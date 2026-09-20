# Run the benchmark suite and record the result as JSON.
#
#   julia --project=benchmark benchmark/run.jl [--tag NAME] [--quick] [--only FILE]
#
# Writes benchmark/results/history/<tag>.json, defaulting the tag to the current
# git SHA. History is gitignored, since absolute timings are machine-specific;
# only baseline.json and python.json are committed, because those are the
# references everything is compared against.
#
# `--only FILE` restricts the run to the `group/name` entries listed in `FILE`, one per
# line, which is how a name `compare.jl --screen` flagged gets re-measured without paying
# for the whole suite again. The memory group is skipped, since it is not screened.

using BenchmarkTools
using JSON3
using Printf

include(joinpath(@__DIR__, "benchmarks.jl"))
# Memory is measured in subprocesses rather than by BenchmarkTools, so it lives outside `SUITE`
# and is merged into the results below. See `memory.jl` for why peak RSS cannot be sampled
# in-process.
include(joinpath(@__DIR__, "memory.jl"))

const USAGE = "Usage: run.jl [--tag NAME] [--quick] [--only FILE]"

function parse_args(args)
    tag, quick, only = nothing, false, nothing
    i = 1
    while i <= length(args)
        if args[i] == "--tag" && i < length(args)
            tag = args[i + 1]
            i += 2
        elseif args[i] == "--only" && i < length(args)
            only = args[i + 1]
            i += 2
        elseif args[i] == "--quick"
            quick = true
            i += 1
        else
            error("unrecognised argument $(args[i]). $USAGE")
        end
    end
    return tag, quick, only
end

"""
    restrict!(suite, path)

Drop every leaf of `suite` whose `group/name` is not listed in `path`, one name per line.

Errors on a listed name the suite does not have, and on a list that leaves nothing: both mean
the caller is measuring something other than what it asked for, and a confirmation pass that
silently measures an empty suite reports no regression.
"""
function restrict!(suite::BenchmarkGroup, path::AbstractString)
    isfile(path) || error("no such name list: $path")
    wanted = Set(filter(!isempty, strip.(readlines(path))))
    isempty(wanted) && error("$path lists no benchmarks")
    have = Set(join(k, "/") for (k, _) in BenchmarkTools.leaves(suite))
    missing_names = sort!(collect(setdiff(wanted, have)))
    isempty(missing_names) ||
        error("$path names benchmarks the suite does not define: " * join(missing_names, ", "))
    for (keys, _) in BenchmarkTools.leaves(suite)
        join(keys, "/") in wanted && continue
        # `leaves` returns the key path from the root, so the parent group is everything but
        # the last element.
        delete!(suite[keys[1:(end - 1)]], last(keys))
    end
    return suite
end

git(cmd) = try
    strip(read(Cmd(`git $cmd`; dir = dirname(@__DIR__)), String))
catch
    "unknown"
end

"""
    environment()

Record everything that affects a timing but is not the code under test.

Absolute times are only comparable within one machine and one Julia version, so a
result without this context cannot be interpreted, and comparing across it would
report hardware differences as regressions.
"""
function environment()
    return (
        julia_version = string(VERSION),
        cpu = Sys.cpu_info()[1].model,
        cpu_threads = Sys.CPU_THREADS,
        nthreads = Threads.nthreads(),
        os = string(Sys.KERNEL),
        arch = string(Sys.ARCH),
        hostname = gethostname(),
        git_sha = git(["rev-parse", "HEAD"]),
        git_branch = git(["rev-parse", "--abbrev-ref", "HEAD"]),
        # A dirty tree means the result does not correspond to any commit, so a
        # regression cannot be attributed. Recorded rather than refused, since
        # measuring uncommitted work is the normal case while iterating.
        git_dirty = !isempty(git(["status", "--porcelain"])),
        timestamp = string(round(Int, time())),
    )
end

"""
    flatten(results) -> Dict{String,Any}

Flatten a nested `BenchmarkGroup` result into `"group/name" => stats`.

Storing the flat form keeps the comparison logic trivial and the JSON diffable,
at the cost of repeating group names. Both are worth it: a benchmark that moves
between groups should read as a rename, not as one deletion and one addition.
"""
function flatten(results::BenchmarkGroup, prefix = "")
    out = Dict{String,Any}()
    for (key, val) in results
        name = isempty(prefix) ? string(key) : "$prefix/$key"
        if val isa BenchmarkGroup
            merge!(out, flatten(val, name))
        else
            t = val
            out[name] = (
                # `min` is the primary comparison statistic: it is the least
                # noisy estimate of the code's cost, since interference can only
                # ever make a sample slower. Median is recorded too, because a
                # gap between them indicates variance worth investigating.
                min_ns = minimum(t).time,
                median_ns = median(t).time,
                mean_ns = mean(t).time,
                max_ns = maximum(t).time,
                allocs = minimum(t).allocs,
                memory_bytes = minimum(t).memory,
                samples = length(t.times),
            )
        end
    end
    return out
end

function main()
    tag, quick, only = parse_args(ARGS)
    env = environment()
    isnothing(tag) && (tag = env.git_sha == "unknown" ? "local" : env.git_sha[1:min(end, 12)])

    isnothing(only) || restrict!(SUITE, only)

    if quick
        # Enough to check the suite runs and to catch a large regression, without
        # the full sampling budget. Not for recording a baseline.
        for leaf in BenchmarkTools.leaves(SUITE)
            leaf[2].params.samples = 20
            leaf[2].params.seconds = 0.5
        end
    end

    nbench = length(BenchmarkTools.leaves(SUITE))
    @info "Running $nbench benchmarks" tag quick only nthreads = env.nthreads
    # A single-threaded process makes every threaded benchmark measure the serial path,
    # silently. That is worse than not measuring it: the numbers look plausible and a real
    # threading regression would be invisible.
    env.nthreads == 1 && @warn "Julia is single-threaded, so threaded benchmarks measure " *
        "the serial path. Re-run with `julia -t auto` for meaningful numbers."
    if nbench == 0
        @warn "The suite is empty. Groups are populated as their milestones land."
    end

    tune!(SUITE)
    results = run(SUITE; verbose = true)

    flat = flatten(results)

    # Memory, unless `--quick` or `--only`: each measurement is a fresh subprocess, so the
    # group costs a couple of minutes, which is not worth paying for a smoke run, and a
    # restricted run is confirming timings the memory group has no part in.
    if quick || !isnothing(only)
        @info "Skipping memory measurements; they spawn a process per configuration"
    else
        @info "Measuring peak memory in subprocesses"
        merge!(flat, run_memory())
    end

    payload = Dict(
        # Bumped whenever the recorded fields change, so `compare.jl` can refuse
        # to compare results whose schemas differ rather than silently
        # misinterpreting old numbers.
        "schema_version" => 1,
        "impl" => "julia",
        "environment" => env,
        "benchmarks" => flat,
    )

    outdir = joinpath(@__DIR__, "results", "history")
    mkpath(outdir)
    outfile = joinpath(outdir, "$tag.json")
    open(outfile, "w") do io
        JSON3.pretty(io, payload)
    end

    @info "Wrote $outfile ($(length(flat)) benchmarks)"
    if !isempty(flat)
        println()
        @printf("%-52s %12s %8s\n", "benchmark", "min", "allocs")
        for name in sort!(collect(keys(flat)))
            s = flat[name]
            @printf("%-52s %12s %8d\n", first(name, 52),
                    BenchmarkTools.prettytime(s.min_ns), s.allocs)
        end
    end
    return outfile
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
