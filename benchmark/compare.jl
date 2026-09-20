# Compare two benchmark runs and fail on regression.
#
#   julia --project=benchmark benchmark/compare.jl BASELINE CANDIDATE [--screen FILE]
#   julia --project=benchmark benchmark/compare.jl results/baseline.json results/history/abc123.json
#
# Exits non-zero if any benchmark's minimum time regresses beyond the threshold,
# or if allocations increase for a benchmark that is supposed to have none.
#
# Ratio gating, not absolute: hosted CI runners vary by more than any regression
# worth catching, so an absolute-time threshold would either fire constantly or
# never. A ratio between two runs on the same machine is meaningful; the same two
# absolute numbers from different machines are not, which is why `environment` is
# recorded and mismatches are called out rather than silently compared.
#
# A ratio of two single measurements still carries no error bar, and the suite makes
# around ninety of them at once. Two mechanisms narrow that, and the second is why
# `--screen` exists:
#
#   * Times are centered on the run's own median, so a runner whose speed drifts
#     between the two measurement halves does not read as a change in every benchmark.
#   * A time regression is a *screening* result, to be reproduced before it fails a
#     build. With `--screen FILE` the flagged names are written to `FILE` and do not
#     set the exit status; re-measuring only those on both revisions and comparing
#     again is what decides. Allocation counts are exact at any duration and fail on
#     the first pass regardless.

using JSON3
using Printf
using Statistics

# A regression must clear this, after centering, to fail the build. Chosen so that ordinary
# measurement noise on a shared runner does not fire, while a real algorithmic regression
# does. Tighten once a self-hosted runner makes the numbers quieter.
const REGRESSION_THRESHOLD = 1.10

# Reported but not failed: worth seeing in the table, not worth blocking on.
const NOTEWORTHY_THRESHOLD = 1.03

# Time ratios are divided by the run's own median before they are classified. Both revisions
# are measured sequentially in one job, so a runner that speeds up or slows down between the
# two halves shifts every benchmark together — one such run moved 36 of 95 entries a uniform
# 1.03-1.14x. That common shift is the machine; a regression is a benchmark moving relative to
# its peers. The shift is reported rather than absorbed silently, and a center that is itself
# past the threshold fails the build, so a change that really does slow everything down cannot
# hide inside its own median.
#
# Below this many timed comparisons there are too few peers to locate the shift, which is the
# case for a confirmation pass over a handful of names: the center is then 1.0 and the ratios
# are compared raw.
const MIN_CENTERING_POPULATION = 20

# A screening pass flagging more than this many names is not reporting a set of candidates to
# re-measure — it is reporting that the measurement itself is not usable, or that the change is
# broad enough to need a person. Either way, confirming them all would cost a second full run,
# so `--screen` fails instead of writing a list that long.
const MAX_CONFIRMABLE = 25

# A ratio needs a baseline long enough to time reliably. Below a microsecond a shared
# runner's variation exceeds `REGRESSION_THRESHOLD` on its own: `points/sanitize!` has
# read 1.03x, 1.11x and 1.18x across three runs of identical code, and
# `points/surface_size` clears the threshold on a 0.8 ns difference. A ratio there
# reports the runner, not the code.
#
# Around fifteen of the suite's timed benchmarks fall below this. They are still measured,
# printed, and listed when they slow down — they just do not fail the build. Allocation
# counts are exact at any duration and stay enforced at every scale, which is what keeps
# those entries meaningful: for the per-point paths, allocating at all is the regression
# that matters, and that is caught however short the benchmark.
const MIN_GATED_NS = 1_000

# Benchmarks whose names match these are expected to be allocation-free, so any
# allocation at all is a failure rather than a slowdown. The per-point correlation
# path is the case that matters: allocating once per grid point would be invisible
# in a microbenchmark and ruinous across millions of image pairs.
const ZEROALLOC_PATTERNS = [
    # The per-point correlation path. Allocating here would be invisible in a
    # microbenchmark and ruinous across millions of image pairs.
    r"^correlate/(surface|point|peak|integral|pyrup)",
    r"^points/(chip_bounds|search_bounds|surface_size|nsearchable|sanitize)",
]

# Benchmarks allowed a small fixed allocation, with the amount that is expected.
# Distinct from the zero-alloc set because "constant" is the property that matters
# for these -- an allocation that grows with the work is a bug, one that does not is
# a returned tuple. Checked as an upper bound so a regression to per-step
# allocation still fails.
const BOUNDED_ALLOC = Dict{Regex,Int}(
    r"^correlate/subpixel" => 8,
)

is_zeroalloc(name) = any(p -> occursin(p, name), ZEROALLOC_PATTERNS)

function _alloc_bound(name)
    for (pat, n) in BOUNDED_ALLOC
        occursin(pat, name) && return n
    end
    return nothing
end

function load(path)
    isfile(path) || error("no such benchmark result: $path")
    data = JSON3.read(read(path, String))
    get(data, :schema_version, 0) == 1 || error(
        "$path has schema version $(get(data, :schema_version, "missing")), " *
        "expected 1. Regenerate it with benchmark/run.jl rather than comparing " *
        "results whose recorded fields differ.")
    return data
end

"""
    check_environment(base, cand)

Warn about differences that make absolute times incomparable.

Not an error: comparing a local run against a committed baseline from another
machine is a normal thing to want, and the ratios are still informative even when
the absolute numbers are not. But a silent comparison across a different CPU or
thread count would attribute hardware differences to the code, so the mismatch is
stated.
"""
function check_environment(base, cand)
    b, c = base.environment, cand.environment
    for field in (:cpu, :julia_version, :nthreads, :arch)
        bv, cv = get(b, field, "?"), get(c, field, "?")
        bv == cv || @warn "environment differs: $field" baseline=bv candidate=cv
    end
    get(c, :git_dirty, false) &&
        @warn "candidate was measured with uncommitted changes, so this result " *
              "does not correspond to any commit"
end

"""
    parse_args(args) -> (baseline, candidate, screen)

`screen` is the path to write flagged names to, or `nothing` to gate on them directly.
"""
function parse_args(args)
    positional, screen = String[], nothing
    i = 1
    while i <= length(args)
        if args[i] == "--screen"
            i < length(args) || error("--screen needs a file to write the flagged names to")
            screen = args[i + 1]
            i += 2
        else
            push!(positional, args[i])
            i += 1
        end
    end
    length(positional) == 2 || error(
        "usage: compare.jl BASELINE CANDIDATE [--screen FILE]\n" *
        "  e.g. compare.jl results/baseline.json results/history/<sha>.json")
    return positional[1], positional[2], screen
end

"""
    center(bb, cb, names) -> (drift, population)

The median time ratio over the gated timed benchmarks, which is what the runner contributed
rather than the code, and how many of them it was taken over.

`drift` is `1.0` when the population is under `MIN_CENTERING_POPULATION`: a handful of names
cannot distinguish a shift shared by all of them from a regression in all of them.
"""
function center(bb, cb, names)
    ratios = Float64[]
    for name in names
        startswith(String(name), "memory/") && continue
        b, c = bb[name], cb[name]
        b.min_ns >= MIN_GATED_NS || continue
        push!(ratios, c.min_ns / b.min_ns)
    end
    length(ratios) < MIN_CENTERING_POPULATION && return 1.0, length(ratios)
    return median(ratios), length(ratios)
end

function main()
    basepath, candpath, screen = parse_args(ARGS)
    base, cand = load(basepath), load(candpath)
    check_environment(base, cand)

    bb, cb = base.benchmarks, cand.benchmarks
    names = sort!(collect(intersect(keys(bb), keys(cb))))
    only_base = sort!(setdiff(collect(keys(bb)), collect(keys(cb))))
    only_cand = sort!(setdiff(collect(keys(cb)), collect(keys(bb))))

    drift, npeers = center(bb, cb, names)

    regressions = String[]
    flagged = String[]
    alloc_failures = String[]
    improvements = String[]
    memory_regressions = String[]
    ungated = String[]

    if drift == 1.0
        println("Time ratios are raw: at $npeers gated, there are too few to locate a runner " *
                "shift against.\n")
    else
        println("Runner shift over $npeers gated benchmarks: ",
                round(drift, digits = 3), "x. Every time ratio below is divided by it, so ",
                "it reads against the rest of the suite rather than against the machine.\n")
    end

    println("| benchmark | baseline | candidate | ratio | allocs |")
    println("|---|---:|---:|---:|---:|")

    for name in names
        b, c = bb[name], cb[name]

        # Memory measurements carry their figure in `memory_bytes` and have zero timings, so a
        # time ratio would be 0/0. They are compared on memory instead — and they are the one
        # group where memory is the *point* rather than a secondary observation.
        if startswith(String(name), "memory/")
            mratio = b.memory_bytes == 0 ? NaN : c.memory_bytes / b.memory_bytes
            mflag = if !isnan(mratio) && mratio >= REGRESSION_THRESHOLD
                push!(memory_regressions,
                      "$name ($(round(mratio, digits = 2))x, " *
                      "$(round(c.memory_bytes/2^20, digits=1)) MiB)")
                " **MORE MEMORY**"
            elseif !isnan(mratio) && mratio <= 1 / NOTEWORTHY_THRESHOLD
                " less"
            else
                ""
            end
            @printf("| %s | %.1f MiB | %.1f MiB | %s |%s |\n", name,
                    b.memory_bytes / 2^20, c.memory_bytes / 2^20,
                    isnan(mratio) ? "—" : @sprintf("%.2fx", mratio),
                    isempty(mflag) ? " " : mflag)
            continue
        end

        # Relative to the suite, not to the raw measurement: `drift` is the shift both
        # revisions' measurements share, so dividing it out leaves what this benchmark did
        # that its peers did not.
        ratio = (c.min_ns / b.min_ns) / drift
        dalloc = c.allocs - b.allocs

        flag = if ratio >= REGRESSION_THRESHOLD
            if b.min_ns < MIN_GATED_NS
                push!(ungated, "$name ($(round(ratio, digits = 2))x, " *
                               "baseline $(prettytime(b.min_ns)))")
                " slower (below the gate)"
            elseif isnothing(screen)
                push!(regressions, "$name ($(round(ratio, digits = 2))x)")
                " **SLOWER**"
            else
                push!(flagged, String(name))
                " slower (to confirm)"
            end
        elseif ratio <= 1 / NOTEWORTHY_THRESHOLD
            push!(improvements, "$name ($(round(1 / ratio, digits = 2))x)")
            " faster"
        elseif ratio >= NOTEWORTHY_THRESHOLD
            " slower"
        else
            ""
        end

        nm = String(name)
        bound = _alloc_bound(nm)
        if is_zeroalloc(nm) && c.allocs > 0
            push!(alloc_failures, "$name ($(c.allocs) allocations)")
            flag *= " **ALLOCATES**"
        elseif !isnothing(bound) && c.allocs > bound
            push!(alloc_failures, "$name ($(c.allocs) allocations, bound $bound)")
            flag *= " **OVER ALLOC BOUND**"
        elseif dalloc > 0
            flag *= " +$dalloc allocs"
        end

        @printf("| %s | %s | %s | %.2fx |%s |\n", name,
                prettytime(b.min_ns), prettytime(c.min_ns), ratio,
                isempty(flag) ? " " : flag)
    end

    if !isempty(memory_regressions)
        println("\nMORE MEMORY beyond $(REGRESSION_THRESHOLD)x:")
        foreach(m -> println("  - ", m), memory_regressions)
    end

    isempty(only_cand) || println("\nNew benchmarks (no baseline): ",
                                  join(only_cand, ", "))
    # A benchmark that vanished is worth surfacing: it usually means a rename,
    # but it can also mean coverage was dropped.
    isempty(only_base) || println("\nMissing from candidate: ",
                                  join(only_base, ", "))

    # Every result carries the Python reference time where one exists, so
    # "match or beat OpenCV" is a number in CI rather than a claim.
    report_python_comparison(cb)

    println()
    if !isempty(improvements)
        println("Improved: ", join(improvements, ", "))
    end

    failed = false
    if !isempty(regressions)
        println("\nREGRESSED beyond $(REGRESSION_THRESHOLD)x:")
        foreach(r -> println("  - ", r), regressions)
        failed = true
    end
    # The suite as a whole moving is a regression the centering would otherwise divide out,
    # so it is checked on the center itself.
    if drift >= REGRESSION_THRESHOLD
        println("\nREGRESSED as a whole: the median of $npeers gated benchmarks is ",
                round(drift, digits = 3), "x.")
        failed = true
    end
    if !isempty(alloc_failures)
        println("\nALLOCATED where none is allowed:")
        foreach(r -> println("  - ", r), alloc_failures)
        failed = true
    end
    # Printed whether or not anything failed: these cleared the ratio threshold and
    # are only unenforced because the baseline is too fast to time reliably. Silence
    # here would let "No regressions" cover a row the table shows as slower.
    if !isempty(ungated)
        println("\nSlower but under $(prettytime(MIN_GATED_NS)), so not gated:")
        foreach(r -> println("  - ", r), ungated)
    end
    if !isnothing(screen)
        # One name per line, so re-running the suite over only these needs no parsing. The
        # file is written even when empty: its absence has to mean the comparison did not
        # run, or a confirmation step would read "nothing to confirm" from a crash.
        write(screen, isempty(flagged) ? "" : join(flagged, "\n") * "\n")
        if isempty(flagged)
            println("\nNothing over $(REGRESSION_THRESHOLD)x to confirm.")
        else
            println("\nTO CONFIRM, beyond $(REGRESSION_THRESHOLD)x on one measurement:")
            foreach(r -> println("  - ", r), flagged)
            println("\nWritten to $screen. A single ratio is a screening result — re-measure " *
                    "these on both revisions before treating any of them as real.")
            if length(flagged) > MAX_CONFIRMABLE
                println("\n$(length(flagged)) of $npeers benchmarks are over the threshold, " *
                        "past the $MAX_CONFIRMABLE this screens. Read the table rather than " *
                        "the list: a change this broad, or a run this noisy, is not a set of " *
                        "candidates to re-measure.")
                failed = true
            end
        end
    end
    # Each check above printed what it found, so this is only the all-clear. With names still
    # to confirm it is not one, and says so: "No regressions" over a list of flagged
    # benchmarks would read as a verdict on them.
    if !failed
        println(isempty(flagged) ? "\nNo regressions." :
                "\nNo confirmed regressions. The names above are unconfirmed and do not fail " *
                "this step.")
    end

    exit(failed ? 1 : 0)
end

"""
    report_python_comparison(cb)

Print the speedup against the Python reference for every benchmark that has one.

The point of the whole suite: the goal is stated as matching or beating OpenCV, so
it should be a measured per-kernel ratio, tracked over the life of the port, and
not only an end-to-end impression.
"""
function report_python_comparison(cb)
    path = joinpath(@__DIR__, "results", "python.json")
    isfile(path) || return
    py = JSON3.read(read(path, String))
    shared = sort!(collect(intersect(keys(py.benchmarks), keys(cb))))
    isempty(shared) && return

    println("\n### vs Python reference\n")
    println("| benchmark | python | julia | speedup |")
    println("|---|---:|---:|---:|")
    for name in shared
        p, j = py.benchmarks[name].min_ns, cb[name].min_ns
        @printf("| %s | %s | %s | %.1fx |\n", name,
                prettytime(p), prettytime(j), p / j)
    end
end

# Local rather than pulling in BenchmarkTools: this script should stay runnable
# with nothing but JSON3, so a CI job can compare two committed results without
# building the package or its benchmark dependencies.
function prettytime(ns::Real)
    ns < 1e3 && return @sprintf("%.1f ns", ns)
    ns < 1e6 && return @sprintf("%.1f us", ns / 1e3)
    ns < 1e9 && return @sprintf("%.1f ms", ns / 1e6)
    return @sprintf("%.2f s", ns / 1e9)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
