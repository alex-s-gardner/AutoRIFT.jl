# How much of the machine every golden case keeps busy, and which stages stop it keeping more.
#
#   julia --project=tools/golden tools/golden/profile_all.jl                  # measure all 22
#   julia --project=tools/golden tools/golden/profile_all.jl --only S1A,S2B    # measure a subset
#   julia --project=tools/golden tools/golden/profile_all.jl --report          # render from disk
#
# `profile_nisar.jl` answers this for one case. This runs it across the whole golden set and ranks what
# comes back, because the question "where is parallelism left on the table" is not a per-case question:
# a stage that is 40% of a Sentinel-2 run and 2% of a Sentinel-1 one is worth less than its worst case
# suggests, and a stage that is serial on all twenty-two is worth more than any single case says.
#
# **Each case is its own process.** Three reasons, all measured rather than stylistic:
#
#   * A capture is 0.4-12 GiB through `read_capture` and stays resident for the run. Twenty-two of them
#     in one process is not a thing this machine can hold, and dropping each before the next relies on
#     the collector releasing an array the profiler may still reference.
#   * `profile_nisar.jl` samples a long multithreaded run, which deadlocks against the collector on
#     macOS through Julia 1.13.0 — see `profiler_gc_deadlock.jl`. A hung run ignores `SIGTERM` and holds
#     its full footprint, so it has to be `SIGKILL`ed from outside. In-process there is nothing outside.
#   * `threaded` is chosen from `Threads.nthreads()` (`kwargs_from_capture`), which is fixed at process
#     start, so the thread count a case is measured at is a property of its process.
#
# On a timeout the case is re-run once with `--no-profile`, which is the arm that has never hung. That
# costs the stage attribution for that case and keeps its timing, which is the right way round: the
# timing is what every other figure is derived from.
#
# **The measurement is untiled** — `--blocks 0` — because that is the configuration the golden gates
# correlate in (`regate.jl`), so a number here is comparable with one there. A blocked run parallelises
# over blocks instead of over grid points and has its own occupancy, which `profile_nisar.jl` measures
# per block size on a single case.
#
# **`preprocess = :none` on every case.** The captures hold the reference's own filtered bytes, so no
# golden case filters (`kwargs_from_capture`), and the filter's contribution to the serial fraction is
# absent from every row below. `tools/ab/README.md` measures that separately, with the filter in.

include("manifest.jl")

using Printf, Serialization, Dates

const WORKER = joinpath(@__DIR__, "profile_nisar.jl")
const TRACE_DIR = joinpath(CACHE, "mem")
const LOG_DIR = joinpath(TRACE_DIR, "profile_all")

argvalue(flag, default) = (i = findfirst(==(flag), ARGS);
                           isnothing(i) ? default : ARGS[i + 1])

# ---------------------------------------------------------------------------
# What to run
# ---------------------------------------------------------------------------
#
# The fragment, the captured run, and the grid — one entry per golden case, in the order `regate.jl`
# gates them. It is a list rather than a rule for the same reason `manifest.json` is: which run number
# holds a case's capture is a fact about what was taken, not something derivable from the product name.
# The optical and radar captures are run 200 and NISAR's is run 100, and NISAR L1's run 200 directory
# exists but is empty — naming the run is what keeps that from silently skipping.
#
# **NISAR is thinned and the others are not.** An untiled whole-grid NISAR run peaks at 49-61 GiB
# against this machine's 96 and takes 5-10 minutes, twice over for the clean and profiled arms; at
# `--stride 4` it searches a sixteenth of the grid in about a minute. So its occupancy is measured over
# a different point set from the rest of the table and its row is marked, not silently mixed in:
# thinning changes which pyramid levels resolve (`tools/golden/README.md`), so it changes the work, not
# only the amount of it.
struct Job
    fragment::String
    class::String
    run::Int
    stride::Int
end

const JOBS = Job[
    Job("LC08_L1TP_009011", "optical", 200, 1),
    Job("LC08_L1TP_062018", "optical", 200, 1),
    Job("LC09_L1GT_215109", "optical", 200, 1),
    Job("S2A_MSIL1C_20200626", "optical", 200, 1),
    Job("S2B_MSIL1C_20200612", "optical", 200, 1),
    Job("LE07_L1TP_061018_20120428", "optical", 200, 1),
    Job("LE07_L1TP_061018_20130314", "optical", 200, 1),
    Job("LE07_L1TP_063018_20040810", "optical", 200, 1),
    Job("LC08_L1TP_060018_20130330_20200912_02_T1_X_LE07", "optical", 200, 1),
    Job("LT05_L1TP_060018_19851028", "optical", 200, 1),
    Job("LT04_L1TP_063018_19880611", "optical", 200, 1),
    Job("LT05_L1GS_001013_19920425", "optical", 200, 1),
    Job("S1A_IW_SLC__1SSH_20150828T162412", "radar", 200, 1),
    Job("S1A_IW_SLC__1SSH_20151120T080202", "radar", 200, 1),
    Job("S1A_IW_SLC__1SSH_20170221T204710", "radar", 200, 1),
    Job("S1B_IW_SLC__1SDH_20180809T204617", "radar", 200, 1),
    Job("S1C_IW_SLC__1SDV_20250416T010214", "radar", 200, 1),
    Job("S1C_IW_SLC__1SSV_20250416T010159", "radar", 200, 1),
    Job("S1A_IW_SLC__1SSV_20240618T025533", "radar", 200, 1),
    Job("S1A_IW_SLC__1SSV_20240618T025528", "radar", 200, 1),
    Job("NISAR_L1_PR_RSLC", "nisar", 100, 4),
    Job("NISAR_L2_PR_GSLC", "nisar", 100, 4),
]

# The product name a fragment resolves to, which is what names the history file.
only_product(j::Job) = only(cases(j.fragment)).product

# The history file `profile_nisar.jl` appends to for a case, by the same rule it writes.
history_path(product::AbstractString) =
    joinpath(TRACE_DIR, "prof_$(first(split(product, "_X_"))).jls")

# ---------------------------------------------------------------------------
# Running one case
# ---------------------------------------------------------------------------

"""
    run_job(j::Job; threads, timeout, profile = true) -> Symbol

Measure one case in its own process, returning `:ok`, `:timeout` or `:failed`.

Output goes to a per-case log rather than to the terminal: a run prints a memory trace line per
hundredth of a second, and twenty-two of those interleaved bury the table this is building. The log
path is printed so a case that fails can be read.

`timeout` is enforced with `SIGKILL` rather than `SIGTERM`, because the failure it exists for is the
profiler/GC deadlock, and a process in it has no thread left to run a signal handler.
"""
function run_job(j::Job; threads::Integer, timeout::Real, profile::Bool = true)
    mkpath(LOG_DIR)
    log = joinpath(LOG_DIR, "$(j.fragment)$(profile ? "" : ".noprofile").log")
    cmd = `julia --project=$(@__DIR__) -t $threads,1 $WORKER $(j.fragment)
           --blocks 0 --run $(j.run) --stride $(j.stride)`
    profile || (cmd = `$cmd --no-profile`)

    t0 = time()
    p = open(log, "w") do io
        run(pipeline(cmd; stdout = io, stderr = io); wait = false)
    end
    # Coarse polling: the question is only whether a minutes-long process is still alive, and a finer
    # interval would wake this one up thousands of times to learn nothing.
    while process_running(p) && time() - t0 < timeout
        sleep(5)
    end
    if process_running(p)
        kill(p, Base.SIGKILL)
        sleep(2)
        return :timeout
    end
    return success(p) ? :ok : :failed
end

# ---------------------------------------------------------------------------
# Reading the rows back
# ---------------------------------------------------------------------------
#
# The untiled row at the grid this pass measured, newest first. `profile_nisar.jl` appends rather than
# replaces, so a case's file holds every row ever taken for it and the selection has to be explicit
# about which one it is reading — a row from an older commit answers about older code.
function latest_row(j::Job, product::AbstractString; since::Union{Nothing,String} = nothing)
    path = history_path(product)
    isfile(path) || return nothing
    rows = try
        deserialize(path)
    catch
        return nothing
    end
    want = filter(rows) do r
        get(r, :block, nothing) == (0, 0) && get(r, :stride, 1) == j.stride &&
            (isnothing(since) || get(r, :stamp, "") >= since)
    end
    isempty(want) && return nothing
    return last(want)
end

# What the run would cost with every working second spread over every thread, as a multiple of what it
# costs now. This is the whole headroom: the run does `occ x wall` thread-seconds of work, and `n`
# threads can retire that in `occ x wall / n`. Everything else in this file is about *where* the gap is.
speedup_ceiling(occ::Real, n::Integer) = n / max(occ, 1e-9)

# The share of the run's wall clock spent at or below `serial_threads` working threads, and the seconds
# that is. Measured from the histogram rather than inverted from the mean occupancy: a run that held five
# threads throughout and one that alternated between ten and one have the same mean and nothing else in
# common, and only the first of those is already parallel.
function serial_wall(s)
    hist = get(s, :hist, Int[])
    isempty(hist) && return (; share = NaN, seconds = NaN)
    cut = min(get(s, :serial_threads, 2) + 1, length(hist))
    share = sum(hist[1:cut]) / max(1, s.nticks)
    return (; share, seconds = share * s.span)
end

# `_stack_label` joins a sample's innermost AutoRIFT frames outward with " ← ", each frame written
# `function @ file:line`. Two views of that chain answer two different questions:
#
#   * the innermost frame is the loop that was executing — the *hotspot*;
#   * the outermost is the pipeline stage that called into it — the *phase*.
#
# The phase is what gets parallelised and the hotspot is what has to be thread-safe to do it, so both are
# worth ranking. Line numbers are dropped from either: a median selection spread over four lines of one
# loop is one piece of work, and four rows of it reads as four small costs instead of one large one.
frame_func(frame::AbstractString) = first(split(frame, " @ "))
hotspot(label::AbstractString) = frame_func(first(split(label, " ← ")))
phase(label::AbstractString) = frame_func(last(split(label, " ← ")))

# ---------------------------------------------------------------------------
# The tables
# ---------------------------------------------------------------------------

function per_case_table(rows)
    println("\n", "="^120)
    println("Per case: how much of the machine the untiled correlation used")
    println("="^120)
    println("occ is CPU seconds over wall seconds; ceiling is thr/occ, the wall clock perfect scaling")
    println("would reach. serial is the share of sampling intervals at or below two working threads,")
    println("so it says how much of the gap is genuinely single-threaded rather than under-occupied.")
    @printf("\n%-26s %-12s %4s %8s %9s %8s %8s %6s %7s %8s %8s %5s\n",
            "case", "class", "thr", "grid Mpx", "searchable", "wall s", "cpu s", "occ",
            "ceiling", "serial %", "serial s", "GC %")
    for (j, _, r) in rows
        n = r.nthreads
        occ = r.clean_occupancy
        sw = isnothing(get(r, :scan, nothing)) ? (; share = NaN, seconds = NaN) : serial_wall(r.scan)
        # Serial seconds are scaled from the profiled arm's span onto the clean arm's wall clock: the
        # share is a property of the run's shape and the seconds a reader budgets against are the clean
        # ones. A profiled span shorter than the clean run means the sample buffer filled, and the share
        # then describes only the part of the run that was sampled.
        @printf("%-26s %-12s %4d %8.1f %9d %8.1f %8.1f %6.2f %6.2fx %8.0f %8.1f %5.1f\n",
                first(j.fragment, 26), j.class * (j.stride > 1 ? "/thin" : ""), n,
                prod(r.grid) / 1e6, get(r, :searchable, -1),
                r.clean_seconds, r.clean_cpu, occ, speedup_ceiling(occ, n),
                100 * sw.share, sw.share * r.clean_seconds,
                100 * r.gc_time / max(r.seconds, eps()))
    end
    tot_wall = sum(r.clean_seconds for (_, _, r) in rows)
    tot_cpu = sum(r.clean_cpu for (_, _, r) in rows)
    n = maximum(r.nthreads for (_, _, r) in rows)
    @printf("\n%d cases, %.1f s of wall clock, %.1f s of CPU, occupancy %.2f of %d threads\n",
            length(rows), tot_wall, tot_cpu, tot_cpu / tot_wall, n)
    @printf("perfect scaling over %d threads would take %.1f s, so %.1f s is on the table\n",
            n, tot_cpu / n, tot_wall - tot_cpu / n)
    return nothing
end

"""
    occupancy_profile(rows)

Print the whole set's working-thread distribution, one row per thread count.

The single most informative figure here, because it settles the question the mean occupancy cannot: a
set whose intervals are bimodal — a tall bar at one thread and another at ten — has serial phases to
parallelise, and one whose intervals cluster in the middle has a granularity or load-balance problem
instead. Those want different work, and the mean is the same for both.
"""
function occupancy_profile(rows)
    hist = Int[]
    for (_, _, r) in rows
        s = get(r, :scan, nothing)
        isnothing(s) && continue
        h = get(s, :hist, Int[])
        length(h) > length(hist) && resize!(append!(hist, zeros(Int, length(h) - length(hist))),
                                            length(h))
        for k in eachindex(h)
            hist[k] += h[k]
        end
    end
    isempty(hist) && return nothing
    total = sum(hist)
    println("\n", "="^120)
    println("Working threads per sampling interval, over every profiled case")
    println("="^120)
    for k in 0:(length(hist) - 1)
        hist[k + 1] == 0 && continue
        share = hist[k + 1] / total
        @printf("%3d threads %6.1f%%  %s\n", k, 100 * share, '#'^ceil(Int, 70 * share))
    end
    return nothing
end

"""
    stage_table(rows; top = 16)

Rank what holds the machine at or below two working threads, by phase and by hotspot.

That is the ranking a parallelisation effort is chosen from, and it is not the ranking by CPU share: the
stages with the most CPU are the ones already spread over every thread. A stage appears here only for
the part of its time that ran with the machine idle beside it.

`serial s` is the set's serial wall clock apportioned by each key's share of the samples taken in those
intervals, so a column sums to the serial total rather than over-counting the intervals two stages
shared. `cases` says how many of the profiled cases contributed, so a total driven by one outlier is
visible rather than averaged into looking general.
"""
function stage_table(rows; top::Int = 16)
    tally = (phase = Dict{String,Float64}(), hotspot = Dict{String,Float64}())
    ncases = (phase = Dict{String,Int}(), hotspot = Dict{String,Int}())
    profiled = 0
    total = 0.0
    cut = 2

    for (_, _, r) in rows
        s = get(r, :scan, nothing)
        isnothing(s) && continue
        haskey(s, :serial) || continue
        profiled += 1
        cut = get(s, :serial_threads, cut)
        # The clean arm's wall clock, apportioned by the profiled arm's shape: the share is what the
        # profile measures and the seconds are what a reader budgets against.
        secs = serial_wall(s).share * r.clean_seconds
        total += secs
        nser = sum(last, s.serial; init = 0)
        nser == 0 && continue
        # Per view, the keys this case has already been counted against: `cases` counts cases, not the
        # dozens of labels one case contributes to a single key.
        seen = (phase = Set{String}(), hotspot = Set{String}())
        for (label, cnt) in s.serial
            cnt == 0 && continue
            for view in (:phase, :hotspot)
                key = view === :phase ? phase(label) : hotspot(label)
                acc, cs, done = getfield(tally, view), getfield(ncases, view), getfield(seen, view)
                acc[key] = get(acc, key, 0.0) + secs * cnt / nser
                key in done || (cs[key] = get(cs, key, 0) + 1; push!(done, key))
            end
        end
    end

    if profiled == 0
        println("\nNo case carries a stage attribution: every profiled arm timed out or was skipped.")
        return nothing
    end

    println("\n", "="^120)
    @printf("What holds the machine at or below %d working threads — %.1f s over %d profiled case%s\n",
            cut, total, profiled, profiled == 1 ? "" : "s")
    for (what, acc, cs) in (("by phase — the stage that would have to run in parallel",
                             tally.phase, ncases.phase),
                            ("by hotspot — the loop inside it",
                             tally.hotspot, ncases.hotspot))
        println("="^120)
        println(what)
        @printf("%10s %7s %6s  %s\n", "serial s", "share", "cases", "stage")
        ranked = sort!(collect(keys(acc)); by = k -> -acc[k])
        for key in first(ranked, top)
            @printf("%10.1f %6.1f%% %6d  %s\n",
                    acc[key], 100 * acc[key] / max(total, eps()), get(cs, key, 0), key)
        end
        rest = sum(acc[k] for k in ranked[min(top + 1, length(ranked) + 1):end]; init = 0.0)
        rest > 0 && @printf("%10.1f %6.1f%% %6s  (%d further)\n",
                            rest, 100 * rest / max(total, eps()), "", length(ranked) - top)
    end
    return nothing
end

# ---------------------------------------------------------------------------

function main()
    threads = parse(Int, argvalue("--threads", "10"))
    timeout = parse(Float64, argvalue("--timeout", "1800"))
    only = argvalue("--only", "")
    jobs = isempty(only) ? JOBS :
           filter(j -> any(f -> occursin(f, j.fragment), split(only, ',')), JOBS)
    isempty(jobs) && error("--only \"$only\" matched no case")

    # A pass is identified by when it started, so the tables read this pass's rows rather than whichever
    # row is newest in a file that may hold months of them.
    since = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS")
    measure = !("--report" in ARGS)

    if measure
        @printf("%d case%s at %d threads, %.0f s timeout each, logs in %s\n",
                length(jobs), length(jobs) == 1 ? "" : "s", threads, timeout, LOG_DIR)
        flush(stdout)
        for (i, j) in enumerate(jobs)
            t0 = time()
            @printf(stderr, "  … %2d/%d %-28s ", i, length(jobs), first(j.fragment, 28))
            flush(stderr)
            state = run_job(j; threads, timeout)
            if state === :timeout
                # The profiled arm is the one that deadlocks; the timing arm has not. Retrying without
                # it keeps the case's occupancy and loses only its attribution.
                @printf(stderr, "timed out after %.0f s, retrying without the profiler … ",
                        time() - t0)
                flush(stderr)
                state = run_job(j; threads, timeout, profile = false)
            end
            @printf(stderr, "%s in %.0f s\n", String(state), time() - t0)
            flush(stderr)
        end
    end

    rows = Tuple{Job,String,Any}[]
    missed = String[]
    for j in jobs
        product = only_product(j)
        r = latest_row(j, product; since = measure ? since : nothing)
        # Falling back to any row when this pass produced none, so `--report` works and a failed case
        # shows its last good measurement rather than vanishing. Flagged either way.
        isnothing(r) && (r = latest_row(j, product))
        isnothing(r) ? push!(missed, j.fragment) : push!(rows, (j, product, r))
    end

    isempty(rows) && error("no case produced a row; read the logs in $LOG_DIR")
    per_case_table(rows)
    occupancy_profile(rows)
    stage_table(rows)
    isempty(missed) || @printf("\n%d case%s without a row: %s\n",
                               length(missed), length(missed) == 1 ? "" : "s",
                               join(missed, ", "))
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
