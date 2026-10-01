# Whole-run profile attribution, shared by every harness that profiles a threaded pass: sizing the
# buffer from a measured runtime, reading CPU occupancy independently of the profiler, scanning a
# profile's raw samples into per-stage, per-thread-count totals, finding a dense window to warm up on,
# and printing a scan's tables.
#
# Extracted from `profile_nisar.jl`, which was the first caller, so `profile_e2e_nisar.jl` can share it
# rather than duplicate it. Needs `_stack_label` from `tools/ab/memtrace.jl`, included by both callers.

# ---------------------------------------------------------------------------
# How much of the machine a run used
# ---------------------------------------------------------------------------
#
# CPU time over wall time is the occupancy figure to trust, and it is measured independently of the
# profiler: `cpu_seconds / wall_seconds` is the mean number of threads that were running, so 10.0 on
# ten threads is saturation and 6.0 is six. A ratio derived from profile samples needs the sampler to
# have kept up and the buffer not to have filled, neither of which holds unconditionally on a run of
# this length; this needs one syscall.
#
# `ri_user_time` and `ri_system_time` are the first two `UInt64` fields of `rusage_info_v4` after its
# 16-byte uuid, so at 8-byte stride they are indices 3 and 4 — the same buffer `rusage!` fills.
#
# **The fields are mach ticks, not nanoseconds.** Reading them as nanoseconds gives an occupancy of
# 0.02 threads on a load verified to be using one, so the timebase conversion is not optional.
# Validated against 1, 2, 4 and 10 concurrent spin loops, which measure 1.00, 1.99, 3.99 and 9.77.

struct MachTimebase
    numer::UInt32
    denom::UInt32
end

function mach_timebase()
    tb = Ref(MachTimebase(0, 0))
    ccall(:mach_timebase_info, Cint, (Ref{MachTimebase},), tb) == 0 ||
        error("mach_timebase_info failed")
    return tb[].numer / tb[].denom
end

const TICK_NS = mach_timebase()

"""
    cpu_seconds!(buf) -> Float64

This process's total CPU time — user plus system, summed over every thread — in seconds.

`buf` is the same 64-`UInt64` scratch [`rusage!`](@ref) uses.
"""
function cpu_seconds!(buf::Vector{UInt64})
    ccall(:proc_pid_rusage, Cint, (Cint, Cint, Ptr{UInt64}),
          getpid(), RUSAGE_INFO_V4, buf) == 0 || error("proc_pid_rusage failed")
    return (buf[3] + buf[4]) * TICK_NS / 1e9
end

# ---------------------------------------------------------------------------
# Sizing the profile buffer
# ---------------------------------------------------------------------------

"""
    plan_profile(seconds, nthreads; depth = 40, slack = 1.5, cap = 2^31) -> (n, delay)

Buffer size and sampling interval for profiling a run expected to last `seconds`.

A profile block is one stack plus six metadata words, written once per *running thread* per tick, so
the requirement grows with the thread count as well as the run: `seconds/delay × nthreads ×
(depth + 6)`. `depth` is a per-sample stack-depth allowance — 40 is comfortable for this pipeline,
whose deepest correlation stacks run to about 30 frames including the C ones.

`delay` is widened rather than truncating the run when the requirement exceeds `cap`. That is the
choice that keeps the answer unbiased: a coarser interval samples the whole run uniformly, while a
full buffer stops recording partway and attributes 100% of the run to the part that fit.
"""
function plan_profile(seconds::Real, nthreads::Integer;
                      depth::Int = 40, slack::Real = 1.5, cap::Int = 2^31)
    delay = 0.002
    words(d) = ceil(Int, slack * (seconds / d) * nthreads * (depth + 6))
    while words(delay) > cap
        delay *= 2
    end
    return (words(delay), delay)
end

# ---------------------------------------------------------------------------
# Reading a profile over the whole run
# ---------------------------------------------------------------------------

"""
    ProfileScan

Whole-run profile totals: `samples` running samples attributed to `stacks`, `by_thread` samples per
thread id, `idle` samples flagged running but parked on a condition variable, `sleeping` samples the
profiler flagged asleep, and `span` the seconds between the first and last sample.

`stacks` is keyed the way [`_stack_label`](@ref) labels a sample — the innermost frames belonging to
this package — so a row names a pipeline stage rather than an FFTW codelet.

The parallelism figures come from grouping samples into the sampling intervals that produced them. The
profiler writes one block per thread per interval, so the running samples sharing an interval are **the
threads that were working at that instant**: `hist[k + 1]` is the number of intervals with exactly `k`
of them, and `nticks` the intervals in the run. That histogram is the answer a sample count cannot give
— a stage holding 5% of the CPU on ten threads is 0.5% of the wall clock, and the same 5% on one thread
is 5% of it — and it is a distribution rather than a mean, so a run that alternates between ten threads
and one is distinguishable from one that uses five throughout.

`ticks` counts, per stage, the intervals that stage appeared in at all, so `ticks / nticks` is its share
of the wall clock. `serial` counts, per stage, only its samples in intervals at or below
[`SERIAL_THREADS`](@ref) working threads: the stages that hold the machine to one core, which is the
ranking a parallelisation effort is chosen from.

Two cross-checks the caller should read rather than assume. `(samples + idle) / nticks` recovers the
thread count, confirming that intervals group by thread and not by anything else; and `samples / nticks`
is an independent estimate of the occupancy that CPU time measures, so the two disagreeing means one of
them is wrong.
"""
struct ProfileScan
    samples::Int
    stacks::Vector{Pair{String,Int}}
    ticks::Dict{String,Int}
    serial::Dict{String,Int}
    hist::Vector{Int}
    nticks::Int
    by_thread::Dict{Int,Int}
    idle::Int
    sleeping::Int
    gc::Int
    span::Float64
end

# The working-thread count at or below which an interval counts as serial.
#
# Two rather than one: a pipeline stage that runs on the main thread while a single straggler finishes
# the previous stage's last chunk is serial for every purpose this measurement serves, and reads as two.
const SERIAL_THREADS = 2

"""
    scan_profile(data, hz; delay, nframes = 6) -> ProfileScan

Attribute every sample in `data` rather than only those inside a time window.

Reads the same block layout [`peak_stacks`](@ref) documents. Kept separate from it because the two
answer different questions and mixing them hides a truncated buffer: a window query over a truncated
profile still returns a plausible answer, whereas this reports a `span` that a caller can compare
against the run's wall clock.

`delay` is the interval the profiler sampled at, and it is what makes the per-stage thread counts
possible: a sample's `clock` is stamped on its own thread, so the blocks written for one tick carry
close but unequal timestamps and cannot be grouped by equality. Binning at `delay` groups them.
"""
function scan_profile(data::Vector{UInt64}, hz::Float64; delay::Real, nframes::Int = 6)
    counts = Dict{String,Int}()
    by_thread = Dict{Int,Int}()
    # One bin per sampling interval, per stage. `Int32` because a bin index is the tick number and a
    # run long enough to overflow it would need six weeks at 2 ms.
    bins = Dict{String,Set{Int32}}()
    allbins = Set{Int32}()
    # Every attributed sample as (interval, stage), so a second aggregation can ask how many threads
    # shared the interval a sample was taken in. Kept rather than streamed because that count is not
    # known until the interval is complete, and the intervals interleave across threads.
    attributed = Tuple{Int32,String}[]
    running = Dict{Int32,Int}()
    period = max(round(UInt64, delay * hz), one(UInt64))
    block_end = 0
    samples = idle = sleeping = gc = 0
    lo = typemax(UInt64)
    hi = zero(UInt64)
    # The first pass fixes the origin the bins are measured from; `lo` is not known until every sample
    # has been read, and a bin index has to be stable across the run.
    for i in 6:length(data)
        (data[i] == 0 && data[i - 1] == 0 && data[i - 2] in 1:3) || continue
        clock = data[i - 3]
        lo = min(lo, clock)
        hi = max(hi, clock)
    end
    lo == typemax(UInt64) && return ProfileScan(0, Pair{String,Int}[], Dict{String,Int}(),
                                                Dict{String,Int}(), Int[], 0,
                                                Dict{Int,Int}(), 0, 0, 0, 0.0)
    for i in 6:length(data)
        (data[i] == 0 && data[i - 1] == 0 && data[i - 2] in 1:3) || continue
        state = data[i - 2]
        clock = data[i - 3]
        tid = Int(data[i - 5])
        stack_hi = block_end + 1
        block_end = i
        bin = Int32((clock - lo) ÷ period)
        push!(allbins, bin)
        if state != 1
            sleeping += 1
            continue
        end
        ips = @view data[stack_hi:(i - 6)]
        isempty(ips) && continue
        key = _stack_label(ips, nframes)
        # `_stack_label` returns `nothing` for a thread parked in `__psynch_cvwait`: flagged running
        # by the profiler and doing nothing. Counted rather than dropped, because on a wide machine
        # the idle fraction is the occupancy question this script exists to answer.
        if isnothing(key)
            idle += 1
            continue
        end
        samples += 1
        key == "(garbage collection)" && (gc += 1)
        counts[key] = get(counts, key, 0) + 1
        push!(get!(bins, key, Set{Int32}()), bin)
        by_thread[tid] = get(by_thread, tid, 0) + 1
        push!(attributed, (bin, key))
        running[bin] = get(running, bin, 0) + 1
    end

    # An interval that produced samples but none of them running is a real observation — every thread
    # asleep or parked — so the histogram is over `allbins`, not over the intervals that did work.
    hist = zeros(Int, isempty(running) ? 1 : maximum(values(running)) + 1)
    for bin in allbins
        hist[get(running, bin, 0) + 1] += 1
    end
    serial = Dict{String,Int}()
    for (bin, key) in attributed
        running[bin] <= SERIAL_THREADS && (serial[key] = get(serial, key, 0) + 1)
    end

    span = hi > lo ? (hi - lo) / hz : 0.0
    return ProfileScan(samples, sort!(collect(counts); by = last, rev = true),
                       Dict(k => length(v) for (k, v) in bins), serial, hist, length(allbins),
                       by_thread, idle, sleeping, gc, span)
end

# ---------------------------------------------------------------------------
# Warming up on a dense window
# ---------------------------------------------------------------------------

"""
    _densest_window(grid, side) -> (n, i, j)

The `side`-square window of `grid` with the most searchable points, found by scanning a coarse lattice
of candidates rather than optimizing: on a NISAR geogrid the footprint is a rotated swath inside its
bounding box, so a corner crop can easily contain nothing to search, and any window with a few thousand
searchable points compiles the same code as any other.
"""
function _densest_window(grid, side::Integer)
    nr, nc = size(grid)
    best = (0, 1, 1)
    for i in 1:max(1, (nr - side) ÷ 8):max(1, nr - side), j in 1:max(1, (nc - side) ÷ 8):max(1, nc - side)
        n = AutoRIFT.nsearchable(grid[i:min(nr, i + side - 1), j:min(nc, j + side - 1)])
        n > best[1] && (best = (n, i, j))
    end
    return best
end

# ---------------------------------------------------------------------------
# Printing a scan
# ---------------------------------------------------------------------------

"""
    report_profile_scan(s::ProfileScan, wall_seconds, nthreads, occupancy; delay, fill,
                        indent = "", top = 14, serial_top = 10)

Print `s`'s occupancy, thread-histogram and stage-attribution tables — the section every caller that
profiles a threaded pass wants, whatever it prints around it.

`occupancy` is the CPU-time-based figure the caller already has — a clean run's `cpu / seconds`, or the
profiled run's own — printed as the `interval check` line's cross-check; it is not recomputed here
because different callers measure it over different runs and neither answer is this function's to
choose. `indent` lets a caller nest this under its own per-configuration heading; `top`/`serial_top`
bound how many stages the two tables print.
"""
function report_profile_scan(s::ProfileScan, wall_seconds::Real, nthreads::Integer, occupancy::Real;
                             delay::Real, fill::Real, indent::AbstractString = "",
                             top::Integer = 14, serial_top::Integer = 10)
    @printf("%sprofile: %d running samples at %.0f ms, span %.0f s of %.0f s wall, buffer %.0f%% full%s\n",
            indent, s.samples, 1000 * delay, s.span, wall_seconds, 100 * fill,
            fill > 0.98 ? "  ** TRUNCATED **" : "")
    # Sample *counts*, not a rate: see the note above `scan_profile` for which ratios of them mean
    # something.
    @printf("%ssamples: %d attributed, %d idle-on-cvwait, %d flagged sleeping\n",
            indent, s.samples, s.idle, s.sleeping)
    @printf("%sGC samples %.1f%% of running\n", indent, 100 * s.gc / max(1, s.samples))
    # How many threads were working at once, over the run's sampling intervals. Read the two
    # cross-checks first: `threads seen` should recover the thread count, and `mean working` should
    # agree with `occupancy`. Where they do, the distribution below is the Amdahl shape of the run,
    # measured rather than inferred from a mean.
    @printf("%sinterval check: %.2f threads seen per interval of %d, mean working %.2f (cpu %.2f)\n",
            indent, (s.samples + s.idle) / max(1, s.nticks), nthreads,
            s.samples / max(1, s.nticks), occupancy)
    println(indent, "working threads per interval:")
    for k in 0:(length(s.hist) - 1)
        s.hist[k + 1] == 0 && continue
        share = s.hist[k + 1] / max(1, s.nticks)
        @printf("%s  %2d %5.1f%% %s\n", indent, k, 100 * share, '#'^ceil(Int, 60 * share))
    end
    ser = sum(s.hist[1:min(SERIAL_THREADS + 1, end)])
    @printf("%sat or below %d working threads: %.1f%% of the run, %.1f s of %.1f s\n",
            indent, SERIAL_THREADS, 100 * ser / max(1, s.nticks),
            s.span * ser / max(1, s.nticks), s.span)
    # Shares of *working* samples — see the note above `scan_profile` for why dividing by
    # running-plus-idle instead would tangle the occupancy question into every stage's share.
    println(indent, "where the run spends its time (share of working samples):")
    @printf("%s  %6s %5s %6s  %s\n", indent, "cpu", "thr", "wall", "stage")
    for (lbl, cnt) in first(s.stacks, top)
        tk = get(s.ticks, lbl, 0)
        @printf("%s  %5.1f%% %5.2f %5.1f%%  %s\n", indent, 100 * cnt / max(1, s.samples),
                tk == 0 ? NaN : cnt / tk, 100 * tk / max(1, s.nticks), lbl)
    end
    # The same stages restricted to the serial intervals, which is a different ordering and the one a
    # parallelisation effort is chosen from: the stage with the most CPU is usually the stage that is
    # already spread across every thread.
    ranked = sort!(collect(s.serial); by = last, rev = true)
    if !isempty(ranked)
        nser = sum(values(s.serial))
        @printf("%swhat holds it there (share of the %d samples in those intervals):\n", indent, nser)
        for (lbl, cnt) in first(ranked, serial_top)
            @printf("%s  %5.1f%% %5.1f s  %s\n", indent, 100 * cnt / nser,
                    s.span * cnt / max(1, s.nticks), lbl)
        end
    end
    return nothing
end
