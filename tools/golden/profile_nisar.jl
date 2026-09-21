# Where a threaded whole-granule run spends its time and its memory, attributed over the whole run.
#
#   julia --project=tools/golden -t 10,1 tools/golden/profile_nisar.jl NISAR_L2_PR_GSLC --blocks 8192
#   julia --project=tools/golden -t 10,1 tools/golden/profile_nisar.jl S2B_MSIL1C --blocks 0 --run 200
#
# The case is a fragment of any golden product name, not only a NISAR one: nothing below reads the
# platform. `--stride S` searches a scattered `1/S^2` of the grid (`_thin`), which is what puts a
# granule whose whole-grid untiled peak does not fit the machine within reach of one run.
#
# `mem_nisar.jl` answers "what is the peak and what set it". This answers the two questions that one
# leaves open on a granule that runs for minutes on ten threads:
#
#   **Where does the wall clock go**, over the whole run rather than at the peak. A NISAR run is 12
#   minutes of correlation whose phases have different shapes — the imagery read, the coarse levels,
#   the base level — and a stage that costs a third of the run is invisible in a peak-window sample.
#
#   **How much of the machine is the run actually using.** Ten threads do not imply ten threads of
#   work: `min(nblocks, nthreads)` caps the tasks, blocks finish at wildly different times because a
#   block whose points a finer level resolved returns before any I/O, and the tail is one thread. A
#   run at 60% occupancy has a different fix than one at 98%.
#
# Two properties of the measurement, both of which the recorded figures needed.
#
# **The profiler's buffer must be sized from the run's length, and it is not free to oversize.** A
# sample costs `stack depth + 6` words *per running thread*, so ten threads at the 2 ms default over
# 716 s need ~165 M words where 60 M were requested. Julia warns on `fetch` and stops recording at
# about a third of the run — a truncation that leaves a peak-window query correct (the peak is early)
# and every whole-run query silently answering about the first third. `plan_profile` sizes the buffer
# from a measured runtime and this reports the fill fraction next to every attribution, so a
# truncated profile is visible in the output rather than in a warning that scrolled past.
#
# **Runtime and attribution come from separate runs.** Sampling ten threads every 2 ms perturbs the
# wall clock it is trying to explain, so the timing row is measured with the profiler off and the
# attribution row is a second run. The two are reported side by side, and their difference is the
# instrument's own cost rather than a discrepancy.
#
# **The first correlation in a process compiles, so it is not a timing.** Measured on the S2B case,
# the untiled configuration takes 19.9 s as the first run in the process and 9.7 s afterwards — the
# JIT is more than half of the first figure, and it lands on whichever configuration is listed first.
# `warmup` correlates a small corner of the grid before anything is recorded, so every row is steady
# state and the row order does not change the answer.
#
# Needs the case's captured inputs — see `tools/golden/README.md` for building them.

include(joinpath(@__DIR__, "correlator.jl"))
include(joinpath(dirname(@__DIR__), "ab", "memtrace.jl"))

using Printf, Serialization, Statistics
using AutoRIFT: halo, block_layout, nsearchable

const TRACE_DIR = joinpath(get(ENV, "AUTORIFT_GOLDEN_CACHE",
                               joinpath(expanduser("~/data/autorift/tests"), "golden_tests")),
                           "mem")

argvalue(flag, default) = (i = findfirst(==(flag), ARGS);
                           isnothing(i) ? default : ARGS[i + 1])

# A block size is `(X, Y)` pixels, with `(0, 0)` for an untiled run.
#
# Both axes are swept independently because a halo need not be square, and on these granules it is very
# much not: the NISAR L2 halo is 2216x1103 px, so the *square* floor is twice the Y floor and a square
# block over-provisions Y by a factor of two. `--blocks 3072` still means 3072x3072, since a square
# sweep is the common case; `--blocks 2304x1152` reaches the anisotropic ones.
"""
    parse_block(s) -> Tuple{Int,Int}

Parse one `--blocks` entry: `"0"` for untiled, `"N"` for `N` square, or `"XxY"`.
"""
function parse_block(s::AbstractString)
    parts = split(s, 'x')
    length(parts) == 1 && return (n = parse(Int, parts[1]); (n, n))
    length(parts) == 2 && return (parse(Int, parts[1]), parse(Int, parts[2]))
    throw(ArgumentError("block size \"$s\" is not `N`, `XxY` or `0`"))
end

block_label(bs::Tuple{Int,Int}) =
    bs == (0, 0) ? "untiled" : bs[1] == bs[2] ? "$(bs[1]) px" : "$(bs[1])x$(bs[2]) px"

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
# What a block size costs in buffers, before running it
# ---------------------------------------------------------------------------
#
# `AutoRIFT.BlockBuffers` holds nine arrays sized to the largest read window in the layout, and
# `_run_blocks!` gives each of `min(nblocks, nthreads)` tasks its own set. So the pool is predictable
# from the layout alone, which is what makes an infeasible block size a calculation rather than a
# failed run.
#
# **The nine arrays total 18 bytes per pixel, not 18 bytes each.** For a `UInt8` pair they are two
# `UInt8` planes, three `Float32` (the two filtered images and one shared scratch) and four `Bool`
# masks: 2 + 12 + 4. Measured directly off the struct's own fields at a 2000x1500 window, which gives
# exactly 18.0 B/px.

"""
    BLOCK_BUFFER_BYTES_PER_PIXEL

Bytes of `AutoRIFT.BlockBuffers` per pixel of read window, for a `UInt8` image pair.

Two `UInt8` planes, three `Float32` and four `Bool` — 18 bytes across all nine arrays.
"""
const BLOCK_BUFFER_BYTES_PER_PIXEL = 18

"""
    block_buffer_bytes(window, nblocks, nthreads) -> Int

Bytes of block buffers a run holds at once: one set per concurrent task, each sized to `window`.
"""
block_buffer_bytes(window, nblocks, nthreads) =
    BLOCK_BUFFER_BYTES_PER_PIXEL * prod(window) * min(nblocks, nthreads)

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
# One configuration
# ---------------------------------------------------------------------------

# The layout figures for a block size, or `nothing` if it cannot produce a layout. Computed before
# running so a rejected size is a message rather than a surprise minutes in.
function layout_figures(grid, p, scene, bs::Tuple{Int,Int})
    bs == (0, 0) && return (nblocks = 1, readamp = 1.0, window = scene)
    local L
    try
        L = block_layout(grid, p, scene, bs)
    catch e
        @printf("  block %-12s REJECTED — %s\n", block_label(bs),
                first(sprint(showerror, e), 160))
        return nothing
    end
    readamp = sum(length(x.read_rows) * length(x.read_cols) for x in L.blocks) / prod(scene)
    biggest = argmax(x -> length(x.read_rows) * length(x.read_cols), L.blocks)
    return (nblocks = length(L.blocks), readamp,
            window = (length(biggest.read_rows), length(biggest.read_cols)))
end

"""
    warmup(a, b, grid, kw) -> Float64

Correlate a small patch of `grid` so the process's first timed run is not also its first compilation.

Returns the seconds it took, which is reported rather than discarded: it is the compile cost the
recorded rows no longer carry.

The patch is chosen where the grid is actually searchable. On a NISAR geogrid the footprint is a
rotated swath inside its bounding box and roughly a third of the points are fill, so a corner crop can
easily contain nothing to search — which would compile the setup and none of the correlator, leaving
the JIT in the first recorded row exactly as before.
"""
function warmup(a, b, grid, kw)
    # The densest 128x128 window of the grid, found by scanning a coarse lattice of candidates rather
    # than optimizing: any window with a few thousand searchable points compiles the same code.
    side = 128
    nr, nc = size(grid)
    best = (0, 1, 1)
    for i in 1:max(1, (nr - side) ÷ 8):max(1, nr - side), j in 1:max(1, (nc - side) ÷ 8):max(1, nc - side)
        n = nsearchable(grid[i:min(nr, i + side - 1), j:min(nc, j + side - 1)])
        n > best[1] && (best = (n, i, j))
    end
    n, i, j = best
    n == 0 && error("no searchable window found for warmup; the grid appears to be entirely fill")
    patch = grid[i:min(nr, i + side - 1), j:min(nc, j + side - 1)]
    t0 = time_ns()
    autorift(b, a, patch; kw...)
    seconds = (time_ns() - t0) / 1e9
    @printf("  warmup: %d x %d grid patch at (%d, %d), %d searchable points, %.1f s compiling\n",
            size(patch)..., i, j, n, seconds)
    flush(stdout)
    return seconds
end

"""
    run_config(a, b, grid, kw; bs, profile, seconds_hint, nthreads) -> NamedTuple

Correlate once at block size `bs`, tracing resident memory, and profile the run when `profile`.

`seconds_hint` sizes the profile buffer (see [`plan_profile`](@ref)) and comes from the unprofiled
run of the same configuration, so the buffer is sized against a measurement rather than a guess.
"""
function run_config(a, b, grid, kw; bs::Tuple{Int,Int}, profile::Bool,
                    seconds_hint::Real, nthreads::Integer, label::AbstractString)
    # The floor this configuration is measured against: what the process holds with the imagery
    # resident and nothing running. Collected after a full collection so the figure is a requirement
    # rather than the previous configuration's garbage.
    GC.gc(true); GC.gc(true)
    buf = zeros(UInt64, 64)
    floor_bytes = last(rusage!(buf))

    nwords, delay = plan_profile(seconds_hint, nthreads)
    if profile
        Profile.clear()
        Profile.init(; n = nwords, delay)
    end

    # Written to stderr with a carriage return so an interactive run gets one updating line. Skipped
    # when stderr is not a terminal, since a redirected run turns the same output into thousands of
    # lines that bury the measurement it is reporting on.
    progress = if isa(stderr, Base.TTY)
        function (trace, _)
            isempty(trace.footprint) && return
            @printf(stderr, "\r  %-22s now %8.0f MiB   peak %8.0f MiB   ", label,
                    last(trace.footprint) / 2^20, maximum(trace.footprint) / 2^20)
            flush(stderr)
        end
    else
        nothing
    end

    g0 = Base.gc_num()
    cpu0 = cpu_seconds!(buf)
    out, trace, seconds = with_trace(; interval = 0.01, progress) do
        if profile
            Profile.@profile(bs == (0, 0) ? autorift(b, a, grid; kw...) :
                             autorift(b, a, grid; kw..., process_block_size = bs))
        else
            bs == (0, 0) ? autorift(b, a, grid; kw...) :
            autorift(b, a, grid; kw..., process_block_size = bs)
        end
    end
    cpu = cpu_seconds!(buf) - cpu0
    gc = Base.GC_Diff(Base.gc_num(), g0)
    isnothing(progress) || println(stderr)

    scan = if profile
        data = Profile.fetch(include_meta = true)
        (; fill = Profile.len_data() / Profile.maxlen_data(), delay,
         result = scan_profile(data, trace.tick_hz; delay))
    else
        nothing
    end

    return (; block = bs, seconds, cpu, occupancy = cpu / seconds, floor_bytes,
            peak = maximum(trace.footprint),
            peak_above_floor = maximum(trace.footprint) - floor_bytes,
            peak_resident = maximum(trace.resident),
            peak_live = maximum(trace.live),
            end_live = last(trace.live),
            gc_bytes = gc.allocd, gc_time = gc.total_time / 1e9,
            gc_pause = gc.pause, gc_full = gc.full_sweep,
            measured = count(!isnan, out.dx),
            trace, scan, dx = out.dx, dy = out.dy)
end

# The reported occupancy is `cpu_seconds / wall_seconds` from [`cpu_seconds!`](@ref) — validated to
# 1.00 / 1.99 / 3.99 / 9.77 on 1, 2, 4 and 10 spin loops. Two sample-derived figures sit beside it and
# neither replaces it.
#
# `running samples / (span / delay x nthreads)` is the tempting one and it is wrong: on a load verified
# by CPU time to be using 1.02 threads of ten it reads **5.26**, and on a genuinely saturating ten-thread
# load 8.55. Two separate biases, both flattering an idle run. The denominator assumes the profiler kept
# up, and it does not — 20% of intervals produce no block at all — while the numerator counts threads
# parked on a condition variable as running, because the profiler flags thread state only coarsely.
#
# `samples / nticks` corrects both and is the `mean working` figure `report` prints: `nticks` counts the
# intervals that actually produced blocks, and a parked thread is separated into `idle` by the
# `__psynch_cvwait` test in [`_stack_label`](@ref). What confirms the correction rather than assuming it
# is the companion ratio `(samples + idle) / nticks`, which recovers the thread count — 9.99 of 10 — so
# an interval holds one block per thread and nothing else.
#
# It still reads below the CPU occupancy, and legitimately: 3.16 against 4.87 on a ten-thread optical
# case. The gap is the worker spin before parking, which burns CPU while doing no work. So CPU time
# bounds how much of the machine was *held* and the samples say how much was *working*.
function report(r, figs, nthreads)
    @printf("  %-14s %6d blocks  %7.1f s  peak %8.2f GiB (%.2f above floor)  readamp %5.2fx  measured %d\n",
            block_label(r.block), figs.nblocks, r.seconds,
            r.peak / 2^30, r.peak_above_floor / 2^30, figs.readamp, r.measured)
    @printf("      resident peak %.2f GiB   live peak %.2f GiB   live at end %.2f GiB   floor %.2f GiB\n",
            r.peak_resident / 2^30, r.peak_live / 2^30, r.end_live / 2^30, r.floor_bytes / 2^30)
    @printf("      allocated %.1f GiB   GC %.1f s (%.1f%% of wall) in %d pauses, %d full\n",
            r.gc_bytes / 2^30, r.gc_time, 100 * r.gc_time / r.seconds, r.gc_pause, r.gc_full)
    # Occupancy from CPU time, which does not depend on the profiler having kept up.
    @printf("      occupancy %.2f of %d threads (%.0f%%)   cpu %.0f s over %.0f s wall\n",
            r.clean_occupancy, nthreads, 100 * r.clean_occupancy / nthreads,
            r.clean_cpu, r.clean_seconds)
    # The buffer arithmetic, whether or not this configuration was profiled: it is what says a block
    # size is memory-feasible before it is run.
    @printf("      buffer prediction: %d B/px x %d x %d px window x %d tasks = %.1f GiB\n",
            BLOCK_BUFFER_BYTES_PER_PIXEL, figs.window[1], figs.window[2],
            min(figs.nblocks, nthreads), block_buffer_bytes(figs.window, figs.nblocks, nthreads) / 2^30)
    isnothing(r.scan) && return nothing

    s = r.scan.result
    @printf("      profile: %d running samples at %.0f ms, span %.0f s of %.0f s wall, buffer %.0f%% full%s\n",
            s.samples, 1000 * r.scan.delay, s.span, r.seconds, 100 * r.scan.fill,
            r.scan.fill > 0.98 ? "  ** TRUNCATED **" : "")
    # Sample *counts*, not a rate: see the note above `report` for which ratios of them mean something.
    @printf("      samples: %d attributed, %d idle-on-cvwait, %d flagged sleeping\n",
            s.samples, s.idle, s.sleeping)
    @printf("      GC samples %.1f%% of running\n", 100 * s.gc / max(1, s.samples))

    # How many threads were working at once, over the run's sampling intervals. Read the two
    # cross-checks first: `threads seen` should recover the thread count, and `mean working` should
    # agree with the occupancy CPU time measured. Where they do, the distribution below is the Amdahl
    # shape of the run, measured rather than inferred from a mean.
    @printf("      interval check: %.2f threads seen per interval of %d, mean working %.2f (cpu %.2f)\n",
            (s.samples + s.idle) / max(1, s.nticks), nthreads,
            s.samples / max(1, s.nticks), r.occupancy)
    println("      working threads per interval:")
    for k in 0:(length(s.hist) - 1)
        s.hist[k + 1] == 0 && continue
        share = s.hist[k + 1] / max(1, s.nticks)
        @printf("        %2d %5.1f%% %s\n", k, 100 * share, '#'^ceil(Int, 60 * share))
    end
    ser = sum(s.hist[1:min(SERIAL_THREADS + 1, end)])
    @printf("      at or below %d working threads: %.1f%% of the run, %.1f s of %.1f s\n",
            SERIAL_THREADS, 100 * ser / max(1, s.nticks),
            s.span * ser / max(1, s.nticks), s.span)
    # Shares of *working* samples. Dividing by running-plus-idle instead would fold the occupancy
    # question into every stage's share and make each one look smaller on a less busy configuration,
    # which is two findings tangled into one column — occupancy is the line above.
    #
    # `wall` is the column to read together with it: a stage at 30% of the CPU over 3% of the intervals
    # costs a tenth of what a stage at 5% of the CPU over 5% of the intervals costs.
    #
    # `thr` is the mean threads inside the stage while it appeared, and it is only meaningful for a stage
    # coarse enough that every thread in it shares a label. These labels are six frames deep and carry
    # line numbers, so a saturated run has its ten threads on ten different lines and every row reads
    # near 1.0 regardless of how busy the machine was. The table below, and the histogram above, are what
    # answer that; `thr` distinguishes only the stages that genuinely run one thread wide, like GC.
    println("      where the run spends its time (share of working samples):")
    @printf("        %6s %5s %6s  %s\n", "cpu", "thr", "wall", "stage")
    for (lbl, cnt) in first(s.stacks, 14)
        tk = get(s.ticks, lbl, 0)
        @printf("        %5.1f%% %5.2f %5.1f%%  %s\n", 100 * cnt / max(1, s.samples),
                tk == 0 ? NaN : cnt / tk, 100 * tk / max(1, s.nticks), lbl)
    end

    # The same stages restricted to the serial intervals, which is a different ordering and the one a
    # parallelisation effort is chosen from: the stage with the most CPU is usually the stage that is
    # already spread across every thread.
    ranked = sort!(collect(s.serial); by = last, rev = true)
    if !isempty(ranked)
        nser = sum(values(s.serial))
        @printf("      what holds it there (share of the %d samples in those intervals):\n", nser)
        for (lbl, cnt) in first(ranked, 10)
            @printf("        %5.1f%% %5.1f s  %s\n", 100 * cnt / nser,
                    s.span * cnt / max(1, s.nticks), lbl)
        end
    end
    return nothing
end

"""
    measure_case(c::GoldenCase; blocks, n, profile) -> Vector{NamedTuple}

Correlate `c`'s captured grid once per entry in `blocks`, tracing memory and optionally profiling.

The imagery and point set are read once and shared across configurations: a NISAR capture is 12 GiB
through `read_capture`, and paying that per configuration costs more than the measurement. Each
configuration's peak is reported against the trace's own settled floor before it started, which is
what makes one process sufficient.
"""
function measure_case(c::GoldenCase; blocks::Vector{Tuple{Int,Int}}, n::Integer = 100,
                      profile::Bool = true, stride::Integer = 1, thin_block::Integer = 128)
    k = read_capture(c; n)
    grid = pointset_from_capture(k)
    # Thinning cuts the points searched and not the resident imagery, so a thinned row's peak is not a
    # fraction of the whole-grid one and the two are not comparable. `stride` travels into the record
    # below for that reason: a row has to say which grid it measured.
    stride > 1 && (grid = _thin(grid, stride; block = thin_block))
    kw = kwargs_from_capture(k)
    # `arImgDisp_s(a, b)` cuts its chip from `b`, and the reference calls it with `I1` second, so
    # `I1` binds to `secondary` — the same binding `compare_correlator` documents and uses.
    a, b = k.arrays["in_I1"], k.arrays["in_I2"]
    scene = size(a)
    p = params(; kw...)
    h = halo(grid, p, scene)
    nthreads = Threads.nthreads(:default)
    @printf("%s\n", c.product)
    @printf("  scene %d x %d px, grid %d x %d, halo %d x %d px, %d searchable points, %d threads\n",
            scene..., size(grid)..., h.X, h.Y, nsearchable(grid), nthreads)
    flush(stdout)

    # Before any recorded row, and blocked as well as untiled: the two take different paths through
    # `_run_blocks!`, so warming only one leaves the other's compilation in its first timing.
    #
    # The blocked warmup uses the smallest size being measured rather than a fixed one. A block must
    # be at least as large as the halo it reads around itself, and on a wide-halo granule that floor
    # is in the hundreds of pixels — a hardcoded 512 px is rejected outright on the NISAR L2 grid,
    # whose halo is 620x370 px over a warmup patch.
    warmup(a, b, grid, kw)
    blocked = filter(!=((0, 0)), blocks)
    if !isempty(blocked)
        warmup(a, b, grid, merge(kw, (; process_block_size = argmin(prod, blocked))))
    end

    results = NamedTuple[]
    for bs in blocks
        figs = layout_figures(grid, p, scene, bs)
        isnothing(figs) && continue
        label = block_label(bs)

        # Clean wall clock first, with the profiler off: sampling ten threads every few milliseconds
        # perturbs the runtime this row is meant to report.
        clean = run_config(a, b, grid, kw; bs, profile = false,
                           seconds_hint = 1, nthreads, label = "$label (timing)")
        @printf("  %-14s %7.1f s clean\n", label, clean.seconds)
        flush(stdout)

        r = if profile
            # The buffer is sized from the clean run's own measured length.
            run_config(a, b, grid, kw; bs, profile = true,
                       seconds_hint = clean.seconds, nthreads, label = "$label (profile)")
        else
            clean
        end

        rec = (; r..., clean_seconds = clean.seconds, clean_peak = clean.peak,
               clean_cpu = clean.cpu, clean_occupancy = clean.occupancy,
               nblocks = figs.nblocks, readamp = figs.readamp, window = figs.window,
               scene, grid = size(grid), halo = (h.X, h.Y), nthreads, stride,
               searchable = nsearchable(grid), case = c.product, run = n)
        push!(results, rec)
        report(rec, figs, nthreads)
        flush(stdout)
    end
    return results
end

# Blocking promises a bit-identical result, so it is checked rather than assumed.
function report_agreement(results)
    base = findfirst(r -> r.block == (0, 0), results)
    isnothing(base) && return nothing
    ref = results[base]
    println("\nagreement against the untiled run:")
    for r in results
        r.block == ref.block && continue
        @printf("  %-14s dx identical %s   dy identical %s   measured %d vs %d\n",
                block_label(r.block), isequal(ref.dx, r.dx), isequal(ref.dy, r.dy),
                r.measured, ref.measured)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# The history, which is append-only
# ---------------------------------------------------------------------------
#
# One file per case, holding **every** row ever measured for it rather than the last run's rows.
#
# Writing the run's own results and nothing else loses the sweep: a five-configuration run followed by a
# one-configuration re-measurement of a single block size leaves a file describing only that size, and
# the four other rows exist afterwards solely in whatever console log the caller happened to keep. That
# is how the untiled, 8192, 6144 and 4096 rows of the first L2 sweep came to survive only in a scratch
# log. Appending costs nothing and the rows are small once the trace is dropped.
#
# A row carries its own `stamp` and `commit`, so a re-measurement is a new row beside the old one rather
# than a replacement, and two rows that disagree can be told apart by when and at what code they were
# taken. `render_history` prints the newest row per block size, which is the reading a caller wants,
# while the superseded ones stay on disk.

history_path(c::GoldenCase) =
    joinpath(TRACE_DIR, "prof_$(first(split(c.product, "_X_"))).jls")

"""
    save_results(c::GoldenCase, results) -> String

Append `results` to `c`'s measurement history and print the table of everything measured so far.

The `DisplacementField`s, the `MemTrace` and the `ProfileScan` struct are all dropped: a record holding
them could not be deserialized without loading this script, which runs a measurement on include. What
is kept is the figures already extracted from the trace plus the scan as plain fields, so a reader needs
nothing but `Serialization`.
"""
function save_results(c::GoldenCase, results)
    mkpath(TRACE_DIR)
    path = history_path(c)
    stamp = Libc.strftime("%Y-%m-%dT%H:%M:%S", time())
    commit = try
        readchomp(`git -C $(dirname(dirname(@__DIR__))) rev-parse --short HEAD`)
    catch
        "unknown"
    end
    plain = map(results) do r
        base = Base.structdiff(r, (; dx = 0, dy = 0, trace = 0, scan = 0))
        scan = isnothing(r.scan) ? nothing :
               (; fill = r.scan.fill, delay = r.scan.delay,
                samples = r.scan.result.samples, idle = r.scan.result.idle,
                sleeping = r.scan.result.sleeping, gc = r.scan.result.gc,
                span = r.scan.result.span, stacks = r.scan.result.stacks,
                # Vectors of pairs rather than `Dict`s, so a reader can zip them against `stacks` and
                # needs no lookup to pair a stage with the intervals it was spread over.
                ticks = [lbl => get(r.scan.result.ticks, lbl, 0)
                         for (lbl, _) in r.scan.result.stacks],
                serial = [lbl => get(r.scan.result.serial, lbl, 0)
                          for (lbl, _) in r.scan.result.stacks],
                hist = r.scan.result.hist, serial_threads = SERIAL_THREADS,
                nticks = r.scan.result.nticks)
        return (; base..., scan, stamp, commit)
    end
    # Read-then-write rather than opening in append mode: `Serialization` writes one value per stream,
    # so appending bytes would produce a file whose second value a single `deserialize` never sees.
    old = isfile(path) ? deserialize(path) : []
    all = vcat(old, plain)
    serialize(path, all)
    @printf("\nwrote %s — %d new row%s, %d in history\n",
            path, length(plain), length(plain) == 1 ? "" : "s", length(all))
    render_history(all)
    return path
end

"""
    render_history(rows)

Print every measured configuration, newest measurement per block size, with absolute figures.

Ratios alone cannot be read as a cost — an instance is sized from minutes and GiB — so runtime and peak
are printed in the units they are budgeted in, with the ratio beside them rather than instead of them.
"""
function render_history(rows)
    isempty(rows) && return nothing
    # Newest row per block size *and* grid, by position: `vcat` appends, so the last occurrence is the
    # newest. The grid is part of the key because a thinned row measures a different computation, not a
    # sample of the same one — see `measure_case`. Rows predating `stride` are whole-grid.
    latest = Dict{Any,Any}()
    for r in rows
        latest[(r.block, get(r, :stride, 1))] = r
    end
    keep = sort!(collect(values(latest));
                 by = r -> (get(r, :stride, 1), r.block == (0, 0) ? 0 : -prod(r.block)))
    # Ratios are against the whole-grid untiled run, so a thinned row has none and prints "—" rather
    # than a ratio against a run that searched sixteen times as many points.
    base = get(latest, ((0, 0), 1), nothing)
    println("\nevery configuration measured for this case (newest per block size and grid):")
    @printf("  %-14s %6s %7s %9s %9s %9s %7s %8s %9s  %s\n",
            "block", "stride", "blocks", "runtime", "vs untiled", "peak GiB", "vs unt", "occ/thr",
            "read amp", "measured")
    for r in keep
        rt = r.clean_seconds
        pk = r.peak / 2^30
        s = get(r, :stride, 1)
        comparable = !isnothing(base) && s == 1
        @printf("  %-14s %6d %7d %7.1f s %9s %9.2f %7s %5.2f/%-2d %8.2fx  %d\n",
                block_label(r.block), s, r.nblocks, rt,
                comparable ? @sprintf("%.2fx", rt / base.clean_seconds) : "—",
                pk, comparable ? @sprintf("%.2fx", pk / (base.peak / 2^30)) : "—",
                r.clean_occupancy, r.nthreads, r.readamp, r.measured)
    end
    n = length(rows) - length(keep)
    n > 0 && @printf("  (%d superseded row%s also on disk)\n", n, n == 1 ? "" : "s")
    return nothing
end

function main()
    isempty(ARGS) && error("usage: profile_nisar.jl <product-fragment> " *
                          "[--blocks 0,3072,2304x1152] [--run N] [--stride S] " *
                          "[--thin-block B] [--no-profile]")
    c = only(cases(ARGS[1]))
    n = parse(Int, argvalue("--run", "100"))
    blocks = parse_block.(split(argvalue("--blocks", "3072"), ','))
    stride = parse(Int, argvalue("--stride", "1"))
    thin_block = parse(Int, argvalue("--thin-block", "128"))
    results = measure_case(c; blocks, n, profile = !("--no-profile" in ARGS), stride, thin_block)
    report_agreement(results)
    save_results(c, results)
    return nothing
end

main()
