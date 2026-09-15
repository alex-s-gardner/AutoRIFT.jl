# Where a threaded whole-granule run spends its time and its memory, attributed over the whole run.
#
#   julia --project=tools/golden -t 10,1 tools/golden/profile_nisar.jl NISAR_L2_PR_GSLC --blocks 8192
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
"""
struct ProfileScan
    samples::Int
    stacks::Vector{Pair{String,Int}}
    by_thread::Dict{Int,Int}
    idle::Int
    sleeping::Int
    gc::Int
    span::Float64
end

"""
    scan_profile(data, hz; nframes = 6) -> ProfileScan

Attribute every sample in `data` rather than only those inside a time window.

Reads the same block layout [`peak_stacks`](@ref) documents. Kept separate from it because the two
answer different questions and mixing them hides a truncated buffer: a window query over a truncated
profile still returns a plausible answer, whereas this reports a `span` that a caller can compare
against the run's wall clock.
"""
function scan_profile(data::Vector{UInt64}, hz::Float64; nframes::Int = 6)
    counts = Dict{String,Int}()
    by_thread = Dict{Int,Int}()
    block_end = 0
    samples = idle = sleeping = gc = 0
    lo = typemax(UInt64)
    hi = zero(UInt64)
    for i in 6:length(data)
        (data[i] == 0 && data[i - 1] == 0 && data[i - 2] in 1:3) || continue
        state = data[i - 2]
        clock = data[i - 3]
        tid = Int(data[i - 5])
        stack_hi = block_end + 1
        block_end = i
        lo = min(lo, clock)
        hi = max(hi, clock)
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
        by_thread[tid] = get(by_thread, tid, 0) + 1
    end
    span = hi > lo ? (hi - lo) / hz : 0.0
    return ProfileScan(samples, sort!(collect(counts); by = last, rev = true),
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
         result = scan_profile(data, trace.tick_hz))
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

# Occupancy comes from CPU time, and the profiler's sample counts cannot substitute for it.
#
# The tempting figure is `running samples / (ticks x nthreads)`, on the reasoning that the profiler
# writes one block per running thread per tick. It does not measure what it appears to: on a load
# verified by CPU time to be using 1.02 threads of ten, that ratio reads **5.26**, and on a genuinely
# saturating ten-thread load it reads 8.55. The profiler samples parked threads too and flags them
# only coarsely, so the numerator counts threads that are doing nothing while the denominator assumes
# they would not have been counted. The bias is large, load-dependent, and in the direction that
# flatters an idle run.
#
# So the sample counts below are counts, not a rate, and occupancy is `cpu_seconds / wall_seconds`
# from [`cpu_seconds!`](@ref) — validated to 1.00 / 1.99 / 3.99 / 9.77 on 1, 2, 4 and 10 spin loops.
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
    # Occupancy: samples per thread against what a saturated run would give. The profiler writes one
    # block per running thread per tick, so `samples / (ticks x nthreads)` is the fraction of the
    # machine the run kept busy.
    # Sample *counts*, not an occupancy: see the note above `report` on why a ratio of profile samples
    # to ticks cannot measure how busy a run was.
    @printf("      samples: %d attributed, %d idle-on-cvwait, %d flagged sleeping\n",
            s.samples, s.idle, s.sleeping)
    @printf("      GC samples %.1f%% of running\n", 100 * s.gc / max(1, s.samples))
    # Shares of *working* samples. Dividing by running-plus-idle instead would fold the occupancy
    # question into every stage's share and make each one look smaller on a less busy configuration,
    # which is two findings tangled into one column — occupancy is the line above.
    println("      where the run spends its time (share of working samples):")
    for (lbl, cnt) in first(s.stacks, 14)
        @printf("        %5.1f%%  %s\n", 100 * cnt / max(1, s.samples), lbl)
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
                      profile::Bool = true)
    k = read_capture(c; n)
    grid = pointset_from_capture(k)
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
               scene, grid = size(grid), halo = (h.X, h.Y), nthreads)
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

function main()
    isempty(ARGS) && error("usage: profile_nisar.jl <product-fragment> " *
                          "[--blocks 0,3072,2304x1152] [--run N] [--no-profile]")
    c = only(cases(ARGS[1]))
    n = parse(Int, argvalue("--run", "100"))
    blocks = parse_block.(split(argvalue("--blocks", "3072"), ','))
    results = measure_case(c; blocks, n, profile = !("--no-profile" in ARGS))
    report_agreement(results)
    mkpath(TRACE_DIR)
    # The fields go (they are the bulk), and so do the `MemTrace` and `ProfileScan` structs: a record
    # holding them cannot be deserialized without loading this script, which defines `main` and would
    # re-run the measurement. The trace is reduced to the figures already extracted from it and the
    # scan to plain fields, so a reader needs nothing but `Serialization`.
    tag = get(ENV, "AUTORIFT_PROFILE_TAG", "")
    path = joinpath(TRACE_DIR,
                    "prof_$(first(split(c.product, "_X_")))$(isempty(tag) ? "" : "_" * tag).jls")
    plain = map(results) do r
        base = Base.structdiff(r, (; dx = 0, dy = 0, trace = 0, scan = 0))
        scan = isnothing(r.scan) ? nothing :
               (; fill = r.scan.fill, delay = r.scan.delay,
                samples = r.scan.result.samples, idle = r.scan.result.idle,
                sleeping = r.scan.result.sleeping, gc = r.scan.result.gc,
                span = r.scan.result.span, stacks = r.scan.result.stacks)
        return (; base..., scan)
    end
    serialize(path, plain)
    @printf("\nwrote %s\n", path)
    return nothing
end

main()
