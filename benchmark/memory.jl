# Peak resident memory, measured in subprocesses.
#
# Deliberately *not* part of `SUITE`. Every other benchmark is a `@benchmarkable` that
# BenchmarkTools runs many times in this process; peak RSS cannot be measured that way, and
# three attempts to do so produced numbers that were wrong in three different directions:
#
#   * `@allocated` counts cumulative churn, not residency. It reported 140 MiB for a 2048²
#     pair whose actual peak was 32 MiB — it sums every temporary ever created.
#   * `Sys.maxrss()` is a *high-water mark*. Measuring several configurations in one process
#     gives the first one's peak and then near-zero for every later one, because the mark has
#     already been set. That produced ratios of 0.05x and 0.00x, which look like a bug in the
#     code rather than in the measurement.
#   * Subtracting a settled baseline within one process still fails, for the same reason.
#
# So: one fresh process per configuration, `Sys.maxrss()` read once at the end, and the
# baseline established by a matching process that allocates the inputs but skips the work.
#
# Why this matters enough to build the machinery. The Python reference's RSS grows without
# plateauing — 376.9 MiB after one 512² pair, 542.3 MiB after thirty, or +5.7 MiB/pair, as *current*
# RSS after an explicit `gc.collect()`. Julia's `Cache` path grows **0.0 MiB of live heap** over the
# same thirty. That difference only shows up if memory is tracked as a metric rather than assumed to
# follow from runtime.
#
# And it only shows up *correctly* if both quantities are tracked. Peak RSS grows ~38 MiB across
# those thirty pairs, which read alone would suggest the same accumulation the reference has. It
# is allocator slack: the live heap is flat. `series_growth` therefore reports both, and the
# difference between them is the difference between "needs process recycling" and "does not".

using JSON3
using Printf

const BENCH_DIR = @__DIR__
# The *benchmark* environment, not the package directory. Every subprocess below needs `AutoRIFT`
# and `FFTW` resolvable, and only this environment is guaranteed to have them instantiated: CI runs
# `Pkg.develop(path=".")` into `benchmark/` and never instantiates the package env itself, so
# pointing at `dirname(BENCH_DIR)` failed with `Package FFTW is required but does not seem to be
# installed`. It worked locally only because the package env happened to be instantiated there.
const PROJECT = BENCH_DIR

"""
    measure_rss(script; threads = "auto") -> Float64

Peak RSS in MiB of a fresh Julia process running `script`.

The process is the unit of measurement, not the function call: `Sys.maxrss()` is a high-water
mark, so a second measurement in the same process reports the first one's peak.
"""
function measure_rss(script::AbstractString; threads = "auto")
    out = read(`$(Base.julia_cmd()) --project=$PROJECT -t $threads -e $script`, String)
    return parse(Float64, strip(out))
end

# A pair of band-limited textures, as source text so each subprocess builds its own. Matches
# `bench_texture` in shape but is written out rather than shared, since the subprocess cannot see
# this module.
const TEXTURE_SRC = """
    using Random
    function tex(n)
        rng = MersenneTwister(0x5EED)
        a = rand(rng, Float32, n, n)
        for _ in 1:2
            a = (a .+ circshift(a, (1, 0)) .+ circshift(a, (0, 1)) .+ circshift(a, (1, 1))) ./ 4
        end
        return a
    end
"""

"""
    floor_breakdown() -> Vector{Pair{String,Float64}}

RSS after each stage of loading, cumulative. Answers "how much of the floor is ours?" — and the
answer is: almost none of it, which is why a trimmed binary is the only lever on the floor.
"""
function floor_breakdown()
    stages = [
        "runtime" => "print(Sys.maxrss()/2^20)",
        "+ FFTW" => "using FFTW; print(Sys.maxrss()/2^20)",
        "+ AutoRIFT" => "using AutoRIFT; print(Sys.maxrss()/2^20)",
    ]
    return [name => measure_rss(src; threads = 1) for (name, src) in stages]
end

"""
    pair_peak(n; threads, kwargs...) -> Float64

Peak RSS in MiB for one `n`-by-`n` pair, over a process that loads and allocates but does not
correlate. The subtraction is what isolates the pipeline from the runtime floor.
"""
function pair_peak(n::Int; threads = "auto", kw = "")
    common = """
        using AutoRIFT
        $TEXTURE_SRC
        a = tex($n); b = circshift(a, (4, -6))
    """
    base = measure_rss(common * "print(Sys.maxrss()/2^20)"; threads)
    work = measure_rss(common * """
        autorift(a, b; chip_size = 32, search_radius = 25$kw)
        print(Sys.maxrss()/2^20)
    """; threads)
    return work - base
end

"""
    series_growth(npairs; n) -> (rss = Float64, live = Float64)

Growth in MiB across `npairs` consecutive pairs through one `Cache`, after the first — reported
two ways, because they answer different questions and only one of them is about leaking.

`rss` is peak resident growth. It includes memory the collector has not returned, so it is what
an instance's memory limit actually sees, and it never falls.

`live` is `Base.gc_live_bytes` after a full collection: memory still reachable. **This is the
leak question.** Measured flat over 30 pairs against ~37 MiB of `rss` growth — so the `rss`
figure is allocator slack, not accumulation, and a long batch run does not need process
recycling. Reporting only `rss` would have implied the opposite.
"""
function series_growth(npairs::Int; n::Int = 512, threads = "auto")
    common = """
        using AutoRIFT
        $TEXTURE_SRC
        base = tex($n)
        series = [circshift(base, (-2k, 3k)) for k in 0:$npairs]
        c = AutoRIFT.init(series[1], series[2]; chip_size = 32, search_radius = 25,
                          threaded = false)
        AutoRIFT.autorift!(c)
        live() = (GC.gc(true); GC.gc(true); Base.gc_live_bytes() / 2^20)
    """
    advance = """
        for k in 3:length(series)
            AutoRIFT.reinit!(c; reference = series[k - 1], secondary = series[k])
            AutoRIFT.autorift!(c)
        end
    """
    first_rss = measure_rss(common * "live(); print(Sys.maxrss()/2^20)"; threads)
    all_rss = measure_rss(common * "live(); " * advance * "print(Sys.maxrss()/2^20)"; threads)
    first_live = measure_rss(common * "print(live())"; threads)
    all_live = measure_rss(common * "live(); " * advance * "print(live())"; threads)
    return (rss = all_rss - first_rss, live = all_live - first_live)
end

"""
    blocked_peak(n; block, threads) -> Float64

Peak RSS in MiB for one `n`-by-`n` pair correlated in `block`-by-`block` grid-point blocks, from an
input that only materializes the window asked for.

The input is a `DiskArrays`-style array whose values are a function of position, so nothing is stored
and the scene is never resident. That is the configuration `process_block_size` exists for, and the
only one where it can be measured: given a resident array, blocking copies out of memory that is
already there, and `docs/plan-tiling.md` records the measurement showing no benefit below ~6000².

Reported against the same no-correlation baseline as [`pair_peak`](@ref), so the two are comparable —
and they must be compared against a *resident* untiled run. Against an untiled run that first
materializes a windowed input, blocking flatters itself: that baseline pays for a copy the blocked
path never makes.
"""
function blocked_peak(n::Int; block::Int = 64, threads = "auto")
    common = """
        using AutoRIFT
        import DiskArrays
        # Values from position, so the scene has no storage and any residency measured is the
        # pipeline's own rather than the input's.
        struct Synth{T} <: DiskArrays.AbstractDiskArray{T,2}
            sz::Tuple{Int,Int}
            seed::Int
        end
        Base.size(a::Synth) = a.sz
        DiskArrays.haschunks(::Synth) = DiskArrays.Chunked()
        DiskArrays.eachchunk(a::Synth) = DiskArrays.GridChunks(a.sz, (512, 512))
        function DiskArrays.readblock!(a::Synth{T}, dest, r::AbstractUnitRange...) where {T}
            for (jj, j) in enumerate(r[2]), (ii, i) in enumerate(r[1])
                dest[ii, jj] = T(0.5 + 0.4 * sin(i * 0.03 + a.seed) * cos(j * 0.021 + a.seed))
            end
            return nothing
        end
        a = Synth{Float32}(($n, $n), 1); b = Synth{Float32}(($n, $n), 2)
        rv = trues($n, $n); sv = trues($n, $n)
    """
    base = measure_rss(common * "print(Sys.maxrss()/2^20)"; threads)
    work = measure_rss(common * """
        autorift(a, b; chip_size = 32, chip_size_max = 32, grid_spacing = 32,
                 search_radius = 12, process_block_size = ($block, $block),
                 reference_valid = rv, secondary_valid = sv)
        print(Sys.maxrss()/2^20)
    """; threads)
    return work - base
end

"""
    run_memory() -> Dict

Every memory measurement, in the same schema `run.jl` emits for timings so `compare.jl` can
treat them alike. `memory_bytes` carries the figure; the timing fields are zero, since these
runs are not timed.
"""
function run_memory()
    results = Dict{String,Any}()
    record!(name, mib) = results[name] = (
        min_ns = 0, median_ns = 0, mean_ns = 0, max_ns = 0, allocs = 0,
        memory_bytes = round(Int, mib * 2^20), samples = 1,
    )

    for (name, mib) in floor_breakdown()
        record!("memory/floor $name", mib)
    end
    for n in (512, 1024, 2048)
        record!("memory/pair $(n)x$(n) serial", pair_peak(n; threads = 1))
        record!("memory/pair $(n)x$(n) threaded", pair_peak(n; threads = "auto"))
    end
    # `upsampling` is included because it is the knob one would *expect* to matter — the
    # refinement workspace grows as its square, 0.2 MiB at 8x to 62.5 MiB at 128x. Measurement
    # says peak RSS barely moves (30.2 against 27.4 MiB), because pooling means few exist at once.
    # Tracked so that claim stays true rather than being remembered.
    #
    # Thread count is *not* tracked as a knob, and that is a finding rather than an omission: it
    # scatters 6.2-9.7 MiB at 512² with no ordering. An earlier reading of one sample per
    # configuration showed peak apparently falling 14.3 -> 5.1 MiB with more threads, which
    # repetition did not reproduce. One subprocess sample is enough to detect a regression against a
    # stored baseline; it is not enough to assert a trend. See `docs/memory.md`.
    for up in (8, 128)
        record!("memory/pair 1024x1024 upsampling $up",
                pair_peak(1024; threads = "auto", kw = ", upsampling = $up"))
    end
    # The rotation search, whose whole cost claim is that it adds *buffers* and not per-point
    # allocation. Measured: 324 allocations against 323 without it, byte-identical at 22.69 MiB of
    # transient allocation, at any angle count — the extra work goes into the workspace's
    # `rotchip`/`rotbest`, so churn does not move at all. Peak above the runtime floor is 13.1 MiB
    # without rotation, 13.0 with three angles, 16.6 with five: three angles is free and the growth
    # from there is the pooled surface copies, not per-point work.
    #
    # Tracked because "one more allocation, whatever the angle count" is exactly the kind of
    # property a later refactor breaks silently.
    record!("memory/pair 1024x1024 rotation x3",
            pair_peak(1024; threads = "auto", kw = ", rotation = true"))
    record!("memory/pair 1024x1024 rotation x5",
            pair_peak(1024; threads = "auto", kw = ", rotation = (-6, -3, 0, 3, 6)"))
    # Blocked processing, from a windowed input, at two sizes and two block sizes — because the
    # numbers say blocking *costs* memory until the scene is large, and the entries exist to pin
    # where that turns over rather than to advertise a reduction:
    #
    #            untiled   blocked 32²   blocked 128²
    #   2048²     69.5        166.5          242.5     MiB above the no-correlation baseline
    #   4096²    156.8        222.9          719.3
    #
    # Two readings. Blocking is 2.4x worse at 2048² and 1.4x at 4096², so the gap closes as the scene
    # grows and the crossover is above both — consistent with `docs/plan-tiling.md`, which measures no
    # benefit below ~6000². And a smaller block is *cheaper*, opposite to the halo-overhead reading
    # alone: it holds less at once, while reading a larger multiple of the scene.
    #
    # The comparison is only meaningful against a resident untiled run. Measured against an untiled
    # run that first materializes a windowed input, blocking looks better than it is, because that
    # baseline is paying for a copy the blocked path never makes.
    for n in (2048, 4096), block in (32, 128)
        record!("memory/blocked $(n)x$(n) block $block",
                blocked_peak(n; block, threads = "auto"))
    end
    g = series_growth(30)
    record!("memory/series 30 pairs 512x512 rss", g.rss)
    # The one that answers "does it leak?". Tracked separately because `rss` never falls and so
    # cannot distinguish accumulation from allocator slack.
    record!("memory/series 30 pairs 512x512 live", g.live)
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    r = run_memory()
    @printf("%-46s %10s\n", "measurement", "MiB")
    for k in sort(collect(keys(r)))
        @printf("%-46s %10.1f\n", k, r[k].memory_bytes / 2^20)
    end
end
