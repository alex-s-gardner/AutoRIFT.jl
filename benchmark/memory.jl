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
const PROJECT = dirname(BENCH_DIR)

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
