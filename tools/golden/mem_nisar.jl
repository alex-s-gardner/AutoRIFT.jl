# Runtime and peak memory for a golden case, blocked and untiled, with the peak pinned to a stack.
#
#   julia --project=tools/golden -t 10,1 tools/golden/mem_nisar.jl NISAR_L1_PR_RSLC_003 --blocks 0,16384
#
# `tools/ab/mem_blocks.jl` answers this question on a Landsat scene, where the halo is 69 px and a
# 1024-pixel block is the best size. A NISAR granule is the other regime: the halo is 2684x1448 px,
# because a Geogrid search-radius field runs from a median of 26 px to a maximum of 1905, and a block
# has to be large relative to *that* rather than to the scene. This measures what block sizes are
# actually worth using there.
#
# Run in-process rather than one subprocess per configuration, unlike `mem_blocks.jl`. A NISAR capture
# is 5.4 GiB of imagery read through `read_capture`, and paying that per configuration costs more than
# the measurement; the trace makes a single process sufficient, since a peak that is *sampled* rather
# than read from a high-water mark can be attributed to the run that caused it. Each configuration's
# peak is reported against the trace's own settled floor before it started.
#
# Needs the case's captured inputs — see `tools/golden/README.md` for building them.

include(joinpath(@__DIR__, "correlator.jl"))
include(joinpath(dirname(@__DIR__), "ab", "memtrace.jl"))
include(joinpath(@__DIR__, "blockspec.jl"))

using Printf, Serialization
using AutoRIFT: halo, block_layout, nsearchable

const TRACE_DIR = joinpath(get(ENV, "AUTORIFT_GOLDEN_CACHE",
                               joinpath(expanduser("~/data/autorift/tests"), "golden_tests")),
                           "mem")

argvalue(flag, default) = (i = findfirst(==(flag), ARGS);
                           isnothing(i) ? default : ARGS[i + 1])

"""
    measure_case(c::GoldenCase; blocks, n, profile) -> Vector{NamedTuple}

Correlate `c`'s captured grid once per entry in `blocks`, tracing resident memory throughout.

`blocks` are block sizes in pixels as `(X, Y)` pairs, with `(0, 0)` for an untiled run, or `nothing` to
take the ladder `_auto_blocks` derives from the halo. The imagery and the point set are read once and
shared, so the figures differ only in the block size — which is the comparison, and which a
per-configuration subprocess would pay 5.4 GiB to reproduce.

`profile` attributes the peak to a stack, and **costs the runtime it is measuring**: sampling every
thread every 2 ms perturbs the wall clock, which is why `profile_nisar.jl` measures its timing row with
the profiler off. It also risks a hang — a sampled multithreaded run can deadlock against the GC on
macOS (`profiler_gc_deadlock.jl`), which is fatal to a long unattended sweep. Pass `false` when the
question is how long a configuration takes rather than where it spends its memory.
"""
function measure_case(c::GoldenCase; blocks::Union{Nothing,Vector{Tuple{Int,Int}}} = nothing,
                      n::Integer = 100, profile::Bool = true)
    k = read_capture(c; n)
    grid = pointset_from_capture(k)
    kw = kwargs_from_capture(k)
    # `arImgDisp_s(a, b)` cuts its chip from `b`, and the reference calls it with `I1` second, so `I1`
    # binds to `secondary` — the same binding `compare_correlator` documents and uses.
    a, b = k.arrays["in_I1"], k.arrays["in_I2"]
    scene = size(a)
    p = params(; kw...)
    h = halo(grid, p, scene)
    @printf("%s\n", c.product)
    @printf("  scene %d x %d px, grid %d x %d, halo %d x %d px, %d searchable points\n",
            scene..., size(grid)..., h.X, h.Y, nsearchable(grid))
    arms = isnothing(blocks) ? _auto_blocks(h, scene) : blocks
    @printf("  arms: %s\n", join(_bslabel.(arms), ", "))
    flush(stdout)

    results = NamedTuple[]
    for bs in arms
        # The block count and read amplification before running, so a configuration that cannot work is
        # a message rather than a surprise mid-run — and so the progress line has a denominator.
        nblocks, readamp = if bs == (0, 0)
            1, 1.0
        else
            local L
            try
                L = block_layout(grid, p, scene, bs)
            catch e
                @printf("  block %12s px: REJECTED — %s\n", _bslabel(bs),
                        first(sprint(showerror, e), 160))
                flush(stdout)
                continue
            end
            (length(L.blocks),
             sum(length(x.read_rows) * length(x.read_cols) for x in L.blocks) / prod(scene))
        end

        # The floor this configuration's peak is measured against: whatever the process holds with the
        # imagery resident and nothing running. Collected first so the figure is a requirement rather
        # than the previous configuration's garbage.
        GC.gc(true); GC.gc(true)
        buf = zeros(UInt64, 64)
        floor_bytes = last(rusage!(buf))

        label = _bslabel(bs)
        if profile
            Profile.clear()
            Profile.init(; n = 60_000_000, delay = 0.002)
        end
        progress = function (trace, _)
            isempty(trace.footprint) && return
            @printf(stderr, "\r  %-10s now %8.0f MiB   peak %8.0f MiB   ", label, last(trace.footprint) / 2^20, maximum(trace.footprint) / 2^20)
            flush(stderr)
        end
        run1() = bs == (0, 0) ? autorift(b, a, grid; kw...) :
                 autorift(b, a, grid; kw..., process_block_size = bs)
        out, trace, seconds = with_trace(; interval = 0.01, progress) do
            profile ? (Profile.@profile run1()) : run1()
        end
        println(stderr)

        ipk = argmax(trace.footprint)
        thresh = 0.99 * trace.footprint[ipk]
        lo = something(findprev(<(thresh), trace.footprint, ipk), 0) + 1
        hi = something(findnext(<(thresh), trace.footprint, ipk), length(trace) + 1) - 1
        stacks = profile ?
                 peak_stacks(Profile.fetch(include_meta = true), trace.tick[lo], trace.tick[hi]) :
                 Tuple{String,Int}[]

        r = (; block = bs, nblocks, readamp, seconds, scene, grid = size(grid),
             halo = (h.X, h.Y), measured = count(!isnan, out.dx),
             floor_bytes, peak = maximum(trace.footprint),
             # Signed, because it legitimately goes negative: the floor is sampled just before the run
             # and the collector can return pages during it, so a configuration that peaks below its own
             # floor is an ordinary outcome for a small case. Left unsigned this wraps to ~2^64 and reads
             # as 17 billion GiB, which is what it did.
             peak_above_floor = Int(maximum(trace.footprint)) - Int(floor_bytes),
             peak_live = maximum(trace.live), stacks,
             dx = out.dx, dy = out.dy)
        push!(results, r)
        @printf("  %-13s %6d blocks  %7.1f s  peak %8.0f MiB (%.0f above floor)  readamp %5.2fx  measured %d\n",
                label, nblocks, seconds, r.peak / 2^20, r.peak_above_floor / 2^20, readamp, r.measured)
        for (lbl, cnt) in first(stacks, 3)
            @printf("      %5.1f%%  %s\n", 100 * cnt / max(1, sum(last, stacks; init = 0)), lbl)
        end
        flush(stdout)
    end
    return results
end

# Blocking promises a bit-identical result, so it is checked here rather than assumed — on a rotated
# grid especially, where the layout is new.
function report_agreement(results)
    base = findfirst(r -> r.block == (0, 0), results)
    isnothing(base) && return
    ref = results[base]
    println("\nagreement against the untiled run:")
    for r in results
        r.block == ref.block && continue
        @printf("  %-13s dx identical %s   dy identical %s   measured %d vs %d\n",
                _bslabel(r.block), isequal(ref.dx, r.dx), isequal(ref.dy, r.dy), r.measured, ref.measured)
    end
    return nothing
end

function main()
    isempty(ARGS) && error("usage: mem_nisar.jl <product-name-fragment> [--blocks a,b,c] [--run N]")
    c = only(cases(ARGS[1]))
    n = parse(Int, argvalue("--run", "100"))
    spec = argvalue("--blocks", "auto")
    blocks = spec == "auto" ? nothing : _parse_blocks(spec)
    results = measure_case(c; blocks, n)
    report_agreement(results)
    mkpath(TRACE_DIR)
    # Without the fields, which are the bulk and which nothing downstream of the agreement check reads.
    path = joinpath(TRACE_DIR, "$(first(split(c.product, "_X_"))).jls")
    serialize(path, [Base.structdiff(r, (; dx = 0, dy = 0)) for r in results])
    @printf("\nwrote %s\n", path)
    return nothing
end

# Only when run as a script. `block_optimum.jl` includes this file for `measure_case`, `_auto_blocks` and
# `_bslabel`, and an unguarded call here would run a sweep of its own before that driver started.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
