# Does reducing the number of unique FFT transform sizes speed up an L1 pass?
#
#   julia --project=tools/golden -t 10,1 laddertest.jl
#
# The premise to check first: an earlier note in this session claimed a NISAR L1 grid needs ~29,800
# distinct transform sizes. That counted distinct *raw* `(radius_x, radius_y)` pairs, which is not what
# gets planned. `AutoRIFT._radius_bucket` already rounds every radius up to a power of two and caps it
# at the pass radius, so the measured ladder is **37 sizes** on L1 and 44 on L2 — a few dozen, each
# reused by tens of thousands of points.
#
# So the question is not whether to quantize but whether the existing quantization is coarse enough.
# Decimating to intervals of 4 would make the ladder *finer*: on L1 it admits 476 x 207 size pairs
# against the power-of-two ladder's 9 x 8, so it can only add plans. The direction that could help is
# coarser still, and this measures three ladders against each other on the same points:
#
#   pow2   — what ships: 8, 16, 32, ... 1024, cap
#   pow4   — every second rung: 8, 32, 128, 512, cap  (about half the sizes)
#   int4   — the proposal: every multiple of 4, which is far *more* sizes, not fewer
#
# Measured on a sample of points rather than the whole grid — the ratio between ladders is the answer,
# and a whole-grid run per ladder costs 10 minutes for a comparison that a sample settles.
#
# One limitation of the method, stated because the `sizes` column looks wrong without it. The radii are
# rewritten and then handed to the *unmodified* correlator, whose own `_radius_bucket` rounds them to
# powers of two again. So `int4` reports 25 sizes rather than the hundreds its ladder admits: what this
# measures is the *cost* of feeding the correlator finer radii, not the plan count a real interval-4
# implementation would produce. That cost is the useful half — the plan count only gets worse.

using AutoRIFT
using AutoRIFT: params, _radius_bucket, rebuild, issearchable, PointSet
using Printf, Statistics, Random

include(joinpath("/Users/gardnera/Documents/GitHub/AutoRIFT.jl", "tools", "golden", "correlator.jl"))

# The three ladders, as functions of a raw radius and the pass cap.
pow2(r, cap) = _radius_bucket(r, cap)
function pow4(r, cap)
    r <= 0 && return 0
    r >= cap && return Int(cap)
    # Round up to 8 * 4^k, so every second power-of-two rung.
    k = max(0, ceil(Int, (log2(r) - 3) / 2))
    return min(8 * 4^k, Int(cap))
end
# Intervals of 4, the decimation asked about: `r` rounded up to a multiple of 4. Included because it is
# the proposal, and measuring it is the only way to show the direction it moves cost.
int4(r, cap) = r <= 0 ? 0 : min(4 * cld(Int(r), 4), Int(cap))

# `single` — one bucket at the cap — is deliberately absent. It makes every point execute the widest
# point's transform (56 ms against 0.32 ms per `src/track.jl`), which on this sample is hours rather
# than a measurement.

"""
    quantize(grid, f) -> PointSet

`grid` with every searchable radius replaced by `f(radius, cap)`.

Rewriting the radii is what makes the ladder testable without touching `src`: the correlator's own
`_radius_bucket` then rounds an already-quantized value to itself, so the transform sizes a pass reaches
are exactly the ladder's.
"""
function quantize(grid::PointSet, f)
    capx, capy = maximum(grid.radius_x), maximum(grid.radius_y)
    rx = similar(grid.radius_x)
    ry = similar(grid.radius_y)
    for i in eachindex(grid.radius_x)
        rx[i] = f(grid.radius_x[i], capx)
        ry[i] = f(grid.radius_y[i], capy)
    end
    return rebuild(grid; radius_x = rx, radius_y = ry)
end

function nsizes(grid::PointSet)
    capx, capy = maximum(grid.radius_x), maximum(grid.radius_y)
    s = Set{Tuple{Int,Int}}()
    for i in eachindex(grid.radius_x)
        grid.radius_x[i] > 0 || continue
        push!(s, (_radius_bucket(grid.radius_x[i], capx), _radius_bucket(grid.radius_y[i], capy)))
    end
    return length(s)
end

function main()
    c = only(cases("NISAR_L1_PR_RSLC"))
    k = read_capture(c; n = 100)
    full = pointset_from_capture(k)
    kw = kwargs_from_capture(k)
    a, b = k.arrays["in_I1"], k.arrays["in_I2"]

    # A contiguous window of the grid, so the sample keeps the spatial structure of the radius field
    # rather than scattering it — cost depends on which radii co-occur in a chunk.
    nr, nc = size(full)
    r0, c0 = nr ÷ 2 - 200, nc ÷ 2 - 200
    grid = full[r0:(r0 + 399), c0:(c0 + 399)]
    @printf("sample %s of grid %s, searchable %d\n\n", string(size(grid)), string(size(full)),
            count(>(0), grid.radius_x))

    results = []
    for (name, f) in (("pow2 (ships)", pow2), ("pow4 (coarser)", pow4), ("int4 (asked)", int4))
        g = quantize(grid, f)
        ns = nsizes(g)
        autorift(b, a, g; kw...)                      # warm: plans this ladder's sizes
        t0 = time_ns()
        out = autorift(b, a, g; kw...)
        wall = (time_ns() - t0) / 1e9
        n = count(!isnan, out.dx)
        push!(results, (name, ns, wall, n))
        line = @sprintf("%-22s %3d sizes   %7.2f s   measured %d", name, ns, wall, n)
        println(line); flush(stdout)
        open("/Users/gardnera/.claude/jobs/69bff82a/tmp/ladder_rows.txt", "a") do io
            println(io, line)
        end
    end
    base = results[1][3]
    println()
    for (name, ns, wall, n) in results
        @printf("%-22s %6.2fx the shipping ladder\n", name, wall / base)
    end
    return nothing
end

main()
