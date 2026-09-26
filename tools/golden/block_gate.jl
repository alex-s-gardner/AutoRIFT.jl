# Gate: a blocked run must reproduce an untiled one on a real granule, bit for bit.
#
#   julia --project=tools/golden -t 10,1 tools/golden/block_gate.jl NISAR_L2_PR_GSLC --stride 4
#   julia --project=tools/golden -t 10,1 tools/golden/block_gate.jl LC08_L1TP_060018_20130330 --blocks 1024
#
# Exits non-zero when they differ, so it can be run over a set and believed.
#
# **Why this did not exist and why the absence mattered.** `test/tile.jl` asserts blocked-equals-untiled
# thoroughly, on grids `gridpoints` builds — axis-aligned, every point searchable, uniform radii. The
# property fails on a *geogrid*, where the radius field is sparse and the points outside the radar footprint
# carry a fill coordinate, and no test or gate covered that case. So a blocked run lost 5% of a Sentinel-1
# granule's points through 22 green golden cases without anything noticing. `dev/plan-16gib.md` has the
# diagnosis; this is the check that would have caught it on the first blocked run and will catch its return.
#
# **Thinned by default**, because the gate has to be cheap enough to run. `--stride 4` searches a scattered
# sixteenth of the grid and finds the defect at 1/8 the wall clock of the whole grid — the disagreement is
# spread across the grid rather than confined to a region, so a sample sees it. Pass `--stride 1` for the
# whole grid when a candidate fix needs the full count.
#
# The comparison is `isequal` over raw bits on `dx` and `dy` plus the measured count, which is what
# "bit-identical" means for a field that is mostly `NaN`: `isequal` makes `NaN` equal to itself and `-0.0`
# distinct from `0.0`. A count alone would pass a run that measured the same number of different points.

using Printf

include(joinpath(@__DIR__, "correlator.jl"))
using AutoRIFT: halo, block_layout, nsearchable

argvalue(flag, default) = (i = findfirst(==(flag), ARGS);
                           isnothing(i) ? default : ARGS[i + 1])

function main()
    isempty(ARGS) && error("usage: block_gate.jl <product-fragment> [--run N] [--stride S] [--blocks PX]")
    frag = ARGS[1]
    call = parse(Int, argvalue("--run", "100"))
    stride = parse(Int, argvalue("--stride", "4"))
    # `WxH` as well as a scalar, because a rectangular block is what `block_size_for` returns on a
    # geogrid — NISAR L2's default is 4432x2206 — and a gate that cannot express the default cannot
    # check it.
    block = let spec = argvalue("--blocks", "0")
        occursin('x', spec) ? Tuple(parse.(Int, split(spec, 'x'))) : (v = parse(Int, spec); (v, v))
    end

    c = only(cases(frag))
    k = read_capture(c; n = call, mmap = CAPTURE_IMAGERY)
    grid = pointset_from_capture(k)
    stride > 1 && (grid = _thin(grid, stride))
    kw = kwargs_from_capture(k)
    a, b = k.arrays["in_I1"], k.arrays["in_I2"]
    p = params(; kw...)
    scene = size(a)
    h = halo(grid, p, scene)

    # A block size the halo permits, unless the caller named one. `block_size_for` is what the production
    # entry points use, so gating on its choice gates what a caller actually gets.
    bs = block[1] > 0 ? block : Tuple(AutoRIFT.block_size_for(grid, p, scene))
    layout = block_layout(grid, p, scene, bs)
    @printf("%s\n  scene %dx%d  grid %dx%d  halo %dx%d  stride %d  %d searchable  block %dx%d, %d blocks\n",
            c.product, scene..., size(grid)..., h.X, h.Y, stride, nsearchable(grid),
            bs[1], bs[2], length(layout.blocks))
    flush(stdout)

    whole = autorift(b, a, grid; kw...)
    tiled = autorift(b, a, grid; kw..., process_block_size = bs)

    nwhole, ntiled = count(isfinite, whole.dx), count(isfinite, tiled.dx)
    samedx, samedy = isequal(whole.dx, tiled.dx), isequal(whole.dy, tiled.dy)
    lost = count(i -> isfinite(whole.dx[i]) && !isfinite(tiled.dx[i]), eachindex(whole.dx))
    gained = count(i -> !isfinite(whole.dx[i]) && isfinite(tiled.dx[i]), eachindex(whole.dx))
    moved = count(i -> isfinite(whole.dx[i]) && isfinite(tiled.dx[i]) &&
                       !isequal(whole.dx[i], tiled.dx[i]), eachindex(whole.dx))

    @printf("  untiled %d, blocked %d: lost %d, gained %d, both measured but differing %d\n",
            nwhole, ntiled, lost, gained, moved)

    # Split by population, because the two have different causes and different fixes: a point with a real
    # coordinate can only be short of its block's window by a bounded amount, while a placeholder
    # coordinate is a window away entirely. An aggregate count cannot tell which is still wrong.
    disagree = i -> !isequal(whole.dx[i], tiled.dx[i]) || !isequal(whole.dy[i], tiled.dy[i])
    for (label, sel) in (("positioned", i -> grid.positioned[i]),
                         ("placeholder", i -> !grid.positioned[i]))
        n = count(sel, eachindex(whole.dx))
        n == 0 && continue
        @printf("    %-11s %8d points: %6d disagree, untiled measured %6d, blocked %6d\n",
                label, n, count(i -> sel(i) && disagree(i), eachindex(whole.dx)),
                count(i -> sel(i) && isfinite(whole.dx[i]), eachindex(whole.dx)),
                count(i -> sel(i) && isfinite(tiled.dx[i]), eachindex(whole.dx)))
    end
    if samedx && samedy
        println("  PASS — dx and dy identical")
        return 0
    end
    @printf("  FAIL — dx identical %s, dy identical %s. A blocked run is documented bit-identical to an\n",
            samedx, samedy)
    println("         untiled one; see `dev/plan-16gib.md` and `tools/golden/block_bisect.jl`.")
    return 1
end

exit(main())
