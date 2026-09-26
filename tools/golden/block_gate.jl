# Gate: a blocked run must reproduce an untiled one on a real granule, bit for bit.
#
#   julia --project=tools/golden -t 10,1 tools/golden/block_gate.jl NISAR_L2_PR_GSLC --stride 4
#   julia --project=tools/golden -t 10,1 tools/golden/block_gate.jl LC08_L1TP_060018_20130330 --blocks 1024
#   julia --project=tools/golden -t 10,1 tools/golden/block_gate.jl S2B_MSIL1C --blocks sweep --stride 1
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
# **`--blocks sweep` walks the whole ladder**, and that is the form to trust. Testing one size is what let
# a real defect hide: every case agreed at the size the gate happened to use while four of them differed at
# smaller sizes, because the agreement depends on how tight the read window is. The untiled run is computed
# once and every size compared against it, so N sizes cost one untiled run plus N blocked ones rather than
# 2N runs. The ladder is `blockspec.jl`'s, the same one `mem_nisar.jl` sweeps, so a size that is gated is a
# size that was measured.
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
include(joinpath(@__DIR__, "blockspec.jl"))
using AutoRIFT: halo, block_layout, nsearchable, block_size_for

argvalue(flag, default) = (i = findfirst(==(flag), ARGS);
                           isnothing(i) ? default : ARGS[i + 1])

function main()
    isempty(ARGS) && error("usage: block_gate.jl <product-fragment> [--run N] [--stride S] " *
                           "[--blocks PX|WxH|a,b,c|sweep]")
    frag = ARGS[1]
    call = parse(Int, argvalue("--run", "100"))
    stride = parse(Int, argvalue("--stride", "4"))
    spec = argvalue("--blocks", "0")

    c = only(cases(frag))
    k = read_capture(c; n = call, mmap = CAPTURE_IMAGERY)
    grid = pointset_from_capture(k)
    stride > 1 && (grid = _thin(grid, stride))
    kw = kwargs_from_capture(k)
    a, b = k.arrays["in_I1"], k.arrays["in_I2"]
    p = params(; kw...)
    scene = size(a)
    h = halo(grid, p, scene)

    # `sweep` takes the whole ladder; `0` takes what `block_size_for` returns, which is what a caller gets
    # unasked and so the one size that must never be skipped.
    arms = if spec == "sweep"
        filter(!=((0, 0)), _auto_blocks(h, scene))
    elseif spec == "0"
        [Tuple(block_size_for(grid, p, scene))]
    else
        filter(!=((0, 0)), _parse_blocks(spec))
    end

    @printf("%s\n  scene %dx%d  grid %dx%d  halo %dx%d  stride %d  %d searchable\n",
            c.product, scene..., size(grid)..., h.X, h.Y, stride, nsearchable(grid))
    @printf("  sizes: %s\n", join(_bslabel.(arms), ", "))
    flush(stdout)

    whole = autorift(b, a, grid; kw...)
    nwhole = count(isfinite, whole.dx)
    @printf("  untiled measured %d\n", nwhole)
    flush(stdout)

    bad = String[]
    for bs in arms
        nblocks = try
            length(block_layout(grid, p, scene, bs).blocks)
        catch e
            @printf("  %-13s REJECTED — %s\n", _bslabel(bs), first(sprint(showerror, e), 120))
            flush(stdout)
            continue
        end
        tiled = autorift(b, a, grid; kw..., process_block_size = bs)
        samedx, samedy = isequal(whole.dx, tiled.dx), isequal(whole.dy, tiled.dy)
        lost = count(i -> isfinite(whole.dx[i]) && !isfinite(tiled.dx[i]), eachindex(whole.dx))
        gained = count(i -> !isfinite(whole.dx[i]) && isfinite(tiled.dx[i]), eachindex(whole.dx))
        moved = count(i -> isfinite(whole.dx[i]) && isfinite(tiled.dx[i]) &&
                           !isequal(whole.dx[i], tiled.dx[i]), eachindex(whole.dx))
        # Split by population: a point with a real coordinate can only be short of its window by a bounded
        # amount, where a placeholder coordinate is a window away entirely, and an aggregate count cannot
        # tell which is wrong.
        disagree = i -> !isequal(whole.dx[i], tiled.dx[i]) || !isequal(whole.dy[i], tiled.dy[i])
        nph = count(i -> !grid.positioned[i] && disagree(i), eachindex(whole.dx))
        npos = count(i -> grid.positioned[i] && disagree(i), eachindex(whole.dx))
        ok = samedx && samedy
        ok || push!(bad, _bslabel(bs))
        @printf("  %-13s %6d blocks  blocked %7d  lost %5d  gained %5d  moved %5d  positioned %5d  placeholder %5d  %s\n",
                _bslabel(bs), nblocks, count(isfinite, tiled.dx), lost, gained, moved,
                npos, nph, ok ? "PASS" : "FAIL")
        flush(stdout)
    end

    if isempty(bad)
        @printf("\nPASS — every size reproduces the untiled run on dx and dy\n")
        return 0
    end
    @printf("\nFAIL — %d of %d sizes differ: %s\n", length(bad), length(arms), join(bad, ", "))
    println("       A blocked run is documented bit-identical to an untiled one; see " *
            "`dev/plan-16gib.md`,")
    println("       `dev/CORRECTNESS.md` item 1b, and `tools/golden/block_bisect.jl`.")
    return 1
end

exit(main())
