# Does the coarse pass search points a block's read window was not sized for?
#
#   julia --project=tools/golden -t 10,1 tools/golden/coarse_span.jl S1B_IW_SLC__1SDH_20180809 2048 --run 200
#
# Two things key on a point's **own** search radius:
#
#   * `block_layout` sizes each block's read window as `_searchable_span` of the block's points, grown by
#     the halo. A point with radius zero is not searchable and contributes nothing to that span.
#   * `_run_one_block!` returns before any I/O when `nsearchable` of the block's points is zero.
#
# `_coarse_points` then builds the coarse set with `_cell_max_radius!`, which gives each coarse point the
# **maximum** radius over a neighbourhood `_sparse_filter_width(_sparse_stride(p))` wide — so a point whose
# own radius is zero inherits a neighbour's and becomes searchable in the coarse pass alone. Its search
# window is then not covered by a span that was reduced without it, and `track!` zero-pads the shortfall:
# the coarse pass searches padding where the untiled run searched imagery, its evidence differs, and the
# mask it produces gates off whole regions of the fine pass.
#
# **Widening the window is not the fix, and this tool is what shows why.** The overshoot column reports by
# how many pixels an unfit point's search window exceeds its block's, and on a Sentinel-1 geogrid it is a
# median of ~46,000 px against a 67,945 px scene — those points are most of the scene away from the block
# they were assigned to. Their coordinates say why: a median x of **1.5**, the scene's left edge. They are
# the grid's *fill* points, outside the radar footprint, carrying the reference's 0-based fill mapped
# through this package's `+1.5` origin convention — which is why a test for `(0, 0)` does not find them.
#
# So the two paths diverge in the other direction from the obvious one. A fill point has radius zero and is
# not searchable, but `_cell_max_radius!` gives it a neighbour's radius and makes it searchable in the
# coarse pass. An untiled run's window is the whole scene, so it zero-pads and correlates that point
# against the scene's corner, producing a measurement for a point that has no position. A blocked run's
# window is where that block's real points are, so the point is wholly outside it and is skipped. **The
# untiled run is the one measuring something spurious**, and the fix is to withhold searchability from a
# point with no coordinate rather than to read more imagery — which changes the whole-scene answer that is
# validated against the reference, so it is a correctness decision and not a mechanical one.
#
# Arithmetic only: this reads the capture for its shapes and never correlates.

using Printf, Statistics

include(joinpath(@__DIR__, "correlator.jl"))
using AutoRIFT: block_layout, halo, nsearchable, issearchable, inbounds, scatter, chip_sizes,
                _level_points, _coarse_points, _coarse_block_layout, _block_points,
                _sparse_stride, _sparse_filter_width, _pass_geometry

argvalue(flag, default) = (i = findfirst(==(flag), ARGS);
                           isnothing(i) ? default : ARGS[i + 1])

function main()
    length(ARGS) >= 2 || error("usage: coarse_span.jl <product-fragment> <block-px> [--run N]")
    frag, block = ARGS[1], parse(Int, ARGS[2])
    call = parse(Int, argvalue("--run", "100"))

    c = only(cases(frag))
    k = read_capture(c; n = call, mmap = CAPTURE_IMAGERY)
    grid = pointset_from_capture(k)
    p = params(; kwargs_from_capture(k)...)
    scene = size(k.arrays["in_I1"])
    layout = block_layout(grid, p, scene, (block, block))
    stride = _sparse_stride(p)
    @printf("%s\n  grid %dx%d  scene %dx%d  %d blocks of %d px  sparse stride %d, filter width %d\n",
            c.product, size(grid)..., scene..., length(layout.blocks), block,
            stride, _sparse_filter_width(stride))

    for cs in chip_sizes(p)
        # Every point wanted, which is the first level's case and the widest the coarse set can be.
        pts = _level_points(grid, p, cs, trues(size(grid)))
        setup = _coarse_points(pts, p, cs)
        if isnothing(setup)
            @printf("  chip %3dx%-3d  coarse grid too small to judge — `_coarse_mask` falls back\n", cs.X, cs.Y)
            continue
        end
        coarse, rows, cols = setup.coarse, setup.rows, setup.cols

        # Searchable in the coarse set at a location that is not searchable in the level's own points.
        inherited = 0
        for (ci, r) in pairs(rows), (cj, col) in pairs(cols)
            issearchable(coarse, CartesianIndex(ci, cj)) &&
                !issearchable(pts, CartesianIndex(r, col)) && (inherited += 1)
        end

        # Of those coarse points, how many cannot fit the window their block was given.
        empty_window = starved = unfit = 0
        for blk in _coarse_block_layout(layout.blocks, setup)
            nrw, ncw = length(blk.read_rows), length(blk.read_cols)
            bc = scatter(_block_points(coarse, blk))
            ns = count(i -> issearchable(bc, i), eachindex(bc))
            ns == 0 && continue
            if nrw == 0 || ncw == 0
                empty_window += 1
                starved += ns
                continue
            end
            unfit += count(i -> issearchable(bc, i) && !inbounds(bc, i, (nrw, ncw)), eachindex(bc))
        end

        # By how many pixels an unfit point's search window exceeds its block's read window. This is the
        # quantity a wider span would have to cover, and it is **not** `_pass_geometry`'s `pad`, which is the
        # point's reach rather than its overshoot.
        overx = overy = 0
        overs = Int[]
        for blk in _coarse_block_layout(layout.blocks, setup)
            nrw, ncw = length(blk.read_rows), length(blk.read_cols)
            (nrw == 0 || ncw == 0) && continue
            bc = scatter(_block_points(coarse, blk))
            for i in eachindex(bc)
                issearchable(bc, i) || continue
                inbounds(bc, i, (nrw, ncw)) && continue
                rx = bc.chip_size_x[i] ÷ 2 + bc.radius_x[i] + ceil(Int, abs(bc.dx_prior[i])) + 2
                ry = bc.chip_size_y[i] ÷ 2 + bc.radius_y[i] + ceil(Int, abs(bc.dy_prior[i])) + 2
                ox = max(1 - (bc.x[i] - rx), (bc.x[i] + rx) - ncw, 0)
                oy = max(1 - (bc.y[i] - ry), (bc.y[i] + ry) - nrw, 0)
                overx = max(overx, ceil(Int, ox)); overy = max(overy, ceil(Int, oy))
                push!(overs, ceil(Int, max(ox, oy)))
            end
        end
        h = halo(grid, p, scene)
        @printf("  chip %3dx%-3d  coarse %dx%d, %6d searchable, %5d of them only via a neighbour's radius\n",
                cs.X, cs.Y, size(coarse)..., nsearchable(coarse), inherited)
        @printf("                 %3d zero-size windows (%d points); %5d do not fit; overshoot median %s max %dx%d against halo %dx%d\n",
                empty_window, starved, unfit,
                isempty(overs) ? "—" : string(round(Int, median(overs))), overx, overy, h.X, h.Y)
    end
    return nothing
end

main()
