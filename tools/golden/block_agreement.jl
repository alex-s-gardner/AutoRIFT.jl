# Does a blocked run reproduce an untiled one on a real granule, and if not, where does it differ?
#
#   julia --project=tools/golden -t 10,1 tools/golden/block_agreement.jl S1B_IW_SLC__1SDH_20180809 2048 --run 200
#
# `profile_nisar.jl` reports whether `dx` is identical and how many points each row measured, which is
# enough to see a disagreement and not enough to classify one. This localizes it, because the candidate
# causes want opposite responses:
#
#   * **An under-computed halo** puts the disagreement on block seams. The discriminator is the lost
#     points' distance to their own block's grid edge against the kept points' — if the two
#     distributions match, the seam is not where the points are being lost.
#   * **A last-bit numerical difference** — a block's integral tables start at the block's origin rather
#     than the scene's, so the normalization accumulates a different number of terms — perturbs the
#     correlation surface *everywhere* and changes `dx` only where the peak is flat enough for the
#     perturbation to move the argmax. The discriminator is how often `correlation` differs while `dx`
#     does not.
#   * **A localized defect** shows up as a minority of blocks behaving differently, with `correlation`
#     and `dx` moving together and by more than a rounding.
#
# `test/tile.jl` asserts the equality this measures, on synthetic grids where it holds. This is the same
# question on a granule, which is where it has been observed to fail.

using Printf, Statistics

include(joinpath(@__DIR__, "correlator.jl"))
using AutoRIFT: block_layout, halo, nsearchable, upsampling, issearchable, inbounds, scatter

argvalue(flag, default) = (i = findfirst(==(flag), ARGS);
                           isnothing(i) ? default : ARGS[i + 1])

function main()
    length(ARGS) >= 2 || error("usage: block_agreement.jl <product-fragment> <block-px> [--run N]")
    frag, block = ARGS[1], parse(Int, ARGS[2])
    call = parse(Int, argvalue("--run", "100"))

    c = only(cases(frag))
    k = read_capture(c; n = call, mmap = CAPTURE_IMAGERY)
    grid = pointset_from_capture(k)
    kw = kwargs_from_capture(k)
    # `arImgDisp_s(a, b)` cuts its chip from `b` and the reference calls it with `I1` second, so `I1`
    # binds to `secondary` — the same binding `profile_nisar.jl` and `compare_correlator` use.
    a, b = k.arrays["in_I1"], k.arrays["in_I2"]
    p = params(; kw...)
    scene = size(a)
    h = halo(grid, p, scene)
    @printf("%s\n  scene %dx%d  grid %dx%d  halo %dx%d  %s imagery  %d searchable  block %d px\n",
            c.product, scene..., size(grid)..., h.X, h.Y, eltype(a), nsearchable(grid), block)
    flush(stdout)

    whole = autorift(b, a, grid; kw...)
    tiled = autorift(b, a, grid; kw..., process_block_size = (block, block))

    fin(A, i) = isfinite(A[i])
    lost = findall(i -> fin(whole.dx, i) && !fin(tiled.dx, i), eachindex(whole.dx))
    kept = findall(i -> fin(whole.dx, i) && fin(tiled.dx, i), eachindex(whole.dx))
    gained = count(i -> !fin(whole.dx, i) && fin(tiled.dx, i), eachindex(whole.dx))
    # A comprehension, not `findall` over `kept`: that returns positions *within* `kept` rather than
    # grid indices, and indexing the field with them silently reads the wrong points.
    moved = [i for i in kept if !isequal(whole.dx[i], tiled.dx[i])]
    @printf("\nuntiled measured %d, blocked %d: lost %d, gained %d, both measured but differing %d\n",
            count(isfinite, whole.dx), count(isfinite, tiled.dx), length(lost), gained, length(moved))

    layout = block_layout(grid, p, scene, (block, block))
    owner = zeros(Int, size(grid))
    for (bi, blk) in pairs(layout.blocks)
        owner[blk.grid_rows, blk.grid_cols] .= bi
    end
    cart = CartesianIndices(size(grid))
    function edge_distance(idx)
        row, col = Tuple(cart[idx])
        blk = layout.blocks[owner[idx]]
        return min(row - first(blk.grid_rows), last(blk.grid_rows) - row,
                   col - first(blk.grid_cols), last(blk.grid_cols) - col)
    end

    # Seam or not. Matching distributions say the halo is not where the points go.
    dlost, dkept = edge_distance.(lost), edge_distance.(kept)
    @printf("\ndistance to own block's grid edge, in grid points — a seam effect concentrates the lost:\n")
    for (label, d) in ("lost" => dlost, "kept" => dkept)
        isempty(d) && continue
        @printf("  %-5s median %5.1f   at distance 0-1: %5.1f%%   n = %d\n",
                label, median(d), 100 * count(<=(1), d) / length(d), length(d))
    end

    # Rounding or not. One upsampling step is the finest difference the output can express.
    if !isempty(moved)
        d = sort(abs.(whole.dx[moved] .- tiled.dx[moved]))
        # `upsampling` is a property of the refinement method, not of `Params`.
        step = 1 / upsampling(first(p.subpixel))
        @printf("\n|dx| difference where both measured, n = %d (one upsampling step is %.4f px):\n",
                length(moved), step)
        @printf("  median %.4f  p90 %.4f  p99 %.4f  max %.4f  within one step %.1f%%\n",
                median(d), d[max(1, round(Int, 0.90 * end))], d[max(1, round(Int, 0.99 * end))],
                last(d), 100 * count(<=(step), d) / length(d))
    end

    # Pervasive or localized. A numerical perturbation moves `correlation` far more often than `dx`.
    bothfin = findall(i -> fin(whole.correlation, i) && fin(tiled.correlation, i),
                      eachindex(whole.correlation))
    cdiff = filter(i -> !isequal(whole.correlation[i], tiled.correlation[i]), bothfin)
    @printf("\ncorrelation differs at %d of %d points where both measured (%.1f%%)\n",
            length(cdiff), length(bothfin), 100 * length(cdiff) / max(length(bothfin), 1))
    if !isempty(cdiff)
        @printf("  dx identical at %d of those (%.1f%%) — high means numerical, low means the two move together\n",
                count(i -> isequal(whole.dx[i], tiled.dx[i]), cdiff),
                100 * count(i -> isequal(whole.dx[i], tiled.dx[i]), cdiff) / length(cdiff))
        cd = [abs(whole.correlation[i] - tiled.correlation[i]) for i in cdiff]
        @printf("  |correlation| difference median %.3g, max %.3g\n", median(cd), maximum(cd))
    end

    # What kind of point is lost. A peak this weak is not a position either run measured.
    for (label, idx) in ("lost" => lost, "kept" => kept)
        cs = [whole.correlation[i] for i in idx if isfinite(whole.correlation[i])]
        isempty(cs) && continue
        @printf("%-5s points: untiled correlation median %.3f, chip sizes %s\n",
                label, median(cs), string(sort(unique(whole.chip_size[idx]))))
    end

    # Localized or spread. A handful of blocks losing everything is a different fault from every block
    # losing a rim.
    byblock = Dict{Int,Int}()
    total = Dict{Int,Int}()
    for i in eachindex(whole.dx)
        isfinite(whole.dx[i]) || continue
        total[owner[i]] = get(total, owner[i], 0) + 1
    end
    for i in lost
        byblock[owner[i]] = get(byblock, owner[i], 0) + 1
    end
    frac = [byblock[key] / total[key] for key in keys(byblock) if get(total, key, 0) > 0]
    isempty(frac) && return nothing
    @printf("\n%d of %d blocks lose a point; median %.0f%% of that block's points, %d lose all of them\n",
            length(byblock), length(layout.blocks), 100 * median(frac), count(>=(0.999), frac))

    # What distinguishes a block that loses *every* point. That is a whole-block failure rather than an
    # edge effect, and `_run_one_block!` has one early return — `nsearchable(bpts) == 0` — which cannot
    # differ between the two paths, since searchability is a property of the points. So the candidates are
    # the block's read window: clipped short at a scene boundary, or too small for its own points to fit,
    # which sends the pass down the zero-padded path.
    allgone = [key for key in keys(byblock) if get(total, key, 0) > 0 && byblock[key] == total[key]]
    function describe(keys_)
        clipped = fits = 0
        for key in keys_
            blk = layout.blocks[key]
            rr, cc = blk.read_rows, blk.read_cols
            (first(rr) == 1 || last(rr) == scene[1] ||
             first(cc) == 1 || last(cc) == scene[2]) && (clipped += 1)
            bpts = scatter(AutoRIFT._block_points(grid, blk))
            wsize = (length(rr), length(cc))
            ok = all(i -> !issearchable(bpts, i) || inbounds(bpts, i, wsize), eachindex(bpts))
            ok && (fits += 1)
        end
        return (clipped, fits, length(keys_))
    end
    for (label, keys_) in ("blocks losing every point" => allgone,
                           "blocks losing some but not all" =>
                               [k for k in keys(byblock) if !(k in allgone)],
                           "a sample of unaffected blocks" =>
                               first([k for k in 1:length(layout.blocks)
                                      if !haskey(byblock, k) && get(total, k, 0) > 0], 200))
        c, f, n = describe(keys_)
        n == 0 && continue
        @printf("  %-32s n=%4d   window clipped at a scene edge %5.1f%%   every point fits its window %5.1f%%\n",
                label, n, 100 * c / n, 100 * f / n)
    end

    # And where they sit, since a rotated footprint puts fill outside it.
    if !isempty(allgone)
        rows = [first(layout.blocks[k].grid_rows) for k in allgone]
        cols = [first(layout.blocks[k].grid_cols) for k in allgone]
        @printf("  blocks losing every point start at grid rows %s, cols %s (grid is %dx%d)\n",
                string(extrema(rows)), string(extrema(cols)), size(grid)...)
        sample = first(sort(allgone), 6)
        for key in sample
            blk = layout.blocks[key]
            bpts = scatter(AutoRIFT._block_points(grid, blk))
            ns = count(i -> issearchable(bpts, i), eachindex(bpts))
            @printf("    block %5d grid %s x %s  window %dx%d  searchable %5d  untiled measured %5d\n",
                    key, string(extrema(blk.grid_rows)), string(extrema(blk.grid_cols)),
                    length(blk.read_rows), length(blk.read_cols), ns, total[key])
        end
    end
    return nothing
end

main()
