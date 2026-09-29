# Where are the points a blocked run moves, and is the geometry it gives them the geometry the untiled
# run gave them?
#
#   julia --project=tools/golden -t 10,1 tools/golden/block_moved.jl S2A_MSIL1C_20200626T204021 768 --run 200
#
# `block_agreement.jl` counts the disagreement and classifies it as seam, numerical or localized from the
# *lost* points' seam distance. On the cases that fail below their default block size nothing is lost —
# S2A at 768 px moves 31 points and loses none — so that discriminator reads an empty set and the
# classification comes from the wrong population. This asks the same questions of the points that moved.
#
# It also checks, without correlating anything, the one difference a block introduces that the halo does
# not describe: `_block_points` rebases every coordinate into the block's read window, and `chip_bounds`
# then computes `floor(x - dx_prior)` in that frame rather than the scene's. The two framings are the same
# real number and not the same floating-point sum, so a point whose `x - dx_prior` sits within a rounding
# of an integer can take a chip one pixel away from the one an untiled run takes. Counted here per block
# so the arithmetic is either implicated or excluded before anything more expensive is tried.
#
# Both runs' fields are serialized, so every later question about the same arm is free.

using Printf, Statistics, Serialization

include(joinpath(@__DIR__, "correlator.jl"))
using AutoRIFT: block_layout, halo, nsearchable, chip_bounds, search_bounds, issearchable,
                scatter, _block_points

argvalue(flag, default) = (i = findfirst(==(flag), ARGS);
                           isnothing(i) ? default : ARGS[i + 1])

# Does rebasing a point into a block's read window change the window the correlator cuts for it?
#
# `chip_bounds` and `search_bounds` are evaluated on the block's own point set and shifted back to scene
# coordinates, which is the composition a blocked run performs. Any point where that differs from the
# untiled evaluation is a point the two runs correlate at different positions.
function rebasing_shifts(grid, layout)
    whole = scatter(grid)
    shifted = Int[]
    for blk in layout.blocks
        bpts = scatter(_block_points(grid, blk))
        # `_block_points` slices `grid[blk.grid_rows, blk.grid_cols]`, so point `k` of the block is this
        # linear index of the grid.
        cart = CartesianIndices(size(grid))
        lin = LinearIndices(size(grid))
        rows, cols = blk.grid_rows, blk.grid_cols
        k = 0
        for c in cols, r in rows
            k += 1
            i = lin[r, c]
            issearchable(whole, i) || continue
            cb = chip_bounds(whole, i)
            sb = search_bounds(whole, i)
            bcb = chip_bounds(bpts, k)
            bsb = search_bounds(bpts, k)
            dr, dc = first(blk.read_rows) - 1, first(blk.read_cols) - 1
            same = first(bcb[1]) + dr == first(cb[1]) && first(bcb[2]) + dc == first(cb[2]) &&
                   first(bsb[1]) + dr == first(sb[1]) && first(bsb[2]) + dc == first(sb[2])
            same || push!(shifted, i)
        end
    end
    return shifted
end

function main()
    length(ARGS) >= 2 || error("usage: block_moved.jl <product-fragment> <block-px> [--run N]")
    frag, block = ARGS[1], parse(Int, ARGS[2])
    call = parse(Int, argvalue("--run", "100"))

    c = only(cases(frag))
    k = read_capture(c; n = call, mmap = CAPTURE_IMAGERY)
    grid = pointset_from_capture(k)
    kw = kwargs_from_capture(k)
    a, b = k.arrays["in_I1"], k.arrays["in_I2"]
    p = params(; kw...)
    scene = size(a)
    bs = (block, block)
    layout = block_layout(grid, p, scene, bs)
    @printf("%s\n  scene %dx%d  grid %dx%d  halo %dx%d  %s imagery  block %d px, %d blocks\n",
            c.product, scene..., size(grid)..., halo(grid, p, scene).X, halo(grid, p, scene).Y,
            eltype(a), block, length(layout.blocks))

    # Free, and it either implicates the rebasing arithmetic or removes it from the list.
    shifted = rebasing_shifts(grid, layout)
    @printf("\nrebasing into a block's frame shifts the cut window at %d of %d searchable points\n",
            length(shifted), nsearchable(grid))
    flush(stdout)

    whole = autorift(b, a, grid; kw...)
    tiled = autorift(b, a, grid; kw..., process_block_size = bs)

    out = joinpath(mkpath(joinpath(@__DIR__, "results", "blockmoved")),
                   "$(c.product)_$(block).jls")
    serialize(out, (; product = c.product, block, call,
                    whole = (dx = whole.dx, dy = whole.dy, correlation = whole.correlation,
                             chip_size = whole.chip_size),
                    tiled = (dx = tiled.dx, dy = tiled.dy, correlation = tiled.correlation,
                             chip_size = tiled.chip_size),
                    shifted, gridsize = size(grid),
                    blocks = [(blk.grid_rows, blk.grid_cols, blk.read_rows, blk.read_cols)
                              for blk in layout.blocks]))
    println("\nfields serialized to $out")

    differ = [i for i in eachindex(whole.dx)
              if !isequal(whole.dx[i], tiled.dx[i]) || !isequal(whole.dy[i], tiled.dy[i])]
    @printf("\n%d points differ on dx or dy\n", length(differ))
    isempty(differ) && return 0

    # Seam or not, asked of the points that actually differ. `block_agreement.jl` asks it of the lost.
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
    measured = [i for i in eachindex(whole.dx) if isfinite(whole.dx[i])]
    ddiff, dall = edge_distance.(differ), edge_distance.(measured)
    @printf("\ndistance to own block's grid edge, in grid points:\n")
    for (label, d) in ("differing" => ddiff, "all measured" => dall)
        @printf("  %-12s median %5.1f   at 0-1: %5.1f%%   at 0: %5.1f%%   n = %d\n",
                label, median(d), 100 * count(<=(1), d) / length(d),
                100 * count(==(0), d) / length(d), length(d))
    end

    # Which level answered. A chip size difference says the disagreement is about *which* level
    # answered rather than what one of them found.
    ncs = count(i -> !isequal(whole.chip_size[i], tiled.chip_size[i]), differ)
    @printf("\nchip size differs at %d of the %d differing points\n", ncs, length(differ))
    @printf("chip sizes there: untiled %s, blocked %s\n",
            string(sort(unique(whole.chip_size[differ]))),
            string(sort(unique(tiled.chip_size[differ]))))

    # How many of the differing points are ones the rebasing arithmetic already flagged.
    inshift = count(in(Set(shifted)), differ)
    @printf("\n%d of the %d differing points are among the %d the rebasing shifts\n",
            inshift, length(differ), length(shifted))

    # Per point, so the pattern is visible rather than summarized away.
    @printf("\n%-8s %-11s %-6s %-5s %5s %9s %9s %9s %9s %7s %7s\n",
            "index", "grid r,c", "block", "edge", "chip", "dx untiled", "dx blocked",
            "dy untiled", "dy blocked", "corr u", "corr b")
    for i in first(differ, 40)
        row, col = Tuple(cart[i])
        @printf("%-8d %-11s %-6d %-5d %5d %9.4f %9.4f %9.4f %9.4f %7.4f %7.4f\n",
                i, "$row,$col", owner[i], edge_distance(i), Int(whole.chip_size[i]),
                whole.dx[i], tiled.dx[i], whole.dy[i], tiled.dy[i],
                whole.correlation[i], tiled.correlation[i])
    end
    return 0
end

exit(main())
