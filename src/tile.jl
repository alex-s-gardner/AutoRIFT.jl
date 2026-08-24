# Splitting a scene into blocks, so peak memory tracks the block rather than the scene.
#
# Layer 3: depends on `points.jl` for the grid and on `track.jl` for the pass geometry that sets
# the halo.
#
# ---------------------------------------------------------------------------
# What a block is, and what the halo is for
# ---------------------------------------------------------------------------
#
# A block is a rectangle of the **output grid**. Its read window is that extent grown by the halo
# and clipped to the scene, so the halo changes what a block *reads* and never what it *writes* —
# which is what keeps assembly a plain copy with no blending, averaging, or seam logic.
#
# The halo exists because correlation at a point reads a neighbourhood. Cut a block without one and
# the answers along every internal edge are wrong rather than merely missing, which is worse: they
# look like measurements.
#
# The reaches **add** rather than competing for a maximum, because each stage acts on the previous
# one's output. `_windowmean_dense!` takes its neighbour count from the array bounds
# (`window.jl`), so a pixel within `filter_width ÷ 2` of the read edge is filtered over a truncated
# window where a whole-scene run gave it a full one — and every pixel the correlator reads must
# have been filtered with a full window.
#
# Deliberately *not* in the halo: the outlier filter and hole fill. Their reach compounds across
# iterations, and on the strided coarse grid `dilate_within(keep, coarse_buffer)` alone reaches
# `coarse_buffer * coarse_stride * grid_spacing` — 1024 px at defaults, more than every other term
# together. A halo covering that would be mostly overlap at any block size worth asking for, so
# rejection runs once on the assembled field instead. See `docs/plan-tiling.md`.

"""
    AutoRIFT.Block

One unit of tiled work: the grid points it writes, and the pixels it reads to do so.

`grid_rows`/`grid_cols` index the **output grid**, and every grid point belongs to exactly one
block — so assembling results is a copy, with no point computed twice and none left out.

`read_rows`/`read_cols` index the **scene**, and are the written extent grown by the layout's halo
and clipped to the image. Neighbouring blocks' read windows overlap; that overlap is the halo doing
its job.
"""
struct Block
    grid_rows::UnitRange{Int}
    grid_cols::UnitRange{Int}
    read_rows::UnitRange{Int}
    read_cols::UnitRange{Int}
end

"""
    AutoRIFT.BlockLayout

How a scene is divided for tiled processing: the blocks, and the halo every block reads beyond
what it writes.

`blocks` is a vector of [`AutoRIFT.Block`](@ref) in column-major order over the grid. `halo` is in
pixels, `(x, y)`.
"""
struct BlockLayout
    blocks::Vector{Block}
    halo::Tuple{Int,Int}
end

Base.length(l::BlockLayout) = length(l.blocks)

"""
    AutoRIFT.halo(grid::PointSet, p::Params, imagesize) -> (hx, hy)

Pixels a block must read beyond what it writes, in `(x, y)`.

The sum of the correlation reach — `chip/2 + radius + ceil(abs(prior))`, which
`_pass_geometry` already computes and which this does not re-derive — and the preprocessing
filter's half-width. Both apply to every point, so both are taken at the grid's maximum.

The outlier filter is **not** included: it runs on the assembled field rather than per block,
because its reach compounds past what a halo can absorb. See the note at the top of this file.
"""
function halo(grid::PointSet, p::Params, imagesize::Tuple{Int,Int})
    # `_pass_geometry`'s pad is exactly the correlation reach plus the reference's 2-pixel slack
    # for the half-pixel grid offset and index truncation.
    _, _, _, _, pad, _ = _pass_geometry(scatter(grid), imagesize)
    w = filter_width(p.preprocess) ÷ 2
    return (pad[1] + w, pad[2] + w)
end

"""
    AutoRIFT.block_layout(grid::PointSet{2}, p::Params, imagesize, block_size) -> BlockLayout

Divide `grid` into blocks of at most `block_size = (X, Y)` grid points each, with the read window
each needs.

`block_size` is in **grid points**, not pixels: it names how much output a block produces, which is
what bounds the memory a block needs. The trailing block in each direction is short when the grid
does not divide evenly.

Throws if a block would be smaller than the halo it reads, since such a block is all overlap.
"""
function block_layout(grid::PointSet{2}, p::Params, imagesize::Tuple{Int,Int},
                      block_size::Tuple{Int,Int})
    bx, by = block_size
    (bx > 0 && by > 0) || throw(ArgumentError(
        "`process_block_size` must be positive in both axes, got $bx by $by"))

    hx, hy = halo(grid, p, imagesize)
    spacing = p.grid_spacing
    # The comparison is in pixels, because that is what the halo is in: `bx` grid points span
    # `bx * spacing` pixels of output. A block narrower than its own halo reads more overlap than
    # data, so it is a configuration error rather than something to silently widen.
    (bx * spacing >= hx && by * spacing >= hy) || throw(ArgumentError(
        "`process_block_size` of $bx by $by grid points spans " *
        "$(bx * spacing) by $(by * spacing) pixels, which is smaller than the " *
        "$hx by $hy pixel halo each block must read around it — so the block would be almost " *
        "entirely overlap. Use at least $(cld(hx, spacing)) by $(cld(hy, spacing)) grid " *
        "points, or reduce the chip size, search radius, or filter width."))

    nr, nc = size(grid)
    nrows, ncols = imagesize
    blocks = Block[]
    for c0 in 1:by:nc, r0 in 1:bx:nr
        grows = r0:min(r0 + bx - 1, nr)
        gcols = c0:min(c0 + by - 1, nc)
        # The pixel extent this block writes, from the coordinates of its corner points. Taken from
        # the grid rather than computed from `spacing` so a caller-supplied grid with its own
        # layout still gets a correct window.
        rlo, rhi = _pixel_span(grid.y, grows, gcols)
        clo, chi = _pixel_span(grid.x, grows, gcols)
        push!(blocks, Block(grows, gcols,
                            max(rlo - hy, 1):min(rhi + hy, nrows),
                            max(clo - hx, 1):min(chi + hx, ncols)))
    end
    return BlockLayout(blocks, (hx, hy))
end

# The integer pixel span of a coordinate field over a sub-block of the grid. `floor`/`ceil` rather
# than `round`: the span must contain every point it covers, and a coordinate carrying the
# half-pixel grid offset would otherwise be truncated to the wrong side.
function _pixel_span(coord::AbstractMatrix, rows, cols)
    lo = hi = coord[first(rows), first(cols)]
    @inbounds for j in cols, i in rows
        v = coord[i, j]
        lo = min(lo, v)
        hi = max(hi, v)
    end
    return floor(Int, lo), ceil(Int, hi)
end

"""
    AutoRIFT._read_block(img, rows, cols)

The sub-window `img[rows, cols]`, materialized.

A seam, and the reason it exists is memory rather than clarity: a `Raster` backed by a disk array
reads only the window asked for, so the Rasters extension's method makes peak memory `O(block)`
where this core method is `O(scene)` — the array is already resident. The core cannot do the lazy
read itself without depending on Rasters.

A `copy` rather than a `view` deliberately. The correlator's inner loop wants contiguous memory,
and a strided view of a large scene would make every chip read stride the full row.
"""
_read_block(img::AbstractMatrix, rows, cols) = img[rows, cols]
