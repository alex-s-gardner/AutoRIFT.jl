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
filter's reach. Both apply to every point, so both are taken at the grid's maximum.

The filter term comes from [`AutoRIFT.filter_reach`](@ref) rather than from `filter_width ÷ 2`,
because the two differ: `Wallis` applies two chained window passes and so reaches twice its
half-width.

The outlier filter is **not** included: it runs on the assembled field rather than per block,
because its reach compounds past what a halo can absorb. See the note at the top of this file.

Throws for a preprocessing method with no finite reach, since no halo makes such a filter
blockwise-reproducible.
"""
function halo(grid::PointSet, p::Params, imagesize::Tuple{Int,Int})
    w = filter_reach(p.preprocess)
    # A whole-image reduction, so there is no halo that makes a block agree with the scene. Named
    # here rather than absorbed, because the alternative is a block whose filter output is quietly
    # different from the untiled run's everywhere, not merely at its edge.
    w >= 0 || throw(ArgumentError(
        "`$(nameof(typeof(p.preprocess)))` estimates its correction from the whole image, so a " *
        "block cannot reproduce it from local data and no halo fixes that. Use a windowed filter " *
        "for tiled processing, or run this pair untiled."))
    # `_pass_geometry`'s pad is exactly the correlation reach plus the reference's 2-pixel slack
    # for the half-pixel grid offset and index truncation.
    _, _, _, _, pad, _ = _pass_geometry(scatter(grid), imagesize)
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

# The pair a block sees: its read window out of both images and both masks.
#
# Built through `_read_block` so a disk-backed input reads only this window. The masks come from
# the *raw* pair's fields rather than from `valid`, because intersecting them is `track!`'s job and
# doing it here would produce an `ImagePair` whose two masks are already the same array.
_block_pair(pair::ImagePair, b::Block) = ImagePair(
    _read_block(pair.reference, b.read_rows, b.read_cols),
    _read_block(pair.secondary, b.read_rows, b.read_cols),
    _read_block(pair.reference_valid, b.read_rows, b.read_cols),
    _read_block(pair.secondary_valid, b.read_rows, b.read_cols))

# A block's points, in its read window's coordinate frame.
#
# Grid coordinates are in scene pixels and the block holds a sub-window, so every coordinate shifts
# by the window's origin. The shift is an **integer**, which is what makes this exact: `chip_bounds`
# and `search_bounds` both `floor` a coordinate, and `floor(u - k) == floor(u) - k` for integer `k`,
# so a point lands on the same pixel of the block that it did on the scene. A fractional offset
# would not commute with the truncation and would move every window by a pixel somewhere.
#
# Radii, priors and chip sizes are shared rather than copied — only the coordinates differ.
function _block_points(pts::PointSet{2}, b::Block)
    sub = pts[b.grid_rows, b.grid_cols]
    return rebuild(sub;
                   x = sub.x .- (first(b.read_cols) - 1),
                   y = sub.y .- (first(b.read_rows) - 1))
end

# ---------------------------------------------------------------------------
# The tiled driver
# ---------------------------------------------------------------------------
#
# Same result as `correlate_multichip`, computed a block at a time so peak memory tracks the block
# rather than the scene.
#
# The structure is forced rather than chosen. Correlation is per-point and so can be split; the
# coarse gate, the dilation, the outlier filter and the hole fill are all neighbourhood or
# whole-grid operations, and running them per block would answer a different question in each.
# So each chip-size level is four steps:
#
#   1. per block, correlate the coarse points          — evidence only
#   2. once, on the assembled coarse grid: gate, dilate, resample
#   3. per block, correlate the fine points the mask kept
#   4. once, on the assembled fine field: reject outliers, fill holes, merge
#
# Steps 1 and 3 are where the memory saving lives, and they are the only steps that touch imagery.
# Steps 2 and 4 work on the grid, which is ~1/1024 the scene at the default spacing and stride.

"""
    AutoRIFT.correlate_tiled(pair, grid, p, block_size) -> MultichipResult

[`AutoRIFT.correlate_multichip`](@ref)'s result, computed in blocks of at most `block_size` grid
points so that peak memory tracks the block rather than the scene.

Bit-identical to the untiled path. Every block is handed the whole grid's
[`AutoRIFT.PassGeometry`](@ref), so it runs the transform the untiled pass would have run; and every
decision that looks at more than one point is taken once, on the assembled grid, rather than per
block.
"""
function correlate_tiled(pair::ImagePair, grid::PointSet{2}, p::Params,
                         block_size::Tuple{Int,Int})
    sizes = chip_sizes(p)
    @assert !isempty(sizes) "no chip-size levels for chip_size $(p.chip_size_base) in " *
                            "[$(p.chip_size_min), $(p.chip_size_max)]"
    _check_measures(p, length(sizes))

    layout = block_layout(grid, p, size(pair), block_size)
    sz = size(grid)
    result = MultichipResult(fill(NaN32, sz), fill(NaN32, sz), fill(NaN32, sz),
                             zeros(UInt16, sz), falses(sz))

    for k in eachindex(sizes)
        cs = sizes[k]
        wanted = result.chip_size .== 0
        any(wanted) || break
        level = _tiled_level(pair, grid, p, cs, wanted, measure_at(p, k), layout)
        isnothing(level) && continue
        _merge_level!(result, level.field, level.filled, cs)
    end
    return result
end

# One chip-size level, in blocks. The block-parallel steps are 1 and 3; everything else is global.
function _tiled_level(pair::ImagePair, grid::PointSet{2}, p::Params, chip_size::Integer,
                      wanted::AbstractMatrix{Bool}, measure::SimilarityMeasure,
                      layout::BlockLayout)
    csx = Int(chip_size)
    csy = chip_size_y(p, csx)

    pts = _level_points(grid, p, csx, csy, wanted)
    nsearchable(pts) == 0 && return nothing

    setup = _coarse_points(pts, p, csx, csy)
    # A coarse grid too small for the filter to judge consistency on. The untiled path searches
    # everything here rather than rejecting on no evidence, and that is defensible for a whole
    # scene. Under tiling it is a configuration error: it would skip the coarse restriction and
    # search every point at full radius, which is ~100x the work with no diagnostic.
    isnothing(setup) && throw(ArgumentError(
        "the coarse grid for chip size $csx is smaller than the outlier filter's window, so the " *
        "coarse pass cannot judge which points are coherent. Untiled, this falls back to " *
        "searching everything; tiled it would do so silently and at roughly a hundred times the " *
        "cost. Use a larger scene, a smaller `grid_spacing`, or a smaller `coarse_stride`."))
    coarse = setup.coarse
    nsearchable(coarse) == 0 && return nothing

    # Step 1, per block: the coarse evidence. Every block runs the transform the whole coarse pass
    # would have run, which is what `geometry` is for.
    cgeom = pass_geometry(coarse)
    cd = _run_blocked(pair, coarse, p, layout, _coarse_block_layout(layout, setup, size(pts)),
                      cgeom, measure; subpixel = NoRefine())

    # Step 2, once: the decisions.
    mask = _coarse_decide(cd, coarse, p, setup.filt, size(pts))
    isnothing(mask) && return nothing

    @inbounds for i in eachindex(pts)
        if !mask[i]
            pts.radius_x[i] = 0
            pts.radius_y[i] = 0
        end
    end
    nsearchable(pts) == 0 && return nothing

    # Step 3, per block: the fine pass.
    fgeom = pass_geometry(pts)
    fine = _run_blocked(pair, pts, p, layout, layout.blocks, fgeom, measure)

    # Step 4, once: reject, fill, and hand back for the merge.
    filled = _reject_and_fill!(fine, pts, p)
    return (; field = fine, filled)
end

# Correlate `pts` block by block, writing into one field.
#
# `blocks` partitions `pts`'s index space; `geometry` is the whole set's, so every block runs the
# transform the untiled pass would have run. Each block reads its own window, shifts its points into
# that window's frame, and writes into its own slice of the output — so no two blocks touch the same
# element and the result is assembled rather than reduced.
#
# One task per block with `threaded = false` inside it, which is the shape `src/api.jl` already
# documents for batch work: one unit of work per task beats threading within a unit. Plans are warmed
# before spawning, because FFTW's planner is not thread-safe.
function _run_blocked(pair::ImagePair, pts::PointSet{2}, p::Params, layout::BlockLayout,
                      blocks::Vector{Block}, geometry::PassGeometry,
                      measure::SimilarityMeasure;
                      subpixel::SubpixelMethod = p.subpixel)
    out = displacement_field(pts)
    # Every block shares the geometry, so one warm-up serves all of them — and it must happen here,
    # on this task, rather than inside a block.
    _warm_pass_plans(geometry.chip_x, geometry.chip_y, geometry.radius_x, geometry.radius_y,
                     measure)

    serial = _serial_params(p)
    if istrue(p.threaded)
        tasks = map(blocks) do b
            StableTasks.@spawn _run_one_block!(out, pair, pts, serial, b, geometry, measure,
                                              subpixel)
        end
        foreach(wait, tasks)
    else
        for b in blocks
            _run_one_block!(out, pair, pts, serial, b, geometry, measure, subpixel)
        end
    end
    return out
end

function _run_one_block!(out::DisplacementField, pair::ImagePair, pts::PointSet{2},
                         p::Params, b::Block, geometry::PassGeometry,
                         measure::SimilarityMeasure, subpixel::SubpixelMethod)
    bpts = _block_points(pts, b)
    # A block all of whose points a previous level resolved, or which the coarse mask emptied.
    nsearchable(bpts) == 0 && return out
    bpair = _block_pair(pair, b)
    bout = displacement_field(bpts)
    track!(bout, bpair, bpts, p; subpixel, measure, geometry)
    # Assembly is a copy: the halo grew what this block read, never what it writes.
    out.dx[b.grid_rows, b.grid_cols] .= bout.dx
    out.dy[b.grid_rows, b.grid_cols] .= bout.dy
    out.correlation[b.grid_rows, b.grid_cols] .= bout.correlation
    out.searched[b.grid_rows, b.grid_cols] .= bout.searched
    return out
end

# The same `Params` with threading off, for use inside a block task. Threading belongs at the block
# level: nesting the intra-pass threading inside per-block tasks would oversubscribe, and
# `src/api.jl:146-157` records the measurement that one unit per task wins.
#
# Built by walking `fieldnames` rather than by listing 22 positional arguments, so adding a field to
# `Params` cannot silently drop it here — the same failure `rebuild` exists to prevent for
# `PointSet`. `@generated` so the splat is resolved at compile time and the result stays concretely
# typed, which `--trim` requires and which a runtime `map` over `fieldnames` would not give.
@generated function _params_serial(p::Params)
    args = [name === :threaded ? :(False()) : :(getfield(p, $(QuoteNode(name))))
            for name in fieldnames(p)]
    return :(Params($(args...)))
end

_serial_params(p::Params) = istrue(p.threaded) ? _params_serial(p) : p

# Blocks over the *coarse* grid, derived from the fine-grid layout.
#
# The coarse grid is the fine grid strided by `coarse_stride`, so a fine-grid block maps to whichever
# coarse points fall inside it. Deriving them rather than laying out afresh is what guarantees the
# two partitions agree: every coarse point belongs to exactly one block, and to the same block its
# fine neighbours do.
#
# `rows`/`cols` are the strided indices `_coarse_points` selected, so `searchsortedfirst` finds where
# each fine-grid range begins in them.
function _coarse_block_layout(layout::BlockLayout, setup, fine_size::Tuple{Int,Int})
    rows, cols = setup.rows, setup.cols
    out = Block[]
    for b in layout.blocks
        r0 = searchsortedfirst(rows, first(b.grid_rows))
        r1 = searchsortedlast(rows, last(b.grid_rows))
        c0 = searchsortedfirst(cols, first(b.grid_cols))
        c1 = searchsortedlast(cols, last(b.grid_cols))
        # A block containing no coarse point contributes nothing to the coarse pass. It still gets
        # a fine pass, because the coarse mask covers it through the resample.
        (r1 >= r0 && c1 >= c0) || continue
        push!(out, Block(r0:r1, c0:c1, b.read_rows, b.read_cols))
    end
    return out
end
