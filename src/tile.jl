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
# `coarse_buffer * coarse_stride * oversample * grid_spacing` — 2048 px at defaults, more than every
# other term together. A halo covering that would be mostly overlap at any block size worth asking
# for, so rejection runs once on the assembled field instead. See `plan-tiling.md`.

"""
    AutoRIFT.Block

One unit of tiled work: the grid points it writes, and the pixels it reads to do so.

`grid_rows`/`grid_cols` index the **output grid**, and every grid point belongs to exactly one
block — so assembling results is a copy, with no point computed twice and none left out.

`read_rows`/`read_cols` index the **scene**: the written extent grown by the layout's halo and clipped
to the image, over the points the block may search. Neighbouring blocks' read windows overlap, and that
overlap is the halo doing its job.

They are a **bound**, not the window a pass reads. A block's searchable coordinates need not be
contiguous, and on a geogrid they are not, so `AutoRIFT._read_windows` derives the windows a pass
actually reads from the point set it is about to correlate and tiles them to fit these bounds. That is
what makes this figure the one `AutoRIFT.block_buffers` sizes from.
"""
struct Block
    grid_rows::UnitRange{Int}
    grid_cols::UnitRange{Int}
    read_rows::UnitRange{Int}
    read_cols::UnitRange{Int}
end

"""
    AutoRIFT.ReadWindow

One window of scene pixels a block reads, as `rows` by `cols`.

A block holds one per tile of its points' coordinate span — usually exactly one. See
`AutoRIFT._read_windows`.
"""
struct ReadWindow
    rows::UnitRange{Int}
    cols::UnitRange{Int}
end

"""
    AutoRIFT.BlockLayout

How a scene is divided for tiled processing: the blocks, and the halo every block reads beyond
what it writes.

`blocks` is a vector of [`AutoRIFT.Block`](@ref) in column-major order over the grid. `halo` is an
[`AutoRIFT.Extent`](@ref) in pixels.
"""
struct BlockLayout
    blocks::Vector{Block}
    halo::Extent
end

Base.length(l::BlockLayout) = length(l.blocks)

"""
    AutoRIFT.halo(grid::PointSet, p::Params, imagesize) -> Extent

Pixels a block must read beyond what it writes, as an [`AutoRIFT.Extent`](@ref).

The sum of the correlation reach — `chip/2 + radius + ceil(abs(prior))`, which
`_pass_geometry` already computes and which this does not re-derive — and the preprocessing
filter's reach. Both apply to every point, so both are taken at their maximum over the run.

Measured against the *coarsest level*, not against `grid` as supplied. A level overwrites the
chip size and floors the radii, so a grid's own `chip_size_x` and `radius_x` are not what any
pass runs — see `AutoRIFT._worst_level_points`.

The filter term comes from [`AutoRIFT.filter_reach`](@ref) rather than from `filter_width ÷ 2`,
because the two differ: `Wallis` applies two chained window passes and so reaches twice its
half-width.

The outlier filter is **not** included: it runs on the assembled field rather than per block,
because its reach compounds past what a halo can absorb. See the note at the top of this file.

Throws for a preprocessing method with no finite reach, since no halo makes such a filter
blockwise-reproducible.
"""
function halo(grid::PointSet, p::Params, imagesize::Tuple{Int,Int})
    w = _filter_halo(p)
    # `_pass_geometry`'s pad is exactly the correlation reach plus the reference's 2-pixel slack
    # for the half-pixel grid offset and index truncation.
    _, _, pad, _ = _pass_geometry(_worst_level_points(grid, p), imagesize)
    ox, oy = _level_centre_offset(p)
    return Extent((pad.X + w + ox, pad.Y + w + oy))
end

"""
    AutoRIFT.halo(p::Params) -> Extent

The halo for a grid [`AutoRIFT.gridpoints`](@ref) would build from `p`, without building it.

`p` alone determines the answer for such a grid: every point carries the same chip size and radius,
so the maxima `halo(grid, p, imagesize)` reduces over are the values in `p` — floored at
`min_search_radius`, exactly as [`AutoRIFT.sanitize!`](@ref) floors them. Given that, the grid
contributes nothing the parameters do not already state.

Exists because a caller choosing a block size needs the halo *before* the run, and building a grid to
ask costs 550 MiB and 50 ms on a Landsat-sized scene — for two integers. The grid form remains the
one to use for a caller-supplied `PointSet`, where per-point radii and priors may vary and the maxima
are genuinely a property of the points.
"""
function halo(p::Params)
    w = _filter_halo(p)
    # The same arithmetic `_pass_geometry` performs per point, with the maxima taken from `p`: half the
    # widest chip, plus the radius, plus the prior, plus the reference's 2-pixel slack.
    rx = max(p.search_radius.X, p.min_search_radius)
    ry = max(p.search_radius.Y, p.min_search_radius)
    hx = p.chip_size_max.X ÷ 2 + rx + ceil(Int, abs(p.dx_prior)) + 2
    hy = p.chip_size_max.Y ÷ 2 + ry + ceil(Int, abs(p.dy_prior)) + 2
    # A decimated level correlates at its cells' centres, which reach further than the grid points the
    # rest of this arithmetic is derived from.
    ox, oy = _level_centre_offset(p)
    return Extent((hx + w + ox, hy + w + oy))
end

# The preprocessing filter's contribution to the halo, and the one configuration that has none.
#
# A whole-image reduction has no halo that makes a block agree with the scene. Named rather than
# absorbed, because the alternative is a block whose filter output is quietly different from the
# untiled run's everywhere, not merely at its edge.
function _filter_halo(p::Params)
    w = filter_reach(p.preprocess)
    w >= 0 || throw(ArgumentError(
        "`$(nameof(typeof(p.preprocess)))` estimates its correction from the whole image, so a " *
        "block cannot reproduce it from local data and no halo fixes that. Use a windowed filter " *
        "for tiled processing, or run this pair untiled."))
    return w
end

"""
    AutoRIFT._worst_level_points(grid::PointSet, p::Params) -> PointSet{1}

`grid`'s points carrying the largest chip size and radius any level will run them at.

The halo has to cover every level, and a level's geometry is not the grid's. `chipsize_level`
ignores `grid.chip_size_x` and sets its own from [`AutoRIFT.chip_sizes`](@ref), so the reach is
set by `chip_size_max` however the caller sized the grid; and `_level_points` ends in
[`AutoRIFT.sanitize!`](@ref), which floors every non-zero radius at `p.min_search_radius`, so a
grid whose radii sit below that floor is searched wider than it asks for.

A third correction applies to a gridded set: the coarse pass's `_cell_max_radius!` gives a point the
widest radius in its neighbourhood, so a point the grid marks unsearchable is searched anyway, and
`_pass_geometry` skips exactly those. Their radii are widened here so it does not — see
`AutoRIFT._widen_for_halo!` for why their *priors* are what makes this matter.

Both corrections matter only for a grid the caller built: [`AutoRIFT.autorift`](@ref)'s own grid
takes `chip_size = p.chip_size_max` already, and its radii come from the same keywords the floor
is compared against. A grid built at a smaller chip size is the case that breaks — at
`chip_size = 32` against `chip_size_max = 128` the coarsest level reaches 46 px further per axis
than the grid implies, and a block sized to the grid reads too little to reproduce it.

A `PointSet{1}`, since `_pass_geometry` reduces over points and does not use the layout.
"""
function _worst_level_points(grid::PointSet, p::Params)
    flat = scatter(grid)
    cs = p.chip_size_max
    n = size(flat.radius_x)
    # Copies, because `sanitize!` writes in place and `scatter` shares the grid's own arrays. The chip
    # sizes are constant over the set, so they are `Uniform` rather than two more grid-sized arrays — see
    # `_level_points`, which makes the same substitution for the same reason.
    pts = rebuild(flat; radius_x = copy(flat.radius_x), radius_y = copy(flat.radius_y),
                  chip_size_x = Uniform(cs.X, n),
                  chip_size_y = Uniform(cs.Y, n))
    _widen_for_halo!(pts, grid, p)
    # The same floor a level applies, applied by the same function, so the two cannot drift.
    sanitize!(pts, p.min_search_radius)
    return pts
end

# Give the points `_cell_max_radius!` can make searchable the grid's widest radius, so `_pass_geometry`
# reduces over them too.
#
# Their **priors** are why this is needed. `_pass_geometry` skips a point with a zero radius, so such a
# point contributes nothing to the halo — yet the widening can hand it a neighbour's radius, after which
# the pass reaches `chip/2 + radius + ceil(abs(prior))` from it. The radius it receives is bounded by one
# already counted; its prior is not, and on a 24-day Sentinel-1 pair a prior is ~18 px of range.
#
# Widening the radii here rather than computing the extra reach separately is what keeps one copy of that
# arithmetic: `_pass_geometry` is the only place it lives, and a second implementation would have to be
# kept in step with it by hand. The widest radius in the grid is used rather than the one each point would
# actually receive, which cannot inflate the halo through the radius term — that is already at its maximum
# over the searchable points — and saves reproducing the neighbourhood reduction.
#
# Mutates `pts`, whose radii `_worst_level_points` has already copied, so nothing reaches the caller's
# grid. A no-op without a grid layout for a neighbourhood to be taken over.
_widen_for_halo!(::PointSet, ::PointSet, ::Params) = nothing

function _widen_for_halo!(pts::PointSet{1}, grid::PointSet{2}, p::Params)
    wide = vec(_widened_searchable(grid, _widening_reach(p)))
    mx = my = 0
    for i in eachindex(pts)
        issearchable(pts, i) || continue
        mx = max(mx, pts.radius_x[i])
        my = max(my, pts.radius_y[i])
    end
    (mx == 0 || my == 0) && return nothing
    for i in eachindex(pts.radius_x, wide)
        (wide[i] && !issearchable(pts, i)) || continue
        pts.radius_x[i] = mx
        pts.radius_y[i] = my
    end
    return nothing
end

"""
    AutoRIFT._level_centre_offset(p::Params) -> (ox, oy)

Pixels past its own grid point that the coarsest level's correlation reaches, from sitting at a cell
centre.

A decimated level correlates at the centre of each `stride`-by-`stride` cell rather than at the cell's
first grid point (see `AutoRIFT._cell_means`), which is up to `(stride - 1) / 2` cells further along
each axis. A halo derived from grid points alone is short by that much, and a block would read too
little to reproduce the untiled run — visible only as a last-bits difference in the peak height at
points near a block seam, since the displacement itself survives a slightly different transform.

Taken over every level rather than at `chip_size_max`, because the stride is set by the chip-to-spacing
ratio and it is the *largest* offset that has to fit, whichever level produces it.

Half a pixel is added for the parity snap `AutoRIFT._cell_means` applies, which can move a node
that much further out again.
"""
function _level_centre_offset(p::Params)
    sx = sy = 0.0
    for cs in chip_sizes(p)
        stride = _level_decimation(p, cs)
        stride <= 1 && continue
        half = (stride - 1) / 2
        # `+ 0.5` for the snap to the chip's pixel parity, which rounds the centre either way.
        sx = max(sx, half * p.grid_spacing.X + 0.5)
        sy = max(sy, half * p.grid_spacing.Y + 0.5)
    end
    return (ceil(Int, sx), ceil(Int, sy))
end

# `want` raised to a whole multiple of `unit`, which is how a size is aligned to the storage's own
# granularity: a block size to the file's chunk grid, a slab height to the chunk rows a read cannot
# subdivide. A non-positive `unit` is no constraint and returns `want` unchanged.
_round_up(want::Integer, unit::Integer) = unit <= 0 ? want : cld(want, unit) * unit

"""
    AutoRIFT.block_layout(grid::PointSet{2}, p::Params, imagesize, block_size) -> BlockLayout

Divide `grid` into blocks spanning at most `block_size = (X, Y)` **pixels** of output each, with the
read window each needs.

Pixels, because pixels are what blocking bounds. A block's output is negligible — 3.2 KiB against
the 6.7 MiB of imagery it reads at a 512-pixel block, and the whole scene's output grid is smaller
than one block's read window — so the quantity worth naming is the imagery held at once. It is also
the unit everything here already works in: the halo, the read windows, and the buffers are all
pixels.

A block must be a whole number of grid points, so the size is a target: the block holds as many points
as fit within it. The count comes from the span a trial block of the grid actually covers rather than
from `grid_spacing`, because a caller-supplied grid need be neither uniformly spaced nor axis-aligned —
a geogrid built from a rotated radar footprint moves both coordinates along both index directions. The
trailing block in each direction is short when the grid does not divide evenly.

A block's read window spans only the points it will search — the grid's searchable points grown by the
reach of the coarse pass's radius widening, which is what any pass can actually reach. A block with no
searchable point reads nothing at all.

Points whose coordinate is a placeholder rather than a position get a **second** window, because a
placeholder lies nowhere near the block's own extent and one window covering both would run from the
block to the scene corner. See [`AutoRIFT.Block`](@ref) and `PointSet`'s `positioned`.

Throws if a block would be smaller than the halo it reads, since such a block is all overlap.
"""
function block_layout(grid::PointSet{2}, p::Params, imagesize::Tuple{Int,Int},
                      block_size)
    # Normalized by the same helper the `process_block_size` keyword uses, rather than by `extent`
    # directly: this is public API in its own right, and `extent` would read a scalar as a square
    # block — which is exactly what that helper exists to refuse.
    bs = _block_size(block_size)
    px, py = bs.X, bs.Y
    (px > 0 && py > 0) || throw(ArgumentError(
        "`process_block_size` must be positive in both axes, got $px by $py pixels"))

    h = halo(grid, p, imagesize)
    hx, hy = h.X, h.Y
    # A block narrower than its own halo reads more overlap than data, so it is a configuration
    # error rather than something to silently widen. Compared directly, both being pixels.
    (px >= hx && py >= hy) || throw(ArgumentError(
        "`process_block_size` of $px by $py pixels is smaller than the $hx by $hy pixel halo each " *
        "block must read around it — so the block would be almost entirely overlap. Use at least " *
        "$hx by $hy pixels, or reduce the chip size, search radius, or filter width."))

    nr, nc = size(grid)
    nrows, ncols = imagesize
    # How many grid points fit in a block along each index axis, measured from the grid rather than
    # walked along one row and one column: a grid's coordinates need not be separable, and a rotated
    # footprint moves both `x` and `y` along either index direction. See `_block_shape`.
    rowpts, colpts = _block_shape(grid, px, py)
    rowstarts = collect(1:rowpts:nr)
    colstarts = collect(1:colpts:nc)

    # Which points any pass can search, once for the whole grid: the coarse pass widens a radius over a
    # neighbourhood, so this is wider than the grid's own searchable set. Two bit arrays at 1/8 byte per
    # point — 0.9 MiB each on the 7.5M-point NISAR grid.
    wide = _widened_searchable(grid, _widening_reach(p))

    blocks = Block[]
    for (ci, c0) in pairs(colstarts), (ri, r0) in pairs(rowstarts)
        grows = r0:(ri == lastindex(rowstarts) ? nr : rowstarts[ri + 1] - 1)
        gcols = c0:(ci == lastindex(colstarts) ? nc : colstarts[ci + 1] - 1)
        # The pixel extent this block writes, from the coordinates of the points it will actually
        # correlate. Taken from the grid rather than computed from `spacing` so a caller-supplied grid
        # with its own layout still gets a correct window.
        #
        # Twice, because a block's points fall into two populations whose coordinates are nowhere near
        # each other: those with a real position, and those whose coordinate is a placeholder for a point
        # outside the footprint. One window spanning both runs from the block to the scene corner.
        #
        # Over the points with a real position only. A placeholder coordinate is excluded *here* because
        # this figure sizes the buffers: spanning it would make the bound the whole scene, where
        # `_read_windows` gives the placeholder population a tile of its own at run time. Sizing from the
        # block's nominal extent instead would cost 1.9x on S1B, since a block's points rarely fill it.
        rlo, rhi, clo, chi = _coordinate_span(grid, grows, gcols, wide, true)
        # A block with nothing to search reads nothing. `_run_one_block!` returns before any I/O for
        # such a block, so the window only has to be empty rather than meaningful.
        rows = isnothing(rlo) ? (1:0) : _clip(rlo - hy, rhi + hy, nrows)
        cols = isnothing(rlo) ? (1:0) : _clip(clo - hx, chi + hx, ncols)
        push!(blocks, Block(grows, gcols, rows, cols))
    end
    return BlockLayout(blocks, h)
end

# Grid points per block along each index axis, as `(rowpts, colpts)`, so a block's read window respects
# `px` by `py` pixels.
#
# This replaces walking one row and one column of the coordinates. That shortcut assumed the grid was
# *separable* — `x` constant down every column, `y` constant across every row — which holds for a grid
# `gridpoints` builds and fails for a rotated footprint sampled onto a map grid. On a NISAR L1 geogrid
# row 1 and column 1 hold one point with real coordinates and fill everywhere else, so walking them
# spanned 216 px instead of the scene and concluded one block covers 57760x50511. Every requested block
# size from 3072 to 16384 px returned a single block — an untiled run wearing a block size, with none of
# the memory bound the caller asked for.
#
# The grid *is* the index-to-pixel mapping, so the four rates below are read from it rather than
# estimated: `rxi` is how far `x` moves per row of index, `rxj` per column, and likewise `ryi`/`ryj` for
# `y`. A block of `a` rows by `b` columns then spans about `a*rxi + b*rxj` pixels of `x` and
# `a*ryi + b*ryj` of `y`, and both have to fit their budget.
#
# **Both index directions charge both axes, which is what a separable calculation gets wrong.** Sizing
# the rows from the `y` budget alone and the columns from the `x` budget alone gives 215 by 124 points at
# an 8192-pixel request on the NISAR grid, whose `x` span is 11187 px — a 37% overshoot of the figure the
# caller asked for. Scaling one factor until both constraints hold is what keeps the request a bound.
#
# On an axis-aligned grid `rxi` and `ryj` are zero, the two constraints decouple, and this reduces to
# `py / ryi` rows by `px / rxj` columns — the separable answer, so a Landsat layout is unchanged. That
# includes a full-width band, where `px` is the scene width and only the row count binds.
_block_shape(grid::PointSet{2}, px::Int, py::Int) =
    _block_shape(_index_rates(grid), size(grid), px, py)

# The four rates, measured once.
#
# Separated from the shape because choosing a block size searches over candidate sizes, and each rate
# is a reduction over the grid: measuring them per candidate costs more than the search it serves.
_index_rates(grid::PointSet{2}) =
    (rxi = _index_rate(grid, grid.x, 1), rxj = _index_rate(grid, grid.x, 2),
     ryi = _index_rate(grid, grid.y, 1), ryj = _index_rate(grid, grid.y, 2))

function _block_shape(r::NamedTuple, gridsize::Tuple{Integer,Integer}, px::Int, py::Int)
    nr, nc = Int(gridsize[1]), Int(gridsize[2])
    rxi, rxj, ryi, ryj = r.rxi, r.rxj, r.ryi, r.ryj
    # The separable answer, from each axis's own dominant direction. Also the starting point for the
    # coupled case, since shrinking from here can only tighten a constraint that already holds.
    rowpts = clamp(floor(Int, py / max(ryi, rxi, EPS_RATE)), 1, nr)
    colpts = clamp(floor(Int, px / max(rxj, ryj, EPS_RATE)), 1, nc)
    # Shrink both together until each axis's total span fits. Halving rather than solving the pair of
    # inequalities exactly: the shape is a target the block snaps to anyway, and a bounded loop cannot
    # be defeated by a degenerate rate the way a division can.
    for _ in 1:MAX_SHRINK
        (rowpts * rxi + colpts * rxj <= px && rowpts * ryi + colpts * ryj <= py) && break
        (rowpts == 1 && colpts == 1) && break
        rowpts = max(1, rowpts ÷ 2)
        colpts = max(1, colpts ÷ 2)
    end
    return (rowpts, colpts)
end

# Blocks a `px` by `py` budget divides a grid of `gridsize` into, by the same arithmetic
# `block_layout` uses and without building one. The count, not the shape, is what a block size is
# chosen against.
function _block_count(rates::NamedTuple, gridsize::Tuple{Integer,Integer}, px::Int, py::Int)
    rowpts, colpts = _block_shape(rates, gridsize, px, py)
    return cld(Int(gridsize[1]), rowpts) * cld(Int(gridsize[2]), colpts)
end

"""
    AutoRIFT.block_size_for(grid::PointSet{2}, p::Params, imagesize; kw...) -> Extent
    AutoRIFT.block_size_for(p::Params, imagesize; kw...) -> Extent

A `process_block_size` for this configuration: `AutoRIFT.BLOCK_HALO_MULTIPLE` times the halo in each
axis, floored at `AutoRIFT.BLOCK_FLOOR` pixels and clamped to the scene.

Keywords: `chunk` as `(rows, cols)` of the storage's own grid, and `floor_pixels`.

**Blocks are the unit of threaded work and the halo is a fixed-width skirt on every one**, so the two
ends of the range fail for different reasons and the best block is interior to them. Too small and a
block reads mostly skirt: on the golden S1B case, halo 684x256, a 768x320 block costs a read
amplification of 7.23x for a *worse* peak than 1024 — 3.24 GiB against 3.01. Too large and the buffers
dominate, at 10.32 GiB by 2048, and one block becomes too much work to balance: on NISAR L1 at 6144 px
a single block holds 19.8% of the granule and 2.37x what a thread should carry, so twelve threads
deliver five.

**The rule is fitted to a measured sweep of every golden case**, not to a model.
`tools/golden/block_optimum.jl` walks each case's ladder and keeps only the arms that reproduce that
case's untiled run, so every row scored is answer-preserving. Against each case's own best arm, this
rule costs a mean of 1.10x and a worst of 1.35x the runtime, and a mean of +0.19 GiB and a worst of
+0.65 GiB of peak — and it is the best of the rules tried under *both* objectives, which do not
otherwise agree. A fixed size cannot do it: the per-case optima span 128 px to 6144 px and
`2240x1152`, and the ratio of the best block to the halo runs from 1.0x to 6.1x.

**Why a multiple of the halo rather than a block count.** An earlier form maximized the size subject
to `blocks_per_thread * nthreads` pieces, and the count it produced does not track the optimum: at the
measured best arm the blocks per thread run from 7 to 3,330 across the set. The halo is what sets both
failure modes above, so it is what the rule is expressed in.

The `Params`-only method is for a caller who has no grid yet, and assumes the grid
[`AutoRIFT.gridpoints`](@ref) would build — uniform radii, axis-aligned, `p.grid_spacing` apart —
exactly as [`AutoRIFT.halo(p)`](@ref) does, and for the same reason: building a grid to choose a block
size costs 550 MiB on a Landsat-sized scene to produce two integers.
"""
block_size_for(grid::PointSet{2}, p::Params, imagesize::Tuple{Integer,Integer}; kw...) =
    _block_size_for(halo(grid, p, imagesize), imagesize; kw...)

block_size_for(p::Params, imagesize::Tuple{Integer,Integer}; kw...) =
    _block_size_for(halo(p), imagesize; kw...)

function _block_size_for(h::Extent, imagesize::Tuple{Integer,Integer};
                         chunk::Tuple{Integer,Integer} = (1, 1),
                         floor_pixels::Integer = 1)
    nrows, ncols = Int(imagesize[1]), Int(imagesize[2])
    # `chunk` is indexed as `imagesize` is — `(rows, cols)` — while a halo and a block size are
    # `(x, y)`, so the axes cross exactly once, here.
    #
    # Clamped to the scene, since a block past it reads the whole thing once and is an untiled run
    # wearing a block size.
    return Extent((min(_round_up(max(BLOCK_HALO_MULTIPLE * h.X, BLOCK_FLOOR, h.X, floor_pixels),
                                chunk[2]), ncols),
                   min(_round_up(max(BLOCK_HALO_MULTIPLE * h.Y, BLOCK_FLOOR, h.Y, floor_pixels),
                                chunk[1]), nrows)))
end

# Multiples of the halo to make a block, and the floor below which the halo stops being the binding
# term. Both are fitted to `tools/golden/block_optimum.jl`'s sweep of all 22 golden cases, which
# measured each case's whole block-size ladder and kept only the arms reproducing that case's untiled
# run. See `block_size_for` for the scoring.
const BLOCK_HALO_MULTIPLE = 2
const BLOCK_FLOOR = 1024



# Halvings allowed while fitting a block to its budget. Twenty takes any grid to a single point, so the
# loop terminates on its own rather than on this bound; it exists so a pathological rate cannot spin.
const MAX_SHRINK = 20

# A rate below this is treated as absent rather than divided by, so an axis a grid does not vary along
# cannot produce an infinite block.
const EPS_RATE = 1e-9

# Pixels of `A` per unit step of index dimension `dim`, measured over the points a block must cover.
#
# **Reduced over pairs where both points are searchable**, which is what makes this robust on a grid whose
# footprint is rotated within its bounding box. Such a grid is mostly fill — 65% of both NISAR grids — and
# this cannot know the fill convention, since they pad with zeros rather than `NaN` and a finiteness test
# finds nothing. Restricting to searchable pairs excludes fill by construction instead of by guessing a
# fill value, and it is the right restriction on its own terms: an unsearchable point is never correlated, so
# its coordinate does not constrain a block.
#
# Both alternatives were measured and both fail, in opposite directions:
#
#   * **The median of the nonzero steps** cannot see a genuinely separable axis. On the L2 GSLC grid `x`
#     really is constant down a column, so every in-footprint step is zero and the only nonzero ones are
#     the two that cross the fill boundary — giving `dx/di = 87666` from a sample of one, and a layout of
#     5.2 million blocks, one per grid point.
#   * **The median of every step** is zero on any grid that is majority fill, which is both NISAR grids.
#     All four rates come out zero, and a zero rate is an infinite block.
#
# Over searchable pairs the same estimator gives 33/34/19/19 px on the rotated L1 grid and 0/48/24/0 on the
# separable L2 grid — the true rates in both cases, with the cross terms vanishing exactly where they
# should.
#
# Subsampled, because this is an estimate of a spacing and reducing over five million points to produce it
# costs more than the layout it informs.
#
# Returns zero when nothing can be measured, which `_block_shape` reads as "this axis does not vary".
function _index_rate(grid::PointSet{2}, A::AbstractMatrix, dim::Int)
    nr, nc = size(A)
    (dim == 1 ? nr : nc) > 1 || return 0.0
    steps = Float64[]
    stride = max(1, (nr * nc) ÷ RATE_SAMPLES)
    ilast = dim == 1 ? nr - 1 : nr
    jlast = dim == 2 ? nc - 1 : nc
    k = 0
    @inbounds for j in 1:jlast, i in 1:ilast
        k += 1
        k % stride == 0 || continue
        i2, j2 = dim == 1 ? (i + 1, j) : (i, j + 1)
        (issearchable(grid, CartesianIndex(i, j)) &&
         issearchable(grid, CartesianIndex(i2, j2))) || continue
        d = Float64(A[i2, j2]) - Float64(A[i, j])
        isfinite(d) && push!(steps, abs(d))
    end
    isempty(steps) && return 0.0
    sort!(steps)
    n = length(steps)
    return isodd(n) ? steps[(n + 1) ÷ 2] : (steps[n ÷ 2] + steps[n ÷ 2 + 1]) / 2
end

# Differences to sample when estimating an index rate. A median over this many is stable far past the
# precision a block layout needs, and it bounds the cost on a five-million-point grid.
const RATE_SAMPLES = 20_000

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

# The pixel window a block must read for one of its two point populations, as `(rlo, rhi, clo, chi)`, or
# four `nothing`s when the block has none of them.
#
# `keep` selects the points any pass can search — `AutoRIFT._widened_searchable`, not the grid's own
# radii — and `want` picks the population by `positioned`, so the two calls per block partition its
# points between the two read windows.
#
# Reduced over a subset rather than over every point, which is what `AutoRIFT._pixel_span` would do. A
# point with a zero radius is never correlated (`issearchable`), so no imagery has to be read for it —
# and on a grid whose footprint is rotated within its bounding box, those points carry a *placeholder*
# coordinate rather than a plausible one. Spanning them together with the real ones is not merely
# wasteful, it is wrong by the width of the scene: a block straddling the footprint edge holds a
# placeholder at 0 and real coordinates in the tens of thousands, so its window becomes `1:57760` — the
# whole scene, read once per such block. Measured on a NISAR L1 grid at an 8192-pixel block, 28 of 209
# blocks each read half the scene or more, for 99x the scene in total. Splitting on `want` is what lets
# the placeholder population have a window of its own instead.
#
# `NaN` coordinates are skipped for the same reason: they cannot bound a window, and `min`/`max` would
# poison the whole span.
function _coordinate_span(grid::PointSet{2}, rows, cols, keep::AbstractMatrix{Bool}, want::Bool)
    rlo = clo = Inf
    rhi = chi = -Inf
    for j in cols, i in rows
        keep[i, j] || continue
        grid.positioned[i, j] == want || continue
        y, x = grid.y[i, j], grid.x[i, j]
        (isfinite(y) && isfinite(x)) || continue
        rlo = min(rlo, y); rhi = max(rhi, y)
        clo = min(clo, x); chi = max(chi, x)
    end
    isfinite(rlo) || return (nothing, nothing, nothing, nothing)
    return (floor(Int, rlo), ceil(Int, rhi), floor(Int, clo), ceil(Int, chi))
end

# Which points any pass can search: the grid's searchable points grown by `AutoRIFT._widening_reach` in
# both index directions.
#
# The grid's own radii are not the answer. `_cell_max_radius!` gives a coarse node the widest radius in
# its neighbourhood, so a point the grid marks unsearchable is searched anyway when a searchable one is
# near enough — and a window sized from the grid's radii alone then falls short. On the golden S1B case
# that costs 20,704 of 26,781 points a blocked run otherwise loses.
#
# Dilation is separable, so this is two linear sweeps per axis rather than a `(2·reach + 1)²` stencil:
# at a reach of 32 over 7.5M points the stencil form is 7 billion writes and takes tens of seconds,
# where this is O(points). Each axis sweeps forward tracking the last set index and backward tracking
# the next, which is a box dilation exactly.
#
# The second axis reads a separate buffer rather than working in place: writing a `true` that a later
# iteration of the same sweep then reads would let the mask grow without bound instead of by `reach`.
function _widened_searchable(grid::PointSet{2}, reach::Int)
    nr, nc = size(grid)
    src = falses(nr, nc)
    for j in 1:nc, i in 1:nr
        src[i, j] = issearchable(grid, CartesianIndex(i, j))
    end
    reach <= 0 && return src
    mid = falses(nr, nc)
    for i in 1:nr
        _dilate_line!(view(mid, i, :), view(src, i, :), reach)
    end
    out = falses(nr, nc)
    for j in 1:nc
        _dilate_line!(view(out, :, j), view(mid, :, j), reach)
    end
    return out
end

# One axis of a box dilation: `dst[k]` is true where `src` is true anywhere within `reach` of `k`.
function _dilate_line!(dst, src, reach::Int)
    n = length(src)
    last = -1
    for k in 1:n
        src[k] && (last = k)
        dst[k] = last >= 0 && k - last <= reach
    end
    next = n + reach + 1
    for k in n:-1:1
        src[k] && (next = k)
        dst[k] |= next - k <= reach
    end
    return dst
end

"""
    AutoRIFT._read_block(img, rows, cols)

The sub-window `img[rows, cols]`, materialized.

The allocating counterpart of [`AutoRIFT._read_block!`](@ref), which is what the driver uses. A
disk-backed `img` reads only this window, so no method for a lazy array is needed — indexing is
already the windowed read.

A `copy` rather than a `view` deliberately. The correlator's inner loop wants contiguous memory,
and a strided view of a large scene would make every chip read stride the full row.
"""
_read_block(img::AbstractMatrix, rows, cols) = img[rows, cols]

"""
    AutoRIFT.BlockBuffers

Reusable storage for one block's raw imagery, so a run allocates one block's worth rather than one
per block — and one set per *run*, not per pass: the layout fixes the largest window, so the coarse
and fine passes of every level want identically-sized arrays.

Peak resident memory is what an instance's limit sees, and a blocked run holds no more *live* data
than an untiled one — measured live-heap growth is ~1 MiB either way. What it does is churn: a fresh
block pair per block, 22.9 MiB each at a 888-pixel read window, 1467 MiB over 64 blocks. The
collector returns none of that to the OS promptly, so `Sys.maxrss` records the high-water mark of
churn rather than a requirement, and a scheduler packing jobs onto a small instance is bounded by
exactly that figure.

Sized to the largest read window in the layout. Smaller blocks — the trailing ones where the grid
does not divide evenly — take a view of the corner, which is why nothing here is sized per block.
Not thread-safe: one set per task, the same contract as a correlation workspace.
"""
struct BlockBuffers{T,M}
    reference::Matrix{T}
    secondary::Matrix{T}
    reference_valid::Matrix{M}
    secondary_valid::Matrix{M}
    # The filter's output for each image, and one shared scratch array for the NaN-encoded copy
    # `_masked_boxmean!` needs when a mask excludes something. `Float32` because every filter
    # produces signed values whatever the input type.
    #
    # Scratch is shared between the two images because the two filter calls are sequential: the
    # reference's is complete before the secondary's begins, so the array is dead in between. One
    # per image would be correct and would waste half of it.
    filtered_reference::Matrix{Float32}
    filtered_secondary::Matrix{Float32}
    filter_scratch::Matrix{Float32}
    # Eroded masks. `_filtered` shrinks a mask by the filter width, so these cannot alias the raw
    # masks above — `track!` intersects the pair's two masks and would then be reading a mask that
    # its own filtering had already narrowed.
    filtered_reference_valid::Matrix{M}
    filtered_secondary_valid::Matrix{M}
end

"""
    AutoRIFT.block_buffers(pair::ImagePair, layout::BlockLayout) -> BlockBuffers

Storage for the largest block in `layout`, to be reused across all of them.
"""
function block_buffers(pair::ImagePair, layout::BlockLayout)
    nr = maximum(length(b.read_rows) for b in layout.blocks)
    nc = maximum(length(b.read_cols) for b in layout.blocks)
    T = eltype(pair)
    return BlockBuffers(Matrix{T}(undef, nr, nc), Matrix{T}(undef, nr, nc),
                        Matrix{Bool}(undef, nr, nc), Matrix{Bool}(undef, nr, nc),
                        Matrix{Float32}(undef, nr, nc), Matrix{Float32}(undef, nr, nc),
                        Matrix{Float32}(undef, nr, nc),
                        Matrix{Bool}(undef, nr, nc), Matrix{Bool}(undef, nr, nc))
end

# A block's raw pair, read into reusable storage.
#
# `_read_block!` writes into a view of the buffer rather than returning a fresh array, so a run's
# raw-read cost is one block's worth however many blocks there are. The views are what let one
# buffer serve a short trailing block as well as a full interior one.
function _block_pair!(buf::BlockBuffers, pair::ImagePair, rows, cols)
    nr, nc = length(rows), length(cols)
    r = @view buf.reference[1:nr, 1:nc]
    s = @view buf.secondary[1:nr, 1:nc]
    rv = @view buf.reference_valid[1:nr, 1:nc]
    sv = @view buf.secondary_valid[1:nr, 1:nc]
    _read_block!(r, pair.reference, rows, cols)
    _read_block!(s, pair.secondary, rows, cols)
    # After the imagery, because a mask derived from an image is computed from the window just read
    # rather than read again.
    _read_mask_block!(rv, pair.reference_valid, pair.reference, r, rows, cols)
    _read_mask_block!(sv, pair.secondary_valid, pair.secondary, s, rows, cols)
    return ImagePair(r, s, rv, sv)
end

# A mask window: read from `mask`, unless it is `img`'s own finiteness — in which case `window`, the
# image window already in the buffer, is the same data and `isfinite` over it is the same answer.
#
# `ImagePair` defaults each valid mask to `FiniteMask` of its image, so without this the common case
# reads every window of a disk-backed pair twice: once as imagery, once through the mask's own
# `getindex`, which asks the same array for the same rows and columns. Measured on a chunked 1024²
# pair at blocks of 256, exactly 2.00x the elements the windows span.
#
# [`AutoRIFT._derived_from`](@ref) is the same test [`AutoRIFT._mask_bytes`](@ref) applies, so what the
# estimate charges and what the read costs cannot diverge. A mask over different storage — a caller's
# own, or a nodata mask over unfilled values — is not derivable and is read.
function _read_mask_block!(dest::AbstractMatrix, mask::AbstractMatrix, img::AbstractMatrix,
                           window::AbstractMatrix, rows, cols)
    _derived_from(mask, img) || return _read_block!(dest, mask, rows, cols)
    dest .= isfinite.(window)
    return dest
end

"""
    AutoRIFT._read_block!(dest, img, rows, cols)

Read `img[rows, cols]` into `dest`, which must have that shape.

The in-place counterpart of [`AutoRIFT._read_block`](@ref), and the form the tiled driver uses so a
run allocates one block's storage rather than one per block. A lazy input's method reads only this
window from disk.
"""
function _read_block!(dest::AbstractMatrix, img::AbstractMatrix, rows, cols)
    size(dest) == (length(rows), length(cols)) || throw(DimensionMismatch(
        "destination is $(size(dest)) but the window is $((length(rows), length(cols)))"))
    return _read_window!(dest, img, rows, cols)
end

# The read itself, after the shape is checked.
#
# Split from the check so a backend can specialize the read without restating it —
# `AutoRIFTDiskArraysExt` reads a window straight into `dest`, which is the same indexing path with
# somewhere to put the result and no block-sized temporary.
#
# `img[rows, cols]`, not `copyto!(dest, view(img, rows, cols))`. A view defers the read, so `copyto!`
# then walks it element by element — and for a lazy array that is one read per pixel rather than one
# read per window. Measured on a lazy GeoTIFF: 99.7% of a blocked run's time was a scalar `getindex`
# reached from here, and a 512² region with one block took 444 s against 0.6 s from memory. Indexing
# asks the array for the whole window, which is the operation a chunked backend is built to serve.
#
# The extra allocation is a block-sized temporary, and a `DiskArrays` backend takes the specialized path
# instead.
_read_window!(dest::AbstractMatrix, img::AbstractMatrix, rows, cols) =
    (copyto!(dest, img[rows, cols]); dest)

# A strided parent has nothing to defer, so it takes the view.
#
# The reason the generic method above must not is that a view of a *lazy* array defers the read, and
# `copyto!` then walks it element by element — one read per pixel rather than one per window. A
# `StridedMatrix` is memory-backed by construction, so the temporary buys nothing and costs a
# window-sized allocation per read. On a whole NISAR L2 grid at a 2304x1152 block that is half the run's
# remaining allocation, and the copy is measurably faster without it.
_read_window!(dest::AbstractMatrix, img::StridedMatrix, rows, cols) =
    (copyto!(dest, view(img, rows, cols)); dest)

# The *filtered* pair a block sees, from raw input.
#
# Filtering per block rather than once over the scene is what makes tiling save memory at all: a
# whole-scene `_prepare` leaves the filtered pair resident, and a block copied out of it then costs
# more than it saves. Filtering the block instead means the scene is never materialized.
#
# `pair` must be **raw**. This filters what it is given, so a pair `_prepare` has already filtered
# comes back filtered twice — not an edge effect but a wrong image everywhere, and one that still
# looks like imagery.
#
# Exact, and the halo is why: `filter_reach` is the neighbourhood a filter's output depends on, and
# `halo` includes it, so a block's filtered values *and* its eroded mask agree with a whole-scene
# filter everywhere the block writes. The mask matters as much as the values, since `_filtered`
# erodes by the filter width and that erosion must not bite at a read edge where the untiled run had
# data.
function _prepared_block_pair(buf::BlockBuffers, pair::ImagePair, rows, cols, p::Params)
    raw = _block_pair!(buf, pair, rows, cols)
    return _prepare_block(buf, raw, p, p.preprocess, length(rows), length(cols))
end

# Filtering a block into pooled storage, for the filters that have an in-place form.
#
# `Highpass` is the one that matters — the default, and the only filter the production driver runs
# inside the correlator — so it is the one pooled. Anything else falls through to the allocating
# path below: correct, and costing a filter output per block, which is worth having as the honest
# fallback rather than blocking those filters from tiled runs.
function _prepare_block(buf::BlockBuffers, raw::ImagePair, p::Params, m::Highpass,
                        nr::Int, nc::Int)
    w = filter_width(m)
    fr = @view buf.filtered_reference[1:nr, 1:nc]
    fs = @view buf.filtered_secondary[1:nr, 1:nc]
    scratch = @view buf.filter_scratch[1:nr, 1:nc]
    rv = @view buf.filtered_reference_valid[1:nr, 1:nc]
    sv = @view buf.filtered_secondary_valid[1:nr, 1:nc]

    highpass!(fr, raw.reference, raw.reference_valid, w; scratch)
    highpass!(fs, raw.secondary, raw.secondary_valid, w; scratch)
    # `_filtered!` does the mask bookkeeping every windowed filter needs: erode by the filter
    # width, and drop any pixel whose filtered value came out non-finite.
    _filtered!(fr, rv, raw.reference_valid, w)
    _filtered!(fs, sv, raw.secondary_valid, w)
    # No `replace_nonfinite` pass: `_filtered!` already zeroed the non-finite values and recorded
    # them in the mask, which is exactly what that function does.
    return ImagePair(fr, fs, rv, sv)
end

# No filter: the block's raw pair is already what the pass correlates.
#
# `_prepare` would reach it by copying twice — `preprocess` of `NoPreprocess` is `copy(img), copy(mask)`
# and `replace_nonfinite` copies the pair again — to produce arrays `_block_pair!` has just written into
# `buf`. Four window-sized allocations per image per block, and on a wide read window that is most of a
# blocked run's allocation: 1282 GiB against 181 on a whole NISAR L2 grid at a 2304x1152 block.
#
# **The non-finite substitution still has to happen**, and skipping it is not a cosmetic difference. A
# float pair carrying no-data — which is what reprojection to a common grid leaves — then hands `NaN` to
# the transform: measured on a 768² pair with a no-data border and an interior hole, returning `raw`
# unchanged loses 122 of 6843 points and changes `dx` where it does not. `replace_nonfinite!` writes into
# the buffer instead of allocating, and is a no-op by dispatch for an integer image, which cannot hold a
# non-finite value at all.
function _prepare_block(::BlockBuffers, raw::ImagePair, ::Params, ::NoPreprocess, ::Int, ::Int)
    replace_nonfinite!(raw.reference, raw.reference_valid)
    replace_nonfinite!(raw.secondary, raw.secondary_valid)
    return raw
end

# Every other filter: allocate, as the untiled path does. Bit-identical either way.
_prepare_block(::BlockBuffers, raw::ImagePair, p::Params, ::PreprocessMethod, ::Int, ::Int) =
    _prepare(raw, p)

# The first point of a block whose search window leaves the block's read window on a side that is not the
# scene's own edge, or `0` when there is none.
#
# The halo is the maximum correlation reach over every point, and a block's window is its points' span
# grown by it, so such a point means the pass reaches further than the layout was built from. `track!` then
# takes its `fits = false` branch and zero-pads the shortfall, correlating against padding where a
# whole-scene run had imagery — a different answer at a real point, and a silent one. A side clipped to the
# scene's own edge is exempt, because a whole-scene run pads there too.
#
# **A point whose own coordinate lies outside the window is not reported here.** Each population is
# correlated against the window that holds its own coordinates — a placeholder coordinate against the
# placeholder window — so a point outside the window under test belongs to the other sub-pass and its
# reach says nothing about this one. Mixing the two would bury a real shortfall of a few hundred pixels
# under a spurious one of forty thousand.
#
# `search_bounds` rather than re-deriving the reach, so this and `inbounds` cannot disagree about what a
# point needs. Returns an index rather than the bounds so that nothing is allocated or formatted on the
# path that finds nothing — which is every block of a grid `gridpoints` built, and is why this reads the
# point set in place rather than flattening it.
function _block_window_shortfall(rows, cols, pts::PointSet, imagesize::Tuple{Int,Int})
    nrows, ncols = length(rows), length(cols)
    (nrows == 0 || ncols == 0) && return 0
    # Sides the window was clipped on, where padding is what an untiled run does as well.
    lo_row = first(rows) == 1
    hi_row = last(rows) == imagesize[1]
    lo_col = first(cols) == 1
    hi_col = last(cols) == imagesize[2]
    # Indexed directly rather than through `scatter`: `search_bounds`, `inbounds` and `issearchable` are
    # already generic over a `PointSet` of any dimension, and `eachindex` is linear over either, so
    # flattening would allocate eleven array views per call for nothing.
    for i in eachindex(pts)
        issearchable(pts, i) || continue
        inbounds(pts, i, (nrows, ncols)) && continue
        # Belongs to another window rather than short of halo: the coordinate is outside this one.
        (1 <= pts.x[i] <= ncols && 1 <= pts.y[i] <= nrows) || continue
        rows, cols = search_bounds(pts, i)
        ((first(rows) < 1 && !lo_row) || (last(rows) > nrows && !hi_row) ||
         (first(cols) < 1 && !lo_col) || (last(cols) > ncols && !hi_col)) || continue
        return i
    end
    return 0
end

# Said once per session, not once per block: a grid that has one such point has thousands, and the finding
# is that the configuration has them at all.
#
# A warning and not an error, which is a judgement rather than caution. Every geogrid case reaches this —
# both NISAR granules and every Sentinel-1 one — so throwing would take the blocked path from quietly
# inexact to unusable on exactly the cases that need it, and blocking is the only route that fits them in
# 16 GiB. `_warn_coarse_fallback` is the same call made for the same reason a few lines above.
function _warn_block_window(wrows, wcols, pts::PointSet, i::Int)
    rows, cols = search_bounds(pts, i)
    # Components interpolated one at a time: showing a `Tuple` reaches `Base.repeat` through `textwidth`,
    # which `--trim` cannot resolve — the same constraint `src/track.jl` records for its messages.
    @warn("a point reaches past its block's read window away from the scene's edge, so this blocked run " *
          "correlates padding where a whole-scene run read imagery and its result differs there. The " *
          "halo is the maximum reach over every point and a block's window is its points' span grown by " *
          "it, so a pass reaching further means the layout was built from a narrower point set than the " *
          "pass runs — on a geogrid that is the coarse pass, whose `_cell_max_radius!` gives a point the " *
          "widest radius in its neighbourhood. See `dev/plan-16gib.md`. Reported once per session.",
          window_rows = length(wrows), window_cols = length(wcols),
          point_rows_from = first(rows), point_rows_to = last(rows),
          point_cols_from = first(cols), point_cols_to = last(cols), maxlog = 1)
    return nothing
end

# The read windows one block needs for the point set a pass is about to correlate, as
# `(windows, assign)` — or `nothing` when one window covers everything, which is the common case and
# every case on a grid `gridpoints` built.
#
# **A block's searchable coordinates need not be contiguous, and on a geogrid they are not.**
# `_cell_means` places a decimated node at the mean coordinate over its cell and averages in the
# placeholder that stands for a point outside the footprint, so a cell straddling the footprint edge
# yields a node somewhere between the swath and the placeholder. The coordinates a block must read
# therefore form a spread rather than a neighbourhood, and one window spanning them is most of the scene:
# 1:57760 on a NISAR L1 grid, read once per such block. `dev/CORRECTNESS.md` item 2 is the defect behind
# the spread; this covers it rather than waiting for it.
#
# **Clustered first, and tiled only where a cluster still does not fit the buffers.** Clustering is
# bucketing at halo resolution and flood-filling the occupied buckets, which is linear in the point count
# where pairwise box merging is quadratic. Two points in adjacent buckets join, so the rule over-merges by
# up to a halo against exact single linkage — the safe direction, since it yields fewer and larger windows
# and cannot leave a point uncovered. It is also the right rule, because a gap narrower than the halo is
# already being read.
#
# Tiling a cluster is the fallback rather than the mechanism, and the order matters. Tiling everything at
# `buffer - 2 * halo` shatters a *compact* cluster wherever the halo is a large fraction of the block: on
# the golden S2B case the halo is 281x267 against a 650x618 block, so the tile is 56x116 and a block that
# needs one window gets dozens. Clustering first leaves such a block at one window and splits only what is
# genuinely spread.
#
# Every window still fits the buffers by construction, which is what keeps the buffer the memory bound
# rather than the data: `process_block_size` means what it says however the coordinates are distributed,
# and the cost of a spread is more sub-passes rather than more memory. Left unbounded the largest window on
# NISAR L2 is 4.84x what the layout sizes for, which would not fit 16 GiB.
function _read_windows(pts::PointSet{2}, b::Block, h::Extent, imagesize::Tuple{Int,Int},
                       maxrows::Int, maxcols::Int)
    # A tile is shrunk by the halo on both sides, since the window is the tile's own span grown by it, and
    # by a further 2 for the `floor`/`ceil` that turn a fractional span into whole pixels. Without that
    # slack a tile whose span is exactly the quantum yields a window two pixels past the buffer, which
    # surfaces as a `BoundsError` inside a block task rather than anywhere useful.
    prows, pcols = maxrows - 2 * h.Y - 2, maxcols - 2 * h.X - 2
    (prows >= 1 && pcols >= 1) || throw(ArgumentError(
        "a read buffer of $maxrows by $maxcols pixels cannot hold a window for a $(h.Y) by $(h.X) " *
        "pixel halo; `process_block_size` must be at least the halo in each axis"))

    rlo = clo = Inf
    rhi = chi = -Inf
    for j in b.grid_cols, i in b.grid_rows
        issearchable(pts, CartesianIndex(i, j)) || continue
        y, x = pts.y[i, j], pts.x[i, j]
        (isfinite(y) && isfinite(x)) || continue
        rlo = min(rlo, y); rhi = max(rhi, y)
        clo = min(clo, x); chi = max(chi, x)
    end
    # Nothing to search. `_run_one_block!` returns before any I/O, so no window is needed at all.
    isfinite(rlo) || return (ReadWindow[], nothing)

    # The whole span fits one tile, so every point shares a window and no assignment is needed. This is
    # the common case and the only one on a grid `gridpoints` built, and it is byte-for-byte the single
    # window a block read before this existed.
    if _window_fits(rlo, rhi, clo, chi, h, imagesize, maxrows, maxcols)
        return ([ReadWindow(_clip(floor(Int, rlo) - h.Y, ceil(Int, rhi) + h.Y, imagesize[1]),
                            _clip(floor(Int, clo) - h.X, ceil(Int, chi) + h.X, imagesize[2]))],
                nothing)
    end

    # One pass collecting the points worth reading, so the guard is applied once rather than at every
    # stage below. `at` is each point's bucket, which the clustering and the assignment both key on.
    by, bx = max(h.Y, 1), max(h.X, 1)
    nbi = fld(floor(Int, rhi - rlo), by) + 1
    nbj = fld(floor(Int, chi - clo), bx) + 1
    occupied = falses(nbi, nbj)
    idx = Int[]
    ys = Float64[]
    xs = Float64[]
    at = Tuple{Int,Int}[]
    lin = LinearIndices((length(b.grid_rows), length(b.grid_cols)))
    for (jj, j) in enumerate(b.grid_cols), (ii, i) in enumerate(b.grid_rows)
        issearchable(pts, CartesianIndex(i, j)) || continue
        y, x = pts.y[i, j], pts.x[i, j]
        (isfinite(y) && isfinite(x)) || continue
        bucket = (fld(floor(Int, y - rlo), by) + 1, fld(floor(Int, x - clo), bx) + 1)
        occupied[bucket...] = true
        push!(idx, lin[ii, jj]); push!(ys, y); push!(xs, x); push!(at, bucket)
    end
    comp = zeros(Int32, nbi, nbj)
    ncomp = 0
    stack = Tuple{Int,Int}[]
    for bj in 1:nbj, bi in 1:nbi
        (occupied[bi, bj] && comp[bi, bj] == 0) || continue
        ncomp += 1
        comp[bi, bj] = ncomp
        push!(stack, (bi, bj))
        while !isempty(stack)
            ci, cj = pop!(stack)
            for dj in -1:1, di in -1:1
                ni, nj = ci + di, cj + dj
                (1 <= ni <= nbi && 1 <= nj <= nbj) || continue
                (occupied[ni, nj] && comp[ni, nj] == 0) || continue
                comp[ni, nj] = ncomp
                push!(stack, (ni, nj))
            end
        end
    end

    # Each cluster's own span, which decides whether it needs tiling at all.
    cspan = fill((Inf, -Inf, Inf, -Inf), ncomp)
    for n in eachindex(idx)
        c = comp[at[n]...]
        s = cspan[c]
        cspan[c] = (min(s[1], ys[n]), max(s[2], ys[n]), min(s[3], xs[n]), max(s[4], xs[n]))
    end
    # A cluster that fits is one window whatever its shape; only a cluster wider than the buffers is cut,
    # and then from its own origin so the cut does not depend on where the block starts.
    fits = [_window_fits(s[1], s[2], s[3], s[4], h, imagesize, maxrows, maxcols) for s in cspan]

    # Group the points by cluster, and within an oversized cluster by tile. Window numbers follow first
    # encounter over a fixed order, so the result is deterministic even though the lookup is a `Dict` —
    # a blocked run has to be reproducible to the bit, and `Dict` iteration order is not.
    seen = Dict{Tuple{Int32,Int,Int},Int}()
    spans = NTuple{4,Float64}[]
    assign = zeros(Int16, length(b.grid_rows), length(b.grid_cols))
    for n in eachindex(idx)
        y, x = ys[n], xs[n]
        c = comp[at[n]...]
        s = cspan[c]
        key = fits[c] ? (c, 0, 0) :
              (c, fld(floor(Int, y - s[1]), prows), fld(floor(Int, x - s[3]), pcols))
        w = get(seen, key, 0)
        if w == 0
            push!(spans, (y, y, x, x))
            w = length(spans)
            seen[key] = w
        else
            t = spans[w]
            spans[w] = (min(t[1], y), max(t[2], y), min(t[3], x), max(t[4], x))
        end
        assign[idx[n]] = w
    end
    windows = [ReadWindow(_clip(floor(Int, s[1]) - h.Y, ceil(Int, s[2]) + h.Y, imagesize[1]),
                          _clip(floor(Int, s[3]) - h.X, ceil(Int, s[4]) + h.X, imagesize[2]))
               for s in spans]
    # The buffers are the memory bound, so a window past them is a defect here rather than something for
    # `_block_pair!` to discover: it surfaces there as a `BoundsError` raised on a worker task, several
    # frames from anything that names a window.
    for w in windows
        (length(w.rows) <= maxrows && length(w.cols) <= maxcols) || error(
            "a read window of $(length(w.rows)) by $(length(w.cols)) pixels exceeds the $maxrows by " *
            "$maxcols pixel buffers it must be read into; the tile quantum is $prows by $pcols")
    end
    return (windows, assign)
end

_clip(lo::Int, hi::Int, n::Int) = max(lo, 1):min(hi, n)

# Whether the window a coordinate span needs fits the buffers, tested on the **window** rather than on the
# span: the window is the span grown by the halo and snapped outward with `floor`/`ceil`, so a span two
# pixels inside the bound can still need a window two pixels past it.
function _window_fits(rlo, rhi, clo, chi, h::Extent, imagesize::Tuple{Int,Int},
                      maxrows::Int, maxcols::Int)
    rows = _clip(floor(Int, rlo) - h.Y, ceil(Int, rhi) + h.Y, imagesize[1])
    cols = _clip(floor(Int, clo) - h.X, ceil(Int, chi) + h.X, imagesize[2])
    return length(rows) <= maxrows && length(cols) <= maxcols
end

# A block's points, in the coordinate frame of one of its read windows.
#
# Grid coordinates are in scene pixels and the block holds a sub-window, so every coordinate shifts
# by that window's origin. The shift is an **integer**, which is what makes this exact: `chip_bounds`
# and `search_bounds` both `floor` a coordinate, and `floor(u - k) == floor(u) - k` for integer `k`,
# so a point lands on the same pixel of the block that it did on the scene. A fractional offset
# would not commute with the truncation and would move every window by a pixel somewhere.
#
# Radii, priors and chip sizes are shared rather than copied — only the coordinates differ.
function _block_points(pts::PointSet{2}, b::Block, rows, cols)
    sub = pts[b.grid_rows, b.grid_cols]
    return rebuild(sub;
                   x = sub.x .- (first(cols) - 1),
                   y = sub.y .- (first(rows) - 1))
end

# The block's own window, which is the one every point with a real position is correlated against.
_block_points(pts::PointSet{2}, b::Block) = _block_points(pts, b, b.read_rows, b.read_cols)

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
    AutoRIFT.Blocked(raw, layout, blocks, buffers, cache_budget)

Correlate each pass a block at a time. See `AutoRIFT.PassRunner`.

`raw` is the **unfiltered** pair, and that is what makes blocking bound memory rather than merely
reorganize it: each block is filtered from its own read window, so the filtered scene the whole-scene
runner holds is never formed.

`blocks` is the partition this runner correlates over — `layout.blocks` for a fine pass, and the
strided subset `AutoRIFT._coarse_block_layout` derives for a coarse one. `layout` is kept
alongside it because `block_buffers` sizes from the whole layout's largest read window, which no pass
changes. `buffers` is `nothing` for a threaded run, where each task takes its own set.

`cache_budget` is the bytes of disk-backed input that may be read into memory up front, carried so that
`reinit!` decides the same way for a later pair as `init` did for the first; `0` keeps the windowed
reads. `raw` is already whatever that decision produced — see [`AutoRIFT._prefetched`](@ref).

!!! warning "`blocks` indexes one grid, and only that grid"
    A `Block` holds *grid index ranges*, so this runner is bound to the grid shape its partition was
    built from. Passing [`AutoRIFT.run_pass`](@ref) a point set of any other shape indexes those
    ranges into the wrong array and throws `BoundsError` from `_block_points` — it does not silently
    correlate the wrong points, but nor is it caught at the call.

    Anything that changes the grid shape — the coarse stride, or a per-level decimation — must
    therefore go through [`AutoRIFT.restrict`](@ref) to re-derive the partition first. That is what
    `restrict` is for, and it is the whole reason it exists as a method on the runner rather than as
    a step inside the coarse pass.
"""
struct Blocked{P<:ImagePair,B<:Union{Nothing,BlockBuffers}} <: PassRunner
    raw::P
    layout::BlockLayout
    blocks::Vector{Block}
    buffers::B
    # Bytes of disk-backed input this runner may read up front, so `reinit!` decides the same way for
    # a pair swapped in later as `init` did for the first one. See `AutoRIFT._prefetched`.
    cache_budget::Int
end

# `pass_geometry(pts)` is computed here rather than by the caller: it is a mechanism of blocking —
# every block must run the transform the whole set would have — and the whole-scene runner has no use
# for it, so it does not belong in a signature the two share.
run_pass(r::Blocked, pts::PointSet{2}, p::Params, measure::SimilarityMeasure,
         subpixel::SubpixelMethod) =
    _run_blocked(r.raw, pts, p, r.layout, r.blocks, pass_geometry(pts), measure, r.buffers,
                 subpixel)

# A pass over a strided subset — the coarse pass, or a decimated chip-size level — so a blocked
# runner has to re-derive which of its blocks hold which of the surviving points. The points
# themselves are already narrowed by the caller; what needs narrowing here is the *partition* of them.
#
# Derived from `r.blocks` rather than `r.layout.blocks`, so restricting twice composes: a decimated
# level restricts, and its coarse pass restricts that result again. `r.layout` is carried through
# untouched because it sizes the buffers from the largest read window, which no striding changes.
restrict(r::Blocked, setup, _gridsize::Tuple{Int,Int}) =
    Blocked(r.raw, r.layout, _coarse_block_layout(r.blocks, setup), r.buffers, r.cache_budget)

# Loud, unlike the whole-scene runner: blocking is asked for when the scene will not fit, so a coarse
# grid too small to filter means every point is searched at full radius — roughly a hundred times the
# work the coarse pass exists to avoid — and silence there reads as a fast run.
#
# The chip size is logged as two integers rather than as the extent itself: showing a `NamedTuple`
# reaches `Base.repeat` via `textwidth`, which `--trim` cannot resolve — the same constraint
# `src/track.jl` records for error messages.
_warn_coarse_fallback(::Blocked, chip_size::Extent) = @warn(
    "coarse grid smaller than the outlier filter's window; searching every point at full radius, " *
    "which costs roughly 100x the restricted pass. Reduce `coarse_stride`, reduce `grid_spacing`, " *
    "or process a larger area per call.", chip_size_x = chip_size.X, chip_size_y = chip_size.Y)

"""
    AutoRIFT.correlate_tiled(raw, grid, p, block_size) -> MultichipResult

[`AutoRIFT.correlate_multichip`](@ref)'s result, computed in blocks spanning at most `block_size`
pixels so that peak memory tracks the block rather than the scene.

`raw` is the **unfiltered** pair, and that is what makes this bound memory rather than merely
reorganize it: each block is filtered from its own read window, so the filtered scene the untiled
path holds resident is never formed. Handing this an already-filtered pair filters it twice.

Bit-identical to the untiled path. Every block is handed the whole grid's
[`AutoRIFT.PassGeometry`](@ref), so it runs the transform the untiled pass would have run; every
decision that looks at more than one point is taken once, on the assembled grid, rather than per
block; and a block's filtered values and eroded mask agree with a whole-scene filter everywhere the
block writes, which is what the filter term in [`AutoRIFT.halo`](@ref) buys.
"""
function correlate_tiled(raw::ImagePair, grid::PointSet{2}, p::Params, block_size)
    layout = block_layout(grid, p, size(raw), block_size)
    # One set for the whole run. `block_buffers` sizes from the layout's largest read window, which
    # no level changes, so allocating per pass would allocate the same nine arrays twice per level.
    # A threaded run takes its own set per task instead, since blocks then write concurrently.
    buffers = istrue(p.threaded) ? nothing : block_buffers(raw, layout)
    # Budget 0: this correlates the pair it is handed, reading windows from wherever that pair lives.
    # Deciding to read it into memory belongs to the entry points, where the caller can say otherwise.
    return _multichip(Blocked(raw, layout, layout.blocks, buffers, 0), grid, p)
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
function _run_blocked(raw::ImagePair, pts::PointSet{2}, p::Params, layout::BlockLayout,
                      blocks::Vector{Block}, geometry::PassGeometry,
                      measure::SimilarityMeasure, buffers::Union{Nothing,BlockBuffers},
                      subpixel::SubpixelMethod)
    out = displacement_field(pts)
    # Every block shares the geometry, so one warm-up serves all of them — and it must happen here,
    # on this task, rather than inside a block. Warmed from the whole point set rather than block by
    # block, because a block's own points reach a subset of these buckets: warming per block would
    # plan the same sizes repeatedly and, worse, plan them concurrently.
    _warm_pass_plans(geometry.chip, geometry.radius, pts, measure)

    serial = _serial_params(p)
    if istrue(p.threaded)
        # As many tasks as there are threads, capped by the block count, each claiming the next
        # unclaimed block until they run out. Bounding the task count is what lets a later change
        # bound the buffer count too; claiming dynamically is what keeps the tasks busy, since
        # per-block cost varies by orders of magnitude — a block whose points a finer level already
        # resolved returns before any I/O, so a static split would leave tasks idle.
        #
        # Dynamic claiming cannot change the result: each block writes a disjoint slice of `out`, so
        # the field is assembled rather than reduced and block order is not an input.
        #
        # One buffer set per *task*, reused across every block that task claims — so a run holds
        # `min(nblocks, nthreads)` sets rather than one per block. At 144 blocks on 8 threads that is
        # 55 MiB against 992.
        #
        # The set is allocated inside `_run_task_blocks!` rather than here, and that is load-bearing
        # rather than tidiness. A variable assigned inside a closure *and* in an enclosing scope is
        # hoisted into a single `Core.Box` shared by every closure built from that frame — so writing
        # `buf = block_buffers(...)` inline here, next to the serial branch's own `buf`, gives all
        # tasks one box and therefore one buffer set. That corrupts 300-900 of 3721 points, varying
        # run to run and concentrated in the blocks claimed second or later. A separate function has
        # its own frame, so the local cannot be captured or shared.
        next = Threads.Atomic{Int}(1)
        ntasks = min(length(blocks), Threads.nthreads())
        tasks = map(1:ntasks) do _
            StableTasks.@spawn _run_task_blocks!(out, next, raw, pts, serial, blocks, layout,
                                                 geometry, measure, subpixel)
        end
        foreach(wait, tasks)
    else
        # Serial: one set serves every block, and the caller's serves every pass.
        serialbuf = isnothing(buffers) ? block_buffers(raw, layout) : buffers
        for b in blocks
            _run_one_block!(out, serialbuf, raw, pts, serial, b, layout.halo, geometry, measure,
                            subpixel)
        end
    end
    return out
end

# One task's share of the blocks: take a buffer set, then claim blocks until they run out.
#
# A function rather than a `begin` block inside the spawn, so its buffer set is a genuine local of
# this frame. Julia boxes a captured variable that is also assigned in the enclosing scope, and a box
# built once in the caller is shared by every task spawned from it — one buffer set for all of them,
# and a corrupted result.
#
# Claiming from a shared counter rather than taking a pre-assigned slice, because per-block cost
# varies by orders of magnitude: a block whose points a finer level already resolved returns before
# any I/O, so a static split would leave tasks idle. It cannot change the result — each block writes a
# disjoint slice of `out`, so the field is assembled rather than reduced and block order is not an
# input.
function _run_task_blocks!(out::DisplacementField, next::Threads.Atomic{Int}, raw::ImagePair,
                           pts::PointSet{2}, p::Params, blocks::Vector{Block},
                           layout::BlockLayout, geometry::PassGeometry,
                           measure::SimilarityMeasure, subpixel::SubpixelMethod)
    buf = block_buffers(raw, layout)
    while true
        k = Threads.atomic_add!(next, 1)
        k <= length(blocks) || break
        _run_one_block!(out, buf, raw, pts, p, blocks[k], layout.halo, geometry, measure, subpixel)
    end
    return out
end

# A block is correlated one read window at a time, each window covering one tile of its points'
# coordinate span. Each sub-pass writes only the points that window holds, so every point is computed
# exactly once and assembly remains a copy.
#
# Usually there is one window and this is one pass over the block, byte-for-byte what it was before
# windows were derived per pass.
function _run_one_block!(out::DisplacementField, buf::BlockBuffers, raw::ImagePair,
                         pts::PointSet{2}, p::Params, b::Block, halo::Extent,
                         geometry::PassGeometry, measure::SimilarityMeasure,
                         subpixel::SubpixelMethod)
    windows, assign = _read_windows(pts, b, halo, size(raw),
                                    size(buf.reference, 1), size(buf.reference, 2))
    if isnothing(assign)
        isempty(windows) && return out
        w = only(windows)
        return _run_block_window!(out, buf, raw, pts, p, b, w.rows, w.cols, nothing, 0,
                                  geometry, measure, subpixel)
    end
    for (k, w) in enumerate(windows)
        _run_block_window!(out, buf, raw, pts, p, b, w.rows, w.cols, assign, k,
                           geometry, measure, subpixel)
    end
    return out
end

# One block, one read window. `assign` names which window owns each point, or `nothing` when one window
# owns all of them — which is the same computation the untiled path performs and must stay byte-identical
# to it.
function _run_block_window!(out::DisplacementField, buf::BlockBuffers, raw::ImagePair,
                            pts::PointSet{2}, p::Params, b::Block, rows, cols,
                            assign::Union{Nothing,AbstractMatrix{Int16}}, k::Int,
                            geometry::PassGeometry, measure::SimilarityMeasure,
                            subpixel::SubpixelMethod)
    bpts = _block_points(pts, b, rows, cols)
    # A block all of whose points a previous level resolved, or which the coarse mask emptied.
    # Checked before reading, so an empty block costs no I/O at all.
    #
    # Zeroing the other windows' points rather than only skipping them at assembly: such a point's
    # coordinate belongs to a *different* window, so searching it here would read the wrong imagery.
    # `_block_points` re-slices the grid for each sub-pass, so mutating these radii cannot reach the
    # caller's own.
    n = isnothing(assign) ? nsearchable(bpts) : _keep_window!(bpts, assign, k)
    n == 0 && return out
    # Before any I/O: a window this pass reaches past changes this block's answer, and saying so after
    # reading would only be slower.
    short = _block_window_shortfall(rows, cols, bpts, size(raw))
    short == 0 || _warn_block_window(rows, cols, bpts, short)
    bpair = _prepared_block_pair(buf, raw, rows, cols, p)
    bout = displacement_field(bpts)
    track!(bout, bpair, bpts, p; subpixel, measure, geometry)
    # Assembly is a copy: the halo grew what this block read, never what it writes.
    if isnothing(assign)
        out.dx[b.grid_rows, b.grid_cols] .= bout.dx
        out.dy[b.grid_rows, b.grid_cols] .= bout.dy
        out.correlation[b.grid_rows, b.grid_cols] .= bout.correlation
        out.peak_ratio[b.grid_rows, b.grid_cols] .= bout.peak_ratio
        out.searched[b.grid_rows, b.grid_cols] .= bout.searched
        return out
    end
    return _merge_window!(out, bout, b, assign, k)
end

# Restrict a block's points to one window, returning how many are left searchable.
#
# Both in one pass rather than zeroing and then counting: the count decides whether any imagery is read
# at all, and a window whose points a finer level resolved is the common case.
function _keep_window!(pts::PointSet, assign::AbstractMatrix{Int16}, k::Int)
    n = 0
    for i in eachindex(pts.radius_x, assign)
        if assign[i] == k
            issearchable(pts, i) && (n += 1)
        else
            pts.radius_x[i] = 0
            pts.radius_y[i] = 0
        end
    end
    return n
end

# Write one window's results into the assembled field, leaving the other windows' points untouched for
# the sub-passes that own them.
function _merge_window!(out::DisplacementField, bout::DisplacementField, b::Block,
                        assign::AbstractMatrix{Int16}, k::Int)
    odx = view(out.dx, b.grid_rows, b.grid_cols)
    ody = view(out.dy, b.grid_rows, b.grid_cols)
    ocor = view(out.correlation, b.grid_rows, b.grid_cols)
    opr = view(out.peak_ratio, b.grid_rows, b.grid_cols)
    osr = view(out.searched, b.grid_rows, b.grid_cols)
    for i in eachindex(assign, bout.dx, odx)
        assign[i] == k || continue
        odx[i] = bout.dx[i]
        ody[i] = bout.dy[i]
        ocor[i] = bout.correlation[i]
        opr[i] = bout.peak_ratio[i]
        osr[i] = bout.searched[i]
    end
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

# Blocks over a *strided subset* of the grid the blocks currently index, derived from them.
#
# Deriving rather than laying out afresh is what guarantees the two partitions agree: every selected
# point belongs to exactly one block, and to the same block its neighbours do.
#
# `blocks` is the partition to map, **not** `layout.blocks`, and that is what makes the operation
# composable. A level may be decimated before its coarse pass strides it again, so this runs twice in
# sequence; taking the full layout each time would silently discard the first striding and map the
# second against the wrong index space. Both stridings are relative to the grid handed to the pass,
# so each must start from the previous result.
#
# `rows`/`cols` are the indices selected *from that same grid*, so `searchsortedfirst` finds where
# each block's range begins in them. The read windows are pixel ranges and carry over unchanged:
# striding the grid changes which points a block writes, never which imagery it must read.
function _coarse_block_layout(blocks::Vector{Block}, setup)
    rows, cols = setup.rows, setup.cols
    out = Block[]
    for b in blocks
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
