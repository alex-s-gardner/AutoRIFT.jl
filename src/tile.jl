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
filter's reach. Both apply to every point, so both are taken at their maximum over the run.

Measured against the *coarsest level*, not against `grid` as supplied. A level overwrites the
chip size and floors the radii, so a grid's own `chip_size_x` and `radius_x` are not what any
pass runs — see [`AutoRIFT._worst_level_points`](@ref).

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
    _, _, _, _, pad, _ = _pass_geometry(_worst_level_points(grid, p), imagesize)
    return (pad[1] + w, pad[2] + w)
end

"""
    AutoRIFT._worst_level_points(grid::PointSet, p::Params) -> PointSet{1}

`grid`'s points carrying the largest chip size and radius any level will run them at.

The halo has to cover every level, and a level's geometry is not the grid's. `chipsize_level`
ignores `grid.chip_size_x` and sets its own from [`AutoRIFT.chip_sizes`](@ref), so the reach is
set by `chip_size_max` however the caller sized the grid; and `_level_points` ends in
[`AutoRIFT.sanitize!`](@ref), which floors every non-zero radius at `p.min_search_radius`, so a
grid whose radii sit below that floor is searched wider than it asks for.

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
    # Copies, because `sanitize!` writes in place and `scatter` shares the grid's own arrays.
    pts = rebuild(flat; radius_x = copy(flat.radius_x), radius_y = copy(flat.radius_y),
                  chip_size_x = fill(cs.X, n),
                  chip_size_y = fill(cs.Y, n))
    # The same floor a level applies, applied by the same function, so the two cannot drift.
    sanitize!(pts, p.min_search_radius)
    return pts
end

"""
    AutoRIFT.block_layout(grid::PointSet{2}, p::Params, imagesize, block_size) -> BlockLayout

Divide `grid` into blocks spanning at most `block_size = (X, Y)` **pixels** of output each, with the
read window each needs.

Pixels, because pixels are what blocking bounds. A block's output is negligible — 3.2 KiB against
the 6.7 MiB of imagery it reads at a 512-pixel block, and the whole scene's output grid is smaller
than one block's read window — so the quantity worth naming is the imagery held at once. It is also
the unit everything here already works in: the halo, the read windows, and the buffers are all
pixels.

A block must be a whole number of grid points, so the size is a target that **snaps outward**: at
`grid_spacing = 32` a request of 500 becomes 512. The conversion walks the grid's own coordinates
rather than dividing by `grid_spacing`, because a caller-supplied grid need not be uniformly spaced.
The trailing block in each direction is short when the grid does not divide evenly.

Throws if a block would be smaller than the halo it reads, since such a block is all overlap.
"""
function block_layout(grid::PointSet{2}, p::Params, imagesize::Tuple{Int,Int},
                      block_size::Tuple{Int,Int})
    px, py = block_size
    (px > 0 && py > 0) || throw(ArgumentError(
        "`process_block_size` must be positive in both axes, got $px by $py pixels"))

    hx, hy = halo(grid, p, imagesize)
    # A block narrower than its own halo reads more overlap than data, so it is a configuration
    # error rather than something to silently widen. Compared directly, both being pixels.
    (px >= hx && py >= hy) || throw(ArgumentError(
        "`process_block_size` of $px by $py pixels is smaller than the $hx by $hy pixel halo each " *
        "block must read around it — so the block would be almost entirely overlap. Use at least " *
        "$hx by $hy pixels, or reduce the chip size, search radius, or filter width."))

    nr, nc = size(grid)
    nrows, ncols = imagesize
    # Where each block starts, walked from the coordinates so no block spans more than the request.
    # `grid.y` runs down rows and `grid.x` across columns, so the row starts take the y extent. The
    # views are free and make the axis explicit at the call rather than a parameter to be got wrong.
    rowstarts = _block_starts(view(grid.y, :, 1), py)
    colstarts = _block_starts(view(grid.x, 1, :), px)

    blocks = Block[]
    for (ci, c0) in pairs(colstarts), (ri, r0) in pairs(rowstarts)
        grows = r0:(ri == lastindex(rowstarts) ? nr : rowstarts[ri + 1] - 1)
        gcols = c0:(ci == lastindex(colstarts) ? nc : colstarts[ci + 1] - 1)
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

# The index each block starts at along one axis, so no block's coordinates span more than `want`
# pixels. `coord` is that axis's coordinates — a gridded `PointSet` repeats the same coordinate down
# every row and across every column, so one row or column describes the whole axis.
#
# Walked rather than divided, because a caller-supplied grid need not be uniformly spaced:
# `gridpoints(xs, ys)` takes arbitrary coordinate vectors, so no single `grid_spacing` describes the
# axis. Accumulating per block also means a grid that is uniform apart from one large jump keeps
# uniform-sized blocks either side of it, where a single points-per-block figure taken from the
# largest step would shrink every block to fit the worst one.
#
# Each block takes as many points as fit, and always at least one — a zero-point block would divide
# the grid into nothing. One point is therefore the only case that may exceed `want`, and it needs a
# gap wider than a whole block to arise.
function _block_starts(coord::AbstractVector, want::Int)
    starts = [firstindex(coord)]
    length(coord) <= 1 && return starts
    origin = first(coord)
    @inbounds for i in (firstindex(coord) + 1):lastindex(coord)
        # Start a new block once this point would carry the current one past `want`. The comparison
        # is on the span from the block's first point, so rounding cannot accumulate across blocks.
        if abs(coord[i] - origin) > want
            push!(starts, i)
            origin = coord[i]
        end
    end
    return starts
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
function _block_pair!(buf::BlockBuffers, pair::ImagePair, b::Block)
    nr, nc = length(b.read_rows), length(b.read_cols)
    r = @view buf.reference[1:nr, 1:nc]
    s = @view buf.secondary[1:nr, 1:nc]
    rv = @view buf.reference_valid[1:nr, 1:nc]
    sv = @view buf.secondary_valid[1:nr, 1:nc]
    _read_block!(r, pair.reference, b.read_rows, b.read_cols)
    _read_block!(s, pair.secondary, b.read_rows, b.read_cols)
    _read_block!(rv, pair.reference_valid, b.read_rows, b.read_cols)
    _read_block!(sv, pair.secondary_valid, b.read_rows, b.read_cols)
    return ImagePair(r, s, rv, sv)
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
    # `img[rows, cols]`, not `copyto!(dest, view(img, rows, cols))`. A view defers the read, so
    # `copyto!` then walks it element by element — and for a lazy array that is one read per pixel
    # rather than one read per window. Measured on a lazy GeoTIFF: 99.7% of a blocked run's time was a
    # scalar `getindex` reached from here, and a 512² region with one block took 444 s against 0.6 s
    # from memory. Indexing asks the array for the whole window, which is the operation a chunked
    # backend is built to serve — `DiskArrays` turns it into one aligned read per touched chunk.
    #
    # The extra allocation is a block-sized temporary, which is the trade: `BlockBuffers` exists to
    # keep a run's storage at one block's worth, and this adds one more of the same order while
    # removing a per-pixel I/O call. For an in-memory input the two forms cost the same.
    copyto!(dest, img[rows, cols])
    return dest
end

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
function _prepared_block_pair(buf::BlockBuffers, pair::ImagePair, b::Block, p::Params)
    raw = _block_pair!(buf, pair, b)
    return _prepare_block(buf, raw, p, p.preprocess, length(b.read_rows), length(b.read_cols))
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

# Every other filter: allocate, as the untiled path does. Bit-identical either way.
_prepare_block(::BlockBuffers, raw::ImagePair, p::Params, ::PreprocessMethod, ::Int, ::Int) =
    _prepare(raw, p)

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
    AutoRIFT.Blocked(raw, layout, blocks, buffers)

Correlate each pass a block at a time. See [`AutoRIFT.PassRunner`](@ref).

`raw` is the **unfiltered** pair, and that is what makes blocking bound memory rather than merely
reorganize it: each block is filtered from its own read window, so the filtered scene the whole-scene
runner holds is never formed.

`blocks` is the partition this runner correlates over — `layout.blocks` for a fine pass, and the
strided subset [`AutoRIFT._coarse_block_layout`](@ref) derives for a coarse one. `layout` is kept
alongside it because `block_buffers` sizes from the whole layout's largest read window, which no pass
changes. `buffers` is `nothing` for a threaded run, where each task takes its own set.

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
    Blocked(r.raw, r.layout, _coarse_block_layout(r.blocks, setup), r.buffers)

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
function correlate_tiled(raw::ImagePair, grid::PointSet{2}, p::Params,
                         block_size::Tuple{Int,Int})
    layout = block_layout(grid, p, size(raw), block_size)
    # One set for the whole run. `block_buffers` sizes from the layout's largest read window, which
    # no level changes, so allocating per pass would allocate the same nine arrays twice per level.
    # A threaded run takes its own set per task instead, since blocks then write concurrently.
    buffers = istrue(p.threaded) ? nothing : block_buffers(raw, layout)
    return _multichip(Blocked(raw, layout, layout.blocks, buffers), grid, p)
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
    # on this task, rather than inside a block.
    _warm_pass_plans(geometry.chip_x, geometry.chip_y, geometry.radius_x, geometry.radius_y,
                     measure)

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
            _run_one_block!(out, serialbuf, raw, pts, serial, b, geometry, measure, subpixel)
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
        _run_one_block!(out, buf, raw, pts, p, blocks[k], geometry, measure, subpixel)
    end
    return out
end

function _run_one_block!(out::DisplacementField, buf::BlockBuffers, raw::ImagePair,
                         pts::PointSet{2}, p::Params, b::Block, geometry::PassGeometry,
                         measure::SimilarityMeasure, subpixel::SubpixelMethod)
    bpts = _block_points(pts, b)
    # A block all of whose points a previous level resolved, or which the coarse mask emptied.
    # Checked before reading, so an empty block costs no I/O at all.
    nsearchable(bpts) == 0 && return out
    bpair = _prepared_block_pair(buf, raw, b, p)
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
