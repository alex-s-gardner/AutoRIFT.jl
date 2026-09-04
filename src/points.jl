# Search points: where to correlate, and with what geometry.
#
# The correlator is fundamentally a *point* operation. Each search center is
# independent: it has its own coordinates, its own search radii, its own
# a-priori displacement, and its own chip size, and nothing about correlating at
# one center depends on any other. The reference obscures this by storing every
# field as a 2-D array and looping `for jj, for ii`, but its own inner loop only
# ever reads scalars (`xGrid[ii, jj]`), and its parallel worker flattens to
# linear indices with `unravel_index`. The grid is a *layout*, not a constraint.
#
# So `PointSet` is parametrized on dimensionality:
#
#   PointSet{1}  scattered centers, at arbitrary real coordinates
#   PointSet{2}  centers laid out on a grid
#
# Both are consumed by the same correlation kernel, which iterates `eachindex`
# and never asks about shape. Only the multi-chip-size search needs `N == 2`, since
# its filtering and resampling steps are neighbourhood operations that require a
# spatial layout; those methods dispatch on `PointSet{2}` and are unavailable for
# scattered points, which is the correct constraint rather than a limitation.
#
# Because a `Matrix` and its `vec` share memory, moving between the two views is
# free: no copy, no conversion.

"""
    AutoRIFT.Uniform(value, size) <: AbstractArray

One value, indexable at every position of a `size`-shaped array, stored once.

A [`PointSet`](@ref) field the caller gave as a scalar holds this instead of a materialized array.
The saving is the whole point: on a 2107x2127 grid a dense `Int` field is 34 MiB, and four such fields
are constant on every grid built from [`params`](@ref) — the two chip extents and the two chip-size
bounds. `Uniform` stores 24 bytes for each.

Read-only, which is why only those four fields use it. `radius_x`/`radius_y` are written per point by
the coarse pass (see `AutoRIFT._apply_coarse_mask!`), so they stay dense; a caller supplying any field
as an array gets that array unchanged.

`IndexLinear`, so `eachindex` and `vec` behave as the correlator expects, and the constant read is
hoisted out of an inner loop rather than fetched per point.
"""
struct Uniform{T,N} <: AbstractArray{T,N}
    value::T
    size::NTuple{N,Int}
end

Base.size(u::Uniform) = u.size
Base.axes(u::Uniform) = map(Base.OneTo, u.size)
Base.IndexStyle(::Type{<:Uniform}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(u::Uniform, ::Int) = u.value
Base.@propagate_inbounds Base.getindex(u::Uniform{T,N}, ::Vararg{Int,N}) where {T,N} = u.value
# `vec` and `similar` on a `Uniform` must not silently materialize: `scatter` takes `vec` of every
# field, and a dense result there would undo the saving at the point the correlator is called.
Base.vec(u::Uniform) = Uniform(u.value, (length(u),))
Base.similar(u::Uniform, ::Type{T}, dims::Dims) where {T} = Array{T}(undef, dims)

# Every field of a `PointSet` must share `x`'s axes. Checked by recursion over a
# `NamedTuple` rather than a loop over a tuple of (name, field) pairs: the fields have ten
# distinct types, so the loop variable is a `Union` and iterating it allocates. This form is
# unrolled at compile time, keeping construction free of the heap and O(1) in point count —
# `scatter` runs it once per chip-size level.
_check_axes(ax, ::@NamedTuple{}) = nothing
@inline function _check_axes(ax, fields::NamedTuple)
    name, field = first(keys(fields)), first(values(fields))
    axes(field) == ax || throw(DimensionMismatch(
        "PointSet field `$name` has axes $(axes(field)), expected $ax to match `x`"))
    return _check_axes(ax, Base.structdiff(fields, NamedTuple{(name,)}))
end

"""
    PointSet{N}

Where to correlate, and with what geometry. Fields are `N`-dimensional arrays of
matching shape, one entry per search center.

`PointSet{1}` holds scattered centers at arbitrary coordinates; `PointSet{2}`
holds centers on a grid. Correlation treats the two identically — each center is
independent — while the multi-chip-size search requires `PointSet{2}` because its
filtering steps are neighbourhood operations.

# Fields
- `x`, `y`: center coordinates in image pixels, as `Float64`. Fractional
  coordinates are meaningful: a center at `x = 10.5` sits between columns.
- `radius_x`, `radius_y`: search half-extent per axis, in pixels. The search
  window spans `2 * radius`. Independent per axis and per point — a glacier
  flowing along x warrants a wide `radius_x` and a narrow `radius_y`. A point
  with either radius zero is skipped.
- `dx_prior`, `dy_prior`: a-priori displacement in pixels. The search window is
  centred on the prior rather than on zero, so a modest radius suffices even
  where motion is large.
- `chip_size_x`, `chip_size_y`: chip extent in pixels, per point.
- `chip_size_min_x`, `chip_size_max_x`: the range of chip sizes a point may be searched at, in
  pixels, bounding the multi-chip-size levels rather than any single pass. A level runs at a point
  only when its chip size falls within them, so a point over thin, fast ice can be restricted to
  fine chips while smooth ice admits coarse ones. Default to zero, which means unbounded — every
  level runs everywhere, and a caller who supplies neither sees no change.

Every field carries its own type parameter, because a field is an [`AutoRIFT.Uniform`](@ref) exactly
when the caller gave *that* field as a scalar, and callers mix freely: `chip_size_x` as a per-point
array beside a scalar `chip_size_y` is ordinary use. One parameter shared across two fields would make
such a pair a `MethodError`.

See also [`pointset`](@ref), [`gridpoints`](@ref), [`npoints`](@ref).
"""
struct PointSet{N,X<:AbstractArray{Float64,N},Y<:AbstractArray{Float64,N},
                RX<:AbstractArray{Int,N},RY<:AbstractArray{Int,N},
                DX<:AbstractArray{Float64,N},DY<:AbstractArray{Float64,N},
                CX<:AbstractArray{Int,N},CY<:AbstractArray{Int,N},
                CN<:AbstractArray{Int,N},CM<:AbstractArray{Int,N}}
    x::X
    y::Y
    radius_x::RX
    radius_y::RY
    dx_prior::DX
    dy_prior::DY
    chip_size_x::CX
    chip_size_y::CY
    chip_size_min_x::CN
    chip_size_max_x::CM

    function PointSet(x::X, y::Y, radius_x::RX, radius_y::RY, dx_prior::DX,
                      dy_prior::DY, chip_size_x::CX, chip_size_y::CY,
                      chip_size_min_x::CN = Uniform(0, size(chip_size_x)),
                      chip_size_max_x::CM = Uniform(0, size(chip_size_x))) where {
                          N,X<:AbstractArray{Float64,N},Y<:AbstractArray{Float64,N},
                          RX<:AbstractArray{Int,N},RY<:AbstractArray{Int,N},
                          DX<:AbstractArray{Float64,N},DY<:AbstractArray{Float64,N},
                          CX<:AbstractArray{Int,N},CY<:AbstractArray{Int,N},
                          CN<:AbstractArray{Int,N},CM<:AbstractArray{Int,N}}
        ax = axes(x)
        _check_axes(ax, (y = y, radius_x = radius_x, radius_y = radius_y,
                         dx_prior = dx_prior, dy_prior = dy_prior,
                         chip_size_x = chip_size_x, chip_size_y = chip_size_y,
                         chip_size_min_x = chip_size_min_x,
                         chip_size_max_x = chip_size_max_x))
        return new{N,X,Y,RX,RY,DX,DY,CX,CY,CN,CM}(
            x, y, radius_x, radius_y, dx_prior, dy_prior,
            chip_size_x, chip_size_y, chip_size_min_x, chip_size_max_x)
    end
end

"""
    npoints(pts::PointSet) -> Int

Number of search centers, including those that will be skipped for having a zero
search radius. See [`nsearchable`](@ref) for the count that will actually be
correlated.
"""
npoints(pts::PointSet) = length(pts.x)

Base.length(pts::PointSet) = npoints(pts)
Base.size(pts::PointSet) = size(pts.x)
Base.eachindex(pts::PointSet) = eachindex(pts.x)
Base.ndims(::PointSet{N}) where {N} = N
Base.isempty(pts::PointSet) = isempty(pts.x)

"""
    nsearchable(pts::PointSet) -> Int

Number of points that will actually be correlated, i.e. those with a non-zero
search radius in both axes.

Typically far smaller than [`npoints`](@ref): the chip-size loop's coarse pass zeroes
the radius wherever it found no coherent motion, and a skipped point costs
almost nothing while a searched one costs microseconds. This ratio is what makes
dynamic work-scheduling worthwhile.
"""
function nsearchable(pts::PointSet)
    n = 0
    @inbounds for i in eachindex(pts)
        issearchable(pts, i) && (n += 1)
    end
    return n
end

"""
    issearchable(pts::PointSet, i) -> Bool

Whether point `i` has a search window with extent in both axes. A window with no
extent in one direction cannot yield a displacement.
"""
@inline issearchable(pts::PointSet, i) =
    @inbounds pts.radius_x[i] > 0 && pts.radius_y[i] > 0

"""
    pointset(x, y; kwargs...) -> PointSet

Build a [`PointSet`](@ref) from center coordinates, filling unspecified geometry
from scalar defaults.

`x` and `y` may be vectors of scattered coordinates, matrices of gridded ones,
or a vector of `Tuple`/`CartesianIndex`. Each geometry keyword accepts a scalar,
broadcast to every point, or an array matching the shape of `x`.

```jldoctest
julia> pts = AutoRIFT.pointset([10.0, 50.5, 200.0], [20.0, 60.0, 30.25];
                               search_radius = 25, chip_size = 32);

julia> AutoRIFT.npoints(pts)
3

julia> pts.x[2], pts.radius_x[2], pts.chip_size_x[2]
(50.5, 25, 32)
```

# Keywords
- `search_radius = 25`: half-extent of the search window, as an
  [`AutoRIFT.Extent`](@ref) — a scalar means square, `(X = …, Y = …)` names the axes.
- `chip_size = 32`: chip extent, likewise.
- `search_radius_x`, `search_radius_y`, `chip_size_x`, `chip_size_y`: per-axis overrides that also
  accept a **per-point array**, which an extent cannot express. That is what they are for: a
  Geogrid-derived search radius varies across the scene.
- `dx_prior = 0.0`, `dy_prior = 0.0`: a-priori displacement.
- `chip_size_min_x = 0`, `chip_size_max_x = 0`: the chip sizes this point may be searched at,
  bounding the multi-chip-size levels. Zero means unbounded. Also accept a per-point array, which
  is how they are usually supplied — ITS_LIVE ships them as rasters.
"""
function pointset(
    x::AbstractArray, y::AbstractArray;
    search_radius = 25,
    search_radius_x = nokw,
    search_radius_y = nokw,
    chip_size = 32,
    chip_size_x = nokw,
    chip_size_y = nokw,
    dx_prior = 0.0,
    dy_prior = 0.0,
    chip_size_min_x = 0,
    chip_size_max_x = 0,
)
    size(x) == size(y) || throw(DimensionMismatch(
        "`x` and `y` must have the same shape, got $(size(x)) and $(size(y))"))

    xs = _tofield(Float64, x, x, :x)
    ys = _tofield(Float64, y, x, :y)

    # `search_radius` and `chip_size` are extents, so they set both axes at once and a scalar means
    # square. The per-axis keywords stay, and are not redundant: they accept a *per-point array*,
    # which an extent cannot express — a Geogrid-derived radius varies across the scene.
    rad = extent(search_radius)
    rx = _tofield(Int, isnokw(search_radius_x) ? rad.X : search_radius_x, x, :search_radius_x)
    ry = _tofield(Int, isnokw(search_radius_y) ? rad.Y : search_radius_y, x, :search_radius_y)

    # The chip extents and the chip-size bounds are read-only, so a scalar stays one value. On a
    # Landsat-sized grid that is four 34 MiB arrays replaced by 96 bytes.
    cs = extent(chip_size)
    csx = _toconstfield(Int, isnokw(chip_size_x) ? cs.X : chip_size_x, x, :chip_size_x)
    csy = _toconstfield(Int, isnokw(chip_size_y) ? cs.Y : chip_size_y, x, :chip_size_y)

    dx = _tofield(Float64, dx_prior, x, :dx_prior)
    dy = _tofield(Float64, dy_prior, x, :dy_prior)

    # Zero means unbounded in both, so the default admits every level and a caller who sets
    # neither keyword gets the behaviour they had before these fields existed.
    cmin = _toconstfield(Int, chip_size_min_x, x, :chip_size_min_x)
    cmax = _toconstfield(Int, chip_size_max_x, x, :chip_size_max_x)

    return PointSet(xs, ys, rx, ry, dx, dy, csx, csy, cmin, cmax)
end

# Coordinates given as points rather than as parallel arrays.
pointset(xy::AbstractVector{<:Tuple{Real,Real}}; kw...) =
    pointset([Float64(p[1]) for p in xy], [Float64(p[2]) for p in xy]; kw...)

# A CartesianIndex is (row, col) = (y, x); silently treating it as (x, y) would
# transpose the whole point set, so be explicit about the swap.
pointset(idx::AbstractVector{<:CartesianIndex{2}}; kw...) =
    pointset([Float64(I[2]) for I in idx], [Float64(I[1]) for I in idx]; kw...)

# Broadcast a scalar, or validate and convert an array, to a field matching the
# shape of the coordinates.
_tofield(::Type{T}, v::Real, like::AbstractArray, ::Symbol) where {T} =
    fill(_exactly(T, v), size(like))

# A scalar for a field nothing writes, kept as one value rather than materialized. Separate from
# `_tofield` rather than a keyword on it, because which fields may take it is a property of the
# *field*, not of the call: see [`AutoRIFT.Uniform`](@ref) for why `radius_x` cannot.
_toconstfield(::Type{T}, v::Real, like::AbstractArray, ::Symbol) where {T} =
    Uniform(_exactly(T, v), size(like))
_toconstfield(::Type{T}, v, like::AbstractArray, name) where {T} = _tofield(T, v, like, name)
function _tofield(::Type{T}, v::AbstractArray, like::AbstractArray, name) where {T}
    size(v) == size(like) || throw(DimensionMismatch(
        "`$name` has shape $(size(v)) but there are $(size(like)) points"))
    return map(x -> _exactly(T, x), v)
end
_tofield(::Type{T}, v, ::AbstractArray, name) where {T} = throw(ArgumentError(
    "`$name` must be a number or an array of them, got a $(typeof(v))"))

# Integer-valued fields must not silently absorb a fractional value: a chip size
# of 32.5 or a radius of 6.7 is a mistake in the caller, not a rounding request.
_exactly(::Type{T}, v::Real) where {T<:Integer} = isinteger(v) ? T(v) :
    throw(ArgumentError("expected an integer value, got $v"))
_exactly(::Type{T}, v::Real) where {T<:AbstractFloat} = T(v)

"""
    gridpoints(imagesize, spacing; kwargs...) -> PointSet{2}
    gridpoints(xs, ys; kwargs...) -> PointSet{2}

Build a gridded [`PointSet`](@ref): centers on a regular lattice, laid out in 2-D
so that the multi-chip-size search can apply its neighbourhood filters.

The first form places centers every `spacing` pixels across an image of size
`imagesize = (nrows, ncols)`, inset far enough from the edges that a chip and its
search window fit. The second takes explicit coordinate vectors.

```jldoctest
julia> pts = AutoRIFT.gridpoints((512, 512), 32; chip_size = 32, search_radius = 25);

julia> ndims(pts), size(pts)
(2, (13, 13))
```

Accepts the same geometry keywords as [`pointset`](@ref).
"""
function gridpoints(imagesize::Tuple{Integer,Integer}, spacing;
                    chip_size = 32, search_radius = 25,
                    search_radius_x = nokw, search_radius_y = nokw, kw...)
    sp = extent(spacing)
    (sp.X > 0 && sp.Y > 0) || throw(ArgumentError(
        "`spacing` must be positive, got $(sp.X) by $(sp.Y)"))

    # Inset so that the widest chip plus its search window lies inside the
    # image. Points that would read outside would either need padding or produce
    # a truncated correlation surface; excluding them keeps the geometry exact.
    #
    # `maximum` on each, because a per-point array is allowed here: the inset has to clear the
    # widest chip and radius anywhere on the grid, not the first one.
    rad = extent(search_radius)
    rx = isnokw(search_radius_x) ? rad.X : search_radius_x
    ry = isnokw(search_radius_y) ? rad.Y : search_radius_y
    cs = extent(chip_size)
    margin_x = cld(cs.X, 2) + maximum(rx) + 1
    margin_y = cld(cs.Y, 2) + maximum(ry) + 1

    nrows, ncols = imagesize
    xs = (margin_x + 1):sp.X:(ncols - margin_x)
    ys = (margin_y + 1):sp.Y:(nrows - margin_y)
    (isempty(xs) || isempty(ys)) && throw(ArgumentError(
        "no search center fits in a $(nrows)x$(ncols) image with chip " *
        "$(cs.X)x$(cs.Y) and search radius $(maximum(rx))x$(maximum(ry)): a margin of " *
        "$(margin_y)x$(margin_x) pixels is needed on each side. Use a smaller " *
        "chip size or search radius."))

    return gridpoints(xs, ys; chip_size, search_radius,
                      search_radius_x, search_radius_y, kw...)
end

function gridpoints(xs::AbstractVector, ys::AbstractVector; kw...)
    # Row index varies with y, column index with x, matching image layout.
    X = [Float64(x) for _ in ys, x in xs]
    Y = [Float64(y) for y in ys, _ in xs]
    return pointset(X, Y; kw...)
end

"""
    scatter(pts::PointSet) -> PointSet{1}

View a `PointSet` as a flat list of points, discarding any grid layout.

Free: `vec` of an array shares its memory, so this allocates nothing and mutating
either view is visible in the other. Used to hand a gridded point set to the
correlator, which has no use for the layout.
"""
scatter(pts::PointSet{1}) = pts
scatter(pts::PointSet) = rebuild(pts;
    x = vec(pts.x), y = vec(pts.y),
    radius_x = vec(pts.radius_x), radius_y = vec(pts.radius_y),
    dx_prior = vec(pts.dx_prior), dy_prior = vec(pts.dy_prior),
    chip_size_x = vec(pts.chip_size_x), chip_size_y = vec(pts.chip_size_y),
    chip_size_min_x = vec(pts.chip_size_min_x),
    chip_size_max_x = vec(pts.chip_size_max_x))

"""
    pts[rows, cols] -> PointSet{2}

Decimate or crop a gridded point set, copying every field.

The coarse pass needs a strided subset of its grid, and doing that by indexing
every field at the call site both repeats the shape and makes it easy to miss one when a
field is added. A copy rather than a view because the caller then modifies the result — the
coarse pass overwrites the radii and the chip sizes.
"""
Base.getindex(pts::PointSet{2}, rows, cols) = PointSet(
    pts.x[rows, cols], pts.y[rows, cols],
    pts.radius_x[rows, cols], pts.radius_y[rows, cols],
    pts.dx_prior[rows, cols], pts.dy_prior[rows, cols],
    pts.chip_size_x[rows, cols], pts.chip_size_y[rows, cols],
    pts.chip_size_min_x[rows, cols], pts.chip_size_max_x[rows, cols])

"""
    rebuild(pts::PointSet; kwargs...) -> PointSet

A copy of `pts` with the named fields replaced, sharing the rest.

Used wherever a point set is derived from another by changing one or two fields — shifting the
coordinates, overriding the chip size, zeroing a radius. Doing that by listing every
positional argument means adding a field to `PointSet` silently drops it from every such site,
which is the failure this exists to prevent.
"""
rebuild(pts::PointSet; x = pts.x, y = pts.y,
        radius_x = pts.radius_x, radius_y = pts.radius_y,
        dx_prior = pts.dx_prior, dy_prior = pts.dy_prior,
        chip_size_x = pts.chip_size_x, chip_size_y = pts.chip_size_y,
        chip_size_min_x = pts.chip_size_min_x,
        chip_size_max_x = pts.chip_size_max_x) =
    PointSet(x, y, radius_x, radius_y, dx_prior, dy_prior, chip_size_x, chip_size_y,
             chip_size_min_x, chip_size_max_x)

"""
    sanitize!(pts::PointSet, min_radius) -> Int

Normalise search radii in place and return the number of points left searchable.

Two rules, both inherited from the reference (`autoRIFT.py:547-551`) and both
deliberate:

1. **Zero is contagious across axes.** A point whose radius is non-positive in
   either axis is unsearchable, so both radii are cleared. A search window with
   no extent in one direction cannot produce a displacement, and leaving a
   positive radius in the other axis would imply otherwise.
2. **The floor applies per axis, and only to searchable points.** A point that is
   searched at all gets at least `min_radius` in each direction, because a
   correlation surface a few pixels across cannot support subpixel refinement.
   Points cleared by rule 1 stay cleared rather than being raised to the floor.
"""
function sanitize!(pts::PointSet, min_radius::Integer)
    rx, ry = pts.radius_x, pts.radius_y
    n = 0
    @inbounds for i in eachindex(rx)
        if rx[i] <= 0 || ry[i] <= 0
            rx[i] = 0
            ry[i] = 0
        else
            rx[i] = max(rx[i], min_radius)
            ry[i] = max(ry[i], min_radius)
            n += 1
        end
    end
    return n
end

"""
    chip_bounds(pts::PointSet, i) -> (rows, cols)

Pixel ranges of the chip cut from the secondary image at point `i`.

The chip is centred on the point *displaced by its a-priori offset*, so the
search begins where motion is expected rather than at zero. Extent is
`chip_size` in each direction, giving an even-sized chip with a well-defined
half-extent.
"""
@inline function chip_bounds(pts::PointSet, i)
    @inbounds begin
        hx = pts.chip_size_x[i] ÷ 2
        hy = pts.chip_size_y[i] ÷ 2
        cx = floor(Int, pts.x[i] - pts.dx_prior[i])
        cy = floor(Int, pts.y[i] - pts.dy_prior[i])
    end
    return (cy - hy):(cy + hy - 1), (cx - hx):(cx + hx - 1)
end

"""
    search_bounds(pts::PointSet, i) -> (rows, cols)

Pixel ranges of the search window cut from the reference image at point `i`.

Larger than the chip by the search radius in each direction, so that correlating
the two yields a surface of exactly `(2 * radius_y, 2 * radius_x)` samples with
zero displacement at its centre.

The window is deliberately **asymmetric**: it extends `radius` pixels below the
chip but only `radius - 1` above, giving a width of `chip + 2 * radius - 1`
rather than `chip + 2 * radius`. That is what makes the correlation surface an
even `2 * radius` samples across, so that zero displacement lands exactly on a
sample rather than between two. A symmetric window would give an odd surface
whose centre is a half-sample offset from zero displacement, biasing every
recovered velocity by half a pixel.
"""
@inline function search_bounds(pts::PointSet, i)
    @inbounds begin
        hx = pts.chip_size_x[i] ÷ 2
        hy = pts.chip_size_y[i] ÷ 2
        rx = pts.radius_x[i]
        ry = pts.radius_y[i]
        cx = floor(Int, pts.x[i])
        cy = floor(Int, pts.y[i])
    end
    return (cy - hy - ry):(cy + hy + ry - 2), (cx - hx - rx):(cx + hx + rx - 2)
end

"""
    surface_size(pts::PointSet, i) -> (nrows, ncols)

Size of the correlation surface at point `i`: `(2 * radius_y, 2 * radius_x)`.

Follows from the chip and search-window extents; asserted in the tests so that a
change to either stays consistent with the peak-offset arithmetic, which assumes
zero displacement sits at index `(radius_y, radius_x)` counting from zero.
"""
@inline function surface_size(pts::PointSet, i)
    @inbounds return (2 * pts.radius_y[i], 2 * pts.radius_x[i])
end

"""
    inbounds(pts::PointSet, i, imagesize) -> Bool

Whether the search window at point `i` lies entirely inside an image of size
`imagesize = (nrows, ncols)`.

Points that fail are either skipped or handled by padding the image, depending on
the caller. Provided so that scattered point sets — where the caller chose the
coordinates and may not have left room — can be validated up front rather than
faulting deep in the correlator.
"""
@inline function inbounds(pts::PointSet, i, imagesize::Tuple{Integer,Integer})
    rows, cols = search_bounds(pts, i)
    return first(rows) >= 1 && last(rows) <= imagesize[1] &&
           first(cols) >= 1 && last(cols) <= imagesize[2]
end

function Base.show(io::IO, pts::PointSet{N}) where {N}
    ns = nsearchable(pts)
    print(io, "PointSet{", N, "}(", npoints(pts), " points")
    N == 2 && print(io, ", ", size(pts, 1), "x", size(pts, 2), " grid")
    ns == npoints(pts) || print(io, ", ", ns, " searchable")
    print(io, ")")
end

Base.size(pts::PointSet, d::Integer) = size(pts.x, d)
# Needed for `end` to resolve inside `pts[rows, cols]`: without it, indexing with `end` is a
# MethodError rather than the obvious thing.
Base.axes(pts::PointSet) = axes(pts.x)
Base.axes(pts::PointSet, d::Integer) = axes(pts.x, d)
