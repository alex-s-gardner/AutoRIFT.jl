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

A [`PointSet`](@ref) field given as a scalar holds this instead of a materialized array. On a
2107x2127 grid a dense `Int` field is 34 MiB; `Uniform` stores 24 bytes.

Read-only, so only the fields nothing writes use it: `chip_size_x`, `chip_size_y`, `chip_size_min_x`,
`chip_size_max_x` and `positioned`. `radius_x` and `radius_y` are written per point by the coarse pass
and stay dense. A field supplied as an array is kept unchanged. `vec` of a `Uniform` is a `Uniform`;
indexing it with ranges returns an ordinary `Array`.
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

# Every field of a `PointSet` must share `x`'s axes.
#
# `map` over the names, not a `for` over a tuple of (name, field) pairs: the fields have ten
# distinct types, so the loop variable is a `Union` and each iteration boxes. `map` over a tuple of
# `Val`s specializes per field, keeping construction off the heap and O(1) in point count —
# `scatter` runs it once per chip-size level.
#
# Not recursion over `Base.structdiff` either, which is equally unrolled and equally allocation-free
# but leaves `--trim=safe` with an unresolved self-invoke it cannot prove terminates.
@inline function _check_axes(ax, fields::NamedTuple{names}) where {names}
    map(map(Val, names)) do vname
        _check_axis(ax, vname, fields[_unval(vname)])
    end
    return nothing
end

_unval(::Val{N}) where {N} = N

@inline function _check_axis(ax, ::Val{name}, field) where {name}
    axes(field) == ax || throw(DimensionMismatch(
        "PointSet field `$name` has axes $(axes(field)), expected $ax to match `x`"))
    return nothing
end

"""
    PointSet{N}

Where to correlate, and with what geometry. Fields are `N`-dimensional arrays of matching shape, one
entry per search center. The constructor throws `DimensionMismatch` if the shapes differ.

`PointSet{1}` holds scattered centers at arbitrary coordinates; `PointSet{2}` holds centers on a grid.
Correlation treats the two identically, but the multi-chip-size search requires `PointSet{2}` because
its filtering steps use neighborhoods.

# Fields
- `x`, `y`: center coordinates in image pixels, `Float64`. Fractional coordinates are allowed: `x =
  10.5` sits between columns.
- `radius_x`, `radius_y`: search half-extent per axis, in pixels; the search window spans
  `2 * radius`. A point with either radius zero is skipped.
- `dx_prior`, `dy_prior`: a-priori displacement in pixels. The search window is centered on the prior
  rather than on zero.
- `chip_size_x`, `chip_size_y`: chip extent in pixels, per point. A gridded multi-chip-size search
  overwrites these at each level; they take effect on a scattered `PointSet{1}`.
- `chip_size_min_x`, `chip_size_max_x`: the range of chip sizes (x extent, in pixels) at which a point
  may be searched. A level runs at a point only when its chip size lies within the range. `0` means
  unbounded and is the default. The finest level ignores these bounds and attempts every point with a
  positive radius.
- `positioned`: `false` marks a point whose `x`, `y` are a placeholder, such as a geogrid point outside
  the image footprint, which is given an arbitrary coordinate and a zero radius. Defaults to `true`.
  It does not affect whether a point is searched ([`AutoRIFT.issearchable`](@ref) tests the radius
  alone). Blocked processing uses it to exclude placeholder coordinates when sizing a block's read
  window.
- `origin`: for a gridded set, the `(row, column)` index of point `[1, 1]` on a lattice shared by
  every grid laid out on it, `(0, 0)` by default. The multi-chip-size search places each coarser
  level's cells and each coarse pass's sample points by this index rather than by the grid's own first
  point, so two grids cut from one map grid at different windows measure every level at the same map
  positions. It must be a multiple of every level's stride on both axes. At `(0, 0)` the lattices
  start at the grid's first point, as in the reference.

Each field has its own type parameter, because a field is an `AutoRIFT.Uniform` exactly when it was
given as a scalar, and fields may be mixed.

See also [`pointset`](@ref), [`gridpoints`](@ref), [`npoints`](@ref).
"""
struct PointSet{N,X<:AbstractArray{Float64,N},Y<:AbstractArray{Float64,N},
                RX<:AbstractArray{Int,N},RY<:AbstractArray{Int,N},
                DX<:AbstractArray{Float64,N},DY<:AbstractArray{Float64,N},
                CX<:AbstractArray{Int,N},CY<:AbstractArray{Int,N},
                CN<:AbstractArray{Int,N},CM<:AbstractArray{Int,N},
                PO<:AbstractArray{Bool,N}}
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
    positioned::PO
    origin::NTuple{2,Int}

    function PointSet(x::X, y::Y, radius_x::RX, radius_y::RY, dx_prior::DX,
                      dy_prior::DY, chip_size_x::CX, chip_size_y::CY,
                      chip_size_min_x::CN = Uniform(0, size(chip_size_x)),
                      chip_size_max_x::CM = Uniform(0, size(chip_size_x)),
                      positioned::PO = Uniform(true, size(chip_size_x)),
                      origin::NTuple{2,Integer} = (0, 0)) where {
                          N,X<:AbstractArray{Float64,N},Y<:AbstractArray{Float64,N},
                          RX<:AbstractArray{Int,N},RY<:AbstractArray{Int,N},
                          DX<:AbstractArray{Float64,N},DY<:AbstractArray{Float64,N},
                          CX<:AbstractArray{Int,N},CY<:AbstractArray{Int,N},
                          CN<:AbstractArray{Int,N},CM<:AbstractArray{Int,N},
                          PO<:AbstractArray{Bool,N}}
        ax = axes(x)
        _check_axes(ax, (y = y, radius_x = radius_x, radius_y = radius_y,
                         dx_prior = dx_prior, dy_prior = dy_prior,
                         chip_size_x = chip_size_x, chip_size_y = chip_size_y,
                         chip_size_min_x = chip_size_min_x,
                         chip_size_max_x = chip_size_max_x,
                         positioned = positioned))
        return new{N,X,Y,RX,RY,DX,DY,CX,CY,CN,CM,PO}(
            x, y, radius_x, radius_y, dx_prior, dy_prior,
            chip_size_x, chip_size_y, chip_size_min_x, chip_size_max_x, positioned, Int.(origin))
    end
end

"""
    npoints(pts::PointSet) -> Int

Number of search centers, including those skipped for a zero search radius. See
[`nsearchable`](@ref) for the number that will be correlated.
"""
npoints(pts::PointSet) = length(pts.x)

Base.length(pts::PointSet) = npoints(pts)
Base.size(pts::PointSet) = size(pts.x)
Base.eachindex(pts::PointSet) = eachindex(pts.x)
Base.ndims(::PointSet{N}) where {N} = N
Base.isempty(pts::PointSet) = isempty(pts.x)

"""
    nsearchable(pts::PointSet) -> Int

Number of points with a non-zero search radius in both axes, which are the points that will be
correlated.

Can be much smaller than [`npoints`](@ref), because the coarse pass zeroes the radius wherever it found
no coherent motion.
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

Whether point `i` has a positive search radius in both axes. A window with no extent in one axis
cannot yield a displacement.
"""
@inline issearchable(pts::PointSet, i) =
    @inbounds pts.radius_x[i] > 0 && pts.radius_y[i] > 0

"""
    pointset(x, y; kwargs...) -> PointSet

Build a [`PointSet`](@ref) from center coordinates, filling unspecified geometry from scalar defaults.

`x` and `y` are arrays of the same shape: vectors for scattered points, matrices for gridded ones.
`pointset(xy; kwargs...)` takes a single vector of `(x, y)` tuples, and `pointset(idx; kwargs...)` a
vector of `CartesianIndex{2}`, read as `(row, col) = (y, x)`. Each geometry keyword accepts a scalar,
applied to every point, or an array matching the shape of `x`. A fractional value for an integer field
throws `ArgumentError`, and a mismatched shape throws `DimensionMismatch`.

```jldoctest
julia> pts = AutoRIFT.pointset([10.0, 50.5, 200.0], [20.0, 60.0, 30.25];
                               search_radius = 25, chip_size = 32);

julia> AutoRIFT.npoints(pts)
3

julia> pts.x[2], pts.radius_x[2], pts.chip_size_x[2]
(50.5, 25, 32)
```

# Keywords
- `search_radius = 25`: half-extent of the search window, an [`AutoRIFT.Extent`](@ref). A scalar means
  square; `(X = …, Y = …)` names the axes.
- `chip_size = 32`: chip extent, likewise.
- `search_radius_x`, `search_radius_y`, `chip_size_x`, `chip_size_y`: per-axis overrides of the
  extents above. Unlike an extent, they accept a per-point array.
- `dx_prior = 0.0`, `dy_prior = 0.0`: a-priori displacement, a scalar or per-point array.
- `chip_size_min_x = 0`, `chip_size_max_x = 0`: the chip sizes at which a point may be searched. `0`
  means unbounded. A scalar or per-point array.
- `positioned = true`: whether each coordinate is a real image position. A scalar or per-point array.
  See [`PointSet`](@ref).
- `origin = (0, 0)`: the lattice index of the first point, for a gridded set. See [`PointSet`](@ref).
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
    positioned = true,
    origin = (0, 0),
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

    # Also a constant field, so a caller who does not mention it pays 24 bytes rather than a
    # grid-sized array — which is every caller but a geogrid.
    pos = _toconstfield(Bool, positioned, x, :positioned)

    return PointSet(xs, ys, rx, ry, dx, dy, csx, csy, cmin, cmax, pos, origin)
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
# *field*, not of the call: see `AutoRIFT.Uniform` for why `radius_x` cannot.
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

Build a gridded [`PointSet`](@ref): centers on a regular lattice laid out in 2-D, as the
multi-chip-size search requires.

The first form places centers every `spacing` pixels (an [`AutoRIFT.Extent`](@ref) or scalar) across an
image of size `imagesize = (nrows, ncols)`, inset from the edges so that a chip and its search window
fit. It throws `ArgumentError` if `spacing` is not positive or no center fits. The second form takes
explicit coordinate vectors; rows of the result vary with `ys` and columns with `xs`.

```jldoctest
julia> pts = AutoRIFT.gridpoints((512, 512), 32; chip_size = 32, search_radius = 25);

julia> ndims(pts), size(pts)
(2, (14, 14))
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

View a `PointSet` as a flat list of points, discarding the grid layout. Each field is `vec` of the
original, so the result shares memory with `pts`: mutating either is visible in the other.
"""
scatter(pts::PointSet{1}) = pts
scatter(pts::PointSet) = rebuild(pts;
    x = vec(pts.x), y = vec(pts.y),
    radius_x = vec(pts.radius_x), radius_y = vec(pts.radius_y),
    dx_prior = vec(pts.dx_prior), dy_prior = vec(pts.dy_prior),
    chip_size_x = vec(pts.chip_size_x), chip_size_y = vec(pts.chip_size_y),
    chip_size_min_x = vec(pts.chip_size_min_x),
    chip_size_max_x = vec(pts.chip_size_max_x),
    positioned = vec(pts.positioned))

"""
    pts[rows, cols] -> PointSet{2}

Decimate or crop a gridded point set, copying every field. `rows` and `cols` index the grid as for a
matrix. The result is independent of `pts`.
"""
Base.getindex(pts::PointSet{2}, rows, cols) = PointSet(
    pts.x[rows, cols], pts.y[rows, cols],
    pts.radius_x[rows, cols], pts.radius_y[rows, cols],
    pts.dx_prior[rows, cols], pts.dy_prior[rows, cols],
    pts.chip_size_x[rows, cols], pts.chip_size_y[rows, cols],
    pts.chip_size_min_x[rows, cols], pts.chip_size_max_x[rows, cols],
    pts.positioned[rows, cols], _sub_origin(pts.origin, rows, cols))

# The lattice index of `pts[rows, cols]`'s first point. A selection of every `step`-th point is on the
# lattice of every `step`-th lattice index, where fine index `i` lies in node `fld(i, step)`; a crop is
# the case `step = 1`.
_sub_origin(o::NTuple{2,Int}, rows::AbstractRange, cols::AbstractRange) =
    (fld(o[1] + first(rows) - 1, step(rows)), fld(o[2] + first(cols) - 1, step(cols)))

"""
    rebuild(pts::PointSet; kwargs...) -> PointSet

A `PointSet` with the named fields replaced. Fields not named are shared with `pts`, not copied. The
keywords are the field names of [`PointSet`](@ref); the replacements must have matching axes.
"""
rebuild(pts::PointSet; x = pts.x, y = pts.y,
        radius_x = pts.radius_x, radius_y = pts.radius_y,
        dx_prior = pts.dx_prior, dy_prior = pts.dy_prior,
        chip_size_x = pts.chip_size_x, chip_size_y = pts.chip_size_y,
        chip_size_min_x = pts.chip_size_min_x,
        chip_size_max_x = pts.chip_size_max_x,
        positioned = pts.positioned, origin = pts.origin) =
    PointSet(x, y, radius_x, radius_y, dx_prior, dy_prior, chip_size_x, chip_size_y,
             chip_size_min_x, chip_size_max_x, positioned, origin)

"""
    sanitize!(pts::PointSet, min_radius) -> Int

Normalize search radii in place and return the number of points left searchable. Two rules, from the
reference (`autoRIFT.py:547-551`):

1. A point whose radius is non-positive in either axis is unsearchable, so both of its radii are set
   to zero.
2. Every other point gets at least `min_radius` in each axis, since a correlation surface only a few
   pixels across cannot support sub-pixel refinement. Points zeroed by rule 1 stay zero.
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

The chip is centered on `floor(x - dx_prior)` and `floor(y - dy_prior)`, so the search begins where
motion is expected. Its extent is `chip_size_y` rows by `chip_size_x` columns, from `c - h` to
`c + h - 1` with `h = chip_size ÷ 2`.
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

The window is centered on `floor(x)` and `floor(y)`, without the prior. It is larger than the chip by
the search radius: it extends `radius` pixels before the chip's first index but only `radius - 1`
after its last, giving a width of `chip + 2 * radius - 1`. Correlating the chip against it yields a
surface of `(2 * radius_y, 2 * radius_x)` samples with zero displacement on a sample. A symmetric
window would give an odd-sized surface whose center is half a sample from zero displacement.
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
