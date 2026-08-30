"""
    AutoRIFTRastersExt

Accept `Raster` input and return a `RasterStack` carrying the output grid's coordinates, CRS, and
per-layer `missingval`.

Builds on `AutoRIFTDimensionalDataExt` rather than repeating it: a `Raster` is an
`AbstractDimArray`, so grid alignment checking and output-dimension construction are shared. What
this adds is everything that needs a *map*: CRS validation between the two inputs, the y-flip, and
conversion from pixel offsets to velocity.

# The y-flip, and why it gets its own boundary

A north-up GeoTIFF stores y **decreasing** — the first row is the northernmost. A south-up file
stores it increasing. `parent()` hands the core whichever order the file used, so the same scene
written both ways produces `dy` of opposite sign: `dx` is unchanged, `dy` is negated. Measured, not
assumed.

That is a bad failure because it stays plausible. A velocity field with `vy` inverted still looks
like ice flow; nothing downstream can detect it, and the error is in the *file*, not the code. So
this extension is where the y direction is read and corrected, exactly once, and the tests assert
that both storage orders give the identical field.

# `vx`/`vy`, not `dx`/`dy`

Two conversions happen here, and the names record that they did:

  * **Orientation.** `dy` becomes north-positive rather than row-positive.
  * **Sign.** The core reports the offset from secondary back to reference, which is the negative
    of how the ground moved. Here it becomes feature motion: `+vx` east, `+vy` north.

`dx`/`dy` are reserved for raw pixel offsets in the array and `DimStack` paths, matching
ITS_LIVE's published naming. A caller who wants the unconverted offsets can correlate
`parent(raster)` directly.
"""
module AutoRIFTRastersExt

import AutoRIFT
import DiskArrays
import Rasters

using Dates: Dates, Day, Millisecond, Month, Period, Year
using DimensionalData: dims, lookup
using DimensionalData.Lookups: ReverseOrdered, order
using Rasters: AbstractRaster, RasterStack, crs

# Sibling extension, loaded first: Rasters depends on DimensionalData, so both of this module's
# triggers being present implies that module exists. Reached through `get_extension` rather than
# `import`, since an extension is not a package and cannot be imported by name.
const DDExt = Base.get_extension(AutoRIFT, :AutoRIFTDimensionalDataExt)

# Seconds in a Julian year, the convention ITS_LIVE publishes velocities in.
const MS_PER_YEAR = 365.25 * 24 * 60 * 60 * 1000

"""
    autorift(reference::AbstractRaster, secondary::AbstractRaster; dt = nothing, kwargs...)

Correlate two rasters, returning a `RasterStack` on the output grid.

Layers are `vx`, `vy`, `correlation`, `chip_size`, and `interpolated`, each with its own
`missingval`. The output grid inherits the inputs' CRS and axis order, at `grid_spacing` times
their pixel size.

# Keywords

All of [`AutoRIFT.params`](@ref)'s, plus:

- `dt = nothing`: the interval between acquisitions. `nothing` leaves `vx`/`vy` in **pixels**; a
  `Dates.Period` such as `Day(16)`, or a `Real` count of years, converts them to **CRS units per
  year** — metres per year for a projected raster.
- `reference_valid`, `secondary_valid`: per-pixel validity masks, as arrays or rasters.
- `process_block_size`: `(X, Y)` pixels per block. Defaults to a halo-derived size, rounded up to the
  rasters' own chunking, for file-backed input; `nothing` — one block — for input already in memory.

```julia
using AutoRIFT, Rasters, ArchGDAL, Dates
out = autorift(Raster("early.tif"), Raster("late.tif"); grid_spacing = 32, dt = Day(16))
out.vx, out.vy, out.correlation
```

`ArchGDAL` has to be loaded to open a file: it is what carries GDAL, through `RastersArchGDALExt`,
and this extension requires it for that reason.

# Reading a raster that is still on disk

`lazy = true` correlates from the file, and this is the case the defaults are tuned for:

```julia
a = Raster("early.tif"; lazy = true)
b = Raster("late.tif"; lazy = true)
out = autorift(a, b; grid_spacing = 8, threaded = true)
```

Two things happen automatically, and both matter on a scene large enough to care about. The run is
**blocked**, so no array the size of the scene is formed — a block reads its own window and nothing
else. And **nodata becomes mask rather than number**: a GDAL raster's
`missingval` marks pixels that are excluded from correlation instead of being read as a dark
measurement, which is what reading a `-9999` fill would amount to.

The answer is **bit-identical** to the same pair materialized, at any block size; the tests assert
that on all five layers, since a windowed read that computed something subtly different would be
worse than one that was merely slow. Measured on a Landsat 8/9 pair over Jakobshavn — 17121x16961,
4.48 M grid points — reading from the two GeoTIFFs peaks at **2.2 GiB against 7.0 GiB** for the same
run from memory.

!!! note "Sign convention"
    `vx` and `vy` are **feature motion in map orientation**: `+vx` points east, `+vy` north. Both
    conversions from the core's convention — the orientation and the sign — happen here, so a
    north-up and a south-up raster of one scene give the same answer.

!!! warning "Velocity is exact only for an axis-aligned grid"
    `dt` scales by the lookup's own pixel size, which assumes displacement along a pixel axis is
    displacement along a map axis. True for any north-up projected raster, so for essentially all
    optical Landsat and Sentinel-2 imagery. It is *not* true for a rotated grid or for radar
    geometry, where the conversion needs Geogrid's per-pixel matrices — not yet implemented, so
    pass `dt = nothing` and convert externally in that case.
"""
function AutoRIFT.autorift(reference::AbstractRaster, secondary::AbstractRaster;
                           dt = nothing, reference_valid = nothing, secondary_valid = nothing,
                           process_block_size = nothing, kwargs...)
    DDExt.check_aligned(reference, secondary)
    _check_crs(reference, secondary)

    # A file-backed raster is correlated where it lies: nodata becomes mask rather than number, and the
    # run is blocked so no array the size of the scene is formed. Both are no-ops for a raster already
    # in memory, so this path is the same computation either way — see `_ondisk` and `_blocks`.
    rimg, rvalid = _lazy_input(reference, DDExt.unwrap(reference_valid))
    simg, svalid = _lazy_input(secondary, DDExt.unwrap(secondary_valid))
    # `params` resolved here as well as inside the core: the block size depends on the halo, which
    # depends on the chip size, search radius and filter width. Resolving keywords is pure and cheap
    # relative to a correlation, and the alternative is guessing at a halo the reads then contradict.
    blocks = _blocks(process_block_size, reference, secondary, AutoRIFT.params(; kwargs...))

    result, grid = AutoRIFT.autorift_with_grid(
        rimg, simg;
        reference_valid = rvalid, secondary_valid = svalid,
        process_block_size = blocks, kwargs...)

    outdims = DDExt.grid_dims(reference, grid)
    vx, vy = _to_velocity(result, reference, dt)
    layers = (vx = vx, vy = vy, correlation = result.correlation,
              chip_size = result.chip_size,
              interpolated = Matrix{Bool}(result.interpolated))
    # Per-layer `missingval`, set at construction rather than left to a caller to discover. `NaN`
    # for the floating-point layers because that is what the core writes for "not measured", which
    # is deliberately distinct from a measured zero; `0` chip size means no level resolved the
    # point; `false` is not missing data but the honest default for a mask.
    return RasterStack(layers, outdims;
                       missingval = (vx = NaN32, vy = NaN32, correlation = NaN32,
                                     chip_size = UInt16(0), interpolated = false))
end

# ---------------------------------------------------------------------------
# Reading a raster where it lies
# ---------------------------------------------------------------------------

# Whether `r`'s data is still on disk.
#
# `DiskArrays.isdisk` rather than a trait test of our own: it is the predicate Rasters itself uses for
# this question, and it follows the property that matters — does a read cost I/O — rather than a list of
# backend types to keep up to date.
_ondisk(r::AbstractRaster) = DiskArrays.isdisk(parent(r))

# The array and validity mask to hand the core.
#
# Two things happen for a file-backed raster, and neither touches the imagery:
#
#   * **Nodata becomes mask.** A GDAL raster's eltype is `Union{Missing,T}`, which `ImagePair` rejects
#     — rightly, since `missing` in an FFT poisons the whole surface. `Rasters.replace_missing` swaps
#     the fill for `zero(T)` *lazily*, and the pixels it swapped are recorded in the mask, so they are
#     excluded from correlation rather than read as a dark measurement.
#   * **A caller's own mask is combined**, not overridden. Nodata is a property of the file; a cloud or
#     shadow mask is a property of the scene, and a caller passing one should not lose the other.
#
# The masks stay lazy: `AutoRIFT.ImagePair` passes a supplied mask through without materializing, and
# `_read_block!` reads windows of it into dense buffers. Materializing here would form exactly the
# scene-sized array that blocking exists to avoid.
function _lazy_input(r::AbstractRaster, supplied)
    # Keyed on whether the raster *declares* nodata, not on where its data lives. A raster read into
    # memory keeps both its `missingval` and its `Union{Missing,T}` eltype, so gating this on
    # disk-backedness would leave the in-memory case unable to correlate at all — and would make the
    # lazy and eager paths answer differently, which is the one thing this must not do.
    mv = Rasters.missingval(r)
    isnothing(mv) && return parent(r), supplied
    filled = parent(Rasters.replace_missing(r, zero(nonmissingtype(eltype(r)))))
    valid = _nodata_mask(r, mv)
    return filled, isnothing(supplied) ? valid : _Both(valid, supplied)
end

# "Not the fill value", as a lazy `Bool` array over the raster's own storage.
#
# `missing` and a sentinel number are the two forms GDAL reports and they need different tests, so this
# dispatches on which one the file declared rather than comparing against a value that may be `missing`
# — `x == missing` is `missing`, not `false`, which would make every pixel indeterminate.
# `missing` and a sentinel number are the two forms GDAL reports, and one type serves both: `missing`
# *is* the sentinel in the first case, so `!isequal(x, value)` covers it — `isequal` rather than `!=`
# because `x == missing` is `missing`, not `false`, which would leave every pixel indeterminate.
_nodata_mask(r::AbstractRaster, mv::Missing) = _NotFill(parent(r), mv)
_nodata_mask(r::AbstractRaster, mv) = _NotFill(parent(r), convert(eltype(r), mv))

# Lazy masks rather than `map`: a `map` over a disk array is itself lazy, but its element type and
# indexing behaviour depend on the backend, where these are `AbstractMatrix{Bool}` by construction and
# forward a windowed read straight to the parent — which is what makes a block read one window rather
# than one pixel at a time. The same shape as `AutoRIFT.FiniteMask`, for the same reason.

# "Not the fill value", for either kind of fill.
struct _NotFill{A,T} <: AbstractMatrix{Bool}
    parent::A
    value::T
end

# Two masks, both of which must pass. Nodata comes from the file and a cloud or shadow mask from the
# caller; a pixel is usable only if neither excludes it.
struct _Both{A,B} <: AbstractMatrix{Bool}
    a::A
    b::B
end

Base.size(m::_NotFill) = size(m.parent)
Base.axes(m::_NotFill) = axes(m.parent)
Base.size(m::_Both) = size(m.a)
Base.axes(m::_Both) = axes(m.a)

Base.@propagate_inbounds Base.getindex(m::_NotFill, I::Int...) = !isequal(m.parent[I...], m.value)
Base.@propagate_inbounds Base.getindex(m::_Both, I::Int...) = m.a[I...] && m.b[I...]

# Windowed reads, one read of the parent each. Without these a block read walks the mask element by
# element, which for a disk-backed parent is one I/O call per pixel — measured at 274 us per element
# against 1.5 ms for the whole window.
Base.@propagate_inbounds Base.getindex(m::_NotFill, r::AbstractUnitRange, c::AbstractUnitRange) =
    map(x -> !isequal(x, m.value), m.parent[r, c])
Base.@propagate_inbounds Base.getindex(m::_Both, r::AbstractUnitRange, c::AbstractUnitRange) =
    m.a[r, c] .& m.b[r, c]

# The block size to correlate at, when the caller did not choose one.
#
# A caller's choice always wins, and an in-memory pair keeps `nothing` — the untiled path — so nothing
# changes for callers who were already passing arrays.
#
# For a file-backed pair the size is set by the **halo**, not by the file's chunking, and measurement is
# why. A block reads its own extent grown by the halo on every side, so redundancy is
# `((block + 2*halo) / block)^2`: at the 56-pixel halo of a default Landsat configuration a block equal
# to the 256-pixel chunk reads 369² and touches 2x2 chunks, so every chunk is decoded 3.7 times and the
# scene is read 1.94 times over. Blocking at the chunk size sounds aligned and is the worst option
# measured — 4.99 s against 3.48 s at 768, with 464k allocations against 66k.
#
# `HALO_BLOCKS` halos per block puts redundancy at `(14/12)^2 = 1.36x`, near the measured optimum. Above
# it the gain flattens while per-block buffers grow with the square of the block; below it the halo
# dominates. Rounded up to a whole number of chunks so a read still starts and ends on a chunk boundary.
#
# The larger of the two rasters' chunk sizes, since one block size serves both: taking the smaller would
# read a partial chunk of the coarser file for every block.
const HALO_BLOCKS = 12

# A floor for the pathological chunkings. A *striped* GeoTIFF — what `Rasters.write` produces by
# default — reports chunks one row tall, so a 384x384 file asks for `(384, 5)`: below the halo, which
# `block_layout` rejects outright, and nearly all overlap even if it did not.
const MIN_BLOCK = 256

function _blocks(supplied, reference::AbstractRaster, secondary::AbstractRaster, p)
    isnothing(supplied) || return supplied
    (_ondisk(reference) && _ondisk(secondary)) || return nothing
    cr = DiskArrays.approx_chunksize(DiskArrays.eachchunk(parent(reference)))
    cs = DiskArrays.approx_chunksize(DiskArrays.eachchunk(parent(secondary)))
    # `approx_chunksize` reports one entry per dimension; a `Raster` here is 2-D by `check_aligned`.
    chunk = (max(cr[1], cs[1]), max(cr[2], cs[2]))
    # The run's actual halo, from the run's actual parameters, so this cannot drift from the value the
    # reads will use. The parameter-only form: the grid form would build a `PointSet` — 550 MiB and
    # 50 ms on a Landsat-sized scene — to arrive at the same two integers.
    hx, hy = AutoRIFT.halo(p)
    # `halo` returns `(x, y)` while `size` is `(rows, cols)` = `(y, x)`, so the axes cross here.
    # Never larger than the scene: a block wider than the image is one block, which is the untiled path
    # wearing a block label, and it makes the trailing-block arithmetic do nothing useful.
    return (min(max(_round_up(HALO_BLOCKS * hy, chunk[1]), MIN_BLOCK), size(reference, 1)),
            min(max(_round_up(HALO_BLOCKS * hx, chunk[2]), MIN_BLOCK), size(reference, 2)))
end

_round_up(want::Int, unit::Int) = unit <= 0 ? want : cld(want, unit) * unit

# ---------------------------------------------------------------------------

# Both images must be in the same coordinate system. Two rasters can share a grid shape and CRS-free
# lookups yet describe different places, and correlating those produces a full field of plausible
# nonsense — so this is checked before any work is done.
function _check_crs(a::AbstractRaster, b::AbstractRaster)
    ca, cb = crs(a), crs(b)
    ca == cb || throw(ArgumentError(
        "the two images are in different coordinate systems, $(ca) and $(cb), so they are not " *
        "co-registered. Reproject one onto the other's CRS before correlating."))
    return nothing
end

# Pixel offsets to map-oriented feature motion, optionally per year.
#
# Two sign corrections, and they are independent:
#
#   * `-r.dx`, `-r.dy` turn the secondary-to-reference offset into feature motion.
#   * `dy` counts down rows. On a north-up raster (y decreasing) a positive row step goes south,
#     so north-positive `vy` needs a second negation — which cancels the first. On a south-up
#     raster the row direction already agrees with the y axis, so only the first applies.
#
# The two cases therefore differ by exactly one sign on `vy`, which is what makes both storage
# orders yield the same field. Working it as a scale factor rather than a branch around two nearly
# identical loops keeps the reasoning in one place.
function _to_velocity(r::AutoRIFT.MultichipResult, reference::AbstractRaster, dt)
    # x is unconditional: map x always increases with column index, so feature motion is just the
    # negated offset. Only y depends on how the file was written, which is the whole asymmetry.
    rowdim = first(dims(reference))
    sy = order(lookup(rowdim)) isa ReverseOrdered ? 1.0f0 : -1.0f0   # see above

    isnothing(dt) && return (-r.dx, sy .* r.dy)

    years = _years(dt)
    px, py = _pixel_sizes(reference)
    return (Float32.(-r.dx .* (px / years)), Float32.(sy .* r.dy .* (py / years)))
end

# Ground spacing per axis, in CRS units. Refuses rather than guesses when the axis is irregular.
function _pixel_sizes(reference::AbstractRaster)
    rowdim, coldim = dims(reference)
    py = DDExt.pixel_size(lookup(rowdim))
    px = DDExt.pixel_size(lookup(coldim))
    (isnothing(px) || isnothing(py)) && throw(ArgumentError(
        "`dt` was given but the raster's lookups are not regularly spaced, so there is no " *
        "single pixel size to scale by and the velocity would be wrong wherever the spacing " *
        "differs. Resample onto a regular grid, or pass `dt = nothing` and convert the pixel " *
        "displacements yourself."))
    return px, py
end

# The acquisition interval in Julian years.
#
# `Year` and `Month` are calendar periods with no fixed length — `Dates.Millisecond(Year(1))`
# throws rather than picking 365 or 366 days — so they are refused with a message naming what to
# use instead. `Date` subtraction yields a `Day`, which is how a caller most naturally arrives
# here, and that converts exactly.
_years(dt::Period) = Dates.value(Millisecond(dt)) / MS_PER_YEAR
_years(dt::Union{Year,Month}) = throw(ArgumentError(
    "`dt = $dt` is a calendar period with no fixed length, so it cannot be converted to a " *
    "velocity denominator. Use a fixed period such as `Day(365)`, a `Real` number of years, or " *
    "the difference of two `Date`s."))
_years(dt::Real) = dt > 0 ? Float64(dt) : throw(ArgumentError(
    "`dt = $dt` must be positive; it is the interval between the two acquisitions, in years."))
_years(dt) = throw(ArgumentError(
    "`dt` must be a `Dates.Period`, a `Real` number of years, or `nothing`, got a $(typeof(dt))."))

end # module
