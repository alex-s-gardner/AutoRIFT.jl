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
import Rasters
import GeoInterface as GI

using Dates: Dates, Day, Millisecond, Month, Period, Year
using DimensionalData: AbstractDimArray, dims, lookup, rebuild
using DimensionalData.Lookups: ForwardOrdered, Lookups, ReverseOrdered, order
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

```julia
using AutoRIFT, Rasters, Dates
out = autorift(Raster("early.tif"), Raster("late.tif"); grid_spacing = 32, dt = Day(16))
out.vx, out.vy, out.correlation
```

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
                           kwargs...)
    DDExt.check_aligned(reference, secondary)
    _check_crs(reference, secondary)

    result, grid = AutoRIFT.autorift_with_grid(
        parent(reference), parent(secondary);
        reference_valid = DDExt.unwrap(reference_valid),
        secondary_valid = DDExt.unwrap(secondary_valid), kwargs...)

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
    rowdim = first(dims(reference))
    sy = order(lookup(rowdim)) isa ReverseOrdered ? 1.0f0 : -1.0f0   # see above
    sx = -1.0f0

    isnothing(dt) && return (sx .* r.dx, sy .* r.dy)

    years = _years(dt)
    px, py = _pixel_sizes(reference)
    return (Float32.(sx .* r.dx .* (px / years)), Float32.(sy .* r.dy .* (py / years)))
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
