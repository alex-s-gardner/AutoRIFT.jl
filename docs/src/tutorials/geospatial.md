
# Geospatial data

[A guided walkthrough](@ref) uses plain arrays and returns pixel offsets. For geolocated images, pass
`autorift` rasters instead. Three things change: the output carries the input's coordinates and CRS,
the signs follow map conventions, and an acquisition interval converts displacements to velocities.
The correlation itself is the same.

## What you need loaded

```@example geo
using AutoRIFT, Rasters, ArchGDAL, Dates
include("../../figures.jl")  # plotting helpers and the example scenes; see [Plotting](@ref)
using DimensionalData.Lookups
using Statistics: median

Base.get_extension(AutoRIFT, :AutoRIFTRastersExt) === nothing &&
    error("the Rasters path is not loaded")
nothing # hide
```

Loading `Rasters` and `ArchGDAL` together activates the geospatial method. `ArchGDAL` supplies GDAL,
which opening a file requires. Without it, `autorift` on a `Raster` silently uses the plain
`DimensionalData` method and returns `dx`/`dy` where you expected `vx`/`vy`.

## A projected pair

Real imagery would be two GeoTIFFs. This page builds a pair from the example photograph so that the
true displacement is known. The structure is the same as a GeoTIFF's: a north-up grid, y
**decreasing**, with a CRS.

```@example geo
# Features move 20 pixels east and 8 pixels north. In `autorift`'s reporting convention that is a
# negative `dx` and a positive `dy`, since the offset points from secondary back to reference and
# north is toward *decreasing* row on a north-up grid. See [Conventions](@ref).
reference_array, secondary_array, _, _ = warped_pair(photo(), 513, (row, col) -> (-20.0, 8.0))

const RES = 30.0  # metres per pixel

function northup(A; epsg = 3413)
    nrows, ncols = Base.size(A)
    y = Y(Projected((RES * (nrows - 1)):(-RES):0.0; order = ReverseOrdered(),
                    span = Regular(-RES), sampling = Intervals(Start()), crs = EPSG(epsg)))
    x = X(Projected(0.0:RES:(RES * (ncols - 1)); order = ForwardOrdered(),
                    span = Regular(RES), sampling = Intervals(Start()), crs = EPSG(epsg)))
    return Raster(A, (y, x))
end

reference, secondary = northup(reference_array), northup(secondary_array)
image_panels(reference_array, secondary_array)
```

EPSG:3413 is NSIDC Sea Ice Polar Stereographic North, a projected CRS in meters. `dt` scales by the
lookup's own units, so a projected CRS gives meters per year and a geographic CRS gives degrees per
year.

## Correlating

The call and keywords are the same as for arrays. The two rasters must have identical lookups and
CRS, otherwise `autorift` throws.

```@example geo
out = autorift(reference, secondary; chip_size = 32, chip_size_max = 32,
               search_radius = 28, grid_spacing = 16)
(type = typeof(out).name.name, layers = propertynames(out))
```

The result is a `RasterStack`. Its layers are named `vx`/`vy` rather than `dx`/`dy` to show that the
sign and orientation conversions were applied.

The output grid is the input grid at `grid_spacing` times the pixel size, inheriting CRS and axis order:

```@example geo
(size = Base.size(out.vx), crs = crs(out.vx), dims = map(name, dims(out)),
 y_order = order(dims(out, Y)))
```

Every layer has its own `missingval`, which is what a GDAL writer needs to record a gap:

```@example geo
(vx = missingval(out.vx), chip_size = missingval(out.chip_size))
```

The floating-point layers use `NaN`. `chip_size` uses `0`, which means no level answered.

## Plotting against projected coordinates

Makie has a Rasters recipe that reads coordinates from the dimensions, so no orientation helper is
needed:

```@example geo
using CairoMakie
fig = Figure(; size = (960, 420))
lim = (-24, 24)
for (i, (layer, title)) in enumerate(((out.vx, "vx (px east)"), (out.vy, "vy (px north)")))
    ax = Axis(fig[1, i]; title, aspect = DataAspect(), xlabel = "x (m)", ylabel = "y (m)")
    hm = heatmap!(ax, layer; colormap = :balance, colorrange = lim)
    i == 2 && Colorbar(fig[1, 3], hm)
end
fig
```

Plotting against the coordinates, rather than through the array helpers in [Plotting](@ref), makes the
convention checkable by eye: the y axis increases upward, so north is up, and positive `vx` is east.
On a raw array neither is visible, which is how a y-sign error survives review.

## The sign conventions

`+vx` points **east** and `+vy` points **north**, whatever row order the file uses. Two conversions
produce this from the core's output:

- **Sign.** The core reports the offset from secondary back to reference, which is the negative of
  the feature's motion. The raster method reports the motion itself.
- **Orientation.** `dy` becomes north-positive rather than row-positive.

The same data through both methods:

```@example geo
array_out = autorift(reference_array, secondary_array; chip_size = 32, chip_size_max = 32,
                     search_radius = 28, grid_spacing = 16)
mid(A) = median(filter(isfinite, collect(A)))
(array = (dx = mid(array_out.dx), dy = mid(array_out.dy)),
 raster = (vx = mid(out.vx), vy = mid(out.vy)))
```

`dx = -20` becomes `vx = +20`: the sign flips, giving eastward motion. `dy = +8` becomes `vy = +8`: the
sign flips and the axis direction flips, and the two cancel. On a north-up raster `vy` therefore equals
`dy`, so a y-sign error in the conversion is invisible in the numbers.

## North-up and south-up give the same answer

The core receives the array in the file's row order, so the same scene stored south-up is vertically
mirrored and the core's `dy` changes sign. The extension reads the lookup's direction and corrects
for it, so the scene gives the same field stored either way:

```@example geo
function southup(A; epsg = 3413)
    nrows, ncols = Base.size(A)
    y = Y(Projected(0.0:RES:(RES * (nrows - 1)); order = ForwardOrdered(),
                    span = Regular(RES), sampling = Intervals(Start()), crs = EPSG(epsg)))
    x = X(Projected(0.0:RES:(RES * (ncols - 1)); order = ForwardOrdered(),
                    span = Regular(RES), sampling = Intervals(Start()), crs = EPSG(epsg)))
    # The array is reversed too: the same ground, written bottom row first.
    return Raster(reverse(A; dims = 1), (y, x))
end

flipped = autorift(southup(reference_array), southup(secondary_array);
                   chip_size = 32, chip_size_max = 32, search_radius = 28, grid_spacing = 16)
(north_up = (mid(out.vx), mid(out.vy)), south_up = (mid(flipped.vx), mid(flipped.vy)),
 y_orders = (order(dims(out, Y)), order(dims(flipped, Y))))
```

The velocities are the same and the storage orders are opposite. The package tests compare the full
field rather than a median, because a field with `vy` inverted still looks like plausible flow.

## Velocity: the `dt` keyword

Without `dt`, `vx`/`vy` are in **pixels**. With the interval between acquisitions they are in **CRS
units per year**, meters per year for a projected raster:

```@example geo
velocity = autorift(reference, secondary; dt = Day(16), chip_size = 32, chip_size_max = 32,
                    search_radius = 28, grid_spacing = 16)
(pixels = mid(out.vx), m_per_yr = mid(velocity.vx))
```

The conversion is:

```
meters per year = pixels × pixel_size / (dt in days / 365.25)
```

```@example geo
mid(out.vx) * RES / (16 / 365.25)
```

This matches the `dt` result. The reverse conversion helps choose a `search_radius`, which must cover the
fastest motion expected over this pair's interval. A published velocity in m/yr converts to pixels the
same way:

```@example geo
# A 5000 m/yr glacier over a 16-day pair, at 30 m pixels.
5000 * (16 / 365.25) / RES
```

That is about 7 pixels, so the radius of 28 used here is generous for that glacier over a 16-day pair. Over a one-year pair the same
glacier moves about 167 pixels, and the radius, the chip size, or a prior has to change. A radius
smaller than the motion is the most common reason a correlation returns nothing; see
[Judging a result](@ref).

`dt` also accepts a `Real` number of years or any fixed-length `Dates.Period` such as `Day(16)` or
`Week(2)`. `Month` and `Year` throw, because they have no fixed length; use `Day(365)` or a number of
years. `dt` also throws if a lookup is irregularly spaced, since no single pixel size exists.

The quality layers `correlation`, `peak_ratio`, `chip_size` and `interpolated` are the same as on the
array path:

```@example geo
quality_panels((dx = collect(out.vx), dy = collect(out.vy),
                correlation = collect(out.correlation), chip_size = collect(out.chip_size)))
```

## Files on disk, and masks

A raster opened with `lazy = true` is correlated from the file without being loaded, block by block:

```julia
a = Raster("early.tif"; lazy = true)
b = Raster("late.tif"; lazy = true)
out = autorift(a, b; dt = Day(16), grid_spacing = 8, threaded = true)
```

A raster's `missingval` becomes a mask, so nodata pixels are excluded rather than read as values such as
`-9999`. For other unusable pixels (clouds, shadow, water), pass `reference_valid` or `secondary_valid` as
a boolean raster or array, `true` where the pixel is valid. [Correlating scenes larger than memory](@ref)
covers blocked runs, and [Masking invalid pixels](@ref) covers masks.

## Not yet supported

`dt` scales by the lookup's pixel size, which assumes that displacement along a pixel axis is
displacement along a map axis. This holds for any north-up projected raster, including nearly all
Landsat and Sentinel-2 imagery. It does **not** hold for a rotated grid or for radar range-Doppler
geometry, where the conversion needs per-pixel geolocation matrices, which are not implemented. For
those, use `dt = nothing` and convert the pixel displacements yourself.

Radar geometry can still be correlated. A `DimArray` that is not a `Raster` uses the `DimensionalData`
method. It returns a `DimStack` of `dx`/`dy` in pixels with no sign or orientation conversion, since
without a CRS there is no north. That method does not accept `dt`.

## See also

- [Conventions](@ref) — the sign and the half pixel, and how to settle a convention question
- [Correlating scenes larger than memory](@ref) — blocking, the cache, and the measured peaks
- [Geospatial](@ref) — the extension's full docstring
- [A guided walkthrough](@ref) — the parameters, on plain arrays
