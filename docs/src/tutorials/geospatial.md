
# Geospatial data

Everything in [A guided walkthrough](@ref) works on plain arrays and returns pixel offsets. If your
images are geolocated, hand `autorift` rasters instead and three things change: the output carries the
input's coordinates and CRS, the sign conventions become map conventions, and with an acquisition
interval the displacements become velocities.

Nothing about the correlation changes. This page is about the layer on top of it.

## What you need loaded

```@example geo
using AutoRIFT, Rasters, ArchGDAL, Dates
include("../../figures.jl")  # plotting helpers and the synthetic scenes; see [Plotting](@ref)
using DimensionalData.Lookups
using Statistics: median

Base.get_extension(AutoRIFT, :AutoRIFTRastersExt) === nothing &&
    error("the Rasters path is not loaded")
nothing # hide
```

`Rasters` triggers the geospatial method. `ArchGDAL` is what carries GDAL, so it is required to open a
file — and without it `autorift` silently falls through to the plain `DimensionalData` method, returning
`dx`/`dy` where you expected `vx`/`vy`. Load both.

## A projected pair

Real imagery would be two GeoTIFFs. This page builds its pair in code so every number on it is
reproducible, but the structure is the same: a north-up grid, y **decreasing**, with a CRS.

```@example geo
# Features move 8 pixels east and 4 pixels north. In `autorift`'s reporting convention that is a
# negative `dx` and a positive `dy`, since the offset points from secondary back to reference and
# north is toward *decreasing* row on a north-up grid. See [Conventions](@ref).
reference_array, secondary_array, _, _ = warped_pair(513, (row, col) -> (-8.0, 4.0); seed = 3)

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

EPSG:3413 is NSIDC Sea Ice Polar Stereographic North, a projected CRS in metres. Any projected CRS
works; a geographic one in degrees does not give meaningful velocities, since `dt` scales by the
lookup's own units.

## Correlating

The same call, the same keywords:

```@example geo
out = autorift(reference, secondary; chip_size = 32, chip_size_max = 32,
               search_radius = 20, grid_spacing = 16)
(type = typeof(out).name.name, layers = propertynames(out))
```

A `RasterStack`, and the layer names are `vx`/`vy` rather than `dx`/`dy` — which is the signal that
conversions happened.

The output grid is the input grid at `grid_spacing` times the pixel size, inheriting CRS and axis order:

```@example geo
(size = Base.size(out.vx), crs = crs(out.vx), dims = map(name, dims(out)),
 y_order = order(dims(out, Y)))
```

Every layer has its own `missingval`, which is what a GDAL writer needs to record a gap:

```@example geo
(vx = missingval(out.vx), chip_size = missingval(out.chip_size))
```

`NaN` for the floating-point layers, `0` for `chip_size` — where zero already means "no level answered"
and is not a chip size.

## Plotting against projected coordinates

Makie has a Rasters recipe that reads the coordinates from the dimensions, so no orientation helper is
needed:

```@example geo
using CairoMakie
fig = Figure(; size = (960, 420))
lim = (-10, 10)
for (i, (layer, title)) in enumerate(((out.vx, "vx (px east)"), (out.vy, "vy (px north)")))
    ax = Axis(fig[1, i]; title, aspect = DataAspect(), xlabel = "x (m)", ylabel = "y (m)")
    hm = heatmap!(ax, layer; colormap = :balance, colorrange = lim)
    i == 2 && Colorbar(fig[1, 3], hm)
end
fig
```

The labelled axes are the reason to plot this way rather than through the array helpers in
[Plotting](@ref): they make the sign convention checkable by eye. A field that should point east must brighten toward
increasing x, and the y axis must increase upward. On a raw array neither is visible, which is why a
y-sign error survives review.

## The sign conventions

`+vx` points **east** and `+vy` points **north**, whatever order the file stored its rows in. Two
conversions get you there from the core's output:

- **Sign.** The core reports the offset from secondary back to reference, which is the negative of how
  the ground moved. Here it becomes feature motion.
- **Orientation.** `dy` becomes north-positive rather than row-positive.

Side by side on the same data:

```@example geo
array_out = autorift(reference_array, secondary_array; chip_size = 32, chip_size_max = 32,
                     search_radius = 20, grid_spacing = 16)
mid(A) = median(filter(isfinite, collect(A)))
(array = (dx = mid(array_out.dx), dy = mid(array_out.dy)),
 raster = (vx = mid(out.vx), vy = mid(out.vy)))
```

`dx = -8` becomes `vx = +8`: the sign flipped, and that is motion eastward. `dy = +4` becomes
`vy = +4`: the sign flipped *and* the axis flipped, so the two cancel. That cancellation is exactly why
a y-sign error is hard to catch — on a north-up raster the wrong answer and the right one differ by two
negations that look like none.

## North-up and south-up give the same answer

`parent()` hands the core whichever row order the file used, so a south-up raster looks vertically
mirrored to the correlator and its `dy` comes back negated. The extension reads the lookup's direction
and corrects for it, so the same scene stored either way yields the same field:

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
                   chip_size = 32, chip_size_max = 32, search_radius = 20, grid_spacing = 16)
(north_up = (mid(out.vx), mid(out.vy)), south_up = (mid(flipped.vx), mid(flipped.vy)),
 y_orders = (order(dims(out, Y)), order(dims(flipped, Y))))
```

Same velocities, opposite storage orders. The package's tests assert this on the full field rather than
on a median, since a field with `vy` inverted still looks like plausible flow and nothing downstream
can detect it.

## Velocity: the `dt` keyword

Without `dt`, `vx`/`vy` are in **pixels**. Give the interval between acquisitions and they become **CRS
units per year** — metres per year for a projected raster:

```@example geo
velocity = autorift(reference, secondary; dt = Day(16), chip_size = 32, chip_size_max = 32,
                    search_radius = 20, grid_spacing = 16)
(pixels = mid(out.vx), m_per_yr = mid(velocity.vx))
```

The conversion is one line, and worth knowing in both directions:

```
metres per year = pixels × pixel_size / (dt in days / 365.25)
```

```@example geo
mid(out.vx) * RES / (16 / 365.25)
```

The same number. The reverse direction is what you need to choose a `search_radius`: a radius has to
cover the fastest motion expected over *this pair's* interval, and a published velocity in m/yr converts
to pixels the same way.

```@example geo
# A 5000 m/yr glacier over a 16-day pair, at 30 m pixels.
5000 * (16 / 365.25) / RES
```

About 7 pixels — so a radius of 20 is generous here, and over a 16-day pair even a fast outlet glacier
needs a modest search. Over a year-long pair the same glacier moves 167 pixels, and the radius, the chip
size, or a prior all have to change. This is the single most common reason a correlation returns nothing:
see [Judging a result](@ref).

`dt` also accepts a `Real` count of years directly, and any `Dates.Period` — `Month(6)`, `Year(1)`.

The quality layers come through unchanged — `correlation`, `peak_ratio`, `chip_size` and
`interpolated` mean exactly what they mean on the array path:

```@example geo
quality_panels((dx = collect(out.vx), dy = collect(out.vy),
                correlation = collect(out.correlation), chip_size = collect(out.chip_size)))
```

## Files on disk, and scenes larger than memory

A real pair comes from files, and a real scene is often larger than memory. `lazy = true` correlates
from the file without materializing it:

```julia
a = Raster("early.tif"; lazy = true)
b = Raster("late.tif"; lazy = true)
out = autorift(a, b; dt = Day(16), grid_spacing = 8, threaded = true)
```

Two things happen automatically. The run is **blocked**, so the filtered `Float32` scene — what an
unblocked run holds resident — is never formed; each block filters its own read window. And **nodata
becomes mask rather than number**: a GDAL raster's `missingval` marks pixels excluded from correlation
rather than being read as a dark measurement, which is what correlating a `-9999` fill would amount to.

The answer is bit-identical to the same pair materialized, at any block size. The memory and the knobs
are [Correlating scenes larger than memory](@ref).

## Masks

Clouds, shadow, water, the edge of a swath — anywhere the imagery is present but not usable. Pass a
boolean raster and those pixels are excluded from every chip and window:

```@example geo
valid = trues(Base.size(reference_array))
valid[200:300, 150:260] .= false   # a cloud, say

masked = autorift(reference, secondary; reference_valid = northup(valid),
                  chip_size = 32, chip_size_max = 32, search_radius = 20, grid_spacing = 16)
(unmasked = count(isfinite, collect(out.vx)), masked = count(isfinite, collect(masked.vx)))
```

```@example geo
field_panels(blank(collect(out.vx), isfinite.(collect(out.vx))),
             blank(collect(masked.vx), isfinite.(collect(masked.vx)));
             titles = ("vx, no mask", "vx, cloud masked"))
```

A point survives if *any* pixel of its chip is valid, so the hole is smaller than you might expect and a
partially covered chip still measures — at a lower correlation. `secondary_valid` is the same for the
other image, and either can be an array or a raster. [Masking invalid pixels](@ref) has the measured
behavior.

## Not yet supported

`dt` scales by the lookup's pixel size, which assumes displacement along a pixel axis is displacement
along a map axis. That is true for any north-up projected raster, so for essentially all optical Landsat
and Sentinel-2 imagery. It is **not** true for a rotated grid or for radar range-Doppler geometry, where
the conversion needs per-pixel geolocation matrices — not implemented. Pass `dt = nothing` there and
convert externally.

Radar geometry still correlates: a `DimArray` whose dimensions are not `X`/`Y` takes the
`DimensionalData` path, returns a `DimStack` of `dx`/`dy` in pixels, and applies no sign or orientation
conversion — correctly, since without a CRS there is no north to point at.

## See also

- [Conventions](@ref) — the sign and the half pixel, and how to settle a convention question
- [Correlating scenes larger than memory](@ref) — blocking, the cache, and the measured peaks
- [Geospatial](@ref) — the extension's full docstring
- [A guided walkthrough](@ref) — the parameters, on plain arrays
