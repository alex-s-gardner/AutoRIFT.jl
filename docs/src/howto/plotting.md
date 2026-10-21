
# Plotting a result

A displacement field is a matrix. Plotting one correctly takes care with two things, and neither raises an
error when missed.

## Orientation

`heatmap!` treats an array's first index as the x axis. A displacement field is indexed `[row, column]`,
so passing one directly draws it transposed, with rows running horizontally. A feature elongated along y
appears elongated along x, and a north-up scene appears rotated.

Transposing fixes the axes. Reversing the columns of the transpose then puts row 1 at the top, where the
first row of an image belongs:

```julia
mapshow(A) = reverse(permutedims(A); dims = 2)
```

Every figure in this documentation goes through this function. Applied once:

```@example plotting
using AutoRIFT, CairoMakie
include("../../figures.jl")  # the helpers this page describes

# A pair that moved only along x, so the correct picture has structure in dx and none in dy.
reference, secondary, _, _ = warped_pair(photo(), 512, (row, col) -> (8.0, 0.0))
out = autorift(reference, secondary; chip_size = 32, search_radius = 20, grid_spacing = 8)

fig = Figure(; size = (900, 420))
heatmap!(Axis(fig[1, 1]; title = "dx, raw heatmap!", aspect = DataAspect()), out.dx)
heatmap!(Axis(fig[1, 2]; title = "dx, through mapshow", aspect = DataAspect()), mapshow(out.dx))
fig
```

On a uniform field the two panels look alike. With an anisotropic field the difference is clear. Here
the displacement is larger in a band of rows:

```@example plotting
striped, striped_sec, _, _ =
    warped_pair(photo(), 512, (row, col) -> (2 + 12 * (abs(row - 256) < 60), 0.0))
out2 = autorift(striped, striped_sec; chip_size = 32, search_radius = 20, grid_spacing = 8)

fig = Figure(; size = (900, 420))
heatmap!(Axis(fig[1, 1]; title = "raw: the band runs vertically", aspect = DataAspect()), out2.dx)
heatmap!(Axis(fig[1, 2]; title = "mapshow: horizontally, as built", aspect = DataAspect()),
         mapshow(out2.dx))
fig
```

The band covers a range of rows, so it should be horizontal. Only the second panel shows it that way.

## Unmeasured points

Not every grid point has a measurement. `dx`, `dy` and `correlation` are `NaN` where nothing was
measured, and most colormaps draw `NaN` in a visible color. A zero displacement is a real measurement and
must look different from a missing one. Two helpers convert explicitly:

```julia
blank(A, keep) = map((v, k) -> k ? Float64(v) : NaN, A, keep)
measured(out) = .!isnan.(out.dx)
```

`chip_size` is an integer layer that uses `0` for "no level answered". It has to be blanked too, or it
plots as a chip size of zero.

```@example plotting
# A decorrelated band, so that more than one chip size answers.
banded = decorrelate(secondary, (row, col) -> abs(row - 256) < 80 ? 1.0 : 0.0; amplitude = 0.2,
                     seed = 3)
partial = autorift(reference, banded; chip_size = 16, chip_size_max = 64, grid_spacing = 8)

fig = Figure(; size = (900, 420))
hm1 = heatmap!(Axis(fig[1, 1]; title = "chip_size, raw", aspect = DataAspect()),
               mapshow(Float64.(partial.chip_size)); colormap = :viridis)
Colorbar(fig[1, 2], hm1)
hm2 = heatmap!(Axis(fig[1, 3]; title = "chip_size, 0 blanked", aspect = DataAspect()),
               mapshow(blank(partial.chip_size, partial.chip_size .> 0)); colormap = :viridis)
Colorbar(fig[1, 4], hm2)
fig
```

The left panel's color scale starts at zero, so the gaps take up most of its range. The right panel's scale
covers only the chip sizes that were used.

## Color scales

Each kind of layer uses a different convention.

- **Displacement** is signed. Use a diverging colormap with symmetric limits taken from a high percentile
  of `abs` rather than the extremes, since one outlier would otherwise set the scale. Share one scale
  across all panels drawn together, so differences in sign and magnitude show.
- **Correlation** is normalized. Fix the range at `(0, 1)`. A fitted scale makes a weak field look as
  strong as a good one.
- **Chip size** takes a few values, each `chip_size_min` times a power of two. A sequential colormap works,
  provided the zeros are blanked.

`displacement_panels` and `quality_panels` apply these:

```@example plotting
displacement_panels(out)
```

```@example plotting
quality_panels(out)
```

## Geospatial output

Given `Raster` input, `autorift` returns a `RasterStack`. Makie's `heatmap` has a Rasters recipe that
reads the coordinates from the dimensions:

```julia
using Rasters, CairoMakie
out = autorift(reference_raster, secondary_raster; dt = Day(16))
heatmap(out.vx; colormap = :balance)
```

`mapshow` is not needed, because the recipe takes the orientation from the lookups. A projected axis also
makes a sign error visible, which an array index does not. See [Conventions](@ref).

## The helpers used here

`docs/figures.jl` in this repository defines these helpers, and every documentation page includes it. It
is not part of the package API. Copy what you need:

| helper | what it does |
|---|---|
| `mapshow(A)` | returns a field in image orientation |
| `blank(A, keep)` | returns `A` as `Float64` with `NaN` where `keep` is false |
| `measured(out)` | marks the points that carry a displacement |
| `image_panels(images...; titles, shared)` | draws input images on a shared grayscale, or one scale per panel |
| `displacement_panels(out)` | draws `dx` and `dy` on one symmetric scale |
| `quality_panels(out)` | draws `correlation` on `(0, 1)` and the chip size |
| `field_panels(fields...; titles)` | draws several displacement-like fields on one symmetric scale |
| `input_field(A, label)` | draws one per-point argument, such as a chip size, radius or prior |
| `surface_panels(chip, window, surface)` | draws a chip, its search window and the correlation surface with the peak marked |
| `mask_panels(masks...; titles)` | draws boolean per-point masks, each title with its point count |
| `warped_pair(source, n, offset)` | returns a reference, a warped secondary and the true `dx` and `dy` |
| `photo()` | returns the example photograph |
| `decorrelate(img, weight; amplitude, seed)` | adds noise scaled by `weight(row, col)` |

## See also

- [Conventions](@ref): the sign of `dy`, and why a projected plot checks it
- [A guided walkthrough](@ref): these helpers in use
