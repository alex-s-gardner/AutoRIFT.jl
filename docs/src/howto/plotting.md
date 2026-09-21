
# Plotting a result

A displacement field is a matrix, and a matrix drawn carelessly is drawn wrong. Two things have to be
handled, and neither produces an error when it is missed.

## Orientation

`heatmap!` treats an array's **first** index as the x axis. A displacement field is indexed
`[row, column]`, so passing one directly draws it transposed — rows run horizontally. A feature
elongated along y appears elongated along x, which inverts the reading of any anisotropy in the field,
and a north-up scene appears rotated.

Transposing fixes the axes. Reversing the resulting columns then puts row 1 at the top, which is where
the first row of an image belongs:

```julia
mapshow(A) = reverse(permutedims(A); dims = 2)
```

That is the whole fix, and every figure in this documentation goes through it. Applied once:

```@example plotting
using AutoRIFT, CairoMakie
include("../../figures.jl")  # the helpers this page describes

# A pair that moved only along x, so the correct picture has structure in dx and none in dy.
reference, secondary, _, _ = warped_pair(512, (row, col) -> (8.0, 0.0); seed = 2)
out = autorift(reference, secondary; chip_size = 32, search_radius = 20, grid_spacing = 8)

fig = Figure(; size = (900, 420))
heatmap!(Axis(fig[1, 1]; title = "dx, raw heatmap!", aspect = DataAspect()), out.dx)
heatmap!(Axis(fig[1, 2]; title = "dx, through mapshow", aspect = DataAspect()), mapshow(out.dx))
fig
```

On a field this uniform the two look similar, which is the danger. Make the field anisotropic and the
difference is unmistakable:

```@example plotting
striped, striped_sec, _, _ =
    warped_pair(512, (row, col) -> (2 + 12 * (abs(row - 256) < 60), 0.0); seed = 2)
out2 = autorift(striped, striped_sec; chip_size = 32, search_radius = 20, grid_spacing = 8)

fig = Figure(; size = (900, 420))
heatmap!(Axis(fig[1, 1]; title = "raw: the band runs vertically", aspect = DataAspect()), out2.dx)
heatmap!(Axis(fig[1, 2]; title = "mapshow: horizontally, as built", aspect = DataAspect()),
         mapshow(out2.dx))
fig
```

The band was built across rows. Only the second panel shows it that way.

## Unmeasured points

Not every grid point gets an answer. `dx` and `dy` are `NaN` where nothing was measured, and
`correlation` likewise — which most colormaps render as a dead color rather than as nothing. Points
that a level did resolve but that hold a genuine zero displacement must not look the same as points
that failed, because zero is a real measurement.

Convert explicitly:

```julia
blank(A, keep) = map((v, k) -> k ? Float64(v) : NaN, A, keep)
measured(out) = .!isnan.(out.dx)
```

`chip_size` needs its own treatment: it is an integer layer using `0` for "no level answered", which
would otherwise plot as a chip size of zero.

```@example plotting
partial = autorift(reference, secondary; chip_size = 16, chip_size_max = 64, grid_spacing = 8)

fig = Figure(; size = (900, 420))
hm1 = heatmap!(Axis(fig[1, 1]; title = "chip_size, raw", aspect = DataAspect()),
               mapshow(Float64.(partial.chip_size)); colormap = :viridis)
Colorbar(fig[1, 2], hm1)
hm2 = heatmap!(Axis(fig[1, 3]; title = "chip_size, 0 blanked", aspect = DataAspect()),
               mapshow(blank(partial.chip_size, partial.chip_size .> 0)); colormap = :viridis)
Colorbar(fig[1, 4], hm2)
fig
```

The left panel's color scale starts at zero and spends most of its range on the gaps. The right one
spends all of it on the three real chip sizes.

## Color scales

Three conventions, each for a different kind of layer.

**Displacement is diverging and signed.** Use a diverging colormap with **symmetric** limits, and take
those limits from a high percentile of `abs` rather than from the extremes — one outlier otherwise sets
the scale for the whole figure. Share one scale across every panel drawn together, so a difference in
sign or magnitude is visible rather than absorbed by per-panel scaling.

**Correlation is already normalized.** Fix the range at `(0, 1)`. A fitted scale makes a weak field look
as strong as a good one, which defeats the purpose of plotting it.

**Chip size is categorical.** It takes a handful of values, all powers of two. A sequential colormap
reads correctly; what matters is blanking the zeros.

```@example plotting
displacement_panels(out)
```

```@example plotting
quality_panels(out)
```

## Geospatial output

Given `Raster` input, `autorift` returns a `RasterStack`, and Makie's `heatmap` has a Rasters recipe
that reads the coordinates from the dimensions:

```julia
using Rasters, CairoMakie
out = autorift(reference_raster, secondary_raster; dt = Day(16))
heatmap(out.vx; colormap = :balance)
```

No `mapshow` there — the recipe handles orientation, because the lookups tell it which way y runs. That
is also what makes it the place to check a sign convention by eye: a projected axis makes an error
visible in a way an array index does not. See [Conventions](@ref).

## The helpers used here

Everything above is packaged in `docs/figures.jl` in this repository, which every documentation page
includes. It is not part of the API — copy what you need:

| helper | what it draws |
|---|---|
| `mapshow(A)` | one field in image orientation |
| `blank(A, keep)` | `NaN` where `keep` is false |
| `measured(out)` | which points carry a displacement |
| `image_panels(images...)` | input images on a shared grayscale |
| `displacement_panels(out)` | `dx` and `dy` on one symmetric scale |
| `quality_panels(out)` | `correlation` at `(0, 1)`, and chip size |
| `field_panels(fields...; titles)` | several displacement-like fields on one scale |
| `input_field(A, label)` | a single per-point argument — a chip size, radius or prior |
| `surface_panels(chip, window, surface)` | one correlation, with its peak marked |

## See also

- [Conventions](@ref) — the sign of `dy`, and why a projected plot is the check for it
- [A guided walkthrough](@ref) — these helpers in use on every step
