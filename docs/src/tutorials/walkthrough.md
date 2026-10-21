# A guided walkthrough

Eight steps over one scene, each adding one argument to the call. Every step is a complete, working
call, so stop wherever the result is good enough.

The first five steps pass keywords. The last three pass a
[`AutoRIFT.PointSet`](@ref AutoRIFT.PointSet), because a setting that varies across the scene is a
per-point array and cannot be a scalar keyword.

## The scene

Two images of the same scene, the second warped from the first so the answer is known. A band across
the middle (Ali's waist and the ropes behind him) moved 22 pixels to the left between them;
everything else moved 2. Noise is added to the second image within the band, so the band's texture
differs between the images as well as its position. A moving surface decorrelates in the same way,
and it is why the chip size matters.

```@example walkthrough
using AutoRIFT
include("../../figures.jl")  # plotting helpers and the example scenes; see [Plotting](@ref)

# Fraction of full speed at image row `row`: 1 inside the band, tapering to 0 outside it.
band(row) = clamp((108 - abs(row - 256)) / 28, 0, 1)

reference, clean, truth_dx, truth_dy =
    warped_pair(photo(), 512, (row, col) -> (2 + 20 * band(row), 0.0))

secondary = decorrelate(clean, (row, col) -> band(row); amplitude = 0.12, seed = 3)

image_panels(reference, secondary)
```

The displacement `autorift` should report:

```@example walkthrough
field_panels(truth_dx, truth_dy; titles = ("true dx (px)", "true dy (px)"))
```

Displacement is in pixels, `dx` along the second index and `dy` along the first, positive downward.
It is the offset from the secondary image back to the reference, so its sign is opposite to the
surface's motion: a band that moved left reports a positive `dx`. [Conventions](@ref) covers this and
the y direction under a map projection.

The figures are drawn in image orientation, with row 1 at the top. [Plotting](@ref) explains the
transposition this takes and why a raw `heatmap!` is wrong.

## Step 1 — two arrays

Nothing but the images.

```@example walkthrough
out = autorift(reference, secondary)

displacement_panels(out)
```

The band is recovered. The defaults are chips of 32, 64 and 128 pixels, a search radius of 25, and an
output point every 32 pixels. On a 512-pixel image that is an 11 × 11 grid, coarse enough that the
figure shows individual points. Every later step measures every 8 pixels so the result reads as a
field:

```@example walkthrough
out = autorift(reference, secondary; grid_spacing = 8)

displacement_panels(out)
```

`out.correlation` says how well each point matched, and `out.chip_size` which chip answered there:

```@example walkthrough
quality_panels(out)
```

Gaps are points with no answer, and they have two sources. One is the image border: a chip and its
search window must fit inside the image. The other is the strips along the band edges, where the
displacement changes from 2 to 22 pixels over 28 rows and the surface is partly decorrelated. A
larger chip answers more of these points but averages the two speeds ([Multiple chip sizes](@ref)). A
gap is drawn blank rather than as zero, because zero displacement is a real measurement.

## Step 2 — chip size and search radius

A chip is the patch of the reference image matched into the secondary; the search radius is how far
it may move. Both are in pixels.

```@example walkthrough
coarse = autorift(reference, secondary; chip_size = 64, search_radius = 25, grid_spacing = 8)
fine = autorift(reference, secondary; chip_size = 16, search_radius = 25, grid_spacing = 8)

field_panels(blank(fine.dx, measured(fine)), blank(coarse.dx, measured(coarse));
             titles = ("chip 16", "chip 64"))
```

`chip_size` is the finest chip tried and `chip_size_max` defaults to four times it, so the first run
tries chips of 16, 32 and 64 pixels and the second 64, 128 and 256. The grid margin is set by the
largest chip and the search radius, so the second run's grid spans only rows 155 to 358, in or near the
band, and nearly all of its answers are near 22.

A chip averages the displacement over its area, so a large chip blurs a sharp boundary. A chip also
needs enough texture to be distinguishable from its surroundings, which a small chip may lack. These
pull in opposite directions, which is why the default searches several sizes rather than one.

The radius must cover the motion; anything beyond it is unreachable. It is also the expensive setting,
because cost grows with the area searched.

## Step 3 — output spacing

`grid_spacing` sets how often a displacement is measured.

```@example walkthrough
sparse = autorift(reference, secondary; grid_spacing = 32)
dense = autorift(reference, secondary; grid_spacing = 8)

field_panels(blank(sparse.dx, measured(sparse)), blank(dense.dx, measured(dense));
             titles = ("spacing 32", "spacing 8"))
```

Both panels sample the same field and agree except along the band edges, where the two grids land on
different positions. Neighboring points at spacing 8 overlap heavily, since their chips are at least
32 pixels wide, so a finer grid costs proportionally more without adding independent information.
Every later step uses 8 because it reads better as a figure.

## Step 4 — a range of chip sizes

`chip_size` and `chip_size_max` together define the sizes to try. Every point is attempted at the
finest, and coarser sizes fill in where that failed.

```@example walkthrough
out = autorift(reference, secondary; chip_size = 16, chip_size_max = 64, grid_spacing = 8)

quality_panels(out)
```

The chip-size map shows where each size was needed. The 16-pixel chip answers the surroundings and
part of the band. The 32-pixel chip answers band points the 16-pixel chip missed, and the 64-pixel chip
answers one small patch. A 16-pixel chip alone leaves much of the band unanswered, and a coarse chip
used everywhere blurs the band edge.

This is not an image pyramid. Every size correlates the images at full resolution; only the chip
changes. See [Multiple chip sizes](@ref) for how the levels combine.

## Step 5 — a prior

Motion larger than the search radius is not found. A prior offsets the search window so that it covers
the motion without widening the radius.

```@example walkthrough
narrow = autorift(reference, secondary; search_radius = 10, grid_spacing = 8)
guided = autorift(reference, secondary; search_radius = 10, dx_prior = 22, grid_spacing = 8)

field_panels(blank(narrow.dx, measured(narrow)), blank(guided.dx, measured(guided));
             titles = ("radius 10", "radius 10, dx_prior 22"))
```

At a radius of 10 the band is out of reach and comes back empty. With a prior of 22 the same radius
measures the band at 22 pixels, because the search finds the departure from the prior. The prior is
one number for the whole scene, so the surroundings, which move 2, are now out of reach. Step 8 fixes
that with a per-point prior.

Where a prior comes from, when the pair is all you have: [First guess](@ref).

## Step 6 — a chip size that varies

A setting that differs from point to point is passed as an array, on a
[`AutoRIFT.PointSet`](@ref AutoRIFT.PointSet), which holds the search points with one entry per point
per field.

Build the grid, then attach the field:

```@example walkthrough
grid = AutoRIFT.gridpoints(size(reference), 8; chip_size = 64, search_radius = 24)

# Coarse chips only where the surface decorrelates. Elsewhere the finest chip already answers, and a
# coarse level there is wasted work that also blurs the edge.
in_band = band.(grid.y) .> 0.5
bounded = AutoRIFT.pointset(grid.x, grid.y; search_radius_x = 24, search_radius_y = 24,
                            chip_size_max_x = ifelse.(in_band, 0, 16))

input_field(bounded.chip_size_max_x, "chip_size_max_x (px, 0 = unbounded)")
```

```@example walkthrough
out = autorift(reference, secondary, bounded;
               chip_size = 16, chip_size_max = 64, grid_spacing = 8)

quality_panels(out)
```

The bound limits every point outside the band to the 16-pixel level. Inside the band the 32- and
64-pixel levels may run, and the map shows answers at 16 and 32 only. The finest chip is attempted at
every point regardless of the bound, so these bounds restrict the coarser levels only. See
[`AutoRIFT.PointSet`](@ref AutoRIFT.PointSet).

A field may stay scalar while another varies. `search_radius_x = 24` above is one number for every
point, beside a per-point `chip_size_max_x`.

## Step 7 — a search radius that varies

Cost grows with the searched area, so use a wide radius only where the motion is large.

```@example walkthrough
# Wide enough for the band, zero outside it. A zero radius skips the point entirely.
radius = ifelse.(in_band, 28, 0)
banded = AutoRIFT.pointset(grid.x, grid.y; search_radius_x = radius, search_radius_y = radius)

input_field(banded.radius_x, "search_radius_x (px)")
```

```@example walkthrough
out = autorift(reference, secondary, banded; chip_size = 32, grid_spacing = 8)

displacement_panels(out)
```

Points outside the band are not searched, so the figure is blank there.
[`AutoRIFT.nsearchable`](@ref AutoRIFT.nsearchable) counts the points with a nonzero radius in both
axes, which sets the cost of the run. Here 1150 of the 2500 grid points are searched:

```@example walkthrough
uniform = AutoRIFT.pointset(grid.x, grid.y; search_radius_x = 28, search_radius_y = 28)
(uniform = AutoRIFT.nsearchable(uniform), banded = AutoRIFT.nsearchable(banded))
```

The radius need not be the same in both axes. A surface moving mostly along x wants a wide
`search_radius_x` and a narrow `search_radius_y`, and the window and the cost follow.

## Step 8 — a prior that varies

Step 5's single prior suits a scene that moves together. A per-point prior suits a scene where the
motion varies.

```@example walkthrough
prior = 2 .+ 20 .* band.(grid.y)
guided = AutoRIFT.pointset(grid.x, grid.y; search_radius_x = 6, search_radius_y = 6,
                           dx_prior = prior, dy_prior = 0)

input_field(guided.dx_prior, "dx_prior (px)")
```

```@example walkthrough
out = autorift(reference, secondary, guided; chip_size = 32, grid_spacing = 8)

field_panels(blank(out.dx, measured(out)), blank(out.dx .- prior, measured(out));
             titles = ("dx (px)", "dx − prior (px)"))
```

A radius of 6 measures a 22-pixel displacement, because the search only has to find what the prior
got wrong. The residual `dx − prior` is near zero except at a few points along the band edges, where
it reaches several pixels. The prior sets the reach of the search, the radius sets its cost, and the
correlation sets the accuracy.

## Points that are not a grid

A [`AutoRIFT.PointSet`](@ref AutoRIFT.PointSet) built from vectors rather than matrices is a set of
scattered points, correlated at one chip size with no grid structure.

```@example walkthrough
xs = [120.0, 200.0, 256.0, 256.0, 300.0, 400.0]
ys = [256.0, 256.0, 100.0, 256.0, 256.0, 256.0]
pts = AutoRIFT.pointset(xs, ys; chip_size_x = 32, chip_size_y = 32,
                        search_radius_x = 28, search_radius_y = 28)

out = autorift(reference, secondary, pts)
round.(out.dx; digits = 2)
```

The five points at row 256 sit in the band and return about 22; the point at row 100 is outside it
and returns about 2. Use this when the points of interest are known and sparse. The multi-chip-size
search needs a grid layout, so a scattered set runs a single chip size.

## Next

- [Geospatial data](@ref) — the same call on rasters, returning velocity in map units
- [Conventions](@ref) — the sign of `dy`, and where the half pixel goes
- [Filtering outliers](@ref) — what to do with the points that are wrong
- [Correlating scenes larger than memory](@ref) — when the scene does not fit
