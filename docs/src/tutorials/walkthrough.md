# A guided walkthrough

Eight steps over one scene. Each adds a single capability to the call before it, and shows what that
argument changed. Stop wherever the result is good enough — every step is a complete, working call.

The first five steps are keywords. The last three pass a
[`AutoRIFT.PointSet`](@ref AutoRIFT.PointSet), which is how a setting that *varies across the scene*
arrives, since a per-point field cannot be a scalar keyword.

## The scene

Two images of the same surface, taken at different times. A band across the middle moved 22 pixels
to the left between them; everything else moved 2. The band also **decorrelates** — its surface
changes between the two images, not just its position — which is what a real moving surface does and
what makes a chip size worth choosing.

```@example walkthrough
using AutoRIFT
include("../../figures.jl")  # plotting helpers and the synthetic scenes; see [Plotting](@ref)

# Fraction of full speed at image row `row`: 1 inside the band, tapering to 0 outside it.
band(row) = clamp((108 - abs(row - 256)) / 28, 0, 1)

reference, clean, truth_dx, truth_dy =
    warped_pair(512, (row, col) -> (2 + 20 * band(row), 0.0); seed = 11)

secondary = decorrelate(clean, (row, col) -> band(row); amplitude = 0.12, seed = 3)

image_panels(reference, secondary)
```

The displacement `autorift` should report:

```@example walkthrough
field_panels(truth_dx, truth_dy; titles = ("true dx (px)", "true dy (px)"))
```

Displacement is in pixels, `dx` along the second index and `dy` along the first, positive downward.
It is the offset from the secondary image back to the reference, so its sign is the opposite of the
surface's motion — a band that moved left reports a positive `dx`. [Conventions](@ref) is the page
for that, and for the y direction under a map projection.

The figures are drawn in image orientation — row 1 at the top — which takes a transposition; see
[Plotting](@ref) for the one function that does it and why a raw `heatmap!` is wrong.

## Step 1 — two arrays

Nothing but the images.

```@example walkthrough
out = autorift(reference, secondary)

displacement_panels(out)
```

The band is recovered. Defaults did the rest: chips of 32, 64 and 128 pixels, a search radius of 25,
and an output point every 32 pixels — which on a 512-pixel image is a 14 × 14 grid, coarse enough
that the figure shows individual points. Every later step measures every 8 pixels so the result reads
as a field:

```@example walkthrough
out = autorift(reference, secondary; grid_spacing = 8)

displacement_panels(out)
```

`out.correlation` says how well each point matched, and `out.chip_size` which chip answered there:

```@example walkthrough
quality_panels(out)
```

Gaps are points with no answer, and they come from two places. The image border is one: a chip and its
search window must fit inside the image. The strips along the band edges are the other, and they are
the more instructive — a chip there spans the transition between two speeds, so no single displacement
describes what it contains. Their position follows the chip size, which is the tell: a larger chip
starts failing further from the edge. A gap is drawn blank rather
than as zero, because zero displacement is a real measurement.

## Step 2 — chip size and search radius

A chip is the patch of the reference image matched into the secondary; the search radius is how far
it may move. Both in pixels.

```@example walkthrough
coarse = autorift(reference, secondary; chip_size = 64, search_radius = 25, grid_spacing = 8)
fine = autorift(reference, secondary; chip_size = 16, search_radius = 25, grid_spacing = 8)

field_panels(blank(fine.dx, measured(fine)), blank(coarse.dx, measured(coarse));
             titles = ("chip 16", "chip 64"))
```

Small chips resolve the band edge and are noisier; large chips are smooth and blur it. A chip must
contain enough texture to be distinguishable from its surroundings, and must be small enough that the
displacement is roughly uniform across it — those pull in opposite directions, which is why the
default searches several sizes rather than one.

The radius must cover the motion: everything beyond it is unreachable. It is also the expensive
knob — cost grows with the area searched, not its width.

## Step 3 — output spacing

`grid_spacing` sets how often a displacement is measured, and changes nothing about how it is
measured.

```@example walkthrough
sparse = autorift(reference, secondary; grid_spacing = 64)
dense = autorift(reference, secondary; grid_spacing = 8)

field_panels(blank(sparse.dx, measured(sparse)), blank(dense.dx, measured(dense));
             titles = ("spacing 64", "spacing 8"))
```

Same field, sampled twice. Neighbouring points at spacing 8 overlap heavily — their chips are 32
pixels wide — so a fine grid costs proportionally more without adding independent information. It
still reads better as a figure, which is why every step here uses 8.

## Step 4 — a range of chip sizes

`chip_size` and `chip_size_max` together define the sizes to try. Every point is attempted at the
finest, and coarser ones fill in where that failed.

```@example walkthrough
out = autorift(reference, secondary; chip_size = 16, chip_size_max = 64, grid_spacing = 8)

quality_panels(out)
```

The chip-size map is the most informative figure here: 16 pixels answers the well-correlated
surroundings, and the decorrelated band needs 32 or 64. Neither uniform choice does both — the fine
chip fails in the band and the coarse chip blurs the edge everywhere else.

This is not an image pyramid. Every size correlates the images at full resolution; only the chip
changes. See [Multiple chip sizes](@ref) for how the levels combine.

## Step 5 — a prior

Motion larger than the search radius is not found. The fix is to tell the search where to look, not
to widen it.

```@example walkthrough
narrow = autorift(reference, secondary; search_radius = 10, grid_spacing = 8)
guided = autorift(reference, secondary; search_radius = 10, dx_prior = 22, grid_spacing = 8)

field_panels(blank(narrow.dx, measured(narrow)), blank(guided.dx, measured(guided));
             titles = ("radius 10", "radius 10, dx_prior 22"))
```

At a radius of 10 the band is out of reach and comes back empty. A prior of 22 offsets the search
window, and the same radius now measures the band at 22 pixels — the search measures the *departure*
from the prior, so a good guess makes a small radius sufficient.

Where a prior comes from, when the pair is all you have: [First guess](@ref).

## Step 6 — a chip size that varies

The first three steps' settings were scalars. A setting that differs from point to point is passed as
an array, on a [`AutoRIFT.PointSet`](@ref AutoRIFT.PointSet) — the search points themselves, one
entry per point per field.

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

The 64-pixel level no longer runs outside the band. Note what the bound does *not* do: the finest
chip is attempted at every point regardless, so these bounds restrict the coarser levels only — see
[`AutoRIFT.PointSet`](@ref AutoRIFT.PointSet).

Any field may stay scalar while another varies. `search_radius_x = 24` above is one number for every
point, beside a per-point `chip_size_max_x`; mixing the two is ordinary.

## Step 7 — a search radius that varies

The radius is the expensive setting, so spending it only where the motion is large is the difference
between a run that finishes and one that does not.

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

[`AutoRIFT.nsearchable`](@ref AutoRIFT.nsearchable) counts the points that will actually be
correlated, which is what the run costs:

```@example walkthrough
uniform = AutoRIFT.pointset(grid.x, grid.y; search_radius_x = 28, search_radius_y = 28)
(uniform = AutoRIFT.nsearchable(uniform), banded = AutoRIFT.nsearchable(banded))
```

The radius need not be isotropic. A surface moving mostly along x wants a wide `search_radius_x` and
a narrow `search_radius_y`; the window follows, and so does the cost.

## Step 8 — a prior that varies

Step 5's prior was one number, which only works when the whole scene moves together. A per-point
prior handles a scene where it does not.

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
got wrong. The residual is what was actually searched for, and it is small everywhere — which is the
whole argument for a prior: accuracy comes from the correlation, reach comes from the prior, and cost
comes from the radius alone.

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

The three points at row 256 sit in the band and return about 22; the point at row 100 is outside it
and returns about 2. Use this when the points of interest are known and sparse — the multi-chip-size
search needs a layout, so a scattered set runs a single scale.

## Next

- [Geospatial data](@ref) — the same call on rasters, returning velocity in map units
- [Conventions](@ref) — the sign of `dy`, and where the half pixel goes
- [Filtering outliers](@ref) — what to do with the points that are wrong
- [Correlating scenes larger than memory](@ref) — when the scene does not fit
