
# Filtering outliers

A correlation returns a displacement at every searched point, including points where the peak was
noise. Those are arbitrary vectors, and one of them can dominate a fit downstream. Real motion is
spatially coherent, so neighboring points move similarly, while a false match agrees with nothing around
it.

[`GardnerFilter`](@ref) tests for this, and `autorift` runs it by default. This page describes what it
does and when to change its settings.

## What it removes

The scene below has one decorrelated patch, where noise added to the secondary image leaves the
correlation nothing to lock onto, in addition to the two speed transitions at the edges of the fast
band:

```@example outliers
using AutoRIFT
include("../../figures.jl")  # plotting helpers and the example scenes; see [Plotting](@ref)
using Statistics: median, quantile

band(row) = clamp((108 - abs(row - 256)) / 28, 0, 1)
reference, clean, truth_dx, _ = warped_pair(photo(), 512, (row, col) -> (2 + 20 * band(row), 0.0))
patch(row, col) = (300 <= row <= 400 && 120 <= col <= 260) ? 1.0 : 0.0
secondary = decorrelate(clean, patch; amplitude = 2.5, seed = 7)

image_panels(reference, secondary)
```

The same call with the filter on and off. The counts include points filled by [Hole filling](@ref):

```@example outliers
settings = (; chip_size = 16, chip_size_max = 64, grid_spacing = 8)
filtered = autorift(reference, secondary; settings...)
raw = autorift(reference, secondary; outliers = :none, settings...)

(measured_raw = AutoRIFT.nmeasured(raw), measured_filtered = AutoRIFT.nmeasured(filtered))
```

```@example outliers
field_panels(blank(raw.dx, measured(raw)), blank(filtered.dx, measured(filtered));
             titles = ("dx, outliers = :none", "dx, GardnerFilter (default)"))
```

The points the filter dropped:

```@example outliers
rejected = measured(raw) .& .!measured(filtered)
input_field(Float64.(rejected), "rejected by the filter"; colormap = :grays)
```

The rejections fall in two places. One is the decorrelated patch, which the filter finds without being
told it exists. The other is the two edges of the fast band, where the speed changes by 20 pixels over
28 rows and a chip straddles two speeds; this is the case described under
[When the filter is the wrong tool](@ref).

## Effect on error

This scene has a known true displacement field, so the error can be computed:

```@example outliers
_, grid = AutoRIFT.autorift_with_grid(reference, secondary; settings...)
truth = reshape([truth_dx[round(Int, grid.y[i]), round(Int, grid.x[i])] for i in eachindex(grid.x)],
                size(filtered.dx))
error_of(out) = filter(isfinite, abs.(out.dx .- truth))

summary(out) = (n = length(error_of(out)),
                median = round(median(error_of(out)); digits = 3),
                p99 = round(quantile(error_of(out), 0.99); digits = 2),
                worst = round(maximum(error_of(out)); digits = 2))
(unfiltered = summary(raw), filtered = summary(filtered))
```

The median barely changes: 0.016 px unfiltered and 0.0 filtered, which is one refinement quantization
step. The filter does not alter the points it keeps. It removes the tail: the 99th percentile falls from
about 38 pixels to 0.06, and the worst point from 47 pixels to 3.6. The filter rejects points; it does
not smooth the field.

## The two stages

Both stages are from autoRIFT's `DISP_FILT`, and a point must pass both.

1. **Agreement.** Keep a point only if at least `min_agree_fraction` of the full `window`×`window`
   neighborhood, counting the point itself, lies within `agree_tolerance` of it. The required count is
   not reduced for truncated windows at the grid border. This removes isolated wild vectors.
2. **Median deviation.** Keep a point only if it lies within `mad_scale` median absolute deviations
   (MAD) of its neighborhood median, with the tolerance floored at two refinement steps. This catches a
   false match sitting among other false matches, where stage 1's neighbors corroborate each other.

Stage 1 runs `outlier_iterations` times and stage 2 one fewer (at least once). A rejected point stays
rejected and no longer counts as an agreeing neighbor, so a rejection can expose its neighbors in the
next pass.

Both stages compare displacements normalized by the local search radius. This is what gives
`agree_tolerance` its meaning: a five-pixel disagreement is negligible where the search reached fifty
pixels and decisive where it reached six, and the radius can vary across a scene. Stage 2 gives the same
result with or without normalization, because the deviation and the MAD scale together. Only its
refinement-step floor depends on the radius.

## The five settings

Each is a keyword to `autorift`, or to [`GardnerFilter`](@ref) if you want an instance.

| keyword | default | raising it |
|---|---:|---|
| `outlier_window` | 5 | wider neighborhood: more evidence per point, and more rejections near any boundary |
| `outlier_iterations` | 3 | more passes of stage 1, letting a rejection expose its neighbors |
| `min_agree_fraction` | 8/25 | more neighbors must agree (stricter) |
| `agree_tolerance` | 0.2 | looser definition of agreement (more permissive) |
| `mad_scale` | 4.0 | wider deviation allowance (more permissive) |

Measured on the scene above, against the same truth field:

```@example outliers
variants = ("default" => GardnerFilter(),
            "agree_tolerance = 0.5" => GardnerFilter(; agree_tolerance = 0.5),
            "mad_scale = 1.5" => GardnerFilter(; mad_scale = 1.5),
            "mad_scale = 10" => GardnerFilter(; mad_scale = 10.0),
            "window = 9" => GardnerFilter(; window = 9),
            "iterations = 1" => GardnerFilter(; iterations = 1))

for (label, f) in variants
    out = autorift(reference, secondary; outliers = f, settings...)
    s = summary(out)
    println(rpad(label, 22), " kept ", lpad(s.n, 4), "   median ", s.median, "   p99 ", s.p99)
end
```

The runs on this page use `chip_size = 16` and `grid_spacing = 8`, so `autorift` rescales the filter
before applying it (see [Grids finer than the chip](@ref)): a window of 5 becomes 9 and the agreement
fraction 8/25 becomes 0.41, and a window of 9 becomes 17. In the table:

- `agree_tolerance` and `mad_scale` trade coverage against the tail. Loosening either keeps 39 to 110
  more points and raises the 99th percentile from 0.06 px to 2.0–3.5 px. Tightening `mad_scale` to 1.5
  removes 23 points and lowers the 99th percentile slightly. Which direction to err depends on whether
  the consumer tolerates a wrong vector or a missing one better.
- `iterations = 1` keeps the most points and has the worst tail (6.3 px). One pass cannot reject an
  outlier whose neighbors are also outliers.
- `window = 9` keeps a quarter fewer points than the default with no improvement in the tail. A wider
  neighborhood requires agreement across more ground, and near the edge of a coherent region that
  agreement is absent.

## Grids finer than the chip

When `grid_spacing` divides `chip_size`, adjacent grid points share most of their imagery, so a wider
window is appropriate. `autorift` calls [`AutoRIFT.rescale`](@ref), which widens the window and raises
the agreement fraction to match. The ratio of `chip_size` to `grid_spacing` used is capped at 2. The
settings you pass are those for one point per chip, and the rescaling is applied on top of them.

## Turning it off

`outliers = :none` keeps every measured point. It is not suitable for production, since the table above
shows what it admits, but it shows which points the correlator failed on and which the filter rejected.
The figures on this page use it for that.

## Hole filling

Rejection leaves gaps, and small gaps are filled from their neighbors. Two keywords control this:

- `fill_window = 3`: the width of the median window a fill is computed over. Must be odd. A point is
  filled if at least `2 * fill_window^2 ÷ 3` of its neighbors are measured (6 of 9 by default).
- `fill_min_hole = 5`: a connected hole smaller than this is also filled, whatever its neighbor count, as
  long as it has at least one measured neighbor. This closes holes whose shape leaves every point short
  of the window criterion. The bound is exclusive, so the default closes holes of one to four points. `0`
  disables it, leaving only the window criterion.

```@example outliers
for (w, h) in ((3, 5), (3, 0), (7, 25))
    out = autorift(reference, secondary; fill_window = w, fill_min_hole = h, settings...)
    println("fill_window = $w, fill_min_hole = $h  →  measured ", AutoRIFT.nmeasured(out),
            ", interpolated ", count(out.interpolated))
end
```

Filled points are marked in the `interpolated` layer. They are not measurements, and their
`correlation` is `NaN`. See [Judging a result](@ref) for whether to keep them. A larger `fill_window` and
`fill_min_hole` (the last row) close more of the ragged edge of the rejected region, at the cost of
interpolating over ground the correlator could not resolve. Filling runs for three passes, each filling
only points that qualify, so a large hole closes from its edge inward.

## When the filter is the wrong tool

The filter assumes the field is spatially coherent at the grid's scale. Where it is not, the filter
rejects real motion:

- A sharp discontinuity (a shear margin, a crack, an object boundary) has neighboring points with
  different true displacements. Expect rejections along it. A finer grid spacing does not help, because
  the discontinuity stays sharp relative to the spacing.
- Turbulent motion with a correlation length at or below the grid spacing cannot be distinguished from
  noise.
- Scattered points: a `PointSet{1}` has no neighborhood, and `autorift` does not apply the filter to it.

For the first two, use `outliers = :none` with a `correlation` threshold, which tests each point alone
and assumes no coherence. It rejects fewer bad points.

## Relation to the normalized median test

The filter resembles the universal outlier detection of Westerweel & Scarano (2005) but differs from it
in ways that change which points survive. The center point is included in both the median and the MAD,
where Westerweel & Scarano exclude it so that the test asks whether a vector agrees with its neighbors
rather than with a set it partly defines. The normalization is by local search radius rather than by
`MAD + ε`, the threshold is 4 rather than their universal 2, and the agreement stage has no counterpart
in their test.

[`OutlierMethod`](@ref) is the extension point for adding another test; see
[`AutoRIFT.reject_outliers`](@ref).

## See also

- [Judging a result](@ref): the per-point threshold, which the filter does not replace
- [`GardnerFilter`](@ref): every keyword and its contract
- [`AutoRIFT.rescale`](@ref): why a fine grid gets a wider window
