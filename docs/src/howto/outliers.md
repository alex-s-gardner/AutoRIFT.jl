
# Filtering outliers

A correlation returns a displacement at every searched point, including the points where the peak was
noise. Those are not small errors — they are arbitrary vectors, and one of them can dominate any fit
downstream. What separates them from real motion is that real motion is spatially coherent: neighbouring
points move similarly, while a false match agrees with nothing around it.

[`GardnerFilter`](@ref) asks that question, and `autorift` runs it by default. This page is about what it
does and when to change its settings.

## What it removes

A scene with one badly decorrelated patch — texture in one image that is simply not in the other, so the
correlation there has nothing to lock onto:

```@example outliers
using AutoRIFT
include("../../figures.jl")  # plotting helpers and the synthetic scenes; see [Plotting](@ref)
using Statistics: median, quantile

band(row) = clamp((108 - abs(row - 256)) / 28, 0, 1)
reference, clean, truth_dx, _ = warped_pair(512, (row, col) -> (2 + 20 * band(row), 0.0); seed = 11)
patch(row, col) = (300 <= row <= 400 && 120 <= col <= 260) ? 1.0 : 0.0
secondary = decorrelate(clean, patch; amplitude = 2.5, seed = 7)

image_panels(reference, secondary)
```

The same call with the filter on and off:

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

And the rejection mask itself — which points the filter dropped:

```@example outliers
rejected = measured(raw) .& .!measured(filtered)
input_field(Float64.(rejected), "rejected by the filter"; colormap = :grays)
```

The rejections concentrate in the decorrelated patch, which is the point: the filter found a region where
the displacements disagree with their neighbours, without being told the region existed.

## How much it is worth

This scene has a known truth field, so the improvement is measurable rather than visual:

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

The **median is identical** — 0.016 px either way, one refinement quantization step. The filter changes
nothing about the points that were already right. What it changes is the tail: the 99th percentile falls
from about 40 pixels to 0.06, and the worst point from 47 pixels to 2.

That shape is worth internalizing. A filter that improved the median would be smoothing the field, which
is a different operation with different costs. This one removes points; the ones it keeps are untouched.

## The two stages

Both are from autoRIFT's `DISP_FILT`, and a point must pass both.

1. **Agreement.** Keep a point only if at least `min_agree_fraction` of its `window`×`window`
   neighbourhood lies within `agree_tolerance` of it. Cheap, and removes isolated wild vectors.
2. **Median deviation.** Keep a point only if it lies within `mad_scale` median absolute deviations of its
   neighbourhood median. Catches the subtler case of a false match sitting among other false matches,
   where stage 1's neighbours corroborate each other.

Both stages compare displacements **normalized by the local search radius**, and that is what gives
`agree_tolerance` its meaning: a five-pixel disagreement is negligible where the search reached fifty
pixels and decisive where it reached six. Since the radius can vary across a scene, a fixed absolute
tolerance would mean different things in different places.

Stage 2 is radius-*invariant* by construction — its tolerance is `mad_scale` times the neighbourhood MAD,
and both the deviation and the MAD scale as one over the radius, so the ratio does not move. Normalizing
there buys consistency of units rather than of behaviour.

The two stages also differ in whether a rejection is final. Stage 1 recomputes its mask from scratch each
iteration, so a point rejected in one pass can return in the next once its neighbours' own rejections
change the counts. Stage 2 intersects with the mask it inherits, so a rejection there sticks.

## The five knobs

Every setting is a keyword to `autorift` directly, or to [`GardnerFilter`](@ref) if you want an instance.

| keyword | default | raising it |
|---|---:|---|
| `outlier_window` | 5 | wider neighbourhood — more evidence per point, and more rejections near any boundary |
| `outlier_iterations` | 3 | more passes of stage 1, letting a rejection expose its neighbours |
| `min_agree_fraction` | 8/25 | more neighbours must agree — stricter |
| `agree_tolerance` | 0.2 | looser definition of agreement — more permissive |
| `mad_scale` | 4.0 | wider deviation allowance — more permissive |

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

Three things to read out of that:

- **`agree_tolerance` and `mad_scale` trade coverage against the tail**, and the trade is not symmetric.
  Loosening either keeps a few dozen more points and costs an order of magnitude in the 99th percentile;
  tightening `mad_scale` to 1.5 halves the tail again for about 20 points. Which side to err on depends
  on whether your consumer can tolerate a wrong vector or a missing one.
- **`iterations = 1` is the worst setting here** — it keeps the most points and has by far the worst tail.
  Stage 1's iteration is doing real work: one pass cannot see an outlier whose neighbours are also
  outliers.
- **`window = 9` costs a quarter of the coverage** for no gain in the tail. A wider neighbourhood demands
  agreement across more ground, and near the edge of any coherent region that agreement is not there.

The exception to that last point is when grid spacing is finer than the chip size, where a wider window is
*correct* rather than merely strict — adjacent points then share most of their imagery. `autorift`
already handles that: it calls [`AutoRIFT.rescale`](@ref) to widen the window and raise the agreement
fraction to match. Whatever you pass is the setting for one point per chip, and the scaling is applied on
top.

## Turning it off

`outliers = :none` keeps everything. It is not a production setting — the table above is what it costs —
but it is the right tool for deciding *which* stage dropped a point, since it separates the correlator's
failures from the filter's rejections. That is what the figures on this page use it for.

## Hole filling

Rejection leaves gaps, and small gaps are filled from their neighbours rather than left empty. Two
keywords control it:

- `fill_window = 3` — the median window a fill is computed over. Must be odd.
- `fill_min_hole = 5` — a connected hole smaller than this is filled whatever its neighbour count, which
  closes a hole whose shape leaves every point short of the window criterion. Exclusive, so the default
  closes holes of one to four points. `0` disables it and leaves filling to the window criterion alone.

```@example outliers
for (w, h) in ((3, 5), (3, 0), (7, 25))
    out = autorift(reference, secondary; fill_window = w, fill_min_hole = h, settings...)
    println("fill_window = $w, fill_min_hole = $h  →  measured ", AutoRIFT.nmeasured(out),
            ", interpolated ", count(out.interpolated))
end
```

Filled points are marked in the `interpolated` layer and are not measurements — their `correlation` is
`NaN`, having no surface of their own. See [Judging a result](@ref) for whether to keep them.

Raising `fill_min_hole` to 25 closes the ragged edges of the rejected region, which is a visual
improvement and an epistemic loss: those points are now interpolated across a region the correlator
could not resolve. The default is small deliberately.

## When the filter is the wrong tool

The filter's premise is that the field is spatially coherent at the grid's scale. Where it genuinely is
not, the filter rejects real motion:

- **A sharp discontinuity** — a shear margin, a crack, an object boundary — has neighbouring points with
  legitimately different displacements. Expect rejections along it, and a finer grid spacing does not help
  because the discontinuity stays sharp relative to the spacing.
- **Genuinely turbulent motion**, where the correlation length is at or below the grid spacing. The filter
  has no way to distinguish that from noise, because at that scale there is no difference to detect.
- **Scattered points** rather than a grid. A `PointSet{1}` has no neighbourhood at all, so there is
  nothing for the filter to work with.

In each case the answer is `outliers = :none` plus a `correlation` gate, which is a per-point test and
makes no coherence assumption. It catches less, and what it catches it catches honestly.

## Relation to the normalized median test

If you come from PIV, this resembles the universal outlier detection of Westerweel & Scarano (2005) but is
not it, and the differences change which points survive. The centre point is included in both the median
and the MAD here, where Westerweel & Scarano exclude it — excluding it is what makes their test ask
whether a vector agrees with its neighbours rather than with a set it partly defines. The normalization is
by local search radius rather than by `MAD + ε`; the threshold is 4 rather than their universal 2; and the
agreement pre-pass has no counterpart there at all.

`OutlierMethod` is the extension point if you want the other test — see
[`AutoRIFT.reject_outliers`](@ref).

## See also

- [Judging a result](@ref) — the per-point gate, which the filter does not replace
- [`GardnerFilter`](@ref) — every keyword and its contract
- [`AutoRIFT.rescale`](@ref) — why a fine grid gets a wider window automatically
