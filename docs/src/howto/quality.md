
# Judging a result

Every point that returns a displacement also returns a `correlation` and a `peak_ratio`. They measure
different things, and only `correlation` is useful as a reliability threshold.

## Threshold on `correlation`

`correlation` is the height of the matched peak, which says how strong the match was. `peak_ratio` is
the peak divided by the best rival elsewhere on the same surface, which says whether the match was
unique. A reliability threshold needs the first.

The comparison below uses the 80,648 directly measured points of a Jakobshavn pair that both this
package and the Python reference resolved. Each point is labelled by whether the two disagree by more
than 0.25 px, which 0.34% of them do:

| measure | AUC | disagreement rate, worst decile | best decile |
|---|---:|---:|---:|
| `correlation` | 0.791 | 1.2% | 0.0% |
| `peak_ratio` | 0.544 | 0.5% | 0.6% |

The disagreement rate falls monotonically across the deciles of `correlation`, from 1.2% to zero. For
`peak_ratio` it is flat at 0.1–0.6% and not monotonic (its highest decile is its worst), so
`peak_ratio` has essentially no skill as a reliability threshold. Pooling points of different
correlation does not explain this: within a band of fixed `correlation`, the AUC of `peak_ratio` is 0.43
over 0.2–0.4 and 0.50 over 0.4–0.6. The two measures are nearly independent, with a Spearman rank
correlation of 0.242. `tools/ab/peak_ratio_skill.jl` computes all of these.

Ambiguity and unreliability are different failures. On this scene, a fast outlet glacier whose flow the
search radius covers, the two implementations disagree where the match is weak, not where it is
contested. A ratio near 1 means two displacements matched nearly equally well, which does not make the
chosen one wrong.

The threshold is one line:

```@example quality
using AutoRIFT
include("../../figures.jl")  # plotting helpers and the example scenes; see [Plotting](@ref)

band(row) = clamp((108 - abs(row - 256)) / 28, 0, 1)
reference, clean, _, _ = warped_pair(photo(), 512, (row, col) -> (2 + 20 * band(row), 0.0))
secondary = decorrelate(clean, (row, col) -> band(row); amplitude = 0.3, seed = 3)

out = autorift(reference, secondary; chip_size = 16, chip_size_max = 64, grid_spacing = 8)

good = out.correlation .> 0.3
(measured = AutoRIFT.nmeasured(out), above_threshold = count(good))
```

The pair: a band across the middle moved 22 pixels and has noise added in the secondary; the rest moved 2.

```@example quality
image_panels(reference, secondary)
```

```@example quality
field_panels(blank(out.dx, measured(out)), blank(out.dx, good);
             titles = ("dx, all measured", "dx, correlation > 0.3"))
```

The right threshold depends on the imagery and on what the result feeds. `0.3` is a common floor, and
below about `0.1` a ZNCC peak carries little information. Choose it by inspecting the correlation field
for your own data.

## The two layers side by side

```@example quality
field_panels(blank(out.correlation, measured(out)),
             blank(min.(out.peak_ratio, 5), measured(out));
             titles = ("correlation", "peak_ratio (clipped at 5)"), colormap = :magma)
```

Each layer shows structure the other does not.

## What `peak_ratio` is for

`peak_ratio` is a diagnostic. A ratio near 1 means some other displacement matched nearly as well, so
the reported one is a choice between two candidates. This happens with periodic texture such as
crevasse fields, dune trains, crop rows, or regular particle seeding, which produces rival peaks one
wavelength apart, each as tall as the true one. A peak height cannot detect this failure. The remedy is
a chip size that spans more than one wavelength.

Two special values:

- `Inf32`: no rival peak was positive, so the primary is the only candidate the surface offers.
- `NaN32`: no ratio could be computed, because the surface was too small to have anything outside the
  exclusion box or its maximum was not positive. This is distinct from a low ratio.

`isfinite` separates both from a real value.

## Finding where `search_radius` is too small

If the true displacement lies outside the search window, the surface does not contain the right peak and
the best available one wins, which looks like a normal measurement. The correlator detects the condition
by the peak landing on the surface's own edge, and sets both `correlation` and `peak_ratio` to zero.

Zero is otherwise unreachable for either layer: a real ZNCC peak is strictly positive, and a real ratio
is at least 1 because the primary is the surface maximum. Any positive threshold on either layer
therefore rejects these points. The displacement is still reported, as a lower bound. To find such
points:

```@example quality
# Motion of 24 pixels against a radius of 25: inside the window's nominal reach, but the surface spans
# -radius to radius-1, so the peak lands on its last sample.
railed_ref, railed_sec, _, _ = warped_pair(photo(), 512, (row, col) -> (24.0, 0.0))
railed = autorift(railed_ref, railed_sec;
                  chip_size = 32, chip_size_max = 32, search_radius = 25, grid_spacing = 32)

m = measured(railed)
(measured = count(m), at_boundary = count(railed.correlation[m] .== 0))
```

All but two measured points are railed. Raise the radius and they resolve:

```@example quality
wider = autorift(railed_ref, railed_sec;
                 chip_size = 32, chip_size_max = 32, search_radius = 32, grid_spacing = 32)
mw = measured(wider)
(at_boundary = count(wider.correlation[mw] .== 0),
 dx = wider.dx[findfirst(mw)])
```

A region of zeros means the radius covers most of the scene but not its fastest part. Raise the radius
there with a per-point radius rather than everywhere ([A guided walkthrough](@ref), step 7).

!!! note "Motion far beyond the radius returns nothing"
    Railing needs the motion to be near the radius. With 30 pixels of motion against a radius of 25,
    the level's coarse pass finds nothing spatially coherent, the level is skipped, and the result is
    empty rather than railed. An all-`NaN` result with a radius you thought generous has the same
    cause.

## The other layers

`chip_size` records which level answered, and `0` means none did. It also gives the spatial averaging
behind each estimate, which distinguishes a sharply resolved displacement from a smoothed one; see
[Multiple chip sizes](@ref).

`interpolated` marks points filled from their neighbors rather than measured. At such a point
`correlation` is `NaN`, because there is no surface of its own, and `peak_ratio` is the median over the
neighborhood the displacement came from.

```@example quality
(interpolated = count(out.interpolated),
 unresolved = count(==(0), out.chip_size))
```

Where they are matters more than how many, because it decides whether excluding them loses a region or
scattered points:

```@example quality
mask_panels(out.interpolated, out.chip_size .== 0;
            titles = ("interpolated", "unresolved"))
```

The interpolated points are 79 points in small scattered clusters, so excluding them loses little. The
unresolved points form a solid block across the decorrelating band, apart from a measured patch at its
center. That loss is a whole region of the scene, and no threshold or filter setting recovers it. The
counts 79 and 1119 do not show this difference; the figure does.

Whether to keep interpolated points depends on the use. They are estimates from neighboring
measurements, not independent measurements. A strain calculation usually excludes them, and a
visualization usually keeps them.

## Removing outliers

A `correlation` threshold tests each point alone, so it cannot reject a displacement that is
well correlated but inconsistent with its neighbors. [`GardnerFilter`](@ref) does, and it runs inside
`autorift` by default; see [Filtering outliers](@ref) for its settings.

## See also

- [How feature tracking works](@ref): why a peak can be wrong
- [Filtering outliers](@ref): the neighborhood test the per-point threshold cannot do
- [`AutoRIFT.MultichipResult`](@ref): every layer, and its contract
