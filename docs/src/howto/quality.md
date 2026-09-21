
# Judging a result

Every point that returns a displacement returns it with a `correlation` and a `peak_ratio`. They
measure different things and only one of them is a reliability gate.

## Gate on `correlation`

`correlation` is the height of the matched peak — how *strong* the match was. `peak_ratio` is the peak
divided by the best rival elsewhere on the same surface — whether the match was *unique*. A threshold
wants the first.

That is measured, not assumed. Over the 80,648 directly measured points of a Jakobshavn pair that both
this package and the Python reference resolved, labelled against "the two disagree by more than
0.25 px" — which 0.34% of them carry:

| measure | AUC | disagreement rate, worst decile | best decile |
|---|---:|---:|---:|
| `correlation` | **0.791** | 1.2% | 0.0% |
| `peak_ratio` | 0.544 | 0.5% | 0.6% |

`correlation` falls monotonically across its deciles, from 1.2% down to zero. `peak_ratio` is flat at
0.1–0.6% and not even monotonic — its highest decile is its worst — so **as a reliability gate on its
own it has essentially no skill.** Nor is that an artifact of pooling points of differing correlation:
within a band of fixed `correlation` the ranking mildly *inverts*, at an AUC of 0.43 over 0.2–0.4 and
0.50 over 0.4–0.6. The two measures are nearly independent, at a Spearman rank correlation of 0.242.
`tools/ab/peak_ratio_skill.jl` computes all of it.

That is not a defect in the quantity, it is what the quantity measures. Ambiguity and unreliability are
different failures, and on this scene — a fast outlet glacier whose flow the search radius covers — the
two implementations disagree where the match is *weak*, not where it is contested. A ratio near 1 does
mean two displacements matched nearly equally well; it does not follow that the one chosen was wrong.

So the gate is one line:

```@example quality
using AutoRIFT
include("../../figures.jl")  # plotting helpers and the synthetic scenes; see [Plotting](@ref)

band(row) = clamp((108 - abs(row - 256)) / 28, 0, 1)
reference, clean, _, _ = warped_pair(512, (row, col) -> (2 + 20 * band(row), 0.0); seed = 11)
secondary = decorrelate(clean, (row, col) -> band(row); amplitude = 0.3, seed = 3)

out = autorift(reference, secondary; chip_size = 16, chip_size_max = 64, grid_spacing = 8)

good = out.correlation .> 0.3
(measured = AutoRIFT.nmeasured(out), above_threshold = count(good))
```

```@example quality
field_panels(blank(out.dx, measured(out)), blank(out.dx, good);
             titles = ("dx, all measured", "dx, correlation > 0.3"))
```

What threshold depends on the imagery and on what the result feeds. `0.3` is a common working floor;
below about `0.1` a ZNCC peak carries little information. There is no universal number — pick it by
looking at the correlation field for your own data.

## The two layers side by side

They disagree, which is the point:

```@example quality
quality_panels(out)
```

```@example quality
field_panels(blank(out.correlation, measured(out)),
             blank(min.(out.peak_ratio, 5), measured(out));
             titles = ("correlation", "peak_ratio (clipped at 5)"), colormap = :magma)
```

`peak_ratio` has structure the correlation field does not, and vice versa. Neither is a proxy for the
other.

## What `peak_ratio` is for

Diagnosis, not thresholding. A ratio near 1 means some other displacement matched nearly as well, so
the reported one is a coin flip between two candidates. That happens with **periodic texture** —
crevasse fields, dune trains, crop rows, a regular particle seeding — which produces rival peaks one
wavelength apart, each as tall as the true one. A peak height cannot see that failure at all.

The fix when you find it is a chip size that spans more than one wavelength, not a tighter gate.

Two special values:

- `Inf32` — no rival peak was positive, so the primary is the only candidate the surface offers.
- `NaN32` — the surface was too small to have anything outside the exclusion box, so no ratio could be
  computed. Distinct from a low ratio.

Testing `isfinite` separates both from a real value.

## Finding where `search_radius` is too small

This is the failure that looks most like a measurement: if the true displacement lies outside the
search window, the surface never contains the right peak and the best available one wins. The
correlator detects the condition — the peak landing on the surface's own edge — and marks it by setting
**both `correlation` and `peak_ratio` to zero**.

Zero is otherwise unreachable for either: a real ZNCC peak is strictly positive, and a real ratio is at
least 1, the primary being the surface maximum. So any positive threshold on either layer already
rejects these points, without the caller needing to know the condition exists.

The displacement is still reported, being the best available lower bound. To find such points
deliberately:

```@example quality
# Motion of 24 pixels against a radius of 25: inside the window's nominal reach, but the surface spans
# -radius to radius-1, so the peak lands on its last sample.
railed_ref, railed_sec, _, _ = warped_pair(512, (row, col) -> (24.0, 0.0); seed = 9)
railed = autorift(railed_ref, railed_sec;
                  chip_size = 32, chip_size_max = 32, search_radius = 25, grid_spacing = 32)

m = measured(railed)
(measured = count(m), at_boundary = count(railed.correlation[m] .== 0))
```

Every measured point is railed. Raise the radius and they resolve:

```@example quality
wider = autorift(railed_ref, railed_sec;
                 chip_size = 32, chip_size_max = 32, search_radius = 32, grid_spacing = 32)
mw = measured(wider)
(at_boundary = count(wider.correlation[mw] .== 0),
 dx = wider.dx[findfirst(mw)])
```

A whole scene of zeros is unmistakable. A *region* of them is the useful case — it says the radius
covers most of the scene but not its fastest part, and is the signal to raise the radius there
specifically rather than everywhere. A per-point radius is how ([A guided walkthrough](@ref), step 7).

!!! note "Beyond the radius entirely, nothing is returned"
    The railing above needs the motion to be near the radius. Push it well past — 30 pixels against a
    radius of 25 — and the level's coarse pass finds nothing spatially coherent to work from, so the
    level is skipped and the result is empty rather than railed. An all-`NaN` result with a radius you
    thought was generous is the same diagnosis by a different route.

## The other layers

`chip_size` records which level answered, and `0` means none did. It is also how much spatial averaging
is behind each estimate, so it distinguishes a sharply-resolved displacement from a smoothed one — see
[Multiple chip sizes](@ref).

`interpolated` marks points filled from their neighbours rather than measured. At such a point
`correlation` is `NaN`, having no surface of its own to report, and `peak_ratio` is the median of the
neighbourhood the displacement came from — since that is what stands behind the value.

```@example quality
(interpolated = count(out.interpolated),
 unresolved = count(==(0), out.chip_size))
```

Whether to keep interpolated points depends on the consumer. They are real estimates with real
uncertainty, just not independent measurements; a strain calculation usually wants them excluded, a
visualization usually does not.

## Removing outliers

A `correlation` gate is a point-by-point test and cannot see a displacement that is individually
well-correlated but inconsistent with its neighbours. That is what [`GardnerFilter`](@ref) does, and it
runs inside `autorift` by default — see [Filtering outliers](@ref) for its knobs.

## See also

- [How feature tracking works](@ref) — why a peak can be wrong in the first place
- [Filtering outliers](@ref) — the neighbourhood test the per-point gate cannot do
- [`AutoRIFT.MultichipResult`](@ref) — every layer, and its contract
