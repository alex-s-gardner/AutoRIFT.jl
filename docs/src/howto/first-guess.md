
# Giving the search a first guess

The search radius is the hard limit on what can be found: motion beyond it is not measured badly, it is
not measured at all. A prior moves the search window so the radius only has to cover the *error* in your
estimate rather than the motion itself.

This is what makes large displacements affordable, and it is the answer to "my correlation returns
nothing".

## The problem a prior solves

A 512² pair whose features move 28 pixels in x and 12 in y:

```@example guess
using AutoRIFT
include("../../figures.jl")  # plotting helpers and the synthetic scenes; see [Plotting](@ref)
using Statistics: median

reference, secondary, truth_dx, _ = warped_pair(512, (row, col) -> (28.0, -12.0); seed = 17)
settings = (; chip_size = 32, chip_size_max = 32, grid_spacing = 16)

image_panels(reference, secondary)
```

```@example guess
for (label, extra) in ("search_radius = 10, no prior" => (; search_radius = 10),
                       "search_radius = 40, no prior" => (; search_radius = 40),
                       "search_radius = 10 + prior" => (; search_radius = 10,
                                                        dx_prior = 28.0, dy_prior = -12.0))
    out = autorift(reference, secondary; settings..., extra...)
    n = AutoRIFT.nmeasured(out)
    println(rpad(label, 30), " measured ", lpad(n, 4),
            n == 0 ? "" :
                "   median correlation $(round(median(filter(isfinite, out.correlation)); digits = 3))")
end
```

A radius of 10 against motion of 28 returns **nothing at all** — not a wrong answer, which is the one
mercy here. A radius of 40 finds it. A radius of 10 *with a prior* also finds it, at the same
correlation, because the prior moved the window to where the features went and the search only had to
cover the remainder.

## Priors as keywords

For a translation-dominated pair, two scalars are the whole interface:

```@example guess
out = autorift(reference, secondary; settings..., search_radius = 10,
               dx_prior = 28.0, dy_prior = -12.0)
field_panels(blank(out.dx, measured(out)), blank(out.dy, measured(out));
             titles = ("dx with prior (28, -12)", "dy with prior (28, -12)"))
```

The reported `dx`/`dy` are **total** displacement, not the residual from the prior. The prior shifts
where the search looks; it does not change what is reported.

## How wrong a prior may be

The prior only has to land the true displacement inside the radius. Here that is a 10-pixel allowance:

```@example guess
for offset in (0, 4, 8, 10, 12, 16)
    out = autorift(reference, secondary; settings..., search_radius = 10,
                   dx_prior = 28.0 - offset, dy_prior = -12.0)
    n = AutoRIFT.nmeasured(out)
    corr = n == 0 ? "—" : string(round(median(filter(isfinite, out.correlation)); digits = 3))
    println("prior wrong by ", lpad(offset, 2), " px:  measured ", lpad(n, 4),
            "   median correlation ", corr)
end
```

Read the boundary carefully. Off by 8, everything works. Off by **exactly 10** — the radius — the points
are still reported but median correlation collapses to **0.0**: the peak landed on the last sample of the
surface, so it is railed rather than located. Off by 12, nothing is returned.

Correlation `0.0` beside a full point count is the signature of a radius that is *just* too small, and it
is why [Judging a result](@ref) recommends gating on correlation. A prior that is confidently wrong is
worse than no prior: a bad guess moves the window away from the answer and the run returns nothing.

## What a prior costs

```@example guess
(wide = AutoRIFT.nsearchable(AutoRIFT.gridpoints(Base.size(reference), 16;
                                                 chip_size = 32, search_radius = 40)),
 narrow = AutoRIFT.nsearchable(AutoRIFT.gridpoints(Base.size(reference), 16;
                                                   chip_size = 32, search_radius = 10)))
```

A wide radius costs two ways. Search area grows as the radius squared, so each point does more work; and
the margin a point needs from the image edge grows with it, so a wide radius searches **fewer** points on
the same scene. Measured on this pair, radius 40 runs about 1.2× the time of radius 10 with a prior —
modest here, but the gap widens with the radius ratio, and on a year-long pair where motion is hundreds
of pixels an unaided search is not merely slow but infeasible.

## Spatially varying priors

A scalar prior assumes one displacement for the whole scene. When motion varies — fast in a channel, slow
at the margins — the prior should vary too, and that requires a [`AutoRIFT.PointSet`](@ref) because a per-point
field cannot be a scalar keyword:

```@example guess
grid = AutoRIFT.gridpoints(Base.size(reference), 16; chip_size = 32, search_radius = 10)
varying = AutoRIFT.pointset(grid.x, grid.y;
                            chip_size_x = 32, chip_size_y = 32,
                            search_radius_x = 10, search_radius_y = 10,
                            dx_prior = fill(28.0, Base.size(grid.x)),
                            dy_prior = fill(-12.0, Base.size(grid.x)))
AutoRIFT.nmeasured(autorift(reference, secondary, varying))
```

Any of `dx_prior`, `dy_prior`, `search_radius_x`, `search_radius_y`, `chip_size_x` and `chip_size_y` takes
a scalar or an array matching the shape of the points, so mixing a varying prior with a uniform radius is
ordinary. [A guided walkthrough](@ref) builds this up step by step.

Where such a prior comes from is your problem to solve and the package makes no assumption about it: a
published velocity field, a coarse run of `autorift` itself at a large radius, or a physical model. For
geolocated data, the conversion is `pixels = (m/yr) × (dt / 365.25) / pixel_size` — see
[Geospatial data](@ref).

## Sparse feature matching

The package can also derive a prior from the imagery itself, by matching sparse keypoints before the
dense search. That needs a detector package loaded, so the following does not execute here:

```julia
using ImageFeatures                      # for ORBGuess
guess = first_guess(reference, secondary, AutoRIFT.ORBGuess())
out = autorift(reference, secondary, guess)
```

`first_guess` returns a `PointSet` whose `dx_prior`/`dy_prior` carry the sparse estimates, with
`search_radius = 6` and `chip_size = 32` by default — deliberately small, since the prior is doing the
reaching. It throws rather than returning almost nothing if fewer than `min_matches = 8` survive: a guess
that found nothing is information about the pair, not a result to pass downstream.

### Which detector

[`ORBGuess`](@ref) needs `ImageFeatures` and is the default. Muckenhuber et al. (2016) compared three
detectors on Sentinel-1 sea ice over Fram Strait and north-east Greenland:

| detector | vectors | time |
|---|---:|---:|
| **ORB** | **177,513** | **66 s** |
| SIFT | 43,260 | 182 s |
| SURF | 25,113 | 99 s |

Four times the vectors of SIFT in a third of the time. ORB is also unencumbered, where SIFT and SURF
were patented when that paper was written — which is why the paper's title says *open-source*.

[`AKAZEGuess`](@ref) needs `AkazeFeatures`, is about 5× slower, and is far more precise under rotation.
Demchev et al. (2017) report A-KAZE outperforming ORB "up to an order of magnitude" on ice drift:
Gaussian scale space blurs speckle and signal alike, while A-KAZE's nonlinear diffusion preserves
edges. Measured against synthetic speckle with known ground truth, at matched keypoint counts on 512²:

| rotation | ORB matches (correct) | A-KAZE matches (correct) |
|---:|---:|---:|
| 0° | 9000 (78.4%) | 8406 (**99.3%**) |
| 3° | 6434 (33.5%) | 6257 (**96.0%**) |
| 8° | 5666 (19.4%) | 5937 (**95.8%**) |

ORB's precision collapses as the field rotates; A-KAZE's does not. In *usable* vectors that is 1.2× at
0° rising to **5.2× at 8°**, against 5× the detection time (0.49 s vs 0.10 s at ~9000 keypoints). So
the two roughly break even on cost per usable vector once there is rotation, and A-KAZE wins outright
on the precision that determines whether the consistency filter has anything left to keep. It is not
the default only because it is unregistered.

Raw matches are not usable directly — descriptor mismatches land anywhere. [`AutoRIFT.consistent_matches`](@ref)
keeps a match only if its nearest neighbours agree with it, which took raw ORB matching from 9.5–62.5%
correct to 100% correct on synthetic tests, at the cost of most of the matches. `first_guess` applies it
for you.

!!! tip "Load `NearestNeighbors` as well"
    The consistency filter's neighbour search is O(n²) without it and O(n log n) with — 90× at 20,000
    matches. Loading a detector does not pull it in, so `using NearestNeighbors` is a separate step and
    worth taking beyond a few thousand matches.

## Rotation

If the scene rotates, a translation prior is not enough. [`scene_rotation`](@ref) fits the single rotation
that best explains a sparse field, and [`RotationSearch`](@ref) centres the per-chip angle search on it:

```julia
guess = first_guess(reference, secondary, AutoRIFT.AKAZEGuess())
out = autorift(reference, secondary, guess;
               rotation = RotationSearch(; about = scene_rotation(guess)))
```

`about` takes the returned value directly — no negation at the call site, since the sign convention
already matches the rest of the package.

### What the angle search recovers

A rotated chip decorrelates against an unrotated window even where the surface is perfectly
trackable. Measured on synthetic speckle, median peak correlation with five angles (0°, ±3°, ±6°)
against none:

| scene rotation | chip 32 | chip 64 |
|---:|---:|---:|
| 0° | 0.571 → 0.571 (0%) | 0.571 → 0.571 (0%) |
| 3° | 0.113 → 0.146 (**+29%**) | 0.112 → 0.279 (**+149%**) |
| 6° | 0.085 → 0.103 (+21%) | 0.043 → 0.052 (+21%) |
| 10° | 0.082 → 0.101 (+23%) | 0.042 → 0.053 (+25%) |

**The gain grows with chip size** — a 64-pixel chip's corners travel twice as far as a 32-pixel
chip's under the same rotation, so it has more to lose and more to recover. And **the benefit is real
but modest against decorrelation**: at 6° and beyond the correlation is weak either way, because
rotating a square chip pulls in padding that was never part of it. That is why ±3° is the default
span.

Cost is **1.7×** for five angles rather than 5×, because the surrounding per-point work — window
extraction, integral images, the peak search — is shared across the angles.

The angle search does not rescue the *sparse* stage. Measured through the first-guess path, usable
vectors fall from 2,327 at 0° rotation to **29 at 10°**, and most of that loss is descriptor matching
rather than dense correlation — which is the case for `AKAZEGuess` on a rotating scene.

### Why `about` subtracts

The chip comes from the **secondary** and is correlated against an **unrotated** reference window, so
it has to be turned *back* to the reference's orientation: the rotations actually applied are
`angles .- about`. That is not a convention to choose. Measured on speckle rotated 8°, peak
correlation of a chip rotated by each candidate:

| chip | no rotation | +8° | −8° |
|---:|---:|---:|---:|
| 32 | 0.723 | 0.345 | 0.597 |
| 64 | 0.328 | 0.132 | **0.641** |
| 128 | 0.142 | 0.025 | **0.670** |

Counter-rotation is the only candidate that recovers anything, and it recovers more the larger the
chip — 0.14 → 0.67 at chip 128, a **4.7×** gain. At chip 32 it still loses to no rotation at all,
because a 32-pixel chip's corners travel only about 2 pixels at 8° while resampling and corner
padding cost more than that.

This is why one scene-level estimate is worth fitting. A scene rotated 8° is outside `±3°` entirely —
every angle tried is wrong by at least 5°, and widening the tuple to reach it pays a correlation per
angle for angles that can only lose. `about = 8.0` moves the whole window instead, searching 5–11° at
the same 3× cost.

!!! warning "`about` is only valid for priors that measured *this* pair"
    A `PointSet`'s priors are not always measured displacements. A prior from a velocity *model* projected
    onto the image axes will fit a confident nonzero angle that describes the projection geometry, not the
    pair's rotation — and then every chip in the run is rotated by it. Nothing in the type distinguishes
    the two cases, which is why `about` is passed explicitly: passing it asserts that these priors came
    from matching these images.

    This is also why a rotating time series should call `autorift` per pair with a freshly fitted `about`
    rather than walking a [`AutoRIFT.Cache`](@ref), which keeps its `Params`. See [Correlating many pairs](@ref).

Two limits worth knowing: the fit removes translation and does not fit scale, so divergence and shear are
residuals; and it fits *one* rotation, so a scene with two regions rotating opposite ways fits near their
average and describes neither. The per-chip search is what handles that.

### What the fit is

An orthogonal Procrustes fit, which in two dimensions reduces to a single `atan2` over two sums.
Given reference positions `p` and displacements `d`, so the secondary position is `p - d`:

```
θ = atan2(Σ (sx·ty − sy·tx), Σ (sx·tx + sy·ty))
```

with `s` the centred `p - d` and `t` the centred `p`. Summing the cross and dot products before
taking the angle makes it a least-squares fit over every point at once, with no wrapping and no
per-point angle to average — the same shape as [`Deramp`](@ref)'s phase estimator, for the same
reason.

The angle returned carries **secondary orientation onto reference**, which is the same direction as
`dx`/`dy`: the negative of the features' own motion. That is why `about` takes the value with no
negation at the call site.

It returns `nothing`, not `NaN`, when there is nothing to fit — fewer than two points, coincident
points, or a field with no rotational component, where both sums vanish and `atan(0, 0)` would report
a confident zero. `nothing` fails at first use, so `RotationSearch(; about = scene_rotation(pts))` on
an unfittable field raises a `MethodError` at the line that made the mistake. A scalar `NaN` would
survive `α / 2` and `round(α)` unremarked and surface much later.

## See also

- [A guided walkthrough](@ref) — priors and per-point fields, built up in order
- [Judging a result](@ref) — correlation `0.0` and the railed peak
- [Geospatial data](@ref) — converting a published velocity to a pixel prior
- [First guess](@ref) — the detector and fitting docstrings
