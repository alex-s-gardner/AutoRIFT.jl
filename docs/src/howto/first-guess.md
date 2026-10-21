
# Giving the search a first guess

The search radius limits what can be found: motion beyond it is not measured. A prior moves the search
window, so the radius only has to cover the error in the estimate rather than the motion itself. This
makes large displacements affordable, and it is the fix when a correlation returns nothing.

## The problem a prior solves

A 512² pair with displacement `dx = 28`, `dy = -12` pixels (the package's sign convention, see
[Conventions](@ref)):

```@example guess
using AutoRIFT
include("../../figures.jl")  # plotting helpers and the example scenes; see [Plotting](@ref)
using Statistics: median

reference, secondary, truth_dx, _ = warped_pair(photo(), 512, (row, col) -> (28.0, -12.0))
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

A radius of 10 against a displacement of 28 measures no points. A radius of 40 finds the motion. A
radius of 10 with a prior finds it at the same median correlation, because the prior centers the window
where the features went and the search only covers the remainder. The prior run also measures more
points (758 against 601), because a smaller radius leaves more grid points far enough from the image edge.

## Priors as keywords

For a pair dominated by translation, two scalars are enough:

```@example guess
out = autorift(reference, secondary; settings..., search_radius = 10,
               dx_prior = 28.0, dy_prior = -12.0)
field_panels(blank(out.dx, measured(out)), blank(out.dy, measured(out));
             titles = ("dx with prior (28, -12)", "dy with prior (28, -12)"))
```

The reported `dx` and `dy` are the total displacement, not the residual from the prior.

## How wrong a prior may be

The prior has to place the true displacement inside the search radius, a 10-pixel allowance here:

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

An error of 8 pixels is harmless. At an error equal to the radius (10), points are still reported but the
median correlation is **0.0**: the peak lies on the last sample of the correlation surface, so it is
located only to the boundary. At 12 pixels nothing is returned.

A median correlation of `0.0` beside a full point count means the radius is too small for the residual
motion. This is one reason [Judging a result](@ref) recommends gating on `correlation`. A wrong prior is
worse than none: it moves the window away from the true displacement, and the run returns nothing.

## What a prior costs

```@example guess
(wide = AutoRIFT.nsearchable(AutoRIFT.gridpoints(Base.size(reference), 16;
                                                 chip_size = 32, search_radius = 40)),
 narrow = AutoRIFT.nsearchable(AutoRIFT.gridpoints(Base.size(reference), 16;
                                                   chip_size = 32, search_radius = 10)))
```

A wide radius costs in two ways. The search area grows as the radius squared, so each point does more
work. The margin a point needs from the image edge also grows with the radius, so a wide radius searches
fewer points on the same scene (625 against 841). On this pair, radius 40 takes about 1.2× the time of
radius 10 with a prior. The gap widens with the radius ratio, and for motion of hundreds of pixels an
unaided search is not feasible.

## Spatially varying priors

A scalar prior assumes one displacement for the whole scene. When motion varies, for example fast in a
channel and slow at the margins, the prior has to vary too. A per-point prior needs an
[`AutoRIFT.PointSet`](@ref):

```@example guess
grid = AutoRIFT.gridpoints(Base.size(reference), 16; chip_size = 32, search_radius = 10)
varying = AutoRIFT.pointset(grid.x, grid.y;
                            chip_size_x = 32, chip_size_y = 32,
                            search_radius_x = 10, search_radius_y = 10,
                            dx_prior = fill(28.0, Base.size(grid.x)),
                            dy_prior = fill(-12.0, Base.size(grid.x)))
AutoRIFT.nmeasured(autorift(reference, secondary, varying))
```

Each of `dx_prior`, `dy_prior`, `search_radius_x`, `search_radius_y`, `chip_size_x` and `chip_size_y`
takes a scalar or an array matching the shape of the points, so a varying prior can be combined with a
uniform radius. [A guided walkthrough](@ref) builds this up step by step.

The package does not supply a prior. Sources include a published velocity field, a coarse `autorift` run
with a large radius, or a physical model. For geolocated data the conversion is
`pixels = (m/yr) × (dt / 365.25) / pixel_size`, followed by the sign convention in [Conventions](@ref).
See [Geospatial data](@ref).

## Sparse feature matching

A prior can also come from the imagery: sparse keypoints are matched before the dense search. This needs
a detector package, so the following is not executed here:

```julia
using ImageFeatures                      # for ORBGuess
guess = first_guess(reference, secondary, AutoRIFT.ORBGuess())
out = autorift(reference, secondary, guess)
```

`first_guess` returns a `PointSet` whose `dx_prior` and `dy_prior` hold the sparse estimates, with
`search_radius = 6` and `chip_size = 32` by default. The radius is small because the prior does the
reaching. It throws an `ArgumentError` if fewer than `min_matches = 8` matches survive the consistency
filter, since a guess that found nothing says the pair is not trackable.

### Which detector

[`ORBGuess`](@ref) needs `ImageFeatures`. Muckenhuber et al. (2016) compared three detectors on
Sentinel-1 sea ice over Fram Strait and north-east Greenland:

| detector | vectors | time |
|---|---:|---:|
| **ORB** | **177,513** | **66 s** |
| SIFT | 43,260 | 182 s |
| SURF | 25,113 | 99 s |

ORB gave four times the vectors of SIFT in a third of the time. SIFT and SURF were patented when that
paper was written.

[`AKAZEGuess`](@ref) needs `AkazeFeatures`, is about 5× slower, and is far more precise under rotation.
Demchev et al. (2017) report A-KAZE outperforming ORB "up to an order of magnitude" on ice drift. A
Gaussian scale space blurs speckle and signal alike, while A-KAZE's nonlinear diffusion preserves edges.
Measured on synthetic speckle with known ground truth, at matched keypoint counts on 512²:

| rotation | ORB matches (correct) | A-KAZE matches (correct) |
|---:|---:|---:|
| 0° | 9000 (78.4%) | 8406 (**99.3%**) |
| 3° | 6434 (33.5%) | 6257 (**96.0%**) |
| 8° | 5666 (19.4%) | 5937 (**95.8%**) |

ORB's precision falls as the field rotates; A-KAZE's does not. A-KAZE gives 1.2× as many correct
vectors at 0° and 5.2× at 8°, for about 5× the detection time (0.49 s against 0.10 s at about 9000
keypoints). With rotation the two cost about the same per correct vector, and A-KAZE leaves more matches
for the consistency filter to keep. `AkazeFeatures` is not in the General registry, so it must be
installed by URL.

Raw descriptor matches are not usable directly, because mismatches land anywhere. A match is kept by
[`AutoRIFT.consistent_matches`](@ref) only if at least `min_agree = 5` of its `neighbours = 12` nearest
matches have a displacement within `tolerance = 6.0` pixels of its own. On synthetic tests this raised
raw ORB matches from 9.5–62.5% correct to 100% correct, at the cost of most of the matches.
`first_guess` applies it.

!!! tip "Load `NearestNeighbors` as well"
    The consistency filter's neighbor search is O(n²) without `NearestNeighbors` and O(n log n) with it,
    90× faster at 20,000 matches. Loading a detector does not load it, so `using NearestNeighbors` is a
    separate step. It matters beyond a few thousand matches.

## Rotation

If the scene rotates, a translation prior is not enough. [`scene_rotation`](@ref) fits the single
rotation that best explains a sparse field, and [`RotationSearch`](@ref) centers the per-chip angle
search on it:

```julia
guess = first_guess(reference, secondary, AutoRIFT.AKAZEGuess())
out = autorift(reference, secondary, guess;
               rotation = RotationSearch(; about = scene_rotation(guess)))
```

`about` takes the returned value directly. The sign convention already matches the rest of the package.

### What the angle search recovers

A rotated chip decorrelates against an unrotated window even where the surface is trackable. This table
gives the median peak correlation on synthetic speckle with five angles (0°, ±3°, ±6°) against none:

| scene rotation | chip 32 | chip 64 |
|---:|---:|---:|
| 0° | 0.571 → 0.571 (0%) | 0.571 → 0.571 (0%) |
| 3° | 0.113 → 0.146 (**+29%**) | 0.112 → 0.279 (**+149%**) |
| 6° | 0.085 → 0.103 (+21%) | 0.043 → 0.052 (+21%) |
| 10° | 0.082 → 0.101 (+23%) | 0.042 → 0.053 (+25%) |

The gain grows with chip size, because a 64-pixel chip's corners move twice as far as a 32-pixel chip's
under the same rotation. At 6° and beyond the correlation is weak either way, because rotating a square
chip pulls in padding that was not part of it.

Five angles cost 1.7× rather than 5×, because the per-point work around the correlation (window
extraction, integral images, the peak search) is shared across angles.

The angle search does not help the sparse stage. Through the first-guess path, usable vectors fall from
2,327 at 0° rotation to 29 at 10°, mostly from descriptor matching rather than dense correlation. This
favors `AKAZEGuess` on a rotating scene.

### Why `about` subtracts

The chip is taken from the secondary and correlated against an unrotated reference window. It must be
rotated back to the reference's orientation, so the rotations applied are `angles .- about`. The panels
below show a scene turned 8° and the chip the correlator takes from it, rotated by each candidate:

```@example guess
img = Float32.(photo())
r0, c0, h = 300, 300, 110
rows, cols = (r0 - h):(r0 + h - 1), (c0 - h):(c0 + h - 1)

window = img[rows, cols]                                        # from the reference, upright
turned = AutoRIFT._rotate_bilinear(img, 8.0, (Float64(c0), Float64(r0)))
chip = turned[rows, cols]                                       # from the secondary, carrying the 8°
centre = (Float64(h), Float64(h))

image_panels(window, chip,
             AutoRIFT._rotate_bilinear(chip, 8.0, centre),
             AutoRIFT._rotate_bilinear(chip, -8.0, centre);
             titles = ("reference window", "chip from the secondary",
                       "chip rotated +8°", "chip rotated −8°"),
             size = (1500, 400))
```

The waistband is horizontal in the reference window and tilted in the chip. Rotating by +8° tilts it
further, and only −8° brings it level. Rotating with the scene's rotation doubles the mismatch.

On speckle rotated 8°, the peak correlation of a chip rotated by each candidate is:

| chip | no rotation | +8° | −8° |
|---:|---:|---:|---:|
| 32 | 0.723 | 0.345 | 0.597 |
| 64 | 0.328 | 0.132 | **0.641** |
| 128 | 0.142 | 0.025 | **0.670** |

Counter-rotation recovers correlation, and more of it for larger chips: 0.14 to 0.67 at chip 128, a 4.7×
gain. At chip 32 no rotation is still better, because a 32-pixel chip's corners move only about 2 pixels
at 8° and resampling and corner padding cost more than that.

A scene rotated 8° is outside the default `±3°` span: every angle tried is wrong by at least 5°, and
widening the tuple to reach 8° pays a correlation for each added angle. `about = 8.0` shifts the whole
span instead, searching 5° to 11° at the same 3× cost.

!!! warning "`about` is only valid for priors that measured *this* pair"
    A `PointSet`'s priors are not always measured displacements. A prior from a velocity model, projected
    onto the image axes, can fit a confident nonzero angle that describes the projection geometry rather
    than the pair's rotation, and every chip in the run is then rotated by it. Nothing in the type
    distinguishes the two cases, so `about` is passed explicitly: passing it asserts that the priors came
    from matching these images.

    For the same reason, a rotating time series should call `autorift` per pair with a freshly fitted
    `about` rather than walking a [`AutoRIFT.Cache`](@ref), which keeps its `Params`. See
    [Correlating many pairs](@ref).

Two limits: the fit removes translation and does not fit scale, so divergence and shear remain as
residuals; and it fits one rotation, so a scene with two regions rotating in opposite directions fits
near their average and describes neither. The per-chip search handles that case.

### What the fit is

An orthogonal Procrustes fit, which in two dimensions is a single `atan2` over two sums. Given reference
positions `p` and displacements `d`, the secondary position is `p - d`:

```
θ = atan2(Σ (sx·ty − sy·tx), Σ (sx·tx + sy·ty))
```

where `s` is the centered `p - d` and `t` the centered `p`. Summing the cross and dot products before
taking the angle gives a least-squares fit over all points at once, with no per-point angle to wrap or
average. [`Deramp`](@ref)'s phase estimator is built the same way.

The angle carries the secondary's orientation onto the reference. This is the same direction as
`dx`/`dy`, the negative of the features' own motion, so `about` takes it without negation.

`scene_rotation` returns `nothing` when there is nothing to fit: fewer than two points, coincident points,
or a field with no rotational component, where both sums vanish and `atan(0, 0)` would report a zero
angle. Passing `nothing` as `about` fails at the `RotationSearch` call.

## See also

- [A guided walkthrough](@ref): priors and per-point fields, built up in order
- [Judging a result](@ref): correlation `0.0` and the peak on the search boundary
- [Geospatial data](@ref): converting a published velocity to a pixel prior
- [First guess](@ref): the detector and fitting docstrings
