
# How feature tracking works

Two images of the same scene are taken at different times, and something in the scene moved. Feature
tracking measures how far, at every grid point, by matching small patches of one image against the
other.

There is no feature detector and no correspondence problem. The method uses the texture already in
the imagery (grain, crevasses, speckle, seeded particles) and finds where each patch of it went.
Every grid point gets an answer, not only the points with distinctive features.

## The chip, the window, and the surface

Every parameter in the package configures one of three objects.

A **chip** is a square patch cut from one image at a grid point. A **search window** is a larger patch
cut from the other image around the same position, reaching `search_radius` pixels further in each
direction. The chip is slid over every position inside the window, and at each one a similarity score
is computed. Those scores make up the **correlation surface**.

```@example tracking
using AutoRIFT
include("../../figures.jl")  # plotting helpers and the example scenes; see [Plotting](@ref)

reference, secondary, _, _ = warped_pair(photo(), 512, (row, col) -> (6.4, -3.2))

chip_size, radius = 32, 16
centre = (256, 256)
h = chip_size ÷ 2

# The chip comes from the secondary image and the window from the reference, which is the arrangement
# that makes the reported offset point from secondary back to reference — see [Conventions](@ref).
chip = secondary[(centre[1] - h):(centre[1] + h - 1), (centre[2] - h):(centre[2] + h - 1)]
# The window reaches `radius` one way and `radius - 1` the other, which is what makes the surface an
# even `2 * radius` with zero displacement on a sample rather than between two.
window = reference[(centre[1] - h - radius):(centre[1] + h + radius - 2),
                   (centre[2] - h - radius):(centre[2] + h + radius - 2)]

ws = AutoRIFT.workspace(eltype(reference), chip_size, radius)
surface = AutoRIFT.correlate!(ws, window, chip, radius)

surface_panels(chip, window, surface)
```

The bright spot in the third panel is the correlation peak. The surface is `2 * radius` across in each
direction with zero displacement at its center, so a spot one cell right of center means `dx = 1`, and
the panel's corners are displacements of ±`radius`. The peak's offset from the center is therefore the
displacement:

```@example tracking
AutoRIFT.peak_offset(surface, (radius, radius))
```

The values are the offset in x, the offset in y, and the correlation at the peak. The pair was built
with an offset of `(6.4, -3.2)`, so the whole-pixel result is within one pixel and the fractional part
is missing. The sub-pixel stage below recovers it.

The correlation surface is not part of the public API. `autorift` does not return one, because a
surface per point would take more memory than the result. It is shown here because every parameter
acts on it.

This is the whole algorithm at one point. The rest of the package concerns chip size, search radius,
number of points, cost, and telling a good answer from a bad one.

!!! note "Other names for the same three things"
    Particle image velocimetry calls the chip an *interrogation window* and the correlation surface a
    *correlation map*. Digital image correlation calls the chip a *subset* and the grid spacing a
    *step size*. Optical flow literature calls the whole operation *block matching*. The objects are
    the same; only the vocabulary differs.

## Normalized cross-correlation

AutoRIFT's default similarity score is **ZNCC**, zero-mean normalized cross-correlation. It subtracts
each patch's mean and divides by its standard deviation before taking the dot product. The result lies
in `[-1, 1]`: 1 is a perfect match and 0 is no relationship.

The normalization makes the score invariant to brightness and contrast, so two images acquired under
different illumination still match. Scaling one image by any positive factor and adding any constant
leaves every correlation value unchanged, apart from floating-point rounding.

```@example tracking
scaled = 0.2f0 .* reference .+ 3.0f0
scaled_window = scaled[(centre[1] - h - radius):(centre[1] + h + radius - 2),
                       (centre[2] - h - radius):(centre[2] + h + radius - 2)]
AutoRIFT.peak_offset(AutoRIFT.correlate!(ws, scaled_window, chip, radius), (radius, radius))
```

The peak offset is the same for an image five times darker, and the correlation differs by about 2e-6. Because amplitude is normalized away, low contrast does not defeat a small chip.
*Decorrelation* does: texture present in one image and absent in the other because the surface changed
between acquisitions. It limits the correlation a small chip can reach however bright the imagery is.

[`AutoRIFT.NCC`](@ref) omits the mean subtraction. [`AutoRIFT.Coherence`](@ref) is the estimator for
complex-valued input, where phase carries information that amplitude does not.

### Escalating from coherence to amplitude

`similarity` accepts a tuple, which assigns a measure to each chip-size level in order, with the last
repeated for any remaining levels. `(:coherence, :zncc)` tries complex coherence at the finest chip and
falls back to amplitude at every coarser one.

This is the escalation of Joughin (2002). The two measures fail differently. Coherence resolves finer
detail than amplitude because phase varies over a shorter distance than brightness does, but phase
variation across a chip destroys it outright, where amplitude degrades gradually. Pairing them by level
uses coherence where it works and leaves the rest to a larger amplitude chip.

Complex matching improves resolution, not accuracy. Joughin found the complex cross-correlation
function "more strongly peaked" in low-correlation regions, so a match that amplitude needs 64×64 to
achieve is available at 24×24. Where speckle decorrelates quickly, this decides whether a narrow shear
zone is resolved or smoothed over. It does not make the displacement more accurate at a point where
both measures work.

Interferometric phase across a chip can reduce or eliminate the peak, most where the deformation is
largest (high shear, steep topography). Amplitude is unaffected there. [`Deramp`](@ref) removes the
linear component of the phase variation, which is the dominant part and cheap to estimate, and the
tuple handles the rest.

The complex path has no reference implementation to match. autoRIFT v2.1.2 has no complex entry point,
and ISCE2's `cuAmpcor` takes `abs` of complex input before correlating, so both reduce to amplitude
matching. The estimator and the escalation follow Joughin (2002). The implementation is tested against
analytic cases: `γ(T, T) = 1`, a known shift, and a known phase ramp.

Joughin, I. (2002). Ice-sheet velocity mapping: a combined interferometric and speckle-tracking
approach. *Annals of Glaciology* 34, 195–201.

## Sub-pixel refinement

The surface is sampled at whole pixels, so its peak is at a whole pixel, while real displacement is
not. Refinement locates the maximum of the underlying continuous surface from the samples near the peak.

[`PyramidRefine`](@ref), the default, upsamples the 5×5 neighborhood of the peak with a cascade of
Gaussian-pyramid steps and takes the argmax. This is the reference implementation's method. The default
`upsampling = 64` locates the peak to 1/64 pixel. [`NoRefine`](@ref) reports the whole-pixel peak, which
suits correlation so weak that a fractional part would be noise.

```@example tracking
refined = autorift(reference, secondary; chip_size = 32, search_radius = 20, grid_spacing = 32)
whole = autorift(reference, secondary; chip_size = 32, search_radius = 20, grid_spacing = 32,
                 subpixel = NoRefine())

using Statistics: median
mid(A) = median(filter(!isnan, A))
(refined = (mid(refined.dx), mid(refined.dy)), whole = (mid(whole.dx), mid(whole.dy)))
```

The true offset is `(6.4, -3.2)`. Refinement lands within a tenth of a pixel of both components.
Without it the answer is the whole-pixel `(6, -3)`.

## Why the peak can be wrong

A correlation surface has one peak per candidate match, and the tallest is not necessarily the right
one. Three failure modes leave different signatures in the output.

**Weak texture.** A chip with little structure correlates weakly with everything, and noise sets the
tallest peak. `correlation` is low, and it is the layer to gate on.

**Ambiguity.** Periodic texture (dunes, crop rows, regular particle seeding) matches at more than one
offset, and the surface has several peaks of similar height. `correlation` can be high while the answer
is wrong. The `peak_ratio` layer diagnoses this: it is the peak divided by the best rival elsewhere on
the surface, so a value near 1 means the match was not unique.

**Motion beyond the radius.** If the true displacement lies outside the search window, the surface
never contains the right peak and the best available one wins. The result looks like a measurement.
When the peak lands on the edge of the surface, the correlator sets both `peak_ratio` and `correlation`
to zero, so `peak_ratio .== 0` selects the points where `search_radius` was too small.

[Judging a result](@ref) covers which layer to threshold and what the thresholds are worth. In short,
`correlation` is the reliability gate and `peak_ratio` diagnoses *why* a point failed.

## Where the parameters come in

| you control | it sets |
|---|---|
| `chip_size`, `chip_size_max` | patch size, which sets how much texture the patch contains and how much spatial averaging is behind each answer |
| `search_radius` | how far the patch may have moved. It is also the dominant cost, which grows with the *area* searched |
| `dx_prior`, `dy_prior` | where the window is centered, so motion larger than the radius is still reachable |
| `grid_spacing` | how often an answer is produced (sampling, not measurement) |
| `similarity` | the similarity score: `ZNCC`, `NCC` or `Coherence` |
| `preprocess` | the image filtering applied before correlation; see [Preprocessing](@ref AutoRIFT.preprocess) |
| `subpixel` | how the fractional part is recovered |

A single chip size is rarely right for a whole scene, so the default searches several; see
[Multiple chip sizes](@ref).

## Further reading

The algorithm is the one described in Gardner et al. (2018), *Increased West Antarctic and unchanged
East Antarctic ice discharge over the last 7 years*, The Cryosphere 12, 521–547
([doi:10.5194/tc-12-521-2018](https://doi.org/10.5194/tc-12-521-2018)).

- [A guided walkthrough](@ref): the same ideas as eight runnable steps
- [Conventions](@ref): the sign of the offsets and the half-pixel convention
- [Multiple chip sizes](@ref): the nested grid and how levels merge
