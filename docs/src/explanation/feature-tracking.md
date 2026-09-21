
# How feature tracking works

Two images of the same scene, taken at different times. Something in the scene moved. Feature
tracking recovers how much, everywhere, by matching small patches of one image against the other.

There is no feature detector and no correspondence problem. The method takes the texture already
present in the imagery — grain, crevasses, speckle, seeded particles, whatever the surface happens to
look like — and asks where each patch of it went. That is what makes it dense: every grid point gets
an answer, not just the points where something distinctive happened to be.

## The chip, the window, and the surface

Three objects, and every parameter in the package configures one of them.

A **chip** is a square patch cut from one image at a grid point. A **search window** is a larger patch
cut from the other image around the same position, reaching `search_radius` pixels further in each
direction. The chip is slid over every position inside the window, and at each one a similarity score
is computed. Those scores make up the **correlation surface**.

```@example tracking
using AutoRIFT
include("../../figures.jl")  # plotting helpers and the synthetic scenes; see [Plotting](@ref)

reference, secondary, _, _ = warped_pair(512, (row, col) -> (6.4, -3.2); seed = 5)

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

The bright spot in the third panel is the answer. Its offset from the surface's centre is the
displacement:

```@example tracking
AutoRIFT.peak_offset(surface, (radius, radius))
```

The offset in x, the offset in y, and the correlation there. The pair was built with an offset of
`(6.4, -3.2)`, so whole pixels get within one and the fractional part is still missing. That is what
the sub-pixel stage is for.

This reaches past the public API — a correlation surface is not part of what `autorift` returns, since
keeping one per point would cost more memory than the result itself. It is shown here because it is
the object every parameter is really about.

This is the whole algorithm at one point. Everything else in the package is about doing it well: at
what size, over what radius, on how many points, at what cost, and how to tell a good answer from a
bad one.

!!! note "Other names for the same three things"
    Particle image velocimetry calls the chip an *interrogation window* and the correlation surface a
    *correlation map*. Digital image correlation calls the chip a *subset* and the grid spacing a
    *step size*. Optical flow literature calls the whole operation *block matching*. The objects are
    the same; only the vocabulary differs.

## Normalized cross-correlation

The similarity score is what makes a match meaningful. AutoRIFT's default is **ZNCC** — zero-mean
normalized cross-correlation — which subtracts each patch's mean and divides by its standard
deviation before taking the dot product. The result lies in `[-1, 1]`: 1 is a perfect match, 0 is no
relationship.

The normalization is the point. It makes the score invariant to brightness and contrast, so two
images acquired under different illumination still match. Concretely: scaling one image by any
positive factor and adding any constant leaves every correlation value unchanged.

```@example tracking
scaled = 0.2f0 .* reference .+ 3.0f0
scaled_window = scaled[(centre[1] - h - radius):(centre[1] + h + radius - 2),
                       (centre[2] - h - radius):(centre[2] + h + radius - 2)]
AutoRIFT.peak_offset(AutoRIFT.correlate!(ws, scaled_window, chip, radius), (radius, radius))
```

The same answer, from an image five times darker. One practical consequence worth knowing: because
amplitude is normalized away, **low contrast is not what defeats a small chip**. What defeats it is
*decorrelation* — texture that is present in one image and not the other, because the surface itself
changed between acquisitions. That lowers the correlation a small chip can reach no matter how bright
the imagery is.

[`AutoRIFT.NCC`](@ref) skips the mean subtraction, and [`AutoRIFT.Coherence`](@ref) is the estimator
for complex-valued input, where phase carries the information amplitude does not.

### Escalating from coherence to amplitude

`similarity` accepts a tuple, which assigns a measure to each chip-size level in order with the last
repeated for any remaining levels. `(:coherence, :zncc)` therefore tries complex coherence at the
finest chip and falls back to amplitude at every coarser one.

That is the escalation of Joughin (2002), and the reason it is worth arranging is that the two
measures fail differently. Coherence resolves finer detail than amplitude because phase varies over a
shorter distance than brightness does — but phase variation across a chip destroys it outright, where
amplitude degrades gracefully. Pairing them by level spends coherence where it can win and leaves
what it cannot resolve to a larger amplitude chip, rather than choosing one measure for the whole
scene.

**Complex matching buys resolution, not accuracy.** Joughin found the complex cross-correlation
function "more strongly peaked" in low-correlation regions, so a match amplitude needs 64×64 to
achieve is available at 24×24. Where speckle decorrelates quickly, that difference decides whether a
narrow shear zone is resolved or smoothed over. It does not make the displacement at a point where
both work any more accurate.

The cost is where it fails. Interferometric phase across a chip can reduce or eliminate the peak
outright, and that is worst exactly where the deformation is largest — high shear, steep
topography — which is where amplitude is unaffected. [`Deramp`](@ref) removes the linear component of
that phase variation, being the part that is both dominant and cheap to estimate, and the tuple
handles the rest.

There is no reference implementation of the complex path to match, which is worth stating plainly.
autoRIFT v2.1.2 has no complex entry point, and ISCE2's `cuAmpcor` takes `abs` of complex input
before correlating, so both reduce to amplitude matching. The estimator and the escalation follow
Joughin (2002); the implementation is verified against analytic cases — `γ(T, T) = 1`, a known shift,
a known phase ramp — rather than against another program's output.

Joughin, I. (2002). Ice-sheet velocity mapping: a combined interferometric and speckle-tracking
approach. *Annals of Glaciology* 34, 195–201.

## Sub-pixel refinement

The surface is sampled at whole pixels, so its peak is a whole pixel. Real displacement is not.
Refinement fits the shape of the surface near its peak and reports where the underlying continuous
maximum lies.

[`PyramidRefine`](@ref) — the default — upsamples the neighbourhood of the peak and re-locates it,
twice, which is the method the reference implementation uses and reaches roughly 1/16-pixel
quantization. [`NoRefine`](@ref) reports the whole-pixel peak, which is the honest output when the
correlation is weak enough that a fractional part would be noise.

```@example tracking
refined = autorift(reference, secondary; chip_size = 32, search_radius = 20, grid_spacing = 32)
whole = autorift(reference, secondary; chip_size = 32, search_radius = 20, grid_spacing = 32,
                 subpixel = NoRefine())

using Statistics: median
mid(A) = median(filter(!isnan, A))
(refined = (mid(refined.dx), mid(refined.dy)), whole = (mid(whole.dx), mid(whole.dy)))
```

The truth is `(6.4, -3.2)`. Refinement lands within a sixteenth of a pixel of both; without it the
answer is the whole-pixel `(6, -3)`.

## Why the peak can be wrong

A correlation surface has one peak per candidate match, and nothing guarantees the tallest one is
the right one. Three failure modes, each with its own signature in the output.

**Weak texture.** A chip with little structure correlates weakly with everything, and the tallest
peak is set by noise. `correlation` is low, and that is the layer to gate on.

**Ambiguity.** Periodic texture — dunes, crop rows, a regular particle seeding — matches at more than
one offset, and the surface has several peaks of similar height. `correlation` can be high while the
answer is wrong. The result's `peak_ratio` layer is the diagnostic: the peak divided by the best rival
elsewhere on the surface, so near 1 means the match was not unique.

**Motion beyond the radius.** If the true displacement lies outside the search window, the surface
never contains the right peak and the best available one wins. This is the failure that looks most
like a measurement. The tell is that `peak_ratio` is exactly zero, which the correlator sets when the
peak lands against the search boundary — so `peak_ratio .== 0` finds every point where
`search_radius` was too small.

[Judging a result](@ref) covers which layer to threshold and what the thresholds are worth; the short
version is that `correlation` is the reliability gate and `peak_ratio` is for diagnosing *why* a
point failed.

## Where the parameters come in

| you control | it sets |
|---|---|
| `chip_size`, `chip_size_max` | how large the patch is, and so how much texture it contains and how much spatial averaging is behind each answer |
| `search_radius` | how far the patch may have moved — and the dominant cost, which grows with the *area* searched |
| `dx_prior`, `dy_prior` | where the window is centred, so motion larger than the radius is still reachable |
| `grid_spacing` | how often an answer is produced, which is sampling and not measurement |
| `measure` | the similarity score: `ZNCC`, `NCC` or `Coherence` |
| `preprocess` | what the images look like before any of this — see [Preprocessing](@ref AutoRIFT.preprocess) |
| `subpixel` | how the fractional part is recovered |

A single chip size is rarely right for a whole scene, which is why the default searches several. That
mechanism is [Multiple chip sizes](@ref).

## Further reading

The algorithm is the one described in Gardner et al. (2018), *Increased West Antarctic and unchanged
East Antarctic ice discharge over the last 7 years*, The Cryosphere 12, 521–547
([doi:10.5194/tc-12-521-2018](https://doi.org/10.5194/tc-12-521-2018)).

- [A guided walkthrough](@ref) — the same ideas as eight runnable steps
- [Conventions](@ref) — signs and the half pixel, which is where correctness actually goes wrong
- [Multiple chip sizes](@ref) — the nested grid and how levels merge
