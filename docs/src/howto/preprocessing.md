
# Choosing a preprocessing filter

Correlation compares brightness patterns. Anything that changes brightness *without* the ground moving —
a different sun angle, a gain change, haze, a sensor artifact — is a difference the correlator has to see
past. Preprocessing removes that before the comparison.

One keyword selects it, by symbol or by instance:

```julia
autorift(reference, secondary; preprocess = :highpass)        # the default
autorift(reference, secondary; preprocess = Highpass(; width = 11))
```

## Why the default is not `:none`

The similarity measure already removes some of this. [`ZNCC`](@ref) subtracts the mean and divides by the
standard deviation of each chip, so a uniform brightening or a gain change is invisible to it — a
filter cannot improve on that. What ZNCC cannot absorb is variation *within* a chip, and a gradient
across the scene is exactly that.

Here is a pair where the secondary image carries an illumination gradient the reference does not:

```@example prep
using AutoRIFT
include("../../figures.jl")  # plotting helpers and the synthetic scenes; see [Plotting](@ref)
using Statistics: median, quantile

reference, clean, truth_dx, _ = warped_pair(512, (row, col) -> (6.0, -2.0); seed = 13)
# Brightness rising left to right, plus a weak vertical trend. Only the secondary image has it.
secondary = Float32[clean[row, col] * (0.4f0 + 1.2f0 * (col / 512)) + 0.3f0 * (row / 512)
                    for row in axes(clean, 1), col in axes(clean, 2)]

image_panels(reference, secondary; titles = ("reference", "secondary, gradient added"))
```

The truth is a uniform 6-pixel shift everywhere, so the error is measurable:

```@example prep
settings = (; chip_size = 16, chip_size_max = 64, grid_spacing = 8)
_, grid = AutoRIFT.autorift_with_grid(reference, secondary; settings...)
truth = reshape([truth_dx[round(Int, grid.y[i]), round(Int, grid.x[i])] for i in eachindex(grid.x)],
                Base.size(autorift(reference, secondary; settings...).dx))
error_of(out) = filter(isfinite, abs.(out.dx .- truth))

for method in (:none, :highpass, :wallis, :sobel, :laplacian)
    out = autorift(reference, secondary; preprocess = method, settings...)
    err = error_of(out)
    println(rpad(string(method), 12),
            " measured ", lpad(AutoRIFT.nmeasured(out), 4),
            "   median correlation ",
            round(median(filter(isfinite, out.correlation)); digits = 3),
            "   median |error| ", round(median(err); digits = 3),
            "   p99 ", round(quantile(err, 0.99); digits = 2))
end
```

Read that table carefully, because it contains the one result most likely to mislead you:

**`:none` has the highest correlation of any option and the worst accuracy.** Median correlation 0.965
against `:highpass`'s 0.812, and median error 0.062 px against 0.016 — four times worse, with a 99th
percentile of 0.31 px against 0.05.

The gradient is itself a strong, smoothly varying signal that both chips share, so it inflates the
correlation coefficient while contributing nothing about where the features went. Filtering removes that
shared low-frequency agreement, and the correlation *falls* to reflect what is actually being matched.

So: **correlation is not how you choose a filter.** It compares points within one run, not one filter
against another. Judge a filter by accuracy against known motion if you have it, and by the spatial
coherence of the field if you do not.

```@example prep
none_out = autorift(reference, secondary; preprocess = :none, settings...)
high_out = autorift(reference, secondary; preprocess = :highpass, settings...)
field_panels(blank(none_out.dx, measured(none_out)), blank(high_out.dx, measured(high_out));
             titles = ("dx, preprocess = :none", "dx, preprocess = :highpass"))
```

Both fields look like a uniform 6-pixel shift. The difference is in the tail, not in the picture — which
is the usual case, and the reason to have a truth field or a gate rather than an opinion.

## Filter width has the same inversion

```@example prep
for width in (3, 5, 11, 21)
    out = autorift(reference, secondary; preprocess = Highpass(; width), settings...)
    err = error_of(out)
    println("width ", lpad(width, 2),
            "   median correlation ", round(median(filter(isfinite, out.correlation)); digits = 3),
            "   median |error| ", round(median(err); digits = 3),
            "   p99 ", round(quantile(err, 0.99); digits = 2))
end
```

Correlation climbs steadily with width — 0.664 at 3, up to 0.953 at 21 — while the 99th percentile of the
error nearly doubles, 0.05 to 0.09 px. A wider highpass passes more of the low-frequency content, so it
approaches `:none` from below in both respects. The default of 5 is near the accuracy optimum here.

## Which filter for which imagery

| filter | what it does | reach | reach for |
|---|---|---:|---|
| [`NoPreprocess`](@ref) | nothing; ZNCC alone | 0 | already-normalized imagery, or a deliberate baseline |
| [`Highpass`](@ref) | removes a windowed local mean | 2 | the default. Optical imagery, most cases |
| [`Wallis`](@ref) | normalizes local mean *and* variance | 2 | scenes mixing high- and low-contrast ground — shadow beside snow |
| [`WallisGapfill`](@ref) | `Wallis`, filling gaps with matched noise | 37 | imagery with holes you would rather fill than mask |
| [`Sobel`](@ref) | gradient magnitude | 2 | edge-dominated scenes; discards flat texture |
| [`Laplacian`](@ref) | isotropic second derivative | 2 | as `Highpass`, with a sharper cut |
| [`Decibel`](@ref) | log of amplitude | 0 | **SAR amplitude**, whose dynamic range is multiplicative |
| [`Destripe`](@ref) | notches periodic stripes in the Fourier domain | 0 | **push-broom detector striping** |
| [`Deramp`](@ref) | removes the linear phase ramp | **-1** | **complex (SLC) input only** |

The last three are domain-specific by construction: a decibel transform on optical reflectance, or a
deramp on real-valued imagery, is not a tuning choice but a category error. `Destripe` also requires
explicit `along_track` and `cross_track` keywords — there is no sensible default for which axis carries
the stripes.

## The reach column, and why `Deramp` cannot be blocked

[`AutoRIFT.filter_reach`](@ref) is how many pixels beyond a region must be read for the filter's output
*inside* that region to match what filtering the whole scene would give. Blocked processing sizes its halo
from it:

```@example prep
for method in (NoPreprocess(), Highpass(), Highpass(; width = 21), Wallis(), Sobel(), Laplacian())
    println(rpad(string(method), 22), " reach ", AutoRIFT.filter_reach(method))
end
```

Note that reach is not the window's half-width: `WallisGapfill`'s 37 comes from its fill *decisions*
depending on how far the nearest real data lies, which no window bounds.

`Deramp` returns **-1**, meaning no finite reach. It estimates from the whole image, so no halo can
reproduce it. Check the sign before using a reach as a width — [`AutoRIFT.halo`](@ref) throws rather than
silently returning a halo shorter than the correlation alone. In practice: `Deramp` and
`process_block_size` are incompatible, and blocking a complex scene means deramping it yourself first.

## Preprocessing is per-pair, not per-image

Both images get the same filter, and the filtered result is cached on the pair — so reusing one image
across several pairs does not re-filter it. That is what makes a time series cheap; see
[Correlating many pairs](@ref).

## See also

- [Judging a result](@ref) — the gate to use once a filter is chosen, and why `correlation` is not it here
- [Masking invalid pixels](@ref) — filters respect the mask, and their reach spreads a masked region
- [How feature tracking works](@ref) — where preprocessing sits relative to the correlation itself
- [`AutoRIFT.preprocess`](@ref) — the filter reference, and every filter's full docstring
