
# Choosing a preprocessing filter

Correlation compares brightness patterns. A different sun angle, a gain change, haze or a sensor artifact
changes brightness without the ground moving, and the correlator has to see past it. Preprocessing
removes that before the comparison.

One keyword selects the filter, by symbol or by instance:

```julia
autorift(reference, secondary; preprocess = :highpass)        # the default
autorift(reference, secondary; preprocess = Highpass(; width = 11))
```

## Why the default is not `:none`

The similarity measure already removes some of this. [`ZNCC`](@ref) subtracts the mean and divides by the
standard deviation of each chip, so a uniform brightening or a gain change is invisible to it. It cannot
absorb variation *within* a chip, and a gradient across the scene is that kind of variation.

Here is a pair where the secondary image carries an illumination gradient the reference does not:

```@example prep
using AutoRIFT
include("../../figures.jl")  # plotting helpers and the example scenes; see [Plotting](@ref)
using Statistics: median, quantile

reference, clean, truth_dx, _ = warped_pair(photo(), 512, (row, col) -> (6.0, -2.0))
# Brightness rising left to right, plus a weak vertical trend. Only the secondary image has it.
secondary = Float32[clean[row, col] * (0.4f0 + 1.2f0 * (col / 512)) + 0.3f0 * (row / 512)
                    for row in axes(clean, 1), col in axes(clean, 2)]

image_panels(reference, secondary; titles = ("reference", "secondary, gradient added"))
```

The true shift is `dx = 6` pixels everywhere, so the error can be measured:

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

`:none` has a higher median correlation (0.838) than `:highpass` (0.582), `:wallis` (0.53) and
`:laplacian` (0.702). Only `:sobel` (0.863) is higher. It also has the largest 99th percentile error, 0.09
px against 0.03 to 0.06 px for the filters, and a median error of 0.016 px against 0.0 for `:highpass` and
`:wallis`.

Correlation therefore does not rank filters. Part of the unfiltered correlation comes from broad
brightness structure common to both images, which says little about where small features moved. Filtering
removes it and the coefficient falls. Compare filters by error against known motion, or, without a truth
field, by the spatial coherence of the result.

## What each filter does

Each panel below is the gradient-added secondary image after one filter, on its own grayscale because
the outputs differ in scale:

```@example prep
mask = trues(size(secondary))
methods = (("none", NoPreprocess()), ("highpass", Highpass()), ("wallis", Wallis()),
           ("sobel", Sobel()), ("laplacian", Laplacian()))

image_panels([first(AutoRIFT.preprocess(secondary, mask, m)) for (_, m) in methods]...;
             titles = first.(methods), size = (1700, 380), shared = false)
```

`:none` shows the gradient as the dominant structure in the frame. The filters differ in how much of the
gradient's effect on contrast they remove. The ratio below is the standard deviation of the right third of
the image over the left third:

```@example prep
using Statistics: std
for (name, m) in methods
    f = first(AutoRIFT.preprocess(secondary, mask, m))
    left, right = std(@view f[:, 1:170]), std(@view f[:, 343:512])
    println(rpad(name, 11), " contrast right/left ", round(right / left; digits = 2))
end
```

The gradient multiplies the texture, so it makes the right side both brighter and higher in contrast.
`:highpass` subtracts a local mean. That removes the brightness difference and leaves the contrast ratio at
2.2, above the unfiltered 1.8. `:wallis` also normalizes local variance and gives 1.0. `:sobel` keeps most
of the trend (2.1). `:laplacian` gives 1.1 because it takes the logarithm first, which turns a
multiplicative gradient into an additive one that the second derivative removes.

```@example prep
none_out = autorift(reference, secondary; preprocess = :none, settings...)
high_out = autorift(reference, secondary; preprocess = :highpass, settings...)
field_panels(blank(none_out.dx, measured(none_out)), blank(high_out.dx, measured(high_out));
             titles = ("dx, preprocess = :none", "dx, preprocess = :highpass"))
```

Both fields show a uniform 6-pixel shift. The difference is in the tail of the error distribution, which a
map does not show.

## Filter width

Correlation and error also trade off with the width of `Highpass`:

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

Correlation rises with width, from 0.437 at 3 to 0.812 at 21, while the 99th percentile error doubles,
from 0.03 to 0.06 px. A wider highpass passes more low-frequency content, so it moves toward `:none` in
both respects. The default width of 5 has a 99th percentile error 0.02 px above the most accurate width
here, which is 3.

## Which filter for which imagery

| filter | what it does | reach | use for |
|---|---|---:|---|
| [`NoPreprocess`](@ref) | nothing; ZNCC alone | 0 | already-normalized imagery, or a baseline |
| [`Highpass`](@ref) | subtracts a windowed local mean | 2 | the default; optical imagery, most cases |
| [`Wallis`](@ref) | normalizes local mean and variance | 4 | scenes mixing high- and low-contrast ground, such as shadow beside snow |
| [`WallisGapfill`](@ref) | `Wallis`, filling interior gaps with matched noise | 37 | imagery with interior holes, such as Landsat 7 after the scan line corrector failure |
| [`Sobel`](@ref) | sum of the x and y Sobel derivatives; directional, not a gradient magnitude | 2 | edge-dominated scenes; discards flat texture |
| [`Laplacian`](@ref) | second derivative of the decibel image | 2 | radar amplitude |
| [`Decibel`](@ref) | `20 log10` of amplitude | 0 | **SAR amplitude**, whose dynamic range is multiplicative |
| [`Destripe`](@ref) | rejects banding in the Fourier domain | **-1** | Landsat 4 and 5 scan-direction banding |
| [`Deramp`](@ref) | removes the linear phase ramp | **-1** | **complex (SLC) input only** |

`Decibel`, `Laplacian`, `Destripe` and `Deramp` are tied to a data type. `Decibel` and `Laplacian` mask
pixels whose amplitude is not positive. `Deramp` throws on real input, and the windowed amplitude filters
throw on complex input. `Destripe` requires explicit `along_track` and `cross_track` keywords, the scan
directions in degrees on the raster's grid.

## The reach column, and why `Deramp` and `Destripe` cannot be blocked

[`AutoRIFT.filter_reach`](@ref) is how many pixels beyond a region must be read for the filter's output
inside that region to match what filtering the whole scene would give. It is an upper bound. Blocked
processing sizes its halo from it:

```@example prep
for method in (NoPreprocess(), Highpass(), Highpass(; width = 21), Wallis(), Sobel(), Laplacian())
    println(rpad(string(method), 22), " reach ", AutoRIFT.filter_reach(method))
end
```

Reach is not the window's half-width. `WallisGapfill`'s 37 comes from its fill decisions, which depend on
how far the nearest real data lies, and no window bounds that.

`Deramp` and `Destripe` return **-1**, meaning no finite reach. Each estimates from the whole image, so no
halo can reproduce the result. Check the sign before using a reach as a width. [`AutoRIFT.halo`](@ref)
throws on a negative reach rather than returning a halo shorter than the correlation alone. Neither filter
can be combined with `process_block_size`; apply it to the whole scene yourself before blocking.

## Filtering is reused across pairs in a cache

Both images get the same filter. A [`AutoRIFT.Cache`](@ref) keeps the filtered result of each image, and
[`reinit!`](@ref) reuses it for an image that is already in either slot, so a time series filters each
acquisition once. See [Correlating many pairs](@ref). A run with `process_block_size` filters each block
from the raw input during the run instead.

## See also

- [Judging a result](@ref): the correlation gate to apply once a filter is chosen
- [Masking invalid pixels](@ref): filters respect the mask, and their reach spreads a masked region
- [How feature tracking works](@ref): where preprocessing sits relative to the correlation itself
- [`AutoRIFT.preprocess`](@ref): the filter reference, and every filter's full docstring
