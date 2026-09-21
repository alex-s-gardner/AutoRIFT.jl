
# Conventions

Two questions account for more wrong answers than any algorithmic choice: which way is positive, and
where does a pixel's centre sit. Both fail the same way — no error, a plausible field, and a residual
that is zero under uniform motion and grows with the velocity gradient. That is why they survive
review: the median stays near zero, the scene correlates above 0.99, and only a difference *map* shows
the structure, always along fast-moving margins where it reads as a physical effect rather than a bug.

This page states what the package does, and how to check it rather than reason about it.

## Array output: `dx` and `dy`

Given two plain arrays, `autorift` returns **pixel offsets in array orientation**:

- `dx` is along the second index, positive toward higher column numbers.
- `dy` is along the first index, positive toward higher row numbers — **downward** in an image.
- Both are the offset from the **secondary image back to the reference**, which is the *negative* of
  how the surface moved.

The last one is the surprise. A feature that moved to the right between acquisitions gives a
**positive** `dx`; one that moved down gives a **negative** `dy`. This matches the reference
implementation, and the package leaves it uncorrected on the array path because an array has no
orientation to be corrected about.

A pair built by hand rather than by a helper, so the direction of motion is read off the construction:

```@example conventions
using AutoRIFT
include("../../figures.jl")  # plotting helpers and the synthetic scenes; see [Plotting](@ref)
using Statistics: median

# The secondary image is the reference shifted by `motion = (columns, rows)`, so every feature — the
# bright blob included — appears that much further right and further down in it. Cropping from an
# oversized texture is what keeps the shift from pulling in out-of-image data.
function moved_pair(n, motion; pad = 40)
    move_col, move_row = motion
    big = texture((n + 2pad, n + 2pad); seed = 4)
    for dr in -8:8, dc in -8:8
        big[pad + n ÷ 2 + dr, pad + n ÷ 3 + dc] += 3 * exp(-(dr^2 + dc^2) / 32)
    end
    rows, cols = (pad + 1):(pad + n), (pad + 1):(pad + n)
    return Float32.(big[rows, cols]), Float32.(big[rows .- move_row, cols .- move_col])
end

reference, secondary = moved_pair(512, (12, 5))
image_panels(reference, secondary)
```

The blob is the check. Its position in each image:

```@example conventions
(reference = Tuple(argmax(reference)), secondary = Tuple(argmax(secondary)))
```

Row 256 → 261 and column 170 → 182: down 5, right 12. And what `autorift` reports:

```@example conventions
out = autorift(reference, secondary; chip_size = 32, search_radius = 20, grid_spacing = 16)
(dx = median(filter(!isnan, out.dx)), dy = median(filter(!isnan, out.dy)))
```

Moved right and down; reported negative `dx` and negative `dy`. Both are the surface motion negated,
and because `dy` also counts rows downward the two conventions happen to agree in sign here — which is
exactly the coincidence that makes a y-sign error hard to spot.

Negate both to get motion in array orientation. To get motion in a *map* orientation, use a raster —
which is where the package does it for you.

## Raster output: `vx` and `vy`

Given `Raster` input, `autorift` returns a `RasterStack` whose layers are named `vx` and `vy` rather
than `dx`/`dy`, and the names record that two conversions happened:

- **Sign.** The offset is negated, so `vx`/`vy` are feature motion rather than its opposite.
- **Orientation.** `vy` becomes north-positive rather than row-positive.

`+vx` points east and `+vy` north, whatever order the file stored its rows in. A north-up GeoTIFF
stores y decreasing and a south-up one stores it increasing; the extension reads the lookup's
direction and corrects for it, so the same scene written both ways yields the identical field.

With `dt`, the units become distance per year rather than pixels:

```
pixels = (metres per year) × (dt in days / 365.25) / pixel_size
```

That is the conversion to reach for when comparing a published velocity against a search radius in
pixels — a radius has to cover the fastest motion expected over the pair's own interval.

The flip lives in exactly one place — the [Geospatial](@ref) extension — because that is the only
place the y direction is actually known. `dx`/`dy` stay reserved for raw pixel offsets on the array and `DimStack`
paths, matching ITS_LIVE's published naming. A caller who wants the unconverted offsets from a raster
can correlate `parent(raster)`.

## The half pixel

An even-sized chip has no centre sample. Extending `-chip/2` to `chip/2 - 1` about an integer position
puts the chip's true centre half a pixel past that position, so the package adds `0.5` when it
translates a point into image coordinates. That is what makes a reported displacement refer to the
chip's centre rather than to a corner of it.

The same question recurs wherever a grid is *resampled* rather than indexed, and it is the
pixel-is-area against pixel-is-point distinction in correlator form. A coarse level's nodes sit at the
**mean coordinate of the cell** they summarize, not at the cell's first point. Getting that wrong
measures the field half a cell from where every consumer assumes it was measured — which appears as a
level-dependent error that no single shift corrects, because the cell size differs per level.

Nothing in the public API requires you to handle this; it is stated because a harness comparing
against arrays captured from another implementation has to account for both sides' conventions, and
because the symptom is a small coherent offset rather than an error.

## How to settle a convention question

Do not reason about it — the reasoning is what fails. Three cheap checks, in order:

1. **Map the difference before believing any summary statistic.** A sign error, a whole-pixel roll, a
   transposed axis and a half-pixel offset produce four distinct pictures and near-identical medians. A
   median of one quantization step is entirely plausible as tie-breaking and has been a half-pixel grid
   error.

2. **Scan the offset rather than deriving it.** Score `+0.0`, `+0.5` and `+1.0` and report which wins.
   A hardcoded convention is correct until something upstream changes, and then fails silently.

3. **Distinguish a bias from tie-breaking by spatial autocorrelation.** Tie-breaking is independent per
   point, so its residual must have lag-1 autocorrelation near zero and block means that shrink as
   `1/n`. A residual that autocorrelates is a convention error whatever its magnitude — a sign bug
   found in this package measured `+0.80` at lag 1, with block means 13× larger than white noise
   allows.

The same discipline applies to argument order and array layout, which fail identically: a transposed
displacement field still reads as a displacement field.

## Plotting

One more orientation trap, on the way out rather than in. `heatmap!` treats an array's **first** index
as the x axis, so passing a `[row, col]` field directly draws it transposed — a feature elongated in y
appears elongated in x, and anisotropy reads inverted. Every figure in this documentation goes through
one function that transposes and flips; see [Plotting](@ref).

## See also

- [How feature tracking works](@ref) — where the offset comes from
- [A guided walkthrough](@ref) — the conventions in use, with figures
- [`autorift`](@ref) — the sign note on the array method
