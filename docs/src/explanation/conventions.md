
# Conventions

Two conventions cause wrong answers that no error flags: which way is positive, and where a pixel's
center sits. Both produce a plausible field whose residual is zero under uniform motion and grows with
the velocity gradient. The median stays near zero, the scene correlates well, and only a difference
*map* shows the structure, along fast-moving margins where it looks like a physical effect.

This page states what the package does and how to check a convention empirically.

## Array output: `dx` and `dy`

Given two plain arrays, `autorift` returns **pixel offsets in array orientation**:

- `dx` is along the second index, positive toward higher column numbers.
- `dy` is along the first index, positive toward higher row numbers — **downward** in an image.
- Both are the offset from the **secondary image back to the reference**, which is the *negative* of
  how the surface moved.

A feature that moved to the right between acquisitions gives a **negative** `dx`, and one that moved
down gives a **negative** `dy`. This matches the reference implementation. The array path does not
correct it because an array has no orientation to correct to.

The following pair is built by hand, so the direction of motion can be read from the construction:

```@example conventions
using AutoRIFT
include("../../figures.jl")  # plotting helpers and the example scenes; see [Plotting](@ref)
using Statistics: median

# The secondary image is the reference shifted by `motion = (columns, rows)`, so every feature,
# including the bright blob, appears that much further right and further down in it. Cropping from an
# oversized photograph keeps the shift from pulling in out-of-image data.
function moved_pair(n, motion; pad = 40)
    move_col, move_row = motion
    big = photo()[1:(n + 2pad), 1:(n + 2pad)]
    for dr in -8:8, dc in -8:8
        big[pad + n ÷ 2 + dr, pad + n ÷ 3 + dc] += 3 * exp(-(dr^2 + dc^2) / 32)
    end
    rows, cols = (pad + 1):(pad + n), (pad + 1):(pad + n)
    return Float32.(big[rows, cols]), Float32.(big[rows .- move_row, cols .- move_col])
end

reference, secondary = moved_pair(512, (12, 5))
image_panels(reference, secondary)
```

The position of the blob in each image shows the motion:

```@example conventions
(reference = Tuple(argmax(reference)), secondary = Tuple(argmax(secondary)))
```

Row 256 → 261 and column 170 → 182: down 5, right 12. And what `autorift` reports:

```@example conventions
out = autorift(reference, secondary; chip_size = 32, search_radius = 20, grid_spacing = 16)
(dx = median(filter(!isnan, out.dx)), dy = median(filter(!isnan, out.dy)))
```

The blob moved right and down, and both `dx` and `dy` are negative: each is the motion negated. `dy`
also counts rows downward, so the negation and the downward axis happen to give the same sign here.
That coincidence is what makes a y-sign error hard to spot.

Negate both to get motion in array orientation. For motion in map orientation, use a raster, where the
package applies the conversion.

## Raster output: `vx` and `vy`

Given `Raster` input, `autorift` returns a `RasterStack` whose layers are named `vx` and `vy` rather
than `dx`/`dy`. The names record two conversions:

- **Sign.** The offset is negated, so `vx`/`vy` are feature motion.
- **Orientation.** `vy` is north-positive rather than row-positive.

`+vx` points east and `+vy` north, whatever order the file stores its rows in. A north-up GeoTIFF
stores y decreasing and a south-up one stores it increasing. The extension reads the lookup's direction
and corrects for it, so the same scene written either way gives the same field.

With `dt`, the units are distance per year rather than pixels:

```
pixels = (metres per year) × (dt in days / 365.25) / pixel_size
```

Use this conversion to compare a published velocity with a search radius in pixels. The radius has to
cover the fastest motion expected over the pair's own interval.

The y-flip is applied in one place, the [Geospatial](@ref) extension, because only there is the y
direction known. `dx`/`dy` are reserved for raw pixel offsets on the array and `DimStack` paths,
matching ITS_LIVE's published naming. To get the unconverted offsets from a raster, correlate
`parent(raster)`.

## The half pixel

An even-sized chip has no center sample. A chip that extends from `-chip/2` to `chip/2 - 1` about an
integer position has its true center half a pixel past that position, so the package adds `0.5` when it
translates a point into image coordinates. A reported displacement then refers to the chip's center
rather than to a corner.

The same question arises wherever a grid is resampled rather than indexed (pixel-is-area against
pixel-is-point). A coarse level's nodes sit at the **mean coordinate of the cell** they summarize, not
at the cell's first point. Placing them elsewhere measures the field away from where consumers assume
it was measured, as a level-dependent error that no single shift corrects because the cell size
differs per level.

The public API does not require you to handle this. It matters when comparing against arrays captured
from another implementation, which has to account for both implementations' conventions, and because
the symptom is a small coherent offset rather than an error.

## How to settle a convention question

Test a convention against data instead of deriving it. Three checks, in order:

1. **Map the difference before reading any summary statistic.** A sign error, a whole-pixel roll, a
   transposed axis and a half-pixel offset produce four distinct difference maps and near-identical
   medians. A median of one quantization step can be tie-breaking, and has also been a half-pixel grid
   error.

2. **Scan the offset.** Score `+0.0`, `+0.5` and `+1.0` and report which wins. A hardcoded convention
   is correct until something upstream changes, and then fails silently.

3. **Separate a bias from tie-breaking by spatial autocorrelation.** Tie-breaking is independent per
   point, so its residual has lag-1 autocorrelation near zero and block means that shrink as `1/n`. A
   residual that autocorrelates indicates a convention error whatever its magnitude. A y-sign error in
   this package's development measured `+0.80` at lag 1, with block means 13× larger than white noise
   allows.

Argument order and array layout fail the same way: a transposed displacement field still looks like a
displacement field.

## Plotting

`heatmap!` treats an array's **first** index as the x axis, so passing a `[row, col]` field directly
draws it transposed: a feature elongated in y appears elongated in x. Every figure in this
documentation goes through one function (`mapshow` in `docs/figures.jl`) that transposes the array and
reverses its columns so row 1 is at the top.

## See also

- [How feature tracking works](@ref): where the offset comes from
- [A guided walkthrough](@ref): the conventions in use, with figures
- [`autorift`](@ref): the sign note on the array method
