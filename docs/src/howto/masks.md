
# Masking invalid pixels

Imagery can contain pixels that are present but not usable: cloud, shadow, saturation, water, the ragged
edge of a swath, the no-data border left by a reprojection. Correlating their values against real texture
produces a displacement with no basis.

Two keywords exclude them. Each takes an array or a raster, and either may be omitted:

```julia
autorift(reference, secondary; reference_valid = a_bool_matrix, secondary_valid = another)
```

`true` means usable.

## Non-finite pixels are masked by default

Without a mask, [`ImagePair`](@ref) treats a `NaN` or infinite pixel as invalid. If your reader already
converts no-data to `NaN`, no further masking is needed:

```@example masks
using AutoRIFT
include("../../figures.jl")  # plotting helpers and the example scenes; see [Plotting](@ref)
using Statistics: median, quantile

reference, secondary, truth_dx, _ = warped_pair(photo(), 512, (row, col) -> (20.0, -8.0))
settings = (; chip_size = 32, chip_size_max = 32, search_radius = 28, grid_spacing = 8)

patch = (236:276, 236:276)   # a 41x41 cloud in the reference image

explicit = trues(Base.size(reference))
explicit[patch...] .= false

as_nan = Float32.(copy(reference))
as_nan[patch...] .= NaN32

(mask = AutoRIFT.nmeasured(autorift(reference, secondary; reference_valid = explicit, settings...)),
 nan = AutoRIFT.nmeasured(autorift(as_nan, secondary; settings...)))
```

The counts are equal because the two routes give the same mask. The default does not treat zero as
no-data, since zero is a valid dark radiance and treating it as no-data would discard dark ground. A
sensor that writes `0` or `-9999` needs an explicit mask.

```@example masks
image_panels(reference, secondary)
```

```@example masks
input_field(Float64.(explicit), "reference_valid (false = masked)"; colormap = :grays)
```

## A masked chip is skipped when most of it is masked

A point is correlated only if more than `min_chip_valid_fraction` (0.65 by default) of its own chip is
valid. The search window around it may hold any amount of masked ground. The fraction is taken from the
mask after preprocessing, which a filter with spatial extent has enlarged (see
[Filters and masks interact](@ref)).

The hole in the output is therefore smaller than the set of points whose chip overlaps the mask, and a
larger chip has a smaller fraction of itself masked by a given cloud:

```@example masks
for chip in (16, 32, 64)
    one_level = (; chip_size = chip, chip_size_max = chip, search_radius = 28, grid_spacing = 8)
    base = autorift(reference, secondary; one_level...)
    masked = autorift(reference, secondary; reference_valid = explicit, one_level...)
    println("chip ", lpad(chip, 2), ": measured ", AutoRIFT.nmeasured(base), " → ",
            AutoRIFT.nmeasured(masked),
            "   lost ", AutoRIFT.nmeasured(base) - AutoRIFT.nmeasured(masked))
end
```

Every size loses a few dozen points, for different reasons. A 16-pixel chip centered in or near the
41×41 cloud is mostly masked, so those points have little valid data to correlate. A 64-pixel chip is at
most 41% masked by the cloud (about 49% once the default 5-pixel highpass widens the mask), so the chip
test removes only the points nearest the cloud's center. The remaining losses at that size come from
other stages, such as the outlier filter, which the counts above do not separate. A multi-chip-size
search can fill a masked region: with `chip_size = 16, chip_size_max = 64`, the 64-pixel level can answer
over the cloud where the 16-pixel level could not, and those points carry their own `chip_size`. See
[Multiple chip sizes](@ref).

## What a partially masked chip reports

Masked pixels become zeros in the filtered scene, and the mask records that they are not measurements.
On a chip that is partly masked, the error does not rise:

```@example masks
masked = autorift(reference, secondary; reference_valid = explicit, settings...)
unmasked, grid = AutoRIFT.autorift_with_grid(reference, secondary; settings...)
truth = reshape([truth_dx[round(Int, grid.y[i]), round(Int, grid.x[i])] for i in eachindex(grid.x)],
                Base.size(masked.dx))

# the masked fraction of each point's own chip
chip = 32
covered = map(eachindex(grid.x)) do i
    rows = (round(Int, grid.y[i]) - chip ÷ 2) .+ (0:(chip - 1))
    cols = (round(Int, grid.x[i]) - chip ÷ 2) .+ (0:(chip - 1))
    n = count(!explicit[r, c] for r in rows, c in cols if checkbounds(Bool, explicit, r, c))
    return n / (chip * chip)
end
covered = reshape(covered, Base.size(masked.dx))

# Points measured both with and without the mask, so each bin compares the same chips.
ok = measured(masked) .& measured(unmasked)
for (lo, hi) in ((0.0, 0.1), (0.1, 0.3), (0.3, 0.6))
    sel = (covered .> lo) .& (covered .<= hi) .& ok
    count(sel) == 0 && continue
    err = filter(isfinite, abs.(masked.dx[sel] .- truth[sel]))
    corr(out) = round(median(filter(isfinite, out.correlation[sel])); digits = 3)
    println("chip ", lpad("$(round(Int, 100lo))-$(round(Int, 100hi))% masked", 12),
            "  n = ", lpad(count(sel), 3),
            "   median correlation ", corr(unmasked), " → ", corr(masked),
            "   median |error| ", round(median(err); digits = 3), " px")
end
```

The median error is within one refinement quantization step in both bins. The median correlation of the
same chips changes little, because only chips more than 65% valid are correlated at all. The chip
validity test, not a `correlation` threshold, is what excludes heavily masked chips; see
[Judging a result](@ref) for the threshold.

## What happens if you do not mask

A fill value left in the imagery loses more points than masking it. The effect is local to the chips that
overlap the fill:

```@example masks
as_fill = Float32.(copy(reference))
as_fill[patch...] .= -9999.0f0

overlapping = (covered .> 0)
function near_patch(label, out)
    sel = overlapping .& measured(out)
    corr = filter(isfinite, out.correlation[sel])
    println(rpad(label, 22), " measured beside the patch: ", lpad(count(sel), 3),
            isempty(corr) ? "" : "   median correlation $(round(median(corr); digits = 3))")
end
near_patch("explicit mask", masked)
near_patch("-9999 left as data", autorift(as_fill, secondary; settings...))
```

With the mask, 31 points whose chip overlaps the patch are measured. With the fill value left as data,
21 are, and they are the points whose chips lie mostly outside the fill. A chip with no texture, such as
one wholly inside a constant block, has a degenerate correlation, and `autorift` reports no measurement for
it. Where the Python reference meets such a chip,
it reports the search-window corner, a systematic bias in one direction over masked ground.

A block of near-constant noise is not degenerate, so it produces arbitrary displacements. Mask it.

## Where a mask does not reach

- **Mask each image where it is invalid.** `reference_valid` and `secondary_valid` are independent. The
  chip validity test uses their intersection ([`AutoRIFT.valid`](@ref) of the pair), but each image is
  filtered with its own mask, so a cloud masked in only one image is still correlated as data in the
  other.
- **A mask is not an outlier filter.** It excludes pixels you know about. Points that fail for other
  reasons are handled by [Filtering outliers](@ref).
- **Masks do not move.** A cloud shadow that moves between acquisitions must be masked in both images at
  both positions.
- **A lazy raster's `missingval` is masked automatically.** A raster read with `lazy = true` has its
  nodata value turned into a mask rather than a dark measurement, with no keyword needed. See
  [Correlating scenes larger than memory](@ref).

## Filters and masks interact

Preprocessing runs before correlation. A filter with spatial extent treats an output computed partly from
masked pixels as invalid, so the mask grows by the filter width. Output is zeroed outside the valid
domain, and the grown mask is what the chip validity test sees. [`AutoRIFT.filter_reach`](@ref) reports
the reach in pixels, and a wide [`Highpass`](@ref) invalidates more ground around a masked region than a
narrow one. [`WallisGapfill`](@ref) takes the other approach for interior gaps: it fills them with noise
matched to the surrounding statistics rather than zeroing them.

## See also

- [Judging a result](@ref): thresholding on `correlation`
- [Filtering outliers](@ref): rejecting points a mask could not anticipate
- [Choosing a preprocessing filter](@ref): `filter_reach` and the gap-filling option
- [`ImagePair`](@ref): the finiteness default, and that zero is not treated as no-data
