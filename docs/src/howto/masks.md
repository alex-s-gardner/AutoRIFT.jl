
# Masking invalid pixels

Imagery has pixels that are present but not usable: cloud, shadow, saturation, water, the ragged edge of
a swath, the no-data border left by a reprojection. They carry a number, and correlating that number
against real texture produces a displacement with nothing behind it.

Two keywords exclude them, and both take an array or a raster:

```julia
autorift(reference, secondary; reference_valid = a_bool_matrix, secondary_valid = another)
```

`true` means usable. Either may be omitted.

## The default already handles non-finite pixels

Without a mask, [`ImagePair`](@ref) derives one from finiteness — a pixel that is `NaN` or infinite is
invalid. So if your reader already converts no-data to `NaN`, masking is done:

```@example masks
using AutoRIFT
include("../../figures.jl")  # plotting helpers and the synthetic scenes; see [Plotting](@ref)
using Statistics: median, quantile

reference, secondary, truth_dx, _ = warped_pair(512, (row, col) -> (6.0, -2.0); seed = 13)
settings = (; chip_size = 32, chip_size_max = 32, search_radius = 20, grid_spacing = 8)

patch = (236:276, 236:276)   # a 41x41 cloud in the reference image

explicit = trues(Base.size(reference))
explicit[patch...] .= false

as_nan = Float32.(copy(reference))
as_nan[patch...] .= NaN32

(mask = AutoRIFT.nmeasured(autorift(reference, secondary; reference_valid = explicit, settings...)),
 nan = AutoRIFT.nmeasured(autorift(as_nan, secondary; settings...)))
```

The same count, because they are the same mask by a different route. What the default does **not** do is
treat zero as no-data — zero is a legitimate dark radiance, and conflating the two throws away genuinely
dark ground. A sensor that writes `0` or `-9999` needs an explicit mask.

```@example masks
input_field(Float64.(explicit), "reference_valid (false = masked)"; colormap = :grays)
```

## A masked chip is skipped only when it is entirely masked

This is the rule that governs everything else on this page. A point is abandoned when its chip contains
**no** valid pixel; a chip with even one valid pixel is correlated.

So the hole in the output is *smaller* than you would get from excluding any overlap, and a bigger chip
loses less ground to a small cloud, not more — a large chip is harder to cover completely:

```@example masks
for chip in (16, 32, 64)
    one_level = (; chip_size = chip, chip_size_max = chip, search_radius = 20, grid_spacing = 8)
    base = autorift(reference, secondary; one_level...)
    masked = autorift(reference, secondary; reference_valid = explicit, one_level...)
    println("chip ", lpad(chip, 2), ": measured ", AutoRIFT.nmeasured(base), " → ",
            AutoRIFT.nmeasured(masked),
            "   lost ", AutoRIFT.nmeasured(base) - AutoRIFT.nmeasured(masked))
end
```

Chip 64 loses nothing at all: no 64×64 chip fits inside a 41×41 hole, so every chip keeps some valid
ground. That also means a multi-chip-size search *recovers* a masked region — the coarse level answers
where the fine level had nothing to work with, and those points come back marked with their own
`chip_size`. See [Multiple chip sizes](@ref).

## What a partially masked chip reports

Masked pixels become zeros in the filtered scene, and the mask is what records that they are not
measurements. The effect on a partially covered chip is a loss of *confidence*, not of accuracy:

```@example masks
masked = autorift(reference, secondary; reference_valid = explicit, settings...)
_, grid = AutoRIFT.autorift_with_grid(reference, secondary; settings...)
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

ok = measured(masked)
for (lo, hi) in ((0.0, 0.0), (0.0, 0.1), (0.1, 0.3), (0.3, 0.6))
    sel = lo == hi ? (covered .== 0) .& ok : (covered .> lo) .& (covered .<= hi) .& ok
    count(sel) == 0 && continue
    err = filter(isfinite, abs.(masked.dx[sel] .- truth[sel]))
    corr = filter(isfinite, masked.correlation[sel])
    println("chip ", lpad(lo == hi ? "unmasked" : "$(round(Int, 100lo))-$(round(Int, 100hi))% masked", 16),
            "  n = ", lpad(count(sel), 4),
            "   median correlation ", round(median(corr); digits = 3),
            "   median |error| ", round(median(err); digits = 3), " px")
end
```

Median error is one refinement quantization step — the same as an unmasked chip — while correlation falls
from about 0.81 to 0.71 as coverage grows. The correlation drop is the honest signal here, and it is
another reason to gate on `correlation` rather than on the mask: see [Judging a result](@ref).

## What happens if you do not mask

Leaving a fill value in the imagery is worse than masking, and the damage is local rather than global:

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

Every point whose chip touches the fill value is lost, where masking keeps most of them. A constant
block has no texture, so the correlation there is degenerate and `autorift` declines to report it — which
is the right outcome, but it costs the partially covered points that a mask would have measured. The
reference implementation reports the search-window corner in this situation, a systematic bias toward one
direction over masked ground.

Fill values are not always constant. A block of near-constant noise is not degenerate, and then it does
produce numbers — arbitrary ones. Mask it.

## Where a mask does not reach

- **Both images need one.** A cloud present in only the secondary image still destroys the match.
  `reference_valid` and `secondary_valid` are independent, and correlation uses the intersection —
  [`AutoRIFT.valid`](@ref) of the pair.
- **A mask is not an outlier filter.** It excludes pixels you already know about. Points that fail for
  reasons you cannot enumerate in advance are [Filtering outliers](@ref).
- **Masks do not move.** A cloud shadow displaced between acquisitions needs masking in both images at
  both positions.
- **A lazy raster's `missingval` is already handled.** Reading a GDAL raster with `lazy = true` turns its
  nodata value into mask rather than into a dark measurement, with no keyword from you. See
  [Correlating scenes larger than memory](@ref).

## Filters and masks interact

Preprocessing runs before correlation, and a filter with any spatial extent spreads a masked pixel's
influence into its neighbours. The filters respect the mask — output is zeroed outside the valid
domain — but the reach is real, and [`AutoRIFT.filter_reach`](@ref) reports it in pixels. A wide
[`Highpass`](@ref) beside a large masked region affects more ground than a narrow one.
[`WallisGapfill`](@ref) exists for the opposite approach: fill the gaps with noise matched to the
surrounding statistics rather than zeroing them.

## See also

- [Judging a result](@ref) — why `correlation` is the gate, including over partially masked chips
- [Filtering outliers](@ref) — rejecting the points a mask could not predict
- [Choosing a preprocessing filter](@ref) — `filter_reach` and the gap-filling option
- [`ImagePair`](@ref) — the finiteness default and what it deliberately does not assume
