
# Multiple chip sizes

Chip size is a trade with no good answer. A small chip resolves detail and follows a sharp boundary,
but it contains little texture, so on a smooth or decorrelated surface it correlates with nothing. A
large chip succeeds there and is blind to detail. Neither choice is right everywhere in one scene.

So AutoRIFT tries several. Each point is attempted at the finest chip size first; coarser levels run
only where the finer ones failed, and the **smallest chip that produces a coherent answer wins**.

## It is not an image pyramid

Worth stating plainly, because the structure invites the wrong name. The imagery is never downsampled.
Every level correlates the pair at full resolution and only the *chip* changes size — there is no
coarse-to-fine warm start from a decimated image.

What does get coarsened is the **grid**. A level with a chip twice as wide posts its points twice as
far apart, which is what makes the levels nest: one coarse point corresponds exactly to a block of
fine ones. Levels are `chip_size * 2^k`, and both axes double together, so a chip's aspect ratio is
the same at every level.

```@example multichip
using AutoRIFT

AutoRIFT.chip_sizes(AutoRIFT.params(; chip_size = 16, chip_size_max = 64))
```

Three levels. `chip_size` is the finest and `chip_size_max` the coarsest; `chip_size_max` must be the
same power-of-two multiple of `chip_size` in both axes, which is what the nesting requires.

The one genuine pyramid in the algorithm is elsewhere: [`PyramidRefine`](@ref) upsamples the
neighbourhood of a correlation peak to find its sub-pixel position. That is a pyramid over a
correlation surface, not over an image.

## What each level contributes

A scene where the right chip size differs by region: a band that both moves fast and decorrelates,
surrounded by a static, well-correlated surface.

```@example multichip
include("../../figures.jl")  # plotting helpers and the synthetic scenes; see [Plotting](@ref)

band(row) = clamp((108 - abs(row - 256)) / 28, 0, 1)

reference, clean, truth_dx, _ =
    warped_pair(512, (row, col) -> (2 + 20 * band(row), 0.0); seed = 11)
secondary = decorrelate(clean, (row, col) -> band(row); amplitude = 0.12, seed = 3)

# One grid for every arm below, so the point counts compare. A grid's margin is set by the coarsest
# chip and the search radius, so building it for chip 64 makes every arm post the same points.
grid = AutoRIFT.gridpoints(size(reference), 8; chip_size = 64, search_radius = 24)

image_panels(reference, secondary)
```

Each level on its own, over that grid:

```@example multichip
single(cs) = autorift(reference, secondary, grid;
                      chip_size = cs, chip_size_max = cs, grid_spacing = 8)

levels = Dict(cs => single(cs) for cs in (16, 32, 64))
[(chip = cs, measured = AutoRIFT.nmeasured(levels[cs])) for cs in (16, 32, 64)]
```

Out of 2500 points: the 16-pixel chip answers 1038, the 32-pixel chip 1775, the 64-pixel chip 2349.
Coverage rises with chip size, which is the whole reason coarse levels exist.

```@example multichip
field_panels(blank(levels[16].dx, measured(levels[16])),
             blank(levels[32].dx, measured(levels[32])),
             blank(levels[64].dx, measured(levels[64]));
             titles = ("chip 16", "chip 32", "chip 64"))
```

Coverage is not accuracy. Where the displacement *varies* across a chip — the tapering edges of the
band — a coarse chip still returns something, and that something is an average of two different
speeds:

```@example multichip
using Statistics: median

truth = [truth_dx[Int(y), Int(x)] for (x, y) in zip(grid.x, grid.y)]
gradient = 0 .< band.(grid.y) .< 1          # 350 points where the speed is changing

function gradient_error(out)
    m = .!isnan.(out.dx) .& gradient
    count(m) == 0 && return (points = 0, median_error = missing)
    return (points = count(m), median_error = round(median(abs.(out.dx[m] .- truth[m])); digits = 2))
end

[(chip = cs, gradient_error(levels[cs])...) for cs in (16, 32, 64)]
```

The 16-pixel chip answers nowhere in the gradient — it has too little texture against the added
noise. The 32-pixel chip answers 26 points to within 2.6 pixels, and the 64-pixel chip answers 273 to
within 4.2. The coarse level's advantage is coverage and its cost is resolution, quantified.

## The merge

Run all three as one call and each point takes its answer from the finest level that succeeded there:

```@example multichip
merged = autorift(reference, secondary, grid; chip_size = 16, chip_size_max = 64, grid_spacing = 8)

[(chip = Int(cs), points = count(==(cs), merged.chip_size))
 for cs in sort(unique(merged.chip_size))]
```

`chip_size == 0` marks a point no level resolved. The remaining 1815 points are drawn from all three
levels, and the layer says which — so a downstream consumer can tell a sharply-resolved displacement
from a heavily-averaged one, which is why the chip size is kept rather than discarded.

```@example multichip
quality_panels(merged)
```

Two things in that figure are worth noticing.

**The merge is not the union of the levels.** Run separately the three levels cover 2378 points
between them; merged they cover 1815. A level does not attempt every point the finer ones left — each
level first runs a sparse **coarse pass** to find where motion is spatially coherent, and searches the
full grid only inside a dilated neighbourhood of that. Since a later level's coarse pass sees only the
points still outstanding, its coverage decision differs from the same level run alone.

**Where levels do overlap they agree.** Comparing the single-level runs point by point where both
produced an answer:

```@example multichip
function overlap(a, b)
    both = .!isnan.(levels[a].dx) .& .!isnan.(levels[b].dx)
    return (points = count(both),
            median_difference = round(median(abs.(levels[a].dx[both] .- levels[b].dx[both]));
                                      digits = 3))
end

(low = overlap(16, 32), high = overlap(32, 64))
```

A sixteenth of a pixel — one quantization step of the sub-pixel refinement. The levels are measuring
the same field, so preferring the finest one that works costs nothing in consistency.

## The coarse pass

Searching every point at full radius is the dominant cost, and most of it is wasted: displacement is
spatially coherent, so a sparse sample tells you where the rest will land. Each level therefore

1. correlates a decimated subset of its points,
2. filters those results for consistency with their neighbours,
3. dilates what survives, and
4. searches the full grid only inside that mask.

Where the coarse pass finds nothing coherent, the fine pass is skipped entirely — and if it validates
too small a fraction of the scene, the whole level is skipped. Three keywords control it:

| keyword | default | effect |
|---|---|---|
| `coarse_stride` | `4` | how sparsely the coarse pass samples, as a rate against one point per chip |
| `coarse_buffer` | `8` | dilation radius of the validity mask, in coarse cells — how far past the coarse evidence the fine pass may reach |
| `min_coarse_valid_fraction` | `0.01` | skip a level whose coarse pass validates less than this |

This is also why a search radius of **zero** excludes a point rather than erroring: setting the radius
per point is the mechanism by which the coarse mask restricts the fine pass, so a zero radius has to
mean "do not search here". [`AutoRIFT.nsearchable`](@ref) counts the points that will actually be
correlated.

!!! note "A coarse level needs room"
    A level's decimated grid has to be large enough to filter. On a small image the coarsest levels
    can end up with a grid too small to be worth running, and they are skipped — so a 256-pixel scene
    at the default `chip_size_max` may return nothing at all. Keep test scenes at 512 pixels or limit
    `chip_size_max` explicitly.

## Choosing the range

`chip_size` is the finest level and the one that sets your best achievable resolution;
`chip_size_max` is how far the search will escalate when the fine chip fails.

- **Both equal** runs a single level: cheapest, and right when the scene is uniform in texture and in
  the smoothness of its motion.
- **The default `4 * chip_size`** gives three levels and is a reasonable starting point for a scene
  with mixed terrain.
- **A wider range** costs little where the fine levels already succeed, since a level only runs on
  what is left, but each additional level widens the grid margin — the margin is set by the coarsest
  chip, so raising `chip_size_max` shrinks the output grid.

Where the right range varies *within* a scene, bound it per point rather than globally: a
[`AutoRIFT.PointSet`](@ref) carries `chip_size_min_x` and `chip_size_max_x` fields. Steps 6 to 8 of
[A guided walkthrough](@ref) do this. Note the asymmetry — the finest level is attempted at every
point regardless, so per-point bounds restrict the coarser levels only.

## See also

- [How feature tracking works](@ref) — one chip, one window, one surface
- [A guided walkthrough](@ref) — the levels in use, step 4 onward
- [`AutoRIFT.correlate_multichip`](@ref) and [`AutoRIFT.chipsize_level`](@ref) — running the loop, or
  one level of it, directly
