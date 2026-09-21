
# Choosing chip size, spacing, and radius

Four numbers decide what a run costs and what it can resolve: `chip_size`, `chip_size_max`,
`grid_spacing`, and `search_radius`. This page is what each one does to cost and to the answer, so
you can pick them without a sweep.

```@example tuning
using AutoRIFT
include("../../figures.jl")  # plotting helpers and the synthetic scenes; see [Plotting](@ref)

reference, secondary, _, _ = warped_pair(512, (row, col) -> (6.0, -2.0); seed = 13)

image_panels(reference, secondary)
```

## What each one costs

The honest way to measure this is over **one fixed set of points**, because every setting below also
changes how many points a scene has room for — a wider chip or radius needs more margin from the
image edge, so a naive sweep conflates cost per point with a shrinking grid. One grid, sized for the
widest arm:

```@example tuning
grid = AutoRIFT.gridpoints(size(reference), 4; chip_size = 128, search_radius = 50)
AutoRIFT.npoints(grid)
```

Measured on those 5,041 points, one Apple M2 Max core:

| setting | value | time | scaling |
|---|---:|---:|---|
| `chip_size` (single level) | 16 / 32 / 64 / 128 | 894 / 1085 / 1294 / 2045 ms | mild — the transform is `O(n² log n)` in a chip that is 64× the area |
| `search_radius` | 5 / 10 / 25 / 50 | 515 / 537 / 687 / 1073 ms | roughly with area |
| `chip_size_max` | 16 / 32 / 64 / 128 | 893 / 894 / 896 / 893 ms | **free** |
| `grid_spacing` | 8 / 16 / 32 / 64 | 334 / 81 / 22 / 6.2 ms | with the point count, which is `spacing⁻²` |

Three things to take from that:

**`grid_spacing` is the dominant cost.** It is quadratic and it is the one knob that changes nothing
about how a displacement is measured — only how often. Halving it quadruples the work for
measurements that already overlap: at spacing 8 with 32-pixel chips, neighbouring chips share
three-quarters of their pixels.

**`search_radius` is the expensive *correctness* knob.** Cost grows with the area searched, so
doubling it roughly quadruples the search. It is also a hard limit: motion beyond it is not measured
badly, it is not measured at all. The way out is not a wider radius but a prior — see
[Giving the search a first guess](@ref).

**Raising `chip_size_max` is free.** A coarse level only runs where every finer level failed, so on a
well-correlated scene the extra levels have almost nothing to do. What it costs instead is *points*:
the grid's margin is set by the coarsest chip, so `chip_size_max = 128` on a 512² scene yields a
smaller output grid than `chip_size_max = 32` does.

## Chip size against resolution

A chip must hold enough texture to be distinguishable, and be small enough that the displacement is
roughly uniform across it. Those pull opposite ways, which is the entire reason for a multi-chip-size
search:

```@example tuning
fine = autorift(reference, secondary; chip_size = 16, chip_size_max = 16, grid_spacing = 8)
coarse = autorift(reference, secondary; chip_size = 64, chip_size_max = 64, grid_spacing = 8)

field_panels(blank(fine.dx, measured(fine)), blank(coarse.dx, measured(coarse));
             titles = ("chip 16", "chip 64"))
```

Fine chips resolve detail and fail more often; coarse chips cover more and average across it.
[Multiple chip sizes](@ref) quantifies the trade with a scene where the right answer differs by
region, and [A guided walkthrough](@ref) builds the range up step by step.

## Where to start

- **`chip_size`** — the finest level, and your resolution ceiling. 32 by default. Drop to 16 for
  well-textured imagery where you want detail; raise it when correlation is poor everywhere.
- **`chip_size_max`** — `4 * chip_size` by default, giving three levels. Set it equal to `chip_size`
  for a single level when the scene is uniform, and remember that raising it shrinks the grid.
- **`grid_spacing`** — the output sampling. There is no accuracy reason to go below about a quarter of
  `chip_size`; below that you are paying quadratically for correlated neighbours.
- **`search_radius`** — cover the motion you expect plus a margin. If that number is large, use a
  prior and a small radius instead.

## Per-point, and one field that does not apply

Any of these may vary across the scene, through a [`AutoRIFT.PointSet`](@ref) rather than a keyword —
steps 6 to 8 of [A guided walkthrough](@ref) do exactly that. One asymmetry to know, because it is
easy to write code that looks right and does nothing:

```@example tuning
points = AutoRIFT.gridpoints(size(reference), 16; chip_size = 64, search_radius = 30)
as_16 = AutoRIFT.pointset(points.x, points.y; chip_size_x = 16, chip_size_y = 16,
                          search_radius_x = 30, search_radius_y = 30)
as_64 = AutoRIFT.pointset(points.x, points.y; chip_size_x = 64, chip_size_y = 64,
                          search_radius_x = 30, search_radius_y = 30)

# Same answer: on a gridded point set the level sets the chip size, not this field.
isequal(autorift(reference, secondary, as_16).dx, autorift(reference, secondary, as_64).dx)
```

On a **gridded** `PointSet{2}`, `chip_size_x`/`chip_size_y` are overwritten per level — the
multi-chip-size loop sets them, and [`AutoRIFT.correlate_multichip`](@ref) documents that. To vary
chip size per point, bound the levels with `chip_size_min_x`/`chip_size_max_x` instead. The field does
take effect on a **scattered** `PointSet{1}`, which runs a single scale.

`search_radius` runs the other way: on any point set the per-point `radius_x`/`radius_y` win and the
`search_radius` keyword is ignored, since the levels do not set the radius.

## See also

- [Multiple chip sizes](@ref) — why the levels exist and how they merge
- [A guided walkthrough](@ref) — each of these settings added one at a time, with figures
- [Giving the search a first guess](@ref) — the alternative to a wide radius
- [Judging a result](@ref) — whether the settings you chose worked
- [`AutoRIFT.params`](@ref) — every keyword, with defaults
