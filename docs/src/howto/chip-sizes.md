
# Choosing chip size, spacing, and radius

Four settings determine what a run costs and what it can resolve: `chip_size`, `chip_size_max`,
`grid_spacing`, and `search_radius`. This page gives the effect of each on cost and on the result.

```@example tuning
using AutoRIFT
include("../../figures.jl")  # plotting helpers and the example scenes; see [Plotting](@ref)

reference, secondary, _, _ = warped_pair(photo(), 512, (row, col) -> (6.0, -2.0))

image_panels(reference, secondary)
```

## What each one costs

Cost is measured over one fixed set of points. A wider chip or radius needs more margin from the image
edge, so the grid shrinks, and a sweep over grids of different size mixes cost per point with point
count. The grid below is sized for the widest setting:

```@example tuning
grid = AutoRIFT.gridpoints(size(reference), 4; chip_size = 128, search_radius = 50)
AutoRIFT.npoints(grid)
```

Measured on those 5,041 points, one Apple M2 Max core:

| setting | value | time | scaling |
|---|---:|---:|---|
| `chip_size` (single level) | 16 / 32 / 64 / 128 | 894 / 1085 / 1294 / 2045 ms | mild: the transform is `O(n² log n)` in a chip that is 64× the area |
| `search_radius` | 5 / 10 / 25 / 50 | 515 / 537 / 687 / 1073 ms | roughly with area |
| `chip_size_max` | 16 / 32 / 64 / 128 | 893 / 894 / 896 / 893 ms | none |
| `grid_spacing` | 8 / 16 / 32 / 64 | 334 / 81 / 22 / 6.2 ms | with the point count, which is `spacing⁻²` |

`grid_spacing` dominates the cost. It is quadratic, and it changes how often a displacement is
measured, not how any one of them is measured. At spacing 8 with 32-pixel chips, neighboring chips
along an axis share three-quarters of their pixels, so halving the spacing quadruples the work for
measurements that already overlap.

`search_radius` costs in proportion to the area searched, so doubling it roughly quadruples the
search. It is also a limit on what can be measured. Motion near the radius is reported as a lower bound
and flagged (see [Judging a result](@ref)), and motion well beyond it is not recovered. The alternative
to a wide radius is a prior; see [Giving the search a first guess](@ref).

Raising `chip_size_max` costs no time. A coarse level runs only where every finer level failed, so on a
well-correlated scene the extra levels have almost nothing to do. It costs points instead: the grid's
margin is set by the coarsest chip, so `chip_size_max = 128` on a 512² scene yields a smaller output
grid than `chip_size_max = 32`.

## Chip size against resolution

A chip must hold enough texture to be distinguishable and be small enough that the displacement is
roughly uniform across it. These requirements conflict, which is the reason for a multi-chip-size
search:

```@example tuning
fine = autorift(reference, secondary; chip_size = 16, chip_size_max = 16, grid_spacing = 8)
coarse = autorift(reference, secondary; chip_size = 64, chip_size_max = 64, grid_spacing = 8)

field_panels(blank(fine.dx, measured(fine)), blank(coarse.dx, measured(coarse));
             titles = ("chip 16", "chip 64"))
```

The shift in this scene is uniform, so both recover it. They differ where the displacement varies
across a chip: fine chips resolve detail and fail more often, and coarse chips cover more ground and
average across it. [Multiple chip sizes](@ref) quantifies the trade on a scene where the right answer
differs by region, and [A guided walkthrough](@ref) builds up the range step by step.

## Where to start

- `chip_size`: the finest level, which sets the resolution. The default is 32. Use 16 for
  well-textured imagery where detail matters, and a larger value when correlation is poor everywhere.
- `chip_size_max`: `4 * chip_size` by default, giving three levels. Set it equal to `chip_size` for a
  single level when the scene is uniform. Raising it shrinks the grid.
- `grid_spacing`: the output sampling. Below about a quarter of `chip_size`, the extra points are
  strongly correlated neighbors and cost time quadratically.
- `search_radius`: the expected motion plus a margin. If that is large, use a prior and a small radius.

## Per-point values

Chip size and search radius can vary across the scene through a [`AutoRIFT.PointSet`](@ref) rather than
a keyword; steps 6 to 8 of [A guided walkthrough](@ref) do this. The chip-size field behaves differently
from what its name suggests:

```@example tuning
points = AutoRIFT.gridpoints(size(reference), 16; chip_size = 64, search_radius = 30)
as_16 = AutoRIFT.pointset(points.x, points.y; chip_size_x = 16, chip_size_y = 16,
                          search_radius_x = 30, search_radius_y = 30)
as_64 = AutoRIFT.pointset(points.x, points.y; chip_size_x = 64, chip_size_y = 64,
                          search_radius_x = 30, search_radius_y = 30)

# Same answer: on a gridded point set the level sets the chip size, not this field.
isequal(autorift(reference, secondary, as_16).dx, autorift(reference, secondary, as_64).dx)
```

On a gridded `PointSet{2}`, `chip_size_x`/`chip_size_y` are overwritten at each level by the
multi-chip-size loop; see [`AutoRIFT.correlate_multichip`](@ref). To vary chip size per point, bound the
levels with `chip_size_min_x`/`chip_size_max_x`. The finest level attempts every point with a positive
radius whatever its bounds, so the bounds restrict only the coarser levels. On a scattered
`PointSet{1}`, which runs a single scale, the field takes effect.

The search radius works the other way. On any point set the per-point `radius_x`/`radius_y` are used and
the `search_radius` keyword is ignored, because the levels do not set the radius.

## See also

- [Multiple chip sizes](@ref): why the levels exist and how they merge
- [A guided walkthrough](@ref): each of these settings added one at a time, with figures
- [Giving the search a first guess](@ref): the alternative to a wide radius
- [Judging a result](@ref): whether the settings you chose worked
- [`AutoRIFT.params`](@ref): every keyword, with defaults
