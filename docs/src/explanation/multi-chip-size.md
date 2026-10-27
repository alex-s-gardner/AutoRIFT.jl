
# Multiple chip sizes

Chip size trades resolution against reliability. A small chip resolves detail and follows a sharp
boundary, but it contains little texture, so on a smooth or decorrelated surface it correlates with
nothing. A large chip succeeds there but cannot resolve detail. No single size is right everywhere in
one scene.

AutoRIFT therefore tries several. Each point is attempted at the finest chip size first, coarser
levels run only where the finer ones failed, and the smallest chip that produces a coherent answer
wins.

## It is not an image pyramid

The imagery is never downsampled. Every level correlates the pair at full resolution and only the
*chip* changes size. There is no coarse-to-fine warm start from a decimated image.

The **grid** is coarsened. A level with a chip twice as wide posts its points twice as far apart, so
one coarse point corresponds to a block of fine ones and the levels nest. Levels are `chip_size * 2^k`,
and both axes double together, so a chip's aspect ratio is the same at every level.

```@example multichip
using AutoRIFT

AutoRIFT.chip_sizes(AutoRIFT.params(; chip_size = 16, chip_size_max = 64))
```

There are three levels. `chip_size` is the finest and `chip_size_max` the coarsest. `chip_size_max`
must be the same power-of-two multiple of `chip_size` in both axes, which the nesting requires.

Where the blocks fall is set by the grid's `origin` (see [`AutoRIFT.PointSet`](@ref)): its first
point's index on a lattice shared by every grid laid out on it. Two grids cut from one map grid at
different windows then measure each level at the same map positions, so where they overlap they agree. A
grid built by [`AutoRIFT.gridpoints`](@ref) has origin `(0, 0)`, which places the blocks from its first
point, as `autoRIFT.py` does; for a grid on the ITS_LIVE map grid, pass `pointset` the origin
`ImagePairGeometry.lattice_origin` gives.

The only pyramid in the algorithm is in [`PyramidRefine`](@ref), which upsamples the neighborhood of a
correlation peak to find its sub-pixel position. It acts on a correlation surface, not an image.

## What each level contributes

In this scene the right chip size differs by region. A horizontal band moves fast and has noise added
to the secondary image, and the surface around it moves slowly and stays well correlated.

```@example multichip
include("../../figures.jl")  # plotting helpers and the example scenes; see [Plotting](@ref)

band(row) = clamp((108 - abs(row - 256)) / 28, 0, 1)

reference, clean, truth_dx, _ =
    warped_pair(photo(), 512, (row, col) -> (2 + 20 * band(row), 0.0))
secondary = decorrelate(clean, (row, col) -> band(row); amplitude = 0.12, seed = 3)

# One grid for every run below, so the point counts compare. A grid's margin is set by the coarsest
# chip and the search radius, so building it for chip 64 makes every run post the same points.
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

Of 2500 grid points, the 16-pixel chip answers 1478, the 32-pixel chip 2074 and the 64-pixel chip
2266. Coverage rises with chip size, which is why coarse levels exist.

```@example multichip
field_panels(blank(levels[16].dx, measured(levels[16])),
             blank(levels[32].dx, measured(levels[32])),
             blank(levels[64].dx, measured(levels[64]));
             titles = ("chip 16", "chip 32", "chip 64"))
```

Coverage is not accuracy. Where the displacement varies across a chip, as at the tapering edges of the
band, a coarse chip still returns a value, and it is an average of two different speeds:

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

In the gradient the 16-pixel chip answers only 12 points, because it has too little texture against
the added noise. The 32-pixel chip answers 88 points with a median error of 4.2 pixels, and the
64-pixel chip answers 253 with a median error of 7.1. A coarse level gains coverage and loses
resolution.

## The merge

Run all three as one call and each point takes its answer from the finest level that succeeded there:

```@example multichip
merged = autorift(reference, secondary, grid; chip_size = 16, chip_size_max = 64, grid_spacing = 8)

[(chip = Int(cs), points = count(==(cs), merged.chip_size))
 for cs in sort(unique(merged.chip_size))]
```

`chip_size == 0` marks a point no level resolved. The other 2026 points come from the 16- and 32-pixel
levels. The 64-pixel level runs on the points the finer two left and answers none of them here. The
`chip_size` layer records which level answered each point, so a consumer can tell a sharply resolved
displacement from a heavily averaged one.

```@example multichip
quality_panels(merged)
```

**The merge is not the union of the levels.** Run separately, the three levels cover 2359 points
between them. Merged, they cover 2026. A level does not attempt every point the finer ones left. Each
level first runs a sparse **coarse pass** to find where motion is spatially coherent, and searches the
full grid only inside a dilated neighborhood of that. A later level's coarse pass sees only the
outstanding points, so its coverage decision differs from the same level run alone.

The difference is spatially structured:

```@example multichip
union_of_levels = measured(levels[16]) .| measured(levels[32]) .| measured(levels[64])

mask_panels(union_of_levels, measured(merged), union_of_levels .& .!measured(merged);
            titles = ("union of the three levels", "the merge", "lost by merging"))
```

The lost points form two ragged horizontal strips along the tapering edges of the band, reaching
toward the left and right image margins. The core of the band and the surface outside it are not lost.
The strips lie where correlation is marginal, so a later level's coarse pass does not accept the
neighborhood and never attempts points that the same level answered when run alone.

The net change differs from the number of lost points: 337 points are lost, but the merge answers four
that no single level did, so the net is 333.

**Where levels overlap, they agree.** Comparing the single-level runs point by point where both
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

The median difference is 0.016 pixel, or 1/64 pixel, the quantization step of the default sub-pixel
refinement. The levels measure the same field, so taking the finest level that works does not make
the result inconsistent.

## The coarse pass

Searching every point at full radius is the dominant cost, and most of it is wasted: displacement is
spatially coherent, so a sparse sample shows where the rest will land. Each level therefore

1. correlates a decimated subset of its points,
2. filters those results for consistency with their neighbors,
3. dilates what survives, and
4. searches the full grid only inside that mask.

If the coarse pass finds nothing coherent, or the filter keeps less than `min_coarse_valid_fraction`
of the coarse points that returned a value, the level is skipped. Three keywords control the coarse pass:

| keyword | default | effect |
|---|---|---|
| `coarse_stride` | `4` | how sparsely the coarse pass samples, as a rate against one point per chip |
| `coarse_buffer` | `8` | dilation radius of the validity mask, in coarse cells — how far past the coarse evidence the fine pass may reach |
| `min_coarse_valid_fraction` | `0.01` | skip a level whose coarse pass validates less than this |

The coarse mask restricts the fine pass by zeroing the search radius at each point outside it, so a
zero search radius at a point excludes the point rather than raising an error.
[`AutoRIFT.nsearchable`](@ref) counts the points that will be correlated.

!!! note "A coarse level needs room"
    A level's decimated grid must be large enough to filter. On a small image the coarsest levels can
    have a grid too small to run, and they are skipped. A 256-pixel scene at the default
    `chip_size_max` may return nothing. Use scenes of 512 pixels or more, or set
    `chip_size_max` explicitly.

## Choosing the range

`chip_size` is the finest level and sets the best achievable resolution. `chip_size_max` is how far
the search escalates when the fine chip fails.

- **Both equal** runs a single level. It is the cheapest option and suits a scene uniform in texture
  and in the smoothness of its motion.
- **The default `4 * chip_size`** gives three levels and is a reasonable start for a scene with mixed
  terrain.
- **A wider range** costs little where the fine levels already succeed, since a level runs only on
  the points still unresolved. The grid margin is set by the coarsest chip, so raising `chip_size_max`
  shrinks the output grid.

Where the right range varies within a scene, bound it per point: a [`AutoRIFT.PointSet`](@ref) carries
`chip_size_min_x` and `chip_size_max_x` fields, and Steps 6 to 8 of [A guided walkthrough](@ref) use
them. The finest level is attempted at every point regardless of these fields, so per-point bounds
restrict the coarser levels only.

## See also

- [How feature tracking works](@ref): one chip, one window, one surface
- [A guided walkthrough](@ref): the levels in use, from Step 4
- [`AutoRIFT.correlate_multichip`](@ref) and [`AutoRIFT.chipsize_level`](@ref): running the loop, or
  one level of it, directly
