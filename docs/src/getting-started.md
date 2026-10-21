
# Getting started

## Install

```julia
using Pkg
Pkg.add(url = "https://github.com/alex-s-gardner/AutoRIFT.jl")
```

AutoRIFT.jl is not in the General registry; install it from its repository.

Correlating arrays needs nothing further. Optional packages add input types and features. Load them with
`using`; installing AutoRIFT does not install them.

| load this | to get |
|---|---|
| `Rasters` + `ArchGDAL` | geospatial input, velocities in map units |
| `DimensionalData` | labeled-array input, offsets in pixels |
| `DiskArrays` | chunked on-disk arrays (Zarr, NetCDF, HDF5) read by window |
| `ImageFeatures` | sparse first guesses from ORB features |
| `AkazeFeatures` | sparse first guesses from A-KAZE features; install with `Pkg.add(url = "https://github.com/nlw0/AkazeFeatures.jl")` |
| `NearestNeighbors` | k-d tree neighbor search in the first-guess consistency filter |
| `ImagePairGeometry` | search grids from a geogrid `PairGeometry`; install with `Pkg.add(url = "https://github.com/alex-s-gardner/ImagePairGeometry.jl")` |
| `Metal` | Apple GPU backend (experimental) |

## One correlation

You need two images of the same scene, on the same grid, differing by motion. This page uses a
photograph and a warped copy of it, so the applied displacement is known:

```@example start
using AutoRIFT
include("../figures.jl")  # plotting helpers and the example scenes; see [Plotting](@ref)

# Sideways motion that is fastest in the middle rows and slower toward the top and bottom edges,
# with a 2 px downward drift. Ice in a valley moves like this.
flow(row, col) = (4.0 + 12.0 * sin(pi * row / 512), -2.0)

reference, secondary, true_dx, true_dy = warped_pair(photo(), 512, flow)

image_panels(reference, secondary)
```

The displacement varies with row: 4 px at the top and bottom edges, 16 px in the middle rows. One
number for the whole frame cannot describe it. The applied field, which the result below should
reproduce:

```@example start
field_panels(true_dx, true_dy; titles = ("true dx (px)", "true dy (px)"))
```

The call:

```@example start
out = autorift(reference, secondary)
```

## Read the result

`out` holds several layers with one value per grid point. The first three to use:

```@example start
displacement_panels(out)
```

- **`dx`** — displacement along the second index (columns), in pixels
- **`dy`** — displacement along the first index (rows), positive downward
- **`correlation`** — the correlation coefficient at the peak; higher is a better match

```@example start
(measured = AutoRIFT.nmeasured(out), of_total = length(out.dx))
```

Points that could not be measured are `NaN`, not zero, because zero is a valid displacement. Filter
before reducing:

```@example start
using Statistics: median
(median_dx = median(filter(isfinite, out.dx)), median_dy = median(filter(isfinite, out.dy)))
```

`dy` is −2, the value applied to the whole scene. `dx` runs from about 4 at the top and bottom edges to
16 in the middle rows, so its median describes no part of the scene. Compare the `dx` panel against the
true field instead.

The grid is 11×11 points by default, coarser than the 512×512 input, so the panels look blocky.
`grid_spacing` sets it; [A guided walkthrough](@ref) shows its effect.

!!! note "The sign is the offset, not the motion"
    `dx`/`dy` point from the secondary image back to the reference, the opposite of how the features
    moved. A feature that moved six columns left reports `dx = +6`. The convention comes from the
    reference implementation; see [Conventions](@ref).

## Judge it before using it

Filter on `correlation` before using the points:

```@example start
good = out.correlation .> 0.3
(kept = count(good), rejected = AutoRIFT.nmeasured(out) - count(good))
```

```@example start
quality_panels(out)
```

By default `autorift` also rejects spatial outliers, so points that disagree with their neighbors are
already removed. [Judging a result](@ref) covers the threshold and why to filter on `correlation`
rather than `peak_ratio`.

## Where to go next

- **[A guided walkthrough](@ref)**: a scene with a decorrelating shear band, in eight steps that each add
  one capability: chip size, grid spacing, multiple chip sizes, priors, and settings that vary across
  the scene.
- **[Geospatial data](@ref)**: what changes for rasters, namely `vx`/`vy` in map orientation and
  velocities from an acquisition interval.
- **[How feature tracking works](@ref)**: what a chip, a search window, and a correlation surface are.
- **[Judging a result](@ref)**: choosing a correlation threshold.
