```@raw html
---
layout: home

hero:
  name: AutoRIFT.jl
  text: Fast dense image motion tracking
  tagline: Two images of the same scene, differing by motion, in. A grid of sub-pixel displacements out.
  actions:
    - theme: brand
      text: Getting started
      link: /getting-started
    - theme: alt
      text: Walkthrough
      link: /tutorials/walkthrough
    - theme: alt
      text: API reference
      link: /reference/index

features:
  - title: Sub-pixel and multi-scale
    details: Several chip sizes in one pass, each point answered at the finest size that worked.
  - title: Coordinate-agnostic
    details: Plain arrays give pixel offsets. Rasters give velocities in map units, signs handled.
  - title: Larger than memory
    details: Blocked processing bounds peak memory by the block rather than the scene, reading lazily.
---
```

Fast, dense image motion tracking by normalized cross-correlation.

Two images of the same scene, differing by motion, in. A grid of sub-pixel displacements out.

```@example home
using AutoRIFT
include("../figures.jl")  # plotting helpers and the synthetic scenes; see [Plotting](@ref)

reference, secondary, _, _ = warped_pair(512, (row, col) -> (6.0, -2.0); seed = 13)
out = autorift(reference, secondary)

image_panels(reference, secondary)
```

```@example home
displacement_panels(out)
```

That is the whole interface for the common case: `autorift(reference, secondary)`, and a result carrying
`dx`, `dy` and `correlation` at every grid point.

## What it is for

Anywhere two images record the same scene at different times and you want to know what moved, and by how
much, everywhere rather than at a few tracked points:

- glacier and ice-sheet velocity
- sea-ice drift
- particle image velocimetry
- digital image correlation and strain mapping
- cell and tissue motion
- video motion estimation

The core knows nothing about any of these. It takes two matrices and returns pixel offsets. Coordinates,
map projections and physical units are a layer on top, reached by passing rasters instead of arrays.

## What it does

- **Sub-pixel, multi-chip-size correlation.** Several chip sizes in one pass, each point answered at the
  finest size that worked — resolution where the imagery supports it, coverage where it does not.
- **Coordinate-agnostic core.** Plain arrays give pixels. Rasters give velocities in map units, with the
  sign and axis conventions handled.
- **Points, not necessarily grids.** A regular grid is the default, but scattered points work, and any
  per-point setting — chip size, search radius, prior — may vary across the scene.
- **Scenes larger than memory.** Blocked processing bounds peak memory by the block rather than the
  scene, reading lazily from disk, with the same answer.
- **Built for batches.** Reusable caches, one pair per task, and no global state.

## Start here

| | |
|---|---|
| **[Getting started](@ref)** | install, one correlation, read the result |
| **[A guided walkthrough](@ref)** | eight steps over one scene, adding a capability at a time |
| **[Geospatial data](@ref)** | rasters in, velocities out |
| **[Judging a result](@ref)** | which points to trust, measured |
| **[API reference](@ref)** | every public name |

## Heritage

AutoRIFT.jl descends from NASA JPL's [autoRIFT](https://github.com/nasa-jpl/autoRIFT) and produces the
[ITS_LIVE](https://its-live.jpl.nasa.gov/) glacier velocity products, its largest deployment. The
algorithm is that lineage: multi-chip-size normalized cross-correlation with the outlier filter of
Gardner et al. (2018).

The implementation, the API, the geospatial handling and the test suite are this package's own, validated
against synthetic ground truth and against production output.

## Citing

If this package contributes to published work, please cite:

> Gardner, A. S., Moholdt, G., Scambos, T., Fahnstock, M., Ligtenberg, S., van den Broeke, M., and Nilsson,
> J. (2018). Increased West Antarctic and unchanged East Antarctic ice discharge over the last 7 years.
> *The Cryosphere*, 12(2), 521–547.
