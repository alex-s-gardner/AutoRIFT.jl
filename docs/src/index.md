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

AutoRIFT.jl tracks image motion by normalized cross-correlation. It takes two images of the same
scene that differ by motion and returns a grid of sub-pixel displacements.

```@example home
using AutoRIFT
include("../figures.jl")  # plotting helpers and the example scenes; see [Plotting](@ref)

# The secondary is the reference turned about 3.4° and shifted, so `dx` runs from about 31 px at the
# top to 1 px at the bottom and `dy` from about -27 px at the left to 3 px at the right.
reference, secondary, _, _ =
    warped_pair(photo(), 512, (row, col) -> (16.0 - 0.06 * (row - 256.5), -12.0 + 0.06 * (col - 256.5)))
out = autorift(reference, secondary; search_radius = 36, grid_spacing = 16)

image_panels(reference, secondary)
```

```@example home
displacement_panels(out)
```

`autorift` returns `dx`, `dy` and `correlation` at every grid point. Both components vary across the
scene.

## What it is for

Any pair of images of one scene, taken at different times, where you want to know what moved and by how
much at every grid point:

- glacier and ice-sheet velocity
- sea-ice drift
- particle image velocimetry
- digital image correlation and strain mapping
- cell and tissue motion
- video motion estimation

The core takes two matrices and returns pixel offsets. Coordinates, map projections and physical units
come from passing rasters instead of arrays. Points may lie on a regular grid, the default, or be
scattered, and any per-point setting (chip size, search radius, prior) may vary across the scene. A cache
reuses buffers and FFT plans across a batch of pairs.

## Heritage

AutoRIFT.jl descends from NASA JPL's [autoRIFT](https://github.com/nasa-jpl/autoRIFT) and produces the
[ITS_LIVE](https://its-live.jpl.nasa.gov/) glacier velocity products. The algorithm is multi-chip-size
normalized cross-correlation with the outlier filter of Gardner et al. (2018).

The implementation, API, geospatial handling and test suite are this package's own. They are validated
against synthetic ground truth and against production output.

## Citing

If this package contributes to published work, please cite:

> Gardner, A. S., Moholdt, G., Scambos, T., Fahnstock, M., Ligtenberg, S., van den Broeke, M., and Nilsson,
> J. (2018). Increased West Antarctic and unchanged East Antarctic ice discharge over the last 7 years.
> *The Cryosphere*, 12(2), 521–547.
