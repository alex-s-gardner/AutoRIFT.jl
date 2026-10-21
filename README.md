# AutoRIFT.jl

[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://alex-s-gardner.github.io/AutoRIFT.jl/stable/)
[![Docs dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://alex-s-gardner.github.io/AutoRIFT.jl/dev/)
[![Build Status](https://github.com/alex-s-gardner/AutoRIFT.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/alex-s-gardner/AutoRIFT.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/alex-s-gardner/AutoRIFT.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/alex-s-gardner/AutoRIFT.jl)

Dense image motion tracking by normalized cross-correlation.

Two images of the same scene that differ by motion go in; a grid of sub-pixel displacements comes
out, at every grid point rather than at a few tracked features.

```julia
using AutoRIFT

out = autorift(reference, secondary)
out.dx, out.dy, out.correlation
```

![A photograph, a copy of it bowed downward, and the recovered displacement field](https://raw.githubusercontent.com/alex-s-gardner/AutoRIFT.jl/main/docs/src/assets/readme_example.png)

The second image is the first bowed downward: 20 pixels at the center column, none at the left and
right edges. The ring ropes, straight in the first image, show the bow. The third panel is
`-out.dy` from `autorift(reference, secondary; grid_spacing = 8)`, which is how far each column
moved down. [`docs/readme_figure.jl`](docs/readme_figure.jl) generates the figure.

Applications include glacier and ice-sheet velocity, sea-ice drift, particle image velocimetry,
digital image correlation and strain mapping, cell and tissue motion, and video motion estimation.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/alex-s-gardner/AutoRIFT.jl")
```

AutoRIFT.jl is not in the General registry; install it from its repository.

Correlating arrays needs nothing further. The packages below add input types and features. Load
them yourself with `using`; installing AutoRIFT does not install them.

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

## Quickstart

Two arrays on the same grid, differing by motion:

```julia
using AutoRIFT

out = autorift(reference, secondary; chip_size = 32, search_radius = 25)

out.dx           # displacement along the second index (columns), in pixels
out.dy           # along the first index (rows), positive downward
out.correlation  # correlation coefficient at the peak, the value to filter on
```

`dx` and `dy` are the offset from the secondary image back to the reference, the negative of how
features moved: a feature that moved six columns left reports `dx = +6`. Unmeasured points are
`NaN`, not zero, because zero displacement is a valid result.

For geolocated rasters, pass the acquisition interval to get velocities in map units, east and
north positive:

```julia
using AutoRIFT, Rasters, ArchGDAL, Dates

out = autorift(raster1, raster2; dt = Day(16))

out.vx, out.vy   # feature motion in CRS units per year, in the inputs' CRS
```

## Features

- **Sub-pixel, multi-chip-size correlation.** Several chip sizes in one pass; each point takes the
  finest size that produced a measurement.
- **Coordinate-agnostic core.** Plain matrices give pixel offsets, so map-projected and radar
  slant-range imagery are handled the same way. The geospatial methods are package extensions, so
  the core does not load GDAL.
- **Points, not only grids.** Each search center has its own coordinates, search radii, prior
  displacement, and chip size. Centers may be scattered at fractional coordinates (single chip size)
  or laid out on a grid (multiple chip sizes).
- **Scenes larger than memory.** Blocked processing bounds peak memory by the block size rather
  than the scene size, reading lazily from disk, and returns the same result as an unblocked run.
- **Batches.** `AutoRIFT.init`, `reinit!`, and `autorift!` reuse buffers and FFT plans across pairs.

## Documentation

The [documentation](https://alex-s-gardner.github.io/AutoRIFT.jl/dev/) starts from four pages:

- **[Getting started](https://alex-s-gardner.github.io/AutoRIFT.jl/dev/getting-started)**: install,
  one correlation, reading the result.
- **[A guided walkthrough](https://alex-s-gardner.github.io/AutoRIFT.jl/dev/tutorials/walkthrough)**:
  one scene in eight steps, each adding a capability, up to chip sizes, search radii, and priors
  that vary across the scene.
- **[Conventions](https://alex-s-gardner.github.io/AutoRIFT.jl/dev/explanation/conventions)**: the
  sign of `dx`/`dy`, where the y flip happens, and the half pixel.
- **[API reference](https://alex-s-gardner.github.io/AutoRIFT.jl/dev/reference/index)**: every
  public name.

Run times and memory against the Python reference are on the
[Benchmarks](https://alex-s-gardner.github.io/AutoRIFT.jl/dev/benchmarks) page.

## Heritage

AutoRIFT.jl descends from NASA JPL's [autoRIFT](https://github.com/nasa-jpl/autoRIFT) and produces
the [ITS_LIVE](https://its-live.jpl.nasa.gov/) glacier velocity products. The algorithm is
multi-chip-size normalized cross-correlation with the outlier filter of Gardner et al. (2018).

The implementation, API, geospatial handling, and test suite are this package's own. They are
validated against synthetic ground truth, in which a texture displaced by a known amount has a known
answer, and against production output. [`dev/`](dev/README.md) holds that validation record.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Citing

If this package contributes to published work, please cite:

> Gardner, A. S., Moholdt, G., Scambos, T., Fahnstock, M., Ligtenberg, S., van den Broeke, M., and
> Nilsson, J. (2018). Increased West Antarctic and unchanged East Antarctic ice discharge over the
> last 7 years. *The Cryosphere*, 12(2), 521–547.

## License

MIT. See [`LICENSE`](LICENSE).
