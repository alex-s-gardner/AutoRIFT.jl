# AutoRIFT.jl

[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://alex-s-gardner.github.io/AutoRIFT.jl/stable/)
[![Docs dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://alex-s-gardner.github.io/AutoRIFT.jl/dev/)
[![Build Status](https://github.com/alex-s-gardner/AutoRIFT.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/alex-s-gardner/AutoRIFT.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/alex-s-gardner/AutoRIFT.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/alex-s-gardner/AutoRIFT.jl)

Fast dense image motion tracking by normalized cross-correlation.

Two images of the same scene that differ by motion go in; a grid of sub-pixel displacements comes
out, at every point rather than at a few tracked features.

```julia
using AutoRIFT

out = autorift(reference, secondary)
out.dx, out.dy, out.correlation
```

![A texture, a copy of it whose middle band moved, and the recovered displacement field](https://raw.githubusercontent.com/alex-s-gardner/AutoRIFT.jl/main/docs/src/assets/readme_example.png)

A band across the middle moved 22 pixels and decorrelated; the surface around it moved 2. The third
panel is what the call above returns. Reproduce it with
[`docs/readme_figure.jl`](docs/readme_figure.jl).

Applications include glacier and ice-sheet velocity, sea-ice drift, particle image velocimetry,
digital image correlation and strain mapping, cell and tissue motion, and video motion estimation.

## Installation

```julia
using Pkg
Pkg.add("AutoRIFT")
```

Nothing further is needed to correlate arrays. Optional packages unlock additional input types, and
each is loaded by you rather than installed as a dependency:

| load this | to get |
|---|---|
| `Rasters` + `ArchGDAL` | geospatial input, velocities in map units |
| `DimensionalData` | labelled-array input, offsets in pixels |
| `ImageFeatures` | sparse first guesses from ORB features |
| `Metal` | the Apple GPU backend (experimental) |

## Quickstart

Two arrays on the same grid, differing by motion:

```julia
using AutoRIFT

out = autorift(reference, secondary; chip_size = 32, search_radius = 25)

out.dx           # displacement along the second index (columns), in pixels
out.dy           # along the first index (rows), positive downward
out.correlation  # how well the chip matched, in [0, 1] — the gate to filter on
```

Unmeasured points are `NaN`, not zero: zero displacement is a real answer and must not be confused
with an absent one.

If the images are geolocated, pass rasters and an acquisition interval instead, and velocities come
back in map units with the axis and sign conventions handled:

```julia
using AutoRIFT, Rasters, ArchGDAL, Dates

out = autorift(raster1, raster2; dt = Day(16))

out.vx, out.vy   # velocity in map units per year, on the input's grid and CRS
```

## What it does

- **Sub-pixel, multi-chip-size correlation.** Several chip sizes in one pass, each point answered at
  the finest size that worked — resolution where the imagery supports it, coverage where it does not.
- **Coordinate-agnostic core.** Plain matrices give pixel offsets, so map-projected and radar
  slant-range imagery are handled identically. The geospatial methods live in package extensions, so
  a core that only correlates two matrices does not pay to load GDAL.
- **Points, not necessarily grids.** Every search center is independent — its own coordinates, search
  radii, prior displacement, and chip size. Centers may be scattered at fractional coordinates or
  laid out on a grid.
- **Scenes larger than memory.** Blocked processing bounds peak memory by the block rather than the
  scene, reading lazily from disk, with a bit-identical answer.
- **Built for batches.** A cache lifecycle (`init` / `reinit!` / `autorift!`) reuses buffers and FFT
  plans across pairs, allocation-free inner loops are asserted in the test suite, and there is no
  global state.

## Documentation

The [documentation](https://alex-s-gardner.github.io/AutoRIFT.jl/dev/) has three doors:

- **[Getting started](https://alex-s-gardner.github.io/AutoRIFT.jl/dev/getting-started)** — install,
  one correlation, read the result.
- **[A guided walkthrough](https://alex-s-gardner.github.io/AutoRIFT.jl/dev/tutorials/walkthrough)** —
  one scene through eight steps, adding one capability at a time, up to chip sizes, search radii and
  priors that vary across the scene.
- **[Conventions](https://alex-s-gardner.github.io/AutoRIFT.jl/dev/explanation/conventions)** — the
  sign of `dx`/`dy`, where the y flip happens, and the half pixel. Worth reading once before trusting
  a result.
- **[API reference](https://alex-s-gardner.github.io/AutoRIFT.jl/dev/reference/index)** — every
  public name.

## Heritage

AutoRIFT.jl descends from NASA JPL's [autoRIFT](https://github.com/nasa-jpl/autoRIFT) and produces
the [ITS_LIVE](https://its-live.jpl.nasa.gov/) glacier velocity products, its largest deployment. The
algorithm is that lineage: multi-chip-size normalized cross-correlation with the outlier filter of
Gardner et al. (2018).

The implementation, the API, the geospatial handling and the test suite are this package's own,
validated against synthetic ground truth — a texture displaced by a known amount has an exactly known
answer — and against production output. [`dev/`](dev/README.md) is the record of that validation.

## Not yet supported

- **Geogrid**, the per-pixel geolocation that maps a rotated radar footprint onto a map projection and
  supplies the per-point search radii a radar pair needs. Complex (SLC) arrays correlate — `Coherence`
  with `Deramp` — but assembling a radar product around them is outside the package.
- **GPU** correlation is experimental: only the correlation pass, only `ZNCC`, only a Metal adapter,
  and slower than the threaded CPU path on an otherwise free machine.

## Development

```bash
julia --project=. -e 'using Pkg; Pkg.test()'

julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=benchmark benchmark/run.jl

julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs -e 'include("docs/make.jl")'
```

The test suite runs on a bare clone: the OpenCV fixture corpus is committed, so no Python is needed.
To regenerate fixtures or refresh the timing baseline, see
[`tools/python_ref/README.md`](tools/python_ref/README.md).
[`benchmark/README.md`](benchmark/README.md) covers the benchmark suite and the regression gate.
[`dev/README.md`](dev/README.md) indexes the contributor-facing validation record, which the source
comments cite by bare filename.

For small instances, [`app/`](app/README.md) builds a trimmed standalone binary: bit-identical
displacements at **27.2 MiB peak RSS against 424.2 MiB**, since 97% of an ordinary process's memory
floor is the Julia runtime rather than AutoRIFT.

The correlation can also run on a GPU — `autorift(a, b; backend = :metal)` after `using Metal` — but
this is **experimental and does not outperform the CPU path on an otherwise free machine**. It is
2.7–3.2× a single CPU core on the pass, and *slower* than the threaded CPU correlator on one pair:
0.65 s against **0.23 s** at 1024² on 8 threads. So it pays only where the cores are already busy and
the device is idle, such as a batch driver running one pair per single-threaded process. `dx`/`dy` are
bit-identical to the CPU's; `correlation` differs by up to 1e-5.
[Correlating on a GPU](https://alex-s-gardner.github.io/AutoRIFT.jl/dev/howto/gpu) covers what agrees
and what does not, and how the kernels avoid needing `Float64` on hardware that has none.

## Citing

If this package contributes to published work, please cite:

> Gardner, A. S., Moholdt, G., Scambos, T., Fahnstock, M., Ligtenberg, S., van den Broeke, M., and
> Nilsson, J. (2018). Increased West Antarctic and unchanged East Antarctic ice discharge over the
> last 7 years. *The Cryosphere*, 12(2), 521–547.

## License

MIT. See [`LICENSE`](LICENSE).
