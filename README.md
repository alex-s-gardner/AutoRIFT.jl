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

## Performance

Two measurements against the Python reference (`nasa-jpl/autoRIFT` v2.1.2 inside `hyp3-autorift` 0.28.4),
both on the golden test set, 12 threads on an Apple M2 Max, one process per case. They answer different
questions: the first is the whole pipeline from a granule, the second is the correlator on identical
inputs.

### From the granule to `dx`/`dy`

Every stage on the Julia side — geogrid, scene read, filter, coregistration, correlation — against the
reference container's own logged wall clock from the last download to the finished product. The scenes are
on local disk before either clock starts. The 15 optical and burst cases are below; the three Sentinel-1
full-SLC pairs and two NISAR granules the chain also reaches are in `dev/GATES.md`, which carries the
per-stage breakdown for all twenty and names what the remaining two need.

Over those twenty the chain spends 7,837 s against the reference's 19,320 s, **2.5x** — a smaller ratio than
the table below because the two NISAR granules are 79% of the Julia total and are the only cases the
reference is not beaten on threefold.

| case | Julia s | Python s | | case | Julia s | Python s | |
|---|---:|---:|---:|---|---:|---:|---:|
| LC08 `009011` | 54.9 | 376 | 6.8x | LT05 `001013` | 49.9 | 190 | 3.8x |
| LC08 `060018` | 67.2 | 328 | 4.9x | LT05 `060018` | 20.7 | 118 | 5.7x |
| LC08 `062018` | 33.0 | 289 | 8.8x | S1A `1SSV_20240618` a | 460.3 | 1536 | 3.3x |
| LC09 `215109` | 39.9 | 327 | 8.2x | S1A `1SSV_20240618` b | 176.3 | 632 | 3.6x |
| LE07 `061018` (2012) | 85.3 | 271 | 3.2x | S1C `1SSV_20250416` | 45.1 | 383 | 8.5x |
| LE07 `061018` (2013) | 79.7 | 358 | 4.5x | S2A `20200626` | 14.4 | 186 | **12.9x** |
| LE07 `063018` | 94.1 | 335 | 3.6x | S2B `20200612` | 21.0 | 187 | 8.9x |
| LT04 `063018` | 29.4 | 195 | 6.6x | **all 15** | **1,271** | **5,711** | **4.5x** |

**Median 6.6x.** Smaller than the correlator's ratio below, because the correlator is 4 to 63% of a Julia
run here and the stages in front of it are GDAL and arithmetic on both sides.

**Peak memory runs the other way on this path**, and the reason is worth stating rather than hiding: the
chain holds the raw band, the filtered band and the crop as whole-scene `Float32` arrays where the
reference writes its filtered scene to disk and reads back a window.

| case | Julia peak | Python peak | |
|---|---:|---:|---:|
| S2B `20200612` | 4.09 GiB | 2.01 GiB | Julia 2.03x |
| LC08 `062018` | 7.87 GiB | 4.42 GiB | Julia 1.78x |
| LT05 `060018` | 8.35 GiB | 5.80 GiB | Julia 1.44x |

Filtering per block would remove it — the correlator already does exactly that internally — except for the
Landsat 4/5 pairs, whose native filter is a band-reject over the whole scene.

Reproduce with `tools/golden/e2e_run.jl`.

### The correlator alone

The same captured inputs on both sides, so neither re-derives the grid. `jl block` is the
`process_block_size` the library picks unasked.

| case | jl block | Julia s | Python s | |
|---|---|---:|---:|---:|
| LC08 `009011` | 1024 | 19.6 | 180.3 | 9x |
| LC08 `060018` | 1024 | 6.1 | 46.1 | 8x |
| LC08 `062018` | 1024 | 8.3 | 86.2 | 10x |
| LC09 `215109` | 1024 | 6.1 | 77.8 | 13x |
| LE07 `061018` (2012) | 1024 | 2.6 | 13.1 | 5x |
| LE07 `061018` (2013) | 1024 | 6.1 | 51.6 | 8x |
| LE07 `063018` | 1024 | 8.9 | 85.5 | 10x |
| LT04 `063018` | 1024 | 5.0 | 60.2 | 12x |
| LT05 `001013` | 1024 | 2.6 | 37.8 | **14x** |
| LT05 `060018` | 1024 | 3.4 | 19.5 | 6x |
| NISAR L1 RSLC | `3593x1843` | 467.2 | 812.2 | **2x** |
| NISAR L2 GSLC | `2883x1531` | 247.3 | 516.4 | **2x** |
| S1A `1SSH_20150828` | 1024 | 16.6 | 111.0 | 7x |
| S1A `1SSH_20151120` | 1024 | 10.5 | 86.8 | 8x |
| S1A `1SSH_20170221` | 1024 | 25.1 | 173.6 | 7x |
| S1A `1SSV_20240618` a | 1024 | 30.1 | 175.5 | 6x |
| S1A `1SSV_20240618` b | 1024 | 13.2 | 63.7 | 5x |
| S1B `1SDH_20180809` | 1024 | 16.9 | 111.1 | 7x |
| S1C `1SDV_20250416` | 1024 | 12.5 | 115.8 | 9x |
| S1C `1SSV_20250416` | 1024 | 4.1 | 33.4 | 8x |
| S2A `20200626` | 1024 | 3.7 | 30.8 | 8x |
| S2B `20200612` | 1024 | 5.9 | 36.0 | 6x |

**Median 8x, best 14x, worst 2x.** The two NISAR granules are the worst because they are the largest:
the mean search window on the L1 grid is 115,000 px against 4,300 on a Landsat one, which puts both
sides in the regime where the transform is the whole cost and the reference's per-point overhead no
longer dominates. Measured over these cases AutoRIFT.jl sustains 31-41 G-butterfly/s of transform work
where the reference reaches 2.3 on the smallest windows and 15.7 on the largest, so the ratio a case
shows is mostly a statement about its window size.

**Julia measures 0.875 to 1.000 of the reference's point count, median 0.987.** A run that measures fewer
points did less work, so read the speedups with that in mind. The remaining differences are catalogued
in [`dev/CORRECTNESS.md`](dev/CORRECTNESS.md) — each one a behaviour reproduced deliberately, or a
deliberate divergence with the measurement behind it.

**Memory is where blocked processing shows.** The Julia columns are the run's peak *above* what the
harness already held, so they are what the correlation itself costs; Python's is `ru_maxrss` for the whole
process, which carries its interpreter and the inputs. Comparable within a column, indicative across:

| case | Julia untiled | Julia blocked | Python, whole process |
|---|---:|---:|---:|
| NISAR L2 GSLC | 56.5 GiB | **6.1 GiB** | 49.4 GiB |
| NISAR L1 RSLC | 35.0 | **11.0** | 26.8 |
| S1B `1SDH_20180809` | 18.9 | **0.4** | 14.9 |

Every block size measured gives an answer bit-identical to the untiled one — asserted on all 22 cases,
`dx` and `dy` both, at equal point counts. The largest correlation footprint at the default block is
NISAR L1's 11.0 GiB.

Reproduce with `tools/golden/e2e_table.jl`; the full record, including block-size sweeps and the memory
budget, is in [`dev/plan-16gib.md`](dev/plan-16gib.md).

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
