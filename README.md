# AutoRIFT.jl

[![Build Status](https://github.com/alex-s-gardner/AutoRIFT.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/alex-s-gardner/AutoRIFT.jl/actions/workflows/CI.yml?query=branch%3Amaster)

Dense feature tracking by normalized cross-correlation: a pure-Julia
reimplementation of NASA JPL's [autoRIFT](https://github.com/nasa-jpl/autoRIFT),
the correlator behind the [ITS_LIVE](https://its-live.jpl.nasa.gov/) glacier
velocity products.

Given two images of the same scene acquired at different times, `autorift`
estimates the displacement of surface features between them, on a grid, to
sub-pixel precision.

> **Status: under construction.** Parameters, point sets, and the test and
> benchmark infrastructure are in place. The correlator itself is next. See
> *Progress* below.

```julia
using AutoRIFT

out = autorift(image1, image2; chip_size = 32, search_radius = 25)
out.dx, out.dy, out.correlation
```

## Design

**Correctness over bug-compatibility.** The goal is to match or beat OpenCV, not
to reproduce the reference's defects. Primitives are verified bit-exact against
OpenCV where OpenCV's semantics are the standard and are correct; the pipeline is
verified against *synthetic ground truth*, where a texture displaced by a known
amount has an exactly known answer — a stronger check than agreement with an
implementation that has no test suite of its own. Where AutoRIFT.jl deliberately
differs from the reference, `REFERENCE.md` says so and why.

**Coordinate-agnostic.** The core works on plain matrices in pixel coordinates, so
map-projected and radar slant-range imagery are handled identically. Load
`Rasters` or `DimensionalData` to pass dimensional arrays and get coordinates and
CRS back; those methods live in package extensions, so a core that only needs to
correlate two matrices does not pay to load GDAL.

**Points, not necessarily grids.** Each search center is independent — its own
coordinates, search radii, prior displacement, and chip size. Centers may be
scattered at arbitrary fractional coordinates or laid out on a grid; only the
multi-scale pyramid requires the latter, and it says so in its signature.

**Built for throughput.** autoRIFT runs on tens of millions of image pairs, so
allocation-free inner loops and reusable buffers are correctness properties here,
asserted in the test suite rather than merely hoped for. A cache lifecycle
(`init` / `reinit!` / `autorift!`) lets buffers and FFT plans be reused across
pairs, which for batch processing matters more than any kernel optimization.

## Reference implementation

The target is **nasa-jpl/autoRIFT@v2.1.2** for the algorithm core, driven by
**ASFHyP3/hyp3-autorift**'s vendored production scripts — HyP3 is what actually
generates ITS_LIVE products, and its drivers override upstream in ways that change
results. See [`REFERENCE.md`](REFERENCE.md) for what changed in v2.0.0 (a great
deal), what the production drivers change (including the order of the pipeline),
and how validation works given that neither upstream repository has a test that
exercises the correlator.

## Progress

- [x] Package scaffold, parameter resolution, point sets
- [x] Test and benchmark infrastructure, OpenCV fixture corpus, Python baseline
- [x] Correlation surface, peak location, subpixel refinement
- [x] Sliding-window reductions, pre-correlation filters, outlier rejection
- [x] The grid loop
- [x] Multi-chip-size search (not a pyramid — the imagery is never downsampled)
- [x] Public API and cache lifecycle
- [x] Rasters / DimensionalData extensions
- [x] Performance milestone — 3.1x serial, 11.1x threaded against the reference
- [x] Memory as a tracked metric, and a trimmed 29.7 MiB binary
- [ ] Complex (SLC) input
- [ ] Geogrid

## Development

```bash
julia --project=. -e 'using Pkg; Pkg.test()'

julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=benchmark benchmark/run.jl
```

The test suite runs on a bare clone: the OpenCV fixture corpus is committed, so no
Python is needed. To regenerate fixtures or refresh the timing baseline, see
[`tools/python_ref/README.md`](tools/python_ref/README.md).
[`benchmark/README.md`](benchmark/README.md) covers the benchmark suite and the
regression gate, and [`docs/memory.md`](docs/memory.md) the memory comparison —
including why peak RSS and live heap answer different questions.

For small instances, [`app/`](app/README.md) builds a trimmed standalone binary:
byte-identical output at **29.7 MiB peak RSS against 424.2 MiB**, since 97% of an
ordinary process's memory floor is the Julia runtime rather than AutoRIFT.
