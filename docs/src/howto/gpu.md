# Correlating on a GPU

> **Experimental, and slower than the CPU path on a machine with free cores.** The device does not beat
> the threaded CPU correlator on a single image pair: 0.65 s against **0.23 s** at 1024², chip 32,
> radius 25, with 8 CPU threads. Every speedup below is against **one** CPU core. That comparison applies
> when the other cores are busy, as in a batch driver that runs one pair per single-threaded process
> and leaves the device idle. On an otherwise free machine, `threaded = true` is faster.
>
> Only Metal (Apple GPUs) is implemented. `backend = :cuda` is accepted by `params`, and correlating with it
> throws an `ArgumentError` saying it is not implemented: there is no CUDA extension, so `using CUDA` does
> not help. Only the correlation pass runs on the device, `correlation` differs from the CPU's by up to
> 1e-5, and nothing is measured beyond 2048².

```julia
using AutoRIFT, Metal
out = autorift(image1, image2; chip_size = 32, search_radius = 25, backend = :metal)
```

`backend` selects where the correlation kernels run. `:cpu` is the default and `:metal` needs `using Metal`.
There is no `:gpu`, because naming the vendor is the point of the keyword. Everything else about the call
is unchanged, and `dx` and `dy` are the values the CPU produces. See [What agrees, and what does not](@ref).

`threaded = true` and a GPU backend cannot be combined. `params` throws an `ArgumentError`, because both
parallelize the same loop.

Measurements are on an Apple M2 Max (38 GPU cores) against this package's CPU path, with Julia 1.12.5 and
Metal.jl 1.10.3.

## What it buys

One correlation pass, `track!` over a dense grid:

| scene | chip | radius | points | CPU | GPU | speedup | GPU µs/point |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 640² | 16 | 20 | 5,329 | 0.55 s | 0.19 s | **3.0×** | 34.8 |
| 640² | 32 | 25 | 4,900 | 0.61 s | 0.25 s | **2.5×** | 50.2 |
| 1024² | 32 | 25 | 13,924 | 1.74 s | 0.68 s | **2.6×** | 48.5 |
| 1024² | 64 | 25 | 3,249 | 0.50 s | 0.30 s | 1.7× | 91.3 |
| 2048² | 32 | 25 | 60,516 | 7.52 s | 2.89 s | **2.6×** | 47.8 |

A whole `autorift` call keeps preprocessing, the coarse pass, the outlier filter and the merge on the host:

| scene | grid spacing | CPU | GPU | speedup |
|---:|---:|---:|---:|---:|
| 640² | 32 | 0.045 s | 0.038 s | 1.2× |
| 1024² | 32 | 0.125 s | 0.077 s | 1.6× |
| 1024² | 8 | 0.450 s | 0.195 s | **2.3×** |
| 2048² | 32 | 0.560 s | 0.283 s | 2.0× |

The whole-call gain is smaller than the per-pass gain because the host-side stages are not accelerated.
Their share is smallest on a dense grid, where the correlation dominates. A small scene gains little: the
fixed cost of kernel launches and upload is most of a 640² run with one point per chip.

### It does not beat a threaded CPU on one pair

| | time |
|---|---:|
| 1024², chip 32, radius 25, 8 CPU threads | **0.23 s** |
| the same on the GPU | 0.65 s |

Intra-pair threading on 8 performance cores is 2.8× the device. The GPU is useful when the CPU is
occupied: a batch driver that runs one pair per process gives each process one core, and the device is
otherwise idle. `benchmark/suite/throughput.jl` measures pair-level parallelism at 2.7× intra-pair threading, so a
batch run wants many single-threaded processes, and the device gives each of them a second engine.

## What agrees, and what does not

`dx` and `dy` are bit-identical to the CPU path, with the two exceptions below. `test/gpu.jl` asserts this
over about 15,000 points covering per-point radii, degenerate chips, peaks against the search boundary,
`NoRefine`, and `UInt8` and `Int16` input.

`correlation` agrees to **1e-5**. `peak_ratio` agrees to a distributional bound: the median relative
difference is below 1e-6, the 95th percentile is below 1e-5, and the points beyond that, whose secondary
peak is one of the cancelling shifts described below, are capped by count rather than by size. These
outliers are a larger share than for `correlation`, because a ratio's denominator is a single sample,
where a mean over the whole surface dilutes any one of them. A textured scene stays within 6.6e-7 over
4900 points, while a scene with a constant patch puts 86 of 4416 points above 1%. `NaN` and `Inf` values of
`peak_ratio` agree exactly, because a count and a sign decide them rather than arithmetic.

The difference in `correlation` comes from the numerator: the device computes it with MPSGraph and the CPU
with FFTW, and two transform libraries sum in different orders. A peak's location is about three orders of
magnitude less sensitive than its value (a 1e-5 perturbation of the surface moved no peak in 3000 weak
surfaces), so `dx` and `dy` are unaffected. FFTW wisdom has the same effect on the CPU: it moves
`correlation` by 3.6e-7 and leaves `dx` and `dy` unchanged (`src/plans.jl`).

The two exceptions:

1. **One subpixel step**, on about 4 points in 15,000, where the upsampled patch has a near-tie. At half of
   them the two candidates from the CPU's own cascade are bit-identical, so neither answer is more
   correct. At the rest they differ by about 2e-7. The difference is at most one step.

2. **Points where the CPU surface is not a correlation coefficient.** See below.

### Where the two disagree: exactly constant windows

Over an exactly constant part of a search window, such as saturated snow or a fill value that no validity
mask caught, `ΣW² − (ΣW)²/n` cancels to within rounding of zero. A `Float32` numerator's absolute error
divided by that denominator, about 1e-6, is a large ratio, and the CPU surface leaves `[-1, 1]`. On a 640²
scene with a saturated 201² block, chip 32, radius 25 and the default `Highpass`, 1.5% of surfaces exceed 1,
with a worst value of 7.1. A near-constant block (one DN of sensor noise) produces no excursion, so this
needs exact constancy.

The device path does not do this, because its gather removes the window mean before transforming. The two
paths differ at those points, and `test/gpu.jl` excludes points whose CPU surface escapes `[-1, 1]`.

The effect is confined to `correlate!`. Through `autorift` at default settings the same scene reports zero
points wrong by more than a pixel and zero `correlation` above 1, at one chip-size level and at three,
because a peak from a vanishing denominator is spatially incoherent and `GardnerFilter` removes it. Only
`outliers = :none` keeps those points. `src/correlate.jl` records the measurement and the fix that would
work if needed: a relative clamp on the variance. Mean removal is not that fix. It measures worse than doing
nothing, because the tables would then be rebuilt on the shifted window.

## How it works

The device path is a batched version of `_track_chunk!`. Each of the eight per-point steps is a kernel over
the whole batch, the buffers stay on the device between them, and only one value per point returns to the
host. Kernels are written against KernelAbstractions.jl and transforms go through AbstractFFTs, so a second
device would need only an adapter file like `ext/AutoRIFTMetalExt.jl`.

Apple GPUs have no `Float64`, which the CPU's summed-area tables need. The device subtracts each search
window's mean before summing and accumulates in double-single `Float32`, which reaches `Float64` accuracy.
The subpixel cascade is the largest stage on both the device and the CPU.

Two budgets control batch sizes: `GPU_MEMORY_BUDGET` (512 MiB) for the correlation batch and
`REFINE_MEMORY_BUDGET` (1 GiB) for the cascade's tile. Neither changes the answer; `test/gpu.jl` asserts that
a pass split into several batches is bit-identical to one batch.

`dev/gpu-feasibility.md` has the error analysis, the per-stage timings, and the kernel design.

## What is not on the device

Preprocessing, the coarse pass and its dilation, the outlier filter, hole filling and the chip-size merge
run on the host, and the pass still returns a host `DisplacementField`. These stages scale with the number
of grid points, not the number of shifts searched, so they cap the whole-call speedup.

Only `ZNCC` has a device kernel. `NCC` differs by one denominator term and `Coherence` needs complex
transforms. A device pass with either throws an `ArgumentError` that names the measure, because a silent
fallback would give a caller a different measure than the one requested. The check runs only on passes
large enough to reach the device (next paragraph), so a smaller pass with `NCC` or `Coherence` runs on the
CPU without error.

A pass with fewer than `GPU_MIN_BATCH` (256) searchable points runs on the CPU, because a batched transform
is slower than FFTW below a few hundred points. This case is routine: the sparse search zeroes most of the
grid, so coarse passes and late chip-size levels reach it often.

## Requirements

Metal.jl 1.10 or later, which provides the batched FFT this depends on (PR #713), and an Apple GPU on
macOS 14 or later. `Metal.functional()` reports whether the hardware answers. On an Intel Mac or in a VM
the package loads and `Metal.functional()` returns `false`, and a pass with `backend = :metal` then throws
an `ArgumentError` instead of falling back to the CPU.

GPU support does not affect a trimmed `--trim=safe` binary. The backend is a `Params` type parameter, so
`app/`'s positional `CPU()` resolves statically, and the kernels are in a package extension that a binary
without Metal never loads.
