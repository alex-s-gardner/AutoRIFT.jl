# Correlating on a GPU

> **Experimental, and slower than the CPU path on a machine with cores free.** The device does not
> beat this package's threaded CPU correlator on a single image pair: 0.65 s against **0.26 s** at
> 1024², chip 32, radius 25 on 8 threads. Every speedup quoted below is against **one** CPU core,
> which is the comparison that matters only when the other cores are already busy — a batch driver
> running one pair per single-threaded process, with the device otherwise idle. On an otherwise free
> machine, `threaded = true` is both faster and the supported path.
>
> Treat `backend` as a preview: only the correlation pass runs on the device, only the Metal adapter
> exists, `correlation` differs from the CPU's by up to 1e-5, and none of it is measured beyond
> 2048² — the full-scene table in `tools/ab/` is CPU-only. Use it to evaluate the device, not to
> produce results you depend on.

```julia
using AutoRIFT, Metal
out = autorift(image1, image2; chip_size = 32, search_radius = 25, backend = :metal)
```

`backend` selects where the correlation kernels run. `:cpu` is the default; `:metal` needs
`using Metal` and `:cuda` needs `using CUDA`. Everything else about the call is unchanged, and
`dx`/`dy` are the same values the CPU produces — see [What agrees, and what does not](#what-agrees-and-what-does-not).

Measured on an Apple M2 Max (38 GPU cores) against this package's own CPU path, Julia 1.12.5,
Metal.jl 1.10.3.

## What it buys

One correlation pass, `track!` over a dense grid:

| scene | chip | radius | points | CPU | GPU | speedup | GPU µs/point |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 640² | 16 | 20 | 5,329 | 0.58 s | 0.18 s | **3.2×** | 34.5 |
| 640² | 32 | 25 | 4,900 | 0.68 s | 0.25 s | **2.8×** | 50.2 |
| 1024² | 32 | 25 | 13,924 | 1.90 s | 0.70 s | **2.7×** | 50.2 |
| 1024² | 64 | 25 | 3,249 | 0.55 s | 0.30 s | 1.9× | 91.2 |
| 2048² | 32 | 25 | 60,516 | 8.44 s | 3.02 s | **2.8×** | 49.9 |

A whole `autorift` — three chip-size levels, with preprocessing, the coarse pass, the outlier
filter and the merge all still on the host:

| scene | grid spacing | CPU | GPU | speedup |
|---:|---:|---:|---:|---:|
| 640² | 32 | 0.037 s | 0.036 s | 1.0× |
| 1024² | 32 | 0.118 s | 0.072 s | 1.6× |
| 1024² | 8 | 0.456 s | 0.194 s | **2.4×** |
| 2048² | 32 | 0.567 s | 0.283 s | 2.0× |

Two things to read off that. The end-to-end gain is smaller than the per-pass one because
**only the pass is on the device** — Amdahl's share of the host-side pipeline sets the ceiling,
and it is largest on a dense grid, which is where the correlation dominates. And a small scene
gains nothing: the fixed cost of the launches and the upload is the whole of a 640² run at one
point per chip.

### It does not beat a threaded CPU on one pair

| | time |
|---|---:|
| 1024², chip 32, radius 25, 8 CPU threads | **0.26 s** |
| the same on the GPU | 0.65 s |

That is not a defect to be fixed later, it is what the numbers say: intra-pair threading across
8 performance cores is 2.5× the device. **The GPU is worth using when the CPU is otherwise
occupied** — a batch driver running one pair per process, where each process has one core and
the device is idle — and not as a way to make a single pair faster on an otherwise free machine.
`benchmark/suite/throughput.jl` already measures pair-level parallelism at 2.7× intra-pair
threading, so the shape a batch run wants is many single-threaded processes, and this gives each
of them a second engine.

`threaded = true` and a GPU backend cannot be combined, and the combination is rejected rather
than silently resolved: both parallelise the same loop.

## What agrees, and what does not

**`dx` and `dy` are bit-identical to the CPU path.** That is the gate, asserted in `test/gpu.jl`
over ~15,000 points across per-point radii, degenerate chips, railed peaks, `NoRefine`, and
`UInt8`/`Int16` input, with two stated exceptions.

`correlation` agrees to **1e-5** and `peak_snr` to a distributional bound — median 2.5e-7 and 99th
percentile 1.0e-6, with a handful of genuine outliers whose *background* includes the cancelling
shifts described below, capped by count rather than by magnitude. The asymmetry is
not a weakness in the port: the numerator is computed by MPSGraph where the CPU uses FFTW, and no
amount of care makes two transform libraries reassociate a sum identically. What makes it
acceptable is that a peak's *location* is three orders of magnitude less sensitive than its value
— measured, a 1e-5 perturbation of the surface moved no peak in 3000 deliberately weak surfaces —
which is the same observation `src/plans.jl` records about FFTW wisdom moving `correlation` by
3.6e-7 with `dx`/`dy` untouched.

The two exceptions:

1. **One subpixel step**, on about 4 points in 15,000, where the upsampled patch has a near-tie.
   Verified benign: at half of them the two candidate samples of the CPU's *own* cascade are
   bit-identical, so neither answer is more correct; at the rest they differ by ~2e-7. Bounded by
   one step, so it never compounds.

2. **Points where the CPU surface is not a correlation coefficient at all.** See below.

### Where the two genuinely disagree: exactly-constant windows

Over an **exactly** constant part of a search window — saturated snow, or a fill value no validity
mask caught — `ΣW² − (ΣW)²/n` cancels to within rounding of zero, and a `Float32` numerator's
absolute error over a ~1e-6 denominator is a large ratio. The CPU surface then leaves `[-1, 1]`:
measured on a 640² scene with a saturated 201² block, chip 32 and radius 25, under the default
`Highpass`, **1.5% of surfaces exceed 1 with a worst value of 7.1**. A *near*-constant block — one
DN of sensor noise — produces no excursion at all, so this needs exact constancy rather than low
contrast.

The device path does not do this, because the gather removes the window mean before transforming.
So the two genuinely differ at those points, and `test/gpu.jl` excludes points whose CPU surface
escapes `[-1, 1]` rather than hiding them under a loose tolerance.

**This is a contract violation in `correlate!`, not a defect in reported output, and it is not
worth fixing.** Through `autorift` at the default settings the same scene reports **zero** points
wrong by more than a pixel and **zero** `correlation` above 1, at one chip-size level and at three:
a garbage peak from a vanishing denominator is spatially incoherent, which is exactly what
`GardnerFilter` tests for. Only `outliers = :none` leaks them, and that setting exists for
diagnosis. `src/correlate.jl` records the measurement, and what the fix would be if it were ever
needed — a relative clamp on the variance, *not* mean-removal, which measures worse than doing
nothing because the tables would then have to be rebuilt on the shifted window.

## How it works, and the two constraints that shaped it

The device path is a batched sibling of `_track_chunk!`: each of the eight per-point steps becomes
a kernel over the whole batch, the buffers live on the device between them, and only five short
vectors — one value per point — come back. No correlation surface crosses the bus.

Kernels are written against **KernelAbstractions.jl** and transforms through **AbstractFFTs**, so
`ext/gpu/` is vendor-neutral and a second device is an adapter file rather than a second
implementation. Only the Metal adapter ships today.

### Apple GPUs have no Float64

`src/integral.jl` requires the summed-area tables to be `Float64`, because the variance
differences two large nearly-equal quantities. Metal has no `Float64`, so the cancellation is
**removed** rather than absorbed: the gather subtracts the search window's own mean, which is exact
for both the numerator (`Σ T′·(W − c) = Σ T′·W`, since `Σ T′ = 0` by construction) and the variance
(shift invariance). The tables then hold O(contrast) values instead of O(DC).

Worst relative error in the denominator, over chips 16–128, DC 1 to 1e4, contrast 1.0 down to
0.001, against a two-pass `Float64` computation:

| | worst |
|---|---:|
| `Float64` tables (the CPU path) | 2.8e-8 |
| `Float32` tables, raw window | **8.4** |
| `Float32` tables, centered window | 3.6e-6 |
| `Float32` **double-single**, centered window | **3.9e-8** |

The naive downgrade is catastrophic, exactly as that file predicts. Centering recovers all but a
factor of ~100, and a double-single accumulator — an unevaluated `Float32` pair with Knuth's
`twosum` — closes the rest to `Float64` accuracy. `twosum` is verified **exact on the Metal
device**: bit-identical to the host with the error term nonzero on 97% of entries. That check is a
precondition, not a nicety — a compiler that reassociated it would zero the error term silently and
leave a kernel that runs and buys nothing.

Centering also makes the **numerator more accurate than the CPU's**, for the same reason:

| numerator arm | worst relative error |
|---|---:|
| FFTW, raw window (**the CPU path today**) | 6.05e-5 |
| MPSGraph, raw window | 6.41e-5 |
| FFTW, centered window | 3.88e-5 |
| MPSGraph, centered window | **3.91e-5** |

### A GPU thread is slow; only width helps

Three kernels were written one-workitem-per-point and all three had to be rewritten. The
measurements are recorded because the arrangement looks right each time:

| | one workitem per point | parallel over elements |
|---|---:|---:|
| subpixel cascade, 64× | 973 µs/pt | 14 µs/pt |
| the cascade's final 320×320 argmax | 113 µs/pt | ~2 µs/pt |
| refinement total | **1012 µs/pt** | **46 µs/pt** |

The first cost 97% of the whole correlator and made the device **7× slower than the CPU**. Its
signature was diagnostic: cost tracking the element count exactly (4× per doubling of the
upsampling) and **flat in the point count** — serial work on one thread, not launch overhead.

The argmax was the subtler one, because a reduction under an *order-dependent* tie rule looks
unsplittable. It splits exactly, because `peak_index`'s rule is a total order on
`(value, row, column)`: reduce per column, then over the column winners. Bit-identical by
construction rather than by tolerance.

### An out-of-bounds read is a `KernelException`, not a `BoundsError`

Worth stating because it changes how the tests have to be written. The search radius is a
*per-point* field, so a point whose surface is narrower than the 5-wide refinement patch is a legal
input — `subpixel_peak` returns its integer peak unrefined. The device kernel originally read the
patch unconditionally, and on a mixed-radius point set that reads past the surface.

On the host that is a `BoundsError` naming a line. On a device it surfaces as a
`KernelException: A BoundsError was thrown on device Apple M2 Max` from whatever later call
happens to synchronize, with no indication of which kernel or which index. It also did not appear
in any single-radius test, only in a pass mixing refinable and unrefinable points — so
`test/gpu.jl` exercises that mix deliberately rather than testing one radius at a time.

Two knobs follow from this and are both memory budgets, because the cost is per *launch* rather
than per element: `GPU_MEMORY_BUDGET` sizes the correlation batch and `REFINE_MEMORY_BUDGET` the
cascade's tile. **Neither changes the answer** — `test/gpu.jl` asserts a pass split into several
batches is bit-identical to one batch, which is what makes them free to retune.

### Where the time goes now

Per point at 1024², chip 32, radius 25, a batch of 1024:

| stage | µs/point |
|---|---:|
| subpixel cascade, 64× | 46.0 |
| gather (chip + window, both mean-removed) | 12.6 |
| peak, boundary, prominence | 5.6 |
| zero-fill and place the transform buffers | 3.5 |
| forward transforms (×2) | 2.2 |
| inverse transform | 2.3 |
| summed-area tables (two passes) | 3.8 |
| normalize | 1.2 |
| spectral multiply | 0.5 |
| **total** | **77.6** |

`gather` and `peak` are the two remaining one-workitem-per-point kernels, and are the obvious next
targets — together 18 µs of 78. The cascade is still the largest single stage even after the
rewrite, which is the same conclusion the CPU profile reached: `docs/gpu-feasibility.md` measures
it at 78 µs of a 97 µs CPU point at chip 16, fixed in chip size because it upsamples a 5×5 patch to
320×320 whatever the surface was.

## What is not on the device

Stated so the ceiling above is not mistaken for a bug: preprocessing, the coarse pass and its
dilation, the outlier filter, hole filling, and the chip-size merge all run on the host, and
`run_pass` still returns a host `DisplacementField`. Those are per-grid-point rather than
per-shift work.

`ZNCC` only. `NCC` differs by one denominator term and `Coherence` needs complex transforms;
neither has a device kernel, and both are **refused by name** rather than silently falling back —
a caller who asked for a measure and got another has no way to tell.

A pass with fewer than `GPU_MIN_BATCH` searchable points runs on the CPU instead, since a batched
transform measures *slower* than FFTW below a few hundred points. That is a routine case rather
than a corner: the sparse search deliberately zeroes most of the grid, so coarse passes and late
chip-size levels reach it constantly.

## Requirements

Metal.jl **1.10 or later** — the batched FFT this depends on landed there (PR #713) — with macOS
14–26 and an Apple GPU. `Metal.functional()` reports whether the hardware answers; on an Intel Mac
or in a VM the package loads and this returns `false`, and selecting `:metal` then errors rather
than falling back.

The GPU support is invisible to a trimmed `--trim=safe` binary, by construction and in three
independent ways: the backend is a `Params` type parameter so `app/`'s positional `CPU()` resolves
statically, the kernels live in a package extension that a binary without Metal never loads, and
nothing new is reachable from `main`. Verified — the trimmed binary still builds with zero verifier
errors and its output is byte-identical.
