# GPU correlation: measured feasibility

Measurements taken on an Apple M2 Max (38 GPU cores, macOS 26.6.1) with Julia 1.12.5,
Metal.jl 1.10.3, against this package at `b4fb528`. Every number below is reproducible
from the probe scripts described at the end.

## Summary

A GPU correlation path is viable. Four things are established:

1. Batched `rfft`/`irfft` over region `(1, 2)` of an `(fy, fx, B)` array works and is
   **8-17x** FFTW's throughput for the same total work, at batch 1024 and above.
2. The Float64 summed-area tables can become Float32 **without loss that reaches the
   output**, by subtracting the search window's own mean before tabulating.
3. `dx` and `dy` are **bit-identical** to the CPU path across 5400 deliberately weak,
   contested surfaces. `correlation` differs by up to 6.1e-6.
4. The MPSGraph numerator over a centered window is **more accurate than the FFTW
   numerator this package computes today** — 3.91e-5 against 6.05e-5 relative to a
   Float64 direct evaluation.

And one thing that changes where the work should go: **the subpixel cascade, not the
transform, is the dominant per-point cost** at the geometry ITS_LIVE runs.

## Transform sizes and throughput

`next_fft_size` maps the window extents real configurations reach as follows:

| chip | radius | window | `next_fft_size` |
|---:|---:|---:|---:|
| 16 | 6 | 27 | 28 |
| 16 | 20 | 55 | 56 |
| 32 | 25 | 81 | 84 |
| 64 | 25 | 113 | 128 |
| 128 | 25 | 177 | 192 |

Batched forward transform, Float32, against FFTW planned with `MEASURE` on the same
data. Ratios are FFTW time over GPU time, so above 1 favours the GPU:

| n | B = 64 | B = 256 | B = 1024 | B = 4096 |
|---:|---:|---:|---:|---:|
| 28 | 0.16 | 0.51 | 3.24 | 8.67 |
| 56 | 0.76 | 1.43 | 6.92 | 8.87 |
| 84 | 0.98 | 2.84 | 8.83 | 11.48 |
| 128 | 1.84 | 8.33 | 13.90 | 16.97 |
| 192 | 2.85 | 6.51 | 12.68 | 13.47 |

The inverse transform runs 5-11x. Two consequences for the design: **batches must be
large** — below 256 the GPU loses at small transform sizes, because per-launch overhead
dominates — and a pass with few searchable points should stay on the CPU. Round-trip
accuracy against FFTW is 4e-7 to 1e-6 relative, which is the transform-library difference
that sets the surface tolerance.

`next_fft_size`'s outputs are adequate for MPSGraph and need no separate padding rule.
Powers of two are modestly better per point (64 at 0.83 ms against 56 at 0.91 ms for
B = 1024) but not enough to justify transforming a larger array; the one size to avoid is
177, which at 27.7 ms is **8x worse** than 192 — and `next_fft_size` already returns 192
there. One caveat for a future padding change: a prime-heavy size is catastrophic on
MPSGraph rather than merely slow.

## Float32 tables

`src/integral.jl` requires Float64 because the variance is `ΣW² − (ΣW)²/n`, a difference
of two large nearly-equal quantities. Apple GPUs have no Float64, so the cancellation has
to be removed rather than absorbed.

Both the numerator and the variance are invariant to subtracting a constant from the
window — the numerator because `Σ T′·(W − c) = Σ T′·W − c·Σ T′` and `Σ T′ = 0` by
construction, the variance by definition. So the window's own mean is subtracted during
the gather and the tables hold O(contrast) values instead of O(DC).

Worst relative error in the *denominator* over chips 16-128, radius 20-25, element types
`Float32`/`UInt8`/`Int16`, DC 1 to 1e4 and contrast 1.0 down to 0.001, scored against a
two-pass Float64 direct computation:

| arm | worst |
|---|---:|
| Float64 tables (what the package does) | 2.8e-8 |
| Float32 tables, raw window | **8.4** |
| Float32 tables, centered window | 3.6e-6 |
| Float32 double-single, centered window | 3.9e-8 |

The naive downgrade is catastrophic, exactly as `src/integral.jl` predicts — at DC 1e4 and
contrast 0.001 the answer is meaningless. Centering recovers all but a factor of ~100,
and the residual is ordinary accumulation error along the recurrence rather than
cancellation. A double-single (Float32 pair, Knuth `twosum`) accumulator closes the rest
and is as accurate as Float64.

**`twosum` is exact on the Metal device**: the pair arithmetic executed in a
KernelAbstractions kernel is bit-identical to the host's, with 12365 of 12800 error terms
nonzero, and the pair reproduces a Float64 reference exactly where plain Float32 errs by
6.7e-7. The Metal compiler does not reassociate it. This had to be checked — a compiler
that reassociates would zero the error term silently and leave a kernel that runs and
quietly buys nothing.

## What reaches the output

The gate is `dx`/`dy` bit-identical, so the question is not how large the surface error is
but whether it moves an argmax.

On 3000 weak surfaces — median peak correlation 0.303, tenth-percentile peak-to-rival
margin 0.005, minimum 1e-6 — a per-shift multiplicative perturbation of the surface moves
the integer peak in:

| perturbation | all surfaces | most contested decile |
|---:|---:|---:|
| 3.9e-8 | 0 | 0 |
| 1.0e-6 | 0 | 0 |
| 3.6e-6 | 0 | 0 |
| 1.0e-5 | 0 | 0 |
| 1.0e-4 | 0.03% | 0.33% |
| 1.0e-3 | 0.17% | 1.67% |

Nothing moves below 1e-5. The peak's *location* is three orders of magnitude less
sensitive than its value, which is the same observation `src/plans.jl` records about FFTW
wisdom.

End to end — batched MPSGraph numerator plus Float32 tables against the package's own
`correlate!`, on 5400 weak decorrelated pairs across chips 16/32/64, three element types,
and contrast 1.0 and 0.05:

| quantity | result |
|---|---|
| integer peak differs, double-single tables | **0 of 5400** |
| integer peak differs, plain Float32 tables | **0 of 5400** |
| subpixel `dx` differs | **0 of 5400** |
| subpixel `dy` differs | **0 of 5400** |
| worst `correlation` relative error | 6.1e-6 |

That plain Float32 and double-single tables produce *identical* integer peaks locates the
remaining error in the numerator, where the tables cannot reach it.

## The numerator, and a correction to the plan's assumption

Numerator relative error against a Float64 direct evaluation, scaled by the largest
numerator on each surface, worst over chips 16/32/64 and contrast 1.0/0.05/0.01:

| arm | worst |
|---|---:|
| FFTW, raw window (**the current CPU path**) | 6.05e-5 |
| FFTW, centered window | 3.88e-5 |
| MPSGraph, raw window | 6.41e-5 |
| MPSGraph, centered window | **3.91e-5** |

Two readings. Centering helps the *numerator* as much as it helps the tables, and for the
same reason: a raw window puts a large DC component into the transform, whose magnitude
sets the Float32 rounding at every frequency, while the numerator is invariant to it. And
MPSGraph and FFTW are equivalent to within 5% of each other's error — so the GPU numerator
over a centered window is **better than what this package computes today**, not a
concession.

The `correlation` tolerance is therefore a property of Float32 transforms at low contrast,
which the CPU path already has, rather than something the GPU introduces. The stated 1e-6
gate on `correlation` was set before this was measured and is **not achievable — nor is it
the right target**, since the CPU path does not meet it against exact arithmetic either.
The honest gate is `dx`/`dy` bit-identical with `correlation` to 1e-5.

## Where the time actually goes

Per-point cost, Float32 input, `PyramidRefine(64)`, minimum of many samples:

| chip | radius | tables | surface | peak+quality | subpixel | total | surface share |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 20 | 2.0 µs | 17.1 µs | 2.1 µs | **78.0 µs** | 97.2 µs | 18% |
| 32 | 25 | 5.0 | 42.0 | 3.4 | **77.8** | 123.1 | 34% |
| 64 | 25 | 10.1 | 71.7 | 3.4 | **77.7** | 152.8 | 47% |
| 128 | 25 | 25.7 | 177.9 | 3.3 | **77.7** | 258.9 | 69% |

**The subpixel cascade is 78 µs regardless of chip size, and it is 80% of a point at chip
16.** That is the geometry ITS_LIVE runs Landsat at (`test/realdata.jl`: chips 16/32/64).
It is fixed-cost because the cascade upsamples a 5x5 patch to 320x320 whatever the surface
was.

This reorders the work. The plan listed the cascade last among seven kernels; on the
evidence it is the **first** thing that must move, because a GPU that accelerates only the
transform is capped at 1.2x at chip 16 and 1.9x at chip 32 by Amdahl. Correlating on the
device and refining on the host is not worth building.

Tables are a consistent 12-14% of surface formation, so the double-single accumulator's
extra arithmetic lands on a small share of a small share.

## Reproducing

Scripts live outside the repository, under the job's scratch directory, since they pin
Metal.jl 1.10.3 in a throwaway environment:

| script | what it answers |
|---|---|
| `fft_probe.jl` | batched transform correctness, throughput, size sensitivity |
| `precision_probe.jl` | Float64 vs Float32 vs centered-Float32 tables |
| `ds_probe.jl` | the double-single variant |
| `device_probe.jl` | `twosum` exactness on device; peak sensitivity |
| `peak_probe.jl` | peak movement on contested surfaces |
| `endtoend_probe.jl` | full GPU numerator + tables vs `correlate!` |
| `numerator_probe.jl` | FFTW vs MPSGraph, raw vs centered |
| `share_probe.jl` | per-point cost breakdown |
