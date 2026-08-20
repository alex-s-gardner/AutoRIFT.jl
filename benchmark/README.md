# Benchmarks

Performance is a tracked deliverable. autoRIFT runs on tens of millions of image
pairs, so a 10% regression in the correlation kernel is a material cost, and
"faster than OpenCV" should be a number rather than a claim.

```bash
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'

# `-t auto` matters: without it the threaded benchmarks silently measure the serial path.
julia --project=benchmark -t auto benchmark/run.jl              # full run
julia --project=benchmark benchmark/run.jl --quick              # reduced sampling
julia --project=benchmark benchmark/run.jl --tag my-experiment

julia --project=benchmark benchmark/compare.jl \
    benchmark/results/baseline.json \
    benchmark/results/history/<tag>.json
```

`compare.jl` exits non-zero on a regression, so it works as a CI gate. It prints
a markdown table of baseline / candidate / ratio, plus a second table of speedup
against the Python reference for every benchmark that has one.

## What is committed, and what is not

| Path | Committed | Why |
|---|---|---|
| `results/baseline.json` | yes | The reference to beat. Updated deliberately, by a human, when an improvement lands. |
| `results/python.json` | yes | OpenCV timings, from `tools/python_ref/bench_python.py`. |
| `results/history/*.json` | no | Absolute times are machine-specific and would be meaningless in another checkout. |

## Gating on ratios, not absolute times

Hosted CI runners vary by more than any regression worth catching, so an absolute
threshold would either fire constantly or never. The CI job measures the merge
base and the pull request **in the same job on the same runner** and compares the
two, which is the only way to get a usable ratio out of shared hardware.

The threshold is 1.10x on the minimum time. Minimum rather than mean because
interference can only ever make a sample slower, so it is the least noisy estimate
of what the code actually costs.

Allocations are treated differently: benchmarks matching `ZEROALLOC_PATTERNS` in
`compare.jl` fail on *any* allocation, not on an increase. Allocating once per grid
point would be invisible in a microbenchmark and ruinous across millions of image
pairs, so it is a correctness property rather than a performance one — and it is
asserted in the test suite as well.

## Structure

Three tiers by timescale. Micro benchmarks use Chairmarks, which collects far
faster than BenchmarkTools and has better statistics at nanosecond scale, keeping
the whole suite runnable in minutes; meso and macro use BenchmarkTools.

```
suite/points.jl       point-set construction and per-point geometry
suite/correlate.jl    the correlation surface, peak location, subpixel refinement
suite/window.jl       sliding-window reductions
suite/preprocess.jl   pre-correlation filters, per Mpixel
suite/multichip.jl    one chip-size level, and per-step breakdown
suite/endtoend.jl     whole scenes at 256, 1024, 4096, plus cold start
suite/throughput.jl   image pairs per second, intra-pair vs pair-level threading
```

Groups are populated as their milestones land. An empty group is deliberate: it
keeps the shape of the suite visible and makes a missing benchmark obvious.

Benchmark names must match the ones `tools/python_ref/bench_python.py` emits, or
the Python comparison silently finds no overlap.

## M8 results

Every number below is measured on this machine (Apple M2 Max, 8 threads, Julia 1.12.5)
rather than estimated, and each was taken as a minimum over many samples — the machine is
shared, and single-shot medians at this scale swing by 3x under load.

### Against the reference

`tools/python_ref/bench_reference.py` runs the real `autoRIFT.runAutorift()` from the
conda-forge `autorift=2.1.2` build. 1024x1024, chip 32, radius 25, spacing 32, a 30x30 =
900-point grid, minimum of three runs:

| | time | measured | displacement |
|---|---:|---:|---|
| reference autoRIFT 2.1.2 (C++ core) | 456 ms | 900/900 | `dx=+6.00 dy=+4.00` |
| AutoRIFT.jl, single-threaded | 148 ms | 900/900 | `vx=+6.00 vy=+4.00` |
| AutoRIFT.jl, 8 threads | 41 ms | 900/900 | — |

**3.1x single-threaded, 11.1x threaded.** The threaded figure compares against what the
reference actually does, not against a parallel reference: the production driver sets
`mpflag = 0` unconditionally, so its own multi-threaded path is unused.

The displacements agree exactly. The reference reports cartesian motion (`Dy = -Dy` at
`autoRIFT.py:1143`), which is what the `Raster` path returns; the array path returns the
raw secondary-to-reference offset, `dx=-6.00`.

### Throughput

512x512, 16 consecutive pairs, 8 threads:

| regime | pairs/sec | |
|---|---:|---|
| serial | 44.7 | |
| intra-pair threading | 116.9 | 2.6x over serial |
| **pair-level, one cache per worker** | **299.4** | 6.7x over serial, 2.6x over intra-pair |

Pair-level wins by enough to be a design conclusion rather than a tuning note, which is why
the `Cache` lifecycle is public API.

### Cold start

Fresh processes. Both halves matter separately: a driver launching a process per pair pays
all of it.

| | first call | wall |
|---|---:|---:|
| before M8 | 3.39 s | 4.28 s |
| + persisted FFTW wisdom | 2.35 s | 3.72 s |
| + precompile workload, wisdom cold | 1.13 s | 2.40 s |
| **+ both, steady state** | **0.13 s** | **1.36 s** |

**26x off the first call, 3.1x off the wall clock.** The cost is `using AutoRIFT` rising
from 0.87 to 1.00 s, plus 5.9 s of precompilation once per install.

### Steady state, 1024x1024

| | before M8 | after |
|---|---:|---:|
| serial | 158.3 ms | 124.9 ms |
| threaded | 46.3 ms | 38.2 ms |
| threaded allocation | 56.3 MiB | 32.5 MiB |
| threaded GC share | 21.8% | 12.0% |

### Measured and rejected

Recorded because knowing what does not work is worth as much as knowing what does; each is
documented at the relevant source file with its numbers.

| change | result | why |
|---|---|---|
| Global integral images | 5.42 ms vs 5.28 ms | Slower. A global table costs one pass per level whether that level searches 900 points or none, and after the finest level resolves what it can the rest search almost nothing. `src/integral.jl` |
| One oversized workspace shared across levels | 4.5e-8 from exact | Forces a 192-point FFT where 84 would do: 5x the arithmetic and different rounding. `src/correlate.jl` |
| Interior/border split on the *horizontal* pyrup pass | 0.29x | 3.4x slower. It would write two columns `2sh` floats apart, destroying the locality the vertical split gains. `src/peak.jl` |
| OpenCV's template-DFT hoisting, tiling, hysteresis clamp | n/a | Structurally inapplicable: every point has its own chip *and* window; transforms are already below the tiling threshold; the clamp never fires (worst excursion 9.0e-5 against a 12.5% band). `src/correlate.jl` |
| VQ-NNF windowed histograms (Gupta & Sintorn 2024) | ≤1% ceiling | The outlier filter runs on the 27x27 grid, not the image: 975 us of a 99 ms pass. `src/window.jl` |
| SSD / SAD similarity measures | SAD 8.7–15.7x slower | SAD cannot use the FFT; the normalisation they skip is only 15% of the cost; and both respond to brightness differences ZNCC removes. `src/types.jl` |
