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
