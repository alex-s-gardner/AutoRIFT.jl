# Memory

Measured on an Apple M2 Max, Julia 1.12.5, against autoRIFT v2.1.2 under Python 3.11. Reproduce
with `julia --project=benchmark benchmark/memory.jl` and
`mamba run -n arift-ref python tools/python_ref/bench_reference.py`.

Runtime was M8's subject. This is the other axis, and it matters for a specific reason: AutoRIFT is
built for batch runs of tens of millions of image pairs, and a scheduler packing those onto small
instances is bounded by resident memory per worker, not by the speed of any one pair.

## The two questions, and why one number cannot answer both

Peak RSS and live heap disagree, and the disagreement is the useful part.

| Over 30 consecutive 512² pairs through one `Cache` | AutoRIFT | autoRIFT v2.1.2 |
|---|---:|---:|
| peak RSS growth | +38.7 MiB | — |
| live heap growth (`gc_live_bytes`, after full GC) | **+0.0 MiB** | — |
| current RSS after `gc.collect()` | — | **+5.7 MiB/pair, no plateau** |

Read `rss` alone and AutoRIFT looks like it accumulates the way the reference does. It does not:
the live heap is flat, so the 38.7 MiB is allocator slack the collector has not returned to the OS.
A long batch run does **not** need process recycling.

The reference does. 376.9 MiB after the first pair, 542.3 MiB after thirty, as *current* RSS after
an explicit collect — so it is retention, not slack. Extrapolated, a thousand-pair worker needs
several GiB. That is why `benchmark/memory.jl` reports both quantities: reporting only `rss` would
have implied the wrong conclusion about our own code, and only `live` would have hidden what an
instance's memory limit actually sees.

Note the crossover. Below roughly ten pairs Python uses *less* memory, because Julia's floor is
higher; above it, Python's growth dominates and never stops.

## Three ways to measure this wrong

Recorded because each failed in a different direction, and two produced numbers that looked like
bugs in the correlator rather than in the measurement.

- **`@allocated` counts cumulative churn, not residency.** It reported 140 MiB for a 2048² pair
  whose actual peak was 32 MiB — it sums every temporary ever created, including ones freed
  immediately. Useful for allocation pressure; meaningless for "will this fit".
- **`Sys.maxrss()` is a high-water mark.** Measuring several configurations in one process gives
  the first one's peak and then near-zero for every later one. That produced ratios of 0.05× and
  0.00×, which read as a broken code path.
- **Subtracting a settled baseline in-process fails for the same reason.** The mark is already set.

Hence the design in `benchmark/memory.jl`: one fresh subprocess per configuration, `maxrss` read
once at the end, and the baseline established by a matching process that allocates the inputs but
skips the correlation.

## The floor is the Julia runtime, not AutoRIFT

| Cumulative RSS after | MiB |
|---|---:|
| bare Julia runtime | 396.8 |
| `+ using FFTW` | 400.6 |
| `+ using AutoRIFT` | 408.0 |

**97% of the floor is the runtime.** AutoRIFT itself is ~7 MiB. There is no optimization inside
this package that reaches the other 397; `--compile=min` recovers 18 MiB, which is not the right
order of magnitude. That is what makes a trimmed binary the only real lever — see
[`app/`](../app/README.md), which runs the same correlation at **29.7 MiB peak against 424.2 MiB**.

## Per-pair peak

Scene size is the only knob that scales peak memory, and it does so roughly linearly in pixel count
— 4× the pixels for 4× the peak, twice over.

| Scene | serial | threaded |
|---|---:|---:|
| 512² | 8.2 MiB | 9.3 MiB |
| 1024² | 28.6 MiB | 32.0 MiB |
| 2048² | 113.3 MiB | 114.5 MiB |

## The non-knobs

Measured because they are what one would expect to expose, and they do not. Recorded so they are
not re-proposed — and, in the first case, so a false positive is not either.

**Thread count does not move peak memory.** Repeated measurement at 512² gives 7.4, 8.0, 8.9 MiB at
one thread and 7.8, 9.3, 6.2 MiB at eight — a 6.2–9.7 MiB scatter with no ordering by thread count.
An earlier reading of this suite appeared to show peak *falling* from 14.3 to 5.1 MiB with more
threads, and a plausible mechanism was available (more threads, earlier collection, less slack).
That reading was noise: three repetitions do not reproduce the trend or its direction. Workspaces
are pooled and few exist at once, so there is nothing here to scale. Threading is not a
memory-versus-speed tradeoff in either direction.

**`upsampling` has no trend** — 30.2 MiB at 8× against 27.4 MiB at 128× on 1024², despite the
refinement workspace itself growing from 0.2 MiB to 62.5 MiB. Pooling again: few exist
simultaneously. It costs time (31 ms → 64 ms), not memory. Tracked in the suite so the claim stays
true rather than being remembered.

Two of these are why a single subprocess measurement is not enough to assert a trend on. The suite
records one figure per configuration for regression detection; a *claim* about a knob needs
repetition, and the thread-count entry above is what happens without it.

**A `max_memory` budget was designed and rejected.** A configuration-derived estimate of peak came
out 1.6–4.1× off, and unsystematically — the gap *shrank* as the estimate grew, so it could not be
corrected with a constant factor. The cause is that peak tracks allocation *rate* and GC timing,
not the resident size of the buffers a configuration implies. A knob that mis-predicts by 4× in an
unpredictable direction is worse than no knob: a caller would size an instance from it and get
OOM-killed. **Tiling large scenes** is the honest version of this, since scene size is the one thing
that does scale, and it is deferred rather than dismissed.

## Practical guidance

- **Batch work: one pair per process or per worker, `threaded = false`.** Also 2.7× faster than
  intra-pair threading (`benchmark/suite/throughput.jl`), so this is not a tradeoff.
- **Reuse a `Cache` across pairs** via `init`/`reinit!`/`autorift!`. The live heap is flat, so this
  is bounded regardless of batch length.
- **No process recycling needed** — the measurement above is what establishes that.
- **Small instances: use the trimmed binary.** 29.7 MiB against 424.2 MiB is the difference between
  3.6% and 43% of a `t3.micro`.
