# Performance

What a correlation costs, and which of the several available kinds of parallelism to spend.

Every number here is measured on an Apple M2 Max, 8 threads, Julia 1.12.5, as a minimum over many
samples — a shared machine's single-shot medians swing by 3× under load. `benchmark/run.jl` reproduces
them.

## One pair

1024 × 1024, chip 32, radius 25, spacing 32 — a 900-point grid:

| | time | allocation |
|---|---:|---:|
| serial | 124.9 ms | — |
| 8 threads | **38.2 ms** | 32.5 MiB |

Inner loops allocate nothing per point; the 32.5 MiB is buffers sized once per pass. That is asserted
in the test suite rather than hoped for, because a per-point allocation is invisible in a single
correlation and decides the throughput of a million of them.

## Parallelism: pair-level beats intra-pair

Two ways to use eight cores. `threaded = true` splits one pair's grid points across threads; the
alternative gives each thread a whole pair and its own [`AutoRIFT.Cache`](@ref). 512 × 512, 16
consecutive pairs:

| regime | pairs/sec | |
|---|---:|---|
| serial | 44.7 | |
| intra-pair threading | 116.9 | 2.6× over serial |
| **pair-level, one cache per worker** | **299.4** | 6.7× over serial, 2.6× over intra-pair |

Pair-level wins by a factor of 2.6, which is why the cache lifecycle is public API rather than an
internal detail: `init`, `reinit!` and `autorift!` exist so a worker can hold its buffers and FFT
plans across pairs. [Correlating many pairs](@ref) is the how-to.

The reason is the working set. One pair's correlation is thousands of independent small problems, and
splitting one of them across threads shares a cache line budget that a whole pair per thread does
not. Prefer pair-level whenever there is more than one pair; reach for `threaded = true` when there
is genuinely only one.

## Cold start, and why it is a separate number

A driver that launches a process per image pair pays startup once per pair, so load and first-call
compilation can exceed the correlation. Fresh processes:

| | first call | wall |
|---|---:|---:|
| no wisdom, no precompile workload | 3.39 s | 4.28 s |
| persisted FFTW wisdom | 2.35 s | 3.72 s |
| precompile workload, wisdom cold | 1.13 s | 2.40 s |
| **both, steady state** | **0.13 s** | **1.36 s** |

**26× off the first call and 3.1× off the wall clock**, at the cost of `using AutoRIFT` rising from
0.87 to 1.00 s and 5.9 s of precompilation once per install. FFTW wisdom is persisted to a scratch
directory and shared across processes, which is what makes the steady-state row reachable at all;
[`AutoRIFT.warm_plans!`](@ref) fills it deliberately.

Load time is measured on every pull request, because a dependency that quietly pulls in a heavy
transitive tree is otherwise discovered in production rather than at review.

## Against the reference implementation

`autoRIFT` 2.1.2's `runAutorift()` from the conda-forge build, on the same bytes — 1024 × 1024, chip
32, radius 25, spacing 32, a 900-point grid:

| | time | measured |
|---|---:|---:|
| autoRIFT 2.1.2 (C++ core) | 456 ms | 900/900 |
| AutoRIFT.jl, serial | 148 ms | 900/900 |
| AutoRIFT.jl, 8 threads | **41 ms** | 900/900 |

**3.1× serial and 11.1× threaded**, with displacements agreeing exactly. The threaded comparison is
against what the reference actually does rather than against a parallel reference: its production
driver sets `mpflag = 0` unconditionally, so its own multi-threaded path is never used.

## A trimmed binary for small instances

97% of an ordinary AutoRIFT process's 408 MiB memory floor is the Julia runtime; the package is about
7 MiB of it. No optimization inside the package reaches the rest, so `app/` builds a statically
compiled executable instead:

| | trimmed binary | `julia -t1`, same work |
|---|---:|---:|
| peak RSS, 512² pair | **27.2 MiB** | 424.2 MiB |
| peak RSS, 2048² pair | **117.8 MiB** | 542.7 MiB |
| wall clock, 512², warm wisdom | **0.06 s** | 1.20 s |
| binary | 3.2 MiB | — |

Every displacement is bit-identical to the library at both sizes, verified by comparing raw output
planes rather than by a tolerance. The 20× wall-clock figure is startup rather than correlation, which
is exactly the cost a process-per-pair driver pays every time.

`correlation` is the one exception, and only across a wisdom boundary: two runs with *different* FFTW
wisdom differ by up to 3.0e-7 on about 20% of points, because the planner picks a different algorithm
for the same transform size and a different algorithm reassociates the same sum differently. Two
library runs disagree the same way. `dx` and `dy` are unaffected. Worth knowing before using a
byte-comparison as a regression check.

`app/README.md` has the build recipe. It needs `--experimental` on Julia 1.12, so it is a recipe
rather than a released artifact.

## What not to reach for

- **A GPU, on an otherwise free machine.** The device is 2.6–3.0× a single CPU core on the correlation
  pass and 0.35× eight threads on a whole pair. It is also not additive: handing one core's work to it
  buys 1.6%, and oversubscribing costs 22%. Its case is one core against it — the process-per-pair
  shape — and [Correlating on a GPU](@ref) is specific about the rest.
- **A coarser transform size.** Sizes are already quantized to what FFTW plans well; coarsening costs
  more than it saves. [Memory](@ref) has that measurement.
- **A larger block, for speed.** Block size trades read amplification against thread occupancy and
  peak memory, and the fastest choice is not the largest. [Correlating scenes larger than
  memory](@ref) sizes it.

`benchmark/README.md` carries the rest of the record, including changes that were measured and
rejected — a global integral image, SAD and SSD similarity measures, windowed-histogram outlier
filtering — each with the numbers that ruled it out. A performance claim in a source comment should
trace to one of those tables.
