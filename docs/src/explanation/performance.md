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

### On a production-sized grid the cold plan costs more than the correlation

Those rows are a 1024² pair reaching three transform sizes. A production grid reaches dozens, and the
cost scales with how many are new. Measured on the golden `S1A_..._20151120` case — an 11.5 M-point
grid, untiled at 10 threads, four fresh processes in sequence:

| run | wall | CPU |
|---|---:|---:|
| first | **28.8 s** | 71.9 s |
| second, third, fourth | 6.5, 6.5, 6.4 s | 48.1, 48.9, 48.3 s |

**22 seconds of planning against a 6.5 second correlation**, and the CPU column is what identifies it:
the extra is 24 CPU-seconds of real work, not a stretched wall clock. The planner, not the correlator.

**Nothing is missing from the package — this is a deployment property.** `__init__` calls
`AutoRIFT.load_wisdom!` so every process imports what is on disk, and
[`AutoRIFT.warm_plans!`](@ref) calls `AutoRIFT.save_wisdom!` so every pass persists what it measured.
The file is keyed by CPU model and FFTW version and lives in a Julia depot scratchspace — 84 KiB here.

What that means for a batch driver is the part worth stating plainly: **a process-per-pair driver in a
fresh container starts with an empty scratchspace, so every pair pays the full planning cost.** The
steady-state row above is unreachable without carrying the wisdom file across processes — bake it into
the image, or mount the scratch directory. On a fleet this is the difference between 6.5 s and 28.8 s
per pair, which no change inside the correlator approaches.

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

## Where the golden set's parallel headroom is

All 22 golden cases, untiled from the reference's captured inputs at 10 threads
(`tools/golden/profile_all.jl`). `occ` is CPU seconds over wall seconds, so it says how much of the
machine the run held; `serial` is the share of sampling intervals at or below two working threads.

| | wall | CPU | occupancy of 10 | on the table |
|---|---:|---:|---:|---:|
| whole set | **328.5 s** | 1913.2 s | **5.82** | 137.2 s |

**The headroom is inside the threaded correlation, not beside it.** Ranked by the time spent at or below
two working threads, the top phases are `_correlate_surface!`, `_track_chunk!`, `_track_claimed!` and
`_track_threaded!` — together over half of it — and the hotspots under them are `peak_index`,
`fft_execute!` and `ifft_execute!`. That is the correlation itself running with the machine idle beside
it, which happens where a pass has too few points to fill ten threads: the coarse levels, whose grids are
decimated by 2, 4 and 8. It is an Amdahl floor rather than a load-balance fault, which is consistent with
ordering chunks longest-first buying nothing — see below.

The stages that run *between* passes — the hole fill, the open-hole scan, the validity intersection, the
grid's sub-pixel phase, the pad — are now a small share, having been threaded one at a time. Reading the
ranking as a priority list is the trap the next section is about.

**One entry in that ranking was self-inflicted, which is the cheapest kind to find.** A reflective
`sizeof`-over-`fieldnames` closure reading the workspace pool's byte bound held `WORKSPACE_LOCK` on every
workspace return and accounted for **9.8%** of the set's serial time. Recording the byte count on the
workspace instead took the set from 363.6 s and 2023.2 CPU seconds to the row above — **−9.7% wall and
−5.4% CPU** — and removed the entry from the table. The CPU saving is larger than the serial share
because the closure also ran during parallel intervals, which that ranking does not count.

## Threading the whole-grid stages: faster in isolation, invisible end to end

The stages that run between correlation passes — the grid's sub-pixel phase, the hole fill, the open-hole
scan, the validity intersection, the pad — are now threaded, and each is faster for it. On a 3400² grid
and a 1008² level field at 10 threads:

| stage | serial | threaded |
|---|---:|---:|
| `_grid_phase` | 23.8 ms | **3.7 ms** |
| `_fill_holes!` | 52.9 ms | **32.9 ms** |

**It does not move a run's wall clock, and that is worth recording so the measurement is not repeated.**
Warmed paired runs, alternating arms so neither carries the other's planning cost:

| case | with threading | without | measured points |
|---|---:|---:|---:|
| `LC08_L1TP_009011`, 5.5 M grid | 20.2 s, occupancy 9.14 | 19.7 s, occupancy 9.15 | 1,708,352 both |
| NISAR L2 GSLC, thinned | 23.0 s, occupancy 6.28 | 23.1 s, occupancy 6.23 | 239,574 both |

The reason is arithmetic rather than a failure: these stages run once per level against a correlation that
runs once per point, so 20 ms saved four times over is 1–2% of a twenty-second run — inside the spread.
**A stage can top the serial-interval ranking and still be a small share of the run**, because that
ranking measures only the intervals where the machine was otherwise idle. Read it as "what would have to
be parallel to fill those gaps", not as "where the time is".

The changes stand on their own terms — bit-identical, and strictly faster where they are the cost — but a
wall-clock claim is not among the things they buy.

## Rejected: ordering correlation chunks longest-first

A chunk's cost spans orders of magnitude on a Geogrid radius field, so handing the expensive ones out
first should tighten the makespan from `1 + max/total` toward `1 + 1/n`. Implemented, measured, removed.

`dev/GATES.md` already bounds the prize at **1.16× ideal**, and the measured gain on the paired runs above
was zero, against a real cost: scoring the chunks means one pass over every point's radii per pass, which
is **155 ms on an 11.5 M-point grid, or 1.24 s per run over eight passes**. A guaranteed 1.24 s for an
unmeasurable share of a 16% ceiling is the wrong trade. Dynamic claiming already absorbs most of the
imbalance, which is why the ceiling is 1.16× and not 4×.

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
