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
| peak RSS growth | +41.2 MiB | — |
| live heap growth (`gc_live_bytes`, after full GC) | **+0.0 MiB** | — |
| current RSS after `gc.collect()` | — | **+5.7 MiB/pair, no plateau** |

Read `rss` alone and AutoRIFT looks like it accumulates the way the reference does. It does not:
the live heap is flat, so the 41.2 MiB is allocator slack the collector has not returned to the OS.
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
[`app/`](../app/README.md), which runs the same correlation at **27.2 MiB peak against 424.2 MiB**.

## Per-pair peak

Scene size is the only knob that scales peak memory, and it does so roughly linearly in pixel count
— 4× the pixels for 4× the peak, twice over.

| Scene | serial | threaded |
|---|---:|---:|
| 512² | 6.6 MiB | 7.9 MiB |
| 1024² | 18.8 MiB | 9.9 MiB |
| 2048² | 42.4 MiB | 26.5 MiB |

The threaded column carries the scatter described under "The non-knobs" below and should not be
read as a trend against the serial one.

### The images are correlated in the caller's own element type

There is no conversion step: an `Int16` sensor product is correlated as `Int16`, and only the
chip is widened, to `Float32`, because it is stored mean-removed. An earlier 8-bit rescaling stage
mapped `mean ± 3σ` of each image onto `0..255` first, and removing it **costs 8.1 MiB of peak at
2048²** — 34.2 against the 42.4 above, measured serial.

That is the honest price of three things. The rescale claimed to buy exact integer accumulation,
which it never did — the numerator accumulates in `Float64` either way — and the repo's own
measurement put 8-bit and `Float32` chips at 27.8 against 27.9 µs on 200², i.e. equal. Accuracy is
marginally *better* without it, since quantizing ahead of a `Float64` accumulation can only lose
information: on a 1024² pair against a known fractional shift, `dx` RMS error 0.02463 against
0.02482. And its scale came from a whole-image statistic, which is the kind of quantity that makes
a block of a scene disagree with the scene.

## Allocation per pair, and where it went

`preprocess` was 92% of a pair's allocations — 31.5 of 34.0 MiB on 1024² — and almost all of that was
`windowmean` scratch. The column pass produces one `Float64` sum per pixel and the row pass consumes
one row of them at a time, so a full-image scratch was written entirely before any of it was read:
every value evicted from cache before use. The NaN-aware path carried two such arrays, `Float64` sums
and `Int32` counts, 12 MiB on 1024² to produce 4 MiB of output.

Processing a band of 16 rows at a time keeps the scratch in L2. It is **faster and smaller at once**,
which is unusual enough to be worth stating plainly — the locality is worth more than the cost of
restarting each band's running sum:

| `windowmean` on 1024² | before | after |
|---|---:|---:|
| dense (gap-free) | 5.6 ms / 8.01 MiB | **3.3 ms / 0.13 MiB** |
| masked (any NaN) | 9.8 ms / 12.01 MiB | **3.7 ms / 0.11 MiB** |

Per whole pair, allocation roughly halves:

| Pair | before | after |
|---|---:|---:|
| 512² dense | 8.08 MiB | **4.20 MiB** |
| 1024² dense | 32.45 MiB | **16.70 MiB** |
| 2048² dense | 129.84 MiB | **66.34 MiB** |
| 1024² masked | 72.49 MiB | **48.87 MiB** |
| 2048² masked | 289.92 MiB | **194.67 MiB** |

Bit-identical throughout, which is the property that makes banding safe: each band reseeds its
running sum from exactly the window its first row sees, so no value is the result of a longer or
differently-ordered accumulation. Verified exactly rather than to a tolerance, end to end, across
dense, masked and Wallis paths at 512² and 1024².

The masked path is not a corner case worth less attention than the dense one — a single NaN anywhere
sends the whole array down it, and reprojection to a common grid routinely leaves a no-data border.

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

**`upsampling` has no trend** — measured 30.2 MiB at 8× against 27.4 MiB at 128× on 1024² before the
banding change, and the *absence* of a trend is what matters, not the absolute figures, despite the
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

## Block size: peak scales with block area, not with block count

`process_block_size` is the one knob a caller has over peak memory, and the figure that decides
whether it is usable is peak against runtime at each size. Measured on the full Landsat 8/9 overlap —
17121×16961 px, a 2127×2107 grid, 922,784 measured points — at 10 worker threads, from a
memory-mapped input. Reproduce with `tools/ab/mem_blocks.jl`.

| block | blocks | peak MiB | vs untiled | runtime | read amplification |
|---|---:|---:|---:|---:|---:|
| untiled | 1 | 4958 | 1.00× | **20.5 s** | 1.00× |
| 8192 px | 9 | 14856 | 3.00× | 42.0 s | 1.03× |
| 4096 px | 25 | 6995 | 1.41× | 30.3 s | 1.06× |
| 2048 px | 81 | 3116 | 0.63× | 24.4 s | 1.13× |
| **1024 px** | 289 | **2140** | **0.43×** | **22.6 s** | 1.26× |
| 512 px | 1089 | 2248 | 0.45× | 22.3 s | 1.55× |
| 256 px | 4160 | 2042 | 0.41× | 24.0 s | 2.21× |

`dx`/`dy` are bit-identical at every size and all 922,784 points are measured at every size, so
nothing below is a quality trade.

**1024 px is the default.** It cuts peak 2.3× for a 10% runtime cost, and the curve is flat from
there to 256 px — 2140, 2248, 2042 MiB across a 14× range of block *counts*. That flatness is the
finding: peak is set by a block's **area**, not by how many blocks there are. Nine arrays sized to the
largest read window are held per task (`AutoRIFT.BlockBuffers`), and the task count is capped at
`min(nblocks, nthreads)`, so the footprint is area × threads however finely the scene is cut.

**Runtime is set by the halo, and that is what bounds how small a block can usefully be.** The halo
is computed from the parameters and the grid *before* any block size is applied, and is the same for
every block — `chip_size_max/2 + radius + |prior| + 2 + filter_reach + level_centre_offset`, which is
69 px here. So it is a fixed-width skirt on a shrinking block, and the imagery a run reads grows as
`((block + 2·halo) / block)²`: 1.26× at 1024 px, 1.55× at 512, 2.21× at 256, 3.82× at 128. Below
1024 px the memory curve has already flattened while that redundancy keeps climbing, so smaller
blocks buy nothing and cost reading. **If a configuration has a wide halo, the block size has to
increase in proportion** — a block only a few halos across is mostly overlap.

Pushed far enough the redundancy stops being merely wasteful. At 128 px (15,624 blocks, 3.82× read
amplification) the run becomes allocation-bound rather than compute-bound: sampled stacks put 6 of 10
workers in `__psynch_mutexwait` beneath `jl_safepoint_start_gc`, all queued on one mutex, and the run
had not finished in 20 minutes against 22.6 s at 1024 px. Every block read allocates a block-sized
temporary (`_read_block!` indexes rather than `copyto!`-ing a view, deliberately — a lazy input needs
one read per window, not one per pixel), so read amplification is also allocation rate.

**8192 px is a misconfiguration, not a baseline.** With 9 blocks on 10 threads every block is in
flight at once, so the run holds nine 8331² working sets — 3× the untiled peak — and it is also one
core short, since `min(nblocks, nthreads)` caps the tasks at 9. A block size chosen so that blocks are
fewer than threads inverts what blocking is for. Keep the block count comfortably above the thread
count.

**The untiled row is a different parallel decomposition, not just a different block size.** It has one
block, so it uses its threads through the intra-pass path while every blocked run uses them per block
(`threaded = false` inside each). That is why it is the fastest row here and still not the one to
choose: 4958 MiB against 2140 is the difference between what fits on an instance and what does not.

**These numbers are for this configuration's 69 px halo.** A wide-halo configuration behaves
differently in kind, not only in degree. On the whole NISAR L1 grid `halo(grid, p, size)` is
**2736×1500 px** — about 7 by 6.7 km at that granule's 2.55 m ground-range and 4.44 m along-track
spacing — because a Geogrid search-radius field is extremely skewed: median 26 px against a maximum
of 1905, a 73× spread, and the halo takes the maximum. `tools/golden/GATES.md` records a per-block
halo that was implemented and reverted at a measured 1.03–1.11× gain, since `chip_size_max/2` alone
floors it at 561 px whatever the block.

A 2736 px halo still permits blocks on a 57760×50511 scene, and it now produces them. Two properties
of a rotated grid had to be handled first, and both used to fail silently rather than loudly.

**A grid's coordinates need not be separable.** [`AutoRIFT.block_layout`](@ref) used to derive its
block boundaries from `grid.y[:, 1]` and `grid.x[1, :]`, assuming a gridded `PointSet` repeats each
coordinate down every row and across every column. A NISAR geogrid is a **rotated radar footprint**
sampled onto a map grid, so `x` varies by 50502 px down a single column and `y` by 43164 px across a
single row; only 43% of points carry real coordinates, and row 1 and column 1 hold *one* valid point
each. Walking them spanned 216 px instead of the scene, so every requested block size up to 16384 px
returned **one block** — an untiled run wearing a block size.

The grid *is* the index-to-pixel mapping, so the block shape now comes from it: four rates, how far `x`
and `y` each move per row and per column of the index space. A block of `a` rows by `b` columns spans
about `a·∂x/∂i + b·∂x/∂j` pixels of `x`, and both axes must fit their budget. **Both index directions
charge both axes**, which a separable calculation gets wrong — sizing rows from the `y` budget and
columns from the `x` budget alone gives an `x` span of 11187 px at an 8192 px request on this grid, a
37% overshoot of what the caller asked for. On an axis-aligned grid two of the four rates are zero, the
constraints decouple, and this reduces to the separable answer, so a Landsat layout is unchanged — a
full-width band included.

**A block's read window must span only the points it will search.** The window came from
[`AutoRIFT._pixel_span`](@ref) over every point in the block, and on a rotated grid the points outside
the footprint carry a *fill* coordinate — zero here, not `NaN`, so a finiteness test does not find
them. A block straddling the footprint edge therefore spanned from 0 to the real coordinates and read
`1:57760`, the whole scene. Measured at an 8192 px block: 28 of 209 blocks each read half the scene or
more, 99× the scene in total. [`AutoRIFT._searchable_span`](@ref) reduces over searchable points only,
which is sound because `_run_one_block!` returns before any I/O for a block with nothing to search.

**And an index rate has to be measured over the points a block must cover.** The rates above came from
the median of each axis's *nonzero* first differences, which cannot see a genuinely separable axis: on
the NISAR L2 grid `x` really is constant down a column, so the only nonzero steps are the two crossing
the fill boundary, giving `∂x/∂i = 87666 px` from a sample of one and a layout of 5.2 million blocks.
The median of *every* step fails the other way — both NISAR grids are ~65% fill, so all four rates come
out zero and a zero rate is an infinite block. Reducing over pairs where both points are searchable
gives the true rates on both: 33/34/19/19 px on the rotated L1 grid, 0/48/24/0 on the separable L2 one.

With all three fixed, both NISAR granules block. Measured whole-grid at `-t 10` on a 96 GiB machine:

| case | block | blocks | runtime | peak | vs untiled | read amp |
|---|---|---:|---:|---:|---:|---:|
| L1 RSLC, 57760×50511 | untiled | 1 | 10.9 min | **55.2 GiB** | 1.00× | 1.00× |
| | 16384 px | 100 | — | 68.6 GiB | 1.24× | 2.89× |
| | 8192 px | 380 | 41.1 min† | 34.8 GiB | 0.63× | 4.51× |
| | 4096 px | 1482 | 45.0 min† | **31.5 GiB** | 0.57× | 8.77× |
| L2 GSLC, 54885×110085 | untiled | 1 | 6.2 min | **85.0 GiB** | 1.00× | 1.00× |
| | 8192 px | 98 | 11.2 min | 47.8 GiB | 0.56× | 0.86× |
| | 6144 px | 162 | 6.2 min | 45.7 GiB | 0.54× | 1.00× |
| | 4096 px | 378 | 7.1 min | 40.1 GiB | 0.47× | 1.33× |
| | 3072 px | 648 | 5.8 min | 31.2 GiB | 0.37× | 1.69× |
| | 2304 px | 1152 | 5.5 min | 30.0 GiB | 0.35× | 2.26× |
| | 2304×1152 px | 2304 | **5.2 min** | **28.8 GiB** | **0.34×** | 3.30× |

† shared the machine with another large job; peak RSS is insensitive to that where wall clock is not.

**L2 is the case that makes blocking a production requirement, and the right block size is the
smallest shape the halo permits.** 2304×1152 px runs the granule in **28.8 GiB and 5.2 minutes** against
an untiled 85.0 GiB and 6.2 — about a third of the peak *and* faster, at 99.98% of the untiled point
count. Every blocked row measures the same 1,781,377 points, so the sizes differ in cost alone. On a 96
GiB machine this is the difference between a memory-optimized instance and a general-purpose one.

**Both axes want sizing separately, because the halo is not square.** At 2216×1103 px it is almost
exactly 2:1, so a *square* block clears the X halo and then over-provisions Y twofold. `2304×1152`
follows the halo's own aspect ratio and is the best row measured; the square floor is 2304 (2048 is
rejected against the 2216 px X halo) while the Y floor is half that.

**Runtime is monotonic in block *count*, not in block size, and the mechanism is thread occupancy.**
Measured as `cpu_seconds / wall_seconds` (`tools/golden/profile_nisar.jl`):

| block | blocks | blocks/thread | occupancy of 10 |
|---|---:|---:|---:|
| 8192 px | 98 | 9.8 | **2.96** |
| 6144 px | 162 | 16.2 | 5.50 |
| 4096 px | 378 | 37.8 | 5.17 |
| 3072 px | 648 | 64.8 | 6.77 |
| 2304 px | 1152 | 115.2 | 7.87 |
| 2304×1152 px | 2304 | 230.4 | **9.03** |

Per-block cost spans orders of magnitude — a block whose points a finer level resolved returns before any
I/O — so a pool with few blocks per thread waits on a handful of expensive ones while the rest idle. **Ten
blocks per thread is not enough; occupancy is still climbing at 230.** Many small blocks is the
configuration that keeps a wide machine fed, and there is no measured turning point on this granule.

Read amplification rises 0.86× → 3.30× across those rows and does not drive the ranking — the fastest row
has the *highest* amplification. Allocation follows it (225 GiB untiled to 1727 GiB, since every block
read allocates a block-sized temporary) and GC still absorbs ≤1.7% of wall clock, so on this granule
neither reading nor allocation is the constraint that block size trades against. Idle threads are.

**The L2 rows above supersede an earlier pass that measured untiled at 80.9 GiB and 12.1 min, and the
runtime half of that is an instrument artifact.** `src/` is unchanged across the interval and both
passes count the same points, so nothing about the package moved. The earlier harness had no warmup and
measured the untiled configuration first, so that row carries the process's JIT compilation; the
blocked rows it measured afterwards did not, which is why the gap appears on untiled alone and why the
recorded ordering made blocking look free. `profile_nisar.jl` correlates a small patch of the grid
before any row is recorded.

Profiler overhead is *not* the explanation, and this is worth stating because it was the first guess.
Every row here is measured twice — once with the profiler off and once with it on at 2 ms — and the
profiled run costs **2–4%** (1.02×, 1.03×, 1.03×, 1.04×), nowhere near the factor of two the
discrepancy would need. Sampling at this rate is cheap enough to ignore; compiling is not.

The 4 GiB peak difference is a genuine measurement spread. Peak is sampled from a shared process whose
floor depends on what the previous configuration left behind, which is why the floor is reported beside
every peak and why 3072 px was re-measured alone — it reproduced at 347.6 s against 340.9 in the sweep.

**A block can also be too large, and the crossover is arithmetic rather than empirical.**
`AutoRIFT.BlockBuffers` holds nine block-sized arrays totalling **18 bytes per pixel** for a `UInt8`
pair — two `UInt8` planes, three `Float32` and four `Bool` — and each of `min(nblocks, nthreads)` tasks
gets its own set. At 16384 px on L1 the read window is 12232×22222, which is 4.56 GiB per set and
**45.6 GiB across ten tasks** before any imagery or workspace; measured peak was 68.6 GiB against an
untiled 55.2. Prediction and measurement agree to 1%, so
`18 bytes × (block + 2·halo)² × min(nblocks, nthreads)` is worth computing before choosing a size.

The 18 bytes are the total across all nine arrays, not the size of each: `18 bytes` per array would
predict 410 GiB for that L1 window and reject every block size this granule can actually run.

**Read amplification below 1.0 is possible**, and L2 shows it at 0.86×: its halo is small relative to
the block and 64% of its grid is fill, so those blocks have no searchable point and read nothing at all.

One failure is open. The L2 8192 px run deadlocked on the first attempt at 44.4 GiB — every allocating
thread queued at `jl_safepoint_start_gc` behind a collection that never started — and completed cleanly
on a second attempt with the machine idle. The difference was a concurrent job holding tens of GiB, so
it needs memory pressure from outside the process. `tools/golden/GATES.md` holds the thread trace.

The amplification is high because the halo is, not because the layout is loose: at a 2684×1448 px halo
even a 16384 px block pays 2.6×. That is the arithmetic in "Runtime is set by the halo" applied to a
wide-halo configuration, and it sets the *floor* on a usable block size — a block below the halo is
rejected outright.

It does not follow that large blocks are the efficient ones, which is what the amplification argument
alone suggests. On L2 the highest-amplification row measured is also the fastest and the cheapest:
1.69× at 3072 px runs in 5.8 min at 31.2 GiB, against 0.86× at 8192 px in 11.2 min at 47.8. Redundant
reading is cheap next to leaving threads idle, so the halo tells you the smallest block you may use and
the thread count tells you which of the permitted sizes to pick.

## Practical guidance

- **Batch work: one pair per process or per worker, `threaded = false`.** Also 2.7× faster than
  intra-pair threading (`benchmark/suite/throughput.jl`), so this is not a tradeoff.
- **`process_block_size = (1024, 1024)` when peak memory matters**, and go as small as the halo allows.
  A block smaller than its own halo is rejected, which is the only floor; within what a granule permits,
  the smallest shape is both the lowest peak and the fastest run.
- **Size the two axes separately against `halo(grid, p, size)`.** A 2:1 halo wants a 2:1 block; a square
  one wastes half of the short axis. On NISAR L2 that is `(2304, 1152)`, which beats `(3072, 3072)` on
  peak, runtime and occupancy at once.
- **Aim for hundreds of blocks per thread, not tens.** Blocks are the unit of threaded work and their
  cost varies by orders of magnitude, so a small pool waits on its slowest members: 9.8 blocks/thread
  runs at 3.0 threads of ten, 64.8 at 6.8 and 230.4 at 9.0. Occupancy was still improving at the smallest
  size measured.
- **Reuse a `Cache` across pairs** via `init`/`reinit!`/`autorift!`. The live heap is flat, so this
  is bounded regardless of batch length.
- **No process recycling needed** — the measurement above is what establishes that.
- **Small instances: use the trimmed binary.** 27.2 MiB against 424.2 MiB is the difference between
  3.1% and 43% of a `t3.micro`.
