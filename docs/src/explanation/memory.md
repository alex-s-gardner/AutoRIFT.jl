# Memory

Peak memory per configuration, what each cache holds resident, and which settings reduce it. AutoRIFT
runs on batches of tens of millions of image pairs, and a scheduler packing them onto small instances is
bounded by resident memory per worker.

Measured on an Apple M2 Max, on Julia 1.12.5. The reference is autoRIFT v2.1.2. No row has been re-timed
on Julia 1.13. Reproduce the pair-level numbers with `julia --project=benchmark benchmark/memory.jl` and
`mamba run -n arift-ref python tools/python_ref/bench_reference.py`.

## Resident memory and live heap

Peak resident set size (RSS) and live heap disagree, so both are reported.

| Over 30 consecutive 512² pairs through one `Cache` | AutoRIFT | autoRIFT v2.1.2 |
|---|---:|---:|
| peak RSS growth | +41.2 MiB | — |
| live heap growth (after full GC) | **+0.0 MiB** | — |
| current RSS after `gc.collect()` | — | **+5.7 MiB/pair, no plateau** |

The AutoRIFT live heap is flat, so the 41.2 MiB RSS growth is allocator slack that the collector has not
returned to the OS. A long batch run does not need process recycling.

The reference retains memory: 376.9 MiB after the first pair and 542.3 MiB after thirty, measured as
current RSS after an explicit collect. Below about ten pairs Python uses less memory because Julia's floor
is higher; above that, Python's growth continues without bound.

Peak RSS is a high-water mark for the process. A benchmark that measures several configurations in one
process reports the first one's peak for all of them, so `benchmark/memory.jl` runs one fresh subprocess
per configuration. `@allocated` counts cumulative allocation, not residency: it reported 140 MiB for a
2048² pair whose peak was 32 MiB.

## The floor is the Julia runtime

| Cumulative RSS after | MiB |
|---|---:|
| bare Julia runtime | 396.8 |
| `+ using FFTW` | 400.6 |
| `+ using AutoRIFT` | 408.0 |

97% of the floor is the runtime, and AutoRIFT itself is about 7 MiB. No change inside the package reaches
the rest, and `--compile=min` recovers 18 MiB. A trimmed executable is the only way below it.
[`app/`](https://github.com/alex-s-gardner/AutoRIFT.jl/tree/main/app) runs the same correlation at
27.2 MiB peak against 424.2 MiB.

## Per-pair peak

Peak grows roughly linearly with pixel count for a single pair:

| Scene | serial | threaded |
|---|---:|---:|
| 512² | 6.6 MiB | 7.9 MiB |
| 1024² | 18.8 MiB | 9.9 MiB |
| 2048² | 42.4 MiB | 26.5 MiB |

The threaded column varies by several MiB between repetitions (see "Settings that do not change peak"
below) and is not a trend against the serial one.

Images are correlated in the caller's element type. An `Int16` product stays `Int16`, and only the chip
is widened to `Float32` because it is stored mean-removed. Rescaling to `UInt8` first with
`AutoRIFT.bytescale` would lower the 2048² serial peak to 34.2 MiB. It does not change speed (27.8
against 27.9 µs per chip on 200²) and it does not improve accuracy: on a 1024² pair with a known
fractional shift, the `dx` RMS error is 0.02463 without it and 0.02482 with it.

### Allocation per pair

The sliding-window mean used by the filters (`windowmean`) works in bands of 16 rows so its scratch stays
in cache, which is both faster and smaller than a full-image scratch (1024²: 3.3 ms and 0.13 MiB dense,
3.7 ms and 0.11 MiB masked). Allocation per whole pair:

| Pair | allocated |
|---|---:|
| 512² dense | 4.20 MiB |
| 1024² dense | 16.70 MiB |
| 2048² dense | 66.34 MiB |
| 1024² masked | 48.87 MiB |
| 2048² masked | 194.67 MiB |

A single `NaN` anywhere in an image sends the whole array down the masked path, and reprojection to a
common grid routinely leaves a no-data border. Expect the masked rows, not the dense ones, for reprojected
data.

## Settings that do not change peak

**Thread count.** Repeated measurement at 512² gives 7.4, 8.0 and 8.9 MiB at one thread and 7.8, 9.3 and
6.2 MiB at eight. The 6.2 to 9.7 MiB scatter has no order by thread count. Workspaces are pooled and few
exist at once, so threading does not trade memory against speed in either direction.

**`upsampling`.** Peak is 30.2 MiB at 8× and 27.4 MiB at 128× on 1024², although the refinement workspace
grows from 0.2 to 62.5 MiB. Higher `upsampling` costs time (31 ms to 64 ms), not memory.

**A `max_memory` budget does not exist.** An estimate of peak derived from the configuration was off by
1.6 to 4.1× in no consistent direction, because peak follows allocation rate and GC timing rather than
buffer sizes. Scene size is the setting that does scale; for large scenes use `process_block_size`
below.

## Block size

`process_block_size` is the main setting that bounds peak memory. Nine arrays sized to the largest read
window are held per task (`AutoRIFT.BlockBuffers`), and the task count is `min(nblocks, nthreads)`. Peak
is therefore set by block area times threads, not by the number of blocks.

For an array input the default is `nothing` (one block). For file-backed Rasters input the default comes
from [`AutoRIFT.block_size_for`](@ref): the halo, floored at 1024 px, rounded up to the file's chunking
and clamped to the scene.

The sweep below is the full Landsat 8/9 overlap (17121×16961 px, a 2127×2107 grid, 922,784 measured
points) at 10 threads from a memory-mapped input. The configuration is `chip_size = 16`,
`chip_size_max = 64` (3 levels), `grid_spacing = 8` and `search_radius = 20`, which gives a halo of
69×69 px. Reproduce with `tools/ab/mem_blocks.jl`. `dx` and `dy` are bit-identical and all 922,784
points are measured at every size.

| block | blocks | peak MiB | vs untiled | runtime | read amplification |
|---|---:|---:|---:|---:|---:|
| untiled | 1 | 4958 | 1.00× | 20.5 s | 1.00× |
| 8192 px | 9 | 14856 | 3.00× | 42.0 s | 1.03× |
| 4096 px | 25 | 6995 | 1.41× | 30.3 s | 1.06× |
| 2048 px | 81 | 3116 | 0.63× | 24.4 s | 1.13× |
| **1024 px** | 289 | **2140** | **0.43×** | **22.6 s** | 1.26× |
| 512 px | 1089 | 2248 | 0.45× | 22.3 s | 1.55× |
| 256 px | 4160 | 2042 | 0.41× | 24.0 s | 2.21× |

1024 px cuts peak 2.3× for a 10% longer runtime, and peak is flat from there down to 256 px.

**The halo sets the smallest useful block.** Each block reads its extent plus a fixed-width halo, computed
from the parameters before any block size is applied:
`chip_size_max/2 + radius + |prior| + 2 + filter_reach + level_centre_offset`, which is 69 px here. See
[`AutoRIFT.halo`](@ref). The imagery a run reads grows as `((block + 2·halo) / block)²`: 1.26× at 1024 px,
1.55× at 512, 2.21× at 256 and 3.82× at 128. A block smaller than its halo is rejected. A block only a few
halos across is mostly overlap, so a wide halo needs a proportionally larger block.

Read amplification is also allocation rate, because every block read allocates a block-sized temporary.
At 128 px (15,624 blocks) the run became allocation-bound, with 6 of 10 workers waiting on one GC mutex,
and had not finished in 20 minutes against 22.6 s at 1024 px.

**Fewer blocks than threads is the one setting that is bad on every axis.** With 9 blocks on 10 threads
every block is in flight at once. The 8192 px row holds nine 8331² working sets, 3× the untiled peak, and
leaves a core idle. Blocks are also the unit of threaded work and their cost varies by orders of magnitude,
so a small pool waits on its slowest member.

**Estimating buffer memory.** For a `UInt8` pair the nine arrays total 18 bytes per pixel (two `UInt8`
planes, three `Float32` and four `Bool`), and 24 bytes for a `Float32` pair. Buffers alone are about
`18 bytes × (block + 2·halo)² × min(nblocks, nthreads)`. At 16384 px on NISAR L1 that is 45.5 GiB before
any imagery or workspace. The 18 bytes are the total over the nine arrays, not per array.

### Wide-halo granules

The NISAR sweeps use a wider halo, so the optimum sits at a different block size.

| | optical (Landsat 8/9) | NISAR L1 RSLC | NISAR L2 GSLC |
|---|---:|---:|---:|
| scene (px) | 17121×16961 | 57760×50511 | 54885×110085 |
| `chip_size` → `chip_size_max` | 16 → 64 | 96×52 → 768×416 | 96×48 → 768×384 |
| `grid_spacing` | 8 | 48 | 48 |
| search radius, x: median / max | 20 / 20 | 34 / 1905 | 34 / 1052 |
| searchable points | — | 35% | 36% |
| halo (px) | 69×69 | 2736×1500 | 2216×1103 |

On NISAR the halo is set by the maximum radius, not the chip, and a Geogrid radius field is heavily
skewed. Reducing `chip_size_max` from 768 to 96 on L2 only narrows the halo from 2216 to 1711 px. Blocks
of 512 or 1024 px are rejected at any chip setting. The halo is not square, so size the two axes
separately: at 2216×1103 a square block clears the x halo and over-provisions y twofold.

Measured at 10 threads on a 96 GiB machine. Occupancy is `cpu_seconds / wall_seconds`.

| case | block | blocks | runtime | peak | vs untiled | occupancy of 10 |
|---|---|---:|---:|---:|---:|---:|
| L1 | untiled | 1 | **9.5 min** | 49.4 GiB | 1.00× | 9.21 |
| | 16384 px | 100 | 30.9 min | 59.6 GiB | 1.21× | 2.33 |
| | 8192 px | 380 | 18.0 min | 33.9 GiB | 0.69× | 4.61 |
| | 4096 px | 1482 | 11.0 min | 27.7 GiB | 0.56× | 8.94 |
| | 2816×1536 px | 5814 | 12.7 min | **24.4 GiB** | **0.49×** | 9.03 |
| L2 | untiled | 1 | 6.2 min | **85.0 GiB** | 1.00× | 6.06 |
| | 8192 px | 98 | 11.2 min | 47.8 GiB | 0.56× | 2.96 |
| | 4096 px | 378 | 7.1 min | 40.1 GiB | 0.47× | 5.17 |
| | 3072 px | 648 | 5.8 min | 31.2 GiB | 0.37× | 6.77 |
| | 2304×1152 px | 2304 | **5.2 min** | 28.8 GiB | 0.34× | 9.03 |
| | 2224×1110 px | 2500 | 6.1 min | **25.9 GiB** | **0.30×** | 7.77 |

Logs for these runs are in `benchmark/results/nisar/`, with the index in `benchmark/results/nisar/README.md`.
The rules that hold on both the optical and the NISAR scenes:

- **Peak flattens once blocks are small.** Optical peak is flat from 1024 to 256 px. On L2, cutting block
  area 3.6× (3072² to 2304×1152) moves peak only 8%, because block buffers are 13 to 21% of peak and the
  imagery, workspaces and output do not shrink with the block.
- **Too few blocks per thread is slow.** Measured occupancy of ten threads: 10 blocks/thread gives 2.3,
  38 gives 4.6, 76 gives 6.6, 148 gives 8.9, 230 gives 9.0, 250 gives 7.8, and 581 gives 9.0 at 21.6×
  read amplification.
- **Runtime is U-shaped in blocks per thread.** Optical runtime is lowest at about 30 to 110 blocks per
  thread. On L2 the fastest row is 230 blocks per thread, and the halo-floor row (250) is 18% slower with
  10% less peak.
- **Redundant reading is cheap next to idle threads.** On L2, 3072 px has 1.69× read amplification and
  runs in 5.8 min at 31.2 GiB, against 0.86× at 8192 px in 11.2 min at 47.8 GiB. The halo gives the
  smallest permitted block and the thread count picks among the permitted sizes.
- **Runtime does not track read amplification.** On L2 the fastest row has the highest amplification
  (3.30×), and GC takes at most 1.7% of wall clock.
- **The formula bounds amplification on a partly empty grid.** `((b + 2h)/b)` per axis predicts 8.52× for
  L2 at 2304×1152 against a measured 3.30×, because 64% of that grid is fill and a block with no
  searchable point reads nothing.

**The untiled run uses a different parallel decomposition.** One block means threads split the points in
each pass, while a blocked run assigns blocks to threads with `threaded = false` inside each. The untiled
row is therefore the fastest on L1, but 4958 MiB against 2140 MiB (optical) or 85.0 GiB against 28.8 GiB
(L2) is the difference between a general-purpose instance and a memory-optimized one.

**Profiled runs on macOS can hang.** A Julia runtime bug in `src/signals-mach.c` deadlocks the sampling
profiler against a thread finishing a collection. It is present through Julia 1.13.0 and fixed on master
(`ca49fc2e2`, not backported). `tools/golden/profiler_gc_deadlock.jl` reproduces it without AutoRIFT code.
All figures here come from unprofiled runs, and an unprofiled worker cannot reach the path.

## A disk-backed pair is read once, not once per pass

For in-memory or memory-mapped input a read is a page touch. For a file, the same amplification is a
decode. A block reads its window once per pass, and a three-level run has up to six passes (a coarse and a
fine pass per chip size), so total reading is a multiple of the sum of the read windows. Measured on a
15901×13435 Landsat 8 GeoTIFF pair (`UInt16`, 0.40 GiB):

| block | blocks | one pass's windows | whole run | vs one pass |
|---|---:|---:|---:|---:|
| 512 px | 924 | 0.96× the scene | 4.41× | 4.6× |
| 1024 px | 238 | 0.74× | 3.60× | 4.9× |
| 2048 px | 63 | 0.66× | 3.37× | 5.1× |

Almost all the redundancy is the repetition across passes, which no block size removes. The count is
fewer than six because a coarse pass sweeps a strided subset of blocks and a level whose points a finer one
resolved may not run.

`cache_budget = :auto` reads the pair into memory once when the pair's bytes are less than the sum of the
read windows times the passes. It is capped at an eighth of `Sys.free_memory()`, divided by the thread
count when `threaded = false`. At 10 threads, one process per arm:

| block | windowed | cached | windowed peak | cached peak |
|---|---:|---:|---:|---:|
| 512 px | 15.2 s | **13.5 s** | 3.56 GiB | 4.16 GiB |
| 1024 px | 14.0 s | 13.6 s | 3.56 GiB | 4.07 GiB |
| 2048 px | 14.1 s | 14.1 s | 4.17 GiB | 4.29 GiB |

The gain is gone by 2048 px, because a larger block reads less to begin with. Caching earns its place when
something else forces a small block, or when the scene is read over a network. It costs the raw pair
(0.5 to 0.6 GiB here against a 0.40 GiB pair), which is half the `Float32` pair for `UInt16` input and a
quarter for `UInt8`. `dx` and `dy` are identical either way (0 of 3,653,760 points differ).

Either the whole pair is cached or none of it is. A blocked run sweeps every block once per pass, so a chunk's
next use is a whole sweep away. Replaying the actual requests through LRU at block 1024 (485 chunks, 121
MiB) gives 3.4× the scene at half the working set and 5.7× at a twentieth. Random replacement gives 5.6×
against LRU's 6.0× at the same capacity, so the access order is the cause, not the eviction policy.

### The read splits on the file's chunk rows

The prefetch reads the scene in row slabs, one task each. A slab boundary inside a stored chunk makes both
neighboring slabs decode it, and for a compressed file the decode is the read. On an 8321×8271 DEFLATE
GeoTIFF at 10 threads with 256-row chunks, an even division into 209-row slabs prefetches in 987 ms
(5.75 s end to end) and 256-row slabs in 278 ms (5.38 s). The pixels are identical. `AutoRIFT._slab_height`
rounds the even division up to whole chunk rows, and an unchunked array reports one row.

## Transform sizes

A point is correlated at its search radius rounded up to a rung of a ladder, capped at the pass maximum
(`AutoRIFT._radius_bucket`). The ladder has four rungs per octave (16, 20, 24, 28, 32, 40, ...), set by
`AutoRIFT.RADIUS_BUCKET_RUNGS`. Distinct radii on the NISAR L1 grid number about 30,000, and the ladder
reduces them to a few hundred transform sizes, each reused by thousands of points.

On NISAR L1 (whole grid, twelve threads) one rung per octave took 590.2 s and four took 467.2 s over the
same 1,799,742 points, because a coarse rung makes a point compute a larger transform than it needs. The
cost of finer rungs is planning: 53 buckets become 388, which adds 110 s once per machine and is removed by
the FFTW wisdom file ([`AutoRIFT.fftw_wisdom_path`](@ref)). Eight rungs per octave would need 996 buckets
for a smaller gain. The ladder is a constant, not a `Params` setting, so there is nothing to tune.

## What an untiled pass holds

Untiled, NISAR L1 peaks at 32 GiB against a 1.29 GiB floor. Most of it is the scene padded by each pass's
halo (6.3 GiB) and the concurrent correlation workspaces (4.4 GiB). Blocking removes the per-pass pad, and
the same granule blocked peaks at 25.2 GiB. Retained correlation workspaces are capped by
`AutoRIFT.WORKSPACE_POOL_BYTES` (default 2 GiB), and those checked out at once by
`AutoRIFT.WORKSPACE_LIVE_BYTES` (default unbounded). The term-by-term breakdown and the paired runs behind
both bounds are in `dev/plan-16gib.md`.

## Timing a large run

Three hazards when timing NISAR-sized runs on this machine:

- **Wall clock is bimodal.** Identical configurations land near 538 s (occupancy 9.6) or near 615 s
  (occupancy 8.6), a 14% split with peak stable to about 0.5 GiB. A single-run wall-clock difference below
  about 15% on this granule means nothing. Interleave arms, and record CPU seconds beside wall clock.
- **Warm up, and do not observe a run while it is timed.** The first row of a sweep carries JIT
  compilation unless a small patch is correlated first. Sampling a live process with `sample` inflated one
  untiled L1 row from about 580 s to 2592 s. Poll with `ps` only.
- **Values are reproducible only within one FFTW wisdom state.** `FFTW_MEASURE` selects by measured
  timing, so a run can persist different wisdom and a later run then executes a different algorithm for the
  same transform size (see `src/plans.jl`). Compare outputs only between runs in the same wisdom state.

## Practical guidance

- **Batch work: one pair per worker, `threaded = false`.** This is 2.6× faster than threading within a
  pair (`benchmark/suite/throughput.jl`). See [Correlating many pairs](@ref).
- **Reuse a `Cache`** through `AutoRIFT.init`, `reinit!` and `autorift!`. Memory is bounded regardless of
  batch length, and no process recycling is needed.
- **Warm the FFTW wisdom before timing anything or sizing a schedule.** On the 11.5 M-point S1A case the
  first run of a sequence took 28.8 s against 6.5 s for the next three, with CPU seconds 71.9 against
  48.3. The difference is planning. See [Performance](@ref) and [`AutoRIFT.warm_plans!`](@ref).
- **Scenes larger than memory: set `process_block_size`.** For file-backed Rasters input the default from
  [`AutoRIFT.block_size_for`](@ref) (the halo, floored at 1024 px) is fitted to a sweep of every golden
  case. For arrays pass a size explicitly. To choose by hand, use `(1024, 1024)` for a narrow halo, size each axis against
  [`AutoRIFT.halo`](@ref), and aim for 100 to 250 blocks per thread without going to the floor. Peak falls
  as blocks shrink but runtime does not: on L1 the lowest peak is `(2816, 1536)` at 0.49× untiled, and the
  fastest blocked row is `4096` square, 13% quicker for 3 GiB more.
- **Budget for the raw pair on top of the blocks when the input is a file.** `cache_budget = nothing` holds
  peak to the blocks alone, and a byte count overrides the automatic choice where free memory is a poor
  guide, such as under a cgroup limit. To keep the raw pair off the heap entirely, memory-map it. A
  resident input is never copied, and the kernel can reclaim its pages under pressure.
- **Small instances: use the trimmed binary.** 27.2 MiB against 424.2 MiB is 3.1% against 43% of a
  `t3.micro`.
