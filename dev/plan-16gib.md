# Fitting every golden case in 16 GiB

The target is a 16 GiB instance, because that is the cheap shape: `m7g.xlarge` is 4 vCPU and 16 GiB,
`c7g.2xlarge` is 8 vCPU and 16 GiB. This records where a run's memory actually is, which of the
plausible levers are real, and the order to take them in.

## The target is met: 22 of 22 cases under 16 GiB

`tools/golden/case_peaks.jl` reports **0 of 22 over 16 GiB** at each case's best recorded configuration.
The binding case is 15.15 GiB and the largest case measured at a block size is NISAR L1 at 12.17 GiB.

Both NISAR granules — the two `memory.md` documents as the expensive ones, at a recorded 24.43 and 25.91
GiB — come in at **12.17 and 10.50 GiB** blocked. The four Sentinel-1 cases that were over had never had a
single blocked row measured between them; blocked at 1024 px they land at 3.50–4.75 GiB, every one of them
faster than untiled:

| case | untiled | 1024 px | above floor | untiled wall | 1024 px wall |
|---|---:|---:|---:|---:|---:|
| S1A `1SSH_20170221` | 24.85 GiB | **3.50** | 0.44 | 27.0 s | 24.9 s |
| S1A `1SSH_20151120` | 22.06 GiB | **4.75** | 1.72 | 18.0 s | **10.4 s** |
| S1B `1SDH_20180809` | 21.96 GiB | **3.53** | 1.02 | 35.0 s | **14.7 s** |
| S1C `1SDV_20250416` | 20.60 GiB | **3.95** | 1.02 | 15.2 s | 12.7 s |

**No code was needed for any of it** — only a block size those sweeps had never been run with. The 2048 px
rows in the same sweeps read 14–19 GiB but sit **0.41–0.48 GiB above floor**, which is residue from the
untiled row that preceded them in the same process rather than the block's own cost; "above floor" is the
comparable column and a production worker runs one configuration per process.

Two caveats, both load-bearing:

- **Two S1 cases are still the census's largest only because they have never been blocked.** `1SSV_20240618T025528`
  at 15.15 GiB and `1SSH_20150828` at 12.79 GiB are untiled-only, and their siblings say both would land
  near 4 GiB. Blocked, the whole set's maximum would be NISAR L1's 12.17 GiB.
- **A blocked run on a geogrid still differs from an untiled one**, by 5.3% of points on S1B. That is
  item 3 and it is a correctness question, not a memory one — the figures above are what the runs cost, not
  a statement that their output is the reference's.

So the remaining obstacle was not a memory lever at all: **a blocked run did not reproduce an untiled one
on a Sentinel-1 geogrid** — 5.3% of points — root-caused to the grid's placeholder coordinates, and now
fixed. See "Step A-3" below for the fix and the gate.

**Two claims in an earlier revision of this section were wrong and are corrected there.** That a blocked
run was "the more nearly correct path" is refuted: the reference measures 25,970 of the 26,781 points a
blocked run was losing, 97.0%, and an untiled run reproduces the reference at them to a median of 0.000 px.
And the placeholder share of the lost set was recorded as 58% where it measures 22.7%. Step 0 has both
measurements.

**Figures measured here and figures in `memory.md` come from different trees.** Re-measured on this one,
NISAR L2 at `2304x1152` measures 147,099 fewer points than the recorded row and L1 at `2816x1536` 2,349
fewer. The arithmetic and the ranking below do not depend on which peak column is current.

Reproduce the measurements below with `tools/ab/mem_stages.jl` and `tools/ab/mem_churn.jl` on the
Landsat pair, `tools/golden/with_copies.jl` against `tools/golden/profile_nisar.jl` on a NISAR granule,
and `tools/ab/halo_terms.jl` for the halo arithmetic, which needs no imagery. Where a figure is
arithmetic rather than measured it says so.

## Peak is live working set, not allocator slack

This is the first thing to settle, because it decides whether the problem is the collector or the
code, and the two call for opposite work.

`--heap-size-hint` forces the collector to a lower target. On the Landsat pair, block 1024 px, ten
threads, `preprocess = :highpass`:

| heap hint | peak footprint | live peak | allocated | GC | wall |
|---|---:|---:|---:|---:|---:|
| default | 2.127 GiB | 1.382 GiB | 33.5 GiB | 0.78 s | 13.7 s |
| 800 MiB | 1.845 GiB | 0.915 GiB | 33.7 GiB | 14.19 s | 27.7 s |
| 400 MiB | 1.870 GiB | 0.912 GiB | 34.0 GiB | 14.66 s | 28.0 s |

**A tighter target buys 13% of peak for 2.0x the wall clock, and then saturates** — 400 MiB and
800 MiB give the same peak, so ~1.85 GiB is a floor the collector cannot go below however hard it is
pushed. GC rises 18x to get there. Peak is what the run holds, not what the collector has failed to
return.

This matters for a second reason. `Sys.total_memory()` is cgroup-constrained, and that is the figure
the collector's target is derived from, so **the same job in a 16 GiB container collects harder than it
does on a 96 GiB host.** Moving a run to a smaller instance does not only risk the OOM killer; it also
pays that GC bill. A configuration that fits 16 GiB only by leaning on heap pressure will be slow
there, which is the opposite of what the instance was chosen for.

## What the peak is made of

Stage by stage on the Landsat pair, `preprocess = :none`, block 1024 px, ten threads, each figure
`phys_footprint` after a full collection (`tools/ab/mem_stages.jl`):

| after | footprint | live |
|---|---:|---:|
| `using AutoRIFT` | 455 MiB | 25 MiB |
| + both planes memory-mapped | 455 MiB | 25 MiB |
| + grid, 2127x2107 = 4.48 Mpts | 729 MiB | 230 MiB |
| + block layout, 289 blocks, halo 67x67 | 798 MiB | 230 MiB |
| + one displacement field | 803 MiB | 371 MiB |
| + warmup | 893 MiB | 375 MiB |
| **peak during the run** | **2385 MiB** | 1638 MiB |
| after the run, result held | 1552 MiB | 453 MiB |

Three things fall out of this, and the first two are not what the block-size sweep in
`docs/src/explanation/memory.md` implies.

**On a narrow-halo scene the peak is grid-sized arrays, not imagery.** The largest read window is
1151x1151 = 1.32 Mpx, so ten tasks of `BlockBuffers` are 170 MiB of the 2385. The grid is 205 MiB by
`summarysize`, a displacement field 73 MiB, and `_level_points` allocates four more grid-sized arrays
per chip-size level — two of which are `fill(chip_size.X, n)`, a constant stored densely where
`AutoRIFT.Uniform` exists for exactly that and costs 24 bytes. That is why the optical curve is flat
from 1024 px down to 256 px: below 1024 px there is no imagery left to remove.

**Mapped input is genuinely free of the footprint.** Mapping 2.17 GiB of planes moves footprint by
0 MiB, confirming that `phys_footprint` excludes clean file-backed pages. The figure to size an
instance from is footprint, and `resident` overstates a lazily-read configuration by the page cache
its own reads populated.

**The Julia runtime is 455 MiB, 2.8% of a 16 GiB budget.** Not worth attacking at this scale, against
27% of the optical peak, which is why the trimmed binary matters for small pairs and not for this.

## Three redundant copies per block read, worth 7.8x of the allocation — removed

Every golden end-to-end run uses `preprocess = :none` (`tools/golden/e2e.jl:764`), because the
reference filters before its correlator and the comparison feeds imagery already filtered. On that
path a block read made three copies of each window that the caller was already holding:

- `_read_window!` materializes `img[rows, cols]` and then copies it into the buffer. Deferring
  through a view is what the generic method must not do, since a lazy array would then be read one
  pixel at a time — but a strided parent has nothing to defer.
- `_prepare_block` has an in-place method for `Highpass` alone. Every other method falls through to
  `_prepare`, and for `NoPreprocess` that is `copy(img), copy(mask)`.
- `_prepare` then calls `replace_nonfinite`, which for an integer image is *documented* as "the copy
  is the whole operation" — a second copy pair, of an operation that cannot change an integer image.

`src/tile.jl` now has a `_prepare_block` method for `NoPreprocess` that substitutes the non-finite
values in place (`AutoRIFT.replace_nonfinite!`) and returns the raw pair, and a `_read_window!` method
that takes a view for a strided parent. `tools/ab/mem_churn.jl`'s `--with-prepare-copy` and
`--with-read-temp` put the old behaviour back, which is what keeps this table reproducible.

Landsat pair, block 1024 px, ten threads, `preprocess = :none`, one process per arm:

| arm | peak | live peak | allocated | GC | wall |
|---|---:|---:|---:|---:|---:|
| both copies | 2.168 GiB | 1.425 GiB | 47.7 GiB | 1.39 s | 9.7 s |
| read temporary only | 2.022 GiB | 1.364 GiB | 18.1 GiB | 0.55 s | 8.9 s |
| **as it ships now** | **1.856 GiB** | **1.051 GiB** | **6.1 GiB** | 0.35 s | 8.2 s |

**Allocation 47.7 -> 6.1 GiB and GC 1.39 -> 0.35 s, `dx` bit-identical** — the same checksum over raw
bits and the same 534,930 points in every arm. Peak falls 14% and live peak 26%, monotonically.

**The wall-clock column here is noise and should not be read.** Five measurements of this scene across
the two endpoint arms give 9.7, 8.9, 8.2, 9.9 and 10.5 s with no ordering by arm. A re-measurement of
the endpoints after the change landed reproduces the allocation exactly — 47.7 and 6.1 GiB — and gives
2.102 and 1.647 GiB of peak, so allocation is the column that reproduces and peak carries ±7%. The
speed result is the NISAR one below, where the run is minutes rather than seconds.

On a 1.32 Mpx window the copies are 13 MiB per task. **On a wide-halo granule they are not.** At NISAR
L2's 22.62 Mpx window the `_prepare` output alone is 4 bytes per pixel held live for the whole of a
block's correlation — 0.84 GiB across ten tasks — and the churn is most of that granule's 1282 GiB.

## Peak scales with the task count exactly where the window is halo-dominated

`docs/src/explanation/memory.md` lists thread count as a non-knob. That was measured on 512-pixel
pairs, where a block's window is a few MiB, and it does not transfer: `BlockBuffers` and one
correlation workspace are held **per task**, so the per-task term is whatever the read window costs.

Two geometries on the same scene, block 1024 px. The optical rows are the whole plane at
`preprocess = :highpass`, so they are comparable to the heap table above; the wide-halo rows are a
5120² sub-window at `preprocess = :none`, `chip_size = 64`, `chip_size_max = 256`,
`grid_spacing = 64`, `search_radius = 500`, which gives a 727x727 halo and a 2415x2415 = 5.83 Mpx
window:

| geometry | threads | peak | live peak | wall |
|---|---:|---:|---:|---:|
| optical, halo 69x69, window 1.32 Mpx | 10 | 2.127 GiB | 1.382 GiB | 13.7 s |
| | 8 | 2.032 GiB | 1.218 GiB | 14.7 s |
| | 4 | 1.872 GiB | 0.949 GiB | 25.0 s |
| | 2 | 1.783 GiB | 0.809 GiB | 47.3 s |
| wide halo, 727x727, window 5.83 Mpx | 10 | 3.433 GiB | 3.564 GiB | 11.0 s |
| | 4 | **2.353 GiB** | 1.821 GiB | 16.6 s |

`dx` is bit-identical across the thread counts in both geometries. Ten to four threads is **-31% of
peak for +51% wall** on the wide-halo geometry against **-12% for +82%** on the optical one.

Note the second column on the wide-halo row: live peak 3.564 GiB against a 3.433 GiB footprint. That
is not a contradiction — `gc_live_bytes` is the live set at the last collection *plus* everything
allocated since, so it overstates. Where the two are close the run is holding what it allocated;
where footprint is 1.5-2x it, the gap is heap the collector has not had cause to reclaim.

So the thread count is a real lever on exactly the two cases that need one, and a poor one everywhere
else. It is also not a free lever — it is the axis the instance has already chosen, since a 16 GiB
instance comes with 2, 4 or 8 vCPU. **The quantity to design against is memory per vCPU, not memory
per process:** 4 GiB/vCPU on these shapes, against a NISAR per-task working set of roughly 0.8 GiB
plus its share of a ~4 GiB fixed cost.

## The read window is sized for the coarsest level and the widest point in the grid

This is where the remaining memory is, and it is arithmetic rather than a measurement.

`halo` takes the maximum of `chip_size_max/2 + radius + |prior| + 2 + filter_reach +
level_centre_offset` over every level and every point. `BlockLayout` carries that single figure,
`restrict` passes each block's read window through unchanged from level to level, and `block_buffers`
sizes to the largest window in the layout. So the **base** level of NISAR L1 — chip 96x52, undecimated,
no centre offset — reads a window sized for a 768x416 chip at a 1905 px radius.

`tools/ab/halo_terms.jl`, NISAR L1 RSLC, recorded halo 2736x1500, at the recorded `2816x1536` block:

| level | halo | window | `BlockBuffers`, 10 tasks |
|---|---:|---:|---:|
| 96x52 (base) | 2231x1149 | 27.9 Mpx | 4.68 GiB |
| 192x104 | 2304x1200 | 29.2 Mpx | 4.90 GiB |
| 384x208 | 2448x1300 | 31.9 Mpx | 5.35 GiB |
| 768x416 | 2736x1500 | 37.6 Mpx | 6.30 GiB |
| **as it ships, every level** | **2736x1500** | **37.6 Mpx** | **6.30 GiB** |

Per level alone is worth 1.6 GiB, because the radius dominates the halo and the chip does not. The
radius is the term to split, and the shape of its distribution is why: over the searchable points of
this grid the x radius has **median 34 and maximum 1905**, the maximum is reached at exactly **4
points**, and p99 is 959. `dev/GATES.md` records that a *spatial* per-block halo is worth only
1.03-1.11x, and the reason is in those numbers — 53-60% of blocks contain at least one wide point at
every block size tried, so partitioning space does not partition the radius.

Partitioning the **radius** does, and it is exactly answer-preserving. `_radius_bucket(r, cap)` is
`min(nextpow2(r), cap)` with `cap` the pass radius. For a class `{r : r <= 2^k}` run at pass radius
`2^k`, every member gets `min(nextpow2(r), 2^k) = nextpow2(r)`, which is what the whole-grid cap gives
it too — so every point is correlated at the same transform size it is now, provided the class caps sit
on the power-of-two ladder and the top class keeps the grid's own maximum.

| level | radius class | halo | window | `BlockBuffers`, 10 tasks |
|---|---:|---:|---:|---:|
| 96x52 | <= 64 | 178x156 | 1.84 Mpx | **0.31 GiB** |
| 96x52 | <= 256 | 562x540 | 4.52 Mpx | 0.76 GiB |
| 96x52 | <= 1024 | 1350x1149 | 13.96 Mpx | 2.34 GiB |
| 96x52 | <= 1905 | 2231x1149 | 23.07 Mpx | 3.87 GiB |
| 768x416 | <= 256 | 1067x891 | 8.98 Mpx | 1.51 GiB |
| 768x416 | <= 1905 | 2736x1500 | 36.94 Mpx | 6.19 GiB |

The workspace follows the same ratio and more steeply, since a transform scales with `chip + 2*radius`
in each axis: the base level at bucket 256 is a 0.34 Mpx transform against 9.5 Mpx at bucket 1905, 28x
smaller.

**The top class is the one that cannot be shrunk, and it does not need to be — it needs to be
throttled.** A class holding a fraction of a percent of the points can run on two tasks rather than
ten and cost 1.24 GiB rather than 6.19, with a wall-clock consequence proportional to its share of the
work rather than to its share of the memory.

## What it measures on NISAR L2

The row itself, whole grid, `2304x1152` blocks, 2304 blocks, `-t 10,1`, `--no-profile`, one process per
arm, back to back on an otherwise idle machine (`tools/golden/with_copies.jl` for the baseline):

| NISAR L2 GSLC, 2304x1152 | baseline | copies removed | |
|---|---:|---:|---:|
| wall | 368.1 s | **258.6 s** | **0.70x** |
| occupancy | 7.46 / 10 (75%) | **9.48 / 10 (95%)** | |
| peak footprint | 10.50 GiB | **9.58 GiB** | 0.91x |
| peak above floor | 8.74 GiB | 7.81 GiB | 0.89x |
| peak resident | 22.06 GiB | 21.11 GiB | |
| live peak | 12.58 GiB | 11.56 GiB | |
| allocated | 1282.5 GiB | **181.4 GiB** | **7.07x less** |
| GC | 12.2 s, 3.3% of wall, 2460 pauses | **1.8 s, 0.7%, 391 pauses** | |
| points measured | 1,634,278 | 1,634,278 | |

**The larger half of the result is speed, not memory: 30% off the wall clock and occupancy from 75% to
95%.** Peak falls 8.8%. Unlike the optical arms, this is not scheduling noise: CPU seconds fall too,
2746 to 2451, so 11% of the work is gone as well as 20 points of occupancy. The overrides fired 5417
and 10834 times — exactly two window reads per `_prepare_block`, one per image — which is what rules
out a signature mismatch reporting the baseline twice.

The arm was measured with `_prepare_block` returning `raw` outright. On this granule that is the same
machine code as the shipped form: the imagery is `UInt8`, and `replace_nonfinite!` for an integer image
is a no-op by dispatch. So the figures stand for the landed change without re-measuring.

The occupancy is the mechanism. 1282 GiB of churn at 3.3% of wall in 2460 pauses is 433 *full*
collections, each of which stops every thread; removing the churn removes the stop-the-world time, and
the threads that were waiting on it are the 20 points of occupancy. This is the same effect
`memory.md` records in the other direction at a 128-pixel block, where read amplification made a run
allocation-bound and left 6 of 10 workers in `__psynch_mutexwait` beneath `jl_safepoint_start_gc`.

### This tree does not reproduce the recorded row, and that is worth looking at

The baseline above is **not** the 28.79 GiB / 309.4 s / 1,781,377-point row in `memory.md`. Same case,
same capture, same `--run 100`, same block shape — `readamp 3.30x` and the `3336 x 6690 px` window
match the record exactly, so the grid and the layout are identical. What differs:

| | recorded | this tree |
|---|---:|---:|
| points measured | 1,781,377 | **1,634,278** |
| process floor before the row | 21.3 GiB | **1.76 GiB** |
| peak | 28.79 GiB | 10.50 GiB |
| peak above floor | 7.5 GiB | 8.74 GiB |

**Peak above floor is the comparable quantity and it is unchanged**, so almost all of 28.8 -> 10.5 GiB
is the process floor falling by 19.5 GiB rather than the row costing less. The uncommitted `src/`
changes in this working tree are the only thing between the two measurements.

The point count is the part that needs a decision rather than a note: **147,099 fewer points, 8.3%,
measured on identical inputs.** That may be intended by the work in progress or it may be a regression;
either way a peak figure and a point count from different trees are not comparable, and the recorded
sweep should be re-run once that tree settles.

### The per-task arithmetic checks out

Sizing the per-task term from the code, `UInt8` imagery, `preprocess = :none`, window 22.62 Mpx:

| per task, at a 22.62 Mpx window | bytes/px | MiB |
|---|---:|---:|
| `BlockBuffers`, the four arrays `:none` touches | 4 | 90 |
| `_prepare` output, live for the block's correlation | 4 | 90 |
| `_prepare` intermediate, dead immediately | 4 | 90 |
| read temporary, in flight | 2 | 45 |
| correlation workspace, widest bucket touched | — | ~436 |
| **total** | | **~750** |

The three filter planes and two eroded masks — 14 of the struct's 18 bytes per pixel — are allocated
`undef` and never touched on the `:none` path, so they are address space rather than footprint. Ten
tasks is ~7.5 GiB; the fixed terms (grid 0.25, output 0.09, pool bound 2.0, runtime 0.45, per-level
fields ~0.8) are ~3.6 GiB, giving **~11 GiB of live against a measured 12.58 GiB live peak** — and
`gc_live_bytes` overstates, since it counts everything allocated since the last collection. The term
model is therefore good to about 15%, which is what makes the remaining prediction worth making.

The per-task term is 68% of that total, and it is a window cost: 4+4+4+2 bytes per pixel of window plus
one workspace. With a per-level, per-class window the common classes go to a 1.84-4.52 Mpx window from
22.62, and the workspace with them. **Predicted per task ~100 MiB against 750, so live ~4-5 GiB and
peak ~4-5 GiB** at the ratio measured here — comfortably inside 16 GiB with room for the eight vCPU
shape rather than the four.

### Removing the copies is answer-preserving, and one of the two was untested

`test/tile.jl` passed in full with both changes — 36,700 tests, including the 122 of "a blocked run
equals an untiled one", the caller-supplied-grid and rotated-grid sets, and the threaded determinism
set. `_read_window!` was reached 3700 times there, so that change was covered.

**`_prepare_block` was reached zero times.** Every blocked-equals-untiled testset ran a *filtered*
configuration, so the `NoPreprocess` fallthrough — the path every golden end-to-end run takes — had no
coverage in the suite at all. That is how the wrong form of the change passed all 36,700 of them:
returning the raw pair without substituting non-finite values loses **122 of 6843 points** on a float
pair with a no-data border, and changes `dx` where it does not.

`test/tile.jl` now has "a blocked run equals an untiled one at preprocess = :none" — 44 tests in 11.4 s,
`Float32` with and without no-data, `UInt8` and `Int16`, at blocks of 512 and 272, all five result fields
under `isequal`, with a point-count floor so an all-`NaN` pair cannot satisfy it. Two ways a check there
goes vacuous and it excludes both: `:none` over *raw* texture measures zero points, since nothing
high-passes it and every level is rejected for incoherence; and a pair carrying no non-finite value
never reaches the substitution at all.

**The testset was confirmed to fail on the defect it exists to catch.** Restoring `return raw` reddens
exactly the `Float32 with no-data` arm, at both block sizes and on all five fields, and nothing else:

```julia
using AutoRIFT: BlockBuffers, ImagePair, Params, NoPreprocess
AutoRIFT._prepare_block(::BlockBuffers, raw::ImagePair, ::Params, ::NoPreprocess, ::Int, ::Int) = raw
include("test/utils.jl"); include("test/tile.jl")
```

## Every golden case, and the ones nobody had blocked

The plan opened by naming two cases over 16 GiB. That was wrong, and it was wrong because it read the
two granules `memory.md` documents rather than the recorded history of all of them. Lowest recorded peak
per case, from `$AUTORIFT_GOLDEN_CACHE/mem/prof_*.jls` alone — no correlation, just deserialization:

`tools/golden/case_peaks.jl` reports it in seconds. Lowest recorded peak per case, with how many of that
case's rows were blocked at all:

| case | best recorded block | peak GiB | blocked rows |
|---|---|---:|---:|
| S1C `1SDV_20250416` | **untiled** | **19.46** | **0 of 4** |
| S1A `1SSH_20170221` | **untiled** | **19.21** | **0 of 42** |
| S1A `1SSH_20151120` | **untiled** | **18.01** | **0 of 12** |
| S1A `1SSV_20240618T025528` | untiled | 15.15 | 0 of 4 |
| S1A `1SSH_20150828` | untiled | 12.79 | 0 of 4 |
| NISAR L1 RSLC | `(2816, 1536)` | 12.17 | 8 of 10 |
| NISAR L2 GSLC | `(2304, 1152)` | 10.50 | 9 of 10 |
| S1B `1SDH_20180809` | `(1024, 1024)` | **3.53** | 3 of 5 — blocked this session |
| the 14 optical and S2 cases | mostly untiled | 2.29 – 6.42 | — |

**Three cases are over 16 GiB, and none of them has ever had a single blocked row.** One history holds 42
rows and every one is `untiled` — so `process_block_size`, the knob `memory.md` documents for exactly this
purpose, has never been applied to the cases that need it. The two NISAR granules are now among the
*cheapest* large cases rather than the most expensive.

S1B is the fourth such case and is no longer over the limit: blocking it is the measurement below.

## Sentinel-1 blocked: 21.96 -> 3.94 GiB, and 2.4x faster

S1B `1SDH_20180809` — one of the four with no blocked row — whole grid, `-t 10,1`, one process, scene
23860x67945, halo 684x256:

| block | blocks | runtime | peak | above floor | occupancy | read amp |
|---|---:|---:|---:|---:|---:|---:|
| untiled | 1 | 35.0 s | 21.96 GiB | 12.41 | 3.68 / 10 | 1.00x |
| 4096 px | 574 | 15.8 s | 20.85 GiB | 0.58 | 7.59 / 10 | 0.89x |
| **2048 px** | 2187 | **14.3 s** | **3.94 GiB** | 0.97 | 7.92 / 10 | 1.38x |
| 1024 px | 9018 | 14.7 s | **3.53 GiB** | 1.02 | 8.52 / 10 | 2.74x |

**Blocking is a pure win here on every axis at once** — 0.18x the peak and 0.41x the runtime at 2048 px,
with occupancy more than doubling. That is a larger margin than anything else in this file, and it needed
no code: only a block size the sweeps had never been run with. It is also why the three cases still over
the limit are very likely not a memory problem at all.

The 4096 px row shows why `above floor` is reported beside `peak`: its peak is 20.85 GiB against a 20.28
GiB floor, so the row itself costs 0.58 GiB and the figure is almost entirely what the process was
already holding when it started. Peak alone would read as "blocking barely helps at 4096".

## But a blocked run does not reproduce an untiled one on Sentinel-1

Blocking promises bit-identical output, and `test/tile.jl` asserts it on synthetic grids. On S1B it fails,
at every block size, and the deficit grows as blocks shrink: 489,081 points at 4096 px, 481,630 at 2048,
481,493 at 1024, against an untiled **507,640**.

**It is not this session's change.** `tools/golden/with_copies.jl` — the per-block copies restored —
reproduces the failure exactly: the same 481,630 points and the same `dx identical false` at 2048 px. So
the defect predates the copy removal, and the identical counts are one more confirmation that removing
them is answer-preserving.

Classified with `tools/golden/block_agreement.jl` at 2048 px, and the classification rules out the two
obvious causes:

| | |
|---|---|
| lost / gained / both-measured-but-differing | 26,781 / 771 / 25,206 |
| lost points' distance to their own block's grid edge | median **7.0** grid points, 14.9% at 0–1 |
| kept points', for comparison | median **6.0**, 16.2% at 0–1 |
| `dx` difference where both measured | median 0.0065 px, p99 0.44, max 2.65; 73.4% within one upsampling step |
| `correlation` differs at | **4.0%** of points where both measured |
| of those, `dx` nevertheless identical at | **0.8%** |
| `correlation` difference | median 0.00225, max **1.05** |
| untiled correlation at lost points | median **0.041**, against 0.277 at kept points |
| blocks affected | **153 of 2187**; median 4% of that block's points; **20 lose all of them** |

**Not a seam effect, so not an under-computed halo.** Lost points sit no closer to their block's edge than
kept ones — further, if anything. A halo that reads too little loses the rim, and this loses the interior.

**Not a last-bit numerical difference either**, which was the next hypothesis and a plausible one: a
block's integral tables start at the block's origin rather than the scene's, so the normalization
accumulates a different number of terms, and such a perturbation would move `correlation` almost
everywhere while moving `dx` only where the peak is flat. The measurement says the opposite — correlation
is bit-identical at **96%** of points, and where it does differ `dx` moves with it 99.2% of the time, by
as much as 1.05 in correlation. That is a different surface, not a rounding.

### Bisected to the coarse gate

`tools/golden/block_bisect.jl` removes one stage of the level machinery at a time. S1B at 2048 px:

| arm | lost | gained | moved |
|---|---:|---:|---:|
| as it runs | 26,781 (5.28%) | 771 | 25,206 |
| outlier filter off | 27,314 (1.57%) | 86 | 12,341 |
| one chip size | 874 (0.26%) | 1 | 6 |
| **one chip size, gate not selecting** | **0** | **0** | **0** |
| **one chip size, outlier filter off** | **0** | **0** | **0** |

**Blocking itself is exactly right.** One chip size with the coarse gate not selecting is bit-identical
over 341,431 points, *with the outlier filter still on* — so the halo, the read windows, the per-block
pass, `_prepare_block`, `_read_window!` and `_reject_and_fill!` on the assembled field are all correct.

The coarse gate is the sole entry point, it contributes 874 points at one level, and **the chip-size
cascade amplifies that 30x**: each level decides which points the next attempts, so a handful of
differently-gated points at the base level compounds to 26,781 over four.

### The mechanism: a span reduced over the wrong radii

Two things key on a point's **own** search radius:

- `block_layout` sizes each block's read window as `_searchable_span` of its points, grown by the halo. A
  point with radius zero is not searchable and contributes nothing to that span.
- `_run_one_block!` returns before any I/O when `nsearchable` of the block's points is zero.

`_coarse_points` then builds the coarse set with `_cell_max_radius!`, which gives each coarse point the
**maximum radius over a neighbourhood** `_sparse_filter_width(_sparse_stride(p))` wide — 9 grid points
here. So a point whose own radius is zero inherits a neighbour's and becomes searchable *in the coarse
pass alone*, at a location the span was reduced without.

Counted with `tools/golden/coarse_span.jl`, arithmetic only, on the same case and block size:

| level | coarse points searchable only via a neighbour's radius | of those, not fitting their block's window |
|---|---:|---:|
| chip 68x16 | 1,130 | **668** |
| chip 136x32 | 1,023 | 374 |
| chip 272x64 | 1,061 | 303 |
| chip 544x128 | 678 | 405 |

and one block per level whose coarse points face a **zero-size** read window — which is block 934 above,
`window 0x0` with 3,503 grid points in it.

A point that does not fit sends `track!` down its `fits = false` branch, which zero-pads. Padding is not
data, so **the coarse pass searches padding where the untiled run searched imagery**, its evidence
differs, and the mask it dilates from that evidence gates off whole regions of the fine pass. The
arithmetic matches the symptom: 668 coarse points each stand for a `stride x stride` = 64-point cell
before dilation, which is ~42,000 fine points against 26,781 observed lost.

**The halo is not what is short, and widening it does not help.** The halo is already the global maximum
reach over every point, and a neighbourhood maximum of those same radii cannot exceed it. What is short is
the **span** the halo is added to.

### Widening the window is not the fix, and the untiled run is the one at fault

The first attempt was to grow the span: reduce it over a grid-index range grown by that reach, and pad the
result by the reach converted to pixels through `_index_rates`. Measured on S1B at 2048 px, it **recovered
29% of the lost points and cost 6.3x the read window** — 1505x2804 px became 3075x8631 — which would undo
the 3.94 GiB result it was meant to protect. Reverted; the patch is kept as
`dev/plan-16gib-span-attempt.patch` rather than in `src/`.

What it established is where the remaining 71% is not. With the span grown, the unfit counts did not move
at all: still 668/374/303/405. So the shortfall is not a position offset, and measuring the **overshoot** —
by how many pixels an unfit point's window exceeds its block's — says why: a median of **~46,000 px** and a
maximum of **66,490** against a 67,945 px scene. Those points are most of the scene away from the block
they were assigned to. No span covers that, and nothing should try.

Their coordinates close it. The inherited coarse points have a median x of **1.5** — the scene's left edge —
with extrema `(1.5, 36334.5)`. They are the geogrid's **fill** points, outside the radar footprint. The fill
presents as 1.5 rather than the 0 `memory.md` records for NISAR because the reference's 0-based fill maps
through this package's `+1.5` grid-origin convention, which is why a test for `(0, 0)` finds none of them
and why this took three wrong hypotheses to reach.

So the two paths diverge in the opposite direction from the obvious one:

- A fill point has radius zero, so `issearchable` rejects it and `_searchable_span` correctly ignores it.
- `_cell_max_radius!` then gives it a neighbour's radius, making it searchable **in the coarse pass alone** —
  1,130 such points at the base level.
- **Untiled** pads the whole scene and correlates that point against the scene's corner, producing a
  measurement for a point that has no position. Its mask is valid there (`UInt8` imagery is finite
  everywhere), so nothing rejects it.
- **Blocked** reads the window its block's real points need, so the fill point is wholly outside it and is
  skipped — the documented behaviour for a chip lying outside the image.

**The untiled run is the one measuring something spurious, and blocked is right to skip it.** Those
spurious coarse measurements then feed `reject_outliers` and the dilated mask, which is how 1,130 points
per level become 26,781 lost after the cascade.

That makes the fix a correctness decision rather than a mechanical one, which is why it is not made here:
withholding searchability from a point with no coordinate changes the **whole-scene** answer, and the
whole-scene answer is what is validated bit-for-bit against autoRIFT. Either `_cell_max_radius!` must not
confer searchability on a point that had none, or the grid's fill points must be excluded before it runs —
and either way the reference's own behaviour on such a point has to be established first. `dev/CORRECTNESS.md`
is where that belongs once it is.

### The guard, and the second population it found

`AutoRIFT._block_window_shortfall` now reports a block whose points reach past its own read window on a side
that is **not** clipped to the scene's edge, and `_run_one_block!` warns once per session before any block
does I/O. Positionless points are excluded by construction: mixing them in would bury a real shortfall of a
few hundred pixels under a spurious one of forty thousand.

Writing it settled what the span attempt had left open. **There are two populations of inherited coarse
point, not one:**

| | overshoot | what it is | right answer |
|---|---:|---|---|
| fill points | median ~46,000 px | outside the footprint, coordinate at the scene edge, no position | skip, as a blocked run does |
| near-span points | tens to hundreds of px | a real coordinate just outside the block's span | read it, which needs a wider window |

The guard fires on the second. On S1B: a window of 1482x1827 px against a point whose search window spans
rows **-19:23** — the point sits 2 px inside the window and needs 21, so it is about 254 px outside the span
on a low-row side that is not the scene's edge. On NISAR L2 at `2304x1152`: a 1978x2830 window against
columns **2705:2907**, 77 px over on an unclipped high-column side.

So the span attempt was addressing a real population after all — the 29% it recovered — and the 71% it did
not is the fill points, which no window should try to cover. That is the split item 3 has to decide, and it
is why the two halves cannot be fixed by one change.

**It warns rather than throws, and that is a judgement rather than caution.** Every geogrid case reaches the
condition — both NISAR granules and every Sentinel-1 one — so an error would take the blocked path from
inexact to unusable on exactly the cases that need it, and blocking is the only route that fits them in 16
GiB. `_warn_coarse_fallback` in the same file is the same call made for the same reason. `track!`'s refusal
of a too-narrow `geometry` is not the precedent it looked like: that condition is unreachable in a correct
configuration, and this one is reached by every real radar grid today.

Verified non-invasive: S1B blocked reports the identical 26,781 lost, 771 gained and 25,206 moved with the
guard as without it, and on the Landsat pair allocation moves 6.1 -> 6.2 GiB with the `dx` checksum
unchanged and wall clock inside its own 8.2-11.1 s spread. Seven tests cover the predicate, including that
it stays silent on a clipped side, on an empty window, on a positionless point, and across every layout
`block_layout` builds.

**This blocks the memory goal rather than sitting beside it.** Blocking is the only route that takes these
four cases under 16 GiB, so they cannot be moved to a 16 GiB instance until a blocked run on them is
trustworthy.

## The in-use workspace bound is neutral, and blocking is why

`AutoRIFT.WORKSPACE_LIVE_BYTES` bounds the bytes of correlation workspace **checked out at once**, where
`WORKSPACE_POOL_BYTES` bounds what the pool *retains*. It was built because the term accounting left
~4.5 GiB of NISAR L2's 7.5 GiB above floor unexplained by buffers, grid and pool, and a task holds one
workspace at a time so ten tasks hold ten.

**It is neutral on peak and on speed.** NISAR L2 blocked at `2304x1152`, interleaved, one process per arm:

| arm | budget | above floor | wall | occupancy | points |
|---|---|---:|---:|---:|---:|
| 1 | 2 GiB | 7.41 GiB | 333.1 s | 7.51 / 10 | 1,634,278 |
| 2 | none | 7.36 GiB | 252.9 s | 9.56 / 10 | 1,634,278 |
| 3 | 2 GiB | 7.52 GiB | 251.9 s | 9.66 / 10 | 1,634,278 |
| 4 | none | 7.58 GiB | 249.8 s | 9.68 / 10 | 1,634,278 |

The pattern is **position, not budget**: the first arm is slow whatever the setting and arms 2-4 are
indistinguishable. A 1 GiB bound also never changed wall clock, which is the useful part of the result —
it means the bound never blocked, so **live workspaces were under 1 GiB throughout**, not 4.5.

**The 4.36 GiB figure the prediction came from is an *untiled* L1 measurement and does not transfer.** An
untiled pass holds the whole grid's radius range, so `pass_geometry` gives it the widest bucket on the
grid; a blocked run's per-block pass spans whatever narrow range that block's points happen to have. So
blocking already bounds live workspaces, and it does so more tightly than a byte budget would.

**Kept rather than reverted**, and off by default. The property it names is real, the mechanism is written
and tested, and a wider-radius case could still bind it — the same reasoning `AutoRIFT._slabbable` records
for a mechanism measured to be a net loss at scene scale. No default is set, because nothing measured
justifies one.

**A non-interleaved sweep of these same arms read as a 25% speed win, and it was an artifact.** Ordered
`none, none, 2 GiB, 1 GiB` it gave 337.7, 329.5, 248.2, 259.0 s and looked like a clear result. Arm 1 paid
a cold page cache for the 11 GiB memory-mapped capture and arm 2 paid a precompile, because `src/` was
edited while arm 1 ran. `memory.md` states the rule this breaks twice over — *a benchmark whose arms are
not interleaved measures the order* — and four interleaved arms cost 22 minutes against a wrong default.

## Grid-sized constants are `Uniform` again

`_level_points` and `_worst_level_points` built their chip-size fields with `fill`, which is two grid-sized
`Matrix{Int}` per level — 72 MiB each on the Landsat grid, 86 on a NISAR one. The value is constant over
the set by construction, and `gridpoints` already carries `chip_size_x` as an `AutoRIFT.Uniform`, so `fill`
was also *densifying* a 24-byte field and forking `PointSet`'s type parameter, specialising the whole
downstream pipeline a second time.

Measured on the Landsat pair at block 1024 px, ten threads: **allocation 6.2 -> 5.9 GiB**, reproduced in
two processes, with the `dx` checksum unchanged and the same 534,930 points. Peak sits inside its own ±7%
scatter, so allocation is the column that carries it. `_coarse_points` still writes its own copy through
`fill!` and is unaffected: it takes `pts[rows, cols]`, and indexing a `Uniform` with ranges materialises an
ordinary `Array`.

## Bit-packing the mask planes is declined

Four of `BlockBuffers`' eighteen bytes per pixel are `Matrix{Bool}`, and a `BitMatrix` would hold the same
information in an eighth of that. On the `preprocess = :none` path only two of the four are touched, so the
saving is 2 -> 0.25 bytes per pixel: **0.44 GiB of NISAR L2's 7.53 above floor, 4.6% of peak.**

Declined on the trade rather than the arithmetic. It puts bit extraction in `_any_valid` and `_filtered!`,
both per-pixel over a chip footprint in the correlation's inner region, to recover 4.6% of a figure that
already has 40% headroom against the target. If a future configuration is buffer-bound — a wide halo with
many threads — this is the first thing to reach for, and the arithmetic above is the estimate to start from.

## The path to making a blocked run agree

The bisection and the population split determine the order, because the two halves interact with
*agreement* in opposite directions. Track A makes blocked match untiled and so can be judged by the
existing gates; track B deliberately changes both paths and so cannot. **A first**, or B's intent and A's
bug are indistinguishable in the residual.

### Step 0 — answered: it does block production, and it is not radar-only

**The reference measures the points a blocked run loses, and an untiled run reproduces it exactly.**
`tools/golden/divergence_fate.jl` reads `autoRIFT_intermediate.nc` — the reference correlator's own
`Dx`/`Dy` on the same grid as the capture, so no coordinate mapping is needed. On S1B at a 2048 px block:

| | |
|---|---|
| orientation check: untiled against the reference, points both measured | 506,544, median \|dx\| difference **0.0000 px**, p90 0.0007 |
| of the 26,781 points a blocked run loses, the reference measured | **25,970 (97.0%)** |
| the reference is nodata at | 811 (3.0%) |
| for scale, reference nodata over everything untiled measured | 1,096 of 507,640 (0.2%) |
| where the reference measured them, \|untiled − reference\| | median **0.000 px**, p90 0.034, max 1.499 |

So the lost set is enriched 15x in reference-nodata and still 97% real. **A blocked run is wrong at those
points, and an earlier reading of this file — that blocked was the more nearly correct path and untiled was
measuring something spurious — is refuted for all but 3% of them.** The orientation guard is what makes that
safe to say: the intermediate is `(x, y)` where the grid is `(y, x)`, and a transposed comparison of two
displacement fields still looks like one.

**And they reach the user.** The product carries 508,185 cells of `v` against the intermediate's 525,474
measured points — **96.7%** — so the crop removes 3.3% and the 5.1% a blocked run loses very largely
survives into the product. Measured as an aggregate rather than by mapping each point, deliberately: the
crop offset would have to be recovered from `window_location.tif`'s geotransform, and a mapping error
fabricates a defect at a location unrelated to the code, which `tools/ab/README.md` records costing a
session.

**It is not radar-only.** `tools/golden/block_gate.jl` fails on every real captured grid tried:

| case | grid | untiled | lost | gained | differing |
|---|---|---:|---:|---:|---:|
| S1B `1SDH_20180809`, whole grid | 3008x2504 | 507,640 | 26,781 (5.3%) | 771 | 25,206 |
| LC08 x LE07 `060018_20130330`, whole grid | 1760x2076 | 698,751 | 2,606 (0.37%) | 24 | 7,259 |
| S2B `MSIL1C_20200612`, whole grid | 1008x1008 | 614,195 | 1,528 (0.25%) | 64 | 4,117 |

An optical ITS_LIVE grid is a captured *geogrid* too — per-point search limits from a raster, with fill —
so it has the same sparse-radius structure, just less of it. Only a grid `gridpoints` builds is clean, which
is precisely why `test/tile.jl` never saw this.

**Consequence for the 16 GiB result.** The peak figures stand, but the four Sentinel-1 cases and both NISAR
granules cannot be *shipped* blocked until step A lands, and the optical cases carry 0.25-0.37% of the same
defect today.

### Step A — partly done: the two populations now have a window each, and it is not enough

**The measured split, which corrects the 42/58 figure this file previously carried.** Splitting the lost
set on whether the point has a real coordinate (`divergence_fate.jl`, S1B at 2048 px, whole grid):

| population | share of lost | reference measured them | \|untiled - reference\| median |
|---|---:|---:|---:|
| placeholder coordinate | 6,077 (22.7%) | 6,015 (99.0%) | 0.000 px |
| real coordinate | 20,704 (77.3%) | 19,955 (96.4%) | 0.001 px |

So the placeholder half is the *minority*, and the reference correlates at a placeholder coordinate too —
99% of them — which is why they are matched rather than declined. `CORRECTNESS.md` carries the encoding.

**What was built.** `PointSet` gained a `positioned` field (`Uniform(true, ...)` by default, so 24 bytes on
every grid but a geogrid and 0.9 MiB on the 7.5M-point NISAR one). `block_layout` spans the searchable set
*grown by* `_widening_reach` — `_level_decimation` max times the `_sparse_filter_width` margin, 32 grid
points on S1B — and takes that span twice, once per population, so a `Block` carries a second read window
for the placeholder points. `_run_one_block!` runs each population against its own window and merges.

Cost, measured before implementing (`reach.jl`): the dilated set is 1.18x the searchable count, per-block
window area grows by a median of 1.000x and a p90 of 1.012x, and **the largest window is unchanged at
4,521,000 px** — so `BlockBuffers` do not grow and peak memory does not move. 410 of 2187 blocks need a
placeholder window, at a constant 176,988 px, 4.2% of a block's own. The constancy confirms every
placeholder point shares one coordinate.

**Effect on the gate, S1B whole grid at 2048 px:**

| | before | after |
|---|---:|---:|
| lost | 26,781 (5.3%) | 22,764 (4.5%) |
| placeholder points blocked recovers | 35 of 6,112 (0.6%) | 2,765 of 6,112 (45%) |
| positioned points lost | 20,704 | 18,734 |

The mechanism works and the defect is not closed.

### Step A-2 — the remainder is `_cell_means`, and two windows cannot cover it

`_decimate_level` does **not** take a subset of the caller's grid. `_cell_means` moves each coarse node to
the *mean coordinate over its cell*, and `src/multichip.jl` already records that the nodata fill is
averaged in rather than excluded, because `cv2.resize` has no concept of it. So a cell straddling the
footprint edge produces a node at a coordinate blended between the real positions and the placeholder
constant — in **neither** population's span, and reachable by no pair of windows: the blend is a continuum
between the footprint and the scene corner, and a 50/50 cell lands mid-scene.

That also explains why the split does not classify them. A node's `positioned` comes from `grid[rows, cols]`
— the cell's *first* point — while its coordinate is the cell's mean, so a blended node can be marked
either way. It is why both populations still disagree (44,055 positioned and 5,935 placeholder points) and
why the read-window guard still fires, at a window of 258x686 for a point reaching to row 273.

### Step A-3 — done: windows are derived per pass and tiled to fit the buffers

**Chosen: match the reference now, and land `CORRECTNESS.md` item 2 once the golden set is frozen.** That
ordering is the cheap one, and the reason is recorded in that item: excluding the fill from the cell mean
returns every node to one of two places, so the clustering finds two windows and the generality goes quiet.
Nothing has to be unwound, because the layout is derived from the point set each pass runs and so holds
whatever `_cell_mean` returns — **implementing item 2 will touch `_cell_mean` and not the blocking code.**

`Block` went back to four fields; `read_rows`/`read_cols` now only size the buffers. `_read_windows`
derives a block's windows from the point set a pass is about to correlate.

**Tiled, not clustered.** A tile is an interval of the block's own coordinate span at
`buffer - 2 * halo`, so distant coordinates land in different tiles and near ones share a tile with no
linkage rule to choose or tune — and every window fits the buffers by construction.

**That last part is what keeps the memory target.** Left unconstrained, the largest cluster window on
NISAR L2 is **145,253,320 px against 30,024,720 today — 4.84x** (S1B is milder at 1.38x). Sizing the
buffers to that on a case already at 10.50 GiB would have broken 16 GiB outright. Tiling holds peak flat
and pays in sub-passes instead.

**What it costs: nothing — it pays.** Measured on NISAR L2 at the block size its recorded peak was set at,
same harness and same configuration as that row:

| NISAR L2 at 2304x1152 | recorded | with this change |
|---|---:|---:|
| peak | 10.50 GiB | **8.04 GiB** |
| wall clock | 368.1 s | **347.0 s** |
| points measured | 1,634,278 | **1,760,838** |
| read amplification | — | 3.41x |

Lower peak, faster, and 126,560 more points — the points a blocked run was losing. The reason it is not a
cost is that a window now tracks the point set *that pass* correlates, where `read_rows`/`read_cols` are
the layout's worst case over every pass: after the coarse mask has narrowed a level's searchable set, the
fine pass reads a window sized to what is left rather than to what the block might have needed.

An earlier estimate in this file of 1.39x read volume was taken from the *unconstrained cluster* geometry,
before tiling and before clustering-first, and does not describe the shipped version.

Windows per block stay small: at the base level, which dominates the work, one window almost everywhere.

### The case nearest the limit, blocked for the first time

`case_peaks.jl` put `S1A_IW_SLC__1SSV_20240618T025528` at **15.15 GiB untiled with no blocked row ever
recorded** — 0.85 GiB under a 16 GiB instance, and the case most likely to fall over on one. Measured with
`mem_nisar.jl`, whose floor holds the capture resident where the recorded row's does not, so read the
column against its own untiled arm rather than against 15.15:

| block | peak | above floor | runtime | agrees with untiled |
|---|---:|---:|---:|---|
| untiled | 21.92 GiB | 17.60 | 49.5 s | — |
| 1024 px | 14.55 GiB | 0.55 | 33.2 s | yes, `dx`/`dy` identical |
| 2048 px | **6.90 GiB** | 1.40 | **30.2 s** | yes, `dx`/`dy` identical |

A third of the untiled peak and 1.6x faster, at 1,570,046 points either way. The case had never been
blocked because nothing had measured it, not because it could not be — the same gap
`tools/golden/case_peaks.jl` was written to expose.

Still unmeasured on this branch: NISAR L1 (recorded 12.17 GiB at 2816x1536) and
`S1A_IW_SLC__1SSH_20150828` (12.79 GiB untiled, also never blocked).

`halo` also gained `_widened_prior_reach`. `_pass_geometry` reduces over searchable points only, so a
point the widening resurrects never contributed its *prior* to the halo — and a Sentinel-1 prior is ~18 px
of range.

`positioned` is retained rather than removed. Tiling by coordinate no longer needs the distinction at run
time, but `block_layout` still does: spanning a placeholder coordinate would make the buffer bound the
whole scene, and sizing from the block's nominal extent instead costs 1.9x on S1B.

**Gate at `--stride 1`, every captured grid:**

| case | before | after |
|---|---|---|
| S1B `1SDH_20180809` | 26,781 lost, 771 gained, 25,206 differing | **PASS** — `dx`/`dy` identical |
| LC08 x LE07 | 2,606 lost, 24 gained, 7,259 differing | **PASS** — `dx`/`dy` identical |
| S2B `MSIL1C_20200612` | 1,528 lost, 64 gained, 4,117 differing | **PASS** — `dx`/`dy` identical |
| NISAR L2 at 2304 px | not gated | **PASS** — `dx`/`dy` identical |

Every case reproduces every placeholder point exactly — 6,112 on S1B, 543 on LC08, 348 on S2B, 16,633 on
NISAR L2 — which is the population step 0 established the reference measures and the product carries.

**One figure in the recorded history is not comparable and should not be read as a regression.** The
`untiled` row for NISAR L2 reports 1,781,775 measured against the 1,760,838 an untiled run produces today.
That row predates this branch and `block_gate.jl` counts `isfinite(dx)` where the profiler rows count their
own way, so the two are not the same quantity. What *is* verified is that blocked and untiled agree exactly
when both are run by the same process on the same code, which is what the gate asserts.

**Two things had to be right beyond deriving windows per pass, and both were found by a real granule rather
than by the suite.**

**Cluster before tiling, not instead of it.** Tiling every spread block at `buffer - 2 * halo` shatters a
*compact* cluster wherever the halo is a large fraction of the block. On S2B the halo is 281x267 against a
650x618 block, so the tile is 56x116 and a block needing one window got dozens — which left 8 points lost
and 85 differing, all on the positioned side, with `_block_window_shortfall` **silent** throughout. The
layout was covering every point's reach; the windows were merely far smaller than they needed to be, which
puts many more points near a window edge. Clustering first took that case to zero.

**Test the window, not the span.** A window is the span grown by the halo and snapped outward with
`floor`/`ceil`, so it can be two pixels wider than the span implies. Comparing the raw span against
`buffer - 2 * halo` let a 3001-column window through against a 3000-column buffer, which surfaced as a
`BoundsError` raised on a worker task several frames from anything naming a window. `_read_windows` now
sizes the tile quantum with that slack and raises a readable error if a window ever exceeds the buffers,
because the buffers are the memory bound and a silent overrun there is the failure that bound exists to
prevent.

**A synthetic reproducer now exists**, so this no longer needs a granule: `test/tile.jl`'s "a blocked run
equals an untiled one on a geogrid-shaped grid" builds a `PointSet` with a sparse radius field and a
diagonal footprint whose edge straddles the decimation cells, and fails in 4.6 s where the S1B gate takes
tens of minutes. Its non-vacuity checks and its placeholder-window assertions pass; only the field
comparison fails.

### Step B — decline a chip that is mostly nodata (`CORRECTNESS.md` item 1)

The 58% at fill coordinates. The fix is that file's fractional validity test — `_any_valid` becoming a
*fraction* over the chip footprint with a per-level threshold — which declines them in **both** paths, so
untiled and blocked agree by construction rather than by reading more imagery.

Three things it needs beyond the code, all of which that file already specifies:

- **A threshold, measured.** Item 1 has the evidence to start from: at chip 768 on NISAR L1, excluding one
  ring of boundary nodes takes the `dx` residual from rms 2.31 px to 0.35.
- **A measurement that is not agreement.** It reduces agreement with production output by construction, so
  only independent ground truth or an internal-consistency argument can say whether it helped.
- **Not piecemeal.** Item 1 subsumes items 2 and 3 of that file; landing it alone while those are open makes
  their comparisons unreadable.

After B the divergence from this cause is zero, because both paths decline the same points.

### Step C — done: `tools/golden/block_gate.jl`

Asserts blocked-equals-untiled on a real captured grid and exits non-zero when it is not, so it can be run
over a set and believed. `--stride 4` thins to a scattered sixteenth for speed.

It lives beside the other golden harnesses rather than in `test/`, because the capture reader is
`tools/golden/correlator.jl` and pulling that into the suite would drag the whole golden stack behind it.
`test/realdata.jl`'s `has_realdata()` guard is the pattern to follow if that ever changes.

**It fails on every real case today**, which is the finding rather than a broken harness — the table in step
0 is its output. Until step A lands it is a reproducer; after A it is the gate. The positive control is
`test/tile.jl`, which asserts the same property on synthetic grids and passes.

**A thinned run is a detector, not a measure.** NISAR L2 at `--stride 4` loses 65% of its points where the
whole grid loses 5.3%, because thinning zeroes most radii while `_cell_max_radius!` still widens over its
full nine-point window — so a far larger share of coarse points inherit a radius they did not have. Use
`--stride 1` for a figure to quote.

## Order of work

1. ~~**Remove the three redundant copies.**~~ **Done**, in `src/preprocess.jl` and `src/tile.jl`.
   Measured on NISAR L2: **30% off the wall clock, occupancy 75% -> 95%, CPU seconds -11%, allocation
   7.07x lower, peak 8.8% lower**, at an identical point count; and 7.8x the allocation on the Landsat
   pair, bit-identical. `test/tile.jl` passes in full — including the new `:none` testset on NaN-bearing
   float input, which is the case that decides the form the change has to take — and Aqua reports no new
   ambiguity, which is the check the `StridedMatrix` method needed against the DiskArrays extension.
2. ~~**Add a `preprocess = :none` arm to `test/tile.jl`'s blocked-equals-untiled set.**~~ **Done** — "a
   blocked run equals an untiled one at preprocess = :none", 44 tests, including a NaN-bearing float pair
   and verified to fail when the substitution is removed.
3. **Decided: do not add a fill-point skip to the coarse pass. Split the population and treat the halves
   separately.** The reference *does* correlate these points — `arImgDisp_u`/`_s` pad the whole image by
   `Px = max(ChipSize)/2 + max(SearchLimit + |Dx0|) + 2` and shift the grid by `Px + 0.5`, so the chip lands
   inside the padded array — and reproducing the reference is the current objective, so an ad-hoc skip would
   be a divergence rather than a fix. The population is also not one thing: of 1,130 inherited coarse points
   at S1B's base level, at 477 distinct coordinates, **58% carry the grid's fill coordinate** and 42% carry a
   real one with merely a zero radius.
   - **3a, the fill half — `dev/CORRECTNESS.md` item 1, "never correlate a chip that is mostly nodata".**
     Already registered as a deliberately reproduced defect, and its specified fix — a *fractional* validity
     test over the chip footprint instead of `_any_valid`'s any-pixel test — declines exactly this
     population, in **both** paths, so untiled and blocked agree by construction rather than by widening a
     window. That file's rule is not to implement correctness debt while cases are red; all twenty-two are
     green, so it is now the work list. What this session adds is a second, independent driver: a blocked run
     cannot reproduce the defect at all, so item 1 is the unblocker for the 16 GiB target and not only the
     highest-value accuracy change.
   - **3b, the real-coordinate half.** Legitimate points the reference measures at a real position, which a
     fractional validity test keeps and should. `_searchable_span` reduces over each point's own radius, so a
     block's window is short of them by tens to hundreds of pixels — an ordinary layout fix, independent of
     the reference question, and the 29% the reverted span attempt recovered. Size the pad from the coarse
     set's own coordinates rather than from a grid-index reach converted to pixels, which is what made that
     attempt cost 6.3x the read window.
4. ~~**Add the read-window guard.**~~ **Done**, in `src/tile.jl` and `test/tile.jl`. It fires on every
   geogrid case, so it warns once per session rather than throwing — see above — and it separated the two
   populations of inherited coarse point that item 3 has to treat differently.
5. ~~**Measure the three remaining over-16 Sentinel-1 cases blocked.**~~ **Done** — 3.50, 4.75 and 3.95 GiB
   at 1024 px, all faster than untiled, closing the 16 GiB target at 22 of 22. Two S1 cases remain
   untiled-only and would drop from 15.15 and 12.79 GiB to roughly 4 if blocked.
6. **Re-run the recorded NISAR sweep once this tree settles.** Its rows and the rows measured here come
   from trees that disagree by 147,099 points on L2 and 2,349 on L1, so neither peak column can be read
   against the other.
7. **Size the read window per pass — but not for peak.** Per-level windows cut the *bytes copied* on the
   finer levels, 27.9 Mpx against 37.6 at the base level of L1, which is runtime and allocation. They do
   **not** cut peak, and an earlier version of this list claimed 1.6 GiB for them wrongly. Peak is the
   maximum over levels of `tasks x window`, and `block_buffers` is already sized to exactly that maximum
   and reused across every level — so sizing per level lowers the average and leaves the maximum where it
   was. `correlate_tiled`'s own comment states the premise: "no level changes" the largest read window.
8. **Split the pass by radius class.** The throttle half is built, measured neutral and off by default —
   see above — so what remains is the split, and its value is now unclear: the term it was meant to shrink
   turned out not to be live workspaces. Re-derive the accounting before building it. Neither half
   works alone, which is why they were separate items and should not have been. Splitting by class leaves
   the top class needing 36.94 Mpx per task at the coarsest level — 6.19 GiB over ten tasks, no better
   than today — because that class is what sets the halo. Throttling without splitting throttles every
   point, including the 99% whose radius is 64 or less. Together: the common classes run at full width on
   a 1.84-4.52 Mpx window, and the top class runs on two or three tasks rather than ten, which costs
   wall clock in proportion to its share of the *work* rather than its share of the memory.
   Answer-preserving on the power-of-two ladder; needs the block output assembly to write where a class
   searched rather than over a whole rectangle, and a `restrict`-style derivation so a class's partition
   composes with a level's decimation. A byte budget rather than a task count for the throttle, so it
   composes with `WORKSPACE_POOL_BYTES`.
9. ~~**Bit-pack the four mask planes in `BlockBuffers`.**~~ **Declined** at 0.44 GiB for bit extraction in
   the mask loop — see above. The arithmetic is recorded for a future buffer-bound configuration.
10. ~~**Return the grid-sized constants to `AutoRIFT.Uniform`.**~~ **Done** — allocation 6.2 -> 5.9 GiB on
   the Landsat pair, bit-identical.

## Strategies from other image-processing systems, and which ones are missing here

| mechanism | what it does there | the analogue here |
|---|---|---|
| `GDAL_CACHEMAX`, default **5% of usable physical RAM**; `GDAL_MAX_DATASET_POOL_RAM_USAGE`, **25% of RAM minus cachemax** | every consumer has a declared byte budget, stated as a fraction of the machine, and the budgets *compose* so the total is bounded. Exceeding one degrades throughput — a flush and a re-read — and never fails | `cache_budget = :auto` caps the prefetch at an eighth of free memory and `WORKSPACE_POOL_BYTES` bounds retention at 2 GiB. **The two largest consumers have no budget at all**: `BlockBuffers` x tasks and live workspaces x tasks. Nothing composes the three |
| `GDALWarpOperation::ChunkAndWarpImage` — estimate a chunk's memory, recursively bisect until it fits `WARP_MEMORY_LIMIT` | block size is an *output* of a memory budget, not an input, and it adapts to local cost | `process_block_size` is uniform and chosen up front. **Missing, and it is also the fix for L1's worst row**: 47% of that grid's work is in one block at 16384 px, which is why occupancy there is 2.33 of 10 |
| ITK `RequestedRegion` propagation, `StreamingImageFilter` | each stage declares the input region it needs and the pipeline propagates it backwards, so nothing reads more than its consumer asked for | `halo` takes one maximum over every level and point and `restrict` passes read windows through unchanged. **Missing — item 8 above is this idea** |
| block-aligned IO via `GetBlockSize` | never split a stored chunk across two readers, since for a compressed file the decode is the read | `_slab_height` rounds up to whole chunk rows. **Done**, and measured at 987 -> 278 ms on an 8321² DEFLATE pair |
| per-worker arenas — `cv::AutoBuffer`, a reused `cv::Mat` scratch | peak is the arena, so it is predictable and budgetable rather than a function of allocation rate | `BlockBuffers` and `WORKSPACE_POOL`. **Done for imagery and workspaces, absent for the `_prepare` output and the read temporary** — which is item 1 |
| bit-packed masks | a validity plane costs an eighth of a byte per pixel | `PaddedMask` pads a `BitMatrix` lazily and saved 6.36 GiB untiled. `BlockBuffers` holds four dense `Matrix{Bool}`. **Half done** — item 9 |
| streaming a filter over a rolling band | scratch stays in cache and never reaches full size | `windowmean` processes 16 rows at a time, faster and smaller at once. **Done inside the filter**, not around it: a block's raw window is fully resident before filtering begins |
| overviews and pyramids | read at the resolution the stage needs | **Not applicable.** The grid is decimated and the imagery never is; there is no image pyramid in the algorithm, and a coarse level correlates a large chip on full-resolution pixels |
| memory-aware task ordering, as in Dask | schedule to minimise live intermediates | **Already free.** Blocks write disjoint slices, so block order is not an input |
| `MALLOC_ARENA_MAX`, jemalloc tuning | bound per-thread allocator arenas, which on glibc can dominate RSS in a threaded process | **Untested, and it should be.** Every figure in this file and in `memory.md` is macOS; the target is Linux, where the allocator and the cgroup accounting both differ |

## Two instrument notes

**`mem_blocks.jl`'s 480 MiB profile buffer does not inflate its peaks.** `tools/ab/mem_child.jl` sets
`PROFILE_SLOTS = 60_000_000`, 480 MiB of `UInt64`, which looks like a constant added to every row. It
is not: the buffer is `undef` and only the portion a run records is ever touched, so it contributes
footprint in proportion to the samples taken. The check is that the recorded 2140 MiB at block 1024 px
reproduces at 2.127 and 2.147 GiB through the unprofiled `mem_churn.jl`. The block-size table's
absolute column stands.

**Single-run peak scatter on this scene is about ±7%.** The both-changes arm of the copy-removal A/B
gave 1.725, 1.856 and 1.965 GiB in three processes, at allocation and wall clock agreeing to 2%. No
conclusion above rests on a peak difference smaller than that, which is why the copy removal is argued
from its allocation and live-peak columns rather than from its peak.

## The optimal block size per case, measured — and where a blocked run still differs

`tools/golden/block_optimum.jl` sweeps each case's ladder at 12 threads and keeps only the arms whose
`dx`/`dy` are identical to that case's untiled run, so a reported row is answer-preserving by
construction. `--report` prints the table; the rows are in `mem/block_optimum.jls`.

**Read peak above the floor, not the peak.** `mem_nisar.jl` holds the capture's imagery resident, so its
floor runs from 3.1 GiB on a Landsat case to 17.1 on NISAR L2, and the total tells you about the harness
rather than the run. Above-floor, blocking wins on every one of the 22:

| case | best block | above floor | untiled above floor |
|---|---|---:|---:|
| NISAR L1 | 6144 | 9.37 GiB | 30.38 |
| NISAR L2 | 2240x1152 | 4.09 | 50.17 |
| S1B `1SDH_20180809` | 1408x512 | 0.61 | 21.18 |
| S1A `1SSH_20170221` | 1280x512 | 0.68 | 21.64 |
| the ten Landsat cases | 384-768 | 0.01-0.83 | 0.91-5.43 |

With a production floor near 2 GiB every case is well inside 16 GiB, NISAR L1 worst at about 11.4.

**The optimum is interior in both directions.** Too small and churn dominates — the S1B arms run 3.24 GiB
at 768x320 with a read amplification of 7.23x against 3.01 GiB at 1024 with 2.88x. Too large and the
buffers do: 10.32 GiB at 2048. So no endpoint rule works, the halo's own shape is not the answer, and the
ratio of chosen block to halo runs from 2.28x to 6.98x across the set — no single multiplier fits.

**A default cannot yet be fitted to this, and the reason is a correctness residual rather than a
modelling difficulty.** Scored against the measured curves, `2 x halo` per axis floored at 1024 is the
best rule available: mean +0.19 GiB and worst +0.65 GiB above each case's own optimum at 1.02x the
runtime, where every alternative tried is +0.76 GiB or worse. On NISAR L1 it picks `5504x3072` — and that
is one of the sizes at which a blocked run does **not** reproduce an untiled one.

**Twelve arms across four cases fail the agreement check**, and they cluster at small blocks: S2A at 192,
384 and 768; S2B at 320, 512 and 960; `S1A_IW_SLC__1SSH_20150828` at 384x192; and NISAR L1 at every arm
but 6144. The counts are 1 to 28 points out of 0.6-1.8 million, and `_block_window_shortfall` stays
silent throughout — so it is not a window shortfall and the layout is covering every point's reach.

It matches the padded-versus-unpadded transform switch `src/multichip.jl` already records: a point
outside the unpadded window takes a different transform whose "peak height differs in the last bits —
enough that a blocked run stops matching an untiled one exactly". A smaller block puts more points near a
window edge, which is exactly the observed gradient. On S2B at 512 px it is 3 points of 843,539, one of
them gained rather than lost.

So `tools/golden/block_gate.jl` passing is weaker evidence than it looked: it was run at one block size
per case, and those happened to be sizes that agree. The gate should sweep sizes, and the residual has to
be understood before any default is chosen — a default that picks a wrong-answer block size is worse than
the badly-calibrated one it would replace.

## Against the Python reference, end to end

`tools/ab/golden_python.py` runs `runAutorift` on the case's **own captured inputs** — the reference's
`xGrid`, search limits and priors as it had them — so neither side re-derives the grid. One process per
case, because `ru_maxrss` is a high-water mark. Both sides at 12 threads; OpenCV is built on GCD here so
`cv2.setNumThreads` is a no-op and the reference uses every core regardless, which is the comparison
wanted. `tools/golden/e2e_table.jl` joins the two.

**22 of 22 cases: median 9x faster, worst 1x, best 34x.**

| case | Julia | Python | speedup |
|---|---:|---:|---:|
| LT05 `L1GS_001013` | 1.1 s | 37.8 s | **34x** |
| LC09 `215109` | 4.5 | 77.8 | 17x |
| LT04 `063018` | 3.8 | 60.2 | 16x |
| S1B `1SDH_20180809` | 15.7 | 111.1 | 7x |
| NISAR L2 | 241.4 | 516.4 | 2x |
| NISAR L1 | 773.3 | 812.2 | **1x** |

The advantage is smallest exactly where it matters most. NISAR L1 is a dead heat, and its own untiled run
takes 569.7 s — so the 6144 block, the only size that agrees, costs 1.36x the runtime of not blocking at
all. Blocking that granule buys 30.38 -> 9.37 GiB above floor and pays 203 s for it.

**The two peak columns are not the same measurement** and the table says so: Julia's is a sampled
resident footprint against the harness's settled floor, Python's is whole-process `ru_maxrss` including
the interpreter and the capture arrays. Runtimes are directly comparable; treat the peak ratio as
indicative.

Point counts agree to 0.875-1.000 of the reference's, which is the pre-existing agreement gap `GATES.md`
tracks rather than anything this branch changed.

### Why NISAR L1 is only level with the reference, and it is not the threading

L1 is the one case where AutoRIFT.jl shows no advantage — 773.3 s against 812.2 s — where the median
across the set is 9x. Two separate things account for it, and only one is a defect.

**Per-point work on L1 really is about twice L2's.** The search-radius field runs to a p99 of 1033x582
against L2's 1052x288, so the y extent is doubled and the search *area* with it. Untiled, L1 measures
3,160 points/s against L2's 6,300 — the ratio the radii predict. Nothing to fix here.

**The rest is load imbalance, forced by the block size.** Work scales with search area, and L1's radius
field is extremely skewed: median 34x20, p99 1033x582, max 1905x830, so the worst point costs about
2,300x the median one. A block is one task on one thread and cannot be subdivided, so a block holding
the skewed region bounds the wall clock however many threads there are. Per-block work at each block
size, with a greedy longest-processing-time schedule over 12 threads:

| block | blocks | non-empty | top block's share | largest / per-thread avg | effective threads |
|---|---:|---:|---:|---:|---:|
| 2752x1536 | 5916 | 2246 | 2.7% | 0.32x | **12.0** |
| 3072 | 2652 | 1029 | 5.8% | 0.70x | **12.0** |
| 4096 | 1482 | 596 | 9.8% | 1.17x | 10.2 |
| 5504x3072 | 1479 | 602 | 9.8% | 1.18x | 10.2 |
| **6144** | 676 | 280 | **19.8%** | **2.37x** | **5.1** |

At 6144 a single block is a fifth of the granule and 2.37x what a thread should carry, so twelve threads
deliver five. `_run_blocked` already claims blocks off a shared atomic counter, so this is not a
scheduling defect — the work *unit* is too coarse, and no scheduler can split one block.

**And 6144 is the only block size on L1 that reproduces an untiled run.** Every smaller arm fails the
agreement check above, so the correctness residual is what forces the coarse block that costs the
parallelism. The discarded profiled sweep measured 3072 at 557.5 s against 6144's 773.3 s, so roughly
1.4x is waiting behind that fix — enough to put L1 at about 1.5x the reference rather than level with it.

Worth stating plainly: **untiled L1 is faster than blocked L1** — 569.7 s against 773.3 — so on this
granule blocking currently buys 30.38 -> 9.37 GiB above floor and pays 36% of the wall clock for it.
Fixing the small-block residual is a memory *and* a speed lever, and it is the same one item.

Measured with `balance.jl`-style per-block work sums rather than by correlating; the numbers need no
imagery beyond the grid.
