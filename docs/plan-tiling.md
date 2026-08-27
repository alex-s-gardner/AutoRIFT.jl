# Tiled processing — an outer `autorift` wrapper

Status: **built, gated against the untiled result, and reading windows from disk-backed input.**

`autorift(a, b; process_block_size = (X, Y))` reads and filters a block at a time, so no
scene-sized array is formed, and `test/tile.jl` asserts equality with an untiled run rather than
leaving it to have been checked once. A lazy `Raster` — or any `DiskArrays`-backed input — works
through the core's generic windowed read, with no dependency in `src/`.

`process_block_size` is in **pixels**; 512 by 512 is the measured best default. Blocking cuts total
peak memory 40% at 4096² and 60% at 6000², and the margin grows with the scene — see "Where this
stands", which corrects an earlier reading of these numbers that had it costing memory instead.

## What is wanted

> support for tiled processing as an outer wrapper to autorift ... accepting a large
> lazy-or-RAM array plus `process_block_size = (X, Y)`, computing its own halo from search
> radius / chip size / priors / filter width.

Two constraints, both binding and both parts of one design rather than separable questions: it
must accept a **lazy** array (a `Raster` backed by disk, not only one already in RAM), and
blocks must be able to run **in parallel**.

## Why this is not just a loop over `autorift`

The correlation at a point reads a *neighbourhood*, so a block cut without overlap produces
wrong answers along every internal edge rather than merely missing them. Several mechanisms
reach outside a block, and the halo is the **sum** of the ones that apply — not the maximum,
because each acts on the output of the one before it:

| mechanism | reach | where it is already computed |
|---|---|---|
| chip half-extent + search radius + prior | `chip/2 + radius + ceil(abs(prior))` per point | `_pass_geometry`, `src/track.jl:176-197` |
| preprocessing filter | `filter_reach(m)` — `width÷2`, but **twice that** for `Wallis` | `filter_reach`, `src/types.jl` |
| fine outlier filter + hole fill | 13 grid points, iterated | `reject_outliers`, `src/outliers.jl`; `_fill_holes!`, `src/multichip.jl` |
| coarse filter + dilation | 14 *coarse* cells = 56 grid points | `_coarse_mask`, `src/multichip.jl` |

**`_pass_geometry` already returns exactly the first one** — `(px + 2, py + 2)`, where the
`+2` is the reference's slack for the half-pixel grid offset and index truncation. So that
part of the halo is not new code. Do not re-derive it.

Why the preprocessing term **adds** rather than competing: `_windowmean_dense!` derives its
neighbour count from the *array* bounds (`src/window.jl:419`), so a pixel within
`filter_width ÷ 2` of the read-window edge is filtered over a truncated window where the
untiled run gave it a full one. Every pixel the correlator reads must have been filtered with
a full window, so the two reaches compose.

The outlier terms are the awkward ones, and they are far larger than a single window width
suggests:

- `reject_outliers` iterates. Stage 1 runs `iterations` times and stage 2 `iterations - 1`,
  and each iteration's `keep` depends on the previous iteration's NaN pattern, so influence
  travels `window ÷ 2` grid points *per pass*. At defaults that is 3×2 + 2×2 = 10, plus three
  `_fill_holes!` passes at `fill_window ÷ 2` = 3, giving 13 grid points.
- The **coarse** path dominates, because it runs on the strided grid where every unit is
  `coarse_stride` fine grid points. `relax` gives 3 passes × 2 = 6 coarse cells, and
  `dilate_within(keep, coarse_buffer = 8)` (`src/multichip.jl:284`) adds 8 more — so 14
  coarse cells, which at `coarse_stride = 4` is 56 fine grid points.

At `grid_spacing = 32` those terms would total **~1792 px**, the dilation alone (1024 px) exceeding
the whole fine-path figure — which at any useful block size is more overlap than data.

So the outlier terms are not in the halo. The coarse pass runs per block as *evidence only*, and the
gate, the dilation and the resample are taken once on the assembled coarse grid, which is ~1/1024 the
size of the imagery. The halo is therefore the correlation-plus-filter term alone: **`(32, 32)` px at
defaults**, costing 1–13% extra reading at block sizes between 512 and 2048 px. Read-to-write is
`((S + 2h)/S)²` for a square block of side `S`, which at `h = 32` is 1.13× for `S = 512`.

## Shape

```julia
autorift(reference, secondary; process_block_size = (512, 512), kwargs...)
```

- `process_block_size` is in **pixels**, and absent ⇒ one block over the whole scene. The wrapper
  must be a no-op when not asked for, and the non-tiled result must stay bit-identical.
- Blocks are laid out over the **output grid** — a block is a whole number of grid points, so a pixel
  request snaps outward — then each block's *read* window is its output
  extent grown by the halo and clipped to the image. So the halo affects what is read, never
  what is written — which is what makes the mosaic a simple copy with no blending, no
  averaging, and no seam logic.
- `min(nblocks, nthreads)` tasks claiming blocks from a shared counter, each with `threaded = false`
  inside it — the pattern `src/api.jl` already documents for batch work: "one pair per task beats
  threading within a pair." Do **not** nest the existing intra-pass threading (`_track_threaded!`)
  inside per-block tasks. Each task holds one `BlockBuffers`, allocated in a function of its own so
  the local cannot be hoisted into a `Core.Box` shared by every task.
- FFTW's planner is not thread-safe (`src/api.jl:351`) — plans must be warmed before the
  block tasks spawn, exactly as `_warm_grid_plans` does now.
- Every block must be given the **whole grid's** [`AutoRIFT.PassGeometry`](@ref) via `track!`'s
  `geometry` keyword. A pass sizes its workspace from its own largest chip and radius, and a
  workspace sizes its FFT buffers from its extents, so a block whose radii happen to be smaller
  would run a shorter transform than the untiled run ran over those same points. Measured on a
  512² grid with two radius regimes: without it, `correlation` differs on 46% of a sub-block's
  points; with it, none. One warm-up then serves every block, since they share the geometry.

## Lazy input

**Needs no extension method, and no `DiskArrays` dependency.** The seam is
`_read_block!(dest, img, rows, cols)`, and its generic implementation — `copyto!` over
`view(img, rows, cols)` — is already a chunk-aware windowed read for any array that supports one. A
lazy `Raster`'s `parent` is a `DiskArrays.AbstractDiskArray`, and measured through a
`readblock!`-counting array, a window spanning four chunks issues four calls and reads exactly the
requested elements: no over-read, no per-element degradation. The same holds for Zarr, NetCDF and
HDF5 backends, which arrive the same way.

In place because the driver reuses one buffer across blocks rather than taking a fresh array per
block. An allocating `_read_block` exists alongside it for callers that want one.

What made `process_block_size` fail to save memory was never the read. Three whole-scene operations
ran before any block was touched: the validity masks, the filtered pair from `_prepare`, and a driver
that then sliced that filtered scene. All three are gone. Filtering happens per block from raw input,
so the filtered scene is never formed — which also removed the double-filtering that made per-block
preparation wrong wherever it was reachable — and the default mask is now an
[`AutoRIFT.FiniteMask`](@ref), which computes `isfinite` on read.

The mask mattered more than it looks. `map(isfinite, img)` read every pixel of an input that may be
on disk and allocated a scene-sized `Matrix{Bool}` per image: 8 MiB at 2048², 32 at 4096², 68.7 at
Landsat's 6000². Constructing an `ImagePair` now touches neither image and allocates 160 bytes.
Materialization happens in `resident`, called from `_prepare` and nowhere else, which puts every
whole-array reduction over a mask — `all(mask)` in `_masked_boxmean!` and `_erode_mask!` — downstream
of a dense one. A blocked run never calls `_prepare`, so it never materializes; the untiled path pays
a single pass and measures the same end to end (26.5 ms against 26.1 with a supplied dense mask).

## Verification

The gate that matters: **a tiled run must equal a non-tiled run**, not approximately.

- `dx`/`dy` bit-identical across the whole field, at several `process_block_size` values
  including sizes that do not divide the image evenly and a size of 1 block (which must be
  the untiled path).
- Include a case where a block boundary falls *through* a real displacement feature, since an
  under-computed halo shows up only there. A boundary through flat texture will pass with a
  halo of zero.
- Compare `isequal` on every field, `correlation` included. A **same-process** comparison shares the
  process's FFTW wisdom and plan cache, so the ~3e-7 planner drift that flips a subpixel step
  between processes is not in play: in-process, an inexact match is a defect. That drift is real
  across processes, and comparing a fresh run against a capture taken earlier has produced false
  alarms three times — so compare within one process and the caveat disappears.
- Memory: `benchmark/memory.jl` records blocked peak RSS at two scene sizes and two block sizes,
  against a resident untiled run. The comparison must be against a *resident* baseline; measured
  against one that first materializes a windowed input, blocking flatters itself by the cost of a
  copy it never makes.

## Files

| File | Change | Done |
|---|---|---|
| `src/tile.jl` | block layout, halo, the tiled driver, `BlockBuffers` | yes |
| `src/api.jl` | `process_block_size` keyword; dispatch to the tiled path when present | yes |
| `src/multichip.jl` | split the level loop so per-block work produces evidence and the gate, dilation and resample happen once on the assembled coarse grid | yes |
| `src/types.jl` | `PassGeometry`; `filter_reach`; `PassRunner` | yes |
| `src/preprocess.jl` | `FiniteMask` and `resident`, so the default mask costs nothing until something reduces over it | yes |
| `src/preprocess.jl` | in-place `highpass!`, `_filtered!`, `_erode_mask!`, `_masked_boxmean!` | yes |
| `test/tile.jl` | layout, halo, `filter_reach`, and the equivalence gate | yes |
| `test/extensions.jl` | windowed reads from a `DiskArrays` input: no read exceeds a block's window, and the result equals the resident one | yes |
| `ext/AutoRIFTRastersExt.jl` | nothing. `_read_block!`'s generic method already reads a window from a disk-backed `parent`; an extension method would only re-implement it | n/a |
| `benchmark/memory.jl` | blocked peak-memory entries, as total process peak against a resident untiled run | yes |

Validation lives in `block_layout` rather than `src/params.jl`, because the halo is what a block
size has to be checked against and `block_layout` is what knows the halo. `process_block_size` is
likewise **not** a `Params` field: it is a scheduling choice that changes peak memory and nothing
else, and adding a 23rd positional field would touch every construction of `Params` including the
trimmed binary's.

## Settled

1. **The outlier filter runs after assembly, not per block.** Its reach compounds to ~1792 px
   (above), so a halo covering it would be mostly overlap at any useful block size. Nothing
   blocks moving it: `resample` is called exactly once in all of `src/` — on the coarse *mask*
   (`src/multichip.jl:287`) — and `_level_points` states priors are never written
   (`src/multichip.jl:207-209`), so there is no coarse-to-fine prior chain to break. The
   docstring at `src/multichip.jl:315` describes the *reference's* behaviour, not this port's.
   The only cross-level channel is the merge's `wanted` mask.
2. **`process_block_size` is a tuple, `(X, Y)`, and only a tuple.** A full-width band is
   expressed by passing the scene width as `X` — the cheaper shape at a given halo, since a
   band has halo on two sides rather than four, but the caller's choice to make rather than
   something inferred from a scalar.
3. **A block smaller than the halo is an error**, not a clip or a silent enlargement: it would
   be entirely overlap.

## Answered while building

- **A block with no searchable points costs no I/O.** `_run_one_block!` checks `nsearchable` before
  reading, so a block every finer level already resolved, or one the coarse mask emptied, returns
  immediately.
- **A block's grid is a clean sub-block.** `gridpoints` insets by a margin derived from chip and
  radius, *not* from image size (`src/points.jl:244`), so the grid origin is scene-independent —
  provided blocks slice the global grid with `pts[rows, cols]` (`src/points.jl:295`) and never call
  `gridpoints` themselves, which `_block_points` respects.
- **The small-coarse-grid fallback is one behaviour, warned about under tiling.** `_coarse_points`
  returns `nothing` when the coarse grid is smaller than the outlier filter's window, and both paths
  then search everything rather than rejecting on no evidence — a tiled run must agree with an
  untiled one point for point, so it cannot substitute a policy of its own. The tiled path warns,
  because the cost is what the coarse pass exists to avoid: ~100× the work, and silence reads as a
  fast run.
- **Block coordinates shift by an integer, and that is what makes them exact.** `chip_bounds` and
  `search_bounds` both `floor` a coordinate, and `floor(u - k) == floor(u) - k` for integer `k`, so
  a point lands on the same pixel of the block that it did on the scene.

## Verified, having been assumed

- **A block can reach `_zeropad`, and that is a cost rather than a defect.** Measured: with
  `chip_size_max = 128` and 8×8 grid-point blocks, some blocks' coarsest level fails `fits` and pads
  (`src/track.jl:146-151`), allocating full copies. Results stay bit-identical — the coordinate shift
  is integral and `okmask` pads to `false`, so padding an in-bounds point changes nothing it
  computes. The earlier note here assumed blocks never reach that branch; they do, and what follows
  is 1.2 ms and 10.6 MB per call, not a wrong answer.

## Where this stands

### Built, and verified bit-identical to an untiled run

| piece | where |
|---|---|
| `PassGeometry`, so a subset runs the transform the whole grid would have | `src/types.jl`, `src/track.jl` |
| `filter_reach`, the neighbourhood a filter's output depends on | `src/types.jl` |
| `Block`, `BlockLayout`, `block_layout`, `halo` | `src/tile.jl` |
| `correlate_tiled` and the four-step level: evidence per block, decisions once | `src/tile.jl`, `src/multichip.jl` |
| `BlockBuffers` — pooled raw reads and pooled `Highpass` filter storage | `src/tile.jl` |
| `process_block_size` on `autorift`, `autorift_with_grid`, `init`/`autorift!` | `src/api.jl` |
| in-place `highpass!`, `_filtered!`, `_erode_mask!`, `_masked_boxmean!` | `src/preprocess.jl` |

Identity is asserted by `test/tile.jl` on `dx`, `dy`, `correlation`, `chip_size` and `interpolated`
across: a scene whose halves move differently with a block boundary on the seam between them, a
caller-supplied grid built below `chip_size_max`, multiple chip-size levels, threaded, `Highpass` and
`Wallis`, block sizes that do not divide the grid evenly, and the small-coarse-grid fallback.

The gate is checked to have teeth rather than assumed to: shortening the halo by 20 px makes it fail
at 252 of 3721 points, and the boundary is confirmed to straddle the discontinuity — `dx` is −3.0 on
one side and +2.0 on the other.

### The measurements, and what they mean

**Total** process peak RSS, threaded. Blocked figures are from an input that materializes only the
window asked for; untiled is a resident scene. Block sizes are in pixels, and each is genuinely
several blocks.

| scene | untiled | 512 px | 1024 px | 2048 px |
|---|---:|---:|---:|---:|
| 4096² | 995.2 MiB | **593.4 MiB** | 613.5 MiB | 718.5 MiB |
| 6000² | 1519.0 MiB | **602.9 MiB** | 612.4 MiB | 742.7 MiB |

**Blocking wins at every size measured, and the margin grows with the scene** — 40% at 4096², 60% at
6000². So it is not only for a scene that will not fit; it is worth reaching for whenever peak memory
matters. 512-pixel blocks are the cheapest, and peak *rises* with the block size, because a larger
block holds more imagery at once while the halo it saves is a fixed width. Halo overhead at these
sizes is 1–13% extra reading.

**Report total peak, not each configuration's own above-baseline delta.** An untiled baseline already
holds the two resident scenes, so subtracting it hides exactly the cost blocking removes. Measured
that way the same runs read as 69.5 against 166.5 MiB at 2048² and support the opposite conclusion.

**Compare against a resident untiled baseline.** Measured against an untiled run that first
materializes a windowed input, blocking looks better than it is — that baseline pays for a copy the
blocked path never makes. An earlier round of this work reported a 35% reduction on exactly that
mistake.

**A block size in grid points is not a block size.** `process_block_size` counts pixels; a figure read
as grid points is `grid_spacing` times larger, so at the default spacing a "128" is 4096 pixels — one
block on any scene up to 4096², which is the untiled computation wearing a block label. Two of the
four rows in the earlier version of this table were exactly that, which is how a table with no block
size varying in it came to be read as a block-size trend.

Per-block allocation, at an 888-pixel read window, before and after pooling: raw read 7.56 MiB → 0,
prepare 15.36 MiB → 0.23 MiB. And per *task* rather than per block, at 144 blocks on 8 threads:
992 MiB → 55 MiB of buffer sets, worth 937 → 873 MiB of peak at 6000² with 512-pixel blocks.

**Read live heap, not RSS, when judging whether something is retained.** Live-heap growth is ~1 MiB
for every configuration measured here, tiled or not. The 300–430 MiB figures an earlier round of this
work reported were allocator slack from per-block churn, not a requirement — and reading them as a
requirement produced a wrong conclusion ("tiling makes memory worse") that took several rounds to
correct. `docs/memory.md` documents the same trap.

### One algorithm, one implementation

The chip-size level and the loop over levels each exist once. `correlate_multichip` and
`correlate_tiled` both build a [`AutoRIFT.PassRunner`](@ref) and hand it to `_multichip`, which runs
`chipsize_level` per level; what differs is three methods on the runner:

| method | `WholeScene` | `Blocked` |
|---|---|---|
| `run_pass` | one `track` call on the filtered scene | `_run_blocked` over its blocks, at the whole set's `pass_geometry` |
| `restrict` | itself — the subset is already the `PointSet` it is given | re-derives the partition via `_coarse_block_layout` |
| `_warn_coarse_fallback` | silent | warns |

The runner carries the pair, which is what makes the prepared-versus-raw distinction unrepresentable
rather than merely documented: `WholeScene` holds a filtered pair and `Blocked` an unfiltered one, so
no call can supply the other kind. That was the twice-filtered-image defect.

The runner is **positional**, and that is a constraint rather than a preference: a keyword carrying an
abstract type produces `unresolved call ... Core.kwcall` under `--trim`. `_run_blocked`'s `subpixel`
is positional for the same reason — it was a keyword annotated `SubpixelMethod`, which survived only
because the blocked path is unreachable from the trimmed entry point.

This removes step-sequence drift and the argument mismatch; it does not make bit-identity automatic.
The numerical risk lives in `run_pass(::Blocked)` and rests on the halo, the integral coordinate
shift, `PassGeometry` and per-block filtering. What it buys is that a failing identity test now means
*numerics in one method* rather than "numerics or a divergent step sequence".

### Three traps that cost time here

- **FFTW wisdom persists to disk.** The first run of changed code can write wisdom that changes what
  every *later* process measures, so comparing a fresh run against a capture taken before the change
  shows a difference that is not in the code. Re-run both sides until each reproduces against itself,
  *then* compare. This produced a false regression on 24% of `correlation` values.
- **`filter_width` is not a filter's reach.** `Wallis` chains two window passes and reaches twice its
  half-width; padding by `width ÷ 2` left 792 of 10201 interior points differing by up to 0.21.
  `filter_reach` exists for this and is pinned by a test that measures the true reach per filter.
- **Which pair a driver takes is part of its contract.** Per-block filtering filters what it is given,
  so handing it a pair `_prepare` had already filtered produced a twice-filtered image — matching a
  twice-filtered scene exactly and the correct one only to 0.09. It reads as an edge effect (the
  differences cluster near read edges, where a filter's output differs most) and is not one: the image
  is wrong everywhere and still looks like imagery. Diagnosing it as a halo problem cost several
  rounds. Bisect by substituting one stage at a time — comparing a block filtered from raw against the
  same block sliced from a filtered scene localizes it in one step — rather than reasoning about which
  pixels a window reaches.

### Loose ends found on the way, each worth its own change

- `Sobel`, `Laplacian`, `Decibel` and `WallisGapfill` are exported, documented as
  `PreprocessMethod`s, resolvable from a keyword symbol, and have `filter_width` methods — but no
  `preprocess` method for real input. All four throw `MethodError` from inside `_prepare` rather than
  failing at the keyword boundary where every other bad value is caught.
- `resample(grown, (nr, nc), Nearest())` is not aligned to the coarse lattice even untiled.
  `resample!(::Nearest)` uses `ys = sr/dr` with `sr = fld(nr, stride)` (`src/resample.jl:58-70`)
  while `_cell_max_radius!` uses `lo = stride÷2, hi = stride-1-lo` (`src/multichip.jl:292-294`). For
  `nr = 13, stride = 4` the radius cell for fine row 4 is rows 2..5 but the mask cell is 1..4, so
  `src/multichip.jl:36-37`'s claim that "a coarse grid point is exactly a block of fine ones" does
  not hold literally for the mask.
- `process_block_size` on an in-RAM array is a pessimization, measured so up to at least 4096². It is
  documented as such rather than warned about: a caller processing a scene that genuinely does not fit
  passes a lazy array and should not be told off for it, and the size at which blocking starts to pay
  depends on the instance rather than on anything the package knows.
