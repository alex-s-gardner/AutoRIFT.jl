# Tiled processing — an outer `autorift` wrapper

Status: **built and bit-identical for in-RAM input; the lazy path that motivates it is not built.**

`autorift(a, b; process_block_size = (X, Y))` works today and gives bit-identical results to an
untiled run. What is missing is the one configuration the feature exists for: a disk-backed input
where a block's read *replaces* a scene that never fits, rather than copying out of one that does.
See "Where this stands" at the end for what remains and what the measurements say.

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
| coarse filter + dilation | 14 *coarse* cells = 56 grid points | `_coarse_pass`, `src/multichip.jl:224-288` |

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

At `grid_spacing = 32` that totals **~1792 px**, and the dilation term alone (1024 px)
exceeds the whole fine-path figure. Overhead depends on the block size the caller asks for:
read-to-write is `((S + 2h)/S)²` for a square block of side `S`, so a 2048² block reads 7.6×
its own pixels at this halo and a 8192² block 2.1×. That does not make tiling impossible; it
makes it useless at the block sizes the feature exists to serve.

The way out is not a bigger halo but to stop making per-block decisions: run the coarse pass
per block as *evidence only*, then take the gate, the dilation and the resample once on the
assembled coarse grid, which is ~1/1024 the size of the imagery. Then the halo falls back to
the correlation-plus-filter term.

## Shape

```julia
autorift(reference, secondary; process_block_size = (2048, 2048), kwargs...)
```

- `process_block_size` absent ⇒ today's behaviour exactly, on one block. The wrapper must be
  a no-op when not asked for, and the non-tiled result must stay bit-identical.
- Blocks are laid out over the **output grid**, then each block's *read* window is its output
  extent grown by the halo and clipped to the image. So the halo affects what is read, never
  what is written — which is what makes the mosaic a simple copy with no blending, no
  averaging, and no seam logic.
- One block per task with `threaded = false` inside it — the pattern `src/api.jl:146-157` already
  documents for batch work: "one pair per task beats threading within a pair." Do **not** nest the
  existing intra-pass threading (`_track_threaded!`, `src/track.jl:431`) inside per-block tasks.
  (As built this is `_serial_params` plus one `BlockBuffers` per task, not one `Cache` per block:
  the buffers are what a block needs, and a `Cache` also carries a grid and a result.)
- FFTW's planner is not thread-safe (`src/api.jl:351`) — plans must be warmed before the
  block tasks spawn, exactly as `_warm_grid_plans` does now.
- Every block must be given the **whole grid's** [`AutoRIFT.PassGeometry`](@ref) via `track!`'s
  `geometry` keyword. A pass sizes its workspace from its own largest chip and radius, and a
  workspace sizes its FFT buffers from its extents, so a block whose radii happen to be smaller
  would run a shorter transform than the untiled run ran over those same points. Measured on a
  512² grid with two radius regimes: without it, `correlation` differs on 46% of a sub-block's
  points; with it, none. One warm-up then serves every block, since they share the geometry.

## Lazy input

Lives in the **Rasters extension**, not the core. The core wrapper asks for a block by index
range; the extension's method materializes it with a Rasters read. A `Raster` wrapping a disk
array reads only the requested window, so peak memory becomes `O(block + halo)` rather than
`O(scene)` — which is the entire point for the small-AWS-instance case that motivated
`docs/memory.md`.

The core must not gain a Rasters dependency to do this. The seam is `_read_block!(dest, img, rows,
cols)` — in place, because the driver reuses one buffer across blocks rather than taking a fresh
array per block. An allocating `_read_block` exists alongside it for callers that want one.

**This is the piece that is not built, and without it the feature has no use case:** blocking a
resident array copies out of memory that is already there, so it only pays when the block read
replaces a load that would not fit.

## Verification

The gate that matters: **a tiled run must equal a non-tiled run**, not approximately.

- `dx`/`dy` bit-identical across the whole field, at several `process_block_size` values
  including sizes that do not divide the image evenly and a size of 1 block (which must be
  the untiled path).
- Include a case where a block boundary falls *through* a real displacement feature, since an
  under-computed halo shows up only there. A boundary through flat texture will pass with a
  halo of zero.
- Compare `isequal` on `dx`/`dy` and hold FFTW wisdom fixed. Note the established caveat:
  `correlation` drifts ~3e-7 through the planner and can flip one subpixel step at 1024²
  (measured: 1 point of 729, `dy -4.0 → -3.984375`) — so **re-run the baseline against itself
  first** before believing any mismatch. That has produced false alarms three times now.
- Memory: `benchmark/memory.jl` should gain a tiled entry showing peak is bounded by block
  size and not by scene size. That is the claim the feature exists to make.

## Files

| File | Change | Done |
|---|---|---|
| `src/tile.jl` | block layout, halo, the tiled driver, `BlockBuffers` | yes |
| `src/api.jl` | `process_block_size` keyword; dispatch to the tiled path when present | yes |
| `src/multichip.jl` | split the level loop so per-block work produces evidence and the gate, dilation and resample happen once on the assembled coarse grid | yes |
| `src/types.jl` | `PassGeometry`; `filter_reach` | yes |
| `src/preprocess.jl` | in-place `highpass!`, `_filtered!`, `_erode_mask!`, `_masked_boxmean!` | yes |
| `test/tile.jl` | layout, halo and `filter_reach` coverage | partly — the equivalence gate is missing |
| `ext/AutoRIFTRastersExt.jl` | lazy windowed `_read_block!`. The extension materializes eagerly today — `parent(reference)` at `ext/AutoRIFTRastersExt.jl:96-99` hands the whole array to the core — so this is new code rather than a change to how `Raster` inputs are unwrapped | **no** |
| `benchmark/memory.jl` | tiled peak-memory entries | **no** |

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
- **The small-coarse-grid fallback is now an error under tiling.** `_coarse_points` returns
  `nothing` and lets its caller decide: the whole-scene path searches everything rather than
  rejecting on no evidence, and `_tiled_level` throws. Per block that fallback would silently skip
  the coarse restriction, which is ~100× the work with no diagnostic.
- **Block coordinates shift by an integer, and that is what makes them exact.** `chip_bounds` and
  `search_bounds` both `floor` a coordinate, and `floor(u - k) == floor(u) - k` for integer `k`, so
  a point lands on the same pixel of the block that it did on the scene.

## Still to verify rather than assume

- An interior block must never need `_zeropad`: it allocates full copies and shifts every
  coordinate (`src/track.jl:146-151`). Blocks are in bounds by construction as laid out, but nothing
  asserts it, so a caller-supplied grid could still reach that branch.

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

Identity was checked against `correlate_multichip` on `dx`, `dy`, `correlation`, `chip_size` and
`interpolated` across: plain, a decorrelated quadrant (so the coarse gate, outlier filter and hole
fill all engage), multiple chip-size levels, threaded, `Highpass` and `Wallis`, and block sizes that
do not divide the grid evenly. Also with `prepare_blocks = true`, i.e. filtering each block from raw
input rather than slicing a pre-filtered scene.

### The measurements, and what they mean

Peak RSS above a baseline process that allocates the inputs but does not correlate:

| scene | pair + masks resident | untiled | blocked |
|---|---:|---:|---:|
| 2048² | 40 MiB | 18 MiB | 48–70 MiB |
| 6000² (Landsat 8) | 274 MiB | 152 MiB | 142–157 MiB |
| 20000×40000 (NISAR SLC) | **13.4 GiB** | will not fit a small instance | ~0.3 GiB at 128² grid-point blocks, 9% halo overhead |

Two things this makes clear. **Blocking does not help a scene that already fits** — at 2048² the
untiled path costs 18 MiB and there is nothing to save. And **the halo is only affordable at scale**:
39.6% overhead at 32² grid-point blocks against 2.3% at 512², so large blocks are the operating
point, not small ones.

Per-block allocation, at an 888-pixel read window, before and after pooling: raw read 7.56 MiB → 0,
prepare 15.36 MiB → 0.23 MiB.

**Read live heap, not RSS, when judging whether something is retained.** Live-heap growth is ~1 MiB
for every configuration measured here, tiled or not. The 300–430 MiB figures an earlier round of this
work reported were allocator slack from per-block churn, not a requirement — and reading them as a
requirement produced a wrong conclusion ("tiling makes memory worse") that took several rounds to
correct. `docs/memory.md` documents the same trap.

### Not built, and load-bearing

1. **Lazy windowed read.** `_read_block!` copies out of a resident array, and the Rasters extension
   still materializes eagerly — `parent(reference)` at `ext/AutoRIFTRastersExt.jl:96-99` hands the
   whole array to the core. For NISAR there is no resident array to copy from, so this is the
   feature rather than an optimization. `_read_block!` is the seam; its method for a disk-backed
   `Raster` should read only the window.
2. **The equivalence gate as a committed test.** `test/tile.jl` covers the layout, the halo and
   `filter_reach`, but tiled-equals-untiled was verified by throwaway scripts and is not guarded.
   That test must include a block boundary falling *through* a displacement feature: a boundary in
   flat texture passes with a halo of zero and proves nothing.
3. **Benchmark entries.** `benchmark/memory.jl` should record the crossover above so a future change
   cannot quietly move it.

### Two traps that cost time here

- **FFTW wisdom persists to disk.** The first run of changed code can write wisdom that changes what
  every *later* process measures, so comparing a fresh run against a capture taken before the change
  shows a difference that is not in the code. Re-run both sides until each reproduces against itself,
  *then* compare. This produced a false regression on 24% of `correlation` values.
- **`filter_width` is not a filter's reach.** `Wallis` chains two window passes and reaches twice its
  half-width; padding by `width ÷ 2` left 792 of 10201 interior points differing by up to 0.21.
  `filter_reach` exists for this and is pinned by a test that measures the true reach per filter.

### Loose ends found on the way, each worth its own change

- `Sobel` and `Laplacian` are exported, documented as `PreprocessMethod`s, and have `filter_width`
  methods — but no `preprocess` method. Calling either throws `MethodError`.
- `resample(grown, (nr, nc), Nearest())` is not aligned to the coarse lattice even untiled.
  `resample!(::Nearest)` uses `ys = sr/dr` with `sr = fld(nr, stride)` (`src/resample.jl:58-70`)
  while `_cell_max_radius!` uses `lo = stride÷2, hi = stride-1-lo` (`src/multichip.jl:292-294`). For
  `nr = 13, stride = 4` the radius cell for fine row 4 is rows 2..5 but the mask cell is 1..4, so
  `src/multichip.jl:36-37`'s claim that "a coarse grid point is exactly a block of fine ones" does
  not hold literally for the mask.
- `process_block_size` on an in-RAM array is a pessimization at small scene sizes. Worth deciding
  whether it should warn, error, or simply be documented as disk-backed-only.
