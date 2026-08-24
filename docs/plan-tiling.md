# Tiled processing — an outer `autorift` wrapper

Status: **designed, not built.** The prerequisite is in place: `track!` accepts a
[`AutoRIFT.PassGeometry`](@ref) so a block can be correlated as the whole grid would have been.

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
| preprocessing window | `filter_width ÷ 2` | `filter_width(::PreprocessMethod)`, `src/types.jl` |
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
- One `Cache` per block, built with `threaded = false`, one block per task. This is the pattern
  `src/api.jl:146-157` already documents for batch work: "one pair per task beats threading
  within a pair." Blocks are that same shape. Do **not** nest the existing intra-pass
  threading (`_track_threaded!`, `src/track.jl:431`) inside per-block tasks.
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

The core must not gain a Rasters dependency to do this. The seam is a small internal function
(`_read_block(img, rows, cols)`) whose core method is a `view`/copy and whose extension method
is a lazy read.

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

| File | Change |
|---|---|
| `src/tile.jl` | new — block layout, halo from `_pass_geometry` + filter widths, mosaic |
| `src/api.jl` | `process_block_size` keyword; dispatch to the tiled path when present |
| `src/multichip.jl` | split the chip-size level loop so per-block work produces evidence and the gate, dilation and resample happen once on the assembled coarse grid |
| `src/params.jl` | validate `process_block_size` (positive in both axes, and at least one halo) |
| `ext/AutoRIFTRastersExt.jl` | lazy `_read_block`. The extension currently materializes eagerly — `parent(reference)` at `ext/AutoRIFTRastersExt.jl:96-99` hands the whole array to the core — so this is new code rather than a change to how `Raster` inputs are unwrapped |
| `test/tile.jl` | new — the equivalence gate above |
| `benchmark/memory.jl` | tiled peak-memory entry |

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

## Still to verify rather than assume

- Does a block with no searchable points short-circuit cleanly? `track!` returns early when the
  pass has nothing searchable (`src/track.jl:124`), so probably yes.
- `gridpoints` insets by a margin derived from chip and radius, *not* from image size
  (`src/points.jl:244`), so the grid origin is scene-independent and a block's grid is a clean
  sub-block — **provided** blocks slice the global grid with `pts[rows, cols]`
  (`src/points.jl:295`) and never call `gridpoints` themselves.
- `_coarse_pass` returns "search everywhere" when the coarse grid is narrower than the filter
  window (`src/multichip.jl:238`). Unreachable on a full scene, reachable per block, where it
  silently skips the coarse restriction — ~100× the work with no diagnostic. Must become an
  error under tiling.
- An interior block must never need `_zeropad`: it allocates full copies and shifts every
  coordinate (`src/track.jl:146-151`). Keep blocks in bounds by construction and error otherwise.
