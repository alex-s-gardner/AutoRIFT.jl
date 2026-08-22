# M12: tiled processing — an outer `autorift` wrapper

Status: **designed, not built.** Task #20.

## What the user asked for

> support for tiled processing as an outer wrapper to autorift ... accepting a large
> lazy-or-RAM array plus `process_block_size = (X, Y)`, computing its own halo from search
> radius / chip size / priors / filter width.

Two constraints stated separately and both binding: it must accept a **lazy** array (a
`Raster` backed by disk, not only one already in RAM), and tiles must be able to run **in
parallel**. An earlier attempt to treat "where does the laziness live" and "should it be
parallel" as separate design questions was rejected as confusing — they are one design.

## Why this is not just a loop over `autorift`

The correlation at a point reads a *neighbourhood*, so a tile cut without overlap produces
wrong answers along every internal edge rather than merely missing them. Three separate
mechanisms reach outside a tile, and the halo is the max over all of them:

| mechanism | reach | where it is already computed |
|---|---|---|
| chip half-extent + search radius + prior | `chip/2 + radius + ceil(abs(prior))` per point | `_pass_geometry`, `src/track.jl:176-197` |
| preprocessing window | `filter_width ÷ 2` | `filter_width(::PreprocessMethod)`, `src/types.jl` |
| outlier filter window | `outlier_window ÷ 2`, over *grid points* not pixels | `GardnerFilter`, `src/outliers.jl` |

**`_pass_geometry` already returns exactly the first one** — `(px + 2, py + 2)`, where the
`+2` is the reference's slack for the half-pixel grid offset and index truncation. So the
halo computation is not new code; it is that function plus two window widths. Do not
re-derive it.

The third row is the awkward one and needs a decision: the outlier filter's window is in
**grid-point** units, so its pixel reach is `(outlier_window ÷ 2) * grid_spacing`, which at
the default 5 and a spacing of 32 is 64 px — comparable to the chip term. Options are to
inflate the halo by it, or to run the outlier filter *after* the mosaic on the assembled
field. The second is cheaper and arguably more correct (a filter that sees the whole field
rejects what a tile-local one cannot), but it changes when rejection happens relative to the
chip-size merge, which currently interleaves them (`_merge_level!`, `src/multichip.jl`).
**Decide this before writing code**; it determines whether tiles are independent.

## Shape

```julia
autorift(reference, secondary; process_block_size = (2048, 2048), kwargs...)
```

- `process_block_size` absent ⇒ today's behaviour exactly, on one block. The wrapper must be
  a no-op when not asked for, and the non-tiled result must stay bit-identical.
- Tiles are laid out over the **output grid**, then each tile's *read* window is its output
  extent grown by the halo and clipped to the image. So the halo affects what is read, never
  what is written — which is what makes the mosaic a simple copy with no blending, no
  averaging, and no seam logic.
- One `Cache` per tile, built with `threaded = false`, one tile per task. This is the pattern
  `src/api.jl:146-157` already documents for batch work: "one pair per task beats threading
  within a pair." Tiles are that same shape. Do **not** nest the existing intra-pass
  threading (`_track_threaded!`, `src/track.jl:431`) inside per-tile tasks.
- FFTW's planner is not thread-safe (`src/api.jl:351`) — plans must be warmed before the
  tile tasks spawn, exactly as `_warm_grid_plans` does now. All tiles share a geometry, so
  one warm-up serves every tile.

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
- Include a case where a tile boundary falls *through* a real displacement feature, since an
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
| `src/params.jl` | validate `process_block_size` (positive, and a sane lower bound relative to chip + radius, since a block smaller than one halo is all overlap) |
| `ext/AutoRIFTRastersExt.jl` | lazy `_read_block` |
| `test/tile.jl` | new — the equivalence gate above |
| `benchmark/memory.jl` | tiled peak-memory entry |

## Open questions to settle first

1. **Outlier filter: inflate the halo, or filter after the mosaic?** (see above — decides
   tile independence)
2. Does a tile with no searchable points short-circuit cleanly? `track!` already returns early
   when `chipx == 0` (`src/track.jl:118`), so probably yes — verify rather than assume.
3. Should `process_block_size` accept a single `Int` for square blocks? Every other size-like
   keyword in this package accepts either a scalar or a tuple; follow that.
