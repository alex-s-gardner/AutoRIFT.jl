# Correctness debt: changes deferred until the golden set agrees

**Every item here is a known defect that AutoRIFT.jl reproduces on purpose.** Agreement with the
production Python is the current objective, and it is not the same objective as being correct. Where the
two conflict this project chooses agreement, because a deliberate difference and a bug look identical in a
comparison — so every difference has to be removed before the remaining ones mean anything.

That trade has a cost, and this file is the invoice. Each item below makes AutoRIFT.jl reproduce behaviour
there is measured reason to think is wrong.

**Read this before implementing any of it:**

- **Do not implement these while golden cases are still red.** Each one *reduces* agreement with current
  production output by construction, because each changes which points are measured or where. Fixing one
  early makes every other comparison unreadable.
- **The gate is `GATES.md`.** When every case there is green, this file becomes the work
  list.
- **Each item names its own evidence.** The measurement that establishes the defect is in `GATES.md` or
  `tools/golden/README.md`; none of these is a suspicion.
- **Landing one needs a new kind of test.** Agreement cannot judge a change that deliberately breaks
  agreement, and on most of these both implementations make the same choice — so only a measurement
  against independent ground truth, or an internal-consistency argument, can say whether the fix helped.
  `tools/golden/README.md` records what that measurement is for each item.

The per-item detail, with the numbers, lives in **`tools/golden/README.md` § "Matched for agreement, not
endorsed"** and in the open-items list above it. This file is the index, ordered by what to do first.

---

## 1. Mask the input: never correlate a chip that is mostly nodata

**The single highest-value change, and the cheapest.** A NISAR geogrid is a rotated radar footprint on a
map grid, so the valid data has a long diagonal boundary and the nodata fill is the *majority* of the
array — 56.8% on L1, 55.7% on L2. Nothing in either implementation keeps a correlation chip off that
boundary, and at a coarse level the chip is large: 768x416 px, so a chip centred one lattice cell (384 px)
inside the edge is still substantially fill.

Measured at chip 768 on NISAR L1: of 1,732 nodes, **88 disagree with the reference by more than a pixel
and every one lies in a one-cell-wide line along the swath boundary** — none in the interior. Excluding
one ring of boundary nodes takes the `dx` residual from rms 2.31 px to 0.35 and the count beyond 1 px to
2 of 1,476. `figs/nisar_l1_replay_chip768.png` maps it; a percentile cannot show it.

Both implementations correlate these chips and neither is measuring ground there, so **agreement says
nothing about whether the answers are right.**

**The change.** Decline a point whose chip footprint is not sufficiently inside the valid region, before
correlating, rather than emitting a value and hoping the coherence filter removes it. It needs a
*fractional* validity test over the chip's own footprint plus a threshold, evaluated per level since the
footprint grows 96 -> 768 px.

The input already exists on both sides and is simply not consulted:

- **Python:** `autoRIFT.py` computes `zero_mask` (`:127`) and `self.I1zeroMask`/`I2zeroMask`
  (`:272-278`), and never tests them when choosing which points to correlate.
- **Julia:** `AutoRIFT.valid(pair)` is already held on the `PassRunner` as `okmask`, and `_any_valid`
  already tests it — but only for *any* valid pixel in the chip, not a fraction.

**This subsumes much of items 2 and 3**: with edge chips declined, the cells those items are about stop
being searched.

## 2. Decimate a coarse level with the mask, not the fill value

Every per-point array is decimated to a coarse level by an unweighted mean over the cell, and every one
encodes nodata as an in-band fill value rather than carrying a mask alongside:

| array | nodata fill | decimated by |
|---|---|---|
| `xGrid`/`yGrid` | `0.5` (after the half-sample snap) | `INTER_AREA`, i.e. a plain block mean |
| `Dx0`/`Dy0` | `0` | `colfilt(..., mean)` |
| `SearchLimitX`/`Y` | `0` | `colfilt(max)` + `colfilt(range)` |

So a cell straddling the swath edge averages real values against the fill constant. On NISAR L1 the
resulting node lands **outside its own cell's real coordinate range for 94.2% of straddling cells** at
chip 768 and 99.0% at chip 384 — and the reference *searches* 3,967 of them, **87% of that level's
nodes**. The correlator is pointed at a place the grid does not describe.

**The change.** Average only the contributing pixels, and mark the node invalid where a cell has none,
instead of emitting a fill-weighted mean.

- **Python:** replace `cv2.resize(..., INTER_AREA)` on the coordinate arrays with a masked block mean, and
  give `colfilt`'s mean and range branches the same treatment.
- **Julia:** `_cell_mean` in `src/multichip.jl` is the single place this is decided; it takes the cell's
  mask instead of averaging blindly, and `_decimate_level` reduces `okmask` onto the cell to supply it.

**Currently matched deliberately**, and the measurement that forced that choice is worth keeping: against
the captured chip-768 lattice the *plain* average differs from the reference by a single constant (the
level's pad) where a fill-excluding average gives **3,288 different offsets**. Diverging here
desynchronizes every downstream comparison, which is exactly what a 115.5 px node offset was doing before
`_cell_centres` was made to match.

## 3. Make one position serve both halves of a level

`INTER_AREA` places a coarse node at the cell's **mean** coordinate; `INTER_CUBIC` reads the level's
answer back from the cell's geometric **centre**. Those are the same point only when the cell is uniform.
Measured on the NISAR L1 chip-768 lattice, in grid coordinates:

| population | n | correlate-vs-read-back gap |
|---|---:|---:|
| fill-free cells | 556 | **0.25 px** — self-consistent |
| cells straddling the boundary | 4,029 | **1,865 px** median, up to 27,538 |

The reference measures at one place and attributes the answer to another, by a median of nearly two
thousand pixels, on 88% of the nodes its coarsest level searches.

**The change** is not really independent: once item 2 is fixed, the masked mean of a valid cell *is* its
centroid, so the two coincide by construction. What matters is that **both halves move together** —
changing the correlation position without the read-back measures the field in one place and reports it at
another, which is measurable as the coarse residual growing rather than shrinking.

### What the end-to-end ladder added, and the condition for acting on items 2 and 3

The ladder localized these two precisely enough to say what fixing them should achieve and how to check it.
Splitting each golden endpoint's residual by the chip size its points resolved at — `rung_endpoint`'s
`_by_level`, `dev/GATES.md` — gives the same answer on every case measured:

  * **every base level agrees with the reference to a median of exactly 0** and a bias under 0.0004 px;
  * the whole-field bias tracks **only** the share of points that never reached the base level, and nothing
    else: 62% at base gives −0.006, 35% gives −0.039, 0% gives −0.089.

So these two items are the whole of two golden reds — `LT05_L1GS_001013`, which has **no** base-level
measurement at all, and `S1C_IW_SLC__1SSV_20250416T010159`, which has 35% — and they are not implicated
anywhere else. A fix should take those two cases' endpoint bias to the base level's, which is zero.

**Deferred by decision until the rest of the golden set is green, and the reason is measurability rather
than caution.** These are the widest-blast-radius items in the register: the node position is a property of
the *grid*, so changing it moves the coarse levels of every radar and NISAR case at once and re-opens the
`3.rdr` and `3.nisar` gates. With four cases still red on a missing COMPASS CSLC, a change here could not be
attributed — a green that appeared and a red that appeared would be indistinguishable from each other and
from the CSLC gap. Act on it when the reds that remain are only these two, and measure against all 22 plus
both radar gates, not against the two targets.

## 4. The rest of the register

Recorded in full, with per-item evidence and revisit conditions, in **`tools/golden/README.md`**:

| item | why it is questionable |
|---|---|
| Per-point chip-size bounds ignored at the base level | a point whose parameter file forbids a chip below 480 m is still correlated at the base chip size |
| The reference reports a search-window corner for a degenerate chip | a fabricated displacement over masked or textureless ground, which also changes which pyramid levels are skipped |
| Even-kernel `colfilt` chunk seam | assumes a left margin of `(k-1)÷2` where `generic_filter` uses `k÷2`; `nchunks - 1` corrupted columns per row. Not reproduced in AutoRIFT.jl |
| `UInt8` quantization before correlating | discards precision the filtered float field already has; the reference's own byte and float paths disagree. Production is moving to `Float32`, so both paths need a gate |
| Scene rotation derived from pixels, not the orbit | the reference's two recovered axes are not perpendicular, which a real cross-track direction cannot be |
| Agreement threshold is a fraction of the full window area | a border point is held to the same absolute neighbour count as an interior one |
| Outlier-filter neighbourhood derived from the X axis alone | on a 64x16 chip the window covers four times as much ground across track as along it |
| Level decimation derived from the X axis alone | coarsens y by 8 for a chip 52 px tall on a 48 px grid |
| Rectangular chips are not obviously handled correctly by *either* implementation | both now make the same choice, so **agreement says nothing about it** |
| The level-grid snap assumes a half-integer grid | `round(x + 0.5) - 0.5` moves an integer-valued grid by half a pixel. **Not matched** — `_cell_centres` reads the phase from the grid |
| Which side's coarse value is more nearly correct is unsettled | judged against a local truth neither wins: the reference is closer on `dx`, while its `dy` carries ~2x our rms at a comparable median |

Two differences run the **other** way, where AutoRIFT.jl is more nearly correct and deliberately does not
match: the Wallis variance (about-the-mean, ~360,000x more accurate than the reference's
`E[x²] - E[x]²`) and a seeded gap-fill RNG. Both are gated on tolerance rather than equality.
