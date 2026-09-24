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

So the two golden endpoint reds — `LT05_L1GS_001013`, which has **no** base-level measurement at all, and
`S1C_IW_SLC__1SSV_20250416T010159`, which has 35% — are entirely a coarse-level effect.

**That is not the same as saying these two items cause them, and an earlier revision of this file said so
without evidence.** Items 2 and 3 describe behaviour AutoRIFT.jl reproduces *deliberately and identically*:
both sides place the coarse node at the same fill-weighted mean and read the answer back from the same cell
centre. A shared defect produces no residual, so it cannot be what makes the two disagree — and fixing it
would *reduce* agreement, like every other item here, rather than turn a red green.

**What produces the bias is therefore something unmatched, and it is not yet identified.** The register's own
candidate in this area is the level-grid snap below — the reference applies `round(x + 0.5) - 0.5` literally,
which moves an integer-valued grid by half a pixel, where `_cell_centres` reads the phase from the grid —
and a half-pixel node offset on a spatially varying field would give exactly a coarse-only bias scaling with
the coarse share. Against that: matching the reference's literal snap was measured to *cost* 8.3 points of
exact match, which is the opposite of what a cause would do.

**That measurement has been made, and the positions agree exactly.** Comparing our coarse lattice against
the reference's captured `lvlN_xgrid`/`lvlN_ygrid` on both cases, through the path the code actually takes —
base to level grid by the chip ratio, then level to coarse pass by the sparse stride, which is 8 on both
sides — the spread is **zero on both axes at every coarse level of both cases**:

| case | coarse chips | spread x | spread y |
|---|---|---:|---:|
| `S1C ... 010159` | 56, 112, 224, 448 | 0 | 0 |
| `LT05_L1GS_001013` | 8, 16, 32, 64 | 0 | 0 |

What remains is one constant per level, which is the level's pad — the same single constant recorded against
the captured chip-768 NISAR lattice under item 2. It cannot be a real translation: the offsets reach ~100 px,
and a node displaced that far would produce a displacement error orders of magnitude larger than the 0.126 px
bias actually seen.

**The search radius is measured and is probably not the cause either.** Comparing the coarse pass's
*inputs* against the reference's captured `searchx`/`searchy`/`dx0`:

| level | stride | `searchx` exact | differing | max abs difference |
|---|---|---:|---:|---:|
| chip 56 | **1** | **100.00%** | 0 | 0 |
| chip 112 | 2 | 98.35% | 182 | 20 |
| chip 224 | 4 | 96.12% | 107 | 58 |
| chip 448 | 8 | 92.94% | 48 | 59 |

The divergence appears exactly where the level decimation begins — stride 1 is exact, so `_coarse_points`
is right and the difference is in `_decimate_level`'s radius. But it is not a simple wrong choice among
reductions. A/B-ing the three candidate forms carried through the same coarse pass, **none is exact and the
two cases disagree about which is best**: `max + range` gives 98.35% on `S1C ... 010159` where a plain
subsample gives 96.69%, while on `LT05_L1GS_001013` those are 97.08% and 98.42%. The differing positions
are contiguous column runs rather than the periodic pattern `colfilt`'s chunk seam would leave, so that
entry is not the explanation either.

**And the signature argues against the radius entirely.** A search window differing on a few percent of
points with a median of zero produces *scattered* errors — a window either contains the same peak or rails
out. What the endpoint shows is a systematic **bias**, a median shift of -0.126 px, which is the signature
of a convention or position offset rather than of occasional window differences.

**The bias predates the end-to-end ladder, measured both ways.** The correlator-only endpoint —
`correlator.jl`, fed `pointset_from_capture` and the captured imagery, with no geometry built — carries it
already:

| case | correlator-only, reference's grid | e2e rung 5.7, our grid | share at base |
|---|---|---|---|
| `LT05_L1GS_001013` | dx **-0.0800**, dy **-0.0914** | dx -0.126, core -0.0888 | 0% |
| `S1C ... 010159` | dx **-0.0768**, dy -0.0145 | dx -0.0446, core -0.0394 | 35% |

So the ladder did not introduce it; it exposed it at a gate that bounds the bias where the older gates
reported it without failing on it, which is also what `3.rdr` at 3/8 and `3.nisar` at 0/2 were failing on.
The e2e core figures reproduce the base-share model on this page to three digits — 0% predicts -0.089
against -0.0888, and 35% predicts -0.039 against -0.0394.

**The stage ladder being green is not evidence against this.** `stages.jl` on
`LT05_L1GS_001013_19920425` is 13 of 13, and its `3.7 coarse correlation` rung is **bitwise exact** —
37,311 nodes, median 0, max 0, bias +0, on both the `UInt8` and `Float32` paths. That rung is fed the
reference's own *coarse* grid, so it establishes that the coarse correlation itself is right and places the
cause downstream of it.

**The read-back's phase-0 alignment costs agreement, and an earlier revision of this section got the
mechanism and the reading both wrong.** Scanning the bicubic read-back's half-sample phase on
`LT05_L1GS_001013` — an experiment, reverted — gives:

| phase (source samples) | agreeing within 1 px | `dx` core | `dy` core |
|---|---:|---:|---:|
| y -0.5 | 19,498 | -0.0645 | -0.0492 |
| y -0.25 | 19,579 | -0.0687 | -0.0630 |
| **0, the current convention** | **17,090** | **-0.0804** | **-0.0900** |
| y +0.5 | 19,502 | -0.0711 | -0.1005 |
| x +0.5 | 19,525 | -0.0414 | -0.0556 |

**Zero is a singular point, not a shifted optimum**: offsets of either sign on either axis improve
agreement by about 2,400 points. That rules out a simple convention offset, which would improve one
direction and worsen the other.

**Two readings to avoid, both of which this file previously asserted.** `n_core` counts points *agreeing
within one pixel*, not points measured (`correlator.jl:422-426`) — coverage is unchanged across the sweep,
and AutoRIFT.jl measures 20,204 points to the reference's 20,016, which is *more*. And the mechanism cannot
be an unfilled node: `_fill_level_holes` ends in `_fill_nan_nearest`, which BFS-fills every `NaN` from the
nearest finite cell and returns unchanged only when the array is entirely `NaN`. The bicubic therefore never
sees a `NaN`, `wsum` is always 1, and no destination can be dropped for want of weight.

**What is left is what zero does that an offset does not.** At exact node alignment the cubic weights are
`(0, 1, 0, 0)`, so each coarse node is reproduced verbatim; at any offset the value is a blend of four
neighbours. Agreement improves under blending, which says the reference's upsampled field is *smoother*
than a verbatim reproduction of its own coarse nodes — so the question is what smooths it. The phase itself
is independently correct, `(j - 0.5) * scale - 0.5` being `cv2.resize`'s mapping and `a = -0.75` being
`INTER_CUBIC`, so moving it to minimize a bias would be fitting the convention to the metric.

### The endpoint reds are data-driven, established by a controlled pair

`LT04_L1TP_063018` and `LT05_L1GS_001013` are a matched pair: **all 21 captured scalars and all
`kwargs_from_capture` keys are identical**, both are 30 m L4/L5 band-2 imagery, both run a four-level
pyramid from chip 8 to chip 64 at `ScaleChipSizeY` 1.0 and grid spacing 4. Fed the reference's own images
and grid through `correlator.jl`, one has a `dx` bias of **-0.00004** and the other **-0.070**.

So the same code on the same parameters diverges on the data alone, and no code path is implicated that the
two do not share. The other twenty cases pass because their data does not trigger it.

**What is distinctive about the red pair is the correlation geometry.** It is the only case in the set where
*both* scenes are `L1GS` — systematic geometric correction, no ground control on either image — so its
apparent displacement is largely residual misregistration: `v` median 1102 m/yr against an `off2vx` of
170 m/yr per pixel is **6.5 px of displacement on an 8-pixel chip**. The chip in the second image overlaps
the first by about a fifth of its area. `LT05_L1TP_060018` pairs an L1TP reference with an L1GS secondary
and is green; `LC09` is L1GT on both sides and is green.

**The residual's shape, fed identical inputs.** Not a tail, not a discrete step, not a uniform offset:

    band                negative   positive
    0.031 .. 0.5 px        6,212      2,526        2.5 : 1
    exactly zero                     2,498 (12.74%)

against the matched green case's **65.8% exactly zero** with symmetric wings. A third of the points
disagree by a sub-pixel amount, one-sided.

**Six mechanisms are eliminated by measurement**, each recorded above: the coarse node positions (exact),
the coarse correlation given the reference's coarse grid (bit-exact), the read-back's phase, kernel and
scale (verified against `cv2.resize`), `_fill_level_holes` leaving `NaN` (impossible — it ends in
`_fill_nan_nearest`), the byte-versus-float correlator path (the reference's two paths are bit-identical
here), and peak conditioning (the matched green case has the same `peak_ratio` distribution, median 1.619
against 1.625, and agrees exactly in every bin). The search window's edge is eliminated in the opposite
direction: the red case never reaches it, at 0.00% against the green case's 6.65%.

### Located: the level's outlier filter, where 74% of the points are discarded

Walking the one level this case resolves at — chip 16, the only `fine` pass in its whole run — stage by
stage against the reference's own per-level records:

| stage | points | mean residual |
|---|---:|---:|
| fine pass, before the filter | 19,267 ours / 19,809 reference | **-0.0067** |
| after `_reject_and_fill!` / `filtDisp` | **5,051 ours / 4,754 reference** | **-0.0441** |
| endpoint, after interpolation, merge and selection | 19,608 | -0.0698 |

**The filter discards 74% of the measured points, and the surviving quarter is disproportionately where the
two disagree**, so the mean moves 6.6-fold in that one step. Interpolation is mean-preserving (verified) and
the remaining 1.6-fold to the endpoint is its population selection.

The *decisions* very nearly agree — 1,261,621 of 1,262,080 mask entries match, 378 only ours and 81 only the
reference — but ours keeps 6% more points, 5,051 against 4,754, and the reference reports `filt_width` 9 over
3 iterations.

**So this is not the correlator, the node positions, the read-back, or the hole fill**, each measured clean
above. It is the outlier rejection, and the register already carries two items about that step: the
agreement threshold being a fraction of the full window area, and the filter neighbourhood being derived
from the X axis alone. This case is where they bite, because it is the only case whose points are
overwhelmingly rejected, which leaves the filter's decisions to determine the answer rather than to trim it.

**Why the twenty others pass** follows: their base level resolves most points, so the filter trims a small
minority and its decisions cannot dominate. This case has no base-level measurement at all — the reference
abandons chips 8, 32 and 64 after their coarse passes and runs exactly one fine pass, at chip 16 — so a
single filter invocation decides the entire product.

### Superseded: the coarse level agrees at 86.7% exact; what is left is a mean, and selection

**With the refinement the pipeline actually uses**, level 16's own estimate on its own decimated grid, before
`_undecimate_level` interpolates it:

| case | exact | median `|r|` | mean |
|---|---:|---:|---:|
| `LT05_L1GS_001013` (red) | **86.69%** | **0** | -0.0067 |
| `LT04_L1TP_063018` (green) | 98.54% | 0 | -0.0003 |

**The refinement varies by level** — `PyramidRefine(16)` at the base, `(32)` at level 2, `(64)` above — and
an earlier revision of this section used `first(p.subpixel)` for every level, which understated level 2's
precision and produced a spurious median of exactly 1/32, half a step of the base's 1/16. Read
`subpixel_at(p, k)`.

**The endpoint's exact fraction follows from interpolation arithmetic, not from a defect.** A 2-D bicubic
blends sixteen coarse nodes, so all sixteen must agree for the result to agree exactly: `0.867^16 = 0.10`
against an observed 12.74%. Interpolating an 87%-exact field cannot do better than that.

**What is not explained is the mean**: -0.0067 at the coarse nodes against -0.0800 at the endpoint.
Interpolation cannot change a mean, and this is verified rather than argued. The kernel is bicubic —
`a = -0.75`, `cv2.INTER_CUBIC` — which is linear *in the data*: a cubic in position applied as a normalized
weighted sum. At stride 2 every fine sample sits at exactly a quarter of a source sample from its node, one
phase for the whole grid, so identical weights apply everywhere. The one worry worth having is that
`resample!` clamps its indices at the array edge, which would redistribute weight onto duplicated boundary
nodes; measured on this case's geometry — a small island of constant difference in a large empty grid, one
touching the array corner so the clamp engages, and one graded so the cubic's negative lobes bite — the
output mean matches the input mean to **six decimal places in all three**. `NaN` taps are skipped and
`wsum` renormalizes over those present, so the operator stays a weighted average over available data. The remaining route is therefore *selection*: which interpolated points survive
`_undecimate_level`'s `valid` mask and `_merge_level!`. Three separate readings in this investigation have
turned out to be population effects rather than value effects, so that is where to look and what to control
for.

**The hole-fill fallback is measured and is a minor term.** For this case the base level resolves nothing, so
level 16's prior is entirely `NaN` and `_fill_level_holes` falls through to its 5x5 median and then the
nearest-neighbour BFS — a path no other case in the set exercises, since every other base level supplies a
prior. Disabling the median step moves the endpoint `dx` bias from -0.0804 to -0.0768, about 4.5%, with
coverage unchanged. A contributor, not the cause.

### Superseded: the coarse correlation agrees; the divergence is after the interpolation

**Isolating the coarse level's own estimate, on its own decimated grid, before `_undecimate_level`
interpolates it** — which every earlier endpoint comparison included:

| case | chip 16, stride 2 | exact | mean | median `|r|` |
|---|---|---:|---:|---:|
| `LT05_L1GS_001013` (red) | 19,774 | 41.42% | **-0.0096** | **0.03125** |
| `LT04_L1TP_063018` (green) | 309,758 | 45.70% | **-0.0046** | **0.03125** |

The coarse estimates agree about equally well — a factor of two apart — where the *endpoints* are -0.070
against +0.0002, a factor of 370. So the coarse correlation is not where the two diverge.

**Both medians are exactly 1/32**, one subpixel quantization step at `PyramidRefine(16)`. So the typical
coarse disagreement is a single step on roughly 55% of points in *both* cases, which is tie-breaking rather
than error, and it is harmless in the green case.

**The open question is therefore how a one-step disagreement becomes a one-signed mean of -0.070 after
interpolation in one case and cancels in the other.** Interpolation is linear and cannot amplify a mean
sevenfold, so the disagreeing points must be weighted asymmetrically. The candidate is that the red case's
entire field comes from this single coarse level — `out_ChipSizeX` resolves to `[16.0]` and nothing else —
where the green case's chip-16 points sit beside chip-8 base measurements that agree exactly.

**Two attributions to retire.** The threshold at the chip size was an artifact of binning a quantity that
scales: the search radius grows with the displacement (median 20, 31, 43, 58, 76 across the `|dx|` bins) so
`|dx|/radius` stays flat near 0.15 throughout, and no point on this case comes near its search edge. And
chip size does not bound measurable displacement at all — the search range does.

**The superseded reading was displacement against the base chip size.** Binning
the residual by the reference's own `dx`, fed identical inputs:

| `|dx|` px | red case median `|r|` | red case mean `r` |
|---|---:|---:|
| 0 .. 1 | 0.0089 | -0.011 |
| 1 .. 3 | 0.0083 | -0.036 |
| 3 .. 6 | 0.0137 | -0.043 |
| **6 .. 10** | **0.0653** | -0.102 |
| **> 10** | **0.1518** | -0.211 |

Flat near 0.01 while `|dx|` stays under 6, then five-fold at 6 to 10 and twelve-fold beyond — **against a
chip of 8 px**. The matched green case settles it: `LT04_L1TP_063018` has a reference `dx` of median 0.57
and p95 1.44, twelve times smaller than its identical 8-pixel chip, and a residual of zero. The red case
runs at median 5.01 and p95 10.82.

**That explains every other observation.** An 8-pixel chip cannot resolve an 11-pixel displacement, which is
why this case has **no base-level points at all** and why its whole answer arrives through the coarse
pyramid and the prior chain — the one region every other measurement pointed at. It is why the residual
worsens toward the patch's interior, where the displacement is largest: exact agreement falls from 17.0% at
distance 3-5 from the nodata edge to 3.5% at 21-40, the *opposite* of the boundary effect the green case
shows (64.1% rising to 98.1%). And it is why twenty cases pass, their displacements being a small fraction
of their chip.

**So the disagreement is not a defect in a step; it is the pyramid's behaviour when the base level cannot
measure.** What differs is how a coarse estimate is carried down and refined when no finer level can
correct it. The fix, and the gate question, both belong there.

**An earlier phrasing of this candidate said chip *overlap*, which is wrong**: in template matching the
chip is always fully covered by the search window, which is the chip plus twice the search radius, so a
large displacement costs no overlap. The regime is right and the mechanism was not.

**The superseded candidate was chip geometry at large displacement** — how each implementation
extracts or pads a chip whose match sits 6.5 px away on an 8-pixel template. A padding or overlap asymmetry
there is one-signed by construction, which is the one property of this residual that noise and conditioning
cannot produce. That is not yet measured.

**What remains is what happens to a coarse estimate after it is correlated**: `_undecimate_level`'s
read-back onto the fine grid and the merge that consumes it. That is the one link in this chain no
measurement here has yet isolated — positions are exact, the radii are near-exact and the wrong shape of
cause, and the correlation is bit-exact.

**So the position hypothesis is closed.** Items 2 and 3 are not the cause, now by measurement as well as by
the argument above; nor is the `round(x + 0.5) - 0.5` snap, which is a position effect; nor the sparse
stride. The cause is downstream of where the coarse node sits — in the coarse pass's *values*, or in how its
result restricts the fine search and is read back. The capture carries `lvlN_dx`, `lvlN_dy`,
`lvlN_searchx`/`searchy` and `lvlN_kept` per level, which is what a comparison of those would be fed.

**Deferred by decision until the rest of the golden set is green, and the reason is measurability rather
than caution.** These are the widest-blast-radius items in the register: the node position is a property of
the *grid*, so changing it moves the coarse levels of every radar and NISAR case at once and re-opens the
`3.rdr` and `3.nisar` gates. With four cases still red on a missing COMPASS CSLC, a change here could not be
attributed — a green that appeared and a red that appeared would be indistinguishable from each other and
from the CSLC gap. Act on it when the reds that remain are only these two, and measure against all 22 plus
both radar gates, not against the two targets.

## 4. Restore the stable Wallis local variance

**Adopted for agreement on 2026-09-23, and it is the clearest case in this file of the trade this
document exists to record.** `_masked_boxstd` computes `sqrt(E[x²] - E[x]²)` clipped at zero, which is
`_preprocess_filt_std` (`autoRIFT.py:47-56`). That is a difference of two large, nearly equal numbers, so
it cancels catastrophically on a bright low-contrast window, and the cancellation can drive the variance
negative — the clip is what stops the square root being `NaN` and the `NaN` reaching the validity mask,
where it would silently discard data. Squaring the deviations about the measured mean costs one extra pass
and has neither problem.

**What it cost to be right, measured on `LT05_L1GS_001013`.** The stable form moved 90% of the filtered
field by a median of 0.113 on a field clamped to ±3, and on scene 1 it moved the band-reject's
fire-or-decline decision: the branch is `(sA/sB >= 2 | sB/sA >= 2) & (max > 500)` on counts of 1588 and
3279, clearing the ratio test by 3.2%, and the more accurate field landed on the other side. So AutoRIFT.jl
*declined* the reject where the reference fired, and a whole scene's output became clamped Wallis instead
of a band-rejected field. Adopting the reference's form:

| scene | stable form | reference's form | the harness's own reproduction of the reference |
|---|---:|---:|---:|
| `... 19920425` | declines the reject, median 0.1983 | **fires, 0.005615** | 0.005387 |
| `... 19920628` | 0.02398 | **0.006666** | 0.00647 |

The decision now agrees on all six scenes of the three `wallis+destripe` cases, and the field tracks the
reference to within 4% of the residual that remains.

**What the fix needs, and why it cannot be judged by the golden set.** Both implementations would then use
the accurate form, so agreement says nothing — the judgement has to be a measurement against an exact
`Float64` truth. That measurement exists and is unambiguous: on a 256² scene of mean 130 and standard
deviation 25 at `width = 5`, the reference's form is off by a median of 0.54 and up to 5.79 where
about-the-mean is off by a median of 1.5e-6.

**A synthetic Gaussian field will not show you the difference.** On a smooth random field of the same mean
and spread the two forms agree to a median of 4.4e-5 — a relative 1.8e-6, three orders below what the real
scene shows. The cancellation needs the bright *low-contrast* regions a real image has and a Gaussian does
not, so a test built on `randn` will report the change as harmless when it is not.

## 5. The rest of the register

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

One difference runs the **other** way, where AutoRIFT.jl is more nearly correct and deliberately does not
match: a seeded gap-fill RNG, which is gated on tolerance rather than equality because the reference draws
from NumPy's unseeded global generator and so cannot match itself either. The Wallis variance used to be
the second such difference and is now item 4 above, adopted for agreement.
