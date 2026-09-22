# The gate ledger

Every measurement that has been confirmed green, with the command that produced it. A gate stays on
this list only while it still passes: `regate.jl` re-runs all of them, and a row that stops holding is
a regression to fix rather than a number to update.

**Deferred correctness work is not here.** A gate records what is *verified*; a defect this project
reproduces on purpose is in [`CORRECTNESS.md`](CORRECTNESS.md), with
`tools/golden/README.md` holding the per-item evidence. When every gate below is green, that file is the
work list.

**Why a ledger rather than a report.** `tools/golden/README.md` records what is *known* about the two
implementations. This records what has been *verified at a commit*, which is a different claim and the
one that decays. A gate whose command cannot be re-run is not a gate.

Machine: Apple M2 Max, 12 cores, macOS 26.5.2. Julia 1.12.5 for every row recorded below; the
toolchain has since moved to **1.13.0**, on which the suite passes 704,596/704,596 and the untiled
point count on the S2B case is unchanged at 612,607. Re-measure before quoting a *runtime* against the
new toolchain. Reference: autoRIFT 2.1.1 in
`micromamba -n arift-ref`, whose `autoRIFT.py` is byte-identical to the pinned v2.1.2; container
`ghcr.io/asfhyp3/hyp3-autorift:0.28.4` for the golden cases.

> **Read case-level figures from the re-measurement sections at the end of this file, not from the
> sections that first recorded them.** Three defects invalidated case-level numbers in sequence. All
> twenty cases have since been re-measured, so no row is awaiting a measurement — but a row's *original*
> section still carries the superseded figure, because these are re-measured rather than edited.
>
> | change | what it broke | re-measured in |
> |---|---|---|
> | `pointset_from_capture` handed AutoRIFT.jl a wrong-sign `Dy0`, the prior *before* `arImgDisp_*` flips it | every `exact`, `bias` and coverage number on all twenty cases | "Re-measurement after the Dy0 sign fix" |
> | `_level_decimation` consulted the y extent, so a level's stride was half the reference's — on three radar pairs no level coarsened at all | every anisotropic chip: eight radar, both NISAR | the NISAR and radar steps below |
> | `_grid_step` read a spacing of zero past a majority nonzero nodata fill | nine of twelve optical, all eight radar, both NISAR | the optical and radar steps below |
> | `_cell_means` places a coarse node at its cell's block mean rather than shifting the cell's first point — deliberate, and it averages the nodata fill in | the core `dx` bias on five of eight radar cases and both NISAR cases; every other gated statistic improved | "the radar and NISAR gates after the cell-mean change" |
>
> Gate 0 (`tools/ab`) is unaffected by any of them: it passes a zero prior on an unrotated synthetic grid,
> so neither the sign convention nor the nodata fill crosses it. **`3.rdr` is 3 of 8 and `3.nisar` 0 of 2**,
> both on core-bias bounds calibrated before the cell-mean placement; they are the open red gates, and the
> final section records why the bounds are not simply widened.

**`regate.jl` covers all 22 cases: `3.opt` twelve optical, `3.rdr` eight radar, `3.nisar` both NISAR.**
`3.nisar` runs a sixteenth of each NISAR grid (`--stride 4 --block 128`, ~1 minute a case), because a
whole-grid NISAR run in `correlator.jl` is untiled and costs 5-10 minutes at a 49-61 GiB peak — the two
cannot even run concurrently on this machine's 96 GiB. **Its thresholds are calibrated to the thinned
grid and are not comparable to the whole-grid figures or to `3.rdr`'s bounds**; thinning changes which
pyramid levels resolve, which is measured under "both NISAR cases re-measured whole-grid" at the end of
this file. The whole-grid figures are a measurement there, not a gate.

## Gate 0 — the verified floor

The agreement that predates the golden work. Everything else is built on it, so it is re-run first and
after every change. All three rows measured at `afdec6d` (2026-09-06), the tip of `golden-tests`.

| # | what | command | expected | measured | state |
|---|---|---|---|---|---|
| 0.1 | correlator alone, `UInt8`, chip 32 | `stage1_julia.jl 1024 32 20` → `stage1_python.py` → `compare.jl chip32` | 100% exact | **100.0% exact** on `dx` and `dy`, median/p99/max all 0.0000, corr 1.00000, 900/900 points, at every correlation gate | **green** |
| 0.2 | whole pipeline, 3072² window | `stage2_julia.jl 3072 16 64 20` → `stage2_python.py` → `compare2.jl full` | 81.8% exact, 98.7% within step, p99 0.0752, level 99.3% | **81.8% exact** `dx` / 82.3% `dy`, 98.7% / 98.9% within step, p99 0.0752 / 0.0719, bias +0.0000 both axes, corr 0.99964, 88,123 shared points | **green** |
| 0.3 | ITS_LIVE granule, real Landsat pair | `test/realdata.jl` | 17 assertions pass, correlation > 0.95 | **17/17 pass**, 45.2 s | **green** |

Gate 0.2's per-level counts on the shared block: chip 16 → 55,202 jl / 55,182 ref, chip 32 → 19,099 /
18,942, chip 64 → 14,164 / 14,092. Coverage 88,465 jl against 88,216 ref, 342 jl-only and 93 ref-only.

**The five `src/` commits on this branch did not regress the benchmark.** `2498548`, `a45dcbd`,
`79ff961`, `f4c3679` and `6facfd0` were each justified against a golden measurement rather than
against this one, so the check was owed. Every headline statistic in row 0.2 reproduces the recorded
figure exactly, and the reference's own level counts are unchanged, so the reference side is
byte-stable across the interval too.

### Two harness faults fixed to reach this gate, neither of which was a package defect

Both would have been read as evidence about AutoRIFT.jl.

**`compare.jl` could not read its own bundle.** `stage1_julia.jl` writes `dtype UInt8` into the
manifest and all four Julia readers parsed every scalar with `parse(Int, ...)`, so the comparison died
with `ArgumentError: invalid base 10 digit 'F' in "Float32"` *after* correlating — the run looked like
a failed measurement rather than a failed reader. One parser now lives in `tools/ab/bundle.jl` and
keeps a non-numeric scalar as a `String`; `compare.jl`, `compare2.jl`, `zoom.jl` and
`bench_figures.jl` all read through it, so adding a scalar to the writer can no longer break three
readers at once.

**The stage-2 bundle halves were from different code.** `python_dx.bin` was two days older than
`julia_dx.bin`, so the recorded comparison was between a Julia run at one commit and a reference run
at another. Row 0.2 was measured with both halves produced in one session, which is now the only way
the gate is quoted.

## Gate 1 — the capture harness observes what it claims to

| # | what | command | expected | measured | state |
|---|---|---|---|---|---|
| 1.1 | the per-level patch targets the module that defines the symbols | see below | `autoRIFT.autoRIFT` module patched; the class path raises | **green** — `arImgDisp_s` and `DISP_FILT.filtDisp` both patched; patching the class now raises `AttributeError` | **green** |

```bash
micromamba run -n arift-ref python -c "
import sys; sys.path[:0] = ['tools/golden', 'tools/ab']
import capture, autoRIFT
m = capture.reference_module(); capture.install_levels(m, {})
assert m.arImgDisp_s.__name__ == 'patched'
assert m.DISP_FILT.filtDisp.__name__ == 'patched_filt'
try: capture.install_levels(autoRIFT.autoRIFT, {}); raise SystemExit('FAIL: silent no-op')
except AttributeError: print('ok')"
```

**`install_levels` had never fired, on any capture ever taken.** It resolved the module as
`from autoRIFT import autoRIFT`, but the package's `__init__` binds that name to the **class**, which
has none of `arImgDisp_u`, `arImgDisp_s` or `DISP_FILT`. `getattr(..., None)` returned `None` and
`wrap_corr` returned silently, so every capture on disk carries `levels: []` — and
`capture_reference` warns about exactly that, attributing it to a capture predating the feature. A
fault that the harness explains away as staleness is worse than a crash.

Three changes make the failure mode unavailable: `reference_module()` resolves through
`sys.modules['autoRIFT.autoRIFT']` and type-checks the result, `wrap_corr` raises rather than
returning when a symbol is absent, and `runAutorift` raises after the call if no per-level record was
written. The last is the important one — the same distinction the stale-`.nc` trap already taught:
that a patch was *installed* does not establish that it *fired*.

## Gate 2 — the reference's window reductions, pinned

`colfilt` supplies the pyramid's grid resize, search-radius widening, prior average and hole fill
(`autoRIFT.py:509-808`). `test/fixtures/` pins `filter2d`, `resize`, `pyrup`, `matchtemplate`,
`disttransform` and `peak` bit-exact but held nothing for `colfilt` or `bwareaopen`, so every
reduction in `src/window.jl` was matched against a reading of the argument list. 135 new cases fix
that: six options × five kernels × two NaN densities × two chunk counts, plus 15 `bwareaopen` cases.

```bash
micromamba run -n arift-ref python tools/python_ref/gen_fixtures.py colfilt bwareaopen
julia --project=. -e 'using TestEnv; TestEnv.activate(); include("test/window.jl")'
```

| # | what | expected | measured | state |
|---|---|---|---|---|
| 2.1 | `max`, `min`, `mean`, `median` against `colfilt` options 0/1/2/3, one chunk, kernels 2/3/4/5/9, with and without NaN | exact | **exact on all 40 cases**, `NaN` for `NaN` | **green** |
| 2.2 | `range` against option 4 | exact on finite values | **exact**, with one documented divergence: an all-NaN window | **green** |
| 2.3 | `mad` against option 6 | same validity, values within Float32 error | **same NaN set**, max difference **4.8e-7** | **green** |
| 2.4 | the even-kernel chunk seam is the reference's, not a Julia defect | one chunk exact; four chunks differ only on derived seam columns | **exact at one chunk for every reducer and both even kernels**; at four chunks every disagreement lies on columns 9/16/23, derived from `colfilt`'s own chunk split | **green** |
| 2.5 | `small_components` against `bwareaopen`, 8-connected, `size1` exclusive | complements within the mask | **exact on all 15 cases**, including the diagonal chain that discriminates 8- from 4-connectivity | **green** |
| 2.6 | no slide: `fixtures_test.jl`, `window.jl`, `multichip.jl` | pass | **36 / 227 / 547,273 pass** | **green** |

Three findings, each of which would otherwise have been read as an AutoRIFT.jl difference:

**The even-kernel disagreement is entirely the reference's chunking.** At `chunkSize = 1` AutoRIFT.jl
matches `colfilt` **exactly** at kernels 2 and 4 for max, min, mean and median. At the default
`chunkSize = 4` the reference's own left margin — `(k-1)//2` where `generic_filter` centres at `k//2`
(`autoRIFT.py:1490-1499`) — puts the first output column of each chunk after the first one column off.
The seams are arithmetic, not data: columns 9, 16, 23 on a 30-column field, for every reducer and both
even kernels. The test derives them from the chunk split rather than hardcoding them, and asserts that
no disagreement occurs anywhere else. A reducer may agree at a seam by chance, so `bad ⊆ seams` is the
assertion rather than `bad == seams`.

**`range` and `max` handle an empty window differently in the reference, and that asymmetry is real.**
`frange` starts from `±inf` and compares with `<`/`>`, which a `NaN` never satisfies; option 0 then
converts a leftover `-inf` back to `NaN` and **option 4 does not** (`autoRIFT.py:1540-1546`). So the
reference returns `-Inf` from an all-NaN window where AutoRIFT.jl returns `NaN`. AutoRIFT.jl's answer
is kept, on two grounds rather than preference: it is what every other reduction returns for an empty
window, and it is the value that composes — a decimated cell's radius goes through
`ceil(Int, ...)` (`_decimate_level`), where `-Inf` and `NaN` both throw. **The divergence is
unreachable on production input**: the captured priors on the golden Landsat case carry 0 `NaN` in
`Dx0` and `Dy0` across all 5,475,584 points, so no window is ever empty.

**A partial fixture run used to truncate the manifest.** Regenerating one group rewrote
`manifest.json` with only that group, dropping the other seven from an index `fixtures_test.jl` gates
on — while leaving their case directories on disk, so the corpus and its index disagreed. The manifest
now merges and records the library versions per group, which the corpus genuinely needs: the original
311 cases were generated under OpenCV 4.10 and these 135 under 4.13, and rewriting the older ones to
match the newer build would move the target the suite is held to rather than verify it.

## Gate 3 — the stage ladder

`tools/golden/stages.jl` feeds AutoRIFT.jl the reference's own dumped input for one stage of
`autorift()` at a time and diffs against that stage's dumped output. Julia output is never chained
into the next Julia stage, so each rung is a statement about one step rather than about a composition
of two dozen.

```bash
CAPTURE_STAGES=1 CAPTURE_STAGE_LEVEL=0 julia --project=tools/golden \
    tools/golden/intermediate.jl LC08_L1TP_009011 --force
julia --project=tools/golden -t 8 tools/golden/stages.jl LC08_L1TP_009011
```

Measured on `LC08_L1TP_009011_20200703` at chip 16 (the base level), grid 2344×2336 = 5,475,584
points, coarse grid 293×292 = 85,556.

| # | stage | reference array | gate | measured | state |
|---|---|---|---|---|---|
| 3.1 | the level's grid | `xGrid0_L0` | exact | **all 5,475,584 equal** | **green** |
| 3.5 | zero / `minSearch` rewrite | `SearchLimitX0_rev1_L0` | exact | **all 5,475,584 equal** | **green** |
| 3.4 | the prior | `Dx00_L0` | exact | **all 5,475,584 equal** | **green** |
| 3.6a | coarse sample lattice | `xGrid0C_L0` | exact | **all 85,556 equal** | **green** |
| 3.6b | coarse radius, both axes | `SearchLimitX0C_L0`, `…Y…` | exact | **all 85,556 equal**, after the fix below | **green** |
| 3.6c | coarse prior, both axes | `Dx0C_L0`, `Dy0C_L0` | exact | **all 85,556 equal** | **green** |

Two stages beyond the committed rungs, measured so the next rung starts from a number rather than an
assumption:

| stage | measured | reading |
|---|---|---|
| coarse correlation `DxC` | **coverage identical** — 35,142 measured on both sides, 0 exclusive either way, and `M0C` agrees on 85,556 of 85,556 (**100%**). `dx` **86.54%** exact, rising to **94.66%** at correlation ≥ 0.5 | agreement improving with peak strength is the shape to expect. 9.61% differ by more than a pixel, concentrated at large radii (median radius 13 against 6 elsewhere) |
| `MC`, the coarse rejection | reference keeps 16,746, AutoRIFT.jl 16,724; **98.47%** of the grid agrees, and the exclusive split is balanced — 644 jl-only against 666 ref-only | the filter's parameters match exactly (width 9, `FracValid` 0.32, 2 iterations). A balanced split is marginal decisions straddling one threshold, not a bias |

### A real bug the endpoint comparison could not see

The reference sets `filtWidth = stride + 1` when the sparse stride is even and `stride` when it is odd
(`autoRIFT.py:618-626`), so the coarse radius reduction is symmetric about the node it samples at.
`_cell_max_radius!` reduced over the stride, which at an even stride reaches one fewer point on the
right.

Measured against the reference's own `SearchLimitX0C`: **1,809 of 85,556** coarse points carried a
radius too small, by up to **152 pixels**, every one downward and none on the grid border. A coarse
point whose radius under-covers its cell searches a narrower window than the reference did, so it
rails out or misses the peak exactly where the prior was doing work — which presents as a correlator
disagreement rather than as a setup difference. With the rule applied both axes are exact on all
85,556 points, and `tools/ab` stage 2 is unchanged at 81.8% exact.

The endpoint comparison for this case reads 63.24% exact and nothing in it points at a window width.
The rung that found it compares one array against the array the reference built for it.

### The argument-order trap, paid once more

Writing the coarse-pass comparison, `ImagePair(I1, I2)` in place of `(I2, I1)` reported **22.25%**
exact where the correct order reports **86.54%** — low enough to look like a finding and high enough
not to look like a bug. `arImgDisp_*(a, b)` cuts its chip from `b` and the reference calls it as
`(self.I2, self.I1)`, so `I1` supplies the chip and binds to AutoRIFT.jl's *secondary*.
`tools/ab/README.md` states this and `correlator.jl` asserts it; every new diagnostic has to re-derive
it, and the cost of getting it wrong is a plausible measurement rather than an error.

### The base level's residual is the `UInt8` quantization, measured on both paths

The fine pass at the base chip size, fed the reference's own post-coarse-mask radii so only the
correlation is under test, agrees on **71.88%** of 2,159,437 points exactly — with **coverage
identical**: 2,159,437 measured on both sides and zero exclusive either way. `DxF` is 100.000% on the
1/16 grid on the reference side, so `exact` is the right statistic here and 71.88% is a real figure
rather than an artifact of comparing unquantized fields.

That number sits below the pipeline benchmark's 81.8%, and the reason is the element type rather than
the pipeline. Running the *same window, same chip, same radius* through stage 1 at each element type
separates them, because the reference has two correlators and dispatches on dtype:

| path | entry point | exact `dx` | within one step | p99 | max |
|---|---|---:|---:|---:|---:|
| `Float32` | `arImgDisp_s` | **100.00%** | **100.00%** | 0.0000 | **0.0000** |
| `UInt8` | `arImgDisp_u` | 84.8% | 97.7% | 4.41 | **35.81** |

The float path is **bit-identical at every one of 3,721 points**. The byte path, on the same imagery
at the same settings, disagrees on 15% and by up to 35.8 px. Production — and therefore every golden
case — takes the byte path, because `uniform_data_type` rescales each scene by its own mean and
standard deviation and quantizes to 256 levels before `runAutorift` is reached
(`autoRIFT.py:359-384`). The capture confirms it: `in_I1` is `UInt8` using all 256 levels.

So the base-level residual on a golden case is **not** evidence of a defect in AutoRIFT.jl's
correlator. Collapsing a filtered float field onto 256 levels creates ties and near-ties the float
field does not have, and a tie broken differently at a plateau puts the peak far away rather than one
step away — which is what a 35.8 px maximum beside a 0.0000 median describes. The quantizer itself is
verified against the reference's own `uniform_data_type` (`tools/ab/README.md`), so the two sides are
quantizing identically and then disagreeing about the surface that results.

This is why `tools/golden/README.md` lists the `UInt8` conversion as **matched, not endorsed**: it is
reproduced to agree with the reference, and it costs accuracy at every point. The float path being
bit-identical is the measurement that says so.

## Gate 4 — the endpoints, and no slide

```bash
julia --project=tools/golden tools/golden/regate.jl --all
```

Every gate above, re-run in one command. A gate whose inputs are absent reports **skipped** rather
than green, because a gate that silently passes when it did not run is worse than one that fails.

| gate | measured |
|---|---|
| 2.x colfilt, bwareaopen, window reductions | **green** — all assertions pass |
| 0.1 the correlator alone | **green** — exact 100.0% |
| 0.2 the whole pipeline, 3072² | **green** — exact 81.8%, within step 98.7% |
| 0.3 the ITS_LIVE granule | **green** — all assertions pass |
| 3.x the stage ladder | **green** — 8 rungs, 8 green, 0 red |

**5 ran, 5 green, 0 red.**

The golden endpoints, on the reference's own captured inputs, after the coarse-radius fix:

| case | both | only jl | only ref | exact `dx` | median | p99 | corr | before |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| S2A Malaspina (**anchor**) | 586,090 | 6,643 | 10,528 | **92.66%** | 0.0625 | 0.418 | +0.997 | 92.67%, only_ref 10,531 |
| LC08 East Greenland | 1,660,132 | 46,761 | 55,928 | 55.12% | 0.111 | 1.375 | +0.996 | 55.14% |

The anchor holds and its coverage improves slightly — `only_ref` 10,531 → 10,528. Neither endpoint
moves materially, which is the expected result and worth stating plainly: the coarse-radius fix
corrected 1,809 of 85,556 coarse points, and a coarse point's radius only changes the answer where the
fine pass would otherwise have missed the peak. The fix is right on its own terms — it is exact against
the reference's own array where it was not before — and it is not what the endpoint gap is made of.

**Where that leaves the endpoint gap.** The ladder has now measured every setup stage as exact and both
correlation passes as coverage-identical, so the gap is neither the setup nor a failure to measure:

| stage | coverage | value agreement |
|---|---|---|
| setup (grid, radii, priors, coarse lattice) | — | **exact**, 5,475,584 and 85,556 points |
| coarse correlation | **identical**, 0 exclusive either way | 86.5% exact, 94.7% at corr ≥ 0.5 |
| coarse rejection `MC` | 98.47% agree, balanced (644 / 666) | — |
| fine correlation | **identical**, 0 exclusive either way | 71.9% exact, 100.000% of reference values on the 1/16 grid |
| endpoint | 46,761 jl-only, 55,928 ref-only | 55.1% exact |

Coverage is identical at both passes and differs by ~100,000 points at the endpoint, so **every
coverage difference is introduced after the fine pass** — by the rejection, the fill and the merge, not
by the correlator. And the base-level value residual is the `UInt8` quantization, measured above. Those
are the two remaining threads, and they are now separated rather than confounded.

## Gate 3 extended — levels 1–3, the fill, and the merge

All four chip-size levels captured with the stage trace and walked rung by rung:

```bash
for L in 0 1 2 3; do
  CAPTURE_STAGES=1 CAPTURE_STAGE_LEVEL=$L julia --project=tools/golden \
    tools/golden/intermediate.jl LC08_L1TP_009011 --run $((200+L)) --force
  julia --project=tools/golden -t 8 tools/golden/stages.jl LC08_L1TP_009011 --run $((200+L)) --all
done
```

| level | chip | grid | rungs | state |
|---|---:|---|---|---|
| 0 | 16 | 2344×2336 | 16 | **16 green** |
| 1 | 32 | 1172×1168 | 16 | **16 green** |
| 2 | 64 | 586×584 | 16 | **16 green** |
| 3 | 128 | 293×292 | 16 | **16 green** |

The stages this adds beyond the setup, all exact at every level:

| rung | reference | measured |
|---|---|---|
| 3.8 filter parameters, coarse and fine | `filtDisp` records | width 9/9, `FracValid` 0.32/0.32 and 0.41/0.41, iterations 2/2 and 3/3 |
| 3.13 fine rejection | `filtDisp kept` | 190,446 of 348,397 at level 1; the nulled field carries exactly that many |
| 3.14 fill median and its gate | `DxFM`, `MM` | **exact** — 5,475,584 at level 0, 1,368,896 at level 1 |
| 3.15 the three-pass fill mask | `MF` | **exact** — every point, every level |
| 3.16 / 3.17 merge and quantization | `ChipSizeX`, `Dx` | level 0 is **97.8%** on the 1/16 grid; levels 1–3 are **0.04%**, **0.02%**, **0.06%** on theirs |

Rung 3.17 is the independent confirmation of the closed coarse-level question: the base level's values
are quantized and the coarse levels' are not, on the reference's own arrays, because both sides replace
the measurement with a bicubic resize. `exact` is meaningful at level 0 and meaningless above it.

### A second real bug: the grid spacing read from the nodata margin

`_cell_centres` took the spacing as `x[1,2] - x[1,1]`. A production grid is zeroed wherever there is no
data (`testautoRIFT.py:394-403`), so on a scene whose first row and column are ocean both values are
`1.5` and the spacing reads as **zero** — the half-cell shift vanishes and every coarse node sits at its
cell's first point, half a cell from where `_undecimate_level` reads it back. On this case that left
99.8% of level-1 nodes 4 or 5 px from the reference's. `_grid_step` takes the mode of adjacent steps
instead.

### What rung 3.1 reports, and why it does not gate

The residual grid difference above the base level is a decision, not a defect. The reference resizes
with `INTER_AREA` and snaps to `round(x + 0.5) - 0.5`; AutoRIFT.jl decimates and shifts to the cell
centre. On a **rotated** grid — `x` varies 1 px per row here — the block mean is not the x-centre of the
column pair and the snap moves it a further half pixel: the cell centre is 3992.5 on both sides, the
reference's block mean is exactly 3993.0, its snapped node 3993.5. Following it would move the
correlation position without moving the read-back, which `src/multichip.jl` records as measuring worse
than matching neither half. The rung reports the offset as a fraction of a cell — **0.250, 0.375,
0.438** at strides 2, 4, 8, converging on half a cell — so a *change* in it means the decimation or the
read-back moved.

### Gate 4 re-measured: read the counts, not the percentage

The two fixes move the two cases in **opposite directions**, and the percentage is misleading on both.

| case | both | exact % | **exact count** | only jl | only ref |
|---|---:|---:|---:|---:|---:|
| S2A before | 586,090 | 92.66% | 543,071 | 6,643 | 10,528 |
| S2A after | 561,410 | **96.74%** | **543,108** | 14,193 | 35,208 |
| *change* | −24,680 | *+4.08 pt* | **+37** | +7,550 | +24,680 |
| LC08 before | 1,660,132 | 55.12% | 915,065 | 46,761 | 55,928 |
| LC08 after | 1,681,367 | 54.45% | **915,504** | **25,591** | **34,693** |
| *change* | +21,235 | *−0.67 pt* | +439 | **−21,170** | **−21,235** |

**S2A's 4-point gain is a coverage artifact and not an improvement.** The 24,680 points that left
`both` are exactly the 24,680 that appeared in `only_ref`, and the exact *count* moved by **+37 of
543,071** — 0.007%. AutoRIFT.jl stopped measuring 24,680 points it used to measure, they were
disproportionately ones it had got wrong, and removing them from the denominator raised the fraction
while the numerator stood still.

**LC08 is a genuine coverage improvement**, and it is unambiguous because *both* exclusive sets shrank:
`only_jl` −21,170 and `only_ref` −21,235, with 21,235 points moving into agreement. The percentage fell
0.67 points only because the denominator grew faster than the numerator.

So: `exact` as a percentage is not a safe headline when coverage moves. **The count and the two
exclusive sets are, and a fix that shrinks both exclusive sets is improving; one that grows them is
not.**

Open, and this is the next thing to measure: **why S2A lost 24,680 measurements.** The stage ladder has
run only on LC08, and the rung that would answer it — the coarse mask `MC2` restricting the fine search
— is the one stage the ladder does not yet cover. Capture S2A's levels and walk it there.

## Gate 3 extended again — the coarse mask, and S2A's lost measurements

The ladder now covers `MC2`, the stage that decides where the fine pass may look — and therefore the
one stage that can lose a measurement the correlator never gets to attempt. Three rungs
(`autoRIFT.py:702-724`): the go/no-go valid fraction, the dilation on the coarse grid, and the
expansion to the fine grid with the radius it produces.

```bash
CAPTURE_STAGES=1 CAPTURE_STAGE_LEVEL=0 julia --project=tools/golden \
    tools/golden/intermediate.jl S2A_MSIL1C_20200626 --run 200 --force
julia --project=tools/golden -t 8 tools/golden/stages.jl S2A_MSIL1C_20200626 --run 200 --all
```

| case, level | rungs | state |
|---|---|---|
| LC08, level 0 | 21 | **21 green** |
| S2A, level 0 | 19 | **19 green** |

### Where S2A's 24,680 lost measurements went

Not `MC2` — that accounts for ~4,600 points. Comparing level assignments directly localized it:

| chip | julia | reference | delta |
|---|---:|---:|---:|
| 0 (unresolved) | 505,997 | 484,982 | +21,015 |
| 24 | 554,187 | 554,112 | **+75** |
| 48 | 21,416 | 38,473 | −17,057 |
| 96 | **0** | 4,033 | −4,033 |

The base level agreed to 75 points. **Chip 96 produced nothing at all**, and chip 48 was short by
17,057 — so the whole loss was above the base level. Chip 96 returned in 0.2 s having correlated zero
coarse points, and the reason was arithmetic: its coarse points sat at x between **16,470 and 27,423**
on a **10,980 px** image. Every window fell outside the scene, nothing correlated, and the level was
dropped for a zero denominator.

**Two bugs, both mine, both in code added earlier in this session:**

`_grid_step` kept only *positive* steps. A step along a row moves `x` by the spacing times the cosine
of the grid's rotation — `8` on a near-axis-aligned Landsat grid and **`−1`** on this Sentinel-2 grid,
rotated near 90°. Keeping positives saw nothing but the jumps out of the zeroed nodata margin and
returned **10979**. The step is signed, and is taken only between points that both carry a coordinate.

`dilate_within` tested `d2 <= r2` where the reference tests `< BuffDistanceC`. On a lattice that
boundary is a whole ring: all **72** disagreeing coarse cells sat at distance *exactly* 8.0, which the
expansion multiplied to **30,084** fine points.

### The result, read as counts

| S2A | both | exact % | **exact count** | only jl | only ref |
|---|---:|---:|---:|---:|---:|
| original baseline | 586,090 | 92.66% | 543,071 | 6,643 | 10,528 |
| after the first two fixes | 561,410 | *96.74%* | 543,108 | 14,193 | 35,208 |
| **after these two** | 586,180 | 92.65% | **543,093** | **6,632** | **10,438** |

Both exclusive sets are now **below the original baseline** — `only_jl` 6,632 against 6,643 and
`only_ref` 10,438 against 10,528 — so the 24,680-point regression is gone rather than traded for a
percentage. Chip 96 is restored from 0 measured points to 46,960, and the level-assignment gap closes
from 21,015 to 3,806. `exact` sits at 92.65% against the baseline's 92.66%, which is the right shape:
the fixes were about coverage, and the count moved by +22.

`tools/ab` stage 2 is unchanged at 81.8% exact, 98.7% within one step, p99 0.0752. `Pkg.test()` passes
— 703,968 assertions — after the `FastGeoProjections` compat bump to `"0.1, 0.2"`, which was failing
at resolution on this branch and on `main`.

### One more matched-not-endorsed difference, measured

The reference's `MC2` expansion is `INTER_NEAREST`, which is **left-aligned**: coarse cell `k` covers
fine `(k-1)*stride+1 .. k*stride`, so the node at `k*stride` is the *last* point of its own cell. But
the radius it reduces for that node is a centred `filtWidth`-wide window, fine `k*stride-4 ..
k*stride+4`. So the reference gathers coherence evidence from fine 4..12 and applies it to fine 1..8 —
**offset by 3**. `_expand_coarse_mask` inverts `_cell_max_radius!`'s own assignment instead, so the
mask lands on exactly the points the evidence came from. Rung 3.11a reports the disagreement (28,032
points on LC08, split 14,016 each way, with both sides searching the same total of 2,615,872) rather
than gating on it.

## Gate 3 — S2A's coarse levels, and where the residual 3,806 points are

All three captured S2A levels walk clean:

| case, level | chip | rungs | state |
|---|---:|---|---|
| S2A, level 0 | 24 | 19 | **19 green** |
| S2A, level 1 | 48 | 19 | **19 green** |
| S2A, level 2 | 96 | 19 | **19 green** |

So every stage the ladder covers — the grid, the search-limit rewrite, the priors, the coarse
lattice/radii/priors, the filter parameters, the rejection, the fill median and its gate, the fill mask,
the merge and the quantization — is exact or reported-as-designed at every level of both cases.

### The residual is the coarse read-back, not the setup

Cross-tabulating the merged `ChipSizeX` point by point localizes what remains:

| reference \ julia | 0 | 24 | 48 | 96 |
|---|---:|---:|---:|---:|
| **0** | 478,350 | 68 | 6,116 | 448 |
| **24** | 34 | **553,857** | 221 | 0 |
| **48** | **7,839** | 261 | 30,124 | 249 |
| **96** | **2,565** | 1 | 648 | 819 |

The base level agrees on 553,857 of 554,112 — **99.95%**. The whole gap is the bold left column: 7,839
+ 2,565 = **10,404** points the reference resolves at chip 48 or 96 and AutoRIFT.jl leaves unresolved,
against 6,632 it resolves that the reference does not.

**It is not the admission gate.** Traced through the reference's own arrays at chip 96: its radius
survives at 44,426 nodes, its `M0` zeroing cuts that to 4,164, and 4,164 coarse measurements become
4,033 full-grid points. AutoRIFT.jl's radius survives at 44,289 — within 0.3% — and its bound gate
leaves 5,575, so it searches *more* coarse points and its level measures 11,424 on the full grid. It
then contributes only 1,516. **More raw coverage, fewer posted points**, which puts the loss in
`_undecimate_level`'s read-back rather than in anything upstream of it.

One genuine difference is identified and quantified there: AutoRIFT.jl *samples* the per-point chip
bounds at the decimated node, where the reference **dilates** its admission mask with
`colfilt(..., 0)` over `6/Scale` cells before inverting and resizing (`autoRIFT.py:531-544`).
Reproducing the reference's rule on the same input admits 17,955 nodes against sampling's 24,746, and
the two masks agree on **94.27%** of the 67,600. That is the right shape for a fix but the wrong sign
to explain this gap on its own — sampling admits *more* — so it wants measuring rather than assuming.

### The read-back measured, and what it leaves

Instrumenting the chip-96 level directly closes the chain:

| step | AutoRIFT.jl | reference |
|---|---:|---:|
| coarse nodes measured, on the 260² grid | **2,935** | **4,164** |
| after the read-back to the full 1040² grid | 46,960 | — |
| contributed to the merge | **1,516** | **4,033** |

`_undecimate_level` spreads each coarse node over its whole `stride²` cell — 2,935 × 16 = 46,960
exactly — and `_merge_level!` then keeps only the points no finer level claimed. The reference does the
same two things (`autoRIFT.py:857-866`: `INTER_NEAREST` of `M0`, then `idxRaw = M0 & (ChipSizeX == 0)`),
and its 4,164 coarse measurements yield 4,033 full-grid points because the finer levels have already
claimed roughly fifteen of every sixteen. So the spread and the gate both match in kind.

**The difference is upstream of both: the coarse pass measures 2,935 nodes where the reference measures
4,164.** That is 70%, and it is what the remaining ~2,500 chip-96 points reduce to. The admission gate
is ruled out — AutoRIFT.jl searches *more* coarse points there (5,575 against 4,164) — so the level is
searching a wider set and succeeding on fewer, which points at the coarse correlation or its rejection
on this scene rather than at any of the setup stages the ladder has verified exact.

### The chip-96 coarse pass, on the intersection of the two search sets

`DxC_rev0_L2` is on a 32×32 grid — the coarse pass of the chip-96 level, not its fine pass. Restricting
to the nodes both sides gave a positive radius:

| | value |
|---|---:|
| searched — julia / reference / **intersection** | 210 / 137 / **115** of 1,024 |
| on the intersection: both measured | **115** |
| only julia / only reference | **0 / 0** |
| exact | **85.22%** |
| median / p99 / max / bias | 0 / 6 / 9 / **+0** |

**The coarse correlator is not the problem at this level.** Coverage on the intersection is perfect —
115 measured by both, zero exclusive either way — and the bias is exactly zero. Both sides also measure
**100%** of what they search (210/210 and 137/137), so neither is failing to correlate. `exact` at
85.22% with a p99 of 6 px is the integer coarse pass choosing between competing peaks on a 96 px chip,
which is the expected regime.

Downstream of it, on the 32² coarse grid: `MC` agrees at 976 of 1,024 nodes and `MC2` at 747.

### Correcting an earlier reading in this file

The line above recording that AutoRIFT.jl "searches *more* coarse points there (5,575 against 4,164)"
and admits "27,744 level points against 4,164" **compared the wrong quantities**. 27,744 is the `MC2`
mask alone, before the radius gate; the radius-gated intersection — what the fine pass actually
receives — is **3,496** against the reference's **4,164**. So AutoRIFT.jl searches slightly *fewer*, at
84%, not 6.7× more.

That also disposes of the candidate fix. Rebuilding the admission mask the reference's way — bounds,
then `colfilt(..., 0)` over `6/Scale` cells, then `INTER_NEAREST` — reconstructs its `M0` to **94.27%**
and moves the fine-pass count from 3,496 to **5,033**, overshooting 4,164 in the other direction. Both
rules land within about 20% of the reference and neither is clearly right, so this is not one
identifiable rule to fix.

### Where that leaves the coarse levels

Every stage the ladder measures is exact or reported-as-designed, at four levels of LC08 and three of
S2A. The coarse correlator agrees with zero bias and identical coverage on the intersection. What
remains is a **composition of near-threshold decisions** — which nodes clear the admission bound, the
`MC` rejection, the dilation and the radius gate — where each factor individually agrees to within a few
percent and their intersection differs by more. The reference's own three-mask intersection at chip 96
is 4,164 from factors of 14,416, 32,160 and 44,426; an intersection is that much more sensitive to each
factor than any factor is on its own.

That is a real explanation and not a satisfying one, so it is worth stating what would settle it: a
per-node comparison of *which* of the three masks each side applies where, on the same 260² grid. The
arrays are all captured. It is not obviously worth the effort — the residual is 3,806 points of 596,618
(0.6%) on a case that already passes the gate — so the honest next step is to leave it and check whether
the other `hps` cases show the same 0.6% or something structurally different.

## Gate 3 — the correlator rung, on both element types

**The dtype pair is now a standing part of every rung that runs a correlator**, not a one-off probe. The
reference has two correlators — `arImgDisp_u` on bytes, `arImgDisp_s` on floats, separate C++ templates
— and production reaches the byte one (`uniform_data_type` quantizes to 256 levels before `runAutorift`,
`autoRIFT.py:359-384`). Running both separates a difference in what the correlator *computes* from a tie
the quantization created in the surface it computes *on*.

### The reference against itself is the floor

Handing the two templates the same information — the captured bytes, and those bytes widened to
`Float32` — gives a disagreement that belongs to neither implementation:

| comparison, chip 96 coarse pass, S2A | both | exact | max |
|---|---:|---:|---:|
| **reference `UInt8` vs its own `Float32`** | 137 | **98.54%** | **7 px** |
| AutoRIFT.jl vs reference, `UInt8` | 137 | 98.54% | 7 px |
| AutoRIFT.jl vs reference, `Float32` | 137 | 98.54% | 5 px |

AutoRIFT.jl sits **exactly on the floor**. The 1.46% is not attributable to it: the reference disagrees
with itself by the same amount on the same nodes, from which template ran. A rung reporting that figure
has found nothing.

**AutoRIFT.jl is bit-identical across the two element types** — every rung reports the same numbers on
`UInt8` and `Float32`, because `_prepare` preserves the element type and one code path handles both. So
the byte/float split is a property of the *reference* here, not a second code path to verify.

### Rung 3.7, across cases and levels

| case | level | chip | coarse nodes | only jl | only ref | exact | bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| LC08 | 0 | 16 | 35,142 | **0** | **0** | 86.54% | −0.092 |
| LC08 | 1 | 32 | 6,630 | **0** | 13 † | 90.86% | −0.521 |
| S2A | 0 | 24 | 11,285 | **0** | **0** | 99.21% | +0.004 |
| S2A | 2 | 96 | 137 | **0** | **0** | 99.27% | −0.037 |

† All 13 are the reference's **degenerate-chip corner**: 10 are exactly `-radius_x` and the other 3 are
one off it at a neighbouring radius. A constant chip carries no displacement information, and the
reference returns the search window's corner there where AutoRIFT.jl reports nothing —
already registered in the matched-not-endorsed table as a difference AutoRIFT.jl keeps deliberately.
The rung counts them separately rather than tolerating them, so the gate cannot come to reward
reproducing a fabricated value.

**Coverage is identical everywhere else** — zero exclusive nodes either way, on four level/case
combinations. `exact` ranges 86.5–99.3%, and both the low figures are LC08, whose 15 m panchromatic band
quantizes harder than S2A's.

### A harness correction

The 85.22% I reported for chip 96 was a harness artifact: it rebuilt the coarse grid with
`_coarse_points` instead of using the reference's captured `xGrid0C`. Rebuilding measures the setup as
well — which rung 3.6 already does separately, and finds exact — so the conflation read as 85% where the
correlator alone reads **99.27%**. Rung 3.7 uses the captured grid.

Ladder totals: **LC08 23 rungs green** at both traced levels, **S2A 21 green**.

## The Float32 capture, and the floor it establishes per level

```bash
CAPTURE_STAGES=1 CAPTURE_STAGE_LEVEL=1 CAPTURE_FLOAT32=1 \
  julia --project=tools/golden tools/golden/intermediate.jl LC08_L1TP_009011 --run 301 --force
julia --project=tools/golden -t 8 tools/golden/stages.jl LC08_L1TP_009011 --dtype-pair 201,301
```

`CAPTURE_FLOAT32` runs the same production pipeline with the pyramid reaching `arImgDisp_s` rather than
`arImgDisp_u`, so the reference's two correlator templates can be compared **against each other on
production imagery**. That difference belongs to neither implementation and bounds what any rung on
that level can be asked to achieve.

### `DataType` is not the lever, and the reason is worth keeping

The class has `uniform_data_type` with a `DataType == 1` branch that keeps a `Float32` field
(`autoRIFT.py:356-404`) — but the container's vendored driver **inlines** the `DataType == 0` arithmetic
instead of calling the method (`vend/testautoRIFT.py:449-481`, the same rescale-and-round written out).
So setting the attribute changes nothing, and `grep uniform_data_type` over the whole container finds
only the definition. Setting a flag and reporting success is precisely the silent no-op this harness has
been bitten by twice, so the arrays are widened in the `runAutorift` patch instead, with the pre-call
dtype asserted `uint8` and the post-call dtype asserted `float32`.

What this measures is **which template correlates the same values**. It does not measure what the
quantization cost — both runs see quantized values, because the driver has already overwritten `obj.I1`
by the time any patch on this method can see it. `tools/ab` stage 1 measures that separately on a
windowed float field.

### The floor is not the same at every level, and that is the finding

| case, level | chip | reference byte vs its own float | AutoRIFT.jl vs reference | attributable |
|---|---:|---:|---:|---|
| S2A, level 2 | 96 | **98.54%** exact, max 7 px | 98.54%, max 5 px | **nothing** |
| LC08, level 1 | 32 | **99.92%** exact, max 21 px | 90.86%, max 399 px | **~9 points** |

Coverage is identical in both pairs — zero exclusive nodes either way on 137 and 6,643 nodes — so the
floor is a value disagreement only.

**This is what the dtype pair is for.** At chip 96 on S2A the reference disagrees with itself as much as
AutoRIFT.jl disagrees with it, so the 1.46% there is quantization tie-breaking and there is nothing to
chase. At chip 32 on LC08 the floor is 99.92% and AutoRIFT.jl is at 90.86% — the shortfall is **real**,
not a tie, and it is the first coarse-level residual on this ladder that cannot be explained away. A
byte-only comparison reports 98.54% and 90.86% and gives no way to tell those two situations apart.

Rung 3.12 also lands with it: the coarse read-back's width-5 median is **exact** against the reference's
own `DxFM` on all 1,368,896 points, once fed the `DxF` state the fill loop left rather than the last
state on the grid — `DxF` is rebound again by the strong interpolation that *follows* this median, and
medianing that reports 25.7% differing. The rung selects its input by role and names the state it chose.

**LC08 level 1: 24 rungs, 24 green.** Next: the 9 points at chip 32 are now the only attributable
coarse-level residual on the ladder, and rungs 3.6, 3.10 and 3.12 are all exact there — so the remaining
candidates are the coarse pass's own peak selection at that chip size and the level's prior.

## Are the chip-32 coarse differences worth chasing? No — measured, not judged

The coarse pass exists to produce `MC` and then `MC2`, the mask that restricts the fine search. Whether
a differing coarse *value* matters is therefore answerable rather than a matter of taste: follow it to
the mask and see whether the fine pass searches a different set.

LC08 level 1, chip 32, 6,643 measured coarse nodes:

| step | measurement |
|---|---|
| coarse values differing | **606** of 6,643 (9.12%) |
| of those, **both sides reject** at `MC` | **353** (58.3%) — no consequence by construction |
| both keep | 112; they disagree about keeping **141** |
| `MC` agreement over the grid | 21,095 of 21,316 (**98.96%**) |
| **`MC2` agreement**, each side dilating its own `MC` | 21,308 of 21,316 (**99.96%**) — 8 cells |
| **fine pass search set** | **348,397 on both sides, symmetric difference 0** |

**The differences are inert.** 58% of them are rejected by both sides anyway; the rest survive the
dilation into 8 differing cells of 21,316; and after expansion to the level grid the fine pass searches
the **identical set** — not merely the same count, the same points. That 348,397 also matches the
reference's own final `SearchLimitX0_rev4_L1` exactly, so the agreement is with the reference's real
behaviour rather than between two reconstructions.

Where the 606 sit is consistent with that: median radius **18.5** against 8 elsewhere, median `|Δ|` 4 px
rising to 399 at the maximum. These are wide-window, weak-peak nodes where an integer coarse pass with no
subpixel refinement picks between competing maxima — and only 13.5% are railing against the search
limit, so it is not a window-size failure either. A coarse estimate exists to say *whether* a
neighbourhood is worth searching, not where the feature is; the fine pass re-measures every point it
admits.

**So this is noise in the strict sense that matters: it has no downstream consequence.** The 9-point
`exact` shortfall at chip 32 is not attributable to a defect, and the earlier note calling it "the first
coarse-level residual that cannot be explained away" was premature — it is explained, by following it to
the stage that consumes it rather than by arguing about its cause.

The general lesson, which applies to every rung above the base level: **a value comparison at an
intermediate stage is only as important as what the next stage does with it.** `exact` on `DxC` is a
diagnostic, not a gate, because the coarse pass's output is a decision and the decision agrees.

## Gate 4 complete — all five `hps` cases, ladder and endpoint

```bash
for cs in LC08_L1TP_009011 LC08_L1TP_062018 LC09_L1GT_215109 S2A_MSIL1C_20200626 S2B_MSIL1C_20200612; do
  CAPTURE_STAGES=1 CAPTURE_STAGE_LEVEL=0 julia --project=tools/golden \
    tools/golden/intermediate.jl "$cs" --run 200 --force
  julia --project=tools/golden -t 8 tools/golden/stages.jl "$cs" --run 200 --all
  julia --project=tools/golden -t 8 tools/golden/correlator.jl "$cs" --run 200
done
```

### The ladder generalizes

| case | rungs |
|---|---|
| LC08 Jakobshavn 009011 | **23 green** |
| LC08 East Greenland 062018 | **23 green** |
| LC09 Antarctic peninsula | **23 green** |
| S2A Malaspina | **21 green** |
| S2B Jakobshavn | **23 green** |

**Five of five, no reds**, on three sensors across Greenland, Alaska and the Antarctic Peninsula — and
three of these cases the ladder had never been run on before. The rungs were not tuned to LC08: every
one is a comparison against an array the reference itself produced, so a case-specific accident would
have shown as a red rather than as a pass.

### The endpoints, against the figures recorded before this work

| case | both | exact | only jl | only ref | *was exact* | *was only ref* | **only-ref change** |
|---|---:|---:|---:|---:|---:|---:|---:|
| S2A Malaspina | 586,090 | 92.65% | 6,632 | 10,438 | *92.67%* | *10,531* | **−93** (−1%) |
| LC08 East Greenland | 691,232 | 63.21% | 38,220 | 35,335 | *63.24%* | *35,638* | **−303** (−1%) |
| S2B Jakobshavn | 605,145 | 67.34% | 7,376 | 12,735 | *67.92%* | *17,970* | **−5,235** (−29%) |
| LC09 Antarctic | 460,136 | **68.42%** | 6,145 | 10,629 | *58.81%* | *27,088* | **−16,459** (−61%) |
| LC08 Jakobshavn | 1,681,367 | 54.45% | 25,591 | 34,693 | *55.14%* | *56,637* | **−21,944** (−39%) |

**`only_ref` fell in all five cases**, by 1% to 61%. That is the honest headline, and it is the right
statistic: it counts points the reference measured and AutoRIFT.jl did not, so it cannot be improved by
narrowing the denominator the way `exact` can. Coverage moved toward the reference everywhere.

`exact` is flat to slightly down on four cases and up 9.6 points on LC09 — and read against a rising
`both` that is the expected direction, since the newly agreeing points are the marginal ones a wider
coverage admits. LC09 is the one case where both moved the same way: +16,459 more shared points *and*
+9.6 points exact.

### What this settles about the residual

**Nothing — every `exact` figure in this section was measured against a wrong-sign y prior and is
superseded.** See "The fast-flow residual was a harness sign error" below. The readings this section
reached, in the order they were reached and refuted:

| reading | refuted by |
|---|---|
| the spread is `UInt8` quantization interacting with scene contrast | `Float32` moves `exact` by 0.15 points |
| the residual is a search-radius effect | radius sweep in `tools/ab` is flat; radius was a proxy |
| the residual is competing-maxima tie-breaking | the residual is spatially autocorrelated at +0.80 |

The `exact` spread across cases — 54% to 93% — was read as tracking the **scene** rather than the
code. It tracked the **prior**, which is largest on fast ice, which is why the spread looked
scene-dependent.

## The fast-flow residual was a harness sign error

The whole-case figures show disagreement concentrated on the fast-flow tongues rather than scattered
by contrast, and that shape is the finding. Three explanations were offered for it here and all three
were wrong; what settled it was refusing to accept a *spatially coherent* residual as tie-breaking.

### The discriminating test, which should have been run first

Tie-breaking is independent per point, so its residual must be spatially white. Measured on the base
level of LC08 Jakobshavn, `dx`:

| | lag 1 | lag 2 | lag 4 | lag 8 | lag 16 |
|---|---:|---:|---:|---:|---:|
| whole base level | +0.64 | +0.56 | +0.51 | +0.33 | +0.09 |
| fastest 10% | **+0.80** | +0.72 | +0.63 | +0.41 | +0.07 |
| fastest 10%, correlation ≥ 0.5 | **+0.71** | +0.57 | +0.48 | +0.31 | −0.04 |

Block means over 16×16 cells have sd **0.418** where white noise would give 0.032 — **13.1×** — and
72.7% are positive. It also survives a strong-peak gate, so no ambiguity argument reaches it. **A
residual that autocorrelates is a convention error, whatever its magnitude.**

### Root cause: `Dy0` is flipped before the correlator sees it

`arImgDisp_u`/`arImgDisp_s` set `Dy0 = -Dy0` as their first act (`autoRIFT.py:1058`, `:1231`),
converting the prior from cartesian-Y to matrix-Y before any chip is cut. `capture.py` records
`self.Dy0`, which is **pre-flip**, and `pointset_from_capture` passed it through unchanged — so
AutoRIFT.jl cut its chip `2 * Dy0` rows from where the reference cut its own.

The correlator flips the prior going *in* and the answer coming *out*. Undoing only the output is the
natural mistake, because `Dy` is what gets compared.

### The ladder that found it, on a 64×64 crop

The crop reproduces the bias in **14 s** against 31 min for the scene, which is what made stepping
through the chain affordable. Each rung is measured against an array the reference itself produced:

| step | measurement | verdict |
|---|---|---|
| raw `Dx_rev0_L0` vs final `out_Dx` | **100.00% equal** on 1,211,482 points | nothing downstream alters values |
| chip/window bounds, C `int()` vs `floor` | identical at all 2,188,984 points | not integer truncation |
| window width, surface size, zero-sample column | identical at every radius, 15 → 181 | not geometry |
| **ZNCC surface vs OpenCV `TM_CCOEFF_NORMED`**, byte-identical chip and window | **max diff 1.2e-6, same argmax**, 4 points | not the measure, not peak selection |
| the reference's own formula on that surface | fails to reproduce its own answer by **+3.2…+4.6 px** | the window it used differs |
| emulating its C++ index arithmetic | reproduces its answer to < 1 px | the chip rectangle differs |
| chip rectangle, reference vs AutoRIFT.jl | x **+0**, y **+28 = 2 × Dy0** | the prior's sign |

The surface comparison is the rung that made it unambiguous: **identical surfaces with a 3.9 px
disagreement** puts the defect downstream of correlation and kills the quantization, measure and
tie-breaking explanations at once.

### The fix, and what it recovers

`pointset_from_capture` negates `Dy0`. Worst 64×64 block, base level, against the reference's own raw
level-0 output:

| | exact | mean residual | p95 \|d\| |
|---|---:|---:|---:|
| `dx` before | 6.10% | +0.9043 | 3.875 |
| `dx` after | **99.61%** | **−0.0004** | **0.000** |
| `dy` before | 8.15% | −0.0158 | 1.875 |
| `dy` after | **99.55%** | **+0.0007** | **0.000** |

### Why it hid, and what that costs the ledger

Two properties, both of which this ledger should now treat as warning signs:

- **The error scales with the prior**, so it is absent on slow ice and largest on fast ice. That makes
  it look like a velocity-dependent physical effect, and it made the case-to-case `exact` spread look
  scene-dependent.
- **A y-axis error surfaces in `dx`.** A chip misplaced in y still correlates best at a similar
  vertical offset, so `dy`'s mean residual is −0.016 px while `dx` carries +0.90. Inspecting the axis
  whose sign is in question finds nothing.

**Every `exact`, `bias` and coverage figure measured through `pointset_from_capture` is superseded** —
that is all twelve optical and all eight radar cases, since every one of them reaches AutoRIFT.jl
through this function. Those rows have to be re-measured rather than edited. Coverage and `bias` are
affected too, not only `exact`: a misplaced chip changes which points are degenerate.

`AutoRIFT.jl was not at fault.` It was handed a wrong-sign prior and propagated it correctly, which is
also why no synthetic test caught this: the package's own tests never construct a prior with the
reference's sign convention. `REFERENCE.md` records the convention itself, all four flip sites, and
the three tests that settle one.

## Superseded: the residual is a search-radius effect, not quantization

**Both claims in this section are wrong.** The `Float32` measurement is sound and is the reason the
quantization explanation was dropped; the search-radius conclusion that replaced it is not — radius is
a proxy for prior magnitude, and a radius sweep in `tools/ab` at fixed prior is flat (84.8% at radius
20 against 87.5% at 120, bias 0.0000). Kept for the `Float32` rows and as a record of the wrong turn.

```bash
julia --project=tools/ab -t 8 tools/golden/compare_figures.jl LC08_L1TP_009011_20200703 --run 200
julia --project=tools/ab -t 8 tools/golden/compare_figures.jl LC08_L1TP_009011_20200703 --run 301
```

The whole-case figures show the disagreement concentrated on the fast-flow tongues, not scattered by
contrast. That is the wrong shape for a quantization tie-break, and running the dtype pair over the whole
case settles it. LC08 Jakobshavn, `dx`, run 200 against run 301:

| comparison | shared | exact | p99 |
|---|---:|---:|---:|
| AutoRIFT.jl vs reference, `UInt8` (production) | 1,660,132 | 55.12% | 0.9375 |
| AutoRIFT.jl vs reference, `Float32` | 1,659,941 | **55.27%** | 0.9375 |
| reference `UInt8` vs its own `Float32` (floor) | 1,715,662 | 95.00% | 0.0072 |
| AutoRIFT.jl `UInt8` vs AutoRIFT.jl `Float32` | 1,595,699 | **100.00%** | 0.0000 |

**Widening to `Float32` moves `exact` by 0.15 points**, and the profile across speed deciles is unchanged
to a tenth of a point. So the quantization is not what the residual is made of. The last row is why the
lever is worth having: AutoRIFT.jl is bit-identical across element types, so the two rows above it differ
only in the reference, and the 95.00% floor is the reference disagreeing with itself.

### It is the search radius, at fixed speed and fixed chip size

Two confounds have to come out first. Fast points preferentially get a coarser chip (chip 16 covers 94.8%
of the slowest speed decile but 52.2% of the ninth), and a coarse level is unquantized, where `exact` is
0% by construction. Holding **both sides at chip 16** removes both, leaving 1,194,613 points at 76.60%:

| speed decile (median px) | 0.06 | 0.14 | 0.44 | 0.79 | 0.96 | 1.13 | 1.38 | 1.96 | 3.10 | 6.60 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exact | 99.7 | 99.3 | 97.4 | 93.0 | 88.8 | 81.1 | 66.4 | 61.5 | 51.5 | **27.8** |
| mean residual | −0.000 | −0.000 | −0.003 | −0.004 | −0.003 | −0.005 | −0.002 | +0.005 | +0.011 | **+0.118** |

The speed dependence survives at one chip size, and the fastest decile carries a **one-sided** residual
where every other decile is centred on zero. Within that decile, binning by search radius separates the
two: agreement falls from 90.7% at radius 15 to 13.4% at radius 72, and the mean residual rises with it.
Speed is the proxy; **radius is the variable**.

| base-level radius | 0–8 | 8–12 | 16–20 | 24–32 | 40–56 | 56–80 | 80–300 |
|---|---:|---:|---:|---:|---:|---:|---:|
| same integer sample | 97.8% | 93.0% | 97.9% | 93.8% | 90.8% | 80.7% | **71.1%** |
| \|Δ\| ≥ 1 px | 0.06% | 0.63% | 0.08% | 0.17% | 0.45% | 3.75% | **13.6%** |
| mean residual | −0.0004 | −0.0029 | +0.0094 | +0.0233 | +0.0827 | +0.1316 | **+0.4171** |

Everything grows monotonically with radius, and at the widest windows one point in seven picks a
**different integer sample** — a different match, not a rounded one.

### What it is not

Each of these is measured, and each rules out a candidate that the fast-flow shape would otherwise fit:

- **Not reachability.** The reference's answer lies inside the window AutoRIFT.jl searched at
  **0.000% outside on all ten speed deciles**, and neither side rails against its search limit at any
  decile (0.000% both). The p95 of `|ref − prior| / radius` never exceeds 0.18, so both sides find their
  peak well inside a window neither one exhausts. A radius too small, a prior too far off, or a coarse
  mask restricting the wrong region would all show here, and none does.
- **Not the sub-pixel refinement kernel.** Where the peak falls inside a pixel is independent of how fast
  the ice moves, so a kernel defect cannot produce a monotonic speed or radius profile — and it would not
  spare the 0–8 radius bin at 97.8%. The residual is also 98.3% on the 1/16 lattice with a median of
  exactly zero, which is a refinement agreeing with the reference's own quantization step.
- **Not the coarse pass's decision.** Already measured separately: at chip 32 the fine pass searches the
  **identical** 348,397 points on both sides, symmetric difference 0.

### The harness bug this was hiding behind, and why the earlier reading was wrong

`cached_run`'s key was `(name, kw, size(grid.x))` — which is identical for run 200 and run 301, because a
`CAPTURE_FLOAT32` capture differs from an ordinary one **only** in `in_I1`'s element type. So a `Float32`
comparison silently reused the `UInt8` field and reported the quantization's effect as zero by
construction. `eltype(k.arrays["in_I1"])` is now in the key.

The earlier conclusion was not built on that bug, but on a narrower measurement: the dtype pair had been
run on one traced level at a time, where the floor at chip 96 (98.54%) does equal AutoRIFT.jl's agreement
and the reading "nothing attributable" is correct *for that level*. Generalizing it to the whole case was
the error. The lesson is the one this ledger already applies elsewhere: **a per-level floor does not
compose into a whole-case one**, because the levels are weighted by how many points each resolves.

# Phase 2 — the remaining seven optical pairs

Four L7/L8×L7 pairs use `_wallis_filter_fill` and three L4/5 pairs use `_wallis_filter` + `_fft_filter`.
Both filters run in `process.py` on the **native** scenes, before geogrid, so neither is visible at the
`runAutorift` boundary the phase-1 captures take.

## Step 1 — the reference's filtered scenes are recoverable

```bash
CAPTURE_STAGES=1 CAPTURE_STAGE_LEVEL=0 julia --project=tools/golden \
  tools/golden/intermediate.jl LE07_L1TP_061018_20120428 --run 200 --force
```

`process.py:312-336` dispatches the filter on platform and writes `Float32` GeoTIFFs to
`Path.cwd()/'filtered'` (`create_filtered_filepath`, `:252`). The container runs in the mounted directory
(`intermediate.jl` does `cd /home/ubuntu/work`), so they persist — the older L7 runs lack them only
because they predate that path.

| | measured | state |
|---|---|---|
| filtered scenes present | **4**: two `Float32` bands, two `uint8` zero masks | **green** |
| reference band | `LE07_..._061018_..._B8.TIF`, 15101×16721, 252,503,821 finite | green |
| secondary band | `LE07_..._060018_..._B8.TIF`, 15721×17221, 270,731,341 finite | green |
| zero masks | same shapes, 89,606,614 and 107,932,572 set | green |
| recorded in `call1.json` | `filtered_dir` plus shape, dtype, finite/nonzero counts and geotransform per file | green |

**The shapes are the finding.** The filtered reference is 15101×**16721** while `in_I1` at the
`runAutorift` boundary is 15101×**11338** — geogrid crops to the common overlap *after* filtering, which is
exactly what `process.py:312`'s `FIXME` describes and why the filter cannot be validated from the
correlator's inputs alone. The filtered scenes are the input a staged comparison needs, and they are now
addressable from the manifest rather than by globbing a directory.

An `hps` pair writes no `filtered/` at all, since it is filtered inside `autorift()`.
`record_filtered_scenes` records `filtered_dir: null` in that case rather than staying silent, so a reader
distinguishes "no filtered scenes" from "not looked for".

## Step 2 — `wallis_gapfill` against `_wallis_filter_fill`, masks exact

```bash
micromamba run -n arift-ref python tools/python_ref/gen_fixtures.py wallisfill
julia --project=. -e 'using TestEnv; TestEnv.activate(); include("test/preprocess.jl")'
```

Nine fixtures — three scenes (scan-line stripes, a wide margin plus interior gaps, a flat patch) × three
`(width, cutoff)` settings — each recording every deterministic array the filter builds. **213 assertions
pass.**

| what | measured | state |
|---|---|---|
| `invalid_data` | **exact**, all 9 cases | green |
| `potential_data` (within 30 px of data) | **exact**, all 9 | green |
| `missing_data` (gaps grown by `buff`) | **exact**, all 9 | green |
| `zero_mask` (the mask the pipeline writes to disk) | **exact**, all 9 | green |
| the fill *set* | deterministic and seed-independent; two seeds agree on `v`, disagree on values | green |
| the local standard deviation | **differs by up to 183** in the input's units | by design, three causes below |

### Every mask is exact; the statistics differ, in three ways and all deliberate

That split is the useful result: the filter's *decisions* — which pixels are gaps, which are reachable,
which are filled, which stay masked — agree exactly, and those are what propagate, because a filled pixel
is marked valid and so stops masking out its neighbours. The *values* differ for three separate reasons,
each a choice the reference made:

1. **The gap zeros are in its statistics.** It detects gaps as `isclose(image, 0)` and then computes the
   local mean and standard deviation over the raw array anyway, so a window touching a gap is normalized by
   a spread the gap itself created. AutoRIFT.jl excludes them via the mask — the difference
   `wallis_gapfill`'s docstring already records.
2. **`E[x²] − E[x]²`, clipped at zero** (`_preprocess_filt_std`), against AutoRIFT.jl's about-the-mean
   form. `REFERENCE.md` measures the reference's median error at 0.54 and AutoRIFT.jl's at 1.5e-6 against
   an exact `Float64` truth.
3. **Mixed border modes inside one filter — new, and not previously recorded.** `_remove_local_mean` uses
   `BORDER_CONSTANT` while `_preprocess_filt_std` uses `BORDER_REFLECT`, so the numerator and the divisor
   of a single Wallis call disagree about the border. Unmasked statistics with the accurate variance still
   differ by 54.7 for this reason alone.

The testset asserts the *shape* of that disagreement rather than tolerating it: masks exact, values not,
and AutoRIFT.jl's divisor finite everywhere it has a neighbourhood. A change that made the values agree
would mean one of the three had been silently adopted.

### A harness error worth recording, because it looked exactly like a defect

The first comparison reported 4,512 pixels differing on `potential_data`. That was the probe passing
`dilate_within(invalid, 30)` where the reference's `distanceTransform(invalid) < 30` measures each *invalid*
pixel's distance to the nearest *valid* one — so the Julia form dilates the **valid** set. With the polarity
right it is exact on all nine cases. The test now asserts the polarity explicitly, since the wrong one runs
happily and produces a plausible number.

## Step 3 — the four L7 / L8×L7 pairs, ladder and endpoint

```bash
CAPTURE_STAGES=1 CAPTURE_STAGE_LEVEL=0 julia --project=tools/golden \
  tools/golden/intermediate.jl <case> --run 200 --force
julia --project=tools/golden -t 8 tools/golden/stages.jl <case> --run 200 --all
julia --project=tools/golden -t 8 tools/golden/correlator.jl <case> --run 200
```

The reference filters these on the native scenes, so the arrays at the `runAutorift` boundary already
carry its `_wallis_filter_fill` output. That makes the fill the reference's and puts only the correlator
chain under test — the decision recorded in this phase's Context, and what lets the same 23-rung ladder run
unchanged.

### The ladder

| case | rungs | note |
|---|---|---|
| `LE07_L1TP_061018_20120428` | **13 green** | base level skipped; both sides agree |
| `LE07_L1TP_061018_20130314` | **23 green** | |
| `LE07_L1TP_063018_20040810` | **23 green** | |
| `LC08_L1TP_060018_20130330` × `LE07` | **23 green** | |

**Four of four, no reds**, including the correlator rung on both element types. Nothing in `stages.jl`
needed a new rung, which is the result the plan was checking for.

### The endpoints

| case | both | exact | exact n | only jl | only ref | bias dx / dy | corr |
|---|---:|---:|---:|---:|---:|---:|---:|
| `LE07_..._20120428` | 106,232 | **0.00%** † | 0 | 11,000 | 32,552 | +0.008 / +0.008 | +0.891 / +0.849 |
| `LE07_..._20130314` | 713,312 | 59.83% | 426,805 | 24,893 | 47,474 | −0.035 / −0.010 | +0.920 / +0.917 |
| `LE07_..._20040810` | 918,395 | 50.18% | 460,854 | 55,935 | 69,334 | −0.010 / −0.004 | +0.973 / +0.975 |
| `LC08_060018` × `LE07` | 672,901 | 58.49% | 393,554 | 26,607 | 49,674 | +0.011 / +0.011 | +0.955 / +0.931 |

Bias is under 0.035 px on every axis of every case and correlation is 0.85–0.98, which is the shape the
`hps` cases show. Three of the four sit at 50–60% exact, comparable to the LC08 `hps` pairs (54–63%).

† **`exact = 0.00%` on the first case is correct by construction, not a failure.** Its base level is
skipped — 102 of 20,525 coarse points survive, 0.50% against a 1% cutoff — so the reference resolves *only*
chips 32 and 64, and 0.03% of its own `dy` values land on any 1/N grid. Above the base chip size both
implementations replace the measurement with a bicubic resize, so nothing is quantized and `exact` is the
wrong statistic there. Bias and correlation are the gate, and both pass.

### Two harness bugs, both of which read as package defects

**The filter-parameter rung indexed `filtDisp` records at `2L + 1`.** That assumes every level runs a
coarse *and* a fine pass. A level below `CoarseCorCutoff` `continue`s out (`autoRIFT.py:704-706`) and
contributes one record, shifting every later level's index — so on the first L7 pair the rung compared the
coarse filter against fine parameters and reported `frac 0.41/0.32, iterations 3/2`. Selecting by grid
shape alone then *regressed a green case*: level 0's coarse grid is 293×292 on the golden Landsat pair and
level 3's fine grid is 293×292 too, both being the full grid over 8. The search is now bounded to the
records between this level's own correlator calls.

**The sign selector compared `exact`, which is zero for both signs when nothing is quantized.** The
comparison tied at zero, `>=` kept `+1`, and the report showed `dy` at sign `+` with a **−0.849
correlation and a 0.761 px bias** — a flipped axis presented as a measured choice. Correlation is now the
discriminant, because it is what a sign error destroys and it stays defined on an unquantized level. The
same run then reports `dy` at sign `−` with bias **+0.0076**.

Both were invisible on the five `hps` cases, whose base levels always resolve. A case class that exercises
a different branch is what found them.

## Step 4 — the metadata route is refuted, and the plan's fallback applies

```bash
CAPTURE_STAGES=1 CAPTURE_STAGE_LEVEL=0 julia --project=tools/golden \
  tools/golden/intermediate.jl LT05_L1TP_060018_19851028 --run 200 --force
grep -E "track angle|banding" <run-dir>/capture.log
```

The plan proposed reading the scan geometry from the MTL rather than deriving it from pixels, deleting four
of the five OpenCV calls. **The measurement says no.** Recorded because it was a good idea that a cheap
check disposed of, and because the reason generalizes.

| what | value |
|---|---|
| reference's logged angles, scene 1 | along **71.60°**, cross **−20.25°** |
| reference's logged angles, scene 2 | along **75.49°**, cross **−16.37°** |
| angles from `CORNER_*_PROJECTION_*_PRODUCT` | along **0.00°**, cross **90.00°** |
| `ORIENTATION` in the MTL | `"NORTH_UP"` |

`ORIENTATION = NORTH_UP` is the explanation: an L1T product is resampled north-up, so *both* the projected
and the lat/lon corner sets are the axis-aligned bounding box of the raster, and every slope from them is 0
or ±90. What `_fft_filter` measures is the **valid-data region inside** that raster — the rotated swath a
north-up product contains surrounded by fill. The rotation exists because the acquisition track does not
line up with map coordinates, and at high latitude no scene's does.

So the corner fields cannot supply it. Whether *some* metadata can is a separate question, and the next
section answers it.

Also confirmed by line order: `apply_landsat_filtering` is called at `process.py:477` and the reprojection
logs at `:482`, so the filter sees the native scene. The rotation is not an artifact of reprojection.

### Correcting the above: the *corner* fields are refuted, metadata is not

The paragraph above concluded too broadly from too narrow a check. The rotation is **orbit geometry** — the
acquisition track does not line up with map coordinates, and at high latitude no scene does — so the place
to look is not the corners but the orbit. Every Landsat L1 product ships an `_ANG.txt` beside the MTL
carrying `EPHEMERIS_ECEF_{X,Y,Z}` at 1 s spacing, which is the satellite's actual trajectory.

**The heading must be projected into the scene's own CRS, not left in true-north terms.** A slope measured
on a UTM raster is a *grid* bearing, and grid north departs from true north by the meridian convergence —
3.2° at 3.7° off the central meridian at latitude 60, and much more near the poles, which is where this data
lives. Transforming two consecutive ECEF positions straight into the raster's CRS and differencing gets that
right without deriving a convergence formula, because the projection applies it.

| scene | orbit along | orbit cross | reference along | reference cross | Δ along | Δ cross |
|---|---:|---:|---:|---:|---:|---:|
| `LT05_L1TP_060018_19851028` | 69.725° | **−20.275°** | 71.602° | **−20.245°** | −1.877° | **−0.030°** |
| `LT05_L1GS_061018_19860123` | 73.589° | **−16.411°** | 75.490° | **−16.370°** | −1.901° | **−0.041°** |

**Cross-track agrees to 0.03–0.04° on both scenes.** That is the confirmation: the orbit, projected into the
scene CRS, is measuring the same physical direction the reference recovers from pixels. An unprojected
true-north heading gives 71.63° for both scenes — which appeared to match scene 1's along-track to 0.03° and
was a coincidence, since the two scenes' grid bearings differ by 3.9° while their true bearings differ by
0.02°.

**Along-track is off by a constant −1.89°** on both scenes, same sign and magnitude. That is not noise: the
reference takes `nanmax` of *two* edge slopes per axis (`autoRIFT.py:143-151`), so its along-track is biased
toward whichever edge is worse-conditioned, while its cross-track happens to land on the good edge. The
orbit's two axes are exactly perpendicular by construction; the reference's are not — −20.245° against
71.602° − 90° = −18.398°, off by 1.85°, which a real cross-track direction cannot be.

So the metadata route is viable and better: `_ANG.txt` carries the geometry once projected, and the MTL
corners do not carry it at all. What it cannot do is *match* the reference's along-track, because that value
is a biased estimator rather than a measurement of anything.

**That is the trade this step exists to surface, and it is the plan's pre-committed decision point.**
Agreement is the current objective, so the derived route is what a matched filter must use — and the four
primitives come back with it, including the `minAreaRect` angle-convention hazard. The orbit route is
recorded in `tools/golden/README.md`'s matched-not-endorsed table as the more correct alternative, with the
measurement above as its justification, to be adopted once agreement is established. `mtl.jl` keeps the
ephemeris reader so that decision is one call away rather than a re-derivation.

### A second finding from the same log: the filter is a no-op on this pair

```
Along track angle is 71.60 degrees
Cross track angle is -20.25 degrees
Power along flight direction (2216) does not exceed banding threshold (500). No banding filter applied.
```

That message is misleading and the arithmetic says why. The condition is
`((sA/sB >= 2) | (sB/sA >= 2)) & ((sA > 500) | (sB > 500))` (`autoRIFT.py:227`), and 2216 > 500 satisfies
the second clause — so it is the **ratio** clause that failed: the two band powers are within a factor of
two of each other. The message blames the threshold for a decision the ratio made.

Both scenes of this pair decline the band-reject, so the filter returns the *clamped* image — `±3` clipped,
NaN→0 — unfiltered. That is still not a pass-through, which is why the branch has to be asserted rather
than inferred from output agreement: a filter that no-ops on both sides agrees trivially.

`gen_warpaffine` is generated and stands regardless, since the band masks are rotated on either route: 12
cases across two dimensions and six angles, recording the bilinear rotation *and* the `== 1` selection the
filter actually consumes.

### Step 4, green: what it established and what Step 5 must do

| # | what | measured | state |
|---|---|---|---|
| 4.1 | `warpAffine` of a `getRotationMatrix2D`, 12 cases | fixtures generated, recording the bilinear rotation *and* the `== 1` selection the filter consumes | **green** |
| 4.2 | the pixel-derived footprint chain, 4 synthetic cases | `connectedComponentsWithStats`, contour moments, `minAreaRect` angle with its post-4.5 range recorded, quadrant rotation, four distance-transform extremes | **green** |
| 4.3 | the reference's real geometry on an L4/5 scene | reproduced **exactly**: along 71.6019°, cross −20.2454°, matching its own log to 4 decimals | **green** |
| 4.4 | the orbit route, projected into the scene CRS | cross-track to **0.03°**; along-track off by a constant **−1.89°**, attributed to `nanmax` | **green** |

4.3 is the one that matters for Step 5: a standalone script reproduces every intermediate of the geometry
half — largest region, centroid, `minAreaRect` angle and size, the four quadrant corners, and both angles —
and lands on the reference's logged values to four decimals. So the derived route is now *specified* by
arrays on disk rather than by a reading of Python, and Step 5 gates against those.

**Retracted, and the retraction is the useful part.** This entry originally claimed the filter *does*
band-reject on this scene, with `sA = 511`, `sB = 2373`, ratio 4.64. That was **two wrong inputs agreeing**:
the probe read the scene from `filtered/`, which is already Wallis-filtered, and applied `_wallis_filter` to
it a second time. Both sides then saw a doubly-filtered field, and their agreement looked like a pass.

Run from the *native* scene through `process.py`'s own `apply_fft_filter`, the numbers are 1187 and 2216,
ratio **1.87 < 2**, and the filter **declines**. The Wallis output's standard deviation says which input is
right without any reference to the filter: 0.770 from the native scene, matching the on-disk filtered scene's
0.771, against 0.972 for the doubly-filtered one.

The lesson is the one this ladder exists for: *a stage fed the wrong input can agree*. `filtered/` is the
filter's **output**, so it is what a filter comparison is gated against and never what it is fed.

**Decision for Step 5, taken here rather than during implementation:** reproduce the derived route, because
agreement is the objective and the reference's along-track is what it is. The orbit route is registered in
`README.md` as the more correct alternative with the measurement above as its justification, to be adopted
once the L4/5 pairs agree. `tools/golden/mtl.jl` keeps the ephemeris reader so that switch is one call.

## Step 5 — the destripe filter, exact against the reference's own chain

```bash
# dump the reference's chain from the native scene, inside the container
python /opt/capture/ds10.py /vsis3/usgs-landsat/.../LT05_..._B2.TIF <out>
julia --project=tools/golden -t 8 <compare>
```

`Destripe` is new in `src/types.jl` and `src/preprocess.jl`, taking the two scan angles as arguments so the
filter is a pure function of image and geometry.

| what | measured | state |
|---|---|---|
| the clamp to ±3, NaN→0 | **exact**, 65,140,551 pixels | green |
| `warpAffine` band masks, 12 fixture cases | **exact**, both parities, all six angles | green |
| band powers on the real scene | julia along **2216** / cross **1187**; reference the same two numbers | green |
| the branch | julia **DECLINE** at ratio 1.867; reference **declined** | green |
| **the filter output** | **exact — 100.0000% on all 38,707,144 valid pixels** | **green** |

### Three details the reference gets wrong or unusually, each found by measurement

**OpenCV interpolates in fixed point, and on a 0/1 mask that changes the answer.** `warpAffine` rounds each
source coordinate to 1/32 (`INTER_BITS = 5`) before splitting it into an integer part and a weight, so a
coordinate within 1/64 of an integer snaps onto it and the interpolated value is *exactly* 1 where a
`Float64` computation gives 0.993903. The consumer tests `== 1` (`autoRIFT.py:225-226`), so that is the
difference between a cell being rejected and kept: ignoring it missed **60 of 3,720** selected cells on an
odd-sized mask, all on the band's rotated edge — precisely where a band-reject decides how much it rejects.

**The rotation direction cannot be checked on counts.** `getRotationMatrix2D` builds the forward map and
`warpAffine` samples the source at `M ⋅ dst`, so the source offset carries the angle's own sign, not its
negation. Getting it backwards rotates the band the other way — and a rotation is area-preserving, so the
*count* of selected cells is identical: 3,804 under both signs, with 3,120 of them in different places. Only
the positions catch it.

**The reference's log labels are swapped.** `rotation_a` is built from `cross_track` (`:208`), so `sA` is the
cross-track power — but it prints as `Along track power` (`:226-227`). The two logged numbers are right and
their names are exchanged. Harmless to the branch, which tests a ratio and a maximum, and to the band chosen,
which is the larger sum — but it would send a reader hunting a swap that is not in the arithmetic.

### And the filter declines on this pair

Ratio 1.867 against a threshold of 2, so the clamped image is returned. That is *not* a pass-through — the
±3 clamp still applies — which is why the branch is asserted explicitly rather than inferred from output
agreement. A filter that no-ops on both sides agrees trivially, and this gate can tell that from real
agreement because it checks the two power sums and the branch as well as the output.

## Step 6 — the three L4/5 pairs, ladder and endpoint

| case | rungs | note |
|---|---|---|
| `LT05_L1TP_060018_19851028` × `LT05_L1GS_061018` | **23 green** | |
| `LT04_L1TP_063018_19880611` × `LT04_..._19880627` | **23 green** | |
| `LT05_L1GS_001013_19920425` × `LT05_..._19920628` | **13 green** | base level skipped; both sides agree. The `P000` case |

| case | both | exact | exact n | only jl | only ref | bias dx / dy | corr |
|---|---:|---:|---:|---:|---:|---:|---:|
| `LT05_060018` | 124,397 | 27.09% | 33,702 | 17,113 | 33,460 | −0.005 / −0.008 | +0.993 / +0.977 |
| `LT04_063018` | 272,917 | 55.95% | 152,693 | 23,374 | 25,681 | **+0.0004 / +0.0004** | +0.982 / +0.955 |
| `LT05_001013` (`P000`) | 17,764 | **0.00%** † | 0 | 1,940 | 2,252 | −0.140 / −0.034 | +0.987 / +0.986 |

Bias is under 0.01 px on the two well-populated pairs and correlation is 0.95–0.99 on all three. `LT04` is
the cleanest golden case measured so far by bias — **0.0004 px on both axes**.

† `LT05_001013` is the `P000` case, the one pair in the whole set exercising the **uncropped product
schema**: `process.py` crops only products with at least one valid pixel, and cropping is what adds the time
axis. It is also the smallest by an order of magnitude — 17,764 shared points against 124k–273k — and its
base level is skipped, so every point is coarse and unquantized, which is why `exact` is 0 by construction.
Its −0.14 px `dx` bias is the largest in the optical set and is worth attention only in proportion to its
size: 17,764 points against 4.4 million across the twelve pairs. Recorded rather than chased.

`LT05_060018`'s 27.09% is the lowest `exact` of the twelve. Its base level *does* resolve, so unlike the
skipped-level cases the figure is comparable — and it is the pair whose destripe filter **declines** on both
scenes, so both sides correlate a clamped-but-unfiltered field. Bias is nonetheless −0.005 px and
correlation +0.993, so the two agree about position and disagree about the last quantization step.

## Step 7 — all twelve optical pairs, and no slide

```bash
julia --project=tools/golden tools/golden/regate.jl --all
julia --project=. -e 'import Pkg; Pkg.test()'
```

| gate | measured |
|---|---|
| 2.x colfilt, bwareaopen, window reductions | **green** |
| 0.1 the correlator alone | **green** — exact 100.0% |
| 0.2 the whole pipeline, 3072² | **green** — exact 81.8%, within step 98.7% |
| 0.3 the ITS_LIVE granule | **green** |
| **3.opt the stage ladder on every optical case** | **green — 12/12** |
| 3.x the stage ladder, base case | **green** — 23 rungs |

**6 ran, 6 green, 0 red.** `Pkg.test()` passes.

`regate.jl` now runs the ladder over **all twelve** optical cases rather than one. That is deliberate: a rung
is only as good as the case classes it has met, and both harness bugs in step 3 — the `filtDisp` index and
the sign selector — were invisible on the five `hps` cases and surfaced only on an L7 pair whose base level is
skipped.

### The twelve optical pairs

| # | case | filter | rungs | both | exact | only jl | only ref | bias dx | corr dx |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | S2A Malaspina | `hps` | 21 | 586,090 | 92.65% | 6,632 | 10,438 | +0.001 | +0.997 |
| 2 | LC09 Antarctic | `hps` | 23 | 460,136 | 68.42% | 6,145 | 10,629 | — | +0.976 |
| 3 | S2B Jakobshavn | `hps` | 23 | 605,145 | 67.34% | 7,376 | 12,735 | — | +0.997 |
| 4 | LC08 East Greenland | `hps` | 23 | 691,232 | 63.21% | 38,220 | 35,335 | — | +0.985 |
| 5 | LC08 Jakobshavn | `hps` | 23 | 1,681,367 | 54.45% | 25,591 | 34,693 | — | +0.997 |
| 6 | `LE07_..._20130314` | `wallis_fill` | 23 | 713,312 | 59.83% | 24,893 | 47,474 | −0.035 | +0.920 |
| 7 | `LC08_060018` × `LE07` | `wallis_fill` | 23 | 672,901 | 58.49% | 26,607 | 49,674 | +0.011 | +0.955 |
| 8 | `LE07_..._20040810` | `wallis_fill` | 23 | 918,395 | 50.18% | 55,935 | 69,334 | −0.010 | +0.973 |
| 9 | `LE07_..._20120428` | `wallis_fill` | 13 | 106,232 | 0.00% † | 11,000 | 32,552 | +0.008 | +0.891 |
| 10 | `LT04_063018` | `fft` | 23 | 272,917 | 55.95% | 23,374 | 25,681 | **+0.0004** | +0.982 |
| 11 | `LT05_060018` | `fft` | 23 | 124,397 | 27.09% | 17,113 | 33,460 | −0.005 | +0.993 |
| 12 | `LT05_001013` (`P000`) | `fft` | 13 | 17,764 | 0.00% † | 1,940 | 2,252 | −0.140 | +0.987 |

† base level skipped on both sides, so every point is coarse and unquantized and `exact` is 0 by
construction. Bias and correlation are the gate there.

**Every stage of every pair is exact or reported-as-designed.** `|bias|` is under 0.035 px on eleven of
twelve and correlation is 0.89–0.997 across the set. `exact` spans 27–93% and tracks the *scene* rather than
the code: it is highest on the two Sentinel-2 pairs and lowest where the base chip size resolves fewest
points. The cause is the search radius, not the `UInt8` quantization — see "The residual is a
search-radius effect, not quantization".

### What this phase added

| | |
|---|---|
| optical pairs validated | 5 → **12 of 12** |
| new preprocessing filters | `Destripe`, exact on 38,707,144 pixels |
| new fixture groups | `wallisfill` (9), `warpaffine` (12), `scenegeometry` (4) |
| package bugs found | the rotation direction and the fixed-point interpolation, both in new code |
| harness bugs found | the `filtDisp` index, the sign selector, and a probe fed a doubly-filtered scene |
| reference behaviours recorded | mixed border modes in one Wallis call; swapped power labels in the log; a misleading decline message; the `nanmax` along-track bias |

The three remaining phase-3-and-beyond groups are unchanged in status: eight Sentinel-1 pairs and two NISAR
pairs need the radar geogrid path, and no golden pair has been compared as a *product* because the
post-correlation chain does not exist in Julia. Those are the next two gates, in that order.

---

# Radar phase — one Sentinel-1 SLC pair, gate by gate

Scope is deliberately **one** S1-SLC pair as a probe, then a reassessment. Radar container time is the
reason: this pair's capture took **2h15m** against ~8 min for an optical one, so the decision about the
remaining eight pairs is made against a measured cost rather than an estimate.

The pair is `S1A_IW_SLC__1SSH_20151120T080202_..._X_..._20151214T080202_..._G0120V02_P002` — the
smallest SLC in the set at 1536x1024 output with 70,214 valid velocity points.

## Step 1 — the capture, green

```bash
julia --project=tools/golden tools/golden/intermediate.jl S1A_IW_SLC__1SSH_20151120T080202 --run 200
```

The reference's own ISCE3 produced `reference.tif`/`secondary.tif` and the capture intercepted
`runAutorift` at the same boundary the twelve optical pairs use, with **no harness change** —
`capture.py` passes its command line to `hyp3_autorift.process.main()`, which reaches
`process_sentinel1_slc_isce3`.

| | |
|---|---|
| grid | 3520 x 3280 |
| image | 23857 x 65978 `UInt8` |
| wall clock | ~2h15m |
| on-disk | 26 GB |

### What `optflag` does, and why the correlator needs no radar mode

`optflag`/`optical_flag` appears five times in `testautoRIFT.py` and **every one is before the capture
boundary**: `loadProduct` per-scene (`:304`), the `ChipSizeMaxX` override it skips (`:376`),
`obj.Dy0 = -1 * obj.Dy0` (`:402`), and the `OverSampleRatio` table (`:477`). `grep optflag` on
`autoRIFT.py` returns nothing. So the radar path differs only in *what arrays and scalars the
correlator is handed*, which is exactly what the capture records.

## Step 2 — every captured scalar, against what the code trace predicted

Recorded **before** running the ladder, so a later surprise is measured against a written expectation.

| scalar | optical | this pair | predicted? |
|---|---|---|---|
| `WallisFilterWidth` | 5 | **21** | yes — `vend/testautoRIFT.py:714` sets 21 for `nc_sensor == 'S1'` |
| `OverSampleRatio` | 16/32/64/64 | **32/64/128/128** | yes — `:477`, the `optflag == 0` branch |
| `DataType` | 0 | **0** | yes — the byte path, as production optical |
| `ChipSize0X` | 16 | **64** | **no** |
| `ScaleChipSizeY` | 1.0 | **0.25** | **no** |
| `GridSpacingX` | 8 | 32 | — |
| `SkipSampleX/Y` | 32 | 32 | — |
| `minSearch` | 6 | 6 | — |
| `FracValid` / `FracSearch` | 0.32 / 0.2 | 0.32 / 0.2 | — |
| `CoarseCorCutoff` | 0.01 | 0.01 | — |

Two facts no reading of the driver would have given, and both matter:

- **`ChipSize0X = 64`**, a 4x larger base chip than any optical case. The pyramid therefore runs
  64/128/256/512 rather than 16/32/64/128, which is why the `OverSampleRatio` keys differ.
- **`ScaleChipSizeY = 0.25`**, so every chip is **64 x 16** — strongly anisotropic, from SAR
  range/azimuth geometry. Every optical pair in the set is 1.0. This exercises the chip-size
  asymmetry already registered as matched-not-endorsed in `README.md`, on a case where the asymmetry
  is 4:1 rather than absent.

### The reference's own level records, which bound what the ladder can assert

| seq | kind | chip | oversample | grid | measured / kept |
|---|---|---|---|---|---|
| 1 | coarse | 64 x 16 | 32 | 440 x 410 | 40,105 |
| 2 | filtDisp | — | 32 | 440 x 410 | kept 132 of 40,105 |
| 3 | coarse | 128 x 32 | 64 | 220 x 205 | 8,620 |
| 4 | filtDisp | — | 64 | 220 x 205 | kept 206 of 8,620 |
| 5 | **fine** | 128 x 32 | 64 | 1760 x 1640 | 128,825 |
| 6 | filtDisp | — | 64 | 1760 x 1640 | kept 15,703 of 128,825 |
| 7 | coarse | 256 x 64 | 128 | 110 x 102 | 1,790 |
| 8 | filtDisp | — | 128 | 110 x 102 | **kept 0** |
| 9 | coarse | 512 x 128 | 128 | 55 x 51 | 397 |
| 10 | filtDisp | — | 128 | 55 x 51 | **kept 0** |

Two structural differences from every optical case, stated as expectations for step 3:

- **Only one fine pass runs, at chip 128 x 32.** The base level (64 x 16) has a coarse pass and a
  `filtDisp` but *no* fine pass, and the two coarsest levels are rejected outright. So the merged
  answer comes from one level, and `exact` at the base chip size is not a meaningful gate here —
  bias and correlation are, as for the two optical pairs whose base level is skipped.
- **`filtDisp` keeps almost nothing at the coarse levels** — 132 of 40,105, then 0 twice. A rung
  asserting a nonempty coarse mask will fail for reasons that are the reference's behaviour, not a
  disagreement.

**`STAGES: 0`** — this capture was taken without `CAPTURE_STAGES=1`, so the loop-local trace is
absent and the stage-by-stage rungs cannot run against it. Re-capturing costs another 2h15m; step 3
records what the ladder can and cannot check without it.

## Step 3 — what the ladder opened, and the two fixes it forced

The ladder itself **cannot run on this capture**: it needs `CAPTURE_STAGES=1`, and taking that trace is
another 2h15m. It fails fast and says so, which is the right behaviour — the message names the
environment variable and the 0-based level index. So step 3's findings came from the endpoint instead,
and both are real.

### The harness: an anisotropic chip could not even be configured

`kwargs_from_capture` scaled `chip_size` by `ScaleChipSizeY` but set `chip_size_max` isotropically from
`in_ChipSizeMaxX`. That is the X-axis maximum — the array's name says so — so Y's bound was 4x too
large and the two axes reached their maxima after different numbers of doublings. `_check_levels`
rejects that outright:

```
`chip_size_max` must be the same multiple of `chip_size` in both axes ...
Got 8 in X and 32 in Y
```

Invisible on all twelve optical pairs, where `ScaleChipSizeY = 1.0` and the two forms coincide.

### The package: `_oversample` consulted the wrong axis

`_oversample` returned `clamp(min(ox, oy), 1, 2)`. The reference's ratio is
`int(self.ChipSize0X / self.GridSpacingX)` (`autoRIFT.py:481`) — **X alone, with no Y term anywhere**.
On a square chip the two agree, which is why twelve optical pairs never caught it. On this pair, 64x16
on a grid spaced 32 gives 2 in X and **0** in Y, so the minimum collapsed the ratio to 1 and the
outlier filter judged over a 5-wide window at `FracValid = 0.32` where the reference used 9 at a
threshold raised by its overlap term.

What the fix bought, same capture, same command:

| | before | after |
|---|---:|---:|
| only jl | 116,948 | **44,208** |
| only ref | 11,245 | **9,897** |
| corr dx | +0.9416 | **+0.9622** |
| corr dy | +0.8430 | **+0.9086** |
| julia measured | 176,211 | 104,819 |

**Matched, not endorsed.** A window derived from X and applied to both axes of a 4:1 chip covers four
times as much ground across track as along it, which is not obviously the right neighbourhood for
judging consistency. Registered in `README.md`; agreement is the objective here.

`test/outliers.jl`'s `rescale matches the reference's grid scaling` asserted the old rule and was
updated to the reference's, with the Sentinel-1 geometry asserted directly.

## Step 4 — the endpoint, green on bias and correlation

```bash
julia --project=tools/golden -t 6 tools/golden/correlator.jl S1A_IW_SLC__1SSH_20151120T080202 --run 200
```

| axis | sign | both | only jl | only ref | median | p99 | corr | bias |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dx | + | 60,611 | 44,208 | 9,897 | 0.331 | 6.355 | **+0.9622** | **+0.0271** |
| dy | **−** | 60,611 | 44,208 | 9,897 | 0.133 | 1.240 | **+0.9086** | **−0.0030** |

### The +0.027 px is 187 outliers, not a systematic offset

`bias` was a mean over every both-measured point, and that is not robust to a heavy tail. Decomposed:

| dx statistic | value | n |
|---|---:|---:|
| mean over all points | +0.0271 | 60,611 |
| median | +0.0057 | 60,611 |
| **mean, agreeing within 1 px** | **−0.0008** | 47,486 (78.3%) |
| mean, within 2 px | +0.0020 | 54,140 |
| points beyond 10 px | — | **187** |
| points beyond 20 px | — | 22 |
| max | 45.93 | — |

On the 78% of points that agree to within a pixel the offset is **−0.0008 px**, three hundred times
smaller than the headline and of the opposite sign. `dy` has no tail at all — zero points beyond 10 px —
and its mean, median and core figures agree at −0.003.

What the 187 are, measured rather than assumed:

- **Two-sided**: 99 positive, 88 negative. A systematic offset is one-sided; this cancels.
- **Not railed**: 0 of 187 sit at their search-radius boundary on either side, so neither implementation
  ran out of window.
- **All at chip 128**, the single level that produces this pair's answer.
- **Not low-correlation junk**: median correlation 0.146 at the tail against 0.148 over the whole pair.
- **Spatially clustered**: rows 1217–2161 of 3520, columns 418–940 of 3280.

The pair correlates at a **median of 0.148** — SAR speckle decorrelation over a 24-day repeat. At that
correlation the peak surface is nearly flat, so which local maximum wins is decided in the last bits.
That is the noise floor a `--dtype-pair` run would have bounded, and it is why `exact` is 0 here.

**So the reported number was the wrong statistic, and the reporting was fixed rather than excused.**
`correlator.jl` now prints `bias` (mean over everything, so a growing tail is visible), `bias core`
(within 1 px, where a systematic error lives), and the tail count that explains any gap. `3.rdr` gates
on the **core** bias at **0.005 px** — an order of magnitude tighter than the optical 0.035 — and bounds
the tail at 400 points separately. Measured: **0.0008 and 0.0025 px**, tail 187.

- **The `dy` sign resolves to `−`, measured not asserted.** This is the one radar-specific expectation
  the plan named: `optflag == 0` pre-flips `Dy0` at `testautoRIFT.py:402`. The selector discriminates
  on correlation, and `dy` correlates +0.909 at `−` — so the flip is confirmed by measurement on a
  60,611-point population, not inferred from the source.
- **`exact` is 0 by construction and is not the gate.** The base level (64x16) runs a coarse pass and
  a `filtDisp` but *no* fine pass, so every reported point comes from the 128x32 level and was
  bicubic-resized rather than quantized. This is the documented `†` case that two optical pairs also
  hit.
- **Julia answers 1.49x the reference's points, and that is the reference rejecting.** Its own
  `filtDisp` records keep 132 of 40,105 at the coarse pass, 15,703 of 128,825 at the fine one, and 0
  at both coarser levels. Julia recovers **86.0%** of what the reference kept; the excess is points
  the reference's filter discarded, not points the correlator disagreed about.

### Not done: the byte-vs-float floor, and why it is memory rather than time

`--dtype-pair` needs a second capture at `CAPTURE_FLOAT32=1`. It was attempted and **OOM-killed at
`AutoRIFT Start`** after 46 minutes:

```
/home/ubuntu/.profile: line 54: 614 Killed "${PIXI_EXE-}" "$@"
```

The constraint is arithmetic, not a flaky run. This scene is 23857 x 65978 = **1.57 billion pixels**:

| path | pair resident |
|---|---:|
| `UInt8` | 2.9 GB |
| `Float32` | **11.7 GB** |

Docker is capped at 32 GB on this machine (the host has 96 GB), and the reference correlator holds both
images plus its pyramid intermediates. An optical scene is ~15x smaller, which is why `--dtype-pair`
works there and the recorded LC08 floor at runs 201/301 exists. A Sentinel-1 SLC does not fit.

**Deliberately left unmeasured** rather than worked around: a spatial subset would have different
statistics, and the tail is spatially clustered (rows 1217-2161, columns 418-940), so a crop either
contains that region or misses the thing being measured. What stands in for it is `tools/ab` stage 1 —
the correlator is bit-identical to `arImgDisp_s` on identical `Float32` input — so the tail is unlikely
to be the correlator itself. That is inference, and it is recorded as inference.

To close it: raise Docker's memory ceiling above ~48 GB, then one capture and one `--dtype-pair` run.
The floor matters more on radar than optical
**Disk is the binding constraint**: 35 GB free with the twelve optical runs holding 63 GB, and this
pair's run reduced from 59 GB to 11 GB by deleting `product/` and `product_sec/` — the per-burst ISCE3
intermediates, which are regenerable and read by no gate. The floor matters more on radar than optical
because SAR amplitude has a long right tail, so `uniform_data_type`'s `mean ± 3σ` window clips a
different fraction than it does on optical reflectance.

## Step 5 — no slide, and the reassessment this scope existed for

```bash
julia --project=tools/golden tools/golden/regate.jl --all
julia --project=. -e 'import Pkg; Pkg.test()'
```

| gate | measured |
|---|---|
| 2.x colfilt, bwareaopen, window reductions | **green** |
| 0.1 the correlator alone | **green** — exact 100.0% |
| 0.2 the whole pipeline, 3072² | **green** — exact 81.8%, within step 98.7% |
| 0.3 the ITS_LIVE granule | **green** |
| 3.opt the stage ladder on every optical case | **green — 12/12** |
| **3.rdr the endpoint on the Sentinel-1 SLC pair** | **green** — bias 0.0271/0.0030, corr +0.962/+0.909, `dy` sign −, both 60,611 |
| 3.x the stage ladder, base case | **green** — 23 rungs |

**7 ran, 7 green, 0 red.** `Pkg.test()` passes. The `_oversample` change touched shipping code and the
optical gates did not move — 12/12 before and after.

`3.rdr` is its own gate rather than a thirteenth case in `3.opt`: that gate's value is that all twelve of
its cases are optical, so a red one names the class that broke.

### What one radar pair cost, and what it bought

| | |
|---|---|
| capture wall clock | **2h15m** (plan estimated 30–60 min) |
| peak on-disk | 59 GB, reducible to 11 GB |
| package bugs found | 1 — `_oversample` consulted the wrong axis |
| harness bugs found | 1 — `chip_size_max` not scaled by `ScaleChipSizeY` |
| plan items not executed | 1 — the byte-vs-float floor, on disk |

### The three questions this scope was chosen to answer

**1. What does a pair cost?** 2h15m and ~60 GB peak, against ~8 min for an optical pair. The ISCE3
coregistration is the long pole, not the correlator. 44 GB of that is `product/` and `product_sec/` —
per-burst intermediates that no gate reads and that can be deleted immediately after the capture.

**2. Did any rung need radar-specific work?** No rung, but two configuration paths did, and both were
invisible across all twelve optical pairs because they only appear when the chip is anisotropic. Neither
is in the correlator: `optflag` never reaches the `autoRIFT` object, and that prediction held — the
correlator has no radar mode, and the byte path, the pyramid and the `filtDisp` chain all behaved as the
optical cases led us to expect once the chip geometry was configured correctly.

**3. Is S1-BURST the same problem?** Unknown, and it takes a different driver — `process_burst` rather
than `process_slc`, with multi-burst mosaicking before the correlator. What this pair establishes is
that the *boundary* is identical: `capture.py` needed no change to intercept a radar run, and the same
`kwargs_from_capture`/`pointset_from_capture` path drives AutoRIFT.jl once `ScaleChipSizeY` is honoured
in both chip bounds.

### What the remaining eight would need

Disk, before time. The twelve optical runs hold 63 GB and this pair's reduced run 11 GB, leaving 35 GB
free — enough for one radar capture at a time if `product/` is deleted as soon as the capture is
verified. Eight pairs at 2h15m is ~18 hours of container time, and three of them are S1-BURST with up to
24 bursts each, so the burst driver's cost is not yet measured.

The one plan item left undone is the **byte-vs-float floor** (`--dtype-pair`), which needs a second
capture of this same pair at `CAPTURE_FLOAT32=1`. It matters more on radar than optical: SAR amplitude
has a long right tail, so `uniform_data_type`'s `mean ± 3σ` window clips a different fraction than it
does on optical reflectance, and the floor is what makes the 9,897 reference-only points attributable.

---

# All eight remaining radar pairs

Run after the probe established the strategy: capture at `--run 200`, verify `call1.json`, prune the
ISCE3 per-burst intermediates, then the endpoint. Sequential, because a capture peaks near 67 GB.

## The eight, all green

Superseded by three later fixes and re-measured at the end of this file, where `3.rdr` is 6 of 8.

| # | case | driver | both | exact | only jl | only ref | core bias dx / dy | corr dx / dy | tail |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `S1A_..._20150828T162412` | SLC | 1,327,618 | 29.03% | 77,340 | 69,830 | −0.0009 / +0.0020 | +0.820 / +0.825 | **0** |
| 2 | `S1A_..._20151120T080202` | SLC | 60,611 | 0.00% † | 44,208 | 9,897 | −0.0008 / −0.0025 | +0.962 / +0.909 | 187 |
| 3 | `S1A_..._20170221T204710` | SLC | 1,534,262 | 74.24% | 42,109 | 182,877 | +0.0040 / −0.0016 | +0.992 / +0.942 | 39 |
| 4 | `S1B_..._20180809T204617` | SLC | 454,314 | 73.53% | 132,083 | 71,160 | −0.0042 / +0.0065 | +0.955 / +0.880 | **0** |
| 5 | `S1C_..._20250416T010214` | SLC | 476,406 | 47.29% | 54,461 | 54,427 | +0.0032 / +0.0003 | +0.981 / +0.846 | **0** |
| 6 | `S1C_..._20250416T010159` | BURST 7 | 33,213 | 38.45% | 13,813 | 8,722 | −0.0007 / −0.0011 | +0.932 / +0.835 | **0** |
| 7 | `S1A_..._20240618T025533` | BURST 10 | 440,025 | **75.81%** | 33,828 | 143,030 | +0.0011 / −0.0004 | +0.993 / +0.982 | 9 |
| 8 | `S1A_..._20240618T025528` | BURST 24 | 1,231,710 | 65.81% | 67,461 | 358,526 | −0.0022 / −0.0013 | +0.988 / +0.951 | 7 |

† Base level runs a coarse pass and a `filtDisp` but **no fine pass**, so every reported point is
bicubic-resized rather than quantized and `exact` is 0 by construction.

**Core bias spans 0.0002 to 0.0065 px across all sixteen axes.** Four of eight report a zero tail beyond
10 px; the two largest are 187 (case 2) and 39 (case 3), against populations of 60,611 and 1,534,262.

**`exact` spans 29% to 76% on the seven that quantize**, tracking the scene rather than the code — the
same pattern the optical set shows at 27–93%. Case 2 is 0 by construction and is the only one.

One reporting note worth keeping: the two 2024-06-18 burst cases share a filename prefix, so an
endpoint log written per-case-prefix silently overwrites one with the other. Both were re-run to
distinct logs to fill this table; a per-case log path should carry the full product name.

## What the eight added over the probe

**The probe was the weakest case in the set, not a typical one.** Its `exact` of 0% and its 187-point
tail both come from the same fact — its base level runs no fine pass, so its answer comes from one
level at a median correlation of 0.148. The other seven run all four levels and reach `exact` of **29% to 76%**, with
correlations of 0.82–0.99. Reading the phase off the probe alone would have understated agreement
substantially: it is the only case at 0%, and the only one whose base level is skipped.

**`process_burst` needs no harness change either.** Three burst pairs — 7, 10 and 24 bursts mosaicked
before the correlator — reach the same `runAutorift` boundary through a different driver, and
`kwargs_from_capture`/`pointset_from_capture` drive AutoRIFT.jl unchanged. The 24-burst pair is the
second-largest case in the whole radar set at 1,231,710 both-measured points.

**Cost, measured rather than estimated.** The probe's 2h15m was not representative:

| driver | pairs | capture wall clock | peak on-disk | pruned to |
|---|---|---|---|---|
| `process_slc` | 5 | 12–50 min | 41–67 GB | 7–11 GB |
| `process_burst` | 3 | 12–50 min | 14–56 GB | 4–16 GB |

All seven new captures took **3h50m total**, against the ~18 hours estimated from the probe. Pruning
`product/` and `product_sec/` immediately after each capture is what made a sequential run possible on a
pool with ~60 GB free: those per-burst ISCE3 intermediates are 70–80% of a run and are read by no gate.

**A capture and a whole-scene endpoint comparison are different costs, and the second is the larger
one.** A capture runs the reference and dumps its arrays; `correlator.jl` then runs AutoRIFT.jl over
every point of the same grid — 5,363,712 points across a 57760x50511 pair on L1 RSLC (2.9 Gpx), and
5,234,944 across 54885x110085 on L2 GSLC (6.0 Gpx). The capture cost is the table above; the endpoint
comparison is hours. Do not quote one for the other — a whole-scene endpoint duration is not a statement
about how long the reference takes.

**`threaded` is what the endpoint's cost turns on, and it was `false` for every golden run before
`kwargs_from_capture` set it.** `Params` defaults it to `false`, so `-t 8` allocated eight threads and
used one. Measured on a 201x201 window of the L1 RSLC grid, all five output fields bit-identical either
way: 148.9 s serial against 22.3 s on eight threads.

**The speedup does not hold across a whole scene, because a scene has a serial phase.** The L1 endpoint
on eight threads ran at ~7.8 cores for its first 2.5 hours and then dropped to **1.0 core**, sampled
over 90 s of CPU time rather than read from an instantaneous `%CPU`. `threaded` parallelises over grid
points within a correlation pass (`src/params.jl:392`), so a serial phase is work outside one: the hole
fill, the merge across levels, or the comparison itself. A window benchmark sees only the parallel part
and overstates what the flag buys on a scene — the 6.7x above did, and an estimate built on it was
wrong by more than a factor of two.

That serial time is **FFTW plan construction**, not any of the three candidates guessed at above, and
it is fixed. See "the serial phase was the FFTW planner" at the end of this file.

## Step: the gate covers every radar case

`3.rdr` now runs all eight rather than one, for the same reason `3.opt` runs all twelve: the two
`_oversample` and `chip_size_max` bugs were invisible on twelve optical pairs and surfaced only on an
anisotropic chip. Thresholds are the weakest measured case less a margin — core bias 0.010 px,
correlation 0.78 (case 1 is the floor at 0.820/0.825), tail 400 points.

Those thresholds were calibrated against the table above, which three later fixes superseded. The eight
pairs are re-measured at the end of this file: coverage and correlation improve on nearly every one, and
two cases now exceed the core-bias threshold.

---

# Re-measurement after the Dy0 sign fix

The prior's sign convention was wrong in two places, not one. `dc55e09` fixed
`pointset_from_capture`; `b10c8ba` fixed the stage ladder's own two `PointSet` constructions and
rung 3.6c's comparison. Every case-level figure recorded before those two commits was measured
through one or the other, so this section re-measures them rather than editing them.

All twelve optical cases are re-measured here. The eight radar rows are not, so the warning at the top
of this file stands until they are.

The nine cases whose grid step carries a zero are re-measured again later in this file, under "the twelve
optical cases, re-measured after the grid-step fix" — read that table for those. The three whose step is
already nonzero reproduce this one exactly, which is what makes them the control there.

Measured at `6dd337b` (2026-09-10). Command per case:

```
julia --project=tools/golden -t 8 tools/golden/correlator.jl <case> --run 200
```

## The stage ladder, and why `3.opt` was red

The fix is visible as a rung before it is visible as a figure. On `LC08_L1TP_009011`, rung 3.7
(coarse correlation against the reference's own level-0 output):

| | before | after |
|---|---|---|
| exact | 86.54% | **99.95%** |
| bias | −0.0916 | **−0.0008** |
| p99 / max | 63 / 342 | **0 / 9** |

Rung 3.6c compares the y prior against the trace, which holds it in cartesian-Y while a `PointSet`
holds matrix-Y. Comparing across the flip reported all 11,562 nonzero points of 85,556 as
disagreements while the chips placed were identical.

All twelve optical cases were red on that one rung and no other. `3.opt` is **12/12** and `3.x` is
**23 of 23 rungs**.

## The optical endpoint, re-measured

All twelve, in the order the earlier table used.

| # | case | filter | both | exact | was | Δ | bias core dx | corr dx | tail |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | S2A Malaspina | `hps` | 586,129 | **94.24%** | 92.65% | +1.59 | +0.0018 | +0.998 | **0** |
| 2 | LC09 Antarctic | `hps` | 464,316 | **83.01%** | 68.42% | +14.59 | −0.0048 | +0.998 | **0** |
| 3 | S2B Jakobshavn | `hps` | 605,987 | **78.85%** | 67.34% | +11.51 | −0.0047 | +1.000 | 2 (dy) |
| 4 | LC08 East Greenland | `hps` | 691,714 | **72.74%** | 63.21% | +9.53 | −0.0226 | +0.991 | 3 |
| 5 | LC08 Jakobshavn | `hps` | 1,662,200 | **71.84%** | 54.45% | +17.39 | −0.0080 | +0.998 | **0** |
| 6 | `LE07_..._20130314` | `wallis_fill` | 713,305 | 59.94% | 59.83% | +0.11 | −0.0326 | +0.920 | **0** |
| 7 | `LC08_060018` × `LE07` | `wallis_fill` | 672,912 | 58.56% | 58.49% | +0.07 | +0.0115 | +0.955 | **0** |
| 8 | `LE07_..._20040810` | `wallis_fill` | 918,180 | **52.91%** | 50.18% | +2.73 | −0.0084 | +0.975 | **0** |
| 9 | `LE07_..._20120428` | `wallis_fill` | 106,228 | 0.00% † | 0.00% | — | +0.0083 | +0.891 | **0** |
| 10 | `LT04_063018` | `fft` | 272,875 | 56.02% | 55.95% | +0.07 | **+0.0008** | +0.982 | **0** |
| 11 | `LT05_060018` | `fft` | 124,397 | 27.09% | 27.09% | 0.00 | −0.0031 | +0.993 | **0** |
| 12 | `LT05_001013` (`P000`) | `fft` | 18,540 | 0.00% † | 0.00% | — | −0.0847 | +0.997 | **0** |

† base level skipped on both sides, so every point is coarse and unquantized and `exact` is 0 by
construction. Bias and correlation are the gate there.

**Eleven of twelve report a zero tail beyond 10 px**, and the two that do not report 2 and 3 points
against populations of 605,987 and 691,714. Before the fix `LC08_L1TP_009011` alone reached a
maximum residual of 399 px; it is now **5.4 px**, with 0 of 1,662,200 points beyond 10 on either
axis. This is the clearest single effect of the fix.

**The gain lands entirely on the five `hps` cases**, +1.6 to +17.4 points, while the four
`wallis_fill` and three `fft` cases move by at most +2.73 and mostly by under 0.15. That split is not
a filter effect: cases 6, 7, 8 and 10 quantize 53–60% of their points, so they had as much room to
gain as case 5 at 72%. The prior is the discriminating quantity — the misplacement was `2 * Dy0`
rows, so a scene whose prior is near zero was never displaced far enough to lose its peak. Case 11 at
27.09% is unchanged to four figures.

**Case 12's bias is the one figure to keep an eye on.** At −0.0847 px it is the largest in the set by
6×, on the smallest population (18,540) and the only `P000` case. It did not move with the fix, so it
is not a sign-convention artifact; it predates this work and is unexplained. (Part of it was the missing
cell-centre shift: the grid-step fix takes it to −0.0691, measured later in this file.)


---

# The coarse-level residual is not the interpolation

The NISAR L2 GSLC case disagrees by +1.03 px at chip 384 and +2.15 px at chip 768 while the base level
agrees at 98.0%. Above the base level a reported value is a `cv2.resize(..., INTER_CUBIC)` of a filtered
field rather than a measurement (`autoRIFT.py:856-866`), so "interpolation" is the available
explanation. It is the wrong one, and the arithmetic says so before any measurement does.

## Why interpolation cannot produce a pixel

Bicubic resampling is a deterministic 16-tap weighted sum. Identical inputs, weights and sample
positions give identical outputs up to float32 summation order, which is bounded by `16 * eps * |value|`:

| displacement | summation-order bound |
|---|---|
| 1 px | 1.9e-06 px |
| 10 px | 1.9e-05 px |
| 30 px | 5.7e-05 px |

The observed residual is 1.1e+05 times the bound at a 10 px displacement. Five orders of magnitude is
not an arithmetic effect.

Measured against OpenCV's own output in `test/fixtures/resize/` — 63 and 64 samples, six scale factors
including a non-integer 0.4286 — the worst disagreement over all twelve `INTER_AREA` fixtures is
**1.79e-07** and over all twelve `INTER_CUBIC` fixtures **4.17e-07**, both at the float32 rounding
floor, several bit-identical. (Only one of the 48 fixtures is asserted by the suite; the rest are
checked here and should be added to it.)

## Every step of the merge chain reproduces the reference exactly

Taken from a stage trace of level 2 at `CAPTURE_STAGES=1 CAPTURE_STAGE_LEVEL=2` (run 201), replaying
each operation on the reference's **own** traced input and comparing against its own traced output:

| line | operation | max abs difference |
|---|---|---|
| `:831` | `INTER_AREA` downsample of `DxF0` | 3.05e-05 — float32 floor |
| `:847` | `DxFM` patched from `DxF0` | **0.000e+00 — bit-exact, 100.00%** |
| `:852` | `DxF` patched from `DxFM` | **0.000e+00 — bit-exact, 100.00%** |
| `:856` | `INTER_CUBIC` upsample of `DxF` | 3.05e-05 — float32 floor |

At the 87,913 points the merge assigns to chip 384, the two agree on definedness for **all** of them.

One difference exists and is inert: after the upsample our field defines 203,648 points the reference
leaves `NaN`, because OpenCV propagates a `NaN` tap over the whole 4x4 kernel footprint while a rule that
skips and renormalizes keeps the point. None of those points is one the merge reads, so it cannot
contribute to the residual — but it is a real difference in the NaN-propagation rule and would matter to
any step that read the field directly.

## What that leaves

Given the reference's own `DxF` the whole chain is reproduced bit-for-bit, so the residual enters
*before* it — in `DxF_rev0`, the level's own raw measurement at chip 384. That is the correlator at a
coarse chip size, not the merge.

Two further constraints on the cause, both measured:

- **The sign differs between the two NISAR cases.** L1 RSLC coarse means are −0.017 / −0.077 / −0.575
  at chips 192 / 384 / 768; L2 GSLC are −0.040 / +1.034 / +2.145. Same code, same ladder, same `Scale`
  values, opposite signs — so not a fixed arithmetic or registration error.
- **Emptiness does not explain it.** L1's coarse levels are *more* sparse (93.2% and 94.6% NaN at chips
  384 and 768) than L2's (90.6%, 92.6%) and its residuals are smaller.

Hypotheses closed by measurement, so they are not re-opened: integer truncation at `:821` (2288 = 2^4 *
143, so every `Scale` divides exactly); a fixed fraction of a coarse pixel (the per-point distributions
differ in shape — chip 384 is right-skewed at median +0.055 against mean +0.259, chip 768 symmetric at
median +0.281 against mean +0.268, so the equal means were coincidence); a whole-coarse-pixel shift
(2.1% of chip-768 points lie within 0.15 px of any `k * Scale`); `InterpMask` (the level scaling survives
restricting to unflagged points); the interpolator and its NaN rule (above).

## Where it is not: five steps and two selection mechanisms, each closed by measurement

Continuing from the section above, on the NISAR L2 GSLC case at chip 384 and 768.

**The raw fine pass agrees.** Re-running the level-2 fine pass on the reference's own traced inputs —
its lattice, its post-`MC2` search radii, its rounded prior, `chip 384 x 192`, oversample 64 — gives
**99.41% bit-exact** over 30,115 both-defined points, with `p50 = p90 = p99 = 0`. Both sides sit exactly
on the 1/64 lattice. What remains is 177 points differing by up to 94.56 px, which is a different
correlation peak rather than a subpixel disagreement, and 548 points the reference reports and
AutoRIFT.jl declines.

**Those points are not what the heatmap shows.** Mapped on the level's own 572² grid, the 548 declined
points lie on a thin diagonal along the footprint edge and the 177 are scattered; `ddx` at the raw pass
renders blank on the same ±2 px scale where the merged residual shows strong blocks, and the merged
chip-384 points fall where level 2 *agreed*. `figs/nisar_l2_level2_classes.png`.

The arithmetic also rules them out on their own: 177 points at up to 94.56 px supply at most 16,832 px
of displacement sum, against the 90,902 px needed to move 87,913 merged points by +1.034.

**Level ownership agrees, and disagreeing about it costs little.** `chip_size` records which level
answered each point, so the two maps can be compared rather than a proxy for them. Over a 401² window,
124,159 both-measured points:

| | n | share | mean ddx | median | std |
|---|---:|---:|---:|---:|---:|
| same owning level | 118,887 | 95.75% | +0.0017 | +0.0000 | 0.2541 |
| different level | 5,272 | 4.25% | −0.0421 | −0.0279 | 0.2542 |

The two groups have the *same* spread, and the largest per-pair mean anywhere in the ownership confusion
matrix is −0.0474 px (`jl 384 -> ref 192`, n = 4,594). Nothing approaches +1.03 or +2.15, so a level
mismatch does not produce a large residual.

Ownership is exclusive and cumulative, which the trace confirms: `ChipSizeX_rev0_L2` holds
0/96/192/384 counts of 3,590,592 / 1,215,214 / 341,225 / 87,913, summing to 2288², and the 96/192/384
counts are identical in the final `out_ChipSizeX` — chip 768's 118,799 comes out of the points still at
0. That is the `ChipSizeX == 0` guard at `autoRIFT.py:863-864`.

**The residual is local, and no single coordinate explains it.** Windows disagree with each other. At
rows 113–513 the chip-384 mean is +1.034 and chip 768 is +2.145; at rows 891–1291 chip 384 is −0.026 and
no chip-768 point appears, with a whole-window mean of −0.00018 px. A sweep of 121² windows across the
grid shows chip 96 within 0.023 px of zero in **every** window while the coarse levels swing widely — and
the swing is not a function of position: row 781 spans −0.073 to +0.108 at chip 384, and row 541 spans
+0.063 to +2.488 at chip 768. A row-dependent law fits the first few windows and is falsified by the
within-row spread.

**So the base level agrees everywhere and the coarse levels disagree locally**, while every step that
builds a coarse level — correlate, median-fill, previous-level fill, `INTER_AREA`, `INTER_CUBIC`, merge
— reproduces the reference exactly on the reference's own inputs. The candidates that were tested and
failed are listed above so they are not retested.

### The cause: two coupled bugs in the coarse grid, not in any step on it

Every step was right and every step ran on the wrong grid. Two independent defects, both in AutoRIFT.jl,
both invisible on a square chip:

1. **The level stride ignored the reference's rule and consulted the y extent.**
   `_level_decimation` returned `min(sx, sy)` over per-axis ratios where the reference resizes a level's
   grid by `ChipSize0X / ChipSizeUniX[i]` (`autoRIFT.py:510-514`) — one factor, from the x extents,
   applied to rows and columns alike. On NISAR that gave 1, 1, 2, 4 against the reference's 1, 2, 4, 8.
   At half the stride a level posts **four estimates per chip footprint** instead of one: four views of
   mostly the same pixels, which the coherence filter cannot separate, so they survive as mutually
   corroborating outliers. That is the blunder texture — a rough field where the reference's is smooth.

2. **`_grid_step` read a spacing of zero, so the cell-centre shift never happened.** It excluded steps
   whose endpoints were zero, on the grounds that zero marks nodata. Both NISAR grids are filled with
   `0.5`, carried to `1.5` by the half-sample snap, and the fill is the *majority* of the array — 55.7%
   on L2, 56.8% on L1. The guard never fired, 2,895,601 zero steps inside the L2 fill outvoted 2,297,235
   real 48 px ones, and the mode came out `0`. `_cell_centres` then shifted by nothing and every coarse
   node sat at its cell's first point, half a cell from where `_undecimate_level` reads it back.

**The coupling is why this resisted single-variable testing.** An earlier attempt at (1) alone measured
*worse* — neighbour disagreement 3.0x to 10.2x, `exact` down — because at the wrong stride and a zero
shift the two errors partly cancel. Fixing either alone breaks that cancellation. Both candidate
"second differences" that were hunted instead have been eliminated by direct measurement: scipy's
even-window origin matches `_window_margins` at w = 2, 4, 8, and the coarse node placement rule is
identical to the reference's `INTER_AREA` block centre once converted to 1-based indices — but it is a
*function of the stride*, which is exactly how one bug masqueraded as two.

**What the ladder was doing wrong.** Every rung in `stages.jl` derived its own stride as `chip ÷ chip0`
— the reference's rule — rather than calling `_level_decimation`. So the ladder compared the reference
against a reimplementation of the reference and stayed green whatever production computed. Rung 3.1's
shape check also *reported* rather than failed. The rungs now call the production function, and a level
whose grid size disagrees with the reference's is red.

Measured at level 2 of the L2 GSLC case, whose reference grid is 572²:

| | our level grid | reduced prior vs reference | rung 3.1 cell |
|---|---|---|---|
| before | 1144 x 1144 | not comparable — shapes differ | `NaN of a 0 px cell` |
| after | **572 x 572** | 327,145 of 327,184 agree (99.988%) | 192 px, offset 0.375 of a cell |

Of the 39 residual nodes, 27 are the documented 1 px `INTER_NEAREST` mapping difference, 8 sit on the
nodata boundary, and 8 exceed 5 px.

**Every level grid and every coarse lattice now matches the reference's traced arrays**, on both cases —
sixteen shapes, no exceptions. The stage trace dumps the level grid at `lvl{3,7,11,15}_xgrid` and the
coarse lattice at `lvl{1,5,9,13}_xgrid`:

| | level grids | coarse lattices |
|---|---|---|
| L1 RSLC, ours and the reference | 2328x2304, 1164x1152, 582x576, 291x288 | 291x288, 145x144, 72x72, 36x36 |
| L2 GSLC, ours and the reference | 2288x2288, 1144x1144, 572x572, 286x286 | 286x286, 143x143, 71x71, 35x35 |

Before the fix three of the four level grids were wrong on each case.

**A windowed pass cannot measure this fix, and the reason is worth recording.** `_coarse_points` returns
`nothing` when a level's coarse lattice is narrower than the outlier filter's window, so a level whose
lattice does not fit is skipped outright. Halving each stride to its correct value halves each lattice,
so windows that previously ran a level now skip it: on a 201² window chip 384 and 768 both return
`nothing`, and on a 61² window *every* level does. That is the extent dependence
`window_endpoint.jl`'s header already documents, made sharper — so the level grids above were verified
on the full grid, where every lattice survives, and a case-level residual still has to come from
`correlator.jl`.

Cost per level rises with chip size despite the point count falling: on a 61² window the four levels take
4.4, 6.2, 8.3 and 18.8 s at strides 1, 2, 4, 8. Chip area grows 64x across the pyramid while the point
count falls 64x, and area wins — the FFT is over the padded chip-plus-search extent, not over the grid.

**Which cases the fix moves.** The stride changes wherever the chip is anisotropic, which is every radar
pair as well as both NISAR ones — not NISAR alone:

| configuration | old stride | new stride | |
|---|---|---|---|
| optical, `ScaleChipSizeY = 1.0`, spacing 16, max 64/128/256 | 1, 2, 4, 8 | 1, 2, 4, 8 | unchanged |
| Sentinel-1, `ScaleChipSizeY = 0.25`, chip 32x8, spacing 32 | 1, 1, 1, 2 | 1, 2, 4, 8 | **changed** |
| NISAR L1, chip 96x52, spacing 48 | 1, 1, 2, 4 | 1, 2, 4, 8 | **changed** |
| NISAR L2, chip 96x48, spacing 48 | 1, 1, 2, 4 | 1, 2, 4, 8 | **changed** |

So no optical case's stride moves — all three optical configurations give the same strides under both
rules — and the eight radar pairs are invalidated along with the two NISAR ones. Every figure above is
read from the real `kwargs_from_capture` path on a capture on disk, radar included: all eight radar
captures are intact, so `3.rdr` was re-measurable without a container run. It is re-measured at the end of
this file, where the radar strides are also given per pair rather than as one Sentinel-1 row — three pairs
decimated by 1, 1, 1, 1 before the fix, not 1, 1, 1, 2.

**The stride is only half the scope.** The grid-step fix travels on a different axis, and an unchanged
stride does not imply an unchanged case: it moves **nine of the twelve optical cases**, measured through
`kwargs_from_capture` on each capture's own `in_xGrid`/`in_yGrid`. The majority-constant fill is not a
NISAR property.

| case | step before | step after |
|---|---|---|
| `LC08_L1TP_009011` | **0, 0** | 8, 8 |
| `LC08_L1TP_062018`, both `LE07_061018`, `LE07_063018`, `LC08_060018`×`LE07`, `LT05_060018`, `LT04_063018` | **0, 0** | −1, −1 |
| `LT05_001013` | **0, 0** | 4, 4 |
| `LC09_L1GT_215109`, `S2A`, `S2B` | 8/−1/12 | unchanged |

The three whose grid arrives without a zero-valued step are the control: they must reproduce the previous
table exactly, and they do — see below.

The stride also changes on a *square* chip whenever `grid_spacing` does not divide `chip_size_min` — 296
of the swept combinations — but no golden case is configured that way, since `_oversample * grid_spacing`
equals `chip_size_min.X` exactly in every one of them.

## Step: the NISAR endpoint, on both cases, after the stride and grid-step fixes

The case-level numbers, from `correlator.jl` over the whole grid — the only source of one. Both cases
run the reference's own captured inputs through AutoRIFT.jl and diff against the reference's own `Dx`/`Dy`.

| | L1 RSLC | L2 GSLC |
|---|---:|---:|
| grid | 2328 x 2304 | 2288 x 2288 |
| scene | 57760 x 50511 (2.9 Gpx) | 54885 x 110085 (6.0 Gpx) |
| both-measured | 1,786,566 | 1,751,658 |
| `dx` exact | **73.90%** (1,320,352) | **72.20%** (1,264,646) |
| `dy` exact | 73.86% (1,319,603) | 75.34% (1,319,639) |
| `dx` correlation | **+0.99972** | +0.99890 |
| `dy` correlation | +0.99884 | +0.99330 |
| `dx` median \|d\| | 0.078 px | 0.132 px |
| `dy` median \|d\| | 0.093 px | 0.054 px |
| `dx` p99 | 0.887 px | 3.608 px |
| `dx` bias / bias core | +0.055 / +0.051 | −0.010 / −0.113 |
| `dy` bias / bias core | +0.027 / −0.028 | **−0.378** / **−0.126** |
| tail >10 px | 78 `dx`, 7 `dy` | 31 `dx`, 0 `dy` |
| only julia / only reference | 11,633 / 14,897 | 30,117 / 11,493 |
| wall clock | 6h30m on 8 threads | 11h41m on 1 thread |

**Read `bias_core`, not `bias`, when asking whether there is a systematic offset** — the reason is at
`correlator.jl:371`. L2's `dy` `bias` of −0.378 is largely a two-sided tail on a flat SAR correlation
surface; its systematic component is −0.126 over the 364,798 points agreeing within a pixel. That is
still the largest core bias of the four axes and is unexplained.

**What the fixes bought, measured against the pre-fix L2 run** (`dx` / `dy`):

| L2 GSLC | pre-fix | post-fix | |
|---|---:|---:|---|
| both-measured | 1,646,459 | **1,751,658** | +105,199 (+6.4%) |
| only reference | **116,692** | **11,493** | −105,199 — the recovered points |
| only julia | 30,805 | 30,117 | ~unchanged |
| `dx` exact | 72.90% | 72.20% | −0.70 pt |
| `dy` exact | 73.08% | **75.34%** | +2.26 pt |
| `dx` correlation | +0.99877 | +0.99890 | + |
| `dy` correlation | +0.99511 | +0.99330 | − |
| `dx` bias core | −0.0131 | −0.1128 | worse |
| `dy` bias core | −0.0366 | −0.1259 | worse |

**The coverage gap is what closed.** Before the fix the reference measured 116,692 points we did not;
now it measures 11,493 — a tenfold reduction, and the direct consequence of the coarse levels finally
running on the reference's own lattice. Those 105,199 recovered points are coarse-level ones, which is
also why `dx` exact fell slightly while `dy` exact rose: the population being scored grew by 6.4%, and
the added points are the unquantized coarse-level kind where exact agreement is unreachable by
construction. A fraction over a changed population is not comparable to itself — the counts are.
`dx` exact **n** rose 1,200,214 → 1,264,646.

**Both core biases got worse**, from −0.013/−0.037 to −0.113/−0.126 px, and that is unexplained. It is
the one number that moved the wrong way on a population that grew, so it is not a denominator artifact.
`bias_core` is the statistic to read here, per `correlator.jl:371`; L2's `dy` `bias` of −0.378 is largely
a two-sided tail on a flat SAR surface.

So the coarse-grid fixes were necessary — the level grids were provably wrong, and 105,199 points of
coverage came back — and they did not close the coarse-level residual. Open, not attributed:

- Both core biases roughly tripling, on both axes, while coverage improved.
- L2's `dy` correlation slipping +0.99511 → +0.99330 where `dx` improved.

**`exact` is the wrong statistic above the base level and these are whole-scene figures**, so a large
part of both cases is coarse-level points where neither side is quantized and exact agreement is
unreachable by construction (recorded earlier in this file). Correlation and bias are the numbers that
carry meaning here; `exact` is reported for continuity with the optical cases.

## Step: the twelve optical cases, re-measured after the grid-step fix

The stride does not move on any optical case, but the grid step moves on nine of the twelve, so the
optical table above is superseded on those nine and confirmed on the other three.

**Gate `3.opt` is 12/12 green** — 254 rungs, 0 red, ~28 s per case on 10 threads. Every capture and its
stage trace is still on disk, so this needed no container run.

The endpoint, from `correlator.jl` over the whole grid, in the previous table's order. `both` is points
both sides measured; `mv` marks a case whose grid step moved:

| # | case | mv | both | Δ both | exact | exact n | Δ | bias core dx | corr dx | tail |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | S2A Malaspina | | 586,129 | **0** | 94.24% | 552,351 | — | +0.00179 | +0.99782 | 0 |
| 2 | LC09 Antarctic | | 464,316 | **0** | 83.01% | 385,409 | — | −0.00479 | +0.99755 | 0 |
| 3 | S2B Jakobshavn | | 605,987 | **0** | 78.85% | 477,819 | — | −0.00471 | +0.99967 | 2 (dy) |
| 4 | LC08 East Greenland | * | 692,088 | +374 | 72.70% | 503,146 | ~ | **+0.00424** | +0.99200 | 1 (dx) |
| 5 | LC08 Jakobshavn | * | 1,685,673 | **+23,473** | 70.87% | 1,194,637 | **+596** | −0.00888 | +0.99931 | 0 |
| 6 | `LE07_..._20130314` | * | 713,575 | +270 | 59.92% | 427,581 | ~ | **−0.02658** | +0.92601 | 0 |
| 7 | `LC08_060018` × `LE07` | * | 672,440 | −472 | 58.60% | 394,041 | ~ | +0.01127 | +0.95656 | 0 |
| 8 | `LE07_..._20040810` | * | 919,043 | +863 | 52.86% | 485,817 | ~ | **+0.00478** | +0.97552 | 0 |
| 9 | `LE07_..._20120428` | * | 112,064 | **+5,836** | 0.00% † | 0 | — | +0.00666 | +0.88999 | 0 |
| 10 | `LT04_063018` | * | 272,851 | −24 | 56.02% | 152,861 | ~ | +0.00102 | +0.98253 | 0 |
| 11 | `LT05_060018` | * | 137,259 | **+12,862** | 24.61% | 33,774 | **+68** | −0.00341 | +0.99210 | 0 |
| 12 | `LT05_001013` (`P000`) | * | 19,408 | +868 | 0.15% † | 29 | **+29** | **−0.06907** | +0.99819 | 0 |

† base level skipped on both sides, so every point is coarse and unquantized.

**The three unmoved cases are the control, and they reproduce the previous table exactly** — identical
`both` counts, exact counts within rounding of the recorded percentage, and core bias agreeing to five
figures. All nine moved cases changed. Prediction and measurement agree in both directions, which is what
separates a scoped change from an untested one.

**`exact` as a *fraction* falls on cases 5, 11 and 12 while its *count* rises.** Case 5 gains 23,473
both-measured points and 596 more exact ones; case 11 gains 12,862 and 68. The added points are the ones
the zero shift had been placing half a cell from where `_undecimate_level` read them back, and they land
at coarse levels where neither side is quantized — so a fraction over a grown population is not
comparable to itself. The counts are, and they rise on every case that moved except where `both` fell.

**Two cases lose coverage** — case 7 by 472 and case 10 by 24, against gains of 23,473 and 12,862
elsewhere. A shifted coarse node can fall outside the image where the unshifted one did not, so a small
two-sided movement is the expected signature rather than a regression.

**Case 12's bias moved, and it was the one recorded as not moving.** `LT05_001013` was flagged above as
the largest bias in the optical set at −0.0847 px, unchanged by the `Dy0` fix and therefore not a
sign-convention artifact. The grid-step fix takes it to **−0.0691** and its coverage from 18,540 to
19,408. So part of it was the missing cell-centre shift. At 4.5× the next largest core bias it is still
the outlier in the set, and the residual is still unexplained.

**Case 4's core bias improves by 5×**, −0.0226 → +0.0042, and case 6's by a fifth. Both are cases whose
`x` step is −1 — a grid rotated near 90°, where a wrong shift moves a node along the wrong axis entirely.

**Coverage across all twelve**: 6,880,833 both-measured, 232,610 julia-only, 333,402 reference-only —
4.85% of `both`. The reference-only fraction is not uniform: 1.4–2.0% on the three `hps` cases with the
highest agreement, against 23.8% on case 9 and 15.0% on case 11, both `wallis_fill`/`fft` cases whose
base level is skipped or nearly so.

**Eleven of twelve report a zero tail beyond 10 px.** The exceptions are 2 points of 605,987 on case 3
(`dy`) and 1 of 692,088 on case 4 (`dx`); case 4's `dx` maximum of 11.03 px is the largest single residual
in the set.

## Step: the eight radar pairs, re-measured after the stride and grid-step fixes

`3.rdr` was the last gate standing on superseded figures. **All eight captures are still on disk** —
`call1.json` present in every `200/capture` — so this needed no container run. The eight endpoints take
**14m37s** total on 10 threads.

Both fixes move every pair, on both axes. The stride change is larger here than anywhere else in the set:
three pairs decimated by **1, 1, 1, 1** before, so no level coarsened at all.

| case | `ScaleChipSizeY` | chip0 | old stride | new stride | old step | new step |
|---|---:|---|---|---|---|---|
| `20150828`, `20151120`, `20250416T010214`, `20240618T025533` | 0.2500 | 64x16 | 1, 1, 1, 2 | 1, 2, 4, 8 | **0, 0** | varies |
| `20170221`, `20180809`, `20240618T025528` | 0.2353 | 68x16 | **1, 1, 1, 1** | 1, 2, 4, 8 | **0, 0** | varies |
| `20250416T010159` | 0.2857 | 56x16 | 1, 1, 1, 2 | 1, 2, 4, 8 | **0, 0** | −6, 2 |

The endpoint, whole grid, against "The eight, all green" above. That baseline predates the `Dy0` fix as
well, so this is the effect of all three defects together and not of the stride alone:

| # | case | driver | both | Δ both | exact | only jl | only ref | Δ only ref | core bias dx / dy | corr dx / dy | tail | gate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `S1A_..._20150828T162412` | SLC | 1,310,564 | −17,054 | 29.41% | 35,053 | 86,884 | +17,054 | **−0.00000** / −0.00029 | +0.903 / +0.906 | 0 | green |
| 2 | `S1A_..._20151120T080202` | SLC | 60,072 | −539 | 0.00% † | 8,668 | 10,436 | +539 | −0.00428 / +0.00408 | +0.968 / +0.930 | 167 | green |
| 3 | `S1A_..._20170221T204710` | SLC | 1,666,631 | **+132,369** | 69.08% | 23,633 | 50,508 | **−132,369** | +0.00715 / +0.00239 | +0.996 / +0.985 | 35 | green |
| 4 | `S1B_..._20180809T204617` | SLC | 489,946 | +35,632 | 69.26% | 12,453 | 35,528 | −35,632 | +0.00479 / +0.00475 | +0.987 / +0.971 | 0 | green |
| 5 | `S1C_..._20250416T010214` | SLC | 484,471 | +8,065 | 46.54% | 26,585 | 46,362 | −8,065 | +0.00560 / −0.00155 | +0.983 / +0.896 | 0 | green |
| 6 | `S1C_..._20250416T010159` | BURST 7 | 35,986 | +2,773 | 36.17% | 3,499 | 5,949 | −2,773 | **−0.04313** / **−0.01656** | +0.941 / +0.850 | 0 | **red** |
| 7 | `S1A_..._20240618T025533` | BURST 10 | 546,899 | **+106,874** | 63.24% | 34,030 | 36,156 | **−106,874** | **+0.01181** / +0.00556 | +0.992 / +0.982 | 14 | **red** |
| 8 | `S1A_..._20240618T025528` | BURST 24 | 1,505,773 | **+274,063** | 54.77% | 73,479 | 84,463 | **−274,063** | −0.00107 / +0.00010 | +0.988 / +0.961 | 12 | green |

† base level runs a coarse pass and a `filtDisp` but no fine pass, so every point is bicubic-resized
rather than quantized and `exact` is 0 by construction.

**Gate `3.rdr` is 6 of 8**, red on cases 6 and 7 against its `0.010 px` core-bias threshold.

**The coverage gap closes, as it did on NISAR.** `only ref` falls **898,469 → 356,286**, from 16.16% of
`both` to **5.84%**, and `both` rises 5,558,159 → 6,100,342 (**+542,183**). Every reference-only point
recovered becomes a both-measured one — the two Δ columns are equal and opposite on all eight rows, so
nothing moved to `only jl`, which itself falls 465,303 → 217,400. The direction matches the NISAR L2
result and has the same cause: the coarse levels finally run on the reference's own lattice.

**Correlation improves on 7 of 8 on each axis.** Case 1 gains most, +0.820 → **+0.903** on `dx` and
+0.825 → +0.906 on `dy`, and its core `dx` bias is now **−0.0000005 px**. Case 7's `dx` slips by 0.001
and case 8's `dy` rises 0.951 → 0.961.

**`exact` falls as a fraction on six of eight while `both` rises**, the same construction as on the
optical and NISAR cases: the recovered points are coarse-level ones where neither side is quantized, so
they cannot be exact. **The exact *count* rises on six and is flat on the other two.** Case 8 is the
extreme — 65.81% → 54.77% on a population that grew by 274,063 — and its count still rises, from a
baseline percentage implying 810,526–810,650 to a measured **824,654**. Case 1's count is flat at 385,404,
which is the one case whose population *fell*; case 2's is 0 either way by construction.

### The two red cases are both `process_burst`, and the bias is systematic

Case 6's core `dx` bias is **−0.0431 px**, 4.3× the gate threshold and **62× its former −0.0007**. Case 7
is +0.0118, 10.7× its former +0.0011. The five `process_slc` pairs stay within 0.0072.

It is not a tail-cancellation artifact, which is what the core statistic exists to exclude: case 6's mean
is −0.0642 and its core −0.0431, so the core carries **67%** of the mean, and case 7's carries 88%. A
two-sided tail would leave the core near zero, as it does on case 2, where the mean is dragged while the
core sits at −0.0043.

Two properties separate the red pair from the green ones, and neither is established as the cause:

- **Both are burst-mosaicked**, but so is case 8, which is the *best* core bias in the set at −0.0011.
  So `process_burst` alone does not predict it; 7 and 10 bursts do while 24 does not.
- **Case 6 is the only `ScaleChipSizeY = 0.2857` pair** and has the smallest chip in the set at 56x16.
  It is also the smallest population, 35,986 points, and had the second-lowest `dy` correlation before
  the fix.

Case 6's `dy` core bias of −0.0166 is also over threshold, so it is not a single-axis effect. Both cases
report a `dx` maximum well above their `dy` one — 6.97 against 2.18 on case 6, 18.04 against 5.93 on
case 7 — so whatever it is, it is stronger along x, which is the axis the stride is now derived from.

**The gate's threshold is not obviously the thing to change.** It was calibrated as "the weakest measured
case less a margin" against figures that are now known to have been measured through three defects, so it
describes the old behaviour rather than a physical requirement. But the two fixes improved coverage and
correlation on these same two cases while the bias grew, which is not the signature of a threshold set
too tight — a case whose systematic offset over its agreeing population grew 62× has changed behaviour
that wants an explanation, not a wider bound. `3.rdr` stays red on 6 and 7 until there is one.

## Step: the serial phase was the FFTW planner

The one-core phase this file records twice — L1 running ~7.8 cores for 2.5 hours then dropping to 1.0 —
is **FFTW plan construction under `FFTW_PATIENT`**. Not the hole fill, the merge or the comparison, each
of which was guessed at and none of which is where the time goes.

**How it was localized.** `sample(1)` on the live process, on an otherwise idle machine so contention
could not explain it: every worker thread parked in `uv_cond_wait`, and the one busy thread was the
*main* thread inside `libfftw3f`, running many different radix codelets (`hb_12`, `r2cb_32`, `n1_64`,
`fftwf_cpy2d_pair_ci`) beneath `apply` at top level. That is the planner timing candidate algorithms,
not a correlation executing. Two candidates were eliminated by direct measurement first: `_fill_holes!`
takes **0.6 s** on the full 2328x2304 L1 grid with its real 66.4% hole pattern, and chunk load imbalance
gives a greedy makespan 1.16x ideal, not the 4.2x an earlier `(2r+1)^2` cost proxy suggested.

**Why `PATIENT` was the wrong flag, measured rather than argued.** Cold plan against warm execution, both
flags, real-to-complex forward:

| size | PATIENT plan | MEASURE plan | execution gain | repaid after |
|---|---:|---:|---:|---:|
| 28x28 | 97 ms | 0.0 ms | **0.98x** — slower | never |
| 84x84 | 517 ms | 0.0 ms | **0.93x** — slower | never |
| 84x160 | 1479 ms | 0.0 ms | 1.13x | 905,696 executions |
| 320x640 | 6346 ms | 0.0 ms | 1.01x | 4,427,357 executions |
| 576x1152 | 16,122 ms | 0.1 ms | 1.00x | never |
| 2304x4608 | **306,787 ms** | 3873 ms | 1.05x | 13,016 executions |

Five minutes to plan one 2304x4608 transform, which is what a 1905-pixel search radius on the L1 grid
reaches. And `src/plans.jl`'s recorded justification for `PATIENT` — 1.41x at 28², 1.28x at 84² — **does
not reproduce**: at those sizes it is not faster at all. `PLAN_FLAGS` is now `FFTW_MEASURE`.

**The endpoint, same window and machine, `-t 10`.** A 201x201 window of the L1 grid, 40,360 searchable
points:

| | before | after | |
|---|---:|---:|---|
| wall clock | 1576.3 s | **104.6 s** | **15.1x** |
| per searchable point | 39.06 ms | **2.59 ms** | |
| peak RSS | 57.97 GiB | 61.45 GiB | 1.06x |

**Peak RSS rises 6%, which is unexplained.** The prediction from the workspace arithmetic was 1.00x, so
this is 6% unaccounted for rather than a known cost. It is not the dominant term in production — imagery
is lazy and a block reads its own window — but the discrepancy stands.

### What it costs in agreement: 0.0013% of one case's exact count

A point is now correlated at its own radius bucketed to a power of two and clamped to the pass maximum
(`AutoRIFT._radius_bucket`), rather than at the pass's widest radius. Transform length reassociates the
same floating-point sum differently, so this perturbs results by construction and every golden figure is
in principle affected. Measured on `LC08_L1TP_009011`, 1,685,673 both-measured points:

| quantity | before | after | delta |
|---|---:|---:|---:|
| both-measured | 1,685,673 | 1,685,673 | **identical** |
| only reference | 30,387 | 30,387 | **identical** |
| only julia | 19,505 | 19,571 | +66 (0.0039% of what julia measures) |
| `dx` exact | 70.87% | 70.87% | identical to 2 dp |
| `dx` exact count | 1,194,637 | 1,194,621 | **−16 of 1.19 M** |
| `dy` exact count | 1,195,999 | 1,195,987 | −12 |
| `dx` correlation | +0.99931 | +0.99931 | identical to 5 dp |
| `dy` correlation | +0.99904 | +0.99904 | identical to 5 dp |
| bias core `dx` | −0.008884 | −0.008883 | +7.6e-07 |
| bias core `dy` | −0.005186 | −0.005185 | +1.8e-06 |
| wall clock | 332.0 s | **75.9 s** | **4.4x** |

So the drift is at the seventh decimal on bias and 16 points in 1.19 million on exact agreement — far
below the level any gate reads, and below the run-to-run wisdom variability this file already records
(`correlation` reproducible to ~1e-7 against a fixed wisdom file, not absolutely). The 4.4x on an
*optical* case is worth noting: optical radii are barely skewed, so almost all of that is the flag.

**Both changes are needed together and neither is sufficient.** Bucketing alone multiplies the plan
count — 43 distinct sizes for the L1 window against 4 — which under `PATIENT` is a large regression, not
a gain. The flag alone leaves every point executing the widest point's transform. The commits are
separate so each is bisectable, but the first is not an improvement on its own.

`Pkg.test()`: **704,173 tests pass**, and the suite itself drops from 16m48s to **11m35s** — the tests
were paying the same planning cost.

## Step: a per-block halo is not worth having, and the reason is arithmetic

The halo is one `Extent` on the `BlockLayout` (`src/tile.jl:60`), taken from the whole grid's maxima
and applied to every block. Since the correlation reach is per point and a Geogrid radius field is
spatially clustered, deriving each block's halo from its own points looks like a large saving. It is
not, and this records the measurement so it is not re-attempted.

**Implemented and measured, then reverted.** A `_block_halo` reducing over each block's own sanitized
radii and priors, capped at the grid-wide halo, with `read_rows`/`read_cols` per block — which are
already per-`Block` fields, so the plumbing needed nothing. All 36,492 tiling tests passed, so a
blocked run still equalled an untiled one. The saving in imagery read, on the real L1 grid:

| block (grid points) | blocks | adaptive | shared | saving |
|---:|---:|---:|---:|---:|
| 64 | 1332 | 454.4 GiB | 505.5 GiB | **1.11x** |
| 128 | 342 | 209.2 GiB | 226.8 GiB | 1.08x |
| 256 | 90 | 97.0 GiB | 102.0 GiB | 1.05x |
| 512 | 25 | 45.3 GiB | 46.6 GiB | 1.03x |

**Why the gain is 5% and not the 18x an earlier estimate gave.** That estimate assumed a block with no
wide-radius point gets a small halo. It does not: the halo is
`chip_size_max/2 + radius + |prior| + 2 + filter_reach + level_centre_offset`, and only the `radius`
term is per point. `chip_size_max/2` alone is 384 px on this configuration, so the *floor* on a
per-block halo is **561 px** against a grid-wide 2684 — and the median block reaches the full 2684
anyway, because 53-60% of blocks contain at least one wide-radius point at every block size tried. A
clustered radius field is not clustered finely enough to matter at block scale.

## Step: the wide halo is a skewed search radius, and the radius is the geogrid's own

The halo above is dominated by the search radius, and a radius of 1905 px is worth interrogating before
it is designed around. It survives interrogation: it is what the geogrid produced.

`window_search_range.tif` — the geogrid's raw output, before `autoRIFT` reads anything — carries band 1
min 0, **max 1905**, mean 72.5 and band 2 min 0, **max 830**, mean 41.6. Those are bit-identical to
`in_SearchLimitX`/`in_SearchLimitY` in the capture, so nothing between the geogrid and the correlator
rescales them.

**The nodata value is not being read as data**, which is the first thing to suspect of a field whose
maximum is 73x its median. The fill value is `-32767` in `window_search_range.tif` and
`window_offset.tif`, and the ITS_LIVE parameter rasters use `32767` for the search ranges and `-32767`
for the velocities. None of the four values appears anywhere in the captured grid: no `32767`, no
`-32767`, nothing with `|v| > 3000`, no `NaN`. The observed maxima in the source rasters — 11576 m/yr
for `vxSearchRange`, 8147 for `vySearchRange` — are that data's own extrema, reported by `gdalinfo` as
statistics separate from the declared nodata.

**The field is genuinely skewed rather than corrupt.** Over the 2.3 M points with real coordinates and
a nonzero chip size, `SearchLimitX` has median 26 and p99 959, then 1486 at p99.9, 1592 at p99.99 and
1905 at the maximum — a continuous tail, not the spike of identical values a misread fill would give.
The maximum is reached at exactly **4 points**, rows 1246-1247 and cols 1105-1106, a 2x2 cluster in the
scene interior; the `-640` prior is 54 points at rows 1316-1333, cols 1113-1140, spatially adjacent to
it. One fast feature, not a fill artifact.

**Converting a pixel radius to a velocity needs `off2vel`, not the pixel spacing.** This is where a
check of whether 1905 px is physically plausible goes wrong: `radius * spacing / days * 365.25` assumes
an offset maps to ground displacement through the pixel size, which in radar geometry it does not. The
geogrid stores the correct projection in `window_rdr_off2vel_x_vec.tif`, whose band 1 means 14.2 m/yr
per pixel of range offset — against 19.4 implied by the naive form, so that step alone is 1.37x off.
Through the stored conversion the median 26 px is 369 m/yr, p99 959 px is 13.6 km/yr and the maximum
1905 px is 27 km/yr.

27 km/yr is still fast for ice, and `off2vel` band 1 itself spans -377 to +393 across the scene, so a
fixed pixel radius maps to wildly different velocities depending on where it sits. Whether those 4
points are fast ice or poorly-conditioned geometry is a geogrid question and is left open. What is
settled is that they are the geogrid's own numbers, correctly carried, with the nodata handled.

**A second finding, which is the one worth acting on.** The guard at `src/tile.jl:236` compares the
requested block size against the grid-wide halo and rejects anything smaller, so on this case the
smallest permitted block is **2684x1448 px** — 6.8 by 6.4 km at this granule's 2.55 m ground-range and
4.44 m along-track spacing.

**That guard is not what stopped blocking here, and an earlier version of this section said it was.** A
2684 px block divides a 57760x50511 scene about 19 by 20 ways. Yet every requested block size from 3072
up to 16384 px returned **one block**, which no halo argument explains — the halo only sets a floor.

Two independent defects, both silent, and both now fixed.

**The grid is not separable.** `block_layout` derived its block boundaries from `grid.y[:, 1]` and
`grid.x[1, :]`, on the assumption a gridded `PointSet` repeats each coordinate down every row and across
every column. A NISAR geogrid is a rotated radar footprint sampled onto a map grid, so it is not
axis-aligned in pixel space: `x` varies by 50502 px down column 1152, `y` by 43164 px across row 1164,
only 43.2% of the 2328x2304 points carry real coordinates, and row 1 and column 1 contain **one** valid
point each. Walking them spanned 216 px rather than the scene, so one block appeared to cover
everything.

The grid is itself the index-to-pixel mapping, so the block shape now comes from it: `x` moves 33 px per
row of index *and* 33 px per column, `y` moves 19 px per each. A block of `a` by `b` index points spans
`a*33 + b*33` px of `x`, and both axes have to fit. Sizing each axis from its own budget alone —
`rowpts = py/dy_di`, `colpts = px/dx_dj` — gives 215 by 124 points at an 8192-px request, whose `x` span
is 11187 px, a 37% overshoot. On an axis-aligned grid two of the four rates are zero and this reduces to
the separable answer.

**The read window spanned unsearchable points.** `_pixel_span` reduced over every point in the block,
and outside the footprint the coordinates are *fill* — zero on this grid, not `NaN`, so finiteness does
not detect them. A block straddling the footprint edge spanned 0 to the real coordinates and read
`1:57760`: at an 8192 px block, 28 of 209 blocks each read half the scene or more, 99x the scene in
total. `_searchable_span` now reduces over searchable points only, which is sound because
`_run_one_block!` returns before any I/O for a block with nothing to search.

Measured after both fixes, on the captured L1 grid:

| block | blocks | max read window | read amplification |
|---:|---:|---:|---:|
| 4096 px | 1444 | 5217x9747 | 8.31x |
| 8192 px | 361 | 7572x14020 | 4.34x |
| 16384 px | 100 | 12277x22519 | 2.85x |

So blocking *is* available on a NISAR grid, at a block size scaled to the halo. The amplification is set
by the 2684x1448 px halo rather than by the layout: even a 16384 px block pays 2.9x, which is what a
fixed-width halo costs when it is a large fraction of the block. Axis-aligned grids are unaffected — the
Landsat sweep in `docs/src/explanation/memory.md` reproduces its previous block counts and amplification, a full-width
band stays a band, and `dx`/`dy`/`correlation` stay bit-identical to an untiled run at every block size.

## Step: what the 44 GiB peak is, and it is not the imagery

The obvious reading of a whole-scene peak near 50 GiB is that the scene is resident and should be read
lazily. Measured on the L1 window at `-t 10`, that is wrong on both counts.

| component | GiB |
|---|---:|
| imagery, both scenes as captured `UInt8` | 5.43 |
| everything reachable **before** the pass | 6.26 |
| maximum reachable **during** the pass (`gc_live_bytes`) | **50.40** |
| peak RSS | 44.17 |

Three findings, each of which rules something out:

- **No padded copy is made.** `_pass_geometry` reports `fits = true` on this window, so the pass reads
  the captured arrays in place. Padding is a real cost on a grid whose points reach outside the scene,
  but it is not what this peak is.
- **The memory is reachable, not uncollected.** `gc_live_bytes` peaks slightly *above* RSS, so the
  collector is not behind — the process genuinely holds it.
- **Lazy imagery would recover 5.43 GiB of ~50.** In production it is worth having, since the blocked
  path reads a window per block; it is not the dominant term and it is unavailable in the harness
  anyway, where `read_capture` materialises `UInt8` matrices from disk.

**The dominant term is workspaces, and their count is larger than it should be.** After one pass the
pool holds **51 workspaces across 28 keys, 9.97 GiB**, with the per-key cap of 2 working as intended.
28 keys is the problem: `_radius_bucket` clamps to *each level's* own maximum radius, and the levels'
maxima differ — 1905x830, 1918x1015, 1689x907, 1828x972 — so the top bucket is a different geometry at
every level rather than one shared entry. **7.74 of the 9.97 GiB is those eight near-duplicate top
buckets.**

Dropping the clamp so every bucket is a clean power of two collapses them to one key per chip size, but
costs 11-45% more transform area for every point in the top bucket (measured: `1792x4096` becomes
`2304x4608` at chip 96x52), which is the regression the clamp exists to prevent. Clamping every level to
one grid-wide maximum instead would collapse them to four keys — but a level's radii can *exceed* the
grid's, since `sanitize!` floors them and the coarse pass rewrites them per level (1918 against a
grid-wide 1905), so a single clamp needs that relationship established first rather than assumed. Left
open deliberately.

## Step: blocking on a real geogrid, and what the outlier filter does to a last-bit difference

Fixing the rotated-grid layout above made blocking actually divide a golden grid, and the first real
multi-block run did not reproduce the untiled answer. **The blocked correlation is right; the
disagreement is the outlier filter amplifying a floating-point difference the package already
documents.** Recorded because the intermediate readings each pointed somewhere else.

Measured on `S2B_MSIL1C_20200612`, a 10980x10980 scene on a 1008x1008 grid with 787,186 searchable
points, at a 2048 px block giving 144 blocks:

| configuration | untiled | blocked | differing |
|---|---:|---:|---:|
| full ladder, `outliers` default | 612,607 | 611,467 | 3713 |
| single chip size, `outliers` default | 481,037 | 481,036 | 1 |
| single chip size, `outliers = :none` | 787,190 | 787,190 | **0, bit-identical** |

The last row is the finding. With rejection off the two paths agree to the last bit, so nothing about
reading, filtering, padding or the layout differs.

**What the one point is.** Grid (918,827), `x = 9812.5`, `y = 10010.5`, radius 6, chip 24, zero prior.
Correlated in isolation through its own block it answers `dx = 0.1875`, `correlation = 0.1534135` —
exactly the untiled values — and its `peak_ratio` comes out **1.6192024 through the whole grid's pass
geometry against 1.6192014 through the block's own**. That is a 6e-7 relative difference in a *quality
metric*, from executing a different-sized transform: `src/plans.jl` already records that `correlation` is
reproducible only to ~1e-7 while `dx`/`dy` are bit-identical, because a peak's location is insensitive to
a perturbation that size. Here the metric feeds `reject_outliers`, which compares each point against its
neighbours and takes a keep-or-drop decision — so a 6e-7 difference lands on either side of a threshold
and the point is kept untiled and dropped blocked.

The ladder then multiplies one point into 3713. A point the base level drops changes what the coarse
gate, the hole fill and every level above it see, so the discrepancy compounds rather than accumulating
linearly.

**Why the synthetic suite cannot see this.** 36,620 tiling assertions pass, and every grid they build
comes from `gridpoints`: uniform radii, no fill, and — before the layout fix — a golden grid collapsed to
one block, so the harness compared an untiled run against itself and reported agreement. That agreement
was vacuous. Reproducing the failure needs a grid whose radius field is skewed enough that a block's own
pass geometry differs materially from the whole grid's, which is a geogrid property.

**Four things this is not**, each measured and ruled out before the above was found: a halo effect at a
seam (only 9% of discrepant points lie within 2 grid points of a boundary, against ~13% by chance);
padding over real imagery (37 of 144 blocks do get `_zeropad`ed, all at the scene edge where the untiled
pass pads too, and a synthetic grid whose points deliberately reach outside the scene stays
bit-identical); per-point radius variety, `preprocess = :none`, a clustered unsearchable region, or a
`grid_spacing` disagreeing with the grid's true spacing (48 declared against 12 measured here — all four
reproduce bit-identically in isolation); and the chip ladder or the fine rejection themselves, since one
point still differs with a single chip size.

**What it means for the bit-identity promise.** `docs/src/explanation/memory.md` states bit-identity as one of two things
`process_block_size` guarantees. That holds for the correlation and fails for the *rejection decision* on
a grid with a skewed radius field, because the decision is a threshold on a quantity only reproducible to
~1e-7. Two honest resolutions, neither applied: hand every block the whole grid's pass geometry for the
quality metrics as well as the transform — which `_run_blocked` already does for `geometry`, so the
remaining difference is that a bucket's workspace is sized to the bucket — or state the promise as
bit-identical `dx`/`dy` *given the same keep mask*, and treat the mask as reproducible only where no point
sits within ~1e-6 of a threshold. The Landsat sweep in `docs/src/explanation/memory.md` is unaffected either way: uniform
radii mean a block's geometry equals the grid's, and those runs are bit-identical at every block size.

## Step: what a whole NISAR scene costs, and a GC deadlock under contention

Blocking works on these grids once the layout is fixed, and the figures are what a production instance
would be sized from. Measured at `-t 10` on an M2 Max with 96 GiB, whole grid, one process per row.

**NISAR L1 RSLC** — 57760x50511 px, grid 2328x2304, halo 2736x1500 px, 1.87 M searchable points:

| block | blocks | runtime | peak | vs untiled | read amp | measured |
|---|---:|---:|---:|---:|---:|---:|
| untiled | 1 | 10.9 min | **55.2 GiB** | 1.00x | 1.00x | 1,798,199 |
| 16384 px | 100 | — | 68.6 GiB | 1.24x | 2.89x | killed |
| 8192 px | 380 | 41.1 min | 34.8 GiB | 0.63x | 4.51x | 1,797,076 |
| 4096 px | 1482 | 45.0 min | **31.5 GiB** | 0.57x | 8.77x | 1,797,076 |

**Superseded — see "Step: the L1 sweep, re-measured on an idle machine" below.** Every runtime in this
table is inflated, and every peak is a few GiB high, because these rows shared the machine with other work.
The re-measured figures are 9.5 min untiled and 24.4-30.9 min across the blocked rows.

**The peaks are the measurement; the blocked runtimes are upper bounds.** Those two rows were re-measured
after `_index_rate` was corrected, on a machine that was also running the L2 job for part of their life. An
earlier uncontended pass over the same block sizes — at the flawed rates, so a slightly different partition
— gave 18.4 and 16.9 min. Peak RSS is insensitive to a competing process in a way wall clock is not, so the
memory column stands and the time column wants a quiet machine before it is quoted.

**NISAR L2 GSLC** — 54885x110085 px (6.0 Gpx), grid 2288x2288, halo 2216x1103 px:

| block | blocks | runtime | peak | vs untiled | read amp | measured |
|---|---:|---:|---:|---:|---:|---:|
| untiled | 1 | 12.1 min | **80.9 GiB** | 1.00x | 1.00x | 1,781,775 |
| 8192 px | 98 | **11.9 min** | **40.8 GiB** | **0.50x** | 0.86x | 1,781,377 |

**Both runtimes in this table are superseded** — see "Step: the L2 block-size sweep" below, which spans
five sizes with the JIT warmed and times each row with the profiler off. The untiled row here carries the
process's compilation, and 8192 px is not the size to choose.

**This is the case that makes blocking a production requirement rather than a tuning knob: peak halves at
identical runtime.** 80.9 GiB against 40.8 on a machine with 96, and 99.98% of the untiled point count. An
instance sized from the untiled figure is memory-optimized; one sized from the blocked figure is not.

Three things worth stating.

**The untiled peaks are the reason blocking matters here.** 55 GiB on L1 and **81 GiB on L2, on a 96 GiB
machine** — a production instance sized from the L2 figure is a memory-optimized instance costing several
times a general-purpose one, for a scene blocking runs at 37 GiB.

**A block can be too large, and the crossover is arithmetic.** `BlockBuffers` holds nine block-sized
arrays — two `UInt8` planes, three `Float32`, four `Bool` — totalling **18 bytes per pixel**, one set per
task, so a run holds `min(nblocks, nthreads)` sets. At 16384 px on L1 the read window is 12232x22222,
which is 4.56 GiB per set and **45.6 GiB across ten tasks** before any imagery or workspace: measured peak
68.6 GiB against an untiled 55.2. The prediction and the measurement agree to 1%, so the rule is usable
rather than empirical — compute `18 bytes x (block + 2*halo)^2 x min(nblocks, nthreads)` and keep it well
under the untiled peak.

The 18 bytes are the total across the nine arrays, not each array's share. `tools/golden/profile_nisar.jl`
holds the constant and measures it off the struct's own fields (exactly 18.0 B/px at a 2000x1500 window);
reading it as 18 bytes *per array* predicts 410 GiB for that L1 window and rejects every block size the
granule runs at.

**L2 reads *less* than the scene when blocked** — 0.87x at 8192 px, against 4.51x for L1 at the same size.
Two reasons compound: its halo is smaller relative to the block, and 64% of its grid is fill, so those
blocks have no searchable point and `_searchable_span` gives them an empty read window. A read
amplification below 1.0 is the signature of a grid whose footprint does not fill its bounding box.

### The deadlock, as first seen — superseded by "The root cause" below

The heading this section carried, "which is contention-dependent", was wrong; keep reading to the root
cause rather than stopping here.

**Reproduced once, then not.** The first attempt at the L2 8192 px row reached 44.4 GiB and stopped dead:
0% CPU across three `sample` runs six minutes apart, unresponsive to `SIGTERM`, killed with `SIGKILL`. All
31 threads sat in a wait — 16 in `__psynch_cvwait`, 4 in `__psynch_mutexwait` — and each of the four was

    ijl_gc_small_alloc / ijl_gc_managed_malloc -> ijl_gc_collect -> jl_safepoint_start_gc -> uv_mutex_lock

so every thread that tried to allocate was queued behind a collection that never started. The trace is at
`~/data/autorift/tests/golden_tests/mem/l2_deadlock_sample.txt`.

**The same configuration then completed cleanly on an idle machine**, in 716 s at 41.8 GiB peak — the row
in the table above. The difference between the two runs was a concurrent L1 job holding tens of GiB, so at
the time this read as a deadlock needing memory pressure from *outside* the process.

**That qualification is wrong on both counts.** It reproduces on an idle machine, and it needs neither
external pressure nor a multi-configuration process: the macOS profiler suspends a thread mid-way through
libpthread's thread-list lock and then blocks on that lock itself, inside the Julia runtime and
reproducible in a script with no AutoRIFT in it. See "The root cause" below. Because it is a race, the
clean retry that motivated the contention theory was simply a run that did not lose it.

Not an artifact of the harness. The sampler thread `mem_nisar.jl` runs appears on no stack in the trace,
and it allocates only two small vectors per sample.

### The deadlock, requalified: an idle machine is enough

**Reproduced on an idle machine with nothing else running**, so the "needs external memory pressure"
qualification above does not hold. Hit while sweeping smaller block sizes: the 2816 px row stopped dead at
**25.4 GiB** — a third of the untiled peak, and well under the 44.4 GiB of the first occurrence — after its
2304 px predecessor had completed normally in the same process.

Same signature as the original, at a different block size and a much lower footprint:

| | first occurrence | this one |
|---|---|---|
| block | 8192 px | 2816 px |
| footprint at stop | 44.4 GiB | **25.4 GiB** |
| other load on machine | concurrent L1 job, tens of GiB | **none** |
| `__psynch_cvwait` | 16 | 16 |
| `__psynch_mutexwait` | 4 | 4 |
| through `ijl_gc_collect -> jl_safepoint_start_gc -> uv_mutex_lock` | yes | yes |

**No thread is collecting.** Zero frames in the sample match `gc_mark`, `sweep` or `gc_scan`, so this is a
collection that never starts rather than one taking a long time — every allocating thread is parked at the
safepoint waiting for a collector that does not exist. Trace at
`~/data/autorift/tests/golden_tests/mem/l2_deadlock_2816_sample.txt`.

**Root cause: the profiler suspends a thread that is holding libpthread's global thread-list lock, then
blocks on that same lock.** A Julia runtime bug, not memory pressure, not the block size, and not this
package — see "The root cause" below. The multi-configuration correlation is incidental; what matters is
that a profiled run allocates hard on many threads for long enough to lose a race.

**A second stuck process was found at the same time**, left over from the five-configuration sweep: 0% CPU,
16 threads in `__psynch_cvwait`, still resident at 11.8 GiB. It had been assumed dead — no output, no
exception, empty stderr — and the assumption was wrong in a way worth recording: **a Julia process
deadlocked this way looks exactly like one that was killed**, since both stop writing and neither leaves a
message. Check `ps` for a 0%-CPU survivor before concluding a run died, and `sample` it before killing it.

The earlier note in this file attributing the 3072 px sweep row to an OOM kill is superseded: that process
was hung, not killed.

### The root cause: the profiler suspends a thread holding libpthread's thread-list lock

**A Julia runtime bug in `src/signals-mach.c`, present in every release through 1.13.0, and nothing to do
with AutoRIFT.** It is *not* a symmetric lock-order inversion between two lock types — an earlier reading
of these traces recorded it that way and was wrong. There is **one** lock, and the profiler freezes its
holder:

`pthread_mach_thread_np(t)` looks `t` up in libpthread's global thread list under
**`_pthread_list_lock`**, an `os_unfair_lock`. A lookup of *another* thread must take it; a self-lookup
takes a lock-free fast path. Both parties here look up other threads.

1. A Julia thread finishes a collection, enters `jl_mach_gc_end` (`signals-mach.c:97`), and calls
   `thread_resume(pthread_mach_thread_np(ptls2->system_id))` to wake the threads it stopped. It is now
   **inside libpthread holding `_pthread_list_lock`**.
2. The profiler's sampling thread picks that thread as its next target and, in
   `jl_thread_suspend_and_get_state2`, calls `thread_suspend` on it — freezing it *mid-critical-section
   with the lock held*. `jl_profile_thread_mach` then unwinds the target's stack and only calls
   `jl_thread_resume` at the very end.
3. Before reaching that resume, the sampler needs `pthread_mach_thread_np` again, blocks on
   `_pthread_list_lock`, and waits on a lock whose holder is suspended and can only be resumed by the
   sampler itself. **The sampler deadlocks against a thread it stopped.**

The lock is process-global, so the two need not be interacting through Julia at all — which is why block
size, memory pressure and the multi-configuration harness are all irrelevant to it.

**The traces say exactly this, and the offsets are the evidence.** Only ever *two* threads are inside
`pthread_mach_thread_np`, at different offsets and with different frames beneath:

| thread | offset | frame beneath | reading |
|---|---|---|---|
| sampler (`jl_profile_thread_mach:797`) | `+56` in every trace | `_os_unfair_lock_lock_slow` → `__ulock_wait2` | **blocked acquiring** the lock |
| victim (`jl_mach_gc_end:97`) | `+76`, `+164` — varies | **none** | **suspended holding** it, frozen at an arbitrary instruction |

A blocked thread is always at the same instruction; a *suspended* one stops wherever it happened to be,
which is why the victim's offset differs between traces and the sampler's never does. And **no thread is
collecting** — zero frames matching `gc_mark`, `sweep` or `gc_scan` — so this is a collection that cannot
finish rather than one taking a long time. Every remaining thread queues at `jl_safepoint_start_gc` behind
it.

**Reproduced without AutoRIFT.** `tools/golden/profiler_gc_deadlock.jl` is allocation churn on every
thread while `Profile` samples at 0.5 ms — no imagery, no correlation:

```bash
julia -t 10,1 tools/golden/profiler_gc_deadlock.jl off       # always completes
julia -t 10,1 tools/golden/profiler_gc_deadlock.jl profile   # hangs ~2 runs in 5
```

Measured 2 hangs in 5 attempts, at rounds 25 and 106 of 200, with the same two-thread signature; the
control arm runs the same workload unprofiled and has never hung. **It is a race, so one clean run proves
nothing** — which is exactly the trap that made the first occurrence look contention-dependent after it
completed on a retry.

**Fixed upstream, but not in any release yet.** `ca49fc2e2` ("[macOS] Handle GC safepoint on-thread",
2026-03-17) deletes `jl_mach_gc_end` and the `suspended_threads` list outright and handles the safepoint on
the signalled thread — so no thread calls `pthread_mach_thread_np` to resume another, and step 1 above
cannot happen. Checked against the tags: `jl_mach_gc_end` is still present in v1.12.5, v1.12.6,
v1.13.0-beta1, v1.13.0-rc1 and v1.13.0, and gone in 1.14-DEV. Not backported to `release-1.12` or
`release-1.13`.

**What this means for the measurements.** Nothing in the recorded figures is suspect: peaks and runtimes
come from unprofiled runs, and a hang costs an attribution rather than a measurement. It does mean the
profiled arm of a long threaded run on macOS may need retrying, and that `profile_nisar.jl`'s split
between a timed run and a separately profiled one is load-bearing rather than tidiness. Production is
unaffected — a worker that does not profile never starts the sampler thread.

## Step: the L2 block-size sweep, and what a threaded whole-granule run actually costs

Seven configurations on NISAR L2 GSLC, whole grid, `-t 10,1` on the M2 Max with 96 GiB. Command:

```bash
julia --project=tools/golden -t 10,1 tools/golden/profile_nisar.jl NISAR_L2_PR_GSLC \
    --blocks 0,8192,6144,4096,3072
julia --project=tools/golden -t 10,1 tools/golden/profile_nisar.jl NISAR_L2_PR_GSLC --blocks 3072
julia --project=tools/golden -t 10,1 tools/golden/profile_nisar.jl NISAR_L2_PR_GSLC --blocks 2304x1152
```

`profile_nisar.jl` differs from `mem_nisar.jl` in three ways that each moved a number: it warms the JIT
before recording anything, it times every row with the profiler **off** and profiles a second run, and it
measures occupancy from CPU time rather than from profile samples. Runtime is the unprofiled figure.

| block | blocks | runtime | peak | floor | vs untiled | occupancy | read amp | alloc | measured |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| untiled | 1 | 6.2 min | **85.0 GiB** | 39.6 | 1.00x | 6.06 / 10 | 1.00x | 225 GiB | 1,781,775 |
| 8192 px | 98 | 11.2 min | 47.8 GiB | 29.9 | 0.56x | **2.96 / 10** | 0.86x | 784 GiB | 1,781,377 |
| 6144 px | 162 | 6.2 min | 45.7 GiB | 29.1 | 0.54x | 5.50 / 10 | 1.00x | 799 GiB | 1,781,377 |
| 4096 px | 378 | 7.1 min | 40.1 GiB | 28.9 | 0.47x | 5.17 / 10 | 1.33x | 897 GiB | 1,781,377 |
| 3072 px | 648 | 5.8 min | 31.2 GiB | 22.5 | 0.37x | 6.77 / 10 | 1.69x | 1063 GiB | 1,781,377 |
| 2304 px | 1152 | 5.5 min | 30.0 GiB | 22.1 | 0.35x | 7.87 / 10 | 2.26x | — | 1,781,377 |
| 2304x1152 px | 2304 | **5.2 min** | **28.8 GiB** | 21.3 | **0.34x** | **9.03 / 10** | 3.30x | 1727 GiB | 1,781,377 |

The 3072 px, 2304 px and 2304x1152 px rows are each from their own process; the first four share one.
Every blocked row measures the same 1,781,377 points, so the sizes differ in cost alone.

**2304x1152 px dominates every axis: 0.34x the untiled peak, 0.83x its runtime, 90% occupancy.** The
previously recorded choice of 8192 px is the worst blocked row measured. Note the ordering is monotonic in
block *count* over all six blocked rows, on peak and runtime alike.

**Both axes need sizing separately, and sweeping squares alone misses the best shape.** The halo is
2216x1103 px — almost exactly 2:1 — so a square block clears X and over-provisions Y twofold. The square
floor is 2304 (2048 is rejected on the X halo) while Y's floor is half that, and following the halo's
aspect ratio is what produced the best row. `--blocks 2304x1152` reaches these; `--blocks N` still means
square.

**Runtime is set by thread occupancy, and occupancy by blocks per thread — hundreds, not tens.**

| block | blocks/thread | occupancy of 10 |
|---|---:|---:|
| 8192 px | 9.8 | 2.96 |
| 6144 px | 16.2 | 5.50 |
| 4096 px | 37.8 | 5.17 |
| 3072 px | 64.8 | 6.77 |
| 2304 px | 115.2 | 7.87 |
| 2304x1152 px | 230.4 | **9.03** |

Per-block cost varies by orders of magnitude — a block whose points a finer level resolved returns before
any I/O — so a pool with few blocks per thread spends the run waiting on a few expensive ones. **There is no
turning point in the measured range**: occupancy was still improving at 230 blocks/thread, so an earlier
version of this section recommending "near 10x the thread count" understated it by an order of magnitude.
Note the untiled row reaches 6.06 through the intra-pass path, so it is a different decomposition rather
than a one-block version of the others.

**Read amplification does not drive the ranking.** It rises 0.86x → 1.69x across the rows while runtime
*falls*. Allocation rises with it (225 → 1063 GiB, since `_read_block!` allocates a block-sized temporary
per read) and GC still absorbs ≤1.0% of wall clock at 699 pauses. Redundant reading is cheap next to idle
threads.

**Where the time goes, over the whole run rather than at the peak.** Shares of *working* samples —
running samples with a stack, excluding the parked ones, since folding those in would mix the occupancy
result into every stage's share:

| stage | 3072 px | 2304x1152 px |
|---|---:|---:|
| FFTW, all stacks (`(FFTW transform)` + `fft_execute!` / `ifft_execute!` under `_numerators_fft!`) | **53.1%** | **45.7%** |
| `_read_block!` under `_prepared_block_pair` | 11.9% | 19.0% |
| `preprocess` under `_prepare_block` | 7.4% | 11.9% |
| `peak_index` under `subpixel_peak` | 3.2% | 2.6% |

Both columns are measured against every stack. The four rows above them in the sweep were read from the
top-14 stacks each run printed, which is complete enough for FFTW (its stacks are all large) and not for
the smaller entries: that treatment gives `_read_block!` 4.6% at 4096 px where a full accounting gives
11.9% at 3072. On the FFTW total the truncated reads are usable — 62.5% untiled, 56.9% at 8192 px, 56.4%
at 6144, 54.4% at 4096.

**The correlator's spectral core is the run**, and the blocked path's own overhead is what grows as blocks
shrink: reading plus preprocessing goes from 19.3% at 3072 px to **30.9%** at 2304x1152, while FFTW falls
53.1% → 45.7%. That is the price paid for occupancy, and at these sizes it is still worth paying — the
2304x1152 row is faster in wall clock despite spending a third of its working time on block handling.
It also locates the ceiling: with ~31% of the run in reading and preprocessing, shrinking blocks further
has less and less headroom, and a real speedup means fewer FFTs per point rather than a better layout.

The `(FFTW transform)` bucket is samples whose stack unwound no further than the codelet, so it is
FFTW's own frames rather than a separate stage; it is listed apart only because those samples cannot be
attributed to a call site.

### Three instrument faults this found, two of them in the recorded figures

**The profile buffer was undersized by 3x, and the failure is silent.** `mem_nisar.jl` requests
`n = 60_000_000` words at `delay = 0.002`. A block costs `stack depth + 6` words per *running thread*, so
ten threads over a 716 s run need ~165 M. Julia warns on `fetch` and stops recording at roughly a third of
the run — which leaves a peak-window query correct, because the peak is early, and every whole-run query
silently describing the first third. `plan_profile` sizes the buffer from a measured runtime and widens
`delay` rather than truncating; the fill fraction is reported next to every attribution (9–13% here).

**Profile-sample occupancy is not occupancy.** The obvious ratio — running samples over
`ticks x nthreads` — reads **5.26** on a load that CPU time and construction both put at 1.0 threads, and
8.55 on a genuinely saturating ten-thread load. The profiler samples parked threads and flags them only
coarsely. `cpu_seconds / wall_seconds` from `proc_pid_rusage` measures 1.00 / 1.99 / 3.99 / 9.77 on 1, 2,
4 and 10 spin loops, so that is the figure quoted above. Its fields are **mach ticks**: read as
nanoseconds they give 0.02 threads for a one-thread load.

**The buffer rule was written as a 9x overcount.** `docs/src/explanation/memory.md` and this file both said
`9 x 18 bytes x (block + 2*halo)^2 x nthreads` while their prose said "18 bytes per pixel" — the nine
arrays *total* 18 B/px for a `UInt8` pair (two `UInt8`, three `Float32`, four `Bool`), measured at exactly
18.0 off the struct's fields. The derived figures in those sections (4.56 GiB per set, 45.6 across ten
tasks) were computed correctly at 18 B/px, so only the formula was wrong — but applied as written it
predicts 410 GiB for that L1 window and rejects every size this granule runs at.

**The untiled 12.1 min in the table above was compilation.** The old harness had no warmup and measured
untiled first, so that row carries the process's JIT; the blocked rows measured after it did not, which is
what made blocking look free. Profiler overhead is not the explanation and this is worth recording because
it was the first guess: measured over all five configurations, a profiled run costs **2–4%** (1.02x, 1.03x,
1.03x, 1.04x).

### The sweep's own casualty was the deadlock, not a kill

The 3072 px row stopped mid-profile in the five-configuration sweep with no exception, empty stderr and no
crash report, which was read as a kernel memory kill. **It was the runtime deadlock**: the process was
still alive at 0% CPU and 11.8 GiB when found later, and its trace carries the same
`jl_mach_gc_end`/`jl_profile_thread_mach` pair as the other two. A hung Julia process and a killed one are
indistinguishable from their output alone, so check `ps` for a 0%-CPU survivor before concluding a run died.

The row was re-measured as the only configuration in a fresh process, at 347.6 s against 340.9 in the
sweep, so the figures are sound. Nothing about the shared-process design is implicated — the cause is the
profiler, and the sweep merely profiled for long enough to hit a race.

## Step: the L1 sweep, re-measured on an idle machine

Six configurations on NISAR L1 RSLC, whole grid, `-t 10,1`, one process each, `--no-profile`. Command:

```bash
for bs in 0 16384 8192 4096 8192x4096 2816x1536; do
  julia --project=tools/golden -t 10,1 tools/golden/profile_nisar.jl \
      NISAR_L1_PR_RSLC --blocks $bs --no-profile
done
```

Scene 57760x50511 px, grid 2328x2304, halo **2736x1500 px**, 1,871,119 searchable points. The square block
floor is 2736; per-axis it is 2736x1500, so `2816x1536` is essentially the smallest legal block.

| block | blocks | blocks/thread | runtime | vs untiled | peak | vs untiled | occupancy | read amp | measured |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| untiled | 1 | — | **567.2 s** | 1.00x | 49.36 GiB | 1.00x | **9.21 / 10** | 1.00x | 1,798,199 |
| 16384 px | 100 | 10 | 1852.3 s | 3.27x | 59.58 GiB | 1.21x | **2.33 / 10** | 2.89x | 1,797,136 |
| 8192 px | 380 | 38 | 1078.4 s | 1.90x | 33.85 GiB | 0.69x | 4.61 / 10 | 4.51x | 1,797,076 |
| 8192x4096 px | 760 | 76 | 828.0 s | 1.46x | 30.41 GiB | 0.62x | 6.60 / 10 | 6.42x | 1,797,076 |
| **4096 px** | 1482 | 148 | **661.5 s** | **1.17x** | 27.72 GiB | 0.56x | **8.94 / 10** | 8.77x | 1,797,076 |
| 2816x1536 px | 5814 | 581 | 762.5 s | 1.34x | **24.43 GiB** | **0.49x** | 9.03 / 10 | 21.60x | 1,797,026 |

**L1 agrees with L2 after all.** An earlier reading of this case — that blocking is a pure loss here —
came from invalid timings (below). Blocking halves the peak: 4096 px runs at 0.56x untiled for 1.17x the
runtime, and 2816x1536 reaches 0.49x. What differs from L2 is only that L1's untiled row is *fast*, at
9.21/10 occupancy, so no blocked row beats it on wall clock.

**The two resources disagree, which they did not on L2.** Peak falls monotonically with block size all the
way to the floor, but runtime bottoms out at 4096 px and rises again by 2816x1536 — read amplification
reaches **21.6x** there, and paying 21x the I/O eventually outruns the occupancy gain. So on L1 the answer
depends on which resource binds: 4096 px for speed, 2816x1536 for memory.

**16384 px is the one configuration that is bad on both axes**, at 1.21x the peak and 3.27x the runtime.
Occupancy explains it: **2.33 of 10 threads**. With 100 blocks over 10 threads and L1's radius field
putting an estimated 47% of all correlation work in a single block — median radius 34 against a maximum of
1905 — the pool cannot balance. The same arithmetic gives 25% in one block at 8192 px, and by 4096 px the
imbalance is diluted enough that occupancy reaches 8.94.

### Every earlier L1 timing was invalid, and the harness was not at fault

Three attempts produced three different untiled figures before this one. None of the spread was the
configuration, the machine, or FFTW wisdom; all of it was **me observing the run**:

| untiled measurement | concurrent activity | result |
|---|---|---|
| first sweep | two scripts each `read_capture`-ing the same 12 GB NISAR capture | 649.8 s |
| "quiet" sweep | my own `sample` calls, twice, on the live process | **2592.1 s** |
| reproducibility test, run 1 | nothing | 587.1 s |
| reproducibility test, run 2 | nothing | 575.5 s |
| this sweep | nothing | 567.2 s |

`sample` suspends every thread to unwind it. On a 34 GiB ten-thread process, calling it twice inflated the
row **4.5x** — and the process looked healthy at every check, because it was: it was being stopped and
restarted thousands of times by the observer. The three clean measurements agree to 3.5%.

**Wisdom was ruled out explicitly**, since it was the leading hypothesis. Two whole-grid runs inside one
process — where nothing but FFTW planning state can differ — measured 587.1 s and 575.5 s with the wisdom
file byte-identical (324,653 bytes) before and after both. `benchmark/results/nisar/l1_reproducibility.log`.

The rule this establishes: **poll a running measurement with `ps` or `pgrep` and nothing heavier.** A
`sample` is a measurement of its own and cannot be taken during one.

## Step: whether fewer FFT transform sizes would help — measured, and it does not

The premise checked first, because it was wrong in a way that matters. Distinct *raw* `(radius_x, radius_y)`
pairs on the NISAR L1 grid number **29,761**, and an earlier note in this session quoted that as the number
of FFT plans a pass builds. It is not: `AutoRIFT._radius_bucket` rounds every radius up to a power of two
and caps it at the pass radius, so the ladder actually reached is **37 sizes** on L1 and 44 on L2 — nine
rungs per axis (8, 16, 32 … 1024, cap), each reused by tens of thousands of points.

So the ladder is already quantized far more aggressively than "intervals of 4", which would admit 476 x 207
size pairs on L1 against 9 x 8. Three ladders measured on a 400x400 window of the L1 grid
(`tools/golden/fft_ladder_test.jl`):

| ladder | sizes reached | runtime | vs shipping | measured points |
|---|---:|---:|---:|---:|
| powers of two — **ships** | 25 | **363.3 s** | **1.00x** | 128,675 |
| every second power of two | 10 | 452.1 s | 1.24x | 127,008 |
| multiples of 4 | 25† | 462.0 s | 1.27x | 130,564 |

**Both alternatives lose, for opposite reasons.** Halving the ladder to every second power of two still
costs **24%**: a coarser rung makes a point execute a *larger* transform than it needs, and that waste
exceeds the planning it saves. Going finer, to multiples of 4, costs **27%** while admitting far more plans.
The shipping ladder is at the minimum of a real tradeoff rather than an arbitrary choice.

The reason there is nothing to win: planning is already amortized to nothing. The wisdom file turns three
cold plans from 822 ms into 0.1 ms, and it is per-size-per-machine, so a production worker pays it once
ever. What remains is execution, which a coarser ladder makes worse.

† The harness rewrites the radii and hands them to the *unmodified* correlator, whose own `_radius_bucket`
re-rounds to powers of two — so this row measures the cost of feeding the correlator finer radii, not the
plan count a real interval-4 implementation would carry. The cost is the half that decides the question;
the plan count only moves against it.

## Step: attribution at the chosen L1 size, and the L2 floor

Two measurements that close the NISAR block-size work.

### NISAR L1 at 4096 px, profiled

The chosen operating point, with the profiled arm this time (`--blocks 4096`, no `--no-profile`).

| | value |
|---|---|
| runtime, timed arm | **676.3 s** (sweep gave 661.5 s — 2% apart) |
| runtime, profiled arm | 695.5 s — **profiling costs 2.8%** |
| peak | 32.47 GiB |
| occupancy | 8.79 / 10 |
| measured | 1,797,076 |
| profile buffer | 15% full — not truncated |

Shares of working samples, and the two rows from the sweep for comparison:

| stage | untiled | 16384 px | **4096 px** |
|---|---:|---:|---:|
| FFTW, all stacks | 69.9% | 64.4% | **56.8%** |
| `_read_block!` | 0.0% | 4.6% | 10.2% |
| `preprocess` under `_prepare_block` | 0.0% | 2.8% | 6.2% |
| blocked-path total | 0.0% | 7.4% | **16.4%** |
| `peak_index` + `pyrup!` | 4.7% | 4.6% | 3.8% |
| garbage collection | 0.0% | 0.1% | 0.5% |

**The correlator's spectral core is the run at every block size, and the blocked path's overhead is the
price of occupancy.** Reading plus preprocessing grows 0% → 7.4% → 16.4% as blocks shrink, exactly the
trend L2 shows (19.3% at 3072 px, 30.9% at 2304×1152). At 4096 px on L1 it is 16.4% for a 3.27×
improvement in runtime over 16384 px — a good trade. It also bounds what layout tuning can still win: the
remaining 57% is FFT execution, which no block size changes.

**Profiling costs 2.8% here, so the L2 finding generalizes.** That figure is worth having because an
earlier attempt to explain a runtime discrepancy blamed profiler overhead; measured, it is small on both
granules.

### NISAR L2 at 2224×1110 px — the floor

The smallest block a 2216×1103 px halo permits, which closes the L2 sweep.

| block | blocks | b/thread | runtime | peak | vs untiled | occupancy | read amp |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2304×1152 px | 2304 | 230 | **309.4 s** | 28.79 GiB | 0.34x | **9.03 / 10** | 3.30x |
| **2224×1110 px** | 2500 | 250 | 365.4 s | **25.91 GiB** | **0.30x** | 7.77 / 10 | 3.50x |

**This settles the open question about whether occupancy keeps improving as blocks shrink. It does not.**
An earlier reading of the L2 sweep said occupancy was still climbing at 230 blocks/thread with no measured
turning point. At the floor — 250 blocks/thread, only 8% more blocks — occupancy *falls* to 7.77 and runtime
rises **18%**, while peak keeps improving to 25.9 GiB.

So both granules have the same shape, and it is the practical rule: **peak falls monotonically to the
floor, runtime does not.** The floor is the memory answer (L2 0.30x, L1 0.49x) and a size somewhat above it
is the speed answer (L2 2304×1152, L1 4096 px). Which to use depends on which resource binds.

### One thing both runs surfaced

Each emitted two warnings that a block's coarse grid is smaller than the outlier filter's window, at the
two coarsest chip levels (192×104 and 384×208 on L1). That is the per-block form of the reference's own
behaviour recorded in `tools/golden/README.md`, and it is block-size dependent — a smaller block reaches it
at more levels. It costs the restricted pass's saving on those levels for the blocks affected, which is
already inside the measured runtimes above rather than additional to them.

## Step: both NISAR cases re-measured whole-grid on 1.13.0, and a NISAR gate

The whole-grid endpoint on both cases, `-t 10`, run sequentially because `correlator.jl` is untiled and
the two peaks do not fit 96 GiB together. Command:

```
julia --project=tools/golden -t 10 tools/golden/correlator.jl NISAR_L1_PR_RSLC --run 100
julia --project=tools/golden -t 10 tools/golden/correlator.jl NISAR_L2_PR_GSLC --run 100
```

**The capture is run 100 on both cases.** L1 also has a `200/` directory, but it is empty — a run named
without a capture in it fails in `read_capture` rather than falling back, so the number has to be right.

| | L1 RSLC | ledger | L2 GSLC | ledger |
|---|---:|---:|---:|---:|
| both-measured | 1,786,566 | 1,786,566 | 1,751,658 | 1,751,658 |
| `dx` exact | 73.91% (1,320,364) | 73.90% (1,320,352) | 72.20% (1,264,618) | 72.20% (1,264,646) |
| `dy` exact | 73.86% (1,319,609) | 73.86% (1,319,603) | 75.34% (1,319,689) | 75.34% (1,319,639) |
| `dx` correlation | +0.99972 | +0.99972 | +0.99890 | +0.99890 |
| `dy` correlation | +0.99884 | +0.99884 | +0.99330 | +0.99330 |
| `dx` bias core | +0.0513 | +0.051 | −0.1128 | −0.113 |
| `dy` bias core | −0.0276 | −0.028 | −0.1259 | −0.126 |
| tail >10 px | 27 `dx`, 0 `dy` | 78, 7 | 31 `dx`, 0 `dy` | 31, 0 |
| only jl / only ref | 11,633 / 14,897 | 11,633 / 14,897 | 30,117 / 11,493 | 30,117 / 11,493 |
| wall clock | **9m36s** | 6h30m on 8 threads | **4m52s** | 11h41m on 1 thread |
| peak RSS | 49.0 GiB | — | 60.7 GiB | — |

**Every agreement statistic reproduces**; coverage and correlation are identical, and the exact counts move
by 12 and 28 points in 1.75 M. The two open findings recorded when these were first measured stand
unchanged: L2's core biases are the larger pair on both axes, and its `dy` correlation is below `dx` where
L1's are matched.

**The runtimes are 41x and 144x faster, which is the FFTW planner fix and not a NISAR-specific effect.**
The recorded L2 figure additionally predates `kwargs_from_capture` setting `threaded`, so it spent 11h41m
on one core; both rows here run at ~9.7 of 10 cores. A runtime from before either fix is not comparable to
one after.

**L1's `dx` tail falls 78 → 27 points beyond 10 px** with `dy` unchanged at 0. Both are under the
sub-0.002% the tail represents either way, and `PLAN_FLAGS` moving `PATIENT` → `MEASURE` changes which
candidate transform the planner picks, so a handful of near-flat peaks resolving differently is the
expected shape of that change rather than an unexplained one.

### The gate: `3.nisar`, on a thinned grid

Nine and five minutes is still far outside the seconds-to-minutes the other gates cost, so the gate runs a
sixteenth of each grid — `--stride 4 --block 128`, 128-px tiles on a 512-px lattice, ~1 minute per case.
Both cases are **green**, at `core 0.0975/0.1411 corr +0.99853/+0.98765` on L1 and
`core 0.1758/0.2343 corr +0.99962/+0.99637` on L2.

**Thinning changes the answers, so its thresholds are calibrated to the thinned run and are not comparable
to the whole-grid figures above or to the 0.010 px bound `3.rdr` holds Sentinel-1 to.** AutoRIFT.jl sees
the sparse grid while the reference's `Dx`/`Dy` come from a capture over the full one, so the two resolve
different pyramid levels — a level's coarse grid is the point grid decimated by 1, 2, 4, 8, and a thinned
one can fall below its filter's width, at which point the level silently produces nothing
(`tools/golden/README.md`). Measured on L1 at `stride 4`:

| tiling | `dx` exact | `dx` corr | `dx` bias core | `dy` bias core |
|---|---:|---:|---:|---:|
| whole grid | **73.91%** | +0.99972 | +0.0513 | −0.0276 |
| 128-px tiles | 16.81% | +0.99853 | +0.0975 | −0.1411 |
| 256-px tiles | 9.41% | +0.99408 | +0.1017 | −0.1382 |
| 512-px tiles | **0.00%** | +0.92317 | +0.1013 | −0.1506 |
| every 16th point | **0.00%** | +0.90625 | +0.1085 | −0.1656 |

Three things this table decides:

- **`exact` is not gateable on a thinned L1** — it spans 73.91% to 0.00% on unchanged code, and
  non-monotonically in tile size. The gate asserts correlation, `bias_core` and the `dy` sign only.
- **Tiles, not a point lattice.** `filtDisp` and the level merge consult each point's neighbors, so
  thinning to every 16th point leaves each survivor without any and its base-level measurement is replaced
  by an interpolated one.
- **The tile size is part of the calibration, so `regate.jl` states `--block` explicitly.** A threshold
  measured at one tiling and re-run at another reports a regression that is only a changed default.

L2 is the case where thinning is benign — `exact` 68.70% against 72.20% whole-grid — because its levels
still clear the filter at this tiling. That the same stride is destructive on one case and not the other is
why the gate is calibrated per case rather than to one shared bound.

`bias_core` is roughly double the whole-grid value at every tiling on both axes, consistently enough to
gate against, and it is a property of the thinning rather than of the correlator: the whole-grid run above
reproduces +0.0513/−0.0276 exactly.

## Step: the benchmark suite on 1.13.0, and one candidate rejected on Amdahl

`Pkg.test()` is green at `5c68d73` — **704,304/704,304 in 3m32s**. The suite was run with
`benchmark/run.jl --quick -t 10` and compared against the committed baseline:

```
julia --project=benchmark -t 10 benchmark/run.jl --quick --tag head1130
julia --project=benchmark benchmark/compare.jl benchmark/results/baseline.json \
    benchmark/results/history/head1130.json
```

`compare.jl` exits 1 on one row, `correlate/peak r50` at 1.11x against a 1.10x threshold. **It is the
toolchain, not this branch.** `baseline.json` was recorded on Julia 1.12.5 and this ran on 1.13.0, so the
two differ by more than the code. Measured on 1.13.0 from both checkouts, same script and machine:

| `peak_index` on a 101x101 surface | min of 5 |
|---|---:|
| `main` (worktree at `f18660a`) | 5.677 us |
| this branch | **5.667 us** |

The branch is marginally the faster of the two, and no commit on it changes `peak_index` itself — the
function is identical to main's, and only its call site in `track.jl` moves. `peak_index` is stable to
0.0% over five repeats in-process, so the 1.11x is not sampling noise either — it is the 1.12.5
baseline. **Re-record `baseline.json` on 1.13.0 before reading that row as a
regression.** No allocation gate fired; `ZEROALLOC_PATTERNS` covers the per-point path, which is the
property that matters across millions of pairs.

### Rejected: integer bit ops in `_radius_bucket`

`_radius_bucket` rounds with `ceil(Int, log2(r))`, one float transcendental per call, where
`leading_zeros` gives the same answer in integer arithmetic. Verified identical for every
`r ∈ -3:3000` against ten caps including 1905 and the powers of two around it, and **10.2x faster in
isolation** — 6.37 ns against 0.63 ns per call over a million radii shaped like a NISAR L1 field.

It is not worth taking. The function is called once per point in `_chunk_buckets` and once per
`(point, bucket)` pair in the `_track_bucket!` rescan, so a 3600-point pass over five buckets makes
43,200 calls — **1.09% of that pass's 25.2 ms, for a saving of 0.98%**. That is below the threshold
where a change to a documented hot function pays for itself, and the rescan it is called from is
deliberate (`_track_bucket!` notes that partitioning would allocate storage proportional to the chunk).
The measurement is recorded so the next reader does not have to take it again.

## Step: the CI benchmark gate's 19 rows, attributed

The Benchmark job compares the merge base and the pull request in one job on one runner, which is the
only comparison free of the toolchain confound. It reports 19 rows past its 1.10x threshold. Two
harness facts first, because the job had not reached its own comparison before this branch:

**The job used to die before comparing anything.** `Pkg.develop(path=".")` writes the absolute
checkout path into `benchmark/Project.toml`, a tracked file, so the second measurement's
`git checkout --detach` aborted. Each checkout is now forced and the tree reset after each run.

**`--quick` skips the memory group**, and a full local suite cannot be recorded in a sandboxed shell:
four attempts died at ~20 minutes with `exit=137` while peak RSS across all Julia processes stayed at
**3.9 GiB of 96**, sampled every 20 s with `ps`. Neither the `gpu` group nor the memory group is
responsible — each completes alone. It is an elapsed-time limit on the shell, so a full-suite recording
belongs to CI or to an interactive terminal.

### The planner accounts for the size-dependent rows

`PLAN_FLAGS` moved `FFTW_PATIENT` → `FFTW_MEASURE`. Every benchmark in `suite/correlate.jl` builds its
workspace once and times `correlate!` against a warm plan, so the suite sees `PATIENT`'s *execution*
gain and never its *planning* cost. Measured by flipping that one const on this tree and re-running the
flagged rows, execution only:

| case | PATIENT | MEASURE | ratio | CI ratio |
|---|---:|---:|---:|---:|
| c32 r6 | 9.31 us | 9.35 us | **1.00x** | 1.10x |
| c32 r25 | 32.71 us | 37.00 us | **1.13x** | 1.15x |
| c64 r25 | 57.92 us | 60.33 us | 1.04x | 1.13x |
| c64 r50 | 131.88 us | 132.50 us | **1.00x** | 1.04x |
| c128 r25 | 142.00 us | 146.54 us | 1.03x | 1.04x |

`c32 r25` reproduces the CI ratio almost exactly, and the two rows CI did *not* flag measure 1.00x and
1.03x here — so the flag is the mechanism and its cost is real, concentrated at mid sizes. It does not
account for all of the gap: `c32 r6` is 1.00x here against CI's 1.10x, and `c64 r25` 1.04x against
1.13x, so a shared runner contributes the remainder.

**The trade is the one `src/plans.jl` records, and the suite is structurally unable to see its
benefit.** Planning one 2304x4608 transform costs 306,787 ms under `PATIENT` against 3,873 ms, the
windowed NISAR endpoint went 1576.3 s → 104.6 s, and the whole-grid cases 6h30m → 9m36s and 11h41m →
4m52s. A 10-15% execution cost at 13-80 us sizes buys 15-144x on a production scene. No benchmark in
the suite pays a cold plan, so no benchmark can show the second number.

### Two rows that are not the planner

**`points/pointset scattered n=100000` at 1.34x is runner noise.** `src/points.jl` has no diff on this
branch. Measured here: **80.88 us**, against CI's *baseline* of 84.2 us and candidate of 113.2 us — so
this tree is faster than the figure CI calls the baseline, on identical code. Local spread over five
repeats is 2.1%.

**`+13 allocs` on `track/*` and `multichip/*` is per chunk, not per point.** `_chunk_buckets` returns a
`Vector{Extent}` holding one entry per distinct bucket, a few dozen at most on a real scene. The count
is constant in the point count, which is why the zero-allocation gate — `correlate/*` and `points/*`,
the per-point path — is unaffected and still passes.

### The policy: execution cost at microbenchmark sizes is accepted

A 10-15% execution regression at the 13-80 us transform sizes is **accepted** in exchange for the
planning cost `FFTW_MEASURE` removes. The rows are real and reproduce; the trade is deliberate.

What this does and does not license:

- It covers the `correlate/{surface,point}` and `correlate/subpixel` rows attributed to `PLAN_FLAGS`
  above, and nothing else. A regression on a *whole-pass* benchmark — `track/*`, `multichip/*`,
  `endtoend/*`, `throughput/*` — is not covered by it: those pay planning as well as execution, so a
  regression there means the trade stopped paying and is a regression to fix.
- It does not license a further execution regression. The figures above are the accepted level, so a
  later change that takes `c32 r25` past 37.00 us is a new question rather than this one already
  answered.
- **The zero-allocation gate is untouched by it.** `ZEROALLOC_PATTERNS` covers the per-point path and
  an allocation appearing there is still a failure, whatever it buys.

The Benchmark job therefore stays red on this branch by decision rather than by oversight. It compares
against the merge base, so those rows clear on their own once this lands and the base carries
`MEASURE` too — the gate is measuring a one-time step change, not an ongoing defect.

## Step: where the coarse-level residual comes from, and one candidate fix rejected

The striped residual in the upper-right of both NISAR comparison figures — the region where ionospheric
defocus broadens the correlation peak — is **entirely a coarse-level effect, and the levels' own
measurements are not what disagrees.** Measured on L1 RSLC over the top-right third of the grid (rows
1–776, cols 1537–2304, 273,447 both-measured points), which is where the figure shows it.

### The base level is exact where the reference measured

Split by the reference's `out_InterpMask`, which marks a value it filled rather than measured:

| level | `InterpMask = 0` (measured) | | `InterpMask = 1` (filled) | |
|---|---:|---:|---:|---:|
| | n | `dx` exact | n | `dx` exact |
| chip 96 | 150,866 | **99.992%** | 23,712 | 85.5% |
| chip 192 | 1,018 | 0.00% | 51 | 0.00% |
| chip 384 | 20,863 | 0.00% | 1,791 | 0.00% |
| chip 768 | 70,926 | 0.00% | 1,805 | 0.00% |

**Effectively every point the reference measured at the base chip size agrees bit-for-bit**, inside the
high-ionosphere region included: **12 points of 150,866 differ in the window, every one of them by exactly
one 1/32 quantization step**, which is tie-breaking on a flat peak and not a systematic error. Over the
whole grid it is 904 of 1,234,293 (0.073%), same magnitude. (An earlier revision of this section rounded
that to 100.00%; the figure panel reports 99.99%, and the exact count is above.)

So the peak broadening the ionosphere causes is not what the whole-scene figure shows: it degrades both
implementations identically, and they still agree to one quantization step.

Above the base level neither side is on its own quantization lattice — 0.00–0.03% of values sit on a
`1/oversample` multiple against 95–96% at chip 96 — because both replace the measurement with a resize.
`exact` is therefore meaningless there, which `stages.jl` already records.

### It is not decimation of the imagery, and not the quantization

**A `Float32` pair would show the same disagreement.** Both NISAR captures are `UInt8` and
`CAPTURE_FLOAT32` needs a container run, but the hypothesis makes a prediction about the bytes already
on disk: if the 256-level collapse were responsible, the residual would concentrate where quantization
costs something. It does not. Binning by how many distinct byte values each chip actually spans:

| level | fewest levels (worst quantized) | middle | most levels (best quantized) |
|---|---:|---:|---:|
| chip 96, median \|ddx\| | 0.0000 (n=1,553) | 0.0000 | 0.0000 |
| chip 768, median \|ddx\| | 0.1516 (n=1,619) | 0.1355 | 0.1471 |

The residual is flat in chip contrast — 0.1355 to 0.1516 px across the terciles, non-monotonically —
while the chips themselves span 209–224 of 256 levels, so quantization has left almost nothing to lose.
The discriminator that settles it is chip 96: the same imagery, the same bytes, the same points, and
100% agreement. A quantization effect cannot be absent at one chip size and present at the next.

### It is the read-back of a decimated level, and it is ours

The capture holds each level's raw measurement on the level's own lattice (`lvl{N}_dx`). Comparing each
side's *merged* fine-grid value against the *reference's own* raw level value at the node standing over
it separates the measurement from the merge:

| level | `ref_merged − ref_raw` rms | `our_merged − ref_raw` rms | ratio |
|---|---:|---:|---:|
| chip 96 (stride 1) | 2.9909 | 2.9909 | **1.000** |
| chip 192 (stride 2) | 1.5904 | 1.6023 | 1.007 |
| chip 384 (stride 4) | 0.0984 | 0.2646 | **2.688** |
| chip 768 (stride 8) | 0.0490 | 0.2542 | **5.190** |

The reference's own read-back returns its raw measurement almost unchanged — 0.049 px rms at chip 768 —
where ours lands 0.254 px away, and the ratio grows with the stride. `figs/nisar_l1_merge_vs_rawlevel_chip768.png`
maps all three: the reference-against-itself panel is blank on the ±0.3 px scale while ours carries the
full striped structure, which is the same structure the whole-scene difference panel shows.

**Where the placements part company.** Reconstructing the reference's level lattice from `in_xGrid` as
`INTER_AREA` to the level shape then `round(x + 0.5) − 0.5` (`autoRIFT.py:109-125`) reproduces every
captured `lvl{N}_xgrid` up to exactly one constant per level — one unique offset, all three coarse
levels, both axes — so its lattice is fully determined. Against it, our `_cell_centres` node sits:

| level | within 1 px of the reference's node, fill-free cells | worst |
|---|---:|---:|
| chip 192 | 100.00% (n=563,661) | 0.5 px |
| chip 384 | 75.06% | 19.25 px at p90 |
| chip 768 | 54.56% | 1313 px at p90, 65 px on fill-free cells |

**The half-cell shift is missing its cross term.** A cell's centre lies `(stride − 1)/2` points further
along in *both* axes, so on a grid where `x` varies with row as well as column the displacement to the
centre is `(dx/dcol + dx/drow) * (stride − 1)/2`. `_cell_centres` applies the `dx/dcol` half only. Fitted
over the L1 grid's real points, `dx/dcol = +33` and `dx/drow = +34` — a swath rotated near 45°, where the
two terms are the same size — and likewise `dy/drow = +19` against `dy/dcol = −19`:

| stride | x shift applied | x shift an affine cell centre needs | x term omitted | y term omitted |
|---|---:|---:|---:|---:|
| 2 | 16.50 | 33.50 | 17.00 | −9.50 |
| 4 | 49.50 | 100.50 | 51.00 | −28.50 |
| 8 | 115.50 | 234.50 | 119.00 | −66.50 |

Measured against each fill-free cell's own block mean, which is the position `_cell_centres` is trying to
reach, adding the cross term accounts for nearly all of the gap:

| stride | current placement | with the cross term |
|---|---|---|
| 2 | med −14.75 px, rms 14.7 | med +2.25, **rms 2.5** |
| 4 | med −45.00 px, rms 45.3 | med +6.00, **rms 8.2** |
| 8 | med −109.48 px, rms 112.1 | med +9.52, **rms 20.2** |

**This is not the half-pixel convention, and rotation matters here for a different reason.** The
`round(x + 0.5) − 0.5` snap is a sub-pixel offset in *image* space and is rotation-independent; this is a
whole-cell offset in *grid* space that scales with the grid step. It vanishes wherever `dx/drow` is zero —
every north-up optical grid — which is why no optical golden case shows it and both NISAR cases do.

Genuine cell-scale curvature exists and is a second-order term: an affine fit *within* a fill-free cell
has a worst-point residual of median 1.2 px at stride 4 and 2.9 px at stride 8. (An earlier note here
attributed the whole offset to curvature on the strength of a 73 px second difference along a row slice;
that slice crossed the nodata fill in its interior, and a per-cell fit is the right test.) Roughly 40% of
the nodes disagreeing with the reference are cells straddling the footprint edge; the rest are interior.

### Rejected: adopting the reference's block-mean placement

Placing each coarse node at its cell's fill-excluding mean coordinate brings the lattice to **100.00%
within 1 px of the reference's on fill-free cells at every level**, median offset 0.0000 — and makes the
answer **worse**:

| | before | after |
|---|---:|---:|
| chip 384, `rms(ours vs raw) / rms(ref vs raw)` | 2.688 | **3.329** |
| chip 768, same | 5.190 | **5.497** |
| L1 whole-grid correlate wall clock | 649.8 s | **7166.3 s** |

`_undecimate_level` reads a coarse node back from the geometric cell centre, so moving where the level
correlates without moving where it is read back measures the field in one place and attributes it to
another. The 11× runtime has the same cause — a node whose cell straddles the footprint edge moves far
enough that its search window grows.

(The pre-existing comment in `_cell_centres` predicted this outcome, but for a reason that turns out to be
wrong — it attributed the hazard to the reference's own two halves disagreeing. They agree to 0.016 px;
see below. The prediction was right and its stated mechanism was not.)

### Also rejected: adding the missing cross term

The cross term is the self-consistent version of the same correction — it makes `_cell_centres` compute
the geometric cell centre its own read-back already assumes, rather than moving to the reference's
position — so it does not carry the desynchronization the block mean does. It is also **slightly worse**,
at no runtime cost:

| chip 768, `rms(ours vs raw) / rms(ref vs raw)` | wall clock |
|---|---:|
| as written | 5.190 — 649.8 s |
| + cross term | **5.455** — 682.6 s |
| node at the cell's exact mean | 5.497 — 7166.3 s |

Both corrections move the lattice much closer to the reference's (109 px → 5.5 px median offset in x at
stride 8 for the cross term; to 0.0 for the exact mean) and neither improves the answer. **So the coarse
residual is not primarily a correlation-position error**. Where it *is* follows below — not the read-back,
which the next section rules out.

### The merge and the read-back are faithful; the coarse measurement is what differs

`chipsize_level` is callable on its own, so our level can be run and its undecimated field compared against
our own merged output. Over the 72,731 chip-768 points in the window:

| | med | rms |
|---|---:|---:|
| `our_merged − our_readback` | +0.0000 | **0.0082** |
| `our_readback − ref_raw` | +0.1167 | 0.2545 |
| `our_merged − ref_raw` | +0.1165 | 0.2542 |

**The merge reproduces our own read-back to 0.008 px**, so the merge is not the defect and the whole 0.254 px
is already present in the read-back's input. Comparing the two sides *on the coarse lattice*, before any
interpolation, both sit **100.00% on the 1/128 quantization lattice** — two genuine measurements — and they
agree at only 2.52%.

**So the residual is in the coarse measurement itself, not in the merge, not in the read-back, and not in
the node position.** This supersedes the earlier statement in this section that `_undecimate_level` was where
the remaining work is; that was written before the read-back was measured separately.

Our `resample` with `Bicubic` was checked against OpenCV's `INTER_CUBIC` directly and **matches exactly**,
including the tap layout and the `a = -0.75` weights: a ramp upsampled by 8 gives the identical −0.0881
node offset on both sides, which is cubic convolution's own behaviour at half-integer offsets rather than a
misalignment.

### Not yet localized, and the harness that would do it is not trustworthy yet

Three attempts to replay our coarse pass on the reference's captured per-level inputs landed at rms 1.9–3.6
px against a 0.25 px target, i.e. they were not reproducing the reference's pass at all. The diagnosis of
those attempts:

- **The per-level prior was missing.** The reference's fine pass is handed `Dx00` — the cell mean of `Dx0`
  over `1/Scale` cells, resized to the lattice (`autoRIFT.py:161-179`) — and `install_levels` recorded
  `dx`, `dy`, `xgrid`, `ygrid`, `searchx`, `searchy` and not the prior. **`capture.py` now records it**
  (`dx0`/`dy0` per level); a re-capture is needed to pick it up.
- **A `dy` sign error in the replay**, which double-negated the prior. Once corrected, single-node
  instrumentation shows `dx` agreeing to ~0.05 px per node while `dy` is offset 4.8–6.1 px with a spread,
  which is still a harness fault on the anisotropic `768x416` chip and not a correlator finding.

What the least-broken replay does show, on a 65² window of the lattice, is that the `dx` disagreement is a
**tail, not a bias**: median +0.0078 px, |Δ| p50 0.055, p75 0.109, p90 0.203, then p95 1.95 and p99 13.06.
A small population of nodes lands on a different peak while the bulk agrees to a fraction of a step. That is
consistent with the ionospheric peak broadening the case is known for, and it is **the hypothesis to test
next**, not a conclusion — the replay has to reproduce the reference when primed with the reference's own
answer before any number from it is usable, and it does not yet.

**The self-consistency gate is the thing to fix first.** `step17`-style replays now assert it explicitly:
prime the pass with the reference's own answer and require rms ≤ 0.1 px, because three earlier rounds of
numbers were reported from a harness that would have failed it.

### The reference is self-consistent, so that is not the explanation either

An earlier note here and in `src/multichip.jl` justified our deliberate placement difference on the
grounds that the reference's two halves disagree with each other. **They do not.** Checked against OpenCV
directly:

- `INTER_AREA` at an integer scale is the exact block mean (max difference 0.000000 against a hand
  computed mean), and for an affine grid the block mean is the value at the cell centroid, fine index
  `k*s + (s-1)/2` (max difference 0.000000).
- `INTER_CUBIC` upsampling places source node `k` at destination `(k + 0.5)*s − 0.5`, which is that same
  `k*s + (s-1)/2` — verified exactly at `k = 1, 2, 3, 10` for `s = 8`.

On the real L1 grid, the position where the reference correlates a node and the position its read-back
attributes it to differ by a **median of 0.016 px, rms 4.94 px = 1.3% of a 384 px cell**, all of it local
curvature within the cell. Its only deliberate self-inconsistency is the `round(x + 0.5) − 0.5` snap, at
most 0.5 px.

That matters twice. It removes the stated reason for our placement differing from the reference's — the
`INTER_AREA` construction is *stronger* than a Jacobian shift, being exact at any rotation and any
curvature. And it means a 0.25 px coarse residual cannot be charged to the reference disagreeing with
itself; the remaining candidate is our own read-back.

### Agreement is not correctness here, and the reference may hold the worse field

The 5.2× ratio above is a self-consistency measurement, not an accuracy one. Judged against an independent
local truth — the base level, where the two agree bit-for-bit on the reference's measured points, averaged
over base-level neighbours within 3 grid points — neither side wins cleanly on L1:

| level | axis | AutoRIFT.jl error vs local truth | autoRIFT.py | closer |
|---|---|---|---|---|
| chip 384 | `dx` | med 0.105, rms 0.181 | med **0.060**, rms **0.170** | reference |
| chip 384 | `dy` | med **0.114**, rms **0.237** | med 0.123, rms 0.512 | AutoRIFT.jl |
| chip 768 | `dx` | med 0.093, rms 0.152 | med **0.078**, rms 0.166 | reference (median) |
| chip 768 | `dy` | med 0.160, rms **0.234** | med **0.103**, rms 0.466 | split |

The reference is better on `dx` and its `dy` carries **roughly twice our rms at a comparable median**,
which is a heavy tail rather than a bias. So converging on it in `dy` would mean adopting the less
accurate field. This is recorded as an agreement-vs-correctness item in `tools/golden/README.md` under
"Matched for agreement, not endorsed", to be settled after the coarse levels agree and against real ground
truth rather than against either implementation — the base-level-neighbour proxy used here is weakest
exactly where the coarse levels do their work, which is where the base level declined to measure.

Reverted; `src/multichip.jl` is unchanged apart from a comment recording the measurement so the candidate
is not retried. The L1 whole-grid figures reproduce exactly after the revert — 73.9% / 73.9% `exact`,
11,633 / 14,897 coverage, 0.7% level disagreement, 665.1 s — so nothing regressed.

`figs/nisar_l1_topright_by_level.png` is the by-level map of the window: `ddx` at chip 96 alone is blank,
`ddx` at chip ≥ 192 carries every stripe, and the stripes coincide with the chip-size panel's coarse
bands rather than with the level-disagreement panel, which covers only 0.88% of points.

## Step: the coarse residual localized to the level's own measurement, and a synthetic reproducer

Continuing the section above, which established that the merge reproduces our own read-back to 0.008 px
and that the two sides' coarse *lattice* values agree at only 2.52%. This narrows that further and
supplies a reproducer that needs no capture and runs in seconds.

### The heat map, which the summary statistics were hiding

`figs/nisar_l1_coarse_lattice_chip768.png` maps the chip-768 difference on the **level's own lattice**, so
one cell is one measurement and nothing is spread by the read-back. It is **a smooth, spatially coherent,
one-signed field of +0.1 to +0.3 px over the fast-flow region** — not scattered outliers: 4.0% of nodes
exceed 0.5 px and **none** exceeds 2 px. The earlier report of "rms 2.1-3.6 px, a heavy tail" came from a
replay harness with a `dy` sign error and a missing prior, and was wrong about the data.

Two mechanisms are ruled out by that shape plus one regression:

- **Not a node-position error.** Regressing the lattice residual on the reference field's own gradient
  gives an implied offset of **+0.005 lattice cells** along the column axis and **R² = 0.0004**, against
  the +0.31 cells the missing cross term would produce. A position error shows up as `gradient x offset`;
  this residual is not proportional to the gradient at all.
- **Not search-radius saturation.** Binned by the reference's own radius, the residual *falls* as the
  radius grows — median +0.083 px in the smallest third against +0.008 in the largest.

### The synthetic reproducer: where the two correlators part company

A 3000² random pair, warped by a linear `dx` gradient of 6 px per 1000 px, correlated at one row of
points by both implementations on byte-identical arrays. No capture, no container, seconds per run:

| chip | window | agreement (73 points) |
|---|---|---:|
| 96x52, 192x104, 384x208 | to 431x751 | rms **0.007 px** — one quantization step |
| 512x280, 576x312, 640x348, 512x512 | to 395x687 | rms 0.007-0.010 px |
| **704x384** | 431x751 | rms **0.360**, max 2.97 |
| **768x416** | 463x815 | rms **0.136**, max 1.15 |
| 768x768, 1024x556 | 815x815, 603x1071 | rms 0.72, 0.44 |

**Agreement is exact-to-one-step up to a ~690 px window and breaks past it.** `512x512` agrees while
`768x768` does not, so it is absolute size and not anisotropy. On a *pure translation* both sides are
bit-exact at every chip size including 768 — the gradient is what exposes it.

### What breaks, mechanically

As the chip grows the correlation peak flattens — peak height falls 0.92 → 0.83 → 0.53 → 0.27 across
96 → 768 — and the peak's plateau widens from 1 sample to 3. The 5x5 refinement patch is clamped to the
surface, so a plateau reaching the patch edge puts the upsampled maximum **on the patch border**, which
maps back exactly to a source node and yields an **exact-integer** displacement with no subpixel part.
The detector is perfect on the synthetic set: at chip 768, **9 border maxima and 9 exact-integer `dx`
values, the same 9 points**; zero of each at chip 96, 192 and 384.

**But this behaviour is matched, not broken.** Asked for its integer peak (`SubPixFlag=False`) the
reference returns the *same* integer peak we do — −2.00 at chip 704, +0.00 at chip 768 — and on 8 of those
9 border points its refined answer is **also** the exact integer, bit-identical to ours. Only one point of
73 differs, and it alone carries the whole chip-768 rms: x=1320, ours −2.00000 against its −0.85156,
1.148 px of the 0.136 total. Decomposed:

| population | n | rms | note |
|---|---:|---:|---|
| refined (non-integer) | 64 | **0.024** | max 0.109 px, sub-step |
| border (exact-integer) | 9 | 0.383 | **8 of 9 bit-identical to the reference** |
| the one mismatch | 1 | — | 1.148 px, a bistable plateau |

So on synthetic data the two correlators agree to a quantization step except at rare bistable plateau
points, and the plateau/border behaviour itself is shared.

### Why that does not yet explain the real case

The real chip-768 residual has a **different shape**, and the synthetic model does not predict it. On the
L1 lattice, top-right third, 72,731 merged points:

| bin | n | share of points | share of squared error | median |
|---|---:|---:|---:|---:|
| \|d\| <= 0.125 | 33,826 | 46.5% | 3.0% | +0.037 |
| 0.125 < \|d\| <= 0.5 | 34,128 | 46.9% | **56.4%** | **+0.227** |
| 0.5 < \|d\| <= 1 | 4,775 | 6.6% | 40.5% | +0.602 |
| \|d\| > 1 | **2** | 0.0% | 0.0% | +1.017 |

**Two points of 72,731 exceed 1 px.** There is no bistable-plateau tail. Instead half the population sits
at a one-signed median of +0.227 px — a broad systematic bias, which is what the heat map shows. The
synthetic experiment reproduces the *quantization-step* agreement and the plateau mechanism but **not**
this bias, so the bias is driven by something the synthetic pair does not contain: real speckle, the
Wallis-filtered texture, a prior the synthetic runs set to zero, or the level's hole-fill.

**The next measurement is the per-level prior**, which is the input the synthetic runs set to zero and the
real level receives as `Dx00` (a cell mean). `capture.py` now records it and `tools/golden/level_replay.jl`
replays a level on it behind a self-consistency gate; an L1 re-capture was started for this and had not
finished when this was written. The reproducer above is the cheap path for everything that does not need
it.

## Step: the L1 re-capture, the replay gate, and `dy` localized to a one-cell position offset

The L1 re-capture completed (~75 min, 186 GB) and carries the per-level prior: 16 `lvl*_dx0`/`_dy0` arrays
across the 8 correlator calls, 72 `lvl*` arrays, `call1.json` listing `dx, dx0, dy, dy0, searchx, searchy,
xgrid, ygrid` on every fine pass. `tools/golden/level_replay.jl` now runs on it.

### Two harness faults the maps found, both of which had been reported as correlator findings

**The `dy` negation was on the wrong side.** `arImgDisp_*` returns cartesian-Y — its last act is
`Dy = -Dy` — so the captured `lvl*_dy` is up-positive while AutoRIFT.jl's `dy` is row-positive. All four
sign combinations were scored rather than reasoned about: negating **our output** gives `dy` rms 0.90 px
against 4.82 for the same sign, and the *prior's* sign changes almost nothing (0.8971 against 0.9019),
because a per-level prior is only a few pixels. The earlier "+4.8 px `dy` offset" was this.

**Every `dx` outlier is a footprint-edge chip.** `figs/nisar_l1_replay_chip768.png` maps it: the nodes
disagreeing by more than a pixel form a **one-cell-wide line along the swath's diagonal boundary**, with
none in the interior. A 768x416 chip centred a cell inside the boundary is still part nodata fill, so the
two sides break a partly-empty correlation differently and neither is measuring ground. This is invisible
in a percentile and obvious in a map. Excluding one ring of 8-connected boundary nodes:

| | n | `dx` med | `dx` rms | `dx` p95 | beyond 1 px |
|---|---:|---:|---:|---:|---:|
| all nodes | 1732 | +0.0078 | 2.3055 | 1.949 | 88 (5.1%) |
| interior only | 1476 | **+0.0078** | 0.3538 | **0.211** | **2** |

So **`dx` at the coarse level agrees to one quantization step over the interior** — median +0.0078 px is
exactly 1/128 — and the level's `dx` measurement is not where the merged residual comes from.

### `dy` is a position error of about one lattice cell, and `dx` is not

`figs/nisar_l1_replay_dy_chip768.png` puts the two side by side on the same interior population. They look
nothing alike: `ddx` is salt-and-pepper about zero, while **`ddy` is smooth diagonal bands alternating
±1-2 px, parallel to the swath edge**. A banded, signed, spatially coherent field is a sampling-position
difference, not tie-breaking.

Regressing `ddy` on the reference's own `dy` field derivatives over the 1,476 interior nodes:

| predictor | correlation | slope |
|---|---:|---:|
| `d(dy)/dcol` | **−0.513** | **−0.942** |
| `d(dy)/drow` | **+0.455** | +0.839 |
| `d²(dy)/dcol²` | +0.328 | +0.299 |
| `d²(dy)/drow²` | +0.350 | +0.266 |

Joint fit **R² = 0.560**, rms 0.574 → 0.377 px, with first-derivative coefficients of −0.815 and +0.528
lattice cells. **So `ddy` is `gradient x offset` with an offset of order one lattice cell** — 384 px at
stride 8 — in opposite senses along the two axes.

Ruled out along the way, each by measurement: the prior (`ddy` vs `dy0` slope +0.09, where a wrong prior
sign would give +2.0); search-boundary railing (**0 of 1476** nodes at the bound on either side); and the
`dy` magnitude itself (the error *falls* as `|dy|` grows, rms 0.835 → 0.211 across quartiles of `|ref dy|`).

**This is the first positive identification of a mechanism**, and it is `dy`-only. The earlier
gradient-regression that returned R² = 0.0004 was run on the *merged* `dx` field, which is why it found
nothing: the defect is in `dy`, at the level's own pass, and worth about 0.4-0.6 px there.

### The gate does not pass yet

Interior, chip 768: **rms 0.479 px against a 0.1 threshold**, `dx` p99 0.31 and `dy` p99 2.66. `dx` is
effectively clean, so the gate is now measuring the `dy` offset above rather than a harness fault. The
threshold stays at 0.1 — it is what a replay on identical inputs should reach — and the next step is to
find which `dy` position the level's pass uses that the reference's does not, now that the offset's size
and sign are known.

## Step: the `dy` offset sweep, and why its answer is not yet trustworthy

Sweeping the node's y coordinate and minimizing the `dy` residual against the reference's own level field is
the direct way to locate a position offset — no convention reading required. Run on the chip-768 level of L1
over a 201² window of the 291×288 lattice, it gives a clean trough:

| y shift | `dy` rms | `dy` MAD | `dx` rms |
|---:|---:|---:|---:|
| 0 | 0.738 | 0.227 | 0.138 |
| −192 | 0.418 | 0.102 | 0.124 |
| −336 | **0.174** | 0.055 | 0.122 |
| −384 | 0.241 | **0.055** | 0.124 |

and the MAD minima across levels land on **exactly one lattice cell** at each stride — −96 at stride 2,
−192 at stride 4, −384 at stride 8, against a grid spacing of 48. That is a tidy result and it is **not yet
believable**, for two reasons the heat maps show and the statistics hid.

**The chip-192 replay is broken, not merely noisy.** `figs/nisar_l1_replay_levels.png` maps it: `ddy` is
saturated across the *entire* 201² window at a ±0.3 px scale and still structured at ±3 px, over 30,828
interior nodes, with MAD 5.1 px at every shift tested — flat, so its "minimum at −96" is the floor of a
broken measurement rather than a located offset. Whatever the replay is doing wrong at that level, the same
harness produced the stride-2 and stride-4 rows of the table above, so the "one lattice cell at every
stride" pattern rests on two rows that cannot be trusted and one that can.

**The chip-768 window is mostly empty.** The same figure shows the level's data occupying only the top-right
corner of a window centred on the lattice: the reference measures 4,585 of 83,808 nodes on that lattice, and
a 201² window at its centre catches a few hundred of them. So the trough above is drawn from a small corner
population, which is exactly the hyper-locality a wider window was meant to rule out — widening the *window*
does not help when the *level* only covers a corner.

**What stands, and what does not.** The `dy` residual at chip 768 is banded, spatially coherent, and fits
`gradient x offset` with R² = 0.56 over the interior — that measurement is on the earlier 65² window inside
the data and is unaffected. The *size* of the offset is not established: −336 and −384 px are
indistinguishable on MAD, the levels that would discriminate the scaling law are broken, and a trough this
shallow over a corner population does not pin a number.

The next step is to fix the replay at chips 192 and 384 — where the reference's own `dy` spans ±957 px
against ±6 at chip 768, so the two levels are not the same kind of measurement — and to place the window on
the level's own data rather than on its lattice centre. Both are harness work, and neither justifies a
change to `src/` yet.

## Step: the coarse-level measurements agree; the whole residual is downstream of them

**The replay gate passes.** The fault was in the replay, not the correlator, and the sweep results above are
withdrawn along with it.

### The defect: every level's captured grid is in its own padded frame

`arImgDisp_*` pads both images by `Px = max(ChipSizeX)/2 + max(SearchLimitX + |Dx0|) + 2` and then shifts the
grid it was handed by `Px + 0.5` **in place** (`arImgDisp_u:78-90`), so `install_levels` records the
*post-shift* grid. The pad is a function of that level's own chip and search extent, so it is one constant
per level and not one constant overall. Measured on L1:

| level | stride | x pad | y pad |
|---|---:|---:|---:|
| chip 96 | 1 | +2231.5 | +1149.5 |
| chip 192 | 2 | +2605.5 | +1228.5 |
| chip 384 | 4 | +788.5 | +274.5 |
| chip 768 | 8 | +933.5 | +371.5 |

Every replay before this treated the captured grid as unpadded, adding only the index base. That placed each
node hundreds of pixels from where the reference correlated — and because the imagery is smooth at that
scale, the result was not an obvious failure but a *plausible-looking* disagreement. It is why the earlier
replays sat at rms 1.9-3.6 px, why chip 192 correlated **−0.03** with the reference over 5,661 nodes, and
why the `dy` residual looked banded and gradient-like: a fixed positional error on a smooth field is exactly
`gradient x offset`. `level_replay.jl` now recovers the pad by reconstructing the level grid from
`in_xGrid` (`INTER_AREA` block mean plus the even-chip snap) and requires the difference to be a *single*
value per axis, erroring otherwise.

A second harness fault compounded it: the window was centred on the level's *lattice*, but a level covers
only part of its lattice and that part moves up the pyramid — chip 768 measures 4,585 of 83,808 nodes. At
chips 192 and 384 the window landed on empty grid (`both-measured 0`). It is now centred on the centroid of
the reference's own measured nodes.

### With both fixed, the levels agree bit-for-bit

65² windows on each level's own data, replaying our fine pass on the reference's captured lattice, priors and
search radii, interior nodes only:

| level | n | `dx` exact | `dx` max | `dy` exact | `dy` max | gate |
|---|---:|---:|---:|---:|---:|---:|
| chip 192 | 3,318 | 99.28% | 229.28 | 99.40% | 36.56 | 113.1 |
| chip 384 | 1,309 | **99.92%** | **0.008** | 99.69% | 0.008 | 0.78 |
| chip 768 | 2,817 | **99.96%** | **0.008** | **99.96%** | 0.008 | **0.060** |

`figs/nisar_l1_level_replay.png` maps it: **1 nonzero node of 2,817** at chip 768, **1 of 1,309** at chip
384, and 24 scattered specks of 3,318 at chip 192 — no structure at any level, on a ±0.05 px scale. The
surviving maxima at chip 192 (229 px) are a handful of points on fast-flow ice, and they are what its gate
figure reports; the median and the 99th percentile are both exactly zero.

**Sanity-checked against a shifted grid rather than assumed:** displacing the grid 500 px takes exact
agreement from **97.06% to 6.58%**, so the pass is genuinely correlating and the agreement is not an echo of
its inputs.

### What this settles, and what it moves

**Our coarse-level measurement is not the defect.** Given the reference's own inputs, our fine pass at chips
384 and 768 reproduces its answer to better than 1 part in 1,000, bit-exact. So the merged residual — median
+0.117 px, rms 0.254 at chip 768 — enters *after* the measurement, and the merge is faithful to our read-back
at 0.008 px rms. What remains between them is the read-back's **input**: the level's hole fill
(`_fill_level_holes` and the reduced prior it fills from) and the `wanted`/coarse mask deciding which nodes
the level measures at all. Those are the only steps left between a measurement that matches to 1e-3 px and a
merged field that differs by 0.25.

Every earlier localization in this file that rested on a replay — the `dy` "one lattice cell" offset, the
R² = 0.56 gradient fit, the per-stride minima — is an artifact of the frame error and should not be carried
forward. The heat maps are what exposed it each time the statistics looked plausible.

## Step: the three-way split, and the decimated grid is where the error enters

Asking the question in the order the pyramid builds — (1) are the decimated inputs identical, (2) do the
levels agree *on* the decimated lattice, (3) do they agree after interpolation to the fine grid — localizes
it exactly.

**(2) The measurements agree.** Handed the reference's own captured lattice, priors and search radii, our
fine pass reproduces its answer at **99.96% bit-exact** at chip 768 and 99.92% at chip 384 (above).

**(3) The interpolation is faithful.** Our merged value reproduces our own read-back at **0.008 px rms**
over 72,731 chip-768 points (above).

**(1) The inputs are NOT identical, and that is where the residual enters.** Comparing our derived
`_decimate_level` output against the reference's captured per-level arrays, in each level's own padded frame:

| level | our searchable | reference | ratio | x grid offset | y grid offset |
|---|---:|---:|---:|---:|---:|
| chip 192 | 474,845 | 262,744 | 1.8x | +16.5 px | +9.5 px |
| chip 384 | 119,615 | 22,661 | 5.3x | +49.5 px | +28.5 px |
| chip 768 | 30,337 | 4,585 | 6.6x | **+115.5 px** | **+66.5 px** |

Both differences are real and independent.

### The grid offset is the missing Jacobian cross term, and it is asymmetric between the axes

The reference's node is the cell's block mean, verified exactly — at node (120,150) of the chip-768 lattice
its coordinate is 19092.5 and the block mean of `in_xGrid` over that cell is 19092.06, snapping to 19092.5;
the same holds at every node checked. Ours is the cell's *first* point plus `dx/dcol * (stride-1)/2`, so it
is short by the `dx/drow` term. With `dx/dcol = +33` and `dx/drow = +34` at stride 8 that is
`33 * 3.5 = 115.5` applied where `67 * 3.5 = 234.5` was needed — and **+115.5 px is exactly the measured
median offset over all 83,808 nodes.**

The correction is not the same on both axes, which is what makes this subtle:

| | applied | needed | adding the cross term |
|---|---|---|---|
| x (`dx/dcol +33`, `dx/drow +34`) | +115.5 | +234.5 | **worse**: +115.5 → +234.5 median |
| y (`dy/drow +19`, `dy/dcol −19`) | +66.5 | ~0 | **fixed**: +66.5 → **0.000** median, 74-98% within 1 px |

On the y axis the two terms have *opposite* signs and cancel, so the correct shift is near zero and our
`+66.5` is pure error. On the x axis they have the *same* sign, so the correct shift is larger than ours —
but adding the row term alone overshoots, because the block mean of a snapped, non-linear grid is not the
value at the affine cell centre. **So neither the current formula nor the cross-term formula is right; the
node has to be the cell's mean coordinate**, which is what the reference computes and what
`_undecimate_level`'s read-back position then has to match.

**An earlier measurement in this file reported this lattice as "100.00% within 1 px on fill-free cells" and
that was wrong.** It compared `_decimate_level(grid, trues(...))` — every point wanted — against a
reconstruction, on a population dominated by the nodata fill where every candidate rule agrees. Restricted
to the nodes the reference actually searched, the same comparison gives 0.0% within 1 px.

### The searchable-set difference is separate and larger

We search 6.6x more nodes than the reference at chip 768 (30,337 against 4,585), with only 35 nodes the
reference searches that we do not. Every one of our 25,787 extra nodes is a point where the reference's
`out_ChipSizeX` records that a *finer* level already won. So our `wanted`/coarse-mask combination lets the
coarse level attempt points the reference had already resolved; those measurements then reach
`_undecimate_level` and the merge. That is a second mechanism for the merged residual, independent of the
grid offset, and it is not yet quantified.

**Both are in `_decimate_level`, and neither is in the correlator or the merge.** That is the answer to
where the coarse-level residual is introduced.

## Step: the cell-mean fix, and whether the reference is right

### The fix

`_cell_centres` now places a coarse node at its cell's **mean coordinate**, snapped back onto the grid's own
sub-pixel lattice, replacing the uniform `(stride - 1) / 2` shift by the modal grid step. That is the
reference's `INTER_AREA` block mean plus its `round(x + 0.5) - 0.5`, and it is correct at any rotation and any
curvature where a Jacobian shift is not.

**The lattice now reproduces the reference's exactly** — `MAD 0.0000`, `100.00%` within 1 px, on every level
and on the nodes the reference actually searches:

| level | before (ref-searched nodes) | after |
|---|---|---|
| chip 192 | med +16.5 px, 0.3% within 1 px | **med 0.0000, MAD 0.0000, 100.00%** |
| chip 384 | med +49.5 px, 0.0% | **med 0.0000, MAD 0.0000, 100.00%** |
| chip 768 | med +115.5 px, 0.0% | **med 0.0000, MAD 0.0000, 100.00%** |

Two details the implementation had to get right, each measured rather than assumed:

- **The nodata fill is averaged in, not excluded.** Excluding it is the more defensible rule and is *not*
  what the reference does: against the captured chip-768 lattice the plain average differs by a single
  constant (the level's pad) where the fill-excluding form gives **3,288 different offsets**. Matched, not
  endorsed.
- **The snap takes its phase from the grid.** `round(x + 0.5) - 0.5` is a snap to half-integers, correct for
  the reference because `runAutorift` has already set `xGrid = round(xGrid) + 0.5`. A `PointSet` carries no
  such guarantee — `gridpoints` gives integers — and the literal form moves every node of an integer grid by
  half a pixel. Verified against OpenCV: on an integer-valued grid the reference's snap pushes block means of
  `15, 55, 95` onto `15.5, 55.5, 95.5`, off the input lattice; on a half-integer grid it is a no-op.

### Whole-grid effect on L1

```
julia --project=tools/golden -t 10 tools/golden/compare_figures.jl NISAR_L1_PR_RSLC --run 100
```

| | before | after |
|---|---:|---:|
| `dx` exact | 73.9% | **76.9%** |
| `dy` exact | 73.9% | **78.2%** |
| level disagreement | 0.7% | **0.4%** |
| jl-only / py-only coverage | 11,633 / 14,897 | **3,410 / 10,159** |
| correlate wall clock | 649.8 s | 767.1 s |

`figs/nisar_l1_after_cellmean_fix.png`: **the broad one-signed red bias over the fast-flow region is gone.**
What remains is thin dark lines tracing the swath's diagonal boundary — the input-masking defect
`tools/golden/README.md` records as item 0, not a coarse-level bias.

**The 18% runtime cost is not the cell averaging, and is work that should always have been done.** Profiled
on a NISAR-L1-shaped 2328x2304 grid, `_cell_centres` costs 107-115 ms per decimating level, **0.33 s of a
767 s run (0.04%)** — against 40 ms for the `_grid_step` scan it replaced, so the helper itself cannot
account for a 117 s difference. The searchable set and the radii are unchanged by the fix (both come from
`windowmax`/`windowrange` over the cell, which does not depend on where the node sits). What changes is how
many points *fit the image*: `track!` skips a point whose chip or search window falls outside it, and the old
rule's nodes were ~115 px from where they belonged, pushing some out of bounds. Counted per level:

| level | searchable | fit the image, old rule | fit, cell mean |
|---|---:|---:|---:|
| chip 192 | 474,845 | 466,010 (98.1%) | **472,340 (99.5%)** |
| chip 384 | 119,615 | 116,901 (97.7%) | **118,848 (99.4%)** |
| chip 768 | 30,337 | 30,007 (98.9%) | **30,020 (99.0%)** |

So about 8,000 coarse points per run were being silently skipped for being out of bounds at a position they
should never have had, and the correlator now does that work. Both helpers are type-stable (`Float64`) and
`_cell_mean` allocates nothing, so there is no optimization to make here — the cost is the correlation, not
the averaging.

Tests pass 704,367/704,367. Two assertions in `test/multichip.jl` were updated: one asserted the node sits at
a fixed offset from its cell's first point, which is the rule that was wrong, and is now stated as the cell
mean plus the snap; the other now also pins the fill-straddling behaviour so the matched-not-endorsed choice
is covered.

### Is the reference correct? No — and we now mirror two of its errors

Tested independently of AutoRIFT.jl, on synthetic grids and on the L1 grid, by calling OpenCV directly.

**1. A coarse node on a footprint-edge cell is placed at a coordinate that is not in the data.** `INTER_AREA`
averages the whole cell, so a cell that is part nodata fill yields a coordinate pulled toward the fill
constant. On a synthetic grid with a diagonal boundary, every straddling cell's node lands outside the range
its real coordinates span. On the L1 grid:

| level | straddling cells | node outside the real data | of those, the reference searches |
|---|---:|---:|---:|
| chip 384 | 15,614 | 15,458 (99.0%) | 4,962 |
| chip 768 | 7,141 | 6,728 (94.2%) | 3,967 |

At chip 768 that is **87.2% of the reference's 4,585 searched nodes** sitting on a straddling cell. The
correlator is then pointed at a place the grid does not describe.

**2. It is internally inconsistent on exactly those cells.** `INTER_AREA` puts the node at the cell *mean*
while `INTER_CUBIC` reads it back from the cell's geometric *centre* — the same position only when the cell is
uniform. Measured on the L1 chip-768 lattice, in grid coordinates:

| population | n | correlate-vs-readback gap |
|---|---:|---:|
| fill-free cells | 556 | **0.25 px** — self-consistent |
| straddling cells | 4,029 | **1,865 px** median, up to 27,538 |

So the reference measures at one place and attributes the answer to another, by a median of nearly two
thousand pixels, on 88% of the nodes its coarsest level searches. An earlier section here concluded the
reference "is self-consistent, and a coarse residual cannot be charged to it disagreeing with itself" — that
was measured on fill-free cells only and is correct only there.

**3. Its snap assumes its own grid convention** (above), which is a latent defect rather than an active one,
since its grid always satisfies the assumption.

**What this means for the golden work.** Matching the reference on (1) and (2) was still the right move: they
are properties of its *grid*, so a difference there desynchronizes everything downstream and makes every other
comparison unreadable — which is exactly what the 115.5 px offset was doing. But both are now on the
matched-not-endorsed register, and both have the same fix as item 0: **mask the input** so a chip whose
footprint is not sufficiently inside the valid region is never correlated. That removes the straddling cells
from the problem rather than arguing about how to average them, and it is a correctness improvement neither
implementation currently has.

## Step: the radar and NISAR gates after the cell-mean change

`_cell_means` (`src/multichip.jl`) places a coarse node at its cell's block mean. The eight radar rows and
both NISAR rows above predate it. Re-measured at the gates' own commands — `correlator.jl <case> --run 200`
for radar, `--run 100 --stride 4 --block 128` for NISAR — against "Step: the eight radar pairs, re-measured
after the stride and grid-step fixes" and "The gate: `3.nisar`, on a thinned grid":

| # | case | both | Δ both | exact dx | core bias dx | core bias dy | corr dx | corr dy | tail dx |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `S1A_..._20150828T162412` | 1,365,687 | +55,123 | 29.41% → **62.29%** | −0.00000 → −0.00019 | −0.00029 → +0.00037 | +0.903 → **+0.9886** | +0.906 → +0.9895 | 0 → 0 |
| 2 | `S1A_..._20151120T080202` | 66,792 | +6,720 | 0.00% † → **32.19%** | −0.00428 → +0.00111 | +0.00408 → +0.00019 | +0.968 → +0.9979 | +0.930 → +0.9949 | 167 → 4 |
| 3 | `S1A_..._20170221T204710` | 1,678,471 | +11,840 | 69.08% → **81.58%** | +0.00715 → **−0.02370** | +0.00239 → −0.00275 | +0.996 → +0.9984 | +0.985 → +0.9952 | 35 → 10 |
| 4 | `S1B_..._20180809T204617` | 501,820 | +11,874 | 69.26% → **76.39%** | +0.00479 → **−0.03830** | +0.00475 → −0.00051 | +0.987 → +0.9915 | +0.971 → +0.9946 | 0 → 0 |
| 5 | `S1C_..._20250416T010214` | 501,072 | +16,601 | 46.54% → **63.06%** | +0.00560 → **−0.01781** | −0.00155 → −0.00083 | +0.983 → +0.9951 | +0.896 → +0.9880 | 0 → 0 |
| 6 | `S1C_..._20250416T010159` | 38,028 | +2,042 | 36.17% → **42.89%** | −0.04313 → **−0.07607** | −0.01656 → −0.01372 | +0.941 → +0.9797 | +0.850 → +0.9481 | 0 → 0 |
| 7 | `S1A_..._20240618T025533` | 575,180 | +28,281 | 63.24% → **72.56%** | +0.01181 → **−0.02253** | +0.00556 → −0.00530 | +0.992 → +0.9985 | +0.982 → +0.9973 | 14 → 4 |
| 8 | `S1A_..._20240618T025528` | 1,558,493 | +52,720 | 54.77% → **72.79%** | −0.00107 → +0.00025 | +0.00010 → −0.00008 | +0.988 → +0.9986 | +0.961 → +0.9984 | 12 → 4 |

**Every case improves on coverage, exact match and both correlations. The core `dx` bias is the only
statistic that gets worse, and it does so on five of the eight.** `both` rises on all eight (+185,201 in
total), `exact` rises as a *fraction* on all eight where it previously fell against a growing population,
and the `dx` tail falls on every case that had one. Cases 3, 4, 5 and 7 flip their `dx` bias from positive
to negative and grow it 3-5x; case 6 was already negative and doubles. Cases 1, 2 and 8 stay within
0.0012 px.

**Gate `3.rdr` is 3 of 8**, red on cases 3, 4, 5, 6 and 7 against its `0.010 px` core-bias threshold. The
three newly red are 3, 4 and 5; 6 and 7 were already over it.

† The `0.00%` was recorded as being 0 "by construction" — `20151120`'s base level running a coarse pass and
a `filtDisp` but no fine pass, so every reported point is bicubic-resized rather than quantized. It is now
32.19%, so that no longer describes this case and the `3.rdr` closure's note that `exact` cannot be
asserted on it is stale.

### NISAR, on the thinned grid the gate uses

| case | exact dx | core bias dx | core bias dy | corr dx | corr dy |
|---|---:|---:|---:|---:|---:|
| L1 `--stride 4 --block 128` | 16.81% → 17.71% | +0.0975 → **−0.7279** | −0.1411 → −0.1731 | +0.99853 → +0.99684 | +0.98765 → +0.99378 |
| L2 `--stride 4 --block 128` | 68.70% → **30.53%** | +0.1758 → **−0.4359** | +0.2343 → **−0.5787** | +0.99962 → +0.99188 | +0.99637 → +0.96092 |

**Gate `3.nisar` is 0 of 2.** Both `dx` biases exceed their bounds by roughly 6x and 2x, and unlike the
radar cases **L2 is worse on every statistic**, not only on bias.

**The thinned run is no longer a proxy for the whole grid.** Placement now depends on the *point grid's own*
cell means, so a grid thinned by `--stride 4` puts its coarse nodes where a whole-grid reference never put
any — and the two sides resolve different pyramid levels to begin with, which is why this gate was already
calibrated to the thinned run rather than to whole-grid figures. The whole-grid L1 measurement moved the
other way over the same change, 73.91% → 76.9% exact on `dx` ("Step: the cell-mean fix, and whether the
reference is right"), so the thinned degradation is the proxy breaking rather than the correlator.

### Why the thresholds are not simply widened

The `dx` bias carries the fill averaging on `CORRECTNESS.md`'s item 2 list: a cell straddling the swath edge
averages real coordinates against the in-band nodata constant, which pulls its node one way only. That is
matched deliberately, and it is measured directly on NISAR L1 — the node lands outside its own cell's real
coordinate range for 94.2% of straddling cells at chip 768. So the statistic these two gates fail on is
dominated by a difference the project is currently choosing to keep, while every statistic that reflects
agreement more broadly improved.

Two ways to make the gates informative again, neither taken here because widening a bound to admit a
measurement is what this ledger exists to prevent:

- **Re-calibrate both gates at this placement**, recording cases 3, 4, 5, 6 and 7 as expected red with
  these figures, so a *new* regression is again the only thing that reds them.
- **Gate something the fill averaging does not dominate** — coverage, exact count, correlation and the tail
  all moved the right way on all eight radar cases and would have caught nothing spuriously.

Either is a calibration decision. What is established here is the attribution: the reds are this change,
they are the `dx` core bias alone on radar, and no other gated statistic regressed.

# Gate 5 — the end-to-end ladder, granule to geogrid

`tools/golden/e2e.jl` starts where the reference starts: at the granule. Every other comparison in
`tools/golden/` is fed the reference's own arrays — `capture.py` dumps the filtered, byte-quantized
pair and the snapped grid at the `runAutorift` boundary, and `pointset_from_capture` turns the
reference's grid into the `PointSet` — so before this, nothing upstream of `runAutorift` had ever run
in Julia on a golden case.

The reference side needs no new compute. Every boundary's answer is already in the cached run
directory: the nine `window_*.tif`, `autoRIFT_intermediate.nc`, the `capture/` arrays, and
`filtered/` for the pairs the driver filters before geogrid.

## The rungs that exist

| rung | Julia produces | reference truth | gate |
|---|---|---|---|
| 5.0 | the output `MapGrid` — geotransform and size | `window_location.tif`'s own | exact |
| 5.5 | the geogrid, all 17 bands | the nine `window_*.tif` | integer bands exact, `Float64` bands ≤ 1e-7 relative |
| 5.6 | the driver's scene-wide parameters | `capture/call1.json` scalars | exact |

The two-tier gate on 5.5 is `ImagePairGeometry`'s own standard and the tiers split where they do for a
reason: `window_location`, `window_offset`, `window_search_range`, the two chip-size files and the
stable-surface mask all pass through a rounding or truncating conversion that absorbs a last-bit
difference, while the off2vel and scale-factor bands do not.

Rungs 5.1 through 5.4 — the granule read, the secondary onto the reference's grid, the filter on
Julia's own read, and the byte rescale — are not yet wired into the ladder. `bytescale` (rung 5.4's
Julia side) exists and is pinned against the reference by fixtures; see below.

## Result: eight of twelve optical cases, every rung green

`FastGeoProjections`, the default, which is what production uses. Four cases are absent because their
two scenes are in different UTM zones and `coregister` refuses them exactly as the reference does
(`GeogridOptical.py:297-298`); reprojecting the secondary is rung 5.2 and does not exist.

`worst float` is the largest absolute disagreement over all six `Float64` bands, in that band's own
units — m/yr per pixel of displacement for the off2vel entries, dimensionless for the scale factors.

| case | platform | grid | rungs | worst float |
|---|---|---:|---|---:|
| `LC08_L1TP_009011` | L8 | 5,503,691 | **23/23** | 1.4e-7 |
| `LC08_L1TP_062018` | L8 | 5,352,100 | **23/23** | 1.5e-6 |
| `LC09_L1GT_215109` | L9 | 5,262,435 | **23/23** | **0** |
| `LE07_L1TP_063018_20040810` | L7 | 5,325,012 | **23/23** | 2.2e-6 |
| `S2A_MSIL1C_20200626` | S2 | 1,092,025 | **23/23** | 1.3e-5 |
| `S2B_MSIL1C_20200612` | S2 | 1,018,081 | **23/23** | 4.4e-7 |
| `LT04_L1TP_063018` | L4 | 5,202,900 | **23/23** | 4.4e-6 |
| `LT05_L1GS_001013` | L5 | 5,066,604 | **23/23** | 1.8e-7 |
| `LC08_L1TP_060018_20130330` | L8×L7 | — | cross-zone, 32608 × 32607 | — |
| `LE07_L1TP_061018_20120428` | L7 | — | cross-zone, 32607 × 32608 | — |
| `LE07_L1TP_061018_20130314` | L7×L8 | — | cross-zone, 32607 × 32608 | — |
| `LT05_L1TP_060018` | L5 | — | cross-zone, 32608 × 32607 | — |

Every integer band is identical to the container's output over **33,822,848 grid points** across the
eight, on four platforms and three projections (32622, 32607, 3413, 3031), bar the single `search_x`
rounding tie below. The grid geotransform and size match exactly on all eight, so the two sides are
comparing the same points before any band is read.

`LC09_L1GT_215109` is the case with no float disagreement at all: its scene is already in EPSG:3031
and the parameter region is the southern polar grid, so the transform is the identity and every band
is bitwise. That is a useful control — it says the residual on the other seven is the reprojection and
nothing else in the kernel.

## Four driver conventions, each of which leaves the case looking two-thirds right

None of these is visible to a comparison fed the reference's own arrays, and each one, when wrong,
leaves `window_location`, both chip-size bands and the stable-surface mask **exact** while corrupting
the bands that depend on the acquisition interval. A case at 11/17 bands reads as a subtle numerical
problem and is a convention.

**`dt` is a whole number of calendar days.** `testGeogridOptical.py:161-165` builds two
`datetime.date` objects from the first eight characters of each scene name's date field, so the time
of day is discarded. The product's `date_dt` is a *different* quantity, computed later in
`netcdf_output.py` from the full timestamps. Measured on `S2B_MSIL1C_20200612`, feeding geogrid the
product's 15.0009490740741 days instead of 15 moves **75 points of `window_offset` and 445 of
`window_search_range`** by one pixel — 0.007% and 0.044% of the grid.

That this was a rounding difference rather than a systematic one was ruled out before the cause was
found, which is the part worth keeping. Perturbing `dt` by one ULP, and then by 1e-6 s, left the
*same* 75 and 445 points differing, so it was not a tie. Recovering the pre-round float — by scaling
the velocity raster by `2^20`, since the offset is linear in it, and dividing back — put every one of
the 75 at a fractional part between 0.5000010 and 0.5024071, against a control median of 0.176 over
the agreeing points and only 0.2% of them within 0.001 of a half. So the reference's float was
systematically *smaller* in magnitude by about 3e-5 relative, which is 39 s on 1,296,082 — and the
82 s of clock time between the two acquisitions is exactly that.

**The pair is taken in acquisition order, not the job's.** Two of the twenty-two jobs name the later
acquisition as the reference, and their products still report the earlier one as `id_img1` with a
positive `date_dt`: on `LE07_L1TP_063018`, `img1` is 20040810 and `date_dt` is +32.000 while the job's
reference is 20040911. Following the job order negated `window_offset` and all four `off2vel` bands —
a velocity field pointing backwards — and *doubled* `window_search_range`, because the short-interval
inflation `max(1, 5 - 4·dt/182)` grows as the interval falls below zero. 191,991 `offset_x` points and
1,890,950 `search_x` points on that case, with every chip-size and mask band still exact.

**`ArchGDAL.toEPSG` returns the base geographic code for a projected CRS.** `LC09_L1GT_215109` is a
Landsat scene over the Antarctic peninsula carrying a custom `PROJCRS["PS         WGS84"]` with no
authority code of its own; `toEPSG` walks down to the `BASEGEOGCRS` and answers **4326**, which is a
real code for a different coordinate system. Nothing errors: the grid-to-scene transform becomes
4326→4326, a no-op, and the pair's centroid stays in metres — so the parameter-region lookup is handed
a point 2,000 km outside the Earth's coordinate range and the case fails on the lookup rather than on
the CRS. The reference's own procedure works and is what `scene_epsg` now reproduces:
`AutoIdentifyEPSG`, then the `PROJCS` authority code, falling back to `OSRFindMatches`
(`GeogridOptical.py:93-123`). That resolves the same scene to **3031** at 100% confidence.

**GDAL's window offsets are zero-based.** `ImagePairGeometry.grid_window` returns one-based
`CartesianIndices`, so reading each parameter raster at `first(xs)` shifts all twelve of them by one
pixel in both axes. The signature is unmistakable once seen and reads as arithmetic until then:
`window_location` stays exact because it reads no parameter raster, while `chip_min` differs by
±24/48/96 — whole steps of the chip-size quantization — the stable-surface mask by ±1, and `offset`
and `search` by up to 39 and 61 pixels. 0.38% to 8.9% of points, all interior, none on the footprint
ring.

## FastGeoProjections against PROJ, measured

The gate runs `FastGeoProjections`, because that is what a production run uses. PROJ is reached through
that package's own `proj_only` keyword — `--proj-only` on the command line — rather than by constructing
against a second library: `FastGeoProjections` already falls back to Proj for any CRS pair it has no
native implementation for, so it is one interface with a backend flag. It is an attribution tool rather
than a requirement: the reference builds its transforms with `osr.CoordinateTransformation`, so forcing
Proj takes the projection library out of the comparison and leaves whatever remains belonging to the
kernel arithmetic.

**The choice is very nearly invisible, and an earlier reading in this file said otherwise.** What the
two transforms do to the quantities the geogrid consumes, measured on `S2A_MSIL1C_20200626`
(3413 → 32607, a 120 m grid over 10 m imagery):

| quantity | fast against PROJ |
|---|---:|
| position | **1.742e-7 m** |
| a one-grid-cell step, relative | **7.278e-11** |
| every integer band | **0 of 1,092,025 differ** |

So the two agree to 174 nanometres in position and to 7e-11 in the *difference* the kernel actually
divides by — and produce the identical geogrid, band for band, at every point.

Across all eight same-CRS optical cases the integer bands differ at **one point of 33.8 million**: on
`LC08_L1TP_062018` a single `search_x` reads 34 against PROJ's 35. That point is a rounding tie, and
this is a measurement rather than an inference — scaling its search range by a relative **1e-9** flips
PROJ to 34 as well, so PROJ's own pre-round value sits within about 3.5e-8 of a pixel of the 34.5
boundary. A 7e-11 step disagreement is enough to straddle it, neither answer is wrong, and PROJ lands
on the container's side only because it is the same library. `rounded_stage` admits a tie and nothing
else; a convention error moves tens of thousands of points.

The `Float64` bands, same case, reported absolutely as well as relatively because the ratio alone
misleads:

| band | max absolute | max relative | median \|value\| |
|---|---:|---:|---:|
| `off2vx_dx` | 5.638e-7 | 7.731e-9 | 74.6 |
| `off2vx_dy` | 1.320e-6 | 1.819e-9 | 726.2 |
| `off2vy_dx` | 5.661e-8 | 7.794e-11 | 726.2 |
| `off2vy_dy` | **1.265e-5** | **1.724e-7** | 74.6 |
| `scale_x` | 7.887e-10 | 7.570e-10 | 1.04 |
| `scale_y` | 1.813e-8 | 1.739e-8 | 1.04 |

`off2vy_dy` is metres per year per pixel of displacement, so its worst disagreement is 1.3e-5 — a
millimetre per year on a hundred-pixel displacement, against velocities of hundreds to thousands, and
six orders below the 1 m/yr the product quantizes to. Its *relative* figure is the largest of the six
only because the ratio is taken where the coefficient is near its own minimum.

**What the earlier reading got wrong**, recorded because it is a method error rather than an
arithmetic one. `ImagePairGeometry`'s `REFERENCE.md` bounds the `Float64` bands at 1e-7 relative, and
that figure is calibrated for **PROJ against PROJ on a different platform**. Applying it to a
comparison that also swaps the projection library gates on a threshold that was never about this
question, and reporting a bare ratio hid the fact that the quantity behind it was 1.3e-5 m/yr. Three
cases then read as red and the conclusion drawn — that PROJ was required — inverted the truth.
`relative_stage` now reports both magnitudes and gates at 1e-6, six times the measured worst across
the eight cases rather than a number borrowed from elsewhere.

## `bytescale`, and the one level that does not close

`uniform_data_type` (`autoRIFT.py:345-379`) had no Julia implementation, so rung 5.4 could not exist.
`AutoRIFT.bytescale` is it, pinned by six fixtures in `test/fixtures/bytescale/` generated from the
reference's own statements.

Three details each move the result by a level: the standard deviation is the **sample** form
(`np.std` is the population form and the reference corrects it by `sqrt(n/(n-1))`); the multiplier is
**256**, written `2**8 - 0`; and `np.clip` runs **before** `np.round`, so a value above the window
becomes exactly 255 rather than rounding to 256 and wrapping to 0 — a bright pixel reported as no
data.

The arithmetic runs in the image's own precision, because `loadProduct` casts to `float32`
(`testautoRIFT.py:120-124`) and every statistic and per-pixel expression after it is `float32`.
Computing in `Float64` is more accurate and disagrees with the reference on **3 pixels of 33,218**;
matching the precision takes that to **1**. The last one is `numpy`'s pairwise-summation block
structure: the mean and standard deviation are reductions over the whole image, so any difference in
accumulation order moves them in the last bits and moves whatever sits nearest a rounding boundary.
Not matched — reproducing `np.mean`'s block structure is brittle for one pixel in thirty thousand —
and recorded rather than absorbed into a wider tolerance.

## What Gate 5 does not yet establish

- **The four cross-zone optical pairs**, which need the secondary reprojected onto the reference's
  grid. `process.py` writes that to `reprojected/`, and `run_reference`'s `prune` default deletes it,
  so rung 5.2 needs either a run with `prune = false` or a comparison that does not use it.
- **Rungs 5.1, 5.3 and 5.4 in the ladder.** The granule read, the filter on Julia's own read, and the
  byte rescale are each implemented and separately tested; none is yet compared against the
  reference's own arrays *inside* the ladder.
- **The endpoint.** Nothing here reaches `autoRIFT_intermediate.nc`; rungs 5.0, 5.5 and 5.6 establish
  that both sides would be handed the same grid, priors, search limits, chip bounds and scene-wide
  parameters, which is the premise the endpoint comparison needs and did not have.
- **The ten radar and NISAR cases.** `coregister` needs the orbit-driven resample for those, which is
  the `ImagePairGeometry` coregistration path.
