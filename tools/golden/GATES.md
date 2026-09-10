# The gate ledger

Every measurement that has been confirmed green, with the command that produced it. A gate stays on
this list only while it still passes: `regate.jl` re-runs all of them, and a row that stops holding is
a regression to fix rather than a number to update.

**Why a ledger rather than a report.** `tools/golden/README.md` records what is *known* about the two
implementations. This records what has been *verified at a commit*, which is a different claim and the
one that decays. A gate whose command cannot be re-run is not a gate.

Machine: Apple M2 Max, 12 cores, macOS 26.5.2, Julia 1.12.5. Reference: autoRIFT 2.1.1 in
`micromamba -n arift-ref`, whose `autoRIFT.py` is byte-identical to the pinned v2.1.2; container
`ghcr.io/asfhyp3/hyp3-autorift:0.28.4` for the golden cases.

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

The `exact` spread across cases — 54% to 93% — tracks the **scene**, not the code:

- Every stage is exact or reported-as-designed on all five, so the pipeline is not case-sensitive.
- Both S2 cases and LC09 sit at 67–93%; both LC08 cases at 54–63%. The 15 m panchromatic band quantizes
  harder onto 256 levels than S2's, and `tools/ab` stage 1 measures that path at 84.8% exact with a
  35.8 px maximum against 100% on the float path.
- So the case-to-case variation is the `UInt8` quantization interacting with scene contrast, which is
  the reference's own preprocessing and not something a change here can recover.

**The correlator work is converged.** Five of five cases pass the ladder, coverage improved on all five,
and the remaining `exact` shortfall is attributable to a quantization step the reference applies before
the correlator sees anything. The gate should move to the post-correlation chain, which is where a
product comparison becomes possible.

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
points, which is the `UInt8` quantization interacting with scene contrast — `tools/ab` stage 1 measures that
path at 84.8% exact with a 35.8 px maximum against 100% on the float path.

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

## Step: the gate covers every radar case

`3.rdr` now runs all eight rather than one, for the same reason `3.opt` runs all twelve: the two
`_oversample` and `chip_size_max` bugs were invisible on twelve optical pairs and surfaced only on an
anisotropic chip. Thresholds are the weakest measured case less a margin — core bias 0.010 px,
correlation 0.78 (case 1 is the floor at 0.820/0.825), tail 400 points.
