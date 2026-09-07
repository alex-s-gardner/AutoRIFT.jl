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
