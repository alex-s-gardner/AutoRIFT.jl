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
