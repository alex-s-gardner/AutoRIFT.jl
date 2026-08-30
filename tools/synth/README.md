# Synthetic benchmark against exact ground truth

Three implementations correlated on scenes whose displacement is known analytically at every pixel:
AutoRIFT.jl, `autoRIFT.py`, and raw OpenCV `matchTemplate` with no autoRIFT machinery around it.

The A/B harness beside this one (`tools/ab/`) compares AutoRIFT.jl against the reference on real
Landsat, where neither answer is known to be right. That comparison found the two indistinguishable
over most of a scene and disagreeing by more than 0.25 px at 6.7% of points, concentrated at maximum
flow, and it could establish which side was closer only through arbiters — OpenCV's own correlation
surface, a chip-size convergence ladder, raw `matchTemplate` end to end. Every one of those is still an
argument about a scene whose truth nobody has.

Here the truth is constructed, so "which is right" is a measurement.

## Running it

```bash
julia --project=tools/synth tools/synth/scenes.jl          # generate the 26 cases (~11 s)
julia --project=tools/synth tools/synth/run_julia.jl       # AutoRIFT.jl arms (~57 s)
micromamba run -n arift-ref  python tools/synth/run_python.py   # the reference arms (~33 s)
micromamba run -n autorift-cv python tools/synth/run_opencv.py  # raw OpenCV arms (~40 s)
julia --project=tools/synth tools/synth/gate.jl            # conventions, before believing anything
julia --project=tools/synth tools/synth/score.jl           # -> results.json and the tables
julia --project=tools/synth tools/synth/figures.jl         # -> plots/
```

The two Python environments are the ones `tools/ab/README.md` and `tools/python_ref/` describe. Scene
bundles and figures are gitignored — they regenerate in under two minutes; `results.json` is the
committed record.

## The arms

Five, split so a difference is attributable to a stage rather than to an implementation as a whole.

| arm | what runs | isolates |
|---|---|---|
| `jl_correlator` | `correlate!` + `subpixel_peak`, one chip | AutoRIFT.jl's matching and refinement |
| `py_correlator` | `arImgDisp_s`, one chip | the reference's matching and refinement |
| `cv_pyrup` | `matchTemplate` + the same `pyrUp` cascade | OpenCV matching, refinement held equal |
| `cv_parabola` | `matchTemplate` + a 3-point parabola fit | what a naive OpenCV user writes |
| `jl_pipeline` / `py_pipeline` | `autorift` / `runAutorift` | pyramid, coarse pass, `DISP_FILT`, fill, merge |

`cv_pyrup` and `cv_parabola` are separate arms because the sub-pixel estimator dominates the result. A
parabola through three samples of a discrete peak is biased toward the sample it is centred on, so an
OpenCV arm scored with one confounds "OpenCV" with "a biased estimator" — measured at 0.0167 px even on
an exact integer shift, where every other arm is exact.

A sixth arm, `jl_rotation`, runs `rotation = RotationSearch()` on the deforming cases. `autoRIFT.py` has
no equivalent, so it is not part of the like-for-like comparison.

## Conventions are measured, not assumed

`gate.jl` scores every arm against the known field under all four sign combinations and reports the one
that matches, plus the runner-up so it is visible whether the field actually distinguishes them. It then
requires each arm to recover a pure translation, where a rigid chip is an exact model and there is no
modelling error to hide behind.

| arm | sign in truth space | integer shift | sub-pixel shift |
|---|---|---:|---:|
| `jl_correlator`, `jl_pipeline`, `cv_pyrup` | `(−dx, −dy)` | **0.0000** | **0.0000** |
| `py_correlator`, `py_pipeline` | `(+dx, +dy)` | **0.0000** | **0.0000** |
| `cv_parabola` | `(−dx, −dy)` | 0.0167 | 0.0611 |

The Julia and OpenCV arms report where the secondary chip sits within the reference window, which is the
negative of feature motion; `arImgDisp_s` returns feature motion directly. Four of five arms recover
both translations exactly. Asserting these signs rather than measuring them is what produced two sign
errors in earlier work on this comparison, which is why the gate exists and why it runs first.

## The cases

One base case, a one-factor sweep around it, and a full crossing of the two factors that interact —
deformation amplitude and chip size, since the mechanism under study is differential motion *within a
chip footprint*, which is their product. 26 scenes at 1024², not a full factorial: crossing every factor
is 432 scenes, most of them re-measuring what one factor already establishes.

- **Deformation mode**, each an analytic field: translation (integer and sub-pixel), rigid rotation at
  0.5° and 2°, uniform divergence, and a `tanh` shear margin at three amplitudes.
- **Noise**: additive Gaussian at SNR 10, 4, 2, plus one case with noise on the secondary only.
- **Decorrelation**: 10%, 25%, 50% of the scene's texture regenerated between acquisitions in patches
  rather than per pixel — what a crevasse field does, and what no existing test in the repository
  produces.
- **Quantization**: the reference's uint8 path (`autoRIFT.py:359-385`), which is what ITS_LIVE
  production ships and what every prior A/B run pinned off so requantization could not masquerade as a
  correlator difference.
- **Texture**: band-limited synthetic against a real Landsat patch from the fast trunk, warped by the
  same analytic fields.

Two properties of the scene generator were measured rather than assumed, because both would otherwise
have put a floor under every number in the tables:

**The warp is not the limiting error.** Against an exact Fourier phase-ramp shift — the analytically
correct answer for a band-limited field — the cubic-spline warp gives an identical recovered
displacement at band limits of 0.5, 0.8 and 1.0 Nyquist. What remains is the 1/16 px upsampling step,
which is a property of the estimators under test.

**Texture is band-limited to half Nyquist.** A sub-pixel shift of a spectrally white field is not
well-posed: energy at Nyquist cannot be translated by any interpolator, local or exact. Recovering a
known 0.729 px shift gives 0.0000 px error at half Nyquist against 0.0263 px at full band, and that
0.026 px would otherwise have been attributed to the correlators. Half Nyquist is also what the real
imagery is: the Landsat patch carries 95.8% of its power there.

## Results

### AutoRIFT.jl and raw OpenCV are the same answer

Bit-identical `Float32` at **99.99–100.00% of points in all 26 cases**. The residual is single 1/16 px
quantization steps where two candidate peaks tie.

This is the finding that makes the rest interpretable. Two independently written implementations — an
FFT-based ZNCC in Julia and `cv2.matchTemplate` in C++ — returning the identical float is not a
near-agreement to be reported as a small error; it means either can stand in for the other, and OpenCV's
maturity is evidence about AutoRIFT.jl rather than a competing opinion.

### Median error against exact truth, px

| case | AutoRIFT.jl | autoRIFT.py | CV+pyrUp | CV+parabola |
|---|---:|---:|---:|---:|
| `translate_integer` | **0.0000** | **0.0000** | **0.0000** | 0.0167 |
| `translate_subpixel` | **0.0000** | **0.0000** | **0.0000** | 0.0611 |
| `translate_subpixel_uint8` | **0.0000** | **0.0000** | **0.0000** | 0.0612 |
| `translate_subpixel_landsat` | 0.0884 | 0.0884 | 0.0884 | 0.1920 |
| `divergence` | 0.0323 | 0.0320 | 0.0323 | 0.0439 |
| `rotation_0p5deg` | 0.0336 | 0.0425 | 0.0336 | 0.0441 |
| `rotation_2deg` | **0.0598** | 0.4706 | **0.0598** | 0.0714 |
| `shear_a2.5_c16` | 0.0001 | 0.0001 | 0.0001 | 0.0325 |
| `shear_a10_c16` | 0.0003 | 0.0003 | 0.0003 | 0.0194 |
| `shear_a20_c16` | **0.0157** | 0.0625 | **0.0157** | 0.0139 |
| `shear_a20_c32` | **0.0005** | 0.0625 | **0.0005** | 0.0057 |
| `shear_a20_c64` | **0.0002** | 0.0624 | **0.0002** | 0.0025 |
| `shear_snr10` | 0.0011 | 0.0013 | 0.0011 | 0.0237 |
| `shear_snr4` | 0.0573 | 0.0623 | 0.0573 | 0.0391 |
| `shear_snr2` | 0.0660 | 0.0757 | 0.0660 | 0.0732 |
| `shear_snr4_secondary` | 0.0214 | 0.0319 | 0.0214 | 0.0308 |
| `shear_decorr0.1` | 0.0043 | 0.0079 | 0.0043 | 0.0238 |
| `shear_decorr0.25` | 0.0617 | 0.0625 | 0.0617 | 0.0414 |
| `shear_decorr0.5` | 0.1768 | 0.2988 | 0.1768 | 0.1693 |
| `shear_uint8` | 0.0003 | 0.0003 | 0.0003 | 0.0194 |
| `shear_landsat` | 0.0319 | 0.0359 | 0.0319 | 0.0249 |
| `shear_landsat_uint8` | 0.0327 | 0.0389 | 0.0327 | 0.0256 |

The remaining chip/amplitude cells are in `results.json`.

AutoRIFT.jl is at least as accurate as the reference in every one of the 26 cases, and materially better
in the deforming ones. Where the scene is a pure translation all three matchers are exact and there is
nothing to choose between them.

### The disagreement reproduces, and it tracks deformation

`shear_a10_c16`, stratified by local deformation rate — the Frobenius norm of the displacement gradient,
which is the right variable because a chip of side `c` sees roughly `c × rate` px of differential motion
across its own footprint:

| deformation px/px | n | AutoRIFT.jl | autoRIFT.py | disagreement | >0.25 px |
|---|---:|---:|---:|---:|---:|
| 0.0000 | 7381 | 0.0000 | 0.0000 | **0.0000** | 0.0% |
| 0.0000–0.0043 | 4477 | 0.0035 | 0.0035 | **0.0000** | 0.0% |
| 0.0043–0.0981 | 2299 | 0.0436 | 0.0698 | 0.0884 | 16.1% |
| 0.0981–0.1454 | 847 | **0.1276** | 0.4032 | 0.3366 | 66.2% |

Zero disagreement across 11,858 points of low deformation, rising to two-thirds of points beyond 0.25 px
in the steepest stratum. This is the real scene's signature reproduced against known truth, and it
confirms the prediction that it would be hard to construct: a pure translation shows no disagreement at
all.

### The mechanism is deformation within the chip, not gradient variation

The uniform-deformation modes settle this. Divergence and rigid rotation apply the *same* deformation
rate at every pixel, so there is no gradient to vary and no ambiguity about which points are hard:

| case | rate px/px | AutoRIFT.jl | autoRIFT.py | ratio |
|---|---:|---:|---:|---:|
| `divergence` | 0.00566 | 0.0323 | 0.0320 | 1.0× |
| `rotation_0p5deg` | 0.01234 | 0.0336 | 0.0425 | 1.3× |
| `rotation_2deg` | 0.04936 | **0.0598** | 0.4706 | **7.9×** |

A 2° rigid rotation is the sharpest case in the whole sweep. Both implementations are fitting one rigid
translation to a chip whose texture has genuinely rotated, and neither models that — but the reference
degrades 7.9× faster.

### Rotation search does not recover it

`rotation = RotationSearch()` over 19 deforming cases: **0 better, 4 worse, 15 unchanged**. It hurts on
the decorrelated and low-SNR cases, where taking the best of three correlations selects noise.

A rigid rotation of the chip cannot represent shear or dilation, and even on the rotation cases the
deformation is *within* the chip rather than of the chip as a whole. So the residual at high deformation
is chip-scale strain that no rigid model captures — which bounds how much of the gap is a defect in
anything and how much is the problem being ill-posed at that deformation rate.

### uint8 quantization changes nothing

`shear_uint8` matches `shear_a10_c16` to four decimals for every arm (0.0003), and
`translate_subpixel_uint8` is exact. On real texture the cost is 0.0009 px. The anomaly is therefore not
an artifact of the float path, and it is present in what ITS_LIVE production ships — closing a gap every
prior A/B run left open by pinning `DataType = 1`.

### Real texture behaves like synthetic

`shear_landsat` gives 0.0319 against `shear_a10_c16`'s 0.0003 — a higher floor from real texture's
noise and anisotropy — but the same ordering and the same ratio between implementations. The synthetic
conclusion is not an artifact of synthetic texture.

## What this does not show

- **`py_pipeline` resolves nothing at chips 32 and 64 here.** Not an accuracy result: it is the
  window-size artifact `tools/ab/README.md` documents, and this sweep pins the arithmetic.
  `ChipSize0_GridSpacing_oversample_ratio = ChipSize0X / GridSpacingX` is 4 at chip 32 and 8 at chip 64,
  so the coarse grid decimates by 16 and 32 to 7×7 and 3×3 while `DispFiltC.FiltWidth` becomes 17 and
  33. The filter is wider than the grid it runs on, `CoarseCorValidFac` falls below `CoarseCorCutoff`,
  and `autorift()` hits `continue`. A 1024² scene is simply too small for those chips at this spacing.
- **Both pipelines lose coverage under heavy decorrelation** — 47.7% and 38.8% at
  `shear_decorr0.5`. That is the outlier filter doing its job on a scene half of which has no
  correspondence to find.
- **Absolute error magnitudes are not field accuracy.** These are single-level correlator runs on
  1024² scenes with a fixed 20 px radius. The comparison between arms transfers; the numbers themselves
  do not.
- **The correlator arms bypass preprocessing.** Every scene is correlated as generated, so a filter
  difference cannot masquerade as a correlator difference — but neither is the Wallis filter production
  uses exercised.
