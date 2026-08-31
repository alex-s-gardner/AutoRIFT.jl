# Synthetic benchmark against exact ground truth

Three implementations correlated on scenes whose displacement is known analytically at every pixel:
AutoRIFT.jl, `autoRIFT.py`, and raw OpenCV `matchTemplate` with no autoRIFT machinery around it.

The A/B harness beside this one (`tools/ab/`) compares AutoRIFT.jl against the reference on real
Landsat, where neither answer is known to be right, so it can only measure *disagreement*. Here the
truth is constructed, so accuracy is a measurement rather than an inference — and the two harnesses
answer different questions: this one asks whether the implementations compute the same thing, that one
asks what they do on ice.

## The result

**All three implementations agree, on every case, to the last bit.** Median error against exact truth
is identical to four decimal places in all 26 cases, and the three are bit-identical `Float32` at
99.99–100.00% of points pairwise. The residual is single 1/16 px steps where two candidate peaks tie.

| pair | bit-identical | largest difference |
|---|---:|---:|
| AutoRIFT.jl vs `autoRIFT.py` | 99.99–100.00% | 3.3750 px (one point of 14,640) |
| AutoRIFT.jl vs raw OpenCV | 99.99–100.00% | 3.3750 px (same point) |
| `autoRIFT.py` vs raw OpenCV | 99.99–100.00% | 0.0625 px |

The 3.375 px outlier is a single grid point in `shear_landsat_uint8` where the true displacement is
zero — a tie broken differently on a quantized, locally degenerate patch, not a systematic difference.

Three independently written implementations — an FFT-based ZNCC in Julia, a spatial-domain matcher in
C++, and `cv2.matchTemplate` called directly — returning the same float means each can serve as an
arbiter for the others.

## The reference's argument order

**`arImgDisp_s(a, b)` cuts its chip from `b` and its search window from `a`** — the reverse of what the
parameter names suggest. `a` binds to `I1` and `b` to `I2`, but the body calls the C++ as
`arSubPixDisp_s_Py(..., I2.shape, I2.ravel(), I1.shape, I1.ravel(), ...)` (`autoRIFT.py:1251-1256`),
and the C++ binds *its* first array to `sec_img`, which is where `chip` is cut from
(`autoriftcoremodule.cpp:413,453`). The two swaps compose rather than cancel.

`runAutorift` inherits this: it correlates with `arImgDisp_s(self.I2, self.I1)`
(`autoRIFT.py:674,747`), so its chip comes from `self.I1` — which must therefore hold the **secondary**
image for the chip to sit on the secondary.

Getting this backwards is not a sign error that a global flip absorbs. The chip is then cut from the
other image, so the estimate describes a different piece of ground: the residual is zero on a pure
translation and grows with displacement wherever the field deforms, reaching 0.47 px on `rotation_2deg` —
small enough to read as an accuracy result and large enough to look significant.

**A translation-only gate cannot detect it**, which is why `gate.jl` also requires `rotation_2deg`. On a
uniform field the displacement of the feature at the reference point and of the feature arriving at the
secondary point are the same number, so a reversed pairing recovers a pure translation at exactly
0.0000 px. Only a deforming field separates them.

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

Five, split so a difference would be attributable to a stage rather than to an implementation as a whole.

| arm | what runs | isolates |
|---|---|---|
| `jl_correlator` | `correlate!` + `subpixel_peak`, one chip | AutoRIFT.jl's matching and refinement |
| `py_correlator` | `arImgDisp_s`, one chip | the reference's matching and refinement |
| `cv_pyrup` | `matchTemplate` + the same `pyrUp` cascade | OpenCV matching, refinement held equal |
| `cv_parabola` | `matchTemplate` + a 3-point parabola fit | what a naive OpenCV user writes |
| `jl_pipeline` / `py_pipeline` | `autorift` / `runAutorift` | pyramid, coarse pass, `DISP_FILT`, fill, merge |

`cv_pyrup` and `cv_parabola` are separate arms because the sub-pixel estimator dominates the result. A
parabola through three samples of a discrete peak is biased toward the sample it is centred on, and it
is the only arm that differs from the others: 0.0167 px even on an exact integer shift, where the rest
are exact.

A sixth arm, `jl_rotation`, runs `rotation = RotationSearch()` on the deforming cases. `autoRIFT.py` has
no equivalent, so it is not part of the like-for-like comparison.

## Conventions are measured, not assumed

`gate.jl` scores every arm against the known field under all four sign combinations, reports the one
that matches plus the runner-up, and requires each arm to recover a pure translation *and* to stay
within 0.15 px on `rotation_2deg`.

All five arms measure `(−dx, −dy)`: every one cuts its chip from the secondary and searches the
reference, so each reports the offset from secondary back to reference, the negative of feature motion.

| arm | integer shift | sub-pixel shift | rotation 2° |
|---|---:|---:|---:|
| `jl_correlator`, `jl_pipeline` | **0.0000** | **0.0000** | 0.0598 |
| `py_correlator`, `py_pipeline` | **0.0000** | **0.0000** | 0.0598 / 0.0596 |
| `cv_pyrup` | **0.0000** | **0.0000** | 0.0598 |
| `cv_parabola` | 0.0167 | 0.0611 | 0.0714 |

## The cases

One base case, a one-factor sweep around it, and a full crossing of deformation amplitude with chip
size. 26 scenes at 1024², not a full factorial: crossing every factor is 432 scenes, most of them
re-measuring what one factor already establishes.

- **Deformation mode**, each an analytic field: translation (integer and sub-pixel), rigid rotation at
  0.5° and 2°, uniform divergence, and a `tanh` shear margin at three amplitudes.
- **Noise**: additive Gaussian at SNR 10, 4, 2, plus one case with noise on the secondary only.
- **Decorrelation**: 10%, 25%, 50% of the texture regenerated between acquisitions in patches rather
  than per pixel — what a crevasse field does, and what no existing test in the repository produces.
- **Quantization**: the reference's uint8 path (`autoRIFT.py:359-385`), which is what ITS_LIVE
  production ships and what every prior A/B run pinned off.
- **Texture**: band-limited synthetic against a real Landsat patch from the fast trunk, warped by the
  same analytic fields.

Two properties of the scene generator were measured rather than assumed, because both would otherwise
have put a floor under every number:

**The warp is not the limiting error.** Against an exact Fourier phase-ramp shift — the analytically
correct answer for a band-limited field — the cubic-spline warp gives an identical recovered
displacement at band limits of 0.5, 0.8 and 1.0 Nyquist. What remains is the 1/16 px upsampling step.

**Texture is band-limited to half Nyquist.** A sub-pixel shift of a spectrally white field is not
well-posed: energy at Nyquist cannot be translated by any interpolator. Recovering a known 0.729 px
shift gives 0.0000 px error at half Nyquist against 0.0263 px at full band. Half Nyquist is also what
the real imagery is: the Landsat patch carries 95.8% of its power there.

## Median error against exact truth, px

Identical across the three implementations in every case, so the table reads as a characterization of
the shared algorithm rather than a comparison.

| case | all three | CV+parabola |
|---|---:|---:|
| `translate_integer` | **0.0000** | 0.0167 |
| `translate_subpixel` | **0.0000** | 0.0611 |
| `translate_subpixel_uint8` | **0.0000** | 0.0612 |
| `translate_subpixel_landsat` | 0.0884 | 0.1920 |
| `divergence` | 0.0323 | 0.0439 |
| `rotation_0p5deg` | 0.0336 | 0.0441 |
| `rotation_2deg` | 0.0598 | 0.0714 |
| `shear_a2.5_c16` | 0.0001 | 0.0325 |
| `shear_a10_c16` | 0.0003 | 0.0194 |
| `shear_a20_c16` | 0.0157 | 0.0139 |
| `shear_a20_c32` | 0.0005 | 0.0057 |
| `shear_a20_c64` | 0.0002 | 0.0025 |
| `shear_snr10` | 0.0011 | 0.0237 |
| `shear_snr4` | 0.0573 | 0.0391 |
| `shear_snr2` | 0.0660 | 0.0732 |
| `shear_snr4_secondary` | 0.0214 | 0.0308 |
| `shear_decorr0.1` | 0.0043 | 0.0238 |
| `shear_decorr0.25` | 0.0617 | 0.0414 |
| `shear_decorr0.5` | 0.1768 | 0.1693 |
| `shear_uint8` | 0.0003 | 0.0194 |
| `shear_landsat` | 0.0319 | 0.0249 |
| `shear_landsat_uint8` | 0.0327 | 0.0256 |

The remaining chip/amplitude cells are in `results.json`.

What the numbers characterize is the algorithm's response to deformation, which all three share: error
rises with the deformation a chip spans, from exact on a pure translation to 0.0598 px at a 2° rotation
and 0.1768 px when half the texture decorrelates. Larger chips help on shear (0.0157 → 0.0002 from
chip 16 to 64 at the steepest amplitude) because the estimate averages more texture.

**uint8 quantization costs nothing measurable**: `shear_uint8` matches `shear_a10_c16` to four decimals
and `translate_subpixel_uint8` is exact, closing a gap every prior A/B run left open by pinning
`DataType = 1`.

**Rotation search does not help.** `rotation = RotationSearch()` over 19 deforming cases: 0 better, 4
worse, 15 unchanged. A rigid rotation of the chip cannot represent shear or dilation, and the
deformation is *within* the chip rather than of it.

## What this does not show

- **Nothing about real imagery.** `tools/ab/` covers that, and finds the same agreement there: the
  correlator bit-identical and the pipeline at a median of 0.0000 px. What this sweep adds is that the
  agreement holds against *known truth* rather than only between the two, across deformation modes,
  noise, decorrelation and quantization.
- **`py_pipeline` resolves nothing at chips 32 and 64 here.** Not an accuracy result: it is the
  window-size artifact `tools/ab/README.md` documents.
  `ChipSize0_GridSpacing_oversample_ratio = ChipSize0X / GridSpacingX` is 4 at chip 32 and 8 at chip
  64, so the coarse grid decimates to 7×7 and 3×3 while `DispFiltC.FiltWidth` becomes 17 and 33. The
  filter is wider than the grid, `CoarseCorValidFac` falls below `CoarseCorCutoff`, and `autorift()`
  hits `continue`. A 1024² scene is too small for those chips at this spacing.
- **Both pipelines lose coverage under heavy decorrelation** — 47.7% and 48.3% at `shear_decorr0.5`.
  That is the outlier filter doing its job on a scene half of which has no correspondence to find.
- **Absolute magnitudes are not field accuracy.** Single-level correlator runs on 1024² scenes at a
  fixed 20 px radius.
- **Preprocessing is bypassed.** Every scene is correlated as generated, so a filter difference cannot
  masquerade as a correlator difference — but the Wallis filter production uses is not exercised.
