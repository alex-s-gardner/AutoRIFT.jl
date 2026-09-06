# Reference implementation

AutoRIFT.jl reimplements the **production** autoRIFT, which is two pieces of
software rather than one:

| Piece | Version | Local path |
|---|---|---|
| Algorithm core (`autoRIFT.py` + `autoriftcoremodule.cpp`) | [nasa-jpl/autoRIFT@v2.1.2](https://github.com/nasa-jpl/autoRIFT/tree/v2.1.2) | `../autoRIFT-v2.1.2` |
| Production drivers (`testautoRIFT.py`, `testGeogrid.py`, `netcdf_output.py`) | [ASFHyP3/hyp3-autorift@develop](https://github.com/ASFHyP3/hyp3-autorift/tree/develop/src/hyp3_autorift/vend) `src/hyp3_autorift/vend/` | `../hyp3-autorift` |

ASF HyP3 is what actually generates ITS_LIVE products, and its vendored drivers
override upstream behaviour in ways that change results. Where the two disagree,
**HyP3 is the reference.**

Recreate the pinned sources with:

```bash
git -C ../autoRIFT worktree add --detach ../autoRIFT-v2.1.2 v2.1.2
git clone -b develop https://github.com/ASFHyP3/hyp3-autorift.git ../hyp3-autorift
```

`../autoRIFT` (v1.5.0) is retained only for archaeology; do not implement against
it. Several v1.5.0 behaviours documented as deliberate quirks turned out to be
bugs that v2.0.0 fixed.

---

## What v2.0.0 changed in the algorithm core

Recorded because the differences are large and the older version is the one most
documentation describes.

**One similarity measure, not two.** All four C++ entry points use
`CV_TM_CCOEFF_NORMED` — mean-removed normalized cross-correlation
(`autoriftcoremodule.cpp:171,263,385,478`). v1.5.0 used the DC-sensitive
`CV_TM_CCORR_NORMED` for float input, which was a bug; the changelog records the
fix. The `ChipI - minChipI` shift that worked around it survives as dead code:
unreachable for unsigned input, and a no-op for float now that the measure is
DC-invariant.

**The grid loop moved into C++.** `arImgDisp_u`/`arImgDisp_s` now make one call
with whole arrays instead of looping per point in Python, and the
`multiprocessing` machinery is gone. Threading is OpenMP `schedule(dynamic, 1)`
over grid rows. Each point writes a distinct output element with no reduction, so
**parallelism does not affect results** — a Julia port may thread the loop however
it likes at no fidelity cost. `MultiThread` remains as a class attribute but
nothing reads it; thread count comes from `OMP_NUM_THREADS`.

**The flat-chip and flat-reference guards were deleted and not reimplemented.**
v1.5.0 skipped any point whose chip was constant, leaving `NaN`. v2.1.2 passes it
to `matchTemplate`, whose normalization yields an all-zero surface, so the peak
lands at index `(0,0)` and the point is written as
`dx = -search_radius_x, dy = +search_radius_y` — a systematic corner-of-window
bias over masked and low-texture areas, where v1.5.0 correctly produced no
estimate. This also flips `M0C = ~isnan(DxC)` to `true` there, changing the
coarse-validity fraction and therefore which pyramid levels are skipped.

*AutoRIFT.jl treats a degenerate chip as no measurement* — a constant chip
carries no information about displacement, so reporting the search-window corner
as an answer is worse than reporting nothing. The reference's behaviour is
available for comparison via a compatibility flag.

**The per-point chip-size bounds do not apply at the base chip size.** The `M0`
gate — `(ChipSizeMinX <= ChipSizeUniX[i]) & (ChipSizeMaxX >= ChipSizeUniX[i])` —
sits inside `if self.ChipSize0X != ChipSizeUniX[i]` (`autoRIFT.py:509-539`), and
the `else` branch taken at the base chip size copies the search limits with no
bounds test at all (`:587-593`). So a point whose parameter file asks for no chip
smaller than 480 m is still correlated at the base chip size; the bounds restrict
only the coarser levels, where the mask is additionally dilated by a `6 / Scale`
maximum filter (`:534-539`).

The effect is large because the bounds cluster spatially: on the golden
Sentinel-2 case 136,800 points are answered at the base chip size against their
own `ChipSizeMinX` of 48, 96 or 192.

*AutoRIFT.jl matches this*, because agreeing with the reference is what makes a
real difference distinguishable from a bug. It is **not** obviously correct — the
finest level is where a chip smaller than the parameter file allows does the most
damage — and it is a candidate to revisit once the two agree, alongside the
degenerate-chip difference above.

**`colfilt` was rewritten** with numba reducers behind
`scipy.ndimage.generic_filter`, which changes three things:

- *Output shape.* Always the input shape. v1.5.0 returned `(H-1, W-1)` for even
  kernels, and those results are then resampled to a fixed target — so every
  non-base pyramid level's search limits, priors, and availability mask differ
  between versions. This is the largest silent numerical delta.
- *Output type.* Always `Float32`, never the input type. A `Bool` input now
  returns 0.0/1.0.
- *Boundary handling.* Per option: `reflect` for max/min/range, constant `NaN` for
  mean/median/MAD/agreement-count. For float input the two are equivalent (the
  reflected indices always land inside the window), but for the `Bool`
  availability mask the border genuinely changes: v1.5.0 padded a boolean array
  with `NaN`, which is truthy, so its border read as available.

Two `colfilt` quirks are worth stating precisely, because they invert between
versions. **Odd kernels are now chunk-invariant** — v1.5.0's chunk-seam
off-by-one is fixed. **Even kernels have a new and different seam defect:** the
code assumes a left margin of `(k-1)÷2` while `generic_filter` uses `k÷2`, so the
first output column of each chunk after the first reads padding where it should
read data, corrupting `nchunks - 1` columns per row. Only the non-base pyramid
levels use even kernels.

**Other fixes that change results:** variance is clipped at zero in
`_preprocess_filt_std`, so the float32 cancellation `NaN`s no longer propagate
into the validity mask; the nodata-fill mask combinator changed from `&` to `|`,
matching the non-fill path; the gap-fill RNG moved from the legacy global
MT19937 to a per-call `default_rng()` (PCG64), still unseeded and so still
irreproducible.

**Removed:** the `Flag` attribute and the grid-too-small early return.
`autoRIFT_ISCE.py` (a parameter-declaration wrapper with no algorithm).

**Unchanged**, and so still described accurately by v1.5.0 analyses: the pyramid
loop and all its resampling modes, the sparse-search decimation, the
distance-transform dilation, the hole filling, the smallest-chip-wins merge, the
`DISP_FILT` outlier test, the subpixel `pyrUp` cascade and every trap in it,
`preprocess_filt_hps`, `uniform_data_type`, and the chip/search-window geometry.

Still **no correlation-peak output**: `minMaxLoc`'s value is discarded at all four
call sites. AutoRIFT.jl returns it.

**Above the base chip size, a reported displacement is not the measured one.** For any
level where `ChipSizeUniX[i] != ChipSize0X` the reference decimates the field, mean- and
median-filters it, resizes it with `INTER_CUBIC` back to the full grid, and then writes
that interpolated value over *every* point the level owns — including the ones it
measured directly: `Dx[idxRaw | idxFill] = DxF[idxRaw | idxFill]`
(`autoRIFT.py:856-866`, and its own comment at `:811` says "replacing the valid
estimates with the bicubic filtered values"). Only the base level reports raw
measurements.

The consequence is worth stating because it looks like a defect in a comparison: a
coarse level's values are **not quantized to `1/upsampling`**, since a bicubic weighted
sum lands anywhere, so two implementations cannot agree exactly there however correct
both are. Measured against AutoRIFT.jl, which does the same thing: 0.01–0.03% of either
side's chip-48 values fall on any of the 1/16, 1/32, 1/64 or 1/128 grids, against 99.4%
at the base level. Bias is under 0.01 px. `tools/golden/README.md` has the numbers.

**The subpixel upsampling factor varies per chip size.** `OverSampleRatio` may be a
scalar or a dict, and when it is a dict the factor is looked up per level —
`overSampleRatio = self.OverSampleRatio[ChipSizeUniX[i]]`
(`autoRIFT.py:652-653`, and again at the fine-search call site). The production
driver always passes a dict, keyed by the four chip sizes `ChipSize0X * [1,2,4,8]`:
`{16, 32, 64, 64}` for optical and `{32, 64, 128, 128}` for radar
(`testautoRIFT.py:488-510`). So a level's displacement is quantized to `1/16` px at
the base chip size but `1/32` and `1/64` at the coarser ones, and the quantization
step is a property of the level rather than of the run.

*AutoRIFT.jl applies one `upsampling` to every level*, since `PyramidRefine` holds a
single factor. On a golden comparison that shows up sharply: at matched chip sizes
the base level agrees exactly on half its points, while the coarser levels agree on
**none** of them, because the two sides are rounding to different grids. Supporting
a per-level factor is the fix; `tools/golden/README.md` records the measurement.

---

## What the production drivers change

The HyP3 drivers are forked from upstream's v2.0.0-era scripts and were never
re-synced; `vend/CHANGES.diff` documents the intent of the older overrides but has
drifted and omits several later additions. Read the vendored `.py` files, not the
diff.

**Only `preprocess_filt_hps` ever runs inside autoRIFT.** The Wallis and FFT
filters are commented out in the driver
(`vend/testautoRIFT.py:421-448`) and performed *outside*, before Geogrid, on
native-projection imagery (`process.py::apply_landsat_filtering`). So the
production order is **filter → reproject → geogrid → correlate**, not autoRIFT's
internal order. Landsat 4/5 get the FFT destripe; Landsat 7 and 8 get the Wallis
gap fill; everything else gets the high-pass inside the correlator.

**Input is always real-valued.** Production feeds detected amplitude GeoTIFFs;
ISCE3 does the detection upstream, so the correlator never sees complex SLC data.
Complex support is therefore a capability AutoRIFT.jl may add, not a requirement
it must match.

**Acquisition times come from STAC metadata with sub-day precision**, not from
filenames parsed to midnight. This propagates into every modelled error estimate.

**`mpflag = 0` unconditionally**, so the reference's own multi-threaded path is
never exercised in production.

**The dt-varying search-range scaling is disabled** — commented out in both
Geogrid branches.

**A stale `autoRIFT_intermediate.nc` in the working directory silently skips
correlation entirely** (`vend/testautoRIFT.py:693-706`). Worth knowing when
comparing against production output.

---

## Validation

Neither repository has a test that exercises the correlator, and hyp3-autorift's one
committed product has zero valid pixels (`P000`), useful only as a schema reference.

**Golden data does exist, outside both repositories.**
`s3://its-live-data/test-space/golden/` holds 22 ITS_LIVE products built by
`hyp3_autorift` 0.28.4 across Landsat 4–9, Sentinel-1/2 and NISAR — the acceptance
set for the Python implementation, publicly readable. `tools/golden/` compares
against them, and the container that produced them runs locally, which makes the
whole production chain reproducible rather than merely readable.

That comparison is **exact wherever the reference is deterministic**, which is every case
whose preprocessing is `hps`. Two runs on a Sentinel-2 granule agree bit for bit on every
plane, and so does a local run against ASF's product; only the `time` coordinate moves,
by `crop.py::numeric_hash` under a per-process hash salt.

**Any pair with a `wallis_fill` scene is not deterministic**, and the reference does not
reproduce itself there. `_wallis_filter_fill` fills Landsat 7's Scan Line Corrector gaps
with `rng.normal` from an **unseeded** `np.random.default_rng()` (`autoRIFT.py:113-125`).

Seeding that one call and changing nothing else makes two runs **bit-identical** — 100.0%
of `vx`, zero coverage difference, every attribute equal — where unseeded runs agree on
3.4%. So the draw is demonstrably the cause, not a coincidence of correlation.

The path it takes is worth stating, because the obvious one is wrong. The fill is *not*
retained as a measurement: it is white noise against real texture, correlates with nothing,
and `filtDisp` rejects it as designed. The effect is in the **rejection**. `filtDisp` is a
neighbourhood test — a point survives only if `FracValid * FiltWidth²` of its neighbours
agree with it (`autoRIFT.py:1600-1626`) — so a rejected fill point is a missing neighbour
for every real point whose window overlaps it, and a different draw rejects a different
set. Coverage therefore swings by ±3,500 measured points between runs in both directions,
and only 9.1% of the disagreeing points are flagged `interp_mask` against 5.8% of the
agreeing ones. Real points are losing and gaining neighbourhood support, not reporting
noise.

The filter is chosen **per scene**, not per pair — `wallis_fill` for `L[EO]07_`, `fft` for
`LT0[45]_`, `hps` otherwise (`vend/testautoRIFT.py:718-723`) — and **one** unseeded scene
is enough to make a pair irreproducible. Measured across all nine optical golden cases,
the split is exact with no exceptions: all five `['hps','hps']` pairs reproduce ASF's
product bit for bit, and all four with at least one `wallis_fill` scene do not, including
a Landsat 8 reference against a Landsat 7 secondary.

On `LE07_L1TP_061018_20120428`, two runs agree on **3.4%** of `vx` at a median difference
of 9 m/yr and a p95 of 34, and coverage moves enough to change the product's own `P<nn>`
name. A local run differs from ASF's golden product by statistically the same amount, so
**that golden product is one draw from a distribution rather than a fixed target**. Those
cases have to be gated against the reference's own run-to-run envelope, measured by
running the container twice.

AutoRIFT.jl's `WallisGapfill` seeds its generator and so is reproducible; that is a
deliberate difference, and it means AutoRIFT.jl cannot match any single draw exactly.

Consequences for how AutoRIFT.jl is validated:

1. **Synthetic ground truth is the primary gate.** A texture displaced by a known
   amount has an exactly known answer, which is stronger than agreement with an
   implementation that has no test suite of its own.
2. **OpenCV fixtures pin the primitives** where OpenCV's semantics are the
   standard and are correct — border handling, resampling, the correlation
   surface, pyramid upsampling, peak tie-breaking. Generated on a current stack:
   v2.1.2 runs on NumPy ≥ 2.0 and Python ≥ 3.10, so the pinned-legacy-environment
   requirement that v1.5.0 imposed is gone.
3. **Whole-pipeline comparison is against the golden products**, and is exact on
   everything except the time coordinate and the wall-clock and version strings in
   the global attributes. The unseeded 10⁶-draw Monte Carlo in `v_error_cal` does
   *not* prevent exact comparison: the standard deviation of that many draws is
   stable well inside the `int16` rounding the product applies, so the sampling
   noise does not reach the file. `tools/golden/README.md` has the measurements.
