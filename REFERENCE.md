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

There is no golden data anywhere: neither repository has a test that exercises the
correlator, and hyp3-autorift's one committed product has zero valid pixels
(`P000`) and is useful only as a schema reference.

Consequences for how AutoRIFT.jl is validated:

1. **Synthetic ground truth is the primary gate.** A texture displaced by a known
   amount has an exactly known answer, which is stronger than agreement with an
   implementation that has no test suite of its own.
2. **OpenCV fixtures pin the primitives** where OpenCV's semantics are the
   standard and are correct — border handling, resampling, the correlation
   surface, pyramid upsampling, peak tie-breaking. Generated on a current stack:
   v2.1.2 runs on NumPy ≥ 2.0 and Python ≥ 3.10, so the pinned-legacy-environment
   requirement that v1.5.0 imposed is gone.
3. **Whole-pipeline comparison is tolerance-based**, against output captured from
   the production container. Three things make exact comparison impossible
   regardless of implementation: an unseeded 10⁶-draw Monte Carlo in
   `v_error_cal`, wall-clock timestamps and version strings in global attributes,
   and a time coordinate jittered by Python's per-process-salted `hash()`
   (`crop.py::numeric_hash`, with `PYTHONHASHSEED` unset — documented as
   deterministic but not).
