# The golden tests: AutoRIFT.jl against ITS_LIVE production output

`s3://its-live-data/test-space/golden/` holds the acceptance products for the Python autoRIFT — 22
ITS_LIVE granules built by `hyp3_autorift` 0.28.4 from the job list in
[`autorift_golden.json.j2`](https://github.com/ASFHyP3/hyp3-testing/blob/develop/hyp3_testing/templates/autorift_golden.json.j2),
publicly readable, each with `ncdump`, STAC, browse and metadata sidecars. Matching them is what
makes AutoRIFT.jl production-ready.

This is a larger target than the correlator. A golden `.nc` is `vx`/`vy`/`v`/`v_error` in m/yr as
`int16`, plus chip sizes, an interpolation mask, a CRS variable, an image-pair metadata variable and
fifteen global attributes, on a 120 m grid cropped to the valid-data extent. Producing one means
running the whole production chain:

    filter → reproject → geogrid → correlate → displacement-to-velocity → stable-shift
    correction → error estimation → crop → netCDF packaging

AutoRIFT.jl owns `correlate`, part of `filter`, and the grid handoff into `geogrid`. Everything after
`correlate` is not yet implemented in Julia, so most of the work here is downstream of the part that
already agrees bit-for-bit with the reference (`tools/ab/README.md`).

## The 22 cases

`manifest.json` is generated from the Jinja job template and committed. It is a lookup rather than a
naming rule, for two reasons that a derived mapping gets wrong:

- **A product is named in acquisition order**, `<img1>_X_<img2>_G0120V02_P<nn>`, which is not the
  job's reference/secondary order. Two of the 22 jobs are reversed relative to their product, and
  `img_pair_info:id_img1` follows the product. So `reference` is not necessarily `img1`.
- **A scene can appear in more than one pair.** `LC08_L1TP_060018_20130330_20200912_02_T1` is in two,
  so matching on one scene name or on a date is ambiguous.

Burst jobs are the exception: a product never names its bursts, so the first reference burst's
timestamp identifies `img1`.

| phase | cases | platforms | blocked on |
|---|---|---|---|
| 3 | 9 | Landsat 7/8/9, Sentinel-2 | the post-correlation chain |
| 4 | 3 | Landsat 4/5 | an FFT destripe filter |
| 5 | 8 | Sentinel-1 SLC and OPERA bursts | the radar geogrid path and ISCE3 detection |
| 6 | 2 | NISAR L1 RSLC, L2 GSLC | as phase 5 |

`P<nn>` in a product name is `roi_valid_percentage` truncated — valid pixels within the ROI, not
within the grid. The grid-wide fraction is lower, 5–42% across the set, so the two are not
comparable and a reader that checks one against the other will see a mismatch that is not one.

## Three product schemas

A reader that assumes one schema mis-reads the others:

| schema | variables | axes | cases |
|---|---|---|---|
| optical | 12 | `(x, y, time)` | Landsat, Sentinel-2, **and NISAR L2 GSLC** |
| radar | 16 | `(x, y, time)` | Sentinel-1, NISAR L1 RSLC — adds `vr`, `va`, `M11`, `M12` |
| uncropped | 11 | `(x, y)` | the one `P000` case |

NISAR L2 GSLC is a radar sensor with the optical schema, because a geocoded product measures
displacement on a map grid rather than in range and azimuth. So `product.jl` reads the schema from
the variables present, not from the platform.

The `P000` case is not a degenerate input to skip. `process.py` crops only products with at least one
valid pixel, and cropping is what adds the time axis — so this is the one case exercising the
uncropped path, and its missing time coordinate is the observable difference.

## Getting the data

```bash
julia --project=tools/golden -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=tools/golden tools/golden/fetch.jl              # 22 products + sidecars, 140 MiB
julia --project=tools/golden tools/golden/fetch.jl --check      # can inputs be reached? no downloads
julia --project=tools/golden tools/golden/run.jl --status       # what is cached
```

Data lives outside the repository, under `~/data/autorift/tests/golden_tests` — override with
`AUTORIFT_GOLDEN_CACHE`.

Sidecars use two naming conventions, which is why they are two lists in `manifest.jl`: `.head`,
`.premet` and `.spatial` hang off the *file* (`<product>.nc.head`), while `.stac.json` hangs off the
*product* (`<product>.stac.json`).

### Input access

The golden products and the parameter shapefile are anonymous. The scenes they were made from are
not, and reach four services with three different credentials, so `fetch.jl --check` resolves them
without downloading — which keeps a credential failure distinguishable from an absent granule.

| source | cases | route | state |
|---|---|---|---|
| Sentinel-2 L1C | 2 | Google Cloud `.SAFE` mirror | anonymous |
| Sentinel-1 SLC and bursts | 8 | CMR + ASF, `~/.netrc` | works |
| NISAR RSLC/GSLC | 2 | CMR + ASF, `~/.netrc` | works |
| Landsat C2 L1 | 9 | requester-pays S3, via `AWS_PROFILE` | works |

**Landsat is not in CMR.** The `Landsat Level-1 Collection 2` collection is registered
(`C3442493460-USGS_EROS`) but indexes **zero granules**; its only data link points at the landsatlook
STAC API. So EarthData.jl reaches Sentinel-1 and NISAR but cannot reach Landsat, and the routes are:

1. **USGS M2M API** — username plus an *application token* from
   <https://ers.cr.usgs.gov/password/appgenerate>. Downloading additionally requires the M2M role on
   the account: without it `login-token` succeeds, `permissions` returns `['user']` rather than
   including `download`, and `download-options` answers HTTP 403.
2. **Requester-pays S3** — `s3://usgs-landsat` with `--request-payer requester`. The only route the
   *container* can use unmodified, since `process.py` builds `/vsis3/` paths.

`~/.netrc` does not authenticate landsatlook: it redirects to `ers.cr.usgs.gov`, whose login is a
CSRF-token form POST that ignores HTTP basic auth.

Which band matters, from `process.py:77-89`: Landsat 4/5 correlate **B2 (green)**, and Landsat 7/8/9
**B8 (panchromatic)**.

## The harness gates on itself first

```bash
julia --project=tools/golden tools/golden/selftest.jl
```

Every product is read and diffed against itself: **22/22 identical**, on every variable, coordinate
and attribute. That proves the reader is deterministic, but a comparator that always reported
agreement would pass it too — so the selftest also injects one difference of each kind that must be
caught, and asserts each is:

| injected | detected as |
|---|---|
| one pixel changed by 3 | `max_abs = 3.0`, not identical |
| one pixel set to nodata | a coverage difference, `only_a = 1` |
| a plane transposed | `DimensionMismatch` |
| `vx:stable_shift` changed | an attribute difference |
| one `x` coordinate shifted | not identical |

The transpose case runs on a non-square product deliberately. `tools/ab/README.md` records that a
transposed displacement field still reads as a displacement field; on a square grid its shape gives
nothing away, and detecting one there needs values checked instead.

## What the comparison does, and does not, tolerate

Exact equality is the default, and a tolerance is a claim requiring a measurement. Nodata is
compared as a *value*, not skipped: where one product measured a pixel and the other did not, that is
a coverage disagreement, which is often more informative than a value one because it points at the
degenerate-chip and stable-shift divergences rather than at arithmetic.

Attributes are compared alongside the pixels. `stable_shift`, the four `error` estimates and
`stable_count` carry as much of the answer as the planes do — a product whose `vx` matched and whose
`stable_shift` did not has not matched.

Two fields cannot match between two runs of the reference *itself*, and are excluded rather than
tolerated. Both are properties of the reference, not of either implementation:

- **the `time` coordinate** — `crop.py::numeric_hash` jitters it by `hash(filename) % 10⁶`
  microseconds, and with `PYTHONHASHSEED` unset Python salts `hash()` per process. Documented as
  deterministic; is not. Measured: two runs land 0.25 s apart, and the amount is recorded in the
  product's own `time:microseconds_added` attribute.
- **`date_created`** — wall clock.

**`v_error` is not a third**, though `netcdf_output.py::v_error_cal` does draw 10⁶ samples from an
unseeded `default_rng()`. Measured on the S2 case: 8,356 pixels have `v == 0` and so take the Monte
Carlo value, and it is `27` in both runs and in golden. The standard deviation of 10⁶ draws is stable
to well within the `int16` rounding the product applies, so the sampling noise does not survive into
the file. The draw is still irreproducible; the *product* is not affected by it.

## The reference container

`ghcr.io/asfhyp3/hyp3-autorift:0.28.4` is the version named in every golden product's `source`
attribute, and an arm64 manifest exists, so the chain that produced the golden data runs locally
unmodified.

```bash
docker pull --platform linux/arm64 ghcr.io/asfhyp3/hyp3-autorift:0.28.4
julia --project=tools/golden tools/golden/run.jl --reproducibility S2B_MSIL1C_20200612
```

Run twice on one granule, the two products differ only in the `time` jitter. That is the
**reproducibility floor**, and measured on `S2B_MSIL1C_20200612` it is **exact**:

| comparison | planes | pixels | exact | differing |
|---|---|---|---|---|
| container run 1 vs run 2 | 7/7 | 615,146 | **100%** | `time` only, 0.254 s |
| container run 1 vs **golden** | 7/7 | 615,146 | **100%** | `time` only, 0.482 s |

The second row is the significant one: this machine reproduces ASF's golden product **bit for bit**,
on `vx`, `vy`, `v`, `v_error`, `chip_size_width`, `chip_size_height` and `interp_mask`, with every
coordinate and every other attribute equal — across a different architecture (arm64 here) and four
months. Both `.nc` files are also the same size to the byte.

So on this case there is no measurement noise to hide behind. **Exact equality is the gate**, and any
difference AutoRIFT.jl shows is a difference in AutoRIFT.jl. That is a considerably harder target
than a tolerance table, and a much more useful one: a tolerance wide enough to absorb a rounding
difference is also wide enough to absorb a bug.

### The whole optical phase, and the one thing that decides reproducibility

All nine phase-3 cases, run twice each. `floor` is run 1 against run 2, `golden` is run 1 against
ASF's product, both on `vx`; `coverage` is how many points one run measured and the other did not.

| case | filters | floor | golden | coverage |
|---|---|---:|---:|---:|
| `LC08_L1TP_009011_20200703` | `hps`, `hps` | **100%** | **100%** | 0 |
| `LC08_L1TP_062018_20200823` | `hps`, `hps` | **100%** | **100%** | 0 |
| `LC09_L1GT_215109_20220125` | `hps`, `hps` | **100%** | **100%** | 0 |
| `S2A_MSIL1C_20200626T204021` | `hps`, `hps` | **100%** | **100%** | 0 |
| `S2B_MSIL1C_20200612T150759` | `hps`, `hps` | **100%** | **100%** | 0 |
| `LE07_L1TP_061018_20130314` | `wallis_fill`, `hps` | 36.2% | 35.9% | 50,350 |
| `LE07_L1TP_063018_20040810` | `wallis_fill`, `wallis_fill` | 29.1% | 25.7% | 109,540 |
| `LC08_L1TP_060018_20130330` | `hps`, `wallis_fill` | 0.5% | 11.9% | 60,200 |
| `LE07_L1TP_061018_20120428` | `wallis_fill`, `wallis_fill` | 3.4% | 3.3% | 30,143 |

**The split is perfect and has one cause.** Every `['hps','hps']` case reproduces exactly — five of
nine, across two Landsat sensors and both Sentinel-2 satellites, on tiles in Greenland, Alaska and the
Antarctic Peninsula. Every case with **at least one** `wallis_fill` scene does not, and in each of
those the `floor` and `golden` columns are close to each other: *a local run differs from ASF's
product by about as much as two local runs differ from each other.*

`_wallis_filter_fill` (`autoRIFT.py:113-125`) fills Landsat 7's Scan Line Corrector gaps with
`rng.normal` from an **unseeded** `np.random.default_rng()`. The driver picks the filter **per scene**,
not per pair (`testautoRIFT.py:718-723`), which is why `LC08_L1TP_060018_20130330` is affected: it is
a Landsat 8 reference against a Landsat 7 secondary, and one unseeded scene is enough.

#### That the draw is the cause is a measurement, not an inference

Patching `numpy.random.default_rng` to seed itself and changing nothing else (`seeded.py`) settles it:

| two runs of the reference | `vx` exact | coverage Δ | whole product |
|---|---:|---:|---|
| unseeded, as shipped | 3.4% | 30,143 | differs |
| **seeded, same seed** | **100.000%** | **0** | **identical** |

Same code, same inputs, same container; one call seeded. So the gap-fill noise does reach the output.

**But not by being retained as a measurement** — that part of the mechanism works as designed. The
fill is white noise against real texture, so it correlates with nothing, and `filtDisp` rejects it.
The effect is *indirect*, and it is in the rejection rather than the acceptance: `filtDisp` is a
neighbourhood test, where a point survives only if at least `FracValid * FiltWidth²` of its neighbours
agree with it (`autoRIFT.py:1600-1626`). A rejected fill point is a **missing neighbour** for every
real point whose window overlaps it, and a different draw rejects a different set — so different real
points clear the agreement threshold.

The signature matches: coverage swings by ±3,500 measured points between runs, in **both** directions
(seeded measures 3,504 more than one unseeded run and 1,371 fewer than the other), and of the 122,945
disagreeing points only 9.1% are flagged `interp_mask` — barely above the 5.8% among the agreeing
ones. So this is not the fill being reported as signal; it is real points losing or gaining the
neighbourhood support they need.

Measured in physical units on `LE07_L1TP_061018_20120428`: two runs land 9 m/yr apart at the median
and 34 at p95, and coverage moves enough to change the product's own `P<nn>` name — `P010`, `P011`,
`P010` across three runs — carrying `stable_shift`, `stable_count` and all four `error` attributes
with it.

So the gate has two tiers, decided by the filters a case uses rather than by its platform:

- **`hps` only** — gate on **exact equality**. Five of nine optical cases, and every S2, L8 and L9 pair
  whose partner is not L7.
- **any `wallis_fill`** — gate on the reference's own run-to-run envelope, measured per case by
  running the container twice. The golden product is one draw from a distribution, not a target.

`REFERENCE.md` already recorded that this RNG is unseeded and that AutoRIFT.jl's `WallisGapfill` is
seeded and therefore reproducible. What this measures is the consequence, and that it is per-scene.

Working directories are kept, not cleaned, under `runs/<product>/<n>/`. They hold the filtered
scenes, the geogrid rasters, and `autoRIFT_intermediate.nc` — `Dx`, `Dy`, `InterpMask`, `ChipSizeX`,
`SearchLimitX/Y`, `noDataMask`, which is the correlator's output before conversion to velocity.
Keeping it is what makes a later product disagreement attributable to the correlator or to the
packaging, rather than to one by elimination.

Note that a stale `autoRIFT_intermediate.nc` in the working directory makes `testautoRIFT.py` skip
correlation entirely (`vend/testautoRIFT.py:693-706`), so a reused directory does not re-correlate.

## What a run leaves behind

A full S2 production run takes about three minutes on 8 threads and writes, besides the product:

| file | what it is |
|---|---|
| `autoRIFT_intermediate.nc` | `Dx`, `Dy`, `InterpMask`, `ChipSizeX`, `SearchLimitX/Y`, `noDataMask` |
| `offset.tif`, `velocity.tif` | displacement and velocity as rasters |
| `window_*.tif` (9 files) | the geogrid: location, search range, chip bounds, the two off2vel vectors, scale factors, stable-surface mask |

For the S2 case the intermediate is a 1008² grid at spacing 12 with chips 24/48/96, against an
`origSize` of 1009² — the reference truncates its grid by one point, which `tools/ab/README.md`
records as costing it 1.6% of the points on a different scene.

That set is what makes Phase 1 possible without re-deriving anything: the same filtered inputs, the
same geogrid, and the reference's own `Dx`/`Dy` to diff AutoRIFT.jl's against directly.

## The correlator, on production imagery

```bash
julia --project=tools/golden -t 8 tools/golden/correlator.jl S2B_MSIL1C_20200612
```

A diagnostic rather than the gate: when the product comparison disagrees, this says whether the
correlator or the packaging is responsible. Both sides get the *same* arrays — the filtered pair,
grid, priors and per-point limits `capture.py` took at the reference's own `runAutorift` boundary — so
a preprocessing difference cannot appear here as a correlator difference.

S2 case, 10980² `UInt8` pair, 1,018,081 grid points, chips 24/48/96 at spacing 12, 139 s on 8 threads:

| axis | sign | both measured | only jl | only ref | exact | median | p99 | corr |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|
| `dx` | + | 598,718 | 16,893 | 19,162 | **67.6%** | 0.079 | 1.19 | **+0.9965** |
| `dy` | − | 598,718 | 16,893 | 19,162 | **69.1%** | 0.072 | 1.04 | **+0.9942** |

Bias is under 0.02 px on both axes and the median disagreement is under one upsampling step, so the
two agree about position. Two harness bugs and one package bug had to be found first, and each was
invisible in the summary statistics that preceded it:

| what was wrong | exact `dx` |
|---|---:|
| first measurement | 27.4% |
| the base level ignores per-point chip-size bounds (`autoRIFT.py:509` vs `:587`) | 38.4% |
| the captured grid needs `+1`, not `+0.5` — a half-pixel offset | **67.6%** |

**Plot before reasoning.** The half-pixel error showed a median `|ddx|` of exactly 1/16 — one
quantization step, entirely plausible as tie-breaking — and was found only by looking at
`tools/golden/residual_maps.jl`: the difference map was blank over flat ice and washed along every
fast-flow margin. That shape *is* the signature, because a grid offset produces no residual under
uniform motion and one proportional to the local velocity gradient. Binning residual against gradient
confirmed it, with exact agreement falling from 66.5% in the flattest decile to 5.6% in the steepest,
and scanning the offset settled the value: 85.0% at `+1.0` against 49.2% at `+0.5` on the base level.

The independent check that the correlator itself was never at fault: `tools/ab` stage 1 still reports
it **bit-identical** to `arImgDisp_s`, 100% exact at chip 32 on a Landsat 8/9 pair. Any golden
disagreement above that floor is the harness or the pipeline around the correlator, and looking there
first would have saved two rounds.

### Every `hps` case against the reference

All five deterministic optical cases, on the reference's own captured inputs, `dx` axis:

| case | both | only ref | exact | within step | median | p99 | level | corr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S2A Malaspina | 586,087 | 10,531 | **92.67%** | **97.66%** | 0.0000 | **0.1250** | **99.77%** | +0.997 |
| LC08 East Greenland | 690,929 | 35,638 | 63.24% | 76.59% | 0.0000 | 0.7500 | 98.96% | +0.985 |
| S2B Jakobshavn | 599,910 | 17,970 | 67.92% | 82.40% | 0.0000 | 0.6698 | 97.60% | +0.997 |
| LC09 Antarctic peninsula | 443,677 | 27,088 | 58.81% | 75.27% | 0.0000 | 0.5625 | 97.29% | +0.951 |
| LC08 Jakobshavn | 1,659,423 | 56,637 | 55.14% | 72.67% | 0.0000 | 0.9375 | 96.09% | +0.996 |
| *L8/L9 benchmark (the gate)* | *87,814* | — | *77.40%* | *97.30%* | *0.0000* | *0.1411* | *99.30%* | — |

**S2A Malaspina passes the gate outright** — 92.67% exact against 77.4%, p99 0.125 against 0.1411,
level agreement 99.77% against 99.3%. That is the existence proof that the harness, the conventions and
the correlator chain reach benchmark quality on a production scene.

The ordering across cases tracks **level agreement** almost monotonically, and far more tightly than it
tracks scene size or sensor. Level agreement is therefore the lever, which is where the residual tail's
enrichment also points.

### The half-pixel grid convention, twice

The LC08 Jakobshavn row above is the *second* measurement of that case. The first read 22.73% exact with
a median residual of exactly one quantization step, and it was wrong: its capture had been taken before
`capture.py` learned to dump inputs *after* `runAutorift` rewrites the grid, so it held the raw `Int32`
grid rather than the `round(xGrid) + 0.5` the correlator actually uses.

Correcting it moved the case from 22.73% to **55.14%**, and the base level from 34.08% to **77.37%** —
the benchmark's own figure. The median went from 1/16 to exactly 0, which is the offset disappearing.

Three things about this are worth keeping, because each cost real time:

**The fractional part of the captured grid identifies the fault immediately.** A correct capture is
`Float32` with every non-zero coordinate at `x.5`; the broken one was `Int32` at `x.0`. `in == out`
shape is the corroborating check, since a pre-rewrite capture carries `origSize` while the correlator
runs on the chopped `rlim`/`clim`. `pointset_from_capture` now *errors* on a non-half-integer grid.

**A forced re-capture silently returned the same arrays.** A stale `autoRIFT_intermediate.nc` makes the
driver skip correlation entirely (`testautoRIFT.py:693-706`), so `runAutorift` is never called and the
patch has nothing to intercept — while the run writes a fresh product, a fresh log and exits 0.
`capture_reference` now removes that file and requires `call1.json` to be newer than the run, because
the patch reporting that it was *installed* does not establish that it *fired*.

**Statistics could not see it and a heatmap could.** A half-pixel offset produces zero residual under
uniform motion and residual proportional to the local velocity gradient, so the median stayed at one
step — readable as tie-breaking — while the difference map showed red/blue dipoles along every fast-flow
margin, scaling with the gradient. That is the same signature, on the same case, that the earlier
half-pixel bug wore.

### Finding: the reference varies upsampling per chip size

`autoRIFT.py:652-653`: `OverSampleRatio` may be a dict, and when it is, the factor is looked up per
level. The driver always passes one — `{24: 16, 48: 32, 96: 64, 192: 64}` here — so the reference
quantizes to 1/16 px at the base chip size and 1/32 and 1/64 at the coarser ones. `PyramidRefine`
held a single factor, so AutoRIFT.jl quantized every level to 1/16.

Fixed: `Params.subpixel` is now a tuple, one method per level, on the same rule `similarity` uses.
`REFERENCE.md` records the reference behaviour.

**It was not the dominant cause.** With the reference's own ladder in place, exact agreement moved
from 27.42% to 27.43% — the plumbing is verified (`subpixel_at` returns 16/32/64 across the three
levels) and the effect is real but small. The fix stands on its own: without it a production
configuration cannot be expressed at all.

### What the base level and the coarse levels each say

Per chip size, over points where both chose that level and the reference did not interpolate:

| chip | step | points | exact | within one step | on the step grid | median |
|---|---|---:|---:|---:|---:|---:|
| 24 | 1/16 | 319,521 | **49.96%** | 84.1% | **99.4%** | 0.031 |
| 48 | 1/32 | 67,606 | 0.00% | 19.2% | **0.01%** | 0.096 |
| 96 | 1/64 | 16,541 | 0.00% | 10.2% | **0.04%** | 0.089 |

The base level behaves as a quantized comparison should: 99.4% of its residuals are exact multiples
of the step, and half are zero. What remains there is tie-breaking at 1/16 px, which no two
implementations can agree about — below a real peak both are choosing from noise. Residuals are
symmetric about zero, median signed difference exactly 0 with the ±1/16 tails within 1% of each
other, so it is not bias.

### The coarse levels are not a quantized comparison at all

"0% exact" at chips 48 and 96 reads like a failure and is not one. Four measurements, in the order
they rule things out:

**Neither side's coarse values sit on any quantization grid.** Multiples of 1/16, 1/32, 1/64 and
1/128 account for 0.01–0.03% of the reference's own chip-48 values and the same of AutoRIFT.jl's. A
level whose values are not quantized cannot agree *exactly* except by coincidence, so `exact` is the
wrong statistic above the base level — unlike chip 24, where 99.4% of values are on the 1/16 grid.

**Because both sides replace the measurement with an interpolated value.** `autoRIFT.py:811` says so
in its own comment — "replacing the valid estimates with the bicubic filtered values for robust and
accurate estimation" — and `:856-866` does it: `DxF` is `cv2.resize(..., INTER_CUBIC)` of the
decimated field, and `Dx[idxRaw | idxFill] = DxF[idxRaw | idxFill]` overwrites even the
directly-measured points. `_undecimate_level` does the same. A bicubic weighted sum lands anywhere.

**It is not a lattice or interpolant difference.** At a coarse *node* — a fine point coinciding with a
coarse sample — Catmull-Rom reproduces its sample exactly, so a lattice error would show up as nodes
agreeing much better than off-nodes. They agree *identically*: median 0.0980 on nodes against 0.0980
off them at chip 48. The interpolant is pinned bit-exact by 12 `INTER_CUBIC` fixtures in
`test/fixtures/resize/`, and for this case the reference's `int(shape/Scale)` ratio is exactly 2 and
4, so the lattice-drift trap `src/multichip.jl` documents is not active either. One level run in
isolation through `chipsize_level`, with no merge and no prior, reproduces the same 0.00% — so it is
not the merge.

**What remains is small and unbiased:**

| level | axis | points | bias | median | within 0.1 px | p95 | corr |
|---|---|---:|---:|---:|---:|---:|---:|
| 48 | `dx` | 80,122 | +0.0009 | 0.098 | 50.7% | 0.459 | +0.972 |
| 48 | `dy` | 80,122 | +0.0031 | 0.098 | 50.8% | 0.428 | +0.958 |
| 96 | `dx` | 20,229 | −0.0015 | 0.094 | 52.5% | 0.466 | +0.921 |
| 96 | `dy` | 20,229 | +0.0079 | 0.104 | 48.8% | 0.542 | +0.978 |

Bias under 0.01 px on both axes at both levels, and half the points within a tenth of a pixel. The
residual is the accumulated difference between two independent implementations of a chain that
decimates, median-filters, area-resizes a prior, hole-fills and bicubic-resizes — each step matched in
kind but not, at Float32, in the last bits. Exact agreement above the base level is not a reachable
target for that chain, so **bias and the within-one-step fraction are the headline there, not
`exact`**.

## Measured results

`results/<product>.<kind>.json` holds each comparison with the machine, versions and commit that
produced it.

| what | state |
|---|---|
| harness self-diff, 22 products | **22/22 identical** |
| injected-fault detection, 5 kinds | **5/5 caught** |
| container vs ASF golden, all 9 optical cases | **5/5 `hps` cases exact**; the 4 with a `wallis_fill` scene cannot be |
| reference reproducibility floor, `hps` | **exact** — every plane, every pixel, `time` only |
| reference reproducibility floor, `wallis_fill` | ±9 m/yr median on L7; golden is one draw from it |
| correlator vs reference `Dx`/`Dy`, all 5 `hps` cases | **S2A Malaspina passes the gate** (92.7% exact, p99 0.125, level 99.8%); the rest between 55.1% and 67.9% |
| correlator vs `arImgDisp_s` (`tools/ab` stage 1) | **bit-identical**, 100% exact at chip 32 |
| whole pipeline vs reference (`tools/ab` stage 2) | **81.8% exact**, 98.7% within one step, p99 0.0752 px — the pyramid is not bit-identical on either harness, only the single-level correlator is |
| AutoRIFT.jl product against golden | needs the post-correlation chain (phase 2) |

### Open, in priority order

1. **The remaining 32%, which is mostly the coarse levels.** Decomposed on the aligned grid:

   | population | points | exact | within 1/16 |
   |---|---:|---:|---:|
   | all both-measured | 598,718 | 67.6% | 82.3% |
   | same chip level | 582,010 | 69.5% | 83.8% |
   | same level, neither side filled | 533,535 | 72.5% | 85.9% |
   | **base level, neither filled** | 450,367 | **85.9%** | **94.5%** |

   The base level is close to the `tools/ab` result and the shortfall is concentrated above it, where
   the reference overwrites measurements with a bicubic resize and exact agreement is unreachable by
   construction (below). Level agreement is now 97.2%. What is left to chase, in order: **level
   agreement**, which carries a 6.55x enrichment in the residual tail and is the one part of that tail
   not explained by the coarse-level bicubic; then the 14% of base-level points beyond one step. Hole
   filling is done, and the outlier filter's parameters and reducers are ruled out.
2. **Coverage, and it is the outlier filter.** 16,893 points AutoRIFT.jl answers alone against 19,162
   the reference does. Disabling the filter and changing nothing else settles which side owns it:

   | outlier filter | both measured | only jl | only ref |
   |---|---:|---:|---:|
   | reference-matched `GardnerFilter` | 598,718 | 16,893 | **19,162** |
   | `NoOutlierFilter` | 617,208 | 186,408 | **672** |

   `only_ref` collapses from 19,162 to 672, and `both` rises by exactly the 18,490 difference. So
   **almost every point the reference answers alone is one AutoRIFT.jl measured and then rejected** —
   the filter is over-rejecting relative to the reference, rather than the correlator failing to
   measure. (The complementary 186,408 confirms the filter is doing real work and must not simply be
   loosened.)

   This also **corrects a claim previously recorded here**: the deliberate degenerate-chip difference
   accounts for at most those 672 residual points, not a substantial share of the gap.

   The cause is not the parameters, which were checked against the reference's own derivation from the
   captured scalars and match exactly — `ChipSize0X/GridSpacingX = 2`, so `FiltWidth` 9 and
   `FracValid` 0.41 fine / 0.32 coarse on both sides, with `agree_tolerance` 0.2, `mad_scale` 4 and
   3/2 iterations. `rescale`/`relax` reproduce `autoRIFT.py:484-505` exactly. The remaining suspects
   are therefore *when* the filter runs relative to the level merge and hole filling, and the
   `windowmedmad` / `count_agreeing` reducer semantics at window borders and on all-NaN
   neighbourhoods.

   One structural difference is already visible and is a candidate rather than a conclusion:
   `_oversample` is **capped at 2** (`src/multichip.jl:697`) where the reference's ratio is uncapped.
   It does not bind on this case — the ratio *is* 2 — so it cannot explain these numbers, but it will
   bind wherever grid spacing divides the chip size more than twice.

   **The disagreement is bidirectional and co-located, which changes what it is.** Mapping the two
   exclusive sets side by side, they are the *same picture*: both are speckle along the same feature
   margins, and neither is at the grid border (8 of 19,162 `only_ref` points fall in the outer 8-pixel
   frame, against 3.1% of the grid by area). A systematic over-rejection would put one set where the
   other is not. Two sets of marginal decisions straddling the same threshold in opposite directions
   look exactly like this — and the counts are nearly balanced, 16,893 against 19,162.

   So the earlier reading, that "the filter over-rejects", is too strong. What the filter-disabled test
   established is that these points *are* filter decisions rather than measurement failures; it does
   not establish a bias, and the map argues against one.

   The reference's captured `InterpMask` splits them further:

   | population | points | reference interpolated it |
   |---|---:|---:|
   | reference answers, AutoRIFT.jl does not | 19,162 | **4,258 (22.2%)** |
   | both answer | 598,718 | 44,937 (7.5%) |
   | reference answers at all | 617,880 | 49,195 (8.0%) |

   A 2.8× enrichment against the baseline, so **hole filling owns about 4,258 of the gap and rejection
   owns the other ~14,900.** Those are different mechanisms and want separate fixes.

### The residual tail is the coarse levels, not an edge artifact

The p99 is 1.19 px against the benchmark's 0.1411, so a small population carries it. Mapping the 6,000
points above the 99th percentile against the things that could plausibly cause it:

| population | share of tail | base rate | enrichment |
|---|---:|---:|---:|
| levels disagree | 15.7% | 2.4% | **6.55x** |
| coarse level (reference chip > 24) | 40.1% | 19.9% | 2.02x |
| reference interpolated | 12.2% | 7.6% | 1.59x |
| base level on both sides | 55.4% | 79.7% | 0.69x |

The tail lives where the ice is fast and the search window widest — median `|dx|` 2.75 px against 0.38
elsewhere, median search radius 20 against 7 — and it is **not** railing against the search limit
(9.7% of the tail within 1.5 px of it, against 21.1% of the rest), so it is not a window-size failure.
The base level is *under*-represented at 0.69x.

So the tail is the coarse-level bicubic regime, which the gate already excludes, plus one actionable
part: the 6.55x on level disagreement, 942 points where the two picked different chip sizes and so
describe different footprints.

**Nodata is not the cause, and the heatmap misleads here.** Level disagreement *falls* toward nodata —
0.62% within one grid cell, rising monotonically to 2.72% beyond sixteen, against a 2.40% base — so the
enrichment runs the wrong way for an edge artifact. What looks like nodata outlines in a mask heatmap is
the surrounding valid region being where the points are, not a signal. Only 0.1% of grid points sit on
nodata at all, because the driver zeroes `xGrid`, `yGrid`, `Dx0`, `Dy0`, `SearchLimit*` and the chip
bounds there before `runAutorift` (`testautoRIFT.py:394-403`), and the capture takes those arrays after
that.

The nodata *buffer* is a separate mechanism and is present on both sides. `_wallis_filter_fill` grows
both the missing-data and the low-variance masks with a `distanceTransform` before treating them as
missing (`autoRIFT.py:84-91,104-110`), by `buff = sqrt(2 * ((w-1)/2)^2) + 0.01`; `_gapfill_buffer` in
`src/types.jl` is that expression and `wallis_gapfill` applies it to both masks. It only runs for
`wallis_fill`, so it cannot affect an `hps` case either way.

### The fill criteria differ, and one is missing

Reading `autoRIFT.py:792-808` against `_fill_holes!`, the reference fills a point on **either** of two
conditions, three passes each:

- a 3×3 area closing — `filter2D(foo, ones(3,3)) >= 6`, six of nine neighbours valid;
- **or** `!bwareaopen(!foo1, 5)` — the point's *connected component of invalid points* is smaller than
  5 pixels, whatever the neighbour count.

Both then require `MM`, that a 3×3 median exists at all.

AutoRIFT.jl implements the first and not the second. At `fill_window = 3` its `needed = 2*9÷3 = 6`
matches the area closing exactly, but nothing corresponds to the connected-component test. A 2×2 hole
is the discriminating case: each of its four points has five valid neighbours, one short of the
threshold, so AutoRIFT.jl leaves it open — while its component is size 4 < 5, so the reference fills
it. That is the right shape to produce a fill-only deficit concentrated in small holes, which is what
the 22.2% enrichment measures.

This is a genuine gap rather than a matched-not-endorsed choice, and the connected-component criterion
is defensible on its own terms: a small hole surrounded by coherent motion is exactly what should be
interpolated, and neighbour-counting misses the ones with awkward shapes.

`small_components` implements it, 8-connected as the reference's `connectivity=2` demands, and
`fill_min_hole = 5` is the default. Two details are load-bearing and were taken from the Python rather
than inferred: eight-connectivity, because under four-connectivity a diagonal pair of holes is two
components of one instead of one of two and the size test then answers differently; and the components
are sized on the hole set *after* the neighbour-count criterion has closed what it can — the
reference's `!foo1` (`autoRIFT.py:803`) — since sizing the raw hole set would judge a large hole by a
size it only has before its edge is filled.

**Measured on the golden S2 case, it closes 1,192 points, not the ~4,258 the `InterpMask` enrichment
predicted:**

| `fill_min_hole` | both | only jl | only ref | exact |
|---|---:|---:|---:|---:|
| 0 (disabled) | 598,718 | 16,893 | 19,162 | 67.58% |
| 5 (reference) | 599,910 | 17,238 | 17,970 | 67.92% |

So about 28% of the predicted deficit, and `only_jl` rises by 345 at the same time — the criterion
fills points the reference does not, as well as the other way. The prediction was too high because it
assumed the two sides' holes have the same *shape*: `InterpMask` counts what the reference interpolated,
but AutoRIFT.jl can only fill a hole its own rejection set actually creates, and those sets differ.
Coverage is a joint function of rejection and filling rather than a sum of two independent deficits.

Exact agreement moves 67.58% → 67.92%, which is real and small. The fix stands on its own — the
criterion is right and the reference has it — but the remaining coverage difference is dominated by
rejection, and the fill side is now close to exhausted as an explanation.
3. **The post-correlation chain** — nothing downstream of `correlate` exists in Julia, so no product
   comparison has run.

Two items came off this list by being measured rather than by being fixed:

- **Per-level upsampling** was a real gap, is implemented, and accounts for 0.01 percentage points.
- **Coarse-level resampling** is not a defect: neither implementation's coarse values are quantized,
  because both replace the measurement with a bicubic-interpolated value, so exact agreement is
  unreachable there. Bias is under 0.01 px.

## The gate: the L8/L9 benchmark's level of agreement, on every golden pair

Exact agreement everywhere is not the target, because above the base chip size it is unreachable on
both sides — the reference overwrites its own measurements with a bicubic resize, so neither field is
quantized and two independent implementations of that chain cannot land on the same value (measured
below). The bicubic step is the thing that will not agree between versions, and no amount of work on
AutoRIFT.jl changes that.

So the gate is the agreement the pre-existing L8/L9 benchmark already achieves, reached on all 22
golden pairs. From `tools/ab/README.md`, stage 2, the whole pipeline on a 3072² window, 87,814 shared
points:

| statistic | L8/L9 benchmark | golden S2, base level | golden S2, all levels |
|---|---:|---:|---:|
| exact | **77.4%** | 85.9% | 67.6% |
| within one upsampling step | **97.3%** | 94.5% | 82.3% |
| median radial | **0.0000 px** | 0 | 0.079 |
| bias, both axes | **+0.0000** | ~0 | < 0.02 px |
| p99 | **0.1411 px** | — | 1.19 |
| same chip level | **99.3%** | — | 97.2% |

and, gated on peak strength, 97.6% exact at correlation ≥ 0.5 — the shape to expect, since a weak peak
is where a tie breaks either way.

Read against that, the golden S2 case is **already at benchmark quality on the base level** (85.9%
exact against 77.4%) and short of it overall, on three specific counts: within-one-step is 82.3%
against 97.3%, p99 is 1.19 px against 0.14, and level agreement is 97.2% against 99.3%. Those three are
the work, and the p99 gap in particular says the residual is not uniformly small — there is a tail the
benchmark does not have.

**The gate is similar statistics, not identical ones.** Each pair is a different scene: the fraction
of interpolated points, the spread of chip sizes and the amount of fast flow all vary, so a pair with
more interpolation legitimately scores lower on `exact` than one with less. What has to hold is that
the numbers sit in the benchmark's neighbourhood and that no pair shows a *structured* residual — a
gradient-correlated difference map, an edge artifact, a bias, a level disagreement well above 1%. A
pair that misses 77.4% by a few points with a structureless residual passes; one that hits it with a
dipole along the flow margin does not.

Two cases need the gate stated differently, and both for reasons that are properties of the reference:

- **`wallis_fill` pairs (L7).** The reference does not reproduce *itself* there — two runs agree on
  3.4% of `vx` — so the comparison is against its own run-to-run envelope, measured by running the
  container twice, rather than against a single product.
- **Coarse levels.** Bias and within-one-step, not `exact`, for the reason above.

## Matched for agreement, not endorsed

**Agreement with the reference is the current objective, and it is not the same objective as being
correct.** Where the two conflict, this exercise chooses agreement — because a deliberate difference
and a bug are indistinguishable in a comparison, so every difference has to be removed before the
remaining ones mean anything. That trade has a cost: each choice below makes AutoRIFT.jl reproduce
behaviour there is reason to think is wrong.

They are listed so the debt is visible and so re-litigating one is a decision rather than a
rediscovery. Each names the condition under which it should be revisited. None should be revisited
before the product comparison passes.

| behaviour | why it is questionable | revisit when |
|---|---|---|
| **Per-point chip-size bounds ignored at the base level** (`autoRIFT.py:509` vs `:587`) | A point whose parameter file asks for no chip smaller than 480 m is still correlated at the base chip size. The finest level is where a chip smaller than the parameter file allows does the most damage, and 136,800 points on the golden S2 case are answered against their own `ChipSizeMinX`. The asymmetry reads as an oversight in the reference — the bounds test sits inside an `if` that excludes the base level — rather than a decision. | Product comparison passes. Then measure what honoring the bounds everywhere does to coverage and to `stable_shift`. |
| **Reference reports a search-window corner for a degenerate chip** | A constant chip carries no information about displacement, so `dx = -radius_x, dy = +radius_y` is a fabricated answer over masked and low-texture ground where v1.5.0 correctly produced none. It also flips `M0C = ~isnan(DxC)` true there, changing which pyramid levels are skipped. | Never adopted as the default. AutoRIFT.jl reports no measurement and **there is currently no flag to reproduce the reference's behaviour** — see below, since the coverage gap may require one. |
| **Even-kernel `colfilt` chunk seam** (`autoRIFT.py`) | The code assumes a left margin of `(k-1)÷2` where `generic_filter` uses `k÷2`, so the first output column of each chunk after the first reads padding where it should read data — `nchunks - 1` corrupted columns per row. Only the non-base pyramid levels use even kernels. | Only matters if a coarse-level residual is traced to it. Not reproduced in AutoRIFT.jl; recorded so the *reference's* coarse values are not assumed clean. |
| **`UInt8` quantization before correlating** | `uniform_data_type` rescales each image by its own mean and standard deviation and quantizes to 256 levels before the correlator sees it (`autoRIFT.py:359-384`), discarding precision the filtered float field already has. Nothing about the correlator needs it, and it costs accuracy at every point — the reference's own byte path disagrees with its float path. It exists because the production driver sets `DataType = 0`. | Once every case agrees. Then correlate the float field directly and measure what the quantization was costing. The reference's two entry points (`arImgDisp_u` vs `arImgDisp_s`) are what make this a matching requirement rather than a choice: `tools/ab/README.md` records that the float path is bit-identical while the byte path fails on 1.7% of points by up to 36 px. |
| **Agreement threshold is a fraction of the full window area** | A point at the grid border is held to the same absolute neighbour count as one in the interior, despite having fewer neighbours to corroborate it. Defensible as conservatism, and it is the reference's behaviour, but it is a choice rather than a derivation. | Only if border coverage turns out to matter to the product's cropped extent. |

Two differences run the *other* way — AutoRIFT.jl is more nearly correct and deliberately does not
match:

- **Wallis variance.** AutoRIFT.jl's about-the-mean form is ~360,000× more accurate than the
  reference's `E[x²] − E[x]²` against an exact `Float64` truth (reference median error 0.54, max
  5.79; about-the-mean 1.5e-6, max 9.5e-6). Kept accurate by decision, so `wallis_fill` cases are
  gated on tolerance rather than on equality.
- **A seeded gap-fill RNG.** Reproducibility is worth more than matching any single draw, and the
  reference cannot match itself here either.

## Closed: hypotheses ruled out, with what ruled them out

Every line here cost real time. Recorded so it is spent once. **A closed hypothesis is closed by a
measurement, not by an argument** — if one is reopened, it should be by a new measurement that
contradicts the old one, not by the reasoning that motivated it the first time.

| hypothesis | ruled out by | verdict |
|---|---|---|
| The correlator disagrees with the reference | `tools/ab` stage 1, re-run: **bit-identical**, 100% exact at chip 32 against `arImgDisp_s` | **The correlator is not the problem, and has never been.** This is the anchor. Any golden disagreement above this floor is the harness or the pipeline around the correlator. Check here *first*. |
| The coarse-level residual is a lattice or interpolant error | At a coarse node Catmull-Rom reproduces its sample exactly, so a lattice error would make nodes agree better than off-nodes. Median 0.0980 on nodes against 0.0980 off them. 12 `INTER_CUBIC` fixtures pin the interpolant bit-exact. `int(shape/Scale)` is exactly 2 and 4 here, so the lattice-drift trap is not active | Not the lattice, not the interpolant |
| The coarse-level residual is the multi-level merge | One level run in isolation through `chipsize_level`, no merge and no prior: same 0.00% exact | Not the merge |
| The coarse levels *should* agree exactly | Multiples of 1/16, 1/32, 1/64, 1/128 account for 0.01–0.03% of **either** side's chip-48 values, against 99.4% at the base level. `autoRIFT.py:856-866` overwrites even directly-measured points with a bicubic resize | **`exact` is the wrong statistic above the base level.** Use bias and within-one-step. Exact agreement is unreachable by construction, on both sides. |
| Per-level upsampling was the dominant cause | Implemented the reference's own `{24:16, 48:32, 96:64}` ladder: exact agreement moved 27.42% → 27.43% | Real gap, correctly fixed, ~0 effect. A production configuration could not be expressed without it. |
| The L7 disagreement is the noise fill being reported as signal | The fill *is* rejected — it correlates with nothing and `filtDisp` drops it as designed. Only 9.1% of disagreeing points are flagged `interp_mask` against 5.8% of agreeing ones | **The mechanism is the rejection, not the retention.** A rejected fill point is a missing neighbour for every real point whose `filtDisp` window overlaps it, so a different draw rejects a different set and coverage swings ±3,500 in both directions. |
| `v_error`'s unseeded 10⁶-draw Monte Carlo prevents exact comparison | The standard deviation of that many draws is stable well inside the `int16` rounding the product applies — 8,356 pixels take the value 27 in every run | **`v_error` is reproducible.** Do not widen a tolerance for it. |
| The golden L7 product is a fixed target | Two container runs agree on 3.4% of `vx`, median 9 m/yr, and coverage moves enough to change the product's own `P<nn>` name | **It is one draw from a distribution.** Gate against the reference's own run-to-run envelope, measured by running the container twice. |
| A median residual of one quantization step is benign tie-breaking | It was a half-pixel grid offset. Invisible in the median; obvious in a heatmap | **See below.** This one nearly closed the investigation at the wrong answer. |
| The outlier filter's *parameters* are mis-derived | Computed the reference's `FiltWidth` and `FracValid` from the captured scalars and compared: 9 and 0.41 fine / 0.32 coarse on both sides, `agree_tolerance` 0.2, `mad_scale` 4, 3/2 iterations. `rescale`/`relax` reproduce `autoRIFT.py:484-505` exactly | Not the parameters. The disagreement is in *when* the filter runs and in reducer semantics, so look there. |
| The coverage gap is mostly the deliberate degenerate-chip difference | Disabling the filter drops `only_ref` from 19,162 to **672** | **It is filter and fill decisions**, not the degenerate-chip choice, which accounts for at most 672 points. Corrects an earlier claim in this file. |
| The filter is systematically *over*-rejecting | Mapped both exclusive sets: same speckle along the same margins, counts nearly balanced (16,893 vs 19,162), and only 8 of 19,162 at the grid border against 3.1% by area | **Bidirectional and co-located, so not a bias** — marginal decisions straddling one threshold in both directions. A systematic over-rejection would put one set where the other is not. |
| The whole coverage gap is one mechanism | Reference `InterpMask` on the `only_ref` points: 22.2% interpolated against a 8.0% baseline | **Two mechanisms**, but not additively. Implementing the fill criterion closed 1,192 of the predicted ~4,258 and *added* 345 the other way, because a hole can only be filled where that side's own rejection created one and the two rejection sets differ. Coverage is joint, not a sum of independent deficits — do not size a fix from an enrichment ratio alone. |
| A matching bicubic interpolator would close the coarse-level residual | The interpolant is already matched: 12 `INTER_CUBIC` fixtures in `test/fixtures/resize/` pin it bit-exact, and the node test settles it independently — at a coarse node Catmull-Rom reproduces its own sample, so a kernel or lattice error would make nodes agree better than off-nodes, and they agree *identically* (median 0.0980 both) | **Writing one would not help.** What differs is the interpolation's *input*, not the interpolation: the chain is decimate → median filter → area-resize → hole-fill → bicubic, and the rejection and fill sets differ slightly upstream, so both sides interpolate slightly different fields. The same interpolator on different inputs gives different outputs. Spend the effort on level agreement (97.2% against the benchmark's 99.3%), which carries a 6.55x tail enrichment. |
| Border/reducer semantics explain the gap | 8 of 19,162 `only_ref` points lie in the outer 8-pixel frame; `count_agreeing` and `windowmedmad` were read against `colfilt` options 5 and 6 and are equivalent on NaN centres, NaN neighbours and odd-window margins | Not the borders, not the reducers |

### The lesson that generalizes: plot before reasoning

The half-pixel bug presented as a median `|ddx|` of exactly 1/16 — one quantization step, entirely
plausible as tie-breaking on weak peaks, and consistent with a correlator that was known to be
bit-identical. Every summary statistic was reassuring. It was found only by looking at the difference
map, which was blank over flat ice and washed along every fast-flow margin.

That shape *is* the signature of a grid offset, and the reason it hides is structural: a shifted grid
produces **zero** residual under uniform motion and a residual proportional to the local velocity
gradient everywhere else. Since most of a scene is slow, the median is dominated by the pixels that
cannot show the error. Binning residual against gradient made it quantitative — exact agreement fell
from 66.5% in the flattest decile to 5.6% in the steepest — and scanning the offset settled the value
empirically rather than by argument: 85.0% at `+1.0` against 49.2% at `+0.5`.

Three habits follow, and they are cheap:

1. **Heatmap the two fields and their difference before computing anything else.** Garbage, an
   off-by-one, a flipped axis and a transpose are each visually unmistakable and each survives a
   plausible-looking median.
2. **Scan the parameter rather than deriving it.** The `+1` vs `+0.5` question had a clean argument on
   both sides — `_shift_points` adds AutoRIFT.jl's own half pixel, so subtracting the reference's
   looks right — and the argument was wrong. `residual_maps.jl` keeps the offset scan and the
   whole-pixel roll test as standing tools.
3. **Bin the residual against the local gradient.** A gradient-correlated residual is a geometry bug.
   An uncorrelated one is arithmetic. This distinguishes them in one plot.

The conventions that produce these bugs are worth naming, since three of the four have now cost
something here: Julia indexes from 1 where Python indexes from 0; Julia is row-major-ish `[row, col]`
where the reference thinks `[col, row]`; projected y increases *upward* while the row index increases
downward; and pixel-is-area against pixel-is-point puts a half pixel between two defensible answers.
`tools/ab/README.md` asserts each of these in code rather than describing them, which is the only form
that keeps working.
