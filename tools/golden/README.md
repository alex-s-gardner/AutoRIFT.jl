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

### Landsat 7 is the exception, and the reference does not reproduce itself there

The same measurement on `LE07_L1TP_061018_20120428` gives a completely different answer. Two runs of
unchanged code on identical inputs:

| comparison | `vx` exact | median \|Δ\| | p95 | max | bias |
|---|---:|---:|---:|---:|---:|
| run 1 vs run 2 | **3.4%** | 9 m/yr | 34 | 96 | −0.15 |
| run 1 vs golden | 3.3% | 9 m/yr | 35 | 101 | −0.24 |
| run 2 vs golden | 3.3% | 9 m/yr | 34 | 111 | −0.03 |

The rows are statistically indistinguishable: **a local run differs from ASF's golden product by
exactly as much as two local runs differ from each other.** The product name changes too — `P010`,
`P011`, `P010` — because coverage moves, and with it `stable_shift`, `stable_count` and all four
`error` attributes on both components.

The cause is `_wallis_filter_fill` (`autoRIFT.py:113-125`): Landsat 7's Scan Line Corrector gaps are
filled with `rng.normal` from an **unseeded** `np.random.default_rng()`, so the *input imagery* is
different on every run and the correlator faithfully reports different displacements. The driver
selects the filter by scene name — `wallis_fill` for `L[EO]07_`, `fft` for `LT0[45]_`, `hps`
otherwise (`testautoRIFT.py:718-723`) — which is why S2 is exact and L7 is not.

So **the golden L7 product is one draw from a distribution, not a fixed target**, and its own
generator lands ±9 m/yr median away from it. Those cases have to be gated against the reference's own
run-to-run envelope, measured per case by running the container twice; only the `hps` cases — L8, L9
and S2 — can be gated exactly.

`REFERENCE.md` already recorded that this RNG is unseeded and that AutoRIFT.jl's `WallisGapfill` is
seeded and therefore reproducible. What this measures is the consequence: 3.4% rather than 100%, and
which of the 22 cases it reaches.

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
| `dx` | + | 592,612 | 16,892 | 25,268 | 27.4% | 0.0625 | 0.88 | **+0.9954** |
| `dy` | − | 592,612 | 16,892 | 25,268 | 30.3% | 0.0625 | 0.82 | **+0.9917** |

Bias is under 0.01 px on both axes and the median disagreement is exactly one upsampling step, so the
two agree about position. Exact agreement at 27–30% against the 77% `tools/ab` measures on a hand-cut
window is the finding, and decomposing it says why:

| population | points | exact `dx` |
|---|---:|---:|
| all both-measured | 592,612 | 27.4% |
| same chip size chosen | 433,781 | 37.5% |
| same chip size, not interpolated by the reference | 404,172 | 39.5% |
| **chip 24 only**, the base level | 319,521 | **50.0%** |

**27% of points chose a different chip size**, and those agree on 0.16% — a point answered at chip 24
by one side and 48 by the other describes a different footprint of ground, so this is not a
correlator disagreement at all. The `dy` sign is measured, not assumed: both signs are scored and the
better kept.

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
| reference reproducibility floor, S2 (`hps`) | **exact** — 7/7 planes, 615,146 px, `time` only |
| container vs ASF golden, S2 | **exact** — bit-identical across arch and four months |
| reference reproducibility floor, L7 (`wallis_fill`) | **±9 m/yr median**, 3.4% exact — unseeded gap-fill RNG |
| container vs ASF golden, L7 | indistinguishable from that floor, so golden is one draw |
| correlator vs reference `Dx`/`Dy`, S2 | corr 0.995/0.992, bias < 0.01 px; base level 50% exact and 99.4% on-grid |
| AutoRIFT.jl product against golden | needs the post-correlation chain (phase 2) |

### Open, in priority order

1. **Chip-size selection** — 27% of points pick a different level, and those agree on 0.16%.
   `tools/ab` sees 0.7% level disagreement on its window, so something about the production
   configuration widens this considerably. Now the largest unexplained effect.
2. **Coverage** — 16,892 points AutoRIFT.jl answers alone against 25,268 the reference does. Partly
   the deliberate degenerate-chip difference in `REFERENCE.md`, but the split is not yet accounted
   for.
3. **The post-correlation chain** — nothing downstream of `correlate` exists in Julia, so no product
   comparison has run.

Two items came off this list by being measured rather than by being fixed:

- **Per-level upsampling** was a real gap, is implemented, and accounts for 0.01 percentage points.
- **Coarse-level resampling** is not a defect: neither implementation's coarse values are quantized,
  because both replace the measurement with a bicubic-interpolated value, so exact agreement is
  unreachable there. Bias is under 0.01 px.
