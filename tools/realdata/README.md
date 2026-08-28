# The real-imagery test case

`test/realdata.jl` correlates a Landsat 8/9 pair over Jakobshavn Isbrae and compares the result
against the ITS_LIVE granule produced from **the same pair**. It is skipped unless the cache below
exists, so a bare clone runs everything else unaffected.

Every other test in the suite uses synthetic texture with an arithmetically known answer. That
verifies the correlator; it does not verify the package against ice. This case brings cloud, saturated
snow, shear margins, crevasse fields that decorrelate, and a velocity field spanning three orders of
magnitude — and it caught a transposed raster read that no synthetic test could, because a square
window transposed still correlates against itself perfectly.

## The pair

| | |
|---|---|
| Reference | `LC09_L1TP_009011_20230618_20230618_02_T1`, band 8 |
| Secondary | `LC08_L1TP_009011_20230626_20230710_02_T1`, band 8 |
| Separation | 8.00 days |
| ITS_LIVE granule | `LC09_..._X_LC08_..._G0120V02_P036.nc` |

Of the 1,740 ITS_LIVE granules at path/row 009/011, four are summer pairs at 6–31 day separation and
this is the only one with usable coverage — 36% valid pixels against 2–3% for the rest.

Band 8 and not band 4: the granule records `x_pixel_size = 15.0`, so ITS_LIVE ran the panchromatic
band. Its `chip_size_width` values are 240/480/960 m, which at 15 m is the 16/32/64 px the test
configures.

## Building the cache

Three files, into `~/data/autorift/tests` (override with `AUTORIFT_TESTDATA`):

```
~/data/autorift/tests/
├── itslive/LC09_L1TP_009011_20230618_..._G0120V02_P036.nc
└── landsat/
    ├── LC09_L1TP_009011_20230618_20230618_02_T1_B8.TIF
    └── LC08_L1TP_009011_20230626_20230710_02_T1_B8.TIF
```

The granule is public and anonymous:

```bash
mkdir -p ~/data/autorift/tests/itslive
aws s3 cp --no-sign-request \
  s3://its-live-data/velocity_image_pair/landsatOLI/v02/N60W050/LC09_L1TP_009011_20230618_20230618_02_T1_X_LC08_L1TP_009011_20230626_20230710_02_T1_G0120V02_P036.nc \
  ~/data/autorift/tests/itslive/
```

The scenes are **not**. `s3://usgs-landsat` is Requester Pays, so either supply AWS credentials and
pass `--request-payer requester`, or download the two band-8 GeoTIFFs from
[EarthExplorer](https://earthexplorer.usgs.gov/) — *Landsat → Landsat Collection 2 Level-1 → Landsat
8-9 OLI/TIRS C2 L1*, searching on Landsat Product Identifier.

Then cut the window:

```bash
julia --project=tools/realdata -e 'using Pkg; Pkg.instantiate()'
julia --project=tools/realdata tools/realdata/prepare.jl
```

This writes `window.jls`: a 2048² window centred on the trunk at (−49.8°, 69.15°), the ITS_LIVE
velocities resampled onto the same grid, and the acquisition separation. The test reads only that
file, so it needs neither ArchGDAL nor Proj — which is why they are in this directory's project
rather than the package's test dependencies.

## Two traps this directory records

**Rasters returns `(X, Y)` order.** `Array(raster)` is therefore indexed `[x, y]` — column first —
while AutoRIFT indexes `[row, col]` like every other Julia matrix. `prepare.jl` transposes on the way
in. Getting this wrong produces a field that is coherent, plausible, and correlates with the
reference at **−0.11**.

**The y axis flips.** A row index increases southward; the granule's northing increases northward. So
`vy` compares against `+out.dy` where `vx` compares against `-out.dx`. The test states this rather
than absorbing it.

## What agreement looks like

Measured on the pair above, over 32,128 comparable grid points:

| | |
|---|---:|
| `vx` correlation | 0.986 |
| `vy` correlation | 0.985 |
| speed correlation | 0.990 |
| speed correlation, ice faster than 2000 m/yr | 0.992 |
| median speed error | −6 m/yr on a 350 m/yr median field |
| flow-direction error on fast ice | 6.2° median |

The assertions sit well inside these, since ITS_LIVE is another correlator's answer rather than
ground truth — it uses a per-point search radius derived from a prior velocity field, which this test
does not reproduce. A regression shows up as a correlation collapse, not as a margin nudged.
