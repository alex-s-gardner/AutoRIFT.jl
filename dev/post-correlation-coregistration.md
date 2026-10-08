# Post-correlation coregistration: tried, measured, rejected

**Conclusion: coregister SAR pairs by resampling the secondary before correlation, as ISCE3 does. Do
not hand the correlator the geometric misregistration instead.** Two designs were built and measured
against ISCE3 on the golden cases. Neither can reproduce the resampled route, and neither is faster
by enough to matter. This file holds the evidence so that the idea is not rebuilt without new
information.

## What was tried

The idea: instead of resampling the secondary SLC onto the reference grid, correlate the raw secondary
and pass the correlator, per point, the offset between the two acquisitions' grids (a few hundred lines
on NISAR, a few lines and up to hundreds of samples on Sentinel-1).

**Design 1: misregistration added to the prior, subtracted afterwards.** `pointset(g; offset)` added
the offset field to `dx_prior`/`dy_prior` and `remove_misregistration` took it back out of the result.
It is wrong in two ways:

- The outlier filter normalizes each displacement by its own search radius before comparing
  neighbours (`nx = dx / rx` in `GardnerFilter`). A large offset divided by radii that vary point to
  point looks incoherent where the measurements agree. On NISAR P094 (offset about −253 lines) a
  noise-free prior field kept 42.0% of its points with the real radii and 80.5% with a constant radius.
  The offset also inflates decimation's `windowrange`, halo inflation and hole filling.
- A fractional prior needs a sub-pixel chip shift. With integer chip placement alone, the floor
  arithmetic leaves errors up to ±1 px and a 0.19 px S-curve in the fraction.

**Design 2: misregistration as its own `PointSet` fields.** The search window moved by the whole-pixel
part, the chip by the fractional remainder through an 8-tap Hann sinc, and the reported displacement
excluded the offset, so the outlier filter and coarse passes saw a resampled-equivalent field. Chip
validity was judged against each image's own mask. Two traps it had to avoid:

- Production puts the reference acquisition in AutoRIFT's secondary (chip) slot. Moving the chip by the
  offset instead of the window attaches each measurement to ground displaced by the offset (252 lines
  on NISAR).
- Correlating raw imagery means one shift per chip; the offset field varies by about 0.05 px across a
  768 × 416 chip on NISAR, which is small but not zero.

Design 2 worked mechanically: yield, filtering and multi-level search behaved as on a resampled pair.
It failed on accuracy, for the reason below.

## Why it cannot match ISCE3

AutoRIFT correlates **detected amplitude**. ISCE3 resamples the **complex** secondary (deramped for
TOPS, Doppler-demodulated for NISAR) and only then takes the magnitude. A post-correlation design has to
apply the fractional offset to amplitude, after detection, and detection roughly doubles the signal's
bandwidth, so a near-critically sampled amplitude image cannot be shifted exactly from its own samples.
The resulting bias is odd in the fractional offset and is not removed by a better kernel.

Measured on NISAR P094 against ISCE3's own `coregistered_secondary.slc`. Each of 250 windows of
128 × 128 shifts the raw secondary by the exact fraction ISCE3's `azimuth.off`/`range.off` give; a
correct interpolator leaves zero bias.

| what is shifted, and how | azimuth bias | range bias, odd part |
|---|---|---|
| amplitude, 8-tap Hann sinc | −0.043 px | 0.034 px |
| amplitude, 32-tap Hann sinc | −0.061 px | 0.056 px |
| amplitude, exact (DFT) shift | −0.063 px | 0.058 px |
| intensity, exact (DFT) shift, then square root | −0.104 px | 0.088 px |
| complex, 8-tap Hann sinc, then magnitude (ISCE3's method) | −0.0001 px | 0.001 px |
| complex, exact (DFT) shift, then magnitude | −0.0075 px | 0.007 px |

The bias grows as the amplitude interpolator improves, so it is intrinsic, not a kernel defect; the
short kernel's smoothing partly hides it. The shift estimator is not the cause: the complex row uses
the same estimator and matches ISCE3 to 0.0001 px.

The same bias appears in the correlator. On Sentinel-1 P089, design 2 minus the resampled route, binned
by the fractional range offset, at 845,017 points both measured:

| subswath | bias range |
|---|---|
| IW1 | +0.017 px at a fraction of −0.22 to −0.030 px at +0.32 |
| IW2 | about ±0.01 px |
| IW3 | about ±0.01 px |

IW1, sampled closest to its bandwidth, is worst. In the products the bias shows as stripes that follow
the contours of the fractional offset: along range on P089, diagonal on NISAR P094, where the offset
field varies in both directions.

## No runtime case either

Once the resampled route was correct, the two routes cost about the same. On NISAR P094 correlation
took 789 s with design 2 and 802 s resampling. Sentinel-1 timings went both ways and were within the
noise of the runs.

## What revisiting it would take

Each chip's fractional shift has to be applied to complex data before detection. That means:

- the correlator takes complex, deramped or Doppler-demodulated imagery;
- each chip is shifted, then detected;
- the preprocessing filter runs per chip instead of per tile (the tile filter cache no longer applies);
- the GPU path does the same.

Then it has to be measured faster than resampling, which costs about as much as the correlation
itself. Without both, the resampled route is the one to keep.

## Differences that were not the coregistration route

Several gaps first attributed to design 2 were defects in geometry or resampling shared by both routes,
since fixed in the pipeline that drives this package (ItsLiveOffsetProduction.jl, `tools/golden`). Check
these before blaming a route:

- `coregistration_offset` read the secondary line from `geo2rdr`'s image-clock time, which carries the
  geogrid's `midtime − orbit_midtime` offset: 1 line for an even line count, 1.5 for an odd one
  (Sentinel-1 IW2, 1515 lines per burst).
- Its height solve oscillated in steep relief and put the ground point off the range sphere.
- `ResampledRSLC` interpolated NISAR's complex data without removing its 968 Hz Doppler centroid.
- Golden Sentinel-1 runs that replay COMPASS sit on COMPASS's static-layer grid, not the annotation
  grid. Products from raw inputs therefore differ from them in noise realization (speed robust sd
  about 64 m/yr on P089) without either being biased.
