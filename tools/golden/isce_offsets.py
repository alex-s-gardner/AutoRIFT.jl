"""ISCE3's own Rdr2Geo, Geo2Rdr and ResampSlc on the golden burst pair.

    micromamba run -n arift-ref python tools/golden/isce_offsets.py <run_dir> <out_dir>
    micromamba run -n arift-ref python tools/golden/isce_offsets.py <run_dir> <out_dir> --bench

The default mode runs the two geometry steps COMPASS runs, with COMPASS's own arguments: a zero-Doppler
`LUT2d()` for each and the burst's own `as_isce3_radargrid()`, taken from `s1_rdr2geo.run` and
`s1_geo2rdr.run`. So `azimuth.off` and `range.off` here are what the resampler consumes, and comparing
them against `radar.jl`'s `coregistration_offset` separates a difference in the geometry from a
difference in convention.

That comparison is what establishes the one line `coregistration_offset` subtracts. Over twenty points
spanning a burst: range agrees to a mean of -0.0 with a spread of 8e-10, and azimuth differs by
+0.99999904 with a spread of 5e-8 — an exact constant, not a geometry error.

`--bench` is the cost of the same work, over every burst of the subswath and including the resample, for
comparison against `tools/golden/coreg_bench.jl`. It reports wall clock per stage, peak resident memory
and **bytes written to disk**, because the disk traffic is not incidental to this implementation: ISCE3
solves the geometry per output pixel and writes it out — three `Float64` topo rasters, two offset
rasters and the resampled SLC per burst — where the Julia side solves a lattice and interpolates. A
comparison that reported only wall clock would miss the larger difference.

Two properties of the measurement:

  * **Peak is `ru_maxrss` for the whole process**, so it carries the interpreter and every raster GDAL
    has cached. The Julia figure `coreg_bench.jl` reports is a sampled footprint above the harness's own
    floor. Comparable in order of magnitude, not in the last hundred MiB.
  * **The unit of `ru_maxrss` is platform-dependent** — bytes on macOS, kibibytes on Linux — so it is
    converted explicitly rather than assumed.

Needs the reference environment for `isce3` and `s1reader`, and the run directory's `dem.tif` and both
SAFEs. Writes `x/y/z.tif`, `topo.vrt`, `azimuth.off` and `range.off` into `out_dir`; the topo rasters are
31 million pixels each, so give it room.
"""
import glob
import os
import resource
import sys
import time

import isce3
import numpy as np
import s1reader
from osgeo import gdal

# `ru_maxrss` is bytes on macOS and kibibytes on Linux. Getting this wrong reports a gibibyte as a
# mebibyte, which reads as the reference being frugal.
MAXRSS_SCALE = 1 if sys.platform == "darwin" else 1024


def peak_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * MAXRSS_SCALE


def dir_bytes(path):
    """Bytes of regular files under `path`, which is what a stage wrote."""
    total = 0
    for root, _, names in os.walk(path):
        for n in names:
            try:
                total += os.path.getsize(os.path.join(root, n))
            except OSError:
                pass
    return total


def load_pair(run):
    """The two SAFEs, their orbits and the DEM the container downloaded, from a run directory."""
    rs = sorted(glob.glob(f"{run}/S1C_IW_SLC__1SSV_20250416*.SAFE"))[0]
    ss = sorted(glob.glob(f"{run}/S1C_IW_SLC__1SSV_20250428*.SAFE"))[0]
    ro = sorted(glob.glob(f"{run}/S1C_OPER_AUX_POEORB*V20250415*.EOF"))[0]
    so = sorted(glob.glob(f"{run}/S1C_OPER_AUX_POEORB*V20250427*.EOF"))[0]
    return rs, ss, ro, so


def run_default(run, out):
    """The two geometry steps and a twenty-point read of the offsets they write."""
    rs, ss, ro, so = load_pair(run)
    rb = s1reader.load_bursts(rs, ro, 1, "VV")[0]
    sb = s1reader.load_bursts(ss, so, 1, "VV")[0]
    print("ref burst sensing_start", rb.sensing_start, "shape", rb.shape)
    print("sec burst sensing_start", sb.sensing_start, "shape", sb.shape)

    dem = isce3.io.Raster(f"{run}/dem.tif")
    proj = isce3.core.make_projection(dem.get_epsg())
    ell = proj.ellipsoid

    ref_grid = rb.as_isce3_radargrid()
    print("ref grid: length %d width %d sensing_start %.6f prf %.6f r0 %.3f dr %.6f"
          % (ref_grid.length, ref_grid.width, ref_grid.sensing_start,
             ref_grid.prf, ref_grid.starting_range, ref_grid.range_pixel_spacing))

    # Step 1 — the reference's topo, exactly as `s1_rdr2geo.run` builds it.
    xs, ys, zs = (isce3.io.Raster(f"{out}/{n}.tif", ref_grid.width, ref_grid.length, 1,
                                  gdal.GDT_Float64, "GTiff") for n in ("x", "y", "z"))
    r2g = isce3.geometry.Rdr2Geo(ref_grid, rb.orbit, ell, isce3.core.LUT2d(),
                                 threshold=1e-8, numiter=25, extraiter=10, lines_per_block=1000)
    r2g.topo(dem, x_raster=xs, y_raster=ys, height_raster=zs)
    vrt = isce3.io.Raster(f"{out}/topo.vrt", [xs, ys, zs])
    vrt.set_epsg(r2g.epsg_out)
    del xs, ys, zs, vrt
    print("topo done, epsg", r2g.epsg_out)

    # Step 2 — the secondary's geo2rdr against that topo, exactly as `s1_geo2rdr.run` does.
    sec_grid = sb.as_isce3_radargrid()
    g2r = isce3.geometry.Geo2Rdr(sec_grid, sb.orbit, ell, isce3.core.LUT2d(), 1e-8, 25, 1000)
    g2r.geo2rdr(isce3.io.Raster(f"{out}/topo.vrt"), out)
    print("geo2rdr done ->", [os.path.basename(p) for p in sorted(glob.glob(f"{out}/*.off"))])

    az = gdal.Open(f"{out}/azimuth.off").ReadAsArray()
    rg = gdal.Open(f"{out}/range.off").ReadAsArray()
    print("\n%-6s %-7s %12s %12s" % ("line", "sample", "azimuth.off", "range.off"))
    for line in (100, 700, 1300):
        for sample in (5000, 9000, 13000):
            print("%-6d %-7d %12.4f %12.4f" % (line, sample, az[line, sample], rg[line, sample]))


def run_bench(run, out, nbursts=None):
    """Every burst of the subswath through `Rdr2Geo`, `Geo2Rdr` and `ResampSlc`, timed.

    One output directory per burst, because `Geo2Rdr` names its products `azimuth.off` and `range.off`
    with no burst in the name and `ResampSlc` reads them by that name. Kept rather than deleted so the
    disk figure is the high-water mark the pipeline actually reaches.
    """
    rs, ss, ro, so = load_pair(run)
    rbs = s1reader.load_bursts(rs, ro, 1, "VV")
    sbs = s1reader.load_bursts(ss, so, 1, "VV")
    n = len(rbs) if nbursts is None else min(nbursts, len(rbs))
    dem = isce3.io.Raster(f"{run}/dem.tif")
    ell = isce3.core.make_projection(dem.get_epsg()).ellipsoid

    print("IW1, %d bursts of %d x %d, isce3 %s" % (n, rbs[0].length, rbs[0].width, isce3.__version__))
    print("\n%-6s %10s %10s %10s %10s %10s" % ("burst", "rdr2geo", "geo2rdr", "vrt", "resamp", "sum"))
    tot = dict(rdr2geo=0.0, geo2rdr=0.0, vrt=0.0, resamp=0.0)
    t_all = time.perf_counter()
    for i in range(n):
        rb, sb = rbs[i], sbs[i]
        bd = os.path.join(out, "b%02d" % (i + 1))
        os.makedirs(bd, exist_ok=True)
        ref_grid = rb.as_isce3_radargrid()

        t = time.perf_counter()
        xs, ys, zs = (isce3.io.Raster(f"{bd}/{nm}.tif", ref_grid.width, ref_grid.length, 1,
                                      gdal.GDT_Float64, "GTiff") for nm in ("x", "y", "z"))
        r2g = isce3.geometry.Rdr2Geo(ref_grid, rb.orbit, ell, isce3.core.LUT2d(),
                                     threshold=1e-8, numiter=25, extraiter=10, lines_per_block=1000)
        r2g.topo(dem, x_raster=xs, y_raster=ys, height_raster=zs)
        vrt = isce3.io.Raster(f"{bd}/topo.vrt", [xs, ys, zs])
        vrt.set_epsg(r2g.epsg_out)
        del xs, ys, zs, vrt
        t_r2g = time.perf_counter() - t

        t = time.perf_counter()
        sec_grid = sb.as_isce3_radargrid()
        g2r = isce3.geometry.Geo2Rdr(sec_grid, sb.orbit, ell, isce3.core.LUT2d(), 1e-8, 25, 1000)
        g2r.geo2rdr(isce3.io.Raster(f"{bd}/topo.vrt"), bd)
        t_g2r = time.perf_counter() - t

        # The resampler's input is the burst VRT `slc_to_vrt_file` writes, whose source rectangle covers
        # the valid window alone — the same rectangle `radar.jl`'s `_zero_outside_valid!` reproduces.
        t = time.perf_counter()
        sb.slc_to_vrt_file(f"{bd}/secondary.vrt")
        t_vrt = time.perf_counter() - t

        t = time.perf_counter()
        resamp = isce3.image.ResampSlc(sec_grid, sb.doppler.lut2d, sb.get_az_carrier_poly())
        resamp.resamp(f"{bd}/secondary.vrt", f"{bd}/coregistered.slc",
                      f"{bd}/range.off", f"{bd}/azimuth.off")
        t_res = time.perf_counter() - t

        tot["rdr2geo"] += t_r2g
        tot["geo2rdr"] += t_g2r
        tot["vrt"] += t_vrt
        tot["resamp"] += t_res
        print("%-6d %10.3f %10.3f %10.3f %10.3f %10.3f"
              % (i + 1, t_r2g, t_g2r, t_vrt, t_res, t_r2g + t_g2r + t_vrt + t_res))
        sys.stdout.flush()

    wall = time.perf_counter() - t_all
    written = dir_bytes(out)
    peak = peak_bytes()
    print("%-6s %10.3f %10.3f %10.3f %10.3f %10.3f"
          % ("total", tot["rdr2geo"], tot["geo2rdr"], tot["vrt"], tot["resamp"], sum(tot.values())))
    print("\nwall        %10.1f s" % wall)
    print("peak rss    %10.2f GiB" % (peak / 2 ** 30))
    print("disk        %10.2f GiB in %s" % (written / 2 ** 30, out))
    print("\nRESULT\t%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.1f\t%d\t%d"
          % (n, tot["rdr2geo"], tot["geo2rdr"], tot["vrt"], tot["resamp"], wall, peak, written))


def main(argv):
    if len(argv) < 3:
        raise SystemExit(__doc__)
    run, out = argv[1], argv[2]
    os.makedirs(out, exist_ok=True)
    if "--bench" in argv:
        i = argv.index("--bursts") if "--bursts" in argv else None
        run_bench(run, out, nbursts=int(argv[i + 1]) if i else None)
    else:
        run_default(run, out)


if __name__ == "__main__":
    main(sys.argv)
