"""ISCE3's own Rdr2Geo and Geo2Rdr on the golden burst pair, and the offsets they write.

    micromamba run -n arift-ref python tools/golden/isce_offsets.py <run_dir> <out_dir>

The two steps COMPASS runs, with COMPASS's own arguments: a zero-Doppler `LUT2d()` for each and the
burst's own `as_isce3_radargrid()`, taken from `s1_rdr2geo.run` and `s1_geo2rdr.run`. So `azimuth.off` and
`range.off` here are what the resampler consumes, and comparing them against
`radar.jl`'s `coregistration_offset` separates a difference in the geometry from a difference in
convention.

That comparison is what establishes the one line `coregistration_offset` subtracts. Over twenty points
spanning a burst: range agrees to a mean of -0.0 with a spread of 8e-10, and azimuth differs by
+0.99999904 with a spread of 5e-8 — an exact constant, not a geometry error.

Needs the reference environment for `isce3` and `s1reader`, and the run directory's `dem.tif` and both
SAFEs. Writes `x/y/z.tif`, `topo.vrt`, `azimuth.off` and `range.off` into `out_dir`; the topo rasters are
31 million pixels each, so give it room.
"""
import os, sys, glob
import numpy as np
import isce3, s1reader
from osgeo import gdal

run = sys.argv[1]
out = sys.argv[2]
os.makedirs(out, exist_ok=True)

rs = sorted(glob.glob(f'{run}/S1C_IW_SLC__1SSV_20250416*.SAFE'))[0]
ss = sorted(glob.glob(f'{run}/S1C_IW_SLC__1SSV_20250428*.SAFE'))[0]
ro = sorted(glob.glob(f'{run}/S1C_OPER_AUX_POEORB*V20250415*.EOF'))[0]
so = sorted(glob.glob(f'{run}/S1C_OPER_AUX_POEORB*V20250427*.EOF'))[0]

rb = s1reader.load_bursts(rs, ro, 1, 'VV')[0]
sb = s1reader.load_bursts(ss, so, 1, 'VV')[0]
print('ref burst sensing_start', rb.sensing_start, 'shape', rb.shape)
print('sec burst sensing_start', sb.sensing_start, 'shape', sb.shape)

dem = isce3.io.Raster(f'{run}/dem.tif')
proj = isce3.core.make_projection(dem.get_epsg())
ell = proj.ellipsoid

ref_grid = rb.as_isce3_radargrid()
print('ref grid: length %d width %d sensing_start %.6f prf %.6f r0 %.3f dr %.6f'
      % (ref_grid.length, ref_grid.width, ref_grid.sensing_start,
         ref_grid.prf, ref_grid.starting_range, ref_grid.range_pixel_spacing))

# Step 1 — the reference's topo, exactly as `s1_rdr2geo.run` builds it.
xs, ys, zs = (isce3.io.Raster(f'{out}/{n}.tif', ref_grid.width, ref_grid.length, 1,
                              gdal.GDT_Float64, 'GTiff') for n in ('x', 'y', 'z'))
r2g = isce3.geometry.Rdr2Geo(ref_grid, rb.orbit, ell, isce3.core.LUT2d(),
                             threshold=1e-8, numiter=25, extraiter=10, lines_per_block=1000)
r2g.topo(dem, x_raster=xs, y_raster=ys, height_raster=zs)
vrt = isce3.io.Raster(f'{out}/topo.vrt', [xs, ys, zs])
vrt.set_epsg(r2g.epsg_out)
del xs, ys, zs, vrt
print('topo done, epsg', r2g.epsg_out)

# Step 2 — the secondary's geo2rdr against that topo, exactly as `s1_geo2rdr.run` does.
sec_grid = sb.as_isce3_radargrid()
g2r = isce3.geometry.Geo2Rdr(sec_grid, sb.orbit, ell, isce3.core.LUT2d(),
                             1e-8, 25, 1000)
g2r.geo2rdr(isce3.io.Raster(f'{out}/topo.vrt'), out)
print('geo2rdr done ->', [os.path.basename(p) for p in sorted(glob.glob(f'{out}/*.off'))])

az = gdal.Open(f'{out}/azimuth.off').ReadAsArray()
rg = gdal.Open(f'{out}/range.off').ReadAsArray()
print('\n%-6s %-7s %12s %12s' % ('line', 'sample', 'azimuth.off', 'range.off'))
for line in (100, 700, 1300):
    for sample in (5000, 9000, 13000):
        print('%-6d %-7d %12.4f %12.4f' % (line, sample, az[line, sample], rg[line, sample]))
