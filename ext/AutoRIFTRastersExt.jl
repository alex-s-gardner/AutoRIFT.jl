"""
    AutoRIFTRastersExt

Accept `Raster` input and return a `RasterStack` carrying the output grid's
coordinates, CRS, and per-layer `missingval`.

Adds to what the DimensionalData extension provides: CRS validation between the
two input images, `Projected` rather than plain `Sampled` lookups on the output,
and pixel-size lookup so displacements can be reported as velocity.

Filled in at milestone M7; the module exists now so that `Pkg.test()` and
precompilation succeed while the core is being built.
"""
module AutoRIFTRastersExt

import AutoRIFT
import Rasters
import GeoInterface as GI

end # module
