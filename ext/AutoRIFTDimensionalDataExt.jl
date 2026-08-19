"""
    AutoRIFTDimensionalDataExt

Accept `AbstractDimArray` input and return a `DimStack` whose dimensions describe
the output grid.

The core is deliberately array-only, in pixel coordinates, so it loads without
any geospatial dependency and works equally on map-projected imagery and radar
slant-range data. This extension is the seam: it reads coordinates off the input
lookups, calls the plain-array core, and rebuilds the result as a stack on the
output grid.

Filled in at milestone M7; the module exists now so that `Pkg.test()` and
precompilation succeed while the core is being built.
"""
module AutoRIFTDimensionalDataExt

import AutoRIFT
import DimensionalData as DD

end # module
