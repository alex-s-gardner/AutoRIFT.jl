"""
    AutoRIFTDiskArraysExt

Teach [`AutoRIFT.ondisk`](@ref) to recognize a chunked backend, so an unblocked run refuses input it
would read a pixel at a time.

Its own extension rather than part of `AutoRIFTRastersExt`, because the property is not about
rasters: a Zarr, NetCDF or HDF5 array reached through `DiskArrays` has it too, and each of those
loads without Rasters. `DiskArrays.isdisk` is the predicate every such backend already answers.
"""
module AutoRIFTDiskArraysExt

import AutoRIFT
import DiskArrays

# More specific than the core's `::AbstractArray` fallback, so this wins by dispatch. Repeating that
# signature — `ondisk(::AbstractArray)`, which `AbstractArray{<:Any,N} where N` also spells — would
# *overwrite* the fallback instead, and precompilation refuses a method overwrite.
#
# So a wrapper that is backed by a disk array without subtyping one needs its own method rather than a
# call to `isdisk` here; `AutoRIFTRastersExt` adds the one for `AbstractRaster`.
AutoRIFT.ondisk(::DiskArrays.AbstractDiskArray) = true

end
