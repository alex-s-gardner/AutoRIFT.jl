"""
    AutoRIFTDiskArraysExt

Teach [`AutoRIFT.ondisk`](@ref) to recognize a chunked backend, so an unblocked run refuses input it
would read a pixel at a time, and `AutoRIFT._chunk_rows` the backend's chunk height, so a threaded
read splits on a boundary the storage shares.

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

# The stored chunk height, so a slab read splits where the backend does. A chunk straddling two slabs
# is decoded by both, and for a compressed file that decode is the read.
#
# `1` for an unchunked backend, which constrains nothing: `approx_chunksize` reports the whole extent
# there, and taking that literally would collapse a threaded read to a single slab.
AutoRIFT._chunk_rows(a::DiskArrays.AbstractDiskArray) = _chunk_rows(DiskArrays.haschunks(a), a)

_chunk_rows(::DiskArrays.Chunked, a) = first(DiskArrays.approx_chunksize(DiskArrays.eachchunk(a)))
_chunk_rows(::DiskArrays.Unchunked, _a) = 1

end
