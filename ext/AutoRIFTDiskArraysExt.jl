"""
    AutoRIFTDiskArraysExt

Teach [`AutoRIFT.ondisk`](@ref) to recognize a chunked backend, so an unblocked run refuses input it
would read a pixel at a time, `AutoRIFT._chunk_rows` the backend's chunk height, so a threaded
read splits on a boundary the storage shares, and `AutoRIFT._read_block!` to read a window straight
into the buffer it was given.

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

# A window read straight into `dest`, with no block-sized temporary.
#
# The core's method is `copyto!(dest, img[rows, cols])`, and the copy is deliberate there: a `view`
# defers the read, so `copyto!` over one walks a lazy array element by element — one I/O call per pixel,
# measured at 444 s against 0.6 s for a single 512² block. Indexing asks for the whole window instead,
# which is the operation a chunked backend is built to serve, and pays a block-sized temporary for it.
#
# `getindex_disk!` is that same indexing path — the one `img[rows, cols]` itself calls — given somewhere
# to put the result. So this keeps one aligned read per touched chunk and drops the temporary: measured
# on a chunked GeoTIFF window, **1,912,496 bytes allocated against 2,352**. That matters because a
# blocked run reads every block once per pass and the volume scales with read amplification, which
# reaches 21.6x on a wide-halo granule — a sweep that allocated 1727 GiB against an untiled 225 GiB,
# and the mechanism behind the one block size measured as allocation-bound rather than compute-bound.
#
# `dest` is a view of a reused buffer and need not be contiguous, which a backend writing through
# `readblock!` could have refused. Verified on a GDAL-backed raster, on the lazy `BroadcastDiskArray` a
# nodata raster becomes, and on a strided view of a larger buffer: all three read correctly.
AutoRIFT._read_window!(dest::AbstractMatrix, img::DiskArrays.AbstractDiskArray, rows, cols) =
    (DiskArrays.getindex_disk!(dest, img, rows, cols); dest)

end
