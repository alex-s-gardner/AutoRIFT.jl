# A lazy image's values, computed once per tile and kept on disk.
#
# **What this is for.** A blocked correlation sweeps every block once per pass — a coarse and a fine sweep
# per chip-size level — so a whole run reads 3.4-4.4x the scene where one pass's windows sum to 0.66-0.96x
# (`src/api.jl:262`). Where those values are *derived* rather than read, the multiplier lands on the
# derivation: a coregistered secondary is resampled four times over, and a NISAR granule's samples are
# decompressed four times over.
#
# **Why not a memory cache.** The access order is a cyclic scan, so a tile's next use is a whole sweep
# away and any cache smaller than the touched set has evicted it by then; `memory.md` replays the real
# request sequence through LRU and random replacement and finds no interior optimum. `cache_budget`
# therefore chooses between holding the *whole* pair and holding none of it — and a NISAR pair is 23 GB
# (L1) to 48 GB (L2) of `Float32`, which is not a choice.
#
# **What this does instead.** Persist each tile the first time it is asked for. The first pass pays the
# resample or the decompression; every later pass reads an uncompressed tile back. Resident memory stays
# proportional to the window being served rather than to the scene, because the tiles live in a scratch
# file and are read with ordinary `read`/`write` rather than mapped — the kernel's page cache holds them
# outside the process's own footprint, where it can reclaim them under pressure.
#
# **Tiles rather than windows, because the windows change.** Each pass reads a block grown by that level's
# halo, so no two passes ask for the same rectangle. A cache keyed on the request would miss every time;
# one keyed on a fixed tile grid hits whatever the request's shape.

"""
    TileCache(parent; tile = 512, dir) <: AbstractMatrix{Float32}

`parent`'s values, computed a tile at a time and cached in a scratch file.

Reads the same values `parent` would return — assert it on a window if you want that checked; nothing
about the tiling reaches the result, since a tile is filled by asking `parent` for exactly that tile.

`tile` is the side of the cache's grid in pixels. It wants to be at least the chunk or burst granularity
of whatever `parent` reads, so that filling one tile is one read of the source rather than a fraction of
one, and small enough that a request's tile-aligned footprint is not much larger than the request.

The file is deleted by [`close`](@ref). It is sparse: only the tiles a run touches are ever written, so a
correlation over part of a scene costs that part.
"""
struct TileCache{P<:AbstractMatrix} <: AbstractMatrix{Float32}
    parent::P
    tile::Int
    dims::Tuple{Int,Int}
    ntiles::Tuple{Int,Int}
    io::IOStream
    path::String
    # `Vector{Bool}`, not `BitVector`: the presence marks are written from whichever task derived the
    # tile, and a `BitVector` packs 64 of them into one word, so two tiles marked at once are a
    # read-modify-write race that silently loses one. A byte per tile is 12 KB on the largest grid here.
    present::Vector{Bool}
    # **A lock per tile, and one more for the file.** Deriving a tile is a resample or a decompression —
    # seconds of work in the large — so holding a single lock across it serializes the very thing a
    # blocked run parallelizes. The per-tile lock is what stops two tasks deriving the same tile and makes
    # the second wait for the first; the file lock is held only around a seek and a transfer, because an
    # `IOStream`'s position is shared state. Taken in that order always — tile, then file — so there is no
    # cycle to deadlock on.
    tiles::Vector{ReentrantLock}
    io_lock::ReentrantLock
    # Atomic because tiles now fill concurrently: these are diagnostics, but a lost count would
    # misreport the reuse the cache is here to deliver.
    filled::Threads.Atomic{Int}
    served::Threads.Atomic{Int}
end

function TileCache(parent::AbstractMatrix; tile::Integer = 512,
                   dir::AbstractString = mktempdir(; cleanup = false))
    tile > 0 || throw(ArgumentError("`tile` must be positive, got $tile"))
    dims = size(parent)
    nt = (cld(dims[1], tile), cld(dims[2], tile))
    mkpath(dir)
    path = joinpath(dir, "tilecache_$(dims[1])x$(dims[2])_$(tile).bin")
    io = open(path, "w+")
    return TileCache(parent, Int(tile), dims, nt, io, path, zeros(Bool, prod(nt)),
                     [ReentrantLock() for _ in 1:prod(nt)], ReentrantLock(),
                     Threads.Atomic{Int}(0), Threads.Atomic{Int}(0))
end

Base.size(c::TileCache) = c.dims

"""
    close(c::TileCache)

Close the scratch file and delete it.
"""
function Base.close(c::TileCache)
    close(c.io)
    isfile(c.path) && rm(c.path)
    return nothing
end

# Bytes per tile, and where a tile starts. Every tile is stored at full size even where the image ends
# inside it, so the offset is arithmetic rather than a lookup; the file is sparse, so the padding of the
# last row and column of tiles occupies nothing.
_tile_bytes(c::TileCache) = c.tile * c.tile * sizeof(Float32)
_tile_offset(c::TileCache, ti::Integer, tj::Integer) =
    ((tj - 1) * c.ntiles[1] + (ti - 1)) * _tile_bytes(c)

# The tile indices covering an index range.
_tile_span(c::TileCache, r::AbstractUnitRange) = (fld(first(r) - 1, c.tile) + 1):(fld(last(r) - 1, c.tile) + 1)

# One tile's rows and columns, clipped to the image.
function _tile_extent(c::TileCache, ti::Integer, tj::Integer)
    rows = ((ti - 1) * c.tile + 1):min(ti * c.tile, c.dims[1])
    cols = ((tj - 1) * c.tile + 1):min(tj * c.tile, c.dims[2])
    return (rows, cols)
end

# A tile's values: from the file when it is already there, from `parent` when it is not — and written on
# the way through, so the next pass reads it back.
#
# **The derivation happens outside the file lock**, so two tiles are derived at once while only their
# transfers serialize. The tile's own lock is held across the whole of its fill, which is what makes a
# second asker wait for the first rather than derive the same tile again.
function _tile!(buf::Matrix{Float32}, c::TileCache, ti::Integer, tj::Integer)
    k = (tj - 1) * c.ntiles[1] + ti
    @lock c.tiles[k] begin
        if c.present[k]
            @lock c.io_lock begin
                seek(c.io, _tile_offset(c, ti, tj))
                read!(c.io, buf)
            end
            Threads.atomic_add!(c.served, 1)
        else
            rows, cols = _tile_extent(c, ti, tj)
            fill!(buf, 0.0f0)
            # `view` into the buffer: an edge tile is narrower than the grid, and the padding stays zero.
            copyto!(view(buf, 1:length(rows), 1:length(cols)), c.parent[rows, cols])
            @lock c.io_lock begin
                seek(c.io, _tile_offset(c, ti, tj))
                write(c.io, buf)
            end
            c.present[k] = true
            Threads.atomic_add!(c.filled, 1)
        end
    end
    return buf
end

function Base.getindex(c::TileCache, rows::AbstractUnitRange, cols::AbstractUnitRange)
    checkbounds(c, rows, cols)
    out = Matrix{Float32}(undef, length(rows), length(cols))
    buf = Matrix{Float32}(undef, c.tile, c.tile)
    for tj in _tile_span(c, cols), ti in _tile_span(c, rows)
        _tile!(buf, c, ti, tj)
        trows, tcols = _tile_extent(c, ti, tj)
        # The part of this tile the request wants, in the tile's own indices and in the output's.
        wr = intersect(rows, trows)
        wc = intersect(cols, tcols)
        (isempty(wr) || isempty(wc)) && continue
        copyto!(view(out, (wr .- first(rows) .+ 1), (wc .- first(cols) .+ 1)),
                view(buf, (wr .- first(trows) .+ 1), (wc .- first(tcols) .+ 1)))
    end
    return out
end

Base.getindex(c::TileCache, i::Integer, j::Integer) = c[i:i, j:j][1, 1]
Base.getindex(c::TileCache, rows::AbstractUnitRange, j::Integer) = c[rows, j:j][:, 1]
Base.getindex(c::TileCache, i::Integer, cols::AbstractUnitRange) = c[i:i, cols][1, :]

# Reading one of these costs I/O, which is what it is for. A blocked run therefore windows its reads, and
# the automatic cache budget will decline to hold a scene this size in memory.
AutoRIFT.ondisk(::TileCache) = true

"""
    cache_report(c::TileCache) -> String

How many tiles were derived and how many were served from the file.

The ratio is what the cache bought: a run that swept the grid `n` times reads `n` tiles for every one it
derives, so `served / filled` approaching `2 * levels - 1` is the multiplicity removed.
"""
cache_report(c::TileCache) =
    string(c.filled[], " tiles derived, ", c.served[], " served from disk (",
           c.filled[] == 0 ? "-" : string(round(c.served[] / c.filled[]; digits = 2)), "x reuse), ",
           round(c.filled[] * _tile_bytes(c) / 2^30; digits = 2), " GiB written")
