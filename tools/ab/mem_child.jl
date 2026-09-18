# One configuration, correlated in a fresh process with its resident memory traced.
#
# Run by `mem_blocks.jl`, one process per block size — for the reason `benchmark/memory.jl` records
# at length: peak memory is a high-water mark, so two configurations measured in one process both
# report the larger. Sampling the trace does not change that, since the pages one configuration
# touched are still resident when the next begins.
#
#   julia --project=tools/ab -t 10,1 tools/ab/mem_child.jl <workdir> <block> <threads> <out.jls>
#
# `block` is the block size in **pixels**, or 0 for one block over the whole scene. The interactive
# thread in `-t N,1` is required: the sampler runs there so it keeps sampling while the correlation
# saturates the default pool.

using Mmap: Mmap
using Printf
using AutoRIFT
using AutoRIFT: params, halo, gridpoints, block_layout

include(joinpath(@__DIR__, "memtrace.jl"))

# The ITS_LIVE optical configuration, and the one `bench_table.jl` measures — so a figure here is
# comparable to that table rather than to a different setting that happens to be nearby.
const CHIP, CHIP_MAX, SPACING, RADIUS, UPSAMPLING = 16, 64, 8, 20, 16

# Profile buffer slots. A whole-scene run at 2 ms on twelve threads records a few million; this is
# sized well above that so the buffer does not fill and stop recording partway through.
const PROFILE_SLOTS = 60_000_000

# ---------------------------------------------------------------------------
# A counting, memory-mapped plane
# ---------------------------------------------------------------------------
#
# The input has to be lazy for the measurement to mean anything: a resident `Matrix` is 1.08 GiB per
# plane that every configuration pays equally, which is a constant added to both sides of the
# comparison and swamps what blocking saves. `Mmap` is the cheapest lazy input there is — the array
# is an ordinary `Matrix{Float32}`, so nothing in the package takes a different path — but it is
# *too* transparent to count reads through, and the read count is what places a trace sample against
# the run's progress.
#
# So the mapping is wrapped. Only `getindex` over two ranges is specialized, because that is the one
# operation a block read performs (`AutoRIFT._read_block!` indexes with the window rather than
# `copyto!`-ing a view, deliberately — see its comment). Everything else falls through to `Base`,
# which is correct and, for a scalar access, no slower than the bare mapping.
struct CountingPlane{T} <: AbstractMatrix{T}
    parent::Matrix{T}
    reads::Threads.Atomic{Int}
    pixels::Threads.Atomic{Int}
end

CountingPlane(a::Matrix{T}) where {T} =
    CountingPlane{T}(a, Threads.Atomic{Int}(0), Threads.Atomic{Int}(0))

Base.size(a::CountingPlane) = size(a.parent)
Base.IndexStyle(::Type{<:CountingPlane}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(a::CountingPlane, i::Int) = a.parent[i]

Base.@propagate_inbounds function Base.getindex(a::CountingPlane, rows::AbstractUnitRange,
                                                cols::AbstractUnitRange)
    Threads.atomic_add!(a.reads, 1)
    Threads.atomic_add!(a.pixels, length(rows) * length(cols))
    return a.parent[rows, cols]
end

# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

function main()
    length(ARGS) == 4 || error("usage: mem_child.jl <workdir> <block-pixels> <threads> <out.jls>")
    work, block, nthreads, out = ARGS[1], parse(Int, ARGS[2]), parse(Int, ARGS[3]), ARGS[4]

    nr, nc = Tuple(parse.(Int, split(read(joinpath(work, "dims.txt"), String))))
    # Mapped rather than read, and left open: closing the file invalidates the mapping, and the
    # process exits when the run does.
    ref = CountingPlane(Mmap.mmap(open(joinpath(work, "ref.bin"), "r"), Matrix{Float32}, (nr, nc)))
    sec = CountingPlane(Mmap.mmap(open(joinpath(work, "sec.bin"), "r"), Matrix{Float32}, (nr, nc)))

    p = params(; chip_size = CHIP, chip_size_max = CHIP_MAX, grid_spacing = SPACING,
               search_radius = RADIUS, preprocess = :highpass, filter_width = 5,
               upsampling = UPSAMPLING, threaded = nthreads > 1)
    bs = block == 0 ? nothing : (block, block)

    # The layout, described before the run rather than after: a block size that cannot work is an
    # error here, and the block count is what the parent's progress bar counts against.
    grid = gridpoints((nr, nc), SPACING; chip_size = CHIP_MAX, search_radius = RADIUS)
    h = halo(grid, p, (nr, nc))
    nblocks, maxread = if block == 0
        1, (nr, nc)
    else
        L = block_layout(grid, p, (nr, nc), bs)
        length(L.blocks), (maximum(length(b.read_rows) for b in L.blocks),
                           maximum(length(b.read_cols) for b in L.blocks))
    end
    @printf("READY blocks=%d maxread=%dx%d halo=%dx%d grid=%dx%d\n",
            nblocks, maxread[1], maxread[2], h.X, h.Y, size(grid)...)
    flush(stdout)

    # The CPU profiler, not `@profile_walltime` — see `peak_stacks`, which records the measurement:
    # the wall-time profiler samples tasks rather than threads, so idle scheduler tasks crowd out the
    # correlation and its samples carry no running/sleeping flag to filter on.
    #
    # 2 ms, not the 1 ms default, and a buffer sized for the whole run rather than trimmed afterwards.
    # A profiler that fills its buffer stops recording, which would lose precisely the late samples a
    # peak tends to sit in; `profile_overflow` below reports it if it happens anyway.
    Profile.init(; n = PROFILE_SLOTS, delay = 0.002)

    total_reads() = ref.reads[] + sec.reads[]
    # Progress against imagery read rather than against blocks, because the read count is what this
    # process can observe from outside the driver and blocks-completed is not. It is reported as a
    # fraction of the scene: each level reads every block's window, so the total crosses 1.0 once per
    # level and the figure says how much imagery has moved rather than how far through the run it is.
    scene_pixels = 2 * nr * nc
    progress = function (trace, reads)
        isempty(trace.resident) && return
        @printf(stderr, "\r  read %5.2f scenes (%d windows)   now %7.0f MiB   peak %7.0f MiB   ", (ref.pixels[] + sec.pixels[]) / scene_pixels, reads, last(trace.footprint) / 2^20, maximum(trace.footprint) / 2^20)
        flush(stderr)
    end

    result, trace, seconds = with_trace(; interval = 0.004, progress, reads = total_reads) do
        Profile.@profile begin
            if block == 0
                autorift(ref, sec, p)
            else
                autorift(ref, sec, p, bs)
            end
        end
    end
    println(stderr)

    # Attribution at the peak, from the profile, before the profile buffer is dropped.
    #
    # A window rather than the single peak sample: the profiler samples on its own timer, so the
    # instant the trace peaked need not have a stack at all. The window is the interval over which
    # the trace sat within 1% of its peak, which is where the memory was actually held.
    ipk = argmax(trace.footprint)
    thresh = 0.99 * trace.footprint[ipk]
    lo = something(findprev(<(thresh), trace.footprint, ipk), 0) + 1
    hi = something(findnext(<(thresh), trace.footprint, ipk), length(trace) + 1) - 1
    data = Profile.fetch(include_meta = true)
    stacks = peak_stacks(data, trace.tick[lo], trace.tick[hi])
    # A profile that filled its buffer stops recording, which would silently make a late peak
    # unattributable. Reported so the figure is not read as "nothing was running".
    overflow = length(data) >= PROFILE_SLOTS - 10_000

    measured = count(!isnan, result.dx)
    meta = (; block, nthreads, nblocks, maxread, halo = (h.X, h.Y), grid = size(grid),
            scene = (nr, nc), seconds, measured,
            reads = total_reads(), read_pixels = ref.pixels[] + sec.pixels[],
            peak_resident = maximum(trace.resident), peak_footprint = maximum(trace.footprint),
            peak_live = maximum(trace.live), final_maxrss = Sys.maxrss(),
            peak_index = ipk, peak_window = (lo, hi), stacks, profile_overflow = overflow,
            chip = CHIP, chip_max = CHIP_MAX, spacing = SPACING, radius = RADIUS,
            upsampling = UPSAMPLING)
    write_trace(out, trace, meta)

    @printf("DONE block=%d blocks=%d %.1f s peak_footprint=%.0f MiB peak_resident=%.0f MiB peak_live=%.0f MiB maxrss=%.0f MiB reads=%d readamp=%.2f measured=%d samples=%d\n",
            block, nblocks, seconds, meta.peak_footprint / 2^20, meta.peak_resident / 2^20,
            meta.peak_live / 2^20, meta.final_maxrss / 2^20, meta.reads,
            meta.read_pixels / (nr * nc * 2), measured, length(trace))
    for (label, n) in first(stacks, 5)
        @printf("  %5.1f%%  %s\n", 100 * n / max(1, sum(last, stacks)), label)
    end
    return nothing
end

main()
