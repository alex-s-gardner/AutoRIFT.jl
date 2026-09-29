# Peak memory against allocation churn: which of the two a change moves, and what it costs.
#
# `mem_blocks.jl` sweeps block size. This sweeps the other three axes that peak turns out to depend
# on — the collector's heap target, the task count, and the redundant copies a block read makes — one
# arm per process, because peak is a high-water mark and two arms in one process both report the
# larger.
#
#   julia --project=tools/ab -t 10,1 tools/ab/mem_churn.jl <workdir> <block> <label> [options]
#
#     --preprocess=none|highpass   filter; `none` is what the golden runs use
#     --with-prepare-copy          restore `_prepare_block`'s fallthrough to `_prepare` for `NoPreprocess`
#     --with-read-temp             restore `_read_window!`'s block-sized temporary on a strided parent
#     --radius=N --chip-max=N --spacing=N --window=N
#                                  a wide-halo geometry on an N-pixel sub-window, which is the shape a
#                                  radar granule has and the optical defaults do not: the read window
#                                  is then set by the halo rather than by the block, which is the
#                                  regime where the task count moves peak
#
# The heap target is set on the process rather than here: pass `--heap-size-hint=800M` to `julia`.
#
# The two `--with-*` options put back the per-block copies `src/` no longer makes, so the A/B that
# removed them stays reproducible against the current baseline. `dev/plan-16gib.md` has the table.

using Mmap: Mmap
using Printf
using AutoRIFT
using AutoRIFT: params, BlockBuffers, ImagePair, Params, NoPreprocess

include(joinpath(@__DIR__, "memtrace.jl"))

const CHIP, CHIP_MAX, SPACING, RADIUS, UPSAMPLING = 16, 64, 8, 20, 16

# A textured, fully covered region of the plane: its corners are fill and its centre carries a fifth
# the high-pass texture of a good window, as `bench_scene.jl` records. A sub-window arm starts here.
const ORIGIN = (1537, 4609)

hasopt(name) = any(startswith("--$name"), ARGS)
optvalue(name, default) = (i = findfirst(startswith("--$name="), ARGS);
                           isnothing(i) ? default : split(ARGS[i], '=')[2])

# `_prepare` for `NoPreprocess` is `copy(img), copy(mask)` in `preprocess` and a second `copy` pair in
# `replace_nonfinite` — four window-sized arrays per image per block, reproducing what `_block_pair!`
# has just written into the buffers. `src/tile.jl` substitutes the non-finite values in place instead.
if hasopt("with-prepare-copy")
    AutoRIFT._prepare_block(::BlockBuffers, raw::ImagePair, p::Params, ::NoPreprocess,
                            ::Int, ::Int) = AutoRIFT._prepare(raw, p)
end

# The generic `_read_window!` materializes `img[rows, cols]` and then copies it in, because a view of a
# *lazy* array would be read one pixel at a time. `src/tile.jl` takes the view for a strided parent,
# which has nothing to defer; this puts the temporary back.
if hasopt("with-read-temp")
    AutoRIFT._read_window!(dest::AbstractMatrix, img::StridedMatrix, rows, cols) =
        (copyto!(dest, img[rows, cols]); dest)
end

function main()
    work, block, label = ARGS[1], parse(Int, ARGS[2]), ARGS[3]
    preprocess = Symbol(optvalue("preprocess", "none"))
    radius = parse(Int, optvalue("radius", string(RADIUS)))
    chipmax = parse(Int, optvalue("chip-max", string(CHIP_MAX)))
    spacing = parse(Int, optvalue("spacing", string(SPACING)))
    window = parse(Int, optvalue("window", "0"))

    nr, nc = Tuple(parse.(Int, split(read(joinpath(work, "dims.txt"), String))))
    ref = Mmap.mmap(open(joinpath(work, "ref.bin"), "r"), Matrix{Float32}, (nr, nc))
    sec = Mmap.mmap(open(joinpath(work, "sec.bin"), "r"), Matrix{Float32}, (nr, nc))
    if window > 0
        rows = ORIGIN[1]:(ORIGIN[1] + window - 1)
        cols = ORIGIN[2]:(ORIGIN[2] + window - 1)
        ref, sec = ref[rows, cols], sec[rows, cols]
    end

    chip = window > 0 ? spacing : CHIP
    p = params(; chip_size = chip, chip_size_max = chipmax, grid_spacing = spacing,
               search_radius = radius, preprocess, filter_width = 5, upsampling = UPSAMPLING,
               threaded = Threads.nthreads() > 1)
    bs = block == 0 ? nothing : (block, block)

    # Warmed so the arm carries neither the planner nor the compiler. Cold plans cost more than
    # anything measured here — see `docs/src/explanation/memory.md`.
    let n = min(1500, minimum(size(ref)) ÷ 2)
        a, b = ref[1:n, 1:n], sec[1:n, 1:n]
        isnothing(bs) ? autorift(a, b, p) : autorift(a, b, p, bs)
    end
    GC.gc(true)
    GC.gc(true)

    before = Base.gc_num()
    out, trace, seconds = with_trace(; interval = 0.01) do
        isnothing(bs) ? autorift(ref, sec, p) : autorift(ref, sec, p, bs)
    end
    after = Base.gc_num()

    # `dx` over raw bits, so `-0.0` and every `NaN` payload counts — a change that is meant to be
    # answer-preserving has to be checked on the bits rather than on a point count.
    checksum = string(hash(reinterpret(UInt32, vec(out.dx))), base = 16)
    h = AutoRIFT.halo(p)
    @printf("RESULT %-22s threads=%2d block=%5d halo=%dx%d peak=%.3f live=%.3f alloc=%6.1f gc=%5.2f wall=%5.1f points=%7d dxsum=%s\n",
            label, Threads.nthreads(), block, h.X, h.Y,
            maximum(trace.footprint) / 2^30, maximum(trace.live) / 2^30,
            (Base.gc_total_bytes(after) - Base.gc_total_bytes(before)) / 2^30,
            (after.total_time - before.total_time) / 1e9, seconds,
            count(isfinite, out.dx), checksum)
end

main()
