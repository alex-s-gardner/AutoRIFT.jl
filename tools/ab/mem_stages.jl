# What is resident before a block is read, and what each stage adds.
#
# `mem_blocks.jl` answers "how large was the peak" and `memtrace.jl` answers "what was running when it
# was". Neither answers "what is the peak made of", and on a narrow-halo scene that is the question
# that matters: every block size below 1024 px and every heap-pressure arm lands at the same peak, so
# the peak is not set by the blocks. This reads the process's own footprint between stages instead of
# deriving it from the configuration, which is what separates the grid from the imagery from the pool.
#
#   julia --project=tools/ab -t 10,1 tools/ab/mem_stages.jl [workdir] [block]
#
# `workdir` holds the planes `bench_scene.jl` writes; it defaults to `$AUTORIFT_BENCH_DIR`.
#
# Every figure is `phys_footprint` after a full collection, so a stage's cost is what survives rather
# than what it churned — see `rusage!` for why footprint and not `resident_size`.

using Mmap: Mmap
using Printf
using AutoRIFT
using AutoRIFT: params, gridpoints, halo, block_layout, displacement_field

include(joinpath(@__DIR__, "memtrace.jl"))

const CHIP, CHIP_MAX, SPACING, RADIUS, UPSAMPLING = 16, 64, 8, 20, 16

# `preprocess = :none`, which is what every golden end-to-end run uses (`tools/golden/e2e.jl`): the
# reference filters before the correlator, so the golden comparison feeds imagery already filtered.
const PREPROCESS = :none

const BUF = zeros(UInt64, 64)

function mark(label)
    GC.gc(true)
    resident, footprint = rusage!(BUF)
    @printf("%-32s footprint %7.0f MiB   resident %7.0f MiB   live %7.0f MiB\n",
            label, footprint / 2^20, resident / 2^20, Base.gc_live_bytes() / 2^20)
end

function main()
    work = length(ARGS) >= 1 ? ARGS[1] :
           get(ENV, "AUTORIFT_BENCH_DIR", expanduser("~/data/autorift/bench"))
    block = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1024

    mark("using AutoRIFT")
    nr, nc = Tuple(parse.(Int, split(read(joinpath(work, "dims.txt"), String))))
    # Mapped and left open: closing the file invalidates the mapping, and the process exits with the
    # run. Mapped pages are clean, so they belong to `resident` and not to `footprint`.
    ref = Mmap.mmap(open(joinpath(work, "ref.bin"), "r"), Matrix{Float32}, (nr, nc))
    sec = Mmap.mmap(open(joinpath(work, "sec.bin"), "r"), Matrix{Float32}, (nr, nc))
    mark("+ planes mapped")

    p = params(; chip_size = CHIP, chip_size_max = CHIP_MAX, grid_spacing = SPACING,
               search_radius = RADIUS, preprocess = PREPROCESS, upsampling = UPSAMPLING,
               threaded = Threads.nthreads() > 1)
    grid = gridpoints((nr, nc), SPACING; chip_size = CHIP_MAX, search_radius = RADIUS)
    mark("+ grid")
    @printf("    grid %dx%d = %.2f Mpts   summarysize %.0f MiB\n",
            size(grid)..., length(grid.x) / 1e6, Base.summarysize(grid) / 2^20)

    layout = block_layout(grid, p, (nr, nc), (block, block))
    mark("+ block layout")
    maxrows = maximum(length(b.read_rows) for b in layout.blocks)
    maxcols = maximum(length(b.read_cols) for b in layout.blocks)
    @printf("    %d blocks   halo %dx%d   largest window %dx%d = %.2f Mpx\n",
            length(layout.blocks), layout.halo.X, layout.halo.Y,
            maxrows, maxcols, maxrows * maxcols / 1e6)

    field = displacement_field(grid)
    mark("+ one displacement field")
    @printf("    field summarysize %.0f MiB\n", Base.summarysize(field) / 2^20)

    # A textured, fully covered window — the plane's corners are fill and its centre carries a fifth
    # the high-pass texture of a good window, as `bench_scene.jl` records. Warming here keeps the
    # planner and the compiler out of the measured run.
    let a = ref[1537:2560, 4609:5632], b = sec[1537:2560, 4609:5632]
        autorift(a, b, p)
    end
    mark("+ warmup (plans, JIT)")

    result, trace, seconds = with_trace(; interval = 0.01) do
        autorift(ref, sec, p, (block, block))
    end
    @printf("%-32s footprint %7.0f MiB   live peak %7.0f MiB   %.1f s\n",
            "peak during the run", maximum(trace.footprint) / 2^20,
            maximum(trace.live) / 2^20, seconds)
    mark("after the run, result held")
    @printf("    result summarysize %.0f MiB   %d points\n",
            Base.summarysize(result) / 2^20, count(isfinite, result.dx))
end

main()
