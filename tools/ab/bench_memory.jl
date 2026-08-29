# Peak resident memory: AutoRIFT.jl against the reference, on the A/B window.
#
# `benchmark/memory.jl` measures the same quantity on synthetic texture and documents why it has to
# be done in subprocesses -- `Sys.maxrss()` is a high-water mark, so two configurations in one
# process report the first one's peak twice. That rule is followed here, and the reference side
# (`bench_memory.py`) follows it too.
#
# Every figure is a *difference*: a process that loads, reads the window and configures the run, and
# one that additionally correlates. Subtracting isolates the pipeline from the runtime floor, which
# is not comparable across the two languages and is not what either implementation controls.
#
#   julia --project=tools/ab tools/ab/bench_memory.jl [npix]

using Printf

const HERE = @__DIR__
const NPIX = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 3072
const CACHE = get(ENV, "AUTORIFT_TESTDATA", expanduser("~/data/autorift/tests"))
# The two images alone, as raw `Float32`. The measured processes read *this*, not `window.jls`.
#
# Deserializing the 756 MiB cache inside a measured process sets a peak of ~1.2 GiB that has nothing
# to do with the pipeline: the transient dominates, the subtraction then reports only the amount by
# which the pipeline exceeded it, and the blocked path — whose whole purpose is a lower peak — comes
# out looking worse than the unblocked one. Reading a bare pair costs `2 * npix^2 * 4` bytes and
# nothing else.
const PAIR = joinpath(HERE, "bench_pair_$(NPIX).bin")

# The inputs and configuration, shared by the baseline and the working run so the subtraction
# cancels the imagery exactly.
const SETUP = """
    using AutoRIFT
    using AutoRIFT: params, autorift, init, autorift!, reinit!
    n = $NPIX
    raw = read("$PAIR")
    ref = Matrix{Float32}(reshape(reinterpret(Float32, raw)[1:(n * n)], n, n))
    sec = Matrix{Float32}(reshape(reinterpret(Float32, raw)[(n * n + 1):(2 * n * n)], n, n))
    raw = nothing; GC.gc()
    kw = (; chip_size = (X = 16, Y = 16), chip_size_max = (X = 64, Y = 64),
          grid_spacing = (X = 8, Y = 8), search_radius = (X = 20, Y = 20),
          preprocess = :highpass, filter_width = 5, subpixel = :pyramid, upsampling = 16,
          threaded = Threads.nthreads() > 1)
    p = params(; kw...)
"""

# Cut the pair out of the cache once, in *this* process, so no measured one pays for it.
function write_pair()
    isfile(PAIR) && filesize(PAIR) == 2 * NPIX^2 * 4 && return
    @eval using Serialization
    w = Base.invokelatest(deserialize, joinpath(CACHE, "window.jls"))
    n = NPIX
    r0 = clamp(1678 - n ÷ 2, 1, size(w.reference, 1) - n + 1)
    c0 = clamp(2495 - n ÷ 2, 1, size(w.reference, 2) - n + 1)
    open(PAIR, "w") do io
        write(io, Matrix{Float32}(w.reference[r0:(r0 + n - 1), c0:(c0 + n - 1)]))
        write(io, Matrix{Float32}(w.secondary[r0:(r0 + n - 1), c0:(c0 + n - 1)]))
    end
    return
end

# Peak RSS and live heap of a fresh process. Both, because they answer different questions: peak is
# what a memory limit sees and never falls, live is what is still reachable. Reporting only peak
# would read allocator slack as accumulation.
function measure(body::AbstractString; threads = "auto")
    src = SETUP * body * """
        GC.gc()
        print(Sys.maxrss() / 2^20, " ", Base.gc_live_bytes() / 2^20)
    """
    out = read(`$(Base.julia_cmd()) --startup-file=no --project=$HERE -t $threads -e $src`, String)
    peak, live = parse.(Float64, split(strip(out)))
    return (; peak, live)
end

function python(args::Vector{String})
    cmd = `micromamba run -n arift-ref python $(joinpath(HERE, "bench_memory.py")) $NPIX $args`
    out = read(pipeline(cmd; stderr = devnull), String)
    # The script prints one line; anything else on stdout is a warning to be skipped.
    line = last(filter(!isempty, split(strip(out), '\n')))
    peak, live = parse.(Float64, split(strip(line)))
    return (; peak, live)
end

function main()
    @printf("window %d x %d, %d threads\n\n", NPIX, NPIX, Threads.nthreads())

    base = measure("")
    oneshot = measure("autorift(ref, sec, p)\n")
    # The blocked path, which exists to bound peak by the block rather than the scene. Included
    # because it is the lever a caller has and the reference has no equivalent of.
    #
    # `process_block_size` is in **pixels** (`src/api.jl`, `block_layout`), snapped outward to a whole
    # number of grid points.
    blocked = measure("""
        c = init(ref, sec; kw..., process_block_size = (1024, 1024))
        autorift!(c)
    """)
    # A second pair through one `Cache`: what a batch driver pays after the first, where buffers and
    # FFT plans are already resident.
    cached = measure("""
        c = init(ref, sec; kw...)
        autorift!(c)
        reinit!(c; reference = ref, secondary = sec)
        autorift!(c)
    """)

    pbase = python(["--baseline"])
    pwork = python(String[])

    @printf("%-34s %10s %10s\n", "", "peak MiB", "live MiB")
    @printf("%-34s %10.1f %10.1f\n", "julia baseline (load + inputs)", base.peak, base.live)
    @printf("%-34s %10.1f %10.1f\n", "julia one-shot", oneshot.peak, oneshot.live)
    @printf("%-34s %10.1f %10.1f\n", "julia blocked 1024 px", blocked.peak, blocked.live)
    @printf("%-34s %10.1f %10.1f\n", "julia two pairs, one Cache", cached.peak, cached.live)
    @printf("%-34s %10.1f %10.1f\n", "python baseline (load + inputs)", pbase.peak, pbase.live)
    @printf("%-34s %10.1f %10.1f\n", "python runAutorift", pwork.peak, pwork.live)

    # Total process peak is the figure to compare, and the above-baseline delta is the one that
    # misleads: each baseline already holds its own copy of the two scenes, so subtracting charges
    # the pipeline only for what it needed *beyond* the imagery and hides the imagery itself. A
    # memory limit sees the total. Both are reported, since the delta is what says where the cost is.
    println("\ntotal process peak — this is the comparison:")
    @printf("  julia one-shot        %8.1f MiB\n", oneshot.peak)
    @printf("  julia blocked 1024 px %8.1f MiB  (%.2fx the one-shot)\n",
            blocked.peak, blocked.peak / oneshot.peak)
    @printf("  python runAutorift    %8.1f MiB\n", pwork.peak)
    @printf("  ratio python/julia    %8.2fx one-shot, %.2fx blocked\n",
            pwork.peak / oneshot.peak, pwork.peak / blocked.peak)

    println("\nabove each side's own baseline (where the cost sits, not what a limit sees):")
    jo = oneshot.peak - base.peak
    jb = blocked.peak - base.peak
    jc = cached.peak - base.peak
    pw = pwork.peak - pbase.peak
    @printf("  julia one-shot        %8.1f MiB\n", jo)
    @printf("  julia blocked 1024 px %8.1f MiB\n", jb)
    @printf("  julia two pairs       %8.1f MiB\n", jc)
    @printf("  python runAutorift    %8.1f MiB\n", pw)

    # Live heap is the retention question: growth here across two pairs would mean a long batch run
    # needs process recycling, where peak growth alone is allocator slack.
    println("\nlive heap after collection (retention, not slack):")
    @printf("  julia one-shot %.1f -> two pairs %.1f MiB (%+.1f)\n",
            oneshot.live, cached.live, cached.live - oneshot.live)
    @printf("  python %.1f MiB\n", pwork.live)
    println("\nfloors (not comparable across languages, reported for completeness):")
    @printf("  julia %.1f MiB   python %.1f MiB\n", base.peak, pbase.peak)
end

main()
