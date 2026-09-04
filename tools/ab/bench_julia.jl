# Wall-clock for the whole pipeline, AutoRIFT.jl side, on the A/B window.
#
# The package's own `benchmark/` suite times kernels against OpenCV primitives; this times the
# end-to-end run against the reference's end-to-end run, which `benchmark/README.md` records as the
# gap that could not be measured without the reference installed. It now is -- see tools/ab/README.md.
#
# Reports the minimum of `--reps` runs. Minimum rather than mean because interference can only make
# a sample slower, so it is the least noisy estimate of what the code costs -- the same rule
# `benchmark/compare.jl` gates on.
#
#   julia --project=tools/ab -t auto tools/ab/bench_julia.jl [npix] [reps]
#
# `-t auto` matters: without it the threaded path is silently measured as serial. The run reports
# the thread count so a comparison cannot quietly compare 1 thread against 12.

using AutoRIFT, Serialization, Statistics, Printf
using AutoRIFT: params, autorift, init, autorift!, reinit!

const NPIX = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 3072
const REPS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
const CACHE = get(ENV, "AUTORIFT_TESTDATA", expanduser("~/data/autorift/tests"))
const OUT = joinpath(@__DIR__, "bench_julia.txt")

function main()
    w = deserialize(joinpath(CACHE, "window.jls"))
    n = NPIX
    r0 = clamp(1678 - n ÷ 2, 1, size(w.reference, 1) - n + 1)
    c0 = clamp(2495 - n ÷ 2, 1, size(w.reference, 2) - n + 1)
    ref = Matrix{Float32}(w.reference[r0:(r0 + n - 1), c0:(c0 + n - 1)])
    sec = Matrix{Float32}(w.secondary[r0:(r0 + n - 1), c0:(c0 + n - 1)])

    kw = (; chip_size = (X = 16, Y = 16), chip_size_max = (X = 64, Y = 64),
          grid_spacing = (X = 8, Y = 8), search_radius = (X = 20, Y = 20),
          preprocess = :highpass, filter_width = 5, subpixel = :pyramid, upsampling = 16,
          threaded = Threads.nthreads() > 1)
    p = params(; kw...)

    @printf("julia %s, %d threads, %d x %d window\n",
            VERSION, Threads.nthreads(), n, n)

    # One untimed run first: it pays JIT compilation and FFTW planning, neither of which the
    # reference's timing includes (its kernels are precompiled C++ and its plans are OpenCV's).
    # Timing the first call would report Julia's compiler, not the pipeline.
    @printf("warmup (compile + plan) ... ")
    tw = @elapsed autorift(ref, sec, p)
    @printf("%.2f s\n", tw)

    # Two shapes, because they answer different questions. `oneshot` is what a user calling
    # `autorift` once pays, including preprocessing and buffer allocation, and is the number
    # comparable to the reference's `runAutorift`. `cached` reuses a `Cache`, which is what a batch
    # driver over millions of pairs actually pays, and has no counterpart in the reference -- it has
    # no batch entry point.
    times = Float64[]
    for _ in 1:REPS
        push!(times, @elapsed autorift(ref, sec, p))
    end

    cache = init(ref, sec; kw...)
    autorift!(cache)                      # prime the cache's own buffers
    ctimes = Float64[]
    for _ in 1:REPS
        reinit!(cache; reference = ref, secondary = sec)
        push!(ctimes, @elapsed autorift!(cache))
    end

    out = autorift(ref, sec, p)
    measured = count(!isnan, out.dx)

    @printf("\none-shot  min %.3f s   median %.3f s   (n=%d)\n",
            minimum(times), median(times), REPS)
    @printf("cached    min %.3f s   median %.3f s   (n=%d)\n",
            minimum(ctimes), median(ctimes), REPS)
    @printf("grid %s = %d points, %d measured (%.1f%%)\n",
            string(size(out.dx)), length(out.dx), measured,
            100 * measured / length(out.dx))
    @printf("throughput %.0f points/s (one-shot min)\n", length(out.dx) / minimum(times))

    open(OUT, "w") do io
        println(io, "npix ", n)
        println(io, "threads ", Threads.nthreads())
        println(io, "julia_version ", VERSION)
        println(io, "grid ", size(out.dx, 1), "x", size(out.dx, 2))
        println(io, "points ", length(out.dx))
        println(io, "measured ", measured)
        println(io, "warmup_s ", tw)
        println(io, "oneshot_min_s ", minimum(times))
        println(io, "oneshot_median_s ", median(times))
        println(io, "cached_min_s ", minimum(ctimes))
        println(io, "cached_median_s ", median(ctimes))
        println(io, "reps ", REPS)
    end
    println("wrote ", OUT)
end

main()
