# Correlation surface, peak location, and subpixel refinement — the inner loop the
# whole package's throughput rests on.
#
# Names match `tools/python_ref/bench_python.py` exactly, so `compare.jl` can print
# a per-kernel speedup against OpenCV. A mismatch silently yields no comparison
# rather than an error, so the names are worth checking whenever either side changes.
#
# What this group is for, beyond regression tracking:
#
#   * Fixing the direct/FFT crossover. Direct evaluation costs
#     `nshifts * chip_area` multiply-accumulates; the FFT gets every shift at once
#     for `O(N log N)` in the search-window area. The crossover constant depends on
#     cache behaviour and on FFTW's plan quality, so `DIRECT_THRESHOLD` is set from
#     these numbers rather than derived from operation counts.
#
#   * Quantifying subpixel refinement. At 128x upsampling a 5x5 patch becomes
#     640x640 — 1.6 MB materialised to locate one maximum. If that dominates,
#     evaluating the composite interpolant directly is the M8 optimisation with the
#     largest expected payoff, and this is the baseline that claim is measured
#     against.
#
#   * Baselining the integral images. OpenCV rebuilds them per grid point over
#     windows that overlap ~2.5x in each direction at typical grid spacing;
#     computing one per pass over the whole padded image is O(N) instead. Also an
#     M8 item, also measured rather than assumed.

let g = addgroup!(SUITE, "correlate")

    for (chip, radius) in CHIP_RADIUS
        win = chip + 2radius - 1

        for (dname, T) in (("uint8", UInt8), ("float32", Float32))
            search = bench_texture((win, win); seed = chip + radius, T = T)
            chipdata = collect(search[(radius + 1):(radius + chip),
                                      (radius + 1):(radius + chip)])
            ws = AutoRIFT.workspace(T, chip, radius)

            g["surface zncc c$chip r$radius $dname"] =
                @benchmarkable AutoRIFT.correlate!($ws, $search, $chipdata, $radius)

            # Surface plus peak: what one grid point actually costs, and the number
            # that multiplies out to a whole-scene time.
            g["point zncc c$chip r$radius $dname"] = @benchmarkable begin
                s = AutoRIFT.correlate!($ws, $search, $chipdata, $radius)
                AutoRIFT.peak_offset(s, ($radius, $radius))
            end
        end
    end

    # Peak location alone, separated from surface formation. Cheap in absolute
    # terms, but it runs once per point per chip-size level.
    for radius in (6, 25, 50)
        surface = bench_texture((2radius, 2radius); seed = radius)
        g["peak r$radius"] = @benchmarkable AutoRIFT.peak_index($surface)
    end

    # Subpixel refinement, per upsampling factor. Not linear in the factor: each
    # step quadruples the area, so the final doubling costs as much as every step
    # before it combined.
    let surface = bench_texture((50, 50); seed = 3)
        for up in (16, 32, 64, 128)
            rw = AutoRIFT.refinement_workspace(up)
            g["subpixel x$up"] =
                @benchmarkable AutoRIFT.subpixel_peak($rw, $surface, (25, 25), $up)
        end
    end

    # One upsampling step at each size the cascade passes through, so the cascade's
    # cost can be attributed to its steps.
    for n in (5, 10, 20, 40, 80)
        src = bench_texture((n, n); seed = n)
        dst = Matrix{Float32}(undef, 2n, 2n)
        g["pyrup $(n)x$(n)"] = @benchmarkable AutoRIFT.pyrup!($dst, $src)
    end

    # Integral images: the denominator's entire cost.
    for n in (81, 113, 177)
        src = bench_texture((n, n); seed = n)
        S = Matrix{Float64}(undef, n + 1, n + 1)
        g["integral $(n)x$(n)"] = @benchmarkable AutoRIFT.integral!($S, $src)
        g["integral_sq $(n)x$(n)"] = @benchmarkable AutoRIFT.integral_sq!($S, $src)
    end
end
