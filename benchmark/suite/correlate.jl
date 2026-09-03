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
#   * Quantifying subpixel refinement, which **does** dominate: 78 us of a 97 us point at chip 16
#     and radius 20, the geometry ITS_LIVE runs Landsat at, and fixed in chip size because the
#     cascade upsamples a 5x5 patch to 320x320 whatever the surface was.
#
#     The composite interpolant is what that invites, and it holds up. `pyrup!` is a separable
#     *linear* map, so `log2(up)` doublings compose into one fixed `(patch*up) x patch` matrix `K`
#     — depending only on the patch size and the upsampling, hence the same for every point — and
#     the cascade is `K·P·K'`. Each output column then needs a 5-vector held in registers, so the
#     upsampled surface is never materialised at all. Prototyped on the device: **5.4x** on the
#     refinement stage and scratch from 1048 MB to 0.109 MB per 1024-point tile, at 4.2e-7 from
#     `pyrup!` with identical displacements over 400 real surfaces. Not yet implemented on either
#     path; `docs/gpu.md` records the measurement.
#
#   * Baselining the integral images, against a global table per pass. **Measured and rejected** —
#     5.42 ms against 5.28 ms, i.e. slightly slower, because a global table costs one pass per
#     chip-size level whether that level searches 900 points or none. `src/integral.jl` carries the
#     numbers and the two plausible explanations that turned out wrong.

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
    # The four-argument form, with the scratch buffer and the tap table supplied — which is what
    # `subpixel_peak` calls at `src/peak.jl:505`. The two-argument convenience overload defaults
    # both, so benchmarking *it* measured a `scratch` allocation and a freshly built tap table per
    # call: 672 bytes at n=5 rising to 70800 at n=80, growing with the work, on a kernel the
    # zero-allocation gate covers. That is allocation the production path does not do.
    for n in (5, 10, 20, 40, 80)
        src = bench_texture((n, n); seed = n)
        dst = Matrix{Float32}(undef, 2n, 2n)
        scratch = Matrix{Float32}(undef, 2n, n)
        taps = AutoRIFT._pyrup_row_taps(n)
        g["pyrup $(n)x$(n)"] = @benchmarkable AutoRIFT.pyrup!($dst, $src, $scratch, $taps)
    end

    # Integral images: the denominator's entire cost.
    for n in (81, 113, 177)
        src = bench_texture((n, n); seed = n)
        S = Matrix{Float64}(undef, n + 1, n + 1)
        g["integral $(n)x$(n)"] = @benchmarkable AutoRIFT.integral!($S, $src)
        g["integral_sq $(n)x$(n)"] = @benchmarkable AutoRIFT.integral_sq!($S, $src)
    end
end
