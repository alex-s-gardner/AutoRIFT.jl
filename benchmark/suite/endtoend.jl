# Whole-pipeline timings on the standard scene sizes.
#
# The headline number. Everything else in the suite measures a part; this measures what a
# caller actually waits for, which is the only figure the speed goal can be stated against.
#
# Two entry points, because they are different products. `autorift` is the one-shot form and
# pays for its grid, its plans, and its filtering on every call. `init` / `reinit!` /
# `autorift!` amortizes all three across pairs, which is the regime that matters when the input
# is tens of millions of pairs — so both are tracked, and the gap between them is the value of
# the cache lifecycle.
#
# The `reinit!` case walks a *series* rather than repeating one pair. Consecutive acquisitions
# are what a real time series looks like, and it is also the case where each new reference is
# the previous secondary — so it exercises the prepared-image reuse, which repeating a single
# pair would not.
#
# Scene sizes come from `SCENE_SIZES`. 4096 is excluded here: at ~4.5 s per sample it dominates
# the suite's runtime without saying anything the 1024 case does not, since the cost is
# proportional to grid points and the grid scales with area.
#
# Not yet comparable to Python. `bench_python.py` measures the correlation kernels, the
# filters, and the resampling, but has no whole-pipeline case — so `speedup_vs_python` cannot
# be computed end-to-end until `obj.runAutorift()` is added there. The per-kernel ratios in
# `correlate/` and `preprocess/` are what the comparison currently rests on.

let g = addgroup!(SUITE, "endtoend")

    for n in (256, 1024)
        ref = bench_texture((n, n); seed = 1)
        # A uniform shift, so the correlation is coherent everywhere and the chip-size loop
        # runs its full course. An uncorrelated pair would be rejected in the coarse pass and
        # would measure the early-exit path instead of the work.
        sec = circshift(ref, (-4, 6))

        g["autorift c32 r25 $(n)x$(n)"] = @benchmarkable AutoRIFT.autorift(
            $ref, $sec; chip_size = 32, search_radius = 25, threaded = false)

        g["autorift threaded c32 r25 $(n)x$(n)"] = @benchmarkable AutoRIFT.autorift(
            $ref, $sec; chip_size = 32, search_radius = 25, threaded = true)

        # A single chip size, against the default's three. `chip_size_max` defaults to
        # `4 * chip_size`, so the default case above already spans 32/64/128 — this is the
        # comparison that isolates what the extra levels cost. They are only reached where a
        # smaller chip failed, so on a cleanly-correlating pair the difference should be small;
        # that is the point of measuring rather than assuming.
        g["autorift c32 only r25 $(n)x$(n)"] = @benchmarkable AutoRIFT.autorift(
            $ref, $sec; chip_size = 32, chip_size_max = 32, search_radius = 25,
            threaded = false)
    end

    # The batch regime: a series of consecutive acquisitions through one cache, against the
    # same series with a fresh `init` per pair. The ratio is what the cache lifecycle buys.
    let n = 512, k = 5
        base = bench_texture((n, n); seed = 3)
        series = [circshift(base, (-2i, 3i)) for i in 0:k]
        kw = (; chip_size = 32, search_radius = 25, threaded = false)

        g["series $k pairs cached $(n)x$(n)"] = @benchmarkable begin
            cache = AutoRIFT.init($series[1], $series[2]; $kw...)
            AutoRIFT.autorift!(cache)
            for i in 3:length($series)
                AutoRIFT.reinit!(cache; reference = $series[i - 1], secondary = $series[i])
                AutoRIFT.autorift!(cache)
            end
        end

        g["series $k pairs fresh $(n)x$(n)"] = @benchmarkable begin
            for i in 2:length($series)
                AutoRIFT.autorift($series[i - 1], $series[i]; $kw...)
            end
        end

        # `reinit!` alone, walking the series so that each call genuinely changes an image.
        # Measuring it against a repeated pair would report the reuse fast path only.
        cache = AutoRIFT.init(series[1], series[2]; kw...)
        AutoRIFT.autorift!(cache)
        g["reinit! series $(n)x$(n)"] = @benchmarkable begin
            for i in 3:length($series)
                AutoRIFT.reinit!($cache; reference = $series[i - 1],
                                 secondary = $series[i])
            end
        end
    end

    # A caller-supplied scattered point set, which the reference cannot express at all: it is
    # the case a sparse validation network or a set of stake positions produces.
    let n = 1024, npts = 2000
        ref = bench_texture((n, n); seed = 5)
        sec = circshift(ref, (-4, 6))
        rng = MersenneTwister(SEED)
        margin = 100
        xs = rand(rng, npts) .* (n - 2margin) .+ margin
        ys = rand(rng, npts) .* (n - 2margin) .+ margin
        pts = AutoRIFT.pointset(xs, ys; chip_size = 32, search_radius = 25)
        g["scattered $npts pts $(n)x$(n)"] = @benchmarkable AutoRIFT.autorift(
            $ref, $sec, $pts; chip_size = 32, threaded = false)
    end

    # The geospatial path against the array path on identical pixels. The extension is meant to be
    # a coordinate wrapper, so the ratio of these two is the whole claim — a `Raster` in must not
    # cost meaningfully more than its `parent`. Both are recorded rather than the ratio, so a
    # regression in either is attributable.
    let n = 1024, res = 10.0
        a = bench_texture((n, n); seed = 1)
        b = circshift(a, (-4, 6))
        x = X(Projected(0.0:res:(res * (n - 1)); order = ForwardOrdered(), span = Regular(res),
                        sampling = Intervals(Start()), crs = EPSG(3031)))
        # y decreasing: north-up, the normal GeoTIFF layout and the one that exercises the y-flip.
        y = Y(Projected((res * (n - 1)):(-res):0.0; order = ReverseOrdered(),
                        span = Regular(-res), sampling = Intervals(Start()), crs = EPSG(3031)))
        ra, rb = Raster(a, (y, x)), Raster(b, (y, x))
        kw = (; chip_size = 32, search_radius = 25, threaded = false)

        g["raster in $(n)x$(n)"] = @benchmarkable AutoRIFT.autorift($ra, $rb; $kw...)
        g["array in $(n)x$(n)"] = @benchmarkable AutoRIFT.autorift($a, $b; $kw...)
    end
end
