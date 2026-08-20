# Sliding-window reductions. Populated at M3.
#
# Eight call sites in the multi-chip-size search, each sweeping the whole grid, so these are a
# meaningful fraction of per-level cost. The Python reference materializes a
# (window^2 x npoints) matrix and grows its output with repeated concatenation,
# which is quadratic; a 10-50x improvement is expected from doing neither.
#
# Measured per reduction (max, min, mean, median, range, MAD, agreement count)
# and per window width. Width matters more than it looks: the search dilates
# masks with windows up to 48 wide, where an O(window^2) implementation is
# hopeless and the O(1)-per-pixel decompositions in ImageMorphology earn their
# dependency.
let g = addgroup!(SUITE, "window")
    # 20% NaN, which is realistic: the coarse pass deliberately invalidates most of the
    # grid, so a benchmark on dense data would measure the wrong thing.
    for n in (512,)
        a = bench_texture((n, n); seed = 5)
        rng = Random.MersenneTwister(6)
        a[rand(rng, n * n) .< 0.2] .= NaN32

        # Extrema at every width the search uses. The point of the monotone deque is
        # that these stay flat as the width grows -- an O(w^2) implementation is ~150x
        # slower at width 48, which is where the mask dilations live.
        for w in (3, 5, 12, 48)
            g["max w$w $(n)x$(n)"] = @benchmarkable AutoRIFT.windowmax($a, $w)
        end
        for w in (3, 5, 12)
            g["mean w$w $(n)x$(n)"] = @benchmarkable AutoRIFT.windowmean($a, $w)
        end
        # Median and MAD only ever see small windows, where a sort beats a running
        # structure.
        for w in (3, 5)
            g["median w$w $(n)x$(n)"] = @benchmarkable AutoRIFT.windowmedian($a, $w)
            g["mad w$w $(n)x$(n)"] = @benchmarkable AutoRIFT.windowmad($a, $w)
        end
        g["count_agreeing w5 $(n)x$(n)"] =
            @benchmarkable AutoRIFT.count_agreeing($a, 5, 0.2f0)
    end

    # The outlier filter, which is what the reductions exist to serve: two stages over
    # the whole grid, iterated. Sized like a real chip-size level rather than an image.
    for n in (256, 512)
        dx = bench_texture((n, n); seed = 7) .* 4.0f0
        dy = bench_texture((n, n); seed = 8) .* 4.0f0
        rx = fill(25, n, n)
        ry = fill(25, n, n)
        v = trues(n, n)
        f = AutoRIFT.outlier_filter()
        g["reject_outliers $(n)x$(n)"] =
            @benchmarkable AutoRIFT.reject_outliers($dx, $dy, $rx, $ry, $v, 64, $f)
    end
end
