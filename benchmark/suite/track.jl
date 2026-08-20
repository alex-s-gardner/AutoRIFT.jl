# The grid loop: correlation over a whole point set.
#
# The number that matters here is grid points per second, since that is what multiplies
# out to a scene time. Reported for the serial and threaded paths separately: the two must
# agree bitwise (asserted in the tests), so any difference is purely scheduling.
#
# Also covers the sparse case, which is what the multi-chip-size search actually produces. Its coarse pass
# zeroes the search radius over most of the grid, and a skipped point costs a comparison
# where a searched one costs microseconds — so throughput on a sparse grid is a different
# measurement from throughput on a dense one, not a scaled version of it.

let g = addgroup!(SUITE, "track")

    for n in (512, 1024)
        ref = bench_texture((n, n); seed = 1)
        sec = bench_texture((n, n); seed = 2)
        pair = AutoRIFT.ImagePair(ref, sec)

        for (cs, r) in ((32, 25), (64, 25))
            pts = AutoRIFT.gridpoints((n, n), 32; chip_size = cs, search_radius = r)
            npts = AutoRIFT.npoints(pts)
            out = AutoRIFT.displacement_field(pts)

            # Integer-only, which is what the chip-size loop's coarse pass uses.
            g["coarse c$cs r$r $(n)x$(n) [$npts pts]"] = @benchmarkable AutoRIFT.track!(
                $out, $pair, $pts, $(AutoRIFT.params(; subpixel = :none)))

            # With refinement, the fine pass.
            g["fine c$cs r$r $(n)x$(n) [$npts pts]"] = @benchmarkable AutoRIFT.track!(
                $out, $pair, $pts, $(AutoRIFT.params(; upsampling = 64)))
        end

        # Threaded against serial at one representative size, to show the scheduling gain
        # without repeating the whole matrix.
        if n == 1024
            pts = AutoRIFT.gridpoints((n, n), 32; chip_size = 32, search_radius = 25)
            out = AutoRIFT.displacement_field(pts)
            g["threaded c32 r25 $(n)x$(n)"] = @benchmarkable AutoRIFT.track!(
                $out, $pair, $pts, $(AutoRIFT.params(; threaded = true)))
        end
    end

    # Sparse: 20% of points searchable, as the coarse pass leaves it. The cost should fall
    # roughly in proportion, and a regression here would mean skipped points are no longer
    # cheap.
    let n = 1024
        pair = AutoRIFT.ImagePair(bench_texture((n, n); seed = 1),
                                  bench_texture((n, n); seed = 2))
        pts = AutoRIFT.gridpoints((n, n), 32; chip_size = 32, search_radius = 25)
        rng = Random.MersenneTwister(11)
        for i in eachindex(pts)
            if rand(rng) > 0.2
                pts.radius_x[i] = 0
                pts.radius_y[i] = 0
            end
        end
        out = AutoRIFT.displacement_field(pts)
        g["sparse 20% c32 r25 $(n)x$(n)"] = @benchmarkable AutoRIFT.track!(
            $out, $pair, $pts, $(AutoRIFT.params(; subpixel = :none)))
    end
end
