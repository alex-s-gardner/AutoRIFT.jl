# Point-set construction and geometry.
#
# These look trivial, and per call they are, but `chip_bounds`/`search_bounds` run
# once per search center per pyramid level — hundreds of thousands of times per
# image pair, tens of millions of times per batch. They must stay allocation-free
# and inline; a regression here is a regression everywhere.

let g = addgroup!(SUITE, "points")

    # Construction. Gridded is the common case; scattered matters for the
    # arbitrary-search-center path.
    for n in (1_000, 100_000)
        xs = collect(range(100.0, 1000.0; length = n))
        ys = copy(xs)
        g["pointset scattered n=$n"] = @benchmarkable AutoRIFT.pointset($xs, $ys;
            chip_size = 32, search_radius = 25)
    end

    for sz in (512, 4096)
        g["gridpoints $(sz)x$(sz)"] = @benchmarkable AutoRIFT.gridpoints(
            ($sz, $sz), 32; chip_size = 32, search_radius = 25)
    end

    pts = AutoRIFT.gridpoints((1024, 1024), 32; chip_size = 32, search_radius = 25)

    # Must be O(1) in the number of points: `vec` shares the underlying data, so
    # the only allocation is the eight reshaped wrappers and the struct itself —
    # a fixed ~1.3 kB regardless of grid size. Benchmarked at two sizes so that a
    # regression to copying shows up as the cost scaling with point count.
    g["scatter 1024x1024"] = @benchmarkable AutoRIFT.scatter($pts)
    g["scatter 4096x4096"] = @benchmarkable AutoRIFT.scatter(p) setup = (
        p = AutoRIFT.gridpoints((4096, 4096), 32; chip_size = 32, search_radius = 25))

    # Per-point geometry, the functions the inner loop calls.
    g["chip_bounds"] = @benchmarkable AutoRIFT.chip_bounds($pts, 1)
    g["search_bounds"] = @benchmarkable AutoRIFT.search_bounds($pts, 1)
    g["surface_size"] = @benchmarkable AutoRIFT.surface_size($pts, 1)

    # Whole-grid sweeps, run once per pyramid level.
    g["nsearchable 1024x1024"] = @benchmarkable AutoRIFT.nsearchable($pts)
    g["sanitize! 1024x1024"] = @benchmarkable AutoRIFT.sanitize!(p, 6) setup = (
        p = AutoRIFT.gridpoints((1024, 1024), 32; chip_size = 32, search_radius = 25))

    # Parameter resolution happens once per call, so it only matters relative to
    # total runtime — but for batch processing of small scenes, "once per call"
    # times ten million is not nothing.
    g["params"] = @benchmarkable AutoRIFT.params()
    g["params with methods"] = @benchmarkable AutoRIFT.params(;
        preprocess = Wallis(width = 21), subpixel = PyramidRefine(upsampling = 32))
end
