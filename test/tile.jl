using AutoRIFT: ImagePair, gridpoints, params, scatter, issearchable,
                chip_bounds, search_bounds, Highpass, Wallis, NoPreprocess, Deramp

@testset "filter_reach is the measured reach" begin
    # The trait exists to size a block's halo, so it has to equal what the filter actually needs
    # rather than what its window width suggests. Measured here by filtering a padded block and
    # finding the smallest pad whose interior matches a whole-image filter bit for bit.
    #
    # This is the check that catches the failure the trait was introduced for: `Wallis` subtracts a
    # local mean and then divides by a local standard deviation computed *about that mean*, so its
    # window is applied twice and its reach is twice its half-width. At `Wallis(5)` a halo of
    # `width ÷ 2` leaves 792 of 10201 interior points differing by up to 0.21 — a visibly wrong
    # filter output, not a rounding artefact.
    n = 512
    img = synthetic_texture(n; seed = 1)
    mask = trues(n, n)
    rows, cols = 200:300, 150:250

    function measured_reach(m)
        whole, = AutoRIFT.preprocess(img, mask, m)
        for pad in 0:24
            rr = (first(rows) - pad):(last(rows) + pad)
            cc = (first(cols) - pad):(last(cols) + pad)
            block, = AutoRIFT.preprocess(img[rr, cc], mask[rr, cc], m)
            inner = block[(pad + 1):(length(rr) - pad), (pad + 1):(length(cc) - pad)]
            all(isequal.(inner, whole[rows, cols])) && return pad
        end
        return -1                      # no finite reach found within the search
    end

    for w in (5, 9, 21)
        for m in (Highpass(w), Wallis(; width = w))
            @test AutoRIFT.filter_reach(m) == measured_reach(m)
        end
        # And the property that motivated the trait: these two are not the same number.
        @test AutoRIFT.filter_reach(Wallis(; width = w)) ==
              2 * AutoRIFT.filter_reach(Highpass(w))
    end

    # The derivative filters are a single convolution, so their reach is the kernel half-width.
    # `Laplacian` takes decibels first, but that step is pointwise and so adds nothing.
    for w in (5, 9)
        for m in (Sobel(; width = w), Laplacian(; width = w))
            @test AutoRIFT.filter_reach(m) == measured_reach(m)
        end
    end

    # Windowless methods reach nothing.
    @test AutoRIFT.filter_reach(NoPreprocess()) == 0
    @test AutoRIFT.filter_reach(Decibel()) == 0

    # `Deramp` estimates from the whole image, so no halo makes a block agree with the scene. A
    # negative reach says "not blockwise-reproducible" rather than naming a large number that would
    # merely be less wrong.
    @test AutoRIFT.filter_reach(Deramp()) < 0

    @testset "WallisGapfill reaches far past its window, but finitely" begin
        # Its reach is set by a *decision* rather than by a window: an invalid pixel is an interior
        # gap only if real data lies within `GAPFILL_REACH`, so influence travels that far plus the
        # dilation and the Wallis window. The declared reach must bound that, because a halo that is
        # too small makes a blocked run silently wrong.
        #
        # The gap below is deliberately deeper than `2 * GAPFILL_REACH`, so its centre is beyond
        # reach of data while its edges are within: the threshold then genuinely discriminates
        # across the probe region. A benign layout measures a reach of zero and passes vacuously,
        # which is the trap this construction avoids.
        m = WallisGapfill(; width = 5)
        reach = AutoRIFT.filter_reach(m)
        @test reach > AutoRIFT.filter_reach(Wallis(; width = 5))
        @test reach >= AutoRIFT.GAPFILL_REACH

        gn = 400
        gimg = synthetic_texture(gn; seed = 3) .* 2000.0f0 .+ 500.0f0
        gmask = trues(gn, gn)
        deep = 2 * ceil(Int, AutoRIFT.GAPFILL_REACH) + 20
        gmask[100:(100 + deep), :] .= false

        # Probe rows straddling the verdict boundary, which sits `GAPFILL_REACH` into the gap:
        # nearer the edge the pixel is filled, deeper in it is excluded.
        boundary = 100 + ceil(Int, AutoRIFT.GAPFILL_REACH)
        prow = (boundary - 5):(boundary + 5)
        pcol = 200:220
        whole_out, whole_v = AutoRIFT.wallis_gapfill(gimg, gmask, 5, 0.25;
                                                    rng = Random.Xoshiro(1))
        # The layout has teeth only if both verdicts appear in the probe.
        @test any(whole_v[prow, pcol])
        @test !all(whole_v[prow, pcol])

        # The validity verdict is what the halo has to reproduce; the filled values themselves are
        # drawn from the generator and so cannot match a differently-sized block's draw.
        function reach_of_verdict()
            for pad in 0:(reach + 20)
                rr = (first(prow) - pad):(last(prow) + pad)
                cc = (first(pcol) - pad):(last(pcol) + pad)
                (first(rr) < 1 || first(cc) < 1 || last(rr) > gn || last(cc) > gn) && return -1
                _, bv = AutoRIFT.wallis_gapfill(gimg[rr, cc], gmask[rr, cc], 5, 0.25;
                                                rng = Random.Xoshiro(1))
                inner = bv[(pad + 1):(length(rr) - pad), (pad + 1):(length(cc) - pad)]
                inner == whole_v[prow, pcol] && return pad
            end
            return -1
        end
        measured = reach_of_verdict()
        @test measured >= 0                       # a finite reach exists at all
        @test measured <= reach                   # and the declared value bounds it
    end
    grid = gridpoints((n, n), 32; chip_size = 32, search_radius = 25)
    @test_throws "no halo fixes that" AutoRIFT.halo(
        grid, params(; preprocess = :deramp, similarity = :coherence), (n, n))
end

@testset "halo is the sum of the reaches, not the largest" begin
    # Each stage acts on the previous one's output, so the correlation reach and the filter's
    # half-width compose. A halo taking the maximum would leave every pixel within
    # `filter_width ÷ 2` of the read edge filtered over a truncated window.
    n = 512
    grid = gridpoints((n, n), 32; chip_size = 32, search_radius = 25)

    # `_pass_geometry`'s pad is chip/2 + radius + ceil(prior) plus the reference's 2-pixel slack.
    # Measured on the widest level's points rather than on `grid`: a level sets its own chip size,
    # so the grid's is not what the reach is built from. Holding `chip_size_max` at the grid's own
    # chip size is what keeps this testset about the *filter* term.
    p0 = params(; preprocess = :none, chip_size = 32, chip_size_max = 32, search_radius = 25)
    pad = AutoRIFT._pass_geometry(AutoRIFT._worst_level_points(grid, p0), (n, n))[5]
    @test AutoRIFT.halo(grid, p0, (n, n)) == pad

    # Same geometry throughout, varying only the filter, so each difference from `pad` is the
    # filter's reach and nothing else.
    filtered(m, w) = AutoRIFT.halo(grid, params(; preprocess = m, filter_width = w,
                                                chip_size = 32, chip_size_max = 32,
                                                search_radius = 25), (n, n))

    # A 5-wide highpass adds 2 in each direction; a 21-wide one adds 10.
    @test filtered(:highpass, 5) == (pad[1] + 2, pad[2] + 2)
    @test filtered(:highpass, 21) == (pad[1] + 10, pad[2] + 10)

    # And `Wallis` adds twice that, because its window is applied twice. Taking `filter_width ÷ 2`
    # here would under-halo every block by half the filter's reach.
    @test filtered(:wallis, 5) == (pad[1] + 4, pad[2] + 4)
    @test filtered(:wallis, 21) == (pad[1] + 20, pad[2] + 20)

    # A wider search radius widens the halo through the correlation term.
    wide = gridpoints((n, n), 32; chip_size = 32, search_radius = 40)
    @test all(AutoRIFT.halo(wide, params(), (n, n)) .>
              AutoRIFT.halo(grid, params(), (n, n)))
end

@testset "halo covers every level, not the grid as supplied" begin
    # A level sets its own chip size and floors the radii, so the grid's own fields are not what
    # any pass runs. Measured against what each level actually asks `_pass_geometry` for: a halo
    # short of that reads too little, and the answers along a block's internal edges are wrong
    # rather than missing.
    n = 1024
    imagesize = (n, n)

    function widest_level_pad(grid, p)
        worst = (0, 0)
        for csx in AutoRIFT.chip_sizes(p)
            lp = AutoRIFT._level_points(grid, p, csx, AutoRIFT.chip_size_y(p, csx),
                                        trues(size(grid)))
            pad = AutoRIFT._pass_geometry(scatter(lp), imagesize)[5]
            worst = (max(worst[1], pad[1]), max(worst[2], pad[2]))
        end
        return worst
    end

    # `chip_size_max` sets the reach however the caller sized the grid. A grid built at the base
    # chip size against a 4x larger maximum is the case that breaks: its own geometry implies a
    # halo 46 px per axis short of what the coarsest level reaches.
    p = params(; chip_size = 32, chip_size_max = 128, grid_spacing = 32, search_radius = 25)
    for gridchip in (32, 64, 128)
        grid = gridpoints(imagesize, 32; chip_size = gridchip, search_radius = 25)
        w = AutoRIFT.filter_reach(p.preprocess)
        need = widest_level_pad(grid, p) .+ w
        @test all(AutoRIFT.halo(grid, p, imagesize) .>= need)
    end

    # `sanitize!` floors a searched point's radius at `min_search_radius`, so a grid whose radii
    # sit below the floor is searched wider than it asks for.
    tight = params(; chip_size = 32, chip_size_max = 32, grid_spacing = 32,
                   search_radius = 3, min_search_radius = 6)
    grid = gridpoints(imagesize, 32; chip_size = 32, search_radius = 3)
    @test all(AutoRIFT.halo(grid, tight, imagesize) .>=
              widest_level_pad(grid, tight) .+ AutoRIFT.filter_reach(tight.preprocess))

    # The grid `autorift` builds for itself already carries the largest chip size, so nothing above
    # widens it — the correction is for caller-supplied grids.
    pd = params()
    gd = AutoRIFT._build_grid(imagesize, pd)
    @test AutoRIFT.halo(gd, pd, imagesize) ==
          AutoRIFT._pass_geometry(scatter(gd), imagesize)[5] .+
          AutoRIFT.filter_reach(pd.preprocess)
end

@testset "blocks partition the grid and read what they need" begin
    n = 1024
    p = params(; chip_size = 32, search_radius = 25)
    grid = gridpoints((n, n), 32; chip_size = 32, search_radius = 25)
    nr, nc = size(grid)
    flat = scatter(grid)
    lin = LinearIndices((nr, nc))

    # In pixels, at `grid_spacing = 32`: sizes that divide the grid evenly, sizes that do not, and
    # sizes covering the whole grid at once.
    for bs in ((256, 256), (224, 160), (128, 352), (32nr, 32nc), (64nr, 64nc))
        layout = AutoRIFT.block_layout(grid, p, (n, n), bs)
        @test length(layout) >= 1

        # Every grid point is written by exactly one block. This is what makes assembly a copy:
        # no point computed twice, none left out, so there is nothing to blend or average.
        written = zeros(Int, nr, nc)
        for b in layout.blocks
            written[b.grid_rows, b.grid_cols] .+= 1
        end
        @test all(==(1), written)

        for b in layout.blocks
            # Read windows are clipped to the scene, never outside it.
            @test first(b.read_rows) >= 1 && last(b.read_rows) <= n
            @test first(b.read_cols) >= 1 && last(b.read_cols) <= n

            # And the halo is *sufficient*: every pixel a point in this block would read in an
            # untiled run lies inside the block's read window. Both windows matter — the chip
            # comes from the secondary and is offset by the prior, the search window from the
            # reference — and an under-computed halo shows up here rather than as a wrong answer
            # much later. Clipped at the scene edge, since an untiled run cannot read there either.
            for j in b.grid_cols, i in b.grid_rows
                idx = lin[i, j]
                issearchable(flat, idx) || continue
                crows, ccols = chip_bounds(flat, idx)
                srows, scols = search_bounds(flat, idx)
                for (rows, cols) in ((crows, ccols), (srows, scols))
                    @test max(first(rows), 1) >= first(b.read_rows)
                    @test min(last(rows), n) <= last(b.read_rows)
                    @test max(first(cols), 1) >= first(b.read_cols)
                    @test min(last(cols), n) <= last(b.read_cols)
                end
            end
        end
    end

    # One block covering the whole grid writes every point, which is what makes the single-block
    # case the untiled computation.
    #
    # Its read window is the grid's own span grown by the halo — *not* the whole scene, because
    # `gridpoints` insets by a margin so no point sits at the image edge (`src/points.jl:244`).
    # The window still contains every pixel any point reads, which the loop above asserted; a
    # block that read the full scene would merely read more than it needs.
    whole = only(AutoRIFT.block_layout(grid, p, (n, n), (n, n)).blocks)
    @test whole.grid_rows == 1:nr && whole.grid_cols == 1:nc
    hx, hy = AutoRIFT.halo(grid, p, (n, n))
    r0 = max(floor(Int, minimum(grid.y)) - hy, 1)
    r1 = min(ceil(Int, maximum(grid.y)) + hy, n)
    c0 = max(floor(Int, minimum(grid.x)) - hx, 1)
    c1 = min(ceil(Int, maximum(grid.x)) + hx, n)
    @test whole.read_rows == r0:r1
    @test whole.read_cols == c0:c1
end

@testset "a block smaller than its halo is an error" begin
    n = 512
    p = params(; chip_size = 32, search_radius = 25)
    grid = gridpoints((n, n), 32; chip_size = 32, search_radius = 25)

    # Such a block is almost entirely overlap, so it defeats the purpose rather than merely
    # costing something. Named as a configuration error, with the minimum stated, rather than
    # silently widened — a caller who asked for a block size to bound memory needs to know the
    # request could not be honoured.
    @test_throws "smaller than the" AutoRIFT.block_layout(grid, p, (n, n), (16, 16))
    @test_throws "Use at least" AutoRIFT.block_layout(grid, p, (n, n), (16, 16))

    @test_throws "must be positive in both axes" AutoRIFT.block_layout(grid, p, (n, n), (0, 8))
    @test_throws "must be positive in both axes" AutoRIFT.block_layout(grid, p, (n, n), (8, -1))
end

@testset "a blocked run equals an untiled one" begin
    # The gate the whole feature rests on. Blocking changes where the work happens, not what is
    # computed, so anything short of equality on every field is a defect rather than a tolerance to
    # widen. Everything below shares one process, so FFTW wisdom is fixed across the comparison.
    # Large enough, at this spacing, that the coarse grid clears the outlier filter's window — so
    # the coarse gate, its dilation and its resample are all genuinely exercised rather than
    # short-circuited by the small-grid fallback.
    n = 1024
    ref, sec = split_pair(n)

    # The feature seam sits at pixel 512, so a block boundary at half the grid's width falls on the
    # discontinuity — the case an under-computed halo fails and a boundary in flat texture would not.
    base = params(; chip_size = 32, chip_size_max = 32, grid_spacing = 16, search_radius = 12)
    pair0 = AutoRIFT._prepare(ImagePair(ref, sec), base)
    grid0 = AutoRIFT._build_grid(size(pair0), base)
    @test size(grid0) == (61, 61)
    # The coarse grid really does clear the filter's window, so the gate is exercised.
    @test !isnothing(AutoRIFT._coarse_points(
        AutoRIFT._level_points(grid0, base, 32, 32, trues(size(grid0))), base, 32, 32))

    for (tag, p) in ("default" => base,
                     "wallis" => params(; chip_size = 32, chip_size_max = 32, grid_spacing = 16,
                                        search_radius = 12, preprocess = :wallis,
                                        filter_width = 5),
                     "multi-level" => params(; chip_size = 32, chip_size_max = 64,
                                             grid_spacing = 16, search_radius = 12),
                     "threaded" => params(; chip_size = 32, chip_size_max = 32, grid_spacing = 16,
                                          search_radius = 12, threaded = true))
        # Each driver takes the pair it is defined on: `correlate_multichip` a filtered scene, and
        # `correlate_tiled` the raw one, which it filters per block. That asymmetry is the feature —
        # a blocked run never forms a filtered scene — so the comparison has to respect it.
        raw = ImagePair(ref, sec)
        pair = AutoRIFT._prepare(raw, p)
        grid = AutoRIFT._build_grid(size(raw), p)
        AutoRIFT._warm_grid_plans(grid, p)
        untiled = AutoRIFT.correlate_multichip(pair, grid, p)

        # In pixels, at `grid_spacing = 16` over a 61-point grid. 496 columns puts a boundary at the
        # feature seam; the next two do not divide the grid evenly; one block covering everything must
        # agree with the whole-scene path trivially; and 128 pixels makes 64 blocks, many more than
        # there are threads — the case that catches a threaded run sharing one buffer set across them.
        for bs in ((976, 496), (976, 400), (320, 320), (208, 704), (976, 976), (128, 128))
            assert_same_result(untiled, AutoRIFT.correlate_tiled(raw, grid, p, bs),
                               "$tag, block $bs")
        end
    end
end

@testset "a blocked run equals an untiled one on a caller-supplied grid" begin
    # The production path: a grid the caller built, at a chip size below `chip_size_max`. The halo
    # has to be sized by what the levels run rather than by what this grid carries, so this is the
    # configuration that fails when it is not — and the one the earlier verification missed.
    n = 1024
    ref, sec = split_pair(n)
    p = params(; chip_size = 32, chip_size_max = 128, grid_spacing = 16, search_radius = 12)
    raw = ImagePair(ref, sec)
    pair = AutoRIFT._prepare(raw, p)

    # Built at the base chip size, not at `chip_size_max`: the halo must cover chip 128 anyway,
    # because that is what the coarsest level runs these points at.
    grid = gridpoints(size(raw), 16; chip_size = 32, search_radius = 12)
    AutoRIFT._warm_grid_plans(grid, p)
    untiled = AutoRIFT.correlate_multichip(pair, grid, p)
    for bs in ((480, 240), (272, 272))
        assert_same_result(untiled, AutoRIFT.correlate_tiled(raw, grid, p, bs),
                           "caller grid, block $bs")
    end
end

@testset "a coarse grid too small to judge falls back, and still agrees" begin
    # Below the relaxed filter's window there is no evidence to restrict the fine search with, so
    # both paths search everything. The tiled path cannot substitute a policy of its own here: it
    # has to agree with the untiled run point for point, which is what this asserts. It warns,
    # because searching every point at full radius is what the coarse pass exists to avoid.
    n = 320
    a, b = shifted_pair((n, n), (2.0, -1.0); seed = 5)
    p = params(; chip_size = 32, chip_size_max = 32, grid_spacing = 32,
               search_radius = 12, coarse_stride = 4)
    raw = ImagePair(a, b)
    pair = AutoRIFT._prepare(raw, p)
    grid = AutoRIFT._build_grid(size(raw), p)

    # The configuration this testset is about: the coarse grid really is too small.
    level = AutoRIFT._level_points(grid, p, 32, 32, trues(size(grid)))
    @test isnothing(AutoRIFT._coarse_points(level, p, 32, 32))

    AutoRIFT._warm_grid_plans(grid, p)
    untiled = AutoRIFT.correlate_multichip(pair, grid, p)
    tiled = @test_logs (:warn, r"coarse grid smaller") match_mode = :any begin
        AutoRIFT.correlate_tiled(raw, grid, p, (128, 128))
    end
    assert_same_result(untiled, tiled, "coarse-grid fallback")
end

@testset "_read_block materializes a window" begin
    a = reshape(collect(1:100), 10, 10)
    b = AutoRIFT._read_block(a, 3:6, 2:5)
    @test b == a[3:6, 2:5]
    # A copy, not a view: the correlator's inner loop wants contiguous memory, and mutating the
    # block must not reach back into the caller's scene.
    @test b !== a
    b[1, 1] = -1
    @test a[3, 2] != -1
end

@testset "a threaded blocked run is deterministic and bounded" begin
    # `_run_blocked` spawns `min(nblocks, nthreads)` tasks, each holding one buffer set and claiming
    # blocks from a shared counter until they run out. Two things can break that, and both are
    # invisible in a single run:
    #
    #   * a buffer set shared between tasks rather than per task, which corrupts whichever blocks a
    #     task claims after its first — the failure a `Core.Box` produces when the set is assigned
    #     inside the spawned closure instead of in a function of its own;
    #   * dynamic claiming changing the answer, which it cannot, since each block writes a disjoint
    #     slice of the output.
    #
    # Asserted as a result rather than as an allocation strategy, and repeated, because a race shows
    # up as run-to-run variation even where one run happens to agree.
    n = 1024
    ref, sec = split_pair(n)
    p = params(; chip_size = 32, chip_size_max = 32, grid_spacing = 16, search_radius = 12,
               threaded = true)
    raw = ImagePair(ref, sec)
    grid = AutoRIFT._build_grid(size(raw), p)
    AutoRIFT._warm_grid_plans(grid, p)
    untiled = AutoRIFT.correlate_multichip(AutoRIFT._prepare(raw, p), grid, p)

    # More blocks than threads on any ordinary machine, so tasks genuinely claim repeatedly.
    for bs in ((320, 320), (128, 128))
        first_run = AutoRIFT.correlate_tiled(raw, grid, p, bs)
        assert_same_result(untiled, first_run, "threaded determinism, block $bs")
        # Repeated: a race shows up as run-to-run variation even where one run happens to agree.
        for _ in 1:2
            again = AutoRIFT.correlate_tiled(raw, grid, p, bs)
            @test isequal(again.dx, first_run.dx)
            @test isequal(again.dy, first_run.dy)
            @test isequal(again.correlation, first_run.correlation)
        end
    end

    # One block and one task: the degenerate end of `min(nblocks, nthreads)`, which must still
    # agree rather than taking a different path.
    nr, nc = size(grid)
    assert_same_result(untiled, AutoRIFT.correlate_tiled(raw, grid, p, (16nr, 16nc)),
                       "threaded, single block")
end

@testset "block size is in pixels, and snaps outward to whole grid points" begin
    # The unit is pixels because pixels are what blocking bounds: a block's output is a few KiB
    # against several MiB of imagery, and the whole scene's output grid is smaller than one block's
    # read window. Grid points would name the wrong quantity, and their meaning would depend on
    # `grid_spacing` — the trap that made `benchmark/memory.jl` measure a single block twice.
    n = 2048
    p = params(; chip_size = 32, chip_size_max = 32, grid_spacing = 32, search_radius = 12)
    grid = AutoRIFT._build_grid((n, n), p)

    for px in (256, 512, 1024)
        layout = AutoRIFT.block_layout(grid, p, (n, n), (px, px))
        # No block writes more than the requested extent.
        for b in layout.blocks
            rlo, rhi = AutoRIFT._pixel_span(grid.y, b.grid_rows, b.grid_cols)
            clo, chi = AutoRIFT._pixel_span(grid.x, b.grid_rows, b.grid_cols)
            @test rhi - rlo <= px
            @test chi - clo <= px
        end
        # Doubling the request quarters the block count, which is what makes the unit an extent
        # rather than a count of anything.
        bigger = AutoRIFT.block_layout(grid, p, (n, n), (2px, 2px))
        @test length(bigger) < length(layout)
    end

    # A request at the scene's own size is one block, and a request larger than the scene still is.
    @test length(AutoRIFT.block_layout(grid, p, (n, n), (n, n))) == 1
    @test length(AutoRIFT.block_layout(grid, p, (n, n), (4n, 4n))) == 1

    # The halo comparison is now like against like, both in pixels, and names the pixel minimum.
    @test_throws "pixels is smaller than the" AutoRIFT.block_layout(grid, p, (n, n), (8, 8))

    # A caller-supplied grid need not be uniformly spaced, so the conversion walks the coordinates
    # rather than dividing by `grid_spacing`. Here the spacing rises from 40 to 400 pixels along the
    # axis, and no single `grid_spacing` describes it.
    xs = Float64[100, 140, 180, 220, 400, 600, 900, 1300]
    uneven = gridpoints(xs, xs; chip_size = 32, search_radius = 12)
    ulayout = AutoRIFT.block_layout(uneven, p, (2000, 2000), (512, 512))
    for b in ulayout.blocks
        clo, chi = AutoRIFT._pixel_span(uneven.x, b.grid_rows, b.grid_cols)
        @test chi - clo <= 512
    end
    # And it packs by extent rather than shrinking every block to fit the widest gap: the closely
    # spaced points share a block instead of each taking one.
    @test maximum(length(b.grid_cols) for b in ulayout.blocks) > 1
end
