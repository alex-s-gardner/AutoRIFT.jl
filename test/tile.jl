using AutoRIFT: ImagePair, gridpoints, params, scatter, issearchable,
                chip_bounds, search_bounds

@testset "halo is the sum of the reaches, not the largest" begin
    # Each stage acts on the previous one's output, so the correlation reach and the filter's
    # half-width compose. A halo taking the maximum would leave every pixel within
    # `filter_width ÷ 2` of the read edge filtered over a truncated window.
    n = 512
    grid = gridpoints((n, n), 32; chip_size = 32, search_radius = 25)

    # `_pass_geometry`'s pad is chip/2 + radius + ceil(prior) plus the reference's 2-pixel slack.
    pad = AutoRIFT._pass_geometry(scatter(grid), (n, n))[5]
    @test AutoRIFT.halo(grid, params(; preprocess = :none), (n, n)) == pad

    # A 5-wide filter adds 2 in each direction; a 21-wide one adds 10.
    @test AutoRIFT.halo(grid, params(; preprocess = :highpass, filter_width = 5), (n, n)) ==
          (pad[1] + 2, pad[2] + 2)
    @test AutoRIFT.halo(grid, params(; preprocess = :highpass, filter_width = 21), (n, n)) ==
          (pad[1] + 10, pad[2] + 10)

    # A wider search radius widens the halo through the correlation term.
    wide = gridpoints((n, n), 32; chip_size = 32, search_radius = 40)
    @test all(AutoRIFT.halo(wide, params(), (n, n)) .>
              AutoRIFT.halo(grid, params(), (n, n)))
end

@testset "blocks partition the grid and read what they need" begin
    n = 1024
    p = params(; chip_size = 32, search_radius = 25)
    grid = gridpoints((n, n), 32; chip_size = 32, search_radius = 25)
    nr, nc = size(grid)
    flat = scatter(grid)
    lin = LinearIndices((nr, nc))

    # Sizes that divide evenly, sizes that do not, and a size covering the whole grid at once.
    for bs in ((8, 8), (7, 5), (3, 11), (nr, nc), (2nr, 2nc))
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
    whole = only(AutoRIFT.block_layout(grid, p, (n, n), (nr, nc)).blocks)
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
    @test_throws "smaller than the" AutoRIFT.block_layout(grid, p, (n, n), (1, 1))
    @test_throws "Use at least" AutoRIFT.block_layout(grid, p, (n, n), (1, 1))

    @test_throws "must be positive in both axes" AutoRIFT.block_layout(grid, p, (n, n), (0, 8))
    @test_throws "must be positive in both axes" AutoRIFT.block_layout(grid, p, (n, n), (8, -1))
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
