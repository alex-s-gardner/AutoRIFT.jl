using AutoRIFT: ImagePair, gridpoints, params, correlate_multichip, chipsize_level,
                MultichipResult, nmeasured, resample, resample!, Nearest, Area, Bicubic,
                dilate_within, windowmax

# Same convention as track.jl's tests: the correlator returns secondary-to-reference, so the
# feature motion is its negative.
motion(r) = (-r.dx, -r.dy)

med(v) = (s = sort(collect(v)); isempty(s) ? NaN : s[(length(s) + 1) ÷ 2])

# A scene with a genuinely featureless band. This is what the pyramid exists for: a chip that
# fits inside the band has zero variance and cannot correlate, while a chip large enough to
# straddle its edge still sees real texture.
#
# Note that merely *low contrast* does not defeat a small chip — ZNCC normalises each window
# by its own scale, so a 50x contrast reduction is invisible to it. The texture has to be
# absent, not faint.
function banded_pair(n, shift, band)
    ref, sec = shifted_pair(n, shift; T = Float32)
    ref[band, :] .= 0.5f0
    sec[band, :] .= 0.5f0
    return ImagePair(ref, sec)
end

@testset "resample: nearest" begin
    a = Float32[1 2; 3 4]
    # Upsampling replicates; downsampling picks.
    @test resample(a, (4, 4), Nearest()) ==
          Float32[1 1 2 2; 1 1 2 2; 3 3 4 4; 3 3 4 4]
    @test size(resample(a, (1, 1), Nearest())) == (1, 1)
    # A mask must survive unchanged in value: nearest never invents an intermediate.
    m = Float32[1 0; 0 1]
    @test sort(unique(resample(m, (6, 6), Nearest()))) == Float32[0, 1]
end

@testset "resample: area" begin
    a = Float32[1 2; 3 4]
    @test resample(a, (1, 1), Area())[1] ≈ 2.5f0        # the mean
    # Averaging down a constant field is that constant, which is the property that keeps a
    # coarse prior unbiased.
    @test all(≈(7.0f0), resample(fill(7.0f0, 8, 8), (2, 2), Area()))

    # NaN is skipped and the weights renormalised, so a partly-missing cell yields the mean
    # of what is present rather than being dragged toward zero.
    b = Float32[1 NaN; 3 4]
    @test resample(b, (1, 1), Area())[1] ≈ Float32((1 + 3 + 4) / 3)
    @test all(isnan, resample(fill(NaN32, 4, 4), (2, 2), Area()))
end

@testset "resample: bicubic" begin
    # Catmull-Rom interpolates through its samples, so resampling to the same size is the
    # identity — which is what keeps an already-known coarse estimate unchanged.
    a = Float32[i + 2j for i in 1:8, j in 1:8]
    @test resample(a, (8, 8), Bicubic()) ≈ a atol = 1e-4

    # A linear ramp stays linear under upsampling: no overshoot on smooth data.
    r = resample(Float32[Float32(j) for i in 1:4, j in 1:8], (8, 16), Bicubic())
    @test all(isfinite, r)
    @test issorted(r[4, :])

    # Mostly-missing neighbourhoods yield NaN rather than a confident value extrapolated
    # from one corner.
    b = fill(NaN32, 8, 8)
    b[1, 1] = 1.0f0
    out = resample(b, (16, 16), Bicubic())
    @test isnan(out[end, end])
end

@testset "dilate_within" begin
    m = falses(11, 11)
    m[6, 6] = true
    d = dilate_within(m, 3)
    # Euclidean, not chessboard: the corner of the 3x3 box is at distance sqrt(18) > 3.
    @test d[6, 6]
    @test d[6, 9]           # exactly 3 away
    @test !d[6, 10]
    @test !d[9, 9]          # distance sqrt(18) ~ 4.24
    @test count(d) == count(i -> (Tuple(i)[1] - 6)^2 + (Tuple(i)[2] - 6)^2 <= 9,
                            CartesianIndices(m))

    @test !any(dilate_within(falses(5, 5), 3))       # empty stays empty
    @test all(dilate_within(trues(5, 5), 1))
end

@testset "exact recovery through the pyramid" begin
    # End to end: the whole multi-scale machinery must not degrade what a single level gets
    # right. Every point resolved, at the finest chip, with the exact shift.
    for (sx, sy) in ((0, 0), (5, -3), (11, 7))
        ref, sec = shifted_pair(512, (sx, sy); T = Float32)
        pair = ImagePair(ref, sec)
        grid = gridpoints((512, 512), 32; chip_size = 32, search_radius = 25)
        r = correlate_multichip(pair, grid, params(; chip_size = 32, chip_size_max = 128))

        @test nmeasured(r) == length(r)
        mx, my = motion(r)
        @test med(filter(!isnan, mx)) ≈ sx atol = 0.05
        @test med(filter(!isnan, my)) ≈ sy atol = 0.05
        # A fully-textured scene is resolved by the finest chip everywhere, so the coarser
        # levels are never needed — which is the behaviour that makes the pyramid cheap on
        # easy scenes.
        @test all(cs -> cs == 32, filter(>(0), vec(r.chip_size)))
    end
end

@testset "coarser levels recover what finer ones cannot" begin
    # The reason the pyramid exists, and the only test that exercises more than one level.
    # Coverage must rise monotonically as coarser chips are permitted.
    n = 512
    pair = banded_pair(n, (6, -4), 200:290)
    grid = gridpoints((n, n), 32; chip_size = 32, search_radius = 25)

    counts = Int[]
    for cmax in (32, 64, 128)
        r = correlate_multichip(pair, grid, params(; chip_size = 32, chip_size_max = cmax))
        push!(counts, nmeasured(r))
        used = sort(unique(Int.(filter(>(0), vec(r.chip_size)))))
        # Every permitted level up to the maximum contributes something.
        @test used ⊆ [32, 64, 128]
        @test 32 in used
        cmax >= 64 && @test 64 in used
    end
    @test counts[1] < counts[2] < counts[3]
    # And the featureless band is not fully resolvable at any scale, so some points remain
    # honestly unmeasured rather than invented.
    @test counts[3] < length(gridpoints((n, n), 32; chip_size = 32, search_radius = 25))
end

@testset "chip_size records the level that won" begin
    n = 512
    pair = banded_pair(n, (6, -4), 200:290)
    grid = gridpoints((n, n), 32; chip_size = 32, search_radius = 25)
    r = correlate_multichip(pair, grid, params(; chip_size = 32, chip_size_max = 128))

    # Measured exactly where a level claimed the point, and unmeasured exactly where none
    # did. This is the invariant that makes `chip_size` interpretable downstream.
    @test all(i -> (r.chip_size[i] == 0) == isnan(r.dx[i]), eachindex(r.dx))
    # Points in the featureless band that did resolve did so with a chip large enough to
    # reach outside it.
    @test !isempty(filter(cs -> cs > 32, vec(r.chip_size)))
end

@testset "chipsize_level in isolation" begin
    # Callable on its own, which is what lets a caller run one scale without the pyramid.
    ref, sec = shifted_pair(512, (5, -3); T = Float32)
    pair = ImagePair(ref, sec)
    grid = gridpoints((512, 512), 32; chip_size = 32, search_radius = 25)
    p = params(; chip_size = 32)

    lvl = chipsize_level(pair, grid, p, 32, trues(size(grid)))
    @test !isnothing(lvl)
    @test med(filter(!isnan, -lvl.field.dx)) ≈ 5 atol = 0.05
    # `filled` records what the hole fill recovered. Even on a fully-textured scene this is
    # not always empty: the outlier filter rejects the occasional point, and the fill then
    # recovers it from its neighbours. Every filled index must be a real point of the grid.
    @test all(i -> 1 <= i <= length(grid), lvl.filled)
    # A filled point has a displacement but no correlation of its own, since nothing was
    # measured there.
    @test all(i -> !isnan(lvl.field.dx[i]), lvl.filled)

    # `wanted` restricts which points are attempted, which is how the pyramid stops a level
    # from redoing work a finer one already did.
    w = falses(size(grid))
    w[1:3, 1:3] .= true
    lvl = chipsize_level(pair, grid, p, 32, w)
    @test !isnothing(lvl)
    @test count(!isnan, lvl.field.dx) <= 9
    # Nothing wanted at all is not an error, just no work.
    @test isnothing(chipsize_level(pair, grid, p, 32, falses(size(grid))))
end

@testset "a level that finds nothing is skipped" begin
    # Two unrelated images: the coarse pass should find no spatial coherence and the level
    # should decline rather than emit noise.
    n = 384
    pair = ImagePair(synthetic_texture(n; seed = 1), synthetic_texture(n; seed = 99))
    grid = gridpoints((n, n), 32; chip_size = 32, search_radius = 25)
    r = correlate_multichip(pair, grid, params(; chip_size = 32, chip_size_max = 64))
    # Whatever survives must be a small minority: uncorrelated images have no coherent
    # displacement to find, and the outlier filter is what enforces that.
    @test nmeasured(r) < length(r) ÷ 2
end

@testset "the coarse pass restricts the fine search" begin
    # The mechanism that makes the pyramid affordable. With the coarse gate effectively
    # disabled, more points are attempted than with it active — the result should be no
    # better, at more cost.
    n = 512
    pair = banded_pair(n, (6, -4), 200:290)
    grid = gridpoints((n, n), 32; chip_size = 32, search_radius = 25)

    strict = correlate_multichip(pair, grid,
                               params(; chip_size = 32, min_coarse_valid_fraction = 0.9))
    loose = correlate_multichip(pair, grid,
                              params(; chip_size = 32, min_coarse_valid_fraction = 0.0))
    # A demanding threshold can reject a whole level; a permissive one cannot make the
    # answer worse where both measure.
    both = findall(i -> !isnan(strict.dx[i]) && !isnan(loose.dx[i]), eachindex(strict.dx))
    @test all(i -> abs(strict.dx[i] - loose.dx[i]) < 0.5, both)
end

@testset "result shape and validation" begin
    ref, sec = shifted_pair(256, (0, 0); T = Float32)
    pair = ImagePair(ref, sec)
    grid = gridpoints((256, 256), 32; chip_size = 32, search_radius = 20)
    r = correlate_multichip(pair, grid, params(; chip_size = 32))
    @test size(r) == size(grid)
    @test size(r.chip_size) == size(grid)
    @test eltype(r.chip_size) === UInt16
    @test r.interpolated isa BitMatrix

    # Every valid parameter set selects at least one level, because `params` requires
    # `chip_size <= chip_size_min <= chip_size_max` with power-of-two ratios. Checked here so
    # the invariant the pyramid relies on is pinned at the boundary that establishes it.
    for (b, mn, mx) in ((32, 32, 32), (32, 64, 64), (16, 64, 128), (32, 32, 128))
        @test !isempty(AutoRIFT.chip_sizes(params(; chip_size = b, chip_size_min = mn,
                                                  chip_size_max = mx)))
    end
    @test_throws ArgumentError params(; chip_size = 32, chip_size_min = 96,
                                      chip_size_max = 96)
end

@testset "the coarse mask sits on the lattice the radii were reduced over" begin
    # `_cell_max_radius!` takes the maximum radius over the cell *centred* on each coarse point,
    # and `_expand_coarse_mask` inverts that assignment. The two have to agree for every fine
    # index, because the mask restricts the fine search and the radii are the evidence behind it.
    #
    # `resample(..., Nearest())` does not: it derives cell boundaries from the size ratio, which
    # for an even stride are left-aligned rather than centred. At `nr = 13, stride = 4` it puts
    # fine row 5 in coarse cell 2 while the radius for that row came from cell 1.
    for stride in (2, 3, 4, 5, 8), n in (7, 13, 20, 33, 64)
        rows = stride:stride:n
        isempty(rows) && continue
        ncoarse = length(rows)
        lo = stride ÷ 2
        hi = stride - 1 - lo

        # Which coarse cell each fine index draws its radius from, read straight off
        # `_cell_max_radius!`'s window: a one-hot radius field at coarse point `k` must reduce to
        # a nonzero maximum exactly for the fine indices that cell covers.
        radius_cell = zeros(Int, n)
        for (k, r) in enumerate(rows), i in max(r - lo, 1):min(r + hi, n)
            radius_cell[i] = k
        end

        mask_cell = zeros(Int, n)
        for k in 1:ncoarse
            onehot = falses(ncoarse, 1)
            onehot[k, 1] = true
            expanded = AutoRIFT._expand_coarse_mask(onehot, (n, 1), stride)
            for i in 1:n
                expanded[i, 1] && (mask_cell[i] = k)
            end
        end

        # Every fine index belongs to exactly one coarse cell, and it is the same one in both
        # directions wherever the radius reduction defines it.
        @test all(mask_cell[i] > 0 for i in 1:n)
        for i in 1:n
            radius_cell[i] == 0 && continue
            @test mask_cell[i] == radius_cell[i]
        end
    end

    # And the case that exposed the defect, asserted concretely rather than only as a property.
    onehot = falses(3, 1)
    onehot[1, 1] = true
    @test vec(AutoRIFT._expand_coarse_mask(onehot, (13, 1), 4)) ==
          [true, true, true, true, true, false, false, false, false, false, false, false, false]
    # `resample` disagrees here, which is why it is not used for this.
    viaresample = vec(resample(Float32.(onehot), (13, 1), Nearest())) .> 0.5f0
    @test viaresample != vec(AutoRIFT._expand_coarse_mask(onehot, (13, 1), 4))

    # A coarse point's own fine index always lands in its own cell, at any stride.
    for stride in (2, 3, 4, 7)
        n = 40
        rows = stride:stride:n
        for (k, r) in enumerate(rows)
            onehot = falses(length(rows), 1)
            onehot[k, 1] = true
            @test AutoRIFT._expand_coarse_mask(onehot, (n, 1), stride)[r, 1]
        end
    end
end
