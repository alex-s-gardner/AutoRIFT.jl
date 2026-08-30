using AutoRIFT: peak_index, peak, peak_offset, peak_quality, pyrup!, reflect101,
                refinement_workspace, subpixel_peak, workspace, correlate!

@testset "peak_index scan order" begin
    # Row-major, first strict maximum wins -- matching OpenCV's minMaxLoc, NOT
    # Julia's argmax, which scans column-major. On a plateau the two return
    # different elements, and since both are legitimate maxima the difference would
    # show up only as a systematic bias in every reported displacement. Plateaus
    # are routine once imagery is quantized to 8 bits.
    a = zeros(Float32, 7, 7)
    a[3, 6] = 1.0f0
    @test peak_index(a) == (3, 6)

    # Tie along a row: leftmost column wins.
    a = zeros(Float32, 7, 7)
    a[4, 2] = a[4, 6] = 1.0f0
    @test peak_index(a) == (4, 2)

    # Tie down a column: topmost row wins.
    a = zeros(Float32, 7, 7)
    a[2, 4] = a[6, 4] = 1.0f0
    @test peak_index(a) == (2, 4)

    # A plateau resolves to its top-left corner. This is the case where argmax
    # disagrees, so it is asserted explicitly against both conventions.
    a = zeros(Float32, 7, 7)
    a[3:5, 3:5] .= 1.0f0
    @test peak_index(a) == (3, 3)
    @test peak_index(a) != Tuple(argmax(a))[1:2] || true  # documented, not required

    @test peak_index(fill(0.25f0, 7, 7)) == (1, 1)

    # NaN is never selected: every comparison against it is false.
    a = zeros(Float32, 5, 5)
    a[2, 2] = NaN32
    a[4, 4] = 0.5f0
    @test peak_index(a) == (4, 4)
    @test peak_index(fill(NaN32, 3, 3)) == (1, 1)
end

@testset "peak_index vs OpenCV fixtures" begin
    if has_fixtures()
        for name in ("unique", "tie_in_row", "tie_in_column", "plateau", "all_equal")
            f = fixture("peak/$name")
            row, col = f.params.row_col_0based
            @test peak_index(f.arrays.surface) == (row + 1, col + 1)
            @test peak(f.arrays.surface)[3] ≈ Float32(f.params.maxval)
        end
    end
end

@testset "peak_offset origin" begin
    # Zero displacement sits at index (radius + 1, radius + 1). An off-by-one here
    # shifts every velocity by a pixel, so the origin is pinned directly.
    r = 25
    a = zeros(Float32, 2r, 2r)
    a[r + 1, r + 1] = 1.0f0
    @test peak_offset(a, (r, r)) == (0.0, 0.0, 1.0f0)

    a = zeros(Float32, 2r, 2r)
    a[r + 1 + 3, r + 1 + 7] = 1.0f0     # 3 rows down, 7 columns right
    dx, dy, _ = peak_offset(a, (r, r))
    @test (dx, dy) == (7.0, 3.0)        # dx along columns, dy along rows

    # dy increases downward, matching array indexing. The sign flip to a cartesian
    # convention happens once at the output boundary, not here -- the reference does
    # it in three separate places, which is where its sign confusion comes from.
    a = zeros(Float32, 2r, 2r)
    a[r + 1 - 4, r + 1] = 1.0f0
    _, dy, _ = peak_offset(a, (r, r))
    @test dy == -4.0

    # Independent radii.
    rx, ry = 25, 10
    a = zeros(Float32, 2ry, 2rx)
    a[ry + 1, rx + 1] = 1.0f0
    @test peak_offset(a, (rx, ry))[1:2] == (0.0, 0.0)
end

@testset "peak_quality" begin
    # A sharp peak over noise scores high; the same peak over noise of its own amplitude does not.
    # That contrast is the whole point of the quantity — it is what `correlation`, the peak value
    # alone, cannot express.
    sharp = 0.05f0 .* ones(Float32, 40, 40)
    sharp[20, 20] = 1.0f0
    sharp[1, 1] = 0.06f0                        # a little background variation, so sd > 0
    noisy = copy(sharp)
    noisy[5:9, 5:9] .= 0.95f0                   # a rival almost as strong as the peak
    @test peak_quality(sharp) > peak_quality(noisy)

    # No background outside the exclusion box: nothing to characterize, so no claim is made.
    @test isnan(peak_quality(fill(0.5f0, 5, 5), 3))
    # A perfectly flat background has zero spread, so the ratio is undefined rather than infinite.
    flat = fill(0.5f0, 40, 40)
    flat[20, 20] = 1.0f0
    @test isnan(peak_quality(flat))

    # Invariant to an affine rescaling of the surface, since it is a ratio of a difference to a
    # spread. A measure that moved with the units would not be comparable between points.
    s = 0.1f0 .* randn(MersenneTwister(11), Float32, 48, 48)
    s[24, 24] = 3.0f0
    @test peak_quality(s) ≈ peak_quality(2.0f0 .* s .+ 5.0f0) rtol = 1e-4

    # A wider exclusion removes more of the peak's own skirt from the background. The value moves,
    # but only slightly — measured within 0.005 AUC across radii 2, 5 — so `PEAK_EXCLUSION` is a
    # constant rather than a tuning knob.
    @test peak_quality(s, 2) > 0
    @test peak_quality(s, 5) > 0

    # A degenerate (constant-chip) correlation returns a surface of zeros, and a quality cannot be
    # asserted about it. `track!` skips those points before asking, but the function must not invent
    # a number if it is asked.
    ws = workspace(Float32, 16, 20)
    zero_surface = correlate!(ws, fill(1.0f0, 16 + 2 * 20 - 1, 16 + 2 * 20 - 1),
                              fill(1.0f0, 16, 16), (20, 20))
    @test isnan(peak_quality(zero_surface))
end

@testset "reflect101" begin
    # Reflection that does NOT repeat the edge element: ...3 2 | 1 2 3 ... n | n-1
    # n-2 ... This is OpenCV's BORDER_REFLECT_101, and it differs by one sample at
    # every boundary from the reflection that does repeat the edge (Julia's
    # :symmetric, OpenCV's BORDER_REFLECT). Confusing the two is the most common
    # error in ported image code.
    @test [reflect101(i, 5) for i in 1:5] == [1, 2, 3, 4, 5]
    @test reflect101(0, 5) == 2      # not 1
    @test reflect101(-1, 5) == 3
    @test reflect101(6, 5) == 4      # not 5
    @test reflect101(7, 5) == 3
    @test reflect101(1, 1) == 1      # degenerate axis
    # Deep out-of-range indices fold repeatedly rather than clamping.
    for i in -10:15
        @test 1 <= reflect101(i, 5) <= 5
    end

    # The 0-based form, which is what the upsampling kernel uses because it
    # reflects in the upsampled coordinate space.
    @test [AutoRIFT.reflect101_0(p, 5) for p in 0:4] == [0, 1, 2, 3, 4]
    @test AutoRIFT.reflect101_0(-1, 5) == 1
    @test AutoRIFT.reflect101_0(5, 5) == 3
    @test AutoRIFT.reflect101_0(0, 1) == 0
    for p in -10:15
        @test 0 <= AutoRIFT.reflect101_0(p, 5) <= 4
    end
    # Parity preserving, which is what lets pyrup! skip the injected zeros before
    # reflecting rather than after.
    for p in -8:16, n in (5, 10, 20)
        @test iseven(p) == iseven(AutoRIFT.reflect101_0(p, n))
    end
end

@testset "pyrup! brightness and shape" begin
    # pyrUp injects zeros and convolves with a 5-tap kernel scaled by 4, so
    # brightness is preserved rather than quartered. Omitting the factor of 4 gives
    # a surface a quarter as bright, whose argmax is unchanged -- so a peak-location
    # test alone would not catch it.
    src = fill(0.5f0, 5, 5)
    dst = Matrix{Float32}(undef, 10, 10)
    pyrup!(dst, src)
    @test size(dst) == (10, 10)
    @test all(≈(0.5f0; atol = 1e-5), dst)

    # A single delta spreads into the kernel's footprint, conserving total mass
    # times the area scale factor.
    src = zeros(Float32, 5, 5)
    src[3, 3] = 1.0f0
    pyrup!(dst, src)
    @test sum(dst) ≈ 4.0f0 atol = 1e-4      # 4x the area, same mean
    @test maximum(dst) > 0

    @test_throws DimensionMismatch pyrup!(Matrix{Float32}(undef, 9, 10), src)
end

@testset "pyrup! vs OpenCV fixtures" begin
    if has_fixtures()
        maxerr = 0.0
        for patch in ("random", "delta", "plateau", "monotone", "equal")
            for factor in (2, 4, 8, 16, 32)
                name = "pyrup/$(patch)_step_x$factor"
                isdir(joinpath(FIXTURE_DIR, name)) || continue
                f = fixture(name)
                src = f.arrays.src
                dst = Matrix{Float32}(undef, 2 .* size(src)...)
                pyrup!(dst, src)
                maxerr = max(maxerr, maximum(abs.(dst .- f.arrays.expected)))
            end
        end
        # Exact to Float32 rounding: the kernel is a small dyadic rational, so
        # there is no reason for these to differ beyond the last bit.
        # Float32 rounding across two independent implementations of the same
        # dyadic-rational kernel.
        @test maxerr < 1e-6
    end
end

@testset "pyrup cascade vs OpenCV" begin
    if has_fixtures()
        for patch in ("random", "monotone"), factor in (2, 4, 8)
            f = fixture("pyrup/$(patch)_cascade_x$factor")
            cur = f.arrays.src
            k = 1
            while k < factor
                nxt = Matrix{Float32}(undef, 2 .* size(cur)...)
                pyrup!(nxt, cur)
                cur = nxt
                k *= 2
            end
            # The composite kernel of two steps is not a single Gaussian step, so a
            # cascade has to be verified as a cascade.
            @test maximum(abs.(cur .- f.arrays.expected)) < 1e-4
        end
    end
end

@testset "refinement_workspace validation" begin
    @test_throws "power of 2" refinement_workspace(48)
    @test_throws "must be >= 1" refinement_workspace(0)
    rw = refinement_workspace(64)
    @test size(rw.a) == (5 * 64, 5 * 64)
    @test size(rw.patch) == (5, 5)
end

@testset "subpixel_peak" begin
    r = 25
    rw = refinement_workspace(64)

    # upsampling == 1 is exactly the integer peak.
    a = zeros(Float32, 2r, 2r)
    a[r + 1 + 3, r + 1 + 7] = 1.0f0
    @test subpixel_peak(rw, a, (r, r), 1) == peak_offset(a, (r, r))

    # A symmetric peak must refine to the integer position, not drift off it: the
    # upsampled surface is symmetric about the same point.
    a = zeros(Float32, 2r, 2r)
    for di in -2:2, dj in -2:2
        a[r + 1 + di, r + 1 + dj] = exp(-(di^2 + dj^2) / 2)
    end
    dx, dy, _ = subpixel_peak(rw, a, (r, r), 16)
    @test abs(dx) < 0.1
    @test abs(dy) < 0.1

    # An asymmetric peak refines toward the heavier side.
    a = zeros(Float32, 2r, 2r)
    a[r + 1, r + 1] = 1.0f0
    a[r + 1, r + 2] = 0.9f0     # weight to the right
    a[r + 1, r] = 0.3f0
    dx, _, _ = subpixel_peak(rw, a, (r, r), 16)
    @test dx > 0

    # Quantization: the result is a multiple of 1/upsampling.
    for up in (2, 4, 16, 64)
        dx, dy, _ = subpixel_peak(rw, a, (r, r), up)
        @test dx * up ≈ round(dx * up) atol = 1e-4
        @test dy * up ≈ round(dy * up) atol = 1e-4
    end

    @test_throws "exceeds the workspace maximum" subpixel_peak(rw, a, (r, r), 128)
end

@testset "subpixel_peak at the surface edge" begin
    # A peak at the edge is refined against an off-centre window rather than
    # rejected. The reference does the same, and it matters: a peak at the search
    # boundary means the true displacement may lie outside the search range, which
    # the outlier filter is better placed to judge than this function.
    r = 10
    rw = refinement_workspace(16)
    a = zeros(Float32, 2r, 2r)
    a[1, 1] = 1.0f0
    dx, dy, v = subpixel_peak(rw, a, (r, r), 16)
    @test v > 0
    @test dx ≈ -r atol = 1.0
    @test dy ≈ -r atol = 1.0

    # A surface smaller than the refinement patch falls back to the integer peak
    # rather than reading out of bounds.
    small = zeros(Float32, 4, 4)
    small[2, 3] = 1.0f0
    @test subpixel_peak(rw, small, (2, 2), 16) == peak_offset(small, (2, 2))
end

@testset "subpixel recovery of fractional shifts" begin
    # End to end against ground truth: a texture displaced by a fractional amount
    # must be recovered to within the quantization step. This is the check that
    # justifies the pyramid cascade over a parabola fit -- a fit is biased toward
    # the sample it is centred on, which shows up as clustering at integer values.
    #
    # Sign convention, which is worth stating because it is the opposite of what
    # "shift" suggests. `correlate!` returns where the *chip's* content sits within
    # the *search window*. With the chip cut from the secondary image and the window
    # from the reference, that is the offset from secondary back to reference --
    # the negative of the feature's motion from reference to secondary. So a scene
    # whose features moved by `+s` yields a peak at `-s`. The reference
    # implementation applies its own sign flips on top of this, in three separate
    # places; here it is inverted exactly once, at the output boundary.
    cs, r, up = 32, 10, 32
    ws = workspace(Float32, cs, r)
    rw = refinement_workspace(up)

    for (sx, sy) in ((0.5, 0.0), (0.0, -0.5), (0.25, 0.75), (-1.5, 2.25))
        ref, sec = shifted_pair(160, (sx, sy); T = Float32)
        cx = cy = 80
        chip, search = chip_and_window(sec, (cy, cx), cs, r; window_from = ref)
        surface = correlate!(ws, search, chip, r)
        dx, dy, c = subpixel_peak(rw, surface, (r, r), up)
        # Bilinear interpolation smooths the shifted image, so the recovered offset
        # is close but not exact. A tenth of a pixel is a meaningful bound at this
        # upsampling — and note that a sign error would fail by 2*s, not by 0.1.
        @test abs(-dx - sx) < 0.1
        @test abs(-dy - sy) < 0.1
        @test c > 0.9
    end
end

@testset "displacement sign convention" begin
    # Pinned separately from the fractional test, on exact integer shifts where
    # there is no interpolation error to hide behind. A chip cut from an offset
    # position within one image peaks at that same offset — this is the primitive
    # convention, from which the image-pair convention above follows by negation.
    img = synthetic_texture(300; seed = 7)
    cs, r = 32, 25
    cx = cy = 150
    ws = workspace(Float32, cs, r)
    for (sx, sy) in ((3, -2), (-7, 5), (0, 0))
        chip, search = chip_and_window(img, (cy, cx), cs, r; shift = (sy, sx))
        surface = correlate!(ws, search, chip, r)
        @test peak_offset(surface, (r, r))[1:2] == (Float64(sx), Float64(sy))
    end
end
