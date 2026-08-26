using AutoRIFT: ImagePair, preprocess, replace_nonfinite, highpass, wallis, valid,
                workspace, correlate!, peak_offset

@testset "ImagePair construction" begin
    a = rand(Float32, 8, 8)
    b = rand(Float32, 8, 8)
    p = ImagePair(a, b)
    @test size(p) == (8, 8)
    @test eltype(p) === Float32
    @test all(p.reference_valid)

    # Non-finite input is invalid by default...
    a2 = copy(a)
    a2[3, 3] = NaN32
    p = ImagePair(a2, b)
    @test !p.reference_valid[3, 3]
    @test p.secondary_valid[3, 3]
    @test !valid(p)[3, 3]          # the intersection

    # ...but zero is NOT, since it is a legitimate radiance value. The reference
    # conflates zero with no-data, which misclassifies genuinely dark pixels.
    a3 = copy(a)
    a3[4, 4] = 0.0f0
    @test ImagePair(a3, b).reference_valid[4, 4]

    # An explicit mask is honoured, for sensors that do use a fill value.
    m = trues(8, 8)
    m[5, 5] = false
    @test !ImagePair(a, b; reference_valid = m).reference_valid[5, 5]

    @test_throws DimensionMismatch ImagePair(rand(Float32, 8, 8), rand(Float32, 8, 9))
    @test_throws DimensionMismatch ImagePair(a, b; reference_valid = trues(4, 4))
end

@testset "highpass removes a gradient" begin
    # What the filter is for: an illumination ramp is exactly the low-frequency content
    # that differs between two acquisitions and must not drive the correlation.
    n = 40
    ramp = Float32[0.01f0 * (i + 2j) for i in 1:n, j in 1:n]
    out = highpass(ramp, trues(n, n), 5)
    # In the interior the ramp is removed almost entirely; a linear function has no
    # high-frequency content, so the residual is rounding.
    interior = out[6:(n - 5), 6:(n - 5)]
    @test maximum(abs, interior) < 1e-4

    # Texture survives.
    tex = synthetic_texture(n; seed = 3)
    out = highpass(tex, trues(n, n), 5)
    @test Statistics.std(out[6:(n - 5), 6:(n - 5)]) > 0.01
    # And the local mean is what was removed, so the result is centred near zero.
    @test abs(Statistics.mean(out[6:(n - 5), 6:(n - 5)])) < 0.02
end

@testset "invalid pixels do not bias the local mean" begin
    # The difference from the reference, which zero-pads its convolution and so lets a
    # no-data border pull the local mean of every pixel near it toward zero.
    n = 20
    img = fill(10.0f0, n, n)
    mask = trues(n, n)
    mask[1:5, :] .= false          # a no-data band along the top
    img[1:5, :] .= 0.0f0           # fill values, as a sensor would write them

    out = highpass(img, mask, 5)
    # Just below the border the image is uniform, so a mask-aware high-pass returns
    # zero there. Including the fill values would return a large positive residual.
    @test abs(out[8, 10]) < 1e-4
    @test abs(out[7, 10]) < 1e-4
    # Inside the invalid band there is no measurement at all.
    @test isnan(out[3, 10])
end

@testset "wallis equalises contrast" begin
    # Two halves of a scene with the same texture but different contrast and brightness,
    # as a shadowed and a sunlit slope would be. Wallis should make them comparable.
    n = 60
    tex = synthetic_texture(n; seed = 5)
    img = copy(tex)
    img[:, 1:(n ÷ 2)] .= 0.2f0 .+ 0.1f0 .* tex[:, 1:(n ÷ 2)]        # dim, low contrast
    img[:, (n ÷ 2 + 1):end] .= 0.7f0 .+ 0.5f0 .* tex[:, (n ÷ 2 + 1):end]  # bright, high

    out = wallis(img, trues(n, n), 5)
    lo = out[10:(n - 10), 8:(n ÷ 2 - 8)]
    hi = out[10:(n - 10), (n ÷ 2 + 8):(n - 8)]
    # After normalisation the two halves have comparable spread, where before they
    # differed fivefold.
    @test 0.5 < Statistics.std(lo) / Statistics.std(hi) < 2.0
    @test Statistics.std(img[:, 1:(n ÷ 2)]) / Statistics.std(img[:, (n ÷ 2 + 1):end]) < 0.3
end

@testset "wallis handles a zero-variance window" begin
    # A uniform region has no texture to normalise, so the result is undefined rather
    # than infinite. In the reference this path produced NaN from a negative variance --
    # a precision artefact rather than a real absence of data.
    n = 20
    img = fill(5.0f0, n, n)
    out = wallis(img, trues(n, n), 5)
    @test all(isnan, out)
    @test !any(isinf, out)

    # min_std floors the divisor, which limits noise amplification in near-textureless
    # areas rather than rejecting them.
    img = fill(5.0f0, n, n)
    img[10, 10] = 5.01f0
    out = wallis(img, trues(n, n), 5, 0.1)
    @test isfinite(out[10, 10])
    @test abs(out[10, 10]) < 1.0        # bounded by the floor, not amplified
end

@testset "preprocess shrinks the validity mask" begin
    # A filter with a w-wide window spreads each invalid pixel over a w-wide
    # neighbourhood: an output computed partly from fill values is not a measurement.
    n = 30
    a = synthetic_texture(n; seed = 1)
    b = synthetic_texture(n; seed = 2)
    ma = trues(n, n)
    ma[15, 15] = false
    p = ImagePair(a, b; reference_valid = ma)

    q = preprocess(p, Highpass(; width = 5))
    # The single invalid pixel invalidates its whole 5x5 neighbourhood.
    @test !q.reference_valid[15, 15]
    @test !q.reference_valid[13, 13]
    @test !q.reference_valid[17, 17]
    @test q.reference_valid[12, 12]      # just outside the footprint
    # The other image is untouched.
    @test all(q.secondary_valid)
    # Output is finite everywhere, with the mask carrying the information about what is
    # real -- so downstream arithmetic never has to guard against NaN.
    @test all(isfinite, q.reference)
    @test count(q.reference_valid) < count(ma)
end

@testset "preprocess accepts Symbols and methods" begin
    n = 20
    p = ImagePair(synthetic_texture(n; seed = 1), synthetic_texture(n; seed = 2))
    @test preprocess(p, :highpass).reference isa Matrix{Float32}
    @test preprocess(p, :none).reference == Float32.(p.reference)
    @test preprocess(p, Highpass(; width = 7)) isa ImagePair
    @test preprocess(p, Wallis(; width = 5)) isa ImagePair
    @test_throws "not recognised" preprocess(p, :sharpen)
end

@testset "non-finite pixels become zero, and the type is preserved" begin
    # The images reach the correlator in the caller's own type: nothing here converts, so an
    # integer image stays 2 bytes per pixel rather than becoming 4.
    for T in (Float32, Float64, Int16, UInt8)
        img = synthetic_texture(20; seed = 7, T)
        out, _ = replace_nonfinite(img, trues(20, 20))
        @test eltype(out) === T
        @test out == img
        # A copy, not an alias: the pipeline must not write through to the caller's array.
        @test out !== img
    end

    # NaN and Inf become zero so downstream arithmetic stays finite. The mask is what records
    # that those pixels carry no information, which is why the value itself can be anything.
    img = Float32[1 NaN32; Inf32 -Inf32]
    out, mask = replace_nonfinite(img, trues(2, 2))
    @test out == Float32[1 0; 0 0]
    @test all(mask)

    # Complex is non-finite if either component is, and the same replacement applies.
    z = ComplexF32[1+2im NaN32+0im; 0+NaN32*im 3-1im]
    zout, _ = replace_nonfinite(z, trues(2, 2))
    @test zout == ComplexF32[1+2im 0; 0 3-1im]

    # An integer image cannot hold a non-finite value, so nothing is replaced.
    iout, _ = replace_nonfinite(Int16[1 -2; 3 -4], trues(2, 2))
    @test iout == Int16[1 -2; 3 -4]
end

@testset "filtering improves correlation across a brightness change" begin
    # The end-to-end justification for this whole file. Two views of the same scene, one
    # with a strong multiplicative and additive brightness change, correlated at a known
    # zero shift. Unfiltered ZNCC already handles a global affine change by construction,
    # so the test uses a *spatially varying* change, which it does not.
    n = 200
    tex = synthetic_texture(n; seed = 11)
    ramp = Float32[0.4f0 + 0.006f0 * j for _ in 1:n, j in 1:n]
    a = tex
    b = tex .* ramp .+ (ramp .- 0.4f0)          # gain and offset both vary across x

    cs, r = 32, 10
    cx = cy = 100
    ws = workspace(Float32, cs, r)

    function corr_at(ref, sec)
        chip, search = chip_and_window(sec, (cy, cx), cs, r; window_from = ref)
        surface = correlate!(ws, search, chip, r)
        return peak_offset(surface, (r, r))
    end

    dx_raw, dy_raw, c_raw = corr_at(a, b)
    fp = preprocess(ImagePair(a, b), Highpass(; width = 5))
    dx_f, dy_f, c_f = corr_at(fp.reference, fp.secondary)

    # Both should find zero displacement -- the scene did not move.
    @test (dx_f, dy_f) == (0.0, 0.0)
    # Filtering raises the peak correlation, which is the quality measure the outlier
    # filter and any downstream weighting depend on.
    @test c_f > c_raw
    @test c_f > 0.95
end

# ---------------------------------------------------------------------------
# The decibel filters and the derivative kernels built on them
# ---------------------------------------------------------------------------

@testset "derivative kernels match OpenCV's getDerivKernels" begin
    # Pinned against `cv2.getDerivKernels(order, 0, w, normalize=false)`, which is what the
    # reference builds `Sobel` and `Laplacian` from. Deriving these independently is a second
    # chance to get the scale wrong -- for widths above 3 they are binomial-difference kernels
    # whose magnitude grows with width, and Julia's `Kernel.sobel` uses a different, normalized,
    # 3x3-only convention.
    expected = Dict(
        (3, 1) => ([-1, 0, 1], [1, 2, 1]),
        (3, 2) => ([1, -2, 1], [1, 2, 1]),
        (5, 1) => ([-1, -2, 0, 2, 1], [1, 4, 6, 4, 1]),
        (5, 2) => ([1, 0, -2, 0, 1], [1, 4, 6, 4, 1]),
        (7, 1) => ([-1, -4, -5, 0, 5, 4, 1], [1, 6, 15, 20, 15, 6, 1]),
        (7, 2) => ([1, 2, -1, -4, -1, 2, 1], [1, 6, 15, 20, 15, 6, 1]),
    )
    for ((w, order), (deriv, smooth)) in expected
        d, s = AutoRIFT._deriv_kernels(w, order)
        @test Int.(d) == deriv
        @test Int.(s) == smooth
    end

    # A derivative kernel sums to zero, which is what makes it blind to a constant offset. This
    # is the property `Laplacian` relies on to remove multiplicative brightness in decibels.
    for w in (3, 5, 7), order in (1, 2)
        d, s = AutoRIFT._deriv_kernels(w, order)
        @test sum(d) == 0
        @test sum(s) > 0
    end

    @test_throws "must be odd and at least 3" AutoRIFT._deriv_kernels(4, 1)
    @test_throws "order must be 1 or 2" AutoRIFT._deriv_kernels(5, 3)
end

@testset "decibel converts amplitude and drops what has no logarithm" begin
    img = Float32[1 10 100; 1000 0 -5; 0.1 0.01 2]
    mask = trues(3, 3)
    out, v = AutoRIFT.decibel(img, mask)

    @test out[1, 1] ≈ 0.0f0                      # 20*log10(1) == 0
    @test out[1, 2] ≈ 20.0f0                     # 20*log10(10)
    @test out[1, 3] ≈ 40.0f0
    @test out[2, 1] ≈ 60.0f0
    # `20 log10` and not `10 log10`: the input is amplitude, so a factor of ten is 20 dB.
    @test out[1, 2] - out[1, 1] ≈ 20.0f0

    # Zero and negative amplitudes have no decibel value. They leave the mask rather than
    # becoming -Inf or NaN, which is where this departs from the reference.
    @test !v[2, 2]
    @test !v[2, 3]
    @test isfinite(out[2, 2])
    @test all(isfinite, out)
    @test count(v) == 7

    # An already-invalid pixel stays invalid whatever its value.
    m2 = trues(3, 3)
    m2[1, 1] = false
    _, v2 = AutoRIFT.decibel(img, m2)
    @test !v2[1, 1]
end

@testset "sobel is the summed kernel, not a gradient magnitude" begin
    # The reference adds the x and y derivative kernels and convolves once, so the filter is
    # directional: `Gx + Gy`. A gradient magnitude would be non-negative everywhere and
    # symmetric under transposition; this is neither, and a future edit that "fixes" it into a
    # magnitude would change every result.
    img = synthetic_texture(48; seed = 4) .+ 1.0f0
    mask = trues(48, 48)
    out = AutoRIFT.sobel(img, mask, 5)

    @test any(<(0), out)                          # signed, so not a magnitude
    inner = 5:44
    @test !all(isapprox.(out[inner, inner], permutedims(AutoRIFT.sobel(permutedims(img),
                                                                      mask, 5))[inner, inner]))

    # Blind to a constant offset, because the kernel sums to zero.
    shifted = AutoRIFT.sobel(img .+ 100.0f0, mask, 5)
    @test out[inner, inner] ≈ shifted[inner, inner] rtol=1e-4
end

@testset "laplacian works in decibels, so it is blind to a brightness scale" begin
    # The log comes first, and that ordering is the whole point: radar brightness varies
    # multiplicatively, so scaling the scene is an additive offset in decibels, which a
    # zero-sum second-derivative kernel removes. On raw amplitude the same scaling would
    # multiply the output.
    img = synthetic_texture(48; seed = 6) .+ 1.0f0
    mask = trues(48, 48)
    inner = 5:44

    plain = AutoRIFT.laplacian(img, mask, 5)
    scaled = AutoRIFT.laplacian(img .* 8.0f0, mask, 5)
    @test plain[inner, inner] ≈ scaled[inner, inner] rtol=1e-3

    # Not the same as a Laplacian of the raw amplitude, which is what makes the ordering
    # observable rather than a stylistic note.
    raw = AutoRIFT._convolve_masked(img, mask, AutoRIFT._summed_deriv_kernel(5, 2))
    @test !isapprox(plain[inner, inner], raw[inner, inner]; rtol = 1e-2)
end

@testset "wallis_gapfill fills interior gaps and excludes the border" begin
    n = 120
    # Landsat-like digital numbers: `std_cutoff` is an absolute threshold on the local standard
    # deviation, so it only means "low contrast" against imagery of a realistic scale.
    img = synthetic_texture(n; seed = 8) .* 2000.0f0 .+ 500.0f0
    mask = trues(n, n)
    # An interior gap, as a post-2003 Landsat-7 scan-line gap would appear.
    mask[40:60, 30:36] .= false
    # And an outer no-data margin deep enough that its far side is beyond `GAPFILL_REACH` from
    # any real pixel. Depth is what distinguishes the two cases: a thin margin is within reach of
    # data and so is filled like any interior gap, which is correct -- only a pixel with no nearby
    # data to be statistically consistent with is excluded.
    margin = 2 * ceil(Int, AutoRIFT.GAPFILL_REACH)
    mask[1:margin, :] .= false

    out, v = AutoRIFT.wallis_gapfill(img, mask, 5, 0.25; rng = Random.Xoshiro(1))

    @test size(out) == (n, n)
    @test all(isfinite, out)
    # The interior gap becomes valid -- that is what stops it masking out its neighbourhood.
    @test all(v[45:55, 31:35])
    # The top of a deep margin is beyond reach of any data, so it stays invalid.
    @test !v[1, 60]
    # Its inner edge is within reach, so it is filled rather than excluded.
    @test v[margin, 60]
    # Real data away from either stays measured, and normalized rather than replaced.
    @test v[80, 80]

    # Reproducible from the seed, which the reference is not: it draws from NumPy's unseeded
    # global generator, so two runs of the same scene disagree.
    again, _ = AutoRIFT.wallis_gapfill(img, mask, 5, 0.25; rng = Random.Xoshiro(1))
    @test isequal(out, again)
    other, _ = AutoRIFT.wallis_gapfill(img, mask, 5, 0.25; rng = Random.Xoshiro(2))
    @test !isequal(out, other)
end

@testset "the local standard deviation carries the Bessel correction" begin
    # The reference scales by `sqrt(n / (n - 1))` with `n` the kernel *area*, not the per-window
    # valid count -- which is what makes it a single multiply. Matching it exactly is what keeps
    # `WallisGapfill`'s `std_cutoff`, an absolute threshold on this quantity, mean the same thing
    # here as there.
    for w in (3, 5, 21)
        n = w * w
        @test AutoRIFT._bessel_factor(w) ≈ Float32(sqrt(n / (n - 1)))
    end
    # A single-pixel window has no spread to correct and must not divide by zero.
    @test AutoRIFT._bessel_factor(1) == 1.0f0

    # The correction is a constant factor on the whole field, so `wallis` scales by exactly its
    # reciprocal. Asserted against a hand-computed window rather than against itself.
    img = Float32[0 0 0 0 0; 0 1 2 3 0; 0 4 5 6 0; 0 7 8 9 0; 0 0 0 0 0]
    mask = trues(5, 5)
    out = AutoRIFT.wallis(img, mask, 3, 0.0)
    centre = img[2:4, 2:4]
    mu = sum(centre) / 9
    sigma = sqrt(sum((centre .- mu) .^ 2) / 9) * sqrt(9 / 8)
    @test out[3, 3] ≈ (img[3, 3] - mu) / sigma rtol=1e-5
end

@testset "every PreprocessMethod runs on the input it documents" begin
    # The gap this closes: `Sobel`, `Laplacian`, `Decibel` and `WallisGapfill` were exported,
    # documented, resolvable from a keyword symbol, and had `filter_width`/`filter_reach`
    # methods -- but no `preprocess` method, so each threw a `MethodError` from inside the run.
    img = synthetic_texture(40; seed = 9) .* 1000.0f0 .+ 500.0f0
    mask = trues(40, 40)
    for m in (NoPreprocess(), Highpass(; width = 5), Wallis(), WallisGapfill(),
              Sobel(), Laplacian(), Decibel())
        out, v = preprocess(img, mask, m)
        @test size(out) == size(img)
        @test size(v) == size(mask)
        @test all(isfinite, out)
        @test any(v)
    end

    # Complex input is still refused for every amplitude filter, naming the two things a caller
    # probably wants instead.
    cimg = rand(ComplexF32, 16, 16)
    cmask = trues(16, 16)
    for m in (Highpass(), Wallis(), WallisGapfill(), Sobel(), Laplacian(), Decibel())
        @test_throws "amplitude filter and the image is complex" preprocess(cimg, cmask, m)
    end

    # And the seed-carrying form forwards to the three-argument one for every filter that
    # ignores it, so `_prepare` need not know which filter it holds.
    for m in (NoPreprocess(), Highpass(; width = 5), Wallis(), Sobel(), Laplacian(), Decibel())
        a, _ = preprocess(img, mask, m)
        b, _ = preprocess(img, mask, m, UInt64(12345))
        @test isequal(a, b)
    end
end
