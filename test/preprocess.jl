using AutoRIFT: ImagePair, preprocess, quantize, highpass, wallis, valid,
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

@testset "quantize to UInt8" begin
    n = 40
    tex = synthetic_texture(n; seed = 7)
    p = ImagePair(tex, tex)
    q = quantize(preprocess(p, :highpass), :uint8)
    @test eltype(q) === UInt8
    # The ±3σ window should use most of the range without saturating everything.
    @test maximum(q.reference) > 200
    @test minimum(q.reference) < 60
    @test count(==(0x00), q.reference) < length(q.reference) ÷ 10
    @test count(==(0xff), q.reference) < length(q.reference) ÷ 10
end

@testset "quantize excludes invalid pixels from its statistics" begin
    # A large fill-valued region would otherwise drag the mean and inflate the spread,
    # compressing the real data into a fraction of the available range.
    n = 40
    tex = synthetic_texture(n; seed = 7)
    img = copy(tex)
    mask = trues(n, n)
    img[1:20, :] .= -999.0f0      # a fill value far outside the data range
    mask[1:20, :] .= false

    p = ImagePair(img, img; reference_valid = mask, secondary_valid = mask)
    q = quantize(p, :uint8)
    # The valid half still spans the range, which it would not if -999 were included.
    live = q.reference[21:n, :]
    @test maximum(live) > 200
    @test minimum(live) < 60
    # Invalid pixels are zeroed rather than mapped somewhere meaningful.
    @test all(==(0x00), q.reference[1:20, :])
end

@testset "quantize handles a uniform image" begin
    # No texture to preserve, so mid-grey is the neutral answer; pinning it to an
    # endpoint would be arbitrary.
    p = ImagePair(fill(5.0f0, 10, 10), fill(5.0f0, 10, 10))
    q = quantize(p, :uint8)
    @test all(==(0x80), q.reference)

    # Nothing valid at all: all zero rather than an error, since the grid will have no
    # searchable points anyway.
    p = ImagePair(fill(5.0f0, 10, 10), fill(5.0f0, 10, 10);
                  reference_valid = falses(10, 10), secondary_valid = falses(10, 10))
    @test all(==(0x00), quantize(p, :uint8).reference)
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
    h = cs ÷ 2
    cx = cy = 100
    ws = workspace(Float32, cs, r)

    function corr_at(ref, sec)
        chip = sec[(cy - h):(cy + h - 1), (cx - h):(cx + h - 1)]
        search = ref[(cy - h - r):(cy + h + r - 2), (cx - h - r):(cx + h + r - 2)]
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
