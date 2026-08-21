# Complex (SLC) correlation: coherence, deramping, and the measure hierarchy.
#
# There is no reference implementation to compare against, and that shapes this whole file.
# autoRIFT v2.1.2 has no complex path (its core exports only real and UInt8 entry points), and
# ISCE2's cuAmpcor takes `abs` before correlating — so neither computes what `Coherence` computes.
# The OpenCV fixture corpus is no help either: `matchTemplate` does not accept complex input.
#
# So every assertion here is against something *derivable* rather than against another program's
# output:
#
#   γ(T, T) = 1                     exactly, by construction of the estimator
#   a known shift                   must peak at that shift
#   a known phase ramp              must be recovered by `ramp_phase` to rounding
#   deramping                       must be idempotent, and must restore a ramped chip's coherence
#   uncorrelated noise              γ ~ 1/sqrt(N), a statistical bound
#
# The last is asserted loosely on purpose: it is a distributional claim, and a tight bound on one
# realisation would be a flaky test rather than a stronger one.

using AutoRIFT: workspace, correlate!, prepare_chip!, peak_index, deramp, ramp_phase,
                Coherence, ZNCC, Deramp, chip_measures, params,
                _cnumerators_direct!, _cnumerators_fft!

# A speckle-like complex field: circular Gaussian, which is what fully-developed speckle is.
# Smoothed so neighbouring samples correlate, otherwise a shifted copy has no structure to find.
function speckle(n::Int; seed = 0xC0FFEE)
    rng = Random.MersenneTwister(seed)
    z = complex.(randn(rng, Float32, n, n), randn(rng, Float32, n, n))
    for _ in 1:2
        z = (z .+ circshift(z, (1, 0)) .+ circshift(z, (0, 1)) .+ circshift(z, (1, 1))) ./ 4
    end
    return z
end

@testset "coherence of a chip with itself is 1" begin
    # The defining property: γ is a normalised inner product, so a chip against itself must give
    # exactly 1 at zero shift. Any normalisation error shows up here first.
    r = 6
    for cs in (16, 32)
        z = speckle(cs + 2r - 1)
        # Cut at (r+1, r+1): the surface index a chip cut at offset `o` peaks at is `o` itself,
        # confirmed against ZNCC on the same data, so this is the zero-displacement centre.
        chip = z[(r + 1):(r + cs), (r + 1):(r + cs)]
        ws = workspace(ComplexF32, cs, r)
        s = correlate!(ws, z, chip, r; measure = Coherence())
        @test peak_index(s) == (r + 1, r + 1)
        @test maximum(s) ≈ 1.0f0 atol = 1e-5
        # And γ is a magnitude ratio, so it is bounded by 1 everywhere — up to Float32 rounding of
        # a Float64 ratio, the same allowance the real measures get.
        @test all(-1.0f0 - 2048eps(1.0f0) .<= s .<= 1.0f0 + 2048eps(1.0f0))
    end
end

@testset "coherence finds a known shift" begin
    # The measurement that matters. A shifted complex chip must peak at the shift, which is what
    # makes this a tracker rather than a similarity function.
    cs, r = 32, 10
    for (sx, sy) in ((0, 0), (3, 0), (0, -4), (5, -3))
        z = speckle(cs + 2r - 1 + 16)
        win = @view z[1:(cs + 2r - 1), 1:(cs + 2r - 1)]
        # Cut the chip offset by (sy, sx) from the centre, so the peak moves by that much.
        chip = z[(r + 1 + sy):(r + sy + cs), (r + 1 + sx):(r + sx + cs)]
        ws = workspace(ComplexF32, cs, r)
        s = correlate!(ws, win, chip, r; measure = Coherence())
        @test peak_index(s) == (r + 1 + sy, r + 1 + sx)
        @test maximum(s) ≈ 1.0f0 atol = 1e-5
    end
end

@testset "coherence is invariant to a global phase offset" begin
    # An SLC pair has an arbitrary absolute phase difference — different orbit, different
    # atmosphere — so a measure sensitive to it would report nothing useful. Invariance comes from
    # removing the *complex* mean in `prepare_chip!`; this is what asserts that was done.
    cs, r = 32, 8
    z = speckle(cs + 2r - 1)
    chip = z[(r + 1):(r + cs), (r + 1):(r + cs)]
    ws = workspace(ComplexF32, cs, r)
    base = copy(correlate!(ws, z, chip, r; measure = Coherence()))
    for θ in (0.5, 2.0, -3.0)
        rotated = correlate!(ws, z, chip .* cis(Float32(θ)), r; measure = Coherence())
        @test all(isapprox.(base, rotated; atol = 1e-5))
    end
end

@testset "the ramp estimator recovers a known ramp" begin
    # `ramp_phase` is the testable half of deramping: build an image whose phase is a known linear
    # function of position and check the gradient comes back. Exact to rounding, since the
    # estimator is an argument of a sum of conjugate products and the products are all equal here.
    n = 64
    mask = trues(n, n)
    for (fx, fy) in ((0.0, 0.0), (0.3, 0.0), (0.0, -0.7), (0.21, 0.13))
        # A constant-amplitude phase ramp: the estimate should be exactly (fx, fy).
        z = [ComplexF32(cis((i - 1) * fy + (j - 1) * fx)) for i in 1:n, j in 1:n]
        px, py = ramp_phase(z, mask)
        @test px ≈ fx atol = 1e-5
        @test py ≈ fy atol = 1e-5
    end
end

@testset "deramping removes the ramp and is idempotent" begin
    n = 48
    mask = trues(n, n)
    z = speckle(n) .* [ComplexF32(cis((i - 1) * 0.4 - (j - 1) * 0.25)) for i in 1:n, j in 1:n]

    d = deramp(z, mask)
    # After deramping there is no linear phase gradient left to find.
    px, py = ramp_phase(d, mask)
    @test abs(px) < 1e-4
    @test abs(py) < 1e-4
    # So a second pass changes nothing: the operation is idempotent, which it must be for
    # `reinit!`'s reuse of a prepared image to be sound.
    @test all(isapprox.(deramp(d, mask), d; atol = 1e-5))
    # Amplitude is untouched — deramping rotates phasors, it does not rescale them.
    @test all(isapprox.(abs.(d), abs.(z); rtol = 1e-5))

    # Axis selection: deramping only x must leave the y gradient intact.
    #
    # The tolerance is 0.01 rather than 1e-3 because the estimate on *speckle* carries sampling
    # error the constant-amplitude case above does not — measured 0.4078 against an imposed 0.4.
    # That is the estimator being honest about a finite noisy sample, not a bias: the assertion
    # that matters is that deramping x left this axis alone, and it did, to ten digits.
    dx_only = deramp(z, mask, :x)
    px2, py2 = ramp_phase(dx_only, mask)
    @test abs(px2) < 1e-4
    @test py2 ≈ 0.4 atol = 0.01
    # Untouched by the x pass, which is the real claim. Not bit-exact: deramping x rotates every
    # sample, so the y estimate is recomputed from different (equally valid) floating-point values.
    # Agreement to ~1e-8 is what that rounding permits.
    @test py2 ≈ last(ramp_phase(z, mask)) atol = 1e-7
end

@testset "deramping restores coherence lost to a phase ramp" begin
    # The reason `Deramp` exists, stated as a measurement. A phase ramp *across the chip* is
    # exactly the failure mode Joughin (2002) describes — it suppresses the correlation peak — and
    # removing it must bring the peak back.
    cs, r = 32, 8
    mask = trues(cs + 2r - 1, cs + 2r - 1)
    z = speckle(cs + 2r - 1)
    # A differential ramp: present in the window, absent from the chip, so the two disagree in
    # phase in a way that grows across the patch.
    ramped = z .* [ComplexF32(cis((i - 1) * 0.6 + (j - 1) * 0.45))
                   for i in axes(z, 1), j in axes(z, 2)]
    chip = z[(r + 1):(r + cs), (r + 1):(r + cs)]

    ws = workspace(ComplexF32, cs, r)
    spoiled = maximum(correlate!(ws, ramped, chip, r; measure = Coherence()))
    # Deramp both sides and correlate again.
    dz = deramp(ramped, mask)
    dchip = deramp(chip, trues(cs, cs))
    restored = maximum(correlate!(ws, dz, dchip, r; measure = Coherence()))

    # Measured: 0.13 spoiled, 0.85 restored — a 6.3x recovery. The ramp really does destroy the
    # peak, and deramping really does bring it back. It does not reach 1.0, and should not: the
    # window and chip ramps are estimated over different extents (81x81 against 32x32), so the two
    # corrections differ slightly and a residual differential ramp remains. That residual is
    # exactly the higher-order phase variation `Deramp`'s docstring says it cannot remove.
    @test spoiled < 0.3
    @test restored > 0.8
    @test restored > 4 * spoiled
end

@testset "the direct and FFT complex numerators agree" begin
    # The complex numerator has two implementations for the same reason the real one does: the FFT
    # is what makes it affordable (113x at chip 128), and the direct loop is what the FFT is checked
    # against. They must agree to Float32 rounding.
    #
    # This also guards the trap in `_cnumerators_fft!`: correlation needs the *conjugate* of the
    # *reversed* chip, and dropping either operation yields a plausible surface with a wrong peak —
    # reversal alone gives convolution, conjugation alone mirrors the displacement.
    worst = 0.0
    for (cs, r) in ((16, 6), (24, 8), (32, 10), (32, 25))
        z = speckle(cs + 2r - 1)
        chip = z[(r + 1):(r + cs), (r + 1):(r + cs)]
        ws = workspace(ComplexF32, cs, r)
        prepare_chip!(ws, chip)
        cd = @view ws.cchip[1:cs, 1:cs]
        nr, nc = 2r, 2r
        d = Matrix{ComplexF32}(undef, nr, nc)
        f = Matrix{ComplexF32}(undef, nr, nc)
        _cnumerators_direct!(d, z, cd, nr, nc, cs, cs)
        _cnumerators_fft!(f, ws, z, cd, nr, nc, cs, cs)
        # Relative to the scale of the numerator, which is what a tolerance on a correlation sum
        # has to be — the absolute magnitude grows with the chip area.
        rel = maximum(abs.(d .- f)) / maximum(abs, d)
        worst = max(worst, rel)
        @test rel < 1e-6
    end
    @info "worst direct-vs-FFT complex numerator deviation" worst
end

@testset "uncorrelated speckle gives low coherence" begin
    # The null case: two independent fields have no shift to find, and γ should sit near the
    # 1/sqrt(N) noise floor rather than near 1. Loose bounds — this is a distributional claim.
    cs, r = 32, 8
    a = speckle(cs + 2r - 1; seed = 0x1111)
    b = speckle(cs; seed = 0x2222)
    ws = workspace(ComplexF32, cs, r)
    s = correlate!(ws, a, b, r; measure = Coherence())
    floor = 1 / sqrt(cs * cs)
    @test maximum(s) < 20 * floor      # well below a real match
    @test all(isfinite, s)
end

@testset "degenerate complex chips are reported, not correlated" begin
    # A constant complex chip has zero variance about its mean, so γ is undefined at every shift —
    # the same condition `prepare_chip!` guards for real input, and the same answer: no
    # measurement, rather than a peak pinned to the surface corner.
    cs, r = 16, 6
    ws = workspace(ComplexF32, cs, r)
    z = speckle(cs + 2r - 1)
    flat = fill(ComplexF32(2 + 3im), cs, cs)
    s = correlate!(ws, z, flat, r; measure = Coherence())
    @test AutoRIFT.degenerate(ws)
    @test all(iszero, s)
end

@testset "Coherence on real input is an error, not a silent fallback" begin
    # A real image has no phase, so there is nothing for coherence to measure. Reporting that is
    # better than degrading to something that looks like NCC and is not documented as such.
    cs, r = 16, 6
    ws = workspace(Float32, cs, r)
    a = rand(Float32, cs + 2r - 1, cs + 2r - 1)
    b = rand(Float32, cs, cs)
    @test_throws ArgumentError correlate!(ws, a, b, r; measure = Coherence())
end

@testset "amplitude filters and uint8 quantization reject complex input" begin
    z = speckle(32)
    m = trues(32, 32)
    # Each of these would produce something plausible-looking and meaningless.
    for filt in (Highpass(), Wallis(), Sobel(), Laplacian(), Decibel())
        @test_throws ArgumentError AutoRIFT.preprocess(z, m, filt)
    end
    @test_throws ArgumentError AutoRIFT.quantize(z, m, QuantizeUInt8())
    # And deramping a real image is the mirror-image error.
    @test_throws ArgumentError AutoRIFT.preprocess(rand(Float32, 32, 32), m, Deramp())
end

@testset "the measure tuple maps onto chip-size levels" begin
    # `chip_measures` is the padding rule, and the rule is what makes escalation expressible
    # without a second way to specify levels.
    p = params(; chip_size = 32, chip_size_max = 128)          # levels 32, 64, 128
    @test length(AutoRIFT.chip_sizes(p)) == 3
    # A scalar applies everywhere.
    @test chip_measures(p) === (ZNCC(), ZNCC(), ZNCC())
    # A tuple assigns in order, last entry repeating — coherence fine, amplitude coarse.
    p2 = params(; chip_size = 32, chip_size_max = 128, similarity = (:coherence, :zncc))
    @test chip_measures(p2) === (Coherence(), ZNCC(), ZNCC())
    # Naming a measure per level is exact rather than padded.
    p3 = params(; chip_size = 32, chip_size_max = 128,
                similarity = (:coherence, :ncc, :zncc))
    @test chip_measures(p3) === (Coherence(), NCC(), ZNCC())
    # More measures than levels is an error: the extras would silently do nothing.
    p4 = params(; chip_size = 32, chip_size_max = 32, similarity = (:coherence, :zncc))
    @test_throws ArgumentError chip_measures(p4)
    # The tuple stays concretely typed, so the kernels specialize on it — and `chip_measures`
    # returns a tuple rather than a vector for the same reason. A `Vector{SimilarityMeasure}` has
    # an abstract eltype, which made the chip-size loop a dynamic dispatch and broke `app/`'s
    # `--trim` build outright; `===` here is what pins the tuple-ness.
    @test p2.similarity isa Tuple{Coherence,ZNCC}
    @test chip_measures(p2) isa Tuple{Coherence,ZNCC,ZNCC}
    # `measure_at` is the indexed form the loop uses, with the last entry repeating.
    @test AutoRIFT.measure_at(p2, 1) === Coherence()
    @test AutoRIFT.measure_at(p2, 2) === ZNCC()
    @test AutoRIFT.measure_at(p2, 3) === ZNCC()
    @test_throws ArgumentError params(; similarity = ())
end

@testset "a coherence pass warms the plans it actually uses" begin
    # `warm_plans!` exists to create plans on the calling task, before any worker races for the
    # planner lock. Warming the wrong *kind* is doubly wrong: it pays FFTW_MEASURE (116-347 ms per
    # size cold) for a plan never executed, and leaves the plans that ARE executed to be built
    # inside a task under the lock — exactly the contention it exists to prevent.
    #
    # This regression-tests a real defect: before `warm_plans!` took a `complex` argument, a
    # coherence run warmed `RFFT_PLANS[(72,72)]`, never touched it, and built its c2c plans lazily.
    z1 = speckle(384; seed = 0xABCD)
    z2 = circshift(z1, (5, -3))
    AutoRIFT.clear_plans!()
    autorift(z1, z2; similarity = :coherence, preprocess = :deramp, quantize = :none,
             chip_size = 32, search_radius = 20, chip_size_max = 32)
    # The complex plans are the ones a coherence surface executes.
    @test !isempty(AutoRIFT.CFFT_PLANS)
    @test !isempty(AutoRIFT.ICFFT_PLANS)
    # And no real plan was measured for a pass that cannot use one.
    @test isempty(AutoRIFT.RFFT_PLANS)
    @test isempty(AutoRIFT.IRFFT_PLANS)

    # The mirror case: a real pass must not warm complex plans.
    AutoRIFT.clear_plans!()
    autorift(abs.(z1), abs.(z2); quantize = :none, chip_size = 32, search_radius = 20,
             chip_size_max = 32)
    @test !isempty(AutoRIFT.RFFT_PLANS)
    @test isempty(AutoRIFT.CFFT_PLANS)
end

@testset "end-to-end: complex pair through autorift" begin
    # The whole pipeline on synthetic SLC, which is what the milestone is for.
    n, shift = 384, (5, -3)
    z1 = speckle(n; seed = 0xABCD)
    z2 = circshift(z1, shift)

    out = autorift(z1, z2; similarity = :coherence, preprocess = :deramp, quantize = :none,
                   chip_size = 32, search_radius = 20, chip_size_max = 32)
    @test out isa MultichipResult
    @test nmeasured(out) > 0
    # Same sign convention as the real path: `dx`/`dy` are the offset from secondary back to
    # reference, so a `circshift` of (5, -3) reads as dy = -5, dx = +3.
    @test med(filter(!isnan, out.dy)) ≈ -5 atol = 0.5
    @test med(filter(!isnan, out.dx)) ≈ 3 atol = 0.5

    # And the hierarchy: coherence at the finest chip, ZNCC above it. Both levels run, and the
    # result must still recover the shift — this is the option the milestone exists to provide.
    hier = autorift(z1, z2; similarity = (:coherence, :zncc), preprocess = :deramp,
                    quantize = :none, chip_size = 32, chip_size_max = 64, search_radius = 20)
    @test nmeasured(hier) > 0
    @test med(filter(!isnan, hier.dy)) ≈ -5 atol = 0.5
    @test med(filter(!isnan, hier.dx)) ≈ 3 atol = 0.5
end
