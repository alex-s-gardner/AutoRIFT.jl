using AutoRIFT: workspace, correlate!, prepare_chip!, next_fft_size,
                integral, integral_sq, boxsum, peak_index, peak, peak_offset,
                pyrup!, reflect101, refinement_workspace, subpixel_peak

@testset "integral images" begin
    A = Float32[1 2 3; 4 5 6; 7 8 9]
    S = integral(A)
    @test size(S) == (4, 4)
    # The leading zero row and column are what let boxsum read four corners with
    # no special case for boxes touching the first row or column.
    @test all(iszero, S[1, :])
    @test all(iszero, S[:, 1])
    @test S[end, end] == sum(A)
    @test boxsum(S, 1, 1, 3, 3) == sum(A)
    @test boxsum(S, 1, 1, 1, 1) == 1
    @test boxsum(S, 2, 2, 2, 2) == sum(A[2:3, 2:3])
    @test boxsum(S, 1, 2, 3, 2) == sum(A[1:3, 2:3])

    S2 = integral_sq(A)
    @test S2[end, end] == sum(abs2, A)
    @test boxsum(S2, 2, 2, 2, 2) == sum(abs2, A[2:3, 2:3])

    # Accumulation is Float64 even for integer input, because the variance is a
    # difference of two large nearly-equal sums and cancels badly otherwise.
    B = fill(UInt8(255), 64, 64)
    @test integral_sq(B)[end, end] == 64 * 64 * 255.0^2
    @test eltype(integral(B)) === Float64

    @test_throws DimensionMismatch AutoRIFT.integral!(zeros(3, 3), A)
end

@testset "next_fft_size" begin
    @test next_fft_size(1) == 1
    @test next_fft_size(8) == 8
    @test next_fft_size(81) == 81      # 3^4
    # A prime length is rounded up: FFTW can be an order of magnitude slower on
    # one, and the padding cost is trivial by comparison.
    @test next_fft_size(83) == 84      # 2^2 * 3 * 7
    for n in (17, 31, 83, 127, 251)
        m = next_fft_size(n)
        @test m >= n
        r = m
        for p in (2, 3, 5, 7)
            while r % p == 0
                r ÷= p
            end
        end
        @test r == 1
    end
end

@testset "prepare_chip!" begin
    ws = workspace(Float32, 8, 4)
    chip = Float32[i + j for i in 1:8, j in 1:8]
    n, ok = prepare_chip!(ws, chip)
    @test ok
    # Mean removed, so the copy sums to zero and its norm is the returned value.
    @test sum(ws.chip[1:8, 1:8]) ≈ 0 atol = 1e-4
    @test n ≈ sqrt(sum(abs2, chip .- Statistics.mean(chip)))

    # A constant chip has zero variance, so the correlation coefficient is
    # undefined at every shift. Reported as "no signal" rather than as a peak.
    _, ok = prepare_chip!(ws, fill(5.0f0, 8, 8))
    @test !ok
    # Nearly constant is also rejected: the norm would be at the scale of the
    # rounding error, and dividing by it amplifies noise into a spurious peak.
    _, ok = prepare_chip!(ws, fill(1.0f0, 8, 8) .+ Float32(1e-10) .* rand(8, 8))
    @test !ok

    @test_throws DimensionMismatch prepare_chip!(ws, zeros(Float32, 16, 16))
end

@testset "workspace validation" begin
    @test_throws "chip size must be positive" workspace(Float32, 0, 5)
    @test_throws "search radius must be positive" workspace(Float32, 8, 0)
    ws = workspace(Float32, (32, 16), (25, 10))
    @test size(ws.chip) == (16, 32)         # (rows, cols) = (y, x)
    @test size(ws.surface) == (20, 50)      # (2*ry, 2*rx)
end

@testset "exact recovery of known shifts" begin
    # The primary correctness gate: a texture displaced by a known integer amount
    # has an exactly known answer, which is a stronger check than agreement with
    # a reference implementation that has no test suite of its own.
    img = synthetic_texture(300; seed = 7)
    cs, r = 32, 25
    ws = workspace(Float32, cs, r)
    cx, cy = 150, 150
    h = cs ÷ 2
    search = img[(cy - h - r):(cy + h + r - 2), (cx - h - r):(cx + h + r - 2)]

    for (sx, sy) in ((0, 0), (3, -2), (7, 11), (-13, 5), (24, -24), (-24, 24))
        chip = img[(cy - h + sy):(cy + h - 1 + sy), (cx - h + sx):(cx + h - 1 + sx)]
        surface = correlate!(ws, search, chip, r)
        dx, dy, c = peak_offset(surface, (r, r))
        @test (dx, dy) == (sx, sy)
        # A chip cut verbatim from the search image is a perfect match.
        @test c > 0.999
    end
end

@testset "self-correlation" begin
    # An image correlated against itself must peak at zero displacement with a
    # coefficient of exactly 1: the numerator and denominator are then the same
    # quantity.
    img = synthetic_texture(120; seed = 3)
    cs, r = 32, 10
    ws = workspace(Float32, cs, r)
    h = cs ÷ 2
    chip = img[(60 - h):(60 + h - 1), (60 - h):(60 + h - 1)]
    search = img[(60 - h - r):(60 + h + r - 2), (60 - h - r):(60 + h + r - 2)]
    surface = correlate!(ws, search, chip, r)
    dx, dy, c = peak_offset(surface, (r, r))
    @test (dx, dy) == (0, 0)
    @test c ≈ 1.0 atol = 1e-5
    # ZNCC is bounded on [-1, 1]; a violation means the normalisation is wrong.
    @test all(-1.0001f0 .<= surface .<= 1.0001f0)
end

@testset "degenerate chip" begin
    # A constant chip yields no measurement. The reference implementation reports
    # the search-window corner instead, because v2.0.0 deleted the guard that used
    # to skip these points -- so over masked or featureless terrain it produces a
    # systematic corner-pinned bias. Reporting nothing is the honest answer.
    ws = workspace(Float32, 16, 8)
    search = synthetic_texture(31; seed = 1)
    surface = correlate!(ws, search, fill(0.5f0, 16, 16), 8)
    @test all(iszero, surface)
end

@testset "surface geometry" begin
    ws = workspace(Float32, 32, 25)
    search = synthetic_texture(81; seed = 2)
    chip = search[26:57, 26:57]
    surface = correlate!(ws, search, chip, 25)
    # Follows from the asymmetric search window, and the peak-offset arithmetic
    # depends on it.
    @test size(surface) == (50, 50)

    # A symmetric window is the wrong size, and saying so explicitly is better
    # than reading out of bounds.
    @test_throws DimensionMismatch correlate!(ws, synthetic_texture(82), chip, 25)
    @test_throws DimensionMismatch correlate!(ws, search, chip, 30)
end

@testset "non-square geometry" begin
    # Chip and radius are independent per axis throughout.
    csx, csy, rx, ry = 32, 16, 25, 10
    ws = workspace(Float32, (csx, csy), (rx, ry))
    img = synthetic_texture(200; seed = 5)
    cx, cy = 100, 100
    search = img[(cy - csy ÷ 2 - ry):(cy + csy ÷ 2 + ry - 2),
                 (cx - csx ÷ 2 - rx):(cx + csx ÷ 2 + rx - 2)]
    for (sx, sy) in ((0, 0), (5, -3), (-9, 7))
        chip = img[(cy - csy ÷ 2 + sy):(cy + csy ÷ 2 - 1 + sy),
                   (cx - csx ÷ 2 + sx):(cx + csx ÷ 2 - 1 + sx)]
        surface = correlate!(ws, search, chip, (rx, ry))
        @test size(surface) == (2ry, 2rx)
        dx, dy, _ = peak_offset(surface, (rx, ry))
        @test (dx, dy) == (sx, sy)
    end
end

@testset "zero allocations" begin
    # A correctness property, not a performance one: allocating once per grid point
    # would be invisible in a microbenchmark and ruinous across millions of pairs.
    ws = workspace(Float32, 32, 25)
    search = synthetic_texture(81; seed = 4)
    chip = search[26:57, 26:57]
    correlate!(ws, search, chip, 25)        # compile
    @test @allocated(correlate!(ws, search, chip, 25)) == 0

    surface = correlate!(ws, search, chip, 25)
    peak_offset(surface, (25, 25))
    @test @allocated(peak_offset(surface, (25, 25))) == 0
end

@testset "direct and FFT paths agree" begin
    # Two strategies for the same numerator, chosen by a threshold. They must agree
    # to rounding, or the threshold would introduce a discontinuity in the output
    # as chip size crosses it.
    img = synthetic_texture(400; seed = 11)
    cs, r = 64, 25
    h = cs ÷ 2
    chip = img[(200 - h):(200 + h - 1), (200 - h):(200 + h - 1)]
    search = img[(200 - h - r):(200 + h + r - 2), (200 - h - r):(200 + h + r - 2)]

    ws = workspace(Float32, cs, r)
    direct = copy(correlate!(ws, search, chip, r))

    # Force the FFT path by lowering the threshold below this case's work.
    old = AutoRIFT.DIRECT_THRESHOLD
    @eval AutoRIFT const DIRECT_THRESHOLD = 1
    try
        ws2 = workspace(Float32, cs, r)
        viafft = copy(correlate!(ws2, search, chip, r))
        @test peak_index(direct) == peak_index(viafft)
        @test maximum(abs.(direct .- viafft)) < 1e-4
    finally
        @eval AutoRIFT const DIRECT_THRESHOLD = $old
    end
end
