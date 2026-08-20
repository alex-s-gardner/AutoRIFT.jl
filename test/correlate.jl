using AutoRIFT: workspace, correlate!, prepare_chip!, next_fft_size,
                integral, integral_sq, boxsum, peak_index, peak, peak_offset,
                pyrup!, reflect101, refinement_workspace, subpixel_peak,
                clear_workspaces!, ImagePair, gridpoints, params, track

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

    # Deliberately *not* the smallest product of small primes. FFTW's radix-2 codelets are
    # far better optimised than its others, so a 2-heavy length beats a smaller 3-heavy one:
    # 81 is itself 3^4 and takes 14.1 us, where 96 = 2^5*3 takes 8.7 us. So the size chosen
    # for 81 must be larger than 81.
    @test next_fft_size(81) > 81
    @test next_fft_size(43) == 48       # matches the measured winner

    # A power of two is chosen when it is close enough, but not unconditionally: at 81 the
    # nearest is 128, and padding that far costs more than the radix-2 advantage buys.
    @test next_fft_size(113) == 128
    @test next_fft_size(81) < 128

    for n in (17, 31, 43, 81, 83, 113, 127, 163, 177, 251)
        m = next_fft_size(n)
        # Never smaller than requested, and never padded so far that the extra arithmetic
        # could dominate the transform it is meant to speed up.
        @test m >= n
        @test m <= ceil(Int, 1.5n)
        # Still 7-smooth, which is the constraint FFTW actually needs.
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

@testset "workspace pool" begin
    # Pooling exists because a threaded 1024² pass allocated 56.3 MiB against the serial path's
    # 34.0, and GC was 21.8% of it — ~96 workspaces per run, 3 chip sizes x 2 passes x 16 chunks.
    # The correctness requirement is that reuse changes nothing at all.
    clear_workspaces!()

    # Taken workspaces are out of the pool, so two concurrent chunks cannot share buffers. That is
    # the same guarantee per-chunk allocation gave.
    a = AutoRIFT.take_workspace!(Float32, 32, 25)
    b = AutoRIFT.take_workspace!(Float32, 32, 25)
    @test a !== b
    AutoRIFT.give_workspace!(a)
    # Returned, so the next take reuses it rather than allocating.
    @test AutoRIFT.take_workspace!(Float32, 32, 25) === a
    AutoRIFT.give_workspace!(a)
    AutoRIFT.give_workspace!(b)

    # Keyed on exact geometry, never "large enough". Reusing an oversized workspace was measured
    # and rejected: the FFT buffer is sized from the workspace's own extents, so a chip-32 point in
    # a chip-128 workspace runs a 192-point transform where 84 would do — 5x the arithmetic and a
    # different rounding, 4.5e-8 from the exact answer. A pooled workspace must be the size a fresh
    # one would be, or it is a different algorithm.
    clear_workspaces!()
    small = AutoRIFT.take_workspace!(Float32, 32, 25)
    AutoRIFT.give_workspace!(small)
    @test AutoRIFT.take_workspace!(Float32, 128, 25) !== small
    @test size(AutoRIFT.take_workspace!(Float32, 32, 25).chip) == (32, 32)

    # The element type is deliberately not part of the key: no buffer here has the image's type,
    # which is the same reason `CorrelationWorkspace` carries no `T`.
    clear_workspaces!()
    w8 = AutoRIFT.take_workspace!(UInt8, 32, 25)
    AutoRIFT.give_workspace!(w8)
    @test AutoRIFT.take_workspace!(Float32, 32, 25) === w8

    # Refinement workspaces pool on `upsampling` alone, since that and the fixed 5x5 patch
    # determine every extent.
    clear_workspaces!()
    r1 = AutoRIFT.take_refinement!(64)
    AutoRIFT.give_refinement!(r1)
    @test AutoRIFT.take_refinement!(64) === r1
    AutoRIFT.give_refinement!(r1)
    @test AutoRIFT.take_refinement!(32) !== r1

    clear_workspaces!()
end

@testset "pooling changes no result" begin
    # The claim that matters: a pooled workspace must give bit-identical answers to a fresh one,
    # on a cold pool and a warm one, serial and threaded. Buffers are not cleared on return —
    # every one is fully written before being read — so this is what proves that safe.
    ref, sec = shifted_pair(400, (6, -4); T = Float32)
    pair = ImagePair(ref, sec)
    pts = gridpoints((400, 400), 32; chip_size = 32, search_radius = 25)
    p = params()

    clear_workspaces!()
    cold = track(pair, pts, p)          # empty pool: every workspace freshly built
    warm = track(pair, pts, p)          # warm pool: every workspace reused
    @test all(isequal.(cold.dx, warm.dx))
    @test all(isequal.(cold.dy, warm.dy))
    @test all(isequal.(cold.correlation, warm.correlation))

    # And across the threading boundary, where several chunks take from the pool at once.
    par = track(pair, pts, params(; threaded = true))
    @test all(isequal.(cold.dx, par.dx))
    @test all(isequal.(cold.dy, par.dy))
    @test all(isequal.(cold.correlation, par.correlation))
end
