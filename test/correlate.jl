using AutoRIFT: workspace, correlate!, prepare_chip!, next_fft_size,
                integral, integral_sq, boxsum, peak_index, peak, peak_offset,
                pyrup!, reflect101, refinement_workspace, subpixel_peak,
                clear_workspaces!, ImagePair, gridpoints, params, track, extent

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

@testset "integral_both! is exactly the two separate tables" begin
    # `integral_both!` traverses four columns at once, which reorders memory access but not
    # arithmetic: each running sum still accumulates down its own column in row order, and the
    # writes across a row still chain left to right. So the fused, blocked form must agree with
    # the one-at-a-time functions *bitwise*, not to a tolerance — a tolerance here would hide
    # exactly the reassociation the blocking is required not to introduce.
    #
    # Widths spanning every remainder mod 4, so the blocked body and its scalar tail are both
    # exercised, and one width below the block so only the tail runs.
    for (m, n) in ((3, 3), (7, 8), (8, 9), (9, 10), (16, 11), (27, 27), (32, 81))
        A = Float32.(sinpi.((1:m) ./ m) .* cospi.(((1:n)') ./ n) .* 37)
        S = Matrix{Float64}(undef, m + 1, n + 1)
        S2 = similar(S)
        AutoRIFT.integral_both!(S, S2, A)
        @test S == AutoRIFT.integral(A)
        @test S2 == AutoRIFT.integral_sq(A)
    end

    # Integer input takes the same path with a widening conversion, and 255² over a large window
    # is where a narrower accumulator would start losing bits.
    B = fill(UInt8(255), 64, 64)
    S = Matrix{Float64}(undef, 65, 65)
    S2 = similar(S)
    AutoRIFT.integral_both!(S, S2, B)
    @test S == AutoRIFT.integral(B)
    @test S2 == AutoRIFT.integral_sq(B)

    S3 = zeros(3, 3)
    @test_throws DimensionMismatch AutoRIFT.integral_both!(S3, S3, Float32[1 2 3; 4 5 6; 7 8 9])
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

    # Four accumulators per column, so the mean-removed copy must still be exactly what a single
    # running sum produces: the partial sums are combined pairwise, which changes the *norm* at the
    # last bit but cannot change `Float32(x - mean)` once `mean` is fixed. Heights spanning every
    # remainder mod 4 exercise the unrolled body and its tail.
    for h in (1, 2, 3, 4, 5, 7, 8, 13, 32)
        w = 6
        ws2 = workspace(Float32, (w, h), 4)
        c = Float32[i * 3 - j * 7 + i * j for i in 1:h, j in 1:w]
        nrm, ok = prepare_chip!(ws2, c)
        @test ok
        mean = sum(Float64, c) / length(c)
        @test ws2.chip[1:h, 1:w] == Float32.(Float64.(c) .- mean)
        @test nrm ≈ sqrt(sum(abs2, Float64.(c) .- mean)) rtol = 1e-13
    end

    # A strided view, which is what `track!` actually passes: the chip is a window into the
    # secondary image, so its columns are not adjacent in memory.
    img = Float32[i + 2j + i * j / 8 for i in 1:40, j in 1:40]
    ws3 = workspace(Float32, 8, 4)
    v = @view img[9:16, 21:28]
    nv, okv = prepare_chip!(ws3, v)
    @test okv
    mv = sum(Float64, v) / length(v)
    @test ws3.chip[1:8, 1:8] == Float32.(Float64.(v) .- mv)
    @test nv ≈ sqrt(sum(abs2, Float64.(v) .- mv)) rtol = 1e-13
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

    for (sx, sy) in ((0, 0), (3, -2), (7, 11), (-13, 5), (24, -24), (-24, 24))
        # The window stays put and the chip is cut `(sx, sy)` away, so the peak must land there.
        chip, search = chip_and_window(img, (cy, cx), cs, r; shift = (sy, sx))
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
    chip, search = chip_and_window(img, (60, 60), cs, r)
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
    chip, search = chip_and_window(img, (200, 200), cs, r)

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

    # A scalar, a plain tuple and an `Extent` name the same geometry, so all three must key the
    # same pool entry. `Extent isa Tuple` is `false`, so normalizing by a tuple test would send an
    # extent down the scalar branch — a workspace sized from `(x, x)` where `x` is the whole pair,
    # and a key that never matches the other two.
    clear_workspaces!()
    s = AutoRIFT.take_workspace!(Float32, 32, 25)
    AutoRIFT.give_workspace!(s)
    @test AutoRIFT.take_workspace!(Float32, (32, 32), (25, 25)) === s
    AutoRIFT.give_workspace!(s)
    @test AutoRIFT.take_workspace!(Float32, extent(32), extent(25)) === s
    AutoRIFT.give_workspace!(s)
    @test length(AutoRIFT.WORKSPACE_POOL) == 1
    @test s.max_chip === extent(32)
    @test s.max_radius === extent(25)

    # And a non-square extent is read as the pair it is, not as its first component.
    ws = workspace(Float32, extent((X = 32, Y = 16)), extent((X = 25, Y = 10)))
    @test size(ws.chip) == (16, 32)
    @test size(ws.surface) == (20, 50)

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

@testset "a prepared window changes no result" begin
    # `correlate_prepared!` skips the window's transform and its integral tables, reading what
    # `prepare_window!` left in the workspace. Bit-identical to `correlate!` is the whole claim:
    # the same spectrum and the same tables, computed once rather than once per chip. Equality
    # rather than a tolerance, because nothing here is reordered — only reused.
    img = synthetic_texture(400; seed = 23)
    for measure in (ZNCC(), NCC()), (cs, r) in ((16, 20), (32, 25), (64, 25), (32, 6))
        chip, search = chip_and_window(img, (200, 200), cs, r)

        ws = workspace(Float32, cs, r)
        plain = copy(correlate!(ws, search, chip, r; measure))

        ws2 = workspace(Float32, cs, r)
        AutoRIFT.prepare_window!(ws2, search, measure)
        prepared = copy(AutoRIFT.correlate_prepared!(ws2, search, chip, r; measure))
        @test plain == prepared
    end

    # Through the rotation search, which is the only caller: every angle after the first reads a
    # spectrum computed for the first, so a stale-window bug would show up here and nowhere else.
    # Compared against the same loop driven by `correlate!`, which recomputes everything per angle.
    chip, search = chip_and_window(img, (200, 200), 32, 25)
    rot = AutoRIFT.RotationSearch((-4.0, -2.0, 0.0, 2.0, 4.0), 0.0)
    ws = workspace(Float32, 32, 25)
    hoisted = copy(AutoRIFT._correlate_rotations!(ws, search, chip, (25, 25), ZNCC(), rot))

    ws2 = workspace(Float32, 32, 25)
    best, found, unhoisted = -Inf32, false, zeros(Float32, 50, 50)
    for a in AutoRIFT.angles(rot)
        s = correlate!(ws2, search, AutoRIFT._rotate_chip(ws2, chip, a), (25, 25); measure = ZNCC())
        AutoRIFT.degenerate(ws2) && continue
        pk = maximum(s)
        pk > best && (best = pk; copyto!(unhoisted, s); found = true)
    end
    @test found
    @test hoisted == unhoisted
end
