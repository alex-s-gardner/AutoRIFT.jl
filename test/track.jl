using AutoRIFT: ImagePair, PointSet, pointset, gridpoints, params, track, track!,
                displacement_field, nmeasured, DisplacementField, Highpass

# Recovered displacement is secondary-to-reference, the negative of feature motion; see
# `peak_offset`'s docstring. Negating here keeps the tests reading in the same direction
# as the shift they impose.
motion(d) = (-d.dx, -d.dy)

function median_of(v)
    s = sort(collect(v))
    isempty(s) && return NaN
    return isodd(length(s)) ? s[(end + 1) ÷ 2] : (s[end ÷ 2] + s[end ÷ 2 + 1]) / 2
end

@testset "exact recovery over a grid" begin
    # The primary correctness gate, now at the level of a whole pass rather than a single
    # point: every searchable point must recover the imposed integer shift.
    #
    # With refinement off the recovery is *exact* — every point returns the integer, which
    # is the strong statement about the correlation itself.
    #
    # With refinement on it lands within a few quantization steps. Measured worst case is
    # 3/upsampling, not 1: the cascade's composite interpolant is a smooth reconstruction
    # of a sampled surface, and its maximum need not fall on the sample that the true
    # displacement occupies. That is inherent to the method rather than a defect — and it
    # is why the reference prefers this over a parabola fit, which would instead bias
    # *toward* the integer and hide the error as false precision.
    #
    # Both are asserted because they check different things: the exactness of the
    # correlation, and the bound on the refinement.
    up = 64
    refine_steps = 3
    for (sx, sy) in ((0, 0), (5, -3), (-7, 11), (20, 20))
        ref, sec = shifted_pair(400, (sx, sy); T = Float32)
        pair = ImagePair(ref, sec)
        pts = gridpoints((400, 400), 32; chip_size = 32, search_radius = 25)

        d = track(pair, pts, params(; subpixel = :none))
        @test nmeasured(d) == length(d)          # nothing failed
        @test all(d.searched)
        mx, my = motion(d)
        @test all(==(Float32(sx)), filter(!isnan, mx))
        @test all(==(Float32(sy)), filter(!isnan, my))
        # A chip cut from a shifted copy of the same texture correlates strongly.
        @test median_of(filter(!isnan, d.correlation)) > 0.9

        d = track(pair, pts, params(; subpixel = :pyramid, upsampling = up))
        mx, my = motion(d)
        @test all(v -> abs(v - sx) <= refine_steps / up, filter(!isnan, mx))
        @test all(v -> abs(v - sy) <= refine_steps / up, filter(!isnan, my))
    end
end

@testset "output shape follows the point set" begin
    ref, sec = shifted_pair(300, (3, 0); T = Float32)
    pair = ImagePair(ref, sec)

    grid = gridpoints((300, 300), 32; chip_size = 32, search_radius = 25)
    dg = track(pair, grid, params())
    @test size(dg) == size(grid)
    @test ndims(dg.dx) == 2

    # Scattered points: the motivating case, and the one the reference cannot express.
    # Arbitrary fractional coordinates, unsorted.
    xs = [100.0, 187.5, 220.25, 140.0]
    ys = [150.0, 120.0, 200.75, 99.5]
    scat = pointset(xs, ys; chip_size = 32, search_radius = 25)
    ds = track(pair, scat, params())
    @test ndims(ds.dx) == 1
    @test length(ds) == 4
    @test nmeasured(ds) == 4
    @test all(v -> abs(v - 3) <= 0.05, filter(!isnan, -ds.dx))
end

@testset "no measurement is distinguishable from zero" begin
    # The distinction the reference loses: a point that was not searched, or whose chip
    # carried no texture, must not report a displacement of zero.
    ref, sec = shifted_pair(300, (0, 0); T = Float32)
    pair = ImagePair(ref, sec)

    pts = gridpoints((300, 300), 32; chip_size = 32, search_radius = 25)
    # Zero the radius at one point: never searched.
    pts.radius_x[2, 2] = 0
    pts.radius_y[2, 2] = 0
    d = track(pair, pts, params())
    @test !d.searched[2, 2]
    @test isnan(d.dx[2, 2])
    @test isnan(d.correlation[2, 2])
    # Its neighbours, which have a genuine zero displacement, report exactly that.
    @test d.searched[3, 3]
    @test abs(d.dx[3, 3]) <= 3 / 64      # zero to within the refinement's resolution
end

@testset "degenerate chip yields no measurement" begin
    # A constant region has zero variance, so the correlation coefficient is undefined at
    # every shift. The reference reports the search-window corner, which over masked or
    # featureless terrain is a systematic corner-pinned bias; this reports nothing.
    n = 300
    ref = synthetic_texture(n; seed = 4)
    sec = copy(ref)
    flat = 100:200
    ref[flat, flat] .= 0.5f0        # a featureless patch in both images
    sec[flat, flat] .= 0.5f0

    pair = ImagePair(ref, sec)
    pts = gridpoints((n, n), 32; chip_size = 32, search_radius = 25)
    d = track(pair, pts, params())

    # Find a point well inside the flat region.
    inflat = findall(i -> 130 <= pts.x[i] <= 170 && 130 <= pts.y[i] <= 170,
                     eachindex(pts))
    @test !isempty(inflat)
    for i in inflat
        @test d.searched[i]          # it was attempted...
        @test isnan(d.dx[i])         # ...and honestly reported no result
    end
    # Textured points still resolve.
    @test nmeasured(d) > 0
end

@testset "a-priori displacement centres the search" begin
    # The prior lets a modest radius cover large motion: the chip follows it while the
    # window stays on the point, so the answer remains relative to the point.
    sx, sy = 18, -14
    ref, sec = shifted_pair(400, (sx, sy); T = Float32)
    pair = ImagePair(ref, sec)

    # A radius too small to reach the true displacement on its own.
    tight = gridpoints((400, 400), 32; chip_size = 32, search_radius = 8)
    d = track(pair, tight, params())
    mx, my = motion(d)
    found = filter(!isnan, mx)
    @test isempty(found) || abs(median_of(found) - sx) > 2   # cannot reach it

    # With the prior, the same radius succeeds.
    withprior = gridpoints((400, 400), 32; chip_size = 32, search_radius = 8,
                           dx_prior = -Float64(sx), dy_prior = -Float64(sy))
    d = track(pair, withprior, params())
    mx, my = motion(d)
    @test median_of(filter(!isnan, mx)) ≈ sx atol = 0.5
    @test median_of(filter(!isnan, my)) ≈ sy atol = 0.5
end

@testset "threaded and serial agree bitwise" begin
    # Each point writes a distinct output element with no reduction, so threading cannot
    # change the result. Asserted rather than argued, at more than one thread count.
    ref, sec = shifted_pair(400, (6, -4); T = Float32)
    pair = ImagePair(ref, sec)
    pts = gridpoints((400, 400), 24; chip_size = 32, search_radius = 20)

    ser = track(pair, pts, params(; threaded = false))
    par = track(pair, pts, params(; threaded = true))
    @test all(isequal.(ser.dx, par.dx))
    @test all(isequal.(ser.dy, par.dy))
    @test all(isequal.(ser.correlation, par.correlation))
    @test ser.searched == par.searched
end

@testset "subpixel refinement" begin
    # A fractional shift is recovered to within the quantization step, and the integer-only
    # path is what the pyramid's coarse pass uses.
    ref, sec = shifted_pair(400, (4.5, -2.25); T = Float32)
    pair = ImagePair(ref, sec)
    pts = gridpoints((400, 400), 32; chip_size = 32, search_radius = 20)

    d = track(pair, pts, params(; subpixel = :pyramid, upsampling = 32))
    mx, my = motion(d)
    @test median_of(filter(!isnan, mx)) ≈ 4.5 atol = 0.3
    @test median_of(filter(!isnan, my)) ≈ -2.25 atol = 0.3

    # Integer-only: every result lands on a whole pixel.
    d = track(pair, pts, params(; subpixel = :none))
    @test all(v -> isnan(v) || v == round(v), d.dx)
    @test all(v -> isnan(v) || v == round(v), d.dy)

    # The override exists so the pyramid can force the coarse pass to skip refinement
    # without rebuilding its parameters.
    p = params(; subpixel = :pyramid, upsampling = 32)
    d = track(pair, pts, p; subpixel = AutoRIFT.NoRefine())
    @test all(v -> isnan(v) || v == round(v), d.dx)
end

@testset "non-square geometry" begin
    # Chip and radius are independent per axis all the way through.
    ref, sec = shifted_pair(400, (5, -3); T = Float32)
    pair = ImagePair(ref, sec)
    pts = gridpoints((400, 400), 32; chip_size = 32, chip_aspect = 0.5,
                     search_radius_x = 25, search_radius_y = 10)
    d = track(pair, pts, params())
    mx, my = motion(d)
    @test median_of(filter(!isnan, mx)) ≈ 5 atol = 0.05
    @test median_of(filter(!isnan, my)) ≈ -3 atol = 0.05
end

@testset "quantized input" begin
    # UInt8 is the reference's default correlation type, and the whole pipeline has to work
    # in it: an 8-bit chip still resolves an integer shift exactly.
    ref, sec = shifted_pair(400, (7, 5); T = Float32)
    pair = AutoRIFT.quantize(ImagePair(ref, sec), :uint8)
    @test eltype(pair) === UInt8
    pts = gridpoints((400, 400), 32; chip_size = 32, search_radius = 25)
    d = track(pair, pts, params())
    mx, my = motion(d)
    @test median_of(filter(!isnan, mx)) ≈ 7 atol = 0.05
    @test median_of(filter(!isnan, my)) ≈ 5 atol = 0.05
end

@testset "points near and beyond the edge" begin
    # A scattered point set is caller-supplied, so a point may sit anywhere. Three regimes,
    # and the distinction between them is the validity mask rather than the padding: the
    # images are padded so the inner loop needs no bounds test, but padding is not data.
    ref, sec = shifted_pair(200, (0, 0); T = Float32)
    pair = ImagePair(ref, sec)
    pts = pointset([100.0, 5.0, 5000.0], [100.0, 100.0, 100.0];
                   chip_size = 32, search_radius = 25)
    d = track(pair, pts, params())

    @test d.searched[1]        # wholly inside
    # Partially overlapping: this chip is 20 of 32 columns of real imagery, so it carries
    # genuine information and is searched. Rejecting it would discard usable data at every
    # scene edge.
    @test d.searched[2]
    @test !isnan(d.dx[2])
    # Wholly outside: the chip is entirely padding, which would correlate perfectly with
    # any other patch of padding. Skipped.
    @test !d.searched[3]
    @test isnan(d.dx[3])
end

@testset "a chip of pure padding is rejected" begin
    # The case the validity mask exists for, isolated. A point far enough outside that its
    # chip contains no real pixel must be skipped rather than correlate padding against
    # padding — which would report a confident, meaningless zero.
    ref, sec = shifted_pair(200, (0, 0); T = Float32)
    pair = ImagePair(ref, sec)
    # Just past the edge by more than a chip half-width.
    pts = pointset([-40.0, 250.0], [100.0, 100.0]; chip_size = 32, search_radius = 25)
    d = track(pair, pts, params())
    @test !any(d.searched)
    @test all(isnan, d.dx)
end

@testset "a masked-out region yields no measurement" begin
    # The same mechanism driven by an explicit mask rather than by the image edge, which is
    # how a cloud or shadow mask reaches the correlator.
    n = 300
    ref = synthetic_texture(n; seed = 12)
    sec = copy(ref)
    m = trues(n, n)
    m[100:200, 100:200] .= false            # masked, though the pixels look fine
    pair = ImagePair(ref, sec; reference_valid = m, secondary_valid = m)
    pts = gridpoints((n, n), 32; chip_size = 32, search_radius = 25)
    d = track(pair, pts, params())

    # Grid points sit every 32 px, so the window has to be wide enough to contain one.
    # Inset from the mask edge so the whole chip is masked, not just its centre.
    inside = findall(i -> 130 <= pts.x[i] <= 170 && 130 <= pts.y[i] <= 170,
                     eachindex(pts))
    @test !isempty(inside)
    @test all(i -> !d.searched[i], inside)
    @test nmeasured(d) > 0                  # unmasked points still resolve
end

@testset "filtering feeds through" begin
    # The pipeline order this package is built around: filter, then correlate. Both paths
    # must find the same displacement, and filtering must help where it is supposed to.
    #
    # Where it is supposed to help is narrower than it first appears, and worth stating
    # because it bounds what preprocessing is for. ZNCC removes each window's own mean and
    # scale, so a brightness change that is *affine across a chip* is already handled — and
    # a gentle ramp is very nearly affine over 32 pixels. Filtering only pays once the
    # variation within a single chip is large. Measured on this texture: at 12% variation
    # across a chip the high-pass slightly lowers the peak (0.891 -> 0.850, since it also
    # attenuates signal), at 80% it raises it (0.808 -> 0.849), and at 400% more so.
    #
    # So the assertion is made at a gradient steep enough for the filter to earn its place,
    # and the shallow case is asserted only to agree on the displacement.
    n = 400
    tex = synthetic_texture(n; seed = 9)
    pts = gridpoints((n, n), 32; chip_size = 32, search_radius = 20)
    p = params()

    for (slope, filter_helps) in ((0.0015f0, false), (0.01f0, true))
        ramp = Float32[0.4f0 + slope * j for _ in 1:n, j in 1:n]
        sec = tex .* ramp .+ (ramp .- 0.4f0)
        raw = track(ImagePair(tex, sec), pts, p)
        filt = track(AutoRIFT.preprocess(ImagePair(tex, sec), Highpass(; width = 5)),
                     pts, p)

        # The scene did not move, and neither path may invent motion.
        @test median_of(filter(!isnan, -raw.dx)) ≈ 0 atol = 0.05
        @test median_of(filter(!isnan, -filt.dx)) ≈ 0 atol = 0.05

        craw = median_of(filter(!isnan, raw.correlation))
        cfilt = median_of(filter(!isnan, filt.correlation))
        filter_helps && @test cfilt > craw
        # Either way the filtered correlation stays high: the filter must not destroy the
        # texture it exists to expose.
        @test cfilt > 0.8
    end
end

@testset "padding is only paid when needed" begin
    # `gridpoints` insets by the chip half-extent plus the search radius, so a gridded pass
    # never reads outside the image and padding it would be pure waste — measured at 1.2 ms
    # and 10.6 MB per call on 1024², which is ~10% of a sparse pass. Asserted as an
    # allocation bound rather than a timing, since that is the part that is deterministic.
    n = 512
    pair = ImagePair(synthetic_texture(n; seed = 1), synthetic_texture(n; seed = 2))
    grid = gridpoints((n, n), 32; chip_size = 32, search_radius = 25)
    out = displacement_field(grid)
    p = params(; subpixel = :none)
    track!(out, pair, grid, p)                     # compile
    # Three padded copies of a 512² image would be ~3 MB; the workspaces and point shift are
    # a small fraction of that.
    @test @allocated(track!(out, pair, grid, p)) < 1_000_000

    # A scattered point set that *does* fall outside still works, which is what the padding
    # is there for. This point's chip is partly outside the image.
    edge = pointset([12.0], [256.0]; chip_size = 32, search_radius = 25)
    d = track(pair, edge, p)
    @test d.searched[1]
    @test !isnan(d.dx[1])
end

@testset "validation" begin
    ref, sec = shifted_pair(200, (0, 0); T = Float32)
    pair = ImagePair(ref, sec)
    pts = gridpoints((200, 200), 32; chip_size = 32, search_radius = 25)
    @test_throws DimensionMismatch track!(
        displacement_field(pointset([1.0], [1.0])), pair, pts, params())
end
