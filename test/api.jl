using AutoRIFT: autorift, autorift!, reinit!, init, Cache, MultichipResult, nmeasured,
                pointset, gridpoints, params, Params, ZNCC, Highpass, QuantizeUInt8,
                PyramidRefine, GardnerFilter

med(v) = (s = sort(collect(v)); isempty(s) ? NaN : s[(length(s) + 1) ÷ 2])
motion(r) = (med(filter(!isnan, -r.dx)), med(filter(!isnan, -r.dy)))

@testset "one-shot autorift" begin
    # The call most users will make, and the shape the whole package exists to support.
    ref, sec = shifted_pair(512, (6, -4); T = Float32)
    r = autorift(ref, sec; chip_size = 32, search_radius = 25)

    @test r isa MultichipResult
    @test nmeasured(r) == length(r)
    mx, my = motion(r)
    @test mx ≈ 6 atol = 0.05
    @test my ≈ -4 atol = 0.05
    # Every documented layer is present and the right shape.
    @test size(r.dx) == size(r.dy) == size(r.correlation) == size(r.chip_size)
    @test size(r.interpolated) == size(r.dx)
    @test eltype(r.chip_size) === UInt16
end

@testset "keywords reach the pipeline" begin
    ref, sec = shifted_pair(512, (5, 0); T = Float32)

    # Grid spacing changes the output resolution, which is the most visible keyword.
    coarse = autorift(ref, sec; chip_size = 32, grid_spacing = 64, search_radius = 25)
    fine = autorift(ref, sec; chip_size = 32, grid_spacing = 24, search_radius = 25)
    @test length(fine) > length(coarse)

    # Refinement off gives exactly integral displacements.
    r = autorift(ref, sec; chip_size = 32, search_radius = 25, subpixel = :none)
    @test all(v -> isnan(v) || v == round(v), r.dx)

    # A bad keyword is rejected at the boundary, naming what was wrong, rather than failing
    # somewhere inside a chip-size level.
    @test_throws "not recognised" autorift(ref, sec; preprocess = :sharpen)
    @test_throws "multiple of 4" autorift(ref, sec; chip_size = 30)
end

@testset "validity masks are honoured" begin
    # How a cloud or shadow mask reaches the correlator. The pixels themselves look fine, so
    # only the mask can exclude them.
    n = 512
    ref, sec = shifted_pair(n, (5, -3); T = Float32)
    m = trues(n, n)
    m[100:250, 100:250] .= false
    r = autorift(ref, sec; chip_size = 32, search_radius = 25,
                 reference_valid = m, secondary_valid = m)
    unmasked = autorift(ref, sec; chip_size = 32, search_radius = 25)

    @test nmeasured(r) < nmeasured(unmasked)
    # What survives is still correct: masking removes points, it does not corrupt them.
    mx, my = motion(r)
    @test mx ≈ 5 atol = 0.05
    @test my ≈ -3 atol = 0.05
end

@testset "cache matches one-shot exactly" begin
    # The batch path must not be a different algorithm. Bitwise, since both run the same code
    # on the same input and any difference would mean the cache is carrying stale state.
    ref, sec = shifted_pair(512, (6, -4); T = Float32)
    kw = (; chip_size = 32, search_radius = 25)

    direct = autorift(ref, sec; kw...)
    c = init(ref, sec; kw...)
    cached = autorift!(c)

    @test c isa Cache
    @test all(isequal.(direct.dx, cached.dx))
    @test all(isequal.(direct.dy, cached.dy))
    @test all(isequal.(direct.correlation, cached.correlation))
    @test direct.chip_size == cached.chip_size
    @test direct.interpolated == cached.interpolated
end

@testset "reinit! advances to the next pair" begin
    # The property that makes the batch driver safe: a reused cache must give the same answer
    # as a fresh one. If it did not, throughput would come at the cost of correctness.
    a_ref, a_sec = shifted_pair(512, (6, -4); T = Float32)
    b_ref, b_sec = shifted_pair(512, (-9, 3); T = Float32)
    kw = (; chip_size = 32, search_radius = 25)

    c = init(a_ref, a_sec; kw...)
    autorift!(c)
    reinit!(c; reference = b_ref, secondary = b_sec)
    reused = autorift!(c)
    fresh = autorift(b_ref, b_sec; kw...)

    @test all(isequal.(reused.dx, fresh.dx))
    @test all(isequal.(reused.dy, fresh.dy))
    mx, my = motion(reused)
    @test mx ≈ -9 atol = 0.05
    @test my ≈ 3 atol = 0.05

    # Swapping one image advances a time series while holding the other fixed.
    c2 = init(a_ref, a_sec; kw...)
    reinit!(c2; secondary = b_sec)
    # The reference must be re-prepared from the *original*, not from the already-filtered and
    # quantized array the cache is holding — filtering twice would silently change the answer.
    @test AutoRIFT.imagepair(c2).reference == AutoRIFT.imagepair(init(a_ref, b_sec; kw...)).reference

    # Nothing to change is a no-op, not an error.
    before = AutoRIFT.imagepair(c2)
    @test reinit!(c2) === c2
    @test AutoRIFT.imagepair(c2) === before
end

@testset "repeated runs do not recompute" begin
    # Idempotence, and the reason the cache tracks freshness at all: a driver that calls
    # `autorift!` twice by accident should not pay twice.
    ref, sec = shifted_pair(384, (4, 2); T = Float32)
    c = init(ref, sec; chip_size = 32, search_radius = 20)
    r1 = autorift!(c)
    r2 = autorift!(c)
    @test r1 === r2               # the same object, not merely an equal one

    # After a reinit the result is recomputed.
    ref2, sec2 = shifted_pair(384, (-6, 1); T = Float32)
    reinit!(c; reference = ref2, secondary = sec2)
    r3 = autorift!(c)
    @test r3 !== r1
    @test med(filter(!isnan, -r3.dx)) ≈ -6 atol = 0.05
end

@testset "cache reuse is cheaper than rebuilding" begin
    # The whole point of the lifecycle. Measured as allocation rather than time, since that is
    # the deterministic part: `init` builds the grid and warms the plans, `autorift!` does not.
    ref, sec = shifted_pair(384, (3, 3); T = Float32)
    kw = (; chip_size = 32, search_radius = 20)
    c = init(ref, sec; kw...)
    autorift!(c)                                   # compile, and warm the plan cache

    ref2, sec2 = shifted_pair(384, (5, 1); T = Float32)
    reused = @allocated begin
        reinit!(c; reference = ref2, secondary = sec2)
        autorift!(c)
    end
    rebuilt = @allocated autorift(ref2, sec2; kw...)
    @test reused < rebuilt
end

@testset "cache rejects a size change" begin
    # The grid and outputs are sized to the original, so a different image size needs a new
    # cache. Said explicitly rather than producing a shape error deep inside a level.
    ref, sec = shifted_pair(256, (0, 0); T = Float32)
    c = init(ref, sec; chip_size = 32, search_radius = 20)
    other, _ = shifted_pair(384, (0, 0); T = Float32)
    @test_throws DimensionMismatch reinit!(c; reference = other)
    msg = sprint(showerror, try reinit!(c; reference = other) catch e; e end)
    @test occursin("init", msg)          # the message says how to proceed
end

@testset "caller-supplied points" begin
    # The production path: eight per-pixel fields that cannot be scalar keywords arrive as a
    # PointSet.
    ref, sec = shifted_pair(512, (6, -4); T = Float32)

    # Gridded goes through the pyramid, so it carries chip sizes.
    grid = gridpoints((512, 512), 32; chip_size = 32, search_radius = 25)
    r = autorift(ref, sec, grid; chip_size = 32)
    @test r isa MultichipResult
    @test med(filter(!isnan, -r.dx)) ≈ 6 atol = 0.05

    # Scattered runs a single scale, since the chip-size loop's coarse pass and merge need a layout.
    pts = pointset([100.0, 200.5, 300.0], [150.0, 250.0, 350.75];
                   chip_size = 32, search_radius = 25)
    d = autorift(ref, sec, pts; chip_size = 32)
    @test d isa AutoRIFT.DisplacementField
    @test count(!isnan, d.dx) == 3
    @test all(v -> abs(v - 6) < 0.1, -filter(!isnan, d.dx))

    # A per-point prior is respected, which is what lets a small radius cover large motion.
    far_ref, far_sec = shifted_pair(512, (20, -16); T = Float32)
    tight = pointset([256.0], [256.0]; chip_size = 32, search_radius = 8,
                     dx_prior = -20.0, dy_prior = 16.0)
    d = autorift(far_ref, far_sec, tight; chip_size = 32)
    @test !isnan(d.dx[1])
    @test -d.dx[1] ≈ 20 atol = 1.0
end

@testset "UInt8 and Float32 correlation paths" begin
    # `quantize` decides the element type the correlator sees. Both must recover the shift; the
    # UInt8 path is the reference's default and the one production uses.
    ref, sec = shifted_pair(512, (7, 5); T = Float32)
    for q in (:uint8, :none)
        r = autorift(ref, sec; chip_size = 32, search_radius = 25, quantize = q,
                     subpixel = :none)
        mx, my = motion(r)
        @test mx == 7
        @test my == 5
    end
end

@testset "threaded and serial agree" begin
    # Threading is a scheduling choice, not an algorithmic one, all the way up to the API.
    ref, sec = shifted_pair(512, (6, -4); T = Float32)
    kw = (; chip_size = 32, search_radius = 25)
    ser = autorift(ref, sec; kw..., threaded = false)
    par = autorift(ref, sec; kw..., threaded = true)
    @test all(isequal.(ser.dx, par.dx))
    @test all(isequal.(ser.dy, par.dy))
    @test ser.chip_size == par.chip_size
end

@testset "the Params path is statically inferrable" begin
    # The keyword form and the positional-`Params` form must be the same computation. That is
    # what the `autorift(ref, sec, ::Params)` docstring promises, and what makes it safe to
    # document as the entry point a trimmed binary uses.
    ref, sec = shifted_pair(512, (6, -4); T = Float32)
    p = Params(ZNCC(), Highpass(), QuantizeUInt8(), PyramidRefine(), GardnerFilter(),
               AutoRIFT.False(),
               32, 32, 128, 1.0, 32, 25, 25, 6, 4, 8, 0.01, 0.0, 0.0, 3, UInt64(0), false)

    # First: the hand-built `Params` is the one `params()` produces for these keywords. If a
    # default drifts, this fails here rather than as a silent behaviour difference in the binary.
    @test p == params(; chip_size = 32, search_radius = 25)

    kwform = autorift(ref, sec; chip_size = 32, search_radius = 25)
    posform = autorift(ref, sec, p)
    @test all(isequal.(kwform.dx, posform.dx))
    @test all(isequal.(kwform.dy, posform.dy))
    @test all(isequal.(kwform.correlation, posform.correlation))

    # The point of the overload: with method *objects*, nothing between the call and the
    # correlation is a runtime value. `@inferred` is the gate — this is what `--trim` needs, and
    # a regression here would break the binary while leaving every other test passing.
    @test (@inferred autorift(ref, sec, p)) isa MultichipResult

    # And the contrast that explains why the overload has to exist at all. `params()` resolves
    # Symbols through a `Dict{Symbol,Any}`, so its return type is not concrete — hence the
    # keyword path cannot be trimmed however concrete the arguments are.
    @test !isconcretetype(Base.infer_return_type(params, Tuple{}))
    @test isconcretetype(typeof(p))
end
