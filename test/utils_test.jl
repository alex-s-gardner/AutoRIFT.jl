# Tests for the test helpers themselves. The synthetic-pair generator is the
# package's primary correctness oracle, so a bug in it would silently weaken
# every truth-based test downstream.

@testset "synthetic_texture" begin
    a = synthetic_texture(64)
    @test size(a) == (64, 64)
    @test eltype(a) === Float32
    @test all(0 .<= a .<= 1)
    # Deterministic in the seed, and different seeds give different textures.
    @test synthetic_texture(32; seed = 1) == synthetic_texture(32; seed = 1)
    @test synthetic_texture(32; seed = 1) != synthetic_texture(32; seed = 2)

    # Band-limited, not white: adjacent pixels must be correlated, or the
    # correlation surface would have a one-pixel-wide peak and subpixel tests
    # would be meaningless.
    v = vec(synthetic_texture(128))
    h = vec(synthetic_texture(128)[:, 2:end])
    lag1 = Statistics.cor(vec(a[:, 1:(end - 1)]), vec(a[:, 2:end]))
    @test lag1 > 0.5

    b = synthetic_texture(32; T = UInt8)
    @test eltype(b) === UInt8
    @test maximum(b) > 200
end

@testset "shifted_pair" begin
    # Integer shift: the two images must match exactly where they overlap, since
    # no interpolation is involved.
    ref, sec = shifted_pair(64, (3, 0))
    @test size(ref) == size(sec) == (64, 64)
    @test ref[:, 1:(end - 3)] ≈ sec[:, 4:end]

    ref, sec = shifted_pair(64, (0, 2))
    @test ref[1:(end - 2), :] ≈ sec[3:end, :]

    ref, sec = shifted_pair(64, (-4, 3))
    @test ref[1:(end - 3), 5:end] ≈ sec[4:end, 1:(end - 4)]

    # Fractional shift: interpolated, so only approximately recoverable, but the
    # displacement must still be detectable in the right direction.
    ref, sec = shifted_pair(64, (2.5, 0))
    @test !isapprox(ref, sec)
    @test Statistics.cor(vec(ref[:, 3:(end - 3)]), vec(sec[:, 3:(end - 3)])) < 0.999

    # Zero shift is the identity.
    ref, sec = shifted_pair(32, (0, 0))
    @test ref ≈ sec
end

@testset "assertion macros" begin
    a = Float32[1.0 2.0; 3.0 NaN]
    @test_identical a copy(a)
    @test_identical a Float32[1.0 2.0; 3.0 NaN]   # NaN == NaN, unlike `==`

    @test_approx a copy(a)
    @test_approx Float32[1.0, 2.0] Float32[1.0 + 1f-8, 2.0]

    # The divergence budget: differences are counted, not tolerated silently.
    x = collect(1:100)
    y = collect(1:100)
    y[1] = 999
    @test_divergence x y 0.02
    @test_throws ErrorException (@test_divergence x y 0.001)
end

@testset "fixture loader" begin
    @test_throws ArgumentError fixture("no_such_fixture")
    # The message must say how to regenerate, not merely that the file is absent.
    msg = sprint(showerror, try fixture("nope") catch e; e end)
    @test occursin("gen_fixtures", msg)
end
