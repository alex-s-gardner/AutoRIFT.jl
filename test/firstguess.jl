# The sparse first-guess stage, and the filter that makes it usable.
#
# `consistent_matches` is tested in the core suite because it needs no detector; `first_guess`
# itself needs `ImageFeatures` and so lives with the extension tests.

using AutoRIFT: consistent_matches

@testset "consistent_matches keeps agreement, drops isolation" begin
    # A block of points that agree, plus one that does not. The agreeing block survives; the
    # dissenter does not, however confident the match that produced it was.
    pts = [(Float64(i), Float64(j)) for i in 1:6 for j in 1:6]
    dx = fill(3.0, length(pts))
    dy = fill(-2.0, length(pts))
    # One rogue displacement in the middle of an otherwise uniform field.
    rogue = 20
    dx[rogue] = 47.0
    dy[rogue] = 91.0

    keep = consistent_matches(pts, dx, dy)
    @test length(keep) == length(pts) - 1
    @test rogue ∉ keep

    # Every survivor is one of the agreeing points.
    @test all(i -> dx[i] == 3.0 && dy[i] == -2.0, keep)
end

@testset "consistent_matches tolerates a smooth field" begin
    # The property that makes this the right filter for ice: a *rotating* or shearing field has no
    # single displacement, but every point still agrees with its neighbours. A global median would
    # reject most of this; a local consistency check keeps it.
    pts = [(Float64(i), Float64(j)) for i in 1:10 for j in 1:10]
    # Solid-body rotation about (5.5, 5.5): displacement grows with radius but varies smoothly.
    dx = [-(p[1] - 5.5) * 0.15 for p in pts]
    dy = [(p[2] - 5.5) * 0.15 for p in pts]
    keep = consistent_matches(pts, dx, dy; tolerance = 1.0)
    # Nearly everything survives, because neighbours differ by far less than the tolerance even
    # though the field spans a wide range of displacements.
    @test length(keep) > 0.9 * length(pts)
    @test maximum(dx) - minimum(dx) > 1.0     # the field really does vary by more than `tolerance`
end

@testset "both neighbour strategies agree" begin
    # The k-d tree in the NearestNeighbors extension must be a pure cost optimisation: same K nearest
    # neighbours, same survivors. It is 12x faster at 3000 points and 90x at 20000, so the temptation
    # to accept "close enough" is real — this pins exact agreement instead.
    #
    # Runs whichever strategy is active against the brute-force one explicitly, so it is meaningful
    # with or without the extension loaded.
    rng = Random.MersenneTwister(7)
    n = 800
    pts = [(rand(rng) * 512, rand(rng) * 512) for _ in 1:n]
    dx = randn(rng, n) .* 2 .+ 20
    dy = randn(rng, n) .* 2 .- 12
    brute = AutoRIFT._neighbour_indices(pts, 12, AutoRIFT._BruteForceNeighbours())
    active = AutoRIFT._neighbour_indices(pts, 12)
    # Index for index, not merely as sets: both sort by distance.
    @test all(i -> collect(brute[i]) == [j for j in active[i] if j != i][1:length(brute[i])],
              eachindex(brute))
end

@testset "consistent_matches edge cases" begin
    # No points at all: an empty answer, not an error. A caller who found no matches gets to
    # decide what that means.
    @test consistent_matches(Tuple{Float64,Float64}[], Float64[], Float64[]) == Int[]

    # Fewer points than the neighbourhood needs. Nothing can be judged consistent, so nothing is
    # kept — silently returning all of them would be the dangerous answer.
    pts = [(1.0, 1.0), (2.0, 2.0)]
    @test consistent_matches(pts, [1.0, 1.0], [1.0, 1.0]; neighbours = 12, min_agree = 5) == Int[]

    # Mismatched lengths are a caller error.
    @test_throws DimensionMismatch consistent_matches([(1.0, 1.0)], [1.0, 2.0], [1.0])
    # An impossible threshold is caught up front rather than silently keeping nothing.
    @test_throws ArgumentError consistent_matches(pts, [1.0, 1.0], [1.0, 1.0];
                                                 neighbours = 4, min_agree = 9)
end

@testset "a FirstGuess without its package names the RIGHT dependency" begin
    # `ORBGuess` and `AKAZEGuess` are declared in the core so they can be documented and dispatched
    # on, but neither works until its extension loads. The error must name the package that would
    # actually help — the first version hardcoded "ImageFeatures" for every subtype, so `AKAZEGuess`
    # sent the reader to install a package that would not have fixed it.
    #
    # This testset runs in the core suite, *before* `test/extensions.jl` loads any detector, which is
    # what makes it meaningful: it asserts the pre-extension state.
    for (guess, pkg, wrong) in ((ORBGuess(), "ImageFeatures", "AkazeFeatures"),
                                (AKAZEGuess(), "AkazeFeatures", "ImageFeatures"))
        err = try
            AutoRIFT._detector(guess)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin(pkg, err.msg)
        @test !occursin(wrong, err.msg)
    end
    @test AutoRIFT.required_package(ORBGuess()) == "ImageFeatures"
    @test AutoRIFT.required_package(AKAZEGuess()) == "AkazeFeatures"
end


# ---------------------------------------------------------------------------
# Chip rotation search
# ---------------------------------------------------------------------------

using AutoRIFT: RotationSearch, NoRotationSearch, angles, params

@testset "RotationSearch construction and the off case" begin
    @test angles(RotationSearch()) == (-3.0, 0.0, 3.0)   # sea_ice_drift's own default
    @test angles(RotationSearch((-1, 0, 1))) == (-1.0, 0.0, 1.0)

    # Empty is a caller error rather than a silent no-op: `NoRotationSearch()` is how you disable it.
    @test_throws ArgumentError RotationSearch(())
    # A repeated angle costs a full correlation and can never win.
    @test_throws ArgumentError RotationSearch((0.0, 3.0, 0.0))

    # The keyword accepts the spellings the docstring promises.
    @test params().rotation === NoRotationSearch()
    @test params(; rotation = nothing).rotation === NoRotationSearch()
    @test params(; rotation = false).rotation === NoRotationSearch()
    @test angles(params(; rotation = true).rotation) == (-3.0, 0.0, 3.0)
    @test angles(params(; rotation = (-2, 0, 2)).rotation) == (-2.0, 0.0, 2.0)
    # A range works too, which the previous two-method `_rotation` silently rejected.
    @test angles(params(; rotation = -6:3:6).rotation) == (-6.0, -3.0, 0.0, 3.0, 6.0)
    @test_throws ArgumentError params(; rotation = "yes")
end

@testset "rotation search is exactly the unrotated path when off" begin
    # The load-bearing property: enabling the *feature* must cost nothing for callers who leave it
    # off. `NoRotationSearch` dispatches to a method that is the original `correlate!` call, so the
    # result must be bit-identical — not close, identical.
    ref, sec = shifted_pair(256, (4, -3); T = Float32)
    kw = (; chip_size = 32, chip_size_max = 32, search_radius = 10)
    off = autorift(ref, sec; kw...)
    explicit = autorift(ref, sec; kw..., rotation = NoRotationSearch())
    @test all(isequal.(off.dx, explicit.dx))
    @test all(isequal.(off.dy, explicit.dy))
    @test all(isequal.(off.correlation, explicit.correlation))

    # And a single zero angle must agree too, since it is the same arithmetic by a different route.
    # Not bit-identical: the search copies the winning surface, and `maximum` over it is a different
    # reduction order than the direct path's. Displacements are unaffected, which is what matters.
    one = autorift(ref, sec; kw..., rotation = RotationSearch((0.0,)))
    @test all(isequal.(off.dx, one.dx))
    @test all(isequal.(off.dy, one.dy))
end

@testset "rotation search recovers correlation lost to rotation" begin
    # The measurement that justifies the cost. A rotated scene decorrelates against unrotated chips;
    # trying several angles recovers part of the peak. See `RotationSearch`'s docstring for the full
    # table — this pins the direction and rough magnitude rather than an exact value, since the
    # amount depends on chip size and rotation angle.
    n = 256
    a = synthetic_texture(n)
    # Rotate about the centre by 3 degrees, by nearest-neighbour resampling — enough to decorrelate
    # a 32-px chip without needing an image-transform dependency in the test suite.
    yc = xc = (n + 1) / 2
    s3, c3 = sincos(deg2rad(3.0))
    b = zeros(Float32, n, n)
    for j in 1:n, i in 1:n
        dy, dx = i - yc, j - xc
        si = round(Int, yc + c3 * dy + s3 * dx)
        sj = round(Int, xc - s3 * dy + c3 * dx)
        (1 <= si <= n && 1 <= sj <= n) && (b[i, j] = a[si, sj])
    end

    kw = (; chip_size = 32, chip_size_max = 32, search_radius = 8, quantize = :none,
          outliers = :none)
    plain = autorift(a, b; kw...)
    rotated = autorift(a, b; kw..., rotation = RotationSearch((-3.0, 0.0, 3.0)))

    cp = med(filter(!isnan, plain.correlation))
    cr = med(filter(!isnan, rotated.correlation))
    # Strictly better, and by a margin well clear of noise — measured 20-150% on speckle depending
    # on chip size.
    @test cr > cp
    @test cr > 1.05 * cp
end
