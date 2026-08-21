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

@testset "RotationSearch `about` shifts the window it searches" begin
    # `angles(m)` returns the rotations applied to the *chip*, which is `m.angles .- m.about`: the
    # chip comes from the secondary and must be turned back to the reference's orientation. Same
    # `angle - alpha0` as `sea_ice_drift`'s `rotate_and_match`. Resolving it in `angles` is what
    # stops the correlator and the tests from disagreeing about the sign.
    @test angles(RotationSearch(; about = 8.0)) == (-11.0, -8.0, -5.0)
    @test angles(RotationSearch((-1, 0, 1); about = -4.5)) == (3.5, 4.5, 5.5)
    # Both spellings of the default agree, because subtracting zero is the identity — *not* because
    # a zero case is branched around. There is no `about == 0` fast path: one was written and
    # measured slower than the unconditional subtraction, which `angles`'s docstring records.
    @test angles(RotationSearch()) === angles(RotationSearch(; about = 0.0))

    # Every angle *tried* must be finite, and the angles tried are `angles .- about` — so both
    # halves are checked. Enforcing it only on `about` was the first version, and it let a NaN
    # angle through to produce an all-NaN chip that vanished into the degenerate path, reported as
    # a textureless chip rather than as bad input.
    @test_throws ArgumentError RotationSearch(; about = NaN)
    @test_throws ArgumentError RotationSearch((-3, 0, 3); about = Inf)
    @test_throws ArgumentError RotationSearch((NaN, 0.0, 3.0))
    @test_throws ArgumentError RotationSearch((-Inf, 0.0, 3.0))

    # An unfittable field cannot be passed through by accident: `scene_rotation` returns `nothing`,
    # so this fails at the line that made the mistake — `about::Real` rejects it — rather than
    # becoming a NaN that survives arithmetic and surfaces somewhere else.
    @test_throws TypeError RotationSearch(; about = scene_rotation([1.0], [2.0], [0.0], [0.0]))
end

# Displacements for a field where the ice has rotated by `motion` degrees, plus a rigid translation
# `shift`. Returns `(dx, dy)` in the package's reference-minus-secondary sense.
#
# One helper rather than the same eight lines in three testsets, because the *sign* is the thing
# under test and three hand-written copies is three chances to write it differently.
#
# The rotation centre is fixed rather than a keyword. It was one, and no assertion could observe it:
# `scene_rotation` removes the centroid, so the fit is centre-invariant by construction — measured
# identical to 1e-14 for pivots at (50, 50), (250, 250) and (-9999, 7). A knob no test can distinguish
# reads as load-bearing at the one call site that sets it. The invariance is pinned directly below
# instead, which is the honest way to state it.
function rotated_field(xs, ys, motion; shift = (0.0, 0.0))
    s, c = sincos(deg2rad(motion))
    xc = yc = 250.0
    # Where each feature ends up in the secondary: carried by the motion, then translated.
    sx = @. xc + c * (xs - xc) - s * (ys - yc) + shift[1]
    sy = @. yc + s * (xs - xc) + c * (ys - yc) + shift[2]
    return xs .- sx, ys .- sy
end

# A rotation about `centre` rather than `rotated_field`'s fixed pivot, for the invariance test only.
function rotated_field_about(xs, ys, motion, centre)
    s, c = sincos(deg2rad(motion))
    xc, yc = centre
    sx = @. xc + c * (xs - xc) - s * (ys - yc)
    sy = @. yc + s * (xs - xc) + c * (ys - yc)
    return xs .- sx, ys .- sy
end

@testset "scene_rotation does not depend on where the rotation was centred" begin
    # The property that lets `rotated_field` hardcode its pivot, and the reason a single scene angle
    # is meaningful at all: the centroid step removes translation, and a rotation about a different
    # centre differs from one about this centre by exactly a translation.
    xs = [0.0, 100.0, 100.0, 0.0]
    ys = [0.0, 0.0, 100.0, 100.0]
    fits = [AutoRIFT.scene_rotation(xs, ys, rotated_field_about(xs, ys, 6.0, ctr)...)
            for ctr in ((50.0, 50.0), (250.0, 250.0), (-9999.0, 7.0))]
    @test all(f -> isapprox(f, -6.0; atol = 1e-9), fits)
end

@testset "scene_rotation fits a known rotation" begin
    # `scene_rotation` reports the rotation carrying **secondary onto reference**, which is the
    # negative of the ice's own motion — the same relationship `dx`/`dy` have to velocity. Pinning
    # that here rather than in a comment: getting it backwards would centre the angle search on
    # exactly the wrong side and be twice as wrong as not centring at all.
    rng = Random.MersenneTwister(11)
    n = 400
    xs = rand(rng, n) .* 500
    ys = rand(rng, n) .* 500
    for motion in (-12.0, -3.0, 0.0, 3.0, 8.5, 25.0), shift in ((0.0, 0.0), (37.0, -64.0))
        dx, dy = rotated_field(xs, ys, motion; shift)
        # A translation must be removed by the centroid step, not absorbed into the angle.
        @test AutoRIFT.scene_rotation(xs, ys, dx, dy) ≈ -motion atol = 1e-9
    end
end

@testset "scene_rotation degenerate cases" begin
    # `nothing`, not `NaN`: a scalar NaN survives arithmetic and surfaces somewhere unrelated, where
    # `nothing` fails at first use. See the docstring.
    #
    # Nothing to fit: a rotation needs two points, since one gives only a translation.
    @test AutoRIFT.scene_rotation(Float64[], Float64[], Float64[], Float64[]) === nothing
    @test AutoRIFT.scene_rotation([1.0], [2.0], [0.5], [0.5]) === nothing
    # Coincident points: both Procrustes sums vanish, and `atan(0, 0)` would report a confident 0.
    @test AutoRIFT.scene_rotation(fill(3.0, 5), fill(4.0, 5), zeros(5), zeros(5)) === nothing
    # A pure translation *is* fittable and is exactly zero rotation, not a small nonzero fit — so it
    # belongs here as the boundary against the coincident case above, which is not fittable at all.
    @test AutoRIFT.scene_rotation([0.0, 10.0, 5.0], [0.0, 3.0, 9.0],
                                  fill(7.0, 3), fill(-2.0, 3)) == 0.0
    @test_throws DimensionMismatch AutoRIFT.scene_rotation([1.0, 2.0], [1.0], [0.0], [0.0])
end

@testset "scene_rotation reads a PointSet directly" begin
    # The form a caller actually uses: `first_guess` returns a `PointSet` whose priors are the
    # sparse displacements, so no unpacking should be needed at the call site.
    xs = [0.0, 100.0, 100.0, 0.0]
    ys = [0.0, 0.0, 100.0, 100.0]
    dx, dy = rotated_field(xs, ys, 6.0)
    pts = AutoRIFT.pointset(xs, ys; dx_prior = dx, dy_prior = dy)
    @test AutoRIFT.scene_rotation(pts) ≈ -6.0 atol = 1e-9

    # And the composition that is the whole point of the feature: the fitted rotation, fed to
    # `about`, produces a chip rotation that undoes the scene's — so `about` and the motion agree
    # in magnitude and the searched angle brackets it.
    m = RotationSearch((0.0,); about = AutoRIFT.scene_rotation(pts))
    @test only(angles(m)) ≈ 6.0 atol = 1e-9
end

@testset "counter-rotating the chip is what recovers correlation" begin
    # The measurement that fixes the sign, and it is not a detail: rotating the chip the *same* way
    # as the scene makes things worse, not better. A chip cut from the rotated secondary is
    # correlated against an unrotated reference window, so it must be turned back.
    #
    # Isolated from displacement deliberately — one chip at the centre of rotation, where the
    # displacement is zero, so the only thing varying is the chip's rotation. Measured peaks, from
    # which `RotationSearch`'s docstring table is taken:
    #
    #     chip     no rotation    +8 deg     -8 deg
    #       32          0.723      0.345      0.597
    #       64          0.328      0.132      0.641
    #      128          0.142      0.025      0.670
    n = 512
    a = synthetic_texture(n)
    motion = 8.0
    b = rotate_bilinear(a, motion)
    ci = cj = 257
    R = 4

    # Peak correlation of the centre chip rotated by `ang`, at chip size `cs`. Chip from the rotated
    # secondary, window from the unrotated reference — which is the arrangement under test.
    function rotated_peak(cs, ang)
        chip, win = chip_and_window(b, (ci, cj), cs, R; window_from = a)
        ws = AutoRIFT.workspace(Float32, (cs, cs), (R, R))
        return maximum(AutoRIFT.correlate!(
            ws, win, AutoRIFT._rotate_chip(ws, chip, ang), (R, R); measure = ZNCC()))
    end

    # Whether counter-rotation beats not rotating is *chip-size dependent*, so the expectation is
    # data rather than a special case below the loop. At chip 32 it loses: those corners travel only
    # ~2 px at 8 degrees, and resampling plus corner padding cost more than that. Pinned in both
    # directions because it is the "gain grows with chip size" effect seen from each end — a change
    # that flipped the 32 case would mean the rotation had stopped being nearly free at small chips.
    for (cs, beats_unrotated) in ((32, false), (64, true), (128, true))
        back, unrotated, wrong = rotated_peak(cs, -motion), rotated_peak(cs, 0.0),
                                 rotated_peak(cs, motion)
        @test (back > unrotated) == beats_unrotated
        # Counter-rotation always beats rotating the wrong way, at every size.
        @test back > wrong
    end

    # `about` must produce exactly that winning angle from the scene rotation — `scene_rotation`
    # returns `-motion`, and the chip needs `+motion`, with no negation at the call site.
    @test only(angles(RotationSearch((0.0,); about = -motion))) == motion
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
    b = rotate_bilinear(a, 3.0)

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
