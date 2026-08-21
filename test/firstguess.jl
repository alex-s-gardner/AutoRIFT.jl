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

@testset "a FirstGuess without ImageFeatures names the dependency" begin
    # `ORBGuess` is declared in the core so it can be documented and dispatched on, but it cannot
    # work until the extension loads. The error must say so — a bare `MethodError` on an internal
    # function would send the reader into the wrong file.
    #
    # This testset runs in the core suite, *before* `test/extensions.jl` loads ImageFeatures, which
    # is what makes it meaningful: it asserts the pre-extension state.
    err = try
        AutoRIFT._detector(ORBGuess())
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("ImageFeatures", err.msg)
end
