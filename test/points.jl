using AutoRIFT: PointSet, pointset, gridpoints, scatter, npoints, nsearchable,
                issearchable, sanitize!, chip_bounds, search_bounds,
                surface_size, inbounds

@testset "scattered points" begin
    # The correlator works on a list of search centers. Arbitrary, unsorted,
    # fractional coordinates are the general case, not a special one.
    xs = [10.0, 200.5, 33.25, 7.0]
    ys = [20.0, 60.0, 300.75, 9.5]
    pts = pointset(xs, ys; search_radius = 25, chip_size = 32)

    @test pts isa PointSet{1}
    @test ndims(pts) == 1
    @test npoints(pts) == 4
    @test length(pts) == 4
    @test pts.x == xs
    @test pts.y == ys
    # Fractional coordinates survive: a center at x = 200.5 sits between columns.
    @test pts.x[2] == 200.5

    # Scalars broadcast to every point.
    @test all(pts.radius_x .== 25)
    @test all(pts.chip_size_x .== 32)
    @test all(pts.dx_prior .== 0)

    # Random centers, which is the motivating case.
    rng = Random.MersenneTwister(42)
    n = 500
    rpts = pointset(rand(rng, n) .* 1000, rand(rng, n) .* 1000; search_radius = 16)
    @test npoints(rpts) == n
    @test nsearchable(rpts) == n
end

@testset "point construction from tuples and indices" begin
    pts = pointset([(10.0, 20.0), (30.0, 40.0)])
    @test pts.x == [10.0, 30.0]
    @test pts.y == [20.0, 40.0]

    # A CartesianIndex is (row, col) = (y, x). Getting this backwards would
    # transpose the entire point set, so it is asserted explicitly.
    pts = pointset([CartesianIndex(5, 9), CartesianIndex(1, 2)])
    @test pts.y == [5.0, 1.0]
    @test pts.x == [9.0, 2.0]
end

@testset "per-point geometry" begin
    # Every geometry field can vary per point, not just globally: this is what
    # lets Geogrid hand over a spatially varying search radius and chip size.
    pts = pointset([10.0, 20.0, 30.0], [10.0, 20.0, 30.0];
                   search_radius_x = [5, 25, 50],
                   search_radius_y = [5, 10, 10],
                   chip_size_x = [16, 32, 64],
                   dx_prior = [0.0, 2.5, -3.0])
    @test pts.radius_x == [5, 25, 50]
    @test pts.radius_y == [5, 10, 10]
    @test pts.chip_size_x == [16, 32, 64]
    @test pts.dx_prior == [0.0, 2.5, -3.0]

    # chip_size_y derives from chip_size_x via the aspect, forced even.
    pts = pointset([1.0], [1.0]; chip_size = 32, chip_aspect = 0.5)
    @test pts.chip_size_y == [16]
    pts = pointset([1.0], [1.0]; chip_size = 32, chip_aspect = 0.7)
    @test iseven(pts.chip_size_y[1])

    # An integer field must not silently swallow a fractional value: a chip size
    # of 32.5 is a caller mistake, not a rounding request.
    @test_throws "expected an integer value" pointset([1.0], [1.0]; chip_size = 32.5)
    @test_throws "expected an integer value" pointset([1.0], [1.0]; search_radius = 6.7)

    @test_throws DimensionMismatch pointset([1.0, 2.0], [1.0])
    @test_throws DimensionMismatch pointset([1.0, 2.0], [1.0, 2.0];
                                            search_radius_x = [1, 2, 3])
    @test_throws "must be a number or an array" pointset([1.0], [1.0];
                                                         chip_size = :big)
end

@testset "gridded points" begin
    pts = gridpoints((512, 512), 32; chip_size = 32, search_radius = 25)
    @test pts isa PointSet{2}
    @test ndims(pts) == 2
    @test size(pts, 1) == size(pts, 2)
    @test npoints(pts) == prod(size(pts))

    # Row index varies with y, column with x, matching image layout. A transpose
    # here would silently swap the velocity components downstream.
    @test pts.x[1, 1] == pts.x[2, 1]
    @test pts.y[1, 1] == pts.y[1, 2]
    @test pts.x[1, 2] > pts.x[1, 1]
    @test pts.y[2, 1] > pts.y[1, 1]

    # Every point's search window must fit in the image: gridpoints insets by
    # the chip half-extent plus the search radius for exactly this reason.
    @test all(i -> inbounds(pts, i, (512, 512)), eachindex(pts))

    # Explicit coordinate vectors.
    pts = gridpoints(100.0:10.0:200.0, 50.0:10.0:100.0; chip_size = 16)
    @test size(pts) == (6, 11)

    # An image too small for even one center is an error naming the margin
    # needed, not an empty result.
    @test_throws "no search center fits" gridpoints((32, 32), 8;
                                                     chip_size = 64, search_radius = 25)
    @test_throws "must be positive" gridpoints((512, 512), 0)
end

@testset "scatter shares memory" begin
    # A gridded set is handed to the correlator as a flat list. `vec` of an array
    # shares storage, so this must allocate nothing and stay a live view.
    grid = gridpoints((256, 256), 32; chip_size = 32, search_radius = 25)
    flat = scatter(grid)
    @test flat isa PointSet{1}
    @test npoints(flat) == npoints(grid)
    @test flat.x == vec(grid.x)

    flat.radius_x[1] = 0
    @test grid.radius_x[1] == 0   # same memory, not a copy

    @test scatter(flat) === flat
end

@testset "sanitize!" begin
    # Rule 1: zero is contagious across axes. A window with no extent in one
    # direction cannot yield a displacement, so both radii are cleared.
    pts = pointset([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0];
                   search_radius_x = [10, 10, 0, -3],
                   search_radius_y = [10, 0, 10, 8])
    n = sanitize!(pts, 6)
    @test pts.radius_x == [10, 0, 0, 0]
    @test pts.radius_y == [10, 0, 0, 0]
    @test n == 1
    @test nsearchable(pts) == 1

    # Rule 2: the floor applies per axis, and only to searchable points. A point
    # cleared by rule 1 stays cleared rather than being raised to the floor.
    pts = pointset([1.0, 2.0, 3.0], [1.0, 2.0, 3.0];
                   search_radius_x = [3, 20, 0],
                   search_radius_y = [20, 2, 5])
    n = sanitize!(pts, 6)
    @test pts.radius_x == [6, 20, 0]
    @test pts.radius_y == [20, 6, 0]
    @test n == 2

    @test issearchable(pts, 1)
    @test !issearchable(pts, 3)
end

@testset "chip and search geometry" begin
    pts = pointset([100.0], [200.0]; chip_size = 32, search_radius = 25)

    rows, cols = chip_bounds(pts, 1)
    @test length(rows) == 32
    @test length(cols) == 32

    srows, scols = search_bounds(pts, 1)
    # The search window exceeds the chip by the radius in each direction, and
    # spans -radius to +radius-1 so that the surface is even-sized with an
    # unambiguous centre.
    @test length(scols) == 32 + 2 * 25 - 1
    @test length(srows) == 32 + 2 * 25 - 1

    # The correlation surface size follows from those two extents, and the
    # peak-offset arithmetic assumes exactly this. Asserting it here means a
    # change to either bound is caught rather than silently biasing every
    # displacement.
    @test surface_size(pts, 1) == (50, 50)
    @test surface_size(pts, 1) == (length(srows) - length(rows) + 1,
                                   length(scols) - length(cols) + 1)

    # Non-square geometry: each axis is independent throughout.
    pts = pointset([100.0], [200.0]; chip_size_x = 64, chip_size_y = 32,
                   search_radius_x = 25, search_radius_y = 10)
    rows, cols = chip_bounds(pts, 1)
    @test length(cols) == 64
    @test length(rows) == 32
    @test surface_size(pts, 1) == (20, 50)

    # The a-priori displacement shifts the chip, not the search window: the
    # search begins where motion is expected while remaining centred on the
    # point, so the recovered displacement stays relative to the point.
    a = pointset([100.0], [100.0]; chip_size = 32, search_radius = 25)
    b = pointset([100.0], [100.0]; chip_size = 32, search_radius = 25,
                 dx_prior = 10.0)
    @test first(chip_bounds(b, 1)[2]) == first(chip_bounds(a, 1)[2]) - 10
    @test search_bounds(a, 1) == search_bounds(b, 1)
end

@testset "inbounds" begin
    # A scattered point set is caller-supplied, so points whose window falls off
    # the image must be detectable up front rather than faulting mid-correlation.
    pts = pointset([100.0, 5.0, 1000.0], [100.0, 100.0, 100.0];
                   chip_size = 32, search_radius = 25)
    @test inbounds(pts, 1, (512, 512))
    @test !inbounds(pts, 2, (512, 512))   # too close to the left edge
    @test !inbounds(pts, 3, (512, 512))   # beyond the right edge
end

@testset "show" begin
    pts = gridpoints((256, 256), 32; chip_size = 32, search_radius = 25)
    s = sprint(show, pts)
    @test occursin("PointSet{2}", s)
    @test occursin("grid", s)

    sanitize!(pointset([1.0], [1.0]; search_radius_x = 0), 6)
    p2 = pointset([1.0, 2.0], [1.0, 2.0]; search_radius_x = [0, 25])
    sanitize!(p2, 6)
    @test occursin("searchable", sprint(show, p2))
end

@testset "PointSet field consistency" begin
    # Mismatched field shapes would produce out-of-bounds reads in the inner
    # loop, so the constructor rejects them.
    @test_throws DimensionMismatch PointSet(
        zeros(3), zeros(2), zeros(Int, 3), zeros(Int, 3),
        zeros(3), zeros(3), zeros(Int, 3), zeros(Int, 3))
    @test_throws DimensionMismatch PointSet(
        zeros(3), zeros(3), zeros(Int, 3), zeros(Int, 2),
        zeros(3), zeros(3), zeros(Int, 3), zeros(Int, 3))
end
