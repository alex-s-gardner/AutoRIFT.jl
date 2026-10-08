# The geogrid handoff: a `PairGeometry` becomes a `PointSet`.
#
# Every assertion here corresponds to a way the conversion could silently misbehave rather than fail:
# an index off by one, a half pixel applied twice, a nodata fill value arriving as a negative search
# radius, a per-point chip bound dropped.

using AutoRIFT
using Aqua
using ImagePairGeometry
using ImagePairGeometry: nodata_from, chip_size_pixels
using StaticArrays: SVector
using Test

# Without the extension, `pointset(::PairGeometry)` does not exist and every assertion below would
# fail on a MethodError rather than on the value it is checking — but a future refactor could instead
# make them all pass against some other method. Asserted, not assumed.
@test Base.get_extension(AutoRIFT, :AutoRIFTImagePairGeometryExt) !== nothing

"""A geometry over a grid the pair covers, with every input band present."""
function ipg_case(; csminy = 360.0, ssm = 1.0)
    grid = MapGrid(geotransform = (295000.0, 120.0, 0.0, 7805000.0, 0.0, -120.0),
                   size = (200, 200), crs = 32624)
    # **Deliberately non-square.** A square footprint gives a square grid window, and a square window
    # hides a transposed band: the shapes agree, the correlator runs, and the answer is wrong by two
    # thirds of its agreement. Every assertion below that compares against a band therefore has a shape
    # to disagree about.
    fp = ImageFootprint(origin = (300000.0, 7800000.0), spacing = (30.0, -30.0),
                        size = (400, 560))
    pair = coregister(fp, fp; dt = 91 * 86400.0)
    win = grid_window(grid, footprint_bounds(IdentityTransform(), pair.coordinate))
    n = size(win)
    inputs = GeometryInputs(dem = fill(500.0, n), dhdx = fill(0.02, n), dhdy = fill(-0.01, n),
                            vx = fill(120.0, n), vy = fill(-80.0, n),
                            srx = fill(400.0, n), sry = fill(300.0, n),
                            csminx = fill(240.0, n), csminy = fill(csminy, n),
                            csmaxx = fill(480.0, n), csmaxy = fill(720.0, n),
                            ssm = fill(ssm, n))
    r = pairgeometry(grid, pair, inputs; window = win, nodata = nodata_from(-32767.0))
    return r, grid, pair, win
end

# One geometry at the default inputs, shared by the testsets that do not vary them: building it runs
# the whole geometry kernel over the window.
const IPG_R, IPG_GRID, IPG_PAIR, IPG_WIN = ipg_case()
const SENTINEL = Int32(-32767)

@testset "pixel positions gain 1.5, and the layout transposes" begin
    pts = AutoRIFT.pointset(IPG_R; pixel_size = 30.0)
    @test pts isa AutoRIFT.PointSet{2}

    # **A `PairGeometry` is indexed `[x, y]` and a `PointSet` `[row, col]`, so the shape reverses.**
    # Asserted on a deliberately non-square window: on a square one the shapes agree whichever layout
    # the bands are in, the correlator runs, and it returns a plausible field having lost two thirds of
    # its agreement with the reference. That is exactly how this went unnoticed.
    @test size(IPG_WIN, 1) != size(IPG_WIN, 2)
    @test size(pts) == reverse(size(IPG_R))

    loc_x = permutedims(IPG_R.location_x)
    loc_y = permutedims(IPG_R.location_y)
    valid = findall(!=(SENTINEL), loc_x)
    # `+1.5`: one for the index base, since geogrid's index is zero-based, and a half for the offset
    # `runAutorift` bakes into its grid before correlating. Asserted over the whole grid at once, since
    # the property is uniform and one assertion per point would report the same fact thousands of times.
    @test all(k -> pts.x[k] == loc_x[k] + 1.5, valid)
    @test all(k -> pts.y[k] == loc_y[k] + 1.5, valid)
    # Half-integers, not integers: a whole-pixel grid is the signature of the offset being dropped.
    @test all(k -> pts.x[k] % 1 == 0.5, valid)
    @test all(k -> pts.y[k] % 1 == 0.5, valid)
end

@testset "invalid points are skipped, not searched" begin
    pts = AutoRIFT.pointset(IPG_R; pixel_size = 30.0)

    invalid = findall(==(SENTINEL), permutedims(IPG_R.location_x))
    @test !isempty(invalid)     # the window overhangs the image, so some points are outside
    # Zero radius is how a point is marked to skip. Passing the fill value through would make the
    # radius negative, which `gridpoints`' margin logic would then size itself from.
    @test all(iszero, pts.radius_x[invalid])
    @test all(iszero, pts.radius_y[invalid])
    @test !any(k -> AutoRIFT.issearchable(pts, k), invalid)

    valid = findall(!=(SENTINEL), permutedims(IPG_R.location_x))
    @test AutoRIFT.nsearchable(pts) == length(valid)
    @test all(k -> AutoRIFT.issearchable(pts, k), valid)
    @test all(>(0), pts.radius_x[valid])
end

@testset "search radius and prior carry through" begin
    pts = AutoRIFT.pointset(IPG_R; pixel_size = 30.0)
    # Every band is transposed together, so a comparison against the geometry transposes too.
    valid = findall(!=(SENTINEL), permutedims(IPG_R.location_x))
    @test all(k -> pts.radius_x[k] == permutedims(IPG_R.search_x)[k], valid)
    @test all(k -> pts.radius_y[k] == permutedims(IPG_R.search_y)[k], valid)
    @test all(k -> pts.dx_prior[k] == permutedims(IPG_R.offset_x)[k], valid)
    # `dy_prior` is negated on the way in, because it is held in the axis `dy` is measured along and
    # the geometry's `offset_y` is in the axis the reference hands `Dy0` over in. See the next testset.
    @test all(k -> pts.dy_prior[k] == -permutedims(IPG_R.offset_y)[k], valid)
end

@testset "the y prior's sign comes from the coordinate system" begin
    # Two factors compose here, and only their product is observable. `y_displacement_sign` answers the
    # driver-level question — a projected image's +y already points down its own second axis, a radar
    # image's azimuth opposes it — while `arImgDisp_*` negates `Dy0` again inside the call and AutoRIFT.jl
    # has no such step. So the stored prior carries the opposite sign to the coordinate's, for both
    # systems. A wrong sign here sends the correlator the wrong way and returns a plausible velocity,
    # which is why the product is pinned rather than left to the reader.
    pts = AutoRIFT.pointset(IPG_R; pixel_size = 30.0)

    # Only the projected branch has a fixture here, so this pins the sign the negation is applied to and
    # the testset above pins the negation itself. The radar branch — where `y_displacement_sign` is
    # `-1.0` and the stored prior is therefore `+offset_y` — is covered end to end by the golden ladder's
    # Sentinel-1 cases rather than here, since a `RadarCoordinate` fixture needs an orbit.
    @test y_displacement_sign(IPG_PAIR.coordinate) === 1.0
    # Passing it explicitly agrees with the default.
    @test AutoRIFT.pointset(IPG_R; pixel_size = 30.0,
                            coordinate = IPG_PAIR.coordinate).dy_prior == pts.dy_prior

    # Something that is not a coordinate is refused rather than silently signed.
    @test_throws "expected a ProjectedCoordinate" AutoRIFT.pointset(IPG_R; pixel_size = 30.0,
                                                                    coordinate = 42)
end

@testset "chip size" begin
    # From a pixel size, derived as the reference does.
    pts = AutoRIFT.pointset(IPG_R; pixel_size = 30.0)
    @test all(==(chip_size_pixels(240.0, 30.0)), pts.chip_size_x)
    @test all(==(8), pts.chip_size_x)

    # Or given directly.
    pts32 = AutoRIFT.pointset(IPG_R; chip_size = 32)
    @test all(==(32), pts32.chip_size_x)
    @test all(==(32), pts32.chip_size_y)

    # Bounds are per point, and a bound of zero means unbounded.
    valid = findall(!=(SENTINEL), permutedims(IPG_R.location_x))
    invalid = findall(==(SENTINEL), permutedims(IPG_R.location_x))
    @test all(k -> pts.chip_size_min_x[k] == permutedims(IPG_R.chip_min_x)[k], valid)
    @test all(k -> pts.chip_size_max_x[k] == permutedims(IPG_R.chip_max_x)[k], valid)
    @test all(iszero, pts.chip_size_min_x[invalid])
    @test all(iszero, pts.chip_size_max_x[invalid])

    # Neither given is an error rather than a silent default: the base extent is not recoverable
    # from a PairGeometry, which stores the bounds but not the base.
    @test_throws "needs the base chip extent" AutoRIFT.pointset(IPG_R)
end

@testset "a geometry with no search band is refused" begin
    # Without `srx`/`sry` every radius would be zero and nothing would be correlated. That is a
    # missing input, not a grid of skips, so it fails with a message saying which.
    n = size(IPG_WIN)
    bare = pairgeometry(IPG_GRID, IPG_PAIR, GeometryInputs(dem = fill(500.0, n));
                        window = IPG_WIN, nodata = nodata_from(-32767.0))
    @test_throws "no search-range band" AutoRIFT.pointset(bare; pixel_size = 30.0)
end

@testset "blocked geometry converts identically" begin
    # The conversion reads only the result, so a blocked run must give the same PointSet.
    n = size(IPG_WIN)
    inputs = GeometryInputs(dem = fill(500.0, n), dhdx = fill(0.02, n), dhdy = fill(-0.01, n),
                            vx = fill(120.0, n), vy = fill(-80.0, n),
                            srx = fill(400.0, n), sry = fill(300.0, n),
                            csminx = fill(240.0, n), csminy = fill(360.0, n),
                            csmaxx = fill(480.0, n), csmaxy = fill(720.0, n), ssm = fill(1.0, n))
    blocked = pairgeometry_blocked(IPG_GRID, IPG_PAIR, InMemoryInputs(inputs, IPG_WIN);
                                  transform = IdentityTransform(), window = IPG_WIN,
                                  blocksize = (16, 16), nodata = nodata_from(-32767.0))
    a = AutoRIFT.pointset(IPG_R; pixel_size = 30.0)
    b = AutoRIFT.pointset(blocked; pixel_size = 30.0)
    for f in (:x, :y, :radius_x, :radius_y, :dx_prior, :dy_prior, :chip_size_x, :chip_size_y,
              :chip_size_min_x, :chip_size_max_x)
        @test getfield(a, f) == getfield(b, f)
    end
end

@testset "extension code quality" begin
    # The same pair of checks `extensions.jl` runs over the other extensions. Here rather than in that
    # loop because this extension loads only when ImagePairGeometry is resolvable, which `extensions.jl`
    # cannot assume.
    mod = Base.get_extension(AutoRIFT, :AutoRIFTImagePairGeometryExt)
    Aqua.test_undefined_exports(mod)
    Aqua.test_stale_deps(mod)
end
