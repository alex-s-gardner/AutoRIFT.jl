# The geogrid handoff: a `PairGeometry` becomes a `PointSet`.
#
# Every assertion here corresponds to a way the conversion could silently misbehave rather than fail:
# an index off by one, a half pixel applied twice, a nodata sentinel arriving as a negative search
# radius, a per-point chip bound dropped.

using AutoRIFT
using Aqua
using ImagePairGeometry
using ImagePairGeometry: nodata_from, chip_size_pixels
using Random
using Test

# Without the extension, `pointset(::PairGeometry)` does not exist and every assertion below would
# fail on a MethodError rather than on the value it is checking — but a future refactor could instead
# make them all pass against some other method. Asserted, not assumed.
@test Base.get_extension(AutoRIFT, :AutoRIFTImagePairGeometryExt) !== nothing

"""A geometry over a grid the pair covers, with every input band present."""
function ipg_case(; csminy = 360.0, ssm = 1.0)
    grid = MapGrid(geotransform = (295000.0, 120.0, 0.0, 7805000.0, 0.0, -120.0),
                   size = (200, 200), crs = 32624)
    fp = ImageFootprint(origin = (300000.0, 7800000.0), spacing = (30.0, -30.0),
                        size = (400, 400))
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

@testset "pixel positions are one-based" begin
    pts = AutoRIFT.pointset(IPG_R; pixel_size = 30.0)
    @test pts isa AutoRIFT.PointSet{2}
    @test size(pts) == size(IPG_R)

    valid = findall(!=(SENTINEL), IPG_R.location_x)
    # Geogrid's index is zero-based; a `PointSet`'s is one-based. Exactly one, and no half pixel —
    # that is added at correlation time for every pyramid level. Asserted over the whole grid at
    # once: the property is uniform, so one assertion per point would report the same fact
    # thousands of times.
    @test all(k -> pts.x[k] == IPG_R.location_x[k] + 1, valid)
    @test all(k -> pts.y[k] == IPG_R.location_y[k] + 1, valid)
    @test all(k -> isinteger(pts.x[k]), valid)
end

@testset "invalid points are skipped, not searched" begin
    pts = AutoRIFT.pointset(IPG_R; pixel_size = 30.0)

    invalid = findall(==(SENTINEL), IPG_R.location_x)
    @test !isempty(invalid)     # the window overhangs the image, so some points are outside
    # Zero radius is how a point is marked to skip. Passing the sentinel through would make the
    # radius negative, which `gridpoints`' margin logic would then size itself from.
    @test all(iszero, pts.radius_x[invalid])
    @test all(iszero, pts.radius_y[invalid])
    @test !any(k -> AutoRIFT.issearchable(pts, k), invalid)

    valid = findall(!=(SENTINEL), IPG_R.location_x)
    @test AutoRIFT.nsearchable(pts) == length(valid)
    @test all(k -> AutoRIFT.issearchable(pts, k), valid)
    @test all(>(0), pts.radius_x[valid])
end

@testset "search radius and prior carry through" begin
    pts = AutoRIFT.pointset(IPG_R; pixel_size = 30.0)
    valid = findall(!=(SENTINEL), IPG_R.location_x)
    @test all(k -> pts.radius_x[k] == IPG_R.search_x[k], valid)
    @test all(k -> pts.radius_y[k] == IPG_R.search_y[k], valid)
    @test all(k -> pts.dx_prior[k] == IPG_R.offset_x[k], valid)
    @test all(k -> pts.dy_prior[k] == IPG_R.offset_y[k], valid)
end

@testset "the y prior's sign comes from the coordinate system" begin
    # A projected image needs no negation: its +y already points down its own second axis. A radar
    # image's azimuth opposes it, and `ImagePairGeometry.y_displacement_sign` is what answers that.
    # A wrong sign here sends the correlator the wrong way and returns a plausible velocity, so the
    # value is read off the coordinate the result carries rather than asked for.
    pts = AutoRIFT.pointset(IPG_R; pixel_size = 30.0)

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
    valid = findall(!=(SENTINEL), IPG_R.location_x)
    invalid = findall(==(SENTINEL), IPG_R.location_x)
    @test all(k -> pts.chip_size_min_x[k] == IPG_R.chip_min_x[k], valid)
    @test all(k -> pts.chip_size_max_x[k] == IPG_R.chip_max_x[k], valid)
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

@testset "a misregistration reaches the prior and comes back out" begin
    # The offset between two images composes with the velocity-derived prior rather than replacing it:
    # `offset_x`/`offset_y` say where the ice is expected to have moved, the misregistration says where
    # the secondary's grid sits, and a chip has to be cut at the sum.
    g = IPG_R
    plain = AutoRIFT.pointset(g; chip_size = 32)

    # A constant offset over the same window, in the correlator's own sign convention.
    off = fill((3.0, -2.0), size(g.location_x))
    with = AutoRIFT.pointset(g; chip_size = 32, offset = off)

    # Added, not substituted: every point moves by exactly the offset, and the points geogrid marked
    # invalid keep the zero prior they had, since those carry no velocity term to add to.
    @test with.dx_prior ≈ plain.dx_prior .+ 3.0
    @test with.dy_prior ≈ plain.dy_prior .- 2.0
    # Nothing else about the grid moves — a misregistration is not a search radius or a chip bound.
    @test with.x == plain.x && with.y == plain.y
    @test with.radius_x == plain.radius_x && with.radius_y == plain.radius_y

    # Omitting it is exactly the old behaviour, which is what makes this additive.
    @test AutoRIFT.pointset(g; chip_size = 32).dx_prior == plain.dx_prior

    # A field over the wrong window is refused rather than broadcast against a mismatched grid.
    @test_throws DimensionMismatch AutoRIFT.pointset(g; chip_size = 32, offset = fill((0.0, 0.0), 3, 3))

    # Removal is the inverse, so a displacement equal to the offset comes back as zero motion.
    dx = fill(3.0f0, size(off)); dy = fill(-2.0f0, size(off))
    c = AutoRIFT.remove_misregistration(dx, dy, off)
    @test all(iszero, c.dx) && all(iszero, c.dy)
    # And it accepts a result object as well as two arrays, since that is what a run returns.
    @test AutoRIFT.remove_misregistration((dx = dx, dy = dy), off).dx == c.dx
    @test_throws DimensionMismatch AutoRIFT.remove_misregistration(dx, dy, fill((0.0, 0.0), 2, 2))

    # An unsolved point stays unsolved rather than becoming the negated offset, which would look like a
    # confident measurement of the misregistration itself.
    nan_dx = fill(NaN32, 2, 2); nan_dy = fill(NaN32, 2, 2)
    n = AutoRIFT.remove_misregistration(nan_dx, nan_dy, fill((5.0, 5.0), 2, 2))
    @test all(isnan, n.dx) && all(isnan, n.dy)
end

@testset "the misregistration round trip recovers the motion" begin
    # The one test that pins every sign at once. A secondary built by shifting the reference by a known
    # misregistration *plus* a known feature motion must, after correlation and removal, report the
    # motion alone. Getting any sign wrong here fails; reasoning about them at each site does not.
    #
    # The correlator returns the offset from secondary back to reference, so a shift of `+s` is reported
    # as `-s` (see `autorift`'s sign note). The offset field is therefore supplied in that same negated
    # sense, which is what the extension's docstring warns about.
    Random.seed!(42)
    n = 400
    ref = randn(Float32, n, n)
    mis = (7.0, -3.0)
    mot = (2.0, 1.0)
    tot = mis .+ mot
    sec = similar(ref)
    for j in 1:n, i in 1:n
        sec[i, j] = ref[clamp(round(Int, i - tot[2]), 1, n), clamp(round(Int, j - tot[1]), 1, n)]
    end

    r = autorift(ref, sec; chip_size = 32, search_radius = 16)
    med(v) = (s = sort(collect(v)); isodd(length(s)) ? s[(length(s) + 1) ÷ 2] :
                                   (s[length(s) ÷ 2] + s[length(s) ÷ 2 + 1]) / 2)

    # The raw measurement carries both, negated.
    @test med(filter(isfinite, vec(r.dx))) ≈ -tot[1] atol = 0.2
    @test med(filter(isfinite, vec(r.dy))) ≈ -tot[2] atol = 0.2

    # Removing the misregistration in the correlator's convention leaves the motion, negated.
    off = fill((-mis[1], -mis[2]), size(r.dx))
    c = AutoRIFT.remove_misregistration(r, off)
    @test med(filter(isfinite, vec(c.dx))) ≈ -mot[1] atol = 0.2
    @test med(filter(isfinite, vec(c.dy))) ≈ -mot[2] atol = 0.2

    # Stated the way a user reads it: the feature motion is the negation of what comes back.
    @test -med(filter(isfinite, vec(c.dx))) ≈ mot[1] atol = 0.2
    @test -med(filter(isfinite, vec(c.dy))) ≈ mot[2] atol = 0.2
end

@testset "extension code quality" begin
    # The same pair of checks `extensions.jl` runs over the other extensions. Here rather than in that
    # loop because this extension loads only when ImagePairGeometry is resolvable, which `extensions.jl`
    # cannot assume.
    mod = Base.get_extension(AutoRIFT, :AutoRIFTImagePairGeometryExt)
    Aqua.test_undefined_exports(mod)
    Aqua.test_stale_deps(mod)
end
