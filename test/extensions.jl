# The geospatial extensions: `Raster` in, `RasterStack` out, and the sign conventions that go
# with knowing which way is north.
#
# Loaded last, deliberately. Everything above this runs with neither Rasters nor DimensionalData
# in the session, so a core file that started depending on them would fail rather than pass
# silently because a later test had loaded them.

using Rasters
using ArchGDAL          # triggers RastersArchGDALExt, and so AutoRIFTRastersExt: see below
using DimensionalData
using DimensionalData.Lookups
using Dates
using AutoRIFT: autorift

# `AutoRIFTRastersExt` needs ArchGDAL as well as Rasters, because reading a raster from a file goes
# through `RastersArchGDALExt` and that is what carries GDAL. Asserted rather than assumed: without
# ArchGDAL the extension does not load, `autorift` on a projected raster falls through to the
# DimensionalData method, and every test below would pass while checking the wrong path -- returning a
# `DimStack` of `dx`/`dy` where a `RasterStack` of `vx`/`vy` is expected.
@test Base.get_extension(AutoRIFT, :AutoRIFTRastersExt) !== nothing

# A north-up projected pair with a known shift. North-up means y *decreasing*, which is what a
# GeoTIFF normally stores and what makes the y-flip question real.
#
# `shift = (drow, dcol)` moves features by that many pixels in array terms, so a positive `drow`
# moves them toward increasing row index — southward on a north-up grid.
function projected_pair(n, shift; res = 10.0, epsg = 3031)
    a = synthetic_texture(n; seed = 1)
    b = circshift(a, shift)
    x = X(Projected(0.0:res:(res * (n - 1)); order = ForwardOrdered(), span = Regular(res),
                    sampling = Intervals(Start()), crs = EPSG(epsg)))
    y = Y(Projected((res * (n - 1)):(-res):0.0; order = ReverseOrdered(),
                    span = Regular(-res), sampling = Intervals(Start()), crs = EPSG(epsg)))
    return Raster(a, (y, x)), Raster(b, (y, x))
end

# Radar range-Doppler: real coordinates, no CRS, and dims that are not X/Y.
function radar_pair(n, shift)
    a = synthetic_texture(n; seed = 1)
    b = circshift(a, shift)
    az = Dim{:azimuth}(Sampled(1.0:1.0:n; order = ForwardOrdered(), span = Regular(1.0),
                               sampling = Intervals(Start())))
    rg = Dim{:range}(Sampled(1.0:1.0:n; order = ForwardOrdered(), span = Regular(1.0),
                             sampling = Intervals(Start())))
    return DimArray(a, (az, rg)), DimArray(b, (az, rg))
end

med(v) = (s = sort(collect(filter(!isnan, v))); isempty(s) ? NaN : s[(length(s) + 1) ÷ 2])

const EXT_KW = (; chip_size = 32, search_radius = 25, subpixel = :none)

@testset "y-flip equivalence" begin
    # The headline check of this milestone, and the reason the extension exists rather than a
    # `parent()` call at the call site.
    #
    # A north-up raster stores y decreasing; a south-up one stores it increasing. `parent()` hands
    # the core whichever order the file used, so the core sees a vertically mirrored image and
    # returns `dy` of the opposite sign — `dx` unchanged, `dy` negated. Nothing downstream can
    # detect that: a velocity field with `vy` inverted still looks like ice flow.
    #
    # So the same scene stored both ways must give the same field, and it is asserted rather than
    # inspected.
    #
    # 405, not a round 400, and the reason is worth stating because it is a property of the grid
    # rather than of this test. `gridpoints` spans `(margin+1):spacing:(n-margin)`, so the leftover
    # at the trailing edge is whatever the step leaves — the grid is *not* generally centred in the
    # image. Flip the storage order and the grid lands on different ground: at n = 400 the two
    # cover 1170–3090 m and 900–2820 m, sharing no sample point at all, and no comparison between
    # them is meaningful. At n = 405 the inset divides evenly (90 lead, 90 trail), the grid is
    # symmetric, and the flipped grids coincide — which is what makes the fields comparable.
    ref, sec = projected_pair(405, (4, 6))
    north = autorift(ref, sec; EXT_KW...)
    south = autorift(reverse(ref; dims = Y), reverse(sec; dims = Y); EXT_KW...)

    # Same ground, opposite storage order.
    @test collect(lookup(south, Y)) == reverse(collect(lookup(north, Y)))
    @test collect(lookup(south, X)) == collect(lookup(north, X))

    # And the same field. This is the assertion the milestone turns on: a sign error here would
    # invert `vy` and still look like ice flow, so it has to be caught mechanically.
    #
    # The displacements are **bit-identical**, which is the strong statement and the one that
    # matters — the y handling is exact, not merely close.
    for layer in (:vx, :vy)
        @test all(isequal.(parent(north[layer]), parent(reverse(south[layer]; dims = Y))))
    end

    # `correlation` agrees to 2e-6 rather than exactly, and that is float arithmetic rather than a
    # defect. Reversing the rows reverses the order in which the integral images accumulate down a
    # column, and floating-point addition is not associative, so the denominator's last bits
    # differ. It cannot move a peak — the displacements above are exact — so the tolerance is the
    # honest assertion here.
    cn = parent(north.correlation)
    cs = parent(reverse(south.correlation; dims = Y))
    @test maximum(abs.(filter(!isnan, cn .- cs))) < 1e-5
    @test count(isnan, cn) == count(isnan, cs)

    # Not vacuous: there is real signal in these fields, and it is the same signal.
    @test med(north.vx) == 6
    @test med(north.vy) == -4
    @test med(south.vx) == 6
    @test med(south.vy) == -4
end

@testset "sign convention is map-oriented feature motion" begin
    # Two conversions happen at this boundary and both are easy to get backwards, so each is
    # asserted with a motion whose direction is unambiguous.
    #
    # `+vx` is east and `+vy` is north, and the values are *feature motion* — not the
    # secondary-to-reference offset the core reports.
    east = autorift(projected_pair(400, (0, 6))...; EXT_KW...)
    @test med(east.vx) == 6            # features moved to higher x: east
    @test med(east.vy) == 0

    south = autorift(projected_pair(400, (4, 0))...; EXT_KW...)
    @test med(south.vx) == 0
    @test med(south.vy) == -4          # higher row index on a north-up grid: south

    north = autorift(projected_pair(400, (-4, 0))...; EXT_KW...)
    @test med(north.vy) == 4

    # And the relationship to the array core is exactly the documented one: `vx = -dx`, and on a
    # north-up raster `vy = +dy`, because the row-down and north-up flips cancel. Asserted so the
    # extension is provably a coordinate wrapper rather than a second algorithm.
    ref, sec = projected_pair(400, (4, 6))
    core = autorift(parent(ref), parent(sec); EXT_KW...)
    st = autorift(ref, sec; EXT_KW...)
    @test all(isequal.(parent(st.vx), -core.dx))
    @test all(isequal.(parent(st.vy), core.dy))
    @test all(isequal.(parent(st.correlation), core.correlation))
end

@testset "output grid is geolocated" begin
    # The output lives on the *grid*, not the image, and the grid is inset from the edges — so its
    # coordinates have to come from the input lookups sampled at the grid's pixel positions.
    n, res, spacing = 512, 10.0, 32
    ref, sec = projected_pair(n, (0, 5); res)
    st = autorift(ref, sec; EXT_KW..., grid_spacing = spacing)

    grid = AutoRIFT._build_grid(size(ref), AutoRIFT.params(; EXT_KW..., grid_spacing = spacing))
    @test size(st.vx) == size(grid)

    xs, ys = lookup(st, X), lookup(st, Y)
    inx, iny = lookup(ref, X), lookup(ref, Y)
    # Every output coordinate is the input coordinate at that grid point's pixel index.
    @test collect(xs) == [inx[round(Int, i)] for i in grid.x[1, :]]
    @test collect(ys) == [iny[round(Int, i)] for i in grid.y[:, 1]]

    # Spacing is the grid step times the input pixel size, and the axis order survives.
    @test step(xs) == spacing * res
    @test step(ys) == -spacing * res
    @test order(ys) isa ReverseOrdered

    # Regular, not Irregular. This matters beyond tidiness: an irregular axis has no single pixel
    # size, so `dt` could not convert to velocity. Indexing a lookup with a vector would lose it.
    @test span(xs) isa Regular
    @test span(ys) isa Regular

    # A single-point grid is degenerate but must not error on the step calculation.
    tiny = autorift(projected_pair(200, (0, 0))...; chip_size = 32, search_radius = 25,
                    grid_spacing = 200)
    @test length(tiny.vx) >= 1
end

@testset "CRS is carried and checked" begin
    ref, sec = projected_pair(300, (0, 4))
    st = autorift(ref, sec; EXT_KW...)
    @test crs(st) == EPSG(3031)
    @test crs(st.vx) == EPSG(3031)

    # Two images in different coordinate systems are not co-registered, and correlating them
    # yields a full field of plausible nonsense — so it is refused, naming both.
    other, _ = projected_pair(300, (0, 4); epsg = 3413)
    err = try autorift(ref, other; EXT_KW...) catch e; e end
    @test err isa ArgumentError
    @test occursin("3031", sprint(showerror, err))
    @test occursin("3413", sprint(showerror, err))
end

@testset "misaligned grids are refused" begin
    # Equal *lookups*, not merely equal sizes. This is the likeliest caller mistake and the most
    # damaging: correlation succeeds, every displacement is offset by the grid difference, and
    # nothing downstream can tell.
    ref, sec = projected_pair(300, (0, 4))
    shifted = Raster(parent(sec), (Y(lookup(ref, Y)), X(lookup(ref, X) .+ 1000.0)))
    @test_throws DimensionMismatch autorift(ref, shifted; EXT_KW...)
    @test occursin("co-registered",
                   sprint(showerror, try autorift(ref, shifted; EXT_KW...) catch e; e end))

    # Different dimension names, same shape.
    named = DimArray(parent(sec), (Dim{:azimuth}(1.0:300.0), Dim{:range}(1.0:300.0)))
    @test_throws DimensionMismatch autorift(
        DimArray(parent(ref), (Y(1.0:300.0), X(1.0:300.0))), named; EXT_KW...)
end

@testset "dt converts pixels to velocity" begin
    # `dt` scales the output and nothing else, so the several spellings of it are tested on the
    # smallest pair that still resolves anything, rather than correlating a 400² image four times
    # to exercise four multiplications.
    #
    # 350 is that floor, and the reason is the outlier filter rather than the correlation: its
    # default window is 5 grid points, so a grid narrower than that cannot judge consistency and
    # every point is dropped. A 300² image gives a 4x4 grid and measures *zero* points; 350² gives
    # 6x6 and measures all of them. Worth stating, because "too small" here is a property of the
    # filter's neighbourhood, not of the image.
    n, res = 350, 10.0
    ref, sec = projected_pair(n, (4, 6); res)
    years = 16 / 365.25

    # Default: pixels, so the caller can convert however they like.
    @test med(autorift(ref, sec; EXT_KW...).vx) == 6

    # A fixed period: metres per year, over a Julian year. Both axes, since the y factor is the
    # one that interacts with the flip.
    st = autorift(ref, sec; EXT_KW..., dt = Day(16))
    @test med(st.vx) ≈ 6 * res / years rtol = 1e-5
    @test med(st.vy) ≈ -4 * res / years rtol = 1e-5

    # `Date` subtraction gives a `Day`, which is how a caller most naturally arrives here.
    @test med(autorift(ref, sec; EXT_KW..., dt = Date(2020, 7, 1) - Date(2020, 6, 15)).vx) ≈
        med(st.vx) rtol = 1e-6
    # A plain number is years.
    @test med(autorift(ref, sec; EXT_KW..., dt = 1.0).vx) ≈ 6 * res rtol = 1e-5

    # Calendar periods have no fixed length, so they are refused rather than silently assumed to
    # be 365 days. The message names what to use instead.
    err = try autorift(ref, sec; EXT_KW..., dt = Year(1)) catch e; e end
    @test err isa ArgumentError
    @test occursin("Day", sprint(showerror, err))
    @test_throws ArgumentError autorift(ref, sec; EXT_KW..., dt = Month(6))
    @test_throws ArgumentError autorift(ref, sec; EXT_KW..., dt = -1.0)
    @test_throws ArgumentError autorift(ref, sec; EXT_KW..., dt = "16 days")

    # An irregular axis has no single pixel size, so velocity would be wrong wherever the spacing
    # differs. Refused, rather than scaled by a nominal value.
    irr = Raster(parent(ref), (Y(lookup(ref, Y)),
                               X(Projected(cumsum(fill(res, n)) .+ randn(n) .* 0.1;
                                           order = ForwardOrdered(), span = Irregular(),
                                           sampling = Intervals(Start()), crs = EPSG(3031)))))
    irr2 = Raster(parent(sec), dims(irr))
    @test_throws ArgumentError autorift(irr, irr2; EXT_KW..., dt = Day(16))
    # But without `dt` the same pair works: nothing needs a pixel size.
    @test med(autorift(irr, irr2; EXT_KW...).vx) == 6
end

@testset "layers and missingval" begin
    ref, sec = projected_pair(300, (0, 4))
    st = autorift(ref, sec; EXT_KW...)

    @test issetequal(keys(st), (:vx, :vy, :correlation, :peak_ratio, :chip_size, :interpolated))
    @test eltype(st.vx) === Float32
    @test eltype(st.vy) === Float32
    @test eltype(st.correlation) === Float32
    @test eltype(st.chip_size) === UInt16
    @test eltype(st.interpolated) === Bool

    # Per-layer, set at construction rather than left for a caller to discover. NaN for the float
    # layers because that is what the core writes for "not measured", which is deliberately
    # distinct from a measured zero.
    mv = Rasters.missingval(st)
    @test isnan(mv.vx) && isnan(mv.vy) && isnan(mv.correlation)
    @test mv.chip_size == 0
    @test mv.interpolated == false

    # Every layer shares the grid dims.
    @test all(l -> dims(st[l]) == dims(st), keys(st))
end

@testset "validity masks reach the correlator" begin
    # Masks may arrive as plain arrays or as rasters, since a caller holding rasters naturally
    # holds its masks the same way.
    n = 400
    ref, sec = projected_pair(n, (0, 5))
    m = trues(n, n)
    m[100:250, 100:250] .= false

    plain = autorift(ref, sec; EXT_KW..., reference_valid = m, secondary_valid = m)
    wrapped = autorift(ref, sec; EXT_KW...,
                       reference_valid = Raster(collect(m), dims(ref)),
                       secondary_valid = Raster(collect(m), dims(ref)))
    @test all(isequal.(parent(plain.vx), parent(wrapped.vx)))
    # Masking removes points rather than corrupting them.
    @test count(isnan, parent(plain.vx)) > count(isnan, parent(autorift(ref, sec; EXT_KW...).vx))
    @test med(plain.vx) == 5
end

@testset "DimStack path: dimensional but unprojected" begin
    # Radar range-Doppler: real coordinates, no CRS, and dims that are not X/Y. An extension that
    # reached for `dims(A, Y)` would work on optical imagery and fail here — which is exactly the
    # data this path exists for.
    ref, sec = radar_pair(400, (4, 6))
    ds = autorift(ref, sec; EXT_KW...)

    @test ds isa DimStack
    @test !(ds isa RasterStack)
    @test map(DimensionalData.name, dims(ds)) == (:azimuth, :range)
    @test issetequal(keys(ds), (:dx, :dy, :correlation, :peak_ratio, :chip_size, :interpolated))

    # `dx`/`dy` here, not `vx`/`vy`: without a CRS there is no map orientation to flip to, so the
    # core's raw secondary-to-reference offsets are the honest output.
    core = autorift(parent(ref), parent(sec); EXT_KW...)
    @test all(isequal.(parent(ds.dx), core.dx))
    @test all(isequal.(parent(ds.dy), core.dy))
    @test med(ds.dx) == -6
    @test med(ds.dy) == -4

    # Coordinates still come from the input lookups.
    grid = AutoRIFT._build_grid(size(ref), AutoRIFT.params(; EXT_KW...))
    @test size(ds.dx) == size(grid)
    @test collect(lookup(ds, Dim{:range})) == [lookup(ref, Dim{:range})[round(Int, i)]
                                               for i in grid.x[1, :]]

    # A projected raster does *not* take this path, even though a Raster is an AbstractDimArray.
    proj = autorift(projected_pair(300, (0, 4))...; EXT_KW...)
    @test proj isa RasterStack
    @test haskey(proj, :vx)
end

@testset "extension overhead is negligible" begin
    # The extension must be a thin coordinate wrapper, not a second cost centre. Measured as
    # allocations rather than time, which is the deterministic part: unwrapping is `parent()` and
    # rewrapping touches only the grid, which is ~100x smaller than the image.
    n = 512
    ref, sec = projected_pair(n, (0, 5))
    a, b = parent(ref), parent(sec)
    autorift(ref, sec; EXT_KW...)          # compile both paths
    autorift(a, b; EXT_KW...)

    wrapped = @allocated autorift(ref, sec; EXT_KW...)
    plain = @allocated autorift(a, b; EXT_KW...)
    @test wrapped < 1.05 * plain
end

# ---------------------------------------------------------------------------
# The sparse first-guess stage (ImageFeatures extension)
# ---------------------------------------------------------------------------

using ImageFeatures
using AutoRIFT: first_guess, ORBGuess

# SAR-like amplitude: fully-developed speckle is a circular Gaussian field, and its magnitude is
# what an amplitude product carries. Smoothed so neighbouring samples correlate, or a shifted copy
# has no structure to match.
function _speckle_amp(n::Int; seed = 1)
    rng = Random.MersenneTwister(seed)
    z = complex.(randn(rng, Float32, n, n), randn(rng, Float32, n, n))
    for _ in 1:2
        z = (z .+ circshift(z, (1, 0)) .+ circshift(z, (0, 1)) .+ circshift(z, (1, 1))) ./ 4
    end
    return abs.(z)
end

@testset "first_guess recovers a large displacement" begin
    # The case the stage exists for: a shift far outside any affordable dense search radius.
    n, shift = 512, (37, -23)
    a = _speckle_amp(n)
    b = circshift(a, shift)

    g = first_guess(a, b, ORBGuess(; num_keypoints = 3000))
    @test g isa AutoRIFT.PointSet
    @test length(g.x) > 100

    # The sign convention is the one thing here that must not be wrong: `dx`/`dy` are the offset
    # from *secondary* back to reference, matching `track!`. `circshift(a, (37, -23))` makes the
    # secondary equal to the reference moved by (+37, -23), so reference-minus-secondary is
    # (-37, +23) — dy = -37, dx = +23. Getting this backwards would centre every search window on
    # the far side of the true peak, which is worse than having no prior at all.
    @test med(g.dx_prior) ≈ 23 atol = 1
    @test med(g.dy_prior) ≈ -37 atol = 1
end

@testset "the guess is what makes a small search radius work" begin
    # The payoff, as a measurement rather than an assertion of intent. A 43-px displacement is
    # invisible to a dense search until the radius reaches it — and a radius that large is both
    # slow and prone to spurious peaks.
    n, shift = 512, (37, -23)
    a = _speckle_amp(n)
    b = circshift(a, shift)
    kw = (; chip_size = 32, chip_size_max = 32)

    # Without a guess, radius 6 finds nothing at all.
    blind = autorift(a, b; search_radius = 6, kw...)
    @test nmeasured(blind) == 0

    # With one, radius 6 measures every point — the prior did the reaching.
    g = first_guess(a, b, ORBGuess(; num_keypoints = 3000))
    guided = autorift(a, b, g)
    @test nmeasured(guided) == length(guided.dx)
    @test med(filter(!isnan, guided.dx)) ≈ 23 atol = 0.5
    @test med(filter(!isnan, guided.dy)) ≈ -37 atol = 0.5
    # And it finds far more of them than a wide blind search does: measured 2093 against 144 at
    # radius 50, because a wide radius forces a coarse grid to stay affordable.
    wide = autorift(a, b; search_radius = 50, kw...)
    @test nmeasured(guided) > 5 * nmeasured(wide)
end

@testset "first_guess rejects what it cannot use" begin
    n = 256
    a = _speckle_amp(n)
    # Mismatched shapes.
    @test_throws DimensionMismatch first_guess(a, _speckle_amp(n + 8), ORBGuess())
    # A featureless image has nothing to scale and nothing to detect.
    @test_throws ArgumentError first_guess(fill(1.0f0, n, n), fill(1.0f0, n, n), ORBGuess())
    # Two unrelated fields: matches exist but none is spatially consistent, so the filter empties
    # and `min_matches` reports it rather than returning a prior built from noise.
    @test_throws ArgumentError first_guess(a, _speckle_amp(n; seed = 99),
                                           ORBGuess(; num_keypoints = 500))
end

@testset "extension code quality" begin
    # Aqua on each extension, per the milestone. Ambiguities are checked across the whole loaded
    # set rather than per module, since that is where an extension would introduce one.
    for ext in (:AutoRIFTDimensionalDataExt, :AutoRIFTRastersExt, :AutoRIFTImageFeaturesExt)
        mod = Base.get_extension(AutoRIFT, ext)
        @test mod !== nothing
        Aqua.test_undefined_exports(mod)
        Aqua.test_stale_deps(mod)
    end
    Aqua.test_ambiguities([AutoRIFT])
end

# ---------------------------------------------------------------------------
# Windowed reads from a disk-backed array
# ---------------------------------------------------------------------------
#
# Here rather than in `test/tile.jl` because it loads a package, and the core testsets deliberately
# run with no optional dependency in the session. The property under test is that the *core* needs
# none: a block is read with `copyto!` over a view, which any array supporting windowed reads answers
# efficiently, so `process_block_size` bounds memory for a lazy `Raster`, Zarr, NetCDF or HDF5 input
# without `src/` knowing any of them exist.

import DiskArrays

# A disk-backed array that counts what is read from it, standing in for any windowed backend —
# Rasters over GeoTIFF, Zarr, NetCDF, HDF5. Counting reads asserts the memory property *structurally*
# rather than by watching RSS, which measures allocator slack as much as requirement: if a
# scene-sized read never happens, no scene-sized array can exist.
mutable struct CountingDisk{T} <: DiskArrays.AbstractDiskArray{T,2}
    size::Tuple{Int,Int}
    seed::Int
    calls::Int
    elements::Int
    widest::Int
end
CountingDisk{T}(size, seed) where {T} = CountingDisk{T}(size, seed, 0, 0, 0)

Base.size(a::CountingDisk) = a.size
DiskArrays.haschunks(::CountingDisk) = DiskArrays.Chunked()
DiskArrays.eachchunk(a::CountingDisk) = DiskArrays.GridChunks(a.size, (256, 256))
# Values are a deterministic function of position, so nothing is stored and the "file" is free.
_disk_value(::Type{T}, i, j, seed) where {T} =
    T(0.5 + 0.4 * sin(i * 0.03 + seed) * cos(j * 0.021 + seed))
# A `Bool` file is a caller-supplied valid mask, and every pixel is valid: what such a mask costs to
# read is the point of it, not which pixels it excludes.
_disk_value(::Type{Bool}, i, j, seed) = true
function DiskArrays.readblock!(a::CountingDisk{T}, dest, r::AbstractUnitRange...) where {T}
    a.calls += 1
    n = prod(length.(r))
    a.elements += n
    a.widest = max(a.widest, n)
    for (jj, j) in enumerate(r[2]), (ii, i) in enumerate(r[1])
        dest[ii, jj] = _disk_value(T, i, j, a.seed)
    end
    return nothing
end

# A *striped* disk array: chunks one row tall, which is what `Rasters.write` produces by default and
# what makes `approx_chunksize` report a block far too small to correlate in.
mutable struct CountingStripe{T} <: DiskArrays.AbstractDiskArray{T,2}
    size::Tuple{Int,Int}
    seed::Int
end
Base.size(a::CountingStripe) = a.size
DiskArrays.haschunks(::CountingStripe) = DiskArrays.Chunked()
DiskArrays.eachchunk(a::CountingStripe) = DiskArrays.GridChunks(a.size, (a.size[1], 1))
function DiskArrays.readblock!(a::CountingStripe{T}, dest, r::AbstractUnitRange...) where {T}
    for (jj, j) in enumerate(r[2]), (ii, i) in enumerate(r[1])
        dest[ii, jj] = _disk_value(T, i, j, a.seed)
    end
    return nothing
end

@testset "a blocked run reads a disk-backed scene once" begin
    # A resident array cannot demonstrate anything about reading — it is already in memory — so the
    # input here is one that only materializes what is asked for, and counts it.
    n = 512
    bs = (256, 256)                      # pixels, so 8 grid points at this spacing
    opts = (; chip_size = 32, chip_size_max = 32, grid_spacing = 32, search_radius = 12,
            process_block_size = bs)
    a = CountingDisk{Float32}((n, n), 1)
    b = CountingDisk{Float32}((n, n), 2)
    lazy = autorift(a, b; opts...)

    # Once, and exactly once. Windowed reads would be 2-9x this: a block reads its own extent grown
    # by the halo, and does it again for every pass of every chip size.
    for img in (a, b)
        @test img.calls >= 1
        @test img.elements == n * n
    end

    # The mask is not read at all, because it is derived: `isfinite` over the array now in memory is
    # the same answer the file would give. Defaulted here, which is the common case — a caller's own
    # mask is data the imagery does not carry and is read like it.
    pair = AutoRIFT.ImagePair(a, b)
    @test AutoRIFT.ondisk(pair.reference_valid)
    @test AutoRIFT._ondisk_bytes(pair) == 2 * n * n * sizeof(Float32)
    prefetched = AutoRIFT._prefetched(pair, typemax(Int), AutoRIFT.params(; chip_size = 32))
    @test !AutoRIFT.ondisk(prefetched.reference)
    @test prefetched.reference_valid isa AutoRIFT.FiniteMask
    @test prefetched.reference_valid.parent === prefetched.reference

    # And the answer is the one a resident array gives, so reading up front is not a different
    # computation. Compared against the run above rather than repeating it.
    resident_ref = [_disk_value(Float32, i, j, 1) for i in 1:n, j in 1:n]
    resident_sec = [_disk_value(Float32, i, j, 2) for i in 1:n, j in 1:n]
    assert_same_result(autorift(resident_ref, resident_sec; opts...), lazy, "lazy equals resident")
end

@testset "a pair too large to hold is read window by window" begin
    # The fallback, which is what a scene larger than the machine takes: no array the size of it is
    # formed at any point, and the answer is the same. `correlate_tiled` is the path that never
    # prefetches — it correlates the pair it is given — so it is what the budget selects when the
    # pair does not fit, and what this asserts against.
    n = 512
    p = AutoRIFT.params(; chip_size = 32, chip_size_max = 32, grid_spacing = 32,
                        search_radius = 12)
    a = CountingDisk{Float32}((n, n), 1)
    b = CountingDisk{Float32}((n, n), 2)
    grid = AutoRIFT._build_grid((n, n), p)
    bs = (256, 256)
    layout = AutoRIFT.block_layout(grid, p, (n, n), bs)
    pair = AutoRIFT.ImagePair(a, b)

    # A budget below the pair's bytes keeps the pair as given — identically, not as a copy.
    @test AutoRIFT._prefetched(pair, AutoRIFT._ondisk_bytes(pair) - 1, p) === pair
    windowed = AutoRIFT.correlate_tiled(pair, grid, p, bs)

    # No single read is scene-sized: every read is a block's window or a chunk of one.
    largest_window = maximum(length(blk.read_rows) * length(blk.read_cols)
                             for blk in layout.blocks)
    for img in (a, b)
        @test img.calls > 1                       # windowed, not one big read
        @test img.widest <= largest_window        # and no read exceeds a block's window
        @test img.widest < n * n                  # so the scene is never materialized at once
    end

    # More than the scene, because each block reads a halo its neighbour also reads — the cost the
    # prefetch above exists to remove. No more than one sweep of the layout's windows per pass, which is
    # what makes `_windowed_bytes` an upper bound.
    #
    # **Not a whole number of sweeps.** A pass reads the windows `_read_windows` derives from the point set
    # it is about to correlate, and those shrink to that set's own span — so a fine pass after the coarse
    # mask has narrowed the searchable points reads less than the layout's window for that block. The
    # layout's window is the bound the buffers are sized to, not a quantum the reads come in.
    total_window = sum(length(blk.read_rows) * length(blk.read_cols) for blk in layout.blocks)
    sweeps = 2 * length(AutoRIFT.chip_sizes(p))
    @test a.elements > n * n
    @test a.elements <= total_window * sweeps
    # The two images of a pair are read together, window for window, so their totals cannot diverge.
    @test a.elements == b.elements
    @test AutoRIFT._windowed_bytes(pair, layout, p) >= a.elements * sizeof(Float32) +
                                                      b.elements * sizeof(Float32)

    # One read per window, not two. The default valid mask above is the image's own finiteness, so a
    # block computes it from the window in hand; a mask over separate storage is data the imagery does
    # not carry, and reads every window again. That second read is what the default must not do, and
    # the run below is the measurement of what it would cost: the same windows, a third time over.
    own_r = CountingDisk{Float32}((n, n), 1)
    own_s = CountingDisk{Float32}((n, n), 2)
    own_m = CountingDisk{Bool}((n, n), 3)
    AutoRIFT.correlate_tiled(AutoRIFT.ImagePair(own_r, own_s; reference_valid = own_m), grid, p, bs)
    @test own_r.elements == a.elements                  # the imagery read is unchanged...
    @test own_m.elements == a.elements                  # ...and the mask adds a full pass of its own
    @test AutoRIFT._windowed_bytes(AutoRIFT.ImagePair(a, b; reference_valid = own_m), layout, p) >
          AutoRIFT._windowed_bytes(pair, layout, p)     # which the estimate charges and the default not

    resident_ref = [_disk_value(Float32, i, j, 1) for i in 1:n, j in 1:n]
    resident_sec = [_disk_value(Float32, i, j, 2) for i in 1:n, j in 1:n]
    assert_same_result(autorift(resident_ref, resident_sec; chip_size = 32, chip_size_max = 32,
                                grid_spacing = 32, search_radius = 12),
                       windowed, "windowed equals resident")
end

@testset "what a prefetch will hold is known before it runs" begin
    # The decision has to be takeable without touching the imagery, or a scene too large for the
    # machine would have to be read to find that out.
    n = 64
    a = CountingDisk{Float32}((n, n), 1)
    b = CountingDisk{Float32}((n, n), 2)
    pair = AutoRIFT.ImagePair(a, b; reference_valid = trues(n, n), secondary_valid = trues(n, n))
    # The images, and not the caller's own masks: those are already in memory.
    @test AutoRIFT._ondisk_bytes(pair) == 2 * n * n * sizeof(Float32)
    # One image on disk, and a mask derived from it adds nothing: it is rebuilt over the prefetched
    # image rather than materialized, so it stores no bytes of its own.
    @test AutoRIFT._ondisk_bytes(AutoRIFT.ImagePair(zeros(Float32, n, n), b)) ==
          n * n * sizeof(Float32)
    # What makes that mask free is one predicate, shared by the estimate and every read that acts on
    # it, so the two cannot diverge. Identity, not equality: two masks over equal arrays are two reads.
    @test AutoRIFT._derived_from(AutoRIFT.FiniteMask(a), a)
    @test !AutoRIFT._derived_from(AutoRIFT.FiniteMask(a), b)
    img = zeros(Float32, n, n)
    @test AutoRIFT._derived_from(AutoRIFT.FiniteMask(img), img)
    @test !AutoRIFT._derived_from(AutoRIFT.FiniteMask(copy(img)), img)
    @test !AutoRIFT._derived_from(trues(n, n), a)   # a caller's own mask is never derived
    # Nothing on disk means nothing to decide, and the pair is returned as it came.
    dense = AutoRIFT.ImagePair(zeros(Float32, n, n), ones(Float32, n, n))
    @test AutoRIFT._ondisk_bytes(dense) == 0
    @test AutoRIFT._prefetched(dense, typemax(Int), AutoRIFT.params(; chip_size = 16)) === dense
    @test a.calls == 0                             # and none of the above read anything

    # A slab read is the whole array however it is split, and a serial run takes one slab.
    p_ser = AutoRIFT.params(; chip_size = 16, threaded = false)
    p_thr = AutoRIFT.params(; chip_size = 16, threaded = true)
    want = [_disk_value(Float32, i, j, 1) for i in 1:n, j in 1:n]
    @test AutoRIFT._slab_read(a, p_ser) == want
    @test a.calls == 1                             # one slab, one read
    @test AutoRIFT._slab_read(a, p_thr) == want
    # A mask read the same way, since a caller's mask over a file is not derivable from the imagery.
    @test AutoRIFT._slab_read(AutoRIFT.FiniteMask(a), p_thr) == map(isfinite, want)

    # A threaded read splits on a boundary the storage shares, so no chunk is decoded by two slabs —
    # unaligned, that decode is paid twice, which `memory.md` measures. The height is a whole
    # number of chunk rows and at least one chunk, whatever the thread count asks for.
    @test AutoRIFT._chunk_rows(a) == 256                      # the extension reads the backend
    @test AutoRIFT._chunk_rows(AutoRIFT.FiniteMask(a)) == 256 # and a mask follows its parent
    @test AutoRIFT._chunk_rows(want) == 1                     # an in-memory array constrains nothing
    for nthr in (1, 4, 64)
        h = AutoRIFT._slab_height(a, AutoRIFT.PREFETCH_SLABS_PER_THREAD * nthr)
        @test h % 256 == 0
        @test h >= 256
    end
end

@testset "the automatic budget compares two read volumes" begin
    # Both volumes are computable from geometry, so the decision is a property of the scene and the
    # layout rather than of the machine — every case below holds whatever memory happens to be free.
    n = 1024
    p = AutoRIFT.params(; chip_size = 32, chip_size_max = 32, grid_spacing = 32, search_radius = 12)
    pair = AutoRIFT.ImagePair(CountingDisk{Float32}((n, n), 1), CountingDisk{Float32}((n, n), 2))
    pairbytes = 2 * n * n * sizeof(Float32)
    @test AutoRIFT._ondisk_bytes(pair) == pairbytes

    # A grid covering the scene reads every block once per sweep, which outweighs one read of the pair.
    full = AutoRIFT.block_layout(AutoRIFT._build_grid((n, n), p), p, (n, n), (256, 256))
    @test AutoRIFT._windowed_bytes(pair, full, p) > pairbytes
    # `free` fixed, so what this asserts is the comparison rather than the machine it ran on.
    @test AutoRIFT._auto_cache_budget(pair, full, p; free = typemax(Int)) == pairbytes

    # A small area of interest in a large file is the other way round: only the blocks holding points
    # are read at all, so windowing those costs less than reading the whole pair would.
    aoi = AutoRIFT.block_layout(
        AutoRIFT.gridpoints(100.0:32.0:300.0, 100.0:32.0:300.0; chip_size = 32, search_radius = 12),
        p, (n, n), (256, 256))
    @test AutoRIFT._windowed_bytes(pair, aoi, p) < pairbytes
    @test AutoRIFT._auto_cache_budget(pair, aoi, p; free = typemax(Int)) == 0

    # The windowed figure counts a caller's own mask, which is read per block, and not one derived
    # from its image, which `_block_pair!` computes from the window already read.
    masked = AutoRIFT.ImagePair(pair.reference, pair.secondary;
                                reference_valid = CountingDisk{Bool}((n, n), 3))
    @test AutoRIFT._windowed_bytes(masked, full, p) > AutoRIFT._windowed_bytes(pair, full, p)

    # Free memory is a ceiling on the answer, never the criterion: a pair that should be cached is
    # not when the machine cannot hold it, and the windowed reads are the fallback. An eighth, so a
    # run competes for a share of what is free rather than the last of it.
    thr = AutoRIFT.params(; chip_size = 32, chip_size_max = 32, grid_spacing = 32,
                          search_radius = 12, threaded = true)
    @test AutoRIFT._auto_cache_budget(pair, full, thr; free = 8 * pairbytes) == pairbytes
    @test AutoRIFT._auto_cache_budget(pair, full, thr; free = 8 * pairbytes - 1) == 0

    # A serial run divides that share by the thread count, because that is the batch shape `autorift!`
    # documents: one pair per task, up to `nthreads` at once, each deciding with no view of the others.
    # `p` is serial, `threaded = false` being the default.
    nt = Threads.nthreads()
    @test AutoRIFT._auto_cache_budget(pair, full, p; free = 8 * pairbytes * nt) == pairbytes
    nt > 1 && @test AutoRIFT._auto_cache_budget(pair, full, p; free = 8 * pairbytes) == 0

    @test pair.reference.calls == 0                # none of the above read anything
end

@testset "cache_budget chooses between the two read strategies" begin
    n = 512
    opts = (; chip_size = 32, chip_size_max = 32, grid_spacing = 32, search_radius = 12,
            process_block_size = (256, 256))
    pairbytes = 2 * n * n * sizeof(Float32)
    mk() = (CountingDisk{Float32}((n, n), 1), CountingDisk{Float32}((n, n), 2))

    # `:auto` is the default, so passing it explicitly must not change what a run reads.
    a, b = mk()
    auto = autorift(a, b; opts...)
    @test a.elements == n * n
    a2, b2 = mk()
    assert_same_result(autorift(a2, b2; opts..., cache_budget = :auto), auto, "explicit :auto")
    @test a2.elements == n * n

    # A budget above the pair reads it once; one below it keeps the windowed reads. The boundary is
    # the pair's own bytes, since the cache holds the whole pair or none of it.
    a3, b3 = mk()
    autorift(a3, b3; opts..., cache_budget = pairbytes)
    @test a3.elements == n * n
    a4, b4 = mk()
    windowed = autorift(a4, b4; opts..., cache_budget = pairbytes - 1)
    @test a4.elements > n * n

    # `nothing` is that fallback named, and the answer does not depend on which strategy ran.
    a5, b5 = mk()
    assert_same_result(autorift(a5, b5; opts..., cache_budget = nothing), auto, "cache_budget = nothing")
    @test a5.elements == a4.elements
    assert_same_result(windowed, auto, "windowed equals cached")

    # Written as a float, which is how a fraction of memory arrives.
    a6, b6 = mk()
    autorift(a6, b6; opts..., cache_budget = 1.0e9)
    @test a6.elements == n * n

    @test_throws "must be a finite, non-negative number of bytes" autorift(mk()...; opts...,
                                                                          cache_budget = -1)
    @test_throws "`cache_budget = :some` is not a choice" autorift(mk()...; opts...,
                                                                  cache_budget = :some)
    @test_throws "must be a number of bytes" autorift(mk()...; opts..., cache_budget = "1GB")

    # The cache carries the budget, so a pair swapped in later is read the way the first one was
    # rather than the way this machine's free memory happens to suggest at that moment.
    a7, b7 = mk()
    cache = AutoRIFT.init(a7, b7; opts..., cache_budget = nothing)
    @test cache.runner.cache_budget == 0
    autorift!(cache)
    a8, b8 = mk()
    AutoRIFT.reinit!(cache; reference = a8, secondary = b8)
    @test cache.runner.cache_budget == 0
    autorift!(cache)
    @test a8.elements == a4.elements               # still windowed, as the first pair was
end

@testset "an unblocked run refuses disk-backed input" begin
    # The unblocked path filters the scene rather than reading windows, and its box filters accumulate
    # elementwise — so a disk-backed image is read one pixel at a time, each read decompressing a whole
    # chunk. On a 15901x13435 GeoTIFF pair that does not finish: 74 minutes and 1.7e9 allocations,
    # still inside the first image's filter. Nothing about the run says so, which is why it must fail
    # at the call rather than run.
    n = 256
    opts = (; chip_size = 16, chip_size_max = 16, grid_spacing = 16, search_radius = 8)
    a = CountingDisk{Float32}((n, n), 1)
    b = CountingDisk{Float32}((n, n), 2)
    @test AutoRIFT.ondisk(a)                                  # the extension's method is loaded
    @test !AutoRIFT.ondisk(zeros(Float32, 4, 4))              # and a plain array is not disk-backed

    # Every unblocked entry point, since each reaches `_prepare` by a different route. The counter
    # proves the refusal is not a slow success: no read happens at all.
    grid = AutoRIFT._build_grid((n, n), AutoRIFT.params(; opts...))
    @test_throws "still on disk" autorift(a, b; opts...)
    @test_throws "still on disk" autorift(a, b, grid; opts...)
    @test_throws "still on disk" autorift(a, b, AutoRIFT.params(; opts...))
    @test_throws "still on disk" AutoRIFT.init(a, b; opts...)
    @test a.calls == 0

    # A scattered point set has no block layout, so blocking cannot rescue it — the error names the
    # same cause, and `read` is the way out.
    @test_throws "still on disk" autorift(a, b, AutoRIFT.scatter(grid); opts...)

    # The message names both remedies, and both work. Blocking reads windows;
    blocked = autorift(a, b; opts..., process_block_size = (128, 128))
    @test a.calls > 0
    # and materializing first is the same computation, which is what makes the advice honest.
    dense_a = [_disk_value(Float32, i, j, 1) for i in 1:n, j in 1:n]
    dense_b = [_disk_value(Float32, i, j, 2) for i in 1:n, j in 1:n]
    assert_same_result(autorift(dense_a, dense_b; opts...), blocked, "materialized equals blocked")
end

# ---------------------------------------------------------------------------
# A raster still on disk
# ---------------------------------------------------------------------------
#
# Every other raster test above is in-memory, and that is why three separate lazy-read defects
# survived to be found by hand on a real scene: `FiniteMask` had no bulk `getindex`, `_read_block!`
# wrapped its input in a view, and `resident` converted elementwise. Each turned one windowed read
# into one read per *pixel* — invisible in memory, where the two cost the same.

# `assert_same_result` compares `MultichipResult` fields; a `RasterStack` has `vx`/`vy` layers instead,
# so stack comparison gets its own helper rather than bending that one.
function assert_same_stack(a, b, tag)
    @testset "$tag" begin
        for l in (:vx, :vy, :correlation, :chip_size, :interpolated)
            @test isequal(parent(a[l]), parent(b[l]))
        end
    end
end

@testset "a file-backed raster correlates where it lies" begin
    # A real GeoTIFF, written and read back, so the whole GDAL path is exercised rather than a mock:
    # tiling, compression, nodata, and the eltype `Union{Missing,T}` that comes with it.
    n = 384
    a = synthetic_texture(n; seed = 3)
    b = circshift(a, (4, -3))
    res = 10.0
    x = X(Projected(0.0:res:(res * (n - 1)); order = ForwardOrdered(), span = Regular(res),
                    sampling = Intervals(Start()), crs = EPSG(3031)))
    y = Y(Projected((res * (n - 1)):(-res):0.0; order = ReverseOrdered(),
                    span = Regular(-res), sampling = Intervals(Start()), crs = EPSG(3031)))

    mktempdir() do dir
        pa, pb = joinpath(dir, "a.tif"), joinpath(dir, "b.tif")
        Rasters.write(pa, Raster(a, (y, x)); force = true)
        Rasters.write(pb, Raster(b, (y, x)); force = true)

        lazy_a = Raster(pa; lazy = true)
        lazy_b = Raster(pb; lazy = true)
        @test parent(lazy_a) isa DiskArrays.AbstractDiskArray

        opts = (; chip_size = 16, chip_size_max = 16, grid_spacing = 16, search_radius = 8)
        lazy = autorift(lazy_a, lazy_b; opts...)
        # `read` materializes but keeps `missingval` and the `Union{Missing,T}` eltype, so this also
        # checks that nodata handling does not depend on where the data lives.
        eager = autorift(read(lazy_a), read(lazy_b); opts...)

        @test count(!isnan, parent(lazy.vx)) > 0.5 * length(parent(lazy.vx))
        assert_same_stack(eager, lazy, "lazy raster equals materialized raster")

        # An explicit block size must give the same answer as the chunk-derived default, at sizes
        # either side of the file's own 256-pixel tiling — a partition bug shows up here and not in a
        # single-block run.
        for bs in (128, 512)
            blocked = autorift(lazy_a, lazy_b; opts..., process_block_size = (bs, bs))
            assert_same_stack(eager, blocked, "lazy raster, $(bs)px blocks")
        end
    end
end

@testset "nodata is excluded rather than correlated" begin
    # An in-band fill value, which is what most sensors write, rather than `missing`. The pixels it
    # marks must produce no measurement: reading a fill value as a number would correlate -9999
    # against -9999 and report a confident zero displacement over the gap.
    n = 384
    fill = -9999.0f0
    a = synthetic_texture(n; seed = 5)
    b = circshift(a, (3, -2))
    gap = 150:250
    a[gap, gap] .= fill
    b[gap, gap] .= fill
    dims2 = (Y(1:n), X(1:n))
    opts = (; chip_size = 16, chip_size_max = 16, grid_spacing = 16, search_radius = 8)
    out = autorift(Raster(a, dims2; missingval = fill),
                   Raster(b, dims2; missingval = fill); opts...)

    grid = AutoRIFT._build_grid((n, n), AutoRIFT.params(; opts...))
    interior = findall(eachindex(IndexCartesian(), parent(out.vx))) do k
        # Well inside the gap, so the whole chip is fill rather than straddling the edge.
        170 <= round(Int, grid.y[k]) <= 230 && 170 <= round(Int, grid.x[k]) <= 230
    end
    @test !isempty(interior)
    @test all(isnan, parent(out.vx)[interior])
    # And the rest of the scene is unaffected: this excludes the gap, it does not poison the run.
    @test count(!isnan, parent(out.vx)) > 0.8 * length(parent(out.vx))
end

@testset "a lazy raster is blocked by default" begin
    # The chunk-derived default is what keeps a file-backed run off the unblocked path, where the
    # whole scene is one window. Asserted through the read counter rather than by timing.
    n = 1024
    ext = Base.get_extension(AutoRIFT, :AutoRIFTRastersExt)
    @test ext !== nothing
    a = CountingDisk{Float32}((n, n), 1)
    b = CountingDisk{Float32}((n, n), 2)
    dims2 = (Y(1:n), X(1:n))
    ra, rb = Raster(a, dims2), Raster(b, dims2)
    @test AutoRIFT.ondisk(ra)
    # The default comes from `AutoRIFT.block_size_for`, rounded up to a whole number of the file's
    # chunks: blocking at the chunk size alone reads 2x2 chunks per block because of the halo, so every
    # chunk is decoded several times over. Measured 4.99 s at 256 against 3.48 s at 768 on real imagery.
    # What this asserts is the part the extension decides — alignment, the floor, and the scene bound.
    pblk = AutoRIFT.params(; chip_size = 16, chip_size_max = 16, grid_spacing = 16, search_radius = 8)
    hx, hy = AutoRIFT.halo(AutoRIFT.gridpoints((n, n), pblk.grid_spacing;
                                               chip_size = pblk.chip_size_max,
                                               search_radius = pblk.search_radius), pblk, (n, n))
    got = ext._blocks(nothing, ra, rb, pblk)
    @test got[1] % 256 == 0 && got[2] % 256 == 0        # still chunk-aligned
    @test got[1] >= hx && got[2] >= hy                   # at least the halo, which `block_layout` demands
    @test all(got .>= ext.MIN_BLOCK)
    @test all(got .<= n)                                 # never larger than the scene
    # Emitted as `(X, Y)`, which is how `process_block_size` is read. Checked against `block_size_for`
    # itself so the two cannot drift, and so a transposed pair — invisible on this square halo — fails
    # the anisotropic assertion in `test/tile.jl` rather than nothing at all.
    @test got == (AutoRIFT.block_size_for(pblk, (n, n); chunk = (256, 256),
                                          floor_pixels = ext.MIN_BLOCK).X,
                  AutoRIFT.block_size_for(pblk, (n, n); chunk = (256, 256),
                                          floor_pixels = ext.MIN_BLOCK).Y)
    # A chunk too small to be a sensible block is raised to `MIN_BLOCK`. `Rasters.write` produces
    # *striped* GeoTIFFs, whose chunks are one row tall, and a 5-pixel block is below the halo — which
    # `block_layout` rejects outright. Bounded above by the scene, so a small image stays one block.
    tiny = Raster(CountingStripe{Float32}((512, 512), 3), (Y(1:512), X(1:512)))
    @test all(ext._blocks(nothing, tiny, tiny, pblk) .>= ext.MIN_BLOCK)
    small = Raster(CountingStripe{Float32}((64, 64), 4), (Y(1:64), X(1:64)))
    @test ext._blocks(nothing, small, small, pblk) == (64, 64)   # bounded by the scene
    # An explicit choice always wins, and an in-memory pair stays unblocked.
    @test ext._blocks((64, 64), ra, rb, pblk) == (64, 64)
    mem = Raster(zeros(Float32, 8, 8), (Y(1:8), X(1:8)))
    @test ext._blocks(nothing, mem, mem, pblk) === nothing
    @test !AutoRIFT.ondisk(mem)

    autorift(ra, rb; chip_size = 16, chip_size_max = 16, grid_spacing = 16, search_radius = 8)
    # The run completing at all is the assertion that it blocked: the unblocked path refuses
    # disk-backed input. A pair this size fits the prefetch budget, so the file is read exactly once
    # rather than a window per block per pass.
    @test a.elements == n * n
    @test b.elements == n * n

    # `cache_budget` reaches the core from here too, and is not forwarded to `AutoRIFT.params` —
    # which takes correlation parameters only and rejects anything else. Every accepted form, since
    # each leaves this method by the same route.
    ropts = (; chip_size = 16, chip_size_max = 16, grid_spacing = 16, search_radius = 8)
    for cb in (:auto, nothing, 0, 1.0e9)
        c, d = CountingDisk{Float32}((n, n), 1), CountingDisk{Float32}((n, n), 2)
        @test autorift(Raster(c, dims2), Raster(d, dims2); ropts..., cache_budget = cb) isa RasterStack
    end
    # And the chunking the alignment reads survives the wrappers `_lazy_input` builds over a raster
    # that declares nodata: the mask is read from the file, so it wants the same slab boundaries.
    @test AutoRIFT._chunk_rows(ra) == 256
    @test AutoRIFT._chunk_rows(parent(ra)) == 256
end
