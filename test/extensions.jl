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

    @test issetequal(keys(st), (:vx, :vy, :correlation, :peak_snr, :chip_size, :interpolated))
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
    @test issetequal(keys(ds), (:dx, :dy, :correlation, :peak_snr, :chip_size, :interpolated))

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

@testset "a blocked run never reads the whole scene" begin
    # The claim `process_block_size` exists to make: peak memory tracks the block, not the scene. A
    # resident array cannot demonstrate it — the scene is already in memory — so the input here is
    # one that only materializes what is asked for.
    n = 512
    p = AutoRIFT.params(; chip_size = 32, chip_size_max = 32, grid_spacing = 32,
                        search_radius = 12)
    a = CountingDisk{Float32}((n, n), 1)
    b = CountingDisk{Float32}((n, n), 2)
    grid = AutoRIFT._build_grid((n, n), p)
    bs = (256, 256)                      # pixels, so 8 grid points at this spacing
    layout = AutoRIFT.block_layout(grid, p, (n, n), bs)

    opts = (; chip_size = 32, chip_size_max = 32, grid_spacing = 32, search_radius = 12,
            process_block_size = bs)
    lazy = autorift(a, b; opts..., reference_valid = trues(n, n),
                    secondary_valid = trues(n, n))

    # No single read is scene-sized: every read is a block's window or a chunk of one.
    largest_window = maximum(length(blk.read_rows) * length(blk.read_cols)
                             for blk in layout.blocks)
    for img in (a, b)
        @test img.calls > 1                       # windowed, not one big read
        @test img.widest <= largest_window        # and no read exceeds a block's window
        @test img.widest < n * n                  # so the scene is never materialized at once
    end

    # The halo is read more than once, by construction — that is the cost blocking pays. Bounded by
    # the sum of the read windows, which is what the layout promises.
    total_window = sum(length(blk.read_rows) * length(blk.read_cols) for blk in layout.blocks)
    @test a.elements <= total_window
    @test a.elements > n * n                      # strictly more than the scene: the halo overlap

    # And the answer is the one a resident array gives, so the windowed read is not a different
    # computation. Compared against the run above rather than repeating it.
    resident_ref = [_disk_value(Float32, i, j, 1) for i in 1:n, j in 1:n]
    resident_sec = [_disk_value(Float32, i, j, 2) for i in 1:n, j in 1:n]
    assert_same_result(autorift(resident_ref, resident_sec; opts...), lazy, "lazy equals resident")
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
    # A sentinel fill value, which is what most sensors write, rather than `missing`. The pixels it
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
    @test ext._ondisk(ra)
    # The default is `HALO_BLOCKS` halos, rounded up to a whole number of the file's chunks: blocking at
    # the chunk size alone reads 2x2 chunks per block because of the halo, so every chunk is decoded
    # several times over. Measured 4.99 s at 256 against 3.48 s at 768 on real imagery.
    pblk = AutoRIFT.params(; chip_size = 16, chip_size_max = 16, grid_spacing = 16, search_radius = 8)
    hx, hy = AutoRIFT.halo(AutoRIFT.gridpoints((n, n), pblk.grid_spacing;
                                               chip_size = pblk.chip_size_max,
                                               search_radius = pblk.search_radius), pblk, (n, n))
    got = ext._blocks(nothing, ra, rb, pblk)
    @test got[1] % 256 == 0 && got[2] % 256 == 0        # still chunk-aligned
    @test got[1] >= ext.HALO_BLOCKS * hy                # and at least the halo-derived target
    @test all(got .<= n)                                 # never larger than the scene
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
    @test !ext._ondisk(mem)

    autorift(ra, rb; chip_size = 16, chip_size_max = 16, grid_spacing = 16, search_radius = 8)
    # Windowed, not one read of the scene: the whole point of the default.
    @test a.calls > 1
    @test a.widest < n * n
end
