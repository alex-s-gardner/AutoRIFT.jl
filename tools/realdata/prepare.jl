# Cut the Jakobshavn Isbrae test case out of two full Landsat scenes and the ITS_LIVE granule
# that covers them, into the cache `test/realdata.jl` reads.
#
# Run once, after placing the three source files in the cache (see tools/realdata/README.md):
#
#     julia --project=tools/realdata tools/realdata/prepare.jl
#
# Writes `window.jls`: the two image windows, the reference velocities resampled onto the same
# grid, and the acquisition separation. Everything the test needs, so the test itself loads no
# GeoTIFF and needs neither ArchGDAL nor Proj.

using Rasters, ArchGDAL, NCDatasets, Proj, Serialization, Statistics, Extents

include(joinpath(@__DIR__, "parameters.jl"))

const CACHE = get(ENV, "AUTORIFT_TESTDATA", expanduser("~/data/autorift/tests"))
const REFERENCE = "LC09_L1TP_009011_20230618_20230618_02_T1_B8.TIF"
const SECONDARY = "LC08_L1TP_009011_20230626_20230710_02_T1_B8.TIF"

# The trunk below the confluence, where the reference field is dense and fast. A window rather
# than the whole scene: 17041x17121 at 15 m is 1.2 GB of imagery for a test, and the fast ice is
# a small part of it.
const CENTER_LONLAT = (-49.8, 69.15)
# The imagery window, and the sub-region of it the test compares.
#
# The two differ because the outlier filter is a neighbourhood operation iterated three times, over
# a grid that the coarse pass decimates: its reach is `(window ÷ 2) * iterations * stride` grid
# points, which at a 9-wide window, three iterations, stride 4 and 8-pixel spacing is 384 pixels.
# A point within that of the imagery edge is filtered against a truncated field and mostly returns
# nothing. Measured directly: the same inner region rises from 2.7% coverage to 13.2% as the
# surrounding imagery grows from nothing to 512 pixels, with no other change.
#
# ITS_LIVE processed the whole 245 km scene, so every point here is deeply interior for it. A test
# window that compares its own edge is comparing an artefact of its own size.
const WINDOW_PX = 3072
const COMPARE_INSET_PX = 512

"""
    scene_window(path, center_utm, npx) -> (Matrix{Float32}, x, y)

`npx`-square window of `path` centred on `center_utm`, as `Float32` with zero for missing, in
AutoRIFT's `(row, col) = (y, x)` layout.

Both scenes are UTM 22N at 15 m but their origins differ by 1200 m, so the window is cut by
*coordinate* rather than by index — indexing both at the same `i, j` would offset them by 80
pixels and the correlator would report that offset as ice motion.

The `permutedims` is load-bearing. A GeoTIFF read by Rasters has dimension order `(X, Y)`, so
`Array` of it is indexed `[x, y]` — column first. AutoRIFT indexes `[row, col]` like every other
Julia matrix, so the two are transposes of each other. For a square window the shapes agree and
the error is silent: the imagery correlates against itself perfectly while every displacement
comes out transposed.
"""
function scene_window(path::AbstractString, center::Tuple{Float64,Float64}, npx::Int)
    r = Raster(path; lazy = true)
    x, y = dims(r, X), dims(r, Y)
    ci = findmin(v -> abs(v - center[1]), parent(x))[2]
    cj = findmin(v -> abs(v - center[2]), parent(y))[2]
    half = npx ÷ 2
    xi = (ci - half):(ci + half - 1)
    yj = (cj - half):(cj + half - 1)
    (first(xi) >= 1 && last(xi) <= length(x) && first(yj) >= 1 && last(yj) <= length(y)) ||
        error("window at $center runs outside $(basename(path))")
    w = r[X(xi), Y(yj)]
    img = permutedims(map(v -> ismissing(v) ? 0.0f0 : Float32(v), Array(w)))
    return img, Vector{Float64}(parent(dims(w, X))), Vector{Float64}(parent(dims(w, Y)))
end

"""
    reference_velocity(path, xs, ys) -> (vx, vy)

ITS_LIVE `vx`/`vy` in m/yr, bilinearly sampled at the UTM window's cell centres.

The granule is EPSG:3413 at 120 m and the window is UTM 22N at 15 m, so every window cell is
projected into the granule's grid and interpolated there.

Bilinear and not nearest. The two grids share a 120 m posting but not a lattice — different
projections, and the window's origin is wherever it was cut — so a window cell falls at an
arbitrary position inside a granule cell, up to 60 m from its centre. Nearest neighbour turns
that offset into a sampling error of half the local velocity gradient, which is negligible on
smooth ice and reaches 456 m/yr at the steepest 1% of the shear margins. That error is a
property of the comparison, not of either velocity field, and it appears as a signed dipole
straddling every margin.
"""
function reference_velocity(path::AbstractString, xs::Vector{Float64}, ys::Vector{Float64})
    ds = NCDataset(path)
    try
        gx = Vector{Float64}(ds["x"][:])
        gy = Vector{Float64}(ds["y"][:])
        rvx = ds["vx"][:, :, 1]
        rvy = ds["vy"][:, :, 1]
        # Both axes are regular, so the enclosing cell is arithmetic rather than a search.
        x0, dx = gx[1], gx[2] - gx[1]
        y0, dy = gy[1], gy[2] - gy[1]
        to3413 = Proj.Transformation("EPSG:32622", "EPSG:3413"; always_xy = true)
        vx = fill(NaN32, length(ys), length(xs))
        vy = similar(vx)
        # Two layouts meet here and neither is negotiable. The granule's `vx` has dimensions
        # `(x, y, time)`, so it is read `[gi, gj]` with `gi` the x index. The output matches the
        # imagery, so it is written `[iy, jx]` with the row first. Writing either one the other
        # way round transposes the comparison, and against a near-square window that produces a
        # plausible-looking field that correlates with nothing.
        for (jx, xv) in pairs(xs), (iy, yv) in pairs(ys)
            px, py = to3413(xv, yv)
            vx[iy, jx] = _bilinear(rvx, (px - x0) / dx + 1, (py - y0) / dy + 1)
            vy[iy, jx] = _bilinear(rvy, (px - x0) / dx + 1, (py - y0) / dy + 1)
        end
        return vx, vy
    finally
        close(ds)
    end
end

# Bilinear sample of `A` at fractional index `(i, j)`, `NaN32` outside it or against any missing
# corner. All four corners must be present: interpolating across the edge of the valid region
# would invent a velocity where the granule declined to report one.
function _bilinear(A, i::Real, j::Real)
    (i < 1 || j < 1 || i > size(A, 1) - 1 || j > size(A, 2) - 1) && return NaN32
    i0, j0 = floor(Int, i), floor(Int, j)
    fi, fj = i - i0, j - j0
    a, b = A[i0, j0], A[i0 + 1, j0]
    c, d = A[i0, j0 + 1], A[i0 + 1, j0 + 1]
    (ismissing(a) || ismissing(b) || ismissing(c) || ismissing(d)) && return NaN32
    lo = (1 - fi) * Float64(a) + fi * Float64(b)
    hi = (1 - fi) * Float64(c) + fi * Float64(d)
    return Float32((1 - fj) * lo + fj * hi)
end

"""
    _window_parameters(xs, ys, date_dt) -> NamedTuple

The production search radii and chip-size bounds, in pixels, on the window's grid.

`production_parameters` returns them on their own 120 m EPSG:3413 lattice; every field is sampled
to the window by nearest neighbour, which is right here because all four are piecewise-constant
over regions far larger than a cell. The search range converts from m/yr through this pair's
separation — see `search_radius_pixels`.
"""
function _window_parameters(xs::Vector{Float64}, ys::Vector{Float64}, date_dt::Real)
    to3413 = Proj.Transformation("EPSG:32622", "EPSG:3413"; always_xy = true)
    corners = [to3413(x, y) for x in (xs[1], xs[end]) for y in (ys[1], ys[end])]
    px = first.(corners)
    py = last.(corners)
    # A cell of margin, so the nearest-neighbour sample at the window edge has a cell to find.
    ex = Extent(X = (minimum(px) - 120, maximum(px) + 120),
                Y = (minimum(py) - 120, maximum(py) + 120))
    st = production_parameters(ex)

    gx = Vector{Float64}(parent(dims(st, X)))
    gy = Vector{Float64}(parent(dims(st, Y)))
    x0, dx = gx[1], gx[2] - gx[1]
    y0, dy = gy[1], gy[2] - gy[1]
    fields = (:search_range_x, :search_range_y, :chip_size_min_x, :chip_size_max_x,
              :prior_vx, :prior_vy, :stable_surface)
    planes = map(f -> Array(st[f]), fields)

    # `Float64` throughout: the search ranges and chip sizes are integers in the rasters but the
    # priors are velocities, and rounding them here rather than on the way out would quantize the
    # search centre to whole metres per year.
    out = map(_ -> zeros(Float64, length(ys), length(xs)), fields)
    for (jx, xv) in pairs(xs), (iy, yv) in pairs(ys)
        qx, qy = to3413(xv, yv)
        gi = clamp(round(Int, (qx - x0) / dx) + 1, 1, size(planes[1], 1))
        gj = clamp(round(Int, (qy - y0) / dy) + 1, 1, size(planes[1], 2))
        for (o, pl) in zip(out, planes)
            v = pl[gi, gj]
            o[iy, jx] = ismissing(v) ? 0.0 : Float64(v)
        end
    end
    named = NamedTuple{fields}(out)
    # The prior is a velocity in m/yr and `dx_prior` is a displacement in pixels, secondary-to-
    # reference — the negative of motion, matching `peak_offset`. The y sign flips again because a
    # row index increases southward while the prior's northing increases northward.
    tometres = date_dt / 365.25 / 15.0
    return (; search_radius_x = search_radius_pixels(named.search_range_x, date_dt, 15.0),
            search_radius_y = search_radius_pixels(named.search_range_y, date_dt, 15.0),
            # Chip sizes are metres in the rasters and pixels everywhere in this package.
            chip_size_min_px = round.(Int, named.chip_size_min_x ./ 15),
            chip_size_max_px = round.(Int, named.chip_size_max_x ./ 15),
            prior_dx = round.(Int, .-named.prior_vx .* tometres),
            prior_dy = round.(Int, named.prior_vy .* tometres),
            # The same prior in m/yr, unrounded. `prior_dx` is quantized to whole pixels because
            # that is what centres a search window; a co-registration shift is a residual against
            # the prior, and rounding it first would fold up to half a pixel — 342 m/yr over an
            # eight-day pair — into the residual it is trying to measure.
            prior_vx = named.prior_vx,
            prior_vy = named.prior_vy,
            # Where a co-registration shift may be estimated: bedrock and other terrain that does
            # not move, which is what makes a residual displacement there an instrument offset
            # rather than ice motion.
            stable_surface = named.stable_surface .> 0)
end

function main()
    ref_path = joinpath(CACHE, "landsat", REFERENCE)
    sec_path = joinpath(CACHE, "landsat", SECONDARY)
    nc = only(filter(endswith(".nc"), readdir(joinpath(CACHE, "itslive"), join = true)))
    for p in (ref_path, sec_path, nc)
        isfile(p) || error("missing $p — see tools/realdata/README.md")
    end

    to_utm = Proj.Transformation("EPSG:4326", "EPSG:32622"; always_xy = true)
    center = to_utm(CENTER_LONLAT...)
    @info "window" center WINDOW_PX

    ref, xs, ys = scene_window(ref_path, center, WINDOW_PX)
    sec, xs2, ys2 = scene_window(sec_path, center, WINDOW_PX)
    # The two windows must be the same ground, not merely the same shape. A tolerance of half a
    # pixel: the scenes' origins differ by a whole number of pixels, so a correct cut agrees
    # exactly and anything else is a misalignment that would read as motion.
    maximum(abs.(xs .- xs2)) <= 7.5 && maximum(abs.(ys .- ys2)) <= 7.5 ||
        error("windows do not cover the same ground: dx up to $(maximum(abs.(xs .- xs2))) m")
    # Stationary ground must correlate at zero shift. This is what catches a transposed read: a
    # square window transposed still correlates against itself, so only a cross-pair check on
    # real terrain distinguishes the two layouts.
    size(ref) == (length(ys), length(xs)) ||
        error("imagery is $(size(ref)) but the axes are $((length(ys), length(xs))) — transposed")

    vx, vy = reference_velocity(nc, xs, ys)
    dt, shift = NCDataset(nc) do ds
        (Float64(ds["img_pair_info"].attrib["date_dt"]),
         # The co-registration ITS_LIVE *already removed* from what it publishes, estimated over
         # stable terrain. Carried so a comparison can put both fields in the same frame: our own
         # output is uncorrected, so it needs the same shift applied.
         (x = Float64(ds["vx"].attrib["stable_shift"]),
          y = Float64(ds["vy"].attrib["stable_shift"])))
    end

    # The parameters production ran with, sampled onto the window. Read rather than chosen: a
    # scalar search radius wide enough for the trunk is several times wider than production used
    # over the rest of the scene, and an over-wide search invites a false peak on low-contrast ice.
    params = _window_parameters(xs, ys, dt)

    out = (; reference = ref, secondary = sec, x = xs, y = ys,
           compare_inset = COMPARE_INSET_PX,
           reference_vx = vx, reference_vy = vy, date_dt = dt,
           pixel_size = 15.0, granule = basename(nc), stable_shift = shift,
           scenes = (REFERENCE, SECONDARY), params...)
    path = joinpath(CACHE, "window.jls")
    serialize(path, out)
    finite = count(!isnan, vx)
    @info "wrote $path" size(ref) dt finite_reference_cells = finite
    return nothing
end

main()
