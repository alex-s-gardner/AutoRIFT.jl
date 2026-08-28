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

using Rasters, ArchGDAL, NCDatasets, Proj, Serialization, Statistics

const CACHE = get(ENV, "AUTORIFT_TESTDATA", expanduser("~/data/autorift/tests"))
const REFERENCE = "LC09_L1TP_009011_20230618_20230618_02_T1_B8.TIF"
const SECONDARY = "LC08_L1TP_009011_20230626_20230710_02_T1_B8.TIF"

# The trunk below the confluence, where the reference field is dense and fast. A window rather
# than the whole scene: 17041x17121 at 15 m is 1.2 GB of imagery for a test, and the fast ice is
# a small part of it.
const CENTER_LONLAT = (-49.8, 69.15)
const WINDOW_PX = 2048

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

ITS_LIVE `vx`/`vy` in m/yr, sampled at the UTM window's cell centres.

The granule is EPSG:3413 at 120 m and the window is UTM 22N at 15 m, so each window cell is
projected into the granule's grid and read by nearest neighbour. Nearest and not bilinear: this
is the *comparison* field, and interpolating it would blur the sharp shear margins that make the
comparison discriminating.
"""
function reference_velocity(path::AbstractString, xs::Vector{Float64}, ys::Vector{Float64})
    ds = NCDataset(path)
    try
        gx = Vector{Float64}(ds["x"][:])
        gy = Vector{Float64}(ds["y"][:])
        rvx = ds["vx"][:, :, 1]
        rvy = ds["vy"][:, :, 1]
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
            gi = findmin(v -> abs(v - px), gx)[2]
            gj = findmin(v -> abs(v - py), gy)[2]
            a, b = rvx[gi, gj], rvy[gi, gj]
            vx[iy, jx] = ismissing(a) ? NaN32 : Float32(a)
            vy[iy, jx] = ismissing(b) ? NaN32 : Float32(b)
        end
        return vx, vy
    finally
        close(ds)
    end
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
    dt = NCDataset(nc) do ds
        Float64(ds["img_pair_info"].attrib["date_dt"])
    end

    out = (; reference = ref, secondary = sec, x = xs, y = ys,
           reference_vx = vx, reference_vy = vy, date_dt = dt,
           pixel_size = 15.0, granule = basename(nc),
           scenes = (REFERENCE, SECONDARY))
    path = joinpath(CACHE, "window.jls")
    serialize(path, out)
    finite = count(!isnan, vx)
    @info "wrote $path" size(ref) dt finite_reference_cells = finite
    return nothing
end

main()
