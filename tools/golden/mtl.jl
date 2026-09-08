# The scan geometry of a Landsat scene, from its `_MTL.txt`.
#
# **Temporary, and deliberately harness-only.** The durable home for this is `ImagePairGeometry`, which
# already owns acquisition geometry — `ImageFootprint`, `ProjectedCoordinate`, `Orbit`, `incidence_angle` —
# and has the Rasters extension to attach it to a raster at load time. Parallax will want the same fields,
# so a parser here would be duplicated there. This exists so the L4/5 destripe gate is not blocked on that
# package, and it is deleted when the real loader lands: nothing outside `tools/golden` calls it, and the
# gate it feeds compares angles against the reference's own logged values rather than against this parse.
#
# **Why metadata rather than the image.** `_fft_filter` recovers the scene footprint from pixels —
# `connectedComponentsWithStats`, `findContours`, `moments`, `minAreaRect`, a `warpAffine`d quadrant map,
# four `distanceTransform` argmaxes — because it only ever receives an array (`autoRIFT.py:153-206`). The
# MTL carries the four corners in projected metres directly, which is the quantity all of that estimates.
# `process.py:312`'s own `FIXME` records that the derived route is degraded by geogrid's corner rounding
# and chopping, so the metadata is also the better value, not merely the cheaper one.

using Printf

"""
    mtl_fields(path) -> Dict{String,String}

Every `KEY = VALUE` pair in a Landsat `_MTL.txt`, as strings.

The format is flat `KEY = VALUE` inside `GROUP`/`END_GROUP` blocks, and keys are unique across groups in
every Landsat collection, so the groups are ignored rather than parsed. Values keep their quotes stripped
and are left as text: a caller that wants a number says so, which keeps a malformed field an error at the
point of use rather than a silent zero here.
"""
function mtl_fields(path::AbstractString)
    isfile(path) || error("no MTL at $path")
    out = Dict{String,String}()
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || s == "END") && continue
        i = findfirst('=', s)
        i === nothing && continue
        key = strip(s[1:prevind(s, i)])
        val = strip(s[nextind(s, i):end])
        (startswith(key, "GROUP") || startswith(key, "END_GROUP")) && continue
        out[String(key)] = String(strip(val, ['"']))
    end
    isempty(out) && error("$path parsed to no fields; is it an MTL?")
    return out
end

"""
    mtl_corners(fields) -> NamedTuple

The four scene corners in projected metres, as `(ul, ur, ll, lr)` of `(x, y)`.

Named for the *product* corners, `CORNER_*_PROJECTION_{X,Y}_PRODUCT`, and not the L1T grid corners: these
are the footprint of the imaged swath, which is what the destripe filter's along- and cross-track angles
describe. A scene missing them is an error — a `Float64` default would put the footprint at the origin and
produce two plausible angles from nothing.
"""
function mtl_corners(fields::AbstractDict)
    get2(c) = let kx = "CORNER_$(c)_PROJECTION_X_PRODUCT", ky = "CORNER_$(c)_PROJECTION_Y_PRODUCT"
        haskey(fields, kx) && haskey(fields, ky) ||
            error("MTL has no $kx / $ky; these are the projected product corners the scan " *
                  "geometry is read from, and a scene without them cannot be destriped from metadata")
        (parse(Float64, fields[kx]), parse(Float64, fields[ky]))
    end
    return (; ul = get2("UL"), ur = get2("UR"), ll = get2("LL"), lr = get2("LR"))
end

"""
    scan_angles(corners; spacing) -> (along_track, cross_track)

The along- and cross-track angles in degrees, on the reference's own convention.

Reproduces `_get_slopes` (`autoRIFT.py:138-151`) exactly, and the exactness matters because it is a
*choice* rather than a derivation: the reference takes `nanmax` of two slopes per axis, so a footprint whose
opposite edges differ slightly resolves to the larger angle rather than to their mean. Reading corners from
metadata changes where the slopes come from, not which of the two is picked.

Its `_calculate_slope` is `atan((y1 - y2) / (x1 - x2))` in degrees — note `atan` of a ratio and not
`atan2`, so the result is in `(-90, 90)` and a vertical edge gives `±90` by the division overflowing rather
than by a branch. Along-track is from the bottom and top edges, cross-track from the two side edges.

**Corners arrive in projected metres and the reference's are in pixels.** A slope is a ratio, so a uniform
scale cancels — but `spacing` is required rather than defaulted because a non-square pixel does *not*
cancel, and silently assuming square is the kind of thing that produces a plausible wrong angle.
"""
function scan_angles(corners; spacing::Tuple{Real,Real})
    sx, sy = Float64(spacing[1]), Float64(spacing[2])
    # Into pixel units, where the reference's slopes are taken. The y sign is irrelevant to a `nanmax`
    # over both edges of an axis, and is left as the projection has it.
    px(p) = (p[1] / sx, p[2] / sy)
    ul, ur, ll, lr = px(corners.ul), px(corners.ur), px(corners.ll), px(corners.lr)
    slope(a, b) = rad2deg(atan((a[2] - b[2]) / (a[1] - b[1])))
    # `_get_slopes(tl, tr, bl, br)`: along-track from the bottom and top edges, cross-track from the sides.
    along = maximum(skipnan((slope(ll, lr), slope(ul, ur))))
    cross = maximum(skipnan((slope(lr, ur), slope(ll, ul))))
    return (along, cross)
end

# `nanmax`: the reference uses `np.nanmax`, which ignores a `NaN` slope rather than propagating it. A `NaN`
# arises when two corners coincide, which a degenerate footprint can produce.
skipnan(t) = (v = filter(!isnan, collect(t)); isempty(v) ? [NaN] : v)

"""
    mtl_scan_angles(path; spacing) -> (along_track, cross_track)

[`scan_angles`](@ref) of [`mtl_corners`](@ref) of [`mtl_fields`](@ref), for the common case.
"""
mtl_scan_angles(path::AbstractString; spacing) =
    scan_angles(mtl_corners(mtl_fields(path)); spacing)

"""
    reference_scan_angles(log) -> Vector{NamedTuple}

The along- and cross-track angles the reference *printed*, one entry per filtered scene.

`_fft_filter` logs both before it decides anything (`autoRIFT.py:201-202`), so the container log carries the
values its own pixel-derived route produced. That is what the metadata route is checked against — the gate
is agreement with the reference, and its log is the only place its intermediate geometry is visible.
"""
function reference_scan_angles(log::AbstractString)
    out = NamedTuple{(:along, :cross),Tuple{Float64,Float64}}[]
    along = nothing
    for line in eachline(log)
        m = match(r"Along track angle is\s+(-?[\d.]+)\s+degrees", line)
        m === nothing || (along = parse(Float64, m.captures[1]); continue)
        m = match(r"Cross track angle is\s+(-?[\d.]+)\s+degrees", line)
        if m !== nothing && along !== nothing
            push!(out, (; along, cross = parse(Float64, m.captures[1])))
            along = nothing
        end
    end
    return out
end
