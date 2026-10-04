# The L4/5 destripe gate (`5.orbit`): the reference's logged scan angles against the orbit route.
#
# The durable geometry lives in two packages, not here: `OpticalDatasets.landsat_ephemeris` parses a
# scene's `_ANG.txt`, `ImagePairGeometry.Orbit`/`ground_track` turn its ephemeris into a ground-track
# displacement in the raster's own CRS, and `AutoRIFT.track_angles` reduces that to `Destripe`'s
# along/cross-track convention. What stays here is harness-only: fetching and caching the `_ANG.txt`
# itself (`scene_ang`, `vsi_text`), reading what the reference logged (`reference_scan_angles`,
# `reference_banding`), and the comparison between the two (`orbit_angle_check`).
#
# **Measured over all six scenes of the three L4/5 pairs**: the orbit's cross-track agrees with the
# reference's to **0.10° at worst**, while its along-track sits **1.45° to 1.89° below** it. The error
# is the reference's and it is all in one axis — the orbit's two directions are perpendicular to the
# last digit by construction, and the reference's are 91.42° to 91.86° apart, never 90°. Its own
# non-perpendicularity accounts for the along-track gap case by case, which is what identifies
# `nanmax` over two edge slopes as the cause (`autoRIFT.py`'s `_get_slopes`): the worse-conditioned
# edge wins and only the along-track pair is affected.

using OpticalDatasets: landsat_ephemeris
using ImagePairGeometry: Orbit, ground_track, fast_transform
using AutoRIFT: track_angles

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

"""
    vsi_text(path) -> String

A whole text object read through GDAL's virtual filesystem.

The route exists because an `_ANG.txt` is only reachable over requester-pays S3: the STAC item's plain
`https` href redirects to an HTML landing page rather than the object, so `Downloads.download` returns a
login form that parses to no fields. `/vsis3` with `AWS_REQUEST_PAYER` is the same access this harness
already uses for the band rasters, and `ArchGDAL.GDAL` re-exports the VSI calls, so this needs no
dependency `ArchGDAL` does not already bring.
"""
function vsi_text(path::AbstractString)
    G = ArchGDAL.GDAL
    ArchGDAL.setconfigoption("AWS_REQUEST_PAYER", "requester")
    h = G.vsifopenl(path, "rb")
    h == C_NULL && error("cannot open $path through GDAL's virtual filesystem; a requester-pays " *
                         "object needs AWS_PROFILE to name credentials that can pay")
    try
        G.vsifseekl(h, 0, 2)                 # SEEK_END
        n = Int(G.vsiftelll(h))
        G.vsifseekl(h, 0, 0)
        buf = Vector{UInt8}(undef, n)
        got = G.vsifreadl(pointer(buf), 1, n, h)
        Int(got) == n || error("short read of $path: $got of $n bytes")
        return String(buf)
    finally
        G.vsifclosel(h)
    end
end

"""
    scene_ang(name, cache) -> String

A local copy of `name`'s `_ANG.txt`, fetched once through the STAC item and cached.

Cached because the gate reads it on every run and the object is requester-pays: 34 KiB is not the cost,
the round trip is. The name is the granule's, so a cache entry is unambiguous.
"""
function scene_ang(name::AbstractString, cache::AbstractString)
    local_path = joinpath(cache, name * "_ANG.txt")
    isfile(local_path) && return local_path
    mkpath(cache)
    url = "https://landsatlook.usgs.gov/stac-server/collections/landsat-c2l1/items/$name"
    item = JSON3.read(String(take!(Downloads.download(url, IOBuffer(); timeout = 60))))
    asset = get(item.assets, Symbol("ANG.txt"), nothing)
    asset === nothing && error("STAC item $name has no `ANG.txt` asset")
    href = asset.alternate.s3.href
    text = vsi_text("/vsis3/" * href[6:end])
    # Through a temporary, so an interrupted fetch does not leave a truncated file that parses.
    tmp = local_path * ".partial"
    write(tmp, text)
    mv(tmp, local_path; force = true)
    return local_path
end

"""
    landsat_scan_angles(name, cache, epsg; spacing) -> (along_track, cross_track)

[`AutoRIFT.track_angles`](@ref) of the orbit's [`ImagePairGeometry.ground_track`](@ref), from the
ephemeris [`scene_ang`](@ref) fetches and caches for `name`.

Sampled at the ephemeris' own midpoint, where the acquisition is centred.
"""
function landsat_scan_angles(name::AbstractString, cache::AbstractString, epsg::Integer;
                             spacing::Tuple{Real,Real})
    e = landsat_ephemeris(scene_ang(name, cache))
    orbit = Orbit(; time = e.time, position = collect(zip(e.x, e.y, e.z)))
    dx, dy = ground_track(orbit, e.time[length(e.time) ÷ 2], fast_transform(4326, epsg))
    return track_angles(dx, dy; spacing)
end

"""
    orbit_angle_check(c::GoldenCase, run, cache) -> Vector{NamedTuple}

Each filtered scene's orbit-derived scan angles beside the reference's own logged pair.

The comparison the register turns on: whether the `_ANG.txt` ephemeris reproduces what `_fft_filter`
recovers from the valid-data region's shape, and where it does not, whether the orbit or the pixels are
the better answer. Scenes are taken in **job order**, since `apply_landsat_filtering(reference,
secondary)` filters and logs them in that order rather than in acquisition order.

**Each scene's EPSG and spacing come from the scene itself**, one per scene: the two scenes of a
cross-zone pair are in different projections, and a bearing is a grid bearing, so using one CRS for both
would put the second scene's angles several degrees out. The granule rather than the run's `filtered/`
copy, which carries the same native grid and which [`prune_run`](@ref) deletes as regenerable.
"""
function orbit_angle_check(c::GoldenCase, run::AbstractString, cache::AbstractString)
    logged = reference_scan_angles(joinpath(run, "capture.log"))
    names = [first(c.reference), first(c.secondary)]
    length(logged) >= length(names) || error(
        "$(run)/capture.log logs $(length(logged)) angle pairs for $(length(names)) filtered " *
        "scenes; the log is truncated or this is not an L4/L5 pair")
    # `names` is in job order and `scene_path` takes acquisition order, so the two are paired by name
    # rather than by position.
    early, late = acquisition_order(c)
    paths = Dict(early => scene_path(c, :reference), late => scene_path(c, :secondary))
    out = NamedTuple[]
    for (i, name) in enumerate(names)
        haskey(paths, name) || error("\"$name\" is neither acquisition of $(c.product)")
        ds = ArchGDAL.read(paths[name])
        epsg = parse(Int, ArchGDAL.toEPSG(ArchGDAL.importWKT(ArchGDAL.getproj(ds))) |> string)
        gt = ArchGDAL.getgeotransform(ds)
        along, cross = landsat_scan_angles(name, cache, epsg; spacing = (gt[2], gt[6]))

        r = logged[i]
        push!(out, (; name, epsg, along, cross, ref_along = r.along, ref_cross = r.cross,
                    d_along = along - r.along, d_cross = cross - r.cross,
                    ours_apart = abs(along - cross), ref_apart = abs(r.along - r.cross)))
    end
    return out
end

"""
    reference_banding(log) -> Vector{Bool}

Whether the reference's band-reject **fired**, one entry per filtered scene.

`_fft_filter` prints its two band powers unconditionally and adds a "No banding filter applied" line
only when it declines (`autoRIFT.py:211-226`), so the decision is readable from the log and does not
have to be inferred from the output — which would be circular, since this is what a rung comparing that
output needs to know.

The decision is a **binary branch on a ratio**, and it can be marginal: on
`LT05_L1GS_001013_19920425` the powers are 1588 and 3279, clearing the `>= 2` test by 3.2%. A scene
that close can be decided differently by two implementations whose input fields differ slightly, and
then its whole output differs — a declined reject returns the clamped input, a fired one returns the
band-rejected field.
"""
function reference_banding(log::AbstractString)
    out = Bool[]
    pending = false
    for line in eachline(log)
        if occursin(r"Cross track power is", line)
            pending && push!(out, true)
            pending = true
        elseif occursin("No banding filter applied", line)
            pending && (push!(out, false); pending = false)
        end
    end
    pending && push!(out, true)
    return out
end
