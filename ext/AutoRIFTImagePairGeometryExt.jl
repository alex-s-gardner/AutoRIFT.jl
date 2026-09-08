# A search grid from a geogrid `PairGeometry`.
#
# `ImagePairGeometry` computes, per grid point, where that point falls in each image of a pair and how
# far the correlator should search around it. This turns that into a `PointSet`. It is a contract
# negotiation rather than a type conversion, and each clause is a place a wrong answer would look
# entirely plausible:
#
# Index base. Geogrid's pixel index is zero-based — `round((x - startingX) / XSize)`, bounds-tested
# against `0` and `nPixels - 1` (`geogridOptical.cpp:723-724,775`). `PointSet` carries one-based
# positions, so every index gains 1.
#
# The half pixel. `autoRIFT.py:890` stores `round(xGrid) + 0.5`, and `_shift_points` reproduces that at
# correlation time for every pyramid level alike (`src/multichip.jl`). So the `+ 0.5` must *not* be
# applied here: doing it twice moves every search centre a pixel.
#
# Missing values. Geogrid marks a point outside the image with `-32767`, while a point is skipped here
# by giving it a zero search radius (`src/points.jl`). Passing the sentinel through as a radius would
# make it negative, and `gridpoints`' margin logic sizes itself from `maximum(radius)`, so the grid
# would be mis-sized rather than the point skipped.
#
# The y sign. Azimuth increases along the track while a north-up raster's `+y` points down, so a radar
# prior needs negating and a projected one does not (`testautoRIFT.py:405-407`, under
# `optical_flag == 0`). `ImagePairGeometry.y_displacement_sign` answers this from the coordinate system,
# and a `PairGeometry` carries its own — so the default is right and a caller has to go out of their
# way to be wrong.
#
# What does not fit. `PointSet` holds ten fields; geogrid produces eighteen numbers per point. The
# displacement-to-velocity operator, the scale factors, the stable-surface mask and the y chip-size
# bounds have no field here, and converting a measured displacement to a velocity needs them:
# `ImagePairGeometry.velocity_conversion` returns those.

module AutoRIFTImagePairGeometryExt

import AutoRIFT
using ImagePairGeometry: PairGeometry, chip_size_pixels, y_displacement_sign

"""
    AutoRIFT.pointset(g::PairGeometry; chip_size = nothing, chip_size_0 = 240.0,
                      pixel_size = nothing, coordinate = g.coordinate) -> PointSet{2}

The search grid in `g`, as a [`PointSet`](@ref).

Pixel positions become one-based, since geogrid's are zero-based, and a point that fell outside the
image gets a search radius of zero — how a point is marked to skip.

The half-pixel offset the reference bakes into its grid is *not* applied here: it is added at
correlation time for every pyramid level, so applying it twice would displace every search centre by a
pixel.

`chip_size` sets the base chip extent in pixels. Given `pixel_size` instead, it is derived as the
reference does — `ceil(chip_size_0 / pixel_size / 4) * 4` — from a chip size in meters. One of the two
is required, since neither is recoverable from `g`.

`coordinate` decides the sign of the y prior, and defaults to the one `g` carries; pass it only to
override that.

Only part of `g` fits a `PointSet`. `ImagePairGeometry.velocity_conversion` returns the operator, the
scale factors and the mask, which converting a measured displacement to a velocity needs.
"""
function AutoRIFT.pointset(g::PairGeometry; chip_size = nothing, chip_size_0 = 240.0,
                           pixel_size = nothing, coordinate = g.coordinate, offset = nothing)
    base = if chip_size !== nothing
        Int(chip_size)
    elseif pixel_size !== nothing
        chip_size_pixels(chip_size_0, pixel_size)
    else
        throw(ArgumentError(
            "pointset needs the base chip extent: pass `chip_size` in pixels, or `pixel_size` in " *
            "meters to derive it from `chip_size_0`. Neither is recoverable from a PairGeometry, " *
            "which stores chip size bounds but not the base."))
    end

    sentinel = Int32(g.nodata.output)
    valid = g.location_x .!= sentinel

    # One-based, and a skipped point still needs a coordinate: `PointSet` has no missing value, so its
    # position is arbitrary and its zero radius is what excludes it.
    x = [v ? Float64(l + 1) : 1.0 for (v, l) in zip(valid, g.location_x)]
    y = [v ? Float64(l + 1) : 1.0 for (v, l) in zip(valid, g.location_y)]

    # A radius is zero where the point is invalid, where the search extent itself is missing, or where
    # geogrid computed no extent at all — each meaning "do not search here".
    rad(band) = [(v && b != sentinel && b > 0) ? Int(b) : 0 for (v, b) in zip(valid, band)]
    rx = rad(g.search_x)
    ry = rad(g.search_y)

    # An absent search-range raster leaves the band uniformly sentinel, which would skip every point.
    # That is a missing input rather than a grid of skips, so say so.
    if all(iszero, rx) && !isempty(rx)
        throw(ArgumentError(
            "every search radius is zero, so no point would be correlated. The PairGeometry has " *
            "no search-range band — it was computed without `srx`/`sry` (and their required " *
            "`dhdx`/`dhdy`). Supply them, or build the PointSet with an explicit radius."))
    end

    dy_flip = y_displacement_sign(coordinate)
    prior(band, flip) = [(v && b != sentinel) ? flip * Float64(b) : 0.0
                         for (v, b) in zip(valid, band)]

    # Chip-size bounds are per point, and zero means unbounded — which is what a missing bound means.
    bound(band) = [(v && b != sentinel && b > 0) ? Int(b) : 0 for (v, b) in zip(valid, band)]

    # The misregistration between the two images, where one was supplied. It *adds* to the
    # velocity-derived prior rather than replacing it: `offset_x`/`offset_y` are where the ice is expected
    # to have moved, this is where the secondary's grid sits relative to the reference's, and a chip has
    # to be cut at the sum of the two. `y_displacement_sign` applies to the velocity term only — the
    # offset is already in the image's own axes, since that is what `pixel_offset` returns.
    mis_x, mis_y = _misregistration(g, offset)

    return AutoRIFT.pointset(x, y;
                             search_radius_x = rx, search_radius_y = ry,
                             chip_size = base,
                             dx_prior = prior(g.offset_x, 1.0) .+ mis_x,
                             dy_prior = prior(g.offset_y, dy_flip) .+ mis_y,
                             chip_size_min_x = bound(g.chip_min_x),
                             chip_size_max_x = bound(g.chip_max_x))
end

# No offset supplied: zero, so the prior is the velocity term alone and nothing changes for a caller that
# has not asked for this. Scalar rather than an array of zeros, since it is only broadcast against.
_misregistration(::PairGeometry, ::Nothing) = (0.0, 0.0)

# An offset field over the same window as the geometry — `ImagePairGeometry.OffsetField` or its lattice
# form — split into the two component arrays the prior needs.
function _misregistration(g::PairGeometry, offset)
    axes(offset) == axes(g.location_x) || throw(DimensionMismatch(
        "the misregistration field has axes $(axes(offset)) but the geometry covers " *
        "$(axes(g.location_x)); both must be over the same window."))
    return ([Float64(o[1]) for o in offset], [Float64(o[2]) for o in offset])
end

"""
    AutoRIFT.remove_misregistration(dx, dy, offset) -> NamedTuple

The correlator's displacement with a misregistration removed, as `(dx, dy)`.

A [`MultichipResult`](@ref)'s `dx`/`dy` include whatever prior the search was centred on, because
`track!` adds the prior back into what it returns. So a run whose prior carried a misregistration returns
a displacement that still contains it, and this subtracts it — leaving the displacement of the *features*
between the two images, which is what a velocity conversion expects.

`offset` is the same field passed to [`AutoRIFT.pointset`](@ref). Applying it in both places with
consistent signs is the whole of the bookkeeping, and getting it wrong is quiet: pass it to one and not
the other and every velocity is wrong by the misregistration, which on a 24-day Sentinel-1 pair is about
18 pixels of range — comparable to the signal being measured.

!!! warning "The offset must be in the correlator's sign convention"
    `dx`/`dy` are the offset from `secondary` back to `reference`, which is the **negative** of the
    displacement of the imaged features — see [`autorift`](@ref)'s note. `ImagePairGeometry.pixel_offset`
    returns the opposite sense: where a ground point sits in the secondary *relative to* the reference.

    So a field taken straight from `pixel_offset` must be negated before it is used here or in
    `pointset`. Measured on a synthetic pair built with a known shift: applying a total shift of `+9`
    samples makes the correlator report `-9`, so a misregistration of `+7` contributes `-7` to what is
    returned, and removing it means subtracting `-7`.

    Both entry points take the field in the same convention, so a caller who negates once and passes the
    same array to both is consistent. What does not work is negating for one and not the other.

`NaN` is preserved: a point the correlator did not solve stays unsolved rather than becoming the negated
offset.
"""
function AutoRIFT.remove_misregistration(dx::AbstractArray, dy::AbstractArray, offset)
    axes(dx) == axes(dy) == axes(offset) || throw(DimensionMismatch(
        "displacement and misregistration must share axes: dx $(axes(dx)), dy $(axes(dy)), " *
        "offset $(axes(offset))."))
    out_x = similar(dx, Float32)
    out_y = similar(dy, Float32)
    for i in eachindex(dx, dy, offset)
        o = offset[i]
        out_x[i] = Float32(dx[i] - o[1])
        out_y[i] = Float32(dy[i] - o[2])
    end
    return (dx = out_x, dy = out_y)
end

AutoRIFT.remove_misregistration(r, offset) =
    AutoRIFT.remove_misregistration(r.dx, r.dy, offset)

end # module
