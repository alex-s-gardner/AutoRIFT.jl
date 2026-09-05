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
                           pixel_size = nothing, coordinate = g.coordinate)
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

    return AutoRIFT.pointset(x, y;
                             search_radius_x = rx, search_radius_y = ry,
                             chip_size = base,
                             dx_prior = prior(g.offset_x, 1.0),
                             dy_prior = prior(g.offset_y, dy_flip),
                             chip_size_min_x = bound(g.chip_min_x),
                             chip_size_max_x = bound(g.chip_max_x))
end

end # module
