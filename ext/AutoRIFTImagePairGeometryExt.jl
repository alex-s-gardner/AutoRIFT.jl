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
using ImagePairGeometry: PairGeometry, ProjectedCoordinate, chip_size_pixels,
                         y_displacement_sign

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

"""
    AutoRIFT.params(g::PairGeometry; chip_size_0 = 240.0, optical, kwargs...) -> Params

The correlator settings the ITS_LIVE driver derives from a geogrid result.

Five keywords a caller would otherwise have to re-derive, each read out of `g` rather than configured
(`testautoRIFT.py:245-250, 330-334`):

  * `chip_size.X = ceil(chip_size_0 / pixel_size / 4) * 4` — the base chip is a fixed *distance*
    divided by the image's pixel size, rounded up to a multiple of four.
  * `chip_size.Y = round(chip_size.X * scale / 2) * 2`, where `scale` is the median of
    `chip_min_y / chip_min_x` over the points where both bounds are present. The parameter chip sizes
    are square on the ground, so that ratio is the y:x *pixel size* ratio — 1.0 wherever the pixel is
    square and about 0.25 on a Sentinel-1 pair, varying per acquisition with the azimuth:range ratio.
  * `chip_size_max` from the largest per-point bound `g` carries, with the same `scale` on Y. Both
    bounds carry it, not just the minimum: the pyramid doubles the two axes together, so a maximum
    scaled on one axis only is reached after a different number of doublings on each.
  * `grid_spacing.X = chip_size.X * grid_spacing_m / chip_size_0`, in pixels — written this way
    rather than as `grid_spacing_m / pixel_size`, which is the same number only when the first
    division is exact.
  * `subpixel`, as the reference's per-level ladder: `16, 32, 64, 64` for optical and
    `32, 64, 128, 128` for radar. A single factor cannot express it, and the reference looks the
    factor up per chip size.

`optical` selects that last pair, matching the driver's `optflag`; it defaults from `g`'s coordinate
system, which is what decides it in production.

Every other keyword is forwarded to [`AutoRIFT.params`](@ref) unchanged, so a caller overrides any of
the above by passing it.
"""
function AutoRIFT.params(g::PairGeometry; chip_size_0 = 240.0,
                         optical::Bool = g.coordinate isa ProjectedCoordinate, kwargs...)
    sentinel = Int32(g.nodata.output)
    pixel_size = abs(g.coordinate.spacing[1])
    chip_x = chip_size_pixels(chip_size_0, pixel_size)

    # Over the points where *both* bounds are present, which is the reference's own condition. A
    # point missing either would contribute a ratio of zero or a division by zero.
    ratios = [Float64(y) / Float64(x)
              for (x, y) in zip(g.chip_min_x, g.chip_min_y)
              if x != sentinel && y != sentinel && x > 0 && y > 0]
    isempty(ratios) && throw(ArgumentError(
        "the geometry carries no point with both chip-size minima present, so the y:x chip ratio " *
        "cannot be derived. It was computed without `csminx`/`csminy`."))
    scale_y = _median!(ratios)

    max_x = maximum(x -> x == sentinel ? Int32(0) : x, g.chip_max_x)
    max_x > 0 || throw(ArgumentError(
        "every chip-size maximum is missing or zero, so no pyramid level could run. The geometry " *
        "was computed without `csmaxx`/`csmaxy`."))

    grid_m = abs(g.geotransform[2])
    spacing = trunc(Int, chip_x * grid_m / chip_size_0)

    return AutoRIFT.params(;
        chip_size = (X = chip_x, Y = _even(chip_x * scale_y)),
        chip_size_max = (X = Int(max_x), Y = _even(Int(max_x) * scale_y)),
        grid_spacing = (X = spacing, Y = spacing),
        subpixel = _subpixel_ladder(optical),
        kwargs...)
end

# `round(x / 2) * 2` — the reference's way of keeping a chip extent even, which the correlator's
# centroid convention requires.
_even(x::Real) = round(Int, x / 2) * 2

# `np.median`: the middle element, or the mean of the two middle ones for an even count. Written here
# rather than taken from `Statistics` because an extension may only load the package's own
# dependencies, and one statistic does not justify making `Statistics` one of them. Sorts in place;
# the caller's vector is a local built for this.
function _median!(v::Vector{Float64})
    sort!(v)
    n = length(v)
    return isodd(n) ? v[(n + 1) ÷ 2] : (v[n ÷ 2] + v[n ÷ 2 + 1]) / 2
end

# One refinement factor per pyramid level, coarsest last. The reference holds these as a dictionary
# keyed by chip size (`testautoRIFT.py:330-334`) and looks the factor up per level; `Params` holds
# the same ladder positionally, with the last entry applying to every remaining level.
_subpixel_ladder(optical::Bool) =
    optical ? (AutoRIFT.PyramidRefine(16), AutoRIFT.PyramidRefine(32), AutoRIFT.PyramidRefine(64),
               AutoRIFT.PyramidRefine(64)) :
    (AutoRIFT.PyramidRefine(32), AutoRIFT.PyramidRefine(64), AutoRIFT.PyramidRefine(128),
     AutoRIFT.PyramidRefine(128))

end # module
