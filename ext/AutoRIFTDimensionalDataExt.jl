"""
    AutoRIFTDimensionalDataExt

Accept `AbstractDimArray` input and return a `DimStack` whose dimensions describe the output
grid.

The core is deliberately array-only, in pixel coordinates, so it loads without any geospatial
dependency and works equally on map-projected imagery and radar slant-range data. This extension
is the seam: it reads coordinates off the input lookups, calls the plain-array core, and rebuilds
the result as a stack on the output grid.

# Why this is separate from the Rasters extension

This one handles the *dimensional but unprojected* case, which in practice means radar
range-Doppler: real coordinates along each axis, no CRS, and dimensions that are not `X` and `Y`
(`Dim{:range}` and `Dim{:azimuth}` are typical). Nothing here may assume otherwise — an
extension that reached for `dims(A, Y)` would work on optical imagery and fail on exactly the
data this exists to support.

Because there is no CRS, there is also no map orientation, so this path reports the core's own
`dx`/`dy` unchanged: pixel offsets from secondary back to reference, `dy` increasing with row
index. The sign flip to feature motion belongs where north is known, which is the Rasters
extension.
"""
module AutoRIFTDimensionalDataExt

import AutoRIFT
import DimensionalData as DD

using DimensionalData: AbstractDimArray, DimStack, dims, lookup, rebuild
using DimensionalData.Lookups: Lookup, isregular

"""
    autorift(reference::AbstractDimArray, secondary::AbstractDimArray; kwargs...) -> DimStack

Correlate two dimensional arrays, returning a stack on the output grid.

Layers are `dx`, `dy` (pixels), `correlation`, `peak_ratio`, `chip_size`, and `interpolated` — see
[`AutoRIFT.MultichipResult`](@ref). Accepts every keyword [`AutoRIFT.params`](@ref) does.

The first dimension is taken as rows and the second as columns, matching the array core. For a
projected raster use the Rasters extension instead, which knows which way is north and returns
map-oriented `vx`/`vy`.

!!! note "Sign convention"
    `dx` and `dy` are pixel offsets from `secondary` back to `reference` — the *negative* of
    feature motion — with `dy` increasing along the first dimension. Unprojected axes have no
    orientation to correct to, so they are reported as measured.
"""
function AutoRIFT.autorift(reference::AbstractDimArray, secondary::AbstractDimArray;
                           reference_valid = nothing, secondary_valid = nothing, kwargs...)
    check_aligned(reference, secondary)
    result, grid = AutoRIFT.autorift_with_grid(
        parent(reference), parent(secondary);
        reference_valid = unwrap(reference_valid),
        secondary_valid = unwrap(secondary_valid), kwargs...)
    return DimStack(layers(result), grid_dims(reference, grid))
end

# ---------------------------------------------------------------------------
# Shared with the Rasters extension
# ---------------------------------------------------------------------------
#
# The Rasters extension builds on this one rather than duplicating it: a `Raster` *is* an
# `AbstractDimArray`, so alignment checking, grid-dimension construction, and mask unwrapping are
# identical there. Only the CRS handling, the y-flip, and the velocity conversion differ, and
# those live in that extension because they are what it adds.
#
# Julia loads an extension only once all of its triggers are available, and Rasters depends on
# DimensionalData — so by the time the Rasters extension loads, this module exists and can be
# reached through `Base.get_extension`. These names are unprefixed for that reason: they are this
# module's small internal interface to its sibling, not private helpers.

"""
    check_aligned(a, b)

Throw unless `a` and `b` describe the same grid.

Equal *lookups*, not merely equal sizes. Two acquisitions on different grids are the likeliest
caller mistake and the most damaging: the correlation succeeds, every displacement is offset by
the grid difference, and nothing downstream can detect it. Matching sizes are not evidence of
co-registration.
"""
function check_aligned(a::AbstractDimArray, b::AbstractDimArray)
    da, db = dims(a), dims(b)
    length(da) == 2 && length(db) == 2 || throw(ArgumentError(
        "expected two 2-D arrays, got $(length(da)) and $(length(db)) dimensions. " *
        "Correlation is defined on a single image pair; slice a time series first."))
    for (x, y) in zip(da, db)
        DD.name(x) == DD.name(y) || throw(DimensionMismatch(
            "dimension names differ: $(map(DD.name, da)) and $(map(DD.name, db)). The two " *
            "images must be on a common grid."))
        lookup(x) == lookup(y) || throw(DimensionMismatch(
            "the `$(DD.name(x))` lookups differ between the two images, so they are not " *
            "co-registered. Correlating them would offset every displacement by the " *
            "difference between the grids without failing. Reproject or resample onto a " *
            "common grid first."))
    end
    return nothing
end

"""
    grid_dims(reference, grid) -> Tuple

Output dimensions: the input's, sampled at the grid's pixel positions.

The grid carries 1-based pixel indices into the input — [`AutoRIFT.gridpoints`](@ref) insets from
the edges, so the first point is not pixel 1. Indexing the input lookup at those positions is
what carries the coordinates, the step, and the axis order through to the output in one
operation, rather than recomputing an origin and spacing here and risking a half-pixel
disagreement with the input.
"""
function grid_dims(reference::AbstractDimArray, grid::AutoRIFT.PointSet{2})
    rowdim, coldim = dims(reference)
    # `grid.y` varies down rows and `grid.x` across columns, per `AutoRIFT.gridpoints`. Taking one
    # column and one row recovers each axis, since the grid is regular by construction.
    rows = @view grid.y[:, 1]
    cols = @view grid.x[1, :]
    return (rebuild(rowdim, sample_lookup(lookup(rowdim), rows)),
            rebuild(coldim, sample_lookup(lookup(coldim), cols)))
end

"""
    sample_lookup(l, positions) -> Lookup

Coordinates at the given 1-based pixel positions, as a lookup of the same kind.

`getindex` on a lookup preserves its type, its order, and its CRS, so a `Projected` input yields
a `Projected` output and a reverse-ordered axis stays reverse-ordered. That is what makes the
Rasters extension's y handling a question about *signs* rather than about metadata.

Indexed with a **range** rather than a vector of the same values, which matters more than it
looks: a vector index produces an `Irregular` span, since `getindex` cannot know the positions
were evenly spaced, and an irregular output grid has no single pixel size — which is exactly what
`dt` needs. The grid *is* regular by construction (`AutoRIFT.gridpoints` steps by `grid_spacing`),
so a range says so and carries `Regular(grid_spacing * input_step)` through.
"""
function sample_lookup(l::Lookup, positions)
    first_px = round(Int, first(positions))
    length(positions) == 1 && return l[first_px:first_px]
    step_px = round(Int, positions[2] - positions[1])
    last_px = round(Int, last(positions))
    return l[first_px:step_px:last_px]
end

"""
    layers(result) -> NamedTuple

The five output layers as plain arrays, ready to become stack layers.

`interpolated` is materialised from its `BitMatrix` into a `Matrix{Bool}`: a packed bit array is
the right internal representation but an awkward one to hand to a consumer writing NetCDF, which
expects a byte per value.
"""
layers(r::AutoRIFT.MultichipResult) = (
    dx = r.dx, dy = r.dy, correlation = r.correlation, peak_ratio = r.peak_ratio,
    chip_size = r.chip_size, interpolated = Matrix{Bool}(r.interpolated))

"""
    unwrap(mask) -> AbstractMatrix or nothing

A validity mask as a plain array. Accepts a dimensional one, since a caller holding rasters
naturally holds its masks the same way.
"""
unwrap(::Nothing) = nothing
unwrap(m::AbstractDimArray) = parent(m)
unwrap(m) = m

"""
    pixel_size(l) -> Float64 or nothing

Ground spacing along the axis `l` describes, or `nothing` if it has none.

Only a regularly-spaced axis has a single spacing. An `Irregular` or `Explicit` lookup describes
an axis whose pixels differ in size, and no scalar can convert those displacements to ground
units — so this reports that rather than returning a nominal value that would be silently wrong
across most of the scene.

The guard is what makes this total: `step` throws a `MethodError` on an irregular lookup rather
than returning anything, so the `nothing` has to come from asking first.
"""
pixel_size(l::Lookup) = isregular(l) ? abs(step(l)) : nothing

end # module
