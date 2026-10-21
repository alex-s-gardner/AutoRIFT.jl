"""
    AutoRIFTDimensionalDataExt

Accept `AbstractDimArray` input and return a `DimStack` whose dimensions describe the output grid.

The extension reads coordinates from the input lookups, calls the plain-array core, and rebuilds the
result as a stack on the output grid.

It handles dimensional arrays without a CRS, such as radar range-Doppler data, whose dimensions need
not be `X` and `Y` (`Dim{:range}` and `Dim{:azimuth}` are typical). The first dimension is rows and the
second is columns. With no CRS there is no map orientation, so it reports the core's `dx` and `dy`
unchanged: pixel offsets from secondary back to reference, `dy` increasing with row index. The
Rasters extension converts these to feature motion.
"""
module AutoRIFTDimensionalDataExt

import AutoRIFT
import DimensionalData as DD

using DimensionalData: AbstractDimArray, DimStack, dims, lookup, rebuild
using DimensionalData.Lookups: Lookup, isregular

"""
    autorift(reference::AbstractDimArray, secondary::AbstractDimArray; kwargs...) -> DimStack

Correlate two dimensional arrays, returning a stack on the output grid.

Layers are `dx`, `dy` (pixels), `correlation`, `peak_ratio`, `chip_size` and `interpolated`; see
[`AutoRIFT.MultichipResult`](@ref). Accepts the keywords of [`AutoRIFT.params`](@ref), plus
`reference_valid` and `secondary_valid` (per-pixel validity masks, arrays or dimensional arrays) and
the blocking keywords of [`AutoRIFT.autorift`](@ref). Throws `ArgumentError` unless both inputs are
2-D, and `DimensionMismatch` unless their dimension names and lookups are equal.

The first dimension is rows and the second is columns. For a projected raster, load `Rasters` for
map-oriented `vx` and `vy`.

!!! note "Sign convention"
    `dx` and `dy` are pixel offsets from `secondary` back to `reference`, the negative of feature
    motion, with `dy` increasing along the first dimension.
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

Both must be 2-D with equal dimension names and equal lookups; equal sizes are not sufficient. Images
on different grids would correlate without error and give displacements offset by the grid
difference. Throws `ArgumentError` for other than two dimensions and `DimensionMismatch` for differing
names or lookups.
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

The grid holds 1-based pixel indices into the input; [`AutoRIFT.gridpoints`](@ref) insets from the
edges, so the first point is not pixel 1. Indexing the input lookup at those positions carries the
coordinates, step and axis order to the output.
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

`getindex` on a lookup preserves its type, order and CRS, so a `Projected` input gives a `Projected`
output and a reverse-ordered axis stays reverse-ordered.

The positions are indexed as a range, not a vector. A vector index gives an `Irregular` span, which has
no single pixel size for `dt` to scale by. The grid is regular (`AutoRIFT.gridpoints` steps by
`grid_spacing`), so a range gives `Regular(grid_spacing * input_step)`.
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

The six output layers (`dx`, `dy`, `correlation`, `peak_ratio`, `chip_size`, `interpolated`) as plain
arrays. `interpolated` is converted from a `BitMatrix` to a `Matrix{Bool}`, one byte per value, as
NetCDF writers expect.
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

Only a regularly spaced axis has a single spacing. An `Irregular` or `Explicit` lookup has pixels of
differing size, so no scalar converts its displacements to ground units and this returns `nothing`.
"""
pixel_size(l::Lookup) = isregular(l) ? abs(step(l)) : nothing

end # module
