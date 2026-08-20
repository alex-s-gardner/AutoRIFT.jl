# Outlier rejection on a displacement field.
#
# Layer 2: depends on `window.jl` and nothing else in the package, so this file is
# potentially useful outside AutoRIFT and is written to stay separable.
#
# The problem it solves: correlation returns a displacement at every searched point,
# including points where the peak was noise. Those false matches are not small errors,
# they are arbitrary vectors, and a single one can dominate any downstream fit. What
# distinguishes them from real motion is that real motion is spatially coherent —
# neighbouring points on the same glacier move similarly — while a false match agrees
# with nothing around it.
#
# Which test asks that question is a choice, so it dispatches on an `OutlierMethod`
# (see `types.jl`). AutoRIFT's own is `GardnerFilter`; the PIV literature's universal
# outlier detection is a candidate for a second, and would differ enough to be worth
# having rather than being a parameter tweak of the first.

"""
    AutoRIFT.outlier_filter(; kwargs...) -> GardnerFilter

Build a [`GardnerFilter`](@ref), AutoRIFT's default outlier method.

A spelling that names the job rather than the author, kept because most callers want the
default filter and not a choice between filters. Takes `GardnerFilter`'s keywords.
"""
outlier_filter(; kwargs...) = GardnerFilter(; kwargs...)

"""
    reject_outliers(dx, dy, radius_x, radius_y, valid, upsampling, method) -> BitMatrix

Which displacements to keep, per `method` — an [`OutlierMethod`](@ref).

`dx` and `dy` are displacement fields in pixels, `radius_x`/`radius_y` the per-point
search radii they were found within, `valid` the points to consider at all, and
`upsampling` the subpixel refinement factor (which sets the quantization floor). Returns a
fresh mask; the inputs are not modified.
"""
function reject_outliers(
    dx::AbstractMatrix, dy::AbstractMatrix,
    radius_x::AbstractMatrix, radius_y::AbstractMatrix,
    valid::AbstractMatrix{Bool}, upsampling::Integer, method::OutlierMethod,
)
    axes(dx) == axes(dy) == axes(radius_x) == axes(radius_y) == axes(valid) ||
        throw(DimensionMismatch(
            "all inputs must share axes: got dx $(axes(dx)), dy $(axes(dy)), " *
            "radius_x $(axes(radius_x)), radius_y $(axes(radius_y)), " *
            "valid $(axes(valid))"))
    return _reject(dx, dy, radius_x, radius_y, valid, upsampling, method)
end

# Nothing to reject. Still a fresh copy, so callers may mutate the result either way.
_reject(dx, _dy, _rx, _ry, valid, _up, ::NoOutlierFilter) =
    copyto!(BitMatrix(undef, size(dx)), valid)

"""
    _reject(..., f::GardnerFilter)

Two stages, run in sequence, both from autoRIFT's `DISP_FILT`:

  1. **Agreement.** Keep a point only if enough of its neighbours have a similar
     displacement. Cheap, and removes isolated wild vectors.

  2. **Median deviation.** Keep a point only if it lies within a few median absolute
     deviations of its neighbourhood median. Catches the subtler case of a false match
     that happens to sit near other false matches.

A point is kept only if it passes both, and rejection is monotone across iterations: once
a point is out it stays out, so removing one outlier can expose its neighbours in the next
pass.

!!! note "Why displacements are normalized by search radius"
    Both stages divide by the local search radius before comparing, but they do not
    both depend on it.

    Stage 1's threshold is `agree_tolerance`, a fixed fraction of the radius, so
    normalising is what gives it meaning: a five-pixel disagreement is negligible
    where the search reached fifty pixels and decisive where it reached six, and the
    radius varies across a scene because it comes from an a-priori velocity field.

    Stage 2 is radius-*invariant* by construction. Its tolerance is `mad_scale` times
    the neighbourhood MAD, and both the deviation and the MAD scale as one over the
    radius, so the ratio is unchanged. Normalising there buys consistency of units
    rather than of behaviour — the one place it changes the verdict is the
    quantization floor, which is an absolute subpixel step and therefore *does* scale
    against the normalized deviation.

See [`GardnerFilter`](@ref) for how this differs from the normalized median test of
Westerweel & Scarano (2005), which it resembles but is not.
"""
function _reject(
    dx::AbstractMatrix, dy::AbstractMatrix,
    radius_x::AbstractMatrix, radius_y::AbstractMatrix,
    valid::AbstractMatrix{Bool}, upsampling::Integer, f::GardnerFilter,
)
    # Work on normalized copies. The reference mutates its arguments and relies on
    # every caller passing a copy; copying here instead makes the function safe to
    # call directly and costs one allocation per invocation rather than per level.
    nx = similar(dx, Float32)
    ny = similar(dy, Float32)
    @inbounds for i in eachindex(nx)
        rx, ry = radius_x[i], radius_y[i]
        # A zero radius means the point was never searched. Division would give Inf
        # or NaN; NaN is the honest value and is what the reductions already skip.
        nx[i] = rx > 0 ? Float32(dx[i] / rx) : NaN32
        ny[i] = ry > 0 ? Float32(dy[i] / ry) : NaN32
    end

    keep = BitMatrix(undef, size(dx))
    copyto!(keep, valid)

    # Stage 1: neighbourhood agreement.
    #
    # `min_agree_fraction` is a fraction of the *full* window area, so the required
    # count is fixed rather than adjusted for truncated windows at the border. That is
    # the reference's behaviour and it is defensible: a point at the edge of the grid
    # has fewer neighbours to corroborate it and correspondingly less evidence, so
    # holding it to the same absolute count is a deliberate conservatism.
    needed = Float32(f.min_agree_fraction * f.window^2)
    tol = Float32(f.agree_tolerance)
    for _ in 1:f.iterations
        _mask_to_nan!(nx, ny, keep)
        ax = count_agreeing(nx, f.window, tol)
        ay = count_agreeing(ny, f.window, tol)
        @inbounds for i in eachindex(keep)
            keep[i] = ax[i] >= needed && ay[i] >= needed
        end
    end

    # Stage 2: distance from the neighbourhood median, scaled by its MAD.
    #
    # The floor on the tolerance is what makes this work where a neighbourhood is
    # nearly uniform. There the MAD collapses toward zero and `mad_scale * MAD` would
    # reject points differing only by the subpixel quantization step — so the
    # tolerance is floored at twice that step, `2 / (upsampling * radius)`, expressed
    # in the same normalized units. Without it, the most coherent parts of the field
    # would be rejected the most aggressively.
    #
    # One fewer iteration than stage 1, per the reference, with a minimum of one.
    n2 = max(f.iterations - 1, 1)
    scale = Float64(f.mad_scale)
    for _ in 1:n2
        _mask_to_nan!(nx, ny, keep)
        # These two calls are 95% of this function's runtime, which is why the median
        # and MAD are fused rather than taken separately.
        medx, madx = windowmedmad(nx, f.window)
        medy, mady = windowmedmad(ny, f.window)
        @inbounds for i in eachindex(keep)
            keep[i] || continue
            rx, ry = radius_x[i], radius_y[i]
            floor_x = rx > 0 ? 2.0 / (upsampling * rx) : Inf
            floor_y = ry > 0 ? 2.0 / (upsampling * ry) : Inf
            tolx = max(scale * madx[i], floor_x)
            toly = max(scale * mady[i], floor_y)
            # A NaN median means no valid neighbour, so there is nothing to be
            # consistent with and the comparison is false — the point is dropped.
            keep[i] = abs(nx[i] - medx[i]) <= tolx && abs(ny[i] - medy[i]) <= toly
        end
    end
    return keep
end

# Invalidate rejected points so the window reductions skip them. This is why the
# reductions have to ignore NaN rather than propagate it: after the first iteration
# most of the field may be NaN, and a propagating reduction would erase the rest.
@inline function _mask_to_nan!(nx, ny, keep)
    @inbounds for i in eachindex(nx)
        if !keep[i]
            nx[i] = NaN32
            ny[i] = NaN32
        end
    end
    return nothing
end
