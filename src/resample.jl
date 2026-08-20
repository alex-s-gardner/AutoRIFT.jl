# Resampling between chip-size levels, and the distance transform that dilates masks.
#
# Layer 2: depends on nothing else in the package.
#
# The multi-chip-size search moves fields between grids that differ by a power of two, and it
# uses a different interpolation for each purpose — averaging down, replicating masks,
# smoothing displacements back up. Those are not interchangeable, and the reference's choice
# in each case is deliberate:
#
#   nearest   masks and labels, where an interpolated value would be meaningless
#   area      averaging down, which is what makes a coarse displacement the mean of the
#             fine ones it covers rather than a sample of one of them
#   bicubic   smoothing a coarse displacement field back up, where the extra smoothness is
#             the point: the result is used as a prior, and a blocky prior would put visible
#             seams in the output
#
# All three are written here rather than delegated, for the same reason the window
# reductions are: the packages that provide them do not handle NaN, and a displacement field
# is NaN wherever there was no measurement.

"""
    Nearest, Area, Bicubic

Resampling methods. See [`resample`](@ref) for which is appropriate where.
"""
struct Nearest end
struct Area end
struct Bicubic end

"""
    resample(A, dstsize, method) -> Matrix

Resample `A` to `dstsize`, ignoring `NaN`.

`method` is [`Nearest`](@ref), [`Area`](@ref), or [`Bicubic`](@ref). A destination sample
with no valid contributor is `NaN`.

Coordinates use the half-sample convention: destination sample `i` maps to source position
`(i - 0.5) * scale + 0.5`, so the two grids cover the same extent and neither is offset by
half a sample relative to the other. Getting this wrong shifts a whole chip-size level by a
fraction of a grid cell, which is invisible in any single level and accumulates across them.
"""
function resample(A::AbstractMatrix, dstsize::Tuple{Int,Int}, method)
    out = similar(A, Float32, dstsize)
    return resample!(out, A, method)
end

resample(A::AbstractMatrix, dstsize, method) =
    resample(A, (Int(dstsize[1]), Int(dstsize[2])), method)

"""
    resample!(out, A, method) -> out

In-place [`resample`](@ref), to the size of `out`.
"""
function resample!(out::AbstractMatrix, A::AbstractMatrix, ::Nearest)
    sr, sc = size(A)
    dr, dc = size(out)
    ys = sr / dr
    xs = sc / dc
    @inbounds for j in 1:dc
        # `(j - 0.5) * scale` is the destination sample's centre in 0-based source
        # coordinates; flooring and adding one gives the source sample containing it.
        sj = clamp(floor(Int, (j - 0.5) * xs) + 1, 1, sc)
        for i in 1:dr
            si = clamp(floor(Int, (i - 0.5) * ys) + 1, 1, sr)
            out[i, j] = Float32(A[si, sj])
        end
    end
    return out
end

# Area averaging: each destination sample is the mean of the source samples its footprint
# covers, weighted by how much of each it overlaps. NaN contributors are skipped and the
# weights renormalised, so a partly-missing cell still yields the mean of what is there —
# which is the behaviour a displacement field needs, and what a plain weighted sum would get
# wrong by treating missing as zero.
function resample!(out::AbstractMatrix, A::AbstractMatrix, ::Area)
    sr, sc = size(A)
    dr, dc = size(out)
    ys = sr / dr
    xs = sc / dc
    @inbounds for j in 1:dc
        # Source interval this destination column covers, in continuous coordinates.
        x0 = (j - 1) * xs
        x1 = j * xs
        j0 = max(floor(Int, x0) + 1, 1)
        j1 = min(ceil(Int, x1), sc)
        for i in 1:dr
            y0 = (i - 1) * ys
            y1 = i * ys
            i0 = max(floor(Int, y0) + 1, 1)
            i1 = min(ceil(Int, y1), sr)
            acc = 0.0
            wsum = 0.0
            for jj in j0:j1
                # Overlap of source column `jj` with the destination footprint.
                wj = min(x1, Float64(jj)) - max(x0, Float64(jj - 1))
                wj <= 0 && continue
                for ii in i0:i1
                    wi = min(y1, Float64(ii)) - max(y0, Float64(ii - 1))
                    wi <= 0 && continue
                    v = Float64(A[ii, jj])
                    isnan(v) && continue
                    w = wi * wj
                    acc += w * v
                    wsum += w
                end
            end
            out[i, j] = wsum > 0 ? Float32(acc / wsum) : NaN32
        end
    end
    return out
end

# Catmull-Rom bicubic, the `a = -0.5` member of the cubic family. OpenCV uses `a = -0.75`;
# the difference is a slightly different overshoot at edges, and it matters here only as a
# smoothness choice rather than a fidelity one, because the result feeds the *prior* for the
# next level rather than the output. Catmull-Rom is chosen for interpolating exactly through
# the source samples, which keeps a coarse estimate unchanged where it was already known.
#
# NaN handling is what makes this longer than a textbook bicubic: any of the sixteen taps may
# be missing, so the weights are renormalised over those present. With more than half
# missing the estimate is not supportable and the result is NaN — otherwise a single valid
# corner would be smeared across a whole empty region.
function resample!(out::AbstractMatrix, A::AbstractMatrix, ::Bicubic)
    sr, sc = size(A)
    dr, dc = size(out)
    ys = sr / dr
    xs = sc / dc
    wx = MVector4()
    wy = MVector4()
    @inbounds for j in 1:dc
        x = (j - 0.5) * xs - 0.5
        j0 = floor(Int, x)
        _cubic_weights!(wx, x - j0)
        for i in 1:dr
            y = (i - 0.5) * ys - 0.5
            i0 = floor(Int, y)
            _cubic_weights!(wy, y - i0)
            acc = 0.0
            wsum = 0.0
            for dj in 0:3
                sj = clamp(j0 + dj, 1, sc)
                for di in 0:3
                    si = clamp(i0 + di, 1, sr)
                    v = Float64(A[si, sj])
                    isnan(v) && continue
                    w = wy[di + 1] * wx[dj + 1]
                    acc += w * v
                    wsum += w
                end
            end
            # The weights sum to 1 when all sixteen taps are present. Requiring at least
            # half the weight keeps a mostly-missing neighbourhood from producing a
            # confident-looking value.
            out[i, j] = wsum >= 0.5 ? Float32(acc / wsum) : NaN32
        end
    end
    return out
end

# A four-element mutable buffer, so the weights are computed once per row and column rather
# than once per output sample. `MVector4` rather than a `Vector` to keep it stack-allocated
# without adding a StaticArrays dependency for four floats.
mutable struct MVector4
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    MVector4() = new(0.0, 0.0, 0.0, 0.0)
end

@inline function Base.getindex(v::MVector4, i::Int)
    i == 1 && return v.a
    i == 2 && return v.b
    i == 3 && return v.c
    return v.d
end

# Catmull-Rom basis at fractional offset `t`, for taps at -1, 0, +1, +2.
@inline function _cubic_weights!(w::MVector4, t::Float64)
    t2 = t * t
    t3 = t2 * t
    w.a = -0.5t3 + t2 - 0.5t
    w.b = 1.5t3 - 2.5t2 + 1.0
    w.c = -1.5t3 + 2.0t2 + 0.5t
    w.d = 0.5t3 - 0.5t2
    return w
end

# ---------------------------------------------------------------------------
# Distance transform
# ---------------------------------------------------------------------------

"""
    dilate_within(mask, radius) -> BitMatrix

Every position within Euclidean `radius` of a `true` in `mask`.

Used to grow the coarse pass's validity mask before it restricts the fine search: a coarse
estimate is evidence that the neighbourhood is worth searching, not only that one point.
The radius is in grid cells and is a genuine Euclidean distance rather than a chessboard
one, which matters at the radii in use — a chessboard ball of radius 8 is 40%
larger in area than a Euclidean one.

Exact, via the two-pass squared-distance transform of Felzenszwalb & Huttenlocher (2012):
separable, O(n) per row and column, and not an approximation like the chamfer masks that
image libraries often default to.
"""
function dilate_within(mask::AbstractMatrix{Bool}, radius::Real)
    d2 = _squared_distance_transform(mask)
    r2 = Float64(radius)^2
    out = BitMatrix(undef, size(mask))
    @inbounds for i in eachindex(out)
        out[i] = d2[i] <= r2
    end
    return out
end

"""
    _squared_distance_transform(mask) -> Matrix{Float64}

Squared Euclidean distance from each position to the nearest `true` in `mask`.

`Inf` where `mask` is empty. Separable: a 1-D transform down each column, then the lower
envelope of parabolas along each row. The second step is the Felzenszwalb–Huttenlocher
construction, which is what makes it exact and linear rather than an iterative
approximation.
"""
function _squared_distance_transform(mask::AbstractMatrix{Bool})
    nr, nc = size(mask)
    d2 = Matrix{Float64}(undef, nr, nc)

    # Pass 1: within each column, distance to the nearest set element in that column.
    @inbounds for j in 1:nc
        # Forward sweep.
        run = Inf
        for i in 1:nr
            run = mask[i, j] ? 0.0 : run + 1.0
            d2[i, j] = run
        end
        # Backward sweep, taking whichever direction is closer.
        run = Inf
        for i in nr:-1:1
            run = mask[i, j] ? 0.0 : run + 1.0
            d2[i, j] = min(d2[i, j], run)
        end
        for i in 1:nr
            v = d2[i, j]
            d2[i, j] = isinf(v) ? Inf : v * v
        end
    end

    # Pass 2: along each row, the lower envelope of the parabolas f(x) = d2[x] + (j - x)^2.
    # `v` holds the parabola vertices in the envelope and `z` the boundaries between them.
    f = Vector{Float64}(undef, nc)
    dd = Vector{Float64}(undef, nc)
    v = Vector{Int}(undef, nc)
    z = Vector{Float64}(undef, nc + 1)
    @inbounds for i in 1:nr
        for j in 1:nc
            f[j] = d2[i, j]
        end
        _lower_envelope!(dd, f, v, z, nc)
        for j in 1:nc
            d2[i, j] = dd[j]
        end
    end
    return d2
end

# One row of the Felzenszwalb–Huttenlocher transform: given `f`, compute
# `dd[j] = min over x of f[x] + (j - x)^2`.
function _lower_envelope!(dd, f, v, z, n::Int)
    # An all-Inf row has no parabola to envelope, and the arithmetic below would produce
    # NaN boundaries rather than propagating Inf.
    if all(isinf, view(f, 1:n))
        @inbounds for j in 1:n
            dd[j] = Inf
        end
        return dd
    end

    k = 1
    @inbounds begin
        v[1] = 1
        z[1] = -Inf
        z[2] = Inf
        for q in 2:n
            isinf(f[q]) && continue
            # Intersection of parabola q with the one currently topmost; pop while it is
            # left of that parabola's existing region.
            s = 0.0
            while true
                p = v[k]
                s = ((f[q] + q * q) - (f[p] + p * p)) / (2q - 2p)
                s > z[k] && break
                k -= 1
                k == 0 && break
            end
            if k == 0
                k = 1
                v[1] = q
                z[1] = -Inf
                z[2] = Inf
            else
                k += 1
                v[k] = q
                z[k] = s
                z[k + 1] = Inf
            end
        end

        k = 1
        for q in 1:n
            while z[k + 1] < q
                k += 1
            end
            p = v[k]
            dd[q] = (q - p)^2 + f[p]
        end
    end
    return dd
end
