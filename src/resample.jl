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
#   bilinear  averaging down a search radius, whose fractional part decides the answer because
#             the result passes through `ceil`
#   bicubic   smoothing a coarse displacement field back up, where the extra smoothness is
#             the point: the result is used as a prior, and a blocky prior would put visible
#             seams in the output
#
# All three are written here rather than delegated, for the same reason the window
# reductions are: the packages that provide them do not handle NaN, and a displacement field
# is NaN wherever there was no measurement.

"""
    Nearest()

Nearest-neighbor resampling: each destination sample takes the value of the source sample containing
its center. Use it for a mask, where an interpolated value between two categories is meaningless. See
[`resample`](@ref).
"""
struct Nearest end

"""
    Area()

Area-averaging resampling: each destination sample is the mean of the source samples its footprint
covers, weighted by overlap and ignoring `NaN`. Use it to decimate a measured field, where a point
sample would discard most of the data. See [`resample`](@ref).
"""
struct Area end

"""
    Bilinear()

Bilinear resampling: the weighted mean of the surrounding 2-by-2 neighborhood, ignoring `NaN`. Use it
to decimate a quantity whose fractional part matters, such as one about to be rounded or taken to a
ceiling. See [`resample`](@ref).
"""
struct Bilinear end

"""
    Bicubic()

Bicubic resampling: a smooth interpolant over the surrounding 4-by-4 neighborhood. Use it to upsample
a displacement field into a prior, where nearest-neighbor would leave visible seams. See
[`resample`](@ref).
"""
struct Bicubic end

"""
    resample(A, dstsize, method) -> Matrix

Resample `A` to `dstsize`, ignoring `NaN`.

`method` is [`Nearest`](@ref), [`Area`](@ref), [`Bilinear`](@ref), or [`Bicubic`](@ref). A
destination sample with no valid contributor is `NaN`.

Coordinates use the half-sample convention: destination sample `i` maps to source position
`(i - 0.5) * scale + 0.5`, so the two grids cover the same extent with no half-sample offset.

`scale` is the source samples per destination sample, per axis, and defaults to the size ratio. Pass
it when the sizes do not determine the correspondence. A source taken as every `stride`-th sample of
the destination has a scale of exactly `1/stride`, but the size ratio `cld(n, stride) / n` equals that
only when `stride` divides `n`; otherwise it drifts, reaching a full cell of offset at the far edge.
"""
function resample(A::AbstractMatrix, dstsize::Tuple{Int,Int}, method;
                  scale::Tuple{Real,Real} = (size(A, 1) / dstsize[1], size(A, 2) / dstsize[2]))
    out = similar(A, Float32, dstsize)
    return resample!(out, A, method; scale)
end

resample(A::AbstractMatrix, dstsize, method; kw...) =
    resample(A, (Int(dstsize[1]), Int(dstsize[2])), method; kw...)

# Every interpolant below reads only the source and writes one output sample per iteration, so the
# split over output columns changes nothing about the result.
"""
    resample!(out, A, method; scale) -> out

In-place [`resample`](@ref), to the size of `out`.
"""
function resample!(out::AbstractMatrix, A::AbstractMatrix, ::Nearest;
                   scale::Tuple{Real,Real} = (size(A, 1) / size(out, 1),
                                              size(A, 2) / size(out, 2)))
    sr, sc = size(A)
    dr, dc = size(out)
    ys, xs = Float64(scale[1]), Float64(scale[2])
    _parallel_slices(1:dc, dr * dc) do cols
        @inbounds for j in cols
            # `(j - 0.5) * scale` is the destination sample's centre in 0-based source
            # coordinates; flooring and adding one gives the source sample containing it.
            sj = clamp(floor(Int, (j - 0.5) * xs) + 1, 1, sc)
            for i in 1:dr
                si = clamp(floor(Int, (i - 0.5) * ys) + 1, 1, sr)
                out[i, j] = Float32(A[si, sj])
            end
        end
    end
    return out
end

# Area averaging: each destination sample is the mean of the source samples its footprint
# covers, weighted by how much of each it overlaps. NaN contributors are skipped and the
# weights renormalised, so a partly-missing cell still yields the mean of what is there —
# which is the behaviour a displacement field needs, and what a plain weighted sum would get
# wrong by treating missing as zero.
function resample!(out::AbstractMatrix, A::AbstractMatrix, ::Area;
                   scale::Tuple{Real,Real} = (size(A, 1) / size(out, 1),
                                              size(A, 2) / size(out, 2)))
    sr, sc = size(A)
    dr, dc = size(out)
    ys, xs = Float64(scale[1]), Float64(scale[2])
    _parallel_slices(1:dc, dr * dc * (ys + 1) * (xs + 1)) do cols
        @inbounds for j in cols
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
    end
    return out
end

# Destination sample `i` placed in the source, under the half-sample convention `resample` documents,
# split into the sample below it and the fraction past that sample. The two interpolating kernels both
# need exactly this, so the convention is written once here rather than re-derived in each.
@inline function _src_frac(i::Integer, scale::Real)
    t = (i - 0.5) * scale - 0.5
    i0 = floor(Int, t)
    return i0, t - i0
end

# Bilinear interpolation over the four samples surrounding the destination centre, with the weights
# renormalised over those that are not `NaN`. Requiring half the weight rather than all of it keeps a
# radius defined on three corners of a cell from collapsing to no radius at all; below that the
# estimate rests on one corner and is not supportable.
#
# `cv2.resize`'s default, and so the reference's for the per-level search radius
# (`autoRIFT.py:580-581`), where the result passes through `ceil`.
#
# **No `@inbounds`, deliberately, where the three kernels around it have one.** Every index here is
# `clamp`ed before use, so the annotation would be safe — but safe is not the bar. Its only effect is to
# remove a bounds check, and on the one path that calls this the whole resize is 0.002% of a run, so
# there is no cost to remove. An `@inbounds` whose benefit has not been measured is silent undefined
# behaviour bought for nothing; add one here only alongside a benchmark that shows the check mattering.
function resample!(out::AbstractMatrix, A::AbstractMatrix, ::Bilinear;
                   scale::Tuple{Real,Real} = (size(A, 1) / size(out, 1),
                                              size(A, 2) / size(out, 2)))
    sr, sc = size(A)
    dr, dc = size(out)
    ys, xs = Float64(scale[1]), Float64(scale[2])
    _parallel_slices(1:dc, 4 * dr * dc) do cols
        for j in cols
            j0, wx = _src_frac(j, xs)
            ja = clamp(j0 + 1, 1, sc)
            jb = clamp(j0 + 2, 1, sc)
            for i in 1:dr
                i0, wy = _src_frac(i, ys)
                ia = clamp(i0 + 1, 1, sr)
                ib = clamp(i0 + 2, 1, sr)
                acc = 0.0
                wsum = 0.0
                for (si, wi) in ((ia, 1 - wy), (ib, wy)), (sj, wj) in ((ja, 1 - wx), (jb, wx))
                    v = Float64(A[si, sj])
                    isnan(v) && continue
                    w = wi * wj
                    acc += w * v
                    wsum += w
                end
                out[i, j] = wsum >= 0.5 ? Float32(acc / wsum) : NaN32
            end
        end
    end
    return out
end

# Bicubic convolution with `a = -0.75`, which is `cv2.INTER_CUBIC` and so the reference's choice
# (`autoRIFT.py:846-847`). Interpolates exactly through the source samples, keeping a coarse estimate
# unchanged where it was already known.
#
# The parameter is a fidelity choice here, not a smoothness preference, because a decimated
# chip-size level's interpolated values *are* the output at every point the level owns —
# `_undecimate_level` writes them into the result. Catmull-Rom (`a = -0.5`) agrees with this at every
# node and at each cell's centre, and departs from it antisymmetrically in between: opposite signs at
# a quarter and three quarters across a cell, scaling with the field's local curvature. Across an
# 8 px step that is ±0.19 px, which is the size of the disagreement it produced along an iceberg
# margin — visible as a one-cell line of alternating sign on the coarse levels alone, and exactly zero
# wherever the finest level answered and no interpolation happened.
#
# NaN handling is what makes this longer than a textbook bicubic: any of the sixteen taps may
# be missing, so the weights are renormalised over those present. With more than half
# missing the estimate is not supportable and the result is NaN — otherwise a single valid
# corner would be smeared across a whole empty region.
function resample!(out::AbstractMatrix, A::AbstractMatrix, ::Bicubic;
                   scale::Tuple{Real,Real} = (size(A, 1) / size(out, 1),
                                              size(A, 2) / size(out, 2)))
    sr, sc = size(A)
    dr, dc = size(out)
    ys, xs = Float64(scale[1]), Float64(scale[2])
    _parallel_slices(1:dc, 16 * dr * dc) do cols
        wx = MVector4()
        wy = MVector4()
        @inbounds for j in cols
            j0, fx = _src_frac(j, xs)
            _cubic_weights!(wx, fx)
            for i in 1:dr
                i0, fy = _src_frac(i, ys)
                _cubic_weights!(wy, fy)
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

# Cubic convolution basis at fractional offset `t`, for taps at -1, 0, +1, +2, with `a = -0.75` —
# OpenCV's `interpolateCubic`, expanded. The outer taps are the same polynomial evaluated at `t + 1`
# and `2 - t`, and the inner two are its `|s| <= 1` branch at `t` and `1 - t`; writing them out keeps
# this to a handful of multiplies per sample.
@inline function _cubic_weights!(w::MVector4, t::Float64)
    a = -0.75
    t2 = t * t
    t3 = t2 * t
    u = 1.0 - t
    w.a = a * (t3 - 2t2 + t)
    w.b = (a + 2.0) * t3 - (a + 3.0) * t2 + 1.0
    w.c = (a + 2.0) * (u * u * u) - (a + 3.0) * (u * u) + 1.0
    w.d = a * (u * u * u - 2.0 * (u * u) + u)
    return w
end

# ---------------------------------------------------------------------------
# Distance transform
# ---------------------------------------------------------------------------

"""
    dilate_within(mask, radius) -> BitMatrix

Every position strictly closer than Euclidean `radius` to a `true` in `mask`.

Used to grow the coarse pass's validity mask before it restricts the fine search. `radius` is in grid
cells and the distance is Euclidean, not chessboard (a chessboard ball of radius 8 has 40% more area).

The comparison is strict, matching the reference's `distance_transform_edt(!MC) < BuffDistanceC`
(`autoRIFT.py:709`). At the default radius of 8, positions at distance exactly 8 are the
axis-aligned ones eight cells out. Admitting them widens the searched region by 72 coarse cells on
the golden Landsat case, which is 30,084 fine points the reference leaves unsearched.

The distance is exact, computed by the two-pass squared-distance transform of Felzenszwalb &
Huttenlocher (2012), which is separable and O(n) per row and column.
"""
function dilate_within(mask::AbstractMatrix{Bool}, radius::Real)
    d2 = _squared_distance_transform(mask)
    r2 = Float64(radius)^2
    out = BitMatrix(undef, size(mask))
    @inbounds for i in eachindex(out)
        out[i] = d2[i] < r2
    end
    return out
end

"""
    small_components(mask, minsize) -> BitMatrix

Positions in a connected component of `mask` holding fewer than `minsize` `true`s.

Components are eight-connected, so two positions touching only at a corner belong to one component,
as in the reference's `connectivity=2` (`autoRIFT.py:1652`). Under four-connectivity a diagonal pair
of holes would be two components of one position, and the size test would give a different answer.

This selects which holes in a displacement field are interpolated across, regardless of shape. An
L-shaped four-position hole, for example, has positions with too few neighbors to pass a neighbor
count.

`minsize` is exclusive, matching `bwareaopen(image, size1)`'s `size < size1`: at `minsize = 5`
components of one to four positions are returned and a five-position component is not.

One union-find pass over the array (union by size, full path compression), so the cost is effectively
linear.
"""
function small_components(mask::AbstractMatrix{Bool}, minsize::Integer)
    Base.require_one_based_indexing(mask)
    out = falses(size(mask))
    minsize <= 1 && return out          # nothing can hold fewer than one position

    nr, nc = size(mask)
    parent = collect(1:length(mask))
    csize = ones(Int, length(mask))

    # Path-halving find. Iterative rather than recursive: a component can span the array, and
    # the recursion depth would follow it.
    function root(a::Int)
        while parent[a] != a
            parent[a] = parent[parent[a]]
            a = parent[a]
        end
        return a
    end

    lin = LinearIndices(mask)
    @inbounds for j in 1:nc, i in 1:nr
        mask[i, j] || continue
        a = root(lin[i, j])
        # Only the four already-visited neighbours under column-major order — up, and the three
        # in the previous column. Their mirror images are unioned when this position is itself
        # the neighbour, so scanning half the ring covers all eight links.
        for (di, dj) in ((-1, 0), (-1, -1), (0, -1), (1, -1))
            ii, jj = i + di, j + dj
            (1 <= ii <= nr && 1 <= jj <= nc) || continue
            mask[ii, jj] || continue
            b = root(lin[ii, jj])
            a == b && continue
            # Union by size, which is what bounds the tree depth.
            csize[a] < csize[b] && ((a, b) = (b, a))
            parent[b] = a
            csize[a] += csize[b]
        end
    end

    @inbounds for i in eachindex(mask)
        mask[i] && (out[i] = csize[root(i)] < minsize)
    end
    return out
end

"""
    _squared_distance_transform(mask) -> Matrix{Float64}

Squared Euclidean distance from each position to the nearest `true` in `mask`.

`Inf` where `mask` is empty. Separable: a 1-D transform down each column, then the lower envelope of
parabolas along each row (Felzenszwalb and Huttenlocher), which is exact and linear.
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
