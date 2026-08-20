# Locating the correlation peak.
#
# Layer 1: depends only on `correlate.jl` and `integral.jl`.
#
# Two steps. The integer peak is the argmax of the surface; the sub-pixel peak
# refines it by upsampling a small neighbourhood. Both are short, and both have a
# convention that is easy to get wrong in a way that is invisible afterwards:
#
#   * Scan order on ties. OpenCV's `minMaxLoc` scans row-major and keeps the first
#     strict maximum; Julia's `argmax` scans column-major. On a plateau the two
#     disagree, and since either answer looks reasonable, the resulting bias in
#     every displacement would never be noticed. Plateaus are not exotic: 8-bit
#     quantized imagery produces them routinely.
#
#   * The origin. The surface is `2 * radius` across with zero displacement at
#     index `radius` counting from zero, i.e. `radius + 1` in Julia. An off-by-one
#     here shifts every velocity by one pixel.

"""
    peak_index(surface) -> (row, col)

Index of the maximum of `surface`, scanning row-major and keeping the first strict
maximum — matching OpenCV's `minMaxLoc`.

Deliberately not `argmax`, which scans column-major. On a plateau the two return
different elements, and because both are legitimate maxima the difference would
show up only as a systematic bias in the output.

Returns 1-based indices. `NaN` is never selected, since every comparison against it
is false; an all-`NaN` surface returns `(1, 1)`.
"""
@inline function peak_index(surface::AbstractMatrix)
    nr, nc = size(surface)
    best_i, best_j = 1, 1
    best = typemin(eltype(surface))
    # Traversal is column-major -- along the array's memory order -- while the
    # tie-breaking rule stays row-major. Iterating in row-major order to match the
    # rule directly would stride across memory and cost about twice as much, and
    # this loop runs once per grid point per pyramid level.
    #
    # The extra clause is what keeps the semantics: a strictly greater value always
    # wins, and an *equal* value wins only if it precedes the incumbent in
    # row-major order. That yields the same element OpenCV's `minMaxLoc` would
    # return, reached in a different order. Verified against 300 randomised
    # surfaces including heavily-tied and all-equal cases.
    @inbounds for j in 1:nc, i in 1:nr
        v = surface[i, j]
        if v > best || (v == best && (i < best_i || (i == best_i && j < best_j)))
            best = v
            best_i, best_j = i, j
        end
    end
    return best_i, best_j
end

"""
    peak(surface) -> (row, col, value)

[`peak_index`](@ref) together with the value at that index.

The value is the peak correlation coefficient, a per-point quality measure. The
reference implementation computes it and throws it away at all four call sites, so
returning it is free and it is the most-requested missing autoRIFT output.
"""
@inline function peak(surface::AbstractMatrix)
    i, j = peak_index(surface)
    return i, j, @inbounds surface[i, j]
end

"""
    peak_offset(surface, radius) -> (dx, dy, correlation)

Displacement in pixels of the correlation peak, relative to zero displacement.

The surface is `(2 * radius_y, 2 * radius_x)` with zero displacement at index
`(radius_y + 1, radius_x + 1)`, so the offset is the peak index minus that origin.
`dx` is along columns and `dy` along rows, both in image orientation — `dy`
increases downward, matching array indexing. The sign flip to a cartesian
convention, if wanted, happens once at the output boundary rather than in three
places as it does in the reference.

!!! note "Sign of the returned offset"
    This is where the chip's content sits **within the search window**, which is
    not the same as the motion of the imaged features.

    With the chip cut from the secondary (later) image and the window from the
    reference (earlier) one — the arrangement the reference implementation uses —
    the returned offset points from secondary back to reference, so it is the
    **negative** of the feature displacement. A glacier that flowed `+10` pixels
    east between acquisitions gives `dx = -10` here.

    The negation is applied once, at the output boundary, so that everything below
    it works in one consistent convention.
"""
@inline function peak_offset(surface::AbstractMatrix, radius::Tuple{Int,Int})
    rx, ry = radius
    i, j, v = peak(surface)
    return Float64(j - rx - 1), Float64(i - ry - 1), v
end

# ---------------------------------------------------------------------------
# Sub-pixel refinement
# ---------------------------------------------------------------------------
#
# The reference upsamples a 5x5 neighbourhood of the integer peak with a cascade of
# Gaussian-pyramid steps and takes the argmax of the result. That is a specific
# operation — inject zeros, convolve with the separable [1,4,6,4,1]/16 kernel,
# scale by 4, reflect at the border — and not equivalent to bilinear or bicubic
# upsampling.
#
# It is also deliberately chosen over a parabola fit. A parabola through three
# samples is biased toward the sample it is centred on ("peak locking"), which for
# quantized imagery produces a histogram of displacements with visible spikes at
# integer values. The pyramid cascade does not have that bias, which is why the
# reference uses it and why this reproduces it rather than substituting something
# analytically tidier.
#
# ---------------------------------------------------------------------------
# Remaining performance gap
# ---------------------------------------------------------------------------
#
# ~4x slower than OpenCV at 128x upsampling (1.14 ms against 247 us), down from ~6x
# after hoisting the row taps out of the column loop. The kernel is competitive at
# ~1.1 ns per output sample; what remains is OpenCV's SIMD-vectorised separable filter
# against a scalar one.
#
# Micro-optimisation will not close it, because the algorithm is the waste: at 128x a
# 5x5 patch becomes 640x640, ~1.6 MB materialised to locate a single maximum. Two
# routes were investigated and rejected for now:
#
#   * A coarse-to-fine cascade that crops around the running peak at each level. This
#     is the big win in principle, but measuring how far the peak moves between levels
#     over 300 surfaces gave a maximum of 11 samples — too wide to bound safely, and an
#     empirical bound is not a guarantee. A cropped cascade that misses the true peak
#     returns a silently wrong displacement, which is not a trade worth making.
#
#   * Splitting the loop by output parity to remove the tap branch. Faster, but it
#     regrouped the float multiplies and so changed the last bit, breaking equivalence
#     with OpenCV. The arithmetic below is deliberately left as-is for that reason.
#
# The honest remaining route is vectorising the separable filter, which is an M8 item.
# The cost is a fixed per-point overhead rather than something that scales with scene
# size, so it matters less than the numbers suggest.

const PYRUP_KERNEL = Float32[1, 4, 6, 4, 1] ./ 16

"""
    pyrup!(dst, src)

One Gaussian-pyramid upsampling step: `dst` becomes `src` at twice the size in each
dimension.

Zeros are injected at odd output positions and the result is convolved with the
separable 5-tap kernel `[1,4,6,4,1]/16`, scaled by 4 so that brightness is
preserved rather than quartered. Border samples are reflected without repeating the
edge (OpenCV's `BORDER_REFLECT_101`), which matters here more than it usually does
because the patch is only 5x5 — the border is most of it.

`dst` must be exactly `2 .* size(src)`.
"""
function pyrup!(dst::AbstractMatrix{Float32}, src::AbstractMatrix{Float32})
    sh, sw = size(src)
    size(dst) == (2sh, 2sw) || throw(DimensionMismatch(
        "pyrup! destination must be $((2sh, 2sw)) for a $((sh, sw)) source, " *
        "got $(size(dst))"))

    # The border convention is the subtle part, and it is not what a naive reading
    # suggests. Reflection happens in the *upsampled* coordinate space, not the
    # source's. The difference shows only at the trailing edge, where it makes the
    # last sample's weight accumulate rather than fold onto an interior neighbour:
    # for a 5-sample row, OpenCV gives the last source sample a weight of 0.875 at
    # the second-to-last output, where reflecting in source coordinates would give
    # 0.75 and put 0.125 on its neighbour. Established by probing OpenCV with unit
    # impulses, since no documentation states it.
    #
    # Only two distinct tap patterns exist per axis, because reflection preserves
    # parity and the injected zeros fall on odd upsampled positions:
    #
    #   even output position -> source taps at relative -1, 0, +1, weights 1/8, 3/4, 1/8
    #   odd  output position -> source taps at relative  0, +1,     weights 1/2, 1/2
    #
    # (Doubled from [1,4,6,4,1]/16 so that brightness is preserved rather than
    # halved per axis.) Precomputing the source indices per output row and column
    # turns the inner loop into at most 3x3 multiply-adds with no branches, no
    # modular arithmetic, and no reflection calls — about an order of magnitude
    # cheaper than evaluating all 25 taps and discarding the 16 that are zero.
    #
    # The row taps depend only on the row, so computing them inside the column loop
    # repeated the same reflection arithmetic `2sw` times over. Hoisting them into a
    # scratch vector is worth 1.5x, and it is the only redundancy here: the tap
    # *arithmetic* below is left exactly as it was, including the parenthesisation,
    # because regrouping these float multiplies changes the last bit and this kernel is
    # verified bit-identical against OpenCV.
    rows = _pyrup_row_taps(sh)
    @inbounds for j in 1:(2sw)
        jm, j0, jp, nj = _pyrup_taps(j, sw)
        for i in 1:(2sh)
            im, i0, ip, ni = rows[i]
            acc = 0.0f0
            if nj == 3
                if ni == 3
                    acc = W_C * (W_C * src[i0, j0]) +
                          W_C * W_S * (src[im, j0] + src[ip, j0]) +
                          W_S * W_C * (src[i0, jm] + src[i0, jp]) +
                          W_S * W_S * (src[im, jm] + src[ip, jm] +
                                       src[im, jp] + src[ip, jp])
                else
                    acc = W_H * (W_C * (src[i0, j0] + src[ip, j0]) +
                                 W_S * (src[i0, jm] + src[ip, jm] +
                                        src[i0, jp] + src[ip, jp]))
                end
            else
                if ni == 3
                    acc = W_H * (W_C * (src[i0, j0] + src[i0, jp]) +
                                 W_S * (src[im, j0] + src[ip, j0] +
                                        src[im, jp] + src[ip, jp]))
                else
                    acc = W_H * W_H * (src[i0, j0] + src[ip, j0] +
                                       src[i0, jp] + src[ip, jp])
                end
            end
            dst[i, j] = acc
        end
    end
    return dst
end

# Tap weights, doubled per axis so the surviving (non-zero) taps sum to 1.
const W_C = 6.0f0 / 8      # centre tap, even output position
const W_S = 1.0f0 / 8      # side taps, even output position
const W_H = 4.0f0 / 8      # both taps, odd output position

"""
    _pyrup_taps(k, n) -> (minus, centre, plus, ntaps)

Source indices contributing to 1-based output position `k` of an axis of length `n`,
and how many of them there are.

An even upsampled position sits on a source sample and takes three taps; an odd one
sits between two and takes two, in which case `centre` and `plus` are the pair and
`minus` is unused. Indices are reflected in the upsampled space (see
[`pyrup!`](@ref)), which is what makes the trailing edge behave as OpenCV's does.
"""
# Row taps for an axis of length `n`, cached across calls.
#
# The cascade calls `pyrup!` at a handful of sizes, over and over, so the tap pattern is
# recomputed for the same `n` thousands of times. A cache keyed on `n` removes that and
# keeps `pyrup!` allocation-free. Small and bounded: one entry per distinct axis length
# a cascade visits, which is `log2(upsampling)` of them.
const PYRUP_ROW_TAPS = Dict{Int,Vector{NTuple{4,Int}}}()
const PYRUP_TAPS_LOCK = ReentrantLock()

function _pyrup_row_taps(n::Int)
    v = get(PYRUP_ROW_TAPS, n, nothing)
    v === nothing || return v
    return lock(PYRUP_TAPS_LOCK) do
        get!(PYRUP_ROW_TAPS, n) do
            [_pyrup_taps(i, n) for i in 1:(2n)]
        end
    end
end

@inline function _pyrup_taps(k::Int, n::Int)
    p = k - 1                       # 0-based upsampled position
    if iseven(p)
        c = p ÷ 2 + 1
        m = reflect101_0(p - 2, 2n) ÷ 2 + 1
        q = reflect101_0(p + 2, 2n) ÷ 2 + 1
        return m, c, q, 3
    else
        c = reflect101_0(p - 1, 2n) ÷ 2 + 1
        q = reflect101_0(p + 1, 2n) ÷ 2 + 1
        return c, c, q, 2
    end
end

"""
    reflect101(i, n) -> Int

Reflect index `i` into `1:n` without repeating the edge element: `…3 2 | 1 2 3 … n |
n-1 n-2…`. OpenCV's `BORDER_REFLECT_101`.

Distinct from the reflection that *does* repeat the edge, which Julia calls
`:symmetric` and OpenCV calls `BORDER_REFLECT`. The two differ by one sample at
every boundary, and confusing them is the most common error in ported image code —
so both conventions are named explicitly wherever they appear rather than left to
the reader.
"""
@inline function reflect101(i::Int, n::Int)
    n == 1 && return 1
    while i < 1 || i > n
        i < 1 && (i = 2 - i)
        i > n && (i = 2n - i)
    end
    return i
end

"""
    reflect101_0(p, n) -> Int

[`reflect101`](@ref) on 0-based positions: reflect `p` into `0:(n-1)` without
repeating the edge.

The 0-based form is what the upsampling kernel needs, because it reflects in the
*upsampled* coordinate space where positions are naturally 0-based offsets from the
output origin. Converting to 1-based first and back afterwards would work but reads
worse than having both forms named.

Reflection preserves parity, which is what lets [`pyrup!`](@ref) skip the injected
zeros before reflecting rather than after.
"""
@inline function reflect101_0(p::Int, n::Int)
    n == 1 && return 0
    m = n - 1
    while p < 0 || p > m
        p < 0 && (p = -p)
        p > m && (p = 2m - p)
    end
    return p
end

"""
    RefinementWorkspace

Ping-pong buffers for the sub-pixel upsampling cascade.

Two buffers, alternated, sized for the deepest cascade the configured upsampling
requires. Allocated once per task rather than per point: at 128x upsampling each
buffer is 640x640, so allocating per point would dominate everything else.
"""
struct RefinementWorkspace
    a::Matrix{Float32}
    b::Matrix{Float32}
    patch::Matrix{Float32}
    max_upsampling::Int
end

"""
    refinement_workspace(upsampling; patch = 5) -> RefinementWorkspace

Allocate cascade buffers for up to `upsampling`-fold refinement.
"""
function refinement_workspace(upsampling::Integer; patch::Integer = 5)
    upsampling >= 1 || throw(ArgumentError(
        "`upsampling` must be >= 1, got $upsampling"))
    ispow2(upsampling) || throw(ArgumentError(
        "`upsampling` must be a power of 2, got $upsampling. The cascade doubles " *
        "at each step, and the final division is by `upsampling`, so a " *
        "non-power-of-two would scale the result wrongly rather than error."))
    n = Int(patch) * Int(upsampling)
    return RefinementWorkspace(
        Matrix{Float32}(undef, n, n),
        Matrix{Float32}(undef, n, n),
        Matrix{Float32}(undef, Int(patch), Int(patch)),
        Int(upsampling),
    )
end

"""
    subpixel_peak(rw, surface, radius, upsampling) -> (dx, dy, correlation)

Displacement of the correlation peak, located to `1/upsampling` of a pixel.

A `patch`-sized neighbourhood of the integer peak is upsampled by a cascade of
[`pyrup!`](@ref) steps and its argmax taken. The neighbourhood is clamped to lie
within the surface, so a peak at the edge is refined against an off-centre window
rather than being rejected — the reference does the same, and it matters because a
peak at the search-window edge means the true displacement may be outside the
search range, which the outlier filter is better placed to judge than this
function.

With `upsampling == 1` this is exactly [`peak_offset`](@ref).
"""
function subpixel_peak(
    rw::RefinementWorkspace, surface::AbstractMatrix{Float32},
    radius::Tuple{Int,Int}, upsampling::Integer,
)
    upsampling == 1 && return peak_offset(surface, radius)
    upsampling <= rw.max_upsampling || throw(ArgumentError(
        "upsampling $upsampling exceeds the workspace maximum " *
        "$(rw.max_upsampling)"))

    rx, ry = radius
    nr, nc = size(surface)
    pi_, pj = peak_index(surface)
    p = size(rw.patch, 1)

    # Clamp the patch to the surface. The reference does the same, and the
    # clamping is why the patch origin has to be tracked separately: the peak is
    # not generally at the patch centre.
    half = p ÷ 2
    i0 = clamp(pi_ - half, 1, max(nr - p + 1, 1))
    j0 = clamp(pj - half, 1, max(nc - p + 1, 1))

    # A surface smaller than the patch cannot be refined; report the integer peak.
    (nr < p || nc < p) && return peak_offset(surface, radius)

    @inbounds for j in 1:p, i in 1:p
        rw.patch[i, j] = surface[i0 + i - 1, j0 + j - 1]
    end

    # Cascade, alternating between the two buffers. Each step doubles the extent, so
    # there are log2(upsampling) of them.
    #
    # Every intermediate is a `SubArray` of the same concrete type, including the
    # first — the patch is viewed rather than used directly, so `cur` never changes
    # type across the loop. Mixing a `Matrix` with a `SubArray` there would make
    # `cur` a union, box it on each iteration, and allocate once per cascade step:
    # small, but multiplied by every grid point of every image pair.
    n = p
    factor = 1
    cur = @view rw.patch[1:p, 1:p]
    use_a = true
    while factor < upsampling
        n *= 2
        factor *= 2
        if use_a
            dst = @view rw.a[1:n, 1:n]
            pyrup!(dst, cur)
            cur = dst
        else
            dst = @view rw.b[1:n, 1:n]
            pyrup!(dst, cur)
            cur = dst
        end
        use_a = !use_a
    end

    si, sj, val = peak(cur)
    # Position within the upsampled patch, converted back to surface coordinates,
    # then to a displacement about the surface origin. The `- 1` terms convert
    # Julia's 1-based indices to offsets before dividing by the upsampling factor.
    row = (si - 1) / factor + (i0 - 1)
    col = (sj - 1) / factor + (j0 - 1)
    return col - rx, row - ry, val
end

subpixel_peak(rw, surface, radius::Integer, upsampling) =
    subpixel_peak(rw, surface, (Int(radius), Int(radius)), upsampling)
