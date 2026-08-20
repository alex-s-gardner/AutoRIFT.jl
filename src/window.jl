# Sliding-window reductions over a NaN-dense field.
#
# Eight call sites in the multi-scale pyramid, each sweeping the whole grid, with
# windows from 3 up to 48 across. The inputs are displacement fields where NaN means
# "no measurement", and NaN is not rare — the coarse pass deliberately invalidates
# most of the grid — so every reduction has to ignore NaN rather than propagate it.
#
# ---------------------------------------------------------------------------
# One semantic, seven reductions
# ---------------------------------------------------------------------------
#
# The reference implements these by padding with NaN and then calling a NaN-skipping
# reducer. For max, min, and range it pads by reflection instead, which looks like a
# different rule but is not: the reflected indices always land inside the part of the
# window that is already in bounds, so reflection and NaN-padding give identical
# results at both window parities (verified against scipy at widths 2 through 6).
#
# So all seven reduce to a single rule: **ignore out-of-bounds neighbours, ignore
# NaN**. Everything here implements that, and the output is the same shape as the
# input.
#
# ---------------------------------------------------------------------------
# Why these are hand-written
# ---------------------------------------------------------------------------
#
# Two candidate packages were measured and both were rejected on evidence:
#
#   * `ImageMorphology.extreme_filter` has the right asymptotics but **propagates
#     NaN** — on a field with 20% NaN it poisoned 14 of 15 outputs. Unusable on a
#     displacement field, which is exactly what these run on.
#
#   * `LocalFilters.localmap` has exactly the right boundary and NaN semantics (it
#     collects only in-bounds neighbours into a reusable buffer) and is competitive
#     at width 3 — 2.9 ms against scipy's 3.8 ms on 512². But it is O(w²) per pixel:
#     50 ms at width 12 and 665 ms at width 48, against scipy's 4.3 ms. The pyramid
#     dilates masks with windows up to 48 wide, so that is not a corner case. Routing
#     a median through it is slower still (119 ms at width 5, against 3.6 ms for the
#     same traversal with a trivial reducer).
#
# So all of these are hand-written, and each uses the cheapest algorithm for the
# window sizes it actually sees:
#
#   max, min, range   monotone deque, separable. O(1) per pixel in the window width,
#                     which is what makes the 48-wide dilations affordable.
#   mean              separable running sum, O(1) per pixel. A gap-free input skips the
#                     per-pixel count entirely, since the neighbour count is then a
#                     function of position; callers that know say so via `hasnan`.
#   median, MAD       gather into a stack buffer and insertion-sort. O(w² log w), but
#                     these windows are always 3 or 5 across, where a running
#                     structure loses to a tight sort on 9 or 25 elements.
#   agreement count   direct, since the centre value is needed and there is nothing
#                     to carry between windows.

"""
    windowmax(A, w) -> Matrix{Float32}
    windowmax!(out, A, w)

Maximum over a `w`-wide sliding window, ignoring `NaN` and out-of-bounds neighbours.

`w` is an `Int` or a `(width, height)` tuple. A window containing only `NaN` yields
`NaN`.

O(1) per pixel in the window width, via the monotone-deque decomposition: separable
into a pass along each axis, and each pass amortises to a constant number of
comparisons per element however wide the window. This matters because the pyramid
dilates masks with windows up to 48 across, where an O(w²) implementation is two
orders of magnitude slower.
"""
windowmax(A::AbstractMatrix, w) = windowmax!(similar(A, Float32), A, w)
windowmax!(out, A, w) = _window_extremum!(out, A, w, >)

"""
    windowmin(A, w) -> Matrix{Float32}
    windowmin!(out, A, w)

Minimum over a `w`-wide sliding window, ignoring `NaN` and out-of-bounds neighbours.
See [`windowmax`](@ref).
"""
windowmin(A::AbstractMatrix, w) = windowmin!(similar(A, Float32), A, w)
windowmin!(out, A, w) = _window_extremum!(out, A, w, <)

"""
    windowrange(A, w) -> Matrix{Float32}

Maximum minus minimum over a `w`-wide sliding window, ignoring `NaN`.

The morphological gradient. Used by the pyramid to measure the spread of a-priori
displacements inside a coarse cell, which determines how far the coarse search must
reach.
"""
function windowrange(A::AbstractMatrix, w)
    hi = windowmax(A, w)
    lo = windowmin(A, w)
    @inbounds for i in eachindex(hi)
        hi[i] -= lo[i]
    end
    return hi
end

# Separable monotone-deque extremum. `better(a, b)` is `>` for a maximum and `<` for
# a minimum, so one implementation serves both without a runtime branch — it
# specializes on the function type.
function _window_extremum!(out, A::AbstractMatrix, w, better::F) where {F}
    wx, wy = _window_size(w, A)
    nr, nc = size(A)
    # Scratch for one axis pass, plus the deque itself. Sized once for the larger
    # axis so a single allocation serves both passes.
    n = max(nr, nc)
    buf = Vector{Float32}(undef, n)
    tmp = Matrix{Float32}(undef, nr, nc)
    dq = Vector{Int}(undef, n)
    vals = Vector{Float32}(undef, n)

    # Pass 1: along columns (contiguous in memory).
    @inbounds for j in 1:nc
        for i in 1:nr
            vals[i] = Float32(A[i, j])
        end
        _deque_pass!(buf, vals, nr, wy, dq, better)
        for i in 1:nr
            tmp[i, j] = buf[i]
        end
    end

    # Pass 2: along rows.
    @inbounds for i in 1:nr
        for j in 1:nc
            vals[j] = tmp[i, j]
        end
        _deque_pass!(buf, vals, nc, wx, dq, better)
        for j in 1:nc
            out[i, j] = buf[j]
        end
    end
    return out
end

# One axis of the sliding extremum, over `vals[1:n]`, window width `w`, into
# `buf[1:n]`.
#
# The deque holds indices whose values are candidates for the extremum of some future
# window, kept in monotone order. An element is dropped as soon as a later element
# beats it, because it can never be the extremum again. That gives amortised O(1) per
# element: each index is pushed once and popped once, whatever the window width.
#
# NaN is dropped on the way in rather than compared: `NaN` fails every comparison, so
# a NaN left in the deque would neither beat nor be beaten and would wedge the
# monotone invariant. An empty deque means every value in the window was NaN, which
# is reported as NaN.
@inline function _deque_pass!(buf, vals, n::Int, w::Int, dq::Vector{Int}, better::F) where {F}
    # Window margins. For an even width the extra element goes to the *left*:
    # `left = w ÷ 2`. That is what `scipy.ndimage.generic_filter` does at its default
    # origin, so it is what the reference's filter does, and it is also what
    # `LocalFilters.localmap` does -- so the whole file agrees on one convention.
    # Getting this backwards shifts even-window results by one pixel, and the pyramid
    # uses even windows at every level above the base.
    left = w ÷ 2
    right = w - 1 - left

    head, tail = 1, 0     # inclusive bounds of the live deque region
    admitted = 0          # every index <= this has been offered to the deque

    @inbounds for i in 1:n
        # Admit everything that has entered the window by this output position. On the
        # first iteration that is a whole half-window; afterwards it is one element.
        # Driving admission from a high-water mark rather than special-casing `i == 1`
        # keeps a single code path, and a single path is why this is correct: the
        # earlier version primed separately and admitted the same index twice.
        target = min(i + right, n)
        while admitted < target
            admitted += 1
            v = vals[admitted]
            # NaN is dropped rather than compared. It fails every comparison, so a
            # NaN in the deque would neither beat nor be beaten and would wedge the
            # monotone invariant.
            isnan(v) && continue
            while tail >= head && !better(vals[dq[tail]], v)
                tail -= 1
            end
            tail += 1
            dq[tail] = admitted
        end

        # Evict indices that have fallen off the left edge.
        oldest = i - left
        while tail >= head && dq[head] < oldest
            head += 1
        end

        # An empty deque means every value in the window was NaN.
        buf[i] = tail >= head ? vals[dq[head]] : NaN32
    end
    return buf
end

"""
    windowmean(A, w) -> Matrix{Float32}
    windowmean!(out, A, w)

Mean over a `w`-wide sliding window, ignoring `NaN` and out-of-bounds neighbours.

A window containing only `NaN` yields `NaN`.

Separable running sums, O(1) per pixel: a NaN-aware box filter, which is the one
reduction here that no available package provides. Sums accumulate in `Float64` because
a running sum over a large window is a long accumulation chain.

Pass `hasnan = false` when the caller already knows the input is gap-free. That skips
both the detection scan and the per-pixel count array — the count is then a function of
position alone — which is worth 1.8x. Results are bit-identical either way, since the
sums are `Float64` on both paths: this trades memory traffic, never precision. The
default `nothing` detects, for callers that do not know.

!!! note "The choice is per-array, not per-window"
    A single `NaN` anywhere forfeits the saving for the whole array, so a scene with a
    no-data border costs about 2.4x one without. That is the honest cost of the split
    and it is not hidden by the average case: reprojection routinely leaves such a
    border. Degrading gracefully would mean detecting per column inside the recurrence,
    which is a larger change than the win justifies for now — but it is the right shape
    if this ever becomes hot.
"""
windowmean(A::AbstractMatrix, w; hasnan = nothing) =
    windowmean!(similar(A, Float32), A, w; hasnan)

function windowmean!(out, A::AbstractMatrix, w; hasnan = nothing)
    wx, wy = _window_size(w, A)
    # Detection is a full extra pass (0.3 ms on 1024²), so it is only paid when the
    # caller cannot say. Every in-package caller can.
    gaps = isnothing(hasnan) ? any(isnan, A) : hasnan
    return gaps ? _windowmean_masked!(out, A, wx, wy) :
                  _windowmean_dense!(out, A, wx, wy)
end

function _windowmean_masked!(out, A::AbstractMatrix, wx::Int, wy::Int)
    nr, nc = size(A)
    # Sum and count carried together: the count is what makes this NaN-aware, since
    # dividing by the window area would be wrong wherever any neighbour is missing.
    sums = Matrix{Float64}(undef, nr, nc)
    counts = Matrix{Int32}(undef, nr, nc)

    _running_sum_cols!(sums, counts, A, wy)
    _running_sum_rows!(sums, counts, wx)

    @inbounds for i in eachindex(out)
        c = counts[i]
        out[i] = c > 0 ? Float32(sums[i] / c) : NaN32
    end
    return out
end

function _windowmean_dense!(out, A::AbstractMatrix, wx::Int, wy::Int)
    nr, nc = size(A)
    lx, ly = wx ÷ 2, wy ÷ 2        # even windows are left-biased
    rx, ry = wx - 1 - lx, wy - 1 - ly
    sums = Matrix{Float64}(undef, nr, nc)

    @inbounds for j in 1:nc
        s = 0.0
        for i in 1:min(ry + 1, nr)
            s += Float64(A[i, j])
        end
        for i in 1:nr
            sums[i, j] = s
            oi = i - ly
            1 <= oi <= nr && (s -= Float64(A[oi, j]))
            ii = i + ry + 1
            ii <= nr && (s += Float64(A[ii, j]))
        end
    end

    rowbuf = Vector{Float64}(undef, nc)
    @inbounds for i in 1:nr
        # Neighbour count factorises across the axes when nothing is missing, so it
        # comes from the window geometry rather than from a stored array.
        ci = min(i + ry, nr) - max(i - ly, 1) + 1
        for j in 1:nc
            rowbuf[j] = sums[i, j]
        end
        s = 0.0
        for j in 1:min(rx + 1, nc)
            s += rowbuf[j]
        end
        for j in 1:nc
            cj = min(j + rx, nc) - max(j - lx, 1) + 1
            out[i, j] = Float32(s / (ci * cj))
            oj = j - lx
            1 <= oj <= nc && (s -= rowbuf[oj])
            jj = j + rx + 1
            jj <= nc && (s += rowbuf[jj])
        end
    end
    return out
end

function _running_sum_cols!(sums, counts, A, w)
    nr, nc = size(A)
    left = w ÷ 2               # even windows are left-biased; see `_deque_pass!`
    right = w - 1 - left
    @inbounds for j in 1:nc
        s = 0.0
        c = 0
        # Prime with the first window.
        for i in 1:min(right + 1, nr)
            v = Float64(A[i, j])
            if !isnan(v)
                s += v
                c += 1
            end
        end
        for i in 1:nr
            sums[i, j] = s
            counts[i, j] = c
            # Slide: drop the element leaving on the left, admit the one entering on
            # the right.
            out_i = i - left
            if 1 <= out_i <= nr
                v = Float64(A[out_i, j])
                if !isnan(v)
                    s -= v
                    c -= 1
                end
            end
            in_i = i + right + 1
            if in_i <= nr
                v = Float64(A[in_i, j])
                if !isnan(v)
                    s += v
                    c += 1
                end
            end
        end
    end
    return sums
end

function _running_sum_rows!(sums, counts, w)
    nr, nc = size(sums)
    left = w ÷ 2               # even windows are left-biased; see `_deque_pass!`
    right = w - 1 - left
    rowbuf = Vector{Float64}(undef, nc)
    cntbuf = Vector{Int32}(undef, nc)
    @inbounds for i in 1:nr
        for j in 1:nc
            rowbuf[j] = sums[i, j]
            cntbuf[j] = counts[i, j]
        end
        s = 0.0
        c = 0
        for j in 1:min(right + 1, nc)
            s += rowbuf[j]
            c += cntbuf[j]
        end
        for j in 1:nc
            sums[i, j] = s
            counts[i, j] = c
            out_j = j - left
            if 1 <= out_j <= nc
                s -= rowbuf[out_j]
                c -= cntbuf[out_j]
            end
            in_j = j + right + 1
            if in_j <= nc
                s += rowbuf[in_j]
                c += cntbuf[in_j]
            end
        end
    end
    return sums
end

"""
    windowmedian(A, w) -> Matrix{Float32}

Median over a `w`-wide sliding window, ignoring `NaN` and out-of-bounds neighbours.

O(w² log w) per pixel, which is acceptable here and only here: the pyramid's median
windows are always 3 or 5 across. A running-median structure would win asymptotically
but lose badly at 9 or 25 elements.

Hand-written rather than delegated. `LocalFilters.localmap` supplies exactly the right
boundary and NaN contract, but routing a general median through it costs 119 ms on
512² where the same traversal with a trivial reducer costs 3.6 ms — the median is
essentially all of the time, not the window machinery. Gathering into a stack buffer
and insertion-sorting the live prefix is 1.8x faster than that, and beats scipy at
width 3.
"""
windowmedian(A::AbstractMatrix, w) = _window_sorted(A, w, _median_of!)

"""
    windowmad(A, w) -> Matrix{Float32}

Median absolute deviation over a `w`-wide sliding window, ignoring `NaN`.

`median(|x - median(x)|)`, the robust scale estimate the outlier filter compares
displacements against. Two sorts of the gathered values rather than one, so roughly twice
the cost of [`windowmedian`](@ref), and subject to the same window-size caveat.

When both statistics are wanted, [`windowmedmad`](@ref) returns them from one traversal.
"""
windowmad(A::AbstractMatrix, w) = _window_sorted(A, w, _mad_of!)

"""
    windowmedmad(A, w) -> (median, mad)

Median and median absolute deviation over the same `w`-wide sliding window, in one pass.

Both are needed together by the outlier filter, and computing them separately gathers and
sorts each window twice. Fusing them is 1.67x faster, and identical to calling
[`windowmedian`](@ref) and [`windowmad`](@ref) in turn by construction rather than by
coincidence: all three route through the same traversal, and the singles are this
function's reducer with one result dropped. The MAD already computes the median on its
way, so the fusion adds no work and needs no extra scratch.
"""
windowmedmad(A::AbstractMatrix, w) = _window_sorted(A, w, _medmad_of!, Val(2))

# Gather-then-reduce over a small stack buffer. `reduce!` receives the buffer and the
# count of live values and may reorder them freely, which is what lets the median sort
# in place with no allocation per window.
#
# `Val(N)` is how many values the reducer returns, and so how many output arrays are
# produced. That is what lets a fused pair like median-and-MAD share this traversal
# rather than copy it: the gather, the left-bias margins, and the all-missing sentinel
# exist once. The window convention in particular is the thing this file's header warns
# is easy to get backwards, so having one transcription of it matters more than the
# handful of lines saved.
function _window_sorted(A::AbstractMatrix, w, reduce!::F, ::Val{N} = Val(1)) where {F,N}
    wx, wy = _window_size(w, A)
    nr, nc = size(A)
    outs = ntuple(_ -> fill(NaN32, nr, nc), Val(N))
    lx, ly = wx ÷ 2, wy ÷ 2        # even windows are left-biased
    rx, ry = wx - 1 - lx, wy - 1 - ly
    buf = Vector{Float32}(undef, wx * wy)

    @inbounds for j in 1:nc, i in 1:nr
        n = 0
        for jj in max(j - lx, 1):min(j + rx, nc)
            for ii in max(i - ly, 1):min(i + ry, nr)
                v = Float32(A[ii, jj])
                isnan(v) && continue
                n += 1
                buf[n] = v
            end
        end
        n == 0 && continue         # every neighbour missing; leave NaN
        vals = reduce!(buf, n)
        # `N` is a compile-time constant, so this unrolls and the tuple never
        # materialises — a one-output reducer costs exactly what it did before.
        ntuple(k -> (outs[k][i, j] = vals[k]), Val(N))
    end
    return N == 1 ? only(outs) : outs
end

# Insertion sort of `buf[1:n]`, then the middle. At n <= 25 this beats calling a
# partial-selection routine, whose setup dominates at that size.
@inline function _median_of!(buf::Vector{Float32}, n::Int)
    _insertion_sort!(buf, n)
    return (_sorted_median(buf, n),)
end

@inline _mad_of!(buf::Vector{Float32}, n::Int) = (_medmad_of!(buf, n)[2],)

# Median and MAD from one sort of the gathered values.
#
# The fusion is free because the MAD already needs the median: it overwrites the buffer
# with deviations from it, so returning that intermediate alongside costs nothing and
# needs no second scratch array. `_median_of!` and `_mad_of!` are this function with one
# of the two results dropped, which is why all three are one traversal.
@inline function _medmad_of!(buf::Vector{Float32}, n::Int)
    _insertion_sort!(buf, n)
    m = _sorted_median(buf, n)
    @inbounds for k in 1:n
        buf[k] = abs(buf[k] - m)
    end
    # The deviations are not sorted even though the values were, so sort again.
    _insertion_sort!(buf, n)
    return (m, _sorted_median(buf, n))
end

@inline function _insertion_sort!(buf::Vector{Float32}, n::Int)
    @inbounds for k in 2:n
        v = buf[k]
        m = k - 1
        while m >= 1 && buf[m] > v
            buf[m + 1] = buf[m]
            m -= 1
        end
        buf[m + 1] = v
    end
    return buf
end

# Mean of the two middle elements for an even count, matching the reference.
@inline function _sorted_median(buf::Vector{Float32}, n::Int)
    @inbounds return isodd(n) ? buf[(n + 1) >> 1] :
                     (buf[n >> 1] + buf[(n >> 1) + 1]) / 2
end

"""
    count_agreeing(A, w, tol) -> Matrix{Float32}

Number of neighbours within `tol` of the window's centre value, itself included.

The reference calls this "displacement distance count". It is the agreement measure
the outlier filter thresholds: a displacement supported by enough of its neighbours
is kept, one that stands alone is rejected. `NaN` neighbours never agree, and a `NaN`
centre agrees with nothing, so both yield a count of zero.
"""
function count_agreeing(A::AbstractMatrix, w, tol::Real)
    wx, wy = _window_size(w, A)
    nr, nc = size(A)
    out = similar(A, Float32)
    t = Float32(tol)
    lx, ly = wx ÷ 2, wy ÷ 2    # even windows are left-biased; see `_deque_pass!`
    rx, ry = wx - 1 - lx, wy - 1 - ly

    @inbounds for j in 1:nc, i in 1:nr
        centre = Float32(A[i, j])
        if isnan(centre)
            out[i, j] = 0.0f0
            continue
        end
        n = 0
        for jj in max(j - lx, 1):min(j + rx, nc)
            for ii in max(i - ly, 1):min(i + ry, nr)
                v = Float32(A[ii, jj])
                # NaN fails this comparison, so missing neighbours simply do not
                # count -- no separate check needed.
                abs(v - centre) < t && (n += 1)
            end
        end
        out[i, j] = Float32(n)
    end
    return out
end

# ---------------------------------------------------------------------------

"""
    _window_size(w, A) -> (wx, wy)

Normalise a window specification to `(width, height)`, clamped to the array.

Clamping matches the reference, and it matters for the coarse pyramid levels: a
48-wide dilation applied to a grid only 20 across would otherwise index out of
bounds.
"""
function _window_size(w, A::AbstractMatrix)
    wx, wy = w isa Tuple ? (Int(w[1]), Int(w[2])) : (Int(w), Int(w))
    (wx >= 1 && wy >= 1) || throw(ArgumentError(
        "window must be at least 1 in each axis, got ($wx, $wy)"))
    return min(wx, size(A, 2)), min(wy, size(A, 1))
end
