# Sliding-window reductions over a NaN-dense field.
#
# Eight call sites in the multi-chip-size search, each sweeping the whole grid, with
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
#     50 ms at width 12 and 665 ms at width 48, against scipy's 4.3 ms. The search
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
comparisons per element however wide the window. This matters because the search
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

The morphological gradient. Used to measure the spread of a-priori
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
    # Getting this backwards shifts even-window results by one pixel, and the search
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

Both paths work in bands of 16 rows rather than over the whole array, which is what keeps
the column-sum scratch in cache — 0.13 MiB on 1024² rather than 8 MiB, and faster for it.
See the implementation notes below.

Pass `hasnan = false` when the caller already knows the input is gap-free. That skips both
the detection scan (12% on 1024²) and the per-pixel count array — the count is then a
function of position alone. Results are bit-identical either way, since the sums are
`Float64` on both paths: this trades memory traffic, never precision. The default
`nothing` detects, for callers that do not know.

!!! note "The choice is per-array, not per-window"
    A single `NaN` anywhere forfeits the saving for the whole array, so a scene with a
    no-data border costs 1.38x one without — and that is the common case, since
    reprojection to a common grid routinely leaves such a border.

    That ratio was 2.4x before both paths were banded, which is why the obvious next step
    is no longer worth taking: degrading gracefully would mean detecting per column inside
    the recurrence, and there is now much less to recover for the complexity. The masked
    path's own scratch is already down to 0.11 MiB.
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

"""
    _window_margins(wx, wy) -> (lx, ly, rx, ry)

How far a window of `(wx, wy)` reaches left/up and right/down from its centre.

For an even width the extra element goes to the **left**, matching
`scipy.ndimage.generic_filter` at its default origin — which is what the reference's filter does,
and what `LocalFilters.localmap` does, so the whole file agrees on one convention. Getting this
backwards shifts even-window results by one pixel, and the chip-size search uses even windows at
every level above the base.

One transcription of that convention rather than four. The file header warns this is the thing
easiest to get backwards; four copies is four chances to.
"""
@inline function _window_margins(wx::Int, wy::Int)
    lx, ly = wx ÷ 2, wy ÷ 2
    return lx, ly, wx - 1 - lx, wy - 1 - ly
end

# Band height for the column-sum scratch in both `windowmean` paths. 16 rows of `Float64` is 128 KiB
# at 1024 columns, which stays in L2 while the row pass reads it. Measured across 4/8/16/32 at both
# window widths and at 128²/256²/512²/1024²: 8 and 16 are within noise of each other and both beat 4
# (too many restarts) and 32 (leaves cache).
#
# Each band reseeds its column recurrence rather than carrying it across, so the redundant work is
# one window prefix per band per column — O(nr/band * wy * nc), which grows with the window. Measured
# on 1024²: 3.16 ms at width 3 rising to 4.4 ms at 49. Mild, and mild in the right place, since
# `windowmean`'s only in-package callers are the preprocessing filters at `filter_width` (default 5).
# The 48-wide windows mentioned at the top of this file belong to the deque reductions, which carry
# no scratch and are untouched by this. A wider default would want a larger band.
const _BAND_ROWS = 16

# The masked path, in the same row bands as the dense one below.
#
# Sum and count are carried together: the count is what makes this NaN-aware, since dividing by the
# window area would be wrong wherever any neighbour is missing.
#
# The saving here is larger than in the dense case, because there are two full-image scratch arrays
# rather than one — `Float64` sums and `Int32` counts, 12 MiB together on 1024² to produce 4 MiB of
# output. Measured 3.7 ms against 9.8 ms and 0.11 MiB against 12.01 MiB, bit-identical.
#
# This path is not a corner case. A single NaN anywhere sends the whole array here, and reprojection
# routinely leaves a no-data border — so a scene that has been warped to a common grid, which is
# every scene in a production run, takes this branch.
#
# `rowbuf`/`cntbuf` look redundant now that `sums` is only 16 rows tall — the row pass could slide
# over `sums` directly. Tried, and it is 6% *slower* (3.55 ms against 3.35): the row pass makes four
# reads per output at a stride of `nb`, and copying the row into a contiguous vector first is worth
# more than the copy costs. Keeping the buffer.
function _windowmean_masked!(out, A::AbstractMatrix, wx::Int, wy::Int)
    nr, nc = size(A)
    lx, ly, rx, ry = _window_margins(wx, wy)
    nb = min(_BAND_ROWS, nr)
    sums = Matrix{Float64}(undef, nb, nc)
    counts = Matrix{Int32}(undef, nb, nc)
    rowbuf = Vector{Float64}(undef, nc)
    cntbuf = Vector{Int32}(undef, nc)

    i0 = 1
    @inbounds while i0 <= nr
        i1 = min(i0 + nb - 1, nr)

        # Column pass over rows i0:i1, seeded with exactly the window row `i0` sees so the result
        # does not depend on the band height.
        for j in 1:nc
            s = 0.0
            c = 0
            for i in max(i0 - ly, 1):min(i0 + ry, nr)
                v = Float64(A[i, j])
                if !isnan(v)
                    s += v
                    c += 1
                end
            end
            for i in i0:i1
                sums[i - i0 + 1, j] = s
                counts[i - i0 + 1, j] = c
                # Slide: drop the element leaving on the left, admit the one entering on the right.
                oi = i - ly
                if 1 <= oi <= nr
                    v = Float64(A[oi, j])
                    if !isnan(v)
                        s -= v
                        c -= 1
                    end
                end
                ii = i + ry + 1
                if ii <= nr
                    v = Float64(A[ii, j])
                    if !isnan(v)
                        s += v
                        c += 1
                    end
                end
            end
        end

        # Row pass, straight into `out`. An empty window — every neighbour missing — is NaN.
        for i in i0:i1
            for j in 1:nc
                rowbuf[j] = sums[i - i0 + 1, j]
                cntbuf[j] = counts[i - i0 + 1, j]
            end
            s = 0.0
            c = 0
            for j in 1:min(rx + 1, nc)
                s += rowbuf[j]
                c += cntbuf[j]
            end
            for j in 1:nc
                out[i, j] = c > 0 ? Float32(s / c) : NaN32
                oj = j - lx
                if 1 <= oj <= nc
                    s -= rowbuf[oj]
                    c -= cntbuf[oj]
                end
                jj = j + rx + 1
                if jj <= nc
                    s += rowbuf[jj]
                    c += cntbuf[jj]
                end
            end
        end
        i0 = i1 + 1
    end
    return out
end

# The dense path, in row bands rather than over the whole image at once.
#
# The column pass produces a sum per pixel and the row pass consumes one row of them at a time, so
# a full-image `Float64` scratch is 8 MiB on 1024² to carry 4 MiB of output — and it is written
# entirely before any of it is read, so every value is evicted from cache before use. Processing a
# band of rows at a time keeps the scratch resident.
#
# This is faster *and* smaller, which is unusual enough to be worth stating: 3.3 ms against 5.6 ms
# and 0.13 MiB against 8.0 MiB on 1024² — the locality is worth more than the cost of restarting
# each band's column recurrence. It also wins at 128², 256² and 512², so there is no crossover
# below which the simpler form is preferable.
#
# Bit-identical to the unbanded version, which is the property that makes banding safe here: each
# band restarts its running sum from the window that its first row sees, so no value is the result
# of a longer or differently-ordered accumulation than before. Verified exactly, not to a tolerance.
#
# The scratch is allocated per call rather than pooled the way `CorrelationWorkspace` is, and now
# deliberately so: banding took it to ~140 KiB, and allocating it measures 0.34 us against a 3.4 ms
# call — 0.01%. Joining the pool would add plumbing for nothing. Before banding, at 8 MiB a call,
# pooling would have been the obvious move; the cheaper change removed the reason for it.
function _windowmean_dense!(out, A::AbstractMatrix, wx::Int, wy::Int)
    nr, nc = size(A)
    lx, ly, rx, ry = _window_margins(wx, wy)
    nb = min(_BAND_ROWS, nr)
    sums = Matrix{Float64}(undef, nb, nc)
    rowbuf = Vector{Float64}(undef, nc)

    i0 = 1
    @inbounds while i0 <= nr
        i1 = min(i0 + nb - 1, nr)

        # Column sums for rows i0:i1. The recurrence restarts here rather than carrying across
        # bands, seeded with exactly the window row `i0` sees — which is what keeps the result
        # independent of the band height.
        for j in 1:nc
            s = 0.0
            for i in max(i0 - ly, 1):min(i0 + ry, nr)
                s += Float64(A[i, j])
            end
            for i in i0:i1
                sums[i - i0 + 1, j] = s
                oi = i - ly
                1 <= oi <= nr && (s -= Float64(A[oi, j]))
                ii = i + ry + 1
                ii <= nr && (s += Float64(A[ii, j]))
            end
        end

        for i in i0:i1
            # Neighbour count factorises across the axes when nothing is missing, so it
            # comes from the window geometry rather than from a stored array.
            ci = min(i + ry, nr) - max(i - ly, 1) + 1
            for j in 1:nc
                rowbuf[j] = sums[i - i0 + 1, j]
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
        i0 = i1 + 1
    end
    return out
end

"""
    windowmedian(A, w) -> Matrix{Float32}

Median over a `w`-wide sliding window, ignoring `NaN` and out-of-bounds neighbours.

O(w² log w) per pixel, which is acceptable here and only here: the chip-size loop's median
windows are always 3 or 5 across. A running-median structure would win asymptotically
but lose badly at 9 or 25 elements.

!!! note "Windowed histograms via integral images, and why they are not used"
    Gupta & Sintorn (2024), *Efficient high-resolution template matching with vector quantized
    nearest neighbour fields* (Pattern Recognition 151:110386), builds an integral image over a
    one-hot encoding so that any rectangle's histogram is available in O(1) regardless of window
    size — which would make this median and MAD constant-time rather than O(w² log w).

    It is not worth doing, and the deciding measurement is where the time actually is. This filter
    runs on the **output grid**, not the image: 27x27 points for a 1024² scene. Measured at 975 us
    against a 99 ms pass, so **1.0%** of it, with the coarse pass another 1.9%. Amdahl's law caps
    the whole win at one percent, and a histogram median is quantized where this one is exact.

    The paper's larger method — vector-quantize the template into `k` codewords, then compare
    *distributions* of codewords through coarse filters — is a different similarity measure rather
    than a faster ZNCC, and its own results show it losing below ~0.5 scale factor. Its geometry is
    also inverted relative to ours: it amortizes one template's codebook over a large query image,
    where AutoRIFT has hundreds of independent chips each with its own search window.

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
    lx, ly, rx, ry = _window_margins(wx, wy)
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
    lx, ly, rx, ry = _window_margins(wx, wy)

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

Clamping matches the reference, and it matters for the coarse chip-size levels: a
48-wide dilation applied to a grid only 20 across would otherwise index out of
bounds.
"""
function _window_size(w, A::AbstractMatrix)
    wx, wy = w isa Tuple ? (Int(w[1]), Int(w[2])) : (Int(w), Int(w))
    (wx >= 1 && wy >= 1) || throw(ArgumentError(
        "window must be at least 1 in each axis, got ($wx, $wy)"))
    return min(wx, size(A, 2)), min(wy, size(A, 1))
end
