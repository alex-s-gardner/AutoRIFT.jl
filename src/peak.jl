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
    # this loop runs once per grid point per chip-size level.
    #
    # The extra clause is what keeps the semantics: a strictly greater value always
    # wins, and an *equal* value wins only if it precedes the incumbent in
    # row-major order. That yields the same element OpenCV's `minMaxLoc` would
    # return, reached in a different order. Verified against 300 randomised
    # surfaces including heavily-tied and all-equal cases.
    #
    # This is the top self-time frame in a profiled pass, so the two-pass alternative was
    # measured: a strict-max sweep in memory order, then a row-major scan for the first
    # element equal to it. That is 2x *slower* at every size and content tried (0.40-0.56x,
    # over random, heavily-tied and all-NaN surfaces at 16² through 64²) -- the row-major
    # second pass strides across memory and costs more than the tie clause it removes. The
    # fused pass below is the fast form, not merely the simple one.
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
    i, j, v = peak(surface)
    return (_offset_at(i, j, radius)..., v)
end

# The displacement a peak at `(i, j)` stands for, for a caller that has already located it.
#
# The arithmetic of `peak_offset` without its scan, and a separate name rather than a method of it
# because it takes no surface and returns two values where `peak_offset` returns three.
@inline function _offset_at(i::Int, j::Int, radius::Tuple{Int,Int})
    rx, ry = radius
    return Float64(j - rx - 1), Float64(i - ry - 1)
end

# How far around the primary peak is excluded when looking for the secondary, in surface samples.
#
# This is what separates a rival peak from the primary's own shoulder, and the requirement is
# one-sided: it must exceed the width of the correlation peak itself. Measured over 4000 real
# surfaces at chip 16, radius 20, the median ratio by exclusion is
#
#   0 -> 1.17    1 -> 1.58    2 -> 1.66    3 -> 1.66    4 -> 1.67    5 -> 1.68    6 -> 1.69
#
# so excluding nothing does collapse the measure — the rival is then the sample beside the peak, and
# 20% of surfaces score below 1.05 where 9% do at any exclusion of 2 or more. Past that it saturates:
# 2 through 6 rank the same surfaces the same way, at a Spearman correlation above 0.99 against 3.
# The value sits in the saturated region with margin on the side that matters.
#
# A constant rather than a `Params` field: making it tunable would imply a decision the caller has
# information to make, and above the peak width the choice does not change the answer.
const PEAK_EXCLUSION = 3

"""
    peak_at_boundary(surface) -> Bool

Whether the integer peak of `surface` lies on its first or last row or column.

The surface spans `-radius … radius - 1`, so this means the true displacement is at or beyond the search
limit: the reported value is a lower bound, and its sub-pixel part along that axis is unrecoverable because
the peak's far side was never sampled. Measured on known translations at chip 16, radius 20, refining such
a peak is *worse* than keeping the integer one — 1.34 px against 0.40 px at a true shift of −18.6, and
0.94 px against exact at −19.0.

`track!` therefore reports **both** `correlation` and `peak_ratio` as zero at such a point, so any positive
threshold on either rejects it without the caller having to know the condition exists. Zero is otherwise
unreachable for both: a real ZNCC peak is strictly positive, and a real ratio is at least 1, the primary
being the surface maximum. The displacement itself is still reported, being the best available bound.
Select these points with `correlation .== 0` to find where `search_radius` needs raising — at the
ITS_LIVE configuration over Jakobshavn 4.2% of points that correlate above 0.3 are affected, and
0.12% at chip 32.

One function rather than the test written out at each site: the two outputs must agree about which points
are railed, and a copy of the condition is how they would drift apart.

    peak_at_boundary(surface, i, j) -> Bool

The same test on a peak the caller has already located. `(i, j)` must be what
[`peak_index`](@ref) would return; the one-argument form is this composed with that call. The
correlator uses this form, having located the peak once for every quantity it derives.
"""
@inline peak_at_boundary(surface::AbstractMatrix, i::Int, j::Int) =
    (i == 1) | (j == 1) | (i == size(surface, 1)) | (j == size(surface, 2))

@inline peak_at_boundary(surface::AbstractMatrix) =
    peak_at_boundary(surface, peak_index(surface)...)

"""
    peak_ratio(surface[, exclusion = $PEAK_EXCLUSION]) -> Float32

The primary-to-secondary peak ratio: the correlation peak divided by the highest rival peak
elsewhere on the surface.

The rival — the secondary peak — is the largest sample lying more than `exclusion` samples from the
primary along either axis. Excluding a box rather than a single sample is what makes the two peaks
*distinct*: the samples immediately around the primary are its own shoulder, and dividing by those
would return a number near 1 at every point regardless of how clean the surface was.

The result is `>= 1` by construction, since the primary is the surface maximum. A ratio near 1 means
some other displacement matched nearly as well, so the reported one is a coin flip between them; a
large ratio means the chosen displacement was the only candidate. This is the measure of an
*ambiguous* surface, and ambiguity is the failure mode that a peak height cannot see: periodic
texture — crevasse fields, dune trains, sea ice floes — produces rival peaks one wavelength apart,
each as tall as the true one.

`Inf32` when no rival is positive at all, meaning the primary is the only candidate the surface
offers. `NaN32` when the surface is too small to have anything outside the exclusion box, which says
"no ratio could be computed" rather than a low ratio. The two are distinct and a caller testing
`isfinite` separates both from a real value.

# Which measure to gate on

**Use `correlation`, not this, to predict whether a displacement is reliable.** The two answer
different questions — how *strong* the match was against whether it was *unique* — and a gate wants
the first. Measured over the 80,648 directly measured points of a Jakobshavn pair that both this
package and `autoRIFT.py` resolved, against the independent label "the two disagree by more than
0.25 px", which 0.34% of them carry:

| measure | AUC | rate in worst decile | rate in best decile |
|---|---:|---:|---:|
| `correlation` | **0.791** | 1.2% | 0.0% |
| this | 0.544 | 0.5% | 0.6% |

`correlation` falls monotonically across its deciles, from 1.2% to zero. This ratio is flat at
0.1–0.6% and *not* monotonic — its highest decile is its worst — so **as a reliability gate on its
own it has essentially no skill.** Nor is the flatness an artifact of pooling points of differing
correlation: within a band of fixed `correlation` the ranking mildly *inverts*, at an AUC of 0.43
over 0.2–0.4 and 0.50 over 0.4–0.6. The two measures are nearly independent, at a Spearman rank
correlation of 0.242. `tools/ab/peak_ratio_skill.jl` computes all of it.

That is not a defect in the quantity, it is what the quantity measures. Ambiguity and unreliability
are different failures, and on this scene — a fast outlet glacier whose flow the search radius covers
— the reference and this package disagree where the match is *weak*, not where it is contested. A
ratio near 1 does mean two displacements matched nearly equally well; it just does not follow that
the one chosen was the wrong one.

So use this to *diagnose* rather than to threshold. A region of low ratios at otherwise healthy
correlation is periodic texture matching at more than one offset — crevasse fields, dunes, sea ice —
and the fix is a chip size that spans more than one wavelength, not a tighter gate.

`exclusion` is `PEAK_EXCLUSION` by default; see that constant for why it is fixed rather than
tunable.

This is a property of the surface alone. The search-boundary condition — where the reported ratio is
forced to zero, below the 1 a real ratio cannot go under — is applied by the caller; see
[`AutoRIFT.peak_at_boundary`](@ref).

    peak_ratio(surface, i, j[, exclusion = $PEAK_EXCLUSION]) -> Float32

The same measure on a peak the caller has already located. `(i, j)` must be what
[`peak_index`](@ref) would return; the form without them is this composed with that call.

The correlator uses this form. Every point needs the displacement, the peak value, the boundary
flag and this ratio, and the four naive calls locate the same peak four times over —
`peak_index` is the top self-time frame in a profiled pass, so at chip 16 and radius 20, the
geometry ITS_LIVE runs Landsat at, that is 3.92 us against 2.15 us, 6% of a whole point.
"""
@inline function peak_ratio(surface::AbstractMatrix, i::Int, j::Int,
                            exclusion::Int = PEAK_EXCLUSION)
    nr, nc = size(surface)
    # `Float32` throughout. A maximum accumulates no error, so there is nothing for a wider type to
    # protect, and one division in the surface's own precision is what lets the device kernel in
    # `ext/gpu/kernels.jl` reach the same value from the same surface — promoting to `Float64` and
    # rounding the quotient back would double-round and put a difference between them that is the
    # arithmetic's rather than the surface's.
    h = Float32(@inbounds surface[i, j])
    # The exclusion box is contiguous along both axes, so how many samples lie outside it follows
    # from the geometry. Counting them in the loop instead would be per-element work for a number
    # the extents already determine.
    rlo, rhi = max(i - exclusion, 1), min(i + exclusion, nr)
    clo, chi = max(j - exclusion, 1), min(j + exclusion, nc)
    # Nothing outside the box to compare against.
    nr * nc - (rhi - rlo + 1) * (chi - clo + 1) == 0 && return NaN32

    # Four independent running maxima rather than one, combined below.
    #
    # This is the whole cost of the function, and the reason is latency, not throughput: `second`
    # depends on the previous iteration's `second`, so a single accumulator serialises the loop on
    # the comparison's latency however wide the loads are. Four chains give the out-of-order engine
    # four independent comparisons to overlap. Measured 3.0-3.3x over 32² through 64², and the
    # two-moment measure this replaced was 35% *faster* than the one-chain form for exactly this
    # reason -- its sum and sum-of-squares are already two independent chains.
    #
    # `ifelse` rather than a branch: the comparison is the same one, so `NaN` is still skipped rather
    # than propagated as `max` would propagate it, and the result carries no data-dependent branch.
    # Verified bit-identical to the one-chain form over 4000 surfaces x 5 exclusions including
    # all-`NaN`, all-zero, all-negative and single-`NaN` cases.
    #
    # Traversal stays column-major, along memory, for the reason `peak_index` documents. No
    # tie-breaking rule is needed where `peak_index` needs one: only the secondary's *value* is used,
    # never its location, so which of several equal rivals wins does not affect the result -- which
    # is also what makes splitting the scan across four chains safe.
    m1 = m2 = m3 = m4 = -Inf32
    @inbounds for c in 1:nc
        if c < clo || c > chi
            # A column clear of the box: every row counts, so the inner loop carries no test at all.
            r = 1
            while r + 3 <= nr
                v1 = Float32(surface[r, c])
                v2 = Float32(surface[r + 1, c])
                v3 = Float32(surface[r + 2, c])
                v4 = Float32(surface[r + 3, c])
                m1 = ifelse(v1 > m1, v1, m1)
                m2 = ifelse(v2 > m2, v2, m2)
                m3 = ifelse(v3 > m3, v3, m3)
                m4 = ifelse(v4 > m4, v4, m4)
                r += 4
            end
            while r <= nr
                v = Float32(surface[r, c])
                m1 = ifelse(v > m1, v, m1)
                r += 1
            end
        else
            # A column straddling the box: the rows above and below it, each a clean range.
            for r in 1:(rlo - 1)
                v = Float32(surface[r, c])
                m1 = ifelse(v > m1, v, m1)
            end
            for r in (rhi + 1):nr
                v = Float32(surface[r, c])
                m2 = ifelse(v > m2, v, m2)
            end
        end
    end
    a = ifelse(m1 > m2, m1, m2)
    b = ifelse(m3 > m4, m3, m4)
    second = ifelse(a > b, a, b)
    # A ratio of two heights only orders surfaces the way the name promises when the primary is a
    # peak in the first place. Written `!(h > 0)` so an all-`NaN` surface — where `peak_index`
    # returns `(1, 1)` having found no candidate at all — lands here rather than falling through
    # every comparison to a finite-looking answer.
    !(h > 0) && return NaN32
    # No positive rival: the primary is unrivalled, not unassessable. `Inf32` says that and keeps
    # the ordering "larger is less ambiguous" intact, where `NaN` would drop the very cleanest
    # points out of any comparison a caller makes.
    second <= 0 && return Inf32
    return h / second
end

@inline peak_ratio(surface::AbstractMatrix, exclusion::Int = PEAK_EXCLUSION) =
    peak_ratio(surface, peak_index(surface)..., exclusion)

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
# Cost, and what has been tried
# ---------------------------------------------------------------------------
#
# 77.7 us at 64x upsampling, down from 119 us, and 1.53x faster than before the interior/border
# split in `pyrup!`'s vertical pass. Earlier steps were the two-pass form and the hoisted tap
# table, together taking the gap to OpenCV from ~6x to ~2.3x. The algorithm is inherently
# expensive — it materialises a 320x320 surface from a 5x5 patch to locate one maximum — but
# refinement is no longer the dominant cost of a fine pass.
#
# Five routes were examined. **The one that worked** was reading the interior taps directly and
# consulting the table only for the four border rows: the table was the bottleneck, since a
# `Vector{NTuple{4,Int}}` indexed per output row yields indices the compiler cannot prove
# contiguous, so the loop stayed scalar. See `pyrup!` for why that split is exact.
#
# The other five were measured and rejected, which is worth recording so they are not retried
# blind:
#
#   * **Evaluating the composite interpolant directly.** `pyrup!` is a separable *linear* map, so
#     `log2(up)` doublings compose into one fixed `(patch*up) x patch` matrix `K` — depending only on
#     the patch size and the upsampling, hence the same for every point — and the whole cascade is
#     `K·P·K'`. Verified to 4.2e-7 of the cascade, with identical displacements over 500 real
#     surfaces, so the factorisation is right. It is nonetheless **1.03-1.06x**, i.e. nothing, from
#     16x through 128x and through BLAS; a hand-written form and a running-argmax form that never
#     materialises the surface are both 3x *worse*.
#
#     The operation counts say why, and they are the thing to check before trying this again: the
#     cascade sums level areas, `n² · (1 + 1/4 + 1/16 + …) → n² · 4/3`, where the composite is
#     `n² · patch`. At 64x that is 520,000 multiplies against 614,250 — a 15% saving, paid back in
#     access pattern. The same change *does* pay 5.4x on a GPU, and the difference is not the
#     arithmetic but the traffic: there the cascade writes every level through device memory, where
#     here 820 kB stays in L2. `docs/gpu.md` records both halves.
#
#   * Cropping the cascade around the running peak at each level. The large win in
#     principle. But the peak's deviation from its own rescaled position was measured at up
#     to 5.5 previous-level samples over 400 surfaces — far too wide to bound, and an
#     empirical bound is not a guarantee. A cropped cascade that misses the peak returns a
#     silently wrong displacement.
#
#   * A 3x3 patch instead of 5x5, which would cost 36% as much. 155 of 200 test surfaces
#     gave a different answer, by up to 3 pixels: a 3x3 patch has no interior at all under a
#     5-tap kernel, so every output depends on reflected border. The reference's choice of
#     5x5 is load-bearing.
#
#   * Splitting the inner loop by row parity to hoist the tap branch. Bit-identical, but
#     1.06x — the branch predictor was already handling it. Note this is *not* the split that
#     eventually worked: parity removes a branch, and the cost was the indirect table read.
#
#   * Reordering the fused arithmetic. Faster, but it moved *further* from OpenCV rather
#     than merely differently, which is the distinction that rules it out. The two-pass form
#     also changes the last bit and is kept anyway; see `pyrup!` for why those two cases
#     differ.
#
# The cost is a fixed per-point overhead rather than something that scales with scene size, so it
# matters most on a dense grid.
#
# The same interior/border split does **not** help the horizontal pass — measured at 0.29x, i.e.
# 3.4x *slower*, bit-identical but far worse. The two passes are not symmetric: the vertical one
# writes two adjacent rows of one column, which is contiguous, while the horizontal one would
# write two adjacent *columns*, which are `2sh` floats apart. Handling one output column at a time
# and paying `_pyrup_taps` once for it keeps the writes sequential, and that locality is worth more
# than the table lookup costs. Recorded because the symmetry is inviting and the answer is
# counter-intuitive.

"""
    pyrup!(dst, src, [scratch])

One Gaussian-pyramid upsampling step: `dst` becomes `src` at twice the size in each
dimension.

Zeros are injected at odd output positions and the result is convolved with the
separable 5-tap kernel `[1,4,6,4,1]/16`, scaled by 4 so that brightness is
preserved rather than quartered. Border samples are reflected without repeating the
edge (OpenCV's `BORDER_REFLECT_101`), which matters here more than it usually does
because the patch is only 5x5 — the border is most of it.

`dst` must be exactly `2 .* size(src)`. `scratch` is an optional buffer of at least
`(2 * size(src, 1), size(src, 2))` for the intermediate of the two-pass form; the
refinement cascade supplies one from its workspace so the hot path allocates nothing.
"""
# Standalone form: allocates its own intermediate. The right trade for a function whose
# standalone use is tests and one-off calls; the cascade passes scratch and allocates
# nothing.
pyrup!(dst::AbstractMatrix{Float32}, src::AbstractMatrix{Float32}) =
    pyrup!(dst, src, Matrix{Float32}(undef, 2 * size(src, 1), size(src, 2)))

function pyrup!(dst::AbstractMatrix{Float32}, src::AbstractMatrix{Float32},
                scratch::AbstractMatrix{Float32},
                rows::Vector{NTuple{4,Int}} = _pyrup_row_taps(size(src, 1)))
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
    # turns each pass into at most three multiply-adds with no branches, no modular
    # arithmetic, and no reflection calls.
    #
    # Applied as two passes rather than one. A single pass costs up to nine loads per
    # output sample; separating it into a vertical pass through an intermediate and then a
    # horizontal pass costs three in each, and is 2.6x faster at the sizes the refinement
    # cascade uses.
    #
    # Separating changes the order of the float multiplies, and so the last bit — but the
    # thing that matters is agreement with OpenCV, not with the fused form. Measured across
    # all 25 upsampling fixtures, both forms sit at 1.788e-7 from OpenCV: exactly as close,
    # because OpenCV is itself separable. An earlier attempt at a different reordering was
    # rejected for drifting *further* from the reference; this one does not, which is the
    # distinction that decides it.
    length(rows) == 2sh || throw(DimensionMismatch(
        "tap table has $(length(rows)) entries but a $(sh)-row source needs $(2sh)"))
    tmp = @view scratch[1:(2sh), 1:sw]

    # Vertical: 2sh x sw, writing down each column so both source and destination are
    # traversed contiguously.
    #
    # Interior rows read their taps directly; only the four border rows consult the table. That
    # split is what makes this fast, and it is exact rather than approximate: for every source row
    # `k` in `2:(sh-1)`, output row `2k-1` takes three taps at `k-1, k, k+1` and output row `2k`
    # takes two at `k, k+1`, with no reflection anywhere. Verified against `_pyrup_row_taps` by
    # construction at every size the cascade visits (5, 10, 20, 40, 80, 160, 320); the rows the
    # formula does *not* cover are always exactly `1, 2, 2sh-1, 2sh`.
    #
    # The table lookup was the bottleneck, not the arithmetic. `rows[i]` is a `Vector{NTuple{4,Int}}`
    # indexed per output row, and the source indices it yields are opaque to the compiler — so the
    # loads cannot be proven contiguous and the loop stays scalar. Reading `src[k-1, j]`,
    # `src[k, j]`, `src[k+1, j]` directly lets it vectorise, and lets each source value be loaded
    # once and used by both output rows instead of twice. Measured 3.1x at 40x40 and 5.3x at
    # 160x160, which is why the vertical pass was 3.65x the horizontal one despite doing the same
    # arithmetic over half the data.
    #
    # The per-element arithmetic is untouched — the same weights applied to the same taps in the
    # same order — and the result is bit-identical. That distinction matters here: an earlier
    # attempt at this pass was reverted for reassociating the sum, and a previous parity-split
    # version was kept only after confirming it changed nothing.
    @inbounds for j in 1:sw
        for k in 2:(sh - 1)
            a = src[k - 1, j]
            b = src[k, j]
            c = src[k + 1, j]
            tmp[2k - 1, j] = W_C * b + W_S * (a + c)
            tmp[2k, j] = W_H * (b + c)
        end
        # The four rows whose taps reflect. Written as a loop over an explicit tuple rather than a
        # special case each, so adding a size where the border set differs would fail the
        # construction check above rather than silently read the wrong row.
        for i in (1, 2, 2sh - 1, 2sh)
            im, i0, ip, ni = rows[i]
            tmp[i, j] = ni == 3 ? W_C * src[i0, j] + W_S * (src[im, j] + src[ip, j]) :
                                  W_H * (src[i0, j] + src[ip, j])
        end
    end

    # Horizontal: the column taps are loop-invariant, so the branch is hoisted out of the
    # inner loop entirely.
    @inbounds for j in 1:(2sw)
        jm, j0, jp, nj = _pyrup_taps(j, sw)
        if nj == 3
            for i in 1:(2sh)
                dst[i, j] = W_C * tmp[i, j0] + W_S * (tmp[i, jm] + tmp[i, jp])
            end
        else
            for i in 1:(2sh)
                dst[i, j] = W_H * (tmp[i, j0] + tmp[i, jp])
            end
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

# The tap pattern for an axis of length `n`.
#
# `pyrup!` needs it for every output row, so computing it inside the column loop meant
# recomputing the same table `2 * width` times over — hoisting it is worth 1.5x.
#
# Built on demand rather than cached in a process-global. It costs ~0.7 us for the largest
# size a cascade uses and ~3 us for every size one visits, against ~560 us for the cascade
# itself, so a global `Dict` with a lock would be machinery the cost does not justify — unlike
# the FFT plans in `src/plans.jl`, which cost milliseconds and are genuinely process-wide.
# `RefinementWorkspace` holds the tables it needs, since it already knows both the patch size
# and the maximum upsampling at construction.
_pyrup_row_taps(n::Int) = [_pyrup_taps(i, n) for i in 1:(2n)]

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
    # Intermediate for `pyrup!`'s two-pass form. Sized for the deepest level, so it serves
    # every shallower one and the whole cascade allocates nothing.
    scratch::Matrix{Float32}
    # Tap tables, one per cascade level, keyed by level rather than by axis length. Built here
    # because the workspace already knows the patch size and the maximum upsampling, which
    # together determine exactly the set of sizes the cascade will visit — so there is nothing
    # to discover at run time and no cache to manage.
    taps::Vector{Vector{NTuple{4,Int}}}
    max_upsampling::Int
end

"""
    AutoRIFT.REFINE_PATCH

Side length of the correlation-surface neighbourhood the sub-pixel cascade upsamples.

The reference's 5, and load-bearing rather than a tunable. A 3x3 patch has no interior at all
under a 5-tap kernel, so every output would depend on reflected border: measured, 155 of 200
surfaces gave a different answer, by up to 3 pixels.

Named because the cascade is implemented twice — here and as a device kernel — and the two must
agree about it, since the refined `dx`/`dy` are asserted bit-identical between them.
"""
const REFINE_PATCH = 5

"""
    refinement_workspace(upsampling; patch = AutoRIFT.REFINE_PATCH) -> RefinementWorkspace

Allocate cascade buffers for up to `upsampling`-fold refinement.
"""
function refinement_workspace(upsampling::Integer; patch::Integer = REFINE_PATCH)
    upsampling >= 1 || throw(ArgumentError(
        "`upsampling` must be >= 1, got $upsampling"))
    ispow2(upsampling) || throw(ArgumentError(
        "`upsampling` must be a power of 2, got $upsampling. The cascade doubles " *
        "at each step, and the final division is by `upsampling`, so a " *
        "non-power-of-two would scale the result wrongly rather than error."))
    p = Int(patch)
    up = Int(upsampling)
    n = p * up
    # One table per doubling: the cascade reads a `p * 2^k` sized source at level k.
    nlevels = up == 1 ? 0 : Int(log2(up))
    taps = [_pyrup_row_taps(p << (k - 1)) for k in 1:max(nlevels, 1)]
    return RefinementWorkspace(
        Matrix{Float32}(undef, n, n),
        Matrix{Float32}(undef, n, n),
        Matrix{Float32}(undef, p, p),
        # `pyrup!`'s vertical pass writes `(2 * rows, cols)` of its *source*, so the deepest
        # step — from `n/2` square up to `n` square — needs `(n, n/2)`. Sizing this `(n, n)`
        # would leave half of it permanently untouched, and it is not free: one workspace is
        # allocated per chunk per pass, so at 16 chunks the waste is real memory traffic rather
        # than idle address space.
        Matrix{Float32}(undef, n, max(n ÷ 2, p)),
        taps,
        up,
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
subpixel_peak(rw::RefinementWorkspace, surface::AbstractMatrix{Float32},
              radius::Tuple{Int,Int}, upsampling::Integer) =
    subpixel_peak(rw, surface, radius, upsampling, peak_index(surface)...)

"""
    subpixel_peak(rw, surface, radius, upsampling, pi, pj) -> (dx, dy, correlation)

[`subpixel_peak`](@ref) refining about an integer peak the caller has already located.

`(pi, pj)` must be what [`peak_index`](@ref) would return for `surface`; the four-argument form
is this with that call supplied, and is what the tests and benchmarks use. The correlator calls
this one: it has already located the peak to derive the displacement, the boundary flag and the
peak ratio, and refining would otherwise scan the surface again for a peak it holds.
"""
function subpixel_peak(
    rw::RefinementWorkspace, surface::AbstractMatrix{Float32},
    radius::Tuple{Int,Int}, upsampling::Integer, pi_::Int, pj::Int,
)
    nr, nc = size(surface)
    p = size(rw.patch, 1)
    # No refinement asked for, or a surface too small to refine on: the integer peak is the answer.
    # Taken from the supplied location rather than re-scanned, which is the whole point of this form.
    (upsampling == 1 || nr < p || nc < p) &&
        return (_offset_at(pi_, pj, radius)..., @inbounds surface[pi_, pj])
    upsampling <= rw.max_upsampling || throw(ArgumentError(
        "upsampling $upsampling exceeds the workspace maximum " *
        "$(rw.max_upsampling)"))

    rx, ry = radius

    # Clamp the patch to the surface. The reference does the same, and the
    # clamping is why the patch origin has to be tracked separately: the peak is
    # not generally at the patch centre.
    half = p ÷ 2
    i0 = clamp(pi_ - half, 1, max(nr - p + 1, 1))
    j0 = clamp(pj - half, 1, max(nc - p + 1, 1))

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
    level = 0
    cur = @view rw.patch[1:p, 1:p]
    use_a = true
    while factor < upsampling
        level += 1
        n *= 2
        factor *= 2
        # `level` indexes the tap table for this step's source size.
        tp = rw.taps[min(level, length(rw.taps))]
        if use_a
            dst = @view rw.a[1:n, 1:n]
            pyrup!(dst, cur, rw.scratch, tp)
            cur = dst
        else
            dst = @view rw.b[1:n, 1:n]
            pyrup!(dst, cur, rw.scratch, tp)
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

# ---------------------------------------------------------------------------
# Refinement workspace pool
# ---------------------------------------------------------------------------
#
# The same take/give discipline as the correlation pool in `correlate.jl`, sharing its lock:
# both are touched only at chunk boundaries, so one lock costs nothing and two would be two
# things to reason about. See that file for why pooling is keyed on exact geometry.

"""
    AutoRIFT.take_refinement!(upsampling) -> RefinementWorkspace

As [`take_workspace!`](@ref), for the sub-pixel cascade's buffers. Keyed on `upsampling` alone,
since that and the fixed 5x5 patch determine every extent.
"""
function take_refinement!(up::Int)
    rw = lock(WORKSPACE_LOCK) do
        pool = get(REFINEMENT_POOL, up, nothing)
        isnothing(pool) || isempty(pool) ? nothing : pop!(pool)
    end
    return isnothing(rw) ? refinement_workspace(up) : rw
end

"""
    AutoRIFT.give_refinement!(rw)

Return a refinement workspace to the pool.
"""
function give_refinement!(rw::RefinementWorkspace)
    lock(WORKSPACE_LOCK) do
        push!(get!(() -> RefinementWorkspace[], REFINEMENT_POOL, rw.max_upsampling), rw)
    end
    return nothing
end
