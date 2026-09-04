# The batched correlator, as KernelAbstractions kernels.
#
# Vendor-neutral and included by each vendor extension; see the head of `plans.jl`.
#
# ---------------------------------------------------------------------------
# What this is a port of, and what may not change
# ---------------------------------------------------------------------------
#
# `_track_chunk!` in `src/track.jl` is the loop this replaces: one point at a time, each doing a
# gather, a chip mean removal, two summed-area tables, an FFT numerator, a per-shift normalisation,
# a peak scan, a peak ratio, and a subpixel cascade. Here each of those is a kernel over the whole
# batch, and the buffers live on the device between them, so no correlation surface crosses the bus.
#
# The gate is that `dx` and `dy` are **bit-identical** to the CPU path — measured, 0 differences
# over 5400 deliberately contested surfaces. Two consequences bind the code below:
#
#   * **`peak_index`'s tie rule is copied verbatim, not re-derived.** OpenCV's `minMaxLoc` keeps the
#     first strict maximum in *row-major* order while the traversal is column-major, and the extra
#     clause is what reconciles them. A plateau is not exotic — 8-bit imagery produces them
#     routinely — so a different rule would bias every displacement on one, invisibly.
#
#   * **The subpixel cascade's arithmetic is the CPU's, in the same order.** `pyrup!` applies the
#     same weights to the same taps; reassociating them would change the last bit of the upsampled
#     surface and so, occasionally, its argmax.
#
# What is *not* bit-identical is the numerator, because MPSGraph and FFTW reassociate a transform
# differently. Measured, the device is the more accurate of the two — 3.91e-5 against the CPU's
# 6.05e-5, relative to exact arithmetic — because the gather subtracts the window mean and so keeps
# a large DC term out of a `Float32` transform. `correlation` therefore agrees to 1e-5 rather than
# exactly. See `docs/gpu-feasibility.md`.

using KernelAbstractions: @kernel, @index, @Const, get_backend, synchronize, allocate

# ---------------------------------------------------------------------------
# Double-single arithmetic
# ---------------------------------------------------------------------------
#
# `twosum` is *exact*: for any `a`, `b` the pair `(s, e)` satisfies `a + b == s + e` exactly with
# `s == fl(a + b)`. That identity is what lets the summed-area recurrence carry the bits a single
# `Float32` drops, recovering the `Float64` accuracy a device without `Float64` cannot have.
#
# It holds only under round-to-nearest with no reassociation and no FMA contraction of the
# subtractions, which is why each step is written out rather than folded. Verified on the Metal
# device: bit-identical to the host, with the error term nonzero on 97% of entries. That check
# matters because the failure is silent — a compiler that reassociated would zero `lo` and leave a
# kernel that runs and is no better than plain `Float32`.
@inline function ds_twosum(a::Float32, b::Float32)
    s = a + b
    bb = s - a
    e = (a - (s - bb)) + (b - bb)
    return s, e
end

# A pair plus a scalar.
@inline function ds_add(hi::Float32, lo::Float32, b::Float32)
    s, e = ds_twosum(hi, b)
    return ds_twosum(s, lo + e)
end

# A pair plus a pair.
@inline function ds_add2(ah::Float32, al::Float32, bh::Float32, bl::Float32)
    s, e = ds_twosum(ah, bh)
    return ds_twosum(s, (al + bl) + e)
end

# The four-corner box difference on a pair table. `boxsum`'s arithmetic, carried in pairs; the
# subtractions are additions of negations, which is exact for a pair since negation is exact.
@inline function ds_boxsum(Thi, Tlo, i, j, h, w, k)
    @inbounds begin
        sh, sl = ds_add2(Thi[i + h, j + w, k], Tlo[i + h, j + w, k],
                         -Thi[i, j + w, k], -Tlo[i, j + w, k])
        sh, sl = ds_add2(sh, sl, -Thi[i + h, j, k], -Tlo[i + h, j, k])
        sh, sl = ds_add2(sh, sl, Thi[i, j, k], Tlo[i, j, k])
    end
    return sh, sl
end

# ---------------------------------------------------------------------------
# Gather
# ---------------------------------------------------------------------------
#
# One workgroup per point, because both the chip mean and the window mean are reductions over the
# point's own data and every element then needs the result. A workitem-per-element kernel would
# have to either compute the mean redundantly per element or run a second pass; a workgroup can
# reduce once into shared state and reuse it.
#
# Written as a serial loop within the workgroup rather than a tree reduction, and deliberately: the
# reduction order fixes the mean, and the mean enters every element of the chip, so a different
# order would perturb the numerator by more than the transform difference this path is otherwise
# bounded by. `prepare_chip!` accumulates in `Float64` for exactly that reason — "an error here
# biases the whole numerator rather than averaging out" — and a double-single accumulator is what
# stands in for that here.
#
# `chip_rows`/`chip_cols` and `win_rows`/`win_cols` arrive as per-point start indices computed on
# the host from `chip_bounds` and `search_bounds`, so the two most delicate index conventions in the
# package — the asymmetric window and the even chip's half-extent — are not restated here.
@kernel function gather_kernel!(chip, window, cnorm, degenerate, @Const(sec), @Const(ref),
                                @Const(chip_r0), @Const(chip_c0), @Const(win_r0),
                                @Const(win_c0), @Const(rxs), @Const(rys), ch::Int32,
                                cw::Int32)
    k = @index(Global)

    @inbounds begin
        cr0 = chip_r0[k]
        cc0 = chip_c0[k]
        wr0 = win_r0[k]
        wc0 = win_c0[k]

        # **This point's own** window extent, not the pass maximum. The search radius is a per-point
        # field, so a point with a small radius has a correspondingly small window — `search_bounds`
        # derives it from `radius_x[i]`/`radius_y[i]` — and reading the pass-maximum extent from its
        # origin runs off the end of the image. On a device that is an out-of-bounds access rather
        # than a `BoundsError`: it appears as a `KernelException` from whatever later call
        # synchronizes, and only under `--check-bounds=yes`.
        #
        # The asymmetric convention, as `workspace` and `search_bounds` both derive it: the window
        # reaches `radius` one way and `radius - 1` the other, so the surface is an even `2 * radius`.
        wh = ch + Int32(2) * rys[k] - Int32(1)
        ww = cw + Int32(2) * rxs[k] - Int32(1)

        # The chip's mean, in double-single. `n` is exact in `Float32` for every chip size the API
        # admits (a multiple of 4, so the area is well inside 2^24).
        n = Float32(ch) * Float32(cw)
        mh = 0.0f0
        ml = 0.0f0
        for j in 1:cw, i in 1:ch
            mh, ml = ds_add(mh, ml, Float32(sec[cr0 + i - 1, cc0 + j - 1]))
        end
        cmean = (mh + ml) / n

        # Mean-removed chip, and its squared norm. The norm is the denominator's chip factor, so it
        # accumulates in double-single as well — it multiplies every shift.
        qh = 0.0f0
        ql = 0.0f0
        for j in 1:cw, i in 1:ch
            d = Float32(sec[cr0 + i - 1, cc0 + j - 1]) - cmean
            chip[i, j, k] = d
            qh, ql = ds_add(qh, ql, d * d)
        end
        sq = qh + ql

        # Zero within rounding rather than exactly zero, as `prepare_chip!` requires: a chip of
        # nearly-identical values has a norm at the scale of the rounding error, and dividing by it
        # would amplify noise into a spurious peak.
        #
        # `eps(Float64)`, matching the CPU, and **not** `eps(Float32)` even though the sum is formed
        # in `Float32` pairs. The threshold names which chips carry no signal, which is a property of
        # the imagery rather than of the accumulator: the two constants differ by nine orders of
        # magnitude (2.3e-13 against 1.2e-4 at chip 32), so `eps(Float32)` would reject chips with a
        # real if small variance — measured, it discarded 95 points of one scene that the CPU
        # measures. The pair accumulator is what makes the tighter constant meaningful here.
        degenerate[k] = !(sq > Float32(eps(Float64)) * n)
        cnorm[k] = sqrt(sq)

        # The window, with *its own* mean removed. This is what makes `Float32` tables and a
        # `Float32` transform sound: the numerator is invariant to it because the chip is
        # mean-removed (`Σ T′ = 0`), and the variance about the window mean is shift-invariant. So
        # this changes no exact-arithmetic quantity and removes the DC term that would otherwise set
        # the rounding scale for both.
        wn = Float32(wh) * Float32(ww)
        sh = 0.0f0
        sl = 0.0f0
        for j in 1:ww, i in 1:wh
            sh, sl = ds_add(sh, sl, Float32(ref[wr0 + i - 1, wc0 + j - 1]))
        end
        wmean = (sh + sl) / wn
        for j in 1:ww, i in 1:wh
            window[i, j, k] = Float32(ref[wr0 + i - 1, wc0 + j - 1]) - wmean
        end
    end
end

# ---------------------------------------------------------------------------
# Summed-area tables
# ---------------------------------------------------------------------------
#
# Two passes, one workitem per column then one per row, which is what makes this parallel at all:
# `integral_both!`'s single traversal carries a running sum down each column *and* adds the previous
# column, so it is sequential in both axes. Splitting it into a down-columns prefix sum and then an
# across-rows prefix sum gives `winx` independent workitems in the first pass and `winy` in the
# second.
#
# Both sums in one kernel per pass, as `integral_both!` fuses them, and for the same reason: every
# caller needing the sum needs the squares, and one traversal reads the window once.
#
# `wh`/`ww` are the *maximum* extents, which size the launch; each point then walks only its own,
# derived from its radius exactly as `gather_kernel!` derives it. A point with a smaller radius
# leaves the rest of its column untouched, and nothing reads it: the row pass and the box sums are
# bounded by the same per-point extent.
@kernel function colsum_kernel!(Shi, Slo, Qhi, Qlo, @Const(window), @Const(rxs), @Const(rys),
                                ch::Int32, cw::Int32, wh::Int32, ww::Int32)
    # One workitem per (column, point).
    idx = @index(Global)
    nj = Int32(ww)
    k = (idx - 1) ÷ nj + 1
    j = (idx - 1) % nj + 1

    @inbounds if j <= cw + Int32(2) * rxs[k] - Int32(1)
        mh = ch + Int32(2) * rys[k] - Int32(1)
        rh = 0.0f0
        rl = 0.0f0
        qh = 0.0f0
        ql = 0.0f0
        for i in 1:mh
            v = window[i, j, k]
            rh, rl = ds_add(rh, rl, v)
            # `v * v` is one rounding, and its error term is recoverable with an FMA. Left out: the
            # squares are the better-conditioned of the two sums, since every term has the same
            # sign and the centered window keeps them small.
            qh, ql = ds_add(qh, ql, v * v)
            Shi[i, j, k] = rh
            Slo[i, j, k] = rl
            Qhi[i, j, k] = qh
            Qlo[i, j, k] = ql
        end
    end
end

# The row pass, into the `(winy + 1, winx + 1)` tables `boxsum` reads. The extra leading row and
# column are zero, which is what lets the box sum read four corners with no branch — see
# `integral!`.
@kernel function rowsum_kernel!(Shi, Slo, Qhi, Qlo, @Const(Chi), @Const(Clo),
                                @Const(Dhi), @Const(Dlo), @Const(rxs), @Const(rys),
                                ch::Int32, cw::Int32, wh::Int32, ww::Int32)
    idx = @index(Global)
    ni = Int32(wh) + Int32(1)
    k = (idx - 1) ÷ ni + 1
    i = (idx - 1) % ni + 1

    @inbounds if i <= ch + Int32(2) * rys[k]
        mw = cw + Int32(2) * rxs[k] - Int32(1)
        # The zero first column, and the whole zero first row when this workitem owns it.
        Shi[i, 1, k] = 0.0f0
        Slo[i, 1, k] = 0.0f0
        Qhi[i, 1, k] = 0.0f0
        Qlo[i, 1, k] = 0.0f0
        if i == 1
            for j in 1:(mw + 1)
                Shi[1, j, k] = 0.0f0
                Slo[1, j, k] = 0.0f0
                Qhi[1, j, k] = 0.0f0
                Qlo[1, j, k] = 0.0f0
            end
        else
            rh = 0.0f0
            rl = 0.0f0
            qh = 0.0f0
            ql = 0.0f0
            for j in 1:mw
                rh, rl = ds_add2(rh, rl, Chi[i - 1, j, k], Clo[i - 1, j, k])
                qh, ql = ds_add2(qh, ql, Dhi[i - 1, j, k], Dlo[i - 1, j, k])
                Shi[i, j + 1, k] = rh
                Slo[i, j + 1, k] = rl
                Qhi[i, j + 1, k] = qh
                Qlo[i, j + 1, k] = ql
            end
        end
    end
end

# ---------------------------------------------------------------------------
# Transform buffers
# ---------------------------------------------------------------------------
#
# Zero-fill and place the window and the reversed chip. Correlation is convolution with a reflected
# kernel, so the chip goes in reversed and at the origin, which puts the valid output at offset
# `(ch, cw)` — the layout `_numerators_fft!` uses and the read below assumes.
#
# The whole padded buffer is written, not just the used corner: a pooled workspace holds whatever
# the previous batch left, and the transform reads all of it.
@kernel function fftfill_kernel!(a, b, @Const(window), @Const(chip), @Const(rxs), @Const(rys),
                                 ch::Int32, cw::Int32, fy::Int32, fx::Int32)
    idx = @index(Global)
    n = Int32(fy) * Int32(fx)
    k = (idx - 1) ÷ n + 1
    r = (idx - 1) % n
    i = r % fy + 1
    j = r ÷ fy + 1

    @inbounds begin
        # This point's own window extent, as the gather wrote it; beyond it the buffer is the zero
        # padding the cyclic transform needs anyway.
        mh = ch + Int32(2) * rys[k] - Int32(1)
        mw = cw + Int32(2) * rxs[k] - Int32(1)
        a[i, j, k] = (i <= mh && j <= mw) ? window[i, j, k] : 0.0f0
        b[i, j, k] = (i <= ch && j <= cw) ? chip[ch - i + 1, cw - j + 1, k] : 0.0f0
    end
end

# The spectral multiply, in place into the window's spectrum.
@kernel function specmul_kernel!(Fa, @Const(Fb))
    i = @index(Global)
    @inbounds Fa[i] *= Fb[i]
end

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
#
# One workitem per (shift, point): the surface element is the numerator over the product of the
# chip norm and the window's standard deviation under the chip, and the second factor is four table
# reads whatever the chip size.
#
# The per-point radius may be smaller than the workspace's, in which case this point's surface is
# the top-left `2*ry_k` by `2*rx_k` corner and the rest is left alone — the same "smaller points
# take views of the corner" arrangement `CorrelationWorkspace` uses. The peak scan honours the same
# per-point extent, so untouched elements are never read.
@kernel function normalize_kernel!(surface, @Const(num), @Const(Shi), @Const(Slo),
                                   @Const(Qhi), @Const(Qlo), @Const(cnorm), @Const(rxs),
                                   @Const(rys), ch::Int32, cw::Int32, nr::Int32, nc::Int32,
                                   fy::Int32, fx::Int32)
    idx = @index(Global)
    n = Int32(nr) * Int32(nc)
    k = (idx - 1) ÷ n + 1
    r = (idx - 1) % n
    i = r % nr + 1
    j = r ÷ nr + 1

    @inbounds begin
        # Outside this point's own surface extent. Zeroed rather than skipped so a pooled buffer
        # cannot leak a previous batch's values into a `maximum` a caller might take.
        if i > 2 * rys[k] || j > 2 * rxs[k]
            surface[i, j, k] = 0.0f0
        else
            nn = Float32(ch) * Float32(cw)
            s1h, s1l = ds_boxsum(Shi, Slo, i, j, ch, cw, k)
            s2h, s2l = ds_boxsum(Qhi, Qlo, i, j, ch, cw, k)

            # `Σ(W - W̄)² = ΣW² - (ΣW)²/n`, with `s1^2` also formed in double-single: squaring is
            # where the leading term would be lost, and this difference is the one `src/integral.jl`
            # requires `Float64` for on the CPU. The window is already centered, so the two terms
            # are no longer large and nearly equal — that is what makes `Float32` sound here, and
            # the pair is what closes the remaining accumulation error.
            p = s1h * s1h
            pe = fma(s1h, s1h, -p) + 2.0f0 * s1h * s1l
            vh, vl = ds_add2(s2h, s2l, -(p / nn), -(pe / nn))
            # Clamped at zero for the reason `_correlate_surface!` clamps: rounding can make a
            # low-contrast window's variance slightly negative, and `sqrt` of that is `NaN`.
            #
            # Then clamped again, *to* zero, below the magnitude at which the variance is
            # indistinguishable from it — the same `eps(Float64) * n` bound `prepare_chip!` applies
            # to the chip's own norm, and applied here for exactly the same reason. A `> 0` test
            # alone is not enough on a summed-area difference: a genuinely constant window leaves a
            # residue of ~1e-9 rather than exact zero, which passes `> 0` and divides a nonzero
            # numerator by it. Measured on a scene with a constant patch, that produced a surface
            # reaching 1.56 at 32 of 2500 shifts and moved the reported displacement 26 px.
            #
            # The CPU reaches exact zero here because its tables are `Float64` over a raw window, so
            # a constant window's `ΣW²` and `(ΣW)²/n` cancel bit for bit. That is luck rather than
            # design — the same cancellation is what makes its *numerator* unreliable at these
            # shifts — so the bound is stated rather than relied upon.
            #
            # Relative to `ΣW²`, the quantity the difference was formed from, rather than an absolute
            # constant: the window's scale is the caller's data and an absolute bound would be a
            # threshold on brightness. `n * eps` is the accumulated rounding of a sum of `n` terms,
            # which is what a variance indistinguishable from zero looks like at this width.
            wvar = vh + vl
            wvar = wvar > nn * eps(Float32) * abs(s2h) ? wvar : 0.0f0

            # The inverse transform is unnormalised, so the `1/(fy*fx)` lands here rather than in a
            # pass over the whole padded buffer — only `nr*nc` of `fy*fx` values are ever read.
            scale = 1.0f0 / (Float32(fy) * Float32(fx))
            den = cnorm[k] * sqrt(wvar)
            surface[i, j, k] = den > 0 ?
                (num[i + ch - 1, j + cw - 1, k] * scale) / den : 0.0f0
        end
    end
end

# ---------------------------------------------------------------------------
# Peak location, boundary, peak ratio
# ---------------------------------------------------------------------------
#
# One workitem per point. `peak_index`, `peak_at_boundary` and `peak_ratio` fused, as
# `_track_chunk!` fuses them and for the same reason: every point needs all three from one located
# peak, and locating it four times over is 6% of a point on the CPU.
#
# **The tie rule is `peak_index`'s, verbatim.** Traversal is column-major, along memory, while the
# tie-breaking is row-major to match OpenCV's `minMaxLoc`; the extra clause reconciles them. This is
# the one kernel where a "simplification" would silently bias every displacement on a plateau, and
# 8-bit imagery produces plateaus routinely.
@kernel function peak_kernel!(peak_i, peak_j, corr, ppr, @Const(surface), @Const(rxs),
                              @Const(rys), exclusion::Int32)
    k = @index(Global)

    @inbounds begin
        nr = 2 * rys[k]
        nc = 2 * rxs[k]
        best_i = Int32(1)
        best_j = Int32(1)
        best = -Inf32
        for j in 1:nc, i in 1:nr
            v = surface[i, j, k]
            if v > best || (v == best && (i < best_i || (i == best_i && j < best_j)))
                best = v
                best_i = Int32(i)
                best_j = Int32(j)
            end
        end
        peak_i[k] = best_i
        peak_j[k] = best_j
        corr[k] = surface[best_i, best_j, k]

        # The primary-to-secondary peak ratio: the peak divided by the largest rival outside the
        # exclusion box, in `Float32` for the reason `peak_ratio` records — so the two paths differ
        # only by the surface each was handed, not by the arithmetic applied to it. `test/gpu.jl`
        # bounds what remains.
        #
        # One running maximum, where the CPU splits into four to break the loop-carried dependency.
        # Deliberate: that split buys 3x there and nothing here, because a workitem per point already
        # gives the device thousands of independent chains to interleave — measured, the pass holds at
        # 50.4 us/point either way. The comparison form still matches the CPU's exactly, which is what
        # the agreement gate needs.
        second = -Inf32
        cnt = Int32(0)
        for c in 1:nc, r in 1:nr
            if !(abs(r - best_i) <= exclusion && abs(c - best_j) <= exclusion)
                v = surface[r, c, k]
                # `v > second`, not `max`: `max` propagates a `NaN` where the comparison skips it,
                # and the CPU form skips. A surface can carry `NaN` from a masked pixel, so the two
                # would disagree exactly where it matters.
                v > second && (second = v)
                cnt += Int32(1)
            end
        end
        h = surface[best_i, best_j, k]
        # The three cases `peak_ratio` documents, in its order: nothing outside the box to compare
        # against, no peak to take a ratio of, and no positive rival to divide by.
        if cnt == 0 || !(h > 0)
            ppr[k] = NaN32
        elseif second <= 0
            ppr[k] = Inf32
        else
            ppr[k] = h / second
        end
    end
end

# ---------------------------------------------------------------------------
# Subpixel refinement
# ---------------------------------------------------------------------------
#
# The largest share of a point at the geometry ITS_LIVE runs — 78 us against 17 for the whole
# surface at chip 16, and fixed in chip size, because the cascade upsamples a 5x5 patch to 320x320
# whatever the surface was. That is why it is on the device rather than left on the host: a GPU that
# accelerated only the transform would be capped at 1.2x there.
#
# ---------------------------------------------------------------------------
# One workitem per output element, and why the obvious arrangement fails
# ---------------------------------------------------------------------------
#
# The parallelism is *within* a point as well as across points: each cascade step is one launch over
# every (element, point) pair, and the steps are separated by launches because step `k + 1` reads
# what step `k` wrote.
#
# The natural alternative — one workitem per point, walking its own cascade serially — is
# catastrophically slower, and the measurement is worth recording because the arrangement looks
# right. At 64x upsampling the cascade writes ~205,000 elements per point, and on one workitem that
# measured **1012 us per point**, against 31 us for every other stage of the pass combined, i.e. 97%
# of the whole correlator and 7x slower end to end than the CPU. The cost tracked the element count
# exactly (4x per doubling of `up`: 2.7, 8.8, 13.7, 47.2, 188.2, 1011.8 us at 2x through 64x) and was
# **flat in the point count** (1628, 1018, 1002, 995 us/pt at 64, 256, 1024, 4096 points) — the
# signature of serial work on one thread rather than of launch overhead. A GPU thread is far slower
# than a CPU thread; the only thing a device offers is width, and a kernel one element wide declines
# it.
#
# So the scratch is per (element, point) rather than per point: `(n, n, batch)` for the ping-pong and
# `(n, n/2, batch)` for the intermediate, which is why `_refine_batch!` tiles over points to bound
# the footprint — at 64x one point's scratch is 820 kB.
#
# The arithmetic is `pyrup!`'s, in the same order and with the same weights. Not for elegance: the
# gate is bit-identical `dx`/`dy`, and reassociating a 5-tap convolution changes the last bit of the
# upsampled surface and so, occasionally, its argmax.
const W_C = 6.0f0 / 8      # centre tap, even output position
const W_S = 1.0f0 / 8      # side taps, even output position
const W_H = 4.0f0 / 8      # both taps, odd output position

# `reflect101` on 0-based positions: reflect `p` into `0:(n-1)` without repeating the edge, which is
# OpenCV's `BORDER_REFLECT_101`. Reflection preserves parity, which is what lets the cascade skip
# the injected zeros before reflecting rather than after.
#
# Every literal is written `Int32(...)`: a bare `2` is an `Int64`, and `2m` on an `Int32` would
# promote the whole expression, which on a device is a `MethodError` at compile time rather than a
# silent widening.
@inline function ka_reflect101_0(p::Int32, n::Int32)
    n == Int32(1) && return Int32(0)
    m = n - Int32(1)
    while p < Int32(0) || p > m
        p < Int32(0) && (p = -p)
        p > m && (p = Int32(2) * m - p)
    end
    return p
end

# Source indices contributing to 1-based output position `k` of an axis of length `n`, and how many.
# An even upsampled position sits on a source sample and takes three taps; an odd one sits between
# two and takes two.
#
# **Reflection happens in the upsampled coordinate space, not the source's**, which is the subtle
# part and is why this mirrors `_pyrup_taps` rather than reflecting source indices. The difference
# shows at the trailing edge, where it makes the last sample's weight accumulate rather than fold
# onto an interior neighbour — established by probing OpenCV with unit impulses, since nothing
# documents it.
@inline function ka_pyrup_taps(k::Int32, n::Int32)
    p = k - Int32(1)
    if iseven(p)
        c = p ÷ Int32(2) + Int32(1)
        m = ka_reflect101_0(p - Int32(2), Int32(2) * n) ÷ Int32(2) + Int32(1)
        q = ka_reflect101_0(p + Int32(2), Int32(2) * n) ÷ Int32(2) + Int32(1)
        return m, c, q, Int32(3)
    else
        c = ka_reflect101_0(p - Int32(1), Int32(2) * n) ÷ Int32(2) + Int32(1)
        q = ka_reflect101_0(p + Int32(1), Int32(2) * n) ÷ Int32(2) + Int32(1)
        return c, c, q, Int32(2)
    end
end

# One cascade step, as a kernel over every (output element, point). `dst` becomes `src` at twice the
# extent in each axis.
#
# Two passes through `tmp`, as `pyrup!` does — a single pass costs up to nine loads per output sample
# where separating costs three in each. The separation changes the order of the multiplies against a
# fused form, but agreement with the *CPU* is what matters and this is the CPU's form.
#
# Two launches rather than one, because the horizontal pass reads what the vertical pass wrote: a
# single kernel would need a barrier across workgroups, which KernelAbstractions does not offer and
# which a device does not provide within one dispatch.
@kernel function pyrup_v_kernel!(tmp, @Const(src), sh::Int32, sw::Int32, nb::Int32)
    # `@index(Global)` is an `Int64`, so it is narrowed here rather than left to promote: an `Int64`
    # reaching `ka_pyrup_taps`'s `Int32` signature is a `MethodError` at device-compile time.
    idx = Int32(@index(Global)) - Int32(1)
    rows = Int32(2) * sh
    n = rows * sw
    t = idx ÷ n + Int32(1)
    r = idx % n
    i = r % rows + Int32(1)
    j = r ÷ rows + Int32(1)

    @inbounds if t <= nb
        im, i0, ip, ni = ka_pyrup_taps(i, sh)
        tmp[i, j, t] = ni == Int32(3) ?
            W_C * src[i0, j, t] + W_S * (src[im, j, t] + src[ip, j, t]) :
            W_H * (src[i0, j, t] + src[ip, j, t])
    end
end

@kernel function pyrup_h_kernel!(dst, @Const(tmp), sh::Int32, sw::Int32, nb::Int32)
    idx = Int32(@index(Global)) - Int32(1)
    rows = Int32(2) * sh
    cols = Int32(2) * sw
    n = rows * cols
    t = idx ÷ n + Int32(1)
    r = idx % n
    i = r % rows + Int32(1)
    j = r ÷ rows + Int32(1)

    @inbounds if t <= nb
        jm, j0, jp, nj = ka_pyrup_taps(j, sw)
        dst[i, j, t] = nj == Int32(3) ?
            W_C * tmp[i, j0, t] + W_S * (tmp[i, jm, t] + tmp[i, jp, t]) :
            W_H * (tmp[i, j0, t] + tmp[i, jp, t])
    end
end

# The clamped patch about each point's integer peak, into the cascade's first buffer.
#
# `i0`/`j0` are returned to the host as well, because the final displacement needs them and
# recomputing them there would put the clamp in two places.
#
# `refinable` marks the points whose surface is at least `patch` across, which is `subpixel_peak`'s
# own precondition — it returns the integer peak unrefined otherwise. **Both halves are needed and
# for different reasons.** A point with a smaller surface cannot supply a `patch`-sized
# neighbourhood, so reading one is an out-of-bounds device access: the search radius is a per-point
# field, and a radius below `patch ÷ 2` is a legal value a caller may set (`min_search_radius`
# defaults to 6 but a `PointSet` carries its own). And the flag has to reach the *peak* kernel too,
# or a point skipped here would be assigned whatever the stale buffer held.
@kernel function refine_patch_kernel!(bufa, i0s, j0s, refinable, @Const(surface),
                                      @Const(peak_i), @Const(peak_j), @Const(rxs),
                                      @Const(rys), patch::Int32, offset::Int32, nb::Int32)
    idx = Int32(@index(Global)) - Int32(1)
    n = patch * patch
    t = idx ÷ n + Int32(1)
    r = idx % n
    i = r % patch + Int32(1)
    j = r ÷ patch + Int32(1)

    @inbounds if t <= nb
        k = t + offset
        nr = Int32(2) * rys[k]
        nc = Int32(2) * rxs[k]
        ok = nr >= patch && nc >= patch
        # Clamp the patch to the surface, as the reference does. The peak is then not generally at
        # the patch centre, which is why the origin is tracked rather than assumed.
        half = patch ÷ Int32(2)
        i0 = clamp(peak_i[k] - half, Int32(1), max(nr - patch + Int32(1), Int32(1)))
        j0 = clamp(peak_j[k] - half, Int32(1), max(nc - patch + Int32(1), Int32(1)))
        if i == Int32(1) && j == Int32(1)
            i0s[t] = i0
            j0s[t] = j0
            refinable[t] = ok
        end
        # Zero-filled rather than skipped where the point cannot be refined: the cascade runs over
        # the whole tile regardless, and reading a pooled buffer's previous contents would put
        # unrelated values through it.
        bufa[i, j, t] = ok ? surface[i0 + i - Int32(1), j0 + j - Int32(1), k] : 0.0f0
    end
end

# The argmax of each point's upsampled patch, in two passes.
#
# **`peak_index`'s tie rule, verbatim**: a strictly greater value wins, and an equal value wins only
# if it precedes the incumbent in *row-major* order, while the traversal is column-major. That is what
# makes the result the element OpenCV's `minMaxLoc` would return, and a plateau in an upsampled patch
# is not exotic — the whole reason `peak_index` carries the clause is that 8-bit imagery produces
# them routinely.
#
# The rule is order-dependent, which is what makes this a two-pass reduction rather than one workitem
# per point. That arrangement was measured and is the wrong one: scanning 320x320 on a single workitem
# cost **113 us per point**, against 14 for the whole convolution cascade beneath it — 84% of the
# refinement and more than half the entire correlator. The same "a GPU thread is slow, only width
# helps" lesson as the cascade itself, in the kernel that had looked too small to matter.
#
# Splitting it is exact rather than approximate, because the rule is a *total order* on
# `(value, row, column)`: a candidate beats the incumbent iff it is greater in value, or equal in
# value and earlier in row-major order. A total order composes, so reducing per column and then over
# the column winners gives the element a single ordered scan would — which the `ndrange`-1 case makes
# testable, since the two paths must then agree bitwise.
#
# One workitem per (column, point). Column-major within the column, so each workitem walks contiguous
# memory, exactly as `peak_index` does and for the same reason.
@kernel function refine_colmax_kernel!(vals, ris, rjs, @Const(cur), n::Int32, nb::Int32)
    idx = Int32(@index(Global)) - Int32(1)
    t = idx ÷ n + Int32(1)
    j = idx % n + Int32(1)

    @inbounds if t <= nb
        bi = Int32(1)
        best = -Inf32
        for i in Int32(1):n
            v = cur[i, j, t]
            # Within one column the column index is fixed, so the tie clause reduces to "an equal
            # value earlier in the column wins" — which `>` alone already gives, since the first such
            # value is the incumbent.
            if v > best
                best = v
                bi = i
            end
        end
        vals[j, t] = best
        ris[j, t] = bi
        rjs[j, t] = j
    end
end

# The column winners combined, and the displacement the overall argmax stands for. `n` entries per
# point, so this is the short pass — 320 against 102,400.
@kernel function refine_peak_kernel!(dx, dy, corr, @Const(vals), @Const(ris), @Const(rjs),
                                     @Const(i0s), @Const(j0s), @Const(refinable),
                                     @Const(surface), @Const(peak_i), @Const(peak_j),
                                     @Const(rxs), @Const(rys), n::Int32, factor::Int32,
                                     offset::Int32, nb::Int32)
    t = Int32(@index(Global))
    @inbounds if t <= nb
        k = t + offset
        if !refinable[t]
            # A surface too small to refine on: the integer peak is the answer, which is what
            # `subpixel_peak` returns in the same case.
            pi_ = peak_i[k]
            pj = peak_j[k]
            dx[k] = Float32(pj - rxs[k] - Int32(1))
            dy[k] = Float32(pi_ - rys[k] - Int32(1))
            corr[k] = surface[pi_, pj, k]
        else
            bi = Int32(1)
            bj = Int32(1)
            best = -Inf32
            for j in Int32(1):n
                v = vals[j, t]
                i = ris[j, t]
                # The full clause here, where columns are being compared: a strictly greater value
                # wins, and an equal one only if it is earlier in row-major order.
                if v > best || (v == best && (i < bi || (i == bi && rjs[j, t] < bj)))
                    best = v
                    bi = i
                    bj = rjs[j, t]
                end
            end
            # Position within the upsampled patch, back to surface coordinates and then to a
            # displacement about the surface origin. The `- 1` terms convert 1-based indices to
            # offsets before dividing.
            row = Float32(bi - Int32(1)) / Float32(factor) + Float32(i0s[t] - Int32(1))
            col = Float32(bj - Int32(1)) / Float32(factor) + Float32(j0s[t] - Int32(1))
            dx[k] = col - Float32(rxs[k])
            dy[k] = row - Float32(rys[k])
            corr[k] = best
        end
    end
end

# The integer peak as the answer, for `NoRefine` or a surface too small to refine on.
@kernel function refine_none_kernel!(dx, dy, corr, @Const(surface), @Const(peak_i),
                                     @Const(peak_j), @Const(rxs), @Const(rys), nb::Int32)
    k = Int32(@index(Global))
    @inbounds if k <= nb
        pi_ = peak_i[k]
        pj = peak_j[k]
        dx[k] = Float32(pj - rxs[k] - Int32(1))
        dy[k] = Float32(pi_ - rys[k] - Int32(1))
        corr[k] = surface[pi_, pj, k]
    end
end
