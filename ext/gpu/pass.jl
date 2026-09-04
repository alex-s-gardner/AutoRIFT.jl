# The batched pass: `_track_chunk!`'s loop, as a sequence of kernel launches over batches of points.
#
# Vendor-neutral and included by each vendor extension; see the head of `plans.jl`.
#
# Structure, and why it is this shape. `_track_chunk!` walks points one at a time, and each point's
# eight steps are separated by no synchronisation because they are on one thread. Here each step is
# a kernel over the whole batch, so the sequence is the same and the parallelism is across points.
# Only the *outputs* return to the host — five vectors of one value per point — so no correlation
# surface crosses the bus, which is what makes the whole thing worth doing.

# Whether a pass has enough searchable points for the device to win.
#
# Not a policy knob: measured. A batched transform is *slower* than the equivalent FFTW calls below
# a few hundred points, because per-launch overhead dominates — 0.16x at 64 points and a 28-point
# transform. The sparse search deliberately zeroes most of the grid, so a coarse pass or a late
# chip-size level routinely lands here; falling back is the common case rather than a corner.
_gpu_worth_it(nsearch::Int) = nsearch >= GPU_MIN_BATCH

# One pass on the device. The signature `_dispatch_pass!` hands over, plus the backend.
#
# `ref`/`sec`/`okmask` are host arrays: the caller has already padded and filtered them, and the
# gather reads windows out of them. They are uploaded once per pass rather than once per batch,
# since every batch reads from the same pair.
function _gpu_pass!(backend_obj, out::AutoRIFT.DisplacementField, ref, sec, okmask,
                    pts::AutoRIFT.PointSet{1}, chip::AutoRIFT.Extent,
                    radius::AutoRIFT.Extent,
                    up::Int, p::AutoRIFT.Params, measure::AutoRIFT.SimilarityMeasure)
    measure isa AutoRIFT.ZNCC || throw(ArgumentError(
        "the GPU path implements `ZNCC` only, but this pass uses " *
        "`$(nameof(typeof(measure)))`. `NCC` differs by one denominator term and `Coherence` " *
        "needs complex transforms; neither has a device kernel yet. Use `backend = :cpu` for " *
        "these measures, or `similarity = :zncc`."))

    # Which points this pass will actually correlate, and their geometry. Collected on the host: the
    # searchability test, the bounds checks and the valid-pixel test are per point and cheap, and
    # doing them here means the device sees a dense batch rather than one with holes.
    sel, chip_r0, chip_c0, win_r0, win_c0, prxs, prys = _gpu_select(ref, okmask, pts)
    n = length(sel)
    n == 0 && return out

    for i in sel
        out.searched[i] = true
    end

    dref = _to_device(backend_obj, ref)
    dsec = _to_device(backend_obj, sec)

    # The pool keys on plain tuples, so the extents are unwrapped here rather than at every
    # kernel launch below, which takes the two axes as separate `Int32`s regardless.
    chipx, chipy = chip.X, chip.Y
    rx, ry = radius.X, radius.Y
    batch = gpu_batch_size((chipx, chipy), (rx, ry))
    ws = take_gpu_workspace!(dref, (chipx, chipy), (rx, ry), batch)
    try
        for lo in 1:batch:n
            hi = min(lo + batch - 1, n)
            _gpu_batch!(backend_obj, ws, out, dref, dsec, pts, sel, chip_r0, chip_c0,
                        win_r0, win_c0, prxs, prys, lo, hi, chipx, chipy, rx, ry, up)
        end
    finally
        # Back to the pool even if a batch threw, for the reason `_track_chunk!`'s `finally` exists:
        # a leaked workspace is not a crash, it silently turns the pool back into per-batch device
        # allocation, and that would be invisible.
        give_gpu_workspace!(ws)
    end
    return out
end

# The points this pass will correlate, with the per-point geometry the kernels need.
#
# Every test here is `_track_chunk!`'s, in the same order and for the same reasons: a zero radius
# means excluded, a chip outside the image is skipped, and a chip with no valid pixel is padding
# rather than imagery. The bounds are taken from `chip_bounds`/`search_bounds` rather than
# recomputed, so the asymmetric window convention and the even chip's half-extent stay in one place.
function _gpu_select(ref, okmask, pts::AutoRIFT.PointSet{1})
    sel = Int[]
    chip_r0 = Int32[]
    chip_c0 = Int32[]
    win_r0 = Int32[]
    win_c0 = Int32[]
    prxs = Int32[]
    prys = Int32[]
    @inbounds for i in eachindex(pts)
        AutoRIFT.issearchable(pts, i) || continue
        chip_rows, chip_cols = AutoRIFT.chip_bounds(pts, i)
        win_rows, win_cols = AutoRIFT.search_bounds(pts, i)
        checkbounds(Bool, ref, chip_rows, chip_cols) || continue
        checkbounds(Bool, ref, win_rows, win_cols) || continue
        AutoRIFT._any_valid(okmask, chip_rows, chip_cols) || continue
        push!(sel, i)
        push!(chip_r0, first(chip_rows))
        push!(chip_c0, first(chip_cols))
        push!(win_r0, first(win_rows))
        push!(win_c0, first(win_cols))
        push!(prxs, pts.radius_x[i])
        push!(prys, pts.radius_y[i])
    end
    return sel, chip_r0, chip_c0, win_r0, win_c0, prxs, prys
end

# One batch, start to finish. Every kernel is launched on the same backend queue, so they order
# themselves; the single `synchronize` before the read-back is what waits for all of them.
function _gpu_batch!(backend_obj, ws::GPUWorkspace, out::AutoRIFT.DisplacementField, dref,
                     dsec, pts::AutoRIFT.PointSet{1}, sel, chip_r0, chip_c0, win_r0, win_c0,
                     prxs, prys, lo::Int, hi::Int, chipx::Int, chipy::Int, rx::Int, ry::Int,
                     up::Int)
    nb = hi - lo + 1
    be = get_backend(ws.chip)

    winx = chipx + 2rx - 1
    winy = chipy + 2ry - 1
    fy, fx = ws.fftsize
    nr, nc = 2ry, 2rx

    # Per-point geometry for this batch. Uploaded as `Int32` vectors rather than passed as a struct
    # of arrays, because the kernels index them by point and nothing else needs the layout.
    d_cr = _to_device(backend_obj, view(chip_r0, lo:hi))
    d_cc = _to_device(backend_obj, view(chip_c0, lo:hi))
    d_wr = _to_device(backend_obj, view(win_r0, lo:hi))
    d_wc = _to_device(backend_obj, view(win_c0, lo:hi))
    d_rx = _to_device(backend_obj, view(prxs, lo:hi))
    d_ry = _to_device(backend_obj, view(prys, lo:hi))

    # Views of the workspace, so a partial final batch runs kernels over exactly its own points
    # rather than over stale ones. Every buffer's trailing dimension is the batch.
    chip = view(ws.chip, :, :, 1:nb)
    window = view(ws.window, :, :, 1:nb)

    # The per-point radii reach the gather, which derives each point's own window extent from them
    # rather than using the pass maximum: a point with a smaller radius has a smaller window, and
    # reading the maximum extent from its origin runs off the image.
    gather_kernel!(be)(chip, window, view(ws.cnorm, 1:nb), view(ws.degenerate, 1:nb),
                       dsec, dref, d_cr, d_cc, d_wr, d_wc, d_rx, d_ry,
                       Int32(chipy), Int32(chipx); ndrange = nb)

    colsum_kernel!(be)(ws.colsum.hi, ws.colsum.lo, ws.colsqsum.hi, ws.colsqsum.lo,
                       window, d_rx, d_ry, Int32(chipy), Int32(chipx), Int32(winy),
                       Int32(winx); ndrange = nb * winx)

    rowsum_kernel!(be)(ws.isum.hi, ws.isum.lo, ws.isqsum.hi, ws.isqsum.lo,
                       ws.colsum.hi, ws.colsum.lo, ws.colsqsum.hi, ws.colsqsum.lo,
                       d_rx, d_ry, Int32(chipy), Int32(chipx), Int32(winy), Int32(winx);
                       ndrange = nb * (winy + 1))

    fftfill_kernel!(be)(ws.fbuf_a, ws.fbuf_b, window, chip, d_rx, d_ry, Int32(chipy),
                        Int32(chipx), Int32(fy), Int32(fx); ndrange = nb * fy * fx)

    # The transform is planned for the *full* batch shape, so a partial final batch transforms its
    # unused tail as well. Cheaper than a second plan: `fftfill_kernel!` writes only `1:nb` and the
    # pooled tail holds whatever the previous batch left, which is finite and never read back —
    # `normalize_kernel!` and the peak scan both run over `1:nb` alone.
    pf = gpu_rfft_plan(ws)
    pb = gpu_irfft_plan(ws)
    mul!(ws.fspec_a, pf, ws.fbuf_a)
    mul!(ws.fspec_b, pf, ws.fbuf_b)
    specmul_kernel!(be)(ws.fspec_a, ws.fspec_b; ndrange = length(ws.fspec_a))
    mul!(ws.fbuf_a, pb, ws.fspec_a)

    normalize_kernel!(be)(view(ws.surface, :, :, 1:nb), ws.fbuf_a,
                          ws.isum.hi, ws.isum.lo, ws.isqsum.hi, ws.isqsum.lo,
                          view(ws.cnorm, 1:nb), d_rx, d_ry,
                          Int32(chipy), Int32(chipx), Int32(nr), Int32(nc),
                          Int32(fy), Int32(fx); ndrange = nb * nr * nc)

    peak_kernel!(be)(view(ws.peak_i, 1:nb), view(ws.peak_j, 1:nb), view(ws.corr, 1:nb),
                     view(ws.ppr, 1:nb), view(ws.surface, :, :, 1:nb), d_rx, d_ry,
                     Int32(AutoRIFT.PEAK_EXCLUSION); ndrange = nb)

    _refine_batch!(backend_obj, ws, nb, d_rx, d_ry, up)

    synchronize(be)

    # Back to the host: five short vectors, one value per point.
    h_dx = Array(view(ws.dx, 1:nb))
    h_dy = Array(view(ws.dy, 1:nb))
    h_corr = Array(view(ws.corr, 1:nb))
    h_ppr = Array(view(ws.ppr, 1:nb))
    h_deg = Array(view(ws.degenerate, 1:nb))
    h_pi = Array(view(ws.peak_i, 1:nb))
    h_pj = Array(view(ws.peak_j, 1:nb))

    @inbounds for b in 1:nb
        i = sel[lo + b - 1]
        # A chip with no texture carries no information about displacement, so it is left as no
        # measurement — `searched` stays true, `dx`/`dy` stay `NaN`. The reference reports the
        # search-window corner here, which over featureless terrain is a corner-pinned bias.
        h_deg[b] && continue

        # Both quality outputs are zero where the peak lay against the search boundary: the
        # displacement is a lower bound with no recoverable sub-pixel part, so neither the peak
        # height nor its ratio to the best rival describes a usable measurement. Zeroing them means any positive
        # threshold on either rejects the point without the caller knowing the condition exists.
        railed = h_pi[b] == 1 || h_pj[b] == 1 ||
                 h_pi[b] == 2 * prys[lo + b - 1] || h_pj[b] == 2 * prxs[lo + b - 1]

        # Back to displacement about the grid point: the surface is centred on the window, which
        # is centred on the point, and the chip was offset by the prior.
        out.dx[i] = Float32(h_dx[b] + pts.dx_prior[i])
        out.dy[i] = Float32(h_dy[b] + pts.dy_prior[i])
        out.correlation[i] = railed ? 0.0f0 : h_corr[b]
        out.peak_ratio[i] = railed ? 0.0f0 : h_ppr[b]
    end
    return out
end

# The subpixel cascade, in tiles over points, with each cascade step a launch over every
# (element, point).
#
# Two things shape this, and the first was settled by measurement rather than by design. The
# parallelism has to be *within* a point as well as across points: a kernel that walked one point's
# cascade serially measured **1012 us per point** against 31 us for every other stage of the pass
# combined — see the note at the head of the refinement section in `kernels.jl`. So each step is one
# launch over `n * n * tile` elements, and consecutive steps are separate launches because step
# `k + 1` reads what step `k` wrote.
#
# And the footprint is per (element, point), which is two orders of magnitude larger than the
# correlation's: at 64x upsampling one point's ping-pong is 2 x 320² `Float32` = 820 kB, against
# ~215 kB for a whole correlation point at chip 32.
#
# **The tile wants to be as large as the budget allows**, because the cost is per *launch* rather than
# per element and the cascade is 12 launches per tile whatever the tile holds. Measured at 64x: the
# same cascade costs 17.6 us per point at a 131-point tile and 13.5 at 2048, while a single deepest
# step runs 6-12x a bare store over the same elements — so the elements are nearly free and the tiles
# are what is paid for. A tile too small was worth 164 us per point against 14.
function _refine_batch!(backend_obj, ws::GPUWorkspace, nb::Int, d_rx, d_ry, up::Int)
    be = get_backend(ws.chip)
    patch = AutoRIFT.REFINE_PATCH

    if up == 1
        refine_none_kernel!(be)(view(ws.dx, 1:nb), view(ws.dy, 1:nb), view(ws.corr, 1:nb),
                                view(ws.surface, :, :, 1:nb), view(ws.peak_i, 1:nb),
                                view(ws.peak_j, 1:nb), d_rx, d_ry, Int32(nb); ndrange = nb)
        return nothing
    end

    n = patch * up
    # Bytes per point: the two ping-pong buffers and the vertical pass's intermediate.
    per = 4 * (2 * n * n + n * max(n ÷ 2, patch))
    tile = max(1, min(nb, REFINE_MEMORY_BUDGET[] ÷ max(per, 1)))

    bufa = _dev(ws.chip, Float32, n, n, tile)
    bufb = _dev(ws.chip, Float32, n, n, tile)
    # The vertical pass writes `(2 * rows, cols)` of its source, so the deepest step needs
    # `(n, n ÷ 2)`. Sized to that rather than `(n, n)`, which would leave half permanently untouched.
    tmp = _dev(ws.chip, Float32, n, max(n ÷ 2, patch), tile)
    i0s = _dev(ws.chip, Int32, tile)
    j0s = _dev(ws.chip, Int32, tile)
    # Column winners for the two-pass argmax: `n` per point, against `n^2` scanned.
    cvals = _dev(ws.chip, Float32, n, tile)
    cris = _dev(ws.chip, Int32, n, tile)
    crjs = _dev(ws.chip, Int32, n, tile)
    # Which points have a surface at least `patch` across. The search radius is a per-point field, so
    # a point too small to refine is a legal input rather than a corner case — and reading a
    # `patch`-sized neighbourhood from a smaller surface is an out-of-bounds device access.
    refinable = _dev(ws.chip, Bool, tile)

    for lo in 1:tile:nb
        hi = min(lo + tile - 1, nb)
        m = hi - lo + 1
        off = Int32(lo - 1)

        refine_patch_kernel!(be)(bufa, i0s, j0s, refinable, view(ws.surface, :, :, 1:nb),
                                 view(ws.peak_i, 1:nb), view(ws.peak_j, 1:nb), d_rx, d_ry,
                                 Int32(patch), off, Int32(m); ndrange = m * patch * patch)

        # Ping-pong, doubling each step. `bufa` holds the patch, so the first step writes `bufb`;
        # `in_a` tracks which buffer holds the *current* value, and it is that buffer the scan reads.
        # Deriving the scan's source from a flag flipped after the write — rather than from one naming
        # the destination — is what keeps the two in step at every depth: at 64x there are six
        # doublings, so a scan of the wrong buffer reads the second-to-last level, which is a
        # plausible surface at half the resolution and shifts every displacement by one pixel.
        sz = patch
        factor = 1
        in_a = true
        while factor < up
            src = in_a ? bufa : bufb
            dst = in_a ? bufb : bufa
            pyrup_v_kernel!(be)(tmp, src, Int32(sz), Int32(sz), Int32(m);
                                ndrange = m * 2 * sz * sz)
            pyrup_h_kernel!(be)(dst, tmp, Int32(sz), Int32(sz), Int32(m);
                               ndrange = m * 4 * sz * sz)
            in_a = !in_a
            sz *= 2
            factor *= 2
        end

        cur = in_a ? bufa : bufb
        refine_colmax_kernel!(be)(cvals, cris, crjs, cur, Int32(sz), Int32(m);
                                  ndrange = m * sz)
        refine_peak_kernel!(be)(view(ws.dx, 1:nb), view(ws.dy, 1:nb), view(ws.corr, 1:nb),
                                cvals, cris, crjs, i0s, j0s, refinable,
                                view(ws.surface, :, :, 1:nb), view(ws.peak_i, 1:nb),
                                view(ws.peak_j, 1:nb), d_rx, d_ry, Int32(sz),
                                Int32(factor), off, Int32(m); ndrange = m)
    end
    return nothing
end
