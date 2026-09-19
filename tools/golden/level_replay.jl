# Replay one pyramid level's fine pass on the reference's own captured inputs, over a window of its lattice.
#
#   julia --project=tools/golden -t 8 tools/golden/level_replay.jl NISAR_L1_PR_RSLC --chip 768
#   julia --project=tools/golden -t 8 tools/golden/level_replay.jl NISAR_L1_PR_RSLC --chip 768 --half 48
#
# **Why this exists.** `correlator.jl` compares the *merged* field, which cannot say whether a coarse-level
# disagreement came from the measurement, the hole fill, the read-back or the merge. Measured separately on
# the NISAR L1 case, the merge reproduces our own read-back to 0.008 px rms while the two sides' coarse
# *lattice* values agree at only 2.52% — so the interesting question is at the level's own pass, and this is
# where it is asked. `GATES.md` records the measurement.
#
# **A window, because the point is to iterate.** A whole-grid `correlator.jl` run on a NISAR case costs
# 10 minutes and 49 GiB; a 65² window of the chip-768 lattice costs about a second. The default window sits
# in the top-right third of the lattice, which is where the residual lives on both NISAR cases.
#
# **It gates on itself before reporting anything.** Priming the pass with the reference's own answer must
# return that answer — a correlator pointed at the truth and searching a window around it has to find the
# peak it was aimed at. Three earlier rounds of this comparison reported numbers from a harness that would
# have failed that check, at rms 1.9-3.6 px against a 0.25 px target, and the failure looked like a
# correlator finding. A row printed after `GATE FAILED` is a statement about this script, not about
# AutoRIFT.jl.
#
# **The prior is the input this needs and the capture has to carry.** `arImgDisp_*` is handed `Dx00`, the
# cell mean of `Dx0` over `1/Scale` cells resized to the lattice (`autoRIFT.py:161-179`) — not the
# full-resolution `in_Dx0`. `capture.py` records it per level as `dx0`/`dy0`; a capture predating that is
# detected and reported rather than silently replayed at a zero prior, since a wrong prior searches the
# wrong window and reads as a measurement difference.
include(joinpath(@__DIR__, "correlator.jl"))
using AutoRIFT: params, extent, ImagePair, WholeScene, run_pass, PointSet
using Statistics, Printf

# The constant that converts a captured level grid back into `in_xGrid`'s frame, per axis.
#
# `INTER_AREA` at an integer scale is a plain block mean and the reference snaps an even chip's grid with
# `round(x + 0.5) - 0.5`, so the level's grid is reconstructible from `in_xGrid` up to one constant — the
# pad `arImgDisp_*` added. Errors rather than guessing if the difference is not a single value, since every
# number downstream assumes the two grids describe the same ground.
function _level_pad(k::Capture, rec, stride::Integer)
    gx = Float64.(k.arrays["in_xGrid"]); gy = Float64.(k.arrays["in_yGrid"])
    function blockmean(A, shape)
        stride == 1 && return A
        H, W = shape
        out = Matrix{Float64}(undef, H, W)
        for cj in 1:W, ci in 1:H
            acc = 0.0; n = 0
            for j in ((cj - 1) * stride + 1):min(cj * stride, size(A, 2)),
                i in ((ci - 1) * stride + 1):min(ci * stride, size(A, 1))
                acc += A[i, j]; n += 1
            end
            out[ci, cj] = round(acc / n + 0.5) - 0.5
        end
        return out
    end
    pxg = Float64.(rec.arrays["xgrid"]); pyg = Float64.(rec.arrays["ygrid"])
    ox = unique(pxg .- blockmean(gx, size(pxg)))
    oy = unique(pyg .- blockmean(gy, size(pyg)))
    length(ox) == 1 && length(oy) == 1 || error(
        "the captured level grid for chip $(Int(rec.chip_size[1])) is not a constant shift of an " *
        "`INTER_AREA` reconstruction of `in_xGrid`: x offsets $(first(ox, 4)), y offsets $(first(oy, 4)). " *
        "The replay cannot place this level's nodes, so no comparison against it would be meaningful.")
    return only(ox), only(oy)
end

"""
    level_replay(c; chip, n = 100, half = 32) -> NamedTuple

Run AutoRIFT.jl's fine pass for chip size `chip` on the reference's captured lattice, priors and search
radii, over a `2half+1` window, and compare against the reference's own `lvl*_dx`/`_dy`.

Returns the two fields, the window, and the gate's residual, so a caller can map or re-analyse without
correlating again.
"""
function level_replay(c::GoldenCase; chip::Integer, n::Integer = 100, half::Integer = 32)
    k = read_capture(c; n)
    kw = kwargs_from_capture(k)
    p = params(; kw...)
    chip0 = Int(k.scalars["ChipSize0X"])

    fines = filter(x -> x.kind == "fine", k.levels)
    hits = filter(x -> Int(x.chip_size[1]) == chip, fines)
    isempty(hits) && throw(ArgumentError(
        "no fine pass at chip $chip in this capture; it has " *
        join((string(Int(x.chip_size[1])) for x in fines), ", ")))
    rec = only(hits)
    csy = Int(rec.chip_size[2])

    haskey(rec.arrays, "dx0") || error(
        "this capture predates `capture.py` recording the per-level prior (`lvl*_dx0`). The pass cannot " *
        "be replayed without it: `arImgDisp_*` is handed `Dx00`, a cell mean of `Dx0` over the level's " *
        "cells, and a replay at the wrong prior searches a different window — which reads as a " *
        "measurement difference. Re-capture with `intermediate.jl $(first(c.product, 24)) --force`.")

    # **The captured level grid is in that level's own PADDED frame, and the pad differs per level.**
    # `arImgDisp_*` pads both images by `Px = max(ChipSizeX)/2 + max(SearchLimitX + |Dx0|) + 2` and then
    # shifts the grid it was handed by `Px + 0.5` *in place* (`arImgDisp_u:78-90`), so what the wrapper
    # records is post-shift. The pad is a function of that level's chip and search extent, so it is not one
    # constant: measured on L1 it is +2231.5/+1149.5 at stride 1, +2605.5/+1228.5 at stride 2,
    # +788.5/+274.5 at 4 and +933.5/+371.5 at 8.
    #
    # Treating every level as unpadded — adding only the index base — puts the replay hundreds of pixels
    # from where the reference correlated, which reads as a total loss of agreement rather than as a frame
    # error: at chip 192 our answer correlated **−0.03** with the reference's over 5,661 nodes. So the pad
    # is recovered here and removed, putting the grid back in `in_xGrid`'s frame where `pointset_from_capture`
    # and the imagery both live.
    #
    # Recovered by reconstruction rather than assumed: `INTER_AREA` at an integer scale is a plain block
    # mean, and the reference's even-chip snap is `round(x + 0.5) - 0.5` (`autoRIFT.py:109-125`), so the
    # level's grid is computable from `in_xGrid` up to exactly that constant. The reconstruction is checked
    # to differ from the capture by a *single* value per axis; anything else means this is not the frame
    # relationship it claims to be, and the replay refuses rather than reporting a shifted comparison.
    xg = rec.arrays["xgrid"]; yg = rec.arrays["ygrid"]
    raw = rec.arrays["dx"];   rawy = rec.arrays["dy"]
    padx, pady = _level_pad(k, rec, chip ÷ chip0)
    rx, ry = _level_search_limits(rec.arrays["searchx"], rec.arrays["searchy"], k)
    # `Dy00` is recorded as the correlator was *handed* it, which is still cartesian-Y: the internal
    # `Dy0 = -Dy0` is the callee's first act (`arImgDisp_u:76`). Same negation `pointset_from_capture`
    # applies to `in_Dy0`, and for the same reason.
    dxp = Float64.(rec.arrays["dx0"]); dyp = .-Float64.(rec.arrays["dy0"])

    # **The window is centred on the level's own measured nodes, not on its lattice.** A level's coverage is
    # a small part of its lattice — chip 768 measures 4,585 of 83,808 nodes — and it moves up the pyramid, so
    # a window at a fixed fraction of the lattice lands on data at one level and on empty grid at the next:
    # centred on the lattice this script reported `both-measured 0` at chips 192 and 384 while working at 768.
    # The centroid of the reference's own measured nodes keeps the window on the population being compared.
    nlr, nlc = size(xg)
    meas_all = .!isnan.(raw)
    any(meas_all) || error("the reference measured nothing at chip $chip in this capture")
    idxs = findall(meas_all)
    ci = clamp(round(Int, median(first.(Tuple.(idxs)))), half + 1, max(nlr - half, half + 1))
    cj = clamp(round(Int, median(last.(Tuple.(idxs)))), half + 1, max(nlc - half, half + 1))
    rows = max(ci - half, 1):min(ci + half, nlr)
    cols = max(cj - half, 1):min(cj + half, nlc)
    w(A) = A[rows, cols]

    a = k.arrays["in_I1"]; b = k.arrays["in_I2"]
    runner = WholeScene(ImagePair(b, a))
    # The level's own subpixel method: the reference varies `OverSampleRatio` per chip size.
    subp = p.subpixel[min(lastindex(p.subpixel), Int(log2(chip ÷ chip0)) + 1)]

    wxg = w(xg); wyg = w(yg); wraw = w(raw); wrawy = w(rawy)
    wrx = w(rx); wry = w(ry)
    csx = fill(Int(chip), size(wxg)); csyy = fill(csy, size(wxg))

    function pass(px, py; yshift = 0.0)
        g = PointSet(Float64.(wxg) .- padx .+ 1, Float64.(wyg) .- pady .+ 1 .+ yshift, wrx, wry,
                     Float64.(px), Float64.(py), csx, csyy, csx, csx)
        return run_pass(runner, g, p, first(p.similarity), subp)
    end

    # **`lvl*_dy` is cartesian-Y, so OUR `dy` is negated before comparing.** The wrapper records what
    # `arImgDisp_*` *returned*, and its last act is `Dy = -Dy`, converting back from matrix-Y
    # ("Y from down being positive to up being positive"); AutoRIFT.jl's `dy` is row-positive. All four
    # sign combinations were scored on the chip-768 level of the L1 case rather than reasoned about:
    # negating our output gives `dy` rms 0.90 px against 4.82 for the same sign, while the prior's sign
    # changes almost nothing (0.8971 against 0.9019) because the per-level prior is only a few pixels.
    # So the negation belongs on the *output*, and that is where it goes.
    ourdy(f) = .-Float64.(f.dy)

    # **Footprint-edge nodes are excluded, and the map is why.** At chip 768 the nodes disagreeing by more
    # than a pixel lie on the swath's diagonal boundary in a one-cell-wide line — every one of them, which
    # a percentile cannot show and a heat map shows at a glance (`figs/nisar_l1_replay_chip768.png`). A
    # 768x416 chip centred a cell inside the boundary is still part nodata fill, so the two sides are
    # breaking a partly-empty correlation differently and neither is measuring ground. Excluding one ring
    # of 8-connected boundary nodes takes the `dx` residual from rms 2.31 to 0.35 and the count beyond 1 px
    # from 88 of 1732 to **2 of 1476**, while the interior median stays at one quantization step.
    interior = falses(size(wraw))
    meas = .!isnan.(wraw)
    for j in axes(meas, 2), i in axes(meas, 1)
        meas[i, j] || continue
        ok = true
        for dj in -1:1, di in -1:1
            (di == 0 && dj == 0) && continue
            ii, jj = i + di, j + dj
            if ii < firstindex(meas, 1) || jj < firstindex(meas, 2) ||
               ii > lastindex(meas, 1) || jj > lastindex(meas, 2) || !meas[ii, jj]
                ok = false
                break
            end
        end
        interior[i, j] = ok
    end

    # The gate: primed with the reference's own answer, in the convention the comparison uses.
    prime(A) = map(v -> isnan(v) ? 0.0 : Float64(round(v)), A)
    fg = pass(prime(wraw), prime(.-Float64.(wrawy)))
    gboth = (.!isnan.(fg.dx)) .& meas .& interior
    gres = count(gboth) == 0 ? NaN :
           sqrt(mean(abs2, vcat(Float64.(fg.dx[gboth]) .- Float64.(wraw[gboth]),
                                ourdy(fg)[gboth] .- Float64.(wrawy[gboth]))))

    f = pass(w(dxp), w(dyp))
    both = (.!isnan.(f.dx)) .& meas .& interior
    return (; field = f, rows, cols, gate_rms = gres, both, interior, our_dy = ourdy(f),
            ref_dx = wraw, ref_dy = Float64.(wrawy), chip, chip_y = csy)
end

function main(args)
    name = isempty(args) ? "NISAR_L1_PR_RSLC" : args[1]
    geti(flag, default) = (i = findfirst(==(flag), args);
                           i === nothing ? default : parse(Int, args[i + 1]))
    chip = geti("--chip", 768)
    half = geti("--half", 32)
    run = geti("--run", 100)
    c = only(cases(name))
    r = level_replay(c; chip, n = run, half)

    @printf("%s  chip %dx%d  window rows %d:%d cols %d:%d  both-measured %d\n",
            first(c.product, 24), r.chip, r.chip_y, first(r.rows), last(r.rows),
            first(r.cols), last(r.cols), count(r.both))
    if !(r.gate_rms <= 0.1)
        @printf("GATE FAILED: primed with the reference's own answer the replay is rms %.4f px, not <= 0.1.\n",
                r.gate_rms)
        println("The rows below are a statement about this harness, not about AutoRIFT.jl.")
    else
        @printf("gate passed: primed with the reference's answer, rms %.4f px\n", r.gate_rms)
    end
    for (lbl, ours, ref) in (("dx", r.field.dx, r.ref_dx), ("dy", r.our_dy, r.ref_dy))
        d = Float64.(ours[r.both]) .- Float64.(ref[r.both])
        isempty(d) && continue
        @printf("  %s  med=%+8.4f  |.| p50=%.4f p75=%.4f p90=%.4f p95=%.4f p99=%.4f  max=%.2f  exact=%.2f%%\n",
                lbl, median(d), quantile(abs.(d), 0.5), quantile(abs.(d), 0.75),
                quantile(abs.(d), 0.9), quantile(abs.(d), 0.95), quantile(abs.(d), 0.99),
                maximum(abs.(d)), 100 * count(iszero, d) / length(d))
    end
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
