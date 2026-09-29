# What becomes of the points a blocked run and an untiled run disagree about?
#
#   julia --project=tools/golden -t 10,1 tools/golden/divergence_fate.jl S1B_IW_SLC__1SDH_20180809 2048 --run 200
#
# A blocked run measures ~5% fewer points than an untiled one on a geogrid (`block_agreement.jl`), and the
# plan's step 0 is to find out whether that matters before anything is changed for it. Two questions, in
# order of how cheaply they settle it:
#
#   1. **What does the *reference* return there?** `autoRIFT_intermediate.nc` is the reference correlator's
#      own `Dx`/`Dy` on the same grid as the capture, so this needs no coordinate mapping. If the reference
#      is nodata at those points then the blocked run agrees with it and the *untiled* run is the outlier —
#      which would make `CORRECTNESS.md` item 1 an agreement-*improving* change rather than the deliberate
#      divergence that file is for, and remove what gates it.
#   2. **Do they carry data in the cropped product?** An ITS_LIVE product is cropped to the valid-data
#      extent, so a point that is nodata there cannot reach a user however the correlator answered.
#
# **The orientation is asserted, not assumed.** The intermediate's arrays are `(x, y)` where the capture's
# grid is `(y, x)`, and a transposed comparison of two displacement fields still looks like a displacement
# field — `tools/ab/README.md` records that trap costing a 0.75 px bias that read as a real defect. So this
# checks agreement over the whole grid first and refuses to report anything if that is not high.

using Printf, Statistics, NCDatasets

include(joinpath(@__DIR__, "correlator.jl"))

argvalue(flag, default) = (i = findfirst(==(flag), ARGS);
                           isnothing(i) ? default : ARGS[i + 1])

nanify(A) = [ismissing(v) ? NaN32 : Float32(v) for v in A]

function main()
    frag, block = ARGS[1], parse(Int, ARGS[2])
    call = parse(Int, argvalue("--run", "100"))
    c = only(cases(frag))

    k = read_capture(c; n = call, mmap = CAPTURE_IMAGERY)
    grid = pointset_from_capture(k)
    kw = kwargs_from_capture(k)
    a, b = k.arrays["in_I1"], k.arrays["in_I2"]

    whole = autorift(b, a, grid; kw...)
    tiled = autorift(b, a, grid; kw..., process_block_size = (block, block))

    path = run_intermediate(run_dir(c, call))
    isnothing(path) && error("no autoRIFT_intermediate.nc under $(run_dir(c, call))")
    refdx = NCDataset(path) do ds
        # `(x, y)` on disk against the grid's `(y, x)`.
        permutedims(nanify(ds["Dx"][:, :]))
    end
    size(refdx) == size(whole.dx) || error(
        "reference Dx is $(size(refdx)) against a grid of $(size(whole.dx)) — not the same grid")

    # The orientation guard. Over points both measured, the untiled run and the reference agree to well
    # under a pixel on every green golden case; a transposed read would show no such agreement.
    both = findall(i -> isfinite(whole.dx[i]) && isfinite(refdx[i]), eachindex(refdx))
    resid = [abs(whole.dx[i] - refdx[i]) for i in both]
    @printf("orientation check: %d points measured by both, |dx| difference median %.4f px, p90 %.4f\n",
            length(both), median(resid), sort(resid)[max(1, round(Int, 0.9 * end))])
    median(resid) < 1.0 || error("untiled and the reference do not agree to a pixel; orientation is wrong")

    lost = findall(i -> isfinite(whole.dx[i]) && !isfinite(tiled.dx[i]), eachindex(whole.dx))
    reffinite = count(i -> isfinite(refdx[i]), lost)
    @printf("\nblocked loses %d points that untiled measured.\n", length(lost))
    @printf("  the reference measured %d of them (%.1f%%) and is nodata at %d (%.1f%%)\n",
            reffinite, 100 * reffinite / max(length(lost), 1),
            length(lost) - reffinite, 100 * (length(lost) - reffinite) / max(length(lost), 1))
    # For scale: how often is the reference nodata where untiled measured, over the whole grid?
    allmeasured = count(i -> isfinite(whole.dx[i]), eachindex(whole.dx))
    refnodata_all = count(i -> isfinite(whole.dx[i]) && !isfinite(refdx[i]), eachindex(whole.dx))
    @printf("  for scale, over the whole grid: untiled measured %d, of which the reference is nodata at %d (%.1f%%)\n",
            allmeasured, refnodata_all, 100 * refnodata_all / max(allmeasured, 1))
    if reffinite > 0
        r = [abs(whole.dx[i] - refdx[i]) for i in lost if isfinite(refdx[i])]
        @printf("  where the reference did measure them, |untiled - reference| median %.3f px, p90 %.3f, max %.3f\n",
                median(r), sort(r)[max(1, round(Int, 0.9 * end))], maximum(r))
    end

    # Split the lost set on whether the point has a position at all, because the two halves call for
    # opposite fixes and an aggregate agreement figure cannot distinguish them.
    #
    # Geogrid marks a point outside the image with `-32767`; `runAutorift` then overwrites it with zero
    # (`testautoRIFT.py:390-391`) before storing `round(xGrid) + 0.5`, so a positionless point reaches the
    # correlator at 0.5 — the scene's outside corner — and a point with a real position lands at a pixel
    # index plus a half. The threshold is therefore below every real point rather than tuned.
    #
    # If the reference measures these too, it is correlating at that corner and agreeing with it means
    # reproducing it, so blocking has to give such a point a window. If the reference is nodata there,
    # untiled is the outlier and making a positionless point permanently unsearchable fixes the blocked
    # disagreement *and* moves both closer to the reference.
    xg = k.arrays["in_xGrid"]
    positionless = i -> Float64(xg[i]) <= 1
    for (label, set) in (("positionless (no real coordinate)", filter(positionless, lost)),
                         ("positioned", filter(!positionless, lost)))
        isempty(set) && (@printf("\n%s: none\n", label); continue)
        nref = count(i -> isfinite(refdx[i]), set)
        @printf("\n%s: %d of %d lost (%.1f%%)\n", label, length(set), length(lost),
                100 * length(set) / length(lost))
        @printf("  the reference measured %d (%.1f%%), nodata at %d\n",
                nref, 100 * nref / length(set), length(set) - nref)
        if nref > 0
            r = [abs(whole.dx[i] - refdx[i]) for i in set if isfinite(refdx[i])]
            @printf("  |untiled - reference| median %.3f px, p90 %.3f, max %.3f\n",
                    median(r), sort(r)[max(1, round(Int, 0.9 * end))], maximum(r))
        end
    end

    # The same split over the whole grid, so the enrichment in the lost set is readable against a base
    # rate rather than on its own.
    npl = count(positionless, eachindex(whole.dx))
    @printf("\nwhole grid: %d of %d points are positionless (%.1f%%); untiled measured %d of them, the reference %d\n",
            npl, length(whole.dx), 100 * npl / length(whole.dx),
            count(i -> positionless(i) && isfinite(whole.dx[i]), eachindex(whole.dx)),
            count(i -> positionless(i) && isfinite(refdx[i]), eachindex(whole.dx)))
    return nothing
end

main()
