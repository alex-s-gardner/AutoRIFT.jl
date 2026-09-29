# Which stage of the level machinery makes a blocked run disagree with an untiled one?
#
#   julia --project=tools/golden -t 10,1 tools/golden/block_bisect.jl S1B_IW_SLC__1SDH_20180809 2048 --run 200
#
# `block_agreement.jl` says how a blocked run differs and where. This says *which stage* it enters at, by
# removing them one at a time. Three things above the per-block pass can move points, and each is
# independently switchable:
#
#   * **the chip-size cascade** — one level's result decides which points the next attempts, so any
#     difference at one level is amplified by every level after it. Collapsing `chip_size_max` to
#     `chip_size` leaves a single level. `subpixel` names one method per level and has to collapse too.
#   * **the coarse gate** — `_coarse_mask` dilates its evidence into a mask that zeroes the radius of every
#     point outside it, so a difference there removes whole regions. `coarse_stride = 1` samples the coarse
#     pass at every point, which makes the mask cover everything and stops it selecting.
#   * **the outlier filter and hole fill** — `reject_outliers` runs twice per level, inside `_coarse_mask`
#     and again on the assembled fine field. `outliers = :none` removes both.
#
# On S1B this isolates it to the coarse gate: one chip size with the gate not selecting agrees exactly,
# with the outlier filter still on. `coarse_span.jl` then names the mechanism.

using Printf

include(joinpath(@__DIR__, "correlator.jl"))

argvalue(flag, default) = (i = findfirst(==(flag), ARGS);
                           isnothing(i) ? default : ARGS[i + 1])

function compare(a, b, grid, kw, label, block)
    whole = autorift(b, a, grid; kw...)
    tiled = autorift(b, a, grid; kw..., process_block_size = (block, block))
    fin(A, i) = isfinite(A[i])
    lost = count(i -> fin(whole.dx, i) && !fin(tiled.dx, i), eachindex(whole.dx))
    gained = count(i -> !fin(whole.dx, i) && fin(tiled.dx, i), eachindex(whole.dx))
    # `isequal`, so `-0.0` counts as a difference and `NaN` does not.
    moved = count(i -> fin(whole.dx, i) && fin(tiled.dx, i) &&
                       !isequal(whole.dx[i], tiled.dx[i]), eachindex(whole.dx))
    nwhole = count(isfinite, whole.dx)
    @printf("  %-44s untiled %7d  blocked %7d  lost %6d (%5.2f%%)  gained %5d  moved %6d\n",
            label, nwhole, count(isfinite, tiled.dx), lost, 100 * lost / max(nwhole, 1), gained, moved)
    flush(stdout)
    return nothing
end

function main()
    length(ARGS) >= 2 || error("usage: block_bisect.jl <product-fragment> <block-px> [--run N]")
    frag, block = ARGS[1], parse(Int, ARGS[2])
    call = parse(Int, argvalue("--run", "100"))

    c = only(cases(frag))
    k = read_capture(c; n = call, mmap = CAPTURE_IMAGERY)
    grid = pointset_from_capture(k)
    kw = kwargs_from_capture(k)
    a, b = k.arrays["in_I1"], k.arrays["in_I2"]
    @printf("%s\n  block %d px, chip %s -> %s\n",
            c.product, block, string(kw.chip_size), string(kw.chip_size_max))

    compare(a, b, grid, kw, "as it runs", block)
    compare(a, b, grid, merge(kw, (; outliers = :none)), "outlier filter off", block)
    one = merge(kw, (; chip_size_max = kw.chip_size, subpixel = first(kw.subpixel)))
    compare(a, b, grid, one, "one chip size", block)
    compare(a, b, grid, merge(one, (; coarse_stride = 1)), "one chip size, gate not selecting", block)
    compare(a, b, grid, merge(one, (; outliers = :none)), "one chip size, outlier filter off", block)
    return nothing
end

main()
