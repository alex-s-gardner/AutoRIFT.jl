# What each term contributes to the halo, per chip-size level and per search-radius class.
#
# `AutoRIFT.halo` returns one `Extent` for a whole run: the maximum of `chip_size_max/2 + radius +
# |prior| + 2 + filter_reach + level_centre_offset` over every level and every point. `BlockLayout`
# carries that one figure, `restrict` passes each block's read window through unchanged, and
# `block_buffers` sizes to the largest window in the layout — so every pass reads a window sized for
# the coarsest chip and the widest radius in the grid, whatever the pass itself needs.
#
# This prints what the same arithmetic gives when the maxima are taken per level and per radius class,
# which is what says whether a narrower window is available and how much it is worth. Arithmetic
# rather than a measurement: the terms are all in `Params` and the grid, so nothing has to be read.
#
#   julia --project=. tools/ab/halo_terms.jl
#
# The configuration is the NISAR L1 RSLC one `docs/src/explanation/memory.md` records, whose recorded
# whole-grid halo is 2736x1500 px. The prior term is per point and the granule is not needed to see
# what it must have contributed: everything else here is computable, so the residual is the prior.

using Printf
using AutoRIFT
using AutoRIFT: params, chip_sizes, _level_decimation, filter_reach

const CHIP = (96, 52)
const CHIP_MAX = (768, 416)
const SPACING = 48
const RADIUS = (1905, 830)          # the geogrid's own maxima; the median over searchable points is 34, 20
const HALO_RECORDED = (2736, 1500)
# The recorded block shape with the lowest peak on this granule, so a window below is comparable to a
# figure in the sweep rather than to a size nothing was measured at.
const BLOCK = (2816, 1536)
const TASKS = 10
# `BlockBuffers` holds nine block-sized arrays. 18 bytes per pixel is the total across all nine for a
# `UInt8` pair — two `UInt8` planes, three `Float32` and four `Bool`.
const BYTES_PER_PIXEL = 18

# The centre offset a *single* level contributes, where `_level_centre_offset` returns the maximum over
# all of them. A decimated level correlates at its cells' centres, which reach further than its grid
# points; an undecimated level reaches no further, so its offset is zero.
function level_offset(p, chip)
    stride = _level_decimation(p, chip)
    stride <= 1 && return (0, 0)
    half = (stride - 1) / 2
    return (ceil(Int, half * p.grid_spacing.X + 0.5), ceil(Int, half * p.grid_spacing.Y + 0.5))
end

buffer_bytes(window) = BYTES_PER_PIXEL * window * TASKS

function main()
    p = params(; chip_size = CHIP, chip_size_max = CHIP_MAX, grid_spacing = SPACING,
               search_radius = RADIUS, preprocess = :none)
    reach = filter_reach(p.preprocess)
    levels = chip_sizes(p)
    offsets = [level_offset(p, c) for c in levels]
    ox, oy = maximum(first, offsets), maximum(last, offsets)

    # The prior the recorded halo implies, given every other term is computable.
    prior_x = HALO_RECORDED[1] - (CHIP_MAX[1] ÷ 2 + RADIUS[1] + 2 + reach + ox)
    prior_y = HALO_RECORDED[2] - (CHIP_MAX[2] ÷ 2 + RADIUS[2] + 2 + reach + oy)

    @printf("levels %s   filter_reach %d   centre offset over all levels %dx%d   implied |prior| %dx%d\n",
            join(("$(c.X)x$(c.Y)" for c in levels), ", "), reach, ox, oy, prior_x, prior_y)
    @printf("halo(p) %dx%d against the recorded whole-grid %dx%d\n\n",
            AutoRIFT.halo(p).X, AutoRIFT.halo(p).Y, HALO_RECORDED...)

    println("Per level, at the whole-grid radius maximum. The window is what a $(BLOCK[1])x$(BLOCK[2]) block reads.")
    println("  chip      stride  offset      halo         window        buffers, $TASKS tasks")
    for (chip, off) in zip(levels, offsets)
        hx = chip.X ÷ 2 + RADIUS[1] + prior_x + 2 + reach + off[1]
        hy = chip.Y ÷ 2 + RADIUS[2] + prior_y + 2 + reach + off[2]
        window = (BLOCK[1] + 2hx) * (BLOCK[2] + 2hy)
        @printf("  %3dx%-3d   %4d   %4dx%-4d  %5dx%-5d  %6.1f Mpx    %5.2f GiB\n",
                chip.X, chip.Y, _level_decimation(p, chip), off[1], off[2], hx, hy,
                window / 1e6, buffer_bytes(window) / 2^30)
    end

    println()
    println("Per level and per radius class. A class cap on the power-of-two ladder `_radius_bucket`")
    println("already rounds to is answer-preserving: a point of radius r <= 2^k buckets to")
    println("min(nextpow2(r), 2^k) = nextpow2(r), which is what the whole-grid cap gives it too.")
    println("The block is the class's own halo where that exceeds 1024 px, since a block below its halo is rejected.")
    println("  chip        cap     halo         block          window        buffers, $TASKS tasks")
    for (chip, off) in zip(levels, offsets), cap in (64, 256, 1024, RADIUS[1])
        capy = min(cap, RADIUS[2])
        px = cap == RADIUS[1] ? prior_x : min(prior_x, cap)
        py = cap == RADIUS[1] ? prior_y : min(prior_y, capy)
        hx = chip.X ÷ 2 + cap + px + 2 + reach + off[1]
        hy = chip.Y ÷ 2 + capy + py + 2 + reach + off[2]
        bx, by = max(1024, hx), max(1024, hy)
        window = (bx + 2hx) * (by + 2hy)
        @printf("  %3dx%-3d   %5d   %5dx%-5d  %5dx%-5d  %6.2f Mpx    %5.2f GiB\n",
                chip.X, chip.Y, cap, hx, hy, bx, by, window / 1e6,
                buffer_bytes(window) / 2^30)
    end
end

main()
