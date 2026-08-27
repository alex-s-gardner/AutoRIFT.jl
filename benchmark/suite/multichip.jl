# Multi-chip-size orchestration, per level and per step.
#
# Measured per step and not just in total, because the steps have very different characters:
# correlation is compute-bound and parallel, while outlier rejection, hole filling, and the
# merge are memory-bandwidth-bound and largely serial. That serial fraction is what caps
# intra-pair thread scaling, so quantifying it is what makes the choice between intra-pair and
# pair-level parallelism an informed one rather than a guess.
#
# Note on naming: these levels differ in *chip size*, not in image resolution. The imagery is
# never downsampled — the reference's `cv2.resize` calls act on the grid arrays, not on `I1`
# and `I2` — so this is a nested multi-chip-size grid rather than an image pyramid. The Laplace
# subpixel solver in `correlate/subpixel` is the one genuine pyramid in the algorithm.

let g = addgroup!(SUITE, "multichip")

    for n in (512, 1024)
        ref = bench_texture((n, n); seed = 1)
        # Coherent motion, so the coarse pass keeps most of the grid and the levels do their
        # work. An uncorrelated pair measures the early-exit path instead.
        sec = circshift(ref, (-4, 6))
        pair = AutoRIFT.ImagePair(ref, sec)
        prepared = AutoRIFT.replace_nonfinite(
            AutoRIFT.preprocess(pair, AutoRIFT.Highpass(; width = 5)))
        p = AutoRIFT.params(; chip_size = 32, search_radius = 25, threaded = false)
        grid = AutoRIFT.gridpoints((n, n), 32; chip_size = 128, search_radius = 25)

        # The whole orchestration, on already-prepared imagery — so this isolates the levels
        # from the filtering that `endtoend` includes.
        g["correlate_multichip c32 $(n)x$(n)"] = @benchmarkable AutoRIFT.correlate_multichip(
            $prepared, $grid, $p)

        # One level, which is what the loop repeats. `wanted` all true asks it to attempt every
        # point, the state the first (finest) level is in.
        wanted = trues(size(grid))
        g["level c32 $(n)x$(n)"] = @benchmarkable AutoRIFT.chipsize_level(
            $prepared, $grid, $p, 32, $wanted)

        # The coarse pass alone: a strided subset correlated to decide where the fine pass is
        # worth attempting. Its cost is what the sparse search trades against, so the ratio to
        # the level above it is the figure of merit.
        #
        # Only at 1024. A 512 scene at 32 px spacing is an 11x11 grid, whose strided subset is
        # a handful of points — microseconds, dominated by setup, and it would report the
        # measurement floor rather than the cost.
        if n == 1024
            cs = AutoRIFT.extent(32)
            pts = AutoRIFT._level_points(grid, p, cs, wanted)
            # A `WholeScene` runner, because that is how a whole-scene pass is executed — the same
            # `_coarse_mask` serves a blocked run through a `Blocked` runner instead.
            #
            # `measure` is positional and has no default — the level's measure is passed explicitly,
            # since `p.similarity` is a tuple and a single level cannot read it without knowing which
            # level it is. Kept in step with `chipsize_level`'s call rather than relying on a default.
            runner = AutoRIFT.WholeScene(prepared)
            g["coarse pass c32 $(n)x$(n)"] = @benchmarkable AutoRIFT._coarse_mask(
                $runner, $pts, $p, $cs, AutoRIFT.measure_at($p, 1))
        end
    end

    # The serial steps, at one representative size. These are the ones that bound thread scaling
    # within a pair, so they are worth their own numbers rather than being folded into a level
    # timing.
    #
    # Measured on a field with real gaps: a quarter of the scene is decorrelated, which is what
    # a cloud edge or a shadowed slope does. Note the gaps do not come from the correlation
    # failing — ZNCC finds a confident peak in decorrelated texture too, so all 900 points come
    # back "measured" and it is the outlier filter that turns 80 of them into gaps.
    #
    # `_reject_and_fill!` is measured as one step because that is the only way it runs: it ends
    # by calling `_fill_holes!`, so timing the fill separately measures an already-filled field
    # and reports 1 us. Splitting them would need the filter's output before the fill, which is
    # not a state the pipeline ever holds.
    let n = 1024
        ref = bench_texture((n, n); seed = 1)
        sec = circshift(ref, (-4, 6))
        gap = (n ÷ 4):(n ÷ 2)
        sec[gap, gap] .= bench_texture((length(gap), length(gap)); seed = 99)
        pair = AutoRIFT.ImagePair(ref, sec)
        p = AutoRIFT.params(; chip_size = 32, search_radius = 25, threaded = false)
        pts = AutoRIFT.gridpoints((n, n), 32; chip_size = 32, search_radius = 25)
        measured = AutoRIFT.track(pair, pts, p)

        g["reject+fill $(n)x$(n)"] = @benchmarkable AutoRIFT._reject_and_fill!(
            d, $pts, $p) setup = (d = deepcopy($measured))
    end
end
