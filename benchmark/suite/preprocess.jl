# Pre-correlation filters. Populated at M3.
#
# Reported as Mpixel/sec against the corresponding cv2 call, since these run over
# the full image rather than the grid and so scale with scene size rather than
# with point count.
#
# `Highpass` first: it is the reference driver's default and therefore the path
# that matters most. Also covers the validity mask each filter produces, which is
# not merely bookkeeping — it decides which pixels are searched at all, so its
# cost is unavoidable and its correctness is load-bearing.
let g = addgroup!(SUITE, "preprocess")
    # Reported per scene size, since these scale with image area rather than with grid
    # point count. Names match tools/python_ref/bench_python.py so compare.jl can print
    # the speedup against cv2.
    for n in SCENE_SIZES
        img = bench_texture((n, n); seed = 1)
        mask = trues(n, n)
        for w in (5, 21)     # 5 is the default; 21 is used for Sentinel-1
            g["highpass w$w $(n)x$(n)"] =
                @benchmarkable AutoRIFT.highpass($img, $mask, $w)
        end
        g["wallis w5 $(n)x$(n)"] = @benchmarkable AutoRIFT.wallis($img, $mask, 5)

        # The full path a caller actually takes: filter both images, shrink the masks,
        # then replace whatever came out non-finite.
        pair = AutoRIFT.ImagePair(img, bench_texture((n, n); seed = 2))
        g["preprocess+prepare $(n)x$(n)"] = @benchmarkable AutoRIFT.replace_nonfinite(
            AutoRIFT.preprocess($pair, AutoRIFT.Highpass(; width = 5)))
    end
end
