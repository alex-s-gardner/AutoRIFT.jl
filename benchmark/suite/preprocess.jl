# Pre-correlation filters and quantization. Populated at M3.
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
end
