# Correlation surface, peak location, and subpixel refinement — the inner loop
# the whole package's throughput rests on. Populated at M2.
#
# What this group must answer:
#
#   * Where the direct/FFT crossover falls. Direct evaluation costs
#     (2*radius)^2 * chip^2 multiply-accumulates; an FFT over the search window
#     costs ~3*N*log(N) plus an integral-image pass. Theory puts the crossover
#     near chip 32 / radius 25, but the constant depends on cache behaviour and
#     FFTW's plan quality, so it is set by measurement here and used as a
#     compile-time-known threshold thereafter.
#
#   * Whether a global integral image beats a per-point one. OpenCV rebuilds the
#     integral image for every grid point, over windows that overlap ~2.5x in
#     each direction at typical grid spacing. Computing one integral image over
#     the whole padded image is O(N) instead of O(points * window^2). Expected to
#     be a large win on the coarse pass; measured, not assumed.
#
#   * Whether exact integer accumulation beats float for the UInt8 path. Integer
#     accumulation is SIMD-friendly, exact, and cannot cancel — but only if the
#     compiler vectorizes it.
#
#   * The cost of subpixel refinement as a fraction of the whole. The reference
#     cascades pyrUp up to seven times on a 5x5 patch, which at upsampling 128
#     materializes a 640x640 surface to find one argmax. Evaluating the composite
#     interpolant directly should be far cheaper; this group is where that claim
#     is checked.
#
# Every measurement is paired with its OpenCV equivalent from
# benchmark/results/python.json, so `speedup_vs_python` is reported per kernel
# rather than only end-to-end.
let g = addgroup!(SUITE, "correlate")
end
