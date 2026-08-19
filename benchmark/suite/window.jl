# Sliding-window reductions. Populated at M3.
#
# Eight call sites in the pyramid, each sweeping the whole grid, so these are a
# meaningful fraction of per-level cost. The Python reference materializes a
# (window^2 x npoints) matrix and grows its output with repeated concatenation,
# which is quadratic; a 10-50x improvement is expected from doing neither.
#
# Measured per reduction (max, min, mean, median, range, MAD, agreement count)
# and per window width. Width matters more than it looks: the pyramid dilates
# masks with windows up to 48 wide, where an O(window^2) implementation is
# hopeless and the O(1)-per-pixel decompositions in ImageMorphology earn their
# dependency.
let g = addgroup!(SUITE, "window")
end
