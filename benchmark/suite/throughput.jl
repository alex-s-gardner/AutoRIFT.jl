# Batch throughput, in image pairs per second. Populated at M8.
#
# The number that actually matters at production scale, and the one the API was
# shaped around. Compares two regimes:
#
#   intra-pair    one pair at a time, threads split across its search points.
#                 Simple, but carries an irreducible serial fraction (filtering,
#                 resampling, the merge) and suffers load imbalance,
#                 because the coarse pass zeroes most of the grid in
#                 spatially-clustered patterns and a skipped point is ~400x
#                 cheaper than a searched one.
#
#   pair-level    one pair per worker, each reusing its own buffers and FFT plans
#                 across pairs via `reinit!`. No serial fraction, no imbalance,
#                 and better cache locality per core.
#
# Pair-level is expected to win, which is why the cache lifecycle is public API
# rather than an implementation detail. The crossover is published either way, so
# the recommendation rests on measurement.
let g = addgroup!(SUITE, "throughput")
end
