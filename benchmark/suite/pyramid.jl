# Multi-scale orchestration. Populated at M5.
#
# Measured per level and per step, not just in total, because the steps have very
# different characters: correlation is compute-bound and parallel, while the
# filtering, resampling, distance transforms, and hole filling are
# memory-bandwidth-bound and largely serial. That serial fraction is what caps
# intra-pair thread scaling, so quantifying it is what makes the choice between
# intra-pair and pair-level parallelism an informed one.
let g = addgroup!(SUITE, "pyramid")
end
