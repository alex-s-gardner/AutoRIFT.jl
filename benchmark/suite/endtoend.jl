# Whole-pipeline timings on the three standard scene sizes. Populated at M6.
#
# The headline number, and the one directly comparable to the Python reference:
# `bench_python.py` runs `obj.runAutorift()` on these same three scenes and emits
# the same JSON schema, so `speedup_vs_python` is a measured ratio rather than a
# claim.
#
# Also covers cold start. If the production driver launches a short-lived process
# per image pair, then load time plus first-call compilation can exceed the
# actual computation, and no amount of kernel optimization would show up in the
# wall clock. Tracked as a first-class metric for that reason, with a
# PrecompileTools workload and persisted FFTW wisdom as the levers.
let g = addgroup!(SUITE, "endtoend")
end
