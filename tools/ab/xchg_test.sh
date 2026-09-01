#!/bin/sh
# Cross-language round trip for the A/B array exchange: Julia writes and Python reads, then Python
# writes and Julia reads. Each side asserts the shape and every element, on a non-square,
# non-symmetric array — the only shape where a transpose is visible rather than silent.
#
# Run this before trusting any A/B diagnostic that moves arrays between the two languages.
#
#   sh tools/ab/xchg_test.sh
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
export AB_XCHG_DIR="${TMPDIR:-/tmp}/ab_xchg_test.$$"
trap 'rm -rf "$AB_XCHG_DIR"' EXIT

julia --project="$HERE" -e '
include("'"$HERE"'/xchg.jl")
xchg_selftest(verbose = false)
# A ramp whose value encodes its own position, so a transpose lands as a value mismatch too.
A = Float32[100i + j for i in 1:4, j in 1:7]
xwrite("jl_to_py", A)
println("julia wrote jl_to_py ", size(A))'

micromamba run -n arift-ref python -c '
import sys; sys.path.insert(0, "'"$HERE"'")
import numpy as np, xchg
xchg.selftest(verbose=False)
A = xchg.read("jl_to_py")
want = np.array([[100*i + j for j in range(1, 8)] for i in range(1, 5)], dtype=np.float32)
assert A.shape == want.shape, "shape: got {} want {}".format(A.shape, want.shape)
assert np.array_equal(A, want), "values differ — orientation is wrong"
print("python read jl_to_py", A.shape, "OK")
B = np.array([[1000*i + j for j in range(1, 6)] for i in range(1, 4)], dtype=np.float64)
xchg.write("py_to_jl", B)
print("python wrote py_to_jl", B.shape)'

julia --project="$HERE" -e '
include("'"$HERE"'/xchg.jl")
B = xread("py_to_jl")
want = Float64[1000i + j for i in 1:3, j in 1:5]
size(B) == size(want) || error("shape: got $(size(B)) want $(size(want))")
B == want || error("values differ — orientation is wrong")
println("julia read py_to_jl ", size(B), " OK")
println("cross-language exchange round trip passed")'
