# Integral images (summed-area tables).
#
# Layer 1: no dependencies on the rest of AutoRIFT.
#
# The correlation denominator needs, at every candidate shift, the sum and the sum
# of squares of the search window under the chip. Computing those directly is
# O(chip area) per shift; an integral image makes each one four array reads,
# independent of chip size. This is what makes a direct-evaluation correlator
# competitive with an FFT one at small chip sizes.
#
# Accumulate in Float64 even for Float32 input. The reason is not fussiness: over
# a 64x64 window of 8-bit values, the sum of squares reaches ~2.7e8, and the
# variance is then computed as `sumsq - sum^2/n` — a difference of two large,
# nearly equal quantities. In Float32 that cancellation destroys most of the
# significant digits precisely when the window is low-contrast, which is exactly
# where the correlation peak is least well determined and most needs an accurate
# denominator. OpenCV accumulates its integral images in double for the same
# reason.

"""
    integral!(S, A)

Fill `S` with the integral image (summed-area table) of `A`.

`S` must be one larger than `A` in each dimension. The extra leading row and
column are zero, which is what lets [`boxsum`](@ref) read four corners with no
bounds checking or branching — the alternative, an `A`-sized table, needs a
special case for every box touching the first row or column.

`S[i+1, j+1]` holds the sum of `A[1:i, 1:j]`.
"""
function integral!(S::AbstractMatrix{Float64}, A::AbstractMatrix)
    m, n = size(A)
    size(S) == (m + 1, n + 1) || throw(DimensionMismatch(
        "integral image must be $(m + 1)x$(n + 1) for a $(m)x$(n) input, " *
        "got $(size(S))"))

    @inbounds begin
        for j in 1:(n + 1)
            S[1, j] = 0.0
        end
        for i in 1:(m + 1)
            S[i, 1] = 0.0
        end
        # Column-major traversal: `S` is written down each column in turn, so the
        # `S[i, j+1]` read of the previous column is far from this column's writes
        # but is sequential in its own right, and `S[i-1, j+1]` is the element
        # just written.
        for j in 1:n
            run = 0.0
            for i in 1:m
                run += Float64(A[i, j])
                S[i + 1, j + 1] = run + S[i + 1, j]
            end
        end
    end
    return S
end

"""
    integral_sq!(S, A)

Fill `S` with the integral image of `A .^ 2`, without materialising the squared
array.

Squaring on the fly rather than in a temporary saves an allocation the size of the
search window at every grid point — which, across the tens of millions of image
pairs this package is built for, is the difference between a steady state and
continuous garbage collection.
"""
function integral_sq!(S::AbstractMatrix{Float64}, A::AbstractMatrix)
    m, n = size(A)
    size(S) == (m + 1, n + 1) || throw(DimensionMismatch(
        "integral image must be $(m + 1)x$(n + 1) for a $(m)x$(n) input, " *
        "got $(size(S))"))

    @inbounds begin
        for j in 1:(n + 1)
            S[1, j] = 0.0
        end
        for i in 1:(m + 1)
            S[i, 1] = 0.0
        end
        for j in 1:n
            run = 0.0
            for i in 1:m
                v = Float64(A[i, j])
                run += v * v
                S[i + 1, j + 1] = run + S[i + 1, j]
            end
        end
    end
    return S
end

"""
    integral(A) -> Matrix{Float64}

Allocating form of [`integral!`](@ref), for tests and one-off use. The hot path
uses the in-place version with a preallocated table.
"""
integral(A::AbstractMatrix) = integral!(zeros(Float64, size(A, 1) + 1,
                                              size(A, 2) + 1), A)

"""
    integral_sq(A) -> Matrix{Float64}

Allocating form of [`integral_sq!`](@ref).
"""
integral_sq(A::AbstractMatrix) = integral_sq!(zeros(Float64, size(A, 1) + 1,
                                                    size(A, 2) + 1), A)

"""
    boxsum(S, i, j, h, w)

Sum of the `h`-by-`w` block of the original array whose top-left corner is at
`(i, j)`, read from its integral image `S` in four accesses.

No bounds checking: the caller is responsible for `i + h <= size(S, 1)` and
`j + w <= size(S, 2)`. This is called once per candidate shift per grid point —
hundreds of millions of times per image pair — so the check is hoisted to the
loop that computes the shift range instead.
"""
@inline function boxsum(S::AbstractMatrix{Float64}, i::Int, j::Int, h::Int, w::Int)
    @inbounds return S[i + h, j + w] - S[i, j + w] - S[i + h, j] + S[i, j]
end
