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
#
# ---------------------------------------------------------------------------
# Why these are per-point and not global
# ---------------------------------------------------------------------------
#
# Each grid point builds an integral image of its own search window, which looks like obvious
# redundancy: neighbouring windows overlap heavily, and one table over the whole image would serve
# every point. `boxsum` already reads four corners at an arbitrary offset, so the change is small.
#
# Measured on 1024x1024 with the default geometry, and it does not pay:
#
#     per-point, 81x81 window    7.24 us x 729 points = 5.28 ms
#     global, 1024x1024          1.81 ms x 3 levels   = 5.42 ms
#
# Slightly *worse*, and the reason is the chip-size loop rather than the arithmetic. A global table
# must be built once per level whether that level searches 729 points or none — and after the
# finest level resolves what it can, later levels search almost nothing. On the measured scene the
# finest level resolved all 729 points and levels 64 and 128 searched zero, so the per-point total
# is about 1x the grid, not the 3x a global table pays regardless.
#
# Two plausible-sounding explanations that turned out wrong, recorded so they are not re-argued:
# `boxsum` locality is *not* the issue (2.15 ns from a 1025x1025 table against 2.16 ns from an
# 82x82 one — identical, since the four corner reads miss cache either way), and neither is the
# 16 MiB the two Float64 tables would occupy. The redundancy is simply smaller than it looks.
#
# This would flip on a scene where the coarse levels do real work: heavily decorrelated imagery, or
# a `chip_size` small relative to the texture scale. Worth re-measuring there rather than treating
# it as settled for every input.

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
    integral!(S::AbstractMatrix{ComplexF64}, A::AbstractMatrix{<:Complex})

Integral image of a complex array, so a window's complex sum — and hence its complex mean — is
four reads.

[`Coherence`](@ref) needs this for the same reason [`ZNCC`](@ref) needs the real version: the
measure removes the mean from *both* sides, and the window's mean is only affordable per shift
through a summed-area table. Removing it from the chip alone would leave the numerator and the
denominator normalised about different points, which caps the self-correlation of a chip below 1
by exactly the ratio of the two norms.
"""
function integral!(S::AbstractMatrix{ComplexF64}, A::AbstractMatrix{<:Complex})
    m, n = size(A)
    size(S) == (m + 1, n + 1) || throw(DimensionMismatch(
        "integral image must be $(m + 1)x$(n + 1) for a $(m)x$(n) input, " *
        "got $(size(S))"))

    @inbounds begin
        for j in 1:(n + 1)
            S[1, j] = zero(ComplexF64)
        end
        for i in 1:(m + 1)
            S[i, 1] = zero(ComplexF64)
        end
        for j in 1:n
            run = zero(ComplexF64)
            for i in 1:m
                run += ComplexF64(A[i, j])
                S[i + 1, j + 1] = run + S[i + 1, j]
            end
        end
    end
    return S
end

"""
    integral_sq!(S, A::AbstractMatrix{<:Complex})

Integral image of `abs2.(A)` — the squared *magnitude* — for complex (SLC) input.

The energy of a complex sample is `|z|²`, not `z²`: the latter is complex and its integral image
would make [`Coherence`](@ref)'s denominator complex, which is meaningless as a normaliser.

Separate method rather than a generic `abs2` in the real version, because `abs2` on a `Real` is
`v*v` and the compiler would produce identical code — but writing it that way would put a
`Complex` union through the real path, which is the hot path for every optical pair.
"""
function integral_sq!(S::AbstractMatrix{Float64}, A::AbstractMatrix{<:Complex})
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
                run += abs2(ComplexF64(A[i, j]))
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
@inline function boxsum(S::AbstractMatrix{<:Union{Float64,ComplexF64}}, i::Int, j::Int,
                        h::Int, w::Int)
    @inbounds return S[i + h, j + w] - S[i, j + w] - S[i + h, j] + S[i, j]
end
