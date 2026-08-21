# A k-d tree for the consistency filter's neighbour search.
#
# Its own extension rather than living inside either detector's, because both need it and neither
# should depend on the other. Loading `NearestNeighbors` alongside `ImageFeatures` or `AkazeFeatures`
# is what turns the filter from O(n²) into O(n log n) — see `AutoRIFT._neighbour_indices` for the
# measurements.
#
# **Not installed automatically**, which is worth being clear about: an extension's trigger list is a
# conjunction of packages the *user* has loaded, not an install directive, so neither detector
# extension can pull this in. A caller who follows `first_guess`'s docstring and loads only a detector
# gets the O(n²) path. `first_guess` says so.

module AutoRIFTNearestNeighborsExt

using AutoRIFT
using AutoRIFT: _NeighbourStrategy
using NearestNeighbors: KDTree, knn

# A *new* strategy type rather than a redefinition of the core's `_neighbour_indices` — see that
# function for why the obvious spellings fail.
struct KDTreeNeighbours <: _NeighbourStrategy end
# More specific than the core's `::Type{<:_NeighbourStrategy}` fallback, so this wins by dispatch
# rather than by replacing it.
AutoRIFT._neighbour_strategy(::Type{AutoRIFT._NeighbourStrategy}) = KDTreeNeighbours()

# Measured 12x at 3000 points rising to 90x at 20000, with **identical output** — both strategies
# take the same K nearest neighbours, so this changes the cost and nothing else.
#
# `K + 1` because a point is its own nearest neighbour and the caller skips it; `true` sorts by
# distance, which the brute-force `partialsortperm` also does, so the two agree index for index
# rather than merely as sets.
function AutoRIFT._neighbour_indices(points::AbstractVector, K::Int, ::KDTreeNeighbours)
    n = length(points)
    data = Matrix{Float64}(undef, 2, n)
    @inbounds for i in 1:n
        data[1, i] = Float64(points[i][1])
        data[2, i] = Float64(points[i][2])
    end
    idxs, _ = knn(KDTree(data), data, min(K + 1, n), true)
    return idxs
end

end # module
