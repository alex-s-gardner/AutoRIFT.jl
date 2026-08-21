# A k-d tree for the consistency filter's neighbour search.
#
# Its own extension rather than living inside either detector's, because both need it and neither
# should depend on the other. Loading `NearestNeighbors` alongside `ImageFeatures` or `AkazeFeatures`
# is what turns the filter from O(n²) into O(n log n) — see `AutoRIFT._neighbour_indices` for the
# measurements. Both detector extensions list it, so installing a detector brings this along.

module AutoRIFTNearestNeighborsExt

using AutoRIFT
using AutoRIFT: _NeighbourStrategy
using NearestNeighbors: KDTree, knn

# A *new* strategy type, and a method on `_neighbour_strategy` selecting it — not a redefinition of
# `_neighbour_indices(::AbstractVector, ::Int)`. Adding a method with a signature the core already
# defines does not extend it, it overwrites it, and Julia then refuses to precompile the extension:
# `Method overwriting is not permitted during Module precompilation`. Found the hard way.
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
