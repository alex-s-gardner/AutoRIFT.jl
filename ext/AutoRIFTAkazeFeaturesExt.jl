# A-KAZE feature tracking, via AkazeFeatures.jl.
#
# A separate extension from the ImageFeatures one, because the two dependencies are unrelated and a
# caller wanting A-KAZE should not be made to load JuliaImages' detector stack as well.
#
# ---------------------------------------------------------------------------
# Why A-KAZE is worth its own adapter when BRISK was not
# ---------------------------------------------------------------------------
#
# Measured against synthetic speckle with known ground truth, at matched keypoint counts on 512²:
#
#     rotation    ORB matches (correct)    A-KAZE matches (correct)
#     0 deg       9000 (78.4%)             8406 (99.3%)
#     3 deg       6434 (33.5%)             6257 (96.0%)
#     8 deg       5666 (19.4%)             5937 (95.8%)
#
# ORB's *precision* collapses as the field rotates; A-KAZE's does not. Both detectors are nominally
# rotation-invariant — ORB steers its BRIEF pattern by an intensity-centroid orientation — so this is
# not about the descriptor's steering but about what survives the scale space. Gaussian smoothing
# blurs speckle and signal to the same extent; nonlinear diffusion does not, so A-KAZE's keypoints
# land on structure that is actually repeatable between acquisitions.
#
# The cost is 5x on detection (0.49 s against 0.10 s at ~9000 keypoints, 512²). In *usable* vectors
# that is a wash at 0 degrees and a 5.2x win at 8 — so once there is any rotation, A-KAZE is at worst
# cost-neutral and considerably better on the precision that decides whether the consistency filter
# has anything left to keep.
#
# It is not the default only because `AkazeFeatures.jl` is unregistered, which a `[weakdeps]` entry
# cannot resolve for an arbitrary user. See `AKAZEGuess`'s docstring.

module AutoRIFTAkazeFeaturesExt

using AutoRIFT
using AutoRIFT: FirstGuess, AKAZEGuess, consistent_matches, pointset
using AkazeFeatures: AKAZE, AKAZEOptions, Create_Nonlinear_Scale_Space, Feature_Detection,
                     Compute_Descriptors

# A-KAZE detects and describes in one object rather than a stateless call, so `_detector` returns the
# options and the work happens in `_akaze_describe`. `omin = 0` because the default (-1) doubles the
# input first, which quadruples the scale-space cost for detail SAR speckle does not carry.
AutoRIFT._detector(g::AKAZEGuess) = g.kwargs

function _togray(img::AbstractMatrix)
    lo, hi = Inf, -Inf
    @inbounds for v in img
        f = Float32(v)
        isfinite(f) || continue
        f < lo && (lo = f)
        f > hi && (hi = f)
    end
    isfinite(lo) && hi > lo || throw(ArgumentError(
        "image has no finite range to scale — every pixel is NaN, Inf, or identical, so there " *
        "is nothing for a feature detector to find."))
    scale = 1.0f0 / (hi - lo)
    out = Matrix{Float64}(undef, size(img))
    @inbounds for i in eachindex(out)
        f = Float32(img[i])
        out[i] = isfinite(f) ? Float64((f - lo) * scale) : 0.0
    end
    return out
end

# Keypoints and their descriptors for one image. Returns `(descriptors, points)` where `points` is a
# vector of `(row, col)` — the same shape the ImageFeatures adapter produces, so `first_guess` need
# not know which detector it called.
function _akaze_describe(img::AbstractMatrix, kwargs)
    ny, nx = size(img)
    opts = AKAZEOptions(; omin = Int32(0), img_width = nx, img_height = ny, kwargs...)
    ak = AKAZE(opts)
    Create_Nonlinear_Scale_Space(ak, img)
    kpts = Feature_Detection(ak)
    isempty(kpts) && throw(ArgumentError(
        "A-KAZE found no keypoints. The image may be constant or already heavily smoothed."))
    desc = Compute_Descriptors(ak, kpts)
    # `Point` has `.x`/`.y` fields rather than being indexable, and A-KAZE's `x` is the *column*.
    # Returning `(row, col)` here keeps the row/column convention in one place.
    return desc, [(Float64(k.pt.y), Float64(k.pt.x)) for k in kpts]
end

# Hamming matching over A-KAZE's packed-binary MLDB descriptors, with Lowe's ratio test.
#
# Written here rather than reused from ImageFeatures because its `match_keypoints` takes that
# package's own `Keypoint` type.
#
# **Packed into `UInt64` words first**, and that is the whole performance story of this function. The
# descriptors are 61 bytes; a byte-wise `count_ones` over an XOR is 61 popcounts per pair, where the
# same 61 bytes repacked into 8 words is 8. Measured 3.6x, with identical match counts:
#
#          n     byte-wise    UInt64 words
#       1000      0.0086 s      0.0025 s
#       2000      0.0345 s      0.0097 s
#
# Still O(n²) in the keypoint count, which at A-KAZE's ~42000 keypoints on 1024² is ~4 s after the
# packing against ~15 s before. A `BKTree` would change the class rather than the constant, but a
# metric tree over Hamming distance only prunes well at small radii and the ratio test needs the
# *two* nearest — so it would have to be a k-nearest query with a loose bound, where the pruning is
# weak. Left as future work with that caveat rather than assumed to be a win.
function _pack_descriptors(d::AbstractMatrix{UInt8})
    nb, n = size(d)
    w = cld(nb, 8)
    out = zeros(UInt64, w, n)
    @inbounds for i in 1:n, b in 1:nb
        # Little-endian within each word; the layout is arbitrary as long as both descriptor sets use
        # the same one, since Hamming distance is invariant to a shared permutation of bit positions.
        out[cld(b, 8), i] |= UInt64(d[b, i]) << (8 * ((b - 1) % 8))
    end
    return out
end

function _hamming_match(d1::AbstractMatrix{UInt8}, p1::Vector, d2::AbstractMatrix{UInt8},
                        p2::Vector, ratio::Float64)
    nb = size(d1, 1)
    size(d2, 1) == nb || throw(DimensionMismatch(
        "descriptors have $(nb) and $(size(d2, 1)) bytes; they must come from the same detector"))
    w1 = _pack_descriptors(d1)
    w2 = _pack_descriptors(d2)
    nw = size(w1, 1)
    pts = Tuple{Float64,Float64}[]
    dxs = Float64[]
    dys = Float64[]
    @inbounds for i in axes(w1, 2)
        best, second, bj = typemax(Int), typemax(Int), 0
        for j in axes(w2, 2)
            dist = 0
            for k in 1:nw
                dist += count_ones(w1[k, i] ⊻ w2[k, j])
            end
            if dist < best
                second, best, bj = best, dist, j
            elseif dist < second
                second = dist
            end
        end
        # Lowe's ratio test: a good match is *distinctively* closer than the runner-up. Without it
        # every keypoint matches something, which is how raw matching reaches 80% wrong.
        (bj != 0 && best < ratio * second) || continue
        push!(pts, p1[i])
        # Reference minus secondary, matching `track!`'s convention — see the ImageFeatures adapter.
        push!(dys, p1[i][1] - p2[bj][1])
        push!(dxs, p1[i][2] - p2[bj][2])
    end
    return pts, dxs, dys
end

function AutoRIFT.first_guess(reference::AbstractMatrix, secondary::AbstractMatrix,
                              method::AKAZEGuess;
                              search_radius = 6, chip_size = 32, min_matches::Integer = 8,
                              ratio::Real = 0.9, kwargs...)
    size(reference) == size(secondary) || throw(DimensionMismatch(
        "reference is $(size(reference)) but secondary is $(size(secondary)); the two images " *
        "must be co-registered to a common grid"))

    opts = AutoRIFT._detector(method)
    da, pa = _akaze_describe(_togray(reference), opts)
    db, pb = _akaze_describe(_togray(secondary), opts)
    pts, dxs, dys = _hamming_match(da, pa, db, pb, Float64(ratio))

    isempty(pts) && throw(ArgumentError(
        "A-KAZE matching found no correspondences passing the ratio test. The pair may be fully " *
        "decorrelated, or the time gap too long for features to persist."))

    keep = consistent_matches(pts, dxs, dys; kwargs...)
    length(keep) >= min_matches || throw(ArgumentError(
        "only $(length(keep)) of $(length(pts)) matches survived the consistency filter, below " *
        "`min_matches = $min_matches`. A-KAZE holds 95-99% precision under rotation in testing, " *
        "so unlike ORB a low survival count here suggests genuinely incoherent motion rather " *
        "than descriptor failure."))

    return pointset([p[2] for p in pts[keep]], [p[1] for p in pts[keep]];
                    search_radius, chip_size,
                    dx_prior = dxs[keep], dy_prior = dys[keep])
end

end # module
