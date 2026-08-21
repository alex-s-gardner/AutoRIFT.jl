# Sparse feature tracking via ImageFeatures.jl.
#
# An extension rather than core, for the same reason the Rasters support is: `ImageFeatures` pulls in
# most of JuliaImages, and the optical glacier path — which is what nearly every call does — has no
# use for a feature detector. Loading it should be the caller's choice.
#
# Nothing here implements a detector. ORB and BRISK are `ImageFeatures`' own, tuned by keyword; what
# this file contributes is the conversion to and from AutoRIFT's conventions, and the filtering that
# makes the output usable — see `src/firstguess.jl` for why raw matches are not.

module AutoRIFTImageFeaturesExt

using AutoRIFT
using AutoRIFT: FirstGuess, ORBGuess, consistent_matches, pointset
using ImageFeatures: ImageFeatures, ORB, create_descriptor, match_keypoints
# `Gray` is what `create_descriptor` dispatches on — a bare `Matrix{Float32}` is a `MethodError`,
# even though the values are already in [0, 1]. `ImageCore` is where the type lives; taking it from
# there rather than from `Images` avoids pulling the whole umbrella package in for one wrapper.
using ImageCore: Gray

# The two detectors this extension supplies. `AutoRIFT._detector`'s fallback throws a message
# naming this dependency, so a caller who forgot `using ImageFeatures` is told what to do rather
# than shown a `MethodError`.
AutoRIFT._detector(g::ORBGuess) = ORB(; g.kwargs...)

# Only ORB. `BRISKGuess` is declared in the core but has no detector here, and the reason is worth
# recording rather than leaving as an apparent omission: **ORB is the only ImageFeatures descriptor
# that detects its own keypoints.** `create_descriptor(img, ::ORB)` exists; the BRISK, BRIEF and
# FREAK methods all require a pre-computed `Features` — keypoints *with orientations* — because those
# descriptors are detector-agnostic by design.
#
# Supporting BRISK therefore means supplying FAST detection, `corner_orientations`, per-`Feature`
# construction, and unwrapping `Feature` back to `Keypoint` for matching — all of which I wrote, and
# then found its `match_keypoints` returns a differently-shaped match than ORB's. Four
# incompatibilities for a detector that was only ever a comparison point against the one
# Muckenhuber et al. (2016) measured as best. `AutoRIFT._detector`'s fallback reports it cleanly.

# ImageFeatures wants a `Gray` image in [0, 1]; SAR amplitude is neither. Scaled by extrema over the
# *finite* values, since a reprojected scene carries NaN in its no-data border and `maximum` over
# those would give NaN for the whole image.
#
# This mirrors Muckenhuber's Eq. 2, which maps sigma0 onto 0-255 between user-chosen bounds
# specifically "to limit the influence of speckle noise". Their bounds are per-sensor constants; ours
# come from the data, which is the more conservative default — a caller with sensor-appropriate
# bounds should clamp before calling.
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
    # `Gray` rather than a bare `Matrix{Float32}`: `create_descriptor` dispatches on the colorant
    # type, so raw floats in [0, 1] are a `MethodError` rather than a working call.
    out = Matrix{Gray{Float32}}(undef, size(img))
    @inbounds for i in eachindex(out)
        f = Float32(img[i])
        # Non-finite becomes zero rather than propagating: the detector has no mask, so a NaN would
        # poison every descriptor whose patch touched it.
        out[i] = Gray(isfinite(f) ? (f - lo) * scale : 0.0f0)
    end
    return out
end

function AutoRIFT.first_guess(reference::AbstractMatrix, secondary::AbstractMatrix,
                              method::FirstGuess;
                              search_radius = 6, chip_size = 32, min_matches::Integer = 8,
                              ratio::Real = 0.9, kwargs...)
    size(reference) == size(secondary) || throw(DimensionMismatch(
        "reference is $(size(reference)) but secondary is $(size(secondary)); the two images " *
        "must be co-registered to a common grid"))

    det = AutoRIFT._detector(method)
    # `create_descriptor(img, ::ORB)` detects its own keypoints, which is what makes ORB the one
    # detector this needs no adapter for — see the note above `AutoRIFT._detector`.
    da, ka = create_descriptor(_togray(reference), det)
    db, kb = create_descriptor(_togray(secondary), det)
    matches = match_keypoints(ka, kb, da, db, ratio)

    isempty(matches) && throw(ArgumentError(
        "feature matching found no correspondences at all. The pair may be fully decorrelated, " *
        "or the time gap too long for features to persist."))

    # AutoRIFT's convention: the displacement is from *secondary* back to reference, matching
    # `track!` — so the sign here is `reference - secondary`, not the other way. Getting this
    # backwards would centre every search window on the far side of the true peak, which is worse
    # than having no prior at all.
    pts = [(Float64(p[1]), Float64(p[2])) for (p, _) in matches]
    dys = Float64[Float64(p[1]) - q[1] for (p, q) in matches]
    dxs = Float64[Float64(p[2]) - q[2] for (p, q) in matches]

    keep = consistent_matches(pts, dxs, dys; kwargs...)
    length(keep) >= min_matches || throw(ArgumentError(
        "only $(length(keep)) of $(length(matches)) matches survived the consistency filter, " *
        "below `min_matches = $min_matches`. Raw descriptor matching on speckle is mostly wrong " *
        "— 9.5-62.5% correct in testing — so a low survival count is normal, but this few means " *
        "no coherent motion field was found. Check the time gap, or loosen `tolerance`."))

    # Row is y and column is x, which is the one place this conversion has to be right: keypoints
    # are `(row, col)` and `PointSet` is `(x, y)`.
    return pointset([p[2] for p in pts[keep]], [p[1] for p in pts[keep]];
                    search_radius, chip_size,
                    dx_prior = dxs[keep], dy_prior = dys[keep])
end

end # module
