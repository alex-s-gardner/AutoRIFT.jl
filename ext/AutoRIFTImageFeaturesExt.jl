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
using AutoRIFT: ORBGuess
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

# ORB's own `create_descriptor(img, ::ORB)` detects its keypoints as well as describing them, which
# is what makes it the one detector needing no adapter beyond this — see the note above
# `AutoRIFT._detector`.
#
# `max_distance`, not a Lowe ratio. The distinction matters and the old name got it wrong:
# `match_keypoints` thresholds the *absolute* normalised Hamming distance and greedily blanks each
# matched column — it has no ratio test at all. The keyword was called `ratio` with a default of 0.9
# by analogy with the A-KAZE path, where 0.9 genuinely is a Lowe ratio.
#
# Measured on 3000 ORB keypoints: threshold 0.1 admits 2642 matches, 0.9 admits 3000. So the old
# default did no descriptor filtering whatsoever. ImageFeatures' own default is 0.1, which is what
# this now is.
#
# Worth stating plainly: this was a *latent* bug, not a wrong answer. End to end the old default gave
# 2093 surviving points at 98.8% within 1 px of truth, the new one 2636 at 99.0% — the consistency
# filter was already rejecting everything the loose threshold let through. The fix yields 26% more
# usable vectors and stops relying on a downstream filter to cover an upstream mistake.
function AutoRIFT._match_features(reference::AbstractMatrix, secondary::AbstractMatrix,
                                  method::ORBGuess; max_distance::Real = 0.1)
    det = AutoRIFT._detector(method)
    da, ka = create_descriptor(AutoRIFT._scale01(Gray{Float32}, reference), det)
    db, kb = create_descriptor(AutoRIFT._scale01(Gray{Float32}, secondary), det)
    matches = match_keypoints(ka, kb, da, db, max_distance)

    # Reference minus secondary, matching `track!`'s convention: `dx`/`dy` are the offset from the
    # secondary back to the reference. Getting this backwards would centre every search window on the
    # far side of the true peak, which is worse than having no prior at all.
    pts = [(Float64(p[1]), Float64(p[2])) for (p, _) in matches]
    dys = Float64[Float64(p[1]) - q[1] for (p, q) in matches]
    dxs = Float64[Float64(p[2]) - q[2] for (p, q) in matches]
    return pts, dxs, dys
end

end # module
