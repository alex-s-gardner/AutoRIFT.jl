# Sparse feature tracking, to seed the dense search.
#
# Layer 2.5: depends on `points.jl` for the output type and on nothing else in the package. The
# detector itself lives in a package extension, because `ImageFeatures` is a heavy dependency and
# the optical path has no use for it.
#
# ---------------------------------------------------------------------------
# Why a first guess at all
# ---------------------------------------------------------------------------
#
# Glacier ice moves metres per day; sea ice moves up to **100 km per day**. At Sentinel-1's 40 m
# pixel that is 2500 pixels of displacement, and a dense search cannot reach it — the correlation
# cost grows as the search area, so a radius wide enough to find the peak is a radius wide enough to
# be unaffordable, and wide enough that a spurious peak becomes likely.
#
# The way out is the structure every published sea-ice tracker uses: a **sparse** stage first, whose
# matches are few but whose search is global, then the dense stage centred on what the sparse stage
# found. `nansencenter/sea_ice_drift` states it plainly — ORB gives "the first guess sea ice drift",
# then "pattern matching ... to retrieve sea ice drift on a regular grid". That second stage is
# `autorift`; this file is the first.
#
# `PointSet` already carries per-point `dx_prior`/`dy_prior`, and `track!` already centres each
# point's search window on them. So the seam needed no new machinery: this file produces a
# `PointSet` and the existing pipeline consumes it.
#
# ---------------------------------------------------------------------------
# Why filtering is not optional
# ---------------------------------------------------------------------------
#
# Raw descriptor matching on speckle is *mostly wrong*, and by a margin that surprised me. Measured
# on synthetic speckle with a known rotation and translation, 3000 ORB matches against ground truth:
#
#     translation only, 60 px      62.5% within 5 px of truth
#     3 deg rotation + 30 px       28.6%
#     8 deg rotation + 80 px       13.1%
#     15 deg rotation + 120 px      9.5%
#
# So a median over raw matches is meaningless once the field is not a rigid shift — which is always,
# since ice rotates and shears. Both reference implementations say so in their own vocabulary:
# Demchev et al. (2017) filter by "the consistency check in the local field of motion", and
# `xdenisx/ice_drift_pc_ncc` by "homogeneity criteria".
#
# [`consistent_matches`](@ref) implements that, and it is *decisive* rather than incremental: the
# same four cases become **100% correct** after filtering, with median errors of 0.0, 0.5, 0.29 and
# 0.81 px. What collapses instead is recall — 1851, 333, 23 and 8 vectors survive. That is the right
# trade for a prior: eight correctly-placed vectors tell the dense search where to look, where three
# thousand mostly-wrong ones would send it somewhere there is nothing to find.

"""
    FirstGuess

Abstract supertype for sparse feature-tracking methods used to seed the dense search.

Concrete subtype: [`ORBGuess`](@ref), declared here but only *working* once `ImageFeatures` is loaded — the detector is a heavy dependency the optical path has no
use for, so the methods live in a package extension.

The types are declared in the core rather than in the extension deliberately. An extension may add
methods to the parent but must not `eval` new bindings into it: that "breaks incremental compilation
because the side effects will not be permanent", which Julia reports as an error rather than
tolerating. So the type is core and the behaviour is the extension, which is also the arrangement
that lets this docstring exist without a conditional dependency.

See [`first_guess`](@ref) for the entry point and [`consistent_matches`](@ref) for why the raw
matches cannot be used directly.
"""
abstract type FirstGuess end

"""
    ORBGuess(; num_keypoints = 3000, kwargs...)

Sparse first guess from ORB features. Needs `ImageFeatures` loaded to do anything.

ORB is the measured choice rather than a default of convenience. Muckenhuber et al. (2016) compared
three detectors on Sentinel-1 sea ice over Fram Strait and north-east Greenland:

| detector | vectors | time |
|---|---:|---:|
| **ORB** | **177,513** | **66 s** |
| SIFT | 43,260 | 182 s |
| SURF | 25,113 | 99 s |

Four times the vectors of SIFT in a third of the time. ORB is also unencumbered, where SIFT and SURF
were patented when that paper was written — which is why its title says *open-source*.

!!! note "A-KAZE would likely be better, and is not available"
    Demchev et al. (2017) report A-KAZE outperforming ORB "up to an order of magnitude" on ice drift:
    Gaussian scale space blurs speckle and signal alike, while A-KAZE's nonlinear diffusion preserves
    edges. It is not in the Julia ecosystem, and its diffusion scale space is precisely the expensive
    part — so it is a real candidate, not a free one. Measure this first.

Keywords are forwarded to `ImageFeatures.ORB`.
"""
struct ORBGuess{K} <: FirstGuess
    kwargs::K
end
ORBGuess(; kwargs...) = ORBGuess(kwargs)

# Deliberately only one concrete `FirstGuess`. A second detector was written (BRISK) and removed:
# ORB is the only ImageFeatures descriptor that detects its own keypoints, so every other one needs
# FAST detection, orientation assignment, per-`Feature` construction and a different match structure
# unwrapped — four pieces of adapter for a comparison point against the detector Muckenhuber et al.
# (2016) already measured as the best of three. The abstract type is what makes adding one later a
# new method rather than a redesign; see `ext/AutoRIFTImageFeaturesExt.jl`.

# The detector itself, from the extension. Defined here so the error a caller sees names the missing
# dependency rather than being a bare `MethodError` on an internal function.
function _detector(g::FirstGuess)
    throw(ArgumentError(
        "`$(nameof(typeof(g)))` needs ImageFeatures to be loaded. Run `using ImageFeatures` " *
        "first — the detector lives in a package extension, because it is a heavy dependency " *
        "that the optical path does not need."))
end

"""
    first_guess(reference, secondary, method::FirstGuess; kwargs...) -> PointSet

Sparse displacement estimates, as a `PointSet` whose `dx_prior`/`dy_prior` carry them.

The output feeds straight into the dense stage:

```julia
guess = first_guess(a, b, AutoRIFT.ORBGuess())      # needs ImageFeatures loaded
out = autorift(a, b, guess)                         # dense search, centred on the guess
```

Requires `ImageFeatures` to be loaded — the methods are in a package extension. See
[`FirstGuess`](@ref).

# Keywords

- `search_radius = 6`: the radius the *dense* stage will use at each returned point. Small on
  purpose: the whole reason for a first guess is that the prior does the reaching, so the dense
  search only has to refine. This is what makes a 100 km/day displacement affordable.
- `chip_size = 32`: chip size for the dense stage.
- `min_matches = 8`: below this many surviving matches, throw rather than return a `PointSize` that
  will silently produce nothing. A first guess that found almost nothing is a signal about the data
  — decorrelated pair, wrong time gap — not a result to pass on.
- Filtering keywords are forwarded to [`consistent_matches`](@ref).
"""
function first_guess end

"""
    consistent_matches(points, dx, dy; neighbours = 12, tolerance = 6.0, min_agree = 5)
        -> Vector{Int}

Indices of the matches whose displacement agrees with their spatial neighbours.

A match survives if at least `min_agree` of its `neighbours` nearest neighbours have a displacement
within `tolerance` pixels of its own. This is the "consistency check in the local field of motion"
of Demchev et al. (2017) and the "homogeneity criteria" of `xdenisx/ice_drift_pc_ncc`.

The premise is physical rather than statistical: ice moves as a field, so a *correct* match has
neighbours that agree with it, while a descriptor mismatch lands somewhere unrelated to its
neighbourhood. That distinction survives rotation and shear, which is what makes it the right filter
here — a global median does not, because there is no global displacement to take a median of.

**This is not the Gardner filter.** [`GardnerFilter`](@ref) rejects outliers from a *gridded*
displacement field using window reductions; these points are scattered, so neighbourhoods come from
a nearest-neighbour search rather than a window. The two also answer different questions: this one
decides which matches are real, the other decides which measurements to keep.

Measured on synthetic speckle with known ground truth, this takes raw ORB matching from 9.5-62.5%
correct to **100% correct** in every case tried, at the cost of most of the matches — see the
discussion at the top of `src/firstguess.jl`.

`points` is a vector of `(row, col)`; `dx`/`dy` are the matched displacements. Returns indices into
all three.
"""
function consistent_matches(points::AbstractVector, dx::AbstractVector, dy::AbstractVector;
                            neighbours::Integer = 12, tolerance::Real = 6.0,
                            min_agree::Integer = 5)
    n = length(points)
    (length(dx) == n && length(dy) == n) || throw(DimensionMismatch(
        "points, dx and dy must be the same length, got $n, $(length(dx)) and $(length(dy))"))
    min_agree <= neighbours || throw(ArgumentError(
        "`min_agree` ($min_agree) cannot exceed `neighbours` ($neighbours) — no match could " *
        "ever survive."))
    n == 0 && return Int[]

    tol = Float64(tolerance)^2
    K = min(Int(neighbours), n - 1)
    K < min_agree && return Int[]     # too few points to judge consistency at all

    keep = Int[]
    d2 = Vector{Float64}(undef, n)
    # O(n²) in the number of matches, deliberately. A k-d tree is asymptotically better, but n here
    # is the *surviving descriptor matches* — thousands, not millions — and measured at 0.13 s for
    # 3000 points against 2.1 s for the ORB detection that produced them. Spending a dependency
    # (NearestNeighbors) and a spatial index to optimise 6% of the stage would be the wrong trade;
    # revisit if the match count ever reaches tens of thousands.
    @inbounds for i in 1:n
        pi = points[i]
        for j in 1:n
            pj = points[j]
            d2[j] = (Float64(pi[1]) - pj[1])^2 + (Float64(pi[2]) - pj[2])^2
        end
        d2[i] = Inf                    # a match is not its own neighbour
        nb = partialsortperm(d2, 1:K)
        agree = 0
        for j in nb
            ddx = dx[i] - dx[j]
            ddy = dy[i] - dy[j]
            ddx * ddx + ddy * ddy <= tol && (agree += 1)
        end
        agree >= min_agree && push!(keep, i)
    end
    return keep
end
