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

Concrete subtypes: [`ORBGuess`](@ref) and [`AKAZEGuess`](@ref), declared here but only *working* once
`ImageFeatures` or `AkazeFeatures` respectively is loaded — the detector is a heavy dependency the optical path has no
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

!!! note "A-KAZE is more precise under rotation, and costs 5x"
    Demchev et al. (2017) report A-KAZE outperforming ORB "up to an order of magnitude" on ice drift:
    Gaussian scale space blurs speckle and signal alike, while A-KAZE's nonlinear diffusion preserves
    edges. Measured here against synthetic speckle with known ground truth, at matched keypoint
    counts on 512²:

    | rotation | ORB matches (correct) | A-KAZE matches (correct) |
    |---:|---:|---:|
    | 0° | 9000 (78.4%) | 8406 (**99.3%**) |
    | 3° | 6434 (33.5%) | 6257 (**96.0%**) |
    | 8° | 5666 (19.4%) | 5937 (**95.8%**) |

    ORB's precision collapses as the field rotates; A-KAZE's does not. In *usable* vectors that is
    1.2x at 0° rising to **5.2x at 8°** — against 5x the detection time (0.49 s vs 0.10 s at ~9000
    keypoints). So the two roughly break even on cost per usable vector once there is rotation, and
    A-KAZE wins outright on the precision that determines whether the consistency filter has
    anything to keep.

    Available as [`AKAZEGuess`](@ref) when `AkazeFeatures` is loaded. Not the default, because it is
    unregistered — see that docstring.

Keywords are forwarded to `ImageFeatures.ORB`.
"""
struct ORBGuess{K} <: FirstGuess
    kwargs::K
end
ORBGuess(; kwargs...) = ORBGuess(kwargs)

"""
    AKAZEGuess(; omax = 4, nsublevels = 4, dthreshold = 0.001)

Sparse first guess from A-KAZE features. Needs `AkazeFeatures` loaded.

More precise than [`ORBGuess`](@ref) under rotation and about five times slower — see the table in
`ORBGuess`. The reason is the scale space: A-KAZE diffuses *nonlinearly*, so edges survive smoothing
while speckle does not, where a Gaussian pyramid blurs both equally. On SAR that difference is the
whole game, and it is why Demchev et al. (2017) chose it.

!!! warning "Unregistered dependency"
    `AkazeFeatures.jl` is a pure-Julia port of Alcantarilla's original, actively maintained, and
    **not in the General registry** — it must be installed by URL:

    ```julia
    Pkg.add(url = "https://github.com/nlw0/AkazeFeatures.jl")
    ```

    That is why `ORBGuess` is the documented default despite being less precise: a registered
    dependency can be a `[weakdeps]` entry that resolves for every user, and an unregistered one
    cannot. Use this when the extra precision matters and you control the environment.
"""
struct AKAZEGuess{K} <: FirstGuess
    kwargs::K
end
AKAZEGuess(; kwargs...) = AKAZEGuess(kwargs)

# Two concrete `FirstGuess`es, and a third was written and removed: BRISK.
# ORB is the only ImageFeatures descriptor that detects its own keypoints, so every other one needs
# FAST detection, orientation assignment, per-`Feature` construction and a different match structure
# unwrapped — four pieces of adapter for a comparison point against the detector Muckenhuber et al.
# (2016) already measured as the best of three. A-KAZE earned its adapter by measuring better;
# BRISK did not. The abstract type is what makes adding one a new method rather than a redesign.

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

    # Neighbour indices from whichever strategy is available: a k-d tree when `NearestNeighbors` is
    # loaded, the brute-force scan otherwise. Both produce the same answer; see `_neighbour_indices`.
    nbrs = _neighbour_indices(points, K)

    keep = Int[]
    @inbounds for i in 1:n
        agree = 0
        for j in nbrs[i]
            j == i && continue          # a match is not its own neighbour
            ddx = dx[i] - dx[j]
            ddy = dy[i] - dy[j]
            ddx * ddx + ddy * ddy <= tol && (agree += 1)
        end
        agree >= min_agree && push!(keep, i)
    end
    return keep
end

# The `K` nearest neighbours of every point, as a vector of index vectors.
#
# Split out because the *complexity class* differs by strategy and the measurement is stark. The
# brute-force scan below is O(n²); a k-d tree is O(n log n), and n is the surviving descriptor-match
# count, which A-KAZE pushes into the tens of thousands:
#
#         n     brute force    k-d tree    speedup
#      3000        0.041 s      0.0033 s      12x
#     10000        0.451 s      0.0113 s      40x
#     20000        2.139 s      0.0239 s      90x
#     40000       ~8.9 s (est)  0.0495 s      ~180x
#
# Measured in place on a 1024² A-KAZE pair (41857 keypoints, 38200 matches), the whole stage now
# breaks down as:
#
#     detection x2          8.91 s
#     descriptor match      4.57 s   (was ~15 s before UInt64 packing)
#     consistency filter    0.045 s  (was ~9 s brute force)
#
# So the filter went from dominating the stage to 0.3% of it, and detection is now the bottleneck —
# which is the right shape, since that is the algorithm rather than the adapter. Identical output at
# every size tested, since both strategies take the same K nearest neighbours.
#
# The tree method lives in its own extension rather than here: `NearestNeighbors` is only needed by
# the sea-ice path, and a core dependency for one optional stage is the wrong trade.
#
# Dispatched on a *strategy* argument rather than by the extension redefining this signature. An
# extension adding a method with the identical signature does not extend, it **overwrites** — which
# Julia reports as `Method overwriting is not permitted during Module precompilation` and then refuses
# to precompile. `_NeighbourStrategy` gives the two implementations distinct signatures, and
# `_neighbour_strategy()` is the one method an extension replaces, returning its own singleton.
abstract type _NeighbourStrategy end
struct _BruteForceNeighbours <: _NeighbourStrategy end

# Which strategy is available.
#
# The seam is a *fallback on the abstract type*: the core defines it for `_NeighbourStrategy` and the
# extension for its own concrete subtype, so the two are genuinely different methods and the more
# specific one wins. A zero-argument function does not work here — the extension's version would have
# the identical signature, which overwrites rather than extends and makes the extension unprecompilable.
_neighbour_strategy(::Type{<:_NeighbourStrategy}) = _BruteForceNeighbours()
_neighbour_strategy() = _neighbour_strategy(_NeighbourStrategy)

_neighbour_indices(points::AbstractVector, K::Int) =
    _neighbour_indices(points, K, _neighbour_strategy())

function _neighbour_indices(points::AbstractVector, K::Int, ::_BruteForceNeighbours)
    n = length(points)
    out = Vector{Vector{Int}}(undef, n)
    d2 = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        pi = points[i]
        for j in 1:n
            pj = points[j]
            d2[j] = (Float64(pi[1]) - pj[1])^2 + (Float64(pi[2]) - pj[2])^2
        end
        d2[i] = Inf
        out[i] = partialsortperm(d2, 1:K)
    end
    return out
end
