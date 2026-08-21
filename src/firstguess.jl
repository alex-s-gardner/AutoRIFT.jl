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

"""
    AutoRIFT.required_package(method::FirstGuess) -> String

Which package must be loaded for `method` to work.

Exists so the "you forgot a dependency" error names the *right* dependency. The first version of that
error hardcoded `ImageFeatures` for every subtype, so `AKAZEGuess` without `AkazeFeatures` told the
caller to load a package that would not have helped.
"""
required_package(::ORBGuess) = "ImageFeatures"
required_package(::AKAZEGuess) = "AkazeFeatures"

# The detector itself, from the extension. Defined here so the error a caller sees names the missing
# dependency rather than being a bare `MethodError` on an internal function.
function _detector(g::FirstGuess)
    pkg = required_package(g)
    throw(ArgumentError(
        "`$(nameof(typeof(g)))` needs $pkg to be loaded. Run `using $pkg` first — the detector " *
        "lives in a package extension, because it is a heavy dependency that the optical path " *
        "does not need."))
end

# Scale an image into [0, 1] for a feature detector, into a caller-chosen element type.
#
# One implementation rather than one per extension: ImageFeatures wants `Gray{Float32}` (it dispatches
# on the colorant) and A-KAZE wants `Float64`, but the arithmetic is identical and
# `Gray{Float32}(::Float32)` is a valid conversion. Two copies of this meant two copies of the
# no-finite-range error message.
#
# Scaled by extrema over the *finite* values, since a reprojected scene carries NaN in its no-data
# border and `maximum` over those gives NaN for the whole image. This mirrors Muckenhuber's Eq. 2,
# which maps sigma0 onto 0-255 between user-chosen bounds specifically "to limit the influence of
# speckle noise"; their bounds are per-sensor constants, ours come from the data, which is the more
# conservative default. A caller with sensor-appropriate bounds should clamp before calling.
function _scale01(::Type{T}, img::AbstractMatrix) where {T}
    lo, hi = Inf32, -Inf32
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
    out = Matrix{T}(undef, size(img))
    @inbounds for i in eachindex(out)
        f = Float32(img[i])
        # Non-finite becomes zero rather than propagating: the detector has no mask, so a NaN would
        # poison every descriptor whose patch touched it.
        out[i] = T(isfinite(f) ? (f - lo) * scale : 0.0f0)
    end
    return out
end

"""
    AutoRIFT._match_features(reference, secondary, method::FirstGuess; kwargs...)
        -> (points, dx, dy)

Detect, describe and match features between two images. **The one method an extension supplies.**

`points` is a vector of `(row, col)`; `dx`/`dy` are the matched displacements, reference minus
secondary to match `track!`'s convention. Everything around this call — scaling, the shape check, the
consistency filter, `min_matches`, building the `PointSet` — belongs to [`first_guess`](@ref) and is
written once.

That split is the whole reason the abstract [`FirstGuess`](@ref) exists. It was not the first
arrangement: each extension originally owned a whole `first_guess` method, which duplicated the
keyword defaults, the sign convention, the NaN policy and two error messages per detector — so adding
a third meant a third copy, and one of them dispatched on the *abstract* type and silently claimed
every future subtype.
"""
function _match_features end

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
- Detector-specific keywords are forwarded to the matcher; see [`ORBGuess`](@ref) and
  [`AKAZEGuess`](@ref).

!!! tip "Load `NearestNeighbors` too"
    The consistency filter's neighbour search is O(n²) without it and O(n log n) with — 90x at 20000
    matches, and A-KAZE produces tens of thousands. It is a weak dependency of this package but *not*
    installed by loading a detector, so `using NearestNeighbors` is a separate step and worth taking
    for anything beyond a few thousand matches.
"""
function first_guess(reference::AbstractMatrix, secondary::AbstractMatrix, method::FirstGuess;
                     search_radius = 6, chip_size = 32, min_matches::Integer = 8,
                     neighbours::Integer = 12, tolerance::Real = 6.0, min_agree::Integer = 5,
                     kwargs...)
    size(reference) == size(secondary) || throw(DimensionMismatch(
        "reference is $(size(reference)) but secondary is $(size(secondary)); the two images " *
        "must be co-registered to a common grid"))

    pts, dxs, dys = _match_features(reference, secondary, method; kwargs...)
    isempty(pts) && throw(ArgumentError(
        "feature matching found no correspondences at all. The pair may be fully decorrelated, " *
        "or the time gap too long for features to persist."))

    keep = consistent_matches(pts, dxs, dys; neighbours, tolerance, min_agree)
    length(keep) >= min_matches || throw(ArgumentError(
        "only $(length(keep)) of $(length(pts)) matches survived the consistency filter, below " *
        "`min_matches = $min_matches`. Raw descriptor matching on speckle is mostly wrong — " *
        "9.5-62.5% correct in testing — so a low survival count is normal, but this few means no " *
        "coherent motion field was found. Check the time gap, or loosen `tolerance`."))

    # Row is y and column is x, which is the one place this conversion has to be right: matchers
    # return `(row, col)` and `PointSet` takes `(x, y)`.
    return pointset([p[2] for p in pts[keep]], [p[1] for p in pts[keep]];
                    search_radius, chip_size,
                    dx_prior = dxs[keep], dy_prior = dys[keep])
end

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

"""
    scene_rotation(guess::PointSet) -> Float64
    scene_rotation(x, y, dx, dy) -> Float64

The single rotation, in degrees, that best explains a sparse displacement field.

Feed it to [`RotationSearch`](@ref)'s `about` so the narrow per-chip angle search is centred on the
scene's actual rotation rather than on zero:

```julia
guess = first_guess(a, b, AKAZEGuess())
out = autorift(a, b, guess; rotation = RotationSearch(; about = scene_rotation(guess)))
```

# Why this rather than the reference's version

`sea_ice_drift` computes `alpha0` in `get_initial_rotation` from **geolocation**: the bearing between
two corners of the secondary scene reprojected into the reference's grid. For a co-registered pair —
which is what [`first_guess`](@ref) requires and checks — that is identically zero, so porting it
would have produced a function that always returns 0.0. The rotation that is actually nonzero here is
the *ice*'s, and the sparse vectors already measure it.

# What it is

An orthogonal Procrustes fit, which for 2D reduces to one `atan2` over two sums — the same shape as
[`Deramp`](@ref)'s estimator and for the same reason: summing the cross and dot products before
taking the angle makes the estimate a least-squares fit over all points at once, with no wrapping and
no per-point angle to average.

Given reference positions `p` and displacements `d` (reference minus secondary, so the secondary
position is `p - d`), the returned angle is the rotation carrying **secondary orientation onto
reference**:

```
θ = atan2(Σ (sx·ty - sy·tx), Σ (sx·tx + sy·ty))
```

with `s` the centred `p - d` and `t` the centred `p`.

That direction, rather than reference-onto-secondary, for the same reason `dx`/`dy` are reference
minus secondary: it is the negative of the imaged features' own motion. So the sign here is
consistent with the rest of the package, and it is what [`RotationSearch`](@ref)'s `about` consumes —
the chip comes from the secondary and must be turned back, which is why that field is *subtracted*.
Pinned by test in both directions, since a sign error would centre the search on the wrong side of
the truth and be twice as wrong as not centring it at all.

# Limitations, stated because they are the reason this is a separate opt-in step

- **Translation is removed, scale is not fitted.** A rigid rotation plus translation is the model;
  divergence and shear are residuals. That is the same model `RotationSearch` itself assumes.
- **One rotation for the whole scene.** A field with two floes rotating opposite ways fits to
  something near their average, which describes neither. The per-chip search is what handles that,
  and `about` only moves where it starts looking.
- **Returns `NaN`** when there is nothing to fit: fewer than two points, or a degenerate
  configuration (all points coincident, or the fit's two sums both zero). `RotationSearch` rejects a
  non-finite `about` rather than silently searching around zero, so the caller must decide.
"""
function scene_rotation(x::AbstractVector, y::AbstractVector, dx::AbstractVector,
                        dy::AbstractVector)
    n = length(x)
    (length(y) == n && length(dx) == n && length(dy) == n) || throw(DimensionMismatch(
        "x, y, dx and dy must be the same length, got $n, $(length(y)), $(length(dx)) and " *
        "$(length(dy))"))
    # Two points define a rotation; one defines only a translation.
    n >= 2 || return NaN

    # Centroids of both point sets. Removing them is what makes this a fit for rotation *alone* —
    # otherwise a pure translation would masquerade as a rotation about a distant centre.
    sx = sy = tx = ty = 0.0
    @inbounds for i in 1:n
        px, py = Float64(x[i]), Float64(y[i])
        tx += px
        ty += py
        sx += px - Float64(dx[i])
        sy += py - Float64(dy[i])
    end
    sx /= n; sy /= n; tx /= n; ty /= n

    cross = dot = 0.0
    @inbounds for i in 1:n
        px, py = Float64(x[i]), Float64(y[i])
        # Source: where the feature is in the secondary. Target: where it is in the reference.
        a = px - Float64(dx[i]) - sx
        b = py - Float64(dy[i]) - sy
        c = px - tx
        d = py - ty
        cross += a * d - b * c
        dot += a * c + b * d
    end
    # Both sums zero means every centred vector vanished — coincident points, or a field that is
    # pure divergence with no rotational component to find. `atan2(0, 0)` is 0.0, which would be a
    # confident wrong answer.
    (cross == 0.0 && dot == 0.0) && return NaN
    return rad2deg(atan(cross, dot))
end

scene_rotation(pts::PointSet) =
    scene_rotation(vec(pts.x), vec(pts.y), vec(pts.dx_prior), vec(pts.dy_prior))

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
# the sea-ice path, and a core dependency for one optional stage is the wrong trade. It is **not**
# pulled in by loading a detector — extension triggers are a conjunction of what the user loaded, not
# an install directive — so `first_guess`'s docstring names it.
#
# Dispatched on a *strategy* argument, and the reason is worth recording because the obvious
# spellings both fail. Verified in a six-line reproduction package, not inferred:
#
#   * **Extension redefines the same signature.** Does not extend — it *overwrites*, and Julia then
#     refuses: `Method overwriting is not permitted during Module precompilation`. Same for a
#     zero-argument `_neighbour_strategy()`, which was the first thing tried here.
#   * **Extension defines a method on a differently-named function, core checks `hasmethod`.** Looks
#     tidiest and is **unsound**: the test package reported the fast path as available with the
#     extension *not* loaded, because `hasmethod` is a runtime query against a world age that a
#     precompiled image can already have advanced past. A wrong answer is worse than an indirection.
#   * **Extension narrows the element type.** Works, but the axis is wrong — availability of a
#     dependency has nothing to do with whether the points are `Float64`.
#
# So: a fallback on the abstract type that the extension specialises. Two distinct methods, resolved
# by ordinary dispatch, correct in both states — verified `_BruteForceNeighbours` without
# `NearestNeighbors` and `KDTreeNeighbours` with it.
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
