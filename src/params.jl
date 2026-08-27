# Keyword resolution and validation.
#
# This is the ONLY file where a `Symbol` is interpreted. Everything downstream
# receives a `Params` whose method choices are type parameters, so no hot-path
# code branches on a Symbol or re-validates a number.
#
# The Symbol -> method mapping follows the pattern Rasters uses for its `source`
# keyword: a Symbol is the documented, convenient spelling, and the method
# object it maps to is equally acceptable and is the only way to pass a
# method-specific parameter.

const SYMBOL2SIMILARITY = Dict{Symbol,SimilarityMeasure}(
    :zncc      => ZNCC(),
    :ncc       => NCC(),
    :coherence => Coherence(),
)

# Preprocessing and subpixel Symbols map to *constructors*, not instances,
# because their defaults depend on other keywords (`filter_width`, `upsampling`).
const SYMBOL2PREPROCESS = Dict{Symbol,Any}(
    :highpass       => Highpass,
    :wallis         => Wallis,
    :wallis_gapfill => WallisGapfill,
    :sobel          => Sobel,
    :laplacian      => Laplacian,
    :decibel        => Decibel,
    :deramp         => Deramp,
    :none           => NoPreprocess,
)

const SYMBOL2SUBPIXEL = Dict{Symbol,Any}(
    :pyramid => PyramidRefine,
    :none    => NoRefine,
)

# Also constructors, since `GardnerFilter` takes the five `outlier_*`/`*_scale` keywords.
const SYMBOL2OUTLIERS = Dict{Symbol,Any}(
    :gardner => GardnerFilter,
    :none    => NoOutlierFilter,
)

# ---------------------------------------------------------------------------
# Symbol resolution
# ---------------------------------------------------------------------------

# One generic resolver rather than four near-identical ones. `kw` is only used
# to build the error message, so a wrong Symbol names the keyword it came from
# and lists what would have been accepted.
function _resolve(table::AbstractDict, x::Symbol, kw::Symbol)
    haskey(table, x) && return table[x]
    valid = join(map(k -> ":$k", sort!(collect(keys(table)))), ", ")
    throw(ArgumentError("`$kw = :$x` is not recognised. Valid options are $valid."))
end

# `similarity` resolves to a *tuple*, one measure per chip-size level. A scalar becomes a 1-tuple,
# whose last entry then applies to every level — so the single-measure case, which is nearly every
# call, is unchanged and stays concretely typed.
_similarity(x::SimilarityMeasure) = (x,)
_similarity(x::Symbol) = (_resolve(SYMBOL2SIMILARITY, x, :similarity),)
# `map` rather than a comprehension: it preserves the tuple, so `S` remains a concrete
# `Tuple{Coherence,ZNCC}` rather than a `Vector{SimilarityMeasure}` the kernels cannot specialize
# on. An empty tuple would type-check and then silently correlate nothing.
function _similarity(x::Tuple)
    isempty(x) && throw(ArgumentError(
        "`similarity` cannot be an empty tuple; name at least one measure."))
    return map(_one_similarity, x)
end
# Deliberately no `AbstractVector` method. It would work, but its return type infers as `Any` —
# `Tuple(::Vector)` has no length in its type — which forfeits exactly the concreteness the tuple
# design exists to preserve, and would be the one accepted spelling that silently costs the
# specialization. A caller with a vector writes `Tuple(v)`, and sees why at the call site.
_similarity(x) = _badtype(:similarity, x,
                         "a Symbol, a `SimilarityMeasure`, or a tuple of either")

# One element of a measure tuple. Separate from `_similarity` because that returns a tuple and
# this must not, or nesting would compound.
_one_similarity(x::SimilarityMeasure) = x
_one_similarity(x::Symbol) = _resolve(SYMBOL2SIMILARITY, x, :similarity)
_one_similarity(x) = _badtype(:similarity, x, "a Symbol or a `SimilarityMeasure`")

_preprocess(x::PreprocessMethod, _width) = x
function _preprocess(x::Symbol, width)
    T = _resolve(SYMBOL2PREPROCESS, x, :preprocess)
    # Methods with no window ignore `filter_width` entirely; the rest take it. Asked via
    # `filter_width` rather than by restating the list of windowless methods here: the trait already
    # encodes it, and a second copy meant adding `Deramp` in two places — where forgetting the second
    # would have made `preprocess = :deramp` call `Deramp(; width = ...)` and throw a `MethodError`
    # from a keyword combination rather than a clear error.
    #
    # `T()` is constructed first because the trait is a property of the instance. Every method here
    # has a zero-argument constructor, which is what makes that safe.
    return filter_width(T()) == 0 ? T() :
           isnokw(width) ? T() : T(; width)
end
_preprocess(x, _width) = _badtype(:preprocess, x, "a Symbol or a `PreprocessMethod`")

_subpixel(x::SubpixelMethod, _upsampling) = x
function _subpixel(x::Symbol, up)
    T = _resolve(SYMBOL2SUBPIXEL, x, :subpixel)
    return T === NoRefine ? NoRefine() :
           isnokw(up) ? T() : T(; upsampling = up)
end
_subpixel(x, _upsampling) = _badtype(:subpixel, x, "a Symbol or a `SubpixelMethod`")

# An instance already carries its own parameters, so the loose keywords would have nothing to
# apply to. Passing both is a contradiction rather than a merge, and saying so beats silently
# preferring one — the alternative is a caller who sets `mad_scale` next to a `GardnerFilter`
# and never learns it was ignored.
function _outliers(x::OutlierMethod, kw::NamedTuple)
    _no_stray_params(x, kw, "a $(typeof(x)) instance, which already carries its own " *
                     "parameters")
    return x
end

function _outliers(x::Symbol, kw::NamedTuple)
    T = _resolve(SYMBOL2OUTLIERS, x, :outliers)
    # A method with no parameters is checked too, since the keywords are just as ignored there
    # and just as likely to be a mistake — but reported against what the caller actually wrote.
    T === NoOutlierFilter && return (_no_stray_params(T(), kw, "`:$x`, which takes no " *
                                                     "parameters"); NoOutlierFilter())

    # The all-default case, which is almost every call, and it is worth its own branch:
    # `params` runs once per image pair across tens of millions of them. `T` comes out of a
    # `Dict{Symbol,Any}` so it is a runtime value, which makes `T(; ...)` a dynamic call, and
    # splatting a generator whose length is not statically known costs ~700 ns and 8
    # allocations even when it forwards nothing. Measured: 1412 ns for `params()` against 744
    # before, entirely here. Naming the concrete type in the common branch makes it a static
    # call at ~1 ns.
    #
    # `all` over a `NamedTuple` of sentinel types is decided at compile time, since `isnokw`
    # dispatches on `NoKW` and each field's type is known — so the branch itself is free.
    if all(isnokw, kw)
        T === GardnerFilter && return GardnerFilter()
        return T()
    end
    # Only forward what the caller actually set, so each method keeps its own defaults rather
    # than having this file restate them. The splat is paid only when a caller overrides
    # something, which is rare and not on any per-pair path.
    return T(; (k => kw[k] for k in keys(kw) if !isnokw(kw[k]))...)
end

# The loose `outlier_*` keywords apply only to a method built from them. Anywhere else they are
# silently dead, which is the failure this prevents: a caller who sets `mad_scale` beside an
# explicit method would otherwise never learn it did nothing.
function _no_stray_params(_method, kw::NamedTuple, what::AbstractString)
    given = filter(k -> !isnokw(kw[k]), keys(kw))
    isempty(given) && return nothing
    throw(ArgumentError(
        "`outliers` was given $what, so $(join(map(k -> "`$k`", given), ", ")) cannot " *
        "also apply. Set them on the method instead, or drop them."))
end

_outliers(x, _kw) = _badtype(:outliers, x, "a Symbol or an `OutlierMethod`")

# `nothing`/`false` off, `true` the paper's default, a tuple the explicit angles. Accepting a `Bool`
# because "should I search rotations" reads as a yes/no question at the call site even though the
# answer carries parameters.
_rotation(::Nothing) = NoRotationSearch()
_rotation(x::Bool) = x ? RotationSearch() : NoRotationSearch()
_rotation(x::RotationMethod) = x
# One method over any real-valued iterable rather than a `Tuple` method plus an `AbstractVector` one:
# `RotationSearch`'s inner constructor already does `Tuple(Float64.(angles))`, so both were
# duplicating its work — and the pair silently rejected a *range*, which is the natural spelling for
# `-6:3:6`.
#
# Constrained to `Real` eltype rather than left as a bare fallback. `RotationSearch("yes")` does throw
# on its own, but as `MethodError: no method matching Float64(::String)` — which says nothing about
# which keyword was wrong. `_badtype` names it.
_rotation(x::Union{Tuple{Vararg{Real}},AbstractVector{<:Real},AbstractRange{<:Real}}) =
    RotationSearch(x)
_rotation(x) = _badtype(:rotation, x,
                       "`nothing`, a `Bool`, a collection of angles, or a `RotationMethod`")

_badtype(kw::Symbol, x, expected) =
    throw(ArgumentError("`$kw` must be $expected, got a $(typeof(x))."))

# ---------------------------------------------------------------------------
# Numeric validation
# ---------------------------------------------------------------------------

# Chip sizes must be multiples of 4 in each axis. A chip's centre has to land on a pixel boundary
# for its displacement to be reported at one, which needs an even extent; and a chip halved once
# still needs an even half-extent, which is the second factor of two.
function _check_chip_size(name::Symbol, cs)
    e = extent(cs)
    for (ax, v) in ((:X, e.X), (:Y, e.Y))
        v > 0 || throw(ArgumentError("`$name` must be positive, got $v in $ax"))
        v % 4 == 0 || throw(ArgumentError(
            "`$name` must be a multiple of 4, got $v in $ax. A chip's centre lands on a pixel " *
            "boundary only for an even extent, and a chip halved once needs an even half-extent."))
    end
    return e
end

# Grid spacing: positive in each axis. Nothing divides by it in `src/`, so there is no further
# constraint — `block_layout` walks the grid's own coordinates rather than assuming a spacing.
function _check_spacing(sp)
    e = extent(sp)
    for (ax, v) in ((:X, e.X), (:Y, e.Y))
        v > 0 || throw(ArgumentError("`grid_spacing` must be positive, got $v in $ax"))
    end
    return e
end

# Search radius: zero is allowed per axis — that is how a caller searches along one axis only — but
# not in both, which would leave no pixel searchable.
function _check_radius(r)
    e = extent(r)
    for (ax, v) in ((:X, e.X), (:Y, e.Y))
        v >= 0 || throw(ArgumentError("`search_radius` must be >= 0, got $v in $ax"))
    end
    (e.X > 0 || e.Y > 0) || throw(ArgumentError(
        "`search_radius` is zero in both axes, so no pixel can be searched."))
    return e
end

# The levels `chip_size .* 2^k` must reach `chip_size_max` in both axes at the same k, so the aspect
# a caller asked for holds at every level rather than drifting across them. That is what makes a
# coarse grid point exactly a block of fine ones — see `src/multichip.jl`.
#
# So `(X=16, Y=32)` to `(X=32, Y=64)` is fine, both ratios being 2, while `(X=16, Y=16)` to
# `(X=128, Y=64)` is not: x would reach its maximum after three doublings and y after two.
function _check_levels(cmin::Extent, cmax::Extent)
    (cmin.X <= cmax.X && cmin.Y <= cmax.Y) || throw(ArgumentError(
        "`chip_size` ($(cmin.X) by $(cmin.Y)) must not exceed `chip_size_max` " *
        "($(cmax.X) by $(cmax.Y)) in either axis."))
    rx, ry = cmax.X ÷ cmin.X, cmax.Y ÷ cmin.Y
    (cmax.X % cmin.X == 0 && cmax.Y % cmin.Y == 0 && ispow2(rx) && ispow2(ry)) ||
        throw(ArgumentError(
            "`chip_size_max` must be a power-of-two multiple of `chip_size` in each axis, " *
            "because chip-size levels double at each step. Got $(cmin.X) to $(cmax.X) in X and " *
            "$(cmin.Y) to $(cmax.Y) in Y."))
    rx == ry || throw(ArgumentError(
        "`chip_size_max` must be the same multiple of `chip_size` in both axes, so the chip's " *
        "aspect ratio is the same at every level. Got $rx in X and $ry in Y — x would reach its " *
        "maximum after a different number of doublings than y."))
    return nothing
end

function _check_fraction(name::Symbol, v::Real)
    0 <= v <= 1 ||
        throw(ArgumentError("`$name` must be in [0, 1], got $v"))
    return Float64(v)
end

function _check_positive(name::Symbol, v::Real)
    v > 0 || throw(ArgumentError("`$name` must be positive, got $v"))
    return v
end

function _check_odd_window(name::Symbol, w::Integer)
    w >= 1 || throw(ArgumentError("`$name` must be >= 1, got $w"))
    isodd(w) ||
        throw(ArgumentError("`$name` must be odd so the window has a centre " *
                            "pixel, got $w"))
    return Int(w)
end

# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------

"""
    AutoRIFT.params(; kwargs...) -> Params

Resolve and validate user keywords into a concrete [`Params`](@ref).

Called once per `autorift` invocation. Every `Symbol` is mapped to a method
object and every number range-checked here, so failures surface at the API
boundary naming the offending keyword rather than deep inside a chip-size level.

Defaults match the autoRIFT reference driver, except where the reference's
choice is a known defect — see the package documentation for the list.

# Keywords

## Method selection
- `similarity = :zncc`: [`ZNCC`](@ref), [`NCC`](@ref), or [`Coherence`](@ref).

  A **tuple** assigns measures to chip-size levels in order, with the last repeated for any
  remaining levels — so `(:coherence, :zncc)` tries complex coherence at the finest chip and
  falls back to amplitude at every coarser one. That is the escalation of Joughin (2002):
  coherence resolves finer detail but is destroyed by phase variation, so points it cannot
  resolve are left to a larger amplitude chip. Requires complex input; see [`Coherence`](@ref).
- `preprocess = :highpass`: pre-correlation filter; see [`PreprocessMethod`](@ref).
- `subpixel = :pyramid`: peak refinement; see [`SubpixelMethod`](@ref).

## Chip and grid geometry, in pixels

Each of these is an **extent** — a quantity with an x and a y component. A scalar means square, and
`(X = …, Y = …)` names the axes:

```julia
chip_size = 32                 # 32 by 32
chip_size = (X = 16, Y = 32)   # taller than wide
```

- `chip_size = 32`: the finest chip-size level. Each axis must be a multiple of 4, so a chip's
  centre lands on a pixel boundary and a chip halved once still has an even half-extent.
- `chip_size_max = 4 * chip_size`: the coarsest level. Levels are `chip_size .* 2^k` up to this, so
  `chip_size_max` must be the **same** power-of-two multiple of `chip_size` in both axes — the chip's
  aspect ratio is therefore the same at every level, which is what makes a coarse grid point exactly
  a block of fine ones.
- `grid_spacing = chip_size`: spacing of output grid points.

## Search geometry, in pixels
- `search_radius = 25`: half-extent of the search window, as an extent. The window spans
  `2 * radius` in each direction. Zero in one axis searches along the other only; zero in both is an
  error, since no pixel could be searched.
- `min_search_radius = 6`: floor applied to non-zero radii, per axis.
- `dx_prior = 0.0`, `dy_prior = 0.0`: a-priori displacement the search window is
  centred on.
- `coarse_stride = 4`: decimation factor for the coarse pass.
- `coarse_buffer = 8`: dilation radius applied to the coarse validity mask
  before it restricts the fine search.
- `min_coarse_valid_fraction = 0.01`: skip a chip-size level whose coarse pass
  validates a smaller fraction than this.

## Outlier rejection and filling
- `rotation = nothing`: try several chip rotations and keep the best, per
  [`RotationSearch`](@ref). `nothing` or `false` disables it, which is the default because it
  multiplies correlation cost by the number of angles. `true` uses ±3°, following
  `nansencenter/sea_ice_drift`; a tuple names the angles. Only useful where the ice rotates —
  see [`RotationSearch`](@ref) for the measurement. To centre a narrow angle window on a scene
  that is already rotated, build the method with `about`: `RotationSearch(; about =
  scene_rotation(guess))`. See [`scene_rotation`](@ref).
- `outliers = :gardner`: which implausible displacements to drop. `:gardner` is the
  two-stage filter of Gardner et al. (2018) that autoRIFT uses; `:none` keeps everything,
  which is useful for telling the correlator's failures from the filter's rejections. An
  [`OutlierMethod`](@ref) instance is also accepted.
- `outlier_window = 5`: neighbourhood width for the filter. Must be odd.
- `outlier_iterations = 3`: iterations of the filter's first stage.
- `min_agree_fraction = 8/25`: fraction of neighbours that must agree.
- `agree_tolerance = 0.2`: agreement threshold, as a fraction of search radius.
- `mad_scale = 4.0`: median-absolute-deviation multiplier.
- `fill_window = 3`: window for median hole filling. Must be odd.

The five keywords after `outliers` are [`GardnerFilter`](@ref)'s own parameters, offered at
the top level for convenience. They cannot be combined with an `outliers` *instance*, which
already carries its own — that is an error rather than a silent override.

## Misc
- `threaded = false`: parallelise over grid points within one image pair. For
  batch processing, leaving this `false` and running one pair per worker is
  usually faster; see the documentation on throughput.
- `progress = false`: show a progress meter.
- `rng_seed = 0`: seed for the noise [`WallisGapfill`](@ref) fills gaps with, so a run is
  reproducible. Read by that filter alone; every other `preprocess` choice ignores it.
"""
function params(;
    similarity = :zncc,
    preprocess = :highpass,
    subpixel = :pyramid,
    filter_width = nokw,
    upsampling = nokw,
    chip_size = 32,
    chip_size_max = nokw,
    grid_spacing = nokw,
    search_radius = 25,
    min_search_radius = 6,
    dx_prior = 0.0,
    dy_prior = 0.0,
    coarse_stride = 4,
    coarse_buffer = 8,
    min_coarse_valid_fraction = 0.01,
    outliers = :gardner,
    rotation = nothing,
    outlier_window = nokw,
    outlier_iterations = nokw,
    min_agree_fraction = nokw,
    agree_tolerance = nokw,
    mad_scale = nokw,
    fill_window = 3,
    threaded = false,
    progress = false,
    rng_seed = 0,
)
    sim = _similarity(similarity)
    pre = _preprocess(preprocess, filter_width)
    sub = _subpixel(subpixel, upsampling)
    # The public keywords are prefixed (`outlier_window`) where the method's own are not
    # (`window`), because at the top level `window` alone would be ambiguous against
    # `fill_window` and `filter_width`. Translated here, in the one file that owns keywords.
    out = _outliers(outliers, (; window = outlier_window, iterations = outlier_iterations,
                               min_agree_fraction, agree_tolerance, mad_scale))
    rot = _rotation(rotation)

    cmin = _check_chip_size(:chip_size, chip_size)
    cmax = isnokw(chip_size_max) ? (X = 4cmin.X, Y = 4cmin.Y) :
        _check_chip_size(:chip_size_max, chip_size_max)
    _check_levels(cmin, cmax)

    spacing = isnokw(grid_spacing) ? cmin : _check_spacing(grid_spacing)
    rad = _check_radius(search_radius)

    return Params(
        sim, pre, sub, out, booltype(threaded), rot,
        cmin, cmax, spacing, rad, Int(min_search_radius),
        Int(_check_positive(:coarse_stride, coarse_stride)),
        Int(coarse_buffer),
        _check_fraction(:min_coarse_valid_fraction, min_coarse_valid_fraction),
        Float64(dx_prior), Float64(dy_prior),
        _check_odd_window(:fill_window, fill_window),
        UInt64(rng_seed),
        Bool(progress),
    )
end
