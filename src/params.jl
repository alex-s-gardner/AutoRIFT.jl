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

const SYMBOL2QUANTIZE = Dict{Symbol,QuantizeMethod}(
    :uint8 => QuantizeUInt8(),
    :none  => NoQuantize(),
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
_similarity(x::AbstractVector) = _similarity(Tuple(x))
_similarity(x) = _badtype(:similarity, x,
                         "a Symbol, a `SimilarityMeasure`, or a tuple of either")

# One element of a measure tuple. Separate from `_similarity` because that returns a tuple and
# this must not, or nesting would compound.
_one_similarity(x::SimilarityMeasure) = x
_one_similarity(x::Symbol) = _resolve(SYMBOL2SIMILARITY, x, :similarity)
_one_similarity(x) = _badtype(:similarity, x, "a Symbol or a `SimilarityMeasure`")

_quantize(x::QuantizeMethod) = x
_quantize(x::Symbol) = _resolve(SYMBOL2QUANTIZE, x, :quantize)
_quantize(x) = _badtype(:quantize, x, "a Symbol or a `QuantizeMethod`")

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

_badtype(kw::Symbol, x, expected) =
    throw(ArgumentError("`$kw` must be $expected, got a $(typeof(x))."))

# ---------------------------------------------------------------------------
# Numeric validation
# ---------------------------------------------------------------------------

# Chip sizes must be multiples of 4. Two independent reasons: the chip needs an
# even half-extent in each direction, and the chip size halves at each
# level, so an odd multiple would break the finest level's grid alignment.
function _check_chip_size(name::Symbol, cs::Integer)
    cs > 0 || throw(ArgumentError("`$name` must be positive, got $cs"))
    cs % 4 == 0 ||
        throw(ArgumentError("`$name` must be a multiple of 4, got $cs. Chip " *
                            "sizes are halved at each chip-size level and need " *
                            "an even half-extent in both directions."))
    return Int(cs)
end

_pair(x::Integer) = (Int(x), Int(x))
_pair(x::Tuple{Integer,Integer}) = (Int(x[1]), Int(x[2]))
_pair(x::AbstractVector) = length(x) == 2 ? (Int(x[1]), Int(x[2])) :
    throw(ArgumentError("expected 1 or 2 values, got $(length(x))"))

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
- `quantize = :uint8`: element type used for correlation; see [`QuantizeMethod`](@ref).
- `subpixel = :pyramid`: peak refinement; see [`SubpixelMethod`](@ref).

## Chip and grid geometry, in pixels
- `chip_size = 32`: finest chip size. Must be a multiple of 4.
- `chip_size_min = chip_size`, `chip_size_max = 4 * chip_size`: range of chip sizes tried.
  Levels are `chip_size * 2^k` within these bounds.
- `chip_aspect = 1.0`: chip height as a multiple of width.
- `grid_spacing = chip_size`: spacing of output grid points.

## Search geometry, in pixels
- `search_radius = 25`: half-extent of the search window. An `Int` sets both
  axes; a `Tuple` sets them separately, as do `search_radius_x` and
  `search_radius_y`. The window spans `2 * radius` in each direction.
- `min_search_radius = 6`: floor applied to non-zero radii, per axis.
- `dx_prior = 0.0`, `dy_prior = 0.0`: a-priori displacement the search window is
  centred on.
- `coarse_stride = 4`: decimation factor for the coarse pass.
- `coarse_buffer = 8`: dilation radius applied to the coarse validity mask
  before it restricts the fine search.
- `min_coarse_valid_fraction = 0.01`: skip a chip-size level whose coarse pass
  validates a smaller fraction than this.

## Outlier rejection and filling
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
- `rng_seed = 0`: seed for gap filling, so results are reproducible.
"""
function params(;
    similarity = :zncc,
    preprocess = :highpass,
    quantize = :uint8,
    subpixel = :pyramid,
    filter_width = nokw,
    upsampling = nokw,
    chip_size = 32,
    chip_size_min = nokw,
    chip_size_max = nokw,
    chip_aspect = 1.0,
    grid_spacing = nokw,
    search_radius = 25,
    search_radius_x = nokw,
    search_radius_y = nokw,
    min_search_radius = 6,
    dx_prior = 0.0,
    dy_prior = 0.0,
    coarse_stride = 4,
    coarse_buffer = 8,
    min_coarse_valid_fraction = 0.01,
    outliers = :gardner,
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
    quant = _quantize(quantize)
    sub = _subpixel(subpixel, upsampling)
    # The public keywords are prefixed (`outlier_window`) where the method's own are not
    # (`window`), because at the top level `window` alone would be ambiguous against
    # `fill_window` and `filter_width`. Translated here, in the one file that owns keywords.
    out = _outliers(outliers, (; window = outlier_window, iterations = outlier_iterations,
                               min_agree_fraction, agree_tolerance, mad_scale))

    base = _check_chip_size(:chip_size, chip_size)
    cmin = isnokw(chip_size_min) ? base : _check_chip_size(:chip_size_min, chip_size_min)
    cmax = isnokw(chip_size_max) ? 4base : _check_chip_size(:chip_size_max, chip_size_max)

    cmin <= cmax || throw(ArgumentError(
        "`chip_size_min` ($cmin) must be <= `chip_size_max` ($cmax)"))
    base <= cmin || throw(ArgumentError(
        "`chip_size` ($base) must be <= `chip_size_min` ($cmin). `chip_size` " *
        "is the finest chip-size level; coarser levels are its powers-of-two " *
        "multiples."))
    cmax % base == 0 || throw(ArgumentError(
        "`chip_size_max` ($cmax) must be a power-of-two multiple of " *
        "`chip_size` ($base), because chip-size levels double at each step."))
    ispow2(cmax ÷ base) || throw(ArgumentError(
        "`chip_size_max` ($cmax) must be a power-of-two multiple of " *
        "`chip_size` ($base), got a ratio of $(cmax ÷ base)."))

    spacing = isnokw(grid_spacing) ? base :
        Int(_check_positive(:grid_spacing, grid_spacing))

    # `search_radius` sets both axes; the per-axis keywords override it.
    rx0, ry0 = _pair(search_radius)
    rx = isnokw(search_radius_x) ? rx0 : Int(search_radius_x)
    ry = isnokw(search_radius_y) ? ry0 : Int(search_radius_y)
    rx >= 0 || throw(ArgumentError("`search_radius_x` must be >= 0, got $rx"))
    ry >= 0 || throw(ArgumentError("`search_radius_y` must be >= 0, got $ry"))
    (rx > 0 || ry > 0) || throw(ArgumentError(
        "`search_radius` is zero in both axes, so no pixel can be searched."))

    return Params(
        sim, pre, quant, sub, out, booltype(threaded),
        base, cmin, cmax, Float64(_check_positive(:chip_aspect, chip_aspect)),
        spacing,
        rx, ry, Int(min_search_radius),
        Int(_check_positive(:coarse_stride, coarse_stride)),
        Int(coarse_buffer),
        _check_fraction(:min_coarse_valid_fraction, min_coarse_valid_fraction),
        Float64(dx_prior), Float64(dy_prior),
        _check_odd_window(:fill_window, fill_window),
        UInt64(rng_seed),
        Bool(progress),
    )
end
