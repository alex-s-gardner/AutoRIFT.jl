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
    :none           => NoPreprocess,
)

const SYMBOL2SUBPIXEL = Dict{Symbol,Any}(
    :pyramid => PyramidRefine,
    :none    => NoRefine,
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

_similarity(x::SimilarityMeasure) = x
_similarity(x::Symbol) = _resolve(SYMBOL2SIMILARITY, x, :similarity)
_similarity(x) = _badtype(:similarity, x, "a Symbol or a `SimilarityMeasure`")

_quantize(x::QuantizeMethod) = x
_quantize(x::Symbol) = _resolve(SYMBOL2QUANTIZE, x, :quantize)
_quantize(x) = _badtype(:quantize, x, "a Symbol or a `QuantizeMethod`")

_preprocess(x::PreprocessMethod, _width) = x
function _preprocess(x::Symbol, width)
    T = _resolve(SYMBOL2PREPROCESS, x, :preprocess)
    # Methods with no window ignore `filter_width` entirely; the rest take it.
    return T <: Union{NoPreprocess,Decibel} ? T() :
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

_badtype(kw::Symbol, x, expected) =
    throw(ArgumentError("`$kw` must be $expected, got a $(typeof(x))."))

# ---------------------------------------------------------------------------
# Numeric validation
# ---------------------------------------------------------------------------

# Chip sizes must be multiples of 4. Two independent reasons: the chip needs an
# even half-extent in each direction, and the pyramid halves chip size at each
# level, so an odd multiple would break the finest level's grid alignment.
function _check_chip_size(name::Symbol, cs::Integer)
    cs > 0 || throw(ArgumentError("`$name` must be positive, got $cs"))
    cs % 4 == 0 ||
        throw(ArgumentError("`$name` must be a multiple of 4, got $cs. Chip " *
                            "sizes are halved at each pyramid level and need " *
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
boundary naming the offending keyword rather than deep inside a pyramid level.

Defaults match the autoRIFT reference driver, except where the reference's
choice is a known defect — see the package documentation for the list.

# Keywords

## Method selection
- `similarity = :zncc`: [`ZNCC`](@ref), [`NCC`](@ref), or [`Coherence`](@ref).
- `preprocess = :highpass`: pre-correlation filter; see [`PreprocessMethod`](@ref).
- `quantize = :uint8`: element type used for correlation; see [`QuantizeMethod`](@ref).
- `subpixel = :pyramid`: peak refinement; see [`SubpixelMethod`](@ref).

## Chip and grid geometry, in pixels
- `chip_size = 32`: finest chip size. Must be a multiple of 4.
- `chip_size_min = chip_size`, `chip_size_max = 4 * chip_size`: pyramid extent.
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
- `min_coarse_valid_fraction = 0.01`: skip a pyramid level whose coarse pass
  validates a smaller fraction than this.

## Outlier rejection and filling
- `outlier_window = 5`: window for the normalized median test. Must be odd.
- `outlier_iterations = 3`: iterations of the test.
- `min_agree_fraction = 8/25`: fraction of neighbours that must agree.
- `agree_tolerance = 0.2`: agreement threshold, as a fraction of search radius.
- `mad_scale = 4.0`: median-absolute-deviation multiplier.
- `fill_window = 3`: window for median hole filling. Must be odd.

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
    outlier_window = 5,
    outlier_iterations = 3,
    min_agree_fraction = 8 / 25,
    agree_tolerance = 0.2,
    mad_scale = 4.0,
    fill_window = 3,
    threaded = false,
    progress = false,
    rng_seed = 0,
)
    sim = _similarity(similarity)
    pre = _preprocess(preprocess, filter_width)
    quant = _quantize(quantize)
    sub = _subpixel(subpixel, upsampling)

    base = _check_chip_size(:chip_size, chip_size)
    cmin = isnokw(chip_size_min) ? base : _check_chip_size(:chip_size_min, chip_size_min)
    cmax = isnokw(chip_size_max) ? 4base : _check_chip_size(:chip_size_max, chip_size_max)

    cmin <= cmax || throw(ArgumentError(
        "`chip_size_min` ($cmin) must be <= `chip_size_max` ($cmax)"))
    base <= cmin || throw(ArgumentError(
        "`chip_size` ($base) must be <= `chip_size_min` ($cmin). `chip_size` " *
        "is the finest pyramid level; coarser levels are its powers-of-two " *
        "multiples."))
    cmax % base == 0 || throw(ArgumentError(
        "`chip_size_max` ($cmax) must be a power-of-two multiple of " *
        "`chip_size` ($base), because pyramid levels double at each step."))
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
        sim, pre, quant, sub, booltype(threaded),
        base, cmin, cmax, Float64(_check_positive(:chip_aspect, chip_aspect)),
        spacing,
        rx, ry, Int(min_search_radius),
        Int(_check_positive(:coarse_stride, coarse_stride)),
        Int(coarse_buffer),
        _check_fraction(:min_coarse_valid_fraction, min_coarse_valid_fraction),
        Float64(dx_prior), Float64(dy_prior),
        _check_odd_window(:outlier_window, outlier_window),
        Int(_check_positive(:outlier_iterations, outlier_iterations)),
        _check_fraction(:min_agree_fraction, min_agree_fraction),
        _check_fraction(:agree_tolerance, agree_tolerance),
        Float64(_check_positive(:mad_scale, mad_scale)),
        _check_odd_window(:fill_window, fill_window),
        UInt64(rng_seed),
        Bool(progress),
    )
end
