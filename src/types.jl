# Method hierarchies, parameter container, and the sentinel/bool-type helpers.
#
# Design contract for this file:
#   * Every user-selectable choice is a singleton or `@kwdef` struct under a
#     documented, exported abstract supertype.
#   * Method-intrinsic tuning parameters live *inside* the method object and are
#     validated and precomputed in its inner constructor, so the hot path never
#     re-checks them. At tens of millions of image pairs this matters.
#   * Shared/geometric parameters (chip size, grid spacing, search radius) are
#     function keywords, not method fields, because several methods read them.

# ---------------------------------------------------------------------------
# Sentinels
# ---------------------------------------------------------------------------

"""
    NoKW

Sentinel for "the user did not pass this keyword", distinct from `nothing`,
which several keywords accept as a meaningful value (e.g. `dt = nothing` means
"report displacement in pixels, not velocity"). Internal; never returned to
users.
"""
struct NoKW end

const nokw = NoKW()

@inline isnokw(::NoKW) = true
@inline isnokw(_) = false
@inline isnokwornothing(::Union{NoKW,Nothing}) = true
@inline isnokwornothing(_) = false

# ---------------------------------------------------------------------------
# Bools as types
# ---------------------------------------------------------------------------

# `threaded` is converted from a `Bool` to a type at the API boundary so the
# threaded and serial code paths compile separately and the unused one is
# eliminated. The abstract parent and the `::BoolAsType` return annotation on
# `booltype` are both load-bearing for inference: without them the compiler
# cannot tell what `booltype` returns and the whole call chain goes unstable.
#
# Static.jl would do this too, but it redefines `<`, `>` and `==` to return
# non-`Bool`s, which causes widespread invalidations. Not worth it for two
# singletons.

abstract type BoolAsType end
struct True <: BoolAsType end
struct False <: BoolAsType end

@inline booltype(x::Bool)::BoolAsType = x ? True() : False()
@inline booltype(x::BoolAsType)::BoolAsType = x
@inline istrue(::True) = true
@inline istrue(::False) = false

# ---------------------------------------------------------------------------
# Similarity measures
# ---------------------------------------------------------------------------

"""
    SimilarityMeasure

Abstract supertype for the metric used to compare a chip against a search
window. Concrete subtypes: [`ZNCC`](@ref), [`NCC`](@ref), [`Coherence`](@ref).

A similarity measure determines how the correlation surface is formed. Given a
chip `T` and the window of the search image `I` at each candidate shift, each
measure defines a numerator and a normalising denominator.
"""
abstract type SimilarityMeasure end

# ---------------------------------------------------------------------------
# Why not SSD and SAD
# ---------------------------------------------------------------------------
#
# Sum-of-squared and sum-of-absolute differences are the textbook cheap alternatives, and the DIC
# literature uses SSD as standard. Neither is offered, for two measured reasons and one physical
# one.
#
# **The normalisation they skip is 15% of the cost.** ZNCC's integral images and per-shift divide
# cost 15% over `NCC` at chips 32 and 64 (44.4 vs 38.7 us, 75.5 vs 65.9 us). That is the entire
# ceiling for any measure whose advantage is avoiding normalisation.
#
# **SAD cannot use the FFT, and that costs an order of magnitude.** SSD expands to
# `sum(T^2) - 2*sum(T*W) + sum(W^2)`, whose cross term is the same correlation ZNCC already
# computes — so SSD is FFT-able and would cost about what ZNCC costs. `sum(abs(T - W))` has no such
# decomposition and must be evaluated directly at every shift: 2.56M multiply-adds per point at
# chip 32, against the 20,000 the direct path is worth using below. Measured with a tight SIMD
# kernel: 387 us at chip 32 and 1187 us at chip 64, so **8.7x and 15.7x slower than ZNCC**.
#
# **And they answer the wrong question.** Two acquisitions differ in sun angle, atmosphere, and
# sensor gain; SSD and SAD respond to that difference directly, where ZNCC's mean-and-scale removal
# does not. That is why the reference unified on ZNCC in v2.0.0 — earlier versions used `NCC` for
# floating-point input, which its own comments describe as a bug.
#
# SSD would be a small `_correlate_surface!` method if a cross-check against a DIC package were
# ever wanted, since the FFT machinery is already there. SAD is not worth having.

"""
    ZNCC()

Zero-mean normalized cross-correlation, also called the *normalized correlation
coefficient* or (in digital image correlation) the Pearson local correlation
coefficient. Equivalent to OpenCV's `TM_CCOEFF_NORMED`.

```math
R = \\frac{\\sum (T - \\bar{T})(I - \\bar{I})}{\\sqrt{\\sum (T - \\bar{T})^2 \\sum (I - \\bar{I})^2}}
```

Invariant to an additive brightness offset in either image, which makes it the
right choice for optical imagery with varying illumination.

The default, and the measure the reference implementation uses for every input
type. (Earlier autoRIFT releases used [`NCC`](@ref) for floating-point input,
which was a bug; v2.0.0 unified on `ZNCC`.)
"""
struct ZNCC <: SimilarityMeasure end

"""
    NCC()

Normalized cross-correlation without mean removal. Equivalent to OpenCV's
`TM_CCORR_NORMED`.

```math
R = \\frac{\\sum T \\cdot I}{\\sqrt{\\sum T^2 \\sum I^2}}
```

!!! warning
    `NCC` is **not** invariant to an additive offset, so a brightness difference
    between the two acquisitions biases the peak. Prefer [`ZNCC`](@ref) unless
    the mean itself carries signal you want to keep. Provided for comparison and
    for reproducing autoRIFT releases before v2.0.0, which used this measure for
    floating-point input.
"""
struct NCC <: SimilarityMeasure end

"""
    Coherence()

Complex coherence magnitude, for single-look-complex (SLC) radar input.

```math
\\gamma = \\frac{\\left| \\sum \\overline{T} \\cdot I \\right|}{\\sqrt{\\sum |T|^2 \\sum |I|^2}}
```

The correlation surface is real-valued (the magnitude), so peak location and
subpixel refinement are unchanged; the phase at the peak is available as an
extra output.
"""
struct Coherence <: SimilarityMeasure end

# ---------------------------------------------------------------------------
# Subpixel refinement
# ---------------------------------------------------------------------------

"""
    SubpixelMethod

Abstract supertype for estimating the correlation peak to sub-pixel precision.
Concrete subtypes: [`PyramidRefine`](@ref), [`NoRefine`](@ref).
"""
abstract type SubpixelMethod end

"""
    NoRefine()

Report the integer-pixel peak with no refinement. Used for the coarse pass,
where only a rough displacement is needed to restrict the fine search.
"""
struct NoRefine <: SubpixelMethod end

"""
    PyramidRefine(; upsampling = 64)

Locate the peak to `1/upsampling` pixel by upsampling a 5x5 neighbourhood of
the correlation surface with a Gaussian pyramid and taking the argmax of the
result. This is the autoRIFT reference approach, and it is deliberately robust
to the "peak locking" bias that a parabola fit suffers on quantized imagery.

`upsampling` must be a power of two greater than one; the reference divides by
`upsampling` after upsampling to the next power of two, so a non-power-of-two
silently produces a wrongly scaled displacement.
"""
struct PyramidRefine <: SubpixelMethod
    upsampling::Int

    function PyramidRefine(upsampling::Integer)
        upsampling > 1 ||
            throw(ArgumentError("`upsampling` must be > 1, got $upsampling"))
        ispow2(upsampling) ||
            throw(ArgumentError("`upsampling` must be a power of 2, got $upsampling"))
        return new(Int(upsampling))
    end
end

PyramidRefine(; upsampling = 64) = PyramidRefine(upsampling)

"""
    upsampling(method::SubpixelMethod)

The subpixel quantization denominator: the peak is located to `1/upsampling` of
a pixel. `1` for [`NoRefine`](@ref).
"""
upsampling(::NoRefine) = 1
upsampling(m::PyramidRefine) = m.upsampling

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

"""
    PreprocessMethod

Abstract supertype for the filter applied to both images before correlation.
Preprocessing suppresses the low-frequency radiometric differences between two
acquisitions so that correlation responds to texture rather than to brightness.

Concrete subtypes: [`Highpass`](@ref), [`Wallis`](@ref), [`WallisGapfill`](@ref),
[`Sobel`](@ref), [`Laplacian`](@ref), [`Decibel`](@ref), [`NoPreprocess`](@ref).
"""
abstract type PreprocessMethod end

"""
    NoPreprocess()

Pass both images through unchanged.
"""
struct NoPreprocess <: PreprocessMethod end

"""
    Highpass(; width = 5)

Subtract a local box mean, i.e. convolve with an identity-minus-box kernel of
side `width`. The cheapest way to remove the illumination gradient between two
scenes, and the default in the autoRIFT reference driver.
"""
struct Highpass <: PreprocessMethod
    width::Int
    Highpass(width::Integer) = new(_check_filter_width(width, :Highpass))
end
Highpass(; width = 5) = Highpass(width)

"""
    Wallis(; width = 5, min_std = 0.0)

Wallis (1976) adaptive contrast filter: subtract the local mean and divide by
the local standard deviation over a `width`-by-`width` window, so that local
contrast is equalized across the scene.

Stronger than [`Highpass`](@ref) where illumination varies sharply, at the cost
of amplifying noise in low-texture areas. `min_std` clamps the divisor to avoid
that amplification; `0.0` disables the clamp.

The reference uses `width = 21` for Sentinel-1 and `5` otherwise.
"""
struct Wallis <: PreprocessMethod
    width::Int
    min_std::Float64

    function Wallis(width::Integer, min_std::Real)
        min_std >= 0 ||
            throw(ArgumentError("`min_std` must be >= 0, got $min_std"))
        return new(_check_filter_width(width, :Wallis), Float64(min_std))
    end
end
Wallis(; width = 5, min_std = 0.0) = Wallis(width, min_std)

"""
    WallisGapfill(; width = 5, min_std = 0.25)

[`Wallis`](@ref) plus synthetic infill of interior no-data gaps, for Landsat-7
imagery acquired after the 2003-05-31 Scan Line Corrector failure. Gap pixels
are filled with noise matched to the local statistics so that they neither
correlate spuriously nor mask out their neighbourhood.

Filling uses the random number generator seeded by the `rng_seed` keyword, so
results are reproducible. (The Python reference draws from NumPy's unseeded
global generator and is therefore not reproducible against itself.)
"""
struct WallisGapfill <: PreprocessMethod
    width::Int
    min_std::Float64

    function WallisGapfill(width::Integer, min_std::Real)
        min_std >= 0 ||
            throw(ArgumentError("`min_std` must be >= 0, got $min_std"))
        return new(_check_filter_width(width, :WallisGapfill), Float64(min_std))
    end
end
WallisGapfill(; width = 5, min_std = 0.25) = WallisGapfill(width, min_std)

"""
    Sobel(; width = 5)

Sum of the Sobel derivative kernels in x and y, giving an edge-emphasising
filter. `width` selects the derivative kernel size.
"""
struct Sobel <: PreprocessMethod
    width::Int
    Sobel(width::Integer) = new(_check_filter_width(width, :Sobel))
end
Sobel(; width = 5) = Sobel(width)

"""
    Laplacian(; width = 5)

Laplacian (isotropic second derivative) of the log-amplitude image. Intended
for radar amplitude.
"""
struct Laplacian <: PreprocessMethod
    width::Int
    Laplacian(width::Integer) = new(_check_filter_width(width, :Laplacian))
end
Laplacian(; width = 5) = Laplacian(width)

"""
    Decibel()

Convert amplitude to decibels, `20 log10(A)`. Intended for radar amplitude,
where the dynamic range is large and multiplicative.
"""
struct Decibel <: PreprocessMethod end

"""
    filter_width(method::PreprocessMethod)

Side length of the filter window, or `0` for methods that take no window.
"""
filter_width(::Union{NoPreprocess,Decibel}) = 0
filter_width(m::Union{Highpass,Wallis,WallisGapfill,Sobel,Laplacian}) = m.width

function _check_filter_width(width::Integer, who::Symbol)
    width >= 3 ||
        throw(ArgumentError("$who `width` must be >= 3, got $width"))
    isodd(width) ||
        throw(ArgumentError("$who `width` must be odd so the filter has a " *
                            "well-defined centre, got $width"))
    return Int(width)
end

# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

"""
    QuantizeMethod

Abstract supertype for how filtered images are converted to the element type
used for correlation. Concrete subtypes: [`QuantizeUInt8`](@ref),
[`NoQuantize`](@ref).
"""
abstract type QuantizeMethod end

"""
    QuantizeUInt8()

Rescale each image so that `[mean - 3 std, mean + 3 std]` maps onto the `UInt8`
range, then round and clamp. Correlating in `UInt8` allows exact integer
accumulation of the correlation numerator, which is both faster and more
accurate than accumulating in `Float32`.
"""
struct QuantizeUInt8 <: QuantizeMethod end

"""
    NoQuantize()

Keep the filtered images in floating point, replacing non-finite values with
zero. Preserves full radiometric precision at the cost of a slower correlation
inner loop.
"""
struct NoQuantize <: QuantizeMethod end

# ---------------------------------------------------------------------------
# Outlier rejection
# ---------------------------------------------------------------------------

"""
    OutlierMethod

Abstract supertype for how implausible displacements are identified and dropped.

Correlation returns a displacement at every searched point, including points where the peak
was noise. Those false matches are not small errors — they are arbitrary vectors, and one can
dominate any downstream fit. What separates them from real motion is spatial coherence:
neighbouring points on the same glacier move similarly, while a false match agrees with
nothing around it. Every method here is some way of asking that question.

Concrete subtypes: [`GardnerFilter`](@ref), [`NoOutlierFilter`](@ref).

Selected with the `outliers` keyword, which accepts either the `Symbol` naming a method or an
instance — the latter being the only way to override a method's own parameters.
"""
abstract type OutlierMethod end

"""
    GardnerFilter(; window = 5, iterations = 3, min_agree_fraction = 8/25,
                    agree_tolerance = 0.2, mad_scale = 4.0)

The two-stage filter of Gardner et al. (2018), as implemented in autoRIFT's `DISP_FILT`.

Stage 1 keeps a point only if `min_agree_fraction` of its neighbourhood lies within
`agree_tolerance` of it — cheap, and removes isolated wild vectors. Stage 2 keeps a point only
if it lies within `mad_scale` median absolute deviations of its neighbourhood median, which
catches the subtler case of a false match sitting near other false matches. Both run on
displacements normalized by the local search radius, and both iterate: rejection is monotone,
so removing one outlier can expose its neighbours.

# Keywords
- `window`: neighbourhood width in grid points. Must be odd and at least 3, so the
  neighbourhood has a centre.
- `iterations`: iterations of stage 1. Stage 2 runs one fewer, minimum one, per the reference.
- `min_agree_fraction`: fraction of the *full* window area that must agree. The default `8/25`
  is eight of a 5x5 window's twenty-five points.
- `agree_tolerance`: how close counts as agreement, as a fraction of the local search radius.
- `mad_scale`: how many median absolute deviations from the neighbourhood median a point may
  lie before stage 2 rejects it.

!!! note "Relation to the normalized median test"
    This is *inspired by* the universal outlier detection of Westerweel & Scarano (2005) but is
    not that test, and the differences change which points survive. The centre point is
    included in both the median and the MAD, where Westerweel & Scarano exclude it — excluding
    it is what makes their test ask whether a vector agrees with its neighbours rather than
    with a set it partly defines. The normalization is by local search radius rather than by
    `MAD + ε` with an absolute `ε`; the threshold is 4 rather than their universal 2; and the
    agreement pre-pass has no counterpart there at all.

    See [`AutoRIFT.reject_outliers`](@ref). Westerweel & Scarano's test proper is a candidate
    for a future `OutlierMethod`.
"""
struct GardnerFilter <: OutlierMethod
    window::Int
    iterations::Int
    min_agree_fraction::Float64
    agree_tolerance::Float64
    mad_scale::Float64

    function GardnerFilter(; window::Integer = 5, iterations::Integer = 3,
                           min_agree_fraction::Real = 8 / 25,
                           agree_tolerance::Real = 0.2, mad_scale::Real = 4.0)
        isodd(window) || throw(ArgumentError(
            "outlier filter `window` must be odd so the neighbourhood has a centre, " *
            "got $window"))
        window >= 3 || throw(ArgumentError(
            "outlier filter `window` must be >= 3, got $window"))
        iterations >= 1 || throw(ArgumentError(
            "outlier filter `iterations` must be >= 1, got $iterations"))
        0 <= min_agree_fraction <= 1 || throw(ArgumentError(
            "`min_agree_fraction` must be in [0, 1], got $min_agree_fraction"))
        # A fraction of the local search radius, so a value above 1 would accept agreement
        # looser than the entire search window — which is every point, making the stage a
        # no-op rather than a filter.
        0 < agree_tolerance <= 1 || throw(ArgumentError(
            "`agree_tolerance` must be in [0, 1] and non-zero, got $agree_tolerance. It " *
            "is a fraction of the local search radius, so 1 already accepts any " *
            "displacement the search could have found."))
        mad_scale > 0 || throw(ArgumentError(
            "`mad_scale` must be positive, got $mad_scale"))
        return new(Int(window), Int(iterations), Float64(min_agree_fraction),
                   Float64(agree_tolerance), Float64(mad_scale))
    end
end

"""
    NoOutlierFilter()

Keep every measured displacement.

Not a sensible production choice — a single false match can dominate a downstream fit — but
the right tool for deciding *which* stage dropped a point, since it separates the correlator's
failures from the filter's rejections.
"""
struct NoOutlierFilter <: OutlierMethod end

"""
    AutoRIFT.window(method::OutlierMethod) -> Int

Neighbourhood width the method needs, in grid points, or `0` if it needs none.

Used to decide whether a decimated grid has enough points to judge consistency at all: a
coarse grid narrower than the window cannot support the filter, and the caller skips it rather
than filtering against a truncated neighbourhood.
"""
window(f::GardnerFilter) = f.window
window(::NoOutlierFilter) = 0

"""
    AutoRIFT.relax(method::OutlierMethod) -> OutlierMethod

A variant of `method` suited to a decimated grid.

The coarse pass runs on a strided subset, so its neighbourhoods span several times more ground
and a point's neighbours are genuinely further away and less like it. Holding them to the fine
pass's standard would reject coherent motion. What "loosened" means is the method's own
business, which is why this returns an instance rather than taking a scale factor — a method
with a universal threshold may legitimately return itself.

For [`GardnerFilter`](@ref) it drops one iteration; the reference derives the same effect from
an overlap fraction.
"""
relax(f::GardnerFilter) = GardnerFilter(;
    window = f.window, iterations = max(f.iterations - 1, 1),
    min_agree_fraction = f.min_agree_fraction,
    agree_tolerance = f.agree_tolerance, mad_scale = f.mad_scale)
relax(m::NoOutlierFilter) = m

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

"""
    Params

Fully resolved, immutable, concretely-typed parameter set. Constructed once by
[`AutoRIFT.params`](@ref) from user keywords; every `Symbol` has been mapped to
a method object and every number validated by the time a `Params` exists, so
nothing downstream ever re-checks or re-dispatches on a `Symbol`.

The method choices are type parameters, not fields, so the correlation and
filtering kernels specialize on them.
"""
struct Params{S<:SimilarityMeasure,P<:PreprocessMethod,Q<:QuantizeMethod,R<:SubpixelMethod,
              O<:OutlierMethod,T<:BoolAsType}
    similarity::S
    preprocess::P
    quantize::Q
    subpixel::R
    outliers::O
    threaded::T

    # Chip geometry (pixels). `chip_size_base` is the finest chip-size level and
    # the reference for grid spacing; levels are base * 2^k.
    chip_size_base::Int
    chip_size_min::Int
    chip_size_max::Int
    chip_aspect::Float64

    # Grid geometry (pixels).
    grid_spacing::Int

    # Search geometry (pixels). Radius, not width: the search window spans
    # 2*radius. Naming this "limit" as the reference does is what produced its
    # off-by-one errors.
    #
    # x and y are independent, not a convenience split of one number. Geogrid
    # projects the a-priori velocity onto each image axis separately, so an ice
    # stream flowing along x legitimately gets a wide x-radius and a narrow
    # y-radius. These scalars are the fallback when no per-pixel radius field is
    # supplied; the per-pixel case carries two separate arrays in lockstep.
    search_radius_x::Int
    search_radius_y::Int
    min_search_radius::Int
    coarse_stride::Int
    coarse_buffer::Int
    min_coarse_valid_fraction::Float64

    # A-priori displacement (pixels), the predictor in a predictor-corrector
    # sense: the search window is centred on it rather than on zero.
    dx_prior::Float64
    dy_prior::Float64

    # Hole filling.
    fill_window::Int

    # Misc.
    rng_seed::UInt64
    progress::Bool
end

"""
    chip_sizes(p::Params) -> Vector{Int}

The chip-size levels actually used, ascending. Levels are
`chip_size_base * 2^k` for `k = 0, 1, ...`, restricted to
`[chip_size_min, chip_size_max]`.

Ascending order is load-bearing: every level only writes where no finer level
already produced a value, so the *smallest* chip that yields a valid estimate
wins. Small chips resolve detail; large chips succeed in low-texture areas.
"""
function chip_sizes(p::Params)
    sizes = Int[]
    cs = p.chip_size_base
    while cs <= p.chip_size_max
        cs >= p.chip_size_min && push!(sizes, cs)
        cs *= 2
    end
    return sizes
end

"""
    chip_size_y(p::Params, chip_size_x::Integer) -> Int

Chip height for a given width, forced even so that the chip has a well-defined
half-extent in both directions.
"""
chip_size_y(p::Params, chip_size_x::Integer) =
    2 * round(Int, chip_size_x * p.chip_aspect / 2)

# Search-radius normalisation lives with the point set it operates on; see
# `sanitize!` in points.jl.
