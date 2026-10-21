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

"""
    AutoRIFT.ImageElement

The element types the correlator accepts: `Real` for optical and amplitude imagery, `Complex` for
single-look-complex radar under [`Coherence`](@ref). Equal to `Union{Real,Complex}`.
"""
const ImageElement = Union{Real,Complex}

"""
    AutoRIFT.Extent

A quantity with an x and a y component, in pixels: `(X = 16, Y = 32)`. Chip size, search radius and
grid spacing are extents.

The type is `NamedTuple{(:X, :Y),Tuple{Int,Int}}`. It is `isbitstype`, so `Params` is too, and it has
the layout of `Tuple{Int,Int}` (two `Int64`s, no padding), which maps to `struct { int64_t x, y; }`
for a C caller. Build one from a scalar or a tuple with [`AutoRIFT.extent`](@ref); a scalar means
square.
"""
const Extent = NamedTuple{(:X, :Y),Tuple{Int,Int}}

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

Abstract supertype for the measure that compares a chip against a search window. Concrete subtypes:
[`ZNCC`](@ref), [`NCC`](@ref), [`Coherence`](@ref).

Given a chip `T` and the window of the search image `I` at each candidate shift, a measure defines a
numerator and a normalizing denominator, which together form the correlation surface.
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

Zero-mean normalized cross-correlation, also called the normalized correlation coefficient or, in
digital image correlation, the Pearson local correlation coefficient. Equivalent to OpenCV's
`TM_CCOEFF_NORMED`.

```math
R = \\frac{\\sum (T - \\bar{T})(I - \\bar{I})}{\\sqrt{\\sum (T - \\bar{T})^2 \\sum (I - \\bar{I})^2}}
```

Invariant to an additive brightness offset in either image, so it suits optical imagery with varying
illumination. The default, and the measure autoRIFT v2.0.0 and later use for every input type.
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
    `NCC` is not invariant to an additive offset, so a brightness difference between the two
    acquisitions biases the peak. Prefer [`ZNCC`](@ref) unless the mean carries signal to keep. It
    reproduces autoRIFT releases before v2.0.0, which used it for floating-point input.
"""
struct NCC <: SimilarityMeasure end

"""
    Coherence()

Complex coherence magnitude, for single-look-complex (SLC) radar input.

```math
\\gamma = \\frac{\\left| \\sum \\overline{T'} \\cdot W' \\right|}
               {\\sqrt{\\sum |T'|^2 \\sum |W'|^2}}
```

where ``T'`` and ``W'`` are the chip and the window with their complex means removed. This is the
complex analogue of [`ZNCC`](@ref) and gives ``\\gamma(T, T) = 1`` exactly. The surface is the real
magnitude, so peak location and sub-pixel refinement use the same code as the real measures.

Requires complex input; use [`ZNCC`](@ref) for a real image.

Coherence resolves finer detail than amplitude (Joughin, 2002) but fails where amplitude does not.

!!! warning "Phase variation destroys the peak"
    Interferometric phase across the chip can reduce or eliminate the correlation peak, most where
    the deformation is largest. Amplitude matching is unaffected there.

    Use [`Deramp`](@ref) as the preprocessing step and give `similarity` a tuple, which runs coherence
    at the finest chip size and a real measure above it, so a point coherence cannot resolve is left
    to a larger amplitude chip.

    ```julia
    autorift(z1, z2; similarity = (:coherence, :zncc), preprocess = :deramp,
             chip_size = 32, chip_size_max = 128)
    ```

See [How feature tracking works](@ref) for what the escalation gains.
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

Report the integer-pixel peak with no refinement. The coarse pass uses it, since it needs only a rough
displacement to restrict the fine search.
"""
struct NoRefine <: SubpixelMethod end

"""
    PyramidRefine(; upsampling = 64)

Locate the peak to `1/upsampling` pixel by upsampling the 5x5 neighborhood of the integer peak with a
Gaussian pyramid and taking the argmax. This is the autoRIFT reference approach. Unlike a parabola
fit, it is not subject to peak-locking bias on quantized imagery.

`upsampling` must be a power of two greater than one; otherwise `ArgumentError` is thrown.
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

The sub-pixel denominator: the peak is located to `1/upsampling` of a pixel. `1` for
[`NoRefine`](@ref).
"""
upsampling(::NoRefine) = 1
upsampling(m::PyramidRefine) = m.upsampling

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

"""
    PreprocessMethod

Abstract supertype for the filter applied to both images before correlation. Preprocessing suppresses
low-frequency radiometric differences between acquisitions so that correlation responds to texture
rather than brightness.

Concrete subtypes: [`Highpass`](@ref), [`Wallis`](@ref), [`WallisGapfill`](@ref), [`Sobel`](@ref),
[`Laplacian`](@ref), [`Decibel`](@ref), [`Destripe`](@ref), [`Deramp`](@ref), [`NoPreprocess`](@ref).
"""
abstract type PreprocessMethod end

"""
    NoPreprocess()

Pass both images through unchanged.
"""
struct NoPreprocess <: PreprocessMethod end

"""
    Highpass(; width = 5)

Subtract a local box mean, equivalent to convolving with an identity-minus-box kernel of side `width`.
It is the cheapest way to remove the illumination gradient between two scenes and is the default in
the autoRIFT reference driver. `width` must be odd and at least 3.
"""
struct Highpass <: PreprocessMethod
    width::Int
    Highpass(width::Integer) = new(_check_filter_width(width, :Highpass))
end
Highpass(; width = 5) = Highpass(width)

"""
    Wallis(; width = 5, min_std = 0.0)

Wallis (1976) adaptive contrast filter: subtract the local mean and divide by the local standard
deviation over a `width`-by-`width` window, equalizing local contrast across the scene. `width` must be
odd and at least 3.

It handles sharply varying illumination better than [`Highpass`](@ref) but amplifies noise in
low-texture areas. `min_std` sets a floor on the divisor to limit that; `0.0` applies no floor.

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

[`Wallis`](@ref) plus synthetic infill of interior no-data gaps, for Landsat-7 imagery acquired after
the 2003-05-31 Scan Line Corrector failure. Gap pixels are filled with noise matched to the local
statistics, so they neither correlate spuriously nor mask out their neighbors.

The generator is seeded by `params`' `rng_seed`, so results are reproducible. The Python reference uses
an unseeded generator and does not reproduce itself.
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
    AutoRIFT.GAPFILL_REACH

How far real data may lie from an invalid pixel for that pixel to count as an interior gap rather
than as outer no-data border, in pixels. From the reference.

An interior gap is filled with noise; a border is left invalid. The distinction matters because
noise is only defensible where there is nearby data for it to be statistically consistent with.
"""
const GAPFILL_REACH = 30.0

# Half-diagonal of a `w`-by-`w` window: the distance at which a pixel's window can still reach a
# gap, and so how far a gap's influence is grown before the normalized output around it is trusted.
_gapfill_buffer(w::Integer) = sqrt(2 * ((Int(w) - 1) / 2)^2) + 0.01

"""
    Sobel(; width = 5)

Sum of the Sobel derivative kernels in x and y, an edge-emphasizing filter. `width` is the derivative
kernel size, odd and at least 3.
"""
struct Sobel <: PreprocessMethod
    width::Int
    Sobel(width::Integer) = new(_check_filter_width(width, :Sobel))
end
Sobel(; width = 5) = Sobel(width)

"""
    Destripe(; along_track, cross_track, band_half = 70, notch_half = 100,
             clamp = 3, power_threshold = 500, ratio = 2, sigma = 10)

Reject the stronger of two band-limited noise directions in the frequency domain, for Landsat 4 and 5,
whose Multispectral Scanner leaves coherent banding along one of two scan directions. The stripes occupy
a thin line through the origin of the Fourier transform, which can be removed without touching the
texture that correlation needs.

`along_track` and `cross_track` are the scan directions in degrees on the raster's own grid. They are
required keywords. [`track_angles`](@ref) computes them from a ground-track displacement.

Only one direction is filtered, and only sometimes. The two band powers are compared, and the reject
applies only if one exceeds the other by `ratio` and the larger exceeds `power_threshold`. Otherwise the
image is returned clamped to `±clamp` but not filtered.

`band_half` and `notch_half` are in pixels and are not scaled to the image. The defaults are the
reference's (`autoRIFT.py:153-243`). Works on the whole image, so it has no finite
[`filter_reach`](@ref) and cannot run in a blocked run.
"""
struct Destripe <: PreprocessMethod
    along_track::Float64
    cross_track::Float64
    band_half::Int
    notch_half::Int
    clamp::Float64
    power_threshold::Float64
    ratio::Float64
    sigma::Float64

    function Destripe(along_track::Real, cross_track::Real, band_half::Integer,
                      notch_half::Integer, clamp::Real, power_threshold::Real,
                      ratio::Real, sigma::Real)
        isfinite(along_track) && isfinite(cross_track) ||
            throw(ArgumentError("scan angles must be finite, got " *
                                "along_track=$along_track, cross_track=$cross_track"))
        band_half > 0 ||
            throw(ArgumentError("`band_half` must be positive, got $band_half"))
        notch_half >= 0 ||
            throw(ArgumentError("`notch_half` must be >= 0, got $notch_half"))
        clamp > 0 || throw(ArgumentError("`clamp` must be positive, got $clamp"))
        ratio >= 1 || throw(ArgumentError("`ratio` must be >= 1, got $ratio"))
        return new(Float64(along_track), Float64(cross_track), Int(band_half), Int(notch_half),
                   Float64(clamp), Float64(power_threshold), Float64(ratio), Float64(sigma))
    end
end

Destripe(; along_track, cross_track, band_half = 70, notch_half = 100, clamp = 3,
         power_threshold = 500, ratio = 2, sigma = 10) =
    Destripe(along_track, cross_track, band_half, notch_half, clamp, power_threshold, ratio, sigma)

"""
    track_angles(dx, dy; spacing) -> (along_track, cross_track)

[`Destripe`](@ref)'s scan angles, in degrees, from a ground-track displacement `(dx, dy)` in a raster's
projected CRS and that raster's pixel `spacing = (sx, sy)`. Any source of a ground-track vector can
supply `(dx, dy)`, such as an orbit ephemeris projected into the raster's CRS.

The angles are computed in pixel units: `along_track = atan((dy / sy) / (dx / sx))`, the arctangent of a
ratio rather than a two-argument `atan`, matching the reference's `_calculate_slope`
(`autoRIFT.py:138-151`). The result lies in `(-90, 90)`, and a vertical displacement gives `±90`.
`cross_track` is the perpendicular direction, folded into the same interval. `spacing` is required
because a non-square pixel changes the ratio.
"""
function track_angles(dx::Real, dy::Real; spacing::Tuple{Real,Real})
    sx, sy = abs(Float64(spacing[1])), abs(Float64(spacing[2]))
    along = rad2deg(atan(Float64(dy) / sy / (Float64(dx) / sx)))
    cross = along + (along > 0 ? -90.0 : 90.0)
    return (along, cross)
end

"""
    Laplacian(; width = 5)

Laplacian (isotropic second derivative) of the log-amplitude image, for radar amplitude. `width` is
the kernel side, odd and at least 3.
"""
struct Laplacian <: PreprocessMethod
    width::Int
    Laplacian(width::Integer) = new(_check_filter_width(width, :Laplacian))
end
Laplacian(; width = 5) = Laplacian(width)

"""
    Decibel()

Convert amplitude to decibels, `20 log10(A)`, for radar amplitude, whose brightness varies
multiplicatively. Pixels with non-positive amplitude are marked invalid.
"""
struct Decibel <: PreprocessMethod end

"""
    Deramp(; axis = :both)

Remove the linear phase ramp from a complex image. Complex input only; a real image throws
`ArgumentError` at the `autorift` call.

A phase ramp is a frequency shift. A signal whose band is off-center aliases when oversampled, and
[`PyramidRefine`](@ref) oversamples the correlation surface, so complex correlation needs the ramp
removed (ISCE2's `cuAmpcor` makes the same point in `cuDeramp.cu`). The linear component is also the
dominant part of the phase variation that suppresses the [`Coherence`](@ref) peak.

The estimate follows `cuAmpcor`'s method 1. For each axis, sum the conjugate product of adjacent valid
pixels and take the argument of that sum:

```math
\\phi_x = \\arg \\sum_{i} \\overline{z_i} \\, z_{i + \\Delta x}
```

Summing the products weights each pixel pair by its amplitude, and no `atan` is taken until after the
sum, so no unwrapping is needed.

`axis` selects `:x`, `:y`, or `:both`.

!!! note "Not a substitute for amplitude fallback"
    A linear ramp is only part of the phase variation. Steep topography and high shear produce
    higher-order variation that this cannot remove; give `similarity` a tuple that falls back to
    amplitude (see [`Coherence`](@ref)).
"""
struct Deramp <: PreprocessMethod
    axis::Symbol
    function Deramp(axis::Symbol)
        axis in (:x, :y, :both) || throw(ArgumentError(
            "`Deramp` axis must be `:x`, `:y`, or `:both`, got `:$axis`"))
        return new(axis)
    end
end
Deramp(; axis = :both) = Deramp(axis)

"""
    RotationMethod

Whether the dense stage correlates each chip at several rotations and keeps the best.

Subtypes: [`RotationSearch`](@ref), [`NoRotationSearch`](@ref). Pass one as `params`' `rotation`.
"""
abstract type RotationMethod end

"""
    NoRotationSearch()

Correlate each chip at its own orientation only. The default `rotation`.
"""
struct NoRotationSearch <: RotationMethod end

# The angles are a *tuple* type parameter, and the reason is `isbits` rather than speed.
#
# The cost is real and was measured: each distinct angle *count* is a distinct `RotationSearch`
# type, hence a distinct `Params`, hence a full recompile of the correlation pipeline — ~600 ms
# per new count (785 / 607 / 520 ms for 3, 5 and 7 angles cold), where new angle *values* at a
# count already seen are free at 7.6 ms. A caller sweeping counts pays that per count, once per
# session.
#
# And the parameterisation buys **nothing** at runtime: against a byte-identical copy of
# `_correlate_rotations!` iterating a `Vector{Float64}` of the same angles, the ratio is
# 1.000 / 1.001 / 1.000 at chip 32 / 64 / 128. Of course it is — the loop body is one
# `_rotate_chip` plus one `correlate!` at 49-550 µs, so unrolling three iterations is unmeasurable.
#
# What a `Vector` field would cost is the deciding measurement: it is `isconcretetype` but **not**
# `isbitstype`, and `Params` is currently `isbitstype` — which is what lets it store inline with no
# dispatch and is load-bearing for the `--trim`ed binary in `app/`. Trading that for a one-time
# ~600 ms of recompile is the wrong direction, so the tuple stays. `test/params.jl` asserts
# `isbitstype` and not merely `isconcretetype` for exactly this reason.
#
# A comment between a docstring and the definition it documents discards the docstring silently, so
# this one stays above it.
"""
    RotationSearch(angles = (-3.0, 0.0, 3.0); about = 0.0)

Correlate each chip at every angle in `angles`, in degrees, and keep the strongest peak. `about` is
the scene's own rotation, in the sense [`scene_rotation`](@ref) reports it, so the rotations applied to
the chip are `angles .- about`. `angles` is any real vector, range or tuple of distinct, finite values;
`about` must be finite. An empty `angles`, a repeated angle or a non-finite value throws
`ArgumentError`.

Correlation cost is multiplied by up to `length(angles)`. The search helps when the scene itself
rotates, since a rotated chip decorrelates against an unrotated window.

```julia
autorift(a, b, guess; rotation = RotationSearch())              # ±3°, 3x cost
autorift(a, b, guess; rotation = RotationSearch((-6, -3, 0, 3, 6)))
autorift(a, b, guess; rotation = RotationSearch(; about = 8.0)) # searches 5-11°
```

Five angles cost 1.7×, not 5×, since the per-point work around the correlation is shared across them.
[`AutoRIFT.angles`](@ref) reports the rotations a pass will try. The search does not correct shear.
See [Giving the search a first guess](@ref) for what the search recovers, why `about` is subtracted,
and where the estimate comes from.
"""
struct RotationSearch{A<:Tuple} <: RotationMethod
    angles::A
    about::Float64
    function RotationSearch(angles, about::Real)
        t = Tuple(Float64.(angles))
        isempty(t) && throw(ArgumentError(
            "`RotationSearch` needs at least one angle; use `NoRotationSearch()` to disable."))
        allunique(t) || throw(ArgumentError(
            "`RotationSearch` angles must be distinct, got $t — a repeated angle costs a full " *
            "correlation and can never win."))
        # The invariant is that every angle *tried* is finite, and the angles tried are
        # `t .- about` — so both halves need checking. Enforcing it only on `about` was the first
        # version, and it let `RotationSearch((NaN, 0.0, 3.0))` construct: `_rotate_chip` then
        # produces an all-NaN chip that disappears into the degenerate path, reported as a
        # textureless chip rather than as the bad input it is.
        a = Float64(about)
        (all(isfinite, t) && isfinite(a)) || throw(ArgumentError(
            "`RotationSearch` needs finite angles and a finite `about`, got angles $t about $a."))
        return new{typeof(t)}(t, a)
    end
end
# The keyword form is the documented one; the inner constructor is positional so `Params`'s
# positional constructor -- the trimmable entry point -- can reach it without a keyword call.
RotationSearch(angles = (-3.0, 0.0, 3.0); about::Real = 0.0) = RotationSearch(angles, about)

"""
    AutoRIFT.angles(method::RotationSearch) -> Tuple

The chip rotations a pass will try, in degrees: `m.angles .- m.about`. These are the rotations applied
to the chip, so `about` is already subtracted. `m.angles` is the search window and `m.about` the scene
rotation it is centered on.

Defined only for [`RotationSearch`](@ref); there is no method for [`NoRotationSearch`](@ref).
"""
angles(m::RotationSearch) = m.angles .- m.about

"""
    filter_width(method::PreprocessMethod)

Side length of the filter window, or `0` for methods that take no window.
"""
filter_width(::Union{NoPreprocess,Decibel,Deramp}) = 0
filter_width(m::Union{Highpass,Wallis,WallisGapfill,Sobel,Laplacian}) = m.width
# A whole-image FFT, so there is no local window and a block halo has nothing to widen. The band masks are
# global by construction: a frequency-domain reject cannot be computed on a tile without changing what it
# rejects, which is why `Destripe` runs before blocking rather than inside it.
filter_width(::Destripe) = 0

"""
    AutoRIFT._poolable(method::PreprocessMethod) -> Bool

Whether `_prepare_block` pools its buffers for `method`, and so whether a [`FilterTileCache`](@ref) can
cache its output — the two are the same question, since both exist for the one filter the production
driver runs inside the correlator. Every other method still allocates per block (`_prepare_block`'s
generic fallback) and is not covered.
"""
_poolable(::PreprocessMethod) = false
_poolable(::Highpass) = true

"""
    AutoRIFT.filter_reach(method::PreprocessMethod) -> Int

How many pixels beyond a region must be supplied for the filter's output inside that region to equal
what filtering the whole image gives. Blocked processing sizes its halo from this.

A negative value means no finite reach: check the sign before using the result as a width. A filter
that estimates from the whole image, [`Deramp`](@ref) or [`Destripe`](@ref), returns `-1`.
[`AutoRIFT.halo`](@ref) throws for such a method.

The value is an upper bound on the reach, which may exceed the true reach:

- `filter_width ÷ 2` for a single convolution ([`Highpass`](@ref), [`Sobel`](@ref),
  [`Laplacian`](@ref)); `0` for [`NoPreprocess`](@ref) and [`Decibel`](@ref).
- Twice that for [`Wallis`](@ref), because excluding a pixel whose own local mean is not finite is a
  window of a window.
- For [`WallisGapfill`](@ref), `GAPFILL_REACH` plus the dilation of its window plus the Wallis reach,
  because whether a pixel is filled depends on the distance to the nearest real data.

Differs from [`filter_width`](@ref) for this reason.
"""
filter_reach(m::PreprocessMethod) = filter_width(m) ÷ 2

# `Wallis` divides by a standard deviation whose *gap handling* is a window of a window: a pixel
# contributes to the variance only where its own local mean is finite, and that mean is itself a
# window, so which pixels a block excludes can differ from the whole image's two windows away. The
# variance terms themselves are both single windows of the raw image and reach half this, which is
# what `test/tile.jl` measures — so this is the bound, and the test asserts it as one.
filter_reach(m::Wallis) = 2 * (filter_width(m) ÷ 2)

# `WallisGapfill` reaches much further than its window, because deciding *whether* a pixel is filled
# is a distance-transform question rather than a windowed one: an invalid pixel is an interior gap
# only if real data lies within `GAPFILL_REACH`, and both that verdict and the low-contrast one are
# then grown by the window's half-diagonal. So the reach is the gap-detection distance plus that
# dilation plus the Wallis window itself.
#
# Finite, and verified so: a valid pixel 40 px away changes no local decision, while one at 20 px
# does. The measured reach is 26 px at width 5 against the 35 px this returns, so the bound holds
# with margin — which matters because a value that is too small makes a blocked run silently wrong
# rather than merely different.
filter_reach(m::WallisGapfill) =
    ceil(Int, GAPFILL_REACH + _gapfill_buffer(filter_width(m))) + 2 * (filter_width(m) ÷ 2)

# A whole-image reduction rather than a window: `deramp` sums adjacent-pixel conjugate products
# over every pixel and takes one `atan2`. No finite reach expresses that, so it is `-1` rather than
# a large number — a caller dividing work into blocks has to be told this cannot be done blockwise
# from local data, not handed a halo that would merely be less wrong.
filter_reach(::Deramp) = -1

# `Destripe` rejects bands of a **whole-image** FFT, and which bands it rejects is decided from that
# spectrum: a band survives only if its power clears `power_threshold` and its ratio to the
# surrounding spectrum clears `ratio`. A window has its own spectrum and so its own verdict, which
# makes the filter's output on a block differ from the scene's everywhere rather than at an edge.
# `filter_width` is 0 because there is no window; the reach is `-1` because no halo substitutes for
# the missing frequency resolution. Run it before blocking, as the production driver does.
filter_reach(::Destripe) = -1

function _check_filter_width(width::Integer, who::Symbol)
    width >= 3 ||
        throw(ArgumentError("$who `width` must be >= 3, got $width"))
    isodd(width) ||
        throw(ArgumentError("$who `width` must be odd so the filter has a " *
                            "well-defined centre, got $width"))
    return Int(width)
end

# ---------------------------------------------------------------------------
# Pass execution
# ---------------------------------------------------------------------------

"""
    AutoRIFT.PassRunner

How a correlation pass over a point set is executed: all at once, or a block at a time.

One chip-size level is the same sequence of steps either way — level points, coarse evidence,
the gate and dilation, the fine pass, rejection and hole filling — and only the *execution* of a
pass differs. `AutoRIFT.WholeScene` correlates an already-filtered pair in one call;
`AutoRIFT.Blocked` reads, filters and correlates one block at a time so peak memory tracks the
block rather than the scene.

Concrete subtypes carry the pair they correlate, and that is load-bearing rather than convenient:
the whole-scene runner holds a **filtered** pair, the blocked runner holds an **unfiltered** one and
filters each block from its own read window. Because the pair travels with the runner, there is no
call that can hand a runner the other kind — a filtered pair filtered again is a wrong image
everywhere, and one that still looks like imagery.

The interface is three methods: `run_pass(runner, pts, p, measure, subpixel)`, which correlates a
point set; `restrict(runner, setup, gridsize)`, which rebuilds a runner for a **differently shaped**
grid; and `_warn_coarse_fallback(runner, chip_size)`, which is where the two differ on policy.

`restrict` is a precondition of `run_pass`, not an optimization. A runner may hold state indexed by
grid shape — `AutoRIFT.Blocked` holds a partition of grid index ranges — so a point set whose shape
differs from the one the runner was built for is a bounds error, not a smaller pass. Every caller
that decimates, strides, or otherwise reshapes the grid must `restrict` first; the coarse pass is
the most visible such caller but it is not the only possible one. `AutoRIFT.WholeScene` holds no
per-shape state and so returns itself, which is why omitting the call is invisible until a blocked
run reaches it.

A runner is always passed **positionally**. A keyword annotated with an abstract type is
unresolvable under `--trim` — verified: it produces `unresolved call ... Core.kwcall` — which is the
same constraint that makes [`measure_at`](@ref)'s result positional.
"""
abstract type PassRunner end

# ---------------------------------------------------------------------------
# Compute backend
# ---------------------------------------------------------------------------

"""
    AutoRIFT.Backend

Where the correlation kernels run. Concrete subtypes: [`AutoRIFT.CPU`](@ref),
[`AutoRIFT.MetalGPU`](@ref), [`AutoRIFT.CUDAGPU`](@ref).

Selected with `params`' `backend` keyword, which takes the Symbol `:cpu`, `:metal` or `:cuda`, or an
instance. `CPU()` is the default and needs no additional package.

The GPU backends are experimental. A device pass is faster than one CPU core but slower than a threaded
CPU run on a single image pair; see [Correlating on a GPU](@ref).

Each backend is a singleton carried as a `Params` type parameter, so the choice is resolved at compile
time and `Params` stays `isbitstype`. The GPU backends are declared in the core so that
`params(; backend = :metal)` can be called before `using Metal`. Correlating with a backend whose
package is not loaded throws `ArgumentError` naming the package from [`AutoRIFT.required_package`](@ref).
"""
abstract type Backend end

"""
    AutoRIFT.CPU()

Run the correlation on the CPU, threaded or not according to `params`' `threaded`. The default.
"""
struct CPU <: Backend end

"""
    AutoRIFT.MetalGPU()

Run the correlation on an Apple GPU through Metal.jl. Needs `using Metal`.

Requires Metal.jl 1.10 or later, which provides the batched FFT it uses.

Experimental, and slower than a threaded CPU run on one image pair; see [`AutoRIFT.Backend`](@ref).
"""
struct MetalGPU <: Backend end

"""
    AutoRIFT.CUDAGPU()

Run the correlation on an NVIDIA GPU through CUDA.jl. Needs `using CUDA`.

Not implemented. The kernels in `ext/gpu/` are written against KernelAbstractions.jl, but only the Metal
adapter exists, so correlating with this backend throws `ArgumentError` and loading CUDA does not help.
"""
struct CUDAGPU <: Backend end

"""
    AutoRIFT.isgpu(backend) -> Bool

Whether `backend` executes on a device with its own memory. `true` for [`AutoRIFT.MetalGPU`](@ref) and
[`AutoRIFT.CUDAGPU`](@ref), `false` for [`AutoRIFT.CPU`](@ref). A new backend defines its own method.
`params` rejects a GPU backend combined with `threaded = true`.
"""
isgpu(::Backend) = false
isgpu(::MetalGPU) = true
isgpu(::CUDAGPU) = true

# Which package carries each backend's kernels. See `required_package` in `src/firstguess.jl`, which
# documents the function and gives the `FirstGuess` methods; the same error shape serves both, since
# both name an extension a caller has to load.
required_package(::MetalGPU) = "Metal"
required_package(::CUDAGPU) = "CUDA"

# ---------------------------------------------------------------------------
# Pass geometry
# ---------------------------------------------------------------------------

"""
    AutoRIFT.PassGeometry

The largest chip and search radius any point in a pass uses, as two [`AutoRIFT.Extent`](@ref) fields:
`chip` and `radius`.

These bound the pass rather than describe every point. A point is correlated at its own radius rounded
up to a rung of the `AutoRIFT.RADIUS_BUCKET_RUNGS` ladder and clamped to `radius`, so the pass uses one
transform length per bucket its points reach. `chip` is uniform within a level.

A different transform length changes `correlation` by about 1e-7 and, rarely, moves a displacement by
one refinement step. A point in the topmost bucket is correlated at the maximum radius of its point
set, which is smaller for a subset than for the whole set. To make a subset reproduce the whole set's
answer, pass the whole set's geometry to [`track!`](@ref); it may widen the pass but not narrow it.

Build one with [`AutoRIFT.pass_geometry`](@ref).
"""
struct PassGeometry
    chip::Extent
    radius::Extent
end

# ---------------------------------------------------------------------------
# Outlier rejection
# ---------------------------------------------------------------------------

"""
    OutlierMethod

Abstract supertype for how implausible displacements are identified and dropped.

Correlation returns a displacement at every searched point, including points where the peak was noise.
False matches are arbitrary vectors that disagree with their neighbors, whereas neighboring points on
one moving surface move similarly. Each method tests that spatial coherence.

Concrete subtypes: [`GardnerFilter`](@ref), [`NoOutlierFilter`](@ref).

Selected with `params`' `outliers` keyword, which takes a Symbol (`:gardner`, `:none`) or an instance.
"""
abstract type OutlierMethod end

"""
    GardnerFilter(; window = 5, iterations = 3, min_agree_fraction = 8/25,
                    agree_tolerance = 0.2, mad_scale = 4.0)

The two-stage filter of Gardner et al. (2018), as implemented in autoRIFT's `DISP_FILT`.

Stage 1 keeps a point only if at least `min_agree_fraction` of the window area lies within
`agree_tolerance` of it, which removes isolated wild vectors. Stage 2 keeps a point only if it lies
within `mad_scale` median absolute deviations of its neighborhood median, which catches a false match
near other false matches. Both stages work on displacements normalized by the local search radius and
iterate, since removing one outlier can expose its neighbors.

# Keywords
- `window = 5`: neighborhood width in grid points. Odd and at least 3.
- `iterations = 3`: iterations of stage 1. Stage 2 runs one fewer, minimum one, as in the reference.
- `min_agree_fraction = 8/25`: fraction of the full window area that must agree, in `[0, 1]`. The
  default is eight of the 25 points of a 5x5 window.
- `agree_tolerance = 0.2`: how close counts as agreement, as a fraction of the local search radius,
  in `(0, 1]`.
- `mad_scale = 4.0`: how many median absolute deviations from the neighborhood median a point may lie
  before stage 2 rejects it. Positive.

An invalid value throws `ArgumentError`.

!!! note "Relation to the normalized median test"
    This is inspired by the universal outlier detection of Westerweel & Scarano (2005) but is not that
    test. The center point is included in both the median and the MAD, where Westerweel & Scarano
    exclude it. The normalization is by local search radius rather than by `MAD + ε` with an absolute
    `ε`. The threshold is 4 rather than their universal 2. The agreement stage has no counterpart
    there.

See [`AutoRIFT.reject_outliers`](@ref).
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

Keep every measured displacement. A single false match can dominate a downstream fit, so this is
for diagnosis: it separates the correlator's failures from the filter's rejections.
"""
struct NoOutlierFilter <: OutlierMethod end

"""
    AutoRIFT.window(method::OutlierMethod) -> Int

Neighborhood width the method needs, in grid points, or `0` if it needs none.

The coarse pass uses it to skip the filter on a decimated grid narrower than the window.
"""
window(f::GardnerFilter) = f.window
window(::NoOutlierFilter) = 0

"""
    AutoRIFT.relax(method::OutlierMethod) -> OutlierMethod

A looser variant of `method` for a decimated grid.

The coarse pass runs on a strided subset, so a point's neighbors lie further away and resemble it
less. Holding them to the fine pass's standard would reject coherent motion. What loosening means is
the method's own: [`GardnerFilter`](@ref) drops one iteration (minimum one), and
[`NoOutlierFilter`](@ref) returns itself.
"""
relax(f::GardnerFilter) = GardnerFilter(;
    window = f.window, iterations = max(f.iterations - 1, 1),
    min_agree_fraction = f.min_agree_fraction,
    agree_tolerance = f.agree_tolerance, mad_scale = f.mad_scale)
relax(m::NoOutlierFilter) = m

"""
    AutoRIFT.rescale(method::OutlierMethod, oversample, stride = 1) -> OutlierMethod

`method` with its neighborhood matched to a grid posted `oversample` times finer than its chip.

When the grid spacing divides the chip size, adjacent points share most of their imagery. At
`chip_size = 16` and `grid_spacing = 8`, a point's 5x5 neighborhood spans two chip widths. The
reference applies two corrections (`autoRIFT.py:484-505`), which [`GardnerFilter`](@ref) implements:

  * The window widens to `(window - 1) * oversample + 1`, so it spans the same ground as at one point
    per chip.
  * The agreement fraction rises to `frac * (1 - overlap) + overlap^2`, where `overlap` is the
    fraction of a neighbor's chip shared with the center's, because overlapping chips agree partly by
    seeing the same pixels.

`stride` is the coarse pass's decimation, which reduces the overlap because decimated neighbors sit
further apart. Pass it for the coarse filter and leave it at `1` for the fine filter. `oversample < 1`
throws `ArgumentError`; `oversample == 1` returns the method unchanged. [`NoOutlierFilter`](@ref)
returns itself.

Separate from [`relax`](@ref): `relax` handles a decimated grid and this handles a grid finer than its
chips. A run can need either, both or neither.

For a Landsat pair over Jakobshavn at `chip_size = 16`, `grid_spacing = 8`, the unscaled 5-wide filter
at 0.32 agreement scores a 0.916 speed correlation against the ITS_LIVE granule, and the rescaled
filter scores 0.996.
"""
function rescale(f::GardnerFilter, oversample::Integer, stride::Integer = 1)
    oversample >= 1 || throw(ArgumentError(
        "`oversample` must be at least 1, got $oversample"))
    oversample == 1 && return f
    overlap = max(1 - stride / oversample, 0.0)
    return GardnerFilter(;
        window = (f.window - 1) * oversample + 1,
        iterations = f.iterations,
        min_agree_fraction = f.min_agree_fraction * (1 - overlap) + overlap^2,
        agree_tolerance = f.agree_tolerance, mad_scale = f.mad_scale)
end
rescale(m::NoOutlierFilter, ::Integer, ::Integer = 1) = m

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

"""
    Params

Fully resolved, immutable parameter set, built by [`AutoRIFT.params`](@ref) from keywords. Every
Symbol has been mapped to a method object and every number validated.

The method choices are type parameters rather than fields, so the kernels specialize on them.
`similarity` and `subpixel` are tuples with one entry per chip-size level; a tuple shorter than the
level list has its last entry repeated, and a scalar keyword gives a 1-tuple. The reference's sub-pixel
denominator depends on chip size, so a level's displacement may be quantized to 1/16 px at the base
chip size and to 1/32 or 1/64 above it. See [`chip_measures`](@ref) and `chip_subpixels`.

The positional constructor is stable API; its arguments follow the field declaration order. Shorter
forms omit `min_chip_valid_fraction` and `fill_min_hole` (defaults `0.65` and `5`) and, in the
shortest, `backend` (default `CPU()`).
"""
struct Params{S<:Tuple{SimilarityMeasure,Vararg{SimilarityMeasure}},P<:PreprocessMethod,
              R<:Tuple{SubpixelMethod,Vararg{SubpixelMethod}},O<:OutlierMethod,T<:BoolAsType,
              W<:RotationMethod,B<:Backend}
    similarity::S
    preprocess::P
    subpixel::R
    outliers::O
    threaded::T
    rotation::W

    # Chip geometry (pixels). Levels are `chip_size_min .* 2^k` up to `chip_size_max`, and
    # `chip_size_max ./ chip_size_min` is the same power of two in both axes — so the aspect is
    # constant across levels, which is what makes a coarse grid point exactly a block of fine ones.
    chip_size_min::Extent
    chip_size_max::Extent

    # Grid geometry (pixels). Also the default reference for how far apart output points sit.
    grid_spacing::Extent

    # Search geometry (pixels). Radius, not width: the search window spans
    # 2*radius. Naming this "limit" as the reference does is what produced its
    # off-by-one errors.
    #
    # x and y are independent, not a convenience split of one number. Geogrid
    # projects the a-priori velocity onto each image axis separately, so an ice
    # stream flowing along x legitimately gets a wide x-radius and a narrow
    # y-radius. This is the fallback when no per-pixel radius field is supplied;
    # the per-pixel case carries two separate arrays in lockstep.
    search_radius::Extent
    min_search_radius::Int
    coarse_stride::Int
    coarse_buffer::Int
    min_coarse_valid_fraction::Float64

    # A chip is correlated only if more than this fraction of its own footprint — not the larger
    # search window around it — is valid. `_any_valid`'s old all-or-nothing test is this at 0.0.
    min_chip_valid_fraction::Float64

    # A-priori displacement (pixels), the predictor in a predictor-corrector
    # sense: the search window is centred on it rather than on zero.
    dx_prior::Float64
    dy_prior::Float64

    # Hole filling. A hole is closed if it has `fill_window`-neighbourhood support or if it is a
    # connected region smaller than `fill_min_hole`; the two criteria are independent, and
    # `fill_min_hole = 0` disables the second.
    fill_window::Int
    fill_min_hole::Int

    # Misc.
    rng_seed::UInt64
    progress::Bool

    # Where the kernels run. Last, so the positional constructor every earlier caller wrote keeps
    # working through the outer method below — see [`AutoRIFT.Backend`](@ref) for why this is a
    # type parameter rather than a plain field.
    backend::B
end

# Holes of one to four points are closed on size alone. The reference's `bwareaopen(..., 5)`
# threshold (`autoRIFT.py:803`), and exclusive: a five-point hole is not small enough.
const _DEFAULT_FILL_MIN_HOLE = 5

# See `params`'s own docstring for the value and the measurement behind it.
const _DEFAULT_MIN_CHIP_VALID_FRACTION = 0.65

# The positional form without a backend, which is the documented stable API and what `app/` calls.
# Appends `CPU()`, so an existing 18-argument call is unchanged in meaning.
#
# The inner constructor is reached directly with the concrete field types rather than by recursion
# through the outer one, so there is exactly one place the field order is written.
Params(similarity, preprocess, subpixel, outliers, threaded, rotation, chip_size_min,
       chip_size_max, grid_spacing, search_radius, min_search_radius, coarse_stride,
       coarse_buffer, min_coarse_valid_fraction, dx_prior, dy_prior, fill_window, rng_seed,
       progress) =
    Params(similarity, preprocess, subpixel, outliers, threaded, rotation, chip_size_min,
           chip_size_max, grid_spacing, search_radius, min_search_radius, coarse_stride,
           coarse_buffer, min_coarse_valid_fraction, _DEFAULT_MIN_CHIP_VALID_FRACTION,
           dx_prior, dy_prior, fill_window, _DEFAULT_FILL_MIN_HOLE, rng_seed, progress, CPU())

# The form without `fill_min_hole`, which the 18-argument one above also routes through. Kept so a
# caller written against the field list before hole size was a criterion still compiles, and gets
# the reference's threshold rather than a silently disabled one.
Params(similarity, preprocess, subpixel, outliers, threaded, rotation, chip_size_min,
       chip_size_max, grid_spacing, search_radius, min_search_radius, coarse_stride,
       coarse_buffer, min_coarse_valid_fraction, dx_prior, dy_prior, fill_window, rng_seed,
       progress, backend) =
    Params(similarity, preprocess, subpixel, outliers, threaded, rotation, chip_size_min,
           chip_size_max, grid_spacing, search_radius, min_search_radius, coarse_stride,
           coarse_buffer, min_coarse_valid_fraction, _DEFAULT_MIN_CHIP_VALID_FRACTION,
           dx_prior, dy_prior, fill_window,
           _DEFAULT_FILL_MIN_HOLE, rng_seed, progress, backend)

"""
    chip_sizes(p::Params) -> Vector{Extent}

The chip-size levels, ascending: `chip_size_min .* 2^k` up to `chip_size_max`. Both axes double
together, because `params` requires `chip_size_max ./ chip_size_min` to be the same power of two in
each axis.

The order matters: a level writes only where no finer level produced a value, so the smallest chip
that yields a valid estimate wins. Small chips resolve detail; large chips succeed in low-texture
areas.
"""
function chip_sizes(p::Params)
    sizes = Extent[]
    cs = p.chip_size_min
    while cs.X <= p.chip_size_max.X
        push!(sizes, cs)
        cs = (X = 2cs.X, Y = 2cs.Y)
    end
    return sizes
end

"""
    AutoRIFT.extent(x) -> Extent

An [`AutoRIFT.Extent`](@ref) from a scalar, a tuple, or a named tuple.

A scalar means square: `32` is `(X = 32, Y = 32)`. `(X = 16, Y = 32)` names the axes, and a bare
`(16, 32)` or a two-element vector is read as x then y. A non-integer value, a named tuple with other
field names, or a vector of another length throws `ArgumentError`.
"""
extent(x::Integer) = Extent((Int(x), Int(x)))
extent(x::Extent) = x
extent(x::NamedTuple{(:X, :Y)}) = Extent((Int(x.X), Int(x.Y)))
extent(x::Tuple{Integer,Integer}) = Extent((Int(x[1]), Int(x[2])))
extent(x::AbstractVector) = length(x) == 2 ? Extent((Int(x[1]), Int(x[2]))) :
    throw(ArgumentError("an extent needs 1 or 2 values, got $(length(x))"))
extent(x::NamedTuple) = throw(ArgumentError(
    "an extent's fields must be `X` and `Y`, got a named tuple with $(length(x)) other field(s)"))

# A fractional value is a caller mistake rather than a rounding request, and anything else is the
# wrong kind entirely. Named here rather than left to a `MethodError`, which would report a method
# table instead of the constraint.
extent(x::Real) = throw(ArgumentError(
    "an extent must be a whole number of pixels: expected an integer value, got $x"))
extent(x) = throw(ArgumentError(
    "an extent must be a number or a pair of numbers, got a $(typeof(x))"))

"""
    chip_measures(p::Params, nlevels::Integer) -> Tuple

The similarity measure for each of `nlevels` chip-size levels, as a tuple, finest first.
`nlevels` defaults to `length(chip_sizes(p))`.

Where `p.similarity` is shorter than `nlevels`, its last entry repeats: `similarity = :zncc` gives ZNCC
at every level, and `(:coherence, :zncc)` gives coherence at the finest chip size and ZNCC at every
coarser one (the escalation of Joughin, 2002). A tuple longer than `nlevels` throws `ArgumentError`.

This is for inspecting a configuration. The chip-size loop calls [`measure_at`](@ref), which does not
allocate.
"""
function chip_measures(p::Params, nlevels::Integer = length(chip_sizes(p)))
    _check_measures(p, nlevels)
    return ntuple(k -> measure_at(p, k), nlevels)
end

"""
    measure_at(p::Params, level::Integer) -> SimilarityMeasure

The measure for chip-size level `level` (1 = finest), with the last tuple entry repeating.

The indexed form of [`chip_measures`](@ref), called by the chip-size loop. It does not allocate, and
the result keeps its concrete type because `p.similarity` is a tuple.
"""
@inline measure_at(p::Params, level::Integer) =
    p.similarity[min(level, length(p.similarity))]

"""
    chip_subpixels(p::Params, nlevels = length(chip_sizes(p))) -> Tuple

The subpixel method each chip-size level will use, finest first, with the last tuple entry
repeating.

The counterpart of [`chip_measures`](@ref) for `p.subpixel`, for inspecting a configuration. The
chip-size loop calls `subpixel_at`, which does not allocate.
"""
function chip_subpixels(p::Params, nlevels::Integer = length(chip_sizes(p)))
    _check_subpixels(p, nlevels)
    return ntuple(k -> subpixel_at(p, k), nlevels)
end

"""
    subpixel_at(p::Params, level::Integer) -> SubpixelMethod

The subpixel method for chip-size level `level` (1 = finest), with the last tuple entry repeating.

The sub-pixel method depends on the level because the reference's `OverSampleRatio` is keyed by chip
size. The production driver passes `{16, 32, 64, 64}` for optical input and `{32, 64, 128, 128}` for
radar, keyed by `ChipSize0X * [1, 2, 4, 8]` (`autoRIFT.py:652-653`, `testautoRIFT.py:488-510`), so a
coarse level locates its peak more finely than the base level. Does not allocate; the result keeps its
concrete type.
"""
@inline subpixel_at(p::Params, level::Integer) =
    p.subpixel[min(level, length(p.subpixel))]

# Shared by `chip_measures` and the chip-size loop. A tuple longer than the level list means the
# extra measures would silently never run, which is a configuration error rather than something to
# quietly ignore.
function _check_measures(p::Params, nlevels::Integer)
    n = length(p.similarity)
    n <= nlevels || throw(ArgumentError(
        "`similarity` names $n measures but there are only $nlevels chip-size levels. Widen " *
        "the `chip_size_min`/`chip_size_max` range, or name fewer measures — the last one " *
        "applies to every remaining level."))
    return nothing
end

# As `_check_measures`, for the subpixel tuple — except that entries past the last level are accepted
# when each repeats the last level's method. Those are what `subpixel_at` would return for them anyway,
# so they change nothing, and they let one ladder type cover every level count: a run whose level count
# is known only at run time keeps a concrete `Params`.
function _check_subpixels(p::Params, nlevels::Integer)
    n = length(p.subpixel)
    (n <= nlevels || _repeats_from(p.subpixel, nlevels)) || throw(ArgumentError(
        "`subpixel` names $n methods but there are only $nlevels chip-size levels. Widen the " *
        "`chip_size_min`/`chip_size_max` range, or name fewer methods — the last one applies to " *
        "every remaining level."))
    return nothing
end

# Whether every entry of `t` after `k` equals entry `k`.
_repeats_from(t::Tuple, k::Integer) = k >= 1 && all(i -> t[i] == t[k], (k + 1):length(t))

# Search-radius normalisation lives with the point set it operates on; see
# `sanitize!` in points.jl.
