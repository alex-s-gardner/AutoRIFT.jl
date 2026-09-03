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
single-look-complex radar under [`Coherence`](@ref).

`Union{Real,Complex}` rather than `Number`, which is what the type walls first said. `Number` admits
types no path here handles, which die as a bare `MethodError` several layers down with no mention of
the actual constraint. Naming the real bound once makes the signatures state what the pipeline
supports instead of merely more than it supports.
"""
const ImageElement = Union{Real,Complex}

"""
    AutoRIFT.Extent

A quantity with an x and a y component, in pixels: `(X = 16, Y = 32)`.

Chip size, search radius and grid spacing are all two-dimensional, and naming the axes is what
makes a transposition visible. `chip.X` reads as itself where `chip[1]` needs the convention looked
up — and a transposed extent is the error this package has already made twice, once in a point set
and once in an FFT size that "for a symmetric test size looks like it works".

A `NamedTuple` rather than a struct, and that is load-bearing three ways. It stays `isbitstype`, so
`Params` does and the trimmed binary keeps its inline layout. It is byte-identical to
`Tuple{Int,Int}` — 16 bytes, offsets 0 and 8 — so it maps to `struct { int64_t x, y; }` with no
padding for a C caller, since the names live in the type rather than the data. And it is writable as
a literal, where a struct would need a constructor call at every site.

Build one from a scalar or a tuple with [`AutoRIFT.extent`](@ref); a scalar means square.
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
\\gamma = \\frac{\\left| \\sum \\overline{T'} \\cdot W' \\right|}
               {\\sqrt{\\sum |T'|^2 \\sum |W'|^2}}
```

where ``T'`` and ``W'`` are the chip and the window with their (complex) means removed — the
complex analogue of [`ZNCC`](@ref), and mean-removed on *both* sides for the same reason: it is
what makes ``\\gamma(T, T) = 1`` exactly. The surface is real — the magnitude — so peak location
and the subpixel cascade are the same code the real measures use.

Requires complex input; a real image has no phase to exploit and [`ZNCC`](@ref) is the measure
for it.

# Why bother, and when not to

Complex matching buys **resolution**, not accuracy. Joughin (2002) found the complex
cross-correlation function "more strongly peaked" in low-correlation regions, so a match that
amplitude needs 64x64 to achieve is available at 24x24 — and on ice, where speckle decorrelates
fast, that difference decides whether a shear margin is resolved or smoothed over.

The cost is that it fails where amplitude does not:

!!! warning "Phase variation destroys the peak"
    Interferometric phase across the chip can *reduce or eliminate* the correlation peak, which
    is worst exactly where the science is most interesting — high shear and steep topography.
    Amplitude matching is unaffected there.

    Two consequences. First, run [`Deramp`](@ref) as the preprocessing step: it removes the
    linear component of that phase variation, which is the part that is both dominant and
    cheap to estimate. Second, expect to fall back. Passing a *tuple* to `similarity` runs
    coherence at the finest chip size and a real measure above it, so a point coherence cannot
    resolve is left for a larger amplitude chip:

    ```julia
    autorift(z1, z2; similarity = (:coherence, :zncc), preprocess = :deramp,
             chip_size = 32, chip_size_max = 128)
    ```

# Provenance

There is no reference implementation of this to match, which is worth stating plainly.
autoRIFT v2.1.2 has no complex path at all — its core exposes only real and `UInt8` entry
points — and ISCE2's `cuAmpcor` takes `abs` of complex input before correlating, so both
reduce to amplitude matching. The estimator above and the escalation strategy follow Joughin
(2002); the implementation is verified against analytic cases (γ(T,T) = 1, a known shift, a
known phase ramp) rather than against another program's output.

Joughin, I. (2002). Ice-sheet velocity mapping: a combined interferometric and speckle-tracking
approach. *Annals of Glaciology* 34, 195-201.
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
    Deramp(; axis = :both)

Remove the linear phase ramp from a complex image. **Complex input only.**

A phase ramp is a frequency shift: it moves the band centre away from zero, and a signal whose
band is off-centre aliases when oversampled. Since [`PyramidRefine`](@ref) oversamples the
correlation surface, this is a correctness requirement for complex correlation rather than an
enhancement — ISCE2's `cuAmpcor` says the same in `cuDeramp.cu`: removing the ramp "is necessary
before oversampling a complex signal".

It also directly addresses the failure mode [`Coherence`](@ref) warns about. Phase variation
across a chip suppresses the correlation peak; the linear component of that variation is both the
dominant part and the part that can be estimated from the chip alone, with no interferogram.

The estimate follows `cuAmpcor`'s method 1. For each axis, sum the conjugate product of adjacent
pixels and take the argument of that sum:

```math
\\phi_x = \\arg \\sum_{i} \\overline{z_i} \\, z_{i + \\Delta x}
```

Summing the product rather than averaging per-pixel phase differences makes the estimate
**amplitude-weighted for free** — a bright pixel pair contributes more than a dim one, which is
what you want when speckle means dim pixels are mostly noise. It also avoids wrapping: no
`atan` is taken until after the sum.

`axis` selects `:x`, `:y`, or `:both`. Deramping one axis is occasionally right for data whose
ramp is known to lie along track.

!!! note "Not a substitute for amplitude fallback"
    A linear ramp is the first term of the phase variation, not all of it. Steep topography and
    high shear produce higher-order variation this cannot remove, which is why
    [`Coherence`](@ref) recommends a measure tuple that falls back to amplitude.
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
    RotationSearch(angles = (-3.0, 0.0, 3.0); about = 0.0)
    NoRotationSearch()

Whether the dense stage tries several chip rotations and keeps the best.

Opt-in, and off by default, because it multiplies the correlation cost by `length(angles)` — the
whole point of `nansencenter/sea_ice_drift`'s `rotate_and_match` is that you pay 3x to recover the
matches a rotating field would otherwise lose. Its own default is `angles=[-3, 0, 3]`, which is
where this default comes from.

```julia
autorift(a, b, guess; rotation = RotationSearch())              # ±3°, 3x cost
autorift(a, b, guess; rotation = RotationSearch((-6, -3, 0, 3, 6)))
```

# When this earns its cost

Sea ice rotates, and a rotated chip decorrelates against an unrotated window — the peak weakens even
though the ice is perfectly trackable. Measured on synthetic speckle, median peak correlation with
five angles (±3°, ±6°, 0°) against none:

| scene rotation | chip 32 | chip 64 |
|---:|---:|---:|
| 0° | 0.571 → 0.571 (0%) | 0.571 → 0.571 (0%) |
| 3° | 0.113 → 0.146 (**+29%**) | 0.112 → 0.279 (**+149%**) |
| 6° | 0.085 → 0.103 (+21%) | 0.043 → 0.052 (+21%) |
| 10° | 0.082 → 0.101 (+23%) | 0.042 → 0.053 (+25%) |

Two things worth reading off that. **The gain grows with chip size** — a 64-px chip's corners travel
twice as far as a 32-px chip's under the same rotation, so it has more to lose and more to recover.
And **the benefit is real but modest against decorrelation**: at 6° and beyond the correlation is
weak with or without the search, because rotating a square chip pulls in padding that was never part
of it. That is why `sea_ice_drift`'s own default is only ±3°.

Cost measured at **1.7x** for five angles, not 5x, because the surrounding per-point work — window
extraction, integral images, the peak search — is shared across angles.

Also measured through the first-guess path: usable vectors fall from 2327 at 0° rotation to 29 at
10°. Most of that loss is in the *sparse* stage's descriptor matching rather than the dense
correlation, so chip rotation alone does not recover it — see [`AKAZEGuess`](@ref), which holds
95-99% match precision where ORB falls to 19%.

It is *not* a substitute for [`Deramp`](@ref) or for the consistency filter, and it does not handle
**shear** — no published sea-ice tracker does. Both references tolerate shear instead by keeping the
filtering neighbourhood small enough that shear looks locally like rotation; deformation is then
computed *from* the vector field afterwards rather than corrected for during matching.

The best-fitting angle is available per point, which makes it a measurement rather than only a
correction — `sea_ice_drift` returns it as `best_a`.

# `about`: centring the search on a scene-level rotation

`about` is the **scene's** rotation, in the same sense [`scene_rotation`](@ref) reports it, and the
chip rotations actually tried are `angles .- about`. This is `rotate_and_match`'s `alpha0`, which
appears in its template call as exactly `angle - alpha0`.

The subtraction is not a convention to be chosen: the chip comes from the *secondary* and is
correlated against an *unrotated* reference window, so it has to be turned **back** to the
reference's orientation. Measured on speckle rotated 8°, peak correlation of a chip rotated by each
candidate — the counter-rotation is the only one that recovers anything, and by more the larger the
chip:

| chip | no rotation | +8° | −8° |
|---:|---:|---:|---:|
| 32 | 0.723 | 0.345 | 0.597 |
| 64 | 0.328 | 0.132 | **0.641** |
| 128 | 0.142 | 0.025 | **0.670** |

At chip 128 that is 0.14 → 0.67, a **4.7x** recovery. At chip 32 counter-rotation still loses to no
rotation at all, because a 32-px chip's corners travel only ~2 px at 8° while resampling and corner
padding cost more than that — which is the same "the gain grows with chip size" effect as the table
above, seen from the other end.

It matters because the search window is narrow on purpose. A scene rotated 8° is outside `±3°`
entirely — every angle tried is wrong by at least 5°, and widening the tuple to reach it costs a
correlation per angle for angles that can only ever lose. One scene-level estimate moves the whole
window instead, so `±3°` around 8° searches 5-11° at the same 3x cost.

[`scene_rotation`](@ref) is where that estimate comes from, and where the choice to fit it from the
sparse vectors rather than from geolocation is argued.
"""
abstract type RotationMethod end

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

The chip rotations a pass will try, in degrees — the rotations **applied to the chip**, so `about` is
already subtracted out.

That is why this is a function rather than a field read: `m.angles` is the search window and
`m.about` is the scene rotation it is centred on, and what `_rotate_chip` needs is neither of those
but `angles .- about`. Resolving it here means the correlator and the tests cannot disagree about the
sign — which mattered, since the first version of this had it backwards and the table in
[`RotationSearch`](@ref) is what caught it.

No `about == 0` fast path. One was written, on the assumption the common case should skip the
subtraction, and it is measurably *slower* than the unconditional form in **both** cases — 2.08
against 1.83 ns at `about = 0` — because subtracting a constant from a tuple is free and the branch
is not.

Only defined for [`RotationSearch`](@ref). There was a `NoRotationSearch` method returning `(0.0,)`,
justified as letting the caller skip a branch — but no such caller exists: the correlation dispatches
on the method *type* first (`_correlate_rotations!` has a `NoRotationSearch` method that never
consults this), which is what makes the off path compile to the unrotated call. A method whose stated
purpose is contradicted by the dispatch above it is worse than no method.
"""
angles(m::RotationSearch) = m.angles .- m.about

"""
    filter_width(method::PreprocessMethod)

Side length of the filter window, or `0` for methods that take no window.
"""
filter_width(::Union{NoPreprocess,Decibel,Deramp}) = 0
filter_width(m::Union{Highpass,Wallis,WallisGapfill,Sobel,Laplacian}) = m.width

"""
    AutoRIFT.filter_reach(method::PreprocessMethod) -> Int

How many pixels beyond a region must be supplied for the filter's output *inside* that region to
equal what filtering the whole image would give.

**Negative means no finite reach**, so check the sign before using the result as a width. A filter
that estimates from the whole image — [`Deramp`](@ref) — cannot be reproduced from a window of any
size, and returns `-1` rather than a large number that would merely be less wrong. Adding a negative reach
to a correlation extent yields a halo *shorter* than the correlation alone, which is exactly the
silent under-read this trait exists to prevent; [`AutoRIFT.halo`](@ref) throws instead.

Separate from [`filter_width`](@ref) because the two differ, and the difference is not a detail: a
filter that applies two chained window passes reaches twice its half-width, since the second pass
consumes the first's output over its own window. Tiled processing sizes its halo from this, so a
value that is too small produces a filter output that is quietly wrong near every block edge rather
than merely different — measured at 792 of 10201 points differing by up to 0.21 for `Wallis(5)` when
padded by `width ÷ 2` instead of twice that.

A reach may also be far larger than the window suggests when the filter's *decisions* are not
windowed: [`WallisGapfill`](@ref) reaches `GAPFILL_REACH` plus its dilation plus its window, because
whether a pixel is filled depends on how far the nearest real data lies.

Pinned per method by a test that measures the true reach and compares it against this trait, so the
two cannot drift.
"""
filter_reach(m::PreprocessMethod) = filter_width(m) ÷ 2

# `Wallis` subtracts a local mean and then divides by a local standard deviation computed *about
# that mean*, so the window is applied twice in sequence and each output depends on a neighbourhood
# twice as wide.
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

Selected with the `backend` keyword, which accepts the `Symbol` naming a backend or an
instance. `CPU()` is the default and the only one that needs no additional package.

The GPU backends are **experimental and do not currently outperform the CPU**: a device pass is
2.7-3.2x one CPU core, but slower than a threaded CPU run on a single image pair. They are worth
selecting where the cores are already committed and the device is idle. See `docs/gpu.md`.

A singleton per backend, carried as a `Params` **type parameter** rather than a field value, so the
grid loop's choice is resolved at compile time and the unused paths are eliminated — the same
arrangement `threaded` uses, for the same reason. That is also what keeps `Params` `isbitstype`, and
what makes the GPU support invisible to a `--trim`ed binary: `app/` builds a `Params` carrying
`CPU()`, so no GPU method is reachable from `main` and none is compiled in. The kernels themselves
live in package extensions, so a caller who never loads `Metal` never pays for it either.

The GPU backends are declared here rather than in their extensions because a `Params` type
parameter must exist before the extension that implements it loads — a caller may construct
`params(; backend = :metal)` and only then `using Metal`. Selecting a backend whose package is
absent throws from [`AutoRIFT.required_package`](@ref)'s message rather than a `MethodError`.
"""
abstract type Backend end

"""
    AutoRIFT.CPU()

Run the correlation on the CPU, threaded or not per the `threaded` keyword. The default.
"""
struct CPU <: Backend end

"""
    AutoRIFT.MetalGPU()

Run the correlation on an Apple GPU through Metal.jl. Needs `using Metal`.

Requires Metal.jl 1.10 or later, which is where the batched FFT this depends on landed.

Experimental, and slower than a threaded CPU run on one image pair — see [`AutoRIFT.Backend`](@ref).
"""
struct MetalGPU <: Backend end

"""
    AutoRIFT.CUDAGPU()

Run the correlation on an NVIDIA GPU through CUDA.jl. Needs `using CUDA`.

**Not implemented.** The kernels in `ext/gpu/` are vendor-neutral, written against
KernelAbstractions.jl, but only the Metal adapter ships — so selecting this backend finds no
extension to load. Declared here because a `Params` type parameter must exist before the extension
that implements it, which is what makes adding the adapter a new file rather than a change here.
"""
struct CUDAGPU <: Backend end

"""
    AutoRIFT.isgpu(backend) -> Bool

Whether `backend` executes on a device with its own memory.

A trait rather than an `isa` union at each call site, so a backend added later declares its own
answer instead of being added to a branch. What it gates is the choices that follow from having a
separate address space — batching, host/device transfer, and the rejection of intra-pair threading.
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

The largest chip and search radius any point in a pass uses.

These size the correlation workspace, and a workspace sizes its own FFT buffers from its extents
— so they set the **transform length** every point in the pass executes, not merely how much
memory it takes. A pass whose maxima are `(32, 32, 25, 25)` runs an 84-point transform where one
with `(32, 32, 10, 10)` runs a 56-point one, and the two agree only to about 4e-7. Pooling depends
on the same property; see `take_workspace!`.

That is why this is a value a caller can supply rather than only something derived per pass. A
subset of a point set has its own, generally smaller, maxima — so correlating a subset computes a
different transform than correlating the whole and reading those points out of it. Passing the
whole set's geometry to the subset is what makes the two agree. Measured: without it, `correlation`
differs on 46% of the points of a sub-block, and `dx`/`dy` can flip a subpixel step.

Build one with [`AutoRIFT.pass_geometry`](@ref).
"""
struct PassGeometry
    chip_x::Int
    chip_y::Int
    radius_x::Int
    radius_y::Int
end

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

"""
    AutoRIFT.rescale(method::OutlierMethod, oversample, stride = 1) -> OutlierMethod

`method` with its neighbourhood matched to a grid posted `oversample` times finer than its chip.

When grid spacing divides the chip size, adjacent grid points share most of their imagery: at
`chip_size = 16` and `grid_spacing = 8` a point's 5x5 neighbourhood spans two chip widths, and
its neighbours are largely the same pixels. Two things follow, and the reference applies both
(`autoRIFT.py:484-505`):

  * The window widens to `(window - 1) * oversample + 1`, so the neighbourhood spans the same
    *ground* it would at one point per chip rather than a fraction of it.
  * The agreement fraction rises to `frac * (1 - overlap) + overlap^2`, where `overlap` is the
    fraction of a neighbour's chip shared with the centre's. Overlapping chips agree partly
    because they see the same pixels, so the evidence a neighbour provides is worth less and
    more neighbours must supply it.

`stride` is the coarse pass's decimation; it reduces the overlap, because decimated neighbours
sit further apart. Pass it for the coarse filter and leave it at one for the fine.

Not folded into [`relax`](@ref): that answers "this grid is decimated", this answers "this grid
is finer than its chips", and a run can need either, both, or neither.

Skipping this is measurable rather than theoretical. Against the ITS_LIVE granule for a Landsat
pair over Jakobshavn at `chip_size = 16`, `grid_spacing = 8` — where the reference would use a
9-wide window at a 0.41 fraction — the unscaled 5-wide filter at 0.32 scores a 0.916 speed
correlation against the rescaled filter's **0.996**, and admits 1.3% of estimates reporting fast
ice where the reference reports slow against **none**.
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

Fully resolved, immutable, concretely-typed parameter set. Constructed once by
[`AutoRIFT.params`](@ref) from user keywords; every `Symbol` has been mapped to
a method object and every number validated by the time a `Params` exists, so
nothing downstream ever re-checks or re-dispatches on a `Symbol`.

The method choices are type parameters, not fields, so the correlation and
filtering kernels specialize on them.

`similarity` is a *tuple* of measures, one per chip-size level, because a run may escalate
between them — coherence at the finest chip, amplitude above it. A scalar keyword resolves to a
1-tuple, and a tuple shorter than the level list has its last entry repeated, so the common case
of one measure everywhere is the 1-tuple and costs nothing. See [`chip_measures`](@ref).
"""
struct Params{S<:Tuple{SimilarityMeasure,Vararg{SimilarityMeasure}},P<:PreprocessMethod,
              R<:SubpixelMethod,O<:OutlierMethod,T<:BoolAsType,W<:RotationMethod,
              B<:Backend}
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

    # A-priori displacement (pixels), the predictor in a predictor-corrector
    # sense: the search window is centred on it rather than on zero.
    dx_prior::Float64
    dy_prior::Float64

    # Hole filling.
    fill_window::Int

    # Misc.
    rng_seed::UInt64
    progress::Bool

    # Where the kernels run. Last, so the positional constructor every earlier caller wrote keeps
    # working through the outer method below — see [`AutoRIFT.Backend`](@ref) for why this is a
    # type parameter rather than a plain field.
    backend::B
end

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
           coarse_buffer, min_coarse_valid_fraction, dx_prior, dy_prior, fill_window, rng_seed,
           progress, CPU())

"""
    chip_sizes(p::Params) -> Vector{Extent}

The chip-size levels actually used, ascending. Levels are `chip_size_min .* 2^k` up to
`chip_size_max`.

Both axes double together, which is well-defined because `params` requires
`chip_size_max ./ chip_size_min` to be the same power of two in each — so the aspect a caller asked
for holds at every level rather than drifting across them.

Ascending order is load-bearing: every level only writes where no finer level
already produced a value, so the *smallest* chip that yields a valid estimate
wins. Small chips resolve detail; large chips succeed in low-texture areas.
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

A scalar means square — `32` is `(X = 32, Y = 32)` — which is what nearly every call wants and what
keeps the common spelling short. `(X = 16, Y = 32)` names the axes; a bare `(16, 32)` is accepted as
x-then-y, matching the order every other pair in this package uses.
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

The similarity measure for each of `nlevels` chip-size levels, in order.

`p.similarity` is a tuple. Where it is shorter than the level list — which includes the usual
case of a single measure — the last entry repeats, so `similarity = :zncc` means ZNCC at every
level and `similarity = (:coherence, :zncc)` means coherence at the finest chip and ZNCC at every
coarser one.

That padding rule is what makes the escalation of Joughin (2002) expressible without a second way
to specify levels: the finest chip tries the sharper, more fragile measure, and the coarser chips
— which exist precisely to succeed where the fine ones failed — use the robust one. A measure
tuple *longer* than the level list is an error, since the extra entries would silently do nothing.

Returns a **tuple**, not a vector, and that is load-bearing rather than stylistic: a
`Vector{SimilarityMeasure}` has an abstract element type, so the measure reaching `chipsize_level`
would be known only as `SimilarityMeasure` — a dynamic dispatch in the chip-size loop, and enough
to make the whole package untrimmable (verified: it produced `unresolved call ... Core.kwcall`
and broke `app/`'s build).

For that reason the chip-size loop itself does not call this — it uses [`measure_at`](@ref), which
needs no allocation and no `ntuple` over a runtime length. **This function is for inspecting a
configuration**, not for running one: answering "what will each level actually use?" in one call,
which is otherwise a loop the caller has to write. The tests use it that way, and so would anyone
debugging a hierarchy.
"""
function chip_measures(p::Params, nlevels::Integer = length(chip_sizes(p)))
    _check_measures(p, nlevels)
    return ntuple(k -> measure_at(p, k), nlevels)
end

"""
    measure_at(p::Params, level::Integer) -> SimilarityMeasure

The measure for chip-size level `level` (1 = finest), with the last tuple entry repeating.

The indexed form of [`chip_measures`](@ref), and what the chip-size loop actually calls: it
allocates nothing, and because `p.similarity` is a tuple the result keeps its concrete type, so
the correlation kernel specializes and `--trim` can resolve the call.
"""
@inline measure_at(p::Params, level::Integer) =
    p.similarity[min(level, length(p.similarity))]

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

# Search-radius normalisation lives with the point set it operates on; see
# `sanitize!` in points.jl.
