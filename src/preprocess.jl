# Pre-correlation filtering.
#
# Layer 2: depends on nothing else in the package.
#
# Two acquisitions of the same scene differ radiometrically as well as geometrically —
# different sun angle, different atmosphere, different sensor gain. Correlation should
# respond to *texture*, not to that difference, so the low-frequency content is removed
# first. This is not a refinement: without it the correlation surface is dominated by
# the brightness mismatch and the peak is unreliable.
#
# ---------------------------------------------------------------------------
# Where these run in the pipeline
# ---------------------------------------------------------------------------
#
# In the production driver, only the high-pass filter runs inside the correlator. The
# Wallis and destriping filters are applied *outside* — before reprojection and before
# the grid is built — because they need the imagery in its native projection, where
# the sensor's own artefacts (Landsat-7 scan-line gaps, Landsat-4/5 striping) are
# axis-aligned. Resampling first would smear them into something no filter can remove.
#
# So the production order is **filter → reproject → build grid → correlate**, and these
# functions have to be callable standalone, ahead of everything else, as well as from
# inside `autorift`. That is why they take and return plain arrays with an explicit
# validity mask rather than reading and writing fields of a state object.

"""
    AutoRIFT.FiniteMask(parent)

The validity mask "every finite pixel of `parent`", computed on read rather than stored.

The default mask when a caller supplies none, and lazy for one reason: a blocked run must not form
any array the size of the scene, and `map(isfinite, img)` is exactly that — 16 MiB per image at
4096², plus a full read of an input that may be on disk. Reading a window of this reads only that
window of `parent`, so `_read_block!`'s `copyto!` over a view costs the block and nothing more.

Materialize with [`AutoRIFT.resident`](@ref) before reducing over the whole thing or reading it more
than once; `_prepare` does, which is what keeps every whole-array reduction downstream of a dense
mask.

`IndexStyle` is forwarded from `parent`, so a Cartesian-indexed disk array is not pushed through
linear index arithmetic it would have to undo.
"""
struct FiniteMask{T,A<:AbstractMatrix{T}} <: AbstractMatrix{Bool}
    parent::A
end

Base.size(m::FiniteMask) = size(m.parent)
Base.axes(m::FiniteMask) = axes(m.parent)
Base.IndexStyle(::Type{FiniteMask{T,A}}) where {T,A} = IndexStyle(A)
Base.@propagate_inbounds Base.getindex(m::FiniteMask, i::Int) = isfinite(m.parent[i])
Base.@propagate_inbounds Base.getindex(m::FiniteMask, I::Int...) = isfinite(m.parent[I...])

"""
    AutoRIFT.resident(mask) -> AbstractMatrix{Bool}

`mask` backed by storage: itself if it already is, one pass over it if it is computed.

Call this before any whole-array reduction over a mask, and before reading one repeatedly. The
filters do both — `_masked_boxmean!` and `_erode_mask!` each begin with `all(mask)` — so `_prepare`
materializes at its entry, which is what puts every such reduction downstream of a dense mask
rather than leaving each one to remember.

A blocked run never calls `_prepare`; it reads windows through `_read_block!` into dense buffers
instead. So the untiled path pays one pass and the blocked path pays nothing, without either
branching on which it is.

Only the lazy masks this package constructs are recognized. A caller supplying a computed
`AbstractMatrix{Bool}` of their own should either materialize it first or add a method here.
"""
resident(m::AbstractMatrix{Bool}) = m
resident(m::FiniteMask) = Matrix{Bool}(m)

# The windowed filters read their mask many times over and begin with a whole-array `all`, so a
# computed mask is the wrong representation for them: `all` on a `FiniteMask` costs 425 us at 1024²
# against 4.6 for a dense one, and the per-window reads then repeat that work.
#
# Materializing here rather than trusting every caller to is what makes the invariant structural.
# `_prepare` already calls `resident`, so these methods are unreachable from the untiled path, and
# `_read_block!` copies into dense buffers, so they are unreachable from the blocked one. They exist
# for the third caller — a future one — that would otherwise be correct but quietly slow.
_masked_boxmean!(out::AbstractMatrix{Float32}, img::AbstractMatrix, mask::FiniteMask, w::Int,
                 scratch::Union{Nothing,AbstractMatrix{Float32}}) =
    _masked_boxmean!(out, img, resident(mask), w, scratch)

_erode_mask!(out::AbstractMatrix{Bool}, mask::FiniteMask, w::Int) =
    _erode_mask!(out, resident(mask), w)

"""
    ImagePair{T}

The two images to correlate, with their validity masks.

`reference` is the earlier acquisition and `secondary` the later one. Naming them
rather than numbering them is deliberate: the reference implementation calls them `I1`
and `I2` and then passes them to its correlator in the opposite order, which is the
kind of mistake a name prevents and a number invites.

`reference_valid` and `secondary_valid` mark pixels carrying real data. Invalid pixels
arise from sensor gaps, cloud masks, and the no-data border left by reprojection, and
they must not be allowed to contribute to a correlation — a chip full of fill values
correlates beautifully with any other chip full of fill values.

The two images and the two masks carry separate type parameters, so a pair may mix containers: a
lazy reference against a resident secondary, a defaulted [`AutoRIFT.FiniteMask`](@ref) beside a
supplied dense one. Both arise in ordinary use — a time series advances one image at a time, and a
caller masking one image explicitly leaves the other to the default.
"""
struct ImagePair{T<:ImageElement,A<:AbstractMatrix{T},B<:AbstractMatrix{T},
                 Mr<:AbstractMatrix{Bool},Ms<:AbstractMatrix{Bool}}
    reference::A
    secondary::B
    reference_valid::Mr
    secondary_valid::Ms

    function ImagePair(reference::A, secondary::B, reference_valid::Mr,
                       secondary_valid::Ms) where {T<:ImageElement,A<:AbstractMatrix{T},
                                                   B<:AbstractMatrix{T},
                                                   Mr<:AbstractMatrix{Bool},
                                                   Ms<:AbstractMatrix{Bool}}
        size(reference) == size(secondary) || throw(DimensionMismatch(
            "reference is $(size(reference)) but secondary is $(size(secondary)); " *
            "the two images must be co-registered to a common grid before " *
            "correlation"))
        size(reference_valid) == size(reference) || throw(DimensionMismatch(
            "reference_valid is $(size(reference_valid)) but reference is " *
            "$(size(reference))"))
        size(secondary_valid) == size(secondary) || throw(DimensionMismatch(
            "secondary_valid is $(size(secondary_valid)) but secondary is " *
            "$(size(secondary))"))
        return new{T,A,B,Mr,Ms}(reference, secondary, reference_valid, secondary_valid)
    end
end

"""
    ImagePair(reference, secondary; reference_valid = nothing, secondary_valid = nothing)

Build an [`ImagePair`](@ref), deriving validity masks from finiteness when not given.

A pixel is treated as invalid if it is not finite. Zero is deliberately *not* treated
as invalid by default: it is a legitimate radiance value, and the reference's habit of
conflating "zero" with "no data" misclassifies genuinely dark pixels. Pass an explicit
mask when the sensor uses a fill value.

The derived mask is an [`AutoRIFT.FiniteMask`](@ref), which computes `isfinite` on read. That keeps
this constructor from touching the imagery at all, so it costs nothing for an input still on disk
and forms no scene-sized array — the property `process_block_size` depends on.
"""
function ImagePair(reference::AbstractMatrix, secondary::AbstractMatrix;
                   reference_valid = nothing, secondary_valid = nothing)
    rv = isnothing(reference_valid) ? FiniteMask(reference) : validmask(reference_valid)
    sv = isnothing(secondary_valid) ? FiniteMask(secondary) : validmask(secondary_valid)
    return ImagePair(reference, secondary, rv, sv)
end

# A supplied mask, as an `AbstractMatrix{Bool}` and without copying.
#
# Passed through as given rather than converted to `Matrix{Bool}`: a lazy `Raster` mask converted
# here would be materialized at construction, which is the scene-sized allocation blocking exists to
# avoid. `resident` is where materialization happens instead, at the one consumer that needs it.
#
# Not copying is also what lets a `Cache` decide by identity whether an image it holds is already
# prepared — a defensive copy would silently turn every reuse into a re-filter.
validmask(m::AbstractMatrix{Bool}) = m
validmask(m::AbstractMatrix) = Matrix{Bool}(m)

Base.size(p::ImagePair) = size(p.reference)
Base.eltype(::ImagePair{T}) where {T} = T

"""
    valid(pair::ImagePair) -> AbstractMatrix{Bool}

Pixels valid in **both** images.

Correlation needs data on both sides, so this is the intersection. It is also what
determines which grid points are searched at all, which makes it consequential well
beyond the filtering: a mask that is wrong here produces a scene-wide error that no
later stage can detect.
"""
valid(p::ImagePair) = p.reference_valid .& p.secondary_valid

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

"""
    preprocess(pair::ImagePair, method) -> ImagePair

Apply a pre-correlation filter to both images, returning a new pair with an updated
validity mask.

`method` is a [`PreprocessMethod`](@ref) or the `Symbol` naming one. The result is
`Float32` regardless of input type, since every filter produces signed values.

The validity mask can only shrink: a filter with a `w`-wide window spreads each
invalid pixel over a `w`-wide neighbourhood, because an output computed partly from
fill values is not a measurement of anything. The reference does not do this
consistently, which lets fill values leak into the correlation as texture.
"""
preprocess(pair::ImagePair, method::Symbol) = preprocess(pair, _preprocess(method, nokw))

# One image at a time is the primitive, and the pair is two calls. Every filter here is
# per-image — nothing about filtering the reference depends on the secondary — and saying so
# is what lets a `Cache` re-filter only the image that changed when a time series advances.
preprocess(pair::ImagePair, m::PreprocessMethod) = ImagePair(
    preprocess(pair.reference, pair.reference_valid, m),
    preprocess(pair.secondary, pair.secondary_valid, m))

"""
    preprocess(img, mask, method) -> (filtered, mask)

Filter a single image, returning it with its shrunken validity mask.

The pair form is this applied twice. Exposed separately because a filtered image is reusable:
in a time series each acquisition is the secondary of one pair and the reference of the next,
so filtering per image rather than per pair halves the work.
"""
# No filter means no conversion: the image passes through with its own element type, so `Int16`
# imagery stays 2 bytes per pixel rather than becoming 4. The correlator handles any `T<:Real`,
# and this stage is memory-bandwidth-bound, so widening for no numerical gain would be the
# expensive kind of harmless. A copy is still made, because the caller's array must not be
# aliased by a pipeline that may write to it.
preprocess(img::AbstractMatrix, mask::AbstractMatrix{Bool}, ::NoPreprocess) =
    (copy(img), copy(mask))

# `{<:Real}` rather than a bare `AbstractMatrix`: these are amplitude filters, and saying so in the
# signature is what makes the complex rejection below unambiguous rather than a tie the compiler
# has to be told how to break. Aqua's ambiguity check catches the alternative.
preprocess(img::AbstractMatrix{<:Real}, mask::AbstractMatrix{Bool}, m::Highpass) =
    _filtered(highpass(img, mask, m.width), mask, m.width)

preprocess(img::AbstractMatrix{<:Real}, mask::AbstractMatrix{Bool}, m::Wallis) =
    _filtered(wallis(img, mask, m.width, m.min_std), mask, m.width)

preprocess(img::AbstractMatrix{<:Real}, mask::AbstractMatrix{Bool}, m::WallisGapfill) =
    preprocess(img, mask, m, UInt64(0))

"""
    preprocess(img, mask, method, seed::UInt64) -> (filtered, mask)

[`preprocess`](@ref) with the seed for the one filter that draws random numbers.

`seed` is positional and concretely typed rather than a keyword carrying an `AbstractRNG`, which
`--trim` cannot resolve. Only [`WallisGapfill`](@ref) reads it; every other method ignores it and
forwards to the three-argument form, so this is what `_prepare` calls without having to know which
filter it holds.
"""
preprocess(img::AbstractMatrix, mask::AbstractMatrix{Bool}, m::PreprocessMethod, ::UInt64) =
    preprocess(img, mask, m)

# A fresh `Xoshiro(seed)` per call rather than the global stream: the two images of a pair, and two
# runs of the same pair, must fill identically, and a shared generator makes the fill depend on call
# order. The reference draws from NumPy's unseeded global generator and so is not reproducible
# against itself.
preprocess(img::AbstractMatrix{<:Real}, mask::AbstractMatrix{Bool}, m::WallisGapfill,
           seed::UInt64) =
    wallis_gapfill(img, mask, m.width, m.min_std; rng = Random.Xoshiro(seed))

# Pointwise, so no window and no mask erosion: `_filtered` is not used and the mask passes
# through with only the pixels the log could not represent removed.
preprocess(img::AbstractMatrix{<:Real}, mask::AbstractMatrix{Bool}, ::Decibel) =
    decibel(img, mask)

preprocess(img::AbstractMatrix{<:Real}, mask::AbstractMatrix{Bool}, m::Sobel) =
    _filtered(sobel(img, mask, m.width), mask, m.width)

preprocess(img::AbstractMatrix{<:Real}, mask::AbstractMatrix{Bool}, m::Laplacian) =
    _filtered(laplacian(img, mask, m.width), mask, m.width)

# ---------------------------------------------------------------------------
# Wallis with gap filling
# ---------------------------------------------------------------------------

"""
    wallis_gapfill(img, mask, width, std_cutoff; rng) -> (Matrix{Float32}, Matrix{Bool})

[`wallis`](@ref), with interior no-data gaps and low-contrast patches replaced by white noise.

For Landsat-7 after the 2003 Scan Line Corrector failure, whose imagery carries wedge-shaped
interior gaps. Correlation cannot use a gap, but leaving one masked costs the whole
neighbourhood of every chip that overlaps it. Filling with noise matched to the surrounding
distribution is the cheaper trade: noise correlates with nothing, so it suppresses the peak
where a chip is mostly gap without discarding chips that merely touch one.

The noise is standard normal and **not** rescaled, which is correct because it is added after
normalization — `wallis` has already divided by the local standard deviation, so valid pixels
are themselves about unit variance.

Three regions are distinguished, and only the middle one is filled:

  * **Measured** — enough valid neighbours and enough contrast. Normalized as usual.
  * **Filled** — an interior gap, or a patch whose local standard deviation is below
    `std_cutoff`, that lies within reach of real data. Replaced by noise and marked *valid*,
    since that is what stops it masking out its neighbours.
  * **Excluded** — beyond `30` pixels from any valid pixel, so there is no nearby data for the
    fill to be consistent with. Marked invalid, as an outer no-data border should be.

`std_cutoff` is an absolute threshold on the local standard deviation, so it is scaled to the
input's own units. Note it does *not* floor the divisor the way [`Wallis`](@ref)'s `min_std`
does; here a low-contrast window is filled rather than clamped.

Gaps are taken from `mask`, not from `img == 0`. The reference detects them as approximately
zero, which conflates a fill value with a genuinely dark pixel — the same conflation
[`ImagePair`](@ref) declines to make. A caller whose sensor writes zeros passes a mask that
says so.
"""
function wallis_gapfill(img::AbstractMatrix{<:Real}, mask::AbstractMatrix{Bool},
                        width::Integer, std_cutoff::Real;
                        rng::Random.AbstractRNG = Random.default_rng())
    axes(img) == axes(mask) || throw(DimensionMismatch(
        "image and mask must share axes: $(axes(img)) vs $(axes(mask))"))
    Base.require_one_based_indexing(img, mask)
    w = Int(width)

    buff = _gapfill_buffer(w)

    # An interior gap is an invalid pixel with real data within `GAPFILL_REACH`. The outer
    # border of a scene fails that test, which is what keeps a whole invalid margin from being
    # filled with noise that has nothing to be consistent with.
    near_data = dilate_within(mask, GAPFILL_REACH)
    gaps = dilate_within(near_data .& .!mask, buff)

    # Where the output means anything at all: real data, plus the grown interior gaps.
    domain = mask .| gaps

    mean = _masked_boxmean(img, mask, w)
    sd = _masked_boxstd(img, mask, mean, w)

    # `!(sd >= cutoff)` rather than `sd < cutoff` so a `NaN` standard deviation — a window with
    # no valid neighbour — counts as low contrast rather than as passing the test.
    cutoff = Float32(std_cutoff)
    flat = similar(mask, Bool)
    @inbounds for i in eachindex(flat, sd)
        flat[i] = !(sd[i] >= cutoff)
    end

    # Fill the gaps and the flat patches, but only inside the domain.
    fill = (gaps .| dilate_within(flat, buff)) .& domain

    out = Matrix{Float32}(undef, size(img))
    v = Matrix{Bool}(undef, size(mask))
    @inbounds for i in eachindex(out, img, mask)
        if !domain[i]
            out[i] = 0.0f0
            v[i] = false
        elseif fill[i]
            # Unit-variance noise, matching what the normalization leaves behind elsewhere.
            out[i] = Float32(randn(rng))
            v[i] = true
        else
            # `mask[i]` holds here without testing it: this branch needs `domain[i] && !fill[i]`,
            # and `gaps ⊆ fill ⊆ domain`, so `!fill` implies `!gaps`, and `(mask | gaps) && !gaps`
            # is `mask`. A non-finite local statistic is still possible, though, and lands below.
            m, s = mean[i], sd[i]
            if isfinite(m) && isfinite(s) && s > 0
                out[i] = (Float32(img[i]) - m) / s
                v[i] = true
            else
                out[i] = 0.0f0
                v[i] = false
            end
        end
    end
    return out, v
end

# ---------------------------------------------------------------------------
# Amplitude-in-decibels, and the derivative filters built on it
# ---------------------------------------------------------------------------

"""
    decibel(img, mask) -> (Matrix{Float32}, Matrix{Bool})

Amplitude to decibels, `20 log10(A)`, with the mask narrowed to the pixels that have one.

`20 log10` and not `10 log10`: the input is amplitude, not power. Intended for radar, where
brightness varies multiplicatively over a large range and a log turns that into an additive
offset the windowed filters can remove.

A non-positive amplitude has no decibel value. Those pixels are dropped from the mask and
written as zero rather than left as `-Inf` or `NaN`, which is where this differs from the
reference: it records a zero mask and then takes the log anyway, so `-Inf` reaches the
correlator as texture.
"""
function decibel(img::AbstractMatrix{<:Real}, mask::AbstractMatrix{Bool})
    axes(img) == axes(mask) || throw(DimensionMismatch(
        "image and mask must share axes: $(axes(img)) vs $(axes(mask))"))
    out = Matrix{Float32}(undef, size(img))
    v = Matrix{Bool}(undef, size(mask))
    @inbounds for i in eachindex(out, img, mask)
        a = Float32(img[i])
        # `a > 0` excludes the fill value the reference conflates with darkness, and also
        # excludes a negative sample, which amplitude data should not contain at all.
        if mask[i] && a > 0
            out[i] = 20.0f0 * log10(a)
            v[i] = true
        else
            out[i] = 0.0f0
            v[i] = false
        end
    end
    return out, v
end

"""
    sobel(img, mask, width) -> Matrix{Float32}

Sum of the x and y Sobel derivative kernels of side `width`, applied as one kernel.

This is `Gx + Gy` — the two derivative kernels added together and convolved once — not the
gradient magnitude `sqrt(Gx² + Gy²)` and not `|Gx| + |Gy|`. It is what the reference does, and
it makes the filter **directional**: the response partially cancels along one diagonal and
reinforces along the other. Worth knowing before reading the output as an edge strength.

Kernels are the unnormalized separable binomial-difference pair OpenCV's `getDerivKernels`
returns, so at `width = 5` the x kernel is `[-1,-2,0,2,1] ⊗ [1,4,6,4,1]`.
"""
sobel(img::AbstractMatrix{<:Real}, mask::AbstractMatrix{Bool}, width::Integer) =
    _convolve_masked(img, mask, _summed_deriv_kernel(Int(width), 1))

"""
    laplacian(img, mask, width) -> Matrix{Float32}

Laplacian of the **decibel** image: `∂²/∂x² + ∂²/∂y²` applied to `20 log10(A)`.

The log comes first, and that ordering is the point. Radar brightness varies
multiplicatively, so a second derivative of raw amplitude scales with local brightness and
reports the same terrain twice as strongly where the scene is twice as bright. In decibels
that variation is an additive offset, which a second derivative removes outright.

Second-order kernels of side `width`, summed across the two axes, so this is one convolution
rather than two.
"""
function laplacian(img::AbstractMatrix{<:Real}, mask::AbstractMatrix{Bool}, width::Integer)
    db, dbmask = decibel(img, mask)
    return _convolve_masked(db, dbmask, _summed_deriv_kernel(Int(width), 2))
end

# The `order`-th derivative along x plus the same along y, as one `w`-by-`w` kernel.
#
# Summing the two separable outer products rather than convolving twice and adding is what the
# reference does, and it is also why one convolution suffices. For `order = 2` this is exactly
# OpenCV's `Laplacian(ksize = w)`; for `order = 1` it is the reference's `Sobel`, which is
# directional rather than a gradient magnitude — see [`sobel`](@ref).
function _summed_deriv_kernel(w::Int, order::Int)
    deriv, smooth = _deriv_kernels(w, order)
    return _outer(deriv, smooth) .+ _outer(smooth, deriv)
end

# The separable derivative kernels OpenCV's `getDerivKernels(order, 0, w, normalize=false)`
# returns: an `order`-th central difference widened by binomial smoothing, and the binomial
# smoother for the perpendicular axis.
#
# Built from Pascal's triangle rather than tabulated, and pinned against OpenCV's own output by
# `test/preprocess.jl`. Both factors are a binomial row: the smoother is row `w-1`, and the
# derivative is the three-tap difference convolved with row `w-3`. That widening is why width 5
# gives `[-1,-2,0,2,1]` rather than a bare `[-1,0,1]`.
function _deriv_kernels(w::Int, order::Int)
    (isodd(w) && w >= 3) || throw(ArgumentError(
        "derivative kernel width must be odd and at least 3, got $w"))
    order in (1, 2) || throw(ArgumentError("derivative order must be 1 or 2, got $order"))
    smooth = _binomial_row(w)
    base = order == 1 ? Float32[-1, 0, 1] : Float32[1, -2, 1]
    return _conv1(base, _binomial_row(w - 2)), smooth
end

# Row `n-1` of Pascal's triangle, as a length-`n` vector: the binomial smoother of width `n`.
_binomial_row(n::Int) = [Float32(binomial(n - 1, k)) for k in 0:(n - 1)]

# Full 1-D convolution, for composing the small kernel factors above.
function _conv1(a::AbstractVector{Float32}, b::AbstractVector{Float32})
    out = zeros(Float32, length(a) + length(b) - 1)
    for (i, ai) in pairs(a), (j, bj) in pairs(b)
        out[i + j - 1] += ai * bj
    end
    return out
end

_outer(col::Vector{Float32}, row::Vector{Float32}) = col .* transpose(row)

# Correlate `img` with `kernel`, excluding invalid pixels from every window rather than letting
# them contribute zeros.
#
# That exclusion is the difference from the reference, which zero-pads: a no-data border there
# biases every output within half a kernel of it, and the bias looks like an edge — exactly what
# a derivative filter is meant to detect. Here a window with no valid pixel is `NaN`, which
# `_filtered` then records in the mask.
#
# Renormalizing by the weight actually used would be wrong for a derivative kernel, whose
# coefficients sum to zero: there is no mean to preserve, so the valid coefficients are applied
# as they stand.
function _convolve_masked(img::AbstractMatrix{<:Real}, mask::AbstractMatrix{Bool},
                          kernel::AbstractMatrix{Float32})
    axes(img) == axes(mask) || throw(DimensionMismatch(
        "image and mask must share axes: $(axes(img)) vs $(axes(mask))"))
    # The other filters in this file return a 1-based `Matrix{Float32}`, and `_filtered` requires
    # one, so declare the assumption rather than appearing generic and producing an offset result.
    Base.require_one_based_indexing(img, mask)
    kh, kw = size(kernel)
    # Left-biased centre, matching the window conventions in `window.jl` so an even width — which
    # the constructors reject, but which a direct call could still reach — lands consistently.
    ci, cj = (kh ÷ 2) + 1, (kw ÷ 2) + 1
    nr, nc = size(img)
    out = Matrix{Float32}(undef, nr, nc)

    # The interior, where every tap of the window is in bounds, so the per-tap range test the edge
    # loop needs cannot fail. Splitting it out is worth a measurement rather than assumed: the taps
    # are cheap enough that the two range checks per tap dominated, and hoisting them out of the
    # interior — which is all but a `kh`-wide frame of the image — cut the filter by a third.
    #
    # `all(mask)` decides the mask test the same way, and for the same reason `_masked_boxmean!` and
    # `_erode_mask!` both check it: a gap-free image is the common case, and on one the test is pure
    # overhead. Both loops are otherwise identical, which is why the body is a macro-free function
    # of the mask predicate rather than two hand-copied loops.
    dense = all(mask)
    ilo, ihi = ci, nr - (kh - ci)
    jlo, jhi = cj, nc - (kw - cj)
    @inbounds for j in 1:nc, i in 1:nr
        interior = ilo <= i <= ihi && jlo <= j <= jhi
        acc = 0.0f0
        any_valid = false
        if interior && dense
            # Nothing to test: every tap is in bounds and every pixel is valid.
            for kj in 1:kw
                jj = j + (kj - cj)
                for ki in 1:kh
                    acc += kernel[ki, kj] * Float32(img[i + (ki - ci), jj])
                end
            end
            any_valid = true
        elseif interior
            for kj in 1:kw
                jj = j + (kj - cj)
                for ki in 1:kh
                    ii = i + (ki - ci)
                    mask[ii, jj] || continue
                    acc += kernel[ki, kj] * Float32(img[ii, jj])
                    any_valid = true
                end
            end
        else
            for kj in 1:kw
                jj = j + (kj - cj)
                1 <= jj <= nc || continue
                for ki in 1:kh
                    ii = i + (ki - ci)
                    1 <= ii <= nr || continue
                    mask[ii, jj] || continue
                    acc += kernel[ki, kj] * Float32(img[ii, jj])
                    any_valid = true
                end
            end
        end
        out[i, j] = any_valid ? acc : NaN32
    end
    return out
end

# ---------------------------------------------------------------------------
# Complex input
# ---------------------------------------------------------------------------

"""
    deramp(img::AbstractMatrix{<:Complex}, mask, axis = :both) -> Matrix{ComplexF32}

Remove the linear phase ramp from a complex image. See [`Deramp`](@ref) for why.

The ramp is estimated from the image itself: for each axis, sum the conjugate product of adjacent
pixels and take the argument of the sum. Following ISCE2's `cuAmpcor` (`cuDeramp.cu`, method 1).

Two properties follow from summing the product rather than averaging per-pixel phase differences,
and both matter for speckle:

  * **Amplitude weighting is free.** `conj(z_i) · z_{i+1}` carries `|z_i||z_{i+1}|`, so bright
    pixel pairs dominate — which is what you want when dim pixels are mostly noise.
  * **No phase wrapping.** The `atan2` is taken once, after the sum, so a per-pixel difference
    near ±π cannot alias. Averaging angles would need unwrapping first.

Invalid pixels are excluded from the estimate but still deramped, so the output stays a complete
image and the mask keeps its own record of what is real.
"""
function deramp(img::AbstractMatrix{<:Complex}, mask::AbstractMatrix{Bool},
                axis::Symbol = :both)
    nr, nc = size(img)

    # Accumulate in ComplexF64: this is a sum over the whole image, so it is exactly the long
    # accumulation chain that `integral.jl` argues for double precision on.
    phase_y = 0.0
    if axis !== :x
        acc = zero(ComplexF64)
        @inbounds for j in 1:nc, i in 1:(nr - 1)
            # Both pixels must be real data for their phase difference to mean anything.
            (mask[i, j] && mask[i + 1, j]) || continue
            acc += conj(ComplexF64(img[i, j])) * ComplexF64(img[i + 1, j])
        end
        phase_y = atan(imag(acc), real(acc))
    end

    phase_x = 0.0
    if axis !== :y
        acc = zero(ComplexF64)
        @inbounds for j in 1:(nc - 1), i in 1:nr
            (mask[i, j] && mask[i, j + 1]) || continue
            acc += conj(ComplexF64(img[i, j])) * ComplexF64(img[i, j + 1])
        end
        phase_x = atan(imag(acc), real(acc))
    end

    # Zero-based offsets, so the ramp is removed *about the first sample* rather than about the
    # origin of a 1-based index. Either is a valid deramp — they differ by a constant phase, which
    # coherence is invariant to after the complex mean is removed — but matching cuAmpcor's
    # convention keeps the recovered coefficients directly comparable to it.
    out = Matrix{ComplexF32}(undef, nr, nc)
    @inbounds for j in 1:nc, i in 1:nr
        out[i, j] = ComplexF32(ComplexF64(img[i, j]) *
                               cis(-((i - 1) * phase_y + (j - 1) * phase_x)))
    end
    return out
end

"""
    ramp_phase(img::AbstractMatrix{<:Complex}, mask) -> (phase_x, phase_y)

The per-sample linear phase gradient [`deramp`](@ref) would remove, in radians per pixel.

Exposed because it is the testable part: a synthetic image built with a known ramp must return
that ramp, which is how the estimator is verified without a reference implementation to compare
against.
"""
function ramp_phase(img::AbstractMatrix{<:Complex}, mask::AbstractMatrix{Bool})
    nr, nc = size(img)
    accy = zero(ComplexF64)
    @inbounds for j in 1:nc, i in 1:(nr - 1)
        (mask[i, j] && mask[i + 1, j]) || continue
        accy += conj(ComplexF64(img[i, j])) * ComplexF64(img[i + 1, j])
    end
    accx = zero(ComplexF64)
    @inbounds for j in 1:(nc - 1), i in 1:nr
        (mask[i, j] && mask[i, j + 1]) || continue
        accx += conj(ComplexF64(img[i, j])) * ComplexF64(img[i, j + 1])
    end
    return atan(imag(accx), real(accx)), atan(imag(accy), real(accy))
end

preprocess(img::AbstractMatrix{<:Complex}, mask::AbstractMatrix{Bool}, m::Deramp) =
    (deramp(img, mask, m.axis), copy(mask))

# Deramping a real image is a no-op that would look like it did something, so it is an error
# naming the mismatch instead.
preprocess(::AbstractMatrix{<:Real}, ::AbstractMatrix{Bool}, ::Deramp) = throw(ArgumentError(
    "`Deramp` needs complex input, but the image is real. A real image has no phase ramp to " *
    "remove. Use `preprocess = :highpass` for real imagery, or pass the complex data."))

# The windowed filters are amplitude operations, and applying one to complex data is very
# unlikely to be what a caller meant: `Highpass` would subtract a local *complex* mean, mixing
# amplitude and phase structure into each other, and `Wallis` would divide by a standard
# deviation computed across that mixture. Rather than silently produce something defensible-looking
# but meaningless, say so and name the two things a caller probably wants.
#
# `Union` in one method rather than one per filter, because the message is the same and the list
# of amplitude filters is the thing being described.
function preprocess(::AbstractMatrix{<:Complex}, ::AbstractMatrix{Bool},
                    m::Union{Highpass,Wallis,WallisGapfill,Sobel,Laplacian,Decibel})
    throw(ArgumentError(
        "`$(nameof(typeof(m)))` is an amplitude filter and the image is complex. Applying it " *
        "would mix amplitude and phase structure into each other rather than filtering either. " *
        "For complex input use `preprocess = :deramp` (see `Coherence`), or take `abs` of the " *
        "images first and correlate the amplitudes."))
end

# The mask bookkeeping every windowed filter needs, written once. Doing it per filter is how
# the reference ended up inconsistent about it.
function _filtered(out::AbstractMatrix{Float32}, mask::AbstractMatrix{Bool}, width::Integer)
    v = _erode_mask(mask, Int(width))
    return _filtered_finish!(out, v)
end

"""
    _filtered!(out, v, mask, width) -> (out, v)

[`_filtered`](@ref)'s work, writing the eroded mask into `v` rather than allocating it.

For tiled processing, where a fresh mask per block is churn `Sys.maxrss` reports as a requirement.
`out` is modified in place, as in the allocating form.
"""
function _filtered!(out::AbstractMatrix{Float32}, v::AbstractMatrix{Bool},
                    mask::AbstractMatrix{Bool}, width::Integer)
    _erode_mask!(v, mask, Int(width))
    return _filtered_finish!(out, v)
end

"""
    AutoRIFT._finishes_nonfinite(method) -> Bool

Whether `method`'s own `preprocess` already zeroed every non-finite output and cleared those pixels
in the mask.

Every filter that ends in [`AutoRIFT._filtered`](@ref) does, because
[`AutoRIFT._filtered_finish!`](@ref) is exactly that pass — so running
[`replace_nonfinite`](@ref) over the result afterwards would allocate two full-size arrays to
recompute a result it already has. On a 17121×16961 scene that is 1.6 GiB per image.

Declared per method rather than inferred, and defaulting to `false`: a new filter that forgets to
declare it pays one redundant pass, where a default of `true` would silently pass non-finite values
into the correlator. [`WallisGapfill`](@ref) is the method that genuinely needs the pass — its random
fill can leave non-finite values behind — and [`Decibel`](@ref) does its own mask bookkeeping without
`_filtered`, so neither declares it.
"""
_finishes_nonfinite(::PreprocessMethod) = false
_finishes_nonfinite(::Union{Highpass,Wallis,Sobel,Laplacian}) = true

# Shared tail: a filter can produce a non-finite value from finite input, so finiteness of the
# output is part of validity too.
function _filtered_finish!(out::AbstractMatrix{Float32}, v::AbstractMatrix{Bool})
    @inbounds for i in eachindex(out)
        # The value becomes zero to keep downstream arithmetic finite; the mask is what records
        # that it carries no information.
        if !isfinite(out[i])
            out[i] = 0.0f0
            v[i] = false
        end
    end
    return out, v
end

# `ImagePair` from two `(image, mask)` tuples, which is what the per-image path returns.
ImagePair((r, rv)::Tuple{AbstractMatrix,AbstractMatrix{Bool}},
          (s, sv)::Tuple{AbstractMatrix,AbstractMatrix{Bool}}) = ImagePair(r, s, rv, sv)

"""
    highpass(img, mask, width) -> Matrix{Float32}

Subtract a local mean of side `width`: the identity minus a box filter.

The cheapest way to remove an illumination gradient, and the only filter the production
driver applies inside the correlator. Invalid pixels are excluded from the local mean
rather than contributing zeros to it, which is where this differs from the reference —
its zero-padded convolution lets a no-data border bias the mean of every pixel near it.
"""
highpass(img::AbstractMatrix, mask::AbstractMatrix{Bool}, width::Integer) =
    highpass!(Matrix{Float32}(undef, size(img)), img, mask, width)

"""
    highpass!(out, img, mask, width; scratch = nothing) -> out

[`highpass`](@ref) writing into `out`, which must have `img`'s shape.

Exists for tiled processing, where the alternative is a fresh output and scratch array per block —
churn that `Sys.maxrss` records as a requirement even though the collector frees it. `scratch` is a
second `Float32` array of the same shape, needed only when `mask` excludes something; pass one to
make the call allocation-free in that case too.
"""
function highpass!(out::AbstractMatrix{Float32}, img::AbstractMatrix,
                   mask::AbstractMatrix{Bool}, width::Integer;
                   scratch::Union{Nothing,AbstractMatrix{Float32}} = nothing)
    axes(out) == axes(img) == axes(mask) || throw(DimensionMismatch(
        "out, img and mask must share axes"))
    # In place over the local mean: each output reads only the mean at its own index, and the
    # mean is not wanted afterwards. Saves a full-size temporary on an image-sized array.
    _masked_boxmean!(out, img, mask, Int(width), scratch)
    @inbounds for i in eachindex(out)
        out[i] = mask[i] && isfinite(out[i]) ? Float32(img[i]) - out[i] : NaN32
    end
    return out
end

"""
    wallis(img, mask, width, min_std = 0.0) -> Matrix{Float32}

Wallis (1976) adaptive contrast filter: subtract the local mean and divide by the local
standard deviation over a `width`-wide window.

Equalises contrast across the scene, which matters where illumination varies sharply —
a shadowed slope and a sunlit one become comparable. The cost is amplified noise in
genuinely textureless areas, since dividing by a small standard deviation magnifies
whatever is there; `min_std` floors the divisor to limit that, and `0.0` disables the
floor.

The local variance is computed in a numerically stable form rather than as
`E[x²] - E[x]²`. That difference-of-large-numbers formula cancels catastrophically for
a bright, low-contrast window, and in the reference it produced negative variances whose
square roots were `NaN` — which then propagated into the validity mask, silently
discarding data. Upstream now clamps the variance at zero, which stops the `NaN`s but
leaves the precision loss; computing the variance about the measured mean avoids both.
"""
function wallis(img::AbstractMatrix, mask::AbstractMatrix{Bool}, width::Integer,
                min_std::Real = 0.0)
    w = Int(width)
    mean = _masked_boxmean(img, mask, w)
    sd = _masked_boxstd(img, mask, mean, w)
    floor_sd = Float32(min_std)
    out = Matrix{Float32}(undef, size(img))
    @inbounds for i in eachindex(out)
        if !mask[i] || !isfinite(mean[i]) || !isfinite(sd[i])
            out[i] = NaN32
            continue
        end
        s = max(sd[i], floor_sd)
        # A window with no contrast at all carries no texture to normalise, so the
        # result is undefined rather than infinite.
        out[i] = s > 0 ? (Float32(img[i]) - mean[i]) / s : NaN32
    end
    return out
end

# Box mean over valid pixels only, via separable running sums with a parallel count.
# The count is what makes it mask-aware: dividing by the window area would be wrong
# wherever any neighbour is invalid, which near a no-data border is everywhere.
#
# Remaining performance gap, measured rather than assumed: ~6x slower than cv2's
# `filter2D` at width 5 (8.1 ms against 1.3 ms on 1024²) and ~1.9x *faster* at width 21,
# because the cost here is flat in the window width and cv2's is not. Down from ~8x:
# `windowmean` now takes a count-free path when nothing is missing, which is the common
# case for whole-image filtering and is bit-identical since the sums stay Float64.
#
# Alternatives tried and rejected: transposing the intermediate to make both passes
# contiguous (5.5 ms, but two extra full-size copies), vectorising the recurrence across
# columns (13 ms — worse, the count-array traffic dominates), and Float32 accumulation
# (1.37x but changed results by 2e-6, since a running sum drifts across a whole row).
#
# A bandwidth estimate puts the floor at 0.4 ms, so what remains is per-element scalar
# work that cv2 vectorises. Closing it means cache tiling, an M8 item. Filtering runs
# once per image rather than once per grid point, so this is not on the hot path.
_masked_boxmean(img::AbstractMatrix, mask::AbstractMatrix{Bool}, w::Int) =
    _masked_boxmean!(Matrix{Float32}(undef, size(img)), img, mask, w, nothing)

# `scratch`, when given, is the NaN-encoded copy this would otherwise allocate. Supplying it is what
# makes a per-block call allocation-free; `nothing` allocates as before.
function _masked_boxmean!(out::AbstractMatrix{Float32}, img::AbstractMatrix,
                          mask::AbstractMatrix{Bool}, w::Int,
                          scratch::Union{Nothing,AbstractMatrix{Float32}})
    # The copy exists only to encode invalidity as NaN, which is what makes the running sum
    # skip those pixels. With nothing masked there is nothing to encode, so read `img`
    # directly — `windowmean!` is generic in its input. On a gap-free image, which is the
    # common case for whole-image filtering, that removes a full-size temporary and a pass.
    #
    # `hasnan` is left to `windowmean!` to determine rather than asserted false: a fully valid
    # mask says every pixel is *marked* usable, not that none of them is NaN, and claiming
    # otherwise would silently take the dense path over an image that needs the masked one.
    all(mask) && return windowmean!(out, img, w)

    masked = isnothing(scratch) ? Matrix{Float32}(undef, size(img)) : scratch
    axes(masked) == axes(img) || throw(DimensionMismatch(
        "scratch must share axes with the image"))
    gaps = false
    @inbounds for i in eachindex(masked)
        if mask[i]
            v = Float32(img[i])
            masked[i] = v
            gaps |= isnan(v)
        else
            masked[i] = NaN32
            gaps = true
        end
    end
    # Tracked during the copy rather than rescanned afterwards: this loop already knows
    # whether it wrote a NaN, and `windowmean` would otherwise spend a full extra pass
    # rediscovering it.
    return windowmean!(out, masked, w; hasnan = gaps)
end

# Standard deviation about the local mean, computed from squared deviations rather than
# from `E[x²] - E[x]²`.
#
# The textbook shortcut is a difference of two large, nearly equal quantities, and it
# cancels catastrophically for a bright low-contrast window — which in the reference
# produced negative variances whose square roots were NaN, and those NaNs propagated
# into the validity mask and silently discarded data. Upstream now clamps the variance
# at zero, which stops the NaNs but leaves the precision loss. Squaring the deviations
# about the measured mean costs one more pass and avoids both.
#
# Bessel-corrected by the window *area*, `sqrt(w² / (w² - 1))`, which is a constant and so
# costs one multiply. It is deliberately not the per-window valid count: the reference scales by
# the kernel area regardless of how many neighbours were usable, and matching it is what keeps
# `WallisGapfill`'s `std_cutoff` — an absolute threshold on this quantity — mean the same thing
# here as there. At width 5 the factor is 1.0206, so omitting it would shift every `Wallis`
# output by 2% and move which pixels the gap filler treats as low-contrast.
function _masked_boxstd(img::AbstractMatrix, mask::AbstractMatrix{Bool},
                        mean::AbstractMatrix{Float32}, w::Int)
    dev2 = Matrix{Float32}(undef, size(img))
    gaps = false
    @inbounds for i in eachindex(dev2)
        if mask[i] && isfinite(mean[i])
            d = Float32(img[i]) - mean[i]
            v = d * d
            dev2[i] = v
            gaps |= isnan(v)
        else
            dev2[i] = NaN32
            gaps = true
        end
    end
    # `windowmean` ignores NaN, so this averages over valid neighbours only, and the
    # loop above already established whether there are any.
    msd = windowmean(dev2, w; hasnan = gaps)
    bessel = _bessel_factor(w)
    @inbounds for i in eachindex(msd)
        msd[i] = sqrt(max(msd[i], 0.0f0)) * bessel
    end
    return msd
end

# `sqrt(n / (n - 1))` for a `w`-by-`w` window. `w == 1` has no spread to correct, and the
# expression would divide by zero, so it is 1 there.
function _bessel_factor(w::Int)
    n = w * w
    return n > 1 ? Float32(sqrt(n / (n - 1))) : 1.0f0
end

# Erode a validity mask by a `w`-wide window: a pixel stays valid only if every pixel in
# its window is valid. The sliding minimum over a 0/1 encoding does exactly that, and is
# O(1) per pixel in the window width — which matters because the widths here reach 21.
#
# A whole-array shortcut first, because it is the common case and it is free to check: if
# nothing is masked, eroding changes nothing. That skips three full-size temporaries and
# two passes on every gap-free image, which is most of them.
_erode_mask(mask::AbstractMatrix{Bool}, w::Int) =
    _erode_mask!(Matrix{Bool}(undef, size(mask)), mask, w)

function _erode_mask!(out::AbstractMatrix{Bool}, mask::AbstractMatrix{Bool}, w::Int)
    if all(mask)
        copyto!(out, mask)
        return out
    end
    f = Matrix{Float32}(undef, size(mask))
    @inbounds for i in eachindex(f)
        f[i] = mask[i] ? 1.0f0 : 0.0f0
    end
    # No NaN by construction, so the sliding minimum takes its cheap path directly.
    lo = windowmin(f, w)
    @inbounds for i in eachindex(out)
        out[i] = lo[i] > 0.5f0
    end
    return out
end

# ---------------------------------------------------------------------------
# Non-finite replacement
# ---------------------------------------------------------------------------

"""
    replace_nonfinite(pair::ImagePair) -> ImagePair

Replace non-finite pixels with zero, leaving the element type alone.

The images reach the correlator in the caller's own type, so `Int16` sensor data is
correlated as `Int16`. The correlator is generic over its element type and widens per
element where it must — the chip to `Float32` because it is stored mean-removed, the sums
to `Float64` — so converting a whole image here would double its memory traffic and change
no result.
"""
replace_nonfinite(pair::ImagePair) = ImagePair(
    replace_nonfinite(pair.reference, pair.reference_valid),
    replace_nonfinite(pair.secondary, pair.secondary_valid))

"""
    replace_nonfinite(img, mask) -> (replaced, mask)

Per-image form, for the same reason [`preprocess`](@ref) has one.
"""
# An integer image cannot hold a non-finite value, so there is nothing to replace and the
# copy is the whole operation.
replace_nonfinite(img::AbstractMatrix{<:Integer}, mask::AbstractMatrix{Bool}) =
    (copy(img), copy(mask))

# Float and complex share one method: `isfinite` of a complex number is false if either
# component is, so the same test and the same replacement serve both. Zero keeps downstream
# arithmetic finite, and the mask is what records that those pixels carry no information.
function replace_nonfinite(img::AbstractMatrix{<:Union{AbstractFloat,Complex}},
                           mask::AbstractMatrix{Bool})
    out = similar(img)
    @inbounds for i in eachindex(out)
        v = img[i]
        out[i] = isfinite(v) ? v : zero(v)
    end
    return out, copy(mask)
end
