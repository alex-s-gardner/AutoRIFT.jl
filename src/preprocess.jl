# Pre-correlation filtering, and conversion to the correlation element type.
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
"""
struct ImagePair{T<:ImageElement,A<:AbstractMatrix{T},M<:AbstractMatrix{Bool}}
    reference::A
    secondary::A
    reference_valid::M
    secondary_valid::M

    function ImagePair(reference::A, secondary::A, reference_valid::M,
                       secondary_valid::M) where {T<:ImageElement,A<:AbstractMatrix{T},
                                                  M<:AbstractMatrix{Bool}}
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
        return new{T,A,M}(reference, secondary, reference_valid, secondary_valid)
    end
end

"""
    ImagePair(reference, secondary; reference_valid = nothing, secondary_valid = nothing)

Build an [`ImagePair`](@ref), deriving validity masks from finiteness when not given.

A pixel is treated as invalid if it is not finite. Zero is deliberately *not* treated
as invalid by default: it is a legitimate radiance value, and the reference's habit of
conflating "zero" with "no data" misclassifies genuinely dark pixels. Pass an explicit
mask when the sensor uses a fill value.
"""
function ImagePair(reference::AbstractMatrix, secondary::AbstractMatrix;
                   reference_valid = nothing, secondary_valid = nothing)
    rv = isnothing(reference_valid) ? map(isfinite, reference) : _boolmask(reference_valid)
    sv = isnothing(secondary_valid) ? map(isfinite, secondary) : _boolmask(secondary_valid)
    return ImagePair(reference, secondary, rv, sv)
end

# A mask as a `Matrix{Bool}`, without copying one that already is.
#
# `Matrix{Bool}` rather than any `AbstractMatrix{Bool}` because a `BitMatrix` packs eight
# entries per byte, which makes a concurrent write to one element a read-modify-write of the
# 64-bit word holding 63 of its neighbours — the same hazard that made `DisplacementField`
# store `searched` unpacked.
#
# Not copying when it need not is what lets a `Cache` decide by identity whether an image it
# holds is already prepared. A defensive copy would silently turn every reuse into a re-filter.
_boolmask(m::Matrix{Bool}) = m
_boolmask(m::AbstractMatrix) = Matrix{Bool}(m)

Base.size(p::ImagePair) = size(p.reference)
Base.eltype(::ImagePair{T}) where {T} = T

"""
    valid(pair::ImagePair) -> Matrix{Bool}

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
function _filtered(out::Matrix{Float32}, mask::AbstractMatrix{Bool}, width::Integer)
    v = _erode_mask(mask, Int(width))
    @inbounds for i in eachindex(out)
        # A filter can produce a non-finite value from finite input (a zero divisor in the
        # Wallis case), so finiteness of the output is part of validity too. The value itself
        # becomes zero to keep downstream arithmetic finite; the mask is what records that it
        # carries no information.
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
function highpass(img::AbstractMatrix, mask::AbstractMatrix{Bool}, width::Integer)
    # In place over the local mean: each output reads only the mean at its own index, and the
    # mean is not wanted afterwards. Saves a full-size temporary on an image-sized array.
    out = _masked_boxmean(img, mask, Int(width))
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
function _masked_boxmean(img::AbstractMatrix, mask::AbstractMatrix{Bool}, w::Int)
    out = Matrix{Float32}(undef, size(img))
    # The copy exists only to encode invalidity as NaN, which is what makes the running sum
    # skip those pixels. With nothing masked there is nothing to encode, so read `img`
    # directly — `windowmean!` is generic in its input. On a gap-free image, which is the
    # common case for whole-image filtering, that removes a full-size temporary and a pass.
    #
    # `hasnan` is left to `windowmean!` to determine rather than asserted false: a fully valid
    # mask says every pixel is *marked* usable, not that none of them is NaN, and claiming
    # otherwise would silently take the dense path over an image that needs the masked one.
    all(mask) && return windowmean!(out, img, w)

    masked = Matrix{Float32}(undef, size(img))
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
# Note this is the *population* standard deviation of the window, not the
# Bessel-corrected sample one. The correction would need the per-window valid count,
# and at the window sizes used here (5 to 21, so 25 to 441 samples) the difference is
# under 2% — well below the contrast variation the filter exists to remove, and not
# worth a second sliding-count pass to recover.
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
    @inbounds for i in eachindex(msd)
        msd[i] = sqrt(max(msd[i], 0.0f0))
    end
    return msd
end

# Erode a validity mask by a `w`-wide window: a pixel stays valid only if every pixel in
# its window is valid. The sliding minimum over a 0/1 encoding does exactly that, and is
# O(1) per pixel in the window width — which matters because the widths here reach 21.
#
# A whole-array shortcut first, because it is the common case and it is free to check: if
# nothing is masked, eroding changes nothing. That skips three full-size temporaries and
# two passes on every gap-free image, which is most of them.
function _erode_mask(mask::AbstractMatrix{Bool}, w::Int)
    all(mask) && return copy(mask)
    f = Matrix{Float32}(undef, size(mask))
    @inbounds for i in eachindex(f)
        f[i] = mask[i] ? 1.0f0 : 0.0f0
    end
    # No NaN by construction, so the sliding minimum takes its cheap path directly.
    lo = windowmin(f, w)
    out = Matrix{Bool}(undef, size(mask))
    @inbounds for i in eachindex(out)
        out[i] = lo[i] > 0.5f0
    end
    return out
end

# ---------------------------------------------------------------------------
# Conversion to the correlation element type
# ---------------------------------------------------------------------------

"""
    quantize(pair::ImagePair, method) -> ImagePair

Convert filtered images to the element type used for correlation.

`method` is a [`QuantizeMethod`](@ref) or the `Symbol` naming one.
"""
quantize(pair::ImagePair, method::Symbol) = quantize(pair, _quantize(method))

# Per image, for the same reason as `preprocess`: the UInt8 scaling is computed from one
# image's own statistics and nothing crosses between the two.
quantize(pair::ImagePair, m::QuantizeMethod) = ImagePair(
    quantize(pair.reference, pair.reference_valid, m),
    quantize(pair.secondary, pair.secondary_valid, m))

"""
    quantize(img, mask, method) -> (converted, mask)

Convert a single filtered image to the correlation element type, with its mask.

See [`preprocess`](@ref) for why the per-image form exists.
"""
# No quantization means the element type is the caller's, so `Int16` sensor data reaches the
# correlator as `Int16`. The correlator is generic over `T<:Real` and converts per element where
# it must — the chip to `Float32` because it is stored mean-removed, the sums to `Float64` — so
# widening the whole image here would double its memory traffic and change no result.
quantize(img::AbstractMatrix{<:Integer}, mask::AbstractMatrix{Bool}, ::NoQuantize) =
    (copy(img), copy(mask))

function quantize(img::AbstractMatrix{<:AbstractFloat}, mask::AbstractMatrix{Bool},
                  ::NoQuantize)
    # Only a float type can hold a non-finite value, so only this method needs to replace one.
    # They become zero so downstream arithmetic stays finite, and the mask records that they
    # carry no information. An integer image cannot be in that state, which is why the method
    # above can be a plain copy rather than this loop with a test that is always false.
    out = similar(img)
    @inbounds for i in eachindex(out)
        v = img[i]
        out[i] = isfinite(v) ? v : zero(v)
    end
    return out, copy(mask)
end

# Complex input under `NoQuantize`: the same non-finite replacement as the float method, since a
# complex value is non-finite if either component is.
function quantize(img::AbstractMatrix{<:Complex}, mask::AbstractMatrix{Bool}, ::NoQuantize)
    out = similar(img)
    @inbounds for i in eachindex(out)
        v = img[i]
        out[i] = isfinite(v) ? v : zero(v)
    end
    return out, copy(mask)
end

quantize(img::AbstractMatrix, mask::AbstractMatrix{Bool}, ::QuantizeUInt8) =
    (_to_uint8(img, mask), copy(mask))

# Quantizing complex data to UInt8 would have to discard the phase, which is the only reason to
# hold complex data in the first place. An error rather than a silent `abs`: a caller who wants
# amplitude matching should say so, and then the whole real pipeline applies unchanged.
quantize(::AbstractMatrix{<:Complex}, ::AbstractMatrix{Bool}, ::QuantizeUInt8) =
    throw(ArgumentError(
        "`quantize = :uint8` cannot represent complex input — scaling to 8-bit integers would " *
        "discard the phase, which is the only thing `Coherence` measures. Use " *
        "`quantize = :none` for complex data, or take `abs` of the images to correlate " *
        "amplitudes with the full real pipeline."))

"""
    _to_uint8(img, mask) -> Matrix{UInt8}

Rescale so that `mean ± 3 std` of the valid pixels spans the `UInt8` range.

Three standard deviations captures essentially all of a roughly normal distribution
while keeping the quantization step small; the tails are clipped, which costs nothing
because correlation depends on texture rather than on extremes.

Statistics are computed over valid pixels only, in `Float64`. Including fill values
would drag the mean and inflate the spread, compressing the real data into a fraction
of the available range and throwing away precision exactly where it is needed.
"""
function _to_uint8(img::AbstractMatrix, mask::AbstractMatrix{Bool})
    n = 0
    s = 0.0
    @inbounds for i in eachindex(img)
        v = Float64(img[i])
        if mask[i] && isfinite(v)
            n += 1
            s += v
        end
    end
    out = zeros(UInt8, size(img))
    n == 0 && return out            # nothing valid; all zero

    mean = s / n
    ss = 0.0
    @inbounds for i in eachindex(img)
        v = Float64(img[i])
        if mask[i] && isfinite(v)
            d = v - mean
            ss += d * d
        end
    end
    # Sample standard deviation (Bessel-corrected), matching the reference. Here the
    # correction is worth applying because the statistics are global rather than
    # per-window, so it costs nothing.
    sd = n > 1 ? sqrt(ss / (n - 1)) : 0.0
    if sd == 0
        # A uniform image has no texture to preserve; mid-grey keeps it neutral rather
        # than pinning it to an endpoint.
        @inbounds for i in eachindex(out)
            out[i] = mask[i] ? 0x80 : 0x00
        end
        return out
    end

    lo = mean - 3sd
    scale = 255.0 / (6sd)
    @inbounds for i in eachindex(out)
        v = Float64(img[i])
        if !mask[i] || !isfinite(v)
            out[i] = 0x00
            continue
        end
        out[i] = round(UInt8, clamp((v - lo) * scale, 0.0, 255.0))
    end
    return out
end
