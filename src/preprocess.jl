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
struct ImagePair{T<:Real,A<:AbstractMatrix{T},M<:AbstractMatrix{Bool}}
    reference::A
    secondary::A
    reference_valid::M
    secondary_valid::M

    function ImagePair(reference::A, secondary::A, reference_valid::M,
                       secondary_valid::M) where {T<:Real,A<:AbstractMatrix{T},
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
    rv = isnothing(reference_valid) ? map(isfinite, reference) : Matrix{Bool}(reference_valid)
    sv = isnothing(secondary_valid) ? map(isfinite, secondary) : Matrix{Bool}(secondary_valid)
    return ImagePair(reference, secondary, rv, sv)
end

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

function preprocess(pair::ImagePair, ::NoPreprocess)
    return ImagePair(Matrix{Float32}(pair.reference), Matrix{Float32}(pair.secondary),
                     copy(pair.reference_valid), copy(pair.secondary_valid))
end

function preprocess(pair::ImagePair, m::Highpass)
    w = m.width
    return _apply_both(pair, w) do img, mask
        highpass(img, mask, w)
    end
end

function preprocess(pair::ImagePair, m::Wallis)
    w = m.width
    return _apply_both(pair, w) do img, mask
        wallis(img, mask, w, m.min_std)
    end
end

# Applies `f` to each image, then shrinks both masks by the filter footprint. Written
# once because every windowed filter needs the same mask bookkeeping, and doing it per
# filter is how the reference ended up inconsistent.
function _apply_both(f, pair::ImagePair, w::Int)
    r = f(pair.reference, pair.reference_valid)
    s = f(pair.secondary, pair.secondary_valid)
    rv = _erode_mask(pair.reference_valid, w)
    sv = _erode_mask(pair.secondary_valid, w)
    # A filter can also produce a non-finite value from finite input (a zero divisor in
    # the Wallis case), so finiteness of the output is part of validity too.
    @inbounds for i in eachindex(rv)
        rv[i] &= isfinite(r[i])
        sv[i] &= isfinite(s[i])
    end
    # Non-finite values are replaced by zero so downstream arithmetic stays finite; the
    # mask is what records that they carry no information.
    @inbounds for i in eachindex(r)
        isfinite(r[i]) || (r[i] = 0.0f0)
        isfinite(s[i]) || (s[i] = 0.0f0)
    end
    return ImagePair(r, s, rv, sv)
end

"""
    highpass(img, mask, width) -> Matrix{Float32}

Subtract a local mean of side `width`: the identity minus a box filter.

The cheapest way to remove an illumination gradient, and the only filter the production
driver applies inside the correlator. Invalid pixels are excluded from the local mean
rather than contributing zeros to it, which is where this differs from the reference —
its zero-padded convolution lets a no-data border bias the mean of every pixel near it.
"""
function highpass(img::AbstractMatrix, mask::AbstractMatrix{Bool}, width::Integer)
    m = _masked_boxmean(img, mask, Int(width))
    out = Matrix{Float32}(undef, size(img))
    @inbounds for i in eachindex(out)
        out[i] = mask[i] && isfinite(m[i]) ? Float32(img[i]) - m[i] : NaN32
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
    return windowmean(masked, w; hasnan = gaps)
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

function quantize(pair::ImagePair, ::NoQuantize)
    # Non-finite values become zero and the mask records that they are not data.
    r = Matrix{Float32}(undef, size(pair))
    s = Matrix{Float32}(undef, size(pair))
    @inbounds for i in eachindex(r)
        vr = Float32(pair.reference[i])
        vs = Float32(pair.secondary[i])
        r[i] = isfinite(vr) ? vr : 0.0f0
        s[i] = isfinite(vs) ? vs : 0.0f0
    end
    return ImagePair(r, s, copy(pair.reference_valid), copy(pair.secondary_valid))
end

function quantize(pair::ImagePair, ::QuantizeUInt8)
    r = _to_uint8(pair.reference, pair.reference_valid)
    s = _to_uint8(pair.secondary, pair.secondary_valid)
    return ImagePair(r, s, copy(pair.reference_valid), copy(pair.secondary_valid))
end

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
