# The public API, and the cache that makes batch processing affordable.
#
# Two entry points. `autorift(reference, secondary; kwargs...)` is the one-shot form and is
# what most callers want. `init` / `reinit!` / `autorift!` is the batch form, and it exists
# because the one-shot form allocates and plans on every call — which across millions of image
# pairs is not a detail.
#
# ---------------------------------------------------------------------------
# Where the Symbols stop
# ---------------------------------------------------------------------------
#
# Every keyword is resolved to a concrete `Params` here, at the boundary, and nothing below
# this file sees a `Symbol` or re-validates a number. That is also why the work is handed to
# `_run` rather than done inline: the images may be `UInt8` or `Float32` depending on
# `quantize`, so the call crosses a type boundary, and putting a function barrier there keeps
# everything downstream monomorphic instead of leaving the whole pipeline inferring a union.

"""
    Cache

Preallocated state for correlating many image pairs with the same configuration.

Holds the resolved parameters, the search grid, the output arrays, and — indirectly, through
the FFT plan cache — the transform plans. Built by [`AutoRIFT.init`](@ref), advanced to the
next pair by [`reinit!`](@ref), and run by [`autorift!`](@ref).

Not thread-safe: one cache per task. Several tasks may each hold their own and run
concurrently, which is the intended shape for batch processing — see [`autorift!`](@ref).
"""
mutable struct Cache{P<:Params}
    params::P
    # The pair as supplied, before filtering or quantization. Kept because `reinit!` may replace
    # only one of the two images, and the other must then be re-prepared from its original —
    # preparing the already-prepared one would filter and quantize it a second time.
    raw::ImagePair
    # The pair the correlator sees: filtered and converted to the correlation element type.
    prepared::ImagePair
    grid::PointSet{2}
    result::Union{Nothing,MultichipResult}
    # Set when the images change and cleared once a run has consumed them. `reinit!`
    # accumulates dirtiness rather than overwriting it, so two swaps before one run still
    # leave the cache correctly marked.
    isfresh::Bool
end

"""
    AutoRIFT.imagepair(cache) -> ImagePair

The pair the correlator will see: filtered and converted to the correlation element type.
"""
imagepair(cache::Cache) = cache.prepared

"""
    AutoRIFT.init(reference, secondary; kwargs...) -> Cache

Prepare to correlate `reference` against `secondary`, allocating buffers and planning
transforms once.

Accepts the same keywords as [`autorift`](@ref). The images are preprocessed and quantized
immediately, since that is configuration-dependent work that does not need repeating per run.

Use this with [`reinit!`](@ref) when processing many pairs: the grid, the output arrays, and
the FFT plans are then built once rather than per pair. For a single pair, call
[`autorift`](@ref) instead.
"""
function CommonSolve.init(reference::AbstractMatrix, secondary::AbstractMatrix;
              reference_valid = nothing, secondary_valid = nothing, kwargs...)
    p = params(; kwargs...)
    raw = ImagePair(reference, secondary; reference_valid, secondary_valid)
    prepared = _prepare(raw, p)
    grid = _build_grid(size(prepared), p)
    # Plans are created here rather than on first use so that a batch driver pays for them
    # once, on the task that builds the cache, and never inside a correlation.
    _warm_grid_plans(grid, p)
    return Cache{typeof(p)}(p, raw, prepared, grid, nothing, true)
end

"""
    reinit!(cache; reference, secondary, reference_valid, secondary_valid) -> cache

Point `cache` at a new image pair, reusing its grid, output arrays, and transform plans.

The new images must match the shape of the originals — the grid and outputs are sized to it.

An omitted image keeps its current value, so `reinit!(cache; secondary = next)` advances one
image of a time series while holding the other fixed. An omitted *mask* follows its image: it
is kept only if that image was also omitted, and otherwise re-derived from finiteness. A mask
describes particular pixels, and carrying one over to a different image would silently mask the
wrong ones.

Only the images that actually changed are re-filtered, and an image already prepared in either
slot is reused — so a walk along consecutive acquisitions, where each new reference is the
previous secondary, filters each image once rather than twice.

Returns the cache, so it composes with [`autorift!`](@ref).
"""
function reinit!(cache::Cache; reference = nothing, secondary = nothing,
                 reference_valid = nothing, secondary_valid = nothing)
    isnothing(reference) && isnothing(secondary) &&
        isnothing(reference_valid) && isnothing(secondary_valid) && return cache

    old, oldprep = cache.raw, cache.prepared
    # Defaults come from the *raw* pair, so replacing one image re-prepares the other from its
    # original rather than filtering an already-filtered array.
    ref = isnothing(reference) ? old.reference : reference
    sec = isnothing(secondary) ? old.secondary : secondary
    # A mask belongs to its image, so an omitted one is carried over only if the cache already
    # holds that same image — in either slot, since a time series walks each acquisition from
    # secondary to reference. Otherwise it is `nothing` and the `ImagePair` constructor derives
    # it from finiteness. Inheriting a mask across a change of image would mask the wrong pixels.
    rv = isnothing(reference_valid) ? _maskfor(ref, old) : reference_valid
    sv = isnothing(secondary_valid) ? _maskfor(sec, old) : secondary_valid
    size(ref) == size(old) && size(sec) == size(old) || throw(DimensionMismatch(
        "new images are $(size(ref)) and $(size(sec)) but the cache was built for " *
        "$(size(old)); the grid and output arrays are sized to that. Build a new " *
        "cache with `init` for a different image size."))

    cache.raw = ImagePair(ref, sec; reference_valid = rv, secondary_valid = sv)
    p = cache.params
    newref = _reprepare(ref, cache.raw.reference_valid, old, oldprep, p)
    newsec = _reprepare(sec, cache.raw.secondary_valid, old, oldprep, p)
    cache.prepared = ImagePair(newref, newsec)
    # `|=` rather than `=`: two swaps before a single run must still leave the cache dirty.
    cache.isfresh |= true
    cache.result = nothing
    return cache
end

"""
    autorift!(cache) -> MultichipResult

Correlate the cache's current image pair, reusing its buffers and plans.

The batch path. A driver that processes many pairs should hold one cache per task and reuse
it, which avoids repeating the allocation and transform planning that would otherwise dominate
when each pair is small:

```julia
Threads.@sync for chunk in Iterators.partition(pairs, cld(length(pairs), nworkers))
    Threads.@spawn begin
        cache = AutoRIFT.init(first(chunk)...; threaded = false, kwargs...)
        for (a, b) in chunk
            reinit!(cache; reference = a, secondary = b)
            write_output(autorift!(cache))
        end
    end
end
```

Note `threaded = false` there. For batch work, one pair per task beats threading within a
pair: there is no serial fraction to bound the speedup and no load imbalance from the sparse
search, and each task keeps its own working set in cache. Intra-pair threading is for the
single-pair case, where it is the only parallelism available.

Calling this twice without an intervening [`reinit!`](@ref) returns the same result rather
than recomputing it.
"""
function autorift!(cache::Cache)
    cache.isfresh || isnothing(cache.result) || return cache.result
    cache.result = correlate_multichip(cache.prepared, cache.grid, cache.params)
    cache.isfresh = false
    return cache.result
end

"""
    autorift(reference, secondary; kwargs...) -> MultichipResult

Estimate the displacement of surface features between two images of the same scene, on a grid,
to sub-pixel precision.

Both images must be co-registered to a common grid and the same size. The result carries `dx`
and `dy` in pixels, the peak `correlation`, the `chip_size` that produced each point, and an
`interpolated` mask — see [`MultichipResult`](@ref). Points where no scale could produce a
coherent estimate are `NaN`, which is deliberately distinct from a measured displacement of
zero.

# Keywords

All of [`AutoRIFT.params`](@ref)'s, plus:

- `reference_valid`, `secondary_valid`: per-pixel validity masks. Defaults to finiteness. Pass
  these for sensors with a fill value, or to apply a cloud or shadow mask — an invalid pixel
  never contributes to a correlation.

```julia
out = autorift(image1, image2; chip_size = 32, search_radius = 25)
out.dx, out.dy, out.correlation
```

For many pairs, [`AutoRIFT.init`](@ref) and [`autorift!`](@ref) reuse buffers and transform
plans across calls.

!!! note "Sign convention"
    `dx` and `dy` are the offset from `secondary` back to `reference`, which is the *negative*
    of the motion of the imaged features — a glacier flowing east gives a negative `dx`. This
    matches the reference implementation. `dy` increases downward, matching array indexing.

    This is the *array* convention, and it is deliberately not what the `Raster` method returns.
    An array has no orientation to be north-up about, so the raw offsets are the honest output
    here; given a CRS, `autorift` instead returns map-oriented `vx`/`vy` for feature motion. The
    flip happens once, in the extension, where the y direction is actually known.
"""
autorift(reference::AbstractMatrix, secondary::AbstractMatrix; kwargs...) =
    autorift!(init(reference, secondary; kwargs...))

"""
    autorift(reference, secondary, grid::PointSet; kwargs...) -> MultichipResult

Correlate at a caller-supplied set of search points rather than a grid synthesized from
`grid_spacing`.

This is the production path: the eight per-pixel fields a grid generator produces — coordinates,
a-priori displacement, per-point search radius and chip-size bounds — cannot be expressed as
scalar keywords, and a [`PointSet`](@ref) is how they arrive.

A gridded `PointSet{2}` runs the full multi-chip-size search. A scattered `PointSet{1}` runs a
single scale via [`track`](@ref), since the chip-size loop's coarse pass and merge are neighbourhood
operations that need a layout.
"""
function autorift(reference::AbstractMatrix, secondary::AbstractMatrix, grid::PointSet;
                  reference_valid = nothing, secondary_valid = nothing, kwargs...)
    p = params(; kwargs...)
    pair = _prepare(ImagePair(reference, secondary; reference_valid, secondary_valid), p)
    return _run(pair, grid, p)
end

# Dispatch on the point set's dimensionality: only a gridded one can run multiple chip sizes.
_run(pair::ImagePair, grid::PointSet{2}, p::Params) = correlate_multichip(pair, grid, p)
_run(pair::ImagePair, pts::PointSet{1}, p::Params) = track(pair, pts, p)

"""
    AutoRIFT.autorift_with_grid(reference, secondary; kwargs...) -> (result, grid)

Correlate, and return the search grid alongside the result.

The same computation as [`autorift`](@ref), which discards the grid. A coordinate-aware caller
needs it: the result is indexed by grid point, and only the grid knows where those points sit in
the image — so mapping a displacement back to a map coordinate is impossible without it.

Exists so the DimensionalData and Rasters extensions share one path into the core rather than
each rebuilding the grid and hoping it matches. Not exported: the array API has no use for it.
"""
function autorift_with_grid(reference::AbstractMatrix, secondary::AbstractMatrix;
                            reference_valid = nothing, secondary_valid = nothing, kwargs...)
    p = params(; kwargs...)
    pair = _prepare(ImagePair(reference, secondary; reference_valid, secondary_valid), p)
    grid = _build_grid(size(pair), p)
    _warm_grid_plans(grid, p)
    return _run(pair, grid, p), grid
end

# ---------------------------------------------------------------------------

# Filter and quantize, in that order. Both are configuration-dependent and neither depends on
# the grid, so a cache does this once per image pair rather than once per run.
#
# Per image, because that is the unit of reuse. In a time series each acquisition is the
# secondary of one pair and the reference of the next, so `reinit!` swapping one image must not
# re-filter the other — on 1024² that is 23 ms and 23 MiB of pure waste per pair.
_prepare(img::AbstractMatrix, mask::AbstractMatrix{Bool}, p::Params) =
    quantize(preprocess(img, mask, p.preprocess)..., p.quantize)


_prepare(pair::ImagePair, p::Params) = ImagePair(
    _prepare(pair.reference, pair.reference_valid, p),
    _prepare(pair.secondary, pair.secondary_valid, p))

# Prepare `img`, unless the cache already holds it prepared — in which case reuse that.
#
# Matched against *both* slots of the old pair, not just the corresponding one. In a time series
# the pairs are consecutive acquisitions, so the new reference is the old secondary: the array
# the caller passes has already been filtered and quantized, just into the other slot. Checking
# only like-for-like would miss that and re-filter every image twice over its lifetime.
#
# Identity (`===`), not equality: comparing 4 MiB of pixels to decide whether to spend 12 ms
# filtering them would cost a large fraction of what it saves, and a caller who mutates an array
# in place and passes it again has changed the image without changing the object. Identity is
# the question that can be answered in a nanosecond and is never wrong in the reuse direction.
# The mask the cache already holds for `img`, or `nothing` if it does not hold that image.
_maskfor(img, old::ImagePair) =
    img === old.reference ? old.reference_valid :
    img === old.secondary ? old.secondary_valid : nothing

function _reprepare(img, mask, old::ImagePair, prep::ImagePair, p::Params)
    img === old.reference && mask === old.reference_valid &&
        return (prep.reference, prep.reference_valid)
    img === old.secondary && mask === old.secondary_valid &&
        return (prep.secondary, prep.secondary_valid)
    return _prepare(img, mask, p)
end

# The grid, when the caller did not supply one. Inset so every search window is in bounds,
# which is what lets `track!` skip padding entirely.
function _build_grid(imagesize::Tuple{Int,Int}, p::Params)
    return gridpoints(imagesize, p.grid_spacing;
                      chip_size = p.chip_size_max,
                      chip_aspect = p.chip_aspect,
                      search_radius_x = p.search_radius_x,
                      search_radius_y = p.search_radius_y,
                      dx_prior = p.dx_prior,
                      dy_prior = p.dy_prior)
end

# Plan the transforms every chip-size level will need, on this task, before any correlation.
#
# FFTW's planner is not thread-safe, so a threaded pass whose first points all miss the cache
# would have every task contend on the planner lock — turning the most parallel part of a run
# into its most serial. Doing it here costs one pass over the level list.
function _warm_grid_plans(grid::PointSet{2}, p::Params)
    rx = maximum(grid.radius_x)
    ry = maximum(grid.radius_y)
    (rx == 0 || ry == 0) && return nothing
    sizes = Tuple{Int,Int}[]
    for cs in chip_sizes(p)
        csy = chip_size_y(p, cs)
        push!(sizes, (next_fft_size(csy + 2ry - 1), next_fft_size(cs + 2rx - 1)))
    end
    warm_plans!(sizes)
    return nothing
end
