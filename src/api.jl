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
# `_run` rather than done inline: the filter may change the element type — an integer image
# filtered by `Highpass` comes back `Float32` — so the call crosses a type boundary, and putting
# a function barrier there keeps everything downstream monomorphic instead of leaving the whole
# pipeline inferring a union.

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
    # The pair as supplied, before filtering. Kept because `reinit!` may replace only one of the
    # two images, and the other must then be re-prepared from its original — preparing the
    # already-prepared one would filter it a second time.
    raw::ImagePair
    # How this cache's passes run, and over which pair: an `AutoRIFT.WholeScene` holding the
    # filtered scene, or an `AutoRIFT.Blocked` holding the raw one and filtering per block.
    #
    # One field rather than a `prepared::Union{Nothing,ImagePair}` beside a
    # `block_size::Union{Nothing,Tuple}`: those two had to agree — a filtered scene *and* a block
    # size is a contradiction, and neither is a run with nothing to correlate — and nothing enforced
    # it. A runner cannot express the disagreement.
    #
    # Blocking stays out of `Params` because it is a scheduling choice rather than an algorithm
    # parameter: it changes peak memory and nothing else, since the two paths agree bit for bit.
    # That also keeps `Params`'s 22 positional fields, and every construction of it including the
    # trimmed binary's, unchanged.
    runner::PassRunner
    grid::PointSet{2}
    result::Union{Nothing,MultichipResult}
    # Set when the images change and cleared once a run has consumed them. `reinit!`
    # accumulates dirtiness rather than overwriting it, so two swaps before one run still
    # leave the cache correctly marked.
    isfresh::Bool
end

"""
    AutoRIFT.imagepair(cache) -> ImagePair

The filtered pair the correlator will see.

Throws for a cache built with `process_block_size`. Such a run filters each block from its own read
window and never forms a filtered scene, so there is no whole-scene pair to return — and returning
the raw one would silently answer a different question.
"""
imagepair(cache::Cache) = imagepair(cache.runner)

imagepair(r::WholeScene) = r.prepared
imagepair(::Blocked) = throw(ArgumentError(
    "this cache was built with `process_block_size`, so no whole-scene filtered pair exists: " *
    "each block is filtered from its own read window, which is what bounds peak memory. Use " *
    "`cache.raw` for the input as supplied, or build the cache without `process_block_size`."))

"""
    AutoRIFT.init(reference, secondary; kwargs...) -> Cache

Prepare to correlate `reference` against `secondary`, allocating buffers and planning
transforms once.

Accepts the same keywords as [`autorift`](@ref). The images are preprocessed immediately, since
that is configuration-dependent work that does not need repeating per run — except under
`process_block_size`, where each block is filtered from its own read window during the run and no
whole-scene filtered copy is ever formed. [`AutoRIFT.imagepair`](@ref) therefore has nothing to
return for such a cache and says so.

Use this with [`reinit!`](@ref) when processing many pairs: the grid, the output arrays, and
the FFT plans are then built once rather than per pair. For a single pair, call
[`autorift`](@ref) instead.
"""
function CommonSolve.init(reference::AbstractMatrix, secondary::AbstractMatrix;
              reference_valid = nothing, secondary_valid = nothing,
              process_block_size = nothing, kwargs...)
    p = params(; kwargs...)
    raw = ImagePair(reference, secondary; reference_valid, secondary_valid)
    # Before the grid and the plans, so a filter that cannot run on this element type is an error at
    # the call that configured it rather than at the first correlation.
    _check_preprocess(eltype(raw), p.preprocess)
    bs = _block_size(process_block_size)
    # A blocked run filters each block from its own read window, so the filtered scene is never
    # formed — which is what bounds peak memory by the block rather than by the scene, and is the
    # reason `process_block_size` exists. Preparing here would defeat it, and would also filter
    # every block twice.
    grid = _build_grid(size(raw), p)
    # Plans are created here rather than on first use so that a batch driver pays for them
    # once, on the task that builds the cache, and never inside a correlation.
    _warm_grid_plans(grid, p)
    # The layout is built here rather than at the first run, so a block size that cannot work is an
    # error at the call that set it. `block_layout` is what knows the halo, so it is what checks.
    return Cache{typeof(p)}(p, raw, _runner(raw, grid, p, bs), grid, nothing, true)
end

# `process_block_size` as a tuple, or `nothing` for one block.
#
# A tuple and only a tuple. A bare `Int` is rejected rather than read as a square block: a
# full-width band is the cheaper shape at a given halo — halo on two sides rather than four — and
# inferring "square" from a scalar would quietly pick the more expensive one on the caller's behalf.
_block_size(::Nothing) = nothing
_block_size(bs::Tuple{Integer,Integer}) = (Int(bs[1]), Int(bs[2]))
_block_size(bs) = throw(ArgumentError(
    "`process_block_size` must be a tuple of two integers, `(X, Y)` grid points, or `nothing` " *
    "for one block over the whole grid. Got a $(typeof(bs)). A scalar is not accepted: a " *
    "full-width band costs less halo than a square block of the same area, so which one you " *
    "want is worth saying."))

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

!!! warning "`Params` is not reinitialised, which matters for `about`"
    The cache keeps its `Params`, and that is the point — the plans and buffers depend on it. But
    [`RotationSearch`](@ref)'s `about` is a **per-pair** quantity, so a cache built with
    `about = scene_rotation(guess)` from the first pair applies that pair's ice rotation to every
    later one, and `reinit!` cannot update it. For a rotating time series, call [`autorift`](@ref)
    per pair with a freshly fitted `about` instead of walking a cache.

Returns the cache, so it composes with [`autorift!`](@ref).
"""
function reinit!(cache::Cache; reference = nothing, secondary = nothing,
                 reference_valid = nothing, secondary_valid = nothing)
    isnothing(reference) && isnothing(secondary) &&
        isnothing(reference_valid) && isnothing(secondary_valid) && return cache

    old = cache.raw
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
    cache.runner = _reinit_runner(cache.runner, cache.raw, old, cache.params)
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
    cache.result = _multichip(cache.runner, cache.grid, cache.params)
    cache.isfresh = false
    return cache.result
end

# The runner a block size implies, and the pair it correlates.
#
# The two take *different pairs*, which is the point rather than an inconsistency: a whole-scene run
# correlates a filtered scene, while a blocked run filters each block from raw and so never forms
# one. Pairing each pair with its runner here is what makes handing the blocked path a prepared pair
# — a twice-filtered image — impossible to write.
_runner(raw::ImagePair, ::PointSet{2}, p::Params, ::Nothing) = WholeScene(_prepare(raw, p))

function _runner(raw::ImagePair, grid::PointSet{2}, p::Params, bs::Tuple{Int,Int})
    layout = block_layout(grid, p, size(raw), bs)
    buffers = istrue(p.threaded) ? nothing : block_buffers(raw, layout)
    return Blocked(raw, layout, layout.blocks, buffers)
end

# The runner for a cache whose images have just changed.
#
# A whole-scene runner refilters, but only what changed: `_reprepare` matches each new image against
# both slots of the old pair, so a time series that walks each acquisition from secondary to reference
# refilters one image per pair rather than two.
_reinit_runner(old_runner::WholeScene, raw::ImagePair, old::ImagePair, p::Params) = WholeScene(
    ImagePair(_reprepare(raw.reference, raw.reference_valid, old, old_runner.prepared, p),
              _reprepare(raw.secondary, raw.secondary_valid, old, old_runner.prepared, p)))

# A blocked runner has nothing to refilter — that happens per block, inside the run — so it only has
# to point at the new pair. The layout and buffers carry over: both are sized from the grid and the
# image size, and `reinit!` rejects a change of image size.
_reinit_runner(old_runner::Blocked, raw::ImagePair, ::ImagePair, ::Params) =
    Blocked(raw, old_runner.layout, old_runner.blocks, old_runner.buffers)

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
- `process_block_size`: `(X, Y)` grid points per block, or `nothing` (the default) for one block
  over the whole scene. Reads and filters the images a block at a time, so no array the size of the
  scene is ever formed. **Bit-identical to the untiled run** — this changes where the work happens,
  not what it computes.

  For inputs that read a window cheaply, which means a lazy `Raster` or any other disk-backed
  array. **On an array already in memory this costs more than it saves** and is measured doing so up
  to at least 4096² (`benchmark/memory.jl`): the scene is already resident, so blocking adds the
  halo without removing anything. Reach for it when the scene is the thing that will not fit.

  A tuple and only a tuple: a full-width band costs halo on two sides where a square block pays it
  on four, so which shape you want is worth saying rather than inferring from a scalar. Pass the
  grid's full width as `X` for a band.

  Each block reads its own extent grown by a halo, so a block reads more than it writes — see
  [`AutoRIFT.halo`](@ref). A block smaller than that halo would be almost entirely overlap and is
  rejected. Smaller blocks hold less at once but read a larger multiple of the scene, since the halo
  is a fixed width around a shrinking interior.

  A preprocessing filter that estimates from the whole image cannot be reproduced block by block,
  and is rejected rather than approximated — see [`AutoRIFT.filter_reach`](@ref).

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
    autorift(reference, secondary, p::Params) -> MultichipResult

Correlate with an already-resolved [`Params`](@ref), bypassing keyword resolution.

Same computation and same result as the keyword form. What differs is that nothing between the
call and the correlation is a runtime value: `Params`'s method choices are type parameters, so
when `p` is built from method *objects* the whole pipeline below this call is statically
resolvable.

That matters in two places. It is the entry point a `--trim`ed binary needs, since
[`AutoRIFT.params`](@ref) resolves Symbols through a `Dict{Symbol,Any}` and the constructor call
after that lookup cannot be resolved at compile time. And it lets a caller who builds `Params`
once reuse it across pairs without re-validating keywords — though [`AutoRIFT.init`](@ref) is
the better tool for that, since it also reuses buffers.

```julia
p = AutoRIFT.Params((ZNCC(),), Highpass(), PyramidRefine(), GardnerFilter(),
                    AutoRIFT.False(), AutoRIFT.NoRotationSearch(), 32, 32, 128, 1.0, 32, 25, 25,
                    6, 4, 8, 0.01, 0.0, 0.0, 3, UInt64(0), false)
out = autorift(image1, image2, p)
```

`Params` has no keyword constructor deliberately: `params()` is the documented way to build one
with defaults, and a second spelling of the defaults would be a second place for them to drift.
The positional form is stable API — the field order is `Params`'s own, in declaration order.

Validity masks are not accepted here. They are per-image data rather than configuration, and the
keyword form or [`ImagePair`](@ref) is where they belong; this overload exists for the case where
*nothing* is a runtime value.
"""
function autorift(reference::AbstractMatrix, secondary::AbstractMatrix, p::Params)
    pair = _prepare(ImagePair(reference, secondary), p)
    grid = _build_grid(size(pair), p)
    _warm_grid_plans(grid, p)
    return correlate_multichip(pair, grid, p)
end

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
                  reference_valid = nothing, secondary_valid = nothing,
                  process_block_size = nothing, kwargs...)
    p = params(; kwargs...)
    raw = ImagePair(reference, secondary; reference_valid, secondary_valid)
    return _run(raw, grid, p, _block_size(process_block_size))
end

# Dispatch on the point set's dimensionality: only a gridded one can run multiple chip sizes.
#
# Whether the scene is filtered at all depends on the block size, which is what `_runner` decides: a
# blocked run filters per block and must never form a filtered scene, since that copy is the
# allocation it exists to avoid.
function _run(raw::ImagePair, grid::PointSet{2}, p::Params, bs::Union{Nothing,Tuple{Int,Int}})
    _check_preprocess(eltype(raw), p.preprocess)
    return _multichip(_runner(raw, grid, p, bs), grid, p)
end

# A scattered set runs one pass at one chip size, so there are no levels to loop over and no coarse
# restriction to apply — `track` is the whole computation.
function _run(raw::ImagePair, pts::PointSet{1}, p::Params, ::Nothing)
    _check_preprocess(eltype(raw), p.preprocess)
    return track(_prepare(raw, p), pts, p)
end

# Whether this filter can run on this element type, checked once at the call that started the run.
#
# `params` cannot do this: it has no image, so it cannot know the element type, and its promise that
# nothing below re-validates (see the note at the top of this file) therefore cannot cover the
# filter/eltype pairing. Without this the mismatch surfaces from inside `_prepare` — mid-run, after
# the grid is built and the plans are warmed — as an error about an array type rather than about the
# keyword that chose it.
#
# Dispatch rather than a condition, so a filter added with no method for the element type it is
# handed produces a `MethodError` here, at the boundary, instead of deep in the filter stack.
_check_preprocess(::Type, ::PreprocessMethod) = nothing
_check_preprocess(::Type{<:Real}, ::Deramp) = throw(ArgumentError(
    "`preprocess = :deramp` needs complex input, but the images are real. A real image has no " *
    "phase ramp to remove. Use `preprocess = :highpass` for real imagery, or pass complex data."))

# Blocks are laid out over a *gridded* point set, since the halo is derived from where points sit
# relative to each other. A scattered set has no such layout, so there is nothing to divide.
_run(::ImagePair, ::PointSet{1}, ::Params, ::Tuple{Int,Int}) = throw(ArgumentError(
    "`process_block_size` needs a gridded `PointSet`, but this one is scattered. Blocks are " *
    "rectangles of the output grid, and a scattered point set has no grid to cut. Drop the " *
    "keyword, or pass a gridded point set."))

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
                            reference_valid = nothing, secondary_valid = nothing,
                            process_block_size = nothing, kwargs...)
    p = params(; kwargs...)
    raw = ImagePair(reference, secondary; reference_valid, secondary_valid)
    # From the raw pair's size, which the filters preserve.
    grid = _build_grid(size(raw), p)
    _warm_grid_plans(grid, p)
    return _run(raw, grid, p, _block_size(process_block_size)), grid
end

# ---------------------------------------------------------------------------

# Filter, then replace non-finite pixels. The filter is configuration-dependent and neither step
# depends on the grid, so a cache does this once per image pair rather than once per run.
#
# Per image, because that is the unit of reuse. In a time series each acquisition is the
# secondary of one pair and the reference of the next, so `reinit!` swapping one image must not
# re-filter the other — on 1024² that is 23 ms and 23 MiB of pure waste per pair.
# `resident` is called here, and only here. Every whole-array reduction over a mask — `all(mask)` in
# `_masked_boxmean!` and in `_erode_mask!` — is downstream of either this function or `_read_block!`,
# and neither can hand one a lazy mask: this materializes at entry, and `_read_block!` copies into a
# dense buffer. So the re-scan a computed mask would cost is unreachable rather than remembered.
#
# A blocked run never reaches here, which is the whole point: it filters per block from raw input, so
# the untiled path pays one pass over the mask and the blocked path pays nothing, with no branch.
_prepare(img::AbstractMatrix, mask::AbstractMatrix{Bool}, p::Params) =
    replace_nonfinite(preprocess(img, resident(mask), p.preprocess, p.rng_seed)...)


_prepare(pair::ImagePair, p::Params) = ImagePair(
    _prepare(pair.reference, pair.reference_valid, p),
    _prepare(pair.secondary, pair.secondary_valid, p))

# Prepare `img`, unless the cache already holds it prepared — in which case reuse that.
#
# Matched against *both* slots of the old pair, not just the corresponding one. In a time series
# the pairs are consecutive acquisitions, so the new reference is the old secondary: the array
# the caller passes has already been filtered, just into the other slot. Checking
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
    # Per level, because the measure can differ per level: a `(:coherence, :zncc)` run needs the
    # complex plan pair at the finest chip and the real pair above it. Warming one kind for the whole
    # grid would leave half the levels to plan inside a worker task, under the planner lock.
    sizes = Tuple{Int,Int}[]
    csizes = chip_sizes(p)
    for k in eachindex(csizes)
        cs = csizes[k]
        csy = chip_size_y(p, cs)
        sz = (next_fft_size(csy + 2ry - 1), next_fft_size(cs + 2rx - 1))
        if _wants_complex_plans(measure_at(p, k))
            warm_plans!((sz,); complex = true)
        else
            push!(sizes, sz)
        end
    end
    isempty(sizes) || warm_plans!(sizes)
    return nothing
end
