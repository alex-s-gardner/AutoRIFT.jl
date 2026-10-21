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
# Deliberately no `AbstractVector` method. It would work, but its return type infers as `Any` —
# `Tuple(::Vector)` has no length in its type — which forfeits exactly the concreteness the tuple
# design exists to preserve, and would be the one accepted spelling that silently costs the
# specialization. A caller with a vector writes `Tuple(v)`, and sees why at the call site.
_similarity(x) = _badtype(:similarity, x,
                         "a Symbol, a `SimilarityMeasure`, or a tuple of either")

# One element of a measure tuple. Separate from `_similarity` because that returns a tuple and
# this must not, or nesting would compound.
_one_similarity(x::SimilarityMeasure) = x
_one_similarity(x::Symbol) = _resolve(SYMBOL2SIMILARITY, x, :similarity)
_one_similarity(x) = _badtype(:similarity, x, "a Symbol or a `SimilarityMeasure`")

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

# `subpixel` resolves to a *tuple*, one method per chip-size level, on the same rule as
# `similarity`: a scalar becomes a 1-tuple whose last entry applies to every level, so the
# single-method case — nearly every call — is unchanged and stays concretely typed.
#
# A tuple is what expresses the reference's per-chip-size `OverSampleRatio`; see `subpixel_at`.
_subpixel(x::SubpixelMethod, _upsampling) = (x,)
_subpixel(x::Symbol, up) = (_one_subpixel(x, up),)
# `map` rather than a comprehension, so `R` stays a concrete `Tuple{PyramidRefine,PyramidRefine}`
# that the refinement kernel can specialize on. `upsampling` is not forwarded to the elements: a
# tuple names its methods, and each carries its own factor, so a loose keyword alongside would have
# two sources for one number.
function _subpixel(x::Tuple, up)
    isempty(x) && throw(ArgumentError(
        "`subpixel` cannot be an empty tuple; name at least one method."))
    isnokw(up) || throw(ArgumentError(
        "`upsampling` cannot be combined with a tuple of `subpixel` methods, which already carry " *
        "their own factors. Drop `upsampling`, or pass a single method."))
    return map(_one_subpixel, x)
end
_subpixel(x, _upsampling) = _badtype(:subpixel, x,
                                    "a Symbol, a `SubpixelMethod`, or a tuple of either")

# One element of a subpixel tuple. Separate from `_subpixel` because that returns a tuple and this
# must not, or nesting would compound.
_one_subpixel(x::SubpixelMethod, _up = nokw) = x
function _one_subpixel(x::Symbol, up = nokw)
    T = _resolve(SYMBOL2SUBPIXEL, x, :subpixel)
    return T === NoRefine ? NoRefine() :
           isnokw(up) ? T() : T(; upsampling = up)
end
_one_subpixel(x, _up = nokw) = _badtype(:subpixel, x, "a Symbol or a `SubpixelMethod`")

# An instance already carries its own parameters, so the loose keywords would have nothing to
# apply to. Passing both is a contradiction rather than a merge, and saying so beats silently
# preferring one — the alternative is a caller who sets `mad_scale` next to a `GardnerFilter`
# and never learns it was ignored.
function _outliers(x::OutlierMethod, kw::NamedTuple)
    # `nameof` rather than the type itself: printing a type reaches the type printer, which a
    # `--trim`ed binary does not have, and this message is built on every call.
    _no_stray_params(x, kw, "a $(nameof(typeof(x))) instance, which already carries its own " *
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

# `nothing`/`false` off, `true` the paper's default, a tuple the explicit angles. Accepting a `Bool`
# because "should I search rotations" reads as a yes/no question at the call site even though the
# answer carries parameters.
_rotation(::Nothing) = NoRotationSearch()
_rotation(x::Bool) = x ? RotationSearch() : NoRotationSearch()
_rotation(x::RotationMethod) = x
# One method over any real-valued iterable rather than a `Tuple` method plus an `AbstractVector` one:
# `RotationSearch`'s inner constructor already does `Tuple(Float64.(angles))`, so both were
# duplicating its work — and the pair silently rejected a *range*, which is the natural spelling for
# `-6:3:6`.
#
# Constrained to `Real` eltype rather than left as a bare fallback. `RotationSearch("yes")` does throw
# on its own, but as `MethodError: no method matching Float64(::String)` — which says nothing about
# which keyword was wrong. `_badtype` names it.
_rotation(x::Union{Tuple{Vararg{Real}},AbstractVector{<:Real},AbstractRange{<:Real}}) =
    RotationSearch(x)
_rotation(x) = _badtype(:rotation, x,
                       "`nothing`, a `Bool`, a collection of angles, or a `RotationMethod`")

# Backends map to instances, since they are singletons carrying no parameters. `:gpu` is
# deliberately absent: a machine can have more than one kind of device, and picking one on the
# package's behalf would be a silent choice about which. Naming the vendor is the whole content of
# the keyword.
const SYMBOL2BACKEND = Dict{Symbol,Backend}(
    :cpu   => CPU(),
    :metal => MetalGPU(),
    :cuda  => CUDAGPU(),
)

_backend(x::Backend) = x
_backend(x::Symbol) = _resolve(SYMBOL2BACKEND, x, :backend)
_backend(x) = _badtype(:backend, x, "a Symbol or an `AutoRIFT.Backend`")

_badtype(kw::Symbol, x, expected) =
    throw(ArgumentError("`$kw` must be $expected, got a $(typeof(x))."))

# ---------------------------------------------------------------------------
# Numeric validation
# ---------------------------------------------------------------------------

# Chip sizes must be multiples of 4 in each axis. A chip's centre has to land on a pixel boundary
# for its displacement to be reported at one, which needs an even extent; and a chip halved once
# still needs an even half-extent, which is the second factor of two.
function _check_chip_size(name::Symbol, cs)
    e = extent(cs)
    for (ax, v) in ((:X, e.X), (:Y, e.Y))
        v > 0 || throw(ArgumentError("`$name` must be positive, got $v in $ax"))
        v % 4 == 0 || throw(ArgumentError(
            "`$name` must be a multiple of 4, got $v in $ax. A chip's centre lands on a pixel " *
            "boundary only for an even extent, and a chip halved once needs an even half-extent."))
    end
    return e
end

# Grid spacing: positive in each axis. Nothing divides by it in `src/`, so there is no further
# constraint — `block_layout` walks the grid's own coordinates rather than assuming a spacing.
function _check_spacing(sp)
    e = extent(sp)
    for (ax, v) in ((:X, e.X), (:Y, e.Y))
        v > 0 || throw(ArgumentError("`grid_spacing` must be positive, got $v in $ax"))
    end
    return e
end

# Search radius: zero is allowed per axis — that is how a caller searches along one axis only — but
# not in both, which would leave no pixel searchable.
function _check_radius(r)
    e = extent(r)
    for (ax, v) in ((:X, e.X), (:Y, e.Y))
        v >= 0 || throw(ArgumentError("`search_radius` must be >= 0, got $v in $ax"))
    end
    (e.X > 0 || e.Y > 0) || throw(ArgumentError(
        "`search_radius` is zero in both axes, so no pixel can be searched."))
    return e
end

# The levels `chip_size .* 2^k` must reach `chip_size_max` in both axes at the same k, so the aspect
# a caller asked for holds at every level rather than drifting across them. That is what makes a
# coarse grid point exactly a block of fine ones — see `src/multichip.jl`.
#
# So `(X=16, Y=32)` to `(X=32, Y=64)` is fine, both ratios being 2, while `(X=16, Y=16)` to
# `(X=128, Y=64)` is not: x would reach its maximum after three doublings and y after two.
function _check_levels(cmin::Extent, cmax::Extent)
    (cmin.X <= cmax.X && cmin.Y <= cmax.Y) || throw(ArgumentError(
        "`chip_size` ($(cmin.X) by $(cmin.Y)) must not exceed `chip_size_max` " *
        "($(cmax.X) by $(cmax.Y)) in either axis."))
    rx, ry = cmax.X ÷ cmin.X, cmax.Y ÷ cmin.Y
    (cmax.X % cmin.X == 0 && cmax.Y % cmin.Y == 0 && ispow2(rx) && ispow2(ry)) ||
        throw(ArgumentError(
            "`chip_size_max` must be a power-of-two multiple of `chip_size` in each axis, " *
            "because chip-size levels double at each step. Got $(cmin.X) to $(cmax.X) in X and " *
            "$(cmin.Y) to $(cmax.Y) in Y."))
    rx == ry || throw(ArgumentError(
        "`chip_size_max` must be the same multiple of `chip_size` in both axes, so the chip's " *
        "aspect ratio is the same at every level. Got $rx in X and $ry in Y — x would reach its " *
        "maximum after a different number of doublings than y."))
    return nothing
end

function _check_fraction(name::Symbol, v::Real)
    0 <= v <= 1 ||
        throw(ArgumentError("`$name` must be in [0, 1], got $v"))
    return Float64(v)
end

function _check_positive(name::Symbol, v::Real)
    v > 0 || throw(ArgumentError("`$name` must be positive, got $v"))
    return v
end

function _check_nonnegative(name::Symbol, v::Real)
    v >= 0 || throw(ArgumentError("`$name` must be >= 0, got $v"))
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

Resolve and validate keywords into a concrete [`Params`](@ref).

Every `Symbol` is mapped to a method object and every number is range-checked, so an invalid value
throws `ArgumentError` naming the keyword. Defaults match the autoRIFT reference driver, except for
a few of its defects; `dev/REFERENCE.md` lists them.

# Keywords

## Method selection
- `similarity = :zncc`: [`ZNCC`](@ref), [`NCC`](@ref), or [`Coherence`](@ref). A tuple assigns
  measures to chip-size levels in order, repeating the last for the remaining levels, so
  `(:coherence, :zncc)` uses coherence at the finest chip size and ZNCC at every coarser one. A tuple
  longer than the number of levels throws when the levels are known. `Coherence` requires complex
  input.
- `preprocess = :highpass`: pre-correlation filter; one of `:highpass`, `:wallis`, `:wallis_gapfill`,
  `:sobel`, `:laplacian`, `:decibel`, `:deramp`, `:none`, or a [`PreprocessMethod`](@ref).
- `filter_width`: window width for a `preprocess` given as a Symbol. Default: the method's own
  (`5`). Ignored by methods without a window.
- `subpixel = :pyramid`: peak refinement (`:pyramid`, `:none`, a [`SubpixelMethod`](@ref), or a tuple
  of either, one per level with the last repeated); see [`SubpixelMethod`](@ref).
- `upsampling`: upsampling factor for a `subpixel` given as a Symbol. Default: `64`. Cannot be combined
  with a tuple `subpixel`.

## Chip and grid geometry, in pixels

Each of these is an extent: a scalar means square and `(X = …, Y = …)` names the axes.

```julia
chip_size = 32                 # 32 by 32
chip_size = (X = 16, Y = 32)   # taller than wide
```

- `chip_size = 32`: the finest chip-size level. Each axis must be a multiple of 4.
- `chip_size_max = 4 * chip_size`: the coarsest level. Levels are `chip_size .* 2^k` up to this, so it
  must be the same power-of-two multiple of `chip_size` in both axes.
- `grid_spacing = chip_size`: spacing of output grid points.

## Search geometry, in pixels
- `search_radius = 25`: half-extent of the search window, as an extent. The window spans
  `2 * radius`. Zero in one axis searches along the other only; zero in both throws.
- `min_search_radius = 6`: floor applied to non-zero radii, per axis.
- `dx_prior = 0.0`, `dy_prior = 0.0`: a-priori displacement the search window is centered on.
- `coarse_stride = 4`: how sparsely the coarse pass samples, as a rate against one point per chip. The
  step in grid points is this times the chip-to-spacing ratio, clamped to 1 to 2, so on a grid finer
  than its chips the coarse pass samples more sparsely in grid points than this number alone suggests.
- `coarse_buffer = 8`: dilation radius, in coarse cells, applied to the coarse validity mask before it
  restricts the fine search.
- `min_coarse_valid_fraction = 0.01`: a chip-size level is skipped when its coarse pass validates a
  smaller fraction of the coarse grid than this.
- `min_chip_valid_fraction = 0.65`: a point is correlated only if more than this fraction of its own
  chip footprint is valid. The search window around the chip may reach the image edge. The value comes
  from `dev/CORRECTNESS.md` item 1, which sweeps the threshold on the swath-boundary nodes of a NISAR
  L1 scene.

## Outlier rejection and filling
- `rotation = nothing`: correlate each chip at several rotations and keep the strongest peak; see
  [`RotationSearch`](@ref). `nothing` or `false` disables it. `true` uses ±3°; a collection of angles
  in degrees or a [`RotationMethod`](@ref) is also accepted. Disabled by default because it
  multiplies correlation cost by the number of angles. See [Giving the search a first guess](@ref) for
  centering the angle window on a scene that is already rotated.
- `outliers = :gardner`: `:gardner` (the filter of Gardner et al. (2018), [`GardnerFilter`](@ref)),
  `:none` (keep everything), or an [`OutlierMethod`](@ref) instance.
- `outlier_window = 5`: neighborhood width for the filter, in grid points. Odd and at least 3.
- `outlier_iterations = 3`: iterations of the filter's first stage.
- `min_agree_fraction = 8/25`: fraction of the window area that must agree.
- `agree_tolerance = 0.2`: agreement threshold, as a fraction of search radius.
- `mad_scale = 4.0`: median-absolute-deviation multiplier.
- `fill_window = 3`: window for median hole filling. Must be odd.
- `fill_min_hole = 5`: a connected hole smaller than this is filled regardless of its neighbor count.
  Exclusive, so the default closes holes of one to four points. `0` leaves filling to the window
  criterion alone.

The five keywords from `outlier_window` to `mad_scale` are [`GardnerFilter`](@ref)'s parameters. They
cannot be combined with an `outliers` instance or with `outliers = :none`; either throws.

## Misc
- `threaded = false`: parallelize over grid points within one image pair. For many pairs, one pair
  per task with `threaded = false` is usually faster; see [Correlating many pairs](@ref).
- `backend = :cpu`: where the correlation kernels run, `:cpu`, `:metal` (needs `using Metal`) or
  `:cuda` (accepted but not implemented; correlating with it throws), or an [`AutoRIFT.Backend`](@ref)
  instance. A GPU backend cannot be combined with `threaded = true`, which parallelizes the same loop.
  The GPU path reports `dx`/`dy` identical to the CPU's and `correlation` within 1e-5. The GPU backend
  is experimental and does not outperform a threaded CPU run on one pair; see
  [Correlating on a GPU](@ref).
- `progress = false`: stored in `Params.progress`. No pass reads it, so it currently has no effect.
- `rng_seed = 0`: seed for the noise [`WallisGapfill`](@ref) fills gaps with, making a run
  reproducible. No other `preprocess` choice reads it.
"""
function params(;
    similarity = :zncc,
    preprocess = :highpass,
    subpixel = :pyramid,
    filter_width = nokw,
    upsampling = nokw,
    chip_size = 32,
    chip_size_max = nokw,
    grid_spacing = nokw,
    search_radius = 25,
    min_search_radius = 6,
    dx_prior = 0.0,
    dy_prior = 0.0,
    coarse_stride = 4,
    coarse_buffer = 8,
    min_coarse_valid_fraction = 0.01,
    min_chip_valid_fraction = _DEFAULT_MIN_CHIP_VALID_FRACTION,
    outliers = :gardner,
    rotation = nothing,
    outlier_window = nokw,
    outlier_iterations = nokw,
    min_agree_fraction = nokw,
    agree_tolerance = nokw,
    mad_scale = nokw,
    fill_window = 3,
    fill_min_hole = 5,
    threaded = false,
    backend = :cpu,
    progress = false,
    rng_seed = 0,
)
    sim = _similarity(similarity)
    pre = _preprocess(preprocess, filter_width)
    sub = _subpixel(subpixel, upsampling)
    # The public keywords are prefixed (`outlier_window`) where the method's own are not
    # (`window`), because at the top level `window` alone would be ambiguous against
    # `fill_window` and `filter_width`. Translated here, in the one file that owns keywords.
    out = _outliers(outliers, (; window = outlier_window, iterations = outlier_iterations,
                               min_agree_fraction, agree_tolerance, mad_scale))
    rot = _rotation(rotation)
    back = _backend(backend)
    # Intra-pair threading and a GPU both parallelise the same loop, and on a device they would
    # contend on one queue rather than compose. Rejected here rather than silently ignoring one:
    # a caller who asked for both has a wrong expectation about which is in effect.
    (isgpu(back) && Bool(threaded)) && throw(ArgumentError(
        "`threaded = true` and `backend = :$(lowercase(string(nameof(typeof(back)))))` cannot " *
        "be combined: both parallelise the grid loop, and the GPU path already batches every " *
        "point of a pass onto one device queue. Drop `threaded`, or use `backend = :cpu`. For " *
        "many pairs, run one pair per process instead — that measures faster than either."))

    cmin = _check_chip_size(:chip_size, chip_size)
    cmax = isnokw(chip_size_max) ? (X = 4cmin.X, Y = 4cmin.Y) :
        _check_chip_size(:chip_size_max, chip_size_max)
    _check_levels(cmin, cmax)

    spacing = isnokw(grid_spacing) ? cmin : _check_spacing(grid_spacing)
    rad = _check_radius(search_radius)

    return Params(
        sim, pre, sub, out, booltype(threaded), rot,
        cmin, cmax, spacing, rad, Int(min_search_radius),
        Int(_check_positive(:coarse_stride, coarse_stride)),
        Int(coarse_buffer),
        _check_fraction(:min_coarse_valid_fraction, min_coarse_valid_fraction),
        _check_fraction(:min_chip_valid_fraction, min_chip_valid_fraction),
        Float64(dx_prior), Float64(dy_prior),
        _check_odd_window(:fill_window, fill_window),
        Int(_check_nonnegative(:fill_min_hole, fill_min_hole)),
        UInt64(rng_seed),
        Bool(progress),
        back,
    )
end

"""
    chip_size_scale(g) -> Float64

The y:x chip-size ratio `g` implies: the median of `chip_min_y / chip_min_x` over the points carrying
both bounds. Chips are square on the ground, so this is the y:x pixel size ratio, `1.0` for a square
pixel and about `0.25` on a Sentinel-1 pair.

`params(g)` derives the base chip extent from this value rounded to an even pixel count; this function
returns it unrounded.

Defined when `ImagePairGeometry` is loaded, for `g::ImagePairGeometry.PairGeometry`.
"""
function chip_size_scale end
