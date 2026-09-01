# Multi-chip-size correlation.
#
# ---------------------------------------------------------------------------
# Not an image pyramid
# ---------------------------------------------------------------------------
#
# Worth stating plainly, because the structure invites the wrong name and this file used to
# carry it. The levels differ in **chip size**; the imagery is never downsampled. Every level
# correlates the same full-resolution `pair` — there is no coarse-to-fine warm start from a
# decimated image, and no image pyramid anywhere in the algorithm.
#
# The reference agrees: its `cv2.resize` calls (`autoRIFT.py:507-590, 812-860`) act only on the
# *grid* arrays — `xGrid`, `yGrid`, `SearchLimitX/Y`, `Dx0`, the masks — coarsening the grid by
# `ChipSizeUniX[i] / ChipSize0X` to match the larger chip. `I1` and `I2` are filtered and
# quantized (`258-397`) and otherwise untouched. So this is a nested multi-chip-size grid.
#
# The one real pyramid in autoRIFT is the subpixel solver: `cv::pyrUp` cascaded in
# `autoriftcoremodule.cpp:292`, which is `PyramidRefine` in `src/peak.jl`. That name is
# correct and deliberate.
#
# Layer 3. This is the orchestration, and unlike everything below it, it is deliberately
# *not* general. The loop is a knot of tuned glaciology heuristics — chip-size-indexed
# levels, a coarse pass that restricts the fine one, distance-transform dilation, hole
# filling, smallest-chip-wins merging — and generalising it would produce a function with
# thirty keywords that nobody could use. It stays concrete, but every step is separately
# callable so a caller can build a variant.
#
# ---------------------------------------------------------------------------
# Why more than one chip size
# ---------------------------------------------------------------------------
#
# Chip size trades resolution against reliability. A small chip resolves fine detail and
# follows a shear margin, but carries little texture, so on a smooth snowfield it correlates
# with nothing. A large chip is reliable there and blind to detail. Neither choice is right
# everywhere in one scene.
#
# So each level is tried in turn, finest first, and a level only writes where no finer level
# already produced a value. The smallest chip that works wins, per point. Levels are
# `chip_size * 2^k`, which is what makes the grids nest: a coarse grid point is exactly a
# block of fine ones.
#
# That correspondence is defined in exactly one place, `_cell_max_radius!`, and inverted in exactly
# one other, `_expand_coarse_mask`. Both directions must use the same cell boundaries — a mask that
# restricts the fine search has to sit on the evidence that produced it.
#
# ---------------------------------------------------------------------------
# Why a coarse pass before each fine pass
# ---------------------------------------------------------------------------
#
# Searching every point at full radius is the dominant cost, and most of it is wasted:
# displacement is spatially coherent, so a sparse sample of points tells you where the rest
# will land. Each level therefore runs on a decimated grid first, filters those results for
# consistency, dilates what survives, and searches the full grid only inside that. Where the
# coarse pass found nothing coherent, the fine pass is skipped entirely.
#
# That is why the search radius is a per-point field rather than a scalar: setting it to zero
# is how a point gets excluded.

"""
    MultichipResult

Displacement over the full grid, assembled from all chip-size levels.

`dx`, `dy`, `correlation`, and `peak_snr` are as in [`DisplacementField`](@ref). `chip_size` records
which level produced each point — `0` where none did — and `interpolated` marks points
filled from their neighbours rather than measured.

`peak_snr` reports how far the peak stood above the rest of its surface, which is a different question
from `correlation`, the peak's height. **To gate on reliability use `correlation`**: against disagreement
with the Python reference it reaches an AUC of 0.842 where `peak_snr` reaches 0.499, and
[`AutoRIFT.peak_quality`](@ref) records the measurement. `peak_snr` is the diagnostic for an *ambiguous*
surface — one with rival peaks — and for the search-boundary condition below.

At an `interpolated` point it is the median of the neighbourhood the displacement itself was taken from,
since that is what stands behind the value; `correlation` is `NaN` there instead, having no surface of its
own to report.

`peak_snr` is **zero** where the correlation peak lay against the search boundary: the displacement is a
lower bound and its sub-pixel part is quantized to whole pixels, so any positive threshold excludes those
points without the caller needing to know about them. Select them with `peak_snr .== 0` to find where
`search_radius` is too small — `AutoRIFT.peak_quality` documents the measurements.

`chip_size` holds the level's **x** extent. That identifies the level on its own, since the aspect
is constant across levels, so the y extent is this times a fixed ratio.

The chip size is worth keeping rather than discarding: it says how much spatial averaging is
behind each estimate, so a downstream consumer can tell a sharply-resolved velocity from a
smoothed one.

`interpolated` is a `BitMatrix` here, unlike `searched` on [`DisplacementField`](@ref), which
is a `Matrix{Bool}`. The difference is deliberate and is about concurrency, not taste: this
one is written only by the single-threaded merge, so the packed representation is safe, while
that one is written from the threaded grid loop where a read-modify-write on a shared word
could lose a neighbour.
"""
struct MultichipResult
    dx::Matrix{Float32}
    dy::Matrix{Float32}
    correlation::Matrix{Float32}
    peak_snr::Matrix{Float32}
    chip_size::Matrix{UInt16}
    interpolated::BitMatrix
end

Base.size(r::MultichipResult) = size(r.dx)
Base.length(r::MultichipResult) = length(r.dx)

"""
    nmeasured(r::MultichipResult) -> Int

How many grid points carry a displacement.
"""
nmeasured(r::MultichipResult) = count(!isnan, r.dx)

# A result with nothing resolved: `NaN` where no displacement was measured, and a zero chip size
# marking every point as still wanted by the level loop.
_empty_result(sz::Tuple{Int,Int}) = MultichipResult(
    fill(NaN32, sz), fill(NaN32, sz), fill(NaN32, sz), fill(NaN32, sz),
    zeros(UInt16, sz), falses(sz))

# The chip sizes a run will step through, checked against the measures it was given.
#
# `params` already guarantees at least one level: it requires `chip_size <= chip_size_min <=
# chip_size_max` and that the bounds are power-of-two multiples of `chip_size`, so the sequence
# always contains `chip_size_min`. Asserted rather than handled, since a caller cannot reach the
# empty case through the public constructor.
#
# The measure check happens here, before any correlation, rather than at the level that would have
# used a surplus entry.
function _level_sizes(p::Params)
    sizes = chip_sizes(p)
    # Interpolates the components rather than the extents: showing a `NamedTuple` reaches
    # `Base.repeat` via `textwidth`, which `--trim` cannot resolve — see `src/track.jl`.
    @assert !isempty(sizes) "no chip-size levels between $(p.chip_size_min.X)x$(p.chip_size_min.Y) " *
                            "and $(p.chip_size_max.X)x$(p.chip_size_max.Y)"
    _check_measures(p, length(sizes))
    return sizes
end

# Restrict a level's search to where the coarse pass found coherent motion, by zeroing the radius
# everywhere else. This is what the coarse pass is *for*, and the reason radius is a per-point
# field. Returns how many points remain searchable, so a caller can stop when none do.
function _apply_coarse_mask!(pts::PointSet{2}, mask::AbstractMatrix{Bool})
    @inbounds for i in eachindex(pts)
        if !mask[i]
            pts.radius_x[i] = 0
            pts.radius_y[i] = 0
        end
    end
    return nsearchable(pts)
end

# ---------------------------------------------------------------------------

"""
    AutoRIFT.WholeScene(prepared)

Correlate each pass in one call, over a whole filtered scene. See [`AutoRIFT.PassRunner`](@ref).

`prepared` is the **filtered** pair — the pair `_prepare` produced. The blocked runner takes an
unfiltered one instead, and holding the pair here is what keeps the two from being confused.

`okmask` is [`valid`](@ref) of that pair: the pixels usable in both images. Held rather than
recomputed because it is a property of the pair, which does not change within a run, while
[`track!`](@ref) needs it once per pass — six times over a three-level run. At a full Landsat
scene the intersection is 35 MiB a call, so recomputing it is ~208 MiB of churn for an answer
already in hand. `Blocked` needs no equivalent: its blocks read mask windows into dense buffers
per block, so there is no whole-scene mask to hold.
"""
struct WholeScene{P<:ImagePair,M<:AbstractMatrix{Bool}} <: PassRunner
    prepared::P
    okmask::M
end

# The mask derived from the pair, so every existing construction keeps working and no caller has to
# know the field exists. `valid` is the one definition of the intersection either way.
WholeScene(prepared::ImagePair) = WholeScene(prepared, valid(prepared))

run_pass(r::WholeScene, pts::PointSet{2}, p::Params, measure::SimilarityMeasure,
         subpixel::SubpixelMethod) =
    track(r.prepared, pts, p; subpixel, measure, okmask = r.okmask)

# A whole-scene pass sees every point of whatever set it is given, so a reshaped grid needs no
# change here: the subset is already expressed in the `PointSet` handed to `run_pass`. Only a
# *partitioned* runner has to re-derive its partition, which is what `Blocked`'s method does.
#
# Returning `r` unchanged is why a missing `restrict` is invisible in a whole-scene run and a
# `BoundsError` in a blocked one — see `AutoRIFT.PassRunner`.
restrict(r::WholeScene, _setup, _gridsize::Tuple{Int,Int}) = r

# Whether to warn when the coarse grid is too small to judge consistency on.
#
# Silent for a whole-scene run and loud for a blocked one, and the asymmetry is about what the
# situation implies rather than about the cost, which is the same either way. A small grid is a
# routine thing to ask a whole scene for — a modest area at a coarse spacing reaches it, and the
# tests do. A blocked run, by contrast, is only asked for when the scene is too large to hold, so a
# coarse grid too small to filter means the configuration is inconsistent with the reason for
# blocking at all.
_warn_coarse_fallback(::WholeScene, ::Extent) = nothing

"""
    correlate_multichip(pair, grid, p) -> MultichipResult

Correlate `pair` over `grid` at every chip-size level, finest chip first.

`grid` must be a gridded [`PointSet`](@ref) — the coarse pass decimates it and the merge
resamples between levels, both of which need a spatial layout. Its `chip_size_x` field is
ignored: the level sets it.

Each level runs [`chipsize_level`](@ref) and contributes only where no finer level already
succeeded, so the smallest chip that yields a coherent estimate wins at every point.

`pair` is the **filtered** pair. [`AutoRIFT.correlate_tiled`](@ref) runs the same loop over an
unfiltered one, filtering per block; both go through `_multichip` so the level sequence exists once.
"""
correlate_multichip(pair::ImagePair, grid::PointSet{2}, p::Params) =
    _multichip(WholeScene(pair), grid, p)

# The chip-size loop, however its passes are executed. One loop, not one per driver, for the same
# reason `chipsize_level` is one function: the two must agree bit for bit, and shared code is what
# makes that structural rather than something to re-verify.
function _multichip(runner::PassRunner, grid::PointSet{2}, p::Params)
    sizes = _level_sizes(p)
    result = _empty_result(size(grid))

    # `eachindex` and two index reads rather than `zip` over a `Vector` and a tuple: `zip` of a
    # `Vector` with anything is `Iterators.Zip{<:Tuple{Vector,Vararg}}`, which `--trim` cannot
    # resolve. Indexing is also how the measure keeps its concrete type — `p.similarity[k]` is
    # typed by the tuple, where an iterated element would not be.
    for k in eachindex(sizes)
        cs = sizes[k]
        # Only points still without a value are candidates, so a level's work shrinks as the
        # finer ones succeed.
        wanted = result.chip_size .== 0
        any(wanted) || break                     # every point resolved
        # `result.dx`/`result.dy` are passed in as the finer levels' standing answer, which a
        # decimated level fills its holes from before interpolating — see `_undecimate_level`.
        level = chipsize_level(runner, grid, p, cs, wanted, measure_at(p, k), result.dx, result.dy)
        isnothing(level) && continue              # level found nothing coherent
        _merge_level!(result, level.field, level.filled, cs)
    end
    return result
end

"""
    chipsize_level(pair, grid, p, chip_size, wanted) -> Union{Nothing,NamedTuple}

One chip-size level: a coarse pass to find where motion is coherent, then a fine pass
restricted to that neighbourhood, then hole filling.

Returns `(; field, filled)` — the displacement field, and the linear indices that were filled
from neighbours rather than measured — or `nothing` if the level found nothing worth
continuing.

`measure` is the similarity measure for this level. Positional rather than a keyword because
`p.similarity` is a tuple and a keyword carrying an abstract `SimilarityMeasure` is unresolvable
under `--trim` — see [`measure_at`](@ref).

`wanted` marks the points this level should attempt — in the loop, those no finer level
resolved. Returns `nothing` if the coarse pass found too little to be worth continuing,
which is what `min_coarse_valid_fraction` decides.

`prior_dx`/`prior_dy` are the finer levels' standing answer over `grid`, which a decimated level
fills its own holes from before interpolating back up — see [`AutoRIFT.correlate_multichip`](@ref).
Omitting them fills from the level's own neighbours alone, which is right for a level run in
isolation and wrong inside the loop, where a finer answer at that very point is available.

Separately callable so a caller can run one chip size without the loop.

`chip_size` is an [`AutoRIFT.Extent`](@ref); a scalar is accepted and means square.

`pair` is the **filtered** pair, which this wraps in an [`AutoRIFT.WholeScene`](@ref) runner. The
[`AutoRIFT.PassRunner`](@ref) form below is what a blocked run uses, with the same body.
"""
chipsize_level(pair::ImagePair, grid::PointSet{2}, p::Params, chip_size,
               wanted::AbstractMatrix{Bool},
               measure::SimilarityMeasure = first(p.similarity),
               prior_dx::Union{Nothing,AbstractMatrix} = nothing,
               prior_dy::Union{Nothing,AbstractMatrix} = nothing) =
    chipsize_level(WholeScene(pair), grid, p, extent(chip_size), wanted, measure,
                   prior_dx, prior_dy)

# One chip-size level, however its passes are executed.
#
# There is one sequence of steps here, not one per driver, and that is the point: a whole-scene run
# and a blocked run must agree bit for bit, and the cheapest way to guarantee they take the same
# steps is for there to be only one place the steps are written. What genuinely differs is pass
# execution, which is `run_pass`, `restrict` and `_warn_coarse_fallback` — three methods rather than
# a second copy of this function.
#
# `runner` is positional: a keyword carrying an abstract type is unresolvable under `--trim`.
#
# `chip_size` is an `Extent` and not a scalar here, because the only caller is the level loop, which
# already has one from `chip_sizes`. The public `ImagePair` form above is where a scalar is accepted.
function chipsize_level(runner::PassRunner, grid::PointSet{2}, p::Params,
                        chip_size::Extent, wanted::AbstractMatrix{Bool},
                        measure::SimilarityMeasure,
                        prior_dx::Union{Nothing,AbstractMatrix} = nothing,
                        prior_dy::Union{Nothing,AbstractMatrix} = nothing)
    # The grid this level runs on. A chip wider than the finest one gets a proportionally coarser
    # grid, so every level posts one estimate per chip rather than several per chip.
    decim = _level_decimation(p, chip_size)
    decim == 1 && return _level_on_grid(runner, grid, p, chip_size, wanted, measure)

    sub = _decimate_level(grid, wanted, decim)
    isnothing(sub) && return nothing
    # `restrict` before the pass: `sub.grid` is a different shape from `grid`, and a `PassRunner` may
    # hold state indexed by grid shape — `Blocked` holds a partition of grid index ranges, which
    # indexed into the decimated grid would be a `BoundsError`. See `AutoRIFT.PassRunner`.
    #
    # `sub.rows`/`sub.cols` are indices into `grid`, which is the space this runner's blocks are in,
    # so the restriction lands in the right place. The level's own coarse pass then restricts the
    # result again, in the decimated space — which composes because `restrict` derives from the
    # runner's current blocks rather than from the full layout.
    got = _level_on_grid(restrict(runner, sub, size(grid)), sub.grid, p, chip_size,
                         sub.wanted, measure)
    isnothing(got) && return nothing
    return _undecimate_level(got, size(grid), sub.rows, sub.cols, prior_dx, prior_dy)
end

# One level on the grid it was handed, without regard to whether that grid was decimated. Split out
# so the decimated and undecimated paths run identically — the two must not drift.
#
# `runner` must already match `grid`'s shape: this does not `restrict`, because it cannot tell
# whether its caller decimated. The caller owns that.
function _level_on_grid(runner::PassRunner, grid::PointSet{2}, p::Params,
                        chip_size::Extent, wanted::AbstractMatrix{Bool},
                        measure::SimilarityMeasure)
    # A level's points: the requested ones, at this level's chip size.
    pts = _level_points(grid, p, chip_size, wanted)
    nsearchable(pts) == 0 && return nothing

    coarse_mask = _coarse_mask(runner, pts, p, chip_size, measure)
    isnothing(coarse_mask) && return nothing
    _apply_coarse_mask!(pts, coarse_mask) == 0 && return nothing

    fine = run_pass(runner, pts, p, measure, p.subpixel)
    filled = _reject_and_fill!(fine, pts, p)
    return (; field = fine, filled)
end

# How much coarser this level's grid is than the caller's, as an integer stride.
#
# The reference resizes its grid by `ChipSize0X / ChipSizeUniX[i]` at every level
# (`autoRIFT.py:507-524`) and resizes the results back afterwards (`820-878`), so a level's grid
# spacing grows with its chip and the chip-to-spacing ratio is the same at every level. That is what
# makes one filter width correct throughout, and what keeps a level from posting sixteen estimates
# per chip footprint — sixteen views of mostly the same pixels, which no coherence filter can tell
# apart.
#
# A stride rather than a resize: the levels are powers of two of the base chip, so the coarse grid is
# exactly every `n`-th point of the fine one, and taking a subset keeps the coordinates the caller
# supplied instead of interpolating new ones.
#
# Derived from the *grid spacing*, not from `chip_size_min`. The invariant to hold is that every
# level sees the same chip-to-spacing ratio, so the stride is whatever makes this level's effective
# spacing proportional to its chip: `chip / (ratio * spacing)`, where the ratio is the finest
# level's. Defining it against `chip_size_min` instead gives a stride of 1 whenever a single coarse
# level runs alone — `chip_size_min` is then that same coarse size — and the ratio jumps to 8, where
# the filter demands 877 of 1089 neighbours agree and nothing survives.
function _level_decimation(p::Params, chip_size::Extent)
    ratio = _oversample(p)
    sx = chip_size.X ÷ max(ratio * p.grid_spacing.X, 1)
    sy = chip_size.Y ÷ max(ratio * p.grid_spacing.Y, 1)
    return max(min(sx, sy), 1)
end

# One point per `stride`-by-`stride` cell of `grid`, placed at the cell's centre.
#
# The centre, not the cell's first point. `_undecimate_level` reads a coarse node back with the
# half-sample convention, which places node `k` at fine position `(k - 0.5) * stride + 0.5` — the cell
# centre. Correlating at the cell's first point instead measures the field half a cell away from where
# every consumer then assumes it was measured, and a displacement field varies over that distance: on a
# Jakobshavn pair the median change across one grid cell is 0.14 px, and the disagreement with the
# reference at the coarse levels was the same 0.13-0.14 px until the two positions were made to
# coincide. The offset grows with `stride`, so it shows up as a level-dependent error that no single
# shift of the output can correct.
#
# The centre falls on a half-integer index for even `stride`, which is why this interpolates the
# coordinates rather than taking a subset: `x` and `y` are `Float64` positions, so a node between two
# fine points is representable, and the reference reaches the same place by resizing its coordinate
# arrays with `INTER_AREA` (`autoRIFT.py:507-524`), which averages each cell and so also lands at the
# centre.
#
# `wanted` is reduced by a maximum over each coarse cell rather than sampled at its centre, which is
# what the reference's `colfilt(..., 0)` does (`autoRIFT.py:534-539`). Sampling would let one fine
# point's state decide for every point in its cell, dropping a region whose sampled node happens to
# be resolved.
#
# `nothing` when the result is too small to filter, the same condition `_coarse_points` applies.
function _decimate_level(grid::PointSet{2}, wanted::AbstractMatrix{Bool}, stride::Int)
    nr, nc = size(grid)
    rows = 1:stride:nr
    cols = 1:stride:nc
    (length(rows) < 3 || length(cols) < 3) && return nothing
    # A coarse point is attempted where any fine point within six coarse cells still wants one,
    # which is the reference's `colfilt(M0, (6/Scale, 6/Scale), 0)` — a maximum, so a dilation of the
    # wanted mask, over six of this level's own cells (`autoRIFT.py:534-539`). Six and not one: the
    # reference's `colfilt(..., 0)` is a maximum, i.e. a dilation of the wanted mask, and taking it
    # over the cell is what stops a sampled node's own state from deciding for its fifteen
    # neighbours. Dilating the *resolved* mask instead — the `logical_not` reading — skips a coarse
    # point whenever anything within `6 * stride` is resolved, which after a successful finest level
    # is everywhere: measured as 259 of 3922 searchable points surviving, and no coarse level
    # reaching the merge at all.
    want = windowmax(map(w -> w ? 1.0f0 : 0.0f0, wanted), 6 * stride)
    keep = [want[i, j] > 0.5f0 for i in rows, j in cols]
    return (; grid = _cell_centres(grid[rows, cols], grid, rows, cols, stride),
            wanted = keep, rows, cols)
end

# `sub`, whose points are the first of each cell of `full`, moved to the centre of the cell each covers.
#
# Only `x` and `y` move. Every other field is a property of the cell — its radius, its prior, its chip
# size — and is already the value this level will use.
#
# The centre, and specifically the centre of the cell as this grid indexes it, because
# `_undecimate_level` reads node `k` back from fine position `(k - 0.5) * stride + 0.5` — the same
# place. Correlating anywhere else measures the field somewhere other than where every consumer then
# assumes, and a displacement field varies over that distance.
#
# The reference's own two halves do *not* coincide here, and matching either one alone is worse than
# matching neither. Instrumenting it on a Jakobshavn pair, the grid it hands each level places node `k`
# a further `stride - 1` cells along than the cell centre, while its `cv2.resize` read-back stays on the
# plain size ratio — so its coarse values land `stride - 1` cells from where they were measured, exactly
# the offset its own read-back carries. Moving our nodes to its correlation positions while keeping a
# consistent read-back took the coarse levels from 1% of points beyond 0.2 px to 39%. Self-consistency
# between the two halves is what the accuracy depends on, not agreement with either half separately.
#
# Per cell, from the rows and columns that cell actually spans, rather than a uniform half-`stride`
# shift. `nr` need not be a multiple of `stride`, so the last cell along an axis can be short, and
# shifting it by a full half-cell puts it past the last grid point: `gridpoints` insets the grid by
# exactly the correlation reach, so a point beyond it no longer fits the unpadded image and the pass
# silently switches to the zero-padded path. The displacements come out the same, but the surface is
# computed by a different transform and its peak height differs in the last bits — enough that a blocked
# run stops matching an untiled one exactly.
#
# The shift is in the grid's own coordinates, taken from `full`'s spacing rather than assumed: a
# `PointSet` carries positions in image pixels, and the grid spacing need not be 1.
#
# No rounding to a pixel lattice here, and the reference's `round(x + 0.5) - 0.5` at
# `autoRIFT.py:525-530` is not a missing step. That snap restores the half-pixel convention its
# `INTER_AREA` grid resize destroys — the reference's `xGrid` already carries `+ 0.5` baked in from
# `runAutorift`, and averaging an even number of such coordinates lands back on an integer. This
# decimates by taking every `stride`-th point and shifting by whole grid cells, so the convention
# survives untouched, and `_shift_points` applies the `+ 0.5` at correlation time for every level
# alike. Snapping on top of that would move a coarse centre half a pixel off the lattice the finest
# level uses.
function _cell_centres(sub::PointSet{2}, full::PointSet{2}, rows, cols, stride::Int)
    stride == 1 && return sub
    # From the first two points along each axis, so a non-square spacing shifts by the right amount on
    # each. A single-point axis cannot happen: `_decimate_level` requires at least three coarse points.
    nr, nc = size(full)
    sx = nc > 1 ? full.x[1, 2] - full.x[1, 1] : 0.0
    sy = nr > 1 ? full.y[2, 1] - full.y[1, 1] : 0.0
    # Half the span of this cell, which is `stride` points except where the grid ran out.
    halfx = [(min(c + stride - 1, nc) - c) / 2 for c in cols]
    halfy = [(min(r + stride - 1, nr) - r) / 2 for r in rows]
    return rebuild(sub; x = sub.x .+ sx .* reshape(halfx, 1, :),
                        y = sub.y .+ sy .* halfy)
end

# `A` with every `NaN` replaced by its nearest finite neighbour, so an interpolant reading a
# neighbourhood never sees one. Only the values are affected; which points are valid is decided by a
# mask the caller keeps, so filling here cannot invent a measurement.
#
# Nearest rather than a smooth extrapolation: this exists to keep a hole from poisoning its
# neighbours, not to estimate anything, and the filled values survive only where the mask allows.
# One multi-source breadth-first sweep, not a search per hole. Every finite cell is a source; the
# frontier grows by one 8-neighbourhood step per level, so the level at which a hole is reached is its
# Chebyshev distance to the nearest finite cell and the value it takes is that neighbour's. Cost is one
# pass over the array however far the holes are from data.
#
# Per-hole ring searching is the obvious alternative and it is unusable here: its cost per hole grows
# with that hole's distance to data, so a scene whose ocean and cloud are contiguous NaN regions
# thousands of cells across takes minutes. Measured on a 512² field with 40% NaN in two blocks —
# the shape a real scene has — 110.5 s against 2.2 ms, for bit-identical output. The sweep is ~2 ms
# slower when every hole is isolated and already touching data, which is the case that was never the
# problem.
function _fill_nan_nearest(A::AbstractMatrix{Float32})
    out = copy(A)
    any(isnan, out) || return out
    nr, nc = size(out)
    # `settled` marks a cell whose value is final: finite in `A`, or already taken from a donor.
    settled = falses(nr, nc)
    # A plain vector as the queue, sized once: each cell is enqueued at most once, so it cannot grow.
    queue = Vector{Int}(undef, length(A))
    head, tail = 1, 0
    @inbounds for k in eachindex(A)
        if !isnan(A[k])
            settled[k] = true
            queue[tail += 1] = k
        end
    end
    # Nothing finite anywhere, so there is no neighbour to take: leave the `NaN`s as they are rather
    # than invent a value. The caller's mask discards them either way.
    tail == 0 && return out
    lin = LinearIndices(out)
    car = CartesianIndices(out)
    @inbounds while head <= tail
        k = queue[head]
        head += 1
        i, j = Tuple(car[k])
        v = out[k]
        for dj in -1:1, di in -1:1
            (di == 0 && dj == 0) && continue
            ii, jj = i + di, j + dj
            (1 <= ii <= nr && 1 <= jj <= nc) || continue
            kk = lin[ii, jj]
            settled[kk] && continue
            settled[kk] = true
            out[kk] = v
            queue[tail += 1] = kk
        end
    end
    return out
end

# A coarse level's values with every hole filled, in the order the reference fills them
# (`autoRIFT.py:823-844`): the finer levels' standing answer at that same place, else a median of this
# level's own neighbourhood, else its nearest finite value however far.
#
# The order is what matters, and it is strictly local-first. Each step reaches further than the last,
# so a hole is filled from the closest evidence that covers it and the long reach is used only where
# nothing nearer exists. Filling nearest-first instead lets a single distant value stand in for a
# whole gap: `_fill_nan_nearest` takes whichever finite cell is nearest in Chebyshev distance, and at
# the edge of a fast-moving feature that cell can lie across the discontinuity. Measured beside an
# iceberg on a Jakobshavn pair, nearest-neighbour and the finer level's own value at the same point
# differ by a median 5.5 px, and those filled values dominate the bicubic footprint of every
# destination along the feature's rim.
#
# `prior` is the finer levels' answer already reduced onto *this level's* grid — `_undecimate_level`
# does that once per axis, since both `dx` and `dy` need it. `nothing` means no finer level has run
# yet, which is the case at the first level and for a caller running one level alone; the median and
# nearest steps still apply.
function _fill_level_holes(A::AbstractMatrix{Float32}, prior::Union{Nothing,AbstractMatrix})
    any(isnan, A) || return copy(A)
    nr, nc = size(A)
    seeded = copy(A)
    # The reference's `colfilt(DxF, (5,5), 3)`, gathered per hole rather than swept over the grid with
    # `windowmedian`. Only holes the prior does not already cover need it, and a level is mostly
    # measurements, so sweeping computes a median at every point and discards nearly all of them —
    # 6.8 ms of an 8.1 ms call on a 186x186 level. The same reasoning `_fill_holes!` records.
    buf = Vector{Float32}(undef, 25)
    @inbounds for j in 1:nc, i in 1:nr
        isnan(seeded[i, j]) || continue
        if !isnothing(prior) && isfinite(prior[i, j])
            seeded[i, j] = prior[i, j]
            continue
        end
        n = 0
        for jj in max(j - 2, 1):min(j + 2, nc)
            for ii in max(i - 2, 1):min(i + 2, nr)
                v = A[ii, jj]
                isnan(v) || (buf[n += 1] = v)
            end
        end
        n > 0 && (seeded[i, j] = _select_median!(buf, n))
    end
    return _fill_nan_nearest(seeded)
end

# A decimated level's result, back on the full grid.
#
# `dx`/`dy` interpolate bicubically, matching the reference's `INTER_CUBIC`: they are continuous
# fields, and nearest-neighbour would post a coarse estimate as a blocky plateau that the merge then
# treats as measured everywhere inside it. `correlation` interpolates the same way, since it
# describes the same estimate. `searched` and the filled indices do not interpolate — they are
# categorical, and a fractional "partly searched" has no meaning.
#
# Every resample here is on the *stride* lattice — `step(rows)` source samples per destination sample —
# and not on the size ratio a bare `resample` would infer. `_decimate_level` takes every `stride`-th
# point and `_cell_centres` moves each to its cell's centre, so coarse node `k` stands at fine position
# `(k - 0.5) * stride + 0.5`; the ratio `length(rows) / nr` describes that lattice only when `stride`
# divides `nr`. Where it does not, the inferred scale is slightly too large, the two lattices drift
# apart along the axis, and every coarse value beyond the crossing point is read one fine cell off. At
# `nr = 371, stride = 2` that is 93 of 371 rows misassigned from row 186 outward, which surfaces as
# rings of chip-size disagreement tracing the level boundaries in the far half of the grid.
#
# Matching the reference's own coarsening more literally — `fld` nodes read back on the size ratio,
# as `cv2.resize` does — is *worse*, not better, and measurably: the reference's nodes come from an
# `INTER_AREA` resize whose cell centres creep away from any fixed stride, so pairing a truncated node
# count with a strided slice describes neither lattice. Measured on a Jakobshavn pair it took the
# coarse levels from a median disagreement of 0.000 px to 0.068 and from 1% of points beyond 0.2 px to
# 13%. The stride lattice is self-consistent, and self-consistency is what the interpolation needs.
#
# `prior_dx`/`prior_dy` are the finer levels' standing answer on the full grid. This reduces them onto
# the level's own grid and `_fill_level_holes` fills from them; see there for why that preference
# matters.
function _undecimate_level(got, fullsize::Tuple{Int,Int}, rows, cols,
                           prior_dx::Union{Nothing,AbstractMatrix} = nothing,
                           prior_dy::Union{Nothing,AbstractMatrix} = nothing)
    field = got.field
    out = DisplacementField(fill(NaN32, fullsize), fill(NaN32, fullsize),
                            fill(NaN32, fullsize), fill(NaN32, fullsize),
                            fill(false, fullsize))
    sc = (Float64(1 / step(rows)), Float64(1 / step(cols)))
    # The prior onto this level's grid: a `step(rows) + 1` wide NaN-aware mean filter, then an
    # `INTER_AREA` resize, both the reference's (`autoRIFT.py:823-839`). The mean filter is not
    # redundant with the area average that follows it — its window is one wider than a cell, so it
    # reaches across each cell boundary and closes a partly-missing neighbourhood before the area step
    # weights what remains. Resizing the raw field instead reproduces the reference's `DxF0` at 1.6% of
    # points against 100% with it, and the prior fills the majority of a coarse level's holes.
    reduce_prior(P) = isnothing(P) ? nothing :
                      resample(windowmean(P, step(rows) + 1), size(field.dx), Area();
                               scale = (Float64(step(rows)), Float64(step(cols))))
    # Bicubic interpolation reads a 4x4 neighbourhood, so a single `NaN` in it poisons every
    # destination it touches — enough to erase all but the interior of a sparse field. The values
    # are interpolated over a hole-free copy, and validity is carried separately by the mask: a
    # destination is kept only where the mask says a measurement stands behind it. Interpolating
    # the raw field instead loses everything within two coarse cells of any hole, which measured as
    # 11,679 coarse values reaching exactly 11,679 fine points where a stride-2 level covers four
    # times that area.
    # Filled only so the interpolant never reads a `NaN`; the mask below decides what survives.
    resample!(out.dx, _fill_level_holes(field.dx, reduce_prior(prior_dx)), Bicubic(); scale = sc)
    resample!(out.dy, _fill_level_holes(field.dy, reduce_prior(prior_dy)), Bicubic(); scale = sc)
    resample!(out.correlation, _fill_nan_nearest(field.correlation), Bicubic(); scale = sc)
    # `Nearest` for the peak quality, where the displacement gets `Bicubic`. Interpolating it would
    # invent a quality between two coarse cells, and a quality is a statement about one specific
    # correlation surface — there is no surface between them to make the interpolated value a
    # statement about. Every fine point in a coarse cell therefore reports that cell's value
    # verbatim, the same treatment `chip_size` receives for the same reason.
    resample!(out.peak_snr, _fill_nan_nearest(field.peak_snr), Nearest(); scale = sc)
    # Keyed on `dx` being finite rather than on `searched`: a point can be searched and yield
    # nothing, and only a real measurement may be spread over its cell. `searched` would post a
    # coarse estimate across every hole the level attempted and failed at — measured as 13,118
    # searched nodes claiming 52,033 fine points and the correlation against the control collapsing
    # from 0.996 to 0.841.
    valid = resample(map(v -> isnan(v) ? 0.0f0 : 1.0f0, field.dx), fullsize, Nearest(); scale = sc)
    filled = Int[]
    wasfilled = falses(size(field.dx))
    @inbounds for idx in got.filled
        wasfilled[idx] = true
    end
    up = resample(map(f -> f ? 1.0f0 : 0.0f0, wasfilled), fullsize, Nearest(); scale = sc)
    @inbounds for i in eachindex(out.dx)
        if valid[i] > 0.5f0 && isfinite(out.dx[i]) && isfinite(out.dy[i])
            out.searched[i] = true
            up[i] > 0.5f0 && push!(filled, i)
        else
            out.dx[i] = NaN32
            out.dy[i] = NaN32
            out.correlation[i] = NaN32
            out.peak_snr[i] = NaN32
            out.searched[i] = false
        end
    end
    return (; field = out, filled)
end

# How many grid points span one chip of the finest level, per axis, taken as the smaller of the
# two so a non-square configuration does not over-widen either axis of the filter window.
#
# Only a grid at least as fine as its chips has overlapping neighbourhoods, so a spacing coarser
# than the chip gives one and the filter is left alone. Integer division matches the reference,
# which rejects a non-dividing pair outright (`autoRIFT.py:474-482`) rather than rounding.
function _oversample(p::Params)
    ox = p.chip_size_min.X ÷ max(p.grid_spacing.X, 1)
    oy = p.chip_size_min.Y ÷ max(p.grid_spacing.Y, 1)
    # Capped at 2. The ratio is how many grid points span one chip, and the filter's window and
    # agreement fraction both grow with it: at 4 the threshold is 186 of 289 neighbours and at 8 it
    # is 877 of 1089, which no real velocity field clears — measured as zero coverage for a 64 px
    # chip on a grid spaced 8. The reference never exceeds 2 because it decimates every level to
    # keep the ratio fixed, so the formula it uses (`autoRIFT.py:498-502`) was only ever exercised
    # there. A caller who posts a grid four times finer than its chips gets the same neighbourhood
    # in ground units that the reference would use, and `_level_decimation` does the rest.
    return clamp(min(ox, oy), 1, 2)
end

# Points for one level: the caller's grid with this level's chip size, and the radius zeroed
# wherever the level should not attempt a point.
#
# A point is attempted only when the level's chip size lies within that point's own
# `chip_size_min_x`/`chip_size_max_x`, matching the reference's per-level mask
# (`autoRIFT.py:534`). Zero in either means unbounded, which is the default, so a grid that
# carries no bounds admits every level exactly as it did before the fields existed.
#
# The bound earns its place on real data. ITS_LIVE's parameter rasters permit a 960 m chip at
# 1.7% of points over Jakobshavn, and running it everywhere instead produces estimates from a
# chip far larger than the ice structure it covers — measured at a 0.73 correlation against the
# reference where the finest level reaches 0.99.
function _level_points(grid::PointSet{2}, p::Params, chip_size::Extent,
                       wanted::AbstractMatrix{Bool})
    n = size(grid)
    rx = Matrix{Int}(undef, n)
    ry = Matrix{Int}(undef, n)
    @inbounds for i in eachindex(grid)
        lo, hi = grid.chip_size_min_x[i], grid.chip_size_max_x[i]
        permitted = (lo == 0 || chip_size.X >= lo) && (hi == 0 || chip_size.X <= hi)
        if wanted[i] && permitted
            rx[i] = grid.radius_x[i]
            ry[i] = grid.radius_y[i]
        else
            rx[i] = 0
            ry[i] = 0
        end
    end
    # Coordinates and priors are shared rather than copied: nothing below writes them, and
    # only the radii (here and by the coarse mask) and the chip sizes are level-specific.
    pts = rebuild(grid; radius_x = rx, radius_y = ry,
                  chip_size_x = fill(chip_size.X, n), chip_size_y = fill(chip_size.Y, n))
    sanitize!(pts, p.min_search_radius)
    return pts
end

# ---------------------------------------------------------------------------
# Coarse pass
# ---------------------------------------------------------------------------

# The coarse pass splits into two halves, and the seam is load-bearing rather than cosmetic.
#
# **Correlating** the decimated points is per-point work: each coarse point's displacement depends
# only on the imagery around it, so a caller processing a scene in blocks can produce this evidence
# a block at a time and scatter it into one coarse-grid array.
#
# **Deciding** from that evidence is not. `reject_outliers` compares a point against its
# neighbourhood, the `min_coarse_valid_fraction` gate is a ratio over the whole coarse grid, and
# `dilate_within` grows across it — so all three need the *assembled* grid. Run per block they would
# each answer a different question: a block over open water would fail the gate and drop a whole
# chip-size level for itself alone, reported as "level found nothing coherent" and indistinguishable
# from a real result.
#
# Splitting them is what lets the halo be the correlation reach alone. The alternative — a halo
# covering the filter and the dilation — reaches 1792 px at defaults, which is more overlap than
# data at any block size worth asking for. See `docs/plan-tiling.md`.
#
# The coarse grid is affordable to hold whole even when the imagery is not: at the default spacing
# and stride it is about 1/1024 the size of the scene.

# How far apart the coarse pass's sample points sit, in points of the level's own grid.
#
# `coarse_stride` is a sampling *rate* against one point per chip, not a step in grid points, so on a
# grid posted finer than its chips the step is the rate times that ratio — the reference's
# `sparseSearchSampleRate * ChipSize0_GridSpacing_oversample_ratio` (`autoRIFT.py:605-614`).
#
# The distinction is load-bearing because `coarse_buffer` is measured in coarse cells and expanded
# back by this same factor: a step of `coarse_stride` alone makes the buffer reach half as far across
# the level grid as the reference's, which silently narrows the region each level is willing to
# search. Where motion is incoherent the effect is a whole neighbourhood that a coarser chip then
# claims instead — on a Jakobshavn pair, 8 buffer cells reached 32 level points against the
# reference's 64.
#
# `coarse_stride` still carries the rate itself into `rescale`'s overlap term, which is written
# against the rate rather than against the step.
_sparse_stride(p::Params) = p.coarse_stride * _oversample(p)

# Which points the coarse pass correlates, and where they sit on the full grid.
#
# `nothing` when the coarse grid is too small for the filter to judge consistency on — the caller
# decides what that means, because the two callers differ: a whole-scene run falls back to searching
# everything rather than rejecting on no evidence, while a block-at-a-time run must treat it as a
# configuration error, since per block it would silently skip the coarse restriction entirely.
function _coarse_points(pts::PointSet{2}, p::Params, chip_size::Extent)
    stride = _sparse_stride(p)
    nr, nc = size(pts)
    rows = stride:stride:nr
    cols = stride:stride:nc
    # Two adjustments, both the reference's (`autoRIFT.py:484-496`). `relax` loosens for the
    # decimation: a coarse point's neighbours span several times more ground and are genuinely
    # less like it. `rescale` matches the neighbourhood to a grid finer than its chips.
    #
    # `coarse_stride` and not `stride` here: the overlap term is the reference's
    # `1 - sparseSearchSampleRate / ratio`, which is written against the sampling *rate* rather than
    # against the grid step the rate produces. See `_sparse_stride`.
    filt = rescale(relax(p.outliers), _oversample(p), p.coarse_stride)
    w = window(filt)
    (length(rows) < w || length(cols) < w) && return nothing

    # The coarse point's radius must cover its whole cell, since it stands in for every fine
    # point inside it — hence the max over the cell rather than a sample of one point.
    #
    # Computed per coarse cell rather than by a full-grid sliding max that is then decimated:
    # the latter discards fifteen sixteenths of its work at stride 4, and measured 113 us and
    # 405 KB per level against 7.9 us here. It also stays in `Int` throughout, where the
    # sliding form needed a Float32 round trip in each direction.
    coarse = pts[rows, cols]
    _cell_max_radius!(coarse.radius_x, pts.radius_x, rows, cols, stride)
    _cell_max_radius!(coarse.radius_y, pts.radius_y, rows, cols, stride)
    fill!(coarse.chip_size_x, chip_size.X)
    fill!(coarse.chip_size_y, chip_size.Y)
    return (; coarse, rows, cols, filt)
end

# The whole-grid half: from coarse displacements to a full-grid mask of where the fine pass should
# look. `nothing` when the level is not worth continuing.
#
# Every operation here is a neighbourhood or a whole-grid reduction, which is why this must see the
# assembled coarse grid rather than one block of it.
function _coarse_decide(cd::DisplacementField, coarse::PointSet{2}, p::Params,
                        filt::OutlierMethod, gridsize::Tuple{Int,Int}, stride::Int)
    measured = map(!isnan, cd.dx)
    any(measured) || return nothing

    keep = reject_outliers(cd.dx, cd.dy, coarse.radius_x, coarse.radius_y,
                           measured, upsampling(p.subpixel), filt)
    @inbounds for i in eachindex(keep)
        measured[i] || (keep[i] = false)
    end

    # Was enough of the coarse grid coherent? Judged only over points that were searchable at
    # all, so a level is not penalised for the points a previous level already resolved.
    denom = 0
    numer = 0
    @inbounds for i in eachindex(keep)
        coarse.radius_x[i] > 0 || continue
        measured[i] || continue
        denom += 1
        keep[i] && (numer += 1)
    end
    denom == 0 && return nothing
    (numer / denom) < p.min_coarse_valid_fraction && return nothing

    # A coherent coarse estimate is evidence about its neighbourhood, not just its own point,
    # so grow it before it restricts the fine search.
    grown = dilate_within(keep, p.coarse_buffer)
    # Back to the full grid, on the same lattice the radii were reduced over.
    return _expand_coarse_mask(grown, gridsize, stride)
end

# A coarse-grid mask on the full grid, inverting the cell assignment `_cell_max_radius!` makes.
#
# Not `resample(..., Nearest())`: that derives its own cell boundaries from the size ratio, and for
# an even `stride` they are not the ones the radii were reduced over. At `nr = 13, stride = 4` the
# coarse point for fine row 4 covers rows 2..5 here, where `Nearest` assigns rows 1..4 to it — so
# the mask restricting the fine search would sit one row off the evidence that produced it. Two
# independent derivations of one correspondence is the defect; the offset is the symptom.
#
# Inverting `_cell_max_radius!` instead makes the two agree by construction: it centres cell `k` on
# fine index `k * stride` with `lo = stride ÷ 2` before and `hi = stride - 1 - lo` after, so the cell
# containing fine index `i` is `fld(i + lo, stride)`, clamped to the ends where the coarse grid stops
# short of the fine one. That also keeps this in integer arithmetic, where `resample` needed a
# `Float32` round trip and a `> 0.5f0` threshold on a Boolean quantity.
function _expand_coarse_mask(mask::AbstractMatrix{Bool}, gridsize::Tuple{Int,Int}, stride::Int)
    nr, nc = gridsize
    sr, sc = size(mask)
    # The same left margin `_cell_max_radius!` reduces over, from the same helper. This is the
    # inverse map — fine index to cell — and it agrees with the forward one only if both take their
    # margin from one place; `test/multichip.jl` pins the round trip.
    lo, _, _, _ = _window_margins(stride, stride)
    out = Matrix{Bool}(undef, nr, nc)
    @inbounds for j in 1:nc
        cj = clamp(fld(j + lo, stride), 1, sc)
        for i in 1:nr
            out[i, j] = mask[clamp(fld(i + lo, stride), 1, sr), cj]
        end
    end
    return out
end

# Correlate a decimated sample of the points, filter for spatial consistency, and dilate what
# survives. Returns a full-grid mask of where the fine pass should look, or `nothing` if too
# little of the coarse grid was coherent for the level to be worth continuing.
#
# The composition of the two halves above, and the only place they are composed — a blocked run takes
# this same path, differing only in the runner it carries.
#
# `measure` is positional and has no default: the one caller always knows the level's measure, and a
# default here would be a second spelling of `chipsize_level`'s that could silently diverge from it.
function _coarse_mask(runner::PassRunner, pts::PointSet{2}, p::Params, chip_size::Extent,
                      measure::SimilarityMeasure)
    setup = _coarse_points(pts, p, chip_size)
    if isnothing(setup)
        # Too few coarse points to judge consistency against their neighbours, so there is no
        # evidence to restrict the fine search with. Searching everything is what this has to do
        # either way — a blocked run must agree with a whole-scene one point for point rather than
        # substitute a policy of its own — and only whether it says so differs.
        _warn_coarse_fallback(runner, chip_size)
        return trues(size(pts))
    end

    coarse = setup.coarse
    nsearchable(coarse) == 0 && return nothing

    # The per-point half, over the strided subset. Integer peaks only: the coarse pass decides
    # *where* to look, and sub-pixel precision would not change that answer while costing most of
    # the pass. The same measure as the fine pass, since judging coherence by a different one would
    # gate on the wrong thing.
    cd = run_pass(restrict(runner, setup, size(pts)), coarse, p, measure, NoRefine())

    # The whole-grid half, once.
    # The same stride `_coarse_points` sliced with, so the mask expands back onto the lattice the
    # evidence was gathered on.
    return _coarse_decide(cd, coarse, p, setup.filt, size(pts), _sparse_stride(p))
end

# Maximum radius over each coarse cell, matching the left-biased window convention the sliding
# reductions use so the two agree at the boundaries.
function _cell_max_radius!(out, radius, rows, cols, stride::Int)
    nr, nc = size(radius)
    # The cell's extent about its centre, from the one place that convention lives. Writing it out
    # here would be a third transcription of a rule `window.jl` documents as the easiest in the
    # package to get backwards.
    lo, _, hi, _ = _window_margins(stride, stride)
    @inbounds for (jo, j) in enumerate(cols), (io, i) in enumerate(rows)
        m = 0
        for jj in max(j - lo, 1):min(j + hi, nc)
            for ii in max(i - lo, 1):min(i + hi, nr)
                m = max(m, radius[ii, jj])
            end
        end
        out[io, jo] = m
    end
    return out
end

# ---------------------------------------------------------------------------
# Fine pass cleanup
# ---------------------------------------------------------------------------

# Reject inconsistent estimates, then fill small holes from their neighbours. Returns the
# indices that were filled.
#
# Filling is worth doing at this stage rather than at the end: a hole filled here can be
# resampled coherently into the next level's prior, whereas a hole left open forces that
# level to search blind.
function _reject_and_fill!(d::DisplacementField, pts::PointSet, p::Params)
    measured = map(!isnan, d.dx)
    keep = reject_outliers(d.dx, d.dy, pts.radius_x, pts.radius_y,
                           measured, upsampling(p.subpixel),
                           rescale(p.outliers, _oversample(p)))
    @inbounds for i in eachindex(keep)
        if !keep[i]
            d.dx[i] = NaN32
            d.dy[i] = NaN32
            d.correlation[i] = NaN32
            d.peak_snr[i] = NaN32
        end
    end
    return _fill_holes!(d, p)
end

# Fill points surrounded by enough measured neighbours with the neighbourhood median.
#
# Three passes, because each fills only points that are already well surrounded: filling one
# ring makes the next ring well surrounded in turn, so a small hole closes from its edge
# inward while a large one is left alone. A single pass with a looser threshold would instead
# invent values in the middle of genuinely empty regions.
#
# Visits the holes rather than the grid. Sweeping the whole grid with `windowmedian` cost
# 0.76 ms per pass on a 118x118 level with 9% holes — and on that level the first pass fills
# nothing, so the entire cost was discarded. Gathering per hole is ~20x faster there and ~90x
# where fills do happen, because the cost scales with the number of holes rather than with the
# grid.
#
# Returns the indices it filled, so the caller can record which points are interpolated
# rather than deducing it later from a missing correlation.
function _fill_holes!(d::DisplacementField, p::Params)
    w = p.fill_window
    lo, _, hi, _ = _window_margins(w, w)
    # Two thirds of a full window: enough neighbours that the median means something, and
    # strict enough that an isolated point does not seed a fill.
    needed = 2 * w^2 ÷ 3
    nr, nc = size(d.dx)
    bufx = Vector{Float32}(undef, w * w)
    bufy = Vector{Float32}(undef, w * w)
    # `peak_snr` is filled alongside the displacement, so a point carrying a value always carries a
    # quality for it — the same rule `chip_size` follows. The filled quality is the neighbourhood's,
    # which is the honest description: the displacement came from those neighbours, so their peak
    # quality is what stands behind it. `interpolated` marks the point either way, so a caller
    # wanting only directly measured peaks can exclude them.
    #
    # `correlation` is deliberately *not* filled, and stays `NaN` at a filled point. It is the peak
    # value of a surface this point has none of, where the median of neighbouring qualities is a
    # summary that still means something.
    bufs = Vector{Float32}(undef, w * w)
    filled = Int[]

    for _ in 1:3
        # Two-phase: collect this pass's fills before applying any, so every point in a pass
        # sees the same field. Filling in place would let one fill seed the next within a
        # single pass, which is what the three-pass structure exists to control.
        pending = Tuple{Int,Float32,Float32,Float32}[]
        @inbounds for j in 1:nc, i in 1:nr
            isnan(d.dx[i, j]) || continue
            n = 0
            ns = 0
            for jj in max(j - lo, 1):min(j + hi, nc)
                for ii in max(i - lo, 1):min(i + hi, nr)
                    v = d.dx[ii, jj]
                    isnan(v) && continue
                    n += 1
                    bufx[n] = v
                    bufy[n] = d.dy[ii, jj]
                    # Counted separately: a neighbour can carry a displacement with no quality —
                    # it may itself have been filled by an earlier pass, or its surface may have
                    # been too small to characterize — and mixing `NaN` into the selection would
                    # poison the median rather than skip the neighbour.
                    #
                    # `peak_quality`'s zero-at-the-search-boundary carries through by majority, since a
                    # median over values including zeros returns zero once half the neighbourhood is
                    # railed. That is the correct reading: a filled displacement is only as trustworthy
                    # as the neighbourhood behind it.
                    s = d.peak_snr[ii, jj]
                    if !isnan(s)
                        ns += 1
                        bufs[ns] = s
                    end
                end
            end
            n >= needed || continue
            # `_select_median!` rather than a sort here: it picks the cheaper selection for `n`,
            # which at the default 3-wide window is an insertion sort.
            push!(pending, (LinearIndices(d.dx)[i, j],
                            _select_median!(bufx, n), _select_median!(bufy, n),
                            ns > 0 ? _select_median!(bufs, ns) : NaN32))
        end
        isempty(pending) && break        # nothing left that qualifies
        @inbounds for (idx, mx, my, ms) in pending
            d.dx[idx] = mx
            d.dy[idx] = my
            d.peak_snr[idx] = ms
            push!(filled, idx)
        end
    end
    return filled
end

# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

# Write a level's results wherever no finer level already produced one.
#
# The gate is what makes the loop work: every level sees the same grid, and the first to
# succeed at a point owns it. Since levels run finest first, that is the smallest chip that
# could resolve the point.
function _merge_level!(result::MultichipResult, level::DisplacementField,
                      filled::Vector{Int}, chip_size::Extent)
    # The x extent identifies the level on its own: the aspect is constant across levels, so
    # `chip_size_y` is `chip_size_x` times a fixed ratio and recording both would be redundant. One
    # `UInt16` per point rather than two also keeps the result array the size it has always been.
    cs = UInt16(chip_size.X)
    # `filled` comes from the fill step itself rather than being inferred from a missing
    # correlation. The inference would work today, but only because of an invariant spanning
    # three files that nothing asserts — that a measured point always has all three of dx, dy
    # and correlation. Recording the fact where it is known is cheaper than maintaining that.
    wasfilled = falses(size(result.dx))
    @inbounds for idx in filled
        wasfilled[idx] = true
    end
    @inbounds for i in eachindex(result.dx)
        result.chip_size[i] == 0 || continue      # a finer level already owns this point
        isnan(level.dx[i]) && continue
        result.dx[i] = level.dx[i]
        result.dy[i] = level.dy[i]
        result.correlation[i] = level.correlation[i]
        result.peak_snr[i] = level.peak_snr[i]
        result.chip_size[i] = cs
        result.interpolated[i] = wasfilled[i]
    end
    return result
end
