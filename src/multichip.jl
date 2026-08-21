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

`dx`, `dy`, and `correlation` are as in [`DisplacementField`](@ref). `chip_size` records
which level produced each point — `0` where none did — and `interpolated` marks points
filled from their neighbours rather than measured.

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

# ---------------------------------------------------------------------------

"""
    correlate_multichip(pair, grid, p) -> MultichipResult

Correlate `pair` over `grid` at every chip-size level, finest chip first.

`grid` must be a gridded [`PointSet`](@ref) — the coarse pass decimates it and the merge
resamples between levels, both of which need a spatial layout. Its `chip_size_x` field is
ignored: the level sets it.

Each level runs [`chipsize_level`](@ref) and contributes only where no finer level already
succeeded, so the smallest chip that yields a coherent estimate wins at every point.
"""
function correlate_multichip(pair::ImagePair, grid::PointSet{2}, p::Params)
    # `params` already guarantees at least one level: it requires `chip_size <= chip_size_min
    # <= chip_size_max` and that the bounds are power-of-two multiples of `chip_size`, so the
    # sequence always contains `chip_size_min`. Asserted rather than handled, since a caller
    # cannot reach the empty case through the public constructor.
    sizes = chip_sizes(p)
    @assert !isempty(sizes) "no chip-size levels for chip_size $(p.chip_size_base) in " *
                            "[$(p.chip_size_min), $(p.chip_size_max)]"
    # Reject a too-long `similarity` tuple before any correlation runs, rather than discovering it
    # at the level that would have used the extra entry.
    _check_measures(p, length(sizes))

    sz = size(grid)
    result = MultichipResult(fill(NaN32, sz), fill(NaN32, sz), fill(NaN32, sz),
                           zeros(UInt16, sz), falses(sz))

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
        level = chipsize_level(pair, grid, p, cs, wanted, measure_at(p, k))
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

Separately callable so a caller can run one chip size without the loop.
"""
function chipsize_level(pair::ImagePair, grid::PointSet{2}, p::Params,
                       chip_size::Integer, wanted::AbstractMatrix{Bool},
                       measure::SimilarityMeasure = first(p.similarity))
    csx = Int(chip_size)
    csy = chip_size_y(p, csx)

    # A level's points: the requested ones, at this level's chip size.
    pts = _level_points(grid, p, csx, csy, wanted)
    nsearchable(pts) == 0 && return nothing

    coarse_mask = _coarse_pass(pair, pts, p, csx, csy, measure)
    isnothing(coarse_mask) && return nothing

    # The coarse result restricts the fine search: zero the radius outside it. This is the
    # whole point of the coarse pass, and the reason radius is a per-point field.
    @inbounds for i in eachindex(pts)
        if !coarse_mask[i]
            pts.radius_x[i] = 0
            pts.radius_y[i] = 0
        end
    end
    nsearchable(pts) == 0 && return nothing

    fine = track(pair, pts, p; measure)
    filled = _reject_and_fill!(fine, pts, p)
    return (; field = fine, filled)
end

# Points for one level: the caller's grid with this level's chip size, and the radius zeroed
# wherever the level should not attempt a point.
#
# `chip_size_min`/`chip_size_max` are not consulted per point here because `Params` carries
# them as scalars; when they become per-point fields (from Geogrid) this is where that filter
# belongs.
function _level_points(grid::PointSet{2}, p::Params, csx::Int, csy::Int,
                       wanted::AbstractMatrix{Bool})
    n = size(grid)
    rx = Matrix{Int}(undef, n)
    ry = Matrix{Int}(undef, n)
    @inbounds for i in eachindex(grid)
        if wanted[i]
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
                  chip_size_x = fill(csx, n), chip_size_y = fill(csy, n))
    sanitize!(pts, p.min_search_radius)
    return pts
end

# ---------------------------------------------------------------------------
# Coarse pass
# ---------------------------------------------------------------------------

# Correlate a decimated sample of the points, filter for spatial consistency, and dilate what
# survives. Returns a full-grid mask of where the fine pass should look, or `nothing` if too
# little of the coarse grid was coherent for the level to be worth continuing.
# `measure` is positional and has no default: the one caller always knows the level's measure, and a
# default here would be a second spelling of `chipsize_level`'s that could silently diverge from it.
function _coarse_pass(pair::ImagePair, pts::PointSet{2}, p::Params, csx::Int, csy::Int,
                      measure::SimilarityMeasure)
    stride = p.coarse_stride
    nr, nc = size(pts)
    rows = stride:stride:nr
    cols = stride:stride:nc
    # Loosened for the decimated grid: its neighbourhoods span `stride` times more ground, so a
    # coarse point's neighbours are genuinely further away and less like it. `relax` is where
    # each method says what that means for it.
    filt = relax(p.outliers)

    # Too few coarse points to judge consistency against their neighbours: the filter needs a
    # neighbourhood, so fall back to searching everything rather than rejecting on no evidence.
    w = window(filt)
    (length(rows) < w || length(cols) < w) && return trues(nr, nc)

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
    fill!(coarse.chip_size_x, csx)
    fill!(coarse.chip_size_y, csy)
    nsearchable(coarse) == 0 && return nothing

    # Integer peaks only: the coarse pass decides *where* to look, and sub-pixel precision
    # would not change that answer while costing most of the pass.
    # The same measure as the fine pass: the coarse pass decides *where* the fine pass looks, so
    # judging coherence by a different measure than the one that will be used there would gate on
    # the wrong thing.
    cd = track(pair, coarse, p; subpixel = NoRefine(), measure)
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
    # Back to the full grid. Nearest-neighbour because this is a mask: an interpolated value
    # between "search" and "do not search" would mean nothing.
    return resample(grown, (nr, nc), Nearest()) .> 0.5f0
end

# Maximum radius over each coarse cell, matching the left-biased window convention the sliding
# reductions use so the two agree at the boundaries.
function _cell_max_radius!(out, radius, rows, cols, stride::Int)
    nr, nc = size(radius)
    lo = stride ÷ 2
    hi = stride - 1 - lo
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
                           measured, upsampling(p.subpixel), p.outliers)
    @inbounds for i in eachindex(keep)
        if !keep[i]
            d.dx[i] = NaN32
            d.dy[i] = NaN32
            d.correlation[i] = NaN32
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
    lo = w ÷ 2
    hi = w - 1 - lo
    # Two thirds of a full window: enough neighbours that the median means something, and
    # strict enough that an isolated point does not seed a fill.
    needed = 2 * w^2 ÷ 3
    nr, nc = size(d.dx)
    bufx = Vector{Float32}(undef, w * w)
    bufy = Vector{Float32}(undef, w * w)
    filled = Int[]

    for _ in 1:3
        # Two-phase: collect this pass's fills before applying any, so every point in a pass
        # sees the same field. Filling in place would let one fill seed the next within a
        # single pass, which is what the three-pass structure exists to control.
        pending = Tuple{Int,Float32,Float32}[]
        @inbounds for j in 1:nc, i in 1:nr
            isnan(d.dx[i, j]) || continue
            n = 0
            for jj in max(j - lo, 1):min(j + hi, nc)
                for ii in max(i - lo, 1):min(i + hi, nr)
                    v = d.dx[ii, jj]
                    isnan(v) && continue
                    n += 1
                    bufx[n] = v
                    bufy[n] = d.dy[ii, jj]
                end
            end
            n >= needed || continue
            _insertion_sort!(bufx, n)
            _insertion_sort!(bufy, n)
            push!(pending, (LinearIndices(d.dx)[i, j],
                            _sorted_median(bufx, n), _sorted_median(bufy, n)))
        end
        isempty(pending) && break        # nothing left that qualifies
        @inbounds for (idx, mx, my) in pending
            d.dx[idx] = mx
            d.dy[idx] = my
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
                      filled::Vector{Int}, chip_size::Integer)
    cs = UInt16(chip_size)
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
        result.chip_size[i] = cs
        result.interpolated[i] = wasfilled[i]
    end
    return result
end
