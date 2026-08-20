# Multi-scale correlation: the pyramid.
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
    PyramidResult

Displacement over the full grid, assembled from all pyramid levels.

`dx`, `dy`, and `correlation` are as in [`DisplacementField`](@ref). `chip_size` records
which level produced each point — `0` where none did — and `interpolated` marks points
filled from their neighbours rather than measured.

The chip size is worth keeping rather than discarding: it says how much spatial averaging is
behind each estimate, so a downstream consumer can tell a sharply-resolved velocity from a
smoothed one.
"""
struct PyramidResult
    dx::Matrix{Float32}
    dy::Matrix{Float32}
    correlation::Matrix{Float32}
    chip_size::Matrix{UInt16}
    interpolated::BitMatrix
end

Base.size(r::PyramidResult) = size(r.dx)
Base.length(r::PyramidResult) = length(r.dx)

"""
    nmeasured(r::PyramidResult) -> Int

How many grid points carry a displacement.
"""
nmeasured(r::PyramidResult) = count(!isnan, r.dx)

# ---------------------------------------------------------------------------

"""
    correlate_pyramid(pair, grid, p) -> PyramidResult

Correlate `pair` over `grid` at every pyramid level, finest chip first.

`grid` must be a gridded [`PointSet`](@ref) — the coarse pass decimates it and the merge
resamples between levels, both of which need a spatial layout. Its `chip_size_x` field is
ignored: the level sets it.

Each level runs [`pyramid_level`](@ref) and contributes only where no finer level already
succeeded, so the smallest chip that yields a coherent estimate wins at every point.
"""
function correlate_pyramid(pair::ImagePair, grid::PointSet{2}, p::Params)
    # `params` already guarantees at least one level: it requires `chip_size <= chip_size_min
    # <= chip_size_max` and that the bounds are power-of-two multiples of `chip_size`, so the
    # sequence always contains `chip_size_min`. Asserted rather than handled, since a caller
    # cannot reach the empty case through the public constructor.
    sizes = chip_sizes(p)
    @assert !isempty(sizes) "no pyramid levels for chip_size $(p.chip_size_base) in " *
                            "[$(p.chip_size_min), $(p.chip_size_max)]"

    sz = size(grid)
    result = PyramidResult(fill(NaN32, sz), fill(NaN32, sz), fill(NaN32, sz),
                           zeros(UInt16, sz), falses(sz))

    for cs in sizes
        # Only points still without a value are candidates, so a level's work shrinks as the
        # finer ones succeed.
        wanted = result.chip_size .== 0
        any(wanted) || break                     # every point resolved
        level = pyramid_level(pair, grid, p, cs, wanted)
        isnothing(level) && continue              # level found nothing coherent
        _merge_level!(result, level, cs)
    end
    return result
end

"""
    pyramid_level(pair, grid, p, chip_size, wanted) -> Union{Nothing,DisplacementField}

One pyramid level: a coarse pass to find where motion is coherent, then a fine pass
restricted to that neighbourhood, then hole filling.

`wanted` marks the points this level should attempt — in the pyramid, those no finer level
resolved. Returns `nothing` if the coarse pass found too little to be worth continuing,
which is what `min_coarse_valid_fraction` decides.

Separately callable so a caller can run a single scale without the pyramid.
"""
function pyramid_level(pair::ImagePair, grid::PointSet{2}, p::Params,
                       chip_size::Integer, wanted::AbstractMatrix{Bool})
    csx = Int(chip_size)
    csy = chip_size_y(p, csx)

    # A level's points: the requested ones, at this level's chip size.
    pts = _level_points(grid, p, csx, csy, wanted)
    nsearchable(pts) == 0 && return nothing

    coarse_mask = _coarse_pass(pair, pts, p, csx, csy)
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

    fine = track(pair, pts, p)
    _reject_and_fill!(fine, pts, p)
    return fine
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
    pts = PointSet(copy(grid.x), copy(grid.y), rx, ry,
                   copy(grid.dx_prior), copy(grid.dy_prior),
                   fill(csx, n), fill(csy, n))
    sanitize!(pts, p.min_search_radius)
    return pts
end

# ---------------------------------------------------------------------------
# Coarse pass
# ---------------------------------------------------------------------------

# Correlate a decimated sample of the points, filter for spatial consistency, and dilate what
# survives. Returns a full-grid mask of where the fine pass should look, or `nothing` if too
# little of the coarse grid was coherent for the level to be worth continuing.
function _coarse_pass(pair::ImagePair, pts::PointSet{2}, p::Params, csx::Int, csy::Int)
    stride = p.coarse_stride
    nr, nc = size(pts)
    rows = stride:stride:nr
    cols = stride:stride:nc
    # Too few coarse points to judge consistency against their neighbours: the outlier filter
    # needs a neighbourhood, so fall back to searching everything rather than rejecting on
    # no evidence.
    (length(rows) < p.outlier_window || length(cols) < p.outlier_window) &&
        return trues(nr, nc)

    # The coarse point's radius must cover its whole cell, since it stands in for every fine
    # point inside it — hence the max over the cell rather than a sample of one point.
    crx = windowmax(Float32.(pts.radius_x), stride)[rows, cols]
    cry = windowmax(Float32.(pts.radius_y), stride)[rows, cols]
    coarse = PointSet(pts.x[rows, cols], pts.y[rows, cols],
                      round.(Int, crx), round.(Int, cry),
                      pts.dx_prior[rows, cols], pts.dy_prior[rows, cols],
                      fill(csx, length(rows), length(cols)),
                      fill(csy, length(rows), length(cols)))
    nsearchable(coarse) == 0 && return nothing

    # Integer peaks only: the coarse pass decides *where* to look, and sub-pixel precision
    # would not change that answer while costing most of the pass.
    cd = track(pair, coarse, p; subpixel = NoRefine())
    measured = .!isnan.(cd.dx)
    any(measured) || return nothing

    keep = reject_outliers(cd.dx, cd.dy, coarse.radius_x, coarse.radius_y,
                           Matrix{Bool}(measured), upsampling(p.subpixel),
                           _coarse_filter(p))
    @inbounds for i in eachindex(keep)
        measured[i] || (keep[i] = false)
    end

    # Was enough of the coarse grid coherent? Judged only over points that were searchable at
    # all, so a level is not penalised for the points a previous level already resolved.
    searchable = coarse.radius_x .> 0
    denom = count(i -> searchable[i] && measured[i], eachindex(keep))
    denom == 0 && return nothing
    numer = count(i -> searchable[i] && keep[i], eachindex(keep))
    (numer / denom) < p.min_coarse_valid_fraction && return nothing

    # A coherent coarse estimate is evidence about its neighbourhood, not just its own point,
    # so grow it before it restricts the fine search.
    grown = dilate_within(keep, p.coarse_buffer)
    # Back to the full grid. Nearest-neighbour because this is a mask: an interpolated value
    # between "search" and "do not search" would mean nothing.
    return resample(grown, (nr, nc), Nearest()) .> 0.5f0
end

# The coarse grid is decimated, so its neighbourhoods span `stride` times more ground and its
# agreement threshold has to be looser — a coarse point's neighbours are genuinely further
# away and less like it. The reference derives this from an overlap fraction; the effect is
# the same and this states it directly.
function _coarse_filter(p::Params)
    return outlier_filter(; window = p.outlier_window,
                          iterations = max(p.outlier_iterations - 1, 1),
                          min_agree_fraction = p.min_agree_fraction,
                          agree_tolerance = p.agree_tolerance,
                          mad_scale = p.mad_scale)
end

# ---------------------------------------------------------------------------
# Fine pass cleanup
# ---------------------------------------------------------------------------

# Reject inconsistent estimates, then fill small holes from their neighbours.
#
# Filling is worth doing at this stage rather than at the end: a hole filled here can be
# resampled coherently into the next level's prior, whereas a hole left open forces that
# level to search blind.
function _reject_and_fill!(d::DisplacementField, pts::PointSet, p::Params)
    measured = .!isnan.(d.dx)
    keep = reject_outliers(d.dx, d.dy, pts.radius_x, pts.radius_y,
                           Matrix{Bool}(measured), upsampling(p.subpixel),
                           _fine_filter(p))
    @inbounds for i in eachindex(keep)
        if !keep[i]
            d.dx[i] = NaN32
            d.dy[i] = NaN32
            d.correlation[i] = NaN32
        end
    end
    _fill_holes!(d, p)
    return d
end

_fine_filter(p::Params) =
    outlier_filter(; window = p.outlier_window, iterations = p.outlier_iterations,
                   min_agree_fraction = p.min_agree_fraction,
                   agree_tolerance = p.agree_tolerance, mad_scale = p.mad_scale)

# Fill points surrounded by enough measured neighbours with the neighbourhood median.
#
# Three passes, because each fills only points that are already well surrounded: filling one
# ring makes the next ring well surrounded in turn, so a small hole closes from its edge
# inward while a large one is left alone. A single pass with a looser threshold would instead
# invent values in the middle of genuinely empty regions.
function _fill_holes!(d::DisplacementField, p::Params)
    w = p.fill_window
    # Two thirds of a full window: enough neighbours that the median means something, and
    # strict enough that an isolated point does not seed a fill.
    needed = 2 * w^2 ÷ 3
    for _ in 1:3
        medx = windowmedian(d.dx, w)
        medy = windowmedian(d.dy, w)
        # Count measured neighbours by taking the mean of an indicator, which reuses the
        # NaN-aware box mean rather than adding another sliding count.
        ind = Matrix{Float32}(undef, size(d.dx))
        @inbounds for i in eachindex(ind)
            ind[i] = isnan(d.dx[i]) ? 0.0f0 : 1.0f0
        end
        frac = windowmean(ind, w; hasnan = false)
        filled = false
        @inbounds for i in eachindex(d.dx)
            isnan(d.dx[i]) || continue
            isnan(medx[i]) && continue
            frac[i] * w^2 >= needed || continue
            d.dx[i] = medx[i]
            d.dy[i] = medy[i]
            filled = true
        end
        filled || break          # nothing left that qualifies
    end
    return d
end

# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

# Write a level's results wherever no finer level already produced one.
#
# The gate is what makes the pyramid work: every level sees the same grid, and the first to
# succeed at a point owns it. Since levels run finest first, that is the smallest chip that
# could resolve the point.
function _merge_level!(result::PyramidResult, level::DisplacementField, chip_size::Integer)
    cs = UInt16(chip_size)
    @inbounds for i in eachindex(result.dx)
        result.chip_size[i] == 0 || continue      # a finer level already owns this point
        isnan(level.dx[i]) && continue
        result.dx[i] = level.dx[i]
        result.dy[i] = level.dy[i]
        result.correlation[i] = level.correlation[i]
        result.chip_size[i] = cs
        # A filled point has no correlation of its own, which is how `interpolated` is
        # distinguished from measured without carrying a second mask through the level.
        result.interpolated[i] = isnan(level.correlation[i])
    end
    return result
end
