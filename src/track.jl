# The grid loop: correlate at every search point.
#
# Layer 2/3 boundary. This is where the per-point primitives of Layer 1 meet a whole
# point set, and it is the only place threading appears.
#
# One function covers every element type. The reference has two copies of this loop —
# `arImgDisp_u` and `arImgDisp_s`, 275 lines each, differing only in which C entry point
# they call and the dtype of their buffers — and four C++ functions behind them that by
# v2.1.2 differ only in element type, since the similarity measure was unified. Julia's
# dispatch collapses all six into one.
#
# ---------------------------------------------------------------------------
# Padding, and why the grid is offset by half a pixel
# ---------------------------------------------------------------------------
#
# A chip near the image edge needs data that is not there. Both images are zero-padded by
# enough that every search window is in bounds — `max(chip)/2 + max(radius + |prior|) + 2`,
# where the `+2` is slack — so the inner loop never has to test its bounds.
#
# The half-pixel offset is subtler and comes from the reference. An even-sized chip has no
# centre sample: extending `-chip/2` to `chip/2 - 1` about an integer position puts the
# chip's true centre half a pixel past it. Adding 0.5 to the grid makes the reported
# displacement refer to that actual centre rather than to a position the chip is not
# symmetric about. It also makes every index truncation below exact, since the grid values
# become integers plus one.

"""
    DisplacementField

Per-point results of a correlation pass.

`dx` and `dy` are displacements in pixels, `correlation` the peak similarity, and
`searched` marks the points that were actually correlated. Non-searched and failed points
are `NaN` in `dx`/`dy` and `false` in `searched` — distinguishing "no measurement" from a
measurement of zero, which the reference conflates.

Fields share the shape of the [`PointSet`](@ref) they came from, so a gridded point set
yields gridded output.
"""
struct DisplacementField{N,A<:AbstractArray{Float32,N},B<:AbstractArray{Bool,N}}
    dx::A
    dy::A
    correlation::A
    searched::B
end

"""
    displacement_field(pts::PointSet) -> DisplacementField

Allocate results for `pts`, initialised to "no measurement".
"""
function displacement_field(pts::PointSet)
    sz = size(pts.x)
    return DisplacementField(fill(NaN32, sz), fill(NaN32, sz), fill(NaN32, sz),
                             falses(sz))
end

Base.size(d::DisplacementField) = size(d.dx)
Base.length(d::DisplacementField) = length(d.dx)

"""
    nmeasured(d::DisplacementField) -> Int

How many points yielded a displacement. Smaller than `count(d.searched)` when some
searched points had a degenerate chip.
"""
nmeasured(d::DisplacementField) = count(!isnan, d.dx)

# ---------------------------------------------------------------------------

"""
    track!(out, pair, pts, p; subpixel = p.subpixel) -> DisplacementField

Correlate `pair` at every searchable point of `pts`, writing into `out`.

The chip is cut from the secondary image and the search window from the reference, so the
returned displacement is the offset from secondary back to reference — see
[`peak_offset`](@ref) for why that is the negative of the feature motion, and note it is
*not* negated here. The single flip to a physical convention happens at the output
boundary.

`subpixel` overrides the parameter set's refinement method, which the pyramid needs: its
coarse pass wants integer peaks only, and paying for refinement there would be wasted.

A point is skipped, leaving `NaN`, if its search radius is zero in either axis, if its
chip lies wholly outside the image, or if its chip contains no valid pixel. The last is
what keeps zero-padding out of the result: the images are padded so the inner loop needs
no bounds test, but padding is not data, and a chip made of it would correlate with any
other such chip. See `valid` on [`ImagePair`](@ref).
"""
function track!(out::DisplacementField, pair::ImagePair, pts::PointSet, p::Params;
                subpixel::SubpixelMethod = p.subpixel)
    size(out) == size(pts.x) || throw(DimensionMismatch(
        "output is $(size(out)) but the point set has $(size(pts.x)) points"))

    flat = scatter(pts)          # free: shares memory, discards only the layout
    pad = _pad_extent(flat)
    ref = _zeropad(pair.reference, pad)
    sec = _zeropad(pair.secondary, pad)
    # The mask is padded as invalid, which is what distinguishes "outside the image" from
    # "dark". Padding exists so the inner loop needs no bounds test; it is not data, and a
    # chip made of it would correlate perfectly with any other such chip.
    okmask = _zeropad_mask(valid(pair), pad)

    up = upsampling(subpixel)
    chipx, chipy, rx, ry = _pass_extents(flat)
    # Plans are built here, on this task, before any spawning. FFTW's planner is not
    # thread-safe, so leaving this to the workers would have every one of them contend on
    # the planner lock at its first point — turning the most parallel part of the run into
    # its most serial.
    _warm_pass_plans(chipx, chipy, rx, ry)

    istrue(p.threaded) ? _track_threaded!(out, ref, sec, okmask, flat, pad, chipx, chipy,
                                          rx, ry, up, p) :
                         _track_chunk!(out, ref, sec, okmask, flat, pad, chipx, chipy,
                                       rx, ry, up, p, eachindex(flat))
    return out
end

"""
    track(pair, pts, p; subpixel = p.subpixel) -> DisplacementField

Allocating form of [`track!`](@ref).
"""
track(pair::ImagePair, pts::PointSet, p::Params; kw...) =
    track!(displacement_field(pts), pair, pts, p; kw...)

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

# Padding wide enough that every search window is in bounds. Derived from the widest chip
# and the largest radius-plus-prior over the whole point set, so one padded copy serves
# every point.
function _pad_extent(pts::PointSet)
    px = py = 0
    @inbounds for i in eachindex(pts)
        issearchable(pts, i) || continue
        px = max(px, pts.chip_size_x[i] ÷ 2 + pts.radius_x[i] +
                     ceil(Int, abs(pts.dx_prior[i])))
        py = max(py, pts.chip_size_y[i] ÷ 2 + pts.radius_y[i] +
                     ceil(Int, abs(pts.dy_prior[i])))
    end
    # The +2 is the reference's slack. Cheap, and it absorbs the half-pixel grid offset
    # plus any rounding in the index truncations.
    return (px + 2, py + 2)
end

function _zeropad(A::AbstractMatrix{T}, pad::Tuple{Int,Int}) where {T}
    px, py = pad
    nr, nc = size(A)
    out = zeros(T, nr + 2py, nc + 2px)
    @inbounds for j in 1:nc, i in 1:nr
        out[i + py, j + px] = A[i, j]
    end
    return out
end

# The mask pads to `false`: everything outside the original image is invalid.
_zeropad_mask(m::AbstractMatrix{Bool}, pad::Tuple{Int,Int}) = _zeropad(m, pad)

# The largest chip and radius any point in the pass uses. Workspaces are sized to these
# once, and smaller points take views of the corner rather than reallocating — which is
# why nothing size-related is a type parameter anywhere in the correlation core.
function _pass_extents(pts::PointSet)
    cx = cy = rx = ry = 0
    @inbounds for i in eachindex(pts)
        issearchable(pts, i) || continue
        cx = max(cx, pts.chip_size_x[i])
        cy = max(cy, pts.chip_size_y[i])
        rx = max(rx, pts.radius_x[i])
        ry = max(ry, pts.radius_y[i])
    end
    return cx, cy, rx, ry
end

# Which transform sizes this pass will use, so they can be planned up front. Only the
# maximum extent is needed: every point either uses it or takes the direct path.
function _warm_pass_plans(chipx::Int, chipy::Int, rx::Int, ry::Int)
    (chipx == 0 || rx == 0) && return nothing
    fy = next_fft_size(chipy + 2ry - 1)
    fx = next_fft_size(chipx + 2rx - 1)
    warm_plans!(((fy, fx),))
    return nothing
end

# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

# One chunk of points, on one task, with one workspace. Split out from the threaded
# driver so the serial path is the same code with a single chunk — the two must agree
# bitwise, and sharing the body is how that is guaranteed rather than tested for.
function _track_chunk!(out::DisplacementField, ref, sec, okmask, pts::PointSet,
                       pad::Tuple{Int,Int}, chipx::Int, chipy::Int,
                       rx::Int, ry::Int, up::Int, p::Params, idx)
    chipx == 0 && return out     # nothing searchable in this pass
    T = eltype(ref)
    ws = workspace(T, (chipx, chipy), (max(rx, 1), max(ry, 1)))
    rw = up > 1 ? refinement_workspace(up) : nothing
    px, py = pad
    padded = size(ref)

    @inbounds for i in idx
        issearchable(pts, i) || continue

        # Grid coordinates in the padded image, offset by half a pixel so that an
        # even-sized chip's reported position refers to its true centre.
        cx = pts.x[i] + px + 0.5
        cy = pts.y[i] + py + 0.5
        hx = pts.chip_size_x[i] ÷ 2
        hy = pts.chip_size_y[i] ÷ 2
        prx = pts.radius_x[i]
        pry = pts.radius_y[i]

        # The chip follows the a-priori displacement; the window stays on the point. So
        # the search begins where motion is expected while the answer remains relative to
        # the point itself.
        ci = floor(Int, cy - pts.dy_prior[i])
        cj = floor(Int, cx - pts.dx_prior[i])
        chip_rows = (ci - hy):(ci + hy - 1)
        chip_cols = (cj - hx):(cj + hx - 1)

        si = floor(Int, cy)
        sj = floor(Int, cx)
        win_rows = (si - hy - pry):(si + hy + pry - 2)
        win_cols = (sj - hx - prx):(sj + hx + prx - 2)

        # Padding is sized so this holds, but a caller-supplied scattered point set can
        # place a point anywhere, so it is checked rather than assumed.
        _in_padded(chip_rows, chip_cols, padded) || continue
        _in_padded(win_rows, win_cols, padded) || continue
        # A chip with no valid pixel is padding, not imagery. Testing the chip rather than
        # the window is deliberate: the window may legitimately overlap the edge, since the
        # correlation only needs the chip to be real.
        _any_valid(okmask, chip_rows, chip_cols) || continue

        out.searched[i] = true

        chip = @view sec[chip_rows, chip_cols]
        window = @view ref[win_rows, win_cols]
        surface = correlate!(ws, window, chip, (prx, pry); measure = p.similarity)
        # A chip with no texture carries no information about displacement, so it is left
        # as no measurement. The reference reports the search-window corner here, which
        # over masked or featureless terrain is a systematic corner-pinned bias.
        degenerate(ws) && continue

        dx, dy, c = isnothing(rw) ? peak_offset(surface, (prx, pry)) :
                                    subpixel_peak(rw, surface, (prx, pry), up)

        # Back to displacement about the grid point: the surface is centred on the window,
        # which is centred on the point, and the chip was offset by the prior.
        out.dx[i] = Float32(dx + pts.dx_prior[i])
        out.dy[i] = Float32(dy + pts.dy_prior[i])
        out.correlation[i] = c
    end
    return out
end

@inline function _in_padded(rows, cols, sz::Tuple{Int,Int})
    return first(rows) >= 1 && last(rows) <= sz[1] &&
           first(cols) >= 1 && last(cols) <= sz[2]
end

@inline function _any_valid(mask, rows, cols)
    @inbounds for j in cols, i in rows
        mask[i, j] && return true
    end
    return false
end

# Threaded driver: one task and one workspace per chunk, indexed by chunk rather than by
# thread. Indexing per-thread state by `threadid()` is unsafe under task migration, and
# per-chunk is correct by construction.
#
# Chunks are deliberately finer than the thread count. The pyramid's sparse search zeroes
# most of the grid in spatially clustered patterns, and a skipped point costs a comparison
# where a searched one costs microseconds — so an even split of the index range leaves some
# tasks with almost nothing to do. Oversubscribing lets the scheduler even that out.
function _track_threaded!(out::DisplacementField, ref, sec, okmask, pts::PointSet,
                          pad::Tuple{Int,Int}, chipx::Int, chipy::Int,
                          rx::Int, ry::Int, up::Int, p::Params)
    n = length(pts)
    nchunks = min(n, max(1, 2 * Threads.nthreads()))
    chunk = cld(n, nchunks)
    tasks = map(Iterators.partition(eachindex(pts), chunk)) do range
        StableTasks.@spawn _track_chunk!(out, ref, sec, okmask, pts, pad, chipx, chipy,
                                         rx, ry, up, p, range)
    end
    foreach(wait, tasks)
    return out
end
