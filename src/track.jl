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

`searched` is a `Matrix{Bool}` rather than a `BitMatrix` deliberately. `BitArray` packs 64
elements per word, so writing one element is a read-modify-write of that word — and the
threaded grid loop has different tasks writing different points that may share a word, which
can lose a neighbour's write. A byte per element is not shared, so the writes are genuinely
independent. Four times the memory for a mask that is a fraction of the output anyway.

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
                             fill(false, sz))
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
    track!(out, pair, pts, p; subpixel = p.subpixel, measure = first(p.similarity))
        -> DisplacementField

Correlate `pair` at every searchable point of `pts`, writing into `out`.

The chip is cut from the secondary image and the search window from the reference, so the
returned displacement is the offset from secondary back to reference — see
[`peak_offset`](@ref) for why that is the negative of the feature motion, and note it is
*not* negated here. The single flip to a physical convention happens at the output
boundary.

`subpixel` overrides the parameter set's refinement method, which the caller needs: a
coarse pass wants integer peaks only, and paying for refinement there would be wasted.

`measure` likewise overrides the similarity measure. `p.similarity` is a tuple, one entry per
chip-size level, so a single level cannot read it without knowing which level it is — the
chip-size loop passes the right one. The default is the first, which is correct for a caller
running `track!` directly on a single scale.

A point is skipped, leaving `NaN`, if its search radius is zero in either axis, if its
chip lies wholly outside the image, or if its chip contains no valid pixel. The last is
what keeps zero-padding out of the result: the images are padded so the inner loop needs
no bounds test, but padding is not data, and a chip made of it would correlate with any
other such chip. See `valid` on [`ImagePair`](@ref).
"""
function track!(out::DisplacementField, pair::ImagePair, pts::PointSet, p::Params;
                subpixel::SubpixelMethod = p.subpixel,
                measure::SimilarityMeasure = first(p.similarity))
    # Interpolating the `Tuple`s directly would be the natural spelling, but showing a `Tuple`
    # reaches `textwidth` and `Base.repeat`, which `--trim` cannot resolve — so this one message
    # would make the whole package untrimmable. Element counts say the same thing here: `out`
    # comes from `displacement_field(pts)`, so a mismatch is a different point set, not a
    # transposed one.
    length(out) == length(pts) || throw(DimensionMismatch(
        "output has $(length(out)) points but the point set has $(length(pts))"))

    flat = scatter(pts)          # free: shares memory, discards only the layout
    chipx, chipy, rx, ry, pad, fits = _pass_geometry(flat, size(pair))
    # Nothing searchable: skip the padding, the planning, and the task spawning entirely.
    # Later chip-size levels hit this, since their coarse pass zeroes most radii.
    chipx == 0 && return out

    # Padding only when a point actually needs it. `gridpoints` insets by the chip half-extent
    # plus the search radius, so a gridded pass never does — and padding it anyway costs 1.2 ms
    # and 10.6 MB per call, which is 2.5% of a dense coarse pass but ~10% of the sparse ones the
    # search produces. A scattered, caller-supplied point set may still need it.
    #
    # The half-pixel offset applies either way: it is what makes an even-sized chip's reported
    # position refer to its true centre rather than to a position the chip is not symmetric
    # about.
    if fits
        ref, sec = pair.reference, pair.secondary
        okmask = valid(pair)
        shifted = _shift_points(flat, (0, 0))
    else
        ref = _zeropad(pair.reference, pad)
        sec = _zeropad(pair.secondary, pad)
        # The mask pads to `false`, which is what distinguishes "outside the image" from "dark".
        okmask = _zeropad(valid(pair), pad)
        shifted = _shift_points(flat, pad)
    end

    up = upsampling(subpixel)
    # Plans are built here, on this task, before any spawning. FFTW's planner is not
    # thread-safe, so leaving this to the workers would have every one of them contend on
    # the planner lock at its first point — turning the most parallel part of the run into
    # its most serial.
    _warm_pass_plans(chipx, chipy, rx, ry, measure)

    istrue(p.threaded) ? _track_threaded!(out, ref, sec, okmask, shifted, chipx, chipy,
                                          rx, ry, up, p, measure) :
                         _track_chunk!(out, ref, sec, okmask, shifted, chipx, chipy,
                                       rx, ry, up, p, measure, eachindex(shifted))
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

# Everything about a pass that is a summary over its points: the largest chip and radius
# any point uses, and how much padding makes every window in bounds.
#
# One traversal rather than two. The pad is the maximum of a per-point *sum*, not a sum of
# maxima — combining the maxima separately would still be safe but would pad further than
# needed, so the accumulation stays per point.
#
# The largest chip and radius are what the workspaces are sized to; smaller points take
# views of the corner rather than reallocating, which is why nothing size-related is a type
# parameter in the correlation core.
function _pass_geometry(pts::PointSet, imagesize::Tuple{Int,Int})
    cx = cy = rx = ry = px = py = 0
    fits = true
    @inbounds for i in eachindex(pts)
        issearchable(pts, i) || continue
        csx, csy = pts.chip_size_x[i], pts.chip_size_y[i]
        prx, pry = pts.radius_x[i], pts.radius_y[i]
        cx = max(cx, csx)
        cy = max(cy, csy)
        rx = max(rx, prx)
        ry = max(ry, pry)
        px = max(px, csx ÷ 2 + prx + ceil(Int, abs(pts.dx_prior[i])))
        py = max(py, csy ÷ 2 + pry + ceil(Int, abs(pts.dy_prior[i])))
        # Whether this point's window lies inside the unpadded image, which decides whether
        # padding is needed at all. One comparison per point, against the alternative of
        # allocating and filling three full-size copies unconditionally.
        fits &= inbounds(pts, i, imagesize)
    end
    # The +2 is the reference's slack: it absorbs the half-pixel grid offset and any
    # rounding in the index truncations.
    return cx, cy, rx, ry, (px + 2, py + 2), fits
end

# Translate points into padded-image coordinates, plus the half-pixel offset. A zero pad is
# the unpadded case, where only the half-pixel offset applies.
#
# The offset is why an even-sized chip's reported position means what it says. Such a chip
# has no centre sample: extending `-chip/2` to `chip/2 - 1` about an integer position puts
# its true centre half a pixel past it. Adding 0.5 makes the displacement refer to that
# centre, and makes every index truncation exact, since the coordinates become integers
# plus one.
function _shift_points(pts::PointSet, pad::Tuple{Int,Int})
    px, py = pad
    return rebuild(pts; x = pts.x .+ (px + 0.5), y = pts.y .+ (py + 0.5))
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

# Which transform sizes this pass will use, so they can be planned up front. Only the
# maximum extent is needed: every point either uses it or takes the direct path.
#
# The measure decides *which kind* of transform to warm — `Coherence` executes complex-to-complex
# plans and the real measures real-to-complex ones, so warming without knowing the measure warms
# the wrong pair half the time.
function _warm_pass_plans(chipx::Int, chipy::Int, rx::Int, ry::Int,
                          measure::SimilarityMeasure)
    (chipx == 0 || rx == 0) && return nothing
    fy = next_fft_size(chipy + 2ry - 1)
    fx = next_fft_size(chipx + 2rx - 1)
    warm_plans!(((fy, fx),); complex = _wants_complex_plans(measure))
    return nothing
end

# Which transform kind a measure executes. A trait rather than an `isa` at the call site, so a
# future measure declares its own answer instead of being added to a branch here.
_wants_complex_plans(::SimilarityMeasure) = false
_wants_complex_plans(::Coherence) = true

# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

# One chunk of points, on one task, with one workspace. Split out from the threaded
# driver so the serial path is the same code with a single chunk — the two must agree
# bitwise, and sharing the body is how that is guaranteed rather than tested for.
function _track_chunk!(out::DisplacementField, ref, sec, okmask, pts::PointSet,
                       chipx::Int, chipy::Int, rx::Int, ry::Int, up::Int,
                       p::Params, measure::SimilarityMeasure, idx)
    T = eltype(ref)
    ws = take_workspace!(T, (chipx, chipy), (rx, ry))
    rw = up > 1 ? take_refinement!(up) : nothing
    try

        @inbounds for i in idx
            issearchable(pts, i) || continue

            prx = pts.radius_x[i]
            pry = pts.radius_y[i]
            # `pts` is already in padded coordinates, so the documented primitives apply
            # directly. Recomputing their arithmetic here would put the asymmetric window
            # convention and the even-chip half-extent -- the two most delicate index
            # conventions in the package -- in a second, untested place.
            chip_rows, chip_cols = chip_bounds(pts, i)
            win_rows, win_cols = search_bounds(pts, i)

            # Padding is sized so this holds, but a caller-supplied scattered point set can
            # place a point anywhere, so it is checked rather than assumed.
            checkbounds(Bool, ref, chip_rows, chip_cols) || continue
            checkbounds(Bool, ref, win_rows, win_cols) || continue
            # A chip with no valid pixel is padding, not imagery. Testing the chip rather than
            # the window is deliberate: the window may legitimately overlap the edge, since the
            # correlation only needs the chip to be real.
            _any_valid(okmask, chip_rows, chip_cols) || continue

            out.searched[i] = true

            chip = @view sec[chip_rows, chip_cols]
            window = @view ref[win_rows, win_cols]
            surface = _correlate_rotations!(ws, window, chip, (prx, pry), measure, p.rotation)
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
    finally
        # Back to the pool even if a point threw. A leaked workspace is not a crash, but it
        # silently turns the pool back into per-chunk allocation, which is the thing this
        # exists to avoid — and that would be invisible.
        give_workspace!(ws)
        isnothing(rw) || give_refinement!(rw)
    end
end

# One correlation, or several at different chip rotations with the best kept.
#
# `NoRotationSearch` is a separate method rather than a one-angle loop, and that is the point: it must
# compile to exactly the call it replaced, with no surface copy and no comparison, so enabling this
# feature cannot cost anything for the callers who do not. The `RotationSearch` method is where the
# expense lives, and it is proportional to the angle count by construction.
@inline _correlate_rotations!(ws, window, chip, radius, measure, ::NoRotationSearch) =
    correlate!(ws, window, chip, radius; measure)

function _correlate_rotations!(ws, window, chip, radius, measure, rot::RotationSearch)
    # `sea_ice_drift`'s `rotate_and_match`: rotate the template, correlate, keep the angle whose peak
    # is highest. Its comparison is on `result.max()`, so this one is too — the strongest peak across
    # angles wins, not the one closest to zero rotation.
    bestpeak = -Inf32
    found = false
    nr, nc = 2radius[2], 2radius[1]
    hold = @view ws.rotbest[1:nr, 1:nc]
    for a in angles(rot)
        rotated = _rotate_chip(ws, chip, a)
        s = correlate!(ws, window, rotated, radius; measure)
        degenerate(ws) && continue
        pk = maximum(s)
        if pk > bestpeak
            bestpeak = pk
            # The surface aliases `ws.surface`, which the next angle overwrites, so the winner must
            # be held elsewhere. Into a workspace buffer rather than a fresh `copy`: that was 877 of
            # the 1177 allocations a 225-point rotation pass made, for no reason but convenience.
            copyto!(hold, s)
            found = true
        end
    end
    if !found
        # Every angle degenerate: report it the way a single degenerate chip is reported, so the
        # caller's existing check still works.
        ws.was_degenerate[] = true
        surf = @view ws.surface[1:(2radius[2]), 1:(2radius[1])]
        fill!(surf, 0.0f0)
        return surf
    end
    ws.was_degenerate[] = false
    # Back into `surface`, because that is what every caller expects a correlation to return — and it
    # is safe: the last angle's surface has already been compared and is no longer needed.
    surf = @view ws.surface[1:nr, 1:nc]
    copyto!(surf, hold)
    return surf
end

# The chip, rotated about its own centre by `deg` degrees, into the workspace's rotation buffer.
#
# Bilinear, and nearest-neighbour was rejected: a rotated chip is correlated against an *unrotated*
# window, so any resampling artefact enters the correlation directly. Bilinear is what
# `sea_ice_drift` uses (`cv2.warpAffine` defaults to it) and what the subpixel refinement already
# assumes elsewhere.
#
# Out-of-chip samples become zero, matching the reference's `borderValue=0`. That is a real
# limitation rather than a detail: rotating a square chip necessarily pulls in corners that were not
# in it, so a large angle dilutes the correlation with padding. It is why the useful angle range is
# small — the reference's own default is ±3°.
function _rotate_chip(ws::CorrelationWorkspace, chip::AbstractMatrix, deg::Float64)
    deg == 0.0 && return chip
    ch, cw = size(chip)
    dst = @view ws.rotchip[1:ch, 1:cw]
    s, c = sincos(deg2rad(deg))
    # Rotate about the chip centre, in the half-pixel convention the grid already uses.
    yc = (ch + 1) / 2
    xc = (cw + 1) / 2
    T = eltype(dst)
    @inbounds for j in 1:cw, i in 1:ch
        dy = i - yc
        dx = j - xc
        # Inverse rotation: where in the source does this destination pixel come from?
        sy = yc + c * dy + s * dx
        sx = xc - s * dy + c * dx
        i0 = floor(Int, sy)
        j0 = floor(Int, sx)
        if i0 < 1 || j0 < 1 || i0 + 1 > ch || j0 + 1 > cw
            dst[i, j] = zero(T)
            continue
        end
        fy = sy - i0
        fx = sx - j0
        v00 = Float64(chip[i0, j0]);     v10 = Float64(chip[i0 + 1, j0])
        v01 = Float64(chip[i0, j0 + 1]); v11 = Float64(chip[i0 + 1, j0 + 1])
        dst[i, j] = T((1 - fy) * ((1 - fx) * v00 + fx * v01) +
                      fy * ((1 - fx) * v10 + fx * v11))
    end
    return dst
end

# Explicit loop rather than `any` over a view. Both short-circuit and neither allocates, but
# the loop is 2x faster in the worst case that matters — an all-invalid window, where there is
# no early exit to take (measured 440 ns against 888 ns on 32x32). The generic path does not
# vectorise across a `SubArray` the way a column-major loop does, and this is the only branch
# here that is not free.
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
# Chunks are deliberately finer than the thread count. The sparse search zeroes
# most of the grid in spatially clustered patterns, and a skipped point costs a comparison
# where a searched one costs microseconds — so an even split of the index range leaves some
# tasks with almost nothing to do. Oversubscribing lets the scheduler even that out.
function _track_threaded!(out::DisplacementField, ref, sec, okmask, pts::PointSet,
                          chipx::Int, chipy::Int, rx::Int, ry::Int, up::Int, p::Params,
                          measure::SimilarityMeasure)
    n = length(pts)
    nchunks = min(n, max(1, 2 * Threads.nthreads()))
    chunk = cld(n, nchunks)
    tasks = map(Iterators.partition(eachindex(pts), chunk)) do range
        StableTasks.@spawn _track_chunk!(out, ref, sec, okmask, pts, chipx, chipy,
                                         rx, ry, up, p, measure, range)
    end
    foreach(wait, tasks)
    return out
end
