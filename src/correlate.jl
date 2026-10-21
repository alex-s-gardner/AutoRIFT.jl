# The correlation surface.
#
# Layer 1: this file and `integral.jl`, `peak.jl` depend on nothing else in
# AutoRIFT. Template matching with sub-pixel peak location is a gap in the Julia
# ecosystem — `TemplateMatching.jl` has the right shape but is a single-author
# 0.1.x with no masks, NaN handling, subpixel refinement, or threading — so this
# is written to be extractable as its own package later. Keeping the dependency
# direction one-way makes that a file move rather than a refactor.
#
# ---------------------------------------------------------------------------
# The math, and why it is arranged this way
# ---------------------------------------------------------------------------
#
# For a chip T and the window W of the search image at shift (i, j), zero-mean
# normalized cross-correlation is
#
#     R(i,j) = Σ(T - T̄)(W - W̄) / sqrt( Σ(T - T̄)² · Σ(W - W̄)² )
#
# Three rearrangements make this cheap, and all three matter:
#
# 1. The chip's mean and norm are computed once, not once per shift.
#
# 2. The window mean drops out of the numerator. Write T' = T - T̄; then
#    Σ T'(W - W̄) = Σ T'W - W̄ Σ T', and Σ T' = 0 by construction. So the
#    numerator is just the raw correlation of the mean-removed chip against the
#    *unmodified* search image — no per-shift mean subtraction, and no need to
#    materialise a mean-removed copy of each window.
#
# 3. The window variance comes from integral images: Σ(W - W̄)² = ΣW² - (ΣW)²/n,
#    and both sums are four array reads. Without this, the denominator alone would
#    cost as much as the numerator.
#
# What remains is one multiply-accumulate over the chip per shift, which is either
# evaluated directly or, for large chips, obtained for all shifts at once by FFT.
#
# ---------------------------------------------------------------------------
# What OpenCV's matchTemplate does differently, and why none of it transfers
# ---------------------------------------------------------------------------
#
# `modules/imgproc/src/templmatch.cpp` is both the correctness reference and the speed target, so
# its tricks are worth knowing. Four were examined; none applies here, and the reasons are
# structural rather than incidental.
#
#   * **The template DFT is hoisted out of the loop.** OpenCV scans *one* template over one image,
#     so it transforms the template once and reuses the spectrum for every tile. Here every grid
#     point cuts its own chip *and* its own search window — nothing is constant across points — so
#     both transforms are genuinely per point. This is the biggest apparent win and it is simply
#     not available.
#
#   * **The image is tiled, each tile transformed separately**, because DFT throughput degrades on
#     large arrays (`blockScale = 4.5`, `minBlockSize = 256`). Our transforms are already 84, 128,
#     and 192 points square for chips 32, 64, and 128 — all below the size at which OpenCV would
#     start tiling. There is nothing to tile.
#
#   * **A hysteresis clamp on the near-zero denominator**: when `|num|` falls within 12.5% of the
#     threshold, OpenCV returns ±1 rather than dividing. Across all 72 correlation fixtures the
#     worst excursion outside [-1, 1] here is 9.0e-5, so on those the branch would never fire.
#
#     **The surface can leave [-1, 1] on input the fixtures do not contain**, and the branch is still
#     absent. Over an **exactly** constant part of a window — saturated snow, or a fill value that no
#     validity mask caught — `ΣW² - (ΣW)²/n` cancels to within rounding of zero while the numerator
#     carries a `Float32` transform's absolute error, and a small error over a ~1e-6 denominator is a
#     large ratio. Measured on a 640² scene with a saturated 201² block, chip 32 radius 25 and the
#     default `Highpass`: 1.5% of surfaces exceed 1, worst **7.1**. A near-constant block instead —
#     one DN of sensor noise — produces **no** excursion at all, so this needs exact constancy rather
#     than merely low contrast.
#
#     It is nonetheless **not reachable in the output**, which is why no clamp is implemented. A
#     garbage peak from a vanishing denominator is spatially incoherent, and that is precisely what
#     [`GardnerFilter`](@ref) tests for: through `autorift` at the default settings the same scene
#     reports **zero** points wrong by more than a pixel and **zero** `correlation` above 1, at one
#     level and at three. Only `outliers = :none` leaks them — 114 points, of which every one above
#     0.8 correlation is wrong — and that setting exists for diagnosis rather than production.
#
#     If it ever does need fixing, the fix is the clamp and **not** mean-removal. Measured on the
#     failing scene: clamping a variance below `n * eps * ΣW²` to zero gives 0 surfaces out of range
#     and 0 wrong peaks against the exact answer, for one comparison per shift. Subtracting the
#     window mean before the transform is *worse* than doing nothing (100 surfaces out of range,
#     against 130) — the tables must then be rebuilt on the shifted window to stay consistent with
#     the transform, which reintroduces the same cancellation in the denominator. That the numerator
#     alone improves is real but irrelevant, since the denominator is the binding term.
#
#   * **`max(sumsq - sum^2/n, 0)` to stop a negative variance**, and returning the whole surface as
#     1 for a zero-variance template. Both of these we already do — see the clamp in
#     `_correlate_surface!` and `degenerate`. Arrived at independently, which is reassuring rather
#     than surprising: it is the only sane way to handle the cancellation.
#
# ---------------------------------------------------------------------------
# Why there is no LoopVectorization dependency
# ---------------------------------------------------------------------------
#
# `@turbo` is the obvious thing to reach for in a file like this, and it was measured on all four of
# this package's hot-loop shapes rather than argued about. It does not earn a dependency here, and
# each shape says no for its own reason:
#
#   * **The normalisation loops** gained 1.8-2.1x — and plain Julia with `ifelse` in place of the
#     branch gains the same 1.8-2.1x, indistinguishable at every geometry tried. The win was the
#     unpredictable branch, not the macro. That form is what the three `_correlate_surface!` methods
#     use.
#
#   * **`integral_both!`** is 3x *slower* under `@turbo`. A summed-area table is a serial prefix
#     recurrence; restructuring it into two vectorizable phases doubles the memory traffic and loses
#     more than vectorization returns.
#
#   * **`prepare_chip!`** gained 1.15-1.41x over the hand-unrolled form, the smallest gain on the
#     cheapest stage — and it needs linear indexing, which the strided-view call site in `track.jl`
#     cannot give.
#
#   * **`_numerators_direct!`** is the one genuine fit, at 2.1-6.8x. But `DIRECT_THRESHOLD` sits
#     below every configuration the public API can reach, so that loop runs only in tests.
#
# The transforms are 68-76% of a `correlate!` call and are inside FFTW, where no Julia-level
# vectorizer reaches at all. Re-measure before adding the dependency, not after.

"""
    CorrelationWorkspace

Preallocated buffers for correlating one chip against one search window.

All buffers the per-point kernel touches, reused across points so the inner loop allocates
nothing. A workspace is sized for the largest chip and search radius of a pass; points needing less
use views of the corner. The buffers are plain `Matrix` with runtime extents, so the kernel is
compiled once rather than per chip size.

The type is not parametrized on the image element type. The chip buffer is `Float32` (stored
mean-removed) and the integral images are `Float64` whatever the image type is.

A workspace holds the buffers for either real imagery (`ZNCC`, `NCC`) or complex imagery
([`Coherence`](@ref)). The other set is empty (`0x0`). At chip 96x52 and radius 1905x830, a 657 MiB
workspace holding both sets becomes 315 MiB for real imagery and 534 MiB for complex. `isqsum` is
used by both, since a sum of squared magnitudes is real. `AutoRIFT.WORKSPACE_POOL` keys on this
choice as well as on the geometry. Correlating real chips with a complex workspace, or the reverse,
throws `ArgumentError`.

Construct with [`workspace`](@ref).
"""
struct CorrelationWorkspace
    # Contiguous copy of the chip, mean-removed. A copy is needed anyway: the chip
    # is a strided view into the secondary image, and both the mean removal and
    # the FFT path require contiguous Float32.
    chip::Matrix{Float32}

    # The same, for complex (SLC) input, which `Coherence` correlates without taking magnitude.
    # A second buffer rather than a type parameter on the first: the struct is deliberately
    # concrete, and parameterising it would make `CorrelationWorkspace{Float32}` and
    # `CorrelationWorkspace{ComplexF32}` distinct types that specialize every method in this file
    # twice — including the real path, which must stay bit-identical.
    #
    # Sized to the chip, so it is 8 bytes per element against the real buffer's 4: 32 KiB at chip
    # 64. Allocated unconditionally, as the FFT scratch already is, so a pooled workspace can
    # serve either measure without reallocating. Measured against the alternative of allocating it
    # only for complex runs: not worth a second pool key.
    cchip::Matrix{ComplexF32}

    # Correlation surface, (2*radius_y, 2*radius_x) at full radius.
    surface::Matrix{Float32}

    # The chip, rotated, for `RotationSearch`. `Float32` regardless of input type: a rotation is a
    # bilinear resample, so the result is fractional even for `UInt8` imagery, and rounding it back
    # to the input type would quantise away the very sub-pixel information the resample computed.
    # Unused — and untouched — when rotation search is off, which is the default.
    rotchip::Matrix{Float32}

    # The best surface found across rotation angles. `correlate!` returns a view of `surface`, which
    # the next angle overwrites, so the winner has to be held somewhere — and holding it here rather
    # than in a fresh `copy` per point is worth 877 of the 1177 allocations a 225-point rotation pass
    # made. Same extent as `surface`; unused when rotation search is off.
    rotbest::Matrix{Float32}

    # Integral images of the search window, sum and sum-of-squares. Float64: see
    # the note in integral.jl.
    isum::Matrix{Float64}
    isqsum::Matrix{Float64}

    # The complex sum table, for `Coherence`. `isqsum` is shared with the real path — a sum of
    # squared *magnitudes* is real either way — but the plain sum is not, because coherence removes
    # a complex mean and the mean of a complex window is complex.
    cisum::Matrix{ComplexF64}

    # Raw correlation numerators, one per shift. Kept separate from the integral
    # images rather than sharing their storage: the integral images are read
    # throughout the normalisation loop, so overlapping them with the numerators
    # would corrupt the surface in a way that is invisible on small test inputs
    # and wrong on real ones. At 2*radius squared in Float64 this is ~20 kB at
    # radius 25 — small enough that the separation costs nothing worth having.
    numerator::Matrix{Float64}

    # The complex numerator, for `Coherence`. `ComplexF32` rather than `ComplexF64`: it is written
    # by the inverse transform, which is `ComplexF32` throughout, so a wider buffer would only
    # widen after the fact. The direct path accumulates in `ComplexF64` and narrows on store, which
    # is the same treatment the real path gives its `Float64` accumulator.
    cnumerator::Matrix{ComplexF32}

    # FFT scratch. Sized to the padded search-window extent; unused by the direct
    # path, but allocated once either way so that a pass can switch strategies
    # without reallocating.
    fbuf_a::Matrix{Float32}
    fbuf_b::Matrix{Float32}
    fspec_a::Matrix{ComplexF32}
    fspec_b::Matrix{ComplexF32}

    # The search window's spectrum, held across the angles of a `RotationSearch`. The window does
    # not change with the angle, so transforming it once per angle repeats a third of the work: the
    # forward transform is 23% of a `correlate!` and the integral tables another 10%, and both are
    # functions of the window alone. `prepare_window!` fills this and the tables; `correlate_prepared!`
    # reads them. Same extent as `fspec_a`; untouched when rotation search is off, which is the
    # default.
    wspec::Matrix{ComplexF32}

    # Complex FFT scratch, for `Coherence`. Full-size rather than the real path's half-spectrum: a
    # complex transform has no conjugate symmetry to exploit.
    #
    # **Three**, and the count was settled by measurement in both directions.
    #
    # Not two. A c2c transform *can* be planned in place, so the spectrum looks redundant — but an
    # FFTW plan encodes its buffer aliasing, and executing a plan built for distinct arrays with
    # `input === output` does not error, it returns wrong numbers: measured 4.6 *relative* error.
    # Planning genuine in-place variants does work and is bit-identical, but measured 1.3-5%
    # *slower* — FFTW's in-place c2c plans are slower here even in isolation — for two extra cached
    # plans and a subtler invariant.
    #
    # Not four either, which is where this started. The second spectrum buffer was redundant: once
    # the search window's transform has been read out of `cbuf_a`, that buffer is dead and can
    # receive the chip's spectrum. It is still a *distinct* array from `cbuf_b`, so the plan's
    # aliasing invariant is untouched — that is precisely what separates this from transforming in
    # place. Freed 9-10% of every complex workspace (55 KiB at chip 32, 288 KiB at chip 128),
    # bit-identical and runtime-neutral.
    #
    # Still the largest buffers here — 3 x fy x fx x 8 bytes, ~170 KiB at chip 32 with radius 25 —
    # and justified by what they buy: the direct numerator they replace measured 47x the ZNCC path
    # at chip 32 and 183x at chip 128.
    cbuf_a::Matrix{ComplexF32}
    cbuf_b::Matrix{ComplexF32}
    cspec_a::Matrix{ComplexF32}

    # Maximum extents this workspace was built for, so a mismatched call is an
    # error rather than a silent out-of-bounds read. Also the pool key: two workspaces are
    # interchangeable exactly when these agree.
    max_chip::Extent
    max_radius::Extent

    # Set by `correlate!` to record whether the last chip carried any signal. A mutable
    # cell so the workspace itself stays immutable and cheap to pass per point; callers
    # read it via `degenerate(ws)` rather than scanning the returned surface, which would
    # be a second pass over it at every grid point.
    was_degenerate::Base.RefValue{Bool}

    # Bytes of buffer above, filled once the buffers exist because the pool's byte bound reads it on
    # every return. Deriving it there instead — `sizeof` over `fieldnames` — put **9.8% of the whole
    # golden set's serial time** in one reflective closure holding `WORKSPACE_LOCK`, since the bound
    # sums the whole pool and a return happens once per chunk per radius bucket. Read from here, that
    # sum is a few hundred integer adds.
    #
    # A cell rather than a plain field for the same reason `was_degenerate` is one: the struct stays
    # immutable, and the value cannot be computed before the buffers it measures are constructed.
    # Summed over the fields rather than re-derived from the extents, so a buffer added above is
    # counted without this having to learn about it. See `workspace_bytes`.
    bytes::Base.RefValue{Int}
end

"""
    workspace([T], chip_size, search_radius) -> CorrelationWorkspace

Allocate correlation buffers for chips up to `chip_size` and search radii up to
`search_radius`, each an `Int` or a `(x, y)` tuple.

Use one workspace per task and do not share it: the buffers are written during correlation.

`T` is the element type of the images to be correlated (default `Float32`). A real `T` allocates
the `ZNCC`/`NCC` buffers and a complex `T` the [`Coherence`](@ref) buffers; the other set is
empty. Any `T<:Real` works, including `Int16` and other integer sensor types. The buffer element
types do not depend on `T`: the chip is `Float32` and the integral images are `Float64`.

A workspace is interchangeable only with one built for the same kind of `T` (real or complex).
[`AutoRIFT.take_workspace!`](@ref) carries that in its pool key.

Throws `ArgumentError` if a chip size or search radius is not positive.
"""
workspace(chip_size, search_radius) = workspace(Float32, chip_size, search_radius)

function workspace(::Type{T}, chip_size, search_radius) where {T<:ImageElement}
    # `extent` and not `isa Tuple`: an `Extent` is a `NamedTuple`, so `isa Tuple` is false for one
    # and the old scalar branch would have taken the whole extent as a single value. `extent`
    # accepts a scalar, a plain tuple and a named tuple, which is what it exists for.
    chip = extent(chip_size)
    rad = extent(search_radius)
    csx, csy = chip.X, chip.Y
    rx, ry = rad.X, rad.Y

    csx > 0 && csy > 0 || throw(ArgumentError(
        "chip size must be positive, got ($csx, $csy)"))
    rx > 0 && ry > 0 || throw(ArgumentError(
        "search radius must be positive, got ($rx, $ry)"))

    # Search window extent, per the reference's asymmetric convention: it reaches
    # `radius` below the chip and `radius - 1` above, so the surface is an even
    # `2 * radius` and zero displacement lands on a sample rather than between two.
    winx = csx + 2rx - 1
    winy = csy + 2ry - 1

    # FFT buffers at a size FFTW handles well. Correlation via FFT is cyclic, so
    # the transform must be at least the linear-output length to avoid wraparound;
    # the padded size is then rounded up to a product of small primes.
    fx = next_fft_size(winx)
    fy = next_fft_size(winy)

    # Which set of buffers this workspace can reach. A real image reaches the `ZNCC`/`NCC` methods of
    # `_correlate_surface!` and a complex one the `Coherence` method — and `Coherence` on a real image
    # throws rather than degrading — so the two sets are mutually exclusive per workspace. The
    # unreachable one is `0x0`: `undef` of zero extent allocates no data, and nothing indexes it
    # because dispatch never selects the path that would.
    #
    # `real ? n : 0` rather than a branch around two constructor calls, so the field order stays
    # readable against the struct and a new field cannot be added to one arm only.
    real = T <: Real
    rn(n) = real ? n : 0
    cn(n) = real ? 0 : n

    ws = CorrelationWorkspace(
        Matrix{Float32}(undef, rn(csy), rn(csx)),
        Matrix{ComplexF32}(undef, cn(csy), cn(csx)),
        Matrix{Float32}(undef, 2ry, 2rx),
        Matrix{Float32}(undef, csy, csx),
        Matrix{Float32}(undef, 2ry, 2rx),
        # `isum` is the real path's plain sum table; `isqsum` is shared, since a sum of squared
        # magnitudes is real whichever measure reads it.
        Matrix{Float64}(undef, rn(winy + 1), rn(winx + 1)),
        Matrix{Float64}(undef, winy + 1, winx + 1),
        Matrix{ComplexF64}(undef, cn(winy + 1), cn(winx + 1)),
        Matrix{Float64}(undef, rn(2ry), rn(2rx)),
        Matrix{ComplexF32}(undef, cn(2ry), cn(2rx)),
        # The real-to-complex FFT scratch, including `wspec`: `Coherence` has no `prepare_window!`
        # method and `_window_prepared` sends it down the unhoisted path (`src/track.jl:535`), so a
        # complex run reaches none of these.
        Matrix{Float32}(undef, rn(fy), rn(fx)),
        Matrix{Float32}(undef, rn(fy), rn(fx)),
        Matrix{ComplexF32}(undef, rn(fy ÷ 2 + 1), rn(fx)),
        Matrix{ComplexF32}(undef, rn(fy ÷ 2 + 1), rn(fx)),
        Matrix{ComplexF32}(undef, rn(fy ÷ 2 + 1), rn(fx)),
        Matrix{ComplexF32}(undef, cn(fy), cn(fx)),
        Matrix{ComplexF32}(undef, cn(fy), cn(fx)),
        Matrix{ComplexF32}(undef, cn(fy), cn(fx)),
        chip,
        rad,
        Ref(false),
        Ref(0),
    )
    # Measured now, once, rather than on every pool return. `bytes` is the only field that depends on
    # the others, which is why it is filled here instead of being passed in.
    ws.bytes[] = sum(f -> f === :bytes ? 0 : sizeof(getfield(ws, f)),
                     fieldnames(CorrelationWorkspace))
    return ws
end

# ---------------------------------------------------------------------------
# Workspace pool
# ---------------------------------------------------------------------------
#
# A chunk allocates its workspaces, uses them for its share of the grid, and drops them. That is
# correct — never indexed by `threadid()`, so it is safe under task migration — but a run does it
# ~96 times: 3 chip sizes x 2 passes x 16 chunks. Measured on 1024^2 threaded, that is 56.3 MiB
# against the serial path's 34.0, and **GC accounts for 21.8% of the threaded pass** against 5.1%
# of the serial one. The allocation *calls* are nothing (2.4 us per chunk, 0.15 ms per run); the
# cost is collection pressure.
#
# So they are pooled and reused. Two things make this safe:
#
#   * **Keyed on exact geometry, not "large enough".** Reusing one oversized workspace across
#     levels was tried and rejected: the FFT buffer is sized from the workspace's own extents, so
#     a chip-32 point in a chip-128 workspace runs a 192-point transform where 84 would do —
#     5x the arithmetic, and a different rounding, measured at 4.5e-8 from the exact answer. A
#     pooled workspace must be the same size as a fresh one or it is not a pool, it is a
#     different algorithm.
#
#   * **Taken and returned, not shared.** A workspace is out of the pool while a chunk holds it,
#     so two concurrent chunks never touch the same buffers. That is the same guarantee
#     per-chunk allocation gave, obtained by a lock around a small vector instead of by the
#     allocator.
#
# A run visits one geometry per (chip size, radius bucket) pair, which on a skewed radius field is
# dozens rather than the handful a uniform one reaches: 48 pairs on a 201² window of a NISAR L1 grid,
# 9.23 GiB if the pool retains one of each. So the pool is **bounded per key**, and the bound is what
# keeps bucketing's memory in proportion to its concurrency rather than to its variety.
#
# `POOL_SLOTS_PER_KEY` is the retention limit, not a limit on how many workspaces may exist: a burst
# of concurrent chunks wider than the bound still gets a workspace each — `take_workspace!` builds one
# when the pool is empty — and the surplus is dropped on return rather than kept. That is the right
# asymmetry, since holding a 0.86 GiB buffer against a future burst that may never come is a
# guaranteed cost against a speculative saving.
#
# **The bound is per key and not across keys, and that is measured rather than assumed.** It does not
# bound the number of keys, and a whole NISAR L1 grid reaches **105 keys holding 9.93 GiB** — of which
# the top buckets are near-duplicates, since `_radius_bucket` clamps to each level's own maximum radius
# and the levels' maxima differ (1905x830, 1918x1015, 1689x907, 1828x972). Every one of those is
# legitimately needed by its own level, and a level that has finished cannot ask for its bucket again,
# so the accumulation looks like pure waste.
#
# The per-key bound does not bound the *number* of keys, which on a skewed radius field is where the
# memory is — hence the second bound, `WORKSPACE_POOL_BYTES`.
const POOL_SLOTS_PER_KEY = 2

"""
    AutoRIFT.WORKSPACE_POOL_BYTES

Bytes of workspace the pool may retain across every geometry, default 2 GiB. `typemax(Int)`
removes the bound, leaving only `POOL_SLOTS_PER_KEY` per geometry.

The per-geometry limit does not bound the number of geometries a run visits. A whole NISAR L1 grid
visits 105 and would retain 9.9 GiB. At 2 GiB, peak footprint on that grid fell from 42.06 to
39.05 GiB (-7.2%) with the same wall clock and CPU time to 0.1%. `docs/src/explanation/memory.md`
has the table.

The bound does not change results: a rebuilt workspace has the same size as the one dropped, so it
runs the same transform.

Eviction is least-recently-used by geometry. The geometry just returned is never evicted, so a
single geometry larger than the bound stays pooled. `tools/golden/pool_peak.jl --budget` measures
the effect.
"""
const WORKSPACE_POOL_BYTES = Ref(2 * 1024^3)

const WORKSPACE_LOCK = ReentrantLock()

# Keys in use order, least-recently-used first. Inert while the bound is `typemax`.
const WORKSPACE_RECENCY = Tuple{Extent,Extent,Bool}[]
# Keyed by (chip, radius, iscomplex) so a checked-out workspace is byte-for-byte what `workspace`
# would have built. The buffers still carry no image element type — the chip is `Float32` because it
# is stored mean-removed, the integral images are `Float64` — but `workspace` allocates only the set
# its `T` can reach, so a real-built and a complex-built workspace of the same geometry are *not*
# interchangeable. Without the flag a process running both measures over one geometry would hand a
# `Coherence` pass a workspace whose complex buffers are `0x0`, and the failure would be an
# out-of-bounds write rather than a `DimensionMismatch`.
#
# A `Bool` rather than `T` itself: what the allocation depends on is real-versus-complex, so keying on
# `T` would split `UInt8` from `Int16` and hold two identical workspaces per geometry.
# `Vector` rather than a single slot because several chunks run concurrently.
const WORKSPACE_POOL = Dict{Tuple{Extent,Extent,Bool},Vector{CorrelationWorkspace}}()

"""
    AutoRIFT.WORKSPACE_LIVE_BYTES

Bytes of correlation workspace that may be checked out at once, across all tasks. `typemax(Int)`
(the default) removes the bound.

This differs from `AutoRIFT.WORKSPACE_POOL_BYTES`, which bounds what the pool retains between uses.
A task holds one workspace at a time, sized by the radius bucket it is in, so a run holds as many
workspaces as it has tasks. On NISAR L2 blocked at `2304x1152`, live workspaces are about 4.5 GiB
of 7.8 GiB above the process floor, three times what the pool retains.

A task that would exceed the bound waits for another task to return its workspace. No workspace is
resized and no point is correlated differently, so results do not change; some tasks wait when all
want a wide bucket at once.

One task always proceeds, however large its workspace, so a bound below a single workspace's size
does not deadlock.

The size of each geometry is its measured `workspace_bytes`. The first workspace of a geometry is
admitted before its size is known, so the bound can be exceeded by one workspace until then.
"""
const WORKSPACE_LIVE_BYTES = Ref(typemax(Int))

# Bytes currently checked out, and the measured size of each geometry seen so far. Both are only
# maintained while the bound is in force.
const WORKSPACE_LIVE = Ref(0)
const WORKSPACE_KEY_BYTES = Dict{Tuple{Extent,Extent,Bool},Int}()
# Signalled by every return, waited on by a take that would exceed the bound. Backed by the pool lock, so
# a waiter releases it and the returning task can make progress.
const WORKSPACE_ROOM = Threads.Condition(WORKSPACE_LOCK)

# Whether the in-use bound is in force. Unset by default, and the guard keeps its bookkeeping — a dictionary
# lookup and a counter, both under the pool lock — off the default path entirely.
@inline _live_bounded() = WORKSPACE_LIVE_BYTES[] != typemax(Int)

# Reserve room for a workspace of this geometry, returning what was reserved. Called with the lock held.
#
# `get(..., 0)` for an unseen geometry, which admits it immediately: its size is unknown until one is built,
# and waiting on an unknown would either deadlock or need a predictor.
function _reserve_live!(key)
    want = get(WORKSPACE_KEY_BYTES, key, 0)
    # `WORKSPACE_LIVE[] > 0` is what guarantees progress: the first task in never waits, so a bound below a
    # single workspace costs nothing but is not a hang.
    while WORKSPACE_LIVE[] > 0 && WORKSPACE_LIVE[] + want > WORKSPACE_LIVE_BYTES[]
        wait(WORKSPACE_ROOM)
    end
    WORKSPACE_LIVE[] += want
    return want
end

# Correct the reservation once the workspace exists and its size is known, and learn that size for next
# time. Reserving then correcting keeps the counter symmetric with `give_workspace!`, which subtracts the
# same measured figure.
function _settle_live!(ws::CorrelationWorkspace, key, reserved::Int)
    actual = workspace_bytes(ws)
    lock(WORKSPACE_LOCK) do
        WORKSPACE_KEY_BYTES[key] = actual
        WORKSPACE_LIVE[] += actual - reserved
    end
    return nothing
end

"""
    AutoRIFT.live_bytes() -> Int

Bytes of correlation workspace currently checked out. `0` unless
`AutoRIFT.WORKSPACE_LIVE_BYTES` is set, since the counter is not maintained otherwise.
"""
live_bytes() = lock(WORKSPACE_LOCK) do
    WORKSPACE_LIVE[]
end

"""
    AutoRIFT.take_workspace!(T, chip_size, search_radius) -> CorrelationWorkspace

A workspace for exactly this geometry, from the pool or newly built.

Return it with [`give_workspace!`](@ref) when done. The caller has exclusive use until then.
Thread safe. Blocks while `AutoRIFT.WORKSPACE_LIVE_BYTES` is exceeded.
"""
function take_workspace!(::Type{T}, chip_size, search_radius) where {T<:ImageElement}
    # Normalized through `extent`, so a scalar, a plain tuple and a named tuple all reach the same
    # pool entry rather than three. `give_workspace!` keys on the workspace's own fields, so this has
    # to produce the same thing — including the complex flag, which it recovers from the buffers.
    key = (extent(chip_size), extent(search_radius), !(T <: Real))
    bounded = _live_bounded()
    reserved = 0
    ws = lock(WORKSPACE_LOCK) do
        # Admission before the pool is consulted, because a pooled workspace occupies the same bytes as a
        # freshly built one — what the bound limits is how many are *out*, not where they came from.
        bounded && (reserved = _reserve_live!(key))
        pool = get(WORKSPACE_POOL, key, nothing)
        (isnothing(pool) || isempty(pool)) && return nothing
        taken = pop!(pool)
        # Touched on a take as well as on a return: a key whose workspaces are all checked out is in
        # active use, and letting it age to the front of the eviction order would drop the entry the
        # moment one came back.
        _pool_bounded() && _touch_workspace_key!(key)
        return taken
    end
    # Built outside the lock: planning and allocation are slow, and holding the lock across them
    # would serialise exactly the burst of concurrent chunks this exists to serve.
    out = isnothing(ws) ? workspace(T, chip_size, search_radius) : ws
    bounded && _settle_live!(out, key, reserved)
    return out
end

"""
    AutoRIFT.give_workspace!(ws)

Return a workspace to the pool, or drop it if that geometry already holds `POOL_SLOTS_PER_KEY`
workspaces. Its buffers are not cleared: each is fully written before it is read on the next use.
The caller must not use `ws` afterwards.
"""
function give_workspace!(ws::CorrelationWorkspace)
    lock(WORKSPACE_LOCK) do
        key = (ws.max_chip, ws.max_radius, iscomplexworkspace(ws))
        if _live_bounded()
            # The same measured figure `_settle_live!` added, so the counter cannot drift across a
            # take/return pair however the workspace was obtained.
            WORKSPACE_LIVE[] -= workspace_bytes(ws)
            # Every return, not only the ones that free room: a waiter cannot tell which return was its
            # own, and `notify` on an empty wait queue costs nothing.
            notify(WORKSPACE_ROOM; all = true)
        end
        pool = get!(() -> CorrelationWorkspace[], WORKSPACE_POOL, key)
        if length(pool) < POOL_SLOTS_PER_KEY
            push!(pool, ws)
            _pool_bounded() && _evict_to_budget!(key)
        end
    end
    return nothing
end

# Whether the across-keys bound is in force. Unset by default, and the guard is what keeps its
# bookkeeping — a recency scan and a sum over the pool, both under the pool lock — off the default path.
@inline _pool_bounded() = WORKSPACE_POOL_BYTES[] != typemax(Int)

"""
    AutoRIFT.pool_bytes() -> Int

Bytes of workspace the pool currently retains, summed over its entries.
"""
pool_bytes() = lock(WORKSPACE_LOCK) do
    sum(workspace_bytes, Iterators.flatten(values(WORKSPACE_POOL)); init = 0)
end

# Drop least-recently-used keys until the pool is inside its bound. Called with the lock held.
#
# `keep` is exempt, being the key whose workspace was just returned. Stopping when it is the only
# candidate left is what keeps a single over-budget geometry pooled instead of rebuilt per chunk: such a
# run holds that much whatever this does, and dropping it would add a GiB-scale allocation to every chunk.
function _evict_to_budget!(keep)
    _touch_workspace_key!(keep)
    held = sum(workspace_bytes, Iterators.flatten(values(WORKSPACE_POOL)); init = 0)
    while held > WORKSPACE_POOL_BYTES[]
        i = findfirst(k -> k != keep && haskey(WORKSPACE_POOL, k), WORKSPACE_RECENCY)
        isnothing(i) && break
        victim = WORKSPACE_RECENCY[i]
        deleteat!(WORKSPACE_RECENCY, i)
        for w in pop!(WORKSPACE_POOL, victim)
            held -= workspace_bytes(w)
        end
    end
    return nothing
end

# Move `key` to the most-recently-used end. Linear over the distinct geometries a run reaches — a hundred
# at worst against a chunk's hundreds of milliseconds.
function _touch_workspace_key!(key)
    i = findfirst(==(key), WORKSPACE_RECENCY)
    isnothing(i) || deleteat!(WORKSPACE_RECENCY, i)
    push!(WORKSPACE_RECENCY, key)
    return nothing
end

"""
    AutoRIFT.workspace_bytes(ws) -> Int

Bytes of buffer `ws` holds, recorded at construction as the sum of its fields' sizes.
"""
workspace_bytes(ws::CorrelationWorkspace) = ws.bytes[]

"""
    AutoRIFT.iscomplexworkspace(ws) -> Bool

Whether `ws` was built for complex imagery, and so carries the [`Coherence`](@ref) buffers rather
than the real ones. Read from `cchip`, which is empty for a real workspace.
"""
iscomplexworkspace(ws::CorrelationWorkspace) = !isempty(ws.cchip)

"""
    AutoRIFT.clear_workspaces!()

Drop every pooled workspace. For tests and benchmarks that measure allocation; not part of
normal operation.
"""
function clear_workspaces!()
    lock(WORKSPACE_LOCK) do
        empty!(WORKSPACE_POOL)
        empty!(REFINEMENT_POOL)
        empty!(WORKSPACE_RECENCY)
    end
    return nothing
end

"""
    next_fft_size(n) -> Int

A transform length `>= n` that FFTW handles efficiently, at most `1.5n`.

Candidates are products of 2, 3, 5 and 7, scored by `m² log₂ m` weighted by how much of `m` is a
power of two, since FFTW's radix-2 codelets are the fastest. The result is not always the smallest
smooth length: `next_fft_size(81)` is 84 = 2²·3·7, not 81 = 3⁴, which measures 14.1 µs against 8.7 µs
for 96 = 2⁵·3. 2-heavy lengths are 1.1 to 1.5 times faster across the window extents this package
uses. A power of two is not always chosen either (not 128 for `n = 81`).
"""
function next_fft_size(n::Integer)
    n <= 1 && return 1
    nn = Int(n)
    best, best_cost = 0, Inf
    for m in nn:ceil(Int, 1.5nn)
        r = m
        twos = 0
        while iseven(r)
            r ÷= 2
            twos += 1
        end
        for p in (3, 5, 7)
            while r % p == 0
                r ÷= p
            end
        end
        r == 1 || continue
        # Work grows as m^2 log m for a 2-D transform. The penalty discounts that by how
        # much of `m` is radix-2, since those codelets are the well-optimised ones: a
        # factorisation that is all twos runs at roughly two-thirds the cost per point of
        # one with none. Calibrated against measured rFFT timings at the window extents this
        # package uses (43, 81, 113, 163, 177), where it reproduces the measured winner.
        frac2 = twos / log2(m)
        cost = m^2 * log2(m) * (1.0 - 0.35frac2)
        if cost < best_cost
            best, best_cost = m, cost
        end
    end
    # A power of two always lies within [n, 1.5n) for n >= 2, so this always succeeds.
    return best
end

# The transform a chip of extent `chip` searched at radius `(ry, rx)` is correlated through.
#
# One home for the mapping, because three callers need it and they must agree exactly: the plan warmed
# ahead of a pass (`_warm_pass_plans`), the plan warmed for a whole grid (`_warm_grid_plans`), and the
# set `precompute_fftw_wisdom` measures up front. A padding convention that differs between them builds
# a plan at a size nothing ever executes, which costs the measurement it was meant to save and raises
# no error.
#
# `2r - 1` is the full cross-correlation support: a chip of `n` against a window reaching `r` either
# side overlaps at `n + 2r - 1` positions. The result is `(row, column)` because an FFT buffer is a
# matrix, where the extent is x-first — naming both axes at every call site is what keeps that
# transposition visible.
_padded_fft_size(chip, ry::Integer, rx::Integer) =
    (next_fft_size(chip.Y + 2ry - 1), next_fft_size(chip.X + 2rx - 1))

"""
    AutoRIFT.RADIUS_BUCKET_RUNGS

Rungs per octave (4) on the ladder [`AutoRIFT._radius_bucket`](@ref) rounds a search radius up to.

Coarse rungs correlate a point through a larger transform than it needs. Fine rungs produce more
distinct transform lengths, each needing its own `FFTW_MEASURE` plan and `WORKSPACE_POOL` entry.
Search radius is skewed on a Geogrid-derived grid: on the NISAR L1 golden pair the median searched
point has radius 34 against a maximum of 1905, and the 1% of points above 1024 carry 40% of the
transform work.

On that pair, whole grid, twelve threads, the run took 590.2 s with one rung per octave and 467.2 s
with four, over the same 1,799,742 points. Four rungs raise the plan set from 53 to 388 buckets. The
extra planning is a one-time cost per machine (110 s on that grid), which [`fftw_wisdom_path`](@ref)
removes from later runs. Eight rungs per octave would need 996 buckets.
"""
const RADIUS_BUCKET_RUNGS = 4

"""
    AutoRIFT._radius_bucket(r, cap) -> Int

The search radius a point of radius `r` is correlated at: `r` rounded up to the next rung of the
`AutoRIFT.RADIUS_BUCKET_RUNGS`-per-octave ladder, never above `cap`.

The rungs are the powers of two and the evenly spaced steps between consecutive ones, so at four per
octave they run 16, 20, 24, 28, 32, 40, 48, 56, 64. The arithmetic is integer.

A point is not correlated at its own exact radius because each distinct radius would need its own
FFTW plan (116 to 347 ms for a new size at the `FFTW_PATIENT` sizes) and its own `WORKSPACE_POOL`
entry. A real scene reaches a few hundred buckets, each reused by thousands of points.

`cap` is the pass maximum. Without it, rounding can exceed the extent the pass was sized for (radius
1905 rounds to 2048, which needs a 2304x4608 transform where the pass geometry needs 1792x4096). With
it, bucketing never costs more memory or work than the pass geometry.

Zero maps to zero: a non-positive radius marks an unsearchable point (see [`AutoRIFT.sanitize!`](@ref)),
which is skipped before any workspace is taken. The bucket depends on the point alone, so a subset
of a point set is correlated as it would be within the whole set (see
[`AutoRIFT.PassGeometry`](@ref)).
"""
@inline function _radius_bucket(r::Integer, cap::Integer)
    r <= 0 && return 0
    r >= cap && return Int(cap)
    ri = Int(r)
    # The power of two at or below `r`. A radius already on a rung is its own bucket, which is what
    # keeps the ladder from rounding a power of two up to the next step above it.
    h = 1 << (8 * sizeof(ri) - leading_zeros(ri) - 1)
    ri == h && return h
    for s in 1:(RADIUS_BUCKET_RUNGS - 1)
        m = (h * (RADIUS_BUCKET_RUNGS + s)) ÷ RADIUS_BUCKET_RUNGS
        m >= ri && return min(m, Int(cap))
    end
    return min(h << 1, Int(cap))
end

# ---------------------------------------------------------------------------
# Chip preparation
# ---------------------------------------------------------------------------

"""
    prepare_chip!(ws, chip) -> (norm, ok)

Copy `chip` into the workspace with its mean removed, and return the Euclidean
norm of the result together with whether the chip carries any signal.

`ok` is `false` for a constant chip. Such a chip has zero variance, so the correlation coefficient
is undefined at every shift. The reference implementation (v2.0.0 onward) reports a peak at the
corner of the surface for these chips, giving a displacement pinned to the corner of the search
window. Here the chip is reported as having no measurement. Throws `ArgumentError` if `ws` was built
for complex imagery, and `DimensionMismatch` if `chip` exceeds the workspace's chip extent.
"""
function prepare_chip!(ws::CorrelationWorkspace, chip::AbstractMatrix)
    ch, cw = size(chip)
    # The mirror of the complex method's check: an empty `chip` means this workspace was built for
    # complex imagery, which allocates the `Coherence` buffers instead of these.
    isempty(ws.chip) && throw(ArgumentError(
        "this workspace was built for complex imagery, so it has no real buffers, and the chip is " *
        "real. Build it with `workspace(Float32, ...)` — or, through the pool, " *
        "`take_workspace!(Float32, ...)`."))
    (ch <= size(ws.chip, 1) && cw <= size(ws.chip, 2)) || throw(DimensionMismatch(
        "chip is $(ch)x$(cw) but the workspace was built for at most " *
        "$(size(ws.chip, 1))x$(size(ws.chip, 2))"))

    dst = @view ws.chip[1:ch, 1:cw]

    # Mean in Float64 regardless of input type: it is subtracted from every
    # element, so an error here biases the whole numerator rather than averaging
    # out.
    #
    # Four accumulators down each column rather than one. A single running sum makes each
    # add wait for the previous one to retire, so the loop runs at the latency of the
    # floating-point adder — measured ~1.6 ns per element, far off what the memory can
    # supply. Four independent chains fill those slots and are worth 1.6-1.9x.
    #
    # Combined pairwise, which is *more* accurate than sequential summation, not less: the
    # partial sums are of comparable magnitude, so the final adds lose fewer low bits. The
    # mean-removed copy this writes is bit-identical to a serial accumulation; the norm
    # differs at the last bit or two.
    #
    # Row-blocked and indexed two-dimensionally, not linearly. `chip` is a strided view into
    # the search image, whose columns are not adjacent, so it is `IndexCartesian` — linear
    # indexing would divide and modulo per element and lose more than the unrolling gains.
    tail = ch - ch % 4
    s = 0.0
    @inbounds for j in 1:cw
        s0 = s1 = s2 = s3 = 0.0
        for i in 1:4:tail
            s0 += Float64(chip[i, j])
            s1 += Float64(chip[i + 1, j])
            s2 += Float64(chip[i + 2, j])
            s3 += Float64(chip[i + 3, j])
        end
        for i in (tail + 1):ch
            s0 += Float64(chip[i, j])
        end
        s += (s0 + s1) + (s2 + s3)
    end
    mean = s / (ch * cw)

    sq = 0.0
    @inbounds for j in 1:cw
        q0 = q1 = q2 = q3 = 0.0
        for i in 1:4:tail
            d0 = Float64(chip[i, j]) - mean
            d1 = Float64(chip[i + 1, j]) - mean
            d2 = Float64(chip[i + 2, j]) - mean
            d3 = Float64(chip[i + 3, j]) - mean
            dst[i, j] = Float32(d0)
            dst[i + 1, j] = Float32(d1)
            dst[i + 2, j] = Float32(d2)
            dst[i + 3, j] = Float32(d3)
            q0 += d0 * d0
            q1 += d1 * d1
            q2 += d2 * d2
            q3 += d3 * d3
        end
        for i in (tail + 1):ch
            d = Float64(chip[i, j]) - mean
            dst[i, j] = Float32(d)
            q0 += d * d
        end
        sq += (q0 + q1) + (q2 + q3)
    end

    # Zero within rounding, not exactly zero: a chip of nearly-identical values
    # gives a norm at the scale of the rounding error, and dividing by it would
    # amplify noise into a spurious peak. `eps` scaled by the element count is the
    # magnitude below which the sum of squares is indistinguishable from zero.
    ok = sq > eps(Float64) * ch * cw
    return sqrt(sq), ok
end

# The complex chip, into the workspace's own complex buffer.
#
# Structurally the real version with a complex accumulator, and a separate method for a narrower
# reason than the one first written here.
#
# The original justification — that a generic method "must not acquire a `Complex` union or an extra
# branch" in the real hot path — is **wrong**, and measurement is what settled it: a single method
# widening through a two-line `_w(::Real)`/`_w(::Complex)` helper is bit-identical *and*
# timing-identical on both `Float32` and `UInt8` chips (27.8 vs 27.9 us, 29.1 vs 29.0 us on 200²).
# No union ever reaches the real path, because Julia specializes on the concrete element type at
# each call site. Type *parameterisation* was the available tool; a `Union` was never the choice.
#
# What does justify two methods is the *destination*: this one writes `ws.cchip` and the real one
# `ws.chip`, which are different fields rather than different types. Unifying would mean dispatching
# the buffer as well as the accumulator, at which point two 20-line bodies read more clearly than
# one with two traits. That is a legibility judgement, and it is worth stating as one rather than
# dressing it as a performance requirement.
#
# The mean is complex — subtracting it removes the *constant* phasor as well as the DC amplitude,
# which is what makes coherence invariant to a global phase offset between the two acquisitions.
# That matters: an SLC pair generally has an arbitrary absolute phase difference, and a measure
# sensitive to it would report nothing useful.
#
# The norm is `sqrt(Σ|z|²)`, so it is real even though the data are not, and `Coherence`'s
# denominator therefore has the same shape as `ZNCC`'s.
function prepare_chip!(ws::CorrelationWorkspace, chip::AbstractMatrix{<:Complex})
    ch, cw = size(chip)
    # An empty `cchip` is a workspace built for real imagery, not one built too small: `workspace`
    # allocates the complex buffers only for a complex `T`. Named separately because the geometry
    # message below would send a reader looking at chip sizes for a mistake in the element type.
    isempty(ws.cchip) && throw(ArgumentError(
        "this workspace was built for real imagery, so it has no complex buffers, and the chip is " *
        "complex. Build it with `workspace(ComplexF32, ...)` — or, through the pool, " *
        "`take_workspace!(ComplexF32, ...)` — so the `Coherence` buffers are allocated."))
    (ch <= size(ws.cchip, 1) && cw <= size(ws.cchip, 2)) || throw(DimensionMismatch(
        "chip is $(ch)x$(cw) but the workspace was built for at most " *
        "$(size(ws.cchip, 1))x$(size(ws.cchip, 2))"))

    dst = @view ws.cchip[1:ch, 1:cw]

    s = zero(ComplexF64)
    @inbounds for j in 1:cw, i in 1:ch
        s += ComplexF64(chip[i, j])
    end
    mean = s / (ch * cw)

    sq = 0.0
    @inbounds for j in 1:cw, i in 1:ch
        d = ComplexF64(chip[i, j]) - mean
        dst[i, j] = ComplexF32(d)
        sq += abs2(d)
    end

    ok = sq > eps(Float64) * ch * cw
    return sqrt(sq), ok
end

# ---------------------------------------------------------------------------
# The surface
# ---------------------------------------------------------------------------

"""
    correlate!(ws, search, chip, radius; measure = ZNCC()) -> surface

Correlate `chip` against `search`, returning a view of the workspace's surface buffer holding the
similarity at every candidate shift.

`radius` is `(radius_x, radius_y)` or an `Int`. The returned surface is `(2 * radius_y, 2 * radius_x)`
with zero displacement at its center; see `AutoRIFT.peak_offset` for converting an index to a
displacement. `search` must be `chip + 2 * radius - 1` in each dimension, otherwise
`DimensionMismatch` is thrown, as it is when the radius exceeds the workspace's.

Returns a surface of zeros if the chip is constant, and sets [`AutoRIFT.degenerate`](@ref) so the
caller can tell that from a genuine zero correlation.

Allocation-free. The result aliases the workspace, so it must be consumed or copied before the next
call on the same workspace.
"""
function correlate!(
    ws::CorrelationWorkspace, search::AbstractMatrix, chip::AbstractMatrix,
    radius::Tuple{Int,Int}; measure::SimilarityMeasure = ZNCC(),
)
    ch, cw = size(chip)
    rx, ry = radius
    nr, nc = 2ry, 2rx

    size(search) == (ch + 2ry - 1, cw + 2rx - 1) || throw(DimensionMismatch(
        "search window is $(size(search)) but a $(ch)x$(cw) chip with radius " *
        "($rx, $ry) requires $((ch + 2ry - 1, cw + 2rx - 1)). The window is " *
        "deliberately asymmetric: it extends `radius` one way and `radius - 1` " *
        "the other, so the surface is an even 2*radius and zero displacement " *
        "lands on a sample."))
    (nr <= size(ws.surface, 1) && nc <= size(ws.surface, 2)) ||
        throw(DimensionMismatch(
            "radius ($rx, $ry) needs a $(nr)x$(nc) surface but the workspace " *
            "was built for at most $(size(ws.surface))"))

    surface = @view ws.surface[1:nr, 1:nc]
    cnorm, ok = prepare_chip!(ws, chip)
    ws.was_degenerate[] = !ok
    if !ok
        fill!(surface, 0.0f0)
        return surface
    end

    _correlate_surface!(surface, ws, search, ch, cw, cnorm, measure)
    return surface
end

correlate!(ws, search, chip, radius::Integer; kw...) =
    correlate!(ws, search, chip, (Int(radius), Int(radius)); kw...)

"""
    correlate_prepared!(ws, search, chip, radius; measure) -> surface

[`AutoRIFT.correlate!`](@ref) for a window [`AutoRIFT.prepare_window!`](@ref) has already been
called on with the same `search` and `measure`.

Skips the window's forward transform and integral tables, about a third of the work, which do not
depend on the chip. The result is bit-identical to `correlate!`.

The caller must pass the same `search` that `prepare_window!` saw. Nothing checks this, and a
different window reads a stale spectrum and returns a surface for the wrong window. The size check
of `correlate!` is not repeated.
"""
function correlate_prepared!(
    ws::CorrelationWorkspace, search::AbstractMatrix, chip::AbstractMatrix,
    radius::Tuple{Int,Int}; measure::SimilarityMeasure = ZNCC(),
)
    ch, cw = size(chip)
    rx, ry = radius
    nr, nc = 2ry, 2rx
    surface = @view ws.surface[1:nr, 1:nc]
    cnorm, ok = prepare_chip!(ws, chip)
    ws.was_degenerate[] = !ok
    if !ok
        fill!(surface, 0.0f0)
        return surface
    end
    _correlate_surface!(surface, ws, search, ch, cw, cnorm, measure, Val(true))
    return surface
end

correlate_prepared!(ws, search, chip, radius::Integer; kw...) =
    correlate_prepared!(ws, search, chip, (Int(radius), Int(radius)); kw...)

"""
    degenerate(ws::CorrelationWorkspace) -> Bool

Whether the last [`correlate!`](@ref) on `ws` was given a chip with no signal.

A constant chip has zero variance, so the correlation coefficient is undefined at every shift and
the surface is filled with zeros. Those zeros are not a measurement of zero displacement.
"""
@inline degenerate(ws::CorrelationWorkspace) = ws.was_degenerate[]

# Everything a measure derives from the search window alone, computed once for a window that
# several chips will be correlated against.
#
# A third of a `correlate!` is this: the window's forward transform is 23% and the integral tables
# another 10%, and neither depends on the chip. Only `_correlate_rotations!` has more than one chip
# per window, so only it calls this; `correlate!` still does the work inline, where the split would
# buy a caller with one chip nothing.
#
# Dispatched on the measure because the tables differ — `NCC` needs no plain sum, and `Coherence`
# needs a complex one.
function prepare_window!(ws::CorrelationWorkspace, search::AbstractMatrix, ::ZNCC)
    S = @view ws.isum[1:(size(search, 1) + 1), 1:(size(search, 2) + 1)]
    S2 = @view ws.isqsum[1:(size(search, 1) + 1), 1:(size(search, 2) + 1)]
    integral_both!(S, S2, search)
    _window_spectrum!(ws, search)
    return nothing
end

function prepare_window!(ws::CorrelationWorkspace, search::AbstractMatrix, ::NCC)
    S2 = @view ws.isqsum[1:(size(search, 1) + 1), 1:(size(search, 2) + 1)]
    integral_sq!(S2, search)
    _window_spectrum!(ws, search)
    return nothing
end

# Dispatch on the measure so each formula compiles to its own kernel with no
# runtime branch in the inner loop.
#
# `prepared` says the window-derived work is already done — `ws.wspec` holds its spectrum and the
# integral tables are filled — so this must not recompute either. It is a compile-time `Bool` type
# parameter rather than a runtime flag so that the branch vanishes in both directions.
function _correlate_surface!(
    surface, ws::CorrelationWorkspace, search::AbstractMatrix,
    ch::Int, cw::Int, cnorm::Float64, ::ZNCC, prepared::Val = Val(false),
)
    nr, nc = size(surface)
    n = ch * cw
    cd = @view ws.chip[1:ch, 1:cw]

    S = @view ws.isum[1:(size(search, 1) + 1), 1:(size(search, 2) + 1)]
    S2 = @view ws.isqsum[1:(size(search, 1) + 1), 1:(size(search, 2) + 1)]
    # One traversal for both tables: every caller needing the sum needs the squares too, and fusing
    # is 1.5-1.66x on the pair — 6-8% of a whole correlation. Bit-identical to two calls.
    prepared === Val(false) && integral_both!(S, S2, search)

    numerators = _numerators_or_prepared!(ws, search, cd, nr, nc, ch, cw, prepared)

    @inbounds for j in 1:nc, i in 1:nr
        s1 = boxsum(S, i, j, ch, cw)
        s2 = boxsum(S2, i, j, ch, cw)
        # Σ(W - W̄)² = ΣW² - (ΣW)²/n. Clamped at zero: the two terms are large and
        # nearly equal for a low-contrast window, so rounding can make the
        # difference slightly negative even in Float64, and `sqrt` of that is NaN.
        wvar = max(s2 - s1 * s1 / n, 0.0)
        den = cnorm * sqrt(wvar)
        # `ifelse` rather than a branch, and it is worth 1.8-2.1x on this loop. A branch here is
        # unpredictable — whether a window has contrast depends on the imagery — and it also
        # blocks vectorization, so the loop pays a mispredict on top of running one shift at a
        # time. `ifelse` evaluates both arms and selects, which is safe because the division is
        # the only cost and `den > 0` is exactly the guard that makes it finite; a zero `den`
        # yields `Inf` or `NaN` in the discarded arm and the select drops it.
        surface[i, j] = ifelse(den > 0, Float32(numerators[i, j] / den), 0.0f0)
    end
    return surface
end

function _correlate_surface!(
    surface, ws::CorrelationWorkspace, search::AbstractMatrix,
    ch::Int, cw::Int, cnorm::Float64, ::NCC, prepared::Val = Val(false),
)
    # NCC differs from ZNCC only in the denominator: the raw sum of squares rather
    # than the variance about the window mean. The numerator is the same
    # correlation, so the whole difference is one term.
    #
    # Note that `ws.chip` still holds the *mean-removed* chip, so this is not
    # OpenCV's TM_CCORR_NORMED. It is retained for comparison rather than as a
    # reference target: v2.1.2 uses ZNCC for every input type, and the DC-sensitive
    # variant older releases used for floating-point input was a bug.
    nr, nc = size(surface)
    cd = @view ws.chip[1:ch, 1:cw]

    S2 = @view ws.isqsum[1:(size(search, 1) + 1), 1:(size(search, 2) + 1)]
    prepared === Val(false) && integral_sq!(S2, search)

    numerators = _numerators_or_prepared!(ws, search, cd, nr, nc, ch, cw, prepared)

    @inbounds for j in 1:nc, i in 1:nr
        wsq = boxsum(S2, i, j, ch, cw)
        den = cnorm * sqrt(max(wsq, 0.0))
        # `ifelse`, for the reason given in the `ZNCC` method.
        surface[i, j] = ifelse(den > 0, Float32(numerators[i, j] / den), 0.0f0)
    end
    return surface
end

# Complex coherence magnitude. Requires complex input; see `Coherence` for why it is worth having
# and when it fails.
#
# Structurally `ZNCC` over complex data, and deliberately so — the same three rearrangements the
# file header describes still apply:
#
#   * The chip's complex mean and its norm are computed once, in `prepare_chip!`.
#   * **The window's mean still drops out of the numerator**, for the same reason and with the
#     same proof: `Σ conj(T')(W - W̄) = Σ conj(T')W - W̄ Σ conj(T')`, and `Σ conj(T') = 0` because
#     the chip was mean-removed. So the numerator correlates against the *unmodified* window.
#   * The window variance comes from integral images: `Σ|W - W̄|² = Σ|W|² - |ΣW|²/n`. This is
#     where complex differs — it needs the complex sum table `cisum` as well as `isqsum`, since
#     `|ΣW|²` is the squared magnitude of a *complex* sum.
#
# Getting that last term wrong is not a rounding matter. Normalising by `Σ|W|²` instead — removing
# the mean from the chip but not the window — caps `γ(T, T)` below 1 by exactly `‖T'‖/‖T‖`, which
# measured 0.9983 on a speckle chip. Predicted and observed agreed to 8 digits, which is how the
# omission was found; the `γ(T,T) == 1` test in `test/complex.jl` is what keeps it found.
#
# The one genuine difference from the real measures: the numerator is `Σ conj(T') · W`, a *complex*
# accumulation whose magnitude is taken per shift. Taking `abs` at the end rather than accumulating
# magnitudes is the entire point — summing `|conj(T')·W|` would discard the phase alignment that
# makes the peak sharp and would give a surface that is large everywhere.
#
# Everything downstream is shared: the surface is real `Float32`, so `peak_index`, `peak_offset`
# and the whole Laplace subpixel cascade are the code the real path uses.
#
function _correlate_surface!(
    surface, ws::CorrelationWorkspace, search::AbstractMatrix{<:Complex},
    ch::Int, cw::Int, cnorm::Float64, ::Coherence,
)
    nr, nc = size(surface)
    n = ch * cw
    cd = @view ws.cchip[1:ch, 1:cw]

    S = @view ws.cisum[1:(size(search, 1) + 1), 1:(size(search, 2) + 1)]
    S2 = @view ws.isqsum[1:(size(search, 1) + 1), 1:(size(search, 2) + 1)]
    integral_both!(S, S2, search)

    numerators = _cnumerators!(ws, search, cd, nr, nc, ch, cw)

    @inbounds for j in 1:nc, i in 1:nr
        s1 = boxsum(S, i, j, ch, cw)
        s2 = boxsum(S2, i, j, ch, cw)
        # Σ|W - W̄|² = Σ|W|² - |ΣW|²/n. Clamped for the same cancellation reason as `ZNCC`.
        wvar = max(s2 - abs2(s1) / n, 0.0)
        den = cnorm * sqrt(wvar)
        # `ifelse`, for the reason given in the `ZNCC` method.
        surface[i, j] = ifelse(den > 0, Float32(abs(numerators[i, j]) / den), 0.0f0)
    end
    return surface
end

"""
    COMPLEX_DIRECT_THRESHOLD

Work below which the complex numerator is evaluated directly, in the same `nshifts * chip_area`
units as [`DIRECT_THRESHOLD`](@ref).

The direct complex loop does four real multiplies and four adds per sample where the real one does
one of each, so the crossover with the transform is much lower: between 256 work (direct 217 ns,
FFT 254) and 576 (FFT 406 ns, direct 477), at radius 1 to 4 and chip 4 to 12.

The threshold is below every configuration the public API can reach (the smallest legal chip is 4
and the default `min_search_radius` is 6, giving 4x4x12x12 = 2304), so the complex path always
transforms. The direct path is used by `test/complex.jl` for verification.
"""
const COMPLEX_DIRECT_THRESHOLD = 512

# Direct or FFT, on the same criterion the real path uses but its own threshold. Both write
# `ws.cnumerator` and return the same view type, so the choice costs one branch per call and
# introduces no instability.
function _cnumerators!(ws::CorrelationWorkspace, search, cd, nr, nc, ch, cw)
    out = @view ws.cnumerator[1:nr, 1:nc]
    if nr * nc * ch * cw <= COMPLEX_DIRECT_THRESHOLD
        _cnumerators_direct!(out, search, cd, nr, nc, ch, cw)
    else
        _cnumerators_fft!(out, ws, search, cd, nr, nc, ch, cw)
    end
    return out
end

function _cnumerators_direct!(out, search, cd, nr, nc, ch, cw)
    @inbounds for j in 1:nc
        for i in 1:nr
            # `ComplexF64` accumulator, narrowed on store: the same treatment the real direct path
            # gives its `Float64` one.
            acc = zero(ComplexF64)
            for jj in 1:cw
                # Column-major inner loop, as in `_numerators_direct!`: both arrays walk
                # contiguous memory.
                for ii in 1:ch
                    acc += conj(ComplexF64(cd[ii, jj])) *
                           ComplexF64(search[i + ii - 1, j + jj - 1])
                end
            end
            out[i, j] = ComplexF32(acc)
        end
    end
    return out
end

# A real image has no phase, so `Coherence` on one is a caller error rather than something to
# silently degrade to `NCC`. Caught here, where the element type is known, with the fix named.
function _correlate_surface!(
    ::Any, ::CorrelationWorkspace, ::AbstractMatrix{<:Real},
    ::Int, ::Int, ::Float64, ::Coherence,
)
    throw(ArgumentError(
        "`Coherence` needs complex input, but the images are real. Coherence exploits the " *
        "phase of an SLC pair; a real image has none, so there is nothing for it to measure. " *
        "Use `similarity = :zncc` for real imagery, or pass the complex data."))
end

# ---------------------------------------------------------------------------
# The numerator: Σ T' · W, for every shift
# ---------------------------------------------------------------------------
#
# Two strategies with the same result to within rounding. Direct evaluation costs
# `nshifts * chip_area` multiply-accumulates; the FFT computes every shift at once
# for `O(N log N)` in the search-window area, which wins once the chip is large
# enough. The crossover is a measured constant, not a derived one — it depends on
# cache behaviour and on FFTW's plan quality — and is set by the M2 benchmark.
#
# The strategy is chosen once per call, on two integers, so it costs nothing per
# shift and introduces no type instability: both branches return a `Matrix{Float64}`
# view of workspace memory.

"""
    DIRECT_THRESHOLD

Work below which the numerator is evaluated directly rather than by FFT, measured as
`nshifts * chip_area` multiply-accumulates.

The FFT was faster at every combination tried, from chip 16 with radius 6 (2x) to chip 64 with
radius 50 (53x). At chip 32 with radius 25 the direct path is 5.2 million multiply-accumulates
against roughly fifty thousand butterflies for the transform. The threshold is therefore below any
configuration real data uses. The direct path remains for very small chips, for verification
(`test/correlate.jl`), and as a fallback needing no FFT plan.

Re-measure with `benchmark/suite/correlate.jl` if the numerator implementation changes.
"""
const DIRECT_THRESHOLD = 20_000

function _numerators!(ws::CorrelationWorkspace, search, cd, nr, nc, ch, cw)
    out = @view ws.numerator[1:nr, 1:nc]
    # One branch per call on two integers, not per shift, so the choice is free
    # and both paths return the same view type — no instability downstream.
    if nr * nc * ch * cw <= DIRECT_THRESHOLD
        _numerators_direct!(out, search, cd, nr, nc, ch, cw)
    else
        _numerators_fft!(out, ws, search, cd, nr, nc, ch, cw)
    end
    return out
end

# `Float32` accumulator, matching the FFT path this one is an alternative to — `_numerators_fft!`
# transforms in `Float32` throughout, so the two agree on the width the numerator is computed at
# rather than differing either side of `DIRECT_THRESHOLD`.
#
# The numerator tolerates a narrower accumulator than the denominator, and the asymmetry is the point:
# it is a sum of products of *mean-removed* values, so the terms straddle zero and no large constant
# accumulates for the sum to cancel against. The variance in `_correlate_surface!` is the opposite
# shape — `sumsq - sum^2/n` differences two large nearly-equal quantities — which is why the integral
# images stay `Float64`. See the head of `integral.jl`.
#
# `Float32` is also *exact* for the default geometry: a 16x16 `UInt8` chip accumulates to at most
# 16646400, inside the 2^24 where `Float32` represents every integer. Above that the relative error is
# ~1e-7, against a peak located to a fraction of a pixel from a 16x-upsampled surface.
function _numerators_direct!(out, search, cd, nr, nc, ch, cw)
    @inbounds for j in 1:nc
        for i in 1:nr
            acc = 0.0f0
            for jj in 1:cw
                # `search` is column-major, so the inner loop over rows walks
                # contiguous memory in both arrays.
                @simd for ii in 1:ch
                    acc += Float32(cd[ii, jj]) * Float32(search[i + ii - 1, j + jj - 1])
                end
            end
            out[i, j] = acc
        end
    end
    return out
end

# The search window's forward transform, into `ws.wspec`.
#
# Split from `_numerators_fft!` so a caller correlating several chips against one window — the
# angles of a `RotationSearch` — transforms it once. `wspec` stays valid until the next call with a
# different window.
function _window_spectrum!(ws::CorrelationWorkspace, search)
    sh, sw = size(search)
    fy, fx = size(ws.fbuf_a)
    a = ws.fbuf_a
    fill!(a, 0.0f0)
    @inbounds for j in 1:sw, i in 1:sh
        a[i, j] = Float32(search[i, j])
    end
    fft_execute!(fft_plan(fy, fx), a, ws.wspec)
    return ws.wspec
end

# The numerators, given a window spectrum already in `ws.wspec`.
function _numerators_fft_prepared!(out, ws::CorrelationWorkspace, cd, nr, nc, ch, cw)
    fy, fx = size(ws.fbuf_a)
    b = ws.fbuf_b
    fill!(b, 0.0f0)
    # Correlation is convolution with a reflected kernel, so the chip goes in
    # reversed. Placed at the origin, which puts the valid output at offset
    # (ch, cw) — see the read below.
    @inbounds for j in 1:cw, i in 1:ch
        b[i, j] = cd[ch - i + 1, cw - j + 1]
    end

    plan = fft_plan(fy, fx)
    fft_execute!(plan, b, ws.fspec_b)
    @inbounds for k in eachindex(ws.fspec_a)
        ws.fspec_a[k] = ws.wspec[k] * ws.fspec_b[k]
    end
    iplan = ifft_plan(fy, fx)
    # `ifft_execute!` applies the 1/n scaling FFTW omits, and destroys `ws.fspec_a` in the
    # process — nothing reads it afterwards.
    ifft_execute!(iplan, ws.fspec_a, b)

    @inbounds for j in 1:nc, i in 1:nr
        out[i, j] = Float64(b[i + ch - 1, j + cw - 1])
    end
    return out
end

function _numerators_fft!(out, ws::CorrelationWorkspace, search, cd, nr, nc, ch, cw)
    _window_spectrum!(ws, search)
    return _numerators_fft_prepared!(out, ws, cd, nr, nc, ch, cw)
end

# The numerators, taking the window spectrum from `ws.wspec` when a caller has already computed it.
# The direct path ignores the distinction: it reads the window itself and holds no spectrum.
function _numerators_or_prepared!(ws, search, cd, nr, nc, ch, cw, ::Val{false})
    return _numerators!(ws, search, cd, nr, nc, ch, cw)
end

function _numerators_or_prepared!(ws, search, cd, nr, nc, ch, cw, ::Val{true})
    out = @view ws.numerator[1:nr, 1:nc]
    if nr * nc * ch * cw <= DIRECT_THRESHOLD
        _numerators_direct!(out, search, cd, nr, nc, ch, cw)
    else
        _numerators_fft_prepared!(out, ws, cd, nr, nc, ch, cw)
    end
    return out
end

# The complex numerator, `Σ conj(T') · W` for every shift, by FFT.
#
# Written into a caller-supplied `ComplexF32` matrix rather than `ws.numerator`, which is
# `Float64`: the numerator is complex here and only becomes real when its magnitude is taken.
#
# Structurally the real version, with one correctness trap worth spelling out. Correlation is
# convolution with a *reflected* kernel, and the complex measure needs `conj(T')` — so the buffer
# takes the conjugate of the reversed chip. Conjugating and reversing commute, but forgetting
# either one produces a plausible surface with the wrong peak: reversal alone gives the
# *convolution* rather than the correlation, and conjugation alone mirrors the displacement.
# `test/complex.jl`'s known-shift cases catch both, and the direct/FFT agreement test catches any
# disagreement between the two paths.
function _cnumerators_fft!(out, ws::CorrelationWorkspace, search, cd, nr, nc, ch, cw)
    sh, sw = size(search)
    fy, fx = size(ws.cbuf_a)

    a, b = ws.cbuf_a, ws.cbuf_b
    fill!(a, zero(ComplexF32))
    fill!(b, zero(ComplexF32))
    @inbounds for j in 1:sw, i in 1:sh
        a[i, j] = ComplexF32(search[i, j])
    end
    @inbounds for j in 1:cw, i in 1:ch
        b[i, j] = conj(cd[ch - i + 1, cw - j + 1])
    end

    plan = cfft_plan(fy, fx)
    cfft_execute!(plan, a, ws.cspec_a)
    # The chip's spectrum lands in `a`, not a fourth buffer: `a` is dead the moment its own
    # transform has been read out, and it is still a *distinct* array from `b`, so the plan's
    # aliasing invariant holds. That is the difference between this and transforming in place —
    # see the workspace comment. Saves 9-10% of the workspace at every chip size, bit-identical
    # and runtime-neutral (three transforms and one elementwise pass either way).
    cfft_execute!(plan, b, a)
    @inbounds for k in eachindex(ws.cspec_a)
        ws.cspec_a[k] *= a[k]
    end
    iplan = icfft_plan(fy, fx)
    # Unnormalised in both directions, so the 1/n is applied on the read below rather than in a
    # separate pass over the whole buffer — only `nr*nc` of `fy*fx` values are ever used.
    cfft_execute!(iplan, ws.cspec_a, a)

    scale = 1.0f0 / (fy * fx)
    @inbounds for j in 1:nc, i in 1:nr
        out[i, j] = a[i + ch - 1, j + cw - 1] * scale
    end
    return out
end
