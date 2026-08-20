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

"""
    CorrelationWorkspace

Preallocated buffers for correlating one chip against one search window.

Everything the per-point kernel touches lives here and is reused across points, so
the inner loop allocates nothing. Sized for the largest chip and search radius a
pass will use; points needing less take views of the corner, which is why the
buffers are plain `Matrix` with runtime extents rather than sized types — a size in
the type would mean recompiling the kernel per chip size, and real runs use several.

Deliberately *not* parametrized on the image element type. None of these buffers has that
type: the chip is `Float32` because it is stored mean-removed, and the integral images are
`Float64`. Carrying a `T` that no field uses would make `CorrelationWorkspace{UInt8}` and
`CorrelationWorkspace{Int16}` distinct types and specialize every downstream method twice for
identical machine code. The element type reaches the kernel through the image arguments, which
is where it belongs.

Construct with [`workspace`](@ref).
"""
struct CorrelationWorkspace
    # Contiguous copy of the chip, mean-removed. A copy is needed anyway: the chip
    # is a strided view into the secondary image, and both the mean removal and
    # the FFT path require contiguous Float32.
    chip::Matrix{Float32}

    # Correlation surface, (2*radius_y, 2*radius_x) at full radius.
    surface::Matrix{Float32}

    # Integral images of the search window, sum and sum-of-squares. Float64: see
    # the note in integral.jl.
    isum::Matrix{Float64}
    isqsum::Matrix{Float64}

    # Raw correlation numerators, one per shift. Kept separate from the integral
    # images rather than sharing their storage: the integral images are read
    # throughout the normalisation loop, so overlapping them with the numerators
    # would corrupt the surface in a way that is invisible on small test inputs
    # and wrong on real ones. At 2*radius squared in Float64 this is ~20 kB at
    # radius 25 — small enough that the separation costs nothing worth having.
    numerator::Matrix{Float64}

    # FFT scratch. Sized to the padded search-window extent; unused by the direct
    # path, but allocated once either way so that a pass can switch strategies
    # without reallocating.
    fbuf_a::Matrix{Float32}
    fbuf_b::Matrix{Float32}
    fspec_a::Matrix{ComplexF32}
    fspec_b::Matrix{ComplexF32}

    # Maximum extents this workspace was built for, so a mismatched call is an
    # error rather than a silent out-of-bounds read.
    max_chip::Tuple{Int,Int}
    max_radius::Tuple{Int,Int}

    # Set by `correlate!` to record whether the last chip carried any signal. A mutable
    # cell so the workspace itself stays immutable and cheap to pass per point; callers
    # read it via `degenerate(ws)` rather than scanning the returned surface, which would
    # be a second pass over it at every grid point.
    was_degenerate::Base.RefValue{Bool}
end

"""
    workspace([T], chip_size, search_radius) -> CorrelationWorkspace

Allocate correlation buffers for chips up to `chip_size` and search radii up to
`search_radius`, each an `Int` or a `(x, y)` tuple.

One workspace per task, never shared: the buffers are written during correlation,
so two tasks sharing one would corrupt each other.

`T` is the element type of the images to be correlated. It is accepted for readability at the
call site and asserted to be a `Real`, but the buffers do not depend on it — the chip is
`Float32` because it is stored mean-removed, and the integral images are `Float64` to hold a
sum of squares without loss. Any `T<:Real` correlates correctly, including `Int16` and other
integer sensor types.
"""
workspace(chip_size, search_radius) = workspace(Float32, chip_size, search_radius)

function workspace(::Type{T}, chip_size, search_radius) where {T<:Real}
    csx, csy = chip_size isa Tuple ? chip_size : (chip_size, chip_size)
    rx, ry = search_radius isa Tuple ? search_radius : (search_radius, search_radius)

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

    return CorrelationWorkspace(
        Matrix{Float32}(undef, csy, csx),
        Matrix{Float32}(undef, 2ry, 2rx),
        Matrix{Float64}(undef, winy + 1, winx + 1),
        Matrix{Float64}(undef, winy + 1, winx + 1),
        Matrix{Float64}(undef, 2ry, 2rx),
        Matrix{Float32}(undef, fy, fx),
        Matrix{Float32}(undef, fy, fx),
        Matrix{ComplexF32}(undef, fy ÷ 2 + 1, fx),
        Matrix{ComplexF32}(undef, fy ÷ 2 + 1, fx),
        (csx, csy),
        (rx, ry),
        Ref(false),
    )
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
# A run visits three geometries, so the pool holds at most three entries per shape plus however
# many a burst of concurrent chunks needs. It is not bounded, and deliberately: bounding it would
# mean either blocking a task or falling back to allocation, and the natural bound — the number
# of chunks in flight — is already small and set by the thread count.

const WORKSPACE_LOCK = ReentrantLock()
# Keyed by (chip, radius) so a checked-out workspace is byte-for-byte what `workspace` would have
# built. Note the element type is *not* part of the key, and does not need to be: no buffer here
# has the image's type — the chip is `Float32` because it is stored mean-removed, the integral
# images are `Float64` — which is the same reason `CorrelationWorkspace` carries no `T`.
# `Vector` rather than a single slot because several chunks run concurrently.
const WORKSPACE_POOL = Dict{Tuple{Tuple{Int,Int},Tuple{Int,Int}},Vector{CorrelationWorkspace}}()

"""
    AutoRIFT.take_workspace!(T, chip_size, search_radius) -> CorrelationWorkspace

A workspace for exactly this geometry, from the pool or newly built.

Return it with [`give_workspace!`](@ref) when done. The caller has exclusive use until then, so
concurrent chunks never share buffers — the same guarantee per-chunk allocation gave, without
the garbage.
"""
function take_workspace!(::Type{T}, chip_size, search_radius) where {T<:Real}
    csx, csy = chip_size isa Tuple ? chip_size : (chip_size, chip_size)
    rx, ry = search_radius isa Tuple ? search_radius : (search_radius, search_radius)
    key = ((Int(csx), Int(csy)), (Int(rx), Int(ry)))
    ws = lock(WORKSPACE_LOCK) do
        pool = get(WORKSPACE_POOL, key, nothing)
        isnothing(pool) || isempty(pool) ? nothing : pop!(pool)
    end
    # Built outside the lock: planning and allocation are slow, and holding the lock across them
    # would serialise exactly the burst of concurrent chunks this exists to serve.
    return isnothing(ws) ? workspace(T, chip_size, search_radius) : ws
end

"""
    AutoRIFT.give_workspace!(ws)

Return a workspace to the pool. Its buffers are not cleared: every one is fully written before
being read on the next use, which is what `prepare_chip!` and the integral images guarantee.
"""
function give_workspace!(ws::CorrelationWorkspace)
    lock(WORKSPACE_LOCK) do
        push!(get!(() -> CorrelationWorkspace[], WORKSPACE_POOL,
                   (ws.max_chip, ws.max_radius)), ws)
    end
    return nothing
end

"""
    AutoRIFT.clear_workspaces!()

Drop every pooled workspace. For tests and benchmarks that measure allocation; not part of
normal operation.
"""
function clear_workspaces!()
    lock(WORKSPACE_LOCK) do
        empty!(WORKSPACE_POOL)
        empty!(REFINEMENT_POOL)
    end
    return nothing
end

# Declared here and filled in `peak.jl`, which is where `RefinementWorkspace` is defined. Split
# because each pool belongs with its type, and this file is included first.
const REFINEMENT_POOL = Dict{Int,Vector{Any}}()

"""
    next_fft_size(n) -> Int

A transform length `>= n` that FFTW handles efficiently.

Not simply the smallest product of small primes, which is the obvious rule and the wrong
one. FFTW's radix-2 codelets are far better optimised than its others, so a 2-heavy length
beats a smaller but 3-heavy one: at `n = 81`, the smallest smooth size *is* 81 (3⁴) and it
takes 14.1 µs, while 96 = 2⁵·3 takes 8.7 µs. Measured 1.1–1.5x available across the window
extents this package uses, always in favour of the more 2-heavy candidate.

So candidates are scored by a cost model rather than by length alone: `m² log₂ m` weighted
by how 2-heavy the factorisation is. Padding a little further costs arithmetic on zeros and
saves more on the transform, but only up to a point — a pure power of two is not
automatically best, because at `n = 81` it would mean padding to 128 where 96 is faster and
smaller. The search is capped at `1.5n` so the padding can never dominate.
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

# ---------------------------------------------------------------------------
# Chip preparation
# ---------------------------------------------------------------------------

"""
    prepare_chip!(ws, chip) -> (norm, ok)

Copy `chip` into the workspace with its mean removed, and return the Euclidean
norm of the result together with whether the chip carries any signal.

`ok` is `false` for a constant chip. Such a chip has zero variance, so the
correlation coefficient is undefined at every shift — the numerator and the chip's
contribution to the denominator both vanish. It is *not* a peak at the surface
origin, which is what the reference implementation reports: v2.0.0 removed the
guard that used to skip these points, so a masked or featureless area now yields a
displacement pinned to the corner of the search window. A constant chip carries no
information about displacement, so the honest answer is no measurement.
"""
function prepare_chip!(ws::CorrelationWorkspace, chip::AbstractMatrix)
    ch, cw = size(chip)
    (ch <= size(ws.chip, 1) && cw <= size(ws.chip, 2)) || throw(DimensionMismatch(
        "chip is $(ch)x$(cw) but the workspace was built for at most " *
        "$(size(ws.chip, 1))x$(size(ws.chip, 2))"))

    dst = @view ws.chip[1:ch, 1:cw]

    # Mean in Float64 regardless of input type: it is subtracted from every
    # element, so an error here biases the whole numerator rather than averaging
    # out.
    s = 0.0
    @inbounds for j in 1:cw, i in 1:ch
        s += Float64(chip[i, j])
    end
    mean = s / (ch * cw)

    sq = 0.0
    @inbounds for j in 1:cw, i in 1:ch
        d = Float64(chip[i, j]) - mean
        dst[i, j] = Float32(d)
        sq += d * d
    end

    # Zero within rounding, not exactly zero: a chip of nearly-identical values
    # gives a norm at the scale of the rounding error, and dividing by it would
    # amplify noise into a spurious peak. `eps` scaled by the element count is the
    # magnitude below which the sum of squares is indistinguishable from zero.
    ok = sq > eps(Float64) * ch * cw
    return sqrt(sq), ok
end

# ---------------------------------------------------------------------------
# The surface
# ---------------------------------------------------------------------------

"""
    correlate!(ws, search, chip, radius; measure = ZNCC()) -> surface

Correlate `chip` against `search`, returning a view of the workspace's surface
buffer holding the similarity at every candidate shift.

The returned surface is `(2 * radius_y, 2 * radius_x)`, with zero displacement at
its centre — see [`AutoRIFT.peak_offset`](@ref) for converting an index to a
displacement. `search` must be `(chip + 2 * radius - 1)` in each dimension, the
asymmetric extent that makes the surface an even size.

Returns a surface of zeros if the chip is constant, and sets
[`AutoRIFT.degenerate`](@ref) so the caller can tell that from a genuine zero
correlation; `AutoRIFT.prepare_chip!` explains why zeros are the right answer rather than
a peak at the origin.

Allocation-free. The result aliases the workspace, so it must be consumed or
copied before the next call on the same workspace.
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
    degenerate(ws::CorrelationWorkspace) -> Bool

Whether the last [`correlate!`](@ref) on `ws` was given a chip with no signal.

A constant chip has zero variance, so the correlation coefficient is undefined at every
shift and the surface is filled with zeros. Those zeros are not a measurement of zero
displacement, and this is how a caller tells the two apart without scanning the surface.
"""
@inline degenerate(ws::CorrelationWorkspace) = ws.was_degenerate[]

# Dispatch on the measure so each formula compiles to its own kernel with no
# runtime branch in the inner loop.
function _correlate_surface!(
    surface, ws::CorrelationWorkspace, search::AbstractMatrix,
    ch::Int, cw::Int, cnorm::Float64, ::ZNCC,
)
    nr, nc = size(surface)
    n = ch * cw
    cd = @view ws.chip[1:ch, 1:cw]

    S = @view ws.isum[1:(size(search, 1) + 1), 1:(size(search, 2) + 1)]
    S2 = @view ws.isqsum[1:(size(search, 1) + 1), 1:(size(search, 2) + 1)]
    integral!(S, search)
    integral_sq!(S2, search)

    numerators = _numerators!(ws, search, cd, nr, nc, ch, cw)

    @inbounds for j in 1:nc, i in 1:nr
        s1 = boxsum(S, i, j, ch, cw)
        s2 = boxsum(S2, i, j, ch, cw)
        # Σ(W - W̄)² = ΣW² - (ΣW)²/n. Clamped at zero: the two terms are large and
        # nearly equal for a low-contrast window, so rounding can make the
        # difference slightly negative even in Float64, and `sqrt` of that is NaN.
        wvar = max(s2 - s1 * s1 / n, 0.0)
        den = cnorm * sqrt(wvar)
        surface[i, j] = den > 0 ? Float32(numerators[i, j] / den) : 0.0f0
    end
    return surface
end

function _correlate_surface!(
    surface, ws::CorrelationWorkspace, search::AbstractMatrix,
    ch::Int, cw::Int, cnorm::Float64, ::NCC,
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
    integral_sq!(S2, search)

    numerators = _numerators!(ws, search, cd, nr, nc, ch, cw)

    @inbounds for j in 1:nc, i in 1:nr
        wsq = boxsum(S2, i, j, ch, cw)
        den = cnorm * sqrt(max(wsq, 0.0))
        surface[i, j] = den > 0 ? Float32(numerators[i, j] / den) : 0.0f0
    end
    return surface
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

Work below which the numerator is evaluated directly rather than by FFT, measured
as `nshifts * chip_area` multiply-accumulates.

Set from measurement, and the measurement was one-sided: FFT won at every
combination tried, from chip 16 with radius 6 (2x) out to chip 64 with radius 50
(53x). The direct loop is not the problem — it sustains ~13 GFLOP/s, close to what
this machine can do — it is simply asked to perform far more arithmetic. At chip 32
with radius 25 the direct path is 5.2 million multiply-accumulates against roughly
fifty thousand butterflies for the transform, and no constant factor closes a gap
like that.

So the threshold sits below any configuration real data uses, and the direct path
survives for three narrower reasons: very small chips where the transform's fixed
cost would dominate, verification (the two paths must agree, which is a test in
`test/correlate.jl`), and as a fallback needing no FFT plan.

Re-measure with `benchmark/suite/correlate.jl` if the numerator implementation
changes; the crossover is a property of the two implementations, not of the math.
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

function _numerators_direct!(out, search, cd, nr, nc, ch, cw)
    @inbounds for j in 1:nc
        for i in 1:nr
            acc = 0.0
            for jj in 1:cw
                # `search` is column-major, so the inner loop over rows walks
                # contiguous memory in both arrays.
                @simd for ii in 1:ch
                    acc += Float64(cd[ii, jj]) * Float64(search[i + ii - 1, j + jj - 1])
                end
            end
            out[i, j] = acc
        end
    end
    return out
end

function _numerators_fft!(out, ws::CorrelationWorkspace, search, cd, nr, nc, ch, cw)
    sh, sw = size(search)
    fy, fx = size(ws.fbuf_a)

    a, b = ws.fbuf_a, ws.fbuf_b
    fill!(a, 0.0f0)
    fill!(b, 0.0f0)
    @inbounds for j in 1:sw, i in 1:sh
        a[i, j] = Float32(search[i, j])
    end
    # Correlation is convolution with a reflected kernel, so the chip goes in
    # reversed. Placed at the origin, which puts the valid output at offset
    # (ch, cw) — see the read below.
    @inbounds for j in 1:cw, i in 1:ch
        b[i, j] = cd[ch - i + 1, cw - j + 1]
    end

    plan = fft_plan(fy, fx)
    mul!(ws.fspec_a, plan, a)
    mul!(ws.fspec_b, plan, b)
    @inbounds for k in eachindex(ws.fspec_a)
        ws.fspec_a[k] *= ws.fspec_b[k]
    end
    iplan = ifft_plan(fy, fx)
    mul!(a, iplan, ws.fspec_a)

    @inbounds for j in 1:nc, i in 1:nr
        out[i, j] = Float64(a[i + ch - 1, j + cw - 1])
    end
    return out
end
