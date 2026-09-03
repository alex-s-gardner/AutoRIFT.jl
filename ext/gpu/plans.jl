# Device buffers and transform plans for the batched correlator.
#
# Vendor-neutral: every array is allocated through `similar` on a backend-supplied prototype and
# every plan through the `AbstractFFTs` interface, so this file is included by each vendor
# extension rather than duplicated in it. Julia extensions cannot depend on one another, but they
# can include a common file, which is what makes one implementation serve several devices.
#
# ---------------------------------------------------------------------------
# Why the batch is a type parameter of nothing
# ---------------------------------------------------------------------------
#
# A pass has one chip size and one transform size — `_level_points` sets the first and the pass
# maxima set the second — so a batch is uniform by construction and needs no regrouping. Only the
# per-point search radius varies, and that changes how much of the zero-padded buffer is filled,
# not its extent. So a workspace is keyed on `(chip, radius, batch)` and holds plain arrays with
# runtime extents, exactly as `CorrelationWorkspace` does and for the same reason: a size in the
# type would recompile every kernel per geometry, and a run uses several.
#
# ---------------------------------------------------------------------------
# Float32 throughout, and why that is not a downgrade
# ---------------------------------------------------------------------------
#
# `src/integral.jl` requires `Float64` summed-area tables because the variance `ΣW² − (ΣW)²/n`
# differences two large nearly-equal quantities. Apple GPUs have no `Float64`, so the cancellation
# is removed instead of absorbed: the window's own mean is subtracted during the gather, which is
# exact for both the numerator (`Σ T′·(W − c) = Σ T′·W` because `Σ T′ = 0`) and the variance (shift
# invariance). The tables then hold O(contrast) values, where `Float32` does not cancel.
#
# Measured, over chips 16-128 and DC 1 to 1e4: raw `Float32` tables err by **8.4** — catastrophic,
# as that file predicts — centered ones by 3.6e-6, and centered with a double-single accumulator by
# 3.9e-8, which matches `Float64`. Centering the window also makes the *numerator* more accurate
# than the CPU's, since a raw window puts a large DC term into the transform whose magnitude sets
# the rounding at every frequency. See `docs/gpu-feasibility.md`.

# Sum tables carried as an unevaluated pair of `Float32`, which is what buys back the mantissa
# `Float64` would have given. `hi + lo` is the value; `lo` holds the bits `hi` dropped.
#
# Verified exact on the Metal device, and that check is a precondition rather than a nicety: the
# `twosum` identity holds only under round-to-nearest with no reassociation, and a compiler that
# reassociated would zero `lo` silently, leaving a kernel that runs and buys nothing. Measured
# bit-identical to the host with 12365 of 12800 error terms nonzero.
struct DevicePairTable{A}
    hi::A
    lo::A
end

"""
    GPUWorkspace

Device buffers for correlating a batch of points at one geometry.

Every array's trailing dimension is the batch, so a kernel indexes its point with the last index
and the transform plans see a `(fy, fx, B)` array they reduce over region `(1, 2)`. Allocated once
per geometry and pooled, for the reason `AutoRIFT.take_workspace!` documents: a pass acquires and
returns rather than allocating, so the device allocator is not on the hot path.
"""
struct GPUWorkspace{T,M<:AbstractArray{T,3},C<:AbstractArray{Complex{T},3},
                    V<:AbstractVector{T},I<:AbstractVector{Int32},B<:AbstractVector{Bool}}
    # Gathered chips, mean-removed, and search windows, mean-removed. Both `Float32` regardless of
    # the image's element type, as `CorrelationWorkspace.chip` is and for the same reason: the chip
    # is stored mean-removed, so it is fractional even for `UInt8` input.
    chip::M
    window::M

    # Zero-padded transform inputs and the spectra. `fbuf_b` receives the reversed chip; the
    # spectra are half-height, since a real transform has conjugate symmetry to exploit.
    fbuf_a::M
    fbuf_b::M
    fspec_a::C
    fspec_b::C

    # Summed-area tables of the centered window, sum and sum-of-squares, as `Float32` pairs.
    isum::DevicePairTable{M}
    isqsum::DevicePairTable{M}

    # Column-wise partial sums, the first stage of the two-pass prefix sum. Separate from the
    # tables because the row pass reads them while writing the tables, and overlapping the two
    # would corrupt a result that looks plausible on a small input.
    colsum::DevicePairTable{M}
    colsqsum::DevicePairTable{M}

    # The correlation surface, one per point.
    surface::M

    # Per-point scalars. The chip norm and the degenerate flag come out of the gather; the peak
    # index comes out of the peak scan and is consumed by the refinement.
    cnorm::V
    peak_i::I
    peak_j::I
    degenerate::B

    # Per-point outputs, read back to the host as five short vectors rather than as surfaces.
    dx::V
    dy::V
    corr::V
    ppr::V
    searched::B

    # Extents this workspace was built for, so a mismatched call errors rather than reading out of
    # bounds — the same guard `CorrelationWorkspace` carries.
    chip_size::Tuple{Int,Int}
    radius::Tuple{Int,Int}
    fftsize::Tuple{Int,Int}
    batch::Int
end

# A `(m, n, B)` device array of the backend's own type, from a prototype.
_dev(proto, ::Type{T}, dims...) where {T} = similar(proto, T, dims...)

_devpair(proto, ::Type{T}, dims...) where {T} =
    DevicePairTable(_dev(proto, T, dims...), _dev(proto, T, dims...))

"""
    gpu_workspace(proto, chip_size, radius, batch) -> GPUWorkspace

Allocate device buffers for `batch` points at one chip size and search radius.

`proto` is any array on the target device; every buffer is `similar` to it, which is what keeps
this file free of a vendor array type.
"""
function gpu_workspace(proto, chip_size::Tuple{Int,Int}, radius::Tuple{Int,Int}, batch::Int)
    csx, csy = chip_size
    rx, ry = radius
    (csx > 0 && csy > 0) || throw(ArgumentError(
        "chip size must be positive, got ($csx, $csy)"))
    (rx > 0 && ry > 0) || throw(ArgumentError(
        "search radius must be positive, got ($rx, $ry)"))
    batch > 0 || throw(ArgumentError("batch must be positive, got $batch"))

    # The asymmetric window convention, as `workspace` derives it: the search reaches `radius` one
    # way and `radius - 1` the other, so the surface is an even `2 * radius`.
    winx = csx + 2rx - 1
    winy = csy + 2ry - 1
    fy = gpu_fft_size(winy)
    fx = gpu_fft_size(winx)
    T = Float32

    return GPUWorkspace(
        _dev(proto, T, csy, csx, batch),
        _dev(proto, T, winy, winx, batch),
        _dev(proto, T, fy, fx, batch),
        _dev(proto, T, fy, fx, batch),
        _dev(proto, Complex{T}, fy ÷ 2 + 1, fx, batch),
        _dev(proto, Complex{T}, fy ÷ 2 + 1, fx, batch),
        _devpair(proto, T, winy + 1, winx + 1, batch),
        _devpair(proto, T, winy + 1, winx + 1, batch),
        _devpair(proto, T, winy, winx, batch),
        _devpair(proto, T, winy, winx, batch),
        _dev(proto, T, 2ry, 2rx, batch),
        _dev(proto, T, batch),
        _dev(proto, Int32, batch),
        _dev(proto, Int32, batch),
        _dev(proto, Bool, batch),
        _dev(proto, T, batch),
        _dev(proto, T, batch),
        _dev(proto, T, batch),
        _dev(proto, T, batch),
        _dev(proto, Bool, batch),
        (csx, csy),
        (rx, ry),
        (fy, fx),
        batch,
    )
end

"""
    gpu_fft_size(n) -> Int

A transform length `>= n` that the device's FFT handles efficiently.

`AutoRIFT.next_fft_size`'s rule, reused rather than replaced: it was calibrated against FFTW's
radix-2 codelets, and measurement on MPSGraph shows its choices are also good there — powers of two
are modestly faster per point (64 at 0.83 ms against 56 at 0.91 ms for a 1024-batch) but not enough
to justify transforming a larger array.

What it protects against is sharper than a few percent. A prime-heavy length is *catastrophic* on
MPSGraph rather than merely slow: 177 measures 27.7 ms against 192's 3.53 ms, an **8x** penalty
where FFTW's is a fraction. `next_fft_size` already returns 192 there, since it admits only
products of 2, 3, 5 and 7 — so the rule that made FFTW fast is what keeps the device off a cliff.
Any future change to the padding rule has to preserve that, which is why this is a named function
with this comment rather than a call at the one site that needs it.
"""
gpu_fft_size(n::Integer) = AutoRIFT.next_fft_size(n)

# ---------------------------------------------------------------------------
# Plan cache
# ---------------------------------------------------------------------------
#
# Keyed on `(fy, fx, batch)`, since an `AbstractFFTs` plan encodes the full array shape and a plan
# executed against a different one throws. Held per backend type, because two devices' plans are
# not interchangeable and a run may in principle touch both.
#
# `mul!` with preallocated output rather than `p * x`, which allocates a fresh device array per
# call — once per pass per direction is not the hot path, but a pass over a large grid runs many
# batches, and the allocator is a global lock on most backends.
#
# Metal.jl caches the underlying MPSGraph per shape internally as well, so this cache is about
# avoiding the plan *object* and its validation, not about the graph.
const GPU_PLAN_LOCK = ReentrantLock()
const GPU_RFFT_PLANS = Dict{Tuple{DataType,Int,Int,Int},Any}()
const GPU_IRFFT_PLANS = Dict{Tuple{DataType,Int,Int,Int},Any}()

"""
    gpu_rfft_plan(ws) -> plan

The forward real-to-complex plan for `ws`'s transform shape, from the cache.

Reduces over region `(1, 2)` of an `(fy, fx, batch)` array, so one call transforms every point of
the batch. Measured 8-17x FFTW's throughput for the same total work at batch 1024 and above, and
*slower* below batch 256 — which is why a sparse pass stays on the CPU rather than being batched
onto the device.
"""
function gpu_rfft_plan(ws::GPUWorkspace)
    key = (typeof(ws.fbuf_a), ws.fftsize[1], ws.fftsize[2], ws.batch)
    p = get(GPU_RFFT_PLANS, key, nothing)
    isnothing(p) || return p
    return lock(GPU_PLAN_LOCK) do
        get!(() -> plan_rfft(ws.fbuf_a, (1, 2)), GPU_RFFT_PLANS, key)
    end
end

"""
    gpu_irfft_plan(ws) -> plan

The inverse plan matching [`gpu_rfft_plan`](@ref).

**Unnormalised**, like FFTW's c2r transform: this is `plan_brfft`, so the `1/(fy*fx)` is applied
when the numerator is read out. Scaling on the read rather than in a pass over the whole buffer is
what `_cnumerators_fft!` does on the CPU, and for the same reason — only `2*radius` squared of
`fy*fx` values are ever used.
"""
function gpu_irfft_plan(ws::GPUWorkspace)
    key = (typeof(ws.fspec_a), ws.fftsize[1], ws.fftsize[2], ws.batch)
    p = get(GPU_IRFFT_PLANS, key, nothing)
    isnothing(p) || return p
    return lock(GPU_PLAN_LOCK) do
        get!(() -> plan_brfft(ws.fspec_a, ws.fftsize[1], (1, 2)), GPU_IRFFT_PLANS, key)
    end
end

# ---------------------------------------------------------------------------
# Workspace pool
# ---------------------------------------------------------------------------
#
# The take/give discipline of `AutoRIFT.take_workspace!`, for the same reason and with one added:
# device allocation is slower than host allocation and serialises on a driver lock, so a pass that
# allocated per batch would spend its time there. Keyed on exact geometry *and* backend type, since
# a buffer belongs to the device it was allocated on.

const GPU_WORKSPACE_POOL = Dict{Tuple{DataType,Tuple{Int,Int},Tuple{Int,Int},Int},Vector{Any}}()

"""
    take_gpu_workspace!(proto, chip_size, radius, batch) -> GPUWorkspace

A device workspace for exactly this geometry, from the pool or newly allocated. Return it with
[`give_gpu_workspace!`](@ref).
"""
function take_gpu_workspace!(proto, chip_size::Tuple{Int,Int}, radius::Tuple{Int,Int},
                             batch::Int)
    key = (typeof(proto), chip_size, radius, batch)
    ws = lock(GPU_PLAN_LOCK) do
        pool = get(GPU_WORKSPACE_POOL, key, nothing)
        isnothing(pool) || isempty(pool) ? nothing : pop!(pool)
    end
    # Allocated outside the lock, as the host pool does: device allocation is the slow part, and
    # holding the lock across it would serialise the concurrency this exists to serve.
    return isnothing(ws) ? gpu_workspace(proto, chip_size, radius, batch) : ws
end

"""
    give_gpu_workspace!(ws)

Return a device workspace to the pool. Buffers are not cleared: the gather fully writes every one
before it is read, and the transform buffers are zeroed by the gather kernel because only part of
them is written.
"""
function give_gpu_workspace!(ws::GPUWorkspace)
    key = (typeof(ws.fbuf_a), ws.chip_size, ws.radius, ws.batch)
    lock(GPU_PLAN_LOCK) do
        push!(get!(() -> [], GPU_WORKSPACE_POOL, key), ws)
    end
    return nothing
end

"""
    clear_gpu_workspaces!()

Drop every pooled device workspace and cached plan. For tests and benchmarks that measure
allocation, and to release device memory; not part of normal operation.
"""
function clear_gpu_workspaces!()
    lock(GPU_PLAN_LOCK) do
        empty!(GPU_WORKSPACE_POOL)
        empty!(GPU_RFFT_PLANS)
        empty!(GPU_IRFFT_PLANS)
    end
    return nothing
end

"""
    gpu_batch_size(chip_size, radius, budget_bytes) -> Int

How many points to correlate at once, from a device-memory budget.

Derived rather than fixed, because the per-point footprint grows with the transform size: about
215 kB at chip 32 and radius 25, four times that at chip 128. **The result must not depend on
this** — a pass split into two batches has to agree bitwise with one batch, which is what makes
the budget free to change and is asserted in `test/gpu.jl`.

Floored at a size where the device still wins. Batched transforms measure *slower* than FFTW below
batch 256 at small transform sizes, since per-launch overhead dominates; a pass with fewer
searchable points than [`GPU_MIN_BATCH`](@ref) is better run on the CPU, which is what
`_gpu_worth_it` decides.
"""
function gpu_batch_size(chip_size::Tuple{Int,Int}, radius::Tuple{Int,Int},
                        budget_bytes::Int = GPU_MEMORY_BUDGET[])
    csx, csy = chip_size
    rx, ry = radius
    winx = csx + 2rx - 1
    winy = csy + 2ry - 1
    fy = gpu_fft_size(winy)
    fx = gpu_fft_size(winx)

    # Per point, in `Float32` elements: chip, window, two padded buffers, two half-spectra (two
    # elements each), four tables and four column buffers as pairs, one surface.
    per = csy * csx + winy * winx + 2 * fy * fx + 2 * 2 * (fy ÷ 2 + 1) * fx +
          4 * (winy + 1) * (winx + 1) + 4 * winy * winx + 4 * ry * rx
    bytes = 4 * per
    return max(1, min(GPU_MAX_BATCH, budget_bytes ÷ max(bytes, 1)))
end

"""
    GPU_MEMORY_BUDGET

Device bytes a single batch of correlation buffers may occupy, which is what
[`gpu_batch_size`](@ref) divides by the per-point footprint.

512 MiB: large enough that chip 32 batches thousands of points and so clears the size at which the
batched transform beats FFTW by an order of magnitude, and small enough to leave the imagery and
the caller's own arrays resident on a shared-memory device, where the GPU has no separate pool to
draw on.

A `Ref` so it can be lowered to force a small batch. **The result must not depend on it** — a pass
split into several batches is bit-identical to one batch — and driving the batch count from the one
knob that controls it is how `test/gpu.jl` asserts that. Lower it and call
[`clear_gpu_workspaces!`](@ref), since a pooled workspace is keyed on its batch size.
"""
const GPU_MEMORY_BUDGET = Ref(512 * 1024 * 1024)

"""
    REFINE_MEMORY_BUDGET

Device bytes the sub-pixel cascade's scratch may occupy, which sets how many points it refines per
tile.

Separate from [`GPU_MEMORY_BUDGET`](@ref) and larger, because the two buy different things. The
correlation's budget bounds a footprint that scales with the transform; this one buys **launch
amortization**: the cascade is 12 kernel launches per tile whatever the tile holds, so a small tile
pays them per handful of points. Measured at 64x upsampling, the same cascade costs 17.6 us per point
at a 131-point tile against 13.5 at 2048, and a tile of 131 made the refinement 164 us per point of a
195 us pass.

1 GiB, which at 64x holds ~1000 points (820 kB each) and at 16x ~16,000. A `Ref` for the same reason
the correlation budget is one — the result must not depend on it.
"""
const REFINE_MEMORY_BUDGET = Ref(1024 * 1024 * 1024)

"""
    GPU_MAX_BATCH

Points per batch, whatever the memory budget allows.

A cap on the transform's trailing dimension rather than on memory. Throughput is already saturated
well before this — 4096 measures within 20% of 1024 per point — so a larger batch buys nothing and
costs a longer tail when the final batch of a pass is partly empty.
"""
const GPU_MAX_BATCH = 4096

"""
    GPU_MIN_BATCH

Searchable points below which a pass runs on the CPU instead of the device.

Measured rather than chosen: a batched transform is *slower* than the equivalent FFTW calls below
batch 256 at the smaller transform sizes — 0.16x at 64 points and a 28-point transform — because
per-launch overhead dominates. A coarse pass or a late chip-size level can easily fall here, since
the search deliberately zeroes most of the grid, so this is a routine case rather than a corner.
"""
const GPU_MIN_BATCH = 256
