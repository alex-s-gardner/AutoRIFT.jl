# FFT plan cache.
#
# Plans are expensive to create and cheap to reuse, and the set of sizes a run
# needs is tiny: chip size is constant within a chip-size pass and the search radius
# takes only a few distinct values, so a whole scene needs on the order of ten
# plans. Caching them is the difference between planning once and planning per grid
# point.
#
# Two constraints from FFTW, both of which shape the design:
#
#   * The planner is not thread-safe. Concurrent planning must be serialised, and
#     the fast path must avoid the lock entirely or every task will queue behind
#     the first one.
#
#   * A plan is not safe to *execute* concurrently on the same plan object with
#     different buffers. Plans are therefore per-size, and the buffers they operate
#     on are per-task (each task has its own `CorrelationWorkspace`); executing a plan with
#     explicit input and output buffers is safe under that arrangement.

using FFTW: FFTW
using FFTW_jll: libfftw3f_path
using Scratch: scratch_dir

# ---------------------------------------------------------------------------
# Why raw C plan pointers rather than FFTW.jl's plan objects
# ---------------------------------------------------------------------------
#
# An `FFTW.rFFTWPlan` is a mutable wrapper that registers a **finalizer** in its inner
# constructor, so it can destroy the underlying C plan when collected. That is right for a plan
# with a bounded lifetime. Ours have none: they live in a `const Dict` for the whole process and
# are never evicted, because the set of sizes a run needs is tiny and every grid point reuses
# them. A destructor we never want is machinery we should not carry.
#
# It also has two concrete costs. `mul!(dst, plan, src)` through an `Any`-typed cache is a dynamic
# call in the hottest path in the package. And the finalizer makes the code untrimmable: Julia's
# `--trim` cannot prove which finalizers run, and FFTW's is reached via `foreach` over a
# `Vector{FFTWPlan}` with `@nospecialize`, so it is unresolvable by construction. Verified against
# a bare ten-line FFTW program: one plan, no cache, still four trim errors, all in plan
# destruction. Holding `Ptr{Cvoid}` avoids needing any of it — measured 37 MiB peak RSS for a
# trimmed binary against 489 MiB for the equivalent Julia process.
#
# The library is named by `FFTW_jll.libfftw3f_path` and not by either obvious alternative, both of
# which were tried and fail in opposite directions. `FFTW.libfftw3f` is a `FakeLazyLibrary`
# resolved by a load-time callback; a trimmed binary has no such callback and the `ccall` dies at
# *runtime* with a `TypeError`. A bare soname (`"libfftw3f.dylib"`) works in a trimmed binary with
# the artifact on the loader path, but fails in an ordinary session, where it is not. The resolved
# artifact path works in both.

const LIBFFTW3F = libfftw3f_path

# FFTW's own flag values, from `fftw3.h`. Hard-coded because the whole point is not to depend on
# FFTW.jl's plan machinery; `FFTW.MEASURE` is the same integer.
const FFTW_MEASURE = UInt32(0)
const FFTW_ESTIMATE = UInt32(1 << 6)

const PLAN_LOCK = ReentrantLock()
# `Ptr{Cvoid}`, so a cache hit yields a concrete type and the `ccall` below is a static call.
const RFFT_PLANS = Dict{Tuple{Int,Int},Ptr{Cvoid}}()
const IRFFT_PLANS = Dict{Tuple{Int,Int},Ptr{Cvoid}}()

# ---------------------------------------------------------------------------
# Wisdom persistence
# ---------------------------------------------------------------------------
#
# `MEASURE` planning is expensive: 116-347 ms per size on this machine, and a default run
# needs three sizes — 822 ms measured for the set, against 158 ms for the correlation itself.
# Within one process that is recovered immediately, since every grid point reuses the plan.
# Across processes it is not recovered at all, and a production driver that launches a process
# per image pair would pay it every time.
#
# FFTW's own answer is wisdom: the planner's measurements, serialised. Importing it turns the
# same three plans from 822 ms into 0.1 ms — measured, an 8000x reduction from a 4.1 KiB file.
#
# Keyed by CPU model *and* FFTW version, because wisdom is neither portable nor stable across
# either. A plan measured on one microarchitecture is not merely suboptimal elsewhere, it
# encodes cache and SIMD-width assumptions that do not hold; and FFTW's serialisation format
# is its own business. Reading another machine's file would be worse than planning fresh.
#
# Every filesystem touch is guarded. A read-only depot, a full disk, or a sandbox with no
# writable scratch directory must degrade to today's behaviour — plan every time — rather than
# fail a correlation. That is the difference between an optimization and a dependency.

# FFTW does not expose its own version as a constant, so take it from the package.
const FFTW_VERSION = string(pkgversion(FFTW))

# `scratch_dir` rather than `Scratch.@get_scratch!`, and the difference is the reason this UUID is
# written out. The macro additionally *records* the access in the depot's usage log, so `Pkg.gc()`
# knows the space is live and does not reclaim it. That bookkeeping stamps a `DateTime`, and
# formatting one reaches `rpad(::String, ::Int, ::Char)` → `Base.repeat` → `textwidth`, which
# `--trim` cannot resolve — four of the errors in a trimmed build, none of them about FFTs.
#
# What is given up is only garbage collection of the directory: `Pkg.gc()` may delete a wisdom file
# that has not been used in a while, and the next run measures its plans again and rewrites it. That
# is precisely the degradation `load_wisdom!` already tolerates for a read-only depot. Deleting a
# cache is a cost of milliseconds; not trimming costs 450 MiB of resident memory.
const PKG_UUID = Base.UUID("52cb0ed0-aa80-430c-bd04-c52888a79add")

# Set once the wisdom file has been read, so a process imports at most once.
const WISDOM_LOADED = Ref(false)
# The resolved wisdom path, and whether resolution has happened. Two refs rather than one because
# `nothing` — no writable scratch space — is a real answer worth caching, not a "not yet" sentinel.
const WISDOM_PATH = Ref{Union{String,Nothing}}(nothing)
const WISDOM_PATH_RESOLVED = Ref(false)
# Sizes planned since the last export, so an export only happens when there is something new.
const WISDOM_DIRTY = Ref(false)

"""
    AutoRIFT.wisdom_path() -> String or nothing

Path to this machine's FFTW wisdom file, or `nothing` if no scratch space is available.

Keyed by CPU model and FFTW version: wisdom is portable across neither, and importing another
machine's would produce plans tuned for the wrong cache hierarchy.
"""
function wisdom_path()
    # Resolved once per process. The path is derived from the CPU model, the FFTW version and the
    # depot — none of which change while a process runs — so recomputing it is pure waste, and it
    # is not free: `Sys.cpu_info()` allocates a vector of per-core structs and `mkpath` stats the
    # directory, together 9.6 us and 4.1 KiB per call.
    #
    # Caching also closes a hole that only opens when the scratch directory is *unwritable*. There
    # `save_wisdom!` returns before clearing `WISDOM_DIRTY`, so the flag stays set and every later
    # `warm_plans!` — once per chip-size level per pair, forever — re-pays the lookup and its
    # failure. Measured at 116 us and 6.6 KiB per call, or 202 extra allocations per image pair.
    # `nothing` is a legitimate cached answer, hence the two-state `Ref` rather than a sentinel.
    WISDOM_PATH_RESOLVED[] && return WISDOM_PATH[]
    WISDOM_PATH_RESOLVED[] = true
    WISDOM_PATH[] = try
        # `Sys.cpu_info()` can report a model string with spaces and slashes ("Apple M2 Max",
        # "Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz"), none of which belong in a filename.
        cpu = replace(Sys.cpu_info()[1].model, r"[^A-Za-z0-9._-]+" => "_")
        dir = scratch_dir(string(PKG_UUID), "fftw_wisdom")
        mkpath(dir)
        joinpath(dir, "$(cpu)-fftw$(FFTW_VERSION).wisdom")
    catch
        # No writable scratch space. Planning still works; it is just never cached.
        nothing
    end
    return WISDOM_PATH[]
end

"""
    AutoRIFT.reset_wisdom_path!()

Forget the cached wisdom path, so the next [`wisdom_path`](@ref) resolves it afresh.

Called from `__init__`, because a path resolved while *precompiling* would otherwise be serialised
into the image and inherited by whatever machine loads it. Also used by tests that move the scratch
directory.
"""
function reset_wisdom_path!()
    WISDOM_PATH_RESOLVED[] = false
    WISDOM_PATH[] = nothing
    return nothing
end

"""
    AutoRIFT.load_wisdom!()

Import this machine's saved FFTW wisdom, if any. Idempotent and never throws.

Called from `__init__`. A missing, unreadable, or corrupt file is not an error — it means the
next plan is measured from scratch, which is what would happen without any of this.
"""
function load_wisdom!()
    WISDOM_LOADED[] && return nothing
    WISDOM_LOADED[] = true
    path = wisdom_path()
    isnothing(path) && return nothing
    try
        # Raw `ccall` rather than `FFTW.import_wisdom`, for the same reason the plans are raw
        # pointers: FFTW.jl's version is wrapped in `@exclusive`, which reaches its plan-lock
        # machinery, and its export writes a 256-space separator with `" "^256` — which is the
        # `Base.repeat` call `--trim` cannot resolve. Only the single-precision wisdom is touched,
        # since every transform here is `Float32`.
        if isfile(path)
            f = ccall(:fopen, Ptr{Cvoid}, (Cstring, Cstring), path, "r")
            f == C_NULL && return nothing
            try
                ccall((:fftwf_import_wisdom_from_file, LIBFFTW3F), Cint, (Ptr{Cvoid},), f)
            finally
                ccall(:fclose, Cint, (Ptr{Cvoid},), f)
            end
        end
    catch
        # A corrupt or partially-written file. Ignore it; the next export overwrites it.
    end
    return nothing
end

"""
    AutoRIFT.save_wisdom!()

Write this machine's FFTW wisdom, if any new size has been planned. Never throws.

Called after [`warm_plans!`](@ref) rather than at exit: an `atexit` hook would miss a process
killed by a scheduler, which for batch work is the normal way for a process to end.
"""
function save_wisdom!()
    WISDOM_DIRTY[] || return nothing
    path = wisdom_path()
    isnothing(path) && return nothing
    try
        # Write to a unique temporary and rename, so two processes exporting at once cannot
        # leave a half-written file that every later process then fails to read. `mv` within one
        # directory is atomic on every filesystem this will run on.
        tmp = path * "." * string(getpid()) * ".tmp"
        f = ccall(:fopen, Ptr{Cvoid}, (Cstring, Cstring), tmp, "w")
        f == C_NULL && return nothing
        try
            ccall((:fftwf_export_wisdom_to_file, LIBFFTW3F), Cvoid, (Ptr{Cvoid},), f)
        finally
            ccall(:fclose, Cint, (Ptr{Cvoid},), f)
        end
        mv(tmp, path; force = true)
        WISDOM_DIRTY[] = false
    catch
        # Read-only depot, full disk, or a race with another process. Planning is unaffected.
    end
    return nothing
end

"""
    fft_plan(ny, nx)

Real-to-complex FFT plan for an `ny`-by-`nx` `Float32` array, from the cache.

Returns FFTW's own `Ptr{Cvoid}` plan handle — see the note at the top of this file for why it is
not an `FFTW.rFFTWPlan`. Execute it with [`fft_execute!`](@ref).

Thread-safe. The common case is a cache hit, which takes no lock — an unsynchronised
read of a `Dict` that is only ever grown under a lock is safe here because a miss
falls through to the locked path and re-checks.
"""
function fft_plan(ny::Int, nx::Int)
    key = (ny, nx)
    p = get(RFFT_PLANS, key, C_NULL)
    p === C_NULL || return p
    return lock(PLAN_LOCK) do
        get!(RFFT_PLANS, key) do
            # MEASURE rather than ESTIMATE: planning is paid once per size per
            # process and reused across every grid point, so the extra planning
            # time is recovered immediately. Wisdom persistence (see `__init__`)
            # removes even that cost on subsequent runs.
            WISDOM_DIRTY[] = true
            # FFTW is row-major and Julia column-major, so an `ny`-by-`nx` Julia array is an
            # `nx`-by-`ny` C array — the dimensions go in reversed. Getting this backwards
            # transposes every transform, which for a symmetric test size looks like it works.
            inb = Matrix{Float32}(undef, ny, nx)
            outb = Matrix{ComplexF32}(undef, ny ÷ 2 + 1, nx)
            plan = ccall((:fftwf_plan_dft_r2c_2d, LIBFFTW3F), Ptr{Cvoid},
                         (Cint, Cint, Ptr{Float32}, Ptr{ComplexF32}, Cuint),
                         nx, ny, inb, outb, FFTW_MEASURE)
            plan == C_NULL && error("FFTW could not plan a $(ny)x$(nx) real-to-complex " *
                                    "transform. This is not a recoverable condition: the " *
                                    "correlator has no fallback for a size FFTW rejects.")
            plan
        end
    end
end

"""
    fft_execute!(plan, input, output)

Run a real-to-complex plan from [`fft_plan`](@ref).

The buffers must be the sizes the plan was created for. FFTW does not check, and a mismatch is a
buffer overrun rather than an error — which is why `CorrelationWorkspace` sizes its FFT buffers
from the same `next_fft_size` the plan is keyed on.
"""
@inline function fft_execute!(plan::Ptr{Cvoid}, input::Matrix{Float32},
                              output::Matrix{ComplexF32})
    ccall((:fftwf_execute_dft_r2c, LIBFFTW3F), Cvoid,
          (Ptr{Cvoid}, Ptr{Float32}, Ptr{ComplexF32}), plan, input, output)
    return output
end

"""
    ifft_plan(ny, nx)

Complex-to-real inverse FFT plan matching [`fft_plan`](@ref), from the cache.

Execute it with [`ifft_execute!`](@ref), which applies the `1/n` scaling FFTW omits.
"""
function ifft_plan(ny::Int, nx::Int)
    key = (ny, nx)
    p = get(IRFFT_PLANS, key, C_NULL)
    p === C_NULL || return p
    return lock(PLAN_LOCK) do
        get!(IRFFT_PLANS, key) do
            WISDOM_DIRTY[] = true
            inb = Matrix{ComplexF32}(undef, ny ÷ 2 + 1, nx)
            outb = Matrix{Float32}(undef, ny, nx)
            plan = ccall((:fftwf_plan_dft_c2r_2d, LIBFFTW3F), Ptr{Cvoid},
                         (Cint, Cint, Ptr{ComplexF32}, Ptr{Float32}, Cuint),
                         nx, ny, inb, outb, FFTW_MEASURE)
            plan == C_NULL && error("FFTW could not plan a $(ny)x$(nx) complex-to-real " *
                                    "transform.")
            plan
        end
    end
end

"""
    ifft_execute!(plan, input, output)

Run a complex-to-real plan from [`ifft_plan`](@ref), scaled.

**FFTW's inverse transform is unnormalised** — it returns `n` times the inverse DFT, where
`AbstractFFTs.plan_irfft` wrapped the same plan in a `ScaledPlan` that divided for us. The scaling
is applied here so callers see the same values as before, and so the omission cannot be
rediscovered per call site: a missing `1/n` scales every correlation numerator by the transform
size, which produces a plausible-looking surface with the wrong normalisation.

!!! warning "The input is destroyed"
    FFTW's c2r transform overwrites its input buffer unless planned with `FFTW_PRESERVE_INPUT`,
    which costs performance. Callers must not read the spectrum afterwards.
"""
@inline function ifft_execute!(plan::Ptr{Cvoid}, input::Matrix{ComplexF32},
                               output::Matrix{Float32})
    ccall((:fftwf_execute_dft_c2r, LIBFFTW3F), Cvoid,
          (Ptr{Cvoid}, Ptr{ComplexF32}, Ptr{Float32}), plan, input, output)
    scale = 1.0f0 / length(output)
    @inbounds @simd for i in eachindex(output)
        output[i] *= scale
    end
    return output
end

"""
    warm_plans!(sizes)

Create the plans for `sizes` on the calling task, before any parallel work starts.

Without this, every task racing to correlate its first point would contend on the
planner lock and serialise — turning the most parallel part of the run into its
most serial. Called once per pass, where the set of sizes is known in advance.
"""
function warm_plans!(sizes)
    for (ny, nx) in sizes
        fft_plan(ny, nx)
        ifft_plan(ny, nx)
    end
    # Persist whatever was measured, so the next process starts warm. Only writes if a plan was
    # actually created — the common case after the first run is that this does nothing.
    save_wisdom!()
    return nothing
end

"""
    clear_plans!()

Drop all cached plans. For tests and benchmarks that need to measure planning cost;
not part of normal operation.
"""
function clear_plans!()
    lock(PLAN_LOCK) do
        empty!(RFFT_PLANS)
        empty!(IRFFT_PLANS)
    end
    return nothing
end
