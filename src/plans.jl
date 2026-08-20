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
#     on are per-task (each task has its own `CorrelationWorkspace`); `mul!` with
#     explicit input and output is safe under that arrangement.

using FFTW: FFTW, plan_rfft, plan_irfft
using LinearAlgebra: mul!
using Scratch: @get_scratch!

const PLAN_LOCK = ReentrantLock()
const RFFT_PLANS = Dict{Tuple{Int,Int},Any}()
const IRFFT_PLANS = Dict{Tuple{Int,Int},Any}()

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

# Set once the wisdom file has been read, so a process imports at most once.
const WISDOM_LOADED = Ref(false)
# Sizes planned since the last export, so an export only happens when there is something new.
const WISDOM_DIRTY = Ref(false)

"""
    AutoRIFT.wisdom_path() -> String or nothing

Path to this machine's FFTW wisdom file, or `nothing` if no scratch space is available.

Keyed by CPU model and FFTW version: wisdom is portable across neither, and importing another
machine's would produce plans tuned for the wrong cache hierarchy.
"""
function wisdom_path()
    try
        # `Sys.cpu_info()` can report a model string with spaces and slashes ("Apple M2 Max",
        # "Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz"), none of which belong in a filename.
        cpu = replace(Sys.cpu_info()[1].model, r"[^A-Za-z0-9._-]+" => "_")
        dir = @get_scratch!("fftw_wisdom")
        return joinpath(dir, "$(cpu)-fftw$(FFTW_VERSION).wisdom")
    catch
        # No writable scratch space. Planning still works; it is just never cached.
        return nothing
    end
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
        isfile(path) && FFTW.import_wisdom(path)
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
        FFTW.export_wisdom(tmp)
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

Thread-safe. The common case is a cache hit, which takes no lock — an unsynchronised
read of a `Dict` that is only ever grown under a lock is safe here because a miss
falls through to the locked path and re-checks.
"""
function fft_plan(ny::Int, nx::Int)
    key = (ny, nx)
    p = get(RFFT_PLANS, key, nothing)
    p === nothing || return p
    return lock(PLAN_LOCK) do
        get!(RFFT_PLANS, key) do
            # MEASURE rather than ESTIMATE: planning is paid once per size per
            # process and reused across every grid point, so the extra planning
            # time is recovered immediately. Wisdom persistence (see `__init__`)
            # removes even that cost on subsequent runs.
            WISDOM_DIRTY[] = true
            plan_rfft(Matrix{Float32}(undef, ny, nx); flags = FFTW.MEASURE)
        end
    end
end

"""
    ifft_plan(ny, nx)

Complex-to-real inverse FFT plan matching [`fft_plan`](@ref), from the cache.
"""
function ifft_plan(ny::Int, nx::Int)
    key = (ny, nx)
    p = get(IRFFT_PLANS, key, nothing)
    p === nothing || return p
    return lock(PLAN_LOCK) do
        get!(IRFFT_PLANS, key) do
            WISDOM_DIRTY[] = true
            plan_irfft(Matrix{ComplexF32}(undef, ny ÷ 2 + 1, nx), ny;
                       flags = FFTW.MEASURE)
        end
    end
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
