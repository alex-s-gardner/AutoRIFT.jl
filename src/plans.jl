# FFT plan cache.
#
# Plans are expensive to create and cheap to reuse, and the set of sizes a run
# needs is tiny: chip size is constant within a pyramid pass and the search radius
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

const PLAN_LOCK = ReentrantLock()
const RFFT_PLANS = Dict{Tuple{Int,Int},Any}()
const IRFFT_PLANS = Dict{Tuple{Int,Int},Any}()

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
