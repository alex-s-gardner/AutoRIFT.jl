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
const FFTW_PATIENT = UInt32(1 << 5)

# The flag every plan in this file is built with, so the four planners cannot drift apart.
#
# `MEASURE`, which times candidate algorithms, rather than `ESTIMATE`, which guesses from a cost model.
# The planning it costs is paid once per size per process, reused by every grid point, and removed
# entirely on later runs by the wisdom file below.
#
# **Not `PATIENT`, which is measurably not worth its planning time at any size this package reaches.**
# `PATIENT` searches compositions of algorithms rather than algorithms, so its planning cost grows
# steeply with the transform while its execution advantage does not. Measured on the real-to-complex
# forward transform, cold plan against warm execution, both flags:
#
#   | size | PATIENT plan | MEASURE plan | execution gain | repaid after |
#   |---|---:|---:|---:|---:|
#   | 28x28 | 97 ms | 0.0 ms | **0.98x** (slower) | never |
#   | 84x84 | 517 ms | 0.0 ms | **0.93x** (slower) | never |
#   | 84x160 | 1479 ms | 0.0 ms | 1.13x | 905,696 executions |
#   | 320x640 | 6346 ms | 0.0 ms | 1.01x | 4,427,357 executions |
#   | 576x1152 | 16,122 ms | 0.1 ms | 1.00x | never |
#   | 2304x4608 | **306,787 ms** | 3873 ms | 1.05x | 13,016 executions |
#
# At the small end `PATIENT` is not faster at all, and at 2304x4608 — a wide search radius on a radar
# grid — it takes **five minutes to plan one transform**. A level executes on the order of a million
# points, so even the sizes where `PATIENT` wins need most of a level's entire point count to break
# even, and the whole-scene cost is dominated by planning rather than correlating.
#
# `EXHAUSTIVE` is further along the same curve and is not worth measuring again.
const PLAN_FLAGS = FFTW_MEASURE

const PLAN_LOCK = ReentrantLock()
# `Ptr{Cvoid}`, so a cache hit yields a concrete type and the `ccall` below is a static call.
const RFFT_PLANS = Dict{Tuple{Int,Int},Ptr{Cvoid}}()
const IRFFT_PLANS = Dict{Tuple{Int,Int},Ptr{Cvoid}}()
# Complex-to-complex, for `Coherence`. Separate caches rather than a flag in the key: these are
# different FFTW plan kinds operating on differently-shaped buffers, and a plan executed with the
# wrong kind is a buffer overrun rather than an error.
const CFFT_PLANS = Dict{Tuple{Int,Int},Ptr{Cvoid}}()
const ICFFT_PLANS = Dict{Tuple{Int,Int},Ptr{Cvoid}}()

# FFTW's sign convention for `fftwf_plan_dft_2d`, from `fftw3.h`. The forward transform uses
# `FFTW_FORWARD`; the inverse uses `FFTW_BACKWARD` and is unnormalised, exactly as the c2r plan is.
const FFTW_FORWARD = Cint(-1)
const FFTW_BACKWARD = Cint(1)

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
# One consequence worth stating, because it will otherwise be mistaken for a bug in the correlator.
# **Wisdom makes `correlation` reproducible only against a fixed wisdom file, not absolutely.** The
# planner picks a different algorithm for the same transform size depending on what it has measured,
# and different algorithms reassociate the same floating-point sum differently. Measured: the same
# binary, run twice with the wisdom file restored in between, gives `correlation` values differing by
# up to 3.6e-7 — while `dx` and `dy` are bit-identical, because a peak's *location* is far from
# sensitive to a 1e-7 perturbation of the surface. That is the honest characterisation: displacements
# are reproducible, the similarity value is reproducible to ~1e-7. Verified across four scene and
# filter combinations at 512² and 1024². It is also why `test/opencv.jl`'s reported worst deviation
# moves between runs without anything having changed.
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
# is precisely the degradation `load_fftw_wisdom!` already tolerates for a read-only depot. Deleting a
# cache is a cost of milliseconds; not trimming costs 450 MiB of resident memory.
const PKG_UUID = Base.UUID("52cb0ed0-aa80-430c-bd04-c52888a79add")

# Set once the wisdom file has been read, so a process imports at most once.
const WISDOM_LOADED = Ref(false)
# Sizes planned since the last export, so an export only happens when there is something new.
const WISDOM_DIRTY = Ref(false)

# Set when `load_fftw_wisdom!` actually imported a file, so the first size measured can tell a machine
# with no FFTW wisdom at all from one whose FFTW wisdom simply does not hold that size.
const WISDOM_IMPORTED = Ref(false)

# Whether the cold-start warning has been emitted in this process. Like `WISDOM_FILENAME`, both of
# these are reset by `reset_fftw_wisdom_path!` in `__init__`: a `Ref` set during precompilation would
# otherwise be serialised into the image and inherited by every process that loads it.
const WISDOM_WARNED = Ref(false)

# The filename component, cached because it is the expensive part and the only part that cannot
# change while the process runs. `Sys.cpu_info()` allocates a vector of per-core structs and the
# regex over its model string costs another 6 us — 12.3 of the 9.6 us a full uncached resolution
# takes. The *directory* is deliberately not cached; see `fftw_wisdom_path`.
#
# `Ref` rather than a `const` binding because `__init__` must be able to re-resolve it: a value
# computed during precompilation would be serialised into the image and inherited by whatever
# machine loads it, which for a CPU-keyed filename is exactly wrong.
const WISDOM_FILENAME = Ref("")

# The directory override, for a deployment whose per-process filesystem does not outlive the process.
# Named for what it holds rather than for the package, since `wisdom` alone names nothing in
# particular.
const FFTW_WISDOM_DIR_VAR = "AUTORIFT_FFTW_WISDOM_DIR"

"""
    AutoRIFT.fftw_wisdom_path() -> String or nothing

Path to this machine's FFTW wisdom file, or `nothing` if no directory is available to hold it.

Keyed by CPU model and FFTW version: wisdom is portable across neither, and importing another
machine's would produce plans tuned for the wrong cache hierarchy. The file name carries both, so a
directory may hold one file per machine type and each process reads only its own.

The directory is a scratch space inside the depot, which on a workstation already outlives any
process — the first run on a machine measures its plans and every run after reads them. Set
`AUTORIFT_FFTW_WISDOM_DIR` to override it, which is what a deployment wants when the process
filesystem is discarded between jobs: pointing it at shared storage turns "measure once per machine"
back into the truth it is on a workstation. Writes there are safe from several processes at once —
see `AutoRIFT.save_fftw_wisdom!`.
"""
function fftw_wisdom_path()
    try
        # `Sys.cpu_info()` can report a model string with spaces and slashes ("Apple M2 Max",
        # "Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz"), none of which belong in a filename. Cached:
        # it is 12.3 us of the cost and cannot change under a running process.
        if isempty(WISDOM_FILENAME[])
            cpu = replace(Sys.cpu_info()[1].model, r"[^A-Za-z0-9._-]+" => "_")
            WISDOM_FILENAME[] = "$(cpu)-fftw$(FFTW_VERSION).wisdom"
        end
        # The directory is resolved every call, and that is not an oversight. `Scratch`'s
        # `with_scratch_directory` redirects `scratch_dir` *dynamically*, and the test suite relies
        # on it to exercise the read-only-depot and corrupt-file paths without touching the real
        # depot. Caching the whole path made that redirect a silent no-op — the isolation tests kept
        # passing while writing into the developer's own scratch space and asserting against the
        # wrong directory. A test that passes for the wrong reason is worse than the 2.3 us this
        # costs (`scratch_dir` 1.0 us, `mkpath` 1.3 us).
        # The override first, so a deployment never depends on where the depot happens to be.
        dir = get(ENV, FFTW_WISDOM_DIR_VAR, "")
        isempty(dir) && (dir = scratch_dir(string(PKG_UUID), "fftw_wisdom"))
        mkpath(dir)
        return joinpath(dir, WISDOM_FILENAME[])
    catch
        # No writable scratch space. Planning still works; it is just never cached.
        return nothing
    end
end

"""
    AutoRIFT.reset_fftw_wisdom_path!()

Forget the cached FFTW wisdom filename and this process's FFTW wisdom state, so the next
[`fftw_wisdom_path`](@ref) re-derives the path and a cold process warns again if it measures.

Called from `__init__`: a filename derived from the CPU model while *precompiling* would otherwise be
serialised into the image and inherited by a machine with a different one.
"""
function reset_fftw_wisdom_path!()
    WISDOM_FILENAME[] = ""
    # The per-process FFTW wisdom state goes with it: whether a file was imported and whether the
    # cold-start warning has been given are both properties of this process, and a value left over
    # from precompilation would make a fresh process silent about measuring everything.
    WISDOM_IMPORTED[] = false
    WISDOM_WARNED[] = false
    return nothing
end

# Called by every plan constructor on a cache miss, where a size is about to be measured.
#
# Measuring *some* sizes is normal and unavoidable: a pass's widest bucket is the grid's own maximum
# rather than a rung of the ladder, so no precomputed file can hold it. What is worth saying out loud is
# measuring with **no FFTW wisdom at all**, which means either a machine that has not run this package
# before or a deployment whose FFTW wisdom is not being found — a container that discards its
# filesystem, or `AUTORIFT_FFTW_WISDOM_DIR` pointing somewhere unwritable.
#
# It is worth a warning rather than a log line because of what it does to measurement. Plan measurement
# is `PLAN_FLAGS` timing candidate algorithms, which on a real image pair reaches minutes, and it lands
# inside whatever the caller is timing — so a benchmark or a profile taken from a run without FFTW
# wisdom describes planning rather than correlation, with nothing in the output to say so.
#
# Once per process: the condition cannot change mid-run, and a correlation visits this path thousands of
# times.
function _measuring_new_size!()
    WISDOM_DIRTY[] = true
    (WISDOM_IMPORTED[] || WISDOM_WARNED[]) && return nothing
    WISDOM_WARNED[] = true
    path = fftw_wisdom_path()
    # The parenthesised call form, because a bare `@warn msg key = val` cannot span lines: a field on
    # its own line parses as a *separate assignment* and silently vanishes from the record.
    @warn("""
          No FFTW wisdom for this machine, so FFTW wisdom is being measured now. This run is slower \
          than a steady-state one, and any benchmark or profile taken from it includes plan \
          measurement rather than only correlation.

          FFTW wisdom is written when the run finishes and read by later processes, so this is a \
          once-per-machine cost — unless the file below does not persist between runs, which is the \
          usual case in a container. Set `AUTORIFT_FFTW_WISDOM_DIR` to a directory that outlives the \
          process, and `AutoRIFT.precompute_fftw_wisdom` to populate it ahead of the first real job.
          """,
          fftw_wisdom_file = isnothing(path) ? "none: no writable directory" : path,
          AUTORIFT_FFTW_WISDOM_DIR = get(ENV, FFTW_WISDOM_DIR_VAR, "unset"))
    return nothing
end

"""
    AutoRIFT.load_fftw_wisdom!()

Import this machine's saved FFTW wisdom, if any. Idempotent and never throws.

Called from `__init__`. A missing, unreadable, or corrupt file is not an error — it means the
next plan is measured from scratch, which is what would happen without any of this.
"""
function load_fftw_wisdom!()
    WISDOM_LOADED[] && return nothing
    WISDOM_LOADED[] = true
    path = fftw_wisdom_path()
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
                # Nonzero is FFTW accepting the file. A corrupt one returns zero, which must leave
                # `WISDOM_IMPORTED` false so the warning fires — silently planning everything from
                # scratch because a cache file is damaged is exactly what it exists to surface.
                ok = ccall((:fftwf_import_wisdom_from_file, LIBFFTW3F), Cint, (Ptr{Cvoid},), f)
                WISDOM_IMPORTED[] = ok != 0
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
    AutoRIFT.save_fftw_wisdom!()

Write this machine's FFTW wisdom, if any new size has been planned. Never throws.

Called after [`warm_plans!`](@ref) rather than at exit: an `atexit` hook would miss a process
killed by a scheduler, which for batch work is the normal way for a process to end.
"""
function save_fftw_wisdom!()
    WISDOM_DIRTY[] || return nothing
    # Cleared up front, on every exit path rather than only the successful one. The distinction that
    # matters is not success versus failure but "worth retrying" versus not, and none of the ways
    # this fails is worth retrying: an unwritable depot, a full disk and a missing scratch directory
    # are all properties of the environment, and they do not become writable between two chip-size
    # levels of the same image pair.
    #
    # Leaving the flag set on failure is what the earlier version did, and it turned one dead export
    # into a permanent one. `warm_plans!` calls this once per chip-size level per pair, so every
    # later call re-derived the path and retried the doomed write: measured 116 us and 6.6 KiB a
    # call, 202 extra allocations per image pair, for the whole life of the process. The cost of
    # clearing eagerly is that a genuinely transient failure — a disk that frees up mid-run — is not
    # retried until some new size is planned. That is the right trade: wisdom is an optimization, and
    # the next process picks it up anyway.
    WISDOM_DIRTY[] = false
    path = fftw_wisdom_path()
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
            # `PLAN_FLAGS` measures rather than estimates: planning is paid once per size
            # per process and reused across every grid point, so the extra planning time
            # is recovered immediately. Wisdom persistence (see `__init__`) removes even
            # that cost on subsequent runs.
            _measuring_new_size!()
            # FFTW is row-major and Julia column-major, so an `ny`-by-`nx` Julia array is an
            # `nx`-by-`ny` C array — the dimensions go in reversed. Getting this backwards
            # transposes every transform, which for a symmetric test size looks like it works.
            inb = Matrix{Float32}(undef, ny, nx)
            outb = Matrix{ComplexF32}(undef, ny ÷ 2 + 1, nx)
            plan = ccall((:fftwf_plan_dft_r2c_2d, LIBFFTW3F), Ptr{Cvoid},
                         (Cint, Cint, Ptr{Float32}, Ptr{ComplexF32}, Cuint),
                         nx, ny, inb, outb, PLAN_FLAGS)
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
            _measuring_new_size!()
            inb = Matrix{ComplexF32}(undef, ny ÷ 2 + 1, nx)
            outb = Matrix{Float32}(undef, ny, nx)
            plan = ccall((:fftwf_plan_dft_c2r_2d, LIBFFTW3F), Ptr{Cvoid},
                         (Cint, Cint, Ptr{ComplexF32}, Ptr{Float32}, Cuint),
                         nx, ny, inb, outb, PLAN_FLAGS)
            plan == C_NULL && error("FFTW could not plan a $(ny)x$(nx) complex-to-real " *
                                    "transform.")
            plan
        end
    end
end

"""
    cfft_plan(ny, nx)   /   icfft_plan(ny, nx)

Complex-to-complex forward and inverse FFT plans for an `ny`-by-`nx` `ComplexF32` array.

Used by [`Coherence`](@ref), whose chip is complex and so cannot go through the real-to-complex
transform the real measures use. Both directions are full `ny`-by-`nx` — there is no conjugate
symmetry to exploit — so a complex transform moves about twice the data of the real one at the
same size. That is the price of phase, and it is far below the price of not transforming at all.

Same `(nx, ny)` argument reversal and the same wisdom persistence as [`fft_plan`](@ref).
"""
function cfft_plan(ny::Int, nx::Int)
    key = (ny, nx)
    p = get(CFFT_PLANS, key, C_NULL)
    p === C_NULL || return p
    return lock(PLAN_LOCK) do
        get!(CFFT_PLANS, key) do
            _measuring_new_size!()
            inb = Matrix{ComplexF32}(undef, ny, nx)
            outb = Matrix{ComplexF32}(undef, ny, nx)
            plan = ccall((:fftwf_plan_dft_2d, LIBFFTW3F), Ptr{Cvoid},
                         (Cint, Cint, Ptr{ComplexF32}, Ptr{ComplexF32}, Cint, Cuint),
                         nx, ny, inb, outb, FFTW_FORWARD, PLAN_FLAGS)
            plan == C_NULL && error("FFTW could not plan a $(ny)x$(nx) forward complex " *
                                    "transform.")
            plan
        end
    end
end

function icfft_plan(ny::Int, nx::Int)
    key = (ny, nx)
    p = get(ICFFT_PLANS, key, C_NULL)
    p === C_NULL || return p
    return lock(PLAN_LOCK) do
        get!(ICFFT_PLANS, key) do
            _measuring_new_size!()
            inb = Matrix{ComplexF32}(undef, ny, nx)
            outb = Matrix{ComplexF32}(undef, ny, nx)
            plan = ccall((:fftwf_plan_dft_2d, LIBFFTW3F), Ptr{Cvoid},
                         (Cint, Cint, Ptr{ComplexF32}, Ptr{ComplexF32}, Cint, Cuint),
                         nx, ny, inb, outb, FFTW_BACKWARD, PLAN_FLAGS)
            plan == C_NULL && error("FFTW could not plan a $(ny)x$(nx) inverse complex " *
                                    "transform.")
            plan
        end
    end
end

"""
    cfft_execute!(plan, input, output)

Run a complex-to-complex plan from [`cfft_plan`](@ref) or [`icfft_plan`](@ref).

**Unnormalised in both directions**, like FFTW's c2r transform — the caller applies `1/n` after
the inverse. Unlike c2r, this transform does *not* destroy its input, so a spectrum may be reused.
"""
@inline function cfft_execute!(plan::Ptr{Cvoid}, input::Matrix{ComplexF32},
                               output::Matrix{ComplexF32})
    ccall((:fftwf_execute_dft, LIBFFTW3F), Cvoid,
          (Ptr{Cvoid}, Ptr{ComplexF32}, Ptr{ComplexF32}), plan, input, output)
    return output
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
    warm_plans!(sizes; complex = false)

Create the plans for `sizes` on the calling task, before any parallel work starts.

Without this, every task racing to correlate its first point would contend on the
planner lock and serialise — turning the most parallel part of the run into its
most serial. Called once per pass, where the set of sizes is known in advance.

`persist` writes the FFTW wisdom measured here when the call finishes. Pass `false` from a caller
that warms many sizes in sequence and persists on its own schedule: FFTW's export has no incremental
form, so every call rewrites the whole accumulated file.

`complex` selects the complex-to-complex pair that [`Coherence`](@ref) executes instead of the
real-to-complex pair the real measures use. It must match the measure the pass will actually run:
warming the wrong kind is doubly wrong, since it pays full `PLAN_FLAGS` planning — hundreds of
milliseconds per size cold — for a plan that is never executed, *and* leaves the plans that are
executed to be created inside a worker task, which is precisely the planner contention this
function exists to prevent. Verified
against a coherence pass before this argument existed: it warmed `RFFT_PLANS[(72,72)]`, never used
it, and built `CFFT_PLANS[(72,72)]` lazily under the lock.
"""
function warm_plans!(sizes; complex::Bool = false, persist::Bool = true)
    for (ny, nx) in sizes
        if complex
            cfft_plan(ny, nx)
            icfft_plan(ny, nx)
        else
            fft_plan(ny, nx)
            ifft_plan(ny, nx)
        end
    end
    # Persist whatever was measured, so the next process starts warm. Only writes if a plan was
    # actually created — the common case after the first run is that this does nothing.
    persist && save_fftw_wisdom!()
    return nothing
end

# Every bucket the ladder can produce for a radius in `1:rmax`.
#
# Data-independent, which is the whole point: `_radius_bucket` quantizes a radius onto a fixed ladder,
# so this is the complete set of buckets reachable by *any* image pair, whatever its time separation.
# A cap of `rmax + 1` is passed so no radius in the range clamps — the clamped bucket is the one case
# this cannot enumerate, and [`precompute_fftw_wisdom`](@ref) says so.
function _bucket_rungs(rmax::Integer)
    out = Set{Int}()
    for r in 1:Int(rmax)
        push!(out, _radius_bucket(r, Int(rmax) + 1))
    end
    return sort!(collect(out))
end

# The transform sizes one chip extent reaches over a set of buckets, in both axes independently.
function _ladder_sizes(chip, rungs)
    out = Set{Tuple{Int,Int}}()
    for by in rungs, bx in rungs
        push!(out, _padded_fft_size(chip, by, bx))
    end
    return out
end

"""
    precompute_fftw_wisdom(p::Params; max_search_radius) -> Int

Measure and persist the FFTW plans `p` can reach at any search radius up to `max_search_radius`, and
return how many distinct transform sizes were planned.

**What this is for.** A plan is measured on first use and persisted by
[`AutoRIFT.fftw_wisdom_path`](@ref), so a process that finds wisdom on disk starts at full speed and
one that does not pays `PLAN_FLAGS` planning for every size its grid happens to reach. Left to
discover sizes as jobs arrive, a shared wisdom file warms up raggedly — whichever pairs run first
contribute their sizes and the rest keep paying. This plans the whole reachable set up front, so the
first real job is as warm as the thousandth.

**Why a radius bound is the only thing it needs.** A point is correlated at its radius rounded up to a
rung of the `AutoRIFT._radius_bucket` ladder, and the rungs are fixed rather than derived from
the data. So although search radius grows with a pair's time separation, that only changes *which*
rungs are reached, never what they are: one precomputation covers every time separation up to the
bound. Pass the widest radius the deployment can produce.

**The one size it cannot cover.** `_radius_bucket` returns the pass maximum verbatim for a radius at or
above it, and that maximum is a property of the grid rather than of the ladder. Sizes involving it are
left to be measured on first use — measured at 10-15% of the sizes a real optical pair reaches.

Costs what it saves: planning is 116-347 ms at the sizes an optical pair reaches and 8.2 s at the
widest a NISAR pair with an 8 km search would, so a few hundred sizes is minutes and a few thousand is
tens of minutes — once per machine type. `max_search_radius` is required because no default is right for
every sensor.

Reports progress no more often than every `progress_interval` seconds, and persists what it has measured
at the same time, so a long precomputation can be interrupted without losing its work and resumed by
running it again.
"""
function precompute_fftw_wisdom(p::Params; max_search_radius::Integer,
                                progress_interval::Real = 10.0)
    max_search_radius >= 1 || throw(ArgumentError(
        "max_search_radius must be at least 1, got $max_search_radius"))
    rungs = _bucket_rungs(max_search_radius)
    levels = chip_sizes(p)
    # Split by plan kind rather than planned per level, because a size reached by two levels is one
    # plan, and warming the wrong kind pays full planning for a plan that is never executed.
    real_sizes = Set{Tuple{Int,Int}}()
    complex_sizes = Set{Tuple{Int,Int}}()
    for k in eachindex(levels)
        dest = _wants_complex_plans(measure_at(p, k)) ? complex_sizes : real_sizes
        union!(dest, _ladder_sizes(levels[k], rungs))
    end

    # Ascending by transform area: the cheap sizes land first, so an operator sees progress early, and
    # the expensive tail is where a report matters most.
    work = vcat([(sz, false) for sz in real_sizes], [(sz, true) for sz in complex_sizes])
    sort!(work; by = w -> prod(first(w)))
    total = length(work)
    total_cost = sum(_plan_cost_weight(first(w)) for w in work; init = 0.0)

    @info "precomputing FFTW wisdom" sizes = total rungs = length(rungs) levels = length(levels) fftw_wisdom_file = fftw_wisdom_path()
    started = time()
    last_report = started
    done_cost = 0.0
    for (i, (sz, complex)) in enumerate(work)
        # `persist = false`: this loop writes FFTW wisdom on its own schedule below, where
        # `warm_plans!` would otherwise rewrite the whole accumulated file once per size.
        warm_plans!((sz,); complex, persist = false)
        done_cost += _plan_cost_weight(sz)
        now = time()
        if i < total && now - last_report >= progress_interval
            save_fftw_wisdom!()
            elapsed = now - started
            @info "precomputing FFTW wisdom" progress = "$i/$total sizes" elapsed_s = round(elapsed; digits = 1) remaining_s = round(elapsed * (total_cost - done_cost) / max(done_cost, eps()); digits = 1)
            last_report = now
        end
    end
    save_fftw_wisdom!()
    @info "FFTW wisdom precomputed" sizes = total seconds = round(time() - started; digits = 1) fftw_wisdom_file = fftw_wisdom_path()
    return total
end

# What one size is expected to cost to plan, relative to the others, for the remaining-time estimate.
#
# The square root of the transform area, which is a measurement rather than a guess: on this package's
# own sizes `FFTW_MEASURE` took 1.52 s at 2.0 Mpt and 8.16 s at 49.5 Mpt, so 25x the area for 5.4x the
# time — an exponent of 0.52. Weighting by count instead would underestimate badly, since the sizes are
# planned cheapest-first and the tail is where the time is.
_plan_cost_weight(sz::Tuple{Int,Int}) = sqrt(float(sz[1]) * float(sz[2]))

"""
    clear_plans!()

Drop all cached plans. For tests and benchmarks that need to measure planning cost;
not part of normal operation.
"""
function clear_plans!()
    lock(PLAN_LOCK) do
        empty!(RFFT_PLANS)
        empty!(IRFFT_PLANS)
        # The complex caches must be emptied here too, and this is not tidiness: `__init__` calls
        # this to discard plans serialised into the precompile image, and a dangling plan handle
        # segfaults inside FFTW rather than erroring. Missing a cache here would reintroduce
        # precisely the crash `test/plans.jl`'s regression test exists to catch.
        empty!(CFFT_PLANS)
        empty!(ICFFT_PLANS)
    end
    return nothing
end
