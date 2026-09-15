# Minimal reproducer for the macOS profiler/GC deadlock that stopped two NISAR sweep runs.
#
#   julia -t 10,1 tools/golden/profiler_gc_deadlock.jl off       # always completes
#   julia -t 10,1 tools/golden/profiler_gc_deadlock.jl profile   # hangs at 0% CPU ~2 runs in 5
#
# No AutoRIFT, no imagery: allocation churn on every thread while `Profile` samples at 0.5 ms. The
# `off` arm is the control and runs the identical workload without the profiler, which is what makes
# the profiler rather than the allocation rate the cause.
#
# **It is a race, so expect to run the `profile` arm several times.** Measured 2 hangs in 5 attempts on
# 10 threads, landing anywhere from round 25 to round 106 of 200; the `off` arm has never hung. A single
# clean `profile` run is therefore not evidence the bug is absent — which is the trap to avoid, since
# it is what made the first occurrence look contention-dependent.
#
# **This is a Julia runtime bug, not a package one** — `src/signals-mach.c` in every release through
# 1.13.0. Two locks are taken in opposite orders:
#
#   * The profiler's sampling thread calls `jl_lock_profile_mach`, then suspends its target while
#     still holding that lock (`jl_profile_thread_mach`, `signals-mach.c:797`). Suspending goes
#     through `pthread_mach_thread_np`, which takes libpthread's internal `os_unfair_lock`.
#   * A thread finishing a collection resumes the threads it stopped, from `jl_mach_gc_end`
#     (`signals-mach.c:97`) — `thread_resume(pthread_mach_thread_np(...))`, which wants that same
#     `os_unfair_lock`, while holding `safepoint_lock`.
#
# Interleave them and the sampler waits on the unfair lock while the collector waits for the sampler
# to release it. Every other thread then piles up at `jl_safepoint_start_gc` behind a collection that
# has begun and can never end, so `sample` shows 0 threads marking or sweeping. The process is
# unkillable by `SIGTERM` and holds its full footprint.
#
# Fixed upstream by `ca49fc2e2` ("[macOS] Handle GC safepoint on-thread", 2026-03-17), which deletes
# `jl_mach_gc_end` and the `suspended_threads` list outright and handles the safepoint on the
# signalled thread. Present in 1.14-DEV; **not** backported to release-1.12 or release-1.13.
#
# Until then, on macOS: do not profile a long multithreaded allocation-heavy run. `profile_nisar.jl`
# times a run with the profiler off and profiles a separate one, so a hang costs the attribution and
# not the measurement.

using Profile, Printf

# Allocation churn: many short-lived arrays, dropped periodically so the collector runs often. The
# point is collection frequency on every thread at once, not the values.
function churn(n)
    a = Vector{Vector{Float64}}(undef, 0)
    for i in 1:n
        push!(a, rand(64))
        i % 512 == 0 && (a = Vector{Vector{Float64}}(undef, 0))
    end
    return length(a)
end

function hammer(iters)
    ts = map(1:Threads.nthreads(:default)) do _
        Threads.@spawn churn(iters)
    end
    return sum(fetch.(ts))
end

function run_rounds()
    for r in 1:200
        hammer(400_000)
        @printf("\r  round %3d", r)
        flush(stdout)
    end
    return nothing
end

function main()
    mode = isempty(ARGS) ? "profile" : ARGS[1]
    mode in ("off", "profile") || error("usage: profiler_gc_deadlock.jl [off|profile]")
    hammer(10_000)  # warm, so a hang is not confused with compilation

    if mode == "profile"
        # A fine interval maximizes suspend/resume cycles per second, which is what makes the
        # interleaving likely rather than rare. The buffer is sized so it cannot fill and stop early.
        Profile.init(; n = 200_000_000, delay = 0.0005)
        Profile.clear()
    end
    @printf("%s, %d threads — expect %s\n",
            mode == "profile" ? "profiling at 0.5 ms" : "no profiler",
            Threads.nthreads(:default), mode == "profile" ? "a hang" : "completion")
    flush(stdout)

    mode == "profile" ? Profile.@profile(run_rounds()) : run_rounds()

    @printf("\nCOMPLETED %s\n", mode)
    return nothing
end

main()
