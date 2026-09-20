# Timing and memory for one *phase* of a process, rather than for the process.
#
# `/usr/bin/time -l` answers "what did this row cost end to end", which includes reading the scene,
# loading a runtime and — on the reference — filtering and casting the imagery. A correlator
# comparison needs the cost from the correlator's own entry point onward, so this brackets the call
# itself: wall clock, CPU time, and the memory high-water reached while it ran.
#
# `phase.py` is the same measurement on the reference's side, printing the same line, so a Julia row
# and a Python row are read on one accounting. Any change here belongs there too.
#
# `proc_pid_rusage` supplies all four figures from one syscall, which is why it is used rather than
# `getrusage` plus `Sys.maxrss`: CPU time and the two memory figures then come from the same read of
# the same kernel structure.

module Phase

# `rusage_info_v4`: a 16-byte uuid, then `UInt64` fields. Indices into a `UInt64` buffer, so the uuid
# occupies 1:2. The two memory figures are bytes; the two CPU figures are **mach absolute time units**,
# not the nanoseconds the header comment on that struct suggests.
const RUSAGE_INFO_V4 = Cint(4)
const I_USER, I_SYSTEM, I_RESIDENT, I_FOOTPRINT = 3, 4, 9, 10

"""
    ns_per_tick() -> Float64

Nanoseconds per mach absolute time unit, from `mach_timebase_info`.

`proc_pid_rusage` reports CPU time in those units, and on Apple silicon one is 125/3 ns rather than 1 —
so treating the figure as nanoseconds understates CPU time by a factor of 42 and makes a saturated
multithreaded run look like it used a twentieth of a core. Queried rather than hard-coded: the ratio is
a machine property, and Intel Macs report 1/1.
"""
function ns_per_tick()
    tb = zeros(UInt32, 2)
    rc = ccall(:mach_timebase_info, Cint, (Ptr{UInt32},), tb)
    rc == 0 || error("mach_timebase_info failed with $rc")
    return tb[1] / tb[2]
end

"""
    rusage!(buf) -> (user_ns, system_ns, resident, footprint)

This process's CPU time and current memory, from one `proc_pid_rusage` call.

`resident` counts every resident page including clean file-backed ones, so a memory-mapped input
inflates it with page cache; `footprint` is the figure macOS enforces its own memory limits against
and excludes that. Both are kept because a lazily-read configuration must not be charged for the
page cache its own reads populated.

`buf` is scratch of at least 64 `UInt64`s, reused so that a sample allocates nothing.
"""
function rusage!(buf::Vector{UInt64})
    rc = ccall(:proc_pid_rusage, Cint, (Cint, Cint, Ptr{UInt64}),
               getpid(), RUSAGE_INFO_V4, buf)
    rc == 0 || error("proc_pid_rusage failed with $rc")
    return (buf[I_USER], buf[I_SYSTEM], buf[I_RESIDENT], buf[I_FOOTPRINT])
end

"""
    measure(f; interval = 0.05) -> (result, metrics)

Run `f()` and return its value with the phase's cost.

`metrics` carries `wall` and `cpu` in seconds, the memory high-water `peak_res`/`peak_foot` reached
during the call, and `start_res`/`start_foot` — what the process already held when the call began,
which is the part of a whole-process peak that belongs to getting the inputs there rather than to
correlating them.

The sampler runs on the **interactive** thread pool: a `:default` task queues behind the
correlation's own tasks and leaves second-long gaps exactly where the peak is, so a process that has
no interactive thread is an error rather than a quietly coarser measurement.
"""
function measure(f; interval::Real = 0.05)
    Threads.nthreadpools() >= 2 && Threads.nthreads(:interactive) >= 1 || error(
        "phase sampling needs an interactive thread so it keeps sampling while the correlation " *
        "saturates the default pool; start Julia with `-t N,1`")
    buf = zeros(UInt64, 64)
    u0, s0, res0, foot0 = rusage!(buf)
    peak_res, peak_foot = res0, foot0
    stop = Threads.Atomic{Bool}(false)
    sampler = Threads.@spawn :interactive begin
        b = zeros(UInt64, 64)
        while !stop[]
            _, _, r, fp = rusage!(b)
            peak_res = max(peak_res, r)
            peak_foot = max(peak_foot, fp)
            sleep(interval)
        end
    end
    t0 = time_ns()
    result = try
        f()
    finally
        stop[] = true
        wait(sampler)
    end
    wall = (time_ns() - t0) / 1e9
    u1, s1, res1, foot1 = rusage!(buf)
    peak_res = max(peak_res, res1)
    peak_foot = max(peak_foot, foot1)
    cpu = ((u1 - u0) + (s1 - s0)) * ns_per_tick() / 1e9
    return result, (; wall, cpu, peak_res, peak_foot, start_res = res0, start_foot = foot0)
end

"""
    report(name, metrics)

Print the phase line `bench_table.jl` parses out of a row's log.

One line of `key=value` pairs, bytes for memory and seconds for time, so the parser needs no column
order and a row that predates a field is missing it rather than misreading its neighbour.
"""
function report(name::AbstractString, m)
    println("PHASE $name wall=$(m.wall) cpu=$(m.cpu) peak_res=$(m.peak_res) " *
            "peak_foot=$(m.peak_foot) start_res=$(m.start_res) start_foot=$(m.start_foot)")
    flush(stdout)
    return nothing
end

end
