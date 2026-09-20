"""Timing and memory for one *phase* of a process, rather than for the process.

`/usr/bin/time -l` answers "what did this row cost end to end", which on the reference includes
reading the scene planes, high-pass filtering them and casting to uint8. A correlator comparison
needs the cost from `runAutorift`'s own entry point onward, so this brackets the call itself: wall
clock, CPU time, and the memory high-water reached while it ran.

`phase.jl` is the same measurement on AutoRIFT.jl's side, printing the same line, so a Python row and
a Julia row are read on one accounting. Any change here belongs there too.

`proc_pid_rusage` supplies all four figures from one syscall, which is why it is used rather than
`resource.getrusage` plus `psutil`: CPU time and the two memory figures then come from the same read
of the same kernel structure, and it is the same call `phase.jl` makes.
"""

import ctypes
import os
import threading
import time

# `rusage_info_v4`: a 16-byte uuid, then `uint64` fields. Indices into a `uint64` view, so the uuid
# occupies 0:1. The two memory figures are bytes; the two CPU figures are **mach absolute time units**,
# not the nanoseconds the header comment on that struct suggests.
RUSAGE_INFO_V4 = 4
I_USER, I_SYSTEM, I_RESIDENT, I_FOOTPRINT = 2, 3, 8, 9

_libc = ctypes.CDLL(None, use_errno=True)


def ns_per_tick():
    """Nanoseconds per mach absolute time unit, from `mach_timebase_info`.

    `proc_pid_rusage` reports CPU time in those units, and on Apple silicon one is 125/3 ns rather
    than 1 — so treating the figure as nanoseconds understates CPU time by a factor of 42. Queried
    rather than hard-coded: the ratio is a machine property, and Intel Macs report 1/1.
    """
    tb = (ctypes.c_uint32 * 2)()
    rc = _libc.mach_timebase_info(ctypes.byref(tb))
    if rc != 0:
        raise OSError(f"mach_timebase_info failed with {rc}")
    return tb[0] / tb[1]


def rusage():
    """This process's `(user_ns, system_ns, resident, footprint)`, from one `proc_pid_rusage` call.

    `resident` counts every resident page including clean file-backed ones; `footprint` is the figure
    macOS enforces its own memory limits against and excludes that. Both are returned because a
    lazily-read configuration must not be charged for the page cache its own reads populated.
    """
    buf = (ctypes.c_uint64 * 64)()
    rc = _libc.proc_pid_rusage(ctypes.c_int(os.getpid()), ctypes.c_int(RUSAGE_INFO_V4),
                               ctypes.byref(buf))
    if rc != 0:
        raise OSError(ctypes.get_errno(), f"proc_pid_rusage failed with {rc}")
    return buf[I_USER], buf[I_SYSTEM], buf[I_RESIDENT], buf[I_FOOTPRINT]


def measure(fn, interval=0.05):
    """Run `fn()` and return `(result, metrics)`, where `metrics` is the phase's cost.

    `wall` and `cpu` are seconds; `peak_res`/`peak_foot` are the high-water reached during the call
    and `start_res`/`start_foot` what the process already held when it began — the part of a
    whole-process peak that belongs to getting the inputs there rather than to correlating them.

    The sampler is a thread rather than a subprocess poll so a sample costs one syscall. It samples
    while `runAutorift` is inside numpy and OpenCV, which release the GIL.
    """
    u0, s0, res0, foot0 = rusage()
    peak = [res0, foot0]
    stop = threading.Event()

    def sampler():
        while not stop.wait(interval):
            _, _, res, foot = rusage()
            peak[0] = max(peak[0], res)
            peak[1] = max(peak[1], foot)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    t0 = time.perf_counter()
    try:
        result = fn()
    finally:
        stop.set()
        thread.join()
    wall = time.perf_counter() - t0
    u1, s1, res1, foot1 = rusage()
    metrics = {
        "wall": wall,
        "cpu": ((u1 - u0) + (s1 - s0)) * ns_per_tick() / 1e9,
        "peak_res": max(peak[0], res1),
        "peak_foot": max(peak[1], foot1),
        "start_res": res0,
        "start_foot": foot0,
    }
    return result, metrics


def report(name, metrics):
    """Print the phase line `bench_table.jl` parses out of a row's log.

    One line of `key=value` pairs, bytes for memory and seconds for time, matching `phase.jl`
    exactly — the parser is shared, so a divergence in this format is a divergence in the table.
    """
    print("PHASE %s wall=%r cpu=%r peak_res=%d peak_foot=%d start_res=%d start_foot=%d"
          % (name, metrics["wall"], metrics["cpu"], metrics["peak_res"], metrics["peak_foot"],
             metrics["start_res"], metrics["start_foot"]), flush=True)
