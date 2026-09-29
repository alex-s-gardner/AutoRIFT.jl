# What a whole NISAR grid costs in peak memory, and how much of it the workspace pool is holding.
#
#   julia --project=tools/golden -t 10,1 tools/golden/pool_peak.jl NISAR_L1_PR_RSLC
#   julia --project=tools/golden -t 10,1 tools/golden/pool_peak.jl NISAR_L1_PR_RSLC --budget 2
#
# `--budget` is GiB across every pooled geometry, `0` or omitted meaning unbounded, which is the default
# `WORKSPACE_POOL_BYTES` ships.
#
# One case per process, because a high-water mark is a property of a process: two configurations in one
# give the first one's peak and then near-zero for the second, which `benchmark/memory.jl` records as
# one of the three ways of getting this wrong.
#
# **What this established.** The pool ends a whole-grid L1 run holding **9.93 GiB across 105 keys**, and
# that looks like the obvious thing to bound — `_radius_bucket` clamps to each level's own maximum
# radius, so the top bucket is a near-duplicate geometry at every level and a finished level can never
# ask for its own again. Bounding the total cuts retention to 1.51 GiB and peak footprint by only
# 2.5-4.3 GiB of 42, because peak is set by what is live *during* a pass rather than by what is retained
# after it. `docs/src/explanation/memory.md` holds both arms and what the runtime cost came out at.
#
# **CPU seconds are reported beside the wall clock, and a conclusion needs both.** A bounded arm that is
# slower at the same CPU seconds was waiting on rebuilds; one whose CPU seconds rose in proportion was
# running at a lower clock, which is what a power-limited machine does and not a property of the code.
# An earlier pass here recorded wall alone and could not tell those apart.
#
# Two peaks are reported and they answer different things. `resident` counts every resident page
# including the clean file-backed ones the mapped imagery populates; `footprint` is what macOS applies
# its own limits to and excludes those. A heap-side change should be read on **`footprint`** —
# `resident` charges a run for page cache its own reads populated.

include("correlator.jl")
include(joinpath(dirname(@__DIR__), "ab", "memtrace.jl"))

using Printf
using AutoRIFT: WORKSPACE_POOL, WORKSPACE_POOL_BYTES, workspace_bytes, clear_workspaces!

argvalue(flag, default) = (i = findfirst(==(flag), ARGS);
                           isnothing(i) ? default : ARGS[i + 1])

gib(x) = Float64(x) / 2^30

# The pool as it stands. Counted over the entries rather than tracked incrementally, because nothing in
# the correlation needs the figure — only this measurement does, and a count here cannot drift from what
# the pool holds.
function pool_state()
    n = sum(length, values(WORKSPACE_POOL); init = 0)
    return (; keys = length(WORKSPACE_POOL), n, bytes = AutoRIFT.pool_bytes())
end

# A bit-level checksum of a displacement plane, so two arms can be compared for equality across
# processes without keeping either. `reinterpret` rather than the values themselves: this has to
# separate `-0.0` from `0.0` and to treat every `NaN` payload as the distinct bits it is, which is
# exactly what a bound claiming bit-identity means.
function checksum(A::AbstractMatrix{Float32})
    acc = 0x0000000000000000
    n = 0
    @inbounds for v in A
        b = UInt64(reinterpret(UInt32, v))
        # An order-dependent mix, so a permutation of the same values does not collide.
        acc = (acc * 0x100000001b3) ⊻ b
        isnan(v) || (n += 1)
    end
    return (acc, n)
end

function main()
    isempty(ARGS) && error("usage: pool_peak.jl <case-fragment> [--run N] [--budget GiB]")
    c = only(cases(ARGS[1]))
    run = parse(Int, argvalue("--run", "100"))
    budget = parse(Float64, argvalue("--budget", "0"))
    WORKSPACE_POOL_BYTES[] = budget <= 0 ? typemax(Int) : round(Int, budget * 2^30)

    # Mapped, so the pair is not on the heap and the peak is the package's own. `xread_mmap` returns the
    # identical array; `tools/ab/xchg.jl` asserts that.
    k = read_capture(c; n = run, mmap = CAPTURE_IMAGERY)
    grid = pointset_from_capture(k)
    kw = kwargs_from_capture(k)
    a, b = k.arrays["in_I1"], k.arrays["in_I2"]

    @printf("case      %s\n", first(c.product, 60))
    @printf("budget    %s\n", budget <= 0 ? "unbounded (default)" : string(budget) * " GiB")
    @printf("scene     %s   grid %s   threads %d\n", string(size(a)), string(size(grid.x)),
            Threads.nthreads())
    flush(stdout)

    # The floor this peak is measured against: what the process holds with the capture read and nothing
    # running. Taken after a full collection so it is a requirement rather than the read's garbage.
    clear_workspaces!()
    GC.gc(true); GC.gc(true)
    buf = zeros(UInt64, 64)
    floor_res, floor_foot = rusage!(buf)

    progress = function (trace, _)
        isempty(trace.footprint) && return
        @printf(stderr, "\r  now %7.1f GiB  peak %7.1f GiB  ", gib(last(trace.footprint)),
                gib(maximum(trace.footprint)))
        flush(stderr)
        return nothing
    end
    hz = tick_rate()
    cpu0 = cpu_seconds!(buf, hz)
    gc0 = Base.gc_time_ns()
    alloc0 = Base.gc_total_bytes(Base.gc_num())
    out, trace, seconds = with_trace(; interval = 0.02, progress) do
        autorift(b, a, grid; kw...)
    end
    cpu = cpu_seconds!(buf, hz) - cpu0
    gcs = (Base.gc_time_ns() - gc0) / 1e9
    alloc = Base.gc_total_bytes(Base.gc_num()) - alloc0
    println(stderr)

    st = pool_state()
    csx, nx = checksum(out.dx)
    csy, _ = checksum(out.dy)

    @printf("\nwall            %8.1f s   cpu %8.1f s   occupancy %5.2f of %d\n",
            seconds, cpu, cpu / seconds, Threads.nthreads())
    # GC share and total allocation, which is what decides whether a per-pass scene pad is worth
    # hoisting: the pad is allocated inside `track!` once per pass and is the largest transient a run
    # makes, so if collection is a small share of the wall clock there is nothing there to win.
    @printf("gc              %8.1f s   %4.1f%% of wall   allocated %8.1f GiB\n",
            gcs, 100 * gcs / seconds, gib(alloc))
    @printf("floor  resident %8.2f GiB   footprint %8.2f GiB\n", gib(floor_res), gib(floor_foot))
    @printf("peak   resident %8.2f GiB   footprint %8.2f GiB\n",
            gib(maximum(trace.resident)), gib(maximum(trace.footprint)))
    @printf("above floor     %8.2f GiB (footprint)\n",
            gib(maximum(trace.footprint) - floor_foot))
    @printf("pool at end     %3d keys / %3d workspaces / %7.3f GiB\n", st.keys, st.n, gib(st.bytes))
    @printf("measured        %8d points\n", nx)
    @printf("checksum dx     %016x\nchecksum dy     %016x\n", csx, csy)
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
