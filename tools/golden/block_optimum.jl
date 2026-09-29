# The block size that minimizes peak memory for every golden case, measured rather than predicted.
#
#   julia --project=tools/golden -t 12,1 tools/golden/block_optimum.jl            # sweep every case
#   julia --project=tools/golden -t 12,1 tools/golden/block_optimum.jl --report   # table from disk
#   julia --project=tools/golden -t 12,1 tools/golden/block_optimum.jl LT05 S1B   # named cases only
#
# `AutoRIFT.block_size_for` picks a default from the halo and a target block count, and it picks badly:
# 3660x1370 on the golden S1B case where 1024x1024 wins on peak, and 6586x2422 on
# `S1A_IW_SLC__1SSV_20240618T025528` where nothing had been measured at all. This sweeps each case's
# ladder so the default can be fitted to measurements instead.
#
# **The optimum is interior, which is why a ladder is needed rather than a rule.** Peak falls as the block
# shrinks; read amplification rises. On S1B: 3.24 GiB at 768x320 with readamp 7.23x, 3.01 GiB at 1024 with
# 2.88x, 10.32 GiB at 2048 with 1.44x. Neither endpoint is the answer and neither is the halo's shape —
# see `_auto_blocks` in `mem_nisar.jl` for the arms and why a shaped one is tried but not assumed.
#
# **Untiled is a real candidate, not just the reference arm.** A small Landsat scene fits in memory whole,
# and blocking it adds buffers without saving anything: `LT05_L1TP_060018` peaks at 2.45 GiB untiled
# against 3.62 GiB at its best block size. A default that always blocks is wrong for those cases.
#
# Each arm's `dx`/`dy` are asserted identical to the untiled run's, so a row is only reported when the
# configuration is answer-preserving. A case whose sweep throws is recorded as failed and the sweep
# continues — one bad case must not cost the other twenty-one.

using Printf, Serialization

include(joinpath(@__DIR__, "mem_nisar.jl"))

const OUT = joinpath(TRACE_DIR, "block_optimum.jls")

# The run number to read for a case: whichever captured run exists, preferring the lowest so the choice
# is reproducible rather than dependent on directory order.
function captured_run(c::GoldenCase)
    dir = run_dir(c, 0)
    base = dirname(dir)
    isdir(base) || return nothing
    ns = Int[]
    for e in readdir(base)
        n = tryparse(Int, e)
        isnothing(n) && continue
        isdir(joinpath(base, e, "capture")) && push!(ns, n)
    end
    return isempty(ns) ? nothing : minimum(ns)
end

# Peak is the objective, because fitting 16 GiB is what block size is chosen for. Runtime breaks a tie
# within 3%, which is inside the run-to-run scatter this repo records for a single blocked run.
function pick(rows)
    isempty(rows) && return nothing
    best = argmin(r -> r.peak, rows)
    close = filter(r -> r.peak <= 1.03 * best.peak, rows)
    return argmin(r -> r.seconds, close)
end

function sweep(frags)
    all = cases()
    want = isempty(frags) ? all :
           filter(c -> any(f -> occursin(f, c.product), frags), all)
    done = isfile(OUT) ? deserialize(OUT) : Dict{String,Any}()
    for (i, c) in enumerate(want)
        n = captured_run(c)
        if isnothing(n)
            @printf("\n[%d/%d] %s\n  SKIP — no captured run\n", i, length(want),
                    first(c.product, 60))
            continue
        end
        haskey(done, c.product) && done[c.product] isa Vector && !isempty(done[c.product]) && begin
            @printf("\n[%d/%d] %s\n  already swept (%d arms)\n", i, length(want),
                    first(c.product, 60), length(done[c.product]))
            continue
        end
        @printf("\n[%d/%d] %s   run %d\n", i, length(want), first(c.product, 60), n)
        flush(stdout)
        try
            # **Unprofiled, for two reasons.** The runtime is part of the answer here, and sampling every
            # thread every 2 ms perturbs it — `profile_nisar.jl` measures its timing row with the
            # profiler off for exactly that reason. And a sampled multithreaded run can deadlock against
            # the GC on macOS (`profiler_gc_deadlock.jl`): a first attempt at this sweep hung on case 15
            # of 22 with no output for 54 minutes, which is fatal to an unattended run.
            rows = measure_case(c; n, profile = false)
            report_agreement(rows)
            base = findfirst(r -> r.block == (0, 0), rows)
            ok = isnothing(base) ? rows :
                 filter(r -> isequal(r.dx, rows[base].dx) && isequal(r.dy, rows[base].dy), rows)
            # Stripped of the fields, which are the bulk; the agreement they were needed for is decided.
            done[c.product] = [Base.structdiff(r, (; dx = 0, dy = 0, stacks = 0)) for r in ok]
        catch e
            @printf("  FAILED — %s\n", first(sprint(showerror, e), 300))
            done[c.product] = :failed
        end
        serialize(OUT, done)
        flush(stdout)
    end
    return done
end

function report()
    isfile(OUT) || (println("nothing swept yet"); return)
    done = deserialize(OUT)
    @printf("\n%-46s %-13s %9s %8s %7s %8s\n",
            "case", "best block", "peak GiB", "wall s", "readamp", "blocks")
    tot = 0
    for k in sort(collect(keys(done)))
        rows = done[k]
        rows === :failed && (@printf("%-46s %-13s\n", first(k, 46), "FAILED"); continue)
        b = pick(rows)
        isnothing(b) && continue
        tot += 1
        @printf("%-46s %-13s %9.2f %8.1f %7.2f %8d\n", first(k, 46), _bslabel(b.block),
                b.peak / 2^30, b.seconds, b.readamp, b.nblocks)
    end
    @printf("\n%d cases with a measured optimum\n", tot)
    # What a default has to reproduce, as the ratio of the chosen block to the halo in each axis. A single
    # multiplier that fits every row is what `block_size_for` could then use.
    println("\nchosen block against halo:")
    for k in sort(collect(keys(done)))
        rows = done[k]
        rows === :failed && continue
        b = pick(rows)
        (isnothing(b) || b.block == (0, 0)) && continue
        @printf("  %-46s halo %5d x %-5d  block %-13s  %.2fx %.2fx\n", first(k, 46),
                b.halo[1], b.halo[2], _bslabel(b.block),
                b.block[1] / b.halo[1], b.block[2] / b.halo[2])
    end
    return nothing
end

if "--report" in ARGS
    report()
else
    sweep(filter(a -> !startswith(a, "--"), ARGS))
    report()
end
