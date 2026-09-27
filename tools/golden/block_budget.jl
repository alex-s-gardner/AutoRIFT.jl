# The fastest block size that fits a memory budget, per golden case.
#
#   julia --project=tools/golden tools/golden/block_budget.jl
#   julia --project=tools/golden tools/golden/block_budget.jl --budget 16 --floor 2
#
# `block_optimum.jl` minimizes **peak** and breaks ties on runtime within 3%, which is the right
# objective for "will it fit at all". This asks the other question: given that it fits, how fast can the
# run be? Those pick different arms whenever peak falls monotonically toward the block floor while
# runtime does not, which is the measured shape on both NISAR granules.
#
# Reads `block_optimum.jls`, so it costs nothing and needs no capture. Arms in that file are already
# filtered to the ones that reproduce an untiled run exactly, so every row here is answer-preserving.
#
# **Peak is reported above the harness floor, and the budget is applied to an estimate.**
# `mem_nisar.jl` holds the capture's imagery resident, so its own floor runs from about 3 GiB on a
# Landsat case to 17 on NISAR L2 — larger than the budget, and not something a production run pays. What
# transfers is each arm's peak *above* that floor, plus whatever a production process holds. `--floor`
# is that allowance and defaults to the 2 GiB `dev/plan-16gib.md` uses.

using Printf, Serialization

include(joinpath(@__DIR__, "mem_nisar.jl"))

# Restated rather than taken from `block_optimum.jl`: including that file runs a sweep, since its
# entry point is not guarded on `PROGRAM_FILE`.
const OUT = joinpath(TRACE_DIR, "block_optimum.jls")

argvalue(flag, default) = (i = findfirst(==(flag), ARGS);
                           isnothing(i) ? default : ARGS[i + 1])

const GiB = 2^30

# The fastest arm whose estimated production peak fits, and `nothing` when none does.
fastest_fitting(rows, cap) = begin
    ok = filter(r -> r.peak_above_floor <= cap, rows)
    isempty(ok) ? nothing : argmin(r -> r.seconds, ok)
end

function main()
    budget = parse(Float64, argvalue("--budget", "16")) * GiB
    floorb = parse(Float64, argvalue("--floor", "2")) * GiB
    cap = budget - floorb
    isfile(OUT) || (println("nothing swept yet — run block_optimum.jl"); return)
    done = deserialize(OUT)
    @printf("budget %.0f GiB, production floor allowance %.0f GiB, so an arm may peak %.2f GiB above the harness floor\n",
            budget / GiB, floorb / GiB, cap / GiB)

    @printf("\n%-44s %-13s %8s %8s %7s | %-13s %8s %8s\n", "case",
            "fastest fit", "GiB", "wall s", "blocks", "min above-floor", "GiB", "wall s")
    nfit = nover = 0
    worst = 0.0
    total = 0.0
    for k in sort(collect(keys(done)))
        rows = done[k]
        rows === :failed && (@printf("%-44s FAILED\n", first(k, 44)); continue)
        isempty(rows) && (@printf("%-44s no answer-preserving arm\n", first(k, 44)); continue)
        f = fastest_fitting(rows, cap)
        m = argmin(r -> r.peak_above_floor, rows)
        if isnothing(f)
            nover += 1
            @printf("%-44s %-13s %8s %8s %7s | %-13s %8.2f %8.1f\n", first(k, 44),
                    "NONE FITS", "-", "-", "-", _bslabel(m.block), m.peak_above_floor / GiB,
                    m.seconds)
            continue
        end
        nfit += 1
        worst = max(worst, f.peak_above_floor / GiB)
        total += f.seconds
        @printf("%-44s %-13s %8.2f %8.1f %7d | %-13s %8.2f %8.1f\n", first(k, 44),
                _bslabel(f.block), f.peak_above_floor / GiB, f.seconds, f.nblocks,
                _bslabel(m.block), m.peak_above_floor / GiB, m.seconds)
    end
    @printf("\n%d cases fit, %d do not\n", nfit, nover)
    @printf("worst above-floor peak among the chosen arms: %.2f GiB, so about %.1f GiB in production\n",
            worst, worst + floorb / GiB)
    @printf("total wall clock over the chosen arms: %.0f s\n", total)

    # Where the two objectives disagree, which is the reason this report exists.
    println("\ncases where the fastest fitting arm is not the minimum-peak arm:")
    ndiff = 0
    for k in sort(collect(keys(done)))
        rows = done[k]
        (rows === :failed || isempty(rows)) && continue
        f = fastest_fitting(rows, cap)
        isnothing(f) && continue
        m = argmin(r -> r.peak_above_floor, rows)
        f.block == m.block && continue
        ndiff += 1
        @printf("  %-44s %-13s %6.1f s at %5.2f GiB  vs  %-13s %6.1f s at %5.2f GiB  (%.2fx faster, %+.2f GiB)\n",
                first(k, 44), _bslabel(f.block), f.seconds, f.peak_above_floor / GiB,
                _bslabel(m.block), m.seconds, m.peak_above_floor / GiB,
                m.seconds / f.seconds, (f.peak_above_floor - m.peak_above_floor) / GiB)
    end
    ndiff == 0 && println("  none — minimizing peak also minimizes runtime everywhere that fits")
    return nothing
end

main()
