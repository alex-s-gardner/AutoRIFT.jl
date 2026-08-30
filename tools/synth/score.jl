# Every arm scored against exact truth, and the tables the README quotes.
#
#   julia --project=tools/synth tools/synth/score.jl
#
# Writes `results.json` -- the per-case, per-arm numbers, which is the committed record -- and prints
# the tables. Runs `gate.jl`'s convention table rather than restating it, so the signs used here are
# the ones measured there.
#
# Errors are reported as medians, not means. A correlator failure is a gross outlier of several pixels,
# not a small perturbation, so a mean over a grid containing a few railed points describes the failures
# rather than the typical accuracy. The fraction beyond 0.25 px is reported alongside, which is where
# the outliers are accounted for.

using JSON3, Printf, Statistics

include(joinpath(@__DIR__, "bundle.jl"))
include(joinpath(@__DIR__, "gate.jl"))

const RESULTS = joinpath(@__DIR__, "results.json")

# JSON has no literal for a non-finite number, so `NaN` becomes `null` on the way out: it means "this
# arm measured nothing here", and `null` is how JSON says that. Writing a string like "NaN" instead
# would silently change the field's type for every consumer.
jsonsafe(x::Real) = isfinite(x) ? x : nothing
jsonsafe(x) = x

# The threshold the real-scene comparison uses, so the two are read on the same scale.
const TOL = 0.25

"""
    score_arm(dir, arm, tdx, tdy) -> Union{NamedTuple,Nothing}

One arm's accuracy on one case: coverage, error quantiles, per-axis bias, and cost.

Bias is reported per axis and separately from the error magnitude because they are different defects: a
median error of 0.2 px with zero bias is noise, while the same error all of one sign is a systematic
offset. The real-scene comparison found the second, so a harness that reported only magnitude could not
have seen it.
"""
function score_arm(dir, arm, tdx, tdy)
    a = read_arm(dir, arm)
    a === nothing && return nothing
    (sx, sy) = ARMS_SIGNS[arm]
    dx, dy = sx .* a.arrays["dx"], sy .* a.arrays["dy"]
    ok = .!isnan.(dx) .& .!isnan.(dy)
    n = count(ok)
    n == 0 && return (; arm, n = 0, coverage = 0.0, median = NaN, p95 = NaN,
                      beyond = NaN, bias_dx = NaN, bias_dy = NaN, seconds = a.seconds)
    ex, ey = dx[ok] .- tdx[ok], dy[ok] .- tdy[ok]
    e = hypot.(ex, ey)
    return (; arm, n, coverage = n / length(ok),
            median = median(e), p95 = quantile(e, 0.95), beyond = mean(e .> TOL),
            bias_dx = median(ex), bias_dy = median(ey), seconds = a.seconds)
end

# The sign per arm, as `gate.jl` measured it.
const ARMS_SIGNS = Dict(ARMS)

# The order arms appear in every table: the two implementations under comparison first, then the
# arbiters, then the rotation arm which is not like-for-like.
const ARM_ORDER = ("jl_correlator", "py_correlator", "cv_pyrup", "cv_parabola",
                   "jl_pipeline", "py_pipeline", "jl_rotation")

const ARM_LABEL = Dict("jl_correlator" => "AutoRIFT.jl", "py_correlator" => "autoRIFT.py",
                       "cv_pyrup" => "OpenCV+pyrUp", "cv_parabola" => "OpenCV+parabola",
                       "jl_pipeline" => "AutoRIFT.jl full", "py_pipeline" => "autoRIFT.py full",
                       "jl_rotation" => "AutoRIFT.jl +rot")

"""
    stratify(dir, tdx, tdy, grad, arms) -> Vector

Each arm's error within quartiles of the local deformation rate.

Stratified by deformation and not by displacement magnitude because that is the variable the effect
tracks: a rigid chip represents a large uniform shift exactly, and fails only where the displacement
*varies* across its own footprint. Quantile edges rather than fixed ones, so every stratum is populated
whatever the case's gradient range.
"""
function stratify(dir, tdx, tdy, grad, arms)
    rows = []
    gs = sort(vec(grad))
    edges = [quantile(gs, q) for q in (0.0, 0.5, 0.8, 0.95, 1.0)]
    for k in 1:(length(edges) - 1)
        lo, hi = edges[k], edges[k + 1]
        sel = (grad .>= lo) .& (grad .<= hi)
        count(sel) < 20 && continue
        entry = Dict{String,Any}("lo" => lo, "hi" => hi, "n" => count(sel))
        for arm in arms
            a = read_arm(dir, arm)
            a === nothing && continue
            (sx, sy) = ARMS_SIGNS[arm]
            dx, dy = sx .* a.arrays["dx"], sy .* a.arrays["dy"]
            ok = sel .& .!isnan.(dx) .& .!isnan.(dy)
            count(ok) < 20 && continue
            e = hypot.(dx[ok] .- tdx[ok], dy[ok] .- tdy[ok])
            entry[arm] = Dict("median" => median(e), "beyond" => mean(e .> TOL), "n" => count(ok))
        end
        push!(rows, entry)
    end
    return rows
end

"""
    disagreement(dir, a, b, grad) -> Vector

How far two arms are from each other, by deformation stratum.

Reported alongside each arm's error against truth because it is the quantity the real-scene comparison
could measure and this one can calibrate: with no truth available, a disagreement is all that is
visible, and knowing how it maps onto actual error is what makes the real-scene number interpretable.
"""
function disagreement(dir, arm_a, arm_b, grad)
    a, b = read_arm(dir, arm_a), read_arm(dir, arm_b)
    (a === nothing || b === nothing) && return []
    (sxa, sya), (sxb, syb) = ARMS_SIGNS[arm_a], ARMS_SIGNS[arm_b]
    dxa, dya = sxa .* a.arrays["dx"], sya .* a.arrays["dy"]
    dxb, dyb = sxb .* b.arrays["dx"], syb .* b.arrays["dy"]
    rows = []
    gs = sort(vec(grad))
    edges = [quantile(gs, q) for q in (0.0, 0.5, 0.8, 0.95, 1.0)]
    for k in 1:(length(edges) - 1)
        lo, hi = edges[k], edges[k + 1]
        ok = (grad .>= lo) .& (grad .<= hi) .& .!isnan.(dxa) .& .!isnan.(dxb)
        count(ok) < 20 && continue
        d = hypot.(dxa[ok] .- dxb[ok], dya[ok] .- dyb[ok])
        push!(rows, Dict("lo" => lo, "hi" => hi, "n" => count(ok),
                         "median" => median(d), "beyond" => mean(d .> TOL)))
    end
    return rows
end

"""
    agreement(dir, arm_a, arm_b) -> Dict

How closely two arms agree with each other, independent of truth: the fraction of points where they
return the identical `Float32` and the largest difference anywhere.

Reported because "these two implementations are the same answer" and "these two are both close to
truth" are different claims, and only the first makes one of them usable as an arbiter for the other.
"""
function agreement(dir, arm_a, arm_b)
    a, b = read_arm(dir, arm_a), read_arm(dir, arm_b)
    (a === nothing || b === nothing) && return Dict{String,Any}()
    dxa, dya = a.arrays["dx"], a.arrays["dy"]
    dxb, dyb = b.arrays["dx"], b.arrays["dy"]
    ok = .!isnan.(dxa) .& .!isnan.(dxb)
    count(ok) == 0 && return Dict{String,Any}()
    same = (dxa[ok] .== dxb[ok]) .& (dya[ok] .== dyb[ok])
    worst = max(maximum(abs.(dxa[ok] .- dxb[ok])), maximum(abs.(dya[ok] .- dyb[ok])))
    return Dict("identical" => mean(same), "max_abs_diff" => jsonsafe(worst), "n" => count(ok))
end

function main()
    names = sort(readdir(SCENES))
    out = Dict{String,Any}()
    cases_out = Dict{String,Any}()

    for name in names
        dir = joinpath(SCENES, name)
        isfile(joinpath(dir, "manifest.txt")) || continue
        b = read_bundle(dir)
        tdx, tdy = b.arrays["true_dx"], b.arrays["true_dy"]
        grad = b.arrays["true_gradient"]
        arms = filter(a -> read_arm(dir, a) !== nothing, ARM_ORDER)

        scores = Dict{String,Any}()
        for arm in arms
            s = score_arm(dir, arm, tdx, tdy)
            s === nothing && continue
            scores[arm] = Dict("n" => s.n, "coverage" => s.coverage,
                               "median" => jsonsafe(s.median), "p95" => jsonsafe(s.p95),
                               "beyond_$(TOL)" => jsonsafe(s.beyond),
                               "bias_dx" => jsonsafe(s.bias_dx),
                               "bias_dy" => jsonsafe(s.bias_dy),
                               "seconds" => jsonsafe(s.seconds))
        end
        cases_out[name] = Dict(
            "case" => read_case(dir),
            "grid" => collect(size(tdx)),
            "max_deformation" => maximum(grad),
            "arms" => scores,
            "strata" => stratify(dir, tdx, tdy, grad, arms),
            "disagreement_jl_py" => disagreement(dir, "jl_correlator", "py_correlator", grad),
            "agreement_jl_cv" => agreement(dir, "jl_correlator", "cv_pyrup"),
        )
    end

    out["tolerance"] = TOL
    out["cases"] = cases_out
    open(RESULTS, "w") do io
        JSON3.pretty(io, out)
    end

    # ---- the printed tables ----
    #
    # `cv_pyrup` is included even though it tracks `jl_correlator` to the last bit at 99.99% of points
    # (see `agreement` below), because that agreement is itself the finding: it is what makes OpenCV an
    # arbiter of the comparison rather than a third opinion in it.
    println("\nMedian error against exact truth, px:\n")
    @printf("%-28s %12s %12s %12s %12s\n", "case", "AutoRIFT.jl", "autoRIFT.py",
            "CV+pyrUp", "CV+parabola")
    println("-"^80)
    for name in sort(collect(keys(cases_out)))
        c = cases_out[name]
        vals = map(("jl_correlator", "py_correlator", "cv_pyrup", "cv_parabola")) do arm
            haskey(c["arms"], arm) && c["arms"][arm]["median"] !== nothing ?
                @sprintf("%.4f", c["arms"][arm]["median"]) : "-"
        end
        @printf("%-28s %12s %12s %12s %12s\n", name, vals...)
    end

    # Per-axis bias, which a magnitude cannot show. The real scene's disagreement was a *signed* offset
    # concentrated in one axis, so an arm with zero median error and a nonzero bias is a different
    # finding from one with the same error spread symmetrically about truth.
    println("\n\nPer-axis bias against truth, px (dx / dy):\n")
    @printf("%-28s %18s %18s %18s\n", "case", "AutoRIFT.jl", "autoRIFT.py", "CV+parabola")
    println("-"^86)
    for name in sort(collect(keys(cases_out)))
        c = cases_out[name]
        vals = map(("jl_correlator", "py_correlator", "cv_parabola")) do arm
            haskey(c["arms"], arm) && c["arms"][arm]["bias_dx"] !== nothing ?
                @sprintf("%+.4f /%+.4f", c["arms"][arm]["bias_dx"], c["arms"][arm]["bias_dy"]) : "-"
        end
        @printf("%-28s %18s %18s %18s\n", name, vals...)
    end

    println("\n\nBy deformation stratum, base shear case (median error px / % beyond $(TOL) px):")
    c = cases_out["shear_a10_c16"]
    @printf("\n%18s %6s %16s %16s %16s\n", "deformation px/px", "n",
            "AutoRIFT.jl", "autoRIFT.py", "OpenCV+pyrUp")
    for s in c["strata"]
        cells = map(("jl_correlator", "py_correlator", "cv_pyrup")) do arm
            haskey(s, arm) ? @sprintf("%.4f / %4.1f%%", s[arm]["median"], 100 * s[arm]["beyond"]) : "-"
        end
        @printf("%8.4f-%8.4f %6d %16s %16s %16s\n", s["lo"], s["hi"], s["n"], cells...)
    end

    # The uniform-deformation modes, where every point carries the same deformation rate by
    # construction. These separate "the chip sees deformation" from "the deformation varies across the
    # scene": there is no gradient to stratify by and no ambiguity about which points are hard, so a
    # single number per arm is the whole result.
    println("\n\nUniform-deformation modes (one deformation rate everywhere, median error px):\n")
    @printf("%-20s %10s %12s %12s %12s\n", "case", "rate", "AutoRIFT.jl", "autoRIFT.py",
            "CV+parabola")
    println("-"^70)
    for name in ("divergence", "rotation_0p5deg", "rotation_2deg")
        haskey(cases_out, name) || continue
        cc = cases_out[name]
        vals = map(("jl_correlator", "py_correlator", "cv_parabola")) do arm
            haskey(cc["arms"], arm) && cc["arms"][arm]["median"] !== nothing ?
                @sprintf("%.4f", cc["arms"][arm]["median"]) : "-"
        end
        @printf("%-20s %10.5f %12s %12s %12s\n", name, cc["max_deformation"], vals...)
    end

    println("\n\nAutoRIFT.jl against autoRIFT.py, by deformation (the quantity the real scene shows):")
    for s in c["disagreement_jl_py"]
        @printf("  %8.4f-%8.4f  n %5d  median %.4f px  beyond %.1f%%\n",
                s["lo"], s["hi"], s["n"], s["median"], 100 * s["beyond"])
    end

    # The arbiter check. Two independently written implementations returning the identical Float32 is
    # not a near-agreement to be reported as a small error; it means one can stand in for the other.
    println("\n\nAutoRIFT.jl against raw OpenCV+pyrUp, over every case:")
    ids = Float64[]
    worst = 0.0
    for (nm, cc) in cases_out
        ag = cc["agreement_jl_cv"]
        isempty(ag) && continue
        push!(ids, ag["identical"])
        ag["max_abs_diff"] === nothing || (worst = max(worst, ag["max_abs_diff"]))
    end
    if !isempty(ids)
        @printf("  bit-identical at %.2f%%-%.2f%% of points (%d cases); largest difference %.4f px\n",
                100 * minimum(ids), 100 * maximum(ids), length(ids), worst)
    end

    # Coverage failures, stated rather than left as a gap in the tables. An arm that answered nothing is
    # not an arm that scored badly, and a table of medians cannot distinguish the two.
    println("\n\nArms that resolved less than half the grid:")
    any_gap = false
    for name in sort(collect(keys(cases_out)))
        for (arm, s) in cases_out[name]["arms"]
            s["coverage"] >= 0.5 && continue
            any_gap = true
            @printf("  %-28s %-14s n %5d  coverage %.1f%%\n",
                    name, arm, s["n"], 100 * s["coverage"])
        end
    end
    any_gap || println("  none")

    @printf("\nwrote %s (%d cases)\n", RESULTS, length(cases_out))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
