# The gate every arm must pass before its other numbers mean anything, and the place each arm's sign
# convention is established.
#
#   julia --project=tools/synth tools/synth/gate.jl
#
# Two things happen here, and the order matters.
#
# First, **signs are measured, not asserted**. Each arm is scored against the known field under all four
# sign combinations and the best is reported. An arm whose best combination is not the one the scorer
# assumes is a defect in the harness, and asserting the convention instead of measuring it is what
# produced two sign errors in earlier work on this comparison.
#
# Second, **exactness on a pure translation**. A rigid chip represents a uniform shift perfectly, so
# there is no modelling error to hide behind: an arm that cannot recover an integer translation to
# 0.000 px has a windowing or convention fault, and its accuracy on a deforming scene would be
# measuring that fault rather than the deformation. The sub-pixel case is held to the upsampling step
# rather than to zero, since 1/16 px is the quantization floor of the estimators themselves.

using Printf, Statistics

include(joinpath(@__DIR__, "bundle.jl"))

const SCENES = joinpath(@__DIR__, "scenes")

# Every arm, and the sign convention that puts it in truth space -- measured by this script, not
# assumed, and recorded here so the scorer applies the same one.
#
# `(-1, -1)` for the Julia and OpenCV arms: both report where the secondary chip sits within the
# reference window, which is the offset from secondary back to reference and so the negative of the
# feature's motion. `(+1, +1)` for the reference arms: `arImgDisp_s` and `runAutorift` already return
# feature motion, and `run_python.py` has additionally undone their cartesian y flip, so both axes are
# in truth space already.
const ARMS = ("jl_correlator" => (-1, -1), "jl_pipeline" => (-1, -1),
              "jl_rotation" => (-1, -1),
              "py_correlator" => (1, 1), "py_pipeline" => (1, 1),
              "cv_pyrup" => (-1, -1), "cv_parabola" => (-1, -1))

const SIGNS = ((1, 1), (-1, -1), (1, -1), (-1, 1))

# What each arm is held to on a pure translation, as a multiple of the case's own tolerance.
#
# `cv_parabola` is allowed more because a parabola through three samples of a discrete correlation peak
# cannot be exact even when the peak sits on a sample: the fit is biased toward the sample it is centred
# on. Measured at 0.017 px on an integer shift and 0.061 px on a sub-pixel one -- small, systematic, and
# the whole reason this arm is reported separately from `cv_pyrup`. Holding it to the cascade arms'
# tolerance would fail it for the property it exists to demonstrate.
const GATE_SLACK = Dict("cv_parabola" => 0.10)

"""
    best_signs(dx, dy, tdx, tdy) -> ((sx, sy), err, second_best_err)

The sign combination that matches truth, its median error, and the runner-up's.

The runner-up is returned because it is what makes the result trustworthy: if the best and second-best
are close, the field does not distinguish them and the convention has not actually been established.
On a translation with a nonzero shift in both axes they differ by twice the shift, which is decisive.
"""
function best_signs(dx, dy, tdx, tdy)
    ok = .!isnan.(dx) .& .!isnan.(dy)
    count(ok) == 0 && return (0, 0), NaN, NaN
    errs = map(SIGNS) do (sx, sy)
        median(abs.(sx .* dx[ok] .- tdx[ok]) .+ abs.(sy .* dy[ok] .- tdy[ok]))
    end
    order = sortperm(collect(errs))
    return SIGNS[order[1]], errs[order[1]], errs[order[2]]
end

"""
    radial_error(dx, dy, tdx, tdy, signs) -> Vector{Float64}

Per-point distance from truth under a given sign convention.
"""
function radial_error(dx, dy, tdx, tdy, (sx, sy))
    ok = .!isnan.(dx) .& .!isnan.(dy)
    return hypot.(sx .* dx[ok] .- tdx[ok], sy .* dy[ok] .- tdy[ok])
end

function main()
    # The translation cases: the only ones where a rigid chip is an exact model, so the only ones that
    # can gate the harness. `divergence` and the shear cases deform the texture inside every chip, and
    # an error there is the thing being measured rather than a fault.
    gates = ("translate_integer" => 0.0, "translate_subpixel" => 1 / 16)
    failures = String[]

    for (case, tol) in gates
        dir = joinpath(SCENES, case)
        isdir(dir) || (@printf("%-22s MISSING -- run scenes.jl\n", case); continue)
        b = read_bundle(dir)
        tdx, tdy = b.arrays["true_dx"], b.arrays["true_dy"]
        @printf("\n=== %s   true (dx, dy) = (%+.3f, %+.3f)   tolerance %.4f px\n",
                case, tdx[1], tdy[1], tol)

        for (arm, expected) in ARMS
            a = read_arm(dir, arm)
            if a === nothing
                @printf("  %-14s not run\n", arm)
                continue
            end
            dx, dy = a.arrays["dx"], a.arrays["dy"]
            got, err, runner = best_signs(dx, dy, tdx, tdy)
            e = radial_error(dx, dy, tdx, tdy, expected)
            med, p95 = median(e), quantile(e, 0.95)
            agree = got == expected
            limit = tol + get(GATE_SLACK, arm, 0.0) + 1e-6
            pass = agree && med <= limit
            pass || push!(failures, "$case/$arm")
            @printf("  %-14s signs %s%s  median %.4f  p95 %.4f  (limit %.4f)  %s\n",
                    arm, string(got), agree ? "" : " != $(expected) !!",
                    med, p95, limit, pass ? "ok" : "FAIL")
            # A convention is only established if the runner-up is clearly worse.
            if agree && runner < 4 * err + 0.5
                @printf("  %-14s   sign not decisive: best %.4f vs next %.4f\n", "", err, runner)
            end
        end
    end

    println()
    if isempty(failures)
        println("gate passed: every arm recovers a pure translation within tolerance, ",
                "under the sign convention the scorer assumes")
    else
        # Fail loudly rather than letting the sweep run: every downstream number from a failing arm
        # would be measuring the fault instead of the comparison.
        error("gate FAILED for: ", join(failures, ", "))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
