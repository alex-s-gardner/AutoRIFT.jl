# Re-run every gate the ledger records as green, in order, and report which still hold.
#
#   julia --project=tools/golden tools/golden/regate.jl              # the cheap gates
#   julia --project=tools/golden tools/golden/regate.jl --all        # including the slow ones
#
# **What this is for.** A ladder whose lower rungs are not re-checked is a ladder that slides: a fix
# justified against one measurement can undo another, and the five `src/` commits that prompted
# `GATES.md` row 0.2 are the case in point — each was justified against a golden comparison and none
# was checked against the pre-existing benchmark. This runs them all so that check is one command
# rather than a decision.
#
# Gates are ordered cheapest first, and by dependency: the fixture comparisons need no reference
# process, the `tools/ab` stages need the `arift-ref` environment, and the golden endpoints need a
# capture on disk. A gate whose inputs are absent is reported **skipped** rather than passed, because a
# gate that silently reports green when it did not run is worse than one that fails.

using Printf

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const LEDGER = joinpath(@__DIR__, "GATES.md")

struct Gate
    id::String
    what::String
    slow::Bool
    # Returns `(state, detail)` where state is `:green`, `:red` or `:skipped`.
    run::Function
end

# Run a command, capture its output, and hand the text to `check`. Output is kept rather than streamed
# so a gate's verdict can be read out of it — every gate here reports its own numbers, and parsing them
# is what makes the comparison against the ledger automatic instead of a reading exercise.
#
# `check` comes first so `do`-block syntax works: `capture_run(cmd) do t ... end` passes the block as
# the *first* argument, which a `(cmd, check)` signature cannot accept.
function capture_run(check::Function, cmd::Cmd)
    io = IOBuffer()
    ok = try
        run(pipeline(cmd; stdout = io, stderr = io))
        true
    catch
        false
    end
    text = String(take!(io))
    ok || return (:red, "command failed: " * last_line(text))
    return check(text)
end

last_line(text) = begin
    ls = filter(!isempty, strip.(split(text, '\n')))
    isempty(ls) ? "(no output)" : ls[end]
end

# The first capture group of `re` in `text`, as a Float64, or `nothing`.
function grab(text, re)
    m = match(re, text)
    m === nothing ? nothing : tryparse(Float64, m.captures[1])
end

const GATES = Gate[
    Gate("2.x", "colfilt, bwareaopen and the window reductions", false, function ()
        script = """
        using AutoRIFT, Test, Random, Statistics, JSON3
        const TD = joinpath(raw"$(ROOT)", "test")
        include(joinpath(TD, "utils.jl"))
        @testset "gate" begin
            include(joinpath(TD, "fixtures_test.jl"))
            include(joinpath(TD, "window.jl"))
            include(joinpath(TD, "multichip.jl"))
        end
        """
        isdir(joinpath(ROOT, "test", "fixtures", "colfilt")) ||
            return (:skipped, "colfilt fixtures absent; regenerate with gen_fixtures.py")
        # `--project=tools/golden` and not `TestEnv`: TestEnv re-resolves the package's test dependencies,
        # and `FastGeoProjections` on `main` has moved past this project's `[compat]` bound, so the
        # resolution fails before any test runs. That failure is present on `main` too and is not what
        # this gate is measuring. The golden project has AutoRIFT dev'd with a manifest that resolves,
        # and none of the files below touch the offending dependency.
        return capture_run(`julia --project=$(@__DIR__) -e $script`) do t
            occursin("Test Summary", t) || return (:red, last_line(t))
            occursin(r"Fail|Error", t) && return (:red, "assertions failed: " * last_line(t))
            return (:green, "all assertions pass")
        end
    end),

    Gate("0.1", "the correlator alone is bit-identical (tools/ab stage 1)", false, function ()
        d = joinpath(ROOT, "tools", "ab")
        run(pipeline(`julia --project=$d $(joinpath(d, "stage1_julia.jl")) 1024 32 20`;
                     stdout = devnull, stderr = devnull))
        run(pipeline(`micromamba run -n arift-ref python $(joinpath(d, "stage1_python.py"))`;
                     stdout = devnull, stderr = devnull))
        return capture_run(`julia --project=$d $(joinpath(d, "compare.jl")) regate1`) do t
            ex = grab(t, r"\|ddx\|\s+exact\s+([\d.]+)%")
            ex === nothing && return (:red, "could not read the exact fraction: " * last_line(t))
            # 100% and nothing less: the float correlator is bit-identical, and any drop off it is a
            # regression rather than a tolerance question.
            return (ex >= 100.0 ? :green : :red, @sprintf("exact %.1f%% (expected 100.0%%)", ex))
        end
    end),

    Gate("0.2", "the whole pipeline against the reference (tools/ab stage 2)", true, function ()
        d = joinpath(ROOT, "tools", "ab")
        # Both halves in one invocation. A bundle whose two sides were written by different code is
        # the fault `GATES.md` row 0.2 records, and running them together is what forecloses it.
        run(pipeline(`julia --project=$d -t 8 $(joinpath(d, "stage2_julia.jl")) 3072 16 64 20`;
                     stdout = devnull, stderr = devnull))
        run(pipeline(`micromamba run -n arift-ref python $(joinpath(d, "stage2_python.py"))`;
                     stdout = devnull, stderr = devnull))
        return capture_run(`julia --project=$d $(joinpath(d, "compare2.jl")) regate2`) do t
            ex = grab(t, r"\|ddx\|\s+exact\s+([\d.]+)%")
            wi = grab(t, r"\|ddx\|\s+exact\s+[\d.]+%\s+within step\s+([\d.]+)%")
            (ex === nothing || wi === nothing) &&
                return (:red, "could not read the statistics: " * last_line(t))
            # The recorded floor, less a tenth of a point for the rounding the report prints at.
            return (ex >= 81.7 && wi >= 98.6 ? :green : :red,
                    @sprintf("exact %.1f%% (floor 81.8%%), within step %.1f%% (floor 98.7%%)", ex, wi))
        end
    end),

    Gate("0.3", "the ITS_LIVE granule comparison (test/realdata.jl)", true, function ()
        cache = get(ENV, "AUTORIFT_TESTDATA", expanduser("~/data/autorift/tests"))
        isdir(joinpath(cache, "landsat")) ||
            return (:skipped, "real-data cache absent at $cache")
        script = """
        using AutoRIFT, Test, Random, Statistics
        using AutoRIFT: params
        const TD = joinpath(raw"$(ROOT)", "test")
        include(joinpath(TD, "utils.jl"))
        @testset "realdata" begin include(joinpath(TD, "realdata.jl")) end
        """
        return capture_run(`julia --project=$(@__DIR__) -t 8 -e $script`) do t
            occursin("Test Summary", t) || return (:red, last_line(t))
            occursin(r"Fail|Error", t) && return (:red, "assertions failed: " * last_line(t))
            return (:green, "all assertions pass")
        end
    end),

    Gate("3.opt", "the stage ladder on every captured optical case", true, function ()
        # **All twelve, not one.** A rung is only as good as the case classes it has met: the `filtDisp`
        # index bug and the sign-selector bug were both invisible on the five `hps` cases and surfaced only
        # on an L7 pair whose base level is skipped. So the no-slide check runs the ladder over every case
        # with a capture on disk rather than over a representative one.
        cases = ["LC08_L1TP_009011", "LC08_L1TP_062018", "LC09_L1GT_215109",
                 "S2A_MSIL1C_20200626", "S2B_MSIL1C_20200612",
                 "LE07_L1TP_061018_20120428", "LE07_L1TP_061018_20130314",
                 "LE07_L1TP_063018_20040810", "LC08_L1TP_060018_20130330_20200912_02_T1_X_LE07",
                 "LT05_L1TP_060018_19851028", "LT04_L1TP_063018_19880611",
                 "LT05_L1GS_001013_19920425"]
        results = String[]
        red = 0
        skipped = 0
        for c in cases
            cmd = `julia --project=$(@__DIR__) -t 8 $(joinpath(@__DIR__, "stages.jl")) $c --run 200 --all`
            state, detail = capture_run(cmd) do t
                occursin("no stage trace", t) && return (:skipped, "no capture")
                m = match(r"(\d+) rungs?, (\d+) green, (\d+) red", t)
                m === nothing && return (:red, last_line(t))
                (parse(Int, m.captures[3]) == 0 ? :green : :red, m.match)
            end
            state === :red && (red += 1)
            state === :skipped && (skipped += 1)
            push!(results, "$(first(c, 22)) $(state === :green ? "ok" : String(state))")
        end
        return (red == 0 ? :green : :red,
                "$(length(cases) - red - skipped)/$(length(cases)) green" *
                (skipped > 0 ? ", $skipped without a capture" : "") *
                (red > 0 ? ": " * join(filter(r -> !endswith(r, "ok"), results), ", ") : ""))
    end),

    Gate("3.rdr", "the endpoint on the Sentinel-1 SLC pair", true, function ()
        # **Its own gate, not folded into `3.opt`.** That gate's value is that all twelve of its cases are
        # optical, so a red one names the class that broke; a mixed list would hide which.
        #
        # Gated on bias and correlation rather than on rungs, for two reasons. The ladder needs
        # `CAPTURE_STAGES=1` and a radar capture is 2h15m, so no stage trace is on disk. And `exact` is 0
        # by construction here: the base level (64x16) runs a coarse pass and a `filtDisp` but no fine
        # pass, so every reported point comes from the 128x32 level bicubic-resized rather than quantized.
        #
        # The `dy` sign is part of the assertion. `optflag == 0` pre-flips `Dy0` (`testautoRIFT.py:402`),
        # and `correlator.jl` picks the sign by correlation rather than asserting it — so a regression that
        # silently reintroduced the flip would show up here as `+` and a collapsed correlation.
        cmd = `julia --project=$(@__DIR__) -t 8 $(joinpath(@__DIR__, "correlator.jl"))
               S1A_IW_SLC__1SSH_20151120T080202 --run 200`
        return capture_run(cmd) do t
            occursin("no call1.json", t) && return (:skipped, "no capture on disk")
            rows = collect(eachmatch(r"^(dx|dy)\s+([+-])\s+(\d+)\s+\d+\s+\d+\s+[\d.]+%\s+\d+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([+-][\d.]+)"m, t))
            length(rows) == 2 || return (:red, "could not read both axes: " * last_line(t))
            # **The core bias, not the mean over everything.** This pair correlates at a median of
            # 0.148 — SAR speckle over a 24-day repeat — so 187 of 60,611 points land on the other side
            # of a nearly flat peak surface, two-sided, by up to 45.9 px. They drag the plain mean to
            # +0.027 px while the 78% agreeing within a pixel sit at −0.0008. Gating on the plain mean
            # would set a threshold around the tail's cancellation, which is noise; gating on the core
            # catches the systematic offset the threshold is for. The tail is bounded separately below.
            bm = match(r"bias core: dx ([+-]?[\d.e-]+), dy ([+-]?[\d.e-]+)", t)
            bm === nothing && return (:red, "no bias core line: " * last_line(t))
            bx, by = abs(parse(Float64, bm.captures[1])), abs(parse(Float64, bm.captures[2]))
            tm = match(r"tail >10px: dx (\d+), dy (\d+) of (\d+)", t)
            tm === nothing && return (:red, "no tail line: " * last_line(t))
            tailx = parse(Int, tm.captures[1])
            corr = Dict(r.captures[1] => parse(Float64, r.captures[4]) for r in rows)
            sign = Dict(r.captures[1] => r.captures[2] for r in rows)
            both = parse(Int, first(rows).captures[3])
            # Floors are the measured values less a margin, so noise does not flip the gate while a real
            # regression does. Bias 0.035 px is the optical figure; correlation and coverage are this
            # pair's own, recorded in GATES.md.
            fails = String[]
            # A systematic offset over the agreeing population is held an order of magnitude tighter
            # than the optical 0.035 px, because that is what the measurement supports: 0.0008 and
            # 0.0025 px here.
            bx <= 0.005 || push!(fails, "dx core bias $bx > 0.005")
            by <= 0.005 || push!(fails, "dy core bias $by > 0.005")
            corr["dx"] >= 0.93 || push!(fails, "dx corr $(corr["dx"]) < 0.93")
            corr["dy"] >= 0.87 || push!(fails, "dy corr $(corr["dy"]) < 0.87")
            sign["dy"] == "-" || push!(fails, "dy sign $(sign["dy"]), expected -")
            both >= 55_000 || push!(fails, "both $both < 55000")
            # The tail is bounded rather than ignored: it cancels today, and a tail that grew would
            # otherwise hide behind a core bias that stayed small.
            tailx <= 400 || push!(fails, "dx tail $tailx > 400 points beyond 10 px")
            isempty(fails) ||  return (:red, join(fails, "; "))
            return (:green, @sprintf("core bias %.4f/%.4f, corr %+.3f/%+.3f, dy sign %s, both %d, tail %d",
                                     bx, by, corr["dx"], corr["dy"], sign["dy"], both, tailx))
        end
    end),

    Gate("3.x", "the stage ladder on the golden Landsat case", false, function ()
        # Run 200 holds the base-level trace; 201..203 hold the coarser levels. Named explicitly because
        # the default run has no stage trace, and a gate that skips is a gate that never runs.
        cmd = `julia --project=$(@__DIR__) -t 8 $(joinpath(@__DIR__, "stages.jl"))
               LC08_L1TP_009011 --run 200 --all`
        return capture_run(cmd) do t
            occursin("no stage trace", t) &&
                return (:skipped, "no stage trace on disk; re-capture with CAPTURE_STAGES=1")
            m = match(r"(\d+) rungs?, (\d+) green, (\d+) red", t)
            m === nothing && return (:red, "could not read the rung count: " * last_line(t))
            red = parse(Int, m.captures[3])
            return (red == 0 ? :green : :red, m.match)
        end
    end),
]

function main(args)
    all = "--all" in args
    @printf("%-6s %-52s %-9s %s\n", "gate", "what", "state", "detail")
    states = Symbol[]
    for g in GATES
        if g.slow && !all
            @printf("%-6s %-52s %-9s %s\n", g.id, g.what, "deferred", "pass --all to run it")
            continue
        end
        state, detail = try
            g.run()
        catch e
            (:red, sprint(showerror, e))
        end
        push!(states, state)
        @printf("%-6s %-52s %-9s %s\n", g.id, g.what, uppercase(String(state)), detail)
    end
    red = count(==(:red), states)
    skipped = count(==(:skipped), states)
    @printf("\n%d ran, %d green, %d red, %d skipped\n",
            length(states), count(==(:green), states), red, skipped)
    red == 0 || @printf("\nA red gate is a regression. `%s` records what each one measured.\n", LEDGER)
    red == 0 || exit(1)
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
