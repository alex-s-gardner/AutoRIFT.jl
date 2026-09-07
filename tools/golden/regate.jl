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
        const TD = joinpath("$(ROOT)", "test")
        include(joinpath(TD, "utils.jl"))
        @testset "gate" begin
            include(joinpath(TD, "fixtures_test.jl"))
            include(joinpath(TD, "window.jl"))
            include(joinpath(TD, "multichip.jl"))
        end
        """
        isdir(joinpath(ROOT, "test", "fixtures", "colfilt")) ||
            return (:skipped, "colfilt fixtures absent; regenerate with gen_fixtures.py")
        return capture_run(`julia --project=$ROOT -e "using TestEnv; TestEnv.activate(); $script"`) do t
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
        const TD = joinpath("$(ROOT)", "test")
        include(joinpath(TD, "utils.jl"))
        @testset "realdata" begin include(joinpath(TD, "realdata.jl")) end
        """
        return capture_run(`julia --project=$ROOT -t 8 -e "using TestEnv; TestEnv.activate(); $script"`) do t
            occursin("Test Summary", t) || return (:red, last_line(t))
            occursin(r"Fail|Error", t) && return (:red, "assertions failed: " * last_line(t))
            return (:green, "all assertions pass")
        end
    end),

    Gate("3.x", "the stage ladder on the golden Landsat case", false, function ()
        cmd = `julia --project=$(@__DIR__) -t 8 $(joinpath(@__DIR__, "stages.jl")) LC08_L1TP_009011 --all`
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
