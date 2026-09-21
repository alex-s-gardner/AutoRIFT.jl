# Re-run every gate the ledger records as green, in order, and report which still hold.
#
#   julia --project=tools/golden tools/golden/regate.jl              # the cheap gates
#   julia --project=tools/golden tools/golden/regate.jl --all        # including the slow ones
#
# The verdict table goes to stdout and per-case progress to stderr, so `regate.jl --all >table.txt`
# leaves the progress on the terminal. `--all` runs for tens of minutes; keep the streams separate or
# redirect both and watch the file.
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

# Live progress on stderr, one flushed line per step, while the verdict table accumulates on stdout.
# Two streams because they answer different questions: the table is the result, and this is only
# evidence that a run lasting tens of minutes is still moving. stdout is block-buffered under a
# redirect, so a single stream into a log file shows nothing at all until the run exits.
function progress(msg::AbstractString)
    print(stderr, "  … ", msg, "\n")
    flush(stderr)
    return nothing
end

# `correlator.jl`'s report, as `(; bx, by, corr, sgn, both, tailx)` — or `(state, detail)` when there is
# nothing to read, which a caller distinguishes with `isa Tuple`.
#
# Shared because both endpoint gates read the same report and each threshold is the *caller's*: this
# parses, and asserts nothing. The regexes are the coupling to `correlator.jl`'s output format, so they
# live once — a column added to the axis table otherwise has to be found in every gate that reads it.
#
# `tailx` is `nothing` when the report carries no tail line. `3.rdr` requires one and treats its absence
# as red; a thinned run has no meaningful tail to bound, so `3.nisar` ignores it.
function parse_endpoint(text)
    occursin("no call1.json", text) && return (:skipped, "no capture on disk")
    # The median, p99 and max columns are `%g`, which switches to exponent form below 1e-4. A
    # fixed-point pattern there reds a case for agreeing *too well*: a median of `6.896e-05` fails to
    # match and the row count falls short.
    num = raw"[\d.]+(?:[eE][+-]?\d+)?"
    row = Regex(raw"^(dx|dy)\s+([+-])\s+(\d+)\s+\d+\s+\d+\s+[\d.]+%\s+\d+\s+" *
                num * raw"\s+" * num * raw"\s+" * num * raw"\s+([+-][\d.]+)", "m")
    rows = collect(eachmatch(row, text))
    length(rows) == 2 || return (:red, "could not read both axes: " * last_line(text))
    # **The core bias, not the mean over everything.** A pair correlating at a median of 0.148 — SAR
    # speckle over a 24-day repeat — puts a few hundred of its points on the other side of a nearly flat
    # peak surface, two-sided, by tens of pixels. On `20151120` that drags the plain mean to +0.027 px
    # while the 78% agreeing within a pixel sit at −0.0008. Gating the plain mean would set a threshold
    # around the tail's cancellation, which is noise; the core catches the systematic offset a threshold
    # is for, and the tail is bounded separately by the caller.
    bm = match(r"bias core: dx ([+-]?[\d.e-]+), dy ([+-]?[\d.e-]+)", text)
    bm === nothing && return (:red, "no bias core line: " * last_line(text))
    tm = match(r"tail >10px: dx (\d+), dy (\d+) of (\d+)", text)
    return (; bx = abs(parse(Float64, bm.captures[1])), by = abs(parse(Float64, bm.captures[2])),
            corr = Dict(r.captures[1] => parse(Float64, r.captures[4]) for r in rows),
            sgn = Dict(r.captures[1] => r.captures[2] for r in rows),
            both = parse(Int, first(rows).captures[3]),
            tailx = tm === nothing ? nothing : parse(Int, tm.captures[1]))
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
        for (i, c) in enumerate(cases)
            progress(@sprintf("3.opt %d/%d %s", i, length(cases), c))
            cmd = `julia --project=$(@__DIR__) -t 8 $(joinpath(@__DIR__, "stages.jl")) $c --run 200 --all`
            state, detail = capture_run(cmd) do t
                occursin("no stage trace", t) && return (:skipped, "no capture")
                m = match(r"(\d+) rungs?, (\d+) green, (\d+) red", t)
                m === nothing && return (:red, last_line(t))
                (parse(Int, m.captures[3]) == 0 ? :green : :red, m.match)
            end
            state === :red && (red += 1)
            state === :skipped && (skipped += 1)
            progress(@sprintf("3.opt %d/%d %s → %s %s", i, length(cases), c, String(state), detail))
            push!(results, "$(first(c, 22)) $(state === :green ? "ok" : String(state))")
        end
        return (red == 0 ? :green : :red,
                "$(length(cases) - red - skipped)/$(length(cases)) green" *
                (skipped > 0 ? ", $skipped without a capture" : "") *
                (red > 0 ? ": " * join(filter(r -> !endswith(r, "ok"), results), ", ") : ""))
    end),

    Gate("3.rdr", "the endpoint on every captured radar case", true, function ()
        # **All eight, not one.** Same reasoning as `3.opt`: a gate is only as good as the case classes
        # it has met. These eight span two drivers — five S1-SLC through `process_slc` and three
        # S1-BURST through `process_burst`, the latter mosaicking 7, 10 and 24 bursts before the
        # correlator — and two pyramid shapes, since `20151120` skips its base level's fine pass where
        # the others run all four levels.
        #
        # **Its own gate, not folded into `3.opt`.** That gate's value is that all twelve of its cases
        # are optical, so a red one names the class that broke; a mixed list would hide which.
        #
        # Gated on the core bias, correlation, the measured `dy` sign and coverage rather than on rungs.
        # The ladder needs `CAPTURE_STAGES=1` and a radar capture costs 12-50 minutes, so no stage trace
        # is on disk. `exact` is not asserted: it spans 32% to 82% across the eight with no expected
        # value to hold any of them to.
        #
        # The `dy` sign is part of the assertion. `optflag == 0` pre-flips `Dy0`
        # (`testautoRIFT.py:402`), and `correlator.jl` picks the sign by correlation rather than
        # asserting it, so a regression that silently reintroduced the flip shows up as `+` and a
        # collapsed correlation.
        cases = ["S1A_IW_SLC__1SSH_20150828T162412", "S1A_IW_SLC__1SSH_20151120T080202",
                 "S1A_IW_SLC__1SSH_20170221T204710", "S1B_IW_SLC__1SDH_20180809T204617",
                 "S1C_IW_SLC__1SDV_20250416T010214", "S1C_IW_SLC__1SSV_20250416T010159",
                 "S1A_IW_SLC__1SSV_20240618T025533", "S1A_IW_SLC__1SSV_20240618T025528"]
        results = String[]
        red = 0
        skipped = 0
        for (i, c) in enumerate(cases)
            progress(@sprintf("3.rdr %d/%d %s", i, length(cases), c))
            cmd = `julia --project=$(@__DIR__) -t 8 $(joinpath(@__DIR__, "correlator.jl")) $c --run 200`
            state, detail = capture_run(cmd) do t
                p = parse_endpoint(t)
                p isa Tuple && return p
                (; bx, by, corr, sgn, both, tailx) = p
                tailx === nothing && return (:red, "no tail line: " * last_line(t))
                fails = String[]
                # A systematic offset over the agreeing population is held an order of magnitude
                # tighter than the optical 0.035 px. The bound is drawn from a placement rule
                # `_cell_means` has since replaced, under which six of the eight sat within 0.0072 px on
                # both axes.
                #
                # **Five cases exceed it and are expected red**: `20170221T204710` at -0.0237,
                # `20180809T204617` at -0.0383, `20250416T010214` at -0.0178, `20250416T010159` at
                # -0.0761/-0.0137 and `20240618T025533` at -0.0225. `GATES.md` records the measurement
                # and why the bound is not the thing to widen — all eight gained coverage, exact match
                # and correlation over the same change, so the `dx` bias is the one statistic the
                # deliberate nodata-fill averaging of `CORRECTNESS.md` item 2 dominates.
                bx <= 0.010 || push!(fails, "dx core bias $bx > 0.010")
                by <= 0.010 || push!(fails, "dy core bias $by > 0.010")
                # The floor is the weakest measured case less a margin. `20250416T010159` is the lowest
                # of the eight at 0.980 and 0.948, where the rest reach 0.988-0.999.
                corr["dx"] >= 0.78 || push!(fails, "dx corr $(corr["dx"]) < 0.78")
                corr["dy"] >= 0.78 || push!(fails, "dy corr $(corr["dy"]) < 0.78")
                sgn["dy"] == "-" || push!(fails, "dy sign $(sgn["dy"]), expected -")
                # The tail is bounded rather than ignored: it cancels today, and a tail that grew would
                # otherwise hide behind a core bias that stayed small. Three of the eight report zero and
                # `20170221T204710` is the largest at 10, two orders inside the bound.
                tailx <= 400 || push!(fails, "dx tail $tailx > 400 beyond 10 px")
                isempty(fails) || return (:red, join(fails, "; "))
                return (:green, @sprintf("core %.4f/%.4f corr %+.3f/%+.3f both %d tail %d",
                                         bx, by, corr["dx"], corr["dy"], both, tailx))
            end
            state === :red && (red += 1)
            state === :skipped && (skipped += 1)
            progress(@sprintf("3.rdr %d/%d %s → %s %s", i, length(cases), c, String(state), detail))
            push!(results, "$(first(c, 24)) $(state === :green ? "ok" : String(state))")
        end
        return (red == 0 ? :green : :red,
                "$(length(cases) - red - skipped)/$(length(cases)) green" *
                (skipped > 0 ? ", $skipped without a capture" : "") *
                (red > 0 ? ": " * join(filter(r -> !endswith(r, "ok"), results), ", ") : ""))
    end),

    Gate("3.nisar", "the endpoint on both NISAR cases, thinned", false, function ()
        # **Thinned, because the whole grid does not fit a gate.** `correlator.jl` is untiled and these
        # are whole-scene comparisons: 1.8 M searchable points on a 2.9 Gpx pair, peaking at 55-81 GiB
        # against this machine's 96, so the two cannot even run concurrently. `--stride 4` searches 128-px
        # tiles on a 512-px lattice — a sixteenth of the grid, ~1 minute per case — which is what puts
        # NISAR in the ladder at all. The whole-grid figures are a measurement in `GATES.md`, not a gate.
        #
        # **The two cases run one after another.** Thinning cuts the points searched, not the resident
        # imagery: the pair, the full-shape output arrays and the FFT workspaces are sized by the scene and
        # the search radius, so a thinned run's peak is not a sixteenth of a whole-grid one and is
        # unmeasured. Overlapping them risks an OOM kill that loses both, against a saving of about a
        # minute.
        #
        # **Thresholds are calibrated to the thinned run, not to the whole-grid figures**, because
        # thinning changes the answers rather than only sampling them. AutoRIFT.jl sees the sparse grid
        # while the reference's `Dx`/`Dy` come from a capture taken over the full one, so the two resolve
        # different pyramid levels: a level's coarse grid is the point grid decimated by 1, 2, 4, 8, and a
        # thinned one can fall below the width its filter needs, at which point the level silently produces
        # nothing (`tools/golden/README.md`). `exact` is the statistic this destroys — L1 reaches 73.90% on
        # the whole grid against 17.71% here and 0.00% at a 512-px tile, and L2 72.20% whole against 30.53%
        # here — so it is *not* asserted on either case.
        #
        # What is asserted is what proved stable across three tilings: correlation and `bias_core`. Both
        # are reproducible bit-for-bit at a given tiling, so a threshold is meaningful; the bound is drawn
        # from the thinned measurement and is not comparable to the 0.010 px `3.rdr` holds Sentinel-1 to.
        #
        # **Both cases exceed their `dx` bound and are expected red**, L1 at -0.7279 and L2 at
        # -0.4359/-0.5787. `_cell_means` places a coarse node at its cell's own mean, so a thinned grid
        # places nodes the whole-grid reference never placed and this run is no longer a proxy for the
        # whole one — over the same change whole-grid L1 `exact` improved. `GATES.md` records the
        # measurement and both ways of making these bounds informative again.
        # Named rather than positional: each bound is the case's own measured value plus a margin, and the
        # two cases differ by more than a factor of two on bias, so a reader at the assertion needs to know
        # which number they are looking at.
        cases = [(case = "NISAR_L1_PR_RSLC", corr_x = 0.99, corr_y = 0.98, bias_x = 0.12, bias_y = 0.17),
                 (case = "NISAR_L2_PR_GSLC", corr_x = 0.99, corr_y = 0.99, bias_x = 0.21, bias_y = 0.27)]
        results = String[]
        red = 0
        skipped = 0
        for (i, g) in enumerate(cases)
            c = g.case
            progress(@sprintf("3.nisar %d/%d %s", i, length(cases), c))
            # Run 100 holds the capture on both cases; L1's run 200 directory exists but is empty, so
            # naming the run explicitly is what keeps this from skipping.
            # `--block` is stated rather than defaulted: the tile size *is* the calibration. At a 512-px
            # tile L1's `dx` correlation is 0.923 against 0.9968 here, so a threshold inherited by one
            # tiling and measured at another reports a regression that is only a changed default.
            cmd = `julia --project=$(@__DIR__) -t 8 $(joinpath(@__DIR__, "correlator.jl"))
                   $c --run 100 --stride 4 --block 128`
            state, detail = capture_run(cmd) do t
                p = parse_endpoint(t)
                p isa Tuple && return p
                (; bx, by, corr, sgn, both) = p
                fails = String[]
                corr["dx"] >= g.corr_x || push!(fails, "dx corr $(corr["dx"]) < $(g.corr_x)")
                corr["dy"] >= g.corr_y || push!(fails, "dy corr $(corr["dy"]) < $(g.corr_y)")
                # L2 carries the larger `dy` core bias of the two, which is the whole-grid finding as well
                # and is unexplained there.
                bx <= g.bias_x || push!(fails, "dx core bias $bx > $(g.bias_x)")
                by <= g.bias_y || push!(fails, "dy core bias $by > $(g.bias_y)")
                # The `dy` sign, for the same reason `3.rdr` asserts it: a reintroduced flip shows up here
                # and nowhere in the value statistics.
                sgn["dy"] == "-" || push!(fails, "dy sign $(sgn["dy"]), expected -")
                isempty(fails) || return (:red, join(fails, "; "))
                return (:green, @sprintf("core %.4f/%.4f corr %+.5f/%+.5f both %d",
                                         bx, by, corr["dx"], corr["dy"], both))
            end
            state === :red && (red += 1)
            state === :skipped && (skipped += 1)
            progress(@sprintf("3.nisar %d/%d %s → %s %s", i, length(cases), c, String(state), detail))
            push!(results, "$(first(c, 12)) $(state === :green ? "ok" : String(state))")
        end
        return (red == 0 ? :green : :red,
                "$(length(cases) - red - skipped)/$(length(cases)) green" *
                (skipped > 0 ? ", $skipped without a capture" : "") *
                (red > 0 ? ": " * join(filter(r -> !endswith(r, "ok"), results), ", ") : ""))
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
    scheduled = count(g -> all || !g.slow, GATES)
    done = 0
    for g in GATES
        if g.slow && !all
            @printf("%-6s %-52s %-9s %s\n", g.id, g.what, "deferred", "pass --all to run it")
            continue
        end
        progress(@sprintf("gate %d/%d %s: %s", done + 1, scheduled, g.id, g.what))
        state, detail = try
            g.run()
        catch e
            (:red, sprint(showerror, e))
        end
        push!(states, state)
        done += 1
        @printf("%-6s %-52s %-9s %s\n", g.id, g.what, uppercase(String(state)), detail)
        # Flush per gate: the table is the only record of a gate that has already run, and a redirect
        # block-buffers stdout, so an interrupted run would otherwise lose every verdict it reached.
        flush(stdout)
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
