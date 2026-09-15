# Recover measurement rows from a `profile_nisar.jl` console log into its append-only history.
#
#   julia --project=tools/golden tools/golden/backfill_history.jl \
#       NISAR_L2_PR_GSLC benchmark/results/nisar/sweep_l2b.log ...
#
# `profile_nisar.jl` used to serialize only the rows of the run that had just finished, so a sweep
# followed by a single-configuration re-measurement left a history describing one block size. The rows it
# dropped survive in the console logs under `benchmark/results/nisar/`, and this parses them back.
#
# Recovered rows carry `recovered = true` and the log they came from. They hold the figures the log
# prints — runtime, peak, floor, occupancy, block count, read amplification, point count — and **not**
# the per-stack profile, which only the serialized record ever had. A consumer that needs stacks must
# re-measure; one that needs the cost table does not.
#
# Kept as a script rather than folded into `profile_nisar.jl`: it is a one-time repair of a specific
# defect, and the harness no longer creates the situation it repairs.

include(joinpath(@__DIR__, "manifest.jl"))

using Printf, Serialization

const TRACE_DIR = joinpath(get(ENV, "AUTORIFT_GOLDEN_CACHE",
                               joinpath(expanduser("~/data/autorift/tests"), "golden_tests")),
                           "mem")

# The two lines a configuration prints, and the ones after it that qualify it. Matched together so a
# half-written row — a run killed mid-configuration — is skipped rather than half-parsed.
const RE_CLEAN = r"^\s{2}(\S+(?: px)?)\s+(\d+\.\d)\s+s clean\s*$"
const RE_ROW = r"^\s{2}(\S+(?: px)?)\s+(\d+) blocks\s+(\d+\.\d) s\s+peak\s+(\d+\.\d+) GiB \((\d+\.\d+) above floor\)\s+readamp\s+(\d+\.\d+)x\s+measured (\d+)"
const RE_OCC = r"occupancy\s+(\d+\.\d+) of (\d+) threads"
const RE_SCENE = r"scene (\d+) x (\d+) px, grid (\d+) x (\d+), halo (\d+) x (\d+) px"

function parse_label(s)
    t = replace(s, " px" => "")
    t == "untiled" && return (0, 0)
    p = split(t, 'x')
    length(p) == 1 && return (n = parse(Int, p[1]); (n, n))
    return (parse(Int, p[1]), parse(Int, p[2]))
end

"""
    rows_from_log(path) -> Vector{NamedTuple}

Every complete configuration in a `profile_nisar.jl` log.

A configuration is complete when its `clean` line, its summary line and its occupancy line are all
present; the last configuration of a killed run is therefore dropped, which is the intent — a row whose
run did not finish is not a measurement.
"""
function rows_from_log(path::AbstractString)
    lines = readlines(path)
    scene = grid = halo = nothing
    for l in lines
        m = match(RE_SCENE, l)
        isnothing(m) && continue
        v = parse.(Int, m.captures)
        scene, grid, halo = (v[1], v[2]), (v[3], v[4]), (v[5], v[6])
        break
    end
    isnothing(scene) && error("no scene line in $path; is this a profile_nisar.jl log?")

    out = NamedTuple[]
    clean = Dict{Tuple{Int,Int},Float64}()
    for (i, l) in pairs(lines)
        mc = match(RE_CLEAN, l)
        if !isnothing(mc)
            clean[parse_label(mc.captures[1])] = parse(Float64, mc.captures[2])
            continue
        end
        m = match(RE_ROW, l)
        isnothing(m) && continue
        bs = parse_label(m.captures[1])
        # Occupancy is two lines below the summary; search forward a few rather than assuming.
        occ = nthreads = nothing
        for j in (i + 1):min(i + 6, length(lines))
            mo = match(RE_OCC, lines[j])
            isnothing(mo) && continue
            occ, nthreads = parse(Float64, mo.captures[1]), parse(Int, mo.captures[2])
            break
        end
        (haskey(clean, bs) && !isnothing(occ)) || continue
        push!(out, (; block = bs, nblocks = parse(Int, m.captures[2]),
                    clean_seconds = clean[bs], seconds = parse(Float64, m.captures[3]),
                    peak = round(Int, parse(Float64, m.captures[4]) * 2^30),
                    peak_above_floor = round(Int, parse(Float64, m.captures[5]) * 2^30),
                    readamp = parse(Float64, m.captures[6]),
                    measured = parse(Int, m.captures[7]),
                    clean_occupancy = occ, nthreads,
                    scene, grid, halo,
                    recovered = true, source = basename(path),
                    scan = nothing))
    end
    return out
end

function main()
    length(ARGS) >= 2 || error("usage: backfill_history.jl <case-fragment> <log> [<log>...]")
    case = only(cases(ARGS[1]))
    path = joinpath(TRACE_DIR, "prof_$(first(split(case.product, "_X_"))).jls")

    recovered = NamedTuple[]
    for log in ARGS[2:end]
        rs = rows_from_log(log)
        @printf("%-40s %d row%s\n", basename(log), length(rs), length(rs) == 1 ? "" : "s")
        append!(recovered, rs)
    end

    # A history written before the trace was stripped embeds `MemTrace` and `ProfileScan`, which this
    # process has no definitions for, so reading it throws `UndefVarError` rather than returning rows.
    # Such a file is set aside rather than overwritten: its rows are still recoverable by anything that
    # loads the harness, and destroying them to let this script succeed would be the same data loss the
    # script exists to repair.
    existing = if isfile(path)
        try
            deserialize(path)
        catch e
            e isa UndefVarError || rethrow()
            aside = path * ".pre-strip"
            mv(path, aside; force = true)
            @printf("moved %s aside: it embeds harness structs this script cannot load (%s)\n",
                    basename(path), e.var)
            []
        end
    else
        []
    end
    # Recovered rows go first, so a later serialized row for the same block size supersedes the parsed
    # one — the serialized record is strictly richer, carrying the profile the log never had.
    serialize(path, vcat(recovered, existing))
    @printf("\n%d recovered + %d existing = %d rows in %s\n",
            length(recovered), length(existing), length(recovered) + length(existing), path)
    return nothing
end

main()
