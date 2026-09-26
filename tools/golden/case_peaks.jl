# The lowest peak ever recorded for each golden case, and which block size produced it.
#
#   julia --project=tools/golden tools/golden/case_peaks.jl [limit-GiB]
#
# Reads `$AUTORIFT_GOLDEN_CACHE/mem/prof_*.jls` and nothing else — no imagery, no correlation, seconds to
# run. It exists because the question "which cases do not fit a 16 GiB instance" was answered from the two
# granules `docs/src/explanation/memory.md` happens to document, and that answer was wrong: four
# Sentinel-1 cases are larger than either, and every row recorded for them is untiled.
#
# A case whose best row is `untiled` has never had `process_block_size` applied to it. That is a gap in
# the measurement rather than a property of the case — see `dev/plan-16gib.md`.

using Serialization, Printf

const MEM = joinpath(get(ENV, "AUTORIFT_GOLDEN_CACHE",
                         joinpath(expanduser("~/data/autorift/tests"), "golden_tests")), "mem")

function main()
    limit = (length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 16.0)
    # Keyed on the file stem, which is `history_path`'s one-file-per-case name — **not** on each record's
    # own `case` field. Only rows written by a recent `profile_nisar.jl` carry that field, so keying on it
    # splits a case's history by vintage and hides its newest rows.
    byStem = Dict{String,Vector{Any}}()
    for f in sort(filter(n -> startswith(n, "prof_") && endswith(n, ".jls"), readdir(MEM)))
        hist = try
            deserialize(joinpath(MEM, f))
        catch
            continue                 # a partial file is skipped rather than fatal
        end
        # Whole-grid rows only: a thinned row searches a fraction of the points, so its peak is not a
        # fraction of the whole-grid one and the two are not comparable.
        whole = [r for r in hist if get(r, :stride, 1) == 1 && get(r, :measured, 0) > 0]
        isempty(whole) || (byStem[replace(f, "prof_" => "", ".jls" => "")] = whole)
    end
    # A per-experiment side file — `..._aniso1.jls` beside `....jls` — is the same case measured under a
    # different setting, and its stem extends the canonical one. Merging on that prefix keeps it from
    # being reported as an extra case, which made one superseded single-row experiment look like a case
    # over the limit.
    for stem in sort(collect(keys(byStem)); by = length, rev = true)
        haskey(byStem, stem) || continue
        others = filter(k -> k != stem && startswith(stem, k), collect(keys(byStem)))
        isempty(others) && continue
        parent = argmax(length, others)      # the longest stem this one extends
        append!(byStem[parent], byStem[stem])
        delete!(byStem, stem)
    end
    rows = NamedTuple[]
    for (stem, whole) in byStem
        best = argmin(r -> r.peak, whole)
        push!(rows, (; case = stem,
                     block = best.block == (0, 0) ? "untiled" : string(best.block),
                     peak = best.peak / 2^30, seconds = get(best, :clean_seconds, NaN),
                     points = get(best, :measured, 0), rows = length(whole),
                     blocked = count(r -> r.block != (0, 0), whole)))
    end
    sort!(rows; by = r -> -r.peak)
    @printf("%-56s %-14s %8s %9s %10s %7s\n",
            "case", "best block", "peak GiB", "runtime", "points", "blocked")
    for r in rows
        @printf("%-56s %-14s %8.2f %8.1f s %10d %4d/%-3d %s\n",
                r.case[1:min(end, 56)], r.block, r.peak, r.seconds, r.points, r.blocked, r.rows,
                r.peak > limit ? " OVER" : "")
    end
    over = filter(r -> r.peak > limit, rows)
    @printf("\n%d cases recorded; %d over %.0f GiB, of which %d have no blocked row at all\n",
            length(rows), length(over), limit, count(r -> r.blocked == 0, over))
end

main()
