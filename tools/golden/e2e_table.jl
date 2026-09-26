# AutoRIFT.jl against the Python reference, end to end, on every golden case.
#
#   julia --project=tools/golden tools/golden/e2e_table.jl [--python FILE]
#
# Reads what the two sweeps recorded and joins them; it measures nothing itself, so it is safe to run
# while a sweep is still going and will simply report fewer rows.
#
#   * `block_optimum.jls` — `tools/golden/block_optimum.jl`'s per-case ladder. The Julia figure is the
#     arm that minimizes peak, which is the configuration a 16 GiB instance would be given, and the
#     untiled arm alongside it because that is what a caller gets without `process_block_size`.
#   * `golden_python.tsv` — `tools/ab/golden_python_all.py`, one process per case so its peak is that
#     case's own high-water mark.
#
# **Both sides run the same correlator call on the same captured inputs.** The capture holds the
# reference's own `xGrid`, search limits and priors at the moment it called `runAutorift`, so neither
# side re-derives the grid — which is where a cross-language comparison usually goes wrong. What is
# compared is the pyramid, the correlator, the coherence filter and the merge.
#
# **The two peaks are not measured the same way and the difference is stated rather than hidden.** The
# Julia figure is a sampled resident footprint over the run (`with_trace`), the Python figure is
# `ru_maxrss` for the process. Both are high-water marks of resident memory, but the Julia one excludes
# whatever the harness held before the run started and the Python one does not, so the Python column
# carries its interpreter and the capture arrays. Read the ratio as indicative, and the runtimes as
# exact.
#
# **Point counts are reported because they bound what a runtime means.** A run that measured fewer
# points did less work. `dev/GATES.md` is where agreement is judged; this only flags a gap.

using Printf, Serialization

const OPT = joinpath(get(ENV, "AUTORIFT_GOLDEN_CACHE",
                         joinpath(expanduser("~/data/autorift/tests"), "golden_tests")),
                     "mem", "block_optimum.jls")

pick(rows) = isempty(rows) ? nothing : begin
    best = argmin(r -> r.peak, rows)
    argmin(r -> r.seconds, filter(r -> r.peak <= 1.03 * best.peak, rows))
end

label(bs) = bs == (0, 0) ? "untiled" : bs[1] == bs[2] ? "$(bs[1])" : "$(bs[1])x$(bs[2])"

function read_python(path)
    out = Dict{String,NamedTuple}()
    isfile(path) || return out
    for (i, line) in enumerate(eachline(path))
        i == 1 && continue
        f = split(line, '\t')
        length(f) >= 9 || continue
        out[f[1]] = (; status = f[2], seconds = something(tryparse(Float64, f[3]), NaN),
                     total = something(tryparse(Float64, f[4]), NaN),
                     peak = something(tryparse(Int, f[5]), 0),
                     measured = something(tryparse(Int, f[6]), 0),
                     threads = f[9])
    end
    return out
end

function main()
    i = findfirst(==("--python"), ARGS)
    pypath = isnothing(i) ? joinpath(get(ENV, "CLAUDE_JOB_DIR", tempdir()), "tmp", "golden_python.tsv") :
             ARGS[i + 1]
    jl = isfile(OPT) ? deserialize(OPT) : Dict{String,Any}()
    py = read_python(pypath)

    @printf("%-44s %9s %8s %9s %10s %8s %9s %7s\n", "case", "jl block", "jl s",
            "jl GiB", "py s", "py GiB", "speedup", "pts jl/py")
    nboth = 0
    ratios = Float64[]
    for k in sort(collect(union(keys(jl), keys(py))))
        rows = get(jl, k, nothing)
        b = (isnothing(rows) || rows === :failed) ? nothing : pick(rows)
        p = get(py, k, nothing)
        jls = isnothing(b) ? NaN : b.seconds
        jlg = isnothing(b) ? NaN : b.peak / 2^30
        pys = isnothing(p) ? NaN : p.seconds
        pyg = isnothing(p) ? NaN : p.peak / 2^30
        sp = (isfinite(jls) && isfinite(pys) && jls > 0) ? pys / jls : NaN
        isfinite(sp) && (nboth += 1; push!(ratios, sp))
        pts = (isnothing(b) || isnothing(p) || p.measured == 0) ? "—" :
              @sprintf("%.3f", b.measured / p.measured)
        @printf("%-44s %9s %8.1f %9.2f %10.1f %8.2f %7s %9s\n", first(k, 44),
                isnothing(b) ? "—" : label(b.block), jls, jlg, pys, pyg,
                isfinite(sp) ? @sprintf("%.0fx", sp) : (isnothing(p) ? "—" : p.status),
                pts)
    end
    if !isempty(ratios)
        sort!(ratios)
        @printf("\n%d cases measured on both sides: speedup median %.0fx, min %.0fx, max %.0fx\n",
                nboth, ratios[(end + 1) ÷ 2], first(ratios), last(ratios))
    end
    # Untiled beside the optimum, since that is what a caller gets with no `process_block_size` and the
    # gap is the whole argument for choosing one.
    println("\nblocked against untiled, Julia:")
    @printf("  %-44s %9s %9s %9s %9s\n", "case", "opt GiB", "unt GiB", "opt s", "unt s")
    for k in sort(collect(keys(jl)))
        rows = jl[k]
        rows === :failed && continue
        b = pick(rows)
        u = findfirst(r -> r.block == (0, 0), rows)
        (isnothing(b) || isnothing(u)) && continue
        @printf("  %-44s %9.2f %9.2f %9.1f %9.1f\n", first(k, 44),
                b.peak / 2^30, rows[u].peak / 2^30, b.seconds, rows[u].seconds)
    end
    return nothing
end

main()
