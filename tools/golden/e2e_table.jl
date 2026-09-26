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

# The row a user actually gets: the block `AutoRIFT.block_size_for` returns, which is
# `BLOCK_HALO_MULTIPLE` times the halo per axis floored at `BLOCK_FLOOR`. Falls back to the fastest
# blocked arm when the default was never measured on that case, and says which it used — reporting a
# swept optimum as though it were the default would overstate what the library delivers unasked.
default_block(h, scene) = (min(max(2 * h[1], 1024), scene[2]), min(max(2 * h[2], 1024), scene[1]))

function pick(rows)
    isempty(rows) && return (nothing, :none)
    h, sc = first(rows).halo, first(rows).scene
    want = default_block(h, sc)
    i = findfirst(r -> r.block == want, rows)
    isnothing(i) || return (rows[i], :default)
    blk = filter(r -> r.block != (0, 0), rows)
    isempty(blk) && return (nothing, :none)
    return (argmin(r -> r.seconds, blk), :fastest)
end

# `mem_nisar.jl` writes one file per case beside the sweep's, and the default was measured there for the
# cases whose default is not on the sweep's ladder. Merged so the table sees every row that exists.
function merged(dir)
    out = Dict{String,Vector{Any}}()
    isdir(dir) || return out
    for f in readdir(dir)
        endswith(f, ".jls") || continue
        f == "block_optimum.jls" && continue
        rows = try deserialize(joinpath(dir, f)) catch; continue end
        rows isa Vector && !isempty(rows) && haskey(first(rows), :halo) || continue
        out[replace(f, ".jls" => "")] = rows
    end
    return out
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
    dir = dirname(OPT)
    jl = isfile(OPT) ? deserialize(OPT) : Dict{String,Any}()
    side = merged(dir)
    py = read_python(pypath)

    # `mem_nisar.jl` names its file from the product's first half, so a side row is attached to the case
    # whose product it prefixes rather than by an exact key match.
    rowsfor(k) = begin
        base = get(jl, k, nothing)
        base = (base === :failed || isnothing(base)) ? Any[] : Any[r for r in base]
        for (sk, sv) in side
            startswith(k, sk) && append!(base, sv)
        end
        base
    end
    ab(r) = (Int(r.peak) - Int(r.floor_bytes)) / 2^30

    @printf("%-40s %-11s %7s %8s %7s %8s %6s %5s\n", "case", "jl block", "jl s",
            "jl GiB↑", "py s", "py GiB", "x", "src")
    ratios = Float64[]
    for k in sort(collect(union(keys(jl), keys(py))))
        rows = rowsfor(k)
        b, src = pick(rows)
        p = get(py, k, nothing)
        jls = isnothing(b) ? NaN : b.seconds
        jlg = isnothing(b) ? NaN : ab(b)
        pys = isnothing(p) ? NaN : p.seconds
        pyg = isnothing(p) ? NaN : p.peak / 2^30
        sp = (isfinite(jls) && isfinite(pys) && jls > 0) ? pys / jls : NaN
        isfinite(sp) && push!(ratios, sp)
        @printf("%-40s %-11s %7.1f %8.2f %7.1f %8.2f %6s %5s\n", first(k, 40),
                isnothing(b) ? "—" : label(b.block), jls, jlg, pys, pyg,
                isfinite(sp) ? @sprintf("%.0fx", sp) : "—",
                src === :default ? "def" : src === :fastest ? "fast" : "—")
    end
    if !isempty(ratios)
        sort!(ratios)
        @printf("\n%d cases on both sides: median %.0fx, min %.0fx, max %.0fx\n",
                length(ratios), ratios[(end + 1) ÷ 2], first(ratios), last(ratios))
    end
    # Untiled beside the default, since that is what a caller gets with no `process_block_size` and the
    # gap is the whole argument for having a default at all. Peaks are above each run's own floor.
    println("\nat the default block against untiled, Julia:")
    @printf("  %-40s %9s %9s %9s %9s\n", "case", "def GiB↑", "unt GiB↑", "def s", "unt s")
    for k in sort(collect(keys(jl)))
        rows = rowsfor(k)
        b, _ = pick(rows)
        u = findfirst(r -> r.block == (0, 0), rows)
        (isnothing(b) || isnothing(u)) && continue
        @printf("  %-40s %9.2f %9.2f %9.1f %9.1f\n", first(k, 40),
                ab(b), ab(rows[u]), b.seconds, rows[u].seconds)
    end
    return nothing
end

main()
