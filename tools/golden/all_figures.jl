# Every golden case's comparison figure, in one pass.
#
#   AUTORIFT_GOLDEN_FIGS=figs julia --project=tools/ab -t 6 tools/golden/all_figures.jl
#   AUTORIFT_GOLDEN_FIGS=figs julia --project=tools/ab -t 6 tools/golden/all_figures.jl --zoom 90
#
# Same layout as `compare_figures.jl` for one case, which mirrors `tools/ab/bench_figures.jl` — so the
# benchmark, one golden case and the whole set are all read the same way.
#
# **A failure on one case does not stop the others.** Twenty cases at minutes each is an hour of
# rendering, and a single bad case aborting the run means finding that out at the end. Each is wrapped,
# and the summary at the end names what failed.
#
# `--project=tools/ab` because CairoMakie lives there.
const R = @__DIR__
include(joinpath(R, "compare_figures.jl"))

# Run 200 where it exists, because that is the comparison every gate reads; the run numbers above it are
# one-off level traces and dtype pairs. The NISAR captures are at 100, so a hardcoded 200 skips them and
# reports "no capture" on the two cases whose figures are most wanted. Prefer 200, fall back to the
# highest ordinary run on disk — 201 and above are the level traces, which carry a partial grid.
const RUN = 200
const RUN_MAX_ORDINARY = 200

function run_for(product::AbstractString)
    base = joinpath(homedir(), "data", "autorift", "tests", "golden_tests", "runs", product)
    isdir(base) || return nothing
    runs = Int[]
    for e in readdir(base)
        n = tryparse(Int, e)
        n === nothing && continue
        n <= RUN_MAX_ORDINARY || continue
        isfile(joinpath(base, e, "capture", "call1.json")) && push!(runs, n)
    end
    isempty(runs) && return nothing
    return RUN in runs ? RUN : maximum(runs)
end

function main(args)
    dir = get(ENV, "AUTORIFT_GOLDEN_FIGS", tempdir())
    mkpath(dir)
    zoom = nothing
    i = findfirst(==("--zoom"), args)
    i === nothing || (zoom = parse(Int, args[i + 1]))
    dlim = 0.3
    d = findfirst(==("--dlim"), args)
    d === nothing || (dlim = parse(Float64, args[d + 1]))
    only_plat = nothing
    j = findfirst(==("--platform"), args)
    j === nothing || (only_plat = args[j + 1])

    cs = cases()
    only_plat === nothing || (cs = filter(c -> startswith(c.platform, only_plat), cs))
    tag = zoom === nothing ? "" : "_zoom$zoom"

    ok = String[]; failed = Tuple{String,String}[]
    for c in cs
        # Skip a case with no capture rather than failing on it: the figure is a diagnostic and a
        # missing capture is a fact about the cache, not an error in the comparison.
        run = run_for(c.product)
        if run === nothing
            push!(failed, (c.product, "no capture at run $RUN or below"))
            continue
        end
        # **The full product name, not a prefix.** Two L8xL7 pairs each contain the other's scene id,
        # so any truncation short of the full name matches both and `cases` refuses. The output name is
        # keyed on the product too, for the same reason: `S1A_..._20240618T025528` and `...T025533`
        # differ only past character 28.
        name = c.product
        out = joinpath(dir, "golden_$(c.platform)_$(c.product).png")
        tag == "" || (out = replace(out, ".png" => "$tag.png"))
        @info "figure" case = first(c.product, 40) platform = c.platform run
        try
            compare_figure(name; path = out, zoom, dlim, run)
            push!(ok, out)
        catch e
            push!(failed, (c.product, sprint(showerror, e)))
            @warn "figure failed" case = first(c.product, 40) error = sprint(showerror, e)
        end
    end

    println("\n", length(ok), " figure(s) written to ", dir)
    for p in ok; println("  ", basename(p)); end
    if !isempty(failed)
        println("\n", length(failed), " case(s) without a figure:")
        for (p, why) in failed; println("  ", first(p, 44), " — ", why); end
    end
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
