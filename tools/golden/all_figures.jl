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

# Run 200 for every case: the run numbers above it are one-off level traces and dtype pairs, and the
# figure wants the comparison every gate reads.
const RUN = 200

function main(args)
    dir = get(ENV, "AUTORIFT_GOLDEN_FIGS", tempdir())
    mkpath(dir)
    zoom = nothing
    i = findfirst(==("--zoom"), args)
    i === nothing || (zoom = parse(Int, args[i + 1]))
    dlim = 1.0
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
        cap = joinpath(homedir(), "data", "autorift", "tests", "golden_tests",
                       "runs", c.product, string(RUN), "capture", "call1.json")
        if !isfile(cap)
            push!(failed, (c.product, "no capture at run $RUN"))
            continue
        end
        # **The full product name, not a prefix.** Two L8xL7 pairs each contain the other's scene id,
        # so any truncation short of the full name matches both and `cases` refuses. The output name is
        # keyed on the product too, for the same reason: `S1A_..._20240618T025528` and `...T025533`
        # differ only past character 28.
        name = c.product
        out = joinpath(dir, "golden_$(c.platform)_$(c.product).png")
        tag == "" || (out = replace(out, ".png" => "$tag.png"))
        @info "figure" case = first(c.product, 40) platform = c.platform
        try
            compare_figure(name; path = out, zoom, dlim, run = RUN)
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
