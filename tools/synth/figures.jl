# The figures for the synthetic comparison: error against deformation for every arm, and per-case error
# maps.
#
#   julia --project=tools/synth tools/synth/figures.jl
#
# Writes PNGs to `plots/`, which is gitignored -- the numbers they show are in `results.json` and the
# README, which are text.

using CairoMakie, Printf, Statistics

include(joinpath(@__DIR__, "bundle.jl"))
include(joinpath(@__DIR__, "gate.jl"))

const PLOTS = joinpath(@__DIR__, "plots")

# The sub-pixel threshold the real-scene comparison is judged against, drawn as a reference line.
const TOL = 0.25

# The arms drawn, in a fixed order and colour so every panel reads the same way. `cv_pyrup` is drawn
# dashed because it tracks `jl_correlator` to the last bit at almost every point: a solid line would be
# invisible underneath, and its absence would read as a missing arm rather than as an exact overlap.
const SERIES = (("jl_correlator", "AutoRIFT.jl", :dodgerblue, :solid),
                ("py_correlator", "autoRIFT.py", :orangered, :solid),
                ("cv_pyrup", "OpenCV + pyrUp", :black, :dash),
                ("cv_parabola", "OpenCV + parabola", :seagreen, :solid))

# The sign that puts each arm in truth space, taken from `gate.jl`'s measured table rather than restated.
# A second copy is how this script came to draw a 40 px error band for an arm whose median error was
# 0.0157 px: the table was corrected in one file and not the other, and a figure has no gate to fail.
const SIGNS = Dict(ARMS)

"""
    arm_error(dir, arm, tdx, tdy) -> Union{Matrix{Float64},Nothing}

Per-point distance from truth, `NaN` where the arm has no answer.

`NaN` and not zero: an unmeasured point is not a perfect one, and on a map the two must not look alike.
"""
function arm_error(dir, arm, tdx, tdy)
    a = read_arm(dir, arm)
    a === nothing && return nothing
    sx, sy = SIGNS[arm]
    dx, dy = sx .* a.arrays["dx"], sy .* a.arrays["dy"]
    return map((x, y, tx, ty) -> isnan(x) || isnan(y) ? NaN : hypot(x - tx, y - ty),
               dx, dy, tdx, tdy)
end

# A grid array as `heatmap!` needs it to appear in image orientation: x horizontal, y vertical,
# increasing downward.
#
# `heatmap!(A)` treats A's *first* index as the x axis, so passing a `[row, col]` array directly draws
# the image transposed. Transposing fixes the axes and reversing the resulting columns puts row 1 at the
# top. Same transform as `tools/ab/bench_figures.jl` applies, for the same reason.
mapshow(A) = reverse(permutedims(A); dims = 2)

"""
    error_vs_deformation(cases) -> path

Median error against the local deformation rate, one panel per case.

The comparison's central figure: every arm on one axis pair, so where the implementations separate and
where they coincide is visible directly rather than inferred from a table. Binned by deformation
quantile, so each point carries a comparable number of samples.
"""
function error_vs_deformation(names; path = joinpath(PLOTS, "error_vs_deformation.png"))
    # Only the cases whose deformation spans a *decade*, which is what this figure plots against. A
    # pure translation is uniform, and `divergence` and `rigid_rotation` are near-uniform by
    # construction -- their deformation is constant in space by design -- so binning them produces a
    # flat line over a meaninglessly narrow x range rather than a trend. Those cases are reported in
    # the tables, where a single number per arm is the honest form.
    drawn = filter(names) do name
        g = read_bundle(joinpath(SCENES, name)).arrays["true_gradient"]
        lo, hi = minimum(g), maximum(g)
        hi > 0 && hi / max(lo, hi / 1e6) > 10
    end
    ncol = 3
    nrow = cld(length(drawn), ncol)
    fig = Figure(; size = (440 * ncol, 320 * nrow))

    for (k, name) in enumerate(drawn)
        dir = joinpath(SCENES, name)
        b = read_bundle(dir)
        tdx, tdy = b.arrays["true_dx"], b.arrays["true_dy"]
        grad = vec(b.arrays["true_gradient"])
        row, col = fldmod1(k, ncol)
        ax = Axis(fig[row, col]; title = name, xlabel = "deformation (px/px)",
                  ylabel = "median error (px)", yscale = log10,
                  # Fixed limits across every panel, so a curve's height means the same thing in each
                  # and the panels can be compared by eye rather than only read individually.
                  limits = (nothing, (5e-4, 3.0)))

        edges = quantile(sort(grad), range(0, 1; length = 9))
        for (arm, label, colour, style) in SERIES
            e = arm_error(dir, arm, tdx, tdy)
            e === nothing && continue
            ev = vec(e)
            xs, ys = Float64[], Float64[]
            for i in 1:(length(edges) - 1)
                sel = (grad .>= edges[i]) .& (grad .<= edges[i + 1]) .& .!isnan.(ev)
                count(sel) < 20 && continue
                push!(xs, median(grad[sel]))
                # Clamped away from zero because the axis is logarithmic and an exactly-recovered
                # stratum is common here: several arms hit 0.0000 px on the low-deformation strata,
                # which has no position on a log axis. The floor is below the 1/16 px quantization
                # step, so it cannot be mistaken for a real value.
                push!(ys, max(median(ev[sel]), 1e-3))
            end
            length(xs) >= 2 && lines!(ax, xs, ys; color = colour, linestyle = style, label,
                                      linewidth = 2)
        end
        hlines!(ax, [TOL]; color = :gray60, linestyle = :dot)
        k == 1 && axislegend(ax; position = :lt, framevisible = false, labelsize = 9)
    end

    mkpath(PLOTS)
    save(path, fig; px_per_unit = 2)
    return path
end

"""
    error_maps(name) -> path

Where each arm errs on one case, at a shared colour scale, beside the deformation field that explains
it.

A map and not a distribution because the real-scene disagreement was spatially coherent: it sat in one
patch at the shear margin, which no histogram of the same numbers would show.
"""
function error_maps(name; path = joinpath(PLOTS, "error_maps_$name.png"))
    dir = joinpath(SCENES, name)
    b = read_bundle(dir)
    tdx, tdy = b.arrays["true_dx"], b.arrays["true_dy"]
    grad = b.arrays["true_gradient"]

    arms = filter(s -> read_arm(dir, s[1]) !== nothing, SERIES)
    fig = Figure(; size = (300 * (length(arms) + 1), 340))
    mapax(k, title) = Axis(fig[1, k]; title, aspect = DataAspect(),
                           xticksvisible = false, yticksvisible = false,
                           xticklabelsvisible = false, yticklabelsvisible = false)

    hm = heatmap!(mapax(1, "deformation (px/px)"), mapshow(grad); colormap = :viridis)
    Colorbar(fig[2, 1], hm; vertical = false, height = 10)

    # One scale across every arm, from the pooled 99th percentile: per-panel scaling would make the
    # worst arm look like the best.
    pooled = Float64[]
    for (arm, _, _, _) in arms
        e = arm_error(dir, arm, tdx, tdy)
        e === nothing || append!(pooled, filter(isfinite, vec(e)))
    end
    top = isempty(pooled) ? 1.0 : max(quantile(pooled, 0.99), 1e-3)

    for (k, (arm, label, _, _)) in enumerate(arms)
        e = arm_error(dir, arm, tdx, tdy)
        h = heatmap!(mapax(k + 1, label), mapshow(e); colormap = :inferno, colorrange = (0, top))
        Colorbar(fig[2, k + 1], h; vertical = false, height = 10, label = "error (px)")
    end

    mkpath(PLOTS)
    save(path, fig; px_per_unit = 2)
    return path
end

function main()
    names = sort(filter(n -> isfile(joinpath(SCENES, n, "manifest.txt")), readdir(SCENES)))
    isempty(names) && error("no cases in $SCENES; run scenes.jl first")

    p = error_vs_deformation(names)
    println("wrote ", p)
    # Maps for the cases the README argues from: the base shear margin, its steepest variant, and real
    # texture. Every case would be 26 figures nobody reads.
    for name in ("shear_a10_c16", "shear_a20_c16", "shear_landsat")
        name in names || continue
        println("wrote ", error_maps(name))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
