# The accuracy figures for the report: AutoRIFT.jl against the Python reference on one window, as
# heatmaps of each field and as distributions of every output.
#
# Reads the stage-2 bundle `stage2_julia.jl` and `stage2_python.py` write, so the two sides are the
# same window through the same parameters. `compare2.jl` prints the statistics; this draws them.
#
# Included by `bench_table.jl`, which embeds the PNGs it returns into the PDF. Standalone:
#
#   julia --project=tools/ab tools/ab/bench_figures.jl

using CairoMakie, Printf, Statistics

const FIG_D = joinpath(@__DIR__, "stage2")
const FIG_PLOTS = joinpath(@__DIR__, "plots")
# The sub-pixel matching-noise limit the comparison is judged against.
const FIG_TOL = 0.2

# The canvas both pages are drawn at. Sized to the aspect of what is left on a landscape letter page
# after the heading — 9.4in by 5.6in — so `bench_table.jl` can scale a page to full width and have it
# fill the page rather than letterbox in the top half.
const FIG_SIZE = (1560, 929)

# The stage-2 bundle: both sides cropped to the grid they share.
#
# `runAutorift` truncates its grid to a whole multiple of `ChipSizeMaxX / ChipSize0X` before
# correlating, so the reference's grid is the top-left sub-block of the one it was handed. Cropping
# AutoRIFT.jl to match compares the same grid points rather than resampling either side.
#
# The reference's displacements are negated on both axes to reach AutoRIFT.jl's secondary-to-reference
# convention. That negation is correct only for the chip/window assignment `stage2_python.py` uses; if
# that assignment changes, this sign must be re-established by scoring both candidates rather than
# carried over.
function load_stage2()
    shapes = Dict{String,Tuple{Int,Int}}()
    scalars = Dict{String,Int}()
    for line in eachline(joinpath(FIG_D, "manifest.txt"))
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        parts = split(s)
        if length(parts) == 2
            scalars[parts[1]] = parse(Int, parts[2])
        else
            shapes[parts[1]] = Tuple(parse.(Int, split(parts[3], "x")))
        end
    end
    rd(name, T, dims) = reshape(collect(reinterpret(T, read(joinpath(FIG_D, "$name.bin")))), dims)
    js = shapes["julia_dx"]
    ps = Tuple(parse.(Int, split(strip(read(joinpath(FIG_D, "python_shape.txt"), String)))))
    nr, nc = min(js[1], ps[1]), min(js[2], ps[2])
    crop(A) = A[1:nr, 1:nc]

    j = (dx = crop(rd("julia_dx", Float32, js)), dy = crop(rd("julia_dy", Float32, js)),
         corr = crop(rd("julia_correlation", Float32, js)),
         chip = crop(rd("julia_chip_size", Int32, js)))
    p = (dx = crop(.-rd("python_dx", Float32, ps)), dy = crop(.-rd("python_dy", Float32, ps)),
         chip = crop(rd("python_chip_size", Int32, ps)))

    jok, pok = .!isnan.(j.dx), .!isnan.(p.dx)
    both = jok .& pok
    radial = sqrt.((j.dx .- p.dx) .^ 2 .+ (j.dy .- p.dy) .^ 2)
    return (; j, p, jok, pok, both, radial, scalars, shape = (nr, nc))
end

# `NaN` where a point has no answer, so an unmeasured point reads as blank rather than as zero — a
# zero displacement is a real measurement and must not look like a missing one.
blank(A, keep) = map((v, k) -> k ? Float64(v) : NaN, A, keep)

# A grid array as `heatmap!` needs it to appear in image orientation: x horizontal, y vertical,
# increasing downward.
#
# `heatmap!(A)` treats A's *first* index as the x axis, so passing a `[row, col]` array directly draws
# the image transposed — rows run horizontally. Transposing fixes the axes, and reversing the resulting
# columns puts row 1 at the top, which is where the northernmost row of a north-up scene belongs.
# Without this, a feature elongated along y appears elongated along x, which inverts the reading of any
# anisotropy in the field.
mapshow(A) = reverse(permutedims(A); dims = 2)

# Page 1 of the comparison: every field as a map, both sides at one color scale.
function heatmap_page(d; path = joinpath(FIG_PLOTS, "fig_heatmaps.png"))
    (; j, p, jok, pok, both, radial) = d
    # No title or caption: the report page supplies both, and duplicating them here costs the panels
    # a quarter of the page height.
    fig = Figure(; size = FIG_SIZE, figure_padding = 12)

    # One symmetric scale across both sides of both axes, so the four maps are directly comparable
    # and a difference in sign or magnitude is visible rather than absorbed by per-panel scaling.
    vals = filter(isfinite, vcat(vec(blank(j.dx, jok)), vec(blank(p.dx, pok)),
                                 vec(blank(j.dy, jok)), vec(blank(p.dy, pok))))
    lim = isempty(vals) ? 1.0 : quantile(abs.(vals), 0.99)
    dlim = (-lim, lim)

    mapax(pos, title) = Axis(fig[pos...]; title, aspect = DataAspect(),
                             xticksvisible = false, yticksvisible = false,
                             xticklabelsvisible = false, yticklabelsvisible = false)

    for (row, (nm, J, P)) in enumerate((("dx", j.dx, p.dx), ("dy", j.dy, p.dy)))
        hm = heatmap!(mapax((row, 1), "AutoRIFT.jl $nm"), mapshow(blank(J, jok));
                      colormap = :balance, colorrange = dlim)
        heatmap!(mapax((row, 2), "autoRIFT.py $nm"), mapshow(blank(P, pok));
                 colormap = :balance, colorrange = dlim)
        Colorbar(fig[row, 3], hm; label = "$nm (px)", width = 12)

        # Signed difference per axis on a diverging scale, so which way each side leans is visible.
        # A magnitude on a sequential scale cannot show that, and the sign is the whole question for a
        # bias: a field of +0.1 px and one of ±0.1 px look identical under `abs`.
        hmd = heatmap!(mapax((row, 4), "difference in $nm"), mapshow(blank(J .- P, both));
                       colormap = :balance, colorrange = (-2 * FIG_TOL, 2 * FIG_TOL))
        Colorbar(fig[row, 5], hmd; label = "AutoRIFT.jl − autoRIFT.py (px)", width = 12)
    end

    colgap!(fig.layout, 10)
    rowgap!(fig.layout, 8)
    save(path, fig; px_per_unit = 2)
    return path
end

# Page 2: the remaining outputs as maps, each beside where the two sides disagree about it.
#
# Chip size and the interpolation flag are categorical, so a difference in them is not a magnitude but
# a disagreement — mapped as such rather than subtracted. The one continuous comparison that belongs
# here is the dx/dy difference distribution, which is what says whether the disagreement is biased.
function histogram_page(d; path = joinpath(FIG_PLOTS, "fig_histograms.png"))
    (; j, p, jok, pok, both, radial, scalars) = d
    ddx, ddy = j.dx[both] .- p.dx[both], j.dy[both] .- p.dy[both]

    fig = Figure(; size = FIG_SIZE, figure_padding = 12)
    mapax(pos, title) = Axis(fig[pos...]; title, aspect = DataAspect(),
                             xticksvisible = false, yticksvisible = false,
                             xticklabelsvisible = false, yticklabelsvisible = false)

    # Which level answered each point, both sides at one scale. `0` means no level resolved it, which
    # is categorically different from a measurement and so is drawn as blank.
    chips = sort(filter(>(0), unique(vcat(vec(j.chip), vec(p.chip)))))
    crange = (minimum(chips) - 0.5, maximum(chips) + 0.5)
    cmap = cgrad(:viridis, length(chips); categorical = true)
    for (row, (nm, C, ok)) in enumerate((("AutoRIFT.jl", j.chip, jok), ("autoRIFT.py", p.chip, pok)))
        hm = heatmap!(mapax((row, 1), "$nm chip size"), mapshow(blank(C, ok .& (C .> 0)));
                      colormap = cmap, colorrange = crange)
        # `height = Relative(0.85)` so a categorical bar matches its map rather than spanning the
        # taller cell the map's `DataAspect` leaves behind.
        Colorbar(fig[row, 2], hm; label = "chip size (px)", width = 12, ticks = chips,
                 height = Relative(0.85))
    end

    # Where the two chose a different level. A level difference changes how much the estimate is
    # smoothed, so this map is the mechanism behind much of the residual spread on the previous page —
    # and it is spatially structured, which a histogram of the same fact cannot show.
    samelevel = both .& (j.chip .== p.chip)
    disagree = map((b, s) -> b ? (s ? 0.0 : 1.0) : NaN, both, samelevel)
    hm = heatmap!(mapax((1, 3), "chip size: agree or disagree"), mapshow(disagree);
                  colormap = cgrad([:gray75, :crimson]; categorical = true), colorrange = (-0.5, 1.5))
    Colorbar(fig[1, 4], hm; width = 12, ticks = (0:1, ["same", "differ"]), height = Relative(0.85))

    # Coverage: which side answered at all. The pyramid and the outlier filter decide *which* points
    # get an answer, so two runs can agree everywhere they overlap and still disagree about most of the
    # grid — the half of the comparison no value statistic can show.
    cov = map((a, b) -> a && b ? 1.0 : a ? 2.0 : b ? 3.0 : 0.0, jok, pok)
    hmc = heatmap!(mapax((2, 3), "coverage"), mapshow(cov);
                   colormap = cgrad([:gray88, :gray30, :dodgerblue, :orangered]; categorical = true),
                   colorrange = (-0.5, 3.5))
    Colorbar(fig[2, 4], hmc; width = 12, height = Relative(0.85),
             ticks = (0:3, ["neither", "both", "jl only", "py only"]))

    # The one histogram on this page: the signed difference per axis. Symmetric about zero is the
    # claim — a shifted center is a systematic bias, which is a different defect from a wide spread.
    #
    # The tail is excluded from the bins rather than clamped into the end one: clamping piles every
    # outlier into a single edge bin, drawing a spike that reads as a real mode. The excluded fraction
    # is stated instead.
    ax = Axis(fig[1:2, 5]; title = "difference in dx and dy",
              xlabel = "AutoRIFT.jl − autoRIFT.py (px)", ylabel = "points")
    for (v, c, nm) in ((ddx, (:dodgerblue, 0.55), "dx"), (ddy, (:orangered, 0.55), "dy"))
        hist!(ax, filter(x -> abs(x) <= 1, v); bins = 121, color = c, label = nm)
    end
    vlines!(ax, [0]; color = :black, linestyle = :dash)
    axislegend(ax; position = :rt, framevisible = false)
    # Below the legend rather than beside it: the distribution is a spike at zero, so the upper corners
    # are the only clear space and both cannot have it.
    text!(ax, 0.97, 0.72;
          text = @sprintf("median %+.4f px (dx)\nmedian %+.4f px (dy)\n\nbeyond ±1 px: %.1f%% of dx,\n\
                           %.1f%% of dy\n\nsame chip size at %.1f%%\nof shared points",
                          median(ddx), median(ddy),
                          100 * mean(abs.(ddx) .> 1), 100 * mean(abs.(ddy) .> 1),
                          100 * count(samelevel) / count(both)),
          space = :relative, align = (:right, :top), fontsize = 10, color = :gray35)

    colgap!(fig.layout, 8)
    rowgap!(fig.layout, 8)
    # The histogram column is given the width of a map so the page reads as maps-plus-one-summary
    # rather than as two unrelated halves.
    colsize!(fig.layout, 5, Relative(0.26))
    save(path, fig; px_per_unit = 2)
    return path
end

# Both pages, and the statistics the report quotes in text.
function accuracy_figures()
    mkpath(FIG_PLOTS)
    d = load_stage2()
    (; j, both, radial) = d
    e = radial[both]
    stats = (; shared = count(both),
             jl_only = count(d.jok .& .!d.pok), ref_only = count(d.pok .& .!d.jok),
             median = median(e), p95 = quantile(e, 0.95),
             bias_dx = median(j.dx[both] .- d.p.dx[both]),
             bias_dy = median(j.dy[both] .- d.p.dy[both]),
             cor_dx = cor(j.dx[both], d.p.dx[both]), cor_dy = cor(j.dy[both], d.p.dy[both]),
             within = mean(e .<= FIG_TOL),
             within_gated = mean(radial[both .& (j.corr .>= 0.5)] .<= FIG_TOL),
             npix = d.scalars["npix"], upsampling = d.scalars["upsampling"],
             chip = d.scalars["chip"], chip_max = d.scalars["chip_max"],
             spacing = d.scalars["grid_spacing"], radius = d.scalars["radius"])
    return (; heatmaps = heatmap_page(d), histograms = histogram_page(d), stats)
end

# Standalone: draw both pages and print what they show.
if abspath(PROGRAM_FILE) == @__FILE__
    r = accuracy_figures()
    @printf("shared %d  median %.4f px  bias %+.4f/%+.4f  within %.1f%%\n",
            r.stats.shared, r.stats.median, r.stats.bias_dx, r.stats.bias_dy,
            100 * r.stats.within)
    println("wrote ", r.heatmaps, "\n      ", r.histograms)
end
