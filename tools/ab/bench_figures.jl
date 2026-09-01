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
# The scale the difference maps are drawn at: one step of the sub-pixel search, which is the finest
# distinction either implementation can draw. A scale rather than a pass/fail line — the two agree
# exactly at most points, so what the maps have to show is *where* the rest sit, not how many clear
# some cutoff.
const FIG_STEP = 1 / 16

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
# The reference's displacements are used as written, in the same sense as AutoRIFT.jl's: with the chip
# cut from the secondary on both sides (see `stage2_python.py`), both report the offset from secondary
# back to reference, and `stage2_python.py` has already undone the reference's cartesian y flip.
#
# The sign is not asserted here. `load_stage2` picks whichever of the four sign combinations minimises
# the disagreement and reports it, so a convention change on either side surfaces as a printed sign
# rather than as a figure full of apparent error.
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
         snr = crop(rd("julia_peak_snr", Float32, js)),
         chip = crop(rd("julia_chip_size", Int32, js)),
         interp = crop(rd("julia_interpolated", Int32, js)))
    rawpx = crop(rd("python_dx", Float32, ps))
    rawpy = crop(rd("python_dy", Float32, ps))

    # The sign that puts the reference in AutoRIFT.jl's sense, measured rather than assumed. Both sides
    # cut their chip from the secondary, so `(+1, +1)` is expected — but asserting it is how a
    # convention change on either side turns into a figure of apparent error instead of a visible flag.
    signs = ((1, 1), (-1, -1), (1, -1), (-1, 1))
    best, err = signs[1], Inf
    for (sx, sy) in signs
        ok = .!isnan.(rawpx) .& .!isnan.(j.dx)
        count(ok) == 0 && continue
        e = median(hypot.(j.dx[ok] .- sx .* rawpx[ok], j.dy[ok] .- sy .* rawpy[ok]))
        e < err && ((best, err) = ((sx, sy), e))
    end
    p = (dx = best[1] .* rawpx, dy = best[2] .* rawpy,
         chip = crop(rd("python_chip_size", Int32, ps)))

    jok, pok = .!isnan.(j.dx), .!isnan.(p.dx)
    both = jok .& pok
    radial = sqrt.((j.dx .- p.dx) .^ 2 .+ (j.dy .- p.dy) .^ 2)
    return (; j, p, jok, pok, both, radial, scalars, shape = (nr, nc), signs = best)
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
        #
        # Scaled to ±4 sub-pixel steps, so the unit of the colour bar is the smallest difference either
        # implementation can express. Anything visible here is at least one step; a point where the two
        # agree exactly is the background colour.
        hmd = heatmap!(mapax((row, 4), "difference in $nm"), mapshow(blank(J .- P, both));
                       colormap = :balance, colorrange = (-4 * FIG_STEP, 4 * FIG_STEP))
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
    #
    # Chip sizes are mapped to their *rank*, not their value. The levels are powers of two, so plotting
    # the value against an evenly divided colour range puts 16 and 32 in the same band of a 16..64
    # scale — 32 becomes invisible, which is exactly the level that answers the shear margins. Ranking
    # gives each level its own colour whatever the sequence.
    chips = sort(filter(>(0), unique(vcat(vec(j.chip), vec(p.chip)))))
    rank = Dict(c => i for (i, c) in enumerate(chips))
    asrank(C) = map(v -> get(rank, v, 0), C)
    crange = (0.5, length(chips) + 0.5)
    cmap = cgrad(:viridis, length(chips); categorical = true)
    for (row, (nm, C, ok)) in enumerate((("AutoRIFT.jl", j.chip, jok), ("autoRIFT.py", p.chip, pok)))
        heatmap!(mapax((row, 1), "$nm chip size"),
                 mapshow(blank(asrank(C), ok .& (C .> 0)));
                 colormap = cmap, colorrange = crange)
        # A legend of named swatches, not a colour bar. Chip size is a small set of discrete levels, and
        # a bar draws them as a continuum with a range between them that no point can take — the ticks
        # then have to undo that impression. One swatch per level says what the map contains.
        Legend(fig[row, 2],
               [PolyElement(; color = cmap[i]) for i in eachindex(chips)],
               ["$c px" for c in chips]; framevisible = false, patchsize = (16, 16),
               rowgap = 8, labelsize = 12, halign = :left, valign = :center,
               tellheight = false, tellwidth = true)
    end

    # Where the two chose a different level. A level difference changes how much the estimate is
    # smoothed, so this map is the mechanism behind much of the residual spread on the previous page —
    # and it is spatially structured, which a histogram of the same fact cannot show.
    # Named swatches rather than a colour bar for the two categorical maps below. A bar implies an
    # ordered range between its ends and stretches to the map's height, so a two- or four-category
    # field reads as a gradient it is not; a legend states the categories and nothing else.
    swatches(pos, colors, labels) =
        Legend(fig[pos...], [PolyElement(; color = c) for c in colors], labels;
               framevisible = false, patchsize = (16, 16), rowgap = 8, labelsize = 12,
               halign = :left, valign = :center, tellheight = false, tellwidth = true)

    samelevel = both .& (j.chip .== p.chip)
    disagree = map((b, s) -> b ? (s ? 0.0 : 1.0) : NaN, both, samelevel)
    heatmap!(mapax((1, 3), @sprintf("chip size: same level at %.1f%% of shared points",
                                    100 * count(samelevel) / max(count(both), 1))),
             mapshow(disagree);
             colormap = cgrad([:gray75, :crimson]; categorical = true), colorrange = (-0.5, 1.5))
    swatches((1, 4), (:gray75, :crimson), ["same level", "different level"])

    # Coverage: which side answered at all. The pyramid and the outlier filter decide *which* points
    # get an answer, so two runs can agree everywhere they overlap and still disagree about most of the
    # grid — the half of the comparison no value statistic can show.
    cov = map((a, b) -> a && b ? 1.0 : a ? 2.0 : b ? 3.0 : 0.0, jok, pok)
    heatmap!(mapax((2, 3), "coverage: which side answered"), mapshow(cov);
             colormap = cgrad([:gray88, :gray30, :dodgerblue, :orangered]; categorical = true),
             colorrange = (-0.5, 3.5))
    swatches((2, 4), (:gray88, :gray30, :dodgerblue, :orangered),
             ["neither", "both", "AutoRIFT.jl only", "autoRIFT.py only"])

    # The one histogram on this page: the signed difference per axis, in units of one sub-pixel step.
    # Symmetric about zero is the claim — a shifted center is a systematic bias, which is a different
    # defect from a wide spread.
    #
    # Steps rather than pixels, and ±8 of them rather than ±1 px. The differences are quantized by the
    # sub-pixel search, so in pixels the whole distribution is a single spike at zero with nothing
    # resolvable either side of it; one bin per step is the finest binning that carries information and
    # the coarsest that loses none.
    #
    # The tail is excluded from the bins rather than clamped into the end one: clamping piles every
    # outlier into a single edge bin, drawing a spike that reads as a real mode. The excluded fraction
    # is stated instead.
    step = 1 / scalars["upsampling"]
    lim = 8
    ax = Axis(fig[1:2, 5]; title = "difference in dx and dy",
              xlabel = "AutoRIFT.jl − autoRIFT.py (steps of 1/$(scalars["upsampling"]) px)",
              ylabel = "points")
    for (v, c, nm) in ((ddx, (:dodgerblue, 0.55), "dx"), (ddy, (:orangered, 0.55), "dy"))
        hist!(ax, filter(x -> abs(x) <= lim, v ./ step);
              bins = range(-lim - 0.5, lim + 0.5; step = 1), color = c, label = nm)
    end
    vlines!(ax, [0]; color = :black, linestyle = :dash)
    axislegend(ax; position = :rt, framevisible = false)
    # Below the legend rather than beside it: the distribution is a spike at zero, so the upper corners
    # are the only clear space and both cannot have it.
    #
    # How often the two agree *exactly*, and how far the tail runs, rather than a fraction clearing some
    # tolerance: the spike at zero is most of the distribution, so a cutoff anywhere in the tail reports
    # its own placement more than it reports the agreement.
    rad = sqrt.(ddx .^ 2 .+ ddy .^ 2)
    text!(ax, 0.97, 0.70;
          text = @sprintf("median %+.4f px (dx)\nmedian %+.4f px (dy)\n\nidentical to the last bit:\n\
                           %.1f%% of dx, %.1f%% of dy\n\nradial tail: p99 %.1f steps,\nmax %.0f steps\n\n\
                           beyond ±%d steps: %.2f%% of dx,\n%.2f%% of dy",
                          median(ddx), median(ddy),
                          100 * mean(ddx .== 0), 100 * mean(ddy .== 0),
                          quantile(rad, 0.99) / step, maximum(rad) / step,
                          lim, 100 * mean(abs.(ddx) .> lim * step),
                          100 * mean(abs.(ddy) .> lim * step)),
          space = :relative, align = (:right, :top), fontsize = 13, color = :gray30)

    colgap!(fig.layout, 8)
    rowgap!(fig.layout, 8)
    # The histogram column is given the width of a map so the page reads as maps-plus-one-summary
    # rather than as two unrelated halves.
    colsize!(fig.layout, 5, Relative(0.26))
    save(path, fig; px_per_unit = 2)
    return path
end

# Page 3: the outputs that are AutoRIFT.jl's alone, which no comparison page can show.
#
# `correlation` and `peak_snr` have no counterpart in the reference's output, and `interpolated` marks
# points whose displacement came from neighbours rather than a surface — so these are properties of the
# result a user gates on, not quantities to diff. Drawn on their own page for that reason.
function outputs_page(d; path = joinpath(FIG_PLOTS, "fig_outputs.png"))
    (; j, jok) = d
    fig = Figure(; size = FIG_SIZE, figure_padding = 12)
    mapax(pos, title) = Axis(fig[pos...]; title, aspect = DataAspect(),
                             xticksvisible = false, yticksvisible = false,
                             xticklabelsvisible = false, yticklabelsvisible = false)

    # Three maps in a row, each with its own colorbar. The maps are square and the page is landscape, so
    # three across fills it without shrinking any of them below legibility.
    #
    # A colour bar is tied to its map's own height rather than given `height = Relative(...)`. The
    # relative form is a fraction of the *grid cell*, and a `DataAspect` map leaves a cell far taller
    # than the square it draws in, so the bar overran the panel it labels. Linking the axis makes the
    # bar exactly as tall as the data it describes, whatever the layout does around it.
    barof(pos, hm, label, ax) =
        Colorbar(fig[pos...], hm; label, width = 12, tellheight = false,
                 halign = :left, height = @lift(($(ax.scene.viewport)).widths[2]))

    measured = jok .& .!isnan.(j.corr)
    axc = mapax((1, 1), "correlation (peak height)")
    hm = heatmap!(axc, mapshow(blank(j.corr, measured)); colormap = :magma, colorrange = (0, 1))
    barof((1, 2), hm, "ZNCC peak", axc)

    # `peak_snr` is zero where the peak lay against the search boundary, so the count of railed points is
    # stated in the title rather than mapped separately: on a scene whose flow the radius covers there
    # are none, and an all-one-colour panel reads as a broken plot rather than as an empty set.
    snrok = jok .& .!isnan.(j.snr)
    snr = map((v, k) -> k ? Float64(v) : NaN, j.snr, snrok)
    finite = filter(isfinite, vec(snr))
    hi = isempty(finite) ? 1.0 : quantile(finite, 0.99)
    nrail = count(iszero, finite)
    axs = mapax((1, 3), @sprintf("peak SNR; %d of %d at the search boundary",
                                 nrail, length(finite)))
    hm2 = heatmap!(axs, mapshow(snr); colormap = :viridis, colorrange = (0, hi))
    barof((1, 4), hm2, "peak above background (sd)", axs)

    # Two categories, so a legend of two swatches rather than a colour bar. A bar implies an ordered
    # range between its ends and sizes itself to the map's height, which for a Boolean leaves a tall
    # gradient labelled `true`/`false` — and those labels name the field rather than what it says about
    # the point. Named swatches state the two conditions, and the fraction interpolated goes in the
    # title, which is the number a reader wants and no legend can carry.
    interp = map((v, k) -> k ? Float64(v != 0) : NaN, j.interp, jok)
    nint = count(i -> jok[i] && j.interp[i] != 0, eachindex(j.interp))
    hm4 = heatmap!(mapax((1, 5), @sprintf("how each point got its displacement (%.1f%% filled)",
                                          100 * nint / max(count(jok), 1))),
                   mapshow(interp);
                   colormap = cgrad([:gray80, :dodgerblue]; categorical = true),
                   colorrange = (-0.5, 1.5))
    Legend(fig[1, 6],
           [PolyElement(; color = :gray80), PolyElement(; color = :dodgerblue)],
           ["measured on its own\ncorrelation surface", "filled from\nits neighbours"];
           framevisible = false, patchsize = (20, 20), rowgap = 12, labelsize = 15,
           halign = :left, valign = :center, tellheight = false, tellwidth = true)

    colgap!(fig.layout, 8)
    rowgap!(fig.layout, 8)
    save(path, fig; px_per_unit = 2)
    return path
end

# Both pages, and the statistics the report quotes in text.
function accuracy_figures()
    mkpath(FIG_PLOTS)
    d = load_stage2()
    (; j, both, radial) = d
    e = radial[both]
    # Reported without a pass/fail threshold. Most shared points agree to the last bit, so a
    # "fraction within X px" says only where X happens to fall in a tail that is already narrower than
    # one step of the sub-pixel search — move X a little and the headline moves a lot, which is a
    # property of the cutoff rather than of the two implementations. What does carry information: how
    # often they agree *exactly*, and how far the tail reaches, both stated in sub-pixel steps.
    step = 1 / d.scalars["upsampling"]
    stats = (; shared = count(both),
             jl_only = count(d.jok .& .!d.pok), ref_only = count(d.pok .& .!d.jok),
             median = median(e), p95 = quantile(e, 0.95), p99 = quantile(e, 0.99),
             pmax = maximum(e),
             exact = mean(e .== 0), within_step = mean(e .<= step),
             steps_p99 = quantile(e, 0.99) / step, steps_max = maximum(e) / step,
             bias_dx = median(j.dx[both] .- d.p.dx[both]),
             bias_dy = median(j.dy[both] .- d.p.dy[both]),
             cor_dx = cor(j.dx[both], d.p.dx[both]), cor_dy = cor(j.dy[both], d.p.dy[both]),
             npix = d.scalars["npix"], upsampling = d.scalars["upsampling"],
             chip = d.scalars["chip"], chip_max = d.scalars["chip_max"],
             spacing = d.scalars["grid_spacing"], radius = d.scalars["radius"],
             signs = d.signs)
    return (; heatmaps = heatmap_page(d), histograms = histogram_page(d),
            outputs = outputs_page(d), stats)
end

# Standalone: draw both pages and print what they show.
if abspath(PROGRAM_FILE) == @__FILE__
    r = accuracy_figures()
    s = r.stats
    @printf("shared %d  exact %.1f%%  median %.4f px  bias %+.4f/%+.4f  p99 %.2f steps  max %.1f steps  sign %s\n",
            s.shared, 100 * s.exact, s.median, s.bias_dx, s.bias_dy,
            s.steps_p99, s.steps_max, string(s.signs))
    println("wrote ", r.heatmaps, "\n      ", r.histograms, "\n      ", r.outputs)
end
