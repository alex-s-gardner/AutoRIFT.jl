# Side-by-side maps of a golden case: the reference, AutoRIFT.jl, and their difference.
#
#   julia --project=tools/golden -t 8 tools/golden/compare_figures.jl LC08_L1TP_009011_20200703
#
# Same layout discipline as `tools/ab/bench_figures.jl`, which it deliberately mirrors so the golden
# comparison and the benchmark are read the same way. Three things carry over because each one changes
# what the figure says:
#
#   **One colour scale across both sides.** Per-panel scaling hides exactly what the figure is for: two
#   fields differing by a constant look identical when each is normalised to its own range.
#
#   **The difference is signed, on a diverging scale.** A magnitude on a sequential scale cannot
#   distinguish a field of +0.1 px from one of ±0.1 px, and that distinction is the difference between
#   a bias and tie-breaking. Exact agreement is the background colour either way.
#
#   The range is `dlim` pixels, ±1 by default. Set it small (a few times the quantization step) to see
#   whether the *sign* of a near-threshold residual is organised, and larger to see how far the worst
#   disagreements actually reach — a tight range saturates them all to the same colour and makes a
#   3-pixel error indistinguishable from a 0.3-pixel one.
#
#   **`NaN` reads as blank.** A zero displacement is a real measurement; an unmeasured point is not, and
#   the two must not look alike.
#
# The reference column is first because it is the thing being matched.

include(joinpath(@__DIR__, "correlator.jl"))

using CairoMakie, Printf, Serialization, Statistics
using AutoRIFT: upsampling

const FIG_SIZE = (1560, 929)

# A grid array as `heatmap!` needs it to appear in image orientation: x horizontal, y vertical,
# increasing downward. `heatmap!(A)` treats A's *first* index as the x axis, so a `[row, col]` array
# passed directly draws transposed — rows run horizontally. Transposing fixes the axes and reversing
# the resulting columns puts row 1 at the top, where the northernmost row of a north-up scene belongs.
# Without this a feature elongated along y appears elongated along x, which inverts the reading of any
# anisotropy.
mapshow(A) = reverse(permutedims(decimate(A)); dims = 2)

blank(A, keep) = map((v, k) -> k ? Float64(v) : NaN, A, keep)

# Nearest-neighbour decimation to roughly `target` cells on the long axis.
#
# A production grid is over a thousand cells across and the figure allocates a few hundred pixels per
# panel, so drawing every cell rasterises far more than the page can show — minutes of rendering for
# detail no reader can see. Nearest, not averaged: averaging a field that is `NaN` where nothing was
# measured would spread those `NaN`s over their neighbours, and averaging a *difference* field would
# cancel a symmetric residual into a misleadingly clean map.
function decimate(A, target::Integer = 700)
    s = max(1, cld(maximum(size(A)), target))
    return s == 1 ? A : A[1:s:end, 1:s:end]
end

"""
    fast_flow_center(dx, dy) -> (row, col)

Grid index of the fastest-moving part of the scene.

The speed-weighted centroid of the fastest 0.5% of points, rather than the single fastest point: one
point is a noisy match away from being anywhere, while the centroid follows the body of the outlet.
Weighting by speed keeps it on the trunk when a scene holds one dominant outlet and several slower
ones — an unweighted centroid of a threshold set drifts toward whichever tributary has more area.

A scene with two comparably fast outlets has no single answer here, and the centroid will sit between
them. Pass `center` explicitly for that case; the log line reports what was chosen either way.
"""
function fast_flow_center(dx, dy)
    speed = map((a, b) -> (isnan(a) || isnan(b)) ? NaN32 : sqrt(a^2 + b^2), dx, dy)
    ok = filter(!isnan, vec(speed))
    isempty(ok) && return (size(dx, 1) ÷ 2, size(dx, 2) ÷ 2)
    thr = quantile(ok, 0.995)
    wi = 0.0; wj = 0.0; w = 0.0
    for idx in CartesianIndices(speed)
        v = speed[idx]
        (isnan(v) || v <= thr) && continue
        wi += idx[1] * v; wj += idx[2] * v; w += v
    end
    w == 0 && return (size(dx, 1) ÷ 2, size(dx, 2) ÷ 2)
    return (round(Int, wi / w), round(Int, wj / w))
end

"""
    cached_run(name, k, grid, kw) -> NamedTuple

AutoRIFT.jl's answer for a golden case, from disk if a run with these settings is already there.

Correlating a production scene costs minutes — over ten on the larger Landsat cases — and a figure is
something you iterate on. Re-deriving the same field for every change of colour scale or panel makes
plotting the slow step, and the whole reason for plotting is to look at a result quickly.

The cache key includes the settings, so changing a parameter re-runs rather than silently plotting the
previous configuration. That is the failure mode worth designing against: a stale figure that looks
current is worse than no figure.
"""
function cached_run(name::AbstractString, k::Capture, grid, kw)
    dir = get(ENV, "AUTORIFT_GOLDEN_CACHE",
              joinpath(homedir(), "data", "autorift", "tests", "golden_tests"))
    key = string(hash((name, kw, size(grid.x))), base = 16)
    path = joinpath(dir, "runs", "field_$(first(name, 24))_$key.jls")

    if isfile(path)
        @info "reusing cached correlator run" path
        return Serialization.deserialize(path)
    end
    @info "correlating (no cached run for these settings)" npoints = length(grid.x)
    t = @elapsed out = autorift(k.arrays["in_I2"], k.arrays["in_I1"], grid; kw...)
    @info "correlated" seconds = round(t; digits = 1)
    mkpath(dirname(path))
    # Only the fields a figure or a statistic reads. Serialising the whole result would write the
    # correlation surfaces too, which are large and never plotted.
    slim = (; out.dx, out.dy, out.chip_size, out.correlation, out.peak_ratio)
    Serialization.serialize(path, slim)
    return slim
end

"""
    compare_figure(name; path) -> String

Write the three-column comparison for one golden case.

Rows are `dx`, `dy` and chip size. The first two share a colour scale set by the 99th percentile of
both sides together, so one outlier cannot flatten the whole field. Chip size is categorical, so its
difference column shows *where* the levels disagree rather than by how much — a signed difference in
level number would imply an ordering the merge does not have.

`dlim` is the half-range of the difference panels in pixels.
"""
function compare_figure(name::AbstractString;
                        path = joinpath(tempdir(), "golden_$(first(name, 24)).png"),
                        zoom::Union{Nothing,Integer} = nothing,
                        center::Union{Nothing,Tuple{Integer,Integer}} = nothing,
                        dlim::Real = 1.0,
                        run::Integer = 100)
    # `run` because a capture is not always at the default: the radar cases live at 200, and the
    # optical ones at 100 or 200 depending on when they were taken. Reading the wrong number fails on
    # a missing directory rather than silently mapping a different comparison.
    k = read_capture(only(cases(name)); n = run)
    grid = pointset_from_capture(k)
    kw = kwargs_from_capture(k)
    out = cached_run(name, k, grid, kw)

    rdx = k.arrays["out_Dx"]
    rdy = k.arrays["out_Dy"]
    rcs = k.arrays["out_ChipSizeX"]

    ny = min(size(out.dx, 1), size(rdx, 1))
    nx = min(size(out.dx, 2), size(rdx, 2))
    jdx = out.dx[1:ny, 1:nx]
    jdy = out.dy[1:ny, 1:nx]
    jcs = Float32.(out.chip_size[1:ny, 1:nx])
    # The reference returns `Dy` up-positive and AutoRIFT.jl row-positive, so one negation makes the
    # two comparable (`correlator.jl` measures which sign wins rather than asserting it).
    pdx = rdx[1:ny, 1:nx]
    pdy = .-rdy[1:ny, 1:nx]
    pcs = rcs[1:ny, 1:nx]

    # A whole-scene panel decimates roughly 3:1, which averages the outlet — the place the residual
    # actually lives — down to a few cells. Cropping to it shows every grid point at full resolution,
    # where whole-scene statistics cannot say whether the disagreement there is structured or
    # scattered.
    if zoom !== nothing
        ci, cj = center === nothing ? fast_flow_center(jdx, jdy) : center
        rows = max(1, ci - zoom):min(ny, ci + zoom)
        cols = max(1, cj - zoom):min(nx, cj + zoom)
        @info "zooming" center = (ci, cj) rows cols
        jdx = jdx[rows, cols]; jdy = jdy[rows, cols]; jcs = jcs[rows, cols]
        pdx = pdx[rows, cols]; pdy = pdy[rows, cols]; pcs = pcs[rows, cols]
    end

    jok = .!isnan.(jdx)
    pok = .!isnan.(pdx)
    both = jok .& pok

    step = 1 / upsampling(first(kw.subpixel))

    fig = Figure(; size = FIG_SIZE, figure_padding = 12)
    mapax(pos, title) = Axis(fig[pos...]; title, aspect = DataAspect(),
                             xticksvisible = false, yticksvisible = false,
                             xticklabelsvisible = false, yticklabelsvisible = false)

    vals = filter(isfinite, vcat(vec(blank(jdx, jok)), vec(blank(pdx, pok)),
                                 vec(blank(jdy, jok)), vec(blank(pdy, pok))))
    lim = isempty(vals) ? 1.0 : quantile(abs.(vals), 0.99)

    for (row, (nm, J, P)) in enumerate((("dx", jdx, pdx), ("dy", jdy, pdy)))
        hm = heatmap!(mapax((row, 1), "autoRIFT.py $nm"), mapshow(blank(P, pok));
                      colormap = :balance, colorrange = (-lim, lim))
        heatmap!(mapax((row, 2), "AutoRIFT.jl $nm"), mapshow(blank(J, jok));
                 colormap = :balance, colorrange = (-lim, lim))
        Colorbar(fig[row, 3], hm; label = "$nm (px)", width = 12)

        d = blank(J .- P, both)
        agree = 100 * count(iszero, filter(isfinite, vec(d))) / max(count(both), 1)
        hmd = heatmap!(mapax((row, 4), @sprintf("difference in %s (%.1f%% exact)", nm, agree)),
                       mapshow(d); colormap = :balance, colorrange = (-dlim, dlim))
        Colorbar(fig[row, 5], hmd; label = "AutoRIFT.jl − autoRIFT.py (px)", width = 12)
    end

    # Chip size, where a disagreement means the two describe different footprints rather than
    # different displacements.
    chips = sort(unique(Int.(filter(>(0), vec(pcs)))))
    crange = (minimum(chips) / 1.2, maximum(chips) * 1.05)
    hmc = heatmap!(mapax((3, 1), "autoRIFT.py chip size"), mapshow(blank(pcs, pok));
                   colormap = :viridis, colorrange = crange)
    heatmap!(mapax((3, 2), "AutoRIFT.jl chip size"), mapshow(blank(jcs, jok));
             colormap = :viridis, colorrange = crange)
    Colorbar(fig[3, 3], hmc; label = "chip size (px)", width = 12)

    # Disagreement is the ink. Agreement is the overwhelming majority, so drawing it dark makes a
    # 96%-agreeing panel read as a 96%-*dis*agreeing one — the reverse of what the title says.
    same = 100 * count(both .& (jcs .== pcs)) / max(count(both), 1)
    lvl = map((j, p, b) -> b ? (j == p ? 0.0 : 1.0) : NaN, jcs, pcs, both)
    heatmap!(mapax((3, 4), @sprintf("level disagrees: %.1f%%", 100 - same)),
             mapshow(lvl); colormap = Reverse(:grays), colorrange = (0, 1))

    # Coverage in the last cell rather than another colour bar: a point one side answered alone is a
    # different finding from a value disagreement, and it has no magnitude to put on a scale.
    #
    # The counts go in the title on two lines; one line overflows the panel and prints outside its
    # axis, which puts a number where a reader cannot tell which panel it belongs to.
    cov = map((j, p) -> j && p ? NaN : (j ? 1.0 : (p ? -1.0 : NaN)), jok, pok)
    heatmap!(mapax((3, 5), @sprintf("coverage\njl %d / py %d",
                                    count(jok .& .!pok), count(pok .& .!jok))),
             mapshow(cov); colormap = :balance, colorrange = (-1, 1))

    colgap!(fig.layout, 10)
    rowgap!(fig.layout, 8)
    save(path, fig; px_per_unit = 1)
    return path
end

function main(args)
    name = isempty(args) ? "LC08_L1TP_009011_20200703" : args[1]
    dir = get(ENV, "AUTORIFT_GOLDEN_FIGS", tempdir())

    # `--zoom N` crops to a 2N+1 box on the fastest flow; `--at ROW,COL` overrides where that is.
    zoom = nothing
    i = findfirst(==("--zoom"), args)
    i === nothing || (zoom = parse(Int, args[i + 1]))
    center = nothing
    j = findfirst(==("--at"), args)
    j === nothing || (center = Tuple(parse.(Int, split(args[j + 1], ","))))
    dlim = 1.0
    d = findfirst(==("--dlim"), args)
    d === nothing || (dlim = parse(Float64, args[d + 1]))
    run = 100
    rr = findfirst(==("--run"), args)
    rr === nothing || (run = parse(Int, args[rr + 1]))

    tag = zoom === nothing ? "" : "_zoom$zoom"
    p = compare_figure(name; path = joinpath(dir, "golden_$(first(name, 24))$tag.png"),
                       zoom, center, dlim, run)
    println("wrote ", p)
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
