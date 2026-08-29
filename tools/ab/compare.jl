# Compare the two sides of a stage-1 bundle and plot the disagreement.
#
# Reads `julia_dx`/`julia_dy` and `python_dx`/`python_dy` from the bundle, puts both in the same
# convention, and writes one PNG per iteration to `tools/ab/plots/`.
#
# **Convention.** AutoRIFT.jl reports displacement secondary-to-reference; the reference reports
# feature motion, which is its negative. Both axes flip together, so the comparison negates the
# reference's pair rather than either component alone. A one-axis flip would be a bug in one of the
# two implementations; a both-axis flip is the documented difference, and `test/realdata.jl` already
# applies it when forming velocities.
#
#   julia --project=tools/ab tools/ab/compare.jl [tag]

using CairoMakie, Statistics, Printf

const D = joinpath(@__DIR__, "stage1")
const PLOTS = joinpath(@__DIR__, "plots")
const TAG = length(ARGS) >= 1 ? ARGS[1] : "stage1"

# The tolerance the comparison is judged against: sub-pixel matching noise for this correlator.
const TOL = 0.2

function manifest()
    shapes = Dict{String,Tuple{Int,Int}}()
    scalars = Dict{String,Int}()
    for line in eachline(joinpath(D, "manifest.txt"))
        line = strip(line)
        (isempty(line) || startswith(line, "#")) && continue
        parts = split(line)
        if length(parts) == 2
            scalars[parts[1]] = parse(Int, parts[2])
        else
            dims = parse.(Int, split(parts[3], "x"))
            shapes[parts[1]] = (dims[1], dims[2])
        end
    end
    return shapes, scalars
end

read_bin(name, T, dims) =
    reshape(collect(reinterpret(T, read(joinpath(D, name * ".bin")))), dims)

function main()
    mkpath(PLOTS)
    shapes, scalars = manifest()
    sz = shapes["julia_dx"]

    jdx = read_bin("julia_dx", Float32, sz)
    jdy = read_bin("julia_dy", Float32, sz)
    jc = read_bin("julia_correlation", Float32, sz)
    # Negated into AutoRIFT.jl's convention, per the note above.
    pdx = .-read_bin("python_dx", Float32, sz)
    pdy = .-read_bin("python_dy", Float32, sz)

    ok = .!isnan.(jdx) .& .!isnan.(jdy) .& .!isnan.(pdx) .& .!isnan.(pdy)
    ex = abs.(jdx .- pdx)
    ey = abs.(jdy .- pdy)
    er = sqrt.((jdx .- pdx) .^ 2 .+ (jdy .- pdy) .^ 2)

    chip = get(scalars, "chip", 0)
    radius = get(scalars, "radius", 0)
    up = get(scalars, "upsampling", 16)

    exo, eyo, ero = ex[ok], ey[ok], er[ok]
    @printf("chip %d  radius %d  upsampling %d  grid %dx%d\n", chip, radius, up, sz...)
    @printf("comparable        %d / %d\n", count(ok), length(ok))
    @printf("coverage  julia %d  python %d\n", count(!isnan, jdx), count(!isnan, pdx))
    for (nm, e) in (("|ddx|", exo), ("|ddy|", eyo), ("radial", ero))
        @printf("%-7s median %7.4f  p95 %7.4f  max %8.4f  within %.1f px %5.1f%%\n",
                nm, median(e), quantile(e, 0.95), maximum(e), TOL, 100 * mean(e .<= TOL))
    end
    # One upsampling step is the finest distinction either side can draw, so a residual at that
    # scale is a tie broken differently rather than a disagreement about the peak.
    step = 1 / up
    @printf("within one upsampling step (%.4f px): dx %.1f%%  dy %.1f%%\n",
            step, 100 * mean(exo .<= step * 1.01), 100 * mean(eyo .<= step * 1.01))
    if count(ok) > 2
        @printf("corr dx %.5f   corr dy %.5f\n",
                cor(jdx[ok], pdx[ok]), cor(jdy[ok], pdy[ok]))
        @printf("bias  dx %+.4f   dy %+.4f  (julia - python, signed median)\n",
                median(jdx[ok] .- pdx[ok]), median(jdy[ok] .- pdy[ok]))
    end

    # Gated by correlation. A 0.2 px claim is about sub-pixel matching noise, which presupposes a
    # match: where the surface has no dominant peak both sides pick from noise, and they are then
    # being asked to agree about something neither one measured. The gate says at what peak
    # strength the two become interchangeable, which is the useful form of the answer.
    println("\n  gate      n   median      p95   within $(TOL) px")
    for g in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
        sel = ok .& (jc .>= g)
        n = count(sel)
        n == 0 && continue
        e = er[sel]
        @printf("  >=%.1f  %5d  %7.4f  %7.4f  %6.1f%%\n",
                g, n, median(e), quantile(e, 0.95), 100 * mean(e .<= TOL))
    end

    fig = Figure(; size = (1500, 950))
    Label(fig[0, 1:3],
          @sprintf("%s — chip %d px, radius %d, upsampling %d — median radial %.4f px",
                   TAG, chip, radius, up, count(ok) > 0 ? median(ero) : NaN);
          fontsize = 19, font = :bold)

    # `map` rather than a comprehension over `eachindex`: the comprehension returns a vector and
    # `heatmap!` needs the grid shape.
    nanify(A) = map((v, keep) -> keep ? Float64(v) : NaN, A, ok)

    # Shared displacement limits, so the two fields are compared by eye rather than by colorbar.
    dlim = let v = filter(isfinite, vcat(nanify(jdx), nanify(pdx), nanify(jdy), nanify(pdy)))
        isempty(v) ? (-1.0, 1.0) : (-maximum(abs.(v)), maximum(abs.(v)))
    end

    for (col, (nm, J, P)) in enumerate((("dx", jdx, pdx), ("dy", jdy, pdy)))
        ax1 = Axis(fig[1, col]; title = "AutoRIFT.jl $nm", aspect = DataAspect())
        hm = heatmap!(ax1, nanify(J); colormap = :balance, colorrange = dlim)
        Colorbar(fig[1, col], hm; label = "px", halign = :right, width = 12, tellwidth = false)
        ax2 = Axis(fig[2, col]; title = "reference $nm (negated)", aspect = DataAspect())
        heatmap!(ax2, nanify(P); colormap = :balance, colorrange = dlim)
    end

    # The difference, on a scale centred at the tolerance so "inside 0.2 px" reads at a glance.
    axd = Axis(fig[1, 3]; title = "radial |difference|", aspect = DataAspect())
    hmd = heatmap!(axd, nanify(er); colormap = :inferno, colorrange = (0, max(TOL * 2, 0.01)))
    Colorbar(fig[1, 3], hmd; label = "px", halign = :right, width = 12, tellwidth = false)

    axs = Axis(fig[2, 3]; title = "AutoRIFT.jl vs reference", xlabel = "reference px",
               ylabel = "AutoRIFT.jl px")
    if count(ok) > 0
        scatter!(axs, pdx[ok], jdx[ok]; markersize = 5, label = "dx")
        scatter!(axs, pdy[ok], jdy[ok]; markersize = 5, label = "dy")
        lo, hi = extrema(vcat(pdx[ok], pdy[ok]))
        lines!(axs, [lo, hi], [lo, hi]; color = :black, linestyle = :dash)
        axislegend(axs; position = :lt)
    end

    axh = Axis(fig[3, 1]; title = "radial difference", xlabel = "px", ylabel = "count")
    if count(ok) > 0
        hist!(axh, ero; bins = 40, color = :steelblue)
        vlines!(axh, [TOL]; color = :red, linestyle = :dash)
    end

    axc = Axis(fig[3, 2]; title = "difference vs correlation", xlabel = "AutoRIFT.jl correlation",
               ylabel = "radial difference (px)")
    if count(ok) > 0
        scatter!(axc, jc[ok], ero; markersize = 5, color = :darkorange)
        hlines!(axc, [TOL]; color = :red, linestyle = :dash)
    end

    # The cumulative curve answers the actual question -- what fraction is inside a tolerance --
    # for every tolerance at once, rather than only at 0.2.
    axcdf = Axis(fig[3, 3]; title = "fraction within tolerance", xlabel = "tolerance (px)",
                 ylabel = "fraction")
    if count(ok) > 0
        ts = range(0, max(1.0, quantile(ero, 0.99)); length = 200)
        lines!(axcdf, ts, [mean(ero .<= t) for t in ts]; color = :seagreen, linewidth = 2)
        vlines!(axcdf, [TOL]; color = :red, linestyle = :dash)
        ylims!(axcdf, 0, 1.02)
    end

    path = joinpath(PLOTS, TAG * ".png")
    save(path, fig)
    println("wrote ", path)
    return path
end

main()
