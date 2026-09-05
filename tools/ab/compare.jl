# Compare the two sides of a stage-1 bundle and plot the disagreement.
#
# Reads `julia_dx`/`julia_dy` and `python_dx`/`python_dy` from the bundle, puts both in the same
# convention, and writes one PNG per iteration to `tools/ab/plots/`.
#
# **Convention.** AutoRIFT.jl reports displacement secondary-to-reference; the reference reports
# feature motion, which is its negative. Both axes flip together, and `stage1_python.py` has
# **already applied** that flip by the time it writes the bundle — it negates `dy` to undo the
# reference's cartesian flip, and its chip/window assignment accounts for the rest. So the planes on
# disk are already in AutoRIFT.jl's convention and this reads them as written.
#
# Negating here as well flips both axes back, which turns a bit-identical result into a whole-pixel
# median bias and a correlation of -0.9996. That reads as an accuracy difference rather than as the
# convention error it is, which is why `stage1_python.py` also compares in-process: two independent
# paths to the same number, and a disagreement between them means the sign is wrong here.
#
#   julia --project=tools/ab tools/ab/compare.jl [tag]

using CairoMakie, Statistics, Printf

const D = joinpath(@__DIR__, "stage1")
const PLOTS = joinpath(@__DIR__, "plots")
const TAG = length(ARGS) >= 1 ? ARGS[1] : "stage1"

# One upsampling step at `upsampling = 16`, the smallest difference either side can express. A
# disagreement below it is about which of two adjacent representable values a peak rounded to, not
# about position.
#
# 0.2 px was the threshold and it no longer discriminates: stage 1 is bit-identical at every chip
# size, so any tolerance above zero reports 100%. Reading the *exact* fraction is what makes a
# regression visible — it drops off 100% the moment the two stop matching to the bit.
const TOL = 1 / 16

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
    # Already in AutoRIFT.jl's convention when written, per the note above.
    pdx = read_bin("python_dx", Float32, sz)
    pdy = read_bin("python_dy", Float32, sz)

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
        @printf("%-7s exact %5.1f%%  within step %5.1f%%  median %7.4f  p99 %7.4f  max %8.4f\n",
                nm, 100 * mean(iszero, e), 100 * mean(e .<= TOL),
                median(e), quantile(e, 0.99), maximum(e))
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
    println("\n  gate      n    exact   within step      p99      max")
    for g in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
        sel = ok .& (jc .>= g)
        n = count(sel)
        n == 0 && continue
        e = er[sel]
        @printf("  >=%.1f  %5d  %6.1f%%       %6.1f%%  %7.4f  %7.4f\n",
                g, n, 100 * mean(iszero, e), 100 * mean(e .<= TOL),
                quantile(e, 0.99), maximum(e))
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
        ax2 = Axis(fig[2, col]; title = "reference $nm", aspect = DataAspect())
        heatmap!(ax2, nanify(P); colormap = :balance, colorrange = dlim)
    end

    # Scaled to one upsampling step, so the map distinguishes exact agreement from a tie broken the
    # other way rather than washing both out against a whole-pixel outlier.
    axd = Axis(fig[1, 3]; title = "radial |difference|", aspect = DataAspect())
    hmd = heatmap!(axd, nanify(er); colormap = :inferno, colorrange = (0, max(TOL, 0.01)),
                   highclip = :cyan)
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
