# Compare the two sides of a stage-2 bundle: values, coverage, and which level answered.
#
# Grid shapes differ by construction. `runAutorift` truncates the grid to a whole multiple of
# `ChipSizeMaxX / ChipSize0X` before correlating (`autoRIFT.py:884-891`), so the reference's grid is
# the top-left `rlim x clim` sub-block of the one it was handed. The comparison therefore takes
# AutoRIFT.jl's matching sub-block rather than resampling either side.
#
#   julia --project=tools/ab tools/ab/compare2.jl [tag]

using CairoMakie, Statistics, Printf

const D = joinpath(@__DIR__, "stage2")
const PLOTS = joinpath(@__DIR__, "plots")
const TAG = length(ARGS) >= 1 ? ARGS[1] : "stage2"

# One upsampling step at `upsampling = 16`: the smallest difference either side can express, so a
# disagreement below it is not a disagreement about position but about which of two adjacent
# representable values a peak rounded to.
#
# 0.2 px was the threshold here and it no longer discriminates: p95 is 0.017 px, and every gate from
# 0 to 0.5 correlation passes between 99.4% and 100%. A tolerance that everything clears measures the
# tolerance, not the code. At one step the same comparison separates 97.3% from 98.8%, and the
# statistic worth reading is the one below — the fraction agreeing *exactly*, which is 77.4%.
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

python_shape() = Tuple(parse.(Int, split(strip(read(joinpath(D, "python_shape.txt"), String)))))

function main()
    mkpath(PLOTS)
    shapes, scalars = manifest()
    jsz = shapes["julia_dx"]
    psz = python_shape()

    jdx_f = read_bin("julia_dx", Float32, jsz)
    jdy_f = read_bin("julia_dy", Float32, jsz)
    jc_f = read_bin("julia_correlation", Float32, jsz)
    jcs_f = read_bin("julia_chip_size", Int32, jsz)

    # Read as written. `stage2_python.py` has already put both axes in AutoRIFT.jl's convention —
    # it negates `Dy` there to undo the reference's final cartesian flip and leaves `Dx` alone.
    # Negating again here flips both back: it turns a bit-identical `dx` into a 0.75 px median bias
    # and a correlation of -0.9996, which reads as an accuracy difference rather than as the
    # convention error it is.
    pdx = read_bin("python_dx", Float32, psz)
    pdy = read_bin("python_dy", Float32, psz)
    pcs = read_bin("python_chip_size", Int32, psz)

    # The reference's grid is the top-left sub-block of the one it was given, so this is a crop
    # rather than an interpolation: the two arrays index the same grid points.
    nr, nc = min(jsz[1], psz[1]), min(jsz[2], psz[2])
    jdx, jdy, jc, jcs = jdx_f[1:nr, 1:nc], jdy_f[1:nr, 1:nc], jc_f[1:nr, 1:nc], jcs_f[1:nr, 1:nc]
    pdx, pdy, pcs = pdx[1:nr, 1:nc], pdy[1:nr, 1:nc], pcs[1:nr, 1:nc]

    jok = .!isnan.(jdx)
    pok = .!isnan.(pdx)
    both = jok .& pok
    er = sqrt.((jdx .- pdx) .^ 2 .+ (jdy .- pdy) .^ 2)

    @printf("compared on the shared %dx%d block (julia %dx%d, reference %dx%d)\n",
            nr, nc, jsz..., psz...)
    @printf("coverage   julia %5d (%.1f%%)   reference %5d (%.1f%%)   both %5d\n",
            count(jok), 100 * mean(jok), count(pok), 100 * mean(pok), count(both))
    @printf("julia only %5d      reference only %5d\n",
            count(jok .& .!pok), count(pok .& .!jok))

    println("\nlevel that answered (shared block):")
    for cs in sort(union(unique(jcs), unique(pcs)))
        @printf("  chip %3d   julia %6d   reference %6d\n",
                cs, count(==(cs), jcs), count(==(cs), pcs))
    end

    if count(both) > 2
        e = er[both]
        println()
        for (nm, v) in (("|ddx|", abs.(jdx[both] .- pdx[both])),
                        ("|ddy|", abs.(jdy[both] .- pdy[both])),
                        ("radial", e))
            # `exact` first, because it is the statistic that still moves: most points agree to the
            # bit, and what a change to the correlator does is move points out of that column.
            @printf("%-7s exact %5.1f%%  within step %5.1f%%  median %7.4f  p99 %8.4f  max %9.4f\n",
                    nm, 100 * mean(iszero, v), 100 * mean(v .<= TOL),
                    median(v), quantile(v, 0.99), maximum(v))
        end
        @printf("bias   dx %+.4f   dy %+.4f  (julia - reference, signed median)\n",
                median(jdx[both] .- pdx[both]), median(jdy[both] .- pdy[both]))
        @printf("corr   dx %.5f   dy %.5f\n",
                cor(jdx[both], pdx[both]), cor(jdy[both], pdy[both]))

        # Gated by peak strength, for the reason compare.jl records: below a real peak the two are
        # being asked to agree about something neither measured.
        println("\n  gate      n    exact   within step      p99      max")
        for g in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
            sel = both .& (jc .>= g)
            count(sel) == 0 && continue
            v = er[sel]
            @printf("  >=%.1f  %5d  %6.1f%%       %6.1f%%  %7.4f  %7.4f\n",
                    g, count(sel), 100 * mean(iszero, v), 100 * mean(v .<= TOL),
                    quantile(v, 0.99), maximum(v))
        end
        # Restricted to points both sides answered at the same chip size: a level difference
        # changes how much the estimate is smoothed, so comparing across levels measures that
        # rather than the pipeline.
        same = both .& (jcs .== pcs)
        if count(same) > 2
            v = er[same]
            @printf("\nsame level only (%d points): exact %.1f%%  within step %.1f%%  p99 %.4f\n",
                    count(same), 100 * mean(iszero, v), 100 * mean(v .<= TOL),
                    quantile(v, 0.99))
        end
    end

    nanify(A, m) = map((v, keep) -> keep ? Float64(v) : NaN, A, m)

    fig = Figure(; size = (1500, 1250))
    Label(fig[0, 1:3],
          @sprintf("%s — chips %d–%d, spacing %d, radius %d — %d shared points, median radial %.4f px",
                   TAG, scalars["chip"], scalars["chip_max"], scalars["grid_spacing"],
                   scalars["radius"], count(both),
                   count(both) > 0 ? median(er[both]) : NaN);
          fontsize = 18, font = :bold)

    dlim = let v = filter(isfinite, vcat(vec(nanify(jdx, jok)), vec(nanify(pdx, pok))))
        isempty(v) ? (-1.0, 1.0) : (-maximum(abs.(v)), maximum(abs.(v)))
    end

    for (col, (nm, J, P)) in enumerate((("dx", jdx, pdx), ("dy", jdy, pdy)))
        ax1 = Axis(fig[1, col]; title = "AutoRIFT.jl $nm", aspect = DataAspect())
        hm = heatmap!(ax1, nanify(J, jok); colormap = :balance, colorrange = dlim)
        Colorbar(fig[1, col], hm; label = "px", halign = :right, width = 12, tellwidth = false)
        ax2 = Axis(fig[2, col]; title = "reference $nm", aspect = DataAspect())
        heatmap!(ax2, nanify(P, pok); colormap = :balance, colorrange = dlim)
    end

    # Scaled to one upsampling step, so the map distinguishes exact agreement from a tie broken the
    # other way rather than washing both out against a whole-pixel outlier. `highclip` marks the few
    # points past a step, which at 0.6% of the grid would otherwise be invisible.
    axd = Axis(fig[1, 3]; title = "radial |difference|", aspect = DataAspect())
    hmd = heatmap!(axd, nanify(er, both); colormap = :inferno, colorrange = (0, max(TOL, 0.01)),
                   highclip = :cyan)
    Colorbar(fig[1, 3], hmd; label = "px", halign = :right, width = 12, tellwidth = false)

    # Where each side has an answer and the other does not, which is the coverage question the
    # value statistics cannot show.
    axcov = Axis(fig[2, 3]; title = "coverage: 1 both, 2 julia only, 3 ref only, 0 neither",
                 aspect = DataAspect())
    cov = map((j, p) -> j && p ? 1.0 : j ? 2.0 : p ? 3.0 : 0.0, jok, pok)
    hmc = heatmap!(axcov, cov; colormap = cgrad(:tab10, 4; categorical = true),
                   colorrange = (-0.5, 3.5))
    Colorbar(fig[2, 3], hmc; halign = :right, width = 12, tellwidth = false, ticks = 0:3)

    axj = Axis(fig[3, 1]; title = "AutoRIFT.jl chip size", aspect = DataAspect())
    hmj = heatmap!(axj, Float64.(jcs); colormap = :viridis)
    Colorbar(fig[3, 1], hmj; label = "px", halign = :right, width = 12, tellwidth = false)
    axp = Axis(fig[3, 2]; title = "reference chip size", aspect = DataAspect())
    heatmap!(axp, Float64.(pcs); colormap = :viridis,
             colorrange = extrema(Float64.(jcs)))

    axs = Axis(fig[3, 3]; title = "AutoRIFT.jl vs reference", xlabel = "reference px",
               ylabel = "AutoRIFT.jl px")
    if count(both) > 0
        scatter!(axs, pdx[both], jdx[both]; markersize = 3, label = "dx")
        scatter!(axs, pdy[both], jdy[both]; markersize = 3, label = "dy")
        lo, hi = extrema(vcat(pdx[both], pdy[both]))
        lines!(axs, [lo, hi], [lo, hi]; color = :black, linestyle = :dash)
        axislegend(axs; position = :lt)
    end

    axh = Axis(fig[4, 1]; title = "radial difference", xlabel = "px", ylabel = "count")
    axc = Axis(fig[4, 2]; title = "difference vs correlation",
               xlabel = "AutoRIFT.jl correlation", ylabel = "radial difference (px)")
    axcdf = Axis(fig[4, 3]; title = "fraction within tolerance", xlabel = "tolerance (px)",
                 ylabel = "fraction")
    if count(both) > 0
        e = er[both]
        hist!(axh, clamp.(e, 0, 5); bins = 50, color = :steelblue)
        vlines!(axh, [TOL]; color = :red, linestyle = :dash)
        scatter!(axc, jc[both], e; markersize = 3, color = :darkorange)
        hlines!(axc, [TOL]; color = :red, linestyle = :dash)
        ts = range(0, max(1.0, quantile(e, 0.99)); length = 200)
        lines!(axcdf, ts, [mean(e .<= t) for t in ts]; color = :seagreen, linewidth = 2)
        vlines!(axcdf, [TOL]; color = :red, linestyle = :dash)
        ylims!(axcdf, 0, 1.02)
    end

    path = joinpath(PLOTS, TAG * ".png")
    save(path, fig)
    println("\nwrote ", path)
end

main()
