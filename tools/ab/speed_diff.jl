# The difference in velocity magnitude: AutoRIFT.jl speed minus reference speed.
#
# A different quantity from `heatmaps.jl`'s radial panel, and the distinction matters. The radial
# difference is |v_j - v_r|, the length of the vector between the two estimates. This is
# |v_j| - |v_r|, the difference in speed, which is signed and cancels whenever the two vectors have
# the same length in different directions. So the radial difference bounds this one, and a speed
# difference near zero does not by itself mean the two agree -- it means they agree about how fast
# the ice moves, which is what a velocity map is read for.
#
# Reported signed, because the sign is the question a speed comparison exists to answer: a bias
# means one implementation systematically reads faster than the other, and that is what would
# propagate into a mass-flux estimate.
#
#   julia --project=tools/ab tools/ab/speed_diff.jl [tag]

using CairoMakie, Statistics, Printf

const D = joinpath(@__DIR__, "stage2")
const PLOTS = joinpath(@__DIR__, "plots")
const TAG = length(ARGS) >= 1 ? ARGS[1] : "speed_difference"

# This pair, from the granule: 15 m panchromatic pixels, 8.0-day separation.
const PIXEL_SIZE = 15.0
const DATE_DT = 8.000149841122685
const TO_MYR = PIXEL_SIZE * 365.25 / DATE_DT

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
    shapes, _ = manifest()
    jsz, psz = shapes["julia_dx"], python_shape()
    nr, nc = min(jsz[1], psz[1]), min(jsz[2], psz[2])
    crop(A) = A[1:nr, 1:nc]

    jdx = crop(read_bin("julia_dx", Float32, jsz))
    jdy = crop(read_bin("julia_dy", Float32, jsz))
    jc = crop(read_bin("julia_correlation", Float32, jsz))
    pdx = crop(read_bin("python_dx", Float32, psz))
    pdy = crop(read_bin("python_dy", Float32, psz))

    both = .!isnan.(jdx) .& .!isnan.(pdx)

    # Speed is a magnitude, so the sign convention cannot affect it -- a useful property, and a trap:
    # this panel looks the same whether the sign is right or wrong, which is why it is not the panel
    # to check a convention with. `compare2.jl`'s `corr` is.
    jsp = sqrt.(jdx .^ 2 .+ jdy .^ 2) .* TO_MYR
    psp = sqrt.(pdx .^ 2 .+ pdy .^ 2) .* TO_MYR
    dsp = jsp .- psp
    # The vector difference, for contrast: it is what the sub-pixel tolerance is stated against, and
    # it is never smaller than |dsp|.
    radial = sqrt.((jdx .- pdx) .^ 2 .+ (jdy .- pdy) .^ 2) .* TO_MYR

    d = dsp[both]
    r = radial[both]
    s = psp[both]

    @printf("shared points            %d\n", count(both))
    @printf("speed difference (julia - reference), m/yr\n")
    @printf("  median   %+8.2f      mean %+8.2f\n", median(d), mean(d))
    @printf("  p5/p95   %+8.2f / %+8.2f\n", quantile(d, 0.05), quantile(d, 0.95))
    @printf("  |.| median %7.2f    p95 %8.2f    max %9.2f\n",
            median(abs.(d)), quantile(abs.(d), 0.95), maximum(abs.(d)))
    @printf("radial (vector) difference, m/yr: median %7.2f   p95 %8.2f\n",
            median(r), quantile(r, 0.95))
    @printf("reference speed: median %.0f   p95 %.0f   max %.0f m/yr\n",
            median(s), quantile(s, 0.95), maximum(s))
    # Relative error on ice fast enough for it to mean something. On slow ice a ratio divides by a
    # number comparable to the matching noise itself and says nothing.
    for thr in (500.0, 1000.0, 2000.0, 5000.0)
        sel = s .>= thr
        count(sel) < 50 && continue
        @printf("  on ice >= %5.0f m/yr (n=%6d): median speed diff %+7.2f  (%.2f%% of speed)\n",
                thr, count(sel), median(d[sel]),
                100 * median(abs.(d[sel]) ./ s[sel]))
    end

    fig = Figure(; size = (1680, 1080))
    Label(fig[0, 1:3],
          @sprintf("Velocity magnitude difference — AutoRIFT.jl − reference — median %+.1f m/yr on %d points",
                   median(d), count(both));
          fontsize = 20, font = :bold)

    show_(A) = map((v, keep) -> keep ? Float64(v) : NaN, A, both)

    splim = (0.0, quantile(vcat(jsp[both], psp[both]), 0.995))
    ax1 = Axis(fig[1, 1]; title = "AutoRIFT.jl speed", aspect = DataAspect())
    hm1 = heatmap!(ax1, show_(jsp); colormap = :viridis, colorrange = splim)
    Colorbar(fig[1, 1], hm1; label = "m/yr", halign = :right, width = 13, tellwidth = false)
    ax2 = Axis(fig[1, 2]; title = "reference speed", aspect = DataAspect())
    heatmap!(ax2, show_(psp); colormap = :viridis, colorrange = splim)

    # Diverging and symmetric about zero, so sign is legible: red faster in AutoRIFT.jl, blue slower.
    # Clipped at the 99th percentile of |difference| rather than the max, so a handful of
    # weak-correlation outliers do not flatten the scale everything else is read on.
    lim = quantile(abs.(d), 0.99)
    ax3 = Axis(fig[1, 3]; title = "speed difference (red = AutoRIFT.jl faster)",
               aspect = DataAspect())
    hm3 = heatmap!(ax3, show_(dsp); colormap = :balance, colorrange = (-lim, lim))
    Colorbar(fig[1, 3], hm3; label = "m/yr", halign = :right, width = 13, tellwidth = false)

    axh = Axis(fig[2, 1]; title = "speed difference", xlabel = "m/yr", ylabel = "count")
    hist!(axh, clamp.(d, -lim * 3, lim * 3); bins = 80, color = :steelblue)
    vlines!(axh, [0.0]; color = :black)
    vlines!(axh, [median(d)]; color = :red, linestyle = :dash)

    # Against speed, to show the difference does not grow with the signal -- a scale error would
    # appear here as a fan opening to the right.
    axs = Axis(fig[2, 2]; title = "difference vs speed", xlabel = "reference speed (m/yr)",
               ylabel = "speed difference (m/yr)")
    scatter!(axs, s, d; markersize = 2, color = (:darkorange, 0.3))
    hlines!(axs, [0.0]; color = :black)
    ylims!(axs, -lim * 4, lim * 4)

    # Speed against speed on fast ice, where a scale or clipping error would show as a slope
    # departure from 1:1 rather than as scatter.
    axc = Axis(fig[2, 3]; title = "AutoRIFT.jl vs reference speed",
               xlabel = "reference (m/yr)", ylabel = "AutoRIFT.jl (m/yr)")
    scatter!(axc, s, jsp[both]; markersize = 2, color = (:seagreen, 0.3))
    lines!(axc, [0, maximum(s)], [0, maximum(s)]; color = :black, linestyle = :dash)

    path = joinpath(PLOTS, TAG * ".png")
    save(path, fig)
    println("wrote ", path)
    return path
end

main()
