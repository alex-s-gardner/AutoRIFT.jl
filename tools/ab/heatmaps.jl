# Output heatmaps: AutoRIFT.jl beside the reference, and their difference.
#
# `compare2.jl` writes a dense diagnostic sheet. This writes the three panels that answer the
# question directly -- the two fields and the difference between them -- large enough to read, in
# velocity units as well as pixels.
#
#   julia --project=tools/ab tools/ab/heatmaps.jl [tag]

using CairoMakie, Statistics, Printf

const D = joinpath(@__DIR__, "stage2")
const PLOTS = joinpath(@__DIR__, "plots")
const TAG = length(ARGS) >= 1 ? ARGS[1] : "heatmaps"
const TOL = 0.2

# This pair, from the granule: 15 m panchromatic pixels and an 8.0-day separation. One pixel of
# displacement is therefore ~685 m/yr, which is what makes a 0.2 px tolerance a ~137 m/yr claim.
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
    shapes, scalars = manifest()
    jsz, psz = shapes["julia_dx"], python_shape()
    nr, nc = min(jsz[1], psz[1]), min(jsz[2], psz[2])

    crop(A) = A[1:nr, 1:nc]
    jdx = crop(read_bin("julia_dx", Float32, jsz))
    jdy = crop(read_bin("julia_dy", Float32, jsz))
    jcs = crop(read_bin("julia_chip_size", Int32, jsz))
    # Negated into AutoRIFT.jl's convention: the reference reports feature motion, this package
    # reports secondary-to-reference. Both axes together -- see tools/ab/README.md.
    pdx = crop(.-read_bin("python_dx", Float32, psz))
    pdy = crop(.-read_bin("python_dy", Float32, psz))
    pcs = crop(read_bin("python_chip_size", Int32, psz))

    jok, pok = .!isnan.(jdx), .!isnan.(pdx)
    both = jok .& pok
    er = sqrt.((jdx .- pdx) .^ 2 .+ (jdy .- pdy) .^ 2)

    # Speed rather than the components, for the overview row: it is the quantity a reader of an
    # ice-velocity map is actually looking at, and it needs no sign convention to interpret.
    jsp = sqrt.(jdx .^ 2 .+ jdy .^ 2) .* TO_MYR
    psp = sqrt.(pdx .^ 2 .+ pdy .^ 2) .* TO_MYR

    show_(A, m) = map((v, keep) -> keep ? Float64(v) : NaN, A, m)

    fig = Figure(; size = (1680, 1180))
    med = count(both) > 0 ? median(er[both]) : NaN
    Label(fig[0, 1:3],
          @sprintf("AutoRIFT.jl vs Python autoRIFT v2.1.2 — %d shared grid points — median difference %.4f px (%.0f m/yr)",
                   count(both), med, med * TO_MYR);
          fontsize = 20, font = :bold)

    splim = (0.0, quantile(filter(!isnan, vcat(vec(jsp), vec(psp))), 0.995))

    ax1 = Axis(fig[1, 1]; title = "AutoRIFT.jl speed", aspect = DataAspect())
    hm1 = heatmap!(ax1, show_(jsp, jok); colormap = :viridis, colorrange = splim)
    Colorbar(fig[1, 1], hm1; label = "m/yr", halign = :right, width = 13, tellwidth = false)

    ax2 = Axis(fig[1, 2]; title = "reference speed", aspect = DataAspect())
    heatmap!(ax2, show_(psp, pok); colormap = :viridis, colorrange = splim)

    # Centred at the tolerance, so "inside 0.2 px" reads as dark at a glance.
    ax3 = Axis(fig[1, 3]; title = "|difference| (0.2 px = tolerance)", aspect = DataAspect())
    hm3 = heatmap!(ax3, show_(er, both); colormap = :inferno, colorrange = (0, TOL * 2))
    Colorbar(fig[1, 3], hm3; label = "px", halign = :right, width = 13, tellwidth = false)

    dlim = let v = filter(isfinite, vcat(vec(show_(jdx, jok)), vec(show_(pdx, pok)),
                                        vec(show_(jdy, jok)), vec(show_(pdy, pok))))
        isempty(v) ? (-1.0, 1.0) : (-quantile(abs.(v), 0.995), quantile(abs.(v), 0.995))
    end

    for (row, (nm, J, P)) in enumerate((("dx", jdx, pdx), ("dy", jdy, pdy)))
        a = Axis(fig[row + 1, 1]; title = "AutoRIFT.jl $nm", aspect = DataAspect())
        h = heatmap!(a, show_(J, jok); colormap = :balance, colorrange = dlim)
        Colorbar(fig[row + 1, 1], h; label = "px", halign = :right, width = 13, tellwidth = false)
        b = Axis(fig[row + 1, 2]; title = "reference $nm", aspect = DataAspect())
        heatmap!(b, show_(P, pok); colormap = :balance, colorrange = dlim)
    end

    # Which chip-size level answered each point. Included because it is the only place the two
    # genuinely differ in structure, and a difference here changes how much a value is smoothed
    # rather than whether it is right.
    csvals = sort(unique(vcat(unique(jcs), unique(pcs))))
    cslim = (Float64(minimum(csvals)), Float64(maximum(csvals)))
    axj = Axis(fig[2, 3]; title = "AutoRIFT.jl chip size", aspect = DataAspect())
    hmj = heatmap!(axj, Float64.(jcs); colormap = :viridis, colorrange = cslim)
    Colorbar(fig[2, 3], hmj; label = "px", halign = :right, width = 13, tellwidth = false)
    axp = Axis(fig[3, 3]; title = "reference chip size", aspect = DataAspect())
    heatmap!(axp, Float64.(pcs); colormap = :viridis, colorrange = cslim)

    path = joinpath(PLOTS, TAG * ".png")
    save(path, fig)

    @printf("shared %dx%d   coverage julia %.1f%%  reference %.1f%%\n",
            nr, nc, 100 * mean(jok), 100 * mean(pok))
    if count(both) > 0
        e = er[both]
        @printf("difference median %.4f px (%.0f m/yr)   p95 %.4f px   within %.1f px %.1f%%\n",
                median(e), median(e) * TO_MYR, quantile(e, 0.95), TOL, 100 * mean(e .<= TOL))
        @printf("bias dx %+.4f px   dy %+.4f px\n",
                median(jdx[both] .- pdx[both]), median(jdy[both] .- pdy[both]))
        @printf("speed: julia median %.0f m/yr   reference median %.0f m/yr\n",
                median(jsp[both]), median(psp[both]))
    end
    println("wrote ", path)
    return path
end

main()
