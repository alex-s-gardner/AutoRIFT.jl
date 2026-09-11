# The four-panel difference figure for a windowed endpoint comparison.
#
# Separate from `window_endpoint.jl` so the comparison needs no plotting stack: CairoMakie lives in
# `tools/ab`, and a gate that pulled it in would make every run pay for it.
#
# Read the difference panels against the field panels above them. A residual that follows the signal is
# a scaling or a gradient effect; one that follows a spatial partition is a level or a mask effect; one
# that is uniform is an offset. A median cannot tell those apart and a map can.
using CairoMakie, Statistics, Printf

"""
    window_figure(r, title; path = nothing) -> String

Plot `window_endpoint`'s result: both `dx` fields on a shared colour range, then `ddx` and `ddy`.

Returns the path written. `r` is the named tuple `window_endpoint` returns.
"""
function window_figure(r, title::AbstractString; path = nothing)
    both = .!isnan.(r.jdx) .& .!isnan.(r.rdx)
    ddx = fill(NaN, size(r.jdx)); ddx[both] .= r.jdx[both] .- r.rdx[both]
    ddy = fill(NaN, size(r.jdy)); ddy[both] .= r.jdy[both] .- r.rdy[both]

    # Nearest-neighbour subsample, so the plot invents no structure the data does not have.
    step = max(1, cld(maximum(size(ddx)), 800))
    sub(A) = A[1:step:end, 1:step:end]

    fig = Figure(size = (1800, 1300))
    Label(fig[0, 1:4], "$(first(title, 60)) — AutoRIFT.jl vs autoRIFT, " *
                       "$(size(r.jdx, 1))×$(size(r.jdx, 2)) grid window";
          fontsize = 20, font = :bold)

    function panel(pos, A, ttl; colormap = :balance, colorrange = nothing)
        ax = Axis(fig[pos...]; title = ttl, aspect = DataAspect(), yreversed = true)
        hidedecorations!(ax, label = false)
        v = A[.!isnan.(A)]
        cr = colorrange === nothing ?
            (isempty(v) ? (-1.0, 1.0) :
             (-quantile(abs.(v), 0.98), quantile(abs.(v), 0.98))) : colorrange
        hm = heatmap!(ax, sub(A)'; colormap, colorrange = cr, nan_color = :gray92)
        Colorbar(fig[pos[1], pos[2] + 1], hm)
    end

    # One colour range across both fields: a difference has to be read against the signal that produced
    # it, and two independently-scaled panels hide whether the fields even agree in magnitude.
    vr = (min(quantile(r.jdx[both], 0.02), quantile(r.rdx[both], 0.02)),
          max(quantile(r.jdx[both], 0.98), quantile(r.rdx[both], 0.98)))
    panel((1, 1), r.jdx, "AutoRIFT.jl dx (px)"; colormap = :viridis, colorrange = vr)
    panel((1, 3), r.rdx, "autoRIFT (Python) dx (px)"; colormap = :viridis, colorrange = vr)
    panel((2, 1), ddx, "ddx = julia − python (px)")
    panel((2, 3), ddy, "ddy = julia − python (px)")

    out = path === nothing ?
        joinpath(dirname(@__DIR__), "..", "figs", "window_$(first(title, 40)).png") : path
    out = normpath(out)
    mkpath(dirname(out)); save(out, fig)
    println("wrote ", out)
    return out
end
