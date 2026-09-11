# The correlator endpoint over a window of the grid, rather than the whole scene.
#
#   julia --project=tools/golden -t 8 tools/golden/window_endpoint.jl NISAR_L2_PR_GSLC_003 --run 100
#   julia --project=tools/ab     -t 8 tools/golden/window_endpoint.jl NISAR_L2_PR_GSLC_003 --figure
#
# `correlator.jl` answers every point on the grid, which on the NISAR scenes is 5.2 million points over
# 6 gigapixels and takes about ten hours. That is the right cost for a gate and the wrong cost for a
# diagnostic: the same question over a 401x401 window is answered in about two minutes, which is the
# difference between testing one hypothesis a day and testing several an hour.
#
# **A windowed point is answered identically to how the full pass answers it.** Every per-point field —
# the node, the search radius, the chip bounds, the prior — travels in the `PointSet`, and the chips are
# cut from the same full images. So restricting the point set changes *which* points are compared and
# nothing about what any one of them measures. What a window cannot do is change which pyramid level
# owns a point: that is decided by the reference's own merge over the whole grid, and is read from the
# capture rather than recomputed.
#
# Per-level statistics are the point of this tool. A whole-scene mean mixes populations that behave
# differently — on the NISAR L2 case the base level agrees at 98% while chip 768 carries a 2 px offset —
# and a single number for the scene reports neither.
#
# **A window is a diagnostic, not a measurement of the case.** The coarse levels are sparse and unevenly
# distributed, so which of them a window even contains depends on where it sits: two windows of the same
# NISAR L1 case gave chip-384 means of −0.077 and −0.043, and one contained no chip-768 point at all.
# Quote a window's per-level figure as a property of that window; a case-level number is `correlator.jl`
# over the whole grid.
include(joinpath(@__DIR__, "correlator.jl"))
using Statistics, Printf, Serialization

"""
    window_endpoint(c; n = 100, half = 200, centre = nothing) -> NamedTuple

Correlate a square window of `c`'s grid and compare it against the reference's output there.

`half` is the window's half-width in grid points, so the window is `2half+1` on a side. `centre`
defaults to the centroid of the points the reference measured, which keeps the window on data; pass it
explicitly to look at a particular feature.

Returns the two fields, the reference's chip sizes, and the row and column ranges, so a caller can plot
or re-analyse without correlating again.
"""
function window_endpoint(c::GoldenCase; n::Integer = 100, half::Integer = 200,
                         centre::Union{Nothing,Tuple{Int,Int}} = nothing)
    k = read_capture(c; n)
    full = pointset_from_capture(k)
    rdx_all = k.arrays["out_Dx"]
    gy, gx = size(full.x)

    ci, cj = if centre === nothing
        # The centroid of the measured points, so the window lands on data rather than on the corner of
        # a rotated footprint. Computed from a stride rather than every point: the answer only has to be
        # good enough to place a window, and a 5-million-point centroid is not free.
        idx = [(i, j) for i in 1:8:gy, j in 1:8:gx if !isnan(rdx_all[i, j])]
        isempty(idx) && error("the reference measured no points in $(c.product)")
        (round(Int, mean(first.(idx))), round(Int, mean(last.(idx))))
    else
        centre
    end
    ci = clamp(ci, half + 1, gy - half)
    cj = clamp(cj, half + 1, gx - half)
    rows, cols = (ci - half):(ci + half), (cj - half):(cj + half)
    @info "window" centre=(ci, cj) rows=extrema(rows) cols=extrema(cols) npoints=length(rows) * length(cols)

    w = PointSet(full.x[rows, cols], full.y[rows, cols],
                 full.radius_x[rows, cols], full.radius_y[rows, cols],
                 full.dx_prior[rows, cols], full.dy_prior[rows, cols],
                 full.chip_size_x[rows, cols], full.chip_size_y[rows, cols],
                 full.chip_size_min_x[rows, cols], full.chip_size_max_x[rows, cols])

    a = k.arrays["in_I1"]; b = k.arrays["in_I2"]
    t = @elapsed out = autorift(b, a, w; kwargs_from_capture(k)...)
    @info "correlated window" seconds=round(t; digits = 1)

    return (; jdx = out.dx, jdy = out.dy,
            rdx = rdx_all[rows, cols],
            # `dy` carries the cartesian-to-matrix flip, the same one `compare_correlator` measures.
            rdy = .-k.arrays["out_Dy"][rows, cols],
            rcs = k.arrays["out_ChipSizeX"][rows, cols],
            rows, cols, chip0 = Int(k.scalars["ChipSize0X"]))
end

"""
    report_window(r)

Print the window's agreement overall and per owning pyramid level.

The per-level breakdown is what the whole-scene mean cannot show: only the base level's values are a
measured argmax, so only there does `exact` mean agreement rather than two interpolations landing on the
same number.
"""
function report_window(r)
    both = .!isnan.(r.jdx) .& .!isnan.(r.rdx)
    for (nm, j, ref) in (("dx", r.jdx, r.rdx), ("dy", r.jdy, r.rdy))
        d = j[both] .- ref[both]
        @printf("\n%s: both %d of %d   mean %+.5f  median %+.5f  std %.4f  max|d| %.3f  exact %.2f%%\n",
                nm, count(both), length(both), mean(d), median(d), std(d),
                maximum(abs.(d)), 100count(==(0), d) / length(d))
        @printf("  %-6s %8s %8s %10s %10s %9s %9s\n",
                "chip", "Scale", "n", "exact%", "mean", "median", "std")
        for cs0 in sort(unique(r.rcs[both]))
            m = both .& (r.rcs .== cs0)
            dd = j[m] .- ref[m]
            @printf("  %-6d %8.0f %8d %9.2f%% %+10.4f %+10.4f %9.4f\n",
                    Int(cs0), cs0 / r.chip0, count(m), 100count(==(0), dd) / count(m),
                    mean(dd), median(dd), std(dd))
        end
    end
    return nothing
end

function main(args)
    isempty(args) && error("usage: window_endpoint.jl <product-fragment> [--run N] [--half H] " *
                           "[--centre I,J] [--figure] [--save PATH]")
    c = only(cases(args[1]))
    opt(flag, default) = (i = findfirst(==(flag), args);
                          i === nothing ? default : args[i + 1])
    n = parse(Int, opt("--run", "100"))
    half = parse(Int, opt("--half", "200"))
    cs = opt("--centre", nothing)
    centre = cs === nothing ? nothing :
        (parse(Int, split(cs, ",")[1]), parse(Int, split(cs, ",")[2]))

    r = window_endpoint(c; n, half, centre)
    report_window(r)

    path = opt("--save", nothing)
    path === nothing || (serialize(path, r); println("\nserialized to ", path))
    if "--figure" in args
        # Loaded here rather than at the top: the comparison itself needs no plotting stack, and
        # `tools/golden` does not carry one.
        Base.require(Main, :CairoMakie)
        include(joinpath(@__DIR__, "window_figure.jl"))
        Base.invokelatest(window_figure, r, c.product)
    end
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
