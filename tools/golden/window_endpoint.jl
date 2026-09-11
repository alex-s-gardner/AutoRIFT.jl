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
# **A windowed pass does not answer a point the way the full pass answers it, and the difference is not
# small.** Every per-point field travels in the `PointSet` and the chips are cut from the same full
# images, so a *single* correlation at a *given* chip size is unaffected. What the window changes is
# **which pyramid level claims the point**. The coarse lattice is laid out from the grid's extent
# (`stages.jl` records the same hazard for the coarse-sampling rung: "not crop-safe … a sub-window
# samples a different set"), so a shifted or cropped extent decimates to different coarse nodes, the
# level-selection mask differs, and a point can be answered by a different level entirely.
#
# Measured on the NISAR L1 case at grid node (246,1722), against the reference's −3.250000:
#
# | window | our `dx` | level |
# |---|---:|---:|
# | half 8, 32, 100, 200 centred on the node | **−3.250000** — exact | 96 |
# | the 401×401 window at rows 124–524 | −4.561503 | **768** |
#
# The point is bit-exact at four extents and off by 1.31 px at a fifth, because the fifth assigned it to
# chip 768 rather than chip 96. Nothing about the correlator differs; the level does.
#
# **So per-level statistics from a window are not comparable to the reference's.** Reading a window's
# residual against the reference's `out_ChipSizeX` compares two levels' answers and reports the
# difference as a per-level disagreement. That artifact produced an apparent 130-point base-level
# disagreement, and coarse-level means that swung between −0.077 and +2.15 across windows of one case.
#
# What a window *is* good for: reproducing a named point at a fixed extent, and bisecting an extent
# dependence by holding the node and varying the window — which is what identified the artifact above.
# **A case-level or per-level number comes from `correlator.jl` over the whole grid, and nowhere else.**
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

    # `jcs` beside `rcs`, because which level answered a point is a result and not a parameter: the two
    # implementations choose independently, and a point they assign to different levels carries the whole
    # difference between two levels' answers rather than a numerical disagreement.
    # `correlation` and `peak_ratio` travel too: the reference exports neither (`minMaxLoc`'s value is
    # discarded at all four call sites), so they are the only way to ask whether a disagreement was
    # predictable from the surface it came from. Note what each describes — `correlation` is `NaN` at an
    # interpolated point, and `peak_ratio` there is the neighbourhood median rather than a measurement of
    # that point — so a level's coverage of them has to be reported beside any figure derived from them.
    return (; jdx = out.dx, jdy = out.dy, jcs = out.chip_size,
            correlation = out.correlation, peak_ratio = out.peak_ratio,
            interpolated = out.interpolated,
            rdx = rdx_all[rows, cols],
            # `dy` carries the cartesian-to-matrix flip, the same one `compare_correlator` measures.
            rdy = .-k.arrays["out_Dy"][rows, cols],
            rcs = k.arrays["out_ChipSizeX"][rows, cols],
            rows, cols, chip0 = Int(k.scalars["ChipSize0X"]))
end

"""
    report_window(r)

Print the window's agreement, and how much of it the extent invalidates.

**The per-level breakdown is deliberately withheld.** Level assignment is extent-dependent, so grouping a
window's residual by the reference's `out_ChipSizeX` compares points the two implementations answered at
*different* levels and reports that as a per-level disagreement. What is printed instead is the level
*disagreement rate*, which is the measure of how far this window's extent has moved the answer away from
the full pass. A window whose levels agree everywhere is one whose residual can be read; one whose levels
disagree is measuring its own extent.
"""
function report_window(r)
    both = .!isnan.(r.jdx) .& .!isnan.(r.rdx)
    # Level agreement first, because it decides whether anything below it means what it appears to.
    samelvl = both .& (r.jcs .== r.rcs)
    difflvl = both .& (r.jcs .!= r.rcs)
    @printf("\nlevel assignment: agree on %d of %d (%.2f%%), differ on %d (%.2f%%)\n",
            count(samelvl), count(both), 100count(samelvl) / count(both),
            count(difflvl), 100count(difflvl) / count(both))
    count(difflvl) > 0 && @printf("  %s %d points are answered at different pyramid levels by the two\n" *
                                  "  implementations, so their residual is a level difference rather than a\n" *
                                  "  disagreement. Per-level figures from this window are not comparable to\n" *
                                  "  the reference's; use `correlator.jl` over the whole grid for those.\n",
                                  "WARNING:", count(difflvl))

    for (nm, j, ref) in (("dx", r.jdx, r.rdx), ("dy", r.jdy, r.rdy))
        d = j[both] .- ref[both]
        @printf("\n%s  all points:        n=%7d  mean %+.5f  median %+.5f  max|d| %.3f  exact %.2f%%\n",
                nm, count(both), mean(d), median(d), maximum(abs.(d)),
                100count(==(0), d) / length(d))
        # The same figures over the points whose level both sides agree on. This is the only subset of a
        # window whose residual is attributable to the computation rather than to the extent.
        if count(samelvl) > 0
            ds = j[samelvl] .- ref[samelvl]
            @printf("%s  same level only:   n=%7d  mean %+.5f  median %+.5f  max|d| %.3f  exact %.2f%%\n",
                    nm, count(samelvl), mean(ds), median(ds), maximum(abs.(ds)),
                    100count(==(0), ds) / length(ds))
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
