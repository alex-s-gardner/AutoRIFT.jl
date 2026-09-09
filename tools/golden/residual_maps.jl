# Heatmaps of dx, dy, chip size and their differences, for a golden case.
#
#   julia --project=tools/ab -t 8 tools/golden/residual_maps.jl S2B_MSIL1C_20200612
#
# Run this *before* reasoning about a disagreement. A map says at a glance what a median cannot: a
# transposed axis, a whole-pixel roll, a sign flip and a sub-pixel grid offset all produce different
# pictures and similar summary statistics.
#
# It earned that place. A half-pixel grid error in this harness showed a median |ddx| of 1/16 -- one
# quantization step, entirely plausible as tie-breaking -- while the difference map showed a wash
# along every fast-flow margin and blank ice elsewhere. That shape is the signature: a grid offset
# produces no residual under uniform motion and one proportional to the local velocity gradient. Fixing
# it moved exact agreement from 38% to 68%.
#
# The whole-pixel roll scan at the end is the cheap companion test. If some roll beats zero, the two
# grids are offset by an integer; if zero wins, they are aligned and a residual is something else.
#
# `--project=tools/ab` rather than `tools/golden`: CairoMakie lives there, and this is a diagnostic
# rather than part of the comparison.
const R = @__DIR__
include(joinpath(R, "correlator.jl"))
using CairoMakie, Statistics, Printf

name = length(ARGS) >= 1 ? ARGS[1] : "S2B_MSIL1C_20200612"
c = only(cases(name))
# `--run N` because a capture is not always at the default: the radar cases live at 200, and reading
# the wrong run number fails on a missing directory rather than silently mapping another comparison.
i = findfirst(==("--run"), ARGS)
run = i === nothing ? 100 : parse(Int, ARGS[i + 1])
r = compare_correlator(c; n = run)
ny, nx = r.overlap
jdx = r.result.dx[1:ny, 1:nx];  jdy = r.result.dy[1:ny, 1:nx]
rdx = r.capture.arrays["out_Dx"][1:ny, 1:nx]
rdy = .-r.capture.arrays["out_Dy"][1:ny, 1:nx]      # sign measured in compare_correlator
jcs = r.result.chip_size[1:ny, 1:nx]
rcs = r.capture.arrays["out_ChipSizeX"][1:ny, 1:nx]

# Subsample so a 2000² grid renders quickly; nearest, so no interpolation invents structure.
step = max(1, cld(max(ny, nx), 700))
sub(a) = a[1:step:end, 1:step:end]

fig = Figure(size = (1500, 1000))
lim = (-8, 8)
for (col, (lbl, j, rr)) in enumerate((("dx", jdx, rdx), ("dy", jdy, rdy)))
    heatmap(fig[1, col], sub(replace(j, NaN => NaN)); colorrange = lim, colormap = :balance,
            axis = (title = "AutoRIFT.jl $lbl", aspect = DataAspect()))
    heatmap(fig[2, col], sub(replace(rr, NaN => NaN)); colorrange = lim, colormap = :balance,
            axis = (title = "reference $lbl", aspect = DataAspect()))
    d = j .- rr
    heatmap(fig[3, col], sub(d); colorrange = (-1, 1), colormap = :balance,
            axis = (title = "$lbl difference (±1 px)", aspect = DataAspect()))
end
# Chip level, where a disagreement means the two describe different footprints.
heatmap(fig[1, 3], sub(Float32.(jcs)); colormap = :viridis,
        axis = (title = "AutoRIFT.jl chip size", aspect = DataAspect()))
heatmap(fig[2, 3], sub(rcs); colormap = :viridis,
        axis = (title = "reference chip size", aspect = DataAspect()))
heatmap(fig[3, 3], sub(Float32.(Float32.(jcs) .!= rcs)); colormap = :grays,
        axis = (title = "level disagrees (white)", aspect = DataAspect()))
out = joinpath(get(ENV, "AUTORIFT_GOLDEN_PLOTS", mktempdir()), "resid_$(first(name, 24)).png")
save(out, fig)
println("wrote ", out)

# A shifted grid shows up as the residual falling when one field is rolled by a pixel. Cheap to test
# and decisive, so test it rather than eyeballing for it.
ok = .!isnan.(jdx) .& .!isnan.(rdx)
base = median(abs.(Float64.(jdx[ok]) .- Float64.(rdx[ok])))
@printf("\nmedian |ddx| at zero shift: %.5f\n", base)
for (di, dj) in ((0,1),(1,0),(0,-1),(-1,0),(1,1),(-1,-1))
    s = circshift(rdx, (di, dj))
    m = .!isnan.(jdx) .& .!isnan.(s)
    v = median(abs.(Float64.(jdx[m]) .- Float64.(s[m])))
    @printf("  reference rolled by (%+d,%+d): %.5f%s\n", di, dj, v, v < base ? "   <-- BETTER" : "")
end
