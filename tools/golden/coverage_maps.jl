# Where the two implementations disagree about *whether* a point has an answer, as opposed to what
# the answer is.
#
#   julia --project=tools/golden -t 8 tools/golden/coverage_maps.jl S2B_MSIL1C_20200612
#
# Two questions, neither answerable from the coverage counts alone.
#
# **Fill or measurement?** The reference's captured `InterpMask` marks the points it interpolated
# rather than measured, which splits the gap into a hole-filling difference and a rejection
# difference. Disabling the outlier filter cannot make this split, because it removes the holes along
# with the rejections.
#
# **Bias or noise?** A systematic over-rejection puts the points only the reference answers somewhere
# the points only AutoRIFT.jl answers are not. Two sets of marginal decisions straddling one threshold
# in opposite directions put them in the same place. The maps distinguish these immediately where the
# counts do not, which is why this plots rather than only tabulating.
#
# The border fraction is here for a third reason: a reducer or window-margin difference concentrates
# at the grid edge, and a decision difference does not.
include(joinpath(@__DIR__, "correlator.jl"))
using Printf, Statistics, CairoMakie

name = isempty(ARGS) ? "S2B_MSIL1C_20200612" : ARGS[1]
c = only(cases(name))
k = read_capture(c; n = 100)
grid = pointset_from_capture(k); kw = kwargs_from_capture(k)
a = k.arrays["in_I1"]; b = k.arrays["in_I2"]
out = autorift(b, a, grid; kw...)

rdx = k.arrays["out_Dx"]; rim = k.arrays["out_InterpMask"]
ny = min(size(out.dx,1), size(rdx,1)); nx = min(size(out.dx,2), size(rdx,2))
j = out.dx[1:ny,1:nx]; r = rdx[1:ny,1:nx]; im = rim[1:ny,1:nx] .!= 0
mj = .!isnan.(j); mr = .!isnan.(r)

only_r = .!mj .& mr
both   = mj .& mr
@printf("only_ref            %7d   of which reference-interpolated %7d (%.1f%%)\n",
        count(only_r), count(only_r .& im), 100*count(only_r .& im)/max(count(only_r),1))
@printf("both measured       %7d   of which reference-interpolated %7d (%.1f%%)\n",
        count(both), count(both .& im), 100*count(both .& im)/max(count(both),1))
# Baseline: how often is the reference interpolated overall? If only_ref is enriched well above
# this, the gap is filling; if it matches, the gap is rejection.
@printf("reference overall   %7d   interpolated %7d (%.1f%%)\n",
        count(mr), count(mr .& im), 100*count(mr .& im)/max(count(mr),1))

# Border vs interior: a reducer/margin difference concentrates at the edge, a decision
# difference does not.
edge = falses(ny, nx); w = 8
edge[1:w,:] .= true; edge[end-w+1:end,:] .= true; edge[:,1:w] .= true; edge[:,end-w+1:end] .= true
@printf("\nonly_ref on border  %7d (%.1f%% of only_ref); border is %.1f%% of grid\n",
        count(only_r .& edge), 100*count(only_r .& edge)/max(count(only_r),1),
        100*count(edge)/length(edge))

# Look at it. Where the disagreement sits is more informative than how much there is.
step = max(1, cld(max(ny,nx), 700)); sub(x) = x[1:step:end, 1:step:end]
fig = Figure(size = (1400, 500))
heatmap(fig[1,1], sub(Float32.(only_r)); colormap = :grays,
        axis = (title = "only reference answers (white)", aspect = DataAspect()))
heatmap(fig[1,2], sub(Float32.(only_r .& im)); colormap = :grays,
        axis = (title = "...and reference interpolated it", aspect = DataAspect()))
heatmap(fig[1,3], sub(Float32.(mj .& .!mr)); colormap = :grays,
        axis = (title = "only AutoRIFT.jl answers", aspect = DataAspect()))
png = joinpath(get(ENV, "AUTORIFT_GOLDEN_FIGS", tempdir()), "coverage_$(first(name, 24)).png")
save(png, fig)
println("\nwrote ", png)
