# Coverage-edge diagnostic: does a disagreement sit at a data boundary?
#
# The value maps in `residual_maps.jl` answer "is the field offset". This answers a different
# question: are the points only one side measured, or the large residuals, concentrated at the
# edge of valid data or at a block boundary? Nodata handling and halo buffering both produce
# edge-localized artefacts that a whole-scene median cannot see.
#   julia --project=tools/ab -t 6 tools/golden/edge_maps.jl <case> <run>
#
# `residual_maps.jl` answers "is the field offset" — a roll, a transpose, a sign flip. This answers a
# different question: is a disagreement *localized to a data boundary*? Nodata handling and halo
# buffering both produce edge-localized artefacts, and a whole-scene median cannot see them because the
# interior dominates the count.
#
# The ring statistics are the measurement; the maps are for looking. `edge_distance` counts grid cells
# inward from the edge of the reference's own measured region, so "at the boundary" is a number rather
# than an impression.
#
# `--project=tools/ab` because CairoMakie lives there.
const R = @__DIR__
include(joinpath(R, "correlator.jl"))
using CairoMakie, Statistics, Printf
name = ARGS[1]; run = parse(Int, ARGS[2])
c = only(cases(name))
r = compare_correlator(c; n = run)
ny, nx = r.overlap
k = r.capture
jdx = r.result.dx[1:ny, 1:nx]
rdx = k.arrays["out_Dx"][1:ny, 1:nx]
jm = .!isnan.(jdx); rm_ = .!isnan.(rdx)
both = jm .& rm_
onlyj = jm .& .!rm_
onlyr = rm_ .& .!jm
d = fill(NaN32, ny, nx)
@inbounds for i in eachindex(jdx, rdx)
    both[i] && (d[i] = Float32(jdx[i] - rdx[i]))
end
# Distance from the edge of the reference's measured region, in grid cells, so "edge effect" is
# measured rather than eyeballed. One dilation step per ring.
function edge_distance(mask)
    dist = fill(typemax(Int32), size(mask))
    front = [i for i in CartesianIndices(mask) if mask[i] &&
             any(!checkbounds(Bool, mask, i + o) || !mask[i + o]
                 for o in (CartesianIndex(1,0), CartesianIndex(-1,0),
                           CartesianIndex(0,1), CartesianIndex(0,-1)))]
    for i in front; dist[i] = 0; end
    cur = front; step = Int32(0)
    while !isempty(cur) && step < 40
        step += Int32(1); nxt = CartesianIndex{2}[]
        for i in cur, o in (CartesianIndex(1,0), CartesianIndex(-1,0), CartesianIndex(0,1), CartesianIndex(0,-1))
            j = i + o
            checkbounds(Bool, mask, j) || continue
            mask[j] && dist[j] > step && (dist[j] = step; push!(nxt, j))
        end
        cur = nxt
    end
    dist
end
ed = edge_distance(rm_)
println("=== ", name, "  grid ", (ny, nx))
println("both=", count(both), "  only_jl=", count(onlyj), "  only_ref=", count(onlyr))
# Is |ddx| elevated near the boundary?
println("\n|ddx| by distance from the reference's data edge:")
for (lo, hi) in ((0,0), (1,2), (3,5), (6,10), (11,20), (21,40), (41,typemax(Int32)))
    sel = [d[i] for i in eachindex(d) if both[i] && !isnan(d[i]) && lo <= ed[i] <= hi]
    isempty(sel) && continue
    a = abs.(sel)
    @printf("  ring %-8s n=%-9d median=%.5f  mean=%+.5f  p99=%.4f  frac>1px=%.4f\n",
            hi == typemax(Int32) ? ">$lo" : "$lo-$hi", length(sel), median(a), mean(sel),
            quantile(a, 0.99), count(>(1.0), a)/length(a))
end
# And where do the exclusive sets sit?
println("\nonly-julia and only-reference points, by distance from that same edge:")
for (nm, m) in (("only_jl", onlyj), ("only_ref", onlyr))
    ds = [ed[i] for i in eachindex(m) if m[i] && ed[i] != typemax(Int32)]
    isempty(ds) && (println("  ", nm, ": none inside the reference's region"); continue)
    @printf("  %-8s n=%-8d median dist=%d  frac at edge(0)=%.3f  frac<=2=%.3f\n",
            nm, length(ds), Int(median(ds)), count(==(0), ds)/length(ds), count(<=(2), ds)/length(ds))
end
out = joinpath(get(ENV, "GOLDEN_FIGDIR", mktempdir()), "edge_$(first(name,24)).png")
fig = Figure(size = (1500, 500))
ax1 = Axis(fig[1,1], title = "coverage: both=grey only_jl=blue only_ref=red", yreversed = true)
cov = fill(0.0f0, ny, nx)
cov[both] .= 1; cov[onlyj] .= 2; cov[onlyr] .= 3
heatmap!(ax1, cov'; colormap = [:white, :grey80, :dodgerblue, :crimson], colorrange = (0, 3))
ax2 = Axis(fig[1,2], title = "dx difference (±1 px)", yreversed = true)
heatmap!(ax2, d'; colormap = :RdBu, colorrange = (-1, 1))
ax3 = Axis(fig[1,3], title = "|ddx| > 1 px (black)", yreversed = true)
big = fill(0.0f0, ny, nx); for i in eachindex(d); (!isnan(d[i]) && abs(d[i]) > 1) && (big[i] = 1); end
heatmap!(ax3, big'; colormap = [:white, :black], colorrange = (0, 1))
for a in (ax1, ax2, ax3); hidedecorations!(a); end
save(out, fig)
println("\nwrote ", out)
