# A dx/dy difference map over one sub-region of the stage-2 comparison, at a scale where a one-step
# disagreement is visible.
#
# The report's own figures cover the whole window, which is the right scale for "do the two agree" and
# the wrong one for "why do they disagree *here*": a feature a dozen grid points across is a few pixels
# on that page. This draws a named region large enough to read cell by cell, beside the chip size that
# owns each point — a level boundary and a feature boundary look alike at low magnification, and telling
# them apart is usually the whole question.
#
#   julia --project=tools/ab tools/ab/zoom.jl <tag> <row0> <row1> <col0> <col1>
#   julia --project=tools/ab tools/ab/zoom.jl terminus 195 215 240 260
#
# Writes `plots/<tag>.png` and prints the region's statistics. Reads the bundle `stage2_julia.jl` and
# `stage2_python.py` wrote, so run those first.

using CairoMakie, Printf, Statistics

const ZD = joinpath(@__DIR__, "stage2")
const ZP = joinpath(@__DIR__, "plots")
# One step of the sub-pixel search: the finest difference either implementation can express, and so the
# unit the difference panels are scaled in. Read from the bundle rather than assumed.
zstep(scalars) = 1 / get(scalars, "upsampling", 16)

function zoom_load()
    shapes = Dict{String,Tuple{Int,Int}}()
    scalars = Dict{String,Int}()
    for line in eachline(joinpath(ZD, "manifest.txt"))
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        parts = split(s)
        length(parts) == 2 ? (scalars[parts[1]] = parse(Int, parts[2])) :
                             (shapes[parts[1]] = Tuple(parse.(Int, split(parts[3], "x"))))
    end
    rd(name, T, dims) = reshape(collect(reinterpret(T, read(joinpath(ZD, "$name.bin")))), dims)
    js = shapes["julia_dx"]
    ps = Tuple(parse.(Int, split(strip(read(joinpath(ZD, "python_shape.txt"), String)))))
    nr, nc = min(js[1], ps[1]), min(js[2], ps[2])
    crop(A) = A[1:nr, 1:nc]
    return (; jdx = crop(rd("julia_dx", Float32, js)), pdx = crop(rd("python_dx", Float32, ps)),
            jdy = crop(rd("julia_dy", Float32, js)), pdy = crop(rd("python_dy", Float32, ps)),
            chip = crop(rd("julia_chip_size", Int32, js)), scalars, size = (nr, nc))
end

# Image orientation, as `bench_figures.jl` uses: `heatmap!` treats the first index as x, so a `[row,
# col]` array drawn directly comes out transposed.
zshow(A) = reverse(permutedims(A); dims = 2)
zblank(A, keep) = map((v, k) -> k ? Float64(v) : NaN, A, keep)

function zoom_region(tag, R, C)
    d = zoom_load()
    step = zstep(d.scalars)
    both = .!isnan.(d.jdx) .& .!isnan.(d.pdx)
    rad = sqrt.((d.jdx .- d.pdx) .^ 2 .+ (d.jdy .- d.pdy) .^ 2)
    sub(A) = A[R, C]

    fig = Figure(; size = (1500, 980), figure_padding = 14)
    ax(pos, t) = Axis(fig[pos...]; title = t, aspect = DataAspect(), titlesize = 13,
                      xticksvisible = false, yticksvisible = false,
                      xticklabelsvisible = false, yticklabelsvisible = false)

    # One scale across both sides so a magnitude difference is visible rather than absorbed by
    # per-panel scaling, taken from the region itself so a slow region is not drawn flat.
    vals = filter(isfinite, vec(sub(zblank(d.jdx, both))))
    lim = isempty(vals) ? 1.0 : max(quantile(abs.(vals), 0.99), 1.0)
    for (row, (nm, J, P)) in enumerate((("dx", d.jdx, d.pdx), ("dy", d.jdy, d.pdy)))
        hm = heatmap!(ax((row, 1), "AutoRIFT.jl $nm"), zshow(sub(zblank(J, both)));
                      colormap = :balance, colorrange = (-lim, lim))
        heatmap!(ax((row, 2), "autoRIFT.py $nm"), zshow(sub(zblank(P, both)));
                 colormap = :balance, colorrange = (-lim, lim))
        Colorbar(fig[row, 3], hm; label = "$nm (px)", width = 11)
        # Scaled to a few sub-pixel steps, so anything visible is at least one step and a point where
        # the two agree exactly is the background colour.
        hd = heatmap!(ax((row, 4), "difference in $nm"), zshow(sub(zblank(J .- P, both)));
                      colormap = :balance, colorrange = (-8 * step, 8 * step))
        Colorbar(fig[row, 5], hd; label = "jl − py (px)", width = 11)
    end

    # Chip size, ranked rather than valued: the levels are powers of two, so a linear colour range puts
    # 16 and 32 in one band and hides the level that answers the margins.
    chips = sort(filter(>(0), unique(vec(d.chip))))
    rank = Dict(c => i for (i, c) in enumerate(chips))
    heatmap!(ax((1, 6), "chip size (px)"),
             zshow(sub(zblank(map(v -> get(rank, v, 0), d.chip), both .& (d.chip .> 0))));
             colormap = cgrad(:viridis, max(length(chips), 2); categorical = true),
             colorrange = (0.5, length(chips) + 0.5))
    Legend(fig[1, 7],
           [PolyElement(; color = cgrad(:viridis, max(length(chips), 2);
                                        categorical = true)[i]) for i in eachindex(chips)],
           string.(chips); framevisible = false, patchsize = (14, 14), labelsize = 11,
           halign = :left, valign = :center, tellheight = false)

    # The disagreement itself, in steps, on a sequential scale — magnitude is the question here and the
    # sign is already in the two panels above.
    hb = heatmap!(ax((2, 6), "radial difference (steps of 1/$(d.scalars["upsampling"]) px)"),
                  zshow(sub(zblank(rad ./ step, both)));
                  colormap = :magma, colorrange = (0, 8))
    Colorbar(fig[2, 7], hb; label = "steps", width = 11)

    s = sub(rad); sb = sub(both)
    @printf("%s: rows %d..%d cols %d..%d | shared=%d  exact=%.1f%%  median=%.2f steps  p95=%.2f  max=%.1f\n",
            tag, first(R), last(R), first(C), last(C), count(sb),
            100 * mean(s[sb] .== 0), median(s[sb]) / step,
            quantile(s[sb], 0.95) / step, maximum(s[sb]) / step)
    mkpath(ZP)
    out = joinpath(ZP, "$tag.png")
    save(out, fig; px_per_unit = 2)
    println("wrote ", out)
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 5 ||
        error("usage: zoom.jl <tag> <row0> <row1> <col0> <col1>")
    zoom_region(ARGS[1], parse(Int, ARGS[2]):parse(Int, ARGS[3]),
                parse(Int, ARGS[4]):parse(Int, ARGS[5]))
end
