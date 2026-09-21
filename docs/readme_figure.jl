# Generates the one figure committed to the repository, `docs/src/assets/readme_example.png`.
#
#     julia --project=docs docs/readme_figure.jl
#
# GitHub cannot run an `@example` block, so the README's figure has to exist as a file where every
# figure on the documentation site is built from a live `autorift` call. That makes this the one image
# that can go stale: **re-run this script whenever the README's quickstart changes**, and keep the two
# in step — the script correlates the same pair the README's code block does.

using AutoRIFT
using CairoMakie

include(joinpath(@__DIR__, "figures.jl"))

const OUT = joinpath(@__DIR__, "src", "assets", "readme_example.png")

# A band across the middle moved 22 pixels, the rest of the surface 2 — a spatially varying field, so
# the recovered panel shows structure. A uniform shift would draw as one flat colour and say nothing.
band(row) = clamp((108 - abs(row - 256)) / 28, 0, 1)

reference, clean, _, _ = warped_pair(512, (row, col) -> (2 + 20 * band(row), 0.0); seed = 11)
secondary = decorrelate(clean, (row, col) -> band(row); amplitude = 0.12, seed = 3)

out = autorift(reference, secondary; grid_spacing = 8)

keep = measured(out)
dx = blank(out.dx, keep)

fig = Figure(; size = (960, 330), figure_padding = 8)
vals = vcat(vec(Float64.(reference)), vec(Float64.(secondary)))
gray = (quantile(vals, 0.01), quantile(vals, 0.99))
for (i, (img, title)) in enumerate(((reference, "reference"), (secondary, "secondary")))
    heatmap!(panel(fig, (1, i), title), mapshow(Float64.(img));
             colormap = :grays, colorrange = gray)
end
# Sequential rather than the diverging scale the site's own panels use: this field is one-signed, and
# a symmetric scale would spend half its range on values that never occur.
hm = heatmap!(panel(fig, (1, 3), "recovered dx (px)"), mapshow(dx); colormap = :magma,
              colorrange = (0, quantile(filter(isfinite, vec(dx)), 0.99)))
Colorbar(fig[1, 4], hm)

mkpath(dirname(OUT))
# `px_per_unit = 1`: a random texture is incompressible, so the default 2× render costs about four
# times the bytes in a file that is committed rather than built.
save(OUT, fig; px_per_unit = 1)
println("wrote $OUT")
