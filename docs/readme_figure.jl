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

# The scene bowed downward, 20 pixels at the centre column and none at the left and right edges. The
# ring ropes are straight in the reference, so the bow is visible in the secondary without the
# recovered panel, and the field varies, so that panel shows structure rather than one flat colour.
# `dy` is the negative of the motion (see Conventions), so a downward bow reports a negative `dy`.
reference, secondary, _, _ =
    warped_pair(photo(), 512, (row, col) -> (0.0, -20.0 * sin(pi * col / 512)))

out = autorift(reference, secondary; grid_spacing = 8)

# Negated to the motion itself, so the panel reads as how far each column moved down.
down = blank(-out.dy, measured(out))

fig = Figure(; size = (960, 330), figure_padding = 8)
vals = vcat(vec(Float64.(reference)), vec(Float64.(secondary)))
gray = (quantile(vals, 0.01), quantile(vals, 0.99))
for (i, (img, title)) in enumerate(((reference, "reference"), (secondary, "secondary")))
    heatmap!(panel(fig, (1, i), title), mapshow(Float64.(img));
             colormap = :grays, colorrange = gray)
end
# Sequential rather than the diverging scale the site's own panels use: this field is one-signed, and
# a symmetric scale would spend half its range on values that never occur.
hm = heatmap!(panel(fig, (1, 3), "recovered downward motion (px)"), mapshow(down);
              colormap = :magma, colorrange = (0, quantile(filter(isfinite, vec(down)), 0.99)))
Colorbar(fig[1, 4], hm)

mkpath(dirname(OUT))
# `px_per_unit = 1`: a photograph compresses poorly, so the default 2× render costs about three times
# the bytes in a file that is committed rather than built.
save(OUT, fig; px_per_unit = 1)
println("wrote $OUT")
