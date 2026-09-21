# Plotting helpers shared by the documentation pages, and the synthetic scenes the examples
# correlate. Included by each page's first `@example` block.
#
# Nothing here is part of AutoRIFT's API. The scenes are built in code rather than read from files so
# that every figure on the site is the output of a live `autorift` call on data the page itself
# produced — a wrong figure then requires a wrong package, which a stored array would not.

using CairoMakie
using Random: MersenneTwister, randn
using Statistics: quantile

# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------

# A grid array as `heatmap!` needs it to appear in image orientation: x horizontal, y vertical,
# increasing downward.
#
# `heatmap!(A)` treats A's *first* index as the x axis, so passing a `[row, col]` array directly draws
# the image transposed — rows run horizontally. Transposing fixes the axes, and reversing the
# resulting columns puts row 1 at the top, which is where the first row of an image belongs. Without
# this, a feature elongated along y appears elongated along x, which inverts the reading of any
# anisotropy in the field.
mapshow(A) = reverse(permutedims(A); dims = 2)

# `NaN` where a point has no answer, so an unmeasured point reads as blank rather than as zero — a
# zero displacement is a real measurement and must not look like a missing one.
blank(A, keep) = map((v, k) -> k ? Float64(v) : NaN, A, keep)

measured(out) = .!isnan.(out.dx)

# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

const PANEL = (; xticksvisible = false, yticksvisible = false,
               xticklabelsvisible = false, yticklabelsvisible = false)

panel(fig, pos, title) = Axis(fig[pos...]; title, aspect = DataAspect(), PANEL...)

# One symmetric scale across every field drawn together, so a difference in sign or magnitude is
# visible rather than absorbed by per-panel scaling. The 99th percentile rather than the extreme,
# which a single outlier would otherwise set.
function symlimits(fields...)
    vals = filter(isfinite, mapreduce(vec, vcat, fields))
    lim = isempty(vals) ? 1.0 : quantile(abs.(vals), 0.99)
    lim = lim > 0 ? lim : 1.0
    return (-lim, lim)
end

"""
    image_panels(images...; titles) -> Figure

A row of input images on a shared grayscale.
"""
function image_panels(images...; titles = ("reference", "secondary"), size = (900, 420))
    fig = Figure(; size, figure_padding = 10)
    # A percentile range rather than the extremes: added noise puts a few pixels far outside the
    # surface's own range, and scaling to those flattens the features the figure exists to show.
    vals = mapreduce(vec, vcat, images)
    lim = (quantile(vals, 0.01), quantile(vals, 0.99))
    local hm
    for (i, (img, title)) in enumerate(zip(images, titles))
        hm = heatmap!(panel(fig, (1, i), title), mapshow(Float64.(img));
                      colormap = :grays, colorrange = lim)
    end
    Colorbar(fig[1, length(images) + 1], hm)
    return fig
end

"""
    displacement_panels(out; titles) -> Figure

`dx` and `dy` on one symmetric diverging scale, blanked where nothing was measured.
"""
function displacement_panels(out; size = (900, 420))
    keep = measured(out)
    dx, dy = blank(out.dx, keep), blank(out.dy, keep)
    fig = Figure(; size, figure_padding = 10)
    lim = symlimits(dx, dy)
    local hm
    for (i, (field, title)) in enumerate(((dx, "dx (px)"), (dy, "dy (px)")))
        hm = heatmap!(panel(fig, (1, i), title), mapshow(field);
                      colormap = :balance, colorrange = lim)
    end
    Colorbar(fig[1, 3], hm)
    return fig
end

"""
    quality_panels(out) -> Figure

`correlation` on a fixed `(0, 1)` scale, and the chip size that answered at each point.
"""
function quality_panels(out; size = (900, 420))
    keep = measured(out)
    fig = Figure(; size, figure_padding = 10)
    # `(0, 1)` is fixed rather than fitted: correlation is already a normalized quantity, and a
    # per-panel scale would make a weak field look as strong as a good one.
    hmc = heatmap!(panel(fig, (1, 1), "correlation"), mapshow(blank(out.correlation, keep));
                   colormap = :magma, colorrange = (0, 1))
    Colorbar(fig[1, 2], hmc)
    # `chip_size == 0` marks a point no level answered, which `blank` turns into a gap rather than a
    # chip size of zero.
    chip = blank(out.chip_size, out.chip_size .> 0)
    hms = heatmap!(panel(fig, (1, 3), "chip size (px)"), mapshow(chip); colormap = :viridis)
    Colorbar(fig[1, 4], hms)
    return fig
end

"""
    input_field(A, label) -> Figure

A single per-point argument — a chip size, search radius or prior — drawn as a heatmap, so passing an
array is something a reader can see rather than a claim.
"""
function input_field(A, label; colormap = :viridis, size = (480, 420))
    fig = Figure(; size, figure_padding = 10)
    hm = heatmap!(panel(fig, (1, 1), label), mapshow(Float64.(A)); colormap)
    Colorbar(fig[1, 2], hm)
    return fig
end

"""
    surface_panels(chip, window, surface) -> Figure

The three objects one correlation involves: the chip, the window it is searched in, and the resulting
correlation surface with its peak marked.

The chip and window share a grayscale so their relative size and shared texture are both visible; the
surface gets its own scale, since a correlation value means something absolute.
"""
function surface_panels(chip, window, surface; size = (1200, 400))
    fig = Figure(; size, figure_padding = 10)
    vals = vcat(vec(Float64.(chip)), vec(Float64.(window)))
    lim = (quantile(vals, 0.01), quantile(vals, 0.99))
    for (i, (img, title)) in enumerate(((chip, "chip $(join(Base.size(chip), "×")) (reference)"),
                                        (window,
                                         "search window $(join(Base.size(window), "×")) (secondary)")))
        heatmap!(panel(fig, (1, i), title), mapshow(Float64.(img));
                 colormap = :grays, colorrange = lim)
    end
    ax = panel(fig, (1, 3), "correlation surface $(join(Base.size(surface), "×"))")
    hm = heatmap!(ax, mapshow(Float64.(surface)); colormap = :magma)
    # The peak in the same coordinates the panel is drawn in: `mapshow` transposes and flips, so the
    # marker has to follow rather than be placed at the raw index.
    pi_, pj = Tuple(argmax(surface))
    scatter!(ax, [Float64(pj)], [Float64(Base.size(surface, 1) - pi_ + 1)];
             marker = :cross, markersize = 18, color = :cyan, strokewidth = 0)
    Colorbar(fig[1, 4], hm)
    return fig
end

"""
    field_panels(fields...; titles) -> Figure

A row of displacement-like fields on one shared symmetric scale, for comparing the same quantity
across settings.
"""
function field_panels(fields...; titles, size = (1100, 400), colormap = :balance)
    fig = Figure(; size, figure_padding = 10)
    lim = symlimits(fields...)
    local hm
    for (i, (field, title)) in enumerate(zip(fields, titles))
        hm = heatmap!(panel(fig, (1, i), title), mapshow(Float64.(field)); colormap,
                      colorrange = lim)
    end
    Colorbar(fig[1, length(fields) + 1], hm)
    return fig
end

# ---------------------------------------------------------------------------
# Synthetic scenes
# ---------------------------------------------------------------------------

# Bilinear sample of `img` at `(row, col)`, zero outside. Bilinear and not nearest-neighbour so a
# fractional displacement is actually present in the data rather than rounded away.
function _sample(img, y, x)
    nr, nc = size(img)
    i0, j0 = floor(Int, y), floor(Int, x)
    fy, fx = y - i0, x - j0
    v = 0.0
    for (di, wy) in ((0, 1 - fy), (1, fy)), (dj, wx) in ((0, 1 - fx), (1, fx))
        i, j = i0 + di, j0 + dj
        if 1 <= i <= nr && 1 <= j <= nc
            v += wy * wx * img[i, j]
        end
    end
    return v
end

# One octave: random values on a grid of `cell`-pixel spacing, interpolated up to `sz`.
function _octave(sz, cell; seed)
    nr, nc = cld(sz[1], cell) + 2, cld(sz[2], cell) + 2
    coarse = randn(MersenneTwister(seed), Float64, nr, nc)
    out = Matrix{Float64}(undef, sz)
    for j in 1:sz[2], i in 1:sz[1]
        out[i, j] = _sample(coarse, 1 + (i - 1) / cell, 1 + (j - 1) / cell)
    end
    return out
end

"""
    texture(sz; seed, cells, weights) -> Matrix{Float64}

A random texture with features at several scales, scaled to `[0, 1]`.

Each entry of `cells` is a feature size in pixels, weighted by the matching entry of `weights`.
Several scales rather than one: white noise correlates only at exactly zero lag, so it says nothing
about the shape of the correlation surface, while a single smooth scale has no fine detail for a small
chip to lock onto. A real surface has both.
"""
function texture(sz::Tuple{Int,Int}; seed::Integer = 0, cells = (24, 8, 3),
                 weights = (1.0, 0.55, 0.3))
    out = zeros(Float64, sz)
    for (k, (cell, w)) in enumerate(zip(cells, weights))
        a = _octave(sz, cell; seed = seed + 100k)
        s = maximum(abs, a)
        out .+= (w / (s > 0 ? s : 1)) .* a
    end
    out .-= minimum(out)
    m = maximum(out)
    m > 0 && (out ./= m)
    return out
end

texture(n::Integer; kw...) = texture((Int(n), Int(n)); kw...)

"""
    warped_pair(sz, offset; seed, pad) -> (reference, secondary, truth_dx, truth_dy)

A texture and a warped copy of it, together with the displacement `autorift` should report, sampled
on the same grid as the images.

`offset(row, col) -> (dx, dy)` is in pixels and in `autorift`'s own convention, so `truth_dx` and
`truth_dy` compare directly against a result's `dx` and `dy`. That convention is the offset from
secondary back to reference, which is the negative of how the surface moved: `offset = (6, 0)` puts
a feature six columns to the *left* in the secondary image. See [Conventions](@ref).

The texture is generated oversized and cropped, so warping it never pulls in out-of-image data — an
artificial edge inside a search window would make the correlation there a measurement of the crop
rather than of the motion.
"""
function warped_pair(sz::Tuple{Int,Int}, offset; seed::Integer = 0, pad::Integer = 48)
    big = texture((sz[1] + 2pad, sz[2] + 2pad); seed)
    rows, cols = (pad + 1):(pad + sz[1]), (pad + 1):(pad + sz[2])
    reference = big[rows, cols]
    secondary = similar(reference)
    tdx, tdy = similar(reference), similar(reference)
    for (jj, j) in enumerate(cols), (ii, i) in enumerate(rows)
        dx, dy = offset(ii, jj)
        tdx[ii, jj], tdy[ii, jj] = dx, dy
        secondary[ii, jj] = _sample(big, i + dy, j + dx)
    end
    return Float32.(reference), Float32.(secondary), tdx, tdy
end

warped_pair(n::Integer, offset; kw...) = warped_pair((Int(n), Int(n)), offset; kw...)

"""
    decorrelate(img, weight; amplitude, seed) -> Matrix

`img` with independent noise added, scaled by `weight(row, col)` in `[0, 1]`.

Noise present in one image of a pair and not the other is what actually limits a chip size: it
lowers the correlation a small chip can reach, so a larger chip is needed to recover the
displacement. Lowering contrast instead does nothing, because `ZNCC` normalizes amplitude away.
"""
function decorrelate(img, weight; amplitude::Real = 0.5, seed::Integer = 0)
    rng = MersenneTwister(seed)
    out = copy(img)
    for j in axes(out, 2), i in axes(out, 1)
        out[i, j] += oftype(out[i, j], amplitude * weight(i, j) * randn(rng))
    end
    return out
end
