
# Getting started

## Install

```julia
using Pkg
Pkg.add("AutoRIFT")
```

Nothing else is required to correlate arrays. Optional packages unlock additional input types, and each
is loaded by you rather than installed as a dependency:

| load this | to get |
|---|---|
| `Rasters` + `ArchGDAL` | geospatial input, velocities in map units |
| `DimensionalData` | labelled-array input, offsets in pixels |
| `ImageFeatures` | sparse first guesses from ORB features |
| `Metal` | the Apple GPU backend |

## One correlation

You need two images of the same scene, on the same grid, differing by motion. A photograph makes the
motion something you can see, so that is what this page uses — warped here rather than photographed
twice, so the answer is known:

```@example start
using AutoRIFT
include("../figures.jl")  # plotting helpers and the example scenes; see [Plotting](@ref)

# The surface slid sideways, fastest down the middle and slowing to the top and bottom edges, with a
# slight downward drift everywhere. Ice in a valley moves like this.
flow(row, col) = (4.0 + 12.0 * sin(pi * row / 512), -2.0)

reference, secondary, true_dx, true_dy = warped_pair(photo(), 512, flow)

image_panels(reference, secondary)
```

Look at the two panels before reading on. The crowd across the middle has shifted noticeably to the
left; Ali's shoulders at the top have barely moved. That difference *is* the measurement — a single
number for the whole frame would throw it away.

Here is the field that was applied, which the result below should reproduce:

```@example start
field_panels(true_dx, true_dy; titles = ("true dx (px)", "true dy (px)"))
```

That is the whole input. The call:

```@example start
out = autorift(reference, secondary)
```

## Read the result

`out` carries one value per grid point in each of several layers. The three you will use first:

```@example start
displacement_panels(out)
```

- **`dx`** — displacement along the second index (columns), in pixels
- **`dy`** — displacement along the first index (rows), positive downward
- **`correlation`** — how well the chip matched, in `[0, 1]`

```@example start
(measured = AutoRIFT.nmeasured(out), of_total = length(out.dx))
```

Points that could not be measured are `NaN`, not zero — zero is a real displacement and must not be
confused with an absent one. So filter before reducing:

```@example start
using Statistics: median
(median_dx = median(filter(isfinite, out.dx)), median_dy = median(filter(isfinite, out.dy)))
```

`dy` comes back at the −2 the whole scene was given. The median `dx` is less interesting, and that is
the point: `dx` is not one number here, it runs from about 4 at the edges to 16 down the middle, so
summarizing it with a median describes no part of the scene. Compare the `dx` panel above against the
true field rather than reducing it.

The grid is coarser than the image — 11×11 points by default, which is why the panels look blocky
beside the 512×512 input. `grid_spacing` sets that, and [A guided walkthrough](@ref) measures what it costs.

!!! note "The sign is the offset, not the motion"
    `dx`/`dy` point from the secondary image back to the reference, which is the *opposite* of how the
    features moved. A feature that moved six columns left reports `dx = +6`. This convention comes from
    the algorithm's heritage and is worth reading once: [Conventions](@ref).

## Judge it before using it

Never take every point. `correlation` is the gate:

```@example start
good = out.correlation .> 0.3
(kept = count(good), rejected = AutoRIFT.nmeasured(out) - count(good))
```

```@example start
quality_panels(out)
```

`autorift` also rejects spatial outliers by default, so points that disagree with their neighbours are
already gone. [Judging a result](@ref) is the page on which threshold to use and why `correlation` rather
than `peak_ratio`.

## Where to go next

- **[A guided walkthrough](@ref)** — a scene with a decorrelating shear band, through eight steps that add one capability at a time:
  chip size, grid spacing, multiple chip sizes, priors, and settings that vary across the scene. This is
  the page to read next.
- **[Geospatial data](@ref)** — if your images are rasters, what changes: `vx`/`vy` in map orientation,
  and velocities from an acquisition interval.
- **[How feature tracking works](@ref)** — what a chip, a search window, and a correlation surface are, if
  the vocabulary above was new.
- **[Judging a result](@ref)** — the gate, measured.
