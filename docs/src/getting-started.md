
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

You need two images of the same scene, on the same grid, differing by motion. Here they are synthetic so
the example is self-contained:

```@example start
using AutoRIFT
include("../figures.jl")  # plotting helpers and the synthetic scenes; see [Plotting](@ref)

# A texture, and a copy of it whose features moved 6 pixels left and 2 pixels down.
reference, secondary, _, _ = warped_pair(512, (row, col) -> (6.0, -2.0); seed = 13)

image_panels(reference, secondary)
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

Six pixels in x and minus two in y, which is what the scene was built with.

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

- **[A guided walkthrough](@ref)** — the same scene through eight steps, adding one capability at a time:
  chip size, grid spacing, multiple chip sizes, priors, and settings that vary across the scene. This is
  the page to read next.
- **[Geospatial data](@ref)** — if your images are rasters, what changes: `vx`/`vy` in map orientation,
  and velocities from an acquisition interval.
- **[How feature tracking works](@ref)** — what a chip, a search window, and a correlation surface are, if
  the vocabulary above was new.
- **[Judging a result](@ref)** — the gate, measured.
