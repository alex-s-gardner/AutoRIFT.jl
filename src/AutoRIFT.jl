"""
    AutoRIFT

Dense feature tracking by normalized cross-correlation: a pure-Julia
reimplementation of NASA JPL's [autoRIFT](https://github.com/nasa-jpl/autoRIFT),
the correlator behind the ITS_LIVE glacier velocity products.

Given two images of the same scene acquired at different times, `autorift`
estimates the displacement of surface features between them on a regular grid,
to sub-pixel precision.

```julia
out = autorift(image1, image2; chip_size = 32, search_radius = 25)
out.dx, out.dy, out.correlation
```

The core operates on plain `AbstractMatrix` in pixel coordinates, so it is
agnostic to whether the images are map-projected or in radar slant-range
geometry. Load `Rasters` or `DimensionalData` to accept dimensional arrays and
receive a stack with coordinates and CRS attached; those methods live in package
extensions, so the core stays cheap to load.
"""
module AutoRIFT

# `init` extends CommonSolve's, a zero-dependency package that exists to own the lifecycle
# names so packages agreeing on them do not each define their own. Costs nothing in load time,
# and a caller who knows the SciML idiom already knows half this API. `reinit!` is not
# CommonSolve's (it belongs to SciMLBase, which would be a heavy dependency for one name), so
# it is ours and is exported.
using CommonSolve: CommonSolve, init
using LinearAlgebra: LinearAlgebra
using StableTasks: StableTasks

"""
    autorift(reference, secondary; kwargs...)

Estimate the displacement of surface features between two images of the same
scene, on a grid, to sub-pixel precision.

Returns a stack of layers on the output grid: `dx` and `dy` in pixels,
`correlation` (the peak similarity), `chip_size_x`/`chip_size_y` recording which
chip-size level produced each estimate, and `interpolated` marking values that were
filled from neighbours rather than measured.

`reference` and `secondary` may be plain matrices in pixel coordinates, or —
with `Rasters` or `DimensionalData` loaded — dimensional arrays, in which case
the result carries coordinates and CRS. Radar slant-range geometry needs no CRS
and works unchanged.

See [`AutoRIFT.params`](@ref) for the full keyword list.
"""
function autorift end

"""
    autorift!(cache)

Run the correlator using preallocated buffers and FFT plans, writing into
`cache.result`.

The path for batch processing: `AutoRIFT.init` allocates and plans once,
`reinit!` swaps in the next image pair, and `autorift!` reuses everything. Across
many pairs this avoids repeating the allocation and FFT planning that would
otherwise dominate.

```julia
cache = AutoRIFT.init(first_pair...)
for (a, b) in pairs
    reinit!(cache; reference = a, secondary = b)
    save(autorift!(cache))
end
```
"""
function autorift! end

include("types.jl")
include("params.jl")
include("points.jl")

# Layer 1: template matching and peak location. These files depend on nothing else
# in the package, so they stay extractable as a standalone package -- the Julia
# ecosystem has no adequate NCC + subpixel-peak implementation.
include("plans.jl")
include("integral.jl")
include("correlate.jl")
include("peak.jl")

# Layer 2: sliding-window reductions and the quality control built on them.
include("window.jl")
include("resample.jl")
include("outliers.jl")
include("preprocess.jl")

# Layer 3: orchestration. The grid loop, then the multi-chip-size search built on it.
include("track.jl")
include("multichip.jl")
include("api.jl")

export autorift, autorift!, reinit!
export SimilarityMeasure, ZNCC, NCC, Coherence
export PreprocessMethod, Highpass, Wallis, WallisGapfill, Sobel, Laplacian,
       Decibel, NoPreprocess
export SubpixelMethod, PyramidRefine, NoRefine
export QuantizeMethod, QuantizeUInt8, NoQuantize
export OutlierMethod, GardnerFilter, NoOutlierFilter
export ImagePair

end # module
