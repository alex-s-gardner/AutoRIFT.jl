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

Returns layers on the output grid: the displacement, `correlation` (the peak
similarity), `chip_size` recording which chip-size level produced each estimate,
and `interpolated` marking values filled from neighbours rather than measured.

What the input is determines both the return type and the displacement convention,
because only some inputs know which way is north:

| input | returns | displacement |
|---|---|---|
| `Matrix` | [`AutoRIFT.MultichipResult`](@ref) | `dx`, `dy` — pixel offset, `dy` down rows |
| `AbstractDimArray` | `DimStack` | `dx`, `dy` — as above, with coordinates |
| `AbstractRaster` | `RasterStack` | `vx`, `vy` — feature motion, `+vy` north |

The last two need `DimensionalData` or `Rasters` loaded. The raster path is the only
one that can orient the result, so it is the only one that reports `vx`/`vy`: the
sign flip from the correlator's secondary-to-reference offset to actual feature
motion, and the flip from row-down to north-up, both happen there. A north-up and a
south-up raster of one scene therefore give the same answer. Radar slant-range
geometry has no CRS and takes the `DimStack` path unchanged.

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

# Import this machine's saved FFTW wisdom, so a fresh process does not re-measure plans it has
# already measured. Measured cost of not doing this: 822 ms for the three sizes a default run
# needs, against 158 ms for the correlation itself — paid on every process launch, which for a
# driver that runs one pair per process is every pair.
#
# `load_wisdom!` never throws and never blocks on anything but a small file read, which is what
# makes it safe here: an `__init__` that can fail makes the package unloadable, and one that can
# hang makes it unloadable in practice. See `src/plans.jl`.
function __init__()
    # Discard any plan cached during precompilation, before anything can execute one.
    #
    # This is not tidiness, it is a segfault. An FFTW plan is a handle to a C structure; caching
    # one in a `const Dict` means the precompile image serialises it, and on reload the handle
    # points nowhere. `mul!` on it crashes the process inside `fftwf_execute_dft_r2c` — verified,
    # and it is why the precompile workload below cannot simply be added without this line.
    #
    # Nothing is lost: what the workload precompiles is Julia code, and the *wisdom* from its
    # planning persists through the file below, so the first real plan is still cheap.
    clear_plans!()
    load_wisdom!()
    return nothing
end

export autorift, autorift!, reinit!
export SimilarityMeasure, ZNCC, NCC, Coherence
export PreprocessMethod, Highpass, Wallis, WallisGapfill, Sobel, Laplacian,
       Decibel, NoPreprocess
export SubpixelMethod, PyramidRefine, NoRefine
export QuantizeMethod, QuantizeUInt8, NoQuantize
export OutlierMethod, GardnerFilter, NoOutlierFilter
export ImagePair

# ---------------------------------------------------------------------------
# Precompilation
# ---------------------------------------------------------------------------
#
# First-call latency is a first-class cost here, not a developer annoyance. A production driver
# may launch a process per image pair, in which case compilation is paid per pair and no amount
# of kernel optimization shows up in the wall clock. Measured before this block: 2.35 s from
# `autorift` returning its first answer, against 4 ms for the second.
#
# The workload is deliberately small but *typed*: what costs time is specialising the pipeline
# for each element type the correlator can see, not running it on large inputs. So this uses the
# smallest images that produce a grid at all and covers the type axes that matter —
#
#   * `UInt8`, the reference's default and what `quantize = :uint8` produces;
#   * `Float32`, what `quantize = :none` produces after filtering;
#   * `Int16`, a raw sensor type that reaches the correlator unwidened.
#
# and the two entry points, since `autorift` and the `init`/`autorift!` cache path compile
# separately. The extensions precompile their own workloads: a package cannot precompile code
# that depends on a module it does not load.
#
# Deliberately *not* here: threading, which compiles the same `_track_chunk!` the serial path
# does; and the several preprocessing methods, which are cheap to specialise and rarely all used
# in one process. Both were measured as adding precompile time without moving TTFX.
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    # 150 px is the smallest square that yields a grid with the default chip size and radius, so
    # this exercises the whole path — grid construction, both passes, the outlier filter, the
    # merge — at the least possible cost. Values are arbitrary but must have texture, or the
    # correlator short-circuits on degenerate chips and the peak code never compiles.
    n = 150
    base = [Float32((i * 7 + j * 13) % 251) / 251 for i in 1:n, j in 1:n]
    shifted = circshift(base, (2, 3))

    @compile_workload begin
        for (a, b) in ((base, shifted),
                       (round.(UInt8, base .* 255), round.(UInt8, shifted .* 255)),
                       (round.(Int16, base .* 10_000), round.(Int16, shifted .* 10_000)))
            autorift(a, b; chip_size = 32, search_radius = 6, chip_size_max = 32)
            # The batch path, which compiles separately from the one-shot one.
            cache = init(a, b; chip_size = 32, search_radius = 6, chip_size_max = 32)
            autorift!(cache)
            reinit!(cache; secondary = b)
            autorift!(cache)
        end
    end
end

end # module
