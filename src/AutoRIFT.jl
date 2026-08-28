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
# `WallisGapfill` fills gaps with noise, and `Params.rng_seed` is what makes that reproducible.
using Random: Random
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
include("firstguess.jl")
include("track.jl")
include("multichip.jl")
include("tile.jl")
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
    # Discard the wisdom path resolved during precompilation, for a milder version of the same
    # reason. It is derived from the depot and the CPU model, and the machine that builds the image
    # need not be the machine that loads it — a relocated depot or a different CPU would otherwise
    # inherit the builder's path and read wisdom measured for the wrong microarchitecture. Cheap to
    # re-resolve: once per process.
    reset_wisdom_path!()
    load_wisdom!()
    return nothing
end

export autorift, autorift!, reinit!
export SimilarityMeasure, ZNCC, NCC, Coherence
export PreprocessMethod, Highpass, Wallis, WallisGapfill, Sobel, Laplacian,
       Decibel, Deramp, NoPreprocess
export SubpixelMethod, PyramidRefine, NoRefine
export OutlierMethod, GardnerFilter, NoOutlierFilter
export ImagePair
export FirstGuess, ORBGuess, AKAZEGuess, first_guess, scene_rotation
export RotationMethod, RotationSearch, NoRotationSearch

"""
    AutoRIFT.PUBLIC_NAMES

Names that are API but not exported: callable as `AutoRIFT.name`, and covered by semantic versioning.

Exports are the names a user needs in scope. These are the ones they may reach for deliberately —
the production input path, the pipeline steps `src/multichip.jl` promises are separately callable, and
the preprocessing functions `src/preprocess.jl` says must be usable ahead of reprojection.

One list rather than a `public` declaration scattered through the files, because three things have to
agree about it: this declaration, the documentation, and the test that every public name has a
docstring. A name absent here is internal and may change in a patch release.
"""
const PUBLIC_NAMES = (
    # Configuration.
    :params, :Params, :chip_sizes, :chip_measures, :measure_at, :filter_width, :filter_reach,
    # Geometry is two-dimensional throughout, and `Extent` is how it is spelled. Public because it
    # appears in `Params`' fields and in every geometry keyword, so a caller building one by hand
    # needs the name — and because a C caller's struct layout is derived from it.
    :Extent, :extent,
    # The grid, which is how per-point fields reach the correlator.
    :PointSet, :pointset, :gridpoints, :scatter, :rebuild, :sanitize!,
    :npoints, :nsearchable, :issearchable, :chip_bounds, :search_bounds,
    # Running it, and the results.
    :init, :autorift_with_grid, :Cache, :imagepair, :MultichipResult, :nmeasured,
    :DisplacementField, :displacement_field, :track, :track!,
    :correlate_multichip, :chipsize_level, :PassGeometry, :pass_geometry,
    # Preprocessing, standalone as well as inside a run.
    :ImageElement, :preprocess, :valid, :replace_nonfinite, :highpass, :highpass!, :wallis,
    :wallis_gapfill, :decibel, :sobel, :laplacian, :deramp, :ramp_phase,
    :FiniteMask, :resident,
    # Post-processing steps a caller may want on their own.
    :reject_outliers, :outlier_filter, :dilate_within, :resample, :resample!,
    :Nearest, :Area, :Bicubic, :window, :relax, :rescale,
    # Blocked processing: `halo` says how much overlap a block size costs. The layout types are
    # deliberately absent — they are the part free to change.
    :halo,
    # FFT plan warming, for a driver that wants it off the hot path.
    :warm_plans!,
    # First-guess plumbing the extensions build on.
    :consistent_matches, :required_package,
)

# `public` exists from Julia 1.11 and this package supports 1.10, so the declaration is guarded.
# `Expr(:public, ...)` rather than the surface syntax because that would be a parse error on 1.10 —
# and it is a lowering-level declaration with no runtime component, so it stays trim-inert.
@static if VERSION >= v"1.11"
    eval(Expr(:public, PUBLIC_NAMES...))
end

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
#   * `Float32`, what a filter produces from real input;
#   * `UInt8` and `Int16`, raw sensor types that reach the correlator unwidened.
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
            # (rotation is covered once, below, rather than per element type)
            cache = init(a, b; chip_size = 32, search_radius = 6, chip_size_max = 32)
            autorift!(cache)
            reinit!(cache; secondary = b)
            autorift!(cache)
        end
        # The rotation search, which `NoRotationSearch` does *not* cover: `_correlate_rotations!`
        # has a separate method per rotation type, so the calls above — all at the default
        # `rotation = nothing` — leave `_rotate_chip`, the `RotationSearch` method, and the whole
        # `Params{…,RotationSearch{…}}` specialisation of the track loop cold. Measured: 606 ms on
        # first call against 2.6 ms warm, which is 23x the fully-precompiled no-rotation call and
        # lands on exactly the sea-ice callers this path exists for. Precompilation is ~1% slower,
        # within noise, because it reuses everything above.
        #
        # Float32 only, and one angle count. The element types above buy nothing here — the
        # rotation buffer is `Float32` regardless of input type — and each distinct *angle count*
        # is a distinct `RotationSearch` type, so covering more than one would multiply precompile
        # cost for a caller who may use neither. Three angles is `RotationSearch`'s default and
        # `sea_ice_drift`'s.
        autorift(base, shifted; chip_size = 32, search_radius = 6, chip_size_max = 32,
                 rotation = true)
    end
end

end # module
