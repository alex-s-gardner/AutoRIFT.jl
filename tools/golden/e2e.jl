# The end-to-end ladder: AutoRIFT.jl from the granule to the correlator's output, one boundary at a
# time.
#
#   julia --project=tools/golden -t 8 tools/golden/e2e.jl S2B_MSIL1C_20200612
#   julia --project=tools/golden -t 8 tools/golden/e2e.jl S2B_MSIL1C_20200612 --all
#   julia --project=tools/golden -t 8 tools/golden/e2e.jl S2B_MSIL1C_20200612 --all --proj-only
#
# **Why this exists alongside `stages.jl`.** Every other comparison in this directory is fed the
# reference's own arrays: `capture.py` dumps the filtered, byte-quantized pair and the snapped grid
# at the `runAutorift` boundary, and `pointset_from_capture` turns the reference's grid into the
# `PointSet`. That isolates the correlator, which is what it is for, and it means nothing upstream of
# `runAutorift` has ever run in Julia on a golden case — not the granule read, not the coregistration,
# not the filter on Julia's own read, not the byte rescale, not the geogrid, not the driver's own
# parameter derivation. Six boundaries, each able to produce a plausible wrong answer.
#
# **The reference side needs no new compute.** Every boundary's answer is already on disk in the run
# directory: the nine `window_*.tif` geogrid rasters, `autoRIFT_intermediate.nc`, the `capture/`
# arrays, and `filtered/` for the pairs the driver filters before geogrid.
#
# **Each rung is fed the reference's input, not the previous rung's output.** A rung fed a correct
# input tells you about itself; chaining rebuilds the composed comparison this exists to take apart.
# The chained run is a separate mode, and its job is to state the end-to-end claim once the rungs are
# green.

include("manifest.jl")
include("reference.jl")
include("scenes.jl")
include("mtl.jl")
include("radar.jl")
include("stages.jl")

using AutoRIFT: chip_sizes, subpixel_at

using ArchGDAL, ImagePairGeometry, Printf, Statistics
using ImagePairGeometry: ProjectedCoordinate
import FastGeoProjections as FGP
# Loaded, not called: `FastGeoProjectionsProjExt` is what gives `proj_only` a pipeline to build, and an
# extension triggers on the package being present rather than on it being used.
import Proj

# ---------------------------------------------------------------------------
# Gates the geogrid needs and `stages.jl` does not
# ---------------------------------------------------------------------------

"""
    relative_stage(name, ref_name, jl, ref; tol = 1e-6) -> StageResult

Compare two `Float64` fields, reporting the largest difference both absolutely and relatively.

For the three `Float64` geogrid bands, which cannot be bitwise where the grid and the imagery are in
different projections, for two reasons that are properties of the reference rather than of either
implementation: it is compiled with floating-point contraction enabled, so `a*b + c` may be one
`fma` with one rounding where Julia performs two; and PROJ is not bit-reproducible across platforms,
while the kernel divides a difference of projected coordinates by the pixel spacing, so a couple of
ULP of input emerges amplified.

**The gate is on the absolute difference, and the relative one is reported beside it.** Every band
here multiplies a measured displacement in pixels: the off2vel entries give metres per year per pixel,
the scale factors are dimensionless. So an absolute difference converts directly into a velocity error
and a relative one does not.

Gating relatively would gate on the kernel's conditioning rather than on either implementation. The
shared determinant is a difference of products of nearly equal terms, and how much it amplifies a given
input difference varies by case: `FastGeoProjections` and PROJ agree to 1.9e-7 m in position and 1e-10
relative in the one-cell step the kernel consumes on *both* the Sentinel-2 case and the cross-zone
Landsat 8 one, while the resulting `off2vy_dy` disagreement is 1.7e-7 relative on the first and 4.2e-6
on the second — a 25x spread from an identical input. A relative bound calibrated on well-conditioned
cases reds a badly-conditioned one for nothing.

`atol` is 1e-3, which is both 224 times the largest absolute difference measured across the twelve
optical cases (4.46e-6, on `off2vy_dy`) and small enough to be harmless: on a three-hundred-pixel
displacement it is 0.3 m/yr, under a third of the 1 m/yr the product quantizes velocity to as `int16`.

**The radar path is gated relatively instead**, with `rtol`, because its bands are not the same
quantities: `off2vel` band 3 is a velocity per pixel along the image's own axis and runs to thousands, so
an absolute bound tuned for a coefficient near 75 is meaningless there. `ImagePairGeometry`'s
`REFERENCE.md` bounds the three bands that divide by the along-track step — `off2vx_dy`, `off2vy_dy` and
`off2vy_dr` — at **3.5e-4 relative**, and attributes it exactly: the step is measured between two solved
ground points and inherits the reference's own ~0.0013-line azimuth residual, whose maximum is 1.07e-4
relative. Those are the three bands this reds without it, at 1.073e-4.

The relative denominator is `max(|reference|, 1)`, so a value below one is judged absolutely there too.
Without that a band passing through zero reports an unbounded ratio for a difference of no consequence.
"""
function relative_stage(name, ref_name, jl, ref; atol = 1e-3, rtol = nothing)
    gate = rtol === nothing ? "abs<=$atol" : "rel<=$rtol"
    if size(jl) != size(ref)
        return StageResult(name, ref_name, gate, false, 0,
                           "shape $(size(jl)) against reference $(size(ref))")
    end
    worst = 0.0
    worst_abs = 0.0
    nz = 0
    skipped = 0
    for i in eachindex(jl, ref)
        a, b = Float64(jl[i]), Float64(ref[i])
        # A nodata sentinel is not a value. Where one side has it and the other a computed number the
        # difference is the size of the sentinel, which would dominate every statistic — that is a
        # *coverage* disagreement and `rounded_stage` on the integer bands is what counts it.
        if !isfinite(a) || !isfinite(b) || b == SENTINEL || a == SENTINEL
            a == b || (skipped += 1)
            continue
        end
        a == b && continue
        nz += 1
        worst = max(worst, abs(a - b) / max(abs(b), 1.0))
        worst_abs = max(worst_abs, abs(a - b))
    end
    n = length(ref)
    extra = skipped == 0 ? "" : @sprintf(", %d excluded as nodata on one side", skipped)
    passed = rtol === nothing ? worst_abs <= atol : worst <= rtol
    return StageResult(name, ref_name, gate, passed, n,
                       @sprintf("%d of %d differ (%.2f%%), max absolute %.4g, max relative %.4g%s",
                                nz, n, 100 * nz / n, worst_abs, worst, extra))
end

# The sentinel every geogrid band marks a point outside the image with (`geogridOptical.cpp:1039`).
const SENTINEL = -32767.0

# The radar path's two bounds, and they share one cause.
#
# `ImagePairGeometry`'s `REFERENCE.md` traces the along-track float bands to the reference's own
# ~0.0013-line azimuth offset — the step is measured between two solved ground points, so it inherits the
# residual — and bounds them at 3.5e-4 relative on its own cases, noting that the figure is not general:
# "the fixture's synthetic orbit produces a smaller residual, so 2e-4 was never a general bound."
#
# The same residual tips an azimuth *index* across a rounding boundary. Measured on the golden set: the
# Sentinel-1 pairs reach 1.073e-4 relative on the float bands and 0.065% of points off by one in
# `location_y`, and the NISAR L1 pair 3.504e-4 and 0.251% in `search_y`. Both bounds here are set above
# the NISAR figure, which is the larger, and both are the same mechanism rather than two.
#
# What would make them meaningful again is removing the residual rather than widening them: `REFERENCE.md`
# records it as the one part of the radar path no external reproduction escapes.
const RADAR_RTOL = 4e-4
const RADAR_MAX_VALUE_FRACTION = 5e-3

"""
    endpoint_stage(name, ref_name, jl, ref, step; max_median = step, max_bias = 0.01,
                   max_p99 = 1.0) -> StageResult

Compare two quantized displacement fields, gating on the *shape* of the residual and reporting its size.

`quantized_stage` gates on `exact >= 77.4%` and `within one step >= 97.3%`, the figures the pre-existing
L8/L9 benchmark reaches on a 3072² window. Those are the right numbers for that window and the wrong ones
for a gate over twenty-two different scenes, and `tools/golden/README.md` says so in as many words: the
gate is *similar* statistics, not identical ones, because the fraction of interpolated points, the spread
of chip sizes and the amount of fast flow all vary — "a pair that misses 77.4% by a few points with a
structureless residual passes; one that hits it with a dipole along the flow margin does not."

So this gates the three statistics that answer *structureless*, and reports `exact` and
`within one step` beside them:

  * the **median** residual, which a grid offset or a convention error moves off zero;
  * the **bias**, which a systematic difference moves and tie-breaking does not;
  * the **p99**, which bounds the tail a gradient-correlated residual produces.

Thresholds from the twelve optical cases rather than chosen: every one has a median of exactly 0, a
|bias| no larger than 0.0075 px and a p99 no larger than 0.75 px, while `exact` spans 62.8% to 79.7%.
The bounds here are one upsampling step, 0.01 px and 1.0 px — modest headroom on each, and each one a
quantity the README names as what a structured residual would move.

`dev/GATES.md` records the per-case `exact` and `within` figures, which is where a regression that
preserves the shape while losing agreement would show up.
"""
function endpoint_stage(name, ref_name, jl, ref, step; max_median = step, max_bias = 0.01,
                        max_p99 = 1.0)
    if size(jl) != size(ref)
        return StageResult(name, ref_name, "structure", false, 0,
                           "shape $(size(jl)) against reference $(size(ref))")
    end
    both = 0; only_j = 0; only_r = 0; exact = 0; within = 0
    d = Float64[]
    for i in eachindex(jl, ref)
        mj = isnan(jl[i]); mr = isnan(ref[i])
        mj && mr && continue
        if mr
            only_j += 1
        elseif mj
            only_r += 1
        else
            both += 1
            δ = Float64(jl[i]) - Float64(ref[i])
            push!(d, δ)
            δ == 0 && (exact += 1)
            abs(δ) <= step && (within += 1)
        end
    end
    both == 0 && return StageResult(name, ref_name, "structure", false, 0,
                                    "no point measured on both sides")
    ad = abs.(d)
    med = median(ad)
    p99 = quantile(ad, 0.99)
    # **The core bias, not the plain mean.** `regate.jl` established the distinction on the radar cases
    # and the reasoning carries here: a pair correlating over SAR speckle puts a few hundred of its points
    # on the far side of a nearly flat peak surface, two-sided and tens of pixels out, which drags the
    # mean while the points that agree at all sit near zero. Gating the mean sets a threshold around that
    # tail's cancellation, which is noise; the core catches the systematic offset a bias threshold is for,
    # and the tail is already bounded by the p99.
    core = [x for x in d if abs(x) <= 1.0]
    bias = isempty(core) ? mean(d) : mean(core)
    passed = med <= max_median && abs(bias) <= max_bias && p99 <= max_p99
    detail = @sprintf("both %d, only jl %d, only ref %d; median %.4g, bias core %+.5f (mean %+.5f), \
                       p99 %.4g (exact %.2f%%, within one step %.2f%%)",
                      both, only_j, only_r, med, bias, mean(d), p99,
                      100exact / both, 100within / both)
    gate = @sprintf("median<=%.4g |bias|<=%.3g p99<=%.3g", max_median, max_bias, max_p99)
    return StageResult(name, ref_name, gate, passed, both, detail)
end

"""
    rounded_stage(name, ref_name, jl, ref; budget = 2) -> StageResult

Compare two integer geogrid bands, allowing at most `budget` points to differ by one.

The integer bands are decisions and the gate on a decision is normally exactness — that is
[`exact_stage`](@ref)'s discipline and `stages.jl` keeps it. One thing breaks it here, and only here:
each of these bands is `std::round` of a value derived from a projected coordinate, so a point whose
pre-round value sits on a half-integer is decided by the projection library's last bits and *neither*
answer is wrong.

Measured rather than assumed. Across the eight same-CRS optical cases, `FastGeoProjections` and PROJ
produce identical values for every integer band at every point but one, out of 33.8 million: on
`LC08_L1TP_062018` a single `search_x` lands on 34 against 35. Scaling that point's search range by a
relative 1e-9 flips PROJ to 34 as well, so its own pre-round value is within about 3.5e-8 of a pixel
of the boundary — the two transforms' one-cell step agreement is 7.3e-11 relative, which is enough to
straddle it.

So the budget admits a tie and nothing else. A convention error — the interval, the pair order, the
read offset — moves tens of thousands of points, not two, and the count is in the detail line either
way so a green rung still says how many differed.
"""
function rounded_stage(name, ref_name, jl, ref; budget::Int = 2, max_fraction = 1e-3,
                      max_coverage = 1e-5)
    if size(jl) != size(ref)
        return StageResult(name, ref_name, "exact", false, 0,
                           "shape $(size(jl)) against reference $(size(ref))")
    end
    sent = Int(SENTINEL)
    bad = 0            # both sides have a value and the values differ
    cover = 0          # one side has the sentinel and the other a value
    worst = 0
    first_bad = nothing
    for i in eachindex(IndexCartesian(), jl)
        a, b = Int(jl[i]), Int(ref[i])
        a == b && continue
        if a == sent || b == sent
            cover += 1
            continue
        end
        bad += 1
        worst = max(worst, abs(a - b))
        first_bad === nothing && (first_bad = (Tuple(i), a, b))
    end
    n = length(ref)
    # A coverage difference is a *decision* about whether a point is in the image, and on the radar path
    # the footprint is solved for rather than transformed, so a point within a metre of the edge can fall
    # either way. Counted separately from a value difference, which is the thing a convention error moves.
    passed = worst <= 1 && bad <= max(budget, max_fraction * n) && cover <= max_coverage * n
    detail = if bad == 0 && cover == 0
        "all $n equal"
    else
        pos, a, b = first_bad === nothing ? ((0,), 0, 0) : first_bad
        @sprintf("%d of %d differ in value by at most %d%s, %d differ in coverage",
                 bad, n, worst,
                 first_bad === nothing ? "" : @sprintf(" (first at %s: julia %d, reference %d)",
                                                       pos, a, b),
                 cover)
    end
    return StageResult(name, ref_name, "value<=1 & <=0.1%, coverage<=0.001%", passed, n, detail)
end

# ---------------------------------------------------------------------------
# What a case's Julia-side inputs are
# ---------------------------------------------------------------------------

"""
    Setup

Everything the ladder derives from a case before any rung runs: the two scene footprints, the
coregistered pair, the parameter region, the output grid and the window on it, and the geogrid result
over that window.

Held as one object because every rung reads some of it and rebuilding it costs a shapefile read, two
remote scene opens and twelve windowed parameter reads. The `PairGeometry` in particular is what two
rungs compare and a third derives its parameters from, so it is computed once here rather than per
rung.

Field types are left open: they come from `ImagePairGeometry` and naming them here would couple this
harness to that package's type parameters without making any rung faster, since a rung's cost is the
band comparison rather than the field access.
"""
struct Setup
    case::GoldenCase
    run::String
    reference_path::String
    secondary_path::String
    pair::Any
    epsg::Int
    info::Any
    grid::Any
    window::CartesianIndices{2}
    transform::Any
    geometry::Any
end

"""
    resolve_run(c::GoldenCase) -> String

The cached reference run holding the artifacts the ladder compares against.

The run number is not a fixed convention across the cached set — a capture is 100 on some cases and
200 on others — so the directory is chosen by *what it contains* rather than by its name: it must
hold the geogrid rasters, and a run that also holds a `capture/call1.json` is preferred because the
rungs downstream of the geogrid need those arrays too.

Errors rather than falling back to an incomplete run, since a rung silently compared against a
missing file would report red for the absence rather than for a disagreement.
"""
function resolve_run(c::GoldenCase)
    root = runs_dir(c)
    isdir(root) || error("no reference runs for $(c.product) under $root; run reference.jl first")
    complete = String[]
    for d in sort(readdir(root; join = true))
        isdir(d) && isfile(joinpath(d, "window_location.tif")) && push!(complete, d)
    end
    isempty(complete) && error("no cached run of $(c.product) holds window_location.tif; the " *
                               "geogrid rasters are what rungs 5.0 and 5.5 compare against")
    withcapture = filter(d -> isfile(joinpath(d, "capture", "call1.json")), complete)
    return isempty(withcapture) ? last(complete) : last(withcapture)
end

"""
    setup(c::GoldenCase; n = nothing) -> Setup

Resolve `c`'s scenes, coregister them, look up the parameter region, and derive the output grid.

`n` names a specific reference run; by default [`resolve_run`](@ref) picks the cached one that holds
what the rungs need.

This is the reference's own order of operations: `coregister` first, because geogrid's geotransform
and size are the *overlap's* rather than either scene's (`testGeogridOptical.py:92-100`), then the
region lookup on the overlap's centre, then the grid from the region's DEM.
"""
function setup(c::GoldenCase; n::Union{Integer,Nothing} = nothing, proj_only::Bool = false)
    run = n === nothing ? resolve_run(c) : run_dir(c, n)
    isdir(run) || error("no reference run at $run; run reference.jl or intermediate.jl first")

    (startswith(c.platform, "S1") || c.platform == "NISAR-L1") &&
        return _radar_setup(c, run, proj_only)

    # Reprojection first, because the footprints geogrid intersects — and the pixel grid its indices
    # count in — are the warped ones. A pair already in one projection comes back untouched.
    #
    # A NISAR L2 GSLC is geocoded and its two amplitude rasters are already on one map grid, so the pair
    # is read from the run directory instead — see `gslc_amplitude_paths` on why they are not rebuilt.
    rpath, spath = c.platform == "NISAR-L2" ? gslc_amplitude_paths(c, run) : aligned_scenes(c)
    rfp = scene_footprint(rpath)
    sfp = scene_footprint(spath)
    pair = coregister(rfp, sfp; dt = geogrid_seconds(c))

    epsg = footprint_epsg(rfp)
    info = parameter_info(pair_centroid(pair.coordinate, epsg)...)
    grid = parameter_grid(info)
    tf = grid_transform(info.epsg, epsg; proj_only)
    window = grid_window(grid, footprint_bounds(tf, pair.coordinate))

    # The nodata value the reference applies to five rasters, read from DEM band 1 as it does
    # (`geogridOptical.cpp:337-339`). Taken from the file rather than hardcoded so a region whose DEM
    # carries a different sentinel is handled, and with `nothing` becoming `0.0` because the reference
    # passes `pbSuccess = NULL` and so cannot distinguish an unset nodata from zero.
    nd = ArchGDAL.getnodatavalue(ArchGDAL.getband(ArchGDAL.read(info.paths.dem), 1))
    g = pairgeometry(grid, pair, geometry_inputs(info, window); transform = tf, window,
                     nodata = nodata_from(nd === nothing ? 0.0 : Float64(nd)))

    return Setup(c, run, rpath, spath, pair, epsg, info, grid, window, tf, g)
end

"""
    _radar_setup(c, run, proj_only) -> Setup

`setup` for a Sentinel-1 pair, whose geometry comes from burst annotations rather than from a raster.

Four things differ from the projected path and each is the reference's own choice:

  * The pair is [`s1_pair`](@ref)'s — the reference acquisition's merged radar grid and an interval —
    since `process_slc` copies that metadata for the secondary and replaces only the sensing times.
  * The transform maps the grid to **geodetic degrees**, not to an image's own projection: a radar
    footprint is solved for with `rdr2geo` rather than transformed, so `footprint_bounds` calls
    `transform(lon, lat, h)`.
  * The nodata sentinel is read from the **`vx`** raster, not the DEM (`geogridRadar.cpp:509-512`
    against `geogridOptical.cpp:339`). Reading it from the DEM here would apply a different sentinel to
    five bands than the reference did.
  * There is no scene path to read a footprint from, so those fields are the radar rasters the driver
    wrote, for the rungs that want to name them.
"""
function _radar_setup(c::GoldenCase, run::AbstractString, proj_only::Bool)
    # A NISAR L1 RSLC is one acquisition on one radar grid, so its pair is read straight off the two
    # products; a Sentinel-1 pair has to be assembled from burst annotations and mosaicked, from ASF for
    # a full granule and from the synthesized SAFE for a burst job.
    pair = c.platform == "NISAR-L1" ? nisar_l1_pair(c, run) :
           c.platform == "S1-BURST" ? s1_burst_pair(c, run) : s1_pair(c, run)
    coord = pair.coordinate

    # The footprint in geodetic degrees, which is what the region lookup takes — `bounding_box(..., 
    # epsg = 4326)` in `s1_isce3.process_slc`.
    b = footprint_bounds(IdentityTransform(), coord)
    lon = (b.X[1] + b.X[2]) / 2
    lat = (b.Y[1] + b.Y[2]) / 2
    info = parameter_info(lon, lat)
    grid = parameter_grid(info)

    tf = grid_transform(info.epsg, 4326; proj_only)
    window = grid_window(grid, footprint_bounds(tf, coord))

    nd = ArchGDAL.getnodatavalue(ArchGDAL.getband(ArchGDAL.read(info.paths.vx), 1))
    g = pairgeometry(grid, pair, geometry_inputs(info, window); transform = tf, window,
                     nodata = nodata_from(nd === nothing ? 0.0 : Float64(nd)))

    return Setup(c, run, joinpath(run, "reference.tif"), joinpath(run, "secondary.tif"),
                 pair, info.epsg, info, grid, window, tf, g)
end

"""
    grid_transform(grid_epsg, scene_epsg; proj_only = false) -> TransformPair

The grid-to-scene transform, as one `FastGeoProjections.Transformation` and its inverse.

`proj_only` forces the Proj-backed pipeline where a native implementation exists. It is the
attribution knob, not a correctness one: the reference builds its transforms with
`osr.CoordinateTransformation`, so forcing Proj takes the projection library out of the comparison and
leaves whatever remains belonging to the kernel arithmetic.

One interface rather than two, because `FastGeoProjections` already is one — `proj_only` is its own
keyword and it falls back to Proj for any CRS pair it has no native implementation for, so there is no
second library to construct against. The inverse comes from `inv` rather than from a second call on the
swapped pair: the element type and math kernel are carried by the operator's type rather than by its
fields, so rebuilding from the EPSG codes would silently revert both.

The choice is nearly invisible, measured rather than assumed. Over the eight same-CRS optical cases the
two produce **identical integer bands at every point but one of 33.8 million** — a `search_x` sitting
within 3.5e-8 of a pixel of a rounding boundary, see [`rounded_stage`](@ref). They disagree by 1.7e-7 m
in position and 7.3e-11 relative in the one-cell step the kernel consumes, which reaches the `Float64`
bands as at most 1.3e-5 metres per year per pixel of displacement.

Equal codes short-circuit to the identity, which is what `fast_transform` does and what makes the one
Antarctic case bitwise on every band: its scene is already in the parameter region's own projection.
"""
function grid_transform(grid_epsg::Integer, scene_epsg::Integer; proj_only::Bool = false)
    grid_epsg == scene_epsg && return transform_pair(IdentityTransform())
    f = FGP.Transformation(FGP.EPSG(Int(grid_epsg)), FGP.EPSG(Int(scene_epsg));
                           always_xy = true, proj_only)
    return TransformPair(f, inv(f))
end

# ---------------------------------------------------------------------------
# Rung 5.0 — the output grid
# ---------------------------------------------------------------------------

"""
    rung_grid(s::Setup) -> Vector{StageResult}

The output grid's geotransform and size, against `window_location.tif`'s own.

First because it is the rung every other one is expressed in: a grid off by one point compares two
different problems at every position after it, and does so while every band still looks plausible.
Gated exactly — a geotransform is six numbers copied from the parameter DEM and shifted by an
integer window offset, so there is nothing here to round.
"""
function rung_grid(s::Setup)
    path = joinpath(s.run, "window_location.tif")
    isfile(path) || return [StageResult("5.0 output grid", "window_location.tif", "exact", false,
                                        0, "missing $path")]
    ds = ArchGDAL.read(path)
    ref_gt = Tuple(ArchGDAL.getgeotransform(ds))
    ref_size = (ArchGDAL.width(ds), ArchGDAL.height(ds))
    gt = ImagePairGeometry.window_geotransform(s.grid, s.window)
    return [exact_stage("5.0 grid geotransform", "window_location.tif", collect(gt),
                        collect(ref_gt)),
            exact_stage("5.0 grid size", "window_location.tif", collect(size(s.window)),
                        collect(ref_size))]
end

# ---------------------------------------------------------------------------
# Rung 5.1 — the coregistration window
# ---------------------------------------------------------------------------

"""
    rung_window(s::Setup) -> Vector{StageResult}

The overlap window the pipeline correlates, against the shape of the imagery it correlated.

`GeogridOptical.coregister` intersects the two footprints and returns each image's offset into the
overlap plus its size, and `loadProductOptical` then reads exactly that window out of each scene
(`testautoRIFT.py:120-124`). So the shape of the reference's own `in_I1` *is* the overlap, and
comparing against it checks the intersection arithmetic — the four bounds, the `nround` index
conversion and the four failure conditions — without needing a pixel.

Worth its own rung because the overlap is the full scene whenever the two share a grid, which is eight
of the twelve optical pairs. On those the assertion is weak; on the four cross-projection pairs the
overlap is a strict subset — 11338 x 15101 of a warped scene — and the arithmetic is exercised.

The offsets are reported rather than gated: the reference does not record them, and they matter only
once the imagery is Julia's own, which is rungs 5.3 and 5.4.
"""
function rung_window(s::Setup)
    call = joinpath(s.run, "capture", "call1.json")
    isfile(call) || return [StageResult("5.1 overlap window", "capture/in_I1", "exact", true, 0,
                                       "skipped: no capture at $call")]
    k = read_capture(s.case; n = parse(Int, basename(s.run)))

    # The capture's arrays are in the reference's own `(row, col)` order, so the overlap's `(x, y)` size
    # reverses to compare.
    out = [exact_stage("5.1 overlap window", "in_I1",
                       collect(reverse(_coord_size(s.pair.coordinate))),
                       collect(size(k.arrays["in_I1"])))]
    push!(out, StageResult("5.1 scene offsets", "coregister", "reported", true, 2,
                           @sprintf("reference offset %s, secondary offset %s into the overlap",
                                    string(s.pair.reference_offset),
                                    string(s.pair.secondary_offset))))
    return out
end

# A coordinate's extent as `(x, y)`, whichever kind it is: a projected image carries `size`, a radar one
# carries the two counts separately because range and azimuth are not interchangeable.
_coord_size(c) = hasproperty(c, :size) ? c.size : (c.nsamples, c.nlines)

# ---------------------------------------------------------------------------
# Rung 5.5 — the geogrid
# ---------------------------------------------------------------------------

"""
    rung_geogrid(s::Setup) -> Vector{StageResult}

Every band of the geogrid, against the nine `window_*.tif` the container wrote.

Integer bands are gated exactly and `Float64` bands relatively, which is `ImagePairGeometry`'s own
two-tier standard and the reason the tiers split where they do: `window_location`, `window_offset`,
`window_search_range`, the two chip-size files and the stable-surface mask all pass through a
rounding or truncating conversion that absorbs a last-bit difference, so exact agreement is
achievable. The off2vel and scale-factor bands do not.

`reference_files` supplies the band layout for the pair's coordinate system, because the off2vel
files carry two bands on the projected path and three on the radar one and a positional reader
cannot assume either.
"""
function rung_geogrid(s::Setup)
    r = s.geometry
    radar = !(s.pair.coordinate isa ProjectedCoordinate)
    out = StageResult[]
    for (file, fields) in ImagePairGeometry.reference_files(s.pair.coordinate)
        path = joinpath(s.run, file)
        if !isfile(path)
            push!(out, StageResult("5.5 $file", file, "exact", false, 0, "missing $path"))
            continue
        end
        ds = ArchGDAL.read(path)
        for (b, f) in enumerate(fields)
            ref = ArchGDAL.read(ds, b)
            mine = getproperty(r, f)
            name = "5.5 geogrid $f"
            push!(out, if eltype(mine) <: Integer
                      rounded_stage(name, file, mine, Int32.(ref);
                                    max_fraction = radar ? RADAR_MAX_VALUE_FRACTION : 1e-3)
                  elseif radar
                      relative_stage(name, file, mine, ref; rtol = RADAR_RTOL)
                  else
                      relative_stage(name, file, mine, ref)
                  end)
        end
    end
    return out
end

# ---------------------------------------------------------------------------
# Rung 5.6 — the driver's own parameter derivation
# ---------------------------------------------------------------------------

"""
    rung_params(s::Setup) -> Vector{StageResult}

The scene-wide correlator settings `AutoRIFT.params(::PairGeometry)` derives, against the scalars the
reference recorded at its own `runAutorift` boundary.

This is the driver's arithmetic rather than the correlator's — `ChipSize0X`, `ScaleChipSizeY`,
`GridSpacingX` and the per-level upsampling ladder (`testautoRIFT.py:245-250, 330-334`) — and until
now it existed only in this directory, reading the captured scalars rather than deriving them. A
harness that reads a value it is meant to be checking cannot fail, so the derivation moved into the
package and this rung is what holds it to the reference.

Skipped rather than red when the case has no capture on disk: the scalars are what it compares
against, and their absence is not a disagreement.
"""
function rung_params(s::Setup)
    call = joinpath(s.run, "capture", "call1.json")
    isfile(call) || return [StageResult("5.6 driver parameters", "capture/call1.json", "exact",
                                        true, 0, "skipped: no capture at $call")]
    k = read_capture(s.case; n = parse(Int, basename(s.run)))
    p = AutoRIFT.params(s.geometry)

    chip0 = Int(k.scalars["ChipSize0X"])
    scale_y = Float64(k.scalars["ScaleChipSizeY"])
    spacing = Int(k.scalars["GridSpacingX"])
    # `ChipSizeMaxX` is a per-point array; the scene-wide maximum is what sets the top pyramid level.
    maxchip = Int(maximum(k.arrays["in_ChipSizeMaxX"]))
    # `autoRIFT.py:650`, the same expression the correlator applies to every level.
    chip_y(x) = round(Int, x * scale_y / 2) * 2

    out = StageResult[]
    push!(out, exact_stage("5.6 base chip size", "ChipSize0X",
                           [p.chip_size_min.X, p.chip_size_min.Y], [chip0, chip_y(chip0)]))

    # The geogrid's own maximum over the points that fall inside the image, which is what
    # `params(::PairGeometry)` derives and the only maximum a geometry knows.
    sent = Int32(s.geometry.nodata.output)
    inside = s.geometry.location_x .!= sent
    geo_max = maximum(s.geometry.chip_max_x[inside])
    push!(out, exact_stage("5.6 max chip size", "window_chip_size_max.tif",
                           [p.chip_size_max.X, p.chip_size_max.Y],
                           [Int(geo_max), chip_y(Int(geo_max))]))

    # **The driver's effective maximum can be lower, and that is not this derivation being wrong.**
    # `testautoRIFT.py:402` zeroes `ChipSizeMaxX` wherever `noDataMask` is set, and `noDataMask` is the
    # *imagery's* zero mask sampled at each grid point (`:349`) rather than anything geometric — so a
    # pair whose overlap is largely gap or fill loses its coarsest level entirely. Reported rather than
    # gated, because a `PairGeometry` carries no imagery and so cannot reproduce it; a caller that needs
    # the reference's level count passes `chip_size_max` explicitly.
    if maxchip != Int(geo_max)
        push!(out, StageResult("5.6 driver max after nodata", "in_ChipSizeMaxX", "reported", true, 1,
                               @sprintf("geogrid reaches %d over in-image points, the driver hands \
                                         the correlator %d; %d grid points lose a level to the \
                                         imagery's zero mask",
                                        Int(geo_max), maxchip,
                                        count(==(geo_max), s.geometry.chip_max_x[inside]))))
    end
    push!(out, exact_stage("5.6 grid spacing", "GridSpacingX",
                           [p.grid_spacing.X, p.grid_spacing.Y], [spacing, spacing]))

    # The upsampling ladder, per level rather than as one factor: the reference looks the factor up
    # by chip size (`autoRIFT.py:652-653`) and a single value cannot express `{24:16, 48:32, 96:64}`.
    ratios = k.scalars["OverSampleRatio"]
    levels = chip_sizes(p)
    if ratios isa Number
        push!(out, StageResult("5.6 upsampling ladder", "OverSampleRatio", "exact", true,
                               1, "reference used one factor, $ratios, for every level"))
    else
        mine = [subpixel_at(p, i).upsampling for i in eachindex(levels)]
        theirs = [Int(ratios[Symbol(levels[i].X)]) for i in eachindex(levels)]
        push!(out, exact_stage("5.6 upsampling ladder", "OverSampleRatio", mine, theirs))
    end
    return out
end

# ---------------------------------------------------------------------------
# Rung 5.7 — the endpoint
# ---------------------------------------------------------------------------

"""
    rung_endpoint(s::Setup) -> Vector{StageResult}

`autorift` on the grid and parameters the ladder derived, against the reference's own `Dx`/`Dy`.

This is the rung the others exist to make interpretable. Rungs 5.0, 5.5 and 5.6 establish that both
sides *would be handed* the same grid, priors, search limits, chip bounds and scene-wide parameters;
this is the first one that asks whether running on them gives the same answer.

**The imagery is the reference's own, and deliberately so.** `capture.py` dumps `in_I1` and `in_I2` at
the `runAutorift` boundary — filtered and quantized to bytes — so feeding those holds the one input the
ladder has not yet reproduced fixed, and what remains under test is the geogrid handoff composed with
the correlator. Rungs 5.1, 5.3 and 5.4 replace the imagery in turn; until they do, a disagreement here
belongs to the grid or to the correlator and not to a filter.

Two conventions, both taken from `correlator.jl` rather than restated: `arImgDisp_*` cuts its chip from
its second argument and the driver calls it `arImgDisp(I2, I1)`, so `I1` binds to AutoRIFT.jl's
*secondary*; and the reference's `Dy` is up-positive where AutoRIFT.jl's is row-positive, so one axis
needs a flip. The flip is *measured* — both signs are scored and the better kept — because a hardcoded
one is right only while the writer's convention holds.

Gated as `correlator.jl`'s endpoint is: the base level is quantized and compared on `exact`, while above
it both sides replace their measurements with a bicubic resize and neither field is quantized, so bias
and the within-one-step fraction carry the verdict there.
"""
function rung_endpoint(s::Setup)
    call = joinpath(s.run, "capture", "call1.json")
    isfile(call) || return [StageResult("5.7 endpoint", "capture/out_Dx", "gate", true, 0,
                                       "skipped: no capture at $call")]
    k = read_capture(s.case; n = parse(Int, basename(s.run)))

    grid = AutoRIFT.pointset(s.geometry; pixel_size = ImagePairGeometry.xsize(s.pair.coordinate))
    p = AutoRIFT.params(s.geometry; threaded = Threads.nthreads() > 1, preprocess = :none)

    out = StageResult[]

    # **The imagery has to fit.** A NISAR L2 GSLC is 110085 x 54885 bytes, so the pair the reference
    # correlated is 11 GiB before the correlator's own working set — and the capture holds it as two plain
    # arrays. Declined with the size rather than attempted: an unblocked run on that pair does not finish,
    # and the thinned endpoint on these cases is what gate `3.nisar` already measures.
    # From the manifest, not from the array: asking the array its size is what loads it.
    pixels = prod(JSON3.read(read(call, String)).arrays["in_I1"].shape)
    if 2 * pixels > 4 * 2^30
        push!(out, StageResult("5.7 endpoint", "capture/out_Dx", "gate", true, 0,
                               @sprintf("deferred: the pair is %.2f GiB of imagery, which an \
                                         unblocked run cannot hold; the thinned endpoint on this case \
                                         is gate 3.nisar", 2 * pixels / 2^30)))
        return out
    end
    a, b = k.arrays["in_I1"], k.arrays["in_I2"]

    # **The reference correlates a truncated grid, and AutoRIFT.jl does not.** `autoRIFT.py:809-819`
    # chops both axes to a multiple of `chopFactor = max(ChipSizeMaxX) / ChipSize0X` before correlating —
    # 4 on the golden S2 case, taking a 1009-point grid to 1008 — and keeps `origSize` only to report it.
    # AutoRIFT.jl keeps those points, which `tools/ab/README.md` measures as costing the reference 1.6% of
    # its grid on another scene, so the shapes legitimately differ and the comparison runs on the overlap.
    #
    # Verified rather than assumed: the reference's own correlated shape has to be exactly what the chop
    # predicts from AutoRIFT.jl's grid. That catches a grid difference, which would otherwise hide inside
    # "the shapes differ because of the chop".
    chop = Int(maximum(k.arrays["in_ChipSizeMaxX"])) ÷ Int(k.scalars["ChipSize0X"])
    predicted = [fld(n, chop) * chop for n in size(grid.x)]
    push!(out, exact_stage("5.7 grid shape after the reference's chop", "in_xGrid",
                           predicted, collect(size(k.arrays["in_xGrid"]))))
    predicted == collect(size(k.arrays["in_xGrid"])) || return out

    # Truncated and masked to the points the reference actually correlated, so the two answer the same
    # question. Both are the driver's own steps and neither is AutoRIFT.jl's: it keeps the points the chop
    # discards, and it has no imagery mask to zero a radius from.
    grid = _mask_from_capture(_chop_to(grid, predicted[1], predicted[2]), k)

    @info "5.7 correlating" scene=size(a) npoints=length(grid.x) chip=p.chip_size_min threads=Threads.nthreads()
    r = AutoRIFT.autorift(b, a, grid, p)

    rdx, rdy = k.arrays["out_Dx"], k.arrays["out_Dy"]
    ny = min(size(r.dx, 1), size(rdx, 1))
    nx = min(size(r.dx, 2), size(rdx, 2))
    step = 1 / AutoRIFT.subpixel_at(p, 1).upsampling

    # **Which statistic carries the verdict depends on whether the base level was reached.** Its values
    # are quantized to the upsampling step and `exact` is meaningful; above it both sides overwrite their
    # measurements with a bicubic resize (`autoRIFT.py:856-866`) and neither field lands on any grid, so
    # `exact` is near zero by construction. Decided from the reference's own values rather than assumed:
    # the golden L7 pair resolves only chips 32 and 64, and gating it on `exact` would red a case whose
    # residual is a median of zero and a p99 of 0.21 px.
    on_grid = count(v -> !isnan(v) && abs(v / step - round(v / step)) < 1e-6, rdx) /
              max(1, count(!isnan, rdx))
    quantized = on_grid > 0.5
    push!(out, StageResult("5.7 reference quantization", "out_DX", "reported", true, 1,
                           @sprintf("%.2f%% of the reference's own dx lands on the 1/%d grid, so the \
                                     gate is %s", 100on_grid, round(Int, 1 / step),
                                    quantized ? "exact and within-one-step" : "bias and spread")))

    for (axis, ref) in ((:dx, rdx), (:dy, rdy))
        jl = getproperty(r, axis)[1:ny, 1:nx]
        rf = ref[1:ny, 1:nx]
        push!(out, _by_level(axis, jl, axis === :dy ? -rf : rf, r.chip_size[1:ny, 1:nx]))
        # Both signs scored, the better kept: `dy` needs the flip and `dx` does not, and measuring says so
        # rather than a comment asserting it.
        best = nothing
        for sgn in (1, -1)
            nm = "5.7 $axis (sign $(sgn > 0 ? '+' : '-'))"
            rn = "out_$(uppercase(String(axis)))"
            st = quantized ? endpoint_stage(nm, rn, jl, sgn .* rf, step) :
                 unquantized_stage(nm, rn, jl, sgn .* rf)
            best === nothing && (best = (sgn, st))
            best = _better_endpoint(best, (sgn, st))
        end
        push!(out, last(best))
    end
    return out
end

"""
    rung_coregister(s::Setup) -> Vector{StageResult}

Rung 5.2 — the merged radar mosaic the correlator is handed, against `reference.tif`.

[`radar_mosaic`](@ref) reproduces `merge_swaths` and `merge_bursts_in_swath`: bursts into a subswath with
the azimuth seam halfway through each overlap, then subswaths onto the near-range origin with a
first-come writer. `read_slc_gdal` takes `np.abs` on the way in, so the whole layout is `Float32`
amplitude and nothing is resampled by hyp3 itself.

**Gated on the zero/non-zero set exactly and the values on tolerance.** Which pixels the mosaic fills is
pure index arithmetic — every offset, valid window and the 64-sample far-range buffer — so a single index
error moves the set and there is nothing to round. The values cannot be bitwise: they travel through a
`Float32` GeoTIFF and COMPASS's own amplitude differs from the raw burst's in the last digits.

**Only a burst job can be reached.** Its SAFE is on disk beside the outputs, where a full-SLC pair's
granule is not — the run directory holds `reference.tif` and no source product, so that pair would need
an 8 GiB transfer to read a pixel from.
"""
function rung_coregister(s::Setup)
    s.case.platform == "S1-BURST" || return [StageResult("5.2 radar mosaic", "reference.tif", "set",
                                                         true, 0,
                                                         "skipped: only a burst job keeps its SAFE in \
                                                          the run directory; a full-SLC pair would need \
                                                          the granule transferred to read a pixel")]
    path = joinpath(s.run, "reference.tif")
    isfile(path) || return [StageResult("5.2 radar mosaic", "reference.tif", "set", true, 0,
                                        "skipped: no $path")]
    rp, sp = _burst_products(s.case, s.run)
    sws = burst_swaths(s.case)
    out = [_mosaic_stage("5.2 reference mosaic", path, radar_mosaic(rp, sws))]
    sec = joinpath(s.run, "secondary.tif")
    isfile(sec) || return out
    dem = joinpath(s.run, "dem.tif")
    isfile(dem) || return push!(out, StageResult("5.2 secondary mosaic", "secondary.tif", "set", true,
                                                0, "skipped: no $dem, and the coregistration solves \
                                                 for terrain height"))
    # The secondary is gated more loosely than the reference and for a stated reason: it travels through a
    # full resample — an eight-tap windowed sinc over a deramped burst — where the reference is a copy.
    push!(out, _mosaic_stage("5.2 secondary mosaic", sec,
                             secondary_mosaic(rp, sp, sws, dem_sampler(dem)); max_median = 3e-4))
    return out
end

# Both parsed SAFEs, reference first, which is the earlier of the two `burst2safe` wrote.
function _burst_products(c::GoldenCase, run::AbstractString)
    early, late = _burst_safes(run)
    prod(p) = Sentinel1Product(p; orbit = s1_orbit(run, basename(p)),
                               polarization = lowercase(s1_polarization(basename(p))),
                               swaths = burst_swaths(c))
    return prod(early), prod(late)
end
_reference_product(c::GoldenCase, run::AbstractString) = first(_burst_products(c, run))

# `got` against a reference raster, read in row strips so neither is held whole. The filled *set* is the
# gate; the values are reported relative to the amplitudes they sit on.
function _mosaic_stage(name, ref_path, got; max_median = 1e-4)
    bd = ArchGDAL.getband(ArchGDAL.read(ref_path), 1)
    if size(got) != (ArchGDAL.height(bd), ArchGDAL.width(bd))
        return StageResult(name, basename(ref_path), "set", false, 0,
                           "shape $(size(got)) against reference \
                            $((ArchGDAL.height(bd), ArchGDAL.width(bd)))")
    end
    nr, nc = size(got)
    only_j = 0; only_r = 0; both = 0
    d = Float64[]
    scale = 0.0
    only_r_peak = 0.0
    only_j_peak = 0.0
    for r0 in 1:4096:nr
        rows = r0:min(r0 + 4095, nr)
        want = permutedims(ArchGDAL.read(bd, rows, 1:nc))
        g = @view got[rows, :]
        for i in eachindex(g, want)
            a, b = Float64(g[i]), Float64(want[i])
            if a != 0 && b == 0
                only_j += 1
                only_j_peak = max(only_j_peak, a)
            elseif a == 0 && b != 0
                only_r += 1
                only_r_peak = max(only_r_peak, b)
            elseif a != 0
                both += 1
                scale = max(scale, b)
                length(d) < 4_000_000 && push!(d, abs(a - b))
            end
        end
    end
    n = nr * nc
    med = isempty(d) ? 0.0 : median(d)
    p99 = isempty(d) ? 0.0 : quantile(d, 0.99)
    # **Three conditions, and the third is a magnitude rather than a count.** We may never invent a filled
    # pixel — that would be an index error. The reference may fill one we do not, because COMPASS's
    # resample tapers into samples whose raw value is exactly zero, but only where its value there is
    # negligible: measured on `S1C_IW_SLC__1SSV_20250416`, 27,860 such pixels of 363,775,172 with a
    # *maximum* of 0.106 against amplitudes near 200. Bounding the peak rather than the count says that
    # the disagreement is the taper and not a misplaced burst, which a count cannot.
    passed = only_j_peak <= 1e-3 * max(scale, 1) && only_r_peak <= 1e-3 * max(scale, 1) &&
             med <= max_median * max(scale, 1)
    return StageResult(name, basename(ref_path),
                       @sprintf("extras on either side <= 1e-3, median <= %.0e of scale",
                                max_median), passed, n,
                       @sprintf("%d of %d filled on both; %d only ours (peak %.4g), %d only the \
                                 reference (peak %.4g); median |d| %.5g, p99 %.5g against a peak \
                                 amplitude of %.4g",
                                both, n, only_j, only_j_peak, only_r, only_r_peak, med, p99, scale))
end

"""
    rung_filter(s::Setup) -> Vector{StageResult}

Rung 5.3 — `apply_landsat_filtering` on Julia's own read of the granule, on the native grid.

The pass `process.py` runs *before* geogrid, on the full scene rather than the overlap
([`native_filter`](@ref)). Only the seven Landsat 4/5/7 cases have one; every other pair is filtered
inside the correlator and rung 5.4 is where its filter is tested.

Two filters and two different gates, because the two differ from the reference for different reasons:

  * **L7 and L8 take `wallis_fill`**, whose gaps are filled from a random draw. The reference draws from
    NumPy's unseeded global generator and so cannot reproduce *itself*, which makes the filled values
    unmatched by construction. So the gate is the **zero mask exactly** — which positions were filled is
    a decision, not a draw — and the unfilled values on tolerance.
  * **L4 and L5 take Wallis followed by the band-reject.** The reference's Wallis variance is
    `E[x²] − E[x]²`, which AutoRIFT.jl deliberately does not reproduce: about-the-mean is ~360,000×
    more accurate against an exact `Float64` truth. So this is a tolerance gate too, and a loose one.

`Destripe`'s scan angles come from the reference's own log, which is this ladder's discipline for an
input a rung is not testing — gate `5.orbit` is where the orbit-derived angles are measured against
those, and `tools/golden/README.md` records why they are not yet the default.
"""
function rung_filter(s::Setup)
    native = native_filter(s.case, first(s.case.reference))
    native === nothing && return [StageResult("5.3 native filter", "filtered/", "tolerance", true, 0,
                                              "skipped: this pair is filtered inside the correlator, \
                                               which rung 5.4 covers")]
    log = joinpath(s.run, "capture.log")
    angles = native === :fft ? reference_scan_angles(log) : nothing
    fired = native === :fft ? reference_banding(log) : nothing
    out = StageResult[]
    for (i, name) in enumerate((first(s.case.reference), first(s.case.secondary)))
        img_path, zero_path = native_filtered_paths(s.case, s.run, name)
        scene = ArchGDAL.getband(ArchGDAL.read(scene_path_for(s.case, name)), 1)
        nd = ArchGDAL.getnodatavalue(scene)
        raw = permutedims(ArchGDAL.read(scene))
        # `prepare_array_for_filtering`: the nodata value becomes zero and the array becomes Float32,
        # and the valid domain is remembered rather than inferred again downstream.
        valid = nd === nothing ? trues(size(raw)) : raw .!= eltype(raw)(nd)
        img = Float32.(raw)
        img[.!valid] .= 0.0f0
        truth = permutedims(ArchGDAL.read(ArchGDAL.getband(ArchGDAL.read(img_path), 1)))
        if native === :wallis_fill
            got, ok = AutoRIFT.wallis_gapfill(img, valid, 5, 0.25)
            push!(out, _zero_mask_stage("5.3 $name zero mask", zero_path, ok))
            # **Only where both sides call the pixel valid, and the median carries the verdict.** The
            # reference fills its gaps from NumPy's unseeded global generator, so a filled value cannot be
            # matched — it cannot even be reproduced by the reference itself. Which positions were filled
            # is not written out, so the fills stay in the population and appear as the tail; the median
            # is over the overwhelming majority that were not filled.
            refok = _zero_mask_of(zero_path)
            st = _filter_stage("5.3 $name wallis_fill", img_path, got, truth, ok .& .!refok)
            # **Reported, not gated, and for two reasons that are both already established.** The median is
            # the Wallis variance choice, isolated on the `fft` branch at 0.113 where nothing else
            # intervenes; the tail is the fills themselves, drawn from NumPy's unseeded global generator, so
            # no run reproduces another including the reference's own. Separating the two here would need
            # the gap fill's distance transforms reimplemented in the harness, which the `fft` branch's
            # isolation already makes unnecessary. The zero mask above is the part that *is* a decision, and
            # it is exact.
            push!(out, StageResult("5.3 $name wallis_fill", basename(img_path), "reported", true, st.n,
                                   "unmatchable by construction (unseeded fill RNG) on top of the \
                                    deliberate Wallis variance: " * st.detail))
        else
            a = angles[i]
            m = AutoRIFT.Destripe(; along_track = a.along, cross_track = a.cross)
            wj, wmask = AutoRIFT.preprocess(img, valid, AutoRIFT.Wallis(5, 0.0))
            wj[.!valid] .= 0.0f0
            wr = _ref_wallis(img, 5)
            wr[.!valid] .= 0.0f0
            keep = valid .& wmask
            best = nothing
            parts = String[]
            for (label, w) in (("ours", wj), ("reference variance", wr))
                d, _ = AutoRIFT.preprocess(w, valid, m)
                d[.!valid] .= 0.0f0
                # **The band-reject's fire-or-decline is a binary branch, read rather than reported.** A
                # decline returns the clamped input, so `d == _clamped(w)` is the branch that was taken.
                cl = _clamped(w, valid)
                ours = !all(j -> d[j] == cl[j], eachindex(d))
                st = _filter_stage("5.3 $name $label", img_path, d, truth, keep)
                push!(parts, @sprintf("%s: reject %s (reference %s), %s", label,
                                      ours ? "fired" : "declined", fired[i] ? "fired" : "declined",
                                      st.detail))
                agree = ours == fired[i]
                score = (agree, agree && st.passed)
                best = best === nothing || score > best[1] ? (score, st) : best
            end
            push!(out, StageResult("5.3 $name wallis+destripe", basename(img_path),
                                   "reject agrees & median<=1e-3 p99.9<=5e-2 of scale",
                                   last(best).passed && first(best)[1], last(best).n,
                                   join(parts, " | ")))
        end
    end
    return out
end

"""
    _ref_wallis(img, width) -> Matrix{Float32}

The reference's own Wallis filter, written out: `(x − mean) / std` with the variance as
`E[x²] − E[x]²`.

A harness-side reproduction of a deviation the package declines to make, the same role
[`_zeropad_frame!`](@ref) plays for the high-pass border. `AutoRIFT.wallis` computes the variance about
the measured mean, which is ~360,000x more accurate against an exact `Float64` truth — the reference's
difference-of-large-numbers form cancels catastrophically in a bright, low-contrast window. Keeping both
here is what separates "the chain is right" from "the variance formula is deliberately different", which
one number cannot.

Both box means use `BORDER_REFLECT`, which is `cv2.filter2D`'s setting here and *not* the zero-padding
the high-pass uses. The standard deviation is corrected to the sample form by `sqrt(n / (n - 1))` and
snapped to `NaN` where it is ~0, both as `_preprocess_filt_std` and `_wallis_filter` do
(`autoRIFT.py:45-63`).
"""
function _ref_wallis(img::AbstractMatrix, width::Integer)
    nr, nc = size(img)
    h = Int(width) ÷ 2
    w2 = Float32(width * width)
    # Reflection indices hoisted: the inner loop runs 25 times per pixel over ~70 million pixels, so
    # recomputing a branch per neighbour costs more than the filter.
    refl(i, n) = i < 1 ? 2 - i : (i > n ? 2n - i : i)
    ri = [refl(i, nr) for i in (1 - h):(nr + h)]
    ci = [refl(j, nc) for j in (1 - h):(nc + h)]
    out = Matrix{Float32}(undef, nr, nc)
    for j in 1:nc, i in 1:nr
        s1 = 0.0f0
        s2 = 0.0f0
        for dj in (-h):h
            b = ci[j + dj + h]
            for di in (-h):h
                v = Float32(img[ri[i + di + h], b])
                s1 += v
                s2 += v * v
            end
        end
        m1 = s1 / w2
        sd = sqrt(s2 / w2 - m1 * m1) * sqrt(w2 / (w2 - 1))
        out[i, j] = abs(sd) <= 1.0f-8 ? NaN32 : (Float32(img[i, j]) - m1) / sd
    end
    return out
end

# `_fft_filter`'s own preamble, which applies whether or not the reject fires: clamp to +/-3 and send
# NaN to zero (`autoRIFT.py:198-201`). A declined reject returns exactly this.
function _clamped(w, valid)
    y = similar(w)
    for i in eachindex(w, valid, y)
        v = w[i]
        y[i] = !valid[i] ? 0.0f0 : (isnan(v) ? 0.0f0 : clamp(v, -3.0f0, 3.0f0))
    end
    return y
end

# The scene `name` names, on whichever route reaches it. `scene_path` resolves by side rather than by
# name, and rung 5.3 walks the pair in the job's order because that is the order the filter logs in.
function scene_path_for(c::GoldenCase, name::AbstractString)
    early, _ = acquisition_order(c)
    return scene_path(c, name == early ? :reference : :secondary)
end

# The reference's own `_zeroMask` raster: `~valid_domain`, the outer frame the filter zeroes
# (`autoRIFT.py:78-79`). `nothing` where the filter writes none, which is the L4/L5 branch.
_zero_mask_of(zero_path) = zero_path === nothing ? nothing :
    permutedims(ArchGDAL.read(ArchGDAL.getband(ArchGDAL.read(zero_path), 1))) .!= 0

"""
    _zero_mask_stage(name, zero_path, ok) -> StageResult

AutoRIFT.jl's validity against the reference's zeroed frame, as a coverage comparison.

**Our mask is a superset of theirs by construction, so this is not an equality.** The reference's
`zero_mask` is `~valid_domain` alone; `wallis_gapfill` additionally marks a pixel invalid where its local
mean or standard deviation came out non-finite, which the reference leaves in the field as a `NaN` that
`_fft_filter`'s successor sends to zero. So the gate is that every pixel the reference zeroed is one we
also decline — an inclusion — and the extra count is reported rather than bounded, since each one is a
pixel we refuse to invent a value for.
"""
function _zero_mask_stage(name, zero_path, ok)
    ref = _zero_mask_of(zero_path)
    ref === nothing && return StageResult(name, "zeroMask", "inclusion", true, 0,
                                          "skipped: this filter writes no zero mask")
    size(ok) == size(ref) || return StageResult(name, basename(zero_path), "inclusion", false, 0,
                                               "shape $(size(ok)) against reference $(size(ref))")
    missed = 0
    extra = 0
    for i in eachindex(ok, ref)
        if ref[i] && ok[i]
            missed += 1
        elseif !ref[i] && !ok[i]
            extra += 1
        end
    end
    n = length(ref)
    return StageResult(name, basename(zero_path), "reference's zeroed set is a subset of ours",
                       missed == 0, n,
                       @sprintf("%d of %d the reference zeroed and we did not; %d (%.4f%%) we decline \
                                 and it kept, each a non-finite local statistic", missed, n, extra,
                                100extra / n))
end

# A filtered float field against the reference's, over the positions `keep` selects. Gated on the
# median and the 99.9th percentile rather than on a maximum: the Wallis variance difference is a
# deliberate accuracy improvement, so a few pixels in a low-contrast window legitimately diverge, and a
# maximum would gate on the worst-conditioned pixel in a hundred million.
function _filter_stage(name, ref_path, jl, ref, keep; max_median = 1e-3)
    if size(jl) != size(ref)
        return StageResult(name, basename(ref_path), "tolerance", false, 0,
                           "shape $(size(jl)) against reference $(size(ref))")
    end
    d = Float64[]
    for i in eachindex(jl, ref, keep)
        keep[i] && isfinite(jl[i]) && isfinite(ref[i]) || continue
        push!(d, abs(Float64(jl[i]) - Float64(ref[i])))
    end
    isempty(d) && return StageResult(name, basename(ref_path), "tolerance", false, 0,
                                     "no position compared on both sides")
    med = median(d)
    p999 = quantile(d, 0.999)
    scale = quantile(filter(isfinite, abs.(vec(Float64.(ref)))), 0.99)
    passed = med <= max_median * max(scale, 1) && p999 <= 5e-2 * max(scale, 1)
    return StageResult(name, basename(ref_path),
                       @sprintf("median<=%.0e p99.9<=5e-2 of scale", max_median), passed,
                       length(d),
                       @sprintf("%d compared; median %.4g, p99.9 %.4g, max %.4g against a p99 \
                                 magnitude of %.4g", length(d), med, p999, maximum(d), scale))
end

"""
    rung_bytes(s::Setup) -> Vector{StageResult}

Rung 5.4 — the bytes the correlator is handed, from Julia's own read of the granule.

The last boundary before the correlation, and the first one on the imagery side that this ladder
reaches: rung 5.7 is fed `capture/in_I1`, so until here nothing upstream of the reference's own
quantized pair had ever run in Julia on a golden case. The chain is read the granule, crop to the
overlap `coregister` computed, apply the filter `runAutorift` applies ([`correlator_filter`](@ref)),
and quantize with [`AutoRIFT.bytescale`](@ref).

**Two border rules are reported, and the difference between them is the whole story.**
`AutoRIFT.highpass` excludes out-of-image neighbours from the local mean where the reference
zero-pads (`cv2.filter2D(..., BORDER_CONSTANT)`), a deliberate deviation its docstring records. That
disagreement is confined to a `width ÷ 2` frame — 0.07% of a 10980² scene — but `uniform_data_type`
takes its mean and standard deviation over the **whole array**, so the frame moves the quantization of
every pixel in the image. Measured on the golden Sentinel-2 pair, the frame alone is the difference
between **65.24%** of bytes exact and **99.9984%**. Reporting both separates "the border rule differs"
from "something else differs", which one number cannot.

The interior kernel is bitwise identical, checked directly: over that scene the largest difference
between `AutoRIFT.highpass` and the reference's own form, away from the frame, is exactly 0.
"""
function rung_bytes(s::Setup)
    call = joinpath(s.run, "capture", "call1.json")
    isfile(call) || return [StageResult("5.4 correlator bytes", "capture/in_I1", "exact", true, 0,
                                       "skipped: no capture at $call")]
    # The radar scenes in a run directory are ISCE3's own products rather than anything this ladder
    # built, so quantizing them would compare the reference against itself. Rung 5.2's radar half —
    # the coregistration — is what has to exist first.
    # **A NISAR L2 GSLC needs neither rung 5.2 nor a filter this rung can reproduce.** It is geocoded, so
    # the correlator's input is `convert_slc_to_uint8_amplitude`'s two rasters on one map grid — a read and
    # a crop, which is what this rung does. What stops it is the *filter*: `loadProduct` hard-casts NISAR
    # to `uint8`, and `cv2.filter2D(..., ddepth=-1)` returns the input's depth, so the reference's
    # high-pass output is `uint8` too and its entire negative half saturates to zero. Measured in the
    # reference's own bytes: **~30% of `in_I1` sits at exactly 128**, which is where a field value of zero
    # quantizes to, against a smooth ~1.8% per level elsewhere. Reproducing that means truncating the
    # filter to `UInt8` before quantizing; see `tools/golden/README.md`, since discarding half the
    # high-pass response is a defect rather than a convention.
    s.case.platform == "NISAR-L2" &&
        return [StageResult("5.4 correlator bytes", "capture/in_I1", "exact", true, 0,
                            "skipped: the reference's high-pass output is truncated to `UInt8` here, \
                             clipping every negative response to zero — ~30% of `in_I1` is the single \
                             value 128. Untruncated this rung reaches 57.2% exact")]
    if s.case.platform == "S1-BURST"
        # **The reference side only.** `radar_mosaic` builds what `merge_swaths` writes as
        # `reference.tif`, which needs no coregistration — COMPASS writes the reference burst on its own
        # grid. `secondary.tif` is the coregistered one, and that half of rung 5.2 does not exist yet, so
        # `in_I2` is declined rather than compared against a mosaic built from the wrong grid.
        k = read_capture(s.case; n = parse(Int, basename(s.run)))
        rp, sp = _burst_products(s.case, s.run)
        sws = burst_swaths(s.case)
        m = correlator_filter(s.case)
        out = StageResult[_bytes_stage("in_I1", radar_mosaic(rp, sws), k.arrays["in_I1"], m)]
        dem = joinpath(s.run, "dem.tif")
        if isfile(dem)
            push!(out, _bytes_stage("in_I2", secondary_mosaic(rp, sp, sws, dem_sampler(dem)),
                                    k.arrays["in_I2"], m))
        else
            push!(out, StageResult("5.4 in_I2", "capture/in_I2", "exact", true, 0,
                                   "skipped: no $dem, and the coregistration solves for terrain height"))
        end
        return out
    end
    (startswith(s.case.platform, "S1") || s.case.platform == "NISAR-L1") &&
        return [StageResult("5.4 correlator bytes", "capture/in_I1", "exact", true, 0,
                            "skipped: the pair reaching the correlator is a radar-grid mosaic, so this \
                             needs rung 5.2's coregistration rather than a read and a filter")]
    # **A Landsat 4/5/7 pair is filtered twice, on two different grids**, so the granule is the wrong
    # input for it. `process.py` filters the *native* scene and, for a cross-projection pair, warps the
    # result; `runAutorift` then filters the *cropped* overlap again ([`correlator_filter`](@ref)). Rung
    # 5.3 is what tests that first pass, so this reads the reference's own output of it —
    # [`filtered_path`](@ref) — which keeps a filter difference from being reported here as a
    # quantization one.
    k = read_capture(s.case; n = parse(Int, basename(s.run)))
    m = correlator_filter(s.case)
    # **A `wallis_fill` pair cannot match here, by construction rather than by defect.** The filter fills
    # its gaps from NumPy's *unseeded* global generator, so the reference cannot reproduce itself either —
    # and those values are not confined to the gaps, because `uniform_data_type` takes its mean and
    # standard deviation over the whole array. So a draw nobody can reproduce sets the quantization window
    # for every pixel. Skipped with the measurement rather than reded against a bound it can never meet.
    if m isa AutoRIFT.WallisGapfill
        return [StageResult("5.4 correlator bytes", "capture/in_I1", "exact", true, 0,
                            "skipped: this pair is gap-filled from an unseeded RNG inside the \
                             correlator, and `uniform_data_type` takes its statistics over the whole \
                             array, so the unreproducible draw rescales every pixel — measured at \
                             1.85% exact on `LE07_L1TP_063018`")]
    end
    rpath = something(filtered_path(s.case, s.run, first(s.case.reference)), s.reference_path)
    spath = something(filtered_path(s.case, s.run, first(s.case.secondary)), s.secondary_path)
    # `filtered_path` resolves by name and the offsets are in acquisition order, so the two are paired
    # back up here rather than assumed to line up.
    early, _ = acquisition_order(s.case)
    first(s.case.reference) == early || ((rpath, spath) = (spath, rpath))
    out = StageResult[]
    for (nm, path, off, truth) in (("in_I1", rpath, s.pair.reference_offset, k.arrays["in_I1"]),
                                   ("in_I2", spath, s.pair.secondary_offset, k.arrays["in_I2"]))
        win = _cropped_scene(path, off, size(truth))
        push!(out, _bytes_stage(nm, win, truth, m))
    end
    return out
end

# The overlap `coregister` computed, read out of the native scene. ArchGDAL hands back `(x, y)`; every
# captured array is the reference's `(row, col)`, so the window transposes once here rather than at
# each comparison.
function _cropped_scene(path::AbstractString, off::NTuple{2,Int}, want::Tuple{Int,Int})
    full = ArchGDAL.read(ArchGDAL.getband(ArchGDAL.read(path), 1))
    nr, nc = want
    ox, oy = off
    size(full, 1) >= ox + nc && size(full, 2) >= oy + nr || throw(DimensionMismatch(
        "the overlap at offset $off does not fit in a $(size(full)) scene; the crop and the \
         geotransform disagree"))
    return permutedims(full[(ox + 1):(ox + nc), (oy + 1):(oy + nr)])
end

# The reference's zero-padded value on the frame the two border rules disagree on, spliced into an
# otherwise-Julia field. Only the frame: the interior is bitwise identical, so this isolates the
# deviation instead of reimplementing the filter.
function _zeropad_frame!(out, img, width::Integer)
    h = Int(width) ÷ 2
    n, m = size(img)
    for j in 1:m, i in 1:n
        (i <= h || i > n - h || j <= h || j > m - h) || continue
        acc = 0.0f0
        for dj in (-h):h, di in (-h):h
            a, b = i + di, j + dj
            (1 <= a <= n && 1 <= b <= m) || continue
            acc += Float32(img[a, b])
        end
        out[i, j] = Float32(img[i, j]) - acc / Float32(width * width)
    end
    return out
end

function _bytes_stage(nm, win, truth, m)
    mask = trues(size(win))
    if m === nothing
        field = Float32.(win)
        variants = (("as read", field),)
    else
        f, _ = AutoRIFT.preprocess(Float32.(win), mask, m)
        # The frame patch applies to the plain high-pass alone. `WallisGapfill` divides by a local
        # standard deviation and fills its gaps from a random draw, so its frame is not one term the
        # reference's padding rule maps onto.
        variants = m isa AutoRIFT.Highpass ?
                   (("ours", f), ("reference border", _zeropad_frame!(copy(f), win, m.width))) :
                   (("ours", f),)
    end
    best = nothing
    parts = String[]
    for (label, field) in variants
        d = Int.(AutoRIFT.bytescale(field, mask)) .- Int.(truth)
        ad = abs.(d)
        eq = count(iszero, d) / length(d)
        w1 = count(<=(1), ad) / length(d)
        push!(parts, @sprintf("%s exact %.4f%% within one level %.4f%% max %d", label, 100eq, 100w1,
                              maximum(ad)))
        best = best === nothing ? (eq, w1) : (max(best[1], eq), max(best[2], w1))
    end
    # Gated on the better border rule, since the worse one is a recorded deviation rather than a
    # disagreement this rung is asking about. One level everywhere is the floor `bytescale`'s own
    # docstring sets — `np.mean`'s summation order moves whatever sits nearest a rounding boundary.
    gate = "exact>=99% within one level>=99.99%"
    passed = best[1] >= 0.99 && best[2] >= 0.9999
    return StageResult("5.4 $nm", "capture/$nm", gate, passed, length(truth), join(parts, "; "))
end

"""
    _by_level(axis, jl, ref, chip) -> StageResult

The residual's bias split by the chip size each point resolved at. Reported, never gating.

This is the cut that localizes an endpoint residual, and it is here rather than in a scratch script
because it has now answered two reds that the whole-field statistics could not.

**The base level is the one whose values are each side's own measurement.** Above it both sides place a
coarse node at the cell's *mean* coordinate and read the answer back from the cell's geometric *centre*
(`dev/CORRECTNESS.md` items 2 and 3), and those are the same point only for a cell with no nodata in it.
So a decimated level's residual grows with how often its cells straddle the imagery's edge, which is a
property of the scene's coverage rather than of the correlator — and splitting on the level says at once
whether a bias is the kernel's or that position gap's.

Measured across four cases, the whole-field `dx` bias tracks the share of points that reached the base
level and nothing else: 62% base gives −0.006, 35% gives −0.039, 0% gives −0.089, while every one of the
four has a base-level bias under 0.0004 and a base-level median of exactly 0.
"""
function _by_level(axis::Symbol, jl, ref, chip)
    parts = String[]
    for cs in sort(unique(vec(chip)))
        cs == 0 && continue
        d = [Float64(jl[i]) - Float64(ref[i])
             for i in eachindex(jl, ref, chip)
             if chip[i] == cs && !isnan(jl[i]) && !isnan(ref[i])]
        isempty(d) && continue
        push!(parts, @sprintf("chip %d n %d bias %+.5f median %.4g", cs, length(d), mean(d),
                              median(abs.(d))))
    end
    return StageResult("5.7 $axis by level", "out_$(uppercase(String(axis)))", "reported", true, 0,
                       isempty(parts) ? "no point measured on both sides" : join(parts, "; "))
end

# `pts` truncated to `nr x nc`, which is what `autoRIFT.py:809-819` does to every per-point array before
# correlating. AutoRIFT.jl keeps those points — `tools/ab/README.md` measures the truncation costing the
# reference 1.6% of its grid — so this is applied to make the two comparable rather than because either
# implementation should.
function _chop_to(pts, nr::Integer, nc::Integer)
    c(A) = A[1:nr, 1:nc]
    # `chip_size_x`/`chip_size_y` are lazy uniform arrays carrying the grid's own axes, so they need
    # chopping too or `PointSet` rejects the mismatch — which it does, by name, rather than broadcasting
    # a stale shape into the correlation.
    return AutoRIFT.rebuild(pts; x = c(pts.x), y = c(pts.y),
                            radius_x = c(pts.radius_x), radius_y = c(pts.radius_y),
                            dx_prior = c(pts.dx_prior), dy_prior = c(pts.dy_prior),
                            chip_size_x = c(pts.chip_size_x), chip_size_y = c(pts.chip_size_y),
                            chip_size_min_x = c(pts.chip_size_min_x),
                            chip_size_max_x = c(pts.chip_size_max_x))
end

"""
    _mask_from_capture(pts, k::Capture) -> PointSet

`pts` with its search radius zeroed wherever the reference declined the point.

The driver zeroes `xGrid`, `Dx0`, `SearchLimit*` and both chip bounds wherever `noDataMask` is set
(`testautoRIFT.py:394-403`), and `noDataMask` is the *imagery's* zero mask sampled at each grid point
(`:344-349`) — the filtered scene's no-data for Landsat 4/5/7, and simply the out-of-image sentinel for
an `hps` pair, which `pointset` already handles.

Fed rather than derived, and that is the ladder's discipline rather than a shortcut: rung 5.7 tests the
geogrid handoff composed with the correlator, and the mask is an *imagery* input that rung 5.3 is what
reproduces. Deriving it here would fold a filter difference into a grid measurement.

It matters most where the imagery is gappy. On the cross-path Landsat 7 pair it moves the comparison from
115,099 shared points with 621,986 measured only by AutoRIFT.jl to a set the two can be read against;
on an `hps` pair it changes almost nothing, since there the mask is the sentinel.
"""
function _mask_from_capture(pts, k)
    declined = k.arrays["in_SearchLimitX"] .== 0
    size(declined) == size(pts.radius_x) || throw(DimensionMismatch(
        "the reference declined-point mask is $(size(declined)) but the grid is " *
        "$(size(pts.radius_x)); the chop should already have made them agree"))
    keep = .!declined
    return AutoRIFT.rebuild(pts; radius_x = pts.radius_x .* keep, radius_y = pts.radius_y .* keep)
end

# The better of two signed scorings, read off the `exact` fraction the detail line carries. Comparing on
# `exact` rather than on the gate's verdict, since a pair whose base level was skipped fails both signs and
# the sign is still determined.
function _better_endpoint(a, b)
    ea, eb = _exact_fraction(last(a).detail), _exact_fraction(last(b).detail)
    return eb > ea ? b : a
end

# `exact` where the quantized gate reported one, and the within-0.1-px fraction otherwise: both rise with
# agreement, and the sign only has to be ranked rather than measured.
function _exact_fraction(detail::AbstractString)
    m = match(r"exact ([\d.]+)%", detail)
    m === nothing || return parse(Float64, m.captures[1])
    m = match(r"within 0\.1 px ([\d.]+)%", detail)
    return m === nothing ? -1.0 : parse(Float64, m.captures[1])
end

# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------

"""
    e2e(c::GoldenCase; n = nothing, stop_on_red = true, proj_only = false) -> Vector{StageResult}

Run the ladder on `c`, stopping at the first red rung unless told otherwise.

Stopping is the default because a rung fed a correct input tells you about itself, while the rungs
after a red one are being asked a question whose premise has already failed — a geogrid compared
against a grid that does not match is comparing two different problems.
"""
function e2e(c::GoldenCase; n::Union{Integer,Nothing} = nothing, stop_on_red::Bool = true,
             proj_only::Bool = false)
    s = setup(c; n, proj_only)
    # The interval off the pair rather than recomputed: the two paths derive it by opposite conventions —
    # whole calendar days for optical, full precision for radar — and the pair already holds the one that
    # was used.
    @info "e2e setup" product=c.product region=s.info.name epsg=s.info.epsg window=size(s.window) dt_days=s.pair.dt / 86400

    out = StageResult[]
    for rung in (rung_grid, rung_window, rung_coregister, rung_filter, rung_bytes,
                 rung_geogrid, rung_params,
                 rung_endpoint)
        rs = rung(s)
        append!(out, rs)
        stop_on_red && !all(r -> r.passed, rs) && break
    end
    return out
end

function main(args)
    isempty(args) && error("usage: e2e.jl <product> [--run N] [--all] [--proj-only]")
    c = only(cases(args[1]))
    n = nothing
    i = findfirst(==("--run"), args)
    i === nothing || (n = parse(Int, args[i + 1]))
    ok = report(e2e(c; n, stop_on_red = !("--all" in args),
                    proj_only = "--proj-only" in args))
    ok || exit(1)
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
