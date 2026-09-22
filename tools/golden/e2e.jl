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
include("stages.jl")

using AutoRIFT: chip_sizes, subpixel_at

using ArchGDAL, ImagePairGeometry, Printf, Statistics
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

The relative denominator is `max(|reference|, 1)`, so a value below one is judged absolutely there too.
Without that a band passing through zero reports an unbounded ratio for a difference of no consequence.
"""
function relative_stage(name, ref_name, jl, ref; atol = 1e-3)
    if size(jl) != size(ref)
        return StageResult(name, ref_name, "abs<=$atol", false, 0,
                           "shape $(size(jl)) against reference $(size(ref))")
    end
    worst = 0.0
    worst_abs = 0.0
    nz = 0
    for i in eachindex(jl, ref)
        a, b = Float64(jl[i]), Float64(ref[i])
        a == b && continue
        nz += 1
        worst = max(worst, abs(a - b) / max(abs(b), 1.0))
        worst_abs = max(worst_abs, abs(a - b))
    end
    n = length(ref)
    return StageResult(name, ref_name, "abs<=$atol", worst_abs <= atol, n,
                       @sprintf("%d of %d differ (%.2f%%), max absolute %.4g, max relative %.4g",
                                nz, n, 100 * nz / n, worst_abs, worst))
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
function rounded_stage(name, ref_name, jl, ref; budget::Int = 2)
    if size(jl) != size(ref)
        return StageResult(name, ref_name, "exact", false, 0,
                           "shape $(size(jl)) against reference $(size(ref))")
    end
    bad = 0
    worst = 0
    first_bad = nothing
    for i in eachindex(IndexCartesian(), jl)
        a, b = Int(jl[i]), Int(ref[i])
        a == b && continue
        bad += 1
        worst = max(worst, abs(a - b))
        first_bad === nothing && (first_bad = (Tuple(i), a, b))
    end
    n = length(ref)
    gate = "exact, <=$budget by 1"
    passed = bad == 0 || (bad <= budget && worst <= 1)
    detail = if bad == 0
        "all $n equal"
    else
        pos, a, b = first_bad
        @sprintf("%d of %d differ by at most %d, first at %s: julia %d, reference %d",
                 bad, n, worst, pos, a, b)
    end
    return StageResult(name, ref_name, gate, passed, n, detail)
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

    # Reprojection first, because the footprints geogrid intersects — and the pixel grid its indices
    # count in — are the warped ones. A pair already in one projection comes back untouched.
    rpath, spath = aligned_scenes(c)
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
            push!(out, eltype(mine) <: Integer ? rounded_stage(name, file, mine, Int32.(ref)) :
                       relative_stage(name, file, mine, ref))
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
    @info "e2e setup" product=c.product region=s.info.name epsg=s.info.epsg window=size(s.window) dt_days=geogrid_seconds(c) / 86400

    out = StageResult[]
    for rung in (rung_grid, rung_geogrid, rung_params)
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
