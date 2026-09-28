# From the raw granule to `dx`/`dy`: the whole Julia chain on a golden case, timed and traced.
#
#   julia --project=tools/golden -t 12,1 tools/golden/e2e_run.jl S2B_MSIL1C_20200612
#   julia --project=tools/golden -t 12,1 tools/golden/e2e_run.jl --all --tsv e2e_julia.tsv
#
# **Every other harness here starts from a capture.** `e2e.jl` compares one stage at a time against the
# reference's own artifacts and then correlates the capture's imagery; `e2e_table.jl` joins two sweeps
# that both read captures. So the figure in `README.md` is a correlator comparison: the same bytes on
# both sides, neither arm deriving the grid. Nothing measured the chain that turns a granule into
# `dx`/`dy`.
#
# This does, on the pieces the ladder validates separately:
#
#   geometry    the scenes resolved, warped to one projection if they are not, the footprints
#               coregistered, the parameter region looked up and the geogrid solved — `setup`, which is
#               rungs 5.0 through 5.2 and 5.5 as one call
#   grid        the geogrid's own point set and parameters — `AutoRIFT.pointset`/`params` of it
#   imagery     the correlator's input built from the granule: the overlap read out of each optical
#               scene with the filter `process.py` applies to a native scene, or the reference mosaic
#               and a lazily resampled secondary for a radar burst pair
#   correlate   `autorift` at the block `block_size_for` picks
#
# Peak is the whole run's, sampled: the point of measuring a chain rather than its last stage is that
# the earlier stages are what a peak is usually made of.
#
# **The scenes are staged locally before the clock starts**, so what is timed is the computation rather
# than 708 MB of requester-pays egress. `--stream` leaves them where `scene_path` resolves them, which is
# what the reference container does — the two differ by the `imagery` stage's I/O and nothing else.
#
# **Argument order.** `arImgDisp_*` cuts its chip from its second argument and the driver calls it
# `arImgDisp(I2, I1)`, so `I1` is AutoRIFT.jl's *secondary* slot; `e2e.jl`'s rung 5.4 pins `in_I1` to the
# scene at the reference offset, so `I1` is the reference *acquisition*. The reference acquisition
# therefore goes in the secondary slot, which is what rung 5.7 does and what this repeats.

include("e2e.jl")
include(joinpath(dirname(@__DIR__), "ab", "memtrace.jl"))

using Printf

argvalue(flag, default) = (i = findfirst(==(flag), ARGS);
                           isnothing(i) ? default : ARGS[i + 1])

# The flags that take a value, so the value is not mistaken for a case name.
const VALUED = ("--run", "--block", "--tsv")

# The case fragments in an argument list: everything that is neither a flag nor a flag's value.
function positional(args)
    out = String[]
    skip = false
    for a in args
        if skip
            skip = false
        elseif startswith(a, "--")
            skip = a in VALUED
        else
            push!(out, a)
        end
    end
    return out
end

# ---------------------------------------------------------------------------
# What the chain reaches
# ---------------------------------------------------------------------------

"""
    stage(c::GoldenCase) -> Vector{String}

Put `c`'s scenes on local disk and have [`scene_path`](@ref) resolve to the copies.

Called before the clock starts, which is the point: the reference streams a Landsat band from a
requester-pays bucket and so does this chain, and 354 MB of egress inside a timed stage measures the
network. A radar burst pair is already local — its SAFE trees are in the run directory — so there is
nothing to stage.
"""
function stage(c::GoldenCase)
    (startswith(c.platform, "L") || c.platform == "S2") || return String[]
    out = String[]
    for which in (:reference, :secondary)
        early, late = acquisition_order(c)
        name = which === :reference ? early : late
        haskey(STAGED, name) && (push!(out, STAGED[name]); continue)
        STAGED[name] = stage_scene(scene_path(c, which))
        push!(out, STAGED[name])
    end
    return out
end

"""
    unsupported(c::GoldenCase) -> Union{Nothing,String}

Why the chain from the granule does not exist for `c`, or `nothing` when it does.

Named per platform rather than discovered by failure, so a case this cannot measure says what is
missing instead of erroring somewhere inside a stage.
"""
function unsupported(c::GoldenCase)
    c.platform == "NISAR-L1" && return "an RSLC is not TOPS and needs its own azimuth resample, which \
                                        does not exist on either side of this repository"
    c.platform == "NISAR-L2" && return "a GSLC reaches the correlator as the driver's own \
                                        `*_adjusted.tif`, so correlating those is not a path from the \
                                        granule"
    if startswith(c.platform, "S1") && c.platform != "S1-BURST"
        return "a full-SLC pair needs the three-subswath mosaic, whose width rule is unresolved, and \
                the granule is not staged"
    end
    return nothing
end

# ---------------------------------------------------------------------------
# The imagery, from the granule
# ---------------------------------------------------------------------------

"""
    native_scene(path, name, c) -> Matrix{Float32}

One optical scene, filtered as `process.py` filters it before geogrid sees it.

**The filter runs on the whole scene and the crop comes after**, which is the order `process.py` uses
and the order `Destripe` requires: its band-reject is a transform of the entire array, so filtering a
crop is a different operation rather than a cheaper one.

`:fft` is Wallis at width 5 followed by the band-reject, with the scan angles derived from the
granule's own `_ANG.txt` ephemeris — the same route gate `5.orbit` measures. `:wallis_fill` is the
gap-filling Wallis. A pair whose platforms name no native filter is read and returned.
"""
function native_scene(path::AbstractString, name::AbstractString, c::GoldenCase)
    ds = ArchGDAL.read(path)
    # ArchGDAL hands back `(x, y)`; every grid index here counts in `(row, col)`.
    img = Float32.(permutedims(ArchGDAL.read(ArchGDAL.getband(ds, 1))))
    m = native_filter(c, name)
    m === nothing && return img

    valid = img .!= 0
    if m === :wallis_fill
        f, _ = AutoRIFT.preprocess(img, valid, AutoRIFT.WallisGapfill(5, 0.25))
        return f
    end
    m === :fft || error("no route for native filter `$m` on $name")
    gt = ArchGDAL.getgeotransform(ds)
    along, cross = orbit_scan_angles(scene_ang(name, joinpath(CACHE, "angcache")), scene_epsg(ds);
                                     spacing = (gt[2], gt[6]))
    w, _ = AutoRIFT.preprocess(img, valid, AutoRIFT.Wallis(5, 0.0))
    w[.!valid] .= 0.0f0
    d, _ = AutoRIFT.preprocess(w, valid, AutoRIFT.Destripe(; along_track = along, cross_track = cross))
    d[.!valid] .= 0.0f0
    return d
end

# The overlap `coregister` found, out of a scene already on the correlation grid.
function crop_overlap(img::AbstractMatrix, off::NTuple{2,Int}, want::Tuple{Int,Int})
    nr, nc = want
    ox, oy = off
    size(img, 1) >= oy + nr && size(img, 2) >= ox + nc || throw(DimensionMismatch(
        "the overlap at offset $off does not fit in a $(size(img)) scene; the crop and the " *
        "geotransform disagree"))
    return img[(oy + 1):(oy + nr), (ox + 1):(ox + nc)]
end

"""
    e2e_imagery(s::Setup) -> (reference, secondary)

The correlator's two images, built from the granule, in acquisition order.

A radar burst pair returns the reference acquisition's mosaic and a [`ResampledMosaic`](@ref) — lazy, so
the secondary is resampled a block at a time by the correlator rather than materialized or written to
disk. An optical pair returns the two overlaps.
"""
function e2e_imagery(s::Setup)
    c = s.case
    if c.platform == "S1-BURST"
        rp, sp = _s1_products(c, s.run)
        sws = burst_swaths(c)
        dem = joinpath(s.run, "dem.tif")
        isfile(dem) || error("no dem.tif in $(s.run); the secondary's resample solves for terrain " *
                            "height, so the pair cannot be coregistered without one")
        return (radar_mosaic(rp, sws), ResampledMosaic(rp, sp, sws, dem_sampler(dem)))
    end
    early, late = acquisition_order(c)
    want = reverse(s.pair.coordinate.size)
    ref = crop_overlap(native_scene(s.reference_path, early, c), s.pair.reference_offset, want)
    sec = crop_overlap(native_scene(s.secondary_path, late, c), s.pair.secondary_offset, want)
    return (ref, sec)
end

# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

struct E2EResult
    case::String
    platform::String
    stages::Vector{Pair{String,Float64}}
    cpu::Float64
    peak::Int
    floor::Int
    scene::Tuple{Int,Int}
    block::Tuple{Int,Int}
    npoints::Int
    measured::Int
    filter::String
end

total_seconds(r::E2EResult) = sum(last, r.stages)

"""
    run_case(c::GoldenCase; n = nothing, block = nothing, trace = true) -> E2EResult

The whole chain on one case, each stage timed and the run's peak sampled.

`block` overrides what [`AutoRIFT.block_size_for`](@ref) picks from the grid's own halo.
"""
function run_case(c::GoldenCase; n::Union{Integer,Nothing} = nothing,
                  block::Union{Integer,Nothing} = nothing, trace::Bool = true,
                  staged::Bool = true, warm::Bool = true)
    reason = unsupported(c)
    isnothing(reason) || error(reason)
    staged && stage(c)

    buf = zeros(UInt64, 64)
    hz = tick_rate()
    stages = Pair{String,Float64}[]
    out = Ref{Any}(nothing)
    shape = Ref((0, 0))
    blk = Ref((0, 0))
    np = Ref(0)
    filt = Ref("")

    work = function ()
        t = @elapsed s = setup(c)
        push!(stages, "geometry" => t)

        t = @elapsed begin
            grid = AutoRIFT.pointset(s.geometry;
                                     pixel_size = ImagePairGeometry.xsize(s.pair.coordinate))
            m = correlator_filter(c)
            p = AutoRIFT.params(s.geometry; threaded = Threads.nthreads() > 1,
                                preprocess = isnothing(m) ? :none : m)
        end
        push!(stages, "grid" => t)
        np[] = length(grid.x)
        filt[] = string(isnothing(m) ? :none : m)

        t = @elapsed (i1, i2) = e2e_imagery(s)
        push!(stages, "imagery" => t)
        shape[] = size(i1)

        # The reference acquisition takes the secondary slot; see the note at the top of this file.
        b = isnothing(block) ? AutoRIFT.block_size_for(grid, p, size(i1)) :
            (; X = Int(block), Y = Int(block))
        blk[] = (b.X, b.Y)
        t = @elapsed out[] = AutoRIFT.autorift(i2, i1, grid, p, (b.X, b.Y))
        push!(stages, "correlate" => t)
        return nothing
    end

    # A cold process spends its first pass compiling — 9.1 s of a Sentinel-2 case's `geometry` stage and
    # 0.4 s of its `grid` — so the measured pass is the second. The peak is unaffected: both passes hold
    # the same arrays, and the collection below returns the first pass's before the second allocates.
    if warm
        work()
        empty!(stages)
        GC.gc(true); GC.gc(true)
    end
    floor_bytes = last(rusage!(buf))
    cpu0 = cpu_seconds!(buf, hz)

    peak = if trace
        _, tr, _ = with_trace(; interval = 0.05) do
            work()
        end
        maximum(tr.footprint)
    else
        work()
        last(rusage!(buf))
    end
    cpu = cpu_seconds!(buf, hz) - cpu0

    return E2EResult(c.product, c.platform, stages, cpu, Int(peak), Int(floor_bytes),
                     shape[], blk[], np[], count(isfinite, out[].dx), filt[])
end

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

function report(r::E2EResult)
    @printf("  %-56s %s\n", first(r.case, 56), r.platform)
    for (name, secs) in r.stages
        @printf("    %-12s %8.1f s\n", name, secs)
    end
    @printf("    %-12s %8.1f s   cpu %.1f\n", "total", total_seconds(r), r.cpu)
    @printf("    scene %d x %d, block %dx%d, %s, %d of %d points measured\n", r.scene...,
            r.block..., r.filter, r.measured, r.npoints)
    @printf("    peak %.2f GiB (%.2f above the floor %.2f)\n", r.peak / 2^30,
            (r.peak - r.floor) / 2^30, r.floor / 2^30)
    flush(stdout)
end

function tsv_line(r::E2EResult)
    st = Dict(r.stages)
    return join([r.case, r.platform,
                 @sprintf("%.2f", get(st, "geometry", NaN)),
                 @sprintf("%.2f", get(st, "grid", NaN)),
                 @sprintf("%.2f", get(st, "imagery", NaN)),
                 @sprintf("%.2f", get(st, "correlate", NaN)),
                 @sprintf("%.2f", total_seconds(r)), @sprintf("%.1f", r.cpu),
                 string(r.peak), string(r.floor),
                 "$(r.scene[1])x$(r.scene[2])", "$(r.block[1])x$(r.block[2])",
                 string(r.npoints), string(r.measured), r.filter], '\t')
end

const TSV_HEADER = join(["case", "platform", "geometry_s", "grid_s", "imagery_s", "correlate_s",
                         "total_s", "cpu_s", "peak_bytes", "floor_bytes", "scene", "block",
                         "npoints", "measured", "filter"], '\t')

function main(args)
    isempty(args) && error("usage: e2e_run.jl <product-name-fragment>... | --all " *
                           "[--run N] [--block N] [--no-trace] [--stream] [--cold] [--tsv FILE]")
    cs = "--all" in args ? cases() : vcat((cases(a) for a in positional(args))...)
    n = "--run" in args ? parse(Int, argvalue("--run", "0")) : nothing
    block = "--block" in args ? parse(Int, argvalue("--block", "0")) : nothing
    tsv = argvalue("--tsv", nothing)
    trace = !("--no-trace" in args)
    staged = !("--stream" in args)
    warm = !("--cold" in args)

    @printf("%d case(s), %d threads\n\n", length(cs), Threads.nthreads())
    rows = E2EResult[]
    for c in cs
        reason = unsupported(c)
        if !isnothing(reason)
            @printf("  %-56s %s\n    skipped: %s\n", first(c.product, 56), c.platform,
                    replace(reason, r"\s+" => " "))
            flush(stdout)
            continue
        end
        try
            r = run_case(c; n, block, trace, staged, warm)
            push!(rows, r)
            report(r)
        catch err
            @printf("  %-56s %s\n    FAILED: %s\n", first(c.product, 56), c.platform,
                    first(sprint(showerror, err), 300))
            flush(stdout)
        end
    end

    if !isnothing(tsv) && !isempty(rows)
        open(tsv, "w") do io
            println(io, TSV_HEADER)
            foreach(r -> println(io, tsv_line(r)), rows)
        end
        @printf("\nwrote %d row(s) to %s\n", length(rows), tsv)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main(ARGS)
end
