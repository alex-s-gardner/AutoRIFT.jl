# The Sentinel-1 radar geometry geogrid is given, built the way the pipeline builds it.
#
# `loadMetadataSlc` (`vend/testGeogrid.py:162-212`) is the whole specification, and two things about it
# make the radar path far cheaper than it looks.
#
# **The coregistration does not enter the geometry.** `s1_isce3.process_slc` calls
# `loadMetadataSlc(safe_ref, orbit_ref, ...)` and then sets `meta_s = copy.copy(meta_r)` with only
# `sensingStart` and `sensingStop` replaced. So every number geogrid consumes — and therefore every
# `window_*.tif` band — comes from the *reference* acquisition's burst annotations and its orbit. What
# COMPASS's per-burst resample and hyp3's `merge_swaths` produce is `secondary.tif`'s pixel values, which
# the geometry rungs never read.
#
# **The mosaic's grid is mostly metadata, and one number of it is not.** hyp3 lays all three subswaths
# on the near-range subswath's range origin and zero-fills between them (`s1_isce3.merge_swaths`), and
# `loadMetadataSlc:210` hands geogrid that mosaic's own shape. Both extents are reproduced here from burst
# annotations and the orbit, so no granule is transferred and no pixels merged — and on three of the five
# golden Sentinel-1 SLC pairs the result is exact against `reference.tif`:
#
#     case                      julia            reference.tif
#     S1A ... 20150828       64751 x 15858      64751 x 15858    exact
#     S1A ... 20151120       66172 x 23856      65978 x 23857    194 wide, 1 short
#     S1A ... 20170221       67945 x 23862      67860 x 24043     85 wide, 181 short
#     S1B ... 20180809       67945 x 23860      67945 x 23860    exact
#     S1C ... 20250416       65643 x 23852      65643 x 23852    exact
#
# **What the two outliers need is a pixel product, not metadata.** `total_rng_samples` is
# `last_rng_samples + floor((far.starting_range - near.starting_range) / dr)`, and `last_rng_samples` is
# `merge_bursts_in_swath`'s `num_rng_samples` — read off the *COMPASS CSLC raster* of the far subswath's
# first burst (`s1_isce3.py:562`), not off the annotation. It equals `samples_per_burst` on the three exact
# cases and is 194 and 85 narrower on the other two. Leading-invalid-sample trimming does not explain it:
# `first_valid_sample` is 45, 592, 580, 524 and 164 across the five against needed deltas of 0, 194, 85, 0
# and 0. The azimuth extent inherits the same dependency, since the per-swath merged length is built from
# the CSLC's own `num_az_samples`.
#
# So the derivation is exact where COMPASS's burst width is the annotation's and needs that width
# otherwise. It is left deriving rather than reading `reference.tif`, so the rung that compares the two
# stays a test.
#
# **`dt` is full precision here, unlike the optical path.** `testGeogrid.py:439` sets
# `repeatTime = (info1.sensingStart - info.sensingStart).total_seconds()` for radar, against `:354`'s
# whole-calendar-day difference for optical. The two conventions are opposite and each is wrong for the
# other path.

using Dates, Printf
using ImagePairGeometry
using ImagePairGeometry: IdentityTransform, LookRight, incidence_angle
using SLCDatasets
using SLCDatasets: annotation, asf_bursts, burst_raster, bursts, merge_bursts, nbursts,
                   open_slc, orbit, seconds_between, Sentinel1Product

# Sentinel-1 IW covers the swath with three subswaths, and a full-SLC job processes all three
# (`s1_isce3.process_slc` defaults `swaths=(1, 2, 3)`).
const S1_SWATHS = 1:3

"""
    s1_polarization(granule) -> String

The channel the pipeline correlates, from the granule name.

`process.py` asks COMPASS for `co-pol`, which is the like-polarized channel: `HH` for an `SH` or `DH`
product and `VV` for an `SV` or `DV` one. The cross-polarized channel of a dual-pol product is never
the one tracked.
"""
function s1_polarization(granule::AbstractString)
    m = match(r"_1S(S|D)(H|V)_", granule)
    m === nothing && throw(ArgumentError(
        "cannot read a polarization out of \"$granule\"; expected a field like `1SSH` or `1SDV`"))
    return m.captures[2] == "H" ? "HH" : "VV"
end

"""
    s1_orbit(dir, granule) -> String

The orbit file in `dir` whose validity window covers `granule`'s acquisition.

A run directory holds the orbits the container downloaded for that pair, one per granule, and their
names carry the window as `V<start>_<stop>`. Chosen by containment rather than by position, because the
two files are not in a defined order and picking the wrong one puts the state vectors a day away — an
error the geometry would absorb into a plausible-looking footprint rather than reject.
"""
function s1_orbit(dir::AbstractString, granule::AbstractString)
    t = DateTime(split(granule, '_')[6], dateformat"yyyymmddTHHMMSS")
    for path in sort(readdir(dir; join = true))
        endswith(path, ".EOF") || continue
        m = match(r"_V(\d{8}T\d{6})_(\d{8}T\d{6})\.EOF$", basename(path))
        m === nothing && continue
        a = DateTime(m.captures[1], dateformat"yyyymmddTHHMMSS")
        b = DateTime(m.captures[2], dateformat"yyyymmddTHHMMSS")
        a <= t <= b && return path
    end
    throw(ErrorException(
        "no orbit file in $dir covers $t. The container downloads one per granule beside its " *
        "outputs; a pruned or partial run has none, and geogrid cannot place a swath without it."))
end

"""
    AsfSwaths(granule, polarization, orbit_path)

An acquisition whose subswaths are read from ASF's burst extractor.

Two fetches per subswath and no granule transfer: the first burst's metadata file carries the whole
subswath's annotation, including how many bursts it has, and the second call is served from the same
cached file. The rasters are fetched only when read, which the geometry never does.
"""
struct AsfSwaths
    granule::String
    polarization::String
    orbit::String
end

"""
    SafeSwaths(product::Sentinel1Product)

An acquisition whose subswaths are read from an already-parsed local SAFE.

This is the route a burst job takes. Its granule name is `burst2safe`'s own — synthesized from the
requested bursts, with a checksum suffix that is not the one the product name carries — so it names
nothing at ASF and the annotations have to come off the container the container built.
"""
struct SafeSwaths
    product::Sentinel1Product
end

"""
    swath_annotation(src, swath) -> SubswathAnnotation

`swath`'s annotation, which is every number the merged grid is derived from bar the orbit.
"""
swath_annotation(src::AsfSwaths, swath::Integer) =
    only(asf_bursts(src.granule, swath, src.polarization, 1:1; orbit = src.orbit)).backend.annotation
swath_annotation(src::SafeSwaths, swath::Integer) = annotation(src.product, swath)

"""
    merged_swath(src, swath) -> SLC

One subswath of the acquisition, its bursts merged.

Wanted for what an annotation does not carry: the orbit, its epoch and the look side.
"""
function merged_swath(src::AsfSwaths, swath::Integer)
    n = nbursts(swath_annotation(src, swath))
    return merge_bursts(asf_bursts(src.granule, swath, src.polarization, 1:n; orbit = src.orbit))
end
merged_swath(src::SafeSwaths, swath::Integer) = merge_bursts(collect(bursts(src.product, swath)))

"""
    s1_mosaic(src, swaths = S1_SWATHS) -> (RadarCoordinate, UtcTime)

The merged radar grid geogrid is handed for the acquisition `src` reaches.

Built from the near-range subswath and then widened to the mosaic, which is what `loadMetadataSlc`
does: the range origin, sample spacing, PRF, wavelength and orbit are the near subswath's, the sample
count spans to the far subswath's far edge, and the line count comes from the sensing interval across
*all* subswaths rather than from any one of them — subswaths are acquired at slightly different azimuth
times, so the union is wider than each.

The incidence angle is recomputed rather than carried over, because it is the angle at the scene centre
and the centre moves when the range extent widens from one subswath to three.

The absolute sensing start comes back alongside, because the pair's interval is the difference of two
acquisitions' and each acquisition's own `sensing_start` is relative to its own orbit epoch.
"""
function s1_mosaic(src::Union{AsfSwaths,SafeSwaths}, swaths = S1_SWATHS)
    ann = [swath_annotation(src, sw) for sw in swaths]

    # **The mosaic's extents are `merge_swaths`'s, not a subswath's.** `loadMetadataSlc:210` takes both
    # `numberOfLines` and `numberOfSamples` straight from the merged shape when it is given one — the
    # closed-form width at `:197-200` is only the fallback — so the grid geogrid is told about is the
    # mosaic hyp3 laid out, and both extents have to be reproduced from its own arithmetic.
    lo = argmin(a -> a.starting_range, ann)
    hi = argmax(a -> a.starting_range, ann)
    dr = lo.range_pixel_spacing
    adt = lo.azimuth_time_interval

    # Per subswath, the merged azimuth length: bursts are placed at their valid-line offsets and the
    # merge runs from the first burst's start to the last burst's start plus one burst
    # (`merge_bursts_in_swath:575-579`).
    swath_lines = [1 + round(Int, (seconds_between(first(a.burst_start), last(a.burst_start)) +
                                   (a.lines_per_burst - 1) * a.azimuth_time_interval) /
                                  a.azimuth_time_interval) for a in ann]

    # **And then the swath stack adds that whole merged length to the *last* burst's start again**
    # (`merge_swaths:437-439`: `burst_sensing_stop = ref_bursts[-1].sensing_start + burst_length`, where
    # `burst_length` spans `burst_az_samples`, the merged swath rather than one burst). So the mosaic
    # reaches about 1.7 times a subswath's height — 15858 rows against 9145 lines of acquisition on
    # `S1A_IW_SLC__1SSH_20150828`. Reproduced because it is the extent geogrid bounds its azimuth index
    # against: the reference's own `window_location` band 2 runs 0..15857.
    start = minimum(a -> first(a.burst_start), ann)
    span = maximum(zip(ann, swath_lines)) do (a, n)
        seconds_between(start, last(a.burst_start)) + (n - 1) * a.azimuth_time_interval
    end
    nlines = 1 + round(Int, span / adt)

    # `floor` here, against the fallback formula's `round`, and the far subswath's own sample count.
    nsamples = hi.samples_per_burst +
               floor(Int, (hi.starting_range - lo.starting_range) / dr)

    prf = 1 / adt

    # The near subswath's merged acquisition supplies what an annotation does not carry: the orbit, its
    # epoch, and the look side.
    near = argmin(i -> ann[i].starting_range, eachindex(ann))
    base = merged_swath(src, collect(swaths)[near])
    c = RadarCoordinate(base)
    base_start = first(_annotation_of(base).burst_start)

    kwargs = (; orbit = c.orbit, starting_range = lo.starting_range, dr,
              # `c.sensing_start` is seconds against the orbit epoch and corresponds to the near
              # subswath's first burst, so the shift to the earliest subswath is added there.
              sensing_start = c.sensing_start + seconds_between(base_start, start),
              prf, nsamples, nlines, look_side = c.look_side, wavelength = lo.wavelength,
              orbit_epoch_offset = c.orbit_epoch_offset)
    coord = RadarCoordinate(; kwargs..., incidence_angle = incidence_angle(; kwargs...))
    return coord, start
end

_annotation_of(s) = s.backend.annotation

"""
    s1_pair(c::GoldenCase, run) -> CoregisteredPair

`c`'s pair as geogrid receives it: the reference acquisition's geometry, and an interval.

The secondary contributes its sensing start and nothing else, since `process_slc` copies `meta_r` and
replaces only the sensing times. So the secondary's own grid — which is where the coregistration lives —
is deliberately absent from this, and a rung that compares the geogrid bands is comparing the reference
acquisition's geometry alone.

`dt` is the full-precision difference of sensing starts (`testGeogrid.py:439`). Using the optical path's
whole-day convention here would be wrong by up to half a day.
"""
function s1_pair(c::GoldenCase, run::AbstractString)
    # Acquisition order, not the job's: the products report a positive `date_dt` with the earlier scene as
    # `img1` even on the two jobs whose reference is the later acquisition, so the pipeline reorders the
    # pair before geogrid sees it exactly as it does on the optical path.
    rg, sg = acquisition_order(c)
    src(g) = AsfSwaths(g, s1_polarization(g), s1_orbit(run, g))
    return _s1_pair(src(rg), src(sg), S1_SWATHS)
end

"""
    s1_burst_pair(c::GoldenCase, run) -> CoregisteredPair

`c`'s pair for a burst job, which reaches `process_slc` over a synthesized SAFE.

A burst job is the full-SLC path with two substitutions and no third
(`s1_isce3.process_sentinel1_burst_isce3:55-73`). `burst2safe` assembles the requested bursts into a
SAFE, and `process_slc` then runs on it unchanged — so `merge_swaths` mosaics whatever bursts that
container holds, and the merged extents follow from its annotations by the same arithmetic as a full
granule's. The substitutions are:

  * **The container is local.** Its name is `burst2safe`'s, not a granule ASF would serve, so the
    annotations are read from the SAFE the run directory holds rather than fetched.
  * **The subswath set is the bursts'.** `swaths = sorted(set(int(g.split('_')[2][2]) for g in
    reference))` (`:58`), so a job over `IW1` bursts alone mosaics one subswath and the range origin,
    width and incidence angle are that subswath's rather than the three-swath union's.

The two SAFEs are matched to the pair by acquisition time rather than by name: `burst2safe` stamps its
own checksum suffix, which does not agree with the one in the product name.
"""
function s1_burst_pair(c::GoldenCase, run::AbstractString)
    early, late = _burst_safes(run)
    pol(safe) = s1_polarization(basename(safe))
    src(safe) = SafeSwaths(Sentinel1Product(safe; orbit = s1_orbit(run, basename(safe)),
                                            polarization = lowercase(pol(safe)),
                                            swaths = burst_swaths(c)))
    return _s1_pair(src(early), src(late), burst_swaths(c))
end

# Shared by both routes, because only the annotation source and the subswath set differ: the reference
# acquisition's merged grid, and the interval between the two sensing starts.
function _s1_pair(ref_src, sec_src, swaths)
    ref, ref_start = s1_mosaic(ref_src, swaths)

    # Only the secondary's sensing start is wanted, so its geometry is built and its grid discarded.
    _, sec_start = s1_mosaic(sec_src, swaths)

    return CoregisteredPair(ref; dt = seconds_between(ref_start, sec_start))
end

"""
    burst_swaths(c::GoldenCase) -> Vector{Int}

The subswaths a burst job mosaics, from its reference burst list.

`s1_isce3.py:58` reads them out of the burst names — `S1_105602_IW2_...` contributes 2 — and off the
reference list alone, so a pair whose secondary reached a subswath the reference did not still
processes only the reference's.
"""
function burst_swaths(c::GoldenCase)
    sw = Int[]
    for g in c.reference
        m = match(r"_IW(\d)_", g)
        m === nothing && throw(ArgumentError(
            "\"$g\" does not name a Sentinel-1 IW subswath; a burst is named " *
            "`S1_<id>_IW<n>_<time>_<pol>_<hash>-BURST`"))
        push!(sw, parse(Int, m.captures[1]))
    end
    return sort!(unique!(sw))
end

# The pair's two SAFEs, earliest first. A burst run holds exactly two, both written by `burst2safe`.
function _burst_safes(run::AbstractString)
    safes = filter(n -> endswith(n, ".SAFE") && isdir(joinpath(run, n)), readdir(run))
    length(safes) == 2 || error(
        "expected the two SAFE products `burst2safe` assembles in $run, found $(length(safes)). " *
        "A burst job's annotations are read from them; a pruned run has none.")
    sort!(safes; by = n -> split(n, '_')[6])
    return joinpath(run, safes[1]), joinpath(run, safes[2])
end

"""
    nisar_l1_pair(c::GoldenCase, run) -> CoregisteredPair

A NISAR L1 RSLC pair, from the two products the run holds.

Much simpler than Sentinel-1 and for one reason: an RSLC is a single acquisition on a single radar grid,
so there is nothing to merge and nothing to mosaic. `loadMetadataRslc` (`testGeogrid.py:240-260`) reads
the zero-Doppler start, the dimensions and the orbit straight off the product, and the orbit travels
*inside* it rather than in a separate `.EOF` — so no orbit file is resolved here.

`dt` is the full-precision difference of the two zero-Doppler starts, which is what `runGeogrid`'s radar
branch takes (`:439`). `CoregisteredPair(::SLC, ::SLC)` computes exactly that.

The products are read from the run directory rather than fetched: they are 11.2 GiB each and the driver
already downloaded them there.
"""
function nisar_l1_pair(c::GoldenCase, run::AbstractString)
    early, late = acquisition_order(c)
    path(name) = begin
        p = joinpath(run, name * ".h5")
        isfile(p) || error("$(name).h5 is not in $run. A NISAR L1 pair is read from the products the " *
                           "driver downloaded there; each is 11.2 GiB.")
        p
    end
    return CoregisteredPair(open_slc(path(early)), open_slc(path(late)))
end

# ---------------------------------------------------------------------------
# Rung 5.2 — the merged radar mosaic the correlator is handed
# ---------------------------------------------------------------------------
#
# `merge_swaths` (`s1_isce3.py:393-530`) is the whole specification, and it is index arithmetic rather
# than signal processing: `read_slc_gdal` takes `np.abs` of each burst raster on the way in, so every
# array downstream is `Float32` amplitude and nothing is resampled. Two nested layouts:
#
#   1. **Bursts into a subswath** (`merge_bursts_in_swath`). Bursts overlap in azimuth, and the seam is
#      put *halfway through* each overlap so no burst contributes its resampling margin.
#   2. **Subswaths into the mosaic** (`merge_swaths`). Each subswath is laid on the near-range one's
#      range origin, and the writer is first-come: a pixel already non-zero is not overwritten.
#
# **The reference burst needs no coregistration.** COMPASS writes the reference burst on its own grid and
# deramping is phase-only, so `abs` of the CSLC is `abs` of the raw burst — measured on
# `S1C_IW_SLC__1SSV_20250416`'s first burst at a median difference of 0.0057 on amplitudes near 200, or
# 3e-5 relative. So the reference side of rung 5.2 is reproducible from the SAFE with no COMPASS run.

"""
    swath_amplitude(p::Sentinel1Product, swath) -> (Matrix{Float32}, Int, Int)

One subswath's bursts merged into a single amplitude raster, with its azimuth and range extents.

Reproduces `merge_bursts_in_swath`. The azimuth seam between two bursts is placed halfway through their
overlap, which is what keeps each burst's resampling margin out of the result, and the range window is
the *first* burst's valid sample range applied to every burst.

A single-burst subswath takes the early return: the burst is written whole, with no valid-region
cropping at all, so its extents are the burst's own.
"""
function swath_amplitude(p::Sentinel1Product, swath::Integer)
    a = annotation(p, swath)
    b = collect(bursts(p, swath))
    n = length(b)
    lpb, spb = a.lines_per_burst, a.samples_per_burst
    dt = a.azimuth_time_interval
    # One handle for the whole subswath: a `.SAFE` stacks every burst of a subswath in one raster, so a
    # burst is a row range of it. `read_pixels` on a single burst cannot be used — it is
    # `first(burst_raster(b))` on a `BurstRaster`, which is not iterable.
    raster = burst_raster(first(b).backend).raster
    amp(i, rows, cols) = abs.(raster[((i - 1) * lpb) .+ rows, cols])

    # The annotation's valid bounds are 1-based inclusive; every index below is the reference's 0-based.
    fvl = a.first_valid_line .- 1
    lvl = a.last_valid_line .- 1
    fvs = a.first_valid_sample .- 1
    lvs = a.last_valid_sample .- 1

    n == 1 && return (Float32.(amp(1, 1:lpb, 1:spb)), lpb, spb)

    # `get_azimuth_reference_offsets`: where each burst's valid region starts and ends in the merged
    # subswath, from its own sensing time and first valid line.
    lims = map(1:n) do i
        s = round(Int, (seconds_between(first(a.burst_start), a.burst_start[i]) + fvl[i] * dt) / dt)
        (s, s + (lvl[i] - fvl[i]) + 1)
    end
    nlines = 1 + round(Int, (seconds_between(first(a.burst_start), last(a.burst_start)) +
                             (lpb - 1) * dt) / dt)
    out = zeros(Float32, nlines, spb)
    for i in 1:n
        # `//` in the reference is floor division and these overlaps are positive, but `fld` says so.
        prev = i > 1 ? fld(lims[i - 1][2] - lims[i][1], 2) : 0
        nxt = i < n ? fld(lims[i][2] - lims[i + 1][1], 2) : 0
        bstart, bend = fvl[i] + prev, 1 + lvl[i] - nxt
        mstart, mend = lims[i][1] + prev, lims[i][2] - nxt
        cols = (fvs[i] + 1):lvs[i]          # `slice(first_valid_sample, last_valid_sample)`
        out[(mstart + 1):mend, cols] = amp(i, (bstart + 1):bend, cols)
    end
    return (out, nlines, spb)
end

"""
    radar_mosaic(p::Sentinel1Product, swaths) -> Matrix{Float32}

The merged amplitude raster `merge_swaths` writes as `reference.tif`.

Each subswath is placed at its own azimuth offset from the earliest sensing start and at its range offset
from the near subswath's origin, and only where the mosaic is still zero — the reference's writer is
first-come, which matters because adjacent subswaths overlap in range.

Two quirks of the extent are reproduced rather than corrected, both already established by
[`s1_mosaic`](@ref): the azimuth span adds a whole merged subswath length to the *last* burst's start, so
the mosaic reaches about 1.7 times a subswath's height; and the width is the **last** subswath's sample
count plus the floored range offset to it, not a union of the three.

A subswath other than the far one is trimmed by 64 samples at its far edge, which is the reference's
`invalid_pixel_buffer` — the resampling margin at a subswath's far range.
"""
function radar_mosaic(p::Sentinel1Product, swaths)
    sws = collect(swaths)
    ann = [annotation(p, sw) for sw in sws]
    merged = [swath_amplitude(p, sw) for sw in sws]

    dr = first(ann).range_pixel_spacing
    dt = first(ann).azimuth_time_interval
    starts = [first(a.burst_start) for a in ann]
    stops = [last(a.burst_start) for a in ann]
    sensing_start = minimum(starts)
    # `burst_sensing_stop` spans the *merged* subswath rather than one burst, which is the 1.7x quirk.
    span = maximum(zip(ann, merged)) do (a, m)
        seconds_between(sensing_start, last(a.burst_start)) + (m[2] - 1) * a.azimuth_time_interval
    end
    total_az = 1 + round(Int, span / dt)

    # `rng_offsets` are measured from the *first* subswath in the list, and `last_rng_samples` ends as the
    # last subswath's own width — not the widest.
    rng_offsets = [sw == first(sws) ? 0 :
                   floor(Int, (annotation(p, sw).starting_range - first(ann).starting_range) / dr)
                   for sw in sws]
    total_rng = last(merged)[3] +
                floor(Int, (last(ann).starting_range - first(ann).starting_range) / dr)

    out = zeros(Float32, total_az, total_rng)
    for k in eachindex(sws)
        a, (slc, nrows, _) = ann[k], merged[k]
        fvl, lvl = a.first_valid_line[1] - 1, a.last_valid_line[1] - 1
        fvs, lvs = a.first_valid_sample[1] - 1, a.last_valid_sample[1] - 1
        az_offset = floor(Int, seconds_between(sensing_start, starts[k]) / dt)
        rng_offset = rng_offsets[k] + fvs
        rng_end = rng_offset + (lvs - fvs)
        buffer = sws[k] == maximum(sws) ? 0 : 64
        # A single-burst subswath was written whole, so its azimuth window is the first burst's valid
        # region; a merged one runs to the *last* burst's last valid line counted from the end.
        slc_az_end = nbursts(a) > 1 ? nrows - (lvl_last(a)) : lvl
        merged_az_end = nbursts(a) > 1 ? az_offset + nrows - lvl_last(a) - fvl : az_offset + (lvl - fvl)
        mrows = (az_offset + 1):merged_az_end
        srows = (fvl + 1):slc_az_end
        mcols = (rng_offset + 1):(rng_end - buffer)
        scols = (fvs + 1):(lvs - buffer)
        dst = view(out, mrows, mcols)
        src = view(slc, srows, scols)
        for i in eachindex(dst, src)
            # First-come: `cond = merged == 0 & slc != 0`.
            (dst[i] == 0 && src[i] != 0) && (dst[i] = src[i])
        end
    end
    return out
end

# `bursts[-1].last_valid_line` of a subswath, 0-based. Named because the reference reaches it as a
# negative Python index, `slice(first_valid_line, -last_valid_line)`, which counts from the end.
lvl_last(a) = a.last_valid_line[end] - 1

"""
    dem_sampler(path) -> Function

Bilinear terrain height from a geographic DEM, as `(lon_degrees, lat_degrees) -> height`.

The DEM the container downloaded for the pair, which sits in the run directory as `dem.tif` on an
EPSG:4326 grid. Read whole rather than windowed: it is a few thousand cells on a side, and the
coregistration solve asks for scattered points rather than a block.
"""
function dem_sampler(path::AbstractString)
    ds = ArchGDAL.read(path)
    gt = ArchGDAL.getgeotransform(ds)
    z = ArchGDAL.read(ArchGDAL.getband(ds, 1))
    nx, ny = size(z)
    return function (lon_d, lat_d)
        px = (lon_d - gt[1]) / gt[2]
        py = (lat_d - gt[4]) / gt[6]
        i = clamp(floor(Int, px), 0, nx - 2)
        j = clamp(floor(Int, py), 0, ny - 2)
        fx, fy = px - i, py - j
        at(p, q) = Float64(z[p + 1, q + 1])
        return (1 - fx) * (1 - fy) * at(i, j) + fx * (1 - fy) * at(i + 1, j) +
               (1 - fx) * fy * at(i, j + 1) + fx * fy * at(i + 1, j + 1)
    end
end

"""
    coregistration_offset(cr, cs, line, sample, height; iters = 4) -> (dline, dsample, h)

Where the secondary images the ground point the reference images at `(line, sample)`, as an offset in
the reference's own pixels.

The orbit-driven coregistration, and the geometry half of rung 5.2's secondary side: `rdr2geo` on the
reference to reach the ground, `geo2rdr` on the secondary to come back. `line` and `sample` are
zero-based, as the reference's indices are.

**The terrain enters as an outer fixed point.** `rdr2geo` takes a constant height — all its callers in
the geogrid supply one — so the DEM is iterated: solve at the current height, look the DEM up at the
resulting position, solve again. Four passes, which is past convergence for Sentinel-1 geometry.

**The offsets are computed from the solved time and range, not through `azimuth_index`.** Those helpers
round to a whole line and sample for the geogrid's benefit, which is exactly the sub-pixel part a
resampler needs.

# The one line, and where it comes from

**`Geo2Rdr` does not define the azimuth offset as `(t - t0) * prf - line`.** It defines it one line
lower, so that is subtracted here. Measured against ISCE3 itself rather than inferred:
`tools/golden/isce_offsets.py` runs `Rdr2Geo` and `Geo2Rdr` with COMPASS's own arguments and reads the
`azimuth.off` and `range.off` the resampler consumes. Over twenty points spanning a burst of
`S1C_IW_SLC__1SSV_20250416`:

    range:   julia - isce3  mean -0.00000000   sd 8.0e-10
    azimuth: julia - isce3  mean +0.99999904   sd 5.1e-08   before this correction

So the geometry agrees with the reference implementation to eight decimal places on both axes and the
difference is a single exact constant. It is the *resampling position* that matters to a caller — the
input line a given output line reads from — and that is what this returns.

Confirmed independently against the imagery before ISCE3 was consulted, which is what said the residual
was real rather than a bookkeeping artifact: correlating `secondary.tif` against the *raw* secondary
burst locates the offset COMPASS actually used, since those are the same acquisition, and a parabola
through the peak over fifteen points gave +1.0162 +/- 0.0310 lines. The same correlation with the
*reference* on both sides peaks at `(0, 0)` with correlation 1.000, so the mosaic mapping is exact.
"""
function coregistration_offset(cr, cs, line::Integer, sample::Integer, height;
                               iters::Integer = 4)
    el = Ellipsoid()
    az = cr.sensing_start + line / cr.prf
    rg = cr.starting_range + sample * cr.dr
    h = 0.0
    llh = ImagePairGeometry.SVector{3,Float64}(0.0, 0.0, 0.0)
    for _ in 1:iters
        llh = ImagePairGeometry.rdr2geo(cr.orbit, el, az, rg; height = h,
                                       wavelength = cr.wavelength, side = cr.look_side)
        h = height(llh[1] / ImagePairGeometry.DEG2RAD, llh[2] / ImagePairGeometry.DEG2RAD)
    end
    xyz = ImagePairGeometry.lonlat_to_xyz(el,
              ImagePairGeometry.SVector{3,Float64}(llh[1], llh[2], h))
    pm, vm = ImagePairGeometry.interpolate(cs.orbit, ImagePairGeometry.orbit_midtime(cs))
    p = ImagePairGeometry.geo2rdr(cs.orbit, xyz, ImagePairGeometry.midtime(cs),
                                 ImagePairGeometry.orbit_midtime(cs), pm, vm)
    # The `- 1` is `Geo2Rdr`'s convention, measured against it; see above.
    return ((p.aztime - cs.sensing_start) * cs.prf - line - 1,
            (p.range - cs.starting_range) / cs.dr - sample, h)
end
