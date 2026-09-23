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
# `loadMetadataSlc:210` hands geogrid that mosaic's own shape.
#
# **Both extents follow the burst shape `merge_bursts_in_swath` reads, and that is the CSLC's rather than
# the annotation's.** `total_rng_samples` is `last_rng_samples + floor((far.starting_range -
# near.starting_range) / dr)`, and `last_rng_samples` is `num_rng_samples` — taken off the *COMPASS CSLC
# raster* of the far subswath's first burst (`s1_isce3.py:565-566`). The azimuth extent inherits the same
# dependency, since the per-swath merged length is built from that raster's `num_az_samples`.
#
# The two agree exactly when the reference burst was written by `rdr2geo`, and not when it was resampled
# against a cached static topographic layer — `write_yaml`'s `use_static_layer` branch sets
# `bool_reference` false, so COMPASS coregisters the reference burst instead of writing it out, and the
# result lands on its own geocoded grid. Which step ran is readable off the burst directory: `x`/`y`/`z`
# beside a polarization-named SLC for `rdr2geo`, `azimuth.off` beside a stem-named one for the resample.
# On `S1A ... 20170221` the difference is 136 to 195 lines taller and 72 to 97 samples narrower per
# subswath, which is exactly its 181-short, 85-wide disagreement.
#
# [`cslc_grid`](@ref) reads that shape from `product/` when the run kept one, and returns `nothing`
# otherwise — where the annotation is already the right answer. So the derivation stays a derivation: it
# reads COMPASS's own intermediate, which is an *input* to `merge_swaths`, never `reference.tif`, which is
# what the rung compares against.
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
function s1_mosaic(src::Union{AsfSwaths,SafeSwaths}, swaths = S1_SWATHS; grid = nothing)
    sws = collect(swaths)
    ann = [swath_annotation(src, sw) for sw in sws]
    # **Both extents follow the burst shape `merge_bursts_in_swath` reads, which is the CSLC's.** It takes
    # `num_az_samples, num_rng_samples` off the first burst's resampled raster (`s1_isce3.py:565-566`), not
    # off the annotation, and the two coincide only when the reference burst was written by `rdr2geo`. A
    # pair whose reference took the resample path instead lands on COMPASS's own grid — 136 to 195 lines
    # taller and 72 to 97 samples narrower — and every extent below inherits that.
    shape = [grid === nothing ? (a.lines_per_burst, a.samples_per_burst) : grid(sw)
             for (a, sw) in zip(ann, sws)]

    # **The mosaic's extents are `merge_swaths`'s, not a subswath's.** `loadMetadataSlc:210` takes both
    # `numberOfLines` and `numberOfSamples` straight from the merged shape when it is given one — the
    # closed-form width at `:197-200` is only the fallback — so the grid geogrid is told about is the
    # mosaic hyp3 laid out, and both extents have to be reproduced from its own arithmetic.
    ilo = argmin(i -> ann[i].starting_range, eachindex(ann))
    ihi = argmax(i -> ann[i].starting_range, eachindex(ann))
    lo, hi = ann[ilo], ann[ihi]
    dr = lo.range_pixel_spacing
    adt = lo.azimuth_time_interval

    # Per subswath, the merged azimuth length: bursts are placed at their valid-line offsets and the
    # merge runs from the first burst's start to the last burst's start plus one burst
    # (`merge_bursts_in_swath:575-579`).
    swath_lines = [1 + round(Int, (seconds_between(first(a.burst_start), last(a.burst_start)) +
                                   (shape[k][1] - 1) * a.azimuth_time_interval) /
                                  a.azimuth_time_interval) for (k, a) in enumerate(ann)]

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
    nsamples = shape[ihi][2] +
               floor(Int, (hi.starting_range - lo.starting_range) / dr)

    prf = 1 / adt

    # The near subswath's merged acquisition supplies what an annotation does not carry: the orbit, its
    # epoch, and the look side.
    base = merged_swath(src, sws[ilo])
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
    return _s1_pair(src(rg), src(sg), S1_SWATHS; grid = cslc_grid(run, S1_SWATHS))
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
    return _s1_pair(src(early), src(late), burst_swaths(c); grid = cslc_grid(run, burst_swaths(c)))
end

# Shared by both routes, because only the annotation source and the subswath set differ: the reference
# acquisition's merged grid, and the interval between the two sensing starts.
function _s1_pair(ref_src, sec_src, swaths; grid = nothing)
    ref, ref_start = s1_mosaic(ref_src, swaths; grid)

    # Only the secondary's sensing start is wanted, so its geometry is built and its grid discarded — and
    # with it the question of which acquisition `product/` holds. The mosaic's shape is one thing rather
    # than one per acquisition, so the grid belongs to the coordinate that is kept.
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
    # A full-SLC granule names the mode without a subswath — `S1A_IW_SLC__1SSH_...` — and
    # `process_slc` defaults to all three. A burst names exactly one.
    all(g -> occursin("_IW_SLC_", g), c.reference) && return collect(S1_SWATHS)
    sw = Int[]
    for g in c.reference
        m = match(r"_IW(\d)_", g)
        m === nothing && throw(ArgumentError(
            "\"$g\" names neither a full-SLC product nor an IW subswath; a burst is named " *
            "`S1_<id>_IW<n>_<time>_<pol>_<hash>-BURST` and an SLC `S1A_IW_SLC__...`"))
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

`offsets` maps a subswath to a per-burst `(dl, ds)` pair, for a reference that was itself coregistered
and resampled rather than copied; see [`resampled_burst_dirs`](@ref). Each subswath then goes through
[`secondary_swath_amplitude`](@ref) with the product on both sides, since the resampling is the same and
only the source acquisition differs.
"""
function radar_mosaic(p::Sentinel1Product, swaths; offsets = nothing, grid = nothing)
    sws = collect(swaths)
    merged = [offsets === nothing ? swath_amplitude(p, sw) :
              secondary_swath_amplitude(p, p, sw, nothing; offsets = offsets(sw),
                                        grid = grid === nothing ? nothing : grid(sw))
              for sw in sws]
    return _stack_swaths(p, sws, merged)
end

"""
    burst_offsets(dir) -> (dl, ds)

COMPASS's per-pixel azimuth and range offsets for one burst, as callables on a 0-based `(line, sample)`.

`Geo2Rdr` marks a pixel it could not solve with -1e6. Those are moved far outside the burst rather than
passed through, so [`resample_burst`](@ref)'s own bounds guard drops them.
"""
function burst_offsets(dir::AbstractString)
    read_off(name) = permutedims(ArchGDAL.read(ArchGDAL.getband(
                                     ArchGDAL.read(joinpath(dir, name)), 1)))
    lookup(A) = (l, s) -> (v = A[l + 1, s + 1]; v <= -1e5 ? -1e7 : v)
    # An ISCE flat file is described by a sibling `.off.xml`, which GDAL only finds if it may list the
    # directory. `GDAL_DISABLE_READDIR_ON_OPEN` is `EMPTY_DIR` for the S3 reads elsewhere in these tools,
    # so it is lifted for these two opens and put back.
    prev = ArchGDAL.getconfigoption("GDAL_DISABLE_READDIR_ON_OPEN", "")
    ArchGDAL.setconfigoption("GDAL_DISABLE_READDIR_ON_OPEN", "NO")
    try
        return (lookup(read_off("azimuth.off")), lookup(read_off("range.off")))
    finally
        ArchGDAL.setconfigoption("GDAL_DISABLE_READDIR_ON_OPEN", prev)
    end
end

"""
    cslc_grid(run, swaths) -> (swath -> (lines, samples)) or `nothing`

The shape `merge_bursts_in_swath` reads per subswath: its **first** burst's resampled raster, since that is
the one whose `shape` it takes `num_az_samples` and `num_rng_samples` from and then applies to every burst
of the subswath.

`nothing` when the run kept no `product/`, or when any subswath has no resampled burst in it — a reference
written by `rdr2geo` has no offsets and its CSLC is the annotation's burst, so the annotation is already
right there.
"""
function cslc_grid(run::AbstractString, swaths)
    isdir(joinpath(run, "product")) || return nothing
    dirs = Dict(sw => resampled_burst_dirs(run, sw) for sw in swaths)
    any(isempty, values(dirs)) && return nothing
    shapes = Dict(sw => burst_offset_grid(first(dirs[sw])) for sw in swaths)
    return sw -> shapes[sw]
end

"""
    burst_offset_grid(dir) -> (lines, samples)

The grid COMPASS resampled one burst onto, read from `azimuth.off`'s header alone.

**This is not the annotation's burst, and `merge_bursts_in_swath` uses this one.** It takes
`num_az_samples, num_rng_samples` from the CSLC raster's own shape (`s1_isce3.py:565-566`) and then applies
the *annotation's* `first_valid_line` and `first_valid_sample` as windows into it. On the burst jobs the two
grids coincide, so the distinction is invisible; on the full-SLC pairs that take the resample path they do
not, and every extent in the mosaic depends on this one:

    subswath   annotation           CSLC and offsets
    IW1        1504 x 21530         1640 x 21458
    IW2        1515 x 25376         1696 x 25279
    IW3        1520 x 24454         1715 x 24369

Dimensions only, so this costs a header read rather than the quarter-gigabyte the offsets themselves are.
"""
function burst_offset_grid(dir::AbstractString)
    prev = ArchGDAL.getconfigoption("GDAL_DISABLE_READDIR_ON_OPEN", "")
    ArchGDAL.setconfigoption("GDAL_DISABLE_READDIR_ON_OPEN", "NO")
    try
        ds = ArchGDAL.read(joinpath(dir, "azimuth.off"))
        return (ArchGDAL.height(ds), ArchGDAL.width(ds))
    finally
        ArchGDAL.setconfigoption("GDAL_DISABLE_READDIR_ON_OPEN", prev)
    end
end

"""
    resampled_burst_dirs(run, swath; secondary = false) -> Vector{String}

The COMPASS burst directories of one subswath that hold `azimuth.off` and `range.off`, in burst order,
or empty when the acquisition was not resampled.

Whether the *reference* acquisition has them is the whole question for a mosaic. `write_yaml` sets
`bool_reference` false whenever a cached static topographic layer is available, so a reference burst is
then coregistered against that layer and resampled rather than written out; without one it runs `rdr2geo`
instead and its directory holds `x`/`y`/`z` and a polarization-named SLC with no offsets at all. So the
presence of the offsets is both the test for which path ran and the input needed to follow it.

The burst identifier encodes the absolute burst number along the track, so sorting the directory names
puts them in acquisition order.
"""
function resampled_burst_dirs(run::AbstractString, swath::Integer; secondary::Bool = false)
    root = joinpath(run, secondary ? "product_sec" : "product")
    isdir(root) || return String[]
    dirs = String[]
    for id in sort!(filter(endswith("_iw$(swath)"), readdir(root)))
        sub = joinpath(root, id)
        for date in sort!(readdir(sub))
            d = joinpath(sub, date)
            isdir(d) && isfile(joinpath(d, "azimuth.off")) && (push!(dirs, d); break)
        end
    end
    return dirs
end

# `merge_swaths`'s swath stack, over per-subswath rasters a caller has already merged. Shared by the
# reference and the secondary because the layout is the reference's in both cases.
function _stack_swaths(p::Sentinel1Product, swaths, merged)
    sws = collect(swaths)
    ann = [annotation(p, sw) for sw in sws]

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

# ---------------------------------------------------------------------------
# The secondary, coregistered onto the reference grid
# ---------------------------------------------------------------------------
#
# `ResampSlc` deramps each chip, interpolates with an eight-tap sinc, and reramps. Only the first two
# reach an amplitude: the reramp is a phase multiply on the finished value.
#
# **Every piece of this was chosen by measurement**, on burst 1 of `S1C_IW_SLC__1SSV_20250416` against
# `secondary.tif` over a 512 x 4096 window, reported as the ratio of means and the correlation:
#
#     no deramp, bicubic                 0.7535   0.73626
#     deramp with the opposite sign      0.7641   0.76199
#     deramp, bicubic                    0.9266   0.98640
#     deramp, eight-tap sinc unwindowed  1.2070   0.98365
#     deramp, eight-tap sinc + Hamming   1.0116   0.99963
#     deramp, eight-tap sinc + Hann      0.9976   0.99957
#
# Hann is also derivable rather than merely best: ISCE's `sinc_coef`, which `Sinc2dInterpolator` is built
# from, weights the sinc by `(1 - pedestal)/2 * cos(pi x / (ns/2)) + (1 + pedestal)/2`, and a pedestal of
# zero makes that exactly `0.5 + 0.5 cos(pi x / 4)` at `ns = 8`.

const CLIGHT = 299792458.0

"""
    range_poly(p::RangePolynomial, range) -> Float64

`p` evaluated at a slant `range` in metres.

**Not `p(range)`.** `RangePolynomial` stores `r0` in metres but keeps the annotation's coefficients in
powers of *slant-range time*, so calling it with a range mixes the two units and returns a number about
sixteen orders of magnitude out. The delta has to be the two-way time: at 838,557 m the FM rate is
−2224.66 Hz/s this way and −3.6e16 Hz/s the other.
"""
range_poly(p, range::Real) = evalpoly(2 * (Float64(range) - p.r0) / CLIGHT, p.coeffs)

"""
    TopsCarrier(slc, coord, lines_per_burst)

The azimuth carrier phase of one Sentinel-1 burst, as `(line, sample) -> radians`.

The TOPS beam sweep puts a quadratic azimuth phase on each burst whose instantaneous frequency reaches a
few kilohertz against a 486 Hz line rate, so the samples of any interpolation chip are aliased with
respect to each other and interpolating the raw complex signal cancels rather than sums — worth 25% of the
amplitude, measured. Removing it first is what `ResampSlc` does.

`s1reader.az_carrier_components` verbatim, and each of its three choices matters:

  * **azimuth time is measured from the middle line index**, `(line - lines_per_burst ÷ 2) * dt` with
    integer division — not from the burst's mid *time*, which is half a line away;
  * **the reference time is a difference**, `dc(r0)/fm(r0) - dc(r)/fm(r)`, so the quadratic's vertex
    carries a constant offset that the beam-centre crossing alone does not;
  * **there is no demodulation term.** The carrier is `pi * kt * (eta - eta_ref)^2` and nothing else.

`kt` is `ks / (1 - ks/ka)` where `ks = 2|v| * steering_rate / wavelength` at the burst mid.
"""
struct TopsCarrier
    fm::Any
    dc::Any
    ks::Float64
    dt::Float64
    r0::Float64
    dr::Float64
    mid_line::Int
    eta_ref0::Float64
end

# Whether the carrier carries the demodulation term as well as the quadratic. `s1reader` has only the
# quadratic; the flag exists because which one reproduces `secondary.tif` better is a measurement.


function TopsCarrier(slc, coord, lines_per_burst::Integer)
    dp = SLCDatasets.deramp_parameters(slc)
    _, v = ImagePairGeometry.interpolate(coord.orbit, ImagePairGeometry.orbit_midtime(coord))
    # The annotation gives the steering rate in degrees per second; `s1reader` holds it in radians.
    ks_rad = dp.azimuth_steering_rate * (abs(dp.azimuth_steering_rate) > 0.1 ? pi / 180 : 1.0)
    ks = ks_rad * 2 * sqrt(sum(abs2, v)) / dp.wavelength
    r0 = dp.starting_range
    eta_ref0 = range_poly(dp.doppler_centroid, r0) / range_poly(dp.azimuth_fm_rate, r0)
    return TopsCarrier(dp.azimuth_fm_rate, dp.doppler_centroid, ks, dp.azimuth_time_interval,
                       r0, dp.range_pixel_spacing, Int(lines_per_burst) ÷ 2, eta_ref0)
end

@inline function (c::TopsCarrier)(line::Integer, sample::Integer)
    r = c.r0 + sample * c.dr
    ka = range_poly(c.fm, r)
    fdc = range_poly(c.dc, r)
    kt = c.ks / (1 - c.ks / ka)
    eta = (line - c.mid_line) * c.dt
    de = eta - (c.eta_ref0 - fdc / ka)
    # The quadratic is `s1reader`'s; the linear term is ISCE3's separate use of the Doppler, which
    # `s1_resample.py` hands `ResampSlc` beside the carrier polynomial. Measured: without it `in_I2`
    # reaches 69.55% of exact bytes against 93.50% with it, and the opposite sign gives 62.78%. Its
    # `eta_ref` subtraction is immaterial — `eta_ref` is ~0.01 s against `eta`'s +/-1.5 — and dropping it
    # changes `in_I2` by 4e-4 of a percent.
    return pi * kt * de * de + 2pi * fdc * de
end

"""
    sinc8(f) -> NTuple{8,Float64}

The eight taps of a Hann-windowed sinc at fractional offset `f`, for taps at −3 through +4.

`Sinc2dInterpolator`'s kernel: ISCE's `sinc_coef` weights `sinc(x)` by
`(1 − pedestal)/2 · cos(πx/(ns/2)) + (1 + pedestal)/2`, which at `pedestal = 0` and `ns = 8` is
`0.5 + 0.5 cos(πx/4)`. Unwindowed the same eight taps overshoot by 21%, and a bicubic undershoots by 7%.
"""
@inline function sinc8(f::Float64)
    # Taps at -3 through +4, which is `Sinc2dInterpolator`'s centring: shifting them to -4 through +3
    # costs `in_I2` 93.50% of exact bytes against 81.76%.
    return ntuple(8) do k
        x = (k - 4) - f
        s = x == 0 ? 1.0 : sinpi(x) / (pi * x)
        s * (0.5 + 0.5 * cospi(x / 4))
    end
end

"""
    deramped_burst(raster, rows, carrier) -> Matrix{ComplexF32}

One burst's samples with its azimuth carrier removed, ready to interpolate.

Done to the whole burst once rather than per interpolation chip: a chip would evaluate the carrier 64
times per output pixel where this evaluates it once per input pixel, which is the difference between
minutes and hours over a subswath.
"""
function deramped_burst(raster, rows::AbstractUnitRange, carrier)
    nr, nc = length(rows), size(raster, 2)
    out = Matrix{ComplexF32}(undef, nr, nc)
    src = raster[rows, :]
    Threads.@threads for j in 1:nc
        for i in 1:nr
            out[i, j] = ComplexF32(src[i, j]) * cis(-Float32(carrier(i - 1, j - 1)))
        end
    end
    return out
end

# A burst's ramp margins zeroed in place, reproducing the source rectangle `slc_to_vrt_file` exposes.
# Bounds are the annotation's, converted to 0-based by the caller, and the array is 1-based.
function _zero_outside_valid!(A::AbstractMatrix, fvl::Integer, lvl::Integer, fvs::Integer, lvs::Integer)
    nr, nc = size(A)
    z = zero(eltype(A))
    @inbounds for j in 1:nc, i in 1:nr
        ((i - 1) < fvl || (i - 1) > lvl || (j - 1) < fvs || (j - 1) > lvs) && (A[i, j] = z)
    end
    return A
end

"""
    resample_burst(deramped, dl, ds, lines, samples) -> Matrix{Float32}

`deramped` read at each reference pixel's own position in the secondary, as amplitude.

`dl` and `ds` are callables giving the offsets at a reference `(line, sample)`; in practice they
interpolate a lattice, since the field moves by 0.0036 lines over 1200 lines and the lattice's own
interpolation error is 1e-5 px.

Amplitude rather than the complex value, so the reramp is omitted: it multiplies the finished sample by a
unit phasor and cannot change a magnitude.
"""
function resample_burst(deramped, dl, ds, lines::AbstractUnitRange,
                        samples::AbstractUnitRange, valid = nothing; doppler = nothing)
    out = zeros(Float32, length(lines), length(samples))
    nr, nc = size(deramped)
    Threads.@threads for (jj, s) in collect(enumerate(samples))
        for (ii, l) in enumerate(lines)
            # `_dopplerLUT.contains(az, rng)`, the third of `ResampSlc::_transformTile`'s five rejections
            # and the one the two bounds tests below do not cover.
            #
            # It is evaluated at the **output** pixel — `az = _sensingStart + iRow / _prf` and
            # `rng = _startingRange + iCol * _rangePixelSpacing` — while the LUT is built from the *source
            # burst's* own shape: `doppler_poly1d_to_lut2d` spans `starting_slant_range` to
            # `starting_slant_range + (samples_per_burst - 1) * dr` in range and `0` to
            # `lines_per_burst * dt` in azimuth (`s1_reader.py:131-168`). `ResampSlc` takes its
            # `_sensingStart` and `_startingRange` from the burst's grid, the first constructor argument,
            # rather than from the `ref_rdr_grid` keyword. So the whole test collapses to the output index
            # lying inside the source burst's dimensions, and it bites whenever the output grid is the
            # larger of the two — which is every full-SLC pair, where the CSLC is 136 to 195 lines taller
            # than the burst it came from.
            if doppler !== nothing
                (l > doppler[1] || s > doppler[2] - 1) && continue
            end
            y = l + dl(l, s) + 1.0
            x = s + ds(l, s) + 1.0
            iy, ix = floor(Int, y), floor(Int, x)
            (iy - 4 < 1 || iy + 4 > nr || ix - 4 < 1 || ix + 4 > nc) && continue
            # The eight-tap support has to sit inside the secondary's own valid region: outside it the
            # reference's resampler has no samples either, and interpolating the zero margin invents a
            # small non-zero value where `secondary.tif` holds none.
            if valid !== nothing
                (iy - 3 < valid[1] || iy + 4 > valid[2] || ix - 3 < valid[3] || ix + 4 > valid[4]) &&
                    continue
            end
            ty = sinc8(y - iy)
            tx = sinc8(x - ix)
            acc = ComplexF64(0)
            for m in 1:8
                row = ComplexF64(0)
                @simd for k in 1:8
                    row += tx[k] * ComplexF64(deramped[iy - 4 + m, ix - 4 + k])
                end
                acc += ty[m] * row
            end
            out[ii, jj] = Float32(abs(acc) / (sum(ty) * sum(tx)))
        end
    end
    return out
end

"""
    secondary_swath_amplitude(rp, sp, swath, dem) -> (Matrix{Float32}, Int, Int)

[`swath_amplitude`](@ref) for the secondary: each burst resampled onto the reference burst's grid first,
then placed by the **reference's** own seam arithmetic.

That the placement is the reference's is not a simplification — `merge_bursts_in_swath` takes
`az_reference_offsets` from `ref_bursts` and the range window from the reference burst's valid samples,
and applies both to the secondary. So the two mosaics share every index and differ only in pixel values.

The offsets come from a lattice of [`coregistration_offset`](@ref) solves, one node every 64 lines and 512
samples, interpolated bilinearly between. That is exact to about 1e-5 px on this field and turns 31
million geometry solves per burst into a few hundred.
"""
function secondary_swath_amplitude(rp::Sentinel1Product, sp::Sentinel1Product, swath::Integer, dem;
                                   offsets = nothing, grid = nothing)
    a = annotation(rp, swath)
    sa = annotation(sp, swath)
    n = nbursts(a)
    lpb, spb = a.lines_per_burst, a.samples_per_burst
    dt = a.azimuth_time_interval
    sraster = burst_raster(first(collect(bursts(sp, swath))).backend).raster
    fvl = a.first_valid_line .- 1
    lvl = a.last_valid_line .- 1
    fvs = a.first_valid_sample .- 1
    lvs = a.last_valid_sample .- 1

    lims = map(1:n) do i
        s = round(Int, (seconds_between(first(a.burst_start), a.burst_start[i]) + fvl[i] * dt) / dt)
        (s, s + (lvl[i] - fvl[i]) + 1)
    end
    # **The output grid is the one COMPASS resampled onto, which is the annotation's burst only sometimes.**
    # `merge_bursts_in_swath` reads its `num_az_samples`/`num_rng_samples` off the first burst's CSLC, so
    # the merged height and the merged width both follow that raster and not `lines_per_burst`.
    nl, ns = grid === nothing ? (lpb, spb) : grid(1)
    nlines = n == 1 ? nl :
             1 + round(Int, (seconds_between(first(a.burst_start), last(a.burst_start)) +
                             (nl - 1) * dt) / dt)
    out = zeros(Float32, nlines, ns)

    for i in 1:n
        ss = open_slc(sp.path; orbit = sp.orbit_path, swath = swath, burst = i,
                      polarization = sp.polarization)
        cs = RadarCoordinate(ss)
        carrier = CarrierFit(TopsCarrier(ss, cs, lpb), lpb, spb)
        dl, ds = if offsets === nothing
            cr = RadarCoordinate(open_slc(rp.path; orbit = rp.orbit_path, swath = swath, burst = i,
                                          polarization = rp.polarization))
            _offset_lattice(cr, cs, dem, lpb, spb)
        else
            offsets(i)
        end
        deramped = deramped_burst(sraster, ((i - 1) * lpb + 1):(i * lpb), carrier)
        # **The resampler's source is the burst zeroed outside its own valid window, not the raw burst.**
        # `slc_to_vrt_file` writes a VRT of the burst's full shape whose `SimpleSource` covers only
        # `first_valid_line:last_valid_line` by `first_valid_sample:last_valid_sample`, with
        # `NoDataValue` 0, so everything outside that rectangle reads as zero
        # (`s1_burst_slc.py:slc_to_vrt_file`). A TOPS burst's ramp-up and ramp-down margins carry small but
        # non-zero amplitudes — a median of 2.24 against a typical 50 on this pair — so reading them fills
        # pixels the reference leaves empty.
        #
        # **Zeroed rather than declined.** Rejecting an output pixel whose interpolation support straddles
        # the boundary is a different rule and a worse one: measured on IW1 burst 1 of
        # `S1A ... 20170221`, guarding declines 161,588 pixels the reference fills. Zeroing the source
        # keeps them, tapered, which is what the reference's own interpolation does with a source rectangle
        # that ends there.
        _zero_outside_valid!(deramped, sa.first_valid_line[i] - 1, sa.last_valid_line[i] - 1,
                             sa.first_valid_sample[i] - 1, sa.last_valid_sample[i] - 1)

        prev = i > 1 ? fld(lims[i - 1][2] - lims[i][1], 2) : 0
        nxt = i < n ? fld(lims[i][2] - lims[i + 1][1], 2) : 0
        bstart, bend = n == 1 ? (0, nl) : (fvl[i] + prev, 1 + lvl[i] - nxt)
        mstart, mend = n == 1 ? (0, nl) : (lims[i][1] + prev, lims[i][2] - nxt)
        # `min(lvs, ns)`: the valid-sample window is the annotation's and the raster it indexes is the
        # CSLC's, which can be the narrower of the two. NumPy's slicing truncates there in silence.
        cols = n == 1 ? (1:ns) : ((fvs[i] + 1):min(lvs[i], ns))
        # No valid-window guard: measured, ISCE3's own criterion is the raster's bounds rather than the
        # annotation's valid region. Guarding on the valid window declines 137,782 pixels `secondary.tif`
        # fills, to avoid filling 2,746 it does not — fifty times the error it removes.
        got = resample_burst(deramped, dl, ds, bstart:(bend - 1), (first(cols) - 1):(last(cols) - 1);
                             doppler = (lpb, spb))
        out[(mstart + 1):mend, cols] = got
    end
    return (out, nlines, ns)
end

# The offset field on a lattice, with bilinear interpolation between nodes. Two closures rather than two
# matrices so the caller reads a position rather than an index.
const _LSTEP, _SSTEP = 64, 512
function _offset_lattice(cr, cs, dem, lpb::Integer, spb::Integer)
    ls = 0:_LSTEP:(lpb + _LSTEP)
    ss = 0:_SSTEP:(spb + _SSTEP)
    DL = Matrix{Float64}(undef, length(ls), length(ss))
    DS = similar(DL)
    Threads.@threads for jj in eachindex(ss)
        for ii in eachindex(ls)
            DL[ii, jj], DS[ii, jj], _ = coregistration_offset(cr, cs, ls[ii], ss[jj], dem)
        end
    end
    lerp(A, l, s) = begin
        fi = l / _LSTEP + 1
        fj = s / _SSTEP + 1
        i = clamp(floor(Int, fi), 1, length(ls) - 1)
        j = clamp(floor(Int, fj), 1, length(ss) - 1)
        u, v = fi - i, fj - j
        (1-u)*(1-v)*A[i,j] + u*(1-v)*A[i+1,j] + (1-u)*v*A[i,j+1] + u*v*A[i+1,j+1]
    end
    return ((l, s) -> lerp(DL, l, s), (l, s) -> lerp(DS, l, s))
end

"""
    secondary_mosaic(rp, sp, swaths, dem) -> Matrix{Float32}

The merged amplitude raster `merge_swaths` writes as `secondary.tif`.

The same swath stack as [`radar_mosaic`](@ref) — the reference's extents, offsets and first-come writer —
over [`secondary_swath_amplitude`](@ref)'s coregistered subswaths.
"""
secondary_mosaic(rp::Sentinel1Product, sp::Sentinel1Product, swaths, dem; offsets = nothing,
                 grid = nothing) =
    _stack_swaths(rp, swaths,
                  [secondary_swath_amplitude(rp, sp, sw, dem;
                                             offsets = offsets === nothing ? nothing : offsets(sw),
                                             grid = grid === nothing ? nothing : grid(sw))
                   for sw in collect(swaths)])

"""
    CarrierFit(c::TopsCarrier, lines, samples)

`c` as the **fitted polynomial** ISCE3 actually evaluates, rather than the carrier itself.

`get_az_carrier_poly` samples the carrier every 50 lines and 500 samples and least-squares fits a
polynomial in normalized range and azimuth time, keeping the terms of `x^j y^i` with `i <= 5`, `j <= 3`
and `i + j <= 5` — eighteen of them (`s1reader.polyfit`, `max_order = True`). `ResampSlc` then evaluates
that fit, so reproducing the carrier exactly is *more* accurate than the reference and therefore different
from it.

**The one-sample and one-line shifts fall out of the fit rather than being applied by hand.** The carrier
is evaluated at index `(y, x)` but fitted against the coordinates of `(y + 1, x + 1)`, so the polynomial
returns, at a given pixel, the carrier of the pixel before it on both axes. That is where
`CARRIER_LINE_SHIFT`'s empirical `-1` came from, and with the fit in place the shift belongs at zero.
"""
struct CarrierFit
    coef::Vector{Float64}
    xmin::Float64
    xnorm::Float64
    ymin::Float64
    ynorm::Float64
    r0::Float64
    dr::Float64
    dt::Float64
end

# The eighteen exponent pairs, in `polyfit`'s own nesting order.
const _CARRIER_TERMS = [(i, j) for i in 0:5 for j in 0:3 if i + j <= 5]

function CarrierFit(c::TopsCarrier, lines::Integer, samples::Integer;
                   ystep::Integer = 50, xstep::Integer = 500)
    xs = 0:xstep:(samples - 1)
    ys = 0:ystep:(lines - 1)
    n = length(xs) * length(ys)
    # The coordinates the fit is against: one sample and one line past the index the value belongs to.
    rg = [c.r0 + (x + 1) * c.dr for x in xs]
    az = [(y + 1) * c.dt for y in ys]
    xmin, xnorm = minimum(rg), max(maximum(rg) - minimum(rg), 1.0)
    ymin, ynorm = minimum(az), max(maximum(az) - minimum(az), 1.0)
    A = Matrix{Float64}(undef, n, length(_CARRIER_TERMS))
    z = Vector{Float64}(undef, n)
    k = 0
    for (jy, y) in enumerate(ys), (jx, x) in enumerate(xs)
        k += 1
        xn = (rg[jx] - xmin) / xnorm
        yn = (az[jy] - ymin) / ynorm
        for (t, (i, j)) in enumerate(_CARRIER_TERMS)
            A[k, t] = xn^j * yn^i
        end
        z[k] = c(y, x)
    end
    return CarrierFit(A \ z, xmin, xnorm, ymin, ynorm, c.r0, c.dr, c.dt)
end

@inline function (f::CarrierFit)(line::Integer, sample::Integer)
    xn = ((f.r0 + sample * f.dr) - f.xmin) / f.xnorm
    yn = (line * f.dt - f.ymin) / f.ynorm
    acc = 0.0
    @inbounds for (t, (i, j)) in enumerate(_CARRIER_TERMS)
        acc += f.coef[t] * xn^j * yn^i
    end
    return acc
end
