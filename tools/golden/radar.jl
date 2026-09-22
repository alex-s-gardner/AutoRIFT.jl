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
using SLCDatasets: annotation, asf_bursts, bursts, merge_bursts, nbursts, open_slc, orbit,
                   seconds_between, Sentinel1Product

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
