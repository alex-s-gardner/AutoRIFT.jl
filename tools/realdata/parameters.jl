# The ITS_LIVE production parameters, read lazily from the rasters the granules point at.
#
# A granule records `autoRIFT_parameter_file` — a shapefile whose polygon over the scene names one
# GeoTIFF per parameter. Those rasters are what production actually ran with, so reading them is
# the difference between comparing against ITS_LIVE and comparing against a guess at ITS_LIVE.

using Rasters, ArchGDAL, NetworkOptions

const PARAMETER_BASE =
    "/vsicurl/https://its-live-data.s3.amazonaws.com/autorift_parameters/v001"

# One entry per parameter the correlator needs. Names follow this package's own, not the file
# names', so a caller reads `chip_size_max_x` rather than decoding `xMaxChipSize`.
#
# `vx0`/`vy0` are the a-priori velocity the search centres on, and `vxSearchRange`/`vySearchRange`
# the half-width around it — both in m/yr, so converting either to pixels needs the pair's
# separation (`search_radius_pixels`). The two go together: production's radius is a few pixels
# because it is centred on the prior, and using that radius without the prior searches a few pixels
# around zero, which finds nothing on fast ice.
const PARAMETER_LAYERS = (
    prior_vx = "vx0",
    prior_vy = "vy0",
    search_range_x = "vxSearchRange",
    search_range_y = "vySearchRange",
    chip_size_min_x = "xMinChipSize",
    chip_size_min_y = "yMinChipSize",
    chip_size_max_x = "xMaxChipSize",
    chip_size_max_y = "yMaxChipSize",
    stable_surface = "StableSurface",
)

"""
    production_parameters(extent; region = "NPS", layers = keys(PARAMETER_LAYERS)) -> RasterStack

The ITS_LIVE production parameters over `extent`, as a lazy `RasterStack` in the region's
projection.

`extent` is an `Extents.Extent` in that projection — EPSG:3413 for `"NPS"`, the northern polar
stereographic grid Greenland is processed on. Nothing is read until the stack is indexed or
materialized.

The rasters are 68480² cloud-optimized GeoTIFFs of about 200 MB each, tiled 256² with six overview
levels, so a windowed read fetches only the tiles it covers — under a second for a 300² window
against roughly 700 MB of parameters that never land on disk. Reading them eagerly would make this
test's cache larger than the imagery it exists to compare.

Certificate verification is configured here rather than left to the caller. Julia's bundled GDAL
does not find the system trust store on its own, and the failure surfaces as `CURL error: SSL
certificate problem` — or, once GDAL has swallowed it, as `Pointer 'hDS' is NULL`, which names
nothing useful. The variable that fixes it is `CURL_CA_BUNDLE`; `GDAL_HTTP_CAINFO` alone does not,
which is worth knowing before spending an afternoon on it.
"""
function production_parameters(extent; region::AbstractString = "NPS",
                               layers = keys(PARAMETER_LAYERS))
    # `EMPTY_DIR` stops GDAL listing the bucket prefix before every open, which for a public bucket
    # is a few seconds of wasted requests per raster.
    ArchGDAL.setconfigoption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    # Set in the process environment, not through `setconfigoption`: the curl handle reads it from
    # there, and a GDAL config option of the same name does not reach it.
    for var in ("CURL_CA_BUNDLE", "SSL_CERT_FILE", "GDAL_HTTP_CAINFO")
        haskey(ENV, var) || (ENV[var] = NetworkOptions.ca_roots_path())
    end

    rasters = map(layers) do name
        file = "$(region)_0120m_$(PARAMETER_LAYERS[name]).tif"
        Raster(joinpath(PARAMETER_BASE, file); lazy = true)
    end
    return RasterStack(NamedTuple{Tuple(layers)}(rasters))[extent]
end

"""
    search_radius_pixels(search_range, date_dt, pixel_size) -> Matrix{Int}

Convert an ITS_LIVE search range in m/yr to a search radius in pixels for a pair separated by
`date_dt` days.

The range is a velocity ceiling rather than a displacement, because one parameter file serves every
pair over a region and pairs differ in separation. A 12,000 m/yr ceiling is 263 m over eight days
and 1,150 m over 35, which at 15 m is 18 px against 77.

Rounded up, so the radius covers the ceiling rather than stopping just inside it, and floored at 1
where the range is zero — a zero radius means "do not search", which the caller expresses by
masking rather than by a radius.
"""
function search_radius_pixels(search_range::AbstractArray, date_dt::Real, pixel_size::Real)
    return map(search_range) do v
        r = ismissing(v) ? 0 : Int(v)
        r <= 0 ? 0 : max(1, ceil(Int, r / 365.25 * date_dt / pixel_size))
    end
end
