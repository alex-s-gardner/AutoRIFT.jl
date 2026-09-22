# The ITS_LIVE production parameters, resolved and read the way the reference resolves and reads
# them.
#
# Every golden product records `autoRIFT_parameter_file` as
# `autorift_parameters/v001/autorift_landice_0120m.shp`, a 122-feature polygon layer covering the
# UTM zones plus the two polar stereographic grids. One feature contains the pair's centroid, and
# its attributes name one cloud-optimized GeoTIFF per parameter and the EPSG code of the output
# grid. So the region is a lookup against the scene, not a configuration choice.
#
# Separate from `tools/realdata/parameters.jl`, which reads the same rasters for a different
# purpose: that one takes a projected extent and returns a lazy `RasterStack` in a region the
# caller names. This one looks the region up, and reads by *grid index* — the window
# `ImagePairGeometry.grid_window` computed — because the geogrid comparison needs the same cells
# the reference read, and a coordinate extent does not pin those.

using ArchGDAL, NetworkOptions, ImagePairGeometry
using GeoFormatTypes: EPSG

const PARAMETER_SHAPEFILE =
    "/vsicurl/https://its-live-data.s3.amazonaws.com/autorift_parameters/v001/autorift_landice_0120m.shp"

"""
    gdal_network_setup()

Configure GDAL for the anonymous `/vsicurl` reads this file makes.

Two settings, each for a failure that names nothing useful:

`CURL_CA_BUNDLE` and friends because Julia's bundled GDAL does not find the system trust store, and
the failure surfaces as `CURL error: SSL certificate problem` — or, once GDAL has swallowed it, as
`Pointer 'hDS' is NULL`. They go in the process environment rather than through
`setconfigoption`: the curl handle reads them from there.

`GDAL_DISABLE_READDIR_ON_OPEN` because otherwise GDAL lists the bucket prefix before every open,
which for a prefix holding thousands of parameter rasters is seconds of wasted requests per file.
"""
function gdal_network_setup()
    for var in ("CURL_CA_BUNDLE", "SSL_CERT_FILE", "GDAL_HTTP_CAINFO")
        haskey(ENV, var) || (ENV[var] = NetworkOptions.ca_roots_path())
    end
    ArchGDAL.setconfigoption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    return nothing
end

# `GeometryInputs` field to shapefile attribute. The attribute names are truncated to ten
# characters because a DBF field name cannot be longer, which is why `vxSearchRan` and
# `StableSurfa` appear cut off — they are, in the file.
#
# `vx0`/`vy0` rather than `vx`/`vy`: the geogrid's `window_offset` is the displacement a *prior*
# velocity implies, and `vx`/`vy` are the measured field the prior is derived from. Established by
# comparison — with `vx0`/`vy0` every integer band is bitwise identical to the container's output.
#
# `dhdx`/`dhdy` rather than `dhdxs`/`dhdys`: same test. The smoothed pair also carries `NaN` over
# its nodata, which the reference's integer conversion cannot represent at all.
const PARAMETER_ATTRIBUTES = (dem = "h", dhdx = "dhdx", dhdy = "dhdy",
                              vx = "vx0", vy = "vy0",
                              srx = "vxSearchRan", sry = "vySearchRan",
                              csminx = "xMinChipSiz", csminy = "yMinChipSiz",
                              csmaxx = "xMaxChipSiz", csmaxy = "yMaxChipSiz",
                              ssm = "StableSurfa")

"""
    parameter_info(lon, lat) -> NamedTuple

The parameter region covering `(lon, lat)`: its `name`, output `epsg`, and one `/vsicurl` path per
[`PARAMETER_ATTRIBUTES`](@ref) entry.

The lookup is containment of the pair's centroid, which resolves uniquely — the UTM-zone and polar
polygons do not overlap in the layer, so there is no priority rule to reproduce.

Throws if no feature contains the point, since a pair outside the parameter coverage cannot be
processed at all and silently falling back to a region would compare against the wrong rasters.
"""
function parameter_info(lon::Real, lat::Real)
    gdal_network_setup()
    ds = ArchGDAL.read(PARAMETER_SHAPEFILE)
    layer = ArchGDAL.getlayer(ds, 0)
    defn = ArchGDAL.layerdefn(layer)
    names = [ArchGDAL.getname(ArchGDAL.getfielddefn(defn, i))
             for i in 0:(ArchGDAL.nfield(defn) - 1)]
    field(f, name) = ArchGDAL.getfield(f, findfirst(==(name), names) - 1)

    pt = ArchGDAL.createpoint(Float64(lon), Float64(lat))
    hit = nothing
    for k in 0:(ArchGDAL.nfeature(layer) - 1)
        ArchGDAL.getfeature(layer, k) do f
            if ArchGDAL.contains(ArchGDAL.getgeom(f), pt)
                hit = (; name = String(field(f, "name")), epsg = Int(field(f, "epsg")),
                       paths = NamedTuple{keys(PARAMETER_ATTRIBUTES)}(
                           map(a -> _vsicurl(String(field(f, a))), values(PARAMETER_ATTRIBUTES))))
            end
        end
        hit === nothing || break
    end
    hit === nothing && error("no ITS_LIVE parameter region contains ($lon, $lat); the pair is " *
                             "outside the coverage of $PARAMETER_SHAPEFILE")
    return hit
end

# The shapefile stores plain HTTPS URLs; GDAL needs the `/vsicurl/` prefix to read one as a raster.
_vsicurl(url::AbstractString) = startswith(url, "/vsicurl/") ? url : "/vsicurl/" * url

"""
    parameter_grid(info) -> MapGrid

The output grid, from the region's DEM.

The DEM is what `geogridOptical` takes its geotransform, size and nodata value from
(`geogridOptical.cpp:337-339`), so every other parameter raster is read at the *DEM's* window and
the grid is the DEM's grid. `NPS_0120m_h.tif` is 68480² at 120 m; a window of it is what a pair
actually covers.
"""
function parameter_grid(info)
    gdal_network_setup()
    dem = ArchGDAL.read(info.paths.dem)
    return MapGrid(geotransform = Tuple(ArchGDAL.getgeotransform(dem)),
                   size = (ArchGDAL.width(dem), ArchGDAL.height(dem)), crs = info.epsg)
end

"""
    parameter_window(path, window) -> Matrix{Float64}

One parameter raster over `window`, as `Float64`.

`Float64` because that is the type the reference reads every one of them as — `GDT_Float64` in each
`RasterIO` call (`geogridOptical.cpp:700-850`) — so a chip size stored as `Int16` reaches the
kernel as a float on both sides and no conversion difference can enter.

`window` is one-based `CartesianIndices`, as [`ImagePairGeometry.grid_window`](@ref) returns it;
GDAL's offsets are zero-based. Getting that wrong shifts every parameter raster by one pixel, which
leaves the location band exact — it reads no parameter — and perturbs every band that does read
one, by whole steps of whatever the raster quantizes to.
"""
function parameter_window(path::AbstractString, window::CartesianIndices{2})
    xs, ys = window.indices
    ds = ArchGDAL.read(path)
    return Float64.(ArchGDAL.read(ds, 1, first(xs) - 1, first(ys) - 1, length(xs), length(ys)))
end

"""
    geometry_inputs(info, window) -> GeometryInputs

Every parameter raster the geogrid reads, over `window`.

Twelve windowed reads against cloud-optimized GeoTIFFs, so only the tiles the window covers are
fetched — a 1009² window costs a few seconds against roughly 2.4 GB of parameters that never land
on disk.
"""
function geometry_inputs(info, window::CartesianIndices{2})
    gdal_network_setup()
    bands = map(p -> parameter_window(p, window), info.paths)
    return GeometryInputs(; bands...)
end
