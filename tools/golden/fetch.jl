# Fetch golden products, and report what their inputs would cost to fetch.
#
#   julia --project=tools/golden tools/golden/fetch.jl              # products for every case
#   julia --project=tools/golden tools/golden/fetch.jl --phase 3    # one phase
#   julia --project=tools/golden tools/golden/fetch.jl --check      # resolve inputs, download none
#
# Products are anonymous and small (145 MB for all 22). Scene inputs are neither, and reach four
# different services with three different credentials, so `--check` resolves them without
# downloading: it answers "can this be fetched at all" separately from fetching, which is what makes
# a credential failure distinguishable from a granule that is genuinely absent.

include("manifest.jl")

using Downloads

const AWS = "aws"

function aws_available()
    try
        run(pipeline(`$AWS --version`; stdout = devnull, stderr = devnull))
        return true
    catch
        return false
    end
end

"""
    fetch_products(cs; force = false) -> NamedTuple

Download each case's product and sidecars from the golden bucket. Anonymous — no credentials.

Existing files are left alone unless `force`, since a product is immutable once written.
"""
function fetch_products(cs::AbstractVector{GoldenCase}; force = false)
    mkpath(golden_dir())
    got = 0
    skipped = 0
    for c in cs
        targets = [(c.product * ".nc", golden_path(c));
                   [(basename(sidecar_path(c, e)), sidecar_path(c, e))
                    for e in (FILE_SIDECARS..., PRODUCT_SIDECARS...)]]
        for (key, dest) in targets
            if isfile(dest) && !force
                skipped += 1
                continue
            end
            url = "$BUCKET/$key"
            # `--no-sign-request` rather than a signed GET: the bucket is public, and an expired or
            # wrong profile in the environment would otherwise turn a working download into a 403.
            try
                run(pipeline(`$AWS s3 cp $url $dest --no-sign-request`;
                             stdout = devnull, stderr = devnull))
                got += 1
            catch
                @warn "failed to fetch" key
            end
        end
    end
    return (; got, skipped)
end

# --- Input resolution ------------------------------------------------------------------------
#
# Each platform's scenes come from a different place, with different auth. Resolution is a HEAD or a
# metadata query: it says whether the granule exists and whether we are allowed to read it, without
# paying for the bytes.

"""
    resolve_inputs(c::GoldenCase) -> NamedTuple

Whether `c`'s scene inputs are reachable, and by what route. Downloads nothing.

`status` is `:ok`, `:unauthorized` (the granule exists, we may not read it), `:missing` (the service
does not have it), or `:unsupported` (no route implemented for this platform yet).
"""
function resolve_inputs(c::GoldenCase)
    if c.platform == "S2"
        return resolve_s2(c)
    elseif startswith(c.platform, "L")
        return resolve_landsat(c)
    elseif c.platform in ("S1-SLC", "S1-BURST") || startswith(c.platform, "NISAR")
        return resolve_asf(c)
    end
    return (; status = :unsupported, route = "none", detail = c.platform)
end

# Sentinel-2 L1C, from the public Google Cloud mirror. `process.py::get_s2_path` prefers an
# `its-live-project` cache and falls back to this, so the fallback is the anonymous route.
function resolve_s2(c::GoldenCase)
    name = c.reference[1]
    tile = "$(name[40:41])/$(name[42:42])/$(name[43:44])"
    url = "https://storage.googleapis.com/gcp-public-data-sentinel-2/tiles/$tile/$name.SAFE/manifest.safe"
    code = head_status(url)
    status = code == 200 ? :ok : code in (401, 403) ? :unauthorized : :missing
    return (; status, route = "gcp-public-data-sentinel-2", detail = "HTTP $code")
end

# Landsat Collection 2 Level-1. Not in CMR: the collection is registered
# (`C3442493460-USGS_EROS`) but indexes zero granules, so EarthData cannot reach it. The STAC item
# is public and names both an authenticated HTTPS href and a requester-pays S3 one; reading the
# item confirms the granule exists and which band autoRIFT wants.
function resolve_landsat(c::GoldenCase)
    name = c.reference[1]
    url = "https://landsatlook.usgs.gov/stac-server/collections/landsat-c2l1/items/$name"
    body = try
        String(take!(Downloads.download(url, IOBuffer(); timeout = 60)))
    catch
        return (; status = :missing, route = "landsatlook-stac", detail = "item not found")
    end
    item = JSON3.read(body)
    # L4/L5 correlate the green band, L7/L8/L9 the panchromatic (`process.py:77-89`).
    key = c.platform in ("L4", "L5") ? :green : :pan
    asset = get(item.assets, key, nothing)
    asset === nothing && return (; status = :missing, route = "landsatlook-stac",
                                 detail = "no $key asset")
    return (; status = :unauthorized, route = "usgs-m2m or requester-pays s3",
            detail = "$(key) band resolved; needs M2M download role or AWS requester-pays")
end

# Sentinel-1 and NISAR, from ASF via CMR. `~/.netrc` with `urs.earthdata.nasa.gov` credentials is
# what authenticates the download; CMR search itself is anonymous.
#
# The granule name to query is the job's input, not the product's. A burst-mosaic product is named
# after a *synthetic* SLC-style identifier that no archive holds — querying that returns nothing,
# while the OPERA burst IDs the job actually lists resolve one-for-one.
#
# `provider=ASF` is required, not a filter: CMR refuses an unconstrained cross-collection granule
# query outright rather than searching everything.
function resolve_asf(c::GoldenCase)
    query = c.reference[1]
    url = "https://cmr.earthdata.nasa.gov/search/granules.umm_json" *
          "?readable_granule_name=$(escape_uri(query))&page_size=5&provider=ASF"
    body = try
        String(take!(Downloads.download(url, IOBuffer(); timeout = 90)))
    catch
        return (; status = :missing, route = "cmr/asf", detail = "search failed")
    end
    r = JSON3.read(body)
    hits = get(r, :hits, 0)
    hits == 0 && return (; status = :missing, route = "cmr/asf", detail = "0 hits for $query")
    # An SLC name matches both its `-SLC` and `-METADATA` granules, so more than one hit is normal
    # and the count is reported rather than treated as ambiguity.
    n = length(c.reference)
    scope = n == 1 ? "1 scene" : "$n bursts, first resolved"
    return (; status = :ok, route = "cmr/asf + ~/.netrc",
            detail = "$scope; $hits CMR granule(s)")
end

escape_uri(s) = replace(s, " " => "%20")

function head_status(url)
    try
        r = Downloads.request(url; method = "HEAD", throw = false, timeout = 60)
        return r.status
    catch
        return 0
    end
end

# --- Entry point ----------------------------------------------------------------------------

function main(args)
    check_only = "--check" in args
    phase = nothing
    i = findfirst(==("--phase"), args)
    i === nothing || (phase = parse(Int, args[i + 1]))

    cs = phase === nothing ? cases() : cases(phase)
    println("$(length(cs)) case(s)", phase === nothing ? "" : " in phase $phase")

    if !aws_available()
        error("the `aws` CLI is required to fetch from the golden bucket")
    end

    if !check_only
        r = fetch_products(cs)
        println("products: $(r.got) fetched, $(r.skipped) already present")
        println("cache: $(golden_dir())")
    end

    println("\ninput resolution (no downloads):")
    for c in cs
        r = resolve_inputs(c)
        mark = r.status == :ok ? "ok  " : r.status == :unauthorized ? "auth" : "MISS"
        println("  $mark  $(rpad(c.platform, 9))  $(r.route)")
        println("        $(r.detail)")
    end
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
