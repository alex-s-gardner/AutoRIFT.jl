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
#
# **The CMR hit and the credential are two separate questions, and this reports both.** CMR search is
# anonymous, so a granule resolves whether or not `~/.netrc` can read it: reporting `:ok` off the hit
# alone claims a route that has never been exercised, and the first thing to exercise it is then a
# container run that dies on `401 Unauthorized` after reaching the driver. `urs_status` is what
# separates "the archive does not have this" from "we may not read it".
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
    urs = urs_status()
    urs === :ok || return (; status = :unauthorized, route = "cmr/asf + ~/.netrc",
                           detail = "$scope; $hits CMR granule(s); Earthdata $urs")
    return (; status = :ok, route = "cmr/asf + ~/.netrc",
            detail = "$scope; $hits CMR granule(s)")
end

"""
    urs_status() -> Symbol

Whether `~/.netrc`'s `urs.earthdata.nasa.gov` credential authenticates: `:ok`, `:rejected`,
`:absent`, `:ambiguous`, or `:unreachable`.

Asked against `urs.earthdata.nasa.gov` itself rather than against a data URL, because an ASF download
is a redirect chain and a 401 anywhere along it is ambiguous between a bad credential and a granule
this account is not approved for. The answer is cached: every radar case shares one credential, and
`--check` over ten cases should not be ten login attempts.

**A duplicate `machine` entry is reported rather than probed, because which one wins is not this
check's to decide.** libcurl takes the first matching entry and Python's `netrc` module — which is
what `hyp3lib.fetch` uses inside the container — takes the last, so two entries for one host with
different passwords mean this check and the container can consult different credentials. A `:ok` here
would then promise a route the container cannot take, and the failure surfaces as
`401 Unauthorized` half an hour into a run, after the driver has already downloaded a SAFE. Neither
entry is knowably the live one from here, so the honest status is `:ambiguous`.
"""
const _URS = Ref{Union{Symbol,Nothing}}(nothing)
function urs_status()
    _URS[] === nothing || return _URS[]
    return _URS[] = _probe_urs()
end

const URS_HOST = "urs.earthdata.nasa.gov"

function _probe_urs()
    path = joinpath(homedir(), ".netrc")
    isfile(path) || return :absent
    netrc_host_count(path, URS_HOST) > 1 && return :ambiguous
    # **Retried, because one 401 does not distinguish a bad credential from a throttled endpoint.**
    # URS answers 401 when it is rate-limiting as well as when it rejects, and this probe is cheap
    # enough to run repeatedly while debugging — which is exactly how a working credential gets
    # throttled. A single reading then condemns it, and the next hour is spent looking for a
    # credential problem that does not exist. Three attempts, spaced, and a `:rejected` only when
    # every one agrees.
    #
    # `Downloads` reads `~/.netrc` through libcurl, so no credential is handled here — the status code
    # is the whole answer and nothing secret enters this process.
    last = :unreachable
    for attempt in 1:3
        attempt > 1 && sleep(2.0 * attempt)
        r = try
            Downloads.request("https://$URS_HOST/api/users/tokens";
                              method = "GET", throw = false, timeout = 60)
        catch
            last = :unreachable
            continue
        end
        r.status == 200 && return :ok
        last = r.status in (401, 403) ? :rejected : :unreachable
    end
    return last
end

"""
    netrc_host_count(path, host) -> Int

How many `machine <host>` entries `path` declares.

Counts tokens rather than lines: `.netrc` is whitespace-delimited, so one entry may span lines or
share one. Only the machine names are inspected — no `login` or `password` token is read, so a
credential cannot leak through this function or its caller's error messages.
"""
function netrc_host_count(path::AbstractString, host::AbstractString)
    toks = split(read(path, String))
    return count(i -> toks[i] == "machine" && i < length(toks) && toks[i + 1] == host,
                 eachindex(toks))
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
