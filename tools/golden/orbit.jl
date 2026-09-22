# The scan geometry from the orbit, against the angles the reference logged for the same scenes.
#
#   julia --project=tools/golden tools/golden/orbit.jl
#
# `_fft_filter` recovers the along- and cross-track directions from the shape of the valid-data region
# (`autoRIFT.py:153-206`); the scene's `_ANG.txt` carries the trajectory they follow from. This prints
# both for every scene of the three Landsat 4/5 pairs, which is the comparison `tools/golden/README.md`
# registers and gate `5.orbit` reads.
#
# Requires `AWS_PROFILE` to name credentials that can pay for `usgs-landsat`, since an `_ANG.txt` is
# reachable only over requester-pays S3 — the STAC item's `https` href redirects to a landing page.

include("e2e.jl")

const L45 = ("LT05_L1TP_060018", "LT04_L1TP_063018", "LT05_L1GS_001013")

function main()
    cache = joinpath(homedir(), "data", "autorift", "tests", "golden_tests", "angcache")
    worst_cross = 0.0
    worst_skew = 0.0
    gaps = Float64[]
    @printf("%-46s %-6s %17s %17s %14s\n", "scene", "epsg", "along (orbit/ref)",
            "cross (orbit/ref)", "axes apart")
    for name in L45
        c = only(cases(name))
        for r in orbit_angle_check(c, resolve_run(c), cache)
            @printf("%-46s %-6d %+7.3f %+7.2f   %+7.3f %+7.2f   %6.2f %6.2f\n",
                    r.name[1:min(end, 46)], r.epsg, r.along, r.ref_along, r.cross, r.ref_cross,
                    r.ours_apart, r.ref_apart)
            worst_cross = max(worst_cross, abs(r.d_cross))
            worst_skew = max(worst_skew, abs(r.ours_apart - 90))
            push!(gaps, r.d_along)
        end
    end
    # The line gate `5.orbit` parses. Cross-track agreement and perpendicularity are the properties the
    # derivation must have; the along-track gap is the reference's own error, so it is reported.
    @printf("SUMMARY scenes %d cross %.4f skew %.3e along_high %.3f %.3f\n",
            length(gaps), worst_cross, worst_skew, -maximum(gaps), -minimum(gaps))
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
