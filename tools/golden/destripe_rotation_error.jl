# What the reference's biased scan angle (CORRECTNESS.md item 5) actually costs: run `Destripe` on
# the same Wallis-filtered native scene with the reference's logged angles and with the orbit's, and
# compare the two destriped fields pixel for pixel.
#
#   AWS_PROFILE=itslive julia --project=tools/golden tools/golden/destripe_rotation_error.jl
#
# A degree difference in the scan angle is not, by itself, a statement about anything a correlator
# sees. This measures the thing that matters: the destriped pixels, and whether either angle flips the
# band-reject's fire/decline decision — `_fft_filter`'s branch is a ratio test against `ratio = 2`,
# and `dev/GATES.md` records at least one L4/5 scene where the reference's own ratio clears its
# threshold by only 3.2%, which is exactly where a few degrees of bias is most likely to matter.
#
# **Declined vs fired is read from the public output, not from `AutoRIFT`'s internals.** `destripe`
# returns the clamped-but-unfiltered image on decline and a band-rejected field on fire
# (`src/preprocess.jl:1267-1332`); reproducing the clamp step here and comparing against it tells the
# two apart without reaching into `_reject_band`'s private sums, so this stays a check of the public
# contract rather than a second implementation of the private one.

include("e2e.jl")

using AutoRIFT: destripe, preprocess, Wallis, Destripe
using Statistics, Printf

const L45 = ("LT05_L1TP_060018", "LT04_L1TP_063018", "LT05_L1GS_001013")

"""The Wallis-filtered field `destripe` is handed, matching `native_scene`'s own order of operations."""
function wallis_field(path::AbstractString)
    ds = ArchGDAL.read(path)
    img = Float32.(permutedims(ArchGDAL.read(ArchGDAL.getband(ds, 1))))
    valid = img .!= 0
    w, _ = preprocess(img, valid, Wallis(5, 0.0))
    w[.!valid] .= 0.0f0
    return w, valid
end

"""Whether `destripe` declined: its output equals the plain clamp-and-zero step, cell for cell."""
function declined(out::AbstractMatrix, w::AbstractMatrix, valid::AbstractMatrix, m::Destripe)
    clamped = Float32.(Base.clamp.(w, Float32(-m.clamp), Float32(m.clamp)))
    clamped[.!valid] .= 0.0f0
    return out == clamped
end

function main()
    cache = joinpath(homedir(), "data", "autorift", "tests", "golden_tests", "angcache")
    for case_name in L45
        c = only(cases(case_name))
        early, late = acquisition_order(c)
        paths = Dict(early => scene_path(c, :reference), late => scene_path(c, :secondary))
        for r in orbit_angle_check(c, resolve_run(c), cache)
            path = paths[r.name]
            w, valid = wallis_field(path)

            m_ref = Destripe(; along_track = r.ref_along, cross_track = r.ref_cross)
            m_orb = Destripe(; along_track = r.along, cross_track = r.cross)
            out_ref = destripe(w, valid, m_ref)
            out_orb = destripe(w, valid, m_orb)

            dec_ref = declined(out_ref, w, valid, m_ref)
            dec_orb = declined(out_orb, w, valid, m_orb)

            d = Float64.(out_ref[valid]) .- Float64.(out_orb[valid])
            rms = sqrt(mean(abs2, d))
            p95 = quantile(abs.(d), 0.95)

            flip = dec_ref != dec_orb
            @printf("%-46s  decline(ref/orb) %-5s/%-5s %s  rms %.4f  p95 %.4f  max %.4f\n",
                    r.name[1:min(end, 46)], dec_ref, dec_orb, flip ? "FLIP" : "    ",
                    rms, p95, maximum(abs.(d)))
        end
    end
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
