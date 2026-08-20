# Agreement with OpenCV, over the whole fixture matrix.
#
# What this establishes and what it does not: OpenCV is the reference for the
# *primitives* where its semantics are the de-facto standard and are correct — the
# correlation surface, peak tie-breaking, pyramid upsampling. It is not the
# reference for the pipeline, which deliberately diverges from Python autoRIFT in
# several places (see REFERENCE.md). So these tests pin the math; correctness of the
# whole is established against synthetic ground truth in test/correlate.jl.
#
# Tolerance rather than bit-equality: OpenCV forms the correlation numerator with a
# blocked DFT and accumulates in Float32, while this implementation evaluates it
# directly in Float64 below a work threshold. Those give the same answer to Float32
# rounding, not to the last bit. What must match exactly is the *peak location*,
# since that is what becomes a displacement — a surface differing by 1e-6 is
# irrelevant, a peak differing by one sample is a whole pixel of velocity error.

using AutoRIFT: workspace, correlate!, peak_index

if !has_fixtures()
    @info "Fixture corpus absent; skipping OpenCV comparison"
else
    @testset "correlation surface vs OpenCV" begin
        worst_err = 0.0
        worst_case = ""
        npeaks = 0
        peak_mismatches = String[]

        # `lowcontrast` is excluded deliberately and tested separately below:
        # OpenCV fails on it outright, so comparing against its output there would
        # assert the wrong answer.
        for cs in (32, 64, 128), r in (6, 10, 25),
            dt in ("uint8", "float32"), variant in ("plain", "dc_offset")

            name = "matchtemplate/zncc_c$(cs)_r$(r)_$(dt)_$(variant)"
            isdir(joinpath(FIXTURE_DIR, name)) || continue
            f = fixture(name)
            T = dt == "uint8" ? UInt8 : Float32

            ws = workspace(T, cs, r)
            got = correlate!(ws, f.arrays.search, f.arrays.chip, r)
            expected = f.arrays.expected

            @test size(got) == size(expected)
            err = maximum(abs.(Float64.(got) .- Float64.(expected)))
            if err > worst_err
                worst_err, worst_case = err, name
            end

            # The load-bearing assertion.
            npeaks += 1
            peak_index(got) == peak_index(expected) ||
                push!(peak_mismatches, "$name: $(peak_index(got)) vs $(peak_index(expected))")
        end

        # Exact count, not a lower bound: 3 chip sizes x 3 radii x (2 dtypes for
        # `plain` + 1 for `dc_offset`, which is float32-only) = 27. Pinning it means
        # a fixture that stops being generated fails here rather than silently
        # shrinking the comparison.
        @test npeaks == 27
        if !isempty(peak_mismatches)
            @info "peak mismatches" peak_mismatches
        end
        @test isempty(peak_mismatches)

        # Float32 rounding across two different summation strategies.
        @test worst_err < 1e-4
        @info "worst surface deviation from OpenCV" worst_err worst_case
    end

    @testset "DC offset separates the measures" begin
        # ZNCC removes both means, so an additive offset in either image leaves the
        # surface unchanged. This is why v2.0.0 unified on it: the DC-sensitive
        # variant that older releases used for floating-point input needed a
        # chip-minimum subtraction hack to paper over exactly this.
        plain = fixture("matchtemplate/zncc_c32_r25_float32_plain")
        offset = fixture("matchtemplate/zncc_c32_r25_float32_dc_offset")

        ws = workspace(Float32, 32, 25)
        s1 = copy(correlate!(ws, plain.arrays.search, plain.arrays.chip, 25))
        s2 = copy(correlate!(ws, offset.arrays.search, offset.arrays.chip, 25))

        @test peak_index(s1) == peak_index(s2)
        # Not merely the same peak: the same surface, since the offset cancels
        # analytically rather than approximately.
        @test maximum(abs.(s1 .- s2)) < 1e-4
    end

    @testset "low contrast: better than OpenCV" begin
        # The case that justifies Float64 integral images, and the one place where
        # agreement with OpenCV would be the *wrong* test.
        #
        # These fixtures carry a signal of 0.001 on a base of 0.5. The window
        # variance is computed as `ΣW² - (ΣW)²/n`, a difference of two large nearly
        # equal quantities; at that contrast, in Float32, the cancellation leaves
        # nothing. OpenCV's guard then emits an all-zero surface — it does not find
        # a wrong peak, it declines to answer at all, and `minMaxLoc` on 2500 tied
        # zeros returns index (1,1).
        #
        # Ground truth is unambiguous here: each chip was cut from the exact centre
        # of its search window, so a correct correlator peaks at the centre with a
        # coefficient of 1. Accumulating the integral images in Float64 recovers
        # that. Asserted against truth rather than against OpenCV, and asserted
        # rather than merely noted, because "we handle a case the reference cannot"
        # is a claim that should fail loudly if it stops being true.
        for cs in (32, 64, 128), r in (6, 10, 25)
            name = "matchtemplate/zncc_c$(cs)_r$(r)_float32_lowcontrast"
            isdir(joinpath(FIXTURE_DIR, name)) || continue
            f = fixture(name)

            @test all(iszero, f.arrays.expected)   # OpenCV gave up

            ws = workspace(Float32, cs, r)
            got = correlate!(ws, f.arrays.search, f.arrays.chip, r)
            @test peak_index(got) == (r + 1, r + 1)
            @test maximum(got) ≈ 1.0f0 atol = 1e-3
            @test all(isfinite, got)
            # A generous bound, and deliberately so: this fixture set is at the edge of what
            # Float32 can represent, and the measured worst excursion beyond [-1, 1] is 9.2e-5 —
            # 92% of this tolerance. That is not slack to be tightened. The coefficient is formed
            # from a Float64 numerator over a Float64 denominator and then rounded to Float32, so
            # a value that should be exactly 1 can land a few ulp above it; at a contrast of
            # 0.001 on a base of 0.5, "a few ulp" is this large. The peak location, asserted
            # above, is the claim that actually matters and it is exact.
            @test all(-1.0001f0 .<= got .<= 1.0001f0)
        end
    end
end
