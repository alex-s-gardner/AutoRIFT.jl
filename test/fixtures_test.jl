# Verify the fixture corpus loads correctly.
#
# This is the M1 exit criterion: unless a reference array generated in Python can
# be asserted against in Julia, none of the later milestones can be verified. The
# risk being tested is the C-order/column-major transposition — a fixture read
# with the wrong memory order still has the right shape and plausible values, and
# would silently invert every comparison that follows.

if !has_fixtures()
    @info "Fixture corpus absent; skipping. Generate with " *
          "`mamba run -n autorift-cv python tools/python_ref/gen_fixtures.py`"
else
    @testset "manifest" begin
        m = JSON3.read(read(MANIFEST_PATH, String))
        @test m.total_cases > 300
        @test haskey(m.groups, :matchtemplate)
        # Recorded so a fixture regenerated under a different OpenCV can be
        # identified as such rather than debugged as a code change.
        @test !isempty(m.opencv_version)
        @test !isempty(m.numpy_version)
    end

    @testset "orientation" begin
        # The transposition test. `derivkernels` is ideal for it: the x- and
        # y-derivative kernels of the same order are transposes of each other, so
        # a loader that got the memory order wrong would return them swapped, and
        # the shapes are non-square, so the error cannot hide.
        d10 = fixture("derivkernels/d10_w5")
        d01 = fixture("derivkernels/d01_w5")
        @test size(d10.arrays.kx) == (5, 1)
        @test size(d10.arrays.ky) == (5, 1)
        # getDerivKernels returns (kx, ky) as column vectors; the first-derivative
        # factor is antisymmetric and the smoothing factor is symmetric, and they
        # exchange roles between the d/dx and d/dy cases.
        @test vec(d10.arrays.kx) == -reverse(vec(d10.arrays.kx))
        @test vec(d10.arrays.ky) == reverse(vec(d10.arrays.ky))
        @test vec(d01.arrays.kx) == reverse(vec(d01.arrays.kx))
        @test vec(d01.arrays.ky) == -reverse(vec(d01.arrays.ky))
        # Unnormalized: OpenCV's derivative kernels for width > 3 are
        # binomial-difference kernels whose magnitude grows with width, unlike
        # Julia's normalized `Kernel.sobel`. Asserted so a port that reaches for
        # the normalized form fails here rather than producing a scaled result.
        @test maximum(abs, d10.arrays.ky) == 6.0f0
    end

    @testset "asymmetric array orientation" begin
        # A non-square case, where a transposed read is a DimensionMismatch rather
        # than a wrong-but-plausible array.
        f = fixture("resize/bilinear_64_0p5")
        @test size(f.arrays.src) == (64, 64)
        @test size(f.arrays.expected) == (32, 32)

        f = fixture("matchtemplate/zncc_c32_r25_uint8_plain")
        @test size(f.arrays.chip) == (32, 32)
        @test size(f.arrays.search) == (81, 81)
        # Independent confirmation of the window geometry: OpenCV's own output
        # size for a 32-pixel chip in an 81-pixel window is 50x50, which is
        # exactly `2 * search_radius`. This is the arithmetic `search_bounds`
        # implements, cross-checked against the reference rather than against my
        # reading of it.
        @test size(f.arrays.expected) == (50, 50)
        @test size(f.arrays.expected) == (2 * 25, 2 * 25)
    end

    @testset "peak tie-breaking convention" begin
        # OpenCV's `minMaxLoc` scans row-major and keeps the first strict maximum.
        # Julia's `argmax` scans column-major. On a plateau these disagree, and
        # the resulting displacement bias is invisible because both answers look
        # reasonable. The fixture records the answer in both spellings so a port
        # cannot quietly transpose it.
        f = fixture("peak/plateau")
        row, col = f.params.row_col_0based
        @test (row, col) == (2, 2)          # top-left of the 3x3 plateau
        @test f.arrays.surface[row + 1, col + 1] == 1.0f0

        # A tie along one row must resolve to the leftmost column...
        f = fixture("peak/tie_in_row")
        r, c = f.params.row_col_0based
        @test (r, c) == (3, 1)
        # ...and a tie down one column to the topmost row. Together these pin both
        # axes of the scan order.
        f = fixture("peak/tie_in_column")
        r, c = f.params.row_col_0based
        @test (r, c) == (1, 3)
    end

    @testset "value round-trip" begin
        # Values must survive the round-trip bit-for-bit, not approximately: these
        # fixtures are the reference for exact comparisons, so a lossy read would
        # quietly weaken every assertion built on them.
        f = fixture("filter2d/box_w5_constant_float32")
        @test eltype(f.arrays.src) === Float32
        @test eltype(f.arrays.expected) === Float32
        @test all(isfinite, f.arrays.expected)
        # A box filter cannot exceed the input range in the interior.
        interior = f.arrays.expected[4:(end - 3), 4:(end - 3)]
        @test maximum(interior) <= maximum(f.arrays.src) + 1f-6

        f = fixture("filter2d/box_w5_constant_uint8")
        @test eltype(f.arrays.src) === UInt8
        # OpenCV keeps the input depth when ddepth = -1, so a UInt8 input gives a
        # UInt8 output — accumulated and rounded in 8 bits, not promoted.
        @test eltype(f.arrays.expected) === UInt8

        # ZNCC is bounded on [-1, 1] by construction; a violation would mean the
        # normalisation is wrong.
        f = fixture("matchtemplate/zncc_c32_r25_float32_plain")
        @test all(-1.0001f0 .<= f.arrays.expected .<= 1.0001f0)
        # The chip was cut from the search window at exactly the centre offset, so
        # the peak must be a near-perfect match at the centre of the surface.
        @test maximum(f.arrays.expected) > 0.99f0
    end

    @testset "DC offset separates the similarity measures" begin
        # The reason ZNCC is the reference measure. Adding a constant to the search
        # window leaves ZNCC unchanged but shifts NCC's peak, which is why earlier
        # autoRIFT releases needed a chip-minimum subtraction hack for float
        # input. Pinned here so the two measures cannot be conflated.
        plain = fixture("matchtemplate/zncc_c32_r25_float32_plain")
        offset = fixture("matchtemplate/zncc_c32_r25_float32_dc_offset")
        @test maximum(offset.arrays.expected) > 0.99f0
        @test argmax(plain.arrays.expected) == argmax(offset.arrays.expected)

        # NCC's peak value degrades under the same offset, since the un-removed
        # mean now dominates the normalisation.
        ncc_plain = fixture("matchtemplate/ncc_c32_r25_float32_plain")
        ncc_offset = fixture("matchtemplate/ncc_c32_r25_float32_dc_offset")
        @test maximum(ncc_offset.arrays.expected) != maximum(ncc_plain.arrays.expected)
    end

    @testset "pyrUp brightness scaling" begin
        # pyrUp injects zeros and convolves with a 5-tap kernel scaled by 4, so
        # brightness is preserved rather than quartered. A port that omits the
        # factor of 4 produces a surface a quarter as bright — whose argmax is
        # unchanged, so it would pass a peak-location test while being wrong.
        f = fixture("pyrup/equal_step_x2")
        @test size(f.arrays.expected) == (10, 10)
        @test all(≈(0.5f0), f.arrays.expected[3:(end - 2), 3:(end - 2)])

        # A single delta spreads into the kernel's footprint, and its peak stays
        # at the delta's upsampled position.
        f = fixture("pyrup/delta_step_x2")
        @test maximum(f.arrays.expected) > 0
        @test sum(f.arrays.expected) > 0
    end
end
