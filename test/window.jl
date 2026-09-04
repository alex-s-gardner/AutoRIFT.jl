using AutoRIFT: windowmax, windowmin, windowmean, windowmedian, windowmad,
                windowrange, count_agreeing

# Brute-force reference: the documented semantics, written as literally as possible.
# Every reduction is checked against this rather than against a hand-computed table,
# because the interesting cases are the ones involving NaN and window truncation at
# the borders, and those are tedious and error-prone to write out by hand.
#
# Window bias for even widths is LEFT (`left = w ÷ 2`), matching
# `scipy.ndimage.generic_filter` at its default origin and therefore the reference.
function brute_window(A, w, f)
    wx, wy = w isa Tuple ? (w[1], w[2]) : (w, w)
    wx = min(wx, size(A, 2))
    wy = min(wy, size(A, 1))
    nr, nc = size(A)
    lx, ly = wx ÷ 2, wy ÷ 2
    rx, ry = wx - 1 - lx, wy - 1 - ly
    out = fill(NaN32, nr, nc)
    for j in 1:nc, i in 1:nr
        vals = Float32[]
        for jj in max(j - lx, 1):min(j + rx, nc), ii in max(i - ly, 1):min(i + ry, nr)
            v = Float32(A[ii, jj])
            isnan(v) || push!(vals, v)
        end
        out[i, j] = isempty(vals) ? NaN32 : Float32(f(vals))
    end
    return out
end

function brute_count(A, w, tol)
    wx, wy = w isa Tuple ? (w[1], w[2]) : (w, w)
    wx = min(wx, size(A, 2))
    wy = min(wy, size(A, 1))
    nr, nc = size(A)
    lx, ly = wx ÷ 2, wy ÷ 2
    rx, ry = wx - 1 - lx, wy - 1 - ly
    out = zeros(Float32, nr, nc)
    for j in 1:nc, i in 1:nr
        c = Float32(A[i, j])
        isnan(c) && continue
        n = 0
        for jj in max(j - lx, 1):min(j + rx, nc), ii in max(i - ly, 1):min(i + ry, nr)
            abs(Float32(A[ii, jj]) - c) < Float32(tol) && (n += 1)
        end
        out[i, j] = Float32(n)
    end
    return out
end

nan_equal(a, b; atol = 2f-4) =
    size(a) == size(b) &&
    all(i -> (isnan(a[i]) && isnan(b[i])) || abs(a[i] - b[i]) <= atol, eachindex(a))

@testset "basic behaviour" begin
    a = Float32[1 5 2; 8 3 9; 4 7 6]
    @test windowmax(a, 3)[2, 2] == 9
    @test windowmin(a, 3)[2, 2] == 1
    @test windowmean(a, 3)[2, 2] ≈ sum(a) / 9
    @test windowmedian(a, 3)[2, 2] == 5
    @test windowrange(a, 3)[2, 2] == 8

    # A window of 1 is the identity.
    @test windowmax(a, 1) == a
    @test windowmedian(a, 1) == a

    # Output is always the input's shape, never shrunk. The reference used to shrink
    # for even windows; v2.0.0 changed that, and the change matters because these
    # results are resampled to a fixed target size.
    for w in (1, 2, 3, 4, 5, 12, 48, (3, 5))
        @test size(windowmax(a, w)) == size(a)
        @test size(windowmean(a, w)) == size(a)
        @test size(windowmedian(a, w)) == size(a)
    end
end

@testset "NaN is ignored, not propagated" begin
    # The property that disqualified `ImageMorphology.extreme_filter`, which
    # propagates NaN and would poison nearly every output on a displacement field.
    a = Float32[1 NaN 3; NaN 5 NaN; 7 NaN 9]
    @test windowmax(a, 3)[2, 2] == 9
    @test windowmin(a, 3)[2, 2] == 1
    @test windowmean(a, 3)[2, 2] ≈ (1 + 3 + 5 + 7 + 9) / 5
    @test windowmedian(a, 3)[2, 2] == 5
    @test count(isnan, windowmax(a, 3)) == 0

    # An all-NaN window is the one case that legitimately yields NaN.
    @test isnan(windowmax(fill(NaN32, 3, 3), 3)[2, 2])
    @test isnan(windowmean(fill(NaN32, 3, 3), 3)[2, 2])
    @test isnan(windowmedian(fill(NaN32, 3, 3), 3)[2, 2])

    # An isolated valid value is reported by every window that contains it.
    a = fill(NaN32, 5, 5)
    a[3, 3] = 42.0f0
    m = windowmax(a, 3)
    @test m[3, 3] == 42
    @test m[2, 2] == 42
    @test isnan(m[1, 1])     # outside the 3x3 neighbourhood of (3,3)
end

@testset "even-window bias is left" begin
    # `scipy.ndimage.generic_filter` uses `left = w ÷ 2` at its default origin, so the
    # reference does too. Getting this backwards shifts every even-window result by a
    # pixel, and the pyramid uses even windows at every level above the base — so this
    # is pinned directly rather than left to the brute-force sweep.
    a = Float32[1 2 3 4 5]
    @test windowmin(a, 2) == Float32[1 1 2 3 4]     # window {i-1, i}
    @test windowmax(a, 2) == Float32[1 2 3 4 5]
    @test windowmin(a, 4) == Float32[1 1 1 2 3]     # left margin 2, right margin 1
end

@testset "against brute force" begin
    # Randomised sweep over shapes, window sizes, and NaN densities. This is where
    # the border truncation and NaN interaction actually get covered; the cases above
    # pin specific conventions.
    fails = String[]
    for trial in 1:30
        rng = Random.MersenneTwister(trial)
        nr, nc = rand(rng, 3:22), rand(rng, 3:22)
        a = rand(rng, Float32, nr, nc)
        nanfrac = (trial % 4) * 0.25          # 0%, 25%, 50%, 75%
        a[rand(rng, nr * nc) .< nanfrac] .= NaN32

        for w in (1, 2, 3, 4, 5, 6, 12, 48, (3, 5), (6, 2))
            nan_equal(windowmax(a, w), brute_window(a, w, maximum)) ||
                push!(fails, "max w=$w trial=$trial")
            nan_equal(windowmin(a, w), brute_window(a, w, minimum)) ||
                push!(fails, "min w=$w trial=$trial")
            nan_equal(windowmean(a, w),
                      brute_window(a, w, v -> sum(Float64.(v)) / length(v))) ||
                push!(fails, "mean w=$w trial=$trial")
            nan_equal(windowrange(a, w),
                      brute_window(a, w, v -> maximum(v) - minimum(v))) ||
                push!(fails, "range w=$w trial=$trial")
        end

        # Median and MAD only ever see small windows in practice.
        for w in (1, 2, 3, 4, 5, (3, 5))
            nan_equal(windowmedian(a, w), brute_window(a, w, Statistics.median)) ||
                push!(fails, "median w=$w trial=$trial")
            nan_equal(windowmad(a, w), brute_window(a, w,
                      v -> (m = Statistics.median(v); Statistics.median(abs.(v .- m))))) ||
                push!(fails, "mad w=$w trial=$trial")
            for tol in (0.1f0, 0.5f0)
                nan_equal(count_agreeing(a, w, tol), brute_count(a, w, tol)) ||
                    push!(fails, "count w=$w tol=$tol trial=$trial")
            end
        end
    end
    isempty(fails) || @info "window mismatches" first(fails, 10)
    @test isempty(fails)
end

@testset "count_agreeing" begin
    # The agreement measure the outlier filter thresholds. A displacement supported by
    # its neighbours is kept; one that stands alone is rejected.
    a = Float32[1 1 1; 1 1 1; 1 1 9]
    c = count_agreeing(a, 3, 0.5)
    @test c[1, 1] == 4        # 2x2 in-bounds window, all equal
    @test c[2, 2] == 8        # 3x3 window, eight agree, the 9 does not
    @test c[3, 3] == 1        # the outlier agrees only with itself

    # NaN never agrees, and a NaN centre agrees with nothing.
    a = Float32[1 1 NaN; 1 1 1; NaN 1 1]
    c = count_agreeing(a, 3, 0.5)
    @test c[1, 3] == 0        # NaN centre
    @test c[2, 2] == 7        # nine neighbours minus two NaN
end

@testset "window size clamping" begin
    # A window wider than the array is clamped to its size, matching the reference.
    # Not hypothetical: the chip-size loop's coarse levels apply 48-wide dilations to grids
    # that may be only tens of points across.
    #
    # Clamping to `n` does *not* make every output the global reduction, because the
    # window stays left-biased: at position 1 of a 5-element axis an `n`-wide window
    # covers indices 1 through 3, not 1 through 5. So the result is compared against
    # the brute-force reference rather than against a global fill.
    a = rand(Float32, 5, 5)
    @test nan_equal(windowmax(a, 48), brute_window(a, 48, maximum))
    @test nan_equal(windowmax(a, 48), windowmax(a, 5))     # 48 clamps to 5
    @test nan_equal(windowmean(a, 100), brute_window(a, 100,
                    v -> sum(Float64.(v)) / length(v)))
    @test size(windowmedian(a, 48)) == (5, 5)

    # The centre of an odd-sized array does see everything, which is the case where a
    # global reduction is the right expectation.
    @test windowmax(a, 9)[3, 3] == maximum(a)

    @test_throws ArgumentError windowmax(a, 0)
    @test_throws ArgumentError windowmax(a, (3, 0))
end

@testset "cost is flat in window width" begin
    # The point of the monotone deque: the extrema must not get dramatically more
    # expensive as the window grows, because the chip-size loop's dilations reach width 48.
    # An O(w^2) implementation would be ~250x slower at 48 than at 3; this asserts a
    # generous bound that only a genuine complexity regression would breach.
    #
    # The only timing-dependent assertion in the suite, and it fired spuriously twice before being
    # made robust. A *minimum* over several trials is the right statistic — noise only ever adds
    # time, so the minimum converges while a single sample wanders — and 384² x 10 iterations puts
    # each trial in the tens of milliseconds, where a millisecond of interference no longer matters.
    #
    # The bound stays loose on purpose: the claim is complexity, not constant factors, so 10x leaves
    # two orders of magnitude of headroom and still fails instantly if the deque is ever replaced by
    # a naive scan.
    a = rand(Float32, 384, 384)
    windowmax(a, 3); windowmax(a, 48)          # compile both widths
    bench(w) = minimum(@elapsed(for _ in 1:10; windowmax(a, w); end) for _ in 1:5)
    t3, t48 = bench(3), bench(48)
    @test t48 < 10 * t3
end

@testset "median selection agrees with a sort at every size" begin
    # Two regimes meet in `_select_median!`: an insertion sort at or below `SELECT_THRESHOLD`
    # and a quickselect above it, and the boundary is where a mistake would hide. Checked
    # against `Statistics.median` on both sides of it.
    rng = MersenneTwister(7)
    for n in (1, 2, 3, 25, 63, 64, 65, 81, 289, 300)
        v = rand(rng, Float32, n)
        buf = copy(v)
        @test AutoRIFT._select_median!(buf, n) ≈ Float32(median(v)) atol = 1e-5
    end

    # Quickselect's worst case is a sorted or constant input, which a window over uniform ice or
    # across a shear margin produces routinely. Median-of-three pivoting is what bounds it, and
    # an unpivoted implementation would still pass the random cases above.
    for n in (65, 121, 289)
        for v in (fill(0.5f0, n), Float32.(collect(1:n)), Float32.(collect(n:-1:1)))
            buf = copy(v)
            @test AutoRIFT._select_median!(buf, n) ≈ Float32(median(v)) atol = 1e-5
        end
    end

    # Randomized, since a hand-picked set cannot cover partition edge cases.
    worst = 0.0f0
    for _ in 1:2000
        n = rand(rng, 1:300)
        v = rand(rng, Float32, n)
        buf = copy(v)
        worst = max(worst, abs(AutoRIFT._select_median!(buf, n) - Float32(median(v))))
    end
    @test worst < 1.0f-5
end
