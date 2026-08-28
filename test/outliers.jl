using AutoRIFT: GardnerFilter, NoOutlierFilter, outlier_filter, reject_outliers, params

# A coherent field with planted outliers, which is the situation the filter exists for:
# real motion is spatially smooth, false correlation peaks are arbitrary vectors that
# agree with nothing around them.
function coherent_field(n; seed = 0, outliers = ())
    dx = fill(3.0f0, n, n)
    dy = fill(-2.0f0, n, n)
    # Gentle spatial variation, so the field is smooth but not constant -- a constant
    # field would make the MAD exactly zero everywhere and only exercise the floor.
    for j in 1:n, i in 1:n
        dx[i, j] += 0.05f0 * i
        dy[i, j] -= 0.05f0 * j
    end
    for (i, j, vx, vy) in outliers
        dx[i, j] = vx
        dy[i, j] = vy
    end
    return dx, dy
end

@testset "construction and validation" begin
    f = outlier_filter()
    @test f isa GardnerFilter
    @test f.window == 5
    @test f.iterations == 3
    @test f.min_agree_fraction ≈ 8 / 25

    @test_throws "must be odd" outlier_filter(; window = 4)
    @test_throws "must be >= 3" outlier_filter(; window = 1)
    @test_throws "must be >= 1" outlier_filter(; iterations = 0)
    @test_throws "in [0, 1]" outlier_filter(; min_agree_fraction = 1.5)
    @test_throws "in [0, 1]" outlier_filter(; agree_tolerance = 0)
    @test_throws "in [0, 1]" outlier_filter(; agree_tolerance = 1.5)
    @test_throws "must be positive" outlier_filter(; mad_scale = -1)
end

@testset "the method is swappable" begin
    # Which consistency test to apply is a choice, so it dispatches on an `OutlierMethod`
    # rather than being wired in. That is what lets a second filter — Westerweel & Scarano's,
    # say — be a new type and one method instead of an edit to the pyramid.
    n = 20
    dx, dy = coherent_field(n)
    rx = fill(25, n, n)
    ry = fill(25, n, n)
    valid = trues(n, n)
    dx[10, 10] = 20.0f0            # a wild vector the default filter should drop
    dy[10, 10] = -18.0f0

    kept = reject_outliers(dx, dy, rx, ry, valid, 64, GardnerFilter())
    @test !kept[10, 10]

    # `:none` keeps everything, which is how you tell the correlator's failures from the
    # filter's rejections.
    all_kept = reject_outliers(dx, dy, rx, ry, valid, 64, NoOutlierFilter())
    @test all(all_kept)
    @test all_kept !== valid        # a fresh mask; callers may mutate it

    # A point never measured stays out regardless of method: `valid` is respected, not
    # overridden.
    part = copy(valid)
    part[3, 3] = false
    @test !reject_outliers(dx, dy, rx, ry, part, 64, NoOutlierFilter())[3, 3]

    # `window` is the one query the scheduler makes of a method: how much neighbourhood it
    # needs, so a grid too small to supply one can skip filtering rather than reject on no
    # evidence. Zero means "needs none".
    @test AutoRIFT.window(GardnerFilter(; window = 7)) == 7
    @test AutoRIFT.window(NoOutlierFilter()) == 0

    # `relax` returns a variant suited to the decimated coarse grid, and what that means is the
    # method's own business — a method with a universal threshold may return itself.
    @test AutoRIFT.relax(GardnerFilter(; iterations = 3)).iterations == 2
    @test AutoRIFT.relax(GardnerFilter(; iterations = 1)).iterations == 1   # floored at one
    @test AutoRIFT.relax(NoOutlierFilter()) === NoOutlierFilter()
    # Everything but the iteration count survives relaxation.
    r = AutoRIFT.relax(GardnerFilter(; window = 7, mad_scale = 2.5))
    @test r.window == 7
    @test r.mad_scale == 2.5
end

@testset "selected by keyword" begin
    # The Symbol spelling, and the instance escape hatch, both reaching `Params`.
    @test AutoRIFT.params().outliers isa GardnerFilter
    @test AutoRIFT.params(; outliers = :none).outliers isa NoOutlierFilter
    @test AutoRIFT.params(; outliers = GardnerFilter(; window = 7)).outliers.window == 7

    # The loose keywords are the default filter's own parameters, forwarded.
    @test AutoRIFT.params(; outlier_window = 7).outliers.window == 7
    @test AutoRIFT.params(; mad_scale = 2.0).outliers.mad_scale == 2.0
    # Unset keywords leave the method's defaults alone rather than restating them.
    @test AutoRIFT.params(; outlier_window = 7).outliers.iterations == 3

    @test_throws "not recognised" AutoRIFT.params(; outliers = :westerweel)
    @test_throws "must be a Symbol or an `OutlierMethod`" AutoRIFT.params(; outliers = 5)

    # Passing both a method and a loose parameter is a contradiction, not a merge: the method
    # already carries its own, so the keyword would be silently dead. Reported against what the
    # caller actually wrote, whether that was an instance or a Symbol.
    @test_throws "GardnerFilter instance" AutoRIFT.params(;
        outliers = GardnerFilter(), mad_scale = 2.0)
    @test_throws "`:none`, which takes no parameters" AutoRIFT.params(;
        outliers = :none, outlier_window = 7)

    # `Params` carries the method as a type parameter, so it must stay concrete — the
    # correlation kernels specialize on it, and an abstract field here would make the whole
    # pipeline type-unstable.
    @test isconcretetype(typeof(AutoRIFT.params()))
    @test isconcretetype(typeof(AutoRIFT.params(; outliers = :none)))

    # The all-default path must not pay for keyword forwarding. `params` runs once per image
    # pair across tens of millions of them, and the method table is a `Dict{Symbol,Any}`, so
    # splatting a generator into a runtime-valued constructor costs ~700 ns and 8 allocations
    # even when it forwards nothing. Asserted as allocations, which is the deterministic part.
    AutoRIFT.params()                       # compile
    @test @allocated(AutoRIFT.params()) <= @allocated(AutoRIFT.params(; mad_scale = 2.0))
end

@testset "a coherent field survives" begin
    # The first thing to get right: the filter must not reject good data. A smoothly
    # varying field is what real ice motion looks like, and if this fails the filter is
    # worse than useless.
    n = 20
    dx, dy = coherent_field(n)
    rx = fill(25, n, n)
    ry = fill(25, n, n)
    valid = trues(n, n)
    keep = reject_outliers(dx, dy, rx, ry, valid, 64, outlier_filter())

    # Interior points have full neighbourhoods and must all survive.
    @test all(keep[4:(n - 3), 4:(n - 3)])
    # Overall survival should be high; edge points have truncated neighbourhoods and
    # are held to the same absolute agreement count, so some loss there is expected and
    # deliberate.
    @test count(keep) / length(keep) > 0.6
end

@testset "isolated outliers are rejected" begin
    n = 20
    # Wild vectors, far outside the search radius in normalized terms.
    spikes = ((6, 6, 20.0f0, 20.0f0), (12, 9, -22.0f0, 18.0f0), (15, 15, 24.0f0, -24.0f0))
    dx, dy = coherent_field(n; outliers = spikes)
    rx = fill(25, n, n)
    ry = fill(25, n, n)
    keep = reject_outliers(dx, dy, rx, ry, trues(n, n), 64, outlier_filter())

    for (i, j, _, _) in spikes
        @test !keep[i, j]
    end
    # And the coherent interior is still kept: rejecting outliers must not cascade into
    # rejecting their neighbours.
    @test keep[3, 3]
    @test keep[10, 10]
end

@testset "rejection is monotone" begin
    # A point once rejected stays rejected, which is what makes iterating meaningful:
    # removing one outlier can expose its neighbours as outliers, but the process
    # cannot oscillate.
    n = 16
    dx, dy = coherent_field(n; outliers = ((8, 8, 30.0f0, 30.0f0),))
    rx = fill(25, n, n)
    ry = fill(25, n, n)
    prev = trues(n, n)
    for iters in 1:4
        keep = reject_outliers(dx, dy, rx, ry, trues(n, n), 64,
                               outlier_filter(; iterations = iters))
        # More iterations may only remove points, never add them back.
        @test all(i -> !keep[i] || prev[i], eachindex(keep))
        prev = keep
    end
end

@testset "already-invalid points stay invalid" begin
    n = 12
    dx, dy = coherent_field(n)
    rx = fill(25, n, n)
    ry = fill(25, n, n)
    valid = trues(n, n)
    valid[5, 5] = false
    valid[6, 7] = false
    keep = reject_outliers(dx, dy, rx, ry, valid, 64, outlier_filter())
    @test !keep[5, 5]
    @test !keep[6, 7]
end

@testset "unsearched points are excluded" begin
    # A zero search radius means the point was never searched, so there is no
    # displacement to judge. Dividing by it would give Inf; the code substitutes NaN,
    # which the window reductions already skip.
    n = 12
    dx, dy = coherent_field(n)
    rx = fill(25, n, n)
    ry = fill(25, n, n)
    rx[3, 3] = 0
    ry[8, 8] = 0
    keep = reject_outliers(dx, dy, rx, ry, trues(n, n), 64, outlier_filter())
    @test !keep[3, 3]
    @test !keep[8, 8]
    @test all(isfinite, dx)     # inputs untouched
end

@testset "normalization by search radius" begin
    # Where normalization does and does not change the verdict, since the two stages
    # differ and the distinction is easy to get backwards.
    #
    # Stage 1 compares against `agree_tolerance`, a fixed fraction of the search
    # radius, so a deviation of a given size in pixels crosses that threshold at one
    # radius and not another. The agreement count is where the radius dependence
    # lives, and it is dramatic: a 3-pixel offset is 0.50 in normalized units at
    # radius 6 (outside the 0.2 tolerance, so the block agrees only with itself) and
    # 0.12 at radius 25 (inside it, so the block agrees with the whole field).
    n = 16
    dx = fill(3.0f0, n, n)
    dx[7:9, 7:9] .= 6.0f0                       # a 3x3 block offset by 3 px
    tol = Float32(outlier_filter().agree_tolerance)

    narrow = AutoRIFT.count_agreeing(dx ./ 6, 5, tol)
    wide = AutoRIFT.count_agreeing(dx ./ 25, 5, tol)
    @test narrow[8, 8] == 9                      # agrees only within its own block
    @test wide[8, 8] == 25                       # agrees with everything in the window

    # Stage 2, by contrast, is radius-*invariant* by construction, and it is worth
    # stating so that the normalization is not mistaken for doing more than it does.
    # Its tolerance is `mad_scale` times the neighbourhood MAD; both the deviation and
    # the MAD scale as 1/radius, so their ratio does not depend on the radius at all.
    # Only the quantization floor introduces a radius dependence there, which the
    # next testset covers.
    for r in (6, 25, 50)
        nx = dx ./ r
        med = AutoRIFT.windowmedian(nx, 5)
        mad = AutoRIFT.windowmad(nx, 5)
        # The ratio of deviation to MAD is the same at every radius.
        @test abs(nx[8, 8] - med[8, 8]) / mad[8, 8] ≈
              abs(dx[8, 8] / 25 - AutoRIFT.windowmedian(dx ./ 25, 5)[8, 8]) /
              AutoRIFT.windowmad(dx ./ 25, 5)[8, 8] rtol = 1e-3
    end
end

@testset "quantization floor protects uniform neighbourhoods" begin
    # In a perfectly uniform neighbourhood the MAD is zero, so `mad_scale * MAD` is
    # zero and any deviation at all -- including one of a single subpixel quantization
    # step -- would be rejected. The floor of twice the quantization step prevents the
    # most coherent parts of the field from being culled the most aggressively.
    n = 14
    r = 25
    up = 64
    step = 1 / up                      # one subpixel step, in pixels
    dx = fill(3.0f0, n, n)
    dy = fill(-2.0f0, n, n)
    dx[7, 7] += Float32(step)          # differs by exactly one quantization step
    keep = reject_outliers(dx, dy, fill(r, n, n), fill(r, n, n), trues(n, n), up,
                           outlier_filter())
    @test keep[7, 7]

    # A deviation far beyond the floor is still caught, so the floor is not simply
    # disabling the stage.
    dx[7, 7] = 3.0f0 + 5.0f0
    keep = reject_outliers(dx, dy, fill(r, n, n), fill(r, n, n), trues(n, n), up,
                           outlier_filter())
    @test !keep[7, 7]
end

@testset "inputs are not modified" begin
    # The reference mutates its arguments and relies on every caller passing a copy.
    # Copying inside instead makes the function safe to call directly, which is worth
    # asserting because the failure mode is silent corruption of the caller's field.
    n = 10
    dx, dy = coherent_field(n; outliers = ((5, 5, 30.0f0, 30.0f0),))
    dx0, dy0 = copy(dx), copy(dy)
    rx, ry = fill(25, n, n), fill(25, n, n)
    valid = trues(n, n)
    reject_outliers(dx, dy, rx, ry, valid, 64, outlier_filter())
    @test dx == dx0
    @test dy == dy0
    @test all(valid)
    @test all(==(25), rx)
end

@testset "shape mismatch is an error" begin
    @test_throws DimensionMismatch reject_outliers(
        zeros(Float32, 5, 5), zeros(Float32, 4, 5), fill(25, 5, 5), fill(25, 5, 5),
        trues(5, 5), 64, outlier_filter())
    @test_throws DimensionMismatch reject_outliers(
        zeros(Float32, 5, 5), zeros(Float32, 5, 5), fill(25, 5, 5), fill(25, 5, 5),
        trues(4, 5), 64, outlier_filter())
end

@testset "rescale matches the reference's grid scaling" begin
    # `autoRIFT.py:484-505` derives the filter's neighbourhood from the chip-size-to-grid-spacing
    # ratio rather than using the configured values directly. The arithmetic is asserted against
    # the reference's own expressions: window `(w-1)*ratio+1`, fraction `frac*(1-ov)+ov^2` for
    # `ov = 1 - stride/ratio`.
    f = outlier_filter()                      # window 5, min_agree_fraction 8/25
    @test AutoRIFT.rescale(f, 1) === f        # a grid no finer than its chips is untouched
    @test AutoRIFT.rescale(f, 1, 4) === f

    r2 = AutoRIFT.rescale(f, 2)
    @test r2.window == 9                      # (5-1)*2+1
    @test r2.min_agree_fraction ≈ 8 / 25 * 0.5 + 0.25    # overlap 1 - 1/2
    r4 = AutoRIFT.rescale(f, 4)
    @test r4.window == 17
    @test r4.min_agree_fraction ≈ 8 / 25 * 0.25 + 0.5625  # overlap 1 - 1/4

    # A stride at least the oversample ratio leaves no overlap, so the fraction is unchanged
    # while the window still widens — decimated neighbours are far apart but each still stands
    # for a chip's worth of ground.
    rc = AutoRIFT.rescale(f, 2, 4)
    @test rc.window == 9
    @test rc.min_agree_fraction ≈ 8 / 25

    # Untouched parameters stay untouched, and the method's own contract is preserved.
    @test r2.iterations == f.iterations
    @test r2.agree_tolerance == f.agree_tolerance
    @test r2.mad_scale == f.mad_scale
    @test AutoRIFT.rescale(NoOutlierFilter(), 4) === NoOutlierFilter()
    @test_throws "must be at least 1" AutoRIFT.rescale(f, 0)

    # `_oversample` is what feeds it, and reads the *finest* chip against the spacing.
    @test AutoRIFT._oversample(params(; chip_size = 32, grid_spacing = 32)) == 1
    @test AutoRIFT._oversample(params(; chip_size = 32, grid_spacing = 16)) == 2
    @test AutoRIFT._oversample(params(; chip_size = 32, grid_spacing = 8)) == 4
    # A spacing coarser than the chip has no overlap to correct for.
    @test AutoRIFT._oversample(params(; chip_size = 16, grid_spacing = 32)) == 1
    # Non-square: the smaller ratio wins, so neither axis is over-widened.
    @test AutoRIFT._oversample(params(; chip_size = (X = 32, Y = 32),
                                      grid_spacing = (X = 8, Y = 32))) == 1
end
