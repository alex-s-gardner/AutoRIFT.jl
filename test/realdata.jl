# Jakobshavn Isbrae against ITS_LIVE, on the Landsat pair ITS_LIVE itself processed.
#
# Every other test in this suite uses synthetic texture with a known answer. That verifies the
# correlator against arithmetic; it cannot verify the package against *ice*. Real imagery brings
# cloud, shadow, saturated snow, shear margins, crevasse fields that decorrelate, and a velocity
# field spanning three orders of magnitude — none of which a shifted-texture test produces.
#
# Skipped unless the cache is present, because the inputs are 700 MB of Landsat that cannot be
# committed. `tools/realdata/README.md` says how to build it.
#
# The comparison is statistical by necessity: ITS_LIVE is not ground truth, it is another
# correlator's answer, run at a per-point search radius this test does not reproduce. So the
# assertions are on agreement *with margin* rather than on equality, and the margins are set from
# measured values with room for the reference to be revised.

if !has_realdata()
    @info "Real-data cache absent; skipping. Build it with " *
          "`julia --project=tools/realdata tools/realdata/prepare.jl` — see tools/realdata/README.md"
else
    w = realdata()

    # ITS_LIVE ran this pair at 240/480/960 m chips on a 120 m grid, which at the panchromatic
    # band's 15 m is 16/32/64 px at a spacing of 8. Read from the granule rather than chosen: it
    # records `x_pixel_size` and the chip sizes it used.
    #
    # The search radius is the one parameter not derivable from the granule, since ITS_LIVE sizes
    # it per point from a prior velocity field. 20 px covers 12,000 m/yr over this pair's 8.0-day
    # separation (262.8 m = 17.5 px) with margin — Jakobshavn's trunk reaches 11,816 m/yr here, so
    # a smaller radius would rail out on exactly the ice this test exists to measure.
    p = params(; chip_size = (X = 16, Y = 16), chip_size_max = (X = 64, Y = 64),
               grid_spacing = (X = 8, Y = 8), search_radius = (X = 20, Y = 20))

    out = autorift(w.reference, w.secondary, p)

    @testset "the pass measures most of the scene" begin
        frac = count(!isnan, out.dx) / length(out.dx)
        # 70% measured. The rest is ocean, cloud and the ice sheet's featureless interior, where
        # there is genuinely nothing to correlate — a run that measured everything would be
        # reporting noise as signal.
        @test 0.5 < frac < 0.95
        @test all(cs -> cs == 0 || cs in (16, 32, 64), out.chip_size)
    end

    # Displacement is in pixels, secondary-to-reference — the negative of feature motion, per
    # `peak_offset`'s docstring. `vy` additionally flips: a row index increases southward while the
    # granule's northing increases northward, so the two disagree by construction and a test that
    # did not flip would report a −0.98 correlation as a failure.
    scale = w.pixel_size * 365.25 / w.date_dt
    vx = .-out.dx .* scale
    vy = out.dy .* scale
    grid = AutoRIFT._build_grid(size(w.reference), p)

    rvx = similar(vx)
    rvy = similar(vy)
    for k in eachindex(IndexCartesian(), vx)
        col = clamp(round(Int, grid.x[k]), 1, size(w.reference_vx, 2))
        row = clamp(round(Int, grid.y[k]), 1, size(w.reference_vx, 1))
        rvx[k] = w.reference_vx[row, col]
        rvy[k] = w.reference_vy[row, col]
    end

    ok = .!isnan.(vx) .& .!isnan.(rvx)
    ours = sqrt.(vx[ok] .^ 2 .+ vy[ok] .^ 2)
    theirs = sqrt.(rvx[ok] .^ 2 .+ rvy[ok] .^ 2)

    @testset "velocities agree with ITS_LIVE" begin
        @test count(ok) > 20_000
        # 0.990 measured, on both components and on speed. This is the assertion the file exists
        # for: it fails for a transposed read, a sign error, a wrong scale factor, or a
        # correlator regression, and none of those are visible in synthetic tests.
        @test cor(vx[ok], rvx[ok]) > 0.95
        @test cor(vy[ok], rvy[ok]) > 0.95
        @test cor(ours, theirs) > 0.95
        # Median error −6 m/yr against a 350 m/yr median field: no systematic bias.
        @test abs(median(ours .- theirs)) < 50
    end

    @testset "fast ice is recovered, not truncated" begin
        fast = theirs .> 2000
        @test count(fast) > 1000
        @test cor(ours[fast], theirs[fast]) > 0.95
        # Flow direction within 6.2 degrees measured. A rail-hit at the search edge would show up
        # here as a direction locked to the diagonal, and in the speed check as a ceiling.
        angles = map(findall(fast)) do i
            a = rad2deg(abs(atan(vy[ok][i], vx[ok][i]) - atan(rvy[ok][i], rvx[ok][i])))
            min(a, 360 - a)
        end
        @test median(angles) < 20
        # The trunk's true speed is reached rather than clipped by the radius.
        @test maximum(ours) > 8000
    end

    @testset "a blocked run equals the untiled one" begin
        # The tiled path on real imagery, where the halo has to be right against genuine texture
        # rather than against a synthetic discontinuity. Same process, so FFTW wisdom is fixed and
        # anything short of equality is a defect.
        blocked = AutoRIFT.correlate_tiled(ImagePair(w.reference, w.secondary), grid, p,
                                           (768, 768))
        @test all(isequal.(blocked.dx, out.dx))
        @test all(isequal.(blocked.dy, out.dy))
        @test all(isequal.(blocked.correlation, out.correlation))
        @test all(isequal.(blocked.chip_size, out.chip_size))
    end
end
