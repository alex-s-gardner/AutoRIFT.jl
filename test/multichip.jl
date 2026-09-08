using AutoRIFT: ImagePair, gridpoints, params, correlate_multichip, chipsize_level,
                MultichipResult, nmeasured, resample, resample!, Nearest, Area, Bicubic,
                dilate_within, small_components, windowmax,
                DisplacementField, _fill_holes!

# Same convention as track.jl's tests: the correlator returns secondary-to-reference, so the
# feature motion is its negative.
motion(r) = (-r.dx, -r.dy)

med(v) = (s = sort(collect(v)); isempty(s) ? NaN : s[(length(s) + 1) ÷ 2])

# A scene with a genuinely featureless band. This is what the pyramid exists for: a chip that
# fits inside the band has zero variance and cannot correlate, while a chip large enough to
# straddle its edge still sees real texture.
#
# Note that merely *low contrast* does not defeat a small chip — ZNCC normalises each window
# by its own scale, so a 50x contrast reduction is invisible to it. The texture has to be
# absent, not faint.
function banded_pair(n, shift, band)
    ref, sec = shifted_pair(n, shift; T = Float32)
    ref[band, :] .= 0.5f0
    sec[band, :] .= 0.5f0
    return ImagePair(ref, sec)
end

@testset "resample: nearest" begin
    a = Float32[1 2; 3 4]
    # Upsampling replicates; downsampling picks.
    @test resample(a, (4, 4), Nearest()) ==
          Float32[1 1 2 2; 1 1 2 2; 3 3 4 4; 3 3 4 4]
    @test size(resample(a, (1, 1), Nearest())) == (1, 1)
    # A mask must survive unchanged in value: nearest never invents an intermediate.
    m = Float32[1 0; 0 1]
    @test sort(unique(resample(m, (6, 6), Nearest()))) == Float32[0, 1]
end

@testset "resample: area" begin
    a = Float32[1 2; 3 4]
    @test resample(a, (1, 1), Area())[1] ≈ 2.5f0        # the mean
    # Averaging down a constant field is that constant, which is the property that keeps a
    # coarse prior unbiased.
    @test all(≈(7.0f0), resample(fill(7.0f0, 8, 8), (2, 2), Area()))

    # NaN is skipped and the weights renormalised, so a partly-missing cell yields the mean
    # of what is present rather than being dragged toward zero.
    b = Float32[1 NaN; 3 4]
    @test resample(b, (1, 1), Area())[1] ≈ Float32((1 + 3 + 4) / 3)
    @test all(isnan, resample(fill(NaN32, 4, 4), (2, 2), Area()))
end

@testset "resample: bicubic" begin
    # Catmull-Rom interpolates through its samples, so resampling to the same size is the
    # identity — which is what keeps an already-known coarse estimate unchanged.
    a = Float32[i + 2j for i in 1:8, j in 1:8]
    @test resample(a, (8, 8), Bicubic()) ≈ a atol = 1e-4

    # A linear ramp stays linear under upsampling: no overshoot on smooth data.
    r = resample(Float32[Float32(j) for i in 1:4, j in 1:8], (8, 16), Bicubic())
    @test all(isfinite, r)
    @test issorted(r[4, :])

    # Mostly-missing neighbourhoods yield NaN rather than a confident value extrapolated
    # from one corner.
    b = fill(NaN32, 8, 8)
    b[1, 1] = 1.0f0
    out = resample(b, (16, 16), Bicubic())
    @test isnan(out[end, end])
end

@testset "resample: an explicit scale overrides the size ratio" begin
    # The default is `size(A) / size(out)` per axis, which is what `cv2.resize` uses and so what the
    # chip-size levels are matched against. `scale` exists for a caller whose two grids stand in a
    # correspondence the sizes do not imply.
    n, stride = 371, 2
    sr = length(1:stride:n)
    src = Float32.(reshape(collect(1:sr), :, 1))

    # On the stride lattice each destination reads the cell it was sliced from.
    strided = resample(src, (n, 1), Nearest(); scale = (1 / stride, 1.0))
    @test all(i -> strided[i, 1] == Float32(min(cld(i, stride), sr)), 1:n)

    # The inferred ratio drifts against it over the far part of the axis, which is the error a
    # chip-size level would otherwise carry.
    ratio = resample(src, (n, 1), Nearest())
    @test count(i -> ratio[i, 1] != strided[i, 1], 1:n) > 90

    # Where the size does divide, the two coincide, which pins the default against the override.
    @test resample(src, (stride * sr, 1), Nearest()) ==
          resample(src, (stride * sr, 1), Nearest(); scale = (1 / stride, 1.0))

    # The bicubic kernel is `a = -0.75`, matching `cv2.INTER_CUBIC`. Both members interpolate through
    # their samples and so agree at the nodes; they part company between them, which is why the
    # parameter is pinned rather than left to taste. Checked against the polynomial OpenCV evaluates.
    cv(t, a = -0.75) = (a * ((t + 1)^3) - 5a * ((t + 1)^2) + 8a * (t + 1) - 4a,
                        (a + 2) * t^3 - (a + 3) * t^2 + 1.0,
                        (a + 2) * ((1 - t)^3) - (a + 3) * ((1 - t)^2) + 1.0,
                        a * ((2 - t)^3) - 5a * ((2 - t)^2) + 8a * (2 - t) - 4a)
    w = AutoRIFT.MVector4()
    for t in 0.0:0.05:1.0
        AutoRIFT._cubic_weights!(w, t)
        v = cv(t)
        @test all(k -> isapprox(w[k], v[k]; atol = 1e-12), 1:4)
        @test sum(w[k] for k in 1:4) ≈ 1.0        # partition of unity, so a constant field survives
    end
end

@testset "dilate_within" begin
    m = falses(11, 11)
    m[6, 6] = true
    d = dilate_within(m, 3)
    # Euclidean, not chessboard: the corner of the 3x3 box is at distance sqrt(18) > 3.
    @test d[6, 6]
    # **Strict**, matching `distance_transform_edt(!MC) < BuffDistanceC` (`autoRIFT.py:709`). The
    # position exactly `radius` away is excluded, and on a lattice that is a whole ring of points: at
    # the default radius of 8 admitting them widened the searched region by 72 coarse cells on the
    # golden Landsat case, 30,084 fine points after expansion.
    @test !d[6, 9]          # exactly 3 away, so outside
    @test d[6, 8]           # 2 away
    @test !d[6, 10]
    @test !d[9, 9]          # distance sqrt(18) ~ 4.24
    @test count(d) == count(i -> (Tuple(i)[1] - 6)^2 + (Tuple(i)[2] - 6)^2 < 9,
                            CartesianIndices(m))

    @test !any(dilate_within(falses(5, 5), 3))       # empty stays empty
    @test all(dilate_within(trues(5, 5), 1))
end

@testset "small_components" begin
    # Eight-connected, which is the reference's `connectivity=2`: a diagonal pair is one
    # component of two rather than two of one, and the size test turns on that.
    m = falses(5, 5)
    m[2, 2] = true
    m[3, 3] = true
    @test all(small_components(m, 3)[[CartesianIndex(2, 2), CartesianIndex(3, 3)]])
    @test !any(small_components(m, 2))          # the pair is size 2, not < 2

    # `minsize` is exclusive, matching `bwareaopen`'s `size < size1`.
    row = falses(7, 7)
    row[3, 2:6] .= true
    @test !any(small_components(row, 5))        # exactly 5 is not smaller than 5
    @test count(small_components(row, 6)) == 5

    # Components are sized independently of each other.
    mixed = falses(9, 9)
    mixed[2, 2] = true
    mixed[6:8, 6:8] .= true
    r = small_components(mixed, 5)
    @test r[2, 2]
    @test !any(r[6:8, 6:8])

    @test !any(small_components(falses(5, 5), 5))     # empty stays empty
    @test !any(small_components(trues(4, 4), 5))      # one component of 16
    @test !any(small_components(trues(3, 3), 0))      # a threshold of 0 selects nothing
end

@testset "hole size fills what neighbour count cannot" begin
    mkfield(dx, dy) = DisplacementField(copy(dx), copy(dy), fill(NaN32, size(dx)),
                                        fill(0.5f0, size(dx)), trues(size(dx)))

    # A 2x2 hole is where the two criteria part. Each of its points has five of nine measured
    # neighbours, one short of `fill_window`'s threshold of six, while the hole is four points
    # and so smaller than `fill_min_hole`. Neither criterion alone closes it.
    function twobytwo(minhole)
        dx = fill(1.0f0, 11, 11)
        dy = fill(2.0f0, 11, 11)
        dx[5:6, 5:6] .= NaN32
        dy[5:6, 5:6] .= NaN32
        d = mkfield(dx, dy)
        filled = _fill_holes!(d, params(; chip_size = 24, fill_min_hole = minhole))
        return d, filled
    end
    @test count(!isnan, first(twobytwo(5)).dx[5:6, 5:6]) == 4
    @test count(!isnan, first(twobytwo(0)).dx[5:6, 5:6]) == 0    # window criterion alone leaves it

    # The filled value is the neighbourhood median, so a uniform field fills with its own value
    # on both axes rather than with the other axis's or with zero.
    d, filled = twobytwo(5)
    @test all(d.dx[5:6, 5:6] .== 1.0f0)
    @test all(d.dy[5:6, 5:6] .== 2.0f0)
    # The four filled points are reported, so a caller can mark them interpolated.
    @test length(filled) == 4

    # The size criterion must not run away on a large hole: three passes close at most three
    # rings, and the interior stays open. A hole judged by its size *before* its edge is filled
    # would swallow the whole region.
    dx = fill(1.0f0, 31, 31)
    dy = fill(2.0f0, 31, 31)
    dx[10:22, 10:22] .= NaN32
    dy[10:22, 10:22] .= NaN32
    big = mkfield(dx, dy)
    _fill_holes!(big, params(; chip_size = 24, fill_min_hole = 5))
    @test all(isnan, big.dx[14:18, 14:18])

    # A hole with no measured neighbour at all stays open however small, since there is nothing
    # for the median to take. This is the reference's `MM` term.
    lone = mkfield(fill(NaN32, 7, 7), fill(NaN32, 7, 7))
    _fill_holes!(lone, params(; chip_size = 24, fill_min_hole = 5))
    @test all(isnan, lone.dx)
end

@testset "exact recovery through the pyramid" begin
    # End to end: the whole multi-scale machinery must not degrade what a single level gets
    # right. Every point resolved, at the finest chip, with the exact shift.
    for (sx, sy) in ((0, 0), (5, -3), (11, 7))
        ref, sec = shifted_pair(512, (sx, sy); T = Float32)
        pair = ImagePair(ref, sec)
        grid = gridpoints((512, 512), 32; chip_size = 32, search_radius = 25)
        r = correlate_multichip(pair, grid, params(; chip_size = 32, chip_size_max = 128))

        @test nmeasured(r) == length(r)
        mx, my = motion(r)
        @test med(filter(!isnan, mx)) ≈ sx atol = 0.05
        @test med(filter(!isnan, my)) ≈ sy atol = 0.05
        # A fully-textured scene is resolved by the finest chip everywhere, so the coarser
        # levels are never needed — which is the behaviour that makes the pyramid cheap on
        # easy scenes.
        @test all(cs -> cs == 32, filter(>(0), vec(r.chip_size)))
    end
end

@testset "coarser levels recover what finer ones cannot" begin
    # The reason the pyramid exists, and the only test that exercises more than one level.
    # Coverage must not fall as coarser chips are permitted.
    n = 512
    pair = banded_pair(n, (6, -4), 200:290)
    grid = gridpoints((n, n), 32; chip_size = 32, search_radius = 25)

    counts = Int[]
    for cmax in (32, 64, 128)
        r = correlate_multichip(pair, grid, params(; chip_size = 32, chip_size_max = cmax))
        push!(counts, nmeasured(r))
        used = sort(unique(Int.(filter(>(0), vec(r.chip_size)))))
        # Every permitted level up to the maximum contributes something.
        @test used ⊆ [32, 64, 128]
        @test 32 in used
        cmax >= 64 && @test 64 in used
    end
    # Non-decreasing rather than strictly increasing. A level coarser than the finest runs on a
    # proportionally coarser grid — `_level_decimation` — so it posts one estimate per chip
    # footprint rather than one per fine grid point, and its estimates are spread back over that
    # footprint. Two successive levels can therefore cover the same points, which is the intended
    # behaviour: permitting a coarser chip may add nothing where the level below already answered.
    @test counts[1] <= counts[2] <= counts[3]
    # Permitting coarser chips must still buy something overall, or the pyramid is pointless.
    @test counts[1] < counts[3]

    # The finest chip alone cannot resolve the band: a 32-wide chip that fits inside it has zero
    # variance. That is what the coarser levels are for, and what the counts above measure.
    fine = correlate_multichip(pair, grid, params(; chip_size = 32, chip_size_max = 32))
    @test nmeasured(fine) < counts[3]
    @test count(isnan, fine.dx) >= 40        # the band spans several grid rows

    # With the whole pyramid most of the band is resolvable, because the points the coarse levels
    # leave open form holes small enough for `fill_min_hole` to close. Filling is part of the answer
    # rather than a fallback, so this asserts the measurement is right where it fills — a filled
    # point that invented motion would fail the median checks below.
    #
    # **A fraction, not an exact count.** The band is set to a single constant, so a chip lying
    # inside it has zero variance and whether it correlates at all is decided by the last bits of a
    # variance sum. Reduction order is not fixed across Julia versions — 1.10 and 1.12 disagree on
    # three of the 196 base-level points here, which cascades to 42 open holes against 8 — so an
    # equality would assert one version's rounding. What the pyramid owes is that nearly every point
    # is answered, and that is what this measures.
    @test counts[3] >= 0.75 * length(grid.x)
    mx, my = motion(correlate_multichip(pair, grid,
                                        params(; chip_size = 32, chip_size_max = 128)))
    @test med(filter(!isnan, mx)) ≈ 6 atol = 0.05
    @test med(filter(!isnan, my)) ≈ -4 atol = 0.05

    # Raising the hole-size criterion is what closes those points, which is what makes them a fill
    # rather than a measurement. Compared across two thresholds on the same run rather than against
    # `counts[3]`: how many holes are small enough to close at the default is float-dependent per
    # above, and on some versions the default closes none of them.
    unfilled = nmeasured(correlate_multichip(
        pair, grid, params(; chip_size = 32, chip_size_max = 128, fill_min_hole = 0)))
    generous = nmeasured(correlate_multichip(
        pair, grid, params(; chip_size = 32, chip_size_max = 128, fill_min_hole = 16)))
    @test unfilled < generous
    # Filling only ever adds, so the default sits between the two.
    @test unfilled <= counts[3] <= generous
end

@testset "chip_size records the level that won" begin
    n = 512
    pair = banded_pair(n, (6, -4), 200:290)
    grid = gridpoints((n, n), 32; chip_size = 32, search_radius = 25)
    r = correlate_multichip(pair, grid, params(; chip_size = 32, chip_size_max = 128))

    # Measured exactly where a level claimed the point, and unmeasured exactly where none
    # did. This is the invariant that makes `chip_size` interpretable downstream.
    @test all(i -> (r.chip_size[i] == 0) == isnan(r.dx[i]), eachindex(r.dx))
    # Points in the featureless band that did resolve did so with a chip large enough to
    # reach outside it.
    @test !isempty(filter(cs -> cs > 32, vec(r.chip_size)))
end

@testset "the finest level that answers owns the point" begin
    # The merge rule that makes `chip_size` meaningful: a coarser level's estimate must never replace a
    # finer one, and a point a finer level answered must not even be offered to a coarser one. Both are
    # asserted by replaying the level loop and checking what each level was asked for against what the
    # merged result attributes to it.
    #
    # Worth pinning because the guard is one `continue` in `_merge_level!` plus one `wanted` mask in
    # `_multichip`, and losing either would degrade accuracy silently: a coarse chip averages more
    # ground, so its answer is smoother and still plausible everywhere it wrongly won.
    n = 512
    pair = banded_pair(n, (6, -4), 200:290)
    grid = gridpoints((n, n), 8; chip_size = 16, search_radius = 20)
    p = params(; chip_size = 16, chip_size_max = 64, grid_spacing = 8, search_radius = 20)
    sizes = AutoRIFT._level_sizes(p)

    result = AutoRIFT._empty_result(size(grid))
    answered = Dict{Int,BitMatrix}()
    for (k, cs) in enumerate(sizes)
        wanted = result.chip_size .== 0
        # Nothing a finer level already owns may be attempted.
        @test !any(wanted .& (result.chip_size .!= 0))
        lvl = chipsize_level(AutoRIFT.WholeScene(pair), grid, p, cs, wanted,
                             AutoRIFT.measure_at(p, k), AutoRIFT.subpixel_at(p, k))
        isnothing(lvl) && continue
        answered[cs.X] = .!isnan.(lvl.field.dx)
        before = copy(result.chip_size)
        AutoRIFT._merge_level!(result, lvl.field, lvl.filled, cs)
        # Every point that already had an owner keeps it.
        kept = before .!= 0
        @test all(result.chip_size[kept] .== before[kept])
    end

    # And no point is attributed to a level coarser than one that answered it.
    for (k, cs) in enumerate(sizes)
        own = result.chip_size .== cs.X
        any(own) || continue
        for f in sizes[1:(k - 1)]
            haskey(answered, f.X) || continue
            @test !any(own .& answered[f.X])
        end
    end
    @test count(!isnan, result.dx) > 0
end

@testset "peak_ratio accompanies every measured point" begin
    n = 512
    pair = banded_pair(n, (6, -4), 200:290)
    grid = gridpoints((n, n), 32; chip_size = 32, search_radius = 25)
    r = correlate_multichip(pair, grid, params(; chip_size = 32, chip_size_max = 128))

    # A point carrying a displacement carries a quality for it, and a point carrying neither carries
    # neither. `chip_size` follows the same rule, and a caller reading `peak_ratio` to gate on quality
    # needs it: a measured point with no quality would silently fail any threshold.
    @test all(i -> isnan(r.dx[i]) == isnan(r.peak_ratio[i]), eachindex(r.dx))
    # A real match stands above its background, and `peak_ratio` reports zero only at a peak against
    # the search boundary — which this scene's 6/-4 px shift at radius 25 does not produce.
    @test all(>(0), filter(isfinite, r.peak_ratio))
end

@testset "peak_ratio is not interpolated across a coarse cell" begin
    # A decimated level posts one estimate per chip footprint. `dx`/`dy` are spread over the cell
    # bicubically, but a quality describes one specific correlation surface and there is no surface
    # between two cells for an interpolated value to be about — so it is resampled `Nearest` and
    # every fine point in a cell must report that cell's value exactly.
    n = 512
    pair = banded_pair(n, (6, -4), 200:290)
    grid = gridpoints((n, n), 8; chip_size = 16, search_radius = 20)
    r = correlate_multichip(pair, grid, params(; chip_size = 16, chip_size_max = 64,
                                               grid_spacing = 8, search_radius = 20))
    nr, nc = size(r.peak_ratio)
    coarse = filter(c -> c > 16, unique(r.chip_size))
    if !isempty(coarse)
        for c in coarse
            stride = Int(c) ÷ 16
            # The cell `_decimate_level` took the point from — `cld(i, stride)`, since it slices
            # `1:stride:n`. Spelled out rather than taken from `resample`, so this asserts the lattice
            # and not merely the resampler's self-consistency.
            srows = length(1:stride:nr)
            scols = length(1:stride:nc)
            cellof(i, ns) = min(cld(i, stride), ns)
            seen = Dict{Tuple{Int,Int},Float32}()
            for j in 1:nc, i in 1:nr
                (r.chip_size[i, j] == c && isfinite(r.peak_ratio[i, j])) || continue
                key = (cellof(i, srows), cellof(j, scols))
                v = get!(seen, key, r.peak_ratio[i, j])
                @test v == r.peak_ratio[i, j]
            end
        end
    end
end

@testset "a coarse level is correlated where it is read back from" begin
    # The round trip a decimated level makes: `_decimate_level` picks where to correlate, and
    # `_undecimate_level` interpolates the answers back with the half-sample convention, which places
    # coarse node `k` at fine position `(k - 0.5) * stride + 0.5`. The two have to name the same place. A
    # level correlated at its cells' first grid point instead measures the field half a cell from where
    # every consumer assumes, which is a displacement error the size of the field's variation over that
    # distance — 0.14 px on a Landsat pair, and growing with `stride`.
    n = 512
    p = params(; chip_size = 16, chip_size_max = 64, grid_spacing = 8, search_radius = 20)
    grid = gridpoints((n, n), 8; chip_size = 16, search_radius = 20)
    nr, nc = size(grid)
    for stride in (2, 4)
        sub = AutoRIFT._decimate_level(grid, trues(nr, nc), stride)
        @test !isnothing(sub)
        # Where the interpolant reads node `k` from, in fine-grid index space, against where the node was
        # actually placed — expressed in the same units by mapping through the grid's own coordinates.
        x0 = grid.x[1, 1]
        sx = grid.x[1, 2] - grid.x[1, 1]
        for k in axes(sub.grid.x, 2)
            placed = (sub.grid.x[1, k] - x0) / sx + 1          # fine index the node sits at
            # The last cell along an axis can be short, so the centre is taken over the rows and columns
            # the cell actually spans rather than a full `stride` of them.
            c = sub.cols[k]
            expected = (c + min(c + stride - 1, nc)) / 2
            @test placed ≈ expected
        end
    end

    # A level that is not decimated must not be moved at all.
    @test AutoRIFT._decimate_level(grid, trues(nr, nc), 1).grid.x == grid.x
end

@testset "a coarse level lands on its own cells when the stride does not divide the grid" begin
    # `_undecimate_level` must read a coarse node back onto the cell `_decimate_level` sliced it from.
    # The two lattices coincide when `stride` divides the grid and diverge when it does not: at
    # `nr = 371, stride = 2` the size ratio is 186/371, which crosses a whole cell at row 186 and
    # shifts every row beyond it. On a real scene that surfaces as rings of chip-size disagreement
    # tracing the level boundaries in the far half of the grid, so the odd size is the case to assert.
    full = gridpoints((3072, 3072), 8; chip_size = 16, search_radius = 20)
    for nr in (371, 368), stride in (2, 4)
        sub = AutoRIFT._decimate_level(full[1:nr, 1:nr], trues(nr, nr), stride)
        @test !isnothing(sub)
        sr = length(sub.rows)
        # `peak_ratio` carries a distinct value per node and is resampled `Nearest`, so it shows directly
        # which coarse cell each destination read — `cld(i, stride)`, the cell the slice took it from.
        # This is the assertion that pins the lattice; the bicubic channel cannot serve, because for an
        # even stride a cell centre falls on a half-integer fine row and so no destination ever
        # coincides with a node.
        tag = Float32.(reshape(1:(sr * sr), sr, sr))
        field = DisplacementField(fill(0.0f0, sr, sr), fill(0.0f0, sr, sr), fill(1.0f0, sr, sr),
                                  tag, trues(sr, sr))
        up = AutoRIFT._undecimate_level((; field, filled = Int[]), (nr, nr), sub.rows, sub.cols)
        for j in 1:nr, i in 1:nr
            @test up.field.peak_ratio[i, j] == tag[min(cld(i, stride), sr), min(cld(j, stride), sr)]
        end
        # A constant field survives interpolation exactly, whatever the lattice — the kernel is a
        # partition of unity, and this is what says the bicubic channel carries no scale error.
        flat = DisplacementField(fill(2.5f0, sr, sr), fill(-1.5f0, sr, sr), fill(1.0f0, sr, sr),
                                 fill(1.0f0, sr, sr), trues(sr, sr))
        upf = AutoRIFT._undecimate_level((; field = flat, filled = Int[]), (nr, nr),
                                         sub.rows, sub.cols)
        @test all(v -> isapprox(v, 2.5f0; atol = 1e-5), filter(!isnan, upf.field.dx))
        @test all(v -> isapprox(v, -1.5f0; atol = 1e-5), filter(!isnan, upf.field.dy))
    end
end

@testset "a coarse hole is filled from the closest evidence that covers it" begin
    # `_fill_level_holes` fills in order of increasing reach: the finer levels' answer at that same
    # point, then a median of this level's own neighbourhood, then its nearest finite value however
    # far. Only the order is asserted here — each step must win over the ones that reach further,
    # because a hole beside a velocity discontinuity is filled across it otherwise.
    #
    # `prior` arrives already reduced onto the level's own grid; `_undecimate_level` does that, and the
    # testset below pins the reduction itself.
    n = 9

    # A broad slow region, a gap, then a thin fast sliver — the shape a fast-flowing feature's edge
    # has. The nearest finite cell to column 6 is the sliver, so a nearest-neighbour fill hands it the
    # sliver's 10.0 even though its neighbourhood is mostly the slow region.
    A = fill(NaN32, n, n)
    A[:, 1:4] .= 1.0f0
    A[:, 7] .= 10.0f0

    # With a prior covering the gap, the prior wins over both the median and the nearest value.
    prior = fill(NaN32, n, n)
    prior[:, 5:6] .= -7.0f0
    got = AutoRIFT._fill_level_holes(A, prior)
    @test all(≈(-7.0f0), got[:, 5:6])
    # The measured values are untouched: filling may not alter what was actually correlated.
    @test got[:, [1, 2, 3, 4, 7]] == A[:, [1, 2, 3, 4, 7]]

    # Without a prior, the median of the level's own neighbourhood answers where it reaches, and being
    # a neighbourhood statistic it lands between the two populations rather than on the nearer one.
    nomedian = AutoRIFT._fill_level_holes(A, nothing)
    @test all(v -> 1.0f0 < v < 10.0f0, nomedian[:, 6])
    @test all(≈(10.0f0), AutoRIFT._fill_nan_nearest(A)[:, 6])

    # The per-hole median must equal the swept one, which is what says the gather is a pure speedup.
    swept = let B = copy(A), m = AutoRIFT.windowmedian(A, 5)
        for i in eachindex(B)
            isnan(B[i]) && isfinite(m[i]) && (B[i] = m[i])
        end
        AutoRIFT._fill_nan_nearest(B)
    end
    @test nomedian == swept

    # A hole beyond the median's reach still gets the nearest value, since nothing nearer exists.
    far = fill(NaN32, n, n)
    far[1, 1] = 3.0f0
    @test all(≈(3.0f0), AutoRIFT._fill_level_holes(far, nothing))

    # Nothing finite anywhere is left alone rather than invented.
    @test all(isnan, AutoRIFT._fill_level_holes(fill(NaN32, n, n), nothing))
end

@testset "the prior is mean-filtered before it is reduced onto a level" begin
    # `_undecimate_level` reduces the finer levels' answer with a `stride + 1` mean filter and *then* an
    # area resize, not the resize alone. The filter's window is one wider than a cell, so it reaches
    # across cell boundaries and closes a partly-missing neighbourhood before the area step weights what
    # is left — the reference's `colfilt(Dx, (Scale+1, Scale+1), 2)` ahead of its `INTER_AREA`
    # (`autoRIFT.py:823-839`). Pinned because dropping it reproduces the reference's own `DxF0` at 1.6%
    # of points instead of 90%, and the prior is what fills most of a coarse level's holes.
    #
    # A single fine value beside a hole is the discriminating case: the filter spreads it into the
    # neighbouring cell, where the bare resize leaves that cell empty.
    stride, n = 2, 9
    rows = cols = 1:stride:(stride * n)
    prior = fill(NaN32, stride * n, stride * n)
    prior[1, 1] = 4.0f0
    field = DisplacementField(fill(NaN32, n, n), fill(NaN32, n, n), fill(NaN32, n, n),
                              fill(NaN32, n, n), trues(n, n))
    field.dx[n, n] = 0.0f0                  # one measurement, far away, so the prior decides cell 1
    field.dy[n, n] = 0.0f0
    up = AutoRIFT._undecimate_level((; field, filled = Int[]), (stride * n, stride * n),
                                    rows, cols, prior, prior)
    # The reduction the level performs, recovered by filling the level's own field with it.
    reduced = AutoRIFT._fill_level_holes(field.dx,
        AutoRIFT.resample(AutoRIFT.windowmean(prior, stride + 1), (n, n), Area();
                          scale = (Float64(stride), Float64(stride))))
    bare = resample(prior, (n, n), Area(); scale = (Float64(stride), Float64(stride)))
    @test reduced[1, 1] ≈ 4.0f0             # the prior reached this cell
    @test reduced[1, 2] ≈ 4.0f0             # and the next one, which the bare resize cannot fill
    @test isnan(bare[1, 2])
    @test size(up.field.dx) == (stride * n, stride * n)
end

@testset "chipsize_level in isolation" begin
    # Callable on its own, which is what lets a caller run one scale without the pyramid.
    ref, sec = shifted_pair(512, (5, -3); T = Float32)
    pair = ImagePair(ref, sec)
    grid = gridpoints((512, 512), 32; chip_size = 32, search_radius = 25)
    p = params(; chip_size = 32)

    lvl = chipsize_level(pair, grid, p, 32, trues(size(grid)))
    @test !isnothing(lvl)
    @test med(filter(!isnan, -lvl.field.dx)) ≈ 5 atol = 0.05
    # `filled` records what the hole fill recovered. Even on a fully-textured scene this is
    # not always empty: the outlier filter rejects the occasional point, and the fill then
    # recovers it from its neighbours. Every filled index must be a real point of the grid.
    @test all(i -> 1 <= i <= length(grid), lvl.filled)
    # A filled point has a displacement but no correlation of its own, since nothing was
    # measured there.
    @test all(i -> !isnan(lvl.field.dx[i]), lvl.filled)

    # `wanted` restricts which points are attempted, which is how the pyramid stops a level
    # from redoing work a finer one already did.
    w = falses(size(grid))
    w[1:3, 1:3] .= true
    lvl = chipsize_level(pair, grid, p, 32, w)
    @test !isnothing(lvl)
    @test count(!isnan, lvl.field.dx) <= 9
    # Nothing wanted at all is not an error, just no work.
    @test isnothing(chipsize_level(pair, grid, p, 32, falses(size(grid))))
end

@testset "each level quantizes at its own upsampling" begin
    # The point of a per-level subpixel method is that the level's *answer* is quantized by it, so
    # this asserts the quantization rather than the plumbing: displacements from a level run at
    # `upsampling = n` are exact multiples of `1/n`.
    #
    # A fractional true shift is what makes the test discriminating — an integer one lands on every
    # grid and would pass whatever the factor was.
    ref, sec = shifted_pair(512, (5.3, -3.7); T = Float32)
    pair = ImagePair(ref, sec)
    grid = gridpoints((512, 512), 32; chip_size = 32, search_radius = 25)

    quantized(v, n) = all(x -> abs(x * n - round(x * n)) < 1e-4, filter(!isnan, v))

    # **Measured points only.** `chipsize_level` returns its field after `_reject_and_fill!` has run,
    # so `field.dx` holds filled values beside measured ones — and a filled value is interpolated
    # from its neighbours, never passed through the cascade, so it has no reason to land on the
    # `1/n` grid. Two of the 196 points here fill to the midpoint of adjacent measurements, which is
    # exactly a half-step off. Asserting over the whole field tests the fill, not the quantization
    # this testset is named for; which points fill is float-dependent, so it also fails on one Julia
    # version and not another.
    measured(lvl, f) = f[setdiff(eachindex(f), lvl.filled)]

    for n in (8, 16, 64)
        p = params(; chip_size = 32, chip_size_max = 32, subpixel = (PyramidRefine(n),))
        lvl = chipsize_level(pair, grid, p, 32, trues(size(grid)), first(p.similarity),
                             AutoRIFT.subpixel_at(p, 1))
        @test !isnothing(lvl)
        @test quantized(measured(lvl, vec(lvl.field.dx)), n)
        @test quantized(measured(lvl, vec(lvl.field.dy)), n)
        # The shift is still recovered, so a finer grid is not being bought with accuracy.
        @test med(filter(!isnan, -lvl.field.dx)) ≈ 5.3 atol = 0.2
    end

    # Two levels with different factors: each obeys its own, which one shared factor could not
    # produce. Run through the whole loop rather than level by level, since that is where
    # `subpixel_at` is consulted.
    p2 = params(; chip_size = 32, chip_size_max = 64,
                subpixel = (PyramidRefine(8), PyramidRefine(64)))
    out = autorift(ref, sec, p2)
    fine = [out.dx[i] for i in eachindex(out.dx) if out.chip_size[i] == 32 && !isnan(out.dx[i])]
    coarse = [out.dx[i] for i in eachindex(out.dx) if out.chip_size[i] == 64 && !isnan(out.dx[i])]
    isempty(fine) || @test quantized(fine, 8)
    # A multiple of 1/64 need not be a multiple of 1/8, and on a fractional shift some are not —
    # which is what shows the two levels used different factors rather than one.
    isempty(coarse) || @test quantized(coarse, 64)
end

@testset "a level that finds nothing is skipped" begin
    # Two unrelated images: the coarse pass should find no spatial coherence and the level
    # should decline rather than emit noise.
    n = 384
    pair = ImagePair(synthetic_texture(n; seed = 1), synthetic_texture(n; seed = 99))
    grid = gridpoints((n, n), 32; chip_size = 32, search_radius = 25)
    r = correlate_multichip(pair, grid, params(; chip_size = 32, chip_size_max = 64))
    # Whatever survives must be a small minority: uncorrelated images have no coherent
    # displacement to find, and the outlier filter is what enforces that.
    @test nmeasured(r) < length(r) ÷ 2
end

@testset "the coarse pass restricts the fine search" begin
    # The mechanism that makes the pyramid affordable. With the coarse gate effectively
    # disabled, more points are attempted than with it active — the result should be no
    # better, at more cost.
    n = 512
    pair = banded_pair(n, (6, -4), 200:290)
    grid = gridpoints((n, n), 32; chip_size = 32, search_radius = 25)

    strict = correlate_multichip(pair, grid,
                               params(; chip_size = 32, min_coarse_valid_fraction = 0.9))
    loose = correlate_multichip(pair, grid,
                              params(; chip_size = 32, min_coarse_valid_fraction = 0.0))
    # A demanding threshold can reject a whole level; a permissive one cannot make the
    # answer worse where both measure.
    both = findall(i -> !isnan(strict.dx[i]) && !isnan(loose.dx[i]), eachindex(strict.dx))
    @test all(i -> abs(strict.dx[i] - loose.dx[i]) < 0.5, both)
end

@testset "the coarse pass samples at the rate times the oversampling" begin
    # `coarse_stride` is a rate against one point per chip, not a step in grid points, so on a grid
    # posted finer than its chips the sparse pass steps by the rate times that ratio — the reference's
    # `sparseSearchSampleRate * ChipSize0_GridSpacing_oversample_ratio` (`autoRIFT.py:605-614`).
    #
    # This is what makes `coarse_buffer` reach as far as the reference's: the buffer is in coarse cells
    # and is expanded back by the same factor, so a step of `coarse_stride` alone covers half the
    # ground. The symptom is not a wrong displacement but a level that declines to search a
    # neighbourhood, leaving a coarser chip to claim it — measured on a Landsat pair as a region where
    # this level's share of the output ran 855 points against 317.
    p = params(; chip_size = 16, chip_size_max = 64, grid_spacing = 8, search_radius = 20)
    @test AutoRIFT._oversample(p) == 2
    @test AutoRIFT._sparse_stride(p) == p.coarse_stride * AutoRIFT._oversample(p)

    # One point per chip leaves nothing to correct for, and the two must coincide there.
    flat = params(; chip_size = 32, chip_size_max = 32, grid_spacing = 32, search_radius = 20)
    @test AutoRIFT._oversample(flat) == 1
    @test AutoRIFT._sparse_stride(flat) == flat.coarse_stride

    # The sparse grid really is sampled at that step, rather than the step merely being computed.
    n = 1024
    grid = gridpoints((n, n), 8; chip_size = 16, search_radius = 20)
    pts = AutoRIFT._level_points(grid, p, AutoRIFT.extent(16), trues(size(grid)))
    setup = AutoRIFT._coarse_points(pts, p, AutoRIFT.extent(16))
    @test !isnothing(setup)
    st = AutoRIFT._sparse_stride(p)
    @test step(setup.rows) == st
    @test step(setup.cols) == st

    # `rescale`'s overlap term keeps taking the rate, not the step: it is written against
    # `sparseSearchSampleRate` itself, and passing the step would change the filter's threshold.
    @test setup.filt == AutoRIFT.rescale(AutoRIFT.relax(p.outliers),
                                         AutoRIFT._oversample(p), p.coarse_stride)
end

@testset "result shape and validation" begin
    ref, sec = shifted_pair(256, (0, 0); T = Float32)
    pair = ImagePair(ref, sec)
    grid = gridpoints((256, 256), 32; chip_size = 32, search_radius = 20)
    r = correlate_multichip(pair, grid, params(; chip_size = 32))
    @test size(r) == size(grid)
    @test size(r.chip_size) == size(grid)
    @test eltype(r.chip_size) === UInt16
    @test r.interpolated isa BitMatrix

    # Every valid parameter set selects at least one level, because `params` requires
    # `chip_size <= chip_size_max` with the same power-of-two ratio in both axes. Checked here so
    # the invariant the level loop relies on is pinned at the boundary that establishes it.
    for (mn, mx) in ((32, 32), (64, 64), (16, 128), (32, 128))
        @test !isempty(AutoRIFT.chip_sizes(params(; chip_size = mn, chip_size_max = mx)))
    end
    @test_throws ArgumentError params(; chip_size = 32, chip_size_max = 96)
end

@testset "grid spacing survives a zeroed nodata margin" begin
    # A production grid is zeroed wherever there is no data (`testautoRIFT.py:394-403`), so a scene
    # whose first rows and columns are ocean has `x[1, 2] == x[1, 1]`. Reading the spacing from the
    # first two points gives zero there, `_cell_centres` then shifts by nothing, and every coarse node
    # sits at its cell's first point — half a cell from where `_undecimate_level` reads it back. On the
    # golden Landsat case that left 99.8% of level-1 nodes 4 or 5 px from the reference's.
    # Zero is the nodata marker, so the margin is zeroed rather than held at the first coordinate --
    # which is what the driver writes and what makes a step touching it describe the margin.
    x = Float64[(c <= 5 || r <= 5) ? 0.0 : 1.5 + 8 * (c - 1) for r in 1:12, c in 1:12]
    y = Float64[(c <= 5 || r <= 5) ? 0.0 : 1.5 + 8 * (r - 1) for r in 1:12, c in 1:12]
    @test iszero(x[1, 1]) && iszero(x[1, 2])       # the trap: adjacent margin points are equal
    @test AutoRIFT._grid_step(x, 2) == 8
    @test AutoRIFT._grid_step(y, 1) == 8

    # A clean grid gives the same answer, so the mode is not a special case for margins.
    xc = Float64[1.5 + 8 * (c - 1) for _ in 1:12, c in 1:12]
    @test AutoRIFT._grid_step(xc, 2) == 8
    # Non-square spacing is read per axis.
    yc = Float64[1.5 + 3 * (r - 1) for r in 1:12, _ in 1:12]
    @test AutoRIFT._grid_step(yc, 1) == 3

    # **The step is signed, because the grid is rotated by an arbitrary amount.** A step along a row
    # moves `x` by the spacing times the cosine of the rotation: 8 on a near-axis-aligned Landsat grid
    # and −1 on a Sentinel-2 grid rotated near 90°. Keeping only positive steps sees nothing but the
    # jumps out of the zeroed margin — on the golden S2A case that returns 10979 where the answer is −1,
    # which puts every chip-96 coarse point outside a 10,980 px image and drops the level entirely.
    xrot = Float64[(c <= 5 || r <= 5) ? 0.0 : 5000.0 - (c - 1) for r in 1:12, c in 1:12]
    @test AutoRIFT._grid_step(xrot, 2) == -1

    # No spacing at all: a single column, or an all-zero grid.
    @test AutoRIFT._grid_step(zeros(4, 4), 2) == 0.0
    @test AutoRIFT._grid_step(reshape(Float64[1.5], 1, 1), 2) == 0.0

    # And the consequence the helper exists for: a decimated node lands at the cell centre, half a
    # cell from its first point, even when the grid's first row and column are margin.
    grid = AutoRIFT.rebuild(gridpoints((200, 200), 8; chip_size = 16, search_radius = 6)[1:12, 1:12];
                            x, y)
    sub = AutoRIFT._decimate_level(grid, trues(12, 12), 2)
    @test sub !== nothing
    # Cell (r, c) spans full columns 2c-1 and 2c, whose x differ by the spacing, so the centre is
    # half a spacing above the first — 4 px here.
    @test sub.grid.x[4, 4] == grid.x[7, 7] + 4
end

@testset "an even sparse stride reduces over an odd window" begin
    # `filtWidth = stride + 1` when the stride is even and `stride` when it is odd
    # (`autoRIFT.py:618-626`), so the coarse radius reduction is symmetric about the node it is
    # sampled at. An even window has a left bias, which would place a coarse point's radius over a
    # cell offset half a step from the point.
    @test AutoRIFT._sparse_filter_width(8) == 9
    @test AutoRIFT._sparse_filter_width(4) == 5
    @test AutoRIFT._sparse_filter_width(2) == 3
    @test AutoRIFT._sparse_filter_width(3) == 3
    @test AutoRIFT._sparse_filter_width(5) == 5
    # Odd either way, which is the property the rule exists for.
    for stride in 1:16
        @test isodd(AutoRIFT._sparse_filter_width(stride))
        @test AutoRIFT._sparse_filter_width(stride) >= stride
    end

    # And the reduction itself is symmetric: a one-hot radius at a coarse node must reach the same
    # distance either side of it. Reducing over the stride instead reaches one fewer point on the
    # right, which under-covers the cell and always downward.
    radius = zeros(Int, 33, 1)
    radius[17, 1] = 5
    stride = 8
    rows = stride:stride:33
    out = zeros(Int, length(rows), 1)
    AutoRIFT._cell_max_radius!(out, radius, rows, 1:1, stride,
                               AutoRIFT._sparse_filter_width(stride))
    # Node 17 is `rows[2]`, so its own cell sees the 5; the neighbours either side must see it
    # equally or not at all.
    @test out[2, 1] == 5
    @test out[1, 1] == out[3, 1]
end

@testset "the coarse mask sits on the lattice the radii were reduced over" begin
    # `_cell_max_radius!` takes the maximum radius over the cell *centred* on each coarse point,
    # and `_expand_coarse_mask` inverts that assignment. The two have to agree for every fine
    # index, because the mask restricts the fine search and the radii are the evidence behind it.
    #
    # `resample(..., Nearest())` does not: it derives cell boundaries from the size ratio, which
    # for an even stride are left-aligned rather than centred. At `nr = 13, stride = 4` it puts
    # fine row 5 in coarse cell 2 while the radius for that row came from cell 1.
    for stride in (2, 3, 4, 5, 8), n in (7, 13, 20, 33, 64)
        rows = stride:stride:n
        isempty(rows) && continue
        ncoarse = length(rows)
        # From the helper the source now uses, so the test cannot drift from it by restating the
        # convention independently — which is the whole failure mode this testset exists for.
        lo, _, hi, _ = AutoRIFT._window_margins(stride, stride)

        # Which coarse cell each fine index draws its radius from, read straight off
        # `_cell_max_radius!`'s window: a one-hot radius field at coarse point `k` must reduce to
        # a nonzero maximum exactly for the fine indices that cell covers.
        radius_cell = zeros(Int, n)
        for (k, r) in enumerate(rows), i in max(r - lo, 1):min(r + hi, n)
            radius_cell[i] = k
        end

        mask_cell = zeros(Int, n)
        for k in 1:ncoarse
            onehot = falses(ncoarse, 1)
            onehot[k, 1] = true
            expanded = AutoRIFT._expand_coarse_mask(onehot, (n, 1), stride)
            for i in 1:n
                expanded[i, 1] && (mask_cell[i] = k)
            end
        end

        # Every fine index belongs to exactly one coarse cell, and it is the same one in both
        # directions wherever the radius reduction defines it.
        @test all(mask_cell[i] > 0 for i in 1:n)
        for i in 1:n
            radius_cell[i] == 0 && continue
            @test mask_cell[i] == radius_cell[i]
        end
    end

    # And the case that exposed the defect, asserted concretely rather than only as a property.
    onehot = falses(3, 1)
    onehot[1, 1] = true
    @test vec(AutoRIFT._expand_coarse_mask(onehot, (13, 1), 4)) ==
          [true, true, true, true, true, false, false, false, false, false, false, false, false]
    # `resample` disagrees here, which is why it is not used for this.
    viaresample = vec(resample(Float32.(onehot), (13, 1), Nearest())) .> 0.5f0
    @test viaresample != vec(AutoRIFT._expand_coarse_mask(onehot, (13, 1), 4))

    # A coarse point's own fine index always lands in its own cell, at any stride.
    for stride in (2, 3, 4, 7)
        n = 40
        rows = stride:stride:n
        for (k, r) in enumerate(rows)
            onehot = falses(length(rows), 1)
            onehot[k, 1] = true
            @test AutoRIFT._expand_coarse_mask(onehot, (n, 1), stride)[r, 1]
        end
    end
end
