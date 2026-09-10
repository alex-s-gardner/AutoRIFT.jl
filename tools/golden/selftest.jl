# The harness's own gate: does the comparison machinery work, before it is used to judge anything?
#
#   julia --project=tools/golden tools/golden/selftest.jl
#
# Two halves, and the second is the one that matters. Reading every golden product and diffing it
# against itself proves the reader is deterministic and the comparator accepts agreement — but a
# comparator that returned "identical" unconditionally would pass that too. So the second half
# injects a known difference of each kind the real comparison must detect and asserts it is caught.
#
# The five faults each represent a way a product comparison fails *silently* rather than loudly: a
# changed value, a changed coverage, a transposed plane, a changed attribute, a shifted grid. The
# transpose is the one worth naming — `tools/ab/README.md` records that a transposed displacement
# field still reads as a displacement field, and on a square grid nothing about its shape gives it
# away. So the injection runs on a non-square product, where a transpose is a shape error; detecting
# one on a square grid requires checking values instead.
#
# The third half gates the *input* path rather than the reader: `pointset_from_capture` is where all
# twenty cases get their correlator inputs, and it went untested while a wrong-sign `Dy0` in it made
# every case-level figure wrong for months. See `REFERENCE.md` on the y sign. That defect was
# undetectable in this file as written — a product comparison judges the answer, and an input
# translated wrongly produces a *plausible* answer.

include("manifest.jl")
include("product.jl")
include("compare.jl")
include("correlator.jl")

using AutoRIFT: chip_bounds, search_bounds, rebuild
using Test

"""
    cpp_rects(k, i, j, chip, rx, ry) -> (chip_rows, chip_cols), (window_rows, window_cols)

The chip and search-window rectangles the reference's C++ cuts at grid point `(i, j)`, as 1-based
inclusive ranges.

Transcribed from `autoriftcoremodule.cpp:129-140` (and the identical block in `arSubPixDisp_u`)
rather than derived, so this is an independent statement of the convention rather than a second
spelling of AutoRIFT.jl's. Three details are load-bearing and each changes the rectangle:

  * **`Dy0` is negated**, because `arImgDisp_*` does `Dy0 = -Dy0` before the call
    (`autoRIFT.py:1058`) and the capture records the pre-flip value.
  * **The two axes have different chip sizes.** `ChipSizeY = round(ChipSizeX * ScaleChipSizeY / 2) * 2`
    (`autoRIFT.py:650`), which is 1.0 on every optical case and **0.25** on Sentinel-1 — so a
    transcription using one `chip` for both axes is right on twelve cases and wrong on eight, in y
    only. Hence `chip_y` is a separate argument.
  * **The arithmetic is `Float32` and truncating.** C's `int(...)` rounds toward zero, not `-Inf`,
    and the C++ holds the grid as `CV_32FC1`. That matters only where a rectangle edge crosses zero,
    which happens at points whose window falls off the left or top of the image — a few dozen per
    scene on the Sentinel-2 cases. Those points are excluded by [`rect_mismatches`](@ref) rather
    than tolerated: the reference reads padded imagery there and AutoRIFT.jl rejects the point, so
    the rectangles are legitimately incomparable. Writing `floor` here would hide that instead, and
    would make this assert AutoRIFT.jl's convention rather than the reference's.
  * **`cv::Range(a, b)` is half-open**, so `(a+1):b` is the 1-based inclusive equivalent.
"""
function cpp_rects(k::Capture, i::Int, j::Int, chip_x::Int, chip_y::Int, rx::Int, ry::Int)
    clx = floor(Float32(chip_x) / 2.0f0)
    cly = floor(Float32(chip_y) / 2.0f0)
    xg = Float32(k.arrays["in_xGrid"][i, j])
    yg = Float32(k.arrays["in_yGrid"][i, j])
    d0x = Float32(k.arrays["in_Dx0"][i, j])
    d0y = Float32(-k.arrays["in_Dy0"][i, j])
    cxs = trunc(Int, -clx - d0x + xg); cxe = trunc(Int, clx - d0x + xg)
    cys = trunc(Int, -cly - d0y + yg); cye = trunc(Int, cly - d0y + yg)
    wxs = trunc(Int, -clx - Float32(rx) + xg); wxe = trunc(Int, clx + Float32(rx) - 1 + xg)
    wys = trunc(Int, -cly - Float32(ry) + yg); wye = trunc(Int, cly + Float32(ry) - 1 + yg)
    return ((cys + 1):cye, (cxs + 1):cxe), ((wys + 1):wye, (wxs + 1):wxe)
end

"""
    rect_mismatches(grid, k; nsample) -> (n, chip, window)

How many sampled base-level points AutoRIFT.jl cuts a different chip or window at than the reference
would, given `grid` from [`pointset_from_capture`](@ref).

Sampled on a stride rather than exhaustively: the grids are millions of points and the conventions
are uniform across them, so a few thousand spread over the whole grid catches a convention error
while keeping this a test rather than a run. Base level only, because the pyramid rewrites both
bounds per level and this checks the handoff, not the pyramid.

Points whose rectangle reaches off the top or left of the image are skipped. The reference pads both
images by `(Py, Px)` before correlating (`autoRIFT.py:1234-1235`) and so reads real array elements
where AutoRIFT.jl is out of bounds and rejects the point — and it is across that boundary that C's
truncation and `floor` diverge. Comparing there would report a difference in padding policy as a
difference in convention.
"""
function rect_mismatches(grid, k::Capture; nsample::Integer = 20_000)
    chip_x = Int(k.scalars["ChipSize0X"])
    chip_y = round(Int, chip_x * Float64(k.scalars["ScaleChipSizeY"]) / 2) * 2
    rx, ry = _level_search_limits(k.arrays["in_SearchLimitX"], k.arrays["in_SearchLimitY"], k)
    nr, nc = size(grid.x)
    lin = LinearIndices((nr, nc))
    cart = CartesianIndices((nr, nc))
    n = 0; bad_chip = 0; bad_win = 0
    for t in 1:max(1, (nr * nc) ÷ nsample):(nr * nc)
        ci = cart[t]; i, j = ci[1], ci[2]
        (Int(rx[i, j]) > 0 && Int(ry[i, j]) > 0) || continue
        Int(k.arrays["in_ChipSizeMinX"][i, j]) == chip_x || continue
        (cr, cc), (wr, wc) = cpp_rects(k, i, j, chip_x, chip_y, Int(rx[i, j]), Int(ry[i, j]))
        jcr, jcc = chip_bounds(grid, lin[i, j])
        jwr, jwc = search_bounds(grid, lin[i, j])
        # Off the low edge of the image: incomparable by construction, see above. **Both**
        # rectangles are tested, not just the reference's — where the reference's edge truncates to
        # 1 and AutoRIFT.jl's floors to 0 the two differ by one, and testing only one side counts
        # that as a convention error. A few points per Sentinel-2 scene sit exactly there.
        any(<(1), (first(cr), first(cc), first(wr), first(wc),
                   first(jcr), first(jcc), first(jwr), first(jwc))) && continue
        n += 1
        (jcr == cr && jcc == cc) || (bad_chip += 1)
        (jwr == wr && jwc == wc) || (bad_win += 1)
    end
    return n, bad_chip, bad_win
end

@testset "golden harness" begin
    cs = cases()

    @testset "manifest" begin
        @test length(cs) == 22
        @test length(unique(c.product for c in cs)) == 22
        # Phase assignment covers every case; an unassigned phase would silently drop cases from
        # every `--phase` run.
        @test all(c -> c.phase in 3:6, cs)
        @test sum(c -> c.phase == 3, cs) == 9
    end

    cached = filter(have_golden, cs)
    if isempty(cached)
        @info "no golden products cached; run `fetch.jl` first" dir=golden_dir()
    else
        @testset "read $(length(cached)) product(s)" begin
            for c in cached
                p = read_product(c)
                @test !isempty(p.planes)
                @test length(p.x) > 0 && length(p.y) > 0
                # Every product carries these; the radar quartet and `time` do not.
                @test all(haskey(p.planes, v) for v in ("vx", "vy", "v", "v_error"))
                @test schema(p) in (:optical, :radar)
                # The grid is 120 m, north-up: x ascends, y descends.
                length(p.x) > 1 && @test all(>(0), diff(p.x))
                length(p.y) > 1 && @test all(<(0), diff(p.y))
            end
        end

        @testset "self-diff is identical" begin
            for c in cached
                @test identical(compare_products(read_product(c), read_product(c)))
            end
        end

        # A non-square case, so a transposed plane is detectable by shape.
        probe = findfirst(c -> have_golden(c) &&
                              (p = read_product(c); size(p.planes["vx"], 1) != size(p.planes["vx"], 2)),
                          cs)
        if probe === nothing
            @info "no non-square product cached; transpose detection not exercised"
        else
            c = cs[probe]
            p = read_product(c)

            @testset "detects injected differences" begin
                q = read_product(c)
                q.planes["vx"][findfirst(!ismissing, q.planes["vx"])] += Int16(3)
                @test !identical(compare_products(p, q))

                q = read_product(c)
                q.planes["vy"][findfirst(!ismissing, q.planes["vy"])] = missing
                d = compare_products(p, q)
                @test !identical(d)
                @test d.vars[findfirst(v -> v.name == "vy", d.vars)].only_a == 1

                q = read_product(c)
                q.planes["v"] = permutedims(q.planes["v"])
                @test_throws "different grids" compare_products(p, q)

                q = read_product(c)
                q.attribs["vx"]["stable_shift"] = 99.9f0
                d = compare_products(p, q)
                @test !identical(d)
                @test haskey(d.attrib_diffs, "vx.stable_shift")

                q = read_product(c)
                q.x[1] += 120.0
                @test !identical(compare_products(p, q))
            end
        end
    end

    # ---- The correlator input path.
    #
    # `pointset_from_capture` translates the reference's captured arrays into a `PointSet`, and three
    # conventions cross that boundary: the index base, the search-limit rewrite, and the `Dy0` sign.
    # All three are asserted here as *rectangles*, which is the form that makes the assertion
    # closed-form — the chip and window a point cuts are integer ranges, so they are equal or they are
    # not, with no correlation to run and no tolerance to choose. The whole check is milliseconds on a
    # cached capture.
    #
    # Each injection below is a real defect this harness has carried or could carry, and each is
    # invisible in a product comparison because it yields a plausible displacement field rather than
    # an error.
    withcap = filter(c -> isfile(joinpath(capture_dir(c, 200), "call1.json")), cs)
    if isempty(withcap)
        @info "no capture at run 200; the correlator input path is not exercised. " *
              "Build one with `intermediate.jl <case> --run 200`."
    else
        @testset "correlator inputs: $(first(c.product, 28))" for c in withcap
            k = read_capture(c; n = 200)
            grid = pointset_from_capture(k)

            n, bad_chip, bad_win = rect_mismatches(grid, k)
            @test n > 0                    # the sample found base-level points at all
            @test bad_chip == 0            # every chip is cut where the reference cuts it
            @test bad_win == 0             # and every search window

            # **The prior's sign, injected.** The `Dy0` case is the defect that made every case-level
            # figure wrong: the capture records the prior before `arImgDisp_*` flips it, so passing it
            # through unchanged puts the chip `2 * Dy0` rows off. Negating must break the chip check
            # and **leave the window check passing** — the window is centred on the point and carries
            # no prior — and that asymmetry is asserted because it is why the bug survived inspection:
            # the window looks right, and the resulting bias lands in `dx` rather than `dy`.
            #
            # **Only where the prior can move an integer rectangle**, which is a property of the case
            # rather than of the code: negating a prior of 0 changes nothing, and negating one of
            # magnitude 1 shifts by 2 only if it is not already integral to the rectangle. Three radar
            # cases have `Dx0` identically zero or `|Dy0| <= 1` everywhere, so an unconditional
            # assertion here would fail on correct code. The guard is `>= 2`, and it is measured from
            # the capture rather than assumed from the platform.
            for (axis, prior, inject) in
                    ((:dy, grid.dy_prior, p -> rebuild(grid; dy_prior = .-p)),
                     (:dx, grid.dx_prior, p -> rebuild(grid; dx_prior = .-p)))
                if maximum(abs, prior) < 2
                    @info "prior too small to move a rectangle; sign injection skipped" case =
                        first(c.product, 28) axis maxabs = maximum(abs, prior)
                    continue
                end
                _, fchip, fwin = rect_mismatches(inject(prior), k)
                @test fchip > 0
                axis === :dy && @test fwin == 0
            end

            # **The index base.** `+1` converts the reference's 0-based grid, and the tempting `+0.5`
            # — on the grounds that `_shift_points` adds AutoRIFT.jl's own half pixel — is wrong: the
            # two half pixels are one convention counted once on each side. A wrong base moves the
            # chip and the window together, so both must be caught.
            #
            # **`-0.5` is deliberately absent, and that is a fact about the grid rather than a gap.**
            # The captured grid is half-integer everywhere, and `floor(x - 0.5) == floor(x)` when
            # `frac(x) == 0.5`, so subtracting half a pixel cannot move an integer rectangle at all.
            # Asserting it were caught would be asserting something false; the direction that *is*
            # detectable is `+0.5`, which crosses to the next integer. This is exactly why the grid
            # offset had to be settled by scanning agreement rather than by checking rectangles —
            # `correlator.jl` records that scan.
            for delta in (+0.5, -1.0, +1.0)
                shifted = rebuild(grid; x = grid.x .+ delta, y = grid.y .+ delta)
                _, schip, swin = rect_mismatches(shifted, k)
                @test schip > 0
                @test swin > 0
            end

            # **The search-limit rewrite.** `runAutorift` raises every nonzero radius to `minSearch`
            # and zeroes both axes when either is zero (`autoRIFT.py:598-602`), and it is the rewritten
            # array the correlator sees. Using the captured radii directly changes which window is
            # searched, so the window check must reject it — and the chip must be untouched, since it
            # does not depend on the radius.
            #
            # Asserted only where the sampled points include one the rewrite actually changed. On two
            # Sentinel-1 cases every base-level point already asks for at least `minSearch`, so the
            # rewrite is a no-op there and the captured radii *are* the correlator's radii. Counting
            # that as an undetected fault would be wrong: there is no fault to detect.
            raw_x = Int.(k.arrays["in_SearchLimitX"])
            raw_y = Int.(k.arrays["in_SearchLimitY"])
            _, rchip, rwin = rect_mismatches(rebuild(grid; radius_x = raw_x, radius_y = raw_y), k)
            @test rchip == 0
            if rwin == 0
                @info "the search-limit rewrite is a no-op on the sampled points; " *
                      "injection not exercised" case = first(c.product, 28)
            else
                @test rwin > 0
            end
        end
    end
end
