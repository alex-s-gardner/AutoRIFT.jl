# The GPU correlator against the CPU one.
#
# Skipped unless a device is functional, the way `test/realdata.jl` skips without its cache, so
# `Pkg.test()` on a machine with no GPU reports a skip rather than a failure.
#
# ---------------------------------------------------------------------------
# What is asserted, and why it is not "identical"
# ---------------------------------------------------------------------------
#
# **`dx` and `dy` are bit-identical**, with two stated exceptions below. `correlation` and `peak_ratio`
# agree to a tolerance, because the numerator is computed by a different transform library and no
# amount of care makes MPSGraph and FFTW reassociate a sum the same way. That asymmetry is the point
# of the gate rather than a weakness in it: a peak's *location* is three orders of magnitude less
# sensitive than its value — measured, a 1e-5 perturbation of the surface moved no peak in 3000 weak
# surfaces — which is the same observation `src/plans.jl` records about FFTW wisdom moving
# `correlation` by 3.6e-7 with `dx`/`dy` untouched.
#
# The two exceptions, both measured rather than allowed for in advance:
#
#   1. **One subpixel step, on a near-tie in the upsampled patch.** ~4 points in 15,000. Verified
#      benign: at half of them the two candidate samples of the *CPU's own* cascade are bit-identical,
#      so neither answer is more correct, and at the rest they differ by ~2e-7. Bounded by one step,
#      so it never compounds.
#
#   2. **Points where the CPU surface is not a correlation coefficient.** On a window that is partly
#      constant the CPU's denominator cancels to near zero while its `Float32` numerator carries the
#      window's DC magnitude, so the ratio escapes `[-1, 1]` — measured up to **3.94**, on 2% of a
#      scene with an interior constant patch. `docs/gpu-feasibility.md` records the mechanism. The
#      device does not have this failure, because it removes the window mean before transforming, so
#      the two genuinely disagree there and the CPU is the wrong one. Those points are excluded by
#      the `[-1, 1]` test rather than papered over with a loose tolerance.

using AutoRIFT: AutoRIFT, params, track, ImagePair, workspace, correlate!

# Each backend is exercised if its package can be loaded *and* the hardware answers. Both halves are
# needed: `Metal` installs and loads on any macOS, while `Metal.functional()` is false on an Intel
# Mac and inside a VM — and the package may be absent entirely on a machine that only runs the CPU
# tests, which must not be an error.
const GPU_BACKENDS = let bs = Tuple{AutoRIFT.Backend,String}[]
    for (mod, backend) in (("Metal", AutoRIFT.MetalGPU()),
                           ("CUDA", AutoRIFT.CUDAGPU()))
        # `@eval import` rather than `Base.require` on a `PkgId`: the package is a weak dependency,
        # so it is in the test target rather than in the package's own deps, and only the ordinary
        # import path resolves it in this environment. A `catch` covers the machine that does not
        # have it at all, which is a skip and not a failure.
        m = try
            @eval (import $(Symbol(mod)); $(Symbol(mod)))
        catch
            continue
        end
        # `functional()` may itself throw on a half-configured driver, which is a skip and not a
        # failure of this package.
        yes = try
            m.functional()
        catch
            false
        end
        yes || continue
        # The extension has to have loaded, or the pass would fall through to the stub that names the
        # missing package and every test below would fail for the wrong reason.
        @test Base.get_extension(AutoRIFT, Symbol("AutoRIFT$(mod)Ext")) !== nothing
        push!(bs, (backend, mod))
    end
    bs
end

if isempty(GPU_BACKENDS)
    @info "No functional GPU backend; skipping the device correlator tests. Load Metal (or CUDA) " *
          "on a machine with a supported device to run them."
else

# A scene with texture at every scale the correlator cares about, plus noise so the surfaces are not
# degenerate anywhere.
function gpu_scene(n = 640; seed = 101, shift = (3, -5), noise = 0.02f0)
    rng = MersenneTwister(seed)
    base = [Float32((i * 7 + j * 13) % 251) / 251 for i in 1:n, j in 1:n] .+
           0.15f0 .* randn(rng, Float32, n, n)
    sec = circshift(base, shift) .+ noise .* randn(rng, Float32, n, n)
    return base, sec
end

# The largest `|correlation|` the CPU surface reaches at each point. Above 1 means its denominator
# cancelled, so the CPU value there is not a correlation coefficient and the two paths are entitled
# to disagree — see the header.
function cpu_in_range(pair, pts, chip, radius)
    ws = workspace(eltype(pair.reference), chip, radius)
    sh = AutoRIFT._shift_points(pts, extent(0))
    ok = trues(length(pts))
    for i in eachindex(sh)
        AutoRIFT.issearchable(sh, i) || continue
        cr, cc = AutoRIFT.chip_bounds(sh, i)
        wr, wc = AutoRIFT.search_bounds(sh, i)
        (checkbounds(Bool, pair.reference, cr, cc) &&
         checkbounds(Bool, pair.reference, wr, wc)) || continue
        s = correlate!(ws, (@view pair.reference[wr, wc]), (@view pair.secondary[cr, cc]),
                       (sh.radius_x[i], sh.radius_y[i]))
        AutoRIFT.degenerate(ws) && continue
        ok[i] = maximum(abs, s) <= 1.0f0 + 1.0f-4
    end
    return ok
end

# One subpixel step: the quantization of the refinement, and the bound on exception 1.
substep(p) = 1 / AutoRIFT.upsampling(p.subpixel)

# Compare a CPU and a GPU pass over the same points, and assert the gate.
function check_pass(pair, pts, pc, pg, chip, radius; label = "")
    a = track(pair, pts, pc)
    b = track(pair, pts, pg)

    # Which points each path attempted, and which it could not measure, must match exactly: those
    # are decided by the searchability, bounds and validity tests, which the device path takes from
    # the same `chip_bounds`/`search_bounds`/`_any_valid` the CPU loop uses.
    @test a.searched == b.searched
    @test count(isnan, a.dx) == count(isnan, b.dx)
    @test findall(isnan, a.dx) == findall(isnan, b.dx)

    both = findall(i -> !isnan(a.dx[i]) && !isnan(b.dx[i]), eachindex(a.dx))
    @test !isempty(both)

    inrange = cpu_in_range(pair, pts, chip, radius)
    cmp = [i for i in both if inrange[i]]

    # Exception 1: a near-tie may move the answer by one subpixel step, no more. Asserted as a
    # *bound* on every point plus a cap on how many are affected, rather than as an exact match with
    # a tolerance — a tolerance would also admit a systematic offset, which is the bug this shape of
    # assertion caught during development (a cascade parity error moved every point by exactly one
    # pixel, which is 64 steps).
    step = substep(pc)
    for i in cmp
        @test abs(a.dx[i] - b.dx[i]) <= step + 1.0f-6
        @test abs(a.dy[i] - b.dy[i]) <= step + 1.0f-6
    end
    moved = count(i -> a.dx[i] != b.dx[i] || a.dy[i] != b.dy[i], cmp)
    @test moved <= max(1, cld(length(cmp), 200))       # measured ~0.03%; 0.5% is generous headroom

    # And the great majority are bit-identical, which is the actual claim. A regression that moved
    # every point by one step would satisfy the bound above and fail here.
    @test count(i -> a.dx[i] == b.dx[i], cmp) >= 0.99 * length(cmp)
    @test count(i -> a.dy[i] == b.dy[i], cmp) >= 0.99 * length(cmp)

    # `correlation` to 1e-5: the transform-library difference, and the tolerance
    # `docs/gpu-feasibility.md` justifies. Relative, since a correlation near zero has no absolute
    # scale worth comparing against.
    for i in cmp
        (isnan(a.correlation[i]) || isnan(b.correlation[i])) && continue
        @test isapprox(a.correlation[i], b.correlation[i];
                       rtol = 1.0f-5, atol = 1.0f-6)
    end

    # `peak_ratio` gets a *distributional* bound rather than a per-point one, for one reason: it
    # reads the whole surface, including shifts where the CPU's denominator has cancelled and the
    # value there is not a correlation at all. A point whose *peak* is a valid correlation may still
    # carry such shifts elsewhere, so the `[-1, 1]` filter above does not exclude them, and the
    # secondary peak may be one of them on either path independently.
    #
    # The tail is worse than `correlation`'s, and the reason is structural rather than incidental: a
    # maximum is dominated by its single largest sample where a mean over `4*rx*ry` samples dilutes
    # it. Measured, the two scenes bracket this — a textured scene gives a maximum relative
    # difference of 6.6e-7 over 4900 points, while the flat-patch scene gives 86 of 4416 points above
    # 1% and a worst case of 58%, all of them where a cancelled shift won the secondary. Bounding the
    # maximum would therefore mean a threshold loose enough to hide any real regression.
    #
    # What the percentile bound does assert is that the kernel computes the *same statistic*: a wrong
    # exclusion box, or `max` in place of the CPU's `>` comparison so a `NaN` propagates, shifts
    # every value at once — which a median catches and a per-point tolerance would only report as
    # thousands of indistinguishable failures.
    pr = [(a.peak_ratio[i], b.peak_ratio[i]) for i in cmp
          if isfinite(a.peak_ratio[i]) && isfinite(b.peak_ratio[i]) && abs(a.peak_ratio[i]) > 1.0f-3]
    if length(pr) >= 100
        rel = sort([abs(x - y) / abs(x) for (x, y) in pr])
        @test rel[cld(length(rel), 2)] < 1.0f-6          # median: tracks the surface difference
        @test rel[cld(95 * length(rel), 100)] < 1.0f-5   # 95th percentile: no systematic shift
        @test count(>(0.01), rel) <= cld(3 * length(rel), 100)   # measured 86 of 4416 at worst
    end
    # The non-finite cases are structural, not numerical: `NaN` means the surface was too small to
    # hold a rival and `Inf` means no rival was positive, and both are decided by counting and sign
    # rather than by arithmetic. So they must land on exactly the same points, with no tolerance.
    @test [i for i in cmp if isnan(a.peak_ratio[i])] == [i for i in cmp if isnan(b.peak_ratio[i])]
    @test [i for i in cmp if isinf(a.peak_ratio[i])] == [i for i in cmp if isinf(b.peak_ratio[i])]
    return a, b
end

@testset "$name" for (backend, name) in GPU_BACKENDS
    kw = (; chip_size = 32, search_radius = 25, grid_spacing = 8, chip_size_max = 32)
    # The vendor extension for *this* backend: the batch-size budget and the fallback threshold live
    # in the shared `ext/gpu/` files, so each extension has its own copy of them.
    ext = Base.get_extension(AutoRIFT, Symbol("AutoRIFT$(name)Ext"))

    @testset "a dense pass agrees with the CPU" begin
        base, sec = gpu_scene()
        pair = ImagePair(base, sec)
        pc = params(; kw...)
        pg = params(; kw..., backend)
        pts = AutoRIFT.scatter(AutoRIFT._build_grid(size(pair), pc))
        check_pass(pair, pts, pc, pg, 32, 25)
    end

    @testset "per-point radii, including points excluded by a zero radius" begin
        # The radius varies per point in production — Geogrid sizes it from an a-priori velocity —
        # and a zero radius is how a point is excluded. The device path must honour both: its
        # surface buffer is sized to the pass maximum, and each point uses only its own corner.
        base, sec = gpu_scene()
        pair = ImagePair(base, sec)
        pc = params(; kw...)
        pg = params(; kw..., backend)
        flat = AutoRIFT.scatter(AutoRIFT._build_grid(size(pair), pc))
        rx = copy(flat.radius_x)
        ry = copy(flat.radius_y)
        for i in eachindex(rx)
            m = i % 5
            rx[i] = (10, 18, 25, 25, 0)[m + 1]
            ry[i] = (25, 18, 10, 25, 0)[m + 1]
        end
        pts = AutoRIFT.rebuild(flat; radius_x = rx, radius_y = ry)
        a, b = check_pass(pair, pts, pc, pg, 32, 25)
        # The excluded points are genuinely absent from both, not merely equal.
        @test count(!, a.searched) >= length(a.dx) ÷ 6
    end

    @testset "a radius too small to refine on" begin
        # A surface narrower than the refinement patch cannot supply a neighbourhood, and
        # `subpixel_peak` returns the integer peak unrefined there. The device path must do the same
        # rather than read past the surface: the radius is a per-point field, so a radius below
        # `REFINE_PATCH ÷ 2` is a legal value a caller may set, and on a device an out-of-range read
        # is a `KernelException` rather than a `BoundsError` — which is how this was found.
        base, sec = gpu_scene()
        pair = ImagePair(base, sec)
        pc = params(; kw...)
        pg = params(; kw..., backend)
        flat = AutoRIFT.scatter(AutoRIFT._build_grid(size(pair), pc))
        rx = copy(flat.radius_x)
        ry = copy(flat.radius_y)
        # 1 and 2 give surfaces 2 and 4 across, both under the 5-wide patch; 25 is refinable, so one
        # pass mixes the two paths — which is what a single-radius test could not exercise.
        for i in eachindex(rx)
            m = i % 3
            rx[i] = (1, 2, 25)[m + 1]
            ry[i] = (2, 1, 25)[m + 1]
        end
        pts = AutoRIFT.rebuild(flat; radius_x = rx, radius_y = ry)
        @test AutoRIFT.nsearchable(pts) > 0
        a, b = check_pass(pair, pts, pc, pg, 32, 25)
        # The unrefinable points really are present, and integer-valued on both paths.
        small = [i for i in eachindex(a.dx) if !isnan(a.dx[i]) && rx[i] < AutoRIFT.REFINE_PATCH]
        @test !isempty(small)
        @test all(i -> b.dx[i] == round(b.dx[i]) && b.dy[i] == round(b.dy[i]), small)
    end

    @testset "a constant region yields no measurement" begin
        # A chip with no texture carries no displacement information, so both paths must leave it
        # `NaN` rather than reporting the search-window corner. The threshold is `eps(Float64) * n`
        # on both sides — the device's sums are `Float32` pairs, but the bound names which chips
        # carry signal, which is a property of the imagery and not of the accumulator.
        base, sec = gpu_scene()
        base = copy(base)
        base[200:400, 200:400] .= 0.5f0
        sec = circshift(base, (3, -5))
        pair = ImagePair(base, sec)
        pc = params(; kw...)
        pg = params(; kw..., backend)
        pts = AutoRIFT.scatter(AutoRIFT._build_grid(size(pair), pc))
        a, b = check_pass(pair, pts, pc, pg, 32, 25)
        @test count(isnan, a.dx) > 0                    # the flat interior really is unmeasurable
        @test count(isnan, a.dx) == count(isnan, b.dx)
    end

    @testset "a peak against the search boundary" begin
        # A true shift beyond the search radius rails the peak, and both quality outputs are zeroed
        # there so any positive threshold rejects the point. The device applies the same rule, from
        # its own peak index.
        base, sec = gpu_scene(; shift = (30, -30), noise = 0.0f0)
        pair = ImagePair(base, sec)
        pc = params(; kw..., search_radius = 10)
        pg = params(; kw..., search_radius = 10, backend)
        pts = AutoRIFT.scatter(AutoRIFT._build_grid(size(pair), pc))
        a, b = check_pass(pair, pts, pc, pg, 32, 10)
        @test count(==(0.0f0), a.correlation) == count(==(0.0f0), b.correlation)
        @test count(==(0.0f0), a.correlation) > 0
        # Zero in both quality outputs together, never one alone.
        @test findall(==(0.0f0), b.correlation) == findall(==(0.0f0), b.peak_ratio)
    end

    @testset "NoRefine reports the integer peak" begin
        base, sec = gpu_scene()
        pair = ImagePair(base, sec)
        pc = params(; kw..., subpixel = :none)
        pg = params(; kw..., subpixel = :none, backend)
        pts = AutoRIFT.scatter(AutoRIFT._build_grid(size(pair), pc))
        a, b = check_pass(pair, pts, pc, pg, 32, 25)
        # With no refinement every displacement is a whole number of pixels, on both paths.
        @test all(i -> isnan(b.dx[i]) || b.dx[i] == round(b.dx[i]), eachindex(b.dx))
    end

    @testset "integer element types" begin
        # `UInt8` and `Int16` reach the correlator unwidened, and the gather is what converts them.
        for (T, scale) in ((UInt8, 255), (Int16, 10_000))
            base, sec = gpu_scene()
            b8 = round.(T, clamp.(base .* scale, 0, scale))
            s8 = circshift(b8, (3, -5))
            pair = ImagePair(b8, s8)
            pc = params(; kw...)
            pg = params(; kw..., backend)
            pts = AutoRIFT.scatter(AutoRIFT._build_grid(size(pair), pc))
            check_pass(pair, pts, pc, pg, 32, 25)
        end
    end

    @testset "the result does not depend on the batch size" begin
        # The batch is derived from a memory budget, so it is free to change — which is only true if
        # the answer does not depend on it. Driven from the one knob that controls it, and asserted
        # **bitwise on every output**, since a batching bug is an indexing bug and would show as a
        # wholesale difference rather than a rounding one.
        base, sec = gpu_scene()
        pair = ImagePair(base, sec)
        pg = params(; kw..., backend)
        pts = AutoRIFT.scatter(AutoRIFT._build_grid(size(pair), pg))

        saved = ext.GPU_MEMORY_BUDGET[]
        results = Any[]
        batches = Int[]
        try
            # Two budgets, not more: a very small one drives the batch to a handful of points and
            # so runs hundreds of launches, which costs minutes for no extra coverage — the property
            # under test is that *some* division into several batches changes nothing.
            for budget in (512 * 1024 * 1024, 32 * 1024 * 1024)
                ext.GPU_MEMORY_BUDGET[] = budget
                # A pooled workspace is keyed on its batch size, so the pool must be dropped for a
                # new budget to take effect.
                ext.clear_gpu_workspaces!()
                push!(batches, ext.gpu_batch_size((32, 32), (25, 25)))
                push!(results, track(pair, pts, pg))
            end
        finally
            ext.GPU_MEMORY_BUDGET[] = saved
            ext.clear_gpu_workspaces!()
        end

        # The budgets really did produce different batch counts, or the test proves nothing.
        @test length(unique(batches)) == length(batches)
        @test minimum(batches) < AutoRIFT.nsearchable(pts)
        # `isequal` throughout: an unmeasured point is `NaN`, which is never `==` itself.
        for r in results[2:end]
            @test isequal(r.dx, results[1].dx)
            @test isequal(r.dy, results[1].dy)
            @test isequal(r.correlation, results[1].correlation)
            @test isequal(r.peak_ratio, results[1].peak_ratio)
            @test r.searched == results[1].searched
        end
    end

    @testset "a sparse pass falls back to the CPU" begin
        # Below `GPU_MIN_BATCH` the batched transform loses to FFTW — measured 0.16x at 64 points —
        # so the device path defers. The fallback is *exactly* the CPU path, so the results must be
        # bit-identical in every output: anything else means the pass was not actually handed over,
        # which is the one failure this test exists to catch.
        #
        # The sparse regime is not contrived. The search deliberately zeroes most of the grid, so a
        # coarse pass or a late chip-size level reaches it routinely.
        base, sec = gpu_scene()
        pair = ImagePair(base, sec)
        pc = params(; kw...)
        pg = params(; kw..., backend)
        flat = AutoRIFT.scatter(AutoRIFT._build_grid(size(pair), pc))
        # Zero all but a handful of radii, which is how a point is excluded.
        rx = copy(flat.radius_x)
        ry = copy(flat.radius_y)
        keep = 40
        for i in eachindex(rx)
            if i > keep
                rx[i] = 0
                ry[i] = 0
            end
        end
        pts = AutoRIFT.rebuild(flat; radius_x = rx, radius_y = ry)
        @test AutoRIFT.nsearchable(pts) == keep
        @test keep < ext.GPU_MIN_BATCH               # the regime the fallback exists for

        a = track(pair, pts, pc)
        b = track(pair, pts, pg)
        # `isequal`, not `==`: the non-searched points are `NaN`, and `NaN == NaN` is false, so `==`
        # on these planes is never true however identical they are.
        @test isequal(a.dx, b.dx)
        @test isequal(a.dy, b.dy)
        @test isequal(a.correlation, b.correlation)
        @test isequal(a.peak_ratio, b.peak_ratio)
        @test a.searched == b.searched
    end

    @testset "an unimplemented measure is refused, not silently wrong" begin
        # `NCC` differs by one denominator term and `Coherence` needs complex transforms; neither has
        # a device kernel. Refused by name rather than falling back, since a caller who asked for a
        # measure and got another has no way to tell.
        base, sec = gpu_scene()
        pair = ImagePair(base, sec)
        pg = params(; kw..., similarity = :ncc, backend)
        pts = AutoRIFT.scatter(AutoRIFT._build_grid(size(pair), pg))
        @test_throws "implements `ZNCC` only" track(pair, pts, pg)
    end

    @testset "the whole pipeline runs on the device" begin
        # `autorift` rather than `track`: the chip-size loop, the coarse pass, the outlier filter and
        # the merge all still run on the host, so this is what checks the device pass composes with
        # them. Multiple levels, so a coarse pass that falls back and a fine pass that does not both
        # appear in one run.
        base, sec = gpu_scene()
        pc = params(; chip_size = 32, chip_size_max = 128, search_radius = 25, grid_spacing = 32)
        pg = params(; chip_size = 32, chip_size_max = 128, search_radius = 25,
                    grid_spacing = 32, backend)
        a = autorift(base, sec, pc)
        b = autorift(base, sec, pg)
        @test size(a) == size(b)
        @test AutoRIFT.nmeasured(b) > 0
        @test a.chip_size == b.chip_size
        # The merged field, to the same one-step bound the per-pass gate uses.
        both = findall(i -> !isnan(a.dx[i]) && !isnan(b.dx[i]), eachindex(a.dx))
        @test length(both) >= 0.9 * AutoRIFT.nmeasured(a)
        step = substep(pc)
        for i in both
            @test abs(a.dx[i] - b.dx[i]) <= step + 1.0f-6
            @test abs(a.dy[i] - b.dy[i]) <= step + 1.0f-6
        end
    end
end

end # GPU_BACKENDS
