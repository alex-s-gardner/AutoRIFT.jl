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

include("manifest.jl")
include("product.jl")
include("compare.jl")

using Test

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
end
