@testset "Symbol resolution" begin
    p = AutoRIFT.params()
    # A tuple, one measure per chip-size level; a scalar keyword becomes a 1-tuple.
    @test p.similarity === (ZNCC(),)
    @test p.preprocess == Highpass(5)
    @test p.subpixel == PyramidRefine(64)
    @test p.threaded === AutoRIFT.False()

    @test AutoRIFT.params(; similarity = :ncc).similarity === (NCC(),)
    @test AutoRIFT.params(; similarity = :coherence).similarity === (Coherence(),)
    @test AutoRIFT.params(; subpixel = :none).subpixel === NoRefine()
    @test AutoRIFT.params(; preprocess = :none).preprocess === NoPreprocess()

    # A method object is accepted wherever a Symbol is, and is the only way to
    # pass a method-specific parameter.
    @test AutoRIFT.params(; similarity = NCC()).similarity === (NCC(),)
    w = Wallis(; width = 21, min_std = 0.1)
    @test AutoRIFT.params(; preprocess = w).preprocess === w

    # `filter_width` applies to whichever windowed method was named.
    @test AutoRIFT.filter_width(AutoRIFT.params(;
        preprocess = :wallis, filter_width = 21).preprocess) == 21
    @test AutoRIFT.filter_width(AutoRIFT.params(;
        preprocess = :sobel, filter_width = 7).preprocess) == 7
    # ...and is ignored by methods that have no window, rather than erroring.
    @test AutoRIFT.filter_width(AutoRIFT.params(;
        preprocess = :decibel, filter_width = 21).preprocess) == 0

    @test AutoRIFT.upsampling(AutoRIFT.params(;
        subpixel = :pyramid, upsampling = 32).subpixel) == 32
    @test AutoRIFT.upsampling(NoRefine()) == 1
end

@testset "search radius, both axes" begin
    # The x and y radii are independent: Geogrid projects the a-priori velocity
    # onto each image axis separately, so a glacier flowing along x gets a wide
    # x-radius and a narrow y-radius. Three spellings of one extent.
    @test AutoRIFT.params(; search_radius = 25).search_radius == (X = 25, Y = 25)
    @test AutoRIFT.params(; search_radius = (25, 10)).search_radius == (X = 25, Y = 10)
    @test AutoRIFT.params(; search_radius = (X = 25, Y = 10)).search_radius == (X = 25, Y = 10)

    # Zero in one axis searches along the other only, which a per-pixel radius field routinely
    # asks for; zero in both means nothing can be searched.
    @test AutoRIFT.params(; search_radius = (25, 0)).search_radius.Y == 0
    @test_throws "zero in both axes" AutoRIFT.params(; search_radius = 0)
    @test_throws "must be >= 0" AutoRIFT.params(; search_radius = (-1, 10))

    # A named tuple with the wrong field names is a mistake worth naming, not a silent reorder.
    @test_throws "must be `X` and `Y`" AutoRIFT.params(; search_radius = (x = 25, y = 10))
end

# Search-radius normalisation is tested in test/points.jl, alongside the
# `PointSet` it operates on.

@testset "chip-size levels" begin
    # Levels are chip_size * 2^k within [min, max], ascending. Ascending order
    # is load-bearing downstream: each level only writes where no finer level
    # succeeded, so the smallest chip that works wins.
    xs(p) = [c.X for c in AutoRIFT.chip_sizes(p)]
    @test xs(AutoRIFT.params(; chip_size = 32)) == [32, 64, 128]
    @test xs(AutoRIFT.params(; chip_size = 32, chip_size_max = 32)) == [32]
    @test xs(AutoRIFT.params(; chip_size = 16, chip_size_max = 128)) == [16, 32, 64, 128]

    # Both axes double together, so a non-square chip keeps its aspect at every level. That is what
    # lets `MultichipResult.chip_size` record the x extent alone and still name the level.
    lv = AutoRIFT.chip_sizes(AutoRIFT.params(; chip_size = (X = 16, Y = 32),
                                            chip_size_max = (X = 64, Y = 128)))
    @test lv == [(X = 16, Y = 32), (X = 32, Y = 64), (X = 64, Y = 128)]
    @test all(c -> c.Y == 2c.X, lv)

    # A scalar means square.
    @test AutoRIFT.params(; chip_size = 32).chip_size_min == (X = 32, Y = 32)
    @test AutoRIFT.extent(32) == (X = 32, Y = 32)
    @test AutoRIFT.extent((16, 32)) == (X = 16, Y = 32)
    @test AutoRIFT.extent((X = 16, Y = 32)) == (X = 16, Y = 32)
end

@testset "chip-size levels must keep one aspect ratio" begin
    # Levels are `chip_size .* 2^k` in both axes at once, so `chip_size_max` has to be the same
    # multiple of `chip_size` in each. Otherwise x would reach its maximum after a different number
    # of doublings than y, and the chip's aspect would drift across levels -- which would break the
    # nesting `src/multichip.jl` depends on, where a coarse grid point is exactly a block of fine
    # ones.
    #
    # Non-square is fine; a *changing* aspect is not. These two cases are the distinction.
    @test AutoRIFT.params(; chip_size = (X = 16, Y = 32),
                          chip_size_max = (X = 32, Y = 64)) isa AutoRIFT.Params
    @test_throws "same multiple" AutoRIFT.params(; chip_size = (X = 16, Y = 16),
                                                 chip_size_max = (X = 128, Y = 64))

    # Applying the per-axis checks alone does not catch that: both cases pass them individually.
    @test_throws "power-of-two multiple" AutoRIFT.params(; chip_size = 16, chip_size_max = 48)
    @test_throws "must not exceed" AutoRIFT.params(; chip_size = 64, chip_size_max = 32)

    # Multiple of 4 per axis, so a chip centre lands on a pixel boundary.
    @test_throws "multiple of 4" AutoRIFT.params(; chip_size = (X = 16, Y = 18))
    @test_throws "multiple of 4" AutoRIFT.params(; chip_size = 18)
end

@testset "validation" begin
    @test_throws "not recognised" AutoRIFT.params(; similarity = :xcorr)
    @test_throws "not recognised" AutoRIFT.params(; preprocess = :sharpen)
    # The message must list the valid options, not just reject the input.
    @test occursin(":zncc", sprint(showerror,
        try AutoRIFT.params(; similarity = :nope) catch e; e end))

    @test_throws "must be a Symbol" AutoRIFT.params(; preprocess = 5)
    @test_throws "must be a Symbol" AutoRIFT.params(; similarity = 1.0)

    @test_throws "multiple of 4" AutoRIFT.params(; chip_size = 30)
    @test_throws "power-of-two multiple" AutoRIFT.params(;
        chip_size = 32, chip_size_max = 96)
    @test_throws "must not exceed" AutoRIFT.params(;
        chip_size = 128, chip_size_max = 64)

    @test_throws "power of 2" PyramidRefine(; upsampling = 48)
    @test_throws "must be > 1" PyramidRefine(; upsampling = 1)

    @test_throws "must be odd" AutoRIFT.params(; outlier_window = 4)
    @test_throws "must be odd" AutoRIFT.params(; fill_window = 2)
    @test_throws "must be odd" Wallis(; width = 6)
    @test_throws "must be >= 3" Highpass(; width = 1)

    @test_throws "must be >= 0" AutoRIFT.params(; search_radius = (-5, 10))
    @test_throws "in [0, 1]" AutoRIFT.params(; agree_tolerance = 1.5)
    @test_throws "in [0, 1]" AutoRIFT.params(; min_agree_fraction = -0.1)
    @test_throws "must be positive" AutoRIFT.params(; mad_scale = 0)
    @test_throws "must be >= 0" Wallis(; min_std = -1)
end

@testset "Params is concrete" begin
    # A field of abstract type would make every downstream kernel type-unstable,
    # so this is a structural guarantee, not a style preference.
    #
    # Checked with a rotation method as well as without. `RotationSearch` carries a tuple type
    # parameter, and the whole reason it is a *tuple* rather than a `Vector` is this property: a
    # `Vector` field is concrete but **not** isbits, which would stop `Params` storing inline.
    # Measured, that parameterisation buys 1.000x at chip 32/64/128 — nothing — so the tuple is
    # here for `isbits` and for the trimmed binary, not for speed. `isconcretetype` alone does not
    # catch a regression to `Vector`; `isbitstype` does, which is why both are asserted.
    for p in (AutoRIFT.params(), AutoRIFT.params(; rotation = true),
              AutoRIFT.params(; rotation = (-6, -3, 0, 3, 6)))
        @test isconcretetype(typeof(p))
        @test isbitstype(typeof(p))
        for name in fieldnames(typeof(p))
            @test isconcretetype(fieldtype(typeof(p), name))
        end
    end
end

@testset "booltype" begin
    @test AutoRIFT.booltype(true) === AutoRIFT.True()
    @test AutoRIFT.booltype(false) === AutoRIFT.False()
    @test AutoRIFT.booltype(AutoRIFT.True()) === AutoRIFT.True()
    @test AutoRIFT.istrue(AutoRIFT.True())
    @test !AutoRIFT.istrue(AutoRIFT.False())
    # The return annotation on `booltype` is what keeps this inferrable; without
    # it the threaded/serial dispatch downstream goes unstable.
    @test inferred_type(AutoRIFT.booltype, (Bool,)) <: AutoRIFT.BoolAsType
end

@testset "nokw sentinel" begin
    # `nokw` must be distinguishable from `nothing`, because several keywords
    # accept `nothing` as a meaningful value.
    @test AutoRIFT.isnokw(AutoRIFT.nokw)
    @test !AutoRIFT.isnokw(nothing)
    @test !AutoRIFT.isnokw(5)
    @test AutoRIFT.isnokwornothing(nothing)
    @test AutoRIFT.isnokwornothing(AutoRIFT.nokw)
    @test !AutoRIFT.isnokwornothing(5)
end

@testset "an extent is C-layout compatible" begin
    # A C binary API is a requirement, and `Extent` is what the geometry fields are made of — so its
    # layout is part of the contract, not an implementation detail. A `NamedTuple` was chosen over a
    # struct precisely because it keeps this property.
    E = AutoRIFT.Extent
    @test isbitstype(E)
    @test sizeof(E) == 2 * sizeof(Int)
    # No padding, and x before y: the layout a `struct { int64_t x, y; }` has.
    @test fieldoffset(E, 1) == 0
    @test fieldoffset(E, 2) == sizeof(Int)
    # The names live in the type, not the data, which is why they are free.
    @test reinterpret(Tuple{Int,Int}, [AutoRIFT.extent((16, 32))])[1] === (16, 32)

    # And `Params` stays `isbitstype`, which `src/types.jl` records as load-bearing for the
    # trimmed binary: it is what lets the struct store inline with no dispatch.
    @test isbitstype(typeof(AutoRIFT.params()))
end
