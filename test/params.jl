@testset "Symbol resolution" begin
    p = AutoRIFT.params()
    @test p.similarity === ZNCC()
    @test p.preprocess == Highpass(5)
    @test p.quantize === QuantizeUInt8()
    @test p.subpixel == PyramidRefine(64)
    @test p.threaded === AutoRIFT.False()

    @test AutoRIFT.params(; similarity = :ncc).similarity === NCC()
    @test AutoRIFT.params(; similarity = :coherence).similarity === Coherence()
    @test AutoRIFT.params(; quantize = :none).quantize === NoQuantize()
    @test AutoRIFT.params(; subpixel = :none).subpixel === NoRefine()
    @test AutoRIFT.params(; preprocess = :none).preprocess === NoPreprocess()

    # A method object is accepted wherever a Symbol is, and is the only way to
    # pass a method-specific parameter.
    @test AutoRIFT.params(; similarity = NCC()).similarity === NCC()
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
    # x-radius and a narrow y-radius. Three spellings, increasing specificity.
    p = AutoRIFT.params(; search_radius = 25)
    @test (p.search_radius_x, p.search_radius_y) == (25, 25)

    p = AutoRIFT.params(; search_radius = (25, 10))
    @test (p.search_radius_x, p.search_radius_y) == (25, 10)

    p = AutoRIFT.params(; search_radius_x = 25, search_radius_y = 10)
    @test (p.search_radius_x, p.search_radius_y) == (25, 10)

    # A per-axis keyword overrides the shared one for that axis only.
    p = AutoRIFT.params(; search_radius = 25, search_radius_y = 10)
    @test (p.search_radius_x, p.search_radius_y) == (25, 10)

    # Zero in one axis is legal as a scalar default (per-pixel fields routinely
    # hold zeros); zero in both means nothing can be searched.
    @test AutoRIFT.params(; search_radius = (25, 0)).search_radius_y == 0
    @test_throws "zero in both axes" AutoRIFT.params(; search_radius = 0)
end

# Search-radius normalisation is tested in test/points.jl, alongside the
# `PointSet` it operates on.

@testset "chip-size levels" begin
    # Levels are chip_size * 2^k within [min, max], ascending. Ascending order
    # is load-bearing downstream: each level only writes where no finer level
    # succeeded, so the smallest chip that works wins.
    @test AutoRIFT.chip_sizes(AutoRIFT.params(; chip_size = 32)) == [32, 64, 128]
    @test AutoRIFT.chip_sizes(AutoRIFT.params(;
        chip_size = 32, chip_size_max = 32)) == [32]
    @test AutoRIFT.chip_sizes(AutoRIFT.params(;
        chip_size = 16, chip_size_max = 128)) == [16, 32, 64, 128]
    # chip_size_min skips the finest levels without changing the level grid.
    @test AutoRIFT.chip_sizes(AutoRIFT.params(;
        chip_size = 16, chip_size_min = 64, chip_size_max = 128)) == [64, 128]

    # Chip height derives from width and is forced even.
    p = AutoRIFT.params(; chip_aspect = 1.0)
    @test AutoRIFT.chip_size_y(p, 32) == 32
    p = AutoRIFT.params(; chip_aspect = 0.5)
    @test AutoRIFT.chip_size_y(p, 32) == 16
    p = AutoRIFT.params(; chip_aspect = 0.7)
    @test iseven(AutoRIFT.chip_size_y(p, 32))
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
    @test_throws "multiple of 4" AutoRIFT.params(; chip_size_min = 18)
    @test_throws "power-of-two multiple" AutoRIFT.params(;
        chip_size = 32, chip_size_max = 96)
    @test_throws "must be <=" AutoRIFT.params(;
        chip_size = 32, chip_size_min = 128, chip_size_max = 64)

    @test_throws "power of 2" PyramidRefine(; upsampling = 48)
    @test_throws "must be > 1" PyramidRefine(; upsampling = 1)

    @test_throws "must be odd" AutoRIFT.params(; outlier_window = 4)
    @test_throws "must be odd" AutoRIFT.params(; fill_window = 2)
    @test_throws "must be odd" Wallis(; width = 6)
    @test_throws "must be >= 3" Highpass(; width = 1)

    @test_throws "must be >= 0" AutoRIFT.params(; search_radius_x = -5)
    @test_throws "in [0, 1]" AutoRIFT.params(; agree_tolerance = 1.5)
    @test_throws "in [0, 1]" AutoRIFT.params(; min_agree_fraction = -0.1)
    @test_throws "must be positive" AutoRIFT.params(; mad_scale = 0)
    @test_throws "must be >= 0" Wallis(; min_std = -1)
end

@testset "Params is concrete" begin
    # A field of abstract type would make every downstream kernel type-unstable,
    # so this is a structural guarantee, not a style preference.
    p = AutoRIFT.params()
    @test isconcretetype(typeof(p))
    for name in fieldnames(typeof(p))
        @test isconcretetype(fieldtype(typeof(p), name))
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
    @test Base.infer_return_type(AutoRIFT.booltype, (Bool,)) <: AutoRIFT.BoolAsType
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
