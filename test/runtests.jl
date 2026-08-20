using AutoRIFT
using Test
using Random
using Statistics

include("utils.jl")

# Extension packages are loaded only after the core testsets, so that the core
# genuinely runs extension-free. This is what keeps the fast-load path honest:
# if a core file starts depending on Rasters, these tests fail rather than
# silently passing because Rasters happened to be loaded.
@testset "AutoRIFT.jl" begin
    @testset "test helpers" begin
        include("utils_test.jl")
    end
    @testset "params" begin
        include("params.jl")
    end
    @testset "points" begin
        include("points.jl")
    end
    @testset "fixtures" begin
        include("fixtures_test.jl")
    end
    @testset "correlate" begin
        include("correlate.jl")
    end
    @testset "peak" begin
        include("peak.jl")
    end
    @testset "window" begin
        include("window.jl")
    end
    @testset "outliers" begin
        include("outliers.jl")
    end
    @testset "preprocess" begin
        include("preprocess.jl")
    end
    @testset "track" begin
        include("track.jl")
    end
    @testset "multichip" begin
        include("multichip.jl")
    end
    @testset "api" begin
        include("api.jl")
    end
    @testset "vs OpenCV" begin
        include("opencv.jl")
    end

    @testset "code quality" begin
        include("aqua.jl")
    end

    # Last, and that ordering is the point: everything above ran with neither Rasters nor
    # DimensionalData in the session, so a core file that started depending on one would fail
    # rather than pass because a later test happened to load it.
    @testset "extensions" begin
        include("extensions.jl")
    end
end
