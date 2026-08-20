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
    @testset "vs OpenCV" begin
        include("opencv.jl")
    end

    @testset "code quality" begin
        include("aqua.jl")
    end
end
