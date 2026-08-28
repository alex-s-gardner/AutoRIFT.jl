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
    @testset "plans" begin
        include("plans.jl")
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
    @testset "tile" begin
        include("tile.jl")
    end
    @testset "api" begin
        include("api.jl")
    end
    @testset "vs OpenCV" begin
        include("opencv.jl")
    end
    # Complex/SLC support. Last of the numerical testsets because it has no fixture corpus behind
    # it — there is no reference implementation of complex correlation to compare against, so
    # every assertion is analytic. See the file header.
    @testset "complex" begin
        include("complex.jl")
    end
    # The sparse first-guess stage. Its filter needs no detector, so it belongs in the core suite;
    # `first_guess` itself needs ImageFeatures and is tested with the extensions below. The
    # "names the dependency" case in here depends on running *before* that load.
    @testset "first guess" begin
        include("firstguess.jl")
    end

    # Real Landsat imagery against ITS_LIVE. Skipped unless the cache is built, and after the
    # synthetic testsets so a failure here reads as "the package disagrees with production" rather
    # than as a broken correlator.
    @testset "real data" begin
        include("realdata.jl")
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
