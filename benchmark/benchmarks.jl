# The benchmark suite. Defines `SUITE::BenchmarkGroup`, which is the entry point
# both `run.jl` and AirspeedVelocity expect.
#
# Performance is a tracked deliverable here, not an afterthought: autoRIFT runs on
# tens of millions of image pairs, so a 10% regression in the correlation kernel
# is a material cost. The suite exists before the algorithms do, so that every
# milestone lands with its numbers recorded rather than reconstructed later.
#
# Three tiers, by timescale:
#
#   micro  ns-us   individual kernels. Measured with Chairmarks, which collects
#                  far faster than BenchmarkTools and has better statistics at
#                  this scale, keeping the whole suite runnable in minutes.
#   meso   ms      one chip-size level, one correlation pass.
#   macro  s       end-to-end scenes, batch throughput in pairs/sec, and cold
#                  start.
#
# Groups are added as their milestones land. An empty group is deliberate: it
# keeps the shape of the suite visible and makes a missing benchmark obvious.

using AutoRIFT
using BenchmarkTools
# The geospatial path is benchmarked alongside the array path, so the extensions have to be loaded
# here. Note this is the opposite of the test suite's arrangement, which loads them last to keep
# the core provably extension-free — here there is nothing to protect and one number to compare.
using Rasters
using DimensionalData
using DimensionalData.Lookups
using Chairmarks
using Random
using Statistics

const SUITE = BenchmarkGroup()

# Fixed seeds throughout: benchmark inputs must not vary run to run, or the
# comparison against a baseline measures noise as well as change.
const SEED = 0x5EED

"""
    bench_texture(sz; seed, T)

Band-limited random texture, the standard benchmark input.

Deliberately not white noise. Correlation cost is data-independent, but the
multi-chip-size search's cost is not: the coarse pass zeroes the search radius wherever it
finds no coherent motion, so a scene with realistic texture exercises a
realistic fraction of the grid. White noise would either correlate everywhere or
nowhere and would misreport end-to-end timings.
"""
function bench_texture(sz::Tuple{Int,Int}; seed = SEED, T::Type = Float32)
    rng = MersenneTwister(seed)
    a = randn(rng, Float64, sz)
    for _ in 1:3
        a = _bench_box3(a)
    end
    a .-= minimum(a)
    m = maximum(a)
    m > 0 && (a ./= m)
    return T <: Integer ? round.(T, a .* typemax(T)) : T.(a)
end

bench_texture(n::Integer; kw...) = bench_texture((Int(n), Int(n)); kw...)

function _bench_box3(a::Matrix{Float64})
    out = similar(a)
    nr, nc = size(a)
    @inbounds for j in 1:nc, i in 1:nr
        s, n = 0.0, 0
        for dj in -1:1, di in -1:1
            ii, jj = i + di, j + dj
            if 1 <= ii <= nr && 1 <= jj <= nc
                s += a[ii, jj]
                n += 1
            end
        end
        out[i, j] = s / n
    end
    return out
end

# The three scene sizes used by every macro benchmark. Small enough to iterate
# on, large enough that the 4096 case is representative of a real Landsat subset.
const SCENE_SIZES = (256, 1024, 4096)

# Chip and radius combinations spanning the useful range. The correlation cost
# scales as (2*radius)^2 * chip^2 for a direct evaluation, so this grid also
# determines where the direct/FFT crossover falls — the constant that M2 fixes by
# measurement rather than by theory.
const CHIP_RADIUS = (
    (32, 6), (32, 25), (64, 25), (128, 25), (64, 50),
)

include("suite/points.jl")
include("suite/correlate.jl")
include("suite/window.jl")
include("suite/preprocess.jl")
include("suite/track.jl")
include("suite/multichip.jl")
include("suite/endtoend.jl")
include("suite/throughput.jl")
# Last, and it adds its group only when a device answers: the CPU groups above must be present in
# every run for a baseline to be comparable across machines.
include("suite/gpu.jl")

SUITE
