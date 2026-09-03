# The correlation kernels on an Apple GPU, through Metal.jl.
#
# This file is the vendor adapter and nothing else: the kernels and the pass driver live in
# `ext/gpu/`, written against KernelAbstractions and `AbstractFFTs`, so a second vendor is another
# file of this shape rather than another copy of them. Julia extensions cannot depend on one
# another, but they can `include` a shared file, which is what makes that arrangement work.
#
# ---------------------------------------------------------------------------
# What the device buys, and where it does not
# ---------------------------------------------------------------------------
#
# Measured on an M2 Max (38 GPU cores) against this package's CPU path — see
# `docs/gpu-feasibility.md` for the full tables:
#
#   * The batched transform is 8-17x FFTW's throughput at batch 1024 and above, and *slower* below
#     batch 256. So a pass with few searchable points falls back to the CPU, which is a routine
#     case rather than a corner: the sparse search deliberately zeroes most of the grid.
#
#   * The sub-pixel cascade, not the transform, is the largest share of a point at the geometry
#     ITS_LIVE runs — 78 us of 97 at chip 16, and fixed in chip size. It is on the device for that
#     reason; a GPU that accelerated only the transform would be capped at 1.2x there.
#
# Apple GPUs have **no `Float64`**, which collides with `src/integral.jl`'s requirement that the
# summed-area tables be `Float64`. The resolution is to remove the cancellation rather than absorb
# it — the gather subtracts the search window's own mean, which is exact for both the numerator and
# the variance — and to carry the sums as `Float32` pairs. Measured as accurate as `Float64`, and
# the resulting numerator is *more* accurate than the CPU's, since a centered window keeps a large
# DC term out of a `Float32` transform.
#
# Requires Metal.jl 1.10, which is where the batched FFT this depends on landed.

module AutoRIFTMetalExt

using AutoRIFT: AutoRIFT
using Metal: Metal, MtlArray
using KernelAbstractions: KernelAbstractions
using LinearAlgebra: mul!

# The transform interface, reached through Metal rather than by depending on `AbstractFFTs`
# directly: an extension may only `using` its own trigger packages and the package's own
# dependencies, and `AbstractFFTs` is neither. Metal re-exports these from its own FFT support, and
# taking them from there is also what pins the requirement — the names exist only from Metal 1.10.
using Metal: plan_rfft, plan_brfft

# The kernels, the buffers and the pass driver, all vendor-neutral.
include(joinpath(@__DIR__, "gpu", "plans.jl"))
include(joinpath(@__DIR__, "gpu", "kernels.jl"))
include(joinpath(@__DIR__, "gpu", "pass.jl"))

# ---------------------------------------------------------------------------
# The vendor adapter
# ---------------------------------------------------------------------------

# Host array to device. `MtlArray` rather than `adapt`, because the array type is the one thing this
# file exists to name; `collect` first so a `SubArray` of the caller's geometry vectors arrives
# contiguous, which `MtlArray` requires.
_to_device(::AutoRIFT.MetalGPU, a::AbstractArray) = MtlArray(collect(a))
# Already resident: no copy. A caller working in `MtlArray`s throughout — the shape a driver holding
# imagery on the device wants — pays no transfer at all.
_to_device(::AutoRIFT.MetalGPU, a::MtlArray) = a

# The pass, wired to the backend `Params` selected. This is the method `src/track.jl`'s stub stands
# in for until this extension loads.
#
# `_gpu_worth_it` is checked here rather than inside `_gpu_pass!` so the fallback is a call to the
# CPU method rather than a flag threaded through the device code: below a few hundred points the
# batched transform loses to FFTW, and a coarse pass routinely has that few.
function AutoRIFT._dispatch_pass!(b::AutoRIFT.MetalGPU, out::AutoRIFT.DisplacementField, ref,
                                  sec, okmask, pts::AutoRIFT.PointSet{1}, chipx::Int,
                                  chipy::Int, rx::Int, ry::Int, p::AutoRIFT.Params,
                                  measure::AutoRIFT.SimilarityMeasure,
                                  subpixel::AutoRIFT.SubpixelMethod)
    Metal.functional() || throw(ArgumentError(
        "`backend = :metal` was selected but Metal is not functional on this machine. Metal.jl " *
        "needs macOS 14 or later and an Apple GPU; `Metal.functional()` reports why. Use " *
        "`backend = :cpu`."))

    if !_gpu_worth_it(AutoRIFT.nsearchable(pts))
        return AutoRIFT._dispatch_pass!(AutoRIFT.CPU(), out, ref, sec, okmask, pts, chipx,
                                        chipy, rx, ry, p, measure, subpixel)
    end
    return _gpu_pass!(b, out, ref, sec, okmask, pts, chipx, chipy, rx, ry,
                      AutoRIFT.upsampling(subpixel), p, measure)
end

end # module
