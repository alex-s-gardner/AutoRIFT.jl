# The device correlator, against the same passes `suite/track.jl` measures on the CPU.
#
# Added to `SUITE` only when a device is functional, so a run on a machine without one produces the
# same result set it always did rather than a group of failures. That does mean `compare.jl` sees no
# GPU entries in a baseline recorded elsewhere, which is the intended behaviour: a missing benchmark
# yields no comparison rather than an error.
#
# **Every timing is wrapped in a device synchronize.** Kernel launches are asynchronous, so an
# unsynchronised measurement times the *encode* and reports a correlator that appears to run in
# microseconds. `AutoRIFT.track!` returns before the queue drains; the barrier is what makes the
# number mean anything.
#
# Names mirror the `track` group's — `gpu fine c32 r25 1024x1024` against `fine c32 r25 1024x1024` —
# so a reader can divide the two directly. There is no automatic ratio: `compare.jl` pairs a
# benchmark against the *same name* in another file, which is the right behaviour for a regression
# gate and the wrong one for a CPU/GPU comparison.

let backend = nothing, syncfn = nothing
    # Metal first, then CUDA. `functional()` guards the case where the package loads but the hardware
    # does not answer — an Intel Mac, or a VM — which is a skip rather than a failure.
    for (mod, b) in (("Metal", AutoRIFT.MetalGPU()), ("CUDA", AutoRIFT.CUDAGPU()))
        m = try
            @eval (import $(Symbol(mod)); $(Symbol(mod)))
        catch
            continue
        end
        ok = try
            m.functional()
        catch
            false
        end
        ok || continue
        backend = b
        # `synchronize()` with no argument is the whole-device barrier in both packages.
        syncfn = () -> m.synchronize()
        break
    end

    if isnothing(backend)
        @info "No functional GPU backend; the `gpu` benchmark group is omitted."
    else
        g = addgroup!(SUITE, "gpu")

        for n in (512, 1024)
            ref = bench_texture((n, n); seed = 1)
            sec = bench_texture((n, n); seed = 2)
            pair = AutoRIFT.ImagePair(ref, sec)

            for (cs, r) in ((32, 25), (64, 25))
                pts = AutoRIFT.gridpoints((n, n), 32; chip_size = cs, search_radius = r)
                npts = AutoRIFT.npoints(pts)
                out = AutoRIFT.displacement_field(pts)

                pc = AutoRIFT.params(; subpixel = :none, backend)
                g["gpu coarse c$cs r$r $(n)x$(n) [$npts pts]"] = @benchmarkable begin
                    AutoRIFT.track!($out, $pair, $pts, $pc)
                    $syncfn()
                end

                pf = AutoRIFT.params(; upsampling = 64, backend)
                g["gpu fine c$cs r$r $(n)x$(n) [$npts pts]"] = @benchmarkable begin
                    AutoRIFT.track!($out, $pair, $pts, $pf)
                    $syncfn()
                end
            end
        end

        # A dense grid at one point per 8 px, which is where the correlation is the largest share of a
        # run and so where the device gains most — the ITS_LIVE configuration's shape.
        let n = 1024
            ref = bench_texture((n, n); seed = 1)
            sec = bench_texture((n, n); seed = 2)
            p = AutoRIFT.params(; chip_size = 32, chip_size_max = 128, search_radius = 25,
                                grid_spacing = 8, backend)
            g["gpu autorift dense $(n)x$(n)"] = @benchmarkable begin
                autorift($ref, $sec, $p)
                $syncfn()
            end
        end
    end
end
