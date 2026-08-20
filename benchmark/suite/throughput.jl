# Batch throughput, in image pairs per second.
#
# The number that actually matters at production scale, and the one the API was shaped around.
# Everything else in the suite measures one pair; this measures the rate, which is not simply the
# reciprocal because the two parallelism regimes differ in kind:
#
#   intra-pair    one pair at a time, threads split across its search points. Simple, but carries
#                 an irreducible serial fraction — filtering, resampling, the merge — and suffers
#                 load imbalance, because the coarse pass zeroes most of the grid in spatially
#                 clustered patterns and a skipped point is ~400x cheaper than a searched one.
#
#   pair-level    one pair per worker, each reusing its own buffers and FFT plans across pairs via
#                 `reinit!`. No serial fraction, no imbalance, and each worker keeps its own
#                 working set in cache.
#
# Pair-level wins, and by enough to be a design conclusion rather than a tuning note: measured at
# 512^2 with 8 threads, **64.0 pairs/sec intra-pair against 171.6 pair-level, a 2.7x difference**.
# That is why the cache lifecycle is public API rather than an implementation detail, and why
# `autorift!`'s docstring recommends `threaded = false` for batch work.
#
# Reported as a rate because that is how a production run is budgeted: tens of millions of pairs
# against a wall-clock deadline.

let g = addgroup!(SUITE, "throughput")

    # 512 rather than 1024: what this measures is the *ratio* between the regimes, which is already
    # clear at a size where a 16-pair batch takes a fraction of a second. A larger scene would
    # multiply the suite's runtime without changing the conclusion.
    n, npairs = 512, 16

    # A time series rather than one pair repeated. Consecutive acquisitions are what a real batch
    # looks like, and it is also the case where each new reference is the previous secondary — so it
    # exercises the prepared-image reuse in `reinit!` that repeating a single pair would not.
    base = bench_texture((n, n); seed = 1)
    series = [circshift(base, (-2k, 3k)) for k in 0:npairs]
    kw = (; chip_size = 32, search_radius = 25)

    # Threads split within each pair; pairs processed in sequence.
    g["intra-pair $npairs pairs $(n)x$(n)"] = @benchmarkable begin
        for k in 1:$npairs
            AutoRIFT.autorift($series[k], $series[k + 1]; $kw..., threaded = true)
        end
    end

    # One pair per worker, each with its own cache. The regime the API exists for.
    g["pair-level $npairs pairs $(n)x$(n)"] = @benchmarkable begin
        chunks = collect(Iterators.partition(1:$npairs, cld($npairs, Threads.nthreads())))
        Threads.@sync for ch in chunks
            Threads.@spawn begin
                # One cache per worker, built once and advanced through that worker's share.
                cache = AutoRIFT.init($series[first(ch)], $series[first(ch) + 1];
                                      $kw..., threaded = false)
                for k in ch
                    AutoRIFT.reinit!(cache; reference = $series[k], secondary = $series[k + 1])
                    AutoRIFT.autorift!(cache)
                end
            end
        end
    end

    # The serial baseline, so the parallel figures have something to divide by that is not itself
    # parallel. Also the honest number for a single-core worker, which is how a scheduler packing
    # many jobs per node will actually run this.
    g["serial $npairs pairs $(n)x$(n)"] = @benchmarkable begin
        for k in 1:$npairs
            AutoRIFT.autorift($series[k], $series[k + 1]; $kw..., threaded = false)
        end
    end
end
