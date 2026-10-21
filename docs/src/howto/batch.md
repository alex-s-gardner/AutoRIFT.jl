
# Correlating many pairs

A single pair is one call. A time series, or a tile of a large production run, is many calls, and the
question is how to organize them. Run one pair per task rather than threading inside a pair, and reuse a
[`AutoRIFT.Cache`](@ref) to reduce allocation. Neither changes the result.

## One pair per task

Parallelize across pairs rather than within one:

```@example batch
using AutoRIFT
include("../../figures.jl")  # plotting helpers and the example scenes; see [Plotting](@ref)

pairs = [warped_pair(photo(), 256, (row, col) -> (Float64(s), 1.0))[1:2] for s in 1:4]
settings = (; chip_size = 16, chip_size_max = 32, search_radius = 12, grid_spacing = 8)

results = Vector{Any}(undef, length(pairs))
Threads.@sync for (i, (reference, secondary)) in enumerate(pairs)
    Threads.@spawn results[i] = autorift(reference, secondary; settings..., threaded = false)
end
map(AutoRIFT.nmeasured, results)
```

Each task sets `threaded = false`. On sixteen 512² pairs across eight threads, one pair per task with
intra-pair threading off ran 1.32× faster than processing the pairs one at a time with `threaded = true`.
Threading within a pair has a serial fraction and load imbalance from the sparse search. Across pairs
there is neither, and each task keeps its own working set in cache.

Use `threaded = true` when one pair is all there is. The threaded and serial paths give identical results.

## Reusing a cache

[`AutoRIFT.init`](@ref) builds the grid, the output arrays, and the FFT plans; [`reinit!`](@ref) points
them at a new pair; [`autorift!`](@ref) runs it.

```@example batch
cache = AutoRIFT.init(pairs[1]...; settings..., threaded = false)
first_result = autorift!(cache)

reinit!(cache; reference = pairs[2][1], secondary = pairs[2][2])
second_result = autorift!(cache)

(first = AutoRIFT.nmeasured(first_result), second = AutoRIFT.nmeasured(second_result))
```

The answer is identical to the single-call form:

```@example batch
direct = autorift(pairs[2]...; settings..., threaded = false)
(dx = isequal(second_result.dx, direct.dx),
 dy = isequal(second_result.dy, direct.dy),
 correlation = isequal(second_result.correlation, direct.correlation))
```

Compare with `isequal` rather than `==`: unmeasured points are `NaN`, and `NaN == NaN` is false.

## What the cache saves

The cache does not shorten wall-clock time on a dense grid. On 512² pairs at `grid_spacing = 8`, building
the cache costs about 2.7 ms and the correlation about 340 ms. A walk of five consecutive pairs, and a
fan-out of sixteen pairs across one, two, four or eight tasks, run at the same speed with or without it.

It reduces allocation: 4.7 MiB per pair fresh against 1.4 MiB reused, a factor of 3.3. This matters when
many tasks run at once, because allocation rate is shared and garbage collection stops every thread.

The cost of `init` is roughly constant while the correlation cost scales with the number of points, so
the cache's share of the work grows as the grid gets coarser:

```@example batch
big_reference, big_secondary, _, _ = warped_pair(photo(), 512, (row, col) -> (3.0, 1.0))
for spacing in (8, 16, 32, 64)
    out = autorift(big_reference, big_secondary; settings..., grid_spacing = spacing,
                   threaded = false)
    println("grid_spacing ", lpad(spacing, 2), "   measured ", lpad(AutoRIFT.nmeasured(out), 4))
end
```

The correlation cost follows those point counts (334 ms, 81 ms, 22 ms, 6.2 ms) while `init` stays near
2.6 ms, so `init` is 0.8%, 3.0%, 10.5% and 28.7% of the total. Reuse pays off most on a coarse grid over
many pairs. On a dense grid the correlation dominates and the cache only saves memory.

## Walking a time series

In a time series, each new reference is the previous secondary. `reinit!` reuses an image that was already
filtered in either slot, so a walk filters each acquisition once rather than twice:

```@example batch
# One photograph, progressively warped, so consecutive frames correlate.
frames = [warped_pair(photo(), 256, (row, col) -> (3.0 * (k - 1), 1.0 * (k - 1)))[2] for k in 1:4]

walk = AutoRIFT.init(frames[1], frames[2]; settings..., threaded = false)
counts = [AutoRIFT.nmeasured(autorift!(walk))]
for k in 2:(length(frames) - 1)
    reinit!(walk; reference = frames[k], secondary = frames[k + 1])
    push!(counts, AutoRIFT.nmeasured(autorift!(walk)))
end
counts
```

An omitted argument keeps the current image, so `reinit!(cache; secondary = next)` advances one image and
holds the other fixed, which suits pairs that share a reference.

A mask follows its image. An omitted mask is kept only if its image was also omitted. Otherwise it is
derived from finiteness, because a mask carried onto a different image would mask the wrong pixels.

## The production shape

```julia
Threads.@sync for chunk in Iterators.partition(pairs, cld(length(pairs), Threads.nthreads()))
    Threads.@spawn begin
        cache = AutoRIFT.init(first(chunk)...; threaded = false, settings...)
        for (a, b) in chunk
            reinit!(cache; reference = a, secondary = b)
            write_output(autorift!(cache))
        end
    end
end
```

Use one cache per task. A cache holds mutable buffers and is not thread-safe.

## Behavior to know

- Calling `autorift!` twice without a `reinit!` returns the same result object without recomputing. To
  keep a result past the next pair, copy it out: the arrays are reused.
- The image size is fixed at `init`. `reinit!` throws a `DimensionMismatch` on a different shape, because
  the grid and output arrays are sized to it. A different size needs a new cache.
- `reinit!` does not change the cache's `Params`. [`RotationSearch`](@ref)'s `about` is a per-pair
  quantity, so a cache built with `about = scene_rotation(guess)` applies the first pair's rotation to
  every later pair. For a rotating series, call `autorift` per pair with a freshly fitted `about`. See
  [Giving the search a first guess](@ref).

## See also

- [Correlating scenes larger than memory](@ref): blocking, which works with everything here
- [`AutoRIFT.init`](@ref), [`reinit!`](@ref), [`autorift!`](@ref): the three docstrings
- [Memory](@ref): measurements, including what a cache holds resident
