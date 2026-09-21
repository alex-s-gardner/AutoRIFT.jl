
# Correlating many pairs

A single pair is one call. A time series, or a tile of a large production run, is many — and there the
question is how to organize the work rather than how to make one correlation faster.

Three things matter, in this order: run **one pair per task** rather than threading inside a pair; reuse a
[`AutoRIFT.Cache`](@ref) to hold the allocations down; and know that neither changes the answer.

## One pair per task

The parallelism to reach for first is across pairs, not within one:

```@example batch
using AutoRIFT
include("../../figures.jl")  # plotting helpers and the synthetic scenes; see [Plotting](@ref)

pairs = [warped_pair(256, (row, col) -> (3.0, 1.0); seed = s)[1:2] for s in 1:4]
settings = (; chip_size = 16, chip_size_max = 32, search_radius = 12, grid_spacing = 8)

results = Vector{Any}(undef, length(pairs))
Threads.@sync for (i, (reference, secondary)) in enumerate(pairs)
    Threads.@spawn results[i] = autorift(reference, secondary; settings..., threaded = false)
end
map(AutoRIFT.nmeasured, results)
```

Note `threaded = false` inside the task. Measured on sixteen 512² pairs across eight threads, one pair per
task with intra-pair threading off runs **1.32× faster** than processing pairs one at a time with
`threaded = true`. Per-pair threading has a serial fraction to bound it and load imbalance from the sparse
search; across pairs there is neither, and each task keeps its own working set in cache.

`threaded = true` is for the case where one pair is all you have. It does not change the result — the
threaded and serial paths agree exactly.

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

Use `isequal` rather than `==` — unmeasured points are `NaN`, and `NaN == NaN` is false, so `==` reports a
difference between a result and itself.

## What the cache actually buys, and what it does not

**It is not a wall-clock speedup.** Measured on 512² pairs at `grid_spacing = 8`, building the cache costs
about 2.7 ms while the correlation costs about 340 ms — so a walk of five consecutive pairs runs at
**1.00×** either way, and so does a fan-out of sixteen pairs across one, two, four, or eight tasks.

What it buys is **allocation**: 4.7 MiB per pair fresh against 1.4 MiB reused, a factor of **3.3**. That
is what matters when many tasks run at once, since allocation rate is shared and garbage collection
stops every thread.

`init`'s cost is roughly flat while the correlation's scales with the number of points, so the cache's
share of the work grows as the grid gets coarser:

```@example batch
big_reference, big_secondary, _, _ = warped_pair(512, (row, col) -> (3.0, 1.0); seed = 21)
for spacing in (8, 16, 32, 64)
    out = autorift(big_reference, big_secondary; settings..., grid_spacing = spacing,
                   threaded = false)
    println("grid_spacing ", lpad(spacing, 2), "   measured ", lpad(AutoRIFT.nmeasured(out), 4))
end
```

Correlation cost tracks those point counts — 334 ms, 81 ms, 22 ms, 6.2 ms — while `init` stays near
2.6 ms throughout. Its share of the total therefore climbs 0.8%, 3.0%, 10.5%, 28.7%. A coarse grid over
many pairs is where reuse is worth arranging; a dense grid is dominated by correlation and the cache is a
memory optimization only.

## Walking a time series

Consecutive acquisitions share an image — each new reference is the previous secondary. `reinit!` reuses an
image already prepared in *either* slot, so a walk filters each acquisition once rather than twice:

```@example batch
# One texture, progressively warped: consecutive frames genuinely correlate.
frames = [warped_pair(256, (row, col) -> (3.0 * (k - 1), 1.0 * (k - 1)); seed = 5)[2] for k in 1:4]

walk = AutoRIFT.init(frames[1], frames[2]; settings..., threaded = false)
counts = [AutoRIFT.nmeasured(autorift!(walk))]
for k in 2:(length(frames) - 1)
    reinit!(walk; reference = frames[k], secondary = frames[k + 1])
    push!(counts, AutoRIFT.nmeasured(autorift!(walk)))
end
counts
```

Omitting an argument keeps the current image, so `reinit!(cache; secondary = next)` advances one image
while holding the other fixed — the right form when every pair shares a common reference.

A mask follows its image. An omitted mask is kept only if its image was also omitted, and otherwise
re-derived from finiteness: carrying a mask onto a different image would mask the wrong pixels.

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

One cache per task, never one shared across tasks — a cache holds mutable buffers and is not thread-safe.

## Three things that will surprise you

- **`autorift!` twice returns the same object**, not a recomputation. It is the cached result until the
  next `reinit!`. If you need to keep a result past the next pair, copy what you need out of it: the
  arrays are reused.
- **Image size is fixed at `init`.** `reinit!` throws on a change of shape, because the grid and output
  arrays are sized to it. A different size needs a new cache.
- **`Params` is not reinitialised**, which matters for exactly one keyword. [`RotationSearch`](@ref)'s
  `about` is a per-pair quantity, so a cache built with `about = scene_rotation(guess)` applies the first
  pair's rotation to every later one. For a rotating series, call `autorift` per pair with a freshly
  fitted `about`. See [Giving the search a first guess](@ref).

## See also

- [Correlating scenes larger than memory](@ref) — blocking, which composes with everything here
- [`AutoRIFT.init`](@ref), [`reinit!`](@ref), [`autorift!`](@ref) — the three docstrings
- [Memory](@ref) — the measured record, including what a cache holds resident
