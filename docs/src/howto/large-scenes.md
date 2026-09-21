# Correlating scenes larger than memory

A pass over a whole scene allocates arrays the size of the scene. `process_block_size` correlates it
a block at a time instead, which bounds peak memory without changing the answer — the result is
bit-identical to the untiled call, and only peak memory, the block shape, buffer reuse and the
threading shape differ.

```julia
out = autorift(image1, image2; chip_size = 32, process_block_size = (1024, 1024))
```

Worth it whenever peak memory matters, not only when the scene cannot fit.

## Choosing a block size

**Peak is set by a block's area, not by the number of blocks.** Nine arrays sized to the largest read
window are held per task, and the task count is capped at `min(nblocks, nthreads)`, so the footprint
is area × threads however finely the scene is cut.

Measured total process peak on a 17121 × 16961 Landsat overlap at 10 threads, from a memory-mapped
input, against an untiled run of the same scene:

| block | blocks | peak | runtime | read amplification |
|---|---:|---:|---:|---:|
| untiled | 1 | 4958 MiB | 20.5 s | 1.00× |
| 2048 px | 81 | 3116 MiB | 24.4 s | 1.13× |
| 1024 px | 289 | **2140 MiB** | 22.6 s | 1.26× |
| 512 px | 1089 | 2248 MiB | 22.3 s | 1.55× |
| 256 px | 4160 | 2042 MiB | 24.0 s | 2.21× |

2140, 2248 and 2042 MiB across a 14× range of block counts is the area rule in one line.
`(1024, 1024)` is the general recommendation: near the floor on peak, and past the point where
shrinking further buys anything.

## Runtime is set by the halo

Every block reads beyond its own edges, because a chip near the boundary needs the search window
around it. That fixed-width margin is the **halo**, and the imagery a run reads grows as

```
((block + 2·halo) / block)²
```

which is the fourth column above, at a 69-pixel halo. Below 1024 px the memory curve has flattened
while that redundancy keeps climbing — and pushed far enough it stops being merely wasteful: each
block read allocates a block-sized temporary, so read amplification is also allocation rate, and a
128-pixel block on this scene becomes GC-bound rather than compute-bound.

**A wide halo therefore needs a proportionally larger block.** A block only a few halos across is
mostly overlap, and one smaller than its own halo is rejected rather than silently clipped.

```julia
halo(grid, p, size(image1))   # the margin this configuration needs, in pixels, per axis
```

The halo need not be square. A configuration with a wide search radius in x and a narrow one in y has
an asymmetric halo, and a block shaped to follow it beats a square block of the same area on peak,
runtime and thread occupancy at once. Size the two axes separately against `halo`; that is the floor,
not the target.

## Keep the block count above the thread count

Fewer blocks than threads puts every block in flight at once. On the scene above at 8192 px that held
nine 8331² working sets for a peak of **14856 MiB — 3× the untiled run**. Going too large is worse
than not blocking at all.

Blocks are also the unit of threaded work and their cost varies by orders of magnitude, so a small
pool waits on its slowest member. Measured occupancy of ten threads:

| blocks/thread | 10 | 38 | 76 | 148 | 230 | 250 | 581 |
|---|---:|---:|---:|---:|---:|---:|---:|
| threads busy | 2.3 | 4.6 | 6.6 | **8.9** | **9.0** | 7.8 | 9.0 |

Aim for 100–250. The 581 row reaches full occupancy at 21.6× read amplification, which is not a trade
worth making.

A block is a whole number of grid points, so the size is a target that snaps **outward**: at
`grid_spacing = 32` a request of 500 becomes 512. Pass a tuple and only a tuple — a full-width band
costs halo on two sides where a square block pays it on four, so the shape is yours to choose rather
than something to infer from a scalar.

A preprocessing filter that estimates from the whole image cannot be reproduced block by block, and
is rejected rather than approximated — see [`AutoRIFT.filter_reach`](@ref).

## Reading a pair that is still on disk

For input already in memory, including a memory-mapped array, nothing scene-sized is formed at all.
For input still on disk, a block's window is read once per pass — three chip sizes, coarse and
fine — which on a 15901 × 13435 GeoTIFF pair is 3.4–4.4× the scene over a run, against 0.66–0.96× for
a single pass. No block size removes that.

`cache_budget` decides whether to read the pair into memory once instead.

```julia
out = autorift(r1, r2; process_block_size = (1024, 1024), cache_budget = :auto)
```

`:auto`, the default, compares two volumes computable before anything is read: the pair's own bytes
against the sum of every block's read window, once per pass. It caches when the first is smaller, and
is capped at an eighth of free memory — divided by the thread count for `threaded = false`, the batch
shape where each task decides for itself — so a run competes for a share of what is free rather than
the last of it.

**Either the whole pair is cached or none of it is.** There is no partial cache to size, because a
blocked run sweeps every block once per pass: a chunk's next use is a whole sweep away, so a cache
below the whole working set has evicted it by then. Simulated on the 15901 × 13435 pair at blocks of
1024, read volume is 3.4× the scene at half the working set and 5.7× at a twentieth, falling below 1×
only at the whole of it.

What it costs and buys: caching a 0.398 GiB pair raised peak from **3.6 to 4.2 GiB** at blocks of
512 px, for **15.2 s against 13.5 s** at 10 threads. The gain narrows as the block grows — 14.1 s
either way at 2048 px — since a larger block reads less to begin with.

- Set a **byte count** on a machine whose free memory is a poor guide to what this process may take: a
  container with a cgroup limit, or a host shared with work Julia cannot see.
- Set **`nothing`** to hold peak to the blocks alone.
- **Memory-map the input** to keep the raw pair off the heap entirely. A resident input is never
  copied, and the kernel can reclaim its pages under pressure where an `Array` cannot.

Input already in memory is never copied whatever this says, and the result does not depend on it.

## What reading from disk costs

The answer is **bit-identical** to the same pair materialized, at any block size, on all five layers —
a windowed read that computed something subtly different would be worse than one that was merely slow.

Measured on a Landsat 8/9 pair over Jakobshavn — 17121 × 16961 `Float32`, 4.48 M grid points, 10
threads — reading from the two GeoTIFFs peaks at **5.4 GiB against 9.0 GiB** for the same run from
memory, in **41.3 s against 39.4 s**. Both measure the same 915,488 points.

The gap is smaller than the raw pair because the lazy run holds that pair too. A `UInt16` or `UInt8`
scene, which is what Landsat level-1 imagery ships as, holds a half or a quarter as much.

## Budget for both

Peak is the blocks plus, when it is taken, the raw pair. Size the blocks against the halo on each
axis, then decide whether the pair's own bytes are worth the runtime at small blocks.

The full measured record — three ways to measure peak memory wrong, where the floor comes from, and
the per-granule sweeps — is in [Memory](@ref).
