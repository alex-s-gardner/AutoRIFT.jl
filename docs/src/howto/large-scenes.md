# Correlating scenes larger than memory

A pass over a whole scene allocates arrays the size of the scene. `process_block_size` correlates the
scene a block at a time instead. This lowers peak memory and gives the same answer: the result is
bit-identical to the untiled call. Only peak memory, the block shape, buffer reuse and the threading
shape differ.

```julia
out = autorift(image1, image2; chip_size = 32, process_block_size = (1024, 1024))
```

Blocking can lower peak memory even when the scene would fit. For a file-backed `Raster`,
`process_block_size` defaults to a size derived from the halo and the file's chunking (see
[`AutoRIFT.block_size_for`](@ref)). For any other input it defaults to `nothing`, one block over the whole
scene. An unblocked run throws an `ArgumentError` for an array that is still on disk
([`AutoRIFT.ondisk`](@ref)), because it would read the array one pixel at a time.

## Choosing a block size

Peak memory is set by a block's area, not by the number of blocks. A threaded run holds nine arrays sized
to the largest read window per task, and the task count is capped at `min(nblocks, nthreads)`, so the
footprint is area × threads however finely the scene is cut. A run with `threaded = false` holds one such
set.

Total process peak on a 17121 × 16961 Landsat overlap at 10 threads, from a memory-mapped input, against an
untiled run of the same scene:

| block | blocks | peak | runtime | read amplification |
|---|---:|---:|---:|---:|
| untiled | 1 | 4958 MiB | 20.5 s | 1.00× |
| 2048 px | 81 | 3116 MiB | 24.4 s | 1.13× |
| 1024 px | 289 | **2140 MiB** | 22.6 s | 1.26× |
| 512 px | 1089 | 2248 MiB | 22.3 s | 1.55× |
| 256 px | 4160 | 2042 MiB | 24.0 s | 2.21× |

Peak stays between 2042 and 2248 MiB across a 14× range of block counts. `(1024, 1024)` is a good general
choice: its peak is near the minimum, and smaller blocks gain nothing further.

## Runtime is set by the halo

A block reads beyond its own edges, because a chip near the boundary needs the search window around it.
That fixed-width margin is the halo. The imagery a run reads grows as

```
((block + 2·halo) / block)²
```

which is the last column above, at a 69-pixel halo. Below 1024 px the memory curve is flat while the
redundant reading keeps growing. Read amplification also raises the allocation rate: a 128-pixel block on
this scene spends its time in garbage collection and had not finished after 20 minutes, against 22.6 s at
1024 px.

A wide halo therefore needs a proportionally larger block. A block only a few halos across is mostly
overlap, and a block smaller than its own halo throws an `ArgumentError`.

```julia
halo(grid, p, size(image1))   # the margin this configuration needs, in pixels, per axis
```

`AutoRIFT.halo(p)` gives the same answer from the parameters alone, without building a grid.

The halo need not be square. A wide search radius in x and a narrow one in y give an asymmetric halo. On a
NISAR L2 granule (halo 2216 × 1103), a `2304 × 1152` block follows it and beats a square block of larger
area on peak, runtime and thread occupancy at once. Size the two axes separately against `halo`. The halo
is the minimum, not the target.

## Keep the block count above the thread count

With fewer blocks than threads, every block is in flight at once. On the scene above at 8192 px, nine
8331² working sets were held at once, for a peak of **14856 MiB, 3× the untiled run**. A block size that is
too large uses more memory than not blocking.

Blocks are also the unit of threaded work, and their cost varies by orders of magnitude, so a small pool
waits on its slowest block. Occupancy of ten threads, by blocks per thread:

| blocks/thread | 10 | 38 | 76 | 148 | 230 | 250 | 581 |
|---|---:|---:|---:|---:|---:|---:|---:|
| threads busy | 2.3 | 4.6 | 6.6 | **8.9** | **9.0** | 7.8 | 9.0 |

Aim for 100–250 blocks per thread. The 581 row reaches full occupancy at 21.6× read amplification, which
costs more than it gains.

A block holds a whole number of grid points, so the size is an upper bound on the block's span: a block
holds as many grid points as fit within the requested pixels. Pass a tuple `(X, Y)` or `(X = …, Y = …)`,
never a scalar. A full-width band pays halo on two sides where a square block pays on four, so the shape
is the caller's choice.

A preprocessing filter that estimates from the whole image cannot be reproduced block by block. It is
rejected with an `ArgumentError`. See [`AutoRIFT.filter_reach`](@ref).

## Reading a pair that is still on disk

For input already in memory, including a memory-mapped array, the filtered scene is never formed. For input
still on disk, each block's window is read once per pass (a coarse and a fine pass at each of three chip
sizes). On a 15901 × 13435 GeoTIFF pair this reads 3.4–4.4× the scene over a run, against 0.66–0.96× for a
single pass. No block size removes the repeat reads.

`cache_budget` decides whether to read the pair into memory once instead.

```julia
out = autorift(r1, r2; process_block_size = (1024, 1024), cache_budget = :auto)
```

`:auto`, the default, compares two volumes that can be computed before anything is read: the pair's own
bytes, and the sum of every block's read window once per pass. It caches the pair when the first is
smaller. The budget is capped at an eighth of free memory, divided by the thread count when
`threaded = false`, since each task then decides for itself.

Either the whole pair is cached or none of it is, so a byte count smaller than the pair caches nothing.
A blocked run sweeps every block once per pass, so a chunk's next use is a whole sweep away, and a cache
smaller than the whole working set has evicted it by then. Replaying the requests for the 15901 × 13435 pair at blocks of 1024 gives a read volume of 3.4× the
scene at half the working set and 5.7× at a twentieth. It falls below 1× only when the whole pair is cached.

Caching a 0.398 GiB pair raised peak from **3.6 to 4.2 GiB** at blocks of 512 px, and cut the runtime from
**15.2 s to 13.5 s** at 10 threads. The gain shrinks as the block grows: at 2048 px the runtime is 14.1 s
either way, since a larger block reads less to begin with.

- Set a byte count when free memory is a poor guide to what this process may use, as in a container with a
  cgroup limit or a host shared with work Julia cannot see.
- Set `nothing` to hold peak to the blocks alone.
- Memory-map the input to keep the raw pair off the heap. A resident input is never copied, and the kernel
  can reclaim a mapped array's pages under pressure, which it cannot do for an `Array`.

Input already in memory is never copied whatever `cache_budget` says, and the result does not depend on it.

## What reading from disk costs

The answer is bit-identical to the same pair held in memory, at any block size.

On a Landsat 8/9 pair over Jakobshavn (17121 × 16961 `Float32`, 4.48 M grid points, 10 threads), reading
from the two GeoTIFFs peaks at **5.4 GiB against 9.0 GiB** for the same run from memory, in **41.3 s
against 39.4 s**. Both measure the same 915,488 points. The gap is smaller than the raw pair because the
file-backed run also holds that pair in memory. A `UInt16` or `UInt8` scene, which is how Landsat level-1
imagery ships, holds a half or a quarter as much.

## Budget for both

Peak is the blocks plus, when it is cached, the raw pair. Size the blocks against the halo on each axis,
then decide whether the pair's own bytes are worth the runtime saved at small blocks.

The full measured record, including how to measure peak memory correctly and where the floor comes from, is
in [Memory](@ref).
