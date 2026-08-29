# The A/B harness: AutoRIFT.jl against the Python reference

Runs both implementations on **identical input arrays with identical parameters** and diffs them per
grid point. Comparing against an ITS_LIVE granule instead conflates at least six confounds — the
granule was made by autoRIFT 1.5.0, at per-point `vx0`/`vy0` priors from the parameter shapefile,
with `stable_shift` already removed, reprojected EPSG:3413→UTM, on a `GeogridOptical` grid, through
an unseeded `np.random.normal` gap fill. None of those survive a direct call, and a granule
correlation of 0.99 is compatible with a systematic half-pixel bias it cannot localize.

## The oracle

`micromamba run -n arift-ref python` has `geo_autorift 2.1.1` with the compiled `autoriftcore`
extension. Its `autoRIFT.py` is byte-identical to `../autoRIFT-v2.1.2`, the version `REFERENCE.md`
pins, so the env is a valid oracle for the pin despite the version label.

## Running it

Requires the real-data cache (`tools/realdata/README.md`). Each stage is Julia, then Python, then a
comparison that writes a PNG to `plots/`.

```bash
# Stage 1 -- the correlator alone: one chip size, no pyramid, no outlier filter.
julia --project=tools/ab tools/ab/stage1_julia.jl 1024 32 20
micromamba run -n arift-ref python tools/ab/stage1_python.py
julia --project=tools/ab tools/ab/compare.jl chip32

# Stage 2 -- the whole pipeline: pyramid, coarse pass, outlier filter, merge.
julia --project=tools/ab tools/ab/stage2_julia.jl 3072 16 64 20
micromamba run -n arift-ref python tools/ab/stage2_python.py
julia --project=tools/ab tools/ab/compare2.jl full
```

Both stages hand the reference the arrays **AutoRIFT.jl already filtered**, so a preprocessing
difference cannot show up as a correlator or pyramid difference. The filter is compared separately.

## Three conventions the harness has to get right

**Grid origin.** Julia's grid is 1-based pixel centres, the reference's 0-based, so the Python side
subtracts 1. Both then add the same `+0.5` internally — `xGrid += Px + 0.5` in `arImgDisp_s` against
`_shift_points` in `track.jl` — which is what makes an even-sized chip's estimate refer to its true
centre.

**Argument order.** The reference's own call sites pass `(self.I2, self.I1)`
(`autoRIFT.py:673,747`); the C++ binds the first to `sec_img` (the chip) and the second to `ref_img`
(the search window). `obj.I1` is the reference scene
(`hyp3-autorift/.../testautoRIFT.py:310`), so the chip comes from the secondary — the same assignment
`track.jl` makes.

**Sign.** AutoRIFT.jl reports displacement secondary-to-reference; the reference reports feature
motion, its negative. **Both axes flip together.** A one-axis flip would be a real bug in one of the
two; the both-axis flip is the documented difference, and `test/realdata.jl` already applies it.

## Why agreement is reported against a correlation gate

A 0.2 px target is a claim about sub-pixel *matching noise*, which presupposes a match. Where the
correlation surface has no dominant peak both implementations pick from noise, and they are then
being asked to agree about something neither one measured — unreachable for any two
implementations, including two builds of the reference itself. The comparison therefore reports the
residual as a function of peak strength; that ladder is the informative form of the answer.

## Measured agreement

Jakobshavn Landsat 8/9 pair, 15 m panchromatic, chips 16/32/64 on a grid spaced 8, radius 20.

Stage 1, the correlator, one level at a time:

| chip | points | median radial | bias dx / dy | within 0.2 px at corr ≥ 0.5 |
|---|---:|---:|---:|---:|
| 16 | 3721 | 0.088 px | +0.0000 / +0.0000 | 95.1% |
| 32 | 900 | 0.062 px | +0.0000 / +0.0000 | 100% |
| 64 | 225 | 0.062 px | +0.0000 / +0.0000 | 100% |

Stage 2, the whole pipeline over the full 3072² window, 85,221 shared points: median radial
0.088 px, `cor(dx) = 0.995`, `cor(dy) = 0.992`, bias `+0.0000` on both axes, coverage 65.5% against
65.3%. Points answered per level: 16 → 54,943 / 55,909, 32 → 18,596 / 19,165, 64 → 15,206 / 13,294.

The signed median bias is **exactly zero** at every chip size and in the full pipeline, and the
median residual sits at or below one upsampling step (1/16 = 0.0625 px) — the finest distinction
either side can draw. The residual is tie-breaking on weak surfaces, not error.

The preprocessing filter agrees to `3e-6` relative in the interior, which is Float32 rounding. It
differs by up to `2e4` in a `filter_width ÷ 2` border ring, where the two use different border
conventions by design. That ring is reachable by the outermost grid points (search windows span
2..3064 of 1..3072 at these settings), but measured against it those points agree *better* than the
interior (p95 0.31 px against 0.45), because a chip made of padding fails the validity check and is
never reported.

## Runtime

`benchmark/` times kernels against OpenCV primitives and records that the end-to-end comparison
"needs the reference installed with its ISCE3/GDAL stack and a real image pair". It is installed, so
these close that gap.

```bash
julia --project=tools/ab -t 12 tools/ab/bench_julia.jl 3072 3
OMP_NUM_THREADS=12 micromamba run -n arift-ref python tools/ab/bench_python.py 3072 3
```

Both sides are handed the same already-filtered arrays and the same grid, so this times the pyramid,
correlator, outlier filter and merge — not two high-pass implementations. Minimum of three runs,
after an untimed warmup that pays Julia's JIT and FFTW planning against numba's `colfilt`
compilation. Apple M2 Max, 12 cores, 3072² window, 137,641 grid points, chips 16/64 spacing 8.

| | AutoRIFT.jl | reference | ratio | per point |
|---|---:|---:|---:|---:|
| 1 thread | 6.019 s | 14.637 s | **2.43×** | 43.7 vs 108.1 µs |
| 12 threads | 1.635 s | 5.245 s | **3.21×** | 11.9 vs 38.7 µs |
| 12 threads, warm `Cache` | 1.440 s | — | **3.64×** | 10.5 µs |

Per-point figures are given because the reference truncates its grid to 368² against 371², so it
solves 1.6% fewer points; normalizing changes the ratio by under 0.05×.

Single-threaded is the algorithmic comparison: **2.43×**. The rest is parallel scaling, where the gap
widens because the two scale differently on this machine — 3.68× against 2.79× from 1 to 12 threads.
The warm-`Cache` row has no counterpart in the reference, which has no batch entry point; it is what
a driver over many pairs pays once buffers and FFT plans are reused.

Every row above is **unblocked**, on both sides. The blocked path costs the following on the same
pair, against the unblocked run at the same thread count:

| block | 12 threads | 1 thread | blocks |
|---|---:|---:|---:|
| unblocked | 1.600 s | 6.136 s | — |
| 256 px | 1.567 s (0.98×) | 7.491 s (1.22×) | 144 |
| 384 px | 1.554 s (**0.97×**) | 7.199 s (1.17×) | 64 |
| 512 px | 1.584 s (0.99×) | 7.082 s (1.15×) | 36 |
| 1024 px | 1.808 s (1.13×) | 6.919 s (1.13×) | 9 |
| 1536 px | 2.635 s (1.65×) | 6.845 s (**1.12×**) | 4 |

The trend **inverts with thread count**, and both directions have the same cause. Blocking trades
redundant work in the halo overlap — which grows as blocks shrink — against parallelism, since blocks
are the unit of threaded work and a scene cut into four cannot fill twelve threads. Serially only the
overlap is visible, so larger blocks are cheaper; threaded, small blocks are free or slightly faster
(better load balance and cache residency) while large ones starve the pool.

So blocking is not a runtime cost at a sensible block size on 12 threads, and is a 12–22% cost
serially. It never changes the answer: blocked and unblocked agree bit for bit on all five output
arrays at every block size from 256 to 2048 px, which is what makes the accuracy comparison above
transfer to a blocked run unchanged.

**Threads cannot be matched below the core count on macOS.** This OpenCV is built with
`Parallel framework: GCD`, and Apple's Grand Central Dispatch ignores `cv2.setNumThreads` —
`getNumThreads()` reports 12 whatever it is set to, and `OMP_NUM_THREADS` reaches only the
reference's own OpenMP loop, not the `matchTemplate` calls inside it. So the 12-thread row is the
honest matched comparison, and any capped run would understate the reference.

## Peak memory

```bash
julia --project=tools/ab -t 12 tools/ab/bench_memory.jl 3072
```

One measurement per process, because `Sys.maxrss` and `ru_maxrss` are high-water marks: a baseline
and a working run in the same process would report the larger twice. Both sides read the same bare
`Float32` pair from `bench_pair_3072.bin` — **not** `window.jls`, whose 756 MiB deserialization sets a
~1.2 GiB peak that has nothing to do with either pipeline and swamps the signal.

Total process peak, 3072² window, 12 threads:

| | peak MiB | live MiB |
|---|---:|---:|
| AutoRIFT.jl one-shot | **635.6** | 175.1 |
| reference `runAutorift` | **651.5** | 651.8 |
| ratio | 1.02× | — |

Peak is a near-tie. The two differ sharply on *retention*: after a full collection the reference's
resident set is essentially its peak (651.8 of 651.5 MiB), while AutoRIFT.jl's live heap is 175 MiB —
the rest is allocator slack the collector has not returned. Across two pairs through one `Cache` the
live heap grows 31.8 MiB and peak grows 23.5 MiB, so a long batch run does not need process recycling.

Compare **total peak**, not the above-baseline delta: each baseline already holds its own copy of the
two scenes, so subtracting charges the pipeline only for what it needed beyond the imagery and hides
the imagery itself. The deltas (190.5 MiB against 304.6 MiB) say where the cost sits; the totals are
what a memory limit sees.

**Blocking does not lower peak on a scene this size**, and the sweep says so plainly:

| block | 12 threads | 1 thread |
|---|---:|---:|
| one-shot | 635.6 | 613.5 |
| 256 px | 660.8 (1.04×) | 638.5 (1.04×) |
| 512 px | 804.3 (1.27×) | 634.3 (1.03×) |
| 1024 px | 952.8 (1.50×) | 660.2 (1.08×) |

At 3072² the two resident input scenes are 72 MiB and the filtered pair is the only scene-sized array
blocking removes, so there is nothing left to save — while each of 12 threads takes its own block
buffers, which is what makes larger blocks cost more under threading. `process_block_size` earns its
keep when the scene will not fit at all; on a scene that fits, it trades peak for nothing. The
`benchmark/memory.jl` figures (−36% at 4096², −56% at 6000²) are measured against *windowed,
never-resident* input, which is the case blocking is for.

## The window-size artifact this harness exposes

On a window small enough that a level's coarse grid falls below the outlier filter's width, **the
reference silently resolves nothing at that level**. At 1024² with chips 16–64 the coarse grids are
7×7 and 3×3 against a filter width of 9, `CoarseCorValidFac` falls under `CoarseCorCutoff`, and
`autorift()` hits `continue` — the reference used chip 16 for every point it answered while
AutoRIFT.jl also used chip 64 for 558. Both then reach all three levels at 3072².

So a coverage comparison on a small window measures the window rather than either implementation.
Compare on a window large enough that every level's coarse grid clears the filter, or expect the
reference to be missing its coarse levels.
