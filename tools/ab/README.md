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
comparison that writes a PNG to `plots/`, which is gitignored — the figures are regenerated rather
than committed, and the numbers they show are recorded here and in `bench_table.json`.

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

Each is easy to get wrong in a way that yields a plausible accuracy result rather than an obvious
failure, so each is asserted at its call site and checked against measurement.

**Argument order.** `arImgDisp_s(a, b)` cuts its chip from **`b`** and its search window from `a`, the
reverse of what the parameter names suggest: `a` binds to `I1`, but the body calls the C++ as
`(I2.ravel(), I1.ravel())` (`autoRIFT.py:1251-1256`) and the C++ binds *its* first array to `sec_img`,
where `chip` is taken (`autoriftcoremodule.cpp:413,453`). Two swaps that compose rather than cancel. So
stage 1 passes `(ref, sec)`. `runAutorift` inherits it — `arImgDisp_s(self.I2, self.I1)` — so stage 2
sets `obj.I1` to the **secondary**.

**Grid origin.** Julia's grid is 1-based pixel centres and the reference's 0-based, so the Python side
subtracts 1. Stage 2 subtracts **1.5**: `runAutorift` snaps an even chip's grid to `round(x + 0.5) - 0.5`
(`autoRIFT.py:526-527`) *and* `arImgDisp_s` adds its own `+0.5`, so the pipeline lands half a pixel past
the bare correlator. Verified by interception — handing `runAutorift` 53.0 makes it correlate at 53.5.

**Sign.** With the chip on the secondary on both sides, both report the offset from secondary back to
reference, so no flip is needed beyond undoing the reference's own cartesian `Dy = -Dy`. The comparison
scripts *measure* the sign rather than asserting it, scoring all four combinations and reporting the
best, so a convention change surfaces as a printed sign instead of a figure full of apparent error.

Getting the pairing or the grid wrong leaves a residual that is zero under uniform motion and grows with
displacement — indistinguishable at a glance from a real disagreement concentrated at fast flow.

## Why agreement is reported against a correlation gate

A 0.2 px target is a claim about sub-pixel *matching noise*, which presupposes a match. Where the
correlation surface has no dominant peak both implementations pick from noise, and they are then
being asked to agree about something neither one measured — unreachable for any two
implementations, including two builds of the reference itself. The comparison therefore reports the
residual as a function of peak strength; that ladder is the informative form of the answer.

## Measured agreement

Jakobshavn Landsat 8/9 pair, 15 m panchromatic, chips 16/32/64 on a grid spaced 8, radius 20.

Stage 1, the correlator, one level at a time on a 512² window:

| chip | points | median \|ddx\| | p95 | max | within 0.2 px |
|---|---:|---:|---:|---:|---:|
| 16 | 841 | 0.0000 | 0.0000 | **0.0000** | 100% |
| 32 | 196 | 0.0000 | 0.0000 | **0.0000** | 100% |
| 64 | 49 | 0.0000 | 0.0000 | **0.0000** | 100% |

The correlator is **bit-identical** — not merely close — at every chip size, on both axes, at every
point. Same for `dy`.

Stage 2, the whole pipeline on a 1024² window, 11,103 shared points: median radial **0.0000 px**,
bias `+0.0000` on both axes, 99.8% within 0.2 px. Only 0.14% of points differ by more than 0.25 px and
0.08% differ in `dy` by more than 0.25 px, with no spatially coherent structure — the densest 12×12
block of the grid holds three of them.

This holds only with the argument order and grid offset above correct; both are asserted at their call
sites in `stage1_python.py` and `stage2_python.py`.

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

## The full scene: every configuration, end to end

The tables above are a 3072² window, where nothing is large enough to force a choice. This one is the
**full Landsat 8/9 overlap** — 17121×16961, 290.4 Mpixel, grid 2127×2107 = 4.48 M points, chips 16/64,
spacing 8, radius 20, upsampling 16 — which is where the choices matter.

```bash
julia --project=tools/ab tools/ab/bench_table.jl              # measure, then render
julia --project=tools/ab tools/ab/bench_table.jl --replot      # re-render from the recorded results
```

Results are recorded in `bench_table.json` with the machine, OS and versions that produced them, and
rendered to `plots/bench_table.pdf` — three pages: this table, then the accuracy comparison against
the reference as maps (`plots/fig_heatmaps.png`) and as distributions (`plots/fig_histograms.png`).
The figures come from `bench_figures.jl`, which reads the stage-2 bundle, so run stage 2 first if it
is stale. Performance and accuracy are in one artifact because neither answers the question alone: a
faster implementation that disagrees is not faster at the same job.

Whole-process wall clock and peak RSS from `/usr/bin/time -l`, one fresh process per row: `ru_maxrss`
is a high-water mark, so two configurations measured in one process both report the larger.

Apple M2 Max, 12 cores, 96 GiB, macOS 26.5.2 · Julia 1.12.5 · AutoRIFT.jl 0.1.0 (`f3bf970`) ·
reference autoRIFT 2.1.1, Python 3.10.20 with NumPy 1.26.4 and OpenCV 4.13.0 · Rasters 0.15.0,
ArchGDAL 0.10.12, DiskArrays 0.4.22. Load average 3.7 during the run, so absolute times are a few
percent pessimistic — equally for every row, so the ratios hold.

| configuration | threads | lazy | blocks | runtime | peak RSS |
|---|---:|:---:|---:|---:|---:|
| python autoRIFT v2.1.2 | 12 | no | 1 | 336.2 s | 8126 MiB |
| AutoRIFT.jl, eager, no blocks | 12 | no | 1 | 21.8 s | 7148 MiB |
| AutoRIFT.jl, eager, no blocks | 1 | no | 1 | 69.8 s | 6701 MiB |
| AutoRIFT.jl, lazy, block 2048 | 12 | yes | 81 | 41.9 s | 3926 MiB |
| juliac binary, lazy, no blocks | 1\* | yes | 1 | 71.6 s | 6532 MiB |
| juliac binary, lazy, block 2048 | 1\* | yes | 81 | 95.7 s | 3504 MiB |
| juliac binary, lazy, block 1024 | 1\* | yes | 289 | 93.9 s | 3299 MiB |
| juliac binary, lazy, block 512 | 1\* | yes | 1122 | 97.1 s | 3352 MiB |
| juliac binary, lazy, block 256 | 1\* | yes | 4422 | 108.8 s | 3354 MiB |

\* The binary is single-threaded: threading in a `--trim=safe` build is blocked upstream by
[JuliaLang/julia#61319](https://github.com/JuliaLang/julia/issues/61319), so the thread pool is
created but a spawned task resolves its entry point in the wrong world age and fails at runtime.

The answer is the same in every row that can be compared byte for byte: lazy at 2048 against eager, 1
thread against 12, and the binary at each block size against its own unblocked run.

Three things this table says that the windowed ones cannot:

**Lazy trades time for memory, and the trade is not free.** 41.9 s against 21.8 s for a 45% lower
peak. Blocking a scene that fits in memory costs 1.9× the runtime and buys nothing; blocking one that
does not fit is the only way to process it at all. That is the whole basis for choosing.

**The two eager rows read raw planes, not rasters.** Measured through
`autorift(::AbstractRaster, ...)` the same configuration peaks at **11590 MiB**, not 7148: `read` keeps
the raster's `missingval`, so a filled copy of each scene is built and the nodata mask is a third array
over the original — three scene-sized arrays where a plain `Matrix` needs one. An in-memory raster
therefore costs 1.6× the equivalent array, and the lazy path avoids it because blocking never forms the
copies.

**Block size is nearly free above the halo, and the floor is the grid.** 2048 through 256 px spans
81 to 4422 blocks and 54× the halo redundancy, yet peak moves by 1.6% and runtime by 16%: what is left
resident is the output grid and the per-block buffers, and only the latter shrinks. The binary's
unblocked row is the one that matters — 6532 against ~3.3 GiB blocked, which is the difference between
fitting on a small instance and not.

## Where the two still differ, and why

The correlator is bit-identical and the pipeline agrees to a median of 0.0000 px. What remains is 0.14%
of points beyond 0.25 px, spatially unstructured, and it is accounted for:

| source | share of disagreeing points | share of agreeing points |
|---|---:|---:|
| a hole fill on one side only | 17% | 1% |
| a different chip-size level chosen | 11% | 0% |

Both are pipeline decisions, not correlator values: the two implementations' outlier filters and
smallest-chip-wins merges accept marginally different point sets, so a point one side measured directly
the other may have filled from neighbours. Where both measure at the same level they return the same
number.

The residual beyond that is tie-breaking at 1/16 px on weak surfaces, which no two implementations can
agree about — below a real correlation peak both are picking from noise.

### Not a bug: `foo.create(cols, rows, …)` in the reference

`autoriftcoremodule.cpp:291` and `:506` call `foo.create(cols, rows, CV_32FC1)` where
`cv::Mat::create` takes `(rows, cols)`, which reads like a transposition. It is harmless, on two
independent grounds. `cv::pyrUp(src, dst, dstsize)` **allocates `dst` to `dstsize` itself**, so the
preceding `create` has no effect whatever its argument order — verified by passing a deliberately
wrong-shaped destination and getting the requested shape back. And `x_count` and `y_count` are both
hard-coded to 5 (`:484-491`), so `cols == rows` at every cascade step regardless.

Recorded because the argument order invites exactly this false positive.

### The chip-centre convention, which both implementations share

An even-sized chip has no centre sample: extending `-chip/2` to `chip/2 - 1` about grid point `p` covers
`p-h … p+h-1`, whose centroid is `p - 0.5`. Both implementations account for this and both put the
estimate at that centroid — AutoRIFT.jl by adding `0.5` to the grid in `_shift_points`
(`src/track.jl`), the reference by `xGrid += Px + 0.5` inside `arImgDisp_s`, plus the even-chip snap in
`runAutorift`. Verified directly: at grid point 30 with chip 16 both cut pixels 22–37.

Measured on an analytic ramp, both show the same bias against `d(p)` and **exactly zero** against
`d(p - 0.5)`, at gradients from 0.02 to 0.12 px/px and on both axes:

| gradient px/px | bias vs `d(p)` | bias vs `d(p - 0.5)` |
|---:|---:|---:|
| 0.02 | −0.0100 | **+0.0000** |
| 0.05 | −0.0250 | **+0.0000** |
| 0.08 | −0.0400 | **+0.0000** |

So the convention is a property of even-chip correlation, not a difference between the two. A caller
matching AutoRIFT.jl output against an external grid should know that a chip's estimate describes the
ground half a pixel below its nominal point; the effect is `0.5 ×` the local velocity gradient, which on
Jakobshavn exceeds the 1/16 px quantization step over 1.3% of the scene and 0.25 px over 0.06%.

## The window-size artifact this harness exposes

On a window small enough that a level's coarse grid falls below the outlier filter's width, **the
reference silently resolves nothing at that level**. `autorift()` hits `continue` because
`CoarseCorValidFac` falls under `CoarseCorCutoff`, and it does so without a warning — the reference used
chip 16 for every point it answered while AutoRIFT.jl also used chip 64 for 558. Both then reach all
three levels at 3072².

The arithmetic, pinned by the single-level runs in `tools/synth/`, is that both quantities scale with
`ChipSize0_GridSpacing_oversample_ratio = ChipSize0X / GridSpacingX` — so raising the chip at a fixed
grid spacing shrinks the coarse grid and widens the filter at the same time:

| chip | ratio | coarse step | coarse grid | `DispFiltC.FiltWidth` | |
|---:|---:|---:|---:|---:|---|
| 16 | 2 | 8 | 15×15 | 9 | resolves |
| 32 | 4 | 16 | 7×7 | 17 | filter wider than grid |
| 64 | 8 | 32 | 3×3 | 33 | filter wider than grid |

So a coverage comparison on a small window measures the window rather than either implementation.
Compare on a window large enough that every level's coarse grid clears the filter, or expect the
reference to be missing its coarse levels.
