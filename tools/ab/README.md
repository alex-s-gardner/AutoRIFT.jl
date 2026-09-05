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

# Which quality measure predicts disagreement -- reads the stage-2 dump, no imagery of its own.
julia --project=tools/ab tools/ab/peak_ratio_skill.jl
```

Both stages hand the reference the arrays **AutoRIFT.jl already filtered**, so a preprocessing
difference cannot show up as a correlator or pyramid difference. The filter is compared separately.

## Four conventions the harness has to get right

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
reference, so no flip is needed beyond undoing the reference's own cartesian `Dy = -Dy`. That one flip
is applied by the *writer*: `stage1_python.py` and `stage2_python.py` negate `Dy` and leave `Dx` alone,
so the planes on disk are already in this package's convention and every reader takes them as written.

`bench_figures.jl` goes further and *measures* the sign, scoring all four combinations and keeping the
best, which is the form to copy in a new diagnostic — a hardcoded flip is only correct as long as the
writer's convention holds, and it fails silently when that changes. Four readers did hardcode one, and
negating an already-negated plane cost `dx` a 0.75 px median bias and `corr dx` of −0.9996 on results
that agree to the bit. `corr` is printed for exactly that reason: a near-perfect *anti*-correlation is
a flipped sign and nothing else.

**Array layout.** Julia is column-major and NumPy row-major, so a 2-D array moved between them is
transposed unless the reader says otherwise — and for a square array that is silent. A transposed
displacement field still looks like a displacement field: it reads as a spatial offset in whatever is
being measured, which is the same signature as a real grid error. Use `xchg.py` / `xchg.jl` for any array
crossing the boundary. They put the element type and both dimensions in the file, so `xread`/`read` take
the layout from the header rather than from a caller-supplied shape, and refuse a file they did not
write. Do not hand-roll `reshape(dims[::-1]).T` in a new diagnostic; that idiom is correct in
`stage1_python.py` and `stage2_python.py` only because those two shapes are checked against the
manifest.

    sh tools/ab/xchg_test.sh          # Julia writes -> Python reads, and back, both asserted

Run that before trusting a diagnostic that moves arrays across the boundary. It uses a non-square array
whose values encode their own position, so a transpose fails on both the shape and the values.

Getting the pairing or the grid wrong leaves a residual that is zero under uniform motion and grows with
displacement — indistinguishable at a glance from a real disagreement concentrated at fast flow. A
layout error is worse: it fabricates a defect at a location that has nothing to do with the code being
diagnosed.

## What agreement is measured against

**Exact agreement, and one upsampling step.** At `upsampling = 16` the finest difference either side
can express is 1/16 = 0.0625 px, so a residual below that is not a disagreement about position — it is
which of two adjacent representable values a peak rounded to. The headline number is therefore the
fraction agreeing *to the bit*, with "within one step" beside it.

A 0.2 px tolerance was the threshold here and it measured nothing: stage 2's p95 is 0.017 px, so every
correlation gate from 0.0 to 0.5 passed between 99.4% and 100%. A threshold everything clears reports
the threshold, not the code. The exact fraction moves — it falls the moment the two stop matching
bit-for-bit, which is what a regression does.

**Gated by peak strength**, because sub-pixel agreement presupposes a match. Where the correlation
surface has no dominant peak both implementations pick from noise, and they are then being asked to
agree about something neither one measured — unreachable for any two implementations, including two
builds of the reference itself. The gate says at what peak strength the two become interchangeable,
which is the informative form of the answer.

## Measured agreement

Jakobshavn Landsat 8/9 pair, 15 m panchromatic, chips 16/32/64 on a grid spaced 8, radius 20.

Stage 1, the correlator, one level at a time on a 512² window:

| chip | points | exact | median | p99 | max |
|---|---:|---:|---:|---:|---:|
| 16 | 841 | **100%** | 0.0000 | 0.0000 | **0.0000** |
| 32 | 196 | **100%** | 0.0000 | 0.0000 | **0.0000** |
| 64 | 49 | **100%** | 0.0000 | 0.0000 | **0.0000** |

The correlator is **bit-identical** — not merely close — at every chip size, on both axes, at every
point. Same for `dy`, and at every correlation gate.

Stage 2, the whole pipeline on the full 3072² window, 87,814 shared points: **77.4% exact**, 97.3%
within one step, median radial 0.0000 px, bias `+0.0000` on both axes, p99 0.1411 px. Exact agreement
rises with peak strength, which is the shape to expect — a weak peak is where a tie can break either
way:

| correlation gate | points | exact | within step | p99 | max |
|---|---:|---:|---:|---:|---:|
| ≥ 0.0 | 85,098 | 77.3% | 97.5% | 0.1363 | 6.3408 |
| ≥ 0.2 | 69,337 | 86.1% | 98.3% | 0.1029 | 2.3138 |
| ≥ 0.4 | 39,142 | 96.0% | 99.4% | 0.0319 | 1.0923 |
| ≥ 0.5 | 17,516 | **97.6%** | 99.7% | 0.0113 | 0.3653 |

The `≥ 0.0` row is 85,098 against the 87,814 above because a gate on correlation drops the points an
interpolated fill answered: those carry a displacement but no peak of their own.

Points answered per level: 16 → 55,022 / 55,182, 32 → 19,162 / 18,942, 64 → 14,075 / 14,092, and the
two pick the **same level at 99.3%** of shared points.

**Use 3072², not a smaller window.** Below it the reference silently resolves nothing above chip 16:
its coarse grid for a level is the fine grid decimated by `sparseSearchSampleRate × ChipSize0/spacing`,
which at 1024² leaves 7×7 and 3×3 grids against a `DispFiltC.FiltWidth` of 9, so `CoarseCorValidFac`
falls below `CoarseCorCutoff` and the level is skipped. A comparison on a smaller window is a comparison
of chip 16 alone, whatever `chip_max` says.

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
| 1 thread | 5.487 s | 14.637 s | **2.67×** | 39.9 vs 106.3 µs |
| 12 threads | 1.288 s | 5.245 s | **4.07×** | 9.4 vs 38.1 µs |
| 12 threads, warm `Cache` | 1.173 s | — | **4.47×** | 8.5 µs |

Per-point figures are given because the reference truncates its grid to 368² against 371², so it
solves 1.6% fewer points; normalizing changes the ratio by under 0.05×.

The reference column is carried over from an earlier run of the same command rather than re-measured
alongside: it is a separate process running unchanged code, and a Python row costs a quarter of an hour
to reproduce. Re-run `bench_python.py` before quoting the ratios as simultaneous.

Single-threaded is the algorithmic comparison: **2.67×**. The rest is parallel scaling, where the gap
widens because the two scale differently on this machine — 4.26× against 2.79× from 1 to 12 threads.
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

Apple M2 Max, 12 cores, 96 GiB, macOS 26.5.2 · Julia 1.12.5 · AutoRIFT.jl 0.1.0 (`11920b9`) ·
reference autoRIFT 2.1.1, Python 3.10.20 with NumPy 1.26.4 and OpenCV 4.13.0 · Rasters 0.15.0,
ArchGDAL 0.10.12, DiskArrays 0.4.22. Load average 2.1 during the run, so absolute times are a few
percent pessimistic — equally for every row, so the ratios hold.

CPU time beside wall clock because the two answer different questions: wall clock is what a user
waits, CPU time is what the work costs. A row using about one core's worth of CPU over its wall
clock was not competing with anything, which is the check that a row is worth reading at all.

| configuration | lazy | blocks | runtime | CPU time | peak RSS |
|---|:---:|---:|---:|---:|---:|
| python autoRIFT v2.1.2† | no | 1 | 176.5 s | 414.3 s | 6883 MiB |
| AutoRIFT.jl, eager, no blocks, 12 threads | no | 1 | 21.3 s | 79.3 s | 7247 MiB |
| AutoRIFT.jl, lazy, block 2048, 12 threads | yes | 81 | 45.5 s | 211.2 s | 3663 MiB |
| AutoRIFT.jl, lazy, block 4096, 12 threads | yes | 25 | 56.6 s | 222.1 s | 9800 MiB |
| AutoRIFT.jl, eager, no blocks, 1 thread | no | 1 | 64.7 s | 66.2 s | 6702 MiB |
| juliac binary, lazy, no blocks, 1 thread\* | yes | 1 | 63.8 s | 65.3 s | 6361 MiB |
| juliac binary, lazy, block 2048, 1 thread\* | yes | 81 | 89.5 s | 90.5 s | 3363 MiB |
| juliac binary, lazy, block 1024, 1 thread\* | yes | 289 | 90.9 s | 92.5 s | 3215 MiB |
| juliac binary, lazy, block 512, 1 thread\* | yes | 1122 | 94.5 s | 95.9 s | 3135 MiB |
| juliac binary, lazy, block 4096, 1 thread\* | yes | 25 | 99.9 s | 101.3 s | 3957 MiB |
| juliac binary, lazy, block 256, 1 thread\* | yes | 4422 | 109.3 s | 110.8 s | 3134 MiB |

† The reference's correlation loop is serial, but OpenCV's own threading is not: it uses 2.4 cores'
worth over the run, which is why its CPU time exceeds its wall clock. `cv2.setNumThreads` does not
constrain it on macOS, where OpenCV dispatches through GCD.

\* The binary is single-threaded: threading in a `--trim=safe` build is blocked upstream by
[JuliaLang/julia#61319](https://github.com/JuliaLang/julia/issues/61319), so the thread pool is
created but a spawned task resolves its entry point in the wrong world age and fails at runtime.

The answer is the same in every row that can be compared byte for byte: lazy at 2048 against eager, 1
thread against 12, and the binary at each block size against its own unblocked run.

Four things this table says that the windowed ones cannot:

**Lazy trades time for memory, and the trade is not free.** 45.5 s against 21.3 s for a 49% lower
peak. Blocking a scene that fits in memory costs 2.1× the runtime and buys nothing; blocking one that
does not fit is the only way to process it at all. That is the whole basis for choosing.

**Block 4096 is the exception, and it is not a memory saving at all.** 9800 MiB against 7247 for the
unblocked row: a block that large holds more per-block buffer than the whole scene costs eagerly, so
blocking there pays 2.7× the runtime to raise peak memory by 35%. The useful range is 2048 and below.

**The two eager rows read raw planes, not rasters.** Measured through
`autorift(::AbstractRaster, ...)` the same configuration peaks at **11590 MiB**, not 7148: `read` keeps
the raster's `missingval`, so a filled copy of each scene is built and the nodata mask is a third array
over the original — three scene-sized arrays where a plain `Matrix` needs one. The lazy path avoids the
copies entirely, because blocking never forms them.

That 1.6× is a property of this code path, not of `Raster`. One pass producing a plain `Matrix{T}`
beside a `BitMatrix` brings resident input from 2.6× the correlator's requirement down to 1.06×, or
1.09 GiB per image rather than 2.34 on this scene. It is deliberately not implemented: measured before
and after, peak RSS is unchanged — 11908 against 11904 MiB — because the ceiling is set inside the
correlator, roughly 5 GiB above where the inputs sit, so shrinking them only widens the headroom
beneath it. The input arrays are the wrong lever for peak memory on this path; the per-thread
workspaces and output are where the 12 GiB comes from.

**Block size is nearly free above the halo, and the floor is the grid.** 2048 through 256 px spans
81 to 4422 blocks and 54× the halo redundancy, yet peak moves by 7.7% and runtime by 24%: what is left
resident is the output grid and the per-block buffers, and only the latter shrinks. The binary's
unblocked row is the one that matters — 6362 against ~3.2 GiB blocked, which is the difference between
fitting on a small instance and not.

## Where the two still differ, and why

The correlator is bit-identical and 77.4% of pipeline points agree to the bit. What remains is 2.7% of
points beyond one upsampling step, and it is the **pyramid**, not the correlator:

| | disagreeing points | agreeing points |
|---|---:|---:|
| chose a different chip-size level | **23.0%** | 0.1% |
| filled from neighbours on the Julia side | **31.8%** | 7.5% |
| median `peak_ratio` | 1.76 | 1.83 |
| median displacement | 0.48 px | 0.56 px |

**51.9%** of the disagreeing points differ in chip level or were filled — decisions made by the outlier
filter and the smallest-chip-wins merge, not values returned by the correlator. A point one side
answered at chip 16 the other may have answered at 32, and the two estimates then describe different
footprints of ground; a point one side measured the other may have filled from neighbours. Both
conditions are far rarer among the agreeing points, 0.1% and 7.5%, which is what makes them the
explanation rather than a coincidence.

Two things this is *not*. It is not concentrated at fast flow: among the top 2% by displacement the rate
is **2.0%**, lower than the 2.7% elsewhere. And it is not a quality effect — the disagreeing points are
only marginally the more ambiguous, at a median `peak_ratio` of 1.76 against 1.83, well inside the
spread of either population. So these are not points where a rival peak beat the right one.

The residual beyond the level and fill differences is tie-breaking at 1/16 px, which no two
implementations can agree about: below a real correlation peak both are picking from noise.

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
reference silently resolves nothing at that level** — `autorift()` hits `continue` because
`CoarseCorValidFac` falls under `CoarseCorCutoff`, with no warning. The level's coarse pass still runs,
so the only visible symptom is a missing chip size in the output.

Each level correlates a coarse grid decimated from the fine one by
`sparseSearchSampleRate × ChipSize0X / GridSpacingX` — 8 at these settings — applied to a grid already
scaled by `ChipSize0X / chip`. `DispFiltC.FiltWidth` is 9. So the coarse grid shrinks with the chip while
the filter does not:

| window | fine grid | coarse grid at chip 16 / 32 / 64 | levels that resolve |
|---:|---:|---|---|
| 1024² | 112 | 14 / 7 / 3 | 16 only |
| 2048² | 240 | 30 / 15 / 7 | 16, 32 |
| **3072²** | 368 | 46 / 23 / 11 | **all three** |

A coverage or chip-size comparison below 3072² therefore measures the window, not either
implementation — verified by intercepting `arImgDisp_s`, which at 1024² is called with chips 32 and 64
for their coarse passes and never for a fine pass.
