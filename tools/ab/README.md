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

## Large disagreements at maximum flow

The global signed bias is exactly zero, but the difference is **not** uniform: at maximum flow it is
larger and spatially coherent, concentrated in `dy`.

!!! warning "These numbers are pending re-derivation"

    Both stage scripts here pair the images in the order `tools/synth/README.md` shows to be reversed:
    `stage1_python.py` calls `arImgDisp_s(sec, ref)` and `stage2_python.py` sets `I1 = reference`, where
    the chip must come from the secondary. On a deforming field that makes the reference's estimate
    describe a different piece of ground, producing a residual that is zero under uniform motion and
    grows with displacement — the same signature as the fast-flow disagreement described below.

    Until stage 1 and stage 2 are re-run with `(ref, sec)` and `I1 = secondary`, **the magnitudes in
    this section are not attributable to either implementation**, and nor is the ruled-out table's
    reasoning about which side sits at the correlation peak.

What is established independently of that: on 26 constructed scenes with exact analytic truth —
translation, rotation, divergence, shear, noise, decorrelation, uint8, real texture — AutoRIFT.jl,
`autoRIFT.py` and raw OpenCV are **bit-identical at 99.99–100.00% of points** (`tools/synth/`). The
correlator, the sub-pixel refinement, the response to deformation and uint8 quantization are therefore
shared, and none of them can explain a real-scene disagreement. What this harness still has to settle,
after re-pairing, is whether any residual survives at all, and if so whether it comes from the
preprocessing filter, the per-point search-radius field, or the pyramid on real scene structure.

### What is observed

Among the top 2% by `|dy|`, **25% differ by more than 0.25 px**, 5.6% by more than 0.5 px, and the
largest reaches 4.77 px, against a 2.4% rate elsewhere. The largest coherent patch is grid rows
113–129, columns 228–236 — 41 points at a median `dy` difference of **+1.38 px** while `dx` differs by
+0.31 px.

The asymmetry between the axes is not a property of the correlator. `|ddx|` and `|ddy|` have identical
medians of 0.0625 px over the whole grid; `dy` only *looks* worse in the difference maps because the
fields differ in range (`dx` spans −0.25→+9.19 px, `dy` −3.19→+7.81). Lag-1 autocorrelation of 0.67
inside the trunk turns ±1-step differences into coherent patches rather than speckle.

### The patch measurement, and why it is inconclusive

Evaluating **OpenCV's own `matchTemplate` surface** at each side's reported displacement, on 117 patch
points at chip 16, put AutoRIFT.jl at the surface maximum and the reference away from it (medians 0.5213
against 0.3575; at the peak for 95 of 117 against 45 of 117; the reference negative at 14 points). Over
a 500-point sample of the rest of the grid the two were indistinguishable (0.6477 against 0.6469).

**That comparison cannot be read as evidence about the reference**, because the surface was built from
the chip/window assignment the stage scripts use, and the reference was run with the reversed pairing.
Evaluating a surface at a displacement that describes different ground puts the value off-peak by
construction. The measurement has to be repeated after re-pairing.

What it does establish, and what survives: `arImgDisp_s` recovers synthetic pure translations exactly —
0.000 px at five shifts, and 0.0000 px across the translation cases in `tools/synth/` — so its windowing
and refinement are sound.

### All three implementations are the same answer

`cv2.matchTemplate(TM_CCOEFF_NORMED)` refined by the same `pyrUp` cascade, with no autoRIFT machinery
around it, returns the **identical `Float32`** to *both* AutoRIFT.jl and `arImgDisp_s` at 99.99–100.00%
of points across all 26 synthetic cases (`tools/synth/`). The residual is single 1/16 px steps where two
candidate peaks tie.

Three independently written implementations agreeing to the last bit makes each an arbiter for the
others, and it means OpenCV's maturity is evidence about both correlators rather than a competing
answer. It also bounds what this section's disagreement can be: not the matching, and not the sub-pixel
refinement.

### Chip-size convergence: an arbiter neither implementation controls

Correlation is more robust at a larger chip, because more texture is averaged. So a point's own estimates
across a ladder of chip sizes — 16, 24, 32, 48, 64 at a fixed 24 px radius — converge on the displacement
that is really there, independently of which implementation reported what. Where those five agree to
within 1 px a trustworthy answer exists, and the question becomes which side matches it.

This ladder found AutoRIFT.jl tracking the converged answer while the reference departed from it, by
4.8× at the worst points. **Like the patch measurement, it is pending re-derivation**: the reference's
rung of every comparison came from the reversed pairing.

Two properties the test needs, independent of that:

**Edge peaks must be rejected.** A large chip spans a velocity gradient, its peak leaves a fixed search
window, and the estimate pins near zero. Holding the radius at 24 out to chip 256 collapses the *control*
from `dy` +6.9 to +0.03 — that failure, not a property of large chips.

**Only the stable subset can be judged.** 58% of the anomalous points have a self-consistent ladder
answer against 95% of controls (median spread 0.699 px against 0.225), so on roughly 40% of them no
converged answer exists and neither implementation can be called right.

### Ruled out

Candidates eliminated by measurements that do not depend on the image pairing, so these stand:

| candidate | evidence against |
|---|---|
| absolute rather than signed peak finding | both call `minMaxLoc(result, NULL, NULL, NULL, &maxLoc)`, the signed maximum |
| a rounded inter-level displacement prior | neither implementation carries one: `Dx00` derives only from `self.Dx0` (`:590`), never from a level's result, and this harness sets it to `0`; the reference's coarse pass feeds the fine pass a *mask*, never a displacement |
| OpenCV `matchTemplate` vs an FFT ZNCC | agree to 8.5e-07 on the same window, Float32 epsilon, same argmax |
| the sub-pixel refinement | identical on the same surface: cloning the C++'s 5×5 ROI before its `pyrUp` cascade changes nothing, and the cascade agrees with `subpixel_peak` to the last bit |
| the correlator and refinement as a whole | bit-identical across all 26 truth-based cases in `tools/synth/`, including rotation, divergence, shear, noise and decorrelation |
| the `-ref_min` / `-chip_min` shift (`cpp:464-465`) | neutral to 1.19e-07 despite shift constants differing by ~7000 on filtered imagery |
| uint8 requantization (`DataType = 0`) | `tools/synth/` measures it at 0.0000 px on translation and 0.0009 px on real texture |
| crevasse-parallel elongated peaks | peaks in the patch are ~2× *sharper* along y than x (curvature ratio 1.96, half-max width 0.50 against 1.00), the opposite of a y-ridge; the structure tensor finds no strong linear texture |
| a coarser chip being chosen | 93% of large-error points used the same chip on both sides |
| grid marshalling shape sensitivity | a 2-D and a 1×N grid give identical answers, `max \|difference\| = 0.0000` |
| a projection or resampling difference | the harness passes raw arrays and pixel-index grids; no CRS is constructed on either side |

### Still open

| candidate | why it is untested |
|---|---|
| the image pairing in these stage scripts | reversed, per the warning above; this is the first thing to fix and may account for all of the residual |
| the preprocessing filter | both sides here correlate arrays AutoRIFT.jl filtered, and `tools/synth/` bypasses filtering entirely, so a Wallis-vs-highpass difference has never been exercised |
| the per-point search-radius field | ITS_LIVE sizes the radius per point from a prior velocity field; this harness and `tools/synth/` both use a single fixed radius |
| the pyramid on real scene structure | the truth-based cases are single-level by construction, so the coarse pass, `DISP_FILT`, hole fill and merge are compared only here |
| a peak at the search-window edge | when the correlation peak lands on the first or last surface index the clamped 5×5 refinement patch puts it on the patch border, and `pyrUp` misplaces the upsampled maximum by up to a full pixel — measured at 0.9375 px on a 19 px shift at radius 20. Affects both implementations identically, `src/peak.jl` included, so it cannot produce a *disagreement*; it is a shared accuracy limit worth guarding in both |

### Not a bug: `foo.create(cols, rows, …)` in the reference

`autoriftcoremodule.cpp:291` and `:506` call `foo.create(cols, rows, CV_32FC1)` where
`cv::Mat::create` takes `(rows, cols)`, which reads like a transposition. It is harmless, on two
independent grounds. `cv::pyrUp(src, dst, dstsize)` **allocates `dst` to `dstsize` itself**, so the
preceding `create` has no effect whatever its argument order — verified by passing a deliberately
wrong-shaped destination and getting the requested shape back. And `x_count` and `y_count` are both
hard-coded to 5 (`:484-491`), so `cols == rows` at every cascade step regardless.

Recorded because the argument order invites exactly this false positive.

### The 1-pixel chip-centre convention difference

The reference centres its chip half a pixel differently: `int(-8 + 1915.5) = 1907` gives 1-based
columns 1878–1893, centre 1885.5, where AutoRIFT.jl uses `round(1886.0) = 1886` and columns
1879–1894, centre 1886.5. The two sample ground **1 px apart**. It cancels in the displacement, since
each offsets chip and window together — which is why the global bias is zero — but they are describing
slightly different ground. Correlating at the reference's placement does not reproduce its answers
(median distance 0.500 px against 0.538), so this is a documentation matter rather than the cause of
the patch.

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
