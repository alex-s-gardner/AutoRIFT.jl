#!/usr/bin/env python3
"""End-to-end benchmark of the reference autoRIFT pipeline.

The kernel benchmarks in `bench_python.py` compare against OpenCV, which is the right target for
`matchTemplate` and the filters but says nothing about the pipeline as a whole. This runs the real
`autoRIFT.runAutorift()` so that `speedup_vs_python` can be stated end to end rather than
per kernel.

Uses the conda-forge build rather than the source tree, which matters: `autoriftcore` is a
compiled C++ extension needing OpenCV headers, and `autorift 2.1.2` is packaged prebuilt for
osx-arm64 and linux-64. That removes the build from the loop entirely.

    mamba create -n arift-ref -c conda-forge python=3.10 autorift=2.1.2
    mamba run -n arift-ref python tools/python_ref/bench_reference.py

# Getting the reference to run at all

Four things are needed that its docstring does not mention, all found by tracing `sys.exit` calls:

  * Every per-point field must be a grid-shaped **float32 array**. The class defaults are plain
    Python ints (`SearchLimitX = 25`), and the C binding rejects those — `testautoRIFT.py` gets
    away with it because it multiplies by a mask, which promotes. Passing `np.float32` scalars is
    not enough either; the scalar branch demands `isinstance(x, np.float32)`.
  * `ScaleChipSizeY` must be set, or chip height is derived from an unset attribute.
  * `preprocess_filt_hps()` then `uniform_data_type()`, in that order, before `runAutorift()`.
  * The grid must survive decimation by `sparseSearchSampleRate` (4): a grid that decimates to a
    single point takes the scalar branch above and dies.

# Timing

`runAutorift()` pays a large first-call cost — 12.7 s against a 456 ms steady state at 1024^2 —
so the minimum of several runs is reported, matching how the Julia suite measures.
"""

import gc
import time

import cv2
import numpy as np
from autoRIFT import autoRIFT

def texture(n, seed=1):
    """Band-limited random texture. Matches `bench_python.py`'s recipe, not its exact pixels."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    for _ in range(3):
        a = cv2.blur(a, (3,3), borderType=cv2.BORDER_REFLECT_101)
    a -= a.min(); a /= a.max()
    return a.astype(np.float32)

def run(n, chip=32, radius=25, spacing=32):
    a = texture(n)
    b = np.roll(np.roll(a, -4, axis=0), 6, axis=1)
    o = autoRIFT()
    o.I1 = a          # reference
    o.I2 = b          # secondary
    m = spacing
    ys, xs = np.meshgrid(np.arange(m, n-m, m), np.arange(m, n-m, m), indexing='ij')
    o.xGrid = xs.astype(np.float32); o.yGrid = ys.astype(np.float32)
    o.ScaleChipSizeY = 1
    o.ChipSize0X = chip
    # Mirror testautoRIFT.py: broadcast the scalars to grid-shaped float32 arrays. The class
    # defaults are plain Python ints, and the C binding demands float32 for every per-point field.
    shp = xs.shape
    o.ChipSizeMinX = np.full(shp, chip, dtype=np.float32)
    o.ChipSizeMaxX = np.full(shp, chip, dtype=np.float32)
    o.SearchLimitX = np.full(shp, radius, dtype=np.float32)
    o.SearchLimitY = np.full(shp, radius, dtype=np.float32)
    o.Dx0 = np.zeros(shp, dtype=np.float32)
    o.Dy0 = np.zeros(shp, dtype=np.float32)
    o.GridSpacingX = spacing
    o.zeroMask = None
    o.preprocess_filt_hps()
    o.uniform_data_type()
    t0 = time.perf_counter_ns()
    o.runAutorift()
    dt = time.perf_counter_ns() - t0
    ok = np.sum(~np.isnan(o.Dx))
    return dt/1e6, o.Dx.shape, ok, np.nanmedian(o.Dx), np.nanmedian(o.Dy)

# 512 is retained although it measures zero points, and the reason is worth seeing rather than
# hiding: a 512 scene at spacing 32 gives a 14x14 grid, whose coarse pass decimates to 3x3 — too
# small for the 5-point outlier filter, so every point is rejected. The reference and this port
# agree on that, which is itself a useful check. The 1024 case is the one to compare.
def rss_growth(n=512, npairs=30):
    """RSS growth per pair, which is where the two implementations differ most.

    Reported as *current* RSS after an explicit `gc.collect()`, not peak: the question is whether
    memory accumulates, and a peak figure cannot distinguish accumulation from allocator slack.
    Julia's equivalent measurement tracks `Base.gc_live_bytes` for the same reason, and its live
    heap is flat across thirty pairs.

    Requires psutil; skipped with a note if absent rather than failing the run.
    """
    try:
        import psutil
    except ImportError:
        print("rss growth: skipped (psutil not installed)")
        return
    proc = psutil.Process()
    cur = lambda: proc.memory_info().rss / 2**20
    run(n)
    gc.collect()
    first = cur()
    for _ in range(npairs - 1):
        run(n)
    gc.collect()
    last = cur()
    print(f"rss growth over {npairs} pairs at {n}x{n}: "
          f"{first:.1f} -> {last:.1f} MiB = {(last-first)/(npairs-1):+.1f} MiB/pair")


for n in (512, 1024):
    best = None; shape = ok = mdx = mdy = None
    for rep in range(3):
        ms, shape, ok, mdx, mdy = run(n)
        best = ms if best is None else min(best, ms)
    print(f"n={n}: {best:9.1f} ms (min of 3)  grid {shape}  measured {ok}  dx={mdx:.2f} dy={mdy:.2f}")

rss_growth()
