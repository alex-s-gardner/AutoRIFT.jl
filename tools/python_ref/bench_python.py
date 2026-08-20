#!/usr/bin/env python
"""Time the OpenCV primitives AutoRIFT.jl replaces, in the same JSON schema the
Julia suite emits.

The result is committed as ``benchmark/results/python.json``, so
``benchmark/compare.jl`` can print a per-kernel ``speedup_vs_python`` alongside
every Julia measurement. That makes "match or beat OpenCV" a tracked number rather
than a claim, and it makes a *regression* against OpenCV visible in the same table
as a regression against the previous Julia run.

    mamba run -n autorift-cv python tools/python_ref/bench_python.py

Benchmark names must match the Julia suite's exactly, or the comparison silently
finds no overlap. The names here are the ones `benchmark/suite/*.jl` uses.

What this does NOT measure: the full autoRIFT pipeline. That needs the reference
installed with its ISCE3/GDAL stack and a real image pair, so the end-to-end
baseline is captured separately from the production container. These numbers cover
the primitives, which is where the port's performance is won or lost.
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmark" / "results" / "python.json"

# OpenCV parallelises internally by default. Disabled so that a Python number and
# a Julia number describe the same thing: one core doing one kernel. Comparing
# single-threaded Julia against multi-threaded OpenCV would understate the port,
# and comparing the reverse would overstate it. Thread scaling is measured
# separately, per-implementation.
cv2.setNumThreads(1)


def timeit(fn, *, min_time=0.5, min_reps=5):
    """Return the minimum wall time over enough repetitions to be meaningful.

    Minimum rather than mean, matching the Julia suite: interference can only make
    a sample slower, so the minimum is the least noisy estimate of the code's
    actual cost.
    """
    fn()  # warm caches and any lazy initialisation
    reps, elapsed = 0, 0.0
    best = float("inf")
    while reps < min_reps or elapsed < min_time:
        t0 = time.perf_counter_ns()
        fn()
        dt = time.perf_counter_ns() - t0
        best = min(best, dt)
        elapsed += dt / 1e9
        reps += 1
    return {
        "min_ns": best,
        "median_ns": best,   # not tracked separately; kept for schema symmetry
        "mean_ns": best,
        "max_ns": best,
        "allocs": 0,          # not measurable here; Julia's value is what matters
        "memory_bytes": 0,
        "samples": reps,
    }


def texture(shape, seed=0, dtype=np.float32):
    """Band-limited random texture, matching the Julia suite's input."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(shape)
    for _ in range(3):
        a = cv2.blur(a, (3, 3), borderType=cv2.BORDER_REFLECT_101)
    a -= a.min()
    if a.max() > 0:
        a /= a.max()
    if np.issubdtype(dtype, np.integer):
        return np.round(a * np.iinfo(dtype).max).astype(dtype)
    return a.astype(dtype)


# Must match `CHIP_RADIUS` in benchmark/benchmarks.jl.
CHIP_RADIUS = ((32, 6), (32, 25), (64, 25), (128, 25), (64, 50))
SCENE_SIZES = (256, 1024, 4096)


def bench_correlate(results):
    """matchTemplate: the kernel the whole port's throughput rests on."""
    for chip, radius in CHIP_RADIUS:
        win = chip + 2 * radius - 1
        for dname, dtype in (("uint8", np.uint8), ("float32", np.float32)):
            search = texture((win, win), seed=chip + radius, dtype=dtype)
            tmpl = np.ascontiguousarray(
                search[radius:radius + chip, radius:radius + chip]
            )
            # v2.1.2 uses CV_TM_CCOEFF_NORMED for every input type; the
            # CCORR_NORMED path that older releases took for float input was a bug.
            name = f"correlate/surface zncc c{chip} r{radius} {dname}"
            results[name] = timeit(
                lambda s=search, t=tmpl: cv2.matchTemplate(s, t, cv2.TM_CCOEFF_NORMED)
            )

            # Surface plus peak location: what one grid point actually costs.
            def point(s=search, t=tmpl):
                r = cv2.matchTemplate(s, t, cv2.TM_CCOEFF_NORMED)
                cv2.minMaxLoc(r)

            results[f"correlate/point zncc c{chip} r{radius} {dname}"] = timeit(point)


def bench_peak(results):
    """minMaxLoc alone, to separate peak location from surface formation."""
    for radius in (6, 25, 50):
        surface = texture((2 * radius, 2 * radius), seed=radius, dtype=np.float32)
        results[f"correlate/peak r{radius}"] = timeit(
            lambda s=surface: cv2.minMaxLoc(s)
        )


def bench_subpixel(results):
    """The pyrUp cascade, at each upsampling factor the reference uses.

    Measured per factor because the cost is not linear in it: each step quadruples
    the surface area, so the last doubling costs as much as everything before it.
    At 128x a 5x5 patch becomes 640x640 — which is why evaluating the composite
    interpolant directly is expected to be a large win, and why that claim needs a
    baseline to be checked against.
    """
    patch = texture((5, 5), seed=3, dtype=np.float32)
    for factor in (16, 32, 64, 128):
        def cascade(p=patch, f=factor):
            cur = p
            k = 1
            while k < f:
                cur = cv2.pyrUp(cur)
                k *= 2
            cv2.minMaxLoc(cur)

        results[f"correlate/subpixel x{factor}"] = timeit(cascade)


def bench_preprocess(results):
    """The high-pass filter: in production, the only filter inside the correlator.

    Reported per scene size since it scales with image area rather than with grid
    point count.
    """
    for size in SCENE_SIZES:
        src = texture((size, size), seed=1, dtype=np.float32)
        for width in (5, 21):   # 5 is the default; 21 is used for Sentinel-1
            k = -np.full((width, width), 1.0 / (width * width), dtype=np.float32)
            k[width // 2, width // 2] += 1.0
            results[f"preprocess/highpass w{width} {size}x{size}"] = timeit(
                lambda s=src, kk=k: cv2.filter2D(
                    s, -1, kk, borderType=cv2.BORDER_CONSTANT
                )
            )

        # The Wallis filter's two box passes, the dominant cost of that path.
        box = np.full((5, 5), 1.0 / 25, dtype=np.float32)

        def wallis_stats(s=src, b=box):
            cv2.filter2D(s, -1, b, borderType=cv2.BORDER_CONSTANT)
            cv2.filter2D(s * s, -1, b, borderType=cv2.BORDER_REFLECT)

        results[f"preprocess/wallis stats w5 {size}x{size}"] = timeit(wallis_stats)


def bench_resample(results):
    """The resampling modes used between chip-size levels."""
    for size in (1024,):
        src = texture((size, size), seed=2, dtype=np.float32)
        for iname, interp in (
            ("nearest", cv2.INTER_NEAREST),
            ("bilinear", cv2.INTER_LINEAR),
            ("area", cv2.INTER_AREA),
            ("cubic", cv2.INTER_CUBIC),
        ):
            for scale in (0.5, 2.0):
                dst = (int(size * scale), int(size * scale))
                results[f"multichip/resample {iname} {scale}x {size}"] = timeit(
                    lambda s=src, d=dst, i=interp: cv2.resize(s, d, interpolation=i)
                )


def bench_window(results):
    """Sliding-window reductions, via the same scipy path v2.1.2's colfilt uses.

    Skipped if scipy is unavailable, since these are the one group that is not
    pure OpenCV.
    """
    try:
        from scipy.ndimage import generic_filter, maximum_filter, median_filter
    except ImportError:
        print("  scipy unavailable; skipping window group", file=sys.stderr)
        return

    for size in (512,):
        a = texture((size, size), seed=5, dtype=np.float64)
        for width in (3, 5, 12, 48):
            results[f"window/max w{width} {size}x{size}"] = timeit(
                lambda x=a, w=width: maximum_filter(x, size=w, mode="reflect")
            )
        for width in (3, 5):
            results[f"window/median w{width} {size}x{size}"] = timeit(
                lambda x=a, w=width: median_filter(x, size=w, mode="constant",
                                                   cval=np.nan)
            )


def bench_morphology(results):
    """The exact Euclidean distance transform used to dilate the coarse mask."""
    rng = np.random.default_rng(4)
    for size in (512, 2048):
        mask = (rng.random((size, size)) > 0.3).astype(np.uint8)
        results[f"window/disttransform {size}x{size}"] = timeit(
            lambda m=mask: cv2.distanceTransform(m, cv2.DIST_L2,
                                                 cv2.DIST_MASK_PRECISE)
        )


def environment():
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return "unknown"

    return {
        "python_version": sys.version.split()[0],
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "cpu": platform.processor() or platform.machine(),
        "cpu_threads": 0,
        "nthreads": 1,           # cv2.setNumThreads(1) above
        "os": platform.system(),
        "arch": platform.machine(),
        "hostname": socket.gethostname(),
        "git_sha": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "timestamp": str(int(time.time())),
    }


GROUPS = (
    ("correlate", bench_correlate),
    ("peak", bench_peak),
    ("subpixel", bench_subpixel),
    ("preprocess", bench_preprocess),
    ("resample", bench_resample),
    ("window", bench_window),
    ("morphology", bench_morphology),
)


def main():
    only = set(sys.argv[1:])
    results = {}
    for name, fn in GROUPS:
        if only and name not in only:
            continue
        print(f"  {name}...", flush=True)
        fn(results)

    payload = {
        "schema_version": 1,
        "impl": "python",
        "environment": environment(),
        "benchmarks": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\n{len(results)} benchmarks written to {OUT}")

    width = max(len(k) for k in results) if results else 0
    for name in sorted(results):
        ns = results[name]["min_ns"]
        unit, div = ("ns", 1) if ns < 1e3 else ("us", 1e3) if ns < 1e6 else \
                    ("ms", 1e6) if ns < 1e9 else ("s", 1e9)
        print(f"  {name:<{width}}  {ns / div:>9.2f} {unit}")


if __name__ == "__main__":
    main()
