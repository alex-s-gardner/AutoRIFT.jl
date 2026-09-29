"""End-to-end runtime and peak memory for the Python reference on one golden case's captured inputs.

    micromamba run -n arift-ref python tools/ab/golden_python.py <capture-dir> [--reps 1] [--cap 1800]

The captured `in_*` arrays are the reference's own inputs at the moment it called the correlator, so
this runs the reference on exactly the data AutoRIFT.jl is measured against — no re-derivation of the
grid, the search limits or the priors, which is where a Python/Julia comparison usually goes wrong.
They are `xchg` files, so they carry their own shape and element type and cannot be read transposed.

`tools/ab/bench_python.py` is the same measurement on a *bundle*, where the chip size and radius are
scalars. A golden capture is a geogrid: `SearchLimitX`, `ChipSizeMinX/MaxX` and `Dx0/Dy0` are per-point
rasters, and handing the reference a scalar instead would be a different, easier problem.

**One case per process, because `ru_maxrss` is a high-water mark.** Two cases measured in one process
both report the larger. The driver spawns one of these per case for that reason.

**`runAutorift` mutates its inputs** — it truncates `xGrid` and zeroes `SearchLimitX` — so the object is
rebuilt per repetition, or a second call would time a smaller problem. `bench_python.py` records the
same constraint.

**Threads.** OpenCV keeps its own pool and does not read `OMP_NUM_THREADS`, so `cv2.setNumThreads` is
called explicitly. On macOS that call is a no-op because OpenCV is built on GCD, which is why the
actual `cv2.getNumThreads()` is reported rather than assumed: the honest comparison is both sides at
full core count, and this records what the run really used.
"""

import json
import os
import resource
import signal
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import xchg  # noqa: E402


class Timeout(Exception):
    pass


def _alarm(_sig, _frm):
    raise Timeout()


def read_capture(d):
    """The `in_*` arrays and the scalars from a capture directory."""
    manifest = json.load(open(os.path.join(d, "call1.json")))
    arrays = {}
    for name, meta in manifest["arrays"].items():
        if not name.startswith("in_"):
            continue
        p = os.path.join(d, name)
        if not os.path.exists(p):
            p = os.path.join(d, meta["file"])
        arrays[name] = xchg.read(p)
    return arrays, manifest["scalars"]


def build(arrays, scalars):
    """A configured `autoRIFT` on a captured geogrid.

    Every field is taken from the capture rather than recomputed. `OverSampleRatio` is stored with
    string keys by JSON and the reference indexes it by the integer chip size, so it is converted back.
    """
    import importlib

    M = importlib.import_module("autoRIFT.autoRIFT")
    obj = M.autoRIFT()
    # Not every capture records `DataType` — the reference only sets it away from its default on the
    # radar path — so it is inferred from the imagery when absent. `0` selects the integer correlator
    # and `1` the float one, and choosing wrong would correlate the right data with the wrong transform.
    dt = int(scalars["DataType"]) if "DataType" in scalars else \
        (0 if arrays["in_I1"].dtype == np.uint8 else 1)
    want = np.uint8 if dt == 0 else np.float32
    obj.I1 = np.ascontiguousarray(arrays["in_I1"], dtype=want)
    obj.I2 = np.ascontiguousarray(arrays["in_I2"], dtype=want)
    obj.DataType = dt
    obj.zeroMask = None
    # Already the reference's own post-rewrite grid — half-integer and with nodata replaced — so it is
    # handed over unchanged. Shifting it here would move every search centre.
    obj.xGrid = np.ascontiguousarray(arrays["in_xGrid"], dtype=np.float32)
    obj.yGrid = np.ascontiguousarray(arrays["in_yGrid"], dtype=np.float32)
    obj.SearchLimitX = np.ascontiguousarray(arrays["in_SearchLimitX"], dtype=np.int32)
    obj.SearchLimitY = np.ascontiguousarray(arrays["in_SearchLimitY"], dtype=np.int32)
    obj.ChipSizeMinX = np.ascontiguousarray(arrays["in_ChipSizeMinX"], dtype=np.int32)
    obj.ChipSizeMaxX = np.ascontiguousarray(arrays["in_ChipSizeMaxX"], dtype=np.int32)
    obj.Dx0 = np.ascontiguousarray(arrays["in_Dx0"], dtype=np.float32)
    obj.Dy0 = np.ascontiguousarray(arrays["in_Dy0"], dtype=np.float32)
    ratio = {int(k): v for k, v in scalars["OverSampleRatio"].items()}
    obj.OverSampleRatio = ratio
    for field, key in (("ChipSize0X", "ChipSize0X"), ("ScaleChipSizeY", "ScaleChipSizeY"),
                       ("GridSpacingX", "GridSpacingX"), ("SkipSampleX", "SkipSampleX"),
                       ("SkipSampleY", "SkipSampleY"), ("WallisFilterWidth", "WallisFilterWidth"),
                       ("BuffDistanceC", "BuffDistanceC"), ("CoarseCorCutoff", "CoarseCorCutoff"),
                       ("sparseSearchSampleRate", "sparseSearchSampleRate"),
                       ("minSearch", "minSearch"), ("FracValid", "FracValid"),
                       ("FracSearch", "FracSearch"), ("FiltWidth", "FiltWidth"),
                       ("Iter", "Iter"), ("MadScalar", "MadScalar"),
                       ("colfiltChunkSize", "colfiltChunkSize"),
                       ("fillFiltWidth", "fillFiltWidth")):
        if key in scalars:
            setattr(obj, field, scalars[key])
    obj.MultiThread = int(scalars.get("MultiThread", 0))
    return obj


def main():
    d = sys.argv[1]
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 1
    cap = int(sys.argv[sys.argv.index("--cap") + 1]) if "--cap" in sys.argv else 1800

    import cv2

    req = os.environ.get("OMP_NUM_THREADS")
    if req:
        cv2.setNumThreads(int(req))

    arrays, scalars = read_capture(d)
    t0 = time.perf_counter()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(cap)
    status, best, measured = "ok", float("nan"), 0
    shape = arrays["in_xGrid"].shape
    try:
        times = []
        for _ in range(reps):
            obj = build(arrays, scalars)
            t = time.perf_counter()
            obj.runAutorift()
            times.append(time.perf_counter() - t)
            measured = int(np.isfinite(np.asarray(obj.Dx)).sum())
        best = min(times)
    except Timeout:
        status = "timeout_%ds" % cap
    except Exception as e:  # a case the reference cannot run is a result, not a crash
        status = "error:" + type(e).__name__ + ":" + str(e)[:120].replace("\t", " ")
    finally:
        signal.alarm(0)

    # macOS reports `ru_maxrss` in bytes where Linux reports kilobytes.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = rss if sys.platform == "darwin" else rss * 1024
    print("RESULT\t%s\t%s\t%.3f\t%.3f\t%d\t%d\t%d\t%d\t%s" % (
        os.path.basename(os.path.dirname(os.path.dirname(d))), status, best,
        time.perf_counter() - t0, peak, measured, shape[0], shape[1],
        cv2.getNumThreads()))


if __name__ == "__main__":
    main()
