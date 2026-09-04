"""Wall-clock for the whole pipeline, reference side, on the same window.

Times `runAutorift` on the arrays `bench_julia.jl` measured against, so the two numbers describe the
same work on the same data. Reports the minimum of `--reps` runs, matching the Julia side and
`benchmark/README.md`'s rule: interference only ever makes a sample slower.

Preprocessing is excluded on both sides -- this side is handed already-filtered arrays, exactly as
the A/B stages do -- so the comparison is of the pyramid, correlator, filter and merge rather than of
two high-pass implementations.

    OMP_NUM_THREADS=12 micromamba run -n arift-ref python tools/ab/bench_python.py [npix] [reps]
"""

import importlib
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.join(HERE, "stage2")
OUT = os.path.join(HERE, "bench_python.txt")


def build(arrays, scalars):
    """A configured `autoRIFT` on the shared window. Rebuilt per repetition: `runAutorift`
    mutates `xGrid`/`SearchLimitX` in place (it truncates the grid and zeroes search limits), so a
    second call on the same object would time a different, smaller problem."""
    M = importlib.import_module("autoRIFT.autoRIFT")
    obj = M.autoRIFT()
    obj.I1 = np.ascontiguousarray(arrays["filtered_reference"], dtype=np.float32)
    obj.I2 = np.ascontiguousarray(arrays["filtered_secondary"], dtype=np.float32)
    obj.DataType = 1
    obj.zeroMask = None
    obj.xGrid = np.ascontiguousarray(arrays["grid_x"] - 1.0, dtype=np.float32)
    obj.yGrid = np.ascontiguousarray(arrays["grid_y"] - 1.0, dtype=np.float32)
    obj.ChipSize0X = scalars["chip"]
    obj.ChipSizeMinX = scalars["chip"]
    obj.ChipSizeMaxX = scalars["chip_max"]
    obj.GridSpacingX = scalars["grid_spacing"]
    obj.SkipSampleX = scalars["grid_spacing"]
    obj.SkipSampleY = scalars["grid_spacing"]
    obj.SearchLimitX = scalars["radius"]
    obj.SearchLimitY = scalars["radius"]
    obj.OverSampleRatio = scalars["upsampling"]
    obj.Dx0 = 0
    obj.Dy0 = 0
    obj.MultiThread = 0
    return obj


def main():
    sys.path.insert(0, HERE)
    from stage2_python import read_bundle

    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    arrays, scalars = read_bundle(BUNDLE)
    shape = arrays["grid_x"].shape

    import cv2
    threads = os.environ.get("OMP_NUM_THREADS", "(unset)")
    # OpenCV keeps its own pool and does not read `OMP_NUM_THREADS`, so without this it runs on
    # every core while the Julia side is capped -- and the comparison silently becomes 12 threads
    # against 10. `matchTemplate` is called per grid point from inside the OpenMP loop, so both
    # pools matter.
    if threads != "(unset)":
        cv2.setNumThreads(int(threads))
    print("python {}  numpy {}  cv2 {}".format(
        sys.version.split()[0], np.__version__, cv2.__version__))
    print("OMP_NUM_THREADS={}  cv2.getNumThreads()={}".format(
        threads, cv2.getNumThreads()))
    print("grid handed in {}".format(shape))

    # One untimed run: numba compiles the `colfilt` reducers on first call, which is a one-off
    # cost the Julia side also excludes from its timing (it warms up too).
    t0 = time.perf_counter()
    build(arrays, scalars).runAutorift()
    warm = time.perf_counter() - t0
    print("warmup (numba compile) ... {:.2f} s".format(warm))

    times = []
    measured = 0
    for _ in range(reps):
        obj = build(arrays, scalars)
        t0 = time.perf_counter()
        obj.runAutorift()
        times.append(time.perf_counter() - t0)
        measured = int(np.isfinite(np.asarray(obj.Dx)).sum())

    n = int(np.prod(obj.Dx.shape))
    print()
    print("runAutorift  min {:.3f} s   median {:.3f} s   (n={})".format(
        min(times), float(np.median(times)), reps))
    print("grid {} = {} points, {} measured ({:.1f}%)".format(
        obj.Dx.shape, n, measured, 100.0 * measured / n))
    print("throughput {:.0f} points/s (min)".format(n / min(times)))

    with open(OUT, "w") as fh:
        fh.write("npix {}\n".format(sys.argv[1] if len(sys.argv) > 1 else "?"))
        fh.write("omp_num_threads {}\n".format(threads))
        fh.write("cv2_threads {}\n".format(cv2.getNumThreads()))
        fh.write("python_version {}\n".format(sys.version.split()[0]))
        fh.write("grid {}x{}\n".format(obj.Dx.shape[0], obj.Dx.shape[1]))
        fh.write("points {}\n".format(n))
        fh.write("measured {}\n".format(measured))
        fh.write("warmup_s {}\n".format(warm))
        fh.write("min_s {}\n".format(min(times)))
        fh.write("median_s {}\n".format(float(np.median(times))))
        fh.write("reps {}\n".format(reps))
    print("wrote", OUT)


if __name__ == "__main__":
    sys.exit(main())
