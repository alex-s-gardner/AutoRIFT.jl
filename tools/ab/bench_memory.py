"""Peak resident memory of the reference pipeline, and of a baseline that skips the correlation.

The difference between the two is the pipeline's cost, isolated from the interpreter floor, NumPy,
OpenCV, and the input arrays -- the same subtraction `benchmark/memory.jl` performs on the Julia
side, so the two numbers are comparable.

One measurement per process. `ru_maxrss` is a high-water mark, so a baseline and a working run in
the same process would report the larger of the two twice.

`--baseline` allocates the inputs and configures the object but never calls `runAutorift`.

    micromamba run -n arift-ref python tools/ab/bench_memory.py [--baseline]
"""

import gc
import importlib
import os
import resource
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def peak_mib():
    """Peak RSS of this process in MiB.

    `ru_maxrss` is bytes on macOS and kilobytes on Linux -- the one platform difference that would
    silently report a 1024x wrong answer.
    """
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / 2**20 if sys.platform == "darwin" else v / 2**10


def live_mib():
    """Resident memory after a full collection, as the closest available analogue to Julia's
    `gc_live_bytes`: what is still reachable rather than what the allocator has kept. Read from
    `psutil` when present, otherwise reported as NaN rather than guessed at."""
    gc.collect()
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**20
    except Exception:
        return float("nan")


def main():
    sys.path.insert(0, HERE)

    baseline = "--baseline" in sys.argv
    npix = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3072

    # The same bare `Float32` pair the Julia side reads, and the grid built to match
    # `gridpoints`. Reading the stage-2 bundle instead would add its own filtered copies and the
    # 79 MiB of arrays beside them to the baseline, which is exactly the transient that made the
    # first version of this measurement meaningless.
    raw = np.fromfile(os.path.join(HERE, "bench_pair_{}.bin".format(npix)), dtype=np.float32)
    ref = np.ascontiguousarray(raw[: npix * npix].reshape(npix, npix, order="F"))
    sec = np.ascontiguousarray(raw[npix * npix:].reshape(npix, npix, order="F"))
    del raw
    gc.collect()

    chip_max, spacing, radius = 64, 8, 20
    # `gridpoints`' inset: the widest chip's half-extent plus the radius plus one, on each side.
    margin = -(-chip_max // 2) + radius + 1
    coords = np.arange(margin + 1, npix - margin + 1, spacing, dtype=np.float32)
    x_grid = np.ascontiguousarray(np.tile(coords, (coords.size, 1)) - 1.0)
    y_grid = np.ascontiguousarray(np.tile(coords[:, None], (1, coords.size)) - 1.0)

    importlib.import_module("autoRIFT.autoRIFT")
    from autoRIFT import autoRIFT

    obj = autoRIFT()
    obj.I1, obj.I2 = ref, sec
    obj.DataType = 1
    obj.zeroMask = None
    obj.xGrid, obj.yGrid = x_grid, y_grid
    obj.ChipSize0X = 16
    obj.ChipSizeMinX = 16
    obj.ChipSizeMaxX = chip_max
    obj.GridSpacingX = spacing
    obj.SkipSampleX = spacing
    obj.SkipSampleY = spacing
    obj.SearchLimitX = radius
    obj.SearchLimitY = radius
    obj.OverSampleRatio = 16
    obj.Dx0 = 0
    obj.Dy0 = 0
    obj.MultiThread = 0

    if not baseline:
        obj.runAutorift()

    print("{:.1f} {:.1f}".format(peak_mib(), live_mib()))


if __name__ == "__main__":
    sys.exit(main())
