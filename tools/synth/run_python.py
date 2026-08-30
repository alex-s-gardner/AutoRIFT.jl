"""The reference implementation's arms of the synthetic comparison.

Two arms, matching the Julia split so a difference is attributable:

  py-correlator  `arImgDisp_s` at one chip size -- matching and refinement alone.
  py-pipeline    `runAutorift` -- the pyramid, coarse pass, `DISP_FILT`, hole fill and merge.

    micromamba run -n arift-ref python tools/synth/run_python.py [case ...]

`DataType` follows the case: `1` keeps the float path, and `0` runs `uniform_data_type`'s uint8
quantization. The quantized cases are already quantized in the bundle so both implementations correlate
identical numbers, which means `DataType = 1` is correct even there -- requantizing values that are
already integers 0-255 would rescale them a second time.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bundle import cases, read_bundle, write_arm

SCENES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenes")


def correlator_arm(ref, sec, arrays, scalars):
    """`arImgDisp_s` on the case's grid, in the matrix convention.

    Argument order follows the reference's own call sites (`autoRIFT.py:673,747`), which pass
    `(self.I2, self.I1)`: the C++ binds the first to the chip and the second to the search window, so
    the chip comes from the secondary and the window from the reference. AutoRIFT.jl assigns them the
    same way.
    """
    from autoRIFT.autoRIFT import arImgDisp_s

    chip = scalars["chip"]
    radius = scalars["radius"]
    oversample = scalars["upsampling"]

    # Julia's grid is 1-based pixel centres and the reference's is 0-based. `arImgDisp_s` adds its own
    # +0.5 internally, which is how an even chip's estimate lands on the point, so only the origin
    # shift is applied here.
    x_grid = np.ascontiguousarray(arrays["grid_x"] - 1.0, dtype=np.float32)
    y_grid = np.ascontiguousarray(arrays["grid_y"] - 1.0, dtype=np.float32)
    shape = x_grid.shape
    ones = np.ones(shape, dtype=np.float32)

    dx, dy = arImgDisp_s(
        np.ascontiguousarray(sec, dtype=np.float32),
        np.ascontiguousarray(ref, dtype=np.float32),
        x_grid.copy(),
        y_grid.copy(),
        ones * np.float32(chip),
        ones * np.float32(chip),
        ones * np.float32(radius),
        ones * np.float32(radius),
        np.zeros(shape, dtype=np.float32),
        np.zeros(shape, dtype=np.float32),
        True,
        oversample,
    )
    dx = np.asarray(dx, dtype=np.float32).reshape(shape)
    dy = np.asarray(dy, dtype=np.float32).reshape(shape)
    # `arImgDisp_s` returns cartesian y (up positive) where AutoRIFT.jl reports dy down rows, so the
    # reference's own final `Dy = -Dy` is undone to put both in the matrix convention.
    return dx, -dy


def pipeline_arm(ref, sec, arrays, scalars, chip_size_max=None):
    """`runAutorift`: the whole reference pipeline.

    `chip_size_max` defaults to the case's chip so this is single-level, matching the Julia pipeline
    arm -- a difference from the correlator arm is then the machinery rather than a coarser chip.
    """
    from autoRIFT import autoRIFT

    chip = scalars["chip"]
    cmax = chip if chip_size_max is None else chip_size_max
    spacing = scalars["grid_spacing"]

    obj = autoRIFT()
    obj.I1 = np.ascontiguousarray(ref, dtype=np.float32)
    obj.I2 = np.ascontiguousarray(sec, dtype=np.float32)
    # The bundle already holds exactly the numbers to correlate, including for the quantized cases, so
    # the float path is right throughout: `DataType = 0` would rescale integers 0-255 a second time.
    obj.DataType = 1
    obj.zeroMask = None
    obj.xGrid = np.ascontiguousarray(arrays["grid_x"] - 1.0, dtype=np.float32)
    obj.yGrid = np.ascontiguousarray(arrays["grid_y"] - 1.0, dtype=np.float32)
    obj.ChipSize0X = chip
    obj.ChipSizeMinX = chip
    obj.ChipSizeMaxX = cmax
    obj.GridSpacingX = spacing
    obj.SkipSampleX = spacing
    obj.SkipSampleY = spacing
    obj.SearchLimitX = scalars["radius"]
    obj.SearchLimitY = scalars["radius"]
    obj.OverSampleRatio = scalars["upsampling"]
    obj.Dx0 = 0
    obj.Dy0 = 0
    obj.MultiThread = 0
    obj.runAutorift()

    dx = np.asarray(obj.Dx, dtype=np.float32)
    dy = np.asarray(obj.Dy, dtype=np.float32)
    chip_used = np.asarray(obj.ChipSizeX, dtype=np.float32)
    # Same cartesian-to-matrix flip as the correlator arm.
    return dx, -dy, chip_used


def main():
    for name in cases(SCENES, sys.argv[1:]):
        path = os.path.join(SCENES, name)
        arrays, scalars = read_bundle(path)
        ref, sec = arrays["reference"], arrays["secondary"]

        t0 = time.time()
        dx, dy = correlator_arm(ref, sec, arrays, scalars)
        t1 = time.time() - t0
        write_arm(path, "py_correlator", {"dx": dx, "dy": dy}, t1)

        t0 = time.time()
        pdx, pdy, pcs = pipeline_arm(ref, sec, arrays, scalars)
        t2 = time.time() - t0
        write_arm(path, "py_pipeline",
                  {"dx": pdx, "dy": pdy, "chip_size": pcs.astype(np.int32)}, t2)

        print("%-28s correlator %5.1f s  pipeline %5.1f s  grid %s / %s"
              % (name, t1, t2, dx.shape, pdx.shape))


if __name__ == "__main__":
    main()
