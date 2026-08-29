"""Stage 2 of the A/B: the reference's whole pipeline on the arrays AutoRIFT.jl filtered.

`runAutorift` and not `arImgDisp_s`: this stage compares the machinery *around* the correlator --
the chip-size pyramid, the sparse/coarse pass, the `DISP_FILT` outlier test, the hole filling and
the smallest-chip-wins merge. Stage 1 already established that the correlator itself agrees.

The preprocessing is deliberately skipped on this side (`obj.I1`/`obj.I2` are set to the arrays
AutoRIFT.jl already high-pass filtered, and no `preprocess_filt_*` is called), so a filter
difference cannot masquerade as a pyramid difference.

    micromamba run -n arift-ref python tools/ab/stage2_python.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.join(HERE, "stage2")

DTYPES = {"Float32": np.float32, "Float64": np.float64, "Int32": np.int32}


def read_bundle(path):
    """The arrays and scalars `stage2_julia.jl` wrote, un-transposed from column-major."""
    arrays, scalars = {}, {}
    with open(os.path.join(path, "manifest.txt")) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 2:
                scalars[parts[0]] = int(parts[1])
                continue
            name, dtype, shape = parts
            dims = tuple(int(v) for v in shape.split("x"))
            raw = np.fromfile(os.path.join(path, name + ".bin"), dtype=DTYPES[dtype])
            arrays[name] = raw.reshape(dims[::-1]).T
    return arrays, scalars


def main():
    from autoRIFT import autoRIFT

    arrays, scalars = read_bundle(BUNDLE)

    obj = autoRIFT()
    # Already high-pass filtered by AutoRIFT.jl, so no preprocessing runs here. `DataType = 1`
    # keeps the float path: `uniform_data_type` would otherwise requantize to uint8 and the two
    # sides would no longer be correlating the same numbers.
    obj.I1 = np.ascontiguousarray(arrays["filtered_reference"], dtype=np.float32)
    obj.I2 = np.ascontiguousarray(arrays["filtered_secondary"], dtype=np.float32)
    obj.DataType = 1
    obj.zeroMask = None

    # Julia's grid is 1-based pixel centres and the reference's is 0-based. `runAutorift` applies
    # its own `round(xGrid) + 0.5`, so only the origin shift is applied here.
    obj.xGrid = np.ascontiguousarray(arrays["grid_x"] - 1.0, dtype=np.float32)
    obj.yGrid = np.ascontiguousarray(arrays["grid_y"] - 1.0, dtype=np.float32)

    chip = scalars["chip"]
    obj.ChipSize0X = chip
    obj.ChipSizeMinX = chip
    obj.ChipSizeMaxX = scalars["chip_max"]
    obj.GridSpacingX = scalars["grid_spacing"]
    obj.SkipSampleX = scalars["grid_spacing"]
    obj.SkipSampleY = scalars["grid_spacing"]
    obj.SearchLimitX = scalars["radius"]
    obj.SearchLimitY = scalars["radius"]
    obj.OverSampleRatio = scalars["upsampling"]
    obj.Dx0 = 0
    obj.Dy0 = 0
    # `mpflag = 0` is what production runs (REFERENCE.md), and the reference's own note says
    # threading cannot change results anyway: each grid point writes a distinct output element.
    obj.MultiThread = 0

    obj.runAutorift()

    shape = obj.Dx.shape
    dx = np.asarray(obj.Dx, dtype=np.float32)
    # Undo the reference's final cartesian flip so both sides are in the matrix convention, then
    # negate both axes into AutoRIFT.jl's secondary-to-reference convention. See tools/ab/compare.jl.
    dy = -np.asarray(obj.Dy, dtype=np.float32)

    chip_size = np.asarray(obj.ChipSizeX, dtype=np.int32)
    interp = np.asarray(obj.InterpMask, dtype=np.int32)

    print("grid            ", shape, " (julia grid was",
          arrays["grid_x"].shape, ")")
    finite = np.isfinite(dx)
    print("python measured ", int(finite.sum()),
          "({:.1f}%)".format(100.0 * finite.mean()))
    print("interpolated    ", int(interp.sum()))
    for cs in sorted(set(chip_size.ravel().tolist())):
        print("  chip {:>3} -> {} points".format(cs, int((chip_size == cs).sum())))

    for name, A in (("python_dx", dx), ("python_dy", dy),
                    ("python_chip_size", chip_size), ("python_interpolated", interp)):
        np.asfortranarray(A).T.tofile(os.path.join(BUNDLE, name + ".bin"))
    with open(os.path.join(BUNDLE, "python_shape.txt"), "w") as fh:
        fh.write("{} {}\n".format(shape[0], shape[1]))
    print("wrote           ", BUNDLE)


if __name__ == "__main__":
    sys.exit(main())
