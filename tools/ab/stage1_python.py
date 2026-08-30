"""Stage 1 of the A/B: the reference correlator on the arrays AutoRIFT.jl filtered.

Reads the bundle `stage1_julia.jl` wrote, calls `arImgDisp_s` on the *same* Float32 arrays at the
*same* grid points, and writes `python_dx`/`python_dy` beside the Julia ones.

`arImgDisp_s` and not `runAutorift`: stage 1 isolates the correlator, so the pyramid loop, the
coarse pass, the outlier filter and the grid truncation are all deliberately bypassed. Those are
compared in later stages, on top of a correlator already known to agree.

    micromamba run -n arift-ref python tools/ab/stage1_python.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.join(HERE, "stage1")

DTYPES = {"Float32": np.float32, "Float64": np.float64}


def read_bundle(path):
    """The arrays and scalars `stage1_julia.jl` wrote.

    Julia writes column-major and NumPy reads row-major, so each array is read with its shape
    reversed and then transposed. That is the transpose the realdata README records as the trap
    which correlates at -0.11 when missed.
    """
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
    from autoRIFT.autoRIFT import arImgDisp_s

    arrays, scalars = read_bundle(BUNDLE)
    chip = scalars["chip"]
    radius = scalars["radius"]
    oversample = scalars["upsampling"]

    ref = np.ascontiguousarray(arrays["filtered_reference"], dtype=np.float32)
    sec = np.ascontiguousarray(arrays["filtered_secondary"], dtype=np.float32)

    # Julia's grid is 1-based pixel centres; the reference's is 0-based. `arImgDisp_s` then adds
    # its own +0.5 internally (`xGrid += Px + 0.5`), which is how an even chip's estimate is
    # reported at the point rather than half a pixel off it -- so nothing is added for that here.
    x_grid = np.ascontiguousarray(arrays["grid_x"] - 1.0, dtype=np.float32)
    y_grid = np.ascontiguousarray(arrays["grid_y"] - 1.0, dtype=np.float32)

    shape = x_grid.shape
    ones = np.ones(shape, dtype=np.float32)

    # `arImgDisp_s(a, b)` cuts its chip from `b` and its search window from `a`, the reverse of what the
    # parameter names suggest: `a` binds to `I1` and `b` to `I2`, but the body calls the C++ as
    # `(I2.ravel(), I1.ravel())` (`autoRIFT.py:1251-1256`) and the C++ binds *its* first array to
    # `sec_img`, where the chip is cut from (`autoriftcoremodule.cpp:413,453`). Two swaps that compose.
    #
    # So `(ref, sec)` puts the chip on the secondary and the window on the reference, matching
    # AutoRIFT.jl. Reversing it is not a sign error: the chip then comes from the other image, so the
    # estimate describes different ground, and the residual grows with displacement wherever the field
    # deforms while staying exactly zero under uniform motion.
    dx, dy = arImgDisp_s(
        ref,
        sec,
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

    # `arImgDisp_s` returns cartesian y (up positive); AutoRIFT.jl reports dy down rows. Undo the
    # reference's own final `Dy = -Dy` so both sides are in the matrix convention.
    dy = -dy

    for name, A in (("python_dx", dx), ("python_dy", dy)):
        # Written column-major so the Julia reader needs no transpose of its own.
        np.asfortranarray(A).T.tofile(os.path.join(BUNDLE, name + ".bin"))

    finite = np.isfinite(dx)
    print("grid            ", shape)
    print("python measured ", int(finite.sum()),
          "({:.1f}%)".format(100.0 * finite.mean()))
    if finite.any():
        print("dx range        ", (float(dx[finite].min()), float(dx[finite].max())))
        print("dy range        ", (float(dy[finite].min()), float(dy[finite].max())))

    jdx = arrays["julia_dx"]
    jdy = arrays["julia_dy"]
    both = finite & np.isfinite(jdx) & np.isfinite(jdy)
    print("comparable      ", int(both.sum()))
    if both.any():
        ex = np.abs(jdx[both] - dx[both])
        ey = np.abs(jdy[both] - dy[both])
        print("|ddx| median/p95/max {:.4f} {:.4f} {:.4f}".format(
            float(np.median(ex)), float(np.percentile(ex, 95)), float(ex.max())))
        print("|ddy| median/p95/max {:.4f} {:.4f} {:.4f}".format(
            float(np.median(ey)), float(np.percentile(ey, 95)), float(ey.max())))
        within = (ex <= 0.2) & (ey <= 0.2)
        print("within 0.2 px   {}/{} ({:.2f}%)".format(
            int(within.sum()), int(both.sum()), 100.0 * within.mean()))


if __name__ == "__main__":
    sys.exit(main())
