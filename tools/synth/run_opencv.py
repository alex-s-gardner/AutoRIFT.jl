"""Raw OpenCV's arms of the synthetic comparison: template matching with no autoRIFT machinery.

Two arms, because a sub-pixel estimator has to come from somewhere and the choice dominates the result:

  cv-pyrup     `matchTemplate` refined by the same `pyrUp` cascade both autoRIFTs use. Refinement held
               equal, so a difference from either autoRIFT arm is the matching or the machinery around
               it -- this is the like-for-like arm.
  cv-parabola  `matchTemplate` refined by a 3-point parabola fit, which is what a naive user writes. A
               parabola peak-locks toward integer offsets, so this arm's error is dominated by its
               estimator; the gap between the two arms is the size of that confound.

    micromamba run -n autorift-cv python tools/synth/run_opencv.py [case ...]

No pyramid, no coarse pass, no outlier filter, no hole fill: whatever these arms get, they get from
`cv2.matchTemplate(TM_CCOEFF_NORMED)` and arithmetic.
"""

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bundle import cases, grid_axes, read_bundle, write_arm

SCENES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenes")

# The refinement patch side, matching `RefinementWorkspace`'s 5x5.
PATCH = 5


def peak_index(surface):
    """Row-major first-strict-maximum, matching `cv2.minMaxLoc` and `AutoRIFT.peak_index`.

    `np.argmax` scans in the same C order for a contiguous array, so it agrees; written out because the
    tie-breaking convention is the thing being relied on and a plateau resolves differently under a
    column-major scan.
    """
    return np.unravel_index(int(np.argmax(surface)), surface.shape)


def refine_pyrup_at(surface, rx, ry, upsampling):
    """The `pyrUp` cascade, following `AutoRIFT.subpixel_peak` (`src/peak.jl:526`) step for step.

    A 5x5 patch about the integer peak, clamped to the surface, doubled by `cv2.pyrUp` until the
    upsampling factor is reached, then the peak of the upsampled patch mapped back. The patch origin is
    tracked separately because clamping means the peak is not generally at the patch centre.
    """
    nr, nc = surface.shape
    pi, pj = peak_index(surface)
    if nr < PATCH or nc < PATCH or upsampling == 1:
        return float(pj - rx), float(pi - ry), float(surface[pi, pj])

    half = PATCH // 2
    i0 = min(max(pi - half, 0), max(nr - PATCH, 0))
    j0 = min(max(pj - half, 0), max(nc - PATCH, 0))
    cur = np.ascontiguousarray(surface[i0:i0 + PATCH, j0:j0 + PATCH], dtype=np.float32)

    factor = 1
    while factor < upsampling:
        n = cur.shape[0] * 2
        cur = cv2.pyrUp(cur, dstsize=(n, n))
        factor *= 2

    si, sj = peak_index(cur)
    row = si / float(factor) + i0
    col = sj / float(factor) + j0
    return col - rx, row - ry, float(cur[si, sj])


def refine_parabola(surface, rx, ry):
    """A 3-point parabola fit in each axis about the integer peak.

    Falls back to the integer peak when the peak sits on the surface boundary, where there is no
    neighbour on one side, and when the denominator vanishes on a locally flat surface.
    """
    nr, nc = surface.shape
    pi, pj = peak_index(surface)
    val = float(surface[pi, pj])
    if not (0 < pi < nr - 1 and 0 < pj < nc - 1):
        return float(pj - rx), float(pi - ry), val

    def offset(a, b, c):
        d = a - 2.0 * b + c
        return 0.0 if d == 0.0 else 0.5 * (a - c) / d

    sy = offset(float(surface[pi - 1, pj]), val, float(surface[pi + 1, pj]))
    sx = offset(float(surface[pi, pj - 1]), val, float(surface[pi, pj + 1]))
    return (pj + sx) - rx, (pi + sy) - ry, val


def opencv_arms(ref, sec, arrays, scalars):
    """Both OpenCV arms in one pass over the grid, sharing each correlation surface.

    One pass because the surface is the expensive part and both estimators read the same one: running
    them separately would double the matching cost and could not change either answer.
    """
    chip = scalars["chip"]
    radius = scalars["radius"]
    upsampling = scalars["upsampling"]
    rows, cols = grid_axes(arrays)
    nr, nc = ref.shape
    h = chip // 2

    shape = (len(rows), len(cols))
    out = {k: np.full(shape, np.nan, dtype=np.float32)
           for k in ("pyrup_dx", "pyrup_dy", "pyrup_corr",
                     "para_dx", "para_dy", "para_corr")}

    for ii, cy in enumerate(rows):
        for jj, cx in enumerate(cols):
            # Same asymmetric window as `correlate!`: `radius` one way and `radius - 1` the other, so
            # the surface is an even 2*radius and zero displacement lands on a sample. Converted from
            # the Julia 1-based centre to 0-based slicing.
            r0 = cy - h - radius - 1
            c0 = cx - h - radius - 1
            r1 = r0 + chip + 2 * radius - 1
            c1 = c0 + chip + 2 * radius - 1
            if r0 < 0 or c0 < 0 or r1 > nr or c1 > nc:
                continue
            chipwin = np.ascontiguousarray(sec[cy - h - 1:cy + h - 1, cx - h - 1:cx + h - 1],
                                           dtype=np.float32)
            searchwin = np.ascontiguousarray(ref[r0:r1, c0:c1], dtype=np.float32)
            if chipwin.std() == 0.0:
                continue
            surface = cv2.matchTemplate(searchwin, chipwin, cv2.TM_CCOEFF_NORMED)

            dx, dy, v = refine_pyrup_at(surface, radius, radius, upsampling)
            out["pyrup_dx"][ii, jj] = dx
            out["pyrup_dy"][ii, jj] = dy
            out["pyrup_corr"][ii, jj] = v

            dx, dy, v = refine_parabola(surface, radius, radius)
            out["para_dx"][ii, jj] = dx
            out["para_dy"][ii, jj] = dy
            out["para_corr"][ii, jj] = v

    return out


def main():
    for name in cases(SCENES, sys.argv[1:]):
        path = os.path.join(SCENES, name)
        arrays, scalars = read_bundle(path)

        t0 = time.time()
        r = opencv_arms(arrays["reference"], arrays["secondary"], arrays, scalars)
        elapsed = time.time() - t0

        # The two arms share the matching cost, so each is charged half rather than the whole: neither
        # would have run the surface twice on its own.
        write_arm(path, "cv_pyrup", {"dx": r["pyrup_dx"], "dy": r["pyrup_dy"],
                                     "correlation": r["pyrup_corr"]}, elapsed / 2)
        write_arm(path, "cv_parabola", {"dx": r["para_dx"], "dy": r["para_dy"],
                                        "correlation": r["para_corr"]}, elapsed / 2)
        measured = int(np.isfinite(r["pyrup_dx"]).sum())
        print("%-28s %5.1f s  measured %d/%d"
              % (name, elapsed, measured, r["pyrup_dx"].size))


if __name__ == "__main__":
    main()
