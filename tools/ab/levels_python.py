"""The reference's per-level intermediates, captured from a real run of the stage-2 comparison.

The stage-2 bundle compares only the *merged* output, which is where a disagreement is visible and not
where it is caused. A coarse chip-size level correlates a decimated grid, filters it, fills its holes,
and interpolates the result back — and a difference introduced at any of those steps arrives at the
merged field looking the same. Comparing per level is what separates them.

Nothing here reimplements the reference. It runs `stage2_python.py` unchanged and patches `cv2.resize`,
which is what the level loop calls at each step, so the arrays recorded are the ones the reference
actually used:

    DxF  before the upsample  the level's own values, holes filled  (`autoRIFT.py:855-856`)
    DxF  after it             the same values on the full grid
    DxF0                      the finer levels' answer reduced onto this level  (`823-839`)
    M0, MF                    the masks that decide which points the level may claim (`858-859`)

Written through `xchg`, so the Julia side reads them without a reshape convention to get wrong — see
`xchg.py` for why that matters. `levels_diff.jl` is the consumer.

    micromamba run -n arift-ref python tools/ab/levels_python.py
"""

import os
import runpy
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xchg  # noqa: E402  -- needs the path set above

HERE = os.path.dirname(os.path.abspath(__file__))
# Counters per kind, so each capture is named by the order the loop produced it: the coarse levels run
# finest-first, and `cubic` arrives in (dx, dy) pairs.
N = {"cubic": 0, "area": 0, "mask": 0}
_resize = cv2.resize


def spy(src, dsize, **kw):
    out = _resize(src, dsize, **kw)
    interp = kw.get("interpolation")
    up = src.shape[0] < dsize[1]
    if interp == cv2.INTER_CUBIC and up:
        # A level's upsample: this is `DxF` (then `DyF`) going from the level's grid to the full one.
        k = N["cubic"]
        N["cubic"] += 1
        xchg.write(f"lvl_pre_{k}", src)
        xchg.write(f"lvl_post_{k}", out)
    elif interp == cv2.INTER_AREA and not up:
        # The prior reduced onto a level: `DxF0`/`DyF0`, plus the coordinate grids, which are told apart
        # downstream by their range rather than by index.
        k = N["area"]
        N["area"] += 1
        xchg.write(f"lvl_area_{k}", out)
    elif interp == cv2.INTER_NEAREST and up and src.dtype == np.uint8:
        # `MF` then `M0`, the masks gating what the level claims.
        k = N["mask"]
        N["mask"] += 1
        xchg.write(f"lvl_mask_{k}", src)
    return out


def main():
    cv2.resize = spy
    sys.argv = ["stage2_python.py"]
    try:
        runpy.run_path(os.path.join(HERE, "stage2_python.py"), run_name="__main__")
    except SystemExit:
        pass
    finally:
        cv2.resize = _resize
    print("captured", N, "->", xchg.DIR, flush=True)


if __name__ == "__main__":
    main()
