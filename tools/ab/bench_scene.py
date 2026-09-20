"""The reference's row of the full-scene table: `runAutorift` over the whole Landsat overlap.

Two accountings, both recorded. `bench_table.jl` times this as a child process under
`/usr/bin/time -l`, which is whole-process wall clock and peak RSS; `phase.py` additionally brackets
`runAutorift` itself, which is the correlator's own cost from its entry point. The second is what a
comparison against AutoRIFT.jl is made on, because the reference's preprocessing is separate calls
while AutoRIFT.jl's is inside `autorift`, and a whole-process figure charges the two differently.
Printing `MEASURED <n>` at the end is the contract, matching `ROW_SCRIPT` on the Julia side, so the
caller can confirm the row did the work rather than exiting early.

The planes come from the work directory `bench_scene.jl` filled, which is what makes this comparable:
both sides read the same `Float32` arrays off the same files rather than each doing its own I/O and
preprocessing.

Preprocessing runs here, unlike the stage-2 A/B, because the whole-process figure measures the
pipeline as a user would invoke it. `DataType = 0` selects the reference's own uint8 path, which is
what its `preprocess_filt_hps` produces and what production runs use.

**The filtered planes are written out**, as `prep_ref.u8` and `prep_sec.u8` in the work directory,
column-major uint8. The Julia rows that compare against the correlator read those and skip their own
preprocessing, so both sides correlate the same bytes and a filter difference cannot appear in the
comparison as a speed or coverage difference.

    AUTORIFT_BENCH_DIR=/path/to/work micromamba run -n arift-ref python tools/ab/bench_scene.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("AUTORIFT_BENCH_DIR")
if not WORK:
    sys.exit("set AUTORIFT_BENCH_DIR to the directory bench_scene.jl wrote")

# Must match `bench_table.jl`'s constants; the table compares rows at one configuration, so a
# divergence here would compare different problems and report it as a speed difference.
CHIP, CHIP_MAX, SPACING, RADIUS, UPSAMPLING = 16, 64, 8, 20, 16


def planes():
    """The two scene planes and their shape, as `bench_scene.jl` wrote them: column-major `Float32`."""
    with open(os.path.join(WORK, "dims.txt")) as fh:
        nr, nc = (int(v) for v in fh.read().split())
    out = []
    for name in ("ref.bin", "sec.bin"):
        a = np.fromfile(os.path.join(WORK, name), dtype=np.float32)
        if a.size != nr * nc:
            sys.exit(f"{name}: expected {nr * nc} values, found {a.size}")
        # Column-major on disk, so read as Fortran order rather than transposing a C-order view --
        # a transpose here would correlate the scene sideways and still produce plausible-looking output.
        out.append(np.reshape(a, (nr, nc), order="F"))
    return out[0], out[1], nr, nc


def main():
    from autoRIFT import autoRIFT

    ref, sec, nr, nc = planes()

    obj = autoRIFT()
    # `I1` holds the *secondary*. `runAutorift` correlates with `arImgDisp_s(self.I2, self.I1)` and that
    # function cuts its chip from its second argument, so the chip comes from `I1`. Putting the secondary
    # there is what matches AutoRIFT.jl and every other row -- see this directory's README.
    obj.I1 = np.ascontiguousarray(sec)
    obj.I2 = np.ascontiguousarray(ref)
    obj.DataType = 0
    obj.zeroMask = None

    # A grid over the whole scene at the shared spacing. `runAutorift` truncates it to a whole multiple
    # of `ChipSizeMaxX / ChipSize0X` itself, so the count here is the request rather than the final one.
    #
    # `- 1.5` for the grid origin, as `stage2_python.py` derives: Julia's grid is 1-based pixel centres
    # against the reference's 0-based, and `runAutorift` adds a further half pixel that `arImgDisp_s`
    # does not.
    inset = CHIP_MAX // 2 + RADIUS + 2
    ys = np.arange(inset, nr - inset, SPACING, dtype=np.float32)
    xs = np.arange(inset, nc - inset, SPACING, dtype=np.float32)
    obj.xGrid, obj.yGrid = np.meshgrid(xs - 1.5, ys - 1.5)

    obj.ChipSize0X = CHIP
    obj.ChipSizeMinX = CHIP
    obj.ChipSizeMaxX = CHIP_MAX
    obj.GridSpacingX = SPACING
    obj.SkipSampleX = SPACING
    obj.SkipSampleY = SPACING
    obj.SearchLimitX = RADIUS
    obj.SearchLimitY = RADIUS
    obj.OverSampleRatio = UPSAMPLING
    obj.Dx0 = 0
    obj.Dy0 = 0
    # `mpflag = 0` is what production runs, and the reference's own note says threading cannot change
    # results: each grid point writes a distinct output element.
    obj.MultiThread = 0

    obj.preprocess_filt_hps()
    obj.uniform_data_type()

    # After `uniform_data_type` these are the uint8 arrays the correlator reads, so this is the
    # correlator's input boundary and the bytes the Julia comparison rows are handed. Written
    # column-major to match `ref.bin`/`sec.bin` and what `read!` on a Julia `Matrix` expects.
    #
    # A column at a time, because `ndarray.tofile` always writes C order: transposing or asking for a
    # Fortran copy first would materialize a second 290 MB plane, and the peak this script reports
    # must not include a temporary that exists only to write a file.
    for name, plane in (("prep_ref.u8", obj.I2), ("prep_sec.u8", obj.I1)):
        if plane.dtype != np.uint8:
            sys.exit(f"{name}: expected uint8 after uniform_data_type, found {plane.dtype}")
        with open(os.path.join(WORK, name), "wb") as fh:
            for j in range(plane.shape[1]):
                fh.write(np.ascontiguousarray(plane[:, j]).tobytes())

    _, metrics = phase.measure(obj.runAutorift)
    phase.report("autorift", metrics)

    print("MEASURED %d" % int(np.count_nonzero(~np.isnan(obj.Dx))))


if __name__ == "__main__":
    main()
