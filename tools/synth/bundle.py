"""Reading and writing the per-case bundles from Python.

The Julia side of this harness writes the format; `bundle.jl` documents it. This is the mirror, so the
Python and OpenCV arms read the same arrays the Julia arms did rather than regenerating anything.
"""

import os

import numpy as np

DTYPES = {"Float32": np.float32, "Float64": np.float64, "Int32": np.int32}
NUMPY_NAMES = {np.dtype(np.float32): "Float32", np.dtype(np.float64): "Float64",
               np.dtype(np.int32): "Int32"}


def read_bundle(path):
    """The arrays and scalars a case bundle holds.

    Arrays are stored column-major, so each is read flat and reshaped against the reversed shape then
    transposed -- the same un-transposing `tools/ab/stage2_python.py` does.
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


def read_case(path):
    """The case's factor settings as strings, parsed by whichever caller needs them."""
    out = {}
    with open(os.path.join(path, "case.txt")) as fh:
        for line in fh:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                out[parts[0]] = parts[1]
    return out


def grid_axes(arrays):
    """The 1-based pixel rows and columns the grid sits on, read back from the grid arrays.

    Recovered rather than recomputed from the margin formula, so an arm cannot silently correlate a
    different grid than the truth was sampled on.
    """
    rows = np.rint(arrays["grid_y"][:, 0]).astype(int)
    cols = np.rint(arrays["grid_x"][0, :]).astype(int)
    return rows, cols


def write_arm(path, name, fields, seconds):
    """One arm's output, in the layout `read_arm` in `bundle.jl` expects.

    Written column-major so the Julia scorer needs no transpose of its own.
    """
    with open(os.path.join(path, name + ".txt"), "w") as fh:
        fh.write("# name dtype shape\n")
        for key, arr in fields.items():
            arr = np.ascontiguousarray(arr)
            fname = "{}_{}".format(name, key)
            np.asfortranarray(arr).T.tofile(os.path.join(path, fname + ".bin"))
            fh.write("{} {} {}\n".format(fname, NUMPY_NAMES[arr.dtype],
                                         "x".join(str(v) for v in arr.shape)))
        fh.write("seconds {}\n".format(seconds))


def cases(root, argv):
    """Case directories to process: every one, or those named on the command line."""
    names = sorted(d for d in os.listdir(root)
                   if os.path.isfile(os.path.join(root, d, "manifest.txt")))
    wanted = [n for n in names if n in argv] if argv else names
    if not wanted:
        raise SystemExit("no case matched {} in {}".format(argv, root))
    return wanted
