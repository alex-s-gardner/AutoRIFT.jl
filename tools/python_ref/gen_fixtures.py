#!/usr/bin/env python
"""Generate OpenCV reference fixtures for the AutoRIFT.jl test suite.

Each case writes a directory under ``test/fixtures/`` holding one flat ``.bin``
per array plus a ``meta.json`` describing shapes, dtypes, and the parameters the
case was generated with. A flat binary format is used rather than ``.npz`` so
that the Julia side needs no NumPy-format reader.

Fixtures are committed, so the test suite runs on a bare clone with no Python.
Regenerate with::

    mamba env create -f tools/python_ref/environment-cv.yml
    mamba run -n autorift-cv python tools/python_ref/gen_fixtures.py

What is pinned here, and why: OpenCV is the comparison target for the primitives
where its semantics are the de-facto standard and are correct — border handling,
resampling, the correlation surface, pyramid upsampling, peak tie-breaking. It is
*not* the target for the parts of the Python autoRIFT that are defective; those
are validated against synthetic ground truth instead. So these fixtures pin the
math, not the reference implementation's behaviour.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = ROOT / "test" / "fixtures"

# Border modes exercised for every filter. The names are the Julia-side spelling;
# the mapping to OpenCV is the point of the test, because it is inverted from
# intuition: OpenCV's BORDER_REFLECT duplicates the edge pixel and corresponds to
# Julia's `:symmetric`, while BORDER_REFLECT_101 does not and corresponds to
# Julia's `:reflect`. Getting this backwards is the single most common porting
# error in image code, so all four are pinned.
BORDERS = {
    "constant": cv2.BORDER_CONSTANT,
    "replicate": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_REFLECT,        # fedcba|abcdef -> Julia :symmetric
    "reflect101": cv2.BORDER_REFLECT_101,  # gfedcb|abcdef -> Julia :reflect
}

INTERPOLATIONS = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "area": cv2.INTER_AREA,
    "cubic": cv2.INTER_CUBIC,
}

SIMILARITIES = {
    "zncc": cv2.TM_CCOEFF_NORMED,
    "ncc": cv2.TM_CCORR_NORMED,
}

DTYPE_NAMES = {
    np.dtype("uint8"): "uint8",
    np.dtype("int8"): "int8",
    np.dtype("uint16"): "uint16",
    np.dtype("int16"): "int16",
    np.dtype("int32"): "int32",
    np.dtype("int64"): "int64",
    np.dtype("float32"): "float32",
    np.dtype("float64"): "float64",
    np.dtype("complex64"): "complex64",
    np.dtype("complex128"): "complex128",
    np.dtype("bool"): "bool",
}


def write_case(name: str, arrays: dict[str, np.ndarray], params: dict) -> None:
    """Write one fixture case, C-order, with a manifest of shapes and dtypes."""
    case_dir = FIXTURE_DIR / name
    case_dir.mkdir(parents=True, exist_ok=True)
    meta = {"arrays": {}, "params": params}
    for key, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        (case_dir / f"{key}.bin").write_bytes(arr.tobytes(order="C"))
        meta["arrays"][key] = {
            "dtype": DTYPE_NAMES[arr.dtype],
            "shape": list(arr.shape),
        }
    (case_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))


def texture(shape, seed=0, dtype=np.float32):
    """Band-limited random texture.

    White noise correlates only at exactly zero lag, so it produces a
    one-pixel-wide correlation peak and tests nothing about surface shape or
    subpixel behaviour. Smoothing gives a finite-width peak, which is both more
    realistic and a stricter test. Matches ``synthetic_texture`` in
    ``test/utils.jl`` in intent, though not bit-for-bit — these fixtures pin
    OpenCV's output for a *given* input, and the input is stored alongside it.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(shape)
    for _ in range(3):
        a = cv2.blur(a, (3, 3), borderType=cv2.BORDER_REFLECT_101)
    a -= a.min()
    if a.max() > 0:
        a /= a.max()
    if np.issubdtype(dtype, np.integer):
        return np.round(a * np.iinfo(dtype).max).astype(dtype)
    return a.astype(dtype)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def gen_filter2d() -> int:
    """Box and high-pass correlation at every border mode and several widths.

    Widths 11 and 13 exist specifically to bracket OpenCV's switch to a
    DFT-based path (kernel area >= 130 on SSE2-era builds), which changes
    rounding. If that threshold moves, these two cases diverge and the suite
    says so rather than drifting silently.
    """
    n = 0
    src32 = texture((64, 64), seed=1, dtype=np.float32)
    src8 = texture((64, 64), seed=1, dtype=np.uint8)

    for width in (3, 5, 7, 9, 11, 13, 15, 21):
        box = np.full((width, width), 1.0 / (width * width), dtype=np.float32)

        # Identity minus box mean: the `Highpass` preprocessing kernel.
        hp = -np.full((width, width), 1.0 / (width * width), dtype=np.float32)
        hp[width // 2, width // 2] += 1.0

        for kname, kernel in (("box", box), ("highpass", hp)):
            for bname, border in BORDERS.items():
                for dname, src in (("float32", src32), ("uint8", src8)):
                    out = cv2.filter2D(src, -1, kernel, borderType=border)
                    write_case(
                        f"filter2d/{kname}_w{width}_{bname}_{dname}",
                        {"src": src, "kernel": kernel, "expected": out},
                        {
                            "kernel": kname,
                            "width": width,
                            "border": bname,
                            "dtype": dname,
                            # filter2D correlates; it does NOT flip the kernel.
                            # Both these kernels are symmetric so it makes no
                            # difference here, but the Sobel case below is not.
                            "correlation_not_convolution": True,
                        },
                    )
                    n += 1
    return n


def gen_derivkernels() -> int:
    """OpenCV's separable derivative kernels, unnormalized.

    Pinned as data rather than derived, because for widths above 3 these are
    binomial-difference kernels whose scale grows with width, and Julia's
    ``Kernel.sobel`` uses a different (normalized, 3x3-only) convention. Deriving
    them independently would be a second chance to get the scale wrong.
    """
    n = 0
    for width in (1, 3, 5, 7):
        for dx, dy in ((1, 0), (0, 1), (2, 0), (0, 2)):
            if width == 1 and (dx > 1 or dy > 1):
                continue
            kx, ky = cv2.getDerivKernels(dx, dy, width, normalize=False, ktype=cv2.CV_32F)
            write_case(
                f"derivkernels/d{dx}{dy}_w{width}",
                {"kx": kx.astype(np.float32), "ky": ky.astype(np.float32)},
                {"dx": dx, "dy": dy, "width": width, "normalize": False},
            )
            n += 1
    return n


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def gen_resize() -> int:
    """All four interpolation modes, at integer and non-integer scale factors.

    Non-integer ratios matter: ``INTER_AREA`` is an exact area average only when
    the ratio is integral, and something else otherwise. ``INTER_NEAREST`` uses
    ``floor(dst * scale)`` rather than the half-pixel-centre convention every
    other mode uses, which puts it half a pixel off from ``INTER_NEAREST_EXACT``.
    Both quirks are load-bearing in the pyramid and so are pinned.
    """
    n = 0
    for size in (64, 63):  # even and odd, since the centre convention differs
        src = texture((size, size), seed=2, dtype=np.float32)
        for iname, interp in INTERPOLATIONS.items():
            for scale in (0.5, 0.25, 0.125, 2.0, 4.0, 3 / 7):
                dst_h = max(1, int(size * scale))
                dst_w = max(1, int(size * scale))
                out = cv2.resize(src, (dst_w, dst_h), interpolation=interp)
                write_case(
                    f"resize/{iname}_{size}_{str(scale).replace('.', 'p').replace('/', 'o')}",
                    {"src": src, "expected": out},
                    {
                        "interpolation": iname,
                        "src_size": size,
                        "dst_shape": [dst_h, dst_w],
                        "scale": scale,
                    },
                )
                n += 1
    return n


def gen_pyrup() -> int:
    """Gaussian-pyramid upsampling, single step and cascaded.

    The reference locates the subpixel peak by cascading ``pyrUp`` on a 5x5
    neighbourhood until the accumulated factor reaches the upsampling ratio, then
    taking the argmax. That is a specific operation — inject zeros, convolve with
    the separable [1,4,6,4,1]/16 kernel, scale by 4, reflect-101 at the border —
    and not equivalent to bilinear or bicubic upsampling. The composite kernel of
    a cascade is also not a single Gaussian, so the cascade is pinned at every
    depth rather than just one step.

    Patch content is chosen to expose the failure modes: a single delta shows the
    kernel's shape directly, a plateau exposes argmax tie-breaking, and an
    all-equal patch is the degenerate case.
    """
    n = 0
    patches = {
        "random": texture((5, 5), seed=3, dtype=np.float32),
        "delta": np.eye(5, dtype=np.float32) * 0.0,
        "plateau": np.zeros((5, 5), dtype=np.float32),
        "monotone": np.arange(25, dtype=np.float32).reshape(5, 5) / 25.0,
        "equal": np.full((5, 5), 0.5, dtype=np.float32),
    }
    patches["delta"][2, 2] = 1.0
    patches["plateau"][1:4, 1:4] = 1.0

    # Single steps, each stored against its immediate input. Verifying one step
    # against its own input proves the whole cascade by induction, so there is no
    # need to store the deep levels: a 128x cascade of a 5x5 patch reaches
    # 640x640, and those arrays alone would be most of the committed fixture
    # bytes while adding no coverage a shallow step does not already give.
    #
    # Capped at 32x (160x160). One step at that size exercises the same code as
    # one step at 640x640; what differs between depths is the *composition*, which
    # the cascade cases below cover.
    for pname, patch in patches.items():
        cur = patch
        factor = 1
        while factor < 32:
            prev = cur
            cur = cv2.pyrUp(cur)
            factor *= 2
            write_case(
                f"pyrup/{pname}_step_x{factor}",
                {"src": prev, "expected": cur},
                {"patch": pname, "factor": factor, "single_step": True},
            )
            n += 1

    # Cascades from the original patch. Two steps are enough to catch a
    # composition error, since the composite kernel of two Gaussian-pyramid steps
    # is not itself a single Gaussian step; 8x gives margin without bulk.
    for pname, patch in patches.items():
        cur = patch
        for factor in (2, 4, 8):
            cur = cv2.pyrUp(cur)
            write_case(
                f"pyrup/{pname}_cascade_x{factor}",
                {"src": patch, "expected": cur},
                {"patch": pname, "factor": factor, "steps": int(np.log2(factor)),
                 "single_step": False},
            )
            n += 1
    return n


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


def gen_matchtemplate() -> int:
    """Full correlation surfaces for both similarity measures.

    The *whole surface* is stored, not just the peak, so the surface and the peak
    location are tested independently — a bug in one would otherwise be masked by
    the other.

    The ``dc_offset`` case is the important one: it adds a constant to the search
    window, which leaves ZNCC unchanged but shifts NCC's peak. That asymmetry is
    why the reference needs its chip-minimum subtraction hack for float input, and
    why ZNCC is the default here.
    """
    n = 0
    for chip in (32, 64, 128):
        for radius in (6, 10, 25):
            win = chip + 2 * radius - 1
            for dname, dtype in (("float32", np.float32), ("uint8", np.uint8)):
                base = texture((win, win), seed=chip + radius, dtype=dtype)
                tmpl = np.ascontiguousarray(
                    base[radius:radius + chip, radius:radius + chip]
                )

                variants = {"plain": base}
                if dtype == np.float32:
                    # Low contrast stresses the normalising denominator.
                    variants["lowcontrast"] = (base * 0.001 + 0.5).astype(np.float32)
                    # A DC offset separates ZNCC from NCC; see above.
                    variants["dc_offset"] = (base + 10.0).astype(np.float32)

                for vname, src in variants.items():
                    for sname, method in SIMILARITIES.items():
                        surface = cv2.matchTemplate(src, tmpl, method)
                        write_case(
                            f"matchtemplate/{sname}_c{chip}_r{radius}_{dname}_{vname}",
                            {"search": src, "chip": tmpl, "expected": surface},
                            {
                                "similarity": sname,
                                "chip_size": chip,
                                "search_radius": radius,
                                "dtype": dname,
                                "variant": vname,
                                # Follows from the window and chip extents; the
                                # peak-offset arithmetic depends on it.
                                "surface_shape": list(surface.shape),
                            },
                        )
                        n += 1
    return n


def gen_peak() -> int:
    """Hand-built surfaces that pin argmax tie-breaking.

    OpenCV's ``minMaxLoc`` scans row-major and keeps the first strict maximum;
    Julia's ``argmax`` scans column-major. On a plateau — common once imagery is
    quantized to 8 bits — the two disagree, and the resulting bias is invisible
    because both answers look reasonable. So the convention is pinned on surfaces
    built to have ties.
    """
    n = 0
    cases = {}

    a = np.zeros((7, 7), dtype=np.float32)
    a[2, 5] = 1.0
    cases["unique"] = a

    a = np.zeros((7, 7), dtype=np.float32)
    a[3, 1] = a[3, 5] = 1.0  # tie along a row: the leftmost must win
    cases["tie_in_row"] = a

    a = np.zeros((7, 7), dtype=np.float32)
    a[1, 3] = a[5, 3] = 1.0  # tie down a column: the topmost must win
    cases["tie_in_column"] = a

    a = np.zeros((7, 7), dtype=np.float32)
    a[2:5, 2:5] = 1.0  # a plateau: the top-left corner must win
    cases["plateau"] = a

    cases["all_equal"] = np.full((7, 7), 0.25, dtype=np.float32)

    for name, surface in cases.items():
        _, maxval, _, maxloc = cv2.minMaxLoc(surface)
        write_case(
            f"peak/{name}",
            {"surface": surface},
            {
                # OpenCV reports (x, y) = (column, row), 0-based. Recorded in
                # both conventions so the Julia side cannot quietly transpose.
                "maxloc_xy": [int(maxloc[0]), int(maxloc[1])],
                "row_col_0based": [int(maxloc[1]), int(maxloc[0])],
                "maxval": float(maxval),
            },
        )
        n += 1
    return n


# ---------------------------------------------------------------------------
# Morphology
# ---------------------------------------------------------------------------


def gen_disttransform() -> int:
    """Exact Euclidean distance transform.

    Convention: distance from each *non-zero* pixel to the nearest *zero* pixel.
    ``DIST_MASK_PRECISE`` gives the true EDT, matching
    ``scipy.ndimage.distance_transform_edt``; the 3x3 and 5x5 masks are chamfer
    approximations and would not.

    The ``stripes`` case reproduces the Landsat-7 scan-line-corrector gap
    pattern, which is what the transform is actually used on.
    """
    n = 0
    rng = np.random.default_rng(4)
    cases = {
        "random": (rng.random((64, 64)) > 0.3).astype(np.uint8),
        "single": np.ones((32, 32), dtype=np.uint8),
        "stripes": np.ones((64, 64), dtype=np.uint8),
        "full": np.ones((16, 16), dtype=np.uint8),
    }
    cases["single"][16, 16] = 0
    cases["stripes"][:, ::8] = 0

    for name, src in cases.items():
        if name == "full":
            # No zero pixel anywhere: OpenCV's output is unspecified, so record
            # only the input and let the Julia side define the behaviour.
            write_case(f"disttransform/{name}", {"src": src}, {"degenerate": True})
            n += 1
            continue
        out = cv2.distanceTransform(src, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        write_case(
            f"disttransform/{name}",
            {"src": src, "expected": out},
            {"metric": "L2", "mask": "precise",
             "convention": "distance from nonzero to nearest zero"},
        )
        n += 1
    return n


# ---------------------------------------------------------------------------


GENERATORS = (
    ("filter2d", gen_filter2d),
    ("derivkernels", gen_derivkernels),
    ("resize", gen_resize),
    ("pyrup", gen_pyrup),
    ("matchtemplate", gen_matchtemplate),
    ("peak", gen_peak),
    ("disttransform", gen_disttransform),
)


def main() -> None:
    import sys

    only = set(sys.argv[1:])
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    written = {}
    for name, fn in GENERATORS:
        if only and name not in only:
            continue
        # Regenerate from scratch so a renamed case cannot leave a stale
        # directory that still passes.
        group_dir = FIXTURE_DIR / name
        if group_dir.exists():
            shutil.rmtree(group_dir)
        count = fn()
        written[name] = count
        total += count
        print(f"  {name}: {count} cases")

    manifest = {
        "generated_by": "tools/python_ref/gen_fixtures.py",
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "groups": written,
        "total_cases": total,
    }
    (FIXTURE_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    print(f"\n{total} cases written to {FIXTURE_DIR}")
    print(f"opencv {cv2.__version__}, numpy {np.__version__}")


if __name__ == "__main__":
    main()
