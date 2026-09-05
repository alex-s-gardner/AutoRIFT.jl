"""Capture the arrays the reference correlator is actually handed, from inside the container.

Driven by `tools/golden/intermediate.jl`, which mounts this file and `tools/ab/xchg.py` into the
container and runs it in place of the ordinary entry point.

**Why intercept rather than reconstruct.** The correlator's inputs are not the scene files. By the
time `runAutorift` is called the pair has been read, high-pass filtered, nodata-filled and cast, and
the grid has been snapped to the even-chip convention. Rebuilding that chain in Julia to feed
AutoRIFT.jl the "same" arrays would mean reimplementing the code under test, so a preprocessing
difference would surface as a correlator difference. Taking the arrays at the call boundary makes the
comparison a comparison of the correlator alone — the same discipline `tools/ab` uses in handing the
reference arrays AutoRIFT.jl already filtered.

Arrays are written through `tools/ab/xchg.py`, so the element type and both dimensions travel in the
file and the Julia reader takes the layout from the header rather than from a caller-supplied shape.
These grids are square, and a square array read with the wrong convention is silently transposed.

**The argument-order trap.** `arImgDisp_s(a, b)` cuts its chip from `b` and its search window from
`a`: `a` binds to `I1`, but the body calls the C++ as `(I2.ravel(), I1.ravel())` and the C++ binds its
first array to `sec_img`, where the chip is taken. Two swaps that compose rather than cancel. Arrays
here are named for the attribute they came from (`I1`, `I2`), so a reader must apply that mapping
itself; `tools/ab/README.md` states it and `tools/ab/stage2_python.py` asserts it.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

import xchg

OUT = Path(os.environ.get('CAPTURE_DIR', '/home/ubuntu/work/capture'))

# Scalars that change the answer. Captured so the Julia side configures `Params` from what the
# reference actually used rather than from what the driver is believed to set — `OverSampleRatio`
# alone is a per-chip-size dictionary assembled at run time (`testautoRIFT.py:488-510`).
SCALARS = (
    'ChipSize0X', 'ChipSizeMinX', 'ChipSizeMaxX', 'ScaleChipSizeY',
    'GridSpacingX', 'SkipSampleX', 'SkipSampleY',
    'OverSampleRatio', 'WallisFilterWidth', 'MultiThread',
    'BuffDistanceC', 'CoarseCorCutoff', 'sparseSearchSampleRate',
    'DataTypeInput', 'ChipSizeMaxXInput', 'preproc_filt_width',
)

# Arrays to take before the call: the filtered pair, the grid, the priors, the per-point limits.
INPUTS = ('I1', 'I2', 'xGrid', 'yGrid', 'Dx0', 'Dy0',
          'SearchLimitX', 'SearchLimitY', 'ChipSizeMinX', 'ChipSizeMaxX', 'zeroMask')

# And after: the answer.
OUTPUTS = ('Dx', 'Dy', 'InterpMask', 'ChipSizeX')


def _write(name, a):
    xchg.write(str(OUT / name), a)
    return {'file': f'{name}.abx', 'dtype': a.dtype.str, 'shape': list(a.shape)}


def _scalar(v):
    """A JSON-representable form of an autoRIFT attribute, or None for an array."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, dict):
        return {str(k): _scalar(x) for k, x in v.items()}
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return None
    return str(v)


def _dump(obj, names, prefix, manifest):
    for name in names:
        v = getattr(obj, name, None)
        if not isinstance(v, np.ndarray) or v.ndim != 2:
            continue
        # `InterpMask` and `zeroMask` are boolean. NumPy stores those one byte per element, so a view
        # as uint8 is the same bytes with a name xchg's type table already has — no value changes.
        # Every other dtype outside the table is recorded as skipped rather than cast, since a cast
        # that loses precision would silently change what is being compared.
        if v.dtype == np.bool_:
            v = v.view(np.uint8)
        if v.dtype.str not in xchg.TAGS:
            manifest['skipped'][f'{prefix}{name}'] = v.dtype.str
            continue
        manifest['arrays'][f'{prefix}{name}'] = _write(f'{prefix}{name}', v)


def install():
    """Patch `autoRIFT.runAutorift` to dump its inputs and outputs around the real call."""
    OUT.mkdir(parents=True, exist_ok=True)
    from autoRIFT.autoRIFT import autoRIFT

    original = autoRIFT.runAutorift
    state = {'n': 0}

    def patched(self):
        state['n'] += 1
        call = state['n']
        manifest = {'call': call, 'arrays': {}, 'scalars': {}, 'skipped': {}}

        _dump(self, INPUTS, 'in_', manifest)
        original(self)
        _dump(self, OUTPUTS, 'out_', manifest)

        for name in SCALARS:
            if hasattr(self, name):
                s = _scalar(getattr(self, name))
                if s is not None:
                    manifest['scalars'][name] = s

        with open(OUT / f'call{call}.json', 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f'[capture] call {call}: {len(manifest["arrays"])} arrays -> {OUT}', flush=True)
        if manifest['skipped']:
            print(f'[capture] skipped (dtype not in xchg): {manifest["skipped"]}', flush=True)

    autoRIFT.runAutorift = patched
    print(f'[capture] runAutorift patched; output -> {OUT}', flush=True)


def main():
    install()
    # The rest of the command line goes to the ordinary entry point, so this runs the real pipeline
    # rather than a reconstruction of it.
    from hyp3_autorift.process import main as process_main

    sys.argv = ['hyp3_autorift'] + sys.argv[1:]
    process_main()


if __name__ == '__main__':
    main()
