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


def install_levels(module, manifest):
    """Patch the per-level correlator and filter calls to dump what each pyramid level decided.

    `runAutorift`'s own inputs and outputs describe the *merged* answer, so which level a
    disagreement came from has to be inferred from the reported `ChipSizeX`. That inference is
    weakest exactly where it matters — a point the two implementations assign to different levels
    is the case under investigation, and it is the case where the merged output cannot say which
    level's decision diverged.

    The pyramid's per-level state is local to `runAutorift`, so it is unreachable from outside. What
    *is* reachable is the calls that loop makes: `arImgDisp_u`/`arImgDisp_s` once for the coarse pass
    and once for the fine pass of each level, and `DISP_FILT.filtDisp` once per pass. Wrapping those
    records each level's raw measurement and each level's rejection mask without reimplementing the
    loop, so the recorded values are the reference's own rather than a reconstruction.

    Calls are numbered in the order they happen. `ChipSizeX` is an argument to the correlator, so the
    level is recorded rather than deduced, and the coarse and fine passes are distinguished by
    `SubPixFlag` — the coarse pass runs with it `False`.
    """
    seq = {'n': 0}
    levels = manifest.setdefault('levels', [])

    def wrap_corr(name):
        original = getattr(module, name, None)
        if original is None:
            return

        def patched(I1, I2, xGrid, yGrid, ChipSizeX, ChipSizeY, SearchLimitX, SearchLimitY,
                    Dx0, Dy0, SubPixFlag, overSampleRatio, *rest):
            seq['n'] += 1
            n = seq['n']
            dx, dy = original(I1, I2, xGrid, yGrid, ChipSizeX, ChipSizeY, SearchLimitX,
                              SearchLimitY, Dx0, Dy0, SubPixFlag, overSampleRatio, *rest)
            rec = {
                'seq': n,
                'kind': 'fine' if SubPixFlag else 'coarse',
                'chip_size_x': float(ChipSizeX),
                'chip_size_y': float(ChipSizeY),
                'oversample': float(overSampleRatio),
                'grid_shape': list(np.shape(xGrid)),
                'measured': int(np.count_nonzero(~np.isnan(dx))),
                'arrays': {},
            }
            # The grid too: a level's coarse pass runs on a decimated grid, and the decimation is
            # what places a coarse estimate over the fine points it stands for.
            for label, arr in (('dx', dx), ('dy', dy), ('xgrid', xGrid), ('ygrid', yGrid),
                               ('searchx', SearchLimitX), ('searchy', SearchLimitY)):
                a = np.asarray(arr)
                if a.ndim != 2 or a.dtype.str not in xchg.TAGS:
                    continue
                rec['arrays'][label] = _write(f'lvl{n}_{label}', a)
            levels.append(rec)
            print(f'[capture] level call {n}: {rec["kind"]} chip {ChipSizeX} '
                  f'grid {rec["grid_shape"]} measured {rec["measured"]}', flush=True)
            return dx, dy

        setattr(module, name, patched)

    wrap_corr('arImgDisp_u')
    wrap_corr('arImgDisp_s')

    # The rejection mask each pass keeps. Paired with the raw `dx` above, this separates "this level
    # never measured the point" from "this level measured it and threw it away", which the merged
    # `ChipSizeX` cannot distinguish.
    disp_filt = getattr(module, 'DISP_FILT', None)
    if disp_filt is not None:
        original_filt = disp_filt.filtDisp

        def patched_filt(self, Dx, Dy, SearchLimitX, SearchLimitY, M, OverSampleRatio):
            seq['n'] += 1
            n = seq['n']
            kept = original_filt(self, Dx, Dy, SearchLimitX, SearchLimitY, M, OverSampleRatio)
            levels.append({
                'seq': n,
                'kind': 'filtDisp',
                'filt_width': int(self.FiltWidth),
                'frac_valid': float(self.FracValid),
                'iterations': int(self.Iter),
                'oversample': float(OverSampleRatio),
                'grid_shape': list(np.shape(Dx)),
                'in_mask': int(np.count_nonzero(M)),
                'kept': int(np.count_nonzero(kept)),
                'arrays': {'kept': _write(f'lvl{n}_kept', np.asarray(kept).view(np.uint8))},
            })
            print(f'[capture] level call {n}: filtDisp width {self.FiltWidth} '
                  f'frac {self.FracValid:.4f} kept {np.count_nonzero(kept)} '
                  f'of {np.count_nonzero(M)}', flush=True)
            return kept

        disp_filt.filtDisp = patched_filt


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

        # Per-level patches go on before the call, since they record what happens inside it. They
        # write into this call's manifest, so a driver that runs the correlator more than once keeps
        # the levels attributed to the right run.
        from autoRIFT import autoRIFT as ar_module

        install_levels(ar_module, manifest)

        original(self)
        # Inputs are taken *after* the call, not before. `runAutorift` rewrites them as its first
        # action — `self.xGrid = np.round(self.xGrid[0:rlim, 0:clim]) + 0.5` and the same for `yGrid`,
        # then truncates `Dx0`, `Dy0`, `SearchLimit*` and the chip bounds to that same window
        # (`autoRIFT.py:883-905`) — and it is the rewritten arrays the correlator sees. Dumping them
        # first captures an integer grid missing the half pixel, which puts every search centre half a
        # pixel from where the reference put it: a residual that is zero under uniform motion and
        # grows with the velocity gradient, so it hides in the median and shows up only as a
        # gradient-correlated difference map.
        #
        # `self.I1`/`self.I2` are also rewritten, by the uniform-data-type conversion, so taking them
        # after is right for the same reason.
        _dump(self, INPUTS, 'in_', manifest)
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
