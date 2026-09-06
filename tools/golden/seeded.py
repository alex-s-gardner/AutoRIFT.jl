"""Run the reference with the gap-fill RNG seeded, to test what it actually changes.

    docker run ... -e PYTHONPATH=/opt/capture python /opt/capture/seeded.py --reference ... --secondary ...

`_wallis_filter_fill` fills Landsat 7's Scan Line Corrector gaps with `rng.normal` from an unseeded
`np.random.default_rng()` (`autoRIFT.py:123`). Two runs of the reference on one L7 pair disagree on
96.6% of `vx`, and this decides whether that draw is the cause: patching `default_rng` to return a
seeded generator makes the fill deterministic and changes nothing else, so if two seeded runs agree
the RNG is responsible, and if they still disagree it is not.

The patch is deliberately narrow — `numpy.random.default_rng` only, not the legacy global — because
`autoRIFT.py:123` is the sole `default_rng` call in the module and `netcdf_output.py::v_error_cal` is
the only other one in the chain. `v_error` is already known to survive `int16` rounding, so seeding it
too costs nothing and keeps the patch a one-liner.
"""

import os
import sys

import numpy as np

SEED = int(os.environ.get('AUTORIFT_SEED', '20260905'))


def install():
    original = np.random.default_rng

    def seeded(seed=None):
        # A caller that passes its own seed keeps it; only the unseeded calls become deterministic.
        return original(SEED if seed is None else seed)

    np.random.default_rng = seeded
    print(f'[seeded] numpy.random.default_rng seeded with {SEED}', flush=True)


def main():
    install()
    from hyp3_autorift.process import main as process_main

    sys.argv = ['hyp3_autorift'] + sys.argv[1:]
    process_main()


if __name__ == '__main__':
    main()
