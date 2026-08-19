# Python reference tooling

Two scripts, one conda environment. Neither is needed to run the test suite —
fixtures are committed — only to regenerate them or to refresh the timing
baseline.

```bash
mamba env create -f tools/python_ref/environment-cv.yml

mamba run -n autorift-cv python tools/python_ref/gen_fixtures.py   # -> test/fixtures/
mamba run -n autorift-cv python tools/python_ref/bench_python.py   # -> benchmark/results/python.json
```

`gen_fixtures.py` accepts group names to regenerate a subset, e.g.
`gen_fixtures.py matchtemplate peak`. Each group directory is deleted and rebuilt,
so a renamed case cannot leave a stale directory behind that still passes.

## Why a modern stack, not a pinned legacy one

autoRIFT v2.1.2 requires Python >= 3.10 and runs on NumPy >= 2.0. Earlier releases
did not — v1.5.0 used `np.bool`, `np.int`, and `np.asscalar`, all removed in NumPy
1.24, which forced a pinned environment. That constraint is gone.

OpenCV *is* pinned, because a few of its behaviours are version-dependent in ways
that would silently invalidate fixtures: `minAreaRect`'s angle convention changed
between 3.x and 4.5, `INTER_NEAREST` differs from `INTER_NEAREST_EXACT` by half a
pixel, and `filter2D` switches to a DFT path once the kernel area passes a
threshold that has itself moved between releases. The version used is recorded in
`test/fixtures/manifest.json`, so a fixture regenerated under a different OpenCV
is identifiable as such rather than debugged as a code change.

## What these fixtures are and are not

They pin **OpenCV's semantics for the primitives where OpenCV is the de-facto
standard and is correct**: border handling, resampling, the correlation surface,
pyramid upsampling, peak tie-breaking, the distance transform.

They are **not** a golden copy of Python autoRIFT's output. That distinction
matters, because AutoRIFT.jl deliberately does not reproduce several defects in
the reference (see `../../REFERENCE.md`). Correctness of the whole pipeline is
established against synthetic ground truth — a texture displaced by a known amount
has an exactly known answer — and agreement with production is assessed
statistically, with a tracked divergence budget.

Neither nasa-jpl/autoRIFT nor ASFHyP3/hyp3-autorift has a test that exercises the
correlator, so there is no upstream golden data to compare against even in
principle.

## Timing baseline

`bench_python.py` times the OpenCV primitives AutoRIFT.jl replaces and writes
`benchmark/results/python.json` in the same schema the Julia suite emits, so
`benchmark/compare.jl` prints a per-kernel speedup alongside every Julia
measurement. `cv2.setNumThreads(1)` is set so both sides describe one core doing
one kernel; thread scaling is measured separately per implementation.

It does not time the full pipeline — that needs the reference installed with its
ISCE3/GDAL stack and a real image pair, so the end-to-end baseline is captured
from the production container instead. The primitives are where the port's
performance is won or lost.
