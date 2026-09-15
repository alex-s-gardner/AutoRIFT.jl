# NISAR block-size sweep logs

Console output of `tools/golden/profile_nisar.jl`, one file per invocation. Committed because the
serialized histories live outside the repository (`$AUTORIFT_GOLDEN_CACHE/mem/`) and a sweep costs about
an hour of machine time per case, so the record has to survive a clean checkout.

Each log carries what the serialized row does not: the per-stack profile attribution, the layout figures
printed before each run, and the progress of the run itself. `docs/memory.md` quotes these; a figure there
should be traceable to a line here.

| log | configurations | note |
|---|---|---|
| `sweep_l2b.log` | untiled, 8192, 6144, 4096, 3072 px | one process; **deadlocked** during the 3072 px profile, so that row is absent |
| `sweep_l2c.log` | 3072 px | re-measured alone, which is how the 3072 row was obtained |
| `sweep_l2small.log` | 2304, 2816, 2304x1152, 3072x1536 px | **deadlocked** during the 2816 px profile; only the 2304 px row completed |
| `sweep_aniso1.log` | 2304x1152 px | alone in its own process — the best configuration measured |

Two of the four runs deadlocked, both while profiling a configuration that was not the first in its
process. `tools/golden/GATES.md` records the thread traces and the reproducer. **A run that stops writing
here has not necessarily died** — check for a 0%-CPU survivor with `ps` before assuming it did.

Recover a log's rows into the append-only history with:

```bash
julia --project=tools/golden tools/golden/backfill_history.jl NISAR_L2_PR_GSLC \
    benchmark/results/nisar/sweep_l2b.log
```
