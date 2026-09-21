# NISAR block-size sweep logs

Console output of `tools/golden/profile_nisar.jl`, one file per invocation. Committed because the
serialized histories live outside the repository (`$AUTORIFT_GOLDEN_CACHE/mem/`) and a sweep costs about
an hour of machine time per case, so the record has to survive a clean checkout.

Each log carries what the serialized row does not: the per-stack profile attribution, the layout figures
printed before each run, and the progress of the run itself. `docs/src/explanation/memory.md` quotes these; a figure there
should be traceable to a line here.

### L1 RSLC (re-measured on an idle machine)

| log | configuration | runtime | peak | occupancy |
|---|---|---:|---:|---:|
| `l1_0.log` | untiled | 567.2 s | 49.36 GiB | 9.21/10 |
| `l1_16384.log` | 16384 px | 1852.3 s | 59.58 GiB | 2.33/10 |
| `l1_8192.log` | 8192 px | 1078.4 s | 33.85 GiB | 4.61/10 |
| `l1_8192x4096.log` | 8192x4096 px | 828.0 s | 30.41 GiB | 6.60/10 |
| `l1_4096.log` | 4096 px — fastest blocked | 661.5 s | 27.72 GiB | 8.94/10 |
| `l1_2816x1536.log` | 2816x1536 px — lowest peak | 762.5 s | 24.43 GiB | 9.03/10 |

Timed arm only (`--no-profile`): the profiled arm doubles the cost and can hit the runtime deadlock, and
peak, runtime, occupancy and read amplification all come from the timed arm.

`l1_reproducibility.log` — two whole-grid untiled runs in one process (587.1 s, 575.5 s), which is what
ruled FFTW wisdom out as the cause of the earlier timing spread.

`l1_4096_profiled.log` — the chosen L1 size with the profiled arm: FFTW 56.8% of working samples, the
blocked path's own reading and preprocessing 16.4%, profiling overhead 2.8%.

`l2_2224x1110.log` — the L2 floor (smallest block a 2216x1103 halo permits): 25.91 GiB at 365.4 s, the
lowest peak measured on that granule and 18% slower than `2304x1152`. This is the row that shows occupancy
*falls* past ~230 blocks/thread rather than continuing to climb.

`l1_fft_ladder.log` — three FFT transform-size ladders compared. The shipping power-of-two ladder wins;
coarser costs 24%, multiples of 4 cost 27%.

**Do not observe a row while it runs.** Three earlier L1 sweeps were invalidated by concurrent activity,
the worst of it a `sample` on the live process, which inflated one untiled row 4.5x. Poll with `ps` or
`pgrep` only.

### L2 GSLC

| log | configurations | note |
|---|---|---|
| `sweep_l2b.log` | untiled, 8192, 6144, 4096, 3072 px | one process; **deadlocked** during the 3072 px profile, so that row is absent |
| `sweep_l2c.log` | 3072 px | re-measured alone, which is how the 3072 row was obtained |
| `sweep_l2small.log` | 2304, 2816, 2304x1152, 3072x1536 px | **deadlocked** during the 2816 px profile; only the 2304 px row completed |
| `sweep_aniso1.log` | 2304x1152 px | alone in its own process — the best configuration measured |

Two of the four runs deadlocked, both while profiling. The cause is a **Julia runtime bug** — the macOS
profiler suspends threads while holding the profile lock, and `jl_mach_gc_end` resumes them through the same
libpthread `os_unfair_lock`, in the opposite order — present in every release through 1.13.0 and fixed on
master by `ca49fc2e2`. `tools/golden/profiler_gc_deadlock.jl` reproduces it without AutoRIFT;
`dev/GATES.md` has the traces and the analysis.

Two consequences for reading these logs. **A run that stops writing has not necessarily died** — check for a
0%-CPU survivor with `ps` before assuming it did. And **the runtimes and peaks are unaffected**, because
they come from the unprofiled arm of each configuration; a hang costs the attribution only.

Recover a log's rows into the append-only history with:

```bash
julia --project=tools/golden tools/golden/backfill_history.jl NISAR_L2_PR_GSLC \
    benchmark/results/nisar/sweep_l2b.log
```
