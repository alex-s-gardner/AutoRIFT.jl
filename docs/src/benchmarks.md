# Benchmarks

Measured run times and memory, against the Python reference and across configurations of this package.
[Performance](@ref) says which settings to change; this page is the record behind it.

Unless a section says otherwise, numbers are measured on an Apple M2 Max with Julia 1.12.5, as the minimum
over many samples (a shared machine's single-shot medians vary by up to 3× under load).
`benchmark/run.jl` reproduces the single-pair numbers; [`benchmark/README.md`](https://github.com/alex-s-gardner/AutoRIFT.jl/blob/main/benchmark/README.md)
covers the suite and the regression gate.

## Against the Python reference

The reference is `nasa-jpl/autoRIFT` v2.1.2 inside `hyp3-autorift` 0.28.4. Both run the golden test set,
12 threads, one process per case.

### From the granule to `dx`/`dy`

Every stage on the Julia side (geogrid, scene read, filter, coregistration, correlation) against the
reference container's own logged wall clock from the last download to the finished product. The scenes are
on local disk before either clock starts. The 15 optical and burst cases:

| case | Julia s | Python s | | case | Julia s | Python s | |
|---|---:|---:|---:|---|---:|---:|---:|
| LC08 `009011` | 54.9 | 376 | 6.8x | LT05 `001013` | 49.9 | 190 | 3.8x |
| LC08 `060018` | 67.2 | 328 | 4.9x | LT05 `060018` | 20.7 | 118 | 5.7x |
| LC08 `062018` | 33.0 | 289 | 8.8x | S1A `1SSV_20240618` a | 460.3 | 1536 | 3.3x |
| LC09 `215109` | 39.9 | 327 | 8.2x | S1A `1SSV_20240618` b | 176.3 | 632 | 3.6x |
| LE07 `061018` (2012) | 85.3 | 271 | 3.2x | S1C `1SSV_20250416` | 45.1 | 383 | 8.5x |
| LE07 `061018` (2013) | 79.7 | 358 | 4.5x | S2A `20200626` | 14.4 | 186 | **12.9x** |
| LE07 `063018` | 94.1 | 335 | 3.6x | S2B `20200612` | 21.0 | 187 | 8.9x |
| LT04 `063018` | 29.4 | 195 | 6.6x | **all 15** | **1,271** | **5,711** | **4.5x** |

The median is 6.6x. It is smaller than the correlator's ratio below because the correlator is 4 to 63% of
a Julia run here, and the stages in front of it are GDAL and arithmetic on both sides.

The chain also reaches three Sentinel-1 full-SLC pairs and two NISAR granules. Over all twenty it takes
7,837 s against the reference's 19,320 s, or 2.5x. The two NISAR granules are 79% of the Julia total and
are the only cases where the ratio is below 3x. `dev/GATES.md` has the per-stage breakdown for all twenty.

Peak memory is higher on this path. The chain holds the raw band, the filtered band and the crop as
whole-scene `Float32` arrays, where the reference writes its filtered scene to disk and reads back a
window:

| case | Julia peak | Python peak | |
|---|---:|---:|---:|
| S2B `20200612` | 4.09 GiB | 2.01 GiB | Julia 2.03x |
| LC08 `062018` | 7.87 GiB | 4.42 GiB | Julia 1.78x |
| LT05 `060018` | 8.35 GiB | 5.80 GiB | Julia 1.44x |

`tools/golden/e2e_run.jl` reproduces these.

### The correlator alone

The same captured inputs on both sides, so neither re-derives the grid. `jl block` is the
`process_block_size` the library picks when none is given.

| case | jl block | Julia s | Python s | |
|---|---|---:|---:|---:|
| LC08 `009011` | 1024 | 19.6 | 180.3 | 9x |
| LC08 `060018` | 1024 | 6.1 | 46.1 | 8x |
| LC08 `062018` | 1024 | 8.3 | 86.2 | 10x |
| LC09 `215109` | 1024 | 6.1 | 77.8 | 13x |
| LE07 `061018` (2012) | 1024 | 2.6 | 13.1 | 5x |
| LE07 `061018` (2013) | 1024 | 6.1 | 51.6 | 8x |
| LE07 `063018` | 1024 | 8.9 | 85.5 | 10x |
| LT04 `063018` | 1024 | 5.0 | 60.2 | 12x |
| LT05 `001013` | 1024 | 2.6 | 37.8 | **14x** |
| LT05 `060018` | 1024 | 3.4 | 19.5 | 6x |
| NISAR L1 RSLC | `3593x1843` | 467.2 | 812.2 | **2x** |
| NISAR L2 GSLC | `2883x1531` | 247.3 | 516.4 | **2x** |
| S1A `1SSH_20150828` | 1024 | 16.6 | 111.0 | 7x |
| S1A `1SSH_20151120` | 1024 | 10.5 | 86.8 | 8x |
| S1A `1SSH_20170221` | 1024 | 25.1 | 173.6 | 7x |
| S1A `1SSV_20240618` a | 1024 | 30.1 | 175.5 | 6x |
| S1A `1SSV_20240618` b | 1024 | 13.2 | 63.7 | 5x |
| S1B `1SDH_20180809` | 1024 | 16.9 | 111.1 | 7x |
| S1C `1SDV_20250416` | 1024 | 12.5 | 115.8 | 9x |
| S1C `1SSV_20250416` | 1024 | 4.1 | 33.4 | 8x |
| S2A `20200626` | 1024 | 3.7 | 30.8 | 8x |
| S2B `20200612` | 1024 | 5.9 | 36.0 | 6x |

The median is 8x, the best 14x and the worst 2x. The two NISAR granules have the largest search windows:
the mean window on the L1 grid is 115,000 px against 4,300 on a Landsat grid. At that size the transform
is the whole cost on both sides. AutoRIFT.jl sustains 31–41 G-butterfly/s of transform work across these
cases, where the reference reaches 2.3 on the smallest windows and 15.7 on the largest, so a case's ratio
mostly reflects its window size.

AutoRIFT.jl measures 0.875 to 1.000 of the reference's point count, median 0.987. A run that measures fewer
points does less work. `dev/CORRECTNESS.md` lists each remaining difference and the measurement behind it.

The Julia memory columns are the run's peak above what the harness already held, so they are the cost of
the correlation itself. Python's is `ru_maxrss` for the whole process, which includes its interpreter and
the inputs. Compare within a column; across columns the numbers are indicative:

| case | Julia untiled | Julia blocked | Python, whole process |
|---|---:|---:|---:|
| NISAR L2 GSLC | 56.5 GiB | **6.1 GiB** | 49.4 GiB |
| NISAR L1 RSLC | 35.0 | **11.0** | 26.8 |
| S1B `1SDH_20180809` | 18.9 | **0.4** | 14.9 |

Every block size measured gives a result bit-identical to the untiled run, in `dx` and `dy` at equal point
counts, on all 22 cases. `tools/golden/e2e_table.jl` reproduces these; `dev/plan-16gib.md` has the
block-size sweeps and the memory budget.

### One pair, against `runAutorift()`

`autoRIFT` 2.1.2's `runAutorift()` from the conda-forge build, on 1024 × 1024, chip 32, radius 25,
spacing 32, a 900-point grid:

| | time | measured |
|---|---:|---:|
| autoRIFT 2.1.2 (C++ core) | 456 ms | 900/900 |
| AutoRIFT.jl, serial | 148 ms | 900/900 |
| AutoRIFT.jl, 8 threads | **41 ms** | 900/900 |

AutoRIFT.jl is 3.1× faster serial and 11.1× faster threaded, and the displacements agree exactly. The
reference's production driver sets `mpflag = 0`, so its multi-threaded path is not used in production; the
threaded row compares against that.

## One pair

1024 × 1024, chip 32, radius 25, spacing 32, a 900-point grid:

| | time | allocation |
|---|---:|---:|
| serial | 124.9 ms | — |
| 8 threads | **38.2 ms** | 32.5 MiB |

Inner loops allocate nothing per point. The 32.5 MiB is buffers sized once per pass, and the test suite
asserts that `correlate!` and `peak_offset` allocate nothing.

## Pair-level against intra-pair threading

512 × 512, 16 consecutive pairs, 8 threads:

| regime | pairs/sec | |
|---|---:|---|
| serial | 44.7 | |
| intra-pair threading | 116.9 | 2.6× over serial |
| **pair-level, one cache per worker** | **299.4** | 6.7× over serial, 2.6× over intra-pair |

## Cold start

Fresh processes, same pair as [One pair](@ref):

| | first call | wall |
|---|---:|---:|
| no wisdom, no precompile workload | 3.39 s | 4.28 s |
| persisted FFTW wisdom | 2.35 s | 3.72 s |
| precompile workload, wisdom cold | 1.13 s | 2.40 s |
| **both, steady state** | **0.13 s** | **1.36 s** |

Wisdom and the precompile workload together cut the first call 26× and the wall clock 3.1×. `using
AutoRIFT` takes 1.00 s with the workload against 0.87 s without, and precompilation takes 5.9 s once per
install.

On a production grid the planning cost is larger. The golden `S1A_..._20151120` case has an 11.5 M-point
grid, run untiled at 10 threads in four fresh processes in sequence:

| run | wall | CPU |
|---|---:|---:|
| first | **28.8 s** | 71.9 s |
| second, third, fourth | 6.5, 6.5, 6.4 s | 48.1, 48.9, 48.3 s |

Planning takes about 22 s against a 6.5 s correlation. The first run also uses 24 more CPU-seconds, so the
extra time is planner work.

## Trimmed binary

`app/` builds a statically compiled executable. 97% of an ordinary process's 408 MiB memory floor is the
Julia runtime, and the package is about 7 MiB of it.

| | trimmed binary | `julia -t1`, same work |
|---|---:|---:|
| peak RSS, 512² pair | **27.2 MiB** | 424.2 MiB |
| peak RSS, 2048² pair | **117.8 MiB** | 542.7 MiB |
| wall clock, 512², warm wisdom | **0.06 s** | 1.20 s |
| binary | 3.2 MiB | — |

Every displacement is bit-identical to the library at both sizes. The 20× wall-clock difference is process
startup. `app/README.md` has the build recipe; it needs `--experimental` on Julia 1.12 and is not a released
artifact.

## Parallel headroom on the golden set

All 22 golden cases, run untiled from the reference's captured inputs at 10 threads
(`tools/golden/profile_all.jl`). Occupancy is CPU seconds over wall seconds. "On the table" is the wall
time that perfect scaling over 10 threads would save.

| | wall | CPU | occupancy of 10 | on the table |
|---|---:|---:|---:|---:|
| whole set | **328.5 s** | 1913.2 s | **5.82** | 137.2 s |

The idle time is inside the threaded correlation, in passes with too few points to fill ten threads, such
as the coarse passes and the decimated grids of the coarser chip-size levels.

The stages that run between passes are threaded. In isolation, on a 3400² grid and a 1008² level field at
10 threads:

| stage | serial | threaded |
|---|---:|---:|
| `_grid_phase` | 23.8 ms | **3.7 ms** |
| `_fill_holes!` | 52.9 ms | **32.9 ms** |

End to end, in warmed paired runs with the arms alternated, threading them makes no measurable difference:

| case | with threading | without | measured points |
|---|---:|---:|---:|
| `LC08_L1TP_009011`, 5.5 M grid | 20.2 s, occupancy 9.14 | 19.7 s, occupancy 9.15 | 1,708,352 both |
| NISAR L2 GSLC, thinned | 23.0 s, occupancy 6.28 | 23.1 s, occupancy 6.23 | 239,574 both |

These stages run once per level, and saving about 20 ms per stage per level is under 2% of a twenty-second
run.

## Elsewhere

- [Memory](@ref): peak memory by configuration, block size, and pool bounds.
- [Correlating scenes larger than memory](@ref): run time and memory against block size.
- [Correlating on a GPU](@ref): device against CPU timings, and where the device time goes.
