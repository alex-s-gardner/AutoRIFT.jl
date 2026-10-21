# Performance

Which kind of parallelism to use, how to avoid paying FFT planning in every process, and what not to
reach for. [Benchmarks](@ref) has the measurements behind each recommendation.

## Parallelism: pair-level beats intra-pair

`threaded = true` splits one pair's grid points across threads. The alternative gives each thread a whole
pair and its own [`AutoRIFT.Cache`](@ref). On 16 consecutive 512² pairs at 8 threads, pair-level runs 2.6×
faster than intra-pair threading.

Intra-pair threading leaves a serial part (preprocessing, resampling, the merge), and the sparse search
leaves some threads with more work than others. A whole pair per thread has neither, and each thread's
working set stays in cache. A worker keeps its buffers and FFT plans across pairs, which is why `init`,
`reinit!` and `autorift!` are public API. [Correlating many pairs](@ref) shows how to use them.

Prefer pair-level whenever there is more than one pair. Use `threaded = true` when there is only one.

## Cold start

A driver that launches a process per image pair pays load and first-call compilation in every process,
and together they can exceed the correlation. Persisted FFTW wisdom and the package's precompile workload
together cut a 1024² pair's first call from 3.39 s to 0.13 s.

FFTW wisdom is saved to a scratch directory and read by every later process. `__init__` calls
`AutoRIFT.load_fftw_wisdom!`, and [`AutoRIFT.warm_plans!`](@ref) calls `AutoRIFT.save_fftw_wisdom!`, so
every process imports what is on disk and every pass saves what it measured. The file is keyed by CPU
model and FFTW version; [`AutoRIFT.fftw_wisdom_path`](@ref) gives its location.
[`AutoRIFT.precompute_fftw_wisdom`](@ref) plans the sizes that search radii up to a given bound reach,
except those that depend on the grid's own maximum radius.

On a production grid, planning costs more than the correlation: 22 s against 6.5 s on an 11.5 M-point
Sentinel-1 grid. A process-per-pair driver in a fresh container starts with an empty scratch directory and
pays that on every pair. Carry the file across processes by baking it into the image, or set
`AUTORIFT_FFTW_WISDOM_DIR` to a mounted directory. A process that has to measure plans without any saved
wisdom emits a warning that names the file it looked for.

`correlation` differs between runs with different FFTW wisdom, by up to 3.0e-7 on about 20% of points,
because the planner can pick a different algorithm for the same transform size. `dx` and `dy` are
unaffected. A byte-for-byte comparison of `correlation` is only a valid regression check between runs in
the same wisdom state.

## Small instances

97% of an ordinary process's memory floor is the Julia runtime. `app/` builds a statically compiled
executable that runs a 512² pair at 27.2 MiB peak RSS, with displacements bit-identical to the library.
`app/README.md` has the build recipe; it is not a released artifact.

## Where a threaded run is idle

On the golden set at 10 threads, a run keeps about 5.8 of the 10 threads busy. The idle time is in
correlation passes with too few points to fill the threads, such as the coarse passes and the decimated
grids of the coarser chip-size levels. The limit is the work available rather than load balancing.

An untiled threaded pass splits its points into 16 chunks per thread, and each task claims the next
unclaimed chunk. Sorting chunks heaviest first is not done: `dev/GATES.md` bounds the gain at 1.16× and the
measured gain was zero, while scoring the chunks costs 155 ms per pass on an 11.5 M-point grid. A blocked
run does order its blocks heaviest first, by the sum of `radius_x × radius_y` over their points.

## What not to reach for

- **A GPU on an otherwise free machine.** On the correlation pass the device is 1.7–3.0× one CPU core,
  and 0.35× eight threads on a whole pair. It does not add to the CPU: moving one core's work to it gains
  1.6%, and oversubscribing costs 22%. It helps when one core is all a process has, as in a
  process-per-pair driver. [Correlating on a GPU](@ref) has the details.
- **A coarser transform size.** Sizes already come from a fixed ladder, and a coarser ladder is slower.
  [Memory](@ref) has the measurement.
- **A larger block, for speed.** Block size trades read amplification against thread occupancy and peak
  memory, and the fastest block is not the largest. [Correlating scenes larger than memory](@ref) sizes it.

`benchmark/README.md` records the changes that were measured and rejected, with their numbers: a global
integral image, SAD and SSD similarity measures, and windowed-histogram outlier filtering.
