# Development notes

The record of validating AutoRIFT.jl against the implementation it descends from. Written for
contributors: none of it is part of the API, and none of it appears on the documentation site.

These files are evidence, not narrative. Each one holds the measurements behind a decision in
`src/`, and a source comment that cites a measurement is worthless once the measurement is gone —
which is why this material is kept here rather than deleted as the port work concluded.

| file | what it holds |
|---|---|
| [`REFERENCE.md`](REFERENCE.md) | the reference implementation: what production actually runs, what changed across its versions, and the conventions — the y sign and the half pixel above all — that a comparison has to agree on before it means anything |
| [`CORRECTNESS.md`](CORRECTNESS.md) | defects AutoRIFT.jl reproduces deliberately, so that a comparison measures a difference rather than two unrelated bugs. Each entry names its evidence and what to change once the golden set agrees |
| [`GATES.md`](GATES.md) | the gate ledger: every confirmed-green measurement with the command that produced it, per golden case |
| [`gpu-feasibility.md`](gpu-feasibility.md) | per-stage device timings behind the GPU backend, and why the correlation agrees to 1e-5 while the displacements agree exactly |
| [`plan-tiling.md`](plan-tiling.md) | the blocked-processing design and the partition invariants it has to hold |
| [`plan-16gib.md`](plan-16gib.md) | where a run's peak memory actually is, which of the plausible levers are real, and the order of work that fits the two NISAR cases on a 16 GiB instance |

Measurements were taken on an Apple M2 Max unless a file says otherwise.

## Citing these files from source

Source comments cite this material by **bare filename** — `` `GATES.md` ``, not a path. A path in a
comment goes stale on the next move and the comment keeps reading as though it were correct. So grep
for the bare name before relocating a measurement, and check that whatever cited it still says
something true.

## Doctests

Doctests run in the documentation build (`makedocs(; doctest = true)`), not in the test suite. Two
reasons, both about `test/runtests.jl`: it would put Documenter on every test job for a check one
docs job already performs, and its testset ordering is load-bearing — the core testsets run with
neither Rasters nor DimensionalData in session, so a core file that started depending on one fails
there rather than passing quietly.

Consequently a `jldoctest` block may assume only `AutoRIFT` itself. Anything needing Rasters, Dates,
or a plot belongs in an `@example` block on a documentation page.

## Examples on the documentation site

Every example runs at build time, so the cost is real and the constraints are: no example over
512x512, and no example reads a file. Synthetic imagery built in the page is what keeps the build
reproducible on a bare checkout.
