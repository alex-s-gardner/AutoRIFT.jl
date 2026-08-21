# A standalone AutoRIFT binary

A trimmed, statically-compiled `autorift` executable: **25.8 MiB peak RSS against 424.2 MiB** for
the equivalent Julia process, 0.06 s to run a 512² pair, and no Julia installation needed at run time.

This exists for one reason. 97% of an ordinary AutoRIFT process's 408 MiB memory floor is the Julia
runtime itself — AutoRIFT is ~7 MiB of it (see [`docs/memory.md`](../docs/memory.md)). No
optimization inside the package reaches the rest, and `--compile=min` recovers only 18 MiB. On a
`t3.micro` the difference is 43% of the instance against 3.1%.

Not in CI, not a dependency of the package, and not part of `Pkg.test()`. JuliaC needs
`--experimental` on Julia 1.12, so this is a supported build recipe rather than a release artifact.

## Measured

| | trimmed binary | `julia -t1`, same work |
|---|---:|---:|
| peak RSS, 512² pair | **25.8 MiB** | 424.2 MiB |
| peak RSS, 2048² pair | **117.8 MiB** | 542.7 MiB |
| wall clock, 512², warm wisdom | **0.06 s** | 1.20 s |
| binary | 3.2 MiB | — |
| bundle (with runtime libraries + FFTW) | 71 MiB | — |

**Every displacement is bit-identical to the library**, at both 512² and 2048² — verified by `cmp` on
the raw output planes rather than by a tolerance. That is the gate for this directory: a smaller
binary that computes something slightly different is not a smaller binary.

One caveat, stated precisely because the first measurement of it was misleading. The whole output
file is byte-identical *when both processes use the same FFTW wisdom*. When they do not — the binary
run cold against a library run warm, or vice versa — `dx` and `dy` remain bit-identical but
`correlation` differs by up to **3.0e-7** on ~20% of points. FFTW's planner picks a different
algorithm for the same transform size, and a different algorithm reassociates the same
floating-point sum differently. It is not a difference between the binary and the library: two
library runs with different wisdom disagree the same way, and once the wisdom file has settled the
two agree byte-for-byte on repeat runs. Worth knowing before using `cmp` as a regression check.

The 20× wall-clock figure is startup, not correlation, and it is only that large because FFTW wisdom
is already cached (the binary shares the same scratch directory as the library, so a first-ever run
pays ~2.8 s to measure plans). For a driver that launches a process per image pair, startup is a per
pair cost and this is the dominant effect.

## Build

```bash
julia --project=app -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=@jcdrv -m JuliaC --experimental \
      --output-exe autorift --bundle app/build --trim=safe app
```

where `@jcdrv` is any environment with `JuliaC` installed (`Pkg.add("JuliaC")`).

Two details that cost a rebuild each. **`--bundle` is relative to the current directory**, not to the
app directory — `--bundle build` from the repository root writes to `./build`, not `app/build`. And
**the bundle directory must not already exist**; a second build into a populated one fails on the
FFTW artifact with `force=true is required`. So `rm -rf app/build` first.

The binary lands at `app/build/bin/autorift`, with libraries beside it under `app/build/lib` —
`--bundle` carries both the Julia runtime libraries and the `FFTW_jll` artifact, so the tree is
self-contained and needs no Julia installation.

## Run

```
autorift <reference> <secondary> <ny> <nx> <output> [chip_size] [search_radius]
```

Images are headerless raw `Float32`, column-major, `ny*nx*4` bytes. Output is three raw `Float32`
planes on the grid — `dx`, `dy`, `correlation` — with `NaN` where no chip size produced a coherent
estimate.

```bash
app/build/bin/autorift ref.bin sec.bin 512 512 out.bin 32 25
```

Deliberately not GeoTIFF: a raster reader pulls in GDAL and defeats the entire purpose. A production
driver is expected to already hold the arrays and to own georeferencing itself.

## What `--trim=safe` costs, concretely

Every restriction below is something the verifier rejected in this app, not a precaution. Each is
worth knowing before adding a line here.

**Symbol keywords are out.** `params(; similarity = :zncc)` resolves through a `Dict{Symbol,Any}`,
so the constructor call after the lookup is a runtime value. The app builds a `Params` from method
*objects* and calls `autorift(reference, secondary, p)` — a documented overload with a test
asserting it is `@inferred`-clean, so this path cannot silently rot.

**`print`/`println` are out.** `print(x)` forwards to `print(Base.stdout::IO, x::Any)`; the global
is typed `IO`, so the call is unresolvable however concrete `x` is. `write(Core.stdout, s)` resolves,
because `Core.stdout` is a concrete type. Two verifier errors came from the usage message alone.

**`open(f, path, mode)` is out.** The do-block form is `open(::Function, ::String,
::Vararg{String})`, and the `Vararg` becomes an `_apply_iterate` splat. Explicit `open` +
`try`/`finally` + `close` is the same thing, resolvable.

**Interpolating a `Tuple` into an error message is out.** Showing one reaches `textwidth` and
`Base.repeat`. `track!`'s `DimensionMismatch` did this; it compares and reports element counts now.

**`Scratch.@get_scratch!` is out.** Not for the path arithmetic — `scratch_dir` trims fine — but
because the macro also records depot usage for `Pkg.gc()`, which stamps a `DateTime`, and formatting
one reaches `rpad(::String, ::Int, ::Char)` → `Base.repeat`. Four errors, none of them about FFTs.
`src/plans.jl` calls `scratch_dir` directly; the cost is that `Pkg.gc()` may reclaim a wisdom file,
which merely means the next run measures its plans again.

**FFTW.jl plan objects are out**, and this was the hard one — an `FFTW.rFFTWPlan` registers a
finalizer in its *inner* constructor, so no caller can opt out, and `unsafe_destroy_plan` is reached
via `foreach` over a `Vector{FFTWPlan}` with `@nospecialize`. Unresolvable by construction; verified
against a bare ten-line FFTW program. `src/plans.jl` holds raw `Ptr{Cvoid}` plan handles instead,
which the cache never evicts anyway, so the destructor was machinery with no use.

**Name the FFTW library as `FFTW_jll.libfftw3f_path`.** Both obvious alternatives fail, in opposite
directions: `FFTW.libfftw3f` is a `FakeLazyLibrary` resolved by a load-time callback that a trimmed
binary does not have, so the `ccall` dies with a `TypeError` *at run time*; and a bare soname works
in the binary but fails in an ordinary session where the artifact is not on the loader path.

## Threading does not work yet, and the failure is silent

`threaded = true` **trims with zero verifier errors and then fails at run time**: the spawned closure
raises a `MethodError` with `args=()` — the task entry point was trimmed away. The verifier does not
follow task entry closures, so this is a clean build that cannot run, which is the worst of the three
possible outcomes.

This is upstream, not ours. A six-line program whose entire body is `Threads.@spawn
atomic_add!(acc, 42)`, with no AutoRIFT involved, fails identically. Re-test on a later Julia; the
change here is one word (`False()` → `True()` in `app_params`).

The loss is small. Threading is a within-pair optimization, and pair-level parallelism measures
**2.7× faster** than intra-pair threading anyway (`benchmark/suite/throughput.jl`) — so the shape a
batch run wants is many single-threaded processes, which is exactly what this binary is.
