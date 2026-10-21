# Contributing

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The test fixtures are committed, so the tests need no Python. To regenerate fixtures, see
[`tools/python_ref/README.md`](tools/python_ref/README.md).

## Documentation

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The build runs every `@example` block and the doctests. [`dev/README.md`](dev/README.md) has the
conventions for documentation examples and doctests. The README's figure is the one committed image;
regenerate it with `julia --project=docs docs/readme_figure.jl` when the README's quickstart changes.

## Benchmarks

[`benchmark/README.md`](benchmark/README.md) covers the benchmark suite and the regression gate.

## Validation record

[`dev/README.md`](dev/README.md) indexes the contributor-facing record of validating AutoRIFT.jl against
the implementation it descends from. Source comments cite those files by bare filename.

## Standalone executable

[`app/README.md`](app/README.md) builds a trimmed standalone binary.
