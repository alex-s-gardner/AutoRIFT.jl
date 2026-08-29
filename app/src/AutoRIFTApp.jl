# A standalone AutoRIFT binary.
#
# Why this exists: the Julia *runtime* is 399 MiB of the 428 MiB an ordinary `using AutoRIFT`
# process occupies before it touches data. On a `t3.micro` that is 43% of the instance, and no
# amount of optimizing AutoRIFT reaches it — `--compile=min` recovers 18 MiB, which is not the
# order of magnitude that matters. A trimmed binary is the only lever on the floor, and it moves
# it to ~37 MiB.
#
# The constraint that shapes every line below: `--trim=safe` requires that every call be
# statically resolvable from `main`. That rules out more than it sounds like. Notably:
#
#   * No Symbol keywords. `params(; similarity = :zncc)` resolves through a `Dict{Symbol,Any}`,
#     so the constructor call after it is a runtime value and unresolvable. Method objects are
#     passed instead — see `AutoRIFT.params` for the documented equivalence.
#   * No `show` on anything, including inside error messages. Interpolating a `Tuple` reaches
#     `Base.repeat` via `textwidth`, which is unresolvable; that is why the errors here are
#     written with pre-formatted integers.
#   * No Rasters, no DimensionalData. Georeferencing is the caller's business; this reads and
#     writes raw arrays.
#
# The I/O format is deliberately headerless raw `Float32`, C-order-free (Julia column-major),
# with the shape passed on the command line. A GeoTIFF reader would pull in the whole GDAL stack
# and defeat the purpose; a production driver is expected to hand off arrays it already has.

module AutoRIFTApp

using AutoRIFT: AutoRIFT, ZNCC, Highpass, PyramidRefine, GardnerFilter,
                Params, False, NoRotationSearch, autorift

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
#
# Hand-rolled rather than ArgParse: the whole dependency would have to trim, and this needs six
# arguments. `tryparse` returns `nothing` on failure rather than throwing, which keeps the error
# path free of the exception machinery.

const USAGE = """
autorift — dense normalized cross-correlation feature tracking

  autorift <reference> <secondary> <ny> <nx> <output> [chip_size] [search_radius]
           [grid_spacing] [block_size] [upsampling]

  <reference>, <secondary>  headerless raw Float32, column-major, ny*nx*4 bytes
  <ny> <nx>                 image shape in pixels
  <output>                  written as three raw Float32 planes: dx, dy, correlation
  [chip_size]               finest chip size, multiple of 4 (default 32)
  [search_radius]           search half-extent in pixels (default 25)
  [grid_spacing]            pixels between search points (default chip_size)
  [block_size]              pixels per block, 0 for one block over the scene (default 0)
  [upsampling]              subpixel refinement factor, power of 2 (default 64)
"""

# `write(Core.stdout, …)` rather than `print`/`println`. `print(x)` forwards to
# `print(Base.stdout::IO, x::Any)` — the global is typed `IO`, so the call is unresolvable no
# matter how concrete `x` is. `Core.stdout` is a concrete `Core.CoreSTDOUT` and `write` on it
# resolves. Two errors in a trimmed build came from nothing but the usage message.
say(s::String) = (write(Core.stdout, s); nothing)

# `open(path, mode)` and an explicit `close`, not `open(f, path, mode)`. The do-block form is
# `open(::Function, ::String, ::Vararg{String})`, and the `Vararg` makes it a `_apply_iterate`
# splat that `--trim` cannot resolve — three more errors, all from the convenience form. The
# `try`/`finally` is what the do-block was providing, written out.
function read_image(path::String, ny::Int, nx::Int)
    a = Matrix{Float32}(undef, ny, nx)
    io = open(path, "r")
    try
        read!(io, a)
    finally
        close(io)
    end
    return a
end

function write_planes(path::String, dx::Matrix{Float32}, dy::Matrix{Float32},
                      corr::Matrix{Float32})
    io = open(path, "w")
    try
        write(io, dx)
        write(io, dy)
        write(io, corr)
    finally
        close(io)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# The trimmable parameter path
# ---------------------------------------------------------------------------
#
# Every method is a concrete instance, so `Params`'s type parameters are known at the call site
# and the whole downstream pipeline specializes. This is the same object `params()` produces for
# the default keywords; the difference is only that no `Dict` lookup stands between the two.
#
# `similarity` is a 1-tuple, not a bare measure: `Params` carries one measure per chip-size level
# so a run can escalate between them, and the last entry applies to every remaining level. A
# 1-tuple therefore means "ZNCC everywhere", and keeps `S` concrete for trimming.
#
# `False()` rather than `false` for `threaded`: `Params` carries threading as a *type* so the grid
# loop's branch is resolved at compile time.
#
# Single-threaded is not only the natural fit for the one-vCPU instance this binary exists for — it
# is currently the only option, and the reason is upstream rather than here. `True()` **trims with
# zero verifier errors and then fails at runtime**: the spawned closure raises a `MethodError` with
# `args=()`, i.e. the task entry point itself was trimmed away. The verifier does not follow task
# entry closures, so this is a clean build that cannot run — the worst failure mode of the three.
#
# Reproduced without AutoRIFT: a six-line program whose whole body is `Threads.@spawn
# atomic_add!(acc, 42)` fails identically. So it is `--trim` and `Threads.@spawn`, not
# `StableTasks` and not the grid loop. Re-test on a later Julia; the change here is one word.
#
# The loss is bounded. Threading is a within-pair optimization, and `benchmark/suite/throughput.jl`
# measures pair-level parallelism at 2.7x intra-pair threading anyway — so the shape this binary
# wants is many single-threaded processes, which is exactly what it is.
function app_params(chip_size::Int, search_radius::Int, grid_spacing::Int, upsampling::Int)
    # The geometry fields are extents — `(X = …, Y = …)` named tuples. Written as literals rather
    # than through `AutoRIFT.extent`, because a positional `Params` call is the trimmable entry point
    # and every value here is already known to be square.
    return Params(
        (ZNCC(),), Highpass(), PyramidRefine(upsampling), GardnerFilter(), False(),
        NoRotationSearch(),
        (X = chip_size, Y = chip_size), (X = 4chip_size, Y = 4chip_size),
        (X = grid_spacing, Y = grid_spacing),
        (X = search_radius, Y = search_radius), 6,
        4, 8, 0.01,
        0.0, 0.0,
        3,
        UInt64(0),
        false,
    )
end

function (@main)(args::Vector{String})::Cint
    if length(args) < 5 || length(args) > 10
        say(USAGE)
        return Cint(2)
    end

    ny = tryparse(Int, args[3])
    nx = tryparse(Int, args[4])
    if ny === nothing || nx === nothing || ny < 1 || nx < 1
        say("error: <ny> and <nx> must be positive integers\n")
        return Cint(2)
    end

    chip_size = 32
    if length(args) >= 6
        c = tryparse(Int, args[6])
        if c === nothing || c < 4 || c % 4 != 0
            say("error: [chip_size] must be a positive multiple of 4\n")
            return Cint(2)
        end
        chip_size = c
    end

    search_radius = 25
    if length(args) >= 7
        r = tryparse(Int, args[7])
        if r === nothing || r < 1
            say("error: [search_radius] must be a positive integer\n")
            return Cint(2)
        end
        search_radius = r
    end

    # Defaults to the chip size, which is what a grid with no overlap between chips means.
    grid_spacing = chip_size
    if length(args) >= 8
        g = tryparse(Int, args[8])
        if g === nothing || g < 1
            say("error: [grid_spacing] must be a positive integer\n")
            return Cint(2)
        end
        grid_spacing = g
    end

    # Zero rather than a missing argument for "one block over the scene": the alternative is a
    # `Union{Nothing,Tuple}` reaching `autorift`, and a runtime-typed union is what `--trim` cannot
    # resolve. Both branches below call a concrete method.
    block_size = 0
    if length(args) >= 9
        b = tryparse(Int, args[9])
        if b === nothing || b < 0
            say("error: [block_size] must be a non-negative integer\n")
            return Cint(2)
        end
        block_size = b
    end

    # `PyramidRefine`'s own default, restated here because a positional `Params` gets no defaults.
    upsampling = 64
    if length(args) >= 10
        u = tryparse(Int, args[10])
        # The same two conditions `PyramidRefine`'s constructor enforces, checked here so a bad value
        # is a usage error rather than an exception from inside the library.
        if u === nothing || u < 2 || (u & (u - 1)) != 0
            say("error: [upsampling] must be a power of 2 greater than 1\n")
            return Cint(2)
        end
        upsampling = u
    end

    reference = read_image(args[1], ny, nx)
    secondary = read_image(args[2], ny, nx)

    p = app_params(chip_size, search_radius, grid_spacing, upsampling)
    # Two calls rather than one with a `Union` argument, for the trimming reason above.
    out = block_size == 0 ? autorift(reference, secondary, p) :
          autorift(reference, secondary, p, (block_size, block_size))
    write_planes(args[5], out.dx, out.dy, out.correlation)
    return Cint(0)
end

end # module
