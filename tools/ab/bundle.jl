# The bundle format the A/B stages write and every reader reads.
#
#     include(joinpath(@__DIR__, "bundle.jl"))
#     shapes, scalars = read_manifest(dir)
#     dx = read_bin(dir, "julia_dx", Float32, shapes["julia_dx"])
#
# A stage writes one raw byte dump per array plus a text manifest: `name dtype RxC` for an array,
# `name value` for a scalar. Raw bytes rather than a container format because the point of the
# harness is that both sides read the same bytes, and NPZ is not a dependency of this package.
#
# **A scalar is not necessarily a number.** `dtype UInt8` records which correlator entry point the
# run used, and a reader that parses every scalar as an `Int` throws on it — `ArgumentError: invalid
# base 10 digit 'F' in "Float32"` — after the measurement it was going to report has already been
# computed and discarded. So values are parsed as `Int` where they parse and kept as `String`
# otherwise, and callers index the result as they always did.
#
# One parser, in one file, because there were four copies and adding one non-numeric scalar to the
# writer broke three of them at once. `xchg.jl` is shared the same way.

# The manifest at `dir`, as `(shapes, scalars)`.
#
# `scalars` is `Dict{String,Any}`: numeric entries arrive as `Int` so arithmetic and `@printf("%d")`
# work unchanged, and a non-numeric one arrives as the `String` it is rather than throwing.
function read_manifest(dir)
    shapes = Dict{String,Tuple{Int,Int}}()
    scalars = Dict{String,Any}()
    for line in eachline(joinpath(dir, "manifest.txt"))
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        parts = split(s)
        if length(parts) == 2
            v = tryparse(Int, parts[2])
            scalars[parts[1]] = v === nothing ? String(parts[2]) : v
        else
            dims = parse.(Int, split(parts[3], "x"))
            shapes[parts[1]] = (dims[1], dims[2])
        end
    end
    return shapes, scalars
end

# One array from the bundle. `dims` comes from the manifest rather than from the caller, so a shape
# and its bytes cannot disagree.
read_bin(dir, name, T, dims) =
    reshape(collect(reinterpret(T, read(joinpath(dir, name * ".bin")))), dims)

# The shape the Python side wrote beside its own arrays, which need not match the Julia grid: the
# reference truncates its grid by one point per axis, so a comparison is over the overlap.
read_python_shape(dir) =
    Tuple(parse.(Int, split(strip(read(joinpath(dir, "python_shape.txt"), String)))))
