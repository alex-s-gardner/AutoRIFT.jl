# Reading and writing the per-case bundles, shared by every Julia script here.
#
# The format is the one `tools/ab` established: raw column-major `.bin` files beside a `manifest.txt`
# naming each array's dtype and shape, plus scalar `key value` lines. Reused rather than reinvented so
# `tools/ab/stage2_python.py`'s reader works on these bundles unchanged.

const UPSAMPLING = 16

const BUNDLE_DTYPES = Dict("Float32" => Float32, "Float64" => Float64, "Int32" => Int32)

"""
    read_bundle(dir) -> (; arrays, scalars)

Every array and scalar a case bundle holds.
"""
function read_bundle(dir::AbstractString)
    arrays = Dict{String,Matrix}()
    scalars = Dict{String,Int}()
    for line in eachline(joinpath(dir, "manifest.txt"))
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        parts = split(s)
        if length(parts) == 2
            scalars[parts[1]] = parse(Int, parts[2])
            continue
        end
        name, dtype, shape = parts
        T = get(BUNDLE_DTYPES, dtype) do
            error("bundle $dir names array $name with unsupported dtype $dtype")
        end
        dims = Tuple(parse.(Int, split(shape, "x")))
        arrays[name] = reshape(collect(reinterpret(T, read(joinpath(dir, name * ".bin")))), dims)
    end
    return (; arrays, scalars)
end

"""
    read_case(dir) -> Dict{String,String}

The case's factor settings, as written by `scenes.jl`. Values stay as strings and are parsed by the
caller that needs them, so adding a factor to `Case` needs no change here.
"""
function read_case(dir::AbstractString)
    out = Dict{String,String}()
    for line in eachline(joinpath(dir, "case.txt"))
        parts = split(strip(line), ' '; limit = 2)
        length(parts) == 2 && (out[parts[1]] = parts[2])
    end
    return out
end

"""
    grid_axes_from(b) -> (rows, cols)

The pixel rows and columns a bundle's grid sits on, recovered from the grid arrays themselves rather
than recomputed from the margin formula. Reading them back means an arm cannot silently correlate a
different grid than the truth was sampled on.
"""
function grid_axes_from(b)
    gy, gx = b.arrays["grid_y"], b.arrays["grid_x"]
    rows = round.(Int, @view gy[:, 1])
    cols = round.(Int, @view gx[1, :])
    return rows, cols
end

"""
    write_arm(dir, name, fields, seconds)

One arm's output, as `<name>_<field>.bin` beside a `<name>.txt` manifest carrying the shapes and the
wall-clock cost.

Cost is recorded per arm because an accuracy difference bought with an order of magnitude more compute
is a different finding from a free one.
"""
function write_arm(dir::AbstractString, name::AbstractString, fields::NamedTuple, seconds::Real)
    open(joinpath(dir, name * ".txt"), "w") do io
        println(io, "# name dtype shape")
        for (k, A) in pairs(fields)
            fname = string(name, "_", k)
            open(joinpath(dir, fname * ".bin"), "w") do bin
                write(bin, A)
            end
            println(io, fname, " ", eltype(A), " ", join(size(A), "x"))
        end
        println(io, "seconds ", seconds)
    end
    return nothing
end

"""
    read_arm(dir, name) -> Union{NamedTuple,Nothing}

An arm's arrays and its cost, or `nothing` if the arm was not run for this case.

`nothing` rather than an error: the rotation arm runs on the shear cases only, and the Python and
OpenCV arms are separate processes that may not have been run yet.
"""
function read_arm(dir::AbstractString, name::AbstractString)
    path = joinpath(dir, name * ".txt")
    isfile(path) || return nothing
    arrays = Dict{String,Matrix}()
    seconds = NaN
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        parts = split(s)
        if length(parts) == 2
            parts[1] == "seconds" && (seconds = parse(Float64, parts[2]))
            continue
        end
        fname, dtype, shape = parts
        T = BUNDLE_DTYPES[dtype]
        dims = Tuple(parse.(Int, split(shape, "x")))
        # Keyed by the field name with the arm prefix removed, so a caller reads `a.dx` regardless of
        # which arm it came from.
        key = replace(fname, name * "_" => "")
        arrays[key] = reshape(collect(reinterpret(T, read(joinpath(dir, fname * ".bin")))), dims)
    end
    return (; arrays, seconds)
end
