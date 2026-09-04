# Array exchange between AutoRIFT.jl and the Python reference, with the layout in the file.
#
# The Julia half of `xchg.py`; see that file for why the header exists. In short: Julia is
# column-major and NumPy is row-major, a square array read with the wrong convention is silently
# transposed rather than an error, and a transposed displacement field still looks like a displacement
# field — it reads as a spatial offset in whatever is being measured. So the shape and element type
# travel in the file, `xread` uses them rather than a caller-supplied size, and a file without the
# magic is refused.
#
#     include("xchg.jl")
#     xwrite("dxf", A)        # 2-D, any type in XCHG_TYPES
#     A = xread("dxf")        # same array, same orientation
#
# `xchg_selftest()` asserts the round trip on a deliberately non-square, non-symmetric array — the
# only shape that catches a transpose.

const XCHG_MAGIC = b"ABXC1\0\0\0"
# Type tags, shared with `xchg.py`. Appending is safe; renumbering is not.
const XCHG_TYPES = Dict{Int,DataType}(1 => Float32, 2 => Float64, 3 => Int32, 4 => Int64, 5 => UInt8)
const XCHG_TAGS = Dict{DataType,Int}(v => k for (k, v) in XCHG_TYPES)

xchg_dir() = get(ENV, "AB_XCHG_DIR", joinpath(@__DIR__, "xchg"))
xchg_path(name) = occursin(Base.Filesystem.path_separator, name) ? name :
                  joinpath(xchg_dir(), name * ".abx")

"""
    xwrite(name, A) -> path

Write `A` with its shape and element type in the header.
"""
function xwrite(name::AbstractString, A::AbstractMatrix)
    T = eltype(A)
    tag = get(XCHG_TAGS, T, nothing)
    tag === nothing &&
        error("xchg: unsupported element type $T; add it to XCHG_TYPES in both xchg.jl and xchg.py")
    p = xchg_path(name)
    mkpath(dirname(p))
    open(p, "w") do io
        write(io, XCHG_MAGIC)
        # Little-endian throughout: both sides run on the same machine, and stating it keeps a
        # cross-endian read an explicit gap rather than a silent scramble.
        write(io, htol(Int64(tag)), htol(Int64(size(A, 1))), htol(Int64(size(A, 2))))
        # Column-major, which is Julia's own order, so the buffer needs no rearrangement here and
        # NumPy reads it with `order="F"`.
        write(io, convert(Matrix{T}, A))
    end
    return p
end

"""
    xread(name) -> Matrix

The array [`xwrite`](@ref) stored, in the orientation it had. The shape comes from the file, never
from the caller.
"""
function xread(name::AbstractString)
    p = xchg_path(name)
    open(p, "r") do io
        read(io, length(XCHG_MAGIC)) == XCHG_MAGIC ||
            error("$p was not written by xchg — refusing to guess its layout")
        tag = ltoh(read(io, Int64))
        nr = ltoh(read(io, Int64))
        nc = ltoh(read(io, Int64))
        T = get(XCHG_TYPES, Int(tag), nothing)
        T === nothing && error("unknown type tag $tag in $p")
        A = Matrix{T}(undef, nr, nc)
        read!(io, A)
        eof(io) || error("$p: trailing bytes after a $(nr)x$(nc) $T array")
        return A
    end
end

"""
    xchg_selftest()

Round trip a non-square, non-symmetric array. Any transpose changes the shape or the values.
"""
function xchg_selftest(; verbose = true)
    A = reshape(Float32.((0:14) .* 1.5), 3, 5)
    A[1, 5] = -7.25f0                                  # breaks any accidental symmetry
    B = xread(xwrite("__selftest", A))
    size(B) == size(A) || error("xchg: shape changed $(size(A)) -> $(size(B))")
    B == A || error("xchg: values changed under round trip")
    for T in (Float64, Int32, UInt8)
        C = T.(round.(abs.(A)))
        xread(xwrite("__selftest", C)) == C || error("xchg: round trip failed for $T")
    end
    rm(xchg_path("__selftest"))
    verbose && println("xchg.jl selftest passed")
    return true
end

if abspath(PROGRAM_FILE) == @__FILE__
    xchg_selftest()
end
