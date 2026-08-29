# Stage 2 of the A/B: the whole pipeline, both sides, on one window.
#
# Stage 1 compared the correlator alone and found it equivalent. This adds everything built on top
# of it -- the chip-size pyramid, the coarse/sparse pass, the outlier filter, the hole filling and
# the smallest-chip-wins merge -- so a disagreement here is attributable to those rather than to
# template matching.
#
# Coverage matters as much as values at this stage: the pyramid and the filter decide *which* points
# get an answer, and two runs can agree on every shared point while disagreeing about most of the
# grid.
#
#   julia --project=tools/ab tools/ab/stage2_julia.jl [npix] [chip_min] [chip_max] [radius] [row] [col]

using AutoRIFT, Serialization, Statistics
using AutoRIFT: params, autorift, _build_grid, imagepair, init

const NPIX = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1024
const CMIN = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 16
const CMAX = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 64
const RADIUS = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 20
const CROW = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 1678
const CCOL = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 2495
const OUT = joinpath(@__DIR__, "stage2")
const CACHE = get(ENV, "AUTORIFT_TESTDATA", expanduser("~/data/autorift/tests"))

dump_array(dir, name, A) = begin
    open(joinpath(dir, name * ".bin"), "w") do io
        write(io, A)
    end
    (name, string(eltype(A)), size(A))
end

function main()
    mkpath(OUT)
    w = deserialize(joinpath(CACHE, "window.jls"))

    n = NPIX
    r0 = clamp(CROW - n ÷ 2, 1, size(w.reference, 1) - n + 1)
    c0 = clamp(CCOL - n ÷ 2, 1, size(w.reference, 2) - n + 1)
    rows, cols = r0:(r0 + n - 1), c0:(c0 + n - 1)
    ref = Matrix{Float32}(w.reference[rows, cols])
    sec = Matrix{Float32}(w.secondary[rows, cols])

    kw = (; chip_size = (X = CMIN, Y = CMIN), chip_size_max = (X = CMAX, Y = CMAX),
          grid_spacing = (X = CMIN ÷ 2, Y = CMIN ÷ 2),
          search_radius = (X = RADIUS, Y = RADIUS),
          preprocess = :highpass, filter_width = 5, subpixel = :pyramid, upsampling = 16)
    p = params(; kw...)

    # The filtered pair, so Python correlates the same arrays. Its own `preprocess_filt_hps` is
    # compared separately; sharing the filtered input keeps a filter difference from showing up
    # here as a pyramid difference.
    pair = imagepair(init(ref, sec; kw...))
    out = autorift(ref, sec, p)
    grid = _build_grid(size(ref), p)

    manifest = Tuple{String,String,Tuple}[]
    push!(manifest, dump_array(OUT, "filtered_reference", Matrix{Float32}(pair.reference)))
    push!(manifest, dump_array(OUT, "filtered_secondary", Matrix{Float32}(pair.secondary)))
    push!(manifest, dump_array(OUT, "grid_x", Matrix{Float64}(grid.x)))
    push!(manifest, dump_array(OUT, "grid_y", Matrix{Float64}(grid.y)))
    push!(manifest, dump_array(OUT, "julia_dx", Matrix{Float32}(out.dx)))
    push!(manifest, dump_array(OUT, "julia_dy", Matrix{Float32}(out.dy)))
    push!(manifest, dump_array(OUT, "julia_correlation", Matrix{Float32}(out.correlation)))
    push!(manifest, dump_array(OUT, "julia_chip_size", Matrix{Int32}(out.chip_size)))
    push!(manifest, dump_array(OUT, "julia_interpolated",
                               Matrix{Int32}(map(v -> v ? Int32(1) : Int32(0), out.interpolated))))

    open(joinpath(OUT, "manifest.txt"), "w") do io
        println(io, "# name dtype shape")
        for (name, T, sz) in manifest
            println(io, name, " ", T, " ", join(sz, "x"))
        end
        println(io, "chip ", CMIN)
        println(io, "chip_max ", CMAX)
        println(io, "grid_spacing ", CMIN ÷ 2)
        println(io, "radius ", RADIUS)
        println(io, "npix ", n)
        println(io, "upsampling 16")
    end

    ok = count(!isnan, out.dx)
    println("grid            ", size(out.dx), " = ", length(out.dx), " points")
    println("julia measured  ", ok, " (", round(100ok / length(out.dx); digits = 1), "%)")
    println("interpolated    ", count(out.interpolated))
    for cs in sort(unique(out.chip_size))
        println("  chip ", lpad(Int(cs), 3), " -> ", count(==(cs), out.chip_size), " points")
    end
    sp = sqrt.(Float64.(w.reference_vx[rows, cols]) .^ 2 .+
               Float64.(w.reference_vy[rows, cols]) .^ 2)
    good = filter(!isnan, sp)
    println("window          rows ", rows, " cols ", cols)
    isempty(good) || println("ITS_LIVE speed  median ", round(median(good); digits = 1),
                             " max ", round(maximum(good); digits = 1), " m/yr")
    println("wrote           ", OUT)
end

main()
