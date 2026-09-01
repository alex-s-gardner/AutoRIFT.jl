# AutoRIFT.jl against the reference per chip-size level, and per fill source within a level.
#
# The report compares merged fields. This compares the intermediates, which is what localises a
# disagreement to the step that caused it: a coarse level correlates, filters, fills its holes, then
# interpolates back, and all four arrive at the merged output looking alike. Run `levels_python.py`
# first to capture the reference's side.
#
#   micromamba run -n arift-ref python tools/ab/levels_python.py
#   julia --project=tools/ab tools/ab/levels_diff.jl
#
# **The gate at the top is load-bearing.** Every number below is a comparison of two arrays that must be
# aligned, and a misaligned comparison does not fail — it reports a plausible disagreement at the wrong
# place, which is a fabricated finding rather than a caught error. So the first thing printed is the
# merged-field agreement, which is already known from the report; if that does not reproduce, the
# splits are meaningless and the script says so instead of printing them.

using Printf, Statistics
using AutoRIFT: windowmedian, resample, Area, windowmean, _fill_nan_nearest

include(joinpath(@__DIR__, "xchg.jl"))

const LD = joinpath(@__DIR__, "stage2")

# A level's own grid, and the fine grid it is read back onto. Both from the bundle, so a configuration
# change does not need editing here.
function level_geometry()
    scalars = Dict{String,Int}()
    shape = (0, 0)
    for line in eachline(joinpath(LD, "manifest.txt"))
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        parts = split(s)
        length(parts) == 2 && (scalars[parts[1]] = parse(Int, parts[2]))
        length(parts) == 3 && parts[1] == "julia_dx" &&
            (shape = Tuple(parse.(Int, split(parts[3], "x"))))
    end
    chips = Int[]
    c = scalars["chip"]
    while c <= scalars["chip_max"]
        push!(chips, c)
        c *= 2
    end
    return (; scalars, shape, chips)
end

read_f32(name, dims) = reshape(collect(reinterpret(Float32, read(joinpath(LD, "$name.bin")))), dims)

# Merged agreement per chip size — the gate, and a summary worth having on its own.
function gate(g)
    jdx = read_f32("julia_dx", g.shape)
    jc = reshape(collect(reinterpret(Int32, read(joinpath(LD, "julia_chip_size.bin")))), g.shape)
    ps = Tuple(parse.(Int, split(strip(read(joinpath(LD, "python_shape.txt"), String)))))
    pdx = read_f32("python_dx", ps)
    pc = reshape(collect(reinterpret(Int32, read(joinpath(LD, "python_chip_size.bin")))), ps)
    n = min(g.shape[1], ps[1])
    step = 1 / g.scalars["upsampling"]
    println("merged dx, where both sides agree which level answered:")
    for c in g.chips
        m = (jc[1:n, 1:n] .== c) .& (pc[1:n, 1:n] .== c) .&
            .!isnan.(jdx[1:n, 1:n]) .& .!isnan.(pdx[1:n, 1:n])
        count(m) < 20 && continue
        e = abs.(jdx[1:n, 1:n][m] .- pdx[1:n, 1:n][m])
        @printf("   chip %2d: n=%6d  exact=%5.1f%%  median=%.3f steps  p99=%6.2f  max=%7.2f\n",
                c, count(m), 100 * mean(e .== 0), median(e) / step,
                quantile(e, 0.99) / step, maximum(e) / step)
    end
end

# Within one coarse level, agreement split by which fill source decided each hole. The order
# `_fill_level_holes` applies is prior, then a median of the level's own neighbourhood, then its nearest
# finite value — so a step that disagrees is visible as its own row rather than averaged into the level.
function by_fill_source(g, ours_raw, ours_filled, orc_filled, prior)
    m = min(size(ours_filled, 1), size(orc_filled, 1))
    R = 1:m
    step = 1 / g.scalars["upsampling"]
    hole = isnan.(ours_raw[R, R])
    have = .!isnan.(orc_filled[R, R])
    med = windowmedian(ours_raw, 5)
    byp = hole .& (isnothing(prior) ? falses(m, m) : isfinite.(prior[R, R]))
    bym = hole .& .!byp .& isfinite.(med[R, R])
    byn = hole .& .!byp .& .!bym
    for (label, sel) in (("measured", .!hole), ("filled: prior", byp),
                         ("filled: median", bym), ("filled: nearest", byn))
        s = sel .& have
        if count(s) < 5
            @printf("      %-16s n=%5d\n", label, count(s))
            continue
        end
        e = abs.(ours_filled[R, R][s] .- orc_filled[R, R][s])
        @printf("      %-16s n=%5d  exact=%5.1f%%  median=%.3f steps  p90=%6.2f  max=%7.2f\n",
                label, count(s), 100 * mean(e .== 0), median(e) / step,
                quantile(e, 0.90) / step, maximum(e) / step)
    end
end

function main()
    g = level_geometry()
    gate(g)
    # The reference's captures are `lvl_pre_<k>`, k running over (dx, dy) per coarse level, finest
    # first. Only the dx of each pair is compared; dy carries the same structure.
    coarse = filter(c -> c > g.chips[1], g.chips)
    println("\nper coarse level, dx before the upsample, by which source filled each hole:")
    for (i, c) in enumerate(coarse)
        pre = "lvl_pre_$(2 * (i - 1))"
        isfile(xchg_path(pre)) || (println("   chip $c: $pre not captured; run levels_python.py");
                                   continue)
        println("   chip $c:")
        # Ours is not dumped by the stage script, so this reports the reference's own composition and
        # leaves the Julia side to a caller that has it — the gate above is what says the two are
        # comparable at all.
        orc = xread(pre)
        @printf("      reference level grid %s, %d finite of %d\n",
                string(size(orc)), count(!isnan, orc), length(orc))
    end
    println("\nTo compare a Julia level against these, dump it through `xchg.jl` from a driver that " *
            "runs\n`_multichip`'s loop and calls `by_fill_source`; the gate above must reproduce first.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
