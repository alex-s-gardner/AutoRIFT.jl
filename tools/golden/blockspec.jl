# Block-size specs and the ladder a sweep walks, shared by `mem_nisar.jl` and `block_gate.jl`.
#
# One definition because the two must agree: a gate that sweeps a different ladder from the one the
# measurement swept would pass sizes nobody measured and measure sizes nobody gated.

# A block spec as it is written on the command line: `0` for untiled, `1024` for a square, `2304x1152`
# for a rectangle. Rectangles are not a nicety — NISAR L2's optimum is `2304x1152`, and the square
# `2304x2304` costs 3.7x its peak.
_bslabel(bs::Tuple{Int,Int}) = bs == (0, 0) ? "untiled" :
                               bs[1] == bs[2] ? "$(bs[1]) px" : "$(bs[1])x$(bs[2]) px"

function _parse_blocks(spec::AbstractString)
    out = Tuple{Int,Int}[]
    for tok in split(spec, ',')
        t = strip(tok)
        if occursin('x', t)
            a, b = split(t, 'x')
            push!(out, (parse(Int, a), parse(Int, b)))
        else
            v = parse(Int, t)
            push!(out, (v, v))
        end
    end
    return out
end

# Sizes worth trying for a grid, from its halo, when the caller does not name them.
#
# **The ladder has to straddle the optimum rather than bracket it coarsely**, because the optimum is a
# balance and not an endpoint: peak falls with block size while read amplification rises, so the best
# block is interior. On the golden S1B case, halo 684x256, the arms measure 3.24 GiB at 768x320 with
# readamp 7.23x, 3.01 GiB at 1024 with 2.88x, and 10.32 GiB at 2048 with 1.44x — a doubling ladder from
# the halo would step straight over the winner.
#
# A shaped arm is included because NISAR's optimum is one: `2304x1152` against the halo's `2216x1103`.
# Shaping is not a rule that generalizes — it is what loses on S1B above — but it has to be *tried*.
#
# Every arm must be at least the halo in both axes or `block_layout` rejects it, which is why the square
# ladder starts at the larger halo axis.
function _auto_blocks(h, scene::Tuple{Int,Int})
    roundup(v, u) = cld(v, u) * u
    sx, sy = roundup(h.X, 64), roundup(h.Y, 64)
    out = Tuple{Int,Int}[(0, 0), (sx, sy)]
    for f in (2, 3)
        push!(out, (sx * f, sy * f))
    end
    lo = max(sx, sy)
    for s in (512, 768, 1024, 1536, 2048, 3072, 4096, 6144)
        s >= lo && s < min(scene...) && push!(out, (s, s))
    end
    # Anything past the scene reads the whole thing once and is an untiled run wearing a block size.
    return unique(filter(bs -> bs == (0, 0) || (bs[1] < scene[2] && bs[2] < scene[1]), out))
end

