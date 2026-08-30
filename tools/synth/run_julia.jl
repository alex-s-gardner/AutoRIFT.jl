# AutoRIFT.jl's arms of the synthetic comparison.
#
# Three arms, kept separate so a difference is attributable:
#
#   jl-correlator  `correlate!` + `subpixel_peak` at one chip size, with no pyramid, no coarse pass,
#                  no outlier filter and no hole fill. Matching and refinement alone.
#   jl-pipeline    `autorift`, the whole package.
#   jl-rotation    the correlator arm with `rotation = RotationSearch()`, on the shear cases only.
#                  `autoRIFT.py` has no equivalent, so this is not part of the like-for-like
#                  comparison -- it asks whether the residual at steep deformation is recoverable by
#                  a rigid rotation search or is chip-scale strain no rigid model can represent.
#
#   julia --project=tools/synth tools/synth/run_julia.jl [case ...]

using AutoRIFT, Printf
using AutoRIFT: workspace, degenerate, refinement_workspace, subpixel_peak,
                _correlate_rotations!, RotationSearch, NoRotationSearch, params

const SCENES = joinpath(@__DIR__, "scenes")

include(joinpath(@__DIR__, "bundle.jl"))

"""
    correlator_arm(ref, sec, rows, cols, chip, radius, upsampling; rotation) -> (dx, dy, corr)

One displacement per grid point from the correlator alone, in the package's own convention: the offset
from secondary back to reference, which is the negative of feature motion.

The chip comes from the secondary and the search window from the reference, matching `track!` and the
reference's own call sites. `NaN32` where the point cannot be measured -- a window that would leave the
scene, or a chip with no signal -- rather than a number that would be scored as if it meant something.
"""
function correlator_arm(ref, sec, rows, cols, chip::Int, radius::Int, upsampling::Int;
                        rotation = NoRotationSearch())
    ws = workspace(Float32, chip, radius)
    rw = refinement_workspace(upsampling)
    dx = fill(NaN32, length(rows), length(cols))
    dy = fill(NaN32, length(rows), length(cols))
    cc = fill(NaN32, length(rows), length(cols))
    nr, nc = size(ref)
    h = chip ÷ 2
    for (jj, cx) in enumerate(cols), (ii, cy) in enumerate(rows)
        # The window is asymmetric by one pixel, per `correlate!`: it reaches `radius` one way and
        # `radius - 1` the other so the surface is an even 2*radius and zero displacement lands on a
        # sample.
        r0, r1 = cy - h - radius, cy + h + radius - 2
        c0, c1 = cx - h - radius, cx + h + radius - 2
        (r0 >= 1 && c0 >= 1 && r1 <= nr && c1 <= nc) || continue
        chipwin = @view sec[(cy - h):(cy + h - 1), (cx - h):(cx + h - 1)]
        searchwin = @view ref[r0:r1, c0:c1]
        surface = _correlate_rotations!(ws, searchwin, chipwin, (radius, radius), ZNCC(), rotation)
        degenerate(ws) && continue
        ddx, ddy, c = subpixel_peak(rw, surface, (radius, radius), upsampling)
        dx[ii, jj] = Float32(ddx)
        dy[ii, jj] = Float32(ddy)
        cc[ii, jj] = c
    end
    return dx, dy, cc
end

"""
    pipeline_arm(ref, sec, chip, radius, spacing; chip_size_max) -> (dx, dy, chip_size, correlation)

`autorift` on the case's parameters: the pyramid, coarse pass, outlier filter, hole fill and merge on
top of the same correlator the arm above uses alone.

`preprocess = :none`: the scenes are already the arrays to correlate, and filtering here would mean the
Python arms were handed different numbers. `chip_size_max = chip` by default keeps this single-level, so
a difference from the correlator arm is the machinery rather than a coarser chip.
"""
function pipeline_arm(ref, sec, chip::Int, radius::Int, spacing::Int; chip_size_max::Int = chip)
    p = params(; chip_size = (X = chip, Y = chip),
               chip_size_max = (X = chip_size_max, Y = chip_size_max),
               grid_spacing = (X = spacing, Y = spacing),
               search_radius = (X = radius, Y = radius),
               preprocess = :none, subpixel = :pyramid, upsampling = UPSAMPLING)
    out = autorift(ref, sec, p)
    return out.dx, out.dy, out.chip_size, out.correlation
end

function main()
    all = sort(readdir(SCENES))
    wanted = isempty(ARGS) ? all : filter(n -> n in ARGS, all)
    isempty(wanted) && error("no case matched $(ARGS) in $SCENES")

    for name in wanted
        dir = joinpath(SCENES, name)
        isfile(joinpath(dir, "manifest.txt")) || continue
        b = read_bundle(dir)
        ref, sec = b.arrays["reference"], b.arrays["secondary"]
        rows, cols = grid_axes_from(b)
        chip, radius = b.scalars["chip"], b.scalars["radius"]
        spacing, up = b.scalars["grid_spacing"], b.scalars["upsampling"]

        t1 = @elapsed dx, dy, cc = correlator_arm(ref, sec, rows, cols, chip, radius, up)
        write_arm(dir, "jl_correlator", (; dx, dy, correlation = cc), t1)

        t2 = @elapsed pdx, pdy, pcs, pcc = pipeline_arm(ref, sec, chip, radius, spacing)
        write_arm(dir, "jl_pipeline", (; dx = pdx, dy = pdy, correlation = pcc,
                                       chip_size = Int32.(pcs)), t2)

        # Rotation only where there is deformation for it to fit. On a pure translation a rotation
        # search cannot help by construction, so running it there would spend 3x to measure nothing.
        t3 = 0.0
        if b.scalars["is_shear"] == 1
            t3 = @elapsed rdx, rdy, rcc =
                correlator_arm(ref, sec, rows, cols, chip, radius, up;
                               rotation = RotationSearch())
            write_arm(dir, "jl_rotation", (; dx = rdx, dy = rdy, correlation = rcc), t3)
        end

        @printf("%-28s correlator %5.1f s  pipeline %5.1f s  rotation %5.1f s\n",
                name, t1, t2, t3)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
