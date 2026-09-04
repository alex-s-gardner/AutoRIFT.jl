# The scene planes every row of `bench_table.jl` reads, split out of `bench_scene.bin`.
#
# `bench_scene.bin` is the full Landsat 8/9 overlap as one file: two `Int64` dimensions, then the
# reference plane and the secondary plane as `Float32`, both column-major. That layout is convenient to
# produce once and inconvenient to measure against, because every row of the table wants the two planes
# as separate files it can `read!` or memory-map — the trimmed binary in particular takes two paths and
# a shape, and has no reader for a combined file.
#
# So this writes `ref.bin`, `sec.bin` and `dims.txt` into the work directory, which is what
# `bench_table.jl` and the standalone binary both expect. Idempotent: it skips a plane that is already
# there at the right size, since the split of a 2.3 GiB file is minutes of I/O and the table is
# re-measured far more often than the scene changes.
#
# The planes are the *bounding box* of a geocoded scene, and a Landsat swath is rotated within it, so
# only 57% of each plane is imagery and the rest is the fill value 0. That matters to anyone cropping a
# sub-window out of these files for a smoke test: the corners are empty, and having data is not
# sufficient either — the bounding-box centre of this pair is covered but carries a fifth the
# high-pass texture of a good window, so nothing correlates there and every level is rejected for
# incoherence. Rows 1537.., cols 4609.. is 100% covered and textured. The table itself correlates the
# whole plane and is unaffected, since a level simply finds nothing in the empty parts.
#
#   julia --project=tools/ab tools/ab/bench_scene.jl [workdir]
#
# `workdir` defaults to `$AUTORIFT_BENCH_DIR`, the same variable `bench_table.jl` reads, so the two
# agree without being told twice.

using Printf

const SCENE = joinpath(@__DIR__, "bench_scene.bin")
const WORK = length(ARGS) >= 1 ? ARGS[1] :
             get(ENV, "AUTORIFT_BENCH_DIR", error("pass a work directory or set AUTORIFT_BENCH_DIR"))

# The header is two `Int64`s; the planes follow it back to back. Read rather than assumed, and checked
# against the file's own length — a shape that disagrees with the byte count means this is not the file
# it claims to be, and every downstream row would silently correlate garbage.
function scene_shape(io)
    nr = read(io, Int64)
    nc = read(io, Int64)
    (nr > 0 && nc > 0) || error("$SCENE: nonsensical shape $(nr)x$(nc)")
    want = 16 + 2 * nr * nc * sizeof(Float32)
    got = filesize(SCENE)
    got == want ||
        error("$SCENE: header says $(nr)x$(nc), which needs $want bytes; the file has $got")
    return nr, nc
end

# One plane, streamed in row-bands rather than read whole. The scene is 1.08 GiB per plane, and holding
# a source and a destination copy at once is 2.2 GiB of the peak this table exists to measure.
function copy_plane(io, path, nr, nc; band = 512)
    if isfile(path) && filesize(path) == nr * nc * sizeof(Float32)
        @printf("  %-8s already present, skipping\n", basename(path))
        skip(io, nr * nc * sizeof(Float32))
        return path
    end
    buf = Matrix{Float32}(undef, nr, min(band, nc))
    open(path, "w") do out
        col = 1
        while col <= nc
            w = min(band, nc - col + 1)
            v = view(buf, :, 1:w)
            read!(io, v)
            write(out, v)
            col += w
        end
    end
    @printf("  %-8s %d x %d  (%.2f GiB)\n", basename(path), nr, nc,
            nr * nc * sizeof(Float32) / 2^30)
    return path
end

function main()
    isfile(SCENE) || error("$SCENE not found; see tools/realdata/README.md for the cache")
    mkpath(WORK)
    open(SCENE, "r") do io
        nr, nc = scene_shape(io)
        @printf("scene %d x %d -> %s\n", nr, nc, WORK)
        copy_plane(io, joinpath(WORK, "ref.bin"), nr, nc)
        copy_plane(io, joinpath(WORK, "sec.bin"), nr, nc)
        write(joinpath(WORK, "dims.txt"), "$nr $nc\n")
    end
    println("wrote dims.txt")
end

main()
