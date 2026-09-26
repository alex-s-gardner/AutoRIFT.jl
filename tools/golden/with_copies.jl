# One `profile_nisar.jl` row with the per-block copies that `src/` no longer makes put back.
#
#   julia --project=tools/golden -t 10,1 tools/golden/with_copies.jl NISAR_L2_PR_GSLC \
#       --blocks 2304x1152 --no-profile
#
# Takes the same arguments as `profile_nisar.jl`, which it drives unmodified — so the arm is comparable
# to a row measured by that harness rather than to a different measurement. It is the *baseline* half of
# the A/B in `dev/plan-16gib.md`; running `profile_nisar.jl` itself gives the other half.
#
# What it restores, both of which cost a window-sized allocation per image per block:
#
#   * **`_prepare_block`'s fallthrough to `_prepare` for `NoPreprocess`.** That is `copy(img),
#     copy(mask)` in `preprocess` and a second `copy` pair in `replace_nonfinite`, reproducing arrays
#     `_block_pair!` has just written into the buffers. `src/tile.jl` substitutes the non-finite values
#     in place instead, which is the part that cannot simply be dropped — see `test/tile.jl`, "a blocked
#     run equals an untiled one at preprocess = :none".
#
#   * **`_read_window!`'s block-sized temporary on a strided parent.** The generic method materializes
#     `img[rows, cols]` and then copies it in, because a view of a *lazy* array would be read one pixel
#     at a time. A mapped plane has nothing to defer, so `src/tile.jl` takes the view.
#
# **The counters are load-bearing.** An override whose signature does not match what the driver calls is
# a silent no-op, and the arm would then report the current code as though it were the baseline. Both
# counts print at exit; `_read_window!` should be twice `_prepare_block`, one read per image per block.
#
# !!! warning "This appends to the append-only history"
#     `profile_nisar.jl` ends in `save_results`, and a row written from here describes internals that are
#     not in `src/`. `render_history` shows the newest row per block size, so such a row supersedes a real
#     measurement of the same size in every later report. Copy the case's
#     `$AUTORIFT_GOLDEN_CACHE/mem/prof_<case>.jls` aside before running this and restore it afterwards.

using AutoRIFT
using AutoRIFT: BlockBuffers, ImagePair, Params, NoPreprocess

const PREPARED = Threads.Atomic{Int}(0)
const WINDOWS = Threads.Atomic{Int}(0)

function AutoRIFT._prepare_block(::BlockBuffers, raw::ImagePair, p::Params, ::NoPreprocess,
                                 ::Int, ::Int)
    Threads.atomic_add!(PREPARED, 1)
    return AutoRIFT._prepare(raw, p)
end

function AutoRIFT._read_window!(dest::AbstractMatrix, img::StridedMatrix, rows, cols)
    Threads.atomic_add!(WINDOWS, 1)
    copyto!(dest, img[rows, cols])
    return dest
end

atexit() do
    println("\nRESTORED _prepare_block=$(PREPARED[]) _read_window!=$(WINDOWS[])")
    flush(stdout)
end

include(joinpath(@__DIR__, "profile_nisar.jl"))
