# One `profile_nisar.jl` row with a bound on workspace bytes checked out at once.
#
#   AUTORIFT_LIVE_GIB=2 julia --project=tools/golden -t 10,1 tools/golden/live_budget.jl \
#       NISAR_L2_PR_GSLC --blocks 2304x1152 --no-profile
#
# `AutoRIFT.WORKSPACE_LIVE_BYTES` is the knob and `AUTORIFT_LIVE_GIB` is how this sets it; unset or `0`
# leaves it off, which reproduces `profile_nisar.jl` exactly and is the baseline arm.
#
# Why it is worth a sweep. `WORKSPACE_POOL_BYTES` bounds what the pool *retains* and is measured free at 2
# GiB. What it does not bound is what is *in use*: a task holds one workspace at a time, so a run holds as
# many as it has tasks, each sized by whichever radius bucket that task is in. Sizing NISAR L2 blocked at
# `2304x1152` from the terms leaves ~4.5 GiB of 7.8 above the process floor unaccounted for by buffers,
# grid and pool — which is that. The bound cannot change an answer, only delay a task, so the sweep is
# peak against wall clock and occupancy with nothing else moving.
#
# Read the `occupancy` line as well as `peak`: the cost of this bound is threads waiting, and a peak that
# falls while occupancy falls with it has bought memory with time rather than for free.

using AutoRIFT

let gib = tryparse(Float64, get(ENV, "AUTORIFT_LIVE_GIB", "0"))
    if !isnothing(gib) && gib > 0
        AutoRIFT.WORKSPACE_LIVE_BYTES[] = round(Int, gib * 2^30)
        println("WORKSPACE_LIVE_BYTES = $(AutoRIFT.WORKSPACE_LIVE_BYTES[]) bytes ($gib GiB)")
    else
        println("WORKSPACE_LIVE_BYTES unbounded (baseline arm)")
    end
    flush(stdout)
end

include(joinpath(@__DIR__, "profile_nisar.jl"))
