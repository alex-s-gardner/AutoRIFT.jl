# Splitting an index range across threads.
#
# The loops this serves — sliding-window reductions, resampling, the whole-scene pad — all have the
# same shape: an outer range whose elements write disjoint parts of the output and read only the
# input, so any partition of that range gives the same answer as running it whole. That makes the
# split a pure scheduling decision, and one helper can carry it for every such loop rather than each
# growing its own copy.

# Slices per thread. More than one so a slice that finishes early has work left to claim: window
# reductions skip a window whose neighbours are all missing, and a no-data border therefore makes
# per-column cost vary by a large factor across one image. Four rather than `track.jl`'s sixteen
# because a slice here is a column band of one array rather than a chunk of correlation points, so
# the spread within a slice is far narrower and the per-slice scratch is paid `SLICES_PER_THREAD`
# times per call.
const SLICES_PER_THREAD = 4

# Work below which the split is not worth making, in element operations — the count a caller passes
# as `cost`. Claiming a slice and waking a task is a few tens of microseconds all told, so a loop
# that would finish in that time must not be split. The coarse levels of the chip-size search reach
# these loops with grids of a few hundred points, which is exactly the case this threshold keeps
# serial.
const PARALLEL_MIN_WORK = 1 << 18

"""
    _parallel_slices(work!, idx, cost)

Run `work!(slice)` over slices of `idx`, on several threads when `cost` justifies it.

`work!` must write only to output positions determined by its slice and read only input. Slices
run concurrently in no particular order, so any dependence between them is a race. `cost` is the
total work of the whole range in element operations; below [`PARALLEL_MIN_WORK`](@ref) the range
runs whole on the calling task.

Disjoint elements are not enough when the output is a `BitArray`: it packs 64 elements into a word,
so two slices whose boundary falls inside a word lose each other's bits. Fill a `Matrix{Bool}`, or
do not split.

Allocate scratch inside `work!` so each slice has its own. A buffer created in the enclosing
function and captured by `work!` is shared by every task, including when it is also reassigned in the
enclosing scope, because Julia boxes such a variable (`src/tile.jl` `_run_task_blocks!` records a
case that silently corrupted several hundred of 3721 points).
"""
function _parallel_slices(work!::F, idx::AbstractUnitRange, cost::Real) where {F}
    nthreads = Threads.nthreads()
    if nthreads == 1 || cost < PARALLEL_MIN_WORK || length(idx) < 2
        work!(idx)
        return nothing
    end
    nslices = min(length(idx), SLICES_PER_THREAD * nthreads)
    slices = collect(Iterators.partition(idx, cld(length(idx), nslices)))
    next = Threads.Atomic{Int}(1)
    ntasks = min(length(slices), nthreads)
    tasks = map(1:ntasks) do _
        Threads.@spawn _parallel_claimed!(work!, slices, next)
    end
    foreach(wait, tasks)
    return nothing
end

# One task's share: claim the next slice until they run out.
#
# A named function rather than a block inside the spawn, for the reason the docstring above gives:
# this frame is where a captured local would otherwise be boxed and shared.
function _parallel_claimed!(work!::F, slices::Vector{<:AbstractUnitRange},
                            next::Threads.Atomic{Int}) where {F}
    while true
        k = Threads.atomic_add!(next, 1)
        k <= length(slices) || break
        work!(slices[k])
    end
    return nothing
end

"""
    _parallel_reduce(accumulate!, init, combine!, idx, cost)

Reduce over slices of `idx` on several threads, and return the combined accumulator.

The counterpart of [`_parallel_slices`](@ref) for a loop that produces a value: `init()` builds one
accumulator per slice, `accumulate!(acc, slice)` folds that slice into it, and `combine!(into, from)`
merges two. `accumulate!` is the first argument so `do`-block syntax reaches it.

Each slice has its own accumulator, and accumulators are combined in slice order rather than
completion order, so the result does not depend on the schedule. It is bit-identical across runs
only if `combine!` is exact: summing integer counts is, whereas floating-point accumulation in a
different grouping is not.

`cost` and its threshold behave as in `_parallel_slices`.
"""
function _parallel_reduce(accumulate!::A, init::I, combine!::C,
                          idx::AbstractUnitRange, cost::Real) where {A,I,C}
    nthreads = Threads.nthreads()
    if nthreads == 1 || cost < PARALLEL_MIN_WORK || length(idx) < 2
        acc = init()
        accumulate!(acc, idx)
        return acc
    end
    nslices = min(length(idx), SLICES_PER_THREAD * nthreads)
    slices = collect(Iterators.partition(idx, cld(length(idx), nslices)))
    # One accumulator per slice, not per task: a task claims several slices, and sharing one
    # accumulator across them would make the result depend on which task claimed what.
    accs = [init() for _ in eachindex(slices)]
    next = Threads.Atomic{Int}(1)
    ntasks = min(length(slices), nthreads)
    tasks = map(1:ntasks) do _
        Threads.@spawn _parallel_reduce_claimed!(accumulate!, accs, slices, next)
    end
    foreach(wait, tasks)
    out = first(accs)
    for k in 2:lastindex(accs)
        combine!(out, accs[k])
    end
    return out
end

# One task's share, folding each claimed slice into that slice's own accumulator. Named for the same
# reason `_parallel_claimed!` is.
function _parallel_reduce_claimed!(accumulate!::A, accs::Vector, slices::Vector{<:AbstractUnitRange},
                                   next::Threads.Atomic{Int}) where {A}
    while true
        k = Threads.atomic_add!(next, 1)
        k <= length(slices) || break
        accumulate!(accs[k], slices[k])
    end
    return nothing
end
