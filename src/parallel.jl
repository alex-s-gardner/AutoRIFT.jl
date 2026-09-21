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

`work!` must write only to output positions determined by the slice it is given, and read only
input — the slices are handed out in no particular order and run concurrently, so any dependence
between them is a race. `cost` is the total work the whole range represents, in element operations;
below [`PARALLEL_MIN_WORK`](@ref) the range runs whole on the calling task.

Disjoint *elements* are not enough when the output is a `BitArray`: it packs 64 of them into one
word, so two slices whose boundary falls inside a word write that word concurrently and lose each
other's bits. A loop filling one needs a `Matrix{Bool}` or must stay whole.

**Scratch belongs inside `work!`.** Allocating it there gives every slice its own, which is what
makes the helper safe for any caller: a buffer created in the enclosing function and captured
instead would be shared by every task at once. Julia also boxes a name that is assigned both inside
a closure and in the enclosing scope, so a captured-and-reassigned buffer is shared even when it
looks per-task — `src/tile.jl` records that failure at `_run_task_blocks!`, where it silently
corrupted several hundred of 3721 points.
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
        StableTasks.@spawn _parallel_claimed!(work!, slices, next)
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
