using Aqua
import CommonSolve

# Everything Aqua checks is enforced. `stale_deps` was disabled while the package was a
# scaffold, since dependencies were declared ahead of the milestones that consume them; every
# one is now imported, so the check is on.
#
# Extensions get their own Aqua pass in test/extensions.jl at M7, run after the
# core testsets so the core stays verified extension-free.
#
# `CommonSolve.init` is excluded from the piracy check. Aqua flags it because both argument
# types are `AbstractMatrix`, owned by Base — but that is the entire point of CommonSolve: it
# exists to be a shared name that packages define their own methods on, so that a caller can
# write `init` without knowing which solver answers. The pattern is what the package is for,
# not an accident of dispatch.
Aqua.test_all(AutoRIFT;
              piracies = (; treat_as_own = [CommonSolve.init]))
