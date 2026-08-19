using Aqua

# Everything Aqua checks is enforced except `stale_deps`, which is disabled only
# while the package is a scaffold: the numeric and threading dependencies are
# declared for the milestones that consume them but are not yet imported, which
# Aqua correctly reports as stale. Re-enable once M4 lands — the deps are all in
# use by then, and leaving this off permanently would let a genuinely unused
# dependency accumulate.
#
# Extensions get their own Aqua pass in test/extensions.jl at M7, run after the
# core testsets so the core stays verified extension-free.
Aqua.test_all(AutoRIFT; stale_deps = false)
