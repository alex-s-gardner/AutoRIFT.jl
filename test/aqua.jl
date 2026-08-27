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

@testset "the public surface is declared, and every name in it is real" begin
    # `AutoRIFT.PUBLIC_NAMES` is what semantic versioning covers beyond the exports. Three things
    # have to agree about it — the declaration, the documentation, and this test — and the failure
    # mode it exists to prevent is a name drifting out of one of them silently.
    meta = Base.Docs.meta(AutoRIFT)
    for name in AutoRIFT.PUBLIC_NAMES
        # Declared but absent would make the `public` declaration a lie and the docs unbuildable.
        @test isdefined(AutoRIFT, name)
        # A public name without a docstring is a name a user cannot learn to use. Read from the
        # module's own docs metadata rather than through `Base.Docs.doc`, which needs the REPL's
        # docsystem loaded and is unavailable in a bare test process.
        @test haskey(meta, Base.Docs.Binding(AutoRIFT, name))
    end

    # No name is both exported and listed: exports are what a user needs in scope, these are what
    # they reach for deliberately, and a name in both lists would leave which one it is ambiguous.
    # `names` includes public names from 1.11, so the exports are taken by their defining property.
    exported = Set(n for n in names(AutoRIFT) if Base.isexported(AutoRIFT, n))
    for name in AutoRIFT.PUBLIC_NAMES
        @test !(name in exported)
    end

    # `public` is a 1.11 feature and this package supports 1.10, so the declaration is guarded.
    # Where it applies, it must have taken effect for every name.
    @static if VERSION >= v"1.11"
        for name in AutoRIFT.PUBLIC_NAMES
            @test Base.ispublic(AutoRIFT, name)
        end
    end

    # The block layout types stay internal on purpose: `process_block_size` promises a bit-identical
    # result and no scene-sized array, and nothing about how the blocks are shaped. Listing these
    # would freeze the part that has to stay free to change.
    for internal in (:Block, :BlockLayout, :BlockBuffers, :block_layout, :block_buffers,
                     :correlate_tiled, :PassRunner, :WholeScene, :Blocked, :run_pass)
        @test isdefined(AutoRIFT, internal)
        @test !(internal in AutoRIFT.PUBLIC_NAMES)
    end
end
