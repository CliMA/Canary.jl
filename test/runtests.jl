using Test
using Pkg

function include_test(_module)
    println("Starting tests for $_module")
    t = @elapsed include(joinpath(_module, "runtests.jl"))
    println("Completed tests for $_module, $(round(Int, t)) seconds elapsed")
    return nothing
end


@testset "Canary" begin
    all_tests = isempty(ARGS) || "all" in ARGS ? true : false

    function has_submodule(sm)
        any(ARGS) do a
            a == sm && return true
            first(split(a, '/')) == sm && return true
            return false
        end
    end

    for submodule in ["Mesh", "Elements"]
        if all_tests || has_submodule(submodule) || "Canary" in ARGS
            include_test(submodule)
        end
    end
end
