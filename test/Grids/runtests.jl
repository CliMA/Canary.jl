using Test
using Pkg

include(joinpath("..", "testhelper.jl"))

function include_test(_module)
    println("Starting tests for Grids submodule: $_module")
    t = @elapsed include(joinpath(_module, "runtests.jl"))
    println("Completed tests for $_module, $(round(Int, t)) seconds elapsed")
    return nothing
end

@testset "Local geometry" begin
    runmpi(joinpath(@__DIR__, "test_local_geometry.jl"))
end

@testset "Min node distance" begin
    runmpi(joinpath(@__DIR__, "test_min_node_distance.jl"))
end

@testset "Canary.Grids submodules" begin
    all_tests = isempty(ARGS) || "all" in ARGS ? true : false

    function has_submodule(sm)
        any(ARGS) do a
            a == sm && return true
            first(split(a, '/')) == sm && return true
            return false
        end
    end

    for submodule in ["Meshes", "Elements"]
        if all_tests || has_submodule(submodule) || "Canary" in ARGS
            include_test(submodule)
        end
    end
end
