using MPI
using Test

include(joinpath("..", "..", "testhelper.jl"))

@testset "Operators" begin
    runmpi(joinpath(@__DIR__, "test_operators.jl"))
end
