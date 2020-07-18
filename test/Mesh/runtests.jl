using MPI
using Test

include(joinpath("..", "testhelper.jl"))

@testset "Mesh BrickMesh" begin
    runmpi(joinpath(@__DIR__, "test_mesh_utilities.jl"))
end

@testset "Mesh Metrics" begin
    runmpi(joinpath(@__DIR__, "test_metrics.jl"))
end

@testset "Mesh Topology" begin
    runmpi(joinpath(@__DIR__, "test_topologies.jl"))
end

@testset "Mesh MPI Tests" begin
    runmpi(joinpath(@__DIR__, "mpi_test_centriod.jl"), ntasks = 3)
    runmpi(joinpath(@__DIR__, "mpi_test_connect.jl"), ntasks = 3)
    runmpi(joinpath(@__DIR__, "mpi_test_connect_1d.jl"), ntasks = 5)
    runmpi(joinpath(@__DIR__, "mpi_test_connect_ell.jl"), ntasks = 2)
    runmpi(joinpath(@__DIR__, "mpi_test_connect_stacked.jl"), ntasks = 3)
    runmpi(joinpath(@__DIR__, "mpi_test_connect_stacked_3d.jl"), ntasks = 2)
    runmpi(joinpath(@__DIR__, "mpi_test_getpartition.jl"), ntasks = 3)
    runmpi(joinpath(@__DIR__, "mpi_test_partition.jl"), ntasks = 3)
    runmpi(joinpath(@__DIR__, "mpi_test_sortcolumns.jl"), ntasks = 1)
end
