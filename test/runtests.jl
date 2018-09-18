using Canary
using Test
using MPI
using Logging

MPI.Init()

include("test_mesh.jl")
include("test_operators.jl")
include("test_metric.jl")

MPI.Finalize()

@testset "MPI Jobs" begin
  # The code below was modified from the MPI.jl file runtests.jl
  #
  # Code coverage command line options; must correspond to src/julia.h
  # and src/ui/repl.c
  JL_LOG_NONE = 0
  JL_LOG_USER = 1
  JL_LOG_ALL = 2
  coverage_opts = Dict{Int, String}(JL_LOG_NONE => "none",
                                    JL_LOG_USER => "user",
                                    JL_LOG_ALL => "all")
  coverage_opt = coverage_opts[Base.JLOptions().code_coverage]
  testdir = dirname(@__FILE__)

  for (n, f) in [(3, "mpi_test_centriod.jl")
                 (3, "mpi_test_getpartition.jl")
                 (5, "mpi_test_getpartition.jl")
                 (3, "mpi_test_partition.jl")
                 (1, "mpi_test_sortcolumns.jl")
                 (4, "mpi_test_sortcolumns.jl")]
    cmd =  `mpiexec -n $n $(Base.julia_cmd()) --startup-file=no --project=$(joinpath(testdir, "..")) --code-coverage=$coverage_opt $(joinpath(testdir, f))`
    @info "Running MPI test..." n f cmd
    @test (run(cmd); true)
  end
end
