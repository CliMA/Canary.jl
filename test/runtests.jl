using Canary
using Test
using MPI

MPI.Init()
MPI.finalize_atexit()

include("test_mesh.jl")

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

  for (n, f) in [(3, "mpi_test_centriod.jl")]
    @test (run(`mpiexec -n $n $(Base.julia_cmd()) --startup-file=no --project=$(joinpath(testdir, "..")) --code-coverage=$coverage_opt $(joinpath(testdir, f))`); true)
  end
end