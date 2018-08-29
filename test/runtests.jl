using Canary
using Test

@testset "Linear Parition" begin
  @test Canary.linearpartition(1,1,1) == 1:1
  @test Canary.linearpartition(20,1,1) == 1:20
  @test Canary.linearpartition(10,1,2) == 1:5
  @test Canary.linearpartition(10,2,2) == 6:10
end
