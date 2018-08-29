using Canary
using Test

@testset "Linear Parition" begin
  @test Canary.linearpartition(1,1,1) == 1:1
  @test Canary.linearpartition(20,1,1) == 1:20
  @test Canary.linearpartition(10,1,2) == 1:5
  @test Canary.linearpartition(10,2,2) == 6:10
end

@testset "Mesh" begin
  let
    (etv, etc, fc) = brickmesh((2:5,4:6), (false,true))
    etv_expect = [ 1  2  5  6
                   2  3  6  7
                   3  4  7  8
                   5  6  9 10
                   6  7 10 11
                   7  8 11 12]'
    fc_expect = Array{Int64,1}[[4, 4, 1, 2],
                               [5, 4, 2, 3],
                               [6, 4, 3, 4]]

    @test etv == etv_expect
    @test fc == fc_expect
    @test etc[:,:,1] == [2 3 2 3
                         4 4 5 5]
    @test etc[:,:,5] == [3 4 3 4
                         5 5 6 6]
  end

  let
    (etv, etc, fc) = brickmesh((-1:2:1,-1:2:1,-1:1:1), (true,true,true))
    etv_expect = [1   5
                  2   6
                  3   7
                  4   8
                  5   9
                  6  10
                  7  11
                  8  12]

    fc_expect = Array{Int64,1}[[1, 2, 1, 3, 5,  7],
                               [1, 4, 1, 2, 5,  6],
                               [2, 2, 5, 7, 9, 11],
                               [2, 4, 5, 6, 9, 10],
                               [2, 6, 1, 2, 3,  4]]

    @test etv == etv_expect
    @test fc == fc_expect

    @test etc[:,:,1] == [-1  1 -1  1 -1  1 -1  1
                         -1 -1  1  1 -1 -1  1  1
                         -1 -1 -1 -1  0  0  0  0]

    @test etc[:,:,2] == [-1  1 -1  1 -1  1 -1  1
                         -1 -1  1  1 -1 -1  1  1
                          0  0  0  0  1  1  1  1]
  end

  let
    x = (-1:2:10,-1:1:1,-4:1:1)
    p = (true,false,true)

    (etv, etc, fc) = brickmesh(x,p)

    n = 50
    (etv_parts, etc_parts, fc_parts) = brickmesh(x,p, part=1, numparts=n)
    for j=2:n
      (etv_j, etc_j, fc_j) = brickmesh(x,p, part=j, numparts=n)
      etv_parts = cat(etv_parts, etv_j; dims=2)
      etc_parts = cat(etc_parts, etc_j; dims=3)
    end

    @test etv == etv_parts
    @test etc == etc_parts
  end
end
