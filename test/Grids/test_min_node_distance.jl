using Test
using MPI
using CUDA
using Canary.Grids
using Logging
using LinearAlgebra

let
    # boiler plate MPI stuff
    MPI.Initialized() || MPI.Init()
    
    if CUDA.has_cuda_gpu()
        ArrayType = CUDA.CuArray
    else
        ArrayType = Array
    end

    mpicomm = MPI.COMM_WORLD

    # Mesh generation parameters
    N = 4
    Nq = N + 1
    Neh = 10
    Nev = 4

    @testset "$(@__FILE__)" begin
        for FT in (Float64, Float32)
            for dim in (2, 3)
                if dim == 2
                    brickrange = (
                        range(FT(0); length = Neh + 1, stop = 1),
                        range(FT(1); length = Nev + 1, stop = 2),
                    )
                elseif dim == 3
                    brickrange = (
                        range(FT(0); length = Neh + 1, stop = 1),
                        range(FT(0); length = Neh + 1, stop = 1),
                        range(FT(1); length = Nev + 1, stop = 2),
                    )
                end

                topl = StackedBrickTopology(mpicomm, brickrange)

                function warpfun(ξ1, ξ2, ξ3)
                    FT = eltype(ξ1)

                    ξ1 ≥ FT(1 // 2) && (ξ1 = FT(1 // 2) + 2 * (ξ1 - FT(1 // 2)))
                    if dim == 2
                        ξ2 ≥ FT(3 // 2) &&
                        (ξ2 = FT(3 // 2) + 2 * (ξ2 - FT(3 // 2)))
                    elseif dim == 3
                        ξ2 ≥ FT(1 // 2) &&
                        (ξ2 = FT(1 // 2) + 2 * (ξ2 - FT(1 // 2)))
                        ξ3 ≥ FT(3 // 2) &&
                        (ξ3 = FT(3 // 2) + 2 * (ξ3 - FT(3 // 2)))
                    end
                    (ξ1, ξ2, ξ3)
                end

                grid = DiscontinuousSpectralElementGrid(
                    topl,
                    FloatType = FT,
                    DeviceArray = ArrayType,
                    polynomialorder = N,
                    meshwarp = warpfun,
                )

                ξ = referencepoints(grid)
                hmnd = (ξ[2] - ξ[1]) / (2Neh)
                vmnd = (ξ[2] - ξ[1]) / (2Nev)

                @test hmnd ≈ min_node_distance(grid, EveryDirection())
                @test vmnd ≈ min_node_distance(grid, VerticalDirection())
                @test hmnd ≈ min_node_distance(grid, HorizontalDirection())

            end
        end
    end
end
