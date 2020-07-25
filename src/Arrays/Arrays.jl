module Arrays

using CUDA
using DoubleFloats
using KernelAbstractions
using LazyArrays
using LinearAlgebra
using MPI
using StaticArrays
using StructArrays

include(joinpath(@__DIR__, "CMBuffers.jl"))
using .CMBuffers

end # module
