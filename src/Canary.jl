module Canary

# We seem to be getting error messages from precompiling since depending on
# MPI.jl.  So we are going to turn it off for now.
#
# The long term solution may be to have an abstract MPI interface so that
# Canary doesn't have to depend directly on the MPI package.
__precompile__(false)

export brickmesh, centroidtocode, partition, connectmesh, mappings
export lglpoints, lgpoints
export barycentricweights, spectralderivative, interpolationmatrix
export creategrid!, computemetric!
export creategrid2d, creategrid3d, computemetric

include("mesh.jl")
include("operators.jl")
include("metric.jl")

end # module
