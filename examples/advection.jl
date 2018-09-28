# # Advection Equation Example
#
# ## Introduction
#
# FILL ME!
#  What's the equation
#  What's the method
#
# ## Commented Program
#

# parameters:
N = 4
# brickN = (12,)
brickN = (1 * 12, 1 * 12)
# brickN = (10, 10, 10)

# First we load the MPI and Canary packages
using MPI
using Canary
using Printf: @sprintf

# Define computational type
DFloat = Float64

# number of dimensions
dim = length(brickN)

# We now initialize MPI as well as get the communicator, rank, and size
MPI.Initialized() || MPI.Init() # only initialize MPI if not initialized
MPI.finalize_atexit()
mpicomm = MPI.COMM_WORLD
mpirank = MPI.Comm_rank(mpicomm)
mpisize = MPI.Comm_size(mpicomm)

# Generate an local view of a fully periodic Cartesian mesh.
if dim == 1
  (Nx, ) = brickN
  local x = range(DFloat(0); length=Nx+1, stop=1)
  mesh = brickmesh((x, ), (true, ); part=mpirank+1, numparts=mpisize)
elseif dim == 2
  (Nx, Ny) = brickN
  local x = range(DFloat(0); length=Nx+1, stop=1)
  local y = range(DFloat(0); length=Ny+1, stop=1)
  mesh = brickmesh((x, y), (true, true); part=mpirank+1, numparts=mpisize)
else
  (Nx, Ny, Nz) = brickN
  local x = range(DFloat(0); length=Nx+1, stop=1)
  local y = range(DFloat(0); length=Ny+1, stop=1)
  local z = range(DFloat(0); length=Nz+1, stop=1)
  mesh = brickmesh((x, y, z), (true, true, true);
                   part=mpirank+1, numparts=mpisize)
end

# Partion the mesh using a Hilbert curve based partitioning
mesh = partition(mpicomm, mesh...)

# Connect the mesh in parallel
mesh = connectmesh(mpicomm, mesh...)

# Get the vmaps
(vmapM, vmapP) = mappings(N, mesh.elemtoelem, mesh.elemtoface, mesh.elemtoordr)

# Create 1-D operators
(ξ, ω) = lglpoints(DFloat, N)
D = spectralderivative(ξ)

# Compute metric terms
(nface, nelem) = size(mesh.elemtoelem)
coord = creategrid(Val(dim), mesh.elemtocoord, ξ)
if dim == 2
  (x, y) = (coord.x, coord.y)
  for j = 1:length(x)
    (x[j], y[j]) = (x[j] .+ sin.(π * x[j]) .* sin.(2 * π * y[j]) / 10,
                    y[j] .+ sin.(2 * π * x[j]) .* sin.(π * y[j]) / 10)
  end
elseif dim == 3
  (x, y, z) = (coord.x, coord.y, coord.z)
  for j = 1:length(x)
    (x[j], y[j], z[j]) = (x[j] + (sin(π * x[j]) * sin(2 * π * y[j]) *
                                  cos(2 * π * z[j])) / 10,
                          y[j] + (sin(π * y[j]) * sin(2 * π * x[j]) *
                                  cos(2 * π * z[j])) / 10,
                          z[j] + (sin(π * z[j]) * sin(2 * π * x[j]) *
                                  cos(2 * π * y[j])) / 10)
  end
end

# Dump the mesh
include("vtk.jl")
writemesh(@sprintf("Advection%dD_rank_%04d_mesh", dim, mpirank), coord...;
          realelems=mesh.realelems)

# Compute the metric terms
metric = computemetric(coord..., D)

# Get the distance from the center
if dim == 1
  statesyms = (:ρ, :Ux)
elseif dim == 2
  statesyms = (:ρ, :Ux, :Uy)
elseif dim == 3
  statesyms = (:ρ, :Ux, :Uy, :Uz)
end

# Create storage for state vector and right-hand side
Q   = NamedTuple{statesyms}(ntuple(j->zero(coord.x), length(statesyms)))
rhs = NamedTuple{statesyms}(ntuple(j->zero(coord.x), length(statesyms)))
if dim == 1
  Q.ρ .= sin.(2 * π * x)
  Q.Ux .= 1
elseif dim == 2
  Q.ρ .= sin.(2 * π * x) .* sin.(2 *  π * y)
  Q.Ux .= 1
  Q.Uy .= -1
elseif dim == 3
  Q.ρ .= sin.(2 * π * x) .* sin.(2 *  π * y) .* sin.(2 * π * y)
  Q.Ux .= 1
  Q.Uy .= -1
  Q.Uz .= 1
end

# set dt and number of steps
dt = [floatmax(DFloat)]
if dim == 1

elseif dim == 2
  (ξx, ξy, ηx, ηy) = (metric.ξx, metric.ξy, metric.ηx, metric.ηy)
  (Ux, Uy) = (Q.Ux, Q.Uy)
  for n = 1:length(Ux)
    loc_dt = 2 ./ max(abs.(Ux[n] * ξx[n] + Uy[n] * ξy[n]),
                      abs.(Ux[n] * ηx[n] + Uy[n] * ηy[n]))
    dt[1] = min(dt[1], loc_dt)
  end
elseif dim == 3
  (ξx, ξy, ξz) = (metric.ξx, metric.ξy, metric.ξz)
  (ηx, ηy, ηz) = (metric.ηx, metric.ηy, metric.ηz)
  (ζx, ζy, ζz) = (metric.ζx, metric.ζy, metric.ζz)
  (Ux, Uy, Uz) = (Q.Ux, Q.Uy, Q.Uz)
  for n = 1:length(Ux)
    loc_dt = 2 ./ max(abs.(Ux[n] * ξx[n] + Uy[n] * ξy[n] + Uz[n] * ξz[n]),
                      abs.(Ux[n] * ηx[n] + Uy[n] * ηy[n] + Uz[n] * ηz[n]),
                      abs.(Ux[n] * ζx[n] + Uy[n] * ζy[n] + Uz[n] * ζz[n]))
    dt[1] = min(dt[1], loc_dt)
  end
end
dt = MPI.Allreduce(dt[1], MPI.MIN, mpicomm)
dt = DFloat(dt / N^sqrt(2))
tend = DFloat(0.1)
nsteps = ceil(Int64, tend / dt)
dt = tend / nsteps
@show (dt, nsteps)

# Here we store the exact solution at the end time.  Later Δ will be used to
# store the difference between exact and the computed solution
Δ   = NamedTuple{statesyms}(ntuple(j->zero(coord.x), length(statesyms)))
if dim == 1
  Δ.ρ .= sin.(2 * π * (x - tend * Q.Ux))
  Δ.Ux .=  Q.Ux
elseif dim == 2
  Δ.ρ .= sin.(2 * π * (x - tend * Q.Ux)) .* sin.(2 *  π * (y - tend * Q.Uy))
  Δ.Ux .=  Q.Ux
  Δ.Uy .=  Q.Uy
elseif dim == 3
  Δ.ρ .= sin.(2 * π * (x - tend * Q.Ux)) .* sin.(2 *  π * (y - tend * Q.Uy)) .*
         sin.(2 * π * (z - tend * Q.Uz))
  Δ.Ux .=  Q.Ux
  Δ.Uy .=  Q.Uy
  Δ.Uz .=  Q.Uz
end

# Fourth-order, low-storage, Runge–Kutta scheme of Carpenter and Kennedy (1994)
# ((5,4) 2N-Storage RK scheme.
#
# Ref:
# @TECHREPORT{CarpenterKennedy1994,
#   author = {M.~H. Carpenter and C.~A. Kennedy},
#   title = {Fourth-order {2N-storage} {Runge-Kutta} schemes},
#   institution = {National Aeronautics and Space Administration},
#   year = {1994},
#   number = {NASA TM-109112},
#   address = {Langley Research Center, Hampton, VA},
# }
RKA = (DFloat(0),
       DFloat(-567301805773)  / DFloat(1357537059087),
       DFloat(-2404267990393) / DFloat(2016746695238),
       DFloat(-3550918686646) / DFloat(2091501179385),
       DFloat(-1275806237668) / DFloat(842570457699 ))

RKB = (DFloat(1432997174477) / DFloat(9575080441755 ),
       DFloat(5161836677717) / DFloat(13612068292357),
       DFloat(1720146321549) / DFloat(2090206949498 ),
       DFloat(3134564353537) / DFloat(4481467310338 ),
       DFloat(2277821191437) / DFloat(14882151754819))

RKC = (DFloat(0),
       DFloat(1432997174477) / DFloat(9575080441755),
       DFloat(2526269341429) / DFloat(6820363962896),
       DFloat(2006345519317) / DFloat(3224310063776),
       DFloat(2802321613138) / DFloat(2924317926251))


# Volume RHS for 2-D
function volumerhs!(rhs, Q::NamedTuple{S, NTuple{3, T}}, metric, D, ω,
                    elems) where {S, T}
  rhsρ = rhs.ρ
  (ρ, Ux, Uy) = (Q.ρ, Q.Ux, Q.Uy)
  Nq = size(ρ, 1)
  J = metric.J
  (ξx, ηx, ξy, ηy) = (metric.ξx, metric.ηx, metric.ξy, metric.ηy)
  # for each element
  for e ∈ elems
    # loop of ξ-grid lines
    for j = 1:Nq
      rhsρ[:, j, e] +=
        D' * (ω[j] * ω .* J[:, j, e].* ρ[:, j, e] .*
              (ξx[:, j, e] .* Ux[:, j, e] + ξy[:, j, e] .* Uy[:, j, e]))
    end
    # loop of η-grid lines
    for i = 1:Nq
      rhsρ[i, :, e] +=
        D' * (ω[i] * ω .* J[i, :, e].* ρ[i, :, e] .*
              (ηx[i, :, e] .* Ux[i, :, e] + ηy[i, :, e] .* Uy[i, :, e]))
    end
  end
end

# Volume RHS for 3-D
function volumerhs!(rhs, Q::NamedTuple{S, NTuple{4, T}}, metric, D, ω,
                    elems) where {S, T}
  rhsρ = rhs.ρ
  (ρ, Ux, Uy, Uz) = (Q.ρ, Q.Ux, Q.Uy, Q.Uz)
  Nq = size(ρ, 1)
  J = metric.J
  (ξx, ηx, ζx) = (metric.ξx, metric.ηx, metric.ζx)
  (ξy, ηy, ζy) = (metric.ξy, metric.ηy, metric.ζy)
  (ξz, ηz, ζz) = (metric.ξz, metric.ηz, metric.ζz)
  for e ∈ elems
    # loop of ξ-grid lines
    for k = 1:Nq
      for j = 1:Nq
        rhsρ[:, j, k, e] +=
          D' * (ω[j] * ω[k] * ω .* J[:, j, k, e] .* ρ[:, j, k, e] .*
                (ξx[:, j, k, e] .* Ux[:, j, k, e] +
                 ξy[:, j, k, e] .* Uy[:, j, k, e] +
                 ξz[:, j, k, e] .* Uz[:, j, k, e]))
      end
    end
    # loop of η-grid lines
    for k = 1:Nq
      for i = 1:Nq
        rhsρ[i, :, k, e] +=
          D' * (ω[i] * ω[k] * ω .* J[i, :, k, e] .* ρ[i, :, k, e] .*
                (ηx[i, :, k, e] .* Ux[i, :, k, e] +
                 ηy[i, :, k, e] .* Uy[i, :, k, e] +
                 ηz[i, :, k, e] .* Uz[i, :, k, e]))
      end
    end
    # loop of ζ-grid lines
    for j = 1:Nq
      for i = 1:Nq
        rhsρ[i, j, :, e] +=
          D' * (ω[i] * ω[j] * ω .* J[i, j, :, e] .* ρ[i, j, :, e] .*
                (ζx[i, j, :, e] .* Ux[i, j, :, e] +
                 ζy[i, j, :, e] .* Uy[i, j, :, e] +
                 ζz[i, j, :, e] .* Uz[i, j, :, e]))
      end
    end
  end
end

# Face RHS for 2-D
function facerhs!(rhs, Q::NamedTuple{S, NTuple{3, T}}, metric, ω, elems, vmapM,
                  vmapP) where {S, T}
  rhsρ = rhs.ρ
  (ρ, Ux, Uy) = (Q.ρ, Q.Ux, Q.Uy)
  nface = 4
  (nx, ny, sJ) = (metric.nx, metric.ny, metric.sJ)
  for e ∈ elems
    for f ∈ 1:nface
      ρM = ρ[vmapM[:, f, e]]
      UxM = Ux[vmapM[:, f, e]]
      UyM = Uy[vmapM[:, f, e]]
      FxM = ρM .* UxM
      FyM = ρM .* UyM

      ρP = ρ[vmapP[:, f, e]]
      UxP = Ux[vmapP[:, f, e]]
      UyP = Uy[vmapP[:, f, e]]
      FxP = ρP .* UxP
      FyP = ρP .* UyP

      nxM = nx[:, f, e]
      nyM = ny[:, f, e]
      λ = max.(abs.(nxM .* UxM + nyM .* UyM), abs.(nxM .* UxP + nyM .* UyP))

      F = (nxM .* (FxM + FxP) + nyM .* (FyM + FyP) + λ .* (ρM - ρP)) / 2
      rhsρ[vmapM[:, f, e]] -= ω .* sJ[:, f, e] .* F
    end
  end
end

# Face RHS for 2-D
function facerhs!(rhs, Q::NamedTuple{S, NTuple{4, T}}, metric, ω, elems, vmapM,
                  vmapP) where {S, T}
  rhsρ = rhs.ρ
  (ρ, Ux, Uy, Uz) = (Q.ρ, Q.Ux, Q.Uy, Q.Uz)
  nface = 6
  (nx, ny, nz, sJ) = (metric.nx, metric.ny, metric.nz, metric.sJ)
  for e ∈ elems
    for f ∈ 1:nface
      ρM = ρ[vmapM[:, f, e]]
      UxM = Ux[vmapM[:, f, e]]
      UyM = Uy[vmapM[:, f, e]]
      UzM = Uz[vmapM[:, f, e]]
      FxM = ρM .* UxM
      FyM = ρM .* UyM
      FzM = ρM .* UzM

      ρP = ρ[vmapP[:, f, e]]
      UxP = Ux[vmapP[:, f, e]]
      UyP = Uy[vmapP[:, f, e]]
      UzP = Uz[vmapP[:, f, e]]
      FxP = ρP .* UxP
      FyP = ρP .* UyP
      FzP = ρP .* UzP

      nxM = nx[:, f, e]
      nyM = ny[:, f, e]
      nzM = nz[:, f, e]
      λ = max.(abs.(nxM .* UxM + nyM .* UyM + nzM .* UzM),
               abs.(nxM .* UxP + nyM .* UyP + nzM .* UzP))

      F = (nxM .* (FxM + FxP) + nyM .* (FyM + FyP) + nzM .* (FzM + FzP) +
           λ .* (ρM - ρP)) / 2
      rhsρ[vmapM[:, f, e]] -= kron(ω, ω) .* sJ[:, f, e] .* F
    end
  end
end

# Update for 2-D
function updatesolution!(rhs, Q::NamedTuple{S, NTuple{3, T}}, metric, ω, elems,
                         rka, rkb, dt) where {S, T}
  J = metric.J
  M = reshape(kron(ω, ω), length(ω), length(ω))
  for (rhsq, q) ∈ zip(rhs, Q)
    for e ∈ elems
      q[:, :, e] += rkb * dt * rhsq[:, :, e] ./ (M .* J[:, :, e])
      rhsq[:, :, e] *= rka
    end
  end
end

# Update for 3-D
function updatesolution!(rhs, Q::NamedTuple{S, NTuple{4, T}}, metric, ω, elems,
                         rka, rkb, dt) where {S, T}
  J = metric.J
  M = reshape(kron(ω, ω, ω), length(ω), length(ω), length(ω))
  for (rhsq, q) ∈ zip(rhs, Q)
    for e ∈ elems
      q[:, :, :, e] += rkb * dt * rhsq[:, :, :, e] ./ (M .* J[:, :, :, e])
      rhsq[:, :, :, e] *= rka
    end
  end
end

# L2 Error: 1-D
function L2energy(Q::NamedTuple{S, NTuple{2, T}}, metric, ω, elems) where {S, T}
  J = metric.J
  Nq = length(ω)
  M = ω
  index = CartesianIndices(ntuple(j->1:Nq, Val(1)))

  energy = [zero(J[1])]
  for q ∈ Q
    for e ∈ elems
      for ind ∈ index
        energy[1] += M[ind] * J[ind, e] * q[ind, e]^2
      end
    end
  end
  energy[1]
end

# L2 Error: 2-D
function L2energy(Q::NamedTuple{S, NTuple{3, T}}, metric, ω, elems) where {S, T}
  J = metric.J
  Nq = length(ω)
  M = reshape(kron(ω, ω), Nq, Nq)
  index = CartesianIndices(ntuple(j->1:Nq, Val(2)))

  energy = [zero(J[1])]
  for q ∈ Q
    for e ∈ elems
      for ind ∈ index
        energy[1] += M[ind] * J[ind, e] * q[ind, e]^2
      end
    end
  end
  energy[1]
end

# L2 Error: 3-D
function L2energy(Q::NamedTuple{S, NTuple{4, T}}, metric, ω, elems) where {S, T}
  J = metric.J
  Nq = length(ω)
  M = reshape(kron(ω, ω, ω), Nq, Nq, Nq)
  index = CartesianIndices(ntuple(j->1:Nq, Val(3)))

  energy = [zero(J[1])]
  for q ∈ Q
    for e ∈ elems
      for ind ∈ index
        energy[1] += M[ind] * J[ind, e] * q[ind, e]^2
      end
    end
  end
  energy[1]
end

# How many MPI neighbors do we have?
numnabr = length(mesh.nabrtorank)

# Create send request array
sendreq = fill(MPI.REQUEST_NULL, numnabr)

# Create recv request array
recvreq = fill(MPI.REQUEST_NULL, numnabr)

# Create send buffer
sendQ = Array{DFloat, 3}(undef, (N+1)^dim, length(Q), length(mesh.sendelems))
recvQ = Array{DFloat, 3}(undef, (N+1)^dim, length(Q), length(mesh.ghostelems))

# Build CartesianIndex map for moving between Cartesian and linear storage of
# dofs
index = CartesianIndices(ntuple(j->1:N+1, dim))
nrealelem = length(mesh.realelems)

# Dump the initial condition
include("vtk.jl")
writemesh(@sprintf("Advection%dD_rank_%04d_step_%05d", dim, mpirank, 0),
          coord...; fields=(("ρ", Q.ρ),), realelems=mesh.realelems)

for step = 1:nsteps
  mpirank == 0 && @show step
  for s = 1:length(RKA)
    # post MPI receives
    for (nnabr, nabrrank, nabrelem) ∈ zip(1:numnabr, mesh.nabrtorank,
                                          mesh.nabrtorecv)
      recvreq[nnabr] = MPI.Irecv!((@view recvQ[:, :, nabrelem]), nabrrank, 777,
                                  mpicomm)
    end

    # wait on (prior) MPI sends
    MPI.Waitall!(sendreq)

    # pack data in send buffer
    for (ne, e) ∈ enumerate(mesh.sendelems)
      for (nf, f) ∈ enumerate(Q)
        sendQ[:, nf, ne] = f[index[:], e]
      end
    end

    # post MPI sends
    for (nnabr, nabrrank, nabrelem) ∈ zip(1:numnabr, mesh.nabrtorank,
                                          mesh.nabrtosend)
      sendreq[nnabr] = MPI.Isend((@view sendQ[:, :, nabrelem]), nabrrank, 777,
                                 mpicomm)
    end

    # volume RHS computation
    volumerhs!(rhs, Q, metric, D, ω, mesh.realelems)

    # wait on MPI receives
    MPI.Waitall!(recvreq)

    # copy data to state vectors
    for elems ∈ mesh.nabrtorecv
      for (nf, f) ∈ enumerate(Q)
        f[index[:], nrealelem .+ elems] = recvQ[:, nf, elems]
      end
    end

    # face RHS computation
    facerhs!(rhs, Q, metric, ω, mesh.realelems, vmapM, vmapP)

    # update solution and scale RHS
    updatesolution!(rhs, Q, metric, ω, mesh.realelems, RKA[s%length(RKA)+1],
                    RKB[s], dt)
  end

  writemesh(@sprintf("Advection%dD_rank_%04d_step_%05d", dim, mpirank, step),
            coord...; fields=(("ρ", Q.ρ),), realelems=mesh.realelems)
end

for (δ, q) ∈ zip(Δ, Q)
  δ .-= q
end
eng = L2energy(Q, metric, ω, mesh.realelems)
eng = MPI.Allreduce(eng, MPI.SUM, mpicomm)
mpirank == 0 && @show sqrt(eng)

err = L2energy(Δ, metric, ω, mesh.realelems)
err = MPI.Allreduce(err, MPI.SUM, mpicomm)
mpirank == 0 && @show sqrt(err)


nothing
