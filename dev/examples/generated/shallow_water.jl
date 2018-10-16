```math
 \mathbf{f}=\left( \mathbf{U}, \frac{\mathbf{U} \otimes \mathbf{U}}{h} + g (h^2 - h^2_b) \mathbf{I}_2 \right).

const DFloat = Float64 #Number Type

N = 1 #polynomial order
#brickN = (10) #1D brickmesh
brickN = (100, 100) #2D brickmesh
tend = DFloat(0.005) #Final Time
δnl = 1.0 #switch to turn on/off nonlinear equations
gravity = 10.0 #gravity
advection=false #Boolean to turn on/off advection or swe

using MPI
using Canary
using Printf: @sprintf

dim = length(brickN)

println("N= ",N)
println("dim= ",dim)
println("brickN= ",brickN)
println("DFloat= ",DFloat)
println("δnl= ",δnl)
println("gravity= ",gravity)
println("advection= ",advection)

MPI.Initialized() || MPI.Init() # only initialize MPI if not initialized
MPI.finalize_atexit()
mpicomm = MPI.COMM_WORLD
mpirank = MPI.Comm_rank(mpicomm)
mpisize = MPI.Comm_size(mpicomm)

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
  mesh = brickmesh((x, y, z), (true, true, true); part=mpirank+1, numparts=mpisize)
end

mesh = partition(mpicomm, mesh...)

mesh = connectmesh(mpicomm, mesh...)

(vmapM, vmapP) = mappings(N, mesh.elemtoelem, mesh.elemtoface, mesh.elemtoordr)

(ξ, ω) = lglpoints(DFloat, N)
D = spectralderivative(ξ)

(nface, nelem) = size(mesh.elemtoelem)
coord = creategrid(Val(dim), mesh.elemtocoord, ξ)
if dim == 1
  x = coord.x
  for j = 1:length(x)
    x[j] = x[j]
 end
elseif dim == 2
  (x, y) = (coord.x, coord.y)
  for j = 1:length(x)
#=    (x[j], y[j]) = (x[j] .+ sin.(π * x[j]) .* sin.(2 * π * y[j]) / 10,
                    y[j] .+ sin.(2 * π * x[j]) .* sin.(π * y[j]) / 10)
=#
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

include(joinpath(@__DIR__, "vtk.jl"))
writemesh(@sprintf("SWE%dD_rank_%04d_mesh", dim, mpirank), coord...;
          realelems=mesh.realelems)

metric = computemetric(coord..., D)

if dim == 1
  statesyms = (:h, :U)
elseif dim == 2
  statesyms = (:h, :U, :V)
elseif dim == 3
  statesyms = (:h, :U, :V, :W)
end

Q   = NamedTuple{statesyms}(ntuple(j->zero(coord.x), length(statesyms)))
rhs = NamedTuple{statesyms}(ntuple(j->zero(coord.x), length(statesyms)))
if dim == 1
    bathymetry = zero(coord.x)
    for i=1:length(coord.x)
        bathymetry[i]=0.1
    end
    r=(x .- 0.5).^2
    Q.h .= 0.5 .* exp.(-32.0 .* r)
    Q.U .= 0
    if (advection)
        δnl=1.0
        gravity=0.0
        Q.U .= (Q.h+bathymetry) .* (1.0)
    end
    #=
  for i=1:length(coord.x)
     bathymetry[i]=2.0
  end
  Q.h .= sin.(2 * π * x) .+ 0.0
  Q.U .= (Q.h+bathymetry) .* (1.0)
=#
elseif dim == 2
    bathymetry = zero(coord.x)
    for i=1:length(coord.x)
        bathymetry[i]=0.2
    end
    r=(x .- 0.5).^2 + (y .- 0.5).^2
    Q.h .= 0.5 .* exp.(-100.0 .* r)
    Q.U .= 0
    Q.V .= 0
    if (advection)
        δnl=1.0
        gravity=0.0
        Q.U .= (Q.h+bathymetry) .* (1.0)
        Q.V .= (Q.h+bathymetry) .* (0.0)
    end
#=
    for i=1:length(coord.x)
     bathymetry[i]=2.0
  end
  r=(x .- 0.5).^2 + (y .- 0.5).^2
  Q.h .= sin.(2 * π * x) .* sin.(2 *  π * y)
  #Q.h .= 0.5 .* exp.(-8.0 .* r)
  Q.U .= (Q.h+bathymetry) .* (1.0)
  Q.V .= (Q.h+bathymetry) .* (1.0)
=#
elseif dim == 3
  Q.h .= sin.(2 * π * x) .* sin.(2 *  π * y) .* sin.(2 * π * z) .+ 2.0
  Q.U .= Q.h .* (1.0)
  Q.V .= Q.h .* (1.0)
  Q.W .= Q.h .* (1.0)
end

dt = [floatmax(DFloat)]
if dim == 1
    (ξx) = (metric.ξx)
    (h,U) = (Q.h+bathymetry,Q.U)
    for n = 1:length(U)
        loc_dt = (2h[n])  ./ (abs.(U[n] * ξx[n]))
        dt[1] = min(dt[1], loc_dt)
    end
elseif dim == 2
    (ξx, ξy) = (metric.ξx, metric.ξy)
    (ηx, ηy) = (metric.ηx, metric.ηy)
    (h,U,V) = (Q.h+bathymetry,Q.U,Q.V)
    for n = 1:length(U)
        loc_dt = (2h[n]) ./ max(abs.(U[n] * ξx[n] + V[n] * ξy[n]),
                          abs.(U[n] * ηx[n] + V[n] * ηy[n]))
        dt[1] = min(dt[1], loc_dt)
    end
elseif dim == 3
    (ξx, ξy, ξz) = (metric.ξx, metric.ξy, metric.ξz)
    (ηx, ηy, ηz) = (metric.ηx, metric.ηy, metric.ηz)
    (ζx, ζy, ζz) = (metric.ζx, metric.ζy, metric.ζz)
    (h,U,V,W) = (Q.h,Q.U,Q.V,Q.W)
    for n = 1:length(U)
        loc_dt = (2h[n]) ./ max(abs.(U[n] * ξx[n] + V[n] * ξy[n] + W[n] * ξz[n]),
                          abs.(U[n] * ηx[n] + V[n] * ηy[n] + W[n] * ηz[n]),
                          abs.(U[n] * ζx[n] + V[n] * ζy[n] + W[n] * ζz[n]))
        dt[1] = min(dt[1], loc_dt)
    end
end
dt = MPI.Allreduce(dt[1], MPI.MIN, mpicomm)
dt = DFloat(dt / N^sqrt(2))
dt = 0.0025
nsteps = ceil(Int64, tend / dt)
dt = tend / nsteps
@show (dt, nsteps)

Δ   = NamedTuple{statesyms}(ntuple(j->zero(coord.x), length(statesyms)))
if dim == 1
  Δ.h .= Q.h
  Δ.U .= Q.U
elseif dim == 2
  Δ.h .= Q.h
  Δ.U .= Q.U
  Δ.V .= Q.V
elseif dim == 3
  u = Q.U ./ Q.h
  v = Q.V ./ Q.h
  w = Q.W ./ Q.h
  Δ.h .= sin.(2 * π * (x - tend * u)) .* sin.(2 *  π * (y - tend * v)) .*
         sin.(2 * π * (z - tend * w)) .+ 2
  Δ.U .=  Q.U
  Δ.V .=  Q.V
  Δ.W .=  Q.W
end

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

function volumerhs!(rhs, Q::NamedTuple{S, NTuple{2, T}}, bathymetry, metric, D, ω, elems, gravity, δnl) where {S, T}
  (rhsh, rhsU) = (rhs.h, rhs.U)
  (h, U) = (Q.h, Q.U)
  Nq = size(h, 1)
  J = metric.J
  ξx = metric.ξx
  for e ∈ elems
      #Get primitive variables and fluxes
      hb=bathymetry[:,e]
      hs=h[:,e]
      ht=hs + hb
      u=U[:,e] ./ ht
      fluxh=U[:,e]
      fluxU=(ht .* u .* u + 0.5 .* gravity .* hs .^2) .* δnl + gravity .* hs .* hb

      rhsh[:,e] += D' * (ω .* J[:,e] .* (ξx[:,e] .* fluxh[:]))
      rhsU[:,e] += D' * (ω .* J[:,e] .* (ξx[:,e] .* fluxU[:])) #assuming dhb/dx=0: need to include it
  end #e ∈ elems
end #function volumerhs-1d

function volumerhs!(rhs, Q::NamedTuple{S, NTuple{3, T}}, bathymetry, metric, D, ω, elems, gravity, δnl) where {S, T}
    (rhsh, rhsU, rhsV) = (rhs.h, rhs.U, rhs.V)
    (h, U, V) = (Q.h, Q.U, Q.V)
    Nq = size(h, 1)
    J = metric.J
    dim=2
    (ξx, ξy) = (metric.ξx, metric.ξy)
    (ηx, ηy) = (metric.ηx, metric.ηy)
    fluxh=Array{DFloat,3}(undef,dim,Nq,Nq)
    fluxU=Array{DFloat,3}(undef,dim,Nq,Nq)
    fluxV=Array{DFloat,3}(undef,dim,Nq,Nq)
    for e ∈ elems
        #Get primitive variables and fluxes
        hb=bathymetry[:,:,e]
        hs=h[:,:,e]
        ht=hs + hb
        u=U[:,:,e] ./ ht
        v=V[:,:,e] ./ ht
        fluxh[1,:,:]=U[:,:,e]
        fluxh[2,:,:]=V[:,:,e]
        fluxU[1,:,:]=(ht .* u .* u + 0.5 .* gravity .* hs .^2) .* δnl + gravity .* hs .* hb
        fluxU[2,:,:]=(ht .* u .* v) .* δnl
        fluxV[1,:,:]=(ht .* v .* u) .* δnl
        fluxV[2,:,:]=(ht .* v .* v + 0.5 .* gravity .* hs .^2) .* δnl + gravity .* hs .* hb

        for j = 1:Nq
            rhsh[:,j,e] += D' * (ω[j] * ω .* J[:,j,e].* (ξx[:,j,e] .* fluxh[1,:,j] + ξy[:,j,e] .* fluxh[2,:,j]))
            rhsU[:,j,e] += D' * (ω[j] * ω .* J[:,j,e].* (ξx[:,j,e] .* fluxU[1,:,j] + ξy[:,j,e] .* fluxU[2,:,j]))
            rhsV[:,j,e] += D' * (ω[j] * ω .* J[:,j,e].* (ξx[:,j,e] .* fluxV[1,:,j] + ξy[:,j,e] .* fluxV[2,:,j]))
        end #j

        for i = 1:Nq
            rhsh[i,:,e] += D' * (ω[i] * ω .* J[i,:,e].* (ηx[i,:,e] .* fluxh[1,i,:] + ηy[i,:,e] .* fluxh[2,i,:]))
            rhsU[i,:,e] += D' * (ω[i] * ω .* J[i,:,e].* (ηx[i,:,e] .* fluxU[1,i,:] + ηy[i,:,e] .* fluxU[2,i,:]))
            rhsV[i,:,e] += D' * (ω[i] * ω .* J[i,:,e].* (ηx[i,:,e] .* fluxV[1,i,:] + ηy[i,:,e] .* fluxV[2,i,:]))
        end #i
    end #e ∈ elems
end #function volumerhs-2d

function fluxrhs!(rhs, Q::NamedTuple{S, NTuple{2, T}}, bathymetry, metric, ω, elems, vmapM, vmapP, gravity, N, δnl) where {S, T}

    (rhsh, rhsU) = (rhs.h, rhs.U)
    (h, U) = (Q.h, Q.U)
    nface = 2
    (nx, sJ) = (metric.nx, metric.sJ)
    nx = reshape(nx, size(vmapM))
    sJ = reshape(sJ, size(vmapM))

    for e ∈ elems
        for f ∈ 1:nface
            #Compute fluxes on M/Left/- side
            hsM = h[vmapM[1, f, e]]
            hbM=bathymetry[vmapM[1, f, e]]
            hM=hsM + hbM
            UM = U[vmapM[1, f, e]]
            uM = UM ./ hM
            fluxhM = UM
            fluxUM = ( hM .* uM .* uM + 0.5 .* gravity .* hsM .^2) .* δnl + gravity .* hsM .* hbM

            #Compute fluxes on P/Right/+ side
            hsP = h[vmapP[1, f, e]]
            hbP=bathymetry[vmapP[1, f, e]]
            hP=hsP + hbP
            UP = U[vmapP[1, f, e]]
            uP = UP ./ hP
            fluxhP = UP
            fluxUP = (hP .* uP .* uP + 0.5 .* gravity .* hsP .^2) .* δnl + gravity .* hsP .* hbP

            #Compute wave speed
            nxM = nx[1, f, e]
            λM=( abs.(nxM .* uM) + sqrt(gravity*hM) ) .* δnl + ( sqrt(gravity*hbM) ) .* (1.0-δnl)
            λP=( abs.(nxM .* uP) + sqrt(gravity*hP) ) .* δnl + ( sqrt(gravity*hbP) ) .* (1.0-δnl)
            λ = max.( λM, λP )

            #Compute Numerical Flux and Update
            fluxh_star = (nxM .* (fluxhM + fluxhP) - λ .* (hsP - hsM)) / 2
            fluxU_star = (nxM .* (fluxUM + fluxUP) - λ .* (UP - UM)) / 2
            rhsh[vmapM[1, f, e]] -= sJ[1, f, e] .* fluxh_star
            rhsU[vmapM[1, f, e]] -= sJ[1, f, e] .* fluxU_star
        end #for f ∈ 1:nface
    end #e ∈ elems
end #function fluxrhs-1d

function fluxrhs!(rhs, Q::NamedTuple{S, NTuple{3, T}}, bathymetry, metric, ω, elems, vmapM, vmapP, gravity, N, δnl) where {S, T}
    (rhsh, rhsU, rhsV) = (rhs.h, rhs.U, rhs.V)
    (h, U, V) = (Q.h, Q.U, Q.V)
    nface = 4
    Nq=N+1
    dim=2
    (nx, ny, sJ) = (metric.nx, metric.ny, metric.sJ)
    fluxhM=Array{DFloat,2}(undef,dim,Nq)
    fluxUM=Array{DFloat,2}(undef,dim,Nq)
    fluxVM=Array{DFloat,2}(undef,dim,Nq)
    fluxhP=Array{DFloat,2}(undef,dim,Nq)
    fluxUP=Array{DFloat,2}(undef,dim,Nq)
    fluxVP=Array{DFloat,2}(undef,dim,Nq)
    for e ∈ elems
        for f ∈ 1:nface
            #Compute fluxes on M/Left/- side
            hsM = h[vmapM[:, f, e]]
            hbM=bathymetry[vmapM[:, f, e]]
            hM=hsM + hbM
            UM = U[vmapM[:, f, e]]
            uM = UM ./ hM
            VM = V[vmapM[:, f, e]]
            vM = VM ./ hM
            fluxhM[1,:] = UM
            fluxhM[2,:] = VM
            fluxUM[1,:] = ( hM .* uM .* uM + 0.5 .* gravity .* hsM .^2) .* δnl + gravity .* hsM .* hbM
            fluxUM[2,:] = ( hM .* uM .* vM ) .* δnl
            fluxVM[1,:] = ( hM .* vM .* uM ) .* δnl
            fluxVM[2,:] = ( hM .* vM .* vM + 0.5 .* gravity .* hsM .^2) .* δnl + gravity .* hsM .* hbM

            #Compute fluxes on P/right/+ side
            hsP = h[vmapP[:, f, e]]
            hbP=bathymetry[vmapP[:, f, e]]
            hP=hsP + hbP
            UP = U[vmapP[:, f, e]]
            uP = UP ./ hP
            VP = V[vmapP[:, f, e]]
            vP = VP ./ hP
            fluxhP[1,:] = UP
            fluxhP[2,:] = VP
            fluxUP[1,:] = ( hP .* uP .* uP + 0.5 .* gravity .* hsP .^2) .* δnl + gravity .* hsP .* hbP
            fluxUP[2,:] = ( hP .* uP .* vP ) .* δnl
            fluxVP[1,:] = ( hP .* vP .* uP ) .* δnl
            fluxVP[2,:] = ( hP .* vP .* vP + 0.5 .* gravity .* hsP .^2) .* δnl + gravity .* hsP .* hbP

            #Compute wave speed
            nxM = nx[:, f, e]
            nyM = ny[:, f, e]
            λM=( abs.(nxM .* uM + nyM .* vM) + sqrt.(gravity*hM) ) .* δnl + ( sqrt.(gravity*hbM) ) .* (1.0-δnl)
            λP=( abs.(nxM .* uP + nyM .* vP) + sqrt.(gravity*hP) ) .* δnl + ( sqrt.(gravity*hbP) ) .* (1.0-δnl)
            λ = max.( λM, λP )

            #Compute Numerical Flux and Update
            fluxh_star = (nxM .* (fluxhM[1,:] + fluxhP[1,:]) + nyM .* (fluxhM[2,:] + fluxhP[2,:]) - λ .* (hsP - hsM)) / 2
            fluxU_star = (nxM .* (fluxUM[1,:] + fluxUP[1,:]) + nyM .* (fluxUM[2,:] + fluxUP[2,:]) - λ .* (UP - UM)) / 2
            fluxV_star = (nxM .* (fluxVM[1,:] + fluxVP[1,:]) + nyM .* (fluxVM[2,:] + fluxVP[2,:]) - λ .* (VP - VM)) / 2
            rhsh[vmapM[:, f, e]] -= ω .* sJ[:, f, e] .* fluxh_star
            rhsU[vmapM[:, f, e]] -= ω .* sJ[:, f, e] .* fluxU_star
            rhsV[vmapM[:, f, e]] -= ω .* sJ[:, f, e] .* fluxV_star
        end #f ∈ 1:nface
    end #e ∈ elems
end #function fluxrhs-2d

function updatesolution!(rhs, Q::NamedTuple{S, NTuple{2, T}}, bathymetry, metric, ω, elems, rka, rkb, dt, advection) where {S, T}
    #Save original velocity
    if advection
        h = Q.h + bathymetry
        u = Q.U ./ h
    end

    J = metric.J
    M =  ω
    for (rhsq, q) ∈ zip(rhs, Q)
        for e ∈ elems
            q[:, e] += rkb * dt * rhsq[:, e] ./ ( M .* J[:, e])
            rhsq[:, e] *= rka
        end
    end
    #Reset velocity
    if advection
        Q.U .= (Q.h+bathymetry) .* u
    end
end #function update-1d

function updatesolution!(rhs, Q::NamedTuple{S, NTuple{3, T}}, bathymetry, metric, ω, elems, rka, rkb, dt, advection) where {S, T}
    #Save original velocity
    if (advection)
        h = Q.h + bathymetry
        u = Q.U ./ h
        v = Q.V ./ h
    end

    J = metric.J
    M = reshape(kron(ω, ω), length(ω), length(ω))
    for (rhsq, q) ∈ zip(rhs, Q)
        for e ∈ elems
            q[:, :, e] += rkb * dt * rhsq[:, :, e] ./ (M .* J[:, :, e])
            rhsq[:, :, e] *= rka
        end
    end
    #Reset velocity
    if (advection)
        Q.U .= (Q.h+bathymetry) .* u
        Q.V .= (Q.h+bathymetry) .* v
    end
end #function update-2d

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
end #end function L2energy-1d

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
end #end function L2energy-2d

numnabr = length(mesh.nabrtorank)

sendreq = fill(MPI.REQUEST_NULL, numnabr)
recvreq = fill(MPI.REQUEST_NULL, numnabr)

sendQ = Array{DFloat, 3}(undef, (N+1)^dim, length(Q), length(mesh.sendelems))
recvQ = Array{DFloat, 3}(undef, (N+1)^dim, length(Q), length(mesh.ghostelems))

index = CartesianIndices(ntuple(j->1:N+1, dim))
nrealelem = length(mesh.realelems)

include(joinpath(@__DIR__, "vtk.jl"))
temp=Q.h + bathymetry
writemesh(@sprintf("SWE%dD_rank_%04d_step_%05d", dim, mpirank, 0),
          coord...; fields=(("hs+hb", temp),), realelems=mesh.realelems)

for step = 1:nsteps
    mpirank == 0 && @show step
    for s = 1:length(RKA)

        for (nnabr, nabrrank, nabrelem) ∈ zip(1:numnabr, mesh.nabrtorank,
                                              mesh.nabrtorecv)
            recvreq[nnabr] = MPI.Irecv!((@view recvQ[:, :, nabrelem]), nabrrank, 777,
                                        mpicomm)
        end

        MPI.Waitall!(sendreq)

        for (ne, e) ∈ enumerate(mesh.sendelems)
            for (nf, f) ∈ enumerate(Q)
                sendQ[:, nf, ne] = f[index[:], e]
            end
        end

        for (nnabr, nabrrank, nabrelem) ∈ zip(1:numnabr, mesh.nabrtorank,
                                              mesh.nabrtosend)
            sendreq[nnabr] = MPI.Isend((@view sendQ[:, :, nabrelem]), nabrrank, 777,
                                       mpicomm)
        end

        volumerhs!(rhs, Q, bathymetry, metric, D, ω, mesh.realelems, gravity, δnl)

        MPI.Waitall!(recvreq)

        for elems ∈ mesh.nabrtorecv
            for (nf, f) ∈ enumerate(Q)
                f[index[:], nrealelem .+ elems] = recvQ[:, nf, elems]
            end
        end

        fluxrhs!(rhs, Q, bathymetry, metric, ω, mesh.realelems, vmapM, vmapP, gravity, N, δnl)

        #call updatesolution
        updatesolution!(rhs, Q, bathymetry, metric, ω, mesh.realelems, RKA[s%length(RKA)+1], RKB[s], dt, advection)
    end #s-stages

    temp=Q.h + bathymetry
    writemesh(@sprintf("SWE%dD_rank_%04d_step_%05d", dim, mpirank, step),
              coord...; fields=(("hs+hb", temp),), realelems=mesh.realelems)
end #step

#extract velocity fields
if dim == 1
    Q.U .= Q.U ./ (Q.h+bathymetry)
    Δ.U .= Δ.U ./ (Δ.h+bathymetry)
    Q.h .= Q.h
    Δ.h .= Δ.h
elseif dim == 2
    Q.U .= Q.U ./ (Q.h+bathymetry)
    Δ.U .= Δ.U ./ (Δ.h+bathymetry)
    Q.V .= Q.V ./ (Q.h+bathymetry)
    Δ.V .= Δ.V ./ (Δ.h+bathymetry)
    Q.h .= Q.h
    Δ.h .= Δ.h
elseif dim == 3
    Q.U .= Q.U ./ Q.h
    Δ.U .= Δ.U ./ Δ.h
    Q.V .= Q.V ./ Q.h
    Δ.V .= Δ.V ./ Δ.h
    Q.W .= Q.W ./ Q.h
    Δ.W .= Δ.W ./ Δ.h
    Q.h .= Q.h
    Δ.h .= Δ.h
end

#Compute Norms
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

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

