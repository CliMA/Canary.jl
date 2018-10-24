include(joinpath(@__DIR__,"vtk.jl"))
using MPI
using Canary
using Printf: @sprintf

# {{{ constants
# note the order of the fields below is also assumed in the code.
const _nstate = 4
const _U, _V, _h, _b = 1:_nstate
const stateid = (U = _U, V = _V, h = _h, b = _b)

const _nvgeo = 8
const _ξx, _ηx, _ξy, _ηy, _MJ, _MJI,
       _x, _y, = 1:_nvgeo
const vgeoid = (ξx = _ξx, ηx = _ηx,
                ξy = _ξy, ηy = _ηy,
                MJ = _MJ, MJI = _MJI,
                x = _x,   y = _y)

const _nsgeo = 4
const _nx, _ny, _sMJ, _vMJI = 1:_nsgeo
const sgeoid = (nx = _nx, ny = _ny, sMJ = _sMJ, vMJI = _vMJI)
# }}}

# {{{ cfl
function cfl(::Val{dim}, ::Val{N}, vgeo, Q, mpicomm, gravity, δnl) where {dim, N}
  DFloat = eltype(Q)
  Np = (N+1)^dim
  (~, ~, nelem) = size(Q)
  dt = [floatmax(DFloat)]

  if dim == 1
    @inbounds for e = 1:nelem, n = 1:Np
      U, ξx = Q[n, _U, e], vgeo[n, _ξx, e]

      loc_dt = 2 / abs(U*ξx)
      dt[1] = min(dt[1], loc_dt)
    end
  end

  if dim == 2
    @inbounds for e = 1:nelem, n = 1:Np
      h, b, U, V = Q[n, _h, e],  Q[n, _b, e], Q[n, _U, e], Q[n, _V, e]
      ξx, ξy, ηx, ηy = vgeo[n, _ξx, e], vgeo[n, _ξy, e],
                       vgeo[n, _ηx, e], vgeo[n, _ηy, e]
      H = h+b
      loc_dt = 2*H / max( abs(U*ξx + V*ξy) + H*sqrt(gravity*H)*δnl, abs(U*ηx + V*ηy) + H*sqrt(gravity*H)*δnl )
      dt[1] = min(dt[1], loc_dt)
    end
  end

  MPI.Allreduce(dt[1], MPI.MIN, mpicomm)
end
# }}}

# {{{ compute geometry
function computegeometry(::Val{dim}, mesh, D, ξ, ω, meshwarp, vmapM, icase) where dim
  # Compute metric terms
  Nq = size(D, 1)
  DFloat = eltype(D)

  (nface, nelem) = size(mesh.elemtoelem)

  crd = creategrid(Val(dim), mesh.elemtocoord, ξ)

  vgeo = zeros(DFloat, Nq^dim, _nvgeo, nelem)
  sgeo = zeros(DFloat, _nsgeo, Nq^(dim-1), nface, nelem)

  (ξx, ηx, ξy, ηy, MJ, MJI, x, y) =
      ntuple(j->(@view vgeo[:, j, :]), _nvgeo)
  J = similar(x)
  (nx, ny, sMJ, vMJI) = ntuple(j->(@view sgeo[ j, :, :, :]), _nsgeo)
  sJ = similar(sMJ)

  X = ntuple(j->(@view vgeo[:, _x+j-1, :]), dim)
  creategrid!(X..., mesh.elemtocoord, ξ)

  @inbounds for j = 1:length(x)
    (x[j], y[j]) = meshwarp(x[j], y[j])
  end

  # Compute the metric terms
  if dim == 1
    computemetric!(x, J, ξx, sJ, nx, D)
  elseif dim == 2
    computemetric!(x, y, J, ξx, ηx, ξy, ηy, sJ, nx, ny, D)
  end

  M = kron(1, ntuple(j->ω, dim)...)
  MJ .= M .* J
  MJI .= 1 ./ MJ
  vMJI .= MJI[vmapM]

  sM = dim > 1 ? kron(1, ntuple(j->ω, dim-1)...) : one(DFloat)
  sMJ .= sM .* sJ

  (vgeo, sgeo)
end
# }}}

#=
# {{{ 1-D
# Volume RHS for 1-D
function volumerhs!(::Val{1}, ::Val{N}, rhs, Q, vgeo, D, elems) where N
  Nq = N + 1

  @inbounds for e in elems
    for i in 1:Nq
      for n in 1:Nq
        rhs[i, _hs, e] += D[n, i] * (vgeo[n, _MJ, e] * vgeo[n, _ξx, e] *
                                    Q[n, _U, e] * Q[n, _h, e])
      end
    end
  end
end

# Face RHS for 1-D
function fluxrhs!(::Val{1}, ::Val{N}, rhs, Q, sgeo, elems, vmapM,
                  vmapP) where N
  Np = N+1
  nface = 2

  @inbounds for e in elems
    for f = 1:nface
      (nxM, ~, sMJ, ~) = sgeo[:, 1, f, e]
      idM, idP = vmapM[1, f, e], vmapP[1, f, e]

      eM, eP = e, ((idP - 1) ÷ Np) + 1
      vidM, vidP = ((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1

      hM = Q[vidM, _h, eM]
      UM = Q[vidM, _U, eM]
      FxM = hM * UM

      hP = Q[vidP, _h, eP]
      UP = Q[vidP, _U, eP]
      FxP = hP * UP

      λ = max(abs(nxM * UM), abs(nxM * UP))

      F = (nxM * (FxM + FxP) + λ * (hM - hP)) / 2
      rhs[vidM, _h, eM] -= sMJ * F
    end
  end
end
# }}}
=#

# {{{ 2-D
# Volume RHS for 2-D
function volumerhs!(::Val{2}, ::Val{N}, rhs, Q, vgeo, D, elems, gravity, δnl) where N
    Nq = N + 1
    DFloat = eltype(Q)
    dim=2
    ~, ~, nelem = size(Q)

    Q = reshape(Q, Nq, Nq, _nstate, nelem)
    rhs = reshape(rhs, Nq, Nq, _nstate, nelem)
    vgeo = reshape(vgeo, Nq, Nq, _nvgeo, nelem)

    #Allocate local flux arrays
    fluxh=Array{DFloat,3}(undef,dim,Nq,Nq)
    fluxU=Array{DFloat,3}(undef,dim,Nq,Nq)
    fluxV=Array{DFloat,3}(undef,dim,Nq,Nq)

    @inbounds for e = 1:nelem

        #Metric Terms
        Jac=vgeo[:,:,_MJ,e]
        ξx=vgeo[:,:,_ξx,e]
        ξy=vgeo[:,:,_ξy,e]
        ηx=vgeo[:,:,_ηx,e]
        ηy=vgeo[:,:,_ηy,e]

        #Get primitive variables and fluxes
        h=Q[:,:,_h,e]
        b=Q[:,:,_b,e]
        H=h + b
        u=Q[:,:,_U,e] ./ H
        v=Q[:,:,_V,e] ./ H

        #Compute Fluxes
        fluxh[1,:,:]=Q[:,:,_U,e]
        fluxh[2,:,:]=Q[:,:,_V,e]
        fluxU[1,:,:]=(H .* u .* u + 0.5 .* gravity .* h.^2) .* δnl + gravity .* h .* b
        fluxU[2,:,:]=H .* u .* v .* δnl
        fluxV[1,:,:]=H .* v .* u .* δnl
        fluxV[2,:,:]=(H .* v .* v + 0.5 .* gravity .* h.^2) .* δnl + gravity .* h .* b

        # loop of ξ-grid lines
        for j = 1:Nq, i = 1:Nq, k = 1:Nq
            rhs[i,j,_h,e] += D[k, i] * Jac[k,j] * (ξx[k,j] * fluxh[1,k,j] + ξy[k,j] * fluxh[2,k,j])
            rhs[i,j,_U,e] += D[k, i] * Jac[k,j] * (ξx[k,j] * fluxU[1,k,j] + ξy[k,j] * fluxU[2,k,j])
            rhs[i,j,_V,e] += D[k, i] * Jac[k,j] * (ξx[k,j] * fluxV[1,k,j] + ξy[k,j] * fluxV[2,k,j])
        end
        # loop of η-grid lines
        for i = 1:Nq, j = 1:Nq, k = 1:Nq
            rhs[i,j,_h,e] += D[k, j] * Jac[i,k] * (ηx[i,k] * fluxh[1,i,k] + ηy[i,k] * fluxh[2,i,k])
            rhs[i,j,_U,e] += D[k, j] * Jac[i,k] * (ηx[i,k] * fluxU[1,i,k] + ηy[i,k] * fluxU[2,i,k])
            rhs[i,j,_V,e] += D[k, j] * Jac[i,k] * (ηx[i,k] * fluxV[1,i,k] + ηy[i,k] * fluxV[2,i,k])
        end
    end
end

# Face RHS for 2-D
function fluxrhs!(::Val{2}, ::Val{N}, rhs, Q, sgeo, elems, boundary, vmapM, vmapP, gravity, δnl) where N

    DFloat = eltype(Q)
    Np = (N+1)^2
    Nfp = N+1
    nface = 4
    (~, ~, nelem) = size(Q)
    dim=2

    #Allocate local flux arrays
    fluxhM=Array{DFloat,1}(undef,dim)
    fluxUM=Array{DFloat,1}(undef,dim)
    fluxVM=Array{DFloat,1}(undef,dim)
    fluxhP=Array{DFloat,1}(undef,dim)
    fluxUP=Array{DFloat,1}(undef,dim)
    fluxVP=Array{DFloat,1}(undef,dim)

    @inbounds for e = 1:nelem
        for f = 1:nface

            #Check Boundary Condition
            bc=boundary[f,e]

            for n = 1:Nfp
                (nxM, nyM, sMJ, ~) = sgeo[:, n, f, e]
                idM, idP = vmapM[n, f, e], vmapP[n, f, e]

                eM, eP = e, ((idP - 1) ÷ Np) + 1
                vidM, vidP = ((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1

                #Compute Quantities on Left side
                hM = Q[vidM, _h, eM]
                bM = Q[vidM, _b, eM]
                HM = hM + bM
                UM = Q[vidM, _U, eM]
                VM = Q[vidM, _V, eM]
                uM = UM / HM
                vM = VM / HM

                #Left Fluxes
                fluxhM[1] = UM
                fluxhM[2] = VM
                fluxUM[1] = (HM * uM * uM + 0.5 * gravity * hM^2) * δnl + gravity * hM * bM
                fluxUM[2] = HM * uM * vM * δnl
                fluxVM[1] = HM * vM * uM * δnl
                fluxVM[2] = (HM * vM * vM + 0.5 * gravity * hM^2) * δnl + gravity * hM * bM

                #Compute Quantities on Right side
                hP = Q[vidP, _h, eP]
                bP = Q[vidP, _b, eP]
                HP = hP + bP
                UP = Q[vidP, _U, eP]
                VP = Q[vidP, _V, eP]

                if bc == 0 #no boundary or periodic
                    #do nothing
                elseif bc == 1 #No-flux
                    Unormal=nxM * UM + nyM * VM
                    UP = UM - 2 * Unormal * nxM
                    VP = VM - 2 * Unormal * nyM
                    hP = hM
                    HP = HM
                end
                uP = UP / HP
                vP = VP / HP

                #Right Fluxes
                fluxhP[1] = UP
                fluxhP[2] = VP
                fluxUP[1] = (HP * uP * uP + 0.5 * gravity * hP^2) * δnl + gravity * hP * bP
                fluxUP[2] = HP * uP * vP * δnl
                fluxVP[1] = HP * vP * uP * δnl
                fluxVP[2] = (HP * vP * vP + 0.5 * gravity * hP^2) * δnl + gravity * hP * bP

                #Compute wave speed
                λM=( abs(nxM * uM + nyM * vM) + sqrt(gravity*HM) ) * δnl + ( sqrt(gravity*bM) ) * (1.0-δnl)
                λP=( abs(nxM * uP + nyM * vP) + sqrt(gravity*HP) ) * δnl + ( sqrt(gravity*bP) ) * (1.0-δnl)
                λ = max( λM, λP )

                #Compute Numerical Flux and Update
                fluxh_star = (nxM * (fluxhM[1] + fluxhP[1]) + nyM * (fluxhM[2] + fluxhP[2]) - λ * (hP - hM)) / 2
                fluxU_star = (nxM * (fluxUM[1] + fluxUP[1]) + nyM * (fluxUM[2] + fluxUP[2]) - λ * (UP - UM)) / 2
                fluxV_star = (nxM * (fluxVM[1] + fluxVP[1]) + nyM * (fluxVM[2] + fluxVP[2]) - λ * (VP - VM)) / 2

                #Update RH
                rhs[vidM, _h, eM] -= sMJ * fluxh_star
                rhs[vidM, _U, eM] -= sMJ * fluxU_star
                rhs[vidM, _V, eM] -= sMJ * fluxV_star
            end
        end
    end
end
# }}}

# {{{ Update solution (for all dimensions)
function updatesolution!(::Val{dim}, ::Val{N}, rhs, Q, vgeo, elems, rka, rkb, dt, advection) where {dim, N}

    DFloat = eltype(Q)
    Nq=(N+1)^dim
    (~, ~, nelem) = size(Q)

    #Store Velocity
    if (advection)

        #Allocate local arrays
        u=Array{DFloat,2}(undef,Nq,nelem)
        v=Array{DFloat,2}(undef,Nq,nelem)

        @inbounds for e = elems, i = 1:Nq
            u[i,e] = Q[i,_U,e] / ( Q[i,_h,e] + Q[i,_b,e] )
            v[i,e] = Q[i,_V,e] / ( Q[i,_h,e] + Q[i,_b,e] )
        end
    end

    @inbounds for e = elems, s = 1:_nstate, i = 1:Nq
        Q[i, s, e] += rkb * dt * rhs[i, s, e] * vgeo[i, _MJI, e]
        rhs[i, s, e] *= rka
    end

    #Reset Velocity
    if (advection)
        @inbounds for e = elems, i = 1:Nq
            Q[i,_U,e] = ( Q[i,_h,e] + Q[i,_b,e] ) * u[i,e]
            Q[i,_V,e] = ( Q[i,_h,e] + Q[i,_b,e] ) * v[i,e]
        end
    end
end

# }}}

# {{{ L2 Error (for all dimensions)
function L2errorsquared(::Val{dim}, ::Val{N}, Q, vgeo, elems, Qex, t) where {dim, N}
  DFloat = eltype(Q)
  Np = (N+1)^dim
  (~, nstate, nelem) = size(Q)

  err = zero(DFloat)

  @inbounds for e = elems, i = 1:Np
    X = ntuple(j -> vgeo[i, _x-1+j, e] - Q[i, _U-1+j, e]*t, Val(dim))
    diff = Q[i, _h, e] - Qex[i, _h, e]

    err += vgeo[i, _MJ, e] * diff^2
  end

  err
end

function L2energysquared(::Val{dim}, ::Val{N}, Q, vgeo, elems) where {dim, N}
  DFloat = eltype(Q)
  Np = (N+1)^dim
  (~, nstate, nelem) = size(Q)

  energy = zero(DFloat)

  @inbounds for e = elems, q = 1:nstate, i = 1:Np
    energy += vgeo[i, _MJ, e] * Q[i, q, e]^2
  end

  energy
end
# }}}

# {{{ RK loop
function lowstorageRK(::Val{dim}, ::Val{N}, mesh, vgeo, sgeo, Q, rhs, D,
                      dt, nsteps, tout, vmapM, vmapP, mpicomm, iplot, icase, gravity, δnl, advection) where {dim, N}
  DFloat = eltype(Q)
  mpirank = MPI.Comm_rank(mpicomm)

  # Fourth-order, low-storage, Runge–Kutta scheme of Carpenter and Kennedy
  # (1994) ((5,4) 2N-Storage RK scheme.
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

  # Create send and recv request array
  nnabr = length(mesh.nabrtorank)
  sendreq = fill(MPI.REQUEST_NULL, nnabr)
  recvreq = fill(MPI.REQUEST_NULL, nnabr)

  # Create send and recv buffer
  sendQ = zeros(DFloat, (N+1)^dim, size(Q,2), length(mesh.sendelems))
  recvQ = zeros(DFloat, (N+1)^dim, size(Q,2), length(mesh.ghostelems))

    nrealelem = length(mesh.realelems)

    t1 = time_ns()
    time=0
    for step = 1:nsteps
        time=time + dt
        if mpirank == 0 && (time_ns() - t1)*1e-9 > tout
            t1 = time_ns()
            @show (step, nsteps)
        end
        for s = 1:length(RKA)
            # post MPI receives
            for n = 1:nnabr
                recvreq[n] = MPI.Irecv!((@view recvQ[:, :, mesh.nabrtorecv[n]]),
                                        mesh.nabrtorank[n], 777, mpicomm)
            end

            # wait on (prior) MPI sends
            MPI.Waitall!(sendreq)

            # pack data in send buffer
            sendQ[:, :, :] .= Q[:, :, mesh.sendelems]

            # post MPI sends
            for n = 1:nnabr
                sendreq[n] = MPI.Isend((@view sendQ[:, :, mesh.nabrtosend[n]]),
                                       mesh.nabrtorank[n], 777, mpicomm)
            end

            # volume RHS computation
            volumerhs!(Val(dim), Val(N), rhs, Q, vgeo, D, mesh.realelems, gravity, δnl)

            # wait on MPI receives
            MPI.Waitall!(recvreq)

            # copy data to state vectors
            Q[:, :, nrealelem+1:end] .= recvQ[:, :, :]

            # face RHS computation
            fluxrhs!(Val(dim), Val(N), rhs, Q, sgeo, mesh.realelems, mesh.elemtobndy, vmapM, vmapP, gravity, δnl)

            # update solution and scale RHS
            updatesolution!(Val(dim), Val(N), rhs, Q, vgeo, mesh.realelems, RKA[s%length(RKA)+1], RKB[s], dt, advection)
        end #s-stages

        #Plot VTK
        if mpirank == 0 && mod(step,iplot) == 0
            println("step=",step," time=",time)
        end
        if dim > 1 && mod(step,iplot) == 0
            (nface, nelem) = size(mesh.elemtoelem)
            X = ntuple(j->reshape((@view vgeo[:, _x+j-1, :]), ntuple(j->N+1,dim)...,
                                  nelem), dim)
            h = reshape((@view Q[:, _h, :]), ntuple(j->(N+1),dim)..., nelem)
            b = reshape((@view Q[:, _b, :]), ntuple(j->(N+1),dim)..., nelem)
            U = reshape((@view Q[:, _U, :]), ntuple(j->(N+1),dim)..., nelem) ./ (h+b)
            V = reshape((@view Q[:, _V, :]), ntuple(j->(N+1),dim)..., nelem) ./ (h+b)
            writemesh(@sprintf("viz/swe%dD_rank_%04d_step_%05d", dim,
                               mpirank, step), X...; fields=(("h", h),("b",b),("U",U),("V",V),), realelems=mesh.realelems)
        end

    end #dt
end #end lowStorageRK
# }}}

# {{{ Swe driver
function swe(mpicomm, ic, ::Val{N}, brickN::NTuple{dim, Int}, tend, iplot, icase, gravity, δnl, advection;
                   meshwarp=(x...)->identity(x), tout = 60) where {N, dim}
  DFloat = Float64
  mpirank = MPI.Comm_rank(mpicomm)
  mpisize = MPI.Comm_size(mpicomm)

  # Generate a local view of a fully periodic Cartesian mesh.
  mesh = brickmesh(ntuple(i->range(DFloat(0); length=brickN[i]+1, stop=1), dim),
                   (fill(true, dim)...,);
                   part=mpirank+1, numparts=mpisize)

  if (icase == 100)
      mesh = brickmesh(ntuple(i->range(DFloat(0); length=brickN[i]+1, stop=1), dim),
                       (fill(false, dim)...,);
                       part=mpirank+1, numparts=mpisize)
  end

  # Partion the mesh using a Hilbert curve based partitioning
  mesh = partition(mpicomm, mesh...)

  # Connect the mesh in parallel
  mesh = connectmesh(mpicomm, mesh...)

  # Get the vmaps
  (vmapM, vmapP) = mappings(N, mesh.elemtoelem, mesh.elemtoface,
                            mesh.elemtoordr)

  # Create 1-D operators
  (ξ, ω) = lglpoints(DFloat, N)
  D = spectralderivative(ξ)

  # Compute the geometry
  (vgeo, sgeo) = computegeometry(Val(dim), mesh, D, ξ, ω, meshwarp, vmapM, icase)
  (nface, nelem) = size(mesh.elemtoelem)

  # Storage for the solution, rhs, and error
  Qexact = zeros(DFloat, (N+1)^dim, _nstate, nelem)
  Q = zeros(DFloat, (N+1)^dim, _nstate, nelem)
  rhs = zeros(DFloat, (N+1)^dim, _nstate, nelem)

  # setup the initial condition
  @inbounds for e = 1:nelem, i = 1:(N+1)^dim
      x, y = vgeo[i, _x, e], vgeo[i, _y, e]
      h, b, U, V = ic(icase,dim,x, y)
      Q[i, _h, e] = h
      Q[i, _b, e] = b
      Q[i, _U, e] = U
      Q[i, _V, e] = V
      Qexact[i, _h, e] = h
      Qexact[i, _b, e] = b
      Qexact[i, _U, e] = U
      Qexact[i, _V, e] = V
  end

  # plot the initial condition
  mkpath("viz")
  # TODO: Fix VTK for 1-D
  if dim > 1
    X = ntuple(j->reshape((@view vgeo[:, _x+j-1, :]), ntuple(j->N+1,dim)...,
                          nelem), dim)
    h = reshape((@view Q[:, _h, :]), ntuple(j->(N+1),dim)..., nelem)
    b = reshape((@view Q[:, _b, :]), ntuple(j->(N+1),dim)..., nelem)
    U = reshape((@view Q[:, _U, :]), ntuple(j->(N+1),dim)..., nelem) ./ (h+b)
    V = reshape((@view Q[:, _V, :]), ntuple(j->(N+1),dim)..., nelem) ./ (h+b)
    writemesh(@sprintf("viz/swe%dD_rank_%04d_step_%05d", dim,
                       mpirank, 0), X...; fields=(("h", h),("b",b),("U",U),("V",V),), realelems=mesh.realelems)
  end

  # Compute time step
  dt = cfl(Val(dim), Val(N), vgeo, Q, mpicomm, gravity, δnl) / N^√2
  #dt=0.025 #for case 20 with N=4 10x10x1
  dt=0.001

  nsteps = ceil(Int64, tend / dt)
  dt = tend / nsteps
  mpirank == 0 && @show (dt, nsteps)

  # Do time stepping
  stats = zeros(DFloat, 3)
  stats[1] = L2energysquared(Val(dim), Val(N), Q, vgeo, mesh.realelems)

  lowstorageRK(Val(dim), Val(N), mesh, vgeo, sgeo, Q, rhs, D, dt, nsteps,
               tout, vmapM, vmapP, mpicomm, iplot, icase, gravity, δnl, advection)

  # plot the final Solution
  # TODO: Fix VTK for 1-D
  if dim > 1
    X = ntuple(j->reshape((@view vgeo[:, _x+j-1, :]), ntuple(j->N+1,dim)...,
                          nelem), dim)
    h = reshape((@view Q[:, _h, :]), ntuple(j->(N+1),dim)..., nelem)
    b = reshape((@view Q[:, _b, :]), ntuple(j->(N+1),dim)..., nelem)
    U = reshape((@view Q[:, _U, :]), ntuple(j->(N+1),dim)..., nelem) ./ (h+b)
    V = reshape((@view Q[:, _V, :]), ntuple(j->(N+1),dim)..., nelem) ./ (h+b)
    writemesh(@sprintf("viz/swe%dD_rank_%04d_step_%05d", dim,
                       mpirank, nsteps), X...; fields=(("h", h),("b",b),("U",U),("V",V),), realelems=mesh.realelems)
  end

  stats[2] = L2energysquared(Val(dim), Val(N), Q, vgeo, mesh.realelems)
  stats[3] = L2errorsquared(Val(dim), Val(N), Q, vgeo, mesh.realelems, Qexact, tend)

  stats = sqrt.(MPI.allreduce(stats, MPI.SUM, mpicomm))

  if mpirank == 0
    @show eng = stats[1]
    @show engdiff = stats[2] - stats[1]
    @show err = stats[3]
  end
end
# }}}

# {{{ main
function main()
  MPI.Initialized() || MPI.Init() # only initialize MPI if not initialized
  MPI.finalize_atexit()
#  MPI.Init()

  mpicomm = MPI.COMM_WORLD
  mpirank = MPI.Comm_rank(mpicomm)
  mpisize = MPI.Comm_size(mpicomm)

#  warping1D(x...) = (x[1] +  sin( π*x[1])/10, zero(x[1]), zero(x[1]))
#  warping2D(x...) = (x[1] +  sin( π*x[1])*sin(2π*x[2])/10,
#                     x[2] +  sin(2π*x[1])*sin( π*x[2])/10,
#                     zero(x[1]))
  warping1D(x...) = (x[1], zero(x[1]), zero(x[1]))
  warping2D(x...) = (x[1], x[2], zero(x[1]))

  #Input Parameters
    N=4
    Ne=10
    iplot=10
    dim=2
    gravity=10.0
    δnl=1
    advection=false
    icase=100
    time_final=0.2

    #For Advection only
    if advection
        gravity=0
        δnl=1
    end

  #Initial Conditions
  function ic(icase,dim,x...)
      if icase == 1 #advection
          r = sqrt( (x[1]-0.5)^2 + (x[2]-0.5)^2 )
          h = 0.5 * exp(-100.0 * r^2)
          b = 1.0
          H = h + b
          U = H*(1.0)
          V = H*(0.0)
          h, b, U, V
      elseif icase == 10 #shallow water with Periodic BCs
          r = sqrt( (x[1]-0.5)^2 + (x[2]-0.5)^2 )
          h = 0.5 * exp(-100.0 * r^2)
          b=1.0
          H = h + b
          U = H*(0.0)
          V = H*(0.0)
          h, b, U, V
      elseif icase == 100 #shallow water with NFBC
          r = sqrt( (x[1]-0.5)^2 + (x[2]-0.5)^2 )
          h = 0.5 * exp(-100.0 * r^2)
          b=1.0
          H = h + b
          U = H*(0.0)
          V = H*(0.0)
          h, b, U, V
      end
  end

    #call swe
    if dim == 2
        mpirank == 0 && println("Running 2d...")
        swe(mpicomm, ic, Val(N), (Ne, Ne), time_final, iplot, icase, gravity, δnl, advection; meshwarp=warping2D)
        mpirank == 0 && println()
    end
#  MPI.Finalize()
end
# }}}

main()


