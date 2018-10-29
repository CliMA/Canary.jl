include(joinpath(@__DIR__,"vtk.jl"))
using MPI
using Canary
using Printf: @sprintf
const HAVE_CUDA = try
  using CUDAnative
  using CUDAdrv
  true
catch
  false
end
if HAVE_CUDA
  macro hascuda(ex)
    return :($(esc(ex)))
  end
else
  macro hascuda(ex)
    return :()
  end
end

# {{{ reshape for CuArray
@hascuda function Base.reshape(A::CuArray, dims::NTuple{N, Int}) where {N}
  @assert prod(dims) == prod(size(A))
  CuArray{eltype(A), length(dims)}(dims, A.buf)
end
# }}}

# {{{ constants
# note the order of the fields below is also assumed in the code.
const _nstate = 4
const _U, _V, _h, _b = 1:_nstate
const stateid = (U = _U, V = _V, h = _h, b = _b)

const _nvgeo = 8
const _ξx, _ηx, _ξy, _ηy, _MJ, _MJI, _x, _y, = 1:_nvgeo

const _nsgeo = 4
const _nx, _ny, _sMJ, _vMJI = 1:_nsgeo
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
      loc_dt = 2*H / max( abs(U*ξx + V*ξy) + H*sqrt(gravity*H)*δnl,
                          abs(U*ηx + V*ηy) + H*sqrt(gravity*H)*δnl )
      dt[1] = min(dt[1], loc_dt)
    end
  end

  MPI.Allreduce(dt[1], MPI.MIN, mpicomm)
end
# }}}

# {{{ compute geometry
function computegeometry(::Val{dim}, mesh, D, ξ, ω, meshwarp, vmapM) where dim
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

# {{{ CPU Kernels
# {{{ Volume RHS for 2D
function volumerhs!(::Val{2}, ::Val{N}, rhs::Array, Q, vgeo, D, elems, gravity, δnl) where N
  DFloat = eltype(Q)
  Nq = N + 1
  dim=2
  nelem = size(Q)[end]

  Q = reshape(Q, Nq, Nq, _nstate, nelem)
  rhs = reshape(rhs, Nq, Nq, _nstate, nelem)
  vgeo = reshape(vgeo, Nq, Nq, _nvgeo, nelem)

  #Allocate Arrays
  s_F = Array{DFloat}(undef, Nq, Nq, _nstate-1)
  s_G = Array{DFloat}(undef, Nq, Nq, _nstate-1)

  @inbounds for e in elems
      for j = 1:Nq, i = 1:Nq
          MJ = vgeo[i, j, _MJ, e]
          ξx, ξy = vgeo[i,j,_ξx,e], vgeo[i,j,_ξy,e]
          ηx, ηy = vgeo[i,j,_ηx,e], vgeo[i,j,_ηy,e]
          U, V = Q[i, j, _U, e], Q[i, j, _V, e]
          h, b = Q[i, j, _h, e], Q[i, j, _b, e]

          #Get primitive variables and fluxes
          H=h + b
          u=U/H
          v=V/H

          fluxh_x = U
          fluxh_y = V
          fluxU_x = (H * u * u + 0.5 * gravity * h^2) * δnl + gravity * h * b
          fluxU_y = (H * u * v) * δnl
          fluxV_x = (H * v * u) * δnl
          fluxV_y = (H * v * v + 0.5 * gravity * h^2) * δnl + gravity * h * b

          s_F[i, j, _h] = MJ * (ξx * fluxh_x + ξy * fluxh_y)
          s_F[i, j, _U] = MJ * (ξx * fluxU_x + ξy * fluxU_y)
          s_F[i, j, _V] = MJ * (ξx * fluxV_x + ξy * fluxV_y)

          s_G[i, j, _h] = MJ * (ηx * fluxh_x + ηy * fluxh_y)
          s_G[i, j, _U] = MJ * (ηx * fluxU_x + ηy * fluxU_y)
          s_G[i, j, _V] = MJ * (ηx * fluxV_x + ηy * fluxV_y)
      end

      # loop of ξ-grid lines
      for s = 1:_nstate-1, j = 1:Nq, i = 1:Nq, n = 1:Nq
          rhs[i, j, s, e] += D[n, i] * s_F[n, j, s]
      end
      # loop of η-grid lines
      for s = 1:_nstate-1, j = 1:Nq, i = 1:Nq, n = 1:Nq
          rhs[i, j, s, e] += D[n, j] * s_G[i, n, s]
      end
  end
end

# }}}

# Flux RHS for 2D
function fluxrhs!(::Val{2}, ::Val{N}, rhs::Array,  Q, sgeo, elems, vmapM, vmapP, elemtobndy, gravity, δnl) where N
    DFloat = eltype(Q)
    Np = (N+1)^2
    Nfp = N+1
    nface = 4

    @inbounds for e in elems
        for f = 1:nface
            for n = 1:Nfp
                nxM, nyM, sMJ = sgeo[_nx, n, f, e], sgeo[_ny, n, f, e], sgeo[_sMJ, n, f, e]
                idM, idP = vmapM[n, f, e], vmapP[n, f, e]

                eM, eP = e, ((idP - 1) ÷ Np) + 1
                vidM, vidP = ((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1

                hM = Q[vidM, _h, eM]
                bM = Q[vidM, _b, eM]
                UM = Q[vidM, _U, eM]
                VM = Q[vidM, _V, eM]

                bc = elemtobndy[f, e]
                hP = bP = UP = VP = zero(eltype(Q))
                if bc == 0
                    hP = Q[vidP, _h, eP]
                    bP = Q[vidP, _b, eM]
                    UP = Q[vidP, _U, eP]
                    VP = Q[vidP, _V, eP]
                elseif bc == 1
                    UnM = nxM * UM + nyM * VM
                    UP = UM - 2 * UnM * nxM
                    VP = VM - 2 * UnM * nyM
                    hP = hM
                    bP = bM
                else
                    error("Invalid boundary conditions $bc on face $f of element $e")
                end
                HM=hM + bM
                uM = UM / HM
                vM = VM / HM
                HP=hP + bP
                uP = UP / HP
                vP = VP / HP

                #Left Fluxes
                fluxhM_x = UM
                fluxhM_y = VM
                fluxUM_x = (HM * uM * uM + 0.5 * gravity * hM^2) * δnl +
                gravity * hM * bM
                fluxUM_y = HM * uM * vM * δnl
                fluxVM_x = HM * vM * uM * δnl
                fluxVM_y = (HM * vM * vM + 0.5 * gravity * hM^2) * δnl +
                gravity * hM * bM

                #Right Fluxes
                fluxhP_x = UP
                fluxhP_y = VP
                fluxUP_x = (HP * uP * uP + 0.5 * gravity * hP^2) * δnl +
                gravity * hP * bP
                fluxUP_y = HP * uP * vP * δnl
                fluxVP_x = HP * vP * uP * δnl
                fluxVP_y = (HP * vP * vP + 0.5 * gravity * hP^2) * δnl +
                gravity * hP * bP

                #Compute wave speed
                λM=( abs(nxM * uM + nyM * vM) + sqrt(gravity*HM) ) * δnl +
                ( sqrt(gravity*bM) ) * (1.0-δnl)
                λP=( abs(nxM * uP + nyM * vP) + sqrt(gravity*HP) ) * δnl +
                ( sqrt(gravity*bP) ) * (1.0-δnl)
                λ = max( λM, λP )

                #Compute Numerical Flux and Update
                fluxhS = (nxM * (fluxhM_x + fluxhP_x) + nyM * (fluxhM_y + fluxhP_y) +
                          - λ * (hP - hM)) / 2
                fluxUS = (nxM * (fluxUM_x + fluxUP_x) + nyM * (fluxUM_y + fluxUP_y) +
                          - λ * (UP - UM)) / 2
                fluxVS = (nxM * (fluxVM_x + fluxVP_x) + nyM * (fluxVM_y + fluxVP_y) +
                          - λ * (VP - VM)) / 2

                #Update RHS
                rhs[vidM, _h, eM] -= sMJ * fluxhS
                rhs[vidM, _U, eM] -= sMJ * fluxUS
                rhs[vidM, _V, eM] -= sMJ * fluxVS
            end
        end
    end
end
# }}}

# {{{ Update solution
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

# {{{ GPU kernels
# {{{ Volume RHS for 2D
@hascuda function knl_volumerhs!(::Val{2}, ::Val{N}, rhs, Q, vgeo, D, nelem, gravity, δnl) where N
    DFloat = eltype(D)
    Nq = N + 1

    #Point Thread to DOF and Block to element
    (i, j, k) = threadIdx()
    e = blockIdx().x

    #Allocate Arrays
    s_D = @cuStaticSharedMem(eltype(D), (Nq, Nq))
    s_F = @cuStaticSharedMem(eltype(Q), (Nq, Nq, _nstate))
    s_G = @cuStaticSharedMem(eltype(Q), (Nq, Nq, _nstate))

    rhsU = rhsV = rhsh = zero(eltype(rhs))
    if i <= Nq && j <= Nq && k == 1 && e <= nelem
        # Load derivative into shared memory
        if k == 1
            s_D[i, j] = D[i, j]
        end

        # Load values needed into registers
        MJ = vgeo[i, j, _MJ, e]
        ξx, ξy = vgeo[i, j, _ξx, e], vgeo[i, j, _ξy, e]
        ηx, ηy = vgeo[i, j, _ηx, e], vgeo[i, j, _ηy, e]
        U, V = Q[i, j, _U, e], Q[i, j, _V, e]
        h, b = Q[i, j, _h, e], Q[i, j, _b, e]
        rhsh, rhsU, rhsV = rhs[i, j, _h, e], rhs[i, j, _U, e], rhs[i, j, _V, e]

        #Get primitive variables and fluxes
        H=h + b
        u=U/H
        v=V/H

        fluxh_x = U
        fluxh_y = V
        fluxU_x = (H * u * u + 0.5 * gravity * h^2) * δnl + gravity * h * b
        fluxU_y = (H * u * v) * δnl
        fluxV_x = (H * v * u) * δnl
        fluxV_y = (H * v * v + 0.5 * gravity * h^2) * δnl + gravity * h * b

        s_F[i, j, _h] = MJ * (ξx * fluxh_x + ξy * fluxh_y)
        s_F[i, j, _U] = MJ * (ξx * fluxU_x + ξy * fluxU_y)
        s_F[i, j, _V] = MJ * (ξx * fluxV_x + ξy * fluxV_y)

        s_G[i, j, _h] = MJ * (ηx * fluxh_x + ηy * fluxh_y)
        s_G[i, j, _U] = MJ * (ηx * fluxU_x + ηy * fluxU_y)
        s_G[i, j, _V] = MJ * (ηx * fluxV_x + ηy * fluxV_y)
    end

    sync_threads()

    @inbounds if i <= Nq && j <= Nq && k == 1 && e <= nelem
        for n = 1:Nq

            #ξ-grid lines
            Dni = s_D[n, i]
            rhsh += Dni * s_F[n, j, _h]
            rhsU += Dni * s_F[n, j, _U]
            rhsV += Dni * s_F[n, j, _V]

            #η-grid lines
            Dnj = s_D[n, j]
            rhsh += Dnj * s_G[i, n, _h]
            rhsU += Dnj * s_G[i, n, _U]
            rhsV += Dnj * s_G[i, n, _V]
        end

        rhs[i, j, _U, e] = rhsU
        rhs[i, j, _V, e] = rhsV
        rhs[i, j, _h, e] = rhsh
    end
    nothing
end
# }}}

# {{{ Face RHS (all dimensions)
@hascuda function knl_fluxrhs!(::Val{dim}, ::Val{N}, rhs, Q, sgeo, nelem, vmapM, vmapP, elemtobndy, gravity, δnl) where {dim, N}
  DFloat = eltype(Q)
  Np = (N+1)^dim
  nface = 2*dim

  (i, j, k) = threadIdx()
  e = blockIdx().x

  Nq = N+1
  half = convert(eltype(Q), 0.5)

  @inbounds if i <= Nq && j <= Nq && k == 1 && e <= nelem
      n = i + (j-1) * Nq
      for lf = 1:2:nface
          for f = lf:lf+1
              nxM, nyM, sMJ = sgeo[_nx, n, f, e], sgeo[_ny, n, f, e], sgeo[_sMJ, n, f, e]
              (idM, idP) = (vmapM[n, f, e], vmapP[n, f, e])

              (eM, eP) = (e, ((idP - 1) ÷ Np) + 1)
              (vidM, vidP) = (((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1)

              hM = Q[vidM, _h, eM]
              bM = Q[vidM, _b, eM]
              UM = Q[vidM, _U, eM]
              VM = Q[vidM, _V, eM]

              bc = elemtobndy[f, e]
              hP = bP = UP = VP = zero(eltype(Q))
              if bc == 0
                  hP = Q[vidP, _h, eP]
                  bP = Q[vidP, _b, eM]
                  UP = Q[vidP, _U, eP]
                  VP = Q[vidP, _V, eP]
              elseif bc == 1
                  UnM = nxM * UM + nyM * VM
                  UP = UM - 2 * UnM * nxM
                  VP = VM - 2 * UnM * nyM
                  hP = hM
                  bP = bM
#              else
#                  error("Invalid boundary conditions $bc on face $f of element $e")
              end
              HM=hM + bM
              uM = UM / HM
              vM = VM / HM
              HP=hP + bP
              uP = UP / HP
              vP = VP / HP

              #Left Fluxes
              fluxhM_x = UM
              fluxhM_y = VM
              fluxUM_x = (HM * uM * uM + 0.5 * gravity * hM * hM) * δnl +
              gravity * hM * bM
              fluxUM_y = HM * uM * vM * δnl
              fluxVM_x = HM * vM * uM * δnl
              fluxVM_y = (HM * vM * vM + 0.5 * gravity * hM * hM) * δnl +
              gravity * hM * bM

              #Right Fluxes
              fluxhP_x = UP
              fluxhP_y = VP
              fluxUP_x = (HP * uP * uP + 0.5 * gravity * hP * hP) * δnl +
              gravity * hP * bP
              fluxUP_y = HP * uP * vP * δnl
              fluxVP_x = HP * vP * uP * δnl
              fluxVP_y = (HP * vP * vP + 0.5 * gravity * hP * hP) * δnl +
              gravity * hP * bP

              #Compute wave speed
              λM=( abs(nxM * uM + nyM * vM) + CUDAnative.sqrt(gravity*HM) ) * δnl +
              ( CUDAnative.sqrt(gravity*bM) ) * (1.0-δnl)
              λP=( abs(nxM * uP + nyM * vP) + CUDAnative.sqrt(gravity*HP) ) * δnl +
              ( CUDAnative.sqrt(gravity*bP) ) * (1.0-δnl)
              λ = max( λM, λP )

              #Compute Numerical Flux and Update
              fluxhS = (nxM * (fluxhM_x + fluxhP_x) + nyM * (fluxhM_y + fluxhP_y) +
                        - λ * (hP - hM)) / 2
              fluxUS = (nxM * (fluxUM_x + fluxUP_x) + nyM * (fluxUM_y + fluxUP_y) +
                        - λ * (UP - UM)) / 2
              fluxVS = (nxM * (fluxVM_x + fluxVP_x) + nyM * (fluxVM_y + fluxVP_y) +
                        - λ * (VP - VM)) / 2

              #Update RHS
              rhs[vidM, _h, eM] -= sMJ * fluxhS
              rhs[vidM, _U, eM] -= sMJ * fluxUS
              rhs[vidM, _V, eM] -= sMJ * fluxVS
          end
          sync_threads()
      end
  end
    nothing
end
# }}}

# {{{ Update solution (for all dimensions)
@hascuda function knl_updatesolution!(::Val{dim}, ::Val{N}, rhs, Q, vgeo, nelem, rka, rkb, dt, advection) where {dim, N}
  (i, j, k) = threadIdx()
  e = blockIdx().x

  Nq = N+1
  @inbounds if i <= Nq && j <= Nq && k <= Nq && e <= nelem
    n = i + (j-1) * Nq + (k-1) * Nq * Nq
    MJI = vgeo[n, _MJI, e]
    for s = 1:_nstate-1
      Q[n, s, e] += rkb * dt * rhs[n, s, e] * MJI
      rhs[n, s, e] *= rka
    end
  end
  nothing
end
# }}}

# {{{ Fill sendQ on device with Q (for all dimensions)
@hascuda function knl_fillsendQ!(::Val{dim}, ::Val{N}, sendQ, Q, sendelems) where {N, dim}
  Nq = N + 1
  (i, j, k) = threadIdx()
  e = blockIdx().x

  @inbounds if i <= Nq && j <= Nq && k <= Nq && e <= length(sendelems)
    n = i + (j-1) * Nq + (k-1) * Nq * Nq
    re = sendelems[e]
    for s = 1:_nstate
      sendQ[n, s, e] = Q[n, s, re]
    end
  end
  nothing
end
# }}}

# {{{ Fill Q on device with recvQ (for all dimensions)
@hascuda function knl_transferrecvQ!(::Val{dim}, ::Val{N}, Q, recvQ, nelem, nrealelem) where {N, dim}
  Nq = N + 1
  (i, j, k) = threadIdx()
  e = blockIdx().x

  @inbounds if i <= Nq && j <= Nq && k <= Nq && e <= nelem
    n = i + (j-1) * Nq + (k-1) * Nq * Nq
    for s = 1:_nstate
      Q[n, s, nrealelem + e] = recvQ[n, s, e]
    end
  end
  nothing
end
# }}}

# {{{ MPI Buffer handling
function fillsendQ!(::Val{dim}, ::Val{N}, sendQ, d_sendQ::Array, Q,
                    sendelems) where {dim, N}
  sendQ[:, :, :] .= Q[:, :, sendelems]
end

@hascuda function fillsendQ!(::Val{dim}, ::Val{N}, sendQ, d_sendQ::CuArray,
                             d_QL, d_sendelems) where {dim, N}
  nsendelem = length(d_sendelems)
  if nsendelem > 0
    @cuda(threads=ntuple(j->N+1, dim), blocks=nsendelem,
          knl_fillsendQ!(Val(dim), Val(N), d_sendQ, d_QL, d_sendelems))
    sendQ .= d_sendQ
  end
end

@hascuda function transferrecvQ!(::Val{dim}, ::Val{N}, d_recvQ::CuArray, recvQ,
                                 d_QL, nrealelem) where {dim, N}
  nrecvelem = size(recvQ)[end]
  if nrecvelem > 0
    d_recvQ .= recvQ
    @cuda(threads=ntuple(j->N+1, dim), blocks=nrecvelem,
          knl_transferrecvQ!(Val(dim), Val(N), d_QL, d_recvQ, nrecvelem,
                             nrealelem))
  end
end

function transferrecvQ!(::Val{dim}, ::Val{N}, d_recvQ::Array, recvQ, Q,
                        nrealelem) where {dim, N}
  Q[:, :, nrealelem+1:end] .= recvQ[:, :, :]
end
# }}}

# {{{ GPU kernel wrappers
@hascuda function volumerhs!(::Val{dim}, ::Val{N}, d_rhsC::CuArray, d_QC, d_vgeoC, d_D, elems, gravity, δnl) where {dim, N}
  nelem = length(elems)
  @cuda(threads=ntuple(j->N+1, dim), blocks=nelem,
        knl_volumerhs!(Val(dim), Val(N), d_rhsC, d_QC, d_vgeoC, d_D, nelem, gravity, δnl))
end

@hascuda function fluxrhs!(::Val{dim}, ::Val{N}, d_rhsL::CuArray, d_QL, d_sgeo, elems, d_vmapM, d_vmapP, d_elemtobndy, gravity, δnl) where {dim, N}
  nelem = length(elems)
  @cuda(threads=(ntuple(j->N+1, dim-1)..., 1), blocks=nelem,
        knl_fluxrhs!(Val(dim), Val(N), d_rhsL, d_QL, d_sgeo, nelem, d_vmapM, d_vmapP, d_elemtobndy, gravity, δnl))
end

@hascuda function updatesolution!(::Val{dim}, ::Val{N}, d_rhsL::CuArray, d_QL, d_vgeoL, elems, rka, rkb, dt, advection) where {dim, N}
  nelem = length(elems)
  @cuda(threads=ntuple(j->N+1, dim), blocks=nelem,
        knl_updatesolution!(Val(dim), Val(N), d_rhsL, d_QL, d_vgeoL, nelem, rka, rkb, dt, advection))
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
# }}}

# {{{ L2 Energy (for all dimensions)
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
                      dt, nsteps, tout, vmapM, vmapP, mpicomm,
                      gravity, δnl, advection;
                      ArrType=ArrType, plotstep=0) where {dim, N}
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
  nsendelem = length(mesh.sendelems)
  nrecvelem = length(mesh.ghostelems)
  nelem = length(mesh.elems)

  d_QL, d_rhsL = ArrType(Q), ArrType(rhs)
  d_vgeoL, d_sgeo = ArrType(vgeo), ArrType(sgeo)
  d_vmapM, d_vmapP = ArrType(vmapM), ArrType(vmapP)
  d_sendelems, d_elemtobndy = ArrType(mesh.sendelems), ArrType(mesh.elemtobndy)
  d_sendQ, d_recvQ = ArrType(sendQ), ArrType(recvQ)
  d_D = ArrType(D)

  Qshape    = (fill(N+1, dim)..., size(Q, 2), size(Q, 3))
  vgeoshape = (fill(N+1, dim)..., _nvgeo, size(Q, 3))

  d_QC = reshape(d_QL, Qshape)
  d_rhsC = reshape(d_rhsL, Qshape...)
  d_vgeoC = reshape(d_vgeoL, vgeoshape)

  start_time = t1 = time_ns()
  for step = 1:nsteps
    for s = 1:length(RKA)
      # post MPI receives
      for n = 1:nnabr
        recvreq[n] = MPI.Irecv!((@view recvQ[:, :, mesh.nabrtorecv[n]]),
                                mesh.nabrtorank[n], 777, mpicomm)
      end

      # wait on (prior) MPI sends
      MPI.Waitall!(sendreq)

      # pack data in send buffer
      fillsendQ!(Val(dim), Val(N), sendQ, d_sendQ, d_QL, d_sendelems)

      # post MPI sends
      for n = 1:nnabr
        sendreq[n] = MPI.Isend((@view sendQ[:, :, mesh.nabrtosend[n]]),
                               mesh.nabrtorank[n], 777, mpicomm)
      end

      # volume RHS computation
      volumerhs!(Val(dim), Val(N), d_rhsC, d_QC, d_vgeoC, d_D, mesh.realelems, gravity, δnl)

      # wait on MPI receives
      MPI.Waitall!(recvreq)

      # copy data to state vectors
      transferrecvQ!(Val(dim), Val(N), d_recvQ, recvQ, d_QL, nrealelem)

      # face RHS computation
      fluxrhs!(Val(dim), Val(N), d_rhsL, d_QL, d_sgeo, mesh.realelems, d_vmapM, d_vmapP, d_elemtobndy, gravity, δnl)

      # update solution and scale RHS
      updatesolution!(Val(dim), Val(N), d_rhsL, d_QL, d_vgeoL, mesh.realelems,
                      RKA[s%length(RKA)+1], RKB[s], dt, advection)
    end
    if step == 1
      @hascuda synchronize()
      start_time = time_ns()
    end
    if mpirank == 0 && (time_ns() - t1)*1e-9 > tout
      @hascuda synchronize()
      t1 = time_ns()
      avg_stage_time = (time_ns() - start_time) * 1e-9 /
                       ((step-1) * length(RKA))
      @show (step, nsteps, avg_stage_time)
    end
    # TODO: Fix VTK for 1-D
    if dim > 0 && plotstep > 0 && step % plotstep == 0
      Q .= d_QL
      X = ntuple(j->reshape((@view vgeo[:, _x+j-1, :]), ntuple(j->N+1,dim)...,
                            nelem), dim)
      h = reshape((@view Q[:, _h, :]), ntuple(j->(N+1),dim)..., nelem)
      b = reshape((@view Q[:, _b, :]), ntuple(j->(N+1),dim)..., nelem)
      U = reshape((@view Q[:, _U, :]), ntuple(j->(N+1),dim)..., nelem) ./ (h.+b)
      V = reshape((@view Q[:, _V, :]), ntuple(j->(N+1),dim)..., nelem) ./ (h.+b)

      writemesh(@sprintf("viz/swe%dD_%s_rank_%04d_step_%05d",
                         dim, ArrType, mpirank, step), X...;
                fields=(("h", h),("b",b),("U",U),("V",V),),
                realelems=mesh.realelems)
    end
  end

  Q .= d_QL
  rhs .= d_rhsL
end
# }}}

# {{{ SWE driver
function swe(::Val{dim}, ::Val{N}, mpicomm, ic, mesh, tend, gravity, δnl,
             advection; meshwarp=(x...)->identity(x), tout = 60, ArrType=Array,
             plotstep=0) where {dim, N}
  DFloat = typeof(tend)

  mpirank = MPI.Comm_rank(mpicomm)
  mpisize = MPI.Comm_size(mpicomm)

  # Partion the mesh using a Hilbert curve based partitioning
  mpirank == 0 && println("[CPU] partitioning mesh...")
  mesh = partition(mpicomm, mesh...)

  # Connect the mesh in parallel
  mpirank == 0 && println("[CPU] connecting mesh...")
  mesh = connectmesh(mpicomm, mesh...)

  # Get the vmaps
  mpirank == 0 && println("[CPU] computing mappings...")
  (vmapM, vmapP) = mappings(N, mesh.elemtoelem, mesh.elemtoface,
                            mesh.elemtoordr)

  # Create 1-D operators
  (ξ, ω) = lglpoints(DFloat, N)
  D = spectralderivative(ξ)

  # Compute the geometry
  mpirank == 0 && println("[CPU] computing metrics...")
  (vgeo, sgeo) = computegeometry(Val(dim), mesh, D, ξ, ω, meshwarp, vmapM)
  (nface, nelem) = size(mesh.elemtoelem)

  # Storage for the solution, rhs, and error
  mpirank == 0 && println("[CPU] creating fields (CPU)...")
  Q = zeros(DFloat, (N+1)^dim, _nstate, nelem)
  Qexact = zeros(DFloat, (N+1)^dim, _nstate, nelem)
  rhs = zeros(DFloat, (N+1)^dim, _nstate, nelem)

  # setup the initial condition
  mpirank == 0 && println("[CPU] computing initial conditions (CPU)...")
  @inbounds for e = 1:nelem, i = 1:(N+1)^dim
    x, y = vgeo[i, _x, e], vgeo[i, _y, e]
    h, b, U, V = ic(x, y)
    Q[i, _h, e] = h
    Q[i, _b, e] = b
    Q[i, _U, e] = U
    Q[i, _V, e] = V
    Qexact[i, _h, e] = h
    Qexact[i, _b, e] = b
    Qexact[i, _U, e] = U
    Qexact[i, _V, e] = V
  end

  # Compute time step
  mpirank == 0 && println("[CPU] computing dt (CPU)...")
  base_dt = cfl(Val(dim), Val(N), vgeo, Q, mpicomm, gravity, δnl) / N^√2
  #base_dt=0.025 #for case 20 with N=4 10x10x1
  base_dt=0.001
  mpirank == 0 && @show base_dt

  nsteps = ceil(Int64, tend / base_dt)
  dt = tend / nsteps
  mpirank == 0 && @show (dt, nsteps, dt * nsteps, tend)

  # Do time stepping
  stats = zeros(DFloat, 3)
  mpirank == 0 && println("[CPU] computing initial energy...")
  stats[1] = L2energysquared(Val(dim), Val(N), Q, vgeo, mesh.realelems)

  # plot the initial condition
  mkpath("viz")
  if dim > 0
    X = ntuple(j->reshape((@view vgeo[:, _x+j-1, :]), ntuple(j->N+1,dim)...,
                          nelem), dim)
    h = reshape((@view Q[:, _h, :]), ntuple(j->(N+1),dim)..., nelem)
    b = reshape((@view Q[:, _b, :]), ntuple(j->(N+1),dim)..., nelem)
    U = reshape((@view Q[:, _U, :]), ntuple(j->(N+1),dim)..., nelem) ./ (h.+b)
    V = reshape((@view Q[:, _V, :]), ntuple(j->(N+1),dim)..., nelem) ./ (h.+b)
    writemesh(@sprintf("viz/swe%dD_%s_rank_%04d_step_%05d",
                       dim, ArrType, mpirank, 0), X...;
              fields=(("h", h),("b",b),("U",U),("V",V),),
              realelems=mesh.realelems)
  end

  mpirank == 0 && println("[DEV] starting time stepper...")
  lowstorageRK(Val(dim), Val(N), mesh, vgeo, sgeo, Q, rhs, D, dt, nsteps, tout,
               vmapM, vmapP, mpicomm, gravity, δnl, advection;
               ArrType=ArrType, plotstep=plotstep)

  mpirank == 0 && println("[CPU] computing final energy...")
  stats[2] = L2energysquared(Val(dim), Val(N), Q, vgeo, mesh.realelems)
  stats[3] = L2errorsquared(Val(dim), Val(N), Q, vgeo, mesh.realelems, Qexact,
                            tend)

  stats = sqrt.(MPI.allreduce(stats, MPI.SUM, mpicomm))

  if  mpirank == 0
    @show eng0 = stats[1]
    @show engf = stats[2]
    @show Δeng = engf - eng0
    @show err = stats[3]
  end
end
# }}}

# {{{ main
function main()
  DFloat = Float64

  MPI.Initialized() ||MPI.Init()
  MPI.finalize_atexit()

  mpicomm = MPI.COMM_WORLD
  mpirank = MPI.Comm_rank(mpicomm)
  mpisize = MPI.Comm_size(mpicomm)

  # FIXME: query via hostname
  @hascuda device!(mpirank % length(devices()))

  #Input Parameters
  N=4
  Ne=10
  iplot=10
  δnl=1
  icase=10
  time_final=DFloat(0.32)
  dim=2
  hardware="cpu"

  #Initial Conditions
  ic = (x...) -> (0.0, 0.0, 0.0, 0.0)
  if icase == 1 #advection
    function ic_1(x...)
      r = sqrt( (x[1]-0.5)^2 + (x[2]-0.5)^2 )
      h = 0.5 * exp(-100.0 * r^2)
      b = 1.0
      H = h + b
      U = H*(1.0)
      V = H*(0.0)
      h, b, U, V
    end
    ic = ic_1
    periodic = (true, true)
    advection = true
    gravity = 0
  elseif icase == 10 #shallow water with Periodic BCs
    function ic10(x...)
      r = sqrt( (x[1]-0.5)^2 + (x[2]-0.5)^2 )
      h = 0.5 * exp(-100.0 * r^2)
      b=1.0
      H = h + b
      U = H*(0.0)
      V = H*(0.0)
      h, b, U, V
    end
    ic = ic10
    periodic = (true, true)
    advection = false
    gravity = 10
  elseif icase == 100 #shallow water with NFBC
    function ic100(x...)
      r = sqrt( (x[1]-0.5)^2 + (x[2]-0.5)^2 )
      h = 0.5 * exp(-100.0 * r^2)
      b=1.0
      H = h + b
      U = H*(0.0)
      V = H*(0.0)
      h, b, U, V
    end
    ic = ic100
    periodic = (false, false)
    advection = false
    gravity = 10
  end

  mesh = brickmesh((range(DFloat(0); length=Ne+1, stop=1),
                    range(DFloat(0); length=Ne+1, stop=1)),
                   periodic; part=mpirank+1, numparts=mpisize)

  if hardware == "cpu"
      mpirank == 0 && println("Running (CPU)...")
      swe(Val(2), Val(N), mpicomm, ic, mesh, time_final, gravity, δnl, advection;
          ArrType=Array, tout = 10, plotstep = iplot)
      mpirank == 0 && println()
  elseif hardware == "gpu"
      @hascuda begin
          mpirank == 0 && println("Running (GPU)...")
          swe(Val(2), Val(N), mpicomm, ic, mesh, time_final, gravity, δnl, advection;
              ArrType=CuArray, tout = 10, plotstep = iplot)
          mpirank == 0 && println()
      end
  end
  nothing
end
# }}}

main()
