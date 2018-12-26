# To think about:
# - How to handle parameters for different case? Dictionaries?

# FIXME: Be consistent with tuple assignments (either with or without parens)
#
# FIXME: Add logging

# FIXME: Add link to https://github.com/paranumal/libparanumal here and in
# advection (also update the license)

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
const _U, _V, _ρ, _E = 1:_nstate
const stateid = (U = _U, V = _V, ρ = _ρ, E = _E)

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

const _γ = 14  // 10
const _p0 = 100000
const _R_gas = 28717 // 100
const _c_p = 100467 // 100
const _c_v = 7175 // 10
const _gravity = 10
# }}}

# {{{ cfl
function cfl(::Val{dim}, ::Val{N}, vgeo, Q, mpicomm) where {dim, N}
  DFloat = eltype(Q)
  γ::DFloat       = _γ
  p0::DFloat      = _p0
  R_gas::DFloat   = _R_gas
  c_p::DFloat     = _c_p
  c_v::DFloat     = _c_v
  gravity::DFloat = _gravity

  Np = (N+1)^dim
  (~, ~, nelem) = size(Q)

  dt = [floatmax(DFloat)]

  @inbounds for e = 1:nelem, n = 1:Np
    ρ, U, V = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e]
    E = Q[n, _E, e]
    P = p0 * (R_gas * E / p0)^(c_p / c_v)
    ξx, ξy, ηx, ηy = vgeo[n, _ξx, e], vgeo[n, _ξy, e],
                     vgeo[n, _ηx, e], vgeo[n, _ηy, e]

    loc_dt = 2ρ / max(abs(U * ξx + V * ξy) + ρ * sqrt(γ * P / ρ),
                      abs(U * ηx + V * ηy) + ρ * sqrt(γ * P / ρ))
    dt[1] = min(dt[1], loc_dt)
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
  computemetric!(x, y, J, ξx, ηx, ξy, ηy, sJ, nx, ny, D)

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
# {{{ 2-D
# Volume RHS for 2-D
function volumerhs!(::Val{2}, ::Val{N}, rhs::Array, Q, vgeo, D, elems) where N
  DFloat = eltype(Q)
  γ::DFloat       = _γ
  p0::DFloat      = _p0
  R_gas::DFloat   = _R_gas
  c_p::DFloat     = _c_p
  c_v::DFloat     = _c_v
  gravity::DFloat = _gravity

  Nq = N + 1

  nelem = size(Q)[end]

  Q = reshape(Q, Nq, Nq, _nstate, nelem)
  rhs = reshape(rhs, Nq, Nq, _nstate, nelem)
  vgeo = reshape(vgeo, Nq, Nq, _nvgeo, nelem)

  s_F = Array{DFloat}(undef, Nq, Nq, _nstate)
  s_G = Array{DFloat}(undef, Nq, Nq, _nstate)

  @inbounds for e in elems
    for j = 1:Nq, i = 1:Nq
      MJ = vgeo[i, j, _MJ, e]
      ξx, ξy = vgeo[i,j,_ξx,e], vgeo[i,j,_ξy,e]
      ηx, ηy = vgeo[i,j,_ηx,e], vgeo[i,j,_ηy,e]

      U, V = Q[i, j, _U, e], Q[i, j, _V, e]
      ρ, E = Q[i, j, _ρ, e], Q[i, j, _E, e]

      P = p0 * (R_gas * E / p0)^(c_p / c_v)

      ρinv = 1 / ρ
      fluxρ_x = U
      fluxU_x = ρinv * U * U + P
      fluxV_x = ρinv * V * U
      fluxE_x = ρinv * U * E

      fluxρ_y = V
      fluxU_y = ρinv * U * V
      fluxV_y = ρinv * V * V + P
      fluxE_y = ρinv * V * E

      s_F[i, j, _ρ] = MJ * (ξx * fluxρ_x + ξy * fluxρ_y)
      s_F[i, j, _U] = MJ * (ξx * fluxU_x + ξy * fluxU_y)
      s_F[i, j, _V] = MJ * (ξx * fluxV_x + ξy * fluxV_y)
      s_F[i, j, _E] = MJ * (ξx * fluxE_x + ξy * fluxE_y)

      s_G[i, j, _ρ] = MJ * (ηx * fluxρ_x + ηy * fluxρ_y)
      s_G[i, j, _U] = MJ * (ηx * fluxU_x + ηy * fluxU_y)
      s_G[i, j, _V] = MJ * (ηx * fluxV_x + ηy * fluxV_y)
      s_G[i, j, _E] = MJ * (ηx * fluxE_x + ηy * fluxE_y)

      # buoyancy term
      rhs[i, j, _V, e] -= MJ * ρ * gravity
    end

    # loop of ξ-grid lines
    for s = 1:_nstate, j = 1:Nq, i = 1:Nq, k = 1:Nq
      rhs[i, j, s, e] += D[k, i] * s_F[k, j, s]
    end
    # loop of η-grid lines
    for s = 1:_nstate, j = 1:Nq, i = 1:Nq, k = 1:Nq
      rhs[i, j, s, e] += D[k, j] * s_G[i, k, s]
    end
  end
end

# Face RHS for 2-D
function facerhs!(::Val{2}, ::Val{N}, rhs::Array, Q, sgeo, elems, vmapM,
                  vmapP, elemtobndy) where N
  DFloat = eltype(Q)
  γ::DFloat       = _γ
  p0::DFloat      = _p0
  R_gas::DFloat   = _R_gas
  c_p::DFloat     = _c_p
  c_v::DFloat     = _c_v
  gravity::DFloat = _gravity

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

        ρM = Q[vidM, _ρ, eM]
        UM = Q[vidM, _U, eM]
        VM = Q[vidM, _V, eM]
        EM = Q[vidM, _E, eM]

        bc = elemtobndy[f, e]
        PM = p0 * (R_gas * EM / p0)^(c_p / c_v)
        if bc == 0
          ρP = Q[vidP, _ρ, eP]
          UP = Q[vidP, _U, eP]
          VP = Q[vidP, _V, eP]
          EP = Q[vidP, _E, eP]
          PP = p0 * (R_gas * EP / p0)^(c_p / c_v)
        elseif bc == 1
          UnM = nxM * UM + nyM * VM
          UP = UM - 2 * UnM * nxM
          VP = VM - 2 * UnM * nyM
          ρP = ρM
          EP = EM
          PP = PM
        else
          error("Invalid boundary conditions $bc on face $f of element $e")
        end

        ρMinv = 1 / ρM
        fluxρM_x = UM
        fluxUM_x = ρMinv * UM * UM + PM
        fluxVM_x = ρMinv * VM * UM
        fluxEM_x = ρMinv * UM * EM

        fluxρM_y = VM
        fluxUM_y = ρMinv * UM * VM
        fluxVM_y = ρMinv * VM * VM + PM
        fluxEM_y = ρMinv * VM * EM

        ρPinv = 1 / ρP
        fluxρP_x = UP
        fluxUP_x = ρPinv * UP * UP + PP
        fluxVP_x = ρPinv * VP * UP
        fluxEP_x = ρPinv * UP * EP

        fluxρP_y = VP
        fluxUP_y = ρPinv * UP * VP
        fluxVP_y = ρPinv * VP * VP + PP
        fluxEP_y = ρPinv * VP * EP

        λM = ρMinv * abs(nxM * UM + nyM * VM) + sqrt(ρMinv * γ * PM)
        λP = ρPinv * abs(nxM * UP + nyM * VP) + sqrt(ρPinv * γ * PP)
        λ  =  max(λM, λP)

        #Compute Numerical Flux and Update
        fluxρS = (nxM * (fluxρM_x + fluxρP_x) + nyM * (fluxρM_y + fluxρP_y) +
                  - λ * (ρP - ρM)) / 2
        fluxUS = (nxM * (fluxUM_x + fluxUP_x) + nyM * (fluxUM_y + fluxUP_y) +
                  - λ * (UP - UM)) / 2
        fluxVS = (nxM * (fluxVM_x + fluxVP_x) + nyM * (fluxVM_y + fluxVP_y) +
                  - λ * (VP - VM)) / 2
        fluxES = (nxM * (fluxEM_x + fluxEP_x) + nyM * (fluxEM_y + fluxEP_y) +
                  - λ * (EP - EM)) / 2


        #Update RHS
        rhs[vidM, _ρ, eM] -= sMJ * fluxρS
        rhs[vidM, _U, eM] -= sMJ * fluxUS
        rhs[vidM, _V, eM] -= sMJ * fluxVS
        rhs[vidM, _E, eM] -= sMJ * fluxES
      end
    end
  end
end
# }}}

# {{{ Update solution (for all dimensions)
function updatesolution!(::Val{dim}, ::Val{N}, rhs::Array, Q, vgeo, elems, rka,
                         rkb, dt) where {dim, N}
  @inbounds for e = elems, s = 1:_nstate, i = 1:(N+1)^dim
    Q[i, s, e] += rkb * dt * rhs[i, s, e] * vgeo[i, _MJI, e]
    rhs[i, s, e] *= rka
  end
end

# }}}
# }}}

# {{{ improved GPU kernles

# {{{ Volume RHS for 2-D
@hascuda function knl_volumerhs!(::Val{2}, ::Val{N}, rhs, Q, vgeo, D, nelem) where N
  DFloat = eltype(D)
  γ::DFloat       = _γ
  p0::DFloat      = _p0
  R_gas::DFloat   = _R_gas
  c_p::DFloat     = _c_p
  c_v::DFloat     = _c_v
  gravity::DFloat = _gravity

  Nq = N + 1

  (i, j, k) = threadIdx()
  e = blockIdx().x

  s_D = @cuStaticSharedMem(eltype(D), (Nq, Nq))
  s_F = @cuStaticSharedMem(eltype(Q), (Nq, Nq, _nstate))
  s_G = @cuStaticSharedMem(eltype(Q), (Nq, Nq, _nstate))

  rhsU = rhsV = rhsρ = rhsE = zero(eltype(rhs))
  if i <= Nq && j <= Nq && k == 1 && e <= nelem
    # Load derivative into shared memory
    if k == 1
      s_D[i, j] = D[i, j]
    end

    # Load values will need into registers
    MJ = vgeo[i, j, _MJ, e]
    ξx, ξy = vgeo[i, j, _ξx, e], vgeo[i, j, _ξy, e]
    ηx, ηy = vgeo[i, j, _ηx, e], vgeo[i, j, _ηy, e]
    U, V = Q[i, j, _U, e], Q[i, j, _V, e]
    ρ, E = Q[i, j, _ρ, e], Q[i, j, _E, e]
    rhsU, rhsV = rhs[i, j, _U, e], rhs[i, j, _V, e]
    rhsρ, rhsE = rhs[i, j, _ρ, e], rhs[i, j, _E, e]

    P = p0 * CUDAnative.pow(R_gas * E / p0, c_p / c_v)

    ρinv = 1 / ρ
    fluxρ_x = U
    fluxU_x = ρinv * U * U + P
    fluxV_x = ρinv * V * U
    fluxE_x = ρinv * U * E

    fluxρ_y = V
    fluxU_y = ρinv * U * V
    fluxV_y = ρinv * V * V + P
    fluxE_y = ρinv * V * E

    s_F[i, j, _ρ] = MJ * (ξx * fluxρ_x + ξy * fluxρ_y)
    s_F[i, j, _U] = MJ * (ξx * fluxU_x + ξy * fluxU_y)
    s_F[i, j, _V] = MJ * (ξx * fluxV_x + ξy * fluxV_y)
    s_F[i, j, _E] = MJ * (ξx * fluxE_x + ξy * fluxE_y)

    s_G[i, j, _ρ] = MJ * (ηx * fluxρ_x + ηy * fluxρ_y)
    s_G[i, j, _U] = MJ * (ηx * fluxU_x + ηy * fluxU_y)
    s_G[i, j, _V] = MJ * (ηx * fluxV_x + ηy * fluxV_y)
    s_G[i, j, _E] = MJ * (ηx * fluxE_x + ηy * fluxE_y)

    # buoyancy term
    rhsV -= MJ * ρ * gravity
  end

  sync_threads()

  @inbounds if i <= Nq && j <= Nq && k == 1 && e <= nelem
    # loop of ξ-grid lines
    for n = 1:Nq
      Dni = s_D[n, i]
      Dnj = s_D[n, j]

      rhsρ += Dni * s_F[n, j, _ρ]
      rhsρ += Dnj * s_G[i, n, _ρ]

      rhsU += Dni * s_F[n, j, _U]
      rhsU += Dnj * s_G[i, n, _U]

      rhsV += Dni * s_F[n, j, _V]
      rhsV += Dnj * s_G[i, n, _V]

      rhsE += Dni * s_F[n, j, _E]
      rhsE += Dnj * s_G[i, n, _E]
    end

    rhs[i, j, _U, e] = rhsU
    rhs[i, j, _V, e] = rhsV
    rhs[i, j, _ρ, e] = rhsρ
    rhs[i, j, _E, e] = rhsE
  end
  nothing
end
# }}}

# {{{ Face RHS (all dimensions)
@hascuda function knl_facerhs!(::Val{dim}, ::Val{N}, rhs, Q, sgeo, nelem, vmapM,
                               vmapP, elemtobndy) where {dim, N}
  DFloat = eltype(Q)
  γ::DFloat       = _γ
  p0::DFloat      = _p0
  R_gas::DFloat   = _R_gas
  c_p::DFloat     = _c_p
  c_v::DFloat     = _c_v
  gravity::DFloat = _gravity

  Np = (N+1) * (N+1)
  nface = 4

  (i, j, k) = threadIdx()
  e = blockIdx().x

  Nq = N+1
  half = convert(eltype(Q), 0.5)

  @inbounds if i <= Nq && j <= Nq && k == 1 && e <= nelem
    n = i + (j-1) * Nq
    for lf = 1:2:nface
      for f = lf:lf+1
        (nxM, nyM, sMJ) = (sgeo[_nx, n, f, e], sgeo[_ny, n, f, e], sgeo[_sMJ, n, f, e])

        (idM, idP) = (vmapM[n, f, e], vmapP[n, f, e])

        (eM, eP) = (e, ((idP - 1) ÷ Np) + 1)
        (vidM, vidP) = (((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1)

        ρM = Q[vidM, _ρ, eM]
        UM = Q[vidM, _U, eM]
        VM = Q[vidM, _V, eM]
        EM = Q[vidM, _E, eM]

        bc = elemtobndy[f, e]
        PM = p0 * CUDAnative.pow(R_gas * EM / p0, c_p / c_v)
        ρP = UP = VP = EP = PP = zero(eltype(Q))
        if bc == 0
          ρP = Q[vidP, _ρ, eP]
          UP = Q[vidP, _U, eP]
          VP = Q[vidP, _V, eP]
          EP = Q[vidP, _E, eP]
          PP = p0 * CUDAnative.pow(R_gas * EP / p0, c_p / c_v)
        elseif bc == 1
          UnM = nxM * UM + nyM * VM
          UP = UM - 2 * UnM * nxM
          VP = VM - 2 * UnM * nyM
          ρP = ρM
          EP = EM
          PP = PM
        end

        ρMinv = 1 / ρM
        fluxρM_x = UM
        fluxUM_x = ρMinv * UM * UM + PM
        fluxVM_x = ρMinv * VM * UM
        fluxEM_x = ρMinv * UM * EM

        fluxρM_y = VM
        fluxUM_y = ρMinv * UM * VM
        fluxVM_y = ρMinv * VM * VM + PM
        fluxEM_y = ρMinv * VM * EM

        ρPinv = 1 / ρP
        fluxρP_x = UP
        fluxUP_x = ρPinv * UP * UP + PP
        fluxVP_x = ρPinv * VP * UP
        fluxEP_x = ρPinv * UP * EP

        fluxρP_y = VP
        fluxUP_y = ρPinv * UP * VP
        fluxVP_y = ρPinv * VP * VP + PP
        fluxEP_y = ρPinv * VP * EP

        λM = ρMinv * abs(nxM * UM + nyM * VM) + CUDAnative.sqrt(ρMinv * γ * PM)
        λP = ρPinv * abs(nxM * UP + nyM * VP) + CUDAnative.sqrt(ρPinv * γ * PP)
        λ  =  max(λM, λP)

        #Compute Numerical Flux and Update
        fluxρS = (nxM * (fluxρM_x + fluxρP_x) + nyM * (fluxρM_y + fluxρP_y) - λ * (ρP - ρM)) / 2
        fluxUS = (nxM * (fluxUM_x + fluxUP_x) + nyM * (fluxUM_y + fluxUP_y) - λ * (UP - UM)) / 2
        fluxVS = (nxM * (fluxVM_x + fluxVP_x) + nyM * (fluxVM_y + fluxVP_y) - λ * (VP - VM)) / 2
        fluxES = (nxM * (fluxEM_x + fluxEP_x) + nyM * (fluxEM_y + fluxEP_y) - λ * (EP - EM)) / 2

        #Update RHS
        rhs[vidM, _ρ, eM] -= sMJ * fluxρS
        rhs[vidM, _U, eM] -= sMJ * fluxUS
        rhs[vidM, _V, eM] -= sMJ * fluxVS
        rhs[vidM, _E, eM] -= sMJ * fluxES
      end
      sync_threads()
    end
  end
  nothing
end
# }}}

# {{{ Update solution (for all dimensions)
@hascuda function knl_updatesolution!(::Val{dim}, ::Val{N}, rhs, Q, vgeo, nelem, rka,
                             rkb, dt) where {dim, N}
  (i, j, k) = threadIdx()
  e = blockIdx().x

  Nq = N+1
  @inbounds if i <= Nq && j <= Nq && k <= Nq && e <= nelem
    n = i + (j-1) * Nq + (k-1) * Nq * Nq
    MJI = vgeo[n, _MJI, e]
    for s = 1:_nstate
      Q[n, s, e] += rkb * dt * rhs[n, s, e] * MJI
      rhs[n, s, e] *= rka
    end
  end
  nothing
end
# }}}

# }}}

# {{{ Fill sendQ on device with Q (for all dimensions)
@hascuda function knl_fillsendQ!(::Val{dim}, ::Val{N}, sendQ, Q,
                        sendelems) where {N, dim}
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
@hascuda function knl_transferrecvQ!(::Val{dim}, ::Val{N}, Q, recvQ, nelem,
                            nrealelem) where {N, dim}
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
@hascuda function volumerhs!(::Val{dim}, ::Val{N}, d_rhsC::CuArray, d_QC,
                             d_vgeoC, d_D, elems) where {dim, N}
  nelem = length(elems)
  @cuda(threads=ntuple(j->N+1, dim), blocks=nelem,
        knl_volumerhs!(Val(dim), Val(N), d_rhsC, d_QC, d_vgeoC, d_D, nelem))
end

@hascuda function facerhs!(::Val{dim}, ::Val{N}, d_rhsL::CuArray, d_QL, d_sgeo,
                           elems, d_vmapM, d_vmapP, d_elemtobndy) where {dim, N}
  nelem = length(elems)
  @cuda(threads=(ntuple(j->N+1, dim-1)..., 1), blocks=nelem,
        knl_facerhs!(Val(dim), Val(N), d_rhsL, d_QL, d_sgeo, nelem, d_vmapM,
                     d_vmapP, d_elemtobndy))
end

@hascuda function updatesolution!(::Val{dim}, ::Val{N}, d_rhsL::CuArray, d_QL,
                                  d_vgeoL, elems, rka, rkb, dt) where {dim, N}
  nelem = length(elems)
  @cuda(threads=ntuple(j->N+1, dim), blocks=nelem,
        knl_updatesolution!(Val(dim), Val(N), d_rhsL, d_QL, d_vgeoL, nelem, rka,
                            rkb, dt))
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
                      dt, nsteps, tout, vmapM, vmapP, mpicomm, iplot;
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
      volumerhs!(Val(dim), Val(N), d_rhsC, d_QC, d_vgeoC, d_D, mesh.realelems)

      # wait on MPI receives
      MPI.Waitall!(recvreq)

      # copy data to state vectors
      transferrecvQ!(Val(dim), Val(N), d_recvQ, recvQ, d_QL, nrealelem)

      # face RHS computation
      facerhs!(Val(dim), Val(N), d_rhsL, d_QL, d_sgeo, mesh.realelems, d_vmapM,
               d_vmapP, d_elemtobndy)

      # update solution and scale RHS
      updatesolution!(Val(dim), Val(N), d_rhsL, d_QL, d_vgeoL, mesh.realelems,
                      RKA[s%length(RKA)+1], RKB[s], dt)
    end
    if step == 1
      @hascuda synchronize()
      start_time = time_ns()
    end
    if mpirank == 0 && (time_ns() - t1)*1e-9 > tout
      @hascuda synchronize()
      t1 = time_ns()
      avg_stage_time = (time_ns() - start_time) * 1e-9 / ((step-1) * length(RKA))
      @show (step, nsteps, avg_stage_time)
    end
    # TODO: Fix VTK for 1-D
      #    if dim > 1 && plotstep > 0 && step % plotstep == 0
    if dim > 1 && mod(step,iplot) == 0
      Q .= d_QL
      convert_set2c_to_set2nc(Val(dim), Val(N), vgeo, Q)
      X = ntuple(j->reshape((@view vgeo[:, _x+j-1, :]), ntuple(j->N+1,dim)...,
                            nelem), dim)
      ρ = reshape((@view Q[:, _ρ, :]), ntuple(j->(N+1),dim)..., nelem)
      U = reshape((@view Q[:, _U, :]), ntuple(j->(N+1),dim)..., nelem)
      V = reshape((@view Q[:, _V, :]), ntuple(j->(N+1),dim)..., nelem)
      E = reshape((@view Q[:, _E, :]), ntuple(j->(N+1),dim)..., nelem)
      E = E .- 300.0
      writemesh(@sprintf("viz/euler%dD_set2c_%s_rank_%04d_step_%05d",
                         dim, ArrType, mpirank, step), X...;
                fields=(("ρ", ρ), ("U", U), ("V", V), ("E", E)),
                realelems=mesh.realelems)
    end
  end
  if mpirank == 0
    avg_stage_time = (time_ns() - start_time) * 1e-9 / ((nsteps-1) * length(RKA))
    @show (nsteps, avg_stage_time)
  end
  Q .= d_QL
  rhs .= d_rhsL
end
# }}}

# {{{ convert_variables
function convert_set2nc_to_set2c(::Val{dim}, ::Val{N}, vgeo, Q) where {dim, N}
  DFloat = eltype(Q)
  γ::DFloat       = _γ
  p0::DFloat      = _p0
  R_gas::DFloat   = _R_gas
  c_p::DFloat     = _c_p
  c_v::DFloat     = _c_v
  gravity::DFloat = _gravity

  Np = (N+1)^dim
  (~, ~, nelem) = size(Q)

  @inbounds for e = 1:nelem, n = 1:Np
    ρ, u, v, E = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e], Q[n, _E, e]
    Q[n, _U, e] = ρ*u
    Q[n, _V, e] = ρ*v
    Q[n, _E, e] = ρ*E
  end
end

function convert_set2c_to_set2nc(::Val{dim}, ::Val{N}, vgeo, Q) where {dim, N}
  DFloat = eltype(Q)
  γ::DFloat       = _γ
  p0::DFloat      = _p0
  R_gas::DFloat   = _R_gas
  c_p::DFloat     = _c_p
  c_v::DFloat     = _c_v
  gravity::DFloat = _gravity

  Np = (N+1)^dim
  (~, ~, nelem) = size(Q)

  @inbounds for e = 1:nelem, n = 1:Np
    ρ, U, V, E = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e], Q[n, _E, e]
    u=U/ρ
    v=V/ρ
    E=E/ρ
    Q[n, _U, e] = u
    Q[n, _V, e] = v
    Q[n, _E, e] = E
  end
end

function convert_set2nc_to_set3c(::Val{dim}, ::Val{N}, vgeo, Q) where {dim, N}
  DFloat = eltype(Q)
  γ::DFloat       = _γ
  p0::DFloat      = _p0
  R_gas::DFloat   = _R_gas
  c_p::DFloat     = _c_p
  c_v::DFloat     = _c_v
  gravity::DFloat = _gravity

  Np = (N+1)^dim
  (~, ~, nelem) = size(Q)

  @inbounds for e = 1:nelem, n = 1:Np
    ρ, u, v, E = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e], Q[n, _E, e]
    y = vgeo[n, _y, e]
    P = p0 * (ρ * R_gas * E / p0)^(c_p / c_v)
    T = P/(ρ*R_gas)
    E = c_v*T + 0.5*(u^2 + v^2) + gravity*y
    Q[n, _U, e] = ρ*u
    Q[n, _V, e] = ρ*v
    Q[n, _E, e] = ρ*E
  end
end

function convert_set3c_to_set2nc(::Val{dim}, ::Val{N}, vgeo, Q) where {dim, N}
  DFloat = eltype(Q)
  γ::DFloat       = _γ
  p0::DFloat      = _p0
  R_gas::DFloat   = _R_gas
  c_p::DFloat     = _c_p
  c_v::DFloat     = _c_v
  gravity::DFloat = _gravity

  Np = (N+1)^dim
  (~, ~, nelem) = size(Q)

  @inbounds for e = 1:nelem, n = 1:Np
    ρ, U, V, E = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e], Q[n, _E, e]
    y = vgeo[n, _y, e]
    u=U/ρ
    v=V/ρ
    E=E/ρ
    P = (R_gas/c_v)*ρ*(E - 0.5*(u^2 + v^2) - gravity*y)
    E=p0/(ρ * R_gas)*( P/p0 )^(c_v/c_p)
    Q[n, _U, e] = u
    Q[n, _V, e] = v
    Q[n, _E, e] = E
  end
end

function convert_set2nc_to_set4c(::Val{dim}, ::Val{N}, vgeo, Q) where {dim, N}
  DFloat = eltype(Q)
  γ::DFloat       = _γ
  p0::DFloat      = _p0
  R_gas::DFloat   = _R_gas
  c_p::DFloat     = _c_p
  c_v::DFloat     = _c_v
  gravity::DFloat = _gravity

  Np = (N+1)^dim
  (~, ~, nelem) = size(Q)

  @inbounds for e = 1:nelem, n = 1:Np
    ρ, u, v, E = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e], Q[n, _E, e]
    y = vgeo[n, _y, e]
    P = p0 * (ρ * R_gas * E / p0)^(c_p / c_v)
    T = P/(ρ*R_gas)
    E = c_v*T
    Q[n, _U, e] = ρ*u
    Q[n, _V, e] = ρ*v
    Q[n, _E, e] = ρ*E
  end
end

function convert_set4c_to_set2nc(::Val{dim}, ::Val{N}, vgeo, Q) where {dim, N}
  DFloat = eltype(Q)
  γ::DFloat       = _γ
  p0::DFloat      = _p0
  R_gas::DFloat   = _R_gas
  c_p::DFloat     = _c_p
  c_v::DFloat     = _c_v
  gravity::DFloat = _gravity

  Np = (N+1)^dim
  (~, ~, nelem) = size(Q)

  @inbounds for e = 1:nelem, n = 1:Np
    ρ, U, V, E = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e], Q[n, _E, e]
    y = vgeo[n, _y, e]
    u=U/ρ
    v=V/ρ
    T=E/(ρ*c_v)
    P=ρ*R_gas*T
    E=p0/(ρ* R_gas)*(P/p0)^(c_v/c_p)
    Q[n, _U, e] = u
    Q[n, _V, e] = v
    Q[n, _E, e] = E
  end
end
# }}}

# {{{ euler driver
function euler(::Val{dim}, ::Val{N}, mpicomm, ic, mesh, tend, iplot;
               meshwarp=(x...)->identity(x),
               tout = 1, ArrType=Array, plotstep=0) where {dim, N}
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
  rhs = zeros(DFloat, (N+1)^dim, _nstate, nelem)

  # setup the initial condition
  mpirank == 0 && println("[CPU] computing initial conditions (CPU)...")
  @inbounds for e = 1:nelem, i = 1:(N+1)^dim
    x, y = vgeo[i, _x, e], vgeo[i, _y, e]
    ρ, U, V, E = ic(x, y)
    Q[i, _ρ, e] = ρ
    Q[i, _U, e] = U
    Q[i, _V, e] = V
    Q[i, _E, e] = E
  end

  # Convert to proper variables
  mpirank == 0 && println("[CPU] converting variables (CPU)...")
  convert_set2nc_to_set4c(Val(dim), Val(N), vgeo, Q)
  convert_set4c_to_set2nc(Val(dim), Val(N), vgeo, Q)
  convert_set2nc_to_set2c(Val(dim), Val(N), vgeo, Q)

  # Compute time step
  mpirank == 0 && println("[CPU] computing dt (CPU)...")
  base_dt = cfl(Val(dim), Val(N), vgeo, Q, mpicomm) / N^√2
  base_dt=0.02 #FXG DT
  mpirank == 0 && @show base_dt

  nsteps = ceil(Int64, tend / base_dt)
  dt = tend / nsteps
  mpirank == 0 && @show (dt, nsteps, dt * nsteps, tend)

  # Do time stepping
  stats = zeros(DFloat, 2)
  mpirank == 0 && println("[CPU] computing initial energy...")
  Q_temp=copy(Q)
  convert_set2c_to_set2nc(Val(dim), Val(N), vgeo, Q_temp)
  stats[1] = L2energysquared(Val(dim), Val(N), Q_temp, vgeo, mesh.realelems)

  # plot the initial condition
  mkpath("viz")
  # TODO: Fix VTK for 1D
  if dim > 1
    X = ntuple(j->reshape((@view vgeo[:, _x+j-1, :]), ntuple(j->N+1,dim)...,
                          nelem), dim)
    ρ = reshape((@view Q_temp[:, _ρ, :]), ntuple(j->(N+1),dim)..., nelem)
    U = reshape((@view Q_temp[:, _U, :]), ntuple(j->(N+1),dim)..., nelem)
    V = reshape((@view Q_temp[:, _V, :]), ntuple(j->(N+1),dim)..., nelem)
    E = reshape((@view Q_temp[:, _E, :]), ntuple(j->(N+1),dim)..., nelem)
    E = E .- 300.0
    writemesh(@sprintf("viz/euler%dD_set2c_%s_rank_%04d_step_%05d",
                       dim, ArrType, mpirank, 0), X...;
              fields=(("ρ", ρ), ("U", U), ("V", V), ("E", E)),
              realelems=mesh.realelems)
  end

  #Call Time-stepping Routine
  mpirank == 0 && println("[DEV] starting time stepper...")
  lowstorageRK(Val(dim), Val(N), mesh, vgeo, sgeo, Q, rhs, D, dt, nsteps, tout,
               vmapM, vmapP, mpicomm, iplot; ArrType=ArrType, plotstep=plotstep)

  # TODO: Fix VTK for 1D
  if dim > 1
    Q_temp=copy(Q)
    convert_set2c_to_set2nc(Val(dim), Val(N), vgeo, Q_temp)
    X = ntuple(j->reshape((@view vgeo[:, _x+j-1, :]), ntuple(j->N+1,dim)...,
                          nelem), dim)
    ρ = reshape((@view Q_temp[:, _ρ, :]), ntuple(j->(N+1),dim)..., nelem)
    U = reshape((@view Q_temp[:, _U, :]), ntuple(j->(N+1),dim)..., nelem)
    V = reshape((@view Q_temp[:, _V, :]), ntuple(j->(N+1),dim)..., nelem)
    E = reshape((@view Q_temp[:, _E, :]), ntuple(j->(N+1),dim)..., nelem)
    E = E .- 300.0
    writemesh(@sprintf("viz/euler%dD_set2c_%s_rank_%04d_step_%05d",
                       dim, ArrType, mpirank, nsteps), X...;
              fields=(("ρ", ρ), ("U", U), ("V", V), ("E", E)),
              realelems=mesh.realelems)
  end

  mpirank == 0 && println("[CPU] computing final energy...")
  stats[2] = L2energysquared(Val(dim), Val(N), Q_temp, vgeo, mesh.realelems)

  stats = sqrt.(MPI.allreduce(stats, MPI.SUM, mpicomm))

  if  mpirank == 0
    @show eng0 = stats[1]
    @show engf = stats[2]
    @show Δeng = engf - eng0
  end
end
# }}}

# {{{ main
function main()
  DFloat = Float64

  # MPI.Init()
  MPI.Initialized() || MPI.Init()
  MPI.finalize_atexit()

  mpicomm = MPI.COMM_WORLD
  mpirank = MPI.Comm_rank(mpicomm)
  mpisize = MPI.Comm_size(mpicomm)

  # FIXME: query via hostname
  @hascuda device!(mpirank % length(devices()))

  #Initial Conditions
  function ic(dim, x...)
    # FIXME: Type generic?
    DFloat = eltype(x)
    γ::DFloat       = _γ
    p0::DFloat      = _p0
    R_gas::DFloat   = _R_gas
    c_p::DFloat     = _c_p
    c_v::DFloat     = _c_v
    gravity::DFloat = _gravity

    u0 = 0
    r = sqrt((x[1]-500)^2 + (x[dim]-350)^2 )
    rc = 250.0
    θ_ref=300.0
    θ_c=0.5
    Δθ=0.0
    if r <= rc
      Δθ = 0.5 * θ_c * (1.0 + cos(π * r/rc))
    end
    θ_k=θ_ref + Δθ
    π_k=1.0 - gravity/(c_p*θ_k)*x[dim]
    c=c_v/R_gas
    ρ_k=p0/(R_gas*θ_k)*(π_k)^c
    ρ = ρ_k
    U = u0
    V = 0.0
    E = θ_k
    ρ, U, V, E
  end

  time_final = DFloat(20.0)
  iplot=100
  Ne = 10
  N  = 4
  dim = 2
  hardware="cpu"
  @show (N,Ne,iplot,time_final,hardware)

  mesh2D = brickmesh((range(DFloat(0); length=Ne+1, stop=1000),
                      range(DFloat(0); length=Ne+1, stop=1000)),
                     (true, false),
                   part=mpirank+1, numparts=mpisize)

  if hardware == "cpu"
     mpirank == 0 && println("Running 2d (CPU)...")
     euler(Val(dim), Val(N), mpicomm, (x...)->ic(dim, x...), mesh2D, time_final, iplot;
        ArrType=Array, tout = 10)
     mpirank == 0 && println()
  elseif hardware == "gpu"
     @hascuda begin
       mpirank == 0 && println("Running 2d (GPU)...")
       euler(Val(dim), Val(N), mpicomm, (x...)->ic(dim, x...), mesh2D, time_final, iplot;
          ArrType=CuArray, tout = 10)
       mpirank == 0 && println()
     end
  end
  nothing
end
# }}}

main()
