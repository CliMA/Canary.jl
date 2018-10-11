include(joinpath(@__DIR__,"vtk.jl"))
using MPI
using Canary
using Printf: @sprintf
using CUDAnative
using CUDAdrv

function Base.reshape(A::CuArray, dims::NTuple{N, Int}) where {N}
  @assert prod(dims) == prod(size(A))
  CuArray{eltype(A), length(dims)}(dims, A.buf)
end

# {{{ constants
# note the order of the fields below is also assumed in the code.
const _nstate = 4
const _Ux, _Uy, _Uz, _ρ = 1:_nstate
const stateid = (Ux = _Ux, Uy = _Uy, Uz = _Uz, ρ = _ρ)

const _nvgeo = 14
const _ξx, _ηx, _ζx, _ξy, _ηy, _ζy, _ξz, _ηz, _ζz, _MJ, _MJI,
       _x, _y, _z = 1:_nvgeo
const vgeoid = (ξx = _ξx, ηx = _ηx, ζx = _ζx,
                ξy = _ξy, ηy = _ηy, ζy = _ζy,
                ξz = _ξz, ηz = _ηz, ζz = _ζz,
                MJ = _MJ, MJI = _MJI,
                 x = _x,   y = _y,   z = _z)

const _nsgeo = 5
const _nx, _ny, _nz, _sMJ, _vMJI = 1:_nsgeo
const sgeoid = (nx = _nx, ny = _ny, nz = _nz, sMJ = _sMJ, vMJI = _vMJI)
# }}}

# {{{ cfl
function cfl(::Val{dim}, ::Val{N}, vgeo, Q, mpicomm) where {dim, N}
  DFloat = eltype(Q)
  Np = (N+1)^dim
  (~, ~, nelem) = size(Q)

  dt = [floatmax(DFloat)]

  if dim == 1
    @inbounds for e = 1:nelem, n = 1:Np
      Ux, ξx = Q[n, _Ux, e], vgeo[n, _ξx, e]

      loc_dt = 2 / abs(Ux*ξx)
      dt[1] = min(dt[1], loc_dt)
    end
  end

  if dim == 2
    @inbounds for e = 1:nelem, n = 1:Np
      Ux, Uy = Q[n, _Ux, e], Q[n, _Uy, e]
      ξx, ξy, ηx, ηy = vgeo[n, _ξx, e], vgeo[n, _ξy, e],
                       vgeo[n, _ηx, e], vgeo[n, _ηy, e]

      loc_dt = 2 / max(abs(Ux*ξx + Uy*ξy), abs(Ux*ηx + Uy*ηy))
      dt[1] = min(dt[1], loc_dt)
    end
  end

  if dim == 3
    @inbounds for e = 1:nelem, n = 1:Np
      Ux, Uy, Uz = Q[n, _Ux, e], Q[n, _Uy, e], Q[n, _Uz, e]
      ξx, ξy, ξz = vgeo[n, _ξx, e], vgeo[n, _ξy, e], vgeo[n, _ξz, e]
      ηx, ηy, ηz = vgeo[n, _ηx, e], vgeo[n, _ηy, e], vgeo[n, _ηz, e]
      ζx, ζy, ζz = vgeo[n, _ζx, e], vgeo[n, _ζy, e], vgeo[n, _ζz, e]

      loc_dt = 2 ./ max(abs(Ux*ξx + Uy*ξy + Uz*ξz),
                        abs(Ux*ηx + Uy*ηy + Uz*ηz),
                        abs(Ux*ζx + Uy*ζy + Uz*ζz))
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

  (ξx, ηx, ζx, ξy, ηy, ζy, ξz, ηz, ζz, MJ, MJI, x, y, z) =
      ntuple(j->(@view vgeo[:, j, :]), _nvgeo)
  J = similar(x)
  (nx, ny, nz, sMJ, vMJI) = ntuple(j->(@view sgeo[ j, :, :, :]), _nsgeo)
  sJ = similar(sMJ)

  X = ntuple(j->(@view vgeo[:, _x+j-1, :]), dim)
  creategrid!(X..., mesh.elemtocoord, ξ)

  @inbounds for j = 1:length(x)
    (x[j], y[j], z[j]) = meshwarp(x[j], y[j], z[j])
  end

  # Compute the metric terms
  if dim == 1
    computemetric!(x, J, ξx, sJ, nx, D)
  elseif dim == 2
    computemetric!(x, y, J, ξx, ηx, ξy, ηy, sJ, nx, ny, D)
  elseif dim == 3
    computemetric!(x, y, z, J, ξx, ηx, ζx, ξy, ηy, ζy, ξz, ηz, ζz, sJ,
                   nx, ny, nz, D)
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

# {{{ 1-D
# Volume RHS for 1-D
function volumerhs!(::Val{1}, ::Val{N}, rhs, Q, vgeo, D, elems) where N
  Nq = N + 1

  @inbounds for e in elems
    for i in 1:Nq
      for n in 1:Nq
        rhs[i, _ρ, e] += D[n, i] * (vgeo[n, _MJ, e] * vgeo[n, _ξx, e] *
                                    Q[n, _Ux, e] * Q[n, _ρ, e])
      end
    end
  end
end

# Face RHS for 1-D
function facerhs!(::Val{1}, ::Val{N}, rhs, Q, sgeo, elems, vmapM,
                  vmapP) where N
  Np = N+1
  nface = 2

  @inbounds for e in elems
    for f = 1:nface
      (nxM, ~, ~, sMJ, ~) = sgeo[:, 1, f, e]
      idM, idP = vmapM[1, f, e], vmapP[1, f, e]

      eM, eP = e, ((idP - 1) ÷ Np) + 1
      vidM, vidP = ((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1

      ρM = Q[vidM, _ρ, eM]
      UxM = Q[vidM, _Ux, eM]
      FxM = ρM * UxM

      ρP = Q[vidP, _ρ, eP]
      UxP = Q[vidP, _Ux, eP]
      FxP = ρP * UxP

      λ = max(abs(nxM * UxM), abs(nxM * UxP))

      F = (nxM * (FxM + FxP) + λ * (ρM - ρP)) / 2
      rhs[vidM, _ρ, eM] -= sMJ * F
    end
  end
end
# }}}

# {{{ 2-D
# Volume RHS for 2-D
function volumerhs!(::Val{2}, ::Val{N}, rhs, Q, vgeo, D, elems) where N
  Nq = N + 1

  ~, ~, nelem = size(Q)
  Q = reshape(Q, Nq, Nq, _nstate, nelem)
  rhs = reshape(rhs, Nq, Nq, _nstate, nelem)
  vgeo = reshape(vgeo, Nq, Nq, _nvgeo, nelem)

  @inbounds for e in elems
    # loop of ξ-grid lines
    for j = 1:Nq, i = 1:Nq, n = 1:Nq
      rhs[i, j, _ρ, e] += D[n, i] * (vgeo[n, j, _MJ, e] * Q[n, j, _ρ, e] *
                                     (vgeo[n, j, _ξx, e] .* Q[n, j, _Ux, e] +
                                      vgeo[n, j, _ξy, e] .* Q[n, j, _Uy, e]))
    end
    # loop of η-grid lines
    for i = 1:Nq, j = 1:Nq, n = 1:Nq
      rhs[i, j, _ρ, e] += D[n, j] * (vgeo[i, n, _MJ, e] * Q[i, n, _ρ, e] *
                                     (vgeo[i, n, _ηx, e] .* Q[i, n, _Ux, e] +
                                      vgeo[i, n, _ηy, e] .* Q[i, n, _Uy, e]))
    end
  end
end

# Face RHS for 2-D
function facerhs!(::Val{2}, ::Val{N}, rhs, Q, sgeo, elems, vmapM,
                  vmapP) where N
  Np = (N+1)^2
  Nfp = N+1
  nface = 4

  @inbounds for e in elems
    for f = 1:nface
      for n = 1:Nfp
        (nxM, nyM, ~, sMJ, ~) = sgeo[:, n, f, e]
        idM, idP = vmapM[n, f, e], vmapP[n, f, e]

        eM, eP = e, ((idP - 1) ÷ Np) + 1
        vidM, vidP = ((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1

        ρM = Q[vidM, _ρ, eM]
        UxM = Q[vidM, _Ux, eM]
        UyM = Q[vidM, _Uy, eM]
        FxM = ρM * UxM
        FyM = ρM * UyM

        ρP = Q[vidP, _ρ, eP]
        UxP = Q[vidP, _Ux, eP]
        UyP = Q[vidP, _Uy, eP]
        FxP = ρP * UxP
        FyP = ρP * UyP

        λ = max(abs(nxM * UxM + nyM * UyM), abs(nxM * UxP + nyM * UyP))

        F = (nxM * (FxM + FxP) + nyM * (FyM + FyP) + λ * (ρM - ρP)) / 2

        rhs[vidM, _ρ, eM] -= sMJ * F
      end
    end
  end
end
# }}}

# {{{ 3-D
# Volume RHS for 3-D
function volumerhs!(::Val{3}, ::Val{N}, rhs, Q, vgeo, D, elems) where N
  Nq = N + 1
  ~, ~, nelem = size(Q)

  Q = reshape(Q, Nq, Nq, Nq, _nstate, nelem)
  rhs = reshape(rhs, Nq, Nq, Nq, _nstate, nelem)
  vgeo = reshape(vgeo, Nq, Nq, Nq, _nvgeo, nelem)

  @inbounds for e in elems
    # loop of ξ-grid lines
    for k = 1:Nq, j = 1:Nq, i = 1:Nq,  n = 1:Nq
      rhs[i, j, k, _ρ, e] +=
        D[n, i] * (vgeo[n, j, k, _MJ, e] * Q[n, j, k, _ρ, e] *
                   (vgeo[n, j, k, _ξx, e] * Q[n, j, k, _Ux, e] +
                    vgeo[n, j, k, _ξy, e] * Q[n, j, k, _Uy, e] +
                    vgeo[n, j, k, _ξz, e] * Q[n, j, k, _Uz, e]))
    end

    # loop of η-grid lines
    for k = 1:Nq, i = 1:Nq, j = 1:Nq, n = 1:Nq
      rhs[i, j, k, _ρ, e] +=
        D[n, j] * (vgeo[i, n, k, _MJ, e] * Q[i, n, k, _ρ, e] *
                   (vgeo[i, n, k, _ηx, e] * Q[i, n, k, _Ux, e] +
                    vgeo[i, n, k, _ηy, e] * Q[i, n, k, _Uy, e] +
                    vgeo[i, n, k, _ηz, e] * Q[i, n, k, _Uz, e]))
    end

    # loop of ζ-grid lines
    for j = 1:Nq, i = 1:Nq, k = 1:Nq, n = 1:Nq
      rhs[i, j, k, _ρ, e] +=
        D[n, k] * (vgeo[i, j, n, _MJ, e] * Q[i, j, n, _ρ, e] *
                   (vgeo[i, j, n, _ζx, e] * Q[i, j, n, _Ux, e] +
                    vgeo[i, j, n, _ζy, e] * Q[i, j, n, _Uy, e] +
                    vgeo[i, j, n, _ζz, e] * Q[i, j, n, _Uz, e]))
    end
  end
end

function kernel_volumerhs_orig!(::Val{3}, ::Val{N}, rhs, Q, vgeo, D,
                                nelem) where N
  Nq = N + 1

  (i, j, k) = threadIdx()
  e = blockIdx().x

  @inbounds if i <= Nq && j <= Nq && k <= Nq && e <= nelem
    # loop of ξ-grid lines
    for n = 1:Nq
      rhs[i, j, k, _ρ, e] +=
        D[n, i] * ( vgeo[n, j, k, _MJ, e] * Q[n, j, k, _ρ, e] *
                   (vgeo[n, j, k, _ξx, e] * Q[n, j, k, _Ux, e] +
                    vgeo[n, j, k, _ξy, e] * Q[n, j, k, _Uy, e] +
                    vgeo[n, j, k, _ξz, e] * Q[n, j, k, _Uz, e]))
    end

    # loop of η-grid lines
    for n = 1:Nq
      rhs[i, j, k, _ρ, e] +=
        D[n, j] * ( vgeo[i, n, k, _MJ, e] * Q[i, n, k, _ρ, e] *
                   (vgeo[i, n, k, _ηx, e] * Q[i, n, k, _Ux, e] +
                    vgeo[i, n, k, _ηy, e] * Q[i, n, k, _Uy, e] +
                    vgeo[i, n, k, _ηz, e] * Q[i, n, k, _Uz, e]))
    end

    # loop of ζ-grid lines
    for n = 1:Nq
      rhs[i, j, k, _ρ, e] +=
        D[n, k] * ( vgeo[i, j, n, _MJ, e] * Q[i, j, n, _ρ, e] *
                   (vgeo[i, j, n, _ζx, e] * Q[i, j, n, _Ux, e] +
                    vgeo[i, j, n, _ζy, e] * Q[i, j, n, _Uy, e] +
                    vgeo[i, j, n, _ζz, e] * Q[i, j, n, _Uz, e]))
    end
  end
  nothing
end

function kernel_volumerhs!(::Val{3}, ::Val{N}, rhs, Q, vgeo, D, nelem) where N
  Nq = N + 1

  (i, j, k) = threadIdx()
  e = blockIdx().x

  s_D = @cuStaticSharedMem(eltype(D), (Nq, Nq))
  s_F = @cuStaticSharedMem(eltype(Q), (Nq, Nq, Nq, _nstate))
  s_G = @cuStaticSharedMem(eltype(Q), (Nq, Nq, Nq, _nstate))
  s_H = @cuStaticSharedMem(eltype(Q), (Nq, Nq, Nq, _nstate))

  rhsρ = zero(eltype(rhs))
  if i <= Nq && j <= Nq && k <= Nq && e <= nelem
    # Load derivative into shared memory
    if k == 1
      s_D[i, j] = D[i, j]
    end

    # Load values will need into registers
    MJ = vgeo[i, j, k, _MJ, e]
    ξx = vgeo[i, j, k, _ξx, e]
    ξy = vgeo[i, j, k, _ξy, e]
    ξz = vgeo[i, j, k, _ξz, e]
    ηx = vgeo[i, j, k, _ηx, e]
    ηy = vgeo[i, j, k, _ηy, e]
    ηz = vgeo[i, j, k, _ηz, e]
    ζx = vgeo[i, j, k, _ζx, e]
    ζy = vgeo[i, j, k, _ζy, e]
    ζz = vgeo[i, j, k, _ζz, e]
    ρ =  Q[i, j, k, _ρ, e]
    Ux = Q[i, j, k, _Ux, e]
    Uy = Q[i, j, k, _Uy, e]
    Uz = Q[i, j, k, _Uz, e]
    rhsρ = rhs[i, j, k, _ρ, e]

    # store flux in shared memory
    s_F[i, j, k, _ρ] = MJ * ρ * (ξx * Ux + ξy * Uy + ξz * Uz)
    s_G[i, j, k, _ρ] = MJ * ρ * (ηx * Ux + ηy * Uy + ηz * Uz)
    s_H[i, j, k, _ρ] = MJ * ρ * (ζx * Ux + ζy * Uy + ζz * Uz)
  end

  sync_threads()

  @inbounds if i <= Nq && j <= Nq && k <= Nq && e <= nelem
    # loop of ξ-grid lines
    for n = 1:Nq
      rhsρ += s_D[n, i] * s_F[n, j, k, _ρ]
    end

    # loop of η-grid lines
    for n = 1:Nq
      rhsρ += s_D[n, j] * s_G[i, n, k, _ρ]
    end

    # loop of ζ-grid lines
    for n = 1:Nq
      rhsρ += s_D[n, k] * s_H[i, j, n, _ρ]
    end

    rhs[i, j, k, _ρ, e] = rhsρ
  end
  nothing
end

# Face RHS for 3-D
function facerhs!(::Val{3}, ::Val{N}, rhs, Q, sgeo, elems, vmapM,
                  vmapP) where N
  Np = (N+1)^3
  Nfp = (N+1)^2
  nface = 6

  @inbounds for e in elems
    for f = 1:nface
      for n = 1:Nfp
        (nxM, nyM, nzM, sMJ, ~) = sgeo[:, n, f, e]
        idM, idP = vmapM[n, f, e], vmapP[n, f, e]

        eM, eP = e, ((idP - 1) ÷ Np) + 1
        vidM, vidP = ((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1

        ρM = Q[vidM, _ρ, eM]
        UxM = Q[vidM, _Ux, eM]
        UyM = Q[vidM, _Uy, eM]
        UzM = Q[vidM, _Uz, eM]
        FxM = ρM * UxM
        FyM = ρM * UyM
        FzM = ρM * UzM

        ρP = Q[vidP, _ρ, eP]
        UxP = Q[vidP, _Ux, eP]
        UyP = Q[vidP, _Uy, eP]
        UzP = Q[vidP, _Uz, eP]
        FxP = ρP * UxP
        FyP = ρP * UyP
        FzP = ρP * UzP

        λ = max(abs(nxM * UxM + nyM * UyM + nzM * UzM),
                abs(nxM * UxP + nyM * UyP + nzM * UzP))

        F = (nxM * (FxM + FxP) + nyM * (FyM + FyP) + nzM * (FzM + FzP) +
             λ * (ρM - ρP)) / 2
        rhs[vidM, _ρ, eM] -= sMJ * F
      end
    end
  end
end

function kernel_facerhs_orig!(::Val{dim}, ::Val{N}, rhs, Q, sgeo, nelem, vmapM,
                              vmapP) where {dim, N}
  if dim == 1
    Np = (N+1)
    nface = 2
  elseif dim == 2
    Np = (N+1) * (N+1)
    nface = 4
  elseif dim == 3
    Np = (N+1) * (N+1) * (N+1)
    nface = 6
  end

  (i, j, k) = threadIdx()
  e = blockIdx().x

  Nq = N+1
  @inbounds if i <= Nq && j <= Nq && k == 1 && e <= nelem
    n = i + (j-1) * Nq
    for f = 1:nface
      (nxM, nyM) = (sgeo[_nx, n, f, e], sgeo[_ny, n, f, e])
      (nzM, sMJ) = (sgeo[_nz, n, f, e], sgeo[_sMJ, n, f, e])

      (idM, idP) = (vmapM[n, f, e], vmapP[n, f, e])

      (eM, eP) = (e, ((idP - 1) ÷ Np) + 1)
      (vidM, vidP) = (((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1)

      ρM = Q[vidM, _ρ, eM]
      UxM = Q[vidM, _Ux, eM]
      UyM = Q[vidM, _Uy, eM]
      UzM = Q[vidM, _Uz, eM]
      FxM = ρM * UxM
      FyM = ρM * UyM
      FzM = ρM * UzM

      ρP = Q[vidP, _ρ, eP]
      UxP = Q[vidP, _Ux, eP]
      UyP = Q[vidP, _Uy, eP]
      UzP = Q[vidP, _Uz, eP]
      FxP = ρP * UxP
      FyP = ρP * UyP
      FzP = ρP * UzP

      λ = max(abs(nxM * UxM + nyM * UyM + nzM * UzM),
              abs(nxM * UxP + nyM * UyP + nzM * UzP))

      F = (nxM * (FxM + FxP) + nyM * (FyM + FyP) + nzM * (FzM + FzP) +
           λ * (ρM - ρP)) / 2
      rhs[vidM, _ρ, eM] -= sMJ * F
      sync_threads() # FIXME: Really only needed every other faces
    end
  end
  nothing
end

function kernel_facerhs!(::Val{dim}, ::Val{N}, rhs, Q, sgeo, nelem, vmapM,
                         vmapP) where {dim, N}
  if dim == 1
    Np = (N+1)
    nface = 2
  elseif dim == 2
    Np = (N+1) * (N+1)
    nface = 4
  elseif dim == 3
    Np = (N+1) * (N+1) * (N+1)
    nface = 6
  end

  (i, j, k) = threadIdx()
  e = blockIdx().x

  Nq = N+1
  half = convert(eltype(Q), 0.5)

  @inbounds if i <= Nq && j <= Nq && k == 1 && e <= nelem
    n = i + (j-1) * Nq
    for lf = 1:2:nface
      for f = lf:lf+1
        (nxM, nyM) = (sgeo[_nx, n, f, e], sgeo[_ny, n, f, e])
        (nzM, sMJ) = (sgeo[_nz, n, f, e], sgeo[_sMJ, n, f, e])

        (idM, idP) = (vmapM[n, f, e], vmapP[n, f, e])

        (eM, eP) = (e, ((idP - 1) ÷ Np) + 1)
        (vidM, vidP) = (((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1)

        ρM = Q[vidM, _ρ, eM]
        UxM = Q[vidM, _Ux, eM]
        UyM = Q[vidM, _Uy, eM]
        UzM = Q[vidM, _Uz, eM]
        FxM = ρM * UxM
        FyM = ρM * UyM
        FzM = ρM * UzM

        ρP = Q[vidP, _ρ, eP]
        UxP = Q[vidP, _Ux, eP]
        UyP = Q[vidP, _Uy, eP]
        UzP = Q[vidP, _Uz, eP]
        FxP = ρP * UxP
        FyP = ρP * UyP
        FzP = ρP * UzP

        λ = max(abs(nxM * UxM + nyM * UyM + nzM * UzM),
                abs(nxM * UxP + nyM * UyP + nzM * UzP))

        F = half * (nxM * (FxM + FxP) + nyM * (FyM + FyP) + nzM * (FzM + FzP) +
                    λ * (ρM - ρP))
        rhs[vidM, _ρ, eM] -= sMJ * F
      end
      sync_threads()
    end
  end
  nothing
end
# }}}

# {{{ Update solution (for all dimensions)
function updatesolution!(::Val{dim}, ::Val{N}, rhs, Q, vgeo, elems, rka,
                         rkb, dt) where {dim, N}
  @inbounds for e = elems, s = 1:_nstate, i = 1:(N+1)^dim
    Q[i, s, e] += rkb * dt * rhs[i, s, e] * vgeo[i, _MJI, e]
    rhs[i, s, e] *= rka
  end
end

function kernel_updatesolution!(::Val{dim}, ::Val{N}, rhs, Q, vgeo, nelem, rka,
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

function kernel_updatesolution_orig!(::Val{dim}, ::Val{N}, rhs, Q, vgeo, nelem,
                                     rka, rkb, dt) where {dim, N}
  (i, j, k) = threadIdx()
  e = blockIdx().x

  Nq = N+1
  if i <= Nq && j <= Nq && k <= Nq && e <= nelem
    n = i + (j-1) * Nq + (k-1) * Nq * Nq
    @inbounds for s = 1:_nstate
      Q[n, s, e] += rkb * dt * rhs[n, s, e] * vgeo[n, _MJI, e]
      rhs[n, s, e] *= rka
    end
  end
  nothing
end

# }}}

# {{{ Fill sendQ on device with Q (for all dimensions)
function kernel_sendQ!(::Val{dim}, ::Val{N}, sendQ, Q, sendelems) where {N, dim}
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
function kernel_recvQ!(::Val{dim}, ::Val{N}, Q, recvQ, nelem,
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

# {{{ L2 Error (for all dimensions)
function L2errorsquared(::Val{dim}, ::Val{N}, Q, vgeo, elems, Qex,
                        t) where {dim, N}
  DFloat = eltype(Q)
  Np = (N+1)^dim
  (~, nstate, nelem) = size(Q)

  err = zero(DFloat)

  @inbounds for e = elems, i = 1:Np
    X = ntuple(j -> vgeo[i, _x-1+j, e] - Q[i, _Ux-1+j, e]*t, Val(dim))
    diff = Q[i, _ρ, e] - Qex.ρ(X...)

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
                      dt, nsteps, tout, vmapM, vmapP, mpicomm) where {dim, N}
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

  (d_QL, d_rhsL) = (CuArray(Q), CuArray(rhs))
  (d_vgeo, d_sgeo) = (CuArray(vgeo), CuArray(sgeo))
  (d_vmapM, d_vmapP) = (CuArray(vmapM), CuArray(vmapP))
  (d_sendelems, d_sendQ) = (CuArray(mesh.sendelems), CuArray(sendQ))
  d_recvQ = CuArray(recvQ)
  (d_D, ) = (CuArray(D), )

  Qshape    = (fill(N+1, dim)..., fill(1, 3-dim)..., size(Q, 2), size(Q, 3))
  vgeoshape = (fill(N+1, dim)..., fill(1, 3-dim)..., _nvgeo, size(Q, 3))

  d_QC = reshape(d_QL, Qshape)
  d_rhsC = reshape(d_rhsL, Qshape...)
  d_vgeoC = reshape(d_vgeo, vgeoshape)

  start_time = t1 = time_ns()
  vthreads=(fill(N+1, dim)...,)
  fthreads=(fill(N+1, dim-1)..., 1)
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
      if nsendelem > 0
        @cuda(threads=vthreads, blocks=nsendelem,
              kernel_sendQ!(Val(dim), Val(N), d_sendQ, d_QL, d_sendelems))
        sendQ .= d_sendQ
      end

      # post MPI sends
      for n = 1:nnabr
        sendreq[n] = MPI.Isend((@view sendQ[:, :, mesh.nabrtosend[n]]),
                               mesh.nabrtorank[n], 777, mpicomm)
      end

      # volume RHS computation
      # volumerhs!(Val(dim), Val(N), rhs, Q, vgeo, D, mesh.realelems)
      @cuda(threads=vthreads, blocks=nrealelem,
            kernel_volumerhs!(Val(dim), Val(N), d_rhsC, d_QC, d_vgeoC,
                              d_D, nrealelem))

      # wait on MPI receives
      MPI.Waitall!(recvreq)

      # copy data to state vectors
      if nrecvelem > 0
        d_recvQ .= recvQ
        @cuda(threads=vthreads, blocks=nrecvelem,
              kernel_recvQ!(Val(dim), Val(N), d_QL, d_recvQ, nrecvelem,
                            nrealelem))
      end

      # face RHS computation
      # facerhs!(Val(dim), Val(N), rhs, Q, sgeo, mesh.realelems, vmapM, vmapP)
      # putting 1 at end of threads tuple enables 1-D
      @cuda(threads=fthreads, blocks=nrealelem,
            kernel_facerhs!(Val(dim), Val(N), d_rhsL, d_QL, d_sgeo,
                            nrealelem, d_vmapM, d_vmapP))

      # update solution and scale RHS
      # updatesolution!(Val(dim), Val(N), rhs, Q, vgeo, mesh.realelems,
      #                 RKA[s%length(RKA)+1], RKB[s], dt)
      @cuda(threads=vthreads, blocks=nrealelem,
            kernel_updatesolution!(Val(dim), Val(N), d_rhsL, d_QL, d_vgeo,
                                   nrealelem, RKA[s%length(RKA)+1], RKB[s],
                                  DFloat(dt)))
    end
    step == 1 && (start_time = time_ns())
    synchronize()
    if mpirank == 0 && (time_ns() - t1)*1e-9 > tout
      t1 = time_ns()
      avg_stage_time = (time_ns() - start_time) * 1e-9 / ((step-1) * length(RKA))
      @show (step, nsteps, avg_stage_time)
    end
  end
  Q .= d_QL
  rhs .= d_rhsL
end
# }}}

# {{{ advection driver
function advection(mpicomm, ic, ::Val{N}, brickN::NTuple{dim, Int}, tend;
                   meshwarp=(x...)->identity(x),
                   tout = 1) where {N, dim}
  DFloat = typeof(tend)

  mpirank = MPI.Comm_rank(mpicomm)
  mpisize = MPI.Comm_size(mpicomm)

  # Generate an local view of a fully periodic Cartesian mesh.
  mesh = brickmesh(ntuple(i->range(DFloat(0); length=brickN[i]+1, stop=1), dim),
                   (fill(true, dim)...,);
                   part=mpirank+1, numparts=mpisize)

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
  (vgeo, sgeo) = computegeometry(Val(dim), mesh, D, ξ, ω, meshwarp, vmapM)
  (nface, nelem) = size(mesh.elemtoelem)

  # Storage for the solution, rhs, and error
  Q = zeros(DFloat, (N+1)^dim, _nstate, nelem)
  rhs = zeros(DFloat, (N+1)^dim, _nstate, nelem)

  # setup the initial condition
  @inbounds for e = 1:nelem, i = 1:(N+1)^dim
    x, y, z = vgeo[i, _x, e], vgeo[i, _y, e], vgeo[i, _z, e]
    Q[i, _ρ, e]  = ic.ρ(x, y, z)
    Q[i, _Ux, e] = ic.Ux(x, y, z)
    dim > 1 && (Q[i, _Uy, e] = ic.Uy(x, y, z))
    dim > 2 && (Q[i, _Uz, e] = ic.Uz(x, y, z))
  end

  # plot the initial condition
  mkpath("viz")
  # TODO: Fix VTK for 1-D
  if dim > 1
    X = ntuple(j->reshape((@view vgeo[:, _x+j-1, :]), ntuple(j->N+1,dim)...,
                          nelem), dim)
    ρ = reshape((@view Q[:, _ρ, :]), ntuple(j->(N+1),dim)..., nelem)
    writemesh(@sprintf("viz/advection%dD_rank_%04d_step_%05d", dim,
                       mpirank, 0), X...; fields=(("ρ", ρ),),
              realelems=mesh.realelems)
  end

  # Compute time step
  dt = cfl(Val(dim), Val(N), vgeo, Q, mpicomm) / N^√2

  nsteps = ceil(Int64, tend / dt)
  dt = tend / nsteps
  mpirank == 0 && @show (dt, nsteps)

  # Do time stepping
  stats = zeros(DFloat, 3)
  stats[1] = L2energysquared(Val(dim), Val(N), Q, vgeo, mesh.realelems)

  lowstorageRK(Val(dim), Val(N), mesh, vgeo, sgeo, Q, rhs, D, dt, nsteps,
               tout, vmapM, vmapP, mpicomm)

  stats[2] = L2energysquared(Val(dim), Val(N), Q, vgeo, mesh.realelems)
  stats[3] = L2errorsquared(Val(dim), Val(N), Q, vgeo, mesh.realelems, ic,
                            tend)

  stats = sqrt.(MPI.allreduce(stats, MPI.SUM, mpicomm))

  if  mpirank == 0
    @show eng = stats[1]
    @show engdiff = stats[2] - stats[1]
    @show err = stats[3]
  end
end
# }}}

# {{{ main
function main()
  # MPI.Init()
  MPI.Initialized() ||MPI.Init()
  MPI.finalize_atexit()

  mpicomm = MPI.COMM_WORLD
  mpirank = MPI.Comm_rank(mpicomm)

  # FIXME: query via hostname
  device!(mpirank % length(devices()))

  warping1D(x...) = (x[1] +  sin( π*x[1])/10, zero(x[1]), zero(x[1]))
  warping2D(x...) = (x[1] +  sin( π*x[1])*sin(2π*x[2])/10,
                     x[2] +  sin(2π*x[1])*sin( π*x[2])/10,
                     zero(x[1]))
  warping3D(x...) = (x[1] + (sin( π*x[1])*sin(2π*x[2])*cos(2π*x[3]))/10,
                     x[2] + (sin( π*x[2])*sin(2π*x[1])*cos(2π*x[3]))/10,
                     x[3] + (sin( π*x[3])*sin(2π*x[1])*cos(2π*x[2]))/10)

  ρ1D(x...) = sin(2π*x[1])
  ρ2D(x...) = sin(2π*x[1])*sin(2π*x[2])
  ρ3D(x...) = sin(2π*x[1])*sin(2π*x[2])*sin(2π*x[3])

  Ux(x...) = -3*one(x[1])/2
  Uy(x...) = -π*one(x[1])
  Uz(x...) =  exp(one(x[1]))

  #=
  mpirank == 0 && println("Running 1d...")
  advection(mpicomm, (ρ=ρ1D, Ux=Ux, Uy=Uy, Uz=Uz), Val(5), (3, ), Float32(π);
            meshwarp=warping1D)
  mpirank == 0 && println()

  mpirank == 0 && println("Running 2d...")
  advection(mpicomm, (ρ=ρ2D, Ux=Ux, Uy=Uy, Uz=Uz), Val(5), (3, 3), Float32(π);
            meshwarp=warping2D)
  mpirank == 0 && println()
  =#

  mpirank == 0 && println("Running 3d...")
  advection(mpicomm, (ρ=ρ3D, Ux=Ux, Uy=Uy, Uz=Uz), Val(5), (30, 30, 30),
            Float64(π); meshwarp=warping3D)

  # MPI.Finalize()
end
# }}}

main()
