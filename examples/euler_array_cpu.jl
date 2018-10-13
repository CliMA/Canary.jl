include(joinpath(@__DIR__,"vtk.jl"))
using MPI
using Canary
using Printf: @sprintf

# {{{ constants
# note the order of the fields below is also assumed in the code.
const _nstate = 5
const _U, _V, _W, _ρ, _E = 1:_nstate
const stateid = (U = _U, V = _V, W = _W, ρ = _ρ, E = _E)

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

# {{{ Pressure
function compute_pressure!(::Val{dim}, ::Val{N}, Pressure, Q, icase) where {dim, N}

    DFloat = eltype(Q)
    Np = (N+1)^dim
    (~, ~, nelem) = size(Q)

    #Gas Constants (will be moved to a Module)
    p0=100000.0
    R_gas=287.17
    c_p=1004.67
    c_v=717.5

    if (icase == 1 || icase == 2 || icase == 3)
        @inbounds for e = 1:nelem, i = 1:Np
            Pressure[i,e] = 0
        end
    elseif (icase == 20)
        @inbounds for e = 1:nelem, i = 1:Np
            Pressure[i,e] = p0*( R_gas * Q[i,_E,e]/ p0 )^(c_p/c_v)
        end
    end
end #function Pressure2D

# {{{ cfl
function cfl(::Val{dim}, ::Val{N}, vgeo, Q, Pressure, mpicomm) where {dim, N}
  DFloat = eltype(Q)
  Np = (N+1)^dim
  (~, ~, nelem) = size(Q)
  γ=1.4
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
      ρ, U, V, P = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e], Pressure[n,e]
      ξx, ξy, ηx, ηy = vgeo[n, _ξx, e], vgeo[n, _ξy, e],
                       vgeo[n, _ηx, e], vgeo[n, _ηy, e]

      loc_dt = 2*ρ / max( abs(U*ξx + V*ξy) + ρ*sqrt(γ*P/ρ), abs(U*ηx + V*ηy)  + ρ*sqrt(γ*P/ρ) )
      dt[1] = min(dt[1], loc_dt)
    end
  end

  if dim == 3
    @inbounds for e = 1:nelem, n = 1:Np
      U, V, W = Q[n, _U, e], Q[n, _V, e], Q[n, _W, e]
      ξx, ξy, ξz = vgeo[n, _ξx, e], vgeo[n, _ξy, e], vgeo[n, _ξz, e]
      ηx, ηy, ηz = vgeo[n, _ηx, e], vgeo[n, _ηy, e], vgeo[n, _ηz, e]
      ζx, ζy, ζz = vgeo[n, _ζx, e], vgeo[n, _ζy, e], vgeo[n, _ζz, e]

      loc_dt = 2 ./ max(abs(U*ξx + V*ξy + W*ξz),
                        abs(U*ηx + V*ηy + W*ηz),
                        abs(U*ζx + V*ζy + W*ζz))
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

  if (icase == 3 || icase == 20)
       @inbounds for j = 1:length(x)
           (x[j], y[j], z[j]) = (x[j]*1000.0, y[j]*1000.0, z[j]*1000.0)
       end
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
                                    Q[n, _U, e] * Q[n, _ρ, e])
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
      UM = Q[vidM, _U, eM]
      FxM = ρM * UM

      ρP = Q[vidP, _ρ, eP]
      UP = Q[vidP, _U, eP]
      FxP = ρP * UP

      λ = max(abs(nxM * UM), abs(nxM * UP))

      F = (nxM * (FxM + FxP) + λ * (ρM - ρP)) / 2
      rhs[vidM, _ρ, eM] -= sMJ * F
    end
  end
end
# }}}

# {{{ 2-D
# Volume RHS for 2-D
function volumerhs!(::Val{2}, ::Val{N}, rhs, Q, Pressure, vgeo, D, elems, gravity) where N
    Nq = N + 1
    DFloat = eltype(Q)
    dim=2
    ~, ~, nelem = size(Q)

    Q = reshape(Q, Nq, Nq, _nstate, nelem)
    rhs = reshape(rhs, Nq, Nq, _nstate, nelem)
    vgeo = reshape(vgeo, Nq, Nq, _nvgeo, nelem)
    Pressure = reshape(Pressure, Nq, Nq, nelem)

    #Allocate local flux arrays
    fluxρ=Array{DFloat,3}(undef,dim,Nq,Nq)
    fluxU=Array{DFloat,3}(undef,dim,Nq,Nq)
    fluxV=Array{DFloat,3}(undef,dim,Nq,Nq)
    fluxE=Array{DFloat,3}(undef,dim,Nq,Nq)

    @inbounds for e = 1:nelem

        #Metric Terms
        Jac=vgeo[:,:,_MJ,e]
        ξx=vgeo[:,:,_ξx,e]
        ξy=vgeo[:,:,_ξy,e]
        ηx=vgeo[:,:,_ηx,e]
        ηy=vgeo[:,:,_ηy,e]

        #Get primitive variables and fluxes
        ρ=Q[:,:,_ρ,e]
        u=Q[:,:,_U,e] ./ ρ
        v=Q[:,:,_V,e] ./ ρ
        E=Q[:,:,_E,e]
        p=Pressure[:,:,e]

        #Compute Fluxes
        fluxρ[1,:,:]=Q[:,:,_U,e]
        fluxρ[2,:,:]=Q[:,:,_V,e]
        fluxU[1,:,:]=ρ .* u .* u + p
        fluxU[2,:,:]=ρ .* u .* v
        fluxV[1,:,:]=ρ .* v .* u
        fluxV[2,:,:]=ρ .* v .* v + p
        fluxE[1,:,:]=Q[:,:,_E,e] .* u
        fluxE[2,:,:]=Q[:,:,_E,e] .* v

        # loop of ξ-grid lines
        for j = 1:Nq, i = 1:Nq, k = 1:Nq
            rhs[i,j,_ρ,e] += D[k, i] * Jac[k,j] * (ξx[k,j] * fluxρ[1,k,j] + ξy[k,j] * fluxρ[2,k,j])
            rhs[i,j,_U,e] += D[k, i] * Jac[k,j] * (ξx[k,j] * fluxU[1,k,j] + ξy[k,j] * fluxU[2,k,j])
            rhs[i,j,_V,e] += D[k, i] * Jac[k,j] * (ξx[k,j] * fluxV[1,k,j] + ξy[k,j] * fluxV[2,k,j])
            rhs[i,j,_E,e] += D[k, i] * Jac[k,j] * (ξx[k,j] * fluxE[1,k,j] + ξy[k,j] * fluxE[2,k,j])
        end
        # loop of η-grid lines
        for i = 1:Nq, j = 1:Nq, k = 1:Nq
            rhs[i,j,_ρ,e] += D[k, j] * Jac[i,k] * (ηx[i,k] * fluxρ[1,i,k] + ηy[i,k] * fluxρ[2,i,k])
            rhs[i,j,_U,e] += D[k, j] * Jac[i,k] * (ηx[i,k] * fluxU[1,i,k] + ηy[i,k] * fluxU[2,i,k])
            rhs[i,j,_V,e] += D[k, j] * Jac[i,k] * (ηx[i,k] * fluxV[1,i,k] + ηy[i,k] * fluxV[2,i,k])
            rhs[i,j,_E,e] += D[k, j] * Jac[i,k] * (ηx[i,k] * fluxE[1,i,k] + ηy[i,k] * fluxE[2,i,k])
        end

        # buoyancy term
        for i = 1:Nq, j = 1:Nq
            rhs[i,j,_V,e] -= Jac[i,j] * ρ[i,j] * gravity
        end
    end
end

# Face RHS for 2-D
function facerhs!(::Val{2}, ::Val{N}, rhs, Q, Pressure, sgeo, elems, boundary, vmapM, vmapP) where N

    DFloat = eltype(Q)
    Np = (N+1)^2
    Nfp = N+1
    nface = 4
    (~, ~, nelem) = size(Q)
    dim=2

    p0=100000.0
    R_gas=287.17
    c_p=1004.67
    c_v=717.5
    γ=1.4

    #Allocate local flux arrays
    fluxρM=Array{DFloat,1}(undef,dim)
    fluxUM=Array{DFloat,1}(undef,dim)
    fluxVM=Array{DFloat,1}(undef,dim)
    fluxEM=Array{DFloat,1}(undef,dim)
    fluxρP=Array{DFloat,1}(undef,dim)
    fluxUP=Array{DFloat,1}(undef,dim)
    fluxVP=Array{DFloat,1}(undef,dim)
    fluxEP=Array{DFloat,1}(undef,dim)

    @inbounds for e = 1:nelem
        for f = 1:nface

            #Check Boundary Condition
            bc=boundary[f,e]

            for n = 1:Nfp
                (nxM, nyM, ~, sMJ, ~) = sgeo[:, n, f, e]
                idM, idP = vmapM[n, f, e], vmapP[n, f, e]

                eM, eP = e, ((idP - 1) ÷ Np) + 1
                vidM, vidP = ((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1

                #Compute Quantities on Left side
                ρM = Q[vidM, _ρ, eM]
                UM = Q[vidM, _U, eM]
                VM = Q[vidM, _V, eM]
                EM = Q[vidM, _E, eM]
                uM = UM / ρM
                vM = VM / ρM
                pM = Pressure[vidM,eM]

                #Left Fluxes
                fluxρM[1] = UM
                fluxρM[2] = VM
                fluxUM[1] = ρM * uM * uM + pM
                fluxUM[2] = ρM * uM * vM
                fluxVM[1] = ρM * vM * uM
                fluxVM[2] = ρM * vM * vM + pM
                fluxEM[1] = EM * uM
                fluxEM[2] = EM * vM

                #Compute Quantities on Right side
                ρP = Q[vidP, _ρ, eP]
                UP = Q[vidP, _U, eP]
                VP = Q[vidP, _V, eP]
                EP = Q[vidP, _E, eP]

                if bc == 0 #no boundary or periodic
                    pP = Pressure[vidP,eP]
                elseif bc == 1 #No-flux
                    Unormal=nxM * UM + nyM * VM
                    UP = UM - 2 * Unormal * nxM
                    VP = VM - 2 * Unormal * nyM
                    ρP = ρM
                    pP = pM
                    EP = EM
                    pP = pM
                end
                uP = UP / ρP
                vP = VP / ρP

                #Right Fluxes
                fluxρP[1] = UP
                fluxρP[2] = VP
                fluxUP[1] = ρP * uP * uP + pP
                fluxUP[2] = ρP * uP * vP
                fluxVP[1] = ρP * vP * uP
                fluxVP[2] = ρP * vP * vP + pP
                fluxEP[1] = EP * uP
                fluxEP[2] = EP * vP

                #Compute wave speed
                λM=abs(nxM * uM + nyM * vM) + sqrt(γ * pM / ρM)
                λP=abs(nxM * uP + nyM * vP) + sqrt(γ * pP / ρP)
                λ = max( λM, λP )

                #Compute Numerical Flux and Update
                fluxρ_star = (nxM * (fluxρM[1] + fluxρP[1]) + nyM * (fluxρM[2] + fluxρP[2]) - λ * (ρP - ρM)) / 2
                fluxU_star = (nxM * (fluxUM[1] + fluxUP[1]) + nyM * (fluxUM[2] + fluxUP[2]) - λ * (UP - UM)) / 2
                fluxV_star = (nxM * (fluxVM[1] + fluxVP[1]) + nyM * (fluxVM[2] + fluxVP[2]) - λ * (VP - VM)) / 2
                fluxE_star = (nxM * (fluxEM[1] + fluxEP[1]) + nyM * (fluxEM[2] + fluxEP[2]) - λ * (EP - EM)) / 2

                #Update RHS
                rhs[vidM, _ρ, eM] -= sMJ * fluxρ_star
                rhs[vidM, _U, eM] -= sMJ * fluxU_star
                rhs[vidM, _V, eM] -= sMJ * fluxV_star
                rhs[vidM, _E, eM] -= sMJ * fluxE_star
            end
        end
    end
end
# }}}

# {{{ 3-D
# Volume RHS for 3-D
function volumerhs!(::Val{3}, ::Val{N}, rhs, Q, Pressure, vgeo, D, elems, gravity) where N

    Nq = N + 1
    DFloat = eltype(Q)
    dim=3
    ~, ~, nelem = size(Q)

    Q = reshape(Q, Nq, Nq, Nq, _nstate, nelem)
    rhs = reshape(rhs, Nq, Nq, Nq, _nstate, nelem)
    vgeo = reshape(vgeo, Nq, Nq, Nq, _nvgeo, nelem)
    Pressure = reshape(Pressure, Nq, Nq, Nq, nelem)

    #Allocate local flux arrays
    fluxρ=Array{DFloat,4}(undef,dim,Nq,Nq,Nq)
    fluxU=Array{DFloat,4}(undef,dim,Nq,Nq,Nq)
    fluxV=Array{DFloat,4}(undef,dim,Nq,Nq,Nq)
    fluxW=Array{DFloat,4}(undef,dim,Nq,Nq,Nq)
    fluxE=Array{DFloat,4}(undef,dim,Nq,Nq,Nq)

    @inbounds for e in elems

        #Metric Terms
        Jac=vgeo[:,:,:,_MJ,e]
        ξx=vgeo[:,:,:,_ξx,e]
        ξy=vgeo[:,:,:,_ξy,e]
        ξz=vgeo[:,:,:,_ξz,e]
        ηx=vgeo[:,:,:,_ηx,e]
        ηy=vgeo[:,:,:,_ηy,e]
        ηz=vgeo[:,:,:,_ηz,e]
        ζx=vgeo[:,:,:,_ζx,e]
        ζy=vgeo[:,:,:,_ζy,e]
        ζz=vgeo[:,:,:,_ζz,e]

        #Get primitive variables and fluxes
        ρ=Q[:,:,:,_ρ,e]
        u=Q[:,:,:,_U,e] ./ ρ
        v=Q[:,:,:,_V,e] ./ ρ
        w=Q[:,:,:,_W,e] ./ ρ
        E=Q[:,:,:,_E,e]
        p=Pressure[:,:,:,e]

        #Compute Fluxes
        fluxρ[1,:,:,:]=Q[:,:,:,_U,e]
        fluxρ[2,:,:,:]=Q[:,:,:,_V,e]
        fluxρ[3,:,:,:]=Q[:,:,:,_W,e]
        fluxU[1,:,:,:]=ρ .* u .* u + p
        fluxU[2,:,:,:]=ρ .* u .* v
        fluxU[3,:,:,:]=ρ .* u .* w
        fluxV[1,:,:,:]=ρ .* v .* u
        fluxV[2,:,:,:]=ρ .* v .* v + p
        fluxV[3,:,:,:]=ρ .* v .* w
        fluxW[1,:,:,:]=ρ .* w .* u
        fluxW[2,:,:,:]=ρ .* w .* v
        fluxW[3,:,:,:]=ρ .* w .* w + p
        fluxE[1,:,:,:]=Q[:,:,:,_E,e] .* u
        fluxE[2,:,:,:]=Q[:,:,:,_E,e] .* v
        fluxE[3,:,:,:]=Q[:,:,:,_E,e] .* w

        # loop of ξ-grid lines
        for k = 1:Nq, j = 1:Nq, i = 1:Nq,  n = 1:Nq
            rhs[i, j, k, _ρ, e] += D[n,i] * Jac[n,j,k] *
                                   (ξx[n,j,k] * fluxρ[1,n,j,k] + ξy[n,j,k] * fluxρ[2,n,j,k] + ξz[n,j,k] * fluxρ[3,n,j,k])
            rhs[i, j, k, _U, e] += D[n,i] * Jac[n,j,k] *
                                   (ξx[n,j,k] * fluxU[1,n,j,k] + ξy[n,j,k] * fluxU[2,n,j,k] + ξz[n,j,k] * fluxU[3,n,j,k])
            rhs[i, j, k, _V, e] += D[n,i] * Jac[n,j,k] *
                                   (ξx[n,j,k] * fluxV[1,n,j,k] + ξy[n,j,k] * fluxV[2,n,j,k] + ξz[n,j,k] * fluxV[3,n,j,k])
            rhs[i, j, k, _W, e] += D[n,i] * Jac[n,j,k] *
                                   (ξx[n,j,k] * fluxW[1,n,j,k] + ξy[n,j,k] * fluxW[2,n,j,k] + ξz[n,j,k] * fluxW[3,n,j,k])
            rhs[i, j, k, _E, e] += D[n,i] * Jac[n,j,k] *
                                   (ξx[n,j,k] * fluxE[1,n,j,k] + ξy[n,j,k] * fluxE[2,n,j,k] + ξz[n,j,k] * fluxE[3,n,j,k])
        end

        # loop of η-grid lines
        for k = 1:Nq, i = 1:Nq, j = 1:Nq, n = 1:Nq
            rhs[i, j, k, _ρ, e] += D[n,j] *  Jac[i,n,k] *
                                   (ηx[i,n,k] * fluxρ[1,i,n,k] + ηy[i,n,k] * fluxρ[2,i,n,k] + ηz[i,n,k] * fluxρ[3,i,n,k])
            rhs[i, j, k, _U, e] += D[n,j] *  Jac[i,n,k] *
                                   (ηx[i,n,k] * fluxU[1,i,n,k] + ηy[i,n,k] * fluxU[2,i,n,k] + ηz[i,n,k] * fluxU[3,i,n,k])
            rhs[i, j, k, _V, e] += D[n,j] *  Jac[i,n,k] *
                                   (ηx[i,n,k] * fluxV[1,i,n,k] + ηy[i,n,k] * fluxV[2,i,n,k] + ηz[i,n,k] * fluxV[3,i,n,k])
            rhs[i, j, k, _W, e] += D[n,j] *  Jac[i,n,k] *
                                   (ηx[i,n,k] * fluxW[1,i,n,k] + ηy[i,n,k] * fluxW[2,i,n,k] + ηz[i,n,k] * fluxW[3,i,n,k])
            rhs[i, j, k, _E, e] += D[n,j] *  Jac[i,n,k] *
                                   (ηx[i,n,k] * fluxE[1,i,n,k] + ηy[i,n,k] * fluxE[2,i,n,k] + ηz[i,n,k] * fluxE[3,i,n,k])
        end

        # loop of ζ-grid lines
        for j = 1:Nq, i = 1:Nq, k = 1:Nq, n = 1:Nq
            rhs[i, j, k, _ρ, e] += D[n,k]  *  Jac[i,j,n] *
                                   (ζx[i,j,n] * fluxρ[1,i,j,n] + ζy[i,j,n] * fluxρ[2,i,j,n] + ζz[i,j,n] * fluxρ[3,i,j,n])
            rhs[i, j, k, _U, e] += D[n,k]  *  Jac[i,j,n] *
                                   (ζx[i,j,n] * fluxU[1,i,j,n] + ζy[i,j,n] * fluxU[2,i,j,n] + ζz[i,j,n] * fluxU[3,i,j,n])
            rhs[i, j, k, _V, e] += D[n,k]  *  Jac[i,j,n] *
                                   (ζx[i,j,n] * fluxV[1,i,j,n] + ζy[i,j,n] * fluxV[2,i,j,n] + ζz[i,j,n] * fluxV[3,i,j,n])
            rhs[i, j, k, _W, e] += D[n,k]  *  Jac[i,j,n] *
                                   (ζx[i,j,n] * fluxW[1,i,j,n] + ζy[i,j,n] * fluxW[2,i,j,n] + ζz[i,j,n] * fluxW[3,i,j,n])
            rhs[i, j, k, _E, e] += D[n,k]  *  Jac[i,j,n] *
                                   (ζx[i,j,n] * fluxE[1,i,j,n] + ζy[i,j,n] * fluxE[2,i,j,n] + ζz[i,j,n] * fluxE[3,i,j,n])
        end

        # buoyancy term
        for i = 1:Nq, j = 1:Nq, k = 1:Nq
            rhs[i,j,k,_W,e] -= Jac[i,j,k] * ρ[i,j,k] * gravity
        end

    end
end

# Face RHS for 3-D
function facerhs!(::Val{3}, ::Val{N}, rhs, Q, Pressure, sgeo, elems, boundary, vmapM, vmapP) where N

    DFloat = eltype(Q)
    Np = (N+1)^3
    Nfp = (N+1)^2
    nface = 6
    (~, ~, nelem) = size(Q)
    dim=3

    p0=100000.0
    R_gas=287.17
    c_p=1004.67
    c_v=717.5
    γ=1.4

    #Allocate local flux arrays
    fluxρM=Array{DFloat,1}(undef,dim)
    fluxUM=Array{DFloat,1}(undef,dim)
    fluxVM=Array{DFloat,1}(undef,dim)
    fluxWM=Array{DFloat,1}(undef,dim)
    fluxEM=Array{DFloat,1}(undef,dim)
    fluxρP=Array{DFloat,1}(undef,dim)
    fluxUP=Array{DFloat,1}(undef,dim)
    fluxVP=Array{DFloat,1}(undef,dim)
    fluxWP=Array{DFloat,1}(undef,dim)
    fluxEP=Array{DFloat,1}(undef,dim)

    @inbounds for e in elems
        for f = 1:nface
            for n = 1:Nfp

                #Check Boundary Condition
                bc=boundary[f,e]

                (nxM, nyM, nzM, sMJ, ~) = sgeo[:, n, f, e]
                idM, idP = vmapM[n, f, e], vmapP[n, f, e]

                eM, eP = e, ((idP - 1) ÷ Np) + 1
                vidM, vidP = ((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1

                #Compute Quantities on Left side
                ρM = Q[vidM, _ρ, eM]
                UM = Q[vidM, _U, eM]
                VM = Q[vidM, _V, eM]
                WM = Q[vidM, _W, eM]
                EM = Q[vidM, _E, eM]
                uM = UM / ρM
                vM = VM / ρM
                wM = WM / ρM
                pM = Pressure[vidM,eM]

                #Left Fluxes
                fluxρM[1] = UM
                fluxρM[2] = VM
                fluxρM[3] = WM
                fluxUM[1] = ρM * uM * uM + pM
                fluxUM[2] = ρM * uM * vM
                fluxUM[3] = ρM * uM * wM
                fluxVM[1] = ρM * vM * uM
                fluxVM[2] = ρM * vM * vM + pM
                fluxVM[3] = ρM * vM * wM
                fluxWM[1] = ρM * wM * uM
                fluxWM[2] = ρM * wM * vM
                fluxWM[3] = ρM * wM * wM + pM
                fluxEM[1] = EM * uM
                fluxEM[2] = EM * vM
                fluxEM[3] = EM * wM

                #Compute Quantities on Right side
                ρP = Q[vidP, _ρ, eP]
                UP = Q[vidP, _U, eP]
                VP = Q[vidP, _V, eP]
                WP = Q[vidP, _W, eP]
                EP = Q[vidP, _E, eP]

                if bc == 0 #no boundary or periodic
                    pP = Pressure[vidP,eP]
                elseif bc == 1 #No-flux
                    Unormal=nxM * UM + nyM * VM + nzM * WM
                    UP = UM - 2 * Unormal * nxM
                    VP = VM - 2 * Unormal * nyM
                    WP = WM - 2 * Unormal * nzM
                    ρP = ρM
                    pP = pM
                    EP = EM
                    pP = pM
                end
                uP = UP / ρP
                vP = VP / ρP
                wP = WP / ρP

                #Right Fluxes
                fluxρP[1] = UP
                fluxρP[2] = VP
                fluxρP[3] = WP
                fluxUP[1] = ρP * uP * uP + pP
                fluxUP[2] = ρP * uP * vP
                fluxUP[3] = ρP * uP * wP
                fluxVP[1] = ρP * vP * uP
                fluxVP[2] = ρP * vP * vP + pP
                fluxVP[3] = ρP * vP * wP
                fluxWP[1] = ρP * wP * uP
                fluxWP[2] = ρP * wP * vP
                fluxWP[3] = ρP * wP * wP + pP
                fluxEP[1] = EP * uP
                fluxEP[2] = EP * vP
                fluxEP[3] = EP * wP

                #Compute wave speed
                λM=abs(nxM * uM + nyM * vM + nzM * wM) + sqrt(γ * pM / ρM)
                λP=abs(nxM * uP + nyM * vP + nzM * wP) + sqrt(γ * pP / ρP)
                λ = max( λM, λP )

                #Compute Numerical Flux and Update
                fluxρ_star = (nxM * (fluxρM[1] + fluxρP[1]) + nyM * (fluxρM[2] + fluxρP[2]) + nzM * (fluxρM[3] + fluxρP[3]) - λ * (ρP - ρM)) / 2
                fluxU_star = (nxM * (fluxUM[1] + fluxUP[1]) + nyM * (fluxUM[2] + fluxUP[2]) + nzM * (fluxUM[3] + fluxUP[3]) - λ * (UP - UM)) / 2
                fluxV_star = (nxM * (fluxVM[1] + fluxVP[1]) + nyM * (fluxVM[2] + fluxVP[2]) + nzM * (fluxVM[3] + fluxVP[3]) - λ * (VP - VM)) / 2
                fluxW_star = (nxM * (fluxWM[1] + fluxWP[1]) + nyM * (fluxWM[2] + fluxWP[2]) + nzM * (fluxWM[3] + fluxWP[3]) - λ * (WP - WM)) / 2
                fluxE_star = (nxM * (fluxEM[1] + fluxEP[1]) + nyM * (fluxEM[2] + fluxEP[2]) + nzM * (fluxEM[3] + fluxEP[3]) - λ * (EP - EM)) / 2

                #Update RHS
                rhs[vidM, _ρ, eM] -= sMJ * fluxρ_star
                rhs[vidM, _U, eM] -= sMJ * fluxU_star
                rhs[vidM, _V, eM] -= sMJ * fluxV_star
                rhs[vidM, _W, eM] -= sMJ * fluxW_star
                rhs[vidM, _E, eM] -= sMJ * fluxE_star
            end
        end
    end
end
# }}}

# {{{ Update solution (for all dimensions)
function updatesolution!(::Val{dim}, ::Val{N}, rhs, Q, vgeo, elems, rka, rkb, dt, icase) where {dim, N}

    DFloat = eltype(Q)
    Nq=(N+1)^dim
    (~, ~, nelem) = size(Q)

    #Store Velocity
    if (icase == 1 || icase == 2 || icase == 3)

        #Allocate local arrays
        ρ=Array{DFloat,2}(undef,Nq,nelem)
        u=Array{DFloat,2}(undef,Nq,nelem)
        v=Array{DFloat,2}(undef,Nq,nelem)
        w=Array{DFloat,2}(undef,Nq,nelem)
        energy=Array{DFloat,2}(undef,Nq,nelem)

        @inbounds for e = elems, i = 1:Nq
            ρ[i, e] = Q[i,_ρ,e]
            u[i, e] = Q[i,_U,e] / ρ[i,e]
            v[i, e] = Q[i,_V,e] / ρ[i,e]
            w[i, e] = Q[i,_W,e] / ρ[i,e]
        end
    end

    @inbounds for e = elems, s = 1:_nstate, i = 1:Nq
        Q[i, s, e] += rkb * dt * rhs[i, s, e] * vgeo[i, _MJI, e]
        rhs[i, s, e] *= rka
    end

    #Reset Velocity
    if (icase == 1 || icase == 2 || icase == 3)
        @inbounds for e = elems, i = 1:Nq
            ρ[i,e] = Q[i,_ρ,e]
            Q[i,_U,e] = ρ[i,e] * u[i,e]
            Q[i,_V,e] = ρ[i,e] * v[i,e]
            Q[i,_W,e] = ρ[i,e] * w[i,e]
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
      #    diff = Q[i, _ρ, e] - Qex.ρ(X...)
    diff = Q[i, _ρ, e] - Qex[i, _ρ, e]

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
function lowstorageRK(::Val{dim}, ::Val{N}, mesh, vgeo, sgeo, Q, rhs, Pressure, D,
                      dt, nsteps, tout, vmapM, vmapP, mpicomm, iplot, icase, gravity) where {dim, N}
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

            # compute Pressure
            compute_pressure!(Val(dim), Val(N), Pressure, Q, icase)

            # volume RHS computation
            volumerhs!(Val(dim), Val(N), rhs, Q, Pressure, vgeo, D, mesh.realelems, gravity)

            # wait on MPI receives
            MPI.Waitall!(recvreq)

            # copy data to state vectors
            Q[:, :, nrealelem+1:end] .= recvQ[:, :, :]

            # face RHS computation
            facerhs!(Val(dim), Val(N), rhs, Q, Pressure, sgeo, mesh.realelems, mesh.elemtobndy, vmapM, vmapP)

            # update solution and scale RHS
            updatesolution!(Val(dim), Val(N), rhs, Q, vgeo, mesh.realelems, RKA[s%length(RKA)+1], RKB[s], dt, icase)
        end #s-stages

        #Plot VTK
        if mpirank == 0 && mod(step,iplot) == 0
            println("step=",step," time=",time)
        end
        if dim > 1 && mod(step,iplot) == 0
            (nface, nelem) = size(mesh.elemtoelem)
            X = ntuple(j->reshape((@view vgeo[:, _x+j-1, :]), ntuple(j->N+1,dim)...,
                                  nelem), dim)
            ρ = reshape((@view Q[:, _ρ, :]), ntuple(j->(N+1),dim)..., nelem)
            U = reshape((@view Q[:, _U, :]), ntuple(j->(N+1),dim)..., nelem) ./ ρ
            V = reshape((@view Q[:, _V, :]), ntuple(j->(N+1),dim)..., nelem) ./ ρ
            W = reshape((@view Q[:, _W, :]), ntuple(j->(N+1),dim)..., nelem) ./ ρ
            θ = reshape((@view Q[:, _E, :]), ntuple(j->(N+1),dim)..., nelem)
            P = reshape((@view Pressure[:, :]), ntuple(j->(N+1),dim)..., nelem)
            if icase == 20
                θ = θ ./ ρ .- 300.0
            end
            writemesh(@sprintf("viz/euler%dD_rank_%04d_step_%05d", dim,
                               mpirank, step), X...; fields=(("ρ", ρ),("U",U),("V",V),("W",W),("θ", θ),("P", P),), realelems=mesh.realelems)
        end

    end #dt
end #end lowStorageRK
# }}}

# {{{ Euler driver
function euler(mpicomm, ic, ::Val{N}, brickN::NTuple{dim, Int}, tend, iplot, icase, gravity;
                   meshwarp=(x...)->identity(x), tout = 60) where {N, dim}
  DFloat = Float64
  mpirank = MPI.Comm_rank(mpicomm)
  mpisize = MPI.Comm_size(mpicomm)

  # Generate a local view of a fully periodic Cartesian mesh.
  mesh = brickmesh(ntuple(i->range(DFloat(0); length=brickN[i]+1, stop=1), dim),
                   (fill(true, dim)...,);
                   part=mpirank+1, numparts=mpisize)

  if (icase == 20)
      mesh = brickmesh(ntuple(i->range(DFloat(0); length=brickN[i]+1, stop=1), dim),
                       (fill(true, dim-1)...,false);
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
  Pressure = zeros(DFloat, (N+1)^dim, nelem)

  # setup the initial condition
  @inbounds for e = 1:nelem, i = 1:(N+1)^dim
      x, y, z = vgeo[i, _x, e], vgeo[i, _y, e], vgeo[i, _z, e]
      ρ, U, V, W, E = ic(icase,gravity,dim,x, y, z)
      Q[i, _ρ, e] = ρ
      Q[i, _U, e] = U
      Q[i, _V, e] = V
      Q[i, _W, e] = W
      Q[i, _E, e] = E
      Qexact[i, _ρ, e] = ρ
      Qexact[i, _U, e] = U
      Qexact[i, _V, e] = V
      Qexact[i, _W, e] = W
      Qexact[i, _E, e] = E
  end

  #Need Pressure for the speed of sound
  compute_pressure!(Val(dim), Val(N), Pressure, Q, icase)

  # plot the initial condition
  mkpath("viz")
  # TODO: Fix VTK for 1-D
  if dim > 1
    X = ntuple(j->reshape((@view vgeo[:, _x+j-1, :]), ntuple(j->N+1,dim)...,
                          nelem), dim)
    ρ = reshape((@view Q[:, _ρ, :]), ntuple(j->(N+1),dim)..., nelem)
    U = reshape((@view Q[:, _U, :]), ntuple(j->(N+1),dim)..., nelem) ./ ρ
    V = reshape((@view Q[:, _V, :]), ntuple(j->(N+1),dim)..., nelem) ./ ρ
    W = reshape((@view Q[:, _W, :]), ntuple(j->(N+1),dim)..., nelem) ./ ρ
    θ = reshape((@view Q[:, _E, :]), ntuple(j->(N+1),dim)..., nelem)
    P = reshape((@view Pressure[:, :]), ntuple(j->(N+1),dim)..., nelem)
      if icase == 20
        θ = θ ./ ρ .- 300.0
    end
    writemesh(@sprintf("viz/euler%dD_rank_%04d_step_%05d", dim,
                       mpirank, 0), X...; fields=(("ρ", ρ),("U",U),("V",V),("W",W),("θ", θ),("P", P),), realelems=mesh.realelems)
  end

  # Compute time step
  dt = cfl(Val(dim), Val(N), vgeo, Q, Pressure, mpicomm) / N^√2
  #dt=0.025 #for case 20 with N=4 10x10x1
  dt=0.02

  nsteps = ceil(Int64, tend / dt)
  dt = tend / nsteps
  mpirank == 0 && @show (dt, nsteps)

  # Do time stepping
  stats = zeros(DFloat, 3)
  stats[1] = L2energysquared(Val(dim), Val(N), Q, vgeo, mesh.realelems)

  lowstorageRK(Val(dim), Val(N), mesh, vgeo, sgeo, Q, rhs, Pressure, D, dt, nsteps,
               tout, vmapM, vmapP, mpicomm, iplot, icase, gravity)

  # plot the final Solution
  # TODO: Fix VTK for 1-D
  if dim > 1
    X = ntuple(j->reshape((@view vgeo[:, _x+j-1, :]), ntuple(j->N+1,dim)...,
                          nelem), dim)
    ρ = reshape((@view Q[:, _ρ, :]), ntuple(j->(N+1),dim)..., nelem)
    U = reshape((@view Q[:, _U, :]), ntuple(j->(N+1),dim)..., nelem) ./ ρ
    V = reshape((@view Q[:, _V, :]), ntuple(j->(N+1),dim)..., nelem) ./ ρ
    W = reshape((@view Q[:, _W, :]), ntuple(j->(N+1),dim)..., nelem) ./ ρ
    θ = reshape((@view Q[:, _E, :]), ntuple(j->(N+1),dim)..., nelem)
    P = reshape((@view Pressure[:, :]), ntuple(j->(N+1),dim)..., nelem)
    if icase == 20
        θ = θ ./ ρ .- 300.0
    end
    writemesh(@sprintf("viz/euler%dD_rank_%04d_step_%05d", dim,
                       mpirank, nsteps), X...; fields=(("ρ", ρ),("U",U),("V",V),("W",W),("θ", θ),("P", P),), realelems=mesh.realelems)
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

  warping1D(x...) = (x[1] +  sin( π*x[1])/10, zero(x[1]), zero(x[1]))
#  warping2D(x...) = (x[1] +  sin( π*x[1])*sin(2π*x[2])/10,
#                     x[2] +  sin(2π*x[1])*sin( π*x[2])/10,
#                     zero(x[1]))
  warping2D(x...) = (x[1], x[2], zero(x[1]))
#  warping3D(x...) = (x[1] + (sin( π*x[1])*sin(2π*x[2])*cos(2π*x[3]))/10,
#                     x[2] + (sin( π*x[2])*sin(2π*x[1])*cos(2π*x[3]))/10,
#                     x[3] + (sin( π*x[3])*sin(2π*x[1])*cos(2π*x[2]))/10)
  warping3D(x...) = (x[1], x[2], x[3])

  #Input Parameters
  N=4
  Ne=10
  iplot=5000
  dim=3

    #Cases
    #=
    icase=1
    gravity=0
    time_final=1.0
    =#

    #=
    icase=2
    time_final=1.0
    icase=3
    time_final=0.5
    =#

    icase=20
    time_final=300.0
    gravity=10.0

  #Initial Conditions
  function ic(icase,gravity,dim,x...)
      if icase == 1
          #          r = sqrt( (x[1]-0.5)^2 + (x[2]-0.5)^2 )
          r = sqrt( (x[1]-0.5)^2 + (x[3]-0.5)^2 )
          ρ = 0.5 * exp(-100.0 * r^2) + 1.0
          U = ρ*(0.0)
          V = ρ*(0.0)
          W = ρ*(1.0)
          E = ρ*(1.0)
          ρ, U, V, W, E
      elseif icase == 2
          r = sqrt( (x[1]-0.5)^2 + (x[2]-0.5)^2 )
          ρ = 0.5 * exp(-100.0 * r^2) + 1.0
          U = ρ*(1.0)
          V = ρ*(0.0)
          W = ρ*(0.0)
          E = ρ*(1.0)
          ρ, U, V, W, E
      elseif icase == 3
          R_gas=287.17
          c_p=1004.67
          c_v=717.5
          p0=100000.0
          u0=1000.0
          c=c_v/R_gas
          r = sqrt( (x[1]-500)^2 + (x[dim]-350)^2 )
          rc = 250.0
          θ_ref=300.0
          θ_c=0.5
          Δθ=0.0
          if r <= rc
              Δθ = 0.5 * θ_c * (1.0 + cos(π * r/rc))
          end
          θ_k=θ_ref + Δθ
          π_k=1.0 - gravity/(c_p*θ_k)*x[dim]
          ρ_k=p0/(R_gas*θ_k)*(π_k)^c
          ρ_k=Δθ + 1.0
          ρ = ρ_k
          U = ρ*(0.0)
          V = ρ*(0.0)
          W = ρ*(u0)
          E = ρ*(1.0)
          ρ, U, V, W, E
      elseif icase == 20
          R_gas=287.17
          c_p=1004.67
          c_v=717.5
          p0=100000.0
          u0=0.0
          c=c_v/R_gas
          r = sqrt( (x[1]-500)^2 + (x[dim]-350)^2 )
          rc = 250.0
          θ_ref=300.0
          θ_c=0.5
          Δθ=0.0
          if r <= rc
              Δθ = 0.5 * θ_c * (1.0 + cos(π * r/rc))
          end
          θ_k=θ_ref + Δθ
          π_k=1.0 - gravity/(c_p*θ_k)*x[dim]
          ρ_k=p0/(R_gas*θ_k)*(π_k)^c
          ρ = ρ_k
          U = ρ*(u0)
          V = ρ*(0.0)
          W = ρ*(0.0)
          E = ρ*θ_k
          ρ, U, V, W, E
      end
  end

    #call euler
    if dim == 2
        mpirank == 0 && println("Running 2d...")
        euler(mpicomm, ic, Val(N), (Ne, Ne), time_final, iplot, icase, gravity; meshwarp=warping2D)
        mpirank == 0 && println()
    elseif dim == 3
        mpirank == 0 && println("Running 3d...")
        euler(mpicomm, ic, Val(N), (Ne, 1, Ne), time_final, iplot, icase, gravity; meshwarp=warping3D)
        mpirank == 0 && println()
    end
#  MPI.Finalize()
end
# }}}

main()
