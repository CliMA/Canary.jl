#--------------------------------Markdown Language Header-----------------------
# # 3D Euler Equations Based on Total Energy
#
#
#-
#
#-
# ## Introduction
#
# This example shows how to solve the 3D Euler equations using vanilla DG.
#
# ## Continuous Governing Equations
# We solve the following equation:
#
# ```math
# \frac{\partial \rho}{\partial t} + \nabla \cdot \mathbf{U} = \nu \nabla^2 \rho \; \; (1.1)
# ```
# ```math
# \frac{\partial \mathbf{U}}{\partial t} + \nabla \cdot \left( \frac{\mathbf{U} \otimes \mathbf{U}}{\rho} + P \mathbf{I}_2 \right) + \rho g \hat{\mathbf{k}}= \nu \nabla^2 \mathbf{U} \; \; (1.2)
# ```
# ```math
# \frac{\partial E}{\partial t} + \nabla \cdot \left( \frac{\mathbf{U} \left(E+P \right)}{\rho} \right = \nu \nabla^2 E \; \; (1.3)
# ```
# where $\mathbf{u}=(u,v,w)$ is the velocity, $\mathbf{U}=\rho \mathbf{u}$, is the momentum, with $\rho$ being the total density and $E=(\gamma-1) \rho \left( c_v T + \frac{1}{2} \mathbf{u} \cdot \mathbf{u} + g z \right)$ is the total energy (internal $+$ kinetic $+$ potential).
# In addition, $\nu$ is the artificial viscosity parameter. We employ periodic boundary conditions in the x-y directions and no-flux boundary conditions in the z-direction.
#
#-
# ## Discontinous Galerkin Method
# To solve Eq.\ (1) we use the discontinuous Galerkin method with basis functions comprised of Lagrange polynomials based on Lobatto points. Multiplying Eq.\ (1) by a test function $\psi$ and integrating within each element $\Omega_e$ such that $\Omega = \bigcup_{e=1}^{N_e} \Omega_e$ we get
#
# ```math
# \int_{\Omega_e} \psi \frac{\partial \mathbf{q}^{(e)}_N}{\partial t} d\Omega_e + \int_{\Omega_e} \psi \nabla \cdot \mathbf{F}^{(e)}_N d\Omega_e =  \int_{\Omega_e} \psi S\left( q^{(e)}_N} \right) d\Omega_e \; \; (2)
# ```
# where $\mathbf{q}^{(e)}_N=\sum_{i=1}^{(N+1)^{dim}} \psi_i(\mathbf{x}) \mathbf{q}_i(t)$ is the finite dimensional expansion with basis functions $\psi(\mathbf{x})$, where $\mathbf{q}=\left( \rho, \mathbf{U}^T, E \right)^T$,
# ```math
# \mathbf{F}=\left( \mathbf{U}, \frac{\mathbf{U} \otimes \mathbf{U}}{\rho} + P \mathbf{I}_2,   \frac{\mathbf{U} \left(E+P \right)}{\rho} \right).
# ```
# and
# ```math
#  S\left( q^{(e)}_N} \right)  = \nu \left( \nabla^2 \rho, \nabla^2 \mathbf{U}, \nabla^2 E \right).
# ```
# Integrating Eq.\ (2) by parts yields
#
# ```math
# \int_{\Omega_e} \psi \frac{\partial \mathbf{q}^{(e)}_N}{\partial t} d\Omega_e + \int_{\Gamma_e} \psi \mathbf{n} \cdot \mathbf{F}^{(*,e)}_N d\Gamma_e - \int_{\Omega_e} \nabla \psi \cdot \mathbf{F}^{(e)}_N d\Omega_e = \int_{\Omega_e} \psi S\left( q^{(e)}_N} \right) d\Omega_e \; \; (3)
# ```
#
# where the second term on the left denotes the flux integral term (computed in "function fluxrhs") and the third term denotes the volume integral term (computed in "function volumerhs").  The superscript $(*,e)$ in the flux integral term denotes the numerical flux. Here we use the Rusanov flux.
#
#-
# ## Local Discontinous Galerkin Method
# To approximate the second order terms on the right hand side of Eq.\ (1) we use the local discontinuous Galerkin (LDG) method, which we described in \texttt{LDG3d.jl}. We will highlight the main steps below for completeness. The operator $\nabla^2$ is approximated by the following two-step process: first we approximate the gradient of $q$ as follows
# ```math
# \mathbf{Q}(x,y,z) = \nabla \vc{q}(x,y,z) \; \; (2)
# ```
# where $\mathbf{Q}$ is an auxiliary vector function, followed by
# ```math
# \nabla \cdot \mathbf{Q} (x,y,z) =  \nabla^2 \vc{q}(x,y,z) \; \; (3)
# ```
# which represents the Laplacian of $\vc{q}$.
#
#-
# ## Commented Program
#
#--------------------------------Markdown Language Header-----------------------
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

const _γ = 14  // 10
const _p0 = 100000
const _R_gas = 28717 // 100
const _c_p = 100467 // 100
const _c_v = 7175 // 10
const _gravity = 10
# }}}

# {{{ courant
function courantnumber(::Val{dim}, ::Val{N}, vgeo, Q, mpicomm) where {dim, N}
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
    Courant = - [floatmax(DFloat)]

    #Compute DT
    @inbounds for e = 1:nelem, n = 1:Np
        ρ, U, V, W, E = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e], Q[n, _W, e], Q[n, _E, e]
        ξx, ξy, ξz = vgeo[n, _ξx, e], vgeo[n, _ξy, e], vgeo[n, _ξz, e]
        ηx, ηy, ηz = vgeo[n, _ηx, e], vgeo[n, _ηy, e], vgeo[n, _ηz, e]
        ζx, ζy, ζz = vgeo[n, _ζx, e], vgeo[n, _ζy, e], vgeo[n, _ζz, e]
        z = vgeo[n, _z, e]
        P = (R_gas/c_v)*(E - (U^2 + V^2 + W^2)/(2*ρ) - ρ*gravity*z)
        u, v, w = U/ρ, V/ρ, W/ρ
        dx=sqrt( (1.0/(2*ξx))^2 + 0*(1.0/(2*ηy))^2  + (1.0/(2*ζz))^2 )
        vel=sqrt( u^2 + v^2 + w^2)
        wave_speed = (vel + sqrt(γ * P / ρ))
        loc_dt = 1.0*dx/wave_speed/N
        dt[1] = min(dt[1], loc_dt)
    end
    dt_min=MPI.Allreduce(dt[1], MPI.MIN, mpicomm)

    #Compute Courant
    @inbounds for e = 1:nelem, n = 1:Np
        ρ, U, V, W, E = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e], Q[n, _W, e], Q[n, _E, e]
        ξx, ξy, ξz = vgeo[n, _ξx, e], vgeo[n, _ξy, e], vgeo[n, _ξz, e]
        ηx, ηy, ηz = vgeo[n, _ηx, e], vgeo[n, _ηy, e], vgeo[n, _ηz, e]
        ζx, ζy, ζz = vgeo[n, _ζx, e], vgeo[n, _ζy, e], vgeo[n, _ζz, e]
        z = vgeo[n, _z, e]
        P = (R_gas/c_v)*(E - (U^2 + V^2 + W^2)/(2*ρ) - ρ*gravity*z)
        u, v, w = U/ρ, V/ρ, W/ρ
        dx=sqrt( (1.0/(2*ξx))^2 + 0*(1.0/(2*ηy))^2  + (1.0/(2*ζz))^2 )
        vel=sqrt( u^2 + v^2 + w^2)
        wave_speed = (vel + sqrt(γ * P / ρ))
        loc_Courant = wave_speed*dt_min*N/dx
        Courant[1] = max(Courant[1], loc_Courant)
    end
    Courant_max=MPI.Allreduce(Courant[1], MPI.MAX, mpicomm)

    (dt_min, Courant_max)
end
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
        ρ, U, V, W = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e], Q[n, _W, e]
        E = Q[n, _E, e]
        P = p0 * (R_gas * E / p0)^(c_p / c_v)

        ξx, ξy, ξz = vgeo[n, _ξx, e], vgeo[n, _ξy, e], vgeo[n, _ξz, e]
        ηx, ηy, ηz = vgeo[n, _ηx, e], vgeo[n, _ηy, e], vgeo[n, _ηz, e]
        ζx, ζy, ζz = vgeo[n, _ζx, e], vgeo[n, _ζy, e], vgeo[n, _ζz, e]

        loc_dt = 2ρ / max(abs(U * ξx + V * ξy + W * ξz) + ρ * sqrt(γ * P / ρ),
                          abs(U * ηx + V * ηy + W * ηz) + ρ * sqrt(γ * P / ρ),
                          abs(U * ζx + V * ζy + W * ζz) + ρ * sqrt(γ * P / ρ))
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
    computemetric!(x, y, z, J, ξx, ηx, ζx, ξy, ηy, ζy, ξz, ηz, ζz, sJ,
                   nx, ny, nz, D)

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
# Volume RHS
function volumerhs!(::Val{dim}, ::Val{N}, rhs::Array, Q, vgeo, D, elems) where {dim, N}
    DFloat = eltype(Q)
    γ::DFloat       = _γ
    p0::DFloat      = _p0
    R_gas::DFloat   = _R_gas
    c_p::DFloat     = _c_p
    c_v::DFloat     = _c_v
    gravity::DFloat = _gravity

    Nq = N + 1
    nelem = size(Q)[end]

    Q = reshape(Q, Nq, Nq, Nq, _nstate, nelem)
    rhs = reshape(rhs, Nq, Nq, Nq, _nstate, nelem)
    vgeo = reshape(vgeo, Nq, Nq, Nq, _nvgeo, nelem)

    s_F = Array{DFloat}(undef, Nq, Nq, Nq, _nstate)
    s_G = Array{DFloat}(undef, Nq, Nq, Nq, _nstate)
    s_H = Array{DFloat}(undef, Nq, Nq, Nq, _nstate)

    @inbounds for e in elems
        for k = 1:Nq, j = 1:Nq, i = 1:Nq
            MJ = vgeo[i, j, k, _MJ, e]
            ξx, ξy, ξz = vgeo[i,j,k,_ξx,e], vgeo[i,j,k,_ξy,e], vgeo[i,j,k,_ξz,e]
            ηx, ηy, ηz = vgeo[i,j,k,_ηx,e], vgeo[i,j,k,_ηy,e], vgeo[i,j,k,_ηz,e]
            ζx, ζy, ζz = vgeo[i,j,k,_ζx,e], vgeo[i,j,k,_ζy,e], vgeo[i,j,k,_ζz,e]
            z = vgeo[i,j,k,_z,e]

            U, V, W = Q[i, j, k, _U, e], Q[i, j, k, _V, e], Q[i, j, k, _W, e]
            ρ, E = Q[i, j, k, _ρ, e], Q[i, j, k, _E, e]
            P = (R_gas/c_v)*(E - (U^2 + V^2 + W^2)/(2*ρ) - ρ*gravity*z)

            ρinv = 1 / ρ
            fluxρ_x = U
            fluxU_x = ρinv * U * U + P
            fluxV_x = ρinv * V * U
            fluxW_x = ρinv * W * U
            fluxE_x = ρinv * U * (E+P)

            fluxρ_y = V
            fluxU_y = ρinv * U * V
            fluxV_y = ρinv * V * V + P
            fluxW_y = ρinv * W * V
            fluxE_y = ρinv * V * (E+P)

            fluxρ_z = W
            fluxU_z = ρinv * U * W
            fluxV_z = ρinv * V * W
            fluxW_z = ρinv * W * W + P
            fluxE_z = ρinv * W * (E+P)

            s_F[i, j, k, _ρ] = MJ * (ξx * fluxρ_x + ξy * fluxρ_y + ξz * fluxρ_z)
            s_F[i, j, k, _U] = MJ * (ξx * fluxU_x + ξy * fluxU_y + ξz * fluxU_z)
            s_F[i, j, k, _V] = MJ * (ξx * fluxV_x + ξy * fluxV_y + ξz * fluxV_z)
            s_F[i, j, k, _W] = MJ * (ξx * fluxW_x + ξy * fluxW_y + ξz * fluxW_z)
            s_F[i, j, k, _E] = MJ * (ξx * fluxE_x + ξy * fluxE_y + ξz * fluxE_z)

            s_G[i, j, k, _ρ] = MJ * (ηx * fluxρ_x + ηy * fluxρ_y + ηz * fluxρ_z)
            s_G[i, j, k, _U] = MJ * (ηx * fluxU_x + ηy * fluxU_y + ηz * fluxU_z)
            s_G[i, j, k, _V] = MJ * (ηx * fluxV_x + ηy * fluxV_y + ηz * fluxV_z)
            s_G[i, j, k, _W] = MJ * (ηx * fluxW_x + ηy * fluxW_y + ηz * fluxW_z)
            s_G[i, j, k, _E] = MJ * (ηx * fluxE_x + ηy * fluxE_y + ηz * fluxE_z)

            s_H[i, j, k, _ρ] = MJ * (ζx * fluxρ_x + ζy * fluxρ_y + ζz * fluxρ_z)
            s_H[i, j, k, _U] = MJ * (ζx * fluxU_x + ζy * fluxU_y + ζz * fluxU_z)
            s_H[i, j, k, _V] = MJ * (ζx * fluxV_x + ζy * fluxV_y + ζz * fluxV_z)
            s_H[i, j, k, _W] = MJ * (ζx * fluxW_x + ζy * fluxW_y + ζz * fluxW_z)
            s_H[i, j, k, _E] = MJ * (ζx * fluxE_x + ζy * fluxE_y + ζz * fluxE_z)

            # buoyancy term
            rhs[i, j, k, _W, e] -= MJ * ρ * gravity
        end

        # loop of ξ-grid lines
        for s = 1:_nstate, k = 1:Nq, j = 1:Nq, i = 1:Nq, n = 1:Nq
            rhs[i, j, k, s, e] += D[n, i] * s_F[n, j, k, s]
        end
        # loop of η-grid lines
        for s = 1:_nstate, k = 1:Nq, j = 1:Nq, i = 1:Nq, n = 1:Nq
            rhs[i, j, k, s, e] += D[n, j] * s_G[i, n, k, s]
        end
        # loop of ζ-grid lines
        for s = 1:_nstate, k = 1:Nq, j = 1:Nq, i = 1:Nq, n = 1:Nq
            rhs[i, j, k, s, e] += D[n, k] * s_H[i, j, n, s]
        end
    end
end

# flux RHS
function fluxrhs!(::Val{dim}, ::Val{N}, rhs::Array, Q, sgeo, vgeo, elems, vmapM,
                  vmapP, elemtobndy) where {dim, N}
    DFloat = eltype(Q)
    γ::DFloat       = _γ
    p0::DFloat      = _p0
    R_gas::DFloat   = _R_gas
    c_p::DFloat     = _c_p
    c_v::DFloat     = _c_v
    gravity::DFloat = _gravity

    Np = (N+1)^dim
    Nfp = (N+1)^(dim-1)
    nface = 2*dim

    @inbounds for e in elems
        for f = 1:nface
            for n = 1:Nfp
                (nxM, nyM, nzM, sMJ, ~) = sgeo[:, n, f, e]
                idM, idP = vmapM[n, f, e], vmapP[n, f, e]

                eM, eP = e, ((idP - 1) ÷ Np) + 1
                vidM, vidP = ((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1

                ρM = Q[vidM, _ρ, eM]
                UM = Q[vidM, _U, eM]
                VM = Q[vidM, _V, eM]
                WM = Q[vidM, _W, eM]
                EM = Q[vidM, _E, eM]
                zM = vgeo[vidM, _z, eM]

                bc = elemtobndy[f, e]
                PM = (R_gas/c_v)*(EM - (UM^2 + VM^2 + WM^2)/(2*ρM) - ρM*gravity*zM)
                if bc == 0
                    ρP = Q[vidP, _ρ, eP]
                    UP = Q[vidP, _U, eP]
                    VP = Q[vidP, _V, eP]
                    WP = Q[vidP, _W, eP]
                    EP = Q[vidP, _E, eP]
                    zP = vgeo[vidP, _z, eP]
                    PP = (R_gas/c_v)*(EP - (UP^2 + VP^2 + WP^2)/(2*ρP) - ρP*gravity*zP)
                elseif bc == 1
                    UnM = nxM * UM + nyM * VM + nzM * WM
                    UP = UM - 2 * UnM * nxM
                    VP = VM - 2 * UnM * nyM
                    WP = WM - 2 * UnM * nzM
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
                fluxWM_x = ρMinv * WM * UM
                fluxEM_x = ρMinv * UM * (EM+PM)

                fluxρM_y = VM
                fluxUM_y = ρMinv * UM * VM
                fluxVM_y = ρMinv * VM * VM + PM
                fluxWM_y = ρMinv * WM * VM
                fluxEM_y = ρMinv * VM * (EM+PM)

                fluxρM_z = WM
                fluxUM_z = ρMinv * UM * WM
                fluxVM_z = ρMinv * VM * WM
                fluxWM_z = ρMinv * WM * WM + PM
                fluxEM_z = ρMinv * WM * (EM+PM)

                ρPinv = 1 / ρP
                fluxρP_x = UP
                fluxUP_x = ρPinv * UP * UP + PP
                fluxVP_x = ρPinv * VP * UP
                fluxWP_x = ρPinv * WP * UP
                fluxEP_x = ρPinv * UP * (EP+PP)

                fluxρP_y = VP
                fluxUP_y = ρPinv * UP * VP
                fluxVP_y = ρPinv * VP * VP + PP
                fluxWP_y = ρPinv * WP * VP
                fluxEP_y = ρPinv * VP * (EP+PP)

                fluxρP_z = WP
                fluxUP_z = ρPinv * UP * WP
                fluxVP_z = ρPinv * VP * WP
                fluxWP_z = ρPinv * WP * WP + PP
                fluxEP_z = ρPinv * WP * (EP+PP)

                λM = ρMinv * abs(nxM * UM + nyM * VM + nzM * WM) + sqrt(ρMinv * γ * PM)
                λP = ρPinv * abs(nxM * UP + nyM * VP + nzM * WP) + sqrt(ρPinv * γ * PP)
                λ  =  max(λM, λP)

                #Compute Numerical Flux and Update
                fluxρS = (nxM * (fluxρM_x + fluxρP_x) + nyM * (fluxρM_y + fluxρP_y) +
                          nzM * (fluxρM_z + fluxρP_z) - λ * (ρP - ρM)) / 2
                fluxUS = (nxM * (fluxUM_x + fluxUP_x) + nyM * (fluxUM_y + fluxUP_y) +
                          nzM * (fluxUM_z + fluxUP_z) - λ * (UP - UM)) / 2
                fluxVS = (nxM * (fluxVM_x + fluxVP_x) + nyM * (fluxVM_y + fluxVP_y) +
                          nzM * (fluxVM_z + fluxVP_z) - λ * (VP - VM)) / 2
                fluxWS = (nxM * (fluxWM_x + fluxWP_x) + nyM * (fluxWM_y + fluxWP_y) +
                          nzM * (fluxWM_z + fluxWP_z) - λ * (WP - WM)) / 2
                fluxES = (nxM * (fluxEM_x + fluxEP_x) + nyM * (fluxEM_y + fluxEP_y) +
                          nzM * (fluxEM_z + fluxEP_z) - λ * (EP - EM)) / 2


                #Update RHS
                rhs[vidM, _ρ, eM] -= sMJ * fluxρS
                rhs[vidM, _U, eM] -= sMJ * fluxUS
                rhs[vidM, _V, eM] -= sMJ * fluxVS
                rhs[vidM, _W, eM] -= sMJ * fluxWS
                rhs[vidM, _E, eM] -= sMJ * fluxES
            end
        end
    end
end
# }}}

# {{{ Volume grad(Q)
function volume_grad!(::Val{dim}, ::Val{N}, rhs::Array, Q, vgeo, D, elems) where {dim, N}
    DFloat = eltype(Q)
    Nq = N + 1
    nelem = size(Q)[end]

    Q = reshape(Q, Nq, Nq, Nq, _nstate, nelem)
    rhs = reshape(rhs, Nq, Nq, Nq, _nstate, dim, nelem)
    vgeo = reshape(vgeo, Nq, Nq, Nq, _nvgeo, nelem)

    #Initialize RHS vector
    fill!( rhs, zero(rhs[1]))

    #Allocate Arrays
    s_F = Array{DFloat}(undef, Nq, Nq, Nq, _nstate, dim)
    s_G = Array{DFloat}(undef, Nq, Nq, Nq, _nstate, dim)
    s_H = Array{DFloat}(undef, Nq, Nq, Nq, _nstate, dim)

    @inbounds for e in elems
        for k = 1:Nq, j = 1:Nq, i = 1:Nq
            MJ = vgeo[i, j, k, _MJ, e]
            ξx, ξy, ξz = vgeo[i,j,k,_ξx,e], vgeo[i,j,k,_ξy,e], vgeo[i,j,k,_ξz,e]
            ηx, ηy, ηz = vgeo[i,j,k,_ηx,e], vgeo[i,j,k,_ηy,e], vgeo[i,j,k,_ηz,e]
            ζx, ζy, ζz = vgeo[i,j,k,_ζx,e], vgeo[i,j,k,_ζy,e], vgeo[i,j,k,_ζz,e]
            z = vgeo[i,j,k,_z,e]

            U, V, W = Q[i, j, k, _U, e], Q[i, j, k, _V, e], Q[i, j, k, _W, e]
            ρ, E = Q[i, j, k, _ρ, e], Q[i, j, k, _E, e]

            #Compute fluxes
            fluxρ = ρ
            fluxU = U
            fluxV = V
            fluxW = W
            fluxE = E

            s_F[i, j, k, _ρ, 1], s_F[i, j, k, _ρ, 2], s_F[i, j, k, _ρ, 3] = MJ * (ξx * fluxρ), MJ * (ξy * fluxρ), MJ * (ξz * fluxρ)
            s_F[i, j, k, _U, 1], s_F[i, j, k, _U, 2], s_F[i, j, k, _U, 3] = MJ * (ξx * fluxU), MJ * (ξy * fluxU), MJ * (ξz * fluxU)
            s_F[i, j, k, _V, 1], s_F[i, j, k, _V, 2], s_F[i, j, k, _V, 3] = MJ * (ξx * fluxV), MJ * (ξy * fluxV), MJ * (ξz * fluxV)
            s_F[i, j, k, _W, 1], s_F[i, j, k, _W, 2], s_F[i, j, k, _W, 3] = MJ * (ξx * fluxW), MJ * (ξy * fluxW), MJ * (ξz * fluxW)
            s_F[i, j, k, _E, 1], s_F[i, j, k, _E, 2], s_F[i, j, k, _E, 3] = MJ * (ξx * fluxE), MJ * (ξy * fluxE), MJ * (ξz * fluxE)

            s_G[i, j, k, _ρ, 1], s_G[i, j, k, _ρ, 2], s_G[i, j, k, _ρ, 3] = MJ * (ηx * fluxρ), MJ * (ηy * fluxρ), MJ * (ηz * fluxρ)
            s_G[i, j, k, _U, 1], s_G[i, j, k, _U, 2], s_G[i, j, k, _U, 3] = MJ * (ηx * fluxU), MJ * (ηy * fluxU), MJ * (ηz * fluxU)
            s_G[i, j, k, _V, 1], s_G[i, j, k, _V, 2], s_G[i, j, k, _V, 3] = MJ * (ηx * fluxV), MJ * (ηy * fluxV), MJ * (ηz * fluxV)
            s_G[i, j, k, _W, 1], s_G[i, j, k, _W, 2], s_G[i, j, k, _W, 3] = MJ * (ηx * fluxW), MJ * (ηy * fluxW), MJ * (ηz * fluxW)
            s_G[i, j, k, _E, 1], s_G[i, j, k, _E, 2], s_G[i, j, k, _E, 3] = MJ * (ηx * fluxE), MJ * (ηy * fluxE), MJ * (ηz * fluxE)

            s_H[i, j, k, _ρ, 1], s_H[i, j, k, _ρ, 2], s_H[i, j, k, _ρ, 3] = MJ * (ζx * fluxρ), MJ * (ζy * fluxρ), MJ * (ζz * fluxρ)
            s_H[i, j, k, _U, 1], s_H[i, j, k, _U, 2], s_H[i, j, k, _U, 3] = MJ * (ζx * fluxU), MJ * (ζy * fluxU), MJ * (ζz * fluxU)
            s_H[i, j, k, _V, 1], s_H[i, j, k, _V, 2], s_H[i, j, k, _V, 3] = MJ * (ζx * fluxV), MJ * (ζy * fluxV), MJ * (ζz * fluxV)
            s_H[i, j, k, _W, 1], s_H[i, j, k, _W, 2], s_H[i, j, k, _W, 3] = MJ * (ζx * fluxW), MJ * (ζy * fluxW), MJ * (ζz * fluxW)
            s_H[i, j, k, _E, 1], s_H[i, j, k, _E, 2], s_H[i, j, k, _E, 3] = MJ * (ζx * fluxE), MJ * (ζy * fluxE), MJ * (ζz * fluxE)
        end

        # loop of ξ-grid lines
        for s = 1:_nstate, k = 1:Nq, j = 1:Nq, i = 1:Nq, n = 1:Nq
            rhs[i, j, k, s, 1, e] -= D[n, i] * s_F[n, j, k, s, 1]
            rhs[i, j, k, s, 2, e] -= D[n, i] * s_F[n, j, k, s, 2]
            rhs[i, j, k, s, 3, e] -= D[n, i] * s_F[n, j, k, s, 3]
        end
        # loop of η-grid lines
        for s = 1:_nstate, k = 1:Nq, j = 1:Nq, i = 1:Nq, n = 1:Nq
            rhs[i, j, k, s, 1, e] -= D[n, j] * s_G[i, n, k, s, 1]
            rhs[i, j, k, s, 2, e] -= D[n, j] * s_G[i, n, k, s, 2]
            rhs[i, j, k, s, 3, e] -= D[n, j] * s_G[i, n, k, s, 3]
        end
        # loop of ζ-grid lines
        for s = 1:_nstate, k = 1:Nq, j = 1:Nq, i = 1:Nq, n = 1:Nq
            rhs[i, j, k, s, 1, e] -= D[n, k] * s_H[i, j, n, s, 1]
            rhs[i, j, k, s, 2, e] -= D[n, k] * s_H[i, j, n, s, 2]
            rhs[i, j, k, s, 3, e] -= D[n, k] * s_H[i, j, n, s, 3]
        end
    end
end
# }}}

# Flux grad(Q)
function flux_grad!(::Val{dim}, ::Val{N}, rhs::Array,  Q, sgeo, elems, vmapM, vmapP, elemtobndy) where {dim, N}
    Np = (N+1)^dim
    Nfp = (N+1)^(dim-1)
    nface = 2*dim

    @inbounds for e in elems
        for f = 1:nface
            for n = 1:Nfp
                (nxM, nyM, nzM, sMJ, ~) = sgeo[:, n, f, e]
                idM, idP = vmapM[n, f, e], vmapP[n, f, e]

                eM, eP = e, ((idP - 1) ÷ Np) + 1
                vidM, vidP = ((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1

                ρM = Q[vidM, _ρ, eM]
                UM = Q[vidM, _U, eM]
                VM = Q[vidM, _V, eM]
                WM = Q[vidM, _W, eM]
                EM = Q[vidM, _E, eM]

                bc = elemtobndy[f, e]
                ρP = UP = VP = WP = EP = zero(eltype(Q))
                if bc == 0
                    ρP = Q[vidP, _ρ, eP]
                    UP = Q[vidP, _U, eP]
                    VP = Q[vidP, _V, eP]
                    WP = Q[vidP, _W, eP]
                    EP = Q[vidP, _E, eP]
                elseif bc == 1
                    UnM = nxM * UM + nyM * VM + nzM * WM
                    UP = UM - 2 * UnM * nxM
                    VP = VM - 2 * UnM * nyM
                    WP = WM - 2 * UnM * nzM
                    ρP = ρM
                    EP = EM
                else
                    error("Invalid boundary conditions $bc on face $f of element $e")
                end

                #Left Fluxes
                fluxρM = ρM
                fluxUM = UM
                fluxVM = VM
                fluxWM = WM
                fluxEM = EM

                #Right Fluxes
                fluxρP = ρP
                fluxUP = UP
                fluxVP = VP
                fluxWP = WP
                fluxEP = EP

                #Compute Numerical/Rusanov Flux
                fluxρS = 0.5*(fluxρM + fluxρP)
                fluxUS = 0.5*(fluxUM + fluxUP)
                fluxVS = 0.5*(fluxVM + fluxVP)
                fluxWS = 0.5*(fluxWM + fluxWP)
                fluxES = 0.5*(fluxEM + fluxEP)

                #Update RHS
                rhs[vidM, _ρ, 1, eM] += sMJ * nxM*fluxρS
                rhs[vidM, _ρ, 2, eM] += sMJ * nyM*fluxρS
                rhs[vidM, _ρ, 3, eM] += sMJ * nzM*fluxρS
                rhs[vidM, _U, 1, eM] += sMJ * nxM*fluxUS
                rhs[vidM, _U, 2, eM] += sMJ * nyM*fluxUS
                rhs[vidM, _U, 3, eM] += sMJ * nzM*fluxUS
                rhs[vidM, _V, 1, eM] += sMJ * nxM*fluxVS
                rhs[vidM, _V, 2, eM] += sMJ * nyM*fluxVS
                rhs[vidM, _V, 3, eM] += sMJ * nzM*fluxVS
                rhs[vidM, _W, 1, eM] += sMJ * nxM*fluxWS
                rhs[vidM, _W, 2, eM] += sMJ * nyM*fluxWS
                rhs[vidM, _W, 3, eM] += sMJ * nzM*fluxWS
                rhs[vidM, _E, 1, eM] += sMJ * nxM*fluxES
                rhs[vidM, _E, 2, eM] += sMJ * nyM*fluxES
                rhs[vidM, _E, 3, eM] += sMJ * nzM*fluxES
            end
        end
    end
end
# }}}

# {{{ Volume div(grad(Q))
function volume_div!(::Val{dim}, ::Val{N}, rhs::Array, Q, vgeo, D, elems) where {dim, N}
    DFloat = eltype(Q)
    Nq = N + 1
    nelem = size(Q)[end]

    Q = reshape(Q, Nq, Nq, Nq, _nstate, dim, nelem)
    rhs = reshape(rhs, Nq, Nq, Nq, _nstate, dim, nelem)
    vgeo = reshape(vgeo, Nq, Nq, Nq, _nvgeo, nelem)

    #Initialize RHS vector
    fill!( rhs, zero(rhs[1]))

    #Allocate Arrays
    s_F = Array{DFloat}(undef, Nq, Nq, Nq, _nstate)
    s_G = Array{DFloat}(undef, Nq, Nq, Nq, _nstate)
    s_H = Array{DFloat}(undef, Nq, Nq, Nq, _nstate)

    @inbounds for e in elems
        for k = 1:Nq, j = 1:Nq, i = 1:Nq
            MJ = vgeo[i, j, k, _MJ, e]
            ξx, ξy, ξz = vgeo[i,j,k,_ξx,e], vgeo[i,j,k,_ξy,e], vgeo[i,j,k,_ξz,e]
            ηx, ηy, ηz = vgeo[i,j,k,_ηx,e], vgeo[i,j,k,_ηy,e], vgeo[i,j,k,_ηz,e]
            ζx, ζy, ζz = vgeo[i,j,k,_ζx,e], vgeo[i,j,k,_ζy,e], vgeo[i,j,k,_ζz,e]

            ρx, ρy, ρz = Q[i,j,k,_ρ,1,e], Q[i,j,k,_ρ,2,e], Q[i,j,k,_ρ,3,e]
            Ux, Uy, Uz = Q[i,j,k,_U,1,e], Q[i,j,k,_U,2,e], Q[i,j,k,_U,3,e]
            Vx, Vy, Vz = Q[i,j,k,_V,1,e], Q[i,j,k,_V,2,e], Q[i,j,k,_V,3,e]
            Wx, Wy, Wz = Q[i,j,k,_W,1,e], Q[i,j,k,_W,2,e], Q[i,j,k,_W,3,e]
            Ex, Ey, Ez = Q[i,j,k,_E,1,e], Q[i,j,k,_E,2,e], Q[i,j,k,_E,3,e]

            #Compute fluxes
            fluxρ_x = ρx
            fluxρ_y = ρy
            fluxρ_z = ρz
            fluxU_x = Ux
            fluxU_y = Uy
            fluxU_z = Uz
            fluxV_x = Vx
            fluxV_y = Vy
            fluxV_z = Vz
            fluxW_x = Wx
            fluxW_y = Wy
            fluxW_z = Wz
            fluxE_x = Ex
            fluxE_y = Ey
            fluxE_z = Ez

            s_F[i, j, k, _ρ] = MJ * (ξx * fluxρ_x + ξy * fluxρ_y + ξz * fluxρ_z)
            s_F[i, j, k, _U] = MJ * (ξx * fluxU_x + ξy * fluxU_y + ξz * fluxU_z)
            s_F[i, j, k, _V] = MJ * (ξx * fluxV_x + ξy * fluxV_y + ξz * fluxV_z)
            s_F[i, j, k, _W] = MJ * (ξx * fluxW_x + ξy * fluxW_y + ξz * fluxW_z)
            s_F[i, j, k, _E] = MJ * (ξx * fluxE_x + ξy * fluxE_y + ξz * fluxE_z)

            s_G[i, j, k, _ρ] = MJ * (ηx * fluxρ_x + ηy * fluxρ_y + ηz * fluxρ_z)
            s_G[i, j, k, _U] = MJ * (ηx * fluxU_x + ηy * fluxU_y + ηz * fluxU_z)
            s_G[i, j, k, _V] = MJ * (ηx * fluxV_x + ηy * fluxV_y + ηz * fluxV_z)
            s_G[i, j, k, _W] = MJ * (ηx * fluxW_x + ηy * fluxW_y + ηz * fluxW_z)
            s_G[i, j, k, _E] = MJ * (ηx * fluxE_x + ηy * fluxE_y + ηz * fluxE_z)

            s_H[i, j, k, _ρ] = MJ * (ζx * fluxρ_x + ζy * fluxρ_y + ζz * fluxρ_z)
            s_H[i, j, k, _U] = MJ * (ζx * fluxU_x + ζy * fluxU_y + ζz * fluxU_z)
            s_H[i, j, k, _V] = MJ * (ζx * fluxV_x + ζy * fluxV_y + ζz * fluxV_z)
            s_H[i, j, k, _W] = MJ * (ζx * fluxW_x + ζy * fluxW_y + ζz * fluxW_z)
            s_H[i, j, k, _E] = MJ * (ζx * fluxE_x + ζy * fluxE_y + ζz * fluxE_z)
        end

        # loop of ξ-grid lines
        for s = 1:_nstate, k = 1:Nq, j = 1:Nq, i = 1:Nq, n = 1:Nq
            rhs[i, j, k, s, 1, e] -= D[n, i] * s_F[n, j, k, s]
        end
        # loop of η-grid lines
        for s = 1:_nstate, k = 1:Nq, j = 1:Nq, i = 1:Nq, n = 1:Nq
            rhs[i, j, k, s, 1, e] -= D[n, j] * s_G[i, n, k, s]
        end
        # loop of ζ-grid lines
        for s = 1:_nstate, k = 1:Nq, j = 1:Nq, i = 1:Nq, n = 1:Nq
            rhs[i, j, k, s, 1, e] -= D[n, k] * s_H[i, j, n, s]
        end
    end
end
# }}}

# Flux div(grad(Q))
function flux_div!(::Val{dim}, ::Val{N}, rhs::Array,  Q, sgeo, elems, vmapM, vmapP, elemtobndy) where {dim, N}
    Np = (N+1)^dim
    Nfp = (N+1)^(dim-1)
    nface = 2*dim

    @inbounds for e in elems
        for f = 1:nface
            for n = 1:Nfp
                (nxM, nyM, nzM, sMJ, ~) = sgeo[:, n, f, e]
                idM, idP = vmapM[n, f, e], vmapP[n, f, e]

                eM, eP = e, ((idP - 1) ÷ Np) + 1
                vidM, vidP = ((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1

                ρxM = Q[vidM, _ρ, 1, eM]
                ρyM = Q[vidM, _ρ, 2, eM]
                ρzM = Q[vidM, _ρ, 3, eM]
                UxM = Q[vidM, _U, 1, eM]
                UyM = Q[vidM, _U, 2, eM]
                UzM = Q[vidM, _U, 3, eM]
                VxM = Q[vidM, _V, 1, eM]
                VyM = Q[vidM, _V, 2, eM]
                VzM = Q[vidM, _V, 3, eM]
                WxM = Q[vidM, _W, 1, eM]
                WyM = Q[vidM, _W, 2, eM]
                WzM = Q[vidM, _W, 3, eM]
                ExM = Q[vidM, _E, 1, eM]
                EyM = Q[vidM, _E, 2, eM]
                EzM = Q[vidM, _E, 3, eM]

                bc = elemtobndy[f, e]
                ρxP = ρyP = ρzP = zero(eltype(Q))
                UxP = UyP = UzP = zero(eltype(Q))
                VxP = VyP = VzP = zero(eltype(Q))
                WxP = WyP = WzP = zero(eltype(Q))
                ExP = EyP = EzP = zero(eltype(Q))
                if bc == 0
                    ρxP = Q[vidP, _ρ, 1, eP]
                    ρyP = Q[vidP, _ρ, 2, eP]
                    ρzP = Q[vidP, _ρ, 3, eP]
                    UxP = Q[vidP, _U, 1, eP]
                    UyP = Q[vidP, _U, 2, eP]
                    UzP = Q[vidP, _U, 3, eP]
                    VxP = Q[vidP, _V, 1, eP]
                    VyP = Q[vidP, _V, 2, eP]
                    VzP = Q[vidP, _V, 3, eP]
                    WxP = Q[vidP, _W, 1, eP]
                    WyP = Q[vidP, _W, 2, eP]
                    WzP = Q[vidP, _W, 3, eP]
                    ExP = Q[vidP, _E, 1, eP]
                    EyP = Q[vidP, _E, 2, eP]
                    EzP = Q[vidP, _E, 3, eP]
                elseif bc == 1
                    ρnM = nxM * ρxM + nyM * ρyM + nzM * ρzM
                    ρxP = ρxM - 2 * ρnM * nxM
                    ρyP = ρyM - 2 * ρnM * nyM
                    ρzP = ρzM - 2 * ρnM * nzM
                    UnM = nxM * UxM + nyM * UyM + nzM * UzM
                    UxP = UxM - 2 * UnM * nxM
                    UyP = UyM - 2 * UnM * nyM
                    UzP = UzM - 2 * UnM * nzM
                    VnM = nxM * VxM + nyM * VyM + nzM * VzM
                    VxP = VxM - 2 * VnM * nxM
                    VyP = VyM - 2 * VnM * nyM
                    VzP = VzM - 2 * VnM * nzM
                    WnM = nxM * WxM + nyM * WyM + nzM * WzM
                    WxP = WxM - 2 * WnM * nxM
                    WyP = WyM - 2 * WnM * nyM
                    WzP = WzM - 2 * WnM * nzM
                    EnM = nxM * ExM + nyM * EyM + nzM * EzM
                    ExP = ExM - 2 * EnM * nxM
                    EyP = EyM - 2 * EnM * nyM
                    EzP = EzM - 2 * EnM * nzM
                else
                    error("Invalid boundary conditions $bc on face $f of element $e")
                end

                #Left Fluxes
                fluxρM_x = ρxM
                fluxρM_y = ρyM
                fluxρM_z = ρzM
                fluxUM_x = UxM
                fluxUM_y = UyM
                fluxUM_z = UzM
                fluxVM_x = VxM
                fluxVM_y = VyM
                fluxVM_z = VzM
                fluxWM_x = WxM
                fluxWM_y = WyM
                fluxWM_z = WzM
                fluxEM_x = ExM
                fluxEM_y = EyM
                fluxEM_z = EzM

                #Right Fluxes
                fluxρP_x = ρxP
                fluxρP_y = ρyP
                fluxρP_z = ρzP
                fluxUP_x = UxP
                fluxUP_y = UyP
                fluxUP_z = UzP
                fluxVP_x = VxP
                fluxVP_y = VyP
                fluxVP_z = VzP
                fluxWP_x = WxP
                fluxWP_y = WyP
                fluxWP_z = WzP
                fluxEP_x = ExP
                fluxEP_y = EyP
                fluxEP_z = EzP

                #Compute Numerical Flux
                fluxρS = 0.5*(nxM * (fluxρM_x + fluxρP_x) + nyM * (fluxρM_y + fluxρP_y) + nzM * (fluxρM_z + fluxρP_z))
                fluxUS = 0.5*(nxM * (fluxUM_x + fluxUP_x) + nyM * (fluxUM_y + fluxUP_y) + nzM * (fluxUM_z + fluxUP_z))
                fluxVS = 0.5*(nxM * (fluxVM_x + fluxVP_x) + nyM * (fluxVM_y + fluxVP_y) + nzM * (fluxVM_z + fluxVP_z))
                fluxWS = 0.5*(nxM * (fluxWM_x + fluxWP_x) + nyM * (fluxWM_y + fluxWP_y) + nzM * (fluxWM_z + fluxWP_z))
                fluxES = 0.5*(nxM * (fluxEM_x + fluxEP_x) + nyM * (fluxEM_y + fluxEP_y) + nzM * (fluxEM_z + fluxEP_z))

                #Update RHS
                rhs[vidM, _ρ, 1, eM] += sMJ * fluxρS
                rhs[vidM, _U, 1, eM] += sMJ * fluxUS
                rhs[vidM, _V, 1, eM] += sMJ * fluxVS
                rhs[vidM, _W, 1, eM] += sMJ * fluxWS
                rhs[vidM, _E, 1, eM] += sMJ * fluxES
            end
        end
    end
end
# }}}

# {{{ Update grad Q solution
function update_gradQ!(::Val{dim}, ::Val{N}, Q, rhs, vgeo, elems) where {dim, N}

    Nq=(N+1)^dim

    @inbounds for e = elems, s = 1:_nstate, i = 1:Nq
        Q[i, s, 1, e] = rhs[i, s, 1, e] * vgeo[i, _MJI, e]
        Q[i, s, 2, e] = rhs[i, s, 2, e] * vgeo[i, _MJI, e]
        Q[i, s, 3, e] = rhs[i, s, 3, e] * vgeo[i, _MJI, e]
    end

end
# }}}

# {{{ Update grad Q solution
function update_divgradQ!(::Val{dim}, ::Val{N}, Q, rhs, vgeo, elems) where {dim, N}

    Nq=(N+1)^dim

    @inbounds for e = elems, s = 1:_nstate, i = 1:Nq
        Q[i, s, e] = rhs[i, s, 1, e] * vgeo[i, _MJI, e]
    end

end
# }}}

# {{{ Update solution (for all dimensions)
function updatesolution!(::Val{dim}, ::Val{N}, rhs::Array, rhs_gradQ, Q, vgeo, elems, rka,
                         rkb, dt, visc) where {dim, N}

    Nq=(N+1)^dim

    @inbounds for e = elems, s = 1:_nstate, i = 1:Nq
        rhs[i, s, e] += visc*rhs_gradQ[i,s,1,e]
        Q[i, s, e] += rkb * dt * rhs[i, s, e] * vgeo[i, _MJI, e]
        rhs[i, s, e] *= rka
    end
end

# }}}
# }}}

# {{{ improved GPU kernles

# {{{ Volume RHS for 3D
@hascuda function knl_volumerhs!(::Val{3}, ::Val{N}, rhs, Q, vgeo, D, nelem) where N
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
    s_F = @cuStaticSharedMem(eltype(Q), (Nq, Nq, Nq, _nstate))
    s_G = @cuStaticSharedMem(eltype(Q), (Nq, Nq, Nq, _nstate))
    s_H = @cuStaticSharedMem(eltype(Q), (Nq, Nq, Nq, _nstate))

    rhsU = rhsV = rhsW = rhsρ = rhsE = zero(eltype(rhs))
    @inbounds if i <= Nq && j <= Nq && k <= Nq && e <= nelem
        # Load derivative into shared memory
        if k == 1
            s_D[i, j] = D[i, j]
        end

        # Load values will need into registers
        MJ = vgeo[i, j, k, _MJ, e]
        ξx, ξy, ξz = vgeo[i,j,k,_ξx,e], vgeo[i,j,k,_ξy,e], vgeo[i,j,k,_ξz,e]
        ηx, ηy, ηz = vgeo[i,j,k,_ηx,e], vgeo[i,j,k,_ηy,e], vgeo[i,j,k,_ηz,e]
        ζx, ζy, ζz = vgeo[i,j,k,_ζx,e], vgeo[i,j,k,_ζy,e], vgeo[i,j,k,_ζz,e]

        U, V, W = Q[i, j, k, _U, e], Q[i, j, k, _V, e], Q[i, j, k, _W, e]
        ρ, E = Q[i, j, k, _ρ, e], Q[i, j, k, _E, e]

        P = p0 * CUDAnative.pow(R_gas * E / p0, c_p / c_v)

        ρinv = 1 / ρ
        fluxρ_x = U
        fluxU_x = ρinv * U * U + P
        fluxV_x = ρinv * V * U
        fluxW_x = ρinv * W * U
        fluxE_x = E * ρinv * U

        fluxρ_y = V
        fluxU_y = ρinv * U * V
        fluxV_y = ρinv * V * V + P
        fluxW_y = ρinv * W * V
        fluxE_y = E * ρinv * V

        fluxρ_z = W
        fluxU_z = ρinv * U * W
        fluxV_z = ρinv * V * W
        fluxW_z = ρinv * W * W + P
        fluxE_z = E * ρinv * W

        s_F[i, j, k, _ρ] = MJ * (ξx * fluxρ_x + ξy * fluxρ_y + ξz * fluxρ_z)
        s_F[i, j, k, _U] = MJ * (ξx * fluxU_x + ξy * fluxU_y + ξz * fluxU_z)
        s_F[i, j, k, _V] = MJ * (ξx * fluxV_x + ξy * fluxV_y + ξz * fluxV_z)
        s_F[i, j, k, _W] = MJ * (ξx * fluxW_x + ξy * fluxW_y + ξz * fluxW_z)
        s_F[i, j, k, _E] = MJ * (ξx * fluxE_x + ξy * fluxE_y + ξz * fluxE_z)

        s_G[i, j, k, _ρ] = MJ * (ηx * fluxρ_x + ηy * fluxρ_y + ηz * fluxρ_z)
        s_G[i, j, k, _U] = MJ * (ηx * fluxU_x + ηy * fluxU_y + ηz * fluxU_z)
        s_G[i, j, k, _V] = MJ * (ηx * fluxV_x + ηy * fluxV_y + ηz * fluxV_z)
        s_G[i, j, k, _W] = MJ * (ηx * fluxW_x + ηy * fluxW_y + ηz * fluxW_z)
        s_G[i, j, k, _E] = MJ * (ηx * fluxE_x + ηy * fluxE_y + ηz * fluxE_z)

        s_H[i, j, k, _ρ] = MJ * (ζx * fluxρ_x + ζy * fluxρ_y + ζz * fluxρ_z)
        s_H[i, j, k, _U] = MJ * (ζx * fluxU_x + ζy * fluxU_y + ζz * fluxU_z)
        s_H[i, j, k, _V] = MJ * (ζx * fluxV_x + ζy * fluxV_y + ζz * fluxV_z)
        s_H[i, j, k, _W] = MJ * (ζx * fluxW_x + ζy * fluxW_y + ζz * fluxW_z)
        s_H[i, j, k, _E] = MJ * (ζx * fluxE_x + ζy * fluxE_y + ζz * fluxE_z)

        rhsU, rhsV, rhsW = (rhs[i, j, k, _U, e],
                            rhs[i, j, k, _V, e],
                            rhs[i, j, k, _W, e])
        rhsρ, rhsE = rhs[i, j, k, _ρ, e], rhs[i, j, k, _E, e]

        # buoyancy term
        rhsW -= MJ * ρ * gravity
    end

    sync_threads()

    @inbounds if i <= Nq && j <= Nq && k <= Nq && e <= nelem
        # loop of ξ-grid lines
        for n = 1:Nq
            Dni = s_D[n, i]
            Dnj = s_D[n, j]
            Dnk = s_D[n, k]

            rhsρ += Dni * s_F[n, j, k, _ρ]
            rhsρ += Dnj * s_G[i, n, k, _ρ]
            rhsρ += Dnk * s_H[i, j, n, _ρ]

            rhsU += Dni * s_F[n, j, k, _U]
            rhsU += Dnj * s_G[i, n, k, _U]
            rhsU += Dnk * s_H[i, j, n, _U]

            rhsV += Dni * s_F[n, j, k, _V]
            rhsV += Dnj * s_G[i, n, k, _V]
            rhsV += Dnk * s_H[i, j, n, _V]

            rhsW += Dni * s_F[n, j, k, _W]
            rhsW += Dnj * s_G[i, n, k, _W]
            rhsW += Dnk * s_H[i, j, n, _W]

            rhsE += Dni * s_F[n, j, k, _E]
            rhsE += Dnj * s_G[i, n, k, _E]
            rhsE += Dnk * s_H[i, j, n, _E]
        end

        rhs[i, j, k, _U, e] = rhsU
        rhs[i, j, k, _V, e] = rhsV
        rhs[i, j, k, _W, e] = rhsW
        rhs[i, j, k, _ρ, e] = rhsρ
        rhs[i, j, k, _E, e] = rhsE
    end
    nothing
end
# }}}

# {{{ Face RHS (all dimensions)
@hascuda function knl_fluxrhs!(::Val{dim}, ::Val{N}, rhs, Q, sgeo, vgeo, nelem, vmapM,
                               vmapP, elemtobndy) where {dim, N}
    DFloat = eltype(Q)
    γ::DFloat       = _γ
    p0::DFloat      = _p0
    R_gas::DFloat   = _R_gas
    c_p::DFloat     = _c_p
    c_v::DFloat     = _c_v
    gravity::DFloat = _gravity

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
                UM = Q[vidM, _U, eM]
                VM = Q[vidM, _V, eM]
                WM = Q[vidM, _W, eM]
                EM = Q[vidM, _E, eM]

                bc = elemtobndy[f, e]
                PM = p0 * CUDAnative.pow(R_gas * EM / p0, c_p / c_v)
                ρP = UP = VP = WP = EP = PP = zero(eltype(Q))
                if bc == 0
                    ρP = Q[vidP, _ρ, eP]
                    UP = Q[vidP, _U, eP]
                    VP = Q[vidP, _V, eP]
                    WP = Q[vidP, _W, eP]
                    EP = Q[vidP, _E, eP]
                    PP = p0 * CUDAnative.pow(R_gas * EP / p0, c_p / c_v)
                elseif bc == 1
                    UnM = nxM * UM + nyM * VM + nzM * WM
                    UP = UM - 2 * UnM * nxM
                    VP = VM - 2 * UnM * nyM
                    WP = WM - 2 * UnM * nzM
                    ρP = ρM
                    EP = EM
                    PP = PM
                end

                ρMinv = 1 / ρM
                fluxρM_x = UM
                fluxUM_x = ρMinv * UM * UM + PM
                fluxVM_x = ρMinv * VM * UM
                fluxWM_x = ρMinv * WM * UM
                fluxEM_x = ρMinv * UM * EM

                fluxρM_y = VM
                fluxUM_y = ρMinv * UM * VM
                fluxVM_y = ρMinv * VM * VM + PM
                fluxWM_y = ρMinv * WM * VM
                fluxEM_y = ρMinv * VM * EM

                fluxρM_z = WM
                fluxUM_z = ρMinv * UM * WM
                fluxVM_z = ρMinv * VM * WM
                fluxWM_z = ρMinv * WM * WM + PM
                fluxEM_z = ρMinv * WM * EM

                ρPinv = 1 / ρP
                fluxρP_x = UP
                fluxUP_x = ρPinv * UP * UP + PP
                fluxVP_x = ρPinv * VP * UP
                fluxWP_x = ρPinv * WP * UP
                fluxEP_x = ρPinv * UP * EP

                fluxρP_y = VP
                fluxUP_y = ρPinv * UP * VP
                fluxVP_y = ρPinv * VP * VP + PP
                fluxWP_y = ρPinv * WP * VP
                fluxEP_y = ρPinv * VP * EP

                fluxρP_z = WP
                fluxUP_z = ρPinv * UP * WP
                fluxVP_z = ρPinv * VP * WP
                fluxWP_z = ρPinv * WP * WP + PP
                fluxEP_z = ρPinv * WP * EP

                λM = ρMinv * abs(nxM * UM + nyM * VM + nzM * WM) + CUDAnative.sqrt(ρMinv * γ * PM)
                λP = ρPinv * abs(nxM * UP + nyM * VP + nzM * WP) + CUDAnative.sqrt(ρPinv * γ * PP)
                λ  =  max(λM, λP)

                #Compute Numerical Flux and Update
                fluxρS = (nxM * (fluxρM_x + fluxρP_x) + nyM * (fluxρM_y + fluxρP_y) +
                          nzM * (fluxρM_z + fluxρP_z) - λ * (ρP - ρM)) / 2
                fluxUS = (nxM * (fluxUM_x + fluxUP_x) + nyM * (fluxUM_y + fluxUP_y) +
                          nzM * (fluxUM_z + fluxUP_z) - λ * (UP - UM)) / 2
                fluxVS = (nxM * (fluxVM_x + fluxVP_x) + nyM * (fluxVM_y + fluxVP_y) +
                          nzM * (fluxVM_z + fluxVP_z) - λ * (VP - VM)) / 2
                fluxWS = (nxM * (fluxWM_x + fluxWP_x) + nyM * (fluxWM_y + fluxWP_y) +
                          nzM * (fluxWM_z + fluxWP_z) - λ * (WP - WM)) / 2
                fluxES = (nxM * (fluxEM_x + fluxEP_x) + nyM * (fluxEM_y + fluxEP_y) +
                          nzM * (fluxEM_z + fluxEP_z) - λ * (EP - EM)) / 2

                #Update RHS
                rhs[vidM, _ρ, eM] -= sMJ * fluxρS
                rhs[vidM, _U, eM] -= sMJ * fluxUS
                rhs[vidM, _V, eM] -= sMJ * fluxVS
                rhs[vidM, _W, eM] -= sMJ * fluxWS
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

@hascuda function fluxrhs!(::Val{dim}, ::Val{N}, d_rhsL::CuArray, d_QL, d_sgeo,
                           d_vgeoL, elems, d_vmapM, d_vmapP, d_elemtobndy) where {dim, N}
    nelem = length(elems)
    @cuda(threads=(ntuple(j->N+1, dim-1)..., 1), blocks=nelem,
          knl_fluxrhs!(Val(dim), Val(N), d_rhsL, d_QL, d_sgeo, d_vgeoL, nelem, d_vmapM,
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

# {{{ Send Data Q
function senddata_Q(::Val{dim}, ::Val{N}, mesh, sendreq, recvreq, sendQ,
                  recvQ, d_sendelems, d_sendQ, d_recvQ, d_QL, mpicomm;
                  ArrType=ArrType) where {dim, N}
    mpirank = MPI.Comm_rank(mpicomm)

    # Create send and recv request array
    nnabr = length(mesh.nabrtorank)
    d_sendelems = ArrType(mesh.sendelems)
    nrealelem = length(mesh.realelems)

    # post MPI receives
    for n = 1:nnabr
        recvreq[n] = MPI.Irecv!((@view recvQ[:, :, mesh.nabrtorecv[n]]),
                                mesh.nabrtorank[n], 777, mpicomm)
    end

    # wait on (prior) MPI sends
    MPI.Waitall!(sendreq)

    # pack data from d_QL into send buffer
#    fillsendQ!(Val(dim), Val(N), sendQ, d_sendQ, d_QL, d_sendelems)
    sendQ[:, :, :] .= d_QL[:, :, d_sendelems]

    # post MPI sends
    for n = 1:nnabr
        sendreq[n] = MPI.Isend((@view sendQ[:, :, mesh.nabrtosend[n]]),
                               mesh.nabrtorank[n], 777, mpicomm)
    end
end
# }}}

# {{{ Send Data Grad(Q)
function senddata_gradQ(::Val{dim}, ::Val{N}, mesh, sendreq, recvreq, sendQ,
                  recvQ, d_sendelems, d_sendQ, d_recvQ, d_QL, mpicomm;
                  ArrType=ArrType) where {dim, N}
    mpirank = MPI.Comm_rank(mpicomm)

    # Create send and recv request array
    nnabr = length(mesh.nabrtorank)
    d_sendelems = ArrType(mesh.sendelems)
    nrealelem = length(mesh.realelems)

    # post MPI receives
    for n = 1:nnabr
        recvreq[n] = MPI.Irecv!((@view recvQ[:, :, :, mesh.nabrtorecv[n]]),
                                mesh.nabrtorank[n], 777, mpicomm)
    end

    # wait on (prior) MPI sends
    MPI.Waitall!(sendreq)

    # pack data from d_QL into send buffer
#    fillsendQ!(Val(dim), Val(N), sendQ, d_sendQ, d_QL, d_sendelems)
    sendQ[:, :, :, :] .= d_QL[:, :, :, d_sendelems]

    # post MPI sends
    for n = 1:nnabr
        sendreq[n] = MPI.Isend((@view sendQ[:, :, :, mesh.nabrtosend[n]]),
                               mesh.nabrtorank[n], 777, mpicomm)
    end
end
# }}}

# {{{ Receive Data Q
function receivedata_Q!(::Val{dim}, ::Val{N}, mesh, recvreq, recvQ,
                        d_recvQ, d_QL) where {dim, N}
    nrealelem = length(mesh.realelems)

    # wait on MPI receives
    MPI.Waitall!(recvreq)

    # copy data to state vector d_QL
    #transferrecvQ!(Val(dim), Val(N), d_recvQ, recvQ, d_QL, nrealelem)
    d_QL[:, :, nrealelem+1:end] .= recvQ[:, :, :]

end
# }}}

# {{{ Receive Data Grad(Q)
function receivedata_gradQ!(::Val{dim}, ::Val{N}, mesh, recvreq, recvQ,
                            d_recvQ, d_QL) where {dim, N}
    nrealelem = length(mesh.realelems)

    # wait on MPI receives
    MPI.Waitall!(recvreq)

    # copy data to state vector d_QL
    #transferrecvQ!(Val(dim), Val(N), d_recvQ, recvQ, d_QL, nrealelem)
    d_QL[:, :, :, nrealelem+1:end] .= recvQ[:, :, :, :]
end
# }}}

# {{{ RK loop
function lowstorageRK(::Val{dim}, ::Val{N}, mesh, vgeo, sgeo, Q, rhs, D,
                      dt, nsteps, tout, vmapM, vmapP, mpicomm, iplot, visc;
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
    # Create send and recv LDG buffer
    sendgradQ = zeros(DFloat, (N+1)^dim, size(Q,2), dim, length(mesh.sendelems))
    recvgradQ = zeros(DFloat, (N+1)^dim, size(Q,2), dim, length(mesh.ghostelems))

    # Store Constants
    nrealelem = length(mesh.realelems)
    nsendelem = length(mesh.sendelems)
    nrecvelem = length(mesh.ghostelems)
    nelem = length(mesh.elems)

    #Create Device Arrays
    d_QL, d_rhsL = ArrType(Q), ArrType(rhs)
    d_vgeoL, d_sgeo = ArrType(vgeo), ArrType(sgeo)
    d_vmapM, d_vmapP = ArrType(vmapM), ArrType(vmapP)
    d_sendelems, d_elemtobndy = ArrType(mesh.sendelems), ArrType(mesh.elemtobndy)
    d_sendQ, d_recvQ = ArrType(sendQ), ArrType(recvQ)
    d_D = ArrType(D)
    #Create Device LDG Arrays
    d_gradQL = zeros(DFloat, (N+1)^dim, _nstate, dim, nelem)
    d_rhs_gradQL = zeros(DFloat, (N+1)^dim, _nstate, dim, nelem)
    d_sendgradQ, d_recvgradQ = ArrType(sendgradQ), ArrType(recvgradQ)

    #Template Reshape Arrays
    Qshape    = (fill(N+1, dim)..., size(Q, 2), size(Q, 3))
    vgeoshape = (fill(N+1, dim)..., _nvgeo, size(Q, 3))
    gradQshape = (fill(N+1, dim)..., size(d_gradQL,2), size(d_gradQL,3), size(d_gradQL,4))

    #Reshape Device Arrays
    d_QC = reshape(d_QL, Qshape)
    d_rhsC = reshape(d_rhsL, Qshape...)
    d_vgeoC = reshape(d_vgeoL, vgeoshape)
    #Reshape Device LDG Arrays
    d_gradQC = reshape(d_gradQL, gradQshape)
    d_rhs_gradQC = reshape(d_rhs_gradQL, gradQshape...)

    start_time = t1 = time_ns()
    for step = 1:nsteps
        for s = 1:length(RKA)

            #---------------1st Order Operators--------------------------#
            # Send Data Q
            senddata_Q(Val(dim), Val(N), mesh, sendreq, recvreq, sendQ,
                       recvQ, d_sendelems, d_sendQ, d_recvQ, d_QL, mpicomm;
                       ArrType=ArrType)

            # volume RHS computation
            volumerhs!(Val(dim), Val(N), d_rhsC, d_QC, d_vgeoC, d_D, mesh.realelems)

            # Receive Data Q
            receivedata_Q!(Val(dim), Val(N), mesh, recvreq, recvQ, d_recvQ, d_QL)

            # face RHS computation
            fluxrhs!(Val(dim), Val(N), d_rhsL, d_QL, d_sgeo, d_vgeoL, mesh.realelems, d_vmapM,
                     d_vmapP, d_elemtobndy)

            #---------------2nd Order Operators--------------------------#
            if (visc > 0)
                # volume grad Q computation
                volume_grad!(Val(dim), Val(N), d_rhs_gradQC, d_QC, d_vgeoC, d_D, mesh.realelems)

                # flux grad Q computation
                flux_grad!(Val(dim), Val(N), d_rhs_gradQL, d_QL, d_sgeo, mesh.realelems, d_vmapM, d_vmapP, d_elemtobndy)

                # Construct grad Q
                update_gradQ!(Val(dim), Val(N), d_gradQL, d_rhs_gradQL, d_vgeoL, mesh.realelems)

                # Send Data grad(Q)
                senddata_gradQ(Val(dim), Val(N), mesh, sendreq, recvreq, sendgradQ,
                               recvgradQ, d_sendelems, d_sendgradQ, d_recvgradQ,
                               d_gradQL, mpicomm;ArrType=ArrType)

                # volume div(grad Q) computation
                volume_div!(Val(dim), Val(N), d_rhs_gradQC, d_gradQC, d_vgeoC, d_D, mesh.realelems)

                # Receive Data grad(Q)
                receivedata_gradQ!(Val(dim), Val(N), mesh, recvreq, recvgradQ, d_recvgradQ, d_gradQL)

                # flux div(grad Q) computation
                flux_div!(Val(dim), Val(N), d_rhs_gradQL, d_gradQL, d_sgeo, mesh.realelems, d_vmapM, d_vmapP, d_elemtobndy)
            end

            #---------------Update Solution--------------------------#
            # update solution and scale RHS
            updatesolution!(Val(dim), Val(N), d_rhsL, d_rhs_gradQL, d_QL, d_vgeoL, mesh.realelems,
                            RKA[s%length(RKA)+1], RKB[s], dt, visc)
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

        # Write VTK file
        if mod(step,iplot) == 0
            Q .= d_QL
            convert_set3c_to_set2nc(Val(dim), Val(N), vgeo, Q)
            X = ntuple(j->reshape((@view vgeo[:, _x+j-1, :]), ntuple(j->N+1,dim)..., nelem), dim)
            ρ = reshape((@view Q[:, _ρ, :]), ntuple(j->(N+1),dim)..., nelem)
            U = reshape((@view Q[:, _U, :]), ntuple(j->(N+1),dim)..., nelem)
            V = reshape((@view Q[:, _V, :]), ntuple(j->(N+1),dim)..., nelem)
            W = reshape((@view Q[:, _W, :]), ntuple(j->(N+1),dim)..., nelem)
            E = reshape((@view Q[:, _E, :]), ntuple(j->(N+1),dim)..., nelem)
            E = E .- 300.0
            writemesh(@sprintf("viz/euler%dD_set3c_%s_rank_%04d_step_%05d",dim, ArrType, mpirank, step), X...;
                      fields=(("ρ", ρ), ("U", U), ("V", V), ("W", W), ("E", E)), realelems=mesh.realelems)
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

    println("[CPU] converting variables (CPU)...")
    @inbounds for e = 1:nelem, n = 1:Np
        ρ, u, v, w, E = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e], Q[n, _W, e], Q[n, _E, e]
        Q[n, _U, e] = ρ*u
        Q[n, _V, e] = ρ*v
        Q[n, _W, e] = ρ*w
        Q[n, _E, e] = ρ*E
    end
end
# }}}

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
        ρ, U, V, W, E = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e], Q[n, _W, e], Q[n, _E, e]
        u=U/ρ
        v=V/ρ
        w=W/ρ
        E=E/ρ
        Q[n, _U, e] = u
        Q[n, _V, e] = v
        Q[n, _W, e] = w
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
        ρ, u, v, w, E = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e], Q[n, _W, e], Q[n, _E, e]
        z = vgeo[n, _z, e]
        P = p0 * (ρ * R_gas * E / p0)^(c_p / c_v)
        T = P/(ρ*R_gas)
        E = c_v*T + 0.5*(u^2 + v^2 + w^2) + gravity*z
        Q[n, _U, e] = ρ*u
        Q[n, _V, e] = ρ*v
        Q[n, _W, e] = ρ*w
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
        ρ, U, V, W, E = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e], Q[n, _W, e], Q[n, _E, e]
        z = vgeo[n, _z, e]
        u=U/ρ
        v=V/ρ
        w=W/ρ
        E=E/ρ
        P = (R_gas/c_v)*ρ*(E - 0.5*(u^2 + v^2 + w^2) - gravity*z)
        E=p0/(ρ * R_gas)*( P/p0 )^(c_v/c_p)
        Q[n, _U, e] = u
        Q[n, _V, e] = v
        Q[n, _W, e] = w
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
        ρ, u, v, w, E = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e], Q[n, _W, e], Q[n, _E, e]
        P = p0 * (ρ * R_gas * E / p0)^(c_p / c_v)
        T = P/(ρ*R_gas)
        E = c_v*T
        Q[n, _U, e] = ρ*u
        Q[n, _V, e] = ρ*v
        Q[n, _W, e] = ρ*w
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
        ρ, U, V, W, E = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e], Q[n, _W, e], Q[n, _E, e]
        u=U/ρ
        v=V/ρ
        w=W/ρ
        T=E/(ρ*c_v)
        P=ρ*R_gas*T
        E=p0/(ρ* R_gas)*(P/p0)^(c_v/c_p)
        Q[n, _U, e] = u
        Q[n, _V, e] = v
        Q[n, _W, e] = w
        Q[n, _E, e] = E
    end
end
# }}}

# {{{ euler driver
function euler(::Val{dim}, ::Val{N}, mpicomm, ic, mesh, tend, iplot, visc;
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
        x, y, z = vgeo[i, _x, e], vgeo[i, _y, e], vgeo[i, _z, e]
        ρ, U, V, W, E = ic(x, y, z)
        Q[i, _ρ, e] = ρ
        Q[i, _U, e] = U
        Q[i, _V, e] = V
        Q[i, _W, e] = W
        Q[i, _E, e] = E
    end

    # Convert to proper variables
    mpirank == 0 && println("[CPU] converting variables (CPU)...")
    convert_set2nc_to_set3c(Val(dim), Val(N), vgeo, Q)

    # Compute time step
    mpirank == 0 && println("[CPU] computing dt (CPU)...")
    (base_dt, Courant) = courantnumber(Val(dim), Val(N), vgeo, Q, mpicomm)
    #base_dt=0.02
    mpirank == 0 && @show (base_dt, Courant)

    nsteps = ceil(Int64, tend / base_dt)
    dt = tend / nsteps
    mpirank == 0 && @show (dt, nsteps, dt * nsteps, tend)

    # Do time stepping
    stats = zeros(DFloat, 2)
    mpirank == 0 && println("[CPU] computing initial energy...")
    Q_temp=copy(Q)
    convert_set3c_to_set2nc(Val(dim), Val(N), vgeo, Q_temp)
    stats[1] = L2energysquared(Val(dim), Val(N), Q_temp, vgeo, mesh.realelems)
    @show (sqrt.(stats[1]))

    # Write VTK file: plot the initial condition
    mkpath("viz")
    X = ntuple(j->reshape((@view vgeo[:, _x+j-1, :]), ntuple(j->N+1,dim)...,
                          nelem), dim)
    ρ = reshape((@view Q_temp[:, _ρ, :]), ntuple(j->(N+1),dim)..., nelem)
    U = reshape((@view Q_temp[:, _U, :]), ntuple(j->(N+1),dim)..., nelem)
    V = reshape((@view Q_temp[:, _V, :]), ntuple(j->(N+1),dim)..., nelem)
    W = reshape((@view Q_temp[:, _W, :]), ntuple(j->(N+1),dim)..., nelem)
    E = reshape((@view Q_temp[:, _E, :]), ntuple(j->(N+1),dim)..., nelem)
    E = E .- 300.0
    writemesh(@sprintf("viz/euler%dD_set3c_%s_rank_%04d_step_%05d",
                       dim, ArrType, mpirank, 0), X...;
              fields=(("ρ", ρ), ("U", U), ("V", V), ("W", W), ("E", E)),
              realelems=mesh.realelems)

    mpirank == 0 && println("[DEV] starting time stepper...")
    lowstorageRK(Val(dim), Val(N), mesh, vgeo, sgeo, Q, rhs, D, dt, nsteps, tout,
                 vmapM, vmapP, mpicomm, iplot, visc; ArrType=ArrType, plotstep=plotstep)

    # Write VTK: final solution
    Q_temp=copy(Q)
    convert_set3c_to_set2nc(Val(dim), Val(N), vgeo, Q_temp)
    X = ntuple(j->reshape((@view vgeo[:, _x+j-1, :]), ntuple(j->N+1,dim)...,
                          nelem), dim)
    ρ = reshape((@view Q_temp[:, _ρ, :]), ntuple(j->(N+1),dim)..., nelem)
    U = reshape((@view Q_temp[:, _U, :]), ntuple(j->(N+1),dim)..., nelem)
    V = reshape((@view Q_temp[:, _V, :]), ntuple(j->(N+1),dim)..., nelem)
    W = reshape((@view Q_temp[:, _W, :]), ntuple(j->(N+1),dim)..., nelem)
    E = reshape((@view Q_temp[:, _E, :]), ntuple(j->(N+1),dim)..., nelem)
    E = E .- 300.0
    writemesh(@sprintf("viz/euler%dD_set3c_%s_rank_%04d_step_%05d",
                       dim, ArrType, mpirank, nsteps), X...;
              fields=(("ρ", ρ), ("U", U), ("V", V), ("W", W), ("E", E)),
              realelems=mesh.realelems)

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
        W = 0.0
        E = θ_k
        ρ, U, V, W, E
    end

    time_final = DFloat(300.0)
    iplot=100
    Ne = 10
    N  = 4
    visc = 0
    dim = 3
    hardware="cpu"
    @show (N,Ne,visc,iplot,time_final,hardware,mpisize)

    mesh3D = brickmesh((range(DFloat(0); length=Ne+1, stop=1000),
                        range(DFloat(0); length=2, stop=1000),
                        range(DFloat(0); length=Ne+1, stop=1000)),
                       (true, true, false),
                       part=mpirank+1, numparts=mpisize)

    if hardware == "cpu"
        mpirank == 0 && println("Running 3d (CPU)...")
        euler(Val(dim), Val(N), mpicomm, (x...)->ic(dim, x...), mesh3D, time_final, iplot, visc;
              ArrType=Array, tout = 10)
        mpirank == 0 && println()
    elseif hardware == "gpu"
        @hascuda begin
            mpirank == 0 && println("Running 3d (GPU)...")
            euler(Val(dim), Val(N), mpicomm, (x...)->ic(dim, x...), mesh3D, time_final, iplot, visc;
                  ArrType=CuArray, tout = 10)
            mpirank == 0 && println()
        end
    end
    nothing
end
# }}}

main()
