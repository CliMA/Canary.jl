#--------------------------------Markdown Language Header-----------------------
# # 2D Compressible Navier-Stokes Equations
#
#
#-
#
#-
# ## Introduction
#
# This example shows how to solve the 2D compressible Navier-Stokes equations using vanilla DG.
#
# ## Continuous Governing Equations
# We solve the following equation:
#
# ```math
# \frac{\partial \rho}{\partial t} + \nabla \cdot \mathbf{U} = 0 \; \; (1.1)
# ```
# ```math
# \frac{\partial \mathbf{U}}{\partial t} + \nabla \cdot \left( \frac{\mathbf{U} \otimes \mathbf{U}}{\rho} + P \mathbf{I}_2 \right) + \rho g \hat{\mathbf{k}}= \nabla \cdot \mathbf{F}_U^{visc} \; \; (1.2)
# ```
# ```math
# \frac{\partial E}{\partial t} + \nabla \cdot \left( \frac{\mathbf{U} \left(E+P \right)}{\rho} \right = \nabla \cdot \mathbf{F}_E^{visc} \; \; (1.3)
# ```
# where $\mathbf{u}=(u,v)$ is the velocity, $\mathbf{U}=\rho \mathbf{u}$, is the momentum, with $\rho$ the total density and $E=(\gamma-1) \rho \left( c_v T + \frac{1}{2} \mathbf{u} \cdot \mathbf{u} + g z \right)$ the total energy (internal $+$ kinetic $+$ potential).
# The viscous fluxes are defined as follows
# ```math
# \mathbf{F}_U^{visc} = \mu \left\[ \nabla \mathbf{u} +  \lambda \left( \nabla \mathbf{u} \right)^T + \nabla \cdot \mathbf{u}  \mathbf{I}_2 \right\]
# ```
# and
# ```math
# \mathbf{F}_E^{visc} =  \mathbf{u} \cdot \mathbf{F}_U^{visc} + \frac{c_p/Pr} \nabla T
# ```
# where $\mu$ is the kinematic (or artificial) viscosity, $\lambda=-\frac{2}{3}$ is the Stokes hypothesis, $Pr \approx 0.71$ is the Prandtl number for air and $T$ is the temperature.
# We employ periodic boundary conditions in the horizontaland no-flux boundary conditions in the vertical.  At the bottom and top of the domain, we need to impose no-flux boundary conditions in $\nabla T$ to avoid a (artificial) thermal boundary layer.
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
# where the second term on the left denotes the flux integral term (computed in "function flux\_rhs") and the third term denotes the volume integral term (computed in "function volume\_rhs").  The superscript $(*,e)$ in the flux integral term denotes the numerical flux. Here we use the Rusanov flux.
#
#-
# ## Local Discontinous Galerkin Method
# To approximate the second order terms on the right hand side of Eq.\ (1) we use the local discontinuous Galerkin (LDG) method, which we described in LDG2d.jl. We will highlight the main steps below for completeness. We employ the following two-step process: first we approximate the gradient of $q$ as follows
# ```math
# \mathbf{Q}(x,y) = \nabla \vc{q}(x,y) \; \; (2)
# ```
# where $\mathbf{Q}$ is an auxiliary vector function, followed by
# ```math
# \nabla \cdot \left \mathbf{F}^{visc}\left( \mathbf{Q} \right) \; \; (3)
# ```
# which completes the approximation of the second order derivatives.
#
#-
# ## Commented Program
#
#
#
# Steps to load MoistThermodynamics.jl
# cd /PATH/TO/CLIMATE/src/
#
# Julia > import Pkg
# ]
# Pkg > dev PlanetParameters/
# Pkg > dev Parameters/
# Pkg > dev Shared/
#
# cd /PATH/TO/myCanary.jl/Canary.jl/
#
#
#--------------------------------Markdown Language Header-----------------------
include(joinpath(@__DIR__,"vtk.jl"))
include("/Users/simone/Work/Tapio/CLIMA/src/Utilities/src/MoistThermodynamics.jl")

using PlanetParameters
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

const _nsd = 2 #number of space dimensions

# {{{ constants
# note the order of the fields below is also assumed in the code.
#
#
# FIND A WAY TO CLEAR PREVIPOUSLY COMPILED CONSTS
#
#
# DEFINE CASE AND PRE_COMPILED QUANTITIES:
#

#_icase = 1    #RTB
#_icase = 1001 # RTB + 1 Passive tracer
_icase = 1003 # RTB + 3 Passive tracers
#_icase = 1010 # Moist case of Pressel et al. 2015 JAMES
if(_icase < 1000)
    DRY_CASE = true
else
    DRY_CASE = false
end

if DRY_CASE

    const _nstate   = _nsd + 2
    const _ntracers = 0
    const _U, _V, _ρ, _E = 1:_nstate
    const stateid = (U = _U, V = _V, ρ = _ρ, E = _E)
    
else
    
    if (_icase == 1001)
        #
        # RTB + passive tracer
        #
        const _ntracers = 1
        const _nstate = (_nsd + 2) + _ntracers
        const _U, _V, _ρ, _E, _qt = 1:_nstate
        const stateid = (U = _U, V = _V, ρ = _ρ, E = _E, qt = _qt)

    elseif (_icase == 1002)
        #
        # Moist dynamics
        #
        error(" _icase 1002 not coded yet")
        
    elseif (_icase == 1003)
        #
        # RTB + 3 passive tracers
        #
        const _ntracers = 3
        const _nstate = (_nsd + 2) + _ntracers
        const _U, _V, _ρ, _E, _qt1, _qt2, _qt3 = 1:_nstate
        const stateid = (U = _U, V = _V, ρ = _ρ, E = _E, qt1 = _qt1, qt2 = _qt2, qt3 = _qt3)

    elseif (_icase == 1010)
        #
        # Moist case of Pressel et al. 2015 JAMES:
        # Total water qt, ql, qi:
        #        
        const _ntracers = 3
        const _nstate = (_nsd + 2) + _ntracers
        const _U, _V, _ρ, _E, _qt1, _qt2, _qt3 = 1:_nstate
        const stateid = (U = _U, V = _V, ρ = _ρ, E = _E, qt1 = _qt1, qt2 = _qt2, qt3 = _qt3)
        
        #const _ntracers = 1
        #const _nstate = (_nsd + 2) + _ntracers
        #const _U, _V, _ρ, _E, _qt = 1:_nstate
        #const stateid = (U = _U, V = _V, ρ = _ρ, E = _E, qt = _qt)
                
    else
        #
        # Moist dynamics
        #
        error(" USER INPUT ERROR: _icase ", _icase,  " not coded yet")
    end
end

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
const _Prandtl = 71 // 100
const _Stokes = -2 // 3

const _C1 = 10/10
const _C2 = 5/10
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
        ρ, U, V, E = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e], Q[n, _E, e]
        ξx, ξy, ηx, ηy = vgeo[n, _ξx, e], vgeo[n, _ξy, e], vgeo[n, _ηx, e], vgeo[n, _ηy, e]
        y = vgeo[n, _y, e]
        P = (R_gas/c_v)*(E - (U^2 + V^2)/(2*ρ) - ρ*gravity*y)
        u, v = U/ρ, V/ρ
        dx=sqrt( (1.0/(2*ξx))^2 + (1.0/(2*ηy))^2 )
        vel=sqrt( u^2 + v^2 )
        wave_speed = (vel + sqrt(γ * P / ρ))
        loc_dt = 1.0*dx/wave_speed/N
        dt[1] = min(dt[1], loc_dt)
    end
    dt_min=MPI.Allreduce(dt[1], MPI.MIN, mpicomm)
    
    #Compute Courant
    @inbounds for e = 1:nelem, n = 1:Np
        ρ, U, V, E = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e], Q[n, _E, e]
        ξx, ξy, ηx, ηy = vgeo[n, _ξx, e], vgeo[n, _ξy, e], vgeo[n, _ηx, e], vgeo[n, _ηy, e]
        y = vgeo[n, _y, e]
        P = (R_gas/c_v)*(E - (U^2 + V^2)/(2*ρ) - ρ*gravity*y)
        u, v = U/ρ, V/ρ
        dx=sqrt( (1.0/(2*ξx))^2 + (1.0/(2*ηy))^2 )
        vel=sqrt( u^2 + v^2 )
        wave_speed = (vel + sqrt(γ * P / ρ))
        loc_Courant = wave_speed*dt_min/dx*N
        dx=sqrt( (1.0/(2*ξx))^2 + (1.0/(2*ηy))^2 )
        vel=sqrt( u^2 + v^2 )
        wave_speed = (vel + sqrt(γ * P / ρ))
        loc_Courant = wave_speed*dt_min/dx*N
        Courant[1] = max(Courant[1], loc_Courant)
    end
    Courant_max=MPI.Allreduce(Courant[1], MPI.MAX, mpicomm)

    (dt_min, Courant_max)
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
# compute wave speed
# SM Jan 7
function compute_wave_speed(ρMinv, ρPinv, nxM, nyM, UM, VM, UP, VP, PM, PP, γ)

    λM = ρMinv * abs(nxM * UM + nyM * VM) + sqrt(ρMinv * γ * PM)
    λP = ρPinv * abs(nxM * UP + nyM * VP) + sqrt(ρPinv * γ * PP)
    λ  =  max(λM, λP)
    
    return λ
    
end

# {{{ CPU Kernels
# build volume fluxes (RENAME IT ?)
# Make the flux construction invisible to the user
# who does not know DG
# SM Jan 7
function build_volume_fluxes_ije(ξx, ξy, MJ, fluxQ_x, fluxQ_y)

    volume_flux = MJ * (ξx * fluxQ_x + ξy * fluxQ_y)
    
    return volume_flux
    
end

# {{{ CPU Kernels
# build surface fluxes (RENAME IT ?)
# Make the flux construction invisible to the user
# who does not know DG
# SM Jan 7
function build_surface_fluxes_ije( nxM, nyM,
                                   fluxQM_x, fluxQP_x,
                                   fluxQM_y, fluxQP_y,
                                   wave_speed, QM, QP)
    
    #Compute Numerical Flux
    fluxQS = (nxM * (fluxQM_x + fluxQP_x) + nyM * (fluxQM_y + fluxQP_y) +
              - wave_speed * (QP - QM)) / 2

    return fluxQS

end

# {{{ CPU Kernels
# Build int_{\Omega} \nabla\psi.F d\Omega
# SM Jan 7
function integrate_volume_rhs!(rhs,                 #in/out
                               e, D, Nq, s_F, s_G)
    
        # loop of ξ-grid lines
        for s = 1:_nstate, j = 1:Nq, i = 1:Nq, k = 1:Nq
            rhs[i, j, s, e] += D[k, i] * s_F[k, j, s]
        end
        # loop of η-grid lines
        for s = 1:_nstate, j = 1:Nq, i = 1:Nq, k = 1:Nq
            rhs[i, j, s, e] += D[k, j] * s_G[i, k, s]
        end
end

#
# {{{ CPU Kernels
# Volume RHS
function volume_rhs!(::Val{2}, ::Val{N}, rhs::Array, Q, vgeo, D, elems) where N
    DFloat          = eltype(Q)
    γ::DFloat       = _γ
    p0::DFloat      = _p0
    R_gas::DFloat   = _R_gas
    c_p::DFloat     = _c_p
    c_v::DFloat     = _c_v
    gravity::DFloat = _gravity

    Nq = N + 1
    nelem = size(Q)[end]

    Q    = reshape(Q, Nq, Nq, _nstate, nelem)
    rhs  = reshape(rhs, Nq, Nq, _nstate, nelem)
    vgeo = reshape(vgeo, Nq, Nq, _nvgeo, nelem)

    #Allocate Arrays
    s_F = Array{DFloat}(undef, Nq, Nq, _nstate)
    s_G = Array{DFloat}(undef, Nq, Nq, _nstate)

    #Allocate and initialize to zero tracer flux quantities
    fluxQT_x = zeros(DFloat, _ntracers)
    fluxQT_y = zeros(DFloat, _ntracers)
    
    q_tr = zeros(DFloat, _ntracers)
    
    @inbounds for e in elems
        for j = 1:Nq, i = 1:Nq
            MJ = vgeo[i, j, _MJ, e]
            ξx, ξy = vgeo[i,j,_ξx,e], vgeo[i,j,_ξy,e]
            ηx, ηy = vgeo[i,j,_ηx,e], vgeo[i,j,_ηy,e]
            y = vgeo[i,j,_y,e]

            #=Moist air constant: Rm
            q_tr[1] = 0.0
            q_tr[2] = 0.0
            q_tr[3] = 0.0
            for itracer = 1:_ntracers
                istate = itracer + (_nsd+2)

                q_tr[itracer]   = Q[i, j, istate, e]
            end        
            R_gas = MoistThermodynamics.gas_constant_air(q_tr[1], q_tr[2], q_tr[3])=#
            R_gas = MoistThermodynamics.gas_constant_air(0.0, 0.0, 0.0)
            
            U, V = Q[i, j, _U, e], Q[i, j, _V, e]
            ρ, E = Q[i, j, _ρ, e], Q[i, j, _E, e]
            P = (R_gas/c_v)*(E - (U^2 + V^2)/(2*ρ) - ρ*gravity*y)
            
            ρinv = 1 / ρ
            fluxρ_x = U
            fluxU_x = ρinv * U * U + P
            fluxV_x = ρinv * V * U
            fluxE_x = ρinv * U * (E+P)

            fluxρ_y = V
            fluxU_y = ρinv * U * V
            fluxV_y = ρinv * V * V + P
            fluxE_y = ρinv * V * (E+P)
            
            # Build volume fluxes
            s_F[i,j,_ρ]  = build_volume_fluxes_ije(ξx, ξy, MJ, fluxρ_x, fluxρ_y)
            s_F[i,j,_U]  = build_volume_fluxes_ije(ξx, ξy, MJ, fluxU_x, fluxU_y)
            s_F[i,j,_V]  = build_volume_fluxes_ije(ξx, ξy, MJ, fluxV_x, fluxV_y)
            s_F[i,j,_E]  = build_volume_fluxes_ije(ξx, ξy, MJ, fluxE_x, fluxE_y)
            
            s_G[i,j,_ρ]  = build_volume_fluxes_ije(ηx, ηy, MJ, fluxρ_x, fluxρ_y)
            s_G[i,j,_U]  = build_volume_fluxes_ije(ηx, ηy, MJ, fluxU_x, fluxU_y)
            s_G[i,j,_V]  = build_volume_fluxes_ije(ηx, ηy, MJ, fluxV_x, fluxV_y)
            s_G[i,j,_E]  = build_volume_fluxes_ije(ηx, ηy, MJ, fluxE_x, fluxE_y)

            
            #Tracers
            @inbounds for itracer = 1:_ntracers
                istate = itracer + (_nsd+2)
                
                QT                = Q[i, j, istate, e]
                fluxQT_x[itracer] = ρinv * U * QT
                fluxQT_y[itracer] = ρinv * V * QT
                
                s_F[i,j,istate] = build_volume_fluxes_ije(ξx, ξy, MJ, fluxQT_x[itracer], fluxQT_y[itracer])
                s_G[i,j,istate] = build_volume_fluxes_ije(ηx, ηy, MJ, fluxQT_x[itracer], fluxQT_y[itracer])
            end
            
            #
            # buoyancy term
            #
            rhs[i, j, _V, e] -= MJ * ρ * gravity
        end

        #
        #Build int_{\Omega} \nabla\psi.F d\Omega
        #
        integrate_volume_rhs!(rhs, e, D, Nq, s_F, s_G)

    end
end

# Flux RHS
function flux_rhs!(::Val{dim}, ::Val{N}, rhs::Array, Q, sgeo, vgeo, elems, vmapM,
                  vmapP, elemtobndy) where {dim, N}
    DFloat = eltype(Q)
    γ::DFloat       = _γ
    p0::DFloat      = _p0
    R_gas::DFloat   = _R_gas
    c_p::DFloat     = _c_p
    c_v::DFloat     = _c_v
    gravity::DFloat = _gravity

    Np    = (N+1)^dim
    Nfp   = (N+1)^(dim-1)
    nface = 2*dim
    q_tr  = zeros(DFloat, 3)

    #Allocate and initialize to zero tracer flux quantities
    QTM = zeros(DFloat, _ntracers)
    QTP = zeros(DFloat, _ntracers)
    fluxQTM_x = zeros(DFloat, _ntracers)
    fluxQTM_y = zeros(DFloat, _ntracers)
    fluxQTP_x = zeros(DFloat, _ntracers)
    fluxQTP_y = zeros(DFloat, _ntracers)
    
    q_tr = zeros(DFloat, _ntracers)
    
    @inbounds for e in elems
        for f = 1:nface
            for n = 1:Nfp
                nxM, nyM, sMJ = sgeo[_nx, n, f, e], sgeo[_ny, n, f, e], sgeo[_sMJ, n, f, e]
                idM, idP = vmapM[n, f, e], vmapP[n, f, e]

                eM, eP = e, ((idP - 1) ÷ Np) + 1
                vidM, vidP = ((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1

                #Left variables
                ρM = Q[vidM, _ρ, eM]
                UM = Q[vidM, _U, eM]
                VM = Q[vidM, _V, eM]
                EM = Q[vidM, _E, eM]
                yM = vgeo[vidM, _y, eM]

                #=Moist air constant: Rm
                q_tr[1] = 0.0
                q_tr[2] = 0.0
                q_tr[3] = 0.0
                for itracer = 1:_ntracers
                    istate = itracer + (_nsd+2)
                    
                    q_tr[itracer] = Q[vidM, istate, eM]
                end
                R_gas = MoistThermodynamics.gas_constant_air(q_tr[1], q_tr[2], q_tr[3])=#
                R_gas = MoistThermodynamics.gas_constant_air(0.0, 0.0, 0.0)
                
                PM = (R_gas/c_v)*(EM - (UM^2 + VM^2)/(2*ρM) - ρM*gravity*yM)
                
                #Tracers
                @inbounds for itracer = 1:_ntracers
                    istate = itracer + (_nsd+2)
                    QTM[itracer] = Q[vidM, istate, eM]
                end
                
                #Right variables
                bc = elemtobndy[f, e]
                ρP = UP = VP = EP = PP = zero(eltype(Q))
                
                #Tracers
                @inbounds for itracer = 1:_ntracers
                    QTP[itracer] = zero(eltype(Q))
                end
                
                if bc == 0
                    ρP = Q[vidP, _ρ, eP]
                    UP = Q[vidP, _U, eP]
                    VP = Q[vidP, _V, eP]
                    EP = Q[vidP, _E, eP]
                    yP = vgeo[vidP, _y, eP]
                    PP = (R_gas/c_v)*(EP - (UP^2 + VP^2)/(2*ρP) - ρP*gravity*yP)
                    #Tracers
                    @inbounds for itracer = 1:_ntracers
                        istate = itracer + (_nsd+2)
                        QTP[itracer] = Q[vidP, istate, eP]
                    end
                    
                elseif bc == 1
                    UnM = nxM * UM + nyM * VM
                    UP = UM - 2 * UnM * nxM
                    VP = VM - 2 * UnM * nyM
                    ρP = ρM
                    EP = EM
                    PP = PM
                    #Tracers
                    @inbounds for itracer = 1:_ntracers
                        QTP[itracer] = QTM[itracer]
                    end
                    
                else
                    error("Invalid boundary conditions $bc on face $f of element $e")
                end
                
                #Left fluxes
                ρMinv = 1 / ρM
                fluxρM_x  = UM
                fluxUM_x  = ρMinv * UM * UM + PM
                fluxVM_x  = ρMinv * VM * UM
                fluxEM_x  = ρMinv * UM * (EM+PM)
               
                fluxρM_y  = VM
                fluxUM_y  = ρMinv * UM * VM
                fluxVM_y  = ρMinv * VM * VM + PM
                fluxEM_y  = ρMinv * VM * (EM+PM)
                
                #Right fluxes
                ρPinv = 1 / ρP
                fluxρP_x = UP
                fluxUP_x = ρPinv * UP * UP + PP
                fluxVP_x = ρPinv * VP * UP
                fluxEP_x = ρPinv * UP * (EP+PP)
                
                fluxρP_y = VP
                fluxUP_y = ρPinv * UP * VP
                fluxVP_y = ρPinv * VP * VP + PP
                fluxEP_y = ρPinv * VP * (EP+PP)
                
                # build surface fluxes 
                #Compute Wave Speed
                λ = compute_wave_speed(ρMinv, ρPinv, nxM, nyM, UM, VM, UP, VP, PM, PP, γ)
                
                #Compute Numerical Flux
                fluxρS  = build_surface_fluxes_ije( nxM, nyM, fluxρM_x, fluxρP_x, fluxρM_y, fluxρP_y, λ, ρM, ρP )
                fluxUS  = build_surface_fluxes_ije( nxM, nyM, fluxUM_x, fluxUP_x, fluxUM_y, fluxUP_y, λ, UM, UP )
                fluxVS  = build_surface_fluxes_ije( nxM, nyM, fluxVM_x, fluxVP_x, fluxVM_y, fluxVP_y, λ, VM, VP )
                fluxES  = build_surface_fluxes_ije( nxM, nyM, fluxEM_x, fluxEP_x, fluxEM_y, fluxEP_y, λ, EM, EP )
                

                #Update RHS
                rhs[vidM, _ρ, eM]  -= sMJ * fluxρS
                rhs[vidM, _U, eM]  -= sMJ * fluxUS
                rhs[vidM, _V, eM]  -= sMJ * fluxVS
                rhs[vidM, _E, eM]  -= sMJ * fluxES

                
                #Tracers
                @inbounds for itracer = 1:_ntracers
                    istate = itracer + (_nsd+2)

                    #Right and left fluxes
                    fluxQTM_x[itracer] = ρMinv * UM * QTM[itracer]
                    fluxQTM_y[itracer] = ρMinv * VM * QTM[itracer]
                    
                    fluxQTP_x[itracer] = ρPinv * UP * QTP[itracer]
                    fluxQTP_y[itracer] = ρPinv * VP * QTP[itracer]

                    #Compute Numerical Flux
                    fluxQTS = build_surface_fluxes_ije( nxM, nyM, fluxQTM_x[itracer], fluxQTP_x[itracer], fluxQTM_y[itracer], fluxQTP_y[itracer], λ, QTM[itracer], QTP[itracer] )

                    #Update RHS LOOP HERE
                    rhs[vidM, istate, eM] -= sMJ * fluxQTS
                end
                
            end
        end
    end
end
# }}}

# {{{ Volume grad(Q)
function volume_grad!(::Val{dim}, ::Val{N}, rhs::Array, Q, vgeo, D, elems) where {dim, N}
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
    rhs = reshape(rhs, Nq, Nq, _nstate, dim, nelem)
    vgeo = reshape(vgeo, Nq, Nq, _nvgeo, nelem)

    #Initialize RHS vector
    fill!( rhs, zero(rhs[1]))

    #Allocate Arrays
    s_F = Array{DFloat}(undef, Nq, Nq, _nstate, dim)
    s_G = Array{DFloat}(undef, Nq, Nq, _nstate, dim)

    q_tr = zeros(DFloat, _ntracers)
    
    @inbounds for e in elems
        for j = 1:Nq, i = 1:Nq
            MJ = vgeo[i, j, _MJ, e]
            ξx, ξy = vgeo[i,j,_ξx,e], vgeo[i,j,_ξy,e]
            ηx, ηy = vgeo[i,j,_ηx,e], vgeo[i,j,_ηy,e]
            y = vgeo[i,j,_y,e]

            U, V = Q[i, j, _U, e], Q[i, j, _V, e]
            ρ, E = Q[i, j, _ρ, e], Q[i, j, _E, e]

            #Moist air constant: Rm
            q_tr[1] = 0.0
            q_tr[2] = 0.0
            q_tr[3] = 0.0
            #for itracer = 1:_ntracers
            #    istate = itracer + (_nsd+2)
            #    
            #    q_tr[itracer]       = Q[i, j, istate, e]
            #end
            #R_gas = MoistThermodynamics.gas_constant_air(q_tr[1], q_tr[2], q_tr[3])
            R_gas = MoistThermodynamics.gas_constant_air(0.0, 0.0, 0.0)

            
            P = (R_gas/c_v)*(E - (U^2 + V^2)/(2*ρ) - ρ*gravity*y)
            
            #Primitive variables
            u=U/ρ
            v=V/ρ
            T=P/(R_gas*ρ)

            #Compute fluxes
            fluxρ  = ρ
            fluxU  = u
            fluxV  = v
            fluxE  = T
            
            s_F[i, j, _ρ, 1], s_F[i, j, _ρ, 2] = build_volume_fluxes_ije(ξx, 0.0, MJ, fluxρ, 0.0),  build_volume_fluxes_ije(0.0, ξy, MJ, 0.0, fluxρ)
            s_F[i, j, _U, 1], s_F[i, j, _U, 2] = build_volume_fluxes_ije(ξx, 0.0, MJ, fluxU, 0.0),  build_volume_fluxes_ije(0.0, ξy, MJ, 0.0, fluxU)
            s_F[i, j, _V, 1], s_F[i, j, _V, 2] = build_volume_fluxes_ije(ξx, 0.0, MJ, fluxV, 0.0),  build_volume_fluxes_ije(0.0, ξy, MJ, 0.0, fluxV)
            s_F[i, j, _E, 1], s_F[i, j, _E, 2] = build_volume_fluxes_ije(ξx, 0.0, MJ, fluxE, 0.0),  build_volume_fluxes_ije(0.0, ξy, MJ, 0.0, fluxE)

            s_G[i, j, _ρ, 1], s_G[i, j, _ρ, 2] = build_volume_fluxes_ije(ηx, 0.0, MJ, fluxρ, 0.0),  build_volume_fluxes_ije(0.0, ηy, MJ, 0.0, fluxρ)
            s_G[i, j, _U, 1], s_G[i, j, _U, 2] = build_volume_fluxes_ije(ηx, 0.0, MJ, fluxU, 0.0),  build_volume_fluxes_ije(0.0, ηy, MJ, 0.0, fluxU)
            s_G[i, j, _V, 1], s_G[i, j, _V, 2] = build_volume_fluxes_ije(ηx, 0.0, MJ, fluxV, 0.0),  build_volume_fluxes_ije(0.0, ηy, MJ, 0.0, fluxV)
            s_G[i, j, _E, 1], s_G[i, j, _E, 2] = build_volume_fluxes_ije(ηx, 0.0, MJ, fluxE, 0.0),  build_volume_fluxes_ije(0.0, ηy, MJ, 0.0, fluxE)

            #Tracers
            for itracer = 1:_ntracers
                istate = itracer + (_nsd+2)
                
                #Compute fluxes
                QT     = Q[i, j, istate, e]
                fluxQT = QT
            
                s_F[i, j, istate, 1], s_F[i, j, istate, 2] = build_volume_fluxes_ije(ξx, 0.0, MJ, fluxQT, 0.0),  build_volume_fluxes_ije(0.0, ξy, MJ, 0.0, fluxQT)
                s_G[i, j, istate, 1], s_G[i, j, istate, 2] = build_volume_fluxes_ije(ηx, 0.0, MJ, fluxQT, 0.0),  build_volume_fluxes_ije(0.0, ηy, MJ, 0.0, fluxQT)
            end
            
        end
        
        # loop of ξ-grid lines
        for s = 1:_nstate, j = 1:Nq, i = 1:Nq, n = 1:Nq
            rhs[i, j, s, 1, e] -= D[n, i] * s_F[n, j, s, 1]
            rhs[i, j, s, 2, e] -= D[n, i] * s_F[n, j, s, 2]
        end
        # loop of η-grid lines
        for s = 1:_nstate, j = 1:Nq, i = 1:Nq, n = 1:Nq
            rhs[i, j, s, 1, e] -= D[n, j] * s_G[i, n, s, 1]
            rhs[i, j, s, 2, e] -= D[n, j] * s_G[i, n, s, 2]
        end
    end
end
# }}}

# Flux grad(Q)
function flux_grad!(::Val{dim}, ::Val{N}, rhs::Array,  Q, sgeo, vgeo, elems, vmapM, vmapP, elemtobndy) where {dim, N}
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

    QTM = zeros(DFloat, _ntracers)
    QTP = zeros(DFloat, _ntracers)
    
    q_tr = zeros(DFloat, _ntracers)
    
    @inbounds for e in elems
        for f = 1:nface
            for n = 1:Nfp
                nxM, nyM, sMJ = sgeo[_nx, n, f, e], sgeo[_ny, n, f, e], sgeo[_sMJ, n, f, e]
                idM, idP = vmapM[n, f, e], vmapP[n, f, e]

                eM, eP = e, ((idP - 1) ÷ Np) + 1
                vidM, vidP = ((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1

                #Left variables
                ρM = Q[vidM, _ρ, eM]
                UM = Q[vidM, _U, eM]
                VM = Q[vidM, _V, eM]
                EM = Q[vidM, _E, eM]
                yM = vgeo[vidM, _y, eM]
                
                #Moist air constant: Rm
                q_tr[1] = 0.0
                q_tr[2] = 0.0
                q_tr[3] = 0.0
                #for itracer = 1:_ntracers
                #    istate = itracer + (_nsd+2)
                #    
                #    q_tr[itracer]       = Q[vidM, istate, eM]
                #end
                #R_gas = MoistThermodynamics.gas_constant_air(q_tr[1], q_tr[2], q_tr[3])
                R_gas = MoistThermodynamics.gas_constant_air(0.0, 0.0, 0.0)
                
                PM = (R_gas/c_v)*(EM - (UM^2 + VM^2)/(2*ρM) - ρM*gravity*yM)
                uM = UM/ρM
                vM = VM/ρM
                TM = PM/(R_gas*ρM)

                #Tracers
                
                #Right variables
                bc = elemtobndy[f, e]
                ρP = UP = VP = EP = PP = zero(eltype(Q))

                #Tracers
                for itracer = 1:_ntracers
                    istate = itracer + (_nsd+2)

                    QTM[itracer] = Q[vidM, istate, eM]                
                    QTP[itracer] = zero(eltype(Q))
                end
                
                if bc == 0
                    ρP = Q[vidP, _ρ, eP]
                    UP = Q[vidP, _U, eP]
                    VP = Q[vidP, _V, eP]
                    EP = Q[vidP, _E, eP]
                    yP = vgeo[vidP, _y, eP]
                    PP = (R_gas/c_v)*(EP - (UP^2 + VP^2)/(2*ρP) - ρP*gravity*yP)
                    uP = UP/ρP
                    vP = VP/ρP
                    TP = PP/(R_gas*ρP)

                    #Tracers
                    for itracer = 1:_ntracers
                        istate = itracer + (_nsd+2)
                        QTP[itracer] = Q[vidP, istate, eP]
                    end
                    
                elseif bc == 1
                    UnM = nxM * UM +  nyM * VM
                    UP = UM - 2 * UnM * nxM
                    VP = VM - 2 * UnM * nyM
                    ρP = ρM
                    EP = EM
                    PP = PM
                    uP = UP/ρP
                    vP = VP/ρP
                    TP = TM

                    #Tracers
                    for itracer = 1:_ntracers
                        istate = itracer + (_nsd+2)
                        QTP[itracer] = QTM[itracer]
                    end
                    
                else
                    error("Invalid boundary conditions $bc on face $f of element $e")
                end

                #Left Fluxes
                fluxρM = ρM
                fluxUM = uM
                fluxVM = vM
                fluxEM = TM
                
                #Right Fluxes
                fluxρP = ρP
                fluxUP = uP
                fluxVP = vP
                fluxEP = TP

                
                #Compute Numerical Flux
                fluxρS = 0.5*(fluxρM + fluxρP)
                fluxUS = 0.5*(fluxUM + fluxUP)
                fluxVS = 0.5*(fluxVM + fluxVP)
                fluxES = 0.5*(fluxEM + fluxEP)

                #Update RHS
                rhs[vidM, _ρ, 1, eM] += sMJ * nxM*fluxρS
                rhs[vidM, _ρ, 2, eM] += sMJ * nyM*fluxρS
                rhs[vidM, _U, 1, eM] += sMJ * nxM*fluxUS
                rhs[vidM, _U, 2, eM] += sMJ * nyM*fluxUS
                rhs[vidM, _V, 1, eM] += sMJ * nxM*fluxVS
                rhs[vidM, _V, 2, eM] += sMJ * nyM*fluxVS
                rhs[vidM, _E, 1, eM] += sMJ * nxM*fluxES
                rhs[vidM, _E, 2, eM] += sMJ * nyM*fluxES

                #Tracers
                for itracer = 1:_ntracers
                    istate = itracer + (_nsd+2)

                    #Compute Numerical Flux
                    fluxQTM = QTM[itracer]
                    fluxQTP = QTP[itracer]
                    fluxQTS = 0.5*(fluxQTM + fluxQTP)

                    #Update RHS
                    rhs[vidM, istate, 1, eM] += sMJ * nxM*fluxQTS
                    rhs[vidM, istate, 2, eM] += sMJ * nyM*fluxQTS
                end
                
            end
        end
    end
end
# }}}

# {{{ Volume div(grad(Q))
function volume_div!(::Val{dim}, ::Val{N}, rhs::Array, gradQ, Q, visc_sgs, vgeo, D, elems) where {dim, N}
    DFloat = eltype(Q)
    γ::DFloat       = _γ
    p0::DFloat      = _p0
    R_gas::DFloat   = _R_gas
    c_p::DFloat     = _c_p
    c_v::DFloat     = _c_v
    gravity::DFloat = _gravity
    Pr::DFloat = _Prandtl
    lambda::DFloat = _Stokes

    Nq = N + 1
    nelem = size(Q)[end]

    Q     = reshape(Q, Nq, Nq, _nstate, nelem)
    gradQ = reshape(gradQ, Nq, Nq, _nstate, dim, nelem)
    rhs   = reshape(rhs, Nq, Nq, _nstate, dim, nelem)
    vgeo  = reshape(vgeo, Nq, Nq, _nvgeo, nelem)

    #Initialize RHS vector
    fill!( rhs, zero(rhs[1]))

    #Allocate Arrays
    s_F = Array{DFloat}(undef, Nq, Nq, _nstate)
    s_G = Array{DFloat}(undef, Nq, Nq, _nstate)

    @inbounds for e in elems
        for j = 1:Nq, i = 1:Nq
            MJ = vgeo[i, j, _MJ, e]
            ξx, ξy = vgeo[i,j,_ξx,e], vgeo[i,j,_ξy,e]
            ηx, ηy = vgeo[i,j,_ηx,e], vgeo[i,j,_ηy,e]
            
            ρx, ρy = gradQ[i, j, _ρ, 1, e], gradQ[i, j, _ρ, 2, e]
            Ux, Uy = gradQ[i, j, _U, 1, e], gradQ[i, j, _U, 2, e]
            Vx, Vy = gradQ[i, j, _V, 1, e], gradQ[i, j, _V, 2, e]
            Ex, Ey = gradQ[i, j, _E, 1, e], gradQ[i, j, _E, 2, e]
            ρ, U, V = Q[i, j, _ρ, e], Q[i, j, _U, e], Q[i, j, _V, e]

           
            #Compute primitive variables
            ux, uy = Ux, Uy
            vx, vy = Vx, Vy
            Tx, Ty = Ex, Ey
            div_u  = ux + vy
            u      = U/ρ
            v      = V/ρ

            #Store viscosity coefficients
            ρ=1 #assumes visc_sgs in compute_viscosity_sgs has density included
            μ=ρ*visc_sgs[1,e]
            κ=ρ*visc_sgs[2,e]
            ν=ρ*visc_sgs[3,e]

            #Compute fluxes
            fluxρ_x = ν*ρx
            fluxρ_y = ν*ρy
            fluxU_x = μ*(2*ux + lambda*div_u)
            fluxU_y = μ*(uy + vx)
            fluxV_x = μ*(vx + uy)
            fluxV_y = μ*(2*vy + lambda*div_u)
            fluxE_x = μ*(u*(2*ux + lambda*div_u) + v*(uy + vx)) + κ*c_p/Pr*Tx
            fluxE_y = μ*(u*(vx + uy) + v*(2*vy + lambda*div_u)) + κ*c_p/Pr*Ty
            
            s_F[i, j, _ρ] = build_volume_fluxes_ije(ξx, ξy, MJ, fluxρ_x, fluxρ_y)
            s_F[i, j, _U] = build_volume_fluxes_ije(ξx, ξy, MJ, fluxU_x, fluxU_y)
            s_F[i, j, _V] = build_volume_fluxes_ije(ξx, ξy, MJ, fluxV_x, fluxV_y)
            s_F[i, j, _E] = build_volume_fluxes_ije(ξx, ξy, MJ, fluxE_x, fluxE_y)
            
            s_G[i, j, _ρ] = build_volume_fluxes_ije(ηx, ηy, MJ, fluxρ_x, fluxρ_y)
            s_G[i, j, _U] = build_volume_fluxes_ije(ηx, ηy, MJ, fluxU_x, fluxU_y)
            s_G[i, j, _V] = build_volume_fluxes_ije(ηx, ηy, MJ, fluxV_x, fluxV_y)
            s_G[i, j, _E] = build_volume_fluxes_ije(ηx, ηy, MJ, fluxE_x, fluxE_y)

            
            #Tracers
            for itracer = 1:_ntracers
                istate = itracer + (_nsd+2)

                QTx = gradQ[i, j, istate, 1, e]
                QTy = gradQ[i, j, istate, 2, e]
                QT  = Q[i, j, istate, e]
                
                fluxQT_x = κ*QTx
                fluxQT_y = κ*QTy
                
                s_F[i, j, istate] = build_volume_fluxes_ije(ξx, ξy, MJ, fluxQT_x, fluxQT_y)
                s_G[i, j, istate] = build_volume_fluxes_ije(ηx, ηy, MJ, fluxQT_x, fluxQT_y)
            end
            
        end

        # loop of ξ-grid lines
        for s = 1:_nstate, j = 1:Nq, i = 1:Nq, n = 1:Nq
            rhs[i, j, s, 1, e] -= D[n, i] * s_F[n, j, s]
        end
        # loop of η-grid lines
        for s = 1:_nstate, j = 1:Nq, i = 1:Nq, n = 1:Nq
            rhs[i, j, s, 1, e] -= D[n, j] * s_G[i, n, s]
        end
    end
end
# }}}

# Flux div(grad(Q))
function flux_div!(::Val{dim}, ::Val{N}, rhs::Array,  gradQ, Q, visc_sgs, sgeo, elems, vmapM, vmapP, elemtobndy) where {dim, N}
    DFloat = eltype(Q)
    γ::DFloat       = _γ
    p0::DFloat      = _p0
    R_gas::DFloat   = _R_gas
    c_p::DFloat     = _c_p
    c_v::DFloat     = _c_v
    gravity::DFloat = _gravity
    Pr::DFloat = _Prandtl
    lambda::DFloat = _Stokes

    Np = (N+1)^dim
    Nfp = (N+1)^(dim-1)
    nface = 2*dim

    #Tracers
    QTxM = zeros(DFloat, _ntracers)
    QTyM = zeros(DFloat, _ntracers)
    QTxP = zeros(DFloat, _ntracers)
    QTyP = zeros(DFloat, _ntracers) 
    QTP  = zeros(DFloat, _ntracers) 
    
    @inbounds for e in elems
        for f = 1:nface
            for n = 1:Nfp
                nxM, nyM, sMJ = sgeo[_nx, n, f, e], sgeo[_ny, n, f, e], sgeo[_sMJ, n, f, e]
                idM, idP = vmapM[n, f, e], vmapP[n, f, e]

                eM, eP = e, ((idP - 1) ÷ Np) + 1
                vidM, vidP = ((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1

                #Left variables
                ρxM = gradQ[vidM, _ρ, 1, eM]
                ρyM = gradQ[vidM, _ρ, 2, eM]
                UxM = gradQ[vidM, _U, 1, eM]
                UyM = gradQ[vidM, _U, 2, eM]
                VxM = gradQ[vidM, _V, 1, eM]
                VyM = gradQ[vidM, _V, 2, eM]
                ExM = gradQ[vidM, _E, 1, eM]
                EyM = gradQ[vidM, _E, 2, eM]
                ρM  = Q[vidM, _ρ, eM]
                UM  = Q[vidM, _U, eM]
                VM  = Q[vidM, _V, eM]              
                
                uM  = UM/ρM
                vM  = VM/ρM
                uxM, uyM = UxM, UyM
                vxM, vyM = VxM, VyM
                TxM, TyM = ExM, EyM

                #Store viscosity coefficients
                ρM=1 #assumes visc_sgs in compute_viscosity_sgs has density included
                μM=ρM*visc_sgs[1,eM]
                κM=ρM*visc_sgs[2,eM]
                νM=ρM*visc_sgs[3,eM]

                #Right variables
                bc = elemtobndy[f, e]
                ρxP = ρyP = UxP = UyP = VxP = VyP = ExP = EyP = zero(eltype(Q))
                EPflag=1.0

                #Tracers
                for itracer = 1:_ntracers
                    istate = itracer + (_nsd+2)
                    
                    QTxM[itracer] = gradQ[vidM, istate, 1, eM]
                    QTyM[itracer] = gradQ[vidM, istate, 2, eM]
                    
                    QTxP[itracer] = zero(eltype(Q))
                    QTyP[itracer] = zero(eltype(Q))
                end
                
                if bc == 0
                    ρxP = gradQ[vidP, _ρ, 1, eP]
                    ρyP = gradQ[vidP, _ρ, 2, eP]
                    UxP = gradQ[vidP, _U, 1, eP]
                    UyP = gradQ[vidP, _U, 2, eP]
                    VxP = gradQ[vidP, _V, 1, eP]
                    VyP = gradQ[vidP, _V, 2, eP]
                    ExP = gradQ[vidP, _E, 1, eP]
                    EyP = gradQ[vidP, _E, 2, eP]
                    
                    ρP = Q[vidP, _ρ, eP]
                    UP = Q[vidP, _U, eP]
                    VP = Q[vidP, _V, eP]
                    uP = UP/ρP
                    vP = VP/ρP
                    uxP, uyP = UxP, UyP
                    vxP, vyP = VxP, VyP
                    TxP, TyP = ExP, EyP
                    
                    #Tracers
                    for itracer = 1:_ntracers
                        istate = itracer + (_nsd+2)
                        
                        QTxP[itracer] = gradQ[vidP, istate, 1, eP]
                        QTyP[itracer] = gradQ[vidP, istate, 2, eP]
                        QTP[itracer]  = Q[vidP, istate, eP]
                    end
                    
                    #Store viscosity coefficients
                    ρP = 1 #assumes visc_sgs in compute_viscosity_sgs has density included
                    μP = ρP*visc_sgs[1,eP]
                    κP = ρP*visc_sgs[2,eP]
                    νP = ρP*visc_sgs[3,eP]
                elseif bc == 1
                    ρnM = nxM * ρxM +  nyM * ρyM
                    ρxP = ρxM - 2 * ρnM * nxM
                    ρyP = ρyM - 2 * ρnM * nyM
                    
                    UnM = nxM * UxM +  nyM * UyM
                    UxP = UxM - 2 * UnM * nxM
                    UyP = UyM - 2 * UnM * nyM
                    
                    VnM = nxM * VxM +  nyM * VyM
                    VxP = VxM - 2 * VnM * nxM
                    VyP = VyM - 2 * VnM * nyM
                    
                    EnM = nxM * ExM +  nyM * EyM
                    ExP = ExM - 2 * EnM * nxM
                    EyP = EyM - 2 * EnM * nyM

                    unM = nxM * uM +  nyM * vM
                    uP = uM - 2 * unM * nxM
                    vP = vM - 2 * unM * nyM
                    uxP, uyP = UxP, UyP #FXG: Not sure about this BC
                    vxP, vyP = VxP, VyP #FXG: Not sure about this BC
                    #TxP, TyP = ExP, EyP #Produces thermal boundary layer
                    TxP, TyP = TxM, TyM
#                    TxP = 0
#                    TyP = -gravity/c_p
#                    EPflag=0

                    #Tracers
                    for itracer = 1:_ntracers
                        istate = itracer + (_nsd+2)
                        
                        QTnM          = nxM * QTxM[itracer] +  nyM * QTyM[itracer]
                        QTxP[itracer] = QTxM[itracer] - 2 * QTnM * nxM
                        QTyP[itracer] = QTyM[itracer] - 2 * QTnM * nyM
                    end
                    
                    #Store viscosity coefficients
                    μP=μM
                    κP=κM
                    νP=νM
                else
                    error("Invalid boundary conditions $bc on face $f of element $e")
                end

                #Left Fluxes
                div_uM=uxM + vyM           
                fluxρM_x = νM*ρxM
                fluxρM_y = νM*ρyM
                fluxUM_x = μM*(2*uxM + lambda*div_uM)
                fluxUM_y = μM*(uyM + vxM)
                fluxVM_x = μM*(vxM + uyM)
                fluxVM_y = μM*(2*vyM + lambda*div_uM)
                fluxEM_x = μM*(uM*(2*uxM + lambda*div_uM) + vM*(uyM + vxM)) + κM*c_p/Pr*TxM
                fluxEM_y = μM*(uM*(vxM + uyM) + vM*(2*vyM + lambda*div_uM)) + κM*c_p/Pr*TyM

                
                #Right Fluxes
                div_uP=uxP + vyP
                fluxρP_x = νP*ρxP
                fluxρP_y = νP*ρyP
                fluxUP_x = μP*(2*uxP + lambda*div_uP)
                fluxUP_y = μP*(uyP + vxP)
                fluxVP_x = μP*(vxP + uyP)
                fluxVP_y = μP*(2*vyP + lambda*div_uP)
                fluxEP_x = μP*(uP*(2*uxP + lambda*div_uP) + vP*(uyP + vxP))*EPflag + κP*c_p/Pr*TxP
                fluxEP_y = μP*(uP*(vxP + uyP) + vP*(2*vyP + lambda*div_uP))*EPflag + κP*c_p/Pr*TyP

                #Compute Numerical Flux
                fluxρS = 0.5*(nxM * (fluxρM_x + fluxρP_x) + nyM * (fluxρM_y + fluxρP_y))
                fluxUS = 0.5*(nxM * (fluxUM_x + fluxUP_x) + nyM * (fluxUM_y + fluxUP_y))
                fluxVS = 0.5*(nxM * (fluxVM_x + fluxVP_x) + nyM * (fluxVM_y + fluxVP_y))
                fluxES = 0.5*(nxM * (fluxEM_x + fluxEP_x) + nyM * (fluxEM_y + fluxEP_y))

                #Update RHS
                rhs[vidM, _ρ, 1, eM] += sMJ * fluxρS
                rhs[vidM, _U, 1, eM] += sMJ * fluxUS
                rhs[vidM, _V, 1, eM] += sMJ * fluxVS
                rhs[vidM, _E, 1, eM] += sMJ * fluxES

                #Tracers
                for itracer = 1:_ntracers
                    istate = itracer + (_nsd+2)

                    #Left and Right Fluxes
                    fluxQTM_x = κM*QTxM[itracer]
                    fluxQTM_y = κM*QTyM[itracer]
                    
                    fluxQTP_x = κP*QTxP[itracer]
                    fluxQTP_y = κP*QTyP[itracer]
                    
                    #Compute Numerical Flux
                    fluxQTS   = 0.5*(nxM * (fluxQTM_x + fluxQTP_x) + nyM * (fluxQTM_y + fluxQTP_y))

                    #Update RHS
                    rhs[vidM, istate, 1, eM] += sMJ * fluxQTS
                end 

               
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
function updatesolution!(::Val{dim}, ::Val{N}, rhs::Array,  rhs_gradQ, Q, vgeo, elems, rka, rkb, dt) where {dim, N}
    
    Np=(N+1)^dim

    @inbounds for e = elems, s = 1:_nstate, i = 1:Np
        rhs[i, s, e] += rhs_gradQ[i,s,1,e]
        Q[i, s, e] += rkb * dt * rhs[i, s, e] * vgeo[i, _MJI, e]
        rhs[i, s, e] *= rka
    end

end
# }}}

# {{{ Store residual for sgs
function store_residual_sgs!(::Val{dim}, ::Val{N}, rhs_sgs::Array, rhs, vgeo, elems) where {dim, N}

    Np=(N+1)^dim
    fill!( rhs_sgs, zero(rhs_sgs[1]))

    @inbounds for e = elems, s = 1:_nstate, i = 1:Np
        rhs_sgs[i, s, e] += rhs[i, s, e] * vgeo[i, _MJI, e]
    end

end
# }}}

#
# {{{ CPU Kernels
# Saturation adjustment
function sat_adjust!(::Val{2}, ::Val{N}, Q, vgeo, elems) where N
    DFloat          = eltype(Q)
    γ::DFloat       = _γ
    p0::DFloat      = _p0
    R_gas::DFloat   = _R_gas
    c_p::DFloat     = _c_p
    c_v::DFloat     = _c_v
    gravity::DFloat = _gravity

    Nq = N + 1
    nelem = size(Q)[end]
    
    Q      = reshape(Q, Nq, Nq, _nstate, nelem)
    vgeo   = reshape(vgeo, Nq, Nq, _nvgeo, nelem)
    q_tr   = zeros(DFloat, 3)
    Qinout = zeros(DFloat, _nstate)
    q_tr   = zeros(DFloat, _ntracers)
    
    @inbounds for e in elems
        for j = 1:Nq, i = 1:Nq
            y = vgeo[i,j,_y,e]

            #Moist air constant: Rm
            #q_tr[1] = Q[i, j, _qt1, e]
            #q_tr[2] = Q[i, j, _qt2, e]
            #q_tr[3] = Q[i, j, _qt3, e]
            #R_gas = MoistThermodynamics.gas_constant_air(q_tr[1], q_tr[2], q_tr[3])
            R_gas = MoistThermodynamics.gas_constant_air(0.0, 0.0, 0.0)

            U, V = Q[i, j, _U, e], Q[i, j, _V, e]
            ρ, E = Q[i, j, _ρ, e], Q[i, j, _E, e]
            P    = (R_gas/c_v)*(E - (U^2 + V^2)/(2*ρ) - ρ*gravity*y)
            @show(E)
            Qinout[_U]   = _U
            Qinout[_V]   = _U
            Qinout[_ρ]   = _ρ
            Qinout[_E]   = _E
            Qinout[_qt1] = q_tr[1]
            Qinout[_qt2] = q_tr[2]
            Qinout[_qt3] = q_tr[3]
            
            #convert_set3c_to_set2nc_scalar(y, Qinout)
            #=
            θ        = Qinout[_E]
            π_k      = 1.0 - gravity/(c_p*θ)*y
            c        = c_v/R_gas
            ρ        = p0/(R_gas*θ)*(π_k)^c
            P        = p0 * (ρ*R_gas*θ/p0)^(c_p/ c_v)
            T        = π_k*θ
            @show(T)
            qt       = q_tr[1]
            T_trial  = 290.0
            E_int    = MoistThermodynamics.internal_energy_sat.(T, ρ, qt);
            T        = MoistThermodynamics.saturation_adjustment.(E_int, ρ, qt);
            θ        = T/π_k
            ρ        = p0/(R_gas*θ)*(π_k)^c
            
            #Obtain ql, qi from T,  ρ, qt
            ql = zeros(size(T)); qi = zeros(size(T))
            MoistThermodynamics.phase_partitioning_eq!(ql, qi, T, ρ, qt);
            =#
            
        end
    end
end
# }}}

# {{{ Compute viscosity for SGS
function compute_viscosity_sgs(::Val{dim}, ::Val{N},  visc_sgs, rhs_sgs, Q, vgeo, visc, mpicomm) where {dim, N}
    DFloat = eltype(Q)
    γ::DFloat       = _γ
    p0::DFloat      = _p0
    R_gas::DFloat   = _R_gas
    c_p::DFloat     = _c_p
    c_v::DFloat     = _c_v
    gravity::DFloat = _gravity
    Pr::DFloat      = _Prandtl
    lambda::DFloat  = _Stokes
    C1::DFloat      = _C1
    C2::DFloat      = _C2

    (Np, nstate, nelem) = size(Q)

    Q_mean_global = zeros(DFloat, nstate)
    ΔQ_global = zeros(DFloat, nstate)
    rhs_max = zeros(DFloat, nstate)

    mpirank = MPI.Comm_rank(mpicomm)
    mpisize = MPI.Comm_size(mpicomm)
    eps=1e-8
    eps=1e-8
#    eps2=1e-8
#    @show (eps, eps2)

    #Compute Q_mean_global
    @inbounds for e = 1:nelem, s=1:nstate, i = 1:Np
        Q_mean_global[s] += Q[i, s, e]
    end
    for s=1:nstate
        Q_mean_global[s] = Q_mean_global[s]/(nelem*Np)
    end
    Q_mean_global=MPI.allreduce(Q_mean_global, MPI.SUM, mpicomm)/mpisize

    #Compute Infinity/Max norm of (Q-Q_mean_global)
    @inbounds for e = 1:nelem, s=1:nstate, i = 1:Np
        ΔQ_global[s]=max( ΔQ_global[s], abs( Q[i, s, e] - Q_mean_global[s] ) )
    end
    ΔQ_global=MPI.allreduce(ΔQ_global, MPI.MAX, mpicomm)

    #Loop through elements
    @inbounds for e = 1:nelem

        #Initialize arrays
        c_max = -1e6
        ρ_max = -1e6
        ds_min = +1e6
        rhs_max = -1e6*ones(DFloat, nstate)
        q_tr = zeros(DFloat, _ntracers)

        #Loop through Element DOF
        for i = 1:Np
            ρ, U, V, E = Q[i, _ρ, e], Q[i, _U, e], Q[i, _V, e], Q[i, _E, e]
            ξx, ξy, ηx, ηy = vgeo[i, _ξx, e], vgeo[i, _ξy, e], vgeo[i, _ηx, e], vgeo[i, _ηy, e]
            y = vgeo[i, _y, e]

            #Moist air constant: Rm
            #q_tr[1] = 0.0
            #q_tr[2] = 0.0
            #q_tr[3] = 0.0
            #for itracer = 1:_ntracers
            #    istate = itracer + (_nsd+2)
            #    
            #    q_tr[itracer]       = Q[i, istate, e]
            #end
            #R_gas = MoistThermodynamics.gas_constant_air(q_tr[1], q_tr[2], q_tr[3])
            R_gas = MoistThermodynamics.gas_constant_air(0.0, 0.0, 0.0)
            
            P = (R_gas/c_v)*(E - (U^2 + V^2)/(2*ρ) - ρ*gravity*y)

            #Compute Max Wave Speed
            u, v = U/ρ, V/ρ
            vel=sqrt( u^2 + v^2 )
            wave_speed = (vel + sqrt(γ * P / ρ))
            c_max = max( c_max, wave_speed )

            #Compute Element Length
            dx, dy = 1.0/(2*ξx), 1.0/(2*ηy)
            ds=min(dx,dy)
            ds_min=min( ds_min, ds )

            #Compute Max Element Density
            ρ_max = max( ρ_max, abs(ρ) )

            #Compute Max Element RHS
            for s=1:nstate
                rhs_max[s]=max( rhs_max[s], abs( rhs_sgs[i,s,e] ) )
            end
        end

        #Compute μ1
        ds=ds_min

        #=
        ρ_flag=U_flag=V_flag=E_flag=1
        if ( ΔQ_global[_ρ] < eps2 )
        end
        if ( ΔQ_global[_U] < eps2 )
            ΔQ_global[_U]=1.0
        end
        if ( ΔQ_global[_V] < eps2 )
            ΔQ_global[_V]=1.0
        end
        if ( ΔQ_global[_E] < eps2 )
            ΔQ_global[_E]=1.0
        end
        =#

        μ1=C1*ds^2*ΔQ_global[_ρ]*max( rhs_max[_ρ]/(ΔQ_global[_ρ]+eps), rhs_max[_U]/(ΔQ_global[_U]+eps), #Nazarov
                                      rhs_max[_V]/(ΔQ_global[_V]+eps), rhs_max[_E]/(ΔQ_global[_E]+eps) )
#        μ1=C1*ds^2*ΔQ_global[_ρ]*max( rhs_max[_ρ], rhs_max[_U],
#                                      rhs_max[_V], rhs_max[_E] )

        #=
        if ( ΔQ_global[_ρ] < eps2 || ΔQ_global[_U] < eps2 || ΔQ_global[_V] < eps2 || ΔQ_global[_E] < eps2 )
            μ1=C1*ds^2*ΔQ_global[_ρ]*rhs_max[_ρ] #Nazarov
            #            μ1=C1*ds^2*ρ_max*rhs_max[_ρ]/(ΔQ_global[_ρ]+eps)
            #μ1=C1*ds^2*rhs_max[_ρ]/(ΔQ_global[_ρ]+eps)
        else
            #            μ1=C1*ds^2*max( rhs_max[_ρ]/(ΔQ_global[_ρ]+eps), rhs_max[_U]/(ΔQ_global[_U]+eps),
            #                                  rhs_max[_V]/(ΔQ_global[_V]+eps), rhs_max[_E]/(ΔQ_global[_E]+eps) )
            #            μ1=C1*ds^2*ρ_max*max( rhs_max[_ρ]/(ΔQ_global[_ρ]+eps), rhs_max[_U]/(ΔQ_global[_U]+eps),
            μ1=C1*ds^2*ΔQ_global[_ρ]*max( rhs_max[_ρ]/(ΔQ_global[_ρ]+eps), rhs_max[_U]/(ΔQ_global[_U]+eps), #Nazarov
                                          rhs_max[_V]/(ΔQ_global[_V]+eps), rhs_max[_E]/(ΔQ_global[_E]+eps) )
        end
        =#

        #Compute μmax
        μmax=C2*ds*ρ_max*c_max #Nazarov

        #Compute μ, κ, ν
        μ_elem = min(μ1, μmax)
        μ_elem = min(μ_elem, visc)
        
        #Store viscosities
#        μ_elem=visc
        visc_sgs[1,e]=μ_elem #μ
        visc_sgs[2,e]=Pr/(γ-1)*μ_elem #κ
        visc_sgs[3,e]=Pr/ρ_max*μ_elem #ν
        visc_sgs[3,e]=0 #μ
        #@show(ds, C1, C2, ρ_max, c_max, visc_sgs[1,e], visc_sgs[2,e], visc_sgs[3,e])

        #=
        if(rhs_max[_U]/(ΔQ_global[_U]+eps) > 50 || rhs_max[_U]/(ΔQ_global[_V]+eps) > 50)
           #        @show(rhs_max[_ρ], ΔQ_global[_ρ], μ1)
           #        @show(ΔQ_global[_U])
           #        @show(ΔQ_global[_V])
            @show(rhs_max[_E], ΔQ_global[_E], rhs_max[_E]/(ΔQ_global[_E]+eps))
            @show(rhs_max[_U], ΔQ_global[_U], rhs_max[_U]/(ΔQ_global[_U]+eps))
            @show(rhs_max[_V], ΔQ_global[_V], rhs_max[_V]/(ΔQ_global[_V]+eps))
         end
        =#
        
        visc_sgs[1,e]=visc #μ
        visc_sgs[2,e]=visc #μ
        visc_sgs[3,e]=0 #μ
        
    end
#    @show (minimum(visc_sgs[1,:]), maximum(visc_sgs[1,:]))
end
# }}}

# {{{ Compute viscosity for SGS
function compute_viscosity_sgs_working(::Val{dim}, ::Val{N},  visc_sgs, rhs_sgs, Q, vgeo, visc, mpicomm) where {dim, N}
    DFloat = eltype(Q)
    γ::DFloat       = _γ
    p0::DFloat      = _p0
    R_gas::DFloat   = _R_gas
    c_p::DFloat     = _c_p
    c_v::DFloat     = _c_v
    gravity::DFloat = _gravity
    Pr::DFloat      = _Prandtl
    lambda::DFloat  = _Stokes
    C1::DFloat      = _C1
    C2::DFloat      = _C2

    (Np, nstate, nelem) = size(Q)

    Q_mean_global = zeros(DFloat, nstate)
    ΔQ_global = zeros(DFloat, nstate)
    rhs_max = zeros(DFloat, nstate)

    mpirank = MPI.Comm_rank(mpicomm)
    mpisize = MPI.Comm_size(mpicomm)
    eps=1e-8
    eps=1e-1
    eps2=1e-2
#    @show (eps, eps2)

    #Compute Q_mean_global
    @inbounds for e = 1:nelem, s=1:nstate, i = 1:Np
        Q_mean_global[s] += Q[i, s, e]
    end
    for s=1:nstate
        Q_mean_global[s] = Q_mean_global[s]/(nelem*Np)
    end
    Q_mean_global=MPI.allreduce(Q_mean_global, MPI.SUM, mpicomm)/mpisize

    #Compute Infinity/Max norm of (Q-Q_mean_global)
    @inbounds for e = 1:nelem, s=1:nstate, i = 1:Np
        ΔQ_global[s]=max( ΔQ_global[s], abs( Q[i, s, e] - Q_mean_global[s] ) )
    end
    ΔQ_global=MPI.allreduce(ΔQ_global, MPI.MAX, mpicomm)

    #Loop through elements
    @inbounds for e = 1:nelem

        #Initialize arrays
        c_max = -1e6
        ρ_max = -1e6
        ds_min = +1e6
        rhs_max = -1e6*ones(DFloat, nstate)

        #Loop through Element DOF
        for i = 1:Np
            ρ, U, V, E = Q[i, _ρ, e], Q[i, _U, e], Q[i, _V, e], Q[i, _E, e]
            ξx, ξy, ηx, ηy = vgeo[i, _ξx, e], vgeo[i, _ξy, e], vgeo[i, _ηx, e], vgeo[i, _ηy, e]
            y = vgeo[i, _y, e]
            P = (R_gas/c_v)*(E - (U^2 + V^2)/(2*ρ) - ρ*gravity*y)

            #Compute Max Wave Speed
            u, v = U/ρ, V/ρ
            vel=sqrt( u^2 + v^2 )
            wave_speed = (vel + sqrt(γ * P / ρ))
            c_max = max( c_max, wave_speed )

            #Compute Element Length
            dx, dy = 1.0/(2*ξx), 1.0/(2*ηy)
            ds=min(dx,dy)
            ds_min=min( ds_min, ds )

            #Compute Max Element Density
            ρ_max = max( ρ_max, abs(ρ) )

            #Compute Max Element RHS
            for s=1:nstate
                rhs_max[s]=max( rhs_max[s], abs( rhs_sgs[i,s,e] ) )
            end
        end

        #Compute μ1
        ds=ds_min

        if ( ΔQ_global[_ρ] < eps2 || ΔQ_global[_U] < eps2 || ΔQ_global[_V] < eps2 || ΔQ_global[_E] < eps2 )
            μ1=C1*ds^2*ΔQ_global[_ρ]*rhs_max[_ρ] #Nazarov
            #            μ1=C1*ds^2*ρ_max*rhs_max[_ρ]/(ΔQ_global[_ρ]+eps)
            #μ1=C1*ds^2*rhs_max[_ρ]/(ΔQ_global[_ρ]+eps)
        else
#            μ1=C1*ds^2*max( rhs_max[_ρ]/(ΔQ_global[_ρ]+eps), rhs_max[_U]/(ΔQ_global[_U]+eps),
#                                  rhs_max[_V]/(ΔQ_global[_V]+eps), rhs_max[_E]/(ΔQ_global[_E]+eps) )
                                  #            μ1=C1*ds^2*ρ_max*max( rhs_max[_ρ]/(ΔQ_global[_ρ]+eps), rhs_max[_U]/(ΔQ_global[_U]+eps),
            μ1=C1*ds^2*ΔQ_global[_ρ]*max( rhs_max[_ρ]/(ΔQ_global[_ρ]+eps), rhs_max[_U]/(ΔQ_global[_U]+eps), #Nazarov
                                          rhs_max[_V]/(ΔQ_global[_V]+eps), rhs_max[_E]/(ΔQ_global[_E]+eps) )
        end
#        μ1=C1*ds^2*ρ_max*rhs_max[_ρ]/(ΔQ_global[_ρ]+eps)

        #Compute μmax
        μmax=C2*ds*ρ_max*c_max #Nazarov
#        μmax=C2*ds*c_max

        #Compute μ, κ, ν
        μ_elem=min(μ1, μmax)

        #Store viscosities
#        μ_elem=visc
        visc_sgs[1,e]=μ_elem #μ
        visc_sgs[2,e]=Pr/(γ-1)*μ_elem #κ
        visc_sgs[3,e]=Pr/ρ_max*μ_elem #ν
        visc_sgs[3,e]=0 #μ

    end
#    @show (minimum(visc_sgs[1,:]), maximum(visc_sgs[1,:]))
end
# }}}

# {{{ improved GPU kernles
# {{{ Volume RHS
@hascuda function knl_volume_rhs!(::Val{dim}, ::Val{N}, rhs, Q, vgeo, D, nelem) where {dim, N}
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
        y = vgeo[i, j, _y, e]

        U, V = Q[i, j, _U, e], Q[i, j, _V, e]
        ρ, E = Q[i, j, _ρ, e], Q[i, j, _E, e]
        P = (R_gas/c_v)*(E - (U*U + V*V)/(2*ρ) - ρ*gravity*y)
        rhsU, rhsV = rhs[i, j, _U, e], rhs[i, j, _V, e]
        rhsρ, rhsE = rhs[i, j, _ρ, e], rhs[i, j, _E, e]

        ρinv = 1 / ρ
        fluxρ_x = U
        fluxU_x = ρinv * U * U + P
        fluxV_x = ρinv * V * U
        fluxE_x = ρinv * U * (E+P)

        fluxρ_y = V
        fluxU_y = ρinv * U * V
        fluxV_y = ρinv * V * V + P
        fluxE_y = ρinv * V * (E+P)

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

# {{{ Flux RHS (all dimensions)
@hascuda function knl_flux_rhs!(::Val{dim}, ::Val{N}, rhs, Q, sgeo, vgeo, nelem, vmapM,vmapP, elemtobndy) where {dim, N}
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

                #Left variables
                ρM = Q[vidM, _ρ, eM]
                UM = Q[vidM, _U, eM]
                VM = Q[vidM, _V, eM]
                EM = Q[vidM, _E, eM]
                yM = vgeo[vidM, _y, eM]
                PM = (R_gas/c_v)*(EM - (UM*UM + VM*VM)/(2*ρM) - ρM*gravity*yM)

                #Right variables
                bc = elemtobndy[f, e]
                ρP = UP = VP = EP = PP = zero(eltype(Q))
                if bc == 0
                    ρP = Q[vidP, _ρ, eP]
                    UP = Q[vidP, _U, eP]
                    VP = Q[vidP, _V, eP]
                    EP = Q[vidP, _E, eP]
                    yP = vgeo[vidP, _y, eP]
                    PP = (R_gas/c_v)*(EP - (UP*UP + VP*VP)/(2*ρP) - ρP*gravity*yP)
                elseif bc == 1
                    UnM = nxM * UM + nyM * VM
                    UP = UM - 2 * UnM * nxM
                    VP = VM - 2 * UnM * nyM
                    ρP = ρM
                    EP = EM
                    PP = PM
                end

                #Left fluxes
                ρMinv = 1 / ρM
                fluxρM_x = UM
                fluxUM_x = ρMinv * UM * UM + PM
                fluxVM_x = ρMinv * VM * UM
                fluxEM_x = ρMinv * UM * (EM+PM)

                fluxρM_y = VM
                fluxUM_y = ρMinv * UM * VM
                fluxVM_y = ρMinv * VM * VM + PM
                fluxEM_y = ρMinv * VM * (EM+PM)

                #Right fluxes
                ρPinv = 1 / ρP
                fluxρP_x = UP
                fluxUP_x = ρPinv * UP * UP + PP
                fluxVP_x = ρPinv * VP * UP
                fluxEP_x = ρPinv * UP * (EP+PP)

                fluxρP_y = VP
                fluxUP_y = ρPinv * UP * VP
                fluxVP_y = ρPinv * VP * VP + PP
                fluxEP_y = ρPinv * VP * (EP+PP)

                #Compute Wave Speed
                λM = ρMinv * abs(nxM * UM + nyM * VM) + CUDAnative.sqrt(ρMinv * γ * PM)
                λP = ρPinv * abs(nxM * UP + nyM * VP) + CUDAnative.sqrt(ρPinv * γ * PP)
                λ  =  max(λM, λP)

                #Compute Numerical Flux
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

# {{{ Volume grad(Q)
@hascuda function knl_volume_grad!(::Val{dim}, ::Val{N}, rhs, Q, vgeo, D, nelem) where {dim, N}
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
    s_F = @cuStaticSharedMem(eltype(Q), (Nq, Nq, _nstate, dim))
    s_G = @cuStaticSharedMem(eltype(Q), (Nq, Nq, _nstate, dim))

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
        y = vgeo[i, j, _y, e]

        U, V = Q[i, j, _U, e], Q[i, j, _V, e]
        ρ, E = Q[i, j, _ρ, e], Q[i, j, _E, e]
        P = (R_gas/c_v)*(E - (U*U + V*V)/(2*ρ) - ρ*gravity*y)

        #Prefetch
        rhsρ1, rhsρ2 = rhs[i, j, _ρ, 1, e], rhs[i, j, _ρ, 2, e]
        rhsU1, rhsU2 = rhs[i, j, _U, 1, e], rhs[i, j, _U, 2, e]
        rhsV1, rhsV2 = rhs[i, j, _V, 1, e], rhs[i, j, _V, 2, e]
        rhsE1, rhsE2 = rhs[i, j, _E, 1, e], rhs[i, j, _E, 2, e]

        #Primitive variables
        u=U/ρ
        v=V/ρ
        T=P/(R_gas*ρ)

        #Compute fluxes
        fluxρ = ρ
        fluxU = u
        fluxV = v
        fluxE = T

        s_F[i, j, _ρ, 1], s_F[i, j, _ρ, 2] = MJ * (ξx * fluxρ), MJ * (ξy * fluxρ)
        s_F[i, j, _U, 1], s_F[i, j, _U, 2] = MJ * (ξx * fluxU), MJ * (ξy * fluxU)
        s_F[i, j, _V, 1], s_F[i, j, _V, 2] = MJ * (ξx * fluxV), MJ * (ξy * fluxV)
        s_F[i, j, _E, 1], s_F[i, j, _E, 2] = MJ * (ξx * fluxE), MJ * (ξy * fluxE)

        s_G[i, j, _ρ, 1], s_G[i, j, _ρ, 2] = MJ * (ηx * fluxρ), MJ * (ηy * fluxρ)
        s_G[i, j, _U, 1], s_G[i, j, _U, 2] = MJ * (ηx * fluxU), MJ * (ηy * fluxU)
        s_G[i, j, _V, 1], s_G[i, j, _V, 2] = MJ * (ηx * fluxV), MJ * (ηy * fluxV)
        s_G[i, j, _E, 1], s_G[i, j, _E, 2] = MJ * (ηx * fluxE), MJ * (ηy * fluxE)

    end

    sync_threads()

    @inbounds if i <= Nq && j <= Nq && k == 1 && e <= nelem
        # loop of ξ-grid lines
        for n = 1:Nq
            Dni = s_D[n, i]
            Dnj = s_D[n, j]

            rhsρ1 += Dni * s_F[n, j, 1, _ρ]
            rhsρ1 += Dnj * s_G[i, n, 1, _ρ]
            rhsρ2 += Dni * s_F[n, j, 2, _ρ]
            rhsρ2 += Dnj * s_G[i, n, 2, _ρ]

            rhsU1 += Dni * s_F[n, j, 1, _U]
            rhsU1 += Dnj * s_G[i, n, 1, _U]
            rhsU2 += Dni * s_F[n, j, 2, _U]
            rhsU2 += Dnj * s_G[i, n, 2, _U]

            rhsV1 += Dni * s_F[n, j, 1, _V]
            rhsV1 += Dnj * s_G[i, n, 1, _V]
            rhsV2 += Dni * s_F[n, j, 2, _V]
            rhsV2 += Dnj * s_G[i, n, 2, _V]

            rhsE1 += Dni * s_F[n, j, 1, _E]
            rhsE1 += Dnj * s_G[i, n, 1, _E]
            rhsE2 += Dni * s_F[n, j, 2, _E]
            rhsE2 += Dnj * s_G[i, n, 2, _E]
        end

        rhs[i, j, _ρ, 1, e], rhs[i, j, _ρ, 2, e] = rhsρ1, rhsρ2
        rhs[i, j, _U, 1, e], rhs[i, j, _U, 2, e] = rhsU1, rhsU2
        rhs[i, j, _V, 1, e], rhs[i, j, _V, 2, e] = rhsV1, rhsV2
        rhs[i, j, _E, 1, e], rhs[i, j, _E, 2, e] = rhsE1, rhsE2
    end
    nothing
end
# }}}

# {{{ Flux grad(Q)
@hascuda function knl_flux_grad!(::Val{dim}, ::Val{N}, rhs, Q, sgeo, vgeo, nelem, vmapM, vmapP, elemtobndy) where {dim, N}
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

                #Left variables
                ρM = Q[vidM, _ρ, eM]
                UM = Q[vidM, _U, eM]
                VM = Q[vidM, _V, eM]
                EM = Q[vidM, _E, eM]
                yM = vgeo[vidM, _y, eM]
                PM = (R_gas/c_v)*(EM - (UM*UM + VM*VM)/(2*ρM) - ρM*gravity*yM)
                uM=UM/ρM
                vM=VM/ρM
                TM=PM/(R_gas*ρM)

                #Right variables
                bc = elemtobndy[f, e]
                ρP = UP = VP = EP = PP = zero(eltype(Q))
                if bc == 0
                    ρP = Q[vidP, _ρ, eP]
                    UP = Q[vidP, _U, eP]
                    VP = Q[vidP, _V, eP]
                    EP = Q[vidP, _E, eP]
                    yP = vgeo[vidP, _y, eP]
                    PP = (R_gas/c_v)*(EP - (UP*UP + VP*VP)/(2*ρP) - ρP*gravity*yP)
                    uP=UP/ρP
                    vP=VP/ρP
                    TP=PP/(R_gas*ρP)
                elseif bc == 1
                    UnM = nxM * UM + nyM * VM
                    UP = UM - 2 * UnM * nxM
                    VP = VM - 2 * UnM * nyM
                    ρP = ρM
                    EP = EM
                    PP = PM
                    uP = UP/ρP
                    vP = VP/ρP
                    TP = TM
                end

                #Left Fluxes
                fluxρM = ρM
                fluxUM = uM
                fluxVM = vM
                fluxEM = TM

                #Right Fluxes
                fluxρP = ρP
                fluxUP = uP
                fluxVP = vP
                fluxEP = TP

                #Compute Numerical Flux
                fluxρS = 0.5*(fluxρM + fluxρP)
                fluxUS = 0.5*(fluxUM + fluxUP)
                fluxVS = 0.5*(fluxVM + fluxVP)
                fluxES = 0.5*(fluxEM + fluxEP)

                #Update RHS
                rhs[vidM, _ρ, 1, eM] += sMJ * nxM*fluxρS
                rhs[vidM, _ρ, 2, eM] += sMJ * nyM*fluxρS
                rhs[vidM, _U, 1, eM] += sMJ * nxM*fluxUS
                rhs[vidM, _U, 2, eM] += sMJ * nyM*fluxUS
                rhs[vidM, _V, 1, eM] += sMJ * nxM*fluxVS
                rhs[vidM, _V, 2, eM] += sMJ * nyM*fluxVS
                rhs[vidM, _E, 1, eM] += sMJ * nxM*fluxES
                rhs[vidM, _E, 2, eM] += sMJ * nyM*fluxES
            end
            sync_threads()
        end
    end
    nothing
end

# {{{ Volume div(grad(Q))
@hascuda function knl_volume_div!(::Val{dim}, ::Val{N}, rhs, gradQ, Q, vgeo, D, nelem) where {dim, N}
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
        ρx, ρy = gradQ[i, j, _ρ, 1, e], gradQ[i, j, _ρ, 2, e]
        Ux, Uy = gradQ[i, j, _U, 1, e], gradQ[i, j, _U, 2, e]
        Vx, Vy = gradQ[i, j, _V, 1, e], gradQ[i, j, _V, 2, e]
        Ex, Ey = gradQ[i, j, _E, 1, e], gradQ[i, j, _E, 2, e]
        ρ, U, V = Q[i, j, _ρ, e], Q[i, j, _U, e], Q[i, j, _V, e]
        rhsU, rhsV = rhs[i, j, _U, 1, e], rhs[i, j, _V, 1, e]
        rhsρ, rhsE = rhs[i, j, _ρ, 1, e], rhs[i, j, _E, 1, e]

        #Compute primitive variables
        ux, uy = Ux, Uy
        vx, vy = Vx, Vy
        Tx, Ty = Ex, Ey
        div_u=ux + vy
        u=U/ρ
        v=V/ρ

        #Compute fluxes
        fluxρ_x = 0*ρx
        fluxρ_y = 0*ρy
        fluxU_x = 2*ux + lambda*div_u
        fluxU_y = uy + vx
        fluxV_x = vx + uy
        fluxV_y = 2*vy + lambda*div_u
        fluxE_x = u*(2*ux + lambda*div_u) + v*(uy + vx) + c_p/Pr*Tx
        fluxE_y = u*(vx + uy) + v*(2*vy + lambda*div_u) + c_p/Pr*Ty

        s_F[i, j, _ρ] = MJ * (ξx * fluxρ_x + ξy * fluxρ_y)
        s_F[i, j, _U] = MJ * (ξx * fluxU_x + ξy * fluxU_y)
        s_F[i, j, _V] = MJ * (ξx * fluxV_x + ξy * fluxV_y)
        s_F[i, j, _E] = MJ * (ξx * fluxE_x + ξy * fluxE_y)

        s_G[i, j, _ρ] = MJ * (ηx * fluxρ_x + ηy * fluxρ_y)
        s_G[i, j, _U] = MJ * (ηx * fluxU_x + ηy * fluxU_y)
        s_G[i, j, _V] = MJ * (ηx * fluxV_x + ηy * fluxV_y)
        s_G[i, j, _E] = MJ * (ηx * fluxE_x + ηy * fluxE_y)
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

        rhs[i, j, _U, 1, e] = rhsU
        rhs[i, j, _V, 1, e] = rhsV
        rhs[i, j, _ρ, 1, e] = rhsρ
        rhs[i, j, _E, 1, e] = rhsE
    end
    nothing
end
# }}}

# {{{ Flux grad(Q)
@hascuda function knl_flux_grad!(::Val{dim}, ::Val{N}, rhs, gradQ, Q, sgeo, nelem, vmapM, vmapP, elemtobndy) where {dim, N}
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

                #Left variables
                ρxM = gradQ[vidM, _ρ, 1, eM]
                ρyM = gradQ[vidM, _ρ, 2, eM]
                UxM = gradQ[vidM, _U, 1, eM]
                UyM = gradQ[vidM, _U, 2, eM]
                VxM = gradQ[vidM, _V, 1, eM]
                VyM = gradQ[vidM, _V, 2, eM]
                ExM = gradQ[vidM, _E, 1, eM]
                EyM = gradQ[vidM, _E, 2, eM]
                ρM = Q[vidM, _ρ, eM]
                UM = Q[vidM, _U, eM]
                VM = Q[vidM, _V, eM]

                uM=UM/ρM
                vM=VM/ρM
                uxM, uyM = UxM, UyM
                vxM, vyM = VxM, VyM
                TxM, TyM = ExM, EyM

                #Right variables
                bc = elemtobndy[f, e]
                ρxP = ρyP = UxP = UyP = VxP = VyP = ExP = EyP = zero(eltype(Q))
                if bc == 0
                    ρxP = gradQ[vidP, _ρ, 1, eP]
                    ρyP = gradQ[vidP, _ρ, 2, eP]
                    UxP = gradQ[vidP, _U, 1, eP]
                    UyP = gradQ[vidP, _U, 2, eP]
                    VxP = gradQ[vidP, _V, 1, eP]
                    VyP = gradQ[vidP, _V, 2, eP]
                    ExP = gradQ[vidP, _E, 1, eP]
                    EyP = gradQ[vidP, _E, 2, eP]

                    ρP = Q[vidP, _ρ, eP]
                    UP = Q[vidP, _U, eP]
                    VP = Q[vidP, _V, eP]
                    uP=UP/ρP
                    vP=VP/ρP
                    uxP, uyP = UxP, UyP
                    vxP, vyP = VxP, VyP
                    TxP, TyP = ExP, EyP
                elseif bc == 1
                    ρnM = nxM * ρxM +  nyM * ρyM
                    ρxP = ρxM - 2 * ρnM * nxM
                    ρyP = ρyM - 2 * ρnM * nyM
                    UnM = nxM * UxM +  nyM * UyM
                    UxP = UxM - 2 * UnM * nxM
                    UyP = UyM - 2 * UnM * nyM
                    VnM = nxM * VxM +  nyM * VyM
                    VxP = VxM - 2 * VnM * nxM
                    VyP = VyM - 2 * VnM * nyM
                    EnM = nxM * ExM +  nyM * EyM
                    ExP = ExM - 2 * EnM * nxM
                    EyP = EyM - 2 * EnM * nyM

                    unM = nxM * uM +  nyM * vM
                    uP = uM - 2 * unM * nxM
                    vP = vM - 2 * unM * nyM
                    uxP, uyP = UxP, UyP #FXG: Not sure about this BC
                    vxP, vyP = VxP, VyP #FXG: Not sure about this BC
                    #TxP, TyP = ExP, EyP #Produces thermal boundary layer
                    TxP, TyP = TxM, TyM
                end

                div_uM=uxM + vyM
                #Left Fluxes
                fluxρM_x = 0*ρxM
                fluxρM_y = 0*ρyM
                fluxUM_x = 2*uxM + lambda*div_uM
                fluxUM_y = uyM + vxM
                fluxVM_x = vxM + uyM
                fluxVM_y = 2*vyM + lambda*div_uM
                fluxEM_x = uM*(2*uxM + lambda*div_uM) + vM*(uyM + vxM) + c_p/Pr*TxM
                fluxEM_y = uM*(vxM + uyM) + vM*(2*vyM + lambda*div_uM) + c_p/Pr*TyM

                div_uP=uxP + vyP
                #Right Fluxes
                fluxρP_x = 0*ρxP
                fluxρP_y = 0*ρyP
                fluxUP_x = 2*uxP + lambda*div_uP
                fluxUP_y = uyP + vxP
                fluxVP_x = vxP + uyP
                fluxVP_y = 2*vyP + lambda*div_uP
                fluxEP_x = uP*(2*uxP + lambda*div_uP) + vP*(uyP + vxP) + c_p/Pr*TxP
                fluxEP_y = uP*(vxP + uyP) + vP*(2*vyP + lambda*div_uP) + c_p/Pr*TyP

                #Compute Numerical Flux
                fluxρS = 0.5*(nxM * (fluxρM_x + fluxρP_x) + nyM * (fluxρM_y + fluxρP_y))
                fluxUS = 0.5*(nxM * (fluxUM_x + fluxUP_x) + nyM * (fluxUM_y + fluxUP_y))
                fluxVS = 0.5*(nxM * (fluxVM_x + fluxVP_x) + nyM * (fluxVM_y + fluxVP_y))
                fluxES = 0.5*(nxM * (fluxEM_x + fluxEP_x) + nyM * (fluxEM_y + fluxEP_y))

                #Update RHS
                rhs[vidM, _ρ, 1, eM] += sMJ * fluxρS
                rhs[vidM, _U, 1, eM] += sMJ * fluxUS
                rhs[vidM, _V, 1, eM] += sMJ * fluxVS
                rhs[vidM, _E, 1, eM] += sMJ * fluxES
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
@hascuda function volume_rhs!(::Val{dim}, ::Val{N}, d_rhsL::CuArray, d_QL,
                              d_vgeoL, d_D, elems) where {dim, N}
    #Constants
    Nq = N+1
    nelem = length(elems)

    #Reshape arrays
    d_rhsC = reshape(d_rhsL, Nq, Nq, _nstate, nelem)
    d_QC = reshape(d_QL, Nq, Nq, _nstate, nelem)
    d_vgeoC = reshape(d_vgeoL, Nq, Nq, _vgeo, nelem)

    @cuda(threads=ntuple(j->N+1, dim), blocks=nelem,
          knl_volume_rhs!(Val(dim), Val(N), d_rhsC, d_QC, d_vgeoC, d_D, nelem))
end

@hascuda function flux_rhs!(::Val{dim}, ::Val{N}, d_rhsL::CuArray, d_QL, d_sgeo,
                           d_vgeoL, elems, d_vmapM, d_vmapP, d_elemtobndy) where {dim, N}
    nelem = length(elems)
    @cuda(threads=(ntuple(j->N+1, dim-1)..., 1), blocks=nelem,
          knl_flux_rhs!(Val(dim), Val(N), d_rhsL, d_QL, d_sgeo, d_vgeoL, nelem, d_vmapM, d_vmapP, d_elemtobndy))
end

@hascuda function volume_grad!(::Val{dim}, ::Val{N}, d_rhs_gradQL::CuArray, d_QL, d_vgeoL, d_D, elems) where {dim, N}

    #Constants
    Nq = N+1
    nelem = length(elems)

    #Reshape arrays
    d_rhs_gradQC = reshape(d_rhs_gradQL, Nq, Nq, _nstate, dim, nelem)
    d_QC = reshape(d_QL, Nq, Nq, _nstate, nelem)
    d_vgeoC = reshape(d_vgeoL, Nq, Nq, _vgeo, nelem)

    @cuda(threads=ntuple(j->N+1, dim), blocks=nelem,
          knl_volume_grad!(Val(dim), Val(N), d_rhsC, d_QC, d_vgeoC, d_D, nelem))
end

@hascuda function flux_grad!(::Val{dim}, ::Val{N}, d_rhsL::CuArray, d_QL, d_sgeo,
                           d_vgeoL, elems, d_vmapM, d_vmapP, d_elemtobndy) where {dim, N}
    nelem = length(elems)
    @cuda(threads=(ntuple(j->N+1, dim-1)..., 1), blocks=nelem,
          knl_flux_grad!(Val(dim), Val(N), d_rhsL, d_QL, d_sgeo, d_vgeoL, nelem, d_vmapM, d_vmapP, d_elemtobndy))
end

@hascuda function volume_div!(::Val{dim}, ::Val{N}, d_rhs_gradQL::CuArray, d_gradQL, QL, d_vgeoL, d_D, elems) where {dim, N}

    #Constants
    Nq = N+1
    nelem = length(elems)

    #Reshape arrays
    d_rhs_gradQC = reshape(d_rhs_gradQL, Nq, Nq, _nstate, dim, nelem)
    d_gradQC = reshape(d_gradQL, Nq, Nq, _nstate, dim, nelem)
    QC = reshape(QL, Nq, Nq, _nstate, nelem)
    d_vgeoC = reshape(d_vgeoL, Nq, Nq, _vgeo, nelem)

    @cuda(threads=ntuple(j->N+1, dim), blocks=nelem,
          knl_volume_div!(Val(dim), Val(N), d_rhs_gradQC, d_gradQC, QC, d_vgeoC, d_D, nelem))
end

@hascuda function flux_div!(::Val{dim}, ::Val{N}, d_rhsL::CuArray, d_gradQL, d_QL, d_sgeo, elems, d_vmapM, d_vmapP, d_elemtobndy) where {dim, N}

    #Constants
    nelem = length(elems)

    @cuda(threads=(ntuple(j->N+1, dim-1)..., 1), blocks=nelem,
          knl_flux_div!(Val(dim), Val(N), d_rhsL, d_gradQL, d_QL, d_sgeo, nelem, d_vmapM, d_vmapP, d_elemtobndy))
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
    γ::DFloat       = _γ
    p0::DFloat      = _p0
    R_gas::DFloat   = _R_gas
    c_p::DFloat     = _c_p
    c_v::DFloat     = _c_v
    gravity::DFloat = _gravity
    
    mpirank = MPI.Comm_rank(mpicomm)
    #@show(_nstate, _ntracers, size(Q), size(rhs))
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

    #Create Device SGS Arrays
    d_rhs_sgs = zeros(DFloat, (N+1)^dim, _nstate, nelem)
    d_visc_sgs = zeros(DFloat, 3, nelem)
    visc_sgsL = zeros(DFloat, (N+1)^dim, 3, nelem)
    
    #Start Time Loop
    start_time = t1 = time_ns()
    for step = 1:nsteps
        for s = 1:length(RKA)

            #---------------1st Order Operators--------------------------#
            # Send Data Q
            senddata_Q(Val(dim), Val(N), mesh, sendreq, recvreq, sendQ,
                       recvQ, d_sendelems, d_sendQ, d_recvQ, d_QL, mpicomm;
                       ArrType=ArrType)

            # volume RHS computation
            volume_rhs!(Val(dim), Val(N), d_rhsL, d_QL, d_vgeoL, d_D, mesh.realelems)

            # Receive Data Q
            receivedata_Q!(Val(dim), Val(N), mesh, recvreq, recvQ, d_recvQ, d_QL)

            # flux RHS computation
            flux_rhs!(Val(dim), Val(N), d_rhsL, d_QL, d_sgeo, d_vgeoL, mesh.realelems, d_vmapM, d_vmapP, d_elemtobndy)

            #---------------2nd Order Operators--------------------------#
            if (visc > 0)

                # Store gridpoint residual
                store_residual_sgs!(Val(dim), Val(N), d_rhs_sgs, d_rhsL, d_vgeoL, mesh.realelems)

                # Compute viscosity coefficient for SGS model
                #                compute_viscosity_sgs_working(Val(dim), Val(N), d_visc_sgs, d_rhs_sgs, d_QL, d_vgeoL, visc, mpicomm)
                compute_viscosity_sgs(Val(dim), Val(N), d_visc_sgs, d_rhs_sgs, d_QL, d_vgeoL, visc, mpicomm)

                # volume grad Q computation
                volume_grad!(Val(dim), Val(N), d_rhs_gradQL, d_QL, d_vgeoL, d_D, mesh.realelems)

                # flux grad Q computation
                flux_grad!(Val(dim), Val(N), d_rhs_gradQL, d_QL, d_sgeo, d_vgeoL, mesh.realelems, d_vmapM, d_vmapP, d_elemtobndy)

                # Construct grad Q
                update_gradQ!(Val(dim), Val(N), d_gradQL, d_rhs_gradQL, d_vgeoL, mesh.realelems)

                # Send Data grad(Q)
                senddata_gradQ(Val(dim), Val(N), mesh, sendreq, recvreq, sendgradQ,
                               recvgradQ, d_sendelems, d_sendgradQ, d_recvgradQ,
                               d_gradQL, mpicomm;ArrType=ArrType)

                # volume div(grad Q) computation
                volume_div!(Val(dim), Val(N), d_rhs_gradQL, d_gradQL, d_QL, d_visc_sgs, d_vgeoL, d_D, mesh.realelems)

                # Receive Data grad(Q)
                receivedata_gradQ!(Val(dim), Val(N), mesh, recvreq, recvgradQ, d_recvgradQ, d_gradQL)

                # flux div(grad Q) computation
                flux_div!(Val(dim), Val(N), d_rhs_gradQL, d_gradQL, d_QL, d_visc_sgs, d_sgeo, mesh.realelems, d_vmapM, d_vmapP, d_elemtobndy)
            end

            #
            # --------------- Saturation adjustment -------------------------- #
            #
            #sat_adjust!(Val(dim), Val(N), d_QL, d_vgeoL, mesh.realelems)
            #
            # End saturation adjustment
            #
            
            
            #---------------Update Solution--------------------------#
            # update solution and scale RHS
            updatesolution!(Val(dim), Val(N), d_rhsL, d_rhs_gradQL, d_QL, d_vgeoL, mesh.realelems,
                            RKA[s%length(RKA)+1], RKB[s], dt)
        end #end time steopping

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
            for e=1:nelem, s=1:3, n=1:(N+1)^dim
                visc_sgsL[n,s,e] = d_visc_sgs[s,e]
            end
            convert_set3c_to_set2nc(Val(dim), Val(N), vgeo, Q)
            X = ntuple(j->reshape((@view vgeo[:, _x+j-1, :]), ntuple(j->N+1,dim)...,
                                  nelem), dim)
            ρ = reshape((@view Q[:, _ρ, :]), ntuple(j->(N+1),dim)..., nelem)
            U = reshape((@view Q[:, _U, :]), ntuple(j->(N+1),dim)..., nelem)
            V = reshape((@view Q[:, _V, :]), ntuple(j->(N+1),dim)..., nelem)
            E = reshape((@view Q[:, _E, :]), ntuple(j->(N+1),dim)..., nelem)
            E = E .- 300.0

            #Write dynamics to file:
            #MU1 = reshape((@view visc_sgsL[:, 1, :]), ntuple(j->(N+1),dim)..., nelem)
            #MU2 = reshape((@view visc_sgsL[:, 2, :]), ntuple(j->(N+1),dim)..., nelem)
            #MU3 = reshape((@view visc_sgsL[:, 3, :]), ntuple(j->(N+1),dim)..., nelem)
            writemesh(@sprintf("viz/nse2d_sgs_%s_rank_%04d_step_%05d",
                               ArrType, mpirank, step), X...;
                      fields=(("Rho", ρ), ("U", U), ("V", V), ("E", E)),
                      realelems=mesh.realelems)
            
            @inbounds for itracer = 1:_ntracers
                #
                # WARNING: correct the call to writemesh to only write Qtracers
                #          to the file, without re-writing the dynamics again
                #
                istate = itracer + (_nsd+2)
                Qtracers = reshape((@view Q[:, istate, :]), ntuple(j->(N+1),dim)..., nelem)
                writemesh(@sprintf("viz/TRACER_%04d_nse2d_sgs_%s_rank_%04d_step_%05d",
                                   itracer,
                                   ArrType, mpirank, step), X...;
                          fields=(("Rho", ρ), ("U", U), ("V", V), ("E", E), ("QT", Qtracers)),
                      realelems=mesh.realelems)
            end
           
            @show (step*dt, minimum(E), maximum(E))
        end
    end
    if (mpirank == 0)
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
        ρ, u, v, E   = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e], Q[n, _E, e]
        Q[n, _U, e]  = ρ*u
        Q[n, _V, e]  = ρ*v
        Q[n, _E, e]  = ρ*E
        
    end
end
# }}}

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
    q_tr = zeros(DFloat, _ntracers)

    @inbounds for e = 1:nelem, n = 1:Np
        ρ, u, v, E = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e], Q[n, _E, e]
        y = vgeo[n, _y, e]
        
        #Moist air constant: Rm
#=        q_tr[1] = 0.0
        q_tr[2] = 0.0
        q_tr[3] = 0.0
        for itracer = 1:_ntracers
            istate = itracer + (_nsd+2)
            
            q_tr[itracer] = Q[n, istate, e]
        end
        R_gas = MoistThermodynamics.gas_constant_air(q_tr[1], q_tr[2], q_tr[3])=#
        R_gas = MoistThermodynamics.gas_constant_air(0.0, 0.0, 0.0)
        
        P = p0 * (ρ * R_gas * E / p0)^(c_p / c_v)
        T = P/(ρ*R_gas)
        E = c_v*T + 0.5*(u^2 + v^2) + gravity*y
        Q[n, _U, e] = ρ*u
        Q[n, _V, e] = ρ*v
        Q[n, _E, e] = ρ*E
    end
end
# }}}

function convert_set2nc_to_set3c_scalar(x_ndim, Q)
    DFloat          = eltype(Q)
    γ::DFloat       = _γ
    p0::DFloat      = _p0
    R_gas::DFloat   = _R_gas
    c_p::DFloat     = _c_p
    c_v::DFloat     = _c_v
    gravity::DFloat = _gravity

    q_tr = zeros(DFloat, _ntracers)
    
    ρ, u, v, E = Q[_ρ], Q[_U], Q[_V], Q[_E]
    
    # Moist air constant: Rm
    #= Get q from q*ρ
    q_tr[1] = 0.0
    q_tr[2] = 0.0
    q_tr[3] = 0.0
    for itracer = 1:_ntracers
        istate = itracer + (_nsd+2)
        
        q_tr[itracer] = Q[istate]
        Q[istate]     = q_tr[itracer] /ρ
        
    end
    R_gas = MoistThermodynamics.gas_constant_air(q_tr[1], q_tr[2], q_tr[3])=#
    R_gas = MoistThermodynamics.gas_constant_air(0.0, 0.0, 0.0)
    
    P = p0 * (ρ * R_gas * E / p0)^(c_p / c_v)
    T = P/(ρ*R_gas)

    E = c_v*T + 0.5*(u^2 + v^2) + gravity*x_ndim
    
    Q[_U] = ρ*u
    Q[_V] = ρ*v
    Q[_E] = ρ*E
    
    return Q
    
end
# }}}

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
    q_tr          = zeros(DFloat, 3)

    q_tr = zeros(DFloat, 3)
    @inbounds for e = 1:nelem, n = 1:Np
        ρ, U, V, E = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e], Q[n, _E, e]
        y = vgeo[n, _y, e]

        # Moist air constant: Rm
        #Get q from q*ρ
#=        q_tr[1] = 0.0
        q_tr[2] = 0.0
        q_tr[3] = 0.0
        for itracer = 1:_ntracers
            istate = itracer + (_nsd+2)
            
            q_tr[itracer]   = Q[n, istate, e]
            #Q[n, istate, e] = q_tr[itracer] /ρ
            
        end
        R_gas = MoistThermodynamics.gas_constant_air(q_tr[1], q_tr[2], q_tr[3])=#
        R_gas = MoistThermodynamics.gas_constant_air(0.0, 0.0, 0.0)
        
        u = U/ρ
        v = V/ρ
        E = E/ρ
        P = (R_gas/c_v)*ρ*(E - 0.5*(u^2 + v^2) - gravity*y)
        E = p0/(ρ * R_gas)*( P/p0 )^(c_v/c_p)
        Q[n, _U, e] = u
        Q[n, _V, e] = v
        Q[n, _E, e] = E
    end
end
# }}}

function convert_set3c_to_set2nc_scalar(x_ndim, Q)
    DFloat          = eltype(Q)
    γ::DFloat       = _γ
    p0::DFloat      = _p0
    R_gas::DFloat   = _R_gas
    c_p::DFloat     = _c_p
    c_v::DFloat     = _c_v
    gravity::DFloat = _gravity
    
    y               = x_ndim    
    q_tr            = zeros(DFloat, 3)
    ρ, U, V, E      = Q[_ρ], Q[_U], Q[_V], Q[_E]

    q_tr  = zeros(DFloat, 3)
    
    #= Calculate air constant R_gas for moist air:
    q_tr[1] = 0.0
    q_tr[2] = 0.0
    q_tr[3] = 0.0
    for itracer = 1:_ntracers
        istate = itracer + (_nsd+2)
        
        q_tr[itracer] = Q[istate]
        #Q[istate]     = q_tr[itracer] /ρ
        
    end    
    R_gas = MoistThermodynamics.gas_constant_air(q_tr[1], q_tr[2], q_tr[3])=#
    R_gas = MoistThermodynamics.gas_constant_air(0.0, 0.0, 0.0)
    
    u = U/ρ
    v = V/ρ
    E = E/ρ
    
    P = (R_gas/c_v)*ρ*(E - 0.5*(u^2 + v^2) - gravity*y)
    #@show(P, E, y, y*gravity)
    E = p0/(ρ * R_gas)*( P/p0 )^(c_v/c_p)
    
    Q[_U] = 0.0 #u
    Q[_V] = 0.0 #v
    Q[_E] = E #θ
    
end
# }}}

# {{{ nse driver
function nse2d_sgs(::Val{dim}, ::Val{N}, mpicomm, ic, mesh, tend, iplot, visc;
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
        x, y  = vgeo[i, _x, e], vgeo[i, _y, e]
        Qinit = ic(x, y)
        
        Q[i, _ρ, e] = Qinit[3]
        Q[i, _U, e] = Qinit[1]
        Q[i, _V, e] = Qinit[2]
        Q[i, _E, e] = Qinit[4]
        
        #Add moist variables
        @inbounds for istate = 5:_nstate
            Q[i, istate, e] = Qinit[istate]
        end
        
    end

    # Convert to proper variables
    mpirank == 0 && println("[CPU] converting variables (CPU)...")
    convert_set2nc_to_set3c(Val(dim), Val(N), vgeo, Q)

    # Compute time step
    mpirank == 0 && println("[CPU] computing dt (CPU)...")
    (base_dt, Courant) = courantnumber(Val(dim), Val(N), vgeo, Q, mpicomm)
#    base_dt=0.01 #FXG DT
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

    # plot the initial condition
    mkpath("viz")
    X = ntuple(j->reshape((@view vgeo[:, _x+j-1, :]), ntuple(j->N+1,dim)...,
                          nelem), dim)
    ρ = reshape((@view Q_temp[:, _ρ, :]), ntuple(j->(N+1),dim)..., nelem)
    U = reshape((@view Q_temp[:, _U, :]), ntuple(j->(N+1),dim)..., nelem)
    V = reshape((@view Q_temp[:, _V, :]), ntuple(j->(N+1),dim)..., nelem)
    E = reshape((@view Q_temp[:, _E, :]), ntuple(j->(N+1),dim)..., nelem)
    E = E .- 300.0
    
    #Write dynamics to file:
    writemesh(@sprintf("viz/nse2d_sgs_%s_rank_%04d_step_%05d",
                       ArrType, mpirank, 0), X...;
              fields=(("Rho", ρ), ("U", U), ("V", V), ("E", E)),
              realelems=mesh.realelems)
    
    #Write tracers to files (one file per tracer):
    for itracer = 1:_ntracers
        #
        # WARNING: correct the call to writemesh to only write Qtracers
        #          to the file, without re-writing the dynamics again
        #
        istate = itracer + (_nsd+2)
        Qtracers = reshape((@view Q[:, istate, :]), ntuple(j->(N+1),dim)..., nelem)
        writemesh(@sprintf("viz/TRACER_%04d_nse2d_sgs_%s_rank_%04d_step_%05d",
                           itracer, 
                           ArrType, mpirank, 0), X...;
                  fields=(("Rho", ρ), ("U", U), ("V", V), ("E", E), ("QT", Qtracers)),
                  realelems=mesh.realelems)
    
    end
    
    mpirank == 0 && println("[DEV] starting time stepper...")
    lowstorageRK(Val(dim), Val(N), mesh, vgeo, sgeo, Q, rhs, D, dt, nsteps, tout,
                 vmapM, vmapP, mpicomm, iplot, visc; ArrType=ArrType, plotstep=plotstep)

    # plot the final solution
    Q_temp=copy(Q)
    convert_set3c_to_set2nc(Val(dim), Val(N), vgeo, Q_temp)
    X = ntuple(j->reshape((@view vgeo[:, _x+j-1, :]), ntuple(j->N+1,dim)...,
                          nelem), dim)
    ρ        = reshape((@view Q_temp[:, _ρ, :]), ntuple(j->(N+1),dim)..., nelem)
    U        = reshape((@view Q_temp[:, _U, :]), ntuple(j->(N+1),dim)..., nelem)
    V        = reshape((@view Q_temp[:, _V, :]), ntuple(j->(N+1),dim)..., nelem)
    E        = reshape((@view Q_temp[:, _E, :]), ntuple(j->(N+1),dim)..., nelem)
    E        = E .- 300.0
    writemesh(@sprintf("viz/nse2d_sgs_%s_rank_%04d_step_%05d",
                       ArrType, mpirank, nsteps), X...;
              fields=(("Rho", ρ), ("U", U), ("V", V), ("E", E)),
              realelems=mesh.realelems)

    #Write tracers to files (one file per tracer):
    for itracer = 1:_ntracers
        #
        # WARNING: correct the call to writemesh to only write Qtracers
        #          to the file, without re-writing the dynamics again
        #
        istate = itracer + (_nsd+2)        
        Qtracers = reshape((@view Q[:, istate, :]), ntuple(j->(N+1),dim)..., nelem)    
        writemesh(@sprintf("viz/TRACER_%04d_nse2d_sgs_%s_rank_%04d_step_%05d",
                           itracer, 
                           ArrType, mpirank, nsteps), X...;
                  fields=(("Rho", ρ), ("U", U), ("V", V), ("E", E), ("QT",  Qtracers)),
                  realelems=mesh.realelems)
    end

    mpirank == 0 && println("[CPU] computing final energy...")
    stats[2] = L2energysquared(Val(dim), Val(N), Q_temp, vgeo, mesh.realelems)

    stats = sqrt.(MPI.allreduce(stats, MPI.SUM, mpicomm))

    if  mpirank == 0
        @show eng0 = stats[1]
        @show engf = stats[2]
        @show Δeng = engf - eng0
        @show (minimum(E), maximum(E))
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
    #
    # Chose a problem icase:
    #
    function ic(dim, x...)
        # FIXME: Type generic?
        DFloat          = eltype(x)
        γ::DFloat       = _γ
        p0::DFloat      = _p0
        R_gas::DFloat   = _R_gas
        c_p::DFloat     = _c_p
        c_v::DFloat     = _c_v
        gravity::DFloat = _gravity

        Qinit = Array{DFloat}(undef, _nstate)

        qt, ql, qi = 0.0, 0.0, 0.0
        
        icase = _icase #defined on top of this file
        if(icase == 1)
            #
            # RTB
            #
            u0    = 0.0
            r     = sqrt((x[1]-500)^2 + (x[dim]-350)^2 )
            rc    = 250.0
            θ_ref = 300.0
            θ_c   = 0.5
            Δθ    = 0.0
            if r <= rc
                Δθ = 0.5 * θ_c * (1.0 + cos(π * r/rc))
            end
            θ   = θ_ref + Δθ
            π_k = 1.0 - gravity/(c_p*θ)*x[dim]
            c   = c_v/R_gas
            ρ   = p0/(R_gas*θ)*(π_k)^c

            U   = u0
            V   = 0.0
        
            Qinit[1] = U
            Qinit[2] = V
            Qinit[3] = ρ
            Qinit[4] = θ
            
            #ρ, U, V, E
        
            return Qinit
            
        elseif(icase == 1001)
            #
            # RTB + 1 passive tracers
            #
            u0     = 0.0
            r      = sqrt((x[1]-500)^2 + (x[dim]-350)^2 )
            rc     = 250.0

            #Thermal
            θ_ref  = 300.0
            θ_c    =   0.5
            Δθ     =   0.0
            
            #Passive
            rtracer  = sqrt((x[1]-500)^2 + (x[dim]-350)^2 )
            rctracer = 250.0
            qt_ref   =   0.0
            qt_c     =   1.0
            Δqt      =   0.0
            
            if r <= rc
                Δθ  = 0.5 * θ_c  * (1.0 + cos(π * r/rc))
                Δqt = 0.5 #* qt_c * (1.0 + cos(π * rtracer/rctracer))
            end
            
            θ   = θ_ref + Δθ
            π_k = 1.0 - gravity/(c_p*θ)*x[dim]
            c   = c_v/R_gas
            ρ   = p0/(R_gas*θ)*(π_k)^c
            qt  = qt_ref + Δqt
            
            U    = u0
            V    = 0.0
            
            Qinit[1] = U
            Qinit[2] = V
            Qinit[3] = ρ
            Qinit[4] = θ
            Qinit[5] = qt
            
            return Qinit

        elseif(icase == 1003)
            #
            # RTB + 2 passive tracers
            #
            
            u0     = 0.0
            r      = sqrt((x[1]-500)^2 + (x[dim]-350)^2 )
            rc     = 250.0

            #Thermal
            θ_ref  = 300.0
            θ_c    =   0.5
            Δθ     =   0.0
            
            #Passive
            rt1  = sqrt((x[1]-500)^2 + (x[dim]-350)^2 )
            rct1 = 250.0
            
            rt2  = sqrt((x[1]-200)^2 + (x[dim]-500)^2 )
            rct2 = 100.0
            
            rt3  = sqrt((x[1]-300)^2 + (x[dim]-400)^2 )
            rct3 = 150.0
            
            qt_ref  =   0.0
            qt_c    =   1.0
            
            Δqt1    =   0.0
            Δqt2    =   0.0
            Δqt3    =   0.0
            
            if r <= rc
                Δθ  = 0.5 * θ_c  * (1.0 + cos(π * r/rc))
            end
            if rt1 <= rct1
                Δqt1 = 0.5 * qt_c * (1.0 + cos(π * rt1/rct1))
            end
            if rt2 <= rct2
                Δqt2 = 0.5 * qt_c * (1.0 + cos(π * rt2/rct2))
            end
            if rt3 <= rct3
                Δqt3 = 0.5 * qt_c * (1.0 + cos(π * rt3/rct3))
            end
            
            θ   = θ_ref + Δθ
            π_k = 1.0 - gravity/(c_p*θ)*x[dim]
            c   = c_v/R_gas
            ρ   = p0/(R_gas*θ)*(π_k)^c

            U    = u0
            V    = 0.0
            
            qt1 = qt_ref + Δqt1
            qt2 = qt_ref + Δqt2
            qt3 = qt_ref + Δqt3
            
            
            Qinit[1] = U
            Qinit[2] = V
            Qinit[3] = ρ
            Qinit[4] = θ
            Qinit[5] = qt1
            Qinit[6] = qt2
            Qinit[7] = qt3
            
            return Qinit
            
        elseif(icase == 1010)
            #
            # Moist bubble: Pressel at al. 2015 JAMES
            #            
            u0  =    0.0
            rc  =  250.0
            r      = sqrt((x[1]-500)^2/rc^2 + (x[dim]-350)^2/rc^2 )
            
            #Thermal
            θ_ref  = 320.0
            θ_c    =   2.0

            #Moisture
            qt_ref  = 0.0196 #kg/kg
            qt      = qt_ref
            ql      = 0.0
            qi      = 0.0
            R_gas   = MoistThermodynamics.gas_constant_air(qt, ql, qi)
            
            Δθ = 0.0
            if r <= 1.0
                Δθ = θ_c * cos(0.5 * π * r)*cos(0.5 * π * r)
            end
            
            θ    = θ_ref + Δθ
            π_k  = 1.0 - gravity/(c_p*θ)*x[dim]
            c    = c_v/R_gas
            ρ    = p0/(R_gas*θ)*(π_k)^c
            P    = p0 * (ρ*R_gas*θ/p0)^(c_p/ c_v)
            T    = π_k*θ
            
            #Saturation adjustment
            T_trial  = 290.0
            E_int    = MoistThermodynamics.internal_energy_sat.(T, ρ, qt);
            T        = MoistThermodynamics.saturation_adjustment.(E_int, ρ, qt);
            θ        = T/π_k
            ρ        = p0/(R_gas*θ)*(π_k)^c
            
            #Obtain ql, qi from T,  ρ, qt
            ql = zeros(size(T)); qi = zeros(size(T))
            MoistThermodynamics.phase_partitioning_eq!(ql, qi, T, ρ, qt);
            
            #Velo
            U    = u0
            V    = 0.0

            #ρtotal = ρ_dry*(1 + qt)
            ρt = ρ*(1.0 + qt)

            Qinit[1] = U
            Qinit[2] = V
            Qinit[3] = ρt
            Qinit[4] = θ
            Qinit[5] = qt
            Qinit[6] = 0.0 #ql
            Qinit[7] = 0.0

            return Qinit
            
        else
            error(" \'ic\': Undefined case for IC. Assign a value to icase in \'main\' ")
            
        end
    end

    #Input Parameters
    time_final = DFloat(700)
    iplot=1000
    Ne = 10
    N  = 4
    visc = 1.5
    dim = _nsd
    hardware="cpu"
    if mpirank == 0
        @show (dim,N,Ne,visc,iplot,time_final,hardware,mpisize)
    end
    
    #Mesh Generation
    mesh2D = brickmesh((range(DFloat(0); length=Ne+1, stop=1000),
                        range(DFloat(0); length=Ne+1, stop=1000)),
                       (true, false),
                       part = mpirank+1, numparts = mpisize)

    #Call Solver
    if hardware == "cpu"
        mpirank == 0 && println("Running 2d (CPU)...")
        nse2d_sgs(Val(dim), Val(N), mpicomm, (x...)->ic(dim, x...), mesh2D, time_final, iplot, visc;
              ArrType=Array, tout = 10)
        mpirank == 0 && println()
    elseif hardware == "gpu"
        @hascuda begin
            mpirank == 0 && println("Running 2d (GPU)...")
            nse2d_sgs(Val(dim), Val(N), mpicomm, (x...)->ic(dim, x...), mesh2D, time_final, iplot, visc;
                  ArrType=CuArray, tout = 10)
            mpirank == 0 && println()
        end
    end
    nothing
end
# }}}

main()
