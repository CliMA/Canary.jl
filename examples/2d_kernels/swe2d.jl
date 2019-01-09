#--------------------------------Markdown Language Header-----------------------
# # 2D Shallow Water Equations
#
#
#-
#
#-
# ## Introduction
#
# This example shows how to solve the 2D shallow water equations using vanilla DG.
#
# ## Continuous Governing Equations
# We solve the following equation:
#
# ```math
# \frac{\partial h_s}{\partial t} + \nabla \cdot \mathbf{U} = \nu \nabla^2 h_s \; \; (1.1)
# ```
# ```math
# \frac{\partial \mathbf{U}}{\partial t} + \nabla \cdot \left( \frac{\mathbf{U} \otimes \mathbf{U}}{h} + g (h^2 - h^2_b) \mathbf{I}_2 \right) + h_s \nabla h_b = \nu \nabla^2 \mathbf{U} \; \; (1.2)
# ```
# where $\mathbf{u}=(u)$ and $\mathbf{U}=h \mathbf{u}$, with $h=h_s(\mathbf{x},t) + h_b(\mathbf{x})$ being the total water column with $h_s$ and $h_b$ being the height of the water surface and depth of the bathymetry (which we assume to be constant for simplicity), respectively, measured from a zero mean sea-level.  In addition, $\nu$ is the artificial viscosity parameter. We employ periodic or no-flux boundary conditions.
#
#-
# ## Discontinous Galerkin Method
# To solve Eq. (1) we use the discontinuous Galerkin method with basis functions comprised of Lagrange polynomials based on Lobatto points. Multiplying Eq. (1) by a test function $\psi$ and integrating within each element $\Omega_e$ such that $\Omega = \bigcup_{e=1}^{N_e} \Omega_e$ we get
#
# ```math
# \int_{\Omega_e} \psi \frac{\partial \mathbf{q}^{(e)}_N}{\partial t} d\Omega_e + \int_{\Omega_e} \psi \nabla \cdot \mathbf{f}^{(e)}_N d\Omega_e =  \int_{\Omega_e} \psi S\left( q^{(e)}_N \right) d\Omega_e \; \; (2)
# ```
# where $\mathbf{q}^{(e)}_N=\sum_{i=1}^{(N+1)^{dim}} \psi_i(\mathbf{x}) \mathbf{q}_i(t)$ is the finite dimensional expansion with basis functions $\psi(\mathbf{x})$, where $\mathbf{q}=\left( h, \mathbf{U}^T \right)^T$ and
# ```math
# \mathbf{f}=\left( \mathbf{U}, \frac{\mathbf{U} \otimes \mathbf{U}}{h} + g (h^2 - h^2_b) \mathbf{I}_2 \right).
# ```
# Integrating Eq. (2) by parts yields
#
# ```math
# \int_{\Omega_e} \psi \frac{\partial \mathbf{q}^{(e)}_N}{\partial t} d\Omega_e + \int_{\Gamma_e} \psi \mathbf{n} \cdot \mathbf{f}^{(*,e)}_N d\Gamma_e - \int_{\Omega_e} \nabla \psi \cdot \mathbf{f}^{(e)}_N d\Omega_e = \int_{\Omega_e} \psi S\left( q^{(e)}_N \right) d\Omega_e \; \; (3)
# ```
#
# where the second term on the left denotes the flux integral term (computed in "function fluxrhs") and the third term denotes the volume integral term (computed in "function volume\_rhs").  The superscript $(*,e)$ in the flux integral term denotes the numerical flux. Here we use the Rusanov flux.
#
#-
# ## Local Discontinous Galerkin Method
# To approximate the second order terms on the right hand side of Eq. (1) we use the local discontinuous Galerkin (LDG) method, which we described in LDG2d.jl. We will highlight the main steps below for completeness. The operator $\nabla^2$ is approximated by the following two-step process: first we approximate the gradient of $q$ as follows
# ```math
# \mathbf{Q}(x,y) = \nabla q(x,y) \; \; (2)
# ```
# where $\mathbf{Q}$ is an auxiliary vector function, followed by
# ```math
# \nabla \cdot \mathbf{Q} (x,y) =  \nabla^2 q(x,y) \; \; (3)
# ```
# which represents the Laplacian of $q$.
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
const _nstate = 4
const _U, _V, _h, _b = 1:_nstate
const stateid = (U = _U, V = _V, h = _h, b = _b)

const _nvgeo = 8
const _ξx, _ηx, _ξy, _ηy, _MJ, _MJI, _x, _y, = 1:_nvgeo

const _nsgeo = 4
const _nx, _ny, _sMJ, _vMJI = 1:_nsgeo
# }}}

# {{{ courant
function courantnumber(::Val{dim}, ::Val{N}, vgeo, Q, mpicomm, gravity, δnl, advection) where {dim, N}
    DFloat = eltype(Q)
    Np = (N+1)^dim
    (~, ~, nelem) = size(Q)
    dt = [floatmax(DFloat)]
    Courant = - [floatmax(DFloat)]
    δ_wave=1
    if advection
        δ_wave=0
    end

    @inbounds for e = 1:nelem, n = 1:Np
        h, b, U, V = Q[n, _h, e],  Q[n, _b, e], Q[n, _U, e], Q[n, _V, e]
        ξx, ξy, ηx, ηy = vgeo[n, _ξx, e], vgeo[n, _ξy, e],
        vgeo[n, _ηx, e], vgeo[n, _ηy, e]
        H = h+b
        u=U/H
        v=V/H
        dx=sqrt( (1.0/(2*ξx))^2 + (1.0/(2*ηy))^2 )
        vel=sqrt( u^2 + v^2 )
        wave_speed = (vel + δ_wave*sqrt(gravity*H)*δnl + gravity*b*(1-δnl))
        loc_dt = 0.5*dx/wave_speed/N
        dt[1] = min(dt[1], loc_dt)
    end
    dt_min=MPI.Allreduce(dt[1], MPI.MIN, mpicomm)

        #Compute Courant
    @inbounds for e = 1:nelem, n = 1:Np
        h, b, U, V = Q[n, _h, e],  Q[n, _b, e], Q[n, _U, e], Q[n, _V, e]
        ξx, ξy, ηx, ηy = vgeo[n, _ξx, e], vgeo[n, _ξy, e],
        vgeo[n, _ηx, e], vgeo[n, _ηy, e]
        H = h+b
        u=U/H
        v=V/H
        dx=sqrt( (1.0/(2*ξx))^2 + (1.0/(2*ηy))^2 )
        vel=sqrt( u^2 + v^2 )
        wave_speed = (vel + δ_wave*sqrt(gravity*H)*δnl + gravity*b*(1-δnl))
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
# {{{ Volume RHS
function volumerhs!(::Val{dim}, ::Val{N}, rhs::Array, Q, vgeo, D, elems, gravity, δnl) where {dim, N}
    DFloat = eltype(Q)
    Nq = N + 1
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

            #Compute fluxes
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

# Flux RHS
function fluxrhs!(::Val{dim}, ::Val{N}, rhs::Array,  Q, sgeo, elems, vmapM, vmapP, elemtobndy, gravity, δnl) where {dim, N}
    DFloat = eltype(Q)
    Np = (N+1)^dim
    Nfp = (N+1)^(dim-1)
    nface = 2*dim

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

# {{{ Volume grad(Q)
function volume_grad!(::Val{dim}, ::Val{N}, rhs::Array, Q, vgeo, D, elems) where {dim, N}
    DFloat = eltype(Q)
    Nq = N + 1
    nelem = size(Q)[end]

    Q = reshape(Q, Nq, Nq, _nstate, nelem)
    rhs = reshape(rhs, Nq, Nq, _nstate, dim, nelem)
    vgeo = reshape(vgeo, Nq, Nq, _nvgeo, nelem)

    #Initialize RHS vector
    fill!( rhs, zero(rhs[1]))

    #Allocate Arrays
    s_F = Array{DFloat}(undef, Nq, Nq, _nstate-1, dim)
    s_G = Array{DFloat}(undef, Nq, Nq, _nstate-1, dim)

    @inbounds for e in elems
        for j = 1:Nq, i = 1:Nq
            MJ = vgeo[i, j, _MJ, e]
            ξx, ξy = vgeo[i,j,_ξx,e], vgeo[i,j,_ξy,e]
            ηx, ηy = vgeo[i,j,_ηx,e], vgeo[i,j,_ηy,e]
            U, V = Q[i, j, _U, e], Q[i, j, _V, e]
            h, b = Q[i, j, _h, e], Q[i, j, _b, e]

            #Compute fluxes
            fluxh = h
            fluxU = U
            fluxV = V

            s_F[i, j, _h, 1] = MJ * (ξx * fluxh)
            s_F[i, j, _h, 2] = MJ * (ξy * fluxh)
            s_F[i, j, _U, 1] = MJ * (ξx * fluxU)
            s_F[i, j, _U, 2] = MJ * (ξy * fluxU)
            s_F[i, j, _V, 1] = MJ * (ξx * fluxV)
            s_F[i, j, _V, 2] = MJ * (ξy * fluxV)

            s_G[i, j, _h, 1] = MJ * (ηx * fluxh)
            s_G[i, j, _h, 2] = MJ * (ηy * fluxh)
            s_G[i, j, _U, 1] = MJ * (ηx * fluxU)
            s_G[i, j, _U, 2] = MJ * (ηy * fluxU)
            s_G[i, j, _V, 1] = MJ * (ηx * fluxV)
            s_G[i, j, _V, 2] = MJ * (ηy * fluxV)
        end

        # loop of ξ-grid lines
        for s = 1:_nstate-1, j = 1:Nq, i = 1:Nq, n = 1:Nq
            rhs[i, j, s, 1, e] -= D[n, i] * s_F[n, j, s, 1]
            rhs[i, j, s, 2, e] -= D[n, i] * s_F[n, j, s, 2]
        end
        # loop of η-grid lines
        for s = 1:_nstate-1, j = 1:Nq, i = 1:Nq, n = 1:Nq
            rhs[i, j, s, 1, e] -= D[n, j] * s_G[i, n, s, 1]
            rhs[i, j, s, 2, e] -= D[n, j] * s_G[i, n, s, 2]
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
                    bP = Q[vidP, _b, eP]
                    UP = Q[vidP, _U, eP]
                    VP = Q[vidP, _V, eP]
                elseif bc == 1
                    UnM = nxM * UM +  nyM * VM
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
                fluxhM = hM
                fluxUM = UM
                fluxVM = VM

                #Right Fluxes
                fluxhP = hP
                fluxUP = UP
                fluxVP = VP

                #Compute Numerical/Rusanov Flux
                fluxhS = 0.5*(fluxhM + fluxhP)
                fluxUS = 0.5*(fluxUM + fluxUP)
                fluxVS = 0.5*(fluxVM + fluxVP)

                #Update RHS
                rhs[vidM, _h, 1, eM] += sMJ * nxM*fluxhS
                rhs[vidM, _h, 2, eM] += sMJ * nyM*fluxhS
                rhs[vidM, _U, 1, eM] += sMJ * nxM*fluxUS
                rhs[vidM, _U, 2, eM] += sMJ * nyM*fluxUS
                rhs[vidM, _V, 1, eM] += sMJ * nxM*fluxVS
                rhs[vidM, _V, 2, eM] += sMJ * nyM*fluxVS
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

    Q = reshape(Q, Nq, Nq, _nstate, dim, nelem)
    rhs = reshape(rhs, Nq, Nq, _nstate, dim, nelem)
    vgeo = reshape(vgeo, Nq, Nq, _nvgeo, nelem)

    #Initialize RHS vector
    fill!( rhs, zero(rhs[1]))

    #Allocate Arrays
    s_F = Array{DFloat}(undef, Nq, Nq, _nstate-1)
    s_G = Array{DFloat}(undef, Nq, Nq, _nstate-1)

    @inbounds for e in elems
        for j = 1:Nq, i = 1:Nq
            MJ = vgeo[i, j, _MJ, e]
            ξx, ξy = vgeo[i,j,_ξx,e], vgeo[i,j,_ξy,e]
            ηx, ηy = vgeo[i,j,_ηx,e], vgeo[i,j,_ηy,e]
            hx, hy = Q[i, j, _h, 1, e], Q[i, j, _h, 2, e]
            Ux, Uy = Q[i, j, _U, 1, e], Q[i, j, _U, 2, e]
            Vx, Vy = Q[i, j, _V, 1, e], Q[i, j, _V, 2, e]

            #Compute fluxes
            fluxh_x = hx
            fluxh_y = hy
            fluxU_x = Ux
            fluxU_y = Uy
            fluxV_x = Vx
            fluxV_y = Vy

            s_F[i, j, _h] = MJ * (ξx * fluxh_x + ξy * fluxh_y)
            s_F[i, j, _U] = MJ * (ξx * fluxU_x + ξy * fluxU_y)
            s_F[i, j, _V] = MJ * (ξx * fluxV_x + ξy * fluxV_y)

            s_G[i, j, _h] = MJ * (ηx * fluxh_x + ηy * fluxh_y)
            s_G[i, j, _U] = MJ * (ηx * fluxU_x + ηy * fluxU_y)
            s_G[i, j, _V] = MJ * (ηx * fluxV_x + ηy * fluxV_y)
        end

        # loop of ξ-grid lines
        for s = 1:_nstate-1, j = 1:Nq, i = 1:Nq, n = 1:Nq
            rhs[i, j, s, 1, e] -= D[n, i] * s_F[n, j, s]
        end
        # loop of η-grid lines
        for s = 1:_nstate-1, j = 1:Nq, i = 1:Nq, n = 1:Nq
            rhs[i, j, s, 1, e] -= D[n, j] * s_G[i, n, s]
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
                nxM, nyM, sMJ = sgeo[_nx, n, f, e], sgeo[_ny, n, f, e], sgeo[_sMJ, n, f, e]
                idM, idP = vmapM[n, f, e], vmapP[n, f, e]

                eM, eP = e, ((idP - 1) ÷ Np) + 1
                vidM, vidP = ((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1

                hxM = Q[vidM, _h, 1, eM]
                hyM = Q[vidM, _h, 2, eM]
                UxM = Q[vidM, _U, 1, eM]
                UyM = Q[vidM, _U, 2, eM]
                VxM = Q[vidM, _V, 1, eM]
                VyM = Q[vidM, _V, 2, eM]

                bc = elemtobndy[f, e]
                hxP = hyP = UxP = UyP = VxP = VyP = zero(eltype(Q))
                if bc == 0
                    hxP = Q[vidP, _h, 1, eP]
                    hyP = Q[vidP, _h, 2, eP]
                    UxP = Q[vidP, _U, 1, eP]
                    UyP = Q[vidP, _U, 2, eP]
                    VxP = Q[vidP, _V, 1, eP]
                    VyP = Q[vidP, _V, 2, eP]
                elseif bc == 1
                    hnM = nxM * hxM +  nyM * hyM
                    hxP = hxM - 2 * hnM * nxM
                    hyP = hyM - 2 * hnM * nyM
                    UnM = nxM * UxM +  nyM * UyM
                    UxP = UxM - 2 * UnM * nxM
                    UyP = UyM - 2 * UnM * nyM
                    VnM = nxM * VxM +  nyM * VyM
                    VxP = VxM - 2 * VnM * nxM
                    VyP = VyM - 2 * VnM * nyM
                else
                    error("Invalid boundary conditions $bc on face $f of element $e")
                end

                #Left Fluxes
                fluxhM_x = hxM
                fluxhM_y = hyM
                fluxUM_x = UxM
                fluxUM_y = UyM
                fluxVM_x = VxM
                fluxVM_y = VyM

                #Right Fluxes
                fluxhP_x = hxP
                fluxhP_y = hyP
                fluxUP_x = UxP
                fluxUP_y = UyP
                fluxVP_x = VxP
                fluxVP_y = VyP

                #Compute Numerical/Rusanov Flux
                fluxhS = 0.5*(nxM * (fluxhM_x + fluxhP_x) + nyM * (fluxhM_y + fluxhP_y))
                fluxUS = 0.5*(nxM * (fluxUM_x + fluxUP_x) + nyM * (fluxUM_y + fluxUP_y))
                fluxVS = 0.5*(nxM * (fluxVM_x + fluxVP_x) + nyM * (fluxVM_y + fluxVP_y))

                #Update RHS
                rhs[vidM, _h, 1, eM] += sMJ * fluxhS
                rhs[vidM, _U, 1, eM] += sMJ * fluxUS
                rhs[vidM, _V, 1, eM] += sMJ * fluxVS
            end
        end
    end
end
# }}}

# {{{ Update grad Q solution
function update_gradQ!(::Val{dim}, ::Val{N}, Q, rhs, vgeo, elems) where {dim, N}

    Nq=(N+1)^dim

    @inbounds for e = elems, s = 1:_nstate-1, i = 1:Nq
        Q[i, s, 1, e] = rhs[i, s, 1, e] * vgeo[i, _MJI, e]
        Q[i, s, 2, e] = rhs[i, s, 2, e] * vgeo[i, _MJI, e]
    end

end
# }}}

# {{{ Update grad Q solution
function update_divgradQ!(::Val{dim}, ::Val{N}, Q, rhs, vgeo, elems) where {dim, N}

    Nq=(N+1)^dim

    @inbounds for e = elems, s = 1:_nstate-1, i = 1:Nq
        Q[i, s, e] = rhs[i, s, 1, e] * vgeo[i, _MJI, e]
    end

end
# }}}

# {{{ Update solution
function updatesolution!(::Val{dim}, ::Val{N}, rhs, rhs_gradQ, Q, vgeo, elems, rka, rkb, dt, advection, visc) where {dim, N}

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
        rhs[i, s, e] += visc*rhs_gradQ[i,s,1,e]
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
                      dt, nsteps, tout, vmapM, vmapP, mpicomm,
                      gravity, δnl, advection, visc;
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

    # Begin Time Loop
    start_time = t1 = time_ns()
    for step = 1:nsteps
        for s = 1:length(RKA)

            #---------------1st Order Operators--------------------------#
            # Send Data Q
            senddata_Q(Val(dim), Val(N), mesh, sendreq, recvreq, sendQ,
                       recvQ, d_sendelems, d_sendQ, d_recvQ, d_QL, mpicomm;
                       ArrType=ArrType)

            # volume RHS computation
            volumerhs!(Val(dim), Val(N), d_rhsC, d_QC, d_vgeoC, d_D, mesh.realelems, gravity, δnl)

            # Receive Data Q
            receivedata_Q!(Val(dim), Val(N), mesh, recvreq, recvQ, d_recvQ, d_QL)

            # face RHS computation
            fluxrhs!(Val(dim), Val(N), d_rhsL, d_QL, d_sgeo, mesh.realelems, d_vmapM, d_vmapP, d_elemtobndy, gravity, δnl)

            #---------------2nd Order Operators--------------------------#
            if (visc > 0)
                # volume grad Q computation
                volume_grad!(Val(dim), Val(N), d_rhs_gradQC, d_QC, d_vgeoC, d_D, mesh.realelems)

                # face grad Q computation
                flux_grad!(Val(dim), Val(N), d_rhs_gradQL, d_QL, d_sgeo, mesh.realelems, d_vmapM, d_vmapP, d_elemtobndy)

                # Construct grad Q
                update_gradQ!(Val(dim), Val(N), d_gradQL, d_rhs_gradQL, d_vgeoL, mesh.realelems)

                # Send Data grad(Q)
                senddata_gradQ(Val(dim), Val(N), mesh, sendreq, recvreq, sendgradQ,
                               recvgradQ, d_sendelems, d_sendgradQ, d_recvgradQ,
                               d_gradQL, mpicomm;ArrType=ArrType)

                # volume div (grad Q) computation
                volume_div!(Val(dim), Val(N), d_rhs_gradQC, d_gradQC, d_vgeoC, d_D, mesh.realelems)

                # Receive Data grad(Q)
                receivedata_gradQ!(Val(dim), Val(N), mesh, recvreq, recvgradQ, d_recvgradQ, d_gradQL)

                # face div (grad Q) computation
                flux_div!(Val(dim), Val(N), d_rhs_gradQL, d_gradQL, d_sgeo, mesh.realelems, d_vmapM, d_vmapP, d_elemtobndy)
            end

            #---------------Update Solution--------------------------#
            # update solution and scale RHS
            updatesolution!(Val(dim), Val(N), d_rhsL, d_rhs_gradQL, d_QL, d_vgeoL, mesh.realelems,
                            RKA[s%length(RKA)+1], RKB[s], dt, advection, visc)
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
        # Write VTK file
        if plotstep > 0 && step % plotstep == 0
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
             advection, visc; meshwarp=(x...)->identity(x), tout = 60, ArrType=Array,
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
    (base_dt, Courant) = courantnumber(Val(dim), Val(N), vgeo, Q, mpicomm, gravity, δnl, advection)
    mpirank == 0 && @show (base_dt,Courant)

    nsteps = ceil(Int64, tend / base_dt)
    dt = tend / nsteps
    mpirank == 0 && @show (dt, nsteps, dt * nsteps, tend)

    # Do time stepping
    stats = zeros(DFloat, 3)
    mpirank == 0 && println("[CPU] computing initial energy...")
    stats[1] = L2energysquared(Val(dim), Val(N), Q, vgeo, mesh.realelems)

    # plot initial condition
    mkpath("viz")
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

    #Call Time-stepping Routine
    mpirank == 0 && println("[DEV] starting time stepper...")
    lowstorageRK(Val(dim), Val(N), mesh, vgeo, sgeo, Q, rhs, D, dt, nsteps, tout,
                 vmapM, vmapP, mpicomm, gravity, δnl, advection, visc;
                 ArrType=ArrType, plotstep=plotstep)

    # plot final solution
    X = ntuple(j->reshape((@view vgeo[:, _x+j-1, :]), ntuple(j->N+1,dim)...,
                          nelem), dim)
    h = reshape((@view Q[:, _h, :]), ntuple(j->(N+1),dim)..., nelem)
    b = reshape((@view Q[:, _b, :]), ntuple(j->(N+1),dim)..., nelem)
    U = reshape((@view Q[:, _U, :]), ntuple(j->(N+1),dim)..., nelem) ./ (h.+b)
    V = reshape((@view Q[:, _V, :]), ntuple(j->(N+1),dim)..., nelem) ./ (h.+b)
    writemesh(@sprintf("viz/swe%dD_%s_rank_%04d_step_%05d",
                       dim, ArrType, mpirank, nsteps), X...;
              fields=(("h", h),("b",b),("U",U),("V",V),),
              realelems=mesh.realelems)

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

    MPI.Initialized() || MPI.Init()
    MPI.finalize_atexit()

    mpicomm = MPI.COMM_WORLD
    mpirank = MPI.Comm_rank(mpicomm)
    mpisize = MPI.Comm_size(mpicomm)

    # FIXME: query via hostname
    @hascuda device!(mpirank % length(devices()))

    #Input Parameters
    N=8
    Ne=10
    visc=0.001
    iplot=10
    δnl=1
    icase=20
    time_final=DFloat(0.32)
    hardware="cpu"
    if mpirank == 0
        @show (N,Ne,visc,iplot,δnl,icase,time_final,hardware,mpisize)
    end

    #Initial Conditions
    ic = (x...) -> (0.0, 0.0, 0.0, 0.0)
    if icase == 1 #advection
        function ic1(x...)
            r = sqrt( (x[1]-0.5)^2 + (x[2]-0.5)^2 )
            h = 0.5 * exp(-100.0 * r^2)
            b = 1.0
            H = h + b
            U = H*(1.0)
            V = H*(0.0)
            h, b, U, V
        end
        ic = ic1
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
    elseif icase == 20 #shallow water with Periodic BCs
        function ic20(x...)
            r = sqrt( (x[1]-0.5)^2 + (x[2]-0.5)^2 )
            rc=0.25
            h=0
            if (r<=rc)
                h = 0.25
            end
            b=1.0
            H = h + b
            U = H*(0.0)
            V = H*(0.0)
            h, b, U, V
        end
        ic = ic20
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
        swe(Val(2), Val(N), mpicomm, ic, mesh, time_final, gravity, δnl, advection, visc;
            ArrType=Array, tout = 10, plotstep = iplot)
        mpirank == 0 && println()
    elseif hardware == "gpu"
        @hascuda begin
            mpirank == 0 && println("Running (GPU)...")
            swe(Val(2), Val(N), mpicomm, ic, mesh, time_final, gravity, δnl, advection, visc;
                ArrType=CuArray, tout = 10, plotstep = iplot)
            mpirank == 0 && println()
        end
    end
    nothing
end
# }}}

main()
