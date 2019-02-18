#--------------------------------Markdown Language Header-----------------------
# # 1D Shallow Water Equations
#
#
#-
#
#-
# ## Introduction
#
# This example shows how to solve the 1D shallow water equations using vanilla DG.
#
# ## Continuous Governing Equations
# We solve the following equation:
#
# ```math
# \frac{\partial h_s}{\partial t} + \nabla \cdot \mathbf{U} = 0 \; \; (1.1)
# ```
# ```math
# \frac{\partial \mathbf{U}}{\partial t} + \nabla \cdot \left( \frac{\mathbf{U} \otimes \mathbf{U}}{h} + g (h^2 - h^2_b) \mathbf{I}_2 \right) + h_s \nabla h_b = 0 \; \; (1.2)
# ```
# where $\mathbf{u}=(u)$ and $\mathbf{U}=h \mathbf{u}$, with $h=h_s(\mathbf{x},t) + h_b(\mathbf{x})$ being the total water column with $h_s$ and $h_b$ being the height of the water surface and depth of the bathymetry (which we assume to be constant for simplicity), respectively, measured from a zero mean sea-level.  We employ periodic boundary conditions.
#
#-
# ## Discontinous Galerkin Method
# To solve Eq. (1) we use the discontinuous Galerkin method with basis functions comprised of Lagrange polynomials based on Lobatto points. Multiplying Eq. (1) by a test function $\psi$ and integrating within each element $\Omega_e$ such that $\Omega = \bigcup_{e=1}^{N_e} \Omega_e$ we get
# ```math
# \int_{\Omega_e} \psi \frac{\partial \mathbf{q}^{(e)}_N}{\partial t} d\Omega_e + \int_{\Omega_e} \psi \nabla \cdot \mathbf{f}^{(e)}_N d\Omega_e =  \int_{\Omega_e} \psi S\left( q^{(e)}_N \right) d\Omega_e \; \; (2)
# ```
# where $\mathbf{q}^{(e)}_N=\sum_{i=1}^{(N+1)^{dim}} \psi_i(\mathbf{x}) \mathbf{q}_i(t)$ is the finite dimensional expansion with basis functions $\psi(\mathbf{x})$, where $\mathbf{q}=\left( h, \mathbf{U}^T \right)^T$ and
# ```math
# \mathbf{f}=\left( \mathbf{U}, \frac{\mathbf{U} \otimes \mathbf{U}}{h} + g (h^2 - h^2_b) \mathbf{I}_2 \right).
# ```
#
# Integrating Eq. (2) by parts yields
# ```math
# \int_{\Omega_e} \psi \frac{\partial \mathbf{q}^{(e)}_N}{\partial t} d\Omega_e + \int_{\Gamma_e} \psi \mathbf{n} \cdot \mathbf{f}^{(*,e)}_N d\Gamma_e - \int_{\Omega_e} \nabla \psi \cdot \mathbf{f}^{(e)}_N d\Omega_e = \int_{\Omega_e} \psi S\left( q^{(e)}_N \right) d\Omega_e \; \; (3)
# ```
# where the second term on the left denotes the flux integral term (computed in "function fluxrhs") and the third term denotes the volume integral term (computed in "function volumerhs").  The superscript $(*,e)$ in the flux integral term denotes the numerical flux. Here we use the Rusanov flux.
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
    using CuArrays
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
const _nstate = 3
const _U, _h, _b = 1:_nstate
const stateid = (U = _U, h = _h, b = _b)

const _nvgeo = 4
const _ξx, _MJ, _MJI, _x = 1:_nvgeo

const _nsgeo = 3
const _nx, _sMJ, _vMJI = 1:_nsgeo
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

    #Compute DT
    @inbounds for e = 1:nelem, n = 1:Np
        h, b, U = Q[n, _h, e],  Q[n, _b, e], Q[n, _U, e]
        ξx = vgeo[n, _ξx, e]
        H = h+b
        u=U/H
        dx=1.0/(2*ξx)
        wave_speed = ( abs(u) + δ_wave*sqrt(gravity*H)*δnl + gravity*b*(1-δnl) )
        loc_dt = 0.5*dx/wave_speed/N
        dt[1] = min(dt[1], loc_dt)
    end
    dt_min=MPI.Allreduce(dt[1], MPI.MIN, mpicomm)

    #Compute Courant
    @inbounds for e = 1:nelem, n = 1:Np
        h, b, U = Q[n, _h, e],  Q[n, _b, e], Q[n, _U, e]
        ξx = vgeo[n, _ξx, e]
        H = h+b
        u=U/H
        dx=1.0/(2*ξx)
        wave_speed = ( abs(u) + δ_wave*sqrt(gravity*H)*δnl + gravity*b*(1-δnl) )
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

    (ξx, MJ, MJI, x) = ntuple(j->(@view vgeo[:, j, :]), _nvgeo)
    J = similar(x)
    (nx, sMJ, vMJI) = ntuple(j->(@view sgeo[ j, :, :, :]), _nsgeo)
    sJ = similar(sMJ)

    X = ntuple(j->(@view vgeo[:, _x+j-1, :]), dim)
    creategrid!(X..., mesh.elemtocoord, ξ)

    @inbounds for j = 1:length(x)
        #    (x[j]) = meshwarp(x[j],)
    end

    # Compute the metric terms
    computemetric!(x, J, ξx, sJ, nx, D)

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
# {{{ Volume RHS for 1D
function volumerhs!(::Val{1}, ::Val{N}, rhs::Array, Q, vgeo, D, elems, gravity, δnl) where N
    DFloat = eltype(Q)
    Nq = N + 1
    nelem = size(Q)[end]

    Q = reshape(Q, Nq, _nstate, nelem)
    rhs = reshape(rhs, Nq, _nstate, nelem)
    vgeo = reshape(vgeo, Nq, _nvgeo, nelem)

    #Allocate Arrays
    s_F = Array{DFloat}(undef, Nq, _nstate-1)

    @inbounds for e in elems
        for i = 1:Nq
            MJ, ξx = vgeo[i,_MJ,e], vgeo[i,_ξx,e]
            h, b, U = Q[i, _h, e], Q[i, _b, e], Q[i, _U, e]

            #Get primitive variables
            H=h + b
            u=U/H

            #Compute fluxes
            fluxh_x = U
            fluxU_x = (H * u * u + 0.5 * gravity * h^2) * δnl + gravity * h * b

            s_F[i, _h] = MJ * (ξx * fluxh_x)
            s_F[i, _U] = MJ * (ξx * fluxU_x)
        end

        # loop of ξ-grid lines
        for s = 1:_nstate-1, i = 1:Nq, k = 1:Nq
            rhs[i, s, e] += D[k, i] * s_F[k, s]
        end
    end
end

# }}}

# Flux RHS for 1D
function fluxrhs!(::Val{1}, ::Val{N}, rhs::Array,  Q, sgeo, elems, vmapM, vmapP, elemtobndy, gravity, δnl) where N
    DFloat = eltype(Q)
    Np = (N+1)
    Nfp = 1
    nface = 2

    @inbounds for e in elems
        for f = 1:nface
            for n = 1:Nfp
                nxM, sMJ = sgeo[_nx, n, f, e], sgeo[_sMJ, n, f, e]
                idM, idP = vmapM[n, f, e], vmapP[n, f, e]

                eM, eP = e, ((idP - 1) ÷ Np) + 1
                vidM, vidP = ((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1

                hM = Q[vidM, _h, eM]
                bM = Q[vidM, _b, eM]
                UM = Q[vidM, _U, eM]

                bc = elemtobndy[f, e]
                hP = bP = UP = VP = zero(eltype(Q))
                if bc == 0
                    hP = Q[vidP, _h, eP]
                    bP = Q[vidP, _b, eM]
                    UP = Q[vidP, _U, eP]
                elseif bc == 1
                    UnM = nxM * UM
                    UP = UM - 2 * UnM * nxM
                    hP = hM
                    bP = bM
                else
                    error("Invalid boundary conditions $bc on face $f of element $e")
                end
                HM=hM + bM
                uM = UM / HM
                HP=hP + bP
                uP = UP / HP

                #Left Fluxes
                fluxhM_x = UM
                fluxUM_x = (HM * uM * uM + 0.5 * gravity * hM^2) * δnl + gravity * hM * bM

                #Right Fluxes
                fluxhP_x = UP
                fluxUP_x = (HP * uP * uP + 0.5 * gravity * hP^2) * δnl + gravity * hP * bP

                #Compute wave speed
                λM=( abs(nxM * uM) + sqrt(gravity*HM) ) * δnl + ( sqrt(gravity*bM) ) * (1.0-δnl)
                λP=( abs(nxM * uP) + sqrt(gravity*HP) ) * δnl + ( sqrt(gravity*bP) ) * (1.0-δnl)
                λ = max( λM, λP )

                #Compute Numerical/Rusanov Flux
                fluxhS = (nxM * (fluxhM_x + fluxhP_x) - λ * (hP - hM)) / 2
                fluxUS = (nxM * (fluxUM_x + fluxUP_x) - λ * (UP - UM)) / 2

                #Update RHS
                rhs[vidM, _h, eM] -= sMJ * fluxhS
                rhs[vidM, _U, eM] -= sMJ * fluxUS
            end
        end
    end
end
# }}}

# {{{ Volume Q
function volumeQ!(::Val{1}, ::Val{N}, rhs::Array, Q, vgeo, D, elems) where N
    DFloat = eltype(Q)
    Nq = N + 1
    nelem = size(Q)[end]

    Q = reshape(Q, Nq, _nstate, nelem)
    rhs = reshape(rhs, Nq, _nstate, nelem)
    vgeo = reshape(vgeo, Nq, _nvgeo, nelem)

    #Initialize RHS vector
    fill!( rhs, zero(rhs[1]))

    #Allocate Arrays
    s_F = Array{DFloat}(undef, Nq, _nstate-1)

    @inbounds for e in elems
        for i = 1:Nq
            MJ, ξx = vgeo[i,_MJ,e], vgeo[i,_ξx,e]
            h, b, U = Q[i, _h, e], Q[i, _b, e], Q[i, _U, e]

            #Get primitive variables
            H=h + b
            u=U/H

            #Compute fluxes
            fluxh = h
            fluxU = U
            s_F[i, _h] = MJ * (ξx * fluxh)
            s_F[i, _U] = MJ * (ξx * fluxU)
        end

        # loop of ξ-grid lines
        for s = 1:_nstate-1, i = 1:Nq, k = 1:Nq
            rhs[i, s, e] -= D[k, i] * s_F[k, s]
        end
    end
end
# }}}

# Flux Q
function fluxQ!(::Val{1}, ::Val{N}, rhs::Array,  Q, sgeo, elems, vmapM, vmapP, elemtobndy) where N
    DFloat = eltype(Q)
    Np = (N+1)
    Nfp = 1
    nface = 2

    @inbounds for e in elems
        for f = 1:nface
            for n = 1:Nfp
                nxM, sMJ = sgeo[_nx, n, f, e], sgeo[_sMJ, n, f, e]
                idM, idP = vmapM[n, f, e], vmapP[n, f, e]

                eM, eP = e, ((idP - 1) ÷ Np) + 1
                vidM, vidP = ((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1

                hM = Q[vidM, _h, eM]
                bM = Q[vidM, _b, eM]
                UM = Q[vidM, _U, eM]

                bc = elemtobndy[f, e]
                hP = bP = UP = VP = zero(eltype(Q))
                if bc == 0
                    hP = Q[vidP, _h, eP]
                    bP = Q[vidP, _b, eM]
                    UP = Q[vidP, _U, eP]
                elseif bc == 1
                    UnM = nxM * UM
                    UP = UM - 2 * UnM * nxM
                    hP = hM
                    bP = bM
                else
                    error("Invalid boundary conditions $bc on face $f of element $e")
                end
                HM=hM + bM
                uM = UM / HM
                HP=hP + bP
                uP = UP / HP

                #Left Fluxes
                fluxhM = hM
                fluxUM = UM

                #Right Fluxes
                fluxhP = hP
                fluxUP = UP

                #Compute Numerical/Rusanov Flux
                fluxhS = 0.5*(fluxhM + fluxhP)
                fluxUS = 0.5*(fluxUM + fluxUP)

                #Update RHS
                rhs[vidM, _h, eM] += sMJ * nxM * fluxhS
                rhs[vidM, _U, eM] += sMJ * nxM * fluxUS
            end
        end
    end
end
# }}}

# {{{ Update grad Q solution
function update_gradQ!(::Val{dim}, ::Val{N}, Q, rhs, vgeo, elems) where {dim, N}

    DFloat = eltype(Q)
    Nq=(N+1)^dim
    (~, ~, nelem) = size(Q)

    @inbounds for e = elems, s = 1:_nstate-1, i = 1:Nq
        Q[i, s, e] = rhs[i, s, e] * vgeo[i, _MJI, e]
    end

end
# }}}

# {{{ Update solution
function updatesolution!(::Val{dim}, ::Val{N}, rhs, Q, rhs_gradQ, vgeo, elems, rka, rkb, dt, advection, visc) where {dim, N}

    DFloat = eltype(Q)
    Nq=(N+1)^dim
    (~, ~, nelem) = size(Q)

    #Store Velocity
    if (advection)

        #Allocate local arrays
        u=Array{DFloat,2}(undef,Nq,nelem)

        @inbounds for e = elems, i = 1:Nq
            u[i,e] = Q[i,_U,e] / ( Q[i,_h,e] + Q[i,_b,e] )
        end
    end

    @inbounds for e = elems, s = 1:_nstate-1, i = 1:Nq
        rhs[i, s, e] += visc*rhs_gradQ[i,s,e]
        Q[i, s, e] += rkb * dt * rhs[i, s, e] * vgeo[i, _MJI, e]
        rhs[i, s, e] *= rka
    end

    #Reset Velocity
    if (advection)
        @inbounds for e = elems, i = 1:Nq
            Q[i,_U,e] = ( Q[i,_h,e] + Q[i,_b,e] ) * u[i,e]
        end
    end
end
# }}}

# {{{ GPU kernels
# {{{ Volume RHS for 1D
@hascuda function knl_volumerhs!(::Val{1}, ::Val{N}, rhs, Q, vgeo, D, nelem, gravity, δnl) where N
    DFloat = eltype(D)
    Nq = N + 1

    #Point Thread to DOF and Block to element
    (i, j, k) = threadIdx()
    e = blockIdx().x

    #Allocate Arrays
    s_D = @cuStaticSharedMem(eltype(D), (Nq, Nq))
    s_F = @cuStaticSharedMem(eltype(Q), (Nq, _nstate))

    rhsU = rhsV = rhsh = zero(eltype(rhs))
    if i <= Nq && j == 1 && k == 1 && e <= nelem
        # Load derivative into shared memory
        if k == 1
            s_D[i, j] = D[i, j]
        end

        # Load values needed into registers
        MJ = vgeo[i, j, _MJ, e]
        ξx = vgeo[i, j, _ξx, e]
        h, b, U = Q[i, _h, e], Q[i, _b, e], Q[i, _U, e]
        rhsh, rhsU = rhs[i, _h, e], rhs[i, _U, e]

        #Get primitive variables and fluxes
        H=h + b
        u=U/H

        fluxh_x = U
        fluxU_x = (H * u * u + 0.5 * gravity * h^2) * δnl + gravity * h * b

        s_F[i, _h] = MJ * (ξx * fluxh_x)
        s_F[i, _U] = MJ * (ξx * fluxU_x)
    end

    sync_threads()

    @inbounds if i <= Nq && j == 1 && k == 1 && e <= nelem
        for n = 1:Nq
            #ξ-grid lines
            Dni = s_D[n, i]
            rhsh += Dni * s_F[n, _h]
            rhsU += Dni * s_F[n, _U]
        end
        rhs[i, _U, e] = rhsU
        rhs[i, _h, e] = rhsh
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

    @inbounds if i <= Nq && j == 1 && k == 1 && e <= nelem
        n = i + (j-1) * Nq
        for lf = 1:2:nface
            for f = lf:lf+1
                nxM, sMJ = sgeo[_nx, n, f, e], sgeo[_sMJ, n, f, e]
                (idM, idP) = (vmapM[n, f, e], vmapP[n, f, e])

                (eM, eP) = (e, ((idP - 1) ÷ Np) + 1)
                (vidM, vidP) = (((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1)

                hM = Q[vidM, _h, eM]
                bM = Q[vidM, _b, eM]
                UM = Q[vidM, _U, eM]

                bc = elemtobndy[f, e]
                hP = bP = UP = VP = zero(eltype(Q))
                if bc == 0
                    hP = Q[vidP, _h, eP]
                    bP = Q[vidP, _b, eM]
                    UP = Q[vidP, _U, eP]
                elseif bc == 1
                    UnM = nxM * UM
                    UP = UM - 2 * UnM * nxM
                    hP = hM
                    bP = bM
                    #              else
                    #                  error("Invalid boundary conditions $bc on face $f of element $e")
                end
                HM=hM + bM
                uM = UM / HM
                HP=hP + bP
                uP = UP / HP

                #Left Fluxes
                fluxhM_x = UM
                fluxUM_x = (HM * uM * uM + 0.5 * gravity * hM * hM) * δnl + gravity * hM * bM

                #Right Fluxes
                fluxhP_x = UP
                fluxUP_x = (HP * uP * uP + 0.5 * gravity * hP * hP) * δnl + gravity * hP * bP

                #Compute wave speed
                λM=( abs(nxM * uM) + CUDAnative.sqrt(gravity*HM) ) * δnl + ( CUDAnative.sqrt(gravity*bM) ) * (1.0-δnl)
                λP=( abs(nxM * uP) + CUDAnative.sqrt(gravity*HP) ) * δnl + ( CUDAnative.sqrt(gravity*bP) ) * (1.0-δnl)
              λ = max( λM, λP )

                #Compute Numerical Flux and Update
                fluxhS = (nxM * (fluxhM_x + fluxhP_x) - λ * (hP - hM)) / 2
                fluxUS = (nxM * (fluxUM_x + fluxUP_x) - λ * (UP - UM)) / 2

                #Update RHS
                rhs[vidM, _h, eM] -= sMJ * fluxhS
                rhs[vidM, _U, eM] -= sMJ * fluxUS
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

# {{{ Send Data
function senddata(::Val{dim}, ::Val{N}, mesh, sendreq, recvreq, sendQ,
                  recvQ, d_sendelems, d_sendQ, d_recvQ, d_QL, mpicomm;
                  ArrType=ArrType) where {dim, N}
    DFloat = eltype(d_QL)
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
    fillsendQ!(Val(dim), Val(N), sendQ, d_sendQ, d_QL, d_sendelems)

    # post MPI sends
    for n = 1:nnabr
        sendreq[n] = MPI.Isend((@view sendQ[:, :, mesh.nabrtosend[n]]),
                               mesh.nabrtorank[n], 777, mpicomm)
    end
end
# }}}

# {{{ Receive Data
function receivedata!(::Val{dim}, ::Val{N}, mesh, recvreq,
                      recvQ, d_recvQ, d_QL) where {dim, N}
    DFloat = eltype(d_QL)
    nrealelem = length(mesh.realelems)

    # wait on MPI receives
    MPI.Waitall!(recvreq)

    # copy data to state vector d_QL
    transferrecvQ!(Val(dim), Val(N), d_recvQ, recvQ, d_QL, nrealelem)

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
    d_gradQL = ArrType(Q)
    d_rhs_gradQL = ArrType(rhs)

    #Template Reshape Arrays
    Qshape    = (fill(N+1, dim)..., size(Q, 2), size(Q, 3))
    vgeoshape = (fill(N+1, dim)..., _nvgeo, size(Q, 3))

    #Reshape Device Arrays
    d_QC = reshape(d_QL, Qshape)
    d_rhsC = reshape(d_rhsL, Qshape...)
    d_vgeoC = reshape(d_vgeoL, vgeoshape)
    d_gradQC = reshape(d_gradQL, Qshape)
    d_rhs_gradQC = reshape(d_rhs_gradQL, Qshape)

    start_time = t1 = time_ns()
    for step = 1:nsteps
        for s = 1:length(RKA)

            #---------------1st Order Operators--------------------------#
            # Send Data
            senddata(Val(dim), Val(N), mesh, sendreq, recvreq, sendQ,
                     recvQ, d_sendelems, d_sendQ, d_recvQ, d_QL, mpicomm;
                     ArrType=ArrType)

            # volume RHS computation
            volumerhs!(Val(dim), Val(N), d_rhsC, d_QC, d_vgeoC, d_D, mesh.realelems, gravity, δnl)

            # Receive Data
            receivedata!(Val(dim), Val(N), mesh, recvreq, recvQ, d_recvQ, d_QL)

            # face RHS computation
            fluxrhs!(Val(dim), Val(N), d_rhsL, d_QL, d_sgeo, mesh.realelems, d_vmapM, d_vmapP, d_elemtobndy, gravity, δnl)

            #---------------2nd Order Operators--------------------------#
            if (visc > 0)
                # volume Q computation
                volumeQ!(Val(dim), Val(N), d_rhs_gradQC, d_QC, d_vgeoC, d_D, mesh.realelems)

                # face Q computation
                fluxQ!(Val(dim), Val(N), d_rhs_gradQL, d_QL, d_sgeo, mesh.realelems, d_vmapM, d_vmapP, d_elemtobndy)

                # Construct grad Q
                update_gradQ!(Val(dim), Val(N), d_gradQL, d_rhs_gradQL, d_vgeoL, mesh.realelems)

                # Send Data
                senddata(Val(dim), Val(N), mesh, sendreq, recvreq, sendQ,
                         recvQ, d_sendelems, d_sendQ, d_recvQ, d_gradQL,
                         mpicomm;ArrType=ArrType)

                # volume grad Q computation
                volumeQ!(Val(dim), Val(N), d_rhs_gradQC, d_gradQC, d_vgeoC, d_D, mesh.realelems)

                # Receive Data
                receivedata!(Val(dim), Val(N), mesh, recvreq, recvQ, d_recvQ, d_gradQL)

                # face grad Q computation
                fluxQ!(Val(dim), Val(N), d_rhs_gradQL, d_gradQL, d_sgeo, mesh.realelems, d_vmapM, d_vmapP, d_elemtobndy)
            end

            #---------------Update Solution--------------------------#
            # update solution and scale RHS
            updatesolution!(Val(dim), Val(N), d_rhsL, d_QL, d_rhs_gradQL, d_vgeoL, mesh.realelems,
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
        # TODO: Fix VTK for 1-D
        if plotstep > 0 && step % plotstep == 0
            Q .= d_QL
            X = ntuple(j->reshape((@view vgeo[:, _x+j-1, :]), ntuple(j->N+1,dim)...,
                                  nelem), dim)
            h = reshape((@view Q[:, _h, :]), ntuple(j->(N+1),dim)..., nelem)
            b = reshape((@view Q[:, _b, :]), ntuple(j->(N+1),dim)..., nelem)
            U = reshape((@view Q[:, _U, :]), ntuple(j->(N+1),dim)..., nelem) ./ (h.+b)
            writemesh(@sprintf("viz/swe%dD_%s_rank_%04d_step_%05d",
                               dim, ArrType, mpirank, step), X...;
                      fields=(("h", h),("b",b),("U",U)),realelems=mesh.realelems)
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
        x = vgeo[i, _x, e]
        h, b, U = ic(x)
        Q[i, _h, e] = h
        Q[i, _b, e] = b
        Q[i, _U, e] = U
        Qexact[i, _h, e] = h
        Qexact[i, _b, e] = b
        Qexact[i, _U, e] = U
    end

    # Compute time step
    mpirank == 0 && println("[CPU] computing dt (CPU)...")
    (base_dt,Courant) = courantnumber(Val(dim), Val(N), vgeo, Q, mpicomm, gravity, δnl, advection)
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
    writemesh(@sprintf("viz/swe%dD_%s_rank_%04d_step_%05d",
                       dim, ArrType, mpirank, 0), X...;
              fields=(("h", h),("b",b),("U",U)),realelems=mesh.realelems)

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
    writemesh(@sprintf("viz/swe%dD_%s_rank_%04d_step_%05d",
                       dim, ArrType, mpirank, nsteps), X...;
              fields=(("h", h),("b",b),("U",U)),realelems=mesh.realelems)

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
    if !Sys.iswindows()
        MPI.finalize_atexit()
    end

    mpicomm = MPI.COMM_WORLD
    mpirank = MPI.Comm_rank(mpicomm)
    mpisize = MPI.Comm_size(mpicomm)

    # FIXME: query via hostname
    @hascuda device!(mpirank % length(devices()))

    #Input Parameters
    N=8
    Ne=10
    visc=0.01
    iplot=10
    δnl=1
    icase=10
    time_final=DFloat(0.2)
    hardware="cpu"
    @show (N,Ne,visc,iplot,δnl,icase,time_final,hardware)

    #Initial Conditions
    if icase == 1 #advection
        function ic1(x...)
            r = sqrt( (x[1]-0.5)^2)
            h = 0.5 * exp(-100.0 * r^2)
            b = 1.0
            H = h + b
            U = H*(1.0)
            h, b, U
        end
        ic = ic1
        periodic = (true, )
        advection = true
        gravity = 0
    elseif icase == 10 #shallow water with Periodic BCs
        function ic10(x...)
            r = sqrt( (x[1]-0.5)^2 )
            h = 0.5 * exp(-100.0 * r^2)
            b=1.0
            H = h + b
            U = H*(0.0)
            h, b, U
        end
        ic = ic10
        periodic = (true, )
        advection = false
        gravity = 10
    elseif icase == 100 #shallow water with NFBC
        function ic100(x...)
            r = sqrt( (x[1]-0.5)^2 )
            h = 0.5 * exp(-100.0 * r^2)
            b=1.0
            H = h + b
            U = H*(0.0)
            h, b, U
        end
        ic = ic100
        periodic = (false, )
        advection = false
        gravity = 10
    end

    mesh = brickmesh((range(DFloat(0); length=Ne+1, stop=1),), periodic; part=mpirank+1, numparts=mpisize)

    if hardware == "cpu"
        mpirank == 0 && println("Running (CPU)...")
        swe(Val(1), Val(N), mpicomm, ic, mesh, time_final, gravity, δnl, advection, visc;
            ArrType=Array, tout = 10, plotstep = iplot)
        mpirank == 0 && println()
    elseif hardware == "gpu"
        @hascuda begin
            mpirank == 0 && println("Running (GPU)...")
            swe(Val(1), Val(N), mpicomm, ic, mesh, time_final, gravity, δnl, advection, visc;
                ArrType=CuArray, tout = 10, plotstep = iplot)
            mpirank == 0 && println()
        end
    end
    nothing
end
# }}}

main()
