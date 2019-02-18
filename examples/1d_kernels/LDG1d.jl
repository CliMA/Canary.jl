#--------------------------------Markdown Language Header-----------------------
# # 1D Diffusion Equation Example
#
#
#-
#
#-
# ## Introduction
#
# This example shows how to construct a 2nd derivative with DG using LDG.
#
# ## Continuous Governing Equations
# We discretize the operator:
# ```math
# \frac{\partial^2 q}{\partial x^2}  \; \; (1)
# ```
# in the following two-step process. First we discretize
# ```math
# Q = \frac{\partial q}{\partial x}  \; \; (2)
# ```
# followed by
# ```math
# \frac{\partial Q}{\partial x} =  \frac{\partial^2 q}{\partial x^2}  \; \; (3)
# ```
#-
# ## Local Discontinous Galerkin (LDG) Method
# Discretizing Eq. (2) we get
# ```math
# \int_{\Omega_e} \psi {Q}^{(e)}_N d\Omega_e = \int_{\Omega_e} \psi \frac{\partial q^{(e)}_N}{\partial x} d\Omega_e \; \; (4)
# ```
# where $q^{(e)}_N=\sum_{i=1}^{(N+1)^{dim}} \psi_i(\mathbf{x}) q_i$ and  ${Q}^{(e)}_N=\sum_{i=1}^{(N+1)^{dim}} \psi_i(\mathbf{x}) {Q}_i$ are the finite dimensional expansion with basis functions $\psi(x)$.
# Integrating Eq. (4) by parts yields
# ```math
# \int_{\Omega_e} \psi {Q}^{(e)}_N d\Omega_e = \left[ \psi q^{(*,e)}_N \right]_{\Gamma_e} - \int_{\Omega_e} \frac{\partial \psi}{\partial x} q^{(e)}_N d\Omega_e \; \; (5)
# ```
# where the first term on the right denotes the flux integral term (computed in "function fluxQ") and the second term on the right denotes the volume integral term (computed in "function volumeQ").  The superscript $(*,e)$ in the flux integral term denotes the numerical flux. Here we use the average flux. In matrix form, Eq. (5) becomes
# ```math
# M^{(e)}_{i,j} Q^{(e)}_j = {F}_{i,j} q^{(*,e)}_j - \widetilde{{D}}^{(e)} q^{(e)}_j \; \; (6)
# ```
#
# Next, integrating Eq. (3) by parts gives a similar form to Eq. (6) as follows
# ```math
# M^{(e)}_{i,j} \frac{\partial^2 q^{(e)}_j}{\partial x^2} = {F}_{i,j} Q^{(*,e)}_j - \widetilde{{D}}^{(e)} Q^{(e)}_j \; \; (7)
# ```
#
# Since we use the average flux for both $q$ and $Q$, we can reuse the same functions to construct ${F}$ and $\widetilde{{D}}$ in both Eqs.\ (6) and (7).  However, this will not be the case in multiple dimensions since Eq. (6) represents a gradient operator while Eq. (7) represents a divergence operator.
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
const _nstate = 2
const _U, _h = 1:_nstate
const stateid = (U = _U, h = _h)

const _nvgeo = 4
const _ξx, _MJ, _MJI, _x = 1:_nvgeo

const _nsgeo = 3
const _nx, _sMJ, _vMJI = 1:_nsgeo
# }}}

# {{{ cfl
function cfl(::Val{dim}, ::Val{N}, vgeo, Q, mpicomm) where {dim, N}
    DFloat = eltype(Q)
    Np = (N+1)^dim
    (~, ~, nelem) = size(Q)
    dt = [floatmax(DFloat)]
    Courant = - [floatmax(DFloat)]

    #Compute DT
    @inbounds for e = 1:nelem, n = 1:Np
        U = Q[n, _U, e]
        ξx = vgeo[n, _ξx, e]
        dx=1.0/(2*ξx)
        wave_speed = abs(U)
        loc_dt = 0.25*dx/wave_speed/N
        dt[1] = min(dt[1], loc_dt)
    end
    dt_min=MPI.Allreduce(dt[1], MPI.MIN, mpicomm)

    #Compute Courant
    @inbounds for e = 1:nelem, n = 1:Np
        U = Q[n, _U, e]
        ξx = vgeo[n, _ξx, e]
        dx=1.0/(2*ξx)
        wave_speed = abs(U)
        loc_Courant = wave_speed*dt[1]/dx*N
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
function volumerhs!(::Val{1}, ::Val{N}, rhs::Array, Q, vgeo, D, elems) where N
    DFloat = eltype(Q)
    Nq = N + 1
    nelem = size(Q)[end]

    Q = reshape(Q, Nq, _nstate, nelem)
    rhs = reshape(rhs, Nq, _nstate, nelem)
    vgeo = reshape(vgeo, Nq, _nvgeo, nelem)

    #Allocate Arrays
    s_F = Array{DFloat}(undef, Nq, _nstate)

    @inbounds for e in elems
        for i = 1:Nq
            MJ, ξx = vgeo[i,_MJ,e], vgeo[i,_ξx,e]
            U = Q[i, _U, e]

            #Get primitive variables
            U=U

            #Compute fluxes
            fluxU_x = 0.5*U^2
            s_F[i, _U] = MJ * (ξx * fluxU_x)
        end

        # loop of ξ-grid lines
        for s = 1:_nstate, i = 1:Nq, k = 1:Nq
            rhs[i, s, e] += D[k, i] * s_F[k, s]
        end
    end
end
# }}}

# Flux RHS for 1D
function fluxrhs!(::Val{1}, ::Val{N}, rhs::Array,  Q, sgeo, elems, vmapM, vmapP, elemtobndy) where N
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

                UM = Q[vidM, _U, eM]
                bc = elemtobndy[f, e]
                UP = zero(eltype(Q))
                if bc == 0
                    UP = Q[vidP, _U, eP]
                elseif bc == 1
                    UnM = nxM * UM
                    UP = UM - 2 * UnM * nxM
                else
                    error("Invalid boundary conditions $bc on face $f of element $e")
                end

                #Left Fluxes
                fluxUM_x = 0.5*UM^2

                #Right Fluxes
                fluxUP_x = 0.5*UP^2

                #Compute wave speed
                λM=abs(nxM * UM)
                λP=abs(nxM * UP)
                λ = max( λM, λP )

                #Compute Numerical/Rusanov Flux
                fluxUS = (nxM * (fluxUM_x + fluxUP_x) - λ * (UP - UM)) / 2

                #Update RHS
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
    s_F = Array{DFloat}(undef, Nq, _nstate)

    @inbounds for e in elems
        for i = 1:Nq
            MJ, ξx = vgeo[i,_MJ,e], vgeo[i,_ξx,e]
            U = Q[i, _U, e]

            #Get primitive variables
            U=U

            #Compute fluxes
            fluxU = U
            s_F[i, _U] = MJ * (ξx * fluxU)
        end

        # loop of ξ-grid lines
        for s = 1:_nstate, i = 1:Nq, k = 1:Nq
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

                UM = Q[vidM, _U, eM]
                bc = elemtobndy[f, e]
                UP = zero(eltype(Q))
                if bc == 0
                    UP = Q[vidP, _U, eP]
                elseif bc == 1
                    UnM = nxM * UM
                    UP = UM - 2 * UnM * nxM
                else
                    error("Invalid boundary conditions $bc on face $f of element $e")
                end

                #Left Fluxes
                fluxUM = UM

                #Right Fluxes
                fluxUP = UP

                #Compute Numerical/Rusanov Flux
                fluxUS = 0.5*(fluxUM + fluxUP)

                #Update RHS
                rhs[vidM, _U, eM] += sMJ * nxM * fluxUS
            end
        end
    end
end
# }}}

# {{{ Update solution
function updatesolution!(::Val{dim}, ::Val{N}, rhs, rhs_gradQ, Q, vgeo, elems, rka, rkb, dt, visc) where {dim, N}

    DFloat = eltype(Q)
    Nq=(N+1)^dim
    (~, ~, nelem) = size(Q)

    @inbounds for e = elems, s = 1:_nstate, i = 1:Nq
        rhs[i, s, e] += visc*rhs_gradQ[i,s,e]
        Q[i, s, e] += rkb * dt * rhs[i, s, e] * vgeo[i, _MJI, e]
        rhs[i, s, e] *= rka
    end

end
# }}}

# {{{ Update grad Q solution
function update_gradQ!(::Val{dim}, ::Val{N}, Q, rhs, vgeo, elems) where {dim, N}

    DFloat = eltype(Q)
    Nq=(N+1)^dim
    (~, ~, nelem) = size(Q)

    @inbounds for e = elems, s = 1:_nstate, i = 1:Nq
        Q[i, s, e] = rhs[i, s, e] * vgeo[i, _MJI, e] / π
    end

end
# }}}

# {{{ GPU kernels
# {{{ Volume RHS for 1D
@hascuda function knl_volumerhs!(::Val{1}, ::Val{N}, rhs, Q, vgeo, D, nelem) where N
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
        U = Q[i, _U, e]
        rhsU = rhs[i, _U, e]

        #Get primitive variables and fluxes
        U=U
        fluxU_x = 0.5*U^2
        s_F[i, _U] = MJ * (ξx * fluxU_x)
    end

    sync_threads()

    @inbounds if i <= Nq && j == 1 && k == 1 && e <= nelem
        for n = 1:Nq
            #ξ-grid lines
            Dni = s_D[n, i]
            rhsU += Dni * s_F[n, _U]
        end
        rhs[i, _U, e] = rhsU
    end
    nothing
end
# }}}

# {{{ Face RHS (all dimensions)
@hascuda function knl_fluxrhs!(::Val{dim}, ::Val{N}, rhs, Q, sgeo, nelem, vmapM, vmapP, elemtobndy) where {dim, N}
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

                UM = Q[vidM, _U, eM]

                bc = elemtobndy[f, e]
                UP = zero(eltype(Q))
                if bc == 0
                    UP = Q[vidP, _U, eP]
                elseif bc == 1
                    UnM = nxM * UM
                    UP = UM - 2 * UnM * nxM
                    #              else
                    #                  error("Invalid boundary conditions $bc on face $f of element $e")
                end

                #Left Fluxes
                fluxUM_x = 0.5*UM^2

                #Right Fluxes
                fluxUP_x = 0.5*UP^2

                #Compute wave speed
                λM=abs(nxM * uM)
                λP=abs(nxM * uP)
                λ = max( λM, λP )

                #Compute Numerical Flux and Update
                fluxUS = (nxM * (fluxUM_x + fluxUP_x) - λ * (UP - UM)) / 2

                #Update RHS
                rhs[vidM, _U, eM] -= sMJ * fluxUS
            end
            sync_threads()
        end
    end
    nothing
end
# }}}

# {{{ Update solution (for all dimensions)
@hascuda function knl_updatesolution!(::Val{dim}, ::Val{N}, rhs, Q, vgeo, nelem, rka, rkb, dt) where {dim, N}
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
@hascuda function volumerhs!(::Val{dim}, ::Val{N}, d_rhsC::CuArray, d_QC, d_vgeoC, d_D, elems) where {dim, N}
    nelem = length(elems)
    @cuda(threads=ntuple(j->N+1, dim), blocks=nelem,
          knl_volumerhs!(Val(dim), Val(N), d_rhsC, d_QC, d_vgeoC, d_D, nelem))
end

@hascuda function fluxrhs!(::Val{dim}, ::Val{N}, d_rhsL::CuArray, d_QL, d_sgeo, elems, d_vmapM, d_vmapP, d_elemtobndy) where {dim, N}
    nelem = length(elems)
    @cuda(threads=(ntuple(j->N+1, dim-1)..., 1), blocks=nelem,
          knl_fluxrhs!(Val(dim), Val(N), d_rhsL, d_QL, d_sgeo, nelem, d_vmapM, d_vmapP, d_elemtobndy))
end

@hascuda function updatesolution!(::Val{dim}, ::Val{N}, d_rhsL::CuArray, d_QL, d_vgeoL, elems, rka, rkb, dt) where {dim, N}
    nelem = length(elems)
    @cuda(threads=ntuple(j->N+1, dim), blocks=nelem,
          knl_updatesolution!(Val(dim), Val(N), d_rhsL, d_QL, d_vgeoL, nelem, rka, rkb, dt))
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
        diff = Q[i, _U, e] - Qex[i, _U, e]
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
                      dt, nsteps, tout, vmapM, vmapP, mpicomm, visc;
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
    d_gradQL = ArrType(Q)
    d_rhs_gradQL = ArrType(rhs)

    Qshape    = (fill(N+1, dim)..., size(Q, 2), size(Q, 3))
    vgeoshape = (fill(N+1, dim)..., _nvgeo, size(Q, 3))

    d_QC = reshape(d_QL, Qshape)
    d_rhsC = reshape(d_rhsL, Qshape...)
    d_vgeoC = reshape(d_vgeoL, vgeoshape)
    d_gradQC = reshape(d_gradQL, Qshape)
    d_rhs_gradQC = reshape(d_rhs_gradQL, Qshape)

    # Send Data
    senddata(Val(dim), Val(N), mesh, sendreq, recvreq, sendQ,
             recvQ, d_sendelems, d_sendQ, d_recvQ, d_QL, mpicomm;
             ArrType=ArrType)

    # Receive Data
    receivedata!(Val(dim), Val(N), mesh, recvreq, recvQ, d_recvQ, d_QL)

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

    # Construct grad Q
    update_gradQ!(Val(dim), Val(N), d_QL, d_rhs_gradQL, d_vgeoL, mesh.realelems)
    Q .= d_QL
    rhs .= d_rhsL
end
# }}}

# {{{ LDG driver
function LDG(::Val{dim}, ::Val{N}, mpicomm, ic, mesh, tend, visc;
             meshwarp=(x...)->identity(x), tout = 60, ArrType=Array,
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
        (U, Uexact) = ic(x)
        Q[i, _U, e] = U
        Qexact[i, _U, e] = Uexact
    end

    # Compute time step
    mpirank == 0 && println("[CPU] computing dt (CPU)...")
    (base_dt,Courant) = cfl(Val(dim), Val(N), vgeo, Q, mpicomm)
    base_dt=0.001  #FXG debug
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
    U = reshape((@view Q[:, _U, :]), ntuple(j->(N+1),dim)..., nelem)
    writemesh(@sprintf("viz/LDG%dD_%s_rank_%04d_step_%05d",
                       dim, ArrType, mpirank, 0), X...;
              fields=(("h", U),("U",U)),realelems=mesh.realelems)

    #Call Time-stepping Routine
    mpirank == 0 && println("[DEV] starting time stepper...")
    lowstorageRK(Val(dim), Val(N), mesh, vgeo, sgeo, Q, rhs, D, dt, nsteps, tout,
                 vmapM, vmapP, mpicomm, visc;
                 ArrType=ArrType, plotstep=plotstep)

    # plot final solution
    X = ntuple(j->reshape((@view vgeo[:, _x+j-1, :]), ntuple(j->N+1,dim)...,
                          nelem), dim)
    U = reshape((@view Q[:, _U, :]), ntuple(j->(N+1),dim)..., nelem)
    writemesh(@sprintf("viz/LDG%dD_%s_rank_%04d_step_%05d",
                       dim, ArrType, mpirank, nsteps), X...;
              fields=(("h", U),("U",U)),realelems=mesh.realelems)

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
    N=4
    Ne=10
    visc=0.01
    iplot=10
    time_final=DFloat(1.0)
    hardware="cpu"
    @show (N,Ne,visc,iplot,time_final,hardware)

    #Initial Conditions
    function ic(x...)
        U = sin( π*x[1] )
        Uexact = - sin( π*x[1] ) #2nd derivative
        (U, Uexact)
    end
    periodic = (true, )

    mesh = brickmesh((range(DFloat(0); length=Ne+1, stop=2),), periodic; part=mpirank+1, numparts=mpisize)

    if hardware == "cpu"
        mpirank == 0 && println("Running (CPU)...")
        LDG(Val(1), Val(N), mpicomm, ic, mesh, time_final, visc;
            ArrType=Array, tout = 10, plotstep = iplot)
        mpirank == 0 && println()
    elseif hardware == "gpu"
        @hascuda begin
            mpirank == 0 && println("Running (GPU)...")
            LDG(Val(1), Val(N), mpicomm, ic, mesh, time_final, visc;
                ArrType=CuArray, tout = 10, plotstep = iplot)
            mpirank == 0 && println()
        end
    end
    nothing
end
# }}}

main()
