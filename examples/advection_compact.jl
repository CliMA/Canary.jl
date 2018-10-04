include("vtk.jl")
using MPI
using Canary
using Printf: @sprintf
const DFloat = Float64

# Fourth-order, low-storage, Runge–Kutta scheme of Carpenter and Kennedy (1994)
# ((5,4) 2N-Storage RK scheme.
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
const RKA = (DFloat(0),
             DFloat(-567301805773)  / DFloat(1357537059087),
             DFloat(-2404267990393) / DFloat(2016746695238),
             DFloat(-3550918686646) / DFloat(2091501179385),
             DFloat(-1275806237668) / DFloat(842570457699 ))

const RKB = (DFloat(1432997174477) / DFloat(9575080441755 ),
             DFloat(5161836677717) / DFloat(13612068292357),
             DFloat(1720146321549) / DFloat(2090206949498 ),
             DFloat(3134564353537) / DFloat(4481467310338 ),
             DFloat(2277821191437) / DFloat(14882151754819))

const RKC = (DFloat(0),
             DFloat(1432997174477) / DFloat(9575080441755),
             DFloat(2526269341429) / DFloat(6820363962896),
             DFloat(2006345519317) / DFloat(3224310063776),
             DFloat(2802321613138) / DFloat(2924317926251))


# We now initialize MPI as well as get the communicator, rank, and size
MPI.Initialized() || MPI.Init() # only initialize MPI if not initialized
MPI.finalize_atexit()
const mpicomm = MPI.COMM_WORLD
const mpirank = MPI.Comm_rank(mpicomm)
const mpisize = MPI.Comm_size(mpicomm)

function createmesh(brickN::NTuple{dim, Int}) where dim

  # Generate an local view of a fully periodic Cartesian mesh.
  mesh = brickmesh(ntuple(i->range(DFloat(0); length=brickN[i]+1, stop=1), dim),
                   (fill(true, dim)...,);
                   part=mpirank+1, numparts=mpisize)

  # Partion the mesh using a Hilbert curve based partitioning
  mesh = partition(mpicomm, mesh...)

  # Connect the mesh in parallel
  mesh = connectmesh(mpicomm, mesh...)

  mesh
end

# TODO: Can we clean this up?
function cfl(dim, metric, Q)
  dt = [floatmax(DFloat)]
  if dim == 1
    ξx = metric.ξx
    Ux = Q.Ux
    for n = 1:length(Ux)
      loc_dt = 2 ./ abs.(Ux[n] * ξx[n])
      dt[1] = min(dt[1], loc_dt)
    end
  elseif dim == 2
    (ξx, ξy, ηx, ηy) = (metric.ξx, metric.ξy, metric.ηx, metric.ηy)
    (Ux, Uy) = (Q.Ux, Q.Uy)
    for n = 1:length(Ux)
      loc_dt = 2 ./ max(abs.(Ux[n] * ξx[n] + Uy[n] * ξy[n]),
                        abs.(Ux[n] * ηx[n] + Uy[n] * ηy[n]))
      dt[1] = min(dt[1], loc_dt)
    end
  elseif dim == 3
    (ξx, ξy, ξz) = (metric.ξx, metric.ξy, metric.ξz)
    (ηx, ηy, ηz) = (metric.ηx, metric.ηy, metric.ηz)
    (ζx, ζy, ζz) = (metric.ζx, metric.ζy, metric.ζz)
    (Ux, Uy, Uz) = (Q.Ux, Q.Uy, Q.Uz)
    for n = 1:length(Ux)
      loc_dt = 2 ./ max(abs.(Ux[n] * ξx[n] + Uy[n] * ξy[n] + Uz[n] * ξz[n]),
                        abs.(Ux[n] * ηx[n] + Uy[n] * ηy[n] + Uz[n] * ηz[n]),
                        abs.(Ux[n] * ζx[n] + Uy[n] * ζy[n] + Uz[n] * ζz[n]))
      dt[1] = min(dt[1], loc_dt)
    end
  end
  dt = MPI.Allreduce(dt[1], MPI.MIN, mpicomm)

  dt
end

function computegeometry(dim, mesh, D, ξ, ω, meshwarp)
  # Compute metric terms
  (nface, nelem) = size(mesh.elemtoelem)
  crd = creategrid(Val(dim), mesh.elemtocoord, ξ)

  # skew the mesh
  dim == 1 && for j = 1:length(crd.x)
    crd.x[j] = meshwarp(crd.x[j])[1]
  end
  dim == 2 && for j = 1:length(crd.x)
    (crd.x[j], crd.y[j]) = meshwarp(crd.x[j], crd.y[j])
  end
  dim == 3 && for j = 1:length(crd.x)
    (crd.x[j], crd.y[j], crd.z[j]) = meshwarp(crd.x[j], crd.y[j], crd.z[j])
  end

  # Compute the metric terms
  metric = computemetric(crd..., D)

  # TODO: scale metric terms with ω (J, sJ, etc.) and compute (scaled) JI

  M = reshape(kron(1, fill(ω, dim)...), fill(length(ω), dim)...)
  MJ = M .* metric.J
  MJI = 1 ./ MJ

  metric = NamedTuple{(keys(metric)..., :MJ, :MJI)}((metric..., MJ, MJI))

  (crd, metric)
end

# Volume RHS for 1-D
# TODO: Clean up!
# TODO: Optimize
function volumerhs!(::Val{N}, rhs, (ρ, Ux)::NamedTuple{(:ρ, :Ux)}, metric, D, ω,
                    elems) where N
  rhsρ = rhs.ρ
  (J, ξx) = (metric.J, metric.ξx)
  Nq = N + 1
  # for each element
  @inbounds for e ∈ elems
    # rhsρ[:, e] += D' * (ω .* J[:, e] .* ξx[:, e] .* Ux[:, e] .* ρ[:, e])
    for i ∈ 1:Nq
      for n ∈ 1:Nq
        rhsρ[i, e] += D[n, i] * (ω[n] * J[n, e] * ξx[n, e] * Ux[n, e] * ρ[n, e])
      end
    end
  end
end

# Face RHS for 1-D
# TODO: Clean up!
# TODO: Optimize
function facerhs!(::Val{N}, rhs, (ρ, Ux)::NamedTuple{(:ρ, :Ux)}, metric, ω,
                  elems, vmapM, vmapP) where N
  rhsρ = rhs.ρ
  nface = 2
  (nx, sJ) = (metric.nx, metric.sJ)
  @inbounds @simd for e ∈ elems
    for f ∈ 1:nface
      ρM = ρ[vmapM[1, f, e]]
      UxM = Ux[vmapM[1, f, e]]
      FxM = ρM * UxM

      ρP = ρ[vmapP[1, f, e]]
      UxP = Ux[vmapP[1, f, e]]
      FxP = ρP * UxP

      nxM = nx[1, f, e]
      λ = max(abs(nxM * UxM), abs(nxM * UxP))

      F = (nxM * (FxM + FxP) + λ * (ρM - ρP)) / 2
      rhsρ[vmapM[1, f, e]] -= sJ[1, f, e] * F
    end
  end
end

# Volume RHS for 2-D
# TODO: Clean up!
# TODO: Optimize
function volumerhs!(::Val{N}, rhs, (ρ, Ux, Uy)::NamedTuple{(:ρ, :Ux, :Uy)},
                    metric, D, ω, elems) where N
  rhsρ = rhs.ρ
  J = metric.J
  (ξx, ηx, ξy, ηy) = (metric.ξx, metric.ηx, metric.ξy, metric.ηy)
  Nq = N + 1
  # for each element
  @inbounds for e ∈ elems
    # loop of ξ-grid lines
    for j = 1:Nq
      #=
      rhsρ[:, j, e] +=
        D' * (ω[j] * ω .* J[:, j, e].* ρ[:, j, e] .*
              (ξx[:, j, e] .* Ux[:, j, e] + ξy[:, j, e] .* Uy[:, j, e]))
      =#
      for i = 1:Nq
        for n = 1:Nq
          rhsρ[i, j, e] +=
            D[n, i] * (ω[j] * ω[n] .* J[n, j, e].* ρ[n, j, e] .*
                    (ξx[n, j, e] .* Ux[n, j, e] + ξy[n, j, e] .* Uy[n, j, e]))
        end
      end
    end
    # loop of η-grid lines
    for i = 1:Nq
      #=
      rhsρ[i, :, e] +=
        D' * (ω[i] * ω .* J[i, :, e].* ρ[i, :, e] .*
              (ηx[i, :, e] .* Ux[i, :, e] + ηy[i, :, e] .* Uy[i, :, e]))
      =#
      for j = 1:Nq
        for n = 1:Nq
          rhsρ[i, j, e] +=
            D[n, j] * (ω[i] * ω[n] .* J[i, n, e].* ρ[i, n, e] .*
                    (ηx[i, n, e] .* Ux[i, n, e] + ηy[i, n, e] .* Uy[i, n, e]))
        end
      end
    end
  end
end

# Face RHS for 2-D
# TODO: Clean up!
# TODO: Optimize
function facerhs!(::Val{N}, rhs, (ρ, Ux, Uy)::NamedTuple{(:ρ, :Ux, :Uy)},
                  metric, ω, elems, vmapM, vmapP) where N
  rhsρ = rhs.ρ
  nface = 4
  (nx, ny, sJ) = (metric.nx, metric.ny, metric.sJ)
  Nfp = N+1
  @inbounds @simd for e ∈ elems
    for f ∈ 1:nface
      for n ∈ 1:Nfp
        ρM = ρ[vmapM[n, f, e]]
        UxM = Ux[vmapM[n, f, e]]
        UyM = Uy[vmapM[n, f, e]]
        FxM = ρM * UxM
        FyM = ρM * UyM

        ρP = ρ[vmapP[n, f, e]]
        UxP = Ux[vmapP[n, f, e]]
        UyP = Uy[vmapP[n, f, e]]
        FxP = ρP * UxP
        FyP = ρP * UyP

        nxM = nx[n, f, e]
        nyM = ny[n, f, e]
        λ = max(abs(nxM * UxM + nyM * UyM), abs(nxM * UxP + nyM * UyP))

        F = (nxM * (FxM + FxP) + nyM * (FyM + FyP) + λ * (ρM - ρP)) / 2
        rhsρ[vmapM[n, f, e]] -= ω[n] * sJ[n, f, e] * F
      end
    end
  end
end

# Volume RHS for 3-D
# TODO: Clean up!
# TODO: Optimize
function volumerhs!(::Val{N}, rhs,
                    (ρ, Ux, Uy, Uz)::NamedTuple{(:ρ, :Ux, :Uy, :Uz)}, metric, D,
                    ω, elems) where N
  rhsρ = rhs.ρ
  J = metric.J
  (ξx, ηx, ζx) = (metric.ξx, metric.ηx, metric.ζx)
  (ξy, ηy, ζy) = (metric.ξy, metric.ηy, metric.ζy)
  (ξz, ηz, ζz) = (metric.ξz, metric.ηz, metric.ζz)
  Nq = N + 1
  @inbounds for e ∈ elems
    # loop of ξ-grid lines
    for k = 1:Nq
      for j = 1:Nq
        #=
        rhsρ[:, j, k, e] +=
          D' * (ω[j] * ω[k] * ω .* J[:, j, k, e] .* ρ[:, j, k, e] .*
                (ξx[:, j, k, e] .* Ux[:, j, k, e] +
                 ξy[:, j, k, e] .* Uy[:, j, k, e] +
                 ξz[:, j, k, e] .* Uz[:, j, k, e]))
        =#
        for i = 1:Nq
          for n = 1:Nq
            rhsρ[i, j, k, e] +=
              D[n, i] * (ω[n] * ω[j] * ω[k] * J[n, j, k, e] * ρ[n, j, k, e] *
                        (ξx[n, j, k, e] * Ux[n, j, k, e] +
                         ξy[n, j, k, e] * Uy[n, j, k, e] +
                         ξz[n, j, k, e] * Uz[n, j, k, e]))
          end
        end
      end
    end
    # loop of η-grid lines
    for k = 1:Nq
      for i = 1:Nq
        #=
        rhsρ[i, :, k, e] +=
          D' * (ω[i] * ω[k] * ω .* J[i, :, k, e] .* ρ[i, :, k, e] .*
                (ηx[i, :, k, e] .* Ux[i, :, k, e] +
                 ηy[i, :, k, e] .* Uy[i, :, k, e] +
                 ηz[i, :, k, e] .* Uz[i, :, k, e]))
        =#
        for j = 1:Nq
          for n = 1:Nq
            rhsρ[i, j, k, e] +=
              D[n, j] * (ω[i] * ω[n] * ω[k] * J[i, n, k, e] * ρ[i, n, k, e] *
                        (ηx[i, n, k, e] * Ux[i, n, k, e] +
                         ηy[i, n, k, e] * Uy[i, n, k, e] +
                         ηz[i, n, k, e] * Uz[i, n, k, e]))
          end
        end
      end
    end
    # loop of ζ-grid lines
    for j = 1:Nq
      for i = 1:Nq
        #=
        rhsρ[i, j, :, e] +=
          D' * (ω[i] * ω[j] * ω .* J[i, j, :, e] .* ρ[i, j, :, e] .*
                (ζx[i, j, :, e] .* Ux[i, j, :, e] +
                 ζy[i, j, :, e] .* Uy[i, j, :, e] +
                 ζz[i, j, :, e] .* Uz[i, j, :, e]))
        =#
        for k = 1:Nq
          for n = 1:Nq
            rhsρ[i, j, k, e] +=
              D[n, k] * (ω[i] * ω[j] * ω[n] * J[i, j, n, e] * ρ[i, j, n, e] *
                         (ζx[i, j, n, e] * Ux[i, j, n, e] +
                          ζy[i, j, n, e] * Uy[i, j, n, e] +
                          ζz[i, j, n, e] * Uz[i, j, n, e]))
          end
        end
      end
    end
  end
end

# Face RHS for 3-D
# TODO: Clean up!
# TODO: Optimize
function facerhs!(::Val{N}, rhs,
                  (ρ, Ux, Uy, Uz)::NamedTuple{(:ρ, :Ux, :Uy, :Uz)}, metric, ω,
                  elems, vmapM, vmapP) where N
  rhsρ = rhs.ρ
  nface = 6
  (nx, ny, nz, sJ) = (metric.nx, metric.ny, metric.nz, metric.sJ)
  ω2 = kron(ω, ω)
  Nfp = (N+1) * (N+1)
  @inbounds @simd for e ∈ elems
    for f ∈ 1:nface
      for n ∈ 1:Nfp
        ρM = ρ[vmapM[n, f, e]]
        UxM = Ux[vmapM[n, f, e]]
        UyM = Uy[vmapM[n, f, e]]
        UzM = Uz[vmapM[n, f, e]]
        FxM = ρM * UxM
        FyM = ρM * UyM
        FzM = ρM * UzM

        ρP = ρ[vmapP[n, f, e]]
        UxP = Ux[vmapP[n, f, e]]
        UyP = Uy[vmapP[n, f, e]]
        UzP = Uz[vmapP[n, f, e]]
        FxP = ρP * UxP
        FyP = ρP * UyP
        FzP = ρP * UzP

        nxM = nx[n, f, e]
        nyM = ny[n, f, e]
        nzM = nz[n, f, e]
        λ = max(abs(nxM * UxM + nyM * UyM + nzM * UzM),
                abs(nxM * UxP + nyM * UyP + nzM * UzP))

        F = (nxM * (FxM + FxP) + nyM * (FyM + FyP) + nzM * (FzM + FzP) +
             λ * (ρM - ρP)) / 2
        rhsρ[vmapM[n, f, e]] -= ω2[n] * sJ[n, f, e] * F
      end
    end
  end
end

# Update solution (for all dimensions)
function updatesolution!(::Val{dim}, ::Val{N}, rhs, Q, metric, ω, elems, rka,
                         rkb, dt) where {dim, N}
  MJI = metric.MJI
  ind = CartesianIndices(ntuple(j->1:N+1, Val(dim)))
  @inbounds for (rhsq, q) ∈ zip(rhs, Q)
    for e ∈ elems
      @simd for i ∈ ind
        q[i, e] += rkb * dt * rhsq[i, e] * MJI[i, e]
        rhsq[i, e] *= rka
      end
    end
  end
end

# L2 Error (for all dimensions)
# TODO: Optimize
function L2energysquared(::Val{dim}, Q, metric, ω, elems) where dim
  MJ = metric.MJ
  Nq = size(MJ, 1)
  energy = [DFloat(0)]
  ind = CartesianIndices(ntuple(j->1:Nq, Val(dim)))
  for q ∈ Q
    for e ∈ elems
      for i ∈ ind
        energy[1] += MJ[i, e] * q[i, e]^2
      end
    end
  end
  energy[1]
end

function lowstorageRK(dim, mesh, metric, Q, rhs, D, ω, dt, nsteps, tout, vmapM,
                      vmapP)
  # TODO: Think about output?

  # Exact polynomial order
  Nq = size(D, 2)
  N = Nq - 1

  # Create send and recv request array
  nmpinabr = length(mesh.nabrtorank)
  sendreq = fill(MPI.REQUEST_NULL, nmpinabr)
  recvreq = fill(MPI.REQUEST_NULL, nmpinabr)

  # Create send and recv buffer
  sendQ = Array{DFloat, 3}(undef, (N+1)^dim, length(Q), length(mesh.sendelems))
  recvQ = Array{DFloat, 3}(undef, (N+1)^dim, length(Q), length(mesh.ghostelems))

  # Index map for MPI communication
  index = CartesianIndices(ntuple(j->1:N+1, dim))
  nrealelem = length(mesh.realelems)

  t1 = time_ns()
  for step = 1:nsteps
    if mpirank == 0 && (time_ns() - t1)*1e-9 > tout
      t1 = time_ns()
      @show (step, nsteps)
    end
    for s = 1:length(RKA)
      # post MPI receives
      map!(recvreq, mesh.nabrtorank, mesh.nabrtorecv) do nabrrank, nabrelem
        MPI.Irecv!((@view recvQ[:, :, nabrelem]), nabrrank, 777, mpicomm)
      end

      # wait on (prior) MPI sends
      MPI.Waitall!(sendreq)

      # pack data in send buffer
      for (ne, e) ∈ enumerate(mesh.sendelems)
        for (nf, f) ∈ enumerate(Q)
          sendQ[:, nf, ne] = f[index[:], e]
        end
      end

      # post MPI sends
      map!(sendreq, mesh.nabrtorank, mesh.nabrtosend) do nabrrank, nabrelem
        MPI.Isend((@view sendQ[:, :, nabrelem]), nabrrank, 777, mpicomm)
      end

      # volume RHS computation
      volumerhs!(Val(N), rhs, Q, metric, D, ω, mesh.realelems)

      # wait on MPI receives
      MPI.Waitall!(recvreq)

      # copy data to state vectors
      for elems ∈ mesh.nabrtorecv
        for (nf, f) ∈ enumerate(Q)
          f[index[:], nrealelem .+ elems] = recvQ[:, nf, elems]
        end
      end

      # face RHS computation
      facerhs!(Val(N), rhs, Q, metric, ω, mesh.realelems, vmapM, vmapP)

      # update solution and scale RHS
      updatesolution!(Val(dim), Val(N), rhs, Q, metric, ω, mesh.realelems,
                      RKA[s%length(RKA)+1], RKB[s], dt)
    end
  end
end

function main(ic, N, brickN::NTuple{dim, Int}, tend;
              meshwarp=(x...)->identity(x),
              tout = 1) where dim

  mesh = createmesh(brickN)

  # Get the vmaps
  (vmapM, vmapP) = mappings(N, mesh.elemtoelem, mesh.elemtoface,
                            mesh.elemtoordr)

  # Create 1-D operators
  (ξ, ω) = lglpoints(DFloat, N)
  D = spectralderivative(ξ)

  # Compute the geometry
  (coord, metric) = computegeometry(dim, mesh, D, ξ, ω, meshwarp)

  # Storage for the solution, rhs, and error
  statesyms = (:ρ, (:Ux, :Uy, :Uz)[1:dim]...)
  Q   = NamedTuple{statesyms}(ntuple(j->zero(coord.x), length(statesyms)))
  rhs = NamedTuple{statesyms}(ntuple(j->zero(coord.x), length(statesyms)))
  Δ   = NamedTuple{statesyms}(ntuple(j->zero(coord.x), length(statesyms)))

  # setup the initial condition
  foreach(f->map!(ic[f], Q[f], coord...), keys(Q))

  # plot the initial condition
  mkpath("viz")
  # TODO: Fix VTK for 1-D
  dim > 1 && writemesh(@sprintf("viz/advection%dD_rank_%04d_step_%05d", dim,
                                mpirank, 0), coord...; fields=(("ρ", Q.ρ),),
                       realelems=mesh.realelems)

  # Compute time step
  dt = DFloat(cfl(dim, metric, Q) / N^√2)

  tend = DFloat(tend)
  nsteps = ceil(Int64, tend / dt)
  dt = tend / nsteps
  @show (dt, nsteps)

  # Do time stepping
  eng = [DFloat(0),  DFloat(0)]
  eng[1] = √MPI.Allreduce(L2energysquared(Val(dim), Q, metric, ω,
                                          mesh.realelems), MPI.SUM, mpicomm)
  lowstorageRK(dim, mesh, metric, Q, rhs, D, ω, dt, nsteps, tout, vmapM, vmapP)
  eng[2] = √MPI.Allreduce(L2energysquared(Val(dim), Q, metric, ω,
                                          mesh.realelems), MPI.SUM, mpicomm)
  mpirank == 0 && @show eng
  mpirank == 0 && @show diff(eng)

  # Compute the error in ρ
  map!(Δ.ρ, 1:length(coord.x), coord... ) do i, x...
    Q.ρ[i] - ic.ρ(ntuple(j -> x[j] - Q[(:Ux, :Uy, :Uz)[j]][i] * tend,
                         Val(dim))...)
  end
  err = √MPI.Allreduce(L2energysquared(Val(dim), Δ, metric, ω, mesh.realelems),
                       MPI.SUM, mpicomm)
  mpirank == 0 && @show err
end

warping1D(x) = (x + sin(π * x)/10,)
warping2D(x, y) = (x + sin(π * x) * sin(2 * π * y) / 10,
                   y + sin(2 * π * x) * sin(π * y) / 10)
warping3D(x, y, z) = (x + (sin(π * x) * sin(2 * π * y) * cos(2 * π * z)) / 10,
                      y + (sin(π * y) * sin(2 * π * x) * cos(2 * π * z)) / 10,
                      z + (sin(π * z) * sin(2 * π * x) * cos(2 * π * y)) / 10)

ρ1D(x) = sin(2 * π * x)
ρ2D(x, y) = sin(2 * π * x) * sin(2 * π * y)
ρ3D(x, y, z) = sin(2 * π * x) * sin(2 * π * y) * sin(2 * π * z)

Ux(x...) = -1.5
Uy(x...) = -π
Uz(x...) =  exp(1)

mpirank == 0 && println("Running 1d...")
main((ρ=ρ1D, Ux=Ux), 5, (3, ), π; meshwarp=warping1D)
mpirank == 0 && println()

mpirank == 0 && println("Running 2d...")
main((ρ=ρ2D, Ux=Ux, Uy=Uy), 5, (3, 3), π; meshwarp=warping2D)
mpirank == 0 && println()

mpirank == 0 && println("Running 3d...")
main((ρ=ρ3D, Ux=Ux, Uy=Uy, Uz=Uz), 5, (3, 3, 3), π; meshwarp=warping3D)

nothing
