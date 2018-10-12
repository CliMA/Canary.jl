#--------------------------------Markdown Language Header-----------------------
# # Euler Equations Example
#
#
# ![](shallow_water2d.png)
#-
#
#-
# ## Introduction
#
# This example shows how to solve the Euler Equations in 1D, 2D, and 3D.
#
# ## Continuous Governing Equations
# We solve the following equation:
#
# ```math
# \frac{\partial \rho}{\partial t} + \nabla \cdot \mathbf{U} = 0 \; \; (1.1)
# ```
# ```math
# \frac{\partial \mathbf{U}}{\partial t} + \nabla \cdot \left( \frac{\mathbf{U} \otimes \mathbf{U}}{\rho} + P \mathbf{I}_2 \right) + \rho g \mathbf{r} = 0 \; \; (1.2)
# ```
# ```math
# \frac{\partial \mathbf{\Theta}}{\partial t} + \nabla \cdot \left( \frac{\Theta \mathbf{U}}{\rho} = 0 \; \; (1.3)
# ```
# where $\mathbf{u}=(u,v,w)$ depending on how many spatial dimensions we are using, and $\mathbf{U}=\rho \mathbf{u}$, $P=P_A \left( \frac{R \Theta}{P_A} \right)^{\gamma}$ is the pressure, $\Theta=\rho \theta$ is the density potential temperature and $\theta=\frac{T}{\pi}$ is the potential temperature where $T$ is temperature and $\pi=\left( \frac{P}{P_A} \right)^{\kappa}$ is the exner function where $P_A$ is atmospheric (sea level) pressure.
# We employ periodic boundary conditions across all four walls of the square domain.
#
#-
# ## Discontinous Galerkin Method
# To solve Eq. (1) in one, two, and three dimensions we use the Discontinuous Galerkin method with basis functions comprised of tensor products
# of one-dimensional Lagrange polynomials based on Lobatto points. Multiplying Eq. (1) by a test function $\psi$ and integrating within each element $\Omega_e$ such that $\Omega = \bigcup_{e=1}^{N_e} \Omega_e$ we get
#
# ```math
# \int_{\Omega_e} \psi \frac{\partial \mathbf{q}^{(e)}_N}{\partial t} d\Omega_e + \int_{\Omega_e} \psi \nabla \cdot \mathbf{f}^{(e)}_N d\Omega_e =  \int_{\Omega_e} \psi S\left( q^{(e)}_N} \right) d\Omega_e \; \; (2)
# ```
# where $\mathbf{q}^{(e)}_N=\sum_{i=1}^{(N+1)^{dim}} \psi_i(\mathbf{x}) \mathbf{q}_i(t)$ is the finite dimensional expansion with basis functions $\psi(\mathbf{x})$, where $\mathbf{q}=\left( h, \mathbf{U}^T \right)^T$ and
#```math
# \mathbf{f}=\left( \mathbf{U}, \frac{\mathbf{U} \otimes \mathbf{U}}{\rho} + P \mathbf{I}_3, \frac{\Theta \mathbf{U}}{\rho} \right).
# ```
# Integrating Eq. (2) by parts yields
#
# ```math
# \int_{\Omega_e} \psi \frac{\partial \mathbf{q}^{(e)}_N}{\partial t} d\Omega_e + \int_{\Gamma_e} \psi \mathbf{n} \cdot \mathbf{f}^{(*,e)}_N d\Gamma_e - \int_{\Omega_e} \nabla \psi \cdot \mathbf{f}^{(e)}_N d\Omega_e = \int_{\Omega_e} \psi S\left( q^{(e)}_N} \right) d\Omega_e \; \; (3)
# ```
#
# where the second term on the left denotes the flux integral term (computed in "function fluxrhs") and the third term denotes the volume integral term (computed in "function volumerhs").  The superscript $(*,e)$ in the flux integral term denotes the numerical flux. Here we use the Rusanov flux.
#
#-
# ## Commented Program
#
#--------------------------------Markdown Language Header-----------------------

# ### Define Input Parameters:
# N is polynomial order and
# brickN(Ne) generates a brick-grid with Ne elements in each direction
N=4 #polynomial order
#brickN=(10) #1D brickmesh
brickN=(10, 10) #2D brickmesh
#brickN=(10, 1, 10) #3D brickmesh
DFloat=Float64 #Number Type
tend=DFloat(5.0) #Final Time
gravity=10.0 #gravity
R_gas=287.17
c_p=1004.67
c_v=717.5
p0=100000.0
γ=1.4 #specific heat ratio cp/cp for air
icase=6 #1=advection; 2=Translating Gaussian; 3=Still Gaussian (Periodic); 4=Still Gaussian (NFBC); 5=Shock tube (NFBC); 6=RTB
iplot=50 #every how many time-steps to plot
warp_grid=false #warp initial grid
u0=100.0 #Initial velocity

# ### Load the MPI and Canary packages where Canary builds the mesh, generates basis functions, and metric terms.
using MPI
using Canary
using Printf: @sprintf

# ### The grid that we create determines the number of spatial dimensions that we are going to use.
dim = length(brickN)

# ###Output the polynomial order, space dimensions, and element configuration
println("N= ",N)
println("dim= ",dim)
println("Ne= ",brickN)
println("DFloat= ",DFloat)
println("gravity= ",gravity)
println("icase= ",icase)
println("u0= ",u0)
println("Time Final= ",tend)
println("warp_grid= ",warp_grid)
println("iplot= ",iplot)

# ### Initialize MPI and get the communicator, rank, and size
MPI.Initialized() || MPI.Init() # only initialize MPI if not initialized
MPI.finalize_atexit()
mpicomm = MPI.COMM_WORLD
mpirank = MPI.Comm_rank(mpicomm)
mpisize = MPI.Comm_size(mpicomm)

# ### Generate a local view of a fully periodic Cartesian mesh.
if dim == 1
  (Nx, ) = brickN
  local x = range(DFloat(0); length=Nx+1, stop=1)
  if icase <= 3
      mesh = brickmesh((x, ), (true, ); part=mpirank+1, numparts=mpisize)
  elseif icase >= 4
      mesh = brickmesh((x, ), (false, ); part=mpirank+1, numparts=mpisize)
  end
elseif dim == 2
  (Nx, Ny) = brickN
  local x = range(DFloat(0); length=Nx+1, stop=1)
  local y = range(DFloat(0); length=Ny+1, stop=1)
  if icase <= 3
      mesh = brickmesh((x, y), (true, true); part=mpirank+1, numparts=mpisize)
  elseif icase >= 4 && icase < 6
      mesh = brickmesh((x, y), (false, false); part=mpirank+1, numparts=mpisize)
  elseif icase >= 6
      mesh = brickmesh((x, y), (true, false); part=mpirank+1, numparts=mpisize)
  end
elseif dim == 3
  (Nx, Ny, Nz) = brickN
  local x = range(DFloat(0); length=Nx+1, stop=1)
  local y = range(DFloat(0); length=Ny+1, stop=1)
  local z = range(DFloat(0); length=Nz+1, stop=1)
  if icase <= 3
      mesh = brickmesh((x, y, z), (true, true, true); part=mpirank+1, numparts=mpisize)
  elseif icase >= 4 && icase < 6
      mesh = brickmesh((x, y, z), (false, false, false); part=mpirank+1, numparts=mpisize)
  elseif icase >= 6
      mesh = brickmesh((x, y, z), (true, true, false); part=mpirank+1, numparts=mpisize)
  end
end

# ### Partition the mesh using a Hilbert curve based partitioning
mesh = partition(mpicomm, mesh...)

# ### Connect the mesh in parallel
mesh = connectmesh(mpicomm, mesh...)

# ### Get the degrees of freedom along the faces of each element.
# vmap(:,f,e) gives the list of local (mpirank) points for the face "f" of element "e".  vmapP points to the outward (or neighbor) element and vmapM for the current element. P=+ or right and M=- or left.
(vmapM, vmapP) = mappings(N, mesh.elemtoelem, mesh.elemtoface, mesh.elemtoordr)

# ### Create 1-D operators
# $\xi$ and $\omega$ are the 1D Lobatto points and weights and $D$ is the derivative of the basis function.
(ξ, ω) = lglpoints(DFloat, N)
D = spectralderivative(ξ)

# ### Compute metric terms
# nface and nelem refers to the total number of faces and elements for this MPI rank. Also, coord contains the dim-tuple coordinates in the mesh.
(nface, nelem) = size(mesh.elemtoelem)
coord = creategrid(Val(dim), mesh.elemtocoord, ξ)
if dim == 1
  x = coord.x
  for j = 1:length(x)
    x[j] = x[j]
 end
elseif dim == 2
  (x, y) = (coord.x, coord.y)
    for j = 1:length(x)
      if warp_grid
          (x[j], y[j]) = (x[j] .+ sin.(π * x[j]) .* sin.(2 * π * y[j]) / 10,
                          y[j] .+ sin.(2 * π * x[j]) .* sin.(π * y[j]) / 10)
      end
      if (icase == 6)
          (x[j], y[j]) =  (x[j] .* 1000.0, y[j] .* 1000.0)
      end
  end
elseif dim == 3
  (x, y, z) = (coord.x, coord.y, coord.z)
  for j = 1:length(x)
      if warp_grid
          (x[j], y[j], z[j]) = (x[j] + (sin(π * x[j]) * sin(2 * π * y[j]) *
                                        cos(2 * π * z[j])) / 10,
                                y[j] + (sin(π * y[j]) * sin(2 * π * x[j]) *
                                        cos(2 * π * z[j])) / 10,
                                z[j] + (sin(π * z[j]) * sin(2 * π * x[j]) *
                                        cos(2 * π * y[j])) / 10)
      end
      if (icase == 6)
          (x[j], y[j], z[j]) =  (x[j] .* 1000.0, y[j] .* 1000.0, z[j] .* 1000.0)
      end
  end
end

# ### First VTK Call
# This first VTK call dumps the mesh out for all mpiranks.
include("vtk.jl")
writemesh(@sprintf("Euler%dD_rank_%04d_mesh", dim, mpirank), coord...;
          realelems=mesh.realelems)

# ### Compute the metric terms
# This call computes the metric terms of the grid such as $\xi_\mathbf{x}$, $\eta_\mathbf{x}$, $\zeta_\mathbf{x}$ for all spatial dimensions $\mathbf{x}$ depending on the dimension of $dim$.
metric = computemetric(coord..., D)

# ### Generate the State Vectors
# We need to create as many velocity vectors as there are dimensions.
if dim == 1
  statesyms = (:ρ, :U, :E)
elseif dim == 2
  statesyms = (:ρ, :U, :V, :E)
elseif dim == 3
  statesyms = (:ρ, :U, :V, :W, :E)
end

# ### Create storage for state vector and right-hand side
# Q holds the solution vector and rhs the rhs-vector which are dim+1 tuples
# In addition, here we generate the initial conditions
Q   = NamedTuple{statesyms}(ntuple(j->zero(coord.x), length(statesyms)))
rhs = NamedTuple{statesyms}(ntuple(j->zero(coord.x), length(statesyms)))

advection=false
eqn_set = 3
Pressure = zero(coord.x)
if dim == 1
    r=(x .- 0.5).^2
    if (icase ==1) #advection
        advection=true
        gravity=0.0
        γ=0
        Q.ρ .= 0.5 .* exp.(-32.0 .* r) .+ 0.1
        Q.U .= Q.ρ .* (1.0)
        Q.E .= Q.ρ .* (1.0)
    elseif (icase == 2) #Translating Gaussian
        gravity=0.0
        Q.ρ .= 0.5 .* exp.(-32.0 .* r) .+ 1.0
        Q.U .= Q.ρ*(1.0)
        for i = 1:length(coord.x)
            Pressure[i]=1.0
        end
        Q.E.= Pressure ./ (γ-1) + 0.5 .* Q.U.^2 ./ Q.ρ
    elseif (icase == 3 || icase == 4) #Still Gaussian (Periodic or NFBC)
        gravity=0.0
        Q.ρ .= 0.5 .* exp.(-32.0 .* r) .+ 1.0
        Q.U .= 0.0
        for i = 1:length(coord.x)
            Pressure[i]=1.0
        end
        Q.E.= Pressure ./ (γ-1) + 0.5 .* Q.U.^2 ./ Q.ρ
    elseif (icase == 5) #Shock tube
        gravity=0.0
        for i = 1:length(coord.x)
            if x[i] <= 0.5
                Q.ρ[i]=1.0
                Pressure[i]=1.0
            elseif x[i] > 0.5
                Q.ρ[i]=0.125
                Pressure[i]=0.1
            end
        end
        Q.U .= 0
        Q.E.= Pressure ./ (γ-1) + 0.5 .* Q.U.^2 ./ Q.ρ
    end
elseif dim == 2
    r=sqrt.( (x .- 0.5).^2 + (y .- 0.5).^2 )
    if (icase == 1) #advection
        advection=true
        gravity=0.0
        γ=0
        Q.ρ .= 0.5 .* exp.(-100.0 .* r.^2) .+ 1.0
        Q.U .= Q.ρ .* (1.0)
        Q.V .= Q.ρ .* (0)
        Q.E .= Q.ρ .* (1.0)
    elseif (icase == 2) #Translating Gaussian (Periodic)
        gravity=0.0
        Q.ρ .= 0.5 .* exp.(-100.0 .* r.^2) .+ 1.0
        Q.U .= Q.ρ*(1.0)
        Q.V .= Q.ρ*(1.0)
        p = zero(coord.x)
        for i = 1:length(coord.x)
            Pressure[i]=1.0
        end
        Q.E.= Pressure ./ (γ-1) + 0.5 .* ( Q.U.^2 + Q.V.^2 )./ Q.ρ
    elseif (icase == 3 || icase == 4) #Still Gaussian (Periodic or NFBC)
        gravity=0.0
        Q.ρ .= 0.5 .* exp.(-100.0 .* r.^2) .+ 1.0
        Q.U .= 0.0
        Q.V .= 0.0
        p = zero(coord.x)
        for i = 1:length(coord.x)
            Pressure[i]=1.0
        end
        Q.E.= Pressure ./ (γ-1) + 0.5 .* ( Q.U.^2 + Q.V.^2 )./ Q.ρ
    elseif (icase == 5) #Shock tube (NFBC)
        gravity=0.0
        rc=0.25
        for i = 1:length(coord.x)
            Q.ρ[i] = 1.0
            Pressure[i]=1.0
            if r[i] <= rc
                Q.ρ[i] = 0.25 * (1.0 + cos(π * r[i]/rc)) + 1.0
                Pressure[i]=0.5
            end
        end
        Q.U .= 0.0
        Q.V .= 0.0
        Q.E .= Pressure ./ (γ-1) + 0.5 .* ( Q.U.^2 + Q.V.^2 )./ Q.ρ
    elseif (icase == 6) #Rising Thermal Bubble (NFBC)
        eqn_set = 2
        gravity=10.0
        r=sqrt.( (x .- 500.0).^2 + (y .- 350.0).^2 )
        rc=250.0
        θ_ref=300.0
        θ_c=0.5
        c=c_v/R_gas
        for i = 1:length(coord.x)
            Δθ=0.0
            if r[i] <= rc
                Δθ = 0.5 * θ_c * (1.0 + cos(π * r[i]/rc))
            end
            θ_k=θ_ref + Δθ
            π_k=1.0 - gravity/(c_p*θ_k)*y[i]
            ρ_k=p0/(R_gas*θ_k)*(π_k)^c
            Q.ρ[i]=ρ_k
            Q.E[i]=ρ_k*θ_k
        end
        Q.U .= Q.ρ .* u0
        Q.V .= 0
    end
elseif dim == 3
    #    r=sqrt.( (x .- 0.5).^2 + (y .- 0.5).^2 + (z .- 0.5).^2 )
    r=sqrt.( (x .- 0.5).^2 + (z .- 0.5).^2 )
    if (icase == 1) #advection
        advection=true
        gravity=0.0
        γ=0
        Q.ρ .= 0.5 .* exp.(-100.0 .* r.^2) .+ 1.0
        Q.U .= Q.ρ .* (1.0)
        Q.V .= Q.ρ .* (0)
        Q.W .= Q.ρ .* (0)
        Q.E .= Q.ρ .* (1.0)
    elseif (icase == 2) #Translating Gaussian (Periodic)
        gravity=0.0
        Q.ρ .= 0.5 .* exp.(-100.0 .* r.^2) .+ 1.0
        Q.U .= Q.ρ*(1.0)
        Q.V .= Q.ρ*(0.0)
        Q.W .= Q.ρ*(0.0)
        p = zero(coord.x)
        for i = 1:length(coord.x)
            Pressure[i]=1.0
        end
        Q.E.= Pressure ./ (γ-1) + 0.5 .* ( Q.U.^2 + Q.V.^2 + Q.W.^2)./ Q.ρ
    elseif (icase == 3 || icase == 4) #Still Gaussian (Periodic or NFBC)
        gravity=0.0
        Q.ρ .= 0.5 .* exp.(-100.0 .* r.^2) .+ 1.0
        Q.U .= 0.0
        Q.V .= 0.0
        Q.W .= 0.0
        for i = 1:length(coord.x)
            Pressure[i]=1.0
        end
        Q.E.= Pressure ./ (γ-1) + 0.5 .* ( Q.U.^2 + Q.V.^2 + Q.W.^2)./ Q.ρ
    elseif (icase == 5) #Shock tube (NFBC)
        gravity=0.0
        for i = 1:length(coord.x)
            if r[i] <= 0.25
                Q.ρ[i] = 0.5 * exp(-r[i]^2) + 1.0
                Q.ρ[i]=1.5
                Pressure[i]=1.0
            elseif r[i] > 0.25
                Q.ρ[i]=1.0
                Pressure[i]=0.1
            end
        end
        Q.U .= 0
        Q.V .= 0
        Q.W .= 0
        Q.E.= Pressure ./ (γ-1) + 0.5 .* ( Q.U.^2 + Q.V.^2 + Q.W.^2)./ Q.ρ
    elseif (icase == 6) #RTB
        eqn_set = 2
        gravity=10.0
        r=sqrt.( (x .- 500.0).^2 + (z .- 350.0).^2 )
        rc=250.0
        θ_ref=300.0
        θ_c=0.5
        c=c_v/R_gas
        for i = 1:length(coord.x)
            Δθ=0.0
            if r[i] <= rc
                Δθ = 0.5 * θ_c * (1.0 + cos(π * r[i]/rc))
            end
            θ_k=θ_ref + Δθ
            π_k=1.0 - gravity/(c_p*θ_k)*z[i]
            ρ_k=p0/(R_gas*θ_k)*(π_k)^c
            Q.ρ[i]=ρ_k
            Q.E[i]=ρ_k*θ_k
        end
        Q.U .= Q.ρ .* u0
        Q.V .= 0
        Q.W .= 0
    end
end

# ### Courant Number and Time-step Volume RHS Routines
# These functions solve the volume term $\int_{\Omega_e} \nabla \psi \cdot \left( \rho \mathbf{u} \right)^{(e)}_N$ for:
# Time-Step computation for Courant Number=1
function courant!(dt, Q, Pressure, metric, γ, dim)

# ### Compute the time-step size and number of time-steps
# Compute a $\Delta t$ such that the Courant number is $1$.
# This is done for each mpirank and then we do an MPI_Allreduce to find the global minimum.
#dt = [floatmax(DFloat)]
if dim == 1
    (ξx) = (metric.ξx)
    (ρ,U) = (Q.ρ,Q.U)
    for n = 1:length(U)
        loc_dt = (2ρ[n])  ./ (abs.(U[n] * ξx[n] + sqrt(γ*Pressure[n]/ρ[n])))
        dt[1] = min(dt[1], loc_dt)
    end
elseif dim == 2
    (ξx, ξy) = (metric.ξx, metric.ξy)
    (ηx, ηy) = (metric.ηx, metric.ηy)
    (ρ,U,V) = (Q.ρ,Q.U,Q.V)
    for n = 1:length(U)
        loc_dt = (2ρ[n]) ./ max(abs.(U[n] * ξx[n] + V[n] * ξy[n] + sqrt(γ*Pressure[n]/ρ[n])),
                          abs.(U[n] * ηx[n] + V[n] * ηy[n] + sqrt(γ*Pressure[n]/ρ[n])))
        dt[1] = min(dt[1], loc_dt)
    end
elseif dim == 3
    (ξx, ξy, ξz) = (metric.ξx, metric.ξy, metric.ξz)
    (ηx, ηy, ηz) = (metric.ηx, metric.ηy, metric.ηz)
    (ζx, ζy, ζz) = (metric.ζx, metric.ζy, metric.ζz)
    (ρ,U,V,W) = (Q.ρ,Q.U,Q.V,Q.W)
    for n = 1:length(U)
        loc_dt = (2ρ[n]) ./ max(abs.(U[n] * ξx[n] + V[n] * ξy[n] + W[n] * ξz[n] + sqrt(γ*Pressure[n]/ρ[n])),
                          abs.(U[n] * ηx[n] + V[n] * ηy[n] + W[n] * ηz[n] + sqrt(γ*Pressure[n]/ρ[n])),
                          abs.(U[n] * ζx[n] + V[n] * ζy[n] + W[n] * ζz[n] + sqrt(γ*Pressure[n]/ρ[n])))
        dt[1] = min(dt[1], loc_dt)
    end
end
dt .= MPI.Allreduce(dt[1], MPI.MIN, mpicomm)
#dt = DFloat(dt / N^sqrt(2))
#nsteps = ceil(Int64, tend / dt)
#dt = tend / nsteps
#@show (dt, nsteps)
end #function courant

# ### Volume RHS Routines
# These functions solve the volume term $\int_{\Omega_e} \nabla \psi \cdot \left( \rho \mathbf{u} \right)^{(e)}_N$ for:
# Pressure1D
function compute_pressure!(Pressure, Q::NamedTuple{S, NTuple{3, T}}, γ, eqn_set) where {S, T}

    #Gas Constants (will be moved to a Module)
    p0=100000.0
    R_gas=287.17
    c_p=1004.67
    c_v=717.5

    if (eqn_set == 3)
        Pressure .= (γ-1)*( Q.E - 0.5 .* Q.U.^2 ./ Q.ρ )
    elseif (eqn_set == 2)
        Pressure .= p0 .* ( R_gas .* Q.E ./ p0 ).^(c_p/c_v)
    end
end #function Pressure1D

# Pressure2D
function compute_pressure!(Pressure, Q::NamedTuple{S, NTuple{4, T}}, γ, eqn_set) where {S, T}

    #Gas Constants (will be moved to a Module)
    p0=100000.0
    R_gas=287.17
    c_p=1004.67
    c_v=717.5

    if (eqn_set == 3)
        Pressure .= (γ-1) .* ( Q.E - 0.5 .* ( Q.U.^2 + Q.V.^2 )./ Q.ρ )
    elseif (eqn_set == 2)
        Pressure .= p0 .* ( R_gas .* Q.E ./ p0 ).^(c_p/c_v)
    end
end #function Pressure2D

# Pressure3D
function compute_pressure!(Pressure, Q::NamedTuple{S, NTuple{5, T}}, γ, eqn_set) where {S, T}

    #Gas Constants (will be moved to a Module)
    p0=100000.0
    R_gas=287.17
    c_p=1004.67
    c_v=717.5

    if (eqn_set == 3)
        Pressure .= (γ-1)*( Q.E - 0.5 .* ( Q.U.^2 + Q.V.^2 + Q.W.^2 )./ Q.ρ )
    elseif (eqn_set == 2)
        Pressure .= p0 .* ( R_gas .* Q.E ./ p0 ).^(c_p/c_v)
    end
end #function Pressure2D


# ### Compute the time-step size and number of time-steps
# Compute a $\Delta t$ such that the Courant number is $1$.
# This is done for each mpirank and then we do an MPI_Allreduce to find the global minimum.
compute_pressure!(Pressure, Q, γ, eqn_set)

#=
dt = [floatmax(DFloat)]
@show (typeof(dt),sizeof(dt))
@show (dt)
courant!(dt, Q, Pressure, metric, γ, dim)
dt = DFloat(dt / N^sqrt(2))
nsteps = ceil(Int64, tend / dt)
dt = tend / nsteps
@show (dt, nsteps)
=#

dt = [floatmax(DFloat)]
if dim == 1
    (ξx) = (metric.ξx)
    (ρ,U) = (Q.ρ,Q.U)
    for n = 1:length(U)
        loc_dt = (2*ρ[n])  ./ (abs.(U[n] * ξx[n] + ρ[n] * sqrt(γ*Pressure[n]/ρ[n])))
        dt[1] = min(dt[1], loc_dt)
    end
elseif dim == 2
    (ξx, ξy) = (metric.ξx, metric.ξy)
    (ηx, ηy) = (metric.ηx, metric.ηy)
    (ρ,U,V) = (Q.ρ,Q.U,Q.V)
    for n = 1:length(U)
        loc_dt = (2*ρ[n]) ./ max(abs.(U[n] * ξx[n] + V[n] * ξy[n] + ρ[n]*sqrt(γ*Pressure[n]/ρ[n])),
                          abs.(U[n] * ηx[n] + V[n] * ηy[n] + ρ[n]*sqrt(γ*Pressure[n]/ρ[n])))
        dt[1] = min(dt[1], loc_dt)
    end
elseif dim == 3
    (ξx, ξy, ξz) = (metric.ξx, metric.ξy, metric.ξz)
    (ηx, ηy, ηz) = (metric.ηx, metric.ηy, metric.ηz)
    (ζx, ζy, ζz) = (metric.ζx, metric.ζy, metric.ζz)
    (ρ,U,V,W,E) = (Q.ρ,Q.U,Q.V,Q.W,Q.E)
    for n = 1:length(U)
        loc_dt = (2*ρ[n]) ./ max(abs.(U[n] * ξx[n] + V[n] * ξy[n] + W[n] * ξz[n] + ρ[n] * sqrt(γ*Pressure[n]/ρ[n])),
                          abs.(U[n] * ηx[n] + V[n] * ηy[n] + W[n] * ηz[n] + ρ[n] * sqrt(γ*Pressure[n]/ρ[n])),
                          abs.(U[n] * ζx[n] + V[n] * ζy[n] + W[n] * ζz[n] + ρ[n] * sqrt(γ*Pressure[n]/ρ[n])))
        dt[1] = min(dt[1], loc_dt)
    end
end
dt = MPI.Allreduce(dt[1], MPI.MIN, mpicomm)
dt = DFloat(dt / N^sqrt(2))
#dt=0.01
nsteps = ceil(Int64, tend / dt)
dt = tend / nsteps
@show (dt, nsteps)

# ### Compute the exact solution at the final time.
# Later Δ will be used to store the difference between the exact and computed solutions.
Δ   = NamedTuple{statesyms}(ntuple(j->zero(coord.x), length(statesyms)))
if dim == 1
  Δ.ρ .= Q.ρ
  Δ.U .= Q.U
  Δ.E .= Q.E
elseif dim == 2
  Δ.ρ .= Q.ρ
  Δ.U .= Q.U
  Δ.V .= Q.V
  Δ.E .= Q.E
elseif dim == 3
  Δ.ρ .= Q.ρ
  Δ.U .= Q.U
  Δ.V .= Q.V
  Δ.W .= Q.W
  Δ.E .= Q.E
end

# ### Store Explicit RK Time-stepping Coefficients
# We use the fourth-order, low-storage, Runge–Kutta scheme of Carpenter and Kennedy (1994)
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

#-------------------------------------------------------------------------------#
#-----Begin Courant, Volume, Flux, Update, and Error Functions for Multiple Dispatch-----#
#-------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------#
#-----Begin Volume, Flux, Update, and Error Functions for Multiple Dispatch-----#
#-------------------------------------------------------------------------------#
# ### Volume RHS Routines
# These functions solve the volume term $\int_{\Omega_e} \nabla \psi \cdot \left( \rho \mathbf{u} \right)^{(e)}_N$ for:
# Volume RHS for 1D
function volumerhs!(rhs, Q::NamedTuple{S, NTuple{3, T}}, Pressure, metric, D, ω, elems, gravity, γ, eqn_set) where {S, T}

    (rhsρ, rhsU, rhsE) = (rhs.ρ, rhs.U, rhs.E)
    (ρ, U, E) = (Q.ρ, Q.U, Q.E)
    Nq = size(ρ, 1)
    J = metric.J
    ξx = metric.ξx
    for e ∈ elems
        #Get primitive variables and fluxes
        ρt=ρ[:,e]
        u=U[:,e] ./ ρt
        p=Pressure[:,e]
        fluxρ=U[:,e]
        fluxU=ρt .* u .* u + p
        fluxE=(E[:,e] + p) .* u
        # loop of ξ-grid lines
        rhsρ[:,e] += D' * (ω .* J[:,e] .* (ξx[:,e] .* fluxρ[:]))
        rhsU[:,e] += D' * (ω .* J[:,e] .* (ξx[:,e] .* fluxU[:]))
        rhsE[:,e] += D' * (ω .* J[:,e] .* (ξx[:,e] .* fluxE[:]))
    end #e ∈ elems
end #function volumerhs-1d

# Volume RHS for 2D
function volumerhs!(rhs, Q::NamedTuple{S, NTuple{4, T}}, Pressure, metric, D, ω, elems, gravity, γ, eqn_set) where {S, T}

    (rhsρ, rhsU, rhsV, rhsE) = (rhs.ρ, rhs.U, rhs.V, rhs.E)
    (ρ, U, V, E) = (Q.ρ, Q.U, Q.V, Q.E)
    Nq = size(ρ, 1)
    J = metric.J
    dim=2
    δP=1.0
    if (eqn_set == 2)
        δP=0.0
    end
    p0=100000.0
    R_gas=287.17
    c_p=1004.67
    c_v=717.5
    (ξx, ξy) = (metric.ξx, metric.ξy)
    (ηx, ηy) = (metric.ηx, metric.ηy)

    #Allocate local flux arrays
    fluxρ=Array{DFloat,3}(undef,dim,Nq,Nq)
    fluxU=Array{DFloat,3}(undef,dim,Nq,Nq)
    fluxV=Array{DFloat,3}(undef,dim,Nq,Nq)
    fluxE=Array{DFloat,3}(undef,dim,Nq,Nq)

    for e ∈ elems

        #Get primitive variables and fluxes
        ρt=ρ[:,:,e]
        u=U[:,:,e] ./ ρt
        v=V[:,:,e] ./ ρt
        p=Pressure[:,:,e]
        fluxρ[1,:,:]=U[:,:,e]
        fluxρ[2,:,:]=V[:,:,e]
        fluxU[1,:,:]=ρt .* u .* u + p
        fluxU[2,:,:]=ρt .* u .* v
        fluxV[1,:,:]=ρt .* v .* u
        fluxV[2,:,:]=ρt .* v .* v + p
        fluxE[1,:,:]=(E[:,:,e] + δP .* p) .* u
        fluxE[2,:,:]=(E[:,:,e] + δP .* p) .* v

        # loop of ξ-grid lines
        for j = 1:Nq
            rhsρ[:,j,e] += D' * (ω[j] * ω .* J[:,j,e].* (ξx[:,j,e] .* fluxρ[1,:,j] + ξy[:,j,e] .* fluxρ[2,:,j]))
            rhsU[:,j,e] += D' * (ω[j] * ω .* J[:,j,e].* (ξx[:,j,e] .* fluxU[1,:,j] + ξy[:,j,e] .* fluxU[2,:,j]))
            rhsV[:,j,e] += D' * (ω[j] * ω .* J[:,j,e].* (ξx[:,j,e] .* fluxV[1,:,j] + ξy[:,j,e] .* fluxV[2,:,j]))
            rhsE[:,j,e] += D' * (ω[j] * ω .* J[:,j,e].* (ξx[:,j,e] .* fluxE[1,:,j] + ξy[:,j,e] .* fluxE[2,:,j]))
        end #j
        # loop of η-grid lines
        for i = 1:Nq
            rhsρ[i,:,e] += D' * (ω[i] * ω .* J[i,:,e].* (ηx[i,:,e] .* fluxρ[1,i,:] + ηy[i,:,e] .* fluxρ[2,i,:]))
            rhsU[i,:,e] += D' * (ω[i] * ω .* J[i,:,e].* (ηx[i,:,e] .* fluxU[1,i,:] + ηy[i,:,e] .* fluxU[2,i,:]))
            rhsV[i,:,e] += D' * (ω[i] * ω .* J[i,:,e].* (ηx[i,:,e] .* fluxV[1,i,:] + ηy[i,:,e] .* fluxV[2,i,:]))
            rhsE[i,:,e] += D' * (ω[i] * ω .* J[i,:,e].* (ηx[i,:,e] .* fluxE[1,i,:] + ηy[i,:,e] .* fluxE[2,i,:]))
        end #i

        #Buoyancy Term
        for j = 1:Nq
            for i = 1:Nq
                rhsV[i,j,e] -= ( ω[i] * ω[j] * J[i,j,e] * ρt[i,j] * gravity )
            end
        end
    end #e ∈ elems
end #function volumerhs-2d

# Volume RHS for 3D
function volumerhs!(rhs, Q::NamedTuple{S, NTuple{5, T}}, Pressure, metric, D, ω, elems, gravity, γ, eqn_set) where {S, T}

    (rhsρ, rhsU, rhsV, rhsW, rhsE) = (rhs.ρ, rhs.U, rhs.V, rhs.W, rhs.E)
    (ρ, U, V, W, E) = (Q.ρ, Q.U, Q.V, Q.W, Q.E)
    Nq = size(ρ, 1)
    J = metric.J
    dim=3
    δP=1.0
    if (eqn_set == 2)
        δP=0.0
    end
    p0=100000.0
    R_gas=287.17
    c_p=1004.67
    c_v=717.5
    (ξx, ξy, ξz) = (metric.ξx, metric.ξy, metric.ξz)
    (ηx, ηy, ηz) = (metric.ηx, metric.ηy, metric.ηz)
    (ζx, ζy, ζz) = (metric.ζx, metric.ζy, metric.ζz)

    #Allocate local flux arrays
    fluxρ=Array{DFloat,4}(undef,dim,Nq,Nq,Nq)
    fluxU=Array{DFloat,4}(undef,dim,Nq,Nq,Nq)
    fluxV=Array{DFloat,4}(undef,dim,Nq,Nq,Nq)
    fluxW=Array{DFloat,4}(undef,dim,Nq,Nq,Nq)
    fluxE=Array{DFloat,4}(undef,dim,Nq,Nq,Nq)

    for e ∈ elems

        #Get primitive variables and fluxes
        ρt=ρ[:,:,:,e]
        u=U[:,:,:,e] ./ ρt
        v=V[:,:,:,e] ./ ρt
        w=W[:,:,:,e] ./ ρt
        p=Pressure[:,:,:,e]
        fluxρ[1,:,:,:]=U[:,:,:,e]
        fluxρ[2,:,:,:]=V[:,:,:,e]
        fluxρ[3,:,:,:]=W[:,:,:,e]
        fluxU[1,:,:,:]=ρt .* u .* u + p
        fluxU[2,:,:,:]=ρt .* u .* v
        fluxU[3,:,:,:]=ρt .* u .* w
        fluxV[1,:,:,:]=ρt .* v .* u
        fluxV[2,:,:,:]=ρt .* v .* v + p
        fluxV[3,:,:,:]=ρt .* v .* w
        fluxW[1,:,:,:]=ρt .* w .* u
        fluxW[2,:,:,:]=ρt .* w .* v
        fluxW[3,:,:,:]=ρt .* w .* w + p
        fluxE[1,:,:,:]=(E[:,:,:,e] + δP .* p) .* u
        fluxE[2,:,:,:]=(E[:,:,:,e] + δP .* p) .* v
        fluxE[3,:,:,:]=(E[:,:,:,e] + δP .* p) .* w

        # loop of ξ-grid lines
        for k = 1:Nq
            for j = 1:Nq
                rhsρ[:,j,k,e] += D' * (ω[j] * ω[k] * ω .* J[:,j,k,e] .*
                          (ξx[:,j,k,e] .* fluxρ[1,:,j,k] + ξy[:,j,k,e] .* fluxρ[2,:,j,k] + ξz[:,j,k,e] .* fluxρ[3,:,j,k]))
                rhsU[:,j,k,e] += D' * (ω[j] * ω[k] * ω .* J[:,j,k,e] .*
                          (ξx[:,j,k,e] .* fluxU[1,:,j,k] + ξy[:,j,k,e] .* fluxU[2,:,j,k] + ξz[:,j,k,e] .* fluxU[3,:,j,k]))
                rhsV[:,j,k,e] += D' * (ω[j] * ω[k] * ω .* J[:,j,k,e] .*
                          (ξx[:,j,k,e] .* fluxV[1,:,j,k] + ξy[:,j,k,e] .* fluxV[2,:,j,k] + ξz[:,j,k,e] .* fluxV[3,:,j,k]))
                rhsW[:,j,k,e] += D' * (ω[j] * ω[k] * ω .* J[:,j,k,e] .*
                          (ξx[:,j,k,e] .* fluxW[1,:,j,k] + ξy[:,j,k,e] .* fluxW[2,:,j,k] + ξz[:,j,k,e] .* fluxW[3,:,j,k]))
                rhsE[:,j,k,e] += D' * (ω[j] * ω[k] * ω .* J[:,j,k,e] .*
                          (ξx[:,j,k,e] .* fluxE[1,:,j,k] + ξy[:,j,k,e] .* fluxE[2,:,j,k] + ξz[:,j,k,e] .* fluxE[3,:,j,k]))
            end #j
        end #k

        # loop of η-grid lines
        for k = 1:Nq
            for i = 1:Nq
                rhsρ[i,:,k,e] += D' * (ω[i] * ω[k] * ω .* J[i,:,k,e] .*
                          (ηx[i,:,k,e] .* fluxρ[1,i,:,k] + ηy[i,:,k,e] .* fluxρ[2,i,:,k] + ηz[i,:,k,e] .* fluxρ[3,i,:,k]))
                rhsU[i,:,k,e] += D' * (ω[i] * ω[k] * ω .* J[i,:,k,e] .*
                          (ηx[i,:,k,e] .* fluxU[1,i,:,k] + ηy[i,:,k,e] .* fluxU[2,i,:,k] + ηz[i,:,k,e] .* fluxU[3,i,:,k]))
                rhsV[i,:,k,e] += D' * (ω[i] * ω[k] * ω .* J[i,:,k,e] .*
                          (ηx[i,:,k,e] .* fluxV[1,i,:,k] + ηy[i,:,k,e] .* fluxV[2,i,:,k] + ηz[i,:,k,e] .* fluxV[3,i,:,k]))
                rhsW[i,:,k,e] += D' * (ω[i] * ω[k] * ω .* J[i,:,k,e] .*
                          (ηx[i,:,k,e] .* fluxW[1,i,:,k] + ηy[i,:,k,e] .* fluxW[2,i,:,k] + ηz[i,:,k,e] .* fluxW[3,i,:,k]))
                rhsE[i,:,k,e] += D' * (ω[i] * ω[k] * ω .* J[i,:,k,e] .*
                          (ηx[i,:,k,e] .* fluxE[1,i,:,k] + ηy[i,:,k,e] .* fluxE[2,i,:,k] + ηz[i,:,k,e] .* fluxE[3,i,:,k]))
            end #i
        end #k

        # loop of ζ-grid lines
        for j = 1:Nq
            for i = 1:Nq
                rhsρ[i,j,:,e] += D' * (ω[i] * ω[j] * ω .* J[i,j,:,e] .*
                          (ζx[i,j,:,e] .* fluxρ[1,i,j,:] + ζy[i,j,:,e] .* fluxρ[2,i,j,:] + ζz[i,j,:,e] .* fluxρ[3,i,j,:]))
                rhsU[i,j,:,e] += D' * (ω[i] * ω[j] * ω .* J[i,j,:,e] .*
                          (ζx[i,j,:,e] .* fluxU[1,i,j,:] + ζy[i,j,:,e] .* fluxU[2,i,j,:] + ζz[i,j,:,e] .* fluxU[3,i,j,:]))
                rhsV[i,j,:,e] += D' * (ω[i] * ω[j] * ω .* J[i,j,:,e] .*
                          (ζx[i,j,:,e] .* fluxV[1,i,j,:] + ζy[i,j,:,e] .* fluxV[2,i,j,:] + ζz[i,j,:,e] .* fluxV[3,i,j,:]))
                rhsW[i,j,:,e] += D' * (ω[i] * ω[j] * ω .* J[i,j,:,e] .*
                          (ζx[i,j,:,e] .* fluxW[1,i,j,:] + ζy[i,j,:,e] .* fluxW[2,i,j,:] + ζz[i,j,:,e] .* fluxW[3,i,j,:]))
                rhsE[i,j,:,e] += D' * (ω[i] * ω[j] * ω .* J[i,j,:,e] .*
                          (ζx[i,j,:,e] .* fluxE[1,i,j,:] + ζy[i,j,:,e] .* fluxE[2,i,j,:] + ζz[i,j,:,e] .* fluxE[3,i,j,:]))
            end #i
        end #j

        #Buoyancy Term
        for k = 1:Nq
            for j = 1:Nq
                for i = 1:Nq
                    rhsW[i,j,k,e] -= ( ω[i] * ω[j] *  ω[k] * J[i,j,k,e] * ρt[i,j,k] * gravity )
                end
            end
        end

    end #e ∈ elems
end #function volumerhs-3d

# ### Flux RHS Routines
# These functions solve the flux integral term $\int_{\Gamma_e} \psi \mathbf{n} \cdot \left( \rho \mathbf{u} \right)^{(*,e)}_N$ for:
# Flux RHS for 1D
function fluxrhs!(rhs, Q::NamedTuple{S, NTuple{3, T}}, Pressure, metric, ω, elems, boundary, vmapM, vmapP, γ, eqn_set) where {S, T}

    (rhsρ, rhsU, rhsE) = (rhs.ρ, rhs.U, rhs.E)
    (ρ, U, E) = (Q.ρ, Q.U, Q.E)
    Nq = size(ρ, 1)
    nface = 2
    (nx, sJ) = (metric.nx, metric.sJ)
    nx = reshape(nx, size(vmapM))
    sJ = reshape(sJ, size(vmapM))

    for e ∈ elems
        for f ∈ 1:nface

            #Check Boundary Condition
            bc=boundary[f,e]

            #Compute fluxes on M/Left/- side
            ρM = ρ[vmapM[1, f, e]]
            UM = U[vmapM[1, f, e]]
            EM = E[vmapM[1, f, e]]
            uM = UM ./ ρM
            pM = Pressure[vmapM[1, f, e]]

            #Left Fluxes
            fluxρM = UM
            fluxUM = ρM .* uM .* uM + pM
            fluxEM = (EM + pM) .* uM

            #Compute fluxes on P/Right/+ side
            ρP = ρ[vmapP[1, f, e]]
            UP = U[vmapP[1, f, e]]
            EP = E[vmapP[1, f, e]]

            if bc == 0 #no boundary or periodic
                pP = Pressure[vmapP[1, f, e]]
            elseif bc == 1 #No-flux
                ρP = ρM
                pP = pM
                UP = -UM
                EP = EM
            end
            uP = UP ./ ρP

            #Right Fluxes
            fluxρP = UP
            fluxUP = ρP .* uP .* uP + pP
            fluxEP = (EP + pP) .* uP

            #Compute wave speed
            nxM = nx[1, f, e]
            λM=abs.(nxM .* uM) + sqrt.(γ .* pM ./ ρM)
            λP=abs.(nxM .* uP) + sqrt.(γ .* pP ./ ρP)
            λ = max.( λM, λP )

            #Compute Numerical Flux and Update
            fluxρ_star = (nxM .* (fluxρM + fluxρP) - λ .* (ρP - ρM)) / 2
            fluxU_star = (nxM .* (fluxUM + fluxUP) - λ .* (UP - UM)) / 2
            fluxE_star = (nxM .* (fluxEM + fluxEP) - λ .* (EP - EM)) / 2
            rhsρ[vmapM[1, f, e]] -= sJ[1, f, e] .* fluxρ_star
            rhsU[vmapM[1, f, e]] -= sJ[1, f, e] .* fluxU_star
            rhsE[vmapM[1, f, e]] -= sJ[1, f, e] .* fluxE_star
        end #for f ∈ 1:nface
    end #e ∈ elems
end #function fluxrhs-1d

# Flux RHS for 2D
function fluxrhs!(rhs, Q::NamedTuple{S, NTuple{4, T}}, Pressure, metric, ω, elems, boundary, vmapM, vmapP, γ, eqn_set) where {S, T}

    (rhsρ, rhsU, rhsV, rhsE) = (rhs.ρ, rhs.U, rhs.V, rhs.E)
    (ρ, U, V, E) = (Q.ρ, Q.U, Q.V, Q.E)
    Nq = size(ρ, 1)
    nface = 4
    dim=2
    δP=1.0
    if (eqn_set == 2)
        δP=0.0
    end
    p0=100000.0
    R_gas=287.17
    c_p=1004.67
    c_v=717.5
    (nx, ny, sJ) = (metric.nx, metric.ny, metric.sJ)

    #Allocate local flux arrays
    fluxρM=Array{DFloat,2}(undef,dim,Nq)
    fluxUM=Array{DFloat,2}(undef,dim,Nq)
    fluxVM=Array{DFloat,2}(undef,dim,Nq)
    fluxEM=Array{DFloat,2}(undef,dim,Nq)
    fluxρP=Array{DFloat,2}(undef,dim,Nq)
    fluxUP=Array{DFloat,2}(undef,dim,Nq)
    fluxVP=Array{DFloat,2}(undef,dim,Nq)
    fluxEP=Array{DFloat,2}(undef,dim,Nq)

    for e ∈ elems
        for f ∈ 1:nface

            #Check Boundary Condition and Store Normal Vectors
            bc=boundary[f,e]
            nxM = nx[:, f, e]
            nyM = ny[:, f, e]

            #Compute fluxes on M/Left/- side
            ρM = ρ[vmapM[:, f, e]]
            UM = U[vmapM[:, f, e]]
            VM = V[vmapM[:, f, e]]
            EM = E[vmapM[:, f, e]]
            uM = UM ./ ρM
            vM = VM ./ ρM
            pM = Pressure[vmapM[:, f, e]]

            #Left Fluxes
            fluxρM[1,:] = UM
            fluxρM[2,:] = VM
            fluxUM[1,:] = ρM .* uM .* uM + pM
            fluxUM[2,:] = ρM .* uM .* vM
            fluxVM[1,:] = ρM .* vM .* uM
            fluxVM[2,:] = ρM .* vM .* vM + pM
            fluxEM[1,:] = (EM + δP .* pM) .* uM
            fluxEM[2,:] = (EM + δP .* pM) .* vM

            #Compute fluxes on P/Right/+ side
            ρP = ρ[vmapP[:, f, e]]
            UP = U[vmapP[:, f, e]]
            VP = V[vmapP[:, f, e]]
            EP = E[vmapP[:, f, e]]

            if bc == 0 #no boundary or periodic
                pP = Pressure[vmapP[:, f, e]]
            elseif bc == 1 #No-flux
                Unormal=nxM .* UM + nyM .* VM
                UP = UM - 2.0 .* Unormal .* nxM
                VP = VM - 2.0 .* Unormal .* nyM
                ρP = ρM
                pP = pM
                EP = EM
            end
            uP = UP ./ ρP
            vP = VP ./ ρP

            #Right Fluxes
            fluxρP[1,:] = UP
            fluxρP[2,:] = VP
            fluxUP[1,:] = ρP .* uP .* uP + pP
            fluxUP[2,:] = ρP .* uP .* vP
            fluxVP[1,:] = ρP .* vP .* uP
            fluxVP[2,:] = ρP .* vP .* vP + pP
            fluxEP[1,:] = (EP + δP .* pP) .* uP
            fluxEP[2,:] = (EP + δP .* pP) .* vP

            #Compute wave speed
            λM=abs.(nxM .* uM + nyM .* vM) + sqrt.(γ .* pM ./ ρM)
            λP=abs.(nxM .* uP + nyM .* vP) + sqrt.(γ .* pP ./ ρP)
            λ = max.( λM, λP )

            #Compute Numerical Flux and Update
            fluxρ_star = (nxM .* (fluxρM[1,:] + fluxρP[1,:]) + nyM .* (fluxρM[2,:] + fluxρP[2,:]) - λ .* (ρP - ρM)) / 2
            fluxU_star = (nxM .* (fluxUM[1,:] + fluxUP[1,:]) + nyM .* (fluxUM[2,:] + fluxUP[2,:]) - λ .* (UP - UM)) / 2
            fluxV_star = (nxM .* (fluxVM[1,:] + fluxVP[1,:]) + nyM .* (fluxVM[2,:] + fluxVP[2,:]) - λ .* (VP - VM)) / 2
            fluxE_star = (nxM .* (fluxEM[1,:] + fluxEP[1,:]) + nyM .* (fluxEM[2,:] + fluxEP[2,:]) - λ .* (EP - EM)) / 2
            rhsρ[vmapM[:, f, e]] -= ω .* sJ[:, f, e] .* fluxρ_star
            rhsU[vmapM[:, f, e]] -= ω .* sJ[:, f, e] .* fluxU_star
            rhsV[vmapM[:, f, e]] -= ω .* sJ[:, f, e] .* fluxV_star
            rhsE[vmapM[:, f, e]] -= ω .* sJ[:, f, e] .* fluxE_star
        end #f ∈ 1:nface
    end #e ∈ elems
end #function fluxrhs-2d

# Flux RHS for 3D
function fluxrhs!(rhs, Q::NamedTuple{S, NTuple{5, T}}, metric, ω, elems, boundary, vmapM, vmapP, γ, eqn_set) where {S, T}

    (rhsρ, rhsU, rhsV, rhsW, rhsE) = (rhs.ρ, rhs.U, rhs.V, rhs.W, rhs.E)
    (ρ, U, V, W, E) = (Q.ρ, Q.U, Q.V, Q.W, Q.E)
    Nq = size(ρ, 1)
    Nq2=Nq*Nq
    nface = 6
    dim=3
    δP=1.0
    if (eqn_set == 2)
        δP=0.0
    end
    p0=100000.0
    R_gas=287.17
    c_p=1004.67
    c_v=717.5
    (nx, ny, nz, sJ) = (metric.nx, metric.ny, metric.nz, metric.sJ)
    nx = reshape(nx, size(vmapM))
    ny = reshape(ny, size(vmapM))
    nz = reshape(nz, size(vmapM))
    sJ = reshape(sJ, size(vmapM))

    #Allocate local flux arrays
    fluxρM=Array{DFloat,2}(undef,dim,Nq2)
    fluxUM=Array{DFloat,2}(undef,dim,Nq2)
    fluxVM=Array{DFloat,2}(undef,dim,Nq2)
    fluxWM=Array{DFloat,2}(undef,dim,Nq2)
    fluxEM=Array{DFloat,2}(undef,dim,Nq2)
    fluxρP=Array{DFloat,2}(undef,dim,Nq2)
    fluxUP=Array{DFloat,2}(undef,dim,Nq2)
    fluxVP=Array{DFloat,2}(undef,dim,Nq2)
    fluxWP=Array{DFloat,2}(undef,dim,Nq2)
    fluxEP=Array{DFloat,2}(undef,dim,Nq2)

    for e ∈ elems
        for f ∈ 1:nface

            #Check Boundary Condition and Store Normal Vectors
            bc=boundary[f,e]
            nxM = nx[:, f, e]
            nyM = ny[:, f, e]
            nzM = nz[:, f, e]

            #Compute fluxes on M/Left/- side
            ρM = ρ[vmapM[:, f, e]]
            UM = U[vmapM[:, f, e]]
            VM = V[vmapM[:, f, e]]
            WM = W[vmapM[:, f, e]]
            EM = E[vmapM[:, f, e]]
            uM = UM ./ ρM
            vM = VM ./ ρM
            wM = WM ./ ρM
            pM = Pressure[vmapM[:, f, e]]

            #Left Fluxes
            fluxρM[1,:] = UM
            fluxρM[2,:] = VM
            fluxρM[3,:] = WM
            fluxUM[1,:] = ρM .* uM .* uM + pM
            fluxUM[2,:] = ρM .* uM .* vM
            fluxUM[3,:] = ρM .* uM .* wM
            fluxVM[1,:] = ρM .* vM .* uM
            fluxVM[2,:] = ρM .* vM .* vM + pM
            fluxVM[3,:] = ρM .* vM .* wM
            fluxWM[1,:] = ρM .* wM .* uM
            fluxWM[2,:] = ρM .* wM .* vM
            fluxWM[3,:] = ρM .* wM .* wM + pM
            fluxEM[1,:] = (EM +  δP .* pM) .* uM
            fluxEM[2,:] = (EM +  δP .* pM) .* vM
            fluxEM[3,:] = (EM +  δP .* pM) .* wM

            #Compute fluxes on P/Right/+ side
            ρP = ρ[vmapP[:, f, e]]
            UP = U[vmapP[:, f, e]]
            VP = V[vmapP[:, f, e]]
            WP = W[vmapP[:, f, e]]
            EP = E[vmapP[:, f, e]]

            if bc == 0 #no boundary or periodic
                pP = Pressure[vmapP[:, f, e]]
            elseif bc == 1 #No-flux
                Unormal=nxM .* UM + nyM .* VM
                UP = UM - 2.0 .* Unormal .* nxM
                VP = VM - 2.0 .* Unormal .* nyM
                WP = WM - 2.0 .* Unormal .* nzM
                ρP = ρM
                pP = pM
                UP = -UM
                VP = -VM
                WP = -WM
                EP = EM
            end
            uP = UP ./ ρP
            vP = VP ./ ρP
            wP = WP ./ ρP

            #Right Fluxes
            fluxρP[1,:] = UP
            fluxρP[2,:] = VP
            fluxρP[3,:] = WP
            fluxUP[1,:] = ρP .* uP .* uP + pP
            fluxUP[2,:] = ρP .* uP .* vP
            fluxUP[3,:] = ρP .* uP .* wP
            fluxVP[1,:] = ρP .* vP .* uP
            fluxVP[2,:] = ρP .* vP .* vP + pP
            fluxVP[3,:] = ρP .* vP .* wP
            fluxWP[1,:] = ρP .* wP .* uP
            fluxWP[2,:] = ρP .* wP .* vP
            fluxWP[3,:] = ρP .* wP .* wP + pP
            fluxEP[1,:] = (EP + δP .* pP) .* uP
            fluxEP[2,:] = (EP + δP .* pP) .* vP
            fluxEP[3,:] = (EP + δP .* pP) .* wP

            #Compute wave speed
            λM=abs.(nxM .* uM + nyM .* vM + nzM .* wM) + sqrt.(γ .* pM ./ ρM)
            λP=abs.(nxM .* uP + nyM .* vP + nzM .* wP) + sqrt.(γ .* pP ./ ρP)
            λ = max.( λM, λP )

            #Compute Numerical Flux and Update
            fluxρ_star = (nxM .* (fluxρM[1,:] + fluxρP[1,:]) + nyM .* (fluxρM[2,:] + fluxρP[2,:]) + nzM .* (fluxρM[3,:] + fluxρP[3,:])
                          - λ .* (ρP - ρM)) / 2
            fluxU_star = (nxM .* (fluxUM[1,:] + fluxUP[1,:]) + nyM .* (fluxUM[2,:] + fluxUP[2,:]) + nzM .* (fluxUM[3,:] + fluxUP[3,:])
                          - λ .* (UP - UM)) / 2
            fluxV_star = (nxM .* (fluxVM[1,:] + fluxVP[1,:]) + nyM .* (fluxVM[2,:] + fluxVP[2,:]) + nzM .* (fluxVM[3,:] + fluxVP[3,:])
                          - λ .* (VP - VM)) / 2
            fluxW_star = (nxM .* (fluxWM[1,:] + fluxWP[1,:]) + nyM .* (fluxWM[2,:] + fluxWP[2,:]) + nzM .* (fluxWM[3,:] + fluxWP[3,:])
                          - λ .* (WP - WM)) / 2
            fluxE_star = (nxM .* (fluxEM[1,:] + fluxEP[1,:]) + nyM .* (fluxEM[2,:] + fluxEP[2,:]) + nzM .* (fluxEM[3,:] + fluxEP[3,:])
                          - λ .* (EP - EM)) / 2
            rhsρ[vmapM[:, f, e]] -= kron(ω, ω) .* sJ[:, f, e] .* fluxρ_star
            rhsU[vmapM[:, f, e]] -= kron(ω, ω) .* sJ[:, f, e] .* fluxU_star
            rhsV[vmapM[:, f, e]] -= kron(ω, ω) .* sJ[:, f, e] .* fluxV_star
            rhsW[vmapM[:, f, e]] -= kron(ω, ω) .* sJ[:, f, e] .* fluxW_star
            rhsE[vmapM[:, f, e]] -= kron(ω, ω) .* sJ[:, f, e] .* fluxE_star
        end #f ∈ 1:nface
    end #e ∈ elems
end #function fluxrhs-3d

# ### Update the solution via RK Method for:
# Update 1D
function updatesolution!(rhs, Q::NamedTuple{S, NTuple{3, T}}, metric, ω, elems, rka, rkb, dt, advection) where {S, T}
    #Save original velocity
    if advection
        ρ = Q.ρ
        u = Q.U ./ ρ
        energy = Q.E ./ ρ
    end

    J = metric.J
    M =  ω
    for (rhsq, q) ∈ zip(rhs, Q)
        for e ∈ elems
            q[:, e] += rkb * dt * rhsq[:, e] ./ ( M .* J[:, e])
            rhsq[:, e] *= rka
        end
    end
    #Reset velocity
    if advection
        Q.U .= Q.ρ .* u
        Q.E .= Q.ρ .* energy
    end
end #function update-1d

# Update 2D
function updatesolution!(rhs, Q::NamedTuple{S, NTuple{4, T}}, metric, ω, elems, rka, rkb, dt, advection) where {S, T}
    #Save original velocity
    if (advection)
        ρ = Q.ρ
        u = Q.U ./ ρ
        v = Q.V ./ ρ
        energy = Q.E ./ ρ
    end

    J = metric.J
    M = reshape(kron(ω, ω), length(ω), length(ω))
    for (rhsq, q) ∈ zip(rhs, Q)
        for e ∈ elems
            q[:, :, e] += rkb * dt * rhsq[:, :, e] ./ (M .* J[:, :, e])
            rhsq[:, :, e] *= rka
        end
    end
    #Reset velocity
    if (advection)
        Q.U .= Q.ρ .* u
        Q.V .= Q.ρ .* v
        Q.E .= Q.ρ .* energy
    end
end #function update-2d

# Update 3D
function updatesolution!(rhs, Q::NamedTuple{S, NTuple{5, T}}, metric, ω, elems, rka, rkb, dt, advection) where {S, T}
    #Save original velocity
    if (advection)
        ρ = Q.ρ
        u = Q.U ./ ρ
        v = Q.V ./ ρ
        w = Q.W ./ ρ
        energy = Q.E ./ ρ
    end

    J = metric.J
    M = reshape(kron(ω, ω, ω), length(ω), length(ω), length(ω))
    for (rhsq, q) ∈ zip(rhs, Q)
        for e ∈ elems
            q[:, :, :, e] += rkb * dt * rhsq[:, :, :, e] ./ (M .* J[:, :, :, e])
            rhsq[:, :, :, e] *= rka
        end
    end
    #Reset velocity
    if (advection)
        Q.U .= Q.ρ .* u
        Q.V .= Q.ρ .* v
        Q.W .= Q.ρ .* w
        Q.E .= Q.ρ .* energy
    end
end #function update-3d

# ### Compute L2 Error Norm for:
# 1D Error
function L2energy(Q::NamedTuple{S, NTuple{3, T}}, metric, ω, elems) where {S, T}
  J = metric.J
  Nq = length(ω)
  M = ω
  index = CartesianIndices(ntuple(j->1:Nq, Val(1)))

  energy = [zero(J[1])]
  for q ∈ Q
    for e ∈ elems
      for ind ∈ index
        energy[1] += M[ind] * J[ind, e] * q[ind, e]^2
      end
    end
  end
  energy[1]
end #end function L2energy-1d

# 2D Error
function L2energy(Q::NamedTuple{S, NTuple{4, T}}, metric, ω, elems) where {S, T}
  J = metric.J
  Nq = length(ω)
  M = reshape(kron(ω, ω), Nq, Nq)
  index = CartesianIndices(ntuple(j->1:Nq, Val(2)))

  energy = [zero(J[1])]
  for q ∈ Q
    for e ∈ elems
      for ind ∈ index
        energy[1] += M[ind] * J[ind, e] * q[ind, e]^2
      end
    end
  end
  energy[1]
end #end function L2energy-2d

# 3D Error
function L2energy(Q::NamedTuple{S, NTuple{5, T}}, metric, ω, elems) where {S, T}
  J = metric.J
  Nq = length(ω)
  M = reshape(kron(ω, ω, ω), Nq, Nq, Nq)
  index = CartesianIndices(ntuple(j->1:Nq, Val(3)))

  energy = [zero(J[1])]
  for q ∈ Q
    for e ∈ elems
      for ind ∈ index
        energy[1] += M[ind] * J[ind, e] * q[ind, e]^2
      end
    end
  end
  energy[1]
end #function L2energy-3d

#-------------------------------------------------------------------------------#
#--------End Volume, Flux, Update, Error Functions for Multiple Dispatch--------#
#-------------------------------------------------------------------------------#

# ### Compute how many MPI neighbors we have
# "mesh.nabrtorank" stands for "Neighbors to rank"
numnabr = length(mesh.nabrtorank)

# ### Create send/recv request arrays
# "sendreq" is the array that we use to send the communication request. It needs to be of the same length as the number of neighboring ranks. Similarly, "recvreq" is the array that we use to receive the neighboring rank information.
sendreq = fill(MPI.REQUEST_NULL, numnabr)
recvreq = fill(MPI.REQUEST_NULL, numnabr)

# ### Create send/recv buffer
# The dimensions of these arrays are (1) degrees of freedom within an element, (2) number of solution vectors, and (3) the number of "send elements" and "ghost elements", respectively.
sendQ = Array{DFloat, 3}(undef, (N+1)^dim, length(Q), length(mesh.sendelems))
recvQ = Array{DFloat, 3}(undef, (N+1)^dim, length(Q), length(mesh.ghostelems))

# Build CartesianIndex map for moving between Cartesian and linear storage of
# dofs
index = CartesianIndices(ntuple(j->1:N+1, dim))
nrealelem = length(mesh.realelems)

# ### Dump the initial condition
# Dump out the initial conditin to VTK prior to entering the time-step loop.
include("vtk.jl")
temp=Q.E
if (icase == 6 || icase == 7)
    temp=( Q.E ./  Q.ρ ) .- 300.0
end
writemesh(@sprintf("Euler%dD_rank_%04d_step_%05d", dim, mpirank, 0),
          coord...; fields=(("ρ", Q.ρ),("U", Q.U),("E", temp)), realelems=mesh.realelems)

# ### Begin Time-step loop
# Go through nsteps time-steps and for each time-step, loop through the s-stages of the explicit RK method.
for step = 1:nsteps
    mpirank == 0 && @show step
    for s = 1:length(RKA)
        # #### Post MPI receives
        # We assume that an MPI_Isend has been posted (non-blocking send) and are waiting to receive any message that has
        # been posted for receiving.  We are looping through the : (1) number of neighbors, (2) neighbor ranks,
        # and (3) neighbor elements.
        for (nnabr, nabrrank, nabrelem) ∈ zip(1:numnabr, mesh.nabrtorank,
                                              mesh.nabrtorecv)
            recvreq[nnabr] = MPI.Irecv!((@view recvQ[:, :, nabrelem]), nabrrank, 777,
                                        mpicomm)
        end

        # #### Wait on (prior) MPI sends
        # WE assume that non-blocking sends have been sent and wait for this to happen. FXG: Why do we need to wait?
        MPI.Waitall!(sendreq)

        # #### Pack data to send buffer
        # For all faces "nf" and all elements "ne" we pack the send data.
        for (ne, e) ∈ enumerate(mesh.sendelems)
            for (nf, f) ∈ enumerate(Q)
                sendQ[:, nf, ne] = f[index[:], e]
            end
        end

        # #### Post MPI sends
        # For all: (1) number of neighbors, (2) neighbor ranks, and (3) neighbor elements we perform a non-blocking send.
        for (nnabr, nabrrank, nabrelem) ∈ zip(1:numnabr, mesh.nabrtorank,
                                              mesh.nabrtosend)
            sendreq[nnabr] = MPI.Isend((@view sendQ[:, :, nabrelem]), nabrrank, 777,
                                       mpicomm)
        end

        # #### Compute Pressure
        # Depending on which Equation Set we are using, compute the pressure for all dimensions
        # call compute_pressure
        compute_pressure!(Pressure, Q, γ, eqn_set)

        # #### Compute RHS Volume Integral
        # Note that it is not necessary to have received all the MPI messages. Here we are interleaving computation
        # with communication in order to curtail latency.  Here we perform the RHS volume integrals.
        # call volumerhs
        volumerhs!(rhs, Q, Pressure, metric, D, ω, mesh.realelems, gravity, γ, eqn_set)

        # #### Wait on MPI receives
        # We need to wait to receive the messages before we move on to t=e flux integrals.
        MPI.Waitall!(recvreq)

        # #### Unpack data from receive buffer
        # The inverse of the Pack datat to send buffer. We now unpack the receive buffer in order to use it in the RHS
        # flux integral.
        for elems ∈ mesh.nabrtorecv
            for (nf, f) ∈ enumerate(Q)
                f[index[:], nrealelem .+ elems] = recvQ[:, nf, elems]
            end
        end

        # #### Compute RHS Flux Integral
        # We compute the flux integral on all "realelems" which are the elements owned by the current mpirank.
        # call fluxrhs
        fluxrhs!(rhs, Q, Pressure, metric, ω, mesh.realelems, mesh.elemtobndy, vmapM, vmapP, γ, eqn_set)

        # #### Update solution and scale RHS
        # We need to update/evolve the solution in time and multiply by the inverse mass matrix.
        #call updatesolution
        updatesolution!(rhs, Q, metric, ω, mesh.realelems, RKA[s%length(RKA)+1], RKB[s], dt, advection)
    end #s-stages

    # #### Write VTK Output
    # After each time-step, we dump out VTK data for Paraview/VisIt.
    if (mod(step,iplot) == 0)
        temp=Q.E
        if (icase == 6 || icase == 7)
            temp=( Q.E ./  Q.ρ ) .- 300.0
        end
        writemesh(@sprintf("Euler%dD_rank_%04d_step_%05d", dim, mpirank, step),
                  coord...; fields=(("ρ", Q.ρ),("U", Q.U),("E", temp)), realelems=mesh.realelems)
    end
end #step

# ### Compute L2 Error Norms
# Since we stored the initial condition, we can now compute the L2 error norms for both the solution and energy.

#extract velocity fields
if dim == 1
    Q.U .= Q.U ./ Q.ρ
    Q.E .= Q.E ./ Q.ρ
    Δ.U .= Δ.U ./ Δ.ρ
    Δ.E .= Δ.E ./ Δ.ρ
elseif dim == 2
    Q.U .= Q.U ./ Q.ρ
    Q.V .= Q.V ./ Q.ρ
    Q.E .= Q.E ./ Q.ρ
    Δ.U .= Δ.U ./ Δ.ρ
    Δ.V .= Δ.V ./ Δ.ρ
    Δ.E .= Δ.E ./ Δ.ρ
elseif dim == 3
    Q.U .= Q.U ./ Q.ρ
    Q.V .= Q.V ./ Q.ρ
    Q.W .= Q.W ./ Q.ρ
    Q.E .= Q.E ./ Q.ρ
    Δ.U .= Δ.U ./ Δ.ρ
    Δ.V .= Δ.V ./ Δ.ρ
    Δ.W .= Δ.W ./ Δ.ρ
    Δ.E .= Δ.E ./ Δ.ρ
end
if (icase == 6 || icase == 7)
    Q.E .= Q.E .- 300.0
    Δ.E .= Δ.E .- 300.0
end
#Compute Norms
for (δ, q) ∈ zip(Δ, Q)
    δ .-= q
end
eng = L2energy(Q, metric, ω, mesh.realelems)
eng = MPI.Allreduce(eng, MPI.SUM, mpicomm)
mpirank == 0 && @show sqrt(eng)

err = L2energy(Δ, metric, ω, mesh.realelems)
err = MPI.Allreduce(err, MPI.SUM, mpicomm)
mpirank == 0 && @show sqrt(err)

# ###Output the polynomial order, space dimensions, and element configuration
println("N= ",N)
println("dim= ",dim)
println("Ne= ",brickN)
println("DFloat= ",DFloat)
println("gravity= ",gravity)
println("icase= ",icase)
println("u0= ",u0)
println("eqn_set= ",eqn_set)
println("dt= ",dt)
println("Time Final= ",tend)
println("warp_grid= ",warp_grid)
println("iplot= ",iplot)

nothing

#-
#md # ## [Plain Program](@id euler-plain-program)
#md #
#md # Below follows a version of the program without any comments.
#md # The file is also available here: [euler.jl](euler.jl)
#md #
#md # ```julia
#md # @__CODE__
#md # ```

