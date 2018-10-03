var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": "DocTestSetup = :(using Canary)"
},

{
    "location": "#Canary.jl-1",
    "page": "Home",
    "title": "Canary.jl",
    "category": "section",
    "text": "An exploration of discontinuous Galerkin methods in Julia."
},

{
    "location": "#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "To install, run the following commands in the Julia REPL:] add \"https://github.com/climate-machine/Canary.jl\"and then runusing Canaryto load the package.If you are using some of the MPI based functions at the REPL you will also need to load MPI with something likeusing MPI\nMPI.Init()\nMPI.finalize_atexit()If you have problems building MPI.jl try explicitly providing the MPI compiler wrappers through the CC and FC environment variables.  Something like the following at your Bourne shell prompt:export CC=mpicc\nexport FC=mpif90Then launch Julia rebuild MPI.jl with] build MPI"
},

{
    "location": "#Building-Documentation-1",
    "page": "Home",
    "title": "Building Documentation",
    "category": "section",
    "text": "You may build the documentation locally by running\njulia --color=yes --project=docs/ -e \'using Pkg; Pkg.instantiate()\'\n\njulia --color=yes --project=docs/ docs/make.jl\nwhere first invocation of julia only needs to be run the first time."
},

{
    "location": "manual/dg_intro/#",
    "page": "Introduction to DG",
    "title": "Introduction to DG",
    "category": "page",
    "text": ""
},

{
    "location": "manual/dg_intro/#Introduction-to-DG-1",
    "page": "Introduction to DG",
    "title": "Introduction to DG",
    "category": "section",
    "text": ""
},

{
    "location": "manual/mesh/#",
    "page": "Mesh",
    "title": "Mesh",
    "category": "page",
    "text": "DocTestSetup = :(using Canary)"
},

{
    "location": "manual/mesh/#Mesh-1",
    "page": "Mesh",
    "title": "Mesh",
    "category": "section",
    "text": ""
},

{
    "location": "manual/metric/#",
    "page": "Metric Terms",
    "title": "Metric Terms",
    "category": "page",
    "text": "DocTestSetup = :(using Canary)"
},

{
    "location": "manual/metric/#Metric-Terms-1",
    "page": "Metric Terms",
    "title": "Metric Terms",
    "category": "section",
    "text": ""
},

{
    "location": "manual/operators/#",
    "page": "Element Operators",
    "title": "Element Operators",
    "category": "page",
    "text": "DocTestSetup = :(using Canary)"
},

{
    "location": "manual/operators/#Element-Operators-1",
    "page": "Element Operators",
    "title": "Element Operators",
    "category": "section",
    "text": ""
},

{
    "location": "examples/generated/advection/#",
    "page": "Advection Equation Example",
    "title": "Advection Equation Example",
    "category": "page",
    "text": "EditURL = \"https://github.com/climate-machine/Canary.jl/blob/master/examples/advection.jl\""
},

{
    "location": "examples/generated/advection/#Advection-Equation-Example-1",
    "page": "Advection Equation Example",
    "title": "Advection Equation Example",
    "category": "section",
    "text": "(Image: )"
},

{
    "location": "examples/generated/advection/#Introduction-1",
    "page": "Advection Equation Example",
    "title": "Introduction",
    "category": "section",
    "text": "This example shows how to solve the Advection or Transport equation in 1D, 2D, and 3D."
},

{
    "location": "examples/generated/advection/#Continuous-Governing-Equations-1",
    "page": "Advection Equation Example",
    "title": "Continuous Governing Equations",
    "category": "section",
    "text": "We solve the following equation:fracpartial rhopartial t + nabla cdot left( rho mathbfu right) = 0   (1)where mathbfu=(uvw) depending on how many spatial dimensions we are using.  We only solve passive transport here so mathbfu is given for all time and does not change. We employ periodic boundary conditions across all the walls of the domain."
},

{
    "location": "examples/generated/advection/#Discontinous-Galerkin-Method-1",
    "page": "Advection Equation Example",
    "title": "Discontinous Galerkin Method",
    "category": "section",
    "text": "To solve Eq. (1) in one, two, and three dimensions we use the Discontinuous Galerkin method with basis functions comprised of tensor products of one-dimensional Lagrange polynomials based on Lobatto points. Multiplying Eq. (1) by a test function psi and integrating within each element Omega_e such that Omega = bigcup_e=1^N_e Omega_e we getint_Omega_e psi fracpartial rho^(e)_Npartial t dOmega_e + int_Omega_e psi nabla cdot left( rho mathbfu right)^(e)_N dOmega_e = 0   (2)where rho^(e)_N=sum_i=1^(N+1)^dim psi_i(mathbfx) rho_i(t) is the finite dimensional expansion with basis functions psi(mathbfx).  Integrating Eq. (2) by parts yieldsint_Omega_e psi fracpartial rho^(e)_Npartial t dOmega_e + int_Gamma_e psi mathbfn cdot left( rho mathbfu right)^(*e)_N dGamma_e - int_Omega_e nabla psi cdot left( rho mathbfu right)^(e)_N dOmega_e = 0   (3)where the second term on the left denotes the flux integral term (computed in \"function fluxrhs\") and the third term denotes the volume integral term (computed in \"function volumerhs\").  The superscript (*e) in the flux integral term denotes the numerical flux. Here we use the Rusanov flux."
},

{
    "location": "examples/generated/advection/#Commented-Program-1",
    "page": "Advection Equation Example",
    "title": "Commented Program",
    "category": "section",
    "text": ""
},

{
    "location": "examples/generated/advection/#Define-key-parameters:-1",
    "page": "Advection Equation Example",
    "title": "Define key parameters:",
    "category": "section",
    "text": "N is polynomial order and brickN(Ne) generates a brick-grid with Ne elements in each directionN = 4 #polynomial order\n#brickN = (10) #1D brickmesh\n#brickN = (10, 10) #2D brickmesh\nbrickN = (10, 10, 10) #3D brickmesh\nDFloat = Float64 #Number Type\ntend = DFloat(0.01) #Final Time"
},

{
    "location": "examples/generated/advection/#Load-the-MPI-and-Canary-packages-where-Canary-builds-the-mesh,-generates-basis-functions,-and-metric-terms.-1",
    "page": "Advection Equation Example",
    "title": "Load the MPI and Canary packages where Canary builds the mesh, generates basis functions, and metric terms.",
    "category": "section",
    "text": "using MPI\nusing Canary\nusing Printf: @sprintf"
},

{
    "location": "examples/generated/advection/#The-grid-that-we-create-determines-the-number-of-spatial-dimensions-that-we-are-going-to-use.-1",
    "page": "Advection Equation Example",
    "title": "The grid that we create determines the number of spatial dimensions that we are going to use.",
    "category": "section",
    "text": "dim = length(brickN)###Output the polynomial order, space dimensions, and element configurationprintln(\"N= \",N)\nprintln(\"dim= \",dim)\nprintln(\"brickN= \",brickN)\nprintln(\"DFloat= \",DFloat)"
},

{
    "location": "examples/generated/advection/#Initialize-MPI-and-get-the-communicator,-rank,-and-size-1",
    "page": "Advection Equation Example",
    "title": "Initialize MPI and get the communicator, rank, and size",
    "category": "section",
    "text": "MPI.Initialized() || MPI.Init() # only initialize MPI if not initialized\nMPI.finalize_atexit()\nmpicomm = MPI.COMM_WORLD\nmpirank = MPI.Comm_rank(mpicomm)\nmpisize = MPI.Comm_size(mpicomm)"
},

{
    "location": "examples/generated/advection/#Generate-a-local-view-of-a-fully-periodic-Cartesian-mesh.-1",
    "page": "Advection Equation Example",
    "title": "Generate a local view of a fully periodic Cartesian mesh.",
    "category": "section",
    "text": "if dim == 1\n  (Nx, ) = brickN\n  local x = range(DFloat(0); length=Nx+1, stop=1)\n  mesh = brickmesh((x, ), (true, ); part=mpirank+1, numparts=mpisize)\nelseif dim == 2\n  (Nx, Ny) = brickN\n  local x = range(DFloat(0); length=Nx+1, stop=1)\n  local y = range(DFloat(0); length=Ny+1, stop=1)\n  mesh = brickmesh((x, y), (true, true); part=mpirank+1, numparts=mpisize)\nelse\n  (Nx, Ny, Nz) = brickN\n  local x = range(DFloat(0); length=Nx+1, stop=1)\n  local y = range(DFloat(0); length=Ny+1, stop=1)\n  local z = range(DFloat(0); length=Nz+1, stop=1)\n  mesh = brickmesh((x, y, z), (true, true, true);\n                   part=mpirank+1, numparts=mpisize)\nend"
},

{
    "location": "examples/generated/advection/#Partition-the-mesh-using-a-Hilbert-curve-based-partitioning-1",
    "page": "Advection Equation Example",
    "title": "Partition the mesh using a Hilbert curve based partitioning",
    "category": "section",
    "text": "mesh = partition(mpicomm, mesh...)"
},

{
    "location": "examples/generated/advection/#Connect-the-mesh-in-parallel-1",
    "page": "Advection Equation Example",
    "title": "Connect the mesh in parallel",
    "category": "section",
    "text": "mesh = connectmesh(mpicomm, mesh...)"
},

{
    "location": "examples/generated/advection/#Get-the-degrees-of-freedom-along-the-faces-of-each-element.-1",
    "page": "Advection Equation Example",
    "title": "Get the degrees of freedom along the faces of each element.",
    "category": "section",
    "text": "vmap(:,f,e) gives the list of local (mpirank) points for the face \"f\" of element \"e\".  vmapP points to the outward (or neighbor) element and vmapM for the current element. P=+ or right and M=- or left.(vmapM, vmapP) = mappings(N, mesh.elemtoelem, mesh.elemtoface, mesh.elemtoordr)"
},

{
    "location": "examples/generated/advection/#Create-1-D-operators-1",
    "page": "Advection Equation Example",
    "title": "Create 1-D operators",
    "category": "section",
    "text": "xiand omega are the 1D Lobatto points and weights and D is the derivative of the basis function.(ξ, ω) = lglpoints(DFloat, N)\nD = spectralderivative(ξ)"
},

{
    "location": "examples/generated/advection/#Compute-metric-terms-1",
    "page": "Advection Equation Example",
    "title": "Compute metric terms",
    "category": "section",
    "text": "nface and nelem refers to the total number of faces and elements for this MPI rank. Also, coord contains the dim-tuple coordinates in the mesh.(nface, nelem) = size(mesh.elemtoelem)\ncoord = creategrid(Val(dim), mesh.elemtocoord, ξ)\nif dim == 1\n  x = coord.x\n  for j = 1:length(x)\n    x[j] = x[j]\n  end\nelseif dim == 2\n  (x, y) = (coord.x, coord.y)\n  for j = 1:length(x)\n    (x[j], y[j]) = (x[j] .+ sin.(π * x[j]) .* sin.(2 * π * y[j]) / 10,\n                    y[j] .+ sin.(2 * π * x[j]) .* sin.(π * y[j]) / 10)\n  end\nelseif dim == 3\n  (x, y, z) = (coord.x, coord.y, coord.z)\n  for j = 1:length(x)\n    (x[j], y[j], z[j]) = (x[j] + (sin(π * x[j]) * sin(2 * π * y[j]) *\n                                  cos(2 * π * z[j])) / 10,\n                          y[j] + (sin(π * y[j]) * sin(2 * π * x[j]) *\n                                  cos(2 * π * z[j])) / 10,\n                          z[j] + (sin(π * z[j]) * sin(2 * π * x[j]) *\n                                  cos(2 * π * y[j])) / 10)\n  end\nend"
},

{
    "location": "examples/generated/advection/#First-VTK-Call-1",
    "page": "Advection Equation Example",
    "title": "First VTK Call",
    "category": "section",
    "text": "This first VTK call dumps the mesh out for all mpiranks.include(joinpath(@__DIR__, \"vtk.jl\"))\nwritemesh(@sprintf(\"Advection%dD_rank_%04d_mesh\", dim, mpirank), coord...;\n          realelems=mesh.realelems)"
},

{
    "location": "examples/generated/advection/#Compute-the-metric-terms-1",
    "page": "Advection Equation Example",
    "title": "Compute the metric terms",
    "category": "section",
    "text": "This call computes the metric terms of the grid such as xi_mathbfx, eta_mathbfx, zeta_mathbfx for all spatial dimensions mathbfx depending on the dimension of dim.metric = computemetric(coord..., D)"
},

{
    "location": "examples/generated/advection/#Generate-the-State-Vectors-1",
    "page": "Advection Equation Example",
    "title": "Generate the State Vectors",
    "category": "section",
    "text": "We need to create as many velocity vectors as there are dimensions.if dim == 1\n  statesyms = (:ρ, :Ux)\nelseif dim == 2\n  statesyms = (:ρ, :Ux, :Uy)\nelseif dim == 3\n  statesyms = (:ρ, :Ux, :Uy, :Uz)\nend"
},

{
    "location": "examples/generated/advection/#Create-storage-for-state-vector-and-right-hand-side-1",
    "page": "Advection Equation Example",
    "title": "Create storage for state vector and right-hand side",
    "category": "section",
    "text": "Q holds the solution vector and rhs the rhs-vector which are dim+1 tuples In addition, here we generate the initial conditionsQ   = NamedTuple{statesyms}(ntuple(j->zero(coord.x), length(statesyms)))\nrhs = NamedTuple{statesyms}(ntuple(j->zero(coord.x), length(statesyms)))\nif dim == 1\n  Q.ρ .= sin.(2 * π * x)\n  Q.Ux .= 1\nelseif dim == 2\n  Q.ρ .= sin.(2 * π * x) .* sin.(2 *  π * y)\n  Q.Ux .= 1\n  Q.Uy .= 1\nelseif dim == 3\n  Q.ρ .= sin.(2 * π * x) .* sin.(2 *  π * y) .* sin.(2 * π * z)Q.ρ .= sin.(2 * π * x) .* sin.(2 *  π * y)  Q.Ux .= 1\n  Q.Uy .= 1\n  Q.Uz .= 1\nend"
},

{
    "location": "examples/generated/advection/#Compute-the-time-step-size-and-number-of-time-steps-1",
    "page": "Advection Equation Example",
    "title": "Compute the time-step size and number of time-steps",
    "category": "section",
    "text": "Compute a Delta t such that the Courant number is 1. This is done for each mpirank and then we do an MPI_Allreduce to find the global minimum.dt = [floatmax(DFloat)]\nif dim == 1\n    (ξx) = (metric.ξx)\n    (Ux) = (Q.Ux)\n    for n = 1:length(Ux)\n        loc_dt = 2 ./ (abs.(Ux[n] * ξx[n]))\n        dt[1] = min(dt[1], loc_dt)\n    end\nelseif dim == 2\n    (ξx, ξy, ηx, ηy) = (metric.ξx, metric.ξy, metric.ηx, metric.ηy)\n    (Ux, Uy) = (Q.Ux, Q.Uy)\n    for n = 1:length(Ux)\n        loc_dt = 2 ./ max(abs.(Ux[n] * ξx[n] + Uy[n] * ξy[n]),\n                          abs.(Ux[n] * ηx[n] + Uy[n] * ηy[n]))\n        dt[1] = min(dt[1], loc_dt)\n    end\nelseif dim == 3\n    (ξx, ξy, ξz) = (metric.ξx, metric.ξy, metric.ξz)\n    (ηx, ηy, ηz) = (metric.ηx, metric.ηy, metric.ηz)\n    (ζx, ζy, ζz) = (metric.ζx, metric.ζy, metric.ζz)\n    (Ux, Uy, Uz) = (Q.Ux, Q.Uy, Q.Uz)\n    for n = 1:length(Ux)\n        loc_dt = 2 ./ max(abs.(Ux[n] * ξx[n] + Uy[n] * ξy[n] + Uz[n] * ξz[n]),\n                          abs.(Ux[n] * ηx[n] + Uy[n] * ηy[n] + Uz[n] * ηz[n]),\n                          abs.(Ux[n] * ζx[n] + Uy[n] * ζy[n] + Uz[n] * ζz[n]))\n        dt[1] = min(dt[1], loc_dt)\n    end\nend\ndt = MPI.Allreduce(dt[1], MPI.MIN, mpicomm)\ndt = DFloat(dt / N^sqrt(2))\nnsteps = ceil(Int64, tend / dt)\ndt = tend / nsteps\n@show (dt, nsteps)"
},

{
    "location": "examples/generated/advection/#Compute-the-exact-solution-at-the-final-time.-1",
    "page": "Advection Equation Example",
    "title": "Compute the exact solution at the final time.",
    "category": "section",
    "text": "Later Δ will be used to store the difference between the exact and computed solutions.Δ   = NamedTuple{statesyms}(ntuple(j->zero(coord.x), length(statesyms)))\nif dim == 1\n  Δ.ρ .= sin.(2 * π * (x - tend * Q.Ux))\n  Δ.Ux .=  Q.Ux\nelseif dim == 2\n  Δ.ρ .= sin.(2 * π * (x - tend * Q.Ux)) .* sin.(2 *  π * (y - tend * Q.Uy))\n  Δ.Ux .=  Q.Ux\n  Δ.Uy .=  Q.Uy\nelseif dim == 3\n  Δ.ρ .= sin.(2 * π * (x - tend * Q.Ux)) .* sin.(2 *  π * (y - tend * Q.Uy)) .* sin.(2 * π * (z - tend * Q.Uz))Δ.ρ .= sin.(2 * π * (x - tend * Q.Ux)) .* sin.(2 *  π * (y - tend * Q.Uy))  Δ.Ux .=  Q.Ux\n  Δ.Uy .=  Q.Uy\n  Δ.Uz .=  Q.Uz\nend"
},

{
    "location": "examples/generated/advection/#Store-Explicit-RK-Time-stepping-Coefficients-1",
    "page": "Advection Equation Example",
    "title": "Store Explicit RK Time-stepping Coefficients",
    "category": "section",
    "text": "We use the fourth-order, low-storage, Runge–Kutta scheme of Carpenter and Kennedy (1994) ((5,4) 2N-Storage RK scheme.Ref: @TECHREPORT{CarpenterKennedy1994,   author = {M.~H. Carpenter and C.~A. Kennedy},   title = {Fourth-order {2N-storage} {Runge-Kutta} schemes},   institution = {National Aeronautics and Space Administration},   year = {1994},   number = {NASA TM-109112},   address = {Langley Research Center, Hampton, VA}, }RKA = (DFloat(0),\n       DFloat(-567301805773)  / DFloat(1357537059087),\n       DFloat(-2404267990393) / DFloat(2016746695238),\n       DFloat(-3550918686646) / DFloat(2091501179385),\n       DFloat(-1275806237668) / DFloat(842570457699 ))\n\nRKB = (DFloat(1432997174477) / DFloat(9575080441755 ),\n       DFloat(5161836677717) / DFloat(13612068292357),\n       DFloat(1720146321549) / DFloat(2090206949498 ),\n       DFloat(3134564353537) / DFloat(4481467310338 ),\n       DFloat(2277821191437) / DFloat(14882151754819))\n\nRKC = (DFloat(0),\n       DFloat(1432997174477) / DFloat(9575080441755),\n       DFloat(2526269341429) / DFloat(6820363962896),\n       DFloat(2006345519317) / DFloat(3224310063776),\n       DFloat(2802321613138) / DFloat(2924317926251))"
},

{
    "location": "examples/generated/advection/#Volume-RHS-Routines-1",
    "page": "Advection Equation Example",
    "title": "Volume RHS Routines",
    "category": "section",
    "text": "These functions solve the volume term int_Omega_e nabla psi cdot left( rho mathbfu right)^(e)_N for: Volume RHS for 1Dfunction volumerhs!(rhs, Q::NamedTuple{S, NTuple{2, T}}, metric, D, ω,\n                    elems) where {S, T}\n  rhsρ = rhs.ρ\n  (ρ, Ux) = (Q.ρ, Q.Ux)\n  Nq = size(ρ, 1)\n  J = metric.J\n  ξx = metric.ξx\n  for e ∈ elemsloop of ξ-grid lines      rhsρ[:,e] += D\' * (ω .* J[:,e] .* ρ[:,e] .* (ξx[:,e] .* Ux[:,e]))\n  end #e ∈ elems\nend #function volumerhs-1dVolume RHS for 2Dfunction volumerhs!(rhs, Q::NamedTuple{S, NTuple{3, T}}, metric, D, ω,\n                    elems) where {S, T}\n    rhsρ = rhs.ρ\n    (ρ, Ux, Uy) = (Q.ρ, Q.Ux, Q.Uy)\n    Nq = size(ρ, 1)\n    J = metric.J\n    (ξx, ηx, ξy, ηy) = (metric.ξx, metric.ηx, metric.ξy, metric.ηy)\n    for e ∈ elemsloop of ξ-grid lines        for j = 1:Nq\n            rhsρ[:, j, e] +=\n                D\' * (ω[j] * ω .* J[:, j, e].* ρ[:, j, e] .*\n                      (ξx[:, j, e] .* Ux[:, j, e] + ξy[:, j, e] .* Uy[:, j, e]))\n        end #jloop of η-grid lines        for i = 1:Nq\n            rhsρ[i, :, e] +=\n                D\' * (ω[i] * ω .* J[i, :, e].* ρ[i, :, e] .*\n                      (ηx[i, :, e] .* Ux[i, :, e] + ηy[i, :, e] .* Uy[i, :, e]))\n        end #i\n    end #e ∈ elems\nend #function volumerhs-2dVolume RHS for 3Dfunction volumerhs!(rhs, Q::NamedTuple{S, NTuple{4, T}}, metric, D, ω,\n                    elems) where {S, T}\n    rhsρ = rhs.ρ\n    (ρ, Ux, Uy, Uz) = (Q.ρ, Q.Ux, Q.Uy, Q.Uz)\n    Nq = size(ρ, 1)\n    J = metric.J\n    (ξx, ηx, ζx) = (metric.ξx, metric.ηx, metric.ζx)\n    (ξy, ηy, ζy) = (metric.ξy, metric.ηy, metric.ζy)\n    (ξz, ηz, ζz) = (metric.ξz, metric.ηz, metric.ζz)\n    for e ∈ elemsloop of ξ-grid lines        for k = 1:Nq\n            for j = 1:Nq\n                rhsρ[:, j, k, e] +=\n                    D\' * (ω[j] * ω[k] * ω .* J[:, j, k, e] .* ρ[:, j, k, e] .*\n                          (ξx[:, j, k, e] .* Ux[:, j, k, e] +\n                           ξy[:, j, k, e] .* Uy[:, j, k, e] +\n                           ξz[:, j, k, e] .* Uz[:, j, k, e]))\n            end #j\n        end #kloop of η-grid lines        for k = 1:Nq\n            for i = 1:Nq\n                rhsρ[i, :, k, e] +=\n                    D\' * (ω[i] * ω[k] * ω .* J[i, :, k, e] .* ρ[i, :, k, e] .*\n                          (ηx[i, :, k, e] .* Ux[i, :, k, e] +\n                           ηy[i, :, k, e] .* Uy[i, :, k, e] +\n                           ηz[i, :, k, e] .* Uz[i, :, k, e]))\n            end #i\n        end #kloop of ζ-grid lines        for j = 1:Nq\n            for i = 1:Nq\n                rhsρ[i, j, :, e] +=\n                    D\' * (ω[i] * ω[j] * ω .* J[i, j, :, e] .* ρ[i, j, :, e] .*\n                          (ζx[i, j, :, e] .* Ux[i, j, :, e] +\n                           ζy[i, j, :, e] .* Uy[i, j, :, e] +\n                           ζz[i, j, :, e] .* Uz[i, j, :, e]))\n            end #i\n        end #j\n    end #e ∈ elems\nend"
},

{
    "location": "examples/generated/advection/#Flux-RHS-Routines-1",
    "page": "Advection Equation Example",
    "title": "Flux RHS Routines",
    "category": "section",
    "text": "These functions solve the flux integral term int_Gamma_e psi mathbfn cdot left( rho mathbfu right)^(*e)_N for: Flux RHS for 1Dfunction fluxrhs!(rhs, Q::NamedTuple{S, NTuple{2, T}}, metric, ω, elems, vmapM,\n                  vmapP) where {S, T}\n    rhsρ = rhs.ρ\n    (ρ, Ux) = (Q.ρ, Q.Ux)\n    nface = 2\n    (nx, sJ) = (metric.nx, metric.sJ)\n    nx = reshape(nx, size(vmapM))\n    sJ = reshape(sJ, size(vmapM))\n\n    for e ∈ elems\n        for f ∈ 1:nface\n            ρM = ρ[vmapM[1, f, e]]\n            UxM = Ux[vmapM[1, f, e]]\n            FxM = ρM .* UxM\n\n            ρP = ρ[vmapP[1, f, e]]\n            UxP = Ux[vmapP[1, f, e]]\n            FxP = ρP .* UxP\n\n            nxM = nx[1, f, e]\n            λ = max.(abs.(nxM .* UxM), abs.(nxM .* UxP))\n\n            F = (nxM .* (FxM + FxP) - λ .* (ρP - ρM)) / 2\n            rhsρ[vmapM[1, f, e]] -= sJ[1, f, e] .* F\n        end #for f ∈ 1:nface\n    end #e ∈ elems\nend #function fluxrhs-1dFlux RHS for 2Dfunction fluxrhs!(rhs, Q::NamedTuple{S, NTuple{3, T}}, metric, ω, elems, vmapM,\n                  vmapP) where {S, T}\n    rhsρ = rhs.ρ\n    (ρ, Ux, Uy) = (Q.ρ, Q.Ux, Q.Uy)\n    nface = 4\n    (nx, ny, sJ) = (metric.nx, metric.ny, metric.sJ)\n    for e ∈ elems\n        for f ∈ 1:nface\n            ρM = ρ[vmapM[:, f, e]]\n            UxM = Ux[vmapM[:, f, e]]\n            UyM = Uy[vmapM[:, f, e]]\n            FxM = ρM .* UxM\n            FyM = ρM .* UyM\n\n            ρP = ρ[vmapP[:, f, e]]\n            UxP = Ux[vmapP[:, f, e]]\n            UyP = Uy[vmapP[:, f, e]]\n            FxP = ρP .* UxP\n            FyP = ρP .* UyP\n\n            nxM = nx[:, f, e]\n            nyM = ny[:, f, e]\n            λ = max.(abs.(nxM .* UxM + nyM .* UyM), abs.(nxM .* UxP + nyM .* UyP))\n\n            F = (nxM .* (FxM + FxP) + nyM .* (FyM + FyP) + λ .* (ρM - ρP)) / 2\n            rhsρ[vmapM[:, f, e]] -= ω .* sJ[:, f, e] .* F\n        end #f ∈ 1:nface\n    end #e ∈ elems\nend #function fluxrhs-2dFLux RHS for 3Dfunction fluxrhs!(rhs, Q::NamedTuple{S, NTuple{4, T}}, metric, ω, elems, vmapM,\n                  vmapP) where {S, T}\n    rhsρ = rhs.ρ\n    (ρ, Ux, Uy, Uz) = (Q.ρ, Q.Ux, Q.Uy, Q.Uz)\n    nface = 6\n    (nx, ny, nz, sJ) = (metric.nx, metric.ny, metric.nz, metric.sJ)\n    nx = reshape(nx, size(vmapM))\n    ny = reshape(ny, size(vmapM))\n    nz = reshape(nz, size(vmapM))\n    sJ = reshape(sJ, size(vmapM))\n    for e ∈ elems\n        for f ∈ 1:nface\n            ρM = ρ[vmapM[:, f, e]]\n            UxM = Ux[vmapM[:, f, e]]\n            UyM = Uy[vmapM[:, f, e]]\n            UzM = Uz[vmapM[:, f, e]]\n            FxM = ρM .* UxM\n            FyM = ρM .* UyM\n            FzM = ρM .* UzM\n\n            ρP = ρ[vmapP[:, f, e]]\n            UxP = Ux[vmapP[:, f, e]]\n            UyP = Uy[vmapP[:, f, e]]\n            UzP = Uz[vmapP[:, f, e]]\n            FxP = ρP .* UxP\n            FyP = ρP .* UyP\n            FzP = ρP .* UzP\n\n            nxM = nx[:, f, e]\n            nyM = ny[:, f, e]\n            nzM = nz[:, f, e]\n            λ = max.(abs.(nxM .* UxM + nyM .* UyM + nzM .* UzM),\n                     abs.(nxM .* UxP + nyM .* UyP + nzM .* UzP))\n\n            F = (nxM .* (FxM + FxP) + nyM .* (FyM + FyP) + nzM .* (FzM + FzP) +\n                 λ .* (ρM - ρP)) / 2\n            rhsρ[vmapM[:, f, e]] -= kron(ω, ω) .* sJ[:, f, e] .* F\n        end #f ∈ 1:nface\n    end #e ∈ elems\nend #function fluxrhs-3d"
},

{
    "location": "examples/generated/advection/#Update-the-solution-via-RK-Method-for:-1",
    "page": "Advection Equation Example",
    "title": "Update the solution via RK Method for:",
    "category": "section",
    "text": "Update 1Dfunction updatesolution!(rhs, Q::NamedTuple{S, NTuple{2, T}}, metric, ω, elems,\n                         rka, rkb, dt) where {S, T}\n    J = metric.J\n    M =  ω\n    for (rhsq, q) ∈ zip(rhs, Q)\n        for e ∈ elems\n            q[:, e] += rkb * dt * rhsq[:, e] ./ ( M .* J[:, e])\n            rhsq[:, e] *= rka\n        end\n    end\nendUpdate 2Dfunction updatesolution!(rhs, Q::NamedTuple{S, NTuple{3, T}}, metric, ω, elems,\n                         rka, rkb, dt) where {S, T}\n    J = metric.J\n    M = reshape(kron(ω, ω), length(ω), length(ω))\n    for (rhsq, q) ∈ zip(rhs, Q)\n        for e ∈ elems\n            q[:, :, e] += rkb * dt * rhsq[:, :, e] ./ (M .* J[:, :, e])\n            rhsq[:, :, e] *= rka\n        end\n    end\nendUpdate 3Dfunction updatesolution!(rhs, Q::NamedTuple{S, NTuple{4, T}}, metric, ω, elems,\n                         rka, rkb, dt) where {S, T}\n    J = metric.J\n    M = reshape(kron(ω, ω, ω), length(ω), length(ω), length(ω))\n    for (rhsq, q) ∈ zip(rhs, Q)\n        for e ∈ elems\n            q[:, :, :, e] += rkb * dt * rhsq[:, :, :, e] ./ (M .* J[:, :, :, e])\n            rhsq[:, :, :, e] *= rka\n        end\n    end\nend"
},

{
    "location": "examples/generated/advection/#Compute-L2-Error-Norm-for:-1",
    "page": "Advection Equation Example",
    "title": "Compute L2 Error Norm for:",
    "category": "section",
    "text": "1D Errorfunction L2energy(Q::NamedTuple{S, NTuple{2, T}}, metric, ω, elems) where {S, T}\n  J = metric.J\n  Nq = length(ω)\n  M = ω\n  index = CartesianIndices(ntuple(j->1:Nq, Val(1)))\n\n  energy = [zero(J[1])]\n  for q ∈ Q\n    for e ∈ elems\n      for ind ∈ index\n        energy[1] += M[ind] * J[ind, e] * q[ind, e]^2\n      end\n    end\n  end\n  energy[1]\nend2D Errorfunction L2energy(Q::NamedTuple{S, NTuple{3, T}}, metric, ω, elems) where {S, T}\n  J = metric.J\n  Nq = length(ω)\n  M = reshape(kron(ω, ω), Nq, Nq)\n  index = CartesianIndices(ntuple(j->1:Nq, Val(2)))\n\n  energy = [zero(J[1])]\n  for q ∈ Q\n    for e ∈ elems\n      for ind ∈ index\n        energy[1] += M[ind] * J[ind, e] * q[ind, e]^2\n      end\n    end\n  end\n  energy[1]\nend3D Errorfunction L2energy(Q::NamedTuple{S, NTuple{4, T}}, metric, ω, elems) where {S, T}\n  J = metric.J\n  Nq = length(ω)\n  M = reshape(kron(ω, ω, ω), Nq, Nq, Nq)\n  index = CartesianIndices(ntuple(j->1:Nq, Val(3)))\n\n  energy = [zero(J[1])]\n  for q ∈ Q\n    for e ∈ elems\n      for ind ∈ index\n        energy[1] += M[ind] * J[ind, e] * q[ind, e]^2\n      end\n    end\n  end\n  energy[1]\nend"
},

{
    "location": "examples/generated/advection/#Compute-how-many-MPI-neighbors-we-have-1",
    "page": "Advection Equation Example",
    "title": "Compute how many MPI neighbors we have",
    "category": "section",
    "text": "\"mesh.nabrtorank\" stands for \"Neighbors to rank\"numnabr = length(mesh.nabrtorank)"
},

{
    "location": "examples/generated/advection/#Create-send/recv-request-arrays-1",
    "page": "Advection Equation Example",
    "title": "Create send/recv request arrays",
    "category": "section",
    "text": "\"sendreq\" is the array that we use to send the communication request. It needs to be of the same length as the number of neighboring ranks. Similarly, \"recvreq\" is the array that we use to receive the neighboring rank information.sendreq = fill(MPI.REQUEST_NULL, numnabr)\nrecvreq = fill(MPI.REQUEST_NULL, numnabr)"
},

{
    "location": "examples/generated/advection/#Create-send/recv-buffer-1",
    "page": "Advection Equation Example",
    "title": "Create send/recv buffer",
    "category": "section",
    "text": "The dimensions of these arrays are (1) degrees of freedom within an element, (2) number of solution vectors, and (3) the number of \"send elements\" and \"ghost elements\", respectively.sendQ = Array{DFloat, 3}(undef, (N+1)^dim, length(Q), length(mesh.sendelems))\nrecvQ = Array{DFloat, 3}(undef, (N+1)^dim, length(Q), length(mesh.ghostelems))Build CartesianIndex map for moving between Cartesian and linear storage of dofsindex = CartesianIndices(ntuple(j->1:N+1, dim))\nnrealelem = length(mesh.realelems)"
},

{
    "location": "examples/generated/advection/#Dump-the-initial-condition-1",
    "page": "Advection Equation Example",
    "title": "Dump the initial condition",
    "category": "section",
    "text": "Dump out the initial conditin to VTK prior to entering the time-step loop.include(joinpath(@__DIR__, \"vtk.jl\"))\nwritemesh(@sprintf(\"Advection%dD_rank_%04d_step_%05d\", dim, mpirank, 0),\n          coord...; fields=((\"ρ\", Q.ρ),), realelems=mesh.realelems)"
},

{
    "location": "examples/generated/advection/#Begin-Time-step-loop-1",
    "page": "Advection Equation Example",
    "title": "Begin Time-step loop",
    "category": "section",
    "text": "Go through nsteps time-steps and for each time-step, loop through the s-stages of the explicit RK method.for step = 1:nsteps\n    mpirank == 0 && @show step\n    for s = 1:length(RKA)"
},

{
    "location": "examples/generated/advection/#Post-MPI-receives-1",
    "page": "Advection Equation Example",
    "title": "Post MPI receives",
    "category": "section",
    "text": "We assume that an MPI_Isend has been posted (non-blocking send) and are waiting to receive any message that has been posted for receiving.  We are looping through the : (1) number of neighbors, (2) neighbor ranks, and (3) neighbor elements.        for (nnabr, nabrrank, nabrelem) ∈ zip(1:numnabr, mesh.nabrtorank,\n                                              mesh.nabrtorecv)\n            recvreq[nnabr] = MPI.Irecv!((@view recvQ[:, :, nabrelem]), nabrrank, 777,\n                                        mpicomm)\n        end"
},

{
    "location": "examples/generated/advection/#Wait-on-(prior)-MPI-sends-1",
    "page": "Advection Equation Example",
    "title": "Wait on (prior) MPI sends",
    "category": "section",
    "text": "WE assume that non-blocking sends have been sent and wait for this to happen. FXG: Why do we need to wait?        MPI.Waitall!(sendreq)"
},

{
    "location": "examples/generated/advection/#Pack-data-to-send-buffer-1",
    "page": "Advection Equation Example",
    "title": "Pack data to send buffer",
    "category": "section",
    "text": "For all faces \"nf\" and all elements \"ne\" we pack the send data.        for (ne, e) ∈ enumerate(mesh.sendelems)\n            for (nf, f) ∈ enumerate(Q)\n                sendQ[:, nf, ne] = f[index[:], e]\n            end\n        end"
},

{
    "location": "examples/generated/advection/#Post-MPI-sends-1",
    "page": "Advection Equation Example",
    "title": "Post MPI sends",
    "category": "section",
    "text": "For all: (1) number of neighbors, (2) neighbor ranks, and (3) neighbor elements we perform a non-blocking send.        for (nnabr, nabrrank, nabrelem) ∈ zip(1:numnabr, mesh.nabrtorank,\n                                              mesh.nabrtosend)\n            sendreq[nnabr] = MPI.Isend((@view sendQ[:, :, nabrelem]), nabrrank, 777,\n                                       mpicomm)\n        end"
},

{
    "location": "examples/generated/advection/#Compute-RHS-Volume-Integral-1",
    "page": "Advection Equation Example",
    "title": "Compute RHS Volume Integral",
    "category": "section",
    "text": "Note that it is not necessary to have received all the MPI messages. Here we are interleaving computation with communication in order to curtail latency.  Here we perform the RHS volume integrals.        volumerhs!(rhs, Q, metric, D, ω, mesh.realelems)"
},

{
    "location": "examples/generated/advection/#Wait-on-MPI-receives-1",
    "page": "Advection Equation Example",
    "title": "Wait on MPI receives",
    "category": "section",
    "text": "We need to wait to receive the messages before we move on to the flux integrals.        MPI.Waitall!(recvreq)"
},

{
    "location": "examples/generated/advection/#Unpack-data-from-receive-buffer-1",
    "page": "Advection Equation Example",
    "title": "Unpack data from receive buffer",
    "category": "section",
    "text": "The inverse of the Pack datat to send buffer. We now unpack the receive buffer in order to use it in the RHS flux integral.        for elems ∈ mesh.nabrtorecv\n            for (nf, f) ∈ enumerate(Q)\n                f[index[:], nrealelem .+ elems] = recvQ[:, nf, elems]\n            end\n        end"
},

{
    "location": "examples/generated/advection/#Compute-RHS-Flux-Integral-1",
    "page": "Advection Equation Example",
    "title": "Compute RHS Flux Integral",
    "category": "section",
    "text": "We compute the flux integral on all \"realelems\" which are the elements owned by the current mpirank.        fluxrhs!(rhs, Q, metric, ω, mesh.realelems, vmapM, vmapP)"
},

{
    "location": "examples/generated/advection/#Update-solution-and-scale-RHS-1",
    "page": "Advection Equation Example",
    "title": "Update solution and scale RHS",
    "category": "section",
    "text": "We need to update/evolve the solution in time and multiply by the inverse mass matrix.        updatesolution!(rhs, Q, metric, ω, mesh.realelems, RKA[s%length(RKA)+1],\n                        RKB[s], dt)\n    end"
},

{
    "location": "examples/generated/advection/#Write-VTK-Output-1",
    "page": "Advection Equation Example",
    "title": "Write VTK Output",
    "category": "section",
    "text": "After each time-step, we dump out VTK data for Paraview/VisIt.    writemesh(@sprintf(\"Advection%dD_rank_%04d_step_%05d\", dim, mpirank, step),\n              coord...; fields=((\"ρ\", Q.ρ),), realelems=mesh.realelems)\nend"
},

{
    "location": "examples/generated/advection/#Compute-L2-Error-Norms-1",
    "page": "Advection Equation Example",
    "title": "Compute L2 Error Norms",
    "category": "section",
    "text": "Since we stored the initial condition, we can now compute the L2 error norms for both the solution and energy.for (δ, q) ∈ zip(Δ, Q)\n    δ .-= q\nend\neng = L2energy(Q, metric, ω, mesh.realelems)\neng = MPI.Allreduce(eng, MPI.SUM, mpicomm)\nmpirank == 0 && @show sqrt(eng)\n\nerr = L2energy(Δ, metric, ω, mesh.realelems)\nerr = MPI.Allreduce(err, MPI.SUM, mpicomm)\nmpirank == 0 && @show sqrt(err)\n\nnothing"
},

{
    "location": "examples/generated/advection/#advection-plain-program-1",
    "page": "Advection Equation Example",
    "title": "Plain Program",
    "category": "section",
    "text": "Below follows a version of the program without any comments. The file is also available here: advection.jlN = 4 #polynomial order\n#brickN = (10) #1D brickmesh\n#brickN = (10, 10) #2D brickmesh\nbrickN = (10, 10, 10) #3D brickmesh\nDFloat = Float64 #Number Type\ntend = DFloat(0.01) #Final Time\n\nusing MPI\nusing Canary\nusing Printf: @sprintf\n\ndim = length(brickN)\n\nprintln(\"N= \",N)\nprintln(\"dim= \",dim)\nprintln(\"brickN= \",brickN)\nprintln(\"DFloat= \",DFloat)\n\nMPI.Initialized() || MPI.Init() # only initialize MPI if not initialized\nMPI.finalize_atexit()\nmpicomm = MPI.COMM_WORLD\nmpirank = MPI.Comm_rank(mpicomm)\nmpisize = MPI.Comm_size(mpicomm)\n\nif dim == 1\n  (Nx, ) = brickN\n  local x = range(DFloat(0); length=Nx+1, stop=1)\n  mesh = brickmesh((x, ), (true, ); part=mpirank+1, numparts=mpisize)\nelseif dim == 2\n  (Nx, Ny) = brickN\n  local x = range(DFloat(0); length=Nx+1, stop=1)\n  local y = range(DFloat(0); length=Ny+1, stop=1)\n  mesh = brickmesh((x, y), (true, true); part=mpirank+1, numparts=mpisize)\nelse\n  (Nx, Ny, Nz) = brickN\n  local x = range(DFloat(0); length=Nx+1, stop=1)\n  local y = range(DFloat(0); length=Ny+1, stop=1)\n  local z = range(DFloat(0); length=Nz+1, stop=1)\n  mesh = brickmesh((x, y, z), (true, true, true);\n                   part=mpirank+1, numparts=mpisize)\nend\n\nmesh = partition(mpicomm, mesh...)\n\nmesh = connectmesh(mpicomm, mesh...)\n\n(vmapM, vmapP) = mappings(N, mesh.elemtoelem, mesh.elemtoface, mesh.elemtoordr)\n\n(ξ, ω) = lglpoints(DFloat, N)\nD = spectralderivative(ξ)\n\n(nface, nelem) = size(mesh.elemtoelem)\ncoord = creategrid(Val(dim), mesh.elemtocoord, ξ)\nif dim == 1\n  x = coord.x\n  for j = 1:length(x)\n    x[j] = x[j]\n  end\nelseif dim == 2\n  (x, y) = (coord.x, coord.y)\n  for j = 1:length(x)\n    (x[j], y[j]) = (x[j] .+ sin.(π * x[j]) .* sin.(2 * π * y[j]) / 10,\n                    y[j] .+ sin.(2 * π * x[j]) .* sin.(π * y[j]) / 10)\n  end\nelseif dim == 3\n  (x, y, z) = (coord.x, coord.y, coord.z)\n  for j = 1:length(x)\n    (x[j], y[j], z[j]) = (x[j] + (sin(π * x[j]) * sin(2 * π * y[j]) *\n                                  cos(2 * π * z[j])) / 10,\n                          y[j] + (sin(π * y[j]) * sin(2 * π * x[j]) *\n                                  cos(2 * π * z[j])) / 10,\n                          z[j] + (sin(π * z[j]) * sin(2 * π * x[j]) *\n                                  cos(2 * π * y[j])) / 10)\n  end\nend\n\ninclude(joinpath(@__DIR__, \"vtk.jl\"))\nwritemesh(@sprintf(\"Advection%dD_rank_%04d_mesh\", dim, mpirank), coord...;\n          realelems=mesh.realelems)\n\nmetric = computemetric(coord..., D)\n\nif dim == 1\n  statesyms = (:ρ, :Ux)\nelseif dim == 2\n  statesyms = (:ρ, :Ux, :Uy)\nelseif dim == 3\n  statesyms = (:ρ, :Ux, :Uy, :Uz)\nend\n\nQ   = NamedTuple{statesyms}(ntuple(j->zero(coord.x), length(statesyms)))\nrhs = NamedTuple{statesyms}(ntuple(j->zero(coord.x), length(statesyms)))\nif dim == 1\n  Q.ρ .= sin.(2 * π * x)\n  Q.Ux .= 1\nelseif dim == 2\n  Q.ρ .= sin.(2 * π * x) .* sin.(2 *  π * y)\n  Q.Ux .= 1\n  Q.Uy .= 1\nelseif dim == 3\n  Q.ρ .= sin.(2 * π * x) .* sin.(2 *  π * y) .* sin.(2 * π * z)\n\n  Q.Ux .= 1\n  Q.Uy .= 1\n  Q.Uz .= 1\nend\n\ndt = [floatmax(DFloat)]\nif dim == 1\n    (ξx) = (metric.ξx)\n    (Ux) = (Q.Ux)\n    for n = 1:length(Ux)\n        loc_dt = 2 ./ (abs.(Ux[n] * ξx[n]))\n        dt[1] = min(dt[1], loc_dt)\n    end\nelseif dim == 2\n    (ξx, ξy, ηx, ηy) = (metric.ξx, metric.ξy, metric.ηx, metric.ηy)\n    (Ux, Uy) = (Q.Ux, Q.Uy)\n    for n = 1:length(Ux)\n        loc_dt = 2 ./ max(abs.(Ux[n] * ξx[n] + Uy[n] * ξy[n]),\n                          abs.(Ux[n] * ηx[n] + Uy[n] * ηy[n]))\n        dt[1] = min(dt[1], loc_dt)\n    end\nelseif dim == 3\n    (ξx, ξy, ξz) = (metric.ξx, metric.ξy, metric.ξz)\n    (ηx, ηy, ηz) = (metric.ηx, metric.ηy, metric.ηz)\n    (ζx, ζy, ζz) = (metric.ζx, metric.ζy, metric.ζz)\n    (Ux, Uy, Uz) = (Q.Ux, Q.Uy, Q.Uz)\n    for n = 1:length(Ux)\n        loc_dt = 2 ./ max(abs.(Ux[n] * ξx[n] + Uy[n] * ξy[n] + Uz[n] * ξz[n]),\n                          abs.(Ux[n] * ηx[n] + Uy[n] * ηy[n] + Uz[n] * ηz[n]),\n                          abs.(Ux[n] * ζx[n] + Uy[n] * ζy[n] + Uz[n] * ζz[n]))\n        dt[1] = min(dt[1], loc_dt)\n    end\nend\ndt = MPI.Allreduce(dt[1], MPI.MIN, mpicomm)\ndt = DFloat(dt / N^sqrt(2))\nnsteps = ceil(Int64, tend / dt)\ndt = tend / nsteps\n@show (dt, nsteps)\n\nΔ   = NamedTuple{statesyms}(ntuple(j->zero(coord.x), length(statesyms)))\nif dim == 1\n  Δ.ρ .= sin.(2 * π * (x - tend * Q.Ux))\n  Δ.Ux .=  Q.Ux\nelseif dim == 2\n  Δ.ρ .= sin.(2 * π * (x - tend * Q.Ux)) .* sin.(2 *  π * (y - tend * Q.Uy))\n  Δ.Ux .=  Q.Ux\n  Δ.Uy .=  Q.Uy\nelseif dim == 3\n  Δ.ρ .= sin.(2 * π * (x - tend * Q.Ux)) .* sin.(2 *  π * (y - tend * Q.Uy)) .* sin.(2 * π * (z - tend * Q.Uz))\n\n  Δ.Ux .=  Q.Ux\n  Δ.Uy .=  Q.Uy\n  Δ.Uz .=  Q.Uz\nend\n\nRKA = (DFloat(0),\n       DFloat(-567301805773)  / DFloat(1357537059087),\n       DFloat(-2404267990393) / DFloat(2016746695238),\n       DFloat(-3550918686646) / DFloat(2091501179385),\n       DFloat(-1275806237668) / DFloat(842570457699 ))\n\nRKB = (DFloat(1432997174477) / DFloat(9575080441755 ),\n       DFloat(5161836677717) / DFloat(13612068292357),\n       DFloat(1720146321549) / DFloat(2090206949498 ),\n       DFloat(3134564353537) / DFloat(4481467310338 ),\n       DFloat(2277821191437) / DFloat(14882151754819))\n\nRKC = (DFloat(0),\n       DFloat(1432997174477) / DFloat(9575080441755),\n       DFloat(2526269341429) / DFloat(6820363962896),\n       DFloat(2006345519317) / DFloat(3224310063776),\n       DFloat(2802321613138) / DFloat(2924317926251))\n\nfunction volumerhs!(rhs, Q::NamedTuple{S, NTuple{2, T}}, metric, D, ω,\n                    elems) where {S, T}\n  rhsρ = rhs.ρ\n  (ρ, Ux) = (Q.ρ, Q.Ux)\n  Nq = size(ρ, 1)\n  J = metric.J\n  ξx = metric.ξx\n  for e ∈ elems\n\n      rhsρ[:,e] += D\' * (ω .* J[:,e] .* ρ[:,e] .* (ξx[:,e] .* Ux[:,e]))\n  end #e ∈ elems\nend #function volumerhs-1d\n\nfunction volumerhs!(rhs, Q::NamedTuple{S, NTuple{3, T}}, metric, D, ω,\n                    elems) where {S, T}\n    rhsρ = rhs.ρ\n    (ρ, Ux, Uy) = (Q.ρ, Q.Ux, Q.Uy)\n    Nq = size(ρ, 1)\n    J = metric.J\n    (ξx, ηx, ξy, ηy) = (metric.ξx, metric.ηx, metric.ξy, metric.ηy)\n    for e ∈ elems\n\n        for j = 1:Nq\n            rhsρ[:, j, e] +=\n                D\' * (ω[j] * ω .* J[:, j, e].* ρ[:, j, e] .*\n                      (ξx[:, j, e] .* Ux[:, j, e] + ξy[:, j, e] .* Uy[:, j, e]))\n        end #j\n\n        for i = 1:Nq\n            rhsρ[i, :, e] +=\n                D\' * (ω[i] * ω .* J[i, :, e].* ρ[i, :, e] .*\n                      (ηx[i, :, e] .* Ux[i, :, e] + ηy[i, :, e] .* Uy[i, :, e]))\n        end #i\n    end #e ∈ elems\nend #function volumerhs-2d\n\nfunction volumerhs!(rhs, Q::NamedTuple{S, NTuple{4, T}}, metric, D, ω,\n                    elems) where {S, T}\n    rhsρ = rhs.ρ\n    (ρ, Ux, Uy, Uz) = (Q.ρ, Q.Ux, Q.Uy, Q.Uz)\n    Nq = size(ρ, 1)\n    J = metric.J\n    (ξx, ηx, ζx) = (metric.ξx, metric.ηx, metric.ζx)\n    (ξy, ηy, ζy) = (metric.ξy, metric.ηy, metric.ζy)\n    (ξz, ηz, ζz) = (metric.ξz, metric.ηz, metric.ζz)\n    for e ∈ elems\n\n        for k = 1:Nq\n            for j = 1:Nq\n                rhsρ[:, j, k, e] +=\n                    D\' * (ω[j] * ω[k] * ω .* J[:, j, k, e] .* ρ[:, j, k, e] .*\n                          (ξx[:, j, k, e] .* Ux[:, j, k, e] +\n                           ξy[:, j, k, e] .* Uy[:, j, k, e] +\n                           ξz[:, j, k, e] .* Uz[:, j, k, e]))\n            end #j\n        end #k\n\n        for k = 1:Nq\n            for i = 1:Nq\n                rhsρ[i, :, k, e] +=\n                    D\' * (ω[i] * ω[k] * ω .* J[i, :, k, e] .* ρ[i, :, k, e] .*\n                          (ηx[i, :, k, e] .* Ux[i, :, k, e] +\n                           ηy[i, :, k, e] .* Uy[i, :, k, e] +\n                           ηz[i, :, k, e] .* Uz[i, :, k, e]))\n            end #i\n        end #k\n\n        for j = 1:Nq\n            for i = 1:Nq\n                rhsρ[i, j, :, e] +=\n                    D\' * (ω[i] * ω[j] * ω .* J[i, j, :, e] .* ρ[i, j, :, e] .*\n                          (ζx[i, j, :, e] .* Ux[i, j, :, e] +\n                           ζy[i, j, :, e] .* Uy[i, j, :, e] +\n                           ζz[i, j, :, e] .* Uz[i, j, :, e]))\n            end #i\n        end #j\n    end #e ∈ elems\nend\n\nfunction fluxrhs!(rhs, Q::NamedTuple{S, NTuple{2, T}}, metric, ω, elems, vmapM,\n                  vmapP) where {S, T}\n    rhsρ = rhs.ρ\n    (ρ, Ux) = (Q.ρ, Q.Ux)\n    nface = 2\n    (nx, sJ) = (metric.nx, metric.sJ)\n    nx = reshape(nx, size(vmapM))\n    sJ = reshape(sJ, size(vmapM))\n\n    for e ∈ elems\n        for f ∈ 1:nface\n            ρM = ρ[vmapM[1, f, e]]\n            UxM = Ux[vmapM[1, f, e]]\n            FxM = ρM .* UxM\n\n            ρP = ρ[vmapP[1, f, e]]\n            UxP = Ux[vmapP[1, f, e]]\n            FxP = ρP .* UxP\n\n            nxM = nx[1, f, e]\n            λ = max.(abs.(nxM .* UxM), abs.(nxM .* UxP))\n\n            F = (nxM .* (FxM + FxP) - λ .* (ρP - ρM)) / 2\n            rhsρ[vmapM[1, f, e]] -= sJ[1, f, e] .* F\n        end #for f ∈ 1:nface\n    end #e ∈ elems\nend #function fluxrhs-1d\n\nfunction fluxrhs!(rhs, Q::NamedTuple{S, NTuple{3, T}}, metric, ω, elems, vmapM,\n                  vmapP) where {S, T}\n    rhsρ = rhs.ρ\n    (ρ, Ux, Uy) = (Q.ρ, Q.Ux, Q.Uy)\n    nface = 4\n    (nx, ny, sJ) = (metric.nx, metric.ny, metric.sJ)\n    for e ∈ elems\n        for f ∈ 1:nface\n            ρM = ρ[vmapM[:, f, e]]\n            UxM = Ux[vmapM[:, f, e]]\n            UyM = Uy[vmapM[:, f, e]]\n            FxM = ρM .* UxM\n            FyM = ρM .* UyM\n\n            ρP = ρ[vmapP[:, f, e]]\n            UxP = Ux[vmapP[:, f, e]]\n            UyP = Uy[vmapP[:, f, e]]\n            FxP = ρP .* UxP\n            FyP = ρP .* UyP\n\n            nxM = nx[:, f, e]\n            nyM = ny[:, f, e]\n            λ = max.(abs.(nxM .* UxM + nyM .* UyM), abs.(nxM .* UxP + nyM .* UyP))\n\n            F = (nxM .* (FxM + FxP) + nyM .* (FyM + FyP) + λ .* (ρM - ρP)) / 2\n            rhsρ[vmapM[:, f, e]] -= ω .* sJ[:, f, e] .* F\n        end #f ∈ 1:nface\n    end #e ∈ elems\nend #function fluxrhs-2d\n\nfunction fluxrhs!(rhs, Q::NamedTuple{S, NTuple{4, T}}, metric, ω, elems, vmapM,\n                  vmapP) where {S, T}\n    rhsρ = rhs.ρ\n    (ρ, Ux, Uy, Uz) = (Q.ρ, Q.Ux, Q.Uy, Q.Uz)\n    nface = 6\n    (nx, ny, nz, sJ) = (metric.nx, metric.ny, metric.nz, metric.sJ)\n    nx = reshape(nx, size(vmapM))\n    ny = reshape(ny, size(vmapM))\n    nz = reshape(nz, size(vmapM))\n    sJ = reshape(sJ, size(vmapM))\n    for e ∈ elems\n        for f ∈ 1:nface\n            ρM = ρ[vmapM[:, f, e]]\n            UxM = Ux[vmapM[:, f, e]]\n            UyM = Uy[vmapM[:, f, e]]\n            UzM = Uz[vmapM[:, f, e]]\n            FxM = ρM .* UxM\n            FyM = ρM .* UyM\n            FzM = ρM .* UzM\n\n            ρP = ρ[vmapP[:, f, e]]\n            UxP = Ux[vmapP[:, f, e]]\n            UyP = Uy[vmapP[:, f, e]]\n            UzP = Uz[vmapP[:, f, e]]\n            FxP = ρP .* UxP\n            FyP = ρP .* UyP\n            FzP = ρP .* UzP\n\n            nxM = nx[:, f, e]\n            nyM = ny[:, f, e]\n            nzM = nz[:, f, e]\n            λ = max.(abs.(nxM .* UxM + nyM .* UyM + nzM .* UzM),\n                     abs.(nxM .* UxP + nyM .* UyP + nzM .* UzP))\n\n            F = (nxM .* (FxM + FxP) + nyM .* (FyM + FyP) + nzM .* (FzM + FzP) +\n                 λ .* (ρM - ρP)) / 2\n            rhsρ[vmapM[:, f, e]] -= kron(ω, ω) .* sJ[:, f, e] .* F\n        end #f ∈ 1:nface\n    end #e ∈ elems\nend #function fluxrhs-3d\n\nfunction updatesolution!(rhs, Q::NamedTuple{S, NTuple{2, T}}, metric, ω, elems,\n                         rka, rkb, dt) where {S, T}\n    J = metric.J\n    M =  ω\n    for (rhsq, q) ∈ zip(rhs, Q)\n        for e ∈ elems\n            q[:, e] += rkb * dt * rhsq[:, e] ./ ( M .* J[:, e])\n            rhsq[:, e] *= rka\n        end\n    end\nend\n\nfunction updatesolution!(rhs, Q::NamedTuple{S, NTuple{3, T}}, metric, ω, elems,\n                         rka, rkb, dt) where {S, T}\n    J = metric.J\n    M = reshape(kron(ω, ω), length(ω), length(ω))\n    for (rhsq, q) ∈ zip(rhs, Q)\n        for e ∈ elems\n            q[:, :, e] += rkb * dt * rhsq[:, :, e] ./ (M .* J[:, :, e])\n            rhsq[:, :, e] *= rka\n        end\n    end\nend\n\nfunction updatesolution!(rhs, Q::NamedTuple{S, NTuple{4, T}}, metric, ω, elems,\n                         rka, rkb, dt) where {S, T}\n    J = metric.J\n    M = reshape(kron(ω, ω, ω), length(ω), length(ω), length(ω))\n    for (rhsq, q) ∈ zip(rhs, Q)\n        for e ∈ elems\n            q[:, :, :, e] += rkb * dt * rhsq[:, :, :, e] ./ (M .* J[:, :, :, e])\n            rhsq[:, :, :, e] *= rka\n        end\n    end\nend\n\nfunction L2energy(Q::NamedTuple{S, NTuple{2, T}}, metric, ω, elems) where {S, T}\n  J = metric.J\n  Nq = length(ω)\n  M = ω\n  index = CartesianIndices(ntuple(j->1:Nq, Val(1)))\n\n  energy = [zero(J[1])]\n  for q ∈ Q\n    for e ∈ elems\n      for ind ∈ index\n        energy[1] += M[ind] * J[ind, e] * q[ind, e]^2\n      end\n    end\n  end\n  energy[1]\nend\n\nfunction L2energy(Q::NamedTuple{S, NTuple{3, T}}, metric, ω, elems) where {S, T}\n  J = metric.J\n  Nq = length(ω)\n  M = reshape(kron(ω, ω), Nq, Nq)\n  index = CartesianIndices(ntuple(j->1:Nq, Val(2)))\n\n  energy = [zero(J[1])]\n  for q ∈ Q\n    for e ∈ elems\n      for ind ∈ index\n        energy[1] += M[ind] * J[ind, e] * q[ind, e]^2\n      end\n    end\n  end\n  energy[1]\nend\n\nfunction L2energy(Q::NamedTuple{S, NTuple{4, T}}, metric, ω, elems) where {S, T}\n  J = metric.J\n  Nq = length(ω)\n  M = reshape(kron(ω, ω, ω), Nq, Nq, Nq)\n  index = CartesianIndices(ntuple(j->1:Nq, Val(3)))\n\n  energy = [zero(J[1])]\n  for q ∈ Q\n    for e ∈ elems\n      for ind ∈ index\n        energy[1] += M[ind] * J[ind, e] * q[ind, e]^2\n      end\n    end\n  end\n  energy[1]\nend\n\nnumnabr = length(mesh.nabrtorank)\n\nsendreq = fill(MPI.REQUEST_NULL, numnabr)\nrecvreq = fill(MPI.REQUEST_NULL, numnabr)\n\nsendQ = Array{DFloat, 3}(undef, (N+1)^dim, length(Q), length(mesh.sendelems))\nrecvQ = Array{DFloat, 3}(undef, (N+1)^dim, length(Q), length(mesh.ghostelems))\n\nindex = CartesianIndices(ntuple(j->1:N+1, dim))\nnrealelem = length(mesh.realelems)\n\ninclude(joinpath(@__DIR__, \"vtk.jl\"))\nwritemesh(@sprintf(\"Advection%dD_rank_%04d_step_%05d\", dim, mpirank, 0),\n          coord...; fields=((\"ρ\", Q.ρ),), realelems=mesh.realelems)\n\nfor step = 1:nsteps\n    mpirank == 0 && @show step\n    for s = 1:length(RKA)\n\n        for (nnabr, nabrrank, nabrelem) ∈ zip(1:numnabr, mesh.nabrtorank,\n                                              mesh.nabrtorecv)\n            recvreq[nnabr] = MPI.Irecv!((@view recvQ[:, :, nabrelem]), nabrrank, 777,\n                                        mpicomm)\n        end\n\n        MPI.Waitall!(sendreq)\n\n        for (ne, e) ∈ enumerate(mesh.sendelems)\n            for (nf, f) ∈ enumerate(Q)\n                sendQ[:, nf, ne] = f[index[:], e]\n            end\n        end\n\n        for (nnabr, nabrrank, nabrelem) ∈ zip(1:numnabr, mesh.nabrtorank,\n                                              mesh.nabrtosend)\n            sendreq[nnabr] = MPI.Isend((@view sendQ[:, :, nabrelem]), nabrrank, 777,\n                                       mpicomm)\n        end\n\n        volumerhs!(rhs, Q, metric, D, ω, mesh.realelems)\n\n        MPI.Waitall!(recvreq)\n\n        for elems ∈ mesh.nabrtorecv\n            for (nf, f) ∈ enumerate(Q)\n                f[index[:], nrealelem .+ elems] = recvQ[:, nf, elems]\n            end\n        end\n\n        fluxrhs!(rhs, Q, metric, ω, mesh.realelems, vmapM, vmapP)\n\n        updatesolution!(rhs, Q, metric, ω, mesh.realelems, RKA[s%length(RKA)+1],\n                        RKB[s], dt)\n    end\n\n    writemesh(@sprintf(\"Advection%dD_rank_%04d_step_%05d\", dim, mpirank, step),\n              coord...; fields=((\"ρ\", Q.ρ),), realelems=mesh.realelems)\nend\n\nfor (δ, q) ∈ zip(Δ, Q)\n    δ .-= q\nend\neng = L2energy(Q, metric, ω, mesh.realelems)\neng = MPI.Allreduce(eng, MPI.SUM, mpicomm)\nmpirank == 0 && @show sqrt(eng)\n\nerr = L2energy(Δ, metric, ω, mesh.realelems)\nerr = MPI.Allreduce(err, MPI.SUM, mpicomm)\nmpirank == 0 && @show sqrt(err)\n\nnothing\n\n# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jlThis page was generated using Literate.jl."
},

{
    "location": "reference/mesh/#",
    "page": "Mesh",
    "title": "Mesh",
    "category": "page",
    "text": "DocTestSetup = :(using Canary)"
},

{
    "location": "reference/mesh/#Canary.brickmesh",
    "page": "Mesh",
    "title": "Canary.brickmesh",
    "category": "function",
    "text": "brickmesh(x, periodic; part=1, numparts=1; boundary)\n\nGenerate a brick mesh with coordinates given by the tuple x and the periodic dimensions given by the periodic tuple.\n\nThe brick can optionally be partitioned into numparts and this returns partition part.  This is a simple Cartesian partition and further partitioning (e.g, based on a space-filling curve) should be done before the mesh is used for computation.\n\nBy default boundary faces will be marked with a one and other faces with a zero.  Specific boundary numbers can also be passed for each face of the brick in boundary.  This will mark the nonperiodic brick faces with the given boundary number.\n\nExamples\n\nWe can build a 3 by 2 element two-dimensional mesh that is periodic in the x_2-direction with\n\njulia> using Canary\njulia> (elemtovert, elemtocoord, elemtobndy, faceconnections) =\n        brickmesh((2:5,4:6), (false,true); boundary=[1 3; 2 4]);\n\nThis returns the mesh structure for\n\n         x_2\n\n          ^\n          |\n         6-  9----10----11----12\n          |  |     |     |     |\n          |  |  4  |  5  |  6  |\n          |  |     |     |     |\n         5-  5-----6-----7-----8\n          |  |     |     |     |\n          |  |  1  |  2  |  3  |\n          |  |     |     |     |\n         4-  1-----2-----3-----4\n          |\n          +--|-----|-----|-----|--> x_1\n             2     3     4     5\n\nThe (number of corners by number of elements) array elemtovert gives the global vertex number for the corners of each element.\n\njulia> elemtovert\n4×6 Array{Int64,2}:\n 1  2  3   5   6   7\n 2  3  4   6   7   8\n 5  6  7   9  10  11\n 6  7  8  10  11  12\n\nNote that the vertices are listed in Cartesian order.\n\nThe (dimension by number of corners by number of elements) array elemtocoord gives the coordinates of the corners of each element.\n\njulia> elemtocoord\n2×4×6 Array{Int64,3}:\n[:, :, 1] =\n 2  3  2  3\n 4  4  5  5\n\n[:, :, 2] =\n 3  4  3  4\n 4  4  5  5\n\n[:, :, 3] =\n 4  5  4  5\n 4  4  5  5\n\n[:, :, 4] =\n 2  3  2  3\n 5  5  6  6\n\n[:, :, 5] =\n 3  4  3  4\n 5  5  6  6\n\n[:, :, 6] =\n 4  5  4  5\n 5  5  6  6\n\nThe (number of faces by number of elements) array elemtobndy gives the boundary number for each face of each element.  A zero will be given for connected faces.\n\njulia> elemtobndy\n4×6 Array{Int64,2}:\n 1  0  0  1  0  0\n 0  0  2  0  0  2\n 0  0  0  0  0  0\n 0  0  0  0  0  0\n\nNote that the faces are listed in Cartesian order.\n\nFinally, the periodic face connections are given in faceconnections which is a list of arrays, one for each connection. Each array in the list is given in the format [e, f, vs...] where\n\ne  is the element number;\nf  is the face number; and\nvs is the global vertices that face associated with.\n\nI the example\n\njulia> faceconnections\n3-element Array{Array{Int64,1},1}:\n [4, 4, 1, 2]\n [5, 4, 2, 3]\n [6, 4, 3, 4]\n\nwe see that face 4 of element 5 is associated with vertices [2 3] (the vertices for face 1 of element 2).\n\n\n\n\n\n"
},

{
    "location": "reference/mesh/#Canary.connectmesh",
    "page": "Mesh",
    "title": "Canary.connectmesh",
    "category": "function",
    "text": "connectmesh(comm::MPI.Comm, elemtovert, elemtocoord, elemtobndy,\n            faceconnections)\n\nThis function takes in a mesh (as returned for example by brickmesh) and returns a connected mesh.  This returns a NamedTuple of:\n\nelems the range of element indices\nrealelems the range of real (aka nonghost) element indices\nghostelems the range of ghost element indices\nsendelems an array of send element indices sorted so that\nelemtocoord element to vertex coordinates; elemtocoord[d,i,e] is the  dth coordinate of corner i of element e\nelemtoelem element to neighboring element; elemtoelem[f,e] is the number of the element neighboring element e across face f.  If there is no neighboring element then elemtoelem[f,e] == e.\nelemtoface element to neighboring element face; elemtoface[f,e] is the face number of the element neighboring element e across face f.  If there is no neighboring element then elemtoface[f,e] == f.\nelemtoordr element to neighboring element order; elemtoordr[f,e] is the ordering number of the element neighboring element e across face f.  If there is no neighboring element then elemtoordr[f,e] == 1.\nelemtobndy element to bounday number; elemtobndy[f,e] is the boundary number of face f of element e.  If there is a neighboring element then elemtobndy[f,e] == 0.\nnabrtorank a list of the MPI ranks for the neighboring processes\nnabrtorecv a range in ghost elements to receive for each neighbor\nnabrtosend a range in sendelems to send for each neighbor\n\n\n\n\n\n"
},

{
    "location": "reference/mesh/#Canary.mappings",
    "page": "Mesh",
    "title": "Canary.mappings",
    "category": "function",
    "text": "mappings(N, elemtoelem, elemtoface, elemtoordr)\n\nThis function takes in a polynomial order N and parts of a mesh (as returned from connectmesh) and returns index mappings for the element surface flux computation.  The returned Tuple contains:\n\nvmapM an array of linear indices into the volume degrees of freedom where vmapM[:,f,e] are the degrees of freedom indices for face f of element  e.\nvmapP an array of linear indices into the volume degrees of freedom where vmapP[:,f,e] are the degrees of freedom indices for the face neighboring face f of element e.\n\n\n\n\n\n"
},

{
    "location": "reference/mesh/#Canary.partition",
    "page": "Mesh",
    "title": "Canary.partition",
    "category": "function",
    "text": "partition(comm::MPI.Comm, elemtovert, elemtocoord, elemtobndy,\n          faceconnections)\n\nThis function takes in a mesh (as returned for example by brickmesh) and returns a Hilbert curve based partitioned mesh.\n\n\n\n\n\n"
},

{
    "location": "reference/mesh/#Mesh-1",
    "page": "Mesh",
    "title": "Mesh",
    "category": "section",
    "text": "brickmesh\nconnectmesh\nmappings\npartition"
},

{
    "location": "reference/metric/#",
    "page": "Metric Terms",
    "title": "Metric Terms",
    "category": "page",
    "text": "DocTestSetup = :(using Canary)"
},

{
    "location": "reference/metric/#Canary.creategrid",
    "page": "Metric Terms",
    "title": "Canary.creategrid",
    "category": "function",
    "text": "creategrid(::Val{1}, elemtocoord::AbstractArray{S, 3},\n           r::AbstractVector{T}) where {S, T}\n\nCreate a grid using elemtocoord (see brickmesh) using the 1-D (-1, 1) reference coordinates r. The element grids are filled using bilinear interpolation of the element coordinates.\n\nThe grid is returned as a tuple of with x array\n\n\n\n\n\ncreategrid(::Val{2}, elemtocoord::AbstractArray{S, 3},\n           r::AbstractVector{T}) where {S, T}\n\nCreate a 2-D tensor product grid using elemtocoord (see brickmesh) using the 1-D (-1, 1) reference coordinates r. The element grids are filled using bilinear interpolation of the element coordinates.\n\nThe grid is returned as a tuple of the x and y arrays\n\n\n\n\n\ncreategrid(::Val{3}, elemtocoord::AbstractArray{S, 3},\n           r::AbstractVector{T}) where {S, T}\n\nCreate a 3-D tensor product grid using elemtocoord (see brickmesh) using the 1-D (-1, 1) reference coordinates r. The element grids are filled using bilinear interpolation of the element coordinates.\n\nThe grid is returned as a tuple of the x, y, z arrays\n\n\n\n\n\n"
},

{
    "location": "reference/metric/#Canary.creategrid!",
    "page": "Metric Terms",
    "title": "Canary.creategrid!",
    "category": "function",
    "text": "creategrid!(x::AbstractArray{T, 2}, elemtocoord::AbstractArray{S, 3},\n            r::AbstractVector{T}) where {S, T}\n\nCreate a 1-D grid using elemtocoord (see brickmesh) using the 1-D (-1, 1) reference coordinates r. The element grids are filled using linear interpolation of the element coordinates.\n\nIf Nq = length(r) and nelem = size(elemtocoord, 3) then the preallocated array x should be (Nq, nelem) == size(x).\n\n\n\n\n\ncreategrid!(x::AbstractArray{T, 3}, y::AbstractArray{T, 3},\n            elemtocoord::AbstractArray{S, 3},\n            r::AbstractVector{T}) where {S, T}\n\nCreate a 2-D tensor product grid using elemtocoord (see brickmesh) using the 1-D (-1, 1) reference coordinates r. The element grids are filled using bilinear interpolation of the element coordinates.\n\nIf Nq = length(r) and nelem = size(elemtocoord, 3) then the preallocated arrays x and y should be (Nq, Nq, nelem) == size(x) == size(y).\n\n\n\n\n\ncreategrid!(x::AbstractArray{T, 3}, y::AbstractArray{T, 3},\n            z::AbstractArray{T, 3}, elemtocoord::AbstractArray{S, 3},\n            r::AbstractVector{T}) where {S, T}\n\nCreate a 3-D tensor product grid using elemtocoord (see brickmesh) using the 1-D (-1, 1) reference coordinates r. The element grids are filled using trilinear interpolation of the element coordinates.\n\nIf Nq = length(r) and nelem = size(elemtocoord, 3) then the preallocated arrays x, y, and z should be (Nq, Nq, Nq, nelem) == size(x) == size(y) == size(z).\n\n\n\n\n\n"
},

{
    "location": "reference/metric/#Canary.computemetric",
    "page": "Metric Terms",
    "title": "Canary.computemetric",
    "category": "function",
    "text": "computemetric(x::AbstractArray{T, 2}, D::AbstractMatrix{T}) where T\n\nCompute the 1-D metric terms from the element grid array x using the derivative matrix D. The derivative matrix D should be consistent with the reference grid r used in creategrid!.\n\nThe metric terms are returned as a \'NamedTuple` of the following arrays:\n\nJ the Jacobian determinant\nξx derivative ∂r / ∂x\'\nsJ the surface Jacobian\n\'nx` outward pointing unit normal in x-direction\n\n\n\n\n\ncomputemetric(x::AbstractArray{T, 3}, y::AbstractArray{T, 3},\n              D::AbstractMatrix{T}) where T\n\nCompute the 2-D metric terms from the element grid arrays x and y using the derivative matrix D. The derivative matrix D should be consistent with the reference grid r used in creategrid!.\n\nThe metric terms are returned as a \'NamedTuple` of the following arrays:\n\nJ the Jacobian determinant\nξx derivative ∂r / ∂x\'\nηx derivative ∂s / ∂x\'\nξy derivative ∂r / ∂y\'\nηy derivative ∂s / ∂y\'\nsJ the surface Jacobian\n\'nx` outward pointing unit normal in x-direction\n\'ny` outward pointing unit normal in y-direction\n\n\n\n\n\ncomputemetric(x::AbstractArray{T, 3}, y::AbstractArray{T, 3},\n              D::AbstractMatrix{T}) where T\n\nCompute the 3-D metric terms from the element grid arrays x, y, and z using the derivative matrix D. The derivative matrix D should be consistent with the reference grid r used in creategrid!.\n\nThe metric terms are returned as a \'NamedTuple` of the following arrays:\n\nJ the Jacobian determinant\nξx derivative ∂r / ∂x\'\nηx derivative ∂s / ∂x\'\nζx derivative ∂t / ∂x\'\nξy derivative ∂r / ∂y\'\nηy derivative ∂s / ∂y\'\nζy derivative ∂t / ∂y\'\nξz derivative ∂r / ∂z\'\nηz derivative ∂s / ∂z\'\nζz derivative ∂t / ∂z\'\nsJ the surface Jacobian\n\'nx` outward pointing unit normal in x-direction\n\'ny` outward pointing unit normal in y-direction\n\'nz` outward pointing unit normal in z-direction\n\n\n\n\n\n"
},

{
    "location": "reference/metric/#Canary.computemetric!",
    "page": "Metric Terms",
    "title": "Canary.computemetric!",
    "category": "function",
    "text": "computemetric!(x::AbstractArray{T, 2},\n               J::AbstractArray{T, 2},\n               ξx::AbstractArray{T, 2},\n               sJ::AbstractArray{T, 3},\n               nx::AbstractArray{T, 3},\n               D::AbstractMatrix{T}) where T\n\nCompute the 1-D metric terms from the element grid arrays x. All the arrays are preallocated by the user and the (square) derivative matrix D should be consistent with the reference grid r used in creategrid!.\n\nIf Nq = size(D, 1) and nelem = size(x, 2) then the volume arrays x, J, and ξx should all be of size (Nq, nelem).  Similarly, the face arrays sJ and nx should be of size (1, nfaces, nelem) with nfaces = 2.\n\n\n\n\n\ncomputemetric!(x::AbstractArray{T, 3}, y::AbstractArray{T, 3},\n               J::AbstractArray{T, 3},\n               ξx::AbstractArray{T, 3}, ηx::AbstractArray{T, 3},\n               ξy::AbstractArray{T, 3}, ηy::AbstractArray{T, 3},\n               sJ::AbstractArray{T, 3},\n               nx::AbstractArray{T, 3}, ny::AbstractArray{T, 3},\n               D::AbstractMatrix{T}) where T\n\nCompute the 2-D metric terms from the element grid arrays x and y. All the arrays are preallocated by the user and the (square) derivative matrix D should be consistent with the reference grid r used in creategrid!.\n\nIf Nq = size(D, 1) and nelem = size(x, 3) then the volume arrays x, y, J, ξx, ηx, ξy, and ηy should all be of size (Nq, Nq, nelem). Similarly, the face arrays sJ, nx, and ny should be of size (Nq, nfaces, nelem) with nfaces = 4.\n\n\n\n\n\ncomputemetric!(x::AbstractArray{T, 4}, y::AbstractArray{T, 4},\n               z::AbstractArray{T, 4}, J::AbstractArray{T, 4},\n               ξx::AbstractArray{T, 4}, ηx::AbstractArray{T, 4},\n               ζx::AbstractArray{T, 4} ξy::AbstractArray{T, 4},\n               ηy::AbstractArray{T, 4}, ζy::AbstractArray{T, 4}\n               ξz::AbstractArray{T, 4}, ηz::AbstractArray{T, 4},\n               ζz::AbstractArray{T, 4} sJ::AbstractArray{T, 3},\n               nx::AbstractArray{T, 3}, ny::AbstractArray{T, 3},\n               nz::AbstractArray{T, 3}, D::AbstractMatrix{T}) where T\n\nCompute the 3-D metric terms from the element grid arrays x, y, and z. All the arrays are preallocated by the user and the (square) derivative matrix D should be consistent with the reference grid r used in creategrid!.\n\nIf Nq = size(D, 1) and nelem = size(x, 4) then the volume arrays x, y, z, J, ξx, ηx, ζx, ξy, ηy, ζy, ξz, ηz, and ζz should all be of size (Nq, Nq, Nq, nelem).  Similarly, the face arrays sJ, nx, ny, and nz should be of size (Nq^2, nfaces, nelem) with nfaces = 6.\n\nThe curl invariant formulation of Kopriva (2006), equation 37, is used.\n\nReference:   Kopriva, David A. \"Metric identities and the discontinuous spectral element   method on curvilinear meshes.\" Journal of Scientific Computing 26.3 (2006):   301-327. https://doi.org/10.1007/s10915-005-9070-8\n\n\n\n\n\n"
},

{
    "location": "reference/metric/#Metric-Terms-1",
    "page": "Metric Terms",
    "title": "Metric Terms",
    "category": "section",
    "text": "creategrid\ncreategrid!\ncomputemetric\ncomputemetric!"
},

{
    "location": "reference/operators/#",
    "page": "Element Operators",
    "title": "Element Operators",
    "category": "page",
    "text": "DocTestSetup = :(using Canary)"
},

{
    "location": "reference/operators/#Canary.baryweights",
    "page": "Element Operators",
    "title": "Canary.baryweights",
    "category": "function",
    "text": "baryweights(r)\n\nreturns the barycentric weights associated with the array of points r\n\nReference:   Jean-Paul Berrut & Lloyd N. Trefethen, \"Barycentric Lagrange Interpolation\",   SIAM Review 46 (2004), pp. 501-517.   https://doi.org/10.1137/S0036144502417715\n\n\n\n\n\n"
},

{
    "location": "reference/operators/#Canary.interpolationmatrix",
    "page": "Element Operators",
    "title": "Canary.interpolationmatrix",
    "category": "function",
    "text": "interpolationmatrix(rsrc::AbstractVector{T}, rdst::AbstractVector{T},\n                    wbsrc=baryweights(rsrc)::AbstractVector{T}) where T\n\nreturns the polynomial interpolation matrix for interpolating between the points rsrc (with associated barycentric weights wbsrc) and rdst\n\nReference:   Jean-Paul Berrut & Lloyd N. Trefethen, \"Barycentric Lagrange Interpolation\",   SIAM Review 46 (2004), pp. 501-517.   https://doi.org/10.1137/S0036144502417715\n\n\n\n\n\n"
},

{
    "location": "reference/operators/#Canary.lglpoints",
    "page": "Element Operators",
    "title": "Canary.lglpoints",
    "category": "function",
    "text": "lglpoints(::Type{T}, N::Integer) where T <: AbstractFloat\n\nreturns the points r and weights w associated with the N+1-point Gauss-Legendre-Lobatto quadrature rule of type T\n\n\n\n\n\n"
},

{
    "location": "reference/operators/#Canary.lgpoints",
    "page": "Element Operators",
    "title": "Canary.lgpoints",
    "category": "function",
    "text": "lgpoints(::Type{T}, N::Integer) where T <: AbstractFloat\n\nreturns the points r and weights w associated with the N+1-point Gauss-Legendre quadrature rule of type T\n\n\n\n\n\n"
},

{
    "location": "reference/operators/#Canary.spectralderivative",
    "page": "Element Operators",
    "title": "Canary.spectralderivative",
    "category": "function",
    "text": "spectralderivative(r::AbstractVector{T},\n                   wb=baryweights(r)::AbstractVector{T}) where T\n\nreturns the spectral differentiation matrix for a polynomial defined on the points r with associated barycentric weights wb\n\nReference:   Jean-Paul Berrut & Lloyd N. Trefethen, \"Barycentric Lagrange Interpolation\",   SIAM Review 46 (2004), pp. 501-517.   https://doi.org/10.1137/S0036144502417715\n\n\n\n\n\n"
},

{
    "location": "reference/operators/#Element-Operators-1",
    "page": "Element Operators",
    "title": "Element Operators",
    "category": "section",
    "text": "baryweights\ninterpolationmatrix\nlglpoints\nlgpoints\nspectralderivative"
},

]}
