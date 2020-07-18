
export DiscontinuousSpectralElementGrid
export get_z

"""
    DiscontinuousSpectralElementGrid(topology; FloatType, DeviceArray,
                                     polynomialorder,
                                     meshwarp = (x...)->identity(x))

Generate a discontinuous spectral element (tensor product,
Legendre-Gauss-Lobatto) grid/mesh from a `topology`, where the order of the
elements is given by `polynomialorder`. `DeviceArray` gives the array type used
to store the data (`CuArray` or `Array`), and the coordinate points will be of
`FloatType`.

The optional `meshwarp` function allows the coordinate points to be warped after
the mesh is created; the mesh degrees of freedom are orginally assigned using a
trilinear blend of the element corner locations.
"""
struct DiscontinuousSpectralElementGrid{
    T,
    dim,
    N,
    Np,
    DA,
    DAT1,
    DAT2,
    DAT3,
    DAT4,
    DAI1,
    DAI2,
    DAI3,
    TOP,
} <: AbstractGrid{T, dim, N, Np, DA}
    "mesh topology"
    topology::TOP

    "volume metric terms"
    vgeo::DAT3

    "surface metric terms"
    sgeo::DAT4

    "element to boundary condition map"
    elemtobndy::DAI2

    "volume DOF to element minus side map"
    vmap⁻::DAI3

    "volume DOF to element plus side map"
    vmap⁺::DAI3

    "list of DOFs that need to be received (in neighbors order)"
    vmaprecv::DAI1

    "list of DOFs that need to be sent (in neighbors order)"
    vmapsend::DAI1

    "An array of ranges in `vmaprecv` to receive from each neighbor"
    nabrtovmaprecv

    "An array of ranges in `vmapsend` to send to each neighbor"
    nabrtovmapsend

    "Array of real elements that do not have a ghost element as a neighbor"
    interiorelems

    "Array of real elements that have at least one ghost element as a neighbor"
    exteriorelems

    "Array indicating if a degree of freedom (real or ghost) is active"
    activedofs

    "1-D lvl weights on the device"
    ω::DAT1

    "1-D derivative operator on the device"
    D::DAT2

    "1-D indefinite integral operator on the device"
    Imat::DAT2

    function DiscontinuousSpectralElementGrid(
        topology::AbstractTopology{dim};
        FloatType,
        DeviceArray,
        polynomialorder,
        meshwarp::Function = (x...) -> identity(x),
    ) where {dim}

        N = polynomialorder
        (ξ, ω) = Elements.lglpoints(FloatType, N)
        Imat = indefinite_integral_interpolation_matrix(ξ, ω)
        D = Elements.spectralderivative(ξ)

        (vmap⁻, vmap⁺) = mappings(
            N,
            topology.elemtoelem,
            topology.elemtoface,
            topology.elemtoordr,
        )

        (vmaprecv, nabrtovmaprecv) = Meshes.commmapping(
            N,
            topology.ghostelems,
            topology.ghostfaces,
            topology.nabrtorecv,
        )
        (vmapsend, nabrtovmapsend) = Meshes.commmapping(
            N,
            topology.sendelems,
            topology.sendfaces,
            topology.nabrtosend,
        )

        (vgeo, sgeo) = computegeometry(topology, D, ξ, ω, meshwarp, vmap⁻)
        Np = (N + 1)^dim
        @assert Np == size(vgeo, 1)

        activedofs = zeros(Bool, Np * length(topology.elems))
        activedofs[1:(Np * length(topology.realelems))] .= true
        activedofs[vmaprecv] .= true

        # Create arrays on the device
        vgeo = DeviceArray(vgeo)
        sgeo = DeviceArray(sgeo)
        elemtobndy = DeviceArray(topology.elemtobndy)
        vmap⁻ = DeviceArray(vmap⁻)
        vmap⁺ = DeviceArray(vmap⁺)
        vmapsend = DeviceArray(vmapsend)
        vmaprecv = DeviceArray(vmaprecv)
        activedofs = DeviceArray(activedofs)
        ω = DeviceArray(ω)
        D = DeviceArray(D)
        Imat = DeviceArray(Imat)

        # FIXME: There has got to be a better way!
        DAT1 = typeof(ω)
        DAT2 = typeof(D)
        DAT3 = typeof(vgeo)
        DAT4 = typeof(sgeo)
        DAI1 = typeof(vmapsend)
        DAI2 = typeof(elemtobndy)
        DAI3 = typeof(vmap⁻)
        TOP = typeof(topology)

        new{
            FloatType,
            dim,
            N,
            Np,
            DeviceArray,
            DAT1,
            DAT2,
            DAT3,
            DAT4,
            DAI1,
            DAI2,
            DAI3,
            TOP,
        }(
            topology,
            vgeo,
            sgeo,
            elemtobndy,
            vmap⁻,
            vmap⁺,
            vmaprecv,
            vmapsend,
            nabrtovmaprecv,
            nabrtovmapsend,
            DeviceArray(topology.interiorelems),
            DeviceArray(topology.exteriorelems),
            activedofs,
            ω,
            D,
            Imat,
        )
    end
end

"""
    referencepoints(::DiscontinuousSpectralElementGrid)

Returns the 1D interpolation points used for the reference element.
"""
function referencepoints(
    ::DiscontinuousSpectralElementGrid{T, dim, N},
) where {T, dim, N}
    ξ, _ = Elements.lglpoints(T, N)
    ξ
end

"""
    min_node_distance(::DiscontinuousSpectralElementGrid,
                      direction::Direction=EveryDirection()))

Returns an approximation of the minimum node distance in physical space along
the reference coordinate directions.  The direction controls which reference
directions are considered.
"""
function min_node_distance(
    grid::DiscontinuousSpectralElementGrid{T, dim, N},
    direction::Direction = EveryDirection(),
) where {T, dim, N}
    topology = grid.topology
    nrealelem = length(topology.realelems)

    if nrealelem > 0
        Nq = N + 1
        Nqk = dim == 2 ? 1 : Nq
        device = grid.vgeo isa Array ? CPU() : CUDADevice()
        min_neighbor_distance = similar(grid.vgeo, Nq^dim, nrealelem)
        event = Event(device)
        event = kernel_min_neighbor_distance!(device, min(Nq * Nq * Nqk, 1024))(
            Val(N),
            Val(dim),
            direction,
            min_neighbor_distance,
            grid.vgeo,
            topology.realelems;
            ndrange = (Nq * Nq * Nqk * nrealelem),
            dependencies = (event,),
        )
        wait(device, event)
        locmin = minimum(min_neighbor_distance)
    else
        locmin = typemax(T)
    end

    MPI.Allreduce(locmin, min, topology.mpicomm)
end

"""
    get_z(grid, z_scale = 1)

Get the Gauss-Lobatto points along the Z-coordinate.

 - `grid`: DG grid
 - `z_scale`: multiplies `z-coordinate`
"""
function get_z(
    grid::DiscontinuousSpectralElementGrid{T, dim, N},
    z_scale = 1,
) where {T, dim, N}
    return reshape(grid.vgeo[(1:((N + 1)^2):((N + 1)^3)), _x3, :], :) * z_scale
end

function Base.getproperty(G::DiscontinuousSpectralElementGrid, s::Symbol)
    if s ∈ keys(vgeoid)
        vgeoid[s]
    elseif s ∈ keys(sgeoid)
        sgeoid[s]
    else
        getfield(G, s)
    end
end

function Base.propertynames(G::DiscontinuousSpectralElementGrid)
    (
        fieldnames(DiscontinuousSpectralElementGrid)...,
        keys(vgeoid)...,
        keys(sgeoid)...,
    )
end
