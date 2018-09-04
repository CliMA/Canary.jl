using MPI

"""
    linearpartition(n, p, np)

Partition the range `1:n` into `np` pieces and return the `p`th piece as a
range.

This will provide an equal partition when `n` is divisible by `np` and
otherwise the ranges will have lengths of either `floor(Int, n/np)` or
`ceil(Int, n/np)`.
"""
linearpartition(n, p, np) = range(div((p-1)*n, np) + 1, stop=div(p*n, np))

"""
    hilbertcode(Y::AbstractArray{T}; bits=8sizeof(T)) where T

Given an array of axes coordinates `Y` stored as `bits`-bit integers
the function returns the Hilbert integer `H`.

The encoding of the Hilbert integer is best described by example.
If 5-bits are used from each of 3 coordinates then the function performs

     X[2]|                       H[0] = A B C D E
         | /X[1]       ------->  H[1] = F G H I J
    axes |/                      H[2] = K L M N O
         0------ X[0]                   high low

where the 15-bit Hilbert integer = `A B C D E F G H I J K L M N O` is stored
in `H`

This function is based on public domain code from John Skilling which can be
found in <https://dx.doi.org/10.1063/1.1751381>.
"""
function hilbertcode(Y::AbstractArray{T}; bits=8sizeof(T)) where T
  # Below is Skilling's AxestoTranspose
  X = deepcopy(Y)
  n = length(X)
  M = one(T) << (bits-1)

  Q = M
  for j = 1:bits-1
    P = Q - one(T)
    for i = 1:n
      if X[i] & Q != zero(T)
        X[1] ⊻= P
      else
        t = (X[1] ⊻ X[i]) & P
        X[1] ⊻= t
        X[i] ⊻= t
      end
    end
    Q >>>= one(T)
  end

  for i = 2:n
    X[i] ⊻= X[i - 1]
  end

  t = zero(T)
  Q = M
  for j = 1:bits-1
    if X[n] & Q != zero(T)
      t ⊻= Q - one(T)
    end
    Q >>>= one(T)
  end

  for i = 1:n
    X[i] ⊻= t
  end

  # Below we transpose X and store it in H, i.e.:
  #
  #   X[0] = A D G J M               H[0] = A B C D E
  #   X[1] = B E H K N   <------->   H[1] = F G H I J
  #   X[2] = C F I L O               H[2] = K L M N O
  #
  # The 15-bit Hilbert integer is then = A B C D E F G H I J K L M N O
  H = zero(X)
  for i = 0:n-1, j = 0:bits-1
    k = i * bits + j
    bit = (X[n - mod(k,n)] >>> div(k,n)) & one(T)
    H[n - i] |= (bit << j)
  end

  return H
end

"""
    centroidtocode(comm::MPI.Comm, elemtocorner; coortocode, CT)

Returns a code for each element based on its centroid.

These element codes can be used to determine a linear ordering for the
partition function.

The communicator `comm` is used to calculate the bounding box for representing
the centroids in coordinates of type `CT`, defaulting to `CT=UInt64`.  These
integer coordinates are converted to a code using the function `coortocode`,
which defaults to `hilbertcode`.

The array containing the element corner coordinates, `elemtocorner`, is used
to compute the centroids.  `elemtocorner` is a dimension by number of corners
by number of elements array.
"""
function centroidtocode(comm::MPI.Comm, elemtocorner; coortocode=hilbertcode,
                        CT=UInt64)
  (d, nvert, nelem) = size(elemtocorner)
  T = eltype(elemtocorner)

  centroids = sum(elemtocorner, dims=2) ./ nvert

  centroidmin = (nelem > 0) ? minimum(centroids, dims=3) : fill(typemax(T),d)
  centroidmax = (nelem > 0) ? maximum(centroids, dims=3) : fill(typemin(T),d)

  centroidmin = MPI.allreduce(centroidmin, MPI.MIN, comm)
  centroidmax = MPI.allreduce(centroidmax, MPI.MAX, comm)

  centroidsize = centroidmax - centroidmin

  code = Array{CT}(undef, d, nelem)
  for e = 1:nelem
    c = (centroids[:,1,e] .- centroidmin) ./ centroidsize
    X = CT.(floor.(typemax(CT).*BigFloat.(c, 16sizeof(CT))))
    code[:,e] = coortocode(X)
  end

  code
end

"""
    brickmesh(x, periodic; part=1, numparts=1)

Generate a brick mesh with coordinates given by the tuple `x` and the
periodic dimensions given by the `periodic` tuple.

The brick can optionally be partitioned into `numparts` and this returns
partition `part`.  This is a simple Cartesian partition and further
partitioning (e.g, based on a space-filling curve) should be done before the
mesh is used for computation.

# Examples

We can build a 3 by 2 element two-dimensional mesh that is periodic in the
\$x_2\$-direction with
```jldoctest brickmesh
julia> using Canary
julia> (etv, etc, fc) = brickmesh((2:5,4:6), (false,true));
```
This returns the mesh structure for

             x_2

              ^
              |
             6-  9----10----11----12
              |  |     |     |     |
              |  |  4  |  5  |  6  |
              |  |     |     |     |
             5-  5-----6-----7-----8
              |  |     |     |     |
              |  |  1  |  2  |  3  |
              |  |     |     |     |
             4-  1-----2-----3-----4
              |
              +--|-----|-----|-----|--> x_1
                 2     3     4     5

The (number of corners by number of elements) array `etv` gives the global
vertex number for the corners of each element.
```jldoctest brickmesh
julia> etv
4×6 Array{Int64,2}:
 1  2  3   5   6   7
 2  3  4   6   7   8
 5  6  7   9  10  11
 6  7  8  10  11  12
```
Note that the vertices are listed in Cartesian order.

The (dimension by number of corners by number of elements) array `etc` gives
the coordinates of the corners of each element.
```jldoctes brickmesh
julia> etc
2×4×6 Array{Int64,3}:
[:, :, 1] =
 2  3  2  3
 4  4  5  5

[:, :, 2] =
 3  4  3  4
 4  4  5  5

[:, :, 3] =
 4  5  4  5
 4  4  5  5

[:, :, 4] =
 2  3  2  3
 5  5  6  6

[:, :, 5] =
 3  4  3  4
 5  5  6  6

[:, :, 6] =
 4  5  4  5
 5  5  6  6
```

Finally, the periodic face connections are given in `fc` which is a list of
arrays, one for each connection.
Each array in the list is given in the format `[e, f, vs...]` where
 - `e`  is the element number;
 - `f`  is the face number; and
 - `vs` is the global vertices that face associated with.
I the example
```jldoctest brickmesh
julia> fc
3-element Array{Array{Int64,1},1}:
 [4, 4, 1, 2]
 [5, 4, 2, 3]
 [6, 4, 3, 4]
```
we see that face `4` of element `5` is associated with vertices `[2 3]` (the
vertices for face `1` of element `2`).
"""
function brickmesh(x, periodic; part=1, numparts=1)
  @assert length(x) == length(periodic)
  @assert length(x) >= 1
  @assert 1 <= part <= numparts

  T = promote_type(eltype.(x)...)
  d = length(x)
  nvert = 2^d

  nelemdim = length.(x).-1
  elemlocal = linearpartition(prod(nelemdim), part, numparts)

  elemtovert = Array{Int}(undef, nvert, length(elemlocal))
  elemtocoord = Array{T}(undef, d, nvert, length(elemlocal))
  faceconnections = Array{Array{Int, 1}}(undef, 0)

  verts = LinearIndices(ntuple(j->1:length(x[j]), d))
  elems = CartesianIndices(ntuple(j->1:length(x[j])-1, d))

  for (e, ec) = enumerate(elems[elemlocal])
    corners = CartesianIndices(ntuple(j->ec[j]:ec[j]+1, d))
    for (v, vc) = enumerate(corners)
      elemtovert[v,e] = verts[vc]

      for j = 1:d
        elemtocoord[j,v,e] = x[j][vc[j]]
      end
    end

    for i=1:d
      if periodic[i] && ec[i]==nelemdim[i]
        js = ntuple(j->(i==j) ? 1 : 1:2, d)
        neighcorners = CartesianIndices(ntuple(j->(i==j) ?
                                               (1:2) : ec[j]:ec[j]+1, d))
        push!(faceconnections,
              vcat(e, 2i, vec(verts[neighcorners[js...]])))
      end
    end
  end

  (elemtovert, elemtocoord, faceconnections)
end

"""
    parallelsortcolumns(comm::MPI.Comm, A;
                        alg::Base.Sort.Algorithm=Base.Sort.DEFAULT_UNSTABLE,
                        lt=isless,
                        by=identity,
                        rev::Union{Bool,Nothing}=nothing)

Sorts the columns of the distributed matrix `A`.

See the documentation of `sort!` for a description of the keyword arguments.

This function assumes `A` has the same number of rows on each MPI rank but can have
a different number of columns.
"""
function parallelsortcolumns(comm::MPI.Comm, A;
                             alg::Base.Sort.Algorithm=Base.Sort.DEFAULT_UNSTABLE,
                             lt=isless,
                             by=identity,
                             rev::Union{Bool,Nothing}=nothing)

  m, n = size(A)
  T = eltype(A)

  csize = MPI.Comm_size(comm)
  crank = MPI.Comm_rank(comm)
  croot = 0

  A = sortslices(A, dims=2, alg=alg, lt=lt, by=by, rev=rev)

  npivots = clamp(n, 0, csize)
  pivots = T[A[i, div(n*p,npivots)+1] for i=1:m, p=0:npivots-1]
  pivotcounts = MPI.Allgather(Cint(length(pivots)), comm)
  pivots = MPI.Allgatherv(pivots, pivotcounts, comm)
  pivots = reshape(pivots, m, div(length(pivots),m))
  pivots = sortslices(pivots, dims=2, alg=alg, lt=lt, by=by, rev=rev)

  # if we don't have any pivots then we must have zero columns
  if size(pivots) == (m, 0)
    return A
  end

  pivots =
    [pivots[i, div(div(length(pivots),m)*r,csize)+1] for i=1:m, r=0:csize-1]

  cols = map(i->view(A,:,i), 1:n)
  sendstarts = [(i<=csize) ? (searchsortedfirst(cols, pivots[:,i], lt=lt,
                                                by=by, rev=rev)-1)*m+1 : n*m+1
                for i=1:csize+1]
  sendcounts = [Cint(sendstarts[i+1]-sendstarts[i]) for i=1:csize]

  B = []
  for r = 0:csize-1
    counts = MPI.Allgather(sendcounts[r+1], comm)
    c = MPI.Gatherv(view(A, sendstarts[r+1]:sendstarts[r+2]-1), counts, r, comm)
    if r == crank
      B = c
    end
  end
  B = reshape(B, m, div(length(B),m))

  sortslices(B, dims=2, alg=alg, lt=lt, by=by, rev=rev)
end
