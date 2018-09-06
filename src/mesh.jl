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

  centroids = sum(elemtocorner, dims=2) ./ nvert
  T = eltype(centroids)

  centroidmin = (nelem > 0) ? minimum(centroids, dims=3) : fill(typemax(T),d)
  centroidmax = (nelem > 0) ? maximum(centroids, dims=3) : fill(typemin(T),d)

  centroidmin = MPI.allreduce(centroidmin, MPI.MIN, comm)
  centroidmax = MPI.allreduce(centroidmax, MPI.MAX, comm)
  centroidsize = centroidmax - centroidmin

  # Fix centroidsize to be nonzero.  It can be zero for a couple of reasons.
  # For example, it will be zero if we have just one element.
  if iszero(centroidsize)
    centroidsize = ones(T, d)
  else
    for i = 1:d
      if iszero(centroidsize[i])
        centroidsize[i] = maximum(centroidsize)
      end
    end
  end

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

"""
    getpartition(comm::MPI.Comm, elemtocode)

Returns an equally weighted partition of a distributed set of elements by sorting
their codes given in `elemtocode`.

The codes for each element, `elemtocode`, are given as an array with a single
entry per local element or as a matrix with a column for each local element.

The partition is returned as a tuple three parts:

 - `partsendorder`: permutation of elements into sending order
 - `partsendstarts`: start entries in the send array for each rank
 - `partrecvstarts`: start entries in the receive array for each rank

Note that both `partsendstarts` and `partrecvstarts` are of length
`MPI.Comm_size(comm)+1` where the last entry has the total number of elements
to send or receive, respectively.
"""
getpartition(comm::MPI.Comm, elemtocode::AbstractVector) =
  getpartition(comm, reshape(elemtocode, 1, length(elemtocode)))

function getpartition(comm::MPI.Comm, elemtocode::AbstractMatrix)
  (ncode, nelem) = size(elemtocode)

  csize = MPI.Comm_size(comm)
  crank = MPI.Comm_rank(comm)

  CT = eltype(elemtocode)

  A = CT[elemtocode;                                 # code
         collect(CT, 1:nelem)';                      # original element number
         fill(CT(MPI.Comm_rank(comm)), (1, nelem));  # original rank
         fill(typemax(CT), (1, nelem))]              # new rank
  m, n = size(A)

  # sort by just code
  A = parallelsortcolumns(comm, A)

  # count the distribution of A
  counts = MPI.Allgather(last(size(A)), comm)
  starts = ones(Int, csize+1)
  for i=1:csize
    starts[i+1] = counts[i] + starts[i]
  end

  # loop to determine new rank
  j = range(starts[crank+1], stop=starts[crank+2]-1)
  for r = 0:csize-1
    k = linearpartition(starts[end]-1, r+1, csize)
    o = intersect(k,j) .- (starts[crank+1]-1)
    A[ncode+3,o] .= r
  end

  # sort by original rank and code
  A = sortslices(A, dims=2, by=x->x[[ncode+2,(1:ncode)...]])

  # count number of elements that are going to be sent
  sendcounts = zeros(Cint, csize)
  for i = 1:last(size(A))
    sendcounts[A[ncode+2,i]+1] += m
  end
  sendstarts = ones(Int, csize+1)
  for i=1:csize
    sendstarts[i+1] = sendcounts[i] + sendstarts[i]
  end

  # communicate columns of A to original rank
  B = []
  for r = 0:csize-1
    rcounts = MPI.Allgather(sendcounts[r+1], comm)
    c = MPI.Gatherv(view(A, sendstarts[r+1]:sendstarts[r+2]-1), rcounts, r,
                    comm)
    if r == crank
      B = c
    end
  end
  B = reshape(B, m, div(length(B),m))

  # check to make sure we didn't drop any elements
  @assert nelem == n == size(B)[2]

  partsendcounts = zeros(Cint, csize)
  for i = 1:last(size(B))
    partsendcounts[B[ncode+3,i]+1] += 1
  end
  partsendstarts = ones(Int, csize+1)
  for i=1:csize
    partsendstarts[i+1] = partsendcounts[i] + partsendstarts[i]
  end

  partsendorder = Int.(B[ncode+1,:])

  partrecvcounts = Cint[]
  for r = 0:csize-1
    c = MPI.Gather(partsendcounts[r+1], r, comm)
    if r == crank
      partrecvcounts = c
    end
  end

  partrecvstarts = ones(Int, csize+1)
  for i=1:csize
    partrecvstarts[i+1] = partrecvcounts[i] + partrecvstarts[i]
  end

  partsendorder, partsendstarts, partrecvstarts
end

"""
    partition(comm::MPI.Comm, elemtovert, elemtocoord, faceconnections)

This function takes in a mesh (as returned for example by `brickmesh`) and
returns a Hilbert curve based partitioned mesh.
"""
function partition(comm::MPI.Comm, elemtovert, elemtocoord, faceconnections)
  (d, nvert, nelem) = size(elemtocoord)

  csize = MPI.Comm_size(comm)
  crank = MPI.Comm_rank(comm)

  nface = 2d
  nfacevert = 2^(d-1)

  # Here we expand the list of face connections into a structure that is easy
  # to partition.  The cost is extra memory transfer.  If this becomes a
  # bottleneck something more efficient may be implemented.
  #
  elemtofaceconnect = zeros(eltype(eltype(faceconnections)), nfacevert, nface,
                            nelem)
  for fc in faceconnections
    elemtofaceconnect[:,fc[2],fc[1]] = fc[3:end]
  end

  elemtocode = centroidtocode(comm, elemtocoord; CT=UInt64)
  sendorder, sendstarts, recvstarts = getpartition(comm, elemtocode)

  elemtovert = elemtovert[:,sendorder]
  elemtocoord = elemtocoord[:,:,sendorder]
  elemtofaceconnect = elemtofaceconnect[:,:,sendorder]

  newelemtovert = []
  newelemtocoord = []
  newelemtofaceconnect = []
  for r = 0:csize-1
    sendrange = sendstarts[r+1]:sendstarts[r+2]-1
    rcounts = MPI.Allgather(Cint(length(sendrange)), comm)

    netv = MPI.Gatherv(view(elemtovert, :, sendrange), rcounts.*Cint(nvert),
                       r, comm)

    netc = MPI.Gatherv(view(elemtocoord, :, :, sendrange),
                       rcounts.*Cint(d*nvert), r, comm)

    netfc = MPI.Gatherv(view(elemtofaceconnect, :, :, sendrange),
                        rcounts.*Cint(nfacevert*nface), r, comm)

    if r == crank
      newelemtovert = netv
      newelemtocoord = netc
      newelemtofaceconnect = netfc
    end
  end

  newnelem = recvstarts[end]-1
  newelemtovert = reshape(newelemtovert, nvert, newnelem)
  newelemtocoord = reshape(newelemtocoord, d, nvert, newnelem)
  newelemtofaceconnect = reshape(newelemtofaceconnect, nfacevert, nface,
                                 newnelem)

  # reorder local elements based on code of new elements
  A = UInt64[centroidtocode(comm, newelemtocoord; CT=UInt64);
             collect(1:newnelem)']
  A = sortslices(A, dims=2)
  newsortorder = view(A,d+1,:)
  newelemtovert = newelemtovert[:,newsortorder]
  newelemtocoord = newelemtocoord[:,:,newsortorder]
  newelemtofaceconnect = newelemtofaceconnect[:,:,newsortorder]

  newfaceconnections = similar(faceconnections, 0)
  for e = 1:newnelem, f = 1:nface
    if newelemtofaceconnect[1,f,e] > 0
      push!(newfaceconnections, vcat(e, f, newelemtofaceconnect[:,f,e]))
    end
  end

  (newelemtovert, newelemtocoord, newfaceconnections)
end

"""
    minmaxflip(x, y)

Returns `x, y` sorted lowest to highest and a bool that indicates if a swap
was needed.
"""
minmaxflip(x, y) = y < x ? (y, x, true) : (x, y, false)

"""
    vertsortandorder(a)

Returns `(a)` and an ordering `o==0`.
"""
vertsortandorder(a) = ((a,), 1)

"""
    vertsortandorder(a, b)

Returns sorted vertex numbers `(a,b)` and an ordering `o` depending on the
order needed to sort the elements.  This ordering is given below including the
vetex ordering for faces.

    o=    0      1

        (a,b)  (b,a)

          a      b
          |      |
          |      |
          b      a
"""
function vertsortandorder(a, b)
  a, b, s1 = minmaxflip(a, b)
  o = s1 ? 2 : 1
  ((a, b), o)
end

"""
    vertsortandorder(a, b, c)

Returns sorted vertex numbers `(a,b,c)` and an ordering `o` depending on the
order needed to sort the elements.  This ordering is given below including the
vetex ordering for faces.

    o=     1         2         3         4         5         6

        (a,b,c)   (c,a,b)   (b,c,a)   (b,a,c)   (c,b,a)   (a,c,b)

          /c\\      /b\\      /a\\      /c\\      /a\\      /b\\
         /   \\    /   \\    /   \\    /   \\    /   \\    /   \\
        /a___b\\  /c___a\\  /b___c\\  /b___a\\  /c___b\\  /a___c\\
"""
function vertsortandorder(a, b, c)
  # Use a (Bose-Nelson Algorithm based) sorting network from
  # <http://pages.ripco.net/~jgamble/nw.html> to sort the vertices.
  b, c, s1 = minmaxflip(b, c)
  a, c, s2 = minmaxflip(a, c)
  a, b, s3 = minmaxflip(a, b)

  if     !s1 && !s2 && !s3
    o = 1
  elseif !s1 &&  s2 &&  s3
    o = 2
  elseif  s1 && !s2 &&  s3
    o = 3
  elseif !s1 && !s2 &&  s3
    o = 4
  elseif  s1 &&  s2 &&  s3
    o = 5
  elseif  s1 && !s2 && !s3
    o = 6
  else
    error("Problem finding vertex ordering $((a,b,c)) with flips
          $((s1,s2,s3))")
  end

  ((a, b, c), o)
end

"""
    vertsortandorder(a, b, c, d)

Returns sorted vertex numbers `(a,b,c,d)` and an ordering `o` depending on the
order needed to sort the elements.  This ordering is given below including the
vetex ordering for faces.

    o=   1      2      3      4      5      6      7      8

       (a,b,  (a,c,  (b,a,  (b,d,  (c,a,  (c,d,  (d,b,  (d,c,
        c,d)   b,d)   c,d)   a,c)   d,b)   a,b)   c,a)   b,a)

       c---d  b---d  c---d  a---c  d---b  a---b  c---a  b---a
       |   |  |   |  |   |  |   |  |   |  |   |  |   |  |   |
       a---b  a---c  b---a  b---d  c---a  c---d  d---b  d---c
"""
function vertsortandorder(a, b, c, d)
  # Use a (Bose-Nelson Algorithm based) sorting network from
  # <http://pages.ripco.net/~jgamble/nw.html> to sort the vertices.
  a, b, s1 = minmaxflip(a, b)
  c, d, s2 = minmaxflip(c, d)
  a, c, s3 = minmaxflip(a, c)
  b, d, s4 = minmaxflip(b, d)
  b, c, s5 = minmaxflip(b, c)

 if     !s1 && !s2 && !s3 && !s4 && !s5
   o = 1
 elseif !s1 && !s2 && !s3 && !s4 &&  s5
   o = 2
 elseif  s1 && !s2 && !s3 && !s4 && !s5
   o = 3
 elseif !s1 && !s2 &&  s3 &&  s4 &&  s5
   o = 4
 elseif  s1 &&  s2 && !s3 && !s4 &&  s5
   o = 5
 elseif !s1 && !s2 &&  s3 &&  s4 && !s5
   o = 6
 elseif  s1 &&  s2 &&  s3 &&  s4 &&  s5
   o = 7
 elseif  s1 &&  s2 &&  s3 &&  s4 && !s5
   o = 8
 else
    error("Problem finding vertex ordering $((a,b,c,d))
            with flips $((s1,s2,s3,s4,s5))")
 end

  ((a, b, c, d), o)
end
