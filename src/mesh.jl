using Base.Iterators: product

"""
    linearpartition(n, p, np)

Partition the range `1:n` into `np` pieces and return the `p`th piece as a
range.

This will provide an equal partition when `n` is divisible by `np` and otherwise
the ranges will have lengths of either `floor(Int, n/np)` or `ceil(Int, n/np)`.
"""
linearpartition(n, p, np) = range(div((p-1)*n, np) + 1, stop=div(p*n, np))

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

The (number of corners by number of elements) array `etv` gives the global vertex
number for the corners of each element.
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
