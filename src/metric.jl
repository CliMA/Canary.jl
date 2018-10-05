"""
    creategrid!(x, elemtocoord, r)

Create a 1-D grid using `elemtocoord` (see [`brickmesh`](@ref)) using the 1-D
`(-1, 1)` reference coordinates `r`. The element grids are filled using linear
interpolation of the element coordinates.

If `Nq = length(r)` and `nelem = size(elemtocoord, 3)` then the preallocated
array `x` should be `Nq * nelem == length(x)`.
"""
function creategrid!(x, e2c, r)
  (d, nvert, nelem) = size(e2c)
  @assert d == 1
  Nq = length(r)
  x = reshape(x, (Nq, nelem))

  # linear blend
  for e = 1:nelem
    for i = 1:Nq
      x[i, e] = ((1 - r[i]) * e2c[1, 1, e] + (1 + r[i])e2c[1, 2, e]) / 2
    end
  end
  nothing
end

"""
    creategrid!(x, y, elemtocoord, r)

Create a 2-D tensor product grid using `elemtocoord` (see [`brickmesh`](@ref))
using the 1-D `(-1, 1)` reference coordinates `r`. The element grids are filled
using bilinear interpolation of the element coordinates.

If `Nq = length(r)` and `nelem = size(elemtocoord, 3)` then the preallocated
arrays `x` and `y` should be `Nq^2 * nelem == size(x) == size(y)`.
"""
function creategrid!(x, y, e2c, r)
  (d, nvert, nelem) = size(e2c)
  @assert d == 2
  Nq = length(r)
  x = reshape(x, (Nq, Nq, nelem))
  y = reshape(y, (Nq, Nq, nelem))

  # bilinear blend of corners
  for (f, n) = zip((x, y), 1:d)
    for e = 1:nelem
      for j = 1:Nq
        for i = 1:Nq
          f[i, j, e] = ((1 - r[i]) * (1 - r[j]) * e2c[n, 1, e] +
                        (1 + r[i]) * (1 - r[j]) * e2c[n, 2, e] +
                        (1 - r[i]) * (1 + r[j]) * e2c[n, 3, e] +
                        (1 + r[i]) * (1 + r[j]) * e2c[n, 4, e]) / 4
        end
      end
    end
  end
  nothing
end

"""
    creategrid!(x, y, z, elemtocoord, r)

Create a 3-D tensor product grid using `elemtocoord` (see [`brickmesh`](@ref))
using the 1-D `(-1, 1)` reference coordinates `r`. The element grids are filled
using trilinear interpolation of the element coordinates.

If `Nq = length(r)` and `nelem = size(elemtocoord, 3)` then the preallocated
arrays `x`, `y`, and `z` should be `Nq^3 * nelem == size(x) == size(y) ==
size(z)`.
"""
function creategrid!(x, y, z, e2c, r)
  (d, nvert, nelem) = size(e2c)
  @assert d == 3
  # TODO: Add asserts?
  Nq = length(r)
  x = reshape(x, (Nq, Nq, Nq, nelem))
  y = reshape(y, (Nq, Nq, Nq, nelem))
  z = reshape(z, (Nq, Nq, Nq, nelem))

  # trilinear blend of corners
  for (f, n) = zip((x,y,z), 1:d)
    for e = 1:nelem
      for k = 1:Nq
        for j = 1:Nq
          for i = 1:Nq
            f[i, j, k, e] = ((1 - r[i]) * (1 - r[j]) *
                               (1 - r[k]) * e2c[n, 1, e] +
                             (1 + r[i]) * (1 - r[j]) *
                               (1 - r[k]) * e2c[n, 2, e] +
                             (1 - r[i]) * (1 + r[j]) *
                               (1 - r[k]) * e2c[n, 3, e] +
                             (1 + r[i]) * (1 + r[j]) *
                               (1 - r[k]) * e2c[n, 4, e] +
                             (1 - r[i]) * (1 - r[j]) *
                               (1 + r[k]) * e2c[n, 5, e] +
                             (1 + r[i]) * (1 - r[j]) *
                               (1 + r[k]) * e2c[n, 6, e] +
                             (1 - r[i]) * (1 + r[j]) *
                               (1 + r[k]) * e2c[n, 7, e] +
                             (1 + r[i]) * (1 + r[j]) *
                               (1 + r[k]) * e2c[n, 8, e]) / 8
          end
        end
      end
    end
  end
  nothing
end

"""
    computemetric!(x, J, ξx, sJ, nx, D)

Compute the 1-D metric terms from the element grid arrays `x`. All the arrays
are preallocated by the user and the (square) derivative matrix `D` should be
consistent with the reference grid `r` used in [`creategrid!`](@ref).

If `Nq = size(D, 1)` and `nelem = div(length(x), Nq)` then the volume arrays
`x`, `J`, and `ξx` should all have length `Nq * nelem`.  Similarly, the face
arrays `sJ` and `nx` should be of length `nface * nelem` with `nface = 2`.
"""
function computemetric!(x, J, ξx, sJ, nx, D)
  Nq = size(D, 1)
  nelem = div(length(J), Nq)
  x = reshape(x, (Nq, nelem))
  J = reshape(J, (Nq, nelem))
  ξx = reshape(ξx, (Nq, nelem))
  nface = 2
  nx = reshape(nx, (1, nface, nelem))
  sJ = reshape(sJ, (1, nface, nelem))

  d = 1

  for e = 1:nelem
    J[:, e] = D * x[:, e]
  end
  ξx .=  1 ./ J

  nx[1, 1, :] .= -sign.(J[ 1, :])
  nx[1, 2, :] .=  sign.(J[Nq, :])
  sJ .= 1
  nothing
end

"""
    computemetric!(x, y, J, ξx, ηx, ξy, ηy, sJ, nx, ny, D)

Compute the 2-D metric terms from the element grid arrays `x` and `y`. All the
arrays are preallocated by the user and the (square) derivative matrix `D`
should be consistent with the reference grid `r` used in [`creategrid!`](@ref).

If `Nq = size(D, 1)` and `nelem = div(length(x), Nq^2)` then the volume arrays
`x`, `y`, `J`, `ξx`, `ηx`, `ξy`, and `ηy` should all be of size `(Nq, Nq,
nelem)`.  Similarly, the face arrays `sJ`, `nx`, and `ny` should be of size
`(Nq, nface, nelem)` with `nface = 4`.
"""
function computemetric!(x, y, J, ξx, ηx, ξy, ηy, sJ, nx, ny, D)
  Nq = size(D, 1)
  nelem = div(length(J), Nq^2)
  d = 2
  x = reshape(x, (Nq, Nq, nelem))
  y = reshape(y, (Nq, Nq, nelem))
  J = reshape(J, (Nq, Nq, nelem))
  ξx = reshape(ξx, (Nq, Nq, nelem))
  ηx = reshape(ηx, (Nq, Nq, nelem))
  ξy = reshape(ξy, (Nq, Nq, nelem))
  ηy = reshape(ηy, (Nq, Nq, nelem))
  nface = 4
  nx = reshape(nx, (Nq, nface, nelem))
  ny = reshape(ny, (Nq, nface, nelem))
  sJ = reshape(sJ, (Nq, nface, nelem))

  # we can resuse this storage
  (ys, yr, xs, xr) = (ξx, ηx, ξy, ηy)

  for e = 1:nelem
    for n = 1:Nq
      # @views xr[:, n, e] = D * x[:, n, e]
      for j = 1:Nq
        xr[j, n, e] = 0
        for i = 1:Nq
          xr[j, n, e] += D[j, i] * x[i, n, e]
        end
      end

      # @views xs[n, :, e] = D * x[n, :, e]
      for j = 1:Nq
        xs[n, j, e] = 0
        for i = 1:Nq
          xs[n, j, e] += D[j, i] * x[n, i, e]
        end
      end

      # @views yr[:, n, e] = D * y[:, n, e]
      for j = 1:Nq
        yr[j, n, e] = 0
        for i = 1:Nq
          yr[j, n, e] += D[j, i] * y[i, n, e]
        end
      end

      # @views ys[n, :, e] = D * y[n, :, e]
      for j = 1:Nq
        ys[n, j, e] = 0
        for i = 1:Nq
          ys[n, j, e] += D[j, i] * y[n, i, e]
        end
      end

    end
  end
  @. J = xr * ys - yr * xs
  @. ξx =  ys / J
  @. ηx = -yr / J
  @. ξy = -xs / J
  @. ηy =  xr / J

  @views nx[:, 1, :] .= -J[ 1,  :, :] .* ξx[ 1,  :, :]
  @views ny[:, 1, :] .= -J[ 1,  :, :] .* ξy[ 1,  :, :]
  @views nx[:, 2, :] .=  J[Nq,  :, :] .* ξx[Nq,  :, :]
  @views ny[:, 2, :] .=  J[Nq,  :, :] .* ξy[Nq,  :, :]
  @views nx[:, 3, :] .= -J[ :,  1, :] .* ηx[ :,  1, :]
  @views ny[:, 3, :] .= -J[ :,  1, :] .* ηy[ :,  1, :]
  @views nx[:, 4, :] .=  J[ :, Nq, :] .* ηx[ :, Nq, :]
  @views ny[:, 4, :] .=  J[ :, Nq, :] .* ηy[ :, Nq, :]
  @. sJ = hypot(nx, ny)
  @. nx = nx / sJ
  @. ny = ny / sJ

  nothing
end

"""
    computemetric!(x, y, z, J, ξx, ηx, ζx, ξy, ηy, ζy, ξz, ηz, ζz, sJ, nx,
                   ny, nz, D)

Compute the 3-D metric terms from the element grid arrays `x`, `y`, and `z`. All
the arrays are preallocated by the user and the (square) derivative matrix `D`
should be consistent with the reference grid `r` used in [`creategrid!`](@ref).

If `Nq = size(D, 1)` and `nelem = div(length(x), Nq^3)` then the volume arrays
`x`, `y`, `z`, `J`, `ξx`, `ηx`, `ζx`, `ξy`, `ηy`, `ζy`, `ξz`, `ηz`, and `ζz`
should all be of length `Nq^3 * nelem`.  Similarly, the face arrays `sJ`, `nx`,
`ny`, and `nz` should be of size `Nq^2 * nface * nelem` with `nface = 6`.

The curl invariant formulation of Kopriva (2006), equation 37, is used.

Reference:
  Kopriva, David A. "Metric identities and the discontinuous spectral element
  method on curvilinear meshes." Journal of Scientific Computing 26.3 (2006):
  301-327. <https://doi.org/10.1007/s10915-005-9070-8>
"""
function computemetric!(x, y, z, J, ξx, ηx, ζx, ξy, ηy, ζy, ξz, ηz, ζz, sJ, nx,
                        ny, nz, D)

  Nq = size(D, 1)
  nelem = div(length(J), Nq^3)

  x = reshape(x, (Nq, Nq, Nq, nelem))
  y = reshape(y, (Nq, Nq, Nq, nelem))
  z = reshape(z, (Nq, Nq, Nq, nelem))
  J = reshape(J, (Nq, Nq, Nq, nelem))
  ξx = reshape(ξx, (Nq, Nq, Nq, nelem))
  ηx = reshape(ηx, (Nq, Nq, Nq, nelem))
  ζx = reshape(ζx, (Nq, Nq, Nq, nelem))
  ξy = reshape(ξy, (Nq, Nq, Nq, nelem))
  ηy = reshape(ηy, (Nq, Nq, Nq, nelem))
  ζy = reshape(ζy, (Nq, Nq, Nq, nelem))
  ξz = reshape(ξz, (Nq, Nq, Nq, nelem))
  ηz = reshape(ηz, (Nq, Nq, Nq, nelem))
  ζz = reshape(ζz, (Nq, Nq, Nq, nelem))

  nface = 6
  nx = reshape(nx, Nq, Nq, nface, nelem)
  ny = reshape(ny, Nq, Nq, nface, nelem)
  nz = reshape(nz, Nq, Nq, nface, nelem)
  sJ = reshape(sJ, Nq, Nq, nface, nelem)

  (xr, xs, xt) = (ξx, ηx, ζx)
  (yr, ys, yt) = (ξy, ηy, ζy)
  (zr, zs, zt) = (ξz, ηz, ζz)

  for e = 1:nelem
    for m = 1:Nq
      for n = 1:Nq
        @views xr[:, n, m, e] = D * x[:, n, m, e]
        @views xs[n, :, m, e] = D * x[n, :, m, e]
        @views xt[n, m, :, e] = D * x[n, m, :, e]
        @views yr[:, n, m, e] = D * y[:, n, m, e]
        @views ys[n, :, m, e] = D * y[n, :, m, e]
        @views yt[n, m, :, e] = D * y[n, m, :, e]
        @views zr[:, n, m, e] = D * z[:, n, m, e]
        @views zs[n, :, m, e] = D * z[n, :, m, e]
        @views zt[n, m, :, e] = D * z[n, m, :, e]
      end
    end
  end


  @. J = (xr * (ys * zt - zs * yt) +
          yr * (zs * xt - xs * zt) +
          zr * (xs * yt - ys * xt))

  JI2 = similar(@view J[:,:,:,1])
  (yzr, yzs, yzt) = (similar(JI2), similar(JI2), similar(JI2))
  (zxr, zxs, zxt) = (similar(JI2), similar(JI2), similar(JI2))
  (xyr, xys, xyt) = (similar(JI2), similar(JI2), similar(JI2))

  for e = 1:nelem
    for k = 1:Nq
      for j = 1:Nq
        for i = 1:Nq
          JI2[i,j,k] = 1 / (2 * J[i,j,k,e])

          yzr[i,j,k] = y[i,j,k,e] * zr[i,j,k,e] - z[i,j,k,e] * yr[i,j,k,e]
          yzs[i,j,k] = y[i,j,k,e] * zs[i,j,k,e] - z[i,j,k,e] * ys[i,j,k,e]
          yzt[i,j,k] = y[i,j,k,e] * zt[i,j,k,e] - z[i,j,k,e] * yt[i,j,k,e]

          zxr[i,j,k] = z[i,j,k,e] * xr[i,j,k,e] - x[i,j,k,e] * zr[i,j,k,e]
          zxs[i,j,k] = z[i,j,k,e] * xs[i,j,k,e] - x[i,j,k,e] * zs[i,j,k,e]
          zxt[i,j,k] = z[i,j,k,e] * xt[i,j,k,e] - x[i,j,k,e] * zt[i,j,k,e]

          xyr[i,j,k] = x[i,j,k,e] * yr[i,j,k,e] - y[i,j,k,e] * xr[i,j,k,e]
          xys[i,j,k] = x[i,j,k,e] * ys[i,j,k,e] - y[i,j,k,e] * xs[i,j,k,e]
          xyt[i,j,k] = x[i,j,k,e] * yt[i,j,k,e] - y[i,j,k,e] * xt[i,j,k,e]
        end
      end
    end
    @views ξx[:, :, :, e] .= 0
    @views ηx[:, :, :, e] .= 0
    @views ζx[:, :, :, e] .= 0
    @views ξy[:, :, :, e] .= 0
    @views ηy[:, :, :, e] .= 0
    @views ζy[:, :, :, e] .= 0
    @views ξz[:, :, :, e] .= 0
    @views ηz[:, :, :, e] .= 0
    @views ζz[:, :, :, e] .= 0
    for m = 1:Nq
      for n = 1:Nq
        @views ξx[n, :, m, e] += D * yzt[n, :, m]
        @views ξx[n, m, :, e] -= D * yzs[n, m, :]

        @views ηx[n, m, :, e] += D * yzr[n, m, :]
        @views ηx[:, n, m, e] -= D * yzt[:, n, m]

        @views ζx[:, n, m, e] += D * yzs[:, n, m]
        @views ζx[n, :, m, e] -= D * yzr[n, :, m]

        @views ξy[n, :, m, e] += D * zxt[n, :, m]
        @views ξy[n, m, :, e] -= D * zxs[n, m, :]

        @views ηy[n, m, :, e] += D * zxr[n, m, :]
        @views ηy[:, n, m, e] -= D * zxt[:, n, m]

        @views ζy[:, n, m, e] += D * zxs[:, n, m]
        @views ζy[n, :, m, e] -= D * zxr[n, :, m]

        @views ξz[n, :, m, e] += D * xyt[n, :, m]
        @views ξz[n, m, :, e] -= D * xys[n, m, :]

        @views ηz[n, m, :, e] += D * xyr[n, m, :]
        @views ηz[:, n, m, e] -= D * xyt[:, n, m]

        @views ζz[:, n, m, e] += D * xys[:, n, m]
        @views ζz[n, :, m, e] -= D * xyr[n, :, m]
      end
    end
    @views ξx[:, :, :, e] = ξx[:, :, :, e] .* JI2
    @views ηx[:, :, :, e] = ηx[:, :, :, e] .* JI2
    @views ζx[:, :, :, e] = ζx[:, :, :, e] .* JI2
    @views ξy[:, :, :, e] = ξy[:, :, :, e] .* JI2
    @views ηy[:, :, :, e] = ηy[:, :, :, e] .* JI2
    @views ζy[:, :, :, e] = ζy[:, :, :, e] .* JI2
    @views ξz[:, :, :, e] = ξz[:, :, :, e] .* JI2
    @views ηz[:, :, :, e] = ηz[:, :, :, e] .* JI2
    @views ζz[:, :, :, e] = ζz[:, :, :, e] .* JI2
  end

  @views nx[:, :, 1, :] .= -J[ 1, :, :, :] .* ξx[ 1, :, :, :]
  @views nx[:, :, 2, :] .=  J[Nq, :, :, :] .* ξx[Nq, :, :, :]
  @views nx[:, :, 3, :] .= -J[ :, 1, :, :] .* ηx[ :, 1, :, :]
  @views nx[:, :, 4, :] .=  J[ :,Nq, :, :] .* ηx[ :,Nq, :, :]
  @views nx[:, :, 5, :] .= -J[ :, :, 1, :] .* ζx[ :, :, 1, :]
  @views nx[:, :, 6, :] .=  J[ :, :,Nq, :] .* ζx[ :, :,Nq, :]

  @views ny[:, :, 1, :] .= -J[ 1, :, :, :] .* ξy[ 1, :, :, :]
  @views ny[:, :, 2, :] .=  J[Nq, :, :, :] .* ξy[Nq, :, :, :]
  @views ny[:, :, 3, :] .= -J[ :, 1, :, :] .* ηy[ :, 1, :, :]
  @views ny[:, :, 4, :] .=  J[ :,Nq, :, :] .* ηy[ :,Nq, :, :]
  @views ny[:, :, 5, :] .= -J[ :, :, 1, :] .* ζy[ :, :, 1, :]
  @views ny[:, :, 6, :] .=  J[ :, :,Nq, :] .* ζy[ :, :,Nq, :]

  @views nz[:, :, 1, :] .= -J[ 1, :, :, :] .* ξz[ 1, :, :, :]
  @views nz[:, :, 2, :] .=  J[Nq, :, :, :] .* ξz[Nq, :, :, :]
  @views nz[:, :, 3, :] .= -J[ :, 1, :, :] .* ηz[ :, 1, :, :]
  @views nz[:, :, 4, :] .=  J[ :,Nq, :, :] .* ηz[ :,Nq, :, :]
  @views nz[:, :, 5, :] .= -J[ :, :, 1, :] .* ζz[ :, :, 1, :]
  @views nz[:, :, 6, :] .=  J[ :, :,Nq, :] .* ζz[ :, :,Nq, :]

  @. sJ = hypot(nx, ny, nz)
  @. nx = nx / sJ
  @. ny = ny / sJ
  @. nz = nz / sJ

  nothing
end

creategrid1d(elemtocoord, r) = creategrid(Val(1), elemtocoord, r)
creategrid2d(elemtocoord, r) = creategrid(Val(2), elemtocoord, r)
creategrid3d(elemtocoord, r) = creategrid(Val(3), elemtocoord, r)

"""
    creategrid(::Val{1}, elemtocoord::AbstractArray{S, 3},
               r::AbstractVector{T}) where {S, T}

Create a grid using `elemtocoord` (see [`brickmesh`](@ref)) using the 1-D `(-1,
1)` reference coordinates `r`. The element grids are filled using bilinear
interpolation of the element coordinates.

The grid is returned as a tuple of with `x` array
"""
function creategrid(::Val{1}, e2c::AbstractArray{S, 3},
                    r::AbstractVector{T}) where {S, T}
  (d, nvert, nelem) = size(e2c)
  @assert d == 1
  Nq = length(r)
  x = Array{T, 2}(undef, Nq, nelem)
  creategrid!(x, e2c, r)
  (x=x, )
end

"""
    creategrid(::Val{2}, elemtocoord::AbstractArray{S, 3},
               r::AbstractVector{T}) where {S, T}

Create a 2-D tensor product grid using `elemtocoord` (see [`brickmesh`](@ref))
using the 1-D `(-1, 1)` reference coordinates `r`. The element grids are filled
using bilinear interpolation of the element coordinates.

The grid is returned as a tuple of the `x` and `y` arrays
"""
function creategrid(::Val{2}, e2c::AbstractArray{S, 3},
                    r::AbstractVector{T}) where {S, T}
  (d, nvert, nelem) = size(e2c)
  @assert d == 2
  Nq = length(r)
  x = Array{T, 3}(undef, Nq, Nq, nelem)
  y = Array{T, 3}(undef, Nq, Nq, nelem)
  creategrid!(x, y, e2c, r)
  (x=x, y=y)
end

"""
    creategrid(::Val{3}, elemtocoord::AbstractArray{S, 3},
               r::AbstractVector{T}) where {S, T}

Create a 3-D tensor product grid using `elemtocoord` (see [`brickmesh`](@ref))
using the 1-D `(-1, 1)` reference coordinates `r`. The element grids are filled
using bilinear interpolation of the element coordinates.

The grid is returned as a tuple of the `x`, `y`, `z` arrays
"""
function creategrid(::Val{3}, e2c::AbstractArray{S, 3},
                    r::AbstractVector{T}) where {S, T}
  (d, nvert, nelem) = size(e2c)
  @assert d == 3
  Nq = length(r)
  x = Array{T, 4}(undef, Nq, Nq, Nq, nelem)
  y = Array{T, 4}(undef, Nq, Nq, Nq, nelem)
  z = Array{T, 4}(undef, Nq, Nq, Nq, nelem)
  creategrid!(x, y, z, e2c, r)
  (x=x, y=y, z=z)
end

"""
    computemetric(x::AbstractArray{T, 2}, D::AbstractMatrix{T}) where T

Compute the 1-D metric terms from the element grid array `x` using the
derivative matrix `D`. The derivative matrix `D` should be consistent with the
reference grid `r` used in [`creategrid!`](@ref).

The metric terms are returned as a 'NamedTuple` of the following arrays:

 - `J` the Jacobian determinant
 - `ξx` derivative ∂r / ∂x'
 - `sJ` the surface Jacobian
 - 'nx` outward pointing unit normal in \$x\$-direction
"""
function computemetric(x::AbstractArray{T, 2},
                       D::AbstractMatrix{T}) where T

  Nq = size(D,1)
  nelem = size(x, 2)
  nface = 2

  J = similar(x)
  ξx = similar(x)

  sJ = Array{T, 3}(undef, 1, nface, nelem)
  nx = Array{T, 3}(undef, 1, nface, nelem)

  computemetric!(x, J, ξx, sJ, nx, D)

  (J=J, ξx=ξx, sJ=sJ, nx=nx)
end


"""
    computemetric(x::AbstractArray{T, 3}, y::AbstractArray{T, 3},
                  D::AbstractMatrix{T}) where T

Compute the 2-D metric terms from the element grid arrays `x` and `y` using the
derivative matrix `D`. The derivative matrix `D` should be consistent with the
reference grid `r` used in [`creategrid!`](@ref).

The metric terms are returned as a 'NamedTuple` of the following arrays:

 - `J` the Jacobian determinant
 - `ξx` derivative ∂r / ∂x'
 - `ηx` derivative ∂s / ∂x'
 - `ξy` derivative ∂r / ∂y'
 - `ηy` derivative ∂s / ∂y'
 - `sJ` the surface Jacobian
 - 'nx` outward pointing unit normal in \$x\$-direction
 - 'ny` outward pointing unit normal in \$y\$-direction
"""
function computemetric(x::AbstractArray{T, 3},
                       y::AbstractArray{T, 3},
                       D::AbstractMatrix{T}) where T
  @assert size(x) == size(y)
  Nq = size(D,1)
  nelem = size(x, 3)
  nface = 4

  J = similar(x)
  ξx = similar(x)
  ηx = similar(x)
  ξy = similar(x)
  ηy = similar(x)

  sJ = Array{T, 3}(undef, Nq, nface, nelem)
  nx = Array{T, 3}(undef, Nq, nface, nelem)
  ny = Array{T, 3}(undef, Nq, nface, nelem)

  computemetric!(x, y, J, ξx, ηx, ξy, ηy, sJ, nx, ny, D)

  (J=J, ξx=ξx, ηx=ηx, ξy=ξy, ηy=ηy, sJ=sJ, nx=nx, ny=ny)
end

"""
    computemetric(x::AbstractArray{T, 3}, y::AbstractArray{T, 3},
                  D::AbstractMatrix{T}) where T

Compute the 3-D metric terms from the element grid arrays `x`, `y`, and `z`
using the derivative matrix `D`. The derivative matrix `D` should be consistent
with the reference grid `r` used in [`creategrid!`](@ref).

The metric terms are returned as a 'NamedTuple` of the following arrays:

 - `J` the Jacobian determinant
 - `ξx` derivative ∂r / ∂x'
 - `ηx` derivative ∂s / ∂x'
 - `ζx` derivative ∂t / ∂x'
 - `ξy` derivative ∂r / ∂y'
 - `ηy` derivative ∂s / ∂y'
 - `ζy` derivative ∂t / ∂y'
 - `ξz` derivative ∂r / ∂z'
 - `ηz` derivative ∂s / ∂z'
 - `ζz` derivative ∂t / ∂z'
 - `sJ` the surface Jacobian
 - 'nx` outward pointing unit normal in \$x\$-direction
 - 'ny` outward pointing unit normal in \$y\$-direction
 - 'nz` outward pointing unit normal in \$z\$-direction

!!! note

   The storage of the volume terms and surface terms from this function are
   slightly different. The volume terms used Cartesian indexing whereas the
   surface terms use linear indexing.
"""
function computemetric(x::AbstractArray{T, 4},
                       y::AbstractArray{T, 4},
                       z::AbstractArray{T, 4},
                       D::AbstractMatrix{T}) where T

  @assert size(x) == size(y) == size(z)
  Nq = size(D,1)
  nelem = size(x, 4)
  nface = 6

  J = similar(x)
  ξx = similar(x)
  ηx = similar(x)
  ζx = similar(x)
  ξy = similar(x)
  ηy = similar(x)
  ζy = similar(x)
  ξz = similar(x)
  ηz = similar(x)
  ζz = similar(x)

  sJ = Array{T, 3}(undef, Nq^2, nface, nelem)
  nx = Array{T, 3}(undef, Nq^2, nface, nelem)
  ny = Array{T, 3}(undef, Nq^2, nface, nelem)
  nz = Array{T, 3}(undef, Nq^2, nface, nelem)

  computemetric!(x, y, z, J, ξx, ηx, ζx, ξy, ηy, ζy, ξz, ηz, ζz, sJ,
                 nx, ny, nz, D)

  (J=J, ξx=ξx, ηx=ηx, ζx=ζx, ξy=ξy, ηy=ηy, ζy=ζy, ξz=ξz, ηz=ηz, ζz=ζz, sJ=sJ,
   nx=nx, ny=ny, nz=nz)
end
