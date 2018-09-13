"""
    creategrid!(x::AbstractArray{T, 3}, y::AbstractArray{T, 3},
                elemtocoord::AbstractArray{S, 3},
                r::AbstractVector{T}) where {S, T}

Create a 2-D tensor product grid using `elemtocoord` (see [`brickmesh`](@ref))
using the 1-D `(-1, 1)` reference coordinates `r`. The element grids are filled
using bilinear interpolation of the element coordinates.

If `Nq = length(r)` and `nelem = size(elemtocoord, 3)` then the preallocated
arrays `x` and `y` should be `(Nq, Nq, nelem) == size(x) == size(y)`.
"""
function creategrid!(x::AbstractArray{T, 3}, y::AbstractArray{T, 3},
                     e2c::AbstractArray{S, 3},
                     r::AbstractVector{T}) where {S, T}
  (d, nvert, nelem) = size(e2c)
  @assert d == 2
  Nq = length(r)

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
  (x, y)
end

"""
    creategrid!(x::AbstractArray{T, 3}, y::AbstractArray{T, 3},
                z::AbstractArray{T, 3}, elemtocoord::AbstractArray{S, 3},
                r::AbstractVector{T}) where {S, T}

Create a 3-D tensor product grid using `elemtocoord` (see [`brickmesh`](@ref))
using the 1-D `(-1, 1)` reference coordinates `r`. The element grids are filled
using trilinear interpolation of the element coordinates.

If `Nq = length(r)` and `nelem = size(elemtocoord, 3)` then the preallocated
arrays `x`, `y`, and `z` should be `(Nq, Nq, Nq, nelem) == size(x) == size(y) ==
size(z)`.
"""
function creategrid!(x::AbstractArray{T, 4}, y::AbstractArray{T, 4},
                     z::AbstractArray{T, 4}, e2c::AbstractArray{S, 3},
                     r::AbstractVector{T}) where {S, T}
  (d, nvert, nelem) = size(e2c)
  @assert d == 3
  # TODO: Add asserts?
  Nq = length(r)

  # trilinear blend of corners
  for (f, n) = zip((x,y,z), 1:d)
    for e = 1:nelem
      for k = 1:Nq
        for j = 1:Nq
          for i = 1:Nq
            f[i, j, k, e] = ((T(1) - r[i]) * (T(1) - r[j]) *
                               (T(1) - r[k]) * e2c[n, 1, e] +
                             (T(1) + r[i]) * (T(1) - r[j]) *
                               (T(1) - r[k]) * e2c[n, 2, e] +
                             (T(1) - r[i]) * (T(1) + r[j]) *
                               (T(1) - r[k]) * e2c[n, 3, e] +
                             (T(1) + r[i]) * (T(1) + r[j]) *
                               (T(1) - r[k]) * e2c[n, 4, e] +
                             (T(1) - r[i]) * (T(1) - r[j]) *
                               (T(1) + r[k]) * e2c[n, 5, e] +
                             (T(1) + r[i]) * (T(1) - r[j]) *
                               (T(1) + r[k]) * e2c[n, 6, e] +
                             (T(1) - r[i]) * (T(1) + r[j]) *
                               (T(1) + r[k]) * e2c[n, 7, e] +
                             (T(1) + r[i]) * (T(1) + r[j]) *
                               (T(1) + r[k]) * e2c[n, 8, e]) / T(8)
          end
        end
      end
    end
  end
  (x, y, z)
end

"""
    computemetric!(x::AbstractArray{T, 3}, y::AbstractArray{T, 3},
                   J::AbstractArray{T, 3},
                   rx::AbstractArray{T, 3}, sx::AbstractArray{T, 3},
                   ry::AbstractArray{T, 3}, sy::AbstractArray{T, 3},
                   sJ::AbstractArray{T, 3},
                   nx::AbstractArray{T, 3}, ny::AbstractArray{T, 3},
                   D::AbstractMatrix{T}) where T

Compute the 2-D metric terms from the element grid arrays `x` and `y`. All the
arrays are preallocated by the user and the (square) derivative matrix `D`
should be consistent with the reference grid `r` used in [`creategrid!`](@ref).

If `Nq = size(D, 1)` and `nelem = size(x, 3)` then the volume arrays `x`, `y`,
`J`, `rx`, `sx`, `ry`, and `sy` should all be of size `(Nq, Nq, nelem)`.
Similarly, the face arrays `sJ`, `nx`, and `ny` should be of size `(Nq, nfaces,
nelem)` with `nfaces = 4`.
"""
function computemetric!(x::AbstractArray{T, 3}, y::AbstractArray{T, 3},
                        J::AbstractArray{T, 3},
                        rx::AbstractArray{T, 3}, sx::AbstractArray{T, 3},
                        ry::AbstractArray{T, 3}, sy::AbstractArray{T, 3},
                        sJ::AbstractArray{T, 3},
                        nx::AbstractArray{T, 3}, ny::AbstractArray{T, 3},
                        D::AbstractMatrix{T}) where T
  nelem = size(J, 3)
  Nq = size(D, 1)
  d = 2

  # we can resuse this storage
  (ys, yr, xs, xr) = (rx, sx, ry, sy)

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
  @. rx =  ys / J
  @. sx = -yr / J
  @. ry = -xs / J
  @. sy =  xr / J

  @views nx[:, 1, :] .= -J[ 1,  :, :] .* rx[ 1,  :, :]
  @views ny[:, 1, :] .= -J[ 1,  :, :] .* ry[ 1,  :, :]
  @views nx[:, 2, :] .=  J[Nq,  :, :] .* rx[Nq,  :, :]
  @views ny[:, 2, :] .=  J[Nq,  :, :] .* ry[Nq,  :, :]
  @views nx[:, 3, :] .= -J[ :,  1, :] .* sx[ :,  1, :]
  @views ny[:, 3, :] .= -J[ :,  1, :] .* sy[ :,  1, :]
  @views nx[:, 4, :] .=  J[ :, Nq, :] .* sx[ :, Nq, :]
  @views ny[:, 4, :] .=  J[ :, Nq, :] .* sy[ :, Nq, :]
  @. sJ = hypot(nx, ny)
  @. nx = nx / sJ
  @. ny = ny / sJ

  (J, rx, sx, ry, sy, sJ, nx, ny)
end

"""
    computemetric!(x::AbstractArray{T, 4}, y::AbstractArray{T, 4},
                   z::AbstractArray{T, 4}, J::AbstractArray{T, 4},
                   rx::AbstractArray{T, 4}, sx::AbstractArray{T, 4},
                   tx::AbstractArray{T, 4} ry::AbstractArray{T, 4},
                   sy::AbstractArray{T, 4}, ty::AbstractArray{T, 4}
                   rz::AbstractArray{T, 4}, sz::AbstractArray{T, 4},
                   tz::AbstractArray{T, 4} sJ::AbstractArray{T, 4},
                   nx::AbstractArray{T, 4}, ny::AbstractArray{T, 4},
                   nz::AbstractArray{T, 4}, D::AbstractMatrix{T}) where T

Compute the 3-D metric terms from the element grid arrays `x`, `y`, and `z`. All
the arrays are preallocated by the user and the (square) derivative matrix `D`
should be consistent with the reference grid `r` used in [`creategrid!`](@ref).

If `Nq = size(D, 1)` and `nelem = size(x, 4)` then the volume arrays `x`, `y`,
`z`, `J`, `rx`, `sx`, `tx`, `ry`, `sy`, `ty`, `rz`, `sz`, and `tz` should all be
of size `(Nq, Nq, Nq, nelem)`.  Similarly, the face arrays `sJ`, `nx`, `ny`, and
`nz` should be of size `(Nq, Nq, nfaces, nelem)` with `nfaces = 6`.

The curl invariant formulation of Kopriva (2006), equation 37, is used.

Reference:
  Kopriva, David A. "Metric identities and the discontinuous spectral element
  method on curvilinear meshes." Journal of Scientific Computing 26.3 (2006):
  301-327. <https://doi.org/10.1007/s10915-005-9070-8>
"""
function computemetric!(x::AbstractArray{T, 4},
                        y::AbstractArray{T, 4},
                        z::AbstractArray{T, 4},
                        J::AbstractArray{T, 4},
                        rx::AbstractArray{T, 4},
                        sx::AbstractArray{T, 4},
                        tx::AbstractArray{T, 4},
                        ry::AbstractArray{T, 4},
                        sy::AbstractArray{T, 4},
                        ty::AbstractArray{T, 4},
                        rz::AbstractArray{T, 4},
                        sz::AbstractArray{T, 4},
                        tz::AbstractArray{T, 4},
                        sJ::AbstractArray{T, 4},
                        nx::AbstractArray{T, 4},
                        ny::AbstractArray{T, 4},
                        nz::AbstractArray{T, 4},
                        D::AbstractMatrix{T}) where T

  nelem = size(J, 4)
  Nq = size(D, 1)
  (xr, xs, xt) = (rx, sx, tx)
  (yr, ys, yt) = (ry, sy, ty)
  (zr, zs, zt) = (rz, sz, tz)

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

  # rx .= (Ds * yzt_zyt - Dt * yzs_zys) ./ (T(2) * J)
  # sx .= (Dt * yzr_zyr - Dr * yzt_zyt) ./ (T(2) * J)
  # tx .= (Dr * yzs_zys - Ds * yzr_zyr) ./ (T(2) * J)

  # ry .= (Ds * zxt_xzt - Dt * zxs_xzs) ./ (T(2) * J)
  # sy .= (Dt * zxr_xzr - Dr * zxt_xzt) ./ (T(2) * J)
  # ty .= (Dr * zxs_xzs - Ds * zxr_xzr) ./ (T(2) * J)

  # rz .= (Ds * xyt_yxt - Dt * xys_yxs) ./ (T(2) * J)
  # sz .= (Dt * xyr_yxr - Dr * xyt_yxt) ./ (T(2) * J)
  # tz .= (Dr * xys_yxs - Ds * xyr_yxr) ./ (T(2) * J)

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
    @views rx[:, :, :, e] .= 0
    @views sx[:, :, :, e] .= 0
    @views tx[:, :, :, e] .= 0
    @views ry[:, :, :, e] .= 0
    @views sy[:, :, :, e] .= 0
    @views ty[:, :, :, e] .= 0
    @views rz[:, :, :, e] .= 0
    @views sz[:, :, :, e] .= 0
    @views tz[:, :, :, e] .= 0
    for m = 1:Nq
      for n = 1:Nq
        @views rx[n, :, m, e] += D * yzt[n, :, m]
        @views rx[n, m, :, e] -= D * yzs[n, m, :]

        @views sx[n, m, :, e] += D * yzr[n, m, :]
        @views sx[:, n, m, e] -= D * yzt[:, n, m]

        @views tx[:, n, m, e] += D * yzs[:, n, m]
        @views tx[n, :, m, e] -= D * yzr[n, :, m]

        @views ry[n, :, m, e] += D * zxt[n, :, m]
        @views ry[n, m, :, e] -= D * zxs[n, m, :]

        @views sy[n, m, :, e] += D * zxr[n, m, :]
        @views sy[:, n, m, e] -= D * zxt[:, n, m]

        @views ty[:, n, m, e] += D * zxs[:, n, m]
        @views ty[n, :, m, e] -= D * zxr[n, :, m]

        @views rz[n, :, m, e] += D * xyt[n, :, m]
        @views rz[n, m, :, e] -= D * xys[n, m, :]

        @views sz[n, m, :, e] += D * xyr[n, m, :]
        @views sz[:, n, m, e] -= D * xyt[:, n, m]

        @views tz[:, n, m, e] += D * xys[:, n, m]
        @views tz[n, :, m, e] -= D * xyr[n, :, m]
      end
    end
    @views rx[:, :, :, e] = rx[:, :, :, e] .* JI2
    @views sx[:, :, :, e] = sx[:, :, :, e] .* JI2
    @views tx[:, :, :, e] = tx[:, :, :, e] .* JI2
    @views ry[:, :, :, e] = ry[:, :, :, e] .* JI2
    @views sy[:, :, :, e] = sy[:, :, :, e] .* JI2
    @views ty[:, :, :, e] = ty[:, :, :, e] .* JI2
    @views rz[:, :, :, e] = rz[:, :, :, e] .* JI2
    @views sz[:, :, :, e] = sz[:, :, :, e] .* JI2
    @views tz[:, :, :, e] = tz[:, :, :, e] .* JI2
  end

  @views nx[:, :, 1, :] .= -J[ 1, :, :, :] .* rx[ 1, :, :, :]
  @views nx[:, :, 2, :] .=  J[Nq, :, :, :] .* rx[Nq, :, :, :]
  @views nx[:, :, 3, :] .= -J[ :, 1, :, :] .* sx[ :, 1, :, :]
  @views nx[:, :, 4, :] .=  J[ :,Nq, :, :] .* sx[ :,Nq, :, :]
  @views nx[:, :, 5, :] .= -J[ :, :, 1, :] .* tx[ :, :, 1, :]
  @views nx[:, :, 6, :] .=  J[ :, :,Nq, :] .* tx[ :, :,Nq, :]

  @views ny[:, :, 1, :] .= -J[ 1, :, :, :] .* ry[ 1, :, :, :]
  @views ny[:, :, 2, :] .=  J[Nq, :, :, :] .* ry[Nq, :, :, :]
  @views ny[:, :, 3, :] .= -J[ :, 1, :, :] .* sy[ :, 1, :, :]
  @views ny[:, :, 4, :] .=  J[ :,Nq, :, :] .* sy[ :,Nq, :, :]
  @views ny[:, :, 5, :] .= -J[ :, :, 1, :] .* ty[ :, :, 1, :]
  @views ny[:, :, 6, :] .=  J[ :, :,Nq, :] .* ty[ :, :,Nq, :]

  @views nz[:, :, 1, :] .= -J[ 1, :, :, :] .* rz[ 1, :, :, :]
  @views nz[:, :, 2, :] .=  J[Nq, :, :, :] .* rz[Nq, :, :, :]
  @views nz[:, :, 3, :] .= -J[ :, 1, :, :] .* sz[ :, 1, :, :]
  @views nz[:, :, 4, :] .=  J[ :,Nq, :, :] .* sz[ :,Nq, :, :]
  @views nz[:, :, 5, :] .= -J[ :, :, 1, :] .* tz[ :, :, 1, :]
  @views nz[:, :, 6, :] .=  J[ :, :,Nq, :] .* tz[ :, :,Nq, :]

  @. sJ = hypot(nx, ny, nz)
  @. nx = nx / sJ
  @. ny = ny / sJ
  @. nz = nz / sJ

  (J, rx, sx, tx, ry, sy, ty, rz, sz, tz, sJ, nx, ny, nz)
end
