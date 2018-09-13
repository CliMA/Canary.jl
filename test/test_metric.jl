const VGEO2D = (x=1, y=2, J=3, rx=4, sx=5, ry=6, sy=7)
const SGEO2D = (sJ = 1, nx = 2, ny = 3)

const VGEO3D = (x = 1, y = 2, z = 3, J = 4, rx = 5, sx = 6, tx = 7, ry = 8,
                sy = 9, ty = 10, rz = 11, sz = 12, tz = 13)
const SGEO3D = (sJ = 1, nx = 2, ny = 3, nz = 4)


@testset "2-D Metric terms" begin
  for T ∈ (Float32, Float64, BigFloat)
    # linear and rotation test
    #{{{
    let
      N = 2

      r, w = Canary.lglpoints(T, N)
      D = Canary.spectralderivative(r)
      Nq = N + 1

      d = 2
      nfaces = 4
      e2c = Array{T, 3}(undef, 2, 4, 4)
      e2c[:, :, 1] = [0 2 0 2;0 0 2 2]
      e2c[:, :, 2] = [2 2 0 0;0 2 0 2]
      e2c[:, :, 3] = [2 0 2 0;2 2 0 0]
      e2c[:, :, 4] = [0 0 2 2;2 0 2 0]
      nelem = size(e2c, 3)

      x_exact = Array{Int, 3}(undef, 3, 3, 4)
      x_exact[:, :, 1] = [0 0 0; 1 1 1; 2 2 2]
      x_exact[:, :, 2] = rotr90(x_exact[:, :, 1])
      x_exact[:, :, 3] = rotr90(x_exact[:, :, 2])
      x_exact[:, :, 4] = rotr90(x_exact[:, :, 3])

      y_exact = Array{Int, 3}(undef, 3, 3, 4)
      y_exact[:, :, 1] = [0 1 2; 0 1 2; 0 1 2]
      y_exact[:, :, 2] = rotr90(y_exact[:, :, 1])
      y_exact[:, :, 3] = rotr90(y_exact[:, :, 2])
      y_exact[:, :, 4] = rotr90(y_exact[:, :, 3])

      J_exact = ones(Int, 3, 3, 4)

      rx_exact = zeros(Int, 3, 3, 4)
      rx_exact[:, :, 1] .= 1
      rx_exact[:, :, 3] .= -1

      ry_exact = zeros(Int, 3, 3, 4)
      ry_exact[:, :, 2] .= 1
      ry_exact[:, :, 4] .= -1

      sx_exact = zeros(Int, 3, 3, 4)
      sx_exact[:, :, 2] .= -1
      sx_exact[:, :, 4] .= 1

      sy_exact = zeros(Int, 3, 3, 4)
      sy_exact[:, :, 1] .= 1
      sy_exact[:, :, 3] .= -1

      sJ_exact = ones(Int, Nq, nfaces, nelem)

      nx_exact = zeros(Int, Nq, nfaces, nelem)
      nx_exact[:, 1, 1] .= -1
      nx_exact[:, 2, 1] .=  1
      nx_exact[:, 3, 2] .=  1
      nx_exact[:, 4, 2] .= -1
      nx_exact[:, 1, 3] .=  1
      nx_exact[:, 2, 3] .= -1
      nx_exact[:, 3, 4] .= -1
      nx_exact[:, 4, 4] .=  1

      ny_exact = zeros(Int, Nq, nfaces, nelem)
      ny_exact[:, 3, 1] .= -1
      ny_exact[:, 4, 1] .=  1
      ny_exact[:, 1, 2] .= -1
      ny_exact[:, 2, 2] .=  1
      ny_exact[:, 3, 3] .=  1
      ny_exact[:, 4, 3] .= -1
      ny_exact[:, 1, 4] .=  1
      ny_exact[:, 2, 4] .= -1

      vgeo = Array{T, 4}(undef, Nq, Nq, length(VGEO2D), nelem)
      sgeo = Array{T, 4}(undef, Nq, nfaces, length(SGEO2D), nelem)
      Canary.creategrid!(ntuple(j->(@view vgeo[:, :, j, :]), d)..., e2c, r)
      Canary.computemetric!(ntuple(j->(@view vgeo[:, :, j, :]), length(VGEO2D))...,
                            ntuple(j->(@view sgeo[:, :, j, :]), length(SGEO2D))...,
      D)

      @test (@view vgeo[:,:,VGEO2D.x,:]) ≈ x_exact
      @test (@view vgeo[:,:,VGEO2D.y,:]) ≈ y_exact
      @test (@view vgeo[:, :, VGEO2D.J, :]) ≈ J_exact
      @test (@view vgeo[:, :, VGEO2D.rx, :]) ≈ rx_exact
      @test (@view vgeo[:, :, VGEO2D.ry, :]) ≈ ry_exact
      @test (@view vgeo[:, :, VGEO2D.sx, :]) ≈ sx_exact
      @test (@view vgeo[:, :, VGEO2D.sy, :]) ≈ sy_exact
      @test (@view sgeo[:, :, SGEO2D.sJ, :]) ≈ sJ_exact
      @test (@view sgeo[:, :, SGEO2D.nx, :]) ≈ nx_exact
      @test (@view sgeo[:, :, SGEO2D.ny, :]) ≈ ny_exact

      nothing
    end
    #}}}

    # Polynomial 2-D test
    #{{{
    let
      N = 4
      T = Float64

      f(r,s) = (9 .* r - (1 .+ r) .* s.^2 + (r .- 1).^2 .* (1 .- s.^2 .+ s.^3),
                10 .* s .+ r.^4 .* (1 .- s) .+ r.^2 .* s .* (1 .+ s))
      fxr(r,s) = 7 .+ s.^2 .- 2 .* s.^3 .+ 2 .* r .* (1 .- s.^2 .+ s.^3)
      fxs(r,s) = -2 .* (1 .+ r) .* s .+ (-1 .+ r).^2 .* s .* (-2 .+ 3 .* s)
      fyr(r,s) = -4 .* r.^3 .* (-1 .+ s) .+ 2 .* r .* s .* (1 .+ s)
      fys(r,s) = 10 .- r.^4 .+ r.^2 .* (1 .+ 2 .* s)

      r, w = Canary.lglpoints(T, N)
      D = Canary.spectralderivative(r)
      Nq = N + 1

      d = 2
      nfaces = 4
      e2c = Array{T, 3}(undef, 2, 4, 1)
      e2c[:, :, 1] = [-1 1 -1 1;-1 -1 1 1]
      nelem = size(e2c, 3)

      vgeo = Array{T, 4}(undef, Nq, Nq, length(VGEO2D), nelem)
      sgeo = Array{T, 4}(undef, Nq, nfaces, length(SGEO2D), nelem)

      Canary.creategrid!(ntuple(j->(@view vgeo[:, :, j, :]), d)..., e2c, r)
      x = @view vgeo[:, :, VGEO2D.x, :]
      y = @view vgeo[:, :, VGEO2D.y, :]

      (xr, xs, yr, ys) = (fxr(x, y), fxs(x,y), fyr(x,y), fys(x,y))
      J = xr .* ys - xs .* yr
      foreach(j->(x[j], y[j]) = f(x[j], y[j]), 1:length(x))

      Canary.computemetric!(ntuple(j->(@view vgeo[:, :, j, :]), length(VGEO2D))...,
                            ntuple(j->(@view sgeo[:, :, j, :]), length(SGEO2D))...,
      D)
      @test J ≈ (@view vgeo[:, :, VGEO2D.J, :])
      @test (@view vgeo[:, :, VGEO2D.rx, :]) ≈  ys ./ J
      @test (@view vgeo[:, :, VGEO2D.sx, :]) ≈ -yr ./ J
      @test (@view vgeo[:, :, VGEO2D.ry, :]) ≈ -xs ./ J
      @test (@view vgeo[:, :, VGEO2D.sy, :]) ≈  xr ./ J

      # TODO: check the normals?
      nx = @view sgeo[:,:,SGEO2D.nx,:]
      ny = @view sgeo[:,:,SGEO2D.ny,:]
      sJ = @view sgeo[:,:,SGEO2D.sJ,:]
      @test hypot.(nx, ny) ≈ ones(T, size(nx))
      @test sJ[:,1,:] .* nx[:,1,:] ≈ -ys[ 1,:,:]
      @test sJ[:,1,:] .* ny[:,1,:] ≈  xs[ 1,:,:]
      @test sJ[:,2,:] .* nx[:,2,:] ≈  ys[Nq,:,:]
      @test sJ[:,2,:] .* ny[:,2,:] ≈ -xs[Nq,:,:]
      @test sJ[:,3,:] .* nx[:,3,:] ≈  yr[:, 1,:]
      @test sJ[:,3,:] .* ny[:,3,:] ≈ -xr[:, 1,:]
      @test sJ[:,4,:] .* nx[:,4,:] ≈ -yr[:,Nq,:]
      @test sJ[:,4,:] .* ny[:,4,:] ≈  xr[:,Nq,:]
    end
  end
  #}}}

  # Constant preserving test?
  #{{{
  let
    N = 4
    T = Float64

    f(r,s) = (9 .* r - (1 .+ r) .* s.^2 + (r .- 1).^2 .* (1 .- s.^2 .+ s.^3),
              10 .* s .+ r.^4 .* (1 .- s) .+ r.^2 .* s .* (1 .+ s))
    fxr(r,s) = 7 .+ s.^2 .- 2 .* s.^3 .+ 2 .* r .* (1 .- s.^2 .+ s.^3)
    fxs(r,s) = -2 .* (1 .+ r) .* s .+ (-1 .+ r).^2 .* s .* (-2 .+ 3 .* s)
    fyr(r,s) = -4 .* r.^3 .* (-1 .+ s) .+ 2 .* r .* s .* (1 .+ s)
    fys(r,s) = 10 .- r.^4 .+ r.^2 .* (1 .+ 2 .* s)

    r, w = Canary.lglpoints(T, N)
    D = Canary.spectralderivative(r)
    Nq = N + 1

    d = 2
    nfaces = 4
    e2c = Array{T, 3}(undef, 2, 4, 1)
    e2c[:, :, 1] = [-1 1 -1 1;-1 -1 1 1]
    nelem = size(e2c, 3)

    vgeo = Array{T, 4}(undef, Nq, Nq, length(VGEO2D), nelem)
    sgeo = Array{T, 4}(undef, Nq, nfaces, length(SGEO2D), nelem)

    Canary.creategrid!(ntuple(j->(@view vgeo[:, :, j, :]), d)..., e2c, r)
    x = @view vgeo[:, :, VGEO2D.x, :]
    y = @view vgeo[:, :, VGEO2D.y, :]

    foreach(j->(x[j], y[j]) = f(x[j], y[j]), 1:length(x))

    Canary.computemetric!(ntuple(j->(@view vgeo[:, :, j, :]), length(VGEO2D))...,
                          ntuple(j->(@view sgeo[:, :, j, :]), length(SGEO2D))...,
                          D)

    (Cx, Cy) = (zeros(T, Nq, Nq), zeros(T, Nq, Nq))

    J  = @view vgeo[:,:,VGEO2D.J ,:]
    rx = @view vgeo[:,:,VGEO2D.rx,:]
    ry = @view vgeo[:,:,VGEO2D.ry,:]
    sx = @view vgeo[:,:,VGEO2D.sx,:]
    sy = @view vgeo[:,:,VGEO2D.sy,:]

    e = 1
    for n = 1:Nq
      Cx[:, n] += D * (J[:, n, e] .* rx[:, n, e])
      Cx[n, :] += D * (J[n, :, e] .* sx[n, :, e])

      Cy[:, n] += D * (J[:, n, e] .* rx[:, n, e])
      Cy[n, :] += D * (J[n, :, e] .* sx[n, :, e])
    end
    @test maximum(abs.(Cx)) ≤ 1000 * eps(T)
    @test maximum(abs.(Cy)) ≤ 1000 * eps(T)
  end
  #}}}
end

@testset "3-D Metric terms" begin
  # linear test
    #{{{
  for T ∈ (Float32, Float64, BigFloat)
    let
      N = 2

      r, w = Canary.lglpoints(T, N)
      D = Canary.spectralderivative(r)
      Nq = N + 1

      d = 3
      nfaces = 6
      e2c = Array{T, 3}(undef, d, 8, 2)
      e2c[:, :, 1] = [0 2 0 2 0 2 0 2;
                      0 0 2 2 0 0 2 2;
                      0 0 0 0 2 2 2 2]
      e2c[:, :, 2] = [2 2 0 0 2 2 0 0;
                      0 2 0 2 0 2 0 2;
                      0 0 0 0 2 2 2 2]

      nelem = size(e2c, 3)

      x_exact = Array{Int, 4}(undef, 3, 3, 3, nelem)
      x_exact[1, :, :, 1] .= 0
      x_exact[2, :, :, 1] .= 1
      x_exact[3, :, :, 1] .= 2
      x_exact[:, 1, :, 2] .= 2
      x_exact[:, 2, :, 2] .= 1
      x_exact[:, 3, :, 2] .= 0

      rx_exact = zeros(Int, 3, 3, 3, nelem)
      rx_exact[:, :, :, 1] .= 1

      ry_exact = zeros(Int, 3, 3, 3, nelem)
      ry_exact[:, :, :, 2] .= 1

      sx_exact = zeros(Int, 3, 3, 3, nelem)
      sx_exact[:, :, :, 2] .= -1

      sy_exact = zeros(Int, 3, 3, 3, nelem)
      sy_exact[:, :, :, 1] .= 1

      tz_exact = ones(Int, 3, 3, 3, nelem)

      y_exact = Array{Int, 4}(undef, 3, 3, 3, nelem)
      y_exact[:, 1, :, 1] .= 0
      y_exact[:, 2, :, 1] .= 1
      y_exact[:, 3, :, 1] .= 2
      y_exact[1, :, :, 2] .= 0
      y_exact[2, :, :, 2] .= 1
      y_exact[3, :, :, 2] .= 2

      z_exact = Array{Int, 4}(undef, 3, 3, 3, nelem)
      z_exact[:, :, 1, 1:2] .= 0
      z_exact[:, :, 2, 1:2] .= 1
      z_exact[:, :, 3, 1:2] .= 2

      J_exact = ones(Int, 3, 3, 3, nelem)

      sJ_exact = ones(Int, Nq, Nq, nfaces, nelem)

      nx_exact = zeros(Int, Nq, Nq, nfaces, nelem)
      nx_exact[:, :, 1, 1] .= -1
      nx_exact[:, :, 2, 1] .=  1
      nx_exact[:, :, 3, 2] .=  1
      nx_exact[:, :, 4, 2] .= -1

      ny_exact = zeros(Int, Nq, Nq, nfaces, nelem)
      ny_exact[:, :, 3, 1] .= -1
      ny_exact[:, :, 4, 1] .=  1
      ny_exact[:, :, 1, 2] .= -1
      ny_exact[:, :, 2, 2] .=  1

      nz_exact = zeros(Int, Nq, Nq, nfaces, nelem)
      nz_exact[:, :, 5, 1:2] .= -1
      nz_exact[:, :, 6, 1:2] .=  1

      vgeo = Array{T, 5}(undef, Nq, Nq, Nq, length(VGEO3D), nelem)
      sgeo = Array{T, 5}(undef, Nq, Nq, nfaces, length(SGEO3D), nelem)
      Canary.creategrid!(ntuple(j->(@view vgeo[:, :, :, j, :]), d)..., e2c, r)
      Canary.computemetric!(ntuple(j->(@view vgeo[:, :, :, j, :]), length(VGEO3D))...,
                            ntuple(j->(@view sgeo[:, :, :, j, :]), length(SGEO3D))...,
      D)

      @test (@view vgeo[:,:,:,VGEO3D.x,:]) ≈ x_exact
      @test (@view vgeo[:,:,:,VGEO3D.y,:]) ≈ y_exact
      @test (@view vgeo[:,:,:,VGEO3D.z,:]) ≈ z_exact
      @test (@view vgeo[:,:,:,VGEO3D.J,:]) ≈ J_exact
      @test (@view vgeo[:,:,:,VGEO3D.rx,:]) ≈ rx_exact
      @test (@view vgeo[:,:,:,VGEO3D.ry,:]) ≈ ry_exact
      @test maximum(abs.(@view vgeo[:,:,:,VGEO3D.rz,:])) ≤ 10 * eps(T)
      @test (@view vgeo[:,:,:,VGEO3D.sx,:]) ≈ sx_exact
      @test (@view vgeo[:,:,:,VGEO3D.sy,:]) ≈ sy_exact
      @test maximum(abs.(@view vgeo[:,:,:,VGEO3D.sz,:])) ≤ 10 * eps(T)
      @test maximum(abs.(@view vgeo[:,:,:,VGEO3D.tx,:])) ≤ 10 * eps(T)
      @test maximum(abs.(@view vgeo[:,:,:,VGEO3D.ty,:])) ≤ 10 * eps(T)
      @test (@view vgeo[:,:,:,VGEO3D.tz,:]) ≈ tz_exact
      @test (@view sgeo[:,:,:,SGEO3D.sJ,:]) ≈ sJ_exact
      @test (@view sgeo[:,:,:,SGEO3D.nx,:]) ≈ nx_exact
      @test (@view sgeo[:,:,:,SGEO3D.ny,:]) ≈ ny_exact
      @test (@view sgeo[:,:,:,SGEO3D.nz,:]) ≈ nz_exact
    end
  end
  #}}}

  # linear test with rotation
  #{{{
  for T ∈ (Float32, Float64, BigFloat)
    θ1 = 2 * T(π) * T( 0.9 )
    θ2 = 2 * T(π) * T(-0.56)
    θ3 = 2 * T(π) * T( 0.33)
    #=
    θ1 = 2 * T(π) * rand(T)
    θ2 = 2 * T(π) * rand(T)
    θ3 = 2 * T(π) * rand(T)
    =#
    #=
    θ1 = T(π) / 6
    θ2 = T(π) / 12
    θ3 = 4 * T(π) / 5
    =#
    Q  = [cos(θ1) -sin(θ1) 0; sin(θ1) cos(θ1) 0; 0 0 1]
    Q *= [cos(θ2) 0 -sin(θ2); 0 1 0; sin(θ2) 0 cos(θ2)]
    Q *= [1 0 0; 0 cos(θ3) -sin(θ3); 0 sin(θ3) cos(θ3)]
    let
      N = 2

      r, w = Canary.lglpoints(T, N)
      D = Canary.spectralderivative(r)
      Nq = N + 1

      d = 3
      nfaces = 6
      e2c = Array{T, 3}(undef, d, 8, 1)
      e2c[:, :, 1] = [0 2 0 2 0 2 0 2;
                      0 0 2 2 0 0 2 2;
                      0 0 0 0 2 2 2 2]
      @views (x, y, z) = (e2c[1, :, 1], e2c[2, :, 1], e2c[3, :, 1])
      for i = 1:length(x)
        (x[i], y[i], z[i]) = Q * [x[i]; y[i]; z[i]]
      end

      nelem = size(e2c, 3)

      xe = Array{T, 4}(undef, 3, 3, 3, nelem)
      xe[1, :, :, 1] .= 0
      xe[2, :, :, 1] .= 1
      xe[3, :, :, 1] .= 2

      ye = Array{T, 4}(undef, 3, 3, 3, nelem)
      ye[:, 1, :, 1] .= 0
      ye[:, 2, :, 1] .= 1
      ye[:, 3, :, 1] .= 2

      ze = Array{T, 4}(undef, 3, 3, 3, nelem)
      ze[:, :, 1, 1] .= 0
      ze[:, :, 2, 1] .= 1
      ze[:, :, 3, 1] .= 2

      for i = 1:length(xe)
        (xe[i], ye[i], ze[i]) = Q * [xe[i]; ye[i]; ze[i]]
      end

      Je = ones(Int, 3, 3, 3, nelem)

      # By construction
      # Q = [xr xs ys; yr ys yt; zr zs zt] = [rx sx tx; ry sy ty; rz sz tz]
      rxe = fill(Q[1,1], 3, 3, 3, nelem)
      rye = fill(Q[2,1], 3, 3, 3, nelem)
      rze = fill(Q[3,1], 3, 3, 3, nelem)
      sxe = fill(Q[1,2], 3, 3, 3, nelem)
      sye = fill(Q[2,2], 3, 3, 3, nelem)
      sze = fill(Q[3,2], 3, 3, 3, nelem)
      txe = fill(Q[1,3], 3, 3, 3, nelem)
      tye = fill(Q[2,3], 3, 3, 3, nelem)
      tze = fill(Q[3,3], 3, 3, 3, nelem)

      sJe = ones(Int, Nq, Nq, nfaces, nelem)

      nxe = zeros(T, Nq, Nq, nfaces, nelem)
      nye = zeros(T, Nq, Nq, nfaces, nelem)
      nze = zeros(T, Nq, Nq, nfaces, nelem)

      fill!(@view(nxe[:,:,1,:]), -Q[1,1])
      fill!(@view(nxe[:,:,2,:]),  Q[1,1])
      fill!(@view(nxe[:,:,3,:]), -Q[1,2])
      fill!(@view(nxe[:,:,4,:]),  Q[1,2])
      fill!(@view(nxe[:,:,5,:]), -Q[1,3])
      fill!(@view(nxe[:,:,6,:]),  Q[1,3])
      fill!(@view(nye[:,:,1,:]), -Q[2,1])
      fill!(@view(nye[:,:,2,:]),  Q[2,1])
      fill!(@view(nye[:,:,3,:]), -Q[2,2])
      fill!(@view(nye[:,:,4,:]),  Q[2,2])
      fill!(@view(nye[:,:,5,:]), -Q[2,3])
      fill!(@view(nye[:,:,6,:]),  Q[2,3])
      fill!(@view(nze[:,:,1,:]), -Q[3,1])
      fill!(@view(nze[:,:,2,:]),  Q[3,1])
      fill!(@view(nze[:,:,3,:]), -Q[3,2])
      fill!(@view(nze[:,:,4,:]),  Q[3,2])
      fill!(@view(nze[:,:,5,:]), -Q[3,3])
      fill!(@view(nze[:,:,6,:]),  Q[3,3])

      vgeo = Array{T, 5}(undef, Nq, Nq, Nq, length(VGEO3D), nelem)
      sgeo = Array{T, 5}(undef, Nq, Nq, nfaces, length(SGEO3D), nelem)
      Canary.creategrid!(ntuple(j->(@view vgeo[:, :, :, j, :]), d)..., e2c, r)
      Canary.computemetric!(ntuple(j->(@view vgeo[:, :, :, j, :]), length(VGEO3D))...,
                            ntuple(j->(@view sgeo[:, :, :, j, :]), length(SGEO3D))...,
      D)

      @test (@view vgeo[:,:,:,VGEO3D.x,:]) ≈ xe
      @test (@view vgeo[:,:,:,VGEO3D.y,:]) ≈ ye
      @test (@view vgeo[:,:,:,VGEO3D.z,:]) ≈ ze
      @test (@view vgeo[:,:,:,VGEO3D.J,:]) ≈ Je
      @test (@view vgeo[:,:,:,VGEO3D.rx,:]) ≈ rxe
      @test (@view vgeo[:,:,:,VGEO3D.ry,:]) ≈ rye
      @test (@view vgeo[:,:,:,VGEO3D.rz,:]) ≈ rze
      @test (@view vgeo[:,:,:,VGEO3D.sx,:]) ≈ sxe
      @test (@view vgeo[:,:,:,VGEO3D.sy,:]) ≈ sye
      @test (@view vgeo[:,:,:,VGEO3D.sz,:]) ≈ sze
      @test (@view vgeo[:,:,:,VGEO3D.tx,:]) ≈ txe
      @test (@view vgeo[:,:,:,VGEO3D.ty,:]) ≈ tye
      @test (@view vgeo[:,:,:,VGEO3D.tz,:]) ≈ tze
      @test (@view sgeo[:,:,:,SGEO3D.sJ,:]) ≈ sJe
      @test (@view sgeo[:,:,:,SGEO3D.nx,:]) ≈ nxe
      @test (@view sgeo[:,:,:,SGEO3D.ny,:]) ≈ nye
      @test (@view sgeo[:,:,:,SGEO3D.nz,:]) ≈ nze
    end
  end
  #}}}

  # Polynomial 3-D test
  #{{{
  for T ∈ (Float32, Float64, BigFloat)
    f(r, s, t) = @.( (s + r*t - (r^2*s^2*t^2)/4,
                      t - ((r*s*t)/2 + 1/2)^3 + 1,
                      r + (r/2 + 1/2)^6*(s/2 + 1/2)^6*(t/2 + 1/2)^6))

    fxr(r, s, t) = @.(t - (r*s^2*t^2)/2)
    fxs(r, s, t) = @.(1 - (r^2*s*t^2)/2)
    fxt(r, s, t) = @.(r - (r^2*s^2*t)/2)
    fyr(r, s, t) = @.(-(3*s*t*((r*s*t)/2 + 1/2)^2)/2)
    fys(r, s, t) = @.(-(3*r*t*((r*s*t)/2 + 1/2)^2)/2)
    fyt(r, s, t) = @.(1 - (3*r*s*((r*s*t)/2 + 1/2)^2)/2)
    fzr(r, s, t) = @.(3*(r/2 + 1/2)^5*(s/2 + 1/2)^6*(t/2 + 1/2)^6 + 1)
    fzs(r, s, t) = @.(3*(r/2 + 1/2)^6*(s/2 + 1/2)^5*(t/2 + 1/2)^6)
    fzt(r, s, t) = @.(3*(r/2 + 1/2)^6*(s/2 + 1/2)^6*(t/2 + 1/2)^5)

    let
      N = 9

      r, w = Canary.lglpoints(T, N)
      D = Canary.spectralderivative(r)
      Nq = N + 1

      d = 3
      nfaces = 6
      e2c = Array{T, 3}(undef, d, 8, 1)
      e2c[:, :, 1] = [-1  1 -1  1 -1  1 -1  1;
                      -1 -1  1  1 -1 -1  1  1;
                      -1 -1 -1 -1  1  1  1  1]

      nelem = size(e2c, 3)

      vgeo = Array{T, 5}(undef, Nq, Nq, Nq, length(VGEO3D), nelem)
      sgeo = Array{T, 5}(undef, Nq, Nq, nfaces, length(SGEO3D), nelem)
      Canary.creategrid!(ntuple(j->(@view vgeo[:, :, :, j, :]), d)..., e2c, r)
      x = @view vgeo[:, :, :, VGEO3D.x, :]
      y = @view vgeo[:, :, :, VGEO3D.y, :]
      z = @view vgeo[:, :, :, VGEO3D.z, :]

      (xr, xs, xt,
       yr, ys, yt,
       zr, zs, zt) = (fxr(x,y,z), fxs(x,y,z), fxt(x,y,z),
                      fyr(x,y,z), fys(x,y,z), fyt(x,y,z),
                      fzr(x,y,z), fzs(x,y,z), fzt(x,y,z))
      J = (xr .* (ys .* zt - yt .* zs) +
           yr .* (zs .* xt - zt .* xs) +
           zr .* (xs .* yt - xt .* ys))

      rx =  (ys .* zt - yt .* zs) ./ J
      ry =  (zs .* xt - zt .* xs) ./ J
      rz =  (xs .* yt - xt .* ys) ./ J
      sx =  (yt .* zr - yr .* zt) ./ J
      sy =  (zt .* xr - zr .* xt) ./ J
      sz =  (xt .* yr - xr .* yt) ./ J
      tx =  (yr .* zs - ys .* zr) ./ J
      ty =  (zr .* xs - zs .* xr) ./ J
      tz =  (xr .* ys - xs .* yr) ./ J

      foreach(j->(x[j], y[j], z[j]) = f(x[j], y[j], z[j]), 1:length(x))

      Canary.computemetric!(ntuple(j->(@view vgeo[:, :, :, j, :]), length(VGEO3D))...,
                            ntuple(j->(@view sgeo[:, :, :, j, :]), length(SGEO3D))...,
      D)

      @test (@view vgeo[:,:,:,VGEO3D.J,:]) ≈ J
      @test (@view vgeo[:,:,:,VGEO3D.rx,:]) ≈ rx
      @test (@view vgeo[:,:,:,VGEO3D.ry,:]) ≈ ry
      @test (@view vgeo[:,:,:,VGEO3D.rz,:]) ≈ rz
      @test (@view vgeo[:,:,:,VGEO3D.sx,:]) ≈ sx
      @test (@view vgeo[:,:,:,VGEO3D.sy,:]) ≈ sy
      @test (@view vgeo[:,:,:,VGEO3D.sz,:]) ≈ sz
      @test (@view vgeo[:,:,:,VGEO3D.tx,:]) ≈ tx
      @test (@view vgeo[:,:,:,VGEO3D.ty,:]) ≈ ty
      @test (@view vgeo[:,:,:,VGEO3D.tz,:]) ≈ tz
      nx = @view sgeo[:,:,:,SGEO3D.nx,:]
      ny = @view sgeo[:,:,:,SGEO3D.ny,:]
      nz = @view sgeo[:,:,:,SGEO3D.nz,:]
      sJ = @view sgeo[:,:,:,SGEO3D.sJ,:]
      @test hypot.(nx, ny, nz) ≈ ones(T, size(nx))
      @test ([sJ[:,:,1,:] .* nx[:,:,1,:], sJ[:,:,1,:] .* ny[:,:,1,:],
              sJ[:,:,1,:] .* nz[:,:,1,:]] ≈
             [-J[ 1,:,:,:] .* rx[ 1,:,:,:], -J[ 1,:,:,:] .* ry[ 1,:,:,:],
              -J[ 1,:,:,:] .* rz[ 1,:,:,:]])
      @test ([sJ[:,:,2,:] .* nx[:,:,2,:], sJ[:,:,2,:] .* ny[:,:,2,:],
              sJ[:,:,2,:] .* nz[:,:,2,:]] ≈
             [ J[Nq,:,:,:] .* rx[Nq,:,:,:],  J[Nq,:,:,:] .* ry[Nq,:,:,:],
               J[Nq,:,:,:] .* rz[Nq,:,:,:]])
      @test sJ[:,:,3,:] .* nx[:,:,3,:] ≈ -J[:, 1,:,:] .* sx[:, 1,:,:]
      @test sJ[:,:,3,:] .* ny[:,:,3,:] ≈ -J[:, 1,:,:] .* sy[:, 1,:,:]
      @test sJ[:,:,3,:] .* nz[:,:,3,:] ≈ -J[:, 1,:,:] .* sz[:, 1,:,:]
      @test sJ[:,:,4,:] .* nx[:,:,4,:] ≈  J[:,Nq,:,:] .* sx[:,Nq,:,:]
      @test sJ[:,:,4,:] .* ny[:,:,4,:] ≈  J[:,Nq,:,:] .* sy[:,Nq,:,:]
      @test sJ[:,:,4,:] .* nz[:,:,4,:] ≈  J[:,Nq,:,:] .* sz[:,Nq,:,:]
      @test sJ[:,:,5,:] .* nx[:,:,5,:] ≈ -J[:,:, 1,:] .* tx[:,:, 1,:]
      @test sJ[:,:,5,:] .* ny[:,:,5,:] ≈ -J[:,:, 1,:] .* ty[:,:, 1,:]
      @test sJ[:,:,5,:] .* nz[:,:,5,:] ≈ -J[:,:, 1,:] .* tz[:,:, 1,:]
      @test sJ[:,:,6,:] .* nx[:,:,6,:] ≈  J[:,:,Nq,:] .* tx[:,:,Nq,:]
      @test sJ[:,:,6,:] .* ny[:,:,6,:] ≈  J[:,:,Nq,:] .* ty[:,:,Nq,:]
      @test sJ[:,:,6,:] .* nz[:,:,6,:] ≈  J[:,:,Nq,:] .* tz[:,:,Nq,:]
    end
  end
  #}}}

  # Constant preserving test
  #{{{
  for T ∈ (Float32, Float64, BigFloat)
    f(r, s, t) = @.( (s + r*t - (r^2*s^2*t^2)/4,
                      t - ((r*s*t)/2 + 1/2)^3 + 1,
                      r + (r/2 + 1/2)^6*(s/2 + 1/2)^6*(t/2 + 1/2)^6))
    let
      N = 5

      r, w = Canary.lglpoints(T, N)
      D = Canary.spectralderivative(r)
      Nq = N + 1

      d = 3
      nfaces = 6
      e2c = Array{T, 3}(undef, d, 8, 1)
      e2c[:, :, 1] = [-1  1 -1  1 -1  1 -1  1;
                      -1 -1  1  1 -1 -1  1  1;
                      -1 -1 -1 -1  1  1  1  1]

      nelem = size(e2c, 3)

      vgeo = Array{T, 5}(undef, Nq, Nq, Nq, length(VGEO3D), nelem)
      sgeo = Array{T, 5}(undef, Nq, Nq, nfaces, length(SGEO3D), nelem)
      Canary.creategrid!(ntuple(j->(@view vgeo[:, :, :, j, :]), d)..., e2c, r)
      x = @view vgeo[:, :, :, VGEO3D.x, :]
      y = @view vgeo[:, :, :, VGEO3D.y, :]
      z = @view vgeo[:, :, :, VGEO3D.z, :]

      foreach(j->(x[j], y[j], z[j]) = f(x[j], y[j], z[j]), 1:length(x))

      Canary.computemetric!(ntuple(j->(@view vgeo[:, :, :, j, :]), length(VGEO3D))...,
                            ntuple(j->(@view sgeo[:, :, :, j, :]), length(SGEO3D))...,
      D)

      (Cx, Cy, Cz) = (zeros(T, Nq, Nq, Nq), zeros(T, Nq, Nq, Nq),
                      zeros(T, Nq, Nq, Nq))

      J  = @view vgeo[:,:,:,VGEO3D.J ,:]
      rx = @view vgeo[:,:,:,VGEO3D.rx,:]
      ry = @view vgeo[:,:,:,VGEO3D.ry,:]
      rz = @view vgeo[:,:,:,VGEO3D.rz,:]
      sx = @view vgeo[:,:,:,VGEO3D.sx,:]
      sy = @view vgeo[:,:,:,VGEO3D.sy,:]
      sz = @view vgeo[:,:,:,VGEO3D.sz,:]
      tx = @view vgeo[:,:,:,VGEO3D.tx,:]
      ty = @view vgeo[:,:,:,VGEO3D.ty,:]
      tz = @view vgeo[:,:,:,VGEO3D.tz,:]

      e = 1
      for m = 1:Nq
        for n = 1:Nq
          Cx[:, n, m] += D * (J[:, n, m, e] .* rx[:, n, m, e])
          Cx[n, :, m] += D * (J[n, :, m, e] .* sx[n, :, m, e])
          Cx[n, m, :] += D * (J[n, m, :, e] .* tx[n, m, :, e])

          Cy[:, n, m] += D * (J[:, n, m, e] .* rx[:, n, m, e])
          Cy[n, :, m] += D * (J[n, :, m, e] .* sx[n, :, m, e])
          Cy[n, m, :] += D * (J[n, m, :, e] .* tx[n, m, :, e])

          Cz[:, n, m] += D * (J[:, n, m, e] .* rx[:, n, m, e])
          Cz[n, :, m] += D * (J[n, :, m, e] .* sx[n, :, m, e])
          Cz[n, m, :] += D * (J[n, m, :, e] .* tx[n, m, :, e])
        end
      end
      @test maximum(abs.(Cx)) ≤ 1000 * eps(T)
      @test maximum(abs.(Cy)) ≤ 1000 * eps(T)
      @test maximum(abs.(Cz)) ≤ 1000 * eps(T)
    end
  end
  #}}}
end
