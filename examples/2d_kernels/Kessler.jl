"""
    Kessler

"""

module Kessler

using PlanetParameters

# Uses Roots.jl from JuliaMath in the saturation adjustment function
using Roots

# Atmospheric equation of state
export kessler_column


"""
    kessler_column

    """
function kessler_column(rho, t, qv, qc, qr, p, z, rainnc, rainncv, dt, icol)

    #::Val{dim}, ::Val{N}, vgeo, Q, mpicomm) where {dim, N}

    DFloat = eltype(rho)
    γ::DFloat       = _γ
    p0::DFloat      = _p0
    R_gas::DFloat   = _R_gas
    c_p::DFloat     = _c_p
    c_v::DFloat     = _c_v
    gravity::DFloat = _gravity

    cp = c_p
    cv = c_v
    
    #NOTICE: many of these hard-coded constants MUST be replaced by the ones
    # in 
    
    const max_cr_sedimentation = 0.75
    const c1                   = 0.001
    const c2                   = 0.001
    const c3                   = 2.2
    const c4                   = 0.875
    const fudge                = 1.0
    const mxfall               = 10.0
    const xlv                  = 2.5e+6
    const ep2                  = 0.6217504

    # constants of Magnus-Tetens formula for saturation water pressure:
    # (see Klemp and Wilhelmson (1978) eq. (2.11),
    # Emanuel (textbook Atmospheric Convection, 1994) eq. 4.4.14)
    const svp1     =   0.6112000
    const svp2     =  17.67000
    const svp3     =  29.65000
    const svpt0    =  273.1500
    const rhowater = 1000.000

    
    prod   = zeros(DFloat, nz)
    vt     = zeros(DFloat, nz)
    prodk  = zeros(DFloat, nz)
    vtden  = zeros(DFloat, nz)
    rdzk   = zeros(DFloat, nz)
    rhok   = zeros(DFloat, nz)
    factor = zeros(DFloat, nz)
    rdzw   = zeros(DFloat, nz)
    
    #Np = (N+1)^dim

    
    #  input arrays
    #
    #  t - potential temperature
    #  qv, qc, qr  - mixing ratio (g/g dry air) of water vapor, cloud water
    #                and rain water
    #  pii         - exner function
    #  dt_in - timestep
    #  z  - height of (t,qv,p,rho) points in meters
    #  dz8w - delta z between t points.
    #
    #  See Klemp and Wilhelmson (1978) Journal of the Atmospheric Sciences
    #  Vol 35, pp 1070-1096 for more details
    #
    #  some more explanations (Andreas Mueller):
    #  es: saturation water pressure according to Magnus-Tetens
    #  qvs: saturation mixing ratio
    #  prod, product: production of qc
    #

    #   f5 = 237.3 * 17.27 * 2.5e6 / cp
    f5     = 17.67*(273.15 - 29.65)*2.5e+6/cp
    ernmax =    0.0
    maxqrp = -100.0

    
    #------------------------------------------------------------------------------
    # parameters for the time split terminal advection
    #------------------------------------------------------------------------------

    max_heating  = 0.0
    max_condense = 0.0
    max_rain     = 0.0
    
    # are all the variables ready for the microphysics?
    # start the microphysics
    # do the sedimentation first
    crmax = 0.0

    #
    # Define region inside the sponge:
    #
    zs  = zmax - bc_zscale
    nz_inside=nz
    #for k = 1:nz
    #    ip     = node_column[icol,k]
    #    if coord[3,ip] <= zs 
    #        nz_inside = k-1
    #    end 
    #end
    
    #=
    #Define region inside the sponge:    
    z         = vgeo[1, 1, _y, 1] 
    zprev     = z
    zs        = zmax - bc_zscale
    nz_inside = nz
    @inbounds for e = 1:nelem
        for j = 1:Nq, i = 1:Nq
            
            z = vgeo[i, j, _y, e]
            
            if (abs(z - zprev) > 1.0e-5)
                if z <= zs
                nzmax          = nzmax + 1
                dataz[nzmax]   = z
                zprev          = z
            end
        end
    end
    
    if(nzmax != nz)
        error(" interpolate_sounding: 1D INTERPOLATION: ops, something is wrong: nz is wrong!\n")
    end
    ###
    =#
    
    #------------------------------------------------------------------------------
    # Terminal velocity calculation and advection, set up coefficients and
    # compute stable timestep
    #------------------------------------------------------------------------------
    do k = 1:nz_inside-1
        rdzk[k] = 1./(z[k+1] - z[k])
    end
    rdzk[nz_inside] = 1.0/(z[nz_inside] - z[nz_inside-1])
    
    do k = 1:nz #nz_inside
        
        prodk[k] = qr[k]
        rhok[k]  = rho[k]
        qrr      = max(0.0, qr[k]*0.001*rhok[k])
        vtden[k] = sqrt(rhok[1]/rhok[k])
        vt[k]    = 36.34*(qrr^0.1364) * vtden[k]
        # vt: terminal fall velocity (Klemp and Wilhelmson (1978) eq. (2.15))
        #       vtmax = max(vt(k), vtmax)
        crmax = max(crmax, vt[k]*dt*rdzk[k])
    end

    nfall         = max(1, int(crmax + 1.0) ) #CM2, Bryan, courant number for big timestep.
    #nfall         = max(1,nint(0.5+crmax/max_cr_sedimentation)) # WRF, courant number for big timestep.
    dtfall        = dt / real(nfall) # splitting so courant number for sedimentation
    time_sediment = dt      # is stable


    #------------------------------------------------------------------------------
    # Terminal velocity calculation and advection
    # Do a time split loop on this for stability.
    #------------------------------------------------------------------------------

    while ( nfall > 0 )

        time_sediment = time_sediment - dtfall
        for k = 1:nz_inside-1
            
            factor[k] = dtfall*rdzk[k]/rhok[k]
            
        end
        factor[nz_inside] = dtfall*rdzk(nz_inside)

        ppt = 0.0

        k       = 1
        ppt     = rhok[k]*prodk[k]*vt[k]*dtfall/rhowater
        rainncv = ppt*1000.0
        rainnc  = rainnc + ppt*1000.0 # unit = mm

        #------------------------------------------------------------------------------
        # Time split loop, Fallout done with flux upstream
        #------------------------------------------------------------------------------
        for k = 1:nz_inside-1
            prodk[k] = prodk[k] - factor[k] &
                * (rhok[k]*prodk[k]*vt[k] &
                   -rhok(k+1)*prodk(k+1)*vt(k+1))
        end
        k = nz_inside
        prodk[k] = prodk[k] - factor[k]*prodk[k]*vt[k]

        
        #------------------------------------------------------------------------------
        # compute new sedimentation velocity, and check/recompute new
        # sedimentation timestep if this isn't the last split step.
        #------------------------------------------------------------------------------
        if ( nfall > 1 ) # this wasn't the last split sedimentation timestep

            nfall = nfall - 1
            crmax = 0.
            for k = 1:nz_inside
                qrr   = max(0.0, prodk[k]*0.001*rhok[k])
                vt[k] = 36.34*(qrr^0.1364) * vtden[k]
                # vt: terminal fall velocity (Klemp and Wilhelmson (1978) eq. (2.15))
                #          vtmax = max(vt[k], vtmax)
                crmax = max(vt[k]*time_sediment*rdzw[k],crmax)
            end

            nfall_new = max(1,nint(0.5+crmax/max_cr_sedimentation))
            if (nfall_new /= nfall )
                nfall  = nfall_new
                dtfall = time_sediment/nfall
            end

        else  # this was the last timestep

            for k = 1:nz_inside
                prod[k] = prodk[k]
            end
            nfall = 0  # exit condition for sedimentation loop
        end
    end

    ##------------------------------------------------------------------------------
    # now the conversion processes
    # Production of rain and deletion of qc
    # Production of qc from supersaturation
    # Evaporation of QR
    #------------------------------------------------------------------------------
    for k = 1:nz_inside
        # autoconversion and accretion: (Klemp and Wilhelmson (1978) eq. (2.13))
        factorn = 1.0 / (1.0 + 2.2*dt*max(0.0, qr[k])^0.875)
        qrprod = qc[k] * (1.0 - factorn) &
            + factorn*0.001*dt*max(qc[k] - 0.001, 0.0)
        qrprod = 0.1*qrprod #only 10% of rain production, SM
        
        rcgs = 0.001*rho[k]

        qc[k] = max(qc[k] - qrprod, 0.0)
        qr[k] = (qr[k] + prod[k] - qr[k])
        qr[k] = max(qr[k] + qrprod,0.0)

        pii=(p[k]/p00)^(rgas/cp)
        temp=t[k]*pii # t: potential temperature, temp: temperature
        pressure=p[k]

        gam     = 2.5e+06/(cp*pii) # see Klemp and Wilhelmson (1978) below eq. (2.9d)

        es      = 611.2*exp(17.67*(temp - 273.15)/(temp - 29.65))
        # es: saturation water pressure according to Magnus-Tetens (see Klemp and Wilhelmson (1978) eq. (2.11),
        # Emanuel (textbook Atmospheric Convection, 1994) eq. 4.4.14)
        qvs     = ep2*es/(pressure - es) # saturation mixing ratio
        
        prod[k] = (qv[k] - qvs)/(1.0 + pressure/(pressure-es)*qvs*f5/(temp-svp3)^2) # production of qc
        
        #>     ern  = min(dt*(((1.6 + 124.9*(rcgs*qr[k])^0.2046) &
        #>          *(rcgs*qr[k])^.525)/(2.55e8/(pressure*qvs) &
        #>          +5.4e5))*(dim(qvs,qv[k])/(rcgs*qvs)), &
        #>          max(-prod[k]-qc[k],0.),qr[k])

        #CM2 Bryan
        ern = (1.6 + 30.3922*((rhok[k]*qr[k])^0.2046))*(1.0-(qv[k]/qvs))*((rhok[k]*qr[k])^0.525)/ ( (2.03e+4 + 9.584e+6/(qvs*p[k])) * rhok[k] )
        
        ern = min(ern*dt,qr[k])

        # ern: evaporation of rain according to Ogura and Takshashi (1971) (see Klemp and Wilhelmson (1978) eq. (2.14))
        
        # Update all variables
        product = max(prod[k], -qc[k]) # deletion of qc cannot be larger than qc
        t [k]   = t[k] + gam*(product - ern)
        qv[k]   = max(qv[k] - product + ern, 0.0)
        qc[k]   =       qc[k] + product
        qr[k]   = qr[k] - ern

    end

return

end
