from scipy import asarray as ar,exp
import numpy as np
import sys
from scipy.integrate import simps

#
import cosmolopy.distance as cd
import cosmolopy.constants as cc
cosmo = {'omega_M_0' : 0.27, 'omega_lambda_0' : 0.73, 'h' : 0.72}
cosmo = cd.set_omega_k_0(cosmo)
c = 3.e18 # A/s
chimax = 1.
mag0 = 25.0
m0set= mag0
d = 10**(73.6/2.5) # From [ergs/s/cm2/A] to [ergs/s/cm2/Hz]
Mpc_cm = 3.08568025e+24 # cm/Mpc

def madau_igm_abs(xtmp, ytmp, zin):
	#
	# Returns; dust attenuated flux
	#
	# xtmp: RF wavelength
	# ytmp: flux in f_lambda
	# z_in: observed redshift
	#
    tau = np.zeros(len(xtmp), dtype='float32')
    xlya = 1216.
    xLL  = 912.
    xLL  = 1216.
    xobs = xtmp * (1.*zin)
    ytmp_abs = np.zeros(len(ytmp), 'float32')

    NH  = get_column(zin)
    tau = (NH/1.6e17) * (xtmp/xLL)**(3.)
    con = (xtmp<xLL)
    ytmp_abs[con] = ytmp[con] * np.exp(-tau[con])
    con = (xtmp>=xLL)
    ytmp_abs[con] = ytmp[con]

    return ytmp_abs

def get_H(x,a):
    #
    # Voigt function.
    #
    I = integrate.quad(lambda y: np.exp(-y**2)/(a**2 + (x - y)**2),-np.inf, np.inf)[0]
    return (a/np.pi)*I

def get_sig_lya(lam_o, z_s, T=1e4):
    #
    # lam_o : Observed wavelength.
    #
    nu0    = 2.466e15 #Hz. Lya freq.
    delnuL = 9.936e7  #Hz. Natural Line Width.

    nu = c / (lam_o * 1e-8)

    # Assume sigma_Lya = 100km/s.
    #sigma_lya = 100.
    Vth    = 12.85 * (T/1e4)**(1/2) * 1e5 # cm/s
    delnuD = (Vth/c) * nu0

    x = (nu - nu0)/delnuD
    a = delnuL / (2.*delnuD)
    H = get_H(x,a)

    sig_lya = 1.041e-13 * (T/1e4)**(-1/2) * H / np.sqrt(np.pi)

    return sig_lya

def get_nH(z):
    #
    # returns : HI density in IGM.
    #
    try:
        nH = np.zeros(len(z),dtype='float32')
    except:
        nH = 0

    # From Cen & Haiman 2000
    nH = 8.5e-5 * ((1.+z)/8)**3 # in cm^-3

    return nH


def get_column(zin, z_r=6.0):
    #
    # Returns : HI column density in IGM.
    #
    delz = 0.1
    z = np.arange(z_r, zin, delz)
    try:
        nH = np.zeros(len(z),dtype='float32')
    except:
        nH = 0

    # From Cen & Haiman 2000
    nH = 8.5e-5 * ((1.+z)/8)**3 # in cm^-3
    NH = 0
    for zz in range(len(z)):
        d1  = cd.luminosity_distance(z[zz]-delz, **cosmo)
        d2  = cd.luminosity_distance(z[zz]+delz, **cosmo)
        dx  = (d2 - d1) * Mpc_cm
        NH += nH[zz] * dx/(1.+z[zz])

    return NH
