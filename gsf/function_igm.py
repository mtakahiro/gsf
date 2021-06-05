from scipy import asarray as ar,exp
import numpy as np
import sys
from scipy.integrate import simps

def madau_igm_abs(xtmp, ytmp, zin, cosmo=None, xLL=1216.):
	'''
	Purpose
	-------
	Apply IMG-attenuation of Madau (1995) of zin to the input flux.

	Parameters
	----------
	xtmp : float array
		Rest-frame wavelength, in AA.
	ytmp : float array
		flux, in f_lambda.
	zin : target redshift of IGM application

	Returns
	-------
	IGM attenuated flux.
	'''
	if cosmo == None:
		from astropy.cosmology import WMAP9 as cosmo

	tau = np.zeros(len(xtmp), dtype='float')
	#xLL = 912.
	xobs = xtmp * (1.*zin)
	ytmp_abs = np.zeros(len(ytmp), 'float')

	NH = get_column(zin, cosmo)
	tau = (NH/1.6e17) * (xtmp/xLL)**(3.)
	con = (xtmp<xLL)
	ytmp_abs[con] = ytmp[con] * np.exp(-tau[con])
	con = (xtmp>=xLL)
	ytmp_abs[con] = ytmp[con]

	return ytmp_abs


def get_H(x,a):
	'''
    Voigt function
	'''
    I = integrate.quad(lambda y: np.exp(-y**2)/(a**2 + (x - y)**2),-np.inf, np.inf)[0]
    return (a/np.pi)*I


def get_sig_lya(lam_o, z_s, T=1e4, c=3e18):
	'''
	Parameters
	----------
	lam_o : float array
		Observed wavelength, in AA.

	'''
	nu0 = 2.466e15 #Hz. Lya freq.
	delnuL = 9.936e7  #Hz. Natural Line Width.

	nu = c / (lam_o * 1e-8)

	# Assume sigma_Lya = 100km/s.
	#sigma_lya = 100.
	Vth = 12.85 * (T/1e4)**(1/2) * 1e5 # cm/s
	delnuD = (Vth/c) * nu0

	x = (nu - nu0)/delnuD
	a = delnuL / (2.*delnuD)
	H = get_H(x,a)

	sig_lya = 1.041e-13 * (T/1e4)**(-1/2) * H / np.sqrt(np.pi)

	return sig_lya


def get_nH(z):
	'''
    Returns
	-------
	HI density in IGM.
	'''
    try:
    	nH = np.zeros(len(z),dtype='float')
    except:
    	nH = 0

    # From Cen & Haiman 2000
    nH = 8.5e-5 * ((1.+z)/8)**3 # in cm^-3

    return nH


def get_column(zin, cosmo, Mpc_cm=3.08568025e+24, z_r=6.0, delz=0.1):
	'''

	Returns
	-------
	HI column density of IGM at zin, in cm^-3.

	'''
	z = np.arange(z_r, zin, delz)
	try:
		nH = np.zeros(len(z),dtype='float')
	except:
		nH = 0

	# From Cen & Haiman 2000
	nH = 8.5e-5 * ((1.+z)/8)**3 # in cm^-3
	NH = 0
	for zz in range(len(z)):
		d1 = cosmo.luminosity_distance(z[zz]-delz).value#, **cosmo)
		d2 = cosmo.luminosity_distance(z[zz]+delz).value#, **cosmo)
		dx = (d2 - d1) * Mpc_cm
		NH += nH[zz] * dx/(1.+z[zz])

	return NH
