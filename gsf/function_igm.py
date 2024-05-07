from scipy import asarray as ar,exp
import numpy as np
import sys
from scipy.integrate import simps
from scipy import integrate


def get_XI(z, zend=5, zstart=8):
	'''
	Very simplified model.
	'''
	zs = np.linspace(zend, zstart, 100)
	if z < zend:
		XI = 0
	elif z > zstart:
		XI = 1.
	else:
		XI = (z-zend) / (zstart-zend) * 1.0
	return XI


def get_dtdz(z, zs, dtdzs):
	'''
	'''
	iix = np.argmin(np.abs(zs[:-1]-z))
	return dtdzs[iix]


def masongronke_igm_abs(xtmp, ytmp, zin, cosmo=None, xLL=1216., c=3e18, ckms=3e5, zobs=6, xLLL=1400):
	'''
	Purpose
	-------

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

	tau = np.zeros(len(xtmp), dtype=float)
	xobs = xtmp * (1.*zin)
	ytmp_abs = np.zeros(len(ytmp), float)
	zs = np.linspace(zobs,zin,1000)

	xtmp_obs = xtmp * (1+zin)
	x_HI = 0.8
	T = 1e0 #1e4 # K
	sigma_0 = 5.9e-14 * (T / 1e4)**(-1/2) # cm2
	a_V = 4.7e-4  * (T / 1e4)**(-1/2) # 
	nu_a = 2.46e15 #Hz
	k_B = 1.380649e-23 / (1e3)**2 #m2 kg s-2 K-1 * (km/m)**2 = km2 kg/s2/K
	m_p = 1.67262192e-27 #kilograms
	delta_nu_d = nu_a * np.sqrt(2 * k_B * T / m_p / ckms**2)

	dtdzs = (cosmo.age(zs)[0:-1].value - cosmo.age(zs)[1:].value) * 1e9 * 365.25 * 3600 * 24 / np.diff(zs) # s

	# xLL = 1390
	for ii in range(len(xtmp_obs)):
		if xtmp[ii] < xLL:
			tau[ii] = 100
		elif xtmp[ii] < xLLL:
			nu = c / xtmp_obs[ii] # Hz
			x = (nu - nu_a) / delta_nu_d
			phi_x = get_H(x,a_V) #

			# tau[ii] = integrate.quad(lambda z: ckms * get_dtdz(z, zs, dtdzs) * x_HI * sigma_0 * phi_x, zobs, zin)[0] *  get_column(zin, cosmo)
			tau[ii] = integrate.quad(lambda z: ckms * get_dtdz(z, zs, dtdzs) * x_HI * (1.88e-7 * (1+z)**3) * sigma_0 * phi_x, zobs, zin)[0]
			print(tau[ii], xtmp_obs[ii])
		else:
			tau[ii] = 1e-9

	# R_b1 = 0.0 # Mpc
	# x_D = 0.8 # neutral fraction
	#NH = get_column(zin, cosmo)
	ytmp_abs = ytmp * np.exp(-tau)
	# con = ()
	# ytmp_abs[con] = ytmp[con] * np.exp(-tau[con])

	return ytmp_abs


def dijkstra_igm_abs(xtmp, ytmp, zin, cosmo=None, xLL=1216., ckms=3e5, 
	R_b1=1.0, delta_v_0=600, alpha_x=1.0, x_HI=None, verbose=False,
	zend=5, zstart=8):
	'''
	Purpose
	-------
	Apply IMG-attenuation of Dijikstra (2014).
	https://www.cambridge.org/core/services/aop-cambridge-core/content/view/S1323358014000332

	Parameters
	----------
	xtmp : float array
		Rest-frame wavelength, in AA.
	ytmp : float array
		flux, in f_lambda.
	zin : 
		target redshift of IGM application
	R_b1 : float
		Bubble size, in Mpc

	Returns
	-------
	IGM attenuated flux.
	'''
	import scipy.interpolate as interpolate
	if cosmo == None:
		from astropy.cosmology import WMAP9 as cosmo

	tau = np.zeros(len(xtmp), dtype=float)
	xobs = xtmp * (1.*zin)
	ytmp_abs = np.zeros(len(ytmp), float)

	xtmp_obs = xtmp * (1+zin)

	if x_HI == None:
		x_HI = get_XI(zin, zend=zend, zstart=zstart) # neutral fraction
	else:
		if verbose:
			print('Neutral fraction, x_HI = %.2f, is provided;'%(x_HI))
		
	x_D = alpha_x * x_HI # x_D is not clear..
	delta_lam = (xtmp - xLL) * (zin + 1)
	delta_lam_fine = (np.linspace(900,2000,1000) - xLL) * (zin + 1)

	delta_v = ckms * delta_lam_fine / (xLL * (1.+zin))
	delta_v_b1 = delta_v
	if R_b1>0:
		delta_v_b1 += cosmo.H(zin).value * R_b1 / (1.+zin) # km / (Mpc s) * Mpc

	tau_fine = 2.3 * x_D * (delta_v_b1/delta_v_0)**(-1) * ((1+zin)/10)**(3/2)
	con_tau = (tau_fine < 0) | (delta_v_b1 == 0)
	tau_fine[con_tau] = 100

	fint = interpolate.interp1d(delta_lam_fine, tau_fine, kind='nearest', fill_value="extrapolate")
	tau = fint(delta_lam)

	# import matplotlib.pyplot as plt
	# plt.close()
	# plt.plot(xtmp, tau[:] )
	# plt.xlim(1200,1500)
	# plt.show()
	# hoge

	ytmp_abs = ytmp * np.exp(-tau)

	return ytmp_abs, x_HI


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

	tau = np.zeros(len(xtmp), dtype=float)
	xobs = xtmp * (1.*zin)
	ytmp_abs = np.zeros(len(ytmp), float)

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
	Purpose
	-------
	Get HI density by using Cen & Haiman 2000.

	Returns
	-------
	HI density in IGM, in cm^-3
	'''
	try:
		nH = np.zeros(len(z),dtype='float')
	except:
		nH = 0

	nH = 8.5e-5 * ((1.+z)/8)**3
	return nH


def get_column(zin, cosmo, Mpc_cm=3.08568025e+24, z_r=6.0, delz=0.1):
	'''
	Returns
	-------
	HI column density of IGM at zin, in cm^-2.
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
