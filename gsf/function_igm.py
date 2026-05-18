# from scipy import asarray as ar,exp
import numpy as np
# import sys
# from scipy.integrate import simps
from scipy import integrate
import os


class inoue_igm(object):
	def __init__(self, MB):
		self._scale_tau = 1.0
		# Load LAF coefficients
		data_path = MB.config_path
		laf_file = os.path.join(data_path, "LAFcoeff.txt")
		data = np.loadtxt(laf_file, unpack=True)
		_, lam, alf1, alf2, alf3 = data
		self._lam = lam[:, np.newaxis]
		self._alf1 = alf1[:, np.newaxis]
		self._alf2 = alf2[:, np.newaxis]
		self._alf3 = alf3[:, np.newaxis]

		# Load DLA coefficients
		dla_file = os.path.join(data_path, "DLAcoeff.txt")
		data = np.loadtxt(dla_file, unpack=True)
		_, lam, adla1, adla2 = data
		self._adla1 = adla1[:, np.newaxis]
		self._adla2 = adla2[:, np.newaxis]
		self._tau = None
		self._zin = 99

	def tau_laf(self, redshift, lam_obs,
				z1_laf = 1.2,
				z2_laf = 4.7
				):
		"""Compute the Lyman series and Lyman-alpha forest optical depth.

		Args:
			redshift (float): Source redshift.
			lam_obs (array): Observed-frame wavelengths in Angstroms.

		Returns:
			array: Optical depth due to the Lyman-alpha forest.
		"""
		# Strip units for the following calculations
		lam = self._lam

		tau_laf_value = np.zeros_like(lam_obs * lam).T

		# Conditions based on observed lam and redshift
		cond0 = lam_obs < lam * (1 + redshift)
		cond1 = cond0 & (lam_obs < lam * (1 + z1_laf))
		cond2 = cond0 & (
			(lam_obs >= lam * (1 + z1_laf)) & (lam_obs < lam * (1 + z2_laf))
		)
		cond3 = cond0 & (lam_obs >= lam * (1 + z2_laf))

		tau_laf_value = np.zeros_like(lam_obs * lam)
		tau_laf_value[cond1] += ((self._alf1 / lam**1.2) * lam_obs**1.2)[cond1]
		tau_laf_value[cond2] += ((self._alf2 / lam**3.7) * lam_obs**3.7)[cond2]
		tau_laf_value[cond3] += ((self._alf3 / lam**5.5) * lam_obs**5.5)[cond3]

		return tau_laf_value.sum(axis=0)


	def tau_dla(self, redshift, lam_obs, z1_dla = 2.0):
		"""Compute the Lyman series and Damped Lyman-alpha (DLA) optical depth.

		Args:
			redshift (float): Source redshift.
			lam_obs (array): Observed-frame wavelengths in Angstroms.

		Returns:
			array: Optical depth due to DLA.
		"""
		# Strip units for the following calculations
		lam = self._lam

		tau_dla_value = np.zeros_like(lam_obs * lam)

		# Conditions based on observed wavelength and redshift
		cond0 = (lam_obs < lam * (1 + redshift)) & (
			lam_obs < lam * (1.0 + z1_dla)
		)
		cond1 = (lam_obs < lam * (1 + redshift)) & ~(
			lam_obs < lam * (1.0 + z1_dla)
		)

		tau_dla_value[cond0] += ((self._adla1 / lam**2) * lam_obs**2)[cond0]
		tau_dla_value[cond1] += ((self._adla2 / lam**3) * lam_obs**3)[cond1]

		return tau_dla_value.sum(axis=0)


	def tau_lc_dla(self, redshift, lam_obs,
				z1_dla = 2.0,
				lam_l = 911.8
				):
		"""Compute the Lyman continuum optical depth for DLA.

		Args:
			redshift (float): Source redshift.
			lam_obs (array): Observed-frame wavelengths in Angstroms.

		Returns:
			array: Optical depth due to Lyman continuum for DLA.
		"""
		# Strip units for the following calculations
		tau_lc_dla_value = np.zeros_like(lam_obs)

		cond0 = lam_obs < lam_l * (1.0 + redshift)
		if redshift < z1_dla:
			tau_lc_dla_value[cond0] = (
				0.2113 * (1.0 + redshift) ** 2
				- 0.07661
				* (1.0 + redshift) ** 2.3
				* (lam_obs[cond0] / lam_l) ** (-0.3)
				- 0.1347 * (lam_obs[cond0] / lam_l) ** 2
			)
		else:
			cond1 = lam_obs >= lam_l * (1.0 + z1_dla)

			tau_lc_dla_value[cond0 & cond1] = (
				0.04696 * (1.0 + redshift) ** 3
				- 0.01779
				* (1.0 + redshift) ** 3.3
				* (lam_obs[cond0 & cond1] / lam_l) ** (-0.3)
				- 0.02916 * (lam_obs[cond0 & cond1] / lam_l) ** 3
			)
			tau_lc_dla_value[cond0 & ~cond1] = (
				0.6340
				+ 0.04696 * (1.0 + redshift) ** 3
				- 0.01779
				* (1.0 + redshift) ** 3.3
				* (lam_obs[cond0 & ~cond1] / lam_l) ** (-0.3)
				- 0.1347 * (lam_obs[cond0 & ~cond1] / lam_l) ** 2
				- 0.2905 * (lam_obs[cond0 & ~cond1] / lam_l) ** (-0.3)
			)

		return tau_lc_dla_value


	def tau_lc_laf(self, redshift, lam_obs,
				z1_laf = 1.2,
				z2_laf = 4.7,
				lam_l = 911.8
				):
		"""Compute the Lyman continuum optical depth for LAF.

		Args:
			redshift (float): Source redshift.
			lam_obs (array): Observed-frame wavelengths in Angstroms.

		Returns:
			array: Optical depth due to Lyman continuum for LAF.
		"""
		# Strip units for the following calculations
		tau_lc_laf_value = np.zeros_like(lam_obs)

		cond0 = lam_obs < lam_l * (1.0 + redshift)

		if redshift < z1_laf:
			tau_lc_laf_value[cond0] = 0.3248 * (
				(lam_obs[cond0] / lam_l) ** 1.2
				- (1.0 + redshift) ** -0.9 * (lam_obs[cond0] / lam_l) ** 2.1
			)
		elif redshift < z2_laf:
			cond1 = lam_obs >= lam_l * (1 + z1_laf)
			tau_lc_laf_value[cond0 & cond1] = 2.545e-2 * (
				(1.0 + redshift) ** 1.6
				* (lam_obs[cond0 & cond1] / lam_l) ** 2.1
				- (lam_obs[cond0 & cond1] / lam_l) ** 3.7
			)
			tau_lc_laf_value[cond0 & ~cond1] = (
				2.545e-2
				* (1.0 + redshift) ** 1.6
				* (lam_obs[cond0 & ~cond1] / lam_l) ** 2.1
				+ 0.3248 * (lam_obs[cond0 & ~cond1] / lam_l) ** 1.2
				- 0.2496 * (lam_obs[cond0 & ~cond1] / lam_l) ** 2.1
			)
		else:

			cond1 = lam_obs > lam_l * (1.0 + z2_laf)
			cond2 = (lam_obs >= lam_l * (1.0 + z1_laf)) & (
				lam_obs < lam_l * (1.0 + z2_laf)
			)
			cond3 = lam_obs < lam_l * (1.0 + z1_laf)

			tau_lc_laf_value[cond0 & cond1] = 5.221e-4 * (
				(1.0 + redshift) ** 3.4
				* (lam_obs[cond0 & cond1] / lam_l) ** 2.1
				- (lam_obs[cond0 & cond1] / lam_l) ** 5.5
			)
			tau_lc_laf_value[cond0 & cond2] = (
				5.221e-4
				* (1.0 + redshift) ** 3.4
				* (lam_obs[cond0 & cond2] / lam_l) ** 2.1
				+ 0.2182 * (lam_obs[cond0 & cond2] / lam_l) ** 2.1
				- 2.545e-2 * (lam_obs[cond0 & cond2] / lam_l) ** 3.7
			)
			tau_lc_laf_value[cond0 & cond3] = (
				5.221e-4
				* (1.0 + redshift) ** 3.4
				* (lam_obs[cond0 & cond3] / lam_l) ** 2.1
				+ 0.3248 * (lam_obs[cond0 & cond3] / lam_l) ** 1.2
				- 3.140e-2 * (lam_obs[cond0 & cond3] / lam_l) ** 2.1
			)

		return tau_lc_laf_value


	def calculate_general_tau(self, redshift, lam_obs):
		"""Compute the total IGM optical depth.

		Args:
			redshift (float): Redshift to evaluate IGM absorption.
			lam_obs (array): Observed-frame wavelengths in Angstroms.

		Returns:
			array: Total IGM absorption optical depth.
		"""
		tau_ls = self.tau_laf(redshift, lam_obs) + self.tau_dla(redshift, lam_obs)
		tau_lc = self.tau_lc_laf(redshift, lam_obs) + self.tau_lc_dla(redshift, lam_obs)

		# Upturn at short wavelengths, low-z
		# k = 1./100
		# l0 = 600-6/k
		# clip = lam_obs/(1+redshift) < 600.
		# tau_clip = 100*(1-1./(1+np.exp(-k*(lam_obs/(1+redshift)-l0))))
		tau_clip = 0.0

		return self._scale_tau * (tau_lc + tau_ls + tau_clip)

	def inoue_igm_abs(self, xtmp, ytmp, zin, cosmo=None, xLL=1215.67, ckms=3e5, 
	# def dijkstra_igm_abs(xtmp, ytmp, zin, cosmo=None, xLL=1215.67, ckms=3e5, 
		R_b1=1.0, delta_v_0=600, alpha_x=1.0, x_HI=None, verbose=False,
		zend=5, zstart=8, log_xHI=False):
		''''''
		# @@@ TBD; need to be careful
		if self._tau is None or np.abs(zin-self._zin) > (1+zin)*0.03 or len(self._tau) != len(xtmp):
			self._zin = zin
			self._tau = self.calculate_general_tau(self._zin, xtmp*(1+self._zin))
		transmission = np.exp(-self._tau)

		# Handle NaNs and values greater than 1
		transmission[transmission != transmission] = 0.0  # squash NaNs
		transmission[transmission > 1] = 1

		# Cut RF <700??
		transmission[xtmp<700] = 0.0  # squash NaNs

		ytmp_abs = ytmp * transmission
		return ytmp_abs, None


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


def masongronke_igm_abs(xtmp, ytmp, zin, cosmo=None, xLL=1216., c=3e18, ckms=3e5, zobs=0, xLLL=1400, x_HI=None,
						zend=5, zstart=8):
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
	T = 1e0 #1e4 # K
	sigma_0 = 5.9e-14 * (T / 1e4)**(-1/2) # cm2
	a_V = 4.7e-4  * (T / 1e4)**(-1/2) # 
	nu_a = 2.46e15 #Hz
	k_B = 1.380649e-23 / (1e3)**2 #m2 kg s-2 K-1 * (km/m)**2 = km2 kg/s2/K
	m_p = 1.67262192e-27 #kilograms
	delta_nu_d = nu_a * np.sqrt(2 * k_B * T / m_p / ckms**2)

	dtdzs = (cosmo.age(zs)[0:-1].value - cosmo.age(zs)[1:].value) * 1e9 * 365.25 * 3600 * 24 / np.diff(zs) # s

	if x_HI == None:
		x_HI = get_XI(zin, zend=zend, zstart=zstart) # neutral fraction
	else:
		if verbose:
			print('Neutral fraction, x_HI = %.10f, is provided;'%(x_HI))

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


def dijkstra_igm_abs(xtmp, ytmp, zin, cosmo=None, xLL=1215.67, ckms=3e5, 
	R_b1=1.0, delta_v_0=600, alpha_x=1.0, x_HI=None, verbose=False,
	zend=5, zstart=8, log_xHI=False, tau_max=200):
	'''
	Purpose
	-------
	Apply IMG-attenuation of Dijikstra (2014). Sec.6.2 Inhomogeneous reionisation & its impact on Lya.
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
	delta_v_0: 
		Lya photons emitted by a galaxy at redshift zg with some velocity off-set delta_v_0

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

	if x_HI == None:
		x_HI = get_XI(zin, zend=zend, zstart=zstart) # neutral fraction
	else:
		if verbose:
			print('Neutral fraction, x_HI = %.10f, is provided;'%(x_HI))

	# This helps speeding up
	x_HI = float("{:.10f}".format(x_HI))

	x_D = alpha_x * x_HI # x_D is not clear..
	delta_lam = (xtmp - xLL) * (zin + 1)
	delta_lam_fine = (np.linspace(900,1500,10000) - xLL) * (zin + 1)

	delta_v = ckms * delta_lam_fine / (xLL * (1.+zin))
	delta_v_b1 = delta_v # 
	if R_b1>0:
		print(R_b1, cosmo.H(zin).value * R_b1 / (1.+zin))
		delta_v_b1 += cosmo.H(zin).value * R_b1 / (1.+zin) # km / (Mpc s) * Mpc

	tau_fine = 2.3 * x_D * (np.abs(delta_v_b1)/delta_v_0)**(-1) * ((1+zin)/10)**(3/2) # Eq.(30)
	con_tau = (tau_fine < 0) | (delta_v_b1 == 0)
	tau_fine[con_tau] = tau_max
	con_tau2 = (delta_v_b1<0)
	tau_fine[con_tau2] = tau_max # By doing this, it assumes the blue side of Lya is completely attenuated
	# import matplotlib.pyplot as plt
	# plt.plot(delta_v_b1, tau_fine)
	# plt.show()
	# print(tau_fine)
	# hoghe

	if False:#True:#
		import matplotlib.pyplot as plt
		plt.close()
		plt.plot(delta_v_b1, tau_fine)
		# plt.xlim(1200,1220)
		# plt.ylim(1e-2,1e1)
		plt.yscale('log')

	fint = interpolate.interp1d(delta_lam_fine, tau_fine, kind='nearest', fill_value="extrapolate")
	tau = fint(delta_lam)
	Transmission = np.exp(-tau)
	ytmp_abs = ytmp * Transmission

	if False:#True:#
		import matplotlib.pyplot as plt
		plt.close()
		# plt.plot(delta_lam_fine, tau_fine)
		# plt.plot(xtmp, tau, ls='--')
		plt.plot(xtmp, Transmission)
		plt.xlim(1200,1220)
		plt.yscale('log')
		plt.show()
		hoge

	if False:#True:#
		import matplotlib.pyplot as plt
		plt.close()
		plt.plot(xtmp, tau)
		plt.xlim(1200,1220)
		# plt.ylim(1e-2,1e1)
		plt.yscale('log')
		plt.show()

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
