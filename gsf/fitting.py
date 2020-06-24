#!/usr/bin/env python
import numpy as np
import sys
import matplotlib.pyplot as plt
from lmfit import Model, Parameters, minimize, fit_report, Minimizer
from numpy import log10
from scipy.integrate import simps
import pickle as cPickle
import os.path
import random
import string
import timeit
from scipy import stats
from scipy.stats import norm
from astropy.io import fits,ascii
import corner

# import from custom codes
from .function import check_line_man, check_line_cz_man, calc_Dn4, savecpkl
from .zfit import check_redshift
from .plot_sed import *
from .writing import get_param

############################
py_v = (sys.version_info[0])
if py_v > 2:
    try:
        raw_input = input
    except NameError:
        pass

################
# Line library
################
LN = ['Mg2', 'Ne5', 'O2', 'Htheta', 'Heta', 'Ne3', 'Hdelta', 'Hgamma', 'Hbeta', 'O3L', 'O3H', 'Mgb', 'Halpha', 'S2L', 'S2H']
LW = [2800, 3347, 3727, 3799, 3836, 3869, 4102, 4341, 4861, 4960, 5008, 5175, 6563, 6717, 6731]
fLW = np.zeros(len(LW), dtype='int')


class Mainbody():

    def __init__(self, inputs, c=3e18, Mpc_cm=3.08568025e+24, m0set=25.0, pixelscale=0.06, Lsun=3.839*1e33, cosmo=None):
        self.update_input(inputs)


    def update_input(self, inputs, c=3e18, Mpc_cm=3.08568025e+24, m0set=25.0, pixelscale=0.06, Lsun=3.839*1e33, cosmo=None):
        '''
        INPUT:
        ==========
        parfile: Ascii file that lists parameters for everything.

        Mpc_cm : cm/Mpc
        pixelscale : arcsec/pixel

        '''

        # Then register;
        self.inputs = inputs
        self.c = c
        self.Mpc_cm = Mpc_cm
        self.m0set  = m0set
        self.pixelscale = pixelscale
        self.Lsun = Lsun #erg s-1

        if cosmo == None:
            from astropy.cosmology import WMAP9 as cosmo
            self.cosmo = cosmo
        else:
            self.cosmo = cosmo

        self.ID   = inputs['ID']
        self.PA   = inputs['PA']
        self.zgal = float(inputs['ZGAL'])
        self.Cz0  = float(inputs['CZ0'])
        self.Cz1  = float(inputs['CZ1'])

        try:
            self.DIR_EXTR = inputs['DIR_EXTR']
        except:
            self.DIR_EXTR = False

        # BPASS Binary template
        try:
            self.f_bpass = int(inputs['BPASS'])
            try:
                self.f_bin = int(inputs['BINARY'])
            except:
                self.f_bin = 1
        except:
            self.f_bpass = 0

        # Nebular emission;
        try:
            self.fneb = int(inputs['ADD_NEBULAE'])
            try:
                self.logU = float(inputs['logU'])
            except:
                self.logU = -2.5
        except:
            self.fneb = 0
            self.logU = 0

        # Data directory;
        self.DIR_TMP  = inputs['DIR_TEMP']
        if not os.path.exists(self.DIR_TMP):
            os.mkdir(self.DIR_TMP)

        # Filter response curve directory, if bb catalog is provided.
        self.DIR_FILT = inputs['DIR_FILT']
        try:
            self.filts    = inputs['FILTER']
            self.filts    = [x.strip() for x in self.filts.split(',')]
        except:
            pass

        self.band = {} #np.zeros((len(self.filts),),'float')
        for ii in range(len(self.filts)):
            fd = np.loadtxt(self.DIR_FILT + self.filts[ii] + '.fil', comments='#')
            self.band['%s_lam'%(self.filts[ii])] = fd[:,1]
            self.band['%s_res'%(self.filts[ii])] = fd[:,2] / np.max(fd[:,2])

        # Filter response curve directory, for RF colors.
        self.filts_rf  = ['u','b','v','j','sz']
        self.band_rf = {} #np.zeros((len(self.filts),),'float')
        for ii in range(len(self.filts_rf)):
            fd = np.loadtxt(self.DIR_FILT + self.filts_rf[ii] + '.fil', comments='#')
            self.band_rf['%s_lam'%(self.filts_rf[ii])] = fd[:,1]
            self.band_rf['%s_res'%(self.filts_rf[ii])] = fd[:,2] / np.max(fd[:,2])


        # Tau comparison?
        try:
            self.ftaucomp = inputs['TAU_COMP']
        except:
            print('No entry: TAU_COMP')
            self.ftaucomp = 0
            print('set to %d' % self.ftaucomp)
            pass

        # Age
        self.age = np.asarray([float(x.strip()) for x in inputs['AGE'].split(',')])
        self.nage = np.arange(0,len(self.age),1)

        # Redshift as a param;
        try:
            self.fzmc = int(inputs['ZMC'])
        except:
            self.fzmc = 0

        # Metallicity
        if self.f_bpass == 0:
            try:
                self.ZFIX = float(inputs['ZFIX'])
                self.delZ = 0.0001
                self.Zmin, self.Zmax = self.ZFIX, self.ZFIX+self.delZ
                self.Zall = np.arange(self.Zmin, self.Zmax, self.delZ) # in logZsun
            except:
                self.Zmax, self.Zmin = float(inputs['ZMAX']), float(inputs['ZMIN'])
                self.delZ = float(inputs['DELZ'])
                if self.Zmax == self.Zmin or self.delZ==0:
                    self.delZ = 0.0001
                    self.Zall = np.arange(self.Zmin, self.Zmax+self.delZ, self.delZ) # in logZsun
                else:
                    self.Zall = np.arange(self.Zmin, self.Zmax, self.delZ) # in logZsun
        else:
            self.Zsun= 0.020
            Zbpass   = [1e-5, 1e-4, 0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.010, 0.020, 0.030, 0.040]
            Zbpass   = np.log10(np.asarray(Zbpass)/self.Zsun)
            try:
                iiz = np.argmin(np.abs(Zbpass[:] - float(inputs['ZFIX']) ) )
                if Zbpass[iiz] - float(inputs['ZFIX']) != 0:
                    print('%.2f is not found in BPASS Z list. %.2f is used instead.'%(float(inputs['ZFIX']),Zbpass[iiz]))
                self.ZFIX = Zbpass[iiz]
                self.delZ = 0.0001
                self.Zmin, self.Zmax = self.ZFIX, self.ZFIX + self.delZ
                self.Zall = np.arange(self.Zmin, self.Zmax, self.delZ) # in logZsun
            except:
                self.Zmax, self.Zmin = float(inputs['ZMAX']), float(inputs['ZMIN'])
                con_z     = np.where((Zbpass >= self.Zmin) & (Zbpass <= self.Zmax))
                self.Zall = Zbpass[con_z]
                self.delZ = 0.0001

        # N of param:
        nAV = 1
        try:
            if int(inputs['ZEVOL']) == 1:
                self.ZEVOL = 1
                self.ndim = int(len(self.nage) * 2 + nAV) # age, Z, and Av.
                print('Metallicity evolution is on.')
                if int(inputs['ZMC']) == 1:
                    self.ndim += 1
                print('No of params are : %d'%(self.ndim))
            else:
                self.ZEVOL = 0
                '''
                if self.delZ < 0.01:
                    self.ndim = int(len(self.nage) + nAV) # age and Av.
                else:
                    self.ndim = int(len(self.nage) + 1 + nAV) # age, Z, and Av.
                '''
                self.ndim = int(len(self.nage) + 1 + nAV) # age, Z, and Av.

                print('Metallicity evolution is off.')
                if int(inputs['ZMC']) == 1:
                    self.ndim += 1
                print('No of params are : %d'%(self.ndim))
        except:
            self.ZEVOL = 0
            pass

        try:
            self.age_fix = [float(x.strip()) for x in inputs['AGEFIX'].split(',')]
        except:
            self.age_fix = []

        # Line
        try:
            self.LW0 = [float(x.strip()) for x in inputs['LINE'].split(',')]
        except:
            self.LW0 = []

        # Dust model specification;
        try:
            self.dust_model = int(inputs['DUST_MODEL'])
        except:
            self.dust_model = 0

        # If FIR data;
        try:
            DT0 = float(inputs['TDUST_LOW'])
            DT1 = float(inputs['TDUST_HIG'])
            dDT = float(inputs['TDUST_DEL'])
            self.Temp= np.arange(DT0,DT1,dDT)
            self.f_dust = True
            print('FIR fit is on.')
        except:
            self.Temp = []
            self.f_dust = False
            pass


        try:
            # Length of each ssp templates.
            self.tau0 = np.asarray([float(x.strip()) for x in inputs['TAU0'].split(',')])
        except:
            self.tau0 = np.asarray([-1.0])

        # IMF
        try:
            self.nimf = int(inputs['NIMF'])
        except:
            self.nimf = 0
            print('Cannot find NIMF. Set to %d.'%(self.nimf))


        '''
        # Read Observed Data
        if self.f_dust:
            self.dict = self.read_data(self.Cz0, self.Cz1, self.zgal, add_fir=True)
        else:
            self.dict = self.read_data(self.Cz0, self.Cz1, self.zgal)
        '''

    def get_lines(self, LW0):
        fLW = np.zeros(len(LW0), dtype='int')
        LW  = LW0
        return LW, fLW


    def read_data(self, Cz0, Cz1, zgal, add_fir=False):
        '''
        Cz0, Cz1 : Normalization coeffs for grism spectra.
        zgal     : Current redshift estimate.

        Note:
        =======
        Can be used for any SFH

        '''
        dict = {}

        ##############
        # Spectrum
        ##############
        dat   = np.loadtxt(self.DIR_TMP + 'spec_obs_' + self.ID + '_PA' + self.PA + '.cat', comments='#')
        NR    = dat[:,0]
        x     = dat[:,1]
        fy00  = dat[:,2]
        ey00  = dat[:,3]

        con0 = (NR<1000)
        xx0  = x[con0]
        fy0  = fy00[con0] * Cz0
        ey0  = ey00[con0] * Cz0
        con1 = (NR>=1000) & (NR<10000)
        xx1  = x[con1]
        fy1  = fy00[con1] * Cz1
        ey1  = ey00[con1] * Cz1

        ##############
        # Broadband
        ##############
        try:
            dat = ascii.read(self.DIR_TMP + 'bb_obs_' + self.ID + '_PA' + self.PA + '.cat', format='no_header')
            NRbb = dat['col1']
            xbb  = dat['col2']
            fybb = dat['col3']
            eybb = dat['col4']
            exbb = dat['col5']
        except: # if no BB;
            print('No BB data.')
            NRbb = np.asarray([])
            xbb  = np.asarray([])
            fybb = np.asarray([])
            eybb = np.asarray([])
            exbb = np.asarray([])

        #con_bb = (eybb>0)
        con_bb = ()

        xx2 = xbb[con_bb]
        ex2 = exbb[con_bb]
        fy2 = fybb[con_bb]
        ey2 = eybb[con_bb]

        xx01 = np.append(xx0,xx1)
        fy01 = np.append(fy0,fy1)
        ey01 = np.append(ey0,ey1)
        xx   = np.append(xx01,xx2)
        fy   = np.append(fy01,fy2)
        ey   = np.append(ey01,ey2)


        wht  = 1./np.square(ey)
        con_wht = (ey<0)
        wht[con_wht] = 0

        # For now...
        #wht2 = check_line_man(fy, x, wht, fy, zgal, self.LW0)
        wht2 = wht[:]

        # Append data;
        if add_fir:
            dat_d = np.loadtxt(self.DIR_TMP + 'spec_dust_obs_' + self.ID + '_PA' + self.PA + '.cat', comments='#')
            x_d   = dat_d[:,1]
            fy_d  = dat_d[:,2]
            ey_d  = dat_d[:,3]

            fy = np.append(fy,fy_d)
            x  = np.append(x,x_d)
            wht= np.append(wht,1./np.square(ey_d))
            # For now...
            #wht2= check_line_man(fy, x, wht, fy, zgal, self.LW0)
            wht2 = wht[:]

        # Into dict

        # Sort data along wave?
        f_sort = False
        if f_sort:
            nrd_yyd = np.zeros((len(NR),6), dtype='float32')
            nrd_yyd[:,0] = NR
            nrd_yyd[:,1] = x
            nrd_yyd[:,2] = fy
            nrd_yyd[:,3] = ey
            nrd_yyd[:,4] = wht
            nrd_yyd[:,5] = wht2

            b = nrd_yyd
            nrd_yyd_sort = b[np.lexsort(([-1,1]*b[:,[1,0]]).T)]
            NR = nrd_yyd_sort[:,0]
            x  = nrd_yyd_sort[:,1]
            fy = nrd_yyd_sort[:,2]
            ey = nrd_yyd_sort[:,3]
            wht = nrd_yyd_sort[:,4]
            wht2= nrd_yyd_sort[:,5]

        sn   = fy/ey
        dict = {'NR':NR, 'x':xx, 'fy':fy, 'ey':ey, 'NRbb':NRbb, 'xbb':xx2, 'exbb':ex2, 'fybb':fy2, 'eybb':ey2, 'wht':wht, 'wht2': wht2, 'sn':sn}

        return dict



    def search_redshift(self, dict, xm_tmp, fm_tmp, zliml=0.01, zlimu=6.0, delzz=0.01, lines=False, prior=None, method='powell'):
        '''
        Purpose:
        =========
        Search redshift space to find the best redshift and probability distribution.


        Input:
        =========

        fm_tmp : a library for various templates. Should be in [ n * len(wavelength)].
        xm_tmp : a wavelength array, common for the templates above, at z=0. Should be in [len(wavelength)].

        prior : Prior for redshift determination. E.g., Eazy z-probability.
        zliml : Lowest redshift for fitting range.
        zlimu : Highest redshift for fitting range.

        method : powell is more accurate. nelder is faster.

        Return:
        =========

        zspace :
        chi2s  :

        '''
        import scipy.interpolate as interpolate

        zspace = np.arange(zliml,zlimu,delzz)
        chi2s  = np.zeros((len(zspace),2), 'float32')
        if prior == None:
            prior = zspace[:] * 0 + 1.0

        # Observed data points;
        NR   = dict['NR']
        con0 = (NR<1000)
        fy0  = dict['fy'][con0] #* Cz0s
        ey0  = dict['ey'][con0] #* Cz0s
        x0   = dict['x'][con0]
        con1 = (NR>=1000) & (NR<10000)
        fy1  = dict['fy'][con1] #* Cz1s
        ey1  = dict['ey'][con1] #* Cz1s
        x1   = dict['x'][con1]
        con2 = (NR>=10000) # BB
        fy2  = dict['fy'][con2]
        ey2  = dict['ey'][con2]
        x2   = dict['x'][con2]

        fy01 = np.append(fy0,fy1)
        fcon = np.append(fy01,fy2)
        ey01 = np.append(ey0,ey1)
        eycon= np.append(ey01,ey2)
        x01  = np.append(x0,x1)
        xobs = np.append(x01,x2)

        wht = 1./np.square(eycon)
        #if lines:
        #    wht2, ypoly = check_line_cz_man(fcon, xobs, wht, fm_s, z)
        #else:
        wht2 = wht

        # Set parameters;
        fit_par_cz = Parameters()
        for nn in range(len(fm_tmp[:,0])):
            fit_par_cz.add('C%d'%nn, value=1., min=0., max=1e5)

        def residual_z(pars,z):
            vals  = pars.valuesdict()

            xm_s = xm_tmp * (1+z)
            fm_s = np.zeros(len(xm_tmp),'float32')

            for nn in range(len(fm_tmp[:,0])):
                fm_s += fm_tmp[nn,:] * pars['C%d'%nn]

            fint = interpolate.interp1d(xm_s, fm_s, kind='cubic', fill_value="extrapolate")
            #fm_int = np.interp(xobs, xm_s, fm_s)
            fm_int = fint(xobs)

            if fcon is None:
                print('Data is none')
                return fm_int
            else:
                return (fm_int - fcon) * np.sqrt(wht2) # i.e. residual/sigma

        # Start redshift search;
        for zz in range(len(zspace)):
            # Best fit
            out_cz = minimize(residual_z, fit_par_cz, args=([zspace[zz]]), method=method)
            keys   = fit_report(out_cz).split('\n')

            csq  = out_cz.chisqr
            rcsq = out_cz.redchi
            fitc_cz = [csq, rcsq]

            #return fitc_cz
            chi2s[zz,0] = csq
            chi2s[zz,1] = rcsq

        self.zspace = zspace
        self.chi2s  = chi2s

        return zspace, chi2s



    def fit_redshift(self, dict, xm_tmp, fm_tmp, delzz=0.01, ezmin=0.01, zliml=0.01, zlimu=6., snlim=0):
        '''
        Purpose:
        ==========
        Find an optimal redshift, before going into a big fit, by using several templates.

        Input:
        ==========
        delzz  : Delta z in redshift search space
        zliml  : Lower limit range for redshift
        zlimu  : Upper limit range for redshift
        ezmin  : Minimum redshift uncertainty.
        snlim  : SN limit for data points. Those below the number will be cut from the fit.

        Note:
        ==========
        Can be used for any SFH.

        '''
        import scipy.interpolate as interpolate

        # For z prior.
        zliml  = self.zgal - 0.5

        # Observed data;
        sn = dict['fy']/dict['ey']
        # Only spec data?
        con_cz = (dict['NR']<10000) & (sn>snlim)
        #con_cz = (dict['NR']<100000) & (sn>snlim)
        fy_cz  = dict['fy'][con_cz] # Already scaled by self.Cz0
        ey_cz  = dict['ey'][con_cz]
        x_cz   = dict['x'][con_cz] # Observed range
        NR_cz  = dict['NR'][con_cz]

        fint = interpolate.interp1d(xm_tmp, fm_tmp, kind='cubic', fill_value="extrapolate")
        #fm_s = np.interp(x_cz, xm_tmp[con_cz], fm_tmp[con_cz])
        #fm_s = np.interp(x_cz, xm_tmp, fm_tmp)
        fm_s = fint(x_cz)

        #
        # If Eazy result exists;
        #
        try:
            eaz_pz = self.inputs['EAZY_PZ']
            f_eazy = True
        except:
            f_eazy = False

        if f_eazy:
            dprob = np.loadtxt(eaz_pz, comments='#')
            zprob = dprob[:,0]
            cprob = dprob[:,1]
            # Then interpolate to a common z grid;
            zz_prob = np.arange(0,13,delzz)
            cprob_s = np.interp(zz_prob, zprob, cprob)
            prior_s = np.exp(-0.5 * cprob_s)
            prior_s /= np.sum(prior_s)
        else:
            zz_prob  = np.arange(0,13,delzz)
            prior_s  = zz_prob * 0 + 1.
            prior_s /= np.sum(prior_s)


        # Plot;
        if self.fzvis==1:
            import matplotlib as mpl
            mpl.use('TkAgg')
            plt.plot(x_cz, fm_s, 'gray', linestyle='--', linewidth=0.5, label='') # Model based on input z.
            plt.plot(x_cz, fy_cz,'b', linestyle='-', linewidth=0.5, label='Obs.') # Observation
            plt.errorbar(x_cz, fy_cz, yerr=ey_cz, color='b', capsize=0, linewidth=0.5) # Observation

        try:
        #if True:
            print('############################')
            print('Start MCMC for redshift fit')
            print('############################')
            res_cz, fitc_cz = check_redshift(fy_cz, ey_cz, x_cz, fm_tmp, xm_tmp/(1+self.zgal), self.zgal, prior_s, NR_cz, zliml, zlimu, delzz, self.nmc_cz, self.nwalk_cz)
            z_cz    = np.percentile(res_cz.flatchain['z'], [16,50,84])
            scl_cz0 = np.percentile(res_cz.flatchain['Cz0'], [16,50,84])
            scl_cz1 = np.percentile(res_cz.flatchain['Cz1'], [16,50,84])

            zrecom  = z_cz[1]
            #if f_scale:
            Czrec0  = scl_cz0[1]
            Czrec1  = scl_cz1[1]

            # Switch to peak redshift:
            # find minimum and maximum of xticks, so we know
            # where we should compute theoretical distribution
            ser = res_cz.flatchain['z']
            xmin, xmax = self.zgal-0.2, self.zgal+0.2
            lnspc = np.linspace(xmin, xmax, len(ser))
            print('\n\n')
            print('Recommended redshift, Cz0 and Cz1, %.5f %.5f %.5f, with chi2/nu=%.3f'%(zrecom, Czrec0, Czrec1, fitc_cz[1]))
            print('\n\n')
            fit_label = 'Proposed model'

        except:
        #else:
            print('!!! z fit failed. No spectral data set?')
            try:
                ezl = float(self.inputs['EZL'])
                ezu = float(self.inputs['EZU'])
                print('Redshift error is taken from input file.')
                if ezl<ezmin:
                    ezl = ezmin #0.03
                if ezu<ezmin:
                    ezu = ezmin #0.03
            except:
                ezl = ezmin
                ezu = ezmin
                print('Redshift error is assumed to %.1f.'%(ezl))

            z_cz    = [self.zprev-ezl, self.zprev, self.zprev+ezu]
            zrecom  = z_cz[1]
            scl_cz0 = [1.,1.,1.]
            scl_cz1 = [1.,1.,1.]
            Czrec0  = scl_cz0[1]
            Czrec1  = scl_cz1[1]
            res_cz  = None

            # If this label is being used, it means that the fit is failed.
            fit_label = 'Current model'

        # New template at zrecom;
        xm_s = xm_tmp / (1+self.zgal) * (1+zrecom)
        #fm_s = np.interp(x_cz, xm_s[con_cz], fm_tmp[con_cz])
        #fm_s = np.interp(x_cz, xm_s, fm_tmp)
        fint = interpolate.interp1d(xm_s, fm_tmp, kind='cubic', fill_value="extrapolate")
        fm_s = fint(x_cz)
        whtl = 1/np.square(ey_cz)

        try:
            wht3, ypoly = check_line_cz_man(fy_cz, x_cz, whtl, fm_s, zrecom, LW=self.LW0)
        except:
            wht3, ypoly = whtl, fy_cz
        con_line = (wht3==0)

        # Visual inspection;
        if self.fzvis==1:
            #
            # Ask interactively;
            #
            plt.plot(x_cz, fm_s, 'r', linestyle='-', linewidth=0.5, label='%s ($z=%.5f$)'%(fit_label,zrecom)) # Model based on recomended z.
            plt.plot(x_cz[con_line], fm_s[con_line], color='orange', marker='o', linestyle='', linewidth=3.)

            # Plot lines for reference
            for ll in range(len(LW)):
                try:
                    conpoly = (x_cz/(1.+zrecom)>3000) & (x_cz/(1.+zrecom)<8000)
                    yline = np.max(ypoly[conpoly])
                    yy    = np.arange(yline/1.02, yline*1.1)
                    xxpre = yy * 0 + LW[ll] * (1.+self.zgal)
                    xx    = yy * 0 + LW[ll] * (1.+zrecom)
                    plt.plot(xxpre, yy/1.02, linewidth=0.5, linestyle='--', color='gray')
                    plt.text(LW[ll] * (1.+self.zgal), yline/1.05, '%s'%(LN[ll]), fontsize=8, color='gray')
                    plt.plot(xx, yy, linewidth=0.5, linestyle='-', color='orangered')
                    plt.text(LW[ll] * (1.+zrecom), yline, '%s'%(LN[ll]), fontsize=8, color='orangered')
                except:
                    pass

            plt.plot(dict['xbb'], dict['fybb'], marker='.', color='r', ms=10, linestyle='', linewidth=0, zorder=4, label='Obs.(BB)')
            plt.scatter(xm_tmp, fm_tmp, color='none', marker='d', s=50, edgecolor='gray', zorder=4, label='Current model ($z=%.5f$)'%(self.zgal))

            try:
                xmin, xmax = np.min(x_cz)/1.1,np.max(x_cz)*1.1
            except:
                xmin, xmax = 2000,10000

            try:
                plt.ylim(0,yline*1.1)
            except:
                pass

            plt.xlim(xmin,xmax)
            plt.xlabel('Wavelength ($\mathrm{\AA}$)')
            plt.ylabel('$F_\\nu$ (arb.)')
            plt.legend(loc=0)

            zzsigma  = ((z_cz[2] - z_cz[0])/2.)/self.zgal
            zsigma   = np.abs(self.zgal-zrecom) / (self.zgal)
            C0sigma  = np.abs(Czrec0-self.Cz0)/self.Cz0
            eC0sigma = ((scl_cz0[2]-scl_cz0[0])/2.)/self.Cz0
            C1sigma  = np.abs(Czrec1-self.Cz1)/self.Cz1
            eC1sigma = ((scl_cz1[2]-scl_cz1[0])/2.)/self.Cz1

            print('\n##############################################################')
            print('Input redshift is %.3f per cent agreement.'%((1.-zsigma)*100))
            print('Error is %.3f per cent.'%(zzsigma*100))
            print('Input Cz0 is %.3f per cent agreement.'%((1.-C0sigma)*100))
            print('Error is %.3f per cent.'%(eC0sigma*100))
            print('Input Cz1 is %.3f per cent agreement.'%((1.-C1sigma)*100))
            print('Error is %.3f per cent.'%(eC1sigma*100))
            print('##############################################################\n')
            plt.show()

            flag_z = raw_input('Do you want to continue with the input redshift, Cz0 and Cz1, %.5f %.5f %.5f? ([y]/n/m) '%(self.zgal, self.Cz0, self.Cz1))
        else:
            flag_z = 'y'

        # Write it to self;
        self.zrecom = zrecom
        self.Czrec0 = Czrec0 * self.Cz0
        self.Czrec1 = Czrec1 * self.Cz1
        self.z_cz   = z_cz
        self.scl_cz0= scl_cz0
        self.scl_cz1= scl_cz1
        self.res_cz = res_cz

        return flag_z


    def get_zdist(self):
        '''
        # Save fig of z-distribution.

        Note:
        =======
        Can be used for any SFH

        '''

        try: # if spectrum;
        #if True:
            fig = plt.figure(figsize=(6.5,2.5))
            fig.subplots_adjust(top=0.96, bottom=0.16, left=0.09, right=0.99, hspace=0.15, wspace=0.25)
            ax1 = fig.add_subplot(111)
            n, nbins, patches = ax1.hist(self.res_cz.flatchain['z'], bins=200, normed=True, color='gray', label='')

            yy = np.arange(0,np.max(n),1)
            xx = yy * 0 + self.z_cz[1]
            ax1.plot(xx,yy,linestyle='-',linewidth=1,color='orangered',label='$z=%.5f_{-%.5f}^{+%.5f}$\n$C_z0=%.3f$\n$C_z1=%.3f$'%(self.z_cz[1],self.z_cz[1]-self.z_cz[0],self.z_cz[2]-self.z_cz[1], self.Cz0, self.Cz1))
            xx = yy * 0 + self.z_cz[0]
            ax1.plot(xx,yy,linestyle='--',linewidth=1,color='orangered')
            xx = yy * 0 + self.z_cz[2]
            ax1.plot(xx,yy,linestyle='--',linewidth=1,color='orangered')
            xx = yy * 0 + self.zgal
            ax1.plot(xx,yy,linestyle='-',linewidth=1,color='royalblue')
            ax1.set_xlabel('Redshift')
            ax1.set_ylabel('$dn/dz$')
            ax1.legend(loc=0)
            plt.savefig('zprob_' + self.ID + '_PA' + seld.PA + '.pdf', dpi=300)
            plt.close()
        #else:
        except:
            print('z-distribution figure is not generated.')
            pass


    def add_param(self, fit_params, sigz=1.0):
        '''

        Note:
        =======
        Can be used for any SFH
        '''

        f_add = False

        # Redshift
        if self.fzmc == 1:
            fit_params.add('zmc', value=self.zgal, min=self.zgal-(self.z_cz[1]-self.z_cz[0])*sigz, max=self.zgal+(self.z_cz[2]-self.z_cz[1])*sigz)
            #self.ndim += 1 # Already added.
            f_add = True

        # Error parameter
        try:
            ferr = self.ferr
            if ferr == 1:
                fit_params.add('f', value=1e-2, min=0, max=1e2)
                self.ndim += 1
                f_add = True
        except:
            ferr = 0
            pass

        # Dust;
        if self.f_dust:
            Tdust = self.Temp
            fit_params.add('TDUST', value=len(Tdust)/2., min=0, max=len(Tdust)-1)
            fit_params.add('MDUST', value=1e6, min=0, max=1e10)
            self.ndim += 2

            dict = read_data(self.Cz0, self.Cz1, self.zgal, add_fir=True)

            f_add = True

        return f_add


    #def main(self, zgal, flag_m, Cz0, Cz1, cornerplot=True, specplot=1, sigz=1.0, ezmin=0.01, ferr=0, f_move=False):
    def main(self, flag_m, cornerplot=True, specplot=1, sigz=1.0, ezmin=0.01, ferr=0, f_move=False):
        '''
        Input:
        ========
        flag_m : related to redshift error in redshift check func.
        ferr   : For error parameter
        zgal   : Input redshift.
        #
        #
        # sigz (float): confidence interval for redshift fit.
        # ezmin (float): minimum error in redshift.
        #
        '''

        print('########################')
        print('### Fitting Function ###')
        print('########################')
        start = timeit.default_timer()

        ID0 = self.ID
        PA0 = self.PA

        inputs = self.inputs
        if not os.path.exists(self.DIR_TMP):
            os.mkdir(self.DIR_TMP)

        # And class;
        #from .function_class import Func
        #from .basic_func import Basic
        #self.fnc  = Func(self.ID, self.PA, self.Zall, self.nage, dust_model=self.dust_model, DIR_TMP=self.DIR_TMP) # Set up the number of Age/ZZ
        #self.bfnc = Basic(self.Zall)

        # Load Spectral library;
        self.lib = self.fnc.open_spec_fits(self, fall=0)
        self.lib_all = self.fnc.open_spec_fits(self, fall=1)
        if self.f_dust:
            self.lib_dust     = self.fnc.open_spec_dust_fits(self, fall=0)
            self.lib_dust_all = self.fnc.open_spec_dust_fits(self, fall=1)

        # For MCMC;
        self.nmc      = int(self.inputs['NMC'])
        self.nwalk    = int(self.inputs['NWALK'])
        self.nmc_cz   = int(self.inputs['NMCZ'])
        self.nwalk_cz = int(self.inputs['NWALKZ'])
        self.Zevol    = int(self.inputs['ZEVOL'])
        self.fzvis    = int(self.inputs['ZVIS'])
        self.fneld    = int(self.inputs['FNELD'])

        try:
            self.ntemp = int(self.inputs['NTEMP'])
        except:
            self.ntemp = 1

        try:
            if int(inputs['DISP']) == 1:
                self.f_disp = True
            else:
                self.f_disp = False
        except:
            self.f_disp = False

        #
        # Dust model specification;
        #
        try:
            dust_model = int(inputs['DUST_MODEL'])
        except:
            dust_model = 0

        fnc  = self.fnc  #Func(Zall, nage, dust_model=dust_model, self.DIR_TMP=self.DIR_TMP) # Set up the number of Age/ZZ
        bfnc = self.bfnc #Basic(Zall)

        # Error parameter
        try:
            self.ferr = int(inputs['F_ERR'])
        except:
            self.ferr = 0
            pass

        #################
        # Observed Data
        #################
        print('READ data with',self.Cz0, self.Cz1, self.zgal)
        dict = self.read_data(self.Cz0, self.Cz1, self.zgal)
        self.dict = dict

        # Call likelihood/prior/posterior function;
        from .posterior_flexible import Post
        class_post = Post(self)

        ###############################
        # Add parameters
        ###############################
        agemax = self.cosmo.age(self.zgal).value #, use_flat=True, **cosmo)/cc.Gyr_s
        fit_params = Parameters()
        try:
            age_fix = inputs['AGEFIX']
            age_fix = [float(x.strip()) for x in age_fix.split(',')]
            aamin = []
            print('\n')
            print('##########################')
            print('AGEFIX is found.\nAge will be fixed to:')
            for age_tmp in age_fix:
                ageind = np.argmin(np.abs(age_tmp-np.asarray(self.age[:])))
                aamin.append(ageind)
                print('%6s Gyr'%(self.age[ageind]))
            print('##########################')
            for aa in range(len(self.age)):
                if aa not in aamin:
                    #print('Not added.')
                    fit_params.add('A'+str(aa), value=0, min=0, max=1e-10)
                else:
                    fit_params.add('A'+str(aa), value=1, min=0, max=1e3)
        except:
            for aa in range(len(self.age)):
                if self.age[aa] == 99:
                    fit_params.add('A'+str(aa), value=0, min=0, max=1e-10)
                elif self.age[aa]>agemax:
                    print('At this redshift, A%d is beyond the age of universe and not used.'%(aa))
                    fit_params.add('A'+str(aa), value=0, min=0, max=1e-10)
                else:
                    fit_params.add('A'+str(aa), value=1, min=0, max=1e3)

        #####################
        # Dust attenuation
        #####################
        try:
            Avmin = float(inputs['AVMIN'])
            Avmax = float(inputs['AVMAX'])
            if Avmin == Avmax:
                Avmax += 0.001
            fit_params.add('Av', value=(Avmax+Avmin)/2., min=Avmin, max=Avmax)
        except:
            Avmin = 0.0
            Avmax = 4.0
            Avini = 0.5
            print('Dust is set in [%.1f:%.1f]/mag. Initial value is set to %.1f'%(Avmin,Avmax,Avini))
            fit_params.add('Av', value=Avini, min=Avmin, max=Avmax)

        #####################
        # Metallicity
        #####################
        if int(inputs['ZEVOL']) == 1:
            for aa in range(len(self.age)):
                if self.age[aa] == 99 or self.age[aa]>agemax:
                    fit_params.add('Z'+str(aa), value=0, min=0, max=1e-10)
                else:
                    fit_params.add('Z'+str(aa), value=0, min=np.min(self.Zall), max=np.max(self.Zall))
        else:
            try:
                ZFIX = float(inputs['ZFIX'])
                aa = 0
                fit_params.add('Z'+str(aa), value=0, min=ZFIX, max=ZFIX+0.0001)
            except:
                aa = 0
                if np.min(self.Zall)==np.max(self.Zall):
                    fit_params.add('Z'+str(aa), value=0, min=np.min(self.Zall), max=np.max(self.Zall)+0.0001)
                else:
                    fit_params.add('Z'+str(aa), value=0, min=np.min(self.Zall), max=np.max(self.Zall))

        ####################################
        # Initial Metallicity Determination
        ####################################
        # Get initial parameters
        out,chidef,Zbest = get_leastsq(inputs,self.Zall,self.fneld,self.age,fit_params,class_post.residual,dict['fy'],dict['wht2'],self.ID,self.PA)

        # Best fit
        csq  = out.chisqr
        rcsq = out.redchi
        fitc = [csq, rcsq] # Chi2, Reduced-chi2
        ZZ   = Zbest # This is really important/does affect lnprob/residual.

        print('\n\n')
        print('#####################################')
        print('Zbest, chi are;',Zbest,chidef)
        print('Params are;',fit_report(out))
        print('#####################################')
        print('\n\n')

        Av_tmp = out.params['Av'].value
        AA_tmp = np.zeros(len(self.age), dtype='float64')
        ZZ_tmp = np.zeros(len(self.age), dtype='float64')
        fm_tmp, xm_tmp = fnc.tmp04_val(out, self.zgal, self.lib)
        #print(self.lib[:,0])

        ########################
        # Check redshift
        ########################
        flag_z = self.fit_redshift(dict, xm_tmp, fm_tmp)

        #################################################
        # Gor for mcmc phase
        #################################################
        if flag_z == 'y' or flag_z == '':
            #zrecom  = self.zprev
            #Czrec0  = self.Cz0
            #Czrec1  = self.Cz1

            # plot z-distribution
            self.get_zdist()

            #######################
            # Add parameters;
            #######################
            out_keep = out #.copy()
            f_add    = self.add_param(fit_params)

            # Then, minimize again.
            if f_add:
                if self.fneld == 1:
                    fit_name = 'nelder'
                else:
                    fit_name = 'powell'
                out = minimize(class_post.residual, fit_params, args=(dict['fy'], dict['wht2'], self.f_dust), method=fit_name) # It needs to define out with redshift constrain.
                print(fit_report(out))

                # Fix params to what we had before.
                out.params['zmc'].value = self.zgal
                out.params['Av'].value  = out_keep.params['Av'].value
                for aa in range(len(self.age)):
                    out.params['A'+str(aa)].value = out_keep.params['A'+str(aa)].value
                    try:
                        out.params['Z'+str(aa)].value = out_keep.params['Z'+str(aa)].value
                    except:
                        out.params['Z0'].value = out_keep.params['Z0'].value


            ##############################
            print('\n\n')
            print('###############################')
            print('Input redshift is adopted.')
            print('Starting long journey in MCMC.')
            print('###############################')
            print('\n\n')

            ################################
            print('\nMinimizer Defined\n')
            mini = Minimizer(class_post.lnprob, out.params, fcn_args=[dict['fy'],dict['wht'],self.f_dust], f_disp=self.f_disp, f_move=f_move)
            print('######################')
            print('### Starting emcee ###')
            print('######################')

            try:
                import multiprocess
            except:
                import multiprocessing as multiprocess

            ncpu0 = int(multiprocess.cpu_count()/2)
            try:
                ncpu = int(inputs['NCPU'])
                if ncpu > ncpu0:
                    print('!!! NCPU is larger than No. of CPU. !!!')
            except:
                ncpu = ncpu0
                pass

            print('No. of CPU is set to %d'%(ncpu))
            start_mc = timeit.default_timer()
            res  = mini.emcee(burn=int(self.nmc/2), steps=self.nmc, thin=10, nwalkers=self.nwalk, params=out.params, is_weighted=True, ntemps=self.ntemp, workers=ncpu)
            stop_mc  = timeit.default_timer()
            tcalc_mc = stop_mc - start_mc
            print('###############################')
            print('### MCMC part took %.1f sec ###'%(tcalc_mc))
            print('###############################')

            #----------- Save pckl file
            #-------- store chain into a cpkl file
            start_mc = timeit.default_timer()
            burnin   = int(self.nmc/2)
            savepath = './'
            cpklname = 'chain_' + self.ID + '_PA' + self.PA + '_corner.cpkl'
            savecpkl({'chain':res.flatchain,
                          'burnin':burnin, 'nwalkers':self.nwalk,'niter':self.nmc,'ndim':self.ndim},
                         savepath+cpklname) # Already burn in
            stop_mc  = timeit.default_timer()
            tcalc_mc = stop_mc - start_mc
            print('#################################')
            print('### Saving chain took %.1f sec'%(tcalc_mc))
            print('#################################')

            Avmc = np.percentile(res.flatchain['Av'], [16,50,84])
            Avpar = np.zeros((1,3), dtype='float64')
            Avpar[0,:] = Avmc

            ####################
            # Best parameters
            ####################
            out = res
            Amc  = np.zeros((len(self.age),3), dtype='float64')
            Ab   = np.zeros(len(self.age), dtype='float64')
            Zmc  = np.zeros((len(self.age),3), dtype='float64')
            Zb   = np.zeros(len(self.age), dtype='float64')
            NZbest = np.zeros(len(self.age), dtype='int')
            f0     = fits.open(self.DIR_TMP + 'ms_' + self.ID + '_PA' + self.PA + '.fits')
            sedpar = f0[1]
            ms     = np.zeros(len(self.age), dtype='float64')
            for aa in range(len(self.age)):
                Ab[aa]    = out.params['A'+str(aa)].value
                Amc[aa,:] = np.percentile(res.flatchain['A'+str(aa)], [16,50,84])
                try:
                    Zb[aa]    = out.params['Z'+str(aa)].value
                    Zmc[aa,:] = np.percentile(res.flatchain['Z'+str(aa)], [16,50,84])
                except:
                    Zb[aa]    = out.params['Z0'].value
                    Zmc[aa,:] = np.percentile(res.flatchain['Z0'], [16,50,84])
                NZbest[aa]= bfnc.Z2NZ(Zb[aa])
                ms[aa]    = sedpar.data['ML_' +  str(NZbest[aa])][aa]

            Avb = out.params['Av'].value

            if self.f_dust:
                Mdust_mc = np.zeros(3, dtype='float64')
                Tdust_mc = np.zeros(3, dtype='float64')
                Mdust_mc[:] = np.percentile(res.flatchain['MDUST'], [16,50,84])
                Tdust_mc[:] = np.percentile(res.flatchain['TDUST'], [16,50,84])
                print(Mdust_mc)
                print(Tdust_mc)

            ####################
            # MCMC corner plot.
            ####################
            if cornerplot:
                fig1 = corner.corner(res.flatchain, labels=res.var_names, \
                label_kwargs={'fontsize':16}, quantiles=[0.16, 0.84], show_titles=False, \
                title_kwargs={"fontsize": 14}, truths=list(res.params.valuesdict().values()), \
                plot_datapoints=False, plot_contours=True, no_fill_contours=True, \
                plot_density=False, levels=[0.68, 0.95, 0.997], truth_color='gray', color='#4682b4')
                fig1.savefig('SPEC_' + self.ID + '_PA' + self.PA + '_corner.pdf')

            # Do analysis on MCMC results.
            # Write to file.
            stop  = timeit.default_timer()
            tcalc = stop - start

            # Then writing;
            start_mc = timeit.default_timer()
            get_param(self, res, fitc, tcalc=tcalc)
            stop_mc  = timeit.default_timer()
            tcalc_mc = stop_mc - start_mc
            print('##############################################')
            print('### Writing params tp file took %.1f sec ###'%(tcalc_mc))
            print('##############################################')

            return False #, self.zgal, self.Cz0, self.Cz1


        elif flag_z == 'm':
            zrecom = raw_input('What is your manual input for redshift? [%.3f] '%(self.zgal))
            if zrecom != '':
                zrecom = float(zrecom)
            else:
                zrecom = self.zgal

            Czrec0 = raw_input('What is your manual input for Cz0? [%.3f] '%(self.Cz0))
            if Czrec0 != '':
                Czrec0 = float(Czrec0)
            else:
                Czrec0 = self.Cz0

            Czrec1 = raw_input('What is your manual input for Cz1? [%.3f] '%(self.Cz1))
            if Czrec1 != '':
                Czrec1 = float(Czrec1)
            else:
                Czrec1 = self.Cz1

            self.zprev = self.zgal   # Input redshift for previous run
            self.zgal  = zrecom # Recommended redshift from previous run
            self.Cz0   = Czrec0
            self.Cz1   = Czrec1
            print('\n\n')
            print('Generate model templates with input redshift and Scale.')
            print('\n\n')
            return True #, self.zgal, self.Cz0, self.Cz1

        else:
            print('\n\n')
            print('Terminated because of redshift estimate.')
            print('Generate model templates with recommended redshift.')
            print('\n\n')

            flag_gen = raw_input('Do you want to make templates with recommended redshift, Cz0, and Cz1 , %.5f %.5f %.5f? ([y]/n) '%(self.zrecom, self.Czrec0, self.Czrec1))
            if flag_gen == 'y' or flag_gen == '':

                self.zprev = self.zgal   # Input redshift for previous run
                self.zgal  = self.zrecom # Recommended redshift from previous run
                self.Cz0   = self.Czrec0
                self.Cz1   = self.Czrec1

                return True #, self.zgal, self.Cz0, self.Cz1

            else:
                print('\n\n')
                print('There is nothing to do.')
                print('Terminating process.')
                print('\n\n')
                return -1 #, self.zgal, self.Czrec0, self.Czrec1



    def quick_fit(self, zgal, flag_m, Cz0, Cz1, specplot=1, sigz=1.0, ezmin=0.01, ferr=0, f_move=False):
        '''
        Purpose:
        ==========
        Fit input data with a prepared template library, to get a chi-min result.

        Input:
        ==========
        flag_m : related to redshift error in redshift check func.
        ferr   : For error parameter
        zgal   : Input redshift.
        #
        #
        # sigz (float): confidence interval for redshift fit.
        # ezmin (float): minimum error in redshift.
        #
        '''
        from .posterior_flexible import Post

        print('########################')
        print('### Fitting Function ###')
        print('########################')
        start = timeit.default_timer()

        ID0 = self.ID
        PA0 = self.PA

        inputs = self.inputs
        if not os.path.exists(self.DIR_TMP):
            os.mkdir(self.DIR_TMP)

        # Load Spectral library;
        self.lib = self.fnc.open_spec_fits(self, fall=0)
        self.lib_all = self.fnc.open_spec_fits(self, fall=1)
        if self.f_dust:
            self.lib_dust     = self.fnc.open_spec_dust_fits(self, fall=0)
            self.lib_dust_all = self.fnc.open_spec_dust_fits(self, fall=1)

        # For MCMC;
        self.nmc      = int(self.inputs['NMC'])
        self.nwalk    = int(self.inputs['NWALK'])
        self.nmc_cz   = int(self.inputs['NMCZ'])
        self.nwalk_cz = int(self.inputs['NWALKZ'])
        self.Zevol    = int(self.inputs['ZEVOL'])
        self.fzvis    = int(self.inputs['ZVIS'])
        self.fneld    = int(self.inputs['FNELD'])

        try:
            self.ntemp = int(self.inputs['NTEMP'])
        except:
            self.ntemp = 1

        try:
            if int(inputs['DISP']) == 1:
                self.f_disp = True
            else:
                self.f_disp = False
        except:
            self.f_disp = False

        #
        # Dust model specification;
        #
        try:
            dust_model = int(inputs['DUST_MODEL'])
        except:
            dust_model = 0

        fnc  = self.fnc  #Func(Zall, nage, dust_model=dust_model, self.DIR_TMP=self.DIR_TMP) # Set up the number of Age/ZZ
        bfnc = self.bfnc #Basic(Zall)

        # Error parameter
        try:
            self.ferr = int(inputs['F_ERR'])
        except:
            self.ferr = 0
            pass

        #################
        # Observed Data
        #################
        dict = self.read_data(self.Cz0, self.Cz1, self.zgal)

        # Call likelihood/prior/posterior function;
        class_post = Post(self)

        ###############################
        # Add parameters
        ###############################
        agemax = self.cosmo.age(zgal).value #, use_flat=True, **cosmo)/cc.Gyr_s
        fit_params = Parameters()
        try:
            age_fix = inputs['AGEFIX']
            age_fix = [float(x.strip()) for x in age_fix.split(',')]
            aamin = []
            print('\n')
            print('##########################')
            print('AGEFIX is found.\nAge will be fixed to:')
            for age_tmp in age_fix:
                ageind = np.argmin(np.abs(age_tmp-np.asarray(self.age[:])))
                aamin.append(ageind)
                print('%6s Gyr'%(self.age[ageind]))
            print('##########################')
            for aa in range(len(self.age)):
                if aa not in aamin:
                    fit_params.add('A'+str(aa), value=0, min=0, max=1e-10)
                else:
                    fit_params.add('A'+str(aa), value=1, min=0, max=1e3)
        except:
            for aa in range(len(self.age)):
                if self.age[aa] == 99:
                    fit_params.add('A'+str(aa), value=0, min=0, max=1e-10)
                elif self.age[aa]>agemax:
                    print('At this redshift, A%d is beyond the age of universe and not used.'%(aa))
                    fit_params.add('A'+str(aa), value=0, min=0, max=1e-10)
                else:
                    fit_params.add('A'+str(aa), value=1, min=0, max=1e3)

        #####################
        # Dust attenuation
        #####################
        try:
            Avmin = float(inputs['AVMIN'])
            Avmax = float(inputs['AVMAX'])
            if Avmin == Avmax:
                Avmax += 0.001
            fit_params.add('Av', value=(Avmax+Avmin)/2., min=Avmin, max=Avmax)
        except:
            Avmin = 0.0
            Avmax = 4.0
            Avini = 0.5
            print('Dust is set in [%.1f:%.1f]/mag. Initial value is set to %.1f'%(Avmin,Avmax,Avini))
            fit_params.add('Av', value=Avini, min=Avmin, max=Avmax)

        #####################
        # Metallicity
        #####################
        if int(inputs['ZEVOL']) == 1:
            for aa in range(len(self.age)):
                if self.age[aa] == 99 or self.age[aa]>agemax:
                    fit_params.add('Z'+str(aa), value=0, min=0, max=1e-10)
                else:
                    fit_params.add('Z'+str(aa), value=0, min=np.min(self.Zall), max=np.max(self.Zall))
        else:
            try:
                ZFIX = float(inputs['ZFIX'])
                aa = 0
                fit_params.add('Z'+str(aa), value=0, min=ZFIX, max=ZFIX+0.0001)
            except:
                aa = 0
                if np.min(self.Zall)==np.max(self.Zall):
                    fit_params.add('Z'+str(aa), value=0, min=np.min(self.Zall), max=np.max(self.Zall)+0.0001)
                else:
                    fit_params.add('Z'+str(aa), value=0, min=np.min(self.Zall), max=np.max(self.Zall))

        ####################################
        # Initial Metallicity Determination
        ####################################
        # Get initial parameters
        print('Start quick fit;')
        out,chidef,Zbest = get_leastsq(inputs,self.Zall,self.fneld,self.age,fit_params,class_post.residual,dict['fy'],dict['wht2'],self.ID,self.PA)

        # Best fit
        csq  = out.chisqr
        rcsq = out.redchi
        fitc = [csq, rcsq] # Chi2, Reduced-chi2
        ZZ   = Zbest # This is really important/does affect lnprob/residual.

        print('\n\n')
        print('#####################################')
        print('Zbest, chi are;',Zbest,chidef)
        print('Params are;',fit_report(out))
        print('#####################################')
        print('\n\n')

        Av_tmp = out.params['Av'].value
        AA_tmp = np.zeros(len(self.age), dtype='float64')
        ZZ_tmp = np.zeros(len(self.age), dtype='float64')
        fm_tmp, xm_tmp = fnc.tmp04_val(out, self.zgal, self.lib)

        ########################
        # Check redshift
        ########################
        if self.fzvis:
            import matplotlib as mpl
            mpl.use('TkAgg')
            flag_z = self.fit_redshift(dict, xm_tmp, fm_tmp)

        return out, fm_tmp, xm_tmp
