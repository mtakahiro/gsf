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
from astropy.io import fits
import string
import timeit

# import from custom codes
from .function import check_line_man, check_line_cz_man, filconv, calc_Dn4, savecpkl
from .zfit import check_redshift
from .plot_sed import *

############################
py_v = (sys.version_info[0])
if py_v > 2:
    try:
        raw_input = input
    except NameError:
        pass

#################
# Physical params
#################
c    = 3.e18 # A/s
mag0 = 25.0 # Magnitude zeropoint set.
d    = 10**(73.6/2.5) # From [ergs/s/cm2/A] to [ergs/s/cm2/Hz]


################
# Line library
################
LN = ['Mg2', 'Ne5', 'O2', 'Htheta', 'Heta', 'Ne3', 'Hdelta', 'Hgamma', 'Hbeta', 'O3L', 'O3H', 'Mgb', 'Halpha', 'S2L', 'S2H']
LW = [2800, 3347, 3727, 3799, 3836, 3869, 4102, 4341, 4861, 4960, 5008, 5175, 6563, 6717, 6731]
fLW = np.zeros(len(LW), dtype='int') # flag.


class Mainbody():

    def __init__(self, inputs, c=3e18, Mpc_cm=3.08568025e+24, m0set=25.0, pixelscale=0.06, Lsun=3.839*1e33, cosmo=None):
        '''
        INPUT:
        ==========
        inputs: Dictionary for input params.

         Mpc_cm : cm/Mpc
         pixelscale : arcsec/pixel

        '''

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
        self.DIR_FILT = inputs['DIR_FILT']

        # Tau comparison?
        try:
            self.ftaucomp = inputs['TAU_COMP']
        except:
            print('No entry: TAU_COMP')
            self.ftaucomp = 0
            print('set to %d' % self.ftaucomp)
            pass

        # Age
        self.age = [float(x.strip()) for x in inputs['AGE'].split(',')]
        self.nage = np.arange(0,len(self.age),1)

        # Redshift as a param;
        try:
            self.fzmc = int(inputs['ZMC'])
        except:
            self.fzmc = 0


        # Metallicity
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

        # N of param:
        if int(inputs['ZEVOL']) == 1:
            self.ndim = int(len(self.nage) * 2 + 1)
            print('Metallicity evolution is on.')
            if int(inputs['ZMC']) == 1:
                self.ndim += 1
            print('No of params are : %d'%(self.ndim))
        else:
            self.ndim = int(len(self.nage) + 1 + 1)
            print('Metallicity evolution is off.')
            if int(inputs['ZMC']) == 1:
                self.ndim += 1
            print('No of params are : %d'%(self.ndim))
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
            self.f_dust = False
            pass


        # Tau for MCMC parameter; not as fitting parameters.
        self.tau0 = [float(x.strip()) for x in inputs['TAU0'].split(',')]

        # IMF
        try:
            self.nimf = int(inputs['NIMF'])
        except:
            self.nimf = 0
            print('Cannot find NIMF. Set to %d.'%(self.nimf))

        # MCMC;
        self.nmc      = int(inputs['NMC'])
        self.nwalk    = int(inputs['NWALK'])
        self.nmc_cz   = int(inputs['NMCZ'])
        self.nwalk_cz = int(inputs['NWALKZ'])
        self.Zevol  = int(inputs['ZEVOL'])
        self.fzvis    = int(inputs['ZVIS'])
        self.fneld    = int(inputs['FNELD'])
        try:
            self.ntemp = int(inputs['NTEMP'])
        except:
            self.ntemp = 1

        try:
            if int(inputs['DISP']) == 1:
                self.f_disp = True
            else:
                self.f_disp = False
        except:
            self.f_disp = False

        # And class;
        from .function_class import Func
        from .basic_func import Basic
        self.fnc  = Func(self.ID, self.PA, self.Zall, self.nage, dust_model=self.dust_model, DIR_TMP=self.DIR_TMP) # Set up the number of Age/ZZ
        self.bfnc = Basic(self.Zall)

        # Spectral library;
        self.lib = self.fnc.open_spec_fits(self.ID, self.PA, fall=0, tau0=self.tau0)
        self.lib_all = self.fnc.open_spec_fits(self.ID, self.PA, fall=1, tau0=self.tau0)
        if self.f_dust:
            self.lib_dust     = self.fnc.open_spec_dust_fits(self.ID, self.PA, self.Temp, fall=0, tau0=self.tau0)
            self.lib_dust_all = self.fnc.open_spec_dust_fits(self.ID, self.PA, self.Temp, fall=1, tau0=self.tau0)

    def get_lines(self, LW0):
        fLW = np.zeros(len(LW0), dtype='int')
        LW  = LW0
        return LW, fLW

    def main(self, ID0, PA0, zgal, flag_m, zprev, Cz0, Cz1, mcmcplot=True, fzvis=1, specplot=1, fneld=0, ntemp=5, sigz=1.0, ezmin=0.01, f_move=False, f_disp=False): # flag_m related to redshift error in redshift check func.
        #
        # sigz (float): confidence interval for redshift fit.
        # ezmin (float): minimum error in redshift.
        #
        print('########################')
        print('### Fitting Function ###')
        print('########################')
        start = timeit.default_timer()

        inputs = self.inputs
        DIR_TMP  = inputs['DIR_TEMP']

        if os.path.exists(DIR_TMP) == False:
            os.mkdir(DIR_TMP)

        # For error parameter
        ferr = 0

        # This needs to be defined here.
        self.zprev = zprev

        #
        # Age
        #
        age = inputs['AGE']
        age = [float(x.strip()) for x in age.split(',')]
        nage = np.arange(0,len(age),1)

        #
        # Metallicity
        #
        try:
            ZFIX = float(inputs['ZFIX'])
            delZ = 0.0001
            Zmin, Zmax = ZFIX, ZFIX+delZ
            Zall = np.arange(Zmin, Zmax, delZ) # in logZsun
        except:
            Zmax, Zmin = float(inputs['ZMAX']), float(inputs['ZMIN'])
            delZ = float(inputs['DELZ'])
            if Zmax == Zmin or delZ==0:
                delZ  = 0.0001
                Zmax = Zmin+delZ
                Zall = np.arange(Zmin, Zmax, delZ) # in logZsun
            else:
                Zall = np.arange(Zmin, Zmax, delZ) # in logZsun


        # For z prior.
        delzz  = 0.001
        zlimu  = 6.
        snlim  = 1
        zliml  = zgal - 0.5
        agemax = self.cosmo.age(zgal).value #, use_flat=True, **cosmo)/cc.Gyr_s

        # N of param:
        if int(inputs['ZEVOL']) == 1:
            ndim = int(len(nage) * 2 + 1)
            print('Metallicity evolution is on.')
            if int(inputs['ZMC']) == 1:
                ndim += 1
            print('No of params are : %d'%(ndim))
        else:
            ndim = int(len(nage) + 1 + 1)
            print('Metallicity evolution is off.')
            if int(inputs['ZMC']) == 1:
                ndim += 1
            print('No of params are : %d'%(ndim))
        pass

        #
        # Line
        #
        try:
            LW0 = inputs['LINE']
            LW0 = [float(x.strip()) for x in LW0.split(',')]
        except:
            LW0 = []

        #
        # Params for MCMC
        #
        nmc      = int(inputs['NMC'])
        nwalk    = int(inputs['NWALK'])
        nmc_cz   = int(inputs['NMCZ'])
        nwalk_cz = int(inputs['NWALKZ'])
        f_Zevol  = int(inputs['ZEVOL'])
        #f_zvis   = int(inputs['ZVIS'])
        try:
            fzmc = int(inputs['ZMC'])
        except:
            fzmc = 0

        #
        # If FIR data;
        #
        try:
            DT0 = float(inputs['TDUST_LOW'])
            DT1 = float(inputs['TDUST_HIG'])
            dDT = float(inputs['TDUST_DEL'])
            Temp= np.arange(DT0,DT1,dDT)
            f_dust = True
            print('FIR fit is on.')
        except:
            f_dust = False
            pass

        #
        # Tau for MCMC parameter; not as fitting parameters.
        #
        tau0 = inputs['TAU0']
        tau0 = [float(x.strip()) for x in tau0.split(',')]

        #
        # Dust model specification;
        #
        try:
            dust_model = int(inputs['DUST_MODEL'])
        except:
            dust_model = 0

        #from .function_class import Func
        #from .basic_func import Basic
        fnc  = self.fnc  #Func(Zall, nage, dust_model=dust_model, DIR_TMP=DIR_TMP) # Set up the number of Age/ZZ
        bfnc = self.bfnc #Basic(Zall)

        # Open ascii file and stock to array.
        #lib = open_spec(ID0, PA0)
        lib     = self.lib #= fnc.open_spec_fits(ID0, PA0, fall=0, tau0=tau0)
        lib_all = self.lib_all #= fnc.open_spec_fits(ID0, PA0, fall=1, tau0=tau0)

        if f_dust:
            lib_dust     = self.lib_dust #= fnc.open_spec_dust_fits(ID0, PA0, Temp, fall=0, tau0=tau0)
            lib_dust_all = self.lib_dust_all #= fnc.open_spec_dust_fits(ID0, PA0, Temp, fall=1, tau0=tau0)

        # Error parameter
        try:
            self.ferr = int(inputs['F_ERR'])
        except:
            self.ferr = 0
            pass

        #################
        # Observed Data
        #################
        ##############
        # Spectrum
        ##############
        dat = np.loadtxt(DIR_TMP + 'spec_obs_' + ID0 + '_PA' + PA0 + '.cat', comments='#')
        NR  = dat[:,0]
        x   = dat[:,1]
        fy00  = dat[:,2]
        ey00  = dat[:,3]

        con0 = (NR<1000)
        fy0  = fy00[con0] * Cz0
        ey0  = ey00[con0] * Cz0
        con1 = (NR>=1000) & (NR<10000)
        fy1  = fy00[con1] * Cz1
        ey1  = ey00[con1] * Cz1

        ##############
        # Broadband
        ##############
        try:
            dat = np.loadtxt(DIR_TMP + 'bb_obs_' + ID0 + '_PA' + PA0 + '.cat', comments='#')
            NRbb = dat[:, 0]
            xbb  = dat[:, 1]
            fybb = dat[:, 2]
            eybb = dat[:, 3]
            exbb = dat[:, 4]
            fy2 = fybb
            ey2 = eybb

            fy01 = np.append(fy0,fy1)
            fy   = np.append(fy01,fy2)
            ey01 = np.append(ey0,ey1)
            ey   = np.append(ey01,ey2)

            wht  = 1./np.square(ey)
            wht2 = check_line_man(fy, x, wht, fy, zprev, LW0)
            sn   = fy/ey
        except: # if no BB;
            xbb  = np.asarray([100.])
            fy   = np.asarray([0.]) #np.append(fy01,fy2)
            ey   = np.asarray([100.]) #np.append(ey01,ey2)
            wht2 = np.asarray([0.]) #check_line_man(fy, x, wht, fy, zprev, LW0)
            sn   = np.asarray([0.]) #fy/ey


        from .posterior_flexible import Post
        class_post = Post(self)
        #from posterior_flexible import residual,lnprob

        ###############################
        # Add parameters
        ###############################
        fit_params = Parameters()
        try:
        #if True:
            age_fix = inputs['AGEFIX']
            age_fix = [float(x.strip()) for x in age_fix.split(',')]
            aamin = []
            print('\n')
            print('##########################')
            print('AGEFIX is found.\nAge will be fixed to:')
            for age_tmp in age_fix:
                ageind = np.argmin(np.abs(age_tmp-np.asarray(age[:])))
                aamin.append(ageind)
                print('%6s Gyr'%(age[ageind]))
            print('##########################')
            for aa in range(len(age)):
                if aa not in aamin:
                    fit_params.add('A'+str(aa), value=0, min=0, max=1e-10)
                else:
                    fit_params.add('A'+str(aa), value=1, min=0, max=1e3)
        except:
            for aa in range(len(age)):
                if age[aa] == 99:
                    fit_params.add('A'+str(aa), value=0, min=0, max=1e-10)
                elif age[aa]>agemax:
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
            for aa in range(len(age)):
                if age[aa] == 99 or age[aa]>agemax:
                    fit_params.add('Z'+str(aa), value=0, min=0, max=1e-10)
                else:
                    fit_params.add('Z'+str(aa), value=0, min=np.min(Zall), max=np.max(Zall))
        else:
            try:
                ZFIX = float(inputs['ZFIX'])
                aa = 0
                fit_params.add('Z'+str(aa), value=0, min=ZFIX, max=ZFIX+0.0001)
            except:
                aa = 0
                if np.min(Zall)==np.max(Zall):
                    fit_params.add('Z'+str(aa), value=0, min=np.min(Zall), max=np.max(Zall)+0.0001)
                else:
                    fit_params.add('Z'+str(aa), value=0, min=np.min(Zall), max=np.max(Zall))

        ####################################
        # Initial Metallicity Determination
        ####################################
        #nZtmp = int((Zmax-Zmin)/delZ)
        ZZtmp = np.arange(Zmin,Zmax,delZ)

        #
        # Get initial parameters
        #
        out,chidef,Zbest = get_leastsq(inputs,ZZtmp,fneld,age,fit_params,class_post.residual,fy,wht2,ID0,PA0)

        #
        # Best fit
        #
        keys = fit_report(out).split('\n')
        for key in keys:
            if key[4:7] == 'chi':
                skey = key.split(' ')
                csq  = float(skey[14])
            if key[4:7] == 'red':
                skey = key.split(' ')
                rcsq = float(skey[7])

        fitc = [csq, rcsq] # Chi2, Reduced-chi2
        ZZ   = Zbest # This is really important/does affect lnprob/residual.

        print('\n\n')
        print('#####################################')
        print('Zbest, chi are;',Zbest,chidef)
        print('Params are;',fit_report(out))
        print('#####################################')
        print('\n\n')

        Av_tmp = out.params['Av'].value
        AA_tmp = np.zeros(len(age), dtype='float64')
        ZZ_tmp = np.zeros(len(age), dtype='float64')
        fm_tmp, xm_tmp = fnc.tmp04_val(ID0, PA0, out, zprev, lib, tau0=tau0)

        ########################
        # Check redshift
        ########################
        zrecom = zprev

        # Observed data.
        con_cz = (NR<10000) #& (sn>snlim)
        fy_cz  = fy[con_cz]
        ey_cz  = ey[con_cz]
        x_cz   = x[con_cz] # Observed range
        NR_cz  = NR[con_cz]

        xm_s = xm_tmp / (1+zprev) * (1+zrecom)
        fm_s = np.interp(x_cz, xm_s, fm_tmp)

        if flag_m == 0:
            dez = 0.5
        else:
            dez = 0.2

        #
        # For Eazy
        #
        try:
            eaz_pz = inputs['EAZY_PZ']
            f_eazy = True
        except:
            f_eazy = False
        if f_eazy:
            dprob = np.loadtxt(eaz_pz, comments='#')
            zprob = dprob[:,0]
            cprob = dprob[:,1]
            # Then interpolate to a common grid;
            zz_prob = np.arange(0,13,delzz)
            cprob_s = np.interp(zz_prob, zprob, cprob)
            prior_s = np.exp(-0.5 * cprob_s)
            prior_s /= np.sum(prior_s)
            '''
            plt.close()
            plt.plot(zz_prob,prior_s,linestyle='-')
            plt.show()
            '''
        else:
            zz_prob  = np.arange(0,13,delzz)
            prior_s  = zz_prob * 0 + 1.
            prior_s /= np.sum(prior_s)

        # Plot;
        if fzvis==1:
            plt.plot(x_cz/(1+zprev)*(1.+zrecom),fm_s,'gray', linestyle='--', linewidth=0.5, label='Default ($z=%.5f$)'%(zprev)) # Model based on input z.
            plt.plot(x_cz, fy_cz,'b', linestyle='-', linewidth=0.5, label='Obs.') # Observation
            plt.errorbar(x_cz, fy_cz, yerr=ey_cz, color='b', capsize=0, linewidth=0.5) # Observation
        try:
            print('############################')
            print('Start MCMC for redshift fit')
            print('############################')
            res_cz, fitc_cz = check_redshift(fy_cz, ey_cz, x_cz, fm_tmp, xm_tmp/(1+zprev), zprev, dez, prior_s, NR_cz, zliml, zlimu, delzz, nmc_cz, nwalk_cz)
            z_cz    = np.percentile(res_cz.flatchain['z'], [16,50,84])
            scl_cz0 = np.percentile(res_cz.flatchain['Cz0'], [16,50,84])
            scl_cz1 = np.percentile(res_cz.flatchain['Cz1'], [16,50,84])

            zrecom  = z_cz[1]
            Czrec0  = scl_cz0[1]
            Czrec1  = scl_cz1[1]

            # Switch to peak redshift:
            from scipy import stats
            from scipy.stats import norm
            # find minimum and maximum of xticks, so we know
            # where we should compute theoretical distribution
            ser = res_cz.flatchain['z']
            xmin, xmax = zprev-0.2, zprev+0.2
            lnspc = np.linspace(xmin, xmax, len(ser))
            print('\n\n')
            print('Recommended redshift, Cz0 and Cz1, %.5f %.5f %.5f, with chi2/nu=%.3f'%(zrecom, Cz0 * Czrec0, Cz1 * Czrec1, fitc_cz[1]))
            print('\n\n')

        except:
            print('z fit failed. No spectral data set?')
            try:
                ezl = float(inputs['EZL'])
                ezu = float(inputs['EZU'])
                print('Redshift error is taken from input file.')
                if ezl<ezmin:
                    ezl = ezmin #0.03
                if ezu<ezmin:
                    ezu = ezmin #0.03
            except:
                ezl = ezmin
                ezu = ezmin
                print('Redshift error is assumed to %.1f.'%(ezl))
            z_cz    = [zprev-ezl,zprev,zprev+ezu]
            zrecom  = z_cz[1]
            scl_cz0 = [1.,1.,1.]
            scl_cz1 = [1.,1.,1.]
            Czrec0  = scl_cz0[1]
            Czrec1  = scl_cz1[1]

        f_fitgauss=0

        xm_s = xm_tmp / (1+zprev) * (1+zrecom)
        fm_s = np.interp(x_cz, xm_s, fm_tmp)
        whtl = 1/np.square(ey_cz)
        try:
            wht3, ypoly = check_line_cz_man(fy_cz, x_cz, whtl, fm_s, zrecom, LW0)
        except:
            wht3, ypoly = whtl, fy_cz
        con_line = (wht3==0)

        if fzvis==1:
            plt.plot(x_cz, fm_s, 'r', linestyle='-', linewidth=0.5, label='Updated model ($z=%.5f$)'%(zrecom)) # Model based on recomended z.
            plt.plot(x_cz[con_line], fm_s[con_line], color='orange', marker='o', linestyle='', linewidth=3.)
            # Plot lines for reference
            for ll in range(len(LW)):
                try:
                    conpoly = (x_cz/(1.+zrecom)>3000) & (x_cz/(1.+zrecom)<8000)
                    yline = np.max(ypoly[conpoly])
                    yy    = np.arange(yline/1.02, yline*1.1)
                    xxpre = yy * 0 + LW[ll] * (1.+zprev)
                    xx    = yy * 0 + LW[ll] * (1.+zrecom)
                    plt.plot(xxpre, yy/1.02, linewidth=0.5, linestyle='--', color='gray')
                    plt.text(LW[ll] * (1.+zprev), yline/1.05, '%s'%(LN[ll]), fontsize=8, color='gray')
                    plt.plot(xx, yy, linewidth=0.5, linestyle='-', color='orangered')
                    plt.text(LW[ll] * (1.+zrecom), yline, '%s'%(LN[ll]), fontsize=8, color='orangered')
                except:
                    pass

            plt.plot(xbb, fybb, '.r', linestyle='', linewidth=0, zorder=4, label='Obs.(BB)')
            plt.plot(xm_tmp, fm_tmp, color='gray', marker='.', ms=0.5, linestyle='', linewidth=0.5, zorder=4, label='Model')
            try:
                xmin, xmax = np.min(x_cz)/1.1,np.max(x_cz)*1.1
            except:
                xmin, xmax = 2000,10000
            plt.xlim(xmin,xmax)
            try:
                plt.ylim(0,yline*1.1)
            except:
                pass
            plt.xlabel('Wavelength ($\mathrm{\AA}$)')
            plt.ylabel('$F_\\nu$ (arb.)')
            plt.legend(loc=0)

            zzsigma  = ((z_cz[2] - z_cz[0])/2.)/zprev
            zsigma   = np.abs(zprev-zrecom) / (zprev)
            C0sigma  = np.abs(Czrec0-Cz0)/Cz0
            eC0sigma = ((scl_cz0[2]-scl_cz0[0])/2.)/Cz0
            C1sigma  = np.abs(Czrec1-Cz1)/Cz1
            eC1sigma = ((scl_cz1[2]-scl_cz1[0])/2.)/Cz1

            print('Input redshift is %.3f per cent agreement.'%((1.-zsigma)*100))
            print('Error is %.3f per cent.'%(zzsigma*100))
            print('Input Cz0 is %.3f per cent agreement.'%((1.-C0sigma)*100))
            print('Error is %.3f per cent.'%(eC0sigma*100))
            print('Input Cz1 is %.3f per cent agreement.'%((1.-C1sigma)*100))
            print('Error is %.3f per cent.'%(eC1sigma*100))
            plt.show()

            #
            # Ask interactively;
            #
            flag_z = raw_input('Do you want to continue with original redshift, Cz0 and Cz1, %.5f %.5f %.5f? ([y]/n/m) '%(zprev, Cz0, Cz1))
        else:
            flag_z = 'y'

        #################################################
        # Gor for mcmc phase
        #################################################
        if flag_z == 'y' or flag_z == '':
            zrecom  = zprev
            Czrec0  = Cz0
            Czrec1  = Cz1

            #######################
            # Added
            #######################
            if fzmc == 1:
                out_keep = out
                #sigz = 1.0 #3.0
                fit_params.add('zmc', value=zrecom, min=zrecom-(z_cz[1]-z_cz[0])*sigz, max=zrecom+(z_cz[2]-z_cz[1])*sigz)
                #print(zrecom,zrecom-(z_cz[1]-z_cz[0])*sigz,zrecom+(z_cz[2]-z_cz[1])*sigz)

                #####################
                # Error parameter
                #####################
                try:
                    ferr = int(inputs['F_ERR'])
                    if ferr == 1:
                        fit_params.add('f', value=1e-2, min=0, max=1e2)
                        ndim += 1
                except:
                    ferr = 0
                    pass

                #####################
                # Dust;
                #####################
                if f_dust:
                    Tdust = np.arange(DT0,DT1,dDT)
                    fit_params.add('TDUST', value=len(Tdust)/2., min=0, max=len(Tdust)-1)
                    #fit_params.add('TDUST', value=1, min=0, max=len(Tdust)-1)
                    fit_params.add('MDUST', value=1e6, min=0, max=1e10)
                    ndim += 2

                    # Append data;
                    dat_d = np.loadtxt(DIR_TMP + 'spec_dust_obs_' + ID0 + '_PA' + PA0 + '.cat', comments='#')
                    x_d   = dat_d[:,1]
                    fy_d  = dat_d[:,2]
                    ey_d  = dat_d[:,3]

                    fy = np.append(fy,fy_d)
                    x  = np.append(x,x_d)
                    wht= np.append(wht,1./np.square(ey_d))
                    wht2= check_line_man(fy, x, wht, fy, zprev, LW0)

                # Then, minimize again.
                if fneld == 1:
                    fit_name = 'nelder'
                else:
                    fit_name = 'powell'
                out = minimize(class_post.residual, fit_params, args=(fy, wht2, f_dust), method=fit_name) # It needs to define out with redshift constrain.
                print(fit_report(out))

                # Fix params to what we had before.
                out.params['zmc'].value = zrecom
                out.params['Av'].value  = out_keep.params['Av'].value
                for aa in range(len(age)):
                    out.params['A'+str(aa)].value = out_keep.params['A'+str(aa)].value
                    try:
                        out.params['Z'+str(aa)].value = out_keep.params['Z'+str(aa)].value
                    except:
                        out.params['Z0'].value = out_keep.params['Z0'].value

            ##############################
            # Save fig of z-distribution.
            ##############################
            try: # if spectrum;
                fig = plt.figure(figsize=(6.5,2.5))
                fig.subplots_adjust(top=0.96, bottom=0.16, left=0.09, right=0.99, hspace=0.15, wspace=0.25)
                ax1 = fig.add_subplot(111)
                #n, nbins, patches = ax1.hist(res_cz.flatchain['z'], bins=200, normed=False, color='gray',label='')
                n, nbins, patches = ax1.hist(res_cz.flatchain['z'], bins=200, normed=True, color='gray',label='')
                if f_fitgauss==1:
                    ax1.plot(lnspc, pdf_g, label='Gaussian fit', color='g', linestyle='-') # plot it

                ax1.set_xlim(m-s*3,m+s*3)
                yy = np.arange(0,np.max(n),1)
                xx = yy * 0 + z_cz[1]
                ax1.plot(xx,yy,linestyle='-',linewidth=1,color='orangered',label='$z=%.5f_{-%.5f}^{+%.5f}$\n$C_z0=%.3f$\n$C_z1=%.3f$'%(z_cz[1],z_cz[1]-z_cz[0],z_cz[2]-z_cz[1], Czrec0, Czrec1))
                xx = yy * 0 + z_cz[0]
                ax1.plot(xx,yy,linestyle='--',linewidth=1,color='orangered')
                xx = yy * 0 + z_cz[2]
                ax1.plot(xx,yy,linestyle='--',linewidth=1,color='orangered')
                xx = yy * 0 + zprev
                ax1.plot(xx,yy,linestyle='-',linewidth=1,color='royalblue')
                ax1.set_xlabel('Redshift')
                ax1.set_ylabel('$dn/dz$')
                ax1.legend(loc=0)
                plt.savefig('zprob_' + ID0 + '_PA' + PA0 + '.pdf', dpi=300)
                plt.close()
            except:
                print('z-distribution figure is not generated.')
                pass

            ##############################
            print('\n\n')
            print('###############################')
            print('Input redshift is adopted.')
            print('Starting long journey in MCMC.')
            print('###############################')
            print('\n\n')
            #################
            # Initialize mm.
            #################
            mm = 0
            # add a noise parameter
            # out.params.add('f', value=1, min=0.001, max=20)
            wht2 = wht

            ################################
            print('\nMinimizer Defined\n')
            mini = Minimizer(class_post.lnprob, out.params, fcn_args=[fy,wht2,f_dust], f_disp=f_disp, f_move=f_move)
            print('######################')
            print('### Starting emcee ###')
            print('######################')
            import multiprocess
            ncpu0 = int(multiprocess.cpu_count()/2)
            try:
                ncpu = int(inputs['NCPU'])
                if ncpu > ncpu0:
                    print('!!! NCPU is larger than No. of CPU. !!!')
                    #print('Now set to %d'%(ncpu0))
                    #ncpu = ncpu0
            except:
                ncpu = ncpu0
                pass

            print('No. of CPU is set to %d'%(ncpu))
            start_mc = timeit.default_timer()
            res  = mini.emcee(burn=int(nmc/2), steps=nmc, thin=10, nwalkers=nwalk, params=out.params, is_weighted=True, ntemps=ntemp, workers=ncpu)
            stop_mc  = timeit.default_timer()
            tcalc_mc = stop_mc - start_mc
            print('###############################')
            print('### MCMC part took %.1f sec ###'%(tcalc_mc))
            print('###############################')

            #----------- Save pckl file
            #-------- store chain into a cpkl file
            start_mc = timeit.default_timer()
            import corner
            burnin   = int(nmc/2)
            savepath = './'
            cpklname = 'chain_' + ID0 + '_PA' + PA0 + '_corner.cpkl'
            savecpkl({'chain':res.flatchain,
                          'burnin':burnin, 'nwalkers':nwalk,'niter':nmc,'ndim':ndim},
                         savepath+cpklname) # Already burn in
            stop_mc  = timeit.default_timer()
            tcalc_mc = stop_mc - start_mc
            print('#################################')
            print('### Saving chain took %.1f sec'%(tcalc_mc))
            print('#################################')

            Avmc = np.percentile(res.flatchain['Av'], [16,50,84])
            #Zmc  = np.percentile(res.flatchain['Z'], [16,50,84])
            Avpar = np.zeros((1,3), dtype='float64')
            Avpar[0,:] = Avmc

            out = res
            ####################
            # Best parameters
            ####################
            Amc  = np.zeros((len(age),3), dtype='float64')
            Ab   = np.zeros(len(age), dtype='float64')
            Zmc  = np.zeros((len(age),3), dtype='float64')
            Zb   = np.zeros(len(age), dtype='float64')
            NZbest = np.zeros(len(age), dtype='int')
            f0     = fits.open(DIR_TMP + 'ms_' + ID0 + '_PA' + PA0 + '.fits')
            sedpar = f0[1]
            ms     = np.zeros(len(age), dtype='float64')
            for aa in range(len(age)):
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

            if f_dust:
                Mdust_mc = np.zeros(3, dtype='float64')
                Tdust_mc = np.zeros(3, dtype='float64')
                Mdust_mc[:] = np.percentile(res.flatchain['MDUST'], [16,50,84])
                Tdust_mc[:] = np.percentile(res.flatchain['TDUST'], [16,50,84])
                print(Mdust_mc)
                print(Tdust_mc)

            ####################
            # MCMC corner plot.
            ####################
            if mcmcplot:
                fig1 = corner.corner(res.flatchain, labels=res.var_names, \
                label_kwargs={'fontsize':16}, quantiles=[0.16, 0.84], show_titles=False, \
                title_kwargs={"fontsize": 14}, truths=list(res.params.valuesdict().values()), \
                plot_datapoints=False, plot_contours=True, no_fill_contours=True, \
                plot_density=False, levels=[0.68, 0.95, 0.997], truth_color='gray', color='#4682b4')
                fig1.savefig('SPEC_' + ID0 + '_PA' + PA0 + '_corner.pdf')
                plt.close()

            #########################
            msmc0 = 0
            for aa in range(len(age)):
                msmc0 += res.flatchain['A'+str(aa)]*ms[aa]
            msmc = np.percentile(msmc0, [16,50,84])

            # Do analysis on MCMC results.
            # Write to file.
            stop  = timeit.default_timer()
            tcalc = stop - start

            # Load writing package;
            from .writing import Analyze
            wrt = Analyze(inputs) # Set up for input

            start_mc = timeit.default_timer()
            wrt.get_param(res, lib_all, zrecom, Czrec0, Czrec1, z_cz[:], scl_cz0[:], scl_cz1[:], fitc[:], tau0=tau0, tcalc=tcalc)
            stop_mc  = timeit.default_timer()
            tcalc_mc = stop_mc - start_mc
            print('##############################################')
            print('### Writing params tp file took %.1f sec ###'%(tcalc_mc))
            print('##############################################')

            return 0, zrecom, Czrec0, Czrec1

        ###################################################################
        elif flag_z == 'm':
            zrecom = float(raw_input('What is your manual input for redshift? '))
            Czrec0 = float(raw_input('What is your manual input for Cz0? '))
            Czrec1 = float(raw_input('What is your manual input for Cz1? '))
            print('\n\n')
            print('Generate model templates with input redshift and Scale.')
            print('\n\n')
            return 1, zrecom, Czrec0, Czrec1

        else:
            print('\n\n')
            print('Terminated because of redshift estimate.')
            print('Generate model templates with recommended redshift.')
            print('\n\n')

            flag_gen = raw_input('Do you want to make templates with recommended redshift, Cz0, and Cz1 , %.5f %.5f %.5f? ([y]/n) '%(zrecom, Czrec0, Czrec1))
            if flag_gen == 'y' or flag_gen == '':
                #return 1, zrecom, Cz0 * Czrec0, Cz1 * Czrec1
                return 1, zrecom, Czrec0, Czrec1
            else:
                print('\n\n')
                print('There is nothing to do.')
                print('Terminating process.')
                print('\n\n')
                return -1, zprev, Czrec0, Czrec1
