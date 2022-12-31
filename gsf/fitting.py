"""
This includes the main part of gsf.

Original copyright:
   Copyright (c) 2021 Takahiro Morishita, STScI
"""
import numpy as np
import sys
import matplotlib.pyplot as plt
from lmfit import Model, Parameters, minimize, fit_report#, Minimizer
from numpy import log10
import pickle as cPickle
import os.path
import random
import string
import timeit
from scipy import stats
from scipy.stats import norm
import scipy.interpolate as interpolate
from astropy.io import fits,ascii
import corner

# import from custom codes
from .function import check_line_man, check_line_cz_man, calc_Dn4, savecpkl, get_leastsq
from .zfit import check_redshift,get_chi2
from .writing import get_param
from .function_class import Func
from .minimizer import Minimizer

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

# NRbb_lim = 10000 # BB data is associated with ids greater than this number.

class Mainbody():
    '''
    The list of (possible) `Mainbody` attributes is given below:

    Attributes
    ----------
    nimf : int 
        0:Salpeter, 1:Chabrier, 2:Kroupa, 3:vanDokkum08.
    Z : float array
        Stellar phase metallicity in logZsun.
    age : float array
        Age, in Gyr.
    fneb : int
        flag for adding nebular emission. 0: No, 1: Yes.
    tau0 : float array
        Width of age bin. If you want to fix it to a specific value, set it to >0.01, in Gyr.
        Otherwise, it would be either minimum value (=0.01; if one age bin), 
        or the width to the next age bin.
    '''

    def __init__(self, inputs, c:float=3e18, Mpc_cm:float=3.08568025e+24, m0set:float=25.0, pixelscale:float=0.06, Lsun:float=3.839*1e33, 
        cosmo=None, idman=None, zman=None, NRbb_lim=10000):
        '''
        Parameters
        ----------
        NRbb_lim : int
            BB data is associated with ids greater than this number.
        '''
        self.update_input(inputs, idman=idman, zman=zman)
        self.NRbb_lim = NRbb_lim


    def update_input(self, inputs, c:float=3e18, Mpc_cm:float=3.08568025e+24, m0set:float=25.0, pixelscale:float=0.06, Lsun:float=3.839*1e33, cosmo=None,
                    idman=None, zman=None, sigz:float=5.0):
        '''
        The purpose of this module is to register/update the parameter attributes in `Mainbody`
        by visiting the configuration file.

        Parameters
        ----------
        inputs : str
            Configuration file that lists parameters.

        Mpc_cm : float, optional
            Conversion factor from Mpc-to-cm

        pixelscale : float, optional
            Conversion factor from pixel-to-arcsec

        '''
        # Then register;
        self.inputs = inputs
        self.c = c
        self.Mpc_cm = Mpc_cm
        self.pixelscale = pixelscale
        self.Lsun = Lsun
        self.sigz = sigz
        self.fitc_cz_prev = None

        # Magzp;
        try:
            self.m0set = float(inputs['MAGZP'])
        except:
            print('MAGZP is not found. Set to %.2f'%(m0set))
            self.m0set = m0set
        self.d = 10**((48.6+self.m0set)/2.5) # Conversion factor from [ergs/s/cm2/A] to Fnu.

        if cosmo == None:
            from astropy.cosmology import WMAP9 as cosmo
            self.cosmo = cosmo
        else:
            self.cosmo = cosmo

        if idman != None:
            self.ID = idman
        else:
            self.ID = inputs['ID']
        print('\nFitting : %s\n'%self.ID)

        # Read catalog;
        try:
            self.CAT_BB = inputs['CAT_BB']
            self.fd_cat = ascii.read(self.CAT_BB)
        except:
            self.CAT_BB = None
            self.fd_cat = None

        if zman != None:
            self.zgal = zman
            self.zmcmin = None
            self.zmcmax = None
        else:
            try:
                self.zgal = float(inputs['ZMC'])
            except:
                iix = np.where(self.fd_cat['id'] == self.ID)
                self.zgal = float(self.fd_cat['redshift'][iix])

            try:
                self.zmcmin = float(inputs['ZMCMIN'])
                self.zmcmax = float(inputs['ZMCMAX'])
            except:
                try:
                    self.zmcmin = self.zgal - float(self.fd_cat['ez_l'][iix])
                    self.zmcmax = self.zgal + float(self.fd_cat['ez_u'][iix])
                except:
                    print('ZMCMIN and ZMCMAX cannot be found. z range is set to z pm 1.0')
                    self.zmcmin = None
                    self.zmcmax = None

        # Data directory;
        self.DIR_TMP = inputs['DIR_TEMP']
        if not os.path.exists(self.DIR_TMP):
            os.mkdir(self.DIR_TMP)

        # Minimization;
        self.fneld = self.inputs['FNELD']

        # Mdyn;
        try:
            #self.Mdyn = float(inputs['MDYN'])
            if int(inputs['F_MDYN']) == 1:
                self.f_Mdyn = True
            else:
                self.f_Mdyn = False
        except:
            self.f_Mdyn = False

        if self.f_Mdyn:
            iix = np.where(self.fd_cat['id'] == int(self.ID))
            try:
                self.logMdyn = float(self.fd_cat['logMdyn'][iix])
                self.elogMdyn = float(self.fd_cat['elogMdyn'][iix])
                self.f_Mdyn = True
            except:
                self.f_Mdyn = False

        print('f_Mdyn is set to %s\n'%self.f_Mdyn)
        
        #self.f_Mdyn = True
        #self.logMdyn = 11.1
        #self.elogMdyn = 0.1

        #if self.f_Mdyn:
        #    # If Mdyn is included.
        #    self.af = asdf.open(self.DIR_TMP + 'spec_all_' + self.ID + '_PA' + self.PA + '.asdf')
        try:
            self.DIR_EXTR = inputs['DIR_EXTR']
        except:
            self.DIR_EXTR = False

        try:
            # Scaling for grism; 
            self.Cz0 = float(inputs['CZ0'])
            self.Cz1 = float(inputs['CZ1'])
            self.Cz2 = float(inputs['CZ2'])
        except:
            self.Cz0 = 1
            self.Cz1 = 1
            self.Cz2 = 1

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
        # Becuase of force_no_neb, add logUs regardless of `ADD_NEBULAE` flag.
        self.fneb = False
        self.nlogU = 0
        try:
            if int(inputs['ADD_NEBULAE']) == 1:
                self.fneb = True
                try:
                    self.logUMIN = float(inputs['logUMIN'])
                    self.logUMAX = float(inputs['logUMAX'])
                    self.DELlogU = float(inputs['DELlogU'])
                except:
                    self.logUMIN = -2.5
                    self.logUMAX = -2.0
                    self.DELlogU = 0.5
                self.logUs = np.arange(self.logUMIN, self.logUMAX, self.DELlogU)
                self.nlogU = len(self.logUs)

                try:
                    self.logUFIX = float(inputs['logUFIX'])
                    self.nlogU = 1
                    self.logUMIN = self.logUFIX
                    self.logUMAX = self.logUFIX
                    self.DELlogU = 0
                    self.logUs = np.asarray([self.logUMAX])
                except:
                    self.logUFIX = None
            else:
                self.fneb = False                
                self.logUMIN = -2.5
                self.logUMAX = -2.0
                self.DELlogU = 0.5
                self.logUs = np.arange(self.logUMIN, self.logUMAX, self.DELlogU)
        except:
            print('No nebular added.')
            self.fneb = False
            self.logUMIN = -2.5
            self.logUMAX = -2.0
            self.DELlogU = 0.5
            self.logUs = np.arange(self.logUMIN, self.logUMAX, self.DELlogU)
            pass
        
        # Outpu directory;
        try:
            self.DIR_OUT = inputs['DIR_OUT']
            if not os.path.exists(self.DIR_OUT):
                os.mkdir(self.DIR_OUT)
        except:
            self.DIR_OUT = './'

        # Filter response curve directory, if bb catalog is provided.
        self.DIR_FILT = inputs['DIR_FILT']
        try:
            self.filts = inputs['FILTER']
            self.filts = [x.strip() for x in self.filts.split(',')]
        except:
            self.filts = []
            for column in self.fd_cat.columns:
                if column[0] == 'F':
                    self.filts.append(column[1:])
            pass

        self.band = {}
        for ii in range(len(self.filts)):
            fd = np.loadtxt(self.DIR_FILT + self.filts[ii] + '.fil', comments='#')
            self.band['%s_lam'%(self.filts[ii])] = fd[:,1]
            self.band['%s_res'%(self.filts[ii])] = fd[:,2] / np.max(fd[:,2])
            ffil_cum = np.cumsum(fd[:,2])
            ffil_cum /= ffil_cum.max()
            con = (ffil_cum>0.05) & (ffil_cum<0.95)
            self.band['%s_fwhm'%(self.filts[ii])] = np.max(fd[:,1][con]) - np.min(fd[:,1][con])       

        # Filter response curve directory, for RF colors.
        self.filts_rf = '93,141,95,220,160'.split(',')
        self.band_rf = {}
        for ii in range(len(self.filts_rf)):
            fd = np.loadtxt(self.DIR_FILT + self.filts_rf[ii] + '.fil', comments='#')
            self.band_rf['%s_lam'%(self.filts_rf[ii])] = fd[:,1]
            self.band_rf['%s_res'%(self.filts_rf[ii])] = fd[:,2] / np.max(fd[:,2])

        # Tau comparison?
        # -> Deprecated;
        # self.ftaucomp = 0

        # Check if func model for SFH;
        try:
            self.SFH_FORM = int(inputs['SFH_FORM'])
        except:
            self.SFH_FORM = -99

        # This is for non-functional form for SFH;
        if self.SFH_FORM == -99:
            # Age
            try:
                self.age = np.asarray([float(x.strip()) for x in inputs['AGE'].split(',')])
            except:
                try:
                    self.delage = float(inputs['DELAGE'])
                except:
                    self.delage = 0.1
                try:
                    self.agemax = float(inputs['AGEMAX'])
                    self.agemin = float(inputs['AGEMIN'])
                except:
                    self.agemax = 14.0
                    self.agemin = 0.003
                logage = np.arange(np.log10(self.agemin), np.log10(self.agemax), self.delage)
                self.age = 10**logage

            try:
                self.age_fix = [float(x.strip()) for x in inputs['AGEFIX'].split(',')]
                aamin = []
                print('\n')
                print('##########################')
                print('AGEFIX is found.\nAge will be fixed to:')
                for age_tmp in self.age_fix:
                    ageind = np.argmin(np.abs(age_tmp-np.asarray(self.age[:])))
                    aamin.append(ageind)
                    print('%6s Gyr'%(self.age[ageind]))
                print('##########################')
                self.aamin = aamin
            except:
                aamin = []
                for nn,age_tmp in enumerate(self.age):
                    aamin.append(nn)
                self.aamin = aamin

            self.npeak = len(self.age)
            self.nage = np.arange(0,len(self.age),1)
            
        else: # This is for functional form for SFH;
            self.agemax = float(inputs['AGEMAX'])
            self.agemin = float(inputs['AGEMIN'])
            self.delage = float(inputs['DELAGE'])

            if self.agemax-self.agemin<self.delage:
                self.delage = 0.0001
                self.agemax = self.agemin + self.delage

            self.ageparam = np.arange(self.agemin, self.agemax, self.delage)
            self.nage = len(self.ageparam)

            self.taumax = float(inputs['TAUMAX'])
            self.taumin = float(inputs['TAUMIN'])
            self.deltau = float(inputs['DELTAU'])

            if self.taumax-self.taumin<self.deltau:
                self.deltau = 0.0001
                self.taumax = self.taumin + self.deltau

            self.tau = np.arange(self.taumin, self.taumax, self.deltau)
            self.ntau = len(self.tau)

            self.npeak = int(inputs['NPEAK'])
            self.age = np.arange(0,self.npeak,1) # This is meaningless.
            aamin = []
            for nn,age_tmp in enumerate(self.age):
                aamin.append(nn)
            self.aamin = aamin

        # SNlimit;
        try:
            self.SNlim = float(inputs['SNLIM']) # SN for non-detection in BB catalog.
        except:
            self.SNlim = 1.0

        # Redshift as a param;
        try:
            self.fzmc = int(inputs['F_ZMC'])
        except:
            self.fzmc = 0
            print('Cannot find ZMC. Set to %d.'%(self.fzmc))

        # Metallicity
        self.has_ZFIX = False
        try:
            self.ZFIX = float(inputs['ZFIX'])
            try:
                self.delZ = float(inputs['DELZ'])
                self.Zmax, self.Zmin = float(inputs['ZMAX']), float(inputs['ZMIN'])
            except:
                self.delZ = 0.0001
                self.Zmin, self.Zmax = self.ZFIX, self.ZFIX + self.delZ
            self.Zall = np.arange(self.Zmin, self.Zmax, self.delZ)
            print('\n##########################')
            print('ZFIX is found.\nZ will be fixed to: %.2f'%(self.ZFIX))
            self.has_ZFIX = True
        except:
            self.Zmax, self.Zmin = float(inputs['ZMAX']), float(inputs['ZMIN'])
            self.delZ = float(inputs['DELZ'])
            if self.Zmax == self.Zmin or self.delZ == 0:
                self.delZ = 0.0
                self.ZFIX = self.Zmin
                self.Zall = np.asarray([self.ZFIX])
                self.has_ZFIX = True
            elif np.abs(self.Zmax - self.Zmin) <= self.delZ:
                self.ZFIX = self.Zmin
                self.Zall = np.asarray([self.ZFIX])
                self.has_ZFIX = True
            else:
                self.Zall = np.arange(self.Zmin, self.Zmax, self.delZ)

        # If BPASS;
        if self.f_bpass == 1:
            try:
                self.BPASS_DIR = inputs['BPASS_DIR']
            except:
                print('BPASS_DIR is not found. Using default.')
                self.BPASS_DIR = '/astro/udfcen3/Takahiro/BPASS/'

            self.BPASS_ver = 'v2.2.1'
            self.Zsun = 0.020
            Zbpass = [1e-5, 1e-4, 0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.010, 0.020, 0.030, 0.040]
            Zbpass = np.log10(np.asarray(Zbpass)/self.Zsun)

            try: # If ZFIX is found;
                iiz = np.argmin(np.abs(Zbpass[:] - float(inputs['ZFIX']) ) )
                if Zbpass[iiz] - float(inputs['ZFIX']) != 0:
                    print('%.2f is not found in BPASS Z list. %.2f is used instead.'%(float(inputs['ZFIX']),Zbpass[iiz]))
                self.ZFIX = Zbpass[iiz]
                self.delZ = 0.0001
                self.Zmin, self.Zmax = self.ZFIX, self.ZFIX + self.delZ
                self.Zall = np.arange(self.Zmin, self.Zmax, self.delZ) # in logZsun
                print('\n##########################')
                print('ZFIX is found.\nZ will be fixed to: %.2f'%(self.ZFIX))
                self.has_ZFIX = True
            except:
                print('ZFIX is not found.')
                print('Metallicities available in BPASS are limited and discrete. ZFIX is recommended.',self.Zall)
                self.Zmax, self.Zmin = float(inputs['ZMAX']), float(inputs['ZMIN'])
                con_z = np.where((Zbpass >= self.Zmin) & (Zbpass <= self.Zmax))
                self.Zall = Zbpass[con_z]
                self.delZ = 0.0001
                self.Zmax,self.Zmin = np.max(self.Zall), np.min(self.Zall)
                print('Final list for log(Z_BPASS/Zsun) is:',self.Zall)
                if len(self.Zall)>1:
                    self.has_ZFIX = False
                    self.ZFIX = None

        # N of param:
        self.has_AVFIX = False
        try:
            Avfix = float(inputs['AVFIX'])
            self.AVFIX = Avfix
            self.nAV = 0
            print('\n##########################')
            print('AVFIX is found.\nAv will be fixed to:\n %.2f'%(Avfix))
            self.has_AVFIX = True
        except:
            try:
                self.Avmin = float(inputs['AVMIN'])
                self.Avmax = float(inputs['AVMAX'])
                if Avmin == Avmax:
                    self.nAV = 0
                    self.AVFIX = Avmin
                    self.has_AVFIX = True
                else:
                    self.nAV = 1
            except:
                self.nAV = 1
                self.Avmin = 0
                self.Avmax = 4.0

        # Z evolution;
        print('\n#############################')
        if self.SFH_FORM == -99:
            if int(inputs['ZEVOL']) == 1:
                self.ZEVOL = 1
                self.ndim = int(self.npeak * 2 + self.nAV) # age, Z, and Av.
                print('Metallicity evolution is on')
            else:
                self.ZEVOL = 0
                print('Metallicity evolution is off')
                try:
                    ZFIX = float(inputs['ZFIX'])
                    self.nZ = 0
                except:
                    if np.max(self.Zall) == np.min(self.Zall):
                        self.nZ = 0
                    else:
                        self.nZ = 1
                self.ndim = int(self.npeak + self.nZ + self.nAV) # age, Z, and Av.
        else:
            if int(inputs['ZEVOL']) == 1:
                self.ZEVOL = 1
                self.nZ = self.npeak
                print('Metallicity evolution is on')
            else:
                self.ZEVOL = 0
                print('Metallicity evolution is off')
                try:
                    ZFIX = float(inputs['ZFIX'])
                    self.nZ = 0
                except:
                    if np.max(self.Zall) == np.min(self.Zall):
                        self.nZ = 0
                    else:
                        self.nZ = 1
                    
            self.ndim = int(self.npeak*3 + self.nZ + self.nAV) # age, Z, and Av.
        print('#############################\n')

        # Redshift
        self.ndim += self.fzmc
        print('\n##########################')
        print('No. of params are : %d'%(self.ndim))

        # Line
        try:
            self.LW0 = [float(x.strip()) for x in inputs['LINE'].split(',')]
        except:
            self.LW0 = []

        # Dust model specification;
        try:
            self.dust_model = int(inputs['DUST_MODEL'])
            if self.dust_model == 0:
                self.dust_model_name = 'Calz'
            elif self.dust_model == 1:
                self.dust_model_name = 'MW'
            elif self.dust_model == 2:
                self.dust_model_name = 'LMC'
            elif self.dust_model == 3:
                self.dust_model_name = 'SMC'
            elif self.dust_model == 4:
                self.dust_model_name = 'KriekConroy'
            else:
                print('Unknown number for dust attenuation. Calzetti.')
                self.dust_model = 0
                self.dust_model_name = 'Calz'
        except:
            self.dust_model = 0
            self.dust_model_name = 'Calz'
        print('Dust attenuation is set to %s\n'%self.dust_model_name)

        # If FIR data;
        try:
            DT0 = float(inputs['TDUST_LOW'])
            DT1 = float(inputs['TDUST_HIG'])
            dDT = float(inputs['TDUST_DEL'])
            try:
                self.TDUSTFIX = float(inputs['TDUSTFIX'])
                if self.TDUSTFIX < DT0 or self.TDUSTFIX > DT1:
                    print('TDUSTFIX is set out of the range. Exiting.')
                    sys.exit()
            except:
                self.TDUSTFIX = None

            if DT0 == DT1:
                self.Temp = [DT0]
            else:
                self.Temp = np.arange(DT0,DT1,dDT)

            if not self.TDUSTFIX == None:
                self.NTDUST = np.argmin(np.abs(self.Temp-self.TDUSTFIX))
            else:
                self.NTDUST = None

            self.f_dust = True

            self.DT0 = DT0
            self.DT1 = DT1
            self.dDT = dDT
            try:
                self.dust_numax = int(inputs['DUST_NUMAX'])
                self.dust_nmodel = int(inputs['DUST_NMODEL'])
            except:
                self.dust_numax = 3
                self.dust_nmodel = 9
            print('FIR fit is on.')
        except:
            self.Temp = []
            self.f_dust = False
            pass

        try:
            self.DIR_DUST = inputs['DIR_DUST']
        except:
            self.DIR_DUST = './'

        try:
            # Length of each ssp templates.
            self.tau0 = np.asarray([float(x.strip()) for x in inputs['TAU0'].split(',')])
            self.ntau0 = len(self.tau0)
        except:
            self.tau0 = np.asarray([-1.0])
            self.ntau0 = 1

        # IMF
        try:
            self.nimf = int(inputs['NIMF'])
        except:
            self.nimf = 0
            print('Cannot find NIMF. Set to %d.'%(self.nimf))

        # Error parameter
        try:
            self.ferr = int(self.inputs['F_ERR'])
        except:
            self.ferr = 0
            pass

        # Nested or ensemble slice sampler?
        self.f_mcmc = False
        self.f_nested = False
        self.f_zeus = False
        try:
            nnested = inputs['MC_SAMP']
            if nnested == 'NEST' or nnested == '1':
                self.f_nested = True
            elif nnested == 'ZEUS' or  nnested == 'SLICE' or nnested == '2':
                self.f_zeus = True
            else:
                self.f_mcmc = True
        except:
            print('Missing MC_SAMP keyword. Setting f_mcmc True.')
            self.f_mcmc = True
            pass

        # Force Age to Age fix?:
        try:
            if int(inputs['FORCE_AGE'])==1:
                self.force_agefix = True
            else:
                self.force_agefix = False
        except:
            self.force_agefix = False

        # SFH prior;
        # try:
        #     self.norder_sfh_prior = int(inputs['norder_sfh_prior'])
        #     self.f_prior_sfh = True
        # except:
        #     self.norder_sfh_prior = None
        #     self.f_prior_sfh = False

        print('\n')


    def get_lines(self, LW0):
        '''
        '''
        fLW = np.zeros(len(LW0), dtype='int')
        LW = LW0
        return LW, fLW


    def read_data(self, Cz0:float, Cz1:float, Cz2:float, zgal:float, add_fir:bool=False, idman=None):
        '''
        Read in observed data. Not model fluxes.

        Parameters
        ----------
        Cz0, Cz1 : float
            Normalization coefficients for grism spectra. Cz0 for G102, Cz1 for G141.
        zgal : float
            Current redshift estimate.

        Returns
        -------
        Dictionary.

        Notes
        -----
        This modeule can be used for any SFHs.
        '''
        print('Reading data with Cz0=%.2f, Cz1=%.2f, Cz2=%.2f, zgal=%.2f'%(Cz0, Cz1, Cz2, zgal))

        ##############
        # Spectrum
        ##############
        NR, x, fy00, ey00 = self.data['spec_obs']['NR'], self.data['spec_obs']['x'], self.data['spec_obs']['fy'], self.data['spec_obs']['ey']
        data_len = self.data['meta']['data_len']

        Cs = [Cz0, Cz1, Cz2]
        xx02 = []
        fy02 = []
        ey02 = []
        for ii in range(len(data_len)):
            if ii == 0:
                con0 = (NR<data_len[ii])
            else:
                con0 = (NR>=np.sum(data_len[:ii])) & (NR<np.sum(data_len[:ii+1]))
            xx02 = np.append(xx02, x[con0])
            fy02 = np.append(fy02, fy00[con0] * Cs[ii])
            ey02 = np.append(ey02, ey00[con0] * Cs[ii])

        ##############
        # Broadband
        ##############
        try:
            NRbb, xbb, fybb, eybb, exbb = self.data['bb_obs']['NR'], self.data['bb_obs']['x'], self.data['bb_obs']['fy'], self.data['bb_obs']['ey'], self.data['bb_obs']['ex']
        except: # if no BB;
            print('No BB data.')
            NRbb = np.asarray([])
            xbb = np.asarray([])
            fybb = np.asarray([])
            eybb = np.asarray([])
            exbb = np.asarray([])

        con_bb = ()
        xx_bb = xbb[con_bb]
        ex_bb = exbb[con_bb]
        fy_bb = fybb[con_bb]
        ey_bb = eybb[con_bb]

        xx = np.append(xx02,xx_bb)
        fy = np.append(fy02,fy_bb)
        ey = np.append(ey02,ey_bb)

        wht = 1./np.square(ey)
        con_wht = (ey<0)
        wht[con_wht] = 0

        # For now...
        #wht2 = check_line_man(fy, x, wht, fy, zgal, self.LW0)
        wht2 = wht[:]

        # Append data;
        if add_fir:
            nr_d, x_d, fy_d, ey_d = self.data['spec_dust_obs']['NR'], self.data['spec_dust_obs']['x'], self.data['spec_dust_obs']['fy'], self.data['spec_dust_obs']['ey']
            NR = np.append(NR,nr_d)
            fy = np.append(fy,fy_d)
            ey = np.append(ey,ey_d)
            x = np.append(x,x_d)
            wht = np.append(wht,1./np.square(ey_d))
            # For now...
            #wht2= check_line_man(fy, x, wht, fy, zgal, self.LW0)
            wht2 = wht[:]

        # Sort data along wave;
        f_sort = False

        if f_sort:
            nrd_yyd = np.zeros((len(NR),6), dtype='float')
            nrd_yyd[:,0] = NR
            nrd_yyd[:,1] = x
            nrd_yyd[:,2] = fy
            nrd_yyd[:,3] = ey
            nrd_yyd[:,4] = wht
            nrd_yyd[:,5] = wht2

            b = nrd_yyd
            nrd_yyd_sort = b[np.lexsort(([-1,1]*b[:,[1,0]]).T)]
            NR = nrd_yyd_sort[:,0]
            x = nrd_yyd_sort[:,1]
            fy = nrd_yyd_sort[:,2]
            ey = nrd_yyd_sort[:,3]
            wht = nrd_yyd_sort[:,4]
            wht2= nrd_yyd_sort[:,5]

        sn = fy/ey
        self.n_optir = len(sn)

        dict = {}
        dict = {'NR':NR, 'x':x, 'fy':fy, 'ey':ey, 'NRbb':NRbb, 'xbb':xx_bb, 'exbb':ex_bb, 'fybb':fy_bb, 'eybb':ey_bb, 'wht':wht, 'wht2': wht2, 'sn':sn}

        return dict


    def fit_redshift(self, xm_tmp, fm_tmp, delzz=0.01, ezmin=0.01, zliml=0.01, 
        zlimu=None, snlim=0, priors=None, f_bb_zfit=True, f_line_check=False, 
        f_norm=True, f_lambda=False, zmax=20, include_bb=False, f_exclude_negative=False):
        '''
        Find the best-fit redshift, before going into a big fit, through an interactive inspection.
        This module is effective only when spec data is provided.
        When spectrun is provided, this does redshift fit, but **by using the SED model guessed from the BB photometry**.
        Thus, providing good BB photometry is critical in this step.

        Parameters
        ----------
        xm_tmp, fm_tmp : float array
            SED model.
        delzz : float
            Delta z in redshift search space
        zliml : float
            Lower limit range for redshift
        zlimu : float
            Upper limit range for redshift
        ezmin : float
            Minimum redshift uncertainty.
        snlim : float
            SN limit for data points. Those below the number will be cut from the fit.
        include_bb : bool
            Turn this True if Redshift fitting is requred for only BB data.
        f_line_check : bool
            If True, line masking.
        priors : dict, optional
            Dictionary that contains z (redshift grid) and chi2 (chi-square).
        f_exclude_negative : bool
            Exclude negative fluxes in spectrum.

        Notes
        -----
        Spectral data must be provided to make this work.

        '''
        self.file_zprob = self.DIR_OUT + 'zprob_' + self.ID + '.txt'

        # NMC for zfit
        self.nmc_cz = int(self.inputs['NMCZ'])

        # For z prior.
        if self.zmcmin != None:
            zliml = self.zmcmin
        else:
            zliml = 0
        if self.zmcmax != None:
            zlimu = self.zmcmax
        else:
            zlimu = 20

        # Observed data;
        sn = self.dict['fy'] / self.dict['ey']

        # Only spec data?
        if include_bb:
            con_cz = ()#(sn>snlim)
        else:
            con_cz = (self.dict['NR']<self.NRbb_lim) #& (sn>snlim)

        if f_exclude_negative:
            con_cz &= (sn>snlim)
            
        fy_cz = self.dict['fy'][con_cz] # Already scaled by self.Cz0
        ey_cz = self.dict['ey'][con_cz]
        x_cz = self.dict['x'][con_cz] # Observed range
        NR_cz = self.dict['NR'][con_cz]

        # kind='cubic' causes an error if len(xm_tmp)<=3;
        fint = interpolate.interp1d(xm_tmp, fm_tmp, kind='nearest', fill_value="extrapolate")
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
            zz_prob = np.arange(0,zmax,delzz)
            cprob_s = np.interp(zz_prob, zprob, cprob)
            prior_s = np.exp(-0.5 * cprob_s)
            prior_s /= np.sum(prior_s)
        else:
            zz_prob = np.arange(zliml,zlimu,delzz)
            if priors != None:
                zprob = priors['z']
                cprob = priors['chi2']

                cprob_s = np.interp(zz_prob, zprob, cprob)
                prior_s = np.exp(-0.5 * cprob_s) / np.sum(cprob_s)
                con_pri = (zz_prob<np.min(zprob)) | (zz_prob>np.max(zprob))
                prior_s[con_pri] = 0
                if f_norm:
                    prior_s /= np.sum(prior_s)
            else:
                prior_s = zz_prob * 0 + 1.
                prior_s /= np.sum(prior_s)

        # Attach prior:
        self.z_prior = zz_prob
        self.p_prior = prior_s

        # Plot;
        if len(self.dict['fy'][con_cz])==0:
            return 'y'

        if self.fzvis==1 and (include_bb | len(fy_cz)>0):
            import matplotlib as mpl
            mpl.use('TkAgg')
            data_model = np.zeros((len(x_cz),4),'float')
            data_model[:,0] = x_cz
            data_model[:,1] = fm_s
            data_model[:,2] = fy_cz
            data_model[:,3] = ey_cz

            data_model_sort = data_model[data_model[:, 0].argsort()]            
            #data_model_sort = np.sort(data_model, axis=0) # This does not work!!

            # Model based on input z.
            plt.plot(data_model_sort[:,0], data_model_sort[:,1], 'gray', linestyle='--', linewidth=0.5, label='')
            # Observation
            # spec;
            con = (self.dict['ey']<1000) & (self.dict['NR']<self.NRbb_lim)
            plt.plot(self.dict['x'][con], self.dict['fy'][con], '.b', linestyle='-', linewidth=0.5, label='Obs.')
            plt.errorbar(self.dict['x'][con], self.dict['fy'][con], yerr=self.dict['ey'][con], color='b', capsize=0, linewidth=0.5)
            # bb;
            con = (self.dict['NR']>=self.NRbb_lim)
            plt.errorbar(self.dict['x'][con], self.dict['fy'][con], yerr=self.dict['ey'][con], ms=5, marker='o', 
                color='orange', capsize=0, linewidth=0.5, ls='None')

            # Write prob distribution;
            if True:
                get_chi2(zz_prob, fy_cz, ey_cz, x_cz, fm_tmp, xm_tmp/(1+self.zgal), self.file_zprob)

            print('############################')
            print('Start MCMC for redshift fit')
            print('############################')
            res_cz, fitc_cz = check_redshift(
                fy_cz, ey_cz, x_cz, fm_tmp, xm_tmp/(1+self.zgal), 
                self.zgal, self.z_prior, self.p_prior,
                NR_cz, self.data['meta']['data_len'], zliml, zlimu, self.nmc_cz, self.nwalk_cz, 
                include_photometry=include_bb
                )

            z_cz = np.percentile(res_cz.flatchain['z'], [16,50,84])
            scl_cz0 = np.percentile(res_cz.flatchain['Cz0'], [16,50,84])
            scl_cz1 = np.percentile(res_cz.flatchain['Cz1'], [16,50,84])
            scl_cz2 = np.percentile(res_cz.flatchain['Cz2'], [16,50,84])

            zrecom = z_cz[1]
            Czrec0 = scl_cz0[1]
            Czrec1 = scl_cz1[1]
            Czrec2 = scl_cz2[1]

            # Switch to peak redshift:
            # find minimum and maximum of xticks, so we know
            # where we should compute theoretical distribution
            ser = res_cz.flatchain['z']
            xmin, xmax = self.zgal-0.2, self.zgal+0.2
            lnspc = np.linspace(xmin, xmax, len(ser))
            print('\n\n')
            print('Recommended redshift, Cz0, Cz1, and Cz2, %.5f %.5f %.5f %.5f, with chi2/nu=%.3f'%(zrecom, Czrec0, Czrec1, Czrec2, fitc_cz[1]))
            print('\n\n')
            fit_label = 'Proposed model'

        else:
            print('fzvis is set to False. z fit not happening.')
            try:
                zmcmin = float(self.inputs['ZMCMIN'])
                zmcmax = float(self.inputs['ZMCMAX'])
                print('Redshift error is taken from input file.')
            except:
                zmcmin = self.zprev-ezmin
                zmcmax = self.zprev+ezmin
                print('Redshift error is assumed to %.1f.'%(ezmin))

            z_cz = [zmcmin, self.zprev, zmcmax]
            zrecom = z_cz[1]
            scl_cz0 = [1.,1.,1.]
            scl_cz1 = [1.,1.,1.]
            scl_cz2 = [1.,1.,1.]
            Czrec0 = scl_cz0[1]
            Czrec1 = scl_cz1[1]
            Czrec2 = scl_cz2[1]
            res_cz = None
            fitc_cz = [99,99]

            # If this label is being used, it means that the fit is failed.
            fit_label = 'Current model'

        # New template at zrecom;
        xm_s = xm_tmp / (1+self.zgal) * (1+zrecom)
        fint = interpolate.interp1d(xm_s, fm_tmp, kind='nearest', fill_value="extrapolate")
        fm_s = fint(x_cz)
        whtl = 1/np.square(ey_cz)

        if f_line_check:
            try:
                wht3, ypoly = check_line_cz_man(fy_cz, x_cz, whtl, fm_s, zrecom, LW=self.LW0)
            except:
                wht3, ypoly = whtl, fy_cz
        else:
            wht3 = whtl
        con_line = (wht3==0)

        # Visual inspection;
        if self.fzvis==1:
            #
            # Ask interactively;
            #
            data_model_new = np.zeros((len(x_cz),4),'float')
            data_model_new[:,0] = x_cz
            data_model_new[:,1] = fm_s
            data_model_new_sort = data_model_new[data_model_new[:, 0].argsort()]

            plt.plot(data_model_new_sort[:,0], data_model_new_sort[:,1], 'r', linestyle='-', linewidth=0.5, label='%s ($z=%.5f$)'%(fit_label,zrecom)) # Model based on recomended z.
            plt.plot(x_cz[con_line], fm_s[con_line], color='orange', marker='o', linestyle='', linewidth=3.)

            # Plot lines for reference
            for ll in range(len(LW)):
                try:
                    conpoly = (x_cz/(1.+zrecom)>3000) & (x_cz/(1.+zrecom)<8000)
                    yline = np.max(ypoly[conpoly])
                    yy = np.arange(yline/1.02, yline*1.1)
                    xxpre = yy * 0 + LW[ll] * (1.+self.zgal)
                    xx = yy * 0 + LW[ll] * (1.+zrecom)
                    plt.plot(xxpre, yy/1.02, linewidth=0.5, linestyle='--', color='gray')
                    plt.text(LW[ll] * (1.+self.zgal), yline/1.05, '%s'%(LN[ll]), fontsize=8, color='gray')
                    plt.plot(xx, yy, linewidth=0.5, linestyle='-', color='orangered')
                    plt.text(LW[ll] * (1.+zrecom), yline, '%s'%(LN[ll]), fontsize=8, color='orangered')
                except:
                    pass

            # Plot data
            data_obsbb = np.zeros((len(self.dict['xbb']),3), 'float')
            data_obsbb[:,0],data_obsbb[:,1] = self.dict['xbb'],self.dict['fybb']
            if len(fm_tmp) == len(self.dict['xbb']): # BB only;
                data_obsbb[:,2] = fm_tmp
            data_obsbb_sort = data_obsbb[data_obsbb[:, 0].argsort()]            
            
            if len(fm_tmp) == len(self.dict['xbb']): # BB only;
                plt.scatter(data_obsbb_sort[:,0], data_obsbb_sort[:,2], color='none', marker='d', s=50, edgecolor='gray', zorder=4, label='Current model ($z=%.5f$)'%(self.zgal))
            else:
                model_spec = np.zeros((len(fm_tmp),2), 'float')
                model_spec[:,0],model_spec[:,1] = xm_tmp,fm_tmp
                model_spec_sort = model_spec[model_spec[:, 0].argsort()]
                plt.plot(model_spec_sort[:,0], model_spec_sort[:,1], marker='.', color='gray', ms=1, linestyle='-', linewidth=0.5, zorder=4, label='Current model ($z=%.5f$)'%(self.zgal))

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

            zzsigma = ((z_cz[2] - z_cz[0])/2.)/self.zgal
            zsigma = np.abs(self.zgal-zrecom) / (self.zgal)
            C0sigma = np.abs(Czrec0-self.Cz0)/self.Cz0
            eC0sigma = ((scl_cz0[2]-scl_cz0[0])/2.)/self.Cz0
            C1sigma = np.abs(Czrec1-self.Cz1)/self.Cz1
            eC1sigma = ((scl_cz1[2]-scl_cz1[0])/2.)/self.Cz1
            C2sigma = np.abs(Czrec2-self.Cz2)/self.Cz2
            eC2sigma = ((scl_cz2[2]-scl_cz2[0])/2.)/self.Cz2

            print('\n##############################################################')
            print('Input redshift is %.3f per cent agreement.'%((1.-zsigma)*100))
            print('Error is %.3f per cent.'%(zzsigma*100))
            print('Input Cz0 is %.3f per cent agreement.'%((1.-C0sigma)*100))
            print('Error is %.3f per cent.'%(eC0sigma*100))
            print('Input Cz1 is %.3f per cent agreement.'%((1.-C1sigma)*100))
            print('Error is %.3f per cent.'%(eC1sigma*100))
            print('Input Cz2 is %.3f per cent agreement.'%((1.-C2sigma)*100))
            print('Error is %.3f per cent.'%(eC2sigma*100))
            print('##############################################################\n')
            plt.show()

            flag_z = raw_input('Do you want to continue with the input redshift, Cz0, Cz1, Cz2, and chi2/nu, %.5f %.5f %.5f %.5f %.5f? ([y]/n/m) '%\
                (self.zgal, self.Cz0, self.Cz1, self.Cz2, self.fitc_cz_prev))
        else:
            flag_z = 'y'

        try:
            self.z_cz_prev = self.z_cz
        except:
            self.z_cz_prev = [self.zgal,self.zgal,self.zgal]

        # Write it to self;
        self.zrecom = zrecom
        self.Czrec0 = Czrec0 * self.Cz0
        self.Czrec1 = Czrec1 * self.Cz1
        self.Czrec2 = Czrec2 * self.Cz2
        self.z_cz = z_cz
        self.scl_cz0 = scl_cz0
        self.scl_cz1 = scl_cz1
        self.scl_cz2 = scl_cz2
        self.res_cz = res_cz
        self.fitc_cz_prev = fitc_cz[1]

        return flag_z


    def get_zdist(self, f_interact=False, f_ascii=True):
        '''
        Saves a plot of z-distribution.

        Parameters
        ----------
        f_interact : bool
            If true, this module returns figure ax.
        '''

        try:
            fig = plt.figure(figsize=(6.5,2.5))
            fig.subplots_adjust(top=0.96, bottom=0.16, left=0.09, right=0.99, hspace=0.15, wspace=0.25)
            ax1 = fig.add_subplot(111)
            n, nbins, patches = ax1.hist(self.res_cz.flatchain['z'], bins=200, density=True, color='gray', label='')

            yy = np.arange(0,np.max(n),1)
            xx = yy * 0 + self.z_cz[1]
            ax1.plot(xx,yy,linestyle='-',linewidth=1,color='orangered',\
                label='$z=%.5f_{-%.5f}^{+%.5f}$\n$C_{z0}=%.3f$\n$C_{z1}=%.3f$\n$C_{z2}=%.3f$'%\
                (self.z_cz[1],self.z_cz[1]-self.z_cz[0],self.z_cz[2]-self.z_cz[1], self.Cz0, self.Cz1, self.Cz2))

            if f_ascii:
                file_ascii_out = self.DIR_OUT + 'zmc_' + self.ID + '.txt'
                fw_ascii = open(file_ascii_out,'w')
                fw_ascii.write('# z pz\n')
                for ii in range(len(n)):
                    fw_ascii.write('%.3f %.3f\n'%(nbins[ii]+(nbins[ii+1]-nbins[ii])/2.,n[ii]))
                fw_ascii.close()

            xx = yy * 0 + self.z_cz[0]
            ax1.plot(xx,yy,linestyle='--',linewidth=1,color='orangered')
            xx = yy * 0 + self.z_cz[2]
            ax1.plot(xx,yy,linestyle='--',linewidth=1,color='orangered')
            xx = yy * 0 + self.zgal
            ax1.plot(xx,yy,linestyle='-',linewidth=1,color='royalblue', label='Input redshift')

            # Prior:
            ax1.plot(self.z_prior, self.p_prior * np.max(yy)/np.max(self.p_prior), linestyle='--', linewidth=1, color='cyan', label='Prior')

            # Label:
            ax1.set_xlabel('Redshift')
            ax1.set_ylabel('$dn/dz$')
            zp_min,zp_max = self.z_cz[0] - (self.z_cz[1]-self.z_cz[0])*3, self.z_cz[2] + (self.z_cz[2]-self.z_cz[1])*3
            ax1.set_xlim(zp_min,zp_max)
            ax1.legend(loc=0)
            
            # Save:
            file_out = self.DIR_OUT + 'zmc_' + self.ID + '.png'
            print('Figure is saved in %s'%file_out)

            if f_interact:
                fig.savefig(file_out, dpi=300)
                # return fig, ax1
            else:
                plt.savefig(file_out, dpi=300)
                plt.close()
                # return True
        except:
            print('z-distribution figure is not generated.')
            pass


        if os.path.exists(self.file_zprob):
            # Also, make zprob;
            fig = plt.figure(figsize=(6.5,2.5))
            fig.subplots_adjust(top=0.96, bottom=0.16, left=0.09, right=0.99, hspace=0.15, wspace=0.25)
            ax1 = fig.add_subplot(111)

            fd = ascii.read(self.file_zprob)
            z = fd['z']
            pz = fd['p(z)']
            pz /= pz.max()

            # prob:
            ax1.plot(z, pz, linestyle='-', linewidth=1, color='r', label='')

            # Label:
            ax1.set_xlabel('Redshift')
            ax1.set_ylabel('$p(z)$')
            file_out = self.DIR_OUT + 'zprob_' + self.ID + '.png'
            if f_interact:
                fig.savefig(file_out, dpi=300)
            else:
                plt.savefig(file_out, dpi=300)
                plt.close()


    def add_param(self, fit_params, sigz=1.0, zmin=None, zmax=None):
        '''
        Add new parameters.

        '''
        f_add = False

        # Redshift
        if self.fzmc == 1:
            if zmin == None:
                self.zmcmin = 0
            if zmax == None:
                self.zmcmax = 15
            fit_params.add('zmc', value=self.zgal, min=self.zmcmin, max=self.zmcmax)
            print('Redshift is set as a free parameter (z in [%.2f:%.2f])'%(self.zmcmin, self.zmcmax))
            f_add = True

        # Dust;
        if self.f_dust:
            Tdust = self.Temp
            if not self.TDUSTFIX == None:
                fit_params.add('TDUST', value=self.NTDUST, vary=False)
            elif len(Tdust)-1>0:
                fit_params.add('TDUST', value=len(Tdust)/2., min=0, max=len(Tdust)-1)
                self.ndim += 1
            else:
                fit_params.add('TDUST', value=0, vary=False)

            fit_params.add('MDUST', value=9, min=0, max=15)
            self.ndim += 1
            self.dict = self.read_data(self.Cz0, self.Cz1, self.Cz2, self.zgal, add_fir=self.f_dust)
            f_add = True

        # Nebular; ver1.6
        if self.fneb:
            fit_params.add('Aneb', value=self.Aini, min=self.Amin, max=self.Amax)
            self.ndim += 1
            if not self.logUFIX == None:
                fit_params.add('logU', value=self.logUFIX, vary=False)
            else:
                fit_params.add('logU', value=np.median(self.logUs), min=self.logUMIN, max=self.logUMAX)
                self.ndim += 1
            f_add = True

        self.fit_params = fit_params

        return f_add


    def set_param(self):
        '''
        Set parameters.
        '''
        print('##################')
        print('Setting parameters')
        print('##################\n')
        agemax = self.cosmo.age(self.zgal).value 
        fit_params = Parameters()
        f_Alog = True
        if f_Alog:
            try:
                self.Amin = float(self.inputs['AMIN'])
                self.Amax = float(self.inputs['AMAX'])
            except:
                self.Amin = -5
                self.Amax = 10
            self.Aini = -1
        else:
            self.Amin = 0
            self.Amax = 1e3
            self.Aini = 1

        if self.SFH_FORM==-99:
            self.age_vary = []
            if len(self.age) != len(self.aamin):
                for aa in range(len(self.age)):
                    if aa not in self.aamin:
                        fit_params.add('A'+str(aa), value=self.Amin, vary=False)
                        self.ndim -= 1                    
                        self.age_vary.append(False)
                    else:
                        fit_params.add('A'+str(aa), value=self.Aini, min=self.Amin, max=self.Amax)
                        self.age_vary.append(True)
            else:
                for aa in range(len(self.age)):
                    if self.age[aa] == 99:
                        fit_params.add('A'+str(aa), value=self.Amin, vary=False)
                        self.ndim -= 1
                        self.age_vary.append(False)
                    elif self.age[aa]>agemax and not self.force_agefix:
                        print('At this redshift, A%d is beyond the age of universe and not being used.'%(aa))
                        fit_params.add('A'+str(aa), value=self.Amin, vary=False)
                        self.ndim -= 1
                        self.age_vary.append(False)
                    else:
                        fit_params.add('A'+str(aa), value=self.Aini, min=self.Amin, max=self.Amax)
                        self.age_vary.append(True)

        else:
            for aa in range(self.npeak):
                tauini = (self.taumin+self.taumax)/2.
                ageini = (self.agemin + self.agemax)/2.
                fit_params.add('A%d'%aa, value=self.Aini, min=self.Amin, max=self.Amax)

                # Check AGE fix; TBD.
                try:
                    AGEFIX = float(self.inputs['AGEFIX'])
                    fit_params.add('AGE%d'%aa, value=AGEFIX, vary=False)
                    self.agemin = AGEFIX
                    self.agemax = AGEFIX
                    self.ndim -= 1
                    print('AGEFIX is found. Set to %.2f'%(AGEFIX))
                except:
                    tcosmo = np.log10(self.cosmo.age(self.zgal).value) 
                    agemax_tmp = self.agemax
                    if agemax_tmp > tcosmo:
                        agemax_tmp = tcosmo
                        print('Maximum age is set to the age of the univese (%.1fGyr) at this redshift.'%(self.cosmo.age(self.zgal).value))
                    if self.npeak>1:
                        if aa == 0:
                            fit_params.add('AGE%d'%aa, value=ageini, min=self.agemin, max=np.log10(1.0))
                        else:
                            ageini = np.log10(1.0)
                            fit_params.add('AGE%d'%aa, value=ageini, min=np.log10(1.0), max=agemax_tmp)
                    else:
                        #fit_params.add('AGE%d'%aa, value=0.0, min=0.0, max=0.01)
                        fit_params.add('AGE%d'%aa, value=ageini, min=self.agemin, max=agemax_tmp)

                # Check Tau fix;
                if self.npeak>1:
                    if aa == 0:
                        fit_params.add('TAU%d'%aa, value=tauini, min=self.taumin, max=self.taumax)
                    else:
                        tauini = np.log10(0.3)
                        fit_params.add('TAU%d'%aa, value=tauini, min=self.taumin, max=np.log10(0.3))
                else:
                    #fit_params.add('TAU%d'%aa, value=-0.8, min=-0.8, max=-0.79)
                    fit_params.add('TAU%d'%aa, value=tauini, min=self.taumin, max=self.taumax)

        #
        # Dust attenuation;
        #
        try:
            Avfix = float(self.inputs['AVFIX'])
            fit_params.add('Av', value=Avfix, vary=False)
            self.Avmin = Avfix
            self.Avmax = Avfix
        except:
            try:
                self.Avmin = float(self.inputs['AVMIN'])
                self.Avmax = float(self.inputs['AVMAX'])
                self.Avini = (self.Avmax+self.Avmin)/2.
                self.Avini = 0.
                if self.Avmin == self.Avmax:
                    fit_params.add('Av', value=self.Avini, vary=False)
                    self.Avmin = self.Avini
                    self.Avmax = self.Avini
                else:
                    fit_params.add('Av', value=self.Avini, min=self.Avmin, max=self.Avmax)
            except:
                self.Avmin = 0.
                self.Avmax = 4.
                self.Avini = 0.5 #(Avmax-Avmin)/2. 
                print('Dust is set in [%.1f:%.1f]/mag. Initial value is set to %.1f'%(self.Avmin,self.Avmax,self.Avini))
                fit_params.add('Av', value=self.Avini, min=self.Avmin, max=self.Avmax)

        #
        # Metallicity;
        #
        if int(self.inputs['ZEVOL']) == 1:
            for aa in range(len(self.age)):
                if self.age[aa] == 99 or self.age[aa]>agemax:
                    fit_params.add('Z'+str(aa), value=0, min=0, max=1e-10)
                else:
                    fit_params.add('Z'+str(aa), value=0, min=self.Zmin, max=self.Zmax)
        else:
            if self.has_ZFIX:
                aa = 0
                fit_params.add('Z'+str(aa), value=self.ZFIX, vary=False)
            else:
                aa = 0
                if np.min(self.Zall)==np.max(self.Zall):
                    fit_params.add('Z'+str(aa), value=np.min(self.Zall), vary=False)
                else:
                    fit_params.add('Z'+str(aa), value=0, min=self.Zmin, max=self.Zmax)

       # Error parameter
        try:
            ferr = self.ferr
            if ferr == 1:
                fit_params.add('logf', value=0, min=-10, max=3) # in linear
                self.ndim += 1
                f_add = True
        except:
            ferr = 0
            pass

        self.fit_params = fit_params
        return fit_params


    def prepare_class(self, add_fir=None):
        '''
        '''
        print('#################')
        print('Preparing library')
        print('#################\n')
        # Load Spectral library;
        self.lib = self.fnc.open_spec_fits(fall=0)
        self.lib_all = self.fnc.open_spec_fits(fall=1, orig=True)

        if self.f_dust:
            self.lib_dust = self.fnc.open_spec_dust_fits(fall=0)
            self.lib_dust_all = self.fnc.open_spec_dust_fits(fall=1)
        if self.fneb:
            self.lib_neb = self.fnc.open_spec_fits(fall=0, f_neb=True)
            self.lib_neb_all = self.fnc.open_spec_fits(fall=1, f_neb=True)

        if add_fir == None:
            add_fir = self.f_dust

        # For MCMC;
        self.nmc = int(self.inputs['NMC'])
        self.nwalk = int(self.inputs['NWALK'])
        self.nmc_cz = int(self.inputs['NMCZ'])
        self.nwalk_cz = int(self.inputs['NWALKZ'])
        self.ZEVOL = int(self.inputs['ZEVOL'])
        self.fzvis = int(self.inputs['ZVIS'])
        if self.f_nested:
            print('Nested sample is on. Nelder is used for time saving analysis.')
            self.fneld = 1 

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
            dust_model = int(self.inputs['DUST_MODEL'])
        except:
            dust_model = 0

        # Error parameter
        try:
            self.ferr = int(self.inputs['F_ERR'])
        except:
            self.ferr = 0
            pass

        #
        # Observed Data;
        #
        self.dict = self.read_data(self.Cz0, self.Cz1, self.Cz2, self.zgal, add_fir=add_fir)

        # Set parameters;
        self.set_param()
    
        return True


    def get_shuffle(self, out, nshuf=3.0, amp=1e-4):
        '''
        Shuffles initial parameters of each walker, to give it extra randomeness.
        '''
        pos = np.zeros((self.nwalk, self.ndim), 'float')
        for ii in range(pos.shape[0]):
            aa = 0
            for aatmp,key in enumerate(out.params.valuesdict()):
                if out.params[key].vary == True:
                    pos[ii,aa] += out.params[key].value
                    # This is critical to avoid parameters fall on the boundary.
                    delpar = (out.params[key].max-out.params[key].min)/1000
                    # or not,
                    delpar = 0
                    if np.random.uniform(0,1) > (1. - 1./self.ndim):        
                        pos[ii,aa] = np.random.uniform(out.params[key].min+delpar, out.params[key].max-delpar)
                    else:
                        if pos[ii,aa]<out.params[key].min+delpar or pos[ii,aa]>out.params[key].max-delpar:
                            pos[ii,aa] = np.random.uniform(out.params[key].min+delpar, out.params[key].max-delpar)

                    aa += 1
        return pos


    def main(self, cornerplot:bool=True, specplot=1, sigz=1.0, ezmin=0.01, ferr=0,
            f_move:bool=False, verbose:bool=False, skip_fitz:bool=False, out=None, f_plot_accept:bool=True,
            f_shuffle:bool=True, amp_shuffle=1e-2, check_converge:bool=True, Zini=None, f_plot_chain:bool=True,
            f_chind:bool=True, ncpu:int=0, f_prior_sfh:bool=False, norder_sfh_prior:int=3, include_bb=True):
        '''
        Main module of this script.

        Parameters
        ----------
        ferr : float
            Used for error parameter.
        skip_fitz : bool 
            Skip redshift fit.
        sigz : float 
            Confidence interval for redshift fit (i.e. n-sigma).
        ezmin : float 
            minimum error in redshift
        f_plot_accept : bool
            Output acceptance plot of mcmc chains.
        f_shuffle : bool
            Randomly shuffle some of initial parameters in walkers.
        check_converge : bool
            Check convergence at every certain number.
        f_plot_chain : book
            Plot MC sample chain.
        '''
        import emcee
        import zeus
        try:
            import multiprocess
        except:
            import multiprocessing as multiprocess

        from .posterior_flexible import Post

        # Call likelihood/prior/posterior function;
        class_post = Post(self)

        # Prepare library, data, etc.
        self.prepare_class()

        print('########################')
        print('### Fitting Function ###')
        print('########################')
        start = timeit.default_timer()        
        if not os.path.exists(self.DIR_TMP):
            os.mkdir(self.DIR_TMP)

        # Initial Z:
        if Zini == None:
            Zini = self.Zall

        ####################################
        # Initial Metallicity Determination
        ####################################
        # Get initial parameters
        if not skip_fitz or out == None:
            
            # Do you want to prepare a template for redshift fit by using only spectrum?;
            f_only_spec = False
            out, chidef, Zbest = get_leastsq(self, Zini, self.fneld, self.age, self.fit_params, class_post.residual,\
            self.dict['fy'], self.dict['ey'], self.dict['wht2'], self.ID, f_only_spec=f_only_spec)

            # Best fit
            csq = out.chisqr
            rcsq = out.redchi
            fitc = [csq, rcsq] # Chi2, Reduced-chi2
            ZZ = Zbest # This is really important/does affect lnprob/residual.
            if self.fitc_cz_prev == None:
                self.fitc_cz_prev = rcsq

            print('\n\n')
            print('#####################################')
            print('Zbest, chi are;',Zbest,chidef)
            print('Params are;',fit_report(out))
            print('#####################################')
            print('\n\n')

            Av_tmp = out.params['Av'].value
            AA_tmp = np.zeros(len(self.age), dtype='float')
            ZZ_tmp = np.zeros(len(self.age), dtype='float')
            nrd_tmp, fm_tmp, xm_tmp = self.fnc.get_template(out, f_val=True, f_nrd=True, f_neb=False)
        else:
            csq = out.chisqr
            rcsq = out.redchi
            fitc = [csq, rcsq]

        ########################
        # Check redshift
        ########################
        if skip_fitz:
            flag_z = 'y'
        else:
            flag_z = self.fit_redshift(xm_tmp, fm_tmp, delzz=0.001, include_bb=include_bb)

        #################################################
        # Gor for mcmc phase
        #################################################
        if flag_z == 'y' or flag_z == '':

            #######################
            # Add parameters;
            #######################
            out_keep = out
            f_add = self.add_param(self.fit_params, sigz=self.sigz, zmin=self.zmcmin, zmax=self.zmcmax)

            # Then, minimize again.
            if f_add:
                if self.fneld == 1:
                    fit_name = 'nelder'
                elif self.fneld == 0:
                    fit_name = 'powell'
                elif self.fneld == 2:
                    fit_name = 'leastsq'
                else:
                    fit_name = self.fneld

                out = minimize(class_post.residual, self.fit_params, args=(self.dict['fy'], self.dict['ey'], self.dict['wht2'], self.f_dust), method=fit_name) 
                # showing this is confusing.
                # print('\nMinimizer refinement;')
                # print(fit_report(out))

                # Fix params to what we had before.
                if self.fzmc:
                    out.params['zmc'].value = self.zgal
                out.params['Av'].value = out_keep.params['Av'].value
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

            print('########################')
            print('### Starting sampling ##')
            print('########################\n')
            start_mc = timeit.default_timer()

            # MCMC;
            if self.f_mcmc or self.f_zeus:
                nburn = int(self.nmc/2)
                if f_shuffle:
                    # ZEUS may fail to run with f_shuffle.
                    pos = self.get_shuffle(out, amp=amp_shuffle)
                else:
                    pos = amp_shuffle * np.random.randn(self.nwalk, self.ndim)
                    aa = 0
                    for aatmp,key in enumerate(out.params.valuesdict()):
                        if out.params[key].vary:
                            pos[:,aa] += out.params[key].value
                            aa += 1

                if self.f_zeus:
                    print('sampling with ZEUS')
                    check_converge = False
                    f_burnin = True
                    if f_burnin:
                        # Burn phase;
                        moves = zeus.moves.DifferentialMove() #GlobalMove()
                        sampler = zeus.EnsembleSampler(self.nwalk, self.ndim, class_post.lnprob_emcee, \
                            args=[out.params, self.dict['fy'], self.dict['ey'], self.dict['wht2'], self.f_dust], \
                            moves=moves, maxiter=1e6,\
                            kwargs={'f_val':True, 'out':out, 'lnpreject':-np.inf, 'f_chind':f_chind, 
                            'f_prior_sfh':f_prior_sfh, 'norder_sfh_prior':norder_sfh_prior},\
                            )
                        # Run MCMC
                        nburn = int(self.nmc/10)

                        print('Running burn-in')
                        sampler.run_mcmc(pos, nburn)
                        print('Done burn-in')

                        # Get the burnin samples
                        burnin = sampler.get_chain()

                        # Set the new starting positions of walkers based on their last positions
                        pos = burnin[-1]

                    # Switch sampler;
                    moves = zeus.moves.GlobalMove()
                    sampler = zeus.EnsembleSampler(self.nwalk, self.ndim, class_post.lnprob_emcee, \
                        args=[out.params, self.dict['fy'], self.dict['ey'], self.dict['wht2'], self.f_dust], \
                        moves=moves, maxiter=1e4,\
                        kwargs={'f_val':True, 'out':out, 'lnpreject':-np.inf, 
                        'f_prior_sfh':f_prior_sfh, 'norder_sfh_prior':norder_sfh_prior},\
                        )

                else:
                    print('sampling with EMCEE')
                    moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2),]
                    sampler = emcee.EnsembleSampler(self.nwalk, self.ndim, class_post.lnprob_emcee, \
                        args=(out.params, self.dict['fy'], self.dict['ey'], self.dict['wht2'], self.f_dust),\
                        #moves=moves,\
                        kwargs={'f_val': True, 'out': out, 'lnpreject':-np.inf, 
                        'f_prior_sfh':f_prior_sfh, 'norder_sfh_prior':norder_sfh_prior},\
                        )

                if check_converge:
                    # Check convergence every number;
                    nevery = int(self.nmc/10)
                    nconverge = 10.
                    if nevery < 1000:
                        nevery = 1000
                    index = 0
                    old_tau = np.inf
                    autocorr = np.empty(self.nmc)
                    for sample in sampler.sample(pos, iterations=self.nmc, progress=True):
                        # Only check convergence every "nevery" steps
                        if sampler.iteration % nevery:
                            continue

                        # Compute the autocorrelation time so far
                        # Using tol=0 means that we'll always get an estimate even
                        # if it isn't trustworthy
                        tau = sampler.get_autocorr_time(tol=0)
                        autocorr[index] = np.mean(tau)
                        index += 1

                        # Check convergence
                        converged = np.all(tau * 100 < sampler.iteration)
                        converged &= np.all(np.abs(old_tau - tau) / tau < nconverge)
                        if converged:
                            print('Converged at %d/%d\n'%(index*nevery,self.nmc))
                            nburn = int(index*nevery / 50) # Burn 50%
                            self.nmc = index*nevery
                            break
                        old_tau = tau
                else:
                    sampler.run_mcmc(pos, self.nmc, progress=True)
                
                flat_samples = sampler.get_chain(discard=nburn, thin=10, flat=True)

                # Plot for chain.
                if f_plot_chain:
                    fig, axes = plt.subplots(self.ndim, figsize=(10, 7), sharex=True)
                    samples = sampler.get_chain()
                    labels = []
                    for key in out.params.valuesdict():
                        if out.params[key].vary:
                            labels.append(key)
                    for i in range(self.ndim):
                        if self.ndim>1:
                            ax = axes[i]
                        else:
                            ax = axes
                        ax.plot(sampler.get_chain()[:,:,i], alpha=0.3, color='k')
                        ax.set_xlim(0, len(samples))
                        ax.yaxis.set_label_coords(-0.1, 0.5)
                        ax.set_ylabel(labels[i])
                    if self.ndim>1:
                        axes[-1].set_xlabel("step number")
                    else:
                        axes.set_xlabel("step number")
                    plt.savefig('%s/chain_%s.png'%(self.DIR_OUT,self.ID))

                # Similar for nested;
                # Dummy just to get structures;
                print('\nRunning dummy sampler. Disregard message from here;\n')
                mini = Minimizer(class_post.lnprob_emcee, out.params, 
                fcn_args=[out.params, self.dict['fy'], self.dict['ey'], self.dict['wht2'], self.f_dust], 
                f_disp=False, nan_policy='omit',
                moves=moves,\
                kwargs={'f_val': True},\
                )
                res = mini.emcee(burn=0, steps=10, thin=1, nwalkers=self.nwalk, 
                params=out.params, is_weighted=True, ntemps=self.ntemp, workers=ncpu, float_behavior='posterior', progress=False)
                print('\nto here.\n')

                # Update;
                var_names = []#res.var_names
                params_value = {}
                ii = 0
                for key in out.params:
                    if out.params[key].vary:
                        var_names.append(key)
                        params_value[key] = np.median(flat_samples[nburn:,ii])
                        ii += 1

                import pandas as pd
                flatchain = pd.DataFrame(data=flat_samples[nburn:,:], columns=var_names)

                class get_res:
                    def __init__(self, flatchain, var_names, params_value, res):
                        self.flatchain = flatchain
                        self.var_names = var_names
                        self.params_value = params_value
                        self.params = res.params
                        for key in var_names:
                            self.params[key].value = params_value[key]

                # Inserting result from res0 into res structure;
                res = get_res(flatchain, var_names, params_value, res)
                res.bic = -99

            elif self.f_nested:
                import dynesty
                from dynesty import NestedSampler

                nlive = self.nmc      # number of live point
                maxmcmc = self.nmc    # maximum MCMC chain length
                nthreads = ncpu       # use one CPU core
                bound = 'multi'   # use MutliNest algorithm for bounds
                sample = 'unif' #'rwalk' # uniform sampling
                tol = 0.01         # the stopping criterion
                ndim_nest = self.ndim #0

                #pars, fy, ey, wht, f_fir
                logl_kwargs = {} #{'pars':out.params, 'fy':dict['fy'], 'ey':dict['ey'], 'wht':dict['wht2'], 'f_fir':self.f_dust}
                logl_args = [self.dict['fy'], self.dict['ey'], self.dict['wht2'], self.f_dust]
                ptform_kwargs = {} #{'pars': out.params}
                ptform_args = []

                from .posterior_nested import Post_nested
                class_post_nested = Post_nested(self, params=out.params)

                # Initialize;
                sampler = NestedSampler(class_post_nested.lnlike, class_post_nested.prior_transform, ndim_nest, walks=self.nwalk,\
                bound=bound, sample=sample, nlive=nlive, ptform_args=ptform_args, logl_args=logl_args, logl_kwargs=logl_kwargs)

                # Run;
                sampler.run_nested(dlogz=tol, maxiter=maxmcmc, print_progress=self.f_disp)                 
                res0 = sampler.results # get results dictionary from sampler
                
                # Dummy just to get structures;
                mini = Minimizer(class_post.lnprob, out.params, fcn_args=[self.dict['fy'], self.dict['ey'], self.dict['wht2'], self.f_dust], f_disp=False, \
                    moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2),])
                res = mini.emcee(burn=0, steps=10, thin=1, nwalkers=self.nwalk, 
                params=out.params, is_weighted=True, ntemps=self.ntemp, workers=ncpu, float_behavior='posterior')

                # Update;
                nburn = int(self.nmc/2)
                var_names = []#res.var_names
                params_value = {}
                ii = 0
                for key in out.params:
                    if out.params[key].vary:
                        var_names.append(key)
                        params_value[key] = np.median(res0.samples[nburn:,ii])
                        ii += 1

                import pandas as pd
                flatchain = pd.DataFrame(data=res0.samples[nburn:,:], columns=var_names)

                class get_res:
                    def __init__(self, flatchain, var_names, params_value, res):
                        self.flatchain = flatchain
                        self.var_names = var_names
                        self.params_value = params_value
                        self.params = res.params
                        for key in var_names:
                            self.params[key].value = params_value[key]

                # Inserting result from res0 into res structure;
                res = get_res(flatchain, var_names, params_value, res)
                res.bic = -99

            else:
                print('Sampling is not specified. Failed. Exiting.')
                return False

            stop_mc  = timeit.default_timer()
            tcalc_mc = stop_mc - start_mc
            print('###############################')
            print('### MCMC part took %.1f sec ###'%(tcalc_mc))
            print('###############################')

            #----------- Save pckl file
            #-------- store chain into a cpkl file
            start_mc = timeit.default_timer()
            burnin = int(self.nmc/2)
            #burnin   = 0 # Since already burnt in.
            savepath = self.DIR_OUT
            cpklname = 'chain_' + self.ID + '_corner.cpkl'
            savecpkl({'chain':flatchain,
                          'burnin':burnin, 'nwalkers':self.nwalk,'niter':self.nmc,'ndim':self.ndim},
                         savepath+cpklname) # Already burn in
            stop_mc = timeit.default_timer()
            tcalc_mc = stop_mc - start_mc
            if verbose:
                print('#################################')
                print('### Saving chain took %.1f sec'%(tcalc_mc))
                print('#################################')


            ####################
            # MCMC corner plot.
            ####################
            if cornerplot:
                levels = [0.68, 0.95, 0.997]
                quantiles = [0.01, 0.99]
                val_truth = []
                for par in var_names:
                    val_truth.append(params_value[par])
                fig1 = corner.corner(flatchain, labels=var_names, \
                label_kwargs={'fontsize':16}, quantiles=quantiles, show_titles=False, \
                title_kwargs={"fontsize": 14}, \
                truths=val_truth, \
                plot_datapoints=False, plot_contours=True, no_fill_contours=True, \
                plot_density=False, levels=levels, truth_color='gray', color='#4682b4')
                fig1.savefig(self.DIR_OUT + 'SPEC_' + self.ID + '_corner.png')
                self.cornerplot_fig = fig1

            # Analyze MCMC results.
            # Write to file.
            stop  = timeit.default_timer()
            tcalc = stop - start

            # Then writing;
            start_mc = timeit.default_timer()
            get_param(self, res, fitc, tcalc=tcalc, burnin=burnin)
            stop_mc = timeit.default_timer()
            tcalc_mc = stop_mc - start_mc

            return 2 # Cannot set to 1, to distinguish from retuen True


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

            Czrec2 = raw_input('What is your manual input for Cz2? [%.3f] '%(self.Cz2))
            if Czrec2 != '':
                Czrec2 = float(Czrec2)
            else:
                Czrec2 = self.Cz2

            self.zprev = self.zgal   # Input redshift for previous run
            self.zgal = zrecom # Recommended redshift from previous run
            self.Cz0 = Czrec0
            self.Cz1 = Czrec1
            self.Cz2 = Czrec2
            print('\n\n')
            print('Generating model templates with input redshift and Scale.')
            print('\n\n')
            return True

        else:
            print('\n\n')
            
            flag_gen = raw_input('Do you want to make templates with recommended redshift, Cz0, Cz1, Cz2, and chi2/nu , %.5f %.5f %.5f %.5f %.5f? ([y]/n) '%\
                (self.zrecom, self.Czrec0, self.Czrec1, self.Czrec2, self.fitc_cz_prev))

            self.get_zdist()

            if flag_gen == 'y' or flag_gen == '':
                self.zprev = self.zgal # Input redshift for previous run
                self.zgal = self.zrecom # Recommended redshift from previous run
                self.Cz0 = self.Czrec0
                self.Cz1 = self.Czrec1
                self.Cz2 = self.Czrec2
                return True
            else:
                print('\n\n')
                print('Terminating process.')
                print('\n\n')
                return False
