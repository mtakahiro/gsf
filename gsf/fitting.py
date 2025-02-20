"""
This includes the main part of gsf.

Original copyright:
   Copyright (c) 2023 Takahiro Morishita, IPAC
"""
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from lmfit import Model, Parameters, minimize, fit_report#, Minimizer
from numpy import log10
import os.path
import random
import string
import timeit
from scipy import stats
from scipy.stats import norm
import scipy.interpolate as interpolate
from astropy.io import fits,ascii
import corner
import emcee
import zeus
import pandas as pd
import asdf
import logging
import pathlib

# import from custom codes
from .function import check_line_man, check_line_cz_man, calc_Dn4, savecpkl, get_leastsq, print_err, str2bool
from .zfit import check_redshift,get_chi2
from .writing import get_param
from .function_class import Func
from .minimizer import Minimizer
from .posterior_flexible import Post
from .function_igm import get_XI

from .Logger.GsfBase import GsfBase
try:
    GSF = os.environ['GSF']
except ValueError:
    print("!! make sure you have set GSF to the path of your GSF repository;\n\
          `export GSF=/your/Github/gsf-directory/`\
          where you can see the `config/` directory.")

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


class Mainbody(GsfBase):
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
        cosmo=None, idman:str=None, zman=None, zman_min=None, zman_max=None, NRbb_lim=10000, verbose=False, configurationfile=None,
        show_list=True):
        '''
        Parameters
        ----------
        NRbb_lim : int
            BB data is associated with ids greater than this number.
        '''
        import gsf
        DIR_GSF = gsf.__path__[0]
        if configurationfile == None:
            configurationfile = f"{GSF}/config/config.yaml"
        super().__init__(configurationfile)
        self.outdir = pathlib.Path('/tmp/logs/')
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.logger = self.logger.getLogger(__name__, "gsf")
        self.logger.info("init class")
        self.flat_fin = {}

        self.verbose = verbose
        flag_input = self.update_input(inputs, idman=idman, zman=zman, zman_min=zman_min, zman_max=zman_max)
        self.NRbb_lim = NRbb_lim
        self.ztemplate = False
        # self.z_prior = None
        # self.p_prior = None

        if not flag_input:
            self.flag_class = False
        else:
            self.flag_class = True

        self.fit_params = None

        self.config_params = {

            'Templates' : ['TAU0', 'NIMF', 'BINARY', 
                           'SFH_FORM',
                           'AMIN', 'AMAX', 
                           'ADD_NEBULAE', 'logUMIN', 'logUMAX', 'DELlogU', 'logUFIX', 
                           'ADD_AGN', 'AGNTAUMIN', 'AGNTAUMAX', 'DELAGNTAU', 'AGNTAUFIX', 'AAGNMIN', 'AAGNMAX',
                           'AGE', 'AGEMIN', 'AGEMAX', 'DELAGE', 'AGEFIX',
                           'ZMIN', 'ZMAX', 'DELZ', 'ZFIX', 'ZEVOL', 
                           'AVMIN', 'AVMAX', 'AVFIX', 'AVEVOL', 'AVPRIOR_SIGMA', 'DUST_MODEL', 
                           'ZMC', 'ZMCMIN', 'ZMCMAX', 'F_ZMC', 
                           'TDUSTMIN', 'TDUSTMAX', 'DELTDUST', 'TDUSTFIX', 'DUST_NUMAX', 'DUST_NMODEL', 'DIR_DUST',
                           'BPASS', 'DIR_BPASS',
                           'TAUMIN', 'TAUMAX', 'DELTAU', 'NPEAK', 
                           'XHIFIX'
                           ],

            'Fitting' : ['MC_SAMP', 'NMC', 'NWALK', 'NMCZ', 'NWALKZ', 
                         'FNELD', 'NCPU', 'F_ERR', 'ZVIS', 'F_MDYN',
                         'NTEMP', 'DISP', 'SIG_TEMP', 'VDISP', 
                         'FORCE_AGE', 'NORDER_SFH_PRIOR', 'NEBULAE_PRIOR',
                         'F_XHI'],

            'Data' : ['ID', 'MAGZP', 'DIR_TEMP', 
                      'CAT_BB', 'CAT_BB_DUST', 'SNLIM',
                      'MORP', 'MORP_FILE', 
                      'SPEC_FILE', 'DIR_EXTR', 'MAGZP_SPEC', 'UNIT_SPEC', 'DIFF_CONV', 
                      'CZ0', 'CZ1', 'CZ2', 'LINE', 'PA', ],

            'Misc' : ['CONFIG', 'DIR_OUT', 'FILTER', 'SKIPFILT', 'FIR_FILTER', 'DIR_FILT', 'USE_UPLIM']

            }

        self.param_names = ['A', 'logU', 'AGE', 'Z', 'AV', 'ZMC', 'TDUST', 'TAU']
        self.check_input(inputs, self.config_params, show_list=show_list)

        return 


    def check_input(self, inputs, dict_config, show_list=False):
        '''
        '''
        keys = np.asarray([key for key in inputs.keys()])
        flag_key = np.zeros(len(keys), int)
 
        for kk,key in enumerate(keys):
            for list in dict_config.keys():
                if key in dict_config[list]:
                    flag_key[kk] = 1
 
        con = flag_key == 0
        if len(keys[con])>0:
 
            self.logger.warning('Some keywords in the config file are not recognized:')
            print(keys[con])
 
            if show_list:
                self.logger.warning('Available input keywords are as follow:')
                print(dict_config)
            

    def get_configfile(self, name=None):
        '''
        Purpose
        -------
        Generate a configuration file.
        '''
        # Version;
        import gsf
        ver = gsf.__version__

        if name == None:
            name = 'default.input'
            
        fw = open(name,'w')
        fw.write('#\n# Input file for ver.%s\n'%ver)

        for key in self.config_params:
            fw.write('\n# %s\n'%key)
            for ele in self.config_params[key]:
                try:
                    value = self.inputs[ele]
                    fw.write('%s\t%s\n'%(ele, value))
                except:
                    fw.write('# %s\t\n'%(ele))
        
        fw.close()
        return name
        

    def update_input(self, inputs, c:float=3e18, Mpc_cm:float=3.08568025e+24, m0set:float=25.0, pixelscale:float=0.06, Lsun:float=3.839*1e33, cosmo=None,
                    idman:str=None, zman=None, zman_min=None, zman_max=None, sigz:float=5.0):
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
        self.key_params_prior = []
        self.key_params_prior_sigma = []

        # Then register;
        self.inputs = inputs
        self.c = c
        self.Mpc_cm = Mpc_cm
        self.pixelscale = pixelscale
        self.Lsun = Lsun
        self.sigz = sigz
        self.fitc_cz_prev = 1e10

        # Set config path;
        try:
            self.config_path = self.inputs['CONFIG']
        except:
            self.config_path = os.path.expandvars('$GSF/config/')
            self.logger.info('CONFIG is set to %s'%self.config_path)

        # Magzp;
        try:
            self.m0set = float(inputs['MAGZP'])
        except:
            self.logger.info('MAGZP is not found. Set to %.2f'%(m0set))
            self.m0set = m0set
        self.d = 10**((48.6+self.m0set)/2.5) # Conversion factor from [ergs/s/cm2/A] to Fnu.

        if cosmo == None:
            from astropy.cosmology import WMAP9 as cosmo
            self.cosmo = cosmo
        else:
            self.cosmo = cosmo

        if idman != None:
            self.ID = '%s'%idman
        else:
            self.ID = '%s'%inputs['ID']
        self.logger.info('Fitting target: %s'%self.ID)

        # Read catalog;
        try:
            self.CAT_BB = inputs['CAT_BB']
            self.fd_cat = ascii.read(self.CAT_BB)
        except:
            self.CAT_BB = None
            self.fd_cat = None

        if zman != None:
            self.zgal = zman
            self.zmcmin = zman_min
            self.zmcmax = zman_max
        else:
            try:
                self.zgal = float(inputs['ZMC'])
            except:
                ids_cat = self.fd_cat['id'].astype('str')
                iix = np.where(ids_cat.value == self.ID)
                if len(iix[0]) == 0:
                    msg = 'id `%s` cannot be found in the catalog, `%s`'%(self.ID,self.CAT_BB)
                    print_err(msg, exit=True)
                self.zgal = float(self.fd_cat['redshift'][iix])

            if self.zgal < 0:
                msg = '%s has negative redshift, %.2f'%(self.ID, self.zgal)
                print_err(msg, exit=False)
                return False

            try:
                self.zmcmin = float(inputs['ZMCMIN'])
                self.zmcmax = float(inputs['ZMCMAX'])
            except:
                try:
                    self.zmcmin = self.zgal - float(self.fd_cat['ez_l'][iix])
                    self.zmcmax = self.zgal + float(self.fd_cat['ez_u'][iix])
                except:
                    self.logger.warning('ZMCMIN and ZMCMAX cannot be found. z range is set to z pm 1.0')
                    self.zmcmin = None
                    self.zmcmax = None

        self.set_zprior(self.zmcmin, self.zmcmax, delzz=0.01, f_eazy=False, eaz_pz=None, zmax=self.zmcmax)

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

        if self.verbose:
            self.logger.info('f_Mdyn is set to %s\n'%self.f_Mdyn)
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
        self.fagn = False
        self.nAGNTAU = 0
        
        self.check_keys(self)

        if 'BPASS' in self.input_keys and str2bool(inputs['BPASS']):
            self.logger.warning('Currently, BPASS does not have option of nebular emission.')
            inputs['ADD_NEBULAE'] = '0'

        # Nebular;
        self.neb_correlate = False
        if 'ADD_NEBULAE' in self.input_keys:
            if str2bool(inputs['ADD_NEBULAE']):
                self.fneb = True
                try:
                    # Correlation between Aneb and LW age? May add some time; see posterior_flexible
                    if inputs['NEBULAE_PRIOR'] == '1':
                        self.neb_correlate = True 
                    else:
                        self.neb_correlate = False
                except:
                    self.neb_correlate = False
                    
                try:
                    self.logUMIN = float(inputs['logUMIN'])
                except:
                    self.logUMIN = -2.5
                try:
                    self.logUMAX = float(inputs['logUMAX'])
                except:
                    self.logUMAX = -2.0
                try:
                    self.DELlogU = float(inputs['DELlogU'])
                    if self.DELlogU<0.1:
                        print_err('`DELlogU` cannot have value smaller than 0.1. Exiting.')
                        sys.exit()
                except:
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

                try:
                    nfneb_tied = str(inputs['NEBULAE_TIED'])
                except:
                    nfneb_tied = '0'
                self.fneb_tied = str2bool(nfneb_tied)

            else:
                self.fneb = False                
                self.logUMIN = -2.5
                self.logUMAX = -2.0
                self.DELlogU = 0.5
                self.logUs = np.arange(self.logUMIN, self.logUMAX, self.DELlogU)
        else:
            if self.verbose:
                print_err('Some error in nebular setup; No nebular added.')
            self.fneb = False
            self.logUMIN = -2.5
            self.logUMAX = -2.0
            self.DELlogU = 0.5
            self.logUs = np.arange(self.logUMIN, self.logUMAX, self.DELlogU)
            pass

        # AGN;
        if 'ADD_AGN' in self.input_keys:
            if str2bool(inputs['ADD_AGN']):
                self.fagn = True
                try:
                    self.AGNTAUMIN = float(inputs['AGNTAUMIN'])
                except:
                    self.AGNTAUMIN = 5
                try:
                    self.AGNTAUMAX = float(inputs['AGNTAUMAX'])
                except:
                    self.AGNTAUMAX = 15
                try:
                    self.DELAGNTAU = float(inputs['DELAGNTAU'])
                    if self.DELAGNTAU<1 and not self.agn_powerlaw:
                        print_err('`DELAGNTAU` cannot have value smaller than 1. Exiting.')
                        sys.exit()
                except:
                    self.DELAGNTAU = 1.0
                self.AGNTAUs = np.arange(self.AGNTAUMIN, self.AGNTAUMAX, self.DELAGNTAU)
                self.nAGNTAU = len(self.AGNTAUs)

                if 'AGNTAUFIX' in self.input_keys:
                    self.AGNTAUFIX = float(inputs['AGNTAUFIX'])
                    self.nAGNTAU = 1
                    self.AGNTAUMIN = self.AGNTAUFIX
                    self.AGNTAUMAX = self.AGNTAUFIX
                    self.DELAGNTAU = 0
                    self.AGNTAUs = np.asarray([self.AGNTAUMAX])
                else:
                    self.AGNTAUFIX = None

            else:
                self.fagn = False                
                self.AGNTAUMIN = 10
                self.AGNTAUMAX = 15
                self.DELAGNTAU = 5
                self.AGNTAUs = np.arange(self.AGNTAUMIN, self.AGNTAUMAX, self.DELAGNTAU)
        else:
            if self.verbose:
                print_err('Some error in agn setup; No agn added.')
            self.fagn = False
            self.AGNTAUMIN = 10
            self.AGNTAUMAX = 15
            self.DELAGNTAU = 5
            self.AGNTAUs = np.arange(self.AGNTAUMIN, self.AGNTAUMAX, self.DELAGNTAU)
            pass

        # Output directory;
        try:
            self.DIR_OUT = inputs['DIR_OUT']
            if not os.path.exists(self.DIR_OUT):
                os.mkdir(self.DIR_OUT)
        except:
            self.DIR_OUT = './'

        # Filter response curve directory, if bb catalog is provided.
        self.DIR_FILT = os.path.join(self.config_path, 'filter/')
        try:
            self.filts = inputs['FILTER']
            self.filts = [x.strip() for x in self.filts.split(',')]
        except:
            self.filts = []
            if not isinstance(self.fd_cat, type(None)):
                for column in self.fd_cat.columns:
                    if column[0] == 'F':
                        self.filts.append(column[1:])

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
                self.logger.info('AGEFIX is found.\nAge will be fixed to:')
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
            self.logger.warning('Cannot find F_ZMC. Set to %d.'%(self.fzmc))

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

            if self.verbose:
                self.logger.info('ZFIX is found.\nZ will be fixed to: %.2f'%(self.ZFIX))

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
                self.DIR_BPASS = inputs['DIR_BPASS']
            except:
                self.logger.warning('DIR_BPASS is not found. Using default.')
                self.DIR_BPASS = '/astro/udfcen3/Takahiro/BPASS/'
                if not os.path.exists(self.DIR_BPASS):
                    msg = 'BPASS directory, %s, not found.'%self.DIR_BPASS
                    print_err(msg, exit=True)

            self.BPASS_ver = 'v2.2.1'
            self.Zsun = 0.020
            Zbpass = [1e-5, 1e-4, 0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.010, 0.020, 0.030, 0.040]
            Zbpass = np.log10(np.asarray(Zbpass)/self.Zsun)

            try: # If ZFIX is found;
                iiz = np.argmin(np.abs(Zbpass[:] - float(inputs['ZFIX']) ) )
                if Zbpass[iiz] - float(inputs['ZFIX']) != 0:
                    self.logger.warning('%.2f is not found in BPASS Z list. %.2f is used instead.'%(float(inputs['ZFIX']),Zbpass[iiz]))
                self.ZFIX = Zbpass[iiz]
                self.delZ = 0.0001
                self.Zmin, self.Zmax = self.ZFIX, self.ZFIX + self.delZ
                self.Zall = np.arange(self.Zmin, self.Zmax, self.delZ) # in logZsun
                self.logger.info('ZFIX is found.\nZ will be fixed to: %.2f'%(self.ZFIX))
                self.has_ZFIX = True
            except:
                self.logger.warning('ZFIX is not found.')
                self.logger.warning('Metallicities available in BPASS are limited and discrete. ZFIX is recommended : ')
                self.logger.warning(self.Zall)
                self.Zmax, self.Zmin = float(inputs['ZMAX']), float(inputs['ZMIN'])
                con_z = np.where((Zbpass >= self.Zmin) & (Zbpass <= self.Zmax))
                self.Zall = Zbpass[con_z]

                if len(self.Zall) == 0:
                    self.logger.warning('No metallicity can be found. Available Zs are:')
                    self.logger.info(Zbpass)
                    sys.exit()

                self.delZ = 0.0001
                self.Zmax,self.Zmin = np.max(self.Zall), np.min(self.Zall)
                self.logger.info('Final list for log(Z_BPASS/Zsun) is : ')
                self.logger.info(self.Zall)
                if len(self.Zall)>1:
                    self.has_ZFIX = False
                    self.ZFIX = None

        # N of param:
        if 'AVEVOL' in inputs:
            self.AVEVOL = str2bool(inputs['AVEVOL'])
        else:
            self.AVEVOL = False

        try:
            _ = inputs['AVFIX']
            self.has_AVFIX = True
        except:
            self.has_AVFIX = False

        if self.has_AVFIX:
            Avfix = float(inputs['AVFIX'])
            self.AVFIX = Avfix
            self.nAV = 0
            self.logger.info('AVFIX is found.\nAv will be fixed to:\n %.2f'%(Avfix))
        else:
            try:
                self.Avmin = float(inputs['AVMIN'])
                self.Avmax = float(inputs['AVMAX'])
                if self.Avmin == self.Avmax:
                    self.nAV = 0
                    self.AVFIX = self.Avmin
                    self.has_AVFIX = True
                else:
                    self.nAV = 1
            except:
                self.nAV = 1
                self.Avmin = 0
                self.Avmax = 4.0

        if 'AVPRIOR_SIGMA' in inputs:
            Av_prior = float(inputs['AVPRIOR_SIGMA'])
            self.key_params_prior.append('AV0')
            self.key_params_prior_sigma.append(Av_prior)

        # Z evolution;
        if self.verbose:
            print('\n#############################')
        if self.SFH_FORM == -99:
            if int(inputs['ZEVOL']) == 1:
                self.ZEVOL = 1
                self.ndim = int(self.npeak * 2 + self.nAV) # age, Z, and Av.
                if self.verbose:
                    self.logger.info('Metallicity evolution is on')
            else:
                self.ZEVOL = 0
                if self.verbose:
                    self.logger.info('Metallicity evolution is off')
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
                if self.verbose:
                    self.logger.info('Metallicity evolution is on')
            else:
                self.ZEVOL = 0
                if self.verbose:
                    self.logger.info('Metallicity evolution is off')
                try:
                    ZFIX = float(inputs['ZFIX'])
                    self.nZ = 0
                except:
                    if np.max(self.Zall) == np.min(self.Zall):
                        self.nZ = 0
                    else:
                        self.nZ = 1
                    
            self.ndim = int(self.npeak*3 + self.nZ + self.nAV) # age, Z, and Av.

        if self.verbose:
            print('#############################\n')

        # Redshift
        self.ndim += self.fzmc
        if self.verbose:
            self.logger.info('No. of params are : %d'%(self.ndim))

        # XHI as a param;
        if 'XHIFIX' in inputs:
            self.x_HI_input = float(inputs['XHIFIX'])
            self.fxhi = False
        else:
            self.x_HI_input = None
            if 'F_XHI' in inputs:
                self.fxhi = str2bool(inputs['F_XHI'])
                if self.fxhi:
                    self.ndim += 1
            else:
                self.fxhi = False

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
                self.logger.warning('Unknown index number for dust model. Setting to Calzetti.')
                self.dust_model = 0
                self.dust_model_name = 'Calz'
        except:
            self.logger.warning('Index number for dust model (DUST_MODEL) cannot be found. Setting to Calzetti.')
            self.dust_model = 0
            self.dust_model_name = 'Calz'

        if self.verbose:
            self.logger.info('Dust model is set to %s\n'%self.dust_model_name)

        # If FIR data;
        try:
            DT0 = float(inputs['TDUSTMIN'])
            DT1 = float(inputs['TDUSTMAX'])
            dDT = float(inputs['DELTDUST'])
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
            self.logger.info('FIR fit is on.')
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
            self.logger.warning('Cannot find NIMF. Set to %d.'%(self.nimf))

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
            self.logger.info('Missing MC_SAMP keyword. Setting f_mcmc True.')
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
        try:
            self.norder_sfh_prior = int(inputs['NORDER_SFH_PRIOR'])
            self.f_prior_sfh = True
        except:
            self.norder_sfh_prior = None
            self.f_prior_sfh = False

        # include ND;
        if 'USE_UPLIM' in inputs:
            self.f_chind = str2bool(inputs['USE_UPLIM'])
        else:
            self.f_chind = True

        self.logger.info('Complete')
        return True


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
        con_spec = (NR<10000)
        if len(NR[con_spec])>0:
            self.has_spectrum = True
        else:
            self.has_spectrum = False

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
            self.has_photometry = True
        except: # if no BB;
            print('No BB data.')
            NRbb = np.asarray([])
            xbb = np.asarray([])
            fybb = np.asarray([])
            eybb = np.asarray([])
            exbb = np.asarray([])
            self.has_photometry = False

        con_bb = ()
        xx_bb = xbb[con_bb]
        ex_bb = exbb[con_bb]
        fy_bb = fybb[con_bb]
        ey_bb = eybb[con_bb]

        xx = np.append(xx02,xx_bb)
        fy = np.append(fy02,fy_bb)
        ey = np.append(ey02,ey_bb)

        wht = 1./np.square(ey)
        con_wht = (ey<=0)
        wht[con_wht] = 0

        # For now...
        #wht2 = check_line_man(fy, x, wht, fy, zgal, self.LW0)
        wht2 = wht[:]

        # Check number of optical/IR data points;
        self.n_optir = len(wht2)

        # Append data;
        if add_fir:
            nr_d, x_d, fy_d, ey_d = self.data['spec_fir_obs']['NR'], self.data['spec_fir_obs']['x'], self.data['spec_fir_obs']['fy'], self.data['spec_fir_obs']['ey']
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

        dict = {}
        dict = {'NR':NR, 'x':x, 'fy':fy, 'ey':ey, 'NRbb':NRbb, 'xbb':xx_bb, 'exbb':ex_bb, 'fybb':fy_bb, 'eybb':ey_bb, 'wht':wht, 'wht2': wht2, 'sn':sn}

        return dict
    

    def plot_data(self,):
        '''
        '''
        return None


    def set_zprior(self, zliml, zlimu, delzz=0.01, priors=None, 
                   f_eazy=False, eaz_pz=None, zmax=20, f_norm=True):
        '''
        '''
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

        self.z_prior = zz_prob
        self.p_prior = prior_s
        return 


    def fit_redshift(self, xm_tmp, fm_tmp, delzz=0.01, ezmin=0.01, #zliml=0.01, zlimu=None, 
                     snlim=0, priors=None, f_line_check=False, fzvis=False,
                     f_norm=True, f_lambda=False, zmax=30, include_photometry=False, 
                     f_exclude_negative=False, return_figure=False):
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
        include_photometry : bool
            Turn this True if Redshift fitting is requred for only BB data.
        f_line_check : bool
            If True, line masking.
        priors : dict, optional
            Dictionary that contains z (redshift grid) and chi2 (chi-square).
        f_exclude_negative : bool
            Exclude negative fluxes in spectrum.
        return_figure : bool
            This only works for notebook

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
            zlimu = zmax

        # Observed data;
        sn = self.dict['fy'] / self.dict['ey']

        # Only spec data?
        if include_photometry:
            if f_exclude_negative:
                con_cz = (sn>snlim)
            else:
                con_cz = ()
        else:
            if f_exclude_negative:
                con_cz = (self.dict['NR']<self.NRbb_lim) & (sn>snlim)
            else:
                con_cz = (self.dict['NR']<self.NRbb_lim) #& (sn>snlim)
            
        fy_cz = self.dict['fy'][con_cz] # Already scaled by self.Cz0
        ey_cz = self.dict['ey'][con_cz]
        x_cz = self.dict['x'][con_cz] # Observed range
        NR_cz = self.dict['NR'][con_cz]
        if len(NR_cz) == 0:
            self.logger.error('No data point exists at SNR > `snlim` (%.1f)'%(snlim))
            self.logger.error('Decrease `snlim` or turn `f_exclude_negative` to False')
            return False

        fint = interpolate.interp1d(xm_tmp, fm_tmp, kind='nearest', fill_value="extrapolate")
        fm_s = fint(x_cz)
        del fint

        #
        # If Eazy result exists;
        #
        try:
            eaz_pz = self.inputs['EAZY_PZ']
            f_eazy = True
        except:
            f_eazy = False
            eaz_pz = None

        # Attach prior:
        self.set_zprior(zliml, zlimu, delzz=delzz, f_eazy=f_eazy, eaz_pz=eaz_pz, zmax=zmax)

        # Plot;
        if len(self.dict['fy'][con_cz])==0:
            return 'y'

        # Figure;
        # mpl.use('TkAgg')
        plt.close()
        fig = plt.figure(figsize=(8,2.8))
        ax1 = fig.add_subplot(111)

        if (include_photometry | len(fy_cz)>0):# and fzvis:# and 
            data_model = np.zeros((len(x_cz),4),'float')
            data_model[:,0] = x_cz
            data_model[:,1] = fm_s
            data_model[:,2] = fy_cz
            data_model[:,3] = ey_cz

            data_model_sort = data_model[data_model[:, 0].argsort()]            
            #data_model_sort = np.sort(data_model, axis=0) # This does not work!!

            # Model based on input z.
            # ax1.plot(data_model_sort[:,0], data_model_sort[:,1], 'b', linestyle='--', linewidth=0.5, label='')

            # Observed data;
            if self.has_spectrum:
                ey_max = 1000
                con = (self.dict['ey']<ey_max) & (self.dict['NR']<self.NRbb_lim) & (self.dict['ey']>=0)
                ax1.errorbar(self.dict['x'][con], self.dict['fy'][con], yerr=self.dict['ey'][con], color='gray', capsize=0, linewidth=0.5, linestyle='', zorder=4)
                ax1.plot(self.dict['x'][con], self.dict['fy'][con], '.r', linestyle='', linewidth=0.5, label='Observed spectrum', zorder=4)

            if include_photometry and self.has_photometry:
                con = (self.dict['NR']>=self.NRbb_lim) & (self.dict['ey']>=0)
                ax1.errorbar(self.dict['x'][con], self.dict['fy'][con], yerr=self.dict['ey'][con], ms=15, marker='None', 
                    color='orange', capsize=0, linewidth=0.5, ls='None', label='', zorder=4)
                ax1.scatter(self.dict['x'][con], self.dict['fy'][con], s=100, marker='o', 
                    color='orange', edgecolor='k', label='Observed photometry', zorder=4)

            # Write prob distribution;
            get_chi2(self.z_prior, fy_cz, ey_cz, x_cz, fm_tmp, xm_tmp/(1+self.zgal), self.file_zprob)

            print('############################')
            print('Start MCMC for redshift fit')
            print('############################')
            res_cz, fitc_cz = check_redshift(
                fy_cz, ey_cz, x_cz, fm_tmp, xm_tmp/(1+self.zgal), 
                self.zgal, self.z_prior, self.p_prior,
                NR_cz, self.data['meta']['data_len'], 
                zliml, zlimu, 
                self.nmc_cz, self.nwalk_cz, 
                include_photometry=include_photometry
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
        data_model_new = np.zeros((len(x_cz),4),'float')
        data_model_new[:,0] = x_cz
        data_model_new[:,1] = fm_s
        data_model_new_sort = data_model_new[data_model_new[:, 0].argsort()]

        ax1.plot(data_model_new_sort[:,0], data_model_new_sort[:,1], 'lightgreen', linestyle='--', linewidth=1, 
                 label='%s ($z=%.5f$)'%(fit_label,zrecom), zorder=3) # Model based on recomended z.
        ax1.plot(x_cz[con_line], fm_s[con_line], color='orange', marker='o', linestyle='', linewidth=3., zorder=5)

        # Plot lines for referenc
        for ll in range(len(LW)):
            try:
                conpoly = (x_cz/(1.+zrecom)>3000) & (x_cz/(1.+zrecom)<8000)
                yline = np.max(ypoly[conpoly])
                yy = np.arange(yline/1.02, yline*1.1)
                xxpre = yy * 0 + LW[ll] * (1.+self.zgal)
                xx = yy * 0 + LW[ll] * (1.+zrecom)
                ax1.plot(xxpre, yy/1.02, linewidth=0.5, linestyle='--', color='gray')
                ax1.text(LW[ll] * (1.+self.zgal), yline/1.05, '%s'%(LN[ll]), fontsize=8, color='gray')
                ax1.plot(xx, yy, linewidth=0.5, linestyle='-', color='orangered')
                ax1.text(LW[ll] * (1.+zrecom), yline, '%s'%(LN[ll]), fontsize=8, color='orangered')
            except:
                pass

        # Plot data
        data_obsbb = np.zeros((len(self.dict['xbb']),3), 'float')
        data_obsbb[:,0],data_obsbb[:,1] = self.dict['xbb'],self.dict['fybb']
        if len(fm_tmp) == len(self.dict['xbb']): # BB only;
            data_obsbb[:,2] = fm_tmp
        data_obsbb_sort = data_obsbb[data_obsbb[:, 0].argsort()]            
        
        if len(fm_tmp) == len(self.dict['xbb']): # BB only;
            ax1.scatter(data_obsbb_sort[:,0], data_obsbb_sort[:,2], color='none', marker='d', s=50, edgecolor='b', zorder=2, label='Current model ($z=%.5f$)'%(self.zgal))
        else:
            model_spec = np.zeros((len(fm_tmp),2), 'float')
            model_spec[:,0],model_spec[:,1] = xm_tmp,fm_tmp
            model_spec_sort = model_spec[model_spec[:, 0].argsort()]
            # ax1.plot(model_spec_sort[:,0], model_spec_sort[:,1], marker='.', color='b', ms=1, linestyle='-', linewidth=0.5, zorder=2)
            ax1.plot(data_model_sort[:,0], data_model_sort[:,1], 'b', linestyle='-', linewidth=3.0, zorder=2, label='Current model ($z=%.5f$)'%(self.zgal))

        try:
            xmin, xmax = np.min(x_cz)/1.1,np.max(x_cz)*1.1
        except:
            xmin, xmax = 2000,10000

        try:
            plt.ylim(0,yline*1.1)
        except:
            pass

        # lines;
        xx = np.arange(xmin,xmax)
        ax1.plot(xx, xx*0, lw=0.5, ls='--', color='gray')

        ax1.set_xlim(xmin,xmax)
        ax1.set_xlabel('Wavelength ($\mathrm{\AA}$)')
        ax1.set_ylabel('$F_\\nu$ (arb.)')
        ax1.legend(loc=0)

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
        print('Estimated error is %.3f per cent.'%(zzsigma*100))
        print('Input Cz0 is %.3f per cent agreement.'%((1.-C0sigma)*100))
        print('Estimated error is %.3f per cent.'%(eC0sigma*100))
        print('Input Cz1 is %.3f per cent agreement.'%((1.-C1sigma)*100))
        print('Estimated error is %.3f per cent.'%(eC1sigma*100))
        print('Input Cz2 is %.3f per cent agreement.'%((1.-C2sigma)*100))
        print('Estimated error is %.3f per cent.'%(eC2sigma*100))
        print('##############################################################\n')

        if fzvis==1:
            # Ask interactively;
            plt.show()
            plt.close()
            flag_z = raw_input('Do you want to continue with the input redshift, Cz0, Cz1, Cz2, and chi2/nu, %.5f %.5f %.5f %.5f %.5f? ([y]/n/m) '%\
                (self.zgal, self.Cz0, self.Cz1, self.Cz2, self.fitc_cz_prev))
        else:
            if not return_figure:
                plt.close()
            self.fig_zfit = fig
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

        if return_figure:
            return fig
        return flag_z


    def get_zdist(self, f_interact=False, f_ascii=True, return_figure=False):
        '''
        Saves a plot of z-distribution.

        Parameters
        ----------
        f_interact : bool
            If true, this module returns figure ax.
        '''
        try:
            fig = plt.figure(figsize=(6.5,4))
            fig.subplots_adjust(top=0.96, bottom=0.16, left=0.09, right=0.99, hspace=0.15, wspace=0.25)
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            n, nbins, _ = ax1.hist(self.res_cz.flatchain['z'], bins=200, density=True, color='gray', label='')

            yy = np.arange(0,np.max(n),1)
            xx = yy * 0 + self.z_cz[1]
            ax1.plot(xx,yy,linestyle='-',linewidth=1,color='orangered',\
                # label='$z=%.5f_{-%.5f}^{+%.5f}$\n$C_{z0}=%.3f$\n$C_{z1}=%.3f$\n$C_{z2}=%.3f$'%\
                label='$z=%.5f_{-%.5f}^{+%.5f}$'%\
                (self.z_cz[1],self.z_cz[1]-self.z_cz[0],self.z_cz[2]-self.z_cz[1]))

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
            
            # # Save:
            # file_out = self.DIR_OUT + 'zmc_' + self.ID + '.png'
            # print('Figure is saved in %s'%file_out)

            # if f_interact or return_figure:
            #     fig.savefig(file_out, dpi=300)
            # else:
            #     plt.savefig(file_out, dpi=300)
            #     plt.close()
        except:
            print('z-distribution figure is not generated.')
            pass

        if os.path.exists(self.file_zprob):
            # Also, make zprob;
            fd = ascii.read(self.file_zprob)
            z = fd['z']
            pz = fd['p(z)']
            pz /= pz.max()

            # prob:
            ax2.plot(z, pz, linestyle='-', linewidth=1, color='r', label='')

            # Label:
            ax2.set_xlabel('Redshift')
            ax2.set_ylabel('$p(z)$')
            file_out = self.DIR_OUT + 'zprob_' + self.ID + '.png'
            if f_interact or return_figure:
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

            fit_params.add('MDUST', value=10, min=0, max=15)
            self.ndim += 1
            self.dict = self.read_data(self.Cz0, self.Cz1, self.Cz2, self.zgal, add_fir=self.f_dust)
            f_add = True

        # Nebular; ver1.6
        if self.fneb:
            self.Anebmin = -10
            self.Anebmax = 10
            if self.fneb_tied:
                # @@@ TBD
                iix = np.argmin(np.abs(self.age-0.01))
                print('Aneb is tied to A%d'%iix)
                fit_params.add('Aneb', value=self.Aini, min=self.Amin, max=self.Amax, expr='A%d if Aneb > A%d'%(iix,iix)) #self.Amax)#, expr='<A%d'%iix)
            else:
                fit_params.add('Aneb', value=self.Aini, min=self.Anebmin, max=self.Anebmax)
            self.ndim += 1
            if not self.logUFIX == None:
                fit_params.add('logU', value=self.logUFIX, vary=False)
            else:
                fit_params.add('logU', value=np.median(self.logUs), min=self.logUMIN, max=self.logUMAX)
                self.ndim += 1
            f_add = True

        # AGN; ver1.9
        if self.fagn:

            self.AAGNmin = -10
            self.AAGNmax = 10
            if 'AAGNMIN' in self.inputs:
                self.AAGNmin = float(self.inputs['AAGNMIN'])
            if 'AAGNMAX' in self.inputs:
                self.AAGNmax = float(self.inputs['AAGNMAX'])

            fit_params.add('Aagn', value=self.Aini, min=self.AAGNmin, max=self.AAGNmax)
            # fit_params.add('Aagn', value=self.Aini, min=-6, max=-5)
            self.ndim += 1
            if not self.AGNTAUFIX == None:
                fit_params.add('AGNTAU', value=self.AGNTAUFIX, vary=False)
            else:
                fit_params.add('AGNTAU', value=np.median(self.AGNTAUs), min=self.AGNTAUMIN, max=self.AGNTAUMAX)
                self.ndim += 1
            f_add = True

        # xhi
        if self.fxhi:
            xhi0 = get_XI(self.zgal)
            fit_params.add('xhi', value=xhi0, min=0, max=1)
            f_add = True

        self.fit_params = fit_params

        return f_add


    def set_param(self):
        '''
        Set parameters.
        '''
        self.logger.info('Setting parameters')
        agemax = self.cosmo.age(self.zgal).value 
        fit_params = Parameters()
        f_Alog = True
        if f_Alog:
            try:
                self.Amin = float(self.inputs['AMIN'])
                self.Amax = float(self.inputs['AMAX'])
            except:
                self.Amin = -10
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
                        self.logger.warning('At this redshift, A%d is beyond the age of universe and not being used.'%(aa))
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
                    self.logger.info('AGEFIX is found. Set to %.2f'%(AGEFIX))
                except:
                    tcosmo = np.log10(self.cosmo.age(self.zgal).value) 
                    agemax_tmp = self.agemax
                    if agemax_tmp > tcosmo:
                        agemax_tmp = tcosmo
                        self.logger.info('Maximum age is set to the age of the univese (%.1fGyr) at this redshift.'%(self.cosmo.age(self.zgal).value))
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
            fit_params.add('AV0', value=Avfix, vary=False)
            self.Avmin = Avfix
            self.Avmax = Avfix
        except:
            try:
                self.Avmin = float(self.inputs['AVMIN'])
                self.Avmax = float(self.inputs['AVMAX'])
                self.Avini = (self.Avmax+self.Avmin)/2.
                self.Avini = 0.
                if self.Avmin == self.Avmax:
                    fit_params.add('AV0', value=self.Avini, vary=False)
                    self.Avmin = self.Avini
                    self.Avmax = self.Avini
                else:
                    fit_params.add('AV0', value=self.Avini, min=self.Avmin, max=self.Avmax)
            except:
                self.Avmin = 0.
                self.Avmax = 4.
                self.Avini = 0.5 #(Avmax-Avmin)/2. 
                self.logger.info('Dust is set in [%.1f:%.1f]/mag. Initial value is set to %.1f'%(self.Avmin,self.Avmax,self.Avini))
                fit_params.add('AV0', value=self.Avini, min=self.Avmin, max=self.Avmax)

        #
        # Metallicity;
        #
        if int(self.inputs['ZEVOL']) == 1:
            for aa in range(len(self.age)):
                if self.age[aa] == 99 or self.age[aa]>agemax:
                    fit_params.add('Z'+str(aa), value=self.Zmin, vary=False)
                    self.ndim -= 1
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


    def check_mainbody(self):
        '''
        To check any issues with the input params.
        '''
        if len(self.age) != len(set(self.age)):
            msg = 'Input age has duplications. Check `AGE`.'
            print_err(msg, exit=True, details=None)
        

    def prepare_class(self, add_fir=None):
        '''
        '''
        self.logger.info('Preparing library')
        # Load Spectral library;
        self.lib = self.fnc.open_spec_fits(fall=0)
        self.lib_all = self.fnc.open_spec_fits(fall=1, orig=True)

        if self.f_dust:
            self.lib_dust = self.fnc.open_spec_dust_fits(fall=0)
            self.lib_dust_all = self.fnc.open_spec_dust_fits(fall=1)
        if self.fneb:
            self.lib_neb = self.fnc.open_spec_fits(fall=0, f_neb=True)
            self.lib_neb_all = self.fnc.open_spec_fits(fall=1, f_neb=True)
        if self.fagn:
            self.lib_agn = self.fnc.open_spec_fits(fall=0, f_agn=True)
            self.lib_agn_all = self.fnc.open_spec_fits(fall=1, f_agn=True)

        if add_fir == None:
            add_fir = self.f_dust

        # For MCMC;
        self.nmc = int(self.inputs['NMC'])
        self.nwalk = int(self.inputs['NWALK'])
        self.nmc_cz = int(self.inputs['NMCZ'])
        self.nwalk_cz = int(self.inputs['NWALKZ'])
        self.ZEVOL = int(self.inputs['ZEVOL'])
        nzvis = int(self.inputs['ZVIS'])
        if nzvis == 1:
            self.fzvis = True
        else:
            self.fzvis = False
        if self.f_nested:
            self.logger.warning('Nested sample is on. Nelder is used to save time')
            self.fneld = 1 

        try:
            self.ntemp = int(self.inputs['NTEMP'])
        except:
            self.ntemp = 1

        try:
            if int(self.inputs['DISP']) == 1:
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
        amp : amplitude, 0 to 1.
        
        Shuffles initial parameters of each walker, to give it extra randomeness.
        '''
        if amp>1:
            amp = 1
        pos = np.zeros((self.nwalk, self.ndim), 'float')
        for ii in range(pos.shape[0]):
            aa = 0
            for aatmp,key in enumerate(out.params.valuesdict()):
                if out.params[key].vary == True:
                    pos[ii,aa] += out.params[key].value
                    # This is critical to avoid parameters fall on the boundary.
                    delpar = (out.params[key].max-out.params[key].min) * amp/2.
                    # or not,
                    # delpar = 0
                    if np.random.uniform(0,1) > (1. - 1./self.ndim):        
                        pos[ii,aa] = np.random.uniform(out.params[key].value-delpar, out.params[key].value+delpar)
                    else:
                        if pos[ii,aa]<out.params[key].min or pos[ii,aa]>out.params[key].max:
                            pos[ii,aa] = np.random.uniform(out.params[key].value-delpar, out.params[key].value+delpar)

                    aa += 1
        return pos


    def main(self, cornerplot:bool=True, specplot=1, sigz=1.0, ezmin=0.01, ferr=0,
            f_move:bool=False, verbose:bool=False, skip_fitz:bool=True, out=None, f_plot_accept:bool=True,
            f_shuffle:bool=True, amp_shuffle=1e-2, check_converge:bool=True, Zini=None, f_plot_chain:bool=True,
            f_chind:bool=True, ncpu:int=0, f_prior_sfh:bool=False, norder_sfh_prior:int=3, include_photometry=True
            ):
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
        # Call likelihood/prior/posterior function;
        class_post = Post(self)

        # Prepare library, data, etc.
        self.prepare_class()

        # Check Main Body;
        self.check_mainbody()

        # Check if zmc is a free param;
        if skip_fitz and self.fzmc==1:
            self.logger.warning('ZMC is 1; skip_fitz is set to False.')
            skip_fitz = False

        print('########################')
        print('### Fitting Function ###')
        print('########################')
        start = timeit.default_timer()        
        if not os.path.exists(self.DIR_TMP):
            os.mkdir(self.DIR_TMP)

        # Initial Z:
        if Zini == None:
            Zini = self.Zall

        # Uplim;
        f_chind = self.f_chind

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

            _, fm_tmp, xm_tmp = self.fnc.get_template(out, f_val=True, f_nrd=True, f_neb=False)
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
            flag_z = self.fit_redshift(xm_tmp, fm_tmp, delzz=0.001, include_photometry=include_photometry, fzvis=self.fzvis)

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

                # Fix params to what we had before.
                if self.fzmc:
                    out.params['zmc'].value = self.zgal
                out.params['AV0'].value = out_keep.params['AV0'].value
                for aa in range(len(self.age)):
                    out.params['A'+str(aa)].value = out_keep.params['A'+str(aa)].value
                    try:
                        out.params['Z'+str(aa)].value = out_keep.params['Z'+str(aa)].value
                    except:
                        out.params['Z0'].value = out_keep.params['Z0'].value

            ##############################
            self.logger.info('Input redshift is adopted')
            self.logger.info('Starting MCMC')
            # self.logger.info('Minimizer Defined')
            # self.logger.info('Starting sampling')
            start_mc = timeit.default_timer()

            # MCMC;
            if self.f_mcmc or self.f_zeus:
                nburn = int(self.nmc/2)
                if f_shuffle and not self.f_zeus:
                    # ZEUS may fail to run with f_shuffle.
                    pos = self.get_shuffle(out, amp=amp_shuffle)
                else:
                    pos = amp_shuffle * np.random.randn(self.nwalk, self.ndim)
                    pos += self.get_shuffle(out, amp=0)
                    # Check boundary;
                    aa = 0
                    for aatmp,key in enumerate(out.params.valuesdict()):
                        if out.params[key].vary:
                            con = (out.params[key].min > pos[:,aa])
                            pos[:,aa][con] = out.params[key].min
                            con = (out.params[key].max < pos[:,aa])
                            pos[:,aa][con] = out.params[key].max
                            # pos[:,aa] = out.params[key].value
                            aa += 1

                if self.f_zeus:
                    self.logger.info('sampling with ZEUS')
                    check_converge = False
                    f_burnin = True
                    if f_burnin:
                        # Burn phase;
                        moves = zeus.moves.DifferentialMove() #GlobalMove()
                        sampler = zeus.EnsembleSampler(self.nwalk, self.ndim, class_post.lnprob_emcee, \
                            args=[out.params, self.dict['fy'], self.dict['ey'], self.dict['wht2'], self.dict['NR'], self.f_dust], \
                            moves=moves, maxiter=1e6,\
                            kwargs={'f_val':True, 'out':out, 'lnpreject':-np.inf, 'f_chind':f_chind, 
                            'f_prior_sfh':f_prior_sfh, 'norder_sfh_prior':norder_sfh_prior, 'SNlim':self.SNlim, 'NRbb_lim':self.NRbb_lim},\
                            )
                        # Run MCMC
                        nburn = int(self.nmc/10)

                        self.logger.info('Running burn-in')
                        sampler.run_mcmc(pos, nburn)
                        self.logger.info('Done burn-in')

                        # Get the burnin samples
                        burnin = sampler.get_chain()

                        # Set the new starting positions of walkers based on their last positions
                        pos = burnin[-1]

                    # Switch sampler;
                    moves = zeus.moves.GlobalMove()
                    sampler = zeus.EnsembleSampler(self.nwalk, self.ndim, class_post.lnprob_emcee, \
                        args=[out.params, self.dict['fy'], self.dict['ey'], self.dict['wht2'], self.dict['NR'], self.f_dust], \
                        moves=moves, maxiter=1e4,\
                        kwargs={'f_val':True, 'out':out, 'lnpreject':-np.inf, 
                        'f_prior_sfh':f_prior_sfh, 'norder_sfh_prior':norder_sfh_prior, 'SNlim':self.SNlim, 'NRbb_lim':self.NRbb_lim},\
                        )

                else:
                    self.logger.info('sampling with EMCEE')
                    moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2),]
                    sampler = emcee.EnsembleSampler(self.nwalk, self.ndim, class_post.lnprob_emcee, \
                        args=(out.params, self.dict['fy'], self.dict['ey'], self.dict['wht2'], self.dict['NR'], self.f_dust),\
                        #moves=moves,\
                        kwargs={'f_val': True, 'out': out, 'lnpreject':-np.inf, 
                        'f_prior_sfh':f_prior_sfh, 'norder_sfh_prior':norder_sfh_prior, 'SNlim':self.SNlim, 'NRbb_lim':self.NRbb_lim},\
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
                            self.logger.info('Converged at %d/%d\n'%(index*nevery,self.nmc))
                            nburn = int(index*nevery / 50) # Burn 50%
                            self.nmc = index*nevery
                            break
                        old_tau = tau
                else:
                    sampler.run_mcmc(pos, self.nmc, progress=True)
                
                flat_samples = sampler.get_chain(discard=nburn, thin=10, flat=True)

                # Plot for chain.
                if f_plot_chain:
                    _, axes = plt.subplots(self.ndim, figsize=(10, 7), sharex=True)
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
                    plt.savefig('%s/gsf_chain_%s.png'%(self.DIR_OUT,self.ID))
                    plt.close()
                    # For memory optimization;
                    del samples, axes


                # Similar for nested;
                # Dummy just to get structures;
                print('\nRunning dummy sampler. Disregard message from here;\n')
                mini = Minimizer(class_post.lnprob_emcee, out.params, 
                fcn_args=[out.params, self.dict['fy'], self.dict['ey'], self.dict['wht2'], self.dict['NR'], self.f_dust], 
                f_disp=False, nan_policy='omit',
                moves=moves,\
                kwargs={'f_val': True, 'NRbb_lim':self.NRbb_lim},\
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
                msg = 'Sampling is not specified. Failed. Exiting.'
                print_err(msg, exit=True)

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

            use_pickl = False
            use_pickl = True
            if use_pickl:
                cpklname = 'gsf_chain_' + self.ID + '.cpkl'
                savecpkl({'chain':flatchain,
                              'burnin':burnin, 'nwalkers':self.nwalk,'niter':self.nmc,'ndim':self.ndim},
                             savepath+cpklname) # Already burn in
            else:
                cpklname = 'gsf_chain_' + self.ID + '.asdf'
                tree = {'chain':flatchain.to_dict(), 'burnin':burnin, 'nwalkers':self.nwalk,'niter':self.nmc,'ndim':self.ndim}
                af = asdf.AsdfFile(tree)
                af.write_to(savepath+cpklname, all_array_compression='zlib')

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

                fig1.savefig(self.DIR_OUT + 'gsf_corner_' + self.ID + '.png')
                self.cornerplot_fig = fig1

            # Analyze MCMC results.
            # Write to file.
            stop = timeit.default_timer()
            tcalc = stop - start

            # Then writing;
            get_param(self, res, fitc, tcalc=tcalc, burnin=burnin)

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
            self.logger.info('Generating model templates with input redshift and Scale.')
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
                self.logger.info('Terminating process.')
                return False


    def quick_fit(self, specplot=1, sigz=1.0, ezmin=0.01, ferr=0, f_move=False, 
        f_get_templates=False, Zini=None, include_photometry=True, f_only_spec=False):
        '''Fits input data with a prepared template library, to get a chi-min result. 
        This function is being used in an example notebook.

        Parameters:
        -----------
        Zini : array
            Array for initial values for metallicity.

        Returns
        -------
        out, chidef, Zbest, xm_tmp, fm_tmp (if f_get_templates is set True).
        '''
        print('#########')
        print('Quick fit')
        print('#########\n')

        # Call likelihood/prior/posterior function;
        class_post = Post(self)

        # Prepare library, data, etc.
        self.prepare_class()

        # Initial Z:
        if Zini == None:
            Zini = self.Zall

        # Temporarily disable zmc;
        self.fzmc = 0
        out, chidef, Zbest = get_leastsq(self, Zini, self.fneld, self.age, self.fit_params, class_post.residual,\
            self.dict['fy'], self.dict['ey'], self.dict['wht2'], self.ID, f_only_spec=f_only_spec)

        if f_get_templates:
            Av_tmp = out.params['AV0'].value
            AA_tmp = np.zeros(len(self.age), dtype='float')
            ZZ_tmp = np.zeros(len(self.age), dtype='float')
            # fm_tmp, xm_tmp = self.fnc.tmp04(out, f_val=True)
            fm_tmp, xm_tmp = self.fnc.get_template(out, f_val=True, f_nrd=False, f_neb=False)

            ########################
            # Check redshift
            ########################
            if self.fzvis:
                flag_z = self.fit_redshift(xm_tmp, fm_tmp, delzz=0.001, include_photometry=include_photometry, fzvis=self.fzvis)

            self.fzmc = 1
            return out, chidef, Zbest, xm_tmp, fm_tmp
        else:
            self.fzmc = 1
            return out, chidef, Zbest


    def search_redshift(self, dict, xm_tmp, fm_tmp, zliml=0.01, zlimu=6.0, delzz=0.01, 
                        lines=False, prior=None, method='powell', include_photometry=False, f_plot=True):
        '''
        This module explores the redshift space to find the best redshift and probability distribution.

        Parameters
        ----------
        dict : dictionary
            Dictionary that includes input data.

        xm_tmp : numpy.array
            Wavelength array, common for fm_tmp below, at z=0. Should be in [len(wavelength)].
        fm_tmp : numpy.array
            Fluxes for various templates. Should be in a shape of [ n * len(wavelength)], 
            where n is the number templates.
        zliml : float
            Lowest redshift for fitting range.
        zlimu : float
            Highest redshift for fitting range.
        prior : numpy.array
            Prior used for the redshift determination. E.g., Eazy z-probability.
        method : str
            Method for minimization. The option must be taken from lmfit. Powell is more accurate. Nelder is faster.

        Returns
        -------
        zspace : numpy.array 
            Array for redshift grid.
        chi2s : numpy.array
            Array of chi2 values corresponding to zspace.
        '''
        zspace = np.arange(zliml,zlimu,delzz)
        chi2s = np.zeros((len(zspace),2), float)
        if prior == None:
            prior = zspace[:] * 0 + 1.0

        data_len = self.data['meta']['data_len']

        NR = dict['NR']
        con0 = (NR<data_len[0])
        x0 = dict['x'][con0]
        fy0 = dict['fy'][con0]
        ey0 = dict['ey'][con0]
        con1 = (NR>=data_len[0]) & (NR<data_len[1]+data_len[0])
        x1 = dict['x'][con1]
        fy1 = dict['fy'][con1]
        ey1 = dict['ey'][con1]
        if include_photometry:
            con2 = (NR>=data_len[1]+data_len[0])
        else:
            con2 = (NR>=data_len[1]+data_len[0]) & (NR<self.NRbb_lim)
        x2 = dict['x'][con2]
        fy2 = dict['fy'][con2]
        ey2 = dict['ey'][con2]

        fy01 = np.append(fy0,fy1)
        fcon = np.append(fy01,fy2)
        ey01 = np.append(ey0,ey1)
        eycon = np.append(ey01,ey2)
        x01 = np.append(x0,x1)
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
            ''''''
            vals  = pars.valuesdict()
            xm_s = xm_tmp * (1+z)
            fm_s = np.zeros(len(xm_tmp),float)

            for nn in range(len(fm_tmp[:,0])):
                fm_s += fm_tmp[nn,:] * pars['C%d'%nn]

            fint = interpolate.interp1d(xm_s, fm_s, kind='nearest', fill_value="extrapolate")
            fm_int = fint(xobs)

            if fcon is None:
                self.logger.warning('Data is none')
                return fm_int
            else:
                return (fm_int - fcon) * np.sqrt(wht2) # i.e. residual/sigma

        # Start redshift search;
        for zz in range(len(zspace)):
            out_cz = minimize(residual_z, fit_par_cz, args=([zspace[zz]]), method=method)
            keys = fit_report(out_cz).split('\n')

            csq  = out_cz.chisqr
            rcsq = out_cz.redchi
            fitc_cz = [csq, rcsq]

            chi2s[zz,0] = csq
            chi2s[zz,1] = rcsq

        self.zspace = zspace
        self.chi2s = chi2s

        if f_plot:
            plt.close()
            fig = plt.figure(figsize=(5,2.5))
            ax1 = fig.add_subplot(111)
            ax1.plot(zspace,chi2s[:,1])
            ax1.set_ylabel('Reduced-$\chi^2$',fontsize=18)
            ax1.set_xlabel('$z$',fontsize=18)
            ax1.set_title('Redshift Fitting Result')
            plt.savefig(os.path.join(self.DIR_OUT, 'zchi_%s.png'%self.ID), dpi=200)

        return zspace, chi2s


    def plot_fit_result(self, out, xmin=2000, xmax=80000):
        '''
        out : class object
            From minimizer.
        '''
        mb = self
        # Read data with the current best-fit Cs.
        dict = mb.read_data(mb.Cz0, mb.Cz1, mb.Cz2, mb.zgal)

        plt.close()
        fig = plt.figure(figsize=(8,2.8))
        ax1 = fig.add_subplot(111)

        # Generate the best-fit model;
        flux_all, wave_all = mb.fnc.get_template(out, f_val=True, lib_all=True)

        # Template
        ax1.errorbar(wave_all, flux_all, ls='-', color='b', zorder=0, label='Best-fit model')

        # plot;
        if self.has_photometry:
            ax1.scatter(dict['xbb'], dict['fybb'], marker='o', c='orange', edgecolor='k', s=150, zorder=2, alpha=1, label='Observed photometry')

        if self.has_spectrum:
            ax1.errorbar(dict['x'], dict['fy'], yerr=dict['ey'], ls='', color='gray', zorder=1, alpha=0.3)
            ax1.scatter(dict['x'], dict['fy'], marker='o', color='r',edgecolor='r', s=10, zorder=1, alpha=1, label='Observed spectrum')

        ax1.set_xlim(xmin, xmax)
        ax1.set_xscale('log')

        ax1.legend(loc=2)
        ax1.set_xlabel('Wavelength [$\mathrm{\AA}$]')
        ax1.set_ylabel('Flux$_\\nu$ (arb.)')
        return fig
    
    @staticmethod
    def check_keys(self):
        '''
        '''
        self.input_keys = []
        for key in self.inputs.keys():
            self.input_keys.append(key)
        return self.input_keys