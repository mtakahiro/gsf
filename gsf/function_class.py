import numpy as np
import sys
import scipy.interpolate as interpolate
import asdf
from astropy.io import fits

from .function import *
from .basic_func import Basic
from .function_igm import dijkstra_igm_abs
from .maketmp_filt import maketemp,maketemp_tau


class Func:
    '''
    The list of (possible) `Func` attributes is given below:

    Attributes
    ----------
    '''
    def __init__(self, MB):
        '''
        Parameters
        ----------
        dust_model : int
            0 for Calzetti.
        '''
        self.ID = MB.ID
        self.ZZ = MB.Zall
        self.age = MB.age
        self.AA = MB.nage
        self.tau0 = MB.tau0
        self.MB = MB

        self.dust_model = MB.dust_model
        self.DIR_TMP = MB.DIR_TMP

        if MB.f_dust:
            self.Temp = MB.Temp

        try:
            self.filts = MB.filts
            self.DIR_FIL = MB.DIR_FILT
        except:
            pass

        # Already Read or not;
        self.f_af = False
        self.f_af0 = False


    def open_spec_fits(self, fall:int=0, orig:bool=False, f_neb=False, f_agn=False):
        '''Load template in obs range.

        Parameters
        ----------
        fall : int
            If 1, returns full spectra.
        orig : :obj:`bool`
            If True, returns the original spectra.

        Returns
        -------
        lib : float array
            
        '''
        ID0 = self.MB.ID
        DIR_TMP = self.DIR_TMP
        ZZ = self.ZZ
        AA = self.AA
        bfnc = self.MB.bfnc
        tau0 = np.arange(0,1,1)
        ntau0 = len(tau0)

        # ASDF;
        if fall == 0:
            app = ''
            hdu0 = self.MB.af['spec']
        elif fall == 1:
            app = 'all_'
            hdu0 = self.MB.af['spec_full']

        if f_neb:
            NZ = len(ZZ)
            NU = len(self.MB.logUs)
            for zz,Z in enumerate(ZZ):
                for uu,logU in enumerate(self.MB.logUs):
                    if zz == 0 and uu == 0:
                        nr = hdu0['colnum']
                        xx = hdu0['wavelength']
                        coln = int(2 + NZ * NU)
                        lib = np.zeros((len(nr), coln), dtype=float)
                        lib[:,0] = nr[:]
                        lib[:,1] = xx[:]
                    if orig:
                        colname = 'fspec_orig_nebular_Z%d'%zz + '_logU%d'%uu
                    else:
                        colname = 'fspec_nebular_Z%d'%zz + '_logU%d'%uu
                    colnall = int(2 + zz * NU + uu) # 2 takes account of wavelength and AV columns.
                    lib[:,colnall] = hdu0[colname]
        elif f_agn:
            NZ = len(ZZ)
            NU = len(self.MB.AGNTAUs)
            for zz,Z in enumerate(ZZ):
                for uu,logU in enumerate(self.MB.AGNTAUs):
                    if zz == 0 and uu == 0:
                        nr = hdu0['colnum']
                        xx = hdu0['wavelength']
                        coln = int(2 + NZ * NU)
                        lib = np.zeros((len(nr), coln), dtype=float)
                        lib[:,0] = nr[:]
                        lib[:,1] = xx[:]
                    if orig:
                        colname = 'fspec_orig_agn_Z%d'%zz + '_AGNTAU%d'%uu
                    else:
                        colname = 'fspec_agn_Z%d'%zz + '_AGNTAU%d'%uu
                    colnall = int(2 + zz * NU + uu) # 2 takes account of wavelength and AV columns.
                    lib[:,colnall] = hdu0[colname]
        else:
            for pp in range(ntau0):
                for zz in range(len(ZZ)):
                    Z = ZZ[zz]
                    NZ = bfnc.Z2NZ(Z)
                    if zz == 0 and pp == 0:
                        nr = hdu0['colnum']
                        xx = hdu0['wavelength']
                        lib = np.zeros((len(nr), 2+len(AA)*len(ZZ)*len(tau0)), dtype='float')
                        lib[:,0] = nr[:]
                        lib[:,1] = xx[:]

                    for aa in range(self.MB.npeak):
                        coln = int(2+aa)
                        if orig:
                            colname = 'fspec_orig_' + str(zz) + '_' + str(aa) + '_' + str(pp)
                        else:
                            colname = 'fspec_' + str(zz) + '_' + str(aa) + '_' + str(pp)
                        colnall = int(2 + pp*len(ZZ)*len(AA) + zz*len(AA) + aa) # 2 takes account of wavelength and AV columns.
                        lib[:,colnall] = hdu0[colname]

        return lib


    def open_spec_dust_fits(self, fall:int = 0):
        '''Loads dust template in obs range.
        '''
        if fall == 0:
            hdu0 = self.MB.af['spec_dust']
        elif fall == 1:
            hdu0 = self.MB.af['spec_dust_full']

        nr = hdu0['colnum']
        xx = hdu0['wavelength']
        
        lib = np.zeros((len(nr), 2+len(self.Temp)), dtype='float')
        lib[:,0] = nr[:]
        lib[:,1] = xx[:]

        for aa in range(len(self.Temp)):
            coln = int(2 + aa)
            colname = 'fspec_' + str(aa)
            colnall = int(2 + aa)
            lib[:,colnall] = hdu0[colname]
        return lib


    def get_total_flux(self, par, f_Alog=True, lib_all=True, pp=0, lib=None, f_get_Mtot=False, f_check_limit=True):
        '''get total flux for a given set of parameter.

        Parameters
        ----------
        par : library
            contains parameters. Needs to include: ['Z','A']

        Returns
        -------
        nr, xx, yy : float arrays
            xx is OBSERVED wavelength at z=MB.zgal
        '''
        Mtot:float = 0
        if lib == None:
            if lib_all:
                lib = self.MB.lib_all
            else:
                lib = self.MB.lib

        for aa in range(self.MB.npeak):
            if self.MB.ZEVOL==1 or aa == 0:
                Z = par['Z'+str(aa)]
                NZ = self.MB.bfnc.Z2NZ(Z)

            if f_check_limit:
                # Check limit;
                if par['A'+str(aa)] < self.MB.Amin:
                    par['A'+str(aa)] = self.MB.Amin
                if par['A'+str(aa)] > self.MB.Amax:
                    par['A'+str(aa)] = self.MB.Amax
                # Z limit:
                if aa == 0 or self.MB.ZEVOL == 1:
                    if par['Z%d'%aa] < self.MB.Zmin:
                        par['Z%d'%aa] = self.MB.Zmin
                    if par['Z%d'%aa] > self.MB.Zmax:
                        par['Z%d'%aa] = self.MB.Zmax

            # Is A in logspace?
            if f_Alog:
                A00 = 10**par['A'+str(aa)]
            else:
                A00 = par['A'+str(aa)]

            coln = int(2 + pp*len(self.ZZ)*len(self.AA) + NZ*len(self.AA) + aa)

            if f_get_Mtot:
                mslist = self.MB.af['ML']['ML_'+str(NZ)][aa]
                Mtot += 10**(par['A%d'%aa] + np.log10(mslist))

            if aa == 0:
                nr = lib[:,0]
                xx = lib[:,1] # This is OBSERVED wavelength range at z=zgal
                yy = A00 * lib[:,coln]
            else:
                yy += A00 * lib[:,coln]

        if f_get_Mtot:
            return nr, xx, yy, Mtot
        else:
            return nr, xx, yy


    def get_total_flux_agn(self, par, f_Alog=True, lib_all=True, lib=None):
        '''get total flux for a given set of parameter.

        Parameters
        ----------
        par : library
            contains parameters
        '''
        if lib == None:
            if lib_all:
                lib = self.MB.lib_agn_all
            else:
                lib = self.MB.lib_agn

        aa = 0
        if self.MB.ZEVOL==1 or aa == 0:
            Z = par['Z'+str(aa)]
            NZ = self.MB.bfnc.Z2NZ(Z)

        try:
            Aagn = par['Aagn']
            AGNTAU = par['AGNTAU']
            nAGNTAU = np.argmin(np.abs(self.MB.AGNTAUs - AGNTAU))
        except: # This is exception for initial minimizing;
            Aagn = -99
            AGNTAU = self.MB.AGNTAUs[0]
            nAGNTAU = 0

        # AGNTAU
        NAGNT = self.MB.nAGNTAU

        # Check limit;
        if Aagn < self.MB.Amin:
            Aagn = self.MB.Amin
        if Aagn > self.MB.Amax:
            Aagn = self.MB.Amax

        # Z limit:
        if aa == 0 or self.MB.ZEVOL == 1:
            if par['Z%d'%aa] < self.MB.Zmin:
                par['Z%d'%aa] = self.MB.Zmin
            if par['Z%d'%aa] > self.MB.Zmax:
                par['Z%d'%aa] = self.MB.Zmax

        # Is A in logspace?
        if f_Alog:
            A00 = 10**Aagn
        else:
            A00 = Aagn

        coln = int(2 + NZ*NAGNT + nAGNTAU)

        if aa == 0:
            nr = lib[:,0]
            xx = lib[:,1] # This is OBSERVED wavelength range at z=zgal
            yy = A00 * lib[:,coln]
        # else:
        #     yy += A00 * lib[:,coln]

        # if True:#lib_all:
        #     import matplotlib.pyplot as plt
        #     plt.close()
        #     plt.plot(xx, yy, ls='None', marker='o')
        #     plt.show()
        #     hoge

        return nr, xx, yy


    def get_total_flux_neb(self, par, f_Alog=True, lib_all=True, lib=None):
        '''get total flux for a given set of parameter.

        Parameters
        ----------
        par : library
            contains parameters
        '''
        if lib is None:
            if lib_all:
                lib = self.MB.lib_neb_all
            else:
                lib = self.MB.lib_neb

        aa = 0
        if self.MB.ZEVOL==1 or aa == 0:
            Z = par['Z'+str(aa)]
            NZ = self.MB.bfnc.Z2NZ(Z)

        try:
            Aneb = par['Aneb']
            logU = par['logU']
            nlogU = np.argmin(np.abs(self.MB.logUs - logU))
        except: # This is exception for initial minimizing;
            Aneb = -99
            logU = self.MB.logUs[0]
            nlogU = 0

        # try:
        #     Aagn = par['Aagn']
        #     AGNTAU = par['AGNTAU']
        #     nAGNTAU = np.argmin(np.abs(self.MB.AGNTAUs - AGNTAU))
        # except: # This is exception for initial minimizing;
        #     Aagn = -99
        #     AGNTAU = self.MB.AGNTAUs[0]
        #     nAGNTAU = 0

        # logU
        NU = self.MB.nlogU
        # # AGNTAU
        # NAGNT = self.MB.nAGNTAU

        # Check limit;
        if Aneb < self.MB.Amin:
            Aneb = self.MB.Amin
        if Aneb > self.MB.Amax:
            Aneb = self.MB.Amax

        # if Aagn < self.MB.Amin:
        #     Aagn = self.MB.Amin
        # if Aagn > self.MB.Amax:
        #     Aagn = self.MB.Amax

        # Z limit:
        if aa == 0 or self.MB.ZEVOL == 1:
            if par['Z%d'%aa] < self.MB.Zmin:
                par['Z%d'%aa] = self.MB.Zmin
            if par['Z%d'%aa] > self.MB.Zmax:
                par['Z%d'%aa] = self.MB.Zmax

        # Is A in logspace?
        if f_Alog:
            A00 = 10**Aneb
            # Aagn00 = 10**Aagn
        else:
            A00 = Aneb
            # Aagn00 = Aagn

        coln = int(2 + NZ*NU + nlogU)

        if aa == 0:
            nr = lib[:,0]
            xx = lib[:,1] # This is OBSERVED wavelength range at z=zgal
            yy = A00 * lib[:,coln]
        # else:
        #     yy += A00 * lib[:,coln]

        return nr, xx, yy


    def get_template_single(self, A00, Av, nmodel, Z, zgal, lib, logU=None, AGNTAU=None, f_apply_dust=True, EBVratio=2.27,
                            f_apply_igm=True, xhi=None):
        '''
        Parameters
        ----------
        EBVratio : float
            E(B-V)_neb / E(B-V)_st. 
            Useful table in https://iopscience.iop.org/article/10.3847/1538-4357/aba35e/pdf

        Returns
        -------
        A00 * yyd_sort, xxd_sort : float arrays
            Flux (fnu) and wavelength (AA; observed frame)

        Notes
        -----
        This function is only used in plot_sed.py.
        Common function for mebular and nonnebular temlates.
        '''
        tau0 = self.tau0
        ZZ = self.ZZ
        AA = self.AA
        bfnc = self.MB.bfnc
        DIR_TMP = self.MB.DIR_TMP
        NZ = bfnc.Z2NZ(Z)

        pp0 = np.random.uniform(low=0, high=len(tau0), size=(1,))
        pp = int(pp0[0])
        if pp>=len(tau0):
            pp += -1

        if logU != None:
            NU = len(self.MB.logUs)
            # Dust attenuation to nebulae
            Av *= EBVratio
            nlogU = np.argmin(np.abs(self.MB.logUs - logU))
            coln = int(2 + NZ*NU + nlogU)
        elif AGNTAU != None:
            NU = len(self.MB.AGNTAUs)
            nAGNTAU = np.argmin(np.abs(self.MB.AGNTAUs - AGNTAU))
            coln = int(2 + NZ*NU + nAGNTAU)
        else:
            coln = int(2 + pp*len(ZZ)*len(AA) + NZ*len(AA) + nmodel)

        nr = lib[:,0]
        xx = lib[:,1] # This is OBSERVED wavelength range at z=zgal
        yy = lib[:,coln]

        if f_apply_igm:
            if xhi == None:
                xhi = self.MB.x_HI_input
            yy, x_HI = dijkstra_igm_abs(xx/(1+zgal), yy, zgal, cosmo=self.MB.cosmo, x_HI=xhi)
            self.MB.x_HI = x_HI

        if f_apply_dust:
            yyd, xxd, nrd = apply_dust(yy, xx/(1+zgal), nr, Av, dust_model=self.dust_model)
        else:
            yyd, xxd, nrd = yy, xx/(1+zgal), nr

        xxd *= (1.+zgal)

        nrd_yyd = np.zeros((len(nrd),3), dtype='float')
        nrd_yyd[:,0] = nrd[:]
        nrd_yyd[:,1] = yyd[:]
        nrd_yyd[:,2] = xxd[:]

        b = nrd_yyd
        nrd_yyd_sort = b[np.lexsort(([-1,1]*b[:,[1,0]]).T)]
        yyd_sort = nrd_yyd_sort[:,1]
        xxd_sort = nrd_yyd_sort[:,2]

        return A00 * yyd_sort, xxd_sort


    def get_template(self, par, f_Alog:bool=True, nprec:int=1, f_val:bool=False, lib_all:bool=False, f_nrd:bool=False, 
        f_apply_dust:bool=True, f_apply_igm=True, xhi=None, deltaz_lim=0.1, f_neb=False, EBVratio:float=2.27, f_agn=False):
        '''Makes model template for a given parameter set, ``par``.

        Parameters
        ----------
        nprec : int
            Precision when redshift is refined. 
        f_apply_dust : bool
            Apply dust attenuation to nebular emission.
        EBVratio : float
            E(B-V)_neb / E(B-V)_st. 
            Useful table in https://iopscience.iop.org/article/10.3847/1538-4357/aba35e/pdf
        f_neb : bool
            Expect to explore nebular template or not.

        Notes
        -----
        This function is only used in plot_sed.py.
        Common function for mebular and nonnebular temlates.
        '''
        ZZ = self.ZZ
        AA = self.AA
        bfnc = self.MB.bfnc

        if f_val:
            par = par.params

        if self.MB.fzmc == 1:
            try:
                zmc = par['zmc'].value
            except:
                zmc = self.MB.zgal
        else:
            zmc = self.MB.zgal

        # AV limit;
        if par['AV0'] < self.MB.Avmin:
            par['AV0'] = self.MB.Avmin
        if par['AV0'] > self.MB.Avmax:
            par['AV0'] = self.MB.Avmax
        Av = par['AV0']

        if f_neb:
            # Dust attenuation to nebulae
            Av *= EBVratio

        if f_neb:
            # Get total flux;
            nr, xx, yy = self.get_total_flux_neb(par, f_Alog=f_Alog, lib_all=lib_all)
        elif f_agn:
            nr, xx, yy = self.get_total_flux_agn(par, f_Alog=f_Alog, lib_all=lib_all)
        else:
            nr, xx, yy = self.get_total_flux(par, f_Alog=f_Alog, lib_all=lib_all)

        # @@@ Filter convolution may need to happpen here
        if round(zmc,nprec) != round(self.MB.zgal,nprec):
            if np.abs(zmc - self.MB.zgal) > deltaz_lim:
                # print('!!! zmc (%.3f) is exploring too far from zgal (%.3f).'%(zmc, self.MB.zgal))

                # @@@ This only work for BB only data set.
                # Get total flux;
                if lib_all:
                    nr_full, xx_full, yy_full = nr, xx, yy
                else:
                    if f_neb:
                        nr_full, xx_full, yy_full = self.get_total_flux_neb(par, f_Alog=f_Alog, lib_all=True)
                    elif f_agn:
                        nr_full, xx_full, yy_full = self.get_total_flux_agn(par, f_Alog=f_Alog, lib_all=True)
                    else:
                        nr_full, xx_full, yy_full = self.get_total_flux(par, f_Alog=f_Alog, lib_all=True)

                xx_s, yy_s = filconv(self.MB.filts, xx_full / (1+self.MB.zgal) * (1+zmc), yy_full, self.MB.DIR_FILT, MB=self.MB, f_regist=False)
                
                if self.MB.f_spec:
                    con_bb = (nr>=self.MB.NRbb_lim)
                    xx[con_bb] = xx_s
                    yy[con_bb] = yy_s
                    xx_s = xx
                    yy_s = yy

            else:
                fint = interpolate.interp1d(xx, yy, kind='nearest', fill_value="extrapolate")
                xx_s = xx / (1+self.MB.zgal) * (1+zmc)
                yy_s = fint(xx_s)

        else:
            xx_s = xx
            yy_s = yy

        xx = xx_s
        yy = yy_s

        if f_apply_igm:
            if xhi == None:
                xhi = self.MB.x_HI_input
            yy, x_HI = dijkstra_igm_abs(xx / (1+zmc), yy, zmc, cosmo=self.MB.cosmo, x_HI=xhi)
            self.MB.x_HI = x_HI

        if f_apply_dust:
            yyd, xxd, nrd = apply_dust(yy, xx/(1+zmc), nr, Av, dust_model=self.dust_model)
            xxd *= (1.+zmc)

            if self.dust_model != 0:
                nrd_yyd = np.zeros((len(nrd),3), dtype=float)
                nrd_yyd[:,0] = nrd[:]
                nrd_yyd[:,1] = yyd[:]
                nrd_yyd[:,2] = xxd[:]
                nrd_yyd_sort = nrd_yyd[nrd_yyd[:,0].argsort()]
                nrd[:],yyd[:],xxd[:] = nrd_yyd_sort[:,0],nrd_yyd_sort[:,1],nrd_yyd_sort[:,2]

            if not f_nrd:
                return yyd[:],xxd[:]
            else:
                return nrd[:],yyd[:],xxd[:]

        else:
            if not f_nrd:
                return yy,xx
            else:
                return nr,yy,xx


    def tmp04_dust(self, par, nprec=1, return_full=False):
        '''
        Makes model template with a given param setself.
        Also dust attenuation.
        '''
        try:
            m_dust = par['MDUST']
            t_dust = par['TDUST']
        except: # This is exception for initial minimizing;
            m_dust = -99
            t_dust = 0

        if return_full:
            lib = self.MB.lib_dust_all
        else:
            lib = self.MB.lib_dust

        nr = lib[:,0]
        xx = lib[:,1] # This is OBSERVED wavelength range at z=zgal
        coln = 2+int(t_dust+0.5)
        yy = 10**m_dust * lib[:,coln]

        if self.MB.fzmc == 1:
            zmc = par.params['zmc'].value
        else:
            zmc = self.MB.zgal

        if return_full:
            return yy, xx

        # How much does this cost in time?
        if round(zmc,nprec) != round(self.MB.zgal,nprec):
            xx_s = xx / (1+self.MB.zgal) * (1+zmc)
            fint = interpolate.interp1d(xx, yy, kind='nearest', fill_value="extrapolate")
            yy_s = fint(xx_s)
        else:
            xx_s = xx
            yy_s = yy

        return yy_s, xx_s


class Func_tau:
    '''
    '''
    def __init__(self, MB):
        '''
        Parameters
        ----------
        dust_model : int
            0 for Calzetti. 1 for MW. 4 for Kriek Conroy
        '''
        self.MB = MB
        self.ID = MB.ID
        self.ZZ = MB.Zall
        self.AA = MB.nage
        self.tau = MB.tau

        self.dust_model = MB.dust_model
        self.DIR_TMP = MB.DIR_TMP

        if MB.f_dust:
            self.Temp = MB.Temp

        try:
            self.filts = MB.filts
            self.DIR_FIL = MB.DIR_FILT
        except:
            pass

        # Already Read or not;
        self.f_af = False
        self.f_af0 = False


    def open_spec_dust_fits(self, fall:int = 0):
        '''
        Load dust template in obs range.
        '''
        ID0 = self.MB.ID
        tau0= self.MB.tau0
        ZZ = self.ZZ
        AA = self.AA
        bfnc = self.MB.bfnc #Basic(ZZ)

        # self.MB.af = asdf.open(self.DIR_TMP + 'spec_all_' + self.ID + '.asdf')
        # self.MB.af0 = asdf.open(self.DIR_TMP + 'spec_all.asdf')

        if fall == 0:
            app = ''
            hdu0 = self.MB.af['spec_dust']
        elif fall == 1:
            app = 'all_'
            hdu0 = self.MB.af['spec_dust_full']

        DIR_TMP = self.DIR_TMP
        nr = hdu0['colnum']
        xx = hdu0['wavelength']
        
        lib = np.zeros((len(nr), 2+len(self.Temp)), dtype='float')
        lib[:,0] = nr[:]
        lib[:,1] = xx[:]

        for aa in range(len(self.Temp)):
            coln = int(2 + aa)
            colname = 'fspec_' + str(aa)
            colnall = int(2 + aa) # 2 takes account of wavelength and AV columns.
            lib[:,colnall] = hdu0[colname]
            if fall==1 and False:
                import matplotlib.pyplot as plt
                plt.close()
                plt.plot(lib[:,1],lib[:,coln],linestyle='-')
                plt.show()
        return lib


    def open_spec_fits(self, fall=0, orig=False, f_neb=False, f_agn=False):
        '''
        Loads template in obs range.
        '''
        ID0 = self.MB.ID
        ZZ = self.ZZ
        AA = self.AA
        bfnc = self.MB.bfnc

        # ASDF;
        if fall == 0:
            app = ''
            hdu0 = self.MB.af['spec']
        elif fall == 1:
            app = 'all_'
            hdu0 = self.MB.af['spec_full']

        DIR_TMP = self.DIR_TMP
        NZ = len(ZZ)

        if f_neb:
            NU = len(self.MB.logUs)
            for zz,Z in enumerate(ZZ):
                for uu,logU in enumerate(self.MB.logUs):
                    if zz == 0 and uu == 0:
                        nr = hdu0['colnum']
                        xx = hdu0['wavelength']
                        coln = int(2 + NZ * NU)
                        lib = np.zeros((len(nr), coln), dtype=float)
                        lib[:,0] = nr[:]
                        lib[:,1] = xx[:]
                    if orig:
                        colname = 'fspec_orig_nebular_Z%d'%zz + '_logU%d'%uu
                    else:
                        colname = 'fspec_nebular_Z%d'%zz + '_logU%d'%uu
                    colnall = int(2 + zz * NU + uu) # 2 takes account of wavelength and AV columns.
                    lib[:,colnall] = hdu0[colname]
        elif f_agn:
            NU = len(self.MB.AGNTAUs)
            for zz,Z in enumerate(ZZ):
                for uu,AGNTAU in enumerate(self.MB.AGNTAUs):
                    if zz == 0 and uu == 0:
                        nr = hdu0['colnum']
                        xx = hdu0['wavelength']
                        coln = int(2 + NZ * NU)
                        lib = np.zeros((len(nr), coln), dtype=float)
                        lib[:,0] = nr[:]
                        lib[:,1] = xx[:]
                    if orig:
                        colname = 'fspec_orig_agn_Z%d'%zz + '_logU%d'%uu
                    else:
                        colname = 'fspec_agn_Z%d'%zz + '_logU%d'%uu
                    colnall = int(2 + zz * NU + uu) # 2 takes account of wavelength and AV columns.
                    lib[:,colnall] = hdu0[colname]
        else:
            NT = self.MB.ntau
            NA = self.MB.nage
            for zz,Z in enumerate(ZZ):
                for tt,TT in enumerate(self.MB.tau):                
                    for ss,TA in enumerate(self.MB.ageparam):
                        if zz == 0 and tt == 0 and ss == 0:
                            nr = hdu0['colnum']
                            xx = hdu0['wavelength']
                            coln = int(2 + NZ * NT * NA)
                            lib = np.zeros((len(nr), coln), dtype=float)
                            lib[:,0] = nr[:]
                            lib[:,1] = xx[:]
                        if orig:
                            colname = 'fspec_orig_' + str(zz) + '_' + str(tt) + '_' + str(ss)
                        else:
                            colname = 'fspec_' + str(zz) + '_' + str(tt) + '_' + str(ss)
                        colnall = int(2 + zz * NT * NA + tt * NA + ss) # 2 takes account of wavelength and AV columns.
                        lib[:,colnall] = hdu0[colname]
        return lib


    def get_total_flux(self, par, f_Alog=True, lib_all=True, pp=0, lib=None, f_get_Mtot=False, f_check_limit=True):
        '''
        '''
        Mtot = 0
        for aa in range(self.MB.npeak):
            if self.MB.ZEVOL==1 or aa == 0:
                if f_check_limit:
                    # Z limit:
                    if par['Z%d'%aa] < self.MB.Zmin:
                        par['Z%d'%aa] = self.MB.Zmin
                    if par['Z%d'%aa] > self.MB.Zmax:
                        par['Z%d'%aa] = self.MB.Zmax
                Z = par['Z%d'%aa]
            else:
                pass

            if f_check_limit:
                # A
                if par['A'+str(aa)] < self.MB.Amin:
                    par['A'+str(aa)] = self.MB.Amin
                if par['A'+str(aa)] > self.MB.Amax:
                    par['A'+str(aa)] = self.MB.Amax

                if par['TAU'+str(aa)] < self.MB.taumin:
                    par['TAU'+str(aa)] = self.MB.taumin
                if par['TAU'+str(aa)] > self.MB.taumax:
                    par['TAU'+str(aa)] = self.MB.taumax

                if par['AGE'+str(aa)] < self.MB.agemin:
                    par['AGE'+str(aa)] = self.MB.agemin
                if par['AGE'+str(aa)] > self.MB.agemax:
                    par['AGE'+str(aa)].value = self.MB.agemax

            # Is A in logspace?
            if f_Alog:
                A00 = 10**par['A'+str(aa)]
            else:
                A00 = par['A'+str(aa)]

            tau,age = par['TAU%d'%aa],par['AGE%d'%aa]

            NZ, NT, NA = self.MB.bfnc.Z2NZ(Z,tau,age)
            coln = int(2 + NZ*self.MB.ntau*self.MB.npeak + NT*self.MB.npeak + NA)
            mslist = self.MB.af['ML']['ML_'+str(NZ)+'_'+str(NT)][NA]
            Mtot += 10**(par['A%d'%aa] + np.log10(mslist))

            if lib_all:
                if aa == 0:
                    nr = self.MB.lib_all[:, 0]
                    xx = self.MB.lib_all[:, 1] # This is OBSERVED wavelength range at z=zgal
                    yy = A00 * self.MB.lib_all[:, coln]
                else:
                    yy += A00 * self.MB.lib_all[:, coln]
            else:
                if aa == 0:
                    nr = self.MB.lib[:, 0]
                    xx = self.MB.lib[:, 1] # This is OBSERVED wavelength range at z=zgal
                    yy = A00 * self.MB.lib[:, coln]
                else:
                    yy += A00 * self.MB.lib[:, coln]

        if f_get_Mtot:
            return nr, xx, yy, Mtot
        else:
            return nr, xx, yy


    def get_total_flux_agn(self, par, f_Alog=True, lib_all=True, pp=0, lib=None, f_get_Mtot=False, f_check_limit=True):
        '''
        '''
        Mtot = 0
        try:
            Aagn = par['Aagn']
            AGNTAU = par['AGNTAU']
            nAGNTAU = np.argmin(np.abs(self.MB.AGNTAUs - AGNTAU))
        except: # This is exception for initial minimizing;
            Aagn = -99
            AGNTAU = self.MB.AGNTAUs[0]
            nAGNTAU = 0

        # logU
        NU = len(self.MB.AGNTAUs)
        # Check limit;
        if f_check_limit:
            if Aagn < self.MB.Amin:
                Aagn = self.MB.Amin
            if Aagn > self.MB.Amax:
                Aagn = self.MB.Amax

        # Is A in logspace?
        if f_Alog:
            A00 = 10**Aagn
        else:
            A00 = Aagn

        aa = 0
        if self.MB.ZEVOL==1 or aa == 0:
            if f_check_limit:
                # Z limit:
                if par['Z%d'%aa] < self.MB.Zmin:
                    par['Z%d'%aa] = self.MB.Zmin
                if par['Z%d'%aa] > self.MB.Zmax:
                    par['Z%d'%aa] = self.MB.Zmax
            Z = par['Z%d'%aa]
            NZ = np.argmin(np.abs(self.MB.Zall-Z))

        coln = int(2 + NZ*NU + nAGNTAU)
        if aa == 0:
            nr = lib[:, 0]
            xx = lib[:, 1] # This is OBSERVED wavelength range at z=zgal
            yy = A00 * lib[:, coln]
        else:
            yy += A00 * lib[:, coln]

        if f_get_Mtot:
            return nr, xx, yy, Mtot
        else:
            return nr, xx, yy


    def get_total_flux_neb(self, par, f_Alog=True, lib_all=True, pp=0, lib=None, f_get_Mtot=False, f_check_limit=True):
        '''
        '''
        Mtot = 0
        try:
            Aneb = par['Aneb']
            logU = par['logU']
            nlogU = np.argmin(np.abs(self.MB.logUs - logU))
        except: # This is exception for initial minimizing;
            Aneb = -99
            logU = self.MB.logUs[0]
            nlogU = 0

        # logU
        NU = len(self.MB.logUs)
        # Check limit;
        if f_check_limit:
            if Aneb < self.MB.Amin:
                Aneb = self.MB.Amin
            if Aneb > self.MB.Amax:
                Aneb = self.MB.Amax

        # Is A in logspace?
        if f_Alog:
            A00 = 10**Aneb
        else:
            A00 = Aneb

        aa = 0
        if self.MB.ZEVOL==1 or aa == 0:
            if f_check_limit:
                # Z limit:
                if par['Z%d'%aa] < self.MB.Zmin:
                    par['Z%d'%aa] = self.MB.Zmin
                if par['Z%d'%aa] > self.MB.Zmax:
                    par['Z%d'%aa] = self.MB.Zmax
            Z = par['Z%d'%aa]
            NZ = np.argmin(np.abs(self.MB.Zall-Z))

        coln = int(2 + NZ*NU + nlogU)
        if aa == 0:
            nr = lib[:, 0]
            xx = lib[:, 1] # This is OBSERVED wavelength range at z=zgal
            yy = A00 * lib[:, coln]
        else:
            yy += A00 * lib[:, coln]

        if f_get_Mtot:
            return nr, xx, yy, Mtot
        else:
            return nr, xx, yy


    def tmp04_dust(self, par, nprec=1):
        '''
        Makes model template with a given param setself.
        Also dust attenuation.
        '''
        try:
            m_dust = par['MDUST']
            t_dust = par['TDUST']
        except: # This is exception for initial minimizing;
            m_dust = -99
            t_dust = 0

        nr = self.MB.lib_dust[:,0]
        xx = self.MB.lib_dust[:,1] # This is OBSERVED wavelength range at z=zgal
        coln= 2+int(t_dust+0.5)
        yy = 10**m_dust * self.MB.lib_dust[:,coln]

        if self.MB.fzmc == 1:
            zmc = par.params['zmc'].value
        else:
            zmc = self.MB.zgal

        # How much does this cost in time?
        if round(zmc,nprec) != round(self.MB.zgal,nprec):
            xx_s = xx / (1+self.MB.zgal) * (1+zmc)
            fint = interpolate.interp1d(xx, yy, kind='nearest', fill_value="extrapolate")
            yy_s = fint(xx_s)
        else:
            xx_s = xx
            yy_s = yy

        return yy_s, xx_s

  
    def get_template(self, par, f_Alog=True, nprec=1, f_val=False, check_bound=False, 
        lib_all=False, lib=None, f_nrd=False, f_apply_dust=True, f_apply_igm=True, xhi=None, f_neb=False, deltaz_lim=0.1):
        '''
        Makes model template with a given param set.
        Also dust attenuation.

        Parameters:
        -----------
        nprec : int
            Precision when redshift is refined. 
        '''
        ZZ = self.ZZ
        AA = self.AA 
        bfnc = self.MB.bfnc
        Mtot = 0
        pp = 0

        if lib == None:
            if lib_all:
                lib = self.MB.lib_all
            else:
                lib = self.MB.lib

        if f_val:
            par = par.params

        if self.MB.fzmc == 1:
            try:
                zmc = par['zmc'].value
            except:
                zmc = self.MB.zgal
        else:
            zmc = self.MB.zgal

        if check_bound:
            # AV limit;
            if par['AV0'] < self.MB.Avmin:
                par['AV0'] = self.MB.Avmin
            if par['AV0'] > self.MB.Avmax:
                par['AV0'] = self.MB.Avmax
        Av = par['AV0']

        if f_neb:
            # @@@ Not clear why f_check_limit cannot work
            nr, xx, yy, Mtot = self.get_total_flux_neb(par, f_Alog=f_Alog, lib_all=lib_all, pp=pp, lib=lib, f_get_Mtot=True, f_check_limit=check_bound)#, f_check_limit=True)
        else:
            # @@@ Not clear why f_check_limit cannot work
            nr, xx, yy, Mtot = self.get_total_flux(par, f_Alog=f_Alog, lib_all=lib_all, pp=pp, lib=lib, f_get_Mtot=True, f_check_limit=check_bound)#, f_check_limit=True)

        # @@@ Filter convolution may need to happpen here
        if round(zmc,nprec) != round(self.MB.zgal,nprec):
            if np.abs(zmc - self.MB.zgal) > deltaz_lim:
                # print('!!! zmc (%.3f) is exploring too far from zgal (%.3f).'%(zmc, self.MB.zgal))

                # @@@ This only work for BB only data set.
                # Get total flux;
                if lib_all:
                    nr_full, xx_full, yy_full = nr, xx, yy
                else:
                    if f_neb:
                        nr_full, xx_full, yy_full = self.get_total_flux_neb(par, f_Alog=f_Alog, lib_all=True, lib=lib)
                    else:
                        nr_full, xx_full, yy_full = self.get_total_flux(par, f_Alog=f_Alog, lib_all=True, lib=lib)

                xx_s, yy_s = filconv(self.MB.filts, xx_full / (1+self.MB.zgal) * (1+zmc), yy_full, self.MB.DIR_FILT, MB=self.MB, f_regist=False)
                
                if self.MB.f_spec:
                    con_bb = (nr>=self.MB.NRbb_lim)
                    xx[con_bb] = xx_s
                    yy[con_bb] = yy_s
                    xx_s = xx
                    yy_s = yy

            else:
                fint = interpolate.interp1d(xx, yy, kind='nearest', fill_value="extrapolate")
                xx_s = xx / (1+self.MB.zgal) * (1+zmc)
                yy_s = fint(xx_s)

        else:
            xx_s = xx
            yy_s = yy

        xx = xx_s
        yy = yy_s

        if f_apply_igm:
            if xhi == None:
                xhi = self.MB.x_HI_input
            yy, x_HI = dijkstra_igm_abs(xx / (1+zmc), yy, zmc, cosmo=self.MB.cosmo, x_HI=xhi)
            self.MB.x_HI = x_HI

        if f_apply_dust:
            yyd, xxd, nrd = apply_dust(yy, xx/(1+zmc), nr, Av, dust_model=self.dust_model)
            xxd *= (1.+zmc)

            if self.dust_model != 0:
                nrd_yyd = np.zeros((len(nrd),3), dtype=float)
                nrd_yyd[:,0] = nrd[:]
                nrd_yyd[:,1] = yyd[:]
                nrd_yyd[:,2] = xxd[:]
                nrd_yyd_sort = nrd_yyd[nrd_yyd[:,0].argsort()]
                nrd[:],yyd[:],xxd[:] = nrd_yyd_sort[:,0],nrd_yyd_sort[:,1],nrd_yyd_sort[:,2]

            if not f_nrd:
                return yyd[:],xxd[:]
            else:
                return nrd[:],yyd[:],xxd[:]

        else:
            if not f_nrd:
                return yy,xx
            else:
                return nr,yy,xx

