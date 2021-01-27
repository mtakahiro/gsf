import numpy as np
import sys
import scipy.interpolate as interpolate
import asdf

from .function import *
from .basic_func import Basic


class Func:
    '''
    '''
    def __init__(self, MB, dust_model=0):
        '''
        dust_model (int) : 0 for Calzetti.
        '''
        self.ID = MB.ID
        self.ZZ = MB.Zall
        self.age = MB.age
        self.AA = MB.nage
        self.tau0 = MB.tau0
        self.MB = MB

        self.dust_model = dust_model
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

    def demo(self):
        ZZ = self.ZZ
        AA = self.AA
        return ZZ, AA

    #############################
    # Load template in obs range.
    #############################
    def open_spec_fits(self, fall=0):
        '''
        '''
        ID0 = self.MB.ID
        tau0= self.MB.tau0 #[0.01,0.02,0.03]

        from astropy.io import fits
        ZZ = self.ZZ
        AA = self.AA
        bfnc = self.MB.bfnc #Basic(ZZ)


        # ASDF;
        if fall == 0:
            app = ''
            hdu0 = self.MB.af['spec']
        elif fall == 1:
            app = 'all_'
            hdu0 = self.MB.af['spec_full']

        DIR_TMP = self.DIR_TMP
        for pp in range(len(tau0)):
            for zz in range(len(ZZ)):
                Z   = ZZ[zz]
                NZ  = bfnc.Z2NZ(Z)
                if zz == 0 and pp == 0:
                    nr = hdu0['colnum']
                    xx = hdu0['wavelength']

                    lib = np.zeros((len(nr), 2+len(AA)*len(ZZ)*len(tau0)), dtype='float')

                    lib[:,0] = nr[:]
                    lib[:,1] = xx[:]

                for aa in range(len(AA)):
                    coln = int(2 + aa)
                    colname = 'fspec_' + str(zz) + '_' + str(aa) + '_' + str(pp)
                    colnall = int(2 + pp*len(ZZ)*len(AA) + zz*len(AA) + aa) # 2 takes account of wavelength and AV columns.
                    lib[:,colnall] = hdu0[colname]

        return lib

    def open_spec_dust_fits(self, fall=0):
        '''
        ##################################
        # Load dust template in obs range.
        ##################################
        '''
        ID0 = self.MB.ID
        tau0= self.MB.tau0 #[0.01,0.02,0.03]

        from astropy.io import fits
        ZZ = self.ZZ
        AA = self.AA
        bfnc = self.MB.bfnc #Basic(ZZ)

        self.MB.af = asdf.open(self.DIR_TMP + 'spec_all_' + self.ID + '.asdf')
        self.MB.af0 = asdf.open(self.DIR_TMP + 'spec_all.asdf')

        if fall == 0:
            app = ''
            hdu0 = self.MB.af['spec_dust']
        elif fall == 1:
            app = 'all_'
            hdu0 = self.MB.af['spec_dust_full']

        DIR_TMP = self.DIR_TMP
        nr = hdu0['colnum']
        xx = hdu0['wavelength']
        
        lib  = np.zeros((len(nr), 2+len(self.Temp)), dtype='float')
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


    def open_spec_fits_dir(self, nage, nz, kk, Av00, zgal, A00):
        '''
        #############################
        # Load template in obs range.
        # But for weird template.
        #############################
        '''
        from astropy.io import fits
        tau0= self.tau0 #[0.01,0.02,0.03]
        ZZ = self.ZZ
        AA = self.AA
        bfnc = self.MB.bfnc #Basic(ZZ)

        self.MB.af = asdf.open(self.DIR_TMP + 'spec_all_' + self.ID + '.asdf')
        self.MB.af0 = asdf.open(self.DIR_TMP + 'spec_all.asdf')

        app = 'all'
        hdu0 = self.MB.af['spec_full']
        DIR_TMP = self.DIR_TMP #'./templates/'

        pp = 0
        zz = nz

        # Luminosity
        mshdu = self.MB.af0['ML']
        Ls = mshdu['Ls_%d'%nz] 

        xx   = hdu0['wavelength'] # at RF;
        nr   = np.arange(0,len(xx),1) #hdu0.data['colnum']

        lib  = np.zeros((len(nr), 2+1), dtype='float')
        lib[:,0] = nr[:]
        lib[:,1] = xx[:]

        aa = nage
        coln = int(2 + aa)
        colname = 'fspec_' + str(zz) + '_' + str(aa) + '_' + str(pp)
        #if kk == 0: # = Tobs.
        #    colname = 'fspec_' + str(zz) + '_' + str(aa) + '_' + str(pp)
        #else: # Tobs - age[kk-1] where kk>=1.
        #    colname = 'fspec_' + str(zz) + '_' + str(aa) + '_' + str(pp) + '_' + str(kk-1)

        #if aa > 0 and aa == kk:
        #    colname = 'fspec_' + str(zz) + '_0' + '_' + str(pp)# + '_0'

        yy0 = hdu0[colname]/Ls[aa]
        yy = flamtonu(xx, yy0)
        lib[:,2] = yy[:]

        if self.dust_model == 0: # Calzetti
            yyd, xxd, nrd = dust_calz(xx, yy, Av00, nr)
        elif self.dust_model == 1: # MW
            yyd, xxd, nrd = dust_mw(xx, yy, Av00, nr)
        elif self.dust_model == 2: # LMC
            yyd, xxd, nrd = dust_gen(xx, yy, Av00, nr, Rv=4.05, gamma=-0.06, Eb=2.8)
        elif self.dust_model == 3: # SMC
            yyd, xxd, nrd = dust_gen(xx, yy, Av00, nr, Rv=4.05, gamma=-0.42, Eb=0.0)
        elif self.dust_model == 4: # Kriek&Conroy with gamma=-0.2
            yyd, xxd, nrd = dust_kc(xx, yy, Av00, nr, Rv=4.05, gamma=-0.2)
        else:
            print('No entry. Dust model is set to Calzetti')
            yyd, xxd, nrd = dust_calz(xx, yy, Av00, nr)

        xxd *= (1.+zgal)
        nrd_yyd = np.zeros((len(nrd),3), dtype='float')
        nrd_yyd[:,0] = nrd[:]
        nrd_yyd[:,1] = yyd[:]
        nrd_yyd[:,2] = xxd[:]

        b = nrd_yyd
        nrd_yyd_sort = b[np.lexsort(([-1,1]*b[:,[1,0]]).T)]
        yyd_sort     = nrd_yyd_sort[:,1]
        xxd_sort     = nrd_yyd_sort[:,2]

        return A00 * yyd_sort, xxd_sort


    def get_template(self, lib, Amp=1.0, T=1.0, Av=0.0, Z=0.0, zgal=1.0, f_bb=False):
        '''
        Purpose:
        ========
        Gets an element template given a set of parameters.
        Not necessarily the most efficient way, but easy to use.

        Input:
        ======
        lib : library dictionary.
        Amp : Amplitude of template. Note that each template has Lbol = 1e10Lsun.
        T   : Age, in Gyr.
        Av  : Dust attenuation in mag.
        Z   : Metallicity in log(Z/Zsun).
        zgal: Redshift.
        f_bb: bool, to calculate bb photometry for the spectrum requested.

        Return:
        =======
        flux, wavelength : Flux in Fnu. Wave in AA.
        lcen, lflux, if f_bb==True.

        '''

        bfnc = self.MB.bfnc
        DIR_TMP = self.MB.DIR_TMP 
        NZ  = bfnc.Z2NZ(Z)

        pp0 = np.random.uniform(low=0, high=len(self.tau0), size=(1,))
        pp  = int(pp0[0])
        if pp>=len(self.tau0):
            pp += -1

        nmodel = np.argmin(np.abs(T-self.age[:]))
        if T - self.age[nmodel] != 0:
            print('T=%.2f is not found in age library. T=%.2f is used.'%(T,self.age[nmodel]))

        coln= int(2 + pp*len(self.ZZ)*len(self.AA) + NZ*len(self.AA) + nmodel)
        nr  = lib[:, 0]
        xx  = lib[:, 1] # This is OBSERVED wavelength range at z=zgal
        yy  = lib[:, coln]

        if self.dust_model == 0:
            yyd, xxd, nrd = dust_calz(xx/(1.+zgal), yy, Av, nr)
        elif self.dust_model == 1:
            yyd, xxd, nrd = dust_mw(xx/(1.+zgal), yy, Av, nr)
        elif self.dust_model == 2: # LMC
            yyd, xxd, nrd = dust_gen(xx/(1.+zgal), yy, Av, nr, Rv=4.05, gamma=-0.06, Eb=2.8)
        elif self.dust_model == 3: # SMC
            yyd, xxd, nrd = dust_gen(xx/(1.+zgal), yy, Av, nr, Rv=4.05, gamma=-0.42, Eb=0.0)
        elif self.dust_model == 4: # Kriek&Conroy with gamma=-0.2
            yyd, xxd, nrd = dust_kc(xx/(1.+zgal), yy, Av, nr, Rv=4.05, gamma=-0.2)
        else:
            yyd, xxd, nrd = dust_calz(xx/(1.+zgal), yy, Av, nr)

        xxd *= (1.+zgal)

        nrd_yyd = np.zeros((len(nrd),3), dtype='float')
        nrd_yyd[:,0] = nrd[:]
        nrd_yyd[:,1] = yyd[:]
        nrd_yyd[:,2] = xxd[:]

        b = nrd_yyd
        nrd_yyd_sort = b[np.lexsort(([-1,1]*b[:,[1,0]]).T)]
        yyd_sort     = nrd_yyd_sort[:,1]
        xxd_sort     = nrd_yyd_sort[:,2]

        if f_bb:
            #fil_cen, fil_flux = filconv(self.filts, xxd_sort, Amp * yyd_sort, self.DIR_FIL)
            fil_cen, fil_flux = filconv_fast(self.MB, xxd_sort, Amp * yyd_sort)
            return Amp * yyd_sort, xxd_sort, fil_flux, fil_cen
        else:
            return Amp * yyd_sort, xxd_sort


    def tmp03(self, A00, Av00, nmodel, Z, zgal, lib):
        '''
        '''
        tau0= self.tau0 #[0.01,0.02,0.03]
        ZZ = self.ZZ
        AA = self.AA
        bfnc = self.MB.bfnc #Basic(ZZ)
        DIR_TMP = self.MB.DIR_TMP #'./templates/'
        NZ  = bfnc.Z2NZ(Z)

        pp0 = np.random.uniform(low=0, high=len(tau0), size=(1,))
        pp  = int(pp0[0])
        if pp>=len(tau0):
            pp += -1

        coln= int(2 + pp*len(ZZ)*len(AA) + NZ*len(AA) + nmodel)
        nr  = lib[:,0]
        xx  = lib[:,1] # This is OBSERVED wavelength range at z=zgal
        yy  = lib[:,coln]

        if self.dust_model == 0:
            yyd, xxd, nrd = dust_calz(xx/(1.+zgal), yy, Av00, nr)
        elif self.dust_model == 1:
            yyd, xxd, nrd = dust_mw(xx/(1.+zgal), yy, Av00, nr)
        elif self.dust_model == 2: # LMC
            yyd, xxd, nrd = dust_gen(xx/(1.+zgal), yy, Av00, nr, Rv=4.05, gamma=-0.06, Eb=2.8)
        elif self.dust_model == 3: # SMC
            yyd, xxd, nrd = dust_gen(xx/(1.+zgal), yy, Av00, nr, Rv=4.05, gamma=-0.42, Eb=0.0)
        elif self.dust_model == 4: # Kriek&Conroy with gamma=-0.2
            yyd, xxd, nrd = dust_kc(xx/(1.+zgal), yy, Av00, nr, Rv=4.05, gamma=-0.2)
        else:
            yyd, xxd, nrd = dust_calz(xx/(1.+zgal), yy, Av00, nr)


        xxd *= (1.+zgal)

        nrd_yyd = np.zeros((len(nrd),3), dtype='float')
        nrd_yyd[:,0] = nrd[:]
        nrd_yyd[:,1] = yyd[:]
        nrd_yyd[:,2] = xxd[:]

        b = nrd_yyd
        nrd_yyd_sort = b[np.lexsort(([-1,1]*b[:,[1,0]]).T)]
        yyd_sort     = nrd_yyd_sort[:,1]
        xxd_sort     = nrd_yyd_sort[:,2]

        return A00 * yyd_sort, xxd_sort


    def tmp04(self, par, f_Alog=True, nprec=1, f_val=False):
        '''
        Purpose:
        ========
        # Making model template with a given param set.
        # Also dust attenuation.

        Input:
        ======
        nprec : Precision when redshift is refined. 
        '''
        ZZ = self.ZZ
        AA = self.AA
        bfnc = self.MB.bfnc #Basic(ZZ)
        Mtot = 0

        if f_val:
            par = par.params

        if self.MB.fzmc == 1:
            zmc = par['zmc']
        else:
            zmc = self.MB.zgal

        pp = 0

        # AV limit;
        if par['Av'] < self.MB.Avmin:
            par['Av'] = self.MB.Avmin
        if par['Av'] > self.MB.Avmax:
            par['Av'] = self.MB.Avmax
        Av00 = par['Av']

        for aa in range(len(AA)):
            if self.MB.ZEVOL==1 or aa == 0:
                Z = par['Z'+str(aa)]
                NZ = bfnc.Z2NZ(Z)
            else:
                pass

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

            coln = int(2 + pp*len(ZZ)*len(AA) + NZ*len(AA) + aa)

            sedpar = self.MB.af['ML'] # For M/L
            mslist = sedpar['ML_'+str(NZ)][aa]
            Mtot += 10**(par['A%d'%aa] + np.log10(mslist))

            if aa == 0:
                nr = self.MB.lib[:, 0]
                xx = self.MB.lib[:, 1] # This is OBSERVED wavelength range at z=zgal
                yy = A00 * self.MB.lib[:, coln]
            else:
                yy += A00 * self.MB.lib[:, coln]
        
        self.MB.logMtmp = np.log10(Mtot)
        # How much does this cost in time?
        if round(zmc,nprec) != round(self.MB.zgal,nprec):
            xx_s = xx / (1+self.MB.zgal) * (1+zmc)
            fint = interpolate.interp1d(xx, yy, kind='nearest', fill_value="extrapolate")
            yy_s = fint(xx_s)
        else:
            xx_s = xx
            yy_s = yy

        xx = xx_s
        yy = yy_s

        if self.dust_model == 0:
            yyd, xxd, nrd = dust_calz(xx/(1.+zmc), yy, Av00, nr)
        elif self.dust_model == 1:
            yyd, xxd, nrd = dust_mw(xx/(1.+zmc), yy, Av00, nr)
        elif self.dust_model == 2: # LMC
            yyd, xxd, nrd = dust_gen(xx/(1.+zmc), yy, Av00, nr, Rv=4.05, gamma=-0.06, Eb=2.8)
        elif self.dust_model == 3: # SMC
            yyd, xxd, nrd = dust_gen(xx/(1.+zmc), yy, Av00, nr, Rv=4.05, gamma=-0.42, Eb=0.0)
        elif self.dust_model == 4: # Kriek&Conroy with gamma=-0.2
            yyd, xxd, nrd = dust_kc(xx/(1.+zmc), yy, Av00, nr, Rv=4.05, gamma=-0.2)
        else:
            yyd, xxd, nrd = dust_calz(xx/(1.+zmc), yy, Av00, nr)
        xxd *= (1.+zmc)

        nrd_yyd = np.zeros((len(nrd),3), dtype='float')
        nrd_yyd[:,0] = nrd[:]
        nrd_yyd[:,1] = yyd[:]
        nrd_yyd[:,2] = xxd[:]
        nrd_yyd_sort = nrd_yyd[nrd_yyd[:,0].argsort()]
        return nrd_yyd_sort[:,1],nrd_yyd_sort[:,2]


    def tmp04_dust(self, par, nprec=1):
        '''
        Purpose:
        ========
        # Making model template with a given param setself.
        # Also dust attenuation.
        '''
        tau0= self.tau0 #[0.01,0.02,0.03]
        ZZ = self.ZZ
        AA = self.AA
        bfnc = self.MB.bfnc #Basic(ZZ)
        DIR_TMP = self.MB.DIR_TMP #'./templates/'

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


class Func_tau:
    '''
    '''

    def __init__(self, MB, dust_model=0):
        '''
        dust_model (int) : 0 for Calzetti.
        '''
        self.MB = MB
        self.ID = MB.ID
        self.ZZ = MB.Zall
        self.AA = MB.nage
        self.tau = MB.tau

        self.dust_model = dust_model
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

    def demo(self):
        ZZ = self.ZZ
        AA = self.AA
        return ZZ, AA

    #############################
    # Load template in obs range.
    #############################
    def open_spec_fits(self, fall=0):
        '''
        '''
        ID0 = self.MB.ID

        from astropy.io import fits
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
        for zz in range(len(ZZ)):
            for tt in range(len(self.MB.tau)):
                for ss in range(len(self.MB.ageparam)):
                    Z = ZZ[zz]
                    TT = self.MB.tau[tt]
                    TA = self.MB.ageparam[ss]

                    if zz == 0 and tt == 0 and ss == 0:
                        nr = hdu0['colnum']
                        xx = hdu0['wavelength']
                        coln = int(2 + len(ZZ) * self.MB.ntau * self.MB.nage)# + self.MB.ntau * self.MB.nage + NA)
                        lib = np.zeros((len(nr), coln), dtype='float')
                        lib[:,0] = nr[:]
                        lib[:,1] = xx[:]

                    colname = 'fspec_' + str(zz) + '_' + str(tt) + '_' + str(ss)
                    colnall = int(2 + zz * self.MB.ntau * self.MB.nage + tt * self.MB.nage + ss) # 2 takes account of wavelength and AV columns.
                                        
                    lib[:,colnall] = hdu0[colname]

        return lib


    def open_spec_dust_fits(self, fall=0):
        '''
        ##################################
        # Load dust template in obs range.
        ##################################
        '''
        ID0 = self.MB.ID
        tau0= self.MB.tau0 #[0.01,0.02,0.03]

        from astropy.io import fits
        ZZ = self.ZZ
        AA = self.AA
        bfnc = self.MB.bfnc #Basic(ZZ)

        self.MB.af = asdf.open(self.DIR_TMP + 'spec_all_' + self.ID + '.asdf')
        self.MB.af0 = asdf.open(self.DIR_TMP + 'spec_all.asdf')

        if fall == 0:
            app = ''
            hdu0 = self.MB.af['spec_dust']
        elif fall == 1:
            app = 'all_'
            hdu0 = self.MB.af['spec_dust_full']

        DIR_TMP = self.DIR_TMP
        nr = hdu0['colnum']
        xx = hdu0['wavelength']
        
        lib  = np.zeros((len(nr), 2+len(self.Temp)), dtype='float')
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


    def open_spec_fits_dir(self, nage, nz, kk, Av00, zgal, A00):
        '''
        #############################
        # Load template in obs range.
        # But for weird template.
        #############################
        '''
        from astropy.io import fits
        tau0= self.tau0 #[0.01,0.02,0.03]
        ZZ = self.ZZ
        AA = self.AA
        bfnc = self.MB.bfnc #Basic(ZZ)

        self.MB.af = asdf.open(self.DIR_TMP + 'spec_all_' + self.ID + '.asdf')
        self.MB.af0 = asdf.open(self.DIR_TMP + 'spec_all.asdf')

        app = 'all'
        hdu0 = self.MB.af['spec_full']
        DIR_TMP = self.DIR_TMP #'./templates/'

        pp = 0
        zz = nz

        # Luminosity
        mshdu = self.MB.af0['ML']
        Ls = mshdu['Ls_%d'%nz] 

        xx   = hdu0['wavelength'] # at RF;
        nr   = np.arange(0,len(xx),1) #hdu0.data['colnum']

        lib  = np.zeros((len(nr), 2+1), dtype='float')
        lib[:,0] = nr[:]
        lib[:,1] = xx[:]

        aa = nage
        coln = int(2 + aa)
        colname = 'fspec_' + str(zz) + '_' + str(aa) + '_' + str(pp)

        yy0 = hdu0[colname]/Ls[aa]
        yy  = flamtonu(xx, yy0)
        lib[:,2] = yy[:]

        if self.dust_model == 0: # Calzetti
            yyd, xxd, nrd = dust_calz(xx, yy, Av00, nr)
        elif self.dust_model == 1: # MW
            yyd, xxd, nrd = dust_mw(xx, yy, Av00, nr)
        elif self.dust_model == 2: # LMC
            yyd, xxd, nrd = dust_gen(xx, yy, Av00, nr, Rv=4.05, gamma=-0.06, Eb=2.8)
        elif self.dust_model == 3: # SMC
            yyd, xxd, nrd = dust_gen(xx, yy, Av00, nr, Rv=4.05, gamma=-0.42, Eb=0.0)
        elif self.dust_model == 4: # Kriek&Conroy with gamma=-0.2
            yyd, xxd, nrd = dust_kc(xx, yy, Av00, nr, Rv=4.05, gamma=-0.2)
        else:
            print('No entry. Dust model is set to Calzetti')
            yyd, xxd, nrd = dust_calz(xx, yy, Av00, nr)

        xxd *= (1.+zgal)
        nrd_yyd = np.zeros((len(nrd),3), dtype='float')
        nrd_yyd[:,0] = nrd[:]
        nrd_yyd[:,1] = yyd[:]
        nrd_yyd[:,2] = xxd[:]

        b = nrd_yyd
        nrd_yyd_sort = b[np.lexsort(([-1,1]*b[:,[1,0]]).T)]
        yyd_sort     = nrd_yyd_sort[:,1]
        xxd_sort     = nrd_yyd_sort[:,2]

        return A00 * yyd_sort, xxd_sort


    def tmp04(self, par, f_Alog=True, nprec=1, f_val=False, check_bound=False, lib_all=False):
        '''
        Purpose:
        ========
        # Making model template with a given param set.
        # Also dust attenuation.

        Input:
        ======
        nprec : Precision when redshift is refined. 
        '''
        ZZ = self.ZZ # Metal
        AA = self.AA # Amp
        bfnc = self.MB.bfnc #Basic(ZZ)
        Mtot = 0
        pp = 0

        if f_val:
            par = par.params

        if self.MB.fzmc == 1:
            zmc = par['zmc']
        else:
            zmc = self.MB.zgal

        if check_bound:
            # AV limit;
            if par['Av'] < self.MB.Avmin:
                par['Av'] = self.MB.Avmin
            if par['Av'] > self.MB.Avmax:
                par['Av'] = self.MB.Avmax
        Av00 = par['Av']

        for aa in range(self.MB.npeak):
            if self.MB.ZEVOL==1 or aa == 0:
                if check_bound:
                    # Z limit:
                    if par['Z%d'%aa] < self.MB.Zmin:
                        par['Z%d'%aa] = self.MB.Zmin
                    if par['Z%d'%aa] > self.MB.Zmax:
                        par['Z%d'%aa] = self.MB.Zmax
                Z = par['Z%d'%aa]
            else:
                pass

            if check_bound:
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
                    par['AGE'+str(aa)] = self.MB.agemax

            # Is A in logspace?
            if f_Alog:
                A00 = 10**par['A'+str(aa)]
            else:
                A00 = par['A'+str(aa)]

            tau,age = par['TAU%d'%aa],par['AGE%d'%aa]

            NZ, NT, NA = bfnc.Z2NZ(Z,tau,age)
            coln = int(2 + NZ*len(self.MB.tau)*len(self.MB.ageparam) + NT*len(self.MB.ageparam) + NA)
            sedpar = self.MB.af['ML'] # For M/L
            mslist = sedpar['ML_'+str(NZ)+'_'+str(NT)][NA]
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


        self.MB.logMtmp = np.log10(Mtot)
        # How much does this cost in time?
        if round(zmc,nprec) != round(self.MB.zgal,nprec):
            xx_s = xx / (1+self.MB.zgal) * (1+zmc)
            fint = interpolate.interp1d(xx, yy, kind='nearest', fill_value="extrapolate")
            yy_s = fint(xx_s)
        else:
            xx_s = xx
            yy_s = yy

        xx = xx_s
        yy = yy_s

        if self.dust_model == 0:
            yyd, xxd, nrd = dust_calz(xx/(1.+zmc), yy, Av00, nr)
        elif self.dust_model == 1:
            yyd, xxd, nrd = dust_mw(xx/(1.+zmc), yy, Av00, nr)
        elif self.dust_model == 2: # LMC
            yyd, xxd, nrd = dust_gen(xx/(1.+zmc), yy, Av00, nr, Rv=4.05, gamma=-0.06, Eb=2.8)
        elif self.dust_model == 3: # SMC
            yyd, xxd, nrd = dust_gen(xx/(1.+zmc), yy, Av00, nr, Rv=4.05, gamma=-0.42, Eb=0.0)
        elif self.dust_model == 4: # Kriek&Conroy with gamma=-0.2
            yyd, xxd, nrd = dust_kc(xx/(1.+zmc), yy, Av00, nr, Rv=4.05, gamma=-0.2)
        else:
            yyd, xxd, nrd = dust_calz(xx/(1.+zmc), yy, Av00, nr)
        xxd *= (1.+zmc)

        if self.dust_model == 0:
            return yyd, xxd
        else: # finction.py needs update for other models.
            nrd_yyd = np.zeros((len(nrd),3), dtype='float')
            nrd_yyd[:,0] = nrd[:]
            nrd_yyd[:,1] = yyd[:]
            nrd_yyd[:,2] = xxd[:]
            nrd_yyd_sort = nrd_yyd[nrd_yyd[:,0].argsort()]
            return nrd_yyd_sort[:,1],nrd_yyd_sort[:,2]


    def tmp04_dust(self, par, nprec=1):
        '''
        Purpose:
        ========
        # Making model template with a given param setself.
        # Also dust attenuation.
        '''
        tau0= self.tau0 #[0.01,0.02,0.03]
        ZZ = self.ZZ
        AA = self.AA
        bfnc = self.MB.bfnc #Basic(ZZ)
        DIR_TMP = self.MB.DIR_TMP #'./templates/'

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

  