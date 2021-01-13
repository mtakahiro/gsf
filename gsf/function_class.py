import numpy as np
import sys
import scipy.interpolate as interpolate
import asdf

from .function import *
from .basic_func import Basic


class Func:
    '''
    '''
    c = 3.e18
    chimax = 1.
    m0set  = 25.0
    d = 10**(73.6/2.5)

    def __init__(self, MB, dust_model=0):
        '''
        dust_model (int) : 0 for Calzetti.
        '''
        self.ID = MB.ID
        self.PA = MB.PA
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

        self.af = asdf.open(self.DIR_TMP + 'spec_all_' + self.ID + '_PA' + self.PA + '.asdf')
        self.af0 = asdf.open(self.DIR_TMP + 'spec_all.asdf')

    def demo(self):
        ZZ = self.ZZ
        AA = self.AA
        return ZZ, AA

    #############################
    # Load template in obs range.
    #############################
    def open_spec_fits(self, MB, fall=0):
        '''
        '''
        ID0 = MB.ID
        PA0 = MB.PA
        tau0= MB.tau0 #[0.01,0.02,0.03]

        from astropy.io import fits
        ZZ = self.ZZ
        AA = self.AA
        bfnc = self.MB.bfnc #Basic(ZZ)

        # ASDF;
        if fall == 0:
            app = ''
            hdu0 = self.af['spec']
        elif fall == 1:
            app = 'all_'
            hdu0 = self.af['spec_full']

        DIR_TMP = self.DIR_TMP
        for pp in range(len(tau0)):
            for zz in range(len(ZZ)):
                Z   = ZZ[zz]
                NZ  = bfnc.Z2NZ(Z)
                if zz == 0 and pp == 0:
                    #f0   = fits.open(DIR_TMP + 'spec_' + app + ID0 + '_PA' + PA0 + '.fits')
                    #hdu0 = f0[1]
                    nr = hdu0['colnum']
                    xx = hdu0['wavelength']

                    lib = np.zeros((len(nr), 2+len(AA)*len(ZZ)*len(tau0)), dtype='float32')

                    lib[:,0] = nr[:]
                    lib[:,1] = xx[:]

                for aa in range(len(AA)):
                    coln = int(2 + aa)
                    colname = 'fspec_' + str(zz) + '_' + str(aa) + '_' + str(pp)
                    colnall = int(2 + pp*len(ZZ)*len(AA) + zz*len(AA) + aa) # 2 takes account of wavelength and AV columns.
                    lib[:,colnall] = hdu0[colname]

        return lib

    def open_spec_dust_fits(self, MB, fall=0):
        '''
        ##################################
        # Load dust template in obs range.
        ##################################
        '''
        ID0 = MB.ID
        PA0 = MB.PA
        tau0= MB.tau0 #[0.01,0.02,0.03]

        from astropy.io import fits
        ZZ = self.ZZ
        AA = self.AA
        bfnc = self.MB.bfnc #Basic(ZZ)

        if fall == 0:
            app = ''
            hdu0 = self.af['spec_dust']
        elif fall == 1:
            app = 'all_'
            hdu0 = self.af['spec_dust_full']

        DIR_TMP = self.DIR_TMP
        nr = hdu0['colnum']
        xx = hdu0['wavelength']
        
        lib  = np.zeros((len(nr), 2+len(self.Temp)), dtype='float32')
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
                hoge
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

        app = 'all'
        hdu0 = self.af['spec_full']
        DIR_TMP = self.DIR_TMP #'./templates/'

        pp = 0
        zz = nz

        # Luminosity
        mshdu = self.af0['ML']
        Ls = mshdu['Ls_%d'%nz] 

        xx   = hdu0['wavelength'] # at RF;
        nr   = np.arange(0,len(xx),1) #hdu0.data['colnum']

        lib  = np.zeros((len(nr), 2+1), dtype='float32')
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
        nrd_yyd = np.zeros((len(nrd),3), dtype='float32')
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

        nrd_yyd = np.zeros((len(nrd),3), dtype='float32')
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

        nrd_yyd = np.zeros((len(nrd),3), dtype='float32')
        nrd_yyd[:,0] = nrd[:]
        nrd_yyd[:,1] = yyd[:]
        nrd_yyd[:,2] = xxd[:]

        b = nrd_yyd
        nrd_yyd_sort = b[np.lexsort(([-1,1]*b[:,[1,0]]).T)]
        yyd_sort     = nrd_yyd_sort[:,1]
        xxd_sort     = nrd_yyd_sort[:,2]

        return A00 * yyd_sort, xxd_sort


    def tmp04(self, par, zgal, lib, f_Alog=True):
        '''
        Purpose:
        ========
        # Making model template with a given param set.
        # Also dust attenuation.
        '''
        tau0= self.tau0 #[0.01,0.02,0.03]
        ZZ = self.ZZ
        AA = self.AA
        bfnc = self.MB.bfnc #Basic(ZZ)
        DIR_TMP = self.MB.DIR_TMP #'./templates/'
        Mtot = 0

        if self.MB.fzmc == 1:
            '''
            try:
                zmc = par['zmc']
            except:
                zmc = zgal
            '''
            zmc = par['zmc']
        else:
            zmc = zgal

        pp = 0

        # AV limit;
        if par['Av'] < self.MB.Avmin:
            par['Av'] = self.MB.Avmin
        if par['Av'] > self.MB.Avmax:
            par['Av'] = self.MB.Avmax
        Av00 = par['Av']

        for aa in range(len(AA)):
            try:
                Ztest = par['Z'+str(len(AA)-1)] # instead of 'ZEVOL'
                Z = par['Z'+str(aa)]
            except:
                # This is in the case with ZEVO=0.
                Z = par['Z0']

            # Check limit;
            if par['A'+str(aa)] < self.MB.Amin:
                par['A'+str(aa)] = self.MB.Amin
            if par['A'+str(aa)] > self.MB.Amax:
                par['A'+str(aa)] = self.MB.Amax
            # Z limit:
            if aa == 0 or self.MB.Zevol == 1:
                if par['Z%d'%aa] < self.MB.Zmin:
                    par['Z%d'%aa] = self.MB.Zmin
                if par['Z%d'%aa] > self.MB.Zmax:
                    par['Z%d'%aa] = self.MB.Zmax

            # Is A in logspace?
            if f_Alog:
                A00 = 10**par['A'+str(aa)]
            else:
                A00 = par['A'+str(aa)]

            NZ = bfnc.Z2NZ(Z)
            coln = int(2 + pp*len(ZZ)*len(AA) + NZ*len(AA) + aa)

            sedpar = self.af['ML'] # For M/L
            mslist = sedpar['ML_'+str(NZ)][aa]
            Mtot += 10**(par['A%d'%aa] + np.log10(mslist))

            if aa == 0:
                nr = lib[:, 0]
                xx = lib[:, 1] # This is OBSERVED wavelength range at z=zgal
                yy = A00 * lib[:, coln]
            else:
                yy += A00 * lib[:, coln]

        # How much does this cost in time?
        if True: #round(zmc,3) != round(zgal,3):
            xx_s = xx / (1+zgal) * (1+zmc)
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

        b = nrd_yyd
        nrd_yyd_sort = b[np.lexsort(([-1,1]*b[:,[1,0]]).T)]
        yyd_sort = nrd_yyd_sort[:,1]
        xxd_sort = nrd_yyd_sort[:,2]

        self.MB.logMtmp = np.log10(Mtot)

        return yyd_sort, xxd_sort


    def tmp04_dust(self, par, zgal, lib):
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

        nr = lib[:,0]
        xx = lib[:,1] # This is OBSERVED wavelength range at z=zgal
        coln= 2+int(t_dust+0.5)
        yy = 10**m_dust * lib[:,coln]
        try:
            zmc = par.params['zmc'].value
        except:
            zmc = zgal

        # How much does this cost in time?
        if True: #round(zmc,3) != round(zgal,3):
            xx_s = xx / (1+zgal) * (1+zmc)
            fint = interpolate.interp1d(xx, yy, kind='nearest', fill_value="extrapolate")
            yy_s = fint(xx_s)
        else:
            xx_s = xx
            yy_s = yy

        return yy_s, xx_s


    def tmp04_val(self, par, zgal, lib, f_Alog=True, Amin=-10):
        '''
        '''
        tau0= self.tau0 #[0.01,0.02,0.03]
        ZZ = self.ZZ
        AA = self.AA
        bfnc = self.MB.bfnc #Basic(ZZ)
        DIR_TMP = self.MB.DIR_TMP #'./templates/'

        try:
            zmc = par.params['zmc'].value
        except:
            zmc = zgal

        pp0 = np.random.uniform(low=0, high=len(tau0), size=(1,))
        pp  = int(pp0[0])
        if pp>=len(tau0):
            pp += -1

        Av00 = par.params['Av'].value
        for aa in range(len(AA)):
            nmodel = aa
            try:
                Ztest = par.params['Z'+str(len(AA)-1)].value # instead of 'ZEVOL'
                Z = par.params['Z'+str(aa)].value
            except:
                # This is in the case with ZEVO=0.
                Z = par.params['Z0'].value

            # Is A in logspace?
            if f_Alog:
                if par.params['A'+str(aa)].value>Amin:
                    A00 = 10**par.params['A'+str(aa)].value
                else:
                    A00 = 0
            else:
                A00 = par.params['A'+str(aa)].value

            NZ  = bfnc.Z2NZ(Z)

            #coln = int(2 + pp*len(ZZ)*len(AA) + zz*len(AA) + aa) # 2 takes account of wavelength and AV columns.
            coln= int(2 + pp*len(ZZ)*len(AA) + NZ*len(AA) + nmodel)
            if aa == 0:
                nr  = lib[:, 0]
                xx  = lib[:, 1] # This is OBSERVED wavelength range at z=zgal
                yy  = A00 * lib[:, coln]
            else:
                yy += A00 * lib[:, coln]

        # How much does this cost in time?
        if zmc != zgal:
            xx_s = xx / (1+zgal) * (1+zmc)
            fint = interpolate.interp1d(xx, yy, kind='nearest', fill_value="extrapolate")
            yy_s = fint(xx_s)
        else:
            xx_s = xx
            yy_s = yy

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

        nrd_yyd = np.zeros((len(nrd),3), dtype='float32')
        nrd_yyd[:,0] = nrd[:]
        nrd_yyd[:,1] = yyd[:]
        nrd_yyd[:,2] = xxd[:]

        b = nrd_yyd
        nrd_yyd_sort = b[np.lexsort(([-1,1]*b[:,[1,0]]).T)]
        yyd_sort     = nrd_yyd_sort[:,1]
        xxd_sort     = nrd_yyd_sort[:,2]

        return yyd_sort, xxd_sort
