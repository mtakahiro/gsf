import numpy as np
import sys
import scipy.interpolate as interpolate

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
        self.ID   = MB.ID
        self.PA   = MB.PA
        self.ZZ   = MB.Zall
        self.age  = MB.age
        self.AA   = MB.nage
        self.tau0 = MB.tau0
        self.MB   = MB

        self.dust_model = dust_model
        self.DIR_TMP    = MB.DIR_TMP

        if MB.f_dust:
            self.Temp = MB.Temp

        try:
            self.filts   = MB.filts
            self.DIR_FIL = MB.DIR_FILT
        except:
            pass

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
        if fall == 0:
            app = ''
        elif fall == 1:
            app = 'all_'

        DIR_TMP = self.DIR_TMP
        for pp in range(len(tau0)):
            for zz in range(len(ZZ)):
                Z   = ZZ[zz]
                NZ  = bfnc.Z2NZ(Z)
                if zz == 0 and pp == 0:
                    f0   = fits.open(DIR_TMP + 'spec_' + app + ID0 + '_PA' + PA0 + '.fits')
                    hdu0 = f0[1]
                    nr   = hdu0.data['colnum']
                    xx   = hdu0.data['wavelength']

                    lib  = np.zeros((len(nr), 2+len(AA)*len(ZZ)*len(tau0)), dtype='float32')

                    lib[:,0] = nr[:]
                    lib[:,1] = xx[:]

                for aa in range(len(AA)):
                    coln = int(2 + aa)

                    colname = 'fspec_' + str(zz) + '_' + str(aa) + '_' + str(pp)
                    colnall = int(2 + pp*len(ZZ)*len(AA) + zz*len(AA) + aa) # 2 takes account of wavelength and AV columns.

                    lib[:,colnall] = hdu0.data[colname]

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
        elif fall == 1:
            app = 'all_'

        DIR_TMP = self.DIR_TMP #'./templates/'
        f0   = fits.open(DIR_TMP + 'spec_dust_' + app + ID0 + '_PA' + PA0 + '.fits')
        hdu0 = f0[1]
        nr   = hdu0.data['colnum']
        xx   = hdu0.data['wavelength']

        lib  = np.zeros((len(nr), 2+len(self.Temp)), dtype='float32')
        lib[:,0] = nr[:]
        lib[:,1] = xx[:]

        for aa in range(len(self.Temp)):
            coln    = int(2 + aa)
            colname = 'fspec_' + str(aa)
            colnall = int(2 + aa) # 2 takes account of wavelength and AV columns.
            lib[:,colnall] = hdu0.data[colname]
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
        app = 'all'
        DIR_TMP = self.DIR_TMP #'./templates/'

        #for pp in range(len(tau0)):
        pp = 0
        zz = nz
        f0 = fits.open(DIR_TMP + 'spec_' + app + '.fits')
        hdu0 = f0[1]

        # Luminosity
        f0    = fits.open(DIR_TMP + 'ms.fits')
        mshdu = f0[1]
        Ls    = np.zeros(len(AA), dtype='float32')
        Ls[:] = mshdu.data['Ls_'+str(zz)][:]

        xx   = hdu0.data['wavelength'] # RF;
        nr   = np.arange(0,len(xx),1) #hdu0.data['colnum']

        lib  = np.zeros((len(nr), 2+1), dtype='float32')
        lib[:,0] = nr[:]
        lib[:,1] = xx[:]

        aa = nage
        coln = int(2 + aa)
        if kk == 0: # = Tobs.
            colname = 'fspec_' + str(zz) + '_' + str(aa) + '_' + str(pp)
        else: # Tobs - age[kk-1] where kk>=1.
            colname = 'fspec_' + str(zz) + '_' + str(aa) + '_' + str(pp) + '_' + str(kk-1)

        if aa >0 and aa == kk:
            colname = 'fspec_' + str(zz) + '_0' + '_' + str(pp)# + '_0'

        yy0 = hdu0.data[colname]/Ls[aa]
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
        =========
        Gets an element template given a set of parameters.
        Not necessarily the most efficient way, but easy to use.

        Input:
        =========
        lib : library dictionary.
        Amp : Amplitude of template. Note that each template has Lbol = 1e10Lsun.
        T   : Age, in Gyr.
        Av  : Dust attenuation in mag.
        Z   : Metallicity in log(Z/Zsun).
        zgal: Redshift.
        f_bb: bool, to calculate bb photometry for the spectrum requested.

        Return:
        ========
        flux, wavelength : Flux in Fnu. Wave in AA.
        lcen, lflux, if f_bb==True.

        '''

        bfnc = self.MB.bfnc #Basic(ZZ)
        DIR_TMP = self.MB.DIR_TMP #'./templates/'
        NZ  = bfnc.Z2NZ(Z)

        pp0 = np.random.uniform(low=0, high=len(self.tau0), size=(1,))
        pp  = int(pp0[0])
        if pp>=len(self.tau0):
            pp += -1

        nmodel = np.argmin(np.abs(T-self.age[:]))
        if T - self.age[nmodel] != 0:
            print('T=%.2 is not found in age library. T=%.2 is used.'%(T,self.age[nmodel]))

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
        nr  = lib[:, 0]
        xx  = lib[:, 1] # This is OBSERVED wavelength range at z=zgal
        yy  = lib[:, coln]

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

    def tmp04(self, par, zgal, lib):
        '''
        # Making model template with a given param setself.
        # Also dust attenuation.
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
        #Cz0s  = vals['Cz0']
        #Cz1s  = vals['Cz1']

        if len(tau0)>1:
            pp0 = np.random.uniform(low=0, high=len(tau0), size=(1,))
            pp  = int(pp0[0])
            if pp>=len(tau0):
                pp += -1
        else:
            pp  = 0

        Av00 = par['Av']
        for aa in range(len(AA)):
            try:
                Ztest = par['Z'+str(len(AA)-1)] # instead of 'ZEVOL'
                Z     = par['Z'+str(aa)]
            except:
                # This is in the case with ZEVO=0.
                Z   = par['Z0']
            A00 = par['A'+str(aa)]
            NZ  = bfnc.Z2NZ(Z)
            coln= int(2 + pp*len(ZZ)*len(AA) + NZ*len(AA) + aa)
            '''
            # Just debbug purpose...
            if len(lib[0,:])<coln:
                print('################################################')
                print('Something is wrong in tmp04 in function_class.py')
                print('aa = %d out of '%(aa),AA)
                print('NZ = %d out of '%(NZ),ZZ)
                print('pp = %d'%(pp))
                print('2 + pp*len(ZZ)*len(AA) + NZ*len(AA) + aa must be <= %d'%(len(lib[0,:])))
                print('################################################')
            '''
            if aa == 0:
                nr  = lib[:, 0]
                xx  = lib[:, 1] # This is OBSERVED wavelength range at z=zgal
                yy  = A00 * lib[:, coln]
            else:
                yy += A00 * lib[:, coln]

        # How much does this cost in time?
        if round(zmc,3) != round(zgal,3):
            xx_s = xx / (1+zgal) * (1+zmc)

            fint = interpolate.interp1d(xx, yy, kind='nearest', fill_value="extrapolate")
            yy_s = fint(xx_s)
            #yy_s = np.interp(xx_s, xx, yy)
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

        nrd_yyd = np.zeros((len(nrd),3), dtype='float32')
        nrd_yyd[:,0] = nrd[:]
        nrd_yyd[:,1] = yyd[:]
        nrd_yyd[:,2] = xxd[:]

        b = nrd_yyd
        nrd_yyd_sort = b[np.lexsort(([-1,1]*b[:,[1,0]]).T)]
        yyd_sort     = nrd_yyd_sort[:,1]
        xxd_sort     = nrd_yyd_sort[:,2]

        return yyd_sort, xxd_sort

    def tmp04_dust(self, par, zgal, lib):
        '''
        # Making model template with a given param setself.
        # Also dust attenuation.
        '''
        tau0= self.tau0 #[0.01,0.02,0.03]
        ZZ = self.ZZ
        AA = self.AA
        bfnc = self.MB.bfnc #Basic(ZZ)
        DIR_TMP = self.MB.DIR_TMP #'./templates/'

        m_dust = par['MDUST']
        t_dust = par['TDUST']
        #print(t_dust,m_dust)

        nr  = lib[:,0]
        xx  = lib[:,1] # This is OBSERVED wavelength range at z=zgal
        coln= 2+int(t_dust+0.5)
        yy  = m_dust * lib[:,coln]
        try:
            zmc = par.params['zmc'].value
        except:
            zmc = zgal
        # How much does this cost in time?
        if round(zmc,3) != round(zgal,3):
            xx_s = xx / (1+zgal) * (1+zmc)
            fint = interpolate.interp1d(xx, yy, kind='nearest', fill_value="extrapolate")
            yy_s = fint(xx_s)
        else:
            xx_s = xx
            yy_s = yy
        return yy_s, xx_s

    def tmp04_val(self, par, zgal, lib):
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
