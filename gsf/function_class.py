# For fitting.
import numpy as np
import sys

from .function import *
from .basic_func import Basic

c = 3.e18 # A/s
chimax = 1.
m0set  = 25.0
d = 10**(73.6/2.5) # From [ergs/s/cm2/A] to [ergs/s/cm2/Hz]

class Func:
    def __init__(self, ZZ, AA, dust_model=0, DIR_TMP='./templates/'):
        self.ZZ   = ZZ
        self.AA   = AA
        try:
            self.delZ = ZZ[1] - ZZ[0]
        except:
            self.delZ = 0.01
        self.dust_model = dust_model
        self.DIR_TMP    = DIR_TMP

    def demo(self):
        ZZ = self.ZZ
        AA = self.AA
        return ZZ, AA

    #############################
    # Load template in obs range.
    #############################
    def open_spec_fits(self, ID0, PA0, fall=0, tau0=[0.01,0.02,0.03]):
        from astropy.io import fits
        ZZ = self.ZZ
        AA = self.AA
        bfnc = Basic(ZZ)
        if fall == 0:
            app = ''
        elif fall == 1:
            app = 'all_'

        DIR_TMP = self.DIR_TMP
        for pp in range(len(tau0)):
            for zz in range(len(ZZ)):
                Z    = ZZ[zz]
                NZ   = bfnc.Z2NZ(Z)

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


    ##################################
    # Load dust template in obs range.
    ##################################
    def open_spec_dust_fits(self, ID0, PA0, Temp, fall=0, tau0=[0.01,0.02,0.03]):
        from astropy.io import fits
        ZZ = self.ZZ
        AA = self.AA
        bfnc = Basic(ZZ)

        if fall == 0:
            app = ''
        elif fall == 1:
            app = 'all_'

        DIR_TMP = self.DIR_TMP #'./templates/'
        f0   = fits.open(DIR_TMP + 'spec_dust_' + app + ID0 + '_PA' + PA0 + '.fits')
        hdu0 = f0[1]
        nr   = hdu0.data['colnum']
        xx   = hdu0.data['wavelength']

        lib  = np.zeros((len(nr), 2+len(Temp)), dtype='float32')
        lib[:,0] = nr[:]
        lib[:,1] = xx[:]

        for aa in range(len(Temp)):
            coln    = int(2 + aa)
            colname = 'fspec_' + str(aa)
            colnall = int(2 + aa) # 2 takes account of wavelength and AV columns.
            lib[:,colnall] = hdu0.data[colname]
            if fall==1 and False:
                import matplotlib.pyplot as plt
                plt.close()
                print(Temp[aa])
                plt.plot(lib[:,1],lib[:,coln],linestyle='-')
                plt.show()
        return lib


    #############################
    # Load template in obs range.
    # But for weird template.
    #############################
    def open_spec_fits_dir(self, ID0, PA0, nage, nz, kk, Av00, zgal, A00, tau0=[0.01,0.02,0.03]):
        from astropy.io import fits
        ZZ = self.ZZ
        AA = self.AA
        bfnc = Basic(ZZ)
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


    def tmp03(self, ID0, PA, A00, Av00, nmodel, Z, zgal, lib, tau0=[0.01,0.02,0.03]):

        ZZ = self.ZZ
        AA = self.AA
        bfnc = Basic(ZZ)
        DIR_TMP = './templates/'
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

    # Making model template with a given param setself.
    # Also dust attenuation.
    def tmp04(self, ID0, PA, par, zgal, lib, tau0=[0.01,0.02,0.03]):
        ZZ = self.ZZ
        AA = self.AA
        bfnc = Basic(ZZ)
        DIR_TMP = './templates/'
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
            if aa == 0:
                nr  = lib[:, 0]
                xx  = lib[:, 1] # This is OBSERVED wavelength range at z=zgal
                yy  = A00 * lib[:, coln]
            else:
                yy += A00 * lib[:, coln]

        # How much does this cost in time?
        if round(zmc,3) != round(zgal,3):
            xx_s = xx / (1+zgal) * (1+zmc)
            yy_s = np.interp(xx_s, xx, yy)
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

    # Making model template with a given param setself.
    # Also dust attenuation.
    def tmp04_dust(self, ID0, PA, par, zgal, lib, tau0=[0.01,0.02,0.03]):
        ZZ = self.ZZ
        AA = self.AA
        bfnc = Basic(ZZ)
        DIR_TMP = './templates/'

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
            yy_s = np.interp(xx_s, xx, yy)
        else:
            xx_s = xx
            yy_s = yy
        return yy_s, xx_s

    def tmp04_val(self, ID0, PA, par, zgal, lib, tau0=[0.01,0.02,0.03]):

        ZZ = self.ZZ
        AA = self.AA
        bfnc = Basic(ZZ)
        DIR_TMP = './templates/'

        try:
            zmc = par.params['zmc'].value
        except:
            zmc = zgal

        #Cz0s  = vals['Cz0']
        #Cz1s  = vals['Cz1']

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
            yy_s = np.interp(xx_s, xx, yy)
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
