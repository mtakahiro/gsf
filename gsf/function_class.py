# For fitting.
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
import numpy as np
import sys
from numpy import log10
from scipy.integrate import simps
import matplotlib.pyplot as plt
from astropy.io import fits

from .function import *
from .basic_func import Basic


c = 3.e18 # A/s
chimax = 1.
m0set  = 25.0
d = 10**(73.6/2.5) # From [ergs/s/cm2/A] to [ergs/s/cm2/Hz]

class Func:
    def __init__(self, ZZ, AA):
        self.ZZ   = ZZ
        self.AA   = AA
        self.delZ = ZZ[1] - ZZ[0]

    def demo(self):
        ZZ = self.ZZ
        AA = self.AA
        return ZZ, AA

    #############################
    # Load template in obs range.
    #############################
    def open_spec_fits(self, ID0, PA0, fall=0, tau0=[0.01,0.02,0.03]):

        ZZ = self.ZZ
        AA = self.AA

        bfnc = Basic(ZZ)

        if fall == 0:
            app = ''
        elif fall == 1:
            app = 'all_'

        DIR_TMP = './templates/'
        for pp in range(len(tau0)):
            for zz in range(len(ZZ)):
                f0 = fits.open(DIR_TMP + 'spec_' + app + ID0 + '_PA' + PA0 + '_'+str(zz)+'.fits')
                hdu0 = f0[1]
                Z    = ZZ[zz]
                NZ   = bfnc.Z2NZ(Z)

                if zz == 0 and pp == 0:
                    nr   = hdu0.data['colnum']
                    xx   = hdu0.data['wavelength']

                    lib  = np.zeros((len(nr), 2+len(AA)*len(ZZ)*len(tau0)), dtype='float32')

                    lib[:,0] = nr[:]
                    lib[:,1] = xx[:]

                for aa in range(len(AA)):
                    coln = int(2 + aa)

                    colname = 'fspec_' + str(zz) + '_' + str(aa) + '_' + str(pp)
                    #colnall = int(2 + aa*len(ZZ) + zz) # 2 takes account of wavelength and AV columns.
                    colnall = int(2 + pp*len(ZZ)*len(AA) + zz*len(AA) + aa) # 2 takes account of wavelength and AV columns.

                    lib[:,colnall] = hdu0.data[colname]


        return lib

    #############################
    # Load template in obs range.
    # But for weird template.
    #############################
    def open_spec_fits_dir(self, ID0, PA0, nage, nz, kk, Av00, zgal, A00, tau0=[0.01,0.02,0.03]):
        ZZ = self.ZZ
        AA = self.AA
        bfnc = Basic(ZZ)
        app = 'all_'
        DIR_TMP = './templates/'

        #for pp in range(len(tau0)):
        pp = 0
        zz = nz
        f0 = fits.open(DIR_TMP + 'spec_' + app + str(zz) + '.fits')
        hdu0 = f0[1]

        # Luminosity
        f0    = fits.open(DIR_TMP + 'ms.fits')
        mshdu = f0[1]
        Ls    = np.zeros(len(AA), dtype='float32')
        Ls[:] = mshdu.data['Ls_'+str(zz)][:]


        xx   = hdu0.data['wavelength']
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
        #print(colname)

        yy0 = hdu0.data[colname]/Ls[aa]

        def flamtonu(lam, flam):
            Ctmp = lam **2/c * 10**((48.6+m0set)/2.5) #/ delx_org
            fnu  = flam * Ctmp
            return fnu

        yy = flamtonu(xx, yy0)
        lib[:,2] = yy[:]

        ########
        yyd, xxd, nrd = dust_calz2(xx, yy, Av00, nr)
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

        #coln = int(2 + pp*len(ZZ)*len(AA) + zz*len(AA) + aa) # 2 takes account of wavelength and AV columns.
        coln= int(2 + pp*len(ZZ)*len(AA) + NZ*len(AA) + nmodel)
        nr  = lib[:, 0]
        xx  = lib[:, 1] # This is OBSERVED wavelength range at z=zgal
        yy  = lib[:, coln]

        yyd, xxd, nrd = dust_calz2(xx/(1.+zgal), yy, Av00, nr)
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

        zmc    = par['zmc']
        #Cz0s  = vals['Cz0']
        #Cz1s  = vals['Cz1']

        pp0 = np.random.uniform(low=0, high=len(tau0), size=(1,))
        pp  = int(pp0[0])
        if pp>=len(tau0):
            pp += -1

        Av00 = par['Av']
        for aa in range(len(AA)):
            nmodel = aa
            Z   = par['Z'+str(aa)]
            A00 = par['A'+str(aa)]
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
        xx_s = xx / (1+zgal) * (1+zmc)
        yy_s = np.interp(xx_s, xx, yy)
        xx = xx_s
        yy = yy_s

        yyd, xxd, nrd = dust_calz2(xx/(1.+zgal), yy, Av00, nr)
        xxd *= (1.+zgal)

        nrd_yyd = np.zeros((len(nrd),3), dtype='float32')
        nrd_yyd[:,0] = nrd[:]
        nrd_yyd[:,1] = yyd[:]
        nrd_yyd[:,2] = xxd[:]

        b = nrd_yyd
        nrd_yyd_sort = b[np.lexsort(([-1,1]*b[:,[1,0]]).T)]
        yyd_sort     = nrd_yyd_sort[:,1]
        xxd_sort     = nrd_yyd_sort[:,2]

        return yyd_sort, xxd_sort

    def tmp04_val(self, ID0, PA, par, zgal, lib, tau0=[0.01,0.02,0.03]):

        ZZ = self.ZZ
        AA = self.AA
        bfnc = Basic(ZZ)
        DIR_TMP = './templates/'

        zmc    = par.params['zmc'].value
        #Cz0s  = vals['Cz0']
        #Cz1s  = vals['Cz1']

        pp0 = np.random.uniform(low=0, high=len(tau0), size=(1,))
        pp  = int(pp0[0])
        if pp>=len(tau0):
            pp += -1

        Av00 = par.params['Av'].value
        for aa in range(len(AA)):
            nmodel = aa
            Z   = par.params['Z'+str(aa)].value
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
        xx_s = xx / (1+zgal) * (1+zmc)
        yy_s = np.interp(xx_s, xx, yy)
        xx = xx_s
        yy = yy_s

        yyd, xxd, nrd = dust_calz2(xx/(1.+zgal), yy, Av00, nr)
        xxd *= (1.+zgal)

        nrd_yyd = np.zeros((len(nrd),3), dtype='float32')
        nrd_yyd[:,0] = nrd[:]
        nrd_yyd[:,1] = yyd[:]
        nrd_yyd[:,2] = xxd[:]

        b = nrd_yyd
        nrd_yyd_sort = b[np.lexsort(([-1,1]*b[:,[1,0]]).T)]
        yyd_sort     = nrd_yyd_sort[:,1]
        xxd_sort     = nrd_yyd_sort[:,2]

        return yyd_sort, xxd_sort

    def tmp04_samp(self, ID0, PA, par, zgal, lib, tau0=[0.01,0.02,0.03]):

        ZZ = self.ZZ
        AA = self.AA
        bfnc = Basic(ZZ)
        DIR_TMP = './templates/'

        #AA00[:]   = par[:len(AA)]
        #ZZ_tmp[:] = par[len(AA)+1:len(AA)+1+len(AA)]
        Av00      = par[len(AA)]
        zmc       = par[len(AA)+1+len(AA)]

        pp0 = np.random.uniform(low=0, high=len(tau0), size=(1,))
        pp  = int(pp0[0])
        if pp>=len(tau0):
            pp += -1

        for aa in range(len(AA)):
            #nmodel = aa
            A00  = par[aa] #AA00[aa] #par.params['A'+str(aa)].value
            Z    = par[len(AA)+1+aa]# ZZ_tmp[aa] #par.params['Z'+str(aa)].value
            NZ   = bfnc.Z2NZ(Z)
            coln = int(2 + pp*len(ZZ)*len(AA) + NZ*len(AA) + aa)
            if aa == 0:
                nr  = lib[:, 0]
                xx  = lib[:, 1] # This is OBSERVED wavelength range at z=zgal
                yy  = A00 * lib[:, coln]
            else:
                yy += A00 * lib[:, coln]

        # How much does this cost in time?
        xx_s = xx / (1+zgal) * (1+zmc)
        yy_s = np.interp(xx_s, xx, yy)
        xx = xx_s
        yy = yy_s

        yyd, xxd, nrd = dust_calz2(xx/(1.+zgal), yy, Av00, nr)
        xxd *= (1.+zgal)

        nrd_yyd = np.zeros((len(nrd),3), dtype='float32')
        nrd_yyd[:,0] = nrd[:]
        nrd_yyd[:,1] = yyd[:]
        nrd_yyd[:,2] = xxd[:]

        b = nrd_yyd
        nrd_yyd_sort = b[np.lexsort(([-1,1]*b[:,[1,0]]).T)]
        yyd_sort     = nrd_yyd_sort[:,1]
        xxd_sort     = nrd_yyd_sort[:,2]

        return yyd_sort, xxd_sort
