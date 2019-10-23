#!/usr/bin/env python
#<examples/doc_nistgauss.py>
import numpy as np
#from lmfit.models import GaussianModel, ExponentialModel, ExpressionModel
import sys
import matplotlib.pyplot as plt
#import lmfit
from numpy import log10
from scipy.integrate import simps
from astropy.io import fits
import pickle
import os
from matplotlib.ticker import FormatStrFormatter

# Custom modules
from .function import *
from .function_class import Func
from .basic_func import Basic
from . import img_scale


lcb   = '#4682b4' # line color, blue

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def loadcpkl(cpklfile):
    """
    Load cpkl files.
    """
    if not os.path.isfile(cpklfile): raise ValueError(' ERR: cannot find the input file')
    f    = open(cpklfile, 'rb')#, encoding='ISO-8859-1')

    if sys.version_info.major == 2:
        data = pickle.load(f)
    elif sys.version_info.major == 3:
        data = pickle.load(f, encoding='latin-1')

    f.close()
    return data


###############
def plot_sfh_pcl2(ID0, PA, Z=np.arange(-1.2,0.4249,0.05), age=[0.01, 0.1, 0.3, 0.7, 1.0, 3.0], f_comp = 0, fil_path = './FILT/', inputs=None, dust_model=0, DIR_TMP='./templates/'):
    #
    #
    #
    import cosmolopy.distance as cd
    import cosmolopy.constants as cc
    cosmo = {'omega_M_0' : 0.27, 'omega_lambda_0' : 0.73, 'h' : 0.72}
    cosmo = cd.set_omega_k_0(cosmo)

    flim = 0.01
    lsfrl = -1 # log SFR low limit
    mmax  = 1000
    Txmax = 4 # Max x value
    lmmin = 10.3

    nage = np.arange(0,len(age),1)
    fnc  = Func(Z, nage, dust_model=dust_model) # Set up the number of Age/ZZ
    bfnc = Basic(Z)

    age = np.asarray(age)

    ################
    # RF colors.
    import os.path
    home = os.path.expanduser('~')
    c      = 3.e18 # A/s
    chimax = 1.
    mag0   = 25.0
    d      = 10**(73.6/2.5) * 1e-18 # From [ergs/s/cm2/A] to [ergs/s/cm2/Hz]
    #d = 10**(-73.6/2.5) # From [ergs/s/cm2/Hz] to [ergs/s/cm2/A]

    #############
    # Plot.
    #############
    fig = plt.figure(figsize=(8,2.8))
    fig.subplots_adjust(top=0.88, bottom=0.16, left=0.07, right=0.99, hspace=0.15, wspace=0.3)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    #ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(133)

    ax1t = ax1.twiny()
    ax2t = ax2.twiny()
    #ax3t = ax3.twiny()
    ax4t = ax4.twiny()

    ##################
    # Fitting Results
    ##################
    #DIR_TMP = './templates/'
    SNlim = 3 # avobe which SN line is shown.

    ###########################
    # Open result file
    ###########################
    file = 'summary_' + ID0 + '_PA' + PA + '.fits'
    hdul = fits.open(file) # open a FITS file
    zbes = hdul[0].header['z']
    chinu= hdul[1].data['chi']

    uv= hdul[1].data['uv']
    vj= hdul[1].data['vj']

    RA   = 0
    DEC  = 0
    rek  = 0
    erekl= 0
    ereku= 0
    mu = 1.0
    nn = 0
    qq = 0
    enn = 0
    eqq = 0
    try:
        RA   = hdul[0].header['RA']
        DEC  = hdul[0].header['DEC']
    except:
        catdir = '/Users/tmorishita/Box Sync/Research/fsps/3dhst/'
        if ID0 == '00141' or ID0 == '00227':
            if ID0 == '00141':
                mu = 1.85
                rek  = np.log10(2.0/mu)
                erekl= 0.1
                ereku= 0.1
                nn = 8.0
                qq = 0.9
                enn = .1
                eqq = .1
            if ID0 == '00227':
                mu = 1.68
                rek  = np.log10(1.8/mu)
                erekl= 0.1
                ereku= 0.1
                nn = 8.0
                qq = 0.9
                enn = .1
                eqq = .1

            try:
                fd = np.loadtxt(catdir+'M1149.cat', comments='#')
                for ii in range(len(fd[:,0])):
                    if fd[ii,0] == int(ID0):
                        RA  = fd[ii,1]
                        DEC = fd[ii,2]
            except:
                RA  = 0
                DEC = 0
        else:
            try:
                fd = np.loadtxt(catdir+'all_str.cat', comments='#')
                ids  = fd[:,0]
                re   = fd[:,6] # in arcsec
                ere  = fd[:,7]
                n    = fd[:,8]
                en   = fd[:,9]
                q    = fd[:,10]
                eq   = fd[:,11]
                for ii in range(len(fd[:,0])):
                    if fd[ii,0] == int(ID0):
                        RA  = fd[ii,1]
                        DEC = fd[ii,2]

                        nn = n[ii]
                        qq = q[ii]
                        enn =en[ii]
                        eqq =eq[ii]

                        dA   = cd.angular_diameter_distance(float(zbes), **cosmo) #AngDiamet
                        dkpc = dA*(2*3.14/360/3600)*10**3 #kpc/arcsec
                        pix  = 1.0

                        rek   = np.log10(re[ii] * pix * dkpc)
                        erekl = rek - np.log10((re[ii]-ere[ii]) * pix * dkpc)
                        ereku = np.log10((re[ii]+ere[ii]) * pix * dkpc) - rek
                        if erekl < 0.01:
                            erekl = 0.01
                        if ereku < 0.01:
                            ereku = 0.01

            except:
                RA  = 0
                DEC = 0


    try:
        SN = hdul[0].header['SN']
    except:
        ###########################
        # Get SN of Spectra
        ###########################
        file = 'templates/spec_obs_' + ID0 + '_PA' + PA + '.cat'
        fds  = np.loadtxt(file, comments='#')
        nrs  = fds[:,0]
        lams = fds[:,1]
        fsp  = fds[:,2]
        esp  = fds[:,3]

        consp = (nrs<10000) & (lams/(1.+zbes)>3600) & (lams/(1.+zbes)<4200)
        if len((fsp/esp)[consp]>10):
            SN = np.median((fsp/esp)[consp])
        else:
            SN = 1


    Asum = 0
    A50 = np.arange(len(age), dtype='float32')
    for aa in range(len(A50)):
        A50[aa] = hdul[1].data['A'+str(aa)][1]
        Asum += A50[aa]

    ####################
    # For cosmology
    ####################
    import cosmolopy.distance as cd
    import cosmolopy.constants as cc
    cosmo = {'omega_M_0' : 0.3, 'omega_lambda_0' : 0.7, 'h' : 0.72}
    cosmo = cd.set_omega_k_0(cosmo)

    Lsun = 3.839 * 1e33 #erg s-1
    Mpc_cm = 3.08568025e+24 # cm/Mpc
    DL = cd.luminosity_distance(zbes, **cosmo) * Mpc_cm # Luminositydistance in cm
    Cons = (4.*np.pi*DL**2/(1.+zbes))

    Tuni = cd.age(zbes, use_flat=True, **cosmo)
    Tuni0 = (Tuni/cc.Gyr_s - age[:])

    delT  = np.zeros(len(age),dtype='float32')
    delTl = np.zeros(len(age),dtype='float32')
    delTu = np.zeros(len(age),dtype='float32')
    for aa in range(len(age)):
        if aa == 0:
            delTl[aa] = age[aa]
            delTu[aa] = (age[aa+1]-age[aa])/2.
            delT[aa]  = delTu[aa] + delTl[aa]
        elif Tuni/cc.Gyr_s < age[aa]:
            delTl[aa] = (age[aa]-age[aa-1])/2.
            delTu[aa] = delTl[aa] #10.
            delT[aa]  = delTu[aa] + delTl[aa]
        elif aa == len(age)-1:
            delTl[aa] = (age[aa]-age[aa-1])/2.
            delTu[aa] = Tuni/cc.Gyr_s - age[aa]
            delT[aa]  = delTu[aa] + delTl[aa]
        else:
            delTl[aa] = (age[aa]-age[aa-1])/2.
            delTu[aa] = (age[aa+1]-age[aa])/2.
            delT[aa]  = delTu[aa] + delTl[aa]


    delT[:]  *= 1e9 # Gyr to yr
    delTl[:] *= 1e9 # Gyr to yr
    delTu[:] *= 1e9 # Gyr to yr
    #print(age, delT, delTu, delTl)
    ##############################
    # Load Pickle
    ##############################
    samplepath = './'
    pfile = 'chain_' + ID0 + '_PA' + PA + '_corner.cpkl'

    niter = 0
    data = loadcpkl(os.path.join(samplepath+'/'+pfile))
    try:
    #if 1>0:
        ndim   = data['ndim']     # By default, use ndim and burnin values contained in the cpkl file, if present.
        burnin = data['burnin']
        nmc    = data['niter']
        nwalk  = data['nwalkers']
        Nburn  = burnin #* nwalk/10/2 # I think this takes 3/4 of samples
        #if nmc>1000:
        #    Nburn  = 500
        samples = data['chain'][:]
    except:
        print(' =   >   NO keys of ndim and burnin found in cpkl, use input keyword values')
        return -1

    ######################
    # Mass-to-Light ratio.
    ######################
    #ms     = np.zeros(len(age), dtype='float32')
    # Wht do you want from MCMC sampler?
    AM = np.zeros((len(age), mmax), dtype='float32') # Mass in each bin.
    AC = np.zeros((len(age), mmax), dtype='float32') # Cumulative mass in each bin.
    AL = np.zeros((len(age), mmax), dtype='float32') # Cumulative light in each bin.
    ZM = np.zeros((len(age), mmax), dtype='float32') # Z.
    ZC = np.zeros((len(age), mmax), dtype='float32') # Cumulative Z.
    ZL = np.zeros((len(age), mmax), dtype='float32') # Light weighted cumulative Z.
    TC = np.zeros((len(age), mmax), dtype='float32') # Mass weighted T.
    TL = np.zeros((len(age), mmax), dtype='float32') # Light weighted T.
    ZMM= np.zeros((len(age), mmax), dtype='float32') # Mass weighted Z.
    ZML= np.zeros((len(age), mmax), dtype='float32') # Light weighted Z.
    SF = np.zeros((len(age), mmax), dtype='float32') # SFR
    Av = np.zeros(mmax, dtype='float32') # SFR


    # ##############################
    # Add simulated scatter in quad
    # if files are available.
    # ##############################
    if inputs:
        f_zev = int(inputs['ZEVOL'])
    else:
        f_zev = 1

    eZ_mean = 0
    try:
        #meanfile = '/Users/tmorishita/Documents/Astronomy/sim_tran/sim_SFH_mean.cat'
        meanfile = './sim_SFH_mean.cat'
        dfile    = np.loadtxt(meanfile, comments='#')
        eA = dfile[:,2]
        eZ = dfile[:,4]
        eAv= np.mean(dfile[:,6])
        if f_zev == 0:
            eZ_mean = np.mean(eZ[:])
            eZ[:]   = age * 0 #+ eZ_mean
        else:
            try:
                f_zev = int(prihdr['ZEVOL'])
                if f_zev == 0:
                    eZ_mean = np.mean(eZ[:])
                    eZ = age * 0
            except:
                pass
    except:
        print('No simulation file (%s).\nError may be underestimated.' % meanfile)
        eA = age * 0
        eZ = age * 0
        eAv= 0

    mm = 0
    #while mm<mmax:
    for mm in range(mmax):
        mtmp  = np.random.randint(len(samples))# + Nburn

        AAtmp = np.zeros(len(age), dtype='float32')
        ZZtmp = np.zeros(len(age), dtype='float32')
        mslist= np.zeros(len(age), dtype='float32')

        Av_tmp = samples['Av'][mtmp]

        f0     = fits.open(DIR_TMP + 'ms_' + ID0 + '_PA' + PA + '.fits')
        sedpar = f0[1]
        f1     = fits.open(DIR_TMP + 'ms.fits')
        mloss  = f1[1].data

        Avrand = np.random.uniform(-eAv, eAv)
        if Av_tmp + Avrand<0:
            Av[mm] = 0
        else:
            Av[mm] = Av_tmp + Avrand

        for aa in range(len(age)):
            AAtmp[aa] = samples['A'+str(aa)][mtmp]/mu
            try:
                ZZtmp[aa] = samples['Z'+str(aa)][mtmp]
            except:
                ZZtmp[aa] = samples['Z0'][mtmp]

            nZtmp      = bfnc.Z2NZ(ZZtmp[aa])
            mslist[aa] = sedpar.data['ML_'+str(nZtmp)][aa]

            ml = mloss['ms_'+str(nZtmp)][aa]

            Arand = np.random.uniform(-eA[aa],eA[aa])
            Zrand = np.random.uniform(-eZ[aa],eZ[aa])
            AM[aa, mm] = AAtmp[aa] * mslist[aa] * 10**Arand
            AL[aa, mm] = AM[aa, mm] / mslist[aa]
            SF[aa, mm] = AAtmp[aa] * mslist[aa] / delT[aa] / ml * 10**Arand
            ZM[aa, mm] = ZZtmp[aa] + Zrand
            ZMM[aa, mm]= (10 ** ZZtmp[aa]) * AAtmp[aa] * mslist[aa] * 10**Zrand
            ZML[aa, mm]= ZMM[aa, mm] / mslist[aa]

        for aa in range(len(age)):
            AC[aa, mm] = np.sum(AM[aa:, mm])
            ZC[aa, mm] = np.log10(np.sum(ZMM[aa:, mm])/AC[aa, mm])
            ZL[aa, mm] = np.log10(np.sum(ZML[aa:, mm])/np.sum(AL[aa:, mm]))
            if f_zev == 0: # To avoid random fluctuation in A.
                ZC[aa, mm] = ZM[aa, mm]

            ACs = 0
            ALs = 0
            for bb in range(aa, len(age), 1):
                tmpAA       = 10**np.random.uniform(-eA[bb],eA[bb])
                tmpTT       = np.random.uniform(-delT[bb]/1e9,delT[bb]/1e9)
                TC[aa, mm] += (age[bb]+tmpTT) * AAtmp[bb] * mslist[bb] * tmpAA
                TL[aa, mm] += (age[bb]+tmpTT) * AAtmp[bb] * tmpAA
                ACs        += AAtmp[bb] * mslist[bb] * tmpAA
                ALs        += AAtmp[bb] * tmpAA

            TC[aa, mm] /= ACs
            TL[aa, mm] /= ALs

    #Avtmp  = np.percentile(samples['Av'][:],[16,50,84])
    Avtmp  = np.percentile(Av[:],[16,50,84])

    #############
    # Plot
    #############
    AMp = np.zeros((len(age),3), dtype='float32')
    ACp = np.zeros((len(age),3), dtype='float32')
    ZMp = np.zeros((len(age),3), dtype='float32')
    ZCp = np.zeros((len(age),3), dtype='float32')
    SFp = np.zeros((len(age),3), dtype='float32')
    for aa in range(len(age)):
       AMp[aa,:] = np.percentile(AM[aa,:], [16,50,84])
       ACp[aa,:] = np.percentile(AC[aa,:], [16,50,84])
       ZMp[aa,:] = np.percentile(ZM[aa,:], [16,50,84])
       ZCp[aa,:] = np.percentile(ZC[aa,:], [16,50,84])
       SFp[aa,:] = np.percentile(SF[aa,:], [16,50,84])

    ###################
    msize = np.zeros(len(age), dtype='float32')
    for aa in range(len(age)):
        if A50[aa]/Asum>flim: # if >1%
            msize[aa] = 150 * A50[aa]/Asum

    conA = (msize>=0)

    ax1.fill_between(age[conA], np.log10(SFp[:,0])[conA], np.log10(SFp[:,2])[conA], linestyle='-', color='k', alpha=0.3)
    #ax1.fill_between(age, np.log10(SFp[:,0]), np.log10(SFp[:,2]), linestyle='-', color='k', alpha=0.3)
    ax1.scatter(age[conA], np.log10(SFp[:,1])[conA], marker='.', c='k', s=msize[conA])
    ax1.errorbar(age[conA], np.log10(SFp[:,1])[conA], xerr=[delTl[:][conA]/1e9,delTu[:][conA]/1e9], yerr=[np.log10(SFp[:,1])[conA]-np.log10(SFp[:,0])[conA], np.log10(SFp[:,2])[conA]-np.log10(SFp[:,1])[conA]], linestyle='-', color='k', lw=0.5, marker='')

    #
    # Fit with delayed exponential??
    #
    minsfr = 1e-10
    def SFH_del(t0, tau, A, tt=np.arange(0.,10,0.1)):
        sfr = np.zeros(len(tt), dtype='float32')+minsfr
        sfr[:] = A * (tt[:]-t0) * np.exp(-(tt[:]-t0)/tau)
        con = (tt[:]-t0<0)
        sfr[:][con] = minsfr
        return sfr

    def SFH_dec(t0, tau, A, tt=np.arange(0.,10,0.1)):
        sfr = np.zeros(len(tt), dtype='float32')+minsfr
        sfr[:] = A * (np.exp(-(tt[:]-t0)/tau))
        con = (tt[:]-t0<0)
        sfr[:][con] = minsfr
        return sfr

    def SFH_cons(t0, tau, A, tt=np.arange(0.,10,0.1)):
        sfr = np.zeros(len(tt), dtype='float32')+minsfr
        sfr[:] = A #* (np.exp(-(tt[:]-t0)/tau))
        con = (tt[:]<t0) | (tt[:]>tau)
        sfr[:][con] = minsfr
        return sfr

    if f_comp == 1:
        #
        # Plot exp model?
        #
        DIR_exp = '/Users/tmorishita/Documents/Astronomy/sedfitter_exp/'
        file = DIR_exp + 'summary_' + ID0 + '_PA' + PA + '.fits'
        hdul = fits.open(file) # open a FITS file

        t0  = 10**float(hdul[1].data['age'][1])
        tau = 10**float(hdul[1].data['tau'][1])
        A   = float(hdul[1].data['A'][1])
        Zexp= hdul[1].data['Z']
        Mexp= float(hdul[1].data['ms'][1])

        deltt0 = 0.01 # in Gyr
        tt0 = np.arange(0.01,10,deltt0)
        sfr = SFH_dec(np.max(age)-t0, tau, A, tt=tt0)
        mtmp = np.sum(sfr * deltt0 * 1e9)
        C_exp = Mexp/mtmp
        ax1.plot(np.max(age) - tt0, np.log10(sfr*C_exp), marker='', linestyle='--', color='r', lw=1.5, alpha=0.7, zorder=-4, label='')
        mctmp = tt0 * 0
        for ii in range(len(tt0)):
            mctmp[ii] = np.sum(sfr[:ii] * deltt0 * 1e9)
        ax2.plot(np.max(age) - tt0, np.log10(mctmp*C_exp), marker='', linestyle='--', color='r', lw=1.5, alpha=0.7, zorder=-4)
        ax4.errorbar(t0, Zexp[1], yerr=[[Zexp[1]-Zexp[0]], [Zexp[2]-Zexp[1]]], marker='s', color='r', alpha=0.7, zorder=-4, capsize=0)
        sfr_exp = sfr
        mc_exp = mctmp*C_exp
        '''
        #
        # Plot delay model?
        #
        DIR_exp = '/Users/tmorishita//Documents/Astronomy/sedfitter_del/'
        file = DIR_exp + 'summary_' + ID0 + '_PA' + PA + '.fits'
        hdul = fits.open(file) # open a FITS file

        t0  = 10**float(hdul[1].data['age'][0])
        tau = 10**float(hdul[1].data['tau'][0])
        A   = float(hdul[1].data['A'][0])
        Mdel= float(hdul[1].data['ms'][1])

        tt0 = np.arange(0.01,10,0.01)
        sfr = SFH_del(np.max(age)-t0, tau, A, tt=tt0)
        mtmp = np.sum(sfr * deltt0 * 1e9)
        C_del = Mdel/mtmp
        '''

    #
    # Mass in each bin
    #
    ax2label = ''
    ax2.fill_between(age[conA], np.log10(ACp[:,0])[conA], np.log10(ACp[:,2])[conA], linestyle='-', color='k', alpha=0.3)
    ax2.errorbar(age[conA], np.log10(ACp[:,1])[conA], xerr=[delTl[:][conA]/1e9,delTu[:][conA]/1e9], yerr=[np.log10(ACp[:,1])[conA]-np.log10(ACp[:,0])[conA],np.log10(ACp[:,2])[conA]-np.log10(ACp[:,1])[conA]], linestyle='-', color='k', lw=0.5, label=ax2label)
    ax2.scatter(age[conA], np.log10(ACp[:,1])[conA], marker='.', c='k', s=msize)

    y2min = np.max([lmmin,np.min(np.log10(ACp[:,0])[conA])])
    y2max = np.max(np.log10(ACp[:,2])[conA])+0.05

    if f_comp == 1:
        y2max = np.max([y2max, np.log10(np.max(mc_exp))+0.05])
        y2min = np.min([y2min, np.log10(np.max(mc_exp))-0.05])

    if (y2max-y2min)<0.2:
        y2min -= 0.2


    #
    # Total Metal
    #
    ax4.fill_between(age[conA], ZCp[:,0][conA], ZCp[:,2][conA], linestyle='-', color='k', alpha=0.3)
    ax4.scatter(age[conA], ZCp[:,1][conA], marker='.', c='k', s=msize[conA])
    ax4.errorbar(age[conA], ZCp[:,1][conA], yerr=[ZCp[:,1][conA]-ZCp[:,0][conA],ZCp[:,2][conA]-ZCp[:,1][conA]], linestyle='-', color='k', lw=0.5)


    fw_sfr = open('SFH_' + ID0 + '_PA' + PA + '.txt', 'w')
    fw_sfr.write('# time_l time_u SFR SFR16 SFR84\n')
    fw_sfr.write('# (Gyr)  (Gyr)  (M/yr) (M/yr) (M/yr)\n')

    fw_met = open('ZH_' + ID0 + '_PA' + PA + '.txt', 'w')
    fw_met.write('# time_l time_u logZ logZ16 logZ84\n')
    fw_met.write('# (Gyr)  (Gyr)  (logZsun) (logZsun) (logZsun)\n')

    for ii in range(len(age)-1,0-1,-1):
        t0 = Tuni/cc.Gyr_s - age[ii]
        fw_sfr.write('%.2f %.2f %.2f %.2f %.2f\n'%(t0-delTl[ii]/1e9, t0+delTl[ii]/1e9, SFp[ii,1], SFp[ii,0], SFp[ii,2]))
        fw_met.write('%.2f %.2f %.2f %.2f %.2f\n'%(t0-delTl[ii]/1e9, t0+delTl[ii]/1e9, ZCp[ii,1], ZCp[ii,0], ZCp[ii,2]))
    fw_sfr.close()
    fw_met.close()

    #########################
    # Title
    #########################
    #ax1.set_title('Each $t$-bin', fontsize=12)
    #ax2.set_title('Net system', fontsize=12)
    #ax3.set_title('Each $t$-bin', fontsize=12)
    #ax4.set_title('Net system', fontsize=12)


    #############
    # Axis
    #############
    #ax1.set_xlabel('$t$ (Gyr)', fontsize=12)
    ax1.set_ylabel('$\log \dot{M_*}/M_\odot$yr$^{-1}$', fontsize=12)
    #ax1.set_ylabel('$\log M_*/M_\odot$', fontsize=12)

    lsfru = 2.8
    if np.max(np.log10(SFp[:,2]))>2.8:
        lsfru = np.max(np.log10(SFp[:,2]))+0.1

    if f_comp == 1:
        lsfru = np.max([lsfru, np.log10(np.max(sfr_exp*C_exp))])


    ax1.set_xlim(0.008, Txmax)
    ax1.set_ylim(lsfrl, lsfru)
    ax1.set_xscale('log')

    #ax2.set_xlabel('$t$ (Gyr)', fontsize=12)
    ax2.set_ylabel('$\log M_*/M_\odot$', fontsize=12)

    ax2.set_xlim(0.008, Txmax)
    ax2.set_ylim(y2min, y2max)
    ax2.set_xscale('log')

    ax2.text(0.01, y2min + 0.07*(y2max-y2min), 'ID: %s\n$z_\mathrm{obs.}:%.2f$\n$\log M_\mathrm{*}/M_\odot:%.2f$\n$\log Z_\mathrm{*}/Z_\odot:%.2f$\n$\log T_\mathrm{*}$/Gyr$:%.2f$\n$A_V$/mag$:%.2f$'%(ID0, zbes, np.log10(ACp[0,1]), ZCp[0,1], np.log10(np.percentile(TC[0,:],50)), Avtmp[1]), fontsize=9)

    #
    # Brief Summary
    #
    #namax = 7
    namax = int(len(age)-1)
    fw = open('SFH_' + ID0 + '_PA' + PA + '_pcl.cat', 'w')
    fw.write('%s %.5e %.5e %.3f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.1f %.1f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n'
    %(ID0, RA, DEC, zbes, ACp[0,1], ACp[0,1]-ACp[0,0], ACp[0,2]-ACp[0,1], ACp[namax,1]/ACp[0,1], ACp[namax,1]/ACp[0,1]-ACp[namax,0]/ACp[0,1], ACp[namax,2]/ACp[0,1]-ACp[namax,1]/ACp[0,1], ZCp[0,1], ZCp[0,1]-ZCp[0,0], ZCp[0,2]-ZCp[0,1], np.percentile(TC[0,:],50), np.percentile(TC[0,:],50)-np.percentile(TC[0,:],16),
    np.percentile(TC[0,:],84)-np.percentile(TC[0,:],50), Avtmp[1], Avtmp[1]-Avtmp[0], Avtmp[2]-Avtmp[1], uv[1], uv[1]-uv[0], uv[2]-uv[1], vj[1], vj[1]-vj[0], vj[2]-vj[1], chinu[1], SN, rek, nn, qq, (erekl+ereku)/2., enn, eqq, np.percentile(TL[0,:],50), np.percentile(TL[0,:],50)-np.percentile(TL[0,:],16), np.percentile(TL[0,:],84)-np.percentile(TL[0,:],50), np.percentile(ZL[0,:],50), np.percentile(ZL[0,:],50)-np.percentile(ZL[0,:],16), np.percentile(ZL[0,:],84)-np.percentile(ZL[0,:],50)))
    fw.close()


    #
    # SFH
    #
    zzall = np.arange(1.,12,0.01)
    Tall  = cd.age(zzall, use_flat=True, **cosmo)/cc.Gyr_s

    fw = open('SFH_' + ID0 + '_PA' + PA + '_sfh.cat', 'w')
    fw.write('%s'%(ID0))
    for mm in range(len(age)):
        mmtmp = np.argmin(np.abs(Tall - Tuni0[mm]))
        zztmp = zzall[mmtmp]
        fw.write(' %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f'%(zztmp, Tuni0[mm], np.log10(ACp[mm,1]), (np.log10(ACp[mm,1])-np.log10(ACp[mm,0])), (np.log10(ACp[mm,2])-np.log10(ACp[mm,1])), ZCp[mm,1], ZCp[mm,1]-ZCp[mm,0], ZCp[mm,2]-ZCp[mm,1], SFp[mm,1], SFp[mm,1]-SFp[mm,0], SFp[mm,2]-SFp[mm,1]))
    fw.write('\n')
    fw.close()
    #print('%s & $%.5e$ & $%.5e$ & $%.3f$ & $%.2f_{-%.2f}^{+%.2f}$ & $%.2f_{-%.2f}^{+%.2f}$ & $%.2f_{-%.2f}^{+%.2f}$ & $%.2f_{-%.2f}^{+%.2f}$ & $%.2f_{-%.2f}^{+%.2f}$ & $%.2f_{-%.2f}^{+%.2f}$ & $%.2f_{-%.2f}^{+%.2f}$ & $%.1f$ & $%.1f$ & $%.2f$\\\\'%(ID0, RA, DEC, zbes, np.log10(ACp[0,1]), (np.log10(ACp[0,1])-np.log10(ACp[0,0])), (np.log10(ACp[0,2])-np.log10(ACp[0,1])), ACp[7,1]/ACp[0,1], ACp[7,1]/ACp[0,1]-ACp[7,0]/ACp[0,1], ACp[7,2]/ACp[0,1]-ACp[7,1]/ACp[0,1], ZCp[0,1], ZCp[0,1]-ZCp[0,0], ZCp[0,2]-ZCp[0,1], np.percentile(TC[0,:],50), np.percentile(TC[0,:],50)-np.percentile(TC[0,:],16), np.percentile(TC[0,:],84)-np.percentile(TC[0,:],50), Avtmp[1], Avtmp[1]-Avtmp[0], Avtmp[2]-Avtmp[1], uv[1], uv[1]-uv[0], uv[2]-uv[1], vj[1], vj[1]-vj[0], vj[2]-vj[1], chinu[1], SN, rek))

    dely2 = 0.1
    while (y2max-y2min)/dely2>7:
        dely2 *= 2.

    y2ticks = np.arange(y2min, y2max, dely2)
    ax2.set_yticks(y2ticks)
    ax2.set_yticklabels(np.arange(y2min, y2max, 0.1), minor=False)

    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    #ax2.yaxis.set_major_locator(plt.MaxNLocator(4))
    #ax3.set_xlabel('$t$ (Gyr)', fontsize=12)
    #ax3.set_ylabel('$\log Z_*/Z_\odot$', fontsize=12)
    y3min, y3max = np.min(Z), np.max(Z)
    #ax3.set_xlim(0.008, Txmax)
    #ax3.set_ylim(y3min, y3max)
    #ax3.set_xscale('log')
    #ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    fwt = open('T_' + ID0 + '_PA' + PA + '_sfh.cat', 'w')
    fwt.write('%s %.3f %.3f %.3f'%(ID0, np.percentile(TC[0,:],50), np.percentile(TC[0,:],16), np.percentile(TC[0,:],84)))
    fwt.close()

    # For redshift
    if zbes<4:
        if zbes<2:
            zred  = [zbes, 2, 3, 6]
            #zredl = ['$z_\mathrm{obs.}$', 2, 3, 6]
            zredl = ['$z_\mathrm{obs.}$', 2, 3, 6]
        elif zbes<2.5:
            zred  = [zbes, 2.5, 3, 6]
            zredl = ['$z_\mathrm{obs.}$', 2.5, 3, 6]
        elif zbes<3.:
            zred  = [zbes, 3, 6]
            zredl = ['$z_\mathrm{obs.}$', 3, 6]
    else:
        zred  = [zbes, 6, 7, 9, 15]
        zredl = ['$z_\mathrm{obs.}$', 6, 7, 9, 15]

    Tzz   = np.zeros(len(zred), dtype='float32')
    for zz in range(len(zred)):
        Tzz[zz] = (Tuni - cd.age(zred[zz], use_flat=True, **cosmo))/cc.Gyr_s
        if Tzz[zz] < 0.01:
            Tzz[zz] = 0.01

    #print(zred, Tzz)
    #ax3t.set_xscale('log')
    #ax3t.set_xlim(0.008, Txmax)

    ax1.set_xlabel('$t_\mathrm{lookback}$/Gyr', fontsize=12)
    ax2.set_xlabel('$t_\mathrm{lookback}$/Gyr', fontsize=12)
    ax4.set_xlabel('$t_\mathrm{lookback}$/Gyr', fontsize=12)
    ax4.set_ylabel('$\log Z_*/Z_\odot$', fontsize=12)

    ax1t.set_xscale('log')
    ax1t.set_xlim(0.008, Txmax)
    ax2t.set_xscale('log')
    ax2t.set_xlim(0.008, Txmax)
    ax4t.set_xscale('log')
    ax4t.set_xlim(0.008, Txmax)

    ax4.set_xlim(0.008, Txmax)
    ax4.set_ylim(y3min-0.05, y3max)
    ax4.set_xscale('log')

    ax4.set_yticks([-0.8, -0.4, 0., 0.4])
    ax4.set_yticklabels(['-0.8', '-0.4', '0', '0.4'])
    #ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    #ax3.yaxis.labelpad = -2
    ax4.yaxis.labelpad = -2


    ax1t.set_xticklabels(zredl[:])
    ax1t.set_xticks(Tzz[:])
    ax1t.tick_params(axis='x', labelcolor='k')
    ax1t.xaxis.set_ticks_position('none')
    #ax1t.plot(Tzz, Tzz*0+y3max+(y3max-y3min)*.00, marker='|', color='k', ms=3, linestyle='None')

    ax2t.set_xticklabels(zredl[:])
    ax2t.set_xticks(Tzz[:])
    ax2t.tick_params(axis='x', labelcolor='k')
    ax2t.xaxis.set_ticks_position('none')
    #ax2t.plot(Tzz, Tzz*0+y3max+(y3max-y3min)*.00, marker='|', color='k', ms=3, linestyle='None')

    ax4t.set_xticklabels(zredl[:])
    ax4t.set_xticks(Tzz[:])
    ax4t.tick_params(axis='x', labelcolor='k')
    ax4t.xaxis.set_ticks_position('none')
    ax4t.plot(Tzz, Tzz*0+y3max+(y3max-y3min)*.00, marker='|', color='k', ms=3, linestyle='None')

    ax1.plot(Tzz, Tzz*0+lsfru+(lsfru-lsfrl)*.00, marker='|', color='k', ms=3, linestyle='None')
    ax2.plot(Tzz, Tzz*0+y2max+(y2max-y2min)*.00, marker='|', color='k', ms=3, linestyle='None')

    ####################
    ## Save
    ####################
    #plt.show()
    #ax1.legend(loc=2, fontsize=8)
    #ax2.legend(loc=3, fontsize=8)
    plt.savefig('SFH_' + ID0 + '_PA' + PA + '_pcl.pdf')
    #if f_comp == 1:
    #    plt.savefig('SFH_' + ID0 + '_PA' + PA + '_comp.pdf')
    #else:
    #    plt.savefig('SFH_' + ID0 + '_PA' + PA + '_pcl.pdf')
