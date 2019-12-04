#!/usr/bin/env python
#<examples/doc_nistgauss.py>
import numpy as np
import sys
import matplotlib.pyplot as plt
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
from .function_igm import *
from . import img_scale

import cosmolopy.distance as cd
import cosmolopy.constants as cc
cosmo = {'omega_M_0' : 0.27, 'omega_lambda_0' : 0.73, 'h' : 0.72}
cosmo = cd.set_omega_k_0(cosmo)
Lsun = 3.839 * 1e33 #erg s-1
Mpc_cm = 3.08568025e+24 # cm/Mpc

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
def plot_sfh(ID0, PA, Z=np.arange(-1.2,0.4249,0.05), age=[0.01, 0.1, 0.3, 0.7, 1.0, 3.0], f_comp = 0, fil_path = './FILT/', inputs=None, dust_model=0, DIR_TMP='./templates/',f_SFMS=False):
    #
    #
    #
    flim = 0.01
    lsfrl = -1 # log SFR low limit
    mmax  = 1000
    Txmax = 4 # Max x value
    lmmin = 9.5 #10.3

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
    fig.subplots_adjust(top=0.88, bottom=0.18, left=0.07, right=0.99, hspace=0.15, wspace=0.3)
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

    # Get SFMS;
    IMF = int(inputs['NIMF'])
    SFMS_16 = get_SFMS(zbes,age,ACp[:,0],IMF=IMF)
    SFMS_50 = get_SFMS(zbes,age,ACp[:,1],IMF=IMF)
    SFMS_84 = get_SFMS(zbes,age,ACp[:,2],IMF=IMF)

    f_rejuv,t_quench,t_rejuv = check_rejuv(age,np.log10(SFp[:,:]),np.log10(ACp[:,:]),np.log10(SFMS_50))
    #print(f_rejuv,t_quench,t_rejuv)

    # Plot MS?
    if f_SFMS:
        #ax1.fill_between(age[conA], np.log10(SFMS_16)[conA], np.log10(SFMS_84)[conA], linestyle='-', color='b', alpha=0.3)
        ax1.fill_between(age[conA], np.log10(SFMS_50)[conA]-0.2, np.log10(SFMS_50)[conA]+0.2, linestyle='-', color='b', alpha=0.3)
        ax1.plot(age[conA], np.log10(SFMS_50)[conA], linestyle='--', color='b', alpha=0.5)

    #
    # Fit with delayed exponential??
    #
    minsfr = 1e-10
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
    ax1.set_ylabel('$\log \dot{M}_*/M_\odot$yr$^{-1}$', fontsize=12)
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
    # Writing SED param in a fits file;
    # Header
    prihdr = fits.Header()
    prihdr['ID']     = ID0
    prihdr['PA']     = PA
    prihdr['z']      = zbes
    prihdr['RA']     = RA
    prihdr['DEC']    = DEC
    prihdr['Re_kpc'] = rek
    prihdr['Ser_n']  = nn
    prihdr['Axis_q'] = qq
    prihdr['e_Re']   = (erekl+ereku)/2.
    prihdr['e_n']    = enn
    prihdr['e_q']    = eqq
    # Add rejuv properties;
    prihdr['f_rejuv']= f_rejuv
    prihdr['t_quen'] = t_quench
    prihdr['t_rejuv']= t_rejuv
    prihdu = fits.PrimaryHDU(header=prihdr)

    col01 = []
    # Redshift
    zmc = hdul[1].data['zmc']
    col50 = fits.Column(name='zmc', format='E', unit='', array=zmc[:])
    col01.append(col50)

    # Stellar mass
    ACP = [np.log10(ACp[0,0]), np.log10(ACp[0,1]), np.log10(ACp[0,2])]
    col50 = fits.Column(name='Mstel', format='E', unit='logMsun', array=ACP[:])
    col01.append(col50)

    # Metallicity_MW
    ZCP = [ZCp[0,0], ZCp[0,1], ZCp[0,2]]
    col50 = fits.Column(name='Z_MW', format='E', unit='logZsun', array=ZCP[:])
    col01.append(col50)

    # Age_mw
    para = [np.log10(np.percentile(TC[0,:],16)), np.log10(np.percentile(TC[0,:],50)), np.log10(np.percentile(TC[0,:],84))]
    col50 = fits.Column(name='T_MW', format='E', unit='logGyr', array=para[:])
    col01.append(col50)

    # Metallicity_LW
    ZCP = [ZL[0,0], ZL[0,1], ZL[0,2]]
    col50 = fits.Column(name='Z_LW', format='E', unit='logZsun', array=ZCP[:])
    col01.append(col50)

    # Age_lw
    para = [np.log10(np.percentile(TL[0,:],16)), np.log10(np.percentile(TL[0,:],50)), np.log10(np.percentile(TL[0,:],84))]
    col50 = fits.Column(name='T_LW', format='E', unit='logGyr', array=para[:])
    col01.append(col50)

    # Dust
    para = [Avtmp[0], Avtmp[1], Avtmp[2]]
    col50 = fits.Column(name='AV', format='E', unit='mag', array=para[:])
    col01.append(col50)

    # U-V
    para = [uv[0], uv[1], uv[2]]
    col50 = fits.Column(name='U-V', format='E', unit='mag', array=para[:])
    col01.append(col50)

    # V-J
    para = [vj[0], vj[1], vj[2]]
    col50 = fits.Column(name='V-J', format='E', unit='mag', array=para[:])
    col01.append(col50)

    colms  = fits.ColDefs(col01)
    dathdu = fits.BinTableHDU.from_columns(colms)
    hdu    = fits.HDUList([prihdu, dathdu])
    hdu.writeto('SFH_' + ID0 + '_PA' + PA + '_param.fits', overwrite=True)


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
            zred  = [zbes, 6]
            zredl = ['$z_\mathrm{obs.}$', 6]
    else:
        zred  = [zbes, 6, 7, 9]
        zredl = ['$z_\mathrm{obs.}$', 6, 7, 9]

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
    plt.savefig('SFH_' + ID0 + '_PA' + PA + '_pcl.png')
    #if f_comp == 1:
    #    plt.savefig('SFH_' + ID0 + '_PA' + PA + '_comp.pdf')
    #else:
    #    plt.savefig('SFH_' + ID0 + '_PA' + PA + '_pcl.pdf')

###############
def get_evolv(ID0, PA, Z=np.arange(-1.2,0.4249,0.05), age=[0.01, 0.1, 0.3, 0.7, 1.0, 3.0], f_comp = 0, fil_path = './FILT/', inputs=None, dust_model=0, DIR_TMP='./templates/', delt_sfh = 0.01):
    #
    # delt_sfh (float): delta t of input SFH in Gyr.
    #
    # Returns: SED as function of age, based on SF and Z histories;
    #
    print('This function may take a while.')
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
    ##############################
    # Load Pickle
    ##############################
    samplepath = './'
    pfile = 'chain_' + ID0 + '_PA' + PA + '_corner.cpkl'

    niter = 0
    data = loadcpkl(os.path.join(samplepath+'/'+pfile))
    try:
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
    #mmax = 10
    #print('mmax is set to 10')
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
    # Make template;
    tbegin = np.min(Tuni/cc.Gyr_s-age)
    tuniv_hr = np.arange(tbegin,Tuni/cc.Gyr_s,delt_sfh) # in Gyr
    sfh_hr_in= np.interp(tuniv_hr,(Tuni/cc.Gyr_s-age)[::-1],SFp[:,1][::-1])
    zh_hr_in = np.interp(tuniv_hr,(Tuni/cc.Gyr_s-age)[::-1],ZCp[:,1][::-1])

    # FSPS
    con_sfh = (tuniv_hr>0)
    import fsps
    nimf = int(inputs['NIMF'])
    try:
        fneb = int(inputs['ADD_NEBULAE'])
    except:
        fneb = 0

    if fneb == 1:
        print('Metallicity is set to logZ/Zsun=%.2f'%(np.max(zh_hr_in)))
        sp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1, imf_type=nimf, logzsol=np.max(zh_hr_in), sfh=3, dust_type=2, dust2=0.0, add_neb_emission=True)
        sp.set_tabular_sfh(tuniv_hr[con_sfh], sfh_hr_in[con_sfh])
    else:
        sp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=3, imf_type=nimf, sfh=3, dust_type=2, dust2=0.0, add_neb_emission=False)
        sp.set_tabular_sfh(tuniv_hr[con_sfh], sfh_hr_in[con_sfh], Z=10**zh_hr_in[con_sfh])

    col01 = []
    t_get = tuniv_hr[con_sfh]
    #con_tget = ((Tuni/cc.Gyr_s-age)>0)
    #t_get = (Tuni/cc.Gyr_s-age)[con_tget][::-1]
    for ss in range(len(t_get)):
        wave0, flux0 = sp.get_spectrum(tage=t_get[ss], peraa=True) # if peraa=True, in unit of L/AA
        if ss == 0:
            spec_mul_nu_conv = np.zeros((len(t_get),len(wave0)),dtype='float32')
        #ax2.plot(wave0, flux0, linestyle='-', color='b')
        #plt.show()
        print('Template %d is done.'%(ss))
        wavetmp  = wave0*(1.+zbes)
        spec_mul_nu = flamtonu(wavetmp, flux0) # Conversion from Flambda to Fnu.
        Lsun = 3.839 * 1e33 #erg s-1
        stmp_common = 1e10 # 1 tmp is in 1e10Lsun

        spec_mul_nu_conv[ss,:] = spec_mul_nu[:]
        spec_mul_nu_conv[ss,:] *= Lsun/(4.*np.pi*DL**2/(1.+zbes))
        Ls = 10**sp.log_lbol
        spec_mul_nu_conv[ss,:] *= (1./Ls)*stmp_common # in unit of erg/s/Hz/cm2/ms[ss].

        consave = (wavetmp/(1.+zbes)<20000) # AA
        if ss == 0:
            nd_ap  = np.arange(0,len(wave0),1)
            col1   = fits.Column(name='wavelength', format='E', unit='AA', disp='obs', array=wavetmp[consave])
            col2   = fits.Column(name='colnum', format='K', unit='', array=nd_ap[consave])
            col00  = [col1, col2]
            col3   = fits.Column(name='age', format='E', unit='Gyr', array=t_get)
            col4   = fits.Column(name='sfh', format='E', unit='Msun/yr', array=sfh_hr_in[con_sfh])
            col5   = fits.Column(name='zh', format='E', unit='Zsun', array=zh_hr_in[con_sfh])
            col01  = [col3,col4,col5]

        colspec_all = fits.Column(name='fspec_'+str(ss), format='E', unit='Fnu', disp='%s'%(t_get[ss]), array=spec_mul_nu_conv[ss,:][consave])
        col00.append(colspec_all)

    coldefs_spec = fits.ColDefs(col00)
    hdu = fits.BinTableHDU.from_columns(coldefs_spec)
    hdu.writeto(DIR_TMP + 'obsspec_' + ID0 + '_PA' + PA + '.fits', overwrite=True)

    coldefs_spec = fits.ColDefs(col01)
    hdu = fits.BinTableHDU.from_columns(coldefs_spec)
    hdu.writeto(DIR_TMP + 'obshist_' + ID0 + '_PA' + PA + '.fits', overwrite=True)


###############
def plot_evolv(ID0, PA, Z=np.arange(-1.2,0.4249,0.05), age=[0.01, 0.1, 0.3, 0.7, 1.0, 3.0], f_comp = 0, fil_path = './FILT/', inputs=None, dust_model=0, DIR_TMP='./templates/', delt_sfh = 0.01, nmc=300):
    #
    # delt_sfh (float): delta t of input SFH in Gyr.
    #
    # Returns: SED as function of age, based on SF and Z histories;
    #
    ################
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

    #############
    # Plot.
    #############
    fig = plt.figure(figsize=(5,2.6))
    fig.subplots_adjust(top=0.96, bottom=0.16, left=0.12, right=0.99, hspace=0.15, wspace=0.15)
    #ax1 = fig.add_subplot(131)
    #ax2 = fig.add_subplot(132)
    #ax3 = fig.add_subplot(133)
    ax2 = fig.add_subplot(121)
    ax3 = fig.add_subplot(122)

    ###########################
    # Open result file
    ###########################
    file = 'summary_' + ID0 + '_PA' + PA + '.fits'
    hdul = fits.open(file) # open a FITS file
    zbes = hdul[0].header['z']
    chinu= hdul[1].data['chi']
    uv= hdul[1].data['uv']
    vj= hdul[1].data['vj']

    try:
        RA   = hdul[0].header['RA']
        DEC  = hdul[0].header['DEC']
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

    # Cosmo;
    DL = cd.luminosity_distance(zbes, **cosmo) * Mpc_cm # Luminositydistance in cm
    Cons = (4.*np.pi*DL**2/(1.+zbes))
    Tuni = cd.age(zbes, use_flat=True, **cosmo)
    Tuni0 = (Tuni/cc.Gyr_s - age[:])

    # Open summary;
    file = 'summary_' + ID0 + '_PA' + PA + '.fits'
    fd   = fits.open(file)[1].data
    #print(fits.open(file)[1].header)
    Avtmp = fd['Av0']
    uvtmp = fd['uv']
    vjtmp = fd['vj']
    #ax2.plot(vj[1],uv[1],color='gray',marker='s',ms=3)

    # SFH
    file = DIR_TMP + 'obshist_' + ID0 + '_PA' + PA + '.fits'
    fd   = fits.open(file)[1].data
    age  = fd['age']
    sfh  = fd['sfh']
    zh   = fd['zh']

    # Open FSPS temp;
    file = DIR_TMP + 'obsspec_' + ID0 + '_PA' + PA + '.fits'
    fd   = fits.open(file)[1].data
    wave = fd['wavelength']
    nr   = fd['colnum']
    uvall= age * 0 - 99
    vjall= age * 0 - 99

    delp = -10
    flag = False
    flag2= False
    #for ii in range(1,len(age),10):
    for ii in range(len(age)-1,-1,delp):
        flux = fd['fspec_%d'%(ii)]
        flux_att = madau_igm_abs(wave/(1.+zbes), flux, zbes)
        flux_d, xxd, nrd = dust_calz(wave/(1.+zbes), flux_att, 0.0, nr)
        #ax1.plot(xxd,flux_d,linestyle='-',lw=0.3+0.1*ii/len(age))
        band0  = ['u','v','j']
        lmconv,fconv = filconv(band0, xxd, flux_d, fil_path) # flux in fnu
        uv = -2.5*np.log10(fconv[0]/fconv[1])
        vj = -2.5*np.log10(fconv[1]/fconv[2])
        uvall[ii] = uv
        vjall[ii] = vj
        #ax2.plot(vjall[ii],uvall[ii],marker='s',ms=5,linestyle='-',zorder=5)

        if flag and not flag2:
            flux_d, xxd, nrd = dust_calz(wave/(1.+zbes), flux_att, Avtmp[1], nr)
            lmconv,fconv = filconv(band0, xxd, flux_d, fil_path) # flux in fnu
            uv_av = -2.5*np.log10(fconv[0]/fconv[1])
            vj_av = -2.5*np.log10(fconv[1]/fconv[2])
            delvj, deluv = vj_av-vj, uv_av-uv
            flag2 = True
        flag = True

    #import matplotlib.colormaps as cm
    conuvj = (vjall>-90)&(uvall>-90)
    ax2.plot(vjall[conuvj],uvall[conuvj],marker='s',markeredgecolor='k',color='none',ms=5,zorder=4)
    ax2.scatter(vjall[conuvj],uvall[conuvj],marker='s',c=age[conuvj],cmap='jet',s=8,zorder=5)
    ax2.plot(vjall[conuvj],uvall[conuvj],marker='',color='k',ms=3,linestyle='-',zorder=3)
    ax3.plot(age,np.log10(sfh),color='b', linewidth=1, linestyle='-',label=r'$\log$SFR$/M_\odot$yr$^{-1}$')
    ax3.plot(age,zh,color='r', linewidth=1, linestyle='-',label=r'$\log Z_*/Z_\odot$')

    '''
    ax1.set_xlim(1000,40000)
    ax1.set_xscale('log')
    ax1.set_ylim(1e-2,1e2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Wavelength')
    '''

    ax2.set_xlim(-0.3,2.3)
    ax2.set_ylim(-0.3,2.3)

    ax2.set_xlabel('$V-J\,$/ mag')
    ax2.set_ylabel('$U-V\,$/ mag')
    ax3.set_xlabel('age / Gyr')
    #ax3.set_ylabel('$\log$SFR$/M_\odot$yr$^{-1}$ or $\log Z_*/Z_\odot$')

    ##
    prog_path = '/Users/tmorishita/GitHub/gsf/gsf/example/misc/'
    data_uvj = np.loadtxt(prog_path+'f2.cat',comments='#')
    x=data_uvj[:,0]
    y=data_uvj[:,1]
    ax2.plot(x,y,color="gray",lw=1,ls="-")

    data_uvj = np.loadtxt(prog_path+'g2.cat',comments='#')
    x=data_uvj[:,0]
    y=data_uvj[:,1]
    ax2.plot(x,y,color="gray",lw=1,ls="-")

    data_uvj = np.loadtxt(prog_path+'h2.cat',comments='#')
    x=data_uvj[:,0]
    y=data_uvj[:,1]
    ax2.plot(x,y,color="gray",lw=1,ls="-")

    try:
        av=np.array([1.2,0.,delvj,deluv])
        X,Y,U,V=zip(av)
        ax2.quiver(X,Y,U,V,angles='xy',scale_units='xy',scale=1,linewidth=1,color="k")
    except:
        pass

    ax2.text(-0.1,2.1,'Quiescent',fontsize=11,color='orangered')
    ax2.text(1.3,-0.2,'Starforming',fontsize=11,color='royalblue')

    ax3.legend(loc=3)
    #plt.show()
    plt.savefig('hist_' + ID0 + '_PA' + PA + '.pdf')
