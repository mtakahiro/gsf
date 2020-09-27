import numpy as np
import sys

import matplotlib.pyplot as plt

from numpy import log10
from scipy.integrate import simps
from astropy.io import fits
import os
from matplotlib.ticker import FormatStrFormatter

# Custom modules
from .function import *
from .function_class import Func
from .basic_func import Basic
from .function_igm import *
#from . import img_scale

lcb   = '#4682b4' # line color, blue

def plot_sfh(MB, f_comp=0, flim=0.01, lsfrl=-1, mmax=1000, Txmin=0.08, Txmax=4, lmmin=7.5, fil_path='./FILT/', \
    inputs=None, dust_model=0, DIR_TMP='./templates/',f_SFMS=False, verbose=False, f_silence=True):

    '''
    Purpose:
    ========
    Star formation history plot.

    Input:
    ======
    flim  : Lower limit for plotting an age bin.
    lsfrl : Lower limit for SFR, in logMsun/yr

    '''
    import os.path
    import time

    if f_silence:
        import matplotlib
        matplotlib.use("Agg")

    fnc  = MB.fnc #Func(ID, PA, Z, nage, dust_model=dust_model, DIR_TMP=DIR_TMP) # Set up the number of Age/ZZ
    bfnc = MB.bfnc #Basic(Z)
    ID   = MB.ID
    PA   = MB.PA
    Z    = MB.Zall
    age  = MB.age  #[0.01, 0.1, 0.3, 0.7, 1.0, 3.0],
    nage = MB.nage
    '''try:
        age = MB.age_fix
    except:
        age  = MB.age
    '''
    tau0 = MB.tau0 #[0.1,0.2,0.3]
    age = np.asarray(age)

    if Txmin > np.min(age):
        Txmin = np.min(age) * 0.8

    ################
    # RF colors.
    home = os.path.expanduser('~')
    c      = MB.c
    chimax = 1.
    m0set  = MB.m0set
    Mpc_cm = MB.Mpc_cm
    d      = 10**(73.6/2.5) * 1e-18 # From [ergs/s/cm2/A] to [ergs/s/cm2/Hz]

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
    file = 'summary_' + ID + '_PA' + PA + '.fits'
    hdul = fits.open(file) # open a FITS file
    zbes = hdul[0].header['z']
    chinu= hdul[1].data['chi']

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
        file = 'templates/spec_obs_' + ID + '_PA' + PA + '.cat'
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
        A50[aa] = 10**hdul[1].data['A'+str(aa)][1]
        Asum += A50[aa]

    ####################
    # For cosmology
    ####################
    DL = MB.cosmo.luminosity_distance(zbes).value * Mpc_cm # Luminositydistance in cm
    Cons = (4.*np.pi*DL**2/(1.+zbes))

    Tuni = MB.cosmo.age(zbes).value #, use_flat=True, **cosmo)
    Tuni0 = (Tuni - age[:])

    delT  = np.zeros(len(age),dtype='float32')
    delTl = np.zeros(len(age),dtype='float32')
    delTu = np.zeros(len(age),dtype='float32')

    if len(age) == 1:
        for aa in range(len(age)):
            try:
                tau_ssp = float(inputs['TAU_SSP'])
            except:
                tau_ssp = 0.01
            delTl[aa] = tau_ssp/2
            delTu[aa] = tau_ssp/2
            delT[aa]  = delTu[aa] + delTl[aa]

    else: # This is only true when CSP...
        for aa in range(len(age)):
            if aa == 0:
                delTl[aa] = age[aa]
                delTu[aa] = (age[aa+1]-age[aa])/2.
                delT[aa]  = delTu[aa] + delTl[aa]
            elif Tuni < age[aa]:
                delTl[aa] = (age[aa]-age[aa-1])/2.
                delTu[aa] = delTl[aa] #10.
                delT[aa]  = delTu[aa] + delTl[aa]
            elif aa == len(age)-1:
                delTl[aa] = (age[aa]-age[aa-1])/2.
                delTu[aa] = Tuni - age[aa]
                delT[aa]  = delTu[aa] + delTl[aa]
            else:
                delTl[aa] = (age[aa]-age[aa-1])/2.
                delTu[aa] = (age[aa+1]-age[aa])/2.
                delT[aa]  = delTu[aa] + delTl[aa]

            if delT[aa] <= 0:
                delT[aa] = 1e10

    delT[:]  *= 1e9 # Gyr to yr
    delTl[:] *= 1e9 # Gyr to yr
    delTu[:] *= 1e9 # Gyr to yr

    ##############################
    # Load Pickle
    ##############################
    samplepath = './'
    pfile = 'chain_' + ID + '_PA' + PA + '_corner.cpkl'

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
        if verbose:
            print('No simulation file (%s).\nError may be underestimated.' % meanfile)
        eA = age * 0
        eZ = age * 0
        eAv= 0

    mm = 0

    #####################
    # Get SED based SFR
    #####################
    tset_SFR_SED = 0.03 # Gyr
    SFR_SED = np.zeros(mmax,dtype='float32')

    # base files opened.
    f0     = fits.open(DIR_TMP + 'ms_' + ID + '_PA' + PA + '.fits')
    f1     = fits.open(DIR_TMP + 'ms.fits')
    sedpar = f0[1]
    mloss  = f1[1].data

    AAtmp = np.zeros(len(age), dtype='float32')
    ZZtmp = np.zeros(len(age), dtype='float32')
    mslist= np.zeros(len(age), dtype='float32')

    for mm in range(mmax):
        delt_tot = 0
        mtmp  = np.random.randint(len(samples))# + Nburn

        try:
            Av_tmp = samples['Av'][mtmp]
        except:
            Av_tmp = MB.AVFIX

        Avrand = np.random.uniform(-eAv, eAv)
        if Av_tmp + Avrand<0:
            Av[mm] = 0
        else:
            Av[mm] = Av_tmp + Avrand

        for aa in range(len(age)):
            try:
                # This is in log.
                AAtmp[aa] = samples['A'+str(aa)][mtmp]
            except:
                AAtmp[aa] = -10
                pass

            try:
                ZZtmp[aa] = samples['Z'+str(aa)][mtmp]
            except:
                try:
                    ZZtmp[aa] = samples['Z0'][mtmp]
                except:
                    ZZtmp[aa] = MB.ZFIX

            nZtmp      = bfnc.Z2NZ(ZZtmp[aa])
            mslist[aa] = sedpar.data['ML_'+str(nZtmp)][aa]

            ml = mloss['ms_'+str(nZtmp)][aa]

            Arand = np.random.uniform(-eA[aa],eA[aa])
            Zrand = np.random.uniform(-eZ[aa],eZ[aa])

            # quantity in log scale;
            AM[aa, mm] = AAtmp[aa] + np.log10(mslist[aa]) + Arand 
            AL[aa, mm] = AM[aa,mm] - np.log10(mslist[aa])
            SF[aa, mm] = AAtmp[aa] + np.log10(mslist[aa] / delT[aa]) + Arand # / ml
            ZM[aa, mm] = ZZtmp[aa] + Zrand
            ZMM[aa, mm]= ZZtmp[aa] + AAtmp[aa] + np.log10(mslist[aa]) + Zrand
            ZML[aa, mm]= ZMM[aa,mm] - np.log10(mslist[aa])

            # SFR from SED. This will be converted in log later;
            if age[aa]<=tset_SFR_SED:
                SFR_SED[mm] += 10**SF[aa, mm] * delT[aa]
                delt_tot    += delT[aa]

        SFR_SED[mm] /= delt_tot
        SFR_SED[mm] = np.log10(SFR_SED[mm])
        for aa in range(len(age)):
            AC[aa, mm] = np.log10(np.sum(10**AM[aa:,mm]))
            ZC[aa, mm] = np.log10(np.sum(10**ZMM[aa:,mm])/10**AC[aa, mm])
            ZL[aa, mm] = np.log10(np.sum(10**ZML[aa:,mm])/np.sum(10**AL[aa:,mm]))
            if f_zev == 0: # To avoid random fluctuation in A.
                ZC[aa,mm] = ZM[aa,mm]

            ACs = 0
            ALs = 0
            for bb in range(aa, len(age), 1):
                tmpAA       = 10**np.random.uniform(-eA[bb],eA[bb])
                tmpTT       = np.random.uniform(-delT[bb]/1e9,delT[bb]/1e9)
                TC[aa, mm] += (age[bb]+tmpTT) * 10**AAtmp[bb] * mslist[bb] * tmpAA
                TL[aa, mm] += (age[bb]+tmpTT) * 10**AAtmp[bb] * tmpAA
                ACs        += 10**AAtmp[bb] * mslist[bb] * tmpAA
                ALs        += 10**AAtmp[bb] * tmpAA

            TC[aa, mm] /= ACs
            TL[aa, mm] /= ALs
            if TC[aa, mm]>0:
                TC[aa, mm] = np.log10(TC[aa, mm])
            if TL[aa, mm]>0:
                TL[aa, mm] = np.log10(TL[aa, mm])

        # Do stuff...
        time.sleep(0.01)
        # Update Progress Bar
        printProgressBar(mm, mmax, prefix = 'Progress:', suffix = 'Complete', length = 40)


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

    SFR_SED_med = np.percentile(SFR_SED[:],[16,50,84])
    f_SFRSED_plot = False
    if f_SFRSED_plot:
        ax1.errorbar(delt_tot/2./1e9, SFR_SED_med[1], xerr=[[delt_tot/2./1e9],[delt_tot/2./1e9]], \
        yerr=[[SFR_SED_med[1]-SFR_SED_med[0]],[SFR_SED_med[2]-SFR_SED_med[1]]], \
        linestyle='', color='orange', lw=1., marker='*',ms=8,zorder=-2)

    ###################
    msize = np.zeros(len(age), dtype='float32')
    for aa in range(len(age)):
        if A50[aa]/Asum>flim: # if >1%
            msize[aa] = 150 * A50[aa]/Asum

    conA = (msize>=0)
    ax1.fill_between(age[conA], SFp[:,0][conA], SFp[:,2][conA], linestyle='-', color='k', alpha=0.3)
    ax1.scatter(age[conA], SFp[:,1][conA], marker='.', c='k', s=msize[conA])
    ax1.errorbar(age[conA], SFp[:,1][conA], xerr=[delTl[:][conA]/1e9,delTu[:][conA]/1e9], yerr=[SFp[:,1][conA]-SFp[:,0][conA], SFp[:,2][conA]-SFp[:,1][conA]], linestyle='-', color='k', lw=0.5, marker='')


    #############
    # Get SFMS in log10;
    #############
    IMF = int(inputs['NIMF'])
    SFMS_16 = get_SFMS(zbes,age,10**ACp[:,0],IMF=IMF)
    SFMS_50 = get_SFMS(zbes,age,10**ACp[:,1],IMF=IMF)
    SFMS_84 = get_SFMS(zbes,age,10**ACp[:,2],IMF=IMF)

    try:
        f_rejuv,t_quench,t_rejuv = check_rejuv(age,SFp[:,:],ACp[:,:],SFMS_50)
    except:
        print('Rejuvenation judge failed. (plot_sfh.py)')
        f_rejuv,t_quench,t_rejuv = 0,0,0

    # Plot MS?
    if f_SFMS:
        ax1.fill_between(age[conA], SFMS_50[conA]-0.2, SFMS_50[conA]+0.2, linestyle='-', color='b', alpha=0.3)
        ax1.plot(age[conA], SFMS_50[conA], linestyle='--', color='b', alpha=0.5)


    #
    # Mass in each bin
    #
    ax2label = ''
    ax2.fill_between(age[conA], ACp[:,0][conA], ACp[:,2][conA], linestyle='-', color='k', alpha=0.3)
    ax2.errorbar(age[conA], ACp[:,1][conA], xerr=[delTl[:][conA]/1e9,delTu[:][conA]/1e9], yerr=[ACp[:,1][conA]-ACp[:,0][conA],ACp[:,2][conA]-ACp[:,1][conA]], linestyle='-', color='k', lw=0.5, label=ax2label)
    ax2.scatter(age[conA], ACp[:,1][conA], marker='.', c='k', s=msize)

    y2min = np.max([lmmin,np.min(ACp[:,0][conA])])
    y2max = np.max(ACp[:,2][conA])+0.05

    if (y2max-y2min)<0.2:
        y2min -= 0.2

    #
    # Total Metal
    #
    ax4.fill_between(age[conA], ZCp[:,0][conA], ZCp[:,2][conA], linestyle='-', color='k', alpha=0.3)
    ax4.scatter(age[conA], ZCp[:,1][conA], marker='.', c='k', s=msize[conA])
    ax4.errorbar(age[conA], ZCp[:,1][conA], yerr=[ZCp[:,1][conA]-ZCp[:,0][conA],ZCp[:,2][conA]-ZCp[:,1][conA]], linestyle='-', color='k', lw=0.5)

    fw_sfr = open('SFH_' + ID + '_PA' + PA + '.txt', 'w')
    fw_sfr.write('# time_l time_u logSFR16 logSFR50 logSFR84 logMstel16 logMstel50 logMstel84 logZ16 logZ50 logZ84\n')
    fw_sfr.write('# (Gyr)  (Gyr)  (M/yr) (M/yr) (M/yr)  (M) (M) (M)  (logZsun) (logZsun) (logZsun)\n')

    for ii in range(len(age)-1,0-1,-1):
        t0 = Tuni - age[ii]
        fw_sfr.write('%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n'\
        %(t0-delTl[ii]/1e9, t0+delTl[ii]/1e9, SFp[ii,0], SFp[ii,1], SFp[ii,2], ACp[ii,0], ACp[ii,1], ACp[ii,2], ZCp[ii,0], ZCp[ii,1], ZCp[ii,2]))
    fw_sfr.close()

    #############
    # Axis
    #############
    ax1.set_ylabel('$\log \dot{M}_*/M_\odot$yr$^{-1}$', fontsize=12)

    lsfru = 2.8
    if np.max(SFp[:,2])>2.8:
        lsfru = np.max(SFp[:,2])+0.1

    ax1.set_xlim(Txmin, Txmax)
    ax1.set_ylim(lsfrl, lsfru)
    ax1.set_xscale('log')

    ax2.set_ylabel('$\log M_*/M_\odot$', fontsize=12)

    ax2.set_xlim(Txmin, Txmax)
    ax2.set_ylim(y2min, y2max)
    ax2.set_xscale('log')

    ax2.text(0.01, y2min + 0.07*(y2max-y2min), 'ID: %s\n$z_\mathrm{obs.}:%.2f$\n$\log M_\mathrm{*}/M_\odot:%.2f$\n$\log Z_\mathrm{*}/Z_\odot:%.2f$\n$\log T_\mathrm{*}$/Gyr$:%.2f$\n$A_V$/mag$:%.2f$'\
        %(ID, zbes, ACp[0,1], ZCp[0,1], np.nanmedian(TC[0,:]), Avtmp[1]), fontsize=9)

    #
    # Brief Summary
    #
    # Writing SED param in a fits file;
    # Header
    prihdr = fits.Header()
    prihdr['ID']     = ID
    prihdr['PA']     = PA
    prihdr['z']      = zbes
    prihdr['RA']     = RA
    prihdr['DEC']    = DEC
    # Add rejuv properties;
    prihdr['f_rejuv']= f_rejuv
    prihdr['t_quen'] = t_quench
    prihdr['t_rejuv']= t_rejuv
    # SFR
    prihdr['tset_SFR']= tset_SFR_SED
    # Version;
    import gsf
    prihdr['version'] = gsf.__version__

    prihdu = fits.PrimaryHDU(header=prihdr)

    col01 = []
    # Redshift
    zmc = hdul[1].data['zmc']
    col50 = fits.Column(name='zmc', format='E', unit='', array=zmc[:])
    col01.append(col50)

    # Stellar mass
    ACP = [ACp[0,0], ACp[0,1], ACp[0,2]]
    col50 = fits.Column(name='Mstel', format='E', unit='logMsun', array=ACP[:])
    col01.append(col50)

    # SFR based on SED
    col50 = fits.Column(name='SFR', format='E', unit='Msun/yr', array=SFR_SED_med[:])
    col01.append(col50)

    # Metallicity_MW
    ZCP = [ZCp[0,0], ZCp[0,1], ZCp[0,2]]
    col50 = fits.Column(name='Z_MW', format='E', unit='logZsun', array=ZCP[:])
    col01.append(col50)

    # Age_mw
    con = (~np.isnan(TC[0,:]))
    para = [np.percentile(TC[0,:][con],16), np.percentile(TC[0,:][con],50), np.percentile(TC[0,:][con],84)]
    col50 = fits.Column(name='T_MW', format='E', unit='logGyr', array=para[:])
    col01.append(col50)

    # Metallicity_LW
    ZCP = [ZL[0,0], ZL[0,1], ZL[0,2]]
    col50 = fits.Column(name='Z_LW', format='E', unit='logZsun', array=ZCP[:])
    col01.append(col50)

    # Age_lw
    con = (~np.isnan(TL[0,:]))
    para = [np.percentile(TL[0,:][con],16), np.percentile(TL[0,:][con],50), np.percentile(TL[0,:][con],84)]
    col50 = fits.Column(name='T_LW', format='E', unit='logGyr', array=para[:])
    col01.append(col50)

    # Dust
    para = [Avtmp[0], Avtmp[1], Avtmp[2]]
    col50 = fits.Column(name='AV', format='E', unit='mag', array=para[:])
    col01.append(col50)

    # 
    colms  = fits.ColDefs(col01)
    dathdu = fits.BinTableHDU.from_columns(colms)
    hdu = fits.HDUList([prihdu, dathdu])
    hdu.writeto('SFH_' + ID + '_PA' + PA + '_param.fits', overwrite=True)

    # Attach to MB;
    MB.sfh_tlook = age
    MB.sfh_tlookl= delTl[:][conA]/1e9
    MB.sfh_tlooku= delTu[:][conA]/1e9
    MB.sfh_sfr16 = SFp[:,0]
    MB.sfh_sfr50 = SFp[:,1]
    MB.sfh_sfr84 = SFp[:,2]
    MB.sfh_mfr16 = ACp[:,0]
    MB.sfh_mfr50 = ACp[:,1]
    MB.sfh_mfr84 = ACp[:,2]
    MB.sfh_zfr16 = ZCp[:,0]
    MB.sfh_zfr50 = ZCp[:,1]
    MB.sfh_zfr84 = ZCp[:,2]

    # SFH
    zzall = np.arange(1.,12,0.01)
    Tall  = MB.cosmo.age(zzall).value # , use_flat=True, **cosmo)

    dely2 = 0.1
    while (y2max-y2min)/dely2>7:
        dely2 *= 2.

    y2ticks = np.arange(y2min, y2max, dely2)
    ax2.set_yticks(y2ticks)
    ax2.set_yticklabels(np.arange(y2min, y2max, 0.1), minor=False)

    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    y3min, y3max = np.min(Z), np.max(Z)

    # For redshift
    if zbes<4:
        if zbes<2:
            zred  = [zbes, 2, 3, 6]
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
    elif zbes<6:
        zred  = [zbes, 6, 7, 9]
        zredl = ['$z_\mathrm{obs.}$', 6, 7, 9]
    else:
        zred  = [zbes, 12]
        zredl = ['$z_\mathrm{obs.}$', 12]

    Tzz   = np.zeros(len(zred), dtype='float32')
    for zz in range(len(zred)):
        Tzz[zz] = (Tuni - MB.cosmo.age(zred[zz]).value)
        if Tzz[zz] < 0.01:
            Tzz[zz] = 0.01

    #ax3t.set_xscale('log')
    #ax3t.set_xlim(Txmin, Txmax)

    ax1.set_xlabel('$t_\mathrm{lookback}$/Gyr', fontsize=12)
    ax2.set_xlabel('$t_\mathrm{lookback}$/Gyr', fontsize=12)
    ax4.set_xlabel('$t_\mathrm{lookback}$/Gyr', fontsize=12)
    ax4.set_ylabel('$\log Z_*/Z_\odot$', fontsize=12)

    ax1t.set_xscale('log')
    ax1t.set_xlim(Txmin, Txmax)
    ax2t.set_xscale('log')
    ax2t.set_xlim(Txmin, Txmax)
    ax4t.set_xscale('log')
    ax4t.set_xlim(Txmin, Txmax)

    ax4.set_xlim(Txmin, Txmax)
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

    # Save
    #plt.show()
    #ax1.legend(loc=2, fontsize=8)
    #ax2.legend(loc=3, fontsize=8)
    plt.savefig('SFH_' + ID + '_PA' + PA + '_pcl.png')


def get_evolv(MB, ID, PA, Z=np.arange(-1.2,0.4249,0.05), age=[0.01, 0.1, 0.3, 0.7, 1.0, 3.0], f_comp=0, fil_path='./FILT/', \
    inputs=None, dust_model=0, DIR_TMP='./templates/', delt_sfh=0.01):

    '''
    Purpose:
    =========
    Reprocess output files to get spectra, UV color, and SFH at higher resolution.

    Input:
    ======
    #
    # delt_sfh (float): delta t of input SFH in Gyr.
    #
    # Returns: SED as function of age, based on SF and Z histories;
    #
    '''

    print('This function may take a while as it runs fsps.')
    flim  = 0.01
    lsfrl = -1 # log SFR low limit
    mmax  = 1000
    Txmax = 4 # Max x value
    lmmin = 10.3

    nage = np.arange(0,len(age),1)
    fnc  = Func(ID, PA, Z, nage, dust_model=dust_model) # Set up the number of Age/ZZ
    bfnc = Basic(Z)
    age = np.asarray(age)

    ################
    # RF colors.
    import os.path
    home = os.path.expanduser('~')
    c      = MB.c #3.e18 # A/s
    m0set  = MB.m0set #25.0
    chimax = 1.
    d      = 10**(73.6/2.5) * 1e-18 # From [ergs/s/cm2/A] to [ergs/s/cm2/Hz]

    ###########################
    # Open result file
    ###########################
    file = 'summary_' + ID + '_PA' + PA + '.fits'
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
        file = 'templates/spec_obs_' + ID + '_PA' + PA + '.cat'
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
    DL = MB.cosmo.luminosity_distance(zbes).value * MB.Mpc_cm # Luminositydistance in cm
    Cons = (4.*np.pi*DL**2/(1.+zbes))

    Tuni = MB.cosmo.age(zbes).value
    Tuni0 = (Tuni - age[:])

    delT  = np.zeros(len(age),dtype='float32')
    delTl = np.zeros(len(age),dtype='float32')
    delTu = np.zeros(len(age),dtype='float32')
    for aa in range(len(age)):
        if aa == 0:
            delTl[aa] = age[aa]
            delTu[aa] = (age[aa+1]-age[aa])/2.
            delT[aa]  = delTu[aa] + delTl[aa]
        elif Tuni < age[aa]:
            delTl[aa] = (age[aa]-age[aa-1])/2.
            delTu[aa] = delTl[aa] #10.
            delT[aa]  = delTu[aa] + delTl[aa]
        elif aa == len(age)-1:
            delTl[aa] = (age[aa]-age[aa-1])/2.
            delTu[aa] = Tuni - age[aa]
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
    pfile = 'chain_' + ID + '_PA' + PA + '_corner.cpkl'

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
    for mm in range(mmax):
        mtmp  = np.random.randint(len(samples))# + Nburn
        AAtmp = np.zeros(len(age), dtype='float32')
        ZZtmp = np.zeros(len(age), dtype='float32')
        mslist= np.zeros(len(age), dtype='float32')

        Av_tmp = samples['Av'][mtmp]

        f0     = fits.open(DIR_TMP + 'ms_' + ID + '_PA' + PA + '.fits')
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
    tbegin = np.min(Tuni-age)
    tuniv_hr = np.arange(tbegin,Tuni,delt_sfh) # in Gyr
    sfh_hr_in= np.interp(tuniv_hr,(Tuni-age)[::-1],SFp[:,1][::-1])
    zh_hr_in = np.interp(tuniv_hr,(Tuni-age)[::-1],ZCp[:,1][::-1])

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

    for ss in range(len(t_get)):
        wave0, flux0 = sp.get_spectrum(tage=t_get[ss], peraa=True)
        if ss == 0:
            spec_mul_nu_conv = np.zeros((len(t_get),len(wave0)),dtype='float32')

        print('Template %d is processed.'%(ss))
        wavetmp  = wave0*(1.+zbes)
        spec_mul_nu = flamtonu(wavetmp, flux0) # Conversion from Flambda to Fnu.
        Lsun = MB.Lsun #3.839 * 1e33 #erg s-1
        stmp_common = 1e10 # 1 tmp is in 1e10Lsun

        spec_mul_nu_conv[ss,:] = spec_mul_nu[:]
        spec_mul_nu_conv[ss,:] *= Lsun/(4.*np.pi*DL**2/(1.+zbes))
        Ls = 10**sp.log_lbol
        spec_mul_nu_conv[ss,:] *= (1./Ls)*stmp_common # in unit of erg/s/Hz/cm2/ms[ss].

        consave = (wavetmp/(1.+zbes)<20000) # AA
        if ss == 0:
            nd_ap  = np.arange(0,len(wave0),1)
            col1   = fits.Column(name='wavelength', format='E', unit='AA', array=wavetmp[consave])#, disp='obs'
            col2   = fits.Column(name='colnum', format='K', unit='', array=nd_ap[consave])
            col00  = [col1, col2]
            col3   = fits.Column(name='age', format='E', unit='Gyr', array=t_get)
            col4   = fits.Column(name='sfh', format='E', unit='Msun/yr', array=sfh_hr_in[con_sfh])
            col5   = fits.Column(name='zh', format='E', unit='Zsun', array=zh_hr_in[con_sfh])
            col01  = [col3,col4,col5]

        colspec_all = fits.Column(name='fspec_'+str(ss), format='E', unit='Fnu', array=spec_mul_nu_conv[ss,:][consave])#, disp='%s'%(t_get[ss])
        col00.append(colspec_all)

    coldefs_spec = fits.ColDefs(col00)
    hdu = fits.BinTableHDU.from_columns(coldefs_spec)
    hdu.writeto(DIR_TMP + 'obsspec_' + ID + '_PA' + PA + '.fits', overwrite=True)

    coldefs_spec = fits.ColDefs(col01)
    hdu = fits.BinTableHDU.from_columns(coldefs_spec)
    hdu.writeto(DIR_TMP + 'obshist_' + ID + '_PA' + PA + '.fits', overwrite=True)


def plot_evolv(MB, ID, PA, Z=np.arange(-1.2,0.4249,0.05), age=[0.01, 0.1, 0.3, 0.7, 1.0, 3.0], f_comp=0, fil_path='./FILT/', \
    inputs=None, dust_model=0, DIR_TMP='./templates/', delt_sfh = 0.01, nmc=300):
    
    '''
    Input:
    ======
    delt_sfh (float): delta t of input SFH in Gyr.

    Returns:
    ========
    SED as function of age, based on SF and Z histories;
    '''

    import os.path

    ################
    flim = 0.01
    lsfrl = -1 # log SFR low limit
    mmax  = 1000
    Txmax = 4 # Max x value
    lmmin = 10.3

    nage = np.arange(0,len(age),1)
    fnc  = Func(ID, PA, Z, nage, dust_model=dust_model) # Set up the number of Age/ZZ
    bfnc = Basic(Z)
    age = np.asarray(age)

    ################
    # RF colors.
    home = os.path.expanduser('~')
    c      = 3.e18 # A/s
    chimax = 1.
    m0set   = 25.0
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
    file = 'summary_' + ID + '_PA' + PA + '.fits'
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
        file = 'templates/spec_obs_' + ID + '_PA' + PA + '.cat'
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
    DL = MB.cosmo.luminosity_distance(zbes).value * MB.Mpc_cm # Luminositydistance in cm
    Cons = (4.*np.pi*DL**2/(1.+zbes))
    Tuni = MB.cosmo.age(zbes).value #, use_flat=True, **cosmo)
    Tuni0 = (Tuni - age[:])

    # Open summary;
    file = 'summary_' + ID + '_PA' + PA + '.fits'
    fd   = fits.open(file)[1].data
    #print(fits.open(file)[1].header)
    Avtmp = fd['Av0']
    uvtmp = fd['uv']
    vjtmp = fd['vj']
    #ax2.plot(vj[1],uv[1],color='gray',marker='s',ms=3)

    # SFH
    file = DIR_TMP + 'obshist_' + ID + '_PA' + PA + '.fits'
    fd   = fits.open(file)[1].data
    age  = fd['age']
    sfh  = fd['sfh']
    zh   = fd['zh']

    # Open FSPS temp;
    file = DIR_TMP + 'obsspec_' + ID + '_PA' + PA + '.fits'
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
    plt.savefig('hist_' + ID + '_PA' + PA + '.pdf')