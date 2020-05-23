#!/usr/bin/env python
#<examples/doc_nistgauss.py>
import numpy as np
import sys
import matplotlib.pyplot as plt
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


def get_logSFMS(logM, T, delm=0.):
    '''
    Steinhurt+14
    '''
    logSFR = (0.84 - 0.026 * T) * (logM - delm) - (6.51 - 0.11 * T)
    return logSFR

def plot_mz(MB, ID0, PA, Z=np.arange(-1.2,0.4249,0.05), age=[0.01, 0.1, 0.3, 0.7, 1.0, 3.0]):
    '''
    '''
    #col = ['darkred', 'r', 'coral','orange','g','lightgreen', 'lightblue', 'b','indigo','violet','k']
    import matplotlib.cm as cm

    flim = 0.01
    lsfrl = -1 # log SFR low limit
    mmax  = 500
    Txmax = 4 # Max x value
    lmmin = 10.3

    age = np.asarray(age)
    nage = np.arange(0,len(age),1)
    fnc  = MB.fnc #Func(Z, nage) # Set up the number of Age/ZZ
    bfnc = MB.bfnc #Basic(Z)



    ################
    # RF colors.
    import os.path
    home = os.path.expanduser('~')
    #fil_path = '/Users/tmorishita/eazy-v1.01/PROG/FILT/'
    fil_path = MB.DIR_FILT #home + '/Dropbox/FILT/'
    fil_u = fil_path+'u.fil'
    fil_b = fil_path+'b.fil'
    fil_v = fil_path+"v.fil"
    fil_j = fil_path+"j.fil"
    fil_k = fil_path+"k.fil"
    fil_f125w = fil_path+"f125w.fil"
    fil_f160w = fil_path+"f160w.fil"
    fil_36 = fil_path+"3.6.fil"
    fil_45 = fil_path+"4.5.fil"

    du = np.loadtxt(fil_u,comments="#")
    lu = du[:,1]
    fu = du[:,2]

    db = np.loadtxt(fil_b,comments="#")
    lb = db[:,1]
    fb = db[:,2]

    dv = np.loadtxt(fil_v,comments="#")
    lv = dv[:,1]
    fv = dv[:,2]

    dj = np.loadtxt(fil_j,comments="#")
    lj = dj[:,1]
    fj = dj[:,2]

    c      = MB.c #3.e18 # A/s
    mag0   = MB.m0set #25.0
    chimax = 1.
    d      = 10**(73.6/2.5) * 1e-18 # From [ergs/s/cm2/A] to [ergs/s/cm2/Hz]
    #d = 10**(-73.6/2.5) # From [ergs/s/cm2/Hz] to [ergs/s/cm2/A]

    #############
    # Plot.
    #############
    fig = plt.figure(figsize=(6,6))
    fig.subplots_adjust(top=0.98, bottom=0.08, left=0.1, right=0.99, hspace=0.18, wspace=0.25)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    #ax3t = ax3.twiny()
    #ax4t = ax4.twiny()


    ##################
    # Fitting Results
    ##################
    DIR_TMP = './templates/'
    SNlim = 3 # avobe which SN line is shown.

    ###########################
    # Open result file
    ###########################
    file = 'summary_' + ID0 + '_PA' + PA + '.fits'
    hdul = fits.open(file) # open a FITS file
    zbes = hdul[0].header['z']

    Asum = 0
    A50 = np.arange(len(age), dtype='float32')
    for aa in range(len(A50)):
        A50[aa] = hdul[1].data['A'+str(aa)][1]
        Asum += A50[aa]

    # Cosmo;
    DL = MB.cosmo.luminosity_distance(zbes).value * MB.Mpc_cm # Luminositydistance in cm
    Cons = (4.*np.pi*DL**2/(1.+zbes))

    Tuni  = MB.cosmo.age(zbes).value #, use_flat=True, **cosmo) # age at zobs.
    Tuni0 = (Tuni - age[:])

    delT  = np.zeros(len(age),dtype='float32')
    delTl = np.zeros(len(age),dtype='float32')
    delTu = np.zeros(len(age),dtype='float32')

    col = np.zeros((len(age),4),dtype='float32')
    for aa in range(len(age)):
        col[aa,:] = cm.nipy_spectral_r((aa+0.1)/(len(age)))

    for aa in range(len(age)):
        if aa == 0:
            delTl[aa] = age[aa]
            delTu[aa] = (age[aa+1]-age[aa])/2.
            delT[aa]  = delTu[aa] + delTl[aa]
        elif Tuni < age[aa]:
            delTl[aa] = (age[aa]-age[aa-1])/2.
            delTu[aa] = 10.
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
    # Wht do you want from MCMC sampler?
    AM = np.zeros((len(age), mmax), dtype='float32') # Mass in each bin.
    AC = np.zeros((len(age), mmax), dtype='float32') # Cumulative mass in each bin.
    ZM = np.zeros((len(age), mmax), dtype='float32') # Z.
    ZC = np.zeros((len(age), mmax), dtype='float32') # Cumulative Z.
    TC = np.zeros((len(age), mmax), dtype='float32') # Mass weighted T.
    ZMM = np.zeros((len(age), mmax), dtype='float32') # Mass weighted Z.
    SF = np.zeros((len(age), mmax), dtype='float32') # SFR

    mm = 0
    while mm<mmax:
        mtmp  = np.random.randint(len(samples))# + Nburn

        AAtmp = np.zeros(len(age), dtype='float32')
        ZZtmp = np.zeros(len(age), dtype='float32')
        mslist= np.zeros(len(age), dtype='float32')

        Av_tmp = samples['Av'][mtmp]

        f0     = fits.open(DIR_TMP + 'ms_' + ID0 + '_PA' + PA + '.fits')
        sedpar = f0[1]

        for aa in range(len(age)):
            AAtmp[aa] = samples['A'+str(aa)][mtmp]
            try:
                ZZtmp[aa] = samples['Z'+str(aa)][mtmp]
            except:
                ZZtmp[aa] = samples['Z0'][mtmp]

            nZtmp      = bfnc.Z2NZ(ZZtmp[aa])
            mslist[aa] = sedpar.data['ML_'+str(nZtmp)][aa]

            AM[aa, mm] = AAtmp[aa] * mslist[aa]
            SF[aa, mm] = AAtmp[aa] * mslist[aa] / delT[aa]
            ZM[aa, mm] = ZZtmp[aa] # AAtmp[aa] * mslist[aa]
            ZMM[aa, mm]= (10 ** ZZtmp[aa]) * AAtmp[aa] * mslist[aa]

        for aa in range(len(age)):
            AC[aa, mm] = np.sum(AM[aa:, mm])
            ZC[aa, mm] = np.log10(np.sum((ZMM)[aa:, mm])/AC[aa, mm])

            ACs = 0
            for bb in range(aa, len(age), 1):
                TC[aa, mm] += age[bb] * AAtmp[bb] * mslist[bb]
                ACs        += AAtmp[bb] * mslist[bb]

            TC[aa, mm] /= ACs

        mm += 1

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
            msize[aa] = 10 + 150 * A50[aa]/Asum

    conA = (msize>=0)

    #
    # 1.M-SFR
    #
    #ax1.fill_between(age[conA], np.log10(SFp[:,0])[conA], np.log10(SFp[:,2])[conA], linestyle='-', color='k', alpha=0.3)
    ax1.scatter(np.log10(ACp[:,1])[conA], np.log10(SFp[:,1])[conA], marker='o', c=col[:], s=msize[conA], edgecolors='k',zorder=2)
    ax1.errorbar(np.log10(ACp[:,1])[conA], np.log10(SFp[:,1])[conA], linestyle='--', color='k', lw=1., marker='', zorder=0, alpha=1.)

    lM      = np.arange(9,13,0.1)
    delSFR  = np.zeros(len(age), dtype='float32')
    delSFRl = np.zeros(len(age), dtype='float32')
    delSFRu = np.zeros(len(age), dtype='float32')

    # Plot Main sequence
    delm = 0
    if MB.nimf == 0:
        # 0.19 is for Kroupa to Salpeter.
        delm = 0.19
    for ii in range(len(Tuni0)):
        lSFR = get_logSFMS(lM, Tuni0[ii], delm=delm)
        ax1.fill_between(lM, lSFR-0.1, lSFR+0.1, linestyle='None', lw=0.5, zorder=-5, alpha=0.5, color=col[ii])
        lSFRtmp = (0.84 - 0.026 * Tuni0[ii]) * np.log10(ACp[ii,1]) - (6.51 - 0.11 * Tuni0[ii])
        delSFR[ii]  = np.log10(SFp[ii,1]) - lSFRtmp
        delSFRl[ii] = np.log10(SFp[ii,0]) - lSFRtmp
        delSFRu[ii] = np.log10(SFp[ii,2]) - lSFRtmp


    #
    # 2.t - delta SRF relation (right top)
    #
    #ax2.plot(age[:][conA], delSFR[conA], marker='', c='k',zorder=1, lw=1, linestyle='-')
    ax2.scatter(age[:][conA], delSFR[conA], marker='o', c=col[:], s=msize[conA], edgecolors='k',zorder=2)
    ax2.errorbar(age[:][conA], delSFR[:][conA], linestyle='--', fmt='--', color='k', lw=1., marker='', zorder=0, alpha=1.)


    #
    # 3.Mass - Z relation (left bottom)
    #
    ax2label = ''
    #ax2.fill_between(age[conA], np.log10(ACp[:,0])[conA], np.log10(ACp[:,2])[conA], linestyle='-', color='k', alpha=0.3)
    ax3.errorbar(np.log10(ACp[:,1])[conA], ZCp[:,1][conA], linestyle='--', zorder=0, color='k', lw=1., label=ax2label, alpha=1.) #, xerr=[np.log10(ACp[:,1])[conA]-np.log10(ACp[:,0])[conA],np.log10(ACp[:,2])[conA]-np.log10(ACp[:,1])[conA]], yerr=[ZCp[:,1][conA]-ZCp[:,0][conA],ZCp[:,2][conA]-ZCp[:,1][conA]]
    ax3.scatter(np.log10(ACp[:,1])[conA], ZCp[:,1][conA], marker='o', c=col[:], s=msize, edgecolors='k',zorder=2)

    #
    # Mass-Z from Gallazzi+05
    #
    lM   = [8.91, 9.11, 9.31, 9.51, 9.72, 9.91, 10.11, 10.31, 10.51, 10.72, 10.91, 11.11, 11.31, 11.51, 11.72, 11.91]
    lZ50 = [-0.6, -0.61, -0.65, -0.61, -.52, -.41, -.23, -.11, -.01, .04, .07, .10, .12, .13, .14, .15]
    lZ16 = [-1.11, -1.07, -1.1, -1.03, -.97, -.90, -.8, -.65, -.41, -.24, -.14, -.09, -.06, -.04, -.03, -.03]
    lZ84 = [-0., -0., -0.05, -0.01, .05, .09, .14, .17, .20, .22, .24, .25, .26, .28, .29, .30]

    lM   = np.asarray(lM)
    lZ50 = np.asarray(lZ50)
    lZ16 = np.asarray(lZ16)
    lZ84 = np.asarray(lZ84)

    ax3.errorbar(lM, lZ50, marker='', color='gray', ms=15, linestyle='-', lw=1, zorder=-2) #, yerr=[lZ50-lZ16, lZ84-lZ50]
    ax3.fill_between(lM, lZ16, lZ84, color='gray', linestyle='None', lw=1, alpha=0.4, zorder=-2)


    #
    # 4.Fundamental Metal
    #
    bsfr = -0.32 # From Mannucci+10
    #ax4.fill_between(age[conA], ZCp[:,0][conA], ZCp[:,2][conA], linestyle='-', color='k', alpha=0.3)
    ax4.scatter((np.log10(ACp[:,1]) + bsfr * np.log10(SFp[:,1]))[conA], ZCp[:,1][conA], marker='o', c=col[:], s=msize[conA], edgecolors='k',zorder=2)
    ax4.errorbar((np.log10(ACp[:,1]) + bsfr * np.log10(SFp[:,1]))[conA], ZCp[:,1][conA], linestyle='--', color='k', lw=1., zorder=0, alpha=1.)


    for iic in range(len(A50)):
        if msize[iic]>10:
            lwe = 1.5
            ax1.errorbar(np.log10(ACp[iic,1]), np.log10(SFp[iic,1]), xerr=[[np.log10(ACp[iic,1])-np.log10(ACp[iic,0])],[np.log10(ACp[iic,2])-np.log10(ACp[iic,1])]], yerr=[[np.log10(SFp[iic,1])-np.log10(SFp[iic,0])], [np.log10(SFp[iic,2])-np.log10(SFp[iic,1])]], linestyle='-', color=col[iic], lw=lwe, marker='', zorder=1)
            ax2.errorbar(age[iic], delSFR[iic], xerr=[[delTl[iic]/1e9],[delTu[iic]/1e9]], yerr=[[delSFR[iic]-delSFRl[iic]], [delSFRu[iic]-delSFR[iic]]], linestyle='-', fmt='-', color=col[iic], lw=lwe, marker='', zorder=1)
            ax3.errorbar(np.log10(ACp[iic,1]), ZCp[iic,1], xerr=[[np.log10(ACp[iic,1])-np.log10(ACp[iic,0])],[np.log10(ACp[iic,2])-np.log10(ACp[iic,1])]], yerr=[[ZCp[iic,1]-ZCp[iic,0]],[ZCp[iic,2]-ZCp[iic,1]]], linestyle='-', color=col[iic], lw=lwe, label=ax2label, zorder=1)

            xerl_ax4 = np.sqrt((np.log10(ACp[iic,1])-np.log10(ACp[iic,0]))**2 + bsfr**2 * (np.log10(SFp[iic,1])-np.log10(SFp[iic,0]))**2)
            xeru_ax4 = np.sqrt((np.log10(ACp[iic,2])-np.log10(ACp[iic,1]))**2 + bsfr**2 * (np.log10(SFp[iic,2])-np.log10(SFp[iic,1]))**2)

            ax4.errorbar((np.log10(ACp[iic,1]) + bsfr * np.log10(SFp[iic,1])), ZCp[iic,1], xerr=[[xerl_ax4],[xeru_ax4]], yerr=[[ZCp[iic,1]-ZCp[iic,0]],[ZCp[iic,2]-ZCp[iic,1]]], linestyle='-', color=col[iic], lw=lwe, label=ax2label, zorder=1)


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

    #y3min, y3max = np.min(Z), np.max(Z)
    y3min, y3max = -0.8, 0.4 #np.min(Z), np.max(Z)

    lsfru = 2.8
    if np.max(np.log10(SFp[:,2]))>2.8:
        lsfru = np.max(np.log10(SFp[:,2]))+0.1

    y2min, y2max = 9.5, 12.5
    ax1.set_xlim(y2min, y2max)
    ax1.set_ylim(lsfrl, lsfru)
    #ax1.set_xscale('log')
    ax1.set_xlabel('$\log M_*/M_\odot$', fontsize=12)

    #ax1.xaxis.labelpad = -3
    #ax1.yaxis.labelpad = -2
    #ax2t.set_yticklabels(())

    #ax2.set_xlabel('$t$ (Gyr)', fontsize=12)
    #ax2.set_ylabel('$\log M_*/M_\odot$', fontsize=12)
    ax2.set_xlabel('$t-t_\mathrm{obs.}$/Gyr', fontsize=12)
    ax2.set_ylabel('$\Delta_\mathrm{SFR}$', fontsize=12)

    #ax2.set_ylim(lsfrl, lsfru)
    ax2.set_ylim(-4, 2.5)
    ax2.set_xlim(0.008, 3.2)
    ax2.set_xscale('log')

    dely2 = 0.1
    while (y2max-y2min)/dely2>7:
        dely2 *= 2.

    #y2ticks = np.arange(y2min, y2max, dely2)
    #ax2.set_yticks(y2ticks)
    #ax2.set_yticklabels(np.arange(y2min, y2max, 0.1), minor=False)
    #ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    ax3.set_xlim(y2min, y2max)
    ax3.set_ylim(y3min, y3max)

    ax3.set_xlabel('$\log M_*/M_\odot$', fontsize=12)
    ax3.set_ylabel('$\log Z_*/Z_\odot$', fontsize=12)

    #ax3.set_xscale('log')
    #ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


    # For redshift
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
    elif zbes<6:
        zred  = [zbes, 6]
        zredl = ['$z_\mathrm{obs.}$', 6]

    Tzz   = np.zeros(len(zred), dtype='float32')
    for zz in range(len(zred)):
        Tzz[zz] = (Tuni - MB.cosmo.age(zred[zz]).value)
        if Tzz[zz] < 0.01:
            Tzz[zz] = 0.01


    #ax3t.set_xscale('log')
    #ax3t.set_xlim(0.008, Txmax)

    ax4.set_xlabel('$\log M_*/M_\odot - 0.32 \log \dot{M_*}/M_\odot$yr$^{-1}$', fontsize=12)
    ax4.set_ylabel('$\log Z_*/Z_\odot$', fontsize=12)

    #ax4t.set_xscale('log')
    #ax4t.set_xlim(0.008, Txmax)

    ax4.set_xlim(9, 12.3)
    ax4.set_ylim(y3min, y3max)
    #ax4.set_xscale('log')
    #ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax3.yaxis.labelpad = -2
    ax4.yaxis.labelpad = -2


    ####################
    ## Save
    ####################
    ax1.legend(loc=1, fontsize=11)
    ax2.legend(loc=3, fontsize=8)
    plt.savefig('MZ_' + ID0 + '_PA' + PA + '_pcl.pdf')
