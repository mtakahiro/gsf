#!/usr/bin/env python
#<examples/doc_nistgauss.py>
import numpy as np
#from lmfit.models import GaussianModel, ExponentialModel, ExpressionModel
import sys
import os
import matplotlib.pyplot as plt
#import lmfit
from numpy import log10
from scipy.integrate import simps
from astropy.io import fits
from matplotlib.ticker import FormatStrFormatter

from .function import *
from .function_class import Func
from .basic_func import Basic
from . import img_scale
from . import corner

lcb = '#4682b4' # line color, blue
col = ['darkred', 'r', 'coral','orange','g','lightgreen', 'lightblue', 'b','indigo','violet','k']


def plot_sim_comp(ID, PA, Z=np.arange(-1.2,0.4249,0.05), age=[0.01, 0.1, 0.3, 0.7, 1.0, 3.0],  f_Z_all=0, tau0=[0.1,0.2,0.3]):

    nage = np.arange(0,len(age),1)
    fnc  = Func(ID, PA, Z, nage, dust_model=dust_model) # Set up the number of Age/ZZ
    bfnc = Basic(Z)

    c      = 3.e18 # A/s
    chimax = 1.
    m0set   = 25.0
    d      = 10**(73.6/2.5) * 1e-18 # From [ergs/s/cm2/A] to [ergs/s/cm2/Hz]
    #d = 10**(-73.6/2.5) # From [ergs/s/cm2/Hz] to [ergs/s/cm2/A]

    #############
    # Plot.
    #############
    fig = plt.figure(figsize=(5.5,5))
    fig.subplots_adjust(top=0.96, bottom=0.16, left=0.1, right=0.99, hspace=0.15, wspace=0.25)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ##################
    # Fitting Results
    ##################
    DIR_TMP = './templates/'
    SNlim = 3 # avobe which SN line is shown.

    ###########################
    # Open input file
    ###########################
    file = DIR_TMP + 'sim_param_' + ID + '_PA' + PA + '.fits'
    hdu0 = fits.open(file) # open a FITS file

    Avin = float(hdu0[0].header['AV'])
    Ain  = np.zeros(len(age), dtype='float64')
    Zin  = np.zeros(len(age), dtype='float64')
    for aa in range(len(age)):
        Ain[aa] = hdu0[0].header['A'+str(aa)]
        Zin[aa] = hdu0[0].header['Z'+str(aa)]

    Ainsum = np.sum(Ain)

    ###########################
    # Open result file
    ###########################
    file = 'summary_' + ID + '_PA' + PA + '.fits'
    hdul = fits.open(file) # open a FITS file

    # Redshift MC
    zp50  = hdul[1].data['zmc'][1]
    zp16  = hdul[1].data['zmc'][0]
    zp84  = hdul[1].data['zmc'][2]

    M50 = hdul[1].data['ms'][1]
    M16 = hdul[1].data['ms'][0]
    M84 = hdul[1].data['ms'][2]
    print('Total stellar mass is %.2e'%(M50))

    A50 = np.zeros(len(age), dtype='float64')
    A16 = np.zeros(len(age), dtype='float64')
    A84 = np.zeros(len(age), dtype='float64')
    for aa in range(len(age)):
        A50[aa] = hdul[1].data['A'+str(aa)][1]
        A16[aa] = hdul[1].data['A'+str(aa)][0]
        A84[aa] = hdul[1].data['A'+str(aa)][2]

    Asum  = np.sum(A50)

    aa   = 0
    Av50 = hdul[1].data['Av'+str(aa)][1]
    Av16 = hdul[1].data['Av'+str(aa)][0]
    Av84 = hdul[1].data['Av'+str(aa)][2]

    Z50 = np.zeros(len(age), dtype='float64')
    Z16 = np.zeros(len(age), dtype='float64')
    Z84 = np.zeros(len(age), dtype='float64')
    NZbest = np.zeros(len(age), dtype='int')
    for aa in range(len(age)):
        Z50[aa] = hdul[1].data['Z'+str(aa)][1]
        Z16[aa] = hdul[1].data['Z'+str(aa)][0]
        Z84[aa] = hdul[1].data['Z'+str(aa)][2]
        NZbest[aa]= bfnc.Z2NZ(Z50[aa])


    ZZ50 = np.sum(Z50*A50)/np.sum(A50)  # Light weighted Z.

    chi   = hdul[1].data['chi'][0]
    chin  = hdul[1].data['chi'][1]
    fitc  = chin

    Cz0   = hdul[0].header['Cz0']
    Cz1   = hdul[0].header['Cz1']
    zbes  = hdul[0].header['z']
    zscl = (1.+zbes)

    ####################
    ## Plot
    ####################
    flim  = 0.1
    msize = np.zeros(len(age), dtype='float64')
    for aa in range(len(age)):
        if Ain[aa]/Ainsum>flim: # if >1%
            msize[aa] = 150 * A50[aa]/Asum

    conA = (msize>=0)

    for aa in range(len(age)):
        if msize[aa]>0:
            ax1.errorbar(Ain[aa], A50[aa]/Asum, yerr=[[A50[aa]-A16[aa]]/Asum, [A84[aa]-A50[aa]]/Asum], marker='', ms=3, c='k', zorder=-1)
            ax1.scatter(Ain[aa], A50[aa]/Asum, marker='o', s=msize[aa], c=col[aa], zorder=1)

            ax2.errorbar(Zin[aa], Z50[aa], yerr=[[Z50[aa]-Z16[aa]], [Z84[aa]-Z50[aa]]], marker='', ms=3, c='k', zorder=-1)
            ax2.scatter(Zin[aa], Z50[aa], marker='o', s=msize[aa], c=col[aa], zorder=1, label='$n=%d$'%(aa))


    Avmax = 1.5
    ax1.errorbar(Avin/Avmax, Av50/Avmax, yerr=[[(Av50-Av16)/Avmax], [(Av84-Av50)/Avmax]], marker='', ms=3, c='k', zorder=-1)
    ax1.scatter(Avin/Avmax, Av50/Avmax, marker='*', s=25, c='gray', zorder=1, label='$A_V$')


    ax1.set_xlabel('$A_\mathrm{in}$', fontsize=12)
    ax1.set_ylabel('$A_\mathrm{out}$', fontsize=12)

    y1min, y1max = 0, 1. #np.max(A50/Asum)+0.1
    ax1.set_xlim(y1min, y1max)
    ax1.set_ylim(y1min, y1max)


    ax2.set_xlabel('$\log Z_*/Z_\odot$', fontsize=12)
    ax2.set_ylabel('$\log Z_*/Z_\odot$', fontsize=12)

    y2min, y2max = np.min(Z), np.max(Z)
    ax2.set_xlim(y2min, y2max)
    ax2.set_ylim(y2min, y2max)

    #
    # y = x
    #
    xx = np.arange(0,1.1,0.1)
    yy = xx
    ax1.plot(xx, yy, color='k', linestyle='--', lw=0.5, zorder=-2)

    xx = np.arange(y2min,y2max+0.1,0.1)
    yy = xx
    ax2.plot(xx, yy, color='k', linestyle='--', lw=0.5, zorder=-2)


    ####################
    ## Save
    ####################
    #plt.show()
    ax1.legend(loc=4, fontsize=11)
    ax2.legend(loc=4, fontsize=11)

    plt.savefig('SIM' + ID + '_PA' + PA + '_comp.pdf', dpi=300)

def plot_sed_Z_sim(ID, PA, Z=np.arange(-1.2,0.4249,0.05), age=[0.01, 0.1, 0.3, 0.7, 1.0, 3.0], tau0=[0.1,0.2,0.3], flim=0.01, fil_path='./', figpdf=False):

    col = ['darkred', 'r', 'coral','orange','g','lightgreen', 'lightblue', 'b','indigo','violet','k']

    nage = np.arange(0,len(age),1)
    fnc  = Func(ID, PA, Z, nage, dust_model=dust_model) # Set up the number of Age/ZZ
    bfnc = Basic(Z)

    ################
    # RF colors.
    import os.path
    home = os.path.expanduser('~')
    #fil_path = '/Users/tmorishita/eazy-v1.01/PROG/FILT/'
    #fil_path = home + '/Dropbox/FILT/'

    c      = 3.e18 # A/s
    chimax = 1.
    m0set   = 25.0
    d      = 10**(73.6/2.5) * 1e-18 # From [ergs/s/cm2/A] to [ergs/s/cm2/Hz]
    #d = 10**(-73.6/2.5) # From [ergs/s/cm2/Hz] to [ergs/s/cm2/A]

    #############
    # Plot.
    #############
    fig = plt.figure(figsize=(5.,2.4))
    fig.subplots_adjust(top=0.96, bottom=0.16, left=0.1, right=0.99, hspace=0.15, wspace=0.25)
    ax1 = fig.add_subplot(111)

    ##################
    # Fitting Results
    ##################
    DIR_TMP = './templates/'
    SNlim = 1 # avobe which SN line is shown.

    ###########################
    # Open result file
    ###########################
    file = 'summary_' + ID + '_PA' + PA + '.fits'
    hdul = fits.open(file) # open a FITS file

    # Redshift MC
    zp50  = hdul[1].data['zmc'][1]
    zp16  = hdul[1].data['zmc'][0]
    zp84  = hdul[1].data['zmc'][2]


    M50 = hdul[1].data['ms'][1]
    M16 = hdul[1].data['ms'][0]
    M84 = hdul[1].data['ms'][2]
    print('Total stellar mass is %.2e'%(M50))

    A50 = np.zeros(len(age), dtype='float64')
    A16 = np.zeros(len(age), dtype='float64')
    A84 = np.zeros(len(age), dtype='float64')
    for aa in range(len(age)):
        A50[aa] = hdul[1].data['A'+str(aa)][1]
        A16[aa] = hdul[1].data['A'+str(aa)][0]
        A84[aa] = hdul[1].data['A'+str(aa)][2]

    Asum  = np.sum(A50)

    aa = 0
    Av50 = hdul[1].data['Av'+str(aa)][1]
    Av16 = hdul[1].data['Av'+str(aa)][0]
    Av84 = hdul[1].data['Av'+str(aa)][2]
    AAv = [Av50]

    Z50 = np.zeros(len(age), dtype='float64')
    Z16 = np.zeros(len(age), dtype='float64')
    Z84 = np.zeros(len(age), dtype='float64')
    NZbest = np.zeros(len(age), dtype='int')
    for aa in range(len(age)):
        Z50[aa] = hdul[1].data['Z'+str(aa)][1]
        Z16[aa] = hdul[1].data['Z'+str(aa)][0]
        Z84[aa] = hdul[1].data['Z'+str(aa)][2]
        NZbest[aa]= bfnc.Z2NZ(Z50[aa])

    ZZ50 = np.sum(Z50*A50)/np.sum(A50)  # Light weighted Z.

    chi   = hdul[1].data['chi'][0]
    chin  = hdul[1].data['chi'][1]
    fitc  = chin

    Cz0   = hdul[0].header['Cz0']
    Cz1   = hdul[0].header['Cz1']
    zbes  = hdul[0].header['z']
    zscl = (1.+zbes)

    ###############################
    # Data taken from
    ###############################
    dat = np.loadtxt(DIR_TMP + 'spec_obs_' + ID + '_PA' + PA + '.cat', comments='#')
    NR = dat[:, 0]
    x  = dat[:, 1]
    fy00 = dat[:, 2]
    ey00 = dat[:, 3]

    con0 = (NR<1000) #& (fy/ey>SNlim)
    xg0  = x[con0]
    fg0  = fy00[con0] * Cz0
    eg0  = ey00[con0] * Cz0
    con1 = (NR>=1000) & (NR<10000) #& (fy/ey>SNlim)
    xg1  = x[con1]
    fg1  = fy00[con1] * Cz1
    eg1  = ey00[con1] * Cz1
    con2 = (NR>=10000)#& (fy/ey>SNlim)
    xg2  = x[con2]
    fg2  = fy00[con2]
    eg2  = ey00[con2]

    fy01 = np.append(fg0,fg1)
    fy   = np.append(fy01,fg2)
    ey01 = np.append(eg0,eg1)
    ey   = np.append(ey01,eg2)

    wht=1./np.square(ey)

    dat = np.loadtxt(DIR_TMP + 'bb_obs_' + ID + '_PA' + PA + '.cat', comments='#')
    NRbb = dat[:, 0]
    xbb  = dat[:, 1]
    fybb = dat[:, 2]
    eybb = dat[:, 3]
    exbb = dat[:, 4]
    snbb = fybb/eybb


    # Original spectrum:
    filepar = DIR_TMP + 'sim_param_' + ID + '_PA' + PA + '.fits'
    hdupar  = fits.open(filepar) # open a FITS file
    Cnorm   = hdupar[0].header['cnorm']

    '''
    file_org = DIR_TMP + 'specorg_' + ID + '_PA' + PA + '.fits'
    hduorg = fits.open(file_org)
    waveorg = hduorg[1].data['wavelength'] # observed.
    fluxorg = hduorg[1].data['fspec_av'] * Cnorm
    #fluxorg_lam = fnutolam(waveorg, fluxorg)
    print(fluxorg)
    ax1.plot(waveorg, fluxorg * c / np.square(waveorg) / d, marker='', linestyle='-', linewidth=0.5, ms=0.1, color='k', label='')

    file_org = DIR_TMP + 'specorg_pix_' + ID + '_PA' + PA + '.fits'
    hduorg = fits.open(file_org)
    waveorg = hduorg[1].data['wavelength'] # observed.
    fluxorg = hduorg[1].data['fspec_av'] * Cnorm
    #fluxorg_lam = fnutolam(waveorg, fluxorg)
    ax1.plot(waveorg, fluxorg * c / np.square(waveorg) / d, marker='', linestyle='--', linewidth=0.5, ms=0.1, color='r', label='')
    print(fluxorg)
    plt.show()
    '''
    ######################
    # Weight by line
    ######################
    wh0  = 1./np.square(eg0)
    LW0 = []
    model = fg0
    wht3 = check_line_man(fy, x, wht, fy, zbes, LW0)


    ######################
    # Mass-to-Light ratio.
    ######################
    ms     = np.zeros(len(age), dtype='float64')
    f0     = fits.open(DIR_TMP + 'ms_' + ID + '_PA' + PA + '.fits')
    sedpar = f0[1]
    for aa in range(len(age)):
        ms[aa] = sedpar.data['ML_' +  str(int(NZbest[aa]))][aa]

    conspec = (NR<10000) #& (fy/ey>1)
    ax1.plot(xg0, fg0 * c / np.square(xg0) / d, marker='', linestyle='-', linewidth=0.5, ms=0.1, color='royalblue', label='')
    ax1.plot(xg1, fg1 * c / np.square(xg1) / d, marker='', linestyle='-', linewidth=0.5, ms=0.1, color='#DF4E00', label='')
    #conbb = (NR>=10000)

    #######################
    # Box for BB photometry
    for ii in range(len(xbb)):
        if eybb[ii]<100 and fybb[ii]/eybb[ii]>1:
            xx = [xbb[ii]-exbb[ii],xbb[ii]-exbb[ii]]
            yy = [(fybb[ii]-eybb[ii])*c/np.square(xbb[ii])/d, (fybb[ii]+eybb[ii])*c/np.square(xbb[ii])/d]
            #ax1.plot(xx, yy, color='k', linestyle='-', linewidth=0.5, zorder=3)
            xx = [xbb[ii]+exbb[ii],xbb[ii]+exbb[ii]]
            yy = [(fybb[ii]-eybb[ii])*c/np.square(xbb[ii])/d, (fybb[ii]+eybb[ii])*c/np.square(xbb[ii])/d]
            #ax1.plot(xx, yy, color='k', linestyle='-', linewidth=0.5, zorder=3)
            xx = [xbb[ii]-exbb[ii],xbb[ii]+exbb[ii]]
            yy = [(fybb[ii]-eybb[ii])*c/np.square(xbb[ii])/d, (fybb[ii]-eybb[ii])*c/np.square(xbb[ii])/d]
            #ax1.plot(xx, yy, color='k', linestyle='-', linewidth=0.5, zorder=3)
            xx = [xbb[ii]-exbb[ii],xbb[ii]+exbb[ii]]
            yy = [(fybb[ii]+eybb[ii])*c/np.square(xbb[ii])/d, (fybb[ii]+eybb[ii])*c/np.square(xbb[ii])/d]
            #ax1.plot(xx, yy, color='k', linestyle='-', linewidth=0.5, zorder=3)

    conbb = (fybb/eybb>3)
    ax1.errorbar(xbb[conbb], fybb[conbb] * c / np.square(xbb[conbb]) / d, yerr=eybb[conbb]*c/np.square(xbb[conbb])/d, color='k', linestyle='', linewidth=0.5, zorder=4)
    ax1.plot(xbb[conbb], fybb[conbb] * c / np.square(xbb[conbb]) / d, '.r', linestyle='', linewidth=0, zorder=4)#, label='Obs.(BB)')


    ################
    # Open ascii file and stock to array.
    lib     = fnc.open_spec_fits(ID, PA, fall=0, tau0=tau0)
    lib_all = fnc.open_spec_fits(ID, PA, fall=1, tau0=tau0)
    #lib_wid = fnc.open_spec_fits_wid(ID, PA, fall=1, tau0=tau0)

    II0   = nage #[0,1,2,3] # Number for templates
    iimax = len(II0)-1


    fwuvj = open(ID + '_PA' + PA + '_uvj.txt', 'w')
    fwuvj.write('# age uv vj\n')

    Asum = np.sum(A50[:])
    for jj in range(len(II0)):
        ii = int(len(II0) - jj - 1) # from old to young templates.
        if jj == 0:
            y0, x0   = fnc.tmp03(ID, PA, A50[ii], AAv[0], ii, Z50[ii], zbes, lib_all, tau0=tau0)
            y0p, x0p = fnc.tmp03(ID, PA, A50[ii], AAv[0], ii, Z50[ii], zbes, lib, tau0=tau0)
            ysump = y0p
            ysum  = y0

            if A50[ii]/Asum > flim:
                ax1.plot(x0, y0 * c/ np.square(x0) / d, '--', lw=0.5, color=col[ii], zorder=-1, label='')

        else:
            y0_r, x0_tmp = fnc.tmp03(ID, PA, A50[ii], AAv[0], ii, Z50[ii], zbes, lib_all, tau0=tau0)
            y0p, x0p     = fnc.tmp03(ID, PA, A50[ii], AAv[0], ii, Z50[ii], zbes, lib, tau0=tau0)

            ysump += y0p
            ysum  += y0_r

            if A50[ii]/Asum > flim:
                ax1.plot(x0, y0_r * c/ np.square(x0) / d, '--', lw=0.5, color=col[ii], zorder=-1, label='')


        ysum_wid = ysum * 0
        #print(ii, A50[ii],age[ii])
        #plt.close()
        for kk in range(0,ii+1,1):
            tt = int(len(II0) - kk - 1)
            nn = int(len(II0) - ii - 1)

            nZ = bfnc.Z2NZ(Z50[tt])
            y0_wid, x0_wid = fnc.open_spec_fits_dir(ID, PA, tt, nZ, nn, AAv[0], zbes, A50[tt], tau0=tau0)
            ysum_wid += y0_wid


        lmrest_wid = x0_wid/(1.+zbes)

        band0 = ['u','v','j']
        lmconv,fconv = filconv(band0, lmrest_wid, ysum_wid, fil_path) # f0 in fnu
        fu_t = fconv[0]
        fv_t = fconv[1]
        fj_t = fconv[2]
        uvt  = -2.5*log10(fu_t/fv_t)
        vjt  = -2.5*log10(fv_t/fj_t)
        fwuvj.write('%.2f %.3f %.3f\n'%(age[ii], uvt, vjt))

    fwuvj.close()


    lmrest = x0/(1.+zbes)
    model2 = ysum
    band0  = ['u','v','j','f140w']
    lmconv,fconv = filconv(band0, lmrest, model2, fil_path) # f0 in fnu
    fu_cnv = fconv[0]
    fv_cnv = fconv[1]
    fj_cnv = fconv[2]
    f140_cnv = fconv[3]

    flw = open(ID + '_PA' + PA + '_mag.txt', 'w')
    flw.write('# ID PA flux_u flux_v flux_j flux_f140w\n')
    flw.write('%s %s %.5f %.5f %.5f %.5f\n'%(ID, PA, fu_cnv, fv_cnv, fj_cnv, f140_cnv))
    flw.close()

    uvtmp = -2.5*log10(fu_cnv/fv_cnv)
    vjtmp = -2.5*log10(fv_cnv/fj_cnv)

    conw = (wht3>0)
    #npar = NDIM
    chi2 = sum((np.square(fy-ysump)*wht3)[conw])
    print('chi2/nu is %.2f'%(chin))
    #nu   = len(fy[conw])-NDIM
    #print('chi2/nu is %.2f'%(chi2/nu))

    #############
    # Main result
    #############
    #ax1.plot(x0, ysum * c/ np.square(x0) / d, '-', lw=1, color='gray', zorder=-1)#, label='$\chi^2/\\nu=%.2f$'%(chin)) # chin
    ymax = np.max([fybb[conbb]*c/np.square(xbb[conbb])/d] or ysum*c/np.square(x0)/d) * 2.


    #######################

    #ax1.text(12000, ymax*0.9, '%s'%(ID), fontsize=10, color='k')
    xboxl = 17000
    xboxu = 28000

    ax1.set_xlabel('Observed wavelength ($\mathrm{\mu m}$)', fontsize=14)
    ax1.set_ylabel('Flux ($10^{-18}\mathrm{erg}/\mathrm{s}/\mathrm{cm}^{2}/\mathrm{\AA}$)', fontsize=13)

    ax1.set_xlim(2200, 88000)
    #ax1.set_xlim(12500, 16000)
    ax1.set_xscale('log')
    ax1.set_ylim(-0.05, ymax)

    #import matplotlib.ticker as ticker
    import matplotlib
    #ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
    ax1.set_xticks([2500, 5000, 10000, 20000, 40000, 80000])
    ax1.set_xticklabels(['0.25', '0.5', '1', '2', '4', '8'])
    #ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    dely1 = 0.5
    while (ymax-0)/dely1>4:
        dely1 *= 2.

    y1ticks = np.arange(0, ymax, dely1)
    ax1.set_yticks(y1ticks)
    ax1.set_yticklabels(np.arange(0, ymax, dely1), minor=False)

    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


    #############
    # Plot
    #############
    eAAl = np.zeros(len(age),dtype='float64')
    eAAu = np.zeros(len(age),dtype='float64')
    eAMl = np.zeros(len(age),dtype='float64')
    eAMu = np.zeros(len(age),dtype='float64')

    MSsum = np.sum(ms)

    Asum = np.sum(A50)
    A50 /= Asum
    A16 /= Asum
    A84 /= Asum

    AM50 = A50 * M50 * ms / MSsum
    CM = M50/np.sum(AM50)

    AM50 = A50 * M50 * ms / MSsum * CM
    AM16 = A16 * M50 * ms / MSsum * CM
    AM84 = A84 * M50 * ms / MSsum * CM

    AC50 = A50 * 0 # Cumulative
    for ii in range(len(A50)):
        eAAl[ii] = A50[ii] - A16[ii]
        eAAu[ii] = A84[ii] - A50[ii]
        eAMl[ii] = AM50[ii] - AM16[ii]
        eAMu[ii] = AM84[ii] - AM50[ii]
        AC50[ii] = np.sum(AM50[ii:])


    ################
    # Lines
    ################
    LN = ['Mg2', 'Ne5', 'O2', 'Htheta', 'Heta', 'Ne3', 'Hdelta', 'Hgamma', 'Hbeta', 'O3', 'O3', 'Mgb', 'Halpha', 'S2L', 'S2H']
    #LW = [2800, 3347, 3727, 3799, 3836, 3869, 4102, 4341, 4861, 4959, 5007, 6563, 6717, 6731]
    FLW = np.zeros(len(LN),dtype='int')

    from scipy.optimize import curve_fit
    from scipy import asarray as ar,exp

    def gaus(x,a,x0,sigma):
        return a*exp(-(x-x0)**2/(2*sigma**2))

    dat = np.loadtxt(DIR_TMP + 'spec_obs_' + ID + '_PA' + PA + '.cat', comments='#')
    NR = dat[:, 0]
    x  = dat[:, 1]
    fy = dat[:, 2]
    ey = dat[:, 3]
    wht=1./np.square(ey)
    ysum_cut = np.interp(x,x0,ysum)

    ########################
    # Magnification
    ########################
    umag = 1

    ################
    # Set the inset.
    ################
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    from mpl_toolkits.axes_grid.inset_locator import inset_axes
    ax2t = inset_axes(ax1, width="40%", height="30%", loc=2)

    ####################
    # For cosmology
    ####################
    DL = MB.cosmo.luminosity_distance(zbes).value * Mpc_cm # Luminositydistance in cm
    Cons = (4.*np.pi*DL**2/(1.+zbes))
    dA     = MB.cosmo.angular_diameter_distance(zbes).value
    dkpc   = dA * (2*3.14/360/3600)*10**3 # kpc/arcsec
    twokpc = 5.0/dkpc/0.06 # in pixel

    ####################
    # LINES
    ####################
    R_grs = 45
    dw    = 4
    dlw   = R_grs * dw # Can affect the SFR.
    ldw   = 7

    # To add lines in the plot,
    # ,manually edit the following file
    # so as Fcont50 have >0.
    flw = open(ID + '_PA' + PA + '_lines_fit.txt', 'w')
    flw.write('# LW flux_line eflux_line flux_cont EW eEW L_line eL_line\n')
    flw.write('# (AA) (Flam_1e-18) (Flam_1e-18) (Flam_1e-18) (AA) (AA) (erg/s) (erg/s)\n')
    flw.write('# Error in EW is 1sigma, by pm eflux_line.\n')
    flw.write('# If EW=-99, it means gaussian fit failed.\n')
    flw.write('# and flux is the sum of excess at WL pm %.1f AA.\n'%(dlw))
    flw.write('# Magnification is corrected; mu=%.3f\n'%(umag))
    try:
        fl = np.loadtxt('table_' + ID + '_PA' + PA + '_lines.txt', comments='#')
        LW      = fl[:,2]
        Fcont50 = fl[:,3]
        Fline50 = fl[:,6]

        for ii in range(len(LW)):
            if Fcont50[ii] > 0:
                WL = LW[ii] * (1.+zbes)
                if ii == 7:
                    contmp = (x > WL - dlw) & (x < WL + dlw*1.5)
                else:
                    contmp = (x > WL - dlw) & (x < WL + dlw)
                FLW[ii] = 1

                xx   = x[contmp]
                yy   = (fy - ysum_cut)[contmp]
                eyy  = ey[contmp]
                yy2  = (ysum_cut)[contmp]

                xyzip = zip(xx,yy,eyy,yy2)
                xyzip = sorted(xyzip)

                xxs  = np.array([p1 for p1,p2,p3,p4 in xyzip])
                yys  = np.array([p2 for p1,p2,p3,p4 in xyzip])
                eyys = np.array([p3 for p1,p2,p3,p4 in xyzip])
                yy2s = np.array([p4 for p1,p2,p3,p4 in xyzip])

                flux = np.zeros(len(xxs), dtype='float64')
                efl  = np.zeros(len(xxs), dtype='float64')
                for ff in range(len(xxs)):
                    flux[ff] = yy2s[ff]/np.square(xxs[ff]) * c/d
                    efl[ff]  = np.square(eyys[ff]/np.square(xxs[ff]) * c/d)

                fmed = np.median(flux) # Median of continuum, model flux
                esum = np.sqrt(simps(efl, xxs))

                try:
                    popt,pcov = curve_fit(gaus,xxs,yys,p0=[Fline50[ii],WL,10],sigma=eyys)
                    xxss = xxs/zscl

                    if ii == 7:
                        popt,pcov = curve_fit(gaus,xxs,yys,p0=[Fline50[ii],WL+20,10],sigma=eyys)
                        xxss = xxs/zscl

                    ax2t.plot(xxs/zscl, (gaus(xxs,*popt)+yy2s) * c/np.square(xxs)/d, '#4682b4', linestyle='-', linewidth=1, alpha=0.8, zorder=20)

                    I1 = simps((gaus(xxs,*popt)) * c/np.square(xxs)/d, xxs)
                    I2 = I1 - simps((gaus(xxs,*popt)) * c/np.square(xxs)/d, xxs)
                    fline = I1

                    Flum = fline*Cons*1e-18 # luminosity in erg/s.
                    elum = esum *Cons*1e-18 # luminosity in erg/s.
                    SFR  = Flum * 6.58*1e-42
                    print('SFR is', SFR/umag)
                    EW_tmp   = simps( ((gaus(xxs,*popt)) * c/np.square(xxs)/d)/yy2s, xxs)
                    EW_tmp_u = simps( ((gaus(xxs,*popt) + eyys/np.sqrt(len(xxs))) * c/np.square(xxs)/d)/yy2s, xxs)

                    if ii == 7:
                        contmp2 = (xxs/zscl>4320.) & (xxs/zscl<4380.)
                        popt,pcov = curve_fit(gaus,xxs[contmp2], yys[contmp2], p0=[Fline50[ii],WL,10], sigma=eyys[contmp2])

                        I1 = simps((gaus(xxs[contmp2],*popt)) * c/np.square(xxs[contmp2])/d, xxs[contmp2])
                        I2 = I1 - simps((gaus(xxs[contmp2],*popt)) * c/np.square(xxs[contmp2])/d, xxs[contmp2])
                        fline = I1

                        Flum = fline*Cons*1e-18 # luminosity in erg/s.
                        elum = esum *Cons*1e-18 # luminosity in erg/s.
                        SFR  = Flum * 6.58*1e-42
                        print('SFR, update, is', SFR/umag)
                        EW_tmp   = simps( ((gaus(xxs[contmp2],*popt)) * c/np.square(xxs[contmp2])/d)/yy2s[contmp2], xxs[contmp2])
                        EW_tmp_u = simps( ((gaus(xxs[contmp2],*popt) + eyys[contmp2]/np.sqrt(len(xxs[contmp2]))) * c/np.square(xxs[contmp2])/d)/yy2s[contmp2], xxs[contmp2])

                    flw.write('%d %.2f %.2f %.2f %.2f %.2f %.2e %.2e %.2f\n'%(LW[ii],fline/umag, esum/umag, fmed/umag, EW_tmp,(EW_tmp_u-EW_tmp), Flum*1e-18/umag, elum*1e-18/umag, SFR/umag))

                except Exception:
                    fsum = np.zeros(len(xxs))
                    for ff in range(len(fsum)):
                        fsum[ff] = (yys[ff]+yy2s[ff])/np.square(xxs[ff])

                    fline = np.sum(fsum) / d*c
                    flw.write('%d %.2f %.2f %.2f %d %d %d %d %d\n'%(LW[ii],fline,esum,fmed, -99, 0, -99, 0, 0))
                    pass

    except:
        pass

    flw.close()

    ############
    # Zoom Line
    ############
    #ax2t.plot(x0/zscl, ysum * c/np.square(x0)/d, '-', lw=0.5, color='gray', zorder=3.)
    conspec = (NR<10000) #& (fy/ey>1)
    ax2t.fill_between(xg1/zscl, (fg1-eg1) * c/np.square(xg1)/d, (fg1+eg1) * c/np.square(xg1)/d, lw=0, color='#DF4E00', zorder=10, alpha=0.7, label='')
    ax2t.fill_between(xg0/zscl, (fg0-eg0) * c/np.square(xg0)/d, (fg0+eg0) * c/np.square(xg0)/d, lw=0, color='royalblue', zorder=10, alpha=0.2, label='')
    ax2t.errorbar(xg1/zscl, fg1 * c/np.square(xg1)/d, yerr=eg1 * c/np.square(xg1)/d, lw=0.5, color='#DF4E00', zorder=10, alpha=1., label='', capsize=0)
    ax2t.errorbar(xg0/zscl, fg0 * c/np.square(xg0)/d, yerr=eg0 * c/np.square(xg0)/d, lw=0.5, color='royalblue', zorder=10, alpha=1., label='', capsize=0)


    # From MCMC chain
    file = 'chain_' + ID + '_PA' + PA + '_corner.cpkl'
    niter = 0
    data = loadcpkl(os.path.join('./'+file))
    try:
        ndim   = data['ndim']     # By default, use ndim and burnin values contained in the cpkl file, if present.
        burnin = data['burnin']
        nmc    = data['niter']
        nwalk  = data['nwalkers']
        Nburn  = burnin #*20
        res    = data['chain'][:]
    except:
        if verbose: print(' =   >   NO keys of ndim and burnin found in cpkl, use input keyword values')

    samples  = res #.chain[:, :, :].reshape((-1, ndim))

    #for kk in range(int(nmc/5)):
    nmc2 = 300
    ytmp = np.zeros((nmc2,len(ysum)), dtype='float64')
    ytmpmax = np.zeros(len(ysum), dtype='float64')
    ytmpmin = np.zeros(len(ysum), dtype='float64')
    for kk in range(0,nmc2,1):
        nr = np.random.randint(len(samples))
        Av_tmp = samples['Av'][nr]
        for ss in range(len(age)):
            AA_tmp = samples['A'+str(ss)][nr]
            ZZ_tmp = samples['Z'+str(ss)][nr]

            if ss == 0:
                mod0_tmp, xm_tmp = fnc.tmp03(ID, PA, AA_tmp, Av_tmp, ss, ZZ_tmp, zbes, lib_all, tau0=tau0)
                fm_tmp = mod0_tmp
            else:
                mod0_tmp, xx_tmp = fnc.tmp03(ID, PA, AA_tmp, Av_tmp, ss, ZZ_tmp, zbes, lib_all, tau0=tau0)
                fm_tmp += mod0_tmp

        ytmp[kk,:] = fm_tmp[:] * c/ np.square(xm_tmp[:]) / d
        ax1.plot(xm_tmp, fm_tmp * c/ np.square(xm_tmp) / d, '-', lw=1, color='gray', zorder=-2, alpha=0.02)
        ax2t.plot(xm_tmp/zscl, fm_tmp * c/np.square(xm_tmp)/d, '-', lw=0.5, color='gray', zorder=3., alpha=0.02)

    #for kk in range(len(ytmp)):
    #    ytmpmin[kk] = np.percentile(ytmp[:,kk],16)
    #    ytmpmax[kk] = np.percentile(ytmp[:,kk],84)
    #ax1.fill_between(xm_tmp, ytmpmin, ytmpmax, linestyle='-', lw=1, facecolor='gray', zorder=-2, alpha=0.4)
    #ax2t.fill_between(xm_tmp/zscl, ytmpmin, ytmpmax, linestyle='-', lw=1, facecolor='gray', zorder=-2, alpha=0.4)

    #######################################
    ax2t.set_xlabel('RF wavelength ($\mathrm{\mu m}$)')
    ax2t.set_xlim(3600, 5400)
    conaa = (x0/zscl>3300) & (x0/zscl<6000)
    ymaxzoom = np.max(ysum[conaa]*c/np.square(x0[conaa])/d) * 1.2
    yminzoom = np.min(ysum[conaa]*c/np.square(x0[conaa])/d) / 1.2
    ax2t.set_ylim(yminzoom, ymaxzoom)
    #ax2t.legend(loc=1, ncol=2, fontsize=7)
    ax1.xaxis.labelpad = -3
    ax2t.xaxis.labelpad = -2
    ax2t.set_yticklabels(())
    ax2t.set_xticks([4000, 5000])
    ax2t.set_xticklabels(['0.4', '0.5'])

    ###############
    # Line name
    ###############
    LN0 = ['Mg2', '$NeIV$', '[OII]', 'H$\theta$', 'H$\eta$', 'Ne3?', 'H$\delta$', 'H$\gamma$', 'H$\\beta$', 'O3', 'O3', 'Mgb', 'Halpha', 'S2L', 'S2H']
    LW0 = [2800, 3347, 3727, 3799, 3836, 3869, 4102, 4341, 4861, 4959, 5007, 5175, 6563, 6717, 6731]
    fsl = 9 # Fontsize for line

    try:
        for ii in range(len(LW)):
            ll = np.argmin(np.abs(LW[ii]-LW0[:]))

            if ll == 2 and FLW[ii] == 1: # FLW is the flag for line fitting.
                yyl = np.arange(yminzoom+(ymaxzoom-yminzoom)*0.5,yminzoom+(ymaxzoom-yminzoom)*0.65, 0.01)
                xxl = yyl * 0 + LW0[ll]
                ax2t.errorbar(xxl, yyl, lw=0.5, color=lcb, zorder=20, alpha=1., label='', capsize=0)
                ax2t.text(xxl[0]-130, yyl[0]*1.28, '%s'%(LN0[ll]),  color=lcb, fontsize=9, rotation=90)

            elif (ll == 9 and FLW[ii] == 1):
                yyl = np.arange(yminzoom+(ymaxzoom-yminzoom)*0.5,yminzoom+(ymaxzoom-yminzoom)*0.65, 0.01)
                xxl = yyl * 0 + LW0[ll]
                ax2t.errorbar(xxl, yyl, lw=0.5, color=lcb, zorder=20, alpha=1., label='', capsize=0)

            elif (ll == 10 and FLW[ii] == 1):
                yyl = np.arange(yminzoom+(ymaxzoom-yminzoom)*0.5,yminzoom+(ymaxzoom-yminzoom)*0.65, 0.01)
                xxl = yyl * 0 + LW0[ll]
                ax2t.errorbar(xxl, yyl, lw=0.5, color=lcb, zorder=20, alpha=1., label='', capsize=0)
                ax2t.text(xxl[0]+40, yyl[0]*0.75, '%s'%(LN0[ll]),  color=lcb, fontsize=9, rotation=90)

            elif FLW[ii] == 1 and (ll == 6 or ll == 7 or ll == 8):
                yyl = np.arange(yminzoom+(ymaxzoom-yminzoom)*0.2,yminzoom+(ymaxzoom-yminzoom)*0.35, 0.01)
                xxl = yyl * 0 + LW0[ll]
                ax2t.errorbar(xxl, yyl, lw=0.5, color=lcb, zorder=20, alpha=1., label='', capsize=0)
                ax2t.text(xxl[0]+40, yyl[0]*0.95, '%s'%(LN0[ll]),  color=lcb, fontsize=9, rotation=90)

            elif ll == 6 or ll == 7 or ll == 8:
                yyl = np.arange(yminzoom+(ymaxzoom-yminzoom)*0.2,yminzoom+(ymaxzoom-yminzoom)*0.35, 0.01)
                xxl = yyl * 0 + LW0[ll]
                ax2t.errorbar(xxl, yyl, lw=0.5, color='gray', zorder=1, alpha=1., label='', capsize=0)
                ax2t.text(xxl[0]+40, yyl[0]*0.95, '%s'%(LN0[ll]),  color='gray', fontsize=9, rotation=90)

            elif FLW[ii] == 1:
                yyl = np.arange(yminzoom+(ymaxzoom-yminzoom)*0.7,yminzoom+(ymaxzoom-yminzoom)*.95, 0.01)
                xxl = yyl * 0 + LW0[ll]
                ax2t.errorbar(xxl, yyl, lw=0.5, color=lcb, zorder=20, alpha=1., label='', capsize=0)
                ax2t.text(xxl[0]+40, yyl[0]*1.25, '%s'%(LN0[ll]),  color=lcb, fontsize=9, rotation=90)

    except:
        pass

    ####################
    # Plot Different Z
    ####################
    '''
    # Deprecated;
    if f_Z_all == 1:
        fileZ = 'Z_' + ID + '_PA' + PA + '.cat'
        Zini, chi, Av = np.loadtxt(fileZ, comments='#', unpack=True, usecols=[1, 2, 3+len(age)])
        Atmp  = np.zeros((len(age),len(Zini)), 'float64')
        Ztmp  = np.zeros((len(age),len(Zini)), 'float64')
        for aa in range(len(age)):
            Atmp[aa,:] = np.loadtxt(fileZ, comments='#', unpack=True, usecols=[3+aa])
            Ztmp[aa,:] = np.loadtxt(fileZ, comments='#', unpack=True, usecols=[3+len(age)+1+aa])

        for jj in range(len(Zini)):
            for aa in range(len(age)):
                if aa == 0:
                    y0_r, x0_r    = fnc.tmp03(ID, PA, Atmp[aa, jj], Av[jj], aa, Ztmp[aa, jj], zbes, lib_all)
                    y0_rp, x0_rp  = fnc.tmp03(ID, PA, Atmp[aa, jj], Av[jj], aa, Ztmp[aa, jj], zbes, lib)
                else:
                    y0_rr, x0_r     = fnc.tmp03(ID, PA, Atmp[aa, jj], Av[jj], aa, Ztmp[aa, jj], zbes, lib_all)
                    y0_rrp, x0_rrp  = fnc.tmp03(ID, PA, Atmp[aa, jj], Av[jj], aa, Ztmp[aa, jj], zbes, lib)
                    y0_r  += y0_rr
                    y0_rp += y0_rrp

            ysum_Z = y0_r
            chi2_ind = sum((np.square(fy-y0_rp)*wht3)[conw])
            print('At Z=%.2f; chi2/nu is %.2f', Zini[jj], chi2_ind/nu)

            if Zini[jj]>=0:
                ax1.plot(x0_r, (ysum_Z)* c/np.square(x0_r)/d, '--', lw=0.3+0.2*jj, color='gray', zorder=-3, alpha=0.7, label='$\log\mathrm{Z_*}=\ \ %.2f$'%(Zini[jj])) # Z here is Zinitial.
            else:
                ax1.plot(x0_r, (ysum_Z)* c/np.square(x0_r)/d, '--', lw=0.3+0.2*jj, color='gray', zorder=-3, alpha=0.7, label='$\log\mathrm{Z_*}=%.2f$'%(Zini[jj])) # Z here is Zinitial.
            ax2t.plot(x0_r/zscl, ysum_Z * c/ np.square(x0_r) / d, '--', lw=0.3+0.2*jj, color='gray', zorder=-3, alpha=0.5)
    '''

    ################
    # RGB
    ################
    try:
    #if 1>0:
        from scipy import misc
        rgb_array = misc.imread('/Users/tmorishita/Box Sync/Research/M18_rgb/rgb_'+str(int(ID))+'.png')
        axicon = fig.add_axes([0.68, 0.53, 0.4, 0.4])
        axicon.imshow(rgb_array, interpolation='nearest', origin='upper')
        axicon.set_xticks([])
        axicon.set_yticks([])
        #xxl = np.arange(0, twokpc, 0.01)
        #axicon.errorbar(xxl + 0.2, xxl*0+0.2, lw=1., color='w', zorder=1, alpha=1., capsize=0)
        #axicon.text(0.1, 1.5, '5kpc', color='w', fontsize=12)
        #axicon.text(10, 10.0, '%s'%(ID), color='w', fontsize=14)
        #axicon.text(11, 10.0, '%s'%(ID), color='w', fontsize=14)
        #axicon.text(10, 11.0, '%s'%(ID), color='w', fontsize=14)
    except:
        pass
    ####################
    ## Save
    ####################
    #plt.show()
    ax1.legend(loc=1, fontsize=11)
    if figpdf:
        plt.savefig('SPEC_' + ID + '_PA' + PA + '_spec.pdf', dpi=300)
    else:
        plt.savefig('SPEC_' + ID + '_PA' + PA + '_spec.png', dpi=150)
    plt.close()

def plot_sed_demo(ID, PA, Z=np.arange(-1.2,0.4249,0.05), age=[0.01, 0.1, 0.3, 0.7, 1.0, 3.0], tau0=[0.1,0.2,0.3], flim=0.01, figpdf=False):
    DIR_TMP = 'templates/'
    col = ['darkred', 'r', 'coral','orange','g','lightgreen', 'lightblue', 'b','indigo','violet','k']

    nage = np.arange(0,len(age),1)

    from .function_class import Func
    from .basic_func import Basic

    fnc  = Func(ID, PA, Z, nage, dust_model=dust_model) # Set up the number of Age/ZZ
    bfnc = Basic(Z)



    ###########################
    # Open input file
    ###########################
    file = DIR_TMP + 'sim_param_' + ID + '_PA' + PA + '.fits'
    hdu0 = fits.open(file) # open a FITS file

    Cnorm = float(hdu0[0].header['Cnorm'])

    Avin = float(hdu0[0].header['AV'])
    Ain  = np.zeros(len(age), dtype='float64')
    Zin  = np.zeros(len(age), dtype='float64')
    for aa in range(len(age)):
        Ain[aa] = hdu0[0].header['A'+str(aa)]
        Zin[aa] = hdu0[0].header['Z'+str(aa)]

    Ainsum = np.sum(Ain)

    zbest = float(hdu0[0].header['z'])

    ################
    # RF colors.
    import os.path
    home = os.path.expanduser('~')
    #fil_path = '/Users/tmorishita/eazy-v1.01/PROG/FILT/'
    fil_path = home + '/Dropbox/FILT/'
    fil_u = fil_path+'u.fil'
    fil_b = fil_path+'b.fil'
    fil_v = fil_path+"v.fil"
    fil_j = fil_path+"j.fil"
    fil_k = fil_path+"k.fil"
    fil_f125w = fil_path+"f125w.fil"
    fil_f160w = fil_path+"f160w.fil"
    fil_36 = fil_path+"3.6.fil"
    fil_45 = fil_path+"4.5.fil"
    fil_jh = fil_path+"f140w.fil"

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

    djh = np.loadtxt(fil_jh,comments="#")
    ljh = djh[:,1]
    fjh = djh[:,2]

    c      = 3.e18 # A/s
    chimax = 1.
    m0set   = 25.0
    d      = 10**(73.6/2.5) * 1e-18 # From [ergs/s/cm2/A] to [ergs/s/cm2/Hz]
    #d = 10**(-73.6/2.5) # From [ergs/s/cm2/Hz] to [ergs/s/cm2/A]

    #############
    # Plot.
    #############
    fig = plt.figure(figsize=(7,5))
    fig.subplots_adjust(top=0.98, bottom=0.12, left=0.06, right=0.99, hspace=0.15, wspace=0.1)
    #ax0 = fig.add_subplot(121)
    ax1 = fig.add_subplot(131)
    ax3 = fig.add_subplot(132)
    ax2 = fig.add_subplot(133)

    ##################
    # Fitting Results
    ##################
    SNlim = 3 # avobe which SN line is shown.

    ###########################
    # Open result file
    ###########################
    file = 'summary_' + ID + '_PA' + PA + '.fits'
    hdul = fits.open(file) # open a FITS file

    # Redshift MC
    zp50  = hdul[1].data['zmc'][1]
    zp16  = hdul[1].data['zmc'][0]
    zp84  = hdul[1].data['zmc'][2]

    M50 = hdul[1].data['ms'][1]
    M16 = hdul[1].data['ms'][0]
    M84 = hdul[1].data['ms'][2]
    print('Total stellar mass is %.2e'%(M50))

    A50 = np.zeros(len(age), dtype='float64')
    A16 = np.zeros(len(age), dtype='float64')
    A84 = np.zeros(len(age), dtype='float64')
    for aa in range(len(age)):
        A50[aa] = hdul[1].data['A'+str(aa)][1]
        A16[aa] = hdul[1].data['A'+str(aa)][0]
        A84[aa] = hdul[1].data['A'+str(aa)][2]

    Asum  = np.sum(A50)

    aa = 0
    Av50 = hdul[1].data['Av'+str(aa)][1]
    Av16 = hdul[1].data['Av'+str(aa)][0]
    Av84 = hdul[1].data['Av'+str(aa)][2]
    AAv = [Av50]

    Z50 = np.zeros(len(age), dtype='float64')
    Z16 = np.zeros(len(age), dtype='float64')
    Z84 = np.zeros(len(age), dtype='float64')
    NZbest = np.zeros(len(age), dtype='int')

    for aa in range(len(age)):
        Z50[aa] = hdul[1].data['Z'+str(aa)][1]
        Z16[aa] = hdul[1].data['Z'+str(aa)][0]
        Z84[aa] = hdul[1].data['Z'+str(aa)][2]
        NZbest[aa]= bfnc.Z2NZ(Z50[aa])


    ZZ50 = np.sum(Z50*A50)/np.sum(A50)  # Light weighted Z.

    chi   = hdul[1].data['chi'][0]
    chin  = hdul[1].data['chi'][1]
    fitc  = chin

    Cz0   = hdul[0].header['Cz0']
    Cz1   = hdul[0].header['Cz1']
    zbes  = hdul[0].header['z']
    zscl  = (1.+zbes)

    SN    = hdul[0].header['SN']


    ######################
    # Weight by line
    ######################
    f0    = fits.open(DIR_TMP + 'ms.fits')
    mshdu = f0[1]
    col02 = []

    f1    = fits.open(DIR_TMP + 'spec_all.fits')
    spechdu = f1[1]
    for zz in range(len(Z)):
        col00 = []
        col01 = []
        for pp in range(len(tau0)):
            Zbest = Z[zz]
            Na  = len(age)
            Nz  = 1

            param = np.zeros((Na, 6), dtype='float64')
            param[:,2] = 1e99

            Ntmp  = 1
            chi2  = np.zeros(Ntmp) + 1e99
            snorm = np.zeros(Ntmp)
            agebest = np.zeros(Ntmp)
            avbest = np.zeros(Ntmp)
            #age_univ = cd.age(zbest, use_flat=True, **cosmo)
            if zz == 0 and pp == 0:
                lm0    = spechdu.data['wavelength']
                spec0  = spechdu.data['fspec_'+str(zz)+'_0_'+str(pp)]

            if zz == 4 and pp == 0:
                for aa in range(len(age)):
                    spec0  = spechdu.data['fspec_'+str(zz)+'_'+str(aa)+'_'+str(pp)]
                    con   = (lm0>4800) & (lm0<5200)
                    Cnorm = np.median(spec0[con])
                    ax1.plot(lm0, spec0/Cnorm + (len(age)-aa), color=col[aa], linewidth=0.5, zorder=2, label='%.2f'%(age[aa]))
                    ax2.plot(lm0, spec0/Cnorm + (len(age)-aa), color='gray', linewidth=0.5, zorder=1)
                    ax3.plot(lm0, spec0/Cnorm + (len(age)-aa), color='gray', linewidth=0.5, zorder=1)


            if zz == 0 and pp == 0:
                for aa in range(len(age)):
                    spec0  = spechdu.data['fspec_'+str(zz)+'_'+str(aa)+'_'+str(pp)]
                    con   = (lm0>4800) & (lm0<5200)
                    Cnorm = np.median(spec0[con])
                    if aa == 0:
                        ax3.plot(lm0, spec0/Cnorm + (len(age)-aa), color=col[aa], linewidth=.3, zorder=2, label='$\log Z/Z_\odot=-0.8$')
                    else:
                        ax3.plot(lm0, spec0/Cnorm + (len(age)-aa), color=col[aa], linewidth=.3, zorder=2, label='')

            if zz == 6 and pp == 0:
                for aa in range(len(age)):
                    spec0  = spechdu.data['fspec_'+str(zz)+'_'+str(aa)+'_'+str(pp)]
                    con   = (lm0>4800) & (lm0<5200)
                    Cnorm = np.median(spec0[con])
                    if aa == 0:
                        ax3.plot(lm0, spec0/Cnorm + (len(age)-aa), color=col[aa], linewidth=.7, zorder=2, label='$\log Z/Z_\odot=0.4$')
                    else:
                        ax3.plot(lm0, spec0/Cnorm + (len(age)-aa), color=col[aa], linewidth=.7, zorder=2, label='')


    ######################
    # Mass-to-Light ratio.
    ######################
    ms     = np.zeros(len(age), dtype='float64')
    f0     = fits.open(DIR_TMP + 'ms_' + ID + '_PA' + PA + '.fits')
    sedpar = f0[1]
    for aa in range(len(age)):
        ms[aa]    = sedpar.data['ML_' +  str(int(NZbest[aa]))][aa]


    ################
    # Open ascii file and stock to array.
    lib     = fnc.open_spec_fits(ID, PA, fall=0, tau0=tau0)
    lib_all = fnc.open_spec_fits(ID, PA, fall=1, tau0=tau0)

    II0   = nage #[0,1,2,3] # Number for templates
    iimax = len(II0)-1


    def fnutolam(lam, fnu):
        Ctmp = lam **2/c * 10**((48.6+m0set)/2.5) #/ delx_org
        flam  = fnu / Ctmp
        return flam

    for aa in range(len(II0)):
        y0, x0   = fnc.tmp03(ID, PA, 1, 0, aa, 0, zbes, lib_all, tau0=tau0)
        #y0p, x0p = fnc.tmp03(ID, PA, 1, 0, aa, 0, zbes, lib, tau0=tau0)

        lm0   = x0/(1.+zbes)
        con   = (lm0>4800) & (lm0<5200)
        flam  = fnutolam(lm0, y0)
        Cnorm = np.median(flam[con])
        con2  = (lm0>3000) & (lm0<5000)
        if aa == 0:
            ax2.plot(lm0[con2], flam[con2]/Cnorm + (len(age)-aa), color=col[aa], linewidth=1., zorder=2, label='$R\sim150$')
        else:
            ax2.plot(lm0[con2], flam[con2]/Cnorm + (len(age)-aa), color=col[aa], linewidth=1., zorder=2)

    ymax = np.max(spec0)
    #NDIM = 19

    #ax1.text(12000, ymax*0.9, '%s'%(ID), fontsize=10, color='k')
    xboxl = 17000
    xboxu = 28000

    ax1.set_xlabel('Wavelength ($\mathrm{\AA}$)', fontsize=14)
    ax1.set_ylabel('F$_{\lambda}$ (arb.)', fontsize=14)

    ax3.set_xlabel('Wavelength ($\mathrm{\AA}$)', fontsize=14)

    ax2.set_xlabel('Wavelength ($\mathrm{\AA}$)', fontsize=14)
    #ax2.set_ylabel('F$_{\lambda}$ (arb)', fontsize=14)

    ax1.set_xlim(2000, 30000)
    ax3.set_xlim(2000, 30000)
    ax2.set_xlim(3200, 5500)
    #ax1.set_xlim(12500, 16000)
    #ax2.set_xscale('log')

    ax1.set_ylim(0., 18)
    ax3.set_ylim(0., 18)
    ax2.set_ylim(1., 13)

    ax1.text(3000, 16.3, '$\log Z/Z_\odot=0$\n$A_V/$mag$=0$', fontsize=10)
    #ax3.text(3000, 16, '$\log Z_*/Z_\odot=0.4$\n$A_V/$mag$=0$', fontsize=10)
    #ax2.text(4400, 12, '$\log Z_*/Z_\odot=0$\n$A_V/$mag$=0$', fontsize=10)

    ax1.set_xscale('log')
    ax3.set_xscale('log')

    #import matplotlib.ticker as ticker
    import matplotlib
    #ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
    ax1.set_xticks([2500, 5000, 10000, 20000])
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.set_yticklabels(())

    ax3.set_xticks([2500, 5000, 10000, 20000])
    ax3.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax3.set_yticklabels(())

    ax2.set_xticks([4000, 5000])
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.set_yticklabels(())

    ax1.legend(loc=1, fontsize=10)
    ax2.legend(loc=1, fontsize=10)
    ax3.legend(loc=1, fontsize=10)


    #plt.show()
    plt.savefig('demo.pdf')
