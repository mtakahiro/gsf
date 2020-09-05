import numpy as np
import sys
import os

import matplotlib.pyplot as plt

from numpy import log10
from scipy.integrate import simps
from astropy.io import fits
from matplotlib.ticker import FormatStrFormatter

from .function import *
from .function_class import Func
from .basic_func import Basic
#from . import img_scale
#from . import corner
import corner

col = ['violet', 'indigo', 'b', 'lightblue', 'lightgreen', 'g', 'orange', 'coral', 'r', 'darkred']#, 'k']
#col = ['darkred', 'r', 'coral','orange','g','lightgreen', 'lightblue', 'b','indigo','violet','k']

def plot_sed(MB, flim=0.01, fil_path='./', scale=1e-19, f_chind=True, figpdf=False, save_sed=True, inputs=False, \
    mmax=300, dust_model=0, DIR_TMP='./templates/', f_label=False, f_bbbox=False, verbose=False, f_silence=True, \
        f_fill=False, f_fancyplot=False, f_Alog=True):
    '''
    Input:
    ======

    SNlim   : SN limit to show flux or up lim in SED.
    f_chind : If include non-detection in chi2 calculation, using Sawicki12.

    Returns:
    ========
    plots

    '''

    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from scipy.optimize import curve_fit
    from scipy import asarray as ar,exp
    import matplotlib
    import scipy.integrate as integrate
    import scipy.special as special
    import os.path
    from astropy.io import ascii
    import time

    if f_silence:
        import matplotlib
        matplotlib.use("Agg")

    def gaus(x,a,x0,sigma):
        return a*exp(-(x-x0)**2/(2*sigma**2))

    col = ['violet', 'indigo', 'b', 'lightblue', 'lightgreen', 'g', 'orange', 'coral', 'r', 'darkred']#, 'k']
    lcb = '#4682b4' # line color, blue

    nage = MB.nage #np.arange(0,len(age),1)
    fnc  = MB.fnc #Func(ID, PA, Z, nage, dust_model=dust_model, DIR_TMP=DIR_TMP) # Set up the number of Age/ZZ
    bfnc = MB.bfnc #Basic(Z)
    ID   = MB.ID
    PA   = MB.PA
    Z    = MB.Zall
    age  = MB.age  #[0.01, 0.1, 0.3, 0.7, 1.0, 3.0],
    try:
        age = MB.age_fix
    except:
        age  = MB.age
    tau0 = MB.tau0 #[0.1,0.2,0.3]

    nstep_plot = 1
    if MB.f_bpass:
        nstep_plot = 30

    SNlim = MB.SNlim

    ################
    # RF colors.
    home = os.path.expanduser('~')
    c      = MB.c
    chimax = 1.
    m0set  = MB.m0set
    Mpc_cm = MB.Mpc_cm
    d = 10**(73.6/2.5) * scale # From [ergs/s/cm2/A] to [ergs/s/cm2/Hz]

    ##################
    # Fitting Results
    ##################
    DIR_FILT = MB.DIR_FILT
    SFILT    = MB.filts

    try:
        f_err = MB.ferr
    except:
        f_err = 0

    ###########################
    # Open result file
    ###########################
    file = 'summary_' + ID + '_PA' + PA + '.fits'
    hdul = fits.open(file) # open a FITS file

    ndim_eff = hdul[0].header['NDIM']

    # Redshift MC
    zp16  = hdul[1].data['zmc'][0]
    zp50  = hdul[1].data['zmc'][1]
    zp84  = hdul[1].data['zmc'][2]

    # Stellar mass MC
    M16 = hdul[1].data['ms'][0]
    M50 = hdul[1].data['ms'][1]
    M84 = hdul[1].data['ms'][2]
    if verbose:
        print('Total stellar mass is %.2e'%(M50))

    # Amplitude MC
    A50 = np.zeros(len(age), dtype='float64')
    A16 = np.zeros(len(age), dtype='float64')
    A84 = np.zeros(len(age), dtype='float64')
    for aa in range(len(age)):
        A16[aa] = 10**hdul[1].data['A'+str(aa)][0]
        A50[aa] = 10**hdul[1].data['A'+str(aa)][1]
        A84[aa] = 10**hdul[1].data['A'+str(aa)][2]

    Asum  = np.sum(A50)

    aa = 0
    Av16 = hdul[1].data['Av'+str(aa)][0]
    Av50 = hdul[1].data['Av'+str(aa)][1]
    Av84 = hdul[1].data['Av'+str(aa)][2]
    AAv = [Av50]

    Z50 = np.zeros(len(age), dtype='float64')
    Z16 = np.zeros(len(age), dtype='float64')
    Z84 = np.zeros(len(age), dtype='float64')
    NZbest = np.zeros(len(age), dtype='int')
    for aa in range(len(age)):
        Z16[aa] = hdul[1].data['Z'+str(aa)][0]
        Z50[aa] = hdul[1].data['Z'+str(aa)][1]
        Z84[aa] = hdul[1].data['Z'+str(aa)][2]
        NZbest[aa]= bfnc.Z2NZ(Z50[aa])

    # Light weighted Z.
    ZZ50 = np.sum(Z50*A50)/np.sum(A50)

    # FIR Dust;
    try:
        MD16 = hdul[1].data['MDUST'][0]
        MD50 = hdul[1].data['MDUST'][1]
        MD84 = hdul[1].data['MDUST'][2]
        TD16 = hdul[1].data['TDUST'][0]
        TD50 = hdul[1].data['TDUST'][1]
        TD84 = hdul[1].data['TDUST'][2]
        nTD16 = hdul[1].data['nTDUST'][0]
        nTD50 = hdul[1].data['nTDUST'][1]
        nTD84 = hdul[1].data['nTDUST'][2]
        DFILT   = inputs['FIR_FILTER'] # filter band string.
        DFILT   = [x.strip() for x in DFILT.split(',')]
        DFWFILT = fil_fwhm(DFILT, DIR_FILT)
        if verbose:
            print('Total dust mass is %.2e'%(MD50))
        f_dust = True
    except:
        f_dust = False

    chi   = hdul[1].data['chi'][0]
    chin  = hdul[1].data['chi'][1]
    fitc  = chin

    Cz0   = hdul[0].header['Cz0']
    Cz1   = hdul[0].header['Cz1']
    zbes  = zp50 #hdul[0].header['z']
    zscl = (1.+zbes)

    ###############################
    # Data taken from
    ###############################
    if MB.f_dust:
        MB.dict = MB.read_data(MB.Cz0, MB.Cz1, MB.zgal, add_fir=True)
    else:
        MB.dict = MB.read_data(MB.Cz0, MB.Cz1, MB.zgal)

    #dat  = np.loadtxt(DIR_TMP + 'spec_obs_' + ID + '_PA' + PA + '.cat', comments='#')
    NR   = MB.dict['NR'] #dat[:, 0]
    x    = MB.dict['x'] #dat[:, 1]
    fy   = MB.dict['fy'] #dat[:, 2]
    ey   = MB.dict['ey'] #dat[:, 3]

    con0 = (NR<1000) #& (fy/ey>SNlim)
    xg0  = x[con0]
    fg0  = fy[con0] * Cz0
    eg0  = ey[con0] * Cz0
    con1 = (NR>=1000) & (NR<10000) #& (fy/ey>SNlim)
    xg1  = x[con1]
    fg1  = fy[con1] * Cz1
    eg1  = ey[con1] * Cz1
    if len(xg0)>0 or len(xg1)>0:
        f_grsm = True
    else:
        f_grsm = False

    # Weight is set to zero for those no data (ey<0).
    wht = fy * 0
    con_wht = (ey>0)
    wht[con_wht] = 1./np.square(ey[con_wht])

    # BB data points;
    NRbb = MB.dict['NRbb'] #dat[:, 0]
    xbb  = MB.dict['xbb'] #dat[:, 1]
    fybb = MB.dict['fybb'] #dat[:, 2]
    eybb = MB.dict['eybb'] #dat[:, 3]
    exbb = MB.dict['exbb'] #dat[:, 4]
    snbb = fybb/eybb

    ######################
    # Weight by line
    ######################
    wh0  = 1./np.square(eg0)
    LW0  = []
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


    #############
    # Plot.
    #############
    # Set the inset.
    if f_grsm or f_dust:
        fig = plt.figure(figsize=(7.,3.2))
        fig.subplots_adjust(top=0.98, bottom=0.16, left=0.1, right=0.99, hspace=0.15, wspace=0.25)
        ax1 = fig.add_subplot(111)
        if f_grsm:
            ax2t = inset_axes(ax1, width="30%", height="20%", loc=1)
        if f_dust:
            ax3t = inset_axes(ax1, width="30%", height="20%", loc=4)
    else:
        fig = plt.figure(figsize=(5.5,2.2))
        fig.subplots_adjust(top=0.98, bottom=0.16, left=0.1, right=0.99, hspace=0.15, wspace=0.25)
        ax1 = fig.add_subplot(111)

    # Plot data;
    conspec = (NR<10000) #& (fy/ey>1)
    ax1.plot(xg0, fg0 * c / np.square(xg0) / d, marker='', linestyle='-', linewidth=0.5, ms=0.1, color='royalblue', label='')
    ax1.plot(xg1, fg1 * c / np.square(xg1) / d, marker='', linestyle='-', linewidth=0.5, ms=0.1, color='#DF4E00', label='')

    #######################################
    # D.Kelson like Box for BB photometry
    #######################################
    #col_dat = 'darkgreen'
    #col_dat = 'tomato'
    col_dat = 'r'
    if f_bbbox:
        for ii in range(len(xbb)):
            if eybb[ii]<100 and fybb[ii]/eybb[ii]>1:
                xx = [xbb[ii]-exbb[ii],xbb[ii]-exbb[ii]]
                yy = [(fybb[ii]-eybb[ii])*c/np.square(xbb[ii])/d, (fybb[ii]+eybb[ii])*c/np.square(xbb[ii])/d]
                ax1.plot(xx, yy, color='k', linestyle='-', linewidth=0.5, zorder=3)
                xx = [xbb[ii]+exbb[ii],xbb[ii]+exbb[ii]]
                yy = [(fybb[ii]-eybb[ii])*c/np.square(xbb[ii])/d, (fybb[ii]+eybb[ii])*c/np.square(xbb[ii])/d]
                ax1.plot(xx, yy, color='k', linestyle='-', linewidth=0.5, zorder=3)
                xx = [xbb[ii]-exbb[ii],xbb[ii]+exbb[ii]]
                yy = [(fybb[ii]-eybb[ii])*c/np.square(xbb[ii])/d, (fybb[ii]-eybb[ii])*c/np.square(xbb[ii])/d]
                ax1.plot(xx, yy, color='k', linestyle='-', linewidth=0.5, zorder=3)
                xx = [xbb[ii]-exbb[ii],xbb[ii]+exbb[ii]]
                yy = [(fybb[ii]+eybb[ii])*c/np.square(xbb[ii])/d, (fybb[ii]+eybb[ii])*c/np.square(xbb[ii])/d]
                ax1.plot(xx, yy, color='k', linestyle='-', linewidth=0.5, zorder=3)
    else: # Normal BB plot;
        # Detection;
        conbb_hs = (fybb/eybb>SNlim)
        ax1.errorbar(xbb[conbb_hs], fybb[conbb_hs] * c / np.square(xbb[conbb_hs]) / d, \
        yerr=eybb[conbb_hs]*c/np.square(xbb[conbb_hs])/d, color='k', linestyle='', linewidth=0.5, zorder=4)
        ax1.plot(xbb[conbb_hs], fybb[conbb_hs] * c / np.square(xbb[conbb_hs]) / d, \
        marker='.', color=col_dat, linestyle='', linewidth=0, zorder=4, ms=8)#, label='Obs.(BB)')

        try:
            # For any data removed fron fit (i.e. IRAC excess):
            data_ex = ascii.read(DIR_TMP + 'bb_obs_' + ID + '_PA' + PA + '_removed.cat')
            NR_ex = data_ex['col1']
        except:
            NR_ex = []

        # Upperlim;
        sigma = 1.0
        leng = np.max(fybb[conbb_hs] * c / np.square(xbb[conbb_hs]) / d) * 0.05 #0.2
        conebb_ls = (fybb/eybb<=SNlim) & (eybb>0)
        
        for ii in range(len(xbb)):
            if NR[ii] in NR_ex[:]:
                conebb_ls[ii] = False
        
        ax1.errorbar(xbb[conebb_ls], eybb[conebb_ls] * c / np.square(xbb[conebb_ls]) / d * sigma, yerr=leng,\
            uplims=eybb[conebb_ls] * c / np.square(xbb[conebb_ls]) / d * sigma, linestyle='',color=col_dat, marker='', ms=4, label='', zorder=4, capsize=3)


    # For any data removed fron fit (i.e. IRAC excess):
    try:
        col_ex = 'lawngreen'
        #col_ex = 'limegreen'
        #col_ex = 'r'
        # Currently, this file is made after FILTER_SKIP;
        data_ex = ascii.read(DIR_TMP + 'bb_obs_' + ID + '_PA' + PA + '_removed.cat')
        x_ex    = data_ex['col2']
        fy_ex   = data_ex['col3']
        ey_ex   = data_ex['col4']
        ex_ex   = data_ex['col5']

        ax1.errorbar(x_ex, fy_ex * c / np.square(x_ex) / d, \
        xerr=ex_ex, yerr=ey_ex*c/np.square(x_ex)/d, color='k', linestyle='', linewidth=0.5, zorder=5)
        ax1.scatter(x_ex, fy_ex * c / np.square(x_ex) / d, marker='s', color=col_ex, edgecolor='k', zorder=5, s=30)
    except:
        pass


    #####################################
    # Open ascii file and stock to array.
    lib     = fnc.open_spec_fits(MB,fall=0)
    lib_all = fnc.open_spec_fits(MB,fall=1)
    if f_dust:
        DT0 = float(inputs['TDUST_LOW'])
        DT1 = float(inputs['TDUST_HIG'])
        dDT = float(inputs['TDUST_DEL'])
        Temp= np.arange(DT0,DT1,dDT)
        lib_dust     = fnc.open_spec_dust_fits(MB,fall=0)
        lib_dust_all = fnc.open_spec_dust_fits(MB,fall=1)

    II0   = nage #[0,1,2,3] # Number for templates
    iimax = len(II0)-1

    #
    # This is for UVJ color time evolution.
    #
    fwuvj = open(ID + '_PA' + PA + '_uvj.txt', 'w')
    fwuvj.write('# age uv vj\n')
    Asum = np.sum(A50[:])

    alp = .8
    for jj in range(len(age)):

        ii = int(len(II0) - jj - 1) # from old to young templates.

        if jj == 0:
            y0, x0 = fnc.tmp03(A50[ii], AAv[0], ii, Z50[ii], zbes, lib_all)
            y0p, x0p = fnc.tmp03(A50[ii], AAv[0], ii, Z50[ii], zbes, lib)
            ysum  = y0
            ysump = y0p
            #if f_fill:
            #    ax1.fill_between(x0[::nstep_plot], (ysum * 0)[::nstep_plot], (ysum * c/ np.square(x0) / d)[::nstep_plot], linestyle='None', lw=0.5, color=col[ii], alpha=alp, zorder=-1, label='')
        else:
            y0_r, x0_tmp = fnc.tmp03(A50[ii], AAv[0], ii, Z50[ii], zbes, lib_all)
            y0p, x0p = fnc.tmp03(A50[ii], AAv[0], ii, Z50[ii], zbes, lib)
            ysum  += y0_r
            ysump += y0p
            #if f_fill:
            #    ax1.fill_between(x0[::nstep_plot], ((ysum - y0_r) * c/ np.square(x0) / d)[::nstep_plot], (ysum * c/ np.square(x0) / d)[::nstep_plot], linestyle='None', lw=0.5, color=col[ii], alpha=alp, zorder=-1, label='')
        '''
        if True:
            alp_fill = 0.8
            if jj == 0:
                yprev = ysum[:] * 0
                ynow = (ysum[:] * c / np.square(x0) / d)
            else:
                yprev = ynow[:]
                ynow = (ysum[:] * c / np.square(x0) / d)
            ax1.fill_between(x0[::nstep_plot], yprev[::nstep_plot], ynow[::nstep_plot], linestyle='None', lw=0.5, color=col[ii], alpha=alp_fill, zorder=-1, label='')
        '''

        try:
            ysum_wid = ysum * 0
            for kk in range(0,ii+1,1):
                tt = int(len(II0) - kk - 1)
                nn = int(len(II0) - ii - 1)

                nZ = bfnc.Z2NZ(Z50[tt])
                y0_wid, x0_wid = fnc.open_spec_fits_dir(tt, nZ, nn, AAv[0], zbes, A50[tt])
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
        except:
            print('Error in writing fw.')
            pass

    fwuvj.close()
    
    # FIR dust plot;
    if f_dust:
        from lmfit import Parameters
        par = Parameters()
        par.add('MDUST',value=MD50)
        par.add('TDUST',value=nTD50)
        par.add('zmc',value=zp50)
        y0d, x0d = fnc.tmp04_dust(par.valuesdict(), zbes, lib_dust_all)
        ax1.plot(x0d, y0d * c/ np.square(x0d) / d, '--', lw=0.5, color='purple', zorder=-1, label='')
        ax3t.plot(x0d, y0d * c/ np.square(x0d) / d, '--', lw=0.5, color='purple', zorder=-1, label='')

        # data;
        ddat  = np.loadtxt(DIR_TMP + 'bb_dust_obs_' + ID + '_PA' + PA + '.cat', comments='#')
        NRbbd = ddat[:, 0]
        xbbd  = ddat[:, 1]
        fybbd = ddat[:, 2]
        eybbd = ddat[:, 3]
        exbbd = ddat[:, 4]
        snbbd = fybbd/eybbd

        try:
            conbbd_hs = (fybbd/eybbd>SNlim)
            ax1.errorbar(xbbd[conbbd_hs], fybbd[conbbd_hs] * c / np.square(xbbd[conbbd_hs]) / d, \
            yerr=eybbd[conbbd_hs]*c/np.square(xbbd[conbbd_hs])/d, color='k', linestyle='', linewidth=0.5, zorder=4)
            ax1.plot(xbbd[conbbd_hs], fybbd[conbbd_hs] * c / np.square(xbbd[conbbd_hs]) / d, \
            '.r', linestyle='', linewidth=0, zorder=4)#, label='Obs.(BB)')
            ax3t.plot(xbbd[conbbd_hs], fybbd[conbbd_hs] * c / np.square(xbbd[conbbd_hs]) / d, \
            '.r', linestyle='', linewidth=0, zorder=4)#, label='Obs.(BB)')
        except:
            pass

        try:
            conebbd_ls = (fybbd/eybbd<=SNlim)
            ax1.errorbar(xbbd[conebbd_ls], eybbd[conebbd_ls] * c / np.square(xbbd[conebbd_ls]) / d, \
            yerr=fybbd[conebbd_ls]*0+np.max(fybbd[conebbd_ls]*c/np.square(xbbd[conebbd_ls])/d)*0.05, \
            uplims=eybbd[conebbd_ls]*c/np.square(xbbd[conebbd_ls])/d, color='r', linestyle='', linewidth=0.5, zorder=4)
            ax3t.errorbar(xbbd[conebbd_ls], eybbd[conebbd_ls] * c / np.square(xbbd[conebbd_ls]) / d, \
            yerr=fybbd[conebbd_ls]*0+np.max(fybbd[conebbd_ls]*c/np.square(xbbd[conebbd_ls])/d)*0.05, \
            uplims=eybbd[conebbd_ls]*c/np.square(xbbd[conebbd_ls])/d, color='r', linestyle='', linewidth=0.5, zorder=4)
        except:
            pass

    #############
    # Main result
    #############
    conbb_ymax = (xbb>0) & (fybb>0) & (eybb>0) & (fybb/eybb>1) # (conbb) &
    ymax = np.max(fybb[conbb_ymax]*c/np.square(xbb[conbb_ymax])/d) * 1.6

    xboxl = 17000
    xboxu = 28000

    ax1.set_xlabel('Observed wavelength ($\mathrm{\mu m}$)', fontsize=12)
    ax1.set_ylabel('Flux ($10^{%d}\mathrm{erg}/\mathrm{s}/\mathrm{cm}^{2}/\mathrm{\AA}$)'%(np.log10(scale)),fontsize=12,labelpad=-2)

    x1max = 22000
    if x1max < np.max(xbb[conbb_ymax]):
        x1max = np.max(xbb[conbb_ymax]) * 1.5
    ax1.set_xlim(2200, 11000)
    ax1.set_xscale('log')
    ax1.set_ylim(-ymax*0.1,ymax)
    ax1.text(2300,-ymax*0.08,'SNlimit:%.1f'%(SNlim),fontsize=8)

    xticks = [2500, 5000, 10000, 20000, 40000, 80000, 110000]
    xlabels= ['0.25', '0.5', '1', '2', '4', '8', '']
    #if f_dust:
    #    xticks = [2500, 5000, 10000, 20000, 40000, 80000, 1e7]
    #    xlabels= ['0.25', '0.5', '1', '2', '4', '8', '1000']
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xlabels)


    dely1 = 0.5
    while (ymax-0)/dely1<1:
        dely1 /= 2.
    while (ymax-0)/dely1>4:
        dely1 *= 2.

    y1ticks = np.arange(0, ymax, dely1)
    ax1.set_yticks(y1ticks)
    ax1.set_yticklabels(np.arange(0, ymax, dely1), minor=False)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.yaxis.labelpad = 1.5

    xx = np.arange(1200,160000)
    yy = xx * 0
    ax1.plot(xx, yy, ls='--', lw=0.5, color='k')

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

    ####################
    # For cosmology
    ####################
    DL = MB.cosmo.luminosity_distance(zbes).value * Mpc_cm #, **cosmo) # Luminositydistance in cm
    Cons = (4.*np.pi*DL**2/(1.+zbes))
    #dA     = MB.cosmo.angular_diameter_distance(zbes).value
    #dkpc   = dA * (2*3.14/360/3600)*10**3 # kpc/arcsec
    #twokpc = 5.0/dkpc/0.06 # in pixel

    if f_grsm:
        print('This function (write_lines) needs to be revised.')
        write_lines(ID,PA,zbes)


    ##########################
    # Zoom in Line regions
    ##########################
    if f_grsm:
        conspec = (NR<10000) #& (fy/ey>1)
        ax2t.fill_between(xg1/zscl, (fg1-eg1) * c/np.square(xg1)/d, (fg1+eg1) * c/np.square(xg1)/d, lw=0, color='#DF4E00', zorder=10, alpha=0.7, label='')
        ax2t.fill_between(xg0/zscl, (fg0-eg0) * c/np.square(xg0)/d, (fg0+eg0) * c/np.square(xg0)/d, lw=0, color='royalblue', zorder=10, alpha=0.2, label='')
        ax2t.errorbar(xg1/zscl, fg1 * c/np.square(xg1)/d, yerr=eg1 * c/np.square(xg1)/d, lw=0.5, color='#DF4E00', zorder=10, alpha=1., label='', capsize=0)
        ax2t.errorbar(xg0/zscl, fg0 * c/np.square(xg0)/d, yerr=eg0 * c/np.square(xg0)/d, lw=0.5, color='royalblue', zorder=10, alpha=1., label='', capsize=0)

        xgrism = np.concatenate([xg0,xg1])
        fgrism = np.concatenate([fg0,fg1])
        egrism = np.concatenate([eg0,eg1])
        con4000b = (xgrism/zscl>3400) & (xgrism/zscl<3800) & (fgrism>0) & (egrism>0)
        con4000r = (xgrism/zscl>4200) & (xgrism/zscl<5000) & (fgrism>0) & (egrism>0)
        print('Median SN at 3400-3800 is;', np.median((fgrism/egrism)[con4000b]))
        print('Median SN at 4200-5000 is;', np.median((fgrism/egrism)[con4000r]))


    #
    # From MCMC chain
    #
    file  = 'chain_' + ID + '_PA' + PA + '_corner.cpkl'
    niter = 0
    data  = loadcpkl(os.path.join('./'+file))
    try:
        ndim   = data['ndim']     # By default, use ndim and burnin values contained in the cpkl file, if present.
        burnin = data['burnin']
        nmc    = data['niter']
        nwalk  = data['nwalkers']
        Nburn  = burnin #*20
        res    = data['chain'][:]
    except:
        if verbose: print(' =   >   NO keys of ndim and burnin found in cpkl, use input keyword values')

    samples  = res

    # Saved template;
    ytmp = np.zeros((mmax,len(ysum)), dtype='float64')
    ytmp_each = np.zeros((mmax,len(ysum),len(age)), dtype='float64')

    ytmpmax = np.zeros(len(ysum), dtype='float64')
    ytmpmin = np.zeros(len(ysum), dtype='float64')

    # MUV;
    DL      = MB.cosmo.luminosity_distance(zbes).value * Mpc_cm # Luminositydistance in cm
    DL10    = Mpc_cm/1e6 * 10 # 10pc in cm
    Fuv     = np.zeros(mmax, dtype='float64') # For Muv
    Fuv28   = np.zeros(mmax, dtype='float64') # For Fuv(1500-2800)
    Lir     = np.zeros(mmax, dtype='float64') # For L(8-1000um)
    UVJ     = np.zeros((mmax,4), dtype='float64') # For UVJ color;

    Cmznu   = 10**((48.6+m0set)/(-2.5)) # Conversion from m0_25 to fnu

    # From random chain;
    alp=0.02
    for kk in range(0, mmax, 1):
        nr = np.random.randint(len(samples['A0']))
        try:
            Av_tmp = samples['Av'][nr]
        except:
            Av_tmp = MB.AVFIX

        try:
            zmc = samples['zmc'][nr]
        except:
            zmc = zbes

        if f_err == 1:
            ferr_tmp = samples['f'][nr]
        else:
            ferr_tmp = 1.0

        for ss in range(len(age)):
            try:
                AA_tmp = 10**samples['A'+str(ss)][nr]
            except:
                AA_tmp = 0
                pass
            
            try:
                Ztest  = samples['Z'+str(len(age)-1)][nr]
                ZZ_tmp = samples['Z'+str(ss)][nr]
            except:
                try:
                    ZZ_tmp = samples['Z0'][nr]
                except:
                    ZZ_tmp = MB.ZFIX

            if ss == 0:
                mod0_tmp, xm_tmp = fnc.tmp03(AA_tmp, Av_tmp, ss, ZZ_tmp, zmc, lib_all)
                fm_tmp = mod0_tmp
            else:
                mod0_tmp, xx_tmp = fnc.tmp03(AA_tmp, Av_tmp, ss, ZZ_tmp, zmc, lib_all)
                fm_tmp += mod0_tmp

            if kk == 0 and f_fill:
                alp_fill = 0.5
                ii = int(len(II0) - ss - 1) # from old to young templates.
                x0 = xm_tmp #fnc.tmp03(AA_tmp, Av_tmp, ss, ZZ_tmp, zmc, lib_all)

                if ss == 0:
                    yprev = fm_tmp[:] * 0
                    ysum = (fm_tmp[:] * c / np.square(x0) / d)[::nstep_plot]
                else:
                    yprev = ysum[:]
                    ysum = (fm_tmp[:] * c / np.square(x0) / d)[::nstep_plot]
                ax1.fill_between(x0[::nstep_plot], yprev[::nstep_plot], ysum, linestyle='None', lw=0.5, color=col[ii], alpha=alp_fill, zorder=-1, label='')
                
            # Each;
            ytmp_each[kk,:,ss] = ferr_tmp * mod0_tmp[:] * c / np.square(xm_tmp[:]) / d

        #
        # Dust component;
        #
        if f_dust:
            if kk == 0:
                par  = Parameters()
                par.add('MDUST',value=samples['MDUST'][nr])
                par.add('TDUST',value=samples['TDUST'][nr])
            par['MDUST'].value = samples['MDUST'][nr]
            par['TDUST'].value = samples['TDUST'][nr]
            model_dust, x1_dust = fnc.tmp04_dust(par.valuesdict(), zbes, lib_dust_all)
            if kk == 0:
                deldt  = (x1_dust[1] - x1_dust[0])
                x1_tot = np.append(xm_tmp,np.arange(np.max(xm_tmp),np.max(x1_dust),deldt))

            model_tot  = np.interp(x1_tot,xx_tmp,fm_tmp) + np.interp(x1_tot,x1_dust,model_dust)
            if f_fill:
                ax1.plot(x1_tot, model_tot * c/ np.square(x1_tot) / d, '-', lw=1, color='gray', zorder=-2, alpha=alp)
            ytmp[kk,:] = ferr_tmp * model_tot[:] * c/np.square(x1_tot[:])/d
        else:
            x1_tot = xm_tmp
            ytmp[kk,:] = ferr_tmp * fm_tmp[:] * c/ np.square(xm_tmp[:]) / d
            if f_fill:
                ax1.plot(x1_tot[::nstep_plot], (fm_tmp * c/ np.square(x1_tot) / d)[::nstep_plot], '-', lw=1, color='gray', zorder=-2, alpha=alp)

        #
        # Grism plot + Fuv flux + LIR.
        #
        if f_grsm:
            if f_fill:
                ax2t.plot(x1_tot/zscl, ytmp[kk,:], '-', lw=0.5, color='gray', zorder=3., alpha=0.02)

        # Get FUV flux;
        Fuv[kk]   = get_Fuv(x1_tot[:]/(1.+zbes), (ytmp[kk,:]/(c/np.square(x1_tot)/d)) * (DL**2/(1.+zbes)) / (DL10**2), lmin=1250, lmax=1650)
        Fuv28[kk] = get_Fuv(x1_tot[:]/(1.+zbes), (ytmp[kk,:]/(c/np.square(x1_tot)/d)) * (4*np.pi*DL**2/(1.+zbes))*Cmznu, lmin=1500, lmax=2800)
        Lir[kk]   = 0

        # Get UVJ Color;
        lmconv,fconv = filconv_fast(MB.filts_rf, MB.band_rf, x1_tot[:]/(1.+zbes), (ytmp[kk,:]/(c/np.square(x1_tot)/d)))
        UVJ[kk,0] = -2.5*np.log10(fconv[0]/fconv[2])
        UVJ[kk,1] = -2.5*np.log10(fconv[1]/fconv[2])
        UVJ[kk,2] = -2.5*np.log10(fconv[2]/fconv[3])
        UVJ[kk,3] = -2.5*np.log10(fconv[4]/fconv[3])

        # Do stuff...
        time.sleep(0.01)
        # Update Progress Bar
        printProgressBar(kk, mmax, prefix = 'Progress:', suffix = 'Complete', length = 40)




    #
    # Plot Median SED;
    #
    ytmp16 = np.zeros(len(x1_tot), dtype='float64')
    ytmp50 = np.zeros(len(x1_tot), dtype='float64')
    ytmp84 = np.zeros(len(x1_tot), dtype='float64')

    for kk in range(len(x1_tot[:])):
        ytmp16[kk] = np.percentile(ytmp[:,kk],16)
        ytmp50[kk] = np.percentile(ytmp[:,kk],50)
        ytmp84[kk] = np.percentile(ytmp[:,kk],84)

    #
    if not f_fill:
        ax1.fill_between(x1_tot[::nstep_plot], ytmp16[::nstep_plot], ytmp84[::nstep_plot], ls='-', lw=.5, color='gray', zorder=-2, alpha=0.5)
    ax1.plot(x1_tot[::nstep_plot], ytmp50[::nstep_plot], '-', lw=.5, color='gray', zorder=-1, alpha=1.)

    # Attach the data point in MB;
    MB.sed_wave_obs   = xbb
    MB.sed_flux_obs   = fybb * c / np.square(xbb) / d
    MB.sed_eflux_obs  = eybb * c / np.square(xbb) / d
    # Attach the best SED to MB;
    MB.sed_wave   = x1_tot
    MB.sed_flux16 = ytmp16
    MB.sed_flux50 = ytmp50
    MB.sed_flux84 = ytmp84


    if f_fancyplot:
        # For each age;
        ytmp_each50       = np.zeros(len(xm_tmp), dtype='float64')
        ytmp_each50_prior = np.zeros(len(xm_tmp), dtype='float64')
        for ss in range(len(age)):
            ii = int(len(II0) - ss - 1) # from old to young templates.
            for kk in range(len(xm_tmp[:])):
                ytmp_each50[kk] = np.percentile(ytmp_each[:,kk,ii],50)
            ax1.fill_between(x1_tot[::nstep_plot], ytmp_each50_prior[::nstep_plot], ytmp_each50[::nstep_plot], linestyle='None', lw=0.5, color=col[ii])
            ytmp_each50_prior[:] += ytmp_each50[:]


    #########################
    # Calculate non-det chi2
    # based on Sawick12
    #########################
    def func_tmp(xint,eobs,fmodel):
        int_tmp = np.exp(-0.5 * ((xint-fmodel)/eobs)**2)
        #int_tmp = np.exp(-0.5 * ((xint-fmodel))**2/fmodel)
        #int_tmp = np.exp(-0.5 * ((xint-fmodel))**2)
        return int_tmp

    if f_chind:
        conw = (wht3>0) & (ey>0) & (fy/ey>SNlim)
    else:
        conw = (wht3>0) & (ey>0) #& (fy/ey>SNlim)


    chi2 = sum((np.square(fy-ysump)*np.sqrt(wht3))[conw])

    # Effective ndim;
    '''
    ndim_eff = MB.ndim
    agemax = MB.cosmo.age(zbes).value
    for aa in range(len(MB.age)):
        if MB.age[aa]>agemax:
            ndim_eff -= 1
    '''

    con_up = (fy==0) & (ey>0) & (fy/ey<=SNlim)
    chi_nd = 0.0
    if f_chind:
        # Chi2 for non detection;
        for nn in range(len(ey[con_up])):
            #result  = integrate.quad(lambda xint: func_tmp(xint, ey[con_up][nn]/SNlim, ysump[con_up][nn]), -ey[con_up][nn]/SNlim, ey[con_up][nn]/SNlim, limit=100)
            result  = integrate.quad(lambda xint: func_tmp(xint, ey[con_up][nn], ysump[con_up][nn]), -ey[con_up][nn]*100, ey[con_up][nn], limit=100)
            chi_nd += np.log(result[0])

    # Number of degree;
    con_nod = (wht3>0) & (ey>0) #& (fy/ey>SNlim)
    nod = int(len(wht3[con_nod])-ndim_eff)

    print('\n')
    print('No-of-detection    : %d'%(len(wht3[conw])))
    print('chi2               : %.2f'%(chi2))
    if f_chind:
        print('No-of-non-detection: %d'%(len(ey[con_up])))
        print('chi2 for non-det   : %.2f'%(- 2 * chi_nd))
    print('No-of-params       : %d'%(ndim_eff))
    print('Degrees-of-freedom : %d'%(nod))
    if nod>0:
        fin_chi2 = (chi2 - 2 * chi_nd) / nod
    else:
        fin_chi2 = -99
    print('Final chi2/nu      : %.2f'%(fin_chi2))

    #
    # plot BB model from best template (blue squares)
    #
    col_dia = 'blue'
    if f_dust:
        ALLFILT = np.append(SFILT,DFILT)
        #for ii in range(len(x1_tot)):
        #    print(x1_tot[ii], model_tot[ii]*c/np.square(x1_tot[ii])/d)
        lbb, fbb, lfwhm   = filconv(ALLFILT, x1_tot, ytmp50, DIR_FILT, fw=True)
        lbb, fbb16, lfwhm = filconv(ALLFILT, x1_tot, ytmp16, DIR_FILT, fw=True)
        lbb, fbb84, lfwhm = filconv(ALLFILT, x1_tot, ytmp84, DIR_FILT, fw=True)

        # plot FIR range;
        ax3t.scatter(lbb, fbb, lw=0.5, color='none', edgecolor=col_dia, \
        zorder=2, alpha=1.0, marker='d', s=50)

    else:
        lbb, fbb, lfwhm   = filconv(SFILT, x1_tot, ytmp50, DIR_FILT, fw=True)
        lbb, fbb16, lfwhm = filconv(SFILT, x1_tot, ytmp16, DIR_FILT, fw=True)
        lbb, fbb84, lfwhm = filconv(SFILT, x1_tot, ytmp84, DIR_FILT, fw=True)

        iix = []
        for ii in range(len(fbb)):
            iix.append(np.argmin(np.abs(lbb[ii]-xbb[:])))
        con_sed = (eybb>0)
        ax1.scatter(lbb[iix][con_sed], fbb[iix][con_sed], lw=0.5, color='none', edgecolor=col_dia, zorder=3, alpha=1.0, marker='d', s=50)


        # Calculate EW, if there is excess band;
        try:
            iix2 = []
            for ii in range(len(fy_ex)):
                iix2.append(np.argmin(np.abs(lbb[:]-x_ex[ii])))

            EW50 = (fy_ex * c / np.square(x_ex) / d - fbb[iix2]) / (fbb[iix2]) * lfwhm[iix2] / (1.+zbes)
            EW16 = (fy_ex * c / np.square(x_ex) / d - fbb16[iix2]) / (fbb[iix2]) * lfwhm[iix2] / (1.+zbes)
            EW84 = (fy_ex * c / np.square(x_ex) / d - fbb84[iix2]) / (fbb[iix2]) * lfwhm[iix2] / (1.+zbes)

            EW50_er1 = ((fy_ex-ey_ex) * c / np.square(x_ex) / d - fbb[iix2]) / (fbb[iix2]) * lfwhm[iix2] / (1.+zbes)
            EW50_er2 = ((fy_ex+ey_ex) * c / np.square(x_ex) / d - fbb[iix2]) / (fbb[iix2]) * lfwhm[iix2] / (1.+zbes)

            cnt50 = fbb[iix2]
            cnt16 = fbb16[iix2]
            cnt84 = fbb84[iix2]

            ew_label = []
            for ii in range(len(fy_ex)):
                lres = MB.band['%s_lam'%MB.filts[iix2[ii]]][:]
                fres = MB.band['%s_res'%MB.filts[iix2[ii]]][:]
                ew_label.append(MB.filts[iix2[ii]])

                print('\n')
                print('EW016 for', x_ex[ii], 'is %d'%EW16[ii])
                print('EW050 for', x_ex[ii], 'is %d'%EW50[ii])
                print('EW084 for', x_ex[ii], 'is %d'%EW84[ii])
                print('%d_{-%d}^{+%d} , for sed error'%(EW50[ii],EW50[ii]-EW84[ii],EW16[ii]-EW50[ii]))
                print('Or, %d\pm{%d} , for flux error'%(EW50[ii],EW50[ii]-EW50_er1[ii]))
        except:
            pass


    if save_sed:
        # Save BB model;
        col_sed = []
        coltmp = fits.Column(name='wave', format='E', unit='AA', array=lbb)
        col_sed.append(coltmp)

        fbb16_nu = flamtonu(lbb, fbb16*1e-18, m0set=25.0)
        coltmp = fits.Column(name='fnu_16', format='E', unit='fnu(m0=25)', array=fbb16_nu)
        col_sed.append(coltmp)

        fbb_nu = flamtonu(lbb, fbb*1e-18, m0set=25.0)
        coltmp = fits.Column(name='fnu_50', format='E', unit='fnu(m0=25)', array=fbb_nu)
        col_sed.append(coltmp)

        fbb84_nu = flamtonu(lbb, fbb84*1e-18, m0set=25.0)
        coltmp = fits.Column(name='fnu_84', format='E', unit='fnu(m0=25)', array=fbb84_nu)
        col_sed.append(coltmp)

        col  = fits.ColDefs(col_sed)
        hdu0 = fits.BinTableHDU.from_columns(col)#, header=hdr)
        hdu0.writeto(ID + '_PA' + PA + '_sed.fits', overwrite=True)

        # Then save full spectrum;
        col00  = []
        col1  = fits.Column(name='wave_model', format='E', unit='AA', array=xm_tmp)
        col00.append(col1)
        col2  = fits.Column(name='f_model_16', format='E', unit='1e-18erg/s/cm2/AA', array=ytmp16[:])
        col00.append(col2)
        col3  = fits.Column(name='f_model_50', format='E', unit='1e-18erg/s/cm2/AA', array=ytmp50[:])
        col00.append(col3)
        col4  = fits.Column(name='f_model_84', format='E', unit='1e-18erg/s/cm2/AA', array=ytmp84[:])
        col00.append(col4)
        col5  = fits.Column(name='wave_obs', format='E', unit='AA', array=xbb)
        col00.append(col5)
        col6  = fits.Column(name='f_obs', format='E', unit='1e-18erg/s/cm2/AA', array=fybb[:] * c / np.square(xbb[:]) / d)
        col00.append(col6)
        col7  = fits.Column(name='e_obs', format='E', unit='1e-18erg/s/cm2/AA', array=eybb[:] * c / np.square(xbb[:]) / d)
        col00.append(col7)

        hdr = fits.Header()
        hdr['redshift'] = zbes
        hdr['id'] = ID

        # Chi square:
        hdr['chi2']     = chi2
        hdr['hierarch No-of-effective-data-points'] = len(wht3[conw])
        hdr['hierarch No-of-nondetectioin'] = len(ey[con_up])
        hdr['hierarch Chi2-of-nondetection'] = chi_nd
        hdr['hierarch No-of-params']  = ndim_eff
        hdr['hierarch Degree-of-freedom']  = nod
        hdr['hierarch reduced-chi2']  = fin_chi2

        # Muv
        MUV = -2.5 * np.log10(Fuv[:]) + 25.0
        hdr['MUV16'] = np.percentile(MUV[:],16)
        hdr['MUV50'] = np.percentile(MUV[:],50)
        hdr['MUV84'] = np.percentile(MUV[:],84)

        # Fuv (!= flux of Muv)
        hdr['FUV16'] = np.percentile(Fuv28[:],16)
        hdr['FUV50'] = np.percentile(Fuv28[:],50)
        hdr['FUV84'] = np.percentile(Fuv28[:],84)

        # LIR
        hdr['LIR16'] = np.percentile(Lir[:],16)
        hdr['LIR50'] = np.percentile(Lir[:],50)
        hdr['LIR84'] = np.percentile(Lir[:],84)

        # UVJ
        try:
            hdr['uv16'] = np.percentile(UVJ[:,0],16)
            hdr['uv50'] = np.percentile(UVJ[:,0],50)
            hdr['uv84'] = np.percentile(UVJ[:,0],84)
            hdr['bv16'] = np.percentile(UVJ[:,1],16)
            hdr['bv50'] = np.percentile(UVJ[:,1],50)
            hdr['bv84'] = np.percentile(UVJ[:,1],84)
            hdr['vj16'] = np.percentile(UVJ[:,2],16)
            hdr['vj50'] = np.percentile(UVJ[:,2],50)
            hdr['vj84'] = np.percentile(UVJ[:,2],84)
            hdr['zj16'] = np.percentile(UVJ[:,3],16)
            hdr['zj50'] = np.percentile(UVJ[:,3],50)
            hdr['zj84'] = np.percentile(UVJ[:,3],84)
        except:
            print('\nError when writinf UVJ colors;\n')
            #print(np.percentile(UVJ[:,0],[16,50,84]))
            pass

        # EW;
        try:
            for ii in range(len(EW50)):
                hdr['EW_%s_16'%(ew_label[ii])] = EW16[ii]
                hdr['EW_%s_50'%(ew_label[ii])] = EW50[ii]
                hdr['EW_%s_84'%(ew_label[ii])] = EW84[ii]
                hdr['EW_%s_e1'%(ew_label[ii])] = EW50_er1[ii]
                hdr['EW_%s_e2'%(ew_label[ii])] = EW50_er2[ii]
                hdr['HIERARCH cnt_%s_16'%(ew_label[ii])]= cnt16[ii]
                hdr['HIERARCH cnt_%s_50'%(ew_label[ii])]= cnt50[ii]
                hdr['HIERARCH cnt_%s_84'%(ew_label[ii])]= cnt84[ii]
        except:
            pass

        # Write;
        colspec = fits.ColDefs(col00)
        hdu0    = fits.BinTableHDU.from_columns(colspec, header=hdr)
        #hdu0.writeto(DIR_TMP + 'gsf_spec_%s.fits'%(ID), overwrite=True)
        hdu0.writeto('gsf_spec_%s.fits'%(ID), overwrite=True)


    #
    # SED params in plot
    #
    if f_label:
        try:
        #if True:
            fd = fits.open('SFH_' + ID + '_PA' + PA + '_param.fits')[1].data
            ax1.text(2300, ymax*0.32,\
            'ID: %s\n$z_\mathrm{obs.}:%.2f$\n$\log M_\mathrm{*}/M_\odot:%.2f$\n$\log Z_\mathrm{*}/Z_\odot:%.2f$\n$\log T_\mathrm{*}$/Gyr$:%.2f$\n$A_V$/mag$:%.2f$\n$\\chi^2/\\nu:%.2f$'\
            %(ID, zbes, fd['Mstel'][1], fd['Z_MW'][1], fd['T_MW'][1], fd['AV'][1], fin_chi2),\
            fontsize=9)
        except:
            print('\nFile is missing : _param.fits\n')
            pass

    #######################################
    ax1.xaxis.labelpad = -3
    if f_grsm:
        ax2t.set_xlabel('RF wavelength ($\mathrm{\mu m}$)')
        ax2t.set_xlim(3600, 5400)
        conaa = (x0/zscl>3300) & (x0/zscl<6000)
        ymaxzoom = np.max(ysum[conaa]*c/np.square(x0[conaa])/d) * 1.2
        yminzoom = np.min(ysum[conaa]*c/np.square(x0[conaa])/d) / 1.2
        ax2t.set_ylim(yminzoom, ymaxzoom)

        ax2t.xaxis.labelpad = -2
        ax2t.set_yticklabels(())
        ax2t.set_xticks([4000, 5000])
        ax2t.set_xticklabels(['0.4', '0.5'])

    if f_dust:
        try:
            contmp = (x1_tot>10*1e4) #& (fybbd/eybbd>SNlim)
            #y3min, y3max = -.2*np.max(fybbd[contmp]*c/np.square(xbbd[contmp])/d), np.max((fybbd)[contmp]*c/np.square(xbbd[contmp])/d)*1.1
            y3min, y3max = -.2*np.max((model_tot * c/ np.square(x1_tot) / d)[contmp]), np.max((model_tot * c/ np.square(x1_tot) / d)[contmp])*1.1
            ax3t.set_ylim(y3min, y3max)
        except:
            if verbose:
                print('y3 limit is not specified.')
            pass
        ax3t.set_xlim(1e5, 2e7)
        ax3t.set_xscale('log')
        ax3t.set_xticks([100000, 1000000, 10000000])
        ax3t.set_xticklabels(['10', '100', '1000'])

    ###############
    # Line name
    ###############
    LN0 = ['Mg2', '$NeIV$', '[OII]', 'H$\theta$', 'H$\eta$', 'Ne3?', 'H$\delta$', 'H$\gamma$', 'H$\\beta$', 'O3', 'O3', 'Mgb', 'Halpha', 'S2L', 'S2H']
    LW0 = [2800, 3347, 3727, 3799, 3836, 3869, 4102, 4341, 4861, 4959, 5007, 5175, 6563, 6717, 6731]
    fsl = 9 # Fontsize for line
    if f_grsm:
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
    ## Save
    ####################
    ax1.legend(loc=1, fontsize=11)
    if figpdf:
        plt.savefig('SPEC_' + ID + '_PA' + PA + '_spec.pdf', dpi=300)
    else:
        plt.savefig('SPEC_' + ID + '_PA' + PA + '_spec.png', dpi=300)


def plot_corner_TZ(ID, PA, Zall=np.arange(-1.2,0.4249,0.05), age=[0.01, 0.1, 0.3, 0.7, 1.0, 3.0]):
    '''
    '''
    import matplotlib
    import matplotlib.cm as cm
    col = ['violet', 'indigo', 'b', 'lightblue', 'lightgreen', 'g', 'orange', 'coral', 'r', 'darkred']#, 'k']
    nage = np.arange(0,len(age),1)
    fnc  = Func(ID, PA, Zall, age, dust_model=dust_model) # Set up the number of Age/ZZ
    bfnc = Basic(Zall)

    fig = plt.figure(figsize=(3,3))
    fig.subplots_adjust(top=0.96, bottom=0.14, left=0.2, right=0.96, hspace=0.15, wspace=0.25)
    ax1 = fig.add_subplot(111)

    DIR_TMP = './templates/'
    ####################
    # MCMC corner plot.
    ####################
    file = 'chain_' + ID + '_PA' + PA + '_corner.cpkl'
    niter = 0
    data = loadcpkl(os.path.join('./'+file))

    try:
        ndim   = data['ndim']     # By default, use ndim and burnin values contained in the cpkl file, if present.
        burnin = data['burnin']
        nmc    = data['niter']
        nwalk  = data['nwalkers']
        Nburn  = burnin #*20
        samples = data['chain'][:]
    except:
        if verbose: print(' =   >   NO keys of ndim and burnin found in cpkl, use input keyword values')


    f0     = fits.open(DIR_TMP + 'ms_' + ID + '_PA' + PA + '.fits')
    sedpar = f0[1]

    getcmap   = matplotlib.cm.get_cmap('jet')
    nc        = np.arange(0, nmc, 1)
    col = getcmap((nc-0)/(nmc-0))

    #for kk in range(0,nmc,1):
    Ntmp = np.zeros(nmc, dtype='float64')
    Avtmp= np.zeros(nmc, dtype='float64')
    Ztmp = np.zeros(nmc, dtype='float64')
    Ttmp = np.zeros(nmc, dtype='float64')
    ACtmp= np.zeros(nmc, dtype='float64')


    for kk in range(0,5000,1):
        #nr = kk # np.random.randint(len(samples))
        nr = np.random.randint(len(samples))

        Avtmp[kk] = samples['Av'][nr]
        for ss in range(len(age)):
            AA_tmp = samples['A'+str(ss)][nr]
            ZZ_tmp = samples['Z'+str(ss)][nr]

            nZtmp  = bfnc.Z2NZ(ZZ_tmp)
            mslist = sedpar.data['ML_'+str(nZtmp)][ss]

            Ztmp[kk]  += (10 ** ZZ_tmp) * AA_tmp * mslist
            Ttmp[kk]  += age[ss] * AA_tmp * mslist
            ACtmp[kk] += AA_tmp * mslist

        Ztmp[kk] /= ACtmp[kk]
        Ttmp[kk] /= ACtmp[kk]
        Ntmp[kk]  = kk


    x1min, x1max = np.log10(0.1), np.log10(4) #np.max(A50/Asum)+0.1
    y1min, y1max = -0.8,.6 #np.max(A50/Asum)+0.1

    xbins = np.arange(x1min, x1max, 0.01)
    ybins = np.arange(y1min, y1max, 0.01)

    xycounts,_,_ = np.histogram2d(np.log10(Ttmp), np.log10(Ztmp), bins=[xbins,ybins])
    ax1.contour(xbins[:-1], ybins[:-1], xycounts.T, 4, linewidths=np.arange(.5, 4, 0.1), colors='k')

    ax1.scatter(np.log10(Ttmp), np.log10(Ztmp), c='r', s=1, marker='.', alpha=0.1)
    #ax2.scatter(np.log10(Ttmp), np.log10(Avtmp), c='r', s=1, marker='.', alpha=0.1)
    #ax3.scatter(np.log10(Ztmp), np.log10(Avtmp), c='r', s=1, marker='.', alpha=0.1)

    ax1.set_xlabel('$\log T_*$/Gyr', fontsize=12)
    ax1.set_ylabel('$\log Z_*/Z_\odot$', fontsize=12)

    #ax2.set_xlabel('$\log T_*$/Gyr', fontsize=12)
    #ax2.set_ylabel('$A_V$/mag', fontsize=12)

    #ax3.set_xlabel('$\log Z_*/Z_\odot$', fontsize=12)
    #ax3.set_ylabel('$A_V$/mag', fontsize=12)

    ax1.set_xlim(x1min, x1max)
    ax1.set_ylim(y1min, y1max)

    ax1.yaxis.labelpad = -1

    plt.savefig('TZ_' + ID + '_PA' + PA + '_corner.pdf')
    plt.close()


def plot_corner_physparam_cum_frame(ID, PA, Zall=np.arange(-1.2,0.4249,0.05), age=[0.01, 0.1, 0.3, 0.7, 1.0, 3.0], tau0=[0.1,0.2,0.3], fig=None, dust_model=0, out_ind=0, snlimbb=1.0, DIR_OUT='./'):
    '''
    # Creat "cumulative" png for gif image.
    #
    #
    # If you like to
    # Creat temporal png for gif image.
    #
    # snlimbb: SN limit to show flux or up lim in SED.
    #
    '''
    col = ['violet', 'indigo', 'b', 'lightblue', 'lightgreen', 'g', 'orange', 'coral', 'r', 'darkred']#, 'k']
    nage = np.arange(0,len(age),1)
    fnc  = Func(ID, PA, Zall, age, dust_model=dust_model) # Set up the number of Age/ZZ
    bfnc = Basic(Zall)

    ###########################
    # Open result file
    ###########################
    # Open ascii file and stock to array.
    lib     = fnc.open_spec_fits(ID, PA, fall=0)
    lib_all = fnc.open_spec_fits(ID, PA, fall=1)

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

    # Repeat no.
    nplot = 1000
    #DIR_OUT = '/astro/udfcen3/Takahiro/sedfitter/corner/' + ID + '_corner/'
    try:
        os.makedirs(DIR_OUT)
    except:
        pass

    # plot Configuration
    K = 4 # No of params.
    Par = ['$\log M_*/M_\odot$', '$\log T$/Gyr', '$A_V$/mag', '$\log Z / Z_\odot$']
    factor = 2.0           # size of one side of one panel
    lbdim = 0.5 * factor   # size of left/bottom margin
    trdim = 0.2 * factor   # size of top/right margin
    whspace = 0.02         # w/hspace size
    plotdim = factor * K + factor * (K - 1.) * whspace
    dim = lbdim + plotdim + trdim
    sclfig = 0.7

    # Create a new figure if one wasn't provided.
    if fig is None:
        fig, axes = plt.subplots(K, K, figsize=(dim*sclfig, dim*sclfig))
    else:
        try:
            axes = np.array(fig.axes).reshape((K, K))
        except:
            raise ValueError("Provided figure has {0} axes, but data has "
                             "dimensions K={1}".format(len(fig.axes), K))
    # Format the figure.
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=lb* 1.06, bottom=lb*.9, right=tr, top=tr*.99,
                        wspace=whspace, hspace=whspace)

    # For spec plot
    ax0 = fig.add_axes([0.62,0.61,0.37,0.33])
    ###############################
    # Data taken from
    ###############################
    DIR_TMP = './templates/'
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

    conspec = (NR<10000) #& (fy/ey>1)
    #ax0.plot(xg0, fg0 * c / np.square(xg0) / d, marker='', linestyle='-', linewidth=0.5, ms=0.1, color='royalblue', label='')
    #ax0.plot(xg1, fg1 * c / np.square(xg1) / d, marker='', linestyle='-', linewidth=0.5, ms=0.1, color='#DF4E00', label='')
    conbb = (fybb/eybb>snlimbb)
    ax0.errorbar(xbb[conbb], fybb[conbb] * c / np.square(xbb[conbb]) / d, yerr=eybb[conbb]*c/np.square(xbb[conbb])/d, color='k', linestyle='', linewidth=0.5, zorder=4)
    ax0.plot(xbb[conbb], fybb[conbb] * c / np.square(xbb[conbb]) / d, '.r', ms=10, linestyle='', linewidth=0, zorder=4)

    conbbe = (fybb/eybb<snlimbb)
    ax0.plot(xbb[conbbe], eybb[conbbe] * c / np.square(xbb[conbbe]) / d, 'vr', ms=10, linestyle='', linewidth=0, zorder=4)

    ymax = np.max(fybb[conbb] * c / np.square(xbb[conbb]) / d) * 1.10
    ax0.set_xlabel('Observed wavelength ($\mathrm{\mu m}$)', fontsize=14)
    ax0.set_ylabel('Flux ($\mathrm{erg}/\mathrm{s}/\mathrm{cm}^{2}/\mathrm{\AA}$)', fontsize=13)
    ax0.set_xlim(2200, 88000)
    #ax1.set_xlim(12500, 16000)
    ax0.set_xscale('log')
    ax0.set_ylim(-0.05, ymax)


    DIR_TMP = './templates/'
    ####################
    # MCMC corner plot.
    ####################
    file = 'chain_' + ID + '_PA' + PA + '_corner.cpkl'
    niter = 0
    data = loadcpkl(os.path.join('./'+file))

    try:
        ndim   = data['ndim']     # By default, use ndim and burnin values contained in the cpkl file, if present.
        burnin = data['burnin']
        nmc    = data['niter']
        nwalk  = data['nwalkers']
        Nburn  = burnin #*20
        samples = data['chain'][:]
    except:
        if verbose: print(' =   >   NO keys of ndim and burnin found in cpkl, use input keyword values')

    f0     = fits.open(DIR_TMP + 'ms_' + ID + '_PA' + PA + '.fits')
    sedpar = f0[1]

    import matplotlib
    import matplotlib.cm as cm
    getcmap   = matplotlib.cm.get_cmap('jet')
    nc        = np.arange(0, nmc, 1)
    col = getcmap((nc-0)/(nmc-0))

    #for kk in range(0,nmc,1):
    Ntmp = np.zeros(nplot, dtype='float64')
    lmtmp= np.zeros(nplot, dtype='float64')
    Avtmp= np.zeros(nplot, dtype='float64')
    Ztmp = np.zeros(nplot, dtype='float64')
    Ttmp = np.zeros(nplot, dtype='float64')
    ACtmp= np.zeros(nplot, dtype='float64')

    files = [] # For movie
    for kk in range(0,nplot,1):

        #nr = kk # np.random.randint(len(samples))
        nr = np.random.randint(len(samples))
        Avtmp[kk] = samples['Av'][nr]
        #Asum = 0
        #for ss in range(len(age)):
        #Asum += np.sum(samples['A'+str(ss)][nr])
        II0   = nage #[0,1,2,3] # Number for templates
        for ss in range(len(age)):
            ii = int(len(II0) - ss - 1) # from old to young templates.
            AA_tmp = samples['A'+str(ii)][nr]
            try:
                ZZ_tmp = samples['Z'+str(ii)][nr]
            except:
                ZZ_tmp = samples['Z0'][nr]
            nZtmp      = bfnc.Z2NZ(ZZ_tmp)
            mslist     = sedpar.data['ML_'+str(nZtmp)][ii]
            lmtmp[kk] += AA_tmp * mslist
            Ztmp[kk]  += (10 ** ZZ_tmp) * AA_tmp * mslist
            Ttmp[kk]  += age[ii] * AA_tmp * mslist
            ACtmp[kk] += AA_tmp * mslist

            # SED
            flim = 0.05
            if ss == 0:
                y0, x0   = fnc.tmp03(AA_tmp, Avtmp[kk], ii, ZZ_tmp, zbes, lib_all, tau0=tau0)
                y0p, x0p = fnc.tmp03(AA_tmp, Avtmp[kk], ii, ZZ_tmp, zbes, lib, tau0=tau0)
                ysump = y0p #* 1e18
                ysum  = y0  #* 1e18
                if AA_tmp/Asum > flim:
                    ax0.plot(x0, y0 * c/ np.square(x0) / d, '--', lw=0.1, color=col[ii], zorder=-1, label='', alpha=0.1)
            else:
                y0_r, x0_tmp = fnc.tmp03(AA_tmp, Avtmp[kk], ii, ZZ_tmp, zbes, lib_all, tau0=tau0)
                y0p, x0p     = fnc.tmp03(AA_tmp, Avtmp[kk], ii, ZZ_tmp, zbes, lib, tau0=tau0)
                ysump += y0p  #* 1e18
                ysum  += y0_r #* 1e18
                if AA_tmp/Asum > flim:
                    ax0.plot(x0, y0_r * c/ np.square(x0) / d, '--', lw=0.1, color=col[ii], zorder=-1, label='', alpha=0.1)
        # Total
        ax0.plot(x0, ysum * c/ np.square(x0) / d, '-', lw=0.1, color='gray', zorder=-1, label='', alpha=0.1)
        ax0.set_xlim(2200, 88000)
        ax0.set_xscale('log')
        ax0.set_ylim(0., ymax)

        # Convert into log
        Ztmp[kk] /= ACtmp[kk]
        Ttmp[kk] /= ACtmp[kk]
        Ntmp[kk]  = kk

        lmtmp[kk] = np.log10(lmtmp[kk])
        Ztmp[kk]  = np.log10(Ztmp[kk])
        Ttmp[kk]  = np.log10(Ttmp[kk])


        NPAR    = [lmtmp[:kk+1], Ttmp[:kk+1], Avtmp[:kk+1], Ztmp[:kk+1]]
        #NPARmin = [np.log10(M16)-0.1, -0.4, 0, -0.6]
        #NPARmax = [np.log10(M84)+0.1, 0.5, 2., 0.5]
        NPARmin = [np.log10(M16)-0.1, -0.4, Av16-0.1, -0.5]
        NPARmax = [np.log10(M84)+0.1, 0.5, Av84+0.1, 0.5]

        #for kk in range(0,nplot,1):
        if kk == nplot-1:
            # Histogram
            for i, x in enumerate(Par):
                ax = axes[i, i]
                x1min, x1max = NPARmin[i], NPARmax[i]
                nbin = 50
                binwidth1 = (x1max-x1min)/nbin
                bins1 = np.arange(x1min, x1max + binwidth1, binwidth1)
                ax.hist(NPAR[i], bins=bins1, orientation='vertical', color='b', histtype='stepfilled', alpha=0.6)
                ax.set_xlim(x1min, x1max)
                #print(x, x1min, x1max)
                #ax2.scatter(np.log10(Ttmp), np.log10(Avtmp), c='r', s=1, marker='.', alpha=0.1)
                #ax3.scatter(np.log10(Ztmp), np.log10(Avtmp), c='r', s=1, marker='.', alpha=0.1)
                #ax.set_xlabel('$\log T_*$/Gyr', fontsize=12)
                #ax.set_ylabel('$\log Z_*/Z_\odot$', fontsize=12)
                ax.set_yticklabels([])
                #ax.set_xticklabels([])
                #ax.set_title('%s'%(Par[i]), fontsize=12)
                if i == K-1:
                    ax.set_xlabel('%s'%(Par[i]), fontsize=12)
                if i < K-1:
                    ax.set_xticklabels([])

        # Scatter and contour
        for i, x in enumerate(Par):
            for j, y in enumerate(Par):
                #print(i,j,Par[j], Par[i])
                if i > j:
                    ax = axes[i, j]
                    ax.scatter(NPAR[j], NPAR[i], c='b', s=1, marker='o', alpha=0.01)
                    ax.set_xlabel('%s'%(Par[j]), fontsize=12)

                    #x1min, x1max = np.min(NPAR[j]), np.max(NPAR[j])
                    #y1min, y1max = np.min(NPAR[i]), np.max(NPAR[i])
                    x1min, x1max = NPARmin[j], NPARmax[j]
                    y1min, y1max = NPARmin[i], NPARmax[i]
                    ax.set_xlim(x1min, x1max)
                    ax.set_ylim(y1min, y1max)

                    if j==0:
                        ax.set_ylabel('%s'%(Par[i]), fontsize=12)
                    if j>0:
                        ax.set_yticklabels([])
                    if i<K-1:
                        ax.set_xticklabels([])

                if i < j:
                    ax = axes[i, j]
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_frame_on(False)
                    ax.set_xticks([])
                    ax.set_yticks([])

                if i == j:
                    ax = axes[i, j]
                    ax.set_yticklabels([])
                    if i == K-1:
                        ax.set_xlabel('%s'%(Par[i]), fontsize=12)
                    if i < K-1:
                        ax.set_xticklabels([])


        if kk%10 == 0 and out_ind == 1:
            fname = DIR_OUT + '%d.png' % kk
            print('Saving frame', fname)
            plt.savefig(fname, dpi=200)
            files.append(fname)

        #plt.savefig(DIR_OUT + '%d.pdf'%(kk))

    plt.savefig(DIR_OUT + 'param_' + ID + '_PA' + PA + '_corner.png', dpi=200)
    plt.close()


def write_lines(ID, PA, zbes, R_grs=45, dw=4, umag=1.0):
    '''
    '''
    dlw   = R_grs * dw # Can affect the SFR.
    ldw   = 7

    ###################################
    # To add lines in the plot,
    # ,manually edit the following file
    # so as Fcont50 have >0.
    ###################################
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

                    if f_grsm:
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


def plot_corner_physparam_summary(MB, fig=None, out_ind=0, DIR_OUT='./', nplot=1000):
    '''
    Purpose:
    ========
    For summary. In the same format as plot_corner_physparam_frame.

    '''

    col = ['violet', 'indigo', 'b', 'lightblue', 'lightgreen', 'g', 'orange', 'coral', 'r', 'darkred']#, 'k']
    import matplotlib
    import matplotlib.cm as cm
    import scipy.stats as stats

    nage = MB.nage #np.arange(0,len(age),1)
    fnc  = MB.fnc #Func(ID, PA, Z, nage, dust_model=dust_model, DIR_TMP=DIR_TMP) # Set up the number of Age/ZZ
    bfnc = MB.bfnc #Basic(Z)
    ID   = MB.ID
    PA   = MB.PA
    Z    = MB.Zall
    try:
        age = MB.age_fix
    except:
        age  = MB.age

    tau0 = MB.tau0 #[0.1,0.2,0.3]
    dust_model = MB.dust_model
    DIR_TMP = MB.DIR_TMP

    #Txmax = 4 # Max x value
    Txmax = np.max(age) + 1.0 # Max x value

    ###########################
    # Open result file
    ###########################
    lib     = fnc.open_spec_fits(MB,fall=0)
    lib_all = fnc.open_spec_fits(MB,fall=1)

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

    Asum = np.sum(A50)
    aa   = 0

    Av16 = hdul[1].data['Av'+str(aa)][0]
    Av50 = hdul[1].data['Av'+str(aa)][1]
    Av84 = hdul[1].data['Av'+str(aa)][2]

    Z50  = np.zeros(len(age), dtype='float64')
    Z16  = np.zeros(len(age), dtype='float64')
    Z84  = np.zeros(len(age), dtype='float64')

    NZbest = np.zeros(len(age), dtype='int')
    for aa in range(len(age)):
        Z50[aa] = hdul[1].data['Z'+str(aa)][1]
        Z16[aa] = hdul[1].data['Z'+str(aa)][0]
        Z84[aa] = hdul[1].data['Z'+str(aa)][2]
        NZbest[aa]= bfnc.Z2NZ(Z50[aa])

    ZZ50  = np.sum(Z50*A50)/np.sum(A50) # Light weighted Z.
    chi   = hdul[1].data['chi'][0]
    chin  = hdul[1].data['chi'][1]
    fitc  = chin
    Cz0   = hdul[0].header['Cz0']
    Cz1   = hdul[0].header['Cz1']
    zbes  = hdul[0].header['z']
    zscl  = (1.+zbes)

    try:
        os.makedirs(DIR_OUT)
    except:
        pass

    # plot Configuration
    Par = ['$\log M_*/M_\odot$', '$\log T_*$/Gyr', '$A_V$/mag', '$\log Z_* / Z_\odot$']
    K   = len(Par) # No of params.
    factor = 2.0           # size of one side of one panel
    lbdim  = 0.5 * factor   # size of left/bottom margin
    trdim  = 0.2 * factor   # size of top/right margin
    whspace= 0.02         # w/hspace size
    plotdim= factor * K + factor * (K - 1.) * whspace
    dim    = lbdim + plotdim + trdim
    sclfig = 0.7


    # Format the figure.
    fig, axes = plt.subplots(K, K, figsize=(dim*sclfig*2, dim*sclfig))
    # Format the figure.
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    #fig.subplots_adjust(left=lb*1.06, bottom=lb*.9, right=tr, top=tr*.99,
    fig.subplots_adjust(left=0.5, bottom=lb*.9, right=tr, top=tr*.99,
                        wspace=whspace, hspace=whspace)

    # For spec plot
    ax0 = fig.add_axes([0.05,0.73,0.37,0.23])
    ax1 = fig.add_axes([0.05,0.40,0.37,0.23])
    ax2 = fig.add_axes([0.05,0.07,0.37,0.23])

    if MB.f_dust:
        MB.dict = MB.read_data(MB.Cz0, MB.Cz1, MB.zgal, add_fir=True)
    else:
        MB.dict = MB.read_data(MB.Cz0, MB.Cz1, MB.zgal)

    # Get data points;
    NRbb   = MB.dict['NRbb']
    xbb    = MB.dict['xbb'] 
    fybb   = MB.dict['fybb']
    eybb   = MB.dict['eybb']
    exbb   = MB.dict['exbb']
    snbb   = fybb/eybb

    # Get spec data points;
    dat = np.loadtxt(DIR_TMP + 'spec_obs_' + ID + '_PA' + PA + '.cat', comments='#')
    NR  = dat[:, 0]
    x   = dat[:, 1]
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
    ey01 = np.append(eg0,eg1)

    fy   = np.append(fy01,fg2)
    ey   = np.append(ey01,eg2)
    wht=1./np.square(ey)

    conspec = (NR<10000) #& (fy/ey>1)
    conbb   = (fybb/eybb>1)
    ax0.errorbar(xbb[conbb], fybb[conbb] * c / np.square(xbb[conbb]) / d, yerr=eybb[conbb]*c/np.square(xbb[conbb])/d, color='k', linestyle='', linewidth=0.5, zorder=4)
    ax0.plot(xbb[conbb], fybb[conbb] * c / np.square(xbb[conbb]) / d, '.r', ms=10, linestyle='', linewidth=0, zorder=4)#, label='Obs.(BB)')

    ####################
    # MCMC corner plot.
    ####################
    file  = 'chain_' + ID + '_PA' + PA + '_corner.cpkl'
    niter = 0
    data  = loadcpkl(os.path.join('./'+file))

    try:
        ndim   = data['ndim']     # By default, use ndim and burnin values contained in the cpkl file, if present.
        burnin = data['burnin']
        nmc    = data['niter']
        nwalk  = data['nwalkers']
        Nburn  = burnin #*20
        samples = data['chain'][:]
    except:
        if verbose: print(' =   >   NO keys of ndim and burnin found in cpkl, use input keyword values')

    f0     = fits.open(DIR_TMP + 'ms_' + ID + '_PA' + PA + '.fits')
    sedpar = f0[1]

    getcmap   = matplotlib.cm.get_cmap('jet')
    nc        = np.arange(0, nmc, 1)
    col = getcmap((nc-0)/(nmc-0))

    #for kk in range(0,nmc,1):
    Ntmp = np.zeros(nplot, dtype='float64')
    lmtmp= np.zeros(nplot, dtype='float64')
    Avtmp= np.zeros(nplot, dtype='float64')
    Ztmp = np.zeros(nplot, dtype='float64')
    Ttmp = np.zeros(nplot, dtype='float64')
    ACtmp= np.zeros(nplot, dtype='float64')

    # Time bin
    Tuni = MB.cosmo.age(zbes).value
    Tuni0 = (Tuni - age[:])
    delT  = np.zeros(len(age),dtype='float64')
    delTl = np.zeros(len(age),dtype='float64')
    delTu = np.zeros(len(age),dtype='float64')

    if len(age) == 1:
        for aa in range(len(age)):
            try:
                tau_ssp = float(inputs['TAU_SSP'])
            except:
                tau_ssp = 0.01
            delTl[aa] = tau_ssp/2
            delTu[aa] = tau_ssp/2
            delT[aa]  = delTu[aa] + delTl[aa]
    else:
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

    ######
    files = [] # For gif animation
    SFmax = 0
    Tsmin = 0
    Tsmax = 0
    Zsmin = 0
    Zsmax = 0
    AMtmp = 0
    AMtmp16 = 0
    AMtmp84 = 0
    for ii in range(len(age)):
        ZZ_tmp = Z50[ii] #samples['Z'+str(ii)][100]
        ZZ_tmp16 = Z16[ii] #samples['Z'+str(ii)][100]
        ZZ_tmp84 = Z84[ii] #samples['Z'+str(ii)][100]
        AA_tmp = np.max(samples['A'+str(ii)][:])
        AA_tmp84 = np.percentile(samples['A'+str(ii)][:],95)
        AA_tmp16 = np.percentile(samples['A'+str(ii)][:],5)
        nZtmp  = bfnc.Z2NZ(ZZ_tmp)
        mslist = sedpar.data['ML_'+str(nZtmp)][ii]
        AMtmp16 += mslist*AA_tmp16
        AMtmp84 += mslist*AA_tmp84

        Tsmax += age[ii] * AA_tmp84 * mslist
        Tsmin += age[ii] * AA_tmp16 * mslist

        Zsmax += 10**ZZ_tmp84 * AA_tmp84 * mslist
        Zsmin += 10**ZZ_tmp16 * AA_tmp16 * mslist

        SFtmp  = AA_tmp * mslist / delT[ii]
        if SFtmp > SFmax:
            SFmax = SFtmp

    delM = np.log10(M84) - np.log10(M16)
    NPARmin = [np.log10(M16)-.1, np.log10(Tsmin/AMtmp16)-0.1, Av16-0.1, np.log10(Zsmin/AMtmp16)-0.2]
    NPARmax = [np.log10(M84)+.1, np.log10(Tsmax/AMtmp84)+0.2, Av84+0.1, np.log10(Zsmax/AMtmp84)+0.2]

    # For redshift
    if zbes<2:
        zred  = [zbes, 2, 3, 6]
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
    else:
        zred  = [zbes, 12]
        zredl = ['$z_\mathrm{obs.}$', 12]

    Tzz   = np.zeros(len(zred), dtype='float64')
    for zz in range(len(zred)):
        Tzz[zz] = (Tuni - MB.cosmo.age(zred[zz]).value) #/ cc.Gyr_s
        if Tzz[zz] < 0.01:
            Tzz[zz] = 0.01


    def density_estimation(m1, m2):
        xmin, xmax = np.min(m1), np.max(m1)
        ymin, ymax = np.min(m2), np.max(m2)
        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([m1, m2])
        kernel = stats.gaussian_kde(values)
        Z = np.reshape(kernel(positions).T, X.shape)
        return X, Y, Z


    for kk in range(0,nplot,1):
        #nr = kk # np.random.randint(len(samples))
        nr = np.random.randint(len(samples))
        Avtmp[kk] = samples['Av'][nr]
        #Asum = 0
        #for ss in range(len(age)):
        #Asum += np.sum(samples['A'+str(ss)][nr])
        ZMM = np.zeros((len(age)), dtype='float64') # Mass weighted Z.
        ZM  = np.zeros((len(age)), dtype='float64') # Light weighted T.
        ZC  = np.zeros((len(age)), dtype='float64') # Light weighted T.
        SF  = np.zeros((len(age)), dtype='float64') # SFR
        AM  = np.zeros((len(age)), dtype='float64') # Light weighted T.


        II0   = nage #[0,1,2,3] # Number for templates
        for ss in range(len(age)):
            ii = int(len(II0) - ss - 1) # from old to young templates.
            AA_tmp = samples['A'+str(ii)][nr]
            try:
                ZZ_tmp = samples['Z'+str(ii)][nr]
            except:
                ZZ_tmp = samples['Z0'][nr]

            nZtmp      = bfnc.Z2NZ(ZZ_tmp)
            mslist     = sedpar.data['ML_'+str(nZtmp)][ii]
            lmtmp[kk] += AA_tmp * mslist
            Ztmp[kk]  += (10 ** ZZ_tmp) * AA_tmp * mslist
            Ttmp[kk]  += age[ii] * AA_tmp * mslist
            ACtmp[kk] += AA_tmp * mslist

            AM[ii] = AA_tmp * mslist
            SF[ii] = AA_tmp * mslist / delT[ii]
            ZM[ii] = ZZ_tmp # AAtmp[aa] * mslist[aa]
            ZMM[ii]= (10 ** ZZ_tmp) * AA_tmp * mslist

            # SED
            flim = 0.05
            if ss == 0:
                y0, x0   = fnc.tmp03(AA_tmp, Avtmp[kk], ii, ZZ_tmp, zbes, lib_all)
                y0p, x0p = fnc.tmp03(AA_tmp, Avtmp[kk], ii, ZZ_tmp, zbes, lib)
                ysump = y0p #* 1e18
                ysum  = y0  #* 1e18
                if AA_tmp/Asum > flim:
                    ax0.plot(x0, y0 * c/ np.square(x0) / d, '--', lw=.1, color=col[ii], zorder=-1, label='', alpha=0.1)
            else:
                y0_r, x0_tmp = fnc.tmp03(AA_tmp, Avtmp[kk], ii, ZZ_tmp, zbes, lib_all)
                y0p, x0p     = fnc.tmp03(AA_tmp, Avtmp[kk], ii, ZZ_tmp, zbes, lib)
                ysump += y0p  #* 1e18
                ysum  += y0_r #* 1e18
                if AA_tmp/Asum > flim:
                    ax0.plot(x0, y0_r * c/ np.square(x0) / d, '--', lw=.1, color=col[ii], zorder=-1, label='', alpha=0.1)

        for ss in range(len(age)):
            ii = ss # from old to young templates.
            AC = np.sum(AM[ss:])
            ZC[ss] = np.log10(np.sum(ZMM[ss:])/AC)

        # Total
        ymax = np.max(fybb[conbb] * c / np.square(xbb[conbb]) / d) * 1.10
        ax0.plot(x0, ysum * c/ np.square(x0) / d, '-', lw=.1, color='gray', zorder=-1, label='', alpha=0.1)
        if len(age)==1:
            ax1.plot(age[:], SF[:], marker='.', linestyle='-', lw=.1, color='k', zorder=-1, label='', alpha=0.01)
            ax2.plot(age[:], ZC[:], marker='.', linestyle='-', lw=.1, color='k', zorder=-1, label='', alpha=0.01)
        else:
            ax1.plot(age[:], SF[:], marker='', linestyle='-', lw=.1, color='k', zorder=-1, label='', alpha=0.1)
            ax2.plot(age[:], ZC[:], marker='', linestyle='-', lw=.1, color='k', zorder=-1, label='', alpha=0.1)

        # Convert into log
        Ztmp[kk] /= ACtmp[kk]
        Ttmp[kk] /= ACtmp[kk]
        Ntmp[kk]  = kk

        lmtmp[kk] = np.log10(lmtmp[kk])
        Ztmp[kk]  = np.log10(Ztmp[kk])
        Ttmp[kk]  = np.log10(Ttmp[kk])

        NPAR    = [lmtmp[:kk+1], Ttmp[:kk+1], Avtmp[:kk+1], Ztmp[:kk+1]]

        #for kk in range(0,nplot,1):
        if kk == nplot-1:
            #
            # Histogram
            #
            for i, x in enumerate(Par):
                ax = axes[i, i]
                x1min, x1max = NPARmin[i], NPARmax[i]
                nbin = 50
                binwidth1 = (x1max-x1min)/nbin
                bins1 = np.arange(x1min, x1max + binwidth1, binwidth1)
                n, bins, patches = ax.hist(NPAR[i], bins=bins1, orientation='vertical', color='b', histtype='stepfilled', alpha=0.6)
                yy = np.arange(0,np.max(n)*1.3,1)
                ax.plot(yy*0+np.percentile(NPAR[i],16), yy, linestyle='--', color='gray', lw=1)
                ax.plot(yy*0+np.percentile(NPAR[i],84), yy, linestyle='--', color='gray', lw=1)
                ax.plot(yy*0+np.percentile(NPAR[i],50), yy, linestyle='-', color='gray', lw=1)
                ax.text(np.percentile(NPAR[i],16), np.max(yy)*1.02, '%.2f'%(np.percentile(NPAR[i],16)), fontsize=9)
                ax.text(np.percentile(NPAR[i],50), np.max(yy)*1.02, '%.2f'%(np.percentile(NPAR[i],50)), fontsize=9)
                ax.text(np.percentile(NPAR[i],84), np.max(yy)*1.02, '%.2f'%(np.percentile(NPAR[i],84)), fontsize=9)

                ax.set_xlim(x1min, x1max)
                ax.set_yticklabels([])
                if i == K-1:
                    ax.set_xlabel('%s'%(Par[i]), fontsize=12)
                if i < K-1:
                    ax.set_xticklabels([])

        # Scatter and contour
        for i, x in enumerate(Par):
            for j, y in enumerate(Par):
                #print(i,j,Par[j], Par[i])
                if i > j:
                    ax = axes[i, j]
                    ax.scatter(NPAR[j], NPAR[i], c='b', s=1, marker='o', alpha=0.01)
                    ax.set_xlabel('%s'%(Par[j]), fontsize=12)

                    if kk == nplot-1:
                        X, Y, Z = density_estimation(NPAR[j], NPAR[i])
                        mZ = np.max(Z)
                        ax.contour(X, Y, Z, levels=[0.68*mZ,0.95*mZ,0.99*mZ], linewidths=[0.8,0.5,0.3], colors='gray')
                        #x1min, x1max = np.min(NPAR[j]), np.max(NPAR[j])
                        #y1min, y1max = np.min(NPAR[i]), np.max(NPAR[i])

                    x1min, x1max = NPARmin[j], NPARmax[j]
                    y1min, y1max = NPARmin[i], NPARmax[i]
                    ax.set_xlim(x1min, x1max)
                    ax.set_ylim(y1min, y1max)


                    if j==0:
                        ax.set_ylabel('%s'%(Par[i]), fontsize=12)
                    if j>0:
                        ax.set_yticklabels([])
                    if i<K-1:
                        ax.set_xticklabels([])
                    if i == 2:
                        ax.yaxis.labelpad = 5.

                if i < j:
                    ax = axes[i, j]
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_frame_on(False)
                    ax.set_xticks([])
                    ax.set_yticks([])

                if i == j:
                    ax = axes[i, j]
                    ax.set_yticklabels([])
                    if i == K-1:
                        ax.set_xlabel('%s'%(Par[i]), fontsize=12)
                    if i < K-1:
                        ax.set_xticklabels([])


        if kk%10 == 0 and out_ind == 1:
            fname = DIR_OUT + '%d.png' % kk
            print('Saving frame', fname)
            plt.savefig(fname, dpi=200)
            files.append(fname)

        #plt.savefig(DIR_OUT + '%d.pdf'%(kk))

    # For the last one
    ax0.plot(xg0, fg0 * c / np.square(xg0) / d, marker='', linestyle='-', linewidth=0.5, ms=0.1, color='royalblue', label='')
    ax0.plot(xg1, fg1 * c / np.square(xg1) / d, marker='', linestyle='-', linewidth=0.5, ms=0.1, color='#DF4E00', label='')

    ax0.set_xlim(2200, 88000)
    ax0.set_xscale('log')
    ax0.set_ylim(0., ymax)
    ax0.set_xlabel('Observed wavelength ($\mathrm{\mu m}$)', fontsize=14)
    ax0.set_ylabel('Flux ($\mathrm{erg}/\mathrm{s}/\mathrm{cm}^{2}/\mathrm{\AA}$)', fontsize=13)
    ax1.set_xlabel('$t$ (Gyr)', fontsize=12)
    ax1.set_ylabel('$\dot{M_*}/M_\odot$yr$^{-1}$', fontsize=12)
    ax1.set_xlim(0.008, Txmax)
    ax1.set_ylim(0, SFmax)
    ax1.set_xscale('log')
    ax2.set_xlabel('$t$ (Gyr)', fontsize=12)
    ax2.set_ylabel('$\log Z_*/Z_\odot$', fontsize=12)
    ax2.set_xlim(0.008, Txmax)
    #ax2.set_ylim(NPARmin[3], NPARmax[3])
    ax2.set_ylim(-0.6, 0.5)
    ax2.set_xscale('log')
    #ax2.yaxis.labelpad = -5

    ax1t = ax1.twiny()
    ax2t = ax2.twiny()
    ax1t.set_xlim(0.008, Txmax)
    ax1t.set_xscale('log')
    ax1t.set_xticklabels(zredl[:])
    ax1t.set_xticks(Tzz[:])
    ax1t.tick_params(axis='x', labelcolor='k')
    ax1t.xaxis.set_ticks_position('none')
    ax1.plot(Tzz, Tzz*0+SFmax, marker='|', color='k', ms=3, linestyle='None')

    ax2t.set_xlim(0.008, Txmax)
    ax2t.set_xscale('log')
    ax2t.set_xticklabels(zredl[:])
    ax2t.set_xticks(Tzz[:])
    ax2t.tick_params(axis='x', labelcolor='k')
    ax2t.xaxis.set_ticks_position('none')
    ax2.plot(Tzz, Tzz*0+0.5, marker='|', color='k', ms=3, linestyle='None')

    plt.savefig(DIR_OUT + 'param_' + ID + '_PA' + PA + '_corner.png', dpi=150)
    #plt.close()


def plot_corner_physparam_frame(ID, PA, Zall=np.arange(-1.2,0.4249,0.05), age=[0.01, 0.1, 0.3, 0.7, 1.0, 3.0], tau0=[0.1,0.2,0.3], fig=None, dust_model=0):
    '''
    #
    # If you like to
    # Creat temporal png for gif image.
    #
    '''
    col = ['violet', 'indigo', 'b', 'lightblue', 'lightgreen', 'g', 'orange', 'coral', 'r', 'darkred']#, 'k']

    nage = np.arange(0,len(age),1)
    fnc  = Func(ID, PA, Zall, age, dust_model=dust_model) # Set up the number of Age/ZZ
    bfnc = Basic(Zall)

    ###########################
    # Open result file
    ###########################
    # Open ascii file and stock to array.
    lib     = fnc.open_spec_fits(fall=0, tau0=tau0)
    lib_all = fnc.open_spec_fits(fall=1, tau0=tau0)

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


    # Repeat no.
    nplot = 1000

    DIR_OUT = '/astro/udfcen3/Takahiro/sedfitter/corner/' + ID + '_corner/'
    try:
        os.makedirs(DIR_OUT)
    except:
        pass

    # plot Configuration
    K = 4 # No of params.
    Par = ['$\log M_*/M_\odot$', '$\log T_*$/Gyr', '$A_V$/mag', '$\log Z_* / Z_\odot$']
    factor = 2.0           # size of one side of one panel
    lbdim = 0.5 * factor   # size of left/bottom margin
    trdim = 0.2 * factor   # size of top/right margin
    whspace = 0.02         # w/hspace size
    plotdim = factor * K + factor * (K - 1.) * whspace
    dim = lbdim + plotdim + trdim
    sclfig = 0.7

    # Create a new figure if one wasn't provided.
    ###############################
    # Data taken from
    ###############################
    DIR_TMP = './templates/'
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

    conspec = (NR<10000) #& (fy/ey>1)
    #ax0.plot(xg0, fg0 * c / np.square(xg0) / d, marker='', linestyle='-', linewidth=0.5, ms=0.1, color='royalblue', label='')
    #ax0.plot(xg1, fg1 * c / np.square(xg1) / d, marker='', linestyle='-', linewidth=0.5, ms=0.1, color='#DF4E00', label='')
    conbb = (fybb/eybb>1)

    DIR_TMP = './templates/'
    ####################
    # MCMC corner plot.
    ####################
    file = 'chain_' + ID + '_PA' + PA + '_corner.cpkl'
    niter = 0
    data = loadcpkl(os.path.join('./'+file))

    try:
        ndim   = data['ndim']     # By default, use ndim and burnin values contained in the cpkl file, if present.
        burnin = data['burnin']
        nmc    = data['niter']
        nwalk  = data['nwalkers']
        Nburn  = burnin #*20
        samples = data['chain'][:]
    except:
        if verbose: print(' =   >   NO keys of ndim and burnin found in cpkl, use input keyword values')

    f0     = fits.open(DIR_TMP + 'ms_' + ID + '_PA' + PA + '.fits')
    sedpar = f0[1]

    import matplotlib
    import matplotlib.cm as cm
    getcmap   = matplotlib.cm.get_cmap('jet')
    nc        = np.arange(0, nmc, 1)
    col = getcmap((nc-0)/(nmc-0))

    #for kk in range(0,nmc,1):
    Ntmp = np.zeros(nplot, dtype='float64')
    lmtmp= np.zeros(nplot, dtype='float64')
    Avtmp= np.zeros(nplot, dtype='float64')
    Ztmp = np.zeros(nplot, dtype='float64')
    Ttmp = np.zeros(nplot, dtype='float64')
    ACtmp= np.zeros(nplot, dtype='float64')


    # Time bin
    Txmax = 4 # Max x value
    Tuni = MB.cosmo.age(zbes).value
    Tuni0 = (Tuni - age[:])
    delT  = np.zeros(len(age),dtype='float64')
    delTl = np.zeros(len(age),dtype='float64')
    delTu = np.zeros(len(age),dtype='float64')
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

    ######
    files = [] # For movie
    SFmax = 0
    Tsmin = 0
    Tsmax = 0
    Zsmin = 0
    Zsmax = 0
    AMtmp = 0
    AMtmp16 = 0
    AMtmp84 = 0
    for ii in range(len(age)):
        ZZ_tmp = Z50[ii] #samples['Z'+str(ii)][100]
        ZZ_tmp16 = Z16[ii] #samples['Z'+str(ii)][100]
        ZZ_tmp84 = Z84[ii] #samples['Z'+str(ii)][100]
        AA_tmp = np.max(samples['A'+str(ii)][:])
        AA_tmp84 = np.percentile(samples['A'+str(ii)][:],95)
        AA_tmp16 = np.percentile(samples['A'+str(ii)][:],5)
        #AA_tmp84 = A84[ii]
        #AA_tmp16 = A16[ii]
        nZtmp  = bfnc.Z2NZ(ZZ_tmp)
        mslist = sedpar.data['ML_'+str(nZtmp)][ii]
        AMtmp16 += mslist*AA_tmp16
        AMtmp84 += mslist*AA_tmp84

        Tsmax += age[ii] * AA_tmp84 * mslist
        Tsmin += age[ii] * AA_tmp16 * mslist

        Zsmax += 10**ZZ_tmp84 * AA_tmp84 * mslist
        Zsmin += 10**ZZ_tmp16 * AA_tmp16 * mslist

        SFtmp  = AA_tmp * mslist / delT[ii]
        if SFtmp > SFmax:
            SFmax = SFtmp

    #NPARmin = [np.log10(M16)-0.1, -0.4, 0, -0.6]
    #NPARmax = [np.log10(M84)+0.1, 0.5, 2., 0.5]
    NPARmin = [np.log10(M16)-0.1, np.log10(Tsmin/AMtmp16)-0.1, Av16-0.1, np.log10(Zsmin/AMtmp16)-0.2]
    NPARmax = [np.log10(M84)+0.1, np.log10(Tsmax/AMtmp84)+0.2, Av84+0.1, np.log10(Zsmax/AMtmp84)+0.2]

    for kk in range(0,nplot,1):
        if kk%10 == 0:
            #print('New plot; %d'%kk)
            fig, axes = plt.subplots(K, K, figsize=(dim*sclfig*2, dim*sclfig))
            # Format the figure.
            lb = lbdim / dim
            tr = (lbdim + plotdim) / dim
            #fig.subplots_adjust(left=lb*1.06, bottom=lb*.9, right=tr, top=tr*.99,
            fig.subplots_adjust(left=0.5, bottom=lb*.9, right=tr, top=tr*.99,
                                wspace=whspace, hspace=whspace)

            # For spec plot
            ax0 = fig.add_axes([0.05,0.73,0.37,0.23])
            ax1 = fig.add_axes([0.05,0.40,0.37,0.23])
            ax2 = fig.add_axes([0.05,0.07,0.37,0.23])

            ax0.errorbar(xbb[conbb], fybb[conbb] * c / np.square(xbb[conbb]) / d, yerr=eybb[conbb]*c/np.square(xbb[conbb])/d, color='k', linestyle='', linewidth=0.5, zorder=4)
            ax0.plot(xbb[conbb], fybb[conbb] * c / np.square(xbb[conbb]) / d, '.r', ms=10, linestyle='', linewidth=0, zorder=4)#, label='Obs.(BB)')
            ax0.plot(xg0, fg0 * c / np.square(xg0) / d, marker='', linestyle='-', linewidth=0.5, ms=0.1, color='royalblue', label='')
            ax0.plot(xg1, fg1 * c / np.square(xg1) / d, marker='', linestyle='-', linewidth=0.5, ms=0.1, color='#DF4E00', label='')


        #nr = kk # np.random.randint(len(samples))
        nr = np.random.randint(len(samples))
        Avtmp[kk] = samples['Av'][nr]
        #Asum = 0
        #for ss in range(len(age)):
        #Asum += np.sum(samples['A'+str(ss)][nr])
        ZMM = np.zeros((len(age)), dtype='float64') # Mass weighted Z.
        ZM  = np.zeros((len(age)), dtype='float64') # Light weighted T.
        ZC  = np.zeros((len(age)), dtype='float64') # Light weighted T.
        SF  = np.zeros((len(age)), dtype='float64') # SFR
        AM  = np.zeros((len(age)), dtype='float64') # Light weighted T.


        II0   = nage #[0,1,2,3] # Number for templates
        for ss in range(len(age)):
            ii = int(len(II0) - ss - 1) # from old to young templates.
            AA_tmp = samples['A'+str(ii)][nr]
            try:
                ZZ_tmp = samples['Z'+str(ii)][nr]
            except:
                ZZ_tmp = samples['Z0'][nr]

            nZtmp      = bfnc.Z2NZ(ZZ_tmp)
            mslist     = sedpar.data['ML_'+str(nZtmp)][ii]
            lmtmp[kk] += AA_tmp * mslist
            Ztmp[kk]  += (10 ** ZZ_tmp) * AA_tmp * mslist
            Ttmp[kk]  += age[ii] * AA_tmp * mslist
            ACtmp[kk] += AA_tmp * mslist

            AM[ii] = AA_tmp * mslist
            SF[ii] = AA_tmp * mslist / delT[ii]
            ZM[ii] = ZZ_tmp # AAtmp[aa] * mslist[aa]
            ZMM[ii]= (10 ** ZZ_tmp) * AA_tmp * mslist

            # SED
            flim = 0.05
            if ss == 0:
                y0, x0   = fnc.tmp03(AA_tmp, Avtmp[kk], ii, ZZ_tmp, zbes, lib_all, tau0=tau0)
                y0p, x0p = fnc.tmp03(AA_tmp, Avtmp[kk], ii, ZZ_tmp, zbes, lib, tau0=tau0)
                ysump = y0p #* 1e18
                ysum  = y0  #* 1e18
                if AA_tmp/Asum > flim:
                    ax0.plot(x0, y0 * c/ np.square(x0) / d, '--', lw=1, color=col[ii], zorder=-1, label='', alpha=0.5)
                #ax1.plot(age[ii], SF[ii], marker='o', lw=1, color=col[ii], zorder=1, label='', alpha=0.5)
                #ax1.errorbar(age[ii], SF[ii], xerr=[[delTl[ii]/1e9], [delTu[ii]/1e9]], ms=10, marker='o', lw=1, color=col[ii], zorder=1, label='', alpha=0.5)
                xx1 = np.arange(age[ii]-delTl[ii]/1e9, age[ii]+delTu[ii]/1e9, 0.01)
                ax1.fill_between(xx1, xx1*0, xx1*0+SF[ii], lw=1, facecolor=col[ii], zorder=1, label='', alpha=0.5)
                #ax2.plot(age[ii], ZM[ii], marker='o', lw=1, color=col[ii], zorder=1, label='', alpha=0.5)
            else:
                y0_r, x0_tmp = fnc.tmp03(AA_tmp, Avtmp[kk], ii, ZZ_tmp, zbes, lib_all, tau0=tau0)
                y0p, x0p     = fnc.tmp03(AA_tmp, Avtmp[kk], ii, ZZ_tmp, zbes, lib, tau0=tau0)
                ysump += y0p  #* 1e18
                ysum  += y0_r #* 1e18
                if AA_tmp/Asum > flim:
                    ax0.plot(x0, y0_r * c/ np.square(x0) / d, '--', lw=1, color=col[ii], zorder=-1, label='', alpha=0.5)
                #ax1.plot(age[ii], SF[ii], marker='o', lw=1, color=col[ii], zorder=1, label='', alpha=0.5)
                #ax1.errorbar(age[ii], SF[ii], xerr=[[delTl[ii]/1e9], [delTu[ii]/1e9]], ms=10, marker='o', lw=1, color=col[ii], zorder=1, label='', alpha=0.5)
                xx1 = np.arange(age[ii]-delTl[ii]/1e9, age[ii]+delTu[ii]/1e9, 0.01)
                ax1.fill_between(xx1, xx1*0, xx1*0+SF[ii], lw=1, facecolor=col[ii], zorder=1, label='', alpha=0.5)
                #ax2.plot(age[ii], ZM[ii], marker='o', lw=1, color=col[ii], zorder=1, label='', alpha=0.5)

        for ss in range(len(age)):
            #ii = int(len(II0) - ss - 1) # from old to young templates.
            ii = ss # from old to young templates.
            AC = np.sum(AM[ss:])
            ZC[ss] = np.log10(np.sum(ZMM[ss:])/AC)
            #ax2.errorbar(age[ii], ZC[ii], xerr=[[delTl[ii]/1e9], [delTu[ii]/1e9]], ms=10*(SF[ii]/np.sum(SF[:]))+1, marker='o', lw=1, color=col[ii], zorder=1, label='', alpha=0.5)
            ax2.errorbar(age[ii], ZC[ii], xerr=[[delTl[ii]/1e9], [delTu[ii]/1e9]], ms=10*(AC/np.sum(AM[:]))+1, marker='o', lw=1, color=col[ii], zorder=1, label='', alpha=0.5)

        # Total
        ymax = np.max(fybb[conbb] * c / np.square(xbb[conbb]) / d) * 1.10
        ax0.plot(x0, ysum * c/ np.square(x0) / d, '-', lw=1., color='gray', zorder=-1, label='', alpha=0.8)
        ax0.set_xlim(2200, 88000)
        ax0.set_xscale('log')
        ax0.set_ylim(0., ymax)
        ax0.set_xlabel('Observed wavelength ($\mathrm{\mu m}$)', fontsize=14)
        ax0.set_ylabel('Flux ($\mathrm{erg}/\mathrm{s}/\mathrm{cm}^{2}/\mathrm{\AA}$)', fontsize=13)


        ax1.plot(age[:], SF[:], marker='', linestyle='-', lw=1, color='k', zorder=-1, label='', alpha=0.5)
        ax1.set_xlabel('$t$ (Gyr)', fontsize=12)
        ax1.set_ylabel('$\log \dot{M_*}/M_\odot$yr$^{-1}$', fontsize=12)
        ax1.set_xlim(0.008, Txmax)
        ax1.set_ylim(0, SFmax)
        ax1.set_xscale('log')


        ax2.plot(age[:], ZC[:], marker='', linestyle='-', lw=1, color='k', zorder=-1, label='', alpha=0.5)
        ax2.set_xlabel('$t$ (Gyr)', fontsize=12)
        ax2.set_ylabel('$\log Z_*/Z_\odot$', fontsize=12)
        ax2.set_xlim(0.008, Txmax)
        #ax2.set_ylim(NPARmin[3], NPARmax[3])
        ax2.set_ylim(-0.6, 0.5)
        ax2.set_xscale('log')
        #ax2.yaxis.labelpad = -5

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

        Tzz   = np.zeros(len(zred), dtype='float64')
        for zz in range(len(zred)):
            Tzz[zz] = (Tuni - MB.cosmo.age(zbes).value)
            if Tzz[zz] < 0.01:
                Tzz[zz] = 0.01


        ax1t = ax1.twiny()
        ax2t = ax2.twiny()
        ax1t.set_xlim(0.008, Txmax)
        ax1t.set_xscale('log')
        ax1t.set_xticklabels(zredl[:])
        ax1t.set_xticks(Tzz[:])
        ax1t.tick_params(axis='x', labelcolor='k')
        ax1t.xaxis.set_ticks_position('none')
        ax1.plot(Tzz, Tzz*0+SFmax, marker='|', color='k', ms=3, linestyle='None')

        ax2t.set_xlim(0.008, Txmax)
        ax2t.set_xscale('log')
        ax2t.set_xticklabels(zredl[:])
        ax2t.set_xticks(Tzz[:])
        ax2t.tick_params(axis='x', labelcolor='k')
        ax2t.xaxis.set_ticks_position('none')
        ax2.plot(Tzz, Tzz*0+0.5, marker='|', color='k', ms=3, linestyle='None')

        # Convert into log
        Ztmp[kk] /= ACtmp[kk]
        Ttmp[kk] /= ACtmp[kk]
        Ntmp[kk]  = kk

        lmtmp[kk] = np.log10(lmtmp[kk])
        Ztmp[kk]  = np.log10(Ztmp[kk])
        Ttmp[kk]  = np.log10(Ttmp[kk])


        #NPAR    = [lmtmp[:kk+1], Ttmp[:kk+1], Avtmp[:kk+1], Ztmp[:kk+1]]
        #NPAR    = [lmtmp[kk-10:kk+1], Ttmp[kk-10:kk+1], Avtmp[kk-10:kk+1], Ztmp[kk-10:kk+1]]
        NPAR    = [lmtmp[kk], Ttmp[kk], Avtmp[kk], Ztmp[kk]]

        #for kk in range(0,nplot,1):
        '''
        if kk == nplot-1:
            # Histogram
            for i, x in enumerate(Par):
                ax = axes[i, i]
                x1min, x1max = NPARmin[i], NPARmax[i]
                nbin = 50
                binwidth1 = (x1max-x1min)/nbin
                bins1 = np.arange(x1min, x1max + binwidth1, binwidth1)
                ax.hist(NPAR[i], bins=bins1, orientation='vertical', color='b', histtype='stepfilled', alpha=0.6)
                ax.set_xlim(x1min, x1max)
                print(x, x1min, x1max)
                #ax2.scatter(np.log10(Ttmp), np.log10(Avtmp), c='r', s=1, marker='.', alpha=0.1)
                #ax3.scatter(np.log10(Ztmp), np.log10(Avtmp), c='r', s=1, marker='.', alpha=0.1)
                ax.set_yticklabels([])
                #ax.set_xticklabels([])
                #ax.set_title('%s'%(Par[i]), fontsize=12)
                if i == K-1:
                    ax.set_xlabel('%s'%(Par[i]), fontsize=12)
                if i < K-1:
                    ax.set_xticklabels([])
        '''

        # Scatter and contour
        for i, x in enumerate(Par):
            for j, y in enumerate(Par):
                #print(i,j,Par[j], Par[i])
                if i > j:
                    ax = axes[i, j]
                    ax.scatter(NPAR[j], NPAR[i], c='b', s=10, marker='o', alpha=0.5)
                    ax.set_xlabel('%s'%(Par[j]), fontsize=12)

                    #x1min, x1max = np.min(NPAR[j]), np.max(NPAR[j])
                    #y1min, y1max = np.min(NPAR[i]), np.max(NPAR[i])
                    x1min, x1max = NPARmin[j], NPARmax[j]
                    y1min, y1max = NPARmin[i], NPARmax[i]
                    ax.set_xlim(x1min, x1max)
                    ax.set_ylim(y1min, y1max)

                    if j==0:
                        ax.set_ylabel('%s'%(Par[i]), fontsize=12)
                    if j>0:
                        ax.set_yticklabels([])
                    if i<K-1:
                        ax.set_xticklabels([])
                    if i == 2:
                        ax.yaxis.labelpad = 5.

                if i < j:
                    ax = axes[i, j]
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_frame_on(False)
                    ax.set_xticks([])
                    ax.set_yticks([])

                if i == j:
                    ax = axes[i, j]
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_frame_on(False)
                    ax.set_xticks([])
                    ax.set_yticks([])

                    '''
                    ax.set_yticklabels([])
                    if i == K-1:
                        ax.set_xlabel('%s'%(Par[i]), fontsize=12)
                    if i < K-1:
                        ax.set_xticklabels([])
                    '''

        if kk%10 == 0:
            fname = DIR_OUT + '%d.png' % (kk)
            #fname = DIR_OUT + '%d.png' % (kk+1)
            print('Saving frame', fname)
            plt.savefig(fname, dpi=150)
            #files.append(fname)
            plt.close()


def plot_corner(ID, PA, Zall=np.arange(-1.2,0.4249,0.05), age=[0.01, 0.1, 0.3, 0.7, 1.0, 3.0],  mcmcplot=1, flim=0.05):
    '''
    '''
    col = ['violet', 'indigo', 'b', 'lightblue', 'lightgreen', 'g', 'orange', 'coral', 'r', 'darkred']#, 'k']
    nage = np.arange(0,len(age),1)
    fnc  = Func(ID, PA, Zall, age, dust_model=dust_model) # Set up the number of Age/ZZ
    bfnc = Basic(Zall)

    ####################
    # MCMC corner plot.
    ####################
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

    #
    # Get label
    #
    label = []
    title = []
    truth = np.zeros(len(age)*2+1, dtype='float64')

    file_sum = 'summary_' + ID + '_PA' + PA + '.fits'
    hdu      = fits.open(file_sum) # open a FITS file

    Asum = 0
    for aa in range(len(age)):
        Asum += hdu[1].data['A'+str(aa)][1]

    for aa in range(len(age)):
        label.append(r'$A_%d$'%(aa))
        if hdu[1].data['A'+str(aa)][1] / Asum > flim:
            truth[aa] = hdu[1].data['A'+str(aa)][1]
            title.append(r'$%.2f$'%(hdu[1].data['A'+str(aa)][1]/ Asum))
        else:
            truth[aa] = np.nan
            title.append(r'$%.2f$'%(hdu[1].data['A'+str(aa)][1]/ Asum))

    label.append(r'$A_V$')
    truth[len(age)] = hdu[1].data['Av0'][1]
    title.append('')

    for aa in range(len(age)):
        label.append(r'$Z_%d$'%(aa))
        title.append('')
        if hdu[1].data['A'+str(aa)][1] / Asum > flim:
            truth[len(age)+1+aa] = hdu[1].data['Z'+str(aa)][1]
        else:
            truth[len(age)+1+aa] = np.nan


    if mcmcplot == 1:
        #fig1 = plt.figure(figsize=(10,10))
        #fig1.subplots_adjust(top=0.96, bottom=0.16, left=0.1, right=0.99, hspace=0.15, wspace=0.25)
        fig1 = corner.corner_TM(res, label_kwargs={'fontsize':14}, show_titles=True, titles=title, title_kwargs={"fontsize": 14, 'color':'orangered'}, plot_datapoints=False, plot_contours=True, no_fill_contours=True, plot_density=False, levels=[0.68, 0.95, 99.7], color='#4682b4', scale_hist=False, truths=truth, truth_color='orangered', labels=label)
        #, labels=label, truth_color='gray'

        fig1.savefig('SPEC_' + ID + '_PA' + PA + '_corner.pdf')
        plt.close()
