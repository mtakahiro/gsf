import numpy as np
import sys
import os
import asdf

import matplotlib.pyplot as plt
from numpy import log10
from scipy.integrate import simps
from astropy.io import fits
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker

from .function import *
from .function_class import Func
from .basic_func import Basic
import corner

col = ['violet', 'indigo', 'b', 'lightblue', 'lightgreen', 'g', 'orange', 'coral', 'r', 'darkred']#, 'k']

def plot_sed(MB, flim=0.01, fil_path='./', scale=1e-19, f_chind=True, figpdf=False, save_sed=True, 
    mmax=300, dust_model=0, DIR_TMP='./templates/', f_label=False, f_bbbox=False, verbose=False, f_silence=True,
    f_fill=False, f_fancyplot=False, f_Alog=True, dpi=300, f_plot_filter=True, f_plot_resid=False, NRbb_lim=10000):
    '''
    Parameters
    ----------
    MB.SNlim : float
        SN limit to show flux or up lim in SED.
    f_chind : bool
        If include non-detection in chi2 calculation, using Sawicki12.
    mmax : int
        Number of mcmc realization for plot. Not for calculation.
    f_fancy : bool
        plot each SED component.
    f_fill: bool
        if True, and so is f_fancy, fill each SED component.

    Returns
    -------
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

    if False:#f_silence:
        import matplotlib
        matplotlib.use("Agg")
    else:
        matplotlib.use("MacOSX")

    def gaus(x,a,x0,sigma):
        return a*exp(-(x-x0)**2/(2*sigma**2))
    
    lcb = '#4682b4' # line color, blue

    fnc  = MB.fnc 
    bfnc = MB.bfnc
    ID   = MB.ID
    Z    = MB.Zall
    age  = MB.age
    nage = MB.nage 
    tau0 = MB.tau0
    
    #col = ['violet', 'indigo', 'b', 'lightblue', 'lightgreen', 'g', 'orange', 'coral', 'r', 'darkred']#, 'k']
    NUM_COLORS = len(age)
    cm = plt.get_cmap('gist_rainbow')
    col = [cm(1 - 1.*i/NUM_COLORS) for i in range(NUM_COLORS)]

    nstep_plot = 1
    if MB.f_bpass:
        nstep_plot = 30

    SNlim = MB.SNlim

    ################
    # RF colors.
    home = os.path.expanduser('~')
    c = MB.c
    chimax = 1.
    m0set = MB.m0set
    Mpc_cm = MB.Mpc_cm
    d = MB.d * scale 

    ##################
    # Fitting Results
    ##################
    DIR_FILT = MB.DIR_FILT
    SFILT = MB.filts

    try:
        f_err = MB.ferr
    except:
        f_err = 0

    ###########################
    # Open result file
    ###########################
    file = MB.DIR_OUT + 'summary_' + ID + '.fits'
    hdul = fits.open(file)
    
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
    A50 = np.zeros(len(age), dtype='float')
    A16 = np.zeros(len(age), dtype='float')
    A84 = np.zeros(len(age), dtype='float')
    for aa in range(len(age)):
        A16[aa] = 10**hdul[1].data['A'+str(aa)][0]
        A50[aa] = 10**hdul[1].data['A'+str(aa)][1]
        A84[aa] = 10**hdul[1].data['A'+str(aa)][2]

    Asum  = np.sum(A50)

    if MB.fneb:
        logU50 = hdul[1].data['logU'][1]
        Aneb50 = 10**hdul[1].data['Aneb'][1]

    aa = 0
    Av16 = hdul[1].data['Av'+str(aa)][0]
    Av50 = hdul[1].data['Av'+str(aa)][1]
    Av84 = hdul[1].data['Av'+str(aa)][2]
    AAv = [Av50]

    Z50 = np.zeros(len(age), dtype='float')
    Z16 = np.zeros(len(age), dtype='float')
    Z84 = np.zeros(len(age), dtype='float')
    NZbest = np.zeros(len(age), dtype='int')
    for aa in range(len(age)):
        Z16[aa] = hdul[1].data['Z'+str(aa)][0]
        Z50[aa] = hdul[1].data['Z'+str(aa)][1]
        Z84[aa] = hdul[1].data['Z'+str(aa)][2]
        NZbest[aa]= bfnc.Z2NZ(Z50[aa])

    # Light weighted Z.
    ZZ50 = np.sum(Z50*A50)/np.sum(A50)

    # FIR Dust;
    if MB.f_dust:
        MD16 = hdul[1].data['MDUST'][0]
        MD50 = hdul[1].data['MDUST'][1]
        MD84 = hdul[1].data['MDUST'][2]
        AD16 = hdul[1].data['ADUST'][0]
        AD50 = hdul[1].data['ADUST'][1]
        AD84 = hdul[1].data['ADUST'][2]
        TD16 = hdul[1].data['TDUST'][0]
        TD50 = hdul[1].data['TDUST'][1]
        TD84 = hdul[1].data['TDUST'][2]
        nTD16 = hdul[1].data['nTDUST'][0]
        nTD50 = hdul[1].data['nTDUST'][1]
        nTD84 = hdul[1].data['nTDUST'][2]
        DFILT = MB.inputs['FIR_FILTER'] # filter band string.
        DFILT = [x.strip() for x in DFILT.split(',')]
        DFWFILT = fil_fwhm(DFILT, DIR_FILT)
        if verbose:
            print('Total dust mass is %.2e'%(MD50))

    chi = hdul[1].data['chi'][0]
    chin = hdul[1].data['chi'][1]
    fitc = chin

    Cz0 = hdul[0].header['Cz0']
    Cz1 = hdul[0].header['Cz1']
    Cz2 = hdul[0].header['Cz2']
    zbes = zp50 
    zscl = (1.+zbes)

    ###############################
    # Data taken from
    ###############################
    if MB.f_dust:
        MB.dict = MB.read_data(Cz0, Cz1, Cz2, zbes, add_fir=True)
    else:
        MB.dict = MB.read_data(Cz0, Cz1, Cz2, zbes)

    NR   = MB.dict['NR']
    x    = MB.dict['x']
    fy   = MB.dict['fy']
    ey   = MB.dict['ey']
    
    con0 = (NR<1000)
    xg0  = x[con0]
    fg0  = fy[con0]
    eg0  = ey[con0]
    con1 = (NR>=1000) & (NR<2000)
    xg1  = x[con1]
    fg1  = fy[con1]
    eg1  = ey[con1]
    con2 = (NR>=1000) & (NR<NRbb_lim)
    xg2  = x[con2]
    fg2  = fy[con2]
    eg2  = ey[con2]

    if len(xg0)>0 or len(xg1)>0 or len(xg2)>0:
        f_grsm = True
    else:
        f_grsm = False

    wht = fy * 0
    con_wht = (ey>0)
    wht[con_wht] = 1./np.square(ey[con_wht])

    # BB data points;
    NRbb = MB.dict['NRbb']
    xbb  = MB.dict['xbb']
    fybb = MB.dict['fybb']
    eybb = MB.dict['eybb']
    exbb = MB.dict['exbb']
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
    ms = np.zeros(len(age), dtype='float')
    af = MB.af
    sedpar = af['ML']

    for aa in range(len(age)):
        ms[aa] = sedpar['ML_' +  str(int(NZbest[aa]))][aa]
    try:
        isochrone = af['isochrone']
        LIBRARY = af['library']
        nimf = af['NIMF']
    except:
        isochrone = ''
        LIBRARY = ''
        nimf = ''

    #############
    # Plot.
    #############
    # Set the inset.
    if f_grsm or MB.f_dust:
        fig = plt.figure(figsize=(7.,3.2))
        fig.subplots_adjust(top=0.98, bottom=0.16, left=0.1, right=0.99, hspace=0.15, wspace=0.25)
        ax1 = fig.add_subplot(111)
        xsize = 0.29
        ysize = 0.25
        if f_grsm:
            ax2t = ax1.inset_axes((1-xsize-0.01,1-ysize-0.01,xsize,ysize))
        if MB.f_dust:
            ax3t = ax1.inset_axes((0.7,.35,.28,.25))

        f_plot_resid = False
        print('Grism data. f_plot_resid is turned off.')

    else:
        if f_plot_resid:
            fig_mosaic = """
            AAAA
            AAAA
            BBBB
            """
            fig,axes = plt.subplot_mosaic(mosaic=fig_mosaic, figsize=(5.5,4.2))
            fig.subplots_adjust(top=0.98, bottom=0.16, left=0.1, right=0.99, hspace=0.15, wspace=0.25)
            ax1 = axes['A']
        else:
            fig = plt.figure(figsize=(5.5,2.2))
            fig.subplots_adjust(top=0.98, bottom=0.16, left=0.1, right=0.99, hspace=0.15, wspace=0.25)
            ax1 = fig.add_subplot(111)

    #######################################
    # D.Kelson like Box for BB photometry
    #######################################
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
            data_ex = ascii.read(DIR_TMP + 'bb_obs_' + ID + '_removed.cat')
            NR_ex = data_ex['col1']
        except:
            NR_ex = []

        # Upperlim;
        sigma = 1.0
        leng = np.max(fybb[conbb_hs] * c / np.square(xbb[conbb_hs]) / d) * 0.05 #0.2
        conebb_ls = (fybb/eybb<=SNlim) & (eybb>0)
        
        for ii in range(len(xbb)):
            if NRbb[ii] in NR_ex[:]:
                conebb_ls[ii] = False

        ax1.errorbar(xbb[conebb_ls], eybb[conebb_ls] * c / np.square(xbb[conebb_ls]) / d * sigma, yerr=leng,\
            uplims=eybb[conebb_ls] * c / np.square(xbb[conebb_ls]) / d * sigma, linestyle='', color=col_dat, marker='', ms=4, label='', zorder=4, capsize=3)


    # For any data removed fron fit (i.e. IRAC excess):
    f_exclude = False
    try:
        col_ex = 'lawngreen'
        # Currently, this file is made after FILTER_SKIP;
        data_ex = ascii.read(DIR_TMP + 'bb_obs_' + ID + '_removed.cat')
        x_ex = data_ex['col2']
        fy_ex = data_ex['col3']
        ey_ex = data_ex['col4']
        ex_ex = data_ex['col5']

        ax1.errorbar(x_ex, fy_ex * c / np.square(x_ex) / d, \
        xerr=ex_ex, yerr=ey_ex*c/np.square(x_ex)/d, color='k', linestyle='', linewidth=0.5, zorder=5)
        ax1.scatter(x_ex, fy_ex * c / np.square(x_ex) / d, marker='s', color=col_ex, edgecolor='k', zorder=5, s=30)
        f_exclude = True
    except:
        pass


    #####################################
    # Open ascii file and stock to array.
    lib = fnc.open_spec_fits(fall=0)
    lib_all = fnc.open_spec_fits(fall=1, orig=True)

    if MB.f_dust:
        MB.lib_dust = fnc.open_spec_dust_fits(fall=0)
        MB.lib_dust_all = fnc.open_spec_dust_fits(fall=1)
    if MB.fneb:
        lib_neb = MB.fnc.open_spec_neb_fits(fall=0)
        lib_neb_all = MB.fnc.open_spec_neb_fits(fall=1, orig=True)

    iimax = len(nage)-1

    # FIR dust plot;
    if MB.f_dust:
        from lmfit import Parameters
        par = Parameters()
        par.add('MDUST',value=AD50)
        par.add('TDUST',value=nTD50)
        par.add('zmc',value=zp50)

        y0d, x0d = fnc.tmp04_dust(par.valuesdict())#, zbes, lib_dust_all)
        y0d_cut, x0d_cut = fnc.tmp04_dust(par.valuesdict())#, zbes, lib_dust)

        # data;
        dat_d = ascii.read(MB.DIR_TMP + 'bb_dust_obs_' + MB.ID + '.cat')
        NRbbd = dat_d['col1']
        xbbd = dat_d['col2']
        fybbd = dat_d['col3']
        eybbd = dat_d['col4']
        exbbd = dat_d['col5']
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


    #
    # This is for UVJ color time evolution.
    #
    Asum = np.sum(A50[:])
    alp = .5

    for jj in range(len(age)):
        ii = int(len(nage) - jj - 1) # from old to young templates.
        if jj == 0:
            y0, x0 = fnc.tmp03(A50[ii], AAv[0], ii, Z50[ii], zbes, lib_all)
            y0p, x0p = fnc.tmp03(A50[ii], AAv[0], ii, Z50[ii], zbes, lib)
            ysum = y0
            ysump = y0p
            nopt = len(ysump)

            f_50_comp = np.zeros((len(age),len(y0)),'float') 
            # Keep each component;
            f_50_comp[ii,:] = y0[:] * c / np.square(x0) / d

            if MB.f_dust:
                ysump[:] += y0d_cut[:nopt]
                ysump = np.append(ysump,y0d_cut[nopt:])
                # Keep each component;
                f_50_comp_dust = y0d * c / np.square(x0d) / d

            if MB.fneb: 
                # Only at one age pixel;
                y0_r, x0_tmp = fnc.tmp03_neb(Aneb50, AAv[0], logU50, ii, Z50[ii], zbes, lib_neb_all)
                y0p, x0p = fnc.tmp03_neb(Aneb50, AAv[0], logU50, ii, Z50[ii], zbes, lib_neb)
                ysum += y0_r
                ysump[:nopt] += y0p

        else:
            y0_r, x0_tmp = fnc.tmp03(A50[ii], AAv[0], ii, Z50[ii], zbes, lib_all)
            y0p, x0p = fnc.tmp03(A50[ii], AAv[0], ii, Z50[ii], zbes, lib)
            ysum += y0_r
            ysump[:nopt] += y0p

            f_50_comp[ii,:] = y0_r[:] * c / np.square(x0_tmp) / d

        # The following needs revised.
        f_uvj = False
        if f_uvj:
            if jj == 0:
                fwuvj = open(MB.DIR_OUT + ID + '_uvj.txt', 'w')
                fwuvj.write('# age uv vj\n')
            ysum_wid = ysum * 0
            for kk in range(0,ii+1,1):
                tt = int(len(nage) - kk - 1)
                nn = int(len(nage) - ii - 1)

                nZ = bfnc.Z2NZ(Z50[tt])
                y0_wid, x0_wid = fnc.open_spec_fits_dir(tt, nZ, nn, AAv[0], zbes, A50[tt])
                ysum_wid += y0_wid

            lmrest_wid = x0_wid/(1.+zbes)
            band0 = ['u','v','j']
            lmconv,fconv = filconv(band0, lmrest_wid, ysum_wid, fil_path) # f0 in fnu
            fu_t = fconv[0]
            fv_t = fconv[1]
            fj_t = fconv[2]
            uvt = -2.5*log10(fu_t/fv_t)
            vjt = -2.5*log10(fv_t/fj_t)
            fwuvj.write('%.2f %.3f %.3f\n'%(age[ii], uvt, vjt))
            fwuvj.close()


    #############
    # Main result
    #############
    conbb_ymax = (xbb>0) & (fybb>0) & (eybb>0) & (fybb/eybb>1)
    ymax = np.max(fybb[conbb_ymax]*c/np.square(xbb[conbb_ymax])/d) * 1.6

    xboxl = 17000
    xboxu = 28000

    ax1.set_xlabel('Observed wavelength ($\mathrm{\mu m}$)', fontsize=12)
    ax1.set_ylabel('Flux ($10^{%d}\mathrm{erg}/\mathrm{s}/\mathrm{cm}^{2}/\mathrm{\AA}$)'%(np.log10(scale)),fontsize=12,labelpad=-2)

    x1min = 2000
    x1max = 100000
    xticks = [2500, 5000, 10000, 20000, 40000, 80000, x1max]
    xlabels= ['0.25', '0.5', '1', '2', '4', '8', '']
    if MB.f_dust:
        x1max = 400000
        xticks = [2500, 5000, 10000, 20000, 40000, 80000, 400000]
        xlabels= ['0.25', '0.5', '1', '2', '4', '8', '']

    #if x1max < np.max(xbb[conbb_ymax]):
    #    x1max = np.max(xbb[conbb_ymax]) * 1.5
    if x1max < np.max(xbb):
        x1max = np.max(xbb) * 1.5
    if x1min > np.min(xbb[conbb_ymax]):
        x1min = np.min(xbb[conbb_ymax]) / 1.5

    ax1.set_xlim(x1min, x1max)
    ax1.set_xscale('log')
    if f_plot_filter:
        scl_yaxis = 0.2
    else:
        scl_yaxis = 0.1
    ax1.set_ylim(-ymax*scl_yaxis,ymax)
    ax1.text(x1min+100,-ymax*0.08,'SNlimit:%.1f'%(SNlim),fontsize=8)

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

    xx = np.arange(100,400000)
    yy = xx * 0
    ax1.plot(xx, yy, ls='--', lw=0.5, color='k')

    #############
    # Plot
    #############
    eAAl = np.zeros(len(age),dtype='float')
    eAAu = np.zeros(len(age),dtype='float')
    eAMl = np.zeros(len(age),dtype='float')
    eAMu = np.zeros(len(age),dtype='float')
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
    FLW = np.zeros(len(LN),dtype='int')

    ####################
    # For cosmology
    ####################
    DL = MB.cosmo.luminosity_distance(zbes).value * Mpc_cm #, **cosmo) # Luminositydistance in cm
    Cons = (4.*np.pi*DL**2/(1.+zbes))
    if f_grsm:
        print('This function (write_lines) needs to be revised.')
        write_lines(ID, zbes, DIR_OUT=MB.DIR_OUT)


    ##########################
    # Zoom in Line regions
    ##########################
    if f_grsm:
        conspec = (NR<10000) #& (fy/ey>1)
        ax2t.errorbar(xg2, fg2 * c/np.square(xg2)/d, yerr=eg2 * c/np.square(xg2)/d, lw=0.5, color='#DF4E00', zorder=10, alpha=1., label='', capsize=0)
        ax2t.errorbar(xg1, fg1 * c/np.square(xg1)/d, yerr=eg1 * c/np.square(xg1)/d, lw=0.5, color='g', zorder=10, alpha=1., label='', capsize=0)
        ax2t.errorbar(xg0, fg0 * c/np.square(xg0)/d, yerr=eg0 * c/np.square(xg0)/d, lw=0.5, linestyle='', color='royalblue', zorder=10, alpha=1., label='', capsize=0)

        xgrism = np.concatenate([xg0,xg1,xg2])
        fgrism = np.concatenate([fg0,fg1,fg2])
        egrism = np.concatenate([eg0,eg1,eg2])
        con4000b = (xgrism/zscl>3400) & (xgrism/zscl<3800) & (fgrism>0) & (egrism>0)
        con4000r = (xgrism/zscl>4200) & (xgrism/zscl<5000) & (fgrism>0) & (egrism>0)
        print('Median SN at 3400-3800 is;', np.median((fgrism/egrism)[con4000b]))
        print('Median SN at 4200-5000 is;', np.median((fgrism/egrism)[con4000r]))


    #
    # From MCMC chain
    #
    file = MB.DIR_OUT + 'chain_' + ID + '_corner.cpkl'
    niter = 0
    data = loadcpkl(file)
    try:
        ndim   = data['ndim']     # By default, use ndim and burnin values contained in the cpkl file, if present.
        burnin = data['burnin']
        nmc    = data['niter']
        nwalk  = data['nwalkers']
        Nburn  = burnin #*20
        res    = data['chain'][:]
    except:
        if verbose: print(' =   >   NO keys of ndim and burnin found in cpkl, use input keyword values')

    samples = res

    # Saved template;
    ytmp = np.zeros((mmax,len(ysum)), dtype='float')
    ytmp_each = np.zeros((mmax,len(ysum),len(age)), dtype='float')

    ytmpmax = np.zeros(len(ysum), dtype='float')
    ytmpmin = np.zeros(len(ysum), dtype='float')

    # MUV;
    DL      = MB.cosmo.luminosity_distance(zbes).value * Mpc_cm # Luminositydistance in cm
    DL10    = Mpc_cm/1e6 * 10 # 10pc in cm
    Fuv     = np.zeros(mmax, dtype='float') # For Muv
    Fuv28   = np.zeros(mmax, dtype='float') # For Fuv(1500-2800)
    Lir     = np.zeros(mmax, dtype='float') # For L(8-1000um)
    UVJ     = np.zeros((mmax,4), dtype='float') # For UVJ color;
    Cmznu   = 10**((48.6+m0set)/(-2.5)) # Conversion from m0_25 to fnu

    # From random chain;
    alp=0.02
    for kk in range(0,mmax,1):
        nr = np.random.randint(Nburn, len(samples['A%d'%MB.aamin[0]]))
        try:
            Av_tmp = samples['Av'][nr]
        except:
            Av_tmp = MB.AVFIX

        try:
            zmc = samples['zmc'][nr]
        except:
            zmc = zbes

        for ss in MB.aamin:
            try:
                AA_tmp = 10**samples['A'+str(ss)][nr]
            except:
                AA_tmp = 0
                pass
            try:
                Ztest = samples['Z'+str(len(age)-1)][nr]
                ZZ_tmp = samples['Z'+str(ss)][nr]
            except:
                try:
                    ZZ_tmp = samples['Z0'][nr]
                except:
                    ZZ_tmp = MB.ZFIX

            if ss == MB.aamin[0]:
                mod0_tmp, xm_tmp = fnc.tmp03(AA_tmp, Av_tmp, ss, ZZ_tmp, zmc, lib_all)
                fm_tmp = mod0_tmp
                if MB.fneb:
                    Aneb_tmp = 10**samples['Aneb'][nr]
                    if not MB.logUFIX == None:
                        logU_tmp = MB.logUFIX
                    else:
                        logU_tmp = samples['logU'][nr]
                    mod0_tmp, xm_tmp = fnc.tmp03_neb(Aneb_tmp, Av_tmp, logU_tmp, ss, ZZ_tmp, zmc, lib_neb_all)
                    fm_tmp += mod0_tmp
            else:
                mod0_tmp, xx_tmp = fnc.tmp03(AA_tmp, Av_tmp, ss, ZZ_tmp, zmc, lib_all)
                fm_tmp += mod0_tmp

            # Each;
            ytmp_each[kk,:,ss] = mod0_tmp[:] * c / np.square(xm_tmp[:]) / d

        #
        # Dust component;
        #
        if MB.f_dust:
            if kk == 0:
                par = Parameters()
                par.add('MDUST',value=samples['MDUST'][nr])
                try:
                    par.add('TDUST',value=samples['TDUST'][nr])
                except:
                    par.add('TDUST',value=0)

            par['MDUST'].value = samples['MDUST'][nr]
            if not MB.TDUSTFIX == None:
                par['TDUST'].value = MB.NTDUST
            else:
                par['TDUST'].value = samples['TDUST'][nr]
                #except:
                #    par['TDUST'].value = 0

            model_dust, x1_dust = fnc.tmp04_dust(par.valuesdict())
            if kk == 0:
                deldt  = (x1_dust[1] - x1_dust[0])
                x1_tot = np.append(xm_tmp,np.arange(np.max(xm_tmp),np.max(x1_dust)*2,deldt))
                # Redefine??
                ytmp = np.zeros((mmax,len(x1_tot)), dtype='float')
                ytmp_dust = np.zeros((mmax,len(x1_dust)), dtype='float')
                ytmp_comp = np.zeros((mmax,len(x1_tot)), dtype='float')

            ytmp_dust[kk,:] = model_dust * c/np.square(x1_dust)/d
            model_tot = np.interp(x1_tot,xx_tmp,fm_tmp) + np.interp(x1_tot,x1_dust,model_dust)

            ytmp[kk,:] = model_tot[:] * c/np.square(x1_tot[:])/d

        else:
            x1_tot = xm_tmp
            ytmp[kk,:] = fm_tmp[:] * c / np.square(xm_tmp[:]) / d

        #
        # Grism plot + Fuv flux + LIR.
        #
        #if f_grsm:
            #ax2t.plot(x1_tot, ytmp[kk,:], '-', lw=0.5, color='gray', zorder=3., alpha=0.02)

        # Get FUV flux;
        Fuv[kk] = get_Fuv(x1_tot[:]/(1.+zbes), (ytmp[kk,:]/(c/np.square(x1_tot)/d)) * (DL**2/(1.+zbes)) / (DL10**2), lmin=1250, lmax=1650)
        Fuv28[kk] = get_Fuv(x1_tot[:]/(1.+zbes), (ytmp[kk,:]/(c/np.square(x1_tot)/d)) * (4*np.pi*DL**2/(1.+zbes))*Cmznu, lmin=1500, lmax=2800)
        Lir[kk] = 0

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
    ytmp16 = np.percentile(ytmp[:,:],16,axis=0)
    ytmp50 = np.percentile(ytmp[:,:],50,axis=0)
    ytmp84 = np.percentile(ytmp[:,:],84,axis=0)
    
    if MB.f_dust:
        ytmp_dust50 = np.percentile(ytmp_dust[:,:],50, axis=0)

    #if not f_fill:
    ax1.fill_between(x1_tot[::nstep_plot], ytmp16[::nstep_plot], ytmp84[::nstep_plot], ls='-', lw=.5, color='gray', zorder=-2, alpha=0.5)
    ax1.plot(x1_tot[::nstep_plot], ytmp50[::nstep_plot], '-', lw=.5, color='gray', zorder=-1, alpha=1.)

    # For grism;
    if f_grsm:
        from astropy.convolution import convolve
        from .maketmp_filt import get_LSF
        LSF, _ = get_LSF(MB.inputs, MB.DIR_EXTR, ID, x1_tot[:], c=3e18)
        spec_grsm16 = convolve(ytmp16[:], LSF, boundary='extend')
        spec_grsm50 = convolve(ytmp50[:], LSF, boundary='extend')
        spec_grsm84 = convolve(ytmp84[:], LSF, boundary='extend')
        if True:
            ax2t.plot(x1_tot[:], ytmp50, '-', lw=0.5, color='gray', zorder=3., alpha=1.0)
        else:
            ax2t.plot(x1_tot[:], spec_grsm50, '-', lw=0.5, color='gray', zorder=3., alpha=1.0)
        
    # Attach the data point in MB;
    MB.sed_wave_obs = xbb
    MB.sed_flux_obs = fybb * c / np.square(xbb) / d
    MB.sed_eflux_obs = eybb * c / np.square(xbb) / d
    # Attach the best SED to MB;
    MB.sed_wave = x1_tot
    MB.sed_flux16 = ytmp16
    MB.sed_flux50 = ytmp50
    MB.sed_flux84 = ytmp84

    if f_fancyplot:
        alp_fancy = 0.5
        #ax1.plot(x1_tot[::nstep_plot], np.percentile(ytmp[:, ::nstep_plot], 50, axis=0), '-', lw=.5, color='gray', zorder=-1, alpha=1.)
        ysumtmp = ytmp[0, ::nstep_plot] * 0
        ysumtmp2 = ytmp[:, ::nstep_plot] * 0
        ysumtmp2_prior = ytmp[0, ::nstep_plot] * 0
        for ss in range(len(age)):
            ii = int(len(nage) - ss - 1)
            # !! Take median after summation;
            ysumtmp2[:,:len(xm_tmp)] += ytmp_each[:, ::nstep_plot, ii]
            if f_fill:
                ax1.fill_between(x1_tot[::nstep_plot], ysumtmp2_prior,  np.percentile(ysumtmp2[:,:], 50, axis=0), linestyle='None', lw=0., color=col[ii], alpha=alp_fancy, zorder=-3)
            else:
                ax1.plot(x1_tot[::nstep_plot], np.percentile(ysumtmp2[:, ::nstep_plot], 50, axis=0), linestyle='--', lw=.5, color=col[ii], alpha=alp_fancy, zorder=-3)
            ysumtmp2_prior[:] = np.percentile(ysumtmp2[:, :], 50, axis=0)
    elif f_fill:
        print('f_fancyplot is False. f_fill is set to False.')

    #########################
    # Calculate non-det chi2
    # based on Sawick12
    #########################
    def func_tmp(xint,eobs,fmodel):
        int_tmp = np.exp(-0.5 * ((xint-fmodel)/eobs)**2)
        return int_tmp

    if f_chind:
        conw = (wht3>0) & (ey>0) & (fy/ey>SNlim)
    else:
        conw = (wht3>0) & (ey>0) #& (fy/ey>SNlim)

    #chi2 = sum((np.square(fy-ysump) * np.sqrt(wht3))[conw])
    try:
        logf = hdul[1].data['logf'][1]
        ey_revised = np.sqrt(ey**2+ ysump**2 * np.exp(logf)**2)
    except:
        ey_revised = ey

    chi2 = sum((np.square(fy-ysump) / ey_revised)[conw])

    chi_nd = 0.0
    if f_chind:
        f_ex = np.zeros(len(fy), 'int')
        if f_exclude:
            for ii in range(len(fy)):
                if x[ii] in x_ex:
                    f_ex[ii] = 1

        con_up = (ey>0) & (fy/ey<=SNlim) & (f_ex == 0)
        from scipy import special
        #x_erf = (ey[con_up] - ysump[con_up]) / (np.sqrt(2) * ey[con_up])
        #f_erf = special.erf(x_erf)
        #chi_nd = np.sum( np.log(np.sqrt(np.pi / 2) * ey[con_up] * (1 + f_erf)) )
        x_erf = (ey_revised[con_up] - ysump[con_up]) / (np.sqrt(2) * ey_revised[con_up])
        f_erf = special.erf(x_erf)
        chi_nd = np.sum( np.log(np.sqrt(np.pi / 2) * ey_revised[con_up] * (1 + f_erf)) )

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
    if MB.f_dust:
        ALLFILT = np.append(SFILT,DFILT)
        #for ii in range(len(x1_tot)):
        #    print(x1_tot[ii], model_tot[ii]*c/np.square(x1_tot[ii])/d)
        lbb, fbb, lfwhm = filconv(ALLFILT, x1_tot, ytmp50, DIR_FILT, fw=True)
        lbb, fbb16, lfwhm = filconv(ALLFILT, x1_tot, ytmp16, DIR_FILT, fw=True)
        lbb, fbb84, lfwhm = filconv(ALLFILT, x1_tot, ytmp84, DIR_FILT, fw=True)

        ax1.plot(x1_tot, ytmp50, '--', lw=0.5, color='purple', zorder=-1, label='')
        ax3t.plot(x1_tot, ytmp50, '--', lw=0.5, color='purple', zorder=-1, label='')

        iix = []
        for ii in range(len(fbb)):
            iix.append(ii)
        con_sed = ()
        ax1.scatter(lbb[iix][con_sed], fbb[iix][con_sed], lw=0.5, color='none', edgecolor=col_dia, zorder=3, alpha=1.0, marker='d', s=50)

        # plot FIR range;
        ax3t.scatter(lbb, fbb, lw=0.5, color='none', edgecolor=col_dia, \
        zorder=2, alpha=1.0, marker='d', s=50)

    else:
        lbb, fbb, lfwhm = filconv(SFILT, x1_tot, ytmp50, DIR_FILT, fw=True, MB=MB, f_regist=False)
        lbb, fbb16, lfwhm = filconv(SFILT, x1_tot, ytmp16, DIR_FILT, fw=True, MB=MB, f_regist=False)
        lbb, fbb84, lfwhm = filconv(SFILT, x1_tot, ytmp84, DIR_FILT, fw=True, MB=MB, f_regist=False)

        iix = []
        for ii in range(len(fbb)):
            iix.append(np.argmin(np.abs(lbb[ii]-xbb[:])))
        con_sed = (eybb>0)
        ax1.scatter(lbb[iix][con_sed], fbb[iix][con_sed], lw=0.5, color='none', edgecolor=col_dia, zorder=3, alpha=1.0, marker='d', s=50)

        if f_plot_resid:
            conbb_hs = (fybb/eybb>SNlim)
            axes['B'].scatter(lbb[iix][conbb_hs], ((fybb*c/np.square(xbb)/d - fbb)/(eybb*c/np.square(xbb)/d))[iix][conbb_hs], lw=0.5, color='none', edgecolor='r', zorder=3, alpha=1.0, marker='.', s=50)
            conbb_hs = (fybb/eybb<=SNlim) & (eybb>0)
            axes['B'].errorbar(lbb[iix][conbb_hs], ((eybb*c/np.square(xbb)/d - fbb)/(eybb*c/np.square(xbb)/d))[iix][conbb_hs], yerr=leng,\
                uplims=((fybb*c/np.square(xbb)/d - fbb)/(eybb*c/np.square(xbb)/d))[iix][conbb_hs] * sigma, linestyle='',\
                color=col_dat, lw=0.5, marker='', ms=4, label='', zorder=4, capsize=1.5)
            axes['B'].set_xscale(ax1.get_xscale())
            axes['B'].set_xlim(ax1.get_xlim())
            axes['B'].set_xticks(ax1.get_xticks())
            axes['B'].set_xticklabels(ax1.get_xticklabels())
            axes['B'].set_xlabel(ax1.get_xlabel())
            xx = np.arange(axes['B'].get_xlim()[0],axes['B'].get_xlim()[1],100)
            axes['B'].plot(xx,xx*0,linestyle='--',lw=0.5,color='k')
            axes['B'].set_ylabel('Residual / $\sigma$')
            axes['A'].set_xlabel('')
            axes['A'].set_xticks(ax1.get_xticks())
            axes['A'].set_xticklabels('')

        # Calculate EW, if there is excess band;
        try:
            iix2 = []
            for ii in range(len(fy_ex)):
                iix2.append(np.argmin(np.abs(lbb[:]-x_ex[ii])))

            # Rest-frame EW;
            # Note about 16/84 in fbb
            EW16 = (fy_ex * c / np.square(x_ex) / d - fbb84[iix2]) / (fbb[iix2]) * lfwhm[iix2] / (1.+zbes)
            EW50 = (fy_ex * c / np.square(x_ex) / d - fbb[iix2]) / (fbb[iix2]) * lfwhm[iix2] / (1.+zbes)
            EW84 = (fy_ex * c / np.square(x_ex) / d - fbb16[iix2]) / (fbb[iix2]) * lfwhm[iix2] / (1.+zbes)

            EW50_er1 = ((fy_ex-ey_ex) * c / np.square(x_ex) / d - fbb[iix2]) / (fbb[iix2]) * lfwhm[iix2] / (1.+zbes)
            EW50_er2 = ((fy_ex+ey_ex) * c / np.square(x_ex) / d - fbb[iix2]) / (fbb[iix2]) * lfwhm[iix2] / (1.+zbes)

            cnt50 = fbb[iix2]
            cnt16 = fbb16[iix2]
            cnt84 = fbb84[iix2]
 
            # Luminosity;
            #Lsun = 3.839 * 1e33 #erg s-1
            L16 = EW16 * cnt16 * (4.*np.pi*DL**2) * scale * (1+zbes) # A * erg/s/A/cm2 * cm2
            L50 = EW50 * cnt50 * (4.*np.pi*DL**2) * scale * (1+zbes) # A * erg/s/A/cm2 * cm2
            L84 = EW84 * cnt84 * (4.*np.pi*DL**2) * scale * (1+zbes) # A * erg/s/A/cm2 * cm2

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
        fbb16_nu = flamtonu(lbb, fbb16*scale, m0set=m0set)
        fbb_nu = flamtonu(lbb, fbb*scale, m0set=m0set)
        fbb84_nu = flamtonu(lbb, fbb84*scale, m0set=m0set)

        # Then save full spectrum;
        col00  = []
        col1  = fits.Column(name='wave_model', format='E', unit='AA', array=x1_tot)
        col00.append(col1)
        col2  = fits.Column(name='f_model_16', format='E', unit='1e%derg/s/cm2/AA'%(np.log10(scale)), array=ytmp16[:])
        col00.append(col2)
        col3  = fits.Column(name='f_model_50', format='E', unit='1e%derg/s/cm2/AA'%(np.log10(scale)), array=ytmp50[:])
        col00.append(col3)
        col4  = fits.Column(name='f_model_84', format='E', unit='1e%derg/s/cm2/AA'%(np.log10(scale)), array=ytmp84[:])
        col00.append(col4)

        # Each component
        # Stellar
        col1 = fits.Column(name='wave_model_stel', format='E', unit='AA', array=x0)
        col00.append(col1)
        for aa in range(len(age)):
            col1 = fits.Column(name='f_model_stel_%d'%aa, format='E', unit='1e%derg/s/cm2/AA'%(np.log10(scale)), array=f_50_comp[aa,:])
            col00.append(col1)
        if MB.f_dust:
            col1 = fits.Column(name='wave_model_dust', format='E', unit='AA', array=x1_dust)
            col00.append(col1)
            col1 = fits.Column(name='f_model_dust', format='E', unit='1e%derg/s/cm2/AA'%(np.log10(scale)), array=ytmp_dust50)
            col00.append(col1)
            
        # Grism;
        if f_grsm:
            col2  = fits.Column(name='f_model_conv_16', format='E', unit='1e%derg/s/cm2/AA'%(np.log10(scale)), array=spec_grsm16)
            col00.append(col2)
            col3  = fits.Column(name='f_model_conv_50', format='E', unit='1e%derg/s/cm2/AA'%(np.log10(scale)), array=spec_grsm50)
            col00.append(col3)
            col4  = fits.Column(name='f_model_conv_84', format='E', unit='1e%derg/s/cm2/AA'%(np.log10(scale)), array=spec_grsm84)
            col00.append(col4)

        # BB for dust
        if MB.f_dust:
            xbb = np.append(xbb,xbbd)
            fybb = np.append(fybb,fybbd)
            eybb = np.append(eybb,eybbd)

        col5 = fits.Column(name='wave_obs', format='E', unit='AA', array=xbb)
        col00.append(col5)
        col6 = fits.Column(name='f_obs', format='E', unit='1e%derg/s/cm2/AA'%(np.log10(scale)), array=fybb[:] * c / np.square(xbb[:]) / d)
        col00.append(col6)
        col7 = fits.Column(name='e_obs', format='E', unit='1e%derg/s/cm2/AA'%(np.log10(scale)), array=eybb[:] * c / np.square(xbb[:]) / d)
        col00.append(col7)

        hdr = fits.Header()
        hdr['redshift'] = zbes
        hdr['id'] = ID
        hdr['hierarch isochrone'] = isochrone
        hdr['library'] = LIBRARY
        hdr['nimf'] = nimf
        hdr['scale'] = scale

        try:
            # Chi square:
            hdr['chi2'] = chi2
            hdr['hierarch No-of-effective-data-points'] = len(wht3[conw])
            hdr['hierarch No-of-nondetectioin'] = len(ey[con_up])
            hdr['hierarch Chi2-of-nondetection'] = chi_nd
            hdr['hierarch No-of-params'] = ndim_eff
            hdr['hierarch Degree-of-freedom']  = nod
            hdr['hierarch reduced-chi2'] = fin_chi2
        except:
            print('Chi seems to be wrong...')
            pass

        try:
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
        except:
            pass

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
                hdr['L_%s_16'%(ew_label[ii])] = L16[ii]
                hdr['L_%s_50'%(ew_label[ii])] = L50[ii]
                hdr['L_%s_84'%(ew_label[ii])] = L84[ii]
        except:
            pass

        # Version;
        import gsf
        hdr['version'] = gsf.__version__

        # Write;
        colspec = fits.ColDefs(col00)
        hdu0 = fits.BinTableHDU.from_columns(colspec, header=hdr)
        hdu0.writeto(MB.DIR_OUT + 'gsf_spec_%s.fits'%(ID), overwrite=True)

        # ASDF;
        tree_spec = {
            'id': ID,
            'redshift': '%.3f'%zbes,
            'isochrone': '%s'%(isochrone),
            'library': '%s'%(LIBRARY),
            'nimf': '%s'%(nimf),
            'scale': scale,
            'version_gsf': gsf.__version__
        }

        # BB;
        tree_spec.update({'wave': lbb})
        tree_spec.update({'fnu_16': fbb16_nu})
        tree_spec.update({'fnu_50': fbb_nu})
        tree_spec.update({'fnu_84': fbb84_nu})
        # full spectrum;
        tree_spec.update({'wave_model': x1_tot})
        tree_spec.update({'f_model_16': ytmp16})
        tree_spec.update({'f_model_50': ytmp50})
        tree_spec.update({'f_model_84': ytmp84})

        # EW;
        try:
            for ii in range(len(EW50)):
                tree_spec.update({'EW_%s_16'%(ew_label[ii]): EW16[ii]})
                tree_spec.update({'EW_%s_50'%(ew_label[ii]): EW50[ii]})
                tree_spec.update({'EW_%s_84'%(ew_label[ii]): EW84[ii]})
                tree_spec.update({'EW_%s_e1'%(ew_label[ii]): EW50_er1[ii]})
                tree_spec.update({'EW_%s_e2'%(ew_label[ii]): EW50_er2[ii]})
                tree_spec.update({'cnt_%s_16'%(ew_label[ii]): cnt16[ii]})
                tree_spec.update({'cnt_%s_50'%(ew_label[ii]): cnt50[ii]})
                tree_spec.update({'cnt_%s_84'%(ew_label[ii]): cnt84[ii]})
                tree_spec.update({'L_%s_16'%(ew_label[ii]): L16[ii]})
                tree_spec.update({'L_%s_50'%(ew_label[ii]): L50[ii]})
                tree_spec.update({'L_%s_84'%(ew_label[ii]): L84[ii]})
        except:
            pass

        # Each component
        # Stellar
        tree_spec.update({'wave_model_stel': x0})
        for aa in range(len(age)):
            tree_spec.update({'f_model_stel_%d'%aa: f_50_comp[aa,:]})
        if MB.f_dust:
            # dust
            tree_spec.update({'wave_model_dust': x1_dust})
            tree_spec.update({'f_model_dust': ytmp_dust50})            
        # BB for dust
        tree_spec.update({'wave_obs': xbb})
        tree_spec.update({'f_obs': fybb[:] * c / np.square(xbb[:]) / d})
        tree_spec.update({'e_obs': eybb[:] * c / np.square(xbb[:]) / d})
        # grism:
        if f_grsm:
            tree_spec.update({'fg0_obs': fg0 * c/np.square(xg0)/d})
            tree_spec.update({'eg0_obs': eg0 * c/np.square(xg0)/d})
            tree_spec.update({'wg0_obs': xg0})
            tree_spec.update({'fg1_obs': fg1 * c/np.square(xg1)/d})
            tree_spec.update({'eg1_obs': eg1 * c/np.square(xg1)/d})
            tree_spec.update({'wg1_obs': xg1})
            tree_spec.update({'fg2_obs': fg2 * c/np.square(xg2)/d})
            tree_spec.update({'eg2_obs': eg2 * c/np.square(xg2)/d})
            tree_spec.update({'wg2_obs': xg2})

        af = asdf.AsdfFile(tree_spec)
        af.write_to(MB.DIR_OUT + 'gsf_spec_%s.asdf'%(ID), all_array_compression='zlib')

    #
    # SED params in plot
    #
    if f_label:
        fd = fits.open(MB.DIR_OUT + 'SFH_' + ID + '.fits')[0].header
        if MB.f_dust:
            label = 'ID: %s\n$z_\mathrm{obs.}:%.2f$\n$\log M_\mathrm{*}/M_\odot:%.2f$\n$\log M_\mathrm{dust}/M_\odot:%.2f$\n$T_\mathrm{dust}/K:%.1f$\n$\log Z_\mathrm{*}/Z_\odot:%.2f$\n$\log T_\mathrm{*}$/Gyr$:%.2f$\n$A_V$/mag$:%.2f$\n$\\chi^2/\\nu:%.2f$'\
            %(ID, zbes, float(fd['Mstel_50']), MD50, TD50, float(fd['Z_MW_50']), float(fd['T_MW_50']), float(fd['AV_50']), fin_chi2)
            ylabel = ymax*0.45
        else:
            label = 'ID: %s\n$z_\mathrm{obs.}:%.2f$\n$\log M_\mathrm{*}/M_\odot:%.2f$\n$\log Z_\mathrm{*}/Z_\odot:%.2f$\n$\log T_\mathrm{*}$/Gyr$:%.2f$\n$A_V$/mag$:%.2f$\n$\\chi^2/\\nu:%.2f$'\
            %(ID, zbes, float(fd['Mstel_50']), float(fd['Z_MW_50']), float(fd['T_MW_50']), float(fd['AV_50']), fin_chi2)
            ylabel = ymax*0.25

        if f_grsm:
            ax1.text(0.02, 0.7, label,\
            fontsize=9, bbox=dict(facecolor='w', alpha=0.7), zorder=10,
            ha='left', va='center', transform=ax1.transAxes)
        else:
            ax1.text(0.77, 0.7, label,\
            fontsize=9, bbox=dict(facecolor='w', alpha=0.7), zorder=10,
            ha='left', va='center', transform=ax1.transAxes)

    ax1.xaxis.labelpad = -3

    if f_grsm:
        if np.max(xg0)<23000: # E.g. WFC3, NIRISS grisms
            conlim = (x0>10000) & (x0<25000)
            xgmin, xgmax = np.min(x0[conlim]),np.max(x0[conlim]), #7500, 17000
            ax2t.set_xlabel('')
            ax2t.set_xlim(xgmin, xgmax)

            conaa = (x0>xgmin-50) & (x0<xgmax+50)
            ymaxzoom = np.max(ysum[conaa]*c/np.square(x0[conaa])/d) * 1.15
            yminzoom = np.min(ysum[conaa]*c/np.square(x0[conaa])/d) / 1.15

            ax2t.set_ylim(yminzoom, ymaxzoom)
            ax2t.xaxis.labelpad = -2
            if xgmax>20000:
                ax2t.set_xticks([8000, 12000, 16000, 20000, 24000])
                ax2t.set_xticklabels(['0.8', '1.2', '1.6', '2.0', '2.4'])
            else:
                ax2t.set_xticks([8000, 10000, 12000, 14000, 16000])
                ax2t.set_xticklabels(['0.8', '1.0', '1.2', '1.4', '1.6'])
        else:
            conlim = (x0>10000) & (x0<54000) # NIRSPEC spectrum;
            xgmin, xgmax = np.min(x0[conlim]),np.max(x0[conlim]), #7500, 17000
            ax2t.set_xlabel('')
            ax2t.set_xlim(xgmin, xgmax)

            conaa = (x0>xgmin-50) & (x0<xgmax+50)
            ymaxzoom = np.max(ysum[conaa]*c/np.square(x0[conaa])/d) * 1.15
            yminzoom = np.min(ysum[conaa]*c/np.square(x0[conaa])/d) / 1.15

            ax2t.set_ylim(yminzoom, ymaxzoom)
            ax2t.xaxis.labelpad = -2
            if xgmax>40000:
                ax2t.set_xticks([8000, 20000, 32000, 44000, 56000])
                ax2t.set_xticklabels(['0.8', '2.0', '3.2', '4.4', '5.6'])
            else:
                ax2t.set_xticks([8000, 20000, 32000, 44000])
                ax2t.set_xticklabels(['0.8', '2.0', '3.2', '4.4'])

    if MB.f_dust:
        try:
            contmp = (x1_tot>10*1e4) #& (fybbd/eybbd>SNlim)
            y3min, y3max = -.2*np.max((model_tot * c/ np.square(x1_tot) / d)[contmp]), np.max((model_tot * c/ np.square(x1_tot) / d)[contmp])*2.0
            ax3t.set_ylim(y3min, y3max)
        except:
            if verbose:
                print('y3 limit is not specified.')
            pass
        ax3t.set_xlim(1e5, 3e7)
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

    # Filters
    ind_remove = np.where((wht3<=0) | (ey<=0))[0]
    if f_plot_filter:
        ax1 = plot_filter(MB, ax1, ymax, scl=scl_yaxis, ind_remove=ind_remove)

    ####################
    ## Save
    ####################
    ax1.legend(loc=1, fontsize=11)
    if figpdf:
        fig.savefig(MB.DIR_OUT + 'SPEC_' + ID + '_spec.pdf', dpi=dpi)
    else:
        fig.savefig(MB.DIR_OUT + 'SPEC_' + ID + '_spec.png', dpi=dpi)


def plot_sed_tau(MB, flim=0.01, fil_path='./', scale=1e-19, f_chind=True, figpdf=False, save_sed=True, 
    mmax=300, dust_model=0, DIR_TMP='./templates/', f_label=False, f_bbbox=False, verbose=False, f_silence=True, 
    f_fill=False, f_fancyplot=False, f_Alog=True, dpi=300, f_plot_filter=True, f_plot_resid=False, NRbb_lim=10000):
    '''
    Parameters
    ----------
    MB.SNlim : float
        SN limit to show flux or up lim in SED.
    f_chind : bool
        If include non-detection in chi2 calculation, using Sawicki12.
    mmax : int
        Number of mcmc realization for plot. Not for calculation.
    f_fancy : bool
        plot each SED component.
    f_fill : bool
        if True, and so is f_fancy, fill each SED component.

    Returns
    -------
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
    
    lcb = '#4682b4' # line color, blue

    fnc  = MB.fnc
    bfnc = MB.bfnc
    ID   = MB.ID
    Z    = MB.Zall
    age  = MB.age
    nage = MB.nage 
    tau0 = MB.tau0
    
    NUM_COLORS = len(age)
    cm = plt.get_cmap('gist_rainbow')
    col = [cm(1 - 1.*i/NUM_COLORS) for i in range(NUM_COLORS)]

    nstep_plot = 1
    if MB.f_bpass:
        nstep_plot = 30

    SNlim = MB.SNlim

    ################
    # RF colors.
    home = os.path.expanduser('~')
    c = MB.c
    chimax = 1.
    m0set = MB.m0set
    Mpc_cm = MB.Mpc_cm
    d = MB.d * scale
    
    ##################
    # Fitting Results
    ##################
    DIR_FILT = MB.DIR_FILT
    SFILT = MB.filts

    try:
        f_err = MB.ferr
    except:
        f_err = 0

    ###########################
    # Open result file
    ###########################
    file = MB.DIR_OUT + 'summary_' + ID + '.fits'
    hdul = fits.open(file) 
    
    ndim_eff = hdul[0].header['NDIM']
    vals = {}

    # Redshift MC
    zp16  = hdul[1].data['zmc'][0]
    zp50  = hdul[1].data['zmc'][1]
    zp84  = hdul[1].data['zmc'][2]
    vals['zmc'] = zp50

    # Stellar mass MC
    M16 = hdul[1].data['ms'][0]
    M50 = hdul[1].data['ms'][1]
    M84 = hdul[1].data['ms'][2]
    if verbose:
        print('Total stellar mass is %.2e'%(M50))

    # Amplitude MC
    A50 = np.zeros(len(age), dtype='float')
    A16 = np.zeros(len(age), dtype='float')
    A84 = np.zeros(len(age), dtype='float')
    for aa in range(len(age)):
        A16[aa] = 10**hdul[1].data['A'+str(aa)][0]
        A50[aa] = 10**hdul[1].data['A'+str(aa)][1]
        A84[aa] = 10**hdul[1].data['A'+str(aa)][2]
        vals['A'+str(aa)] = np.log10(A50[aa])

    Asum  = np.sum(A50)

    # TAU MC
    # AGE MC
    TAU50 = np.zeros(len(age), dtype='float')
    TAU16 = np.zeros(len(age), dtype='float')
    TAU84 = np.zeros(len(age), dtype='float')
    AGE50 = np.zeros(len(age), dtype='float')
    AGE16 = np.zeros(len(age), dtype='float')
    AGE84 = np.zeros(len(age), dtype='float')
    for aa in range(len(age)):
        TAU16[aa] = 10**hdul[1].data['TAU'+str(aa)][0]
        TAU50[aa] = 10**hdul[1].data['TAU'+str(aa)][1]
        TAU84[aa] = 10**hdul[1].data['TAU'+str(aa)][2]
        AGE16[aa] = 10**hdul[1].data['AGE'+str(aa)][0]
        AGE50[aa] = 10**hdul[1].data['AGE'+str(aa)][1]
        AGE84[aa] = 10**hdul[1].data['AGE'+str(aa)][2]
        vals['TAU'+str(aa)] = np.log10(TAU50[aa])
        vals['AGE'+str(aa)] = np.log10(AGE50[aa])

    aa = 0
    Av16 = hdul[1].data['Av'+str(aa)][0]
    Av50 = hdul[1].data['Av'+str(aa)][1]
    Av84 = hdul[1].data['Av'+str(aa)][2]
    AAv = [Av50]
    vals['Av'] = Av50

    Z50 = np.zeros(len(age), dtype='float')
    Z16 = np.zeros(len(age), dtype='float')
    Z84 = np.zeros(len(age), dtype='float')
    for aa in range(len(age)):
        Z16[aa] = hdul[1].data['Z'+str(aa)][0]
        Z50[aa] = hdul[1].data['Z'+str(aa)][1]
        Z84[aa] = hdul[1].data['Z'+str(aa)][2]
        vals['Z'+str(aa)] = Z50[aa]

    # Light weighted Z.
    ZZ50 = np.sum(Z50*A50)/np.sum(A50)

    # FIR Dust;
    try:
        MD16 = hdul[1].data['MDUST'][0]
        MD50 = hdul[1].data['MDUST'][1]
        MD84 = hdul[1].data['MDUST'][2]
        AD16 = hdul[1].data['ADUST'][0]
        AD50 = hdul[1].data['ADUST'][1]
        AD84 = hdul[1].data['ADUST'][2]
        TD16 = hdul[1].data['TDUST'][0]
        TD50 = hdul[1].data['TDUST'][1]
        TD84 = hdul[1].data['TDUST'][2]
        nTD16 = hdul[1].data['nTDUST'][0]
        nTD50 = hdul[1].data['nTDUST'][1]
        nTD84 = hdul[1].data['nTDUST'][2]
        DFILT = MB.inputs['FIR_FILTER'] # filter band string.
        DFILT = [x.strip() for x in DFILT.split(',')]
        DFWFILT = fil_fwhm(DFILT, DIR_FILT)
        if verbose:
            print('Total dust mass is %.2e'%(MD50))
        f_dust = True
    except:
        f_dust = False

    chi = hdul[1].data['chi'][0]
    chin = hdul[1].data['chi'][1]
    fitc = chin

    Cz0 = hdul[0].header['Cz0']
    Cz1 = hdul[0].header['Cz1']
    Cz2 = hdul[0].header['Cz2']
    zbes = zp50 
    zscl = (1.+zbes)

    ###############################
    # Data taken from
    ###############################
    if MB.f_dust:
        MB.dict = MB.read_data(Cz0, Cz1, Cz2, zbes, add_fir=True)
    else:
        MB.dict = MB.read_data(Cz0, Cz1, Cz2, zbes)

    NR   = MB.dict['NR']
    x    = MB.dict['x']
    fy   = MB.dict['fy']
    ey   = MB.dict['ey']
    
    con0 = (NR<1000)
    xg0  = x[con0]
    fg0  = fy[con0]
    eg0  = ey[con0]
    con1 = (NR>=1000) & (NR<2000) #& (fy/ey>SNlim)
    xg1  = x[con1]
    fg1  = fy[con1]
    eg1  = ey[con1]
    con2 = (NR>=2000) & (NR<NRbb_lim) #& (fy/ey>SNlim)
    xg2  = x[con2]
    fg2  = fy[con2]
    eg2  = ey[con2]
    if len(xg0)>0 or len(xg1)>0 or len(xg2)>0:
        f_grsm = True
    else:
        f_grsm = False

    # Weight is set to zero for those no data (ey<0).
    wht = fy * 0
    con_wht = (ey>0)
    wht[con_wht] = 1./np.square(ey[con_wht])

    # BB data points;
    NRbb = MB.dict['NRbb']
    xbb  = MB.dict['xbb']
    fybb = MB.dict['fybb']
    eybb = MB.dict['eybb']
    exbb = MB.dict['exbb']
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
    af = MB.af
    sedpar = af['ML']
    try:
        isochrone = af['isochrone']
        LIBRARY = af['library']
        nimf = af['nimf']
    except:
        isochrone = ''
        LIBRARY = ''
        nimf = ''

    #############
    # Plot.
    #############
    # Set the inset.
    if f_grsm or f_dust:
        fig = plt.figure(figsize=(7.,3.2))
        fig.subplots_adjust(top=0.98, bottom=0.16, left=0.1, right=0.99, hspace=0.15, wspace=0.25)
        ax1 = fig.add_subplot(111)
        xsize = 0.29
        ysize = 0.25
        if f_grsm:
            ax2t = ax1.inset_axes((1-xsize-0.01,1-ysize-0.01,xsize,ysize))
        if f_dust:
            ax3t = ax1.inset_axes((0.7,.35,.28,.25))
        f_plot_resid = False
    else:
        if f_plot_resid:
            fig_mosaic = """
            AAAA
            AAAA
            BBBB
            """
            fig,axes = plt.subplot_mosaic(mosaic=fig_mosaic, figsize=(5.5,4.2))
            fig.subplots_adjust(top=0.98, bottom=0.16, left=0.1, right=0.99, hspace=0.15, wspace=0.25)
            ax1 = axes['A']
        else:
            fig = plt.figure(figsize=(5.5,2.2))
            fig.subplots_adjust(top=0.98, bottom=0.16, left=0.1, right=0.99, hspace=0.15, wspace=0.25)
            ax1 = fig.add_subplot(111)

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
            data_ex = ascii.read(DIR_TMP + 'bb_obs_' + ID + '_removed.cat')
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
    f_exclude = False
    try:
        col_ex = 'lawngreen'
        #col_ex = 'limegreen'
        #col_ex = 'r'
        # Currently, this file is made after FILTER_SKIP;
        data_ex = ascii.read(DIR_TMP + 'bb_obs_' + ID + '_removed.cat')
        x_ex = data_ex['col2']
        fy_ex = data_ex['col3']
        ey_ex = data_ex['col4']
        ex_ex = data_ex['col5']

        ax1.errorbar(x_ex, fy_ex * c / np.square(x_ex) / d, \
        xerr=ex_ex, yerr=ey_ex*c/np.square(x_ex)/d, color='k', linestyle='', linewidth=0.5, zorder=5)
        ax1.scatter(x_ex, fy_ex * c / np.square(x_ex) / d, marker='s', color=col_ex, edgecolor='k', zorder=5, s=30)
        f_exclude = True
    except:
        pass


    #####################################
    # Open ascii file and stock to array.
    MB.lib = fnc.open_spec_fits(fall=0)
    MB.lib_all = fnc.open_spec_fits(fall=1, orig=True)

    if f_dust:
        DT0 = float(MB.inputs['TDUST_LOW'])
        DT1 = float(MB.inputs['TDUST_HIG'])
        dDT = float(MB.inputs['TDUST_DEL'])
        Temp = np.arange(DT0,DT1,dDT)
        MB.lib_dust = fnc.open_spec_dust_fits(fall=0)
        MB.lib_dust_all = fnc.open_spec_dust_fits(fall=1)

    # FIR dust plot;
    if f_dust:
        from lmfit import Parameters
        par = Parameters()
        par.add('MDUST',value=AD50)
        par.add('TDUST',value=nTD50)
        par.add('zmc',value=zp50)

        y0d, x0d = fnc.tmp04_dust(par.valuesdict())#, zbes, lib_dust_all)
        y0d_cut, x0d_cut = fnc.tmp04_dust(par.valuesdict())#, zbes, lib_dust)
        
        # data;
        dat_d = ascii.read(MB.DIR_TMP + 'bb_dust_obs_' + MB.ID + '.cat')
        NRbbd = dat_d['col1']
        xbbd = dat_d['col2']
        fybbd = dat_d['col3']
        eybbd = dat_d['col4']
        exbbd = dat_d['col5']
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


    #
    # This is for UVJ color time evolution.
    #
    Asum = np.sum(A50[:])
    alp = .5

    # Get total templates
    y0p, x0p = MB.fnc.tmp04(vals, f_val=False, check_bound=False)
    y0, x0 = MB.fnc.tmp04(vals, f_val=False, check_bound=False, lib_all=True)

    ysum = y0
    f_50_comp = y0[:] * c / np.square(x0) / d

    ysump = y0p
    nopt = len(ysump)

    if f_dust:
        ysump[:] += y0d_cut[:nopt]
        ysump = np.append(ysump,y0d_cut[nopt:])
        f_50_comp_dust = y0d * c / np.square(x0d) / d

    # Plot each best fit:
    vals_each = vals.copy()
    for aa in range(len(age)):
        vals_each['A%d'%aa] = -99
    for aa in range(len(age)):
        vals_each['A%d'%aa] = vals['A%d'%aa]
        y0tmp, x0tmp = MB.fnc.tmp04(vals_each, f_val=False, check_bound=False, lib_all=True)
        if aa == 0:
            y0keep = y0tmp
        else:
            y0keep += y0tmp
        ax1.plot(x0tmp, y0tmp * c / np.square(x0tmp) / d, linestyle='--', lw=0.5, color=col[aa])
        vals_each['A%d'%aa] = 0

    # Plot best fit;
    ax1.plot(x0, f_50_comp, linestyle='-', lw=0.5, color='k')

    #############
    # Main result
    #############
    conbb_ymax = (xbb>0) & (fybb>0) & (eybb>0) & (fybb/eybb>1) # (conbb) &
    ymax = np.max(fybb[conbb_ymax]*c/np.square(xbb[conbb_ymax])/d) * 1.6

    xboxl = 17000
    xboxu = 28000

    x1max = 22000
    if x1max < np.max(xbb[conbb_ymax]):
        x1max = np.max(xbb[conbb_ymax]) * 1.5
    ax1.set_xlim(2000, 11000)
    ax1.set_xscale('log')
    if f_plot_filter:
        scl_yaxis = 0.2
    else:
        scl_yaxis = 0.1
    ax1.set_ylim(-ymax*scl_yaxis,ymax)
    ax1.text(2100,-ymax*0.08,'SNlimit:%.1f'%(SNlim),fontsize=8)

    ax1.set_xlabel('Observed wavelength ($\mathrm{\mu m}$)', fontsize=12)
    ax1.set_ylabel('Flux ($10^{%d}\mathrm{erg}/\mathrm{s}/\mathrm{cm}^{2}/\mathrm{\AA}$)'%(np.log10(scale)),fontsize=12,labelpad=-2)

    xticks = [2500, 5000, 10000, 20000, 40000, 80000, 110000]
    xlabels= ['0.25', '0.5', '1', '2', '4', '8', '']
    if f_dust:
        xticks = [2500, 5000, 10000, 20000, 40000, 80000, 400000]
        xlabels= ['0.25', '0.5', '1', '2', '4', '8', '']

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

    xx = np.arange(1200,400000)
    yy = xx * 0
    ax1.plot(xx, yy, ls='--', lw=0.5, color='k')

    #############
    # Plot
    #############
    ms = np.zeros(len(age), dtype='float')
    af = MB.af
    sedpar = af['ML']

    eAAl = np.zeros(len(age),dtype='float')
    eAAu = np.zeros(len(age),dtype='float')
    eAMl = np.zeros(len(age),dtype='float')
    eAMu = np.zeros(len(age),dtype='float')
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
    FLW = np.zeros(len(LN),dtype='int')

    ####################
    # For cosmology
    ####################
    DL = MB.cosmo.luminosity_distance(zbes).value * Mpc_cm
    Cons = (4.*np.pi*DL**2/(1.+zbes))

    if f_grsm:
        print('This function (write_lines) needs to be revised.')
        write_lines(ID, zbes, DIR_OUT=MB.DIR_OUT)


    ##########################
    # Zoom in Line regions
    ##########################
    if f_grsm:
        conspec = (NR<10000) #& (fy/ey>1)
        ax2t.errorbar(xg2, fg2 * c/np.square(xg2)/d, yerr=eg2 * c/np.square(xg2)/d, lw=0.5, color='#DF4E00', zorder=10, alpha=1., label='', capsize=0)
        ax2t.errorbar(xg1, fg1 * c/np.square(xg1)/d, yerr=eg1 * c/np.square(xg1)/d, lw=0.5, color='g', zorder=10, alpha=1., label='', capsize=0)
        ax2t.errorbar(xg0, fg0 * c/np.square(xg0)/d, yerr=eg0 * c/np.square(xg0)/d, lw=0.5, linestyle='', color='royalblue', zorder=10, alpha=1., label='', capsize=0)

        xgrism = np.concatenate([xg0,xg1,xg2])
        fgrism = np.concatenate([fg0,fg1,fg2])
        egrism = np.concatenate([eg0,eg1,eg2])
        con4000b = (xgrism/zscl>3400) & (xgrism/zscl<3800) & (fgrism>0) & (egrism>0)
        con4000r = (xgrism/zscl>4200) & (xgrism/zscl<5000) & (fgrism>0) & (egrism>0)
        print('Median SN at 3400-3800 is;', np.median((fgrism/egrism)[con4000b]))
        print('Median SN at 4200-5000 is;', np.median((fgrism/egrism)[con4000r]))


    #
    # From MCMC chain
    #
    file = MB.DIR_OUT + 'chain_' + ID + '_corner.cpkl'
    niter = 0
    data = loadcpkl(file)
    ndim = data['ndim'] 
    burnin = data['burnin']
    nmc = data['niter']
    nwalk = data['nwalkers']
    Nburn = burnin
    res = data['chain'][:]

    samples = res

    # Saved template;
    ytmp = np.zeros((mmax,len(ysum)), dtype='float')
    ytmp_each = np.zeros((mmax,len(ysum),len(age)), dtype='float')

    ytmpmax = np.zeros(len(ysum), dtype='float')
    ytmpmin = np.zeros(len(ysum), dtype='float')

    # MUV;
    DL      = MB.cosmo.luminosity_distance(zbes).value * Mpc_cm # Luminositydistance in cm
    DL10    = Mpc_cm/1e6 * 10 # 10pc in cm
    Fuv     = np.zeros(mmax, dtype='float') # For Muv
    Fuv28   = np.zeros(mmax, dtype='float') # For Fuv(1500-2800)
    Lir     = np.zeros(mmax, dtype='float') # For L(8-1000um)
    UVJ     = np.zeros((mmax,4), dtype='float') # For UVJ color;

    Cmznu   = 10**((48.6+m0set)/(-2.5)) # Conversion from m0_25 to fnu

    # From random chain;
    alp=0.02
    for kk in range(0,mmax,1):
        nr = np.random.randint(Nburn, len(samples['A%d'%MB.aamin[0]]))
        try:
            Av_tmp = samples['Av'][nr]
        except:
            Av_tmp = MB.AVFIX
        vals['Av'] = Av_tmp

        try:
            zmc = samples['zmc'][nr]
        except:
            zmc = zbes
        vals['zmc'] = zmc

        for ss in MB.aamin:
            try:
                AA_tmp = 10**samples['A'+str(ss)][nr]
            except:
                AA_tmp = 0
            vals['A%d'%ss] = np.log10(AA_tmp)

            if ss == 0 or MB.ZEVOL:
                try:
                    ZZtmp = samples['Z%d'%ss][nr]
                except:
                    ZZtmp = MB.ZFIX
                vals['Z%d'%ss] = ZZtmp

        mod0_tmp, xm_tmp = fnc.tmp04(vals, f_val=False, check_bound=False, lib_all=True)
        fm_tmp = mod0_tmp

        if False:
            # Each;
            ytmp_each[kk,:,ss] = mod0_tmp[:] * c / np.square(xm_tmp[:]) / d
            #if kk == 100:
            #    ax1.plot(xm_tmp[:], ytmp_each[kk,:,ss], color=col[ss], linestyle='--')

        #
        # Dust component;
        #
        if f_dust:
            if kk == 0:
                par = Parameters()
                par.add('MDUST',value=samples['MDUST'][nr])
                try:
                    par.add('TDUST',value=samples['TDUST'][nr])
                except:
                    par.add('TDUST',value=0)

            par['MDUST'].value = samples['MDUST'][nr]
            if not MB.TDUSTFIX == None:
                par['TDUST'].value = MB.NTDUST
            else:
                par['TDUST'].value = samples['TDUST'][nr]

            model_dust, x1_dust = fnc.tmp04_dust(par.valuesdict())#, zbes, lib_dust_all)
            if kk == 0:
                deldt  = (x1_dust[1] - x1_dust[0])
                x1_tot = np.append(xm_tmp,np.arange(np.max(xm_tmp),np.max(x1_dust),deldt))
                # Redefine??
                ytmp = np.zeros((mmax,len(x1_tot)), dtype='float')
                ytmp_dust = np.zeros((mmax,len(x1_dust)), dtype='float')

            ytmp_dust[kk,:] = model_dust * c/np.square(x1_dust)/d
            model_tot = np.interp(x1_tot,xm_tmp,fm_tmp) + np.interp(x1_tot,x1_dust,model_dust)

            ytmp[kk,:] = model_tot[:] * c/np.square(x1_tot[:])/d

        else:
            x1_tot = xm_tmp
            ytmp[kk,:] = fm_tmp[:] * c / np.square(xm_tmp[:]) / d

        # plot random sed;
        plot_mc = True
        if plot_mc:
            ax1.plot(x1_tot, ytmp[kk,:], '-', lw=1, color='gray', zorder=-2, alpha=0.02)

        # Grism plot + Fuv flux + LIR.
        #if f_grsm:
        #    ax2t.plot(x1_tot, ytmp[kk,:], '-', lw=0.5, color='gray', zorder=3., alpha=0.02)

        if True:
            # Get FUV flux;
            Fuv[kk] = get_Fuv(x1_tot[:]/(1.+zbes), (ytmp[kk,:]/(c/np.square(x1_tot)/d)) * (DL**2/(1.+zbes)) / (DL10**2), lmin=1250, lmax=1650)
            Fuv28[kk] = get_Fuv(x1_tot[:]/(1.+zbes), (ytmp[kk,:]/(c/np.square(x1_tot)/d)) * (4*np.pi*DL**2/(1.+zbes))*Cmznu, lmin=1500, lmax=2800)
            Lir[kk] = 0

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

    print('')

    #
    # Plot Median SED;
    #
    ytmp16 = np.percentile(ytmp[:,:],16,axis=0)
    ytmp50 = np.percentile(ytmp[:,:],50,axis=0)
    ytmp84 = np.percentile(ytmp[:,:],84,axis=0)
    
    if f_dust:
        ytmp_dust50 = np.percentile(ytmp_dust[:,:],50, axis=0)

    # For grism;
    if f_grsm:
        from astropy.convolution import convolve
        from .maketmp_filt import get_LSF
        LSF, _ = get_LSF(MB.inputs, MB.DIR_EXTR, ID, x1_tot[:], c=3e18)
        spec_grsm16 = convolve(ytmp16[:], LSF, boundary='extend')
        spec_grsm50 = convolve(ytmp50[:], LSF, boundary='extend')
        spec_grsm84 = convolve(ytmp84[:], LSF, boundary='extend')
        if False:#True:
            ax2t.plot(x1_tot[:], ytmp50, '-', lw=0.5, color='gray', zorder=3., alpha=1.0)
        else:
            ax2t.plot(x1_tot[:], spec_grsm50, '-', lw=0.5, color='gray', zorder=3., alpha=1.0)


    #if not f_fill:
    ax1.fill_between(x1_tot[::nstep_plot], ytmp16[::nstep_plot], ytmp84[::nstep_plot], ls='-', lw=.5, color='gray', zorder=-2, alpha=0.5)
    ax1.plot(x1_tot[::nstep_plot], ytmp50[::nstep_plot], '-', lw=.5, color='gray', zorder=-1, alpha=1.)

    # Attach the data point in MB;
    MB.sed_wave_obs = xbb
    MB.sed_flux_obs = fybb * c / np.square(xbb) / d
    MB.sed_eflux_obs = eybb * c / np.square(xbb) / d
    # Attach the best SED to MB;
    MB.sed_wave = x1_tot
    MB.sed_flux16 = ytmp16
    MB.sed_flux50 = ytmp50
    MB.sed_flux84 = ytmp84


    #########################
    # Calculate non-det chi2
    # based on Sawick12
    #########################
    #chi2,fin_chi2 = get_chi2(fy, ey, wht3, ysump, ndim_eff, SNlim=1.0, f_chind=f_chind, f_exclude=f_exclude, xbb=xbb, x_ex=x_ex)
    def func_tmp(xint,eobs,fmodel):
        int_tmp = np.exp(-0.5 * ((xint-fmodel)/eobs)**2)
        return int_tmp

    if f_chind:
        conw = (wht3>0) & (ey>0) & (fy/ey>SNlim)
    else:
        conw = (wht3>0) & (ey>0)

    chi2 = sum((np.square(fy-ysump) * np.sqrt(wht3))[conw])

    chi_nd = 0.0
    if f_chind:
        f_ex = np.zeros(len(fy), 'int')
        if f_exclude:
            for ii in range(len(fy)):
                if x[ii] in x_ex:
                    f_ex[ii] = 1

        con_up = (ey>0) & (fy/ey<=SNlim) & (f_ex == 0)
        from scipy import special
        x_erf = (ey[con_up] - ysump[con_up]) / (np.sqrt(2) * ey[con_up])
        f_erf = special.erf(x_erf)
        chi_nd = np.sum( np.log(np.sqrt(np.pi / 2) * ey[con_up] * (1 + f_erf)) )

    # Number of degree;
    con_nod = (wht3>0) & (ey>0) #& (fy/ey>SNlim)
    if MB.ferr:
        ndim_eff -= 1
        
    nod = int(len(wht3[con_nod])-ndim_eff)

    if nod>0:
        fin_chi2 = (chi2 - 2 * chi_nd) / nod
    else:
        fin_chi2 = -99

    if f_chind:
        conw = (wht3>0) & (ey>0) & (fy/ey>SNlim)
        con_up = (ey>0) & (fy/ey<=SNlim) & (f_ex == 0)
    else:
        conw = (wht3>0) & (ey>0)

    # Print results;
    print('\n')
    print('No-of-detection    : %d'%(len(wht3[conw])))
    print('chi2               : %.2f'%(chi2))
    if f_chind:
        print('No-of-non-detection: %d'%(len(ey[con_up])))
        print('chi2 for non-det   : %.2f'%(- 2 * chi_nd))
    print('No-of-params       : %d'%(ndim_eff))
    print('Degrees-of-freedom : %d'%(nod))
    print('Final chi2/nu      : %.2f'%(fin_chi2))


    if False:
        from lmfit import Model, Parameters, minimize, fit_report, Minimizer
        from .posterior_flexible import Post
        class_post = Post(MB)
        residual = class_post.residual
        MB.set_param()
        fit_params = MB.fit_params #Parameters()
        for key in vals.keys():
            try:
                fit_params[key].value=vals[key]
            except:
                pass
        out_tmp = minimize(residual, fit_params, args=(fy, ey, wht3, False), method='differential_evolution') # nelder is the most efficient.
        csq = out_tmp.chisqr
        rcsq = out_tmp.redchi
        print(csq, rcsq)

    #
    # plot BB model from best template (blue squares)
    #
    col_dia = 'blue'
    if f_dust:
        ALLFILT = np.append(SFILT,DFILT)
        #for ii in range(len(x1_tot)):
        #    print(x1_tot[ii], model_tot[ii]*c/np.square(x1_tot[ii])/d)
        lbb, fbb, lfwhm = filconv(ALLFILT, x1_tot, ytmp50, DIR_FILT, fw=True)
        lbb, fbb16, lfwhm = filconv(ALLFILT, x1_tot, ytmp16, DIR_FILT, fw=True)
        lbb, fbb84, lfwhm = filconv(ALLFILT, x1_tot, ytmp84, DIR_FILT, fw=True)

        ax1.plot(x1_tot, ytmp50, '--', lw=0.5, color='purple', zorder=-1, label='')
        ax3t.plot(x1_tot, ytmp50, '--', lw=0.5, color='purple', zorder=-1, label='')

        iix = []
        for ii in range(len(fbb)):
            iix.append(ii)
        con_sed = ()
        ax1.scatter(lbb[iix][con_sed], fbb[iix][con_sed], lw=0.5, color='none', edgecolor=col_dia, zorder=3, alpha=1.0, marker='d', s=50)

        # plot FIR range;
        ax3t.scatter(lbb, fbb, lw=0.5, color='none', edgecolor=col_dia, \
        zorder=2, alpha=1.0, marker='d', s=50)

    else:
        lbb, fbb, lfwhm = filconv(SFILT, x1_tot, ytmp50, DIR_FILT, fw=True, MB=MB, f_regist=False)
        lbb, fbb16, lfwhm = filconv(SFILT, x1_tot, ytmp16, DIR_FILT, fw=True, MB=MB, f_regist=False)
        lbb, fbb84, lfwhm = filconv(SFILT, x1_tot, ytmp84, DIR_FILT, fw=True, MB=MB, f_regist=False)

        iix = []
        for ii in range(len(fbb)):
            iix.append(np.argmin(np.abs(lbb[ii]-xbb[:])))
        con_sed = (eybb>0)
        ax1.scatter(lbb[iix][con_sed], fbb[iix][con_sed], lw=0.5, color='none', edgecolor=col_dia, zorder=3, alpha=1.0, marker='d', s=50)

        if f_plot_resid:
            conbb_hs = (fybb/eybb>SNlim)
            axes['B'].scatter(lbb[iix][conbb_hs], ((fybb*c/np.square(xbb)/d - fbb)/(eybb*c/np.square(xbb)/d))[iix][conbb_hs], lw=0.5, color='none', edgecolor='r', zorder=3, alpha=1.0, marker='.', s=50)
            conbb_hs = (fybb/eybb<=SNlim) & (eybb>0)
            axes['B'].errorbar(lbb[iix][conbb_hs], ((eybb*c/np.square(xbb)/d - fbb)/(eybb*c/np.square(xbb)/d))[iix][conbb_hs], yerr=leng,\
                uplims=((fybb*c/np.square(xbb)/d - fbb)/(eybb*c/np.square(xbb)/d))[iix][conbb_hs] * sigma, linestyle='',\
                color=col_dat, lw=0.5, marker='', ms=4, label='', zorder=4, capsize=1.5)
            axes['B'].set_xscale(ax1.get_xscale())
            axes['B'].set_xlim(ax1.get_xlim())
            axes['B'].set_xticks(ax1.get_xticks())
            axes['B'].set_xticklabels(ax1.get_xticklabels())
            axes['B'].set_xlabel(ax1.get_xlabel())
            xx = np.arange(axes['B'].get_xlim()[0],axes['B'].get_xlim()[1],100)
            axes['B'].plot(xx,xx*0,linestyle='--',lw=0.5,color='k')
            axes['B'].set_ylabel('Residual / $\sigma$')
            axes['A'].set_xlabel('')
            axes['A'].set_xticks(ax1.get_xticks())
            axes['A'].set_xticklabels('')

        # Calculate EW, if there is excess band;
        try:
            iix2 = []
            for ii in range(len(fy_ex)):
                iix2.append(np.argmin(np.abs(lbb[:]-x_ex[ii])))

            # Rest-frame EW;
            # Note about 16/84 in fbb
            EW16 = (fy_ex * c / np.square(x_ex) / d - fbb84[iix2]) / (fbb[iix2]) * lfwhm[iix2] / (1.+zbes)
            EW50 = (fy_ex * c / np.square(x_ex) / d - fbb[iix2]) / (fbb[iix2]) * lfwhm[iix2] / (1.+zbes)
            EW84 = (fy_ex * c / np.square(x_ex) / d - fbb16[iix2]) / (fbb[iix2]) * lfwhm[iix2] / (1.+zbes)

            EW50_er1 = ((fy_ex-ey_ex) * c / np.square(x_ex) / d - fbb[iix2]) / (fbb[iix2]) * lfwhm[iix2] / (1.+zbes)
            EW50_er2 = ((fy_ex+ey_ex) * c / np.square(x_ex) / d - fbb[iix2]) / (fbb[iix2]) * lfwhm[iix2] / (1.+zbes)

            cnt50 = fbb[iix2]
            cnt16 = fbb16[iix2]
            cnt84 = fbb84[iix2]
 
            # Luminosity;
            #Lsun = 3.839 * 1e33 #erg s-1
            L16 = EW16 * cnt16 * (4.*np.pi*DL**2) * scale * (1+zbes) # A * erg/s/A/cm2 * cm2
            L50 = EW50 * cnt50 * (4.*np.pi*DL**2) * scale * (1+zbes) # A * erg/s/A/cm2 * cm2
            L84 = EW84 * cnt84 * (4.*np.pi*DL**2) * scale * (1+zbes) # A * erg/s/A/cm2 * cm2

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
            print('\nEW calculation; Failed.\n')
            pass


    if save_sed:
        fbb16_nu = flamtonu(lbb, fbb16*scale, m0set=m0set)
        fbb_nu = flamtonu(lbb, fbb*scale, m0set=m0set)
        fbb84_nu = flamtonu(lbb, fbb84*scale, m0set=m0set)

        # Then save full spectrum;
        col00  = []
        col1  = fits.Column(name='wave_model', format='E', unit='AA', array=x1_tot)
        col00.append(col1)
        col2  = fits.Column(name='f_model_16', format='E', unit='1e%derg/s/cm2/AA'%(np.log10(scale)), array=ytmp16[:])
        col00.append(col2)
        col3  = fits.Column(name='f_model_50', format='E', unit='1e%derg/s/cm2/AA'%(np.log10(scale)), array=ytmp50[:])
        col00.append(col3)
        col4  = fits.Column(name='f_model_84', format='E', unit='1e%derg/s/cm2/AA'%(np.log10(scale)), array=ytmp84[:])
        col00.append(col4)

        f_sed_each = False
        if f_sed_each:
            # Each component
            # Stellar
            col1 = fits.Column(name='wave_model_stel', format='E', unit='AA', array=x0)
            col00.append(col1)
            for aa in range(len(age)):
                col1 = fits.Column(name='f_model_stel_%d'%aa, format='E', unit='1e%derg/s/cm2/AA'%(np.log10(scale)), array=f_50_comp[aa,:])
                col00.append(col1)
            if f_dust:
                col1 = fits.Column(name='wave_model_dust', format='E', unit='AA', array=x1_dust)
                col00.append(col1)
                col1 = fits.Column(name='f_model_dust', format='E', unit='1e%derg/s/cm2/AA'%(np.log10(scale)), array=ytmp_dust50)
                col00.append(col1)
            
        # BB for dust
        if f_dust:
            xbb = np.append(xbb,xbbd)
            fybb = np.append(fybb,fybbd)
            eybb = np.append(eybb,eybbd)

        col5  = fits.Column(name='wave_obs', format='E', unit='AA', array=xbb)
        col00.append(col5)
        col6  = fits.Column(name='f_obs', format='E', unit='1e%derg/s/cm2/AA'%(np.log10(scale)), array=fybb[:] * c / np.square(xbb[:]) / d)
        col00.append(col6)
        col7  = fits.Column(name='e_obs', format='E', unit='1e%derg/s/cm2/AA'%(np.log10(scale)), array=eybb[:] * c / np.square(xbb[:]) / d)
        col00.append(col7)

        hdr = fits.Header()
        hdr['redshift'] = zbes
        hdr['id'] = ID
        hdr['hierarch isochrone'] = isochrone
        hdr['library'] = LIBRARY
        hdr['nimf'] = nimf
        hdr['scale'] = scale

        try:
            # Chi square:
            hdr['chi2'] = chi2
            hdr['hierarch No-of-effective-data-points'] = len(wht3[conw])
            hdr['hierarch No-of-nondetectioin'] = len(ey[con_up])
            hdr['hierarch Chi2-of-nondetection'] = chi_nd
            hdr['hierarch No-of-params'] = ndim_eff
            hdr['hierarch Degree-of-freedom']  = nod
            hdr['hierarch reduced-chi2'] = fin_chi2
        except:
            print('Chi seems to be wrong...')
            pass

        try:
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
        except:
            pass

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
                hdr['L_%s_16'%(ew_label[ii])] = L16[ii]
                hdr['L_%s_50'%(ew_label[ii])] = L50[ii]
                hdr['L_%s_84'%(ew_label[ii])] = L84[ii]
        except:
            pass

        # Version;
        import gsf
        hdr['version'] = gsf.__version__

        # Write;
        colspec = fits.ColDefs(col00)
        hdu0 = fits.BinTableHDU.from_columns(colspec, header=hdr)
        hdu0.writeto(MB.DIR_OUT + 'gsf_spec_%s.fits'%(ID), overwrite=True)

        # ASDF;
        tree_spec = {
            'id': ID,
            'redshift': '%.3f'%zbes,
            'isochrone': '%s'%(isochrone),
            'library': '%s'%(LIBRARY),
            'nimf': '%s'%(nimf),
            'scale': scale,
            'version_gsf': gsf.__version__
        }

        # BB;
        tree_spec.update({'wave': lbb})
        tree_spec.update({'fnu_16': fbb16_nu})
        tree_spec.update({'fnu_50': fbb_nu})
        tree_spec.update({'fnu_84': fbb84_nu})
        # full spectrum;
        tree_spec.update({'wave_model': x1_tot})
        tree_spec.update({'f_model_16': ytmp16})
        tree_spec.update({'f_model_50': ytmp50})
        tree_spec.update({'f_model_84': ytmp84})

        # EW;
        try:
            for ii in range(len(EW50)):
                tree_spec.update({'EW_%s_16'%(ew_label[ii]): EW16[ii]})
                tree_spec.update({'EW_%s_50'%(ew_label[ii]): EW50[ii]})
                tree_spec.update({'EW_%s_84'%(ew_label[ii]): EW84[ii]})
                tree_spec.update({'EW_%s_e1'%(ew_label[ii]): EW50_er1[ii]})
                tree_spec.update({'EW_%s_e2'%(ew_label[ii]): EW50_er2[ii]})
                tree_spec.update({'cnt_%s_16'%(ew_label[ii]): cnt16[ii]})
                tree_spec.update({'cnt_%s_50'%(ew_label[ii]): cnt50[ii]})
                tree_spec.update({'cnt_%s_84'%(ew_label[ii]): cnt84[ii]})
                tree_spec.update({'L_%s_16'%(ew_label[ii]): L16[ii]})
                tree_spec.update({'L_%s_50'%(ew_label[ii]): L50[ii]})
                tree_spec.update({'L_%s_84'%(ew_label[ii]): L84[ii]})
        except:
            pass

        # Each component
        # Stellar
        tree_spec.update({'wave_model_stel': x0})

        if f_sed_each:
            for aa in range(len(age)):
                tree_spec.update({'f_model_stel_%d'%aa: f_50_comp[aa,:]})

        if f_dust:
            # dust
            tree_spec.update({'wave_model_dust': x1_dust})
            tree_spec.update({'f_model_dust': ytmp_dust50})            
        # BB for dust
        tree_spec.update({'wave_obs': xbb})
        tree_spec.update({'f_obs': fybb[:] * c / np.square(xbb[:]) / d})
        tree_spec.update({'e_obs': eybb[:] * c / np.square(xbb[:]) / d})
        # grism:
        if f_grsm:
            tree_spec.update({'fg0_obs': fg0 * c/np.square(xg0)/d})
            tree_spec.update({'eg0_obs': eg0 * c/np.square(xg0)/d})
            tree_spec.update({'wg0_obs': xg0})
            tree_spec.update({'fg1_obs': fg1 * c/np.square(xg1)/d})
            tree_spec.update({'eg1_obs': eg1 * c/np.square(xg1)/d})
            tree_spec.update({'wg1_obs': xg1})
            tree_spec.update({'fg2_obs': fg2 * c/np.square(xg2)/d})
            tree_spec.update({'eg2_obs': eg2 * c/np.square(xg2)/d})
            tree_spec.update({'wg2_obs': xg2})

        af = asdf.AsdfFile(tree_spec)
        af.write_to(MB.DIR_OUT + 'gsf_spec_%s.asdf'%(ID), all_array_compression='zlib')

    #
    # SED params in plot
    #
    if f_label:
        fd = fits.open(MB.DIR_OUT + 'SFH_' + ID + '.fits')[0].header
        if f_dust:
            label = 'ID: %s\n$z_\mathrm{obs.}:%.2f$\n$\log M_*/M_\odot:%.2f$\n$\log M_\mathrm{dust}/M_\odot:%.2f$\n$T_\mathrm{dust}/K:%.1f$\n$\log Z_*/Z_\odot:%.2f$\n$\log T_0$/Gyr$:%.2f$\n$\log \\tau$/Gyr$:%.2f$\n$A_V$/mag$:%.2f$\n$\\chi^2/\\nu:%.2f$'\
            %(ID, zbes, float(fd['Mstel_50']), MD50, TD50, float(fd['Z_MW_50']), float(fd['T_MW_50']), float(fd['TAU_50']), float(fd['AV_50']), fin_chi2)
            ylabel = ymax*0.45
        else:
            label = 'ID: %s\n$z_\mathrm{obs.}:%.2f$\n$\log M_*/M_\odot:%.2f$\n$\log Z_*/Z_\odot:%.2f$\n$\log T_0$/Gyr$:%.2f$\n$\log \\tau$/Gyr$:%.2f$\n$A_V$/mag$:%.2f$\n$\\chi^2/\\nu:%.2f$'\
            %(ID, zbes, float(fd['Mstel_50']), float(fd['Z_MW_50']), float(fd['T_MW_50']), float(fd['TAU_50']), float(fd['AV_50']), fin_chi2)
            ylabel = ymax*0.4

        ax1.text(2200, ylabel, label,\
        fontsize=7, bbox=dict(facecolor='w', alpha=0.7), zorder=10)
        
    #######################################
    ax1.xaxis.labelpad = -3
    if f_grsm:
        conlim = (x0>10000) & (x0<25000)
        xgmin, xgmax = np.min(x0[conlim]),np.max(x0[conlim]), #7500, 17000
        ax2t.set_xlabel('')
        ax2t.set_xlim(xgmin, xgmax)

        conaa = (x0>xgmin-50) & (x0<xgmax+50)
        ymaxzoom = np.max(ysum[conaa]*c/np.square(x0[conaa])/d) * 1.15
        yminzoom = np.min(ysum[conaa]*c/np.square(x0[conaa])/d) / 1.15

        ax2t.set_ylim(yminzoom, ymaxzoom)
        ax2t.xaxis.labelpad = -2
        if xgmax>20000:
            ax2t.set_xticks([8000, 12000, 16000, 20000, 24000])
            ax2t.set_xticklabels(['0.8', '1.2', '1.6', '2.0', '2.4'])
        else:
            ax2t.set_xticks([8000, 10000, 12000, 14000, 16000])
            ax2t.set_xticklabels(['0.8', '1.0', '1.2', '1.4', '1.6'])

    if f_dust:
        try:
            contmp = (x1_tot>10*1e4)
            y3min, y3max = -.2*np.max((model_tot * c/ np.square(x1_tot) / d)[contmp]), np.max((model_tot * c/ np.square(x1_tot) / d)[contmp])*2.0
            ax3t.set_ylim(y3min, y3max)
        except:
            if verbose:
                print('y3 limit is not specified.')
            pass
        ax3t.set_xlim(1e5, 3e7)
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

                if ll == 2 and FLW[ii] == 1: 
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

    # Filters
    ind_remove = np.where((wht3<=0) | (ey<=0))[0]
    if f_plot_filter:
        ax1 = plot_filter(MB, ax1, ymax, scl=scl_yaxis, ind_remove=ind_remove)

    ####################
    ## Save
    ####################
    ax1.legend(loc=1, fontsize=11)
    if figpdf:
        fig.savefig(MB.DIR_OUT + 'SPEC_' + ID + '_spec.pdf', dpi=dpi)
    else:
        fig.savefig(MB.DIR_OUT + 'SPEC_' + ID + '_spec.png', dpi=dpi)


def plot_filter(MB, ax, ymax, scl=0.3, cmap='gist_rainbow', alp=0.4, ind_remove=[]):
    '''
    Add filter response curve to ax1.

    '''
    NUM_COLORS = len(MB.filts)
    cm = plt.get_cmap(cmap)
    cols = [cm(1 - 1.*i/NUM_COLORS) for i in range(NUM_COLORS)]

    wavecen = []
    for ii,filt in enumerate(MB.filts):
        wave = MB.band['%s_lam'%filt]
        flux = MB.band['%s_res'%filt]
        #wavecen.append(np.median(wave * flux)/np.median(flux))
        con = (flux/flux.max()>0.1)
        wavecen.append(np.min(wave[con]))
    wavecen = np.asarray(wavecen)
    wavecen_sort = np.sort(wavecen)

    for ii,filt in enumerate(MB.filts):
        iix = np.argmin(np.abs(wavecen_sort[:]-wavecen[ii]))
        col = cols[iix]
        wave = MB.band['%s_lam'%filt]
        flux = MB.band['%s_res'%filt]
        if ii in ind_remove:
            continue
        ax.plot(wave, ((flux / np.max(flux))*0.8 - 1) * ymax * scl, linestyle='-', color='k', lw=0.2)
        ax.fill_between(wave, (wave*0 - ymax)*scl, ((flux / np.max(flux))*0.8 - 1) * ymax * scl, linestyle='-', lw=0, color=col, alpha=alp)

    return ax


def plot_corner_physparam_summary(MB, fig=None, out_ind=0, DIR_OUT='./', mmax=300, TMIN=0.0001, tau_lim=0.01, f_plot_filter=True, 
    scale=1e-19, NRbb_lim=10000):
    '''
    Purpose
    -------
    For summary. In the same format as plot_corner_physparam_frame.

    Notes
    -----
    Tau model not supported.
    '''

    col = ['violet', 'indigo', 'b', 'lightblue', 'lightgreen', 'g', 'orange', 'coral', 'r', 'darkred']#, 'k']
    import matplotlib
    import matplotlib.cm as cm
    import scipy.stats as stats

    nage = MB.nage 
    fnc  = MB.fnc 
    bfnc = MB.bfnc 
    ID   = MB.ID
    Z    = MB.Zall
    age  = MB.age
    d = MB.d * scale
    c = MB.c

    tau0 = MB.tau0 
    dust_model = MB.dust_model
    DIR_TMP = MB.DIR_TMP

    Txmax = np.max(age) + 1.0 

    ###########################
    # Open result file
    ###########################
    lib = fnc.open_spec_fits(fall=0)
    lib_all = fnc.open_spec_fits(fall=1)

    file = MB.DIR_OUT + 'summary_' + ID + '.fits'
    hdul = fits.open(file) # open a FITS file

    # Redshift MC
    zp50  = hdul[1].data['zmc'][1]
    zp16  = hdul[1].data['zmc'][0]
    zp84  = hdul[1].data['zmc'][2]

    M50 = hdul[1].data['ms'][1]
    M16 = hdul[1].data['ms'][0]
    M84 = hdul[1].data['ms'][2]
    print('Total stellar mass is %.2e'%(M50))

    A50 = np.zeros(len(age), dtype='float')
    A16 = np.zeros(len(age), dtype='float')
    A84 = np.zeros(len(age), dtype='float')
    for aa in range(len(age)):
        A50[aa] = hdul[1].data['A'+str(aa)][1]
        A16[aa] = hdul[1].data['A'+str(aa)][0]
        A84[aa] = hdul[1].data['A'+str(aa)][2]

    Asum = np.sum(A50)
    aa = 0

    Av16 = hdul[1].data['Av'+str(aa)][0]
    Av50 = hdul[1].data['Av'+str(aa)][1]
    Av84 = hdul[1].data['Av'+str(aa)][2]

    Z50  = np.zeros(len(age), dtype='float')
    Z16  = np.zeros(len(age), dtype='float')
    Z84  = np.zeros(len(age), dtype='float')

    NZbest = np.zeros(len(age), dtype='int')
    for aa in range(len(age)):
        Z50[aa] = hdul[1].data['Z'+str(aa)][1]
        Z16[aa] = hdul[1].data['Z'+str(aa)][0]
        Z84[aa] = hdul[1].data['Z'+str(aa)][2]
        #NZbest[aa]= bfnc.Z2NZ(Z50[aa])

    ZZ50 = np.sum(Z50*A50)/np.sum(A50) # Light weighted Z.
    chi = hdul[1].data['chi'][0]
    chin = hdul[1].data['chi'][1]
    fitc = chin
    Cz0 = hdul[0].header['Cz0']
    Cz1 = hdul[0].header['Cz1']
    Cz2 = hdul[0].header['Cz2']
    zbes = hdul[0].header['z']
    zscl = (1.+zbes)

    # plot Configuration
    if MB.fzmc == 1:
        Par = ['$\log M_*/M_\odot$', '$\log T_*$/Gyr', '$A_V$/mag', '$\log Z_* / Z_\odot$', '$z$']
    else:
        Par = ['$\log M_*/M_\odot$', '$\log T_*$/Gyr', '$A_V$/mag', '$\log Z_* / Z_\odot$']

    K = len(Par) # No of params.
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
        MB.dict = MB.read_data(MB.Cz0, MB.Cz1, MB.Cz2, MB.zgal, add_fir=True)
    else:
        MB.dict = MB.read_data(MB.Cz0, MB.Cz1, MB.Cz2, MB.zgal)

    # Get data points;
    NRbb = MB.dict['NRbb']
    xbb = MB.dict['xbb'] 
    fybb = MB.dict['fybb']
    eybb = MB.dict['eybb']
    exbb = MB.dict['exbb']
    snbb = fybb/eybb

    # Get spec data points;
    dat = np.loadtxt(DIR_TMP + 'spec_obs_' + ID + '.cat', comments='#')
    NR  = dat[:,0]
    x   = dat[:,1]
    fy00 = dat[:,2]
    ey00 = dat[:,3]

    con0 = (NR<1000) #& (fy/ey>SNlim)
    xg0  = x[con0]
    fg0  = fy00[con0] * Cz0
    eg0  = ey00[con0] * Cz0
    con1 = (NR>=1000) & (NR<2000) #& (fy/ey>SNlim)
    xg1  = x[con1]
    fg1  = fy00[con1] * Cz1
    eg1  = ey00[con1] * Cz1
    con2 = (NR>=2000) & (NR<NRbb_lim) #& (fy/ey>SNlim)
    xg2  = x[con2]
    fg2  = fy00[con2] * Cz2
    eg2  = ey00[con2] * Cz2

    con_bb = (NR>=NRbb_lim)
    xg_bb = x[con_bb]
    fg_bb = fy00[con_bb]
    eg_bb = ey00[con_bb]

    fy01 = np.append(fg0,fg1)
    ey01 = np.append(eg0,eg1)
    fy02 = np.append(fy01,fg2)
    ey02 = np.append(ey01,eg2)

    fy = np.append(fy01,fg_bb)
    ey = np.append(ey01,eg_bb)
    wht = 1./np.square(ey)

    # BB photometry
    conspec = (NR<NRbb_lim)
    sigma = 1.
    conbb = (fybb/eybb > sigma)
    ax0.errorbar(xbb[conbb], fybb[conbb] * c / np.square(xbb[conbb]) / d, yerr=eybb[conbb]*c/np.square(xbb[conbb])/d, color='k', linestyle='', linewidth=0.5, zorder=4)
    ax0.plot(xbb[conbb], fybb[conbb] * c / np.square(xbb[conbb]) / d, '.r', ms=10, linestyle='', linewidth=0, zorder=4)

    conebb_ls = (fybb/eybb <= sigma)
    leng = np.max(fybb[conebb_ls] * c / np.square(xbb[conebb_ls]) / d) * 0.05
    ax0.errorbar(xbb[conebb_ls], eybb[conebb_ls] * c / np.square(xbb[conebb_ls]) / d * sigma, yerr=leng,\
        uplims=eybb[conebb_ls] * c / np.square(xbb[conebb_ls]) / d * sigma, linestyle='',color='r', marker='', ms=4, label='', zorder=4, capsize=3)
    
    ####################
    # MCMC corner plot.
    ####################
    file  = MB.DIR_OUT + 'chain_' + ID + '_corner.cpkl'
    niter = 0
    data  = loadcpkl(file)

    try:
        ndim   = data['ndim']     # By default, use ndim and burnin values contained in the cpkl file, if present.
        burnin = data['burnin']
        nmc    = data['niter']
        nwalk  = data['nwalkers']
        Nburn  = burnin
        samples = data['chain'][:]
    except:
        if verbose: print(' =   >   NO keys of ndim and burnin found in cpkl, use input keyword values')

    af = MB.af
    sedpar = af['ML']

    getcmap = matplotlib.cm.get_cmap('jet')
    nc = np.arange(0, nmc, 1)
    col = getcmap((nc-0)/(nmc-0))

    #for kk in range(0,nmc,1):
    Ntmp = np.zeros(mmax, dtype='float')
    lmtmp= np.zeros(mmax, dtype='float')
    Avtmp= np.zeros(mmax, dtype='float')
    Ztmp = np.zeros(mmax, dtype='float')
    Ttmp = np.zeros(mmax, dtype='float')
    ACtmp= np.zeros(mmax, dtype='float')
    redshifttmp = np.zeros(mmax, dtype='float')

    # Time bin
    Tuni = MB.cosmo.age(zbes).value
    Tuni0 = (Tuni - age[:])
    delT  = np.zeros(len(age),dtype='float')
    delTl = np.zeros(len(age),dtype='float')
    delTu = np.zeros(len(age),dtype='float')

    if len(age) == 1:
        for aa in range(len(age)):
            try:
                tau_ssp = float(MB.inputs['TAU_SSP'])
            except:
                tau_ssp = tau_lim
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

            if delT[aa] < tau_lim:
                # This is because fsps has the minimum tau = tau_lim
                delT[aa] = tau_lim


    delT[:]  *= 1e9 # Gyr to yr
    delTl[:] *= 1e9 # Gyr to yr
    delTu[:] *= 1e9 # Gyr to yr

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

        if aa == 0 or MB.ZEVOL:
            ZZ_tmp = Z50[ii]
            ZZ_tmp16 = Z16[ii]
            ZZ_tmp84 = Z84[ii]
        else:
            ZZ_tmp = Z50[0]
            ZZ_tmp16 = Z16[0]
            ZZ_tmp84 = Z84[0]
        
        try:
            AA_tmp = 10**np.max(samples['A'+str(ii)][:])
            AA_tmp84 = 10**np.percentile(samples['A'+str(ii)][:],95)
            AA_tmp16 = 10**np.percentile(samples['A'+str(ii)][:],5)
        except:
            AA_tmp = 0
            AA_tmp84 = 0
            AA_tmp16 = 0

        nZtmp = bfnc.Z2NZ(ZZ_tmp)
        mslist = sedpar['ML_'+str(nZtmp)][ii]

        AMtmp16 += mslist*AA_tmp16
        AMtmp84 += mslist*AA_tmp84

        Tsmax += age[ii] * AA_tmp84 * mslist
        Tsmin += age[ii] * AA_tmp16 * mslist

        Zsmax += 10**ZZ_tmp84 * AA_tmp84 * mslist
        Zsmin += 10**ZZ_tmp16 * AA_tmp16 * mslist

        SFtmp = AA_tmp * mslist / delT[ii]
        if SFtmp > SFmax:
            SFmax = SFtmp
    if SFmax > 0.5e4:
        SFmax = 0.5e4

    delM = np.log10(M84) - np.log10(M16)
    if MB.fzmc == 1:
        NPARmin = [np.log10(M16)-.1, np.log10(Tsmin/AMtmp16)-0.1, Av16-0.1, np.log10(Zsmin/AMtmp16)-0.2, np.percentile(samples['zmc'],1)-0.1]
        NPARmax = [np.log10(M84)+.1, np.log10(Tsmax/AMtmp84)+0.2, Av84+0.1, np.log10(Zsmax/AMtmp84)+0.2, np.percentile(samples['zmc'],99)+0.1]
    else:
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

    Tzz   = np.zeros(len(zred), dtype='float')
    for zz in range(len(zred)):
        Tzz[zz] = (Tuni - MB.cosmo.age(zred[zz]).value) #/ cc.Gyr_s
        if Tzz[zz] < TMIN:
            Tzz[zz] = TMIN


    def density_estimation(m1, m2):
        xmin, xmax = np.min(m1), np.max(m1)
        ymin, ymax = np.min(m2), np.max(m2)
        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([m1, m2])
        kernel = stats.gaussian_kde(values)
        Z = np.reshape(kernel(positions).T, X.shape)
        return X, Y, Z


    for kk in range(0,mmax,1):
        nr = np.random.randint(len(samples))
        try:
            Avtmp[kk] = samples['Av'][nr]
        except:
            Avtmp[kk] = MB.AVFIX

        ZMM = np.zeros((len(age)), dtype='float') # Mass weighted Z.
        ZM = np.zeros((len(age)), dtype='float') # Light weighted T.
        ZC = np.zeros((len(age)), dtype='float') # Light weighted T.
        SF = np.zeros((len(age)), dtype='float') # SFR
        AM = np.zeros((len(age)), dtype='float') # Light weighted T.
        II0 = nage

        for ss in range(len(age)):
            ii = int(len(II0) - ss - 1) # from old to young templates.
            
            try:
                AA_tmp = 10**samples['A'+str(ii)][nr]
            except:
                AA_tmp = 0
                pass

            try:
                ZZ_tmp = samples['Z'+str(ii)][nr]
            except:
                try:
                    ZZ_tmp = samples['Z0'][nr]
                except:
                    ZZ_tmp = MB.ZFIX

            nZtmp = bfnc.Z2NZ(ZZ_tmp)
            mslist = sedpar['ML_'+str(nZtmp)][ii]
            lmtmp[kk] += AA_tmp * mslist
            Ztmp[kk] += (10 ** ZZ_tmp) * AA_tmp * mslist
            Ttmp[kk] += age[ii] * AA_tmp * mslist
            ACtmp[kk] += AA_tmp * mslist
            if MB.fzmc == 1:
                redshifttmp[kk] = samples['zmc'][nr]

            AM[ii] = AA_tmp * mslist
            SF[ii] = AA_tmp * mslist / delT[ii]
            ZM[ii] = ZZ_tmp
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
            if AC > 0:
                ZC[ss] = np.log10(np.sum(ZMM[ss:])/AC)
            else:
                ZC[ss] = -99

        # Plot Total
        ax0.plot(x0, ysum * c/ np.square(x0) / d, '-', lw=.1, color='gray', zorder=-1, label='', alpha=0.1)

        if len(age)==1:
            ax1.plot(age[:], SF[:], marker='.', linestyle='-', lw=.1, color='k', zorder=-1, label='', alpha=0.01)
            ax2.plot(age[:], ZC[:], marker='.', linestyle='-', lw=.1, color='k', zorder=-1, label='', alpha=0.01)
        else:
            ax1.plot(age[:], SF[:], marker='', linestyle='-', lw=.1, color='k', zorder=-1, label='', alpha=0.1)
            ax2.plot(age[:], ZC[:], marker='', linestyle='-', lw=.1, color='k', zorder=-1, label='', alpha=0.1)

        # Get ymax
        if f_plot_filter:
            scl_yaxis = 0.2
        else:
            scl_yaxis = 0

        ymax_bb = np.max(fybb[conbb] * c / np.square(xbb[conbb]) / d) * 1.10
        ymax_temp = np.max(ysum * c/ np.square(x0) / d) * 1.10
        ymax = np.max([ymax_bb, ymax_temp])

        # Convert into log
        Ztmp[kk] /= ACtmp[kk]
        Ttmp[kk] /= ACtmp[kk]
        Ntmp[kk]  = kk

        lmtmp[kk] = np.log10(lmtmp[kk])
        Ztmp[kk]  = np.log10(Ztmp[kk])
        Ttmp[kk]  = np.log10(Ttmp[kk])

        if MB.fzmc == 1:
            NPAR = [lmtmp[:kk+1], Ttmp[:kk+1], Avtmp[:kk+1], Ztmp[:kk+1], redshifttmp[:kk+1]]
        else:
            NPAR = [lmtmp[:kk+1], Ttmp[:kk+1], Avtmp[:kk+1], Ztmp[:kk+1]]

        if kk == mmax-1:
            # Histogram
            for i, x in enumerate(Par):
                ax = axes[i, i]
                x1min, x1max = NPARmin[i], NPARmax[i]
                nbin = 50
                binwidth1 = (x1max-x1min)/nbin
                bins1 = np.arange(x1min, x1max + binwidth1, binwidth1)
                n, bins, patches = ax.hist(NPAR[i], bins=bins1, orientation='vertical', color='b', histtype='stepfilled', alpha=0.6)
                yy = np.arange(0,np.max(n)*1.3,1)

                try:
                    ax.plot(yy*0+np.percentile(NPAR[i],16), yy, linestyle='--', color='gray', lw=1)
                    ax.plot(yy*0+np.percentile(NPAR[i],84), yy, linestyle='--', color='gray', lw=1)
                    ax.plot(yy*0+np.percentile(NPAR[i],50), yy, linestyle='-', color='gray', lw=1)

                    ax.text(np.percentile(NPAR[i],16), np.max(yy)*1.02, '%.2f'%(np.percentile(NPAR[i],16)), fontsize=9)
                    ax.text(np.percentile(NPAR[i],50), np.max(yy)*1.02, '%.2f'%(np.percentile(NPAR[i],50)), fontsize=9)
                    ax.text(np.percentile(NPAR[i],84), np.max(yy)*1.02, '%.2f'%(np.percentile(NPAR[i],84)), fontsize=9)
                except:
                    print('Failed at i,x=',i,x)

                ax.set_xlim(x1min, x1max)
                ax.set_yticklabels([])
                if i == K-1:
                    ax.set_xlabel('%s'%(Par[i]), fontsize=12)
                if i < K-1:
                    ax.set_xticklabels([])

        # Scatter and contour
        for i, x in enumerate(Par):
            for j, y in enumerate(Par):
                if i > j:
                    ax = axes[i, j]
                    ax.scatter(NPAR[j], NPAR[i], c='b', s=1, marker='o', alpha=0.01)
                    ax.set_xlabel('%s'%(Par[j]), fontsize=12)

                    if kk == mmax-1:
                        try:
                            Xcont, Ycont, Zcont = density_estimation(NPAR[j], NPAR[i])
                            mZ = np.max(Zcont)
                            ax.contour(Xcont, Ycont, Zcont, levels=[0.68*mZ, 0.95*mZ, 0.99*mZ], linewidths=[0.8,0.5,0.3], colors='gray')
                        except:
                            print('Error occurs when density estimation. Maybe because some params are fixed.')
                            pass

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
            fname = MB.DIR_OUT +  + '%d.png' % kk
            print('Saving frame', fname)
            plt.savefig(fname, dpi=200)
            files.append(fname)

    # For the last one
    ax0.plot(xg0, fg0 * c / np.square(xg0) / d, marker='', linestyle='-', linewidth=0.5, ms=0.1, color='royalblue', label='')
    ax0.plot(xg1, fg1 * c / np.square(xg1) / d, marker='', linestyle='-', linewidth=0.5, ms=0.1, color='g', label='')
    ax0.plot(xg2, fg2 * c / np.square(xg2) / d, marker='', linestyle='-', linewidth=0.5, ms=0.1, color='#DF4E00', label='')

    ax0.set_xlim(2200, 88000)
    ax0.set_xscale('log')
    ax0.set_ylim(-ymax * scl_yaxis, ymax)
    ax0.set_xlabel('Observed wavelength ($\mathrm{\mu m}$)', fontsize=14)
    ax0.set_ylabel('Flux ($10^{%d}\mathrm{erg}/\mathrm{s}/\mathrm{cm}^{2}/\mathrm{\AA}$)'%(np.log10(scale)),fontsize=12,labelpad=-2)
    ax1.set_xlabel('$t$ (Gyr)', fontsize=12)
    ax1.set_ylabel('$\dot{M_*}/M_\odot$yr$^{-1}$', fontsize=12)
    ax1.set_xlim(np.min(age)*0.8, Txmax)
    ax1.set_ylim(0, SFmax)
    ax1.set_xscale('log')
    ax2.set_xlabel('$t$ (Gyr)', fontsize=12)
    ax2.set_ylabel('$\log Z_*/Z_\odot$', fontsize=12)
    ax2.set_xlim(np.min(age)*0.8, Txmax)
    if round(np.min(Z),2) == round(np.max(Z),2):
        ax2.set_ylim(-0.8, 0.5)
    else:
        ax2.set_ylim(np.min(Z)-0.05, np.max(Z)+0.05)
    ax2.set_xscale('log')
    #ax2.yaxis.labelpad = -5

    ax1t = ax1.twiny()
    ax2t = ax2.twiny()
    ax1t.set_xlim(0.008, Txmax)
    ax1t.set_xscale('log')

    ax1t.xaxis.set_major_locator(ticker.FixedLocator(Tzz[:]))
    ax1t.xaxis.set_major_formatter(ticker.FixedFormatter(zredl[:]))
    #ax1t.set_xticklabels(zredl[:])
    #ax1t.set_xticks(Tzz[:])

    ax1t.tick_params(axis='x', labelcolor='k')
    ax1t.xaxis.set_ticks_position('none')
    ax1.plot(Tzz, Tzz*0+SFmax, marker='|', color='k', ms=3, linestyle='None')

    ax2t.set_xlim(0.008, Txmax)
    ax2t.set_xscale('log')
    #ax2t.set_xticklabels(zredl[:])
    #ax2t.set_xticks(Tzz[:])
    ax2t.xaxis.set_major_locator(ticker.FixedLocator(Tzz[:]))
    ax2t.xaxis.set_major_formatter(ticker.FixedFormatter(zredl[:]))

    ax2t.tick_params(axis='x', labelcolor='k')
    ax2t.xaxis.set_ticks_position('none')
    ax2.plot(Tzz, Tzz*0+0.5, marker='|', color='k', ms=3, linestyle='None')

    # Filters
    ind_remove = np.where((wht<=0) | (ey<=0))[0]
    if f_plot_filter:
        ax1 = plot_filter(MB, ax1, ymax, scl=scl_yaxis, ind_remove=ind_remove)
        xx = np.arange(2200,100000,100)
        ax0.plot(xx, xx * 0, linestyle='--', lw=0.5, color='k')

    plt.savefig(MB.DIR_OUT + 'param_' + ID + '_corner.png', dpi=150)


def plot_corner_physparam_cumulative_frame(ID, Zall=np.arange(-1.2,0.4249,0.05), age=[0.01, 0.1, 0.3, 0.7, 1.0, 3.0], \
    tau0=[0.1,0.2,0.3], fig=None, dust_model=0, out_ind=0, snlimbb=1.0, DIR_OUT='./', NRbb_lim=10000):
    '''
    Creat "cumulative" png for gif image.
    
    If you like to creat temporal png for gif image.

    Parameters
    ----------
    snlimbb : float
        SN limit to show flux or up lim in SED.
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

    file = 'summary_' + ID + '.fits'
    hdul = fits.open(file) # open a FITS file

    # Redshift MC
    zp50  = hdul[1].data['zmc'][1]
    zp16  = hdul[1].data['zmc'][0]
    zp84  = hdul[1].data['zmc'][2]

    M50 = hdul[1].data['ms'][1]
    M16 = hdul[1].data['ms'][0]
    M84 = hdul[1].data['ms'][2]
    print('Total stellar mass is %.2e'%(M50))

    A50 = np.zeros(len(age), dtype='float')
    A16 = np.zeros(len(age), dtype='float')
    A84 = np.zeros(len(age), dtype='float')
    for aa in range(len(age)):
        A50[aa] = hdul[1].data['A'+str(aa)][1]
        A16[aa] = hdul[1].data['A'+str(aa)][0]
        A84[aa] = hdul[1].data['A'+str(aa)][2]

    Asum  = np.sum(A50)
    aa   = 0
    Av50 = hdul[1].data['Av'+str(aa)][1]
    Av16 = hdul[1].data['Av'+str(aa)][0]
    Av84 = hdul[1].data['Av'+str(aa)][2]
    Z50 = np.zeros(len(age), dtype='float')
    Z16 = np.zeros(len(age), dtype='float')
    Z84 = np.zeros(len(age), dtype='float')
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
    Cz2   = hdul[0].header['Cz2']
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
    dat = np.loadtxt(DIR_TMP + 'spec_obs_' + ID + '.cat', comments='#')
    NR = dat[:, 0]
    x  = dat[:, 1]
    fy00 = dat[:, 2]
    ey00 = dat[:, 3]

    con0 = (NR<1000) #& (fy/ey>SNlim)
    xg0  = x[con0]
    fg0  = fy00[con0] * Cz0
    eg0  = ey00[con0] * Cz0
    con1 = (NR>=1000) & (NR<2000) #& (fy/ey>SNlim)
    xg1  = x[con1]
    fg1  = fy00[con1] * Cz1
    eg1  = ey00[con1] * Cz1
    con2 = (NR>=2000) & (NR<NRbb_lim) #& (fy/ey>SNlim)
    xg2  = x[con2]
    fg2  = fy00[con2] * Cz2
    eg2  = ey00[con2] * Cz2

    con_bb = (NR>=NRbb_lim)#& (fy/ey>SNlim)
    xg_bb  = x[con_bb]
    fg_bb  = fy00[con_bb]
    eg_bb  = ey00[con_bb]

    fy01 = np.append(fg0,fg1)
    ey01 = np.append(eg0,eg1)
    fy02 = np.append(fy01,fg2)
    ey02 = np.append(ey01,eg2)

    fy = np.append(fy02,fg_bb)
    ey = np.append(ey02,eg_bb)
    wht=1./np.square(ey)

    dat = np.loadtxt(DIR_TMP + 'bb_obs_' + ID + '.cat', comments='#')
    NRbb = dat[:,0]
    xbb  = dat[:,1]
    fybb = dat[:,2]
    eybb = dat[:,3]
    exbb = dat[:,4]
    snbb = fybb/eybb

    conspec = (NR<NRbb_lim) #& (fy/ey>1)
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
    file = 'chain_' + ID + '_corner.cpkl'
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

    f0     = fits.open(DIR_TMP + 'ms_' + ID + '.fits')
    sedpar = f0[1]

    import matplotlib
    import matplotlib.cm as cm
    getcmap   = matplotlib.cm.get_cmap('jet')
    nc        = np.arange(0, nmc, 1)
    col = getcmap((nc-0)/(nmc-0))

    #for kk in range(0,nmc,1):
    Ntmp = np.zeros(nplot, dtype='float')
    lmtmp= np.zeros(nplot, dtype='float')
    Avtmp= np.zeros(nplot, dtype='float')
    Ztmp = np.zeros(nplot, dtype='float')
    Ttmp = np.zeros(nplot, dtype='float')
    ACtmp= np.zeros(nplot, dtype='float')

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
                y0p, x0p  = fnc.tmp03(AA_tmp, Avtmp[kk], ii, ZZ_tmp, zbes, lib, tau0=tau0)
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


        NPAR = [lmtmp[:kk+1], Ttmp[:kk+1], Avtmp[:kk+1], Ztmp[:kk+1]]
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
                ax.set_yticklabels([])
                if i == K-1:
                    ax.set_xlabel('%s'%(Par[i]), fontsize=12)
                if i < K-1:
                    ax.set_xticklabels([])

        # Scatter and contour
        for i, x in enumerate(Par):
            for j, y in enumerate(Par):
                if i > j:
                    ax = axes[i, j]
                    ax.scatter(NPAR[j], NPAR[i], c='b', s=1, marker='o', alpha=0.01)
                    ax.set_xlabel('%s'%(Par[j]), fontsize=12)

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

    plt.savefig(DIR_OUT + 'param_' + ID + '_corner.png', dpi=200)
    plt.close()


def write_lines(ID, zbes, R_grs=45, dw=4, umag=1.0, DIR_OUT='./'):
    '''
    '''
    dlw = R_grs * dw # Can affect the SFR.
    ldw = 7

    ###################################
    # To add lines in the plot,
    # ,manually edit the following file
    # so as Fcont50 have >0.
    ###################################
    flw = open(DIR_OUT + ID + '_lines_fit.txt', 'w')
    flw.write('# LW flux_line eflux_line flux_cont EW eEW L_line eL_line\n')
    flw.write('# (AA) (Flam_1e-18) (Flam_1e-18) (Flam_1e-18) (AA) (AA) (erg/s) (erg/s)\n')
    flw.write('# Error in EW is 1sigma, by pm eflux_line.\n')
    flw.write('# If EW=-99, it means gaussian fit failed.\n')
    flw.write('# and flux is the sum of excess at WL pm %.1f AA.\n'%(dlw))
    flw.write('# Magnification is corrected; mu=%.3f\n'%(umag))
    try:
        fl = np.loadtxt(DIR_OUT + 'table_' + ID + '_lines.txt', comments='#')
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

                flux = np.zeros(len(xxs), dtype='float')
                efl  = np.zeros(len(xxs), dtype='float')
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


def plot_corner_physparam_frame(ID, PA, Zall=np.arange(-1.2,0.4249,0.05), age=[0.01, 0.1, 0.3, 0.7, 1.0, 3.0], \
    tau0=[0.1,0.2,0.3], fig=None, dust_model=0):
    '''
    If you like to
    Creat temporal png for gif image.
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

    file = 'summary_' + ID + '.fits'
    hdul = fits.open(file) # open a FITS file

    # Redshift MC
    zp50  = hdul[1].data['zmc'][1]
    zp16  = hdul[1].data['zmc'][0]
    zp84  = hdul[1].data['zmc'][2]

    M50 = hdul[1].data['ms'][1]
    M16 = hdul[1].data['ms'][0]
    M84 = hdul[1].data['ms'][2]
    print('Total stellar mass is %.2e'%(M50))

    A50 = np.zeros(len(age), dtype='float')
    A16 = np.zeros(len(age), dtype='float')
    A84 = np.zeros(len(age), dtype='float')
    for aa in range(len(age)):
        A50[aa] = hdul[1].data['A'+str(aa)][1]
        A16[aa] = hdul[1].data['A'+str(aa)][0]
        A84[aa] = hdul[1].data['A'+str(aa)][2]

    Asum  = np.sum(A50)
    aa   = 0
    Av50 = hdul[1].data['Av'+str(aa)][1]
    Av16 = hdul[1].data['Av'+str(aa)][0]
    Av84 = hdul[1].data['Av'+str(aa)][2]
    Z50 = np.zeros(len(age), dtype='float')
    Z16 = np.zeros(len(age), dtype='float')
    Z84 = np.zeros(len(age), dtype='float')
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
    dat = np.loadtxt(DIR_TMP + 'spec_obs_' + ID + '.cat', comments='#')
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

    dat = np.loadtxt(DIR_TMP + 'bb_obs_' + ID + '.cat', comments='#')
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
    file = 'chain_' + ID + '_corner.cpkl'
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

    f0     = fits.open(DIR_TMP + 'ms_' + ID + '.fits')
    sedpar = f0[1]

    import matplotlib
    import matplotlib.cm as cm
    getcmap   = matplotlib.cm.get_cmap('jet')
    nc        = np.arange(0, nmc, 1)
    col = getcmap((nc-0)/(nmc-0))

    #for kk in range(0,nmc,1):
    Ntmp = np.zeros(nplot, dtype='float')
    lmtmp= np.zeros(nplot, dtype='float')
    Avtmp= np.zeros(nplot, dtype='float')
    Ztmp = np.zeros(nplot, dtype='float')
    Ttmp = np.zeros(nplot, dtype='float')
    ACtmp= np.zeros(nplot, dtype='float')


    # Time bin
    Txmax = 4 # Max x value
    Tuni = MB.cosmo.age(zbes).value
    Tuni0 = (Tuni - age[:])
    delT  = np.zeros(len(age),dtype='float')
    delTl = np.zeros(len(age),dtype='float')
    delTu = np.zeros(len(age),dtype='float')
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
        ZMM = np.zeros((len(age)), dtype='float') # Mass weighted Z.
        ZM  = np.zeros((len(age)), dtype='float') # Light weighted T.
        ZC  = np.zeros((len(age)), dtype='float') # Light weighted T.
        SF  = np.zeros((len(age)), dtype='float') # SFR
        AM  = np.zeros((len(age)), dtype='float') # Light weighted T.


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

        Tzz   = np.zeros(len(zred), dtype='float')
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

        NPAR    = [lmtmp[kk], Ttmp[kk], Avtmp[kk], Ztmp[kk]]

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
