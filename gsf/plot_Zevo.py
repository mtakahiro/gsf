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

#Zset  = [-0.40, 0.0, 0.4]#, 0.20, 0.35]
#Zset  = [-0.30, -0.20]#, 0.20, 0.35]
lcb = '#4682b4' # line color, blue
col = ['darkred', 'r', 'coral','orange','g','lightgreen', 'lightblue', 'b','indigo','violet','k']

def make_rgb(id0,xcen,ycen):
    fits_dir = '/Volumes/EHDD1/3DHST/IMG/'
    blue_fn  = fits_dir + 'goodsn/goodsn_3dhst.v4.0.F850LP_orig_sci.fits'
    green_fn = fits_dir + 'goodsn/goodsn_3dhst.v4.0.F125W_orig_sci.fits'
    red_fn   = fits_dir + 'goodsn_3dhst_v4.0_f160w/goodsn_3dhst_v4.0_f160w/goodsn_3dhst.v4.0.F160W_orig_sci.fits'

    # Parameters
    sig_fract = 5.0
    per_fract = 5.0-4
    max_iter = 5e1
    min_val = 0.
    max_val = .8
    non_linear_fact = .01
    axis_tag = True

    Rsq = 20.

    mb0 = 25.
    mg0 = 25.
    mr0 = 25.

    ymin = int(ycen - Rsq)
    ymax = int(ycen + Rsq)
    xmin = int(xcen - Rsq)
    xmax = int(xcen + Rsq)

    # Blue image
    hdulist = fits.open(blue_fn)
    img_header = hdulist[0].header
    img_data = hdulist[0].data[ymin:ymax +1,xmin:xmax +1]
    hdulist.close()
    width=img_data.shape[0]
    height=img_data.shape[1]
    #print "Blue file = ", blue_fn, "(", width, ",", height, ")"
    img_data_b = np.array(img_data, dtype=float)
    sky, num_iter = img_scale.sky_median_sig_clip(img_data_b, sig_fract, per_fract, max_iter, low_cut=False, high_cut=True)
    img_data_b = img_data_b - sky
    img_data_b *= 10**((25-mb0)/-2.5)
    rgb_array = np.empty((width,height,3), dtype=float)
    b = img_scale.asinh(img_data_b, scale_min = min_val, scale_max = max_val, non_linear=non_linear_fact)

    # Green image
    hdulist = fits.open(green_fn)
    img_header = hdulist[0].header
    img_data = hdulist[0].data[ymin:ymax +1,xmin:xmax +1]
    hdulist.close()
    width=img_data.shape[0]
    height=img_data.shape[1]
    #print "Green file = ", green_fn, "(", width, ",", height, ")"
    img_data_g = np.array(img_data, dtype=float)
    img_data_g *= 10**((25-mg0)/-2.5)
    sky, num_iter = img_scale.sky_median_sig_clip(img_data_g, sig_fract, per_fract, max_iter, low_cut=False, high_cut=True)
    img_data_g = img_data_g - sky
    g = img_scale.asinh(img_data_g, scale_min = min_val, scale_max = max_val, non_linear=non_linear_fact)

    # Red image
    hdulist = fits.open(red_fn)
    img_header = hdulist[0].header
    img_data = hdulist[0].data[ymin:ymax +1,xmin:xmax +1]
    hdulist.close()
    width=img_data.shape[0]
    height=img_data.shape[1]
    #print "Red file = ", red_fn, "(", width, ",", height, ")"
    img_data_r = np.array(img_data, dtype=float)
    img_data_r *= 10**((25-mr0)/-2.5)
    sky, num_iter = img_scale.sky_median_sig_clip(img_data_r, sig_fract, per_fract, max_iter, low_cut=False, high_cut=True)
    img_data_r = img_data_r - sky
    r = img_scale.asinh(img_data_r, scale_min = min_val, scale_max = max_val, non_linear=non_linear_fact)

    # RGB image with Matplotlib
    rgb_array[:,:,0] = r
    rgb_array[:,:,1] = g
    rgb_array[:,:,2] = b

    return rgb_array

###############
import pickle
def loadcpkl(cpklfile):
    """
    Load cpkl files.
    """
    if not os.path.isfile(cpklfile):
        raise ValueError(' ERR: cannot find the input file')
    f    = open(cpklfile, 'rb')#, encoding='ISO-8859-1')

    if sys.version_info.major == 2:
        data = pickle.load(f)
    elif sys.version_info.major == 3:
        data = pickle.load(f, encoding='latin-1')

    f.close()
    return data

def plot_corner_TZ(ID, PA, Zall=np.arange(-1.2,0.4249,0.05), age=[0.01, 0.1, 0.3, 0.7, 1.0, 3.0]):
    nage = np.arange(0,len(age),1)
    fnc  = Func(Zall, age, dust_model=dust_model) # Set up the number of Age/ZZ
    bfnc = Basic(Zall)

    fig = plt.figure(figsize=(3,3))
    fig.subplots_adjust(top=0.96, bottom=0.14, left=0.2, right=0.96, hspace=0.15, wspace=0.25)
    ax1 = fig.add_subplot(111)
    #ax2 = fig.add_subplot(132)
    #ax3 = fig.add_subplot(133)

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
    Ntmp = np.zeros(nmc, dtype='float32')
    Avtmp= np.zeros(nmc, dtype='float32')
    Ztmp = np.zeros(nmc, dtype='float32')
    Ttmp = np.zeros(nmc, dtype='float32')
    ACtmp= np.zeros(nmc, dtype='float32')


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

# Creat "cumulative" png for gif image.
def plot_corner_param(ID, PA, Zall=np.arange(-1.2,0.4249,0.05), age=[0.01, 0.1, 0.3, 0.7, 1.0, 3.0], tau0=[0.1,0.2,0.3], fig=None, out_ind=0, snlimbb=1.0):
    #
    # Returns: plots.
    #
    # snlimbb: SN limit to show flux or up lim in SED.
    #
    nage = np.arange(0,len(age),1)
    fnc  = Func(Zall, age, dust_model=dust_model) # Set up the number of Age/ZZ
    bfnc = Basic(Zall)

    ###########################
    # Open result file
    ###########################
    # Open ascii file and stock to array.
    lib     = fnc.open_spec_fits(ID, PA, fall=0, tau0=tau0)
    lib_all = fnc.open_spec_fits(ID, PA, fall=1, tau0=tau0)

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

    A50 = np.zeros(len(age), dtype='float32')
    A16 = np.zeros(len(age), dtype='float32')
    A84 = np.zeros(len(age), dtype='float32')
    for aa in range(len(age)):
        A50[aa] = hdul[1].data['A'+str(aa)][1]
        A16[aa] = hdul[1].data['A'+str(aa)][0]
        A84[aa] = hdul[1].data['A'+str(aa)][2]

    Asum  = np.sum(A50)
    aa   = 0
    Av50 = hdul[1].data['Av'+str(aa)][1]
    Av16 = hdul[1].data['Av'+str(aa)][0]
    Av84 = hdul[1].data['Av'+str(aa)][2]
    Z50 = np.zeros(len(age), dtype='float32')
    Z16 = np.zeros(len(age), dtype='float32')
    Z84 = np.zeros(len(age), dtype='float32')
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
    Ntmp = np.zeros(nplot, dtype='float32')
    lmtmp= np.zeros(nplot, dtype='float32')
    Avtmp= np.zeros(nplot, dtype='float32')
    Ztmp = np.zeros(nplot, dtype='float32')
    Ttmp = np.zeros(nplot, dtype='float32')
    ACtmp= np.zeros(nplot, dtype='float32')
    col = ['darkred', 'r', 'coral','orange','g','lightgreen', 'lightblue', 'b','indigo','violet','k']

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
            ZZ_tmp = samples['Z'+str(ii)][nr]

            nZtmp      = bfnc.Z2NZ(ZZ_tmp)
            mslist     = sedpar.data['ML_'+str(nZtmp)][ii]
            lmtmp[kk] += AA_tmp * mslist
            Ztmp[kk]  += (10 ** ZZ_tmp) * AA_tmp * mslist
            Ttmp[kk]  += age[ii] * AA_tmp * mslist
            ACtmp[kk] += AA_tmp * mslist

            # SED
            flim = 0.05
            if ss == 0:
                y0, x0   = fnc.tmp03(ID, PA, AA_tmp, Avtmp[kk], ii, ZZ_tmp, zbes, lib_all, tau0=tau0)
                y0p, x0p = fnc.tmp03(ID, PA, AA_tmp, Avtmp[kk], ii, ZZ_tmp, zbes, lib, tau0=tau0)
                ysump = y0p #* 1e18
                ysum  = y0  #* 1e18
                if AA_tmp/Asum > flim:
                    ax0.plot(x0, y0 * c/ np.square(x0) / d, '--', lw=0.1, color=col[ii], zorder=-1, label='', alpha=0.1)
            else:
                y0_r, x0_tmp = fnc.tmp03(ID, PA, AA_tmp, Avtmp[kk], ii, ZZ_tmp, zbes, lib_all, tau0=tau0)
                y0p, x0p     = fnc.tmp03(ID, PA, AA_tmp, Avtmp[kk], ii, ZZ_tmp, zbes, lib, tau0=tau0)
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

# Creat temporal png for gif image.
def plot_corner_param2(ID, PA, Zall=np.arange(-1.2,0.4249,0.05), age=[0.01, 0.1, 0.3, 0.7, 1.0, 3.0], tau0=[0.1,0.2,0.3], fig=None, out_ind=0):
    nage = np.arange(0,len(age),1)
    fnc  = Func(Zall, age, dust_model=dust_model) # Set up the number of Age/ZZ
    bfnc = Basic(Zall)

    ###########################
    # Open result file
    ###########################
    # Open ascii file and stock to array.
    lib     = fnc.open_spec_fits(ID, PA, fall=0, tau0=tau0)
    lib_all = fnc.open_spec_fits(ID, PA, fall=1, tau0=tau0)

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

    A50 = np.zeros(len(age), dtype='float32')
    A16 = np.zeros(len(age), dtype='float32')
    A84 = np.zeros(len(age), dtype='float32')
    for aa in range(len(age)):
        A50[aa] = hdul[1].data['A'+str(aa)][1]
        A16[aa] = hdul[1].data['A'+str(aa)][0]
        A84[aa] = hdul[1].data['A'+str(aa)][2]

    Asum  = np.sum(A50)
    aa   = 0
    Av50 = hdul[1].data['Av'+str(aa)][1]
    Av16 = hdul[1].data['Av'+str(aa)][0]
    Av84 = hdul[1].data['Av'+str(aa)][2]
    Z50 = np.zeros(len(age), dtype='float32')
    Z16 = np.zeros(len(age), dtype='float32')
    Z84 = np.zeros(len(age), dtype='float32')
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
    #nplot = 100

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
    ax0.errorbar(xbb[conbb], fybb[conbb] * c / np.square(xbb[conbb]) / d, yerr=eybb[conbb]*c/np.square(xbb[conbb])/d, color='k', linestyle='', linewidth=0.5, zorder=4)
    ax0.plot(xbb[conbb], fybb[conbb] * c / np.square(xbb[conbb]) / d, '.r', ms=10, linestyle='', linewidth=0, zorder=4)#, label='Obs.(BB)')
    #ax0.plot(xg0, fg0 * c / np.square(xg0) / d, marker='', linestyle='-', linewidth=0.5, ms=0.1, color='royalblue', label='')
    #ax0.plot(xg1, fg1 * c / np.square(xg1) / d, marker='', linestyle='-', linewidth=0.5, ms=0.1, color='#DF4E00', label='')

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
    Ntmp = np.zeros(nplot, dtype='float32')
    lmtmp= np.zeros(nplot, dtype='float32')
    Avtmp= np.zeros(nplot, dtype='float32')
    Ztmp = np.zeros(nplot, dtype='float32')
    Ttmp = np.zeros(nplot, dtype='float32')
    ACtmp= np.zeros(nplot, dtype='float32')
    col = ['darkred', 'r', 'coral','orange','g','lightgreen', 'lightblue', 'b','indigo','violet','k']


    # Time bin
    import cosmolopy.distance as cd
    import cosmolopy.constants as cc
    cosmo = {'omega_M_0' : 0.27, 'omega_lambda_0' : 0.73, 'h' : 0.72}
    cosmo = cd.set_omega_k_0(cosmo)

    Txmax = 4 # Max x value
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
            delTu[aa] = 10.
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
    delM = np.log10(M84) - np.log10(M16)
    #NPARmin = [np.log10(M16)-delM*1., np.log10(Tsmin/AMtmp16)-0.1, Av16-0.1, np.log10(Zsmin/AMtmp16)-0.2]
    #NPARmax = [np.log10(M84)+delM*1., np.log10(Tsmax/AMtmp84)+0.2, Av84+0.1, np.log10(Zsmax/AMtmp84)+0.2]
    NPARmin = [np.log10(M16)-.1, np.log10(Tsmin/AMtmp16)-0.1, Av16-0.1, np.log10(Zsmin/AMtmp16)-0.2]
    NPARmax = [np.log10(M84)+.1, np.log10(Tsmax/AMtmp84)+0.2, Av84+0.1, np.log10(Zsmax/AMtmp84)+0.2]
    #print(np.log10(Zsmin/AMtmp16),np.log10(Zsmax/AMtmp84),np.log10(Tsmin/AMtmp16),np.log10(Tsmax/AMtmp84))

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
        Tzz[zz] = (Tuni - cd.age(zred[zz], use_flat=True, **cosmo))/cc.Gyr_s
        if Tzz[zz] < 0.01:
            Tzz[zz] = 0.01


    import scipy.stats as stats
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
        ZMM = np.zeros((len(age)), dtype='float32') # Mass weighted Z.
        ZM  = np.zeros((len(age)), dtype='float32') # Light weighted T.
        ZC  = np.zeros((len(age)), dtype='float32') # Light weighted T.
        SF  = np.zeros((len(age)), dtype='float32') # SFR
        AM  = np.zeros((len(age)), dtype='float32') # Light weighted T.


        II0   = nage #[0,1,2,3] # Number for templates
        for ss in range(len(age)):
            ii = int(len(II0) - ss - 1) # from old to young templates.
            AA_tmp = samples['A'+str(ii)][nr]
            ZZ_tmp = samples['Z'+str(ii)][nr]

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
                y0, x0   = fnc.tmp03(ID, PA, AA_tmp, Avtmp[kk], ii, ZZ_tmp, zbes, lib_all, tau0=tau0)
                y0p, x0p = fnc.tmp03(ID, PA, AA_tmp, Avtmp[kk], ii, ZZ_tmp, zbes, lib, tau0=tau0)
                ysump = y0p #* 1e18
                ysum  = y0  #* 1e18
                if AA_tmp/Asum > flim:
                    ax0.plot(x0, y0 * c/ np.square(x0) / d, '--', lw=.1, color=col[ii], zorder=-1, label='', alpha=0.1)
                #ax1.plot(age[ii], SF[ii], marker='o', lw=1, color=col[ii], zorder=1, label='', alpha=0.5)
                #ax1.errorbar(age[ii], SF[ii], xerr=[[delTl[ii]/1e9], [delTu[ii]/1e9]], ms=10, marker='', lw=.1, color=col[ii], zorder=1, label='', alpha=0.1)
                #xx1 = np.arange(age[ii]-delTl[ii]/1e9, age[ii]+delTu[ii]/1e9, 0.01)
                #ax1.fill_between(xx1, xx1*0, xx1*0+SF[ii], lw=.1, facecolor=col[ii], zorder=1, label='', alpha=0.1)
                #ax2.plot(age[ii], ZM[ii], marker='o', lw=1, color=col[ii], zorder=1, label='', alpha=0.5)
            else:
                y0_r, x0_tmp = fnc.tmp03(ID, PA, AA_tmp, Avtmp[kk], ii, ZZ_tmp, zbes, lib_all, tau0=tau0)
                y0p, x0p     = fnc.tmp03(ID, PA, AA_tmp, Avtmp[kk], ii, ZZ_tmp, zbes, lib, tau0=tau0)
                ysump += y0p  #* 1e18
                ysum  += y0_r #* 1e18
                if AA_tmp/Asum > flim:
                    ax0.plot(x0, y0_r * c/ np.square(x0) / d, '--', lw=.1, color=col[ii], zorder=-1, label='', alpha=0.1)
                #ax1.plot(age[ii], SF[ii], marker='o', lw=1, color=col[ii], zorder=1, label='', alpha=0.5)
                #ax1.errorbar(age[ii], SF[ii], xerr=[[delTl[ii]/1e9], [delTu[ii]/1e9]], ms=10, marker='', lw=.1, color=col[ii], zorder=1, label='', alpha=0.1)
                #xx1 = np.arange(age[ii]-delTl[ii]/1e9, age[ii]+delTu[ii]/1e9, 0.01)
                #ax1.fill_between(xx1, xx1*0, xx1*0+SF[ii], lw=.1, facecolor=col[ii], zorder=1, label='', alpha=0.1)
                #ax2.plot(age[ii], ZM[ii], marker='o', lw=1, color=col[ii], zorder=1, label='', alpha=0.5)

        for ss in range(len(age)):
            #ii = int(len(II0) - ss - 1) # from old to young templates.
            ii = ss # from old to young templates.
            AC = np.sum(AM[ss:])
            ZC[ss] = np.log10(np.sum(ZMM[ss:])/AC)
            #ax2.errorbar(age[ii], ZC[ii], xerr=[[delTl[ii]/1e9], [delTu[ii]/1e9]], ms=10*(SF[ii]/np.sum(SF[:]))+1, marker='o', lw=1, color=col[ii], zorder=1, label='', alpha=0.5)
            #ax2.errorbar(age[ii], ZC[ii], xerr=[[delTl[ii]/1e9], [delTu[ii]/1e9]], ms=10*(AC/np.sum(AM[:]))+1, marker='o', lw=1, color=col[ii], zorder=1, label='', alpha=0.5)
            #ax2.errorbar(age[ii], ZC[ii], xerr=[[delTl[ii]/1e9], [delTu[ii]/1e9]], ms=1, marker='', lw=.1, color=col[ii], zorder=1, label='', alpha=0.1)

        # Total
        ymax = np.max(fybb[conbb] * c / np.square(xbb[conbb]) / d) * 1.10
        ax0.plot(x0, ysum * c/ np.square(x0) / d, '-', lw=.1, color='gray', zorder=-1, label='', alpha=0.1)
        ax1.plot(age[:], SF[:], marker='', linestyle='-', lw=.1, color='k', zorder=-1, label='', alpha=0.1)
        ax2.plot(age[:], ZC[:], marker='', linestyle='-', lw=.1, color='k', zorder=-1, label='', alpha=0.1)

        # Convert into log
        Ztmp[kk] /= ACtmp[kk]
        Ttmp[kk] /= ACtmp[kk]
        Ntmp[kk]  = kk

        lmtmp[kk] = np.log10(lmtmp[kk])
        Ztmp[kk]  = np.log10(Ztmp[kk])
        Ttmp[kk]  = np.log10(Ttmp[kk])

        #NPAR    = [lmtmp[:kk+1], Ttmp[:kk+1], Avtmp[:kk+1], Ztmp[:kk+1]]
        #NPAR    = [lmtmp[kk-10:kk+1], Ttmp[kk-10:kk+1], Avtmp[kk-10:kk+1], Ztmp[kk-10:kk+1]]
        #NPAR    = [lmtmp[kk], Ttmp[kk], Avtmp[kk], Ztmp[kk]]
        NPAR    = [lmtmp[:kk+1], Ttmp[:kk+1], Avtmp[:kk+1], Ztmp[:kk+1]]

        #for kk in range(0,nplot,1):
        if kk == nplot-1:
            # Histogram
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
    ax1.set_ylabel('$\log \dot{M_*}/M_\odot$yr$^{-1}$', fontsize=12)
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
    #print(x, x1min, x1max)
    #ax0.text('')

    plt.savefig(DIR_OUT + 'param_' + ID + '_PA' + PA + '_corner.png', dpi=150)
    plt.close()

# Creat temporal png for gif image.
def plot_corner_tmp(ID, PA, Zall=np.arange(-1.2,0.4249,0.05), age=[0.01, 0.1, 0.3, 0.7, 1.0, 3.0], tau0=[0.1,0.2,0.3], fig=None):
    nage = np.arange(0,len(age),1)
    fnc  = Func(Zall, age, dust_model=dust_model) # Set up the number of Age/ZZ
    bfnc = Basic(Zall)

    ###########################
    # Open result file
    ###########################
    # Open ascii file and stock to array.
    lib     = fnc.open_spec_fits(ID, PA, fall=0, tau0=tau0)
    lib_all = fnc.open_spec_fits(ID, PA, fall=1, tau0=tau0)

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

    A50 = np.zeros(len(age), dtype='float32')
    A16 = np.zeros(len(age), dtype='float32')
    A84 = np.zeros(len(age), dtype='float32')
    for aa in range(len(age)):
        A50[aa] = hdul[1].data['A'+str(aa)][1]
        A16[aa] = hdul[1].data['A'+str(aa)][0]
        A84[aa] = hdul[1].data['A'+str(aa)][2]

    Asum  = np.sum(A50)
    aa   = 0
    Av50 = hdul[1].data['Av'+str(aa)][1]
    Av16 = hdul[1].data['Av'+str(aa)][0]
    Av84 = hdul[1].data['Av'+str(aa)][2]
    Z50 = np.zeros(len(age), dtype='float32')
    Z16 = np.zeros(len(age), dtype='float32')
    Z84 = np.zeros(len(age), dtype='float32')
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
    Ntmp = np.zeros(nplot, dtype='float32')
    lmtmp= np.zeros(nplot, dtype='float32')
    Avtmp= np.zeros(nplot, dtype='float32')
    Ztmp = np.zeros(nplot, dtype='float32')
    Ttmp = np.zeros(nplot, dtype='float32')
    ACtmp= np.zeros(nplot, dtype='float32')
    col = ['darkred', 'r', 'coral','orange','g','lightgreen', 'lightblue', 'b','indigo','violet','k']


    # Time bin
    import cosmolopy.distance as cd
    import cosmolopy.constants as cc
    cosmo = {'omega_M_0' : 0.27, 'omega_lambda_0' : 0.73, 'h' : 0.72}
    cosmo = cd.set_omega_k_0(cosmo)

    Txmax = 4 # Max x value
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
            delTu[aa] = 10.
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
        ZMM = np.zeros((len(age)), dtype='float32') # Mass weighted Z.
        ZM  = np.zeros((len(age)), dtype='float32') # Light weighted T.
        ZC  = np.zeros((len(age)), dtype='float32') # Light weighted T.
        SF  = np.zeros((len(age)), dtype='float32') # SFR
        AM  = np.zeros((len(age)), dtype='float32') # Light weighted T.


        II0   = nage #[0,1,2,3] # Number for templates
        for ss in range(len(age)):
            ii = int(len(II0) - ss - 1) # from old to young templates.
            AA_tmp = samples['A'+str(ii)][nr]
            ZZ_tmp = samples['Z'+str(ii)][nr]

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
                y0, x0   = fnc.tmp03(ID, PA, AA_tmp, Avtmp[kk], ii, ZZ_tmp, zbes, lib_all, tau0=tau0)
                y0p, x0p = fnc.tmp03(ID, PA, AA_tmp, Avtmp[kk], ii, ZZ_tmp, zbes, lib, tau0=tau0)
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
                y0_r, x0_tmp = fnc.tmp03(ID, PA, AA_tmp, Avtmp[kk], ii, ZZ_tmp, zbes, lib_all, tau0=tau0)
                y0p, x0p     = fnc.tmp03(ID, PA, AA_tmp, Avtmp[kk], ii, ZZ_tmp, zbes, lib, tau0=tau0)
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

        Tzz   = np.zeros(len(zred), dtype='float32')
        for zz in range(len(zred)):
            Tzz[zz] = (Tuni - cd.age(zred[zz], use_flat=True, **cosmo))/cc.Gyr_s
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
    nage = np.arange(0,len(age),1)
    fnc  = Func(Zall, age, dust_model=dust_model) # Set up the number of Age/ZZ
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
    truth = np.zeros(len(age)*2+1, dtype='float32')

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

def plot_sim_comp(ID0, PA, Z=np.arange(-1.2,0.4249,0.05), age=[0.01, 0.1, 0.3, 0.7, 1.0, 3.0],  f_Z_all=0, tau0=[0.1,0.2,0.3]):

    nage = np.arange(0,len(age),1)
    fnc  = Func(Z, nage, dust_model=dust_model) # Set up the number of Age/ZZ
    bfnc = Basic(Z)

    c      = 3.e18 # A/s
    chimax = 1.
    mag0   = 25.0
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
    file = DIR_TMP + 'sim_param_' + ID0 + '_PA' + PA + '.fits'
    hdu0 = fits.open(file) # open a FITS file

    Avin = float(hdu0[0].header['AV'])
    Ain  = np.zeros(len(age), dtype='float32')
    Zin  = np.zeros(len(age), dtype='float32')
    for aa in range(len(age)):
        Ain[aa] = hdu0[0].header['A'+str(aa)]
        Zin[aa] = hdu0[0].header['Z'+str(aa)]

    Ainsum = np.sum(Ain)

    ###########################
    # Open result file
    ###########################
    file = 'summary_' + ID0 + '_PA' + PA + '.fits'
    hdul = fits.open(file) # open a FITS file

    # Redshift MC
    zp50  = hdul[1].data['zmc'][1]
    zp16  = hdul[1].data['zmc'][0]
    zp84  = hdul[1].data['zmc'][2]

    M50 = hdul[1].data['ms'][1]
    M16 = hdul[1].data['ms'][0]
    M84 = hdul[1].data['ms'][2]
    print('Total stellar mass is %.2e'%(M50))

    A50 = np.zeros(len(age), dtype='float32')
    A16 = np.zeros(len(age), dtype='float32')
    A84 = np.zeros(len(age), dtype='float32')
    for aa in range(len(age)):
        A50[aa] = hdul[1].data['A'+str(aa)][1]
        A16[aa] = hdul[1].data['A'+str(aa)][0]
        A84[aa] = hdul[1].data['A'+str(aa)][2]

    Asum  = np.sum(A50)

    aa   = 0
    Av50 = hdul[1].data['Av'+str(aa)][1]
    Av16 = hdul[1].data['Av'+str(aa)][0]
    Av84 = hdul[1].data['Av'+str(aa)][2]

    Z50 = np.zeros(len(age), dtype='float32')
    Z16 = np.zeros(len(age), dtype='float32')
    Z84 = np.zeros(len(age), dtype='float32')
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
    msize = np.zeros(len(age), dtype='float32')
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

    plt.savefig('SIM' + ID0 + '_PA' + PA + '_comp.pdf', dpi=300)

def plot_sed_Z(ID0, PA, Z=np.arange(-1.2,0.4249,0.05), age=[0.01, 0.1, 0.3, 0.7, 1.0, 3.0], \
f_Z_all=0, tau0=[0.1,0.2,0.3], flim=0.01, fil_path='./', SNlim=1.5, figpdf=False, \
save_sed=True, inputs=False, nmc2=300, dust_model=0, DIR_TMP='./templates/'):
    #
    # Returns: plots.
    #
    # snlimbb: SN limit to show flux or up lim in SED.
    #
    col = ['darkred', 'r', 'coral','orange','g','lightgreen', 'lightblue', 'b','indigo','violet','k']

    nage = np.arange(0,len(age),1)
    fnc  = Func(Z, nage, dust_model=dust_model, DIR_TMP=DIR_TMP) # Set up the number of Age/ZZ
    bfnc = Basic(Z)

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
    fig = plt.figure(figsize=(5.,2.4))
    fig.subplots_adjust(top=0.96, bottom=0.16, left=0.1, right=0.99, hspace=0.15, wspace=0.25)
    ax1 = fig.add_subplot(111)

    ##################
    # Fitting Results
    ##################
    #DIR_TMP  = './templates/'
    DIR_FILT = fil_path
    if inputs:
        SFILT = inputs['FILTER'] # filter band string.
        SFILT = [x.strip() for x in SFILT.split(',')]
    else:
        SFILT = []

    try:
        f_err = int(inputs['F_ERR'])
    except:
        f_err = 0
    ###########################
    # Open result file
    ###########################
    file = 'summary_' + ID0 + '_PA' + PA + '.fits'
    hdul = fits.open(file) # open a FITS file

    # Redshift MC
    zp16  = hdul[1].data['zmc'][0]
    zp50  = hdul[1].data['zmc'][1]
    zp84  = hdul[1].data['zmc'][2]

    # Stellar mass MC
    M16 = hdul[1].data['ms'][0]
    M50 = hdul[1].data['ms'][1]
    M84 = hdul[1].data['ms'][2]
    print('Total stellar mass is %.2e'%(M50))

    # Amplitude MC
    A50 = np.zeros(len(age), dtype='float32')
    A16 = np.zeros(len(age), dtype='float32')
    A84 = np.zeros(len(age), dtype='float32')
    for aa in range(len(age)):
        A50[aa] = hdul[1].data['A'+str(aa)][1]
        A16[aa] = hdul[1].data['A'+str(aa)][0]
        A84[aa] = hdul[1].data['A'+str(aa)][2]

    Asum  = np.sum(A50)

    aa = 0
    Av16 = hdul[1].data['Av'+str(aa)][0]
    Av50 = hdul[1].data['Av'+str(aa)][1]
    Av84 = hdul[1].data['Av'+str(aa)][2]
    AAv = [Av50]

    Z50 = np.zeros(len(age), dtype='float32')
    Z16 = np.zeros(len(age), dtype='float32')
    Z84 = np.zeros(len(age), dtype='float32')
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
    #if True:
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
        print('Total dust mass is %.2e'%(MD50))
        f_dust = True
    except:
        f_dust = False

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
    dat  = np.loadtxt(DIR_TMP + 'spec_obs_' + ID0 + '_PA' + PA + '.cat', comments='#')
    NR   = dat[:, 0]
    x    = dat[:, 1]
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
    if len(xg0)>0 or len(xg1)>0:
        f_grsm = True
    else:
        f_grsm = False

    con2 = (NR>=10000)#& (fy/ey>SNlim)
    xg2  = x[con2]
    fg2  = fy00[con2]
    eg2  = ey00[con2]

    xg01 = np.append(xg0,xg1)
    fy01 = np.append(fg0,fg1)
    ey01 = np.append(eg0,eg1)
    xg   = np.append(xg01,xg2)
    fy   = np.append(fy01,fg2)
    ey   = np.append(ey01,eg2)

    wht=1./np.square(ey)

    dat = np.loadtxt(DIR_TMP + 'bb_obs_' + ID0 + '_PA' + PA + '.cat', comments='#')
    NRbb = dat[:, 0]
    xbb  = dat[:, 1]
    fybb = dat[:, 2]
    eybb = dat[:, 3]
    exbb = dat[:, 4]
    snbb = fybb/eybb

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
    ms     = np.zeros(len(age), dtype='float32')
    f0     = fits.open(DIR_TMP + 'ms_' + ID0 + '_PA' + PA + '.fits')
    sedpar = f0[1]
    for aa in range(len(age)):
        ms[aa] = sedpar.data['ML_' +  str(int(NZbest[aa]))][aa]

    conspec = (NR<10000) #& (fy/ey>1)
    ax1.plot(xg0, fg0 * c / np.square(xg0) / d, marker='', linestyle='-', linewidth=0.5, ms=0.1, color='royalblue', label='')
    ax1.plot(xg1, fg1 * c / np.square(xg1) / d, marker='', linestyle='-', linewidth=0.5, ms=0.1, color='#DF4E00', label='')

    conbb = (NR>=10000)

    #######################
    # D.Kelson like Box for BB photometry
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

    try:
        conbb_hs = (fybb/eybb>SNlim)
        ax1.errorbar(xbb[conbb_hs], fybb[conbb_hs] * c / np.square(xbb[conbb_hs]) / d, \
        yerr=eybb[conbb_hs]*c/np.square(xbb[conbb_hs])/d, color='k', linestyle='', linewidth=0.5, zorder=4)
        ax1.plot(xbb[conbb_hs], fybb[conbb_hs] * c / np.square(xbb[conbb_hs]) / d, \
        '.r', linestyle='', linewidth=0, zorder=4)#, label='Obs.(BB)')
    except:
        pass
    try:
        conebb_ls = (fybb/eybb<=SNlim)
        ax1.errorbar(xbb[conebb_ls], eybb[conebb_ls] * c / np.square(xbb[conebb_ls]) / d * SNlim, \
        yerr=fybb[conebb_ls]*0+np.max(fybb[conebb_ls]*c/np.square(xbb[conebb_ls])/d)*0.05, \
        uplims=eybb[conebb_ls]*c/np.square(xbb[conebb_ls])/d*SNlim, color='r', linestyle='', linewidth=0.5, zorder=4)
    except:
        pass

    #####################################
    # Open ascii file and stock to array.
    lib     = fnc.open_spec_fits(ID0, PA, fall=0, tau0=tau0)
    lib_all = fnc.open_spec_fits(ID0, PA, fall=1, tau0=tau0)
    if f_dust:
        DT0 = float(inputs['TDUST_LOW'])
        DT1 = float(inputs['TDUST_HIG'])
        dDT = float(inputs['TDUST_DEL'])
        Temp= np.arange(DT0,DT1,dDT)
        lib_dust     = fnc.open_spec_dust_fits(ID0, PA, Temp, fall=0, tau0=tau0)
        lib_dust_all = fnc.open_spec_dust_fits(ID0, PA, Temp, fall=1, tau0=tau0)

    II0   = nage #[0,1,2,3] # Number for templates
    iimax = len(II0)-1

    fwuvj = open(ID0 + '_PA' + PA + '_uvj.txt', 'w')
    fwuvj.write('# age uv vj\n')
    Asum = np.sum(A50[:])
    for jj in range(len(II0)):
        ii = int(len(II0) - jj - 1) # from old to young templates.
        if jj == 0:
            y0, x0   = fnc.tmp03(ID0, PA, A50[ii], AAv[0], ii, Z50[ii], zbes, lib_all, tau0=tau0)
            y0p, x0p = fnc.tmp03(ID0, PA, A50[ii], AAv[0], ii, Z50[ii], zbes, lib, tau0=tau0)
            ysump = y0p
            ysum  = y0
            if A50[ii]/Asum > flim:
                ax1.plot(x0, y0 * c/ np.square(x0) / d, '--', lw=0.5, color=col[ii], zorder=-1, label='')
        else:
            y0_r, x0_tmp = fnc.tmp03(ID0, PA, A50[ii], AAv[0], ii, Z50[ii], zbes, lib_all, tau0=tau0)
            y0p, x0p     = fnc.tmp03(ID0, PA, A50[ii], AAv[0], ii, Z50[ii], zbes, lib, tau0=tau0)
            ysump += y0p
            ysum  += y0_r
            if A50[ii]/Asum > flim:
                ax1.plot(x0, y0_r * c/ np.square(x0) / d, '--', lw=0.5, color=col[ii], zorder=-1, label='')
        ysum_wid = ysum * 0

        for kk in range(0,ii+1,1):
            tt = int(len(II0) - kk - 1)
            nn = int(len(II0) - ii - 1)

            nZ = bfnc.Z2NZ(Z50[tt])
            y0_wid, x0_wid = fnc.open_spec_fits_dir(ID0, PA, tt, nZ, nn, AAv[0], zbes, A50[tt], tau0=tau0)
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

    ################
    # Set the inset.
    ################
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    from mpl_toolkits.axes_grid.inset_locator import inset_axes
    if f_grsm:
        ax2t = inset_axes(ax1, width="40%", height="30%", loc=2)

    # FIR dust plot;
    if f_dust:
        ax3t = inset_axes(ax1, width="40%", height="30%", loc=1)
        from lmfit import Parameters
        par = Parameters()
        par.add('MDUST',value=MD50)
        par.add('TDUST',value=nTD50)
        par.add('zmc',value=zp50)
        y0d, x0d = fnc.tmp04_dust(ID0, PA, par.valuesdict(), zbes, lib_dust_all, tau0=tau0)
        ax1.plot(x0d, y0d * c/ np.square(x0d) / d, '--', lw=0.5, color='purple', zorder=-1, label='')
        ax3t.plot(x0d, y0d * c/ np.square(x0d) / d, '--', lw=0.5, color='purple', zorder=-1, label='')
        #ax1.plot(x0d, y0d, '--', lw=0.5, color='purple', zorder=-1, label='')
        #ax3t.plot(x0d, y0d, '--', lw=0.5, color='purple', zorder=-1, label='')

        # data;
        ddat  = np.loadtxt(DIR_TMP + 'bb_dust_obs_' + ID0 + '_PA' + PA + '.cat', comments='#')
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
            ax1.errorbar(xbbd[conebbd_ls], eybbd[conebbd_ls] * c / np.square(xbbd[conebbd_ls]) / d * SNlim, \
            yerr=fybbd[conebbd_ls]*0+np.max(fybbd[conebbd_ls]*c/np.square(xbbd[conebbd_ls])/d)*0.05, \
            uplims=eybbd[conebbd_ls]*c/np.square(xbbd[conebbd_ls])/d*SNlim, color='r', linestyle='', linewidth=0.5, zorder=4)
            ax3t.errorbar(xbbd[conebbd_ls], eybbd[conebbd_ls] * c / np.square(xbbd[conebbd_ls]) / d * SNlim, \
            yerr=fybbd[conebbd_ls]*0+np.max(fybbd[conebbd_ls]*c/np.square(xbbd[conebbd_ls])/d)*0.05, \
            uplims=eybbd[conebbd_ls]*c/np.square(xbbd[conebbd_ls])/d*SNlim, color='r', linestyle='', linewidth=0.5, zorder=4)
        except:
            pass

        #print(xbbd[conbbd_hs], fybbd[conbbd_hs] * c / np.square(xbbd[conbbd_hs]) / d)
        #print(np.max(y0d * c/ np.square(x0d) / d))
        #plt.show()

    conw = (wht3>0)
    chi2 = sum((np.square(fy-ysump)*wht3)[conw])
    print('chi2/nu is %.2f'%(chin))
    #############
    # Main result
    #############
    conbb_ymax = (xbb>0.5)# (conbb) &
    ymax = np.max(fybb[conbb_ymax]*c/np.square(xbb[conbb_ymax])/d) * 2.

    xboxl = 17000
    xboxu = 28000

    ax1.set_xlabel('Observed wavelength ($\mathrm{\mu m}$)', fontsize=14)
    ax1.set_ylabel('Flux ($10^{-18}\mathrm{erg}/\mathrm{s}/\mathrm{cm}^{2}/\mathrm{\AA}$)', fontsize=13)

    x1max = 22000
    if x1max < np.max(xbb[conbb_ymax]):
        x1max = np.max(xbb[conbb_ymax]) * 1.1
    ax1.set_xlim(2200, x1max)
    ax1.set_xscale('log')
    ax1.set_ylim(-ymax*0.1, ymax)

    #import matplotlib.ticker as ticker
    import matplotlib
    #ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
    xticks = [2500, 5000, 10000, 20000, 40000, 80000]
    xlabels= ['0.25', '0.5', '1', '2', '4', '8']
    #if f_dust:
    #    xticks = [2500, 5000, 10000, 20000, 40000, 80000, 1e7]
    #    xlabels= ['0.25', '0.5', '1', '2', '4', '8', '1000']
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xlabels)

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
    eAAl = np.zeros(len(age),dtype='float32')
    eAAu = np.zeros(len(age),dtype='float32')
    eAMl = np.zeros(len(age),dtype='float32')
    eAMu = np.zeros(len(age),dtype='float32')
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

    dat = np.loadtxt(DIR_TMP + 'spec_obs_' + ID0 + '_PA' + PA + '.cat', comments='#')
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

    ####################
    # For cosmology
    ####################
    import cosmolopy.distance as cd
    import cosmolopy.constants as cc
    cosmo = {'omega_M_0' : 0.27, 'omega_lambda_0' : 0.73, 'h' : 0.72}
    cosmo = cd.set_omega_k_0(cosmo)

    Lsun = 3.839 * 1e33 #erg s-1
    Mpc_cm = 3.08568025e+24 # cm/Mpc
    DL = cd.luminosity_distance(zbes, **cosmo) * Mpc_cm # Luminositydistance in cm
    Cons = (4.*np.pi*DL**2/(1.+zbes))

    dA     = cd.angular_diameter_distance(zbes, **cosmo)
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
    flw = open(ID0 + '_PA' + PA + '_lines_fit.txt', 'w')
    flw.write('# LW flux_line eflux_line flux_cont EW eEW L_line eL_line\n')
    flw.write('# (AA) (Flam_1e-18) (Flam_1e-18) (Flam_1e-18) (AA) (AA) (erg/s) (erg/s)\n')
    flw.write('# Error in EW is 1sigma, by pm eflux_line.\n')
    flw.write('# If EW=-99, it means gaussian fit failed.\n')
    flw.write('# and flux is the sum of excess at WL pm %.1f AA.\n'%(dlw))
    flw.write('# Magnification is corrected; mu=%.3f\n'%(umag))
    try:
        fl = np.loadtxt('table_' + ID0 + '_PA' + PA + '_lines.txt', comments='#')
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

                flux = np.zeros(len(xxs), dtype='float32')
                efl  = np.zeros(len(xxs), dtype='float32')
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

    ############
    # Zoom Line
    ############
    if f_grsm:
        conspec = (NR<10000) #& (fy/ey>1)
        ax2t.fill_between(xg1/zscl, (fg1-eg1) * c/np.square(xg1)/d, (fg1+eg1) * c/np.square(xg1)/d, lw=0, color='#DF4E00', zorder=10, alpha=0.7, label='')
        ax2t.fill_between(xg0/zscl, (fg0-eg0) * c/np.square(xg0)/d, (fg0+eg0) * c/np.square(xg0)/d, lw=0, color='royalblue', zorder=10, alpha=0.2, label='')
        ax2t.errorbar(xg1/zscl, fg1 * c/np.square(xg1)/d, yerr=eg1 * c/np.square(xg1)/d, lw=0.5, color='#DF4E00', zorder=10, alpha=1., label='', capsize=0)
        ax2t.errorbar(xg0/zscl, fg0 * c/np.square(xg0)/d, yerr=eg0 * c/np.square(xg0)/d, lw=0.5, color='royalblue', zorder=10, alpha=1., label='', capsize=0)

        con4000b = (xg01/zscl>3400) & (xg01/zscl<3800) & (fy01>0) & (ey01>0)
        con4000r = (xg01/zscl>4200) & (xg01/zscl<5000) & (fy01>0) & (ey01>0)
        print('Median SN at 3400-3800 is;', np.median((fy01/ey01)[con4000b]))
        print('Median SN at 4200-5000 is;', np.median((fy01/ey01)[con4000r]))

    # From MCMC chain
    file = 'chain_' + ID0 + '_PA' + PA + '_corner.cpkl'
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
    ytmp = np.zeros((nmc2,len(ysum)), dtype='float32')
    ytmpmax = np.zeros(len(ysum), dtype='float32')
    ytmpmin = np.zeros(len(ysum), dtype='float32')

    # MUV;
    Mpc_cm  = 3.08568025e+24 # cm/Mpc
    DL      = cd.luminosity_distance(zbes, **cosmo) * Mpc_cm # Luminositydistance in cm
    DL10    = Mpc_cm/1e6 * 10 # 10pc in cm
    Fuv     = np.zeros(nmc2, dtype='float64') # For Muv
    Fuv28   = np.zeros(nmc2, dtype='float64') # For Fuv(1500-2800)
    Lir     = np.zeros(nmc2, dtype='float64') # For L(8-1000um)
    Cmznu   = 10**((48.6+mag0)/(-2.5)) # Conversion from m0_25 to fnu

    for kk in range(0,nmc2,1):
        nr = np.random.randint(len(samples))
        Av_tmp = samples['Av'][nr]
        for ss in range(len(age)):
            AA_tmp = samples['A'+str(ss)][nr]
            try:
                Ztest  = samples['Z'+str(len(age)-1)][nr]
                ZZ_tmp = samples['Z'+str(ss)][nr]
            except:
                ZZ_tmp = samples['Z0'][nr]

            if ss == 0:
                mod0_tmp, xm_tmp = fnc.tmp03(ID0, PA, AA_tmp, Av_tmp, ss, ZZ_tmp, zbes, lib_all, tau0=tau0)
                fm_tmp = mod0_tmp
            else:
                mod0_tmp, xx_tmp = fnc.tmp03(ID0, PA, AA_tmp, Av_tmp, ss, ZZ_tmp, zbes, lib_all, tau0=tau0)
                fm_tmp += mod0_tmp

        if f_err == 1:
            ferr_tmp = samples['f'][nr]
        else:
            ferr_tmp = 1.0

        # Dust component;
        if f_dust:
            if kk == 0:
                par  = Parameters()
                par.add('MDUST',value=samples['MDUST'][nr])
                par.add('TDUST',value=samples['TDUST'][nr])
            model_dust, x1_dust = fnc.tmp04_dust(ID0, PA, par.valuesdict(), zbes, lib_dust_all, tau0=tau0)
            if kk == 0:
                deldt  = (x1_dust[1] - x1_dust[0])
                x1_tot = np.append(xm_tmp,np.arange(np.max(xm_tmp),np.max(x1_dust),deldt))
                ytmp   = np.zeros((nmc2,len(x1_tot)), dtype='float32')
            model_tot  = np.interp(x1_tot,xx_tmp,fm_tmp) + np.interp(x1_tot,x1_dust,model_dust)
            ax1.plot(x1_tot, model_tot * c/ np.square(x1_tot) / d, '-', lw=1, color='gray', zorder=-2, alpha=0.02)
            if f_grsm:
                ax2t.plot(x1_tot/zscl, model_tot * c/np.square(x1_tot)/d, '-', lw=0.5, color='gray', zorder=3., alpha=0.02)
            ax3t.plot(x1_tot, model_tot * c/ np.square(x1_tot) / d, '-', lw=1, color='gray', zorder=-2, alpha=0.02)
            ytmp[kk,:] = ferr_tmp * model_tot[:] * c/np.square(x1_tot[:])/d
            Fuv[kk]    = get_Fuv(x1_tot[:]/(1.+zbes), model_tot * (DL**2/(1.+zbes)) / (DL10**2), lmin=1250, lmax=1650)
            Fuv28[kk]  = get_Fuv(x1_tot[:]/(1.+zbes), model_tot * (4*np.pi*DL**2/(1.+zbes))*Cmznu, lmin=1500, lmax=2800)
            Lir[kk]    = get_Fint(x1_tot[:]/(1.+zbes), model_tot * (4*np.pi*DL**2/(1.+zbes))*Cmznu, lmin=80000, lmax=10000*1e3)
        else:
            ytmp[kk,:] = ferr_tmp * fm_tmp[:] * c/ np.square(xm_tmp[:]) / d
            ax1.plot(xm_tmp, fm_tmp * c/ np.square(xm_tmp) / d, '-', lw=1, color='gray', zorder=-2, alpha=0.02)
            if f_grsm:
                ax2t.plot(xm_tmp/zscl, fm_tmp * c/np.square(xm_tmp)/d, '-', lw=0.5, color='gray', zorder=3., alpha=0.02)
            Fuv[kk]   = get_Fuv(xm_tmp[:]/(1.+zbes), fm_tmp * (DL**2/(1.+zbes)) / (DL10**2), lmin=1250, lmax=1650)
            Fuv28[kk] = get_Fuv(xm_tmp[:]/(1.+zbes), fm_tmp * (4*np.pi*DL**2/(1.+zbes))*Cmznu, lmin=1500, lmax=2800)
            Lir[kk]   = 0 #get_Fint(xm_tmp[:]/(1.+zbes), fm_tmp * (4*np.pi*DL**2/(1.+zbes))*Cmznu, lmin=80000, lmax=10000*1e3)

    # plot BB model;
    #from .maketmp_filt import filconv
    if f_dust:
        ALLFILT = np.append(SFILT,DFILT)
        #for ii in range(len(x1_tot)):
        #    print(x1_tot[ii], model_tot[ii]*c/np.square(x1_tot[ii])/d)
        lbb, fbb, lfwhm = filconv(ALLFILT, x1_tot, model_tot*c/np.square(x1_tot)/d, DIR_FILT, fw=True)
        ax1.scatter(lbb, fbb, lw=1, color='none', edgecolor='b', \
        zorder=2, alpha=1.0, marker='s', s=10)
        ax3t.scatter(lbb, fbb, lw=1, color='none', edgecolor='b', \
        zorder=2, alpha=1.0, marker='s', s=10)
        if save_sed == True:
            fbb_nu = flamtonu(lbb, fbb*1e-18, m0set=25.0)
            fw = open(ID0 + '_PA' + PA + '_sed.txt', 'w')
            fw.write('# wave fnu       filt_width No.Filt\n')
            fw.write('# (AA) (m0=25.0) (AA)       ()\n')
            for ii in range(len(lbb)):
                fw.write('%.2f %.5f %.2f %s\n'%(lbb[ii],fbb_nu[ii],lfwhm[ii],ALLFILT[ii]))
            fw.close()
    else:
        lbb, fbb, lfwhm = filconv(SFILT, xm_tmp, fm_tmp*c/np.square(xm_tmp)/d, DIR_FILT, fw=True)
        ax1.scatter(lbb, fbb, lw=1, color='none', edgecolor='b', zorder=2, alpha=1.0, marker='s', s=10)
        if save_sed == True:
            fbb_nu = flamtonu(lbb, fbb*1e-18, m0set=25.0)
            fw = open(ID0 + '_PA' + PA + '_sed.txt', 'w')
            fw.write('# wave fnu       filt_width No.Filt\n')
            fw.write('# (AA) (m0=25.0) (AA)       ()\n')
            for ii in range(len(lbb)):
                fw.write('%.2f %.5f %.2f %s\n'%(lbb[ii],fbb_nu[ii],lfwhm[ii],SFILT[ii]))
            fw.close()

    if save_sed == True:
        col00  = []
        ytmp16 = np.zeros(len(xm_tmp), dtype='float32')
        ytmp50 = np.zeros(len(xm_tmp), dtype='float32')
        ytmp84 = np.zeros(len(xm_tmp), dtype='float32')
        for kk in range(len(xm_tmp[:])):
            ytmp16[kk] = np.percentile(ytmp[:,kk],16)
            ytmp50[kk] = np.percentile(ytmp[:,kk],50)
            ytmp84[kk] = np.percentile(ytmp[:,kk],84)
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
        hdr['id'] = ID0

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

        # Write;
        colspec = fits.ColDefs(col00)
        hdu0    = fits.BinTableHDU.from_columns(colspec, header=hdr)
        hdu0.writeto(DIR_TMP + 'gsf_spec_%s.fits'%(ID0), overwrite=True)

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
            print('y3 limit is not specified.')
            pass
        ax3t.set_xlim(1e5, 2e7)
        ax3t.set_xscale('log')
        #ax3t.set_yticklabels(())
        ax3t.set_xticks([100000, 1000000, 10000000])
        ax3t.set_xticklabels(['10', '100', '1000'])
        #plt.show()

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
    # Plot Different Z
    ####################
    if f_Z_all == 1:
        fileZ = 'Z_' + ID0 + '_PA' + PA + '.cat'
        Zini, chi, Av = np.loadtxt(fileZ, comments='#', unpack=True, usecols=[1, 2, 3+len(age)])
        Atmp  = np.zeros((len(age),len(Zini)), 'float32')
        Ztmp  = np.zeros((len(age),len(Zini)), 'float32')
        for aa in range(len(age)):
            Atmp[aa,:] = np.loadtxt(fileZ, comments='#', unpack=True, usecols=[3+aa])
            Ztmp[aa,:] = np.loadtxt(fileZ, comments='#', unpack=True, usecols=[3+len(age)+1+aa])

        for jj in range(len(Zini)):
            for aa in range(len(age)):
                if aa == 0:
                    y0_r, x0_r    = fnc.tmp03(ID0, PA, Atmp[aa, jj], Av[jj], aa, Ztmp[aa, jj], zbes, lib_all)
                    y0_rp, x0_rp  = fnc.tmp03(ID0, PA, Atmp[aa, jj], Av[jj], aa, Ztmp[aa, jj], zbes, lib)
                else:
                    y0_rr, x0_r     = fnc.tmp03(ID0, PA, Atmp[aa, jj], Av[jj], aa, Ztmp[aa, jj], zbes, lib_all)
                    y0_rrp, x0_rrp  = fnc.tmp03(ID0, PA, Atmp[aa, jj], Av[jj], aa, Ztmp[aa, jj], zbes, lib)
                    y0_r  += y0_rr
                    y0_rp += y0_rrp

            ysum_Z = y0_r
            chi2_ind = sum((np.square(fy-y0_rp)*wht3)[conw])
            print('At Z=%.2f; chi2/nu is %.2f', Zini[jj], chi2_ind/nu)

            if Zini[jj]>=0:
                ax1.plot(x0_r, (ysum_Z)* c/np.square(x0_r)/d, '--', lw=0.3+0.2*jj, color='gray', zorder=-3, alpha=0.7, label='$\log\mathrm{Z_*}=\ \ %.2f$'%(Zini[jj])) # Z here is Zinitial.
            else:
                ax1.plot(x0_r, (ysum_Z)* c/np.square(x0_r)/d, '--', lw=0.3+0.2*jj, color='gray', zorder=-3, alpha=0.7, label='$\log\mathrm{Z_*}=%.2f$'%(Zini[jj])) # Z here is Zinitial.
            if f_grsm:
                ax2t.plot(x0_r/zscl, ysum_Z * c/ np.square(x0_r) / d, '--', lw=0.3+0.2*jj, color='gray', zorder=-3, alpha=0.5)

    ################
    # RGB
    ################
    try:
        from scipy import misc
        rgb_array = misc.imread('/Users/tmorishita/Box Sync/Research/M18_rgb/rgb_'+str(int(ID0))+'.png')
        axicon = fig.add_axes([0.68, 0.53, 0.4, 0.4])
        axicon.imshow(rgb_array, interpolation='nearest', origin='upper')
        axicon.set_xticks([])
        axicon.set_yticks([])
    except:
        pass
    ####################
    ## Save
    ####################
    #plt.show()
    ax1.legend(loc=1, fontsize=11)
    if figpdf:
        plt.savefig('SPEC_' + ID0 + '_PA' + PA + '_spec.pdf', dpi=300)
    else:
        plt.savefig('SPEC_' + ID0 + '_PA' + PA + '_spec.png', dpi=150)


def plot_sed_Z_sim(ID0, PA, Z=np.arange(-1.2,0.4249,0.05), age=[0.01, 0.1, 0.3, 0.7, 1.0, 3.0],  f_Z_all=0, tau0=[0.1,0.2,0.3], flim=0.01, fil_path='./', figpdf=False):

    col = ['darkred', 'r', 'coral','orange','g','lightgreen', 'lightblue', 'b','indigo','violet','k']

    nage = np.arange(0,len(age),1)
    fnc  = Func(Z, nage, dust_model=dust_model) # Set up the number of Age/ZZ
    bfnc = Basic(Z)

    ################
    # RF colors.
    import os.path
    home = os.path.expanduser('~')
    #fil_path = '/Users/tmorishita/eazy-v1.01/PROG/FILT/'
    #fil_path = home + '/Dropbox/FILT/'

    c      = 3.e18 # A/s
    chimax = 1.
    mag0   = 25.0
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
    file = 'summary_' + ID0 + '_PA' + PA + '.fits'
    hdul = fits.open(file) # open a FITS file

    # Redshift MC
    zp50  = hdul[1].data['zmc'][1]
    zp16  = hdul[1].data['zmc'][0]
    zp84  = hdul[1].data['zmc'][2]


    M50 = hdul[1].data['ms'][1]
    M16 = hdul[1].data['ms'][0]
    M84 = hdul[1].data['ms'][2]
    print('Total stellar mass is %.2e'%(M50))

    A50 = np.zeros(len(age), dtype='float32')
    A16 = np.zeros(len(age), dtype='float32')
    A84 = np.zeros(len(age), dtype='float32')
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

    Z50 = np.zeros(len(age), dtype='float32')
    Z16 = np.zeros(len(age), dtype='float32')
    Z84 = np.zeros(len(age), dtype='float32')
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
    dat = np.loadtxt(DIR_TMP + 'spec_obs_' + ID0 + '_PA' + PA + '.cat', comments='#')
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

    dat = np.loadtxt(DIR_TMP + 'bb_obs_' + ID0 + '_PA' + PA + '.cat', comments='#')
    NRbb = dat[:, 0]
    xbb  = dat[:, 1]
    fybb = dat[:, 2]
    eybb = dat[:, 3]
    exbb = dat[:, 4]
    snbb = fybb/eybb


    # Original spectrum:
    filepar = DIR_TMP + 'sim_param_' + ID0 + '_PA' + PA + '.fits'
    hdupar  = fits.open(filepar) # open a FITS file
    Cnorm   = hdupar[0].header['cnorm']

    '''
    file_org = DIR_TMP + 'specorg_' + ID0 + '_PA' + PA + '.fits'
    hduorg = fits.open(file_org)
    waveorg = hduorg[1].data['wavelength'] # observed.
    fluxorg = hduorg[1].data['fspec_av'] * Cnorm
    #fluxorg_lam = fnutolam(waveorg, fluxorg)
    print(fluxorg)
    ax1.plot(waveorg, fluxorg * c / np.square(waveorg) / d, marker='', linestyle='-', linewidth=0.5, ms=0.1, color='k', label='')

    file_org = DIR_TMP + 'specorg_pix_' + ID0 + '_PA' + PA + '.fits'
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
    ms     = np.zeros(len(age), dtype='float32')
    f0     = fits.open(DIR_TMP + 'ms_' + ID0 + '_PA' + PA + '.fits')
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
    lib     = fnc.open_spec_fits(ID0, PA, fall=0, tau0=tau0)
    lib_all = fnc.open_spec_fits(ID0, PA, fall=1, tau0=tau0)
    #lib_wid = fnc.open_spec_fits_wid(ID0, PA, fall=1, tau0=tau0)

    II0   = nage #[0,1,2,3] # Number for templates
    iimax = len(II0)-1


    fwuvj = open(ID0 + '_PA' + PA + '_uvj.txt', 'w')
    fwuvj.write('# age uv vj\n')

    Asum = np.sum(A50[:])
    for jj in range(len(II0)):
        ii = int(len(II0) - jj - 1) # from old to young templates.
        if jj == 0:
            y0, x0   = fnc.tmp03(ID0, PA, A50[ii], AAv[0], ii, Z50[ii], zbes, lib_all, tau0=tau0)
            y0p, x0p = fnc.tmp03(ID0, PA, A50[ii], AAv[0], ii, Z50[ii], zbes, lib, tau0=tau0)
            ysump = y0p
            ysum  = y0

            if A50[ii]/Asum > flim:
                ax1.plot(x0, y0 * c/ np.square(x0) / d, '--', lw=0.5, color=col[ii], zorder=-1, label='')

        else:
            y0_r, x0_tmp = fnc.tmp03(ID0, PA, A50[ii], AAv[0], ii, Z50[ii], zbes, lib_all, tau0=tau0)
            y0p, x0p     = fnc.tmp03(ID0, PA, A50[ii], AAv[0], ii, Z50[ii], zbes, lib, tau0=tau0)

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
            y0_wid, x0_wid = fnc.open_spec_fits_dir(ID0, PA, tt, nZ, nn, AAv[0], zbes, A50[tt], tau0=tau0)
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

    flw = open(ID0 + '_PA' + PA + '_mag.txt', 'w')
    flw.write('# ID PA flux_u flux_v flux_j flux_f140w\n')
    flw.write('%s %s %.5f %.5f %.5f %.5f\n'%(ID0, PA, fu_cnv, fv_cnv, fj_cnv, f140_cnv))
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

    #ax1.text(12000, ymax*0.9, '%s'%(ID0), fontsize=10, color='k')
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
    eAAl = np.zeros(len(age),dtype='float32')
    eAAu = np.zeros(len(age),dtype='float32')
    eAMl = np.zeros(len(age),dtype='float32')
    eAMu = np.zeros(len(age),dtype='float32')

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

    dat = np.loadtxt(DIR_TMP + 'spec_obs_' + ID0 + '_PA' + PA + '.cat', comments='#')
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
    import cosmolopy.distance as cd
    import cosmolopy.constants as cc
    cosmo = {'omega_M_0' : 0.27, 'omega_lambda_0' : 0.73, 'h' : 0.72}
    cosmo = cd.set_omega_k_0(cosmo)

    Lsun = 3.839 * 1e33 #erg s-1
    Mpc_cm = 3.08568025e+24 # cm/Mpc
    DL = cd.luminosity_distance(zbes, **cosmo) * Mpc_cm # Luminositydistance in cm
    Cons = (4.*np.pi*DL**2/(1.+zbes))

    dA     = cd.angular_diameter_distance(zbes, **cosmo)
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
    flw = open(ID0 + '_PA' + PA + '_lines_fit.txt', 'w')
    flw.write('# LW flux_line eflux_line flux_cont EW eEW L_line eL_line\n')
    flw.write('# (AA) (Flam_1e-18) (Flam_1e-18) (Flam_1e-18) (AA) (AA) (erg/s) (erg/s)\n')
    flw.write('# Error in EW is 1sigma, by pm eflux_line.\n')
    flw.write('# If EW=-99, it means gaussian fit failed.\n')
    flw.write('# and flux is the sum of excess at WL pm %.1f AA.\n'%(dlw))
    flw.write('# Magnification is corrected; mu=%.3f\n'%(umag))
    try:
        fl = np.loadtxt('table_' + ID0 + '_PA' + PA + '_lines.txt', comments='#')
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

                flux = np.zeros(len(xxs), dtype='float32')
                efl  = np.zeros(len(xxs), dtype='float32')
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
    file = 'chain_' + ID0 + '_PA' + PA + '_corner.cpkl'
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
    ytmp = np.zeros((nmc2,len(ysum)), dtype='float32')
    ytmpmax = np.zeros(len(ysum), dtype='float32')
    ytmpmin = np.zeros(len(ysum), dtype='float32')
    for kk in range(0,nmc2,1):
        nr = np.random.randint(len(samples))
        Av_tmp = samples['Av'][nr]
        for ss in range(len(age)):
            AA_tmp = samples['A'+str(ss)][nr]
            ZZ_tmp = samples['Z'+str(ss)][nr]

            if ss == 0:
                mod0_tmp, xm_tmp = fnc.tmp03(ID0, PA, AA_tmp, Av_tmp, ss, ZZ_tmp, zbes, lib_all, tau0=tau0)
                fm_tmp = mod0_tmp
            else:
                mod0_tmp, xx_tmp = fnc.tmp03(ID0, PA, AA_tmp, Av_tmp, ss, ZZ_tmp, zbes, lib_all, tau0=tau0)
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
    if f_Z_all == 1:
        fileZ = 'Z_' + ID0 + '_PA' + PA + '.cat'
        Zini, chi, Av = np.loadtxt(fileZ, comments='#', unpack=True, usecols=[1, 2, 3+len(age)])
        Atmp  = np.zeros((len(age),len(Zini)), 'float32')
        Ztmp  = np.zeros((len(age),len(Zini)), 'float32')
        for aa in range(len(age)):
            Atmp[aa,:] = np.loadtxt(fileZ, comments='#', unpack=True, usecols=[3+aa])
            Ztmp[aa,:] = np.loadtxt(fileZ, comments='#', unpack=True, usecols=[3+len(age)+1+aa])

        for jj in range(len(Zini)):
            for aa in range(len(age)):
                if aa == 0:
                    y0_r, x0_r    = fnc.tmp03(ID0, PA, Atmp[aa, jj], Av[jj], aa, Ztmp[aa, jj], zbes, lib_all)
                    y0_rp, x0_rp  = fnc.tmp03(ID0, PA, Atmp[aa, jj], Av[jj], aa, Ztmp[aa, jj], zbes, lib)
                else:
                    y0_rr, x0_r     = fnc.tmp03(ID0, PA, Atmp[aa, jj], Av[jj], aa, Ztmp[aa, jj], zbes, lib_all)
                    y0_rrp, x0_rrp  = fnc.tmp03(ID0, PA, Atmp[aa, jj], Av[jj], aa, Ztmp[aa, jj], zbes, lib)
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


    ################
    # RGB
    ################
    try:
    #if 1>0:
        from scipy import misc
        rgb_array = misc.imread('/Users/tmorishita/Box Sync/Research/M18_rgb/rgb_'+str(int(ID0))+'.png')
        axicon = fig.add_axes([0.68, 0.53, 0.4, 0.4])
        axicon.imshow(rgb_array, interpolation='nearest', origin='upper')
        axicon.set_xticks([])
        axicon.set_yticks([])
        #xxl = np.arange(0, twokpc, 0.01)
        #axicon.errorbar(xxl + 0.2, xxl*0+0.2, lw=1., color='w', zorder=1, alpha=1., capsize=0)
        #axicon.text(0.1, 1.5, '5kpc', color='w', fontsize=12)
        #axicon.text(10, 10.0, '%s'%(ID0), color='w', fontsize=14)
        #axicon.text(11, 10.0, '%s'%(ID0), color='w', fontsize=14)
        #axicon.text(10, 11.0, '%s'%(ID0), color='w', fontsize=14)
    except:
        pass
    ####################
    ## Save
    ####################
    #plt.show()
    ax1.legend(loc=1, fontsize=11)
    if figpdf:
        plt.savefig('SPEC_' + ID0 + '_PA' + PA + '_spec.pdf', dpi=300)
    else:
        plt.savefig('SPEC_' + ID0 + '_PA' + PA + '_spec.png', dpi=150)
    plt.close()


def plot_sed_demo(ID0, PA, Z=np.arange(-1.2,0.4249,0.05), age=[0.01, 0.1, 0.3, 0.7, 1.0, 3.0],  f_Z_all=0, tau0=[0.1,0.2,0.3], flim=0.01, figpdf=False):
    DIR_TMP = 'templates/'
    col = ['darkred', 'r', 'coral','orange','g','lightgreen', 'lightblue', 'b','indigo','violet','k']

    nage = np.arange(0,len(age),1)

    from .function_class import Func
    from .basic_func import Basic

    fnc  = Func(Z, nage, dust_model=dust_model) # Set up the number of Age/ZZ
    bfnc = Basic(Z)

    import cosmolopy.distance as cd
    import cosmolopy.constants as cc
    cosmo = {'omega_M_0' : 0.3, 'omega_lambda_0' : 0.7, 'h' : 0.72}
    cosmo = cd.set_omega_k_0(cosmo)
    c = 3e18
    pixelscale = 0.06 # arcsec/pixel
    Mpc_cm = 3.08568025e+24 # cm/Mpc
    m0set = 25.0


    ###########################
    # Open input file
    ###########################
    file = DIR_TMP + 'sim_param_' + ID0 + '_PA' + PA + '.fits'
    hdu0 = fits.open(file) # open a FITS file

    Cnorm = float(hdu0[0].header['Cnorm'])

    Avin = float(hdu0[0].header['AV'])
    Ain  = np.zeros(len(age), dtype='float32')
    Zin  = np.zeros(len(age), dtype='float32')
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
    mag0   = 25.0
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
    file = 'summary_' + ID0 + '_PA' + PA + '.fits'
    hdul = fits.open(file) # open a FITS file

    # Redshift MC
    zp50  = hdul[1].data['zmc'][1]
    zp16  = hdul[1].data['zmc'][0]
    zp84  = hdul[1].data['zmc'][2]

    M50 = hdul[1].data['ms'][1]
    M16 = hdul[1].data['ms'][0]
    M84 = hdul[1].data['ms'][2]
    print('Total stellar mass is %.2e'%(M50))

    A50 = np.zeros(len(age), dtype='float32')
    A16 = np.zeros(len(age), dtype='float32')
    A84 = np.zeros(len(age), dtype='float32')
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

    Z50 = np.zeros(len(age), dtype='float32')
    Z16 = np.zeros(len(age), dtype='float32')
    Z84 = np.zeros(len(age), dtype='float32')
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

            param = np.zeros((Na, 6), dtype='float32')
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
    ms     = np.zeros(len(age), dtype='float32')
    f0     = fits.open(DIR_TMP + 'ms_' + ID0 + '_PA' + PA + '.fits')
    sedpar = f0[1]
    for aa in range(len(age)):
        ms[aa]    = sedpar.data['ML_' +  str(int(NZbest[aa]))][aa]


    ################
    # Open ascii file and stock to array.
    lib     = fnc.open_spec_fits(ID0, PA, fall=0, tau0=tau0)
    lib_all = fnc.open_spec_fits(ID0, PA, fall=1, tau0=tau0)

    II0   = nage #[0,1,2,3] # Number for templates
    iimax = len(II0)-1


    def fnutolam(lam, fnu):
        Ctmp = lam **2/c * 10**((48.6+m0set)/2.5) #/ delx_org
        flam  = fnu / Ctmp
        return flam

    for aa in range(len(II0)):
        y0, x0   = fnc.tmp03(ID0, PA, 1, 0, aa, 0, zbes, lib_all, tau0=tau0)
        #y0p, x0p = fnc.tmp03(ID0, PA, 1, 0, aa, 0, zbes, lib, tau0=tau0)

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
    NDIM = 19

    #ax1.text(12000, ymax*0.9, '%s'%(ID0), fontsize=10, color='k')
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
