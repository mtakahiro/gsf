import numpy as np
import sys
import matplotlib.pyplot as plt
import os.path
import string
from astropy.cosmology import WMAP9 as cosmo
from astropy.io import fits
import asdf

# From gsf
from .fitting import Mainbody
from .maketmp_filt import maketemp,maketemp_tau
from .function_class import Func,Func_tau
from .basic_func import Basic,Basic_tau

import timeit
start = timeit.default_timer()


def run_gsf_template(inputs, fplt=0, tau_lim=0.001, idman=None, nthin=1, delwave=10,
    f_IGM=True):
    '''
    Purpose
    -------
    This is only for 0 and 1, to get templates.
    Not for fitting, nor plotting.

    '''

    MB = Mainbody(inputs, c=3e18, Mpc_cm=3.08568025e+24, m0set=25.0, pixelscale=0.06, cosmo=cosmo, idman=idman)

    if os.path.exists(MB.DIR_TMP) == False:
        os.mkdir(MB.DIR_TMP)

    #
    # Then load Func and Basic with param range.
    #
    if MB.SFH_FORM == -99:
        MB.fnc = Func(MB) # Set up the number of Age/ZZ
        MB.bfnc = Basic(MB)
    else:
        MB.fnc = Func_tau(MB) # Set up the number of Age/ZZ
        MB.bfnc = Basic_tau(MB)


    #
    # 0. Make basic templates
    #
    if fplt == 0 or fplt == 1 or fplt == 2:
        #
        # 0. Make basic templates
        #
        if fplt==0:
            lammax = 40000 * (1.+MB.zgal) # AA
            if MB.f_dust:
                lammax = 2000000 * (1.+MB.zgal) # AA

            if MB.SFH_FORM == -99:
                if MB.f_bpass == 1:
                    from .maketmp_z0 import make_tmp_z0_bpass
                    make_tmp_z0_bpass(MB, lammax=lammax)
                else:
                    from .maketmp_z0 import make_tmp_z0
                    make_tmp_z0(MB, lammax=lammax)
            else:
                from .maketmp_z0_tau import make_tmp_z0
                make_tmp_z0(MB, lammax=lammax)            

    #
    # 1. Start making redshifted templates.
    #
    if fplt<2:
        #
        # 1. Start making redshifted templates.
        #
        if MB.SFH_FORM == -99:
            maketemp(MB, tau_lim=tau_lim, nthin=nthin, delwave=delwave, f_IGM=f_IGM)
        else:
            maketemp_tau(MB, tau_lim=tau_lim, nthin=nthin, delwave=delwave, f_IGM=f_IGM)

    return MB


def run_gsf_all(parfile, fplt, cornerplot=True, f_Alog=True, idman=None, zman=None, f_label=True, f_symbol=True, 
    f_SFMS=False, f_fill=True, save_sed=True, figpdf=False, mmax=300, skip_sfh=False, f_fancyplot=False, 
    skip_zhist=False, tau_lim=0.001, tset_SFR_SED=0.1, f_shuffle=False, amp_shuffle=1e-2, Zini=None, 
    nthin=1, delwave=1, f_plot_resid=False, scale=1e-19, f_plot_filter=True, f_prior_sfh=False, norder_sfh_prior=3):
    '''
    Purpose
    -------
    gsf pipeline, which runs all steps.

    Parameters
    ----------
    delwave : float
        If >0, the input templates get smoothing to delwave. 
        For fsps, this seems to be critical, so it has the same delwave over the template wavelength range.
    '''

    ######################
    # Read from Input file
    ######################
    from .function import read_input
    inputs = read_input(parfile)

    MB = Mainbody(inputs, c=3e18, Mpc_cm=3.08568025e+24, m0set=25.0, pixelscale=0.06, cosmo=cosmo, idman=idman, zman=zman)
    
    # Register some params;
    MB.tau_lim = tau_lim
    MB.nthin = nthin
    MB.delwave = delwave

    if os.path.exists(MB.DIR_TMP) == False:
        os.mkdir(MB.DIR_TMP)

    #
    # Then load Func and Basic with param range.
    #
    if MB.SFH_FORM == -99:
        MB.fnc = Func(MB) # Set up the number of Age/ZZ
        MB.bfnc = Basic(MB)
    else:
        MB.fnc = Func_tau(MB) # Set up the number of Age/ZZ
        MB.bfnc = Basic_tau(MB)

    #
    # Make templates based on input redsfift.
    #
    flag_suc = True
    if fplt == 0 or fplt == 1:
        #
        # 0. Make basic templates
        #
        if fplt==0:
            lammax = 40000 * (1.+MB.zgal) # AA
            if MB.f_dust:
                lammax = 2000000 * (1.+MB.zgal) # AA

            if MB.SFH_FORM == -99:
                if MB.f_bpass == 1:
                    from .maketmp_z0 import make_tmp_z0_bpass
                    make_tmp_z0_bpass(MB, lammax=lammax)
                else:
                    from .maketmp_z0 import make_tmp_z0
                    make_tmp_z0(MB, lammax=lammax)
            else:
                from .maketmp_z0_tau import make_tmp_z0
                make_tmp_z0(MB, lammax=lammax)            

    if not flag_suc:
        sys.exit()

    if fplt <= 2:
        MB.zprev = MB.zgal 
        MB.ndim_keep = MB.ndim
        #
        # 1. Start making redshifted templates, at z=MB.zgal.
        #
        if MB.SFH_FORM == -99:
            flag_suc = maketemp(MB, tau_lim=MB.tau_lim, nthin=MB.nthin, delwave=MB.delwave)
        else:
            flag_suc = maketemp_tau(MB, tau_lim=MB.tau_lim, nthin=MB.nthin, delwave=MB.delwave)

        if not flag_suc:
            return False

        #
        # 2. Main fitting part.
        #
        flag_suc = MB.main(cornerplot=cornerplot, f_shuffle=f_shuffle, amp_shuffle=amp_shuffle, Zini=Zini, 
            f_prior_sfh=f_prior_sfh, norder_sfh_prior=norder_sfh_prior)

        while (flag_suc and flag_suc!=2):

            MB.ndim = MB.ndim_keep
            print('\n\n')
            print('Making templates...')
            print('\n\n')

            # Make temp at the new z
            if MB.SFH_FORM == -99:
                flag_suc = maketemp(MB, tau_lim=MB.tau_lim, nthin=MB.nthin, delwave=MB.delwave)
            else:
                flag_suc = maketemp_tau(MB, tau_lim=MB.tau_lim, nthin=MB.nthin, delwave=MB.delwave)

            print('\n\n')
            print('Going into another round with updated templates and redshift.')
            print('\n\n')

            flag_suc = MB.main(cornerplot=cornerplot, f_shuffle=f_shuffle, amp_shuffle=amp_shuffle, Zini=Zini, 
                f_prior_sfh=f_prior_sfh, norder_sfh_prior=norder_sfh_prior)

        # Total calculation time
        stop = timeit.default_timer()
        print('The whole process took;',stop - start)

    if not flag_suc:
        sys.exit()

    if fplt <= 3 and flag_suc:

        # Use the final redshift;
        hd_sum = fits.open(os.path.join(MB.DIR_OUT, 'summary_%s.fits'%MB.ID))[0].header
        MB.zgal = hd_sum['ZMC']

        if not MB.ztemplate:
            if MB.SFH_FORM == -99:
                flag_suc = maketemp(MB, tau_lim=tau_lim, nthin=nthin, delwave=delwave)
            else:
                flag_suc = maketemp_tau(MB, tau_lim=tau_lim, nthin=nthin, delwave=delwave)
            if not flag_suc:
                return False

        if MB.SFH_FORM == -99:
            from .plot_sfh import plot_sfh
            from .plot_sed import plot_sed            
        else:
            from .plot_sfh import plot_sfh_tau as plot_sfh
            from .plot_sed import plot_sed_tau as plot_sed            

        if not skip_sfh:
            plot_sfh(MB, fil_path=MB.DIR_FILT, mmax=mmax,
            dust_model=MB.dust_model, DIR_TMP=MB.DIR_TMP, f_silence=True, 
            f_SFMS=f_SFMS, f_symbol=f_symbol, skip_zhist=skip_zhist, tau_lim=tau_lim, tset_SFR_SED=tset_SFR_SED)

        plot_sed(MB, fil_path=MB.DIR_FILT,
        figpdf=figpdf, save_sed=save_sed, mmax=mmax,
        dust_model=MB.dust_model, DIR_TMP=MB.DIR_TMP, f_label=f_label, f_fill=f_fill, 
        f_fancyplot=f_fancyplot, f_plot_resid=f_plot_resid, scale=scale, f_plot_filter=f_plot_filter)

    '''
    if fplt == 4:
        from .plot_sfh import get_evolv
        get_evolv(MB, MB.ID, MB.PA, MB.Zall, MB.age, f_comp=MB.ftaucomp, fil_path=MB.DIR_FILT, inputs=MB.inputs, dust_model=MB.dust_model, DIR_TMP=MB.DIR_TMP)


    if fplt == 5:
        from .plot_sfh import plot_evolv
        plot_evolv(MB, MB.ID, MB.PA, MB.Zall, MB.age, f_comp=MB.ftaucomp, fil_path=MB.DIR_FILT, inputs=MB.inputs, dust_model=MB.dust_model, DIR_TMP=MB.DIR_TMP, nmc=10)
    '''

    if fplt == 6:
        # Use the final redshift;
        hd_sum = fits.open(os.path.join(MB.DIR_OUT, 'summary_%s.fits'%MB.ID))[0].header
        MB.zgal = hd_sum['ZMC']
        
        if MB.SFH_FORM == -99:
            flag_suc = maketemp(MB, tau_lim=tau_lim, nthin=nthin, delwave=delwave)
        else:
            flag_suc = maketemp_tau(MB, tau_lim=tau_lim, nthin=nthin, delwave=delwave)

        if MB.SFH_FORM == -99:
            from .plot_sed import plot_corner_physparam_frame,plot_corner_physparam_summary
            plot_corner_physparam_summary(MB)
        else:
            #from .plot_sed_logA import plot_corner_physparam_summary_tau as plot_corner_physparam_summary
            print('One for Tau model is TBD...')


if __name__ == "__main__":
    '''
    '''
    parfile = sys.argv[1]
    fplt = int(sys.argv[2])

    run_gsf_all(parfile, fplt)
