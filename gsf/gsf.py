import numpy as np
import sys
import matplotlib.pyplot as plt
import os.path
import string
from astropy.cosmology import WMAP9 as cosmo
import asdf

# From gsf
from .fitting import Mainbody
from .maketmp_filt import maketemp,maketemp_tau
from .function_class import Func,Func_tau
from .basic_func import Basic,Basic_tau

import timeit
start = timeit.default_timer()


def run_gsf_template(inputs, fplt=0, tau_lim=0.001, idman=None):
    '''
    Purpose:
    ========
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

    #
    # 1. Start making redshifted templates.
    #
    if fplt<2:
        #
        # 1. Start making redshifted templates.
        #
        if MB.SFH_FORM == -99:
            maketemp(MB, tau_lim=tau_lim)
        else:
            maketemp_tau(MB, tau_lim=tau_lim)

    # Read temp from asdf;
    # This has to happend after fplt==1 and before fplt>=2.
    MB.af = asdf.open(MB.DIR_TMP + 'spec_all_' + MB.ID + '.asdf')
    MB.af0 = asdf.open(MB.DIR_TMP + 'spec_all.asdf')

    '''
    #
    # 2. Load templates
    #
    MB.lib = MB.fnc.open_spec_fits(MB, fall=0)
    MB.lib_all = MB.fnc.open_spec_fits(MB, fall=1)
    if MB.f_dust:
        MB.lib_dust = MB.fnc.open_spec_dust_fits(MB, fall=0)
        MB.lib_dust_all = MB.fnc.open_spec_dust_fits(MB, fall=1)
    '''

    # How to get SED?
    if False:
        import matplotlib.pyplot as plt
        for T in MB.age[:1]:
            y0, x0 = MB.fnc.get_template(MB.lib_all, Amp=1.0, T=T, Av=0.0, Z=-1.0, zgal=MB.zgal)
            plt.plot(x0/(1.+MB.zgal),y0,linestyle='-',lw=1.0, label='$T=%.2f$\n$z=%.2f$'%(T,MB.zgal))
            plt.xlim(600,20000)
            plt.xlabel('Rest frame wavelength')
            plt.xscale('log')
        plt.legend(loc=0)
        plt.show()

    return MB


def run_gsf_all(parfile, fplt, cornerplot=True, f_Alog=True, idman=None, f_label=True, f_symbol=True, \
    f_SFMS=True, f_fill=True, save_sed=True, figpdf=False, mmax=300, skip_sfh=False, f_fancyplot=False, \
    skip_zhist=False, tau_lim=0.001, tset_SFR_SED=0.1, f_shuffle=True):
    '''
    Purpose:
    ========
    Run all steps.

    '''

    ######################
    # Read from Input file
    ######################
    from .function import read_input
    inputs = read_input(parfile)

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
    # Make templates based on input redsfift.
    #
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

        #
        # 1. Start making redshifted templates.
        #
        if MB.SFH_FORM == -99:
            maketemp(MB, tau_lim=tau_lim)
        else:
            maketemp_tau(MB, tau_lim=tau_lim)

    # Read temp from asdf;
    # This has to happend after fplt==1 and before fplt>=2.
    MB.af = asdf.open(MB.DIR_TMP + 'spec_all_' + MB.ID + '.asdf')
    MB.af0 = asdf.open(MB.DIR_TMP + 'spec_all.asdf')

    flag_suc = 0
    if fplt <= 2:
        #
        # 2. Mian fitting part.
        #
        MB.zprev = MB.zgal #zrecom # redshift from previous run
        MB.ndim_keep = MB.ndim # This is needed for iteration.
        flag_suc = MB.main(cornerplot=cornerplot, f_shuffle=f_shuffle)
        while (flag_suc and flag_suc!=-1):

            MB.ndim = MB.ndim_keep
            print('\n\n')
            print('Making templates...')
            print('\n\n')
            maketemp(MB)
            print('\n\n')
            print('Going into another trial with updated templates and redshift.')
            print('\n\n')

            flag_suc = MB.main(cornerplot=cornerplot, f_shuffle=f_shuffle)
            # If still in the loop, read again.
            MB.af = asdf.open(MB.DIR_TMP + 'spec_all_' + MB.ID + '.asdf')

        # Total calculation time
        stop = timeit.default_timer()
        print('The whole process took;',stop - start)

    if fplt <= 3 and flag_suc != -1:# and False:
        if f_Alog:
            if MB.SFH_FORM == -99:
                from .plot_sfh_logA import plot_sfh
                from .plot_sed_logA import plot_sed            
            else:
                from .plot_sfh_logA import plot_sfh_tau as plot_sfh
                from .plot_sed_logA import plot_sed_tau as plot_sed            
        else:
            from .plot_sfh import plot_sfh
            from .plot_sed import plot_sed

        if not skip_sfh:
            plot_sfh(MB, f_comp=MB.ftaucomp, fil_path=MB.DIR_FILT, mmax=mmax,
            inputs=MB.inputs, dust_model=MB.dust_model, DIR_TMP=MB.DIR_TMP, f_silence=True, 
            f_SFMS=f_SFMS, f_symbol=f_symbol, skip_zhist=skip_zhist, tau_lim=tau_lim, tset_SFR_SED=tset_SFR_SED)

        plot_sed(MB, fil_path=MB.DIR_FILT,
        figpdf=figpdf, save_sed=save_sed, inputs=MB.inputs, mmax=mmax,
        dust_model=MB.dust_model, DIR_TMP=MB.DIR_TMP, f_label=f_label, f_fill=f_fill, f_fancyplot=f_fancyplot)

    '''
    if fplt == 4:
        from .plot_sfh import get_evolv
        get_evolv(MB, MB.ID, MB.PA, MB.Zall, MB.age, f_comp=MB.ftaucomp, fil_path=MB.DIR_FILT, inputs=MB.inputs, dust_model=MB.dust_model, DIR_TMP=MB.DIR_TMP)


    if fplt == 5:
        from .plot_sfh import plot_evolv
        plot_evolv(MB, MB.ID, MB.PA, MB.Zall, MB.age, f_comp=MB.ftaucomp, fil_path=MB.DIR_FILT, inputs=MB.inputs, dust_model=MB.dust_model, DIR_TMP=MB.DIR_TMP, nmc=10)
    '''

    if fplt == 6:
        from .plot_sed_logA import plot_corner_physparam_frame,plot_corner_physparam_summary
        plot_corner_physparam_summary(MB)
        #plot_corner, plot_corner_TZ, plot_corner_param2, plot_corner_tmp
        #plot_corner_physparam_frame(ID0, PA0, Zall, age, tau0, dust_model=dust_model)


    if fplt == 8:
        '''
        See MZ evolution
        '''
        from .plot_MZ import plot_mz
        plot_mz(MB, MB.ID, MB.PA, MB.Zall, MB.age)


if __name__ == "__main__":
    '''
    '''
    parfile = sys.argv[1]
    fplt = int(sys.argv[2])

    run_gsf_all(parfile, fplt)
