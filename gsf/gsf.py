##########################
# 2018.10.05
# version 1.0.0
##########################

import numpy as np
import sys
import matplotlib.pyplot as plt
import os.path
import string
from astropy.cosmology import WMAP9 as cosmo

# From gsf
from .fitting import Mainbody
from .function_class import Func
from .basic_func import Basic
from .maketmp_filt import maketemp
from .maketmp_z0 import make_tmp_z0
from .maketmp_z0 import make_tmp_z0_bpass

import timeit
start = timeit.default_timer()

def run_gsf_template(inputs, fplt=0):
    '''
    Purpose:
    ==========
    This is only for 0 and 1, to get templates.
    Not for fitting, nor plotting.

    '''

    MB = Mainbody(inputs, c=3e18, Mpc_cm=3.08568025e+24, m0set=25.0, pixelscale=0.06, cosmo=cosmo)
    if os.path.exists(MB.DIR_TMP) == False:
        os.mkdir(MB.DIR_TMP)

    MB.fnc  = Func(MB) # Set up the number of Age/ZZ
    MB.bfnc = Basic(MB)

    #
    # 0. Make basic templates
    #
    if fplt<1:
        lammax = 80000 / (1.+MB.zgal) # AA
        if MB.f_bpass == 1:
            make_tmp_z0_bpass(MB, lammax=lammax)
        else:
            make_tmp_z0(MB, lammax=lammax)

    #
    # 1. Start making redshifted templates.
    #
    if fplt<2:
        maketemp(MB)

    #
    # 2. Load templates
    #
    MB.lib = MB.fnc.open_spec_fits(MB, fall=0)
    MB.lib_all = MB.fnc.open_spec_fits(MB, fall=1)
    if MB.f_dust:
        MB.lib_dust     = MB.fnc.open_spec_dust_fits(MB, fall=0)
        MB.lib_dust_all = MB.fnc.open_spec_dust_fits(MB, fall=1)

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


def run_gsf_all(parfile, fplt, cornerplot=True):
    '''
    #########################################
    # What do you need before running this?
    #
    # 1. Broadband photometry (ID + '_bb_ksirac.cat')
    # 2. Moffat parameters
    # (ID + '_PA' + PA + '_inp{0,1}_moffat.cat')
    # 3. Extracted spectra
    # (ID + '_PA' + PA + '_inp0_tmp3.cat')
    #########################################
    '''
    flag_suc = 0 #True

    ######################
    # Read from Input file
    ######################
    from .function import read_input
    inputs = read_input(parfile)

    MB = Mainbody(inputs, c=3e18, Mpc_cm=3.08568025e+24, m0set=25.0, pixelscale=0.06, cosmo=cosmo)

    if os.path.exists(MB.DIR_TMP) == False:
        os.mkdir(MB.DIR_TMP)

    #
    # Then load Func and Basic with param range.
    #
    MB.fnc  = Func(MB) # Set up the number of Age/ZZ
    MB.bfnc = Basic(MB)

    #
    # Make templates based on input redsfift.
    #
    if fplt == 0 or fplt == 1 or fplt == 2:
        #
        # 0. Make basic templates
        #
        if fplt == 0:
            lammax = 40000 * (1.+MB.zgal) # AA
            if MB.f_bpass == 1:
                make_tmp_z0_bpass(MB, lammax=lammax)
            else:
                make_tmp_z0(MB, lammax=lammax)


        #
        # 1. Start making redshifted templates.
        #
        if fplt < 2:
            maketemp(MB)


        #
        # 2. Mian fitting part.
        #
        MB.zprev = MB.zgal #zrecom # redshift from previous run

        flag_suc = MB.main(0, cornerplot=cornerplot)

        while (flag_suc and flag_suc!=-1):

            print('\n\n')
            print('Making templates...')
            print('\n\n')
            maketemp(MB)
            print('\n\n')
            print('Going into another trial with updated templates and redshift.')
            print('\n\n')

            flag_suc = MB.main(1, cornerplot=cornerplot)

        # Total calculation time
        stop = timeit.default_timer()
        print('The whole process took;',stop - start)


    if fplt <= 3 and flag_suc != -1:
        from .plot_sfh import plot_sfh
        from .plot_sed import plot_sed
        #plot_sfh(MB, f_comp=MB.ftaucomp, fil_path=MB.DIR_FILT,
        #inputs=MB.inputs, dust_model=MB.dust_model, DIR_TMP=MB.DIR_TMP, f_SFMS=True)
        plot_sed(MB, fil_path=MB.DIR_FILT,
        SNlim=1.0, figpdf=False, save_sed=True, inputs=MB.inputs, nmc_rand=100,
        dust_model=MB.dust_model, DIR_TMP=MB.DIR_TMP, f_label=True)


    if fplt == 4:
        from .plot_sfh import get_evolv
        get_evolv(MB, MB.ID, MB.PA, MB.Zall, MB.age, f_comp=MB.ftaucomp, fil_path=MB.DIR_FILT, inputs=MB.inputs, dust_model=MB.dust_model, DIR_TMP=MB.DIR_TMP)


    if fplt == 5:
        from .plot_sfh import plot_evolv
        plot_evolv(MB, MB.ID, MB.PA, MB.Zall, MB.age, f_comp=MB.ftaucomp, fil_path=MB.DIR_FILT, inputs=MB.inputs, dust_model=MB.dust_model, DIR_TMP=MB.DIR_TMP, nmc=10)


    if fplt == 6:
        from .plot_sed import plot_corner_physparam_frame,plot_corner_physparam_summary
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
    fplt    = int(sys.argv[2])

    run_gsf_all(parfile, fplt)
