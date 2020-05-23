#!/usr/bin/env python
##########################
# 2018.10.05
# version 1.0.0
##########################

import numpy as np
import sys
import matplotlib.pyplot as plt
import os.path
import string

# Custom modules
from .fitting import Mainbody
from .function_class import Func
from .basic_func import Basic
from .maketmp_filt import maketemp
from .maketmp_z0 import make_tmp_z0

import timeit
start = timeit.default_timer()

def run_gsf(parfile, fplt, mcmcplot=True):
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

    fplt = int(fplt)
    #
    # Get info from param file.
    #
    input0 = []
    input1 = []
    file = open(parfile,'r')
    while 1:
        line = file.readline()
        if not line:
            break
        else:
            cols = str.split(line)
            if len(cols)>0 and cols[0] != '#':
                    input0.append(cols[0])
                    input1.append(cols[1])
    file.close()

    inputs = {}
    for i in range(len(input0)):
        inputs[input0[i]]=input1[i]

    ######################
    # Read from Input file
    ######################
    from astropy.cosmology import WMAP9 as cosmo
    MB = Mainbody(inputs, c=3e18, Mpc_cm=3.08568025e+24, m0set=25.0, pixelscale=0.06, cosmo=cosmo)

    if os.path.exists(MB.DIR_TMP) == False:
        os.mkdir(MB.DIR_TMP)

    #
    # Redshift initial guess.
    #
    zrecom   = MB.zgal
    flag_suc = 0 #True

    #
    # Grism spectrum normalization
    #
    Czrec0   = MB.Cz0
    Czrec1   = MB.Cz1

    #
    # Then load Func and Basic with param range.
    #
    fnc  = Func(MB.ID, MB.PA, MB.Zall, MB.nage, dust_model=MB.dust_model) # Set up the number of Age/ZZ
    bfnc = Basic(MB.Zall)

    #
    # Make templates based on input redsfift.
    #
    if fplt == 0 or fplt == 1 or fplt == 2:
        #
        # Params for MCMC
        #
        nmc      = int(inputs['NMC'])
        nwalk    = int(inputs['NWALK'])
        nmc_cz   = int(inputs['NMCZ'])
        nwalk_cz = int(inputs['NWALKZ'])
        f_Zevol  = int(inputs['ZEVOL'])
        fzvis    = int(inputs['ZVIS'])
        fneld    = int(inputs['FNELD'])
        try:
            ntemp = int(inputs['NTEMP'])
        except:
            ntemp = 1
        try:
            if int(inputs['DISP']) == 1:
                f_disp = True
            else:
                f_disp = False
        except:
            f_disp = False


        #
        # 0. Make basic templates
        #
        if fplt == 0:
            zmin   = 1.0
            lammax = 80000/(1.+zmin)
            make_tmp_z0(MB, lammax=lammax)


        #
        # 1. Start making redshifted templates.
        #
        if fplt != 2:
            maketemp(MB, zrecom)

        #
        # 2. Mian fitting part.
        #
        zprev   = zrecom # redshift from previous run
        Czprev0 = Czrec0
        Czprev1 = Czrec1

        flag_suc, zrecom, Czrec0, Czrec1 = MB.main(MB.ID, MB.PA, zrecom, 0, zprev, Czprev0, Czprev1, fzvis=MB.fzvis, fneld=MB.fneld, ntemp=MB.ntemp, mcmcplot=mcmcplot, f_disp=MB.f_disp)

        while (flag_suc == 1):
            print('\n\n')
            print('Making templates...')
            print('\n\n')
            maketemp(inputs, zrecom, MB.Zall, MB.age, fneb=MB.fneb, DIR_TMP=MB.DIR_TMP)
            print('\n\n')
            print('Going into another trial with updated templates and redshift.')
            print('\n\n')
            zprev     = zrecom # redshift from previous run
            Czprev0  *= Czrec0
            Czprev1  *= Czrec1
            flag_suc, zrecom, Czrec0, Czrec1 = MB.main(MB.ID, MB.PA, zrecom, 1, zprev, Czprev0, Czprev1, fzvis=MB.fzvis, fneld=MB.fneld, ntemp=MB.ntemp, mcmcplot=mcmcplot, f_disp=MB.f_disp)

        # Total calculation time
        stop = timeit.default_timer()
        print('The whole process took;',stop - start)

    if fplt <= 3 and flag_suc >= 0:
        from .plot_sfh import plot_sfh
        from .plot_sed import plot_sed
        #plot_sfh(MB, MB.ID, MB.PA, MB.Zall, MB.age, f_comp=MB.ftaucomp, fil_path=MB.DIR_FILT,
        #inputs=inputs, dust_model=MB.dust_model, DIR_TMP=MB.DIR_TMP, f_SFMS=True)
        plot_sed(MB, MB.ID, MB.PA, Z=MB.Zall, age=MB.age, tau0=MB.tau0, fil_path=MB.DIR_FILT,
        SNlim=1.0, figpdf=False, save_sed=True, inputs=inputs, nmc2=300,
        dust_model=MB.dust_model, DIR_TMP=MB.DIR_TMP, f_label = True)

    if fplt == 4:
        from .plot_sfh import get_evolv
        get_evolv(MB, MB.ID, MB.PA, MB.Zall, MB.age, f_comp=MB.ftaucomp, fil_path=MB.DIR_FILT, inputs=inputs, dust_model=MB.dust_model, DIR_TMP=MB.DIR_TMP)

    if fplt == 5:
        from .plot_sfh import plot_evolv
        plot_evolv(MB, MB.ID, MB.PA, MB.Zall, MB.age, f_comp=MB.ftaucomp, fil_path=MB.DIR_FILT, inputs=inputs, dust_model=MB.dust_model, DIR_TMP=MB.DIR_TMP, nmc=10)

    if fplt == 6:
        from .plot_sed import plot_corner_physparam_frame,plot_corner_physparam_summary
        plot_corner_physparam_summary(MB, MB.ID, MB.PA, MB.Zall, MB.age, MB.tau0, dust_model=MB.dust_model)
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

    run_gsf(parfile, fplt)
