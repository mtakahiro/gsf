#!/usr/bin/env python
##########################
# 2018.10.05
# version 1.0.0
##########################

#########################################
# What do you need before running this?
#
# 1. Broadband photometry (ID + '_bb_ksirac.cat')
# 2. Moffat parameters
# (ID + '_PA' + PA + '_inp{0,1}_moffat.cat')
# 3. Extracted spectra
# (ID + '_PA' + PA + '_inp0_tmp3.cat')
#########################################
################
# Input
################
#parfile = sys.argv[1]
#fplt    = int(sys.argv[2]) # flag for plot.

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

#if __name__ == "__main__":
def main(parfile, fplt, mcmcplot=True):
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
    MB = Mainbody(inputs)

    ID0  = inputs['ID']
    PA0  = inputs['PA']
    zgal = float(inputs['ZGAL'])
    Cz0  = float(inputs['CZ0'])
    Cz1  = float(inputs['CZ1'])
    try:
        DIR_EXTR = inputs['DIR_EXTR']
    except:
        DIR_EXTR = False
    try:
        fneb = int(inputs['ADD_NEBULAE'])
    except:
        fneb = 0

    DIR_TMP  = inputs['DIR_TEMP']
    DIR_FILT = inputs['DIR_FILT']

    try:
        ftaucomp = inputs['TAU_COMP']
    except:
        print('No entry: TAU_COMP')
        ftaucomp = 0
        print('set to %d' % ftaucomp)
        pass

    if os.path.exists(DIR_TMP) == False:
        os.mkdir(DIR_TMP)

    #
    # Age
    #
    age = inputs['AGE']
    age = [float(x.strip()) for x in age.split(',')]
    nage = np.arange(0,len(age),1)

    #
    # Metallicity
    #
    try:
        ZFIX = float(inputs['ZFIX'])
        Zmin, Zmax = ZFIX, ZFIX+0.0001
        Zall = np.arange(Zmin, Zmax, 0.0001) # in logZsun
    except:
        Zmax, Zmin = float(inputs['ZMAX']), float(inputs['ZMIN'])
        delZ = float(inputs['DELZ'])
        if Zmax == Zmin or delZ==0:
            delZ = 0.0001
            Zall = np.arange(Zmin, Zmax+delZ, delZ) # in logZsun
        else:
            Zall = np.arange(Zmin, Zmax, delZ) # in logZsun

    #
    # Line
    #
    try:
        LW0 = inputs['LINE']
        LW0 = [float(x.strip()) for x in LW0.split(',')]
    except:
        LW0 = []

    #
    # Dust model specification;
    #
    try:
        dust_model = int(inputs['DUST_MODEL'])
    except:
        dust_model = 0


    #
    # Tau for MCMC parameter; not as fitting parameters.
    # = Age bin size;
    #
    tau0 = inputs['TAU0']
    tau0 = [float(x.strip()) for x in tau0.split(',')]
    #
    # IMF
    #
    try:
        nimf = int(inputs['NIMF'])
    except:
        nimf = 0
        print('Cannot find NIMF. Set to %d.'%(nimf))

    #
    # Redshift initial guess.
    #
    zrecom   = zgal
    flag_suc = 1
    #
    # Grism spectrum normalization
    #
    Czrec0   = Cz0
    Czrec1   = Cz1

    #
    # Then load Func and Basic with param range.
    #
    fnc  = Func(Zall, nage, dust_model=dust_model) # Set up the number of Age/ZZ
    bfnc = Basic(Zall)

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


        ######################
        # Make basic templates
        if fplt == 0:
            zmin   = 1.0
            lammax = 80000/(1.+zmin)
            #
            # Then run;
            #
            make_tmp_z0(nimf, Zall, age, lammax=lammax, tau0=tau0, fneb=fneb, DIR_TMP=DIR_TMP)


        # ##################################
        # Start making redshifted templates.
        # Then, fit.
        # ##################################
        if fplt != 2:
            maketemp(inputs, zrecom, Zall, age, fneb=fneb, DIR_TMP=DIR_TMP)
        zprev   = zrecom # redshift from previous run
        Czprev0 = Czrec0
        Czprev1 = Czrec1

        flag_suc, zrecom, Czrec0, Czrec1 = MB.main(ID0, PA0, zrecom, 0, zprev, Czprev0, Czprev1, fzvis=fzvis, fneld=fneld, ntemp=ntemp, mcmcplot=mcmcplot, f_disp=f_disp)
        #flag_suc = 0
        while (flag_suc == 1):
            print('\n\n')
            print('Making templates...')
            print('\n\n')
            maketemp(inputs, zrecom, Zall, age, fneb=fneb, DIR_TMP=DIR_TMP)
            print('\n\n')
            print('Going into another trial with updated templates and redshift.')
            print('\n\n')
            zprev     = zrecom # redshift from previous run
            Czprev0  *= Czrec0
            Czprev1  *= Czrec1
            flag_suc, zrecom, Czrec0, Czrec1 = MB.main(ID0, PA0, zrecom, 1, zprev, Czprev0, Czprev1, fzvis=fzvis, fneld=fneld, ntemp=ntemp, mcmcplot=mcmcplot, f_disp=f_disp)

        #
        # Total calculation time
        #
        stop = timeit.default_timer()
        print('The whole process took;',stop - start)

    if fplt <= 3:
        from .plot_sfh import plot_sfh
        from .plot_sed import plot_sed
        plot_sfh(ID0, PA0, Zall, age, f_comp=ftaucomp, fil_path=DIR_FILT,
        inputs=inputs, dust_model=dust_model, DIR_TMP=DIR_TMP, f_SFMS=True)
        plot_sed(ID0, PA0, Z=Zall, age=age, tau0=tau0, fil_path=DIR_FILT,
        SNlim=1.0, figpdf=False, save_sed=True, inputs=inputs, nmc2=300,
        dust_model=dust_model, DIR_TMP=DIR_TMP, f_label = True)

    if fplt == 4:
        from .plot_sfh import get_evolv
        get_evolv(ID0, PA0, Zall, age, f_comp=ftaucomp, fil_path=DIR_FILT, inputs=inputs, dust_model=dust_model, DIR_TMP=DIR_TMP)

    if fplt == 5:
        from .plot_sfh import plot_evolv
        plot_evolv(ID0, PA0, Zall, age, f_comp=ftaucomp, fil_path=DIR_FILT, inputs=inputs, dust_model=dust_model, DIR_TMP=DIR_TMP, nmc=10)

    if fplt == 6:
        from .plot_sed import plot_corner_physparam_frame,plot_corner_physparam_summary
        #plot_corner, plot_corner_TZ, plot_corner_param2, plot_corner_tmp
        plot_corner_physparam_summary(ID0, PA0, Zall, age, tau0, dust_model=dust_model)
        #plot_corner_physparam_frame(ID0, PA0, Zall, age, tau0, dust_model=dust_model)


    '''
    if fplt == 8:
        from .plot_MZ import plot_mz
        plot_mz(ID0, PA0, Zall, age)
    '''
