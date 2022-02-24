import sys
from gsf import gsf
import argparse

import numpy as np
import sys
import matplotlib.pyplot as plt
import os.path
import string
from astropy.cosmology import WMAP9 as cosmo
import asdf

# From gsf
from gsf.fitting import Mainbody
from gsf.maketmp_filt import maketemp,maketemp_tau
from gsf.function_class import Func,Func_tau
from gsf.basic_func import Basic,Basic_tau

from gsf.maketmp_z0_tau import make_tmp_z0 as make_tmp_z0_tau
from gsf.maketmp_z0 import make_tmp_z0
from gsf.maketmp_z0 import make_tmp_z0_bpass

import timeit
start = timeit.default_timer()

from multiprocessing import Pool,TimeoutError,cpu_count
import multiprocessing
from functools import partial


def merge_asdf(MB, file_out='spec_all.asdf', clean=True):
    '''
    '''
    tree = {}
    entries = ['spec','ML','lick']
    for zz,Z in enumerate(MB.Zall):
        tree_tmp = asdf.open(MB.DIR_TMP + 'spec_all_Z%.1f.asdf'%Z)
        if zz == 0:
            for key in tree_tmp.keys():
                tree[key] = tree_tmp[key]
        else:
            for key in entries:
                tree[key] = tree[key] | tree_tmp[key]                    
    af = asdf.AsdfFile(tree)
    af.write_to(MB.DIR_TMP + file_out, all_array_compression='zlib')

    if clean:
        for zz,Z in enumerate(MB.Zall):
            os.system('rm %s'%(MB.DIR_TMP + 'spec_all_Z%.1f.asdf'%Z))


def mp_func_tau(Zforce, MB=None, lammax=None):
    '''
    '''
    make_tmp_z0_tau(MB, lammax=lammax, Zforce=Zforce)
    return Zforce


def mp_func(Zforce, MB=None, lammax=None):
    '''
    '''
    make_tmp_z0(MB, lammax=lammax, Zforce=Zforce)
    return Zforce


def mp_func_bpass(Zforce, MB=None, lammax=None):
    '''
    '''
    make_tmp_z0_bpass(MB, lammax=lammax, Zforce=Zforce)
    return Zforce


if __name__ == '__main__':
    '''
    '''
    parser = argparse.ArgumentParser(description='Run gsf.')
    parser.add_argument('parfile', metavar='parfile', type=str, help='Configuration file.')
    parser.add_argument('--id', default=1, help='Object ID. This can be anything here.', type=str)
    parser.add_argument('--zmax', default=12.0, help='Maximum redshift for templates.', type=float)
    args = parser.parse_args()

    parfile = args.parfile
    idman = args.id
    zman = args.zmax

    ######################
    # Read from Input file
    ######################
    from gsf.function import read_input
    inputs = read_input(parfile)

    MB = Mainbody(inputs, c=3e18, Mpc_cm=3.08568025e+24, m0set=25.0, pixelscale=0.06, cosmo=cosmo, idman=idman, zman=zman)

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
    #
    # 0. Make basic templates
    #
    lammax = 200000 * (1.+MB.zgal) # AA
    if MB.f_dust:
        lammax = 2000000 * (1.+MB.zgal) # AA

    agents = min(cpu_count(),48)
    chunksize = int(len(MB.Zall)/agents) + 1
    kwargs = {'MB':MB, 'lammax':lammax}
    file_out = 'spec_all.asdf'

    if MB.SFH_FORM == -99:
        if MB.f_bpass == 1:
            with Pool(processes=agents) as pool:
                result = pool.map(partial(mp_func_bpass, **kwargs), MB.Zall, chunksize)
        else:
            with Pool(processes=agents) as pool:
                result = pool.map(partial(mp_func, **kwargs), MB.Zall, chunksize)
    else:
        with Pool(processes=agents) as pool:
            result = pool.map(partial(mp_func_tau, **kwargs), MB.Zall, chunksize)

    # Then add all asdf file into one
    merge_asdf(MB, file_out=file_out, clean=True)
