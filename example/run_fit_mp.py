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


def mp_func(index, **kwargs):
    '''
    '''
    fplt = kwargs['fplt']
    z = kwargs['z']
    id = kwargs['id']
    delwave = kwargs['delwave']
    parfile = kwargs['parfile']

    gsf.run_gsf_all(parfile, fplt, idman=str(id[index]), zman=z, nthin=1, delwave=delwave)


if __name__ == "__main__":
    '''
    '''
    parser = argparse.ArgumentParser(description='Run gsf.')
    parser.add_argument('parfile', metavar='parfile', type=str, help='Configuration file.')
    parser.add_argument('fplt', metavar='fplt', type=int, help='Flag for run (int: 0,1,2,3).')
    parser.add_argument('--id', default=None, help='Manual input for object ID.')
    parser.add_argument('--z', default=None, help='Redshift.', type=float)
    parser.add_argument('--delwave', default=20, help='Delta wavelength (AA).', type=float)
    args = parser.parse_args()

    from gsf.function import read_input
    from astropy.io import ascii
    inputs = read_input(args.parfile)
    MB = Mainbody(inputs, c=3e18, Mpc_cm=3.08568025e+24, m0set=25.0, pixelscale=0.06, cosmo=cosmo, idman=1, zman=args.z)
    fd = ascii.read(MB.CAT_BB)
    ids = fd['id']#[:1]
    indices = np.arange(0,len(ids),1)

    kwargs = {}
    kwargs['id'] = ids
    kwargs['parfile'] = args.parfile
    kwargs['fplt'] = args.fplt
    kwargs['z'] = args.z
    kwargs['delwave'] = args.delwave

    agents = min(cpu_count(),48)
    chunksize = int(len(fd['id'].value)/agents) + 1
    with Pool(processes=agents) as pool:
        result = pool.map(partial(mp_func, **kwargs), indices, chunksize)
        


