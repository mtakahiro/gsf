import sys
from gsf import gsf
import argparse
import os
import fsps

def test_fitting():
    '''
    '''
    parfile = 'gds_43114.input'
    fplt = 0
    id = 0
    z = 1.9
    delwave = 5.0

    os.chdir('./example/')

    gsf.run_gsf_all(parfile, fplt, nthin=1, delwave=delwave, zman=z) #, idman=id
