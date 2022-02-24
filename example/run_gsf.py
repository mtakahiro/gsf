import sys
from gsf import gsf
import argparse

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

    gsf.run_gsf_all(args.parfile, args.fplt, idman=args.id, zman=args.z, nthin=1, delwave=args.delwave)
