import sys
from gsf import gsf

import argparse
parser = argparse.ArgumentParser(description='Run gsf.')
parser.add_argument('parfile', metavar='parfile', type=str, help='Configuration file.')
parser.add_argument('fplt', metavar='fplt', type=int, help='Flag for run (int: 0,1,2,3).')
parser.add_argument('--id', default=None, help='Manual input for object ID.')
args = parser.parse_args()

gsf.run_gsf_all(args.parfile, args.fplt, idman=args.id)
