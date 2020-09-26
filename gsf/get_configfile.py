import numpy as np

def get_configfile():
    '''
    Purpose:
    ========
    To generate a configuration file.
    '''

    # Version;
    import __init__ as vers
    ver = vers.__version__

    fw = open('default.input','w')
    fw.write('\
#\n\
# Input file for ver.%s\n\
#\n\
# params for MCMC\n\
#\n\
NMC         1000\n\
NWALK       50\n\
NMCZ        20\n\
NWALKZ      10\n\
TAU0        -1\n\
FNELD       0\n\
NCPU        0 # Number of multiprocessing.\n\
F_ERR       0\n\
#\n\
# Params for data\n\
#\n\
DIR_TEMP    ./templates/ # Directory of extracted spectra. \n\
DIR_EXTR    ./ # Directory of extracted spectra.\n\
#SPEC_FILE  #\n\
DIR_FILT    /Users/tmorishita/GitHub/gsf/gsf/example/filter/\n\
CAT_BB      # Directory of extracted spectra.\n\
FILTER      308,309,310,311,350,351,352,353,354,355,356,357,1,4,6,202,203,204,205\n\
SNLIM       1.\n\
#\n\
# Params for SED\n\
#\n\
AGE         0.01,0.021,0.0457,0.1,0.21,0.457,1.,2.1,4.57,10.\n\
ZFIX        0.0 # Max metalicity\n\
#ZMAX       0.41 # Max metalicity\n\
#ZMIN       -0.8 # Min metalicity\n\
#DELZ       0.1\n\
#AVFIX      1.0\n\
AVMIN       0\n\
AVMAX       4.0\n\
ZEVOL       0 # Evolution in Z; 1=yes, 2=no.\n\
ZMC         0 # redshift as a free parameter in mcmc.\n\
EZL         0.3  #0.216 # redshift lower error.\n\
EZU         0.3 #206 # redshift upper error.\n\
NIMF        1 # Choice of IMF. 0=Salpeter, 1=Chabrier, 2=Kroupa, 3=van Dokkum, 4=Dave, 5=tabulated, specified in imf.dat file located in the data directory.\n\
#\n\
# Params for target\n\
#\n\
#ID         \n\
PA          00\n\
#ZGAL       # zpeak from eazy.\n\
CZ0         1.0 # Initial guess of spectral normalization for G102.\n\
CZ1         1.0 # Initial guess of spectral normalization for G141.\n\
LINE        0. # Emission line, iin AA.\n\
ZVIS        0 # Visual inspection of redshift. 1=yes, 0=no. If you are not confident with zgal, cz1/2, then ZVIS should be 1 for iteration.\n\
ADD_NEBULAE 0 # This cannot be done with BPASS yet.\n\
logU        -2.0\n\
BPASS       0 # BPASS may take longer, as it has higher res than fsps.\n\
#\n\
# Grism\n\
#\n\
MORP        moffat\n\
MORP_FILE   ./output/l3_nis_f200w_G150C_s00010_moffat.txt\n\
ZVIS        1\n\
    '%ver)


if __name__ == "__main__":
    '''
    '''
    get_configfile()
