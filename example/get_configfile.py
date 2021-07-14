import numpy as np

def get_configfile(name=None):
    '''
    Purpose
    -------
    Generate a configuration file.
    '''

    # Version;
    import gsf.__init__ as vers
    ver = vers.__version__

    if name == None:
        name = 'default.input'
        
    fw = open(name,'w')
    fw.write('\
#\n\
# Input file for ver.%s\n\
#\n\
#\n\
# Params for SED\n\
#\n\
TAU0        -1 # in Gyr, tau for age pixels.; 99 for csp, -1 for ssp.\n\
AGE         0.01,0.021,0.0457,0.1,0.21,0.457,1.,2.1,4.57,10.\n\
#AGE_FIX    1.0 # if age is fixed. Comma separated float, taken from AGE.\n\
ZFIX        0.0 # Metalicity fixed to this value.\n\
#ZMAX       0.41 # Max metalicity.\n\
#ZMIN       -0.8 # Min metalicity.\n\
#DELZ       0.05\n\
ZEVOL       0 # Variation in Z at each age pixel; 1=yes, 0=no.\n\
AVMIN       0\n\
AVMAX       4.0\n\
#AVFIX      1.0 # Dust attenuation, fixed to this value.\n\
ZMC         0 # redshift as a free parameter in mcmc; 1=yes, 2=no.\n\
EZL         0.3 # redshift lower error.\n\
EZU         0.3 # redshift upper error.\n\
NIMF        1 # Choice of IMF. 0=Salpeter, 1=Chabrier, 2=Kroupa, 3=van Dokkum, 4=Dave, 5=tabulated, specified in imf.dat file located in the data directory.\n\
ADD_NEBULAE 0 # Add nebular lines; 1=yes, 2=no. This cannot be done when BPASS is on.\n\
logU        -2.0 # Ionizing parameter, in logU.\n\
BPASS       0 # BPASS library; 1=yes, 0=no (fsps). This may take longer to calculate, as it has a higher res than fsps.\n\
#\n\
# params for MCMC\n\
#\n\
NMC         1000\n\
NWALK       50\n\
NMCZ        20\n\
NWALKZ      10\n\
FNELD       nelder # powell\n\
NCPU        0 # Number of multiprocessing.\n\
F_ERR       0\n\
ZVIS        1 # Visual inspection of spectral fit.\n\
#\n\
# Params for data\n\
#\n\
DIR_TEMP    ./templates/ # Directory of the template library. \n\
DIR_EXTR    ./ # Directory of extracted spectra.\n\
#SPEC_FILE  #\n\
DIR_FILT    /Users/tmorishita/GitHub/gsf/gsf/example/filter/\n\
CAT_BB      # Broadband photometric catalog.\n\
FILTER      308,309,310,311,350,351,352,353,354,355,356,357,1,4,6,202,203,204,205\n\
SNLIM       1.\n\
#\n\
# Params for target\n\
#\n\
#ID         \n\
PA          00\n\
#ZGAL       # zpeak from eazy.\n\
CZ0         1.0 # Initial guess of spectral normalization for G102.\n\
CZ1         1.0 # Initial guess of spectral normalization for G141.\n\
LINE        0. # Emission line, iin AA.\n\
#\n\
# Grism\n\
#\n\
MORP        moffat\n\
MORP_FILE   ./output/l3_nis_f200w_G150C_s00010_moffat.txt\n\
#\n\
# DUST\n\
#\n\
#FIR_FILTER	325,326,329,330,1014\n\
#CAT_BB_DUST	fir_flux.txt\n\
#TDUST_LOW	0\n\
#TDUST_HIG	22\n\
#TDUST_DEL	1\n\
#DIR_DUST    /PATH/To/DL07spec/\n\
    '%ver)


if __name__ == "__main__":
    '''
    '''
    import sys
    try:
        name = sys.argv[1]
    except:
        name = None
    get_configfile(name=name)
