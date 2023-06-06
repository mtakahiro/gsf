from scipy import asarray as ar,exp
import numpy as np
import sys
from scipy.integrate import simps
import pickle as cPickle
import os
import scipy.interpolate as interpolate
from scipy.interpolate import interp1d
import logging
from colorama import Fore, Back, Style
from datetime import datetime

################
# Line library
################
LN0 = ['Mg2', 'Ne5', 'O2', 'Htheta', 'Heta', 'Ne3', 'Hdelta', 'Hgamma', 'Hbeta', 'O3L', 'O3H', 'Mgb', 'Halpha', 'S2L', 'S2H']
LW0 = [2800, 3347, 3727, 3799, 3836, 3869, 4102, 4341, 4861, 4960, 5008, 5175, 6563, 6717, 6731]
fLW = np.zeros(len(LW0), dtype='int') # flag.
c = 3.e18 # A/s


def print_err(msg, exit=False, details=None):
    '''
    '''
    now = datetime.now()
    print(Fore.RED)
    print('$$$ =================== $$$')
    print('$$$  gsf error message  $$$')
    print('%s'%now)
    if not details == None:
        # @@@ This does not work??
        exc_type, exc_obj, exc_tb = details
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

    print('$$$ =================== $$$')
    print(Fore.RED)
    print(msg)
    print(Fore.RED)
    print(Style.RESET_ALL)

    if exit:
        print(Fore.CYAN)
        print('Exiting.')
        print(Style.RESET_ALL)
        sys.exit()


def str2bool(v):
    '''
    '''
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_uvbeta(lm, flam, zbes, lam_blue=1650, lam_red=2300):
    '''
    Purpose
    -------
    get UV beta_lambda slope.

    Parameters
    ----------
    lm : float array
        in lambda
    flam : float array
        in flambda 
    '''
    con_uv = (lm/(1.+zbes)>lam_blue) & (lm/(1.+zbes)<lam_red)
    #flam = fnutolam(lm,fl)
    try:
        fit_results = np.polyfit(np.log10(lm/(1.+zbes))[con_uv], np.log10(flam)[con_uv], 1) #, w=flam[con_uv])
        beta = fit_results[0]
        if np.isnan(beta):
            beta = -99
    except:
        beta = -99
    return beta


def func_tmp(xint,eobs,fmodel):
    '''
    '''
    int_tmp = np.exp(-0.5 * ((xint-fmodel)/eobs)**2)
    return int_tmp


def get_chi2(fy, ey, wht3, ysump, ndim_eff, SNlim=1.0, f_chind=True, f_exclude=False, xbb=None, x_ex=None):
    '''
    '''
    from scipy import special
    if f_chind:
        conw = (wht3>0) & (ey>0) & (fy/ey>SNlim)
    else:
        conw = (wht3>0) & (ey>0)

    resid = fy-ysump
    chi2 = sum((np.square(resid) * np.sqrt(wht3))[conw])

    chi_nd = 0.0
    if f_chind:
        f_ex = np.zeros(len(fy), 'int')
        for ii in range(len(fy)):
            if f_exclude:
                if xbb[ii] in x_ex:
                    f_ex[ii] = 1

        con_up = (ey>0) & (fy/ey<=SNlim) & (f_ex == 0)
        x_erf = (ey[con_up] - ysump[con_up]) / (np.sqrt(2) * ey[con_up])
        f_erf = special.erf(x_erf)
        chi_nd = np.sum( np.log(np.sqrt(np.pi / 2) * ey[con_up] * (1 + f_erf)) )

    # Number of degree;
    con_nod = (wht3>0) & (ey>0) #& (fy/ey>SNlim)
    nod = int(len(wht3[con_nod])-ndim_eff)
    if nod>0:
        fin_chi2 = (chi2 - 2 * chi_nd) / nod
    else:
        fin_chi2 = -99

    return chi2,fin_chi2


def get_ind(wave,flux):
    '''
    Gets Lick index for input

    Returns
    -------
    equivalent width
    '''    
    lml     = [4268, 5143, 5233, 5305, 5862, 4828, 4628, 4985, 5669, 5742, 4895, 4895, 5818, 6068]
    lmcl    = [4283, 5161, 5246, 5312, 5879, 4848, 4648, 5005, 5689, 5762, 5069, 5154, 5938, 6191]
    lmcr    = [4318, 5193, 5286, 5352, 5911, 4877, 4668, 5925, 5709, 5782, 5134, 5197, 5996, 6274]
    lmr     = [4336, 5206, 5318, 5363, 5950, 4892, 4688, 5945, 5729, 5802, 5366, 5366, 6105, 6417]

    W = np.zeros(len(lml), dtype='float')
    for ii in range(len(lml)):
        con_cen = (wave>lmcl[ii]) & (wave<lmcr[ii])
        con_sid = ((wave<lmcl[ii]) & (wave>lml[ii])) | ((wave<lmr[ii]) & (wave>lmcr[ii]))

        Ic = np.mean(flux[con_cen])
        Is = np.mean(flux[con_sid])

        delam = lmcr[ii] - lmcl[ii]

        if ii < 10:
            W[ii] = (1. - Ic/Is) * delam
        elif 1. - Ic/Is > 0:
            W[ii] = -2.5 * np.log10(1. - Ic/Is)
        else:
            W[ii] = -99

    return W


def printProgressBar (iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r", emojis=['']):
    '''
    Call in a loop to create terminal progress bar.

    Parameters
    ----------
    iteration : int 
        current iteration
    total : int 
        total iterations
    prefix : str
        prefix string
    suffix : str
        suffix string
    decimals : int
        positive number of decimals in percent complete
    length : int
        character length of bar
    fill : str
        bar fill character
    printEnd : str 
        end character
    '''
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    if fill == None:
        if float(percent) < 33:
            fill = emojis[0]
        elif float(percent) < 66:
            fill = emojis[1]
        elif float(percent) < 99:
            fill = emojis[2]
        else:
            fill = emojis[3]

    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    string = '(%d/%d)'%(iteration,total)
    print(f'\r{prefix} |{bar}| {percent}% {suffix} {string}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


def get_input():
    '''
    Gets a default dictionary for input params.

    '''
    inputs = {'ID':'10000', 'PA':'00', 'ZMC':0.01, 'CZ0':1.0, 'CZ1':1.0, 'BPASS':0,
    'DIR_OUT':'./output/', 'DIR_TEMP':'./templates/', 'DIR_FILT':'./filter/',
    'NIMF':0, 'NMC':100, 'NWALK':50, 'NMCZ':30, 'NWALKZ':20,
    'ZEVOL':0, 'ZVIS':0, 'FNELD':'differential_evolution', 'MC_SAMP':'EMCEE'}

    return inputs


def read_input(parfile):
    '''
    Gets info from param file.

    Returns
    -------
    Input dictionary.

    '''
    input0 = []
    input1 = []
    file = open(parfile,'r')
    while 1:
        line = file.readline()
        if not line:
            break
        else:
            cols = str.split(line)
            if len(cols)>0 and cols[0][0] != '#':
                input0.append(cols[0])
                input1.append(cols[1])
    file.close()
    inputs = {}
    for i in range(len(input0)):
        inputs[input0[i]]=input1[i]

    return inputs


def write_input(inputs, file_out='gsf.input'):
    '''
    Gets an ascii format param file.    

    Returns
    -------
    file_out
    '''
    import gsf
    fw = open(file_out, 'w')
    fw.write('# gsf ver %s\n'%(gsf.__version__))
    for key in inputs.keys():
        fw.write('%s %s\n'%(key,inputs[key]))
    return True


def loadcpkl(cpklfile):
    """
    Load cpkl files.
    """
    import pickle
    # msg = 'cpkl will be deprecated from gsf. Rerun your fit with the latest version, so your sampler will be saved in asdf.'
    # print_err(msg)

    if not os.path.isfile(cpklfile): raise ValueError(' ERR: cannot find the input file')
    f = open(cpklfile, 'rb') #, encoding='ISO-8859-1')

    if sys.version_info.major == 2:
        data = pickle.load(f)
    elif sys.version_info.major == 3:
        data = pickle.load(f, encoding='latin-1')

    f.close()
    return data


def get_leastsq(MB, ZZtmp, fneld, age, fit_params, residual, fy, ey, wht, ID0, 
    chidef=None, Zbest=0, f_keep=False, f_only_spec=False):
    '''
    Get initial parameters at various Z
    '''
    from lmfit import Model, Parameters, minimize, fit_report, Minimizer

    if len(fy)<2:
        print('Not enough data for quick fit. Exiting.')
        return False

    file = 'Z_' + ID0 + '.cat'
    fwz = open(file, 'w')
    fwz.write('# ID Zini chi/nu AA Av Zbest\n')

    if fneld == 1:
        fit_name = 'nelder'
    elif fneld == 0:
        fit_name = 'powell'
    elif fneld == 2:
        fit_name = 'leastsq'
    else:
        fit_name = fneld

    fwz.write('# minimizer: %s\n' % fit_name)

    if MB.has_ZFIX:
        ZZtmp = [MB.ZFIX]

    for zz in range(len(ZZtmp)):
        ZZ = ZZtmp[zz]
        for aa in range(MB.npeak):
            if MB.ZEVOL == 1 or aa == 0:
                fit_params['Z'+str(aa)].value = ZZ

        f_fir = False
        out_tmp = minimize(residual, fit_params, args=(fy, ey, wht, f_fir), 
            method=fit_name, kws={'f_only_spec':f_only_spec})
            
        csq = out_tmp.chisqr
        rcsq = out_tmp.redchi
        fitc = [csq, rcsq] # Chi2, Reduced-chi2

        fwz.write('%s %.2f %.5f'%(ID0, ZZ, fitc[1]))

        AA_tmp = np.zeros(MB.npeak, dtype='float')
        ZZ_tmp = np.zeros(MB.npeak, dtype='float')
        for aa in range(MB.npeak):
            AA_tmp[aa] = out_tmp.params['A'+str(aa)].value
            fwz.write(' %.5f'%(AA_tmp[aa]))

        Av_tmp = out_tmp.params['AV'].value
        fwz.write(' %.5f'%(Av_tmp))
        for aa in range(MB.npeak):
            if MB.ZEVOL == 1 or aa == 0:
                ZZ_tmp[aa] = out_tmp.params['Z'+str(aa)].value
                fwz.write(' %.5f'%(ZZ_tmp[aa]))

        fwz.write('\n')
        if chidef==None:
            chidef = fitc[1]
            out = out_tmp
        elif fitc[1]<chidef:
            chidef = fitc[1]
            out = out_tmp

    fwz.close()

    if not f_keep:
        os.system('rm %s'%file)
    
    return out,chidef,Zbest


def check_rejuv(age,SF,MS,SFMS_50,lm_old=10.0,delMS=0.2):
    '''
    A Function to check rejuvenation.

    Parameters
    ----------
    delMS : float
        Scatter around the Main Sequence. 0.2 dex, in default.
    '''
    age_set = 1.0
    con_old = (age>=age_set)
    con_mid = (age<age_set) & (age>np.min(age))

    f_quen  = 0
    f_rejuv = 0.
    t_quench= 10.0
    t_rejuv = np.min(age)
    if np.max(MS[con_old])>lm_old:

        # Set the most recent quenching;
        #for ii in range(len(age)-1,-1,-1): # 4 to 0.01
        for ii in range(len(age)): # 0.01 to 4
            #if (age[ii]<age_set) and (age[ii]>np.min(age)) and SF[ii,2]<SFMS_50[ii]-delMS:
            if (age[ii]<age_set) and (age[ii]>np.min(age)) and SF[ii,2]<SFMS_50[ii]-delMS and f_rejuv == 0:
                if ii>0 and f_quen==0:
                    t_rejuv = age[ii-1]
                f_quen  = 1
            if f_quen==1 and SF[ii,2]>SFMS_50[ii]-delMS:
                    t_quench = age[ii-1] # Time the most recent quenching has set.
                    f_rejuv  = 1
                    break

    #print(f_rejuv,t_quench,t_rejuv)
    return f_rejuv,t_quench,t_rejuv


def get_SFMS(red,age,mass,IMF=1,get_param=False):
    '''
    Gets SFMS at age ago from z=red.

    Parameters
    ----------
    red : float
        Observed redshift
    age : array
        lookback time, in Gyr.
    mass : array
        stellar mass (array) at each age, in Msun (not logM). 

    Returns
    -------
    SFR, in logMsun/yr.

    Notes
    -----
    From Speagle+14 Eq28.
    Chabrier IMF, in default
    '''
    from astropy.cosmology import WMAP9
    cosmo = WMAP9

    CIMF = 0
    if IMF == 0:#
        CIMF = 0.23
        print('SFMS is shifted to Salpeter IMF.')
    elif IMF == 2:
        CIMF = 0.04
        print('SFMS is shifted to Kroupa IMF.')

    x = np.log10(mass) - CIMF #np.arange(6,13,0.1)
    tz = cosmo.age(z=red).value - age # in Gyr
    alp = (0.84 - 0.026*tz)
    beta = - (6.51 - 0.11*tz)
    y1 = alp * x + beta # in log Msun/yr
    con = (y1<=0)
    y1[con] = -10
    if get_param:
        return y1, [alp,beta]
    return y1


def fit_spec(lm, fobs, eobs, ftmp):
    '''
    Fitting. (Not sure)
    '''
    s = np.sum(fobs*ftmp/eobs**2)/np.sum(ftmp**2/eobs**2)
    chi2 = np.sum(((fobs-s*ftmp)/eobs)**2)
    return chi2, s


def fit_specphot(lm, fobs, eobs, ftmp, fbb, ebb, ltmp_bb, ftmp_bb):
    I1   = np.sum(fobs*ftmp/eobs**2) + np.sum(fbb*ftmp_bb/ebb**2)
    I2   = np.sum(ftmp**2/eobs**2)   + np.sum(ftmp_bb**2/ebb**2)
    s    = I1/I2
    chi2 = np.sum(((fobs-s*ftmp)/eobs)**2) + np.sum(((fbb-s*ftmp_bb)/ebb)**2)
    return chi2, s


def SFH_del(t0, tau, A, tt=None, minsfr = 1e-10):
    '''
    SFH
    '''
    try:
        if tt == None:
            tt = np.arange(0.,10,0.1)
    except:
        pass
    sfr = np.zeros(len(tt), dtype='float')+minsfr
    sfr[:] = A * (tt[:]-t0) * np.exp(-(tt[:]-t0)/tau)
    con = (tt[:]-t0<0)
    sfr[:][con] = minsfr
    return sfr


def SFH_dec(t0, tau, A, tt=None, minsfr = 1e-10):
    '''
    '''
    try:
        if tt == None:
            tt = np.arange(0.,10,0.1)
    except:
        pass
    sfr = np.zeros(len(tt), dtype='float')+minsfr
    sfr[:] = A * (np.exp(-(tt[:]-t0)/tau))
    con = (tt[:]-t0<0)
    sfr[:][con] = minsfr
    return sfr


def SFH_cons(t0, tau, A, tt=None, minsfr = 1e-10):
    '''
    '''
    try:
        if tt == None:
            tt = np.arange(0.,10,0.1)
    except:
        pass
    sfr = np.zeros(len(tt), dtype='float')+minsfr
    sfr[:] = A #* (np.exp(-(tt[:]-t0)/tau))
    con = (tt[:]<t0) | (tt[:]>tau)
    sfr[:][con] = minsfr
    return sfr


def get_Fint(lmtmp, ftmp, lmin=1400, lmax=1500):
    '''
    Parameters
    ----------
    lmtmp : 
        Rest frame wave (AA)
    ftmp :
        Fnu ()

    Returns
    -------
    integrated flux.
    '''
    
    con = (lmtmp>lmin) & (lmtmp<lmax) & (ftmp>0)
    if len(lmtmp[con])>0:
        lamS,spec = lmtmp[con], ftmp[con] # Two columns with wavelength and flux density
        # Equivalent to the following;
        I1 = simps(spec*lamS*1.,lamS)
        I2 = simps(lamS*1.,lamS)
        fnu = I1
    return fnu


def get_Fuv(lmtmp, ftmp, lmin=1400, lmax=1500):
    '''Get RF UV (or any wavelength) flux density.

    Parameters
    ----------
    lmtmp : float array
        Rest-frame wavelength, in AA.
    ftmp : float array
        Fnu
    
    Returns
    -------
    Flux density estimated over lmin:lmax. Not integrated sum.
    '''
    con = (lmtmp>lmin) & (lmtmp<lmax) & (ftmp>0)
    if len(lmtmp[con])>0:
        lamS,spec = lmtmp[con], ftmp[con] # Two columns with wavelength and flux density
        I1 = simps(spec*lamS*1.,lamS)
        I2 = simps(lamS*1.,lamS)
        fnu = I1/I2
    else:
        fnu = None
    return fnu


def data_int(lmobs, lmtmp, ftmp):
    '''

    Parameters
    ----------
    lmobs : 
        Observed wavelength.
    lmtmp, ftmp: 
        Those to be interpolated.
    '''
    ftmp_int  = np.interp(lmobs,lmtmp,ftmp) # Interpolate model flux to observed wavelength axis.
    return ftmp_int


def fnutonu(fnu, m0set=25.0, m0input=-48.6):
    '''
    Converts from Fnu (cgs) to Fnu (m0=m0set)
    
    Parameters
    ----------
    fnu : float array
        flux in cgs, with magnitude zero point of m0input.
    m0set : float
        Target mag zero point.
    m0input : float
        Original value for magzp. If erg/s/cm2/Hz, -48.6.
    '''
    Ctmp = 10**((m0set-m0input)/2.5)
    fnu_new = fnu * Ctmp
    return fnu_new


def flamtonu(lam, flam, m0set=25.0, m0=-48.6):
    '''
    Converts from Flam to Fnu, with mag zeropoint of m0set.
    
    '''
    Ctmp = lam**2/c * 10**((m0set-m0)/2.5) #/ delx_org
    fnu = flam * Ctmp
    return fnu


def fnutolam(lam, fnu, m0set=25.0, m0=-48.6):
    '''
    Converts from Fnu to Flam, from mag zeropoint of m0set (to -48.6).

    Parameters
    ----------
    m0set : float
        current magzp.
    m0 : float
        target magzp. The default, -48.6, is for flam (erg/s/cm2/lambda).
    '''
    Ctmp = lam**2/c * 10**((m0set-m0)/2.5)
    flam = fnu / Ctmp
    return flam


def delta(x,A):
    '''
    '''
    yy = np.zeros(len(x),float)
    iix = np.argmin(np.abs(x))
    if len(x)%2 == 0:
        logger= logging.getLogger( __name__ )
        print(logger)
        print('Input array has an even number.')
        yy[iix] = A/2.
        yy[iix+1] = A/2.
    else:
        yy[iix] = A
    return yy


def gauss(x,A,sig):
    return A * np.exp(-0.5*x**2/sig**2)


def moffat(xx, A, x0, gamma, alp):
    yy = A * (1. + (xx-x0)**2/gamma**2)**(-alp)
    return yy


def get_fit(x, y, xer, yer, nsfh:str = 'Del.'):
    '''
    '''
    from lmfit import Model, Parameters, minimize, fit_report, Minimizer

    fit_params = Parameters()
    #fit_params.add('t0', value=1., min=0, max=np.max(tt))
    fit_params.add('t0', value=.5, min=0, max=14)
    fit_params.add('tau', value=.1, min=0, max=100)
    fit_params.add('A', value=1, min=0, max=5000)

    def residual_tmp(pars):
        vals = pars.valuesdict()
        t0_tmp, tau_tmp, A_tmp = vals['t0'],vals['tau'],vals['A']

        if nsfh == 'Del.':
            model = SFH_del(t0_tmp, tau_tmp, A_tmp, tt=x)
        elif nsfh == 'Decl.':
            model = SFH_dec(t0_tmp, tau_tmp, A_tmp, tt=x)
        elif nsfh == 'Cons.':
            model = SFH_cons(t0_tmp, tau_tmp, A_tmp, tt=x)

        con = (model>0)
        resid = (np.log10(model[con]) - y[con]) / yer[con]
        return resid

    out = minimize(residual_tmp, fit_params, method='powell')
    print(fit_report(out))

    t0 = out.params['t0'].value
    tau = out.params['tau'].value
    A = out.params['A'].value
    param = [t0, tau, A]

    keys = fit_report(out).split('\n')
    for key in keys:
        if key[4:7] == 'chi':
            skey = key.split(' ')
            csq  = float(skey[14])
        if key[4:7] == 'red':
            skey = key.split(' ')
            rcsq = float(skey[7])

    return param, rcsq


def savecpkl(data, cpklfile, verbose=True):
    """
    Save data into cpklfile.
    """
    if verbose: print(' => Saving data to cpklfile '+cpklfile)
    f = open(cpklfile,'wb')
    cPickle.dump(data, f, 2)
    f.close()


def apply_dust(yy, xx, nr, Av, dust_model=0):
	'''
	xx : float array
		RF Wavelength
	'''
	if dust_model == 0:
		yyd, xxd, nrd = dust_calz(xx, yy, Av, nr)
	elif dust_model == 1:
		yyd, xxd, nrd = dust_mw(xx, yy, Av, nr)
	elif dust_model == 2: # LMC
		yyd, xxd, nrd = dust_gen(xx, yy, Av, nr, Rv=4.05, gamma=-0.06, Eb=2.8)
	elif dust_model == 3: # SMC
		yyd, xxd, nrd = dust_gen(xx, yy, Av, nr, Rv=4.05, gamma=-0.42, Eb=0.0)
	elif dust_model == 4: # Kriek&Conroy with gamma=-0.2
		yyd, xxd, nrd = dust_kc(xx, yy, Av, nr, Rv=4.05, gamma=-0.2)
	else:
		yyd, xxd, nrd = dust_calz(xx, yy, Av, nr)
	return yyd, xxd, nrd


def dust_gen(lm, fl, Av, nr, Rv=4.05, gamma=-0.05, Eb=3.0, lmlimu=3.115, lmv=5000/10000, f_Alam=False):
    '''
    For general purpose (Noll+09).
    This function is much better than previous, but is hard to impliment for the current version.
    A difference from dust_gen is Eb is defined as a function of gamma.

    Parameters
    ----------
    lm : float array
        wavelength, at RF.
    fl : float array
        fnu
    Av : float
        in mag
    nr : int array
        index, to be used for sorting.
    Rv : 
        from Calzetti+00
    gamma :
        gamma.
    Eb:
        Eb
    '''
    Kl = np.zeros(len(lm), dtype='float')

    lmm  = lm/10000. # in micron
    con1 = (lmm<=0.63)
    con2 = (lmm>0.63)  & (lmm<=lmlimu)
    con3 = (lmm>lmlimu)

    Kl1 = (2.659 * (-2.156 + 1.509/lmm[con1] - 0.198/lmm[con1]**2 + 0.011/lmm[con1]**3) + Rv)
    Kl2 = (2.659 * (-1.857 + 1.040/lmm[con2]) + Rv)
    Kl3 = (2.659 * (-1.857 + 1.040/lmlimu + lmm[con3] * 0) + Rv)

    #nr0 = nr[con0]
    nr1 = nr[con1]
    nr2 = nr[con2]
    nr3 = nr[con3]

    #lmm0 = lmm[con0]
    lmm1 = lmm[con1]
    lmm2 = lmm[con2]
    lmm3 = lmm[con3]

    #fl0 = fl[con0]
    fl1 = fl[con1]
    fl2 = fl[con2]
    fl3 = fl[con3]

    Kl   = np.concatenate([Kl1,Kl2,Kl3])
    nrd  = np.concatenate([nr1,nr2,nr3])
    lmmc = np.concatenate([lmm1,lmm2,lmm3])
    flc  = np.concatenate([fl1,fl2,fl3])

    # Bump;
    lm0   = 2175 / 10000 # micron m by Noll et al. (2009)
    dellm = 350 / 10000 # micron m by Noll et al. (2009)
    D = Eb * (lmmc * dellm)**2 / ((lmmc**2-lm0**2)**2+(lmmc * dellm)**2)
    #
    Alam   = Av / Rv * (Kl + D) * (lmmc / lmv)**gamma
    fl_cor = flc[:] * 10**(-0.4*Alam[:])

    if f_Alam:
        return fl_cor, lmmc*10000., nrd, Alam
    else:
        return fl_cor, lmmc*10000., nrd


def dust_kc(lm, fl, Av, nr, Rv=4.05, gamma=0, lmlimu=3.115, lmv=5000/10000, f_Alam=False):
    '''
    Dust model by Kriek&Conroy13
    
    Parameters
    ----------
    lm : float array
        RF Wavelength.
    fl : float array
        in fnu
    Av : float
        in mag
    nr : int array
        index, to be used for sorting.
    Rv : float
        from Calzetti+00
    gamma : float
        See Eq.1
    '''
    Kl = np.zeros(len(lm), dtype='float')

    lmm  = lm/10000. # in micron
    con1 = (lmm<=0.63)
    con2 = (lmm>0.63)  & (lmm<=lmlimu)
    con3 = (lmm>lmlimu)

    Kl1 = (2.659 * (-2.156 + 1.509/lmm[con1] - 0.198/lmm[con1]**2 + 0.011/lmm[con1]**3) + Rv)
    Kl2 = (2.659 * (-1.857 + 1.040/lmm[con2]) + Rv)
    Kl3 = (2.659 * (-1.857 + 1.040/lmlimu + lmm[con3] * 0) + Rv)

    #nr0 = nr[con0]
    nr1 = nr[con1]
    nr2 = nr[con2]
    nr3 = nr[con3]

    #lmm0 = lmm[con0]
    lmm1 = lmm[con1]
    lmm2 = lmm[con2]
    lmm3 = lmm[con3]

    #fl0 = fl[con0]
    fl1 = fl[con1]
    fl2 = fl[con2]
    fl3 = fl[con3]

    Kl   = np.concatenate([Kl1,Kl2,Kl3])
    nrd  = np.concatenate([nr1,nr2,nr3])
    lmmc = np.concatenate([lmm1,lmm2,lmm3])
    flc  = np.concatenate([fl1,fl2,fl3])

    Eb = 0.85 - 1.9 * gamma
    #print('Eb is %.2f'%(Eb))

    # Bump;
    lm0   = 2175 / 10000 # micron m by Noll et al. (2009)
    dellm = 350 / 10000 # micron m by Noll et al. (2009)
    D = Eb * (lmmc * dellm)**2 / ((lmmc**2-lm0**2)**2+(lmmc * dellm)**2)
    #
    Alam   = Av / Rv * (Kl + D) * (lmmc / lmv)**gamma
    fl_cor = flc[:] * 10**(-0.4*Alam[:])

    if f_Alam:
        return fl_cor, lmmc*10000., nrd, Alam
    else:
        return fl_cor, lmmc*10000., nrd


def dust_calz(lm, fl, Av:float, nr, Rv:float = 4.05, lmlimu:float = 3.115, f_Alam:bool = False):
    '''
    Parameters
    ----------
    lm : float array
        wavelength, at RF.
    fl : float array
        fnu
    Av : float
        in mag
    nr : int array
        index, to be used for sorting.
    Rv : float
        from Calzetti+00
    lmlimu : float
        Upper limit. 2.2 in Calz+00
    '''
    Kl = lm[:]*0 #np.zeros(len(lm), dtype='float')
    nrd = lm[:]*0 #np.zeros(len(lm), dtype='float')
    lmmc = lm[:]*0 #np.zeros(len(lm), dtype='float')
    flc = lm[:]*0 #np.zeros(len(lm), dtype='float')

    lmm = lm/10000. # in micron
    con1 = (lmm<=0.63)
    con2 = (lmm>0.63)  & (lmm<=lmlimu)
    con3 = (lmm>lmlimu)

    Kl[con1] = (2.659 * (-2.156 + 1.509/lmm[con1] - 0.198/lmm[con1]**2 + 0.011/lmm[con1]**3) + Rv)
    Kl[con2] = (2.659 * (-1.857 + 1.040/lmm[con2]) + Rv)
    Kl[con3] = (2.659 * (-1.857 + 1.040/lmlimu + lmm[con3] * 0) + Rv)

    nrd[con1] = nr[con1]
    nrd[con2] = nr[con2]
    nrd[con3] = nr[con3]

    lmmc[con1] = lmm[con1]
    lmmc[con2] = lmm[con2]
    lmmc[con3] = lmm[con3]

    flc[con1] = fl[con1]
    flc[con2] = fl[con2]
    flc[con3] = fl[con3]

    Alam = Kl * Av / Rv
    fl_cor = flc[:] * 10**(-0.4*Alam[:])

    if f_Alam:
        return fl_cor, lmmc*10000., nrd, Alam
    else:
        return fl_cor, lmmc*10000., nrd


def dust_mw(lm, fl, Av, nr, Rv=3.1, f_Alam=False):
    '''
    Parameters
    ----------
    lm : float array
        wavelength, at RF, in AA.
    fl : float array
        fnu
    Av : float
        mag
    nr : int array
        index, to be used for sorting.
    Rv : float
        3.1 for MW.
    '''
    Kl = np.zeros(len(lm), dtype='float')

    lmm  = lm/10000. # into micron
    xx   = 1./lmm

    con0 = (xx>=8.0)
    con1 = (xx>=5.9) & (xx<8.0)
    con2 = (xx>=3.3) & (xx<5.9)
    con3 = (xx>=1.1) & (xx<3.3)
    con4 = (xx<1.1)

    nr0 = nr[con0]
    nr1 = nr[con1]
    nr2 = nr[con2]
    nr3 = nr[con3]
    nr4 = nr[con4]

    lmm0 = lmm[con0]
    lmm1 = lmm[con1]
    lmm2 = lmm[con2]
    lmm3 = lmm[con3]
    lmm4 = lmm[con4]

    fl0 = fl[con0]
    fl1 = fl[con1]
    fl2 = fl[con2]
    fl3 = fl[con3]
    fl4 = fl[con4]

    ax4 =  0.574 * (1./lmm4)**1.61
    bx4 = -0.527 * (1./lmm4)**1.61

    yy  = ((1./lmm3) - 1.82)
    ax3 = 1. + 0.17699 * yy - 0.50447 * yy**2 - 0.02427 * yy**3 + 0.72085 * yy**4\
          + 0.01979 * yy**5 - 0.77530 * yy**6 + 0.32999 * yy**7
    bx3 = 1.41338 * yy + 2.28305 * yy**2 + 1.07233 * yy**3 - 5.38434 * yy**4\
          - 0.62251 * yy**5 + 5.30260 * yy**6 - 2.09002 * yy**7

    Fax2 = Fbx2 = lmm2 * 0
    ax2  = 1.752 - 0.316 * (1./lmm2) - 0.104/(((1./lmm2)-4.67)**2+0.341) + Fax2
    bx2  = -3.090 + 1.825 * (1./lmm2) + 1.206/(((1./lmm2)-4.62)**2+0.263) + Fbx2

    Fax1 = -0.04473 * ((1./lmm1) - 5.9)**2 - 0.009779 * ((1./lmm1) - 5.9)**3
    Fbx1 = 0.2130 * ((1./lmm1) - 5.9)**2 + 0.1207 * ((1./lmm1) - 5.9)**3
    ax1  = 1.752 - 0.316 * (1./lmm1) - 0.104/(((1./lmm1)-4.67)**2+0.341) + Fax1
    bx1  = -3.090 + 1.825 * (1./lmm1) + 1.206/(((1./lmm1)-4.62)**2+0.263) + Fbx1

    Fax0 = -0.04473 * ((1./lmm0) - 5.9)**2 - 0.009779 * ((1./lmm0) - 5.9)**3
    Fbx0 = 0.2130 * ((1./lmm0) - 5.9)**2 + 0.1207 * ((1./lmm0) - 5.9)**3
    ax0  = 1.752 - 0.316 * (1./lmm0) - 0.104/(((1./lmm0)-4.67)**2+0.341) + Fax0
    bx0  = -3.090 + 1.825 * (1./lmm0) + 1.206/(((1./lmm0)-4.62)**2+0.263) + Fbx0

    nrd  = np.concatenate([nr0,nr1,nr2,nr3,nr4])
    lmmc = np.concatenate([lmm0,lmm1,lmm2,lmm3,lmm4])
    flc  = np.concatenate([fl0,fl1,fl2,fl3,fl4])
    ax   = np.concatenate([ax0,ax1,ax2,ax3,ax4])
    bx   = np.concatenate([bx0,bx1,bx2,bx3,bx4])

    Alam   = Av * (ax + bx / Rv)

    if False:
        import matplotlib.pyplot as plt
        plt.plot(1/lmmc,Alam/Av,linestyle='-')
        plt.xlim(0,3)
        plt.ylim(0,2.0)
        plt.show()

    fl_cor = flc[:] * 10**(-0.4*Alam[:])

    if f_Alam:
        return fl_cor, lmmc*10000., nrd, Alam
    else:
        return fl_cor, lmmc*10000., nrd


def check_line(data,wave,wht,model):
    '''
    '''
    R_grs = 50
    dw  = 10
    ldw = 5
    dlw = R_grs * 2
    lsig = 1.5 # significance of lines.
    ii0 = 0 + dw
    ii9 = len(data) - dw
    wd  = data * wht
    er  = 1./np.sqrt(wht)
    wht2 = wht

    for ii in range(ii0, ii9, 1):
        concont = (((wave>wave[ii]-dw*R_grs) & (wave<wave[ii]-(dw-ldw)*R_grs)) \
                   | ((wave<wave[ii]+dw*R_grs) & ((wave>wave[ii]+(dw-ldw)*R_grs))))

        xcont = wave[concont]
        ycont = data[concont]
        wycont = wht[concont]

        if len(xcont)>5:
            try:
                z = np.polyfit(xcont, ycont, 1, w=wycont)
                p = np.poly1d(z)
                fconttmp  = p(wave[ii])
                fconttmp1 = p(wave[ii-1])
                fconttmp2 = p(wave[ii+1])

                if data[ii] > er[ii]*lsig + fconttmp and data[ii-1] > er[ii-1]*lsig + fconttmp1 and data[ii+1] > er[ii+1]*lsig + fconttmp2:
                    #print wave[ii]/(1.+zgal), dlw/(1+zgal)
                    for jj in range(len(LW)):
                        #print wave[ii],  LW[jj]*(1.+zgal) - dlw, LW[jj]*(1.+zgal) + dlw, LN[jj]
                        if wave[ii] > LW[jj]*(1.+zgal) - dlw\
                           and wave[ii] < LW[jj]*(1.+zgal) + dlw:
                            wht2[ii-dw:ii+dw] *= 0
                            fLW[jj] = 1
                            print(p, LN[jj], fconttmp, data[ii], wave[ii]/(1.+zgal))
                elif wht2[ii] != 0:
                    wht2[ii] = wht[ii]

            except Exception:
                print('Error in Line Check.')
                pass

    return wht2


def filconv_cen(band0, l0, f0, DIR='FILT/'):
    '''
    Convolution of templates with filter response curves.

    Parameters
    ----------
    '''

    fnu  = np.zeros(len(band0), dtype='float')
    lcen = np.zeros(len(band0), dtype='float')
    for ii in range(len(band0)):
        fd = np.loadtxt(DIR + '%s.fil'%str(band0[ii]), comments='#')
        lfil = fd[:,1]
        ffil = fd[:,2]

        lmin  = np.min(lfil)
        lmax = np.max(lfil)
        imin = 0
        imax = 0

        lcen[ii] = np.sum(lfil*ffil)/np.sum(ffil)

        lamS,spec = l0, f0 #Two columns with wavelength and flux density
        lamF,filt = lfil, ffil #Two columns with wavelength and response in the range [0,1]
        #filt_int = np.interp(lamS,lamF,filt)  #Interpolate Filter to common(spectra) wavelength axis
        fint = interpolate.interp1d(lamF, filt, kind='nearest', fill_value="extrapolate")
        filt_int = fint(lamS)
        filtSpec = filt_int * spec #Calculate throughput
        wht = 1. #/(er1[con_rf])**2

        if len(lamS)>0: #./3*len(x0[con_org]): # Can be affect results.
            I1  = simps(spec/lamS**2*c*filt_int*lamS,lamS)   #Denominator for Fnu
            I2  = simps(filt_int/lamS,lamS)                  #Numerator
            fnu[ii] = I1/I2/c         #Average flux density
        else:
            I1  = 0
            I2  = 0
            fnu[ii] = 0

    return lcen, fnu


def filconv_fast(filts, band, l0, f0, fw=False):
    '''
    Parameters
    ----------
    filts, band : 
        From MB.filts and MB.band, respectively.
    f0: 
        Flux for spectrum, in fnu
    l0: 
        Wavelength for spectrum, in AA (that matches filter response curve's.)

    '''
    fnu  = np.zeros(len(filts), dtype='float')
    lcen = np.zeros(len(filts), dtype='float')
    if fw:
        fwhm = np.zeros(len(filts[:]), dtype='float')

    for ii in range(len(filts[:])):
        lfil = band['%s_lam'%(filts[ii])]
        ffil = band['%s_res'%(filts[ii])]

        if fw:
            ffil_cum = np.cumsum(ffil)
            ffil_cum/= ffil_cum.max()
            con      = (ffil_cum>0.05) & (ffil_cum<0.95)
            fwhm[ii] = np.max(lfil[con]) - np.min(lfil[con])

        lmin  = np.min(lfil)
        lmax  = np.max(lfil)
        imin  = 0
        imax  = 0

        con = (l0>lmin) & (l0<lmax) #& (f0>0)
        lcen[ii] = np.sum(lfil*ffil)/np.sum(ffil)
        if len(l0[con])>1:
            lamS,spec = l0[con], f0[con]                     # Two columns with wavelength and flux density
            lamF,filt = lfil, ffil                 # Two columns with wavelength and response in the range [0,1]
            #filt_int = np.interp(lamS,lamF,filt)  # Interpolate Filter to the same wavelength axis of l0, or lamS
            fint = interpolate.interp1d(lamF, filt, kind='nearest', fill_value="extrapolate")
            filt_int = fint(lamS)

            wht = 1.


            # This does not work sometimes;
            delS = lamS[1]-lamS[0]
            I1 = np.sum(spec/lamS**2*c*filt_int*lamS*delS)   #Denominator for Fnu
            I2 = np.sum(filt_int/lamS*delS)                  #Numerator
            if I2>0:
                fnu[ii] = I1/I2/c #Average flux density
            else:
                fnu[ii] = 0
        else:
            fnu[ii] = 0

    if fw:
        return lcen, fnu, fwhm
    else:
        return lcen, fnu


def filconv(band0, l0, f0, DIR, fw=False, f_regist=True, MB=None):
    '''
    Parameters
    ----------
    f0 : float array
        Flux for spectrum, in fnu
    l0 : float array
        Wavelength for spectrum, in AA (that matches filter response curve's.)
    f_regist : bool
        If True, read filter response curves and register those to MB.
    '''
    if MB==None:
        f_regist = True
    if f_regist:
        lfil_lib = {}
        ffil_lib = {}

    fnu = np.zeros_like(band0, dtype=float)
    lcen = np.zeros_like(band0, dtype=float)
    if fw:
        fwhm = np.zeros_like(band0, dtype=float)
    
    for ii in range(len(band0)):
        if not f_regist:
            try:
                lfil = MB.lfil_lib['%s'%str(band0[ii])]
                ffil = MB.ffil_lib['%s'%str(band0[ii])]
                lmin = MB.lfil_lib['%s_lmin'%str(band0[ii])]
                lmax = MB.lfil_lib['%s_lmax'%str(band0[ii])]
                lcen[ii] = MB.lfil_lib['%s_lcen'%str(band0[ii])]
                if fw:
                    fwhm = MB.filt_fwhm
            except:
                f_regist = True
                lfil_lib = {}
                ffil_lib = {}
                
        if f_regist:
            fd = np.loadtxt(DIR + '%s.fil'%str(band0[ii]), comments='#')
            lfil = fd[:,1]
            ffil = fd[:,2]
            ffil /= np.max(ffil)
            lmin = np.min(lfil)
            lmax = np.max(lfil)

            if fw:
                ffil_cum = np.cumsum(ffil)
                ffil_cum/= ffil_cum.max()
                con = (ffil_cum>0.05) & (ffil_cum<0.95)
                fwhm[ii] = np.max(lfil[con]) - np.min(lfil[con])

            con = (l0>lmin) & (l0<lmax)
            delw = np.nanmin(np.diff(l0))
            if delw > np.nanmin(np.diff(ffil)):
                lfil_new = np.arange(lmin,lmax,delw)
                fint = interpolate.interp1d(lfil, ffil, kind='nearest', fill_value="extrapolate")
                ffil = fint(lfil_new)
                lfil = lfil_new

            lfil_lib['%s'%str(band0[ii])] = lfil
            ffil_lib['%s'%str(band0[ii])] = ffil

            lcen[ii] = np.sum(lfil*ffil)/np.sum(ffil)
            lfil_lib['%s_lmin'%str(band0[ii])] = lmin
            lfil_lib['%s_lmax'%str(band0[ii])] = lmax
            lfil_lib['%s_lcen'%str(band0[ii])] = lcen[ii]

        con = (l0>lmin) & (l0<lmax) #& (f0>0)
        if len(l0[con])>1:
            fint = interp1d(lfil, ffil, kind='nearest', fill_value="extrapolate")
            filt_int = fint(l0[con])

            # This does not work sometimes;
            #delS = l0[con][1]-l0[con][0]
            I1 = np.sum(f0[con]/l0[con]**2*c*filt_int*l0[con])
            I2 = np.sum(filt_int/l0[con])
            if I2>0:
                fnu[ii] = I1/I2/c
            else:
                fnu[ii] = 0
        else:
            fnu[ii] = 0

    if MB != None and f_regist:
        MB.lfil_lib = lfil_lib
        MB.ffil_lib = ffil_lib
        if fw:
            MB.filt_fwhm = fwhm
        
    if fw:
        return lcen, fnu, fwhm
    else:
        return lcen, fnu


"""
def filconv(band0, l0, f0, DIR, fw=False, f_regist=True, MB=None):
    '''
    This one does not improve much.
    
    Parameters
    ----------
    f0 : float array
        Flux for spectrum, in fnu
    l0 : float array
        Wavelength for spectrum, in AA (that matches filter response curve's.)
    f_regist : bool
        If True, read filter response curves and register those to MB.
    '''
    if MB==None:
        f_regist = True
    if f_regist:
        lfil_lib = {}
        ffil_lib = {}

    if fw:
        fwhm = np.zeros_like(band0, dtype=float)

    if not f_regist:
        try:
            ffil_lib = MB.ffil_lib
            if fw:
                fwhm = MB.filt_fwhm
        except:
            f_regist = True
            ffil_lib = {}
            # lfils = np.zeros_like(band0, dtype=float)
            # ffils = np.zeros_like(band0, dtype=float)
            # lmins = np.zeros_like(band0, dtype=float)
            # lmaxs = np.zeros_like(band0, dtype=float)

    if f_regist:
        lcen = np.zeros_like(band0, dtype=float)
        for ii in range(len(band0)):
            ffil_lib['%s'%band0[ii]] = {}
            
            fd = np.loadtxt(DIR + '%s.fil'%str(band0[ii]), comments='#')
            lfil = fd[:,1]
            ffil = fd[:,2]
            ffil /= np.max(ffil)
            lmin = np.min(lfil)
            lmax = np.max(lfil)

            if fw:
                ffil_cum = np.cumsum(ffil)
                ffil_cum/= ffil_cum.max()
                con = (ffil_cum>0.05) & (ffil_cum<0.95)
                fwhm[ii] = np.max(lfil[con]) - np.min(lfil[con])

            con = (l0>lmin) & (l0<lmax)
            delw = np.nanmin(np.diff(l0))
            if delw > np.nanmin(np.diff(ffil)):
                lfil_new = np.arange(lmin,lmax,delw)
                fint = interpolate.interp1d(lfil, ffil, kind='nearest', fill_value="extrapolate")
                ffil = fint(lfil_new)
                lfil = lfil_new

            ffil_lib['%s'%band0[ii]]['lfil'] = lfil
            ffil_lib['%s'%band0[ii]]['ffil'] = ffil
            ffil_lib['%s'%band0[ii]]['lmin'] = lmin
            ffil_lib['%s'%band0[ii]]['lmax'] = lmax
            lcen[ii] = np.sum(lfil*ffil)/np.sum(ffil)
        
        ffil_lib['lcen'] = lcen

    fnu = np.zeros_like(band0, dtype=float)
    for ii in range(len(band0)):
        con = (l0>ffil_lib['%s'%band0[ii]]['lmin']) & (l0<ffil_lib['%s'%band0[ii]]['lmax']) #& (f0>0)

        if len(l0[con])>1:
            fint = interp1d(ffil_lib['%s'%band0[ii]]['lfil'], ffil_lib['%s'%band0[ii]]['ffil'], kind='nearest', fill_value="extrapolate")
            filt_int = fint(l0[con])

            # This does not work sometimes;
            I1 = np.sum(f0[con]/l0[con]**2*c*filt_int*l0[con])
            I2 = np.sum(filt_int/l0[con])
            if I2>0:
                fnu[ii] = I1/I2/c
            else:
                fnu[ii] = 0
        else:
            fnu[ii] = 0

    if MB != None and f_regist:
        MB.ffil_lib = ffil_lib
        if fw:
            MB.filt_fwhm = fwhm
        
    if fw:
        return ffil_lib['lcen'], fnu, fwhm
    else:
        return ffil_lib['lcen'], fnu
"""


def fil_fwhm(band0, DIR):
    '''
    Parameters
    ----------
    f0 :
        in fnu
    '''
    fwhm = np.zeros(len(band0), dtype='float')
    for ii in range(len(band0)):
        fd = np.loadtxt(DIR + band0[ii] + '.fil', comments='#')
        lfil = fd[:,1]
        ffil = fd[:,2]

        fsum = np.sum(ffil)
        fcum = np.zeros(len(ffil), dtype='float')
        lam0,lam1 = 0,0

        for jj in range(len(ffil)):
            fcum[jj] = np.sum(ffil[:jj])/fsum
            if lam0 == 0 and fcum[jj]>0.05:
                lam0 = lfil[jj]
            if lam1 == 0 and fcum[jj]>0.95:
                lam1 = lfil[jj]

        fwhm[ii] = lam1 - lam0

    return fwhm


def calc_Dn4(x0, y0, z0):
    con1 = (x0/(1+z0)>3750) & (x0/(1+z0)<3950)
    con2 = (x0/(1+z0)>4050) & (x0/(1+z0)<4250)
    D41  = np.average(y0[con1])
    D42  = np.average(y0[con2])
    if D41>0 and D42>0:
        D4 = D42/D41
        return D4
    else:
        return -99


def gaus(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))


def detect_line(xcont, ycont, wycont, zgal):
    ################
    # Line library
    ################
    LN = ['Mg2', 'Ne5', 'O2', 'Htheta', 'Heta', 'Ne3', 'Hdelta', 'Hgamma', 'Hbeta', 'O3', 'Halpha', 'S2L', 'S2H']
    LW = [2800, 3347, 3727, 3799, 3836, 3869, 4102, 4341, 4861, 4983, 6563, 6717, 6731]
    fLW = np.zeros(len(LW), dtype='int') # flag.

    R_grs = 50
    dw  = 5
    ldw = 5
    dlw = R_grs * 2
    lsig = 3 # significance of lines.
    er   = 1./np.sqrt(wycont)

    con = (xcont<20000)
    z = np.polyfit(xcont[con], ycont[con], 5, w=wycont[con])
    p = np.poly1d(z)
    ypoly = p(xcont)

    wht2   = wycont
    flag_l = 0

    for ii in range(len(xcont)):
        if ycont[ii] > er[ii]*lsig + ypoly[ii] and ycont[ii-1] > er[ii-1]*lsig + ypoly[ii-1] and ycont[ii+1] > er[ii+1]*lsig + ypoly[ii+1] and wht2[ii]:
            for jj in range(len(LW)):
                if xcont[ii]/(1.+zgal) > LW[jj] -  dw and xcont[ii]/(1.+zgal) < LW[jj] +  dw:
                    wht2[ii-dw:ii+dw] *= 0
                    flag_l  = 1
                    fLW[jj] = 1

    return wht2,flag_l


def check_line_cz(ycont,xcont,wycont,model,zgal):
    er = 1./np.sqrt(wycont)
    try:
        wht2, flag_l = detect_line(xcont, ycont, wycont, zgal)
        if flag_l == 1:
            wycont = wht2
            wht2, flag_l = detect_line(xcont, ycont, wycont,zgal)
    except Exception:
        wht2 = wycont
        pass

    z = np.polyfit(xcont, ycont, 5, w=wht2)
    p = np.poly1d(z)
    ypoly = p(xcont)

    return wht2, ypoly


def check_line_cz_man(ycont,xcont,wycont,model,zgal,LW=LW0,norder=5.):
    '''
    Parameters
    ----------
    LW : 
        List for emission lines to be masked.

    Returns
    -------
    wht : 
        Processed weight, where wavelength at line exists is masked.
    ypoly : 
        Fitted continuum flux.
    '''

    er  = 1./np.sqrt(wycont)
    try:
        wht2, flag_l = detect_line_man(xcont, ycont, wycont, zgal, LW, model)
    except Exception:
        print('Error in Line Check.')
        wht2 = wycont
        pass

    z     = np.polyfit(xcont, ycont, norder, w=wht2)
    p     = np.poly1d(z)
    ypoly = p(xcont)

    return wht2, ypoly

def detect_line_man(xcont, ycont, wycont, zgal, LW, model):
    ################
    # Line library
    ################
    #LN = ['Mg2', 'Ne5', 'O2', 'Htheta', 'Heta', 'Ne3', 'Hdelta', 'Hgamma', 'Hbeta', 'O3', 'Halpha', 'S2L', 'S2H']
    #LW = [2800, 3347, 3727, 3799, 3836, 3869, 4102, 4341, 4861, 4983, 6563, 6717, 6731]
    fLW = np.zeros(len(LW), dtype='int') # flag.

    #R_grs = 45
    #R_grs = 23.0
    R_grs = (xcont[1] - xcont[0])
    dw   = 1
    lsig = 1.5 # significance of lines.
    er   = 1./np.sqrt(wycont)

    con   = (xcont<20000)
    z     = np.polyfit(xcont[con], ycont[con], 5, w=wycont[con])
    p     = np.poly1d(z)
    ypoly = p(xcont)

    wht2   = wycont
    flag_l = 0

    for ii in range(len(xcont)):
        if 1 > 0:
            for jj in range(len(LW)):
                if xcont[ii]/(1.+zgal) > LW[jj] - dw*R_grs and xcont[ii]/(1.+zgal) < LW[jj] + dw*R_grs:
                    wht2[int(ii-dw):int(ii+dw)] *= 0
                    flag_l  = 1

    return wht2,flag_l


def check_line_man(data,xcont,wht,model,zgal,LW=LW0,lsig=1.5):
    '''
    Parameters
    ----------
    lsig : float
        which sigma to detect lines.
    '''
    fLW = np.zeros(len(LW), dtype='int') # flag.
    R_grs = (xcont[1] - xcont[0])
    dw    = 1.

    er = wht * 0
    con_wht = (wht>0)
    er[con_wht] = 1./np.sqrt(wht[con_wht])

    wht2 = np.zeros(len(wht),'float')
    wht2[:]= wht[:]
    flag_l = 0

    for ii in range(len(xcont)):
        for jj in range(len(LW)):
            if LW[jj]>0:
                if xcont[ii]/(1.+zgal) > LW[jj] - dw*R_grs and xcont[ii]/(1.+zgal) < LW[jj] + dw*R_grs:
                    wht2[int(ii-dw):int(ii+dw)] *= 0
                    flag_l  = 1

    return wht2
