import numpy as np
import sys
from lmfit import Model, Parameters, minimize, fit_report, Minimizer
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from .function import check_line_cz_man


def lnprob_cz(pars, zprior, prior, zliml, zlimu, args, kwargs):
    '''
    '''
    resid = residual_z(pars, *args, **kwargs) # i.e. (data - model) * wht
    z = pars['z']
    s_z = 1 #pars['f_cz']
    resid *= 1/s_z
    resid *= resid

    if False:
        nzz = np.argmin(np.abs(zprior-z))

        # For something unacceptable;
        if nzz<0 or zprior[nzz]<zliml or zprior[nzz]>zlimu or prior[nzz]<=0:
            return -np.inf
        else:
            respr = np.log(prior[nzz])
            resid += np.log(2 * np.pi * s_z**2)
            return -0.5 * np.sum(resid) + respr
    else:
        resid += np.log(2 * np.pi * s_z**2)
        return -0.5 * np.sum(resid)


def residual_z(pars, xm_tmp, fm_tmp, xobs, fobs, eobs, NR, data_len, NRbb_lim=10000, include_photometry=True, f_line_check=False):
    '''
    '''
    vals = pars.valuesdict()
    z = vals['z']
    Cz0s = vals['Cz0']
    Cz1s = vals['Cz1']
    Cz2s = vals['Cz2']

    xm_s = xm_tmp * (1+z)
    fint = interpolate.interp1d(xm_s, fm_tmp, kind='nearest', fill_value="extrapolate")
    # @@@ Maybe for future consideration.
    # fint = interpolate.interp1d(xm_s, fm_tmp, kind='linear', fill_value="extrapolate")
    fm_s = fint(xobs)

    Cs = [Cz0s, Cz1s, Cz2s]
    fy02 = []
    ey02 = []
    for ii in range(len(data_len)):
        if ii == 0:
            con0 = (NR<data_len[ii])
        else:
            con0 = (NR>=np.sum(data_len[:ii])) & (NR<np.sum(data_len[:ii+1]))
        fy02 = np.append(fy02, fobs[con0] * Cs[ii])
        ey02 = np.append(ey02, eobs[con0] * Cs[ii])

    con_bb = (NR>=NRbb_lim) # BB
    fy_bb = fobs[con_bb]
    ey_bb = eobs[con_bb]
    if include_photometry and len(fy_bb)>0:
        fcon = np.append(fy02,fy_bb)
        eycon = np.append(ey02,ey_bb)
        wht = 1./np.square(eycon)
    else:
        fcon = fy02
        eycon = ey02
        wht = 1./np.square(eycon)

    if f_line_check:
        try:
            wht2, ypoly = check_line_cz_man(fcon, xobs, wht, fm_s, z)
        except:
            wht2 = wht
    else:
        wht2 = wht
        
    if fobs is None:
        print('Data is none')
        return fm_s
    else:
        return (fm_s - fcon) * np.sqrt(wht2) # i.e. residual/sigma


def check_redshift(fobs, eobs, xobs, fm_tmp, xm_tmp, zbest, zprior, prior, NR, data_len, zliml, zlimu, \
    nmc_cz=100, nwalk_cz=10, nthin=5, f_line_check=False, f_vary=True, NRbb_lim=10000, include_photometry=True):
    '''
    Fit observed flux with a template to get redshift probability.

    Parameters
    ----------
    zbest : 
        Initial value for redshift.
    zprior : 
        Redshift grid for prior.
    prior : 
        Prior for redshift determination. E.g., Eazy z-probability.
    zliml : 
        Lowest redshift for fitting range.
    zlimu : 
        Highest redshift for fitting range.
    f_vary : bool
        If want to fix redshift.
    fm_tmp :
        Template spectrum at RF.
    xm_tmp :
        Template spectrum at RF.
    fobs : 
        Observed spectrum. (Already scaled with Cz0prev.)
    eobs : 
        Observed spectrum. (Already scaled with Cz0prev.)
    xobs : 
        Observed spectrum. (Already scaled with Cz0prev.)


    Returns
    -------
    res_cz :

    fitc_cz :

    '''
    if zliml == None or zlimu == None:
        print('z range is not set for the z-fit function. Exiting.')
        print('Specify `ZMCMIN` and `ZMCMIN` in your input file.')
        sys.exit()

    fit_par_cz = Parameters()
    fit_par_cz.add('z', value=zbest, min=zliml, max=zlimu, vary=f_vary)
    fit_par_cz.add('Cz0', value=1, min=0.5, max=1.5)
    fit_par_cz.add('Cz1', value=1, min=0.5, max=1.5)
    fit_par_cz.add('Cz2', value=1, min=0.5, max=1.5)

    # Get Best fit
    args_res = (xm_tmp, fm_tmp, xobs, fobs, eobs, NR, data_len)
    kwargs_res = {'include_photometry':include_photometry, 'f_line_check':f_line_check, 'NRbb_lim':NRbb_lim}

    out_cz  = minimize(residual_z, fit_par_cz, args=args_res, method='nelder')
    keys = fit_report(out_cz).split('\n')
    for key in keys:
        if key[4:7] == 'chi':
            skey = key.split(' ')
            csq = float(skey[14])
        if key[4:7] == 'red':
            skey = key.split(' ')
            rcsq = float(skey[7])

    fitc_cz = [csq, rcsq] # Chi2, Reduced-chi2

    mini_cz = Minimizer(lnprob_cz, out_cz.params, (zprior, prior, zliml, zlimu, args_res, kwargs_res))
    res_cz = mini_cz.emcee(burn=int(nmc_cz/2), steps=nmc_cz, thin=nthin, nwalkers=nwalk_cz, params=out_cz.params, is_weighted=True)

    return res_cz, fitc_cz


def get_chi2(zz_prob, fy_cz, ey_cz, x_cz, fm_tmp, xm_tmp, file_zprob, rms_lim=1e4):
    '''
    Parameters
    ----------
    zz_prob : float array
        redshift array for fit.
    fy_cz, ey_cz, x_cz : 
        observed values.
    fm_tmp, xm_tmp :
        template.
    file_zprob : str
        output file
    '''
    mask = (ey_cz<rms_lim)
    prob_cz = np.zeros(len(zz_prob), float)
 
    fw = open(file_zprob, 'w')
    fw.write('# z p(z)\n')
 
    for ii in range(len(zz_prob)):
        z = zz_prob[ii]
        xm_s = xm_tmp * (1+z)
        fint = interpolate.interp1d(xm_s, fm_tmp, kind='nearest', fill_value="extrapolate")
        fm_s = fint(x_cz)

        wht = 1./np.square(ey_cz)
        lnprob_cz = -0.5 * np.nansum( np.square((fm_s - fy_cz)[mask] * np.sqrt(wht[mask])) ) # i.e. (residual/sigma)^2
        prob_cz[ii] = np.exp(lnprob_cz)
        fw.write('%.3f %.3e\n'%(z,prob_cz[ii]))
        
    fw.close()
