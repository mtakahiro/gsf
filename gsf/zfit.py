import numpy as np
from lmfit import Model, Parameters, minimize, fit_report, Minimizer

from .function import check_line_cz_man

########################################
def check_redshift(fobs,eobs,xobs,fm_tmp,xm_tmp,zbest,dez,prior,NR,zliml, zlimu, delzz=0.01, nmc_cz=100, nwalk_cz=10):

    fit_par_cz = Parameters()
    fit_par_cz.add('z', value=zbest, min=zliml, max=zlimu)
    fit_par_cz.add('Cz0', value=1, min=0.5, max=1.5)
    fit_par_cz.add('Cz1', value=1, min=0.5, max=1.5)

    ##############################
    def residual_z(pars):
        vals  = pars.valuesdict()
        z     = vals['z']
        Cz0s  = vals['Cz0']
        Cz1s  = vals['Cz1']

        xm_s = xm_tmp * (1+z)
        fm_s = np.interp(xobs, xm_s, fm_tmp)

        con0 = (NR<1000)
        fy0  = fobs[con0] * Cz0s
        ey0  = eobs[con0] * Cz0s
        con1 = (NR>=1000) & (NR<10000)
        fy1  = fobs[con1] * Cz1s
        ey1  = eobs[con1] * Cz1s
        con2 = (NR>=10000) # BB
        fy2  = fobs[con2]
        ey2  = eobs[con2]

        fy01 = np.append(fy0,fy1)
        fcon = np.append(fy01,fy2)
        ey01 = np.append(ey0,ey1)
        eycon= np.append(ey01,ey2)

        wht = 1./np.square(eycon)

        wht2, ypoly = check_line_cz_man(fcon, xobs, wht, fm_s, z)

        if fobs is None:
            print('Data is none')
            return fm_s
        else:
            return (fm_s - fcon) * np.sqrt(wht2) # i.e. residual/sigma

    ###############################
    def lnprob_cz(pars):
        resid  = residual_z(pars) # i.e. (data - model) * wht
        z      = pars['z']
        s_z    = 1 #pars['f_cz']
        resid *= 1 / s_z
        resid *= resid
        nzz   = int(z/delzz)
        # Something unacceptable;
        if nzz<0 or z<zliml:
             return -np.inf
        else:
            respr = np.log(prior[nzz])
        resid += np.log(2 * np.pi * s_z**2) + respr
        return -0.5 * np.sum(resid)

    #################################

    out_cz  = minimize(residual_z, fit_par_cz, method='nelder')

    #
    # Best fit
    #
    keys = fit_report(out_cz).split('\n')
    for key in keys:
        if key[4:7] == 'chi':
            skey = key.split(' ')
            csq  = float(skey[14])
        if key[4:7] == 'red':
            skey = key.split(' ')
            rcsq = float(skey[7])

    fitc_cz = [csq, rcsq] # Chi2, Reduced-chi2

    zrecom  = out_cz.params['z'].value
    Czrec0  = out_cz.params['Cz0'].value
    Czrec1  = out_cz.params['Cz1'].value

    mini_cz = Minimizer(lnprob_cz, out_cz.params)
    res_cz  = mini_cz.emcee(burn=int(nmc_cz/2), steps=nmc_cz, thin=10, nwalkers=nwalk_cz, params=out_cz.params, is_weighted=True)

    return res_cz, fitc_cz
