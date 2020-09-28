import numpy as np
import sys
import scipy.integrate as integrate
from scipy.integrate import cumtrapz

from .function import *

class Post:
    '''
    #####################
    # Function for MCMC
    #####################
    '''
    def __init__(self, mainbody):
        self.mb = mainbody

    def residual(self, pars, fy, ey, wht, f_fir=False, out=False):
        '''
        Input:
        ======
        out   : model as second output. For lnprob func.
        f_fir : Bool. If dust component is on or off.

        Returns:
        ========
        residual of model and data.

        '''

        vals = pars.valuesdict()
        model, x1 = self.mb.fnc.tmp04(vals, self.mb.zgal, self.mb.lib)

        if self.mb.f_dust:
            model_dust, x1_dust = self.mb.fnc.tmp04_dust(vals, self.mb.zgal, self.mb.lib_dust)
            n_optir = len(model)

            # Add dust flux to opt/IR grid.
            model[:] += model_dust[:n_optir]

            # then append only FIR flux grid.
            model = np.append(model,model_dust[n_optir:])
            x1 = np.append(x1,x1_dust[n_optir:])

        if self.mb.ferr:
            try:
                f = vals['f']
            except:
                f = 0
        else:
            f = 0 # temporary... (if f is param, then take from vals dictionary.)

        #con_res = (model>0) & (wht>0) #& (ey>0)
        sig = np.sqrt(1./wht + (f**2*model**2))

        if fy is None:
            print('Data is none')
            resid = model #[con_res]
        else:
            resid = (model - fy) / sig

        if not out:
            return resid # i.e. residual/sigma. Because is_weighted = True.
        else:
            return resid, model # i.e. residual/sigma. Because is_weighted = True.


    def func_tmp(self, xint, eobs, fmodel):
        '''
        A function used for chi2 calculation for non-detection in lnprob.
        '''
        int_tmp = np.exp(-0.5 * ((xint-fmodel)/eobs)**2)
        return int_tmp


    def lnprob(self, pars, fy, ey, wht, f_fir, f_chind=True, SNlim=1.0):
        '''
        Returns:
        ========
        log posterior

        '''

        vals   = pars.valuesdict()
        if self.mb.ferr == 1:
            f = vals['f']
        else:
            f = 0

        resid, model = self.residual(pars, fy, ey, wht, f_fir, out=True)
        sig = np.sqrt(1./wht+f**2*model**2)
        chi_nd = 0.0

        if f_chind:
            con_up = (fy==0) & (fy/ey<=SNlim) & (ey>0)
            # This may be a bit cost of time;
            for nn in range(len(ey[con_up])):
                result = integrate.quad(lambda xint: self.func_tmp(xint, ey[con_up][nn]/SNlim, model[con_up][nn]), -ey[con_up][nn], ey[con_up][nn], limit=100)
                if result[0] > 0:
                    chi_nd += np.log(result[0])

            con_res = (model>=0) & (wht>0) & (fy>0) # Instead of model>0, model>=0 is for Lyman limit where flux=0.
            lnlike  = -0.5 * (np.sum(resid[con_res]**2 + np.log(2 * 3.14 * sig[con_res]**2)) - 2 * chi_nd)

        else:
            con_res = (model>=0) & (wht>0) # Instead of model>0, model>=0 is for Lyman limit where flux=0.
            lnlike  = -0.5 * (np.sum(resid[con_res]**2 + np.log(2 * 3.14 * sig[con_res]**2)))

        #print(np.log(2 * 3.14 * 1) * len(sig[con_res]), np.sum(np.log(2 * 3.14 * sig[con_res]**2)))
        #Av   = vals['Av']
        #if Av<0:
        #     return -np.inf
        #else:
        #    respr = 0 #np.log(1)

        respr = 0 # Flat prior...
        lnposterior = lnlike + respr

        return lnposterior
