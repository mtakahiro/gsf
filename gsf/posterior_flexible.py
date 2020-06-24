import numpy as np
import sys
from .function import *

class Post:
    '''
    #####################
    # Function for MCMC
    #####################
    '''
    def __init__(self, mainbody):
        self.mb = mainbody

    def residual(self, pars, fy, wht2, f_fir=False, out=False):
        '''
        #
        # Returns: residual of model and data.
        # out: model as second output. For lnprob func.
        # f_fir: Bool. If dust component is on or off.
        #
        '''

        vals = pars.valuesdict()
        model, x1 = self.mb.fnc.tmp04(vals, self.mb.zprev, self.mb.lib)
        if self.mb.f_dust:
            model_dust, x1_dust = self.mb.fnc.tmp04_dust(vals, self.mb.zprev, self.mb.lib_dust)
            n_optir = len(model)

            # Add dust flux to opt/IR grid.
            model[:]+= model_dust[:n_optir]

            # then append only FIR flux grid.
            model = np.append(model,model_dust[n_optir:])
            x1    = np.append(x1,x1_dust[n_optir:])

        if self.mb.ferr:
            f = vals['f']
        else:
            f = 0 # temporary... (if f is param, then take from vals dictionary.)
        con_res = (model>0) & (wht2>0) #& (ey>0)
        sig     = np.sqrt(1./wht2[con_res] + (f**2*model**2)[con_res])

        if not out:
            if fy is None:
                print('Data is none')
                return model[con_res]
            else:
                return (model - fy)[con_res] / sig # i.e. residual/sigma. Because is_weighted = True.
        if out:
            if fy is None:
                print('Data is none')
                return model[con_res], model
            else:
                return (model - fy)[con_res] / sig, model # i.e. residual/sigma. Because is_weighted = True.


    def lnprob(self, pars, fy, wht2, f_fir):
        '''
        #
        # Returns: log posterior
        #
        '''

        vals   = pars.valuesdict()
        if self.mb.ferr == 1:
            f = vals['f']
        else:
            f = 0

        resid, model = self.residual(pars, fy, wht2, f_fir, out=True)
        con_res = (model>0) & (wht2>0)
        sig     = np.sqrt(1./wht2+f**2*model**2)

        lnlike  = -0.5 * np.sum(resid**2 + np.log(2 * 3.14 * sig[con_res]**2))
        #print(np.log(2 * 3.14 * 1) * len(sig[con_res]), np.sum(np.log(2 * 3.14 * sig[con_res]**2)))
        #Av   = vals['Av']
        #if Av<0:
        #     return -np.inf
        #else:
        #    respr = 0 #np.log(1)
        respr = 0 # Flat prior...

        lnposterior = lnlike + respr
        return lnposterior
