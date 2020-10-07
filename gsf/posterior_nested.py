import numpy as np
import sys
import scipy.integrate as integrate
from scipy.integrate import cumtrapz

from .function import *

class Post_nested:
    '''
    #####################
    # Function for MCMC
    #####################
    '''
    def __init__(self, mainbody, params=None):
        self.mb = mainbody
        self.params = params

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
            '''try:
                print(vals['MDUST'])
                print(model_dust[:n_optir],x1_dust[:n_optir])
                print(model_dust[n_optir:],x1_dust[n_optir:])
            except:
                pass
            '''
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


    def get_dict(self, pars):

        ii = 0
        for key in self.params:
            if self.params[key].vary:
                self.params[key].value = pars[ii]
                ii += 1
        return self.params

    def residual_nest(self, pars, fy, ey, wht, f_fir=False, out=False):
        '''
        Input:
        ======
        out   : model as second output. For lnprob func.
        f_fir : Bool. If dust component is on or off.

        Returns:
        ========
        residual of model and data.

        '''

        vals = self.get_dict(pars)

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


    def prior_transform(self, pars):
        """
        A function defining the tranform between the parameterisation in the unit hypercube
        to the true parameters.

        Args:
            theta (tuple): a tuple containing the parameters.
            
        Returns:
            tuple: a new tuple or array with the transformed parameters.
        """

        ii = 0
        for key in self.params:
            if self.params[key].vary:
                cmin = self.params[key].min
                cmax = self.params[key].max
                cprim = pars[ii]
                pars[ii] = cprim*(cmax-cmin) + cmin
                ii += 1
        '''
        mmu = 0.     # mean of Gaussian prior on m
        msigma = 10. # standard deviation of Gaussian prior on m
        m = mmu + msigma*ndtri(mprime) # convert back to m
        '''

        return pars


    def lnlike(self, pars, fy, ey, wht, f_fir, f_chind=True, SNlim=1.0):
        '''
        Returns:
        ========
        log likelihood

        Note:
        =====
        This is a copy from lnprob, with respr=0.

        '''
        vals = pars#.valuesdict()
        if self.mb.ferr == 1:
            f = vals['f']
        else:
            f = 0

        resid, model = self.residual_nest(pars, fy, ey, wht, f_fir, out=True)
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

        #lnlike += 700000
        #print(norm - 0.5*chisq)
        #print(lnlike)
        #print(vals,lnlike)
        return lnlike


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

        respr = 0 # Flat prior...
        lnposterior = lnlike + respr

        return lnposterior
