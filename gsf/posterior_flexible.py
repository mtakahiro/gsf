import numpy as np
import sys
import scipy.integrate as integrate
from scipy.integrate import cumtrapz
from scipy import special

from .function import *

class Post:
    '''
    #####################
    # Function for MCMC
    #####################
    '''
    def __init__(self, mainbody):
        self.mb = mainbody
        self.scale = 1

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

        sig = wht[:] * 0
        con_res = (wht>0)
        con_res_r = (wht==0)
        sig[con_res] = np.sqrt(1./wht[con_res] + (f**2*model[con_res]**2))
        sig[con_res_r] = wht[con_res_r] * 0 + np.inf
        #sig = np.sqrt(1./wht + (f**2*model**2))

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


    def lnprob(self, pars, fy, ey, wht, f_fir, f_chind=True, SNlim=1.0, f_scale=False, lnpreject=-1e10):
        '''
        Input:
        ======
        f_chind (bool) : If true, includes non-detection in likelihood calculation.
        lnpreject : A replaced value when lnprob gets -inf value.

        Returns:
        ========
        log posterior

        '''
        vals = pars.valuesdict()
        if self.mb.ferr == 1:
            f = vals['f']
        else:
            f = 0

        resid, model = self.residual(pars, fy, ey, wht, f_fir, out=True)
        con_res = (model>=0) & (wht>0) & (fy>0) & (ey>0) # Instead of model>0; model>=0 is for Lyman limit where flux=0. This already exclude upper limit.
        sig_con = np.sqrt(1./wht[con_res]+f**2*model[con_res]**2) # To avoid error message.
        chi_nd = 0.0

        con_up = (fy==0) & (fy/ey<=SNlim) & (ey>0)
        if f_chind and len(fy[con_up])>0:
            x_erf = (ey[con_up]/SNlim - model[con_up]) / (np.sqrt(2) * ey[con_up]/SNlim)
            f_erf = special.erf(x_erf)
            if np.min(f_erf) <= -1:
                return lnpreject
            else:
                chi_nd = np.sum( np.log(np.sqrt(np.pi / 2) * ey[con_up]/SNlim * (1 + f_erf)) )
            #con_res = (model>=0) & (wht>0) & (fy>0) # Instead of model>0, model>=0 is for Lyman limit where flux=0.
            lnlike  = -0.5 * (np.sum(resid[con_res]**2 + np.log(2 * 3.14 * sig_con**2)) - 2 * chi_nd)
        else:
            #con_res = (model>=0) & (wht>0) # Instead of model>0, model>=0 is for Lyman limit where flux=0.
            lnlike  = -0.5 * (np.sum(resid[con_res]**2 + np.log(2 * 3.14 * sig_con**2)))

        # Scale likeligood; Do not make this happen yet.
        if f_scale and self.scale == 1:
            self.scale = np.abs(lnlike) * 0.001
            print('scale is set to',self.scale)

        lnlike /= self.scale

        #print(np.log(2 * 3.14 * 1) * len(sig[con_res]), np.sum(np.log(2 * 3.14 * sig[con_res]**2)))
        #Av   = vals['Av']
        #if Av<0:
        #     return -np.inf
        #else:
        #    respr = 0 #np.log(1)

        # Prior
        respr = 0

        # Prior for redshift:
        if self.mb.fzmc == 1:
            zprior = self.mb.z_prior
            prior = self.mb.p_prior

            nzz = np.argmin(np.abs(zprior-vals['zmc']))
            # For something unacceptable;
            if nzz<0 or prior[nzz]<=0:
                return lnpreject
            else:
                respr += np.log(prior[nzz])
            
        lnposterior = lnlike + respr
        return lnposterior
