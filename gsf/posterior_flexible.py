import numpy as np
import sys
import scipy.integrate as integrate
from scipy.integrate import cumtrapz
from scipy import special,stats

from .function import *

class Post:
    '''
    Class function for MCMC
    '''
    def __init__(self, mainbody):
        self.mb = mainbody
        self.scale = 1
        self.gauss_Mdyn = None
        self.Na = len(self.mb.age)


    def residual(self, pars, fy, ey, wht, f_fir, out=False, f_val=False, f_penlize=True):
        '''
        Parameters
        ----------
        out : 
            model as second output. For lnprob func.
        f_fir : bool
            If dust component is on or off.

        Returns
        -------
        residual of model and data.
        '''
        if f_val:
            vals = pars
        else:
            vals = pars.valuesdict()

        model, x1 = self.mb.fnc.tmp04(vals)

        if self.mb.f_dust:
            model_dust, x1_dust = self.mb.fnc.tmp04_dust(vals)
            n_optir = len(model)

            # Add dust flux to opt/IR grid.
            model[:] += model_dust[:n_optir]
            # then append only FIR flux grid.
            model = np.append(model,model_dust[n_optir:])
            x1 = np.append(x1,x1_dust[n_optir:])

        if self.mb.ferr:
            try:
                logf = vals['logf'] #.
            except:
                logf = -np.inf
        else:
            logf = -np.inf # temporary... (if f is param, then take from vals dictionary.)

        sig = wht[:] * 0
        con_res = (wht>0)
        con_res_r = (wht==0)
        sig[con_res] = ey[con_res]**2 + model[con_res]**2 * np.exp(logf)**2
        sig[con_res_r] = wht[con_res_r] * 0 + np.inf

        if fy is None:
            print('Data is none')
            resid = model #[con_res]
        else:
            resid = (model - fy) / np.sqrt(sig)

        if self.mb.ferr and f_penlize:
            # Penalize redisual;
            tmp = (model - fy)**2 / sig + np.log(2*3.14*sig**2)
            con_res = (tmp>0) & (~np.isinf(tmp))
            resid[con_res] = np.sqrt(tmp[con_res])

        if not out:
            return resid # i.e. residual/sigma. Because is_weighted = True.
        else:
            return resid, model # i.e. residual/sigma. Because is_weighted = True.

    """
    def residual(self, pars, fy, ey, wht, f_fir=False, out=False, f_val=False, SNlim=1.0, f_chind=True):
        '''
        '''
        if f_val:
            vals = pars
        else:
            vals = pars.valuesdict()

        model, x1 = self.mb.fnc.tmp04(vals)

        from scipy import special
        if f_chind:
            conw = (wht>0) & (ey>0) & (fy/ey>SNlim)
        else:
            conw = (wht>0) & (ey>0)

        if self.mb.ferr:
            try:
                f = vals['f']
            except:
                f = 0
        else:
            f = 0 # temporary... (if f is param, then take from vals dictionary.)

        if self.mb.f_dust:
            model_dust, x1_dust = self.mb.fnc.tmp04_dust(vals)
            n_optir = len(model)

            # Add dust flux to opt/IR grid.
            model[:] += model_dust[:n_optir]
            # then append only FIR flux grid.
            model = np.append(model,model_dust[n_optir:])
            x1 = np.append(x1,x1_dust[n_optir:])

        # Add sigma?
        sig = wht[:] * 0
        con_res = (wht>0)
        con_res_r = (wht==0)
        sig[con_res] = np.sqrt(1./wht[con_res] + (f**2*model[con_res]**2))
        sig[con_res_r] = wht[con_res_r] * 0 + np.inf

        if fy is None:
            print('Data is none')
            resid = model #[con_res]
        else:
            resid = (model - fy) / sig

        if not out:
            return resid # i.e. residual/sigma. Because is_weighted = True.
        else:
            return resid, model # i.e. residual/sigma. Because is_weighted = True.

        return resid
    """

    def func_tmp(self, xint, eobs, fmodel):
        '''
        A function used for chi2 calculation for non-detection in lnprob.
        '''
        int_tmp = np.exp(-0.5 * ((xint-fmodel)/eobs)**2)
        return int_tmp


    def swap_pars(self, pars):
        '''
        '''
        Amax = -99
        aamax = 0
        for aa in range(self.Na):
            if pars['A%d'%aa]>Amax:
                Amax = pars['A%d'%aa]
                aamax = aa
        if aamax>0:
            Amax2 = pars['A%d'%(aamax-1)]
            pars['A%d'%(aamax-1)] = Amax
            pars['A%d'%aamax] = Amax2
        return pars

    def swap_pars_inv(self, pars):
        '''
        '''
        Amax = -99
        aamax = self.Na-1
        for aa in range(self.Na):
            if pars['A%d'%aa]>Amax:
                Amax = pars['A%d'%aa]
                aamax = aa
        if aamax<self.Na-1:
            Amax2 = pars['A%d'%(aamax+1)]
            pars['A%d'%(aamax+1)] = Amax
            pars['A%d'%aamax] = Amax2
        return pars


    def lnprob_emcee(self, pos, pars, fy, ey, wht, f_fir, f_chind=True, SNlim=1.0, f_scale=False, 
    lnpreject=-np.inf, f_like=False, flat_prior=False, gauss_prior=True, f_val=True, nsigma=1.0, out=None):
        '''
        Parameters
        ----------
        f_chind : bool
            If true, includes non-detection in likelihood calculation.
        lnpreject : 
            A replaced value when lnprob gets -inf value.
        flat_prior : 
            Assumes flat prior for Mdyn. Used only when MB.f_Mdyn==True.
        gauss_prior : 
            Assumes gaussian prior for Mdyn. Used only when MB.f_Mdyn==True.
        pos : 
            This is a critical parameter, to make use of EMCEE, while keeping pars a dictionary obtained by lmfit.
            This is pos of sampler.run_mcmc(pos, self.nmc, progress=True).
        out : 
            Just for keywords.

        Returns:
        --------
        If f_like, log Likelihood. Else, log Posterior prob.
        '''
        if f_val:
            vals = pars
        else:
            vals = pars.valuesdict()
        if self.mb.ferr == 1:
            logf = vals['logf']
        else:
            logf = -np.inf
                
        if False:
            # Checking multiple peak model
            if self.mb.SFH_FORM != -99 and self.mb.npeak>1:
                for aa in range(0,self.mb.npeak-1,1):
                    if vals['A'+str(aa)] > vals['A'+str(aa+1)]:
                        return lnpreject

        # Check range:
        if not out==None:
            ii = 0
            for key in out.params:
                if out.params[key].vary:
                    cmin = out.params[key].min
                    cmax = out.params[key].max
                    if pos[ii]<cmin or pos[ii]>cmax or np.isnan(pos[ii]):
                        return lnpreject
                    vals[key].value = pos[ii]
                    ii += 1

        resid, model = self.residual(vals, fy, ey, wht, f_fir, out=True, f_val=f_val, f_penlize=False)

        con_res = (model>=0) & (wht>0) & (fy>0) & (ey>0)
        sig_con = np.sqrt(ey[con_res]**2 + model[con_res]**2 * np.exp(2 * logf))
        chi_nd = 0.0

        con_up = (ey>0) & (fy/ey<=SNlim)
        if f_chind and len(fy[con_up])>0:
            x_erf = (ey[con_up]/SNlim - model[con_up]) / (np.sqrt(2) * ey[con_up]/SNlim)
            f_erf = special.erf(x_erf)
            if np.min(f_erf) <= -1:
                return lnpreject
            else:
                chi_nd = np.sum( np.log(np.sqrt(np.pi / 2) * ey[con_up]/SNlim * (1 + f_erf)) )
            lnlike = -0.5 * (np.sum(resid[con_res]**2 + np.log(2 * 3.14 * sig_con**2)) - 2 * chi_nd)
        else:
            lnlike = -0.5 * (np.sum(resid[con_res]**2 + np.log(2 * 3.14 * sig_con**2)))

        #chi2,fin_chi2 = get_chi2(fy, ey, wht, model, self.mb.ndim, SNlim=SNlim, f_chind=f_chind, f_exclude=False, xbb=None, x_ex=None)
        #lnlike = -0.5 * (fin_chi2)
        #print(lnlike,'hoge2')

        # Scale likeligood; Do not make this happen yet.
        if f_scale:
            if self.scale == 1:
                self.scale = np.abs(lnlike) * 0.001
                print('scale is set to',self.scale)
            lnlike += self.scale

        if np.isinf(np.abs(lnlike)):
            print('Error in lnlike')
            return lnpreject

        # If no prior, return log likeligood.
        if f_like:
            return lnlike

        # Prior
        respr = 0

        if self.mb.f_Mdyn:
            # Prior from dynamical mass:
            if gauss_prior and self.gauss_Mdyn == None:
                self.gauss_Mdyn = stats.norm(self.mb.logMdyn, self.mb.elogMdyn)
            #logMtmp = self.get_mass(vals)
            logMtmp = self.mb.logMtmp
            #print(logMtmp)
            if flat_prior:
                if logMtmp > self.mb.logMdyn + self.mb.elogMdyn * nsigma:
                    #pars = self.swap_pars(pars)
                    #print(logMtmp, self.mb.logMdyn + self.mb.elogMdyn)
                    return lnpreject
                else:
                    respr += 0
            elif gauss_prior:
                if logMtmp > self.mb.logMdyn + self.mb.elogMdyn * nsigma:
                    #pars = self.swap_pars(pars)
                    #return lnpreject
                    pass
                #elif logMtmp < self.mb.logMdyn - self.mb.elogMdyn * nsigma:
                #    pars = self.swap_pars_inv(pars)
                #    return lnpreject
                p_gauss = self.gauss_Mdyn.pdf(logMtmp) #/ self.gauss_cnst
                respr += np.log(p_gauss)

        # Prior for redshift:
        if self.mb.fzmc == 1:
            zprior = self.mb.z_prior
            prior = self.mb.p_prior

            nzz = np.argmin(np.abs(zprior-vals['zmc']))
            # For something unacceptable;
            if nzz<0 or prior[nzz]<=0:
                print('z Posterior unacceptable.')
                return lnpreject
            else:
                respr += np.log(prior[nzz])

        lnposterior = lnlike + respr
        if not np.isfinite(lnposterior):
            print('Posterior unacceptable.')
            return lnpreject

        return lnposterior

