import numpy as np
import sys
import scipy.integrate as integrate
from scipy.integrate import cumtrapz
from scipy import stats
from numpy import exp as np_exp
from numpy import log as np_log
from scipy.special import erf
from scipy.stats import lognorm

class Post():
    '''
    Class function for MCMC
    '''
    def __init__(self, mainbody):
        self.mb = mainbody
        self.scale = 1
        self.gauss_Mdyn = None
        self.Na = len(self.mb.age)
        self.mb.prior = None


    def residual(self, pars, fy:float, ey:float, wht:float, f_fir:bool, out:bool=False, 
        f_val:bool=False, f_penlize:bool=True, f_only_spec:bool=False, verbose=False):
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

        model, x1 = self.mb.fnc.get_template(vals)

        if self.mb.f_dust:
            model_dust, x1_dust = self.mb.fnc.tmp04_dust(vals)
            n_optir = self.mb.n_optir #len(model)

            # Add dust flux to opt/IR grid.
            model[:] += model_dust[:n_optir]
            # then append only FIR flux grid.
            model = np.append(model,model_dust[n_optir:])
            x1 = np.append(x1,x1_dust[n_optir:])

        if self.mb.fneb:
            n_optir = self.mb.n_optir
            model_neb, x1_neb = self.mb.fnc.get_template(vals, f_neb=True)
            model[:n_optir] += model_neb

        if self.mb.ferr:
            try:
                logf = vals['logf']
            except:
                logf = -np.inf
        else:
            logf = -np.inf # temporary... (if f is param, then take from vals dictionary.)

        if f_only_spec:
            contmp = (self.mb.dict['NR']<10000)
            fy = fy[contmp]
            ey = ey[contmp]
            wht = wht[contmp]
            model = model[contmp]

        sig = wht[:] * 0
        con_res = (wht>0)
        con_res_r = (wht==0)
        sig[con_res] = ey[con_res]**2 + model[con_res]**2 * np_exp(logf)**2
        sig[con_res_r] = wht[con_res_r] * 0 + np.inf

        if fy is None:
            if verbose:
                print('Data is none')
            resid = model
        else:
            resid = (model - fy) / np.sqrt(sig)

        if self.mb.ferr and f_penlize:
            # Penalize redisual;
            tmp = (model - fy)**2 / sig + np_log(2*3.14*sig**2)
            con_res = (tmp>0) & (~np.isinf(tmp))
            resid[con_res] = np.sqrt(tmp[con_res])

        if not out:
            return resid # i.e. residual/sigma. Because is_weighted = True.
        else:
            return resid, model # i.e. residual/sigma. Because is_weighted = True.


    def func_tmp(self, xint, eobs, fmodel):
        '''
        A function used for chi2 calculation for non-detection in lnprob.
        '''
        int_tmp = np_exp(-0.5 * ((xint-fmodel)/eobs)**2)
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


    def lnprob_emcee(self, pos, pars, fy:float, ey:float, wht:float, f_fir:bool, f_chind:bool=True, SNlim:float=1.0, f_scale:bool=False, 
        lnpreject=-np.inf, f_like:bool=False, flat_prior:bool=False, gauss_prior:bool=True, f_val:bool=True, nsigma:float=1.0, out=None,
        f_prior_sfh=False, alpha_sfh_prior=100, norder_sfh_prior=3, verbose=False):
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
        sig_con = np.sqrt(ey[con_res]**2 + model[con_res]**2 * np_exp(2 * logf))
        chi_nd = 0.0

        con_up = (ey>0) & (fy/ey<=SNlim)
        if f_chind and len(fy[con_up])>0:
            x_erf = (ey[con_up]/SNlim - model[con_up]) / (np.sqrt(2) * ey[con_up]/SNlim)
            f_erf = erf(x_erf)
            if np.min(f_erf) <= -1:
                #return lnpreject
                lnlike = -0.5 * (np.sum(resid[con_res]**2 + np_log(2 * 3.14 * sig_con**2)))
            else:
                chi_nd = np.sum( np_log(np.sqrt(np.pi / 2) * ey[con_up]/SNlim * (1 + f_erf)) )
            lnlike = -0.5 * (np.sum(resid[con_res]**2 + np_log(2 * 3.14 * sig_con**2)) - 2 * chi_nd)
        else:
            lnlike = -0.5 * (np.sum(resid[con_res]**2 + np_log(2 * 3.14 * sig_con**2)))

        # Scale likeligood; Do not make this happen yet.
        if f_scale:
            if self.scale == 1:
                self.scale = np.abs(lnlike) * 0.001
                if verbose:
                    print('scale is set to',self.scale)
            lnlike += self.scale

        if np.isinf(np.abs(lnlike)):
            if verbose:
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
                respr += np_log(p_gauss)

        # Prior for redshift:
        if self.mb.fzmc == 1:
            zprior = self.mb.z_prior
            prior = self.mb.p_prior

            nzz = np.argmin(np.abs(zprior-vals['zmc']))
            # For something unacceptable;
            if nzz<0 or prior[nzz]<=0:
                # print('z Posterior unacceptable.')
                return lnpreject
            else:
                respr += np_log(prior[nzz])

        # Prior for SFH;
        if f_prior_sfh:
            respr += self.get_sfh_prior(vals, norder=norder_sfh_prior, alpha=alpha_sfh_prior)

        # lognormal-prior for any params;
        for ii,key_param in enumerate(self.mb.key_params_prior):
            # key_param = 'Av'
            sigma = self.mb.key_params_prior_sigma[ii]
            respr += self.get_lognormal_prior(vals, key_param, sigma=sigma, mu=0)

        lnposterior = lnlike + respr
        if not np.isfinite(lnposterior):
            return lnpreject

        return lnposterior


    def get_sfh_prior(self, vals, norder=3, alpha=100.0):
        '''
        Fit SFH with an n-order polynomial and reflect the residual to the prior.
        Fit should be done in log-log space.
        '''
        t = np.log10(self.mb.age)
        y = []
        vary = []
        for aa in range(self.Na):
            y.append(vals['A%d'%aa])
            vary.append(vals['A%d'%aa].vary)
        y = np.asarray(y)

        con = (vary)
        z = np.polyfit(t[con], y[con], norder)
        yfit = np.polyval(z,t[con])
        yres = (y[con] - yfit)

        if False:
            import matplotlib.pyplot as plt
            plt.plot(t,y)
            plt.plot(t[con],yfit)
            plt.show()
            hoge

        lnprior = -0.5 * (np.sum(yres**2) * alpha)

        return lnprior
    

    def get_lognormal_prior(self, vals, key_param, mu=0, sigma=100.0):
        '''
        '''
        y = vals[key_param]

        if self.mb.prior == None:
            self.mb.prior = {}
            for key_param in self.mb.fit_params:
                self.mb.prior[key_param] = None

        if self.mb.prior[key_param] == None:
            self.mb.logger.info('Using lognormal prior for %s'%key_param)
            self.mb.prior[key_param] = lognorm(sigma)

        lnprior = np.log(self.mb.prior[key_param].pdf(y))

        return lnprior