import numpy as np
import sys
import scipy.integrate as integrate
from scipy.integrate import cumtrapz
from scipy import special,stats

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
        self.gauss_Mdyn = None
        self.Na = len(self.mb.age)

    def residual(self, pars, fy, ey, wht, f_fir=False, out=False, f_val=False):
        '''
        Input:
        ======
        out   : model as second output. For lnprob func.
        f_fir : Bool. If dust component is on or off.

        Returns:
        ========
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
        if False:
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

    def lnprob(self, pars, fy, ey, wht, f_fir, f_chind=True, SNlim=1.0, f_scale=False, 
    lnpreject=-np.inf, f_like=False, flat_prior=False, gauss_prior=True, f_val=False, nsigma=1.0):
        '''
        Input:
        ======
        f_chind (bool) : If true, includes non-detection in likelihood calculation.
        lnpreject : A replaced value when lnprob gets -inf value.
        flat_prior : Assumes flat prior for Mdyn. Used only when MB.f_Mdyn==True.
        gauss_prior : Assumes gaussian prior for Mdyn. Used only when MB.f_Mdyn==True.

        Returns:
        ========
        If f_like, log Likelihood. Else, log Posterior prob.
        '''
        if f_val:
            vals = pars
        else:
            vals = pars.valuesdict()
        if self.mb.ferr == 1:
            f = vals['f']
        else:
            f = 0

        if False:
            # Checking multiple peak model
            if self.mb.SFH_FORM != -99 and self.mb.npeak>1:
                for aa in range(0,self.mb.npeak-1,1):
                    if vals['A'+str(aa)] > vals['A'+str(aa+1)]:
                        return lnpreject

        resid, model = self.residual(pars, fy, ey, wht, f_fir, out=True, f_val=f_val)
        con_res = (model>=0) & (wht>0) & (fy>0) & (ey>0) # Instead of model>0; model>=0 is for Lyman limit where flux=0. This already exclude upper limit.
        sig_con = np.sqrt(1./wht[con_res]+f**2*model[con_res]**2) # To avoid error message.
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

        # Scale likeligood; Do not make this happen yet.
        if f_scale:
            if self.scale == 1:
                self.scale = np.abs(lnlike) * 0.001
                print('scale is set to',self.scale)
            lnlike /= self.scale

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
                return lnpreject
            else:
                respr += np.log(prior[nzz])

        lnposterior = lnlike + respr
        
        if not np.isfinite(lnposterior):
            return -np.inf
        return lnposterior


    """
    def get_mass(self,pars):
        '''
        Purpose:
        ========
        Quickly calculate stellar mass for a given param set.

        Return:
        =======
        Stellar mass in logMsun.
        '''
        sedpar = self.af['ML'] # For M/L

        Mtot = 0
        for aa in range(len(self.mb.age)):
            # Checking AA limit too;
            if pars['A%d'%aa] < self.mb.Amin or pars['A%d'%aa] > self.mb.Amax:
                return np.inf

            if self.mb.Zevol == 1:
                # Checking Z limit too;
                if pars['Z%d'%aa] < self.mb.Zmin or pars['Z%d'%aa] > self.mb.Zmax:
                    return np.inf
                ZZtmp = pars['Z%d'%aa]
                nZtmp = self.mb.bfnc.Z2NZ(ZZtmp)
            else:
                # Checking Z limit too;
                if pars['Z%d'%0] < self.mb.Zmin or pars['Z%d'%0] > self.mb.Zmax:
                    return np.inf
                #ZZtmp = pars['Z%d'%0]
                nZtmp = aa #self.mb.bfnc.Z2NZ(ZZtmp)

            mslist = sedpar['ML_'+str(nZtmp)][aa]
            Mtot += 10**(pars['A%d'%aa] + np.log10(mslist))
        
        return np.log10(Mtot)
    """