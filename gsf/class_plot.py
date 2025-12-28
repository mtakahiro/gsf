import numpy as np
# import sys
import asdf
import matplotlib.pyplot as plt
# from numpy import log10
# from scipy.integrate import simps
import os
# import time
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker
# import scipy.interpolate as interpolate
from astropy.io import fits
from astropy import units as u

# Custom modules
from .function import *
# from .function_class import Func
# from .basic_func import Basic
from .function_igm import *

class PLOT(object):
    '''
    '''
    def __init__(self, mb, f_silence=True):
        ''''''
        self.mb = mb

        if f_silence:
            import matplotlib
            matplotlib.use("Agg")
        else:
            import matplotlib

        print('\n### Running plot_sfh_tau ###\n')

        try:
            if not self.mb.ZFIX == None:
                skip_zhist = True
        except:
            pass

        NUM_COLORS = len(self.mb.age)
        cm = plt.get_cmap('gist_rainbow_r')
        self.col = np.atleast_2d([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

        return 


    def get_sfh_figure_format(self, Tzz, zredl, lsfrl, lsfru, y2min, y2max, Txmin, Txmax):
        """"""
        self.axes['ax1'].set_xlabel('$t_\mathrm{lookback}$/Gyr', fontsize=12)
        self.axes['ax2'].set_xlabel('$t_\mathrm{lookback}$/Gyr', fontsize=12)

        # This has to come before set_xticks;
        self.axes['ax1t'].set_xscale('log')
        self.axes['ax2t'].set_xscale('log')

        self.axes['ax1t'].xaxis.set_major_locator(ticker.FixedLocator(Tzz[:]))
        self.axes['ax1t'].xaxis.set_major_formatter(ticker.FixedFormatter(zredl[:]))
        self.axes['ax1t'].tick_params(axis='x', labelcolor='k')
        self.axes['ax1t'].xaxis.set_ticks_position('none')
        self.axes['ax1t'].plot(Tzz, Tzz*0+lsfru+(lsfru-lsfrl)*.00, marker='|', color='k', ms=3, linestyle='None')

        self.axes['ax2t'].xaxis.set_major_locator(ticker.FixedLocator(Tzz[:]))
        self.axes['ax2t'].xaxis.set_major_formatter(ticker.FixedFormatter(zredl[:]))
        self.axes['ax2t'].tick_params(axis='x', labelcolor='k')
        self.axes['ax2t'].xaxis.set_ticks_position('none')
        self.axes['ax2t'].plot(Tzz, Tzz*0+y2max+(y2max-y2min)*.00, marker='|', color='k', ms=3, linestyle='None')

        # This has to come after set_xticks;
        self.axes['ax1t'].set_xlim(Txmin, Txmax)
        self.axes['ax2t'].set_xlim(Txmin, Txmax)
        self.axes['ax1'].legend(loc=0, fontsize=9)

        dely2 = 0.1
        while (y2max-y2min)/dely2>7:
            dely2 *= 2.

        y2ticks = np.arange(y2min, y2max, dely2)
        self.axes['ax2'].set_yticks(y2ticks)
        self.axes['ax2'].set_yticklabels(np.arange(y2min, y2max, dely2), minor=False)
        self.axes['ax2'].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        return 


    def define_axis(self, f_log_sfh=True, skip_zhist=True):
        ''''''
        self.axes = {'ax1':None, 'ax1t':None, 'ax2':None, 'ax2t':None, 'ax4':None, 'ax4t':None}
        if f_log_sfh:
            self.axes['fig'] = plt.figure(figsize=(8,2.8))
            self.axes['fig'].subplots_adjust(top=0.88, bottom=0.18, left=0.07, right=0.99, hspace=0.15, wspace=0.3)
        else:
            self.axes['fig'] = plt.figure(figsize=(8.2,2.8))
            self.axes['fig'].subplots_adjust(top=0.88, bottom=0.18, left=0.1, right=0.99, hspace=0.15, wspace=0.3)

        if skip_zhist:
            if f_log_sfh:
                self.axes['fig'] = plt.figure(figsize=(5.5,2.8))
                self.axes['fig'].subplots_adjust(top=0.88, bottom=0.18, left=0.1, right=0.99, hspace=0.15, wspace=0.3)
            else:
                self.axes['fig'] = plt.figure(figsize=(6.2,2.8))
                self.axes['fig'].subplots_adjust(top=0.88, bottom=0.18, left=0.1, right=0.99, hspace=0.15, wspace=0.3)
            self.axes['ax1'] = self.axes['fig'].add_subplot(121)
            self.axes['ax2'] = self.axes['fig'].add_subplot(122)
        else:
            self.axes['ax1'] = self.axes['fig'].add_subplot(131)
            self.axes['ax2'] = self.axes['fig'].add_subplot(132)
            self.axes['ax4'] = self.axes['fig'].add_subplot(133)
            self.axes['ax4t'] = self.axes['ax4'].twiny()

        self.axes['ax1t'] = self.axes['ax1'].twiny()
        self.axes['ax2t'] = self.axes['ax2'].twiny()
        return self.axes


    def update_axis(self, f_log_sfh=True, skip_zhist=True, lsfrl=-1):
        ''''''
        # For redshift
        if self.zbes<4:
            if self.zbes<2:
                zred  = [self.zbes, 2, 3, 6]
                zredl = ['$z_\mathrm{obs.}$', 2, 3, 6]
            elif self.zbes<2.5:
                zred  = [self.zbes, 2.5, 3, 6]
                zredl = ['$z_\mathrm{obs.}$', 2.5, 3, 6]
            elif self.zbes<3.:
                zred  = [self.zbes, 3, 6]
                zredl = ['$z_\mathrm{obs.}$', 3, 6]
            else:
                zred  = [self.zbes, 6]
                zredl = ['$z_\mathrm{obs.}$', 6]
        elif self.zbes<6:
            zred  = [self.zbes, 5, 6, 9]
            zredl = ['$z_\mathrm{obs.}$', 5, 6, 9]
        elif self.zbes<12:
            zred  = [self.zbes, 12]
            zredl = ['$z_\mathrm{obs.}$', 12]
        else:
            zred  = [self.zbes, 20]
            zredl = ['$z_\mathrm{obs.}$', 20]

        Tzz = np.zeros(len(zred), dtype=float)
        for zz in range(len(zred)):
            Tzz[zz] = (self.Tuni - self.mb.cosmo.age(zred[zz]).value)
            if Tzz[zz] < self.Txmin:
                Tzz[zz] = self.Txmin
        
        lsfru = 2.8
        if np.max(self.SFp[:,2])>2.8:
            lsfru = np.max(self.SFp[:,2])+0.1
        if np.min(self.SFp[:,2])>lsfrl:
            lsfrl = np.min(self.SFp[:,2])+0.1

        if f_log_sfh:
            self.axes['ax1'].set_ylim(lsfrl, lsfru)
            self.axes['ax1'].set_ylabel('$\log \dot{M}_*/M_\odot$yr$^{-1}$', fontsize=12)
        else:
            self.axes['ax1'].set_ylim(0, 10**lsfru)
            self.axes['ax1'].set_ylabel('$\dot{M}_*/M_\odot$yr$^{-1}$', fontsize=12)

        self.axes['ax1'].set_xlim(self.Txmin, self.Txmax)
        self.axes['ax1'].set_xscale('log')

        self.axes['ax2'].set_ylabel('$\log M_*/M_\odot$', fontsize=12)
        self.axes['ax2'].set_xlim(self.Txmin, self.Txmax)
        self.axes['ax2'].set_ylim(self.y2min, self.y2max)
        self.axes['ax2'].set_xscale('log')
        self.axes['ax2'].text(np.min(self.mb.age*1.05), self.y2min + 0.07*(self.y2max-self.y2min), 'ID: %s\n$z_\mathrm{obs.}:%.2f$\n$\log M_\mathrm{*}/M_\odot:%.2f$\n$\log Z_\mathrm{*}/Z_\odot:%.2f$\n$\log T_\mathrm{*}$/Gyr$:%.2f$\n$A_V$/mag$:%.2f$'\
            %(self.mb.ID, self.zbes, self.ACp[0,1], self.ZCp[0,1], np.nanmedian(self.TC[0,:]), self.Avtmp[1]), fontsize=9, bbox=dict(facecolor='w', alpha=0.7), zorder=10)

        # SFH
        # zzall = np.arange(1.,12,0.01)
        # Tall = self.mb.cosmo.age(zzall).value # , use_flat=True, **cosmo)

        dely2 = 0.1
        while (self.y2max-self.y2min)/dely2>7:
            dely2 *= 2.

        y2ticks = np.arange(self.y2min, self.y2max, dely2)
        self.axes['ax2'].set_yticks(y2ticks)
        self.axes['ax2'].set_yticklabels(np.arange(self.y2min, self.y2max, dely2), minor=False)
        self.axes['ax2'].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        if not skip_zhist:
            y3min, y3max = np.min([np.min(self.mb.Zall),-0.8]), np.max([np.max(self.mb.Zall),0.4])
            self.axes['ax4'].set_xlim(self.Txmin, self.Txmax)
            self.axes['ax4'].set_ylim(y3min-0.05, y3max)
            self.axes['ax4'].set_xscale('log')
            self.axes['ax4'].set_yticks([-0.8, -0.4, 0., 0.4])
            self.axes['ax4'].set_yticklabels(['-0.8', '-0.4', '0', '0.4'])
            
            #self.axes['ax4'].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            #ax3.yaxis.labelpad = -2
            self.axes['ax4'].yaxis.labelpad = -2
            self.axes['ax4'].set_xlabel('$t_\mathrm{lookback}$/Gyr', fontsize=12)
            self.axes['ax4'].set_ylabel('$\log Z_*/Z_\odot$', fontsize=12)
            self.axes['ax4t'].set_xscale('log')
            self.axes['ax4t'].xaxis.set_major_locator(ticker.FixedLocator(Tzz[:]))
            self.axes['ax4t'].xaxis.set_major_formatter(ticker.FixedFormatter(zredl[:]))
            self.axes['ax4t'].tick_params(axis='x', labelcolor='k')
            self.axes['ax4t'].xaxis.set_ticks_position('none')
            self.axes['ax4t'].plot(Tzz, Tzz*0+y3max+(y3max-y3min)*.00, marker='|', color='k', ms=3, linestyle='None')
            self.axes['ax4t'].set_xlim(self.Txmin, self.Txmax)

        _ = self.get_sfh_figure_format(Tzz, zredl, lsfrl, lsfru, self.y2min, self.y2max, self.Txmin, self.Txmax)

        return


    @staticmethod
    def sfr_tau(t0, tau0, Z=0.0, sfh=0, tt=np.arange(0,13,0.1), Mtot=1.,
        sfrllim=1e-20):
        '''
        Parameters
        ----------
        sfh : int
            1:exponential, 4:delayed exp, 5:, 6:lognormal
        ML : float
            Total Mass.
        tt : float
            Lookback time, in Gyr
        tau0: float
            in Gyr
        t0 : float
            age, in Gyr

        Returns
        -------
        SFR : 
            in Msun/yr
        MFR :
            in Msun

        '''
        yy = np.zeros(len(tt), dtype=float) 
        yyms = np.zeros(len(tt), dtype=float)
        con = (tt<=t0)
        if sfh == 1:
            yy[con] = np.exp((tt[con]-t0)/tau0)
        elif sfh == 4:
            yy[con] = (t0-tt[con]) * np.exp((tt[con]-t0)/tau0)
        elif sfh == 6: # lognorm
            con = (tt>0)
            yy[con] = 1. / np.sqrt(2*np.pi*tau0**2) * np.exp(-(np.log(tt[con])-np.log(t0))**2/(2*tau0**2)) / tt[con]

        # Total mass calculation;
        #deltt = (tt[1] - tt[0]) #* 1e9
        yyms[:] = np.cumsum(yy[::-1])[::-1] #* deltt * 1e9 # in Msun
        
        # Normalization;
        deltt = tt[1] - tt[0]
        C = Mtot/np.max(yyms)
        yyms *= C
        yy *= C / deltt / 1e9 # in Msun/yr

        yy[~con] = sfrllim #1e-20
        yyms[~con] = sfrllim #1e-20
        return tt, yy, yyms


    def plot_sfh_tau(self, f_comp=0, flim=0.01, lsfrl=-1, mmax=1000, Txmin=0.08, Txmax=4, lmmin=8.5, fil_path='./FILT/',
        dust_model=0, f_SFMS=False, f_symbol=True, verbose=False, DIR_TMP=None,
        f_log_sfh=True, dpi=250, TMIN=0.0001, tau_lim=0.01, skip_zhist=True, tsets_SFR_SED=[0.001,0.003,0.01,0.03,0.1,0.3], tset_SFR_SED=0.1, return_figure=False,
        f_sfh_yaxis_force=True):
        '''
        Purpose
        -------
        Star formation history plot.

        Parameters
        ----------
        flim : float
            Lower limit for plotting an age bin.
        lsfrl : float
            Lower limit for SFR, in logMsun/yr
        f_SFMS : bool
            If true, plot SFR of the main sequence of a ginen stellar mass at each lookback time.
        tset_SFR_SED : float
            in Gyr. Time scale over which SFR estimate is averaged.
        '''
        ###########################
        # Open result file
        ###########################
        file = self.mb.DIR_OUT + 'gsf_params_' + self.mb.ID + '.fits'
        hdul = fits.open(file) # open a FITS file
        try:
            self.zbes = hdul[0].header['zmc']
        except:
            self.zbes = hdul[0].header['z']
        try:
            self.RA   = hdul[0].header['RA']
            self.DEC  = hdul[0].header['DEC']
        except:
            self.RA  = 0
            self.DEC = 0
        try:
            SN = hdul[0].header['SN']
        except:
            ###########################
            # Get SN of Spectra
            ###########################
            file = os.path.join(DIR_TMP, 'spec_obs_' + self.mb.ID + '.cat')
            fds  = np.loadtxt(file, comments='#')
            nrs  = fds[:,0]
            lams = fds[:,1]
            fsp  = fds[:,2]
            esp  = fds[:,3]

            consp = (nrs<10000) & (lams/(1.+self.zbes)>3600) & (lams/(1.+self.zbes)<4200)
            if len((fsp/esp)[consp]>10):
                SN = np.median((fsp/esp)[consp])
            else:
                SN = 1

        Asum = 0
        A50 = np.arange(len(self.mb.age), dtype=float)
        for aa in range(len(A50)):
            A50[aa] = 10**hdul[1].data['A'+str(aa)][1]
            Asum += A50[aa]

        ####################
        # For cosmology
        ####################
        self.Tuni = self.mb.cosmo.age(self.zbes).value #, use_flat=True, **cosmo)
        delT  = np.zeros(len(self.mb.age),dtype=float)
        delTl = np.zeros(len(self.mb.age),dtype=float)
        delTu = np.zeros(len(self.mb.age),dtype=float)

        if len(self.mb.age) == 1:
        #if tau0[0] < 0: # SSP;
            for aa in range(len(self.mb.age)):
                try:
                    tau_ssp = float(self.mb.inputs['TAU_SSP'])
                except:
                    tau_ssp = tau_lim
                delTl[aa] = tau_ssp/2
                delTu[aa] = tau_ssp/2
                if self.mb.age[aa] < tau_lim:
                    delT[aa] = tau_lim
                else:
                    delT[aa] = delTu[aa] + delTl[aa]
        else: # This is only true when CSP...
            for aa in range(len(self.mb.age)):
                if aa == 0:
                    delTl[aa] = self.mb.age[aa]
                    delTu[aa] = (self.mb.age[aa+1]-self.mb.age[aa])/2.
                    delT[aa]  = delTu[aa] + delTl[aa]
                elif self.Tuni < self.mb.age[aa]:
                    delTl[aa] = (self.mb.age[aa]-self.mb.age[aa-1])/2.
                    delTu[aa] = self.Tuni-self.mb.age[aa] #delTl[aa] #10.
                    delT[aa]  = delTu[aa] + delTl[aa]
                elif aa == len(self.mb.age)-1:
                    delTl[aa] = (self.mb.age[aa]-self.mb.age[aa-1])/2.
                    delTu[aa] = self.Tuni - self.mb.age[aa]
                    delT[aa]  = delTu[aa] + delTl[aa]
                else:
                    delTl[aa] = (self.mb.age[aa]-self.mb.age[aa-1])/2.
                    delTu[aa] = (self.mb.age[aa+1]-self.mb.age[aa])/2.
                    if self.mb.age[aa]+delTu[aa]>self.Tuni:
                        delTu[aa] = self.Tuni-self.mb.age[aa]
                    delT[aa] = delTu[aa] + delTl[aa]

        mask_age = (delT<=0) # For those age_template > age_universe
        delT[mask_age] = np.inf
        delT[:] *= 1e9 # Gyr to yr
        delTl[:] *= 1e9 # Gyr to yr
        delTu[:] *= 1e9 # Gyr to yr

        ##############################
        # Load Pickle
        ##############################
        samplepath = self.mb.DIR_OUT 
        pfile = 'gsf_chain_' + self.mb.ID + '.cpkl'

        data = loadcpkl(os.path.join(samplepath+'/'+pfile))
        try:
            samples = data['chain'][:]
        except:
            msg = ' =   >   NO keys of ndim and burnin found in cpkl, use input keyword values'
            print_err(msg, exit=False)
            return -1

        ######################
        # Mass-to-Light ratio.
        ######################
        AM = np.zeros((len(self.mb.age), mmax), dtype=float) # Mass in each bin.
        AC = np.zeros((len(self.mb.age), mmax), dtype=float) -99 # Cumulative mass in each bin.
        AL = np.zeros((len(self.mb.age), mmax), dtype=float) # Cumulative light in each bin.
        ZM = np.zeros((len(self.mb.age), mmax), dtype=float) # Z.
        ZC = np.zeros((len(self.mb.age), mmax), dtype=float) -99 # Cumulative Z.
        ZL = np.zeros((len(self.mb.age), mmax), dtype=float) -99 # Light weighted cumulative Z.
        TC = np.zeros((len(self.mb.age), mmax), dtype=float) # Mass weighted T.
        TL = np.zeros((len(self.mb.age), mmax), dtype=float) # Light weighted T.
        ZMM = np.zeros((len(self.mb.age), mmax), dtype=float) # Mass weighted Z.
        ZML= np.zeros((len(self.mb.age), mmax), dtype=float) # Light weighted Z.
        SF = np.zeros((len(self.mb.age), mmax), dtype=float) # SFR
        Av = np.zeros(mmax, dtype=float) # SFR

        # ##############################
        # Add simulated scatter in quad
        # if files are available.
        # ##############################
        try:
            f_zev = int(self.mb.inputs['ZEVOL'])
        except:
            f_zev = 1

        try:
            meanfile = './sim_SFH_mean.cat'
            dfile = np.loadtxt(meanfile, comments='#')
            eA = dfile[:,2]
            eZ = dfile[:,4]
            eAv= np.mean(dfile[:,6])
            if f_zev == 0:
                eZ[:] = self.mb.age * 0 #+ eZ_mean
            else:
                try:
                    f_zev = int(prihdr['ZEVOL'])
                    if f_zev == 0:
                        eZ = self.mb.age * 0
                except:
                    pass
        except:
            if verbose:
                self.mb.logger.warning('No simulation file (%s).\nError may be underestimated.' % meanfile)
            eA = self.mb.age * 0
            eZ = self.mb.age * 0
            eAv = 0

        ##################
        # Define axis
        ##################
        _ = self.define_axis(f_log_sfh=f_log_sfh, skip_zhist=skip_zhist)

        #####################
        # Get SED based SFR
        #####################
        SFRs_SED = np.zeros((mmax,len(tsets_SFR_SED)),dtype=float)

        # ASDF;
        af = self.mb.af #asdf.open(self.mb.DIR_TMP + 'spec_all_' + self.mb.ID + '.asdf')
        af0 = asdf.open(self.mb.DIR_TMP + 'spec_all.asdf')
        sedpar = af['ML'] # For M/L
        sedpar0 = af0['ML'] # For mass loss frac.

        ttmin = 0.001
        tt = np.arange(ttmin,self.Tuni+0.5,ttmin/10)
        xSF = np.zeros((len(tt), mmax), dtype=float) # SFR
        ySF = np.zeros((len(tt), mmax), dtype=float) # SFR
        yMS = np.zeros((len(tt), mmax), dtype=float) # MFR
        ySF_each = np.zeros((self.mb.npeak, len(tt), mmax), dtype=float) # SFR
        yMS_each = np.zeros((self.mb.npeak, len(tt), mmax), dtype=float) # MFR

        ZZmc = np.zeros((self.mb.npeak, mmax), dtype=float) 
        TTmc = np.zeros((self.mb.npeak, mmax), dtype=float) 
        TAmc = np.zeros((self.mb.npeak, mmax), dtype=float) 

        if Txmin > np.min(tt):
            Txmin = np.min(tt) * 0.8

        mm = 0
        plot_each = True
        while mm<mmax:
            mtmp = np.random.randint(len(samples))# + Nburn
            if self.mb.has_AVFIX:
                Av_tmp = self.mb.AVFIX
            else:
                try:
                    Av_tmp = samples['AV0'][mtmp]
                except:
                    Av_tmp = samples['AV'][mtmp]

            for aa in range(self.mb.npeak):
                AAtmp = samples['A%d'%aa][mtmp]
                ltautmp = samples['TAU%d'%aa][mtmp]
                lagetmp = samples['AGE%d'%aa][mtmp]
                if aa == 0 or self.mb.ZEVOL:
                    try:
                        ZZtmp = samples['Z%d'%aa][mtmp]
                    except:
                        ZZtmp = self.mb.ZFIX

                ZZmc[aa,mm] = ZZtmp
                TAmc[aa,mm] = lagetmp
                TTmc[aa,mm] = ltautmp

                nZtmp,nttmp,natmp = self.mb.bfnc.Z2NZ(ZZtmp, ltautmp, lagetmp)
                mslist = sedpar['ML_'+str(nZtmp)+'_'+str(nttmp)][natmp]
                # f_m_sur = sedpar0['frac_mass_survive_%d'%nZtmp][natmp]

                xSF[:,mm], ySF_each[aa,:,mm], yMS_each[aa,:,mm] = PLOT.sfr_tau(10**lagetmp, 10**ltautmp, ZZtmp, sfh=self.mb.SFH_FORM, tt=tt, Mtot=10**AAtmp*mslist)
                ySF[:,mm] += ySF_each[aa,:,mm]
                yMS[:,mm] += yMS_each[aa,:,mm]

                # SFR from SED. This will be converted in log later;
                for t in range(len(tsets_SFR_SED)):
                    iix = np.argmin(np.abs(tt-tsets_SFR_SED[t]))
                    con_sfr = (tt<tsets_SFR_SED[t])
                    if len(ySF_each[aa,:,mm][con_sfr])>0:
                        SFRs_SED[mm,t] += np.sum(ySF_each[aa,:,mm][con_sfr])

            Av[mm] = Av_tmp
            if plot_each:
                self.axes['ax1'].plot(xSF[:,mm], np.log10(ySF[:,mm]), linestyle='-', color='k', alpha=0.01, zorder=-1, lw=0.5)
                self.axes['ax2'].plot(xSF[:,mm], np.log10(yMS[:,mm]), linestyle='-', color='k', alpha=0.01, zorder=-1, lw=0.5)

            # Convert SFRs_SED to log
            for t in range(len(tsets_SFR_SED)):
                if SFRs_SED[mm,t] > 0:
                    SFRs_SED[mm,t] = np.log10(SFRs_SED[mm,t])
                else:
                    SFRs_SED[mm,t] = -99

            mm += 1

        self.Avtmp = np.percentile(Av[:],[16,50,84])

        #############
        # Plot
        #############
        xSFp = np.zeros((len(tt),3), dtype=float)
        ySFp = np.zeros((len(tt),3), dtype=float)
        yMSp = np.zeros((len(tt),3), dtype=float)
        ySFp_each = np.zeros((self.mb.npeak, len(tt), 3), dtype=float)
        yMSp_each = np.zeros((self.mb.npeak, len(tt), 3), dtype=float)
        for ii in range(len(tt)):
            xSFp[ii,:] = np.percentile(xSF[ii,:], [16,50,84])
            ySFp[ii,:] = np.percentile(ySF[ii,:], [16,50,84])
            yMSp[ii,:] = np.percentile(yMS[ii,:], [16,50,84])
            for aa in range(self.mb.npeak):
                ySFp_each[aa,ii,:] = np.percentile(ySF_each[aa,ii,:], [16,50,84])
                yMSp_each[aa,ii,:] = np.percentile(yMS_each[aa,ii,:], [16,50,84])

        for aa in range(self.mb.npeak):
            self.axes['ax1'].plot(xSFp[:,1], np.log10(ySFp_each[aa,:,1]), linestyle='-', color=self.col[aa], alpha=1., zorder=-1, lw=0.5)
            self.axes['ax2'].plot(xSFp[:,1], np.log10(ySFp_each[aa,:,1]), linestyle='-', color=self.col[aa], alpha=1., zorder=-1, lw=0.5)

        self.axes['ax1'].plot(xSFp[:,1], np.log10(ySFp[:,1]), linestyle='-', color='k', alpha=1., zorder=-1, lw=0.5)
        self.axes['ax2'].plot(xSFp[:,1], np.log10(yMSp[:,1]), linestyle='-', color='k', alpha=1., zorder=-1, lw=0.5)

        ACp = np.zeros((len(tt),3),float)
        SFp = np.zeros((len(tt),3),float)
        ACp[:] = np.log10(yMSp[:,:])
        SFp[:] = np.log10(ySFp[:,:])

        ###################
        msize = np.zeros(len(self.mb.age), dtype=float)
        # Metal
        ZCp = np.zeros((self.mb.npeak,3),float)
        TCp = np.zeros((self.mb.npeak,3),float)
        TTp = np.zeros((self.mb.npeak,3),float)
        for aa in range(len(self.mb.age)):
            if A50[aa]/Asum>flim: # if >1%
                msize[aa] = 200 * A50[aa]/Asum

            ZCp[aa,:] = np.percentile(ZZmc[aa,:], [16,50,84])
            TCp[aa,:] = np.percentile(TTmc[aa,:], [16,50,84])
            TTp[aa,:] = np.percentile(TAmc[aa,:], [16,50,84])

        if False:
            conA = (msize>=0)
            if f_log_sfh:
                self.axes['ax1'].fill_between(age[conA], SFp[:,0][conA], SFp[:,2][conA], linestyle='-', color='k', alpha=0.5, zorder=-1)
                self.axes['ax1'].errorbar(age, SFp[:,1], linestyle='-', color='k', marker='', zorder=-1, lw=.5)
            else:
                self.axes['ax1'].fill_between(age[conA], 10**SFp[:,0][conA], 10**SFp[:,2][conA], linestyle='-', color='k', alpha=0.5, zorder=-1)
                self.axes['ax1'].errorbar(age, 10**SFp[:,1], linestyle='-', color='k', marker='', zorder=-1, lw=.5)

        #############
        # Get SFMS in log10;
        #############
        IMF = int(self.mb.inputs['NIMF'])
        SFMS_16 = get_SFMS(self.zbes,tt,10**ACp[:,0],IMF=IMF)
        SFMS_50 = get_SFMS(self.zbes,tt,10**ACp[:,1],IMF=IMF)
        SFMS_84 = get_SFMS(self.zbes,tt,10**ACp[:,2],IMF=IMF)

        #try:
        if False:
            f_rejuv,t_quench,t_rejuv = check_rejuv(age,SFp[:,:],ACp[:,:],SFMS_50)
        else:
            if verbose:
                self.mb.logger.warning('Failed to call rejuvenation module.')
            self.f_rejuv,self.t_quench,self.t_rejuv = 0,0,0

        # Plot MS?
        conA = ()
        if f_SFMS:
            if f_log_sfh:
                self.axes['ax1'].fill_between(tt[conA], SFMS_50[conA]-0.2, SFMS_50[conA]+0.2, linestyle='-', color='b', alpha=0.3, zorder=-2)
                self.axes['ax1'].plot(tt[conA], SFMS_50[conA], linestyle='--', color='k', alpha=0.5, zorder=-2)

        # Plot limit;
        y2min = np.nanmax([lmmin,np.min(np.log10(yMSp[:,1]))])
        y2max = np.nanmax(np.log10(yMSp[:,1]))+0.05
        if np.abs(y2max-y2min) < 0.2:
            y2min -= 0.2

        # Total Metal
        if not skip_zhist:
            self.axes['ax4'].fill_between(self.mb.age[conA], ZCp[:,0][conA], ZCp[:,2][conA], linestyle='-', color='k', alpha=0.5)
            self.axes['ax4'].errorbar(self.mb.age[conA], ZCp[:,1][conA], linestyle='-', color='k', lw=0.5, zorder=1)
            
            for ii in range(len(self.mb.age)):
                aa = len(self.mb.age) -1 - ii
                if msize[aa]>0:
                    self.axes['ax4'].errorbar(self.mb.age[aa], ZCp[aa,1], xerr=[[delTl[aa]/1e9],[delTu[aa]/1e9]], yerr=[[ZCp[aa,1]-ZCp[aa,0]],[ZCp[aa,2]-ZCp[aa,1]]], linestyle='-', color=self.col[aa], lw=1, zorder=1)
                    self.axes['ax4'].scatter(self.mb.age[aa], ZCp[aa,1], marker='.', c=[self.col[aa]], edgecolor='k', s=msize[aa], zorder=2)

        #
        # percs = [16,50,84]
        # zmc = hdul[1].data['zmc']
        self.ACP = [ACp[0,0], ACp[0,1], ACp[0,2]]
        self.ACp = ACp
        self.ZCP = [ZCp[0,0], ZCp[0,1], ZCp[0,2]]
        self.ZCp = ZCp
        self.ZLP = self.ZCP #[ZLp[0,0], ZLp[0,1], ZLp[0,2]]
        self.TLW = TTp[0,:]
        self.TMW = TTp[0,:]
        self.TAW = TCp[0,:]
        self.TC = TC
        self.TL = TL
        self.SFRs_SED = SFRs_SED
        self.xSFp = xSFp
        self.SFp = SFp
        self.delT = delT
        self.delTl = delTl
        self.delTu = delTu
        self.hdul = hdul
        self.Txmin = Txmin
        self.Txmax = Txmax
        self.y2min = y2min
        self.y2max = y2max

        # Update axis
        self.update_axis(f_log_sfh=f_log_sfh, skip_zhist=skip_zhist, lsfrl=lsfrl)

        # Write files
        tree_sfh = self.save_files(tsets_SFR_SED=tsets_SFR_SED, taumodel=True)

        # Attach to MB;
        self.mb.sfh_tlook = self.mb.age
        self.mb.sfh_tlookl= delTl[:]/1e9
        self.mb.sfh_tlooku= delTu[:]/1e9
        self.mb.sfh_sfr16 = SFp[:,0]
        self.mb.sfh_sfr50 = SFp[:,1]
        self.mb.sfh_sfr84 = SFp[:,2]
        self.mb.sfh_mfr16 = ACp[:,0]
        self.mb.sfh_mfr50 = ACp[:,1]
        self.mb.sfh_mfr84 = ACp[:,2]
        self.mb.sfh_zfr16 = ZCp[:,0]
        self.mb.sfh_zfr50 = ZCp[:,1]
        self.mb.sfh_zfr84 = ZCp[:,2]

        # Save
        self.axes['fig'].savefig(self.mb.DIR_OUT + 'gsf_sfh_' + self.mb.ID + '.png', dpi=dpi)

        if return_figure:
            return self.axes['fig']

        self.axes['fig'].clear()
        plt.close()

        if return_figure:
            return tree_sfh, self.axes['fig']

        return tree_sfh


    def save_files(self, tsets_SFR_SED=[], taumodel=False):
        ''''''
        #
        # Brief Summary
        #
        # Writing SED param in a fits file;
        # Header
        prihdr = fits.Header()
        prihdr['ID'] = self.mb.ID
        prihdr['z'] = self.zbes
        prihdr['RA'] = self.RA
        prihdr['DEC'] = self.DEC
        # Add rejuv properties;
        prihdr['f_rejuv'] = self.f_rejuv
        prihdr['t_quen'] = self.t_quench
        prihdr['t_rejuv'] = self.t_rejuv
        # SFR
        # prihdr['tset_SFR'] = tset_SFR_SED
        prihdr['tsets_SFR'] = ','.join(['%s'%s for s in tsets_SFR_SED])
        # Version;
        import gsf
        prihdr['version'] = gsf.__version__

        percs = [16,50,84]
        zmc = self.hdul[1].data['zmc']
        # ACP = [ACp[0,0], ACp[0,1], ACp[0,2]]
        # ZCP = [ZCp[0,0], ZCp[0,1], ZCp[0,2]]
        # ZLP = [ZLp[0,0], ZLp[0,1], ZLp[0,2]]
        con = (~np.isnan(self.TC[0,:]))
        TMW = [np.percentile(self.TC[0,:][con],16), np.percentile(self.TC[0,:][con],50), np.percentile(self.TC[0,:][con],84)]
        con = (~np.isnan(self.TL[0,:]))
        TLW = [np.percentile(self.TL[0,:][con],16), np.percentile(self.TL[0,:][con],50), np.percentile(self.TL[0,:][con],84)]

        for ii in range(len(percs)):
            prihdr['zmc_%d'%percs[ii]] = ('%.3f'%zmc[ii],'redshift')
        for ii in range(len(percs)):
            prihdr['HIERARCH Mstel_%d'%percs[ii]] = ('%.3f'%self.ACP[ii], 'Stellar mass, logMsun')
        for t in range(len(tsets_SFR_SED)):
            SFR_SED_med_tmp = np.nanpercentile(self.SFRs_SED[:,t],[16,50,84])
            for ii in range(len(percs)):
                prihdr['HIERARCH SFR_%dMyr_%d'%(tsets_SFR_SED[t]*1e3, percs[ii])] = ('%.3f'%SFR_SED_med_tmp[ii], 'SFR, logMsun/yr')
        for ii in range(len(percs)):
            prihdr['HIERARCH Z_MW_%d'%percs[ii]] = ('%.3f'%self.ZCP[ii], 'Mass-weighted metallicity, logZsun')
        for ii in range(len(percs)):
            prihdr['HIERARCH Z_LW_%d'%percs[ii]] = ('%.3f'%self.ZLP[ii], 'Light-weighted metallicity, logZsun')
        for ii in range(len(percs)):
            prihdr['HIERARCH T_MW_%d'%percs[ii]] = ('%.3f'%TMW[ii], 'Mass-weighted age, logGyr')
        for ii in range(len(percs)):
            prihdr['HIERARCH T_LW_%d'%percs[ii]] = ('%.3f'%TLW[ii], 'Light-weighted age, logGyr')
        for ii in range(len(percs)):
            prihdr['AV0_%d'%percs[ii]] = ('%.3f'%self.Avtmp[ii], 'Dust attenuation, mag')
        prihdu = fits.PrimaryHDU(header=prihdr)

        # For SFH plot;
        t0 = self.Tuni - self.mb.age[:]
        col02 = []
        col50 = fits.Column(name='time', format='E', unit='Gyr', array=self.mb.age[:])
        col02.append(col50)
        col50 = fits.Column(name='time_l', format='E', unit='Gyr', array=self.mb.age[:]-self.delTl[:]/1e9)
        col02.append(col50)
        col50 = fits.Column(name='time_u', format='E', unit='Gyr', array=self.mb.age[:]+self.delTl[:]/1e9)
        col02.append(col50)
        col50 = fits.Column(name='SFR16', format='E', unit='logMsun/yr', array=self.SFp[:,0])
        col02.append(col50)
        col50 = fits.Column(name='SFR50', format='E', unit='logMsun/yr', array=self.SFp[:,1])
        col02.append(col50)
        col50 = fits.Column(name='SFR84', format='E', unit='logMsun/yr', array=self.SFp[:,2])
        col02.append(col50)
        col50 = fits.Column(name='Mstel16', format='E', unit='logMsun', array=self.ACp[:,0])
        col02.append(col50)
        col50 = fits.Column(name='Mstel50', format='E', unit='logMsun', array=self.ACp[:,1])
        col02.append(col50)
        col50 = fits.Column(name='Mstel84', format='E', unit='logMsun', array=self.ACp[:,2])
        col02.append(col50)
        col50 = fits.Column(name='Z16', format='E', unit='logZsun', array=self.ZCp[:,0])
        col02.append(col50)
        col50 = fits.Column(name='Z50', format='E', unit='logZsun', array=self.ZCp[:,1])
        col02.append(col50)
        col50 = fits.Column(name='Z84', format='E', unit='logZsun', array=self.ZCp[:,2])
        col02.append(col50)
        
        colms = fits.ColDefs(col02)
        dathdu = fits.BinTableHDU.from_columns(colms)
        hdu = fits.HDUList([prihdu, dathdu])
        file_sfh = self.mb.DIR_OUT + 'SFH_' + self.mb.ID + '.fits'
        hdu.writeto(file_sfh, overwrite=True)

        # ASDF;
        tree_sfh = {}
        tree_sfh['header'] = {}
        tree_sfh['sfh'] = {}

        # Dump physical parameters;
        for key in prihdu.header:
            if key not in tree_sfh:
                if key.split('_')[0] == 'SFR':
                    tree_sfh['header'].update({'%s'%key: 10**float(prihdu.header[key]) * u.solMass / u.yr})
                elif key.split('_')[0] == 'Mstel':
                    tree_sfh['header'].update({'%s'%key: 10**float(prihdu.header[key]) * u.solMass})
                elif key.split('_')[0] == 'T':
                    tree_sfh['header'].update({'%s'%key: 10**float(prihdu.header[key]) * u.Gyr})
                elif key.split('_')[0] == 'AV0':
                    tree_sfh['header'].update({'%s'%key: float(prihdu.header[key]) * u.mag})
                else:
                    tree_sfh['header'].update({'%s'%key: prihdu.header[key]})

        # Mask values when age>age_uni;
        mask_age = self.xSFp[:,1] > self.Tuni
        arrays = [self.SFp[:,0],self.SFp[:,1],self.SFp[:,2],self.ACp[:,0],self.ACp[:,1],self.ACp[:,2]]#
        if not taumodel:
            arrays += [self.ZCp[:,0],self.ZCp[:,1],self.ZCp[:,2]]
        for arr in arrays:
            arr[mask_age] = np.nan

        tree_sfh['sfh'].update({'time': self.mb.age * u.Gyr})
        tree_sfh['sfh'].update({'time_l': (self.mb.age[:]-self.delTl[:]/1e9) * u.Gyr})
        tree_sfh['sfh'].update({'time_u': (self.mb.age[:]+self.delTl[:]/1e9) * u.Gyr})
        tree_sfh['sfh'].update({'SFR16': 10**self.SFp[:,0] * u.Msun / u.yr})
        tree_sfh['sfh'].update({'SFR50': 10**self.SFp[:,1] * u.Msun / u.yr})
        tree_sfh['sfh'].update({'SFR84': 10**self.SFp[:,2] * u.Msun / u.yr})
        tree_sfh['sfh'].update({'Mstel16': 10**self.ACp[:,0] * u.Msun})
        tree_sfh['sfh'].update({'Mstel50': 10**self.ACp[:,1] * u.Msun})
        tree_sfh['sfh'].update({'Mstel84': 10**self.ACp[:,2] * u.Msun})
        tree_sfh['sfh'].update({'logZ16': self.ZCp[:,0]})
        tree_sfh['sfh'].update({'logZ50': self.ZCp[:,1]})
        tree_sfh['sfh'].update({'logZ84': self.ZCp[:,2]})

        af = asdf.AsdfFile(tree_sfh)
        af.write_to(os.path.join(self.mb.DIR_OUT, 'gsf_sfh_%s.asdf'%(self.mb.ID)), all_array_compression='zlib')

        # Attach to MB;
        conA = ()
        self.mb.sfh_tlook = self.mb.age
        self.mb.sfh_tlookl= self.delTl[:][conA]/1e9
        self.mb.sfh_tlooku= self.delTu[:][conA]/1e9
        self.mb.sfh_sfr16 = self.SFp[:,0]
        self.mb.sfh_sfr50 = self.SFp[:,1]
        self.mb.sfh_sfr84 = self.SFp[:,2]
        self.mb.sfh_mfr16 = self.ACp[:,0]
        self.mb.sfh_mfr50 = self.ACp[:,1]
        self.mb.sfh_mfr84 = self.ACp[:,2]
        self.mb.sfh_zfr16 = self.ZCp[:,0]
        self.mb.sfh_zfr50 = self.ZCp[:,1]
        self.mb.sfh_zfr84 = self.ZCp[:,2]

        return tree_sfh
