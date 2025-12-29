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
from lmfit import Parameters
from astropy.convolution import convolve

# Custom modules
from .function import *
# from .function_class import Func
# from .basic_func import Basic
from .function_igm import *
from .maketmp_filt import get_LSF

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


    def update_axis_sfh(self, f_log_sfh=True, skip_zhist=True, lsfrl=-1):
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


    def plot_sfh(self, flim=0.01, lsfrl=-3, mmax=1000, Txmin=0.08, Txmax=4, lmmin=5, fil_path='./FILT/',
        dust_model=0, f_SFMS=False, f_symbol=True, verbose=False, f_silence=True, DIR_TMP=None,
        f_log_sfh=True, dpi=250, TMIN=0.0001, tau_lim=0.01, skip_zhist=False, 
        tsets_SFR_SED=[0.001,0.003,0.01,0.03,0.1,0.3], tset_SFR_SED=0.1, f_sfh_yaxis_force=True,
        return_figure=False):
        '''
        Purpose
        -------
        Star formation history plot.

        Parametes
        ---------
        flim : float
            Lower limit for plotting an age bin.
        lsfrl : float
            Lower limit for SFR, in logMsun/yr
        f_SFMS : bool
            If true, plot SFR of the main sequence of a ginen stellar mass at each lookback time.
        tset_SFR_SED : float
            in Gyr. Time scale over which SFR estimate is averaged.
        '''
        MB = self.mb
        MB.logger.info('Running plot_sfh')

        bfnc = MB.bfnc
        ID = MB.ID
        Z = MB.Zall
        age = MB.age
        age = np.asarray(age)

        try:
            if not MB.ZFIX == None:
                skip_zhist = True
        except:
            pass

        if Txmin > np.min(age):
            Txmin = np.min(age) * 0.8

        ###########################
        # Open result file
        ###########################
        self.open_result_file(DIR_TMP=DIR_TMP,)

        file = self.mb.DIR_OUT + 'gsf_params_' + self.mb.ID + '.fits'
        hdul = fits.open(file) # open a FITS file

        Asum = 0
        A50 = np.arange(len(age), dtype=float)
        for aa in range(len(A50)):
            A50[aa] = 10**hdul[1].data['A'+str(aa)][1]
            Asum += A50[aa]

        ####################
        # For cosmology
        ####################
        self.Tuni = self.mb.cosmo.age(self.zbes).value #, use_flat=True, **cosmo)
        Tuni0 = (self.Tuni - age[:])
        delT  = np.zeros(len(age),dtype=float)
        delTl = np.zeros(len(age),dtype=float)
        delTu = np.zeros(len(age),dtype=float)

        if len(age) == 1:
            for aa in range(len(age)):
                try:
                    tau_ssp = float(MB.inputs['TAU_SSP'])
                except:
                    tau_ssp = tau_lim
                delTl[aa] = tau_ssp/2.
                delTu[aa] = tau_ssp/2.
                if age[aa] < tau_lim:
                    # This is because fsps has the minimum tau = tau_lim
                    delT[aa] = tau_lim
                else:
                    delT[aa] = delTu[aa] + delTl[aa]
        else: 
            # @@@ Note: This is only true when CSP...?
            for aa in range(len(age)):
                if aa == 0:
                    delTl[aa] = age[aa]
                    delTu[aa] = (age[aa+1]-age[aa])/2.
                    delT[aa] = delTu[aa] + delTl[aa]
                elif self.Tuni < age[aa]:
                    delTl[aa] = (age[aa]-age[aa-1])/2.
                    delTu[aa] = self.Tuni-age[aa] #delTl[aa] #10.
                    delT[aa]  = delTu[aa] + delTl[aa]
                elif aa == len(age)-1:
                    delTl[aa] = (age[aa]-age[aa-1])/2.
                    delTu[aa] = self.Tuni - age[aa]
                    delT[aa]  = delTu[aa] + delTl[aa]
                else:
                    delTl[aa] = (age[aa]-age[aa-1])/2.
                    delTu[aa] = (age[aa+1]-age[aa])/2.
                    if age[aa]+delTu[aa]>self.Tuni:
                        delTu[aa] = self.Tuni-age[aa]
                    delT[aa] = delTu[aa] + delTl[aa]

                if delTu[aa]<0:
                    delTu[aa] = 1e3

        mask_age = (delT<=0) # For those age_template > age_universe
        delT[mask_age] = np.inf
        delT[:] *= 1e9 # Gyr to yr
        delTl[:] *= 1e9 # Gyr to yr
        delTu[:] *= 1e9 # Gyr to yr

        ##############################
        # Load Pickle
        ##############################
        samplepath = MB.DIR_OUT 

        niter = 0
        use_pickl = False
        use_pickl = True
        if use_pickl:
            pfile = 'gsf_chain_' + ID + '.cpkl'
            data = loadcpkl(os.path.join(samplepath+'/'+pfile))
        else:
            pfile = 'gsf_chain_' + ID + '.asdf'
            data = asdf.open(os.path.join(samplepath+'/'+pfile))

        try:
            if use_pickl:
                samples = data['chain'][:]
            else:
                samples = data['chain']
        except:
            msg = ' =   >   NO keys of ndim and burnin found in cpkl, use input keyword values'
            print_err(msg, exit=False)
            return -1

        ######################
        # Mass-to-Light ratio.
        ######################
        AM = np.zeros((len(age), mmax), dtype=float) # Mass in each bin.
        AC = np.zeros((len(age), mmax), dtype=float) -99 # Cumulative mass in each bin.
        AL = np.zeros((len(age), mmax), dtype=float) # Cumulative light in each bin.
        ZM = np.zeros((len(age), mmax), dtype=float) # Z.
        ZC = np.zeros((len(age), mmax), dtype=float) -99 # Cumulative Z.
        ZL = np.zeros((len(age), mmax), dtype=float) -99 # Light weighted cumulative Z.
        TC = np.zeros((len(age), mmax), dtype=float) # Mass weighted T.
        TL = np.zeros((len(age), mmax), dtype=float) # Light weighted T.
        ZMM= np.zeros((len(age), mmax), dtype=float) # Mass weighted Z.
        ZML= np.zeros((len(age), mmax), dtype=float) # Light weighted Z.
        SF = np.zeros((len(age), mmax), dtype=float) # SFR
        Av = np.zeros(mmax, dtype=float) # SFR

        ##################
        # Define axis
        ##################
        _ = self.define_axis(f_log_sfh=f_log_sfh, skip_zhist=skip_zhist)

        # ##############################
        # Add simulated scatter in quad
        # if files are available.
        # ##############################
        try:
            f_zev = int(MB.inputs['ZEVOL'])
        except:
            f_zev = 1

        try:
            meanfile = './sim_SFH_mean.cat'
            dfile = np.loadtxt(meanfile, comments='#')
            eA = dfile[:,2]
            eZ = dfile[:,4]
            eAv= np.mean(dfile[:,6])
            if f_zev == 0:
                eZ[:] = age * 0 #+ eZ_mean
            else:
                try:
                    f_zev = int(prihdr['ZEVOL'])
                    if f_zev == 0:
                        eZ = age * 0
                except:
                    pass
        except:
            if verbose:
                MB.logger.warning('No simulation file (%s).\nError may be underestimated.' % meanfile)
            eA = age * 0
            eZ = age * 0
            eAv = 0

        mm = 0

        #####################
        # Get SED based SFR
        #####################
        # f_SFRSED_plot = False
        # SFR_SED = np.zeros(mmax,dtype=float)
        SFRs_SED = np.zeros((mmax,len(tsets_SFR_SED)),dtype=float)

        # ASDF;
        af = MB.af #asdf.open(MB.DIR_TMP + 'spec_all_' + MB.ID + '.asdf')
        af0 = asdf.open(MB.DIR_TMP + 'spec_all.asdf')
        sedpar = af['ML'] # For M/L
        sedpar0 = af0['ML'] # For mass loss frac.

        AAtmp = np.zeros(len(age), dtype=float)
        ZZtmp = np.zeros(len(age), dtype=float)
        mslist= np.zeros(len(age), dtype=float)

        for mm in range(mmax):
            delt_tot = 0
            mtmp  = np.random.randint(len(samples))# + Nburn

            if MB.has_AVFIX:
                Av_tmp = MB.AVFIX
            else:
                try:
                    Av_tmp = samples['AV0'][mtmp]
                except:
                    Av_tmp = samples['AV'][mtmp]

            Avrand = np.random.uniform(-eAv, eAv)
            if Av_tmp + Avrand<0:
                Av[mm] = 0
            else:
                Av[mm] = Av_tmp + Avrand

            for aa in range(len(age)):
                try:
                    # This is in log.
                    AAtmp[aa] = samples['A'+str(aa)][mtmp]
                except:
                    AAtmp[aa] = -10
                    pass

                try:
                    ZZtmp[aa] = samples['Z'+str(aa)][mtmp]
                except:
                    try:
                        ZZtmp[aa] = samples['Z0'][mtmp]
                    except:
                        ZZtmp[aa] = MB.ZFIX

                nZtmp = bfnc.Z2NZ(ZZtmp[aa])
                mslist[aa] = sedpar['ML_'+str(nZtmp)][aa]
                Arand = np.random.uniform(-eA[aa],eA[aa])
                Zrand = np.random.uniform(-eZ[aa],eZ[aa])
                f_m_sur = sedpar0['frac_mass_survive_%d'%nZtmp][aa]

                # quantity in log scale;
                AM[aa, mm] = AAtmp[aa] + np.log10(mslist[aa]) + Arand 
                AL[aa, mm] = AM[aa,mm] - np.log10(mslist[aa])
                SF[aa, mm] = AAtmp[aa] + np.log10(mslist[aa] / delT[aa] / f_m_sur) + Arand # log Msun/yr
                ZM[aa, mm] = ZZtmp[aa] + Zrand
                ZMM[aa, mm]= ZZtmp[aa] + AAtmp[aa] + np.log10(mslist[aa]) + Zrand
                ZML[aa, mm]= ZMM[aa,mm] - np.log10(mslist[aa])

                # SFR from SED. This will be converted in log later;
                # if True:
                #     if age[aa]<=tset_SFR_SED:
                #         SFR_SED[mm] += 10**SF[aa, mm] * delT[aa]
                #         delt_tot += delT[aa]

            # if True:
            #     SFR_SED[mm] /= delt_tot
            #     if SFR_SED[mm] > 0:
            #         SFR_SED[mm] = np.log10(SFR_SED[mm])
            #     else:
            #         SFR_SED[mm] = -99

            for aa in range(len(age)):

                if Tuni0[aa]<0:
                    continue

                if np.sum(10**AM[aa:,mm])>0:
                    AC[aa, mm] = np.log10(np.sum(10**AM[aa:,mm]))
                    ZC[aa, mm] = np.log10(np.sum(10**ZMM[aa:,mm])/10**AC[aa, mm])
                if np.sum(10**AL[aa:,mm])>0:
                    ZL[aa, mm] = np.log10(np.sum(10**ZML[aa:,mm])/np.sum(10**AL[aa:,mm]))
                if f_zev == 0: # To avoid random fluctuation in A.
                    ZC[aa,mm] = ZM[aa,mm]

                ACs = 0
                ALs = 0
                for bb in range(aa, len(age), 1):

                    if Tuni0[bb]<0:
                        continue

                    tmpAA = 10**np.random.uniform(-eA[bb],eA[bb])
                    tmpTT = np.random.uniform(-delT[bb]/1e9/2.,delT[bb]/1e9/2.)

                    TC[aa, mm] += (age[bb]+tmpTT) * 10**AAtmp[bb] * mslist[bb] * tmpAA
                    TL[aa, mm] += (age[bb]+tmpTT) * 10**AAtmp[bb] * tmpAA
                    ACs += 10**AAtmp[bb] * mslist[bb] * tmpAA
                    ALs += 10**AAtmp[bb] * tmpAA

                TC[aa, mm] /= ACs
                TL[aa, mm] /= ALs

                if TC[aa, mm]>0:
                    TC[aa, mm] = np.log10(TC[aa, mm])
                else:
                    TC[aa, mm] = np.nan

                if TL[aa, mm]>0:
                    TL[aa, mm] = np.log10(TL[aa, mm])
                else:
                    TL[aa, mm] = np.nan

            # Get SFR from SFH;
            if True:
                # tset_SFR_SED = 0.03
                SFH_for_interp = np.asarray([s for s in 10**SF[:, mm]] + [0])
                age_for_interp = np.asarray([s for s in np.log10(age)] + [np.nanmax(np.log10(age+delT[aa]/1e9*2))])
                fint_sfr = interpolate.interp1d(age_for_interp, SFH_for_interp, kind='nearest', fill_value="extrapolate")
                delt_int = np.nanmin(age)/10 # in Gyr
                times_int = np.arange(0,np.nanmax(age),delt_int)
                sfr_int = fint_sfr(np.log10(times_int))

                fint_delt = interpolate.interp1d(np.log10(age), delT, kind='nearest', fill_value="extrapolate")
                delt_interp = fint_delt(np.log10(times_int))

                # con = (~np.isinf(sfr_int))
                # con2 = (~np.isinf(10**SF[:, mm]))
                # print(np.nansum(sfr_int[con])*delt_int, np.nansum(10**SF[:, mm][con2]))
                # hoge
                # con_sfr = (times_int<tset_SFR_SED)
                # SFR_SED_tmp = np.log10(np.nansum(sfr_int[con_sfr]*delt_int)/(tset_SFR_SED))
                # SFR_SED[mm] = SFR_SED_tmp

                for t in range(len(tsets_SFR_SED)):
                    iix = np.argmin(np.abs(times_int-tsets_SFR_SED[t]))
                    # print(tsets_SFR_SED[t], delt_interp[iix]/1e9/2.)
                    con_sfr = (times_int<tsets_SFR_SED[t]+delt_interp[iix]/1e9/2.)
                    SFRs_SED[mm,t] = np.log10(np.nansum(sfr_int[con_sfr]*delt_int)/(tsets_SFR_SED[t]))
                # print(SFR_SED[mm], SFR_SED_tmp, tset_SFR_SED, delt_int, len(sfr_int[con_sfr]))
                # plt.close()
                # ax1.plot(times_int, np.log10(sfr_int), color='green', alpha=0.1)
                # ax1.set_xscale('log')
                # # ax1.set_ylim(-3, 3)
                # plt.savefig('tmp.png')
                # hoge

            # Do stuff...
            # time.sleep(0.01)
            # Update Progress Bar
            printProgressBar(mm, mmax, prefix = 'Progress:', suffix = 'Complete', length = 40)

        self.Avtmp = np.percentile(Av[:],[16,50,84])

        #############
        # Plot
        #############
        AMp = np.zeros((len(age),3), dtype=float)
        ACp = np.zeros((len(age),3), dtype=float)
        ZMp = np.zeros((len(age),3), dtype=float)
        ZCp = np.zeros((len(age),3), dtype=float)
        ZLp = np.zeros((len(age),3), dtype=float)
        SFp = np.zeros((len(age),3), dtype=float)
        for aa in range(len(age)):
            AMp[aa,:] = np.nanpercentile(AM[aa,:], [16,50,84])
            ACp[aa,:] = np.nanpercentile(AC[aa,:], [16,50,84])
            ZMp[aa,:] = np.nanpercentile(ZM[aa,:], [16,50,84])
            ZCp[aa,:] = np.nanpercentile(ZC[aa,:], [16,50,84])
            ZLp[aa,:] = np.nanpercentile(ZL[aa,:], [16,50,84])
            SFp[aa,:] = np.nanpercentile(SF[aa,:], [16,50,84])

        # SFR_SED_med = np.nanpercentile(SFR_SED[:],[16,50,84])
        # if f_SFRSED_plot:
        #     ax1.errorbar(delt_tot/2./1e9, SFR_SED_med[1], xerr=[[delt_tot/2./1e9],[delt_tot/2./1e9]], \
        #     yerr=[[SFR_SED_med[1]-SFR_SED_med[0]],[SFR_SED_med[2]-SFR_SED_med[1]]], \
        #     linestyle='', color='orange', lw=1., marker='*',ms=8,zorder=-2)

        ###################
        msize = np.zeros(len(age), dtype=float)
        for aa in range(len(age)):
            if A50[aa]/Asum>flim: # if >1%
                msize[aa] = 200 * A50[aa]/Asum

        conA = (msize>=0)
        if f_log_sfh:
            self.axes['ax1'].fill_between(age[conA], SFp[:,0][conA], SFp[:,2][conA], linestyle='-', color='k', alpha=0.5, zorder=-1)
            self.axes['ax1'].errorbar(age, SFp[:,1], linestyle='-', color='k', marker='', zorder=-1, lw=.5)
        else:
            self.axes['ax1'].fill_between(age[conA], 10**SFp[:,0][conA], 10**SFp[:,2][conA], linestyle='-', color='k', alpha=0.5, zorder=-1)
            self.axes['ax1'].errorbar(age, 10**SFp[:,1], linestyle='-', color='k', marker='', zorder=-1, lw=.5)

        if f_symbol:
            tbnd = 0.0001
            for aa in range(len(age)):
                agebin = np.arange(age[aa]-delTl[aa]/1e9, age[aa]+delTu[aa]/1e9, delTu[aa]/1e10)
                tbnd = age[aa]+delT[aa]/2./1e9
                
                if f_log_sfh:
                    self.axes['ax1'].errorbar(age[aa], SFp[aa,1], xerr=[[delTl[aa]/1e9], [delTu[aa]/1e9]], \
                        yerr=[[SFp[aa,1]-SFp[aa,0]], [SFp[aa,2]-SFp[aa,1]]], linestyle='', color=self.col[aa], marker='', zorder=1, lw=1.)
                    if msize[aa]>0:
                        self.axes['ax1'].scatter(age[aa], SFp[aa,1], marker='.', color=self.col[aa], edgecolor='k', s=msize[aa], zorder=1)
                else:
                    self.axes['ax1'].errorbar(age[aa], 10**SFp[aa,1], xerr=[[delTl[aa]/1e9], [delTu[aa]/1e9]], \
                        yerr=[[10**SFp[aa,1]-10**SFp[aa,0]], [10**SFp[aa,2]-10**SFp[aa,1]]], linestyle='', color=self.col[aa], marker='.', zorder=1, lw=1.)
                    if msize[aa]>0:
                        self.axes['ax1'].scatter(age[aa], 10**SFp[aa,1], marker='.', color=self.col[aa], edgecolor='k', s=msize[aa], zorder=1)

        #############
        # Get SFMS in log10;
        #############
        IMF = int(MB.inputs['NIMF'])
        SFMS_16 = get_SFMS(self.zbes,age,10**ACp[:,0],IMF=IMF)
        SFMS_50 = get_SFMS(self.zbes,age,10**ACp[:,1],IMF=IMF)
        SFMS_84 = get_SFMS(self.zbes,age,10**ACp[:,2],IMF=IMF)

        #try:
        if False:
            f_rejuv,t_quench,t_rejuv = check_rejuv(age,SFp[:,:],ACp[:,:],SFMS_50)
        else:
            if verbose:
                MB.logger.warning('Failed to call rejuvenation module.')
            self.f_rejuv,self.t_quench,self.t_rejuv = 0,0,0

        # Plot MS?
        if f_SFMS:
            if f_log_sfh:
                self.axes['ax1'].fill_between(age[conA], SFMS_50[conA]-0.2, SFMS_50[conA]+0.2, linestyle='-', color='b', alpha=0.3, zorder=-2, label='SFMS')
                self.axes['ax1'].plot(age[conA], SFMS_50[conA], linestyle='--', color='k', alpha=0.5, zorder=-2)
            else:
                self.axes['ax1'].fill_between(age[conA], 10**(SFMS_50[conA]-0.2), 10**(SFMS_50[conA]+0.2), linestyle='-', color='b', alpha=0.3, zorder=-2, label='SFMS')
                self.axes['ax1'].plot(age[conA], 10**SFMS_50[conA], linestyle='--', color='k', alpha=0.5, zorder=-2)

        #
        # Mass in each bin
        #
        ax2label = ''
        self.axes['ax2'].fill_between(age[conA], ACp[:,0][conA], ACp[:,2][conA], linestyle='-', color='k', alpha=0.5)
        self.axes['ax2'].errorbar(age[conA], ACp[:,1][conA], xerr=[delTl[:][conA]/1e9,delTu[:][conA]/1e9],
            yerr=[ACp[:,1][conA]-ACp[:,0][conA],ACp[:,2][conA]-ACp[:,1][conA]], linestyle='-', color='k', lw=0.5, label=ax2label, zorder=1)

        if f_symbol:
            tbnd = 0.0001
            mtmp = 0
            for ii in range(len(age)):
                aa = len(age) -1 - ii
                agebin = np.arange(0, age[aa], delTu[aa]/1e10)
                self.axes['ax2'].errorbar(age[aa], ACp[aa,1], xerr=[[delTl[aa]/1e9],[delTu[aa]/1e9]],
                    yerr=[[ACp[aa,1]-ACp[aa,0]],[ACp[aa,2]-ACp[aa,1]]], linestyle='-', color=self.col[aa], lw=1, zorder=2)

                tbnd = age[aa]+delT[aa]/2./1e9
                mtmp = ACp[aa,1]
                if msize[aa]>0:
                    self.axes['ax2'].scatter(age[aa], ACp[aa,1], marker='.', c=[self.col[aa]], edgecolor='k', s=msize[aa], zorder=2)

        y2min = np.max([lmmin,np.min(ACp[:,0][conA])])
        y2max = np.max(ACp[:,2][conA])+0.05
        if np.abs(y2max-y2min) < 0.2:
            y2min -= 0.2

        #
        # Total Metal
        #
        if not skip_zhist:
            self.axes['ax4'].fill_between(age[conA], ZCp[:,0][conA], ZCp[:,2][conA], linestyle='-', color='k', alpha=0.5)
            self.axes['ax4'].errorbar(age[conA], ZCp[:,1][conA], linestyle='-', color='k', lw=0.5, zorder=1)
            
            for ii in range(len(age)):
                aa = len(age) -1 - ii
                if msize[aa]>0:
                    self.axes['ax4'].errorbar(age[aa], ZCp[aa,1], xerr=[[delTl[aa]/1e9],[delTu[aa]/1e9]], yerr=[[ZCp[aa,1]-ZCp[aa,0]],[ZCp[aa,2]-ZCp[aa,1]]], linestyle='-', color=self.col[aa], lw=1, zorder=1)
                    self.axes['ax4'].scatter(age[aa], ZCp[aa,1], marker='.', c=[self.col[aa]], edgecolor='k', s=msize[aa], zorder=2)

        # Attach to MB;
        MB.sfh_tlook = age
        MB.sfh_tlookl= delTl[:][conA]/1e9
        MB.sfh_tlooku= delTu[:][conA]/1e9
        MB.sfh_sfr16 = SFp[:,0]
        MB.sfh_sfr50 = SFp[:,1]
        MB.sfh_sfr84 = SFp[:,2]
        MB.sfh_mfr16 = ACp[:,0]
        MB.sfh_mfr50 = ACp[:,1]
        MB.sfh_mfr84 = ACp[:,2]
        MB.sfh_zfr16 = ZCp[:,0]
        MB.sfh_zfr50 = ZCp[:,1]
        MB.sfh_zfr84 = ZCp[:,2]

        ####
        #
        # percs = [16,50,84]
        # zmc = hdul[1].data['zmc']
        self.ACP = [ACp[0,0], ACp[0,1], ACp[0,2]]
        self.ACp = ACp
        self.ZCP = [ZCp[0,0], ZCp[0,1], ZCp[0,2]]
        self.ZCp = ZCp
        self.ZLP = self.ZCP #[ZLp[0,0], ZLp[0,1], ZLp[0,2]]
        # self.TLW = TTp[0,:]
        # self.TMW = TTp[0,:]
        # self.TAW = TCp[0,:]
        self.TC = TC
        self.TL = TL
        self.SFRs_SED = SFRs_SED
        # self.xSFp = xSFp
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
        self.update_axis_sfh(f_log_sfh=f_log_sfh, skip_zhist=skip_zhist, lsfrl=lsfrl)

        # Write files
        tree_sfh = self.save_files_sfh(tsets_SFR_SED=tsets_SFR_SED, taumodel=False)

        # Attach to self.mb;
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
    

    def open_result_file(self, DIR_TMP=''):
        ''''''
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
            self.SN = hdul[0].header['SN']
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
                self.SN = np.median((fsp/esp)[consp])
            else:
                self.SN = 1
        return 
    

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
        self.open_result_file(DIR_TMP=DIR_TMP,)

        file = self.mb.DIR_OUT + 'gsf_params_' + self.mb.ID + '.fits'
        hdul = fits.open(file) # open a FITS file
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
        # self.TLW = TTp[0,:]
        # self.TMW = TTp[0,:]
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
        self.update_axis_sfh(f_log_sfh=f_log_sfh, skip_zhist=skip_zhist, lsfrl=lsfrl)

        # Write files
        tree_sfh = self.save_files_sfh(tsets_SFR_SED=tsets_SFR_SED, taumodel=True)

        # Attach to self.mb;
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
    

    def save_files_sfh(self, tsets_SFR_SED=[], taumodel=False):
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
        if taumodel:
            for ii in range(len(percs)):
                prihdr['HIERARCH TAU_%d'%percs[ii]] = ('%.3f'%self.TAW[ii], 'Tau, logGyr')
        prihdu = fits.PrimaryHDU(header=prihdr)

        # For SFH plot;
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
        if not taumodel:
            arrays = [self.SFp[:,0],self.SFp[:,1],self.SFp[:,2],self.ACp[:,0],self.ACp[:,1],self.ACp[:,2], self.ZCp[:,0],self.ZCp[:,1],self.ZCp[:,2]]
            for arr in arrays:
                arr = np.nan
        else:
            mask_age = self.xSFp[:,1] > self.Tuni
            arrays = [self.SFp[:,0],self.SFp[:,1],self.SFp[:,2],self.ACp[:,0],self.ACp[:,1],self.ACp[:,2]]#
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

        # Attach to self.mb;
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


    def define_axis_sed(self, f_grsm=False, f_dust=False, f_plot_filter=True):
        """"""
        self.axes = {'ax1':None, 'ax2t':None, 'ax3t':None, 'axes':None}
        if f_grsm or f_dust:
            fig = plt.figure(figsize=(7.,3.2))
            fig.subplots_adjust(top=0.98, bottom=0.16, left=0.1, right=0.99, hspace=0.15, wspace=0.25)
            self.axes['ax1'] = fig.add_subplot(111)
            xsize = 0.29
            ysize = 0.25
            if f_grsm:
                self.axes['ax2t'] = self.axes['ax1'].inset_axes((1-xsize-0.01,1-ysize-0.01,xsize,ysize))
            if self.mb.f_dust:
                self.axes['ax3t'] = self.axes['ax1'].inset_axes((0.7,.35,.28,.25))
            self.f_plot_resid = False
            self.mb.logger.info('Grism data. f_plot_resid is turned off.')
        else:
            if self.f_plot_resid:
                fig_mosaic = """
                AAAA
                AAAA
                BBBB
                """
                fig,axes = plt.subplot_mosaic(mosaic=fig_mosaic, figsize=(5.5,4.))
                fig.subplots_adjust(top=0.98, bottom=0.16, left=0.08, right=0.99, hspace=0.15, wspace=0.25)
                self.axes['ax1'] = axes['A']
                self.axes['axes'] = axes
            else:
                if f_plot_filter:
                    fig = plt.figure(figsize=(5.5,2.))
                else:
                    fig = plt.figure(figsize=(5.5,1.8))
                fig.subplots_adjust(top=0.98, bottom=0.16, left=0.08, right=0.99, hspace=0.15, wspace=0.25)
                self.axes['ax1'] = fig.add_subplot(111)

        self.axes['fig'] = fig
        return self.axes


    def plot_sed_tau(self, flim=0.01, fil_path='./', scale=1e-19, f_chind=True, figpdf=False, save_sed=True, 
        mmax=300, dust_model=0, DIR_TMP='./templates/', f_label=False, f_bbbox=False, verbose=False, f_silence=True, 
        f_fill=False, f_fancyplot=False, f_Alog=True, dpi=300, f_plot_filter=True, f_plot_resid=False, NRbb_lim=10000,
        f_apply_igm=True,
        return_figure=False, percs=[16,50,84], 
        col_dat='r', col_dia='blue',
        lcb = '#4682b4', # line color, blue
        sigma=1.0, use_pickl=True,
        ):
        '''
        Parameters
        ----------
        self.mb.SNlim : float
            SN limit to show flux or up lim in SED.
        f_chind : bool
            If include non-detection in chi2 calculation, using Sawicki12.
        mmax : int
            Nuself.mber of mcmc realization for plot. Not for calculation.
        f_fancy : bool
            plot each SED component.
        f_fill : bool
            if True, and so is f_fancy, fill each SED component.

        Returns
        -------
        plots
        '''
        print('\n### Running plot_sed_tau ###\n')

        fnc  = self.mb.fnc
        age  = self.mb.age
        self.f_plot_resid = f_plot_resid
        
        nstep_plot = 1
        if self.mb.f_bpass:
            nstep_plot = 30

        SNlim = self.mb.SNlim

        ################
        # RF colors.
        c = self.mb.c
        m0set = self.mb.m0set
        Mpc_cm = self.mb.Mpc_cm
        
        ##################
        # Fitting Results
        ##################
        DIR_FILT = self.mb.DIR_FILT
        SFILT = self.mb.filts

        ###########################
        # Open result file
        ###########################
        file = self.mb.DIR_OUT + 'gsf_params_' + self.mb.ID + '.fits'
        hdul = fits.open(file) 
        
        ndim_eff = hdul[0].header['NDIM']
        vals = {}

        # Redshift MC
        zp16  = hdul[1].data['zmc'][0]
        zp50  = hdul[1].data['zmc'][1]
        zp84  = hdul[1].data['zmc'][2]
        vals['zmc'] = zp50

        # Stellar mass MC
        M16 = hdul[1].data['ms'][0]
        M50 = hdul[1].data['ms'][1]
        M84 = hdul[1].data['ms'][2]
        if verbose:
            print('Total stellar mass is %.2e'%(M50))

        # Amplitude MC
        A50 = np.zeros(len(age), dtype='float')
        A16 = np.zeros(len(age), dtype='float')
        A84 = np.zeros(len(age), dtype='float')
        for aa in range(len(age)):
            A16[aa] = 10**hdul[1].data['A'+str(aa)][0]
            A50[aa] = 10**hdul[1].data['A'+str(aa)][1]
            A84[aa] = 10**hdul[1].data['A'+str(aa)][2]
            vals['A'+str(aa)] = np.log10(A50[aa])

        Asum  = np.sum(A50)

        # TAU MC
        # AGE MC
        TAU50 = np.zeros(len(age), dtype='float')
        TAU16 = np.zeros(len(age), dtype='float')
        TAU84 = np.zeros(len(age), dtype='float')
        AGE50 = np.zeros(len(age), dtype='float')
        AGE16 = np.zeros(len(age), dtype='float')
        AGE84 = np.zeros(len(age), dtype='float')
        for aa in range(len(age)):
            TAU16[aa] = 10**hdul[1].data['TAU'+str(aa)][0]
            TAU50[aa] = 10**hdul[1].data['TAU'+str(aa)][1]
            TAU84[aa] = 10**hdul[1].data['TAU'+str(aa)][2]
            AGE16[aa] = 10**hdul[1].data['AGE'+str(aa)][0]
            AGE50[aa] = 10**hdul[1].data['AGE'+str(aa)][1]
            AGE84[aa] = 10**hdul[1].data['AGE'+str(aa)][2]
            vals['TAU'+str(aa)] = np.log10(TAU50[aa])
            vals['AGE'+str(aa)] = np.log10(AGE50[aa])

        if self.mb.fneb:
            logU50 = hdul[1].data['logU'][1]
            Aneb50 = 10**hdul[1].data['Aneb'][1]

        aa = 0
        Av16 = hdul[1].data['AV'+str(aa)][0]
        Av50 = hdul[1].data['AV'+str(aa)][1]
        Av84 = hdul[1].data['AV'+str(aa)][2]
        AAv = [Av50]
        vals['AV0'] = Av50

        Z50 = np.zeros(len(age), dtype='float')
        Z16 = np.zeros(len(age), dtype='float')
        Z84 = np.zeros(len(age), dtype='float')
        for aa in range(len(age)):
            Z16[aa] = hdul[1].data['Z'+str(aa)][0]
            Z50[aa] = hdul[1].data['Z'+str(aa)][1]
            Z84[aa] = hdul[1].data['Z'+str(aa)][2]
            vals['Z'+str(aa)] = Z50[aa]

        # Light weighted Z.
        ZZ50 = np.sum(Z50*A50)/np.sum(A50)

        # FIR Dust;
        if self.mb.f_dust:
            MD16 = hdul[1].data['MDUST'][0]
            MD50 = hdul[1].data['MDUST'][1]
            MD84 = hdul[1].data['MDUST'][2]
            AD16 = hdul[1].data['ADUST'][0]
            AD50 = hdul[1].data['ADUST'][1]
            AD84 = hdul[1].data['ADUST'][2]
            TD16 = hdul[1].data['TDUST'][0]
            TD50 = hdul[1].data['TDUST'][1]
            TD84 = hdul[1].data['TDUST'][2]
            nTD16 = hdul[1].data['nTDUST'][0]
            nTD50 = hdul[1].data['nTDUST'][1]
            nTD84 = hdul[1].data['nTDUST'][2]
            DFILT = self.mb.inputs['FIR_FILTER'] # filter band string.
            DFILT = [x.strip() for x in DFILT.split(',')]
            # DFWFILT = fil_fwhm(DFILT, DIR_FILT)
            if verbose:
                print('Total dust mass is %.2e'%(MD50))
            f_dust = True
        else:
            DFILT = []
            f_dust = False

        Cz0 = hdul[0].header['Cz0']
        Cz1 = hdul[0].header['Cz1']
        Cz2 = hdul[0].header['Cz2']
        zbes = zp50 
        zscl = (1.+zbes)

        ###############################
        # Data taken from
        ###############################
        if self.mb.f_dust:
            self.mb.dict = self.mb.read_data(Cz0, Cz1, Cz2, zbes, add_fir=True)
        else:
            self.mb.dict = self.mb.read_data(Cz0, Cz1, Cz2, zbes)

        NR   = self.mb.dict['NR']
        x    = self.mb.dict['x']
        fy   = self.mb.dict['fy']
        ey   = self.mb.dict['ey']
        
        con0 = (NR<1000)
        xg0  = x[con0]
        fg0  = fy[con0]
        eg0  = ey[con0]
        con1 = (NR>=1000) & (NR<2000) #& (fy/ey>SNlim)
        xg1  = x[con1]
        fg1  = fy[con1]
        eg1  = ey[con1]
        con2 = (NR>=2000) & (NR<NRbb_lim) #& (fy/ey>SNlim)
        xg2  = x[con2]
        fg2  = fy[con2]
        eg2  = ey[con2]
        if len(xg0)>0 or len(xg1)>0 or len(xg2)>0:
            f_grsm = True
        else:
            f_grsm = False

        # Weight is set to zero for those no data (ey<0).
        wht = fy * 0
        con_wht = (ey>0)
        wht[con_wht] = 1./np.square(ey[con_wht])

        # BB data points;
        NRbb = self.mb.dict['NRbb']
        xbb  = self.mb.dict['xbb']
        fybb = self.mb.dict['fybb']
        eybb = self.mb.dict['eybb']
        exbb = self.mb.dict['exbb']
        snbb = fybb/eybb

        ######################
        # Weight by line
        ######################
        wh0  = 1./np.square(eg0)
        LW0  = []
        model = fg0
        wht3 = check_line_man(fy, x, wht, fy, zbes, LW0)

        ######################
        # Mass-to-Light ratio.
        ######################
        af = self.mb.af
        sedpar = af['ML']
        try:
            isochrone = af['isochrone']
            LIBRARY = af['library']
            nimf = af['nimf']
        except:
            isochrone = ''
            LIBRARY = ''
            nimf = ''

        # Initiate figure;
        _ = self.define_axis_sed(f_grsm=f_grsm, f_dust=self.mb.f_dust, f_plot_filter=f_plot_filter)

        # Determine scale here;
        if scale == None:
            conbb_hs = (fybb/eybb > SNlim)
            if len(fybb[conbb_hs])>0:
                scale = 10**(int(np.log10(np.nanmax(fybb[conbb_hs] * c / np.square(xbb[conbb_hs])) / self.mb.d))) / 10
            else:
                scale = 1e-19
                self.mb.logger.info('no data point has SN > %.1f. Setting scale to %.1e'%(SNlim, scale))
        d_scale = self.mb.d * scale

        # Plot BB data points;
        _, leng = self.plot_bb_sed(xbb, fybb, eybb, exbb, NRbb, d_scale, SNlim, c=c, col_dat='r', f_bbbox=f_bbbox, sigma = sigma)

        # For any data removed fron fit (i.e. IRAC excess):
        f_exclude = False
        col_ex = 'lawngreen'
        x_ex = []
        fy_ex = []
        ey_ex = []
        try:
            #col_ex = 'limegreen'
            #col_ex = 'r'
            # Currently, this file is made after FILTER_SKIP;
            data_ex = ascii.read(DIR_TMP + 'bb_obs_' + self.mb.ID + '_removed.cat')
            x_ex = data_ex['col2']
            fy_ex = data_ex['col3']
            ey_ex = data_ex['col4']
            ex_ex = data_ex['col5']

            self.axes['ax1'].errorbar(x_ex, fy_ex * c / np.square(x_ex) /d_scale, \
            xerr=ex_ex, yerr=ey_ex*c/np.square(x_ex)/d_scale, color='k', linestyle='', linewidth=0.5, zorder=5)
            self.axes['ax1'].scatter(x_ex, fy_ex * c / np.square(x_ex) /d_scale, marker='s', color=col_ex, edgecolor='k', zorder=5, s=30)
            f_exclude = True
        except:
            pass

        #####################################
        # Open ascii file and stock to array.
        _ = self.setup_sed_library()

        if self.mb.f_dust:
            _, _, y0d_cut, y0d, x0d = self.plot_dust_sed(AD50, nTD50, zp50, c=c, d_scale=d_scale, SNlim=SNlim)

        #
        # This is for UVJ color time evolution.
        #
        Asum = np.sum(A50[:])
        # alp = .5

        # Get total templates
        y0p, _ = self.mb.fnc.get_template(vals, f_val=False, check_bound=False)
        y0, x0 = self.mb.fnc.get_template(vals, f_val=False, check_bound=False, lib_all=True)

        ysum = y0
        f_50_comp = y0[:] * c / np.square(x0) /d_scale

        ysump = y0p
        nopt = len(ysump)

        if f_dust:
            ysump[:] += y0d_cut[:nopt]
            ysump = np.append(ysump,y0d_cut[nopt:])
            f_50_comp_dust = y0d * c / np.square(x0d) /d_scale

        if self.mb.fneb: 
            # Only at one age pixel;
            y0p, x0p = self.mb.fnc.get_template(vals, f_val=False, check_bound=False, f_neb=True)
            y0_r, x0_tmp = self.mb.fnc.get_template(vals, f_val=False, check_bound=False, f_neb=True, lib_all=True)
            ysum += y0_r
            ysump[:nopt] += y0p

        # Plot each best fit:
        vals_each = vals.copy()
        for aa in range(len(age)):
            vals_each['A%d'%aa] = -99
        for aa in range(len(age)):
            vals_each['A%d'%aa] = vals['A%d'%aa]
            y0tmp, x0tmp = self.mb.fnc.get_template(vals_each, f_val=False, check_bound=False, lib_all=True)
            if aa == 0:
                y0keep = y0tmp
            else:
                y0keep += y0tmp
            self.axes['ax1'].plot(x0tmp, y0tmp * c / np.square(x0tmp) /d_scale, linestyle='--', lw=0.5, color=self.col[aa])
            vals_each['A%d'%aa] = 0

        # Plot best fit;
        self.axes['ax1'].plot(x0, f_50_comp, linestyle='-', lw=0.5, color='k')

        #############
        # Main result
        #############
        if self.mb.has_photometry:
            conbb_ymax = (xbb>0) & (fybb>0) & (eybb>0) & (fybb/eybb>SNlim)
            if len(fybb[conbb_ymax]):
                ymax = np.nanmax(fybb[conbb_ymax]*c/np.square(xbb[conbb_ymax])/d_scale) * 1.6
            else:
                ymax = np.nanmax(fybb*c/np.square(xbb)/d_scale) * 1.6
        else:
            ymax = None

        x1max = 100000
        if self.mb.has_photometry:
            if x1max < np.nanmax(xbb):
                x1max = np.nanmax(xbb) * 1.5
            if len(fybb[conbb_ymax]):
                if x1min > np.nanmin(xbb[conbb_ymax]):
                    x1min = np.nanmin(xbb[conbb_ymax]) / 1.5
        else:
            x1min = 2000

        #############
        # Plot
        #############
        ms = np.zeros(len(age), dtype='float')
        af = self.mb.af
        sedpar = af['ML']

        eAAl = np.zeros(len(age),dtype='float')
        eAAu = np.zeros(len(age),dtype='float')
        eAMl = np.zeros(len(age),dtype='float')
        eAMu = np.zeros(len(age),dtype='float')
        MSsum = np.sum(ms)
        Asum = np.sum(A50)
        A50 /= Asum
        A16 /= Asum
        A84 /= Asum

        AM50 = A50 * M50 * ms / MSsum
        CM = M50/np.sum(AM50)

        AM50 = A50 * M50 * ms / MSsum * CM
        AM16 = A16 * M50 * ms / MSsum * CM
        AM84 = A84 * M50 * ms / MSsum * CM

        AC50 = A50 * 0 # Cumulative
        for ii in range(len(A50)):
            eAAl[ii] = A50[ii] - A16[ii]
            eAAu[ii] = A84[ii] - A50[ii]
            eAMl[ii] = AM50[ii] - AM16[ii]
            eAMu[ii] = AM84[ii] - AM50[ii]
            AC50[ii] = np.sum(AM50[ii:])

        ####################
        # For cosmology
        ####################
        DL = self.mb.cosmo.luminosity_distance(zbes).value * Mpc_cm

        if f_grsm:
            print('This function (write_lines) needs to be revised.')
            PLOT.write_lines(self.mb.ID, zbes, DIR_OUT=self.mb.DIR_OUT)

        ##########################
        # Zoom in Line regions
        ##########################
        if f_grsm:
            conspec = (NR<10000) #& (fy/ey>1)
            self.axes['ax2t'].errorbar(xg2, fg2 * c/np.square(xg2)/d_scale, yerr=eg2 * c/np.square(xg2)/d_scale, lw=0.5, color='#DF4E00', zorder=10, alpha=1., label='', capsize=0)
            self.axes['ax2t'].errorbar(xg1, fg1 * c/np.square(xg1)/d_scale, yerr=eg1 * c/np.square(xg1)/d_scale, lw=0.5, color='g', zorder=10, alpha=1., label='', capsize=0)
            self.axes['ax2t'].errorbar(xg0, fg0 * c/np.square(xg0)/d_scale, yerr=eg0 * c/np.square(xg0)/d_scale, lw=0.5, linestyle='', color='royalblue', zorder=10, alpha=1., label='', capsize=0)

            xgrism = np.concatenate([xg0,xg1,xg2])
            fgrism = np.concatenate([fg0,fg1,fg2])
            egrism = np.concatenate([eg0,eg1,eg2])
            con4000b = (xgrism/zscl>3400) & (xgrism/zscl<3800) & (fgrism>0) & (egrism>0)
            con4000r = (xgrism/zscl>4200) & (xgrism/zscl<5000) & (fgrism>0) & (egrism>0)
            print('Median SN at 3400-3800 is;', np.median((fgrism/egrism)[con4000b]))
            print('Median SN at 4200-5000 is;', np.median((fgrism/egrism)[con4000r]))

            # TEST;
            self.axes['ax1'].errorbar(xg2, fg2 * c/np.square(xg2)/d_scale, yerr=eg2 * c/np.square(xg2)/d_scale, lw=0.5, color='#DF4E00', zorder=10, alpha=1., label='', capsize=0)
            self.axes['ax1'].errorbar(xg1, fg1 * c/np.square(xg1)/d_scale, yerr=eg1 * c/np.square(xg1)/d_scale, lw=0.5, color='g', zorder=10, alpha=1., label='', capsize=0)
            self.axes['ax1'].errorbar(xg0, fg0 * c/np.square(xg0)/d_scale, yerr=eg0 * c/np.square(xg0)/d_scale, lw=0.5, linestyle='', color='royalblue', zorder=10, alpha=1., label='', capsize=0)

        #
        # From MCMC chain
        #
        ndim, Nburn, samples = PLOT.read_mcmc_chain(self.mb.ID, samplepath=self.mb.DIR_OUT, use_pickl=use_pickl)

        # Saved template;
        ytmp = np.zeros((mmax,len(ysum)), dtype='float')
        ytmp_each = np.zeros((mmax,len(ysum),len(age)), dtype='float')

        # MUV;
        DL      = self.mb.cosmo.luminosity_distance(zbes).value * Mpc_cm # Luminositydistance in cm
        DL10    = Mpc_cm/1e6 * 10 # 10pc in cm
        Fuv     = np.zeros(mmax, dtype='float') # For Muv
        Fuv2800   = np.zeros(mmax, dtype='float') # For Fuv(1500-2800)
        Lir     = np.zeros(mmax, dtype='float') # For L(8-1000um)
        UVJ     = np.zeros((mmax,4), dtype='float') # For UVJ color;
        Cmznu   = 10**((48.6+m0set)/(-2.5)) # Conversion from m0_25 to fnu

        # From random chain;
        for kk in range(0,mmax,1):
            nr = np.random.randint(Nburn, len(samples['A%d'%self.mb.aamin[0]]))

            if self.mb.fxhi:
                xhi = samples['xhi'][nr]
            else:
                xhi = self.mb.x_HI_input

            try:
                Av_tmp = samples['AV0'][nr]
            except:
                Av_tmp = self.mb.AVFIX
            vals['AV0'] = Av_tmp

            try:
                zmc = samples['zmc'][nr]
            except:
                zmc = zbes
            vals['zmc'] = zmc

            # Tau model;
            vals['TAU0'] = samples['TAU0'][nr]
            vals['AGE0'] = samples['AGE0'][nr]

            for ss in self.mb.aamin:
                try:
                    AA_tmp = 10**samples['A'+str(ss)][nr]
                except:
                    AA_tmp = 0
                vals['A%d'%ss] = np.log10(AA_tmp)

                if ss == 0 or self.mb.ZEVOL:
                    try:
                        ZZtmp = samples['Z%d'%ss][nr]
                    except:
                        ZZtmp = self.mb.ZFIX
                    vals['Z%d'%ss] = ZZtmp

            mod0_tmp, xm_tmp = self.mb.fnc.get_template(vals, f_val=False, check_bound=False, lib_all=True)
            fm_tmp = mod0_tmp

            # @@@ Something is wrong here, or wrong with the nebular templates.
            if self.mb.fneb:
                Aneb_tmp = 10**samples['Aneb'][nr]
                if not self.mb.logUFIX == None:
                    logU_tmp = self.mb.logUFIX
                else:
                    logU_tmp = samples['logU'][nr]

                mod0_tmp, _ = fnc.get_template_single(Aneb_tmp, Av_tmp, vals['TAU0'], vals['AGE0'], ZZtmp, zmc, self.mb.lib_neb_all, logU=logU_tmp, f_apply_igm=f_apply_igm, xhi=xhi)

                fm_tmp += mod0_tmp
                # # Make no emission line template;
                # mod0_tmp_nl, xm_tmp_nl = fnc.get_template(0, Av_tmp, ss, ZZ_tmp, zmc, lib_neb_all, logU=logU_tmp)
                # fm_tmp_nl += mod0_tmp_nl

            if False:
                # Each;
                ytmp_each[kk,:,ss] = mod0_tmp[:] * c / np.square(xm_tmp[:]) /d_scale
                #if kk == 100:
                #    self.axes['ax1'].plot(xm_tmp[:], ytmp_each[kk,:,ss], color=col[ss], linestyle='--')

            #
            # Dust component;
            #
            if f_dust:
                if kk == 0:
                    par = Parameters()
                    par.add('MDUST',value=samples['MDUST'][nr])
                    try:
                        par.add('TDUST',value=samples['TDUST'][nr])
                    except:
                        par.add('TDUST',value=0)

                par['MDUST'].value = samples['MDUST'][nr]
                if not self.mb.TDUSTFIX == None:
                    par['TDUST'].value = self.mb.NTDUST
                else:
                    par['TDUST'].value = samples['TDUST'][nr]

                model_dust, x1_dust = fnc.tmp04_dust(par.valuesdict())#, zbes, lib_dust_all)
                if kk == 0:
                    deldt  = (x1_dust[1] - x1_dust[0])
                    x1_tot = np.append(xm_tmp,np.arange(np.max(xm_tmp),np.max(x1_dust),deldt))
                    # Redefine??
                    ytmp = np.zeros((mmax,len(x1_tot)), dtype='float')
                    ytmp_dust = np.zeros((mmax,len(x1_dust)), dtype='float')

                ytmp_dust[kk,:] = model_dust * c/np.square(x1_dust)/d_scale
                model_tot = np.interp(x1_tot,xm_tmp,fm_tmp) + np.interp(x1_tot,x1_dust,model_dust)

                ytmp[kk,:] = model_tot[:] * c/np.square(x1_tot[:])/d_scale

            else:
                x1_tot = xm_tmp
                ytmp[kk,:] = fm_tmp[:] * c / np.square(xm_tmp[:]) /d_scale

            # plot random sed;
            plot_mc = True
            if plot_mc:
                self.axes['ax1'].plot(x1_tot, ytmp[kk,:], '-', lw=1, color='gray', zorder=-2, alpha=0.02)

            # Grism plot + Fuv flux + LIR.
            #if f_grsm:
            #    self.axes['ax2t'].plot(x1_tot, ytmp[kk,:], '-', lw=0.5, color='gray', zorder=3., alpha=0.02)

            if True:
                # Get FUV flux;
                Fuv[kk] = get_Fuv(x1_tot[:]/(1.+zbes), (ytmp[kk,:]/(c/np.square(x1_tot)/d_scale)) * (DL**2/(1.+zbes)) / (DL10**2), lmin=1250, lmax=1650)
                Fuv2800[kk] = get_Fuv(x1_tot[:]/(1.+zbes), (ytmp[kk,:]/(c/np.square(x1_tot)/d_scale)) * (4*np.pi*DL**2/(1.+zbes))*Cmznu, lmin=1500, lmax=2800)
                Lir[kk] = 0

                # Get UVJ Color;
                lmconv,fconv = filconv_fast(self.mb.filts_rf, self.mb.band_rf, x1_tot[:]/(1.+zbes), (ytmp[kk,:]/(c/np.square(x1_tot)/d_scale)))
                UVJ[kk,0] = -2.5*np.log10(fconv[0]/fconv[2])
                UVJ[kk,1] = -2.5*np.log10(fconv[1]/fconv[2])
                UVJ[kk,2] = -2.5*np.log10(fconv[2]/fconv[3])
                UVJ[kk,3] = -2.5*np.log10(fconv[4]/fconv[3])

            # Do stuff...
            # time.sleep(0.01)
            # Update Progress Bar
            printProgressBar(kk, mmax, prefix = 'Progress:', suffix = 'Complete', length = 40)

        print('')

        #
        # Plot Median SED;
        #
        ytmp16 = np.nanpercentile(ytmp[:,:],16,axis=0)
        ytmp50 = np.nanpercentile(ytmp[:,:],50,axis=0)
        ytmp84 = np.nanpercentile(ytmp[:,:],84,axis=0)
        
        if f_dust:
            ytmp_dust50 = np.nanpercentile(ytmp_dust[:,:],50, axis=0)

        # For grism;
        if f_grsm:
            LSF = get_LSF(self.mb.inputs, self.mb.DIR_EXTR, self.mb.ID, x1_tot[:]/(1.+zbes), c=3e18)
            try:
                spec_grsm16 = convolve(ytmp16[:], LSF, boundary='extend')
                spec_grsm50 = convolve(ytmp50[:], LSF, boundary='extend')
                spec_grsm84 = convolve(ytmp84[:], LSF, boundary='extend')
            except:
                spec_grsm16 = ytmp16[:]
                spec_grsm50 = ytmp50[:]
                spec_grsm84 = ytmp84[:]

            if True:
                self.axes['ax2t'].plot(x1_tot[:], ytmp50, '-', lw=0.5, color='gray', zorder=3., alpha=1.0)
            else:
                self.axes['ax2t'].plot(x1_tot[:], spec_grsm50, '-', lw=0.5, color='gray', zorder=3., alpha=1.0)

        #if not f_fill:
        self.axes['ax1'].fill_between(x1_tot[::nstep_plot], ytmp16[::nstep_plot], ytmp84[::nstep_plot], ls='-', lw=.5, color='gray', zorder=-2, alpha=0.5)
        self.axes['ax1'].plot(x1_tot[::nstep_plot], ytmp50[::nstep_plot], '-', lw=.5, color='gray', zorder=-1, alpha=1.)

        # Attach the data point in self.mb;
        self.mb.sed_wave_obs = xbb
        self.mb.sed_flux_obs = fybb * c / np.square(xbb) /d_scale
        self.mb.sed_eflux_obs = eybb * c / np.square(xbb) /d_scale
        # Attach the best SED to self.mb;
        self.mb.sed_wave = x1_tot
        self.mb.sed_flux16 = ytmp16
        self.mb.sed_flux50 = ytmp50
        self.mb.sed_flux84 = ytmp84

        # Calculate non-det chi2
        chi2, conw, con_up, chi_nd, nod, fin_chi2 = PLOT.show_chi2(hdul, fy, ey, ysump, x, wht3, ndim_eff, x_ex=x_ex, SNlim=SNlim, f_chind=f_chind, f_exclude=f_exclude)

        # plot BB model from best template (blue squares)
        lbb, fbb, fbb16, fbb84, ew_label, EW16, EW50, EW84, EW50_er1, EW50_er2, cnt16, cnt50, cnt84, L16, L50, L84 = self.plot_bbmodel_sed(zbes, xbb, fybb, eybb, x_ex, fy_ex, ey_ex, x1_tot, ytmp16, ytmp50, ytmp84, 
                                                                                                                                           SFILT, DFILT, DIR_FILT, scale, d_scale, DL, leng, sigma, 
                                                                                                                                           SNlim=SNlim, c=c, col_dat=col_dat, col_dia=col_dat)

        if save_sed:
            fbb16_nu = flamtonu(lbb, fbb16*scale, m0set=m0set)
            fbb_nu = flamtonu(lbb, fbb*scale, m0set=m0set)
            fbb84_nu = flamtonu(lbb, fbb84*scale, m0set=m0set)

            # Then save full spectrum;
            col00  = []
            col1  = fits.Column(name='wave_model', format='E', unit='AA', array=x1_tot)
            col00.append(col1)
            col2  = fits.Column(name='f_model_16', format='E', unit='1e%derg/s/cm2/AA'%(np.log10(scale)), array=ytmp16[:])
            col00.append(col2)
            col3  = fits.Column(name='f_model_50', format='E', unit='1e%derg/s/cm2/AA'%(np.log10(scale)), array=ytmp50[:])
            col00.append(col3)
            col4  = fits.Column(name='f_model_84', format='E', unit='1e%derg/s/cm2/AA'%(np.log10(scale)), array=ytmp84[:])
            col00.append(col4)

            f_sed_each = False
            if f_sed_each:
                # Each component
                # Stellar
                col1 = fits.Column(name='wave_model_stel', format='E', unit='AA', array=x0)
                col00.append(col1)
                for aa in range(len(age)):
                    col1 = fits.Column(name='f_model_stel_%d'%aa, format='E', unit='1e%derg/s/cm2/AA'%(np.log10(scale)), array=f_50_comp[aa,:])
                    col00.append(col1)
                if f_dust:
                    col1 = fits.Column(name='wave_model_dust', format='E', unit='AA', array=x1_dust)
                    col00.append(col1)
                    col1 = fits.Column(name='f_model_dust', format='E', unit='1e%derg/s/cm2/AA'%(np.log10(scale)), array=ytmp_dust50)
                    col00.append(col1)
                
            # BB for dust
            if f_dust:
                xbb = np.append(xbb,xbbd)
                fybb = np.append(fybb,fybbd)
                eybb = np.append(eybb,eybbd)

            col5  = fits.Column(name='wave_obs', format='E', unit='AA', array=xbb)
            col00.append(col5)
            col6  = fits.Column(name='f_obs', format='E', unit='1e%derg/s/cm2/AA'%(np.log10(scale)), array=fybb[:] * c / np.square(xbb[:]) /d_scale)
            col00.append(col6)
            col7  = fits.Column(name='e_obs', format='E', unit='1e%derg/s/cm2/AA'%(np.log10(scale)), array=eybb[:] * c / np.square(xbb[:]) /d_scale)
            col00.append(col7)

            hdr = fits.Header()
            hdr['redshift'] = zbes
            hdr['id'] = self.mb.ID
            hdr['hierarch isochrone'] = isochrone
            hdr['library'] = LIBRARY
            hdr['nimf'] = nimf
            hdr['scale'] = scale

            try:
                # Chi square:
                hdr['chi2'] = chi2
                hdr['hierarch No-of-effective-data-points'] = len(wht3[conw])
                hdr['hierarch No-of-nondetectioin'] = len(ey[con_up])
                hdr['hierarch Chi2-of-nondetection'] = chi_nd
                hdr['hierarch No-of-params'] = ndim_eff
                hdr['hierarch Degree-of-freedom']  = nod
                hdr['hierarch reduced-chi2'] = fin_chi2
            except:
                print('Chi seems to be wrong...')
                pass

            try:
                # Muv
                MUV = -2.5 * np.log10(Fuv[:]) + self.mb.m0set
                hdr['MUV16'] = -2.5 * np.log10(np.nanpercentile(Fuv[:],16)) + self.mb.m0set
                hdr['MUV50'] = -2.5 * np.log10(np.nanpercentile(Fuv[:],50)) + self.mb.m0set
                hdr['MUV84'] = -2.5 * np.log10(np.nanpercentile(Fuv[:],84)) + self.mb.m0set

                # Fuv (!= flux of Muv)
                hdr['FUV16'] = np.nanpercentile(Fuv2800[:],16)
                hdr['FUV50'] = np.nanpercentile(Fuv2800[:],50)
                hdr['FUV84'] = np.nanpercentile(Fuv2800[:],84)

                # LIR
                hdr['LIR16'] = np.nanpercentile(Lir[:],16)
                hdr['LIR50'] = np.nanpercentile(Lir[:],50)
                hdr['LIR84'] = np.nanpercentile(Lir[:],84)
            except:
                pass

            # UV beta;
            beta_16 = get_uvbeta(x1_tot, ytmp16, zbes)
            beta_50 = get_uvbeta(x1_tot, ytmp50, zbes)
            beta_84 = get_uvbeta(x1_tot, ytmp84, zbes)
            hdr['UVBETA16'] = beta_16
            hdr['UVBETA50'] = beta_50
            hdr['UVBETA84'] = beta_84

            # UVJ
            try:
                hdr['uv16'] = np.nanpercentile(UVJ[:,0],16)
                hdr['uv50'] = np.nanpercentile(UVJ[:,0],50)
                hdr['uv84'] = np.nanpercentile(UVJ[:,0],84)
                hdr['bv16'] = np.nanpercentile(UVJ[:,1],16)
                hdr['bv50'] = np.nanpercentile(UVJ[:,1],50)
                hdr['bv84'] = np.nanpercentile(UVJ[:,1],84)
                hdr['vj16'] = np.nanpercentile(UVJ[:,2],16)
                hdr['vj50'] = np.nanpercentile(UVJ[:,2],50)
                hdr['vj84'] = np.nanpercentile(UVJ[:,2],84)
                hdr['zj16'] = np.nanpercentile(UVJ[:,3],16)
                hdr['zj50'] = np.nanpercentile(UVJ[:,3],50)
                hdr['zj84'] = np.nanpercentile(UVJ[:,3],84)
            except:
                print('\nError when writinf UVJ colors;\n')
                pass

            # EW;
            try:
                for ii in range(len(EW50)):
                    hdr['EW_%s_16'%(ew_label[ii])] = EW16[ii]
                    hdr['EW_%s_50'%(ew_label[ii])] = EW50[ii]
                    hdr['EW_%s_84'%(ew_label[ii])] = EW84[ii]
                    hdr['EW_%s_e1'%(ew_label[ii])] = EW50_er1[ii]
                    hdr['EW_%s_e2'%(ew_label[ii])] = EW50_er2[ii]
                    hdr['HIERARCH cnt_%s_16'%(ew_label[ii])]= cnt16[ii]
                    hdr['HIERARCH cnt_%s_50'%(ew_label[ii])]= cnt50[ii]
                    hdr['HIERARCH cnt_%s_84'%(ew_label[ii])]= cnt84[ii]
                    hdr['L_%s_16'%(ew_label[ii])] = L16[ii]
                    hdr['L_%s_50'%(ew_label[ii])] = L50[ii]
                    hdr['L_%s_84'%(ew_label[ii])] = L84[ii]
            except:
                pass

            # Version;
            import gsf
            from astropy import units as u
            hdr['version'] = gsf.__version__

            # Write;
            colspec = fits.ColDefs(col00)
            hdu0 = fits.BinTableHDU.from_columns(colspec, header=hdr)
            hdu0.writeto(self.mb.DIR_OUT + 'gsf_spec_%s.fits'%(self.mb.ID), overwrite=True)

            # ASDF;
            tree_spec = {
                'id': self.mb.ID,
                'redshift': '%.3f'%zbes,
                'isochrone': '%s'%(isochrone),
                'library': '%s'%(LIBRARY),
                'nimf': '%s'%(nimf),
                'scale': scale,
                'version_gsf': gsf.__version__
            }
            tree_spec['model'] = {}
            tree_spec['obs'] = {}
            tree_spec['header'] = {}

            # Dump physical parameters;
            for key in hdr:
                if key not in tree_spec:
                    if key[:-3] == 'SFRUV':
                        tree_spec['header'].update({'%s'%key: hdr[key] * u.solMass / u.yr})
                    else:
                        tree_spec['header'].update({'%s'%key: hdr[key]})

            # BB;
            tree_spec.update({'wave': lbb})
            tree_spec.update({'fnu_16': fbb16_nu})
            tree_spec.update({'fnu_50': fbb_nu})
            tree_spec.update({'fnu_84': fbb84_nu})
            # full spectrum;
            tree_spec.update({'wave_model': x1_tot})
            tree_spec.update({'f_model_16': ytmp16})
            tree_spec.update({'f_model_50': ytmp50})
            tree_spec.update({'f_model_84': ytmp84})

            # EW;
            try:
                for ii in range(len(EW50)):
                    tree_spec.update({'EW_%s_16'%(ew_label[ii]): EW16[ii]})
                    tree_spec.update({'EW_%s_50'%(ew_label[ii]): EW50[ii]})
                    tree_spec.update({'EW_%s_84'%(ew_label[ii]): EW84[ii]})
                    tree_spec.update({'EW_%s_e1'%(ew_label[ii]): EW50_er1[ii]})
                    tree_spec.update({'EW_%s_e2'%(ew_label[ii]): EW50_er2[ii]})
                    tree_spec.update({'cnt_%s_16'%(ew_label[ii]): cnt16[ii]})
                    tree_spec.update({'cnt_%s_50'%(ew_label[ii]): cnt50[ii]})
                    tree_spec.update({'cnt_%s_84'%(ew_label[ii]): cnt84[ii]})
                    tree_spec.update({'L_%s_16'%(ew_label[ii]): L16[ii]})
                    tree_spec.update({'L_%s_50'%(ew_label[ii]): L50[ii]})
                    tree_spec.update({'L_%s_84'%(ew_label[ii]): L84[ii]})
            except:
                pass

            # Each component
            # Stellar
            tree_spec.update({'wave_model_stel': x0})

            if f_sed_each:
                for aa in range(len(age)):
                    tree_spec.update({'f_model_stel_%d'%aa: f_50_comp[aa,:]})

            if f_dust:
                # dust
                tree_spec.update({'wave_model_dust': x1_dust})
                tree_spec.update({'f_model_dust': ytmp_dust50})            
            # BB for dust
            tree_spec.update({'wave_obs': xbb})
            tree_spec.update({'f_obs': fybb[:] * c / np.square(xbb[:]) /d_scale})
            tree_spec.update({'e_obs': eybb[:] * c / np.square(xbb[:]) /d_scale})
            # grism:
            if f_grsm:
                tree_spec.update({'fg0_obs': fg0 * c/np.square(xg0)/d_scale})
                tree_spec.update({'eg0_obs': eg0 * c/np.square(xg0)/d_scale})
                tree_spec.update({'wg0_obs': xg0})
                tree_spec.update({'fg1_obs': fg1 * c/np.square(xg1)/d_scale})
                tree_spec.update({'eg1_obs': eg1 * c/np.square(xg1)/d_scale})
                tree_spec.update({'wg1_obs': xg1})
                tree_spec.update({'fg2_obs': fg2 * c/np.square(xg2)/d_scale})
                tree_spec.update({'eg2_obs': eg2 * c/np.square(xg2)/d_scale})
                tree_spec.update({'wg2_obs': xg2})

            af = asdf.AsdfFile(tree_spec)
            af.write_to(self.mb.DIR_OUT + 'gsf_spec_%s.asdf'%(self.mb.ID), all_array_compression='zlib')

        # Make a new dict
        gsf_dict = self.save_files_sed(percs=percs)

        #
        # SED params in plot
        #
        if f_label:
            fs_label = 8
            fd = fits.open(self.mb.DIR_OUT + 'SFH_' + self.mb.ID + '.fits')[0].header
            if f_dust:
                label = 'ID: %s\n$z:%.2f$\n$\log M_*/M_\odot:%.2f$\n$\log M_\mathrm{dust}/M_\odot:%.2f$\n$T_\mathrm{dust}/K:%.1f$\n$\log Z_*/Z_\odot:%.2f$\n$\log T_0$/Gyr$:%.2f$\n$\log \\tau$/Gyr$:%.2f$\n$A_V$/mag$:%.2f$\n$\\chi^2/\\nu:%.2f$'\
                %(self.mb.ID, zbes, float(fd['Mstel_50']), MD50, TD50, float(fd['Z_MW_50']), float(fd['T_MW_50']), float(fd['TAU_50']), float(fd['AV0_50']), fin_chi2)
            else:
                label = 'ID: %s\n$z:%.2f$\n$\log M_*/M_\odot:%.2f$\n$\log Z_*/Z_\odot:%.2f$\n$\log T_0$/Gyr$:%.2f$\n$\log \\tau$/Gyr$:%.2f$\n$A_V$/mag$:%.2f$\n$\\chi^2/\\nu:%.2f$'\
                %(self.mb.ID, zbes, float(fd['Mstel_50']), float(fd['Z_MW_50']), float(fd['T_MW_50']), float(fd['TAU_50']), float(fd['AV0_50']), fin_chi2)

            if f_grsm:
                self.axes['ax1'].text(0.02, 0.68, label,\
                fontsize=fs_label, bbox=dict(facecolor='w', alpha=0.8, lw=1.), zorder=10,
                ha='left', va='center', transform=self.axes['ax1'].transAxes)
            else:
                self.axes['ax1'].text(0.02, 0.68, label,\
                fontsize=fs_label, bbox=dict(facecolor='w', alpha=0.8, lw=1.), zorder=10,
                ha='left', va='center', transform=self.axes['ax1'].transAxes)
            
        #######################################
        if f_grsm:
            conlim = (x0>10000) & (x0<25000)
            xgmin, xgmax = np.min(x0[conlim]),np.max(x0[conlim]), #7500, 17000
            self.axes['ax2t'].set_xlabel('')
            self.axes['ax2t'].set_xlim(xgmin, xgmax)

            conaa = (x0>xgmin-50) & (x0<xgmax+50)
            ymaxzoom = np.max(ysum[conaa]*c/np.square(x0[conaa])/d_scale) * 1.15
            yminzoom = np.min(ysum[conaa]*c/np.square(x0[conaa])/d_scale) / 1.15

            self.axes['ax2t'].set_ylim(yminzoom, ymaxzoom)
            self.axes['ax2t'].xaxis.labelpad = -2
            if xgmax>20000:
                self.axes['ax2t'].set_xticks([8000, 12000, 16000, 20000, 24000])
                self.axes['ax2t'].set_xticklabels(['0.8', '1.2', '1.6', '2.0', '2.4'])
            else:
                self.axes['ax2t'].set_xticks([8000, 10000, 12000, 14000, 16000])
                self.axes['ax2t'].set_xticklabels(['0.8', '1.0', '1.2', '1.4', '1.6'])

        if f_dust:
            try:
                contmp = (x1_tot>10*1e4)
                y3min, y3max = -.2*np.max((model_tot * c/ np.square(x1_tot) /d_scale)[contmp]), np.max((model_tot * c/ np.square(x1_tot) /d_scale)[contmp])*2.0
                self.axes['ax3t'].set_ylim(y3min, y3max)
            except:
                if verbose:
                    print('y3 limit is not specified.')
                pass

        # Figure configure;
        ax1 = self.update_axis_sed(x1min, x1max, ymax, scale, ey, wht3, f_plot_filter=False)

        ####################
        ## Save
        ####################
        if figpdf:
            self.axes['fig'].savefig(self.mb.DIR_OUT + 'gsf_spec_' + self.mb.ID + '.pdf', dpi=dpi)
        else:
            self.axes['fig'].savefig(self.mb.DIR_OUT + 'gsf_spec_' + self.mb.ID + '.png', dpi=dpi)

        if return_figure:
            return self.axes['fig']

        self.axes['fig'].clear()
        plt.close()


    def plot_bb_sed(self, xbb, fybb, eybb, exbb, NRbb, d_scale, SNlim, c=3e5, col_dat = 'r', f_bbbox=False, sigma=1.0):
        """"""
        #######################################
        # D.Kelson like Box for BB photometry
        #######################################
        if f_bbbox:
            for ii in range(len(xbb)):
                if eybb[ii]<100 and fybb[ii]/eybb[ii]>1:
                    xx = [xbb[ii]-exbb[ii],xbb[ii]-exbb[ii]]
                    yy = [(fybb[ii]-eybb[ii])*c/np.square(xbb[ii])/d_scale, (fybb[ii]+eybb[ii])*c/np.square(xbb[ii])/d_scale]
                    self.axes['ax1'].plot(xx, yy, color='k', linestyle='-', linewidth=0.5, zorder=3)
                    xx = [xbb[ii]+exbb[ii],xbb[ii]+exbb[ii]]
                    yy = [(fybb[ii]-eybb[ii])*c/np.square(xbb[ii])/d_scale, (fybb[ii]+eybb[ii])*c/np.square(xbb[ii])/d_scale]
                    self.axes['ax1'].plot(xx, yy, color='k', linestyle='-', linewidth=0.5, zorder=3)
                    xx = [xbb[ii]-exbb[ii],xbb[ii]+exbb[ii]]
                    yy = [(fybb[ii]-eybb[ii])*c/np.square(xbb[ii])/d_scale, (fybb[ii]-eybb[ii])*c/np.square(xbb[ii])/d_scale]
                    self.axes['ax1'].plot(xx, yy, color='k', linestyle='-', linewidth=0.5, zorder=3)
                    xx = [xbb[ii]-exbb[ii],xbb[ii]+exbb[ii]]
                    yy = [(fybb[ii]+eybb[ii])*c/np.square(xbb[ii])/d_scale, (fybb[ii]+eybb[ii])*c/np.square(xbb[ii])/d_scale]
                    self.axes['ax1'].plot(xx, yy, color='k', linestyle='-', linewidth=0.5, zorder=3)
        else: # Normal BB plot;
            # Detection;
            conbb_hs = (fybb/eybb>SNlim)
            self.axes['ax1'].errorbar(xbb[conbb_hs], fybb[conbb_hs] * c / np.square(xbb[conbb_hs]) /d_scale, \
            yerr=eybb[conbb_hs]*c/np.square(xbb[conbb_hs])/d_scale, color='k', linestyle='', linewidth=0.5, zorder=4)
            self.axes['ax1'].plot(xbb[conbb_hs], fybb[conbb_hs] * c / np.square(xbb[conbb_hs]) /d_scale, \
            marker='.', color=col_dat, linestyle='', linewidth=0, zorder=4, ms=8)#, label='Obs.(BB)')
            try:
                # For any data removed fron fit (i.e. IRAC excess):
                #data_ex = ascii.read(DIR_TMP + 'bb_obs_' + ID + '_removed.cat')
                NR_ex = MB.data['bb_obs_removed']['NR']# data_ex['col1']
            except:
                NR_ex = []

            # Upperlim;
            if len(fybb[conbb_hs]):
                leng = np.nanmax(fybb[conbb_hs] * c / np.square(xbb[conbb_hs]) /d_scale) * 0.05 #0.2
            else:
                leng = None
            conebb_ls = (fybb/eybb<=SNlim) & (eybb>0)
            
            for ii in range(len(xbb)):
                if NRbb[ii] in NR_ex[:]:
                    conebb_ls[ii] = False

            self.axes['ax1'].errorbar(xbb[conebb_ls], eybb[conebb_ls] * c / np.square(xbb[conebb_ls]) /d_scale * sigma, yerr=leng,\
                uplims=eybb[conebb_ls] * c / np.square(xbb[conebb_ls]) /d_scale * sigma, linestyle='', color=col_dat, marker='', ms=4, label='', zorder=4, capsize=3)

        return self.axes['ax1'], leng


    def setup_sed_library(self):
        """"""
        self.mb.lib = self.mb.fnc.open_spec_fits(fall=0)
        self.mb.lib_all = self.mb.fnc.open_spec_fits(fall=1, orig=True)
        if self.mb.f_dust:
            self.mb.lib_dust = self.mb.fnc.open_spec_dust_fits(fall=0)
            self.mb.lib_dust_all = self.mb.fnc.open_spec_dust_fits(fall=1)
        if self.mb.fneb:
            self.mb.lib_neb = self.mb.fnc.open_spec_fits(fall=0, f_neb=True)
            self.mb.lib_neb_all = self.mb.fnc.open_spec_fits(fall=1, orig=True, f_neb=True)
        if self.mb.fagn:
            self.mb.lib_agn = self.mb.fnc.open_spec_fits(fall=0, f_agn=True)
            self.mb.lib_agn_all = self.mb.fnc.open_spec_fits(fall=1, orig=True, f_agn=True)
        return self.mb


    @staticmethod
    def write_lines(ID, zbes, R_grs=45, dw=4, umag=1.0, ldw = 7, DIR_OUT='./'):
        '''
        TBD
        '''
        dlw = R_grs * dw # Can affect the SFR.
        
        ###################################
        # To add lines in the plot,
        # ,manually edit the following file
        # so as Fcont50 have >0.
        ###################################
        flw = open(DIR_OUT + ID + '_lines_fit.txt', 'w')
        flw.write('# LW flux_line eflux_line flux_cont EW eEW L_line eL_line\n')
        flw.write('# (AA) (Flam_1e-18) (Flam_1e-18) (Flam_1e-18) (AA) (AA) (erg/s) (erg/s)\n')
        flw.write('# Error in EW is 1sigma, by pm eflux_line.\n')
        flw.write('# If EW=-99, it means gaussian fit failed.\n')
        flw.write('# and flux is the sum of excess at WL pm %.1f AA.\n'%(dlw))
        flw.write('# Magnification is corrected; mu=%.3f\n'%(umag))
        try:
            fl = np.loadtxt(DIR_OUT + 'table_' + ID + '_lines.txt', comments='#')
            LW      = fl[:,2]
            Fcont50 = fl[:,3]
            Fline50 = fl[:,6]
            for ii in range(len(LW)):
                if Fcont50[ii] > 0:
                    WL = LW[ii] * (1.+zbes)
                    if ii == 7:
                        contmp = (x > WL - dlw) & (x < WL + dlw*1.5)
                    else:
                        contmp = (x > WL - dlw) & (x < WL + dlw)
                    FLW[ii] = 1

                    xx   = x[contmp]
                    yy   = (fy - ysum_cut)[contmp]
                    eyy  = ey[contmp]
                    yy2  = (ysum_cut)[contmp]

                    xyzip = zip(xx,yy,eyy,yy2)
                    xyzip = sorted(xyzip)

                    xxs  = np.array([p1 for p1,p2,p3,p4 in xyzip])
                    yys  = np.array([p2 for p1,p2,p3,p4 in xyzip])
                    eyys = np.array([p3 for p1,p2,p3,p4 in xyzip])
                    yy2s = np.array([p4 for p1,p2,p3,p4 in xyzip])

                    flux = np.zeros(len(xxs), dtype=float)
                    efl  = np.zeros(len(xxs), dtype=float)
                    for ff in range(len(xxs)):
                        flux[ff] = yy2s[ff]/np.square(xxs[ff]) * c/d_scale
                        efl[ff]  = np.square(eyys[ff]/np.square(xxs[ff]) * c/d_scale)

                    fmed = np.median(flux) # Median of continuum, model flux
                    esum = np.sqrt(simps(efl, xxs))

                    try:
                        popt,pcov = curve_fit(gaus,xxs,yys,p0=[Fline50[ii],WL,10],sigma=eyys)
                        xxss = xxs/zscl

                        if ii == 7:
                            popt,pcov = curve_fit(gaus,xxs,yys,p0=[Fline50[ii],WL+20,10],sigma=eyys)
                            xxss = xxs/zscl

                        if f_grsm:
                            ax2t.plot(xxs/zscl, (gaus(xxs,*popt)+yy2s) * c/np.square(xxs)/d_scale, '#4682b4', linestyle='-', linewidth=1, alpha=0.8, zorder=20)

                        I1 = simps((gaus(xxs,*popt)) * c/np.square(xxs)/d_scale, xxs)
                        I2 = I1 - simps((gaus(xxs,*popt)) * c/np.square(xxs)/d_scale, xxs)
                        fline = I1

                        Flum = fline*Cons*1e-18 # luminosity in erg/s.
                        elum = esum *Cons*1e-18 # luminosity in erg/s.
                        SFR  = Flum * 6.58*1e-42
                        print('SFR is', SFR/umag)
                        EW_tmp   = simps( ((gaus(xxs,*popt)) * c/np.square(xxs)/d_scale)/yy2s, xxs)
                        EW_tmp_u = simps( ((gaus(xxs,*popt) + eyys/np.sqrt(len(xxs))) * c/np.square(xxs)/d_scale)/yy2s, xxs)

                        if ii == 7:
                            contmp2 = (xxs/zscl>4320.) & (xxs/zscl<4380.)
                            popt,pcov = curve_fit(gaus,xxs[contmp2], yys[contmp2], p0=[Fline50[ii],WL,10], sigma=eyys[contmp2])

                            I1 = simps((gaus(xxs[contmp2],*popt)) * c/np.square(xxs[contmp2])/d_scale, xxs[contmp2])
                            I2 = I1 - simps((gaus(xxs[contmp2],*popt)) * c/np.square(xxs[contmp2])/d_scale, xxs[contmp2])
                            fline = I1

                            Flum = fline*Cons*1e-18 # luminosity in erg/s.
                            elum = esum *Cons*1e-18 # luminosity in erg/s.
                            SFR  = Flum * 6.58*1e-42
                            print('SFR, update, is', SFR/umag)
                            EW_tmp   = simps( ((gaus(xxs[contmp2],*popt)) * c/np.square(xxs[contmp2])/d_scale)/yy2s[contmp2], xxs[contmp2])
                            EW_tmp_u = simps( ((gaus(xxs[contmp2],*popt) + eyys[contmp2]/np.sqrt(len(xxs[contmp2]))) * c/np.square(xxs[contmp2])/d_scale)/yy2s[contmp2], xxs[contmp2])

                        flw.write('%d %.2f %.2f %.2f %.2f %.2f %.2e %.2e %.2f\n'%(LW[ii],fline/umag, esum/umag, fmed/umag, EW_tmp,(EW_tmp_u-EW_tmp), Flum*1e-18/umag, elum*1e-18/umag, SFR/umag))

                    except Exception:
                        fsum = np.zeros(len(xxs))
                        for ff in range(len(fsum)):
                            fsum[ff] = (yys[ff]+yy2s[ff])/np.square(xxs[ff])

                        fline = np.sum(fsum) /d_scale*c
                        flw.write('%d %.2f %.2f %.2f %d %d %d %d %d\n'%(LW[ii],fline,esum,fmed, -99, 0, -99, 0, 0))
                        pass

        except:
            pass
        flw.close()


    @staticmethod
    def read_mcmc_chain(ID, samplepath='', use_pickl=True):
        if use_pickl:
            pfile = 'gsf_chain_' + ID + '.cpkl'
            data = loadcpkl(os.path.join(samplepath+'/'+pfile))
        else:
            pfile = 'gsf_chain_' + ID + '.asdf'
            data = asdf.open(os.path.join(samplepath+'/'+pfile))

        try:
            ndim   = data['ndim']     # By default, use ndim and burnin values contained in the cpkl file, if present.
            burnin = data['burnin']
            nmc    = data['niter']
            nwalk  = data['nwalkers']
            Nburn  = burnin
            if use_pickl:
                samples = data['chain'][:]
            else:
                samples = data['chain']
            return ndim, Nburn, samples
        except:
            msg = ' =   >   NO keys of ndim and burnin found in cpkl, use input keyword values'
            print_err(msg, exit=False)
            return False, False, False


    @staticmethod
    def show_chi2(hdul, fy, ey, ysump, x, wht3, ndim_eff, x_ex=[], SNlim=3, f_chind=True, f_exclude=False):
        """ based on Sawick12 """
        from scipy import special

        if f_chind:
            conw = (wht3>0) & (ey>0) & (fy/ey>SNlim)
        else:
            conw = (wht3>0) & (ey>0) #& (fy/ey>SNlim)

        try:
            logf = hdul[1].data['logf'][1]
            ey_revised = np.sqrt(ey**2+ ysump**2 * np.exp(logf)**2)
        except:
            ey_revised = ey

        chi2 = sum((np.square(fy-ysump) / ey_revised)[conw])

        chi_nd = 0.0
        if f_chind:
            f_ex = np.zeros(len(fy), 'int')
            if f_exclude:
                for ii in range(len(fy)):
                    if x[ii] in x_ex:
                        f_ex[ii] = 1

            con_up = (ey>0) & (fy/ey<=SNlim) & (f_ex == 0)
            x_erf = (ey_revised[con_up] - ysump[con_up]) / (np.sqrt(2) * ey_revised[con_up])
            f_erf = special.erf(x_erf)
            chi_nd = np.sum( np.log(np.sqrt(np.pi / 2) * ey_revised[con_up] * (1 + f_erf)) )

        # Number of degree;
        con_nod = (wht3>0) & (ey>0) #& (fy/ey>SNlim)
        nod = int(len(wht3[con_nod])-ndim_eff)

        print('\n')
        print('No-of-detection    : %d'%(len(wht3[conw])))
        print('chi2               : %.2f'%(chi2))
        if f_chind:
            print('No-of-non-detection: %d'%(len(ey[con_up])))
            print('chi2 for non-det   : %.2f'%(- 2 * chi_nd))
        print('No-of-params       : %d'%(ndim_eff))
        print('Degrees-of-freedom : %d'%(nod))
        if nod>0:
            fin_chi2 = (chi2 - 2 * chi_nd) / nod
        else:
            fin_chi2 = -99
        print('Final chi2/nu      : %.2f'%(fin_chi2))    
        return chi2, conw, con_up, chi_nd, nod, fin_chi2


    def plot_bbmodel_sed(self, zbes, xbb, fybb, eybb, x_ex, fy_ex, ey_ex, x1_tot, ytmp16, ytmp50, ytmp84, 
                         SFILT, DFILT, DIR_FILT, scale, d_scale, DL, leng, sigma, 
                         SNlim=2, c=3e5, col_dat='r', col_dia='blue'):
        """"""    
        if self.mb.f_dust:
            ALLFILT = np.append(SFILT,DFILT)
            lbb, fbb, lfwhm = filconv(ALLFILT, x1_tot, ytmp50, DIR_FILT, fw=True, MB=self.mb, f_regist=False)
            lbb, fbb16, lfwhm = filconv(ALLFILT, x1_tot, ytmp16, DIR_FILT, fw=True, MB=self.mb, f_regist=False)
            lbb, fbb84, lfwhm = filconv(ALLFILT, x1_tot, ytmp84, DIR_FILT, fw=True, MB=self.mb, f_regist=False)

            self.axes['ax1'].plot(x1_tot, ytmp50, '--', lw=0.5, color='purple', zorder=-1, label='')
            self.axes['ax3t'].plot(x1_tot, ytmp50, '--', lw=0.5, color='purple', zorder=-1, label='')

            iix = []
            for ii in range(len(fbb)):
                iix.append(ii)
            con_sed = ()
            self.axes['ax1'].scatter(lbb[iix][con_sed], fbb[iix][con_sed], lw=0.5, color='none', edgecolor=col_dia, zorder=3, alpha=1.0, marker='d', s=50)

            # plot FIR range;
            self.axes['ax3t'].scatter(lbb, fbb, lw=0.5, color='none', edgecolor=col_dia, \
            zorder=2, alpha=1.0, marker='d', s=50)

        else:
            lbb, fbb, lfwhm = filconv(SFILT, x1_tot, ytmp50, DIR_FILT, fw=True, MB=self.mb, f_regist=False)
            lbb, fbb16, lfwhm = filconv(SFILT, x1_tot, ytmp16, DIR_FILT, fw=True, MB=self.mb, f_regist=False)
            lbb, fbb84, lfwhm = filconv(SFILT, x1_tot, ytmp84, DIR_FILT, fw=True, MB=self.mb, f_regist=False)

            iix = []
            for ii in range(len(fbb)):
                iix.append(np.argmin(np.abs(lbb[ii]-xbb[:])))
            con_sed = (eybb>0)
            self.axes['ax1'].scatter(lbb[iix][con_sed], fbb[iix][con_sed], lw=0.5, color='none', edgecolor=col_dia, zorder=3, alpha=1.0, marker='d', s=50)

            if self.f_plot_resid:
                conbb_hs = (fybb/eybb>SNlim)
                axes['B'].scatter(lbb[iix][conbb_hs], ((fybb*c/np.square(xbb)/d_scale - fbb)/(eybb*c/np.square(xbb)/d_scale))[iix][conbb_hs], lw=0.5, color='none', edgecolor='r', zorder=3, alpha=1.0, marker='.', s=50)
                conbb_hs = (fybb/eybb<=SNlim) & (eybb>0)
                axes['B'].errorbar(lbb[iix][conbb_hs], ((eybb*c/np.square(xbb)/d_scale - fbb)/(eybb*c/np.square(xbb)/d_scale))[iix][conbb_hs], yerr=leng,\
                    uplims=((fybb*c/np.square(xbb)/d_scale - fbb)/(eybb*c/np.square(xbb)/d_scale))[iix][conbb_hs] * sigma, linestyle='',\
                    color=col_dat, lw=0.5, marker='', ms=4, label='', zorder=4, capsize=1.5)
                axes['B'].set_xscale(self.axes['ax1'].get_xscale())
                axes['B'].set_xlim(self.axes['ax1'].get_xlim())
                axes['B'].set_xticks(self.axes['ax1'].get_xticks())
                axes['B'].set_xticklabels(self.axes['ax1'].get_xticklabels())
                axes['B'].set_xlabel(self.axes['ax1'].get_xlabel())
                xx = np.arange(axes['B'].get_xlim()[0],axes['B'].get_xlim()[1],100)
                axes['B'].plot(xx,xx*0,linestyle='--',lw=0.5,color='k')
                axes['B'].set_ylabel('Residual / $\sigma$')
                axes['A'].set_xlabel('')
                axes['A'].set_xticks(self.axes['ax1'].get_xticks())
                axes['A'].set_xticklabels('')

            # Calculate EW, if there is excess band;
            try:
                iix2 = []
                for ii in range(len(fy_ex)):
                    iix2.append(np.argmin(np.abs(lbb[:]-x_ex[ii])))

                # Rest-frame EW;
                # Note about 16/84 in fbb
                EW16 = (fy_ex * c / np.square(x_ex) /d_scale - fbb84[iix2]) / (fbb[iix2]) * lfwhm[iix2] / (1.+zbes)
                EW50 = (fy_ex * c / np.square(x_ex) /d_scale - fbb[iix2]) / (fbb[iix2]) * lfwhm[iix2] / (1.+zbes)
                EW84 = (fy_ex * c / np.square(x_ex) /d_scale - fbb16[iix2]) / (fbb[iix2]) * lfwhm[iix2] / (1.+zbes)

                EW50_er1 = ((fy_ex-ey_ex) * c / np.square(x_ex) /d_scale - fbb[iix2]) / (fbb[iix2]) * lfwhm[iix2] / (1.+zbes)
                EW50_er2 = ((fy_ex+ey_ex) * c / np.square(x_ex) /d_scale - fbb[iix2]) / (fbb[iix2]) * lfwhm[iix2] / (1.+zbes)

                cnt50 = fbb[iix2] # in Flam
                cnt16 = fbb16[iix2] # in Flam
                cnt84 = fbb84[iix2] # in Flam
    
                # Luminosity;
                #Lsun = 3.839 * 1e33 #erg s-1
                L16 = EW16 * cnt16 * (4.*np.pi*DL**2) * scale * (1+zbes) # A * erg/s/A/cm2 * cm2
                L50 = EW50 * cnt50 * (4.*np.pi*DL**2) * scale * (1+zbes) # A * erg/s/A/cm2 * cm2
                L84 = EW84 * cnt84 * (4.*np.pi*DL**2) * scale * (1+zbes) # A * erg/s/A/cm2 * cm2

                ew_label = []
                for ii in range(len(fy_ex)):
                    lres = self.mb.band['%s_lam'%self.mb.filts[iix2[ii]]][:]
                    fres = self.mb.band['%s_res'%self.mb.filts[iix2[ii]]][:]
                    ew_label.append(self.mb.filts[iix2[ii]])

                    print('\n')
                    print('EW016 for', x_ex[ii], 'is %d'%EW16[ii])
                    print('EW050 for', x_ex[ii], 'is %d'%EW50[ii])
                    print('EW084 for', x_ex[ii], 'is %d'%EW84[ii])
                    print('%d_{-%d}^{+%d} , for sed error'%(EW50[ii],EW50[ii]-EW84[ii],EW16[ii]-EW50[ii]))
                    print('Or, %d\pm{%d} , for flux error'%(EW50[ii],EW50[ii]-EW50_er1[ii]))
            except:
                print('\nEW calculation; Failed.\n')
                EW16, EW50, EW84, EW50_er1, EW50_er2, cnt16, cnt50, cnt84, L16, L50, L84 = None, None, None, None, None, None, None, None, None, None, None
                ew_label = []
                pass

        return lbb, fbb, fbb16, fbb84, ew_label, EW16, EW50, EW84, EW50_er1, EW50_er2, cnt16, cnt50, cnt84, L16, L50, L84


    def plot_dust_sed(self, AD50, nTD50, zp50, c=3e5, d_scale=1, SNlim=3):
        """"""
        par = Parameters()
        par.add('MDUST',value=AD50)
        par.add('TDUST',value=nTD50)
        par.add('zmc',value=zp50)

        y0d, x0d = self.mb.fnc.tmp04_dust(par.valuesdict())#, zbes, lib_dust_all)
        y0d_cut, _ = self.mb.fnc.tmp04_dust(par.valuesdict())#, zbes, lib_dust)

        # data;
        xbbd, fybbd, eybbd = self.mb.data['spec_fir_obs']['x'], self.mb.data['spec_fir_obs']['fy'], self.mb.data['spec_fir_obs']['ey']

        try:
            conbbd_hs = (fybbd/eybbd>SNlim)
            self.axes['ax1'].errorbar(xbbd[conbbd_hs], fybbd[conbbd_hs] * c / np.square(xbbd[conbbd_hs]) /d_scale, \
            yerr=eybbd[conbbd_hs]*c/np.square(xbbd[conbbd_hs])/d_scale, color='k', linestyle='', linewidth=0.5, zorder=4)
            self.axes['ax1'].plot(xbbd[conbbd_hs], fybbd[conbbd_hs] * c / np.square(xbbd[conbbd_hs]) /d_scale, \
            '.r', linestyle='', linewidth=0, zorder=4)#, label='Obs.(BB)')
            self.axes['ax3t'].plot(xbbd[conbbd_hs], fybbd[conbbd_hs] * c / np.square(xbbd[conbbd_hs]) /d_scale, \
            '.r', linestyle='', linewidth=0, zorder=4)#, label='Obs.(BB)')
        except:
            pass

        try:
            conebbd_ls = (fybbd/eybbd<=SNlim)
            self.axes['ax1'].errorbar(xbbd[conebbd_ls], eybbd[conebbd_ls] * c / np.square(xbbd[conebbd_ls]) /d_scale, \
            yerr=fybbd[conebbd_ls]*0+np.max(fybbd[conebbd_ls]*c/np.square(xbbd[conebbd_ls])/d_scale)*0.05, \
            uplims=eybbd[conebbd_ls]*c/np.square(xbbd[conebbd_ls])/d_scale, color='r', linestyle='', linewidth=0.5, zorder=4)
            self.axes['ax3t'].errorbar(xbbd[conebbd_ls], eybbd[conebbd_ls] * c / np.square(xbbd[conebbd_ls]) /d_scale, \
            yerr=fybbd[conebbd_ls]*0+np.max(fybbd[conebbd_ls]*c/np.square(xbbd[conebbd_ls])/d_scale)*0.05, \
            uplims=eybbd[conebbd_ls]*c/np.square(xbbd[conebbd_ls])/d_scale, color='r', linestyle='', linewidth=0.5, zorder=4)
        except:
            pass

        return self.axes['ax1'], self.axes['ax3t'], y0d_cut, y0d, x0d


    def save_files_sed(self, percs=[16,50,84]):
        gsf_dict = {}
        gsf_dict['primary_params'] = {}

        # a single ASDF;
        tree_shf = asdf.open(os.path.join(self.mb.DIR_OUT, 'gsf_sfh_%s.asdf'%(self.mb.ID)))
        gsf_dict['sfh'] = {}
        gsf_dict = PLOT.modify_keys_sed(tree_shf, 'sfh', gsf_dict=gsf_dict)
        tsets_SFR = tree_shf['header']['tsets_SFR'].split(',')
        tsets_SFR = [float(s) for s in tsets_SFR]

        tree_sed = asdf.open(os.path.join(self.mb.DIR_OUT, 'gsf_spec_%s.asdf'%(self.mb.ID)))
        gsf_dict['sed'] = {}

        lbls_sed_skip = ['header']
        lbls_sed_skip = [s.upper() for s in lbls_sed_skip]
        key_skip = ['BITPIX', 'EXTEND', 'SIMPLE', 'NAXIS']
        key_skip += lbls_sed_skip

        label = 'sed'
        gsf_dict = PLOT.modify_keys_sed(tree_sed, label, gsf_dict=gsf_dict, key_skip=key_skip)

        keys_param_sed = ['MUV', 'SFRUV', 'SFRUV_STEL', 'SFRUV_BETA', 'SFRUV_UNCOR', 'UVBETA', 'UVBETA_OBS', 'UV', 'VJ']
        keys_param_sfh = ['ZMC', 'MSTEL', 'T_LW', 'T_MW', 'Z_LW', 'Z_MW', 'AV0']
        for key in keys_param_sed:
            for perc in percs:
                try:
                    gsf_dict['primary_params']['%s_%d'%(key, perc)] = gsf_dict['sed']['%s_%d'%(key, perc)]
                except:
                    print('%s_%d'%(key, perc),'cannot be found')

        for key in keys_param_sfh:
            for perc in percs:
                try:
                    gsf_dict['primary_params']['%s_%d'%(key, perc)] = gsf_dict['sfh']['%s_%d'%(key, perc)]
                except:
                    print('%s_%d'%(key, perc),'cannot be found')

        key = 'SFR'
        for t in range(len(tsets_SFR)):
            for perc in percs:
                gsf_dict['primary_params']['%s_%dMYR_%d'%(key, tsets_SFR[t]*1e3, perc)] = gsf_dict['sfh']['%s_%dMYR_%d'%(key, tsets_SFR[t]*1e3, perc)]

        af = asdf.AsdfFile(gsf_dict)
        af.write_to(os.path.join(self.mb.DIR_OUT, 'gsf_%s.asdf'%(self.mb.ID)), all_array_compression='zlib')
        return gsf_dict


    @staticmethod
    def modify_keys_sed(fd_sfh, label, gsf_dict=None, key_skip=['BITPIX', 'EXTEND', 'SIMPLE', 'NAXIS']):
        '''
        label : 'sfh' or 'sed' 
        '''
        q_labels = []
        percs = [16,50,84]
        
        if gsf_dict == None:
            gsf_dict = {}
            gsf_dict[label] = {}
            
        if label not in gsf_dict:
            gsf_dict[label] = {}

        for key in fd_sfh['header'].keys():
            key_mod = key
            for perc in percs:
                key_mod = key_mod.replace('_%d'%perc,'%d'%perc)
                key_mod = key_mod.replace('%d'%perc,'_%d'%perc)
            key_mod = key_mod.upper()

            if key_mod in key_skip:
                continue

            gsf_dict[label][key_mod] = fd_sfh['header'][key]

            key_q = key_mod
            for perc in percs:
                key_q = key_q.replace('_%d'%perc,'')
            if key_q not in q_labels:
                q_labels.append(key_q)

        for key in fd_sfh.keys():
            key_mod = key
            for perc in percs:
                key_mod = key_mod.replace('_%d'%perc,'%d'%perc)
                key_mod = key_mod.replace('%d'%perc,'_%d'%perc)
            key_mod = key_mod.upper()

            if key_mod in key_skip:
                continue

            gsf_dict[label][key_mod] = fd_sfh[key]

            key_q = key_mod
            for perc in percs:
                key_q = key_q.replace('_%d'%perc,'')
            if key_q not in q_labels:
                q_labels.append(key_q)

        gsf_dict[label]['quantities'.upper()] = q_labels
        return gsf_dict


    def update_axis_sed(self, x1min, x1max, ymax, scale, ey, wht3, f_plot_filter=False):
        """"""
        self.axes['ax1'].set_xlabel('Observed wavelength [$\mathrm{\mu m}$]', fontsize=11)
        self.axes['ax1'].set_ylabel('$f_\lambda$ [$10^{%d}\mathrm{erg}/\mathrm{s}/\mathrm{cm}^{2}/\mathrm{\AA}$]'%(np.log10(scale)),fontsize=11,labelpad=2)

        xticks = [2500, 5000, 10000, 20000, 40000, 80000, x1max]
        xlabels= ['0.25', '0.5', '1', '2', '4', '8', '']
        if self.mb.f_dust:
            x1max = 400000
            xticks = [2500, 5000, 10000, 20000, 40000, 80000, 400000]
            xlabels= ['0.25', '0.5', '1', '2', '4', '8', '']

        if x1min > 2500:
            xticks = xticks[1:]
            xlabels = xlabels[1:]

        self.axes['ax1'].set_xlim(x1min, x1max)
        self.axes['ax1'].set_xscale('log')
        if f_plot_filter:
            scl_yaxis = 0.2
        else:
            scl_yaxis = 0.1

        if not ymax == None:
            self.axes['ax1'].set_ylim(-ymax*scl_yaxis,ymax)

        self.axes['ax1'].set_xticks(xticks)
        self.axes['ax1'].set_xticklabels(xlabels)

        ###############
        # Line name
        ###############
        if False:
            add_line_names()

        # Filters
        ind_remove = np.where((wht3<=0) | (ey<=0))[0]
        if f_plot_filter:
            _ = PLOT.plot_filter(ymax, scl=scl_yaxis, ind_remove=ind_remove)

        xx = np.arange(100,400000)
        yy = xx * 0
        self.axes['ax1'].plot(xx, yy, ls='--', lw=0.5, color='k')
        self.axes['ax1'].legend(loc=1, fontsize=11)
        self.axes['ax1'].xaxis.labelpad = -3

        if self.mb.f_dust:
            self.axes['ax3t'].set_xlim(1e4, 3e7)
            self.axes['ax3t'].set_xscale('log')
            self.axes['ax3t'].set_xticks([10000, 1000000, 10000000])
            self.axes['ax3t'].set_xticklabels(['1', '100', '1000'])

        return self.axes['ax1'] #ax1t, ax2, ax2t

    def plot_filter(self, ymax, scl=0.3, cmap='gist_rainbow', alp=0.4, 
                    ind_remove=[], nmax=1000, plot_log=False):
        '''
        Add filter response curve to ax1.
        '''
        ax, filt_responses = self.plot_filter_core(ymax, scl=scl, cmap=cmap, alp=alp, 
                    ind_remove=ind_remove, nmax=nmax, plot_log=plot_log)
        self.mb.filt_responses = filt_responses
        return ax


    def plot_filter_core(self, ymax, scl=0.3, cmap='gist_rainbow', alp=0.4, 
                    ind_remove=[], nmax=1000, plot_log=False):
        ''''''
        filts = []
        for key0 in self.mb.band.keys():
            key = key0.split('_')[0]
            if key not in filts:
                filts.append(key)

        NUM_COLORS = len(filts)
        cm = plt.get_cmap(cmap)
        cols = [cm(1 - 1.*i/NUM_COLORS) for i in range(NUM_COLORS)]

        filt_responses = {'colors':[],'filters':{}}
        wavecen = []
        for ii,filt in enumerate(filts):
            wave = self.mb.band['%s_lam'%filt]
            flux = self.mb.band['%s_res'%filt]
            con = (flux/flux.max()>0.1)
            wavecen.append(np.min(wave[con]))
        wavecen = np.asarray(wavecen)
        wavecen_sort = np.sort(wavecen)

        for ii,filt in enumerate(filts):
            iix = np.argmin(np.abs(wavecen_sort[:]-wavecen[ii]))
            col = cols[iix]
            wave = self.mb.band['%s_lam'%filt]
            flux = self.mb.band['%s_res'%filt]
            
            if len(wave) > nmax:
                nthin = int(len(wave)/nmax)
            else:
                nthin = 1

            filt_responses['filters'][filt] = {}
            wave_tmp = np.zeros(len(wave[::nthin]), float)
            res_tmp = np.zeros(len(wave[::nthin]), float)

            wave_tmp[:] = wave[::nthin]
            res_tmp[:] = flux[::nthin]

            filt_responses['filters'][filt]['wave'] = wave_tmp
            filt_responses['filters'][filt]['response'] = res_tmp

            # Get fwhm;
            fsum = np.nansum(res_tmp)
            fcum = np.zeros(len(res_tmp), dtype=float)
            lam0,lam1 = 0,0
            wave_median = 0
            for jj in range(len(res_tmp)):
                fcum[jj] = np.nansum(res_tmp[:jj])/fsum
                if lam0 == 0 and fcum[jj]>0.05:
                    lam0 = wave_tmp[jj]
                if lam1 == 0 and fcum[jj]>0.95:
                    lam1 = wave_tmp[jj]
                if wave_median == 0 and fcum[jj]>0.50:
                    wave_median = wave_tmp[jj]
            fwhm = lam1 - lam0
            filt_responses['filters'][filt]['wave_mean'] = wave_median
            filt_responses['filters'][filt]['fwhm'] = fwhm
            filt_responses['colors'].append(col)

            if ii in ind_remove:
                continue

            if not plot_log:
                self.mb.axes['ax1'].plot(wave, ((flux / np.nanmax(flux))*0.8 - 1) * ymax * scl, linestyle='-', color='k', lw=0.2)
                self.mb.axes['ax1'].fill_between(wave, (wave*0 - ymax)*scl, ((flux / np.nanmax(flux))*0.8 - 1) * ymax * scl, linestyle='-', lw=0, color=col, alpha=alp)
            else:
                self.mb.axes['ax1'].plot(wave, ((flux / np.nanmax(flux))*0.8 - 1) * ymax * scl, linestyle='-', color='k', lw=0.2)
                self.mb.axes['ax1'].fill_between(wave, ((flux / np.nanmax(flux))*0.8 - 1) * ymax * scl * 0.001, ((flux / np.nanmax(flux))*0.8 - 1) * ymax * scl, linestyle='-', lw=0, color=col, alpha=alp)

        return self.mb.axes['ax1'],filt_responses
