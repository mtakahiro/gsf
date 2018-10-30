#!/usr/bin/env python
import numpy as np
import sys
import matplotlib.pyplot as plt
from lmfit import Model, Parameters, minimize, fit_report, Minimizer
from numpy import log10
from scipy.integrate import simps
import pickle as cPickle
import os.path
import random
from astropy.io import fits
import string
import timeit

# import from custom codes
from .function import check_line_man, check_line_cz_man, filconv, calc_Dn4, savecpkl
from .zfit import check_redshift
from .plot_Zevo import *
from .plot_sfh import plot_sfh_pcl2


############################
py_v = (sys.version_info[0])
if py_v > 2:
    try:
        raw_input = input
    except NameError:
        pass

#################
# Physical params
#################
c    = 3.e18 # A/s
mag0 = 25.0 # Magnitude zeropoint set.
d    = 10**(73.6/2.5) # From [ergs/s/cm2/A] to [ergs/s/cm2/Hz]


################
# Line library
################
LN = ['Mg2', 'Ne5', 'O2', 'Htheta', 'Heta', 'Ne3', 'Hdelta', 'Hgamma', 'Hbeta', 'O3L', 'O3H', 'Mgb', 'Halpha', 'S2L', 'S2H']
LW = [2800, 3347, 3727, 3799, 3836, 3869, 4102, 4341, 4861, 4960, 5008, 5175, 6563, 6717, 6731]
fLW = np.zeros(len(LW), dtype='int') # flag.


################
# RF colors.
################
home = os.path.expanduser('~')
#fil_path = home + '/eazy-v1.01/PROG/FILT/'
fil_path = home + '/Dropbox/FILT/'
eaz_path = home + '/Dropbox/OUTPUT_M1149/' # From which z-prior is taken.


#####################
# Function fo MCMC
#####################
class Mainbody():
    def __init__(self, inputs):
        self.inputs = inputs

    def get_lines(self, LW0):
        fLW = np.zeros(len(LW0), dtype='int')
        LW  = LW0
        return LW, fLW

    def main(self, ID0, PA0, zgal, flag_m, zprev, Cz0, Cz1, mcmcplot=1, fzvis=1, specplot=1, fneld=0, ntemp=5): # flag_m related to redshift error in redshift check func.

        start = timeit.default_timer()

        inputs = self.inputs
        DIR_TMP  = inputs['DIR_TEMP']

        if os.path.exists(DIR_TMP) == False:
            os.mkdir(DIR_TMP)

        #
        # Age
        #
        age = inputs['AGE']
        age = [float(x.strip()) for x in age.split(',')]
        nage = np.arange(0,len(age),1)

        #
        # Metallicity
        #
        Zmax, Zmin = float(inputs['ZMAX']), float(inputs['ZMIN'])
        delZ = float(inputs['DELZ'])
        Zall = np.arange(Zmin, Zmax, delZ) # in logZsun

        # For minimizer.
        delZtmp = delZ
        #delZtmp = 0.4 # to increase speed.

        # For z prior.
        delzz  = 0.001
        zlimu  = 6.
        snlim  = 1
        zliml  = zgal - 0.5


        # N of param:
        try:
            ndim = int(inputs['NDIM'])
        except:
            if int(inputs['ZEVOL']) == 1:
                ndim = int(len(nage) * 2 + 1)
            else:
                ndim = int(len(nage) + 1 + 1)

            pass

        #
        # Line
        #
        LW0 = inputs['LINE']
        LW0 = [float(x.strip()) for x in LW0.split(',')]

        #
        # Params for MCMC
        #
        nmc      = int(inputs['NMC'])
        nwalk    = int(inputs['NWALK'])
        nmc_cz   = int(inputs['NMCZ'])
        nwalk_cz = int(inputs['NWALKZ'])
        f_Zevol  = int(inputs['ZEVOL'])
        #f_zvis   = int(inputs['ZVIS'])

        #
        # Tau for MCMC parameter; not as fitting parameters.
        #
        tau0 = inputs['TAU0']
        tau0 = [float(x.strip()) for x in tau0.split(',')]
        #tau0     = [0.1,0.2,0.3] # Gyr


        from .function_class import Func
        from .basic_func import Basic
        fnc  = Func(Zall, nage) # Set up the number of Age/ZZ
        bfnc = Basic(Zall)

        from .writing import Analyze
        wrt = Analyze(inputs) # Set up for input


        # Open ascii file and stock to array.
        #lib = open_spec(ID0, PA0)
        lib     = fnc.open_spec_fits(ID0, PA0, fall=0, tau0=tau0)
        lib_all = fnc.open_spec_fits(ID0, PA0, fall=1, tau0=tau0)

        #####################
        # Model templates.
        #####################
        chimax = 1.

        #################
        # Observed Data
        #################

        ##############
        # Spectrum
        ##############
        dat = np.loadtxt(DIR_TMP + 'spec_obs_' + ID0 + '_PA' + PA0 + '.cat', comments='#')
        NR  = dat[:,0]
        x   = dat[:,1]
        fy00  = dat[:,2]
        ey00  = dat[:,3]

        con0 = (NR<1000)
        fy0  = fy00[con0] * Cz0
        ey0  = ey00[con0] * Cz0
        con1 = (NR>=1000) & (NR<10000)
        fy1  = fy00[con1] * Cz1
        ey1  = ey00[con1] * Cz1
        con2 = (NR>=10000) # BB
        fy2  = fy00[con2]
        ey2  = ey00[con2]

        fy01 = np.append(fy0,fy1)
        fy   = np.append(fy01,fy2)
        ey01 = np.append(ey0,ey1)
        ey   = np.append(ey01,ey2)

        wht  = 1./np.square(ey)
        sn   = fy/ey

        ##############
        # Broadband
        ##############
        dat = np.loadtxt(DIR_TMP + 'bb_obs_' + ID0 + '_PA' + PA0 + '.cat', comments='#')
        NRbb = dat[:, 0]
        xbb  = dat[:, 1]
        fybb = dat[:, 2]
        eybb = dat[:, 3]
        exbb = dat[:, 4]
        wht2 = check_line_man(fy, x, wht, fy, zprev,LW0)

        #####################
        # Function fo MCMC
        #####################
        def residual(pars): # x, y, wht are taken from out of the definition.
            vals = pars.valuesdict()
            Av   = vals['Av']
            ######################
            '''
            for aa in range(len(age)):
                if aa == 0:
                    mod0, x1 = fnc.tmp03(ID0, PA0, vals['A'+str(aa)], Av, aa, vals['Z'+str(aa)], zprev, lib, tau0)
                    model = mod0
                else:
                    mod0, xxbs = fnc.tmp03(ID0, PA0, vals['A'+str(aa)], Av, aa, vals['Z'+str(aa)], zprev, lib, tau0)
                    model += mod0
            '''
            model, x1 = fnc.tmp04(ID0, PA0, vals, zprev, lib, tau0=tau0)
            ######################
            if fy is None:
                print('Data is none')
                return model
            else:
                return (model - fy) * np.sqrt(wht2) # i.e. residual/sigma

        def lnprob(pars):
            vals = pars.valuesdict()
            Av = vals['Av']
            resid  = residual(pars)
            s      = 1. #pars['f']
            resid *= 1 / s
            resid *= resid
            resid += np.log(2 * np.pi * s**2)
            if Av<0:
                 return -np.inf
            else:
                respr = np.log(1)
            return -0.5 * np.sum(resid) + respr


        ###############################
        # Add parameters
        ###############################
        fit_params = Parameters()
        for aa in range(len(age)):
            if age[aa] == 99:
                fit_params.add('A'+str(aa), value=0, min=0, max=0.01)
            else:
                fit_params.add('A'+str(aa), value=1, min=0, max=400)

        #fit_params.add('Av', value=1., min=0, max=4.0)
        fit_params.add('Av', value=0.2, min=0, max=4.0)
        #fit_params.add('Z', value=0, min=np.min(Zall), max=np.max(Zall))
        for aa in range(len(age)):
            if age[aa] == 99:
                fit_params.add('Z'+str(aa), value=1, min=0, max=0.01)
            else:
                fit_params.add('Z'+str(aa), value=0, min=np.min(Zall), max=np.max(Zall))

        ##################################
        # Metallicity determination
        ##################################
        chidef = 1e5
        Zbest  = 0

        fwz = open('Z_' + ID0 + '_PA' + PA0 + '.cat', 'w')
        fwz.write('# ID Zini chi/nu AA Av Zbest\n')

        nZtmp = int((Zmax-Zmin)/delZtmp)

        if fneld == 1:
            for zz in range(0,nZtmp,1):
            #for zz in range(2,7,2):
                ZZ = zz * delZtmp + np.min(Zall)
                for aa in range(len(age)):
                    fit_params['Z'+str(aa)].value = ZZ

                out_tmp = minimize(residual, fit_params, method='nelder') # nelder is the most efficient.
                #print(ZZ, bfnc.Z2NZ(ZZ))
                #print(fit_report(out_tmp))
                #print('\n')

                keys = fit_report(out_tmp).split('\n')
                csq  = 99999
                rcsq = 99999
                for key in keys:
                    if key[4:7] == 'chi':
                        skey = key.split(' ')
                        csq  = float(skey[14])
                    if key[4:7] == 'red':
                        skey = key.split(' ')
                        rcsq = float(skey[7])

                fitc = [csq, rcsq] # Chi2, Reduced-chi2

                fwz.write('%s %.2f %.5f'%(ID0, ZZ, fitc[1]))

                AA_tmp = np.zeros(len(age), dtype='float32')
                ZZ_tmp = np.zeros(len(age), dtype='float32')
                for aa in range(len(age)):
                    AA_tmp[aa] = out_tmp.params['A'+str(aa)].value
                    fwz.write(' %.5f'%(AA_tmp[aa]))

                Av_tmp = out_tmp.params['Av'].value
                fwz.write(' %.5f'%(Av_tmp))
                for aa in range(len(age)):
                    ZZ_tmp[aa] = out_tmp.params['Z'+str(aa)].value
                    fwz.write(' %.5f'%(ZZ_tmp[aa]))

                fwz.write('\n')
                if fitc[1]<chidef:
                    chidef = fitc[1]
                    out    = out_tmp

        else:
            for zz in range(0,nZtmp,2):
                ZZ = zz * delZtmp + np.min(Zall)
                for aa in range(len(age)):
                    fit_params['Z'+str(aa)].value = ZZ

                #
                out_tmp = minimize(residual, fit_params, method='powell') # powel is the more accurate.
                #print(ZZ, bfnc.Z2NZ(ZZ))
                #print(fit_report(out_tmp))
                #print('\n')

                keys = fit_report(out_tmp).split('\n')
                csq  = 99999
                rcsq = 99999
                for key in keys:
                    if key[4:7] == 'chi':
                        skey = key.split(' ')
                        csq  = float(skey[14])
                    if key[4:7] == 'red':
                        skey = key.split(' ')
                        rcsq = float(skey[7])

                fitc = [csq, rcsq] # Chi2, Reduced-chi2
                fwz.write('%s %.2f %.5f'%(ID0, ZZ, fitc[1]))

                AA_tmp = np.zeros(len(age), dtype='float32')
                ZZ_tmp = np.zeros(len(age), dtype='float32')
                for aa in range(len(age)):
                    AA_tmp[aa] = out_tmp.params['A'+str(aa)].value
                    fwz.write(' %.5f'%(AA_tmp[aa]))

                Av_tmp = out_tmp.params['Av'].value
                fwz.write(' %.5f'%(Av_tmp))
                #Z_tmp  = out_tmp.params['Z'].value
                for aa in range(len(age)):
                    ZZ_tmp[aa] = out_tmp.params['Z'+str(aa)].value
                    fwz.write(' %.5f'%(ZZ_tmp[aa]))

                fwz.write('\n')
                if fitc[1]<chidef:
                    chidef = fitc[1]
                    out    = out_tmp


        #
        # Best fit
        #
        keys = fit_report(out).split('\n')
        for key in keys:
            if key[4:7] == 'chi':
                skey = key.split(' ')
                csq  = float(skey[14])
            if key[4:7] == 'red':
                skey = key.split(' ')
                rcsq = float(skey[7])

        fitc = [csq, rcsq] # Chi2, Reduced-chi2
        #fitc = fit_report_chi(out) # Chi2, Reduced-chi2
        ZZ   = Zbest # This is really important/does affect lnprob/residual.


        print('\n\n')
        print('#####################################')
        print('Zbest, chi are;',Zbest,chidef)
        print('Params are;',fit_report(out))
        print('#####################################')
        print('\n\n')
        fwz.close()

        Av_tmp = out.params['Av'].value
        #Z_tmp  = Zbest
        AA_tmp = np.zeros(len(age), dtype='float32')
        ZZ_tmp = np.zeros(len(age), dtype='float32')
        for aa in range(len(age)):
            AA_tmp[aa] = out.params['A'+str(aa)].value
            ZZ_tmp[aa] = out.params['Z'+str(aa)].value
            if aa == 0:
                mod0_tmp, xm_tmp = fnc.tmp03(ID0, PA0, AA_tmp[aa], Av_tmp, aa, ZZ_tmp[aa], zprev, lib, tau0)
                fm_tmp = mod0_tmp
            else:
                mod0_tmp, xx_tmp = fnc.tmp03(ID0, PA0, AA_tmp[aa], Av_tmp, aa, ZZ_tmp[aa], zprev, lib, tau0)
                fm_tmp += mod0_tmp


        ########################
        # Check redshift
        ########################
        zrecom = zprev

        # Observed data.
        con_cz = (NR<10000) #& (sn>snlim)
        fy_cz  = fy[con_cz]
        ey_cz  = ey[con_cz]
        x_cz   = x[con_cz] # Observed range
        NR_cz  = NR[con_cz]

        xm_s = xm_tmp / (1+zprev) * (1+zrecom)
        fm_s = np.interp(x_cz, xm_s, fm_tmp)


        if fzvis==1:
            plt.plot(x_cz/(1+zprev)*(1.+zrecom),fm_s,'gray', linestyle='--', linewidth=0.5, label='Default ($z=%.5f$)'%(zprev)) # Model based on input z.
            plt.plot(x_cz, fy_cz,'b', linestyle='-', linewidth=0.5, label='Obs.') # Observation
            plt.errorbar(x_cz, fy_cz, yerr=ey_cz, color='b', capsize=0, linewidth=0.5) # Observation


        if flag_m == 0:
            dez = 0.5
        else:
            dez = 0.2

        #
        # For Eazy
        #
        '''
        dprob = np.loadtxt(eaz_path + 'photz_' + str(int(ID0)) + '.pz', comments='#')
        zprob = dprob[:,0]
        cprob = dprob[:,1]

        zz_prob = np.arange(0,13,delzz)
        cprob_s = np.interp(zz_prob, zprob, cprob)
        prior_s = 1/cprob_s
        prior_s /= np.sum(prior_s)
        '''

        zz_prob  = np.arange(0,13,delzz)
        prior_s  = zz_prob * 0 + 1.
        prior_s /= np.sum(prior_s)

        print('################\nStart MCMC for redshift fit\n################')
        res_cz, fitc_cz = check_redshift(fy_cz,ey_cz,x_cz,fm_tmp,xm_tmp/(1+zprev),zprev,dez,prior_s,NR_cz, zliml, zlimu, delzz, nmc_cz, nwalk_cz)
        z_cz    = np.percentile(res_cz.flatchain['z'], [16,50,84])
        scl_cz0 = np.percentile(res_cz.flatchain['Cz0'], [16,50,84])
        scl_cz1 = np.percentile(res_cz.flatchain['Cz1'], [16,50,84])

        zrecom  = z_cz[1]
        Czrec0  = scl_cz0[1]
        Czrec1  = scl_cz1[1]

        xm_s = xm_tmp / (1+zprev) * (1+zrecom)
        fm_s = np.interp(x_cz, xm_s, fm_tmp)

        whtl        = 1/np.square(ey_cz)
        wht2, ypoly = check_line_cz_man(fy_cz, x_cz, whtl, fm_s, zrecom, LW0)
        con_line    = (wht2==0)


        print('\n\n')
        print('Recommended redshift, Cz0 and Cz1, %.5f %.5f %.5f, with chi2/nu=%.3f'%(zrecom, Cz0 * Czrec0, Cz1 * Czrec1, fitc_cz[1]))
        print('\n\n')

        if fzvis==1:
            plt.plot(x_cz, fm_s, 'r', linestyle='-', linewidth=0.5, label='Updated model ($z=%.5f$)'%(zrecom)) # Model based on recomended z.
            plt.plot(x_cz[con_line], fm_s[con_line], color='orange', marker='o', linestyle='', linewidth=3.)

            # Plot lines for reference
            for ll in range(len(LW)):
                conpoly = (x_cz>12000) & (x_cz<16500)
                yline = np.max(ypoly[conpoly])
                yy    = np.arange(yline/1.02, yline*1.1)
                xxpre = yy * 0 + LW[ll] * (1.+zprev)
                xx    = yy * 0 + LW[ll] * (1.+zrecom)
                plt.plot(xxpre, yy/1.02, linewidth=0.5, linestyle='--', color='gray')
                plt.text(LW[ll] * (1.+zprev), yline/1.05, '%s'%(LN[ll]), fontsize=8, color='gray')
                plt.plot(xx, yy, linewidth=0.5, linestyle='-', color='orangered')
                plt.text(LW[ll] * (1.+zrecom), yline, '%s'%(LN[ll]), fontsize=8, color='orangered')

            plt.plot(xbb, fybb, '.r', linestyle='', linewidth=0, zorder=4, label='Obs.(BB)')
            plt.plot(xm_tmp, fm_tmp, color='gray', marker='.', ms=0.5, linestyle='', linewidth=0.5, zorder=4, label='Model')
            xmin, xmax = np.min(x_cz)/1.1,np.max(x_cz)*1.1
            plt.xlim(xmin,xmax)
            plt.ylim(0,yline*1.1)
            plt.xlabel('Wavelength ($\mathrm{\AA}$)')
            plt.ylabel('$F_\\nu$ (arb.)')
            plt.legend(loc=0)

            zzsigma  = ((z_cz[2] - z_cz[0])/2.)/zprev
            zsigma   = np.abs(zprev-zrecom) / (zprev)
            C0sigma  = np.abs(Czrec0-Cz0)/Cz0
            eC0sigma = ((scl_cz0[2]-scl_cz0[0])/2.)/Cz0
            C1sigma  = np.abs(Czrec1-Cz1)/Cz1
            eC1sigma = ((scl_cz1[2]-scl_cz1[0])/2.)/Cz1

            print('Input redshift is %.3f per cent agreement.'%((1.-zsigma)*100))
            print('Error is %.3f per cent.'%(zzsigma*100))
            print('Input Cz0 is %.3f per cent agreement.'%((1.-C0sigma)*100))
            print('Error is %.3f per cent.'%(eC0sigma*100))
            print('Input Cz1 is %.3f per cent agreement.'%((1.-C1sigma)*100))
            print('Error is %.3f per cent.'%(eC1sigma*100))

            plt.show()

            #
            # Ask interactively;
            #
            flag_z = raw_input('Do you want to continue with original redshift, Cz0 and Cz1, %.5f %.5f %.5f? ([y]/n/m) '%(zprev, Cz0, Cz1))
        else:
            flag_z = 'y'


        #################################################
        # Gor for mcmc phase
        #################################################
        if flag_z == 'y' or flag_z == '':
            zrecom  = zprev
            Czrec0  = Cz0
            Czrec1  = Cz1

            ##############################
            # Save fig of z-distribution.
            ##############################
            fig = plt.figure(figsize=(6.5,2.5))
            fig.subplots_adjust(top=0.96, bottom=0.16, left=0.09, right=0.99, hspace=0.15, wspace=0.25)
            ax1 = fig.add_subplot(111)
            n, nbins, patches = ax1.hist(res_cz.flatchain['z'], bins=200, normed=False, color='gray',label='')

            yy = np.arange(0,np.max(n),1)
            xx = yy * 0 + z_cz[1]
            ax1.plot(xx,yy,linestyle='-',linewidth=1,color='orangered',label='$z=%.5f_{-%.5f}^{+%.5f}$\n$C_z0=%.3f$\n$C_z1=%.3f$'%(z_cz[1],z_cz[1]-z_cz[0],z_cz[2]-z_cz[1], Czrec0, Czrec1))
            xx = yy * 0 + z_cz[0]
            ax1.plot(xx,yy,linestyle='--',linewidth=1,color='orangered')
            xx = yy * 0 + z_cz[2]
            ax1.plot(xx,yy,linestyle='--',linewidth=1,color='orangered')
            xx = yy * 0 + zprev
            ax1.plot(xx,yy,linestyle='-',linewidth=1,color='royalblue')
            ax1.set_xlabel('Redshift')
            ax1.set_ylabel('$dn/dz$')
            ax1.legend(loc=0)
            plt.savefig('zprob_' + ID0 + '_PA' + PA0 + '.pdf', dpi=300)
            plt.close()

            ##############################
            print('\n\n')
            print('Input redshift is adopted.')
            print('Starting long journey in MCMC.')
            print('\n\n')
            #################
            # Initialize mm.
            #################
            mm = 0
            # add a noise parameter
            # out.params.add('f', value=1, min=0.001, max=20)
            wht2 = wht


            ################################
            print('Defined Minimizer\n')
            mini = Minimizer(lnprob, out.params)
            print('################\nStarting emcee\n################\n')
            import multiprocessing
            ncpu0 = int(multiprocessing.cpu_count()/2)
            try:
                ncpu = int(inputs['NCPU'])
                if ncpu > ncpu0:
                    print('!!! NCPU is larger than No. of CPU. !!!')
                    #print('Now set to %d'%(ncpu0))
                    #ncpu = ncpu0
            except:
                ncpu = ncpu0
                pass
            print('No. of CPU is set to %d'%(ncpu))
            start_mc = timeit.default_timer()
            res  = mini.emcee(burn=int(nmc/2), steps=nmc, thin=10, nwalkers=nwalk, params=out.params, is_weighted=True, ntemps=ntemp, workers=ncpu)
            stop_mc  = timeit.default_timer()
            tcalc_mc = stop_mc - start_mc
            print('#######################\nMCMC part took %.1f sec\n#######################'%(tcalc_mc))


            #----------- Save pckl file
            #-------- store chain into a cpkl file
            import corner
            burnin   = int(nmc/2)
            savepath = './'
            cpklname = 'chain_' + ID0 + '_PA' + PA0 + '_corner.cpkl'
            savecpkl({'chain':res.flatchain,
                          'burnin':burnin, 'nwalkers':nwalk,'niter':nmc,'ndim':ndim},
                         savepath+cpklname) # Already burn in

            Avmc = np.percentile(res.flatchain['Av'], [16,50,84])
            #Zmc  = np.percentile(res.flatchain['Z'], [16,50,84])

            Avpar = np.zeros((1,3), dtype='float32')
            Avpar[0,:] = Avmc

            out = res
            ##############################
            # Best parameters
            Amc  = np.zeros((len(age),3), dtype='float32')
            Ab   = np.zeros(len(age), dtype='float32')
            Zmc  = np.zeros((len(age),3), dtype='float32')
            Zb   = np.zeros(len(age), dtype='float32')
            NZbest = np.zeros(len(age), dtype='int')
            f0     = fits.open(DIR_TMP + 'ms_' + ID0 + '_PA' + PA0 + '.fits')
            sedpar = f0[1]
            ms     = np.zeros(len(age), dtype='float32')
            for aa in range(len(age)):
                Ab[aa]    = out.params['A'+str(aa)].value
                Amc[aa,:] = np.percentile(res.flatchain['A'+str(aa)], [16,50,84])
                Zb[aa]    = out.params['Z'+str(aa)].value
                Zmc[aa,:] = np.percentile(res.flatchain['Z'+str(aa)], [16,50,84])
                NZbest[aa]= bfnc.Z2NZ(Zb[aa])
                ms[aa]    = sedpar.data['ML_' +  str(NZbest[aa])][aa]

            Avb   = out.params['Av'].value

            ####################
            # MCMC corner plot.
            ####################
            if mcmcplot == 1:
                fig1 = corner.corner(res.flatchain, labels=res.var_names, label_kwargs={'fontsize':16}, quantiles=[0.16, 0.84], show_titles=False, title_kwargs={"fontsize": 14}, truths=list(res.params.valuesdict().values()), plot_datapoints=False, plot_contours=True, no_fill_contours=True, plot_density=False, levels=[0.68, 0.95, 0.997], truth_color='gray', color='#4682b4')
                fig1.savefig('SPEC_' + ID0 + '_PA' + PA0 + '_corner.pdf')
                plt.close()


            #########################
            msmc0 = 0
            for aa in range(len(age)):
                msmc0 += res.flatchain['A'+str(aa)]*ms[aa]

            msmc = np.percentile(msmc0, [16,50,84])

            # Do analysis on MCMC results.
            # Write to file.
            stop  = timeit.default_timer()
            tcalc = stop - start

            wrt.get_param(res, lib_all, zrecom, Czrec0, Czrec1, z_cz[:], scl_cz0[:], scl_cz1[:], fitc[:], tau0=tau0, tcalc=tcalc)


            ##########
            # Plot
            ##########
            if specplot == 1:
                plt.close()
                try:
                    DIR_FILT = inputs['DIR_FILT']
                except:
                    DIR_FILT = './'

                plot_sed_Z(ID0, PA0, Zall, age, tau0=tau0, fil_path=DIR_FILT)
                plot_sfh_pcl2(ID0, PA0, Zall, age, f_comp=ftaucomp, fil_path=DIR_FILT)


            return 0, zrecom, Czrec0, Czrec1

        ###################################################################
        elif flag_z == 'm':
            zrecom = float(raw_input('What is your manual input for redshift? '))
            Czrec0 = float(raw_input('What is your manual input for Cz0? '))
            Czrec1 = float(raw_input('What is your manual input for Cz1? '))
            print('\n\n')
            print('Generate model templates with input redshift and Scale.')
            print('\n\n')
            return 1, zrecom, Czrec0, Czrec1

        else:
            print('\n\n')
            print('Terminated because of redshift estimate.')
            print('Generate model templates with recommended redshift.')
            print('\n\n')

            flag_gen = raw_input('Do you want to make templates with recommended redshift, Cz0, and Cz1 , %.5f %.5f %.5f? ([y]/n) '%(zrecom, Czrec0, Czrec1))
            if flag_gen == 'y' or flag_gen == '':
                #return 1, zrecom, Cz0 * Czrec0, Cz1 * Czrec1
                return 1, zrecom, Czrec0, Czrec1
            else:
                print('\n\n')
                print('There is nothing to do.')
                print('\n\n')
                return 0, zprev, Czrec0, Czrec1
