# The purpose of this code is to figure out Z and redshift (with 1-sig range).
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sys
import os
import fsps

from astropy.io import fits

from astropy.modeling.models import Moffat1D
from astropy.convolution import convolve, convolve_fft

import cosmolopy.distance as cd
import cosmolopy.constants as cc
cosmo = {'omega_M_0' : 0.27, 'omega_lambda_0' : 0.73, 'h' : 0.72}
cosmo = cd.set_omega_k_0(cosmo)
c = 3e18 # speed of light in AA/s
pixelscale = 0.06 # arcsec/pixel
Mpc_cm = 3.08568025e+24 # cm/Mpc
m0set = 25.0

# Custom package
from .function import *
from .function_class import Func
from .basic_func import Basic
from .function_igm import *

col  = ['b', 'skyblue', 'g', 'orange', 'r']

def fit_spec(lm, fobs, eobs, ftmp):
    s = np.sum(fobs*ftmp/eobs**2)/np.sum(ftmp**2/eobs**2)
    chi2 = np.sum(((fobs-s*ftmp)/eobs)**2)
    return chi2, s


def fit_specphot(lm, fobs, eobs, ftmp, fbb, ebb, ltmp_bb, ftmp_bb):
    I1   = np.sum(fobs*ftmp/eobs**2) + np.sum(fbb*ftmp_bb/ebb**2)
    I2   = np.sum(ftmp**2/eobs**2)   + np.sum(ftmp_bb**2/ebb**2)
    s    = I1/I2
    chi2 = np.sum(((fobs-s*ftmp)/eobs)**2) + np.sum(((fbb-s*ftmp_bb)/ebb)**2)
    return chi2, s


def filconv(band0, l0, f0, DIR): # f0 in fnu
    #home = os.path.expanduser('~')
    #DIR  = home + '/Dropbox/FILT/'
    fnu  = np.zeros(len(band0), dtype='float32')
    lcen = np.zeros(len(band0), dtype='float32')
    fwhm = np.zeros(len(band0), dtype='float32')

    for ii in range(len(band0)):
        fd = np.loadtxt(DIR + band0[ii] + '.fil', comments='#')
        lfil = fd[:,1]
        ffil = fd[:,2]

        lmin  = np.min(lfil)
        lmax  = np.max(lfil)
        imin  = 0
        imax  = 0

        lcen[ii] = np.sum(lfil*ffil)/np.sum(ffil)
        lamS,spec = l0, f0                     # Two columns with wavelength and flux density
        lamF,filt = lfil, ffil                 # Two columns with wavelength and response in the range [0,1]
        filt_int  = np.interp(lamS,lamF,filt)  # Interpolate Filter to common(spectra) wavelength axis
        wht       = 1. #/(er1[con_rf])**2

        if len(lamS)>0: #./3*len(x0[con_org]): # Can be affect results.
            I1  = simps(spec/lamS**2*c*filt_int*lamS,lamS)   #Denominator for Fnu
            I2  = simps(filt_int/lamS,lamS)                  #Numerator
            fnu[ii] = I1/I2/c         #Average flux density
        else:
            I1  = 0
            I2  = 0
            fnu[ii] = 0

    return lcen, fnu


def fil_fwhm(band0, DIR): # f0 in fnu
    #
    # FWHM
    #
    fwhm = np.zeros(len(band0), dtype='float32')
    for ii in range(len(band0)):
        fd = np.loadtxt(DIR + band0[ii] + '.fil', comments='#')
        lfil = fd[:,1]
        ffil = fd[:,2]

        fsum = np.sum(ffil)
        fcum = np.zeros(len(ffil), dtype='float32')
        lam0,lam1 = 0,0

        for jj in range(len(ffil)):
            fcum[jj] = np.sum(ffil[:jj])/fsum
            if lam0 == 0 and fcum[jj]>0.05:
                lam0 = lfil[jj]
            if lam1 == 0 and fcum[jj]>0.95:
                lam1 = lfil[jj]

        fwhm[ii] = lam1 - lam0

    return fwhm

from scipy.integrate import simps
def data_int(lmobs, lmtmp, ftmp):
    # lmobs: Observed wavelength.
    # lmtmp, ftmp: Those to be interpolated.
    ftmp_int  = np.interp(lmobs,lmtmp,ftmp) # Interpolate model flux to observed wavelength axis.
    return ftmp_int

def flamtonu(lam, flam):
    Ctmp = lam **2/c * 10**((48.6+m0set)/2.5) #/ delx_org
    fnu  = flam * Ctmp
    return fnu

def fnutolam(lam, fnu):
    Ctmp = lam **2/c * 10**((48.6+m0set)/2.5) #/ delx_org
    flam  = fnu / Ctmp
    return flam

def gauss(x,A,sig):
    return A * np.exp(-0.5*x**2/sig**2)

def moffat(xx, A, x0, gamma, alp):
    yy = A * (1. + (xx-x0)**2/gamma**2)**(-alp)
    return yy

def get_filt(LIBFILT, NFILT):
    #f = open(LIBFILT + '.info', 'r')
    f = open(LIBFILT + '', 'r')



###################################################
### SIMULATION of SPECTRA.
###################################################
def sim_spec(lmin, fin, sn): # wave_obs, wave_temp, flux_temp, sn_obs

    frand = fin * 0
    erand = fin * 0

    for ii in range(len(lmin)):
        if fin[ii]>0 and sn[ii]>0:
            erand[ii] = fin[ii]/sn[ii]
            frand[ii] = np.random.normal(fin[ii],erand[ii],1)
        else:
            erand[ii] = 1e10
            frand[ii] = np.random.normal(fin[ii],erand[ii],1)

    return frand, erand


###################################################
# Make SPECTRA at given z and filter set.
###################################################
def maketemp(inputs, zbest, Z=np.arange(-1.2,0.45,0.1), age=[0.01, 0.1, 0.3, 0.7, 1.0, 3.0], fneb=0):
    #
    # inputs      : Configuration file.
    # zbest(float): Best redshift at this iteration. Templates are generated based on this reshift.
    # Z (array)   : Stellar phase metallicity in logZsun.
    # age (array) : Age, in Gyr.
    # fneb (int)  : flag for adding nebular emissionself.
    #
    nage = np.arange(0,len(age),1)
    fnc  = Func(Z, nage) # Set up the number of Age/ZZ
    bfnc = Basic(Z)

    ID = inputs['ID']
    PA = inputs['PA']
    try:
        DIR_EXTR = inputs['DIR_EXTR']
        if len(DIR_EXTR)==0:
            DIR_EXTR = False
    except:
        DIR_EXTR = False
    DIR_FILT = inputs['DIR_FILT']
    CAT_BB   = inputs['CAT_BB']
    SFILT    = inputs['FILTER'] # filter band string.
    SFILT    = [x.strip() for x in SFILT.split(',')]
    FWFILT   = fil_fwhm(SFILT, DIR_FILT)

    #
    # Tau for MCMC parameter; not as fitting parameters.
    #
    tau0 = inputs['TAU0']
    tau0 = [float(x.strip()) for x in tau0.split(',')]
    #tau0     = [0.1,0.2,0.3] # Gyr

    print('############################')
    print('Making templates at %.4f'%(zbest))
    print('############################')

    ####################################################
    # Get extracted spectra.
    ####################################################
    #
    # Get ascii data.
    #
    #ninp1 = 0
    #ninp2 = 0
    try:
        spec_files = inputs['SPEC_FILE'] # filter band string.
        spec_files = [x.strip() for x in spec_files.split(',')]
        ninp0 = np.zeros(len(spec_files), dtype='int')
        for ff, spec_file in enumerate(spec_files):
            try:
                fd0   = np.loadtxt(DIR_EXTR + spec_file, comments='#')
                lm0tmp= fd0[:,0]
                fobs0 = fd0[:,1]
                eobs0 = fd0[:,2]
                ninp0[ff] = len(lm0tmp)#[con_tmp])
            except Exception:
                print('File, %s, can be open.'%(spec_file))
                pass

        # Constructing arrays.
        lm   = np.zeros(np.sum(ninp0[:]),dtype='float32')
        fobs = np.zeros(np.sum(ninp0[:]),dtype='float32')
        eobs = np.zeros(np.sum(ninp0[:]),dtype='float32')
        fgrs = np.zeros(np.sum(ninp0[:]),dtype='int')  # FLAG for G102/G141.
        for ff, spec_file in enumerate(spec_files):
            try:
                fd0   = np.loadtxt(DIR_EXTR + spec_file, comments='#')
                lm0tmp= fd0[:,0]
                fobs0 = fd0[:,1]
                eobs0 = fd0[:,2]
                for ii1 in range(ninp0[ff]):
                    if ff==0:
                        ii = ii1
                    else:
                        ii = ii1 + np.sum(ninp0[:ff])
                    fgrs[ii] = ff
                    lm[ii]   = lm0tmp[ii1]
                    fobs[ii] = fobs0[ii1]
                    eobs[ii] = eobs0[ii1]

            except Exception:
                pass
    except:
        print('No spec file is provided.')
        pass

    #############################
    # Extracting BB photometry:
    #############################
    fd0 = np.loadtxt(CAT_BB, comments='#')
    id0 = fd0[:,0]
    for ii in range(len(id0)):
        if int(id0[ii]) == int(ID):
            ii0 = ii
            break
    if (int(id0[ii0]) !=  int(ID)):
        return -1

    fd  = fd0[ii0,:]
    id  = fd[0]
    fbb = np.zeros(len(SFILT), dtype='float32')
    ebb = np.zeros(len(SFILT), dtype='float32')
    for ii in range(len(SFILT)):
        fbb[ii] = fd[ii*2+1]
        ebb[ii] = fd[ii*2+2]


    #############################
    # Getting Morphology params.
    #############################
    Amp    = 0
    f_morp = False
    try:
        if inputs['MORP'] == 'moffat' or inputs['MORP'] == 'gauss':
            f_morp = True
            try:
                mor_file = inputs['MORP_FILE']
                fm = np.loadtxt(DIR_EXTR + mor_file, comments='#')
                Amp   = fm[0]
                gamma = fm[1]
                if inputs['MORP'] == 'moffat':
                    alp   = fm[2]
                else:
                    alp   = 0
            except Exception:
                print('Error in reading morphology params.')
                return -1
        else:
            print('MORP Keywords does not match.')
            print('No morphology convolution.')
    except:
        pass

    ############################
    # Template convolution;
    ############################
    try:
        sig_temp = float(inputs['SIG_TEMP'])
    except:
        sig_temp = 50.
        print('Template resolution is unknown.')
        print('Set to %.1f km/s.'%(sig_temp))
    dellam = lm[1] - lm[0] # AA/pix
    R_temp = c/(sig_temp*1e3*1e10)
    sig_temp_pix = np.median(lm) / R_temp / dellam # delta v in pixel;

    # If grism;
    if f_morp:
        print('Templates convolution (intrinsic morphology).')
        if gamma>sig_temp_pix:
            sig_conv = np.sqrt(gamma**2-sig_temp_pix**2)
        else:
            sig_conv = 0
            print('Template resolution is broader than Morphology.')
            print('No convolution is applied to templates.')

        xMof = np.arange(-5, 5.1, .1) # dimension must be even.
        if inputs['MORP'] == 'moffat' and Amp>0 and alp>0:
            LSF = moffat(xMof, Amp, 0, np.sqrt(gamma**2-sig_temp_pix**2), alp)
            print(np.sqrt(gamma**2-sig_temp_pix**2))
            print('Template convolution with Moffat.')
            #print('params are;',Amp, 0, gamma, alp)
        elif inputs['MORP'] == 'gauss':
            sigma = gamma
            LSF = gauss(xMof, Amp, np.sqrt(sigma**2-sig_temp_pix**2))
            print('Template convolution with Gaussian.')
            print('params is sigma;',sigma)
        else:
            print('Something is wrong.')
            return -1
    else: # For slit spectroscopy. To be updated...
        print('Templates convolution (intrinsic velocity).')
        f_disp = False
        try:
            vdisp = float(inputs['VDISP'])
            dellam = lm[1] - lm[0] # AA/pix
            R_disp = c/(vdisp*1e3*1e10)
            vdisp_pix = np.median(lm) / R_disp / dellam # delta v in pixel;
            print('Templates are convolved at %.2f km/s.'%(vdisp))
            sig_conv = np.sqrt(vdisp_pix**2-sig_temp_pix**2)
        except:
            vdisp = 0.
            print('Templates are not convolved.')
            sig_conv = np.sqrt(sig_temp_pix**2)
            pass
        xMof = np.arange(-5, 5.1, .1) # dimension must be even.
        Amp  = 1.
        LSF  = gauss(xMof, Amp, np.sqrt(sigma**2-sig_temp_pix**2))

    ####################################
    # Start generating templates
    ####################################
    DIR_TMP = './templates/'
    f0    = fits.open(DIR_TMP + 'ms.fits')
    mshdu = f0[1]
    col00 = []
    col01 = []
    col02 = []
    for zz in range(len(Z)):
        for pp in range(len(tau0)):
            f1      = fits.open(DIR_TMP + 'spec_all.fits')
            spechdu = f1[1]
            Zbest   = Z[zz]

            Na      = len(age)
            Nz      = 1
            param   = np.zeros((Na, 6), dtype='float32')
            param[:,2] = 1e99
            Ntmp    = 1
            chi2    = np.zeros(Ntmp) + 1e99
            snorm   = np.zeros(Ntmp)
            agebest = np.zeros(Ntmp)
            avbest  = np.zeros(Ntmp)
            age_univ= cd.age(zbest, use_flat=True, **cosmo)

            if zz == 0 and pp == 0:
                lm0    = spechdu.data['wavelength']
                if fneb == 1:
                    spec0 = spechdu.data['efspec_'+str(zz)+'_0_'+str(pp)]
                    #logU  = f1[0].header['logU']
                else:
                    spec0 = spechdu.data['fspec_'+str(zz)+'_0_'+str(pp)]

            lmbest   = np.zeros((Ntmp, len(lm0)), dtype='float32')
            fbest    = np.zeros((Ntmp, len(lm0)), dtype='float32')
            lmbestbb = np.zeros((Ntmp, len(SFILT)), dtype='float32')
            fbestbb  = np.zeros((Ntmp, len(SFILT)), dtype='float32')

            A = np.zeros(Na, dtype='float32') + 1

            spec_mul = np.zeros((Na, len(lm0)), dtype='float32')
            spec_mul_nu = np.zeros((Na, len(lm0)), dtype='float32')
            spec_mul_nu_conv = np.zeros((Na, len(lm0)), dtype='float32')

            ftmp_nu_int = np.zeros((Na, len(lm)), dtype='float32')
            ftmpbb = np.zeros((Na, len(SFILT)), dtype='float32')
            ltmpbb = np.zeros((Na, len(SFILT)), dtype='float32')
            spec_av_tmp = np.zeros((Na, len(lm0)), dtype='float32')

            ms    = np.zeros(Na, dtype='float32')
            Ls    = np.zeros(Na, dtype='float32')
            ms[:] = mshdu.data['ms_'+str(zz)][:] # [:] is necessary.
            Ls[:] = mshdu.data['Ls_'+str(zz)][:]

            for ss in range(Na):
                wave = spechdu.data['wavelength']
                if fneb == 1:
                    spec_mul[ss] = spechdu.data['efspec_'+str(zz)+'_'+str(ss)+'_'+str(pp)]
                else:
                    spec_mul[ss] = spechdu.data['fspec_'+str(zz)+'_'+str(ss)+'_'+str(pp)]

                ###################
                # IGM attenuation.
                ###################
                spec_av_tmp = madau_igm_abs(wave, spec_mul[ss,:],zbest)
                spec_mul_nu[ss,:] = flamtonu(wave, spec_av_tmp)

                if DIR_EXTR:
                    spec_mul_nu_conv[ss,:] = convolve(spec_mul_nu[ss], LSF, boundary='extend')
                else:
                    spec_mul_nu_conv[ss,:] = spec_mul_nu[ss]

                spec_sum = 0*spec_mul[0] # This is dummy file.
                DL = cd.luminosity_distance(zbest, **cosmo) * Mpc_cm # Luminositydistance in cm
                wavetmp = wave*(1.+zbest)
                spec_av  = flamtonu(wavetmp, spec_sum) # Conversion from Flambda to Fnu.
                ftmp_int = data_int(lm, wavetmp, spec_av)

                Lsun = 3.839 * 1e33 #erg s-1
                ftmpbb[ss,:]           *= Lsun/(4.*np.pi*DL**2/(1.+zbest))
                spec_mul_nu_conv[ss,:] *= Lsun/(4.*np.pi*DL**2/(1.+zbest))

                stmp_common = 1e10 # 1 tmp is in 1e10Lsun
                ftmpbb[ss,:]      *= (1./Ls[ss])*stmp_common
                spec_mul_nu_conv[ss,:] *= (1./Ls[ss])*stmp_common
                ms[ss]            *= (1./Ls[ss])*stmp_common # 1 unit template has this mass in [Msolar].

                ftmp_nu_int[ss,:]  = data_int(lm, wavetmp, spec_mul_nu_conv[ss,:])
                ltmpbb[ss,:], ftmpbb[ss,:] = filconv(SFILT, wavetmp, spec_mul_nu_conv[ss,:], DIR_FILT)

                ##########################################
                # Writing out the templates to fits table.
                ##########################################
                if ss == 0 and pp == 0 and zz == 0:
                    # First file
                    nd1    = np.arange(0,len(lm),1)
                    nd3    = np.arange(10000,10000+len(ltmpbb[ss,:]),1)
                    nd_ap  = np.append(nd1,nd3)
                    lm_ap  = np.append(lm, ltmpbb[ss,:])

                    col1   = fits.Column(name='wavelength', format='E', unit='AA', array=lm_ap)
                    col2   = fits.Column(name='colnum', format='K', unit='', array=nd_ap)
                    col00 = [col1, col2]

                    # Second file
                    col3   = fits.Column(name='wavelength', format='E', unit='AA', array=wavetmp)
                    nd     = np.arange(0,len(wavetmp),1)
                    col4   = fits.Column(name='colnum', format='K', unit='', array=nd)
                    col01 = [col3, col4]


                spec_ap = np.append(ftmp_nu_int[ss,:], ftmpbb[ss,:])
                colspec = fits.Column(name='fspec_'+str(zz)+'_'+str(ss)+'_'+str(pp), format='E', unit='Fnu', disp='%s'%(age[ss]), array=spec_ap)
                col00.append(colspec)

                colspec_all = fits.Column(name='fspec_'+str(zz)+'_'+str(ss)+'_'+str(pp), format='E', unit='Fnu', disp='%s'%(age[ss]), array=spec_mul_nu_conv[ss,:])
                col01.append(colspec_all)


            #########################
            # Summarize the ML
            #########################
            if pp == 0:
                colms = fits.Column(name='ML_'+str(zz), format='E', unit='Msun/1e10Lsun', array=ms)
                col02.append(colms)


    #########################
    # Summarize the templates
    #########################
    coldefs_spec = fits.ColDefs(col00)
    hdu = fits.BinTableHDU.from_columns(coldefs_spec)
    hdu.writeto(DIR_TMP + 'spec_' + ID + '_PA' + PA + '.fits', overwrite=True)

    coldefs_spec = fits.ColDefs(col01)
    hdu2 = fits.BinTableHDU.from_columns(coldefs_spec)
    hdu2.writeto(DIR_TMP + 'spec_all_' + ID + '_PA' + PA + '.fits', overwrite=True)

    coldefs_ms = fits.ColDefs(col02)
    hdu3 = fits.BinTableHDU.from_columns(coldefs_ms)
    hdu3.writeto(DIR_TMP + 'ms_' + ID + '_PA' + PA + '.fits', overwrite=True)

    ##########################################
    # For observation.
    # Write out for the Multi-component fitting.
    ##########################################
    lamliml = 0.
    lamlimu = 20000.
    fw = open(DIR_TMP + 'spec_obs_' + ID + '_PA' + PA + '.cat', 'w')
    for ii in range(len(lm)):
        if fgrs[ii]==0: # G102
            if lm[ii]/(1.+zbest) > lamliml and lm[ii]/(1.+zbest) < lamlimu:
                fw.write('%d %.5f %.5e %.5e\n'%(ii, lm[ii], fobs[ii], eobs[ii]))
            else:
                fw.write('%d %.5f 0 1000\n'%(ii, lm[ii]))
        elif fgrs[ii]==1: # G141
            if lm[ii]/(1.+zbest) > lamliml and lm[ii]/(1.+zbest) < lamlimu:
                fw.write('%d %.5f %.5e %.5e\n'%(ii+1000, lm[ii], fobs[ii], eobs[ii]))
            else:
                fw.write('%d %.5f 0 1000\n'%(ii+1000, lm[ii]))

    for ii in range(len(ltmpbb[0,:])):
        if  ebb[ii]>1000:
            fw.write('%d %.5f 0 1000\n'%(ii+10000, ltmpbb[0,ii]))
        else:
            fw.write('%d %.5f %.5e %.5e\n'%(ii+10000, ltmpbb[0,ii], fbb[ii], ebb[ii]))
    fw.close()

    fw = open(DIR_TMP + 'bb_obs_' + ID + '_PA' + PA + '.cat', 'w')
    for ii in range(len(ltmpbb[0,:])):
        if ebb[ii]>1000:
            fw.write('%d %.5f 0 1000 %.1f\n'%(ii+10000, ltmpbb[0,ii], FWFILT[ii]/2.))
        else:
            fw.write('%d %.5f %.5e %.5e %.1f\n'%(ii+10000, ltmpbb[0,ii], fbb[ii], ebb[ii], FWFILT[ii]/2.))
    fw.close()
