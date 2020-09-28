# The purpose of this code is to figure out Z and redshift (with 1-sig range).
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sys
import os

from astropy.io import fits
from scipy.integrate import simps

from astropy.modeling.models import Moffat1D
from astropy.convolution import convolve, convolve_fft

# Custom modules
from .function import *
from .function_igm import *
col  = ['b', 'skyblue', 'g', 'orange', 'r']

def sim_spec(lmin, fin, sn):
    '''
    Purpose:
    ========
    SIMULATION of SPECTRA.
    
    Input:
    ======
    wave_obs, wave_temp, flux_temp, sn_obs
    Return: frand, erand
    '''

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


def maketemp(MB, ebblim=1e10, lamliml=0., lamlimu=20000., ncolbb=10000):
    '''
    Purpose:
    ========
    Make SPECTRA at given z and filter set.
    
    Input:
    ======
    inputs      : Configuration file.
    zbest(float): Best redshift at this iteration. Templates are generated based on this reshift.
    Z (array)   : Stellar phase metallicity in logZsun.
    age (array) : Age, in Gyr.
    fneb (int)  : flag for adding nebular emissionself.
    '''    
    inputs = MB.inputs
    ID = MB.ID #inputs['ID']
    PA = MB.PA #inputs['PA']
    age  = MB.age #=[0.01, 0.1, 0.3, 0.7, 1.0, 3.0]
    nage = MB.nage #np.arange(0,len(age),1)
    Z  = MB.Zall #=np.arange(-1.2,0.45,0.1),
    fneb = MB.fneb
    DIR_TMP = MB.DIR_TMP# './templates/'
    zbest = MB.zgal
    tau0 = MB.tau0

    fnc  = MB.fnc #Func(ID, PA, Z, nage) # Set up the number of Age/ZZ
    bfnc = MB.bfnc #Basic(Z)

    import asdf
    af = asdf.open(DIR_TMP + 'spec_all.asdf')
    mshdu = af['ML']
    spechdu = af['spec']

    # ASDF Big tree;
    # Create header;
    tree = {
        'isochrone': af['isochrone'],
        'library': af['library'],
        'nimf': af['nimf'],
        'version_gsf': af['version_gsf']
    }
    tree_spec = {}
    tree_spec_full = {}
    tree_ML = {}

    try:
        DIR_EXTR = inputs['DIR_EXTR']
        if len(DIR_EXTR)==0:
            DIR_EXTR = False
    except:
        DIR_EXTR = False
    DIR_FILT = inputs['DIR_FILT']
    try:
        CAT_BB_IND = inputs['CAT_BB_IND']
    except:
        CAT_BB_IND = False
    try:
        CAT_BB = inputs['CAT_BB']
    except:
        CAT_BB = False

    try:
        SFILT  = inputs['FILTER'] # filter band string.
        SFILT  = [x.strip() for x in SFILT.split(',')]
        FWFILT = fil_fwhm(SFILT, DIR_FILT)
    except:
        SFILT = []
        FWFILT= []
    if len(FWFILT)==0:
        print('########################')
        print('Filter is not detected!!')
        print('Make sure your \nfilter directory is correct.')
        print('########################')
    try:
        SKIPFILT = inputs['SKIPFILT']
        SKIPFILT = [x.strip() for x in SKIPFILT.split(',')]
    except:
        SKIPFILT = []

    # If FIR data;
    try:
        DFILT   = inputs['FIR_FILTER'] # filter band string.
        DFILT   = [x.strip() for x in DFILT.split(',')]
        DFWFILT = fil_fwhm(DFILT, DIR_FILT)
        CAT_BB_DUST = inputs['CAT_BB_DUST']
        DT0 = float(inputs['TDUST_LOW'])
        DT1 = float(inputs['TDUST_HIG'])
        dDT = float(inputs['TDUST_DEL'])
        f_dust = True
        print('FIR is implemented.')
    except:
        print('No FIR is implemented.')
        f_dust = False
        pass


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
    f_spec = False
    try:
        spec_files = inputs['SPEC_FILE'] #.replace('$ID','%s'%(ID))
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
                print('File, %s/%s, cannot be open.'%(DIR_EXTR,spec_file))
                pass
        # Constructing arrays.
        lm   = np.zeros(np.sum(ninp0[:]),dtype='float64')
        fobs = np.zeros(np.sum(ninp0[:]),dtype='float64')
        eobs = np.zeros(np.sum(ninp0[:]),dtype='float64')
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
                f_spec = True
            except Exception:
                pass
    except:
        print('No spec file is provided.')
        pass

    #############################
    # READ BB photometry from CAT_BB:
    #############################
    from astropy.io import ascii
    if CAT_BB:
        #fd0 = np.loadtxt(CAT_BB, comments='#')
        fd0 = ascii.read(CAT_BB)

        id0 = fd0['id']
        ii0 = np.argmin(np.abs(id0[:]-int(ID)))
        if int(id0[ii0]) !=  int(ID):
            print('Cannot find the column for %d in the input BB catalog!'%(int(ID)))
            return -1
        id  = fd0['id'][ii0]

        fbb = np.zeros(len(SFILT), dtype='float64')
        ebb = np.zeros(len(SFILT), dtype='float64')

        for ii in range(len(SFILT)):
            fbb[ii] = fd0['F%s'%(SFILT[ii])][ii0]
            ebb[ii] = fd0['E%s'%(SFILT[ii])][ii0]

    elif CAT_BB_IND: # if individual photometric catalog; made in get_sdss.py
        fd0 = fits.open(DIR_EXTR + CAT_BB_IND)
        hd0 = fd0[1].header
        bunit_bb = float(hd0['bunit'][:5])
        lmbb0= fd0[1].data['wavelength']
        fbb0 = fd0[1].data['flux'] * bunit_bb
        ebb0 = 1/np.sqrt(fd0[1].data['inverse_variance']) * bunit_bb

        unit  = 'nu'
        try:
            unit = inputs['UNIT_SPEC']
        except:
            print('No param for UNIT_SPEC is found.')
            print('BB flux unit is assumed to Fnu.')
            pass

        if unit == 'lambda':
            print('#########################')
            print('Changed BB from Flam to Fnu')
            snbb0= fbb0/ebb0
            fbb  = flamtonu(lmbb0, fbb0)
            ebb  = fbb/snbb0
        else:
            snbb0= fbb0/ebb0
            fbb  = fbb0
            ebb  = ebb0

    else:
        fbb = np.zeros(len(SFILT), dtype='float64')
        ebb = np.zeros(len(SFILT), dtype='float64')
        for ii in range(len(SFILT)):
            fbb[ii] = 0
            ebb[ii] = -99 #1000

    # Dust flux;
    if f_dust:
        fdd = ascii.read(CAT_BB_DUST)
        try:
            id0 = fdd['id']
            ii0 = np.argmin(np.abs(id0[:]-int(ID)))
            if int(id0[ii0]) != int(ID):
                return -1
        except:
            return -1
        id = fdd['id']

        fbb_d = np.zeros(len(DFILT), dtype='float64')
        ebb_d = np.zeros(len(DFILT), dtype='float64')
        for ii in range(len(DFILT)):
            fbb_d[ii] = fdd['F%s'%(DFILT[ii])][ii0]
            ebb_d[ii] = fdd['E%s'%(DFILT[ii])][ii0]

    #############################
    # Getting Morphology params.
    #############################
    Amp    = 0
    f_morp = False
    if f_spec:
        try:
            if inputs['MORP'] == 'moffat' or inputs['MORP'] == 'gauss':
                f_morp = True
                try:
                    mor_file = inputs['MORP_FILE'].replace('$ID','%s'%(ID))
                    #fm = np.loadtxt(DIR_EXTR + mor_file, comments='#')
                    from astropy.io import ascii
                    fm    = ascii.read(DIR_EXTR + mor_file)
                    Amp   = fm['A']
                    gamma = fm['gamma']
                    if inputs['MORP'] == 'moffat':
                        alp   = fm['alp']
                    else:
                        alp   = 0
                except Exception:
                    print('Error in reading morphology params.')
                    print('No morphology convolution.')
                    pass
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

        #
        sig_inst = 0 #65 #km/s for Manga

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
                print('Template convolution with Moffat.')
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
                #R_disp = c/(vdisp*1e3*1e10)
                R_disp = c/(np.sqrt(vdisp**2-sig_inst**2)*1e3*1e10)
                vdisp_pix = np.median(lm) / R_disp / dellam # delta v in pixel;
                print('Templates are convolved at %.2f km/s.'%(vdisp))
                if vdisp_pix-sig_temp_pix>0:
                    sig_conv = np.sqrt(vdisp_pix**2-sig_temp_pix**2)
                else:
                    sig_conv = 0
            except:
                vdisp = 0.
                print('Templates are not convolved.')
                sig_conv = 0 #np.sqrt(sig_temp_pix**2)
                pass
            xMof = np.arange(-5, 5.1, .1) # dimension must be even.
            Amp  = 1.
            LSF  = gauss(xMof, Amp, sig_conv)
    else:
        lm = []


    ####################################
    # Start generating templates
    ####################################
    col00 = []
    col01 = []
    col02 = []
    for zz in range(len(Z)):
        for pp in range(len(tau0)):
            Zbest   = Z[zz]
            Na      = len(age)
            Nz      = 1
            param   = np.zeros((Na, 6), dtype='float64')
            param[:,2] = 1e99
            Ntmp    = 1
            chi2    = np.zeros(Ntmp) + 1e99
            snorm   = np.zeros(Ntmp)
            agebest = np.zeros(Ntmp)
            avbest  = np.zeros(Ntmp)
            age_univ= MB.cosmo.age(zbest).value #, use_flat=True, **cosmo)

            if zz == 0 and pp == 0:
                lm0    = spechdu['wavelength']

            lmbest   = np.zeros((Ntmp, len(lm0)), dtype='float64')
            fbest    = np.zeros((Ntmp, len(lm0)), dtype='float64')
            lmbestbb = np.zeros((Ntmp, len(SFILT)), dtype='float64')
            fbestbb  = np.zeros((Ntmp, len(SFILT)), dtype='float64')

            A = np.zeros(Na, dtype='float64') + 1

            spec_mul = np.zeros((Na, len(lm0)), dtype='float64')
            spec_mul_nu = np.zeros((Na, len(lm0)), dtype='float64')
            spec_mul_nu_conv = np.zeros((Na, len(lm0)), dtype='float64')

            ftmpbb = np.zeros((Na, len(SFILT)), dtype='float64')
            ltmpbb = np.zeros((Na, len(SFILT)), dtype='float64')

            ftmp_nu_int = np.zeros((Na, len(lm)), dtype='float64')
            spec_av_tmp = np.zeros((Na, len(lm)), dtype='float64')

            ms    = np.zeros(Na, dtype='float64')
            Ls    = np.zeros(Na, dtype='float64')
            ms[:] = mshdu['ms_'+str(zz)][:] # [:] is necessary.
            Ls[:] = mshdu['Ls_'+str(zz)][:]
            Fuv   = np.zeros(Na, dtype='float64')

            for ss in range(Na):
                wave = spechdu['wavelength']
                if fneb == 1:
                    spec_mul[ss] = spechdu['efspec_'+str(zz)+'_'+str(ss)+'_'+str(pp)]
                else:
                    spec_mul[ss] = spechdu['fspec_'+str(zz)+'_'+str(ss)+'_'+str(pp)]

                ###################
                # IGM attenuation.
                ###################
                spec_av_tmp = madau_igm_abs(wave, spec_mul[ss,:], zbest, cosmo=MB.cosmo)
                spec_mul_nu[ss,:] = flamtonu(wave, spec_av_tmp)
                if len(lm)>0:
                    try:
                        spec_mul_nu_conv[ss,:] = convolve(spec_mul_nu[ss], LSF, boundary='extend')
                    except:
                        spec_mul_nu_conv[ss,:] = spec_mul_nu[ss]
                        if zz==0 and ss==0:
                            print('Kernel is too small. No convolution.')
                else:
                    spec_mul_nu_conv[ss,:] = spec_mul_nu[ss]

                spec_sum = 0*spec_mul[0] # This is dummy file.
                DL = MB.cosmo.luminosity_distance(zbest).value * MB.Mpc_cm # Luminositydistance in cm
                wavetmp = wave*(1.+zbest)
                #spec_av  = flamtonu(wavetmp, spec_sum) # Conversion from Flambda to Fnu.
                #ftmp_int = data_int(lm, wavetmp, spec_av)

                Lsun = 3.839 * 1e33 #erg s-1
                stmp_common = 1e10 # so 1 template is in 1e10Lsun

                spec_mul_nu_conv[ss,:] *= Lsun/(4.*np.pi*DL**2/(1.+zbest))
                spec_mul_nu_conv[ss,:] *= (1./Ls[ss])*stmp_common # in unit of erg/s/Hz/cm2/ms[ss].
                ms[ss] *= (1./Ls[ss])*stmp_common # M/L; 1 unit template has this mass in [Msolar].

                if f_spec:
                    ftmp_nu_int[ss,:] = data_int(lm, wavetmp, spec_mul_nu_conv[ss,:])
                ltmpbb[ss,:], ftmpbb[ss,:] = filconv(SFILT, wavetmp, spec_mul_nu_conv[ss,:], DIR_FILT)

                # UV magnitude;
                #print('%s AA is used as UV reference.'%(xm_tmp[iiuv]))
                #print(ms[ss], (Lsun/(4.*np.pi*DL**2/(1.+zbest))))
                #print('m-M=',5*np.log10(DL/Mpc_cm*1e6/10))

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
                    col00  = [col1, col2]
                    # ASDF
                    tree_spec.update({'wavelength':lm_ap})
                    tree_spec.update({'colnum':nd_ap})

                    # Second file
                    col3   = fits.Column(name='wavelength', format='E', unit='AA', array=wavetmp)
                    nd     = np.arange(0,len(wavetmp),1)
                    col4   = fits.Column(name='colnum', format='K', unit='', array=nd)
                    col01 = [col3, col4]
                    # ASDF
                    tree_spec_full.update({'wavelength':wavetmp})
                    tree_spec_full.update({'colnum':nd})

                spec_ap = np.append(ftmp_nu_int[ss,:], ftmpbb[ss,:])
                colspec = fits.Column(name='fspec_'+str(zz)+'_'+str(ss)+'_'+str(pp), format='E', unit='Fnu', array=spec_ap)#, disp='%s'%(age[ss])
                col00.append(colspec)
                # ASDF
                tree_spec.update({'fspec_'+str(zz)+'_'+str(ss)+'_'+str(pp): spec_ap})

                colspec_all = fits.Column(name='fspec_'+str(zz)+'_'+str(ss)+'_'+str(pp), format='E', unit='Fnu', array=spec_mul_nu_conv[ss,:])#, disp='%s'%(age[ss])
                col01.append(colspec_all)
                # ASDF
                tree_spec_full.update({'fspec_'+str(zz)+'_'+str(ss)+'_'+str(pp): spec_mul_nu_conv[ss,:]})

            #########################
            # Summarize the ML
            #########################
            if pp == 0:
                colms = fits.Column(name='ML_'+str(zz), format='E', unit='Msun/1e10Lsun', array=ms)
                col02.append(colms)
                # ASDF
                tree_ML.update({'ML_'+str(zz): ms})


    #########################
    # Summarize the templates
    #########################
    '''
    coldefs_spec = fits.ColDefs(col00)
    hdu = fits.BinTableHDU.from_columns(coldefs_spec)
    hdu.writeto(DIR_TMP + 'spec_' + ID + '_PA' + PA + '.fits', overwrite=True)

    coldefs_spec = fits.ColDefs(col01)
    hdu2 = fits.BinTableHDU.from_columns(coldefs_spec)
    hdu2.writeto(DIR_TMP + 'spec_all_' + ID + '_PA' + PA + '.fits', overwrite=True)

    coldefs_ms = fits.ColDefs(col02)
    hdu3 = fits.BinTableHDU.from_columns(coldefs_ms)
    hdu3.writeto(DIR_TMP + 'ms_' + ID + '_PA' + PA + '.fits', overwrite=True)
    '''

    tree.update({'spec' : tree_spec})
    tree.update({'spec_full' : tree_spec_full})
    tree.update({'ML' : tree_ML})

    ######################
    # Add dust component;
    ######################
    if f_dust:
        tree_spec_dust = {}
        tree_spec_dust_full = {}

        if DT0 == DT1:
            Temp = [DT0]
        else:
            Temp = np.arange(DT0,DT1,dDT)

        dellam_d = 1e3
        lambda_d = np.arange(1e4,1e7,dellam_d) # RF wavelength, in AA. #* (1.+zbest) # 1um to 1000um;

        # c in AA/s.
        kb = 1.380649e-23 # Boltzmann constant, in J/K
        hp = 6.62607015e-34 # Planck constant, in J*s

        # from Eq.3 of Bianchi 13
        kabs0 = 4.0 # in cm2/g
        beta_d= 2.08 #
        lam0  = 250.*1e4 # mu m to AA
        #kappa *= (1e8)**2 # AA2/g

        from astropy.modeling import models
        from astropy import units as u
        for tt in range(len(Temp)):
            if tt == 0:
                # For full;
                nd_d  = np.arange(0,len(lambda_d),1)

                # ASDF
                tree_spec_dust_full.update({'wavelength': lambda_d*(1.+zbest)})
                tree_spec_dust_full.update({'colnum': nd_d})

            '''
            nu_d = c / lambda_d # 1/s = Hz
            nu_d_hp = nu_d * hp # This is recommended, as BT_nu equation may cause overflow.
            BT_nu = 2 * hp * nu_d[:]**3 / c**2 / (np.exp(nu_d_hp/(kb*Temp[tt]))-1) # J*s * (1/s)^3 / (AA/s)^2 / sr = J / AA^2 / sr = J/s/AA^2/Hz/sr.
            # if optically thin;
            #kappa = nu_d ** beta_d
            fnu_d = 1.0 / (4.*np.pi*DL**2/(1.+zbest)) * kappa * BT_nu # 1/cm2 * AA2/g * J/s/AA^2/Hz/sr = J/s/cm^2/Hz/g/sr
            fnu_d *= 1.989e+33 # J/s/cm^2/Hz/Msun/sr; i.e. 1 flux is in 1Msun
            fnu_d *= 1e7 # erg/s/cm^2/Hz/Msun/sr.
            '''

            bb = models.BlackBody(temperature=Temp[tt]*u.K)
            wav = lambda_d * u.AA
            BT_nu = bb(wav) # erg/Hz/s/sr/cm2

            kappa = kabs0 * (lam0/wav)**beta_d # cm2/g
            
            # if optically thin;
            #kappa = nu_d ** beta_d
            fnu_d = (1+zbest)/DL**2 * kappa * BT_nu # 1/cm2 * cm2/g * erg/Hz/s/sr/cm2 = erg/s/cm^2/Hz/g/sr
            fnu_d *= 1.989e+33 # erg/s/cm^2/Hz/Msun/sr; i.e. 1 flux is in 1Msun

            '''
            # Redefine;
            nu_d = c / wav  * u.Hz
            beta = 2.0
            nu_1 = c / lam0 # 1/s
            t_nu = (nu_d / nu_1)**beta
            k870 = 0.05 # m2/kg
            nu_870 = 343 * 1e9 # in Hz

            Snu0 = (1+zbest)/DL**2 * k870 * (nu_d / nu_870)**beta * flux # 1/cm2 * m2/kg * J/s/AA^2/Hz/sr. = 10000 * J/s/AA^2/Hz/sr/kg
            Snu0 *= 1.989e+30 * 10000 # J/s/AA^2/Hz/sr/Msun
            Snu0 *= 1e+7 # erg/s/AA^2/Hz/Msun/sr.
            Snu0 *= (1e8)**2 # erg/s/cm^2/Hz/Msun/sr.
            '''

            if True:
                print('Somehow, crazy scale is required for FIR normalization...')
                fnu_d *= 1e30
            
            if False:
                flam_d = fnutolam(wav, fnu_d)
                plt.plot(wav, flam_d, '.-')
                plt.xlim(1e4, 1e6)
                plt.show()
                hoge

            
            #colspec_d = fits.Column(name='fspec_'+str(tt), format='E', unit='Fnu(erg/s/cm^2/Hz/Msun)', disp='%.2f'%(Temp[tt]), array=fnu_d)
            #col03.append(colspec_d)
            # ASDF
            fnu_d = fnu_d.value
            tree_spec_dust_full.update({'fspec_'+str(tt): fnu_d})

            # Convolution;
            #ltmpbb_d, ftmpbb_d = filconv(DFILT,lambda_d*(1.+zbest),fnu_d,DIR_FILT)
            ALLFILT = np.append(SFILT,DFILT)
            ltmpbb_d, ftmpbb_d = filconv(ALLFILT,lambda_d*(1.+zbest),fnu_d,DIR_FILT)
            if False:
                #plt.plot(nu_d/1e9/(1.+zbest),fnu_d)
                #nubb_d = c / ltmpbb_d
                #plt.plot(nubb_d/1e9, ftmpbb_d, 'x')
                plt.plot(lambda_d/1e4,fnu_d)
                plt.plot(lambda_d*(1.+zbest)/1e4,fnu_d)
                plt.plot(ltmpbb_d/1e4, ftmpbb_d, 'x')
                plt.show()
            if tt == 0:
                # For conv;
                col3   = fits.Column(name='wavelength', format='E', unit='AA', array=ltmpbb_d)
                nd_db  = np.arange(0,len(ltmpbb_d),1)
                col4   = fits.Column(name='colnum', format='K', unit='', array=nd_db)
                col04 = [col3, col4]
                # ASDF
                tree_spec_dust.update({'wavelength': ltmpbb_d})
                tree_spec_dust.update({'colnum': nd_db})

            #colspec_db = fits.Column(name='fspec_'+str(tt), format='E', unit='Fnu', disp='%.2f'%(Temp[tt]), array=ftmpbb_d)
            #col04.append(colspec_db)
            tree_spec_dust.update({'fspec_'+str(tt): ftmpbb_d})

        '''
        coldefs_d = fits.ColDefs(col03)
        hdu4 = fits.BinTableHDU.from_columns(coldefs_d)
        hdu4.writeto(DIR_TMP + 'spec_dust_all_' + ID + '_PA' + PA + '.fits', overwrite=True)

        coldefs_db = fits.ColDefs(col04)
        hdu5 = fits.BinTableHDU.from_columns(coldefs_db)
        hdu5.writeto(DIR_TMP + 'spec_dust_' + ID + '_PA' + PA + '.fits', overwrite=True)
        '''
        tree.update({'spec_dust' : tree_spec_dust})
        tree.update({'spec_dust_full' : tree_spec_dust_full})
        print('dust updated.')

    # Save;
    af = asdf.AsdfFile(tree)
    af.write_to(DIR_TMP + 'spec_all_' + ID + '_PA' + PA + '.asdf', all_array_compression='zlib')


    ##########################################
    # For observation.
    # Write out for the Multi-component fitting.
    ##########################################
    fw = open(DIR_TMP + 'spec_obs_' + ID + '_PA' + PA + '.cat', 'w')
    fw.write('# BB data (>%d) in this file are not used in fitting.\n'%(ncolbb))
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
        if SFILT[ii] in SKIPFILT:# data point to be skiped;
            fw.write('%d %.5f %.5e %.5e\n'%(ii+ncolbb, ltmpbb[0,ii], 0.0, fbb[ii]))
            #fw.write('%d %.5f %.5e %.5e\n'%(ii+ncolbb, ltmpbb[0,ii], 0.0, 1000))
        elif  ebb[ii]>ebblim:
            fw.write('%d %.5f 0 1000\n'%(ii+ncolbb, ltmpbb[0,ii]))
        else:
            fw.write('%d %.5f %.5e %.5e\n'%(ii+ncolbb, ltmpbb[0,ii], fbb[ii], ebb[ii]))

    fw.close()
    fw = open(DIR_TMP + 'spec_dust_obs_' + ID + '_PA' + PA + '.cat', 'w')
    if f_dust:
        nbblast = len(ltmpbb[0,:])
        for ii in range(len(ebb_d[:])):
            if  ebb_d[ii]>ebblim:
                fw.write('%d %.5f 0 1000\n'%(ii+ncolbb+nbblast, ltmpbb_d[ii+nbblast]))
            else:
                fw.write('%d %.5f %.5e %.5e\n'%(ii+ncolbb+nbblast, ltmpbb_d[ii+nbblast], fbb_d[ii], ebb_d[ii]))
    fw.close()

    # BB phot
    fw     = open(DIR_TMP + 'bb_obs_' + ID + '_PA' + PA + '.cat', 'w')
    fw_rem = open(DIR_TMP + 'bb_obs_' + ID + '_PA' + PA + '_removed.cat', 'w')
    for ii in range(len(ltmpbb[0,:])):
        if SFILT[ii] in SKIPFILT:# data point to be skiped;
            fw.write('%d %.5f %.5e %.5e %.1f\n'%(ii+ncolbb, ltmpbb[0,ii], 0.0, fbb[ii], FWFILT[ii]/2.))
            fw_rem.write('%d %.5f %.5e %.5e %.1f\n'%(ii+ncolbb, ltmpbb[0,ii], fbb[ii], ebb[ii], FWFILT[ii]/2.))
        elif ebb[ii]>ebblim:
            fw.write('%d %.5f 0 1000 %.1f\n'%(ii+ncolbb, ltmpbb[0,ii], FWFILT[ii]/2.))
        elif ebb[ii]<=0:
            fw.write('%d %.5f 0 -99 %.1f\n'%(ii+ncolbb, ltmpbb[0,ii], FWFILT[ii]/2.))
        else:
            fw.write('%d %.5f %.5e %.5e %.1f\n'%(ii+ncolbb, ltmpbb[0,ii], fbb[ii], ebb[ii], FWFILT[ii]/2.))
    fw.close()
    fw_rem.close()

    # Dust
    fw = open(DIR_TMP + 'bb_dust_obs_' + ID + '_PA' + PA + '.cat', 'w')
    if f_dust:
        for ii in range(len(ebb_d[:])):
            if  ebb_d[ii]>ebblim:
                fw.write('%d %.5f 0 1000 %.1f\n'%(ii+ncolbb+nbblast, ltmpbb_d[ii+nbblast], DFWFILT[ii]/2.))
            else:
                fw.write('%d %.5f %.5e %.5e %.1f\n'%(ii+ncolbb+nbblast, ltmpbb_d[ii+nbblast], fbb_d[ii], ebb_d[ii], DFWFILT[ii]/2.))
    fw.close()
