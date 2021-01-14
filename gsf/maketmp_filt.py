# The purpose of this code is to figure out Z and redshift (with 1-sig range).
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sys
import os
from scipy.integrate import simps
import asdf

from astropy.io import fits,ascii
from astropy.modeling.models import Moffat1D
from astropy.convolution import convolve, convolve_fft

# Custom modules
from .function import *
from .function_igm import *
col  = ['b', 'skyblue', 'g', 'orange', 'r']


def get_spectrum_draine(lambda_d, DL, zbest, numin, numax, ndmodel, DIR_DUST='/Users/tmorishita/Downloads/DL07spec/', phi=0.055):
    '''
    Purpose:
    ========

    Input:
    ======
    lambda_d : Wavelength array, in AA.
    phi (default: 0.055): Eq.34 of Draine & Li 2007.

    Return:
    =======
    Interpolated dust emission in Fnu of m0=25.0. In units of Fnu/Msun

    Ref:
    ====
    umins = ['0.10', '0.15', '0.20', '0.30', '0.40', '0.50', '0.70', '0.80', '1.00', '1.20',\
            '1.50', '2.00', '2.50', '3.00', '4.00', '5.00', '7.00', '8.00', '10.0', '12.0', '15.0',\
            '20.0', '25.0']
    umaxs = ['1e3', '1e4', '1e5', '1e6', '1e7']

    '''
    from .function import fnutonu
    import scipy.interpolate as interpolate

    Htokg = 1.66054e-27 # kg/H
    kgtomsun = 1.989e+30 # kg/Msun
    MsunperH = Htokg / kgtomsun # Msun/H

    Jytoerg = 1e-23 # erg/s/cm2/Hz / Jy
    c = 3e18
    Mpc_cm = 3.08568025e+24

    umins = ['0.10', '0.15', '0.20', '0.30', '0.40', '0.50', '0.70', '0.80', '1.00', '1.20',\
            '1.50', '2.00', '2.50', '3.00', '4.00', '5.00', '7.00', '8.00', '12.0', '15.0',\
            '20.0', '25.0']
    umaxs = ['1e3', '1e4', '1e5', '1e6', '1e7']
        
    dust_model = DIR_DUST+'draine07_models.txt'
    fd_model = ascii.read(dust_model)

    umin = umins[numin]
    umax = umaxs[numax]
    dmodel = fd_model['name'][ndmodel]

    # See README of Draine's table.
    #dU = float(umin)/100.
    #U = np.arange(float(umin), float(umax), dU)
    #Umean = np.mean(U)
    #print(Umean)

    gamma = 0.01
    Umean = (1-gamma) * float(umin) + (gamma * float(umin) * np.log(float(umax)/float(umin))) / (1-float(umin)/float(umax))
    #print(Umean)

    #try:
    if True:
        #if dmodel == 'MW3.1_60':
        if ndmodel == 6 or ndmodel == 1:
            data_start = 55
        else:
            data_start = 36

        file_dust = DIR_DUST + 'U%s/U%s_%s_%s.txt'%(umin, umin, umax, dmodel)
        print(file_dust)
        fd = ascii.read(file_dust, data_start=data_start)

        wave = fd['col1'] # in mu m.
        flux = fd['col2'] # erg/s H-1
        flux_dens = fd['col3'] # j_nu: Jy cm2 sr-1 H-1
        
        fobs = flux_dens * Jytoerg / (4.*np.pi*DL**2/(1.+zbest)) / MsunperH
        # Jy cm2 sr-1 H-1 * erg/s/cm2/Hz / Jy / (cm2 * sr) / (Msun/H) = erg/s/cm2/Hz / Msun

        freq = c / (wave*1e4) # 1/Hz

        ftot = np.sum(flux/ MsunperH) # erg/s H-1 / (Msun/H) = erg/s/Msun
        #Mh = ftot * phi # erg/s/Msun * g/(erg/s) = g/Msun

        # Get Mdust to MH2 ratio;
        #ftot2 = np.sum(flux * freq)
        #MdtoMh = phi / Umean * ftot2 / (Htokg*1e3) # g/(erg/s)/H / 1 * erg/s/Msun / g * Msun/H = 1/Msun 
        #print(MdtoMh)
        MdtoMh = 0.01 #1.0
        Mdust = 1.0 * MdtoMh #* Mh * kgtomsun * mh # Msun/template
 
        # Then;
        fnu = fnutonu(fobs) / Mdust # Flux density per 1Msun for dust.

        fint = interpolate.interp1d(wave*1e4, fnu, kind='nearest', fill_value="extrapolate")
        yy_s = fint(lambda_d)
        con_yys = (lambda_d<1e4) # Interpolation cause some error??
        yy_s[con_yys] = 0

    #except:
    #    print('Something is wrong.',file_dust)
    #    yy_s = lambda_d * 0

    return yy_s


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


def check_library(MB, af):
    '''
    Purpose:
    ========
    Check library if it has a consistency setup as input file.

    Return:
    =======
    True is no problem. 
    '''

    # Z needs special care in z0 script, to avoid Zfix.
    if False:
        Zmax_tmp, Zmin_tmp = float(MB.inputs['ZMAX']), float(MB.inputs['ZMIN'])
        delZ_tmp = float(MB.inputs['DELZ'])
        if Zmax_tmp == Zmin_tmp or delZ_tmp==0:
            delZ_tmp = 0.0001
        Zall = np.arange(Zmin_tmp, Zmax_tmp+delZ_tmp, delZ_tmp) # in logZsun
    else:
        Zall = MB.Zall
    
    flag = True
    # Matallicity:
    for aa in range(len(Zall)):
        if Zall[aa] != af['Z%d'%(aa)]:
            print('Z:', Zall[aa], af['Z%d'%(aa)])
            flag = False
    # Age:
    for aa in range(len(MB.age)):
        if MB.age[aa] != af['age%d'%(aa)]:
            print('age:', MB.age[aa], af['age%d'%(aa)])
            flag = False

    # Tau (e.g. ssp/csp):
    for aa in range(len(MB.tau0)):
        if MB.tau0[aa] != af['tau0%d'%(aa)]:
            print('tau0:', MB.tau0[aa], af['tau0%d'%(aa)])
            flag = False

    # IMF:
    if MB.nimf != af['nimf']:
        print('nimf:', MB.nimf, af['nimf'])
        flag = False

    return flag


def maketemp(MB, ebblim=1e10, lamliml=0., lamlimu=50000., ncolbb=10000, tau_lim=0.001):
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
    ID = MB.ID
    PA = MB.PA
    age = MB.age
    nage = MB.nage
    Z = MB.Zall
    fneb = MB.fneb
    DIR_TMP = MB.DIR_TMP
    zbest = MB.zgal
    tau0 = MB.tau0
    fnc = MB.fnc
    bfnc = MB.bfnc

    af = asdf.open(DIR_TMP + 'spec_all.asdf')
    mshdu = af['ML']
    spechdu = af['spec']

    # Consistency check:
    flag = check_library(MB, af)
    if not flag:
        print('\n!!!\nThere is inconsistency in z0 library and input file. Exiting.\n!!!\n')
        sys.exit()

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
        print('FIR is implemented.\n')
    except:
        print('No FIR is implemented.\n')
        f_dust = False
        pass


    print('############################')
    print('Making templates at z=%.4f'%(zbest))
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
        lm   = np.zeros(np.sum(ninp0[:]),dtype='float')
        fobs = np.zeros(np.sum(ninp0[:]),dtype='float')
        eobs = np.zeros(np.sum(ninp0[:]),dtype='float')
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
    if CAT_BB:
        #fd0 = np.loadtxt(CAT_BB, comments='#')
        fd0 = ascii.read(CAT_BB)

        id0 = fd0['id']
        ii0 = np.argmin(np.abs(id0[:]-int(ID)))
        if int(id0[ii0]) !=  int(ID):
            print('Cannot find the column for [ID: %d] in the input BB catalog!'%(int(ID)))
            return -1
        id = fd0['id'][ii0]

        fbb = np.zeros(len(SFILT), dtype='float')
        ebb = np.zeros(len(SFILT), dtype='float')

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
        fbb = np.zeros(len(SFILT), dtype='float')
        ebb = np.zeros(len(SFILT), dtype='float')
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

        fbb_d = np.zeros(len(DFILT), dtype='float')
        ebb_d = np.zeros(len(DFILT), dtype='float')
        for ii in range(len(DFILT)):
            fbb_d[ii] = fdd['F%s'%(DFILT[ii])][ii0]
            ebb_d[ii] = fdd['E%s'%(DFILT[ii])][ii0]

    #############################
    # Getting Morphology params.
    #############################
    Amp = 0
    f_morp = False
    if f_spec:
        try:
            if inputs['MORP'] == 'moffat' or inputs['MORP'] == 'gauss':
                f_morp = True
                try:
                    mor_file = inputs['MORP_FILE'].replace('$ID','%s'%(ID))
                    #fm = np.loadtxt(DIR_EXTR + mor_file, comments='#')
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
            Zbest = Z[zz]
            Na = len(age)
            Ntmp = 1
            age_univ= MB.cosmo.age(zbest).value #, use_flat=True, **cosmo)

            if zz == 0 and pp == 0:
                lm0 = spechdu['wavelength']

            lmbest = np.zeros((Ntmp, len(lm0)), dtype='float')
            fbest = np.zeros((Ntmp, len(lm0)), dtype='float')
            lmbestbb = np.zeros((Ntmp, len(SFILT)), dtype='float')
            fbestbb = np.zeros((Ntmp, len(SFILT)), dtype='float')

            A = np.zeros(Na, dtype='float') + 1

            spec_mul = np.zeros((Na, len(lm0)), dtype='float')
            spec_mul_nu = np.zeros((Na, len(lm0)), dtype='float')
            spec_mul_nu_conv = np.zeros((Na, len(lm0)), dtype='float')

            ftmpbb = np.zeros((Na, len(SFILT)), dtype='float')
            ltmpbb = np.zeros((Na, len(SFILT)), dtype='float')

            ftmp_nu_int = np.zeros((Na, len(lm)), dtype='float')
            spec_av_tmp = np.zeros((Na, len(lm)), dtype='float')

            ms = np.zeros(Na, dtype='float')
            Ls = np.zeros(Na, dtype='float')
            ms[:] = mshdu['ms_'+str(zz)][:] # [:] is necessary.
            Ls[:] = mshdu['Ls_'+str(zz)][:]
            Fuv = np.zeros(Na, dtype='float')

            for ss in range(Na):
                wave = spechdu['wavelength']
                if fneb == 1:
                    spec_mul[ss] = spechdu['efspec_'+str(zz)+'_'+str(ss)+'_'+str(pp)]
                else:
                    spec_mul[ss] = spechdu['fspec_'+str(zz)+'_'+str(ss)+'_'+str(pp)]

                ###################
                # IGM attenuation.
                ###################
                f_IGM = True
                if f_IGM:
                    spec_av_tmp = madau_igm_abs(wave, spec_mul[ss,:], zbest, cosmo=MB.cosmo)
                else:
                    spec_av_tmp = spec_mul[ss,:]

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

                Lsun = 3.839 * 1e33 #erg s-1
                stmp_common = 1e10 # so 1 template is in 1e10Lsun

                spec_mul_nu_conv[ss,:] *= Lsun/(4.*np.pi*DL**2/(1.+zbest))
                spec_mul_nu_conv[ss,:] *= (1./Ls[ss])*stmp_common # in unit of erg/s/Hz/cm2/ms[ss].
                ms[ss] *= (1./Ls[ss])*stmp_common # M/L; 1 unit template has this mass in [Msolar].

                if f_spec:
                    ftmp_nu_int[ss,:] = data_int(lm, wavetmp, spec_mul_nu_conv[ss,:])
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
                # ASDF
                tree_spec.update({'fspec_'+str(zz)+'_'+str(ss)+'_'+str(pp): spec_ap})

                # ASDF
                tree_spec_full.update({'fspec_'+str(zz)+'_'+str(ss)+'_'+str(pp): spec_mul_nu_conv[ss,:]})

            #########################
            # Summarize the ML
            #########################
            if pp == 0:
                colms = fits.Column(name='ML_'+str(zz), format='E', unit='Msun/%.1eLsun'%(stmp_common), array=ms)
                col02.append(colms)
                # ASDF
                tree_ML.update({'ML_'+str(zz): ms})


    #########################
    # Summarize the templates
    #########################
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
        lambda_d = np.arange(1e3, 1e7, dellam_d) # RF wavelength, in AA. #* (1.+zbest) # 1um to 1000um; This has to be wide enough, to cut dust contribution at <1um.

        '''
        # c in AA/s.
        kb = 1.380649e-23 # Boltzmann constant, in J/K
        hp = 6.62607015e-34 # Planck constant, in J*s
        # from Eq.3 of Bianchi 13
        kabs0 = 4.0 # in cm2/g
        beta_d= 2.08 #
        lam0  = 250.*1e4 # mu m to AA
        
        from astropy.modeling import models
        from astropy import units as u
        '''

        print('Reading dust table...')
        for tt in range(len(Temp)):
            if tt == 0:
                # For full;
                nd_d  = np.arange(0,len(lambda_d),1)

                # ASDF
                tree_spec_dust_full.update({'wavelength': lambda_d*(1.+zbest)})
                tree_spec_dust_full.update({'colnum': nd_d})

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

            #numin, numax, nmodel = 8, 3, 9
            numin, numax, nmodel = tt, 3, 9
            fnu_d = get_spectrum_draine(lambda_d, DL, zbest, numin, numax, nmodel, DIR_DUST=MB.DIR_DUST)

            if False:
                for nn in range(0,11,1):
                    try:
                        fnu_d_tmp = get_spectrum_draine(lambda_d, DL, zbest, numin, numax, nn, DIR_DUST=MB.DIR_DUST)
                        plt.plot(lambda_d * (1+zbest), fnu_d_tmp, label='%d'%nn)
                        plt.xlim(2000, 5000000)
                        plt.xscale('log')
                        plt.yscale('log')
                    except:
                        print('Errir in ',nn)
                plt.legend()
                plt.show()

            # ASDF
            tree_spec_dust_full.update({'fspec_'+str(tt): fnu_d})

            # Convolution;
            ALLFILT = np.append(SFILT,DFILT)
            ltmpbb_d, ftmpbb_d = filconv(ALLFILT,lambda_d*(1.+zbest),fnu_d,DIR_FILT)

            if f_spec:
                ftmp_nu_int_d = data_int(lm, lambda_d*(1.+zbest), fnu_d)
                ltmpbb_d = np.append(lm, ltmpbb_d)
                ftmpbb_d = np.append(ftmp_nu_int_d, ftmpbb_d)
                nd_db = np.arange(0, len(ftmpbb_d), 1)

            if tt == 0:
                # For conv;
                col3   = fits.Column(name='wavelength', format='E', unit='AA', array=ltmpbb_d)
                nd_db  = np.arange(0,len(ltmpbb_d),1)
                col4   = fits.Column(name='colnum', format='K', unit='', array=nd_db)
                col04 = [col3, col4]
                # ASDF
                tree_spec_dust.update({'wavelength': ltmpbb_d})
                tree_spec_dust.update({'colnum': nd_db})

            tree_spec_dust.update({'fspec_'+str(tt): ftmpbb_d})

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
        nbblast = len(ltmpbb[0,:])+len(lm)
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

    print('Done making templates at z=%.2f.\n'%zbest)
