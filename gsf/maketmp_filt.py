import matplotlib.pyplot as plt
import numpy as np
import scipy
import sys
import os
from scipy.integrate import simps
import scipy.interpolate as interpolate
import asdf

from astropy.io import fits,ascii
from astropy.modeling.models import Moffat1D
from astropy.convolution import convolve, convolve_fft
import astropy.units as u

from .function import *
from .function_igm import *



def maketemp(MB, ebblim=1e10, lamliml=0., lamlimu=50000., ncolbb=10000, 
    tau_lim=0.001, tmp_norm=1e10, nthin=1, delwave=0, lammax=300000, f_IGM=True):
    '''
    Make SPECTRA at given z and filter set.
    Also, after v1.8, through this function library and data are register to the main class object, MB.
    
    Parameters
    ----------
    inputs : str
        Configuration file.
    zbest : float
        Best redshift at this iteration. Templates are generated based on this reshift.
    Z : array
        Stellar phase metallicity in logZsun.
    age : array
        Age, in Gyr.
    fneb : int
        flag for adding nebular emissionself.
    tmp_norm : float
        Normalization of the stored templated. i.e. each template is in units of tmp_norm [Lsun].
    '''
    inputs = MB.inputs
    age = MB.age
    nage = MB.nage
    Z = MB.Zall
    fneb = MB.fneb
    DIR_TMP = MB.DIR_TMP
    zbest = MB.zgal
    tau0 = MB.tau0

    delz = 1.0
    MB.zbests = np.arange(MB.zgal, MB.zgal + 0.01, delz)

    try:
        af = MB.af0
    except:
        if not os.path.exists(os.path.join(DIR_TMP, 'spec_all.asdf')):
            msg = 'The z=0 template is missing in %s directory.\nCheck your configuration file (`DIR_TEMP`) or run with flag=0.'%(DIR_TMP)
            print_err(msg, exit=True)
        af = asdf.open(os.path.join(DIR_TMP, 'spec_all.asdf'))
        MB.af0 = af

    mshdu = af['ML']
    spechdu = af['spec']

    # Consistency check:
    flag = check_library(MB, af)
    if not flag:
        msg = 'There is inconsistency in z0 library and input file. Exiting.'
        print_err(msg, exit=True)

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
    tree_SFR = {}

    try:
        DIR_EXTR = MB.DIR_EXTR
        if len(DIR_EXTR)==0:
            DIR_EXTR = False
    except:
        DIR_EXTR = False
    DIR_FILT = MB.DIR_FILT
    try:
        CAT_BB_IND = inputs['CAT_BB_IND']
    except:
        CAT_BB_IND = False
    try:
        CAT_BB = inputs['CAT_BB']
    except:
        CAT_BB = False

    try:
        SFILT = MB.filts
        FWFILT = fil_fwhm(SFILT, DIR_FILT)
        MB.logger.info('Filters in the input catalog are: ' + ','.join(MB.filts))
    except:
        msg = 'Filter is not detected!!\nMake sure your filter directory is correct.'
        print_err(msg, exit=True)

    try:
        SKIPFILT = inputs['SKIPFILT']
        SKIPFILT = [x.strip() for x in SKIPFILT.split(',')]
    except:
        SKIPFILT = []

    # If FIR data;
    if MB.f_dust:
        DFILT = inputs['FIR_FILTER'] # filter band string.
        DFILT = [x.strip() for x in DFILT.split(',')]
        DFWFILT = fil_fwhm(DFILT, DIR_FILT)
        CAT_BB_DUST = inputs['CAT_BB_DUST']
        DT0 = float(inputs['TDUSTMIN'])
        DT1 = float(inputs['TDUSTMAX'])
        dDT = float(inputs['DELTDUST'])
        MB.logger.info('FIR is implemented.')
    else:
        MB.logger.info('No FIR is implemented.')

    MB.logger.info('Making templates at z=%.4f'%(zbest))
    ####################################################
    # Get extracted spectra.
    ####################################################
    #
    # Read ascii files
    #
    MB.f_spec = False
    data_meta = {'data_len':np.zeros(3,int),'data_origin':[],'data_index':[]}

    try:
        spec_files = [x.strip() for x in inputs['SPEC_FILE'].split(',')]
    except:
        spec_files = []
        MB.logger.info('No spec file is provided.')
        pass
    ninp0 = np.zeros(len(spec_files), dtype='int')

    # THIS PART IS JUST TO GET THE TOTAL ARRAY NUMBER;
    for ff, spec_file in enumerate(spec_files):
        try:
            if spec_file.split('.')[-1] == 'asdf':
                id_asdf = int(spec_file.split('_')[2])
                fd0 = asdf.open(os.path.join(DIR_EXTR, spec_file))
                lm0tmp = fd0[id_asdf]['wavelength'].to(u.angstrom)
                ninp0[ff] = len(lm0tmp)
            elif spec_file.split('.')[-1] == 'fits':
                fd0 = fits.open(os.path.join(DIR_EXTR, spec_file))[1].data
                eobs0 = fd0['full_err']
                spec_mask = (eobs0>0)
                lm0tmp = fd0['wave'][spec_mask]
                ninp0[ff] = len(lm0tmp)
            else:
                fd0 = np.loadtxt(os.path.join(DIR_EXTR, spec_file), comments='#')
                lm0tmp = fd0[:,0]
                ninp0[ff] = len(lm0tmp)
        except Exception:
            MB.logger.error('File, %s, cannot be open.'%(os.path.join(DIR_EXTR, spec_file)))
            pass

    # Then, Constructing arrays.
    lm = np.zeros(np.sum(ninp0[:]),dtype=float)
    fobs = np.zeros(np.sum(ninp0[:]),dtype=float)
    eobs = np.zeros(np.sum(ninp0[:]),dtype=float)
    fgrs = np.zeros(np.sum(ninp0[:]),dtype=int) # FLAG for each grism.
    for ff, spec_file in enumerate(spec_files):
        try:
            if spec_file.split('.')[-1] == 'asdf':
                id_asdf = int(spec_file.split('_')[2])
                fd0 = asdf.open(os.path.join(DIR_EXTR, spec_file))
                lm0tmp = fd0[id_asdf]['wavelength'].to(u.angstrom).value
                fobs0 = fd0[id_asdf]['flux'].value
                eobs0 = np.sqrt(fd0[id_asdf]['fluxvar']).value
            elif spec_file.split('.')[-1] == 'fits':
                fd0 = fits.open(os.path.join(DIR_EXTR, spec_file))[1].data
                eobs0 = fd0['full_err']
                if True:
                    spec_mask = (eobs0>0)
                else:
                    spec_mask = ()
                lm0tmp = fd0['wave'][spec_mask]
                if lm0tmp.max() < 10:
                    MB.logger.warning('Wave column in the input spec file seems to be um. Scaling to AA.')
                    lm0tmp *= 1e4
                try:
                    magzp_spec = float(inputs['MAGZP_SPEC'])
                    magzp = float(inputs['MAGZP'])
                    C_spec = 10**((magzp-magzp_spec)/(2.5))
                except:
                    MB.logger.info('`MAGZP_SPEC` not found. Assuming same as `MAGZP`')
                    C_spec = 1
                fobs0 = fd0['flux'][spec_mask] * C_spec
                eobs0 = fd0['full_err'][spec_mask] * C_spec
            else:
                fd0 = np.loadtxt(os.path.join(DIR_EXTR, spec_file), comments='#')
                lm0tmp = fd0[:,0]
                fobs0 = fd0[:,1]
                eobs0 = fd0[:,2]

            for ii1 in range(ninp0[ff]):
                if ff==0:
                    ii = ii1
                else:
                    ii = ii1 + np.sum(ninp0[:ff])
                fgrs[ii] = ff
                lm[ii] = lm0tmp[ii1]
                fobs[ii] = fobs0[ii1]
                eobs[ii] = eobs0[ii1]

            data_meta['data_len'][ff] = len(lm0tmp)
            data_meta['data_origin'] = np.append(data_meta['data_origin'], '%s'%spec_file)
            data_meta['data_index'] = np.append(data_meta['data_index'], '%d'%ff)
            MB.f_spec = True

        except Exception:
            print('No spec data is registered.')
            pass

    if ncolbb < np.sum(data_meta['data_len']):
        MB.logger.info('ncolbb is updated')
        ncolbb = np.sum(data_meta['data_len'])
        MB.NRbb_lim = ncolbb
    data_meta['data_len'] = np.asarray(data_meta['data_len'])

    #############################
    # READ BB photometry from CAT_BB:
    #############################
    if CAT_BB:
        key_id = 'id'
        fd0 = ascii.read(CAT_BB)
        try:
            id0 = fd0[key_id].astype('str')
        except:
            key_id = 'ID'
            id0 = fd0[key_id].astype('str')

        ii0 = np.where(id0[:]==MB.ID)
        try:
            if len(ii0[0]) == 0:
                msg = 'Could not find the column for [ID: %s] in the input BB catalog! Exiting.'%(MB.ID)
                print_err(msg, exit=True)
            id = fd0[key_id][ii0]
        except:
            msg = 'Could not find the column for [ID: %s] in the input BB catalog! Exiting.'%(MB.ID)
            print_err(msg, exit=True)

        fbb = np.zeros(len(SFILT), dtype=float)
        ebb = np.zeros(len(SFILT), dtype=float)
        for ii in range(len(SFILT)):
            try:
                fbb[ii] = fd0['F%s'%(SFILT[ii])][ii0]
                ebb[ii] = fd0['E%s'%(SFILT[ii])][ii0]
            except:
                msg = 'Could not find flux inputs for filter %s in the input BB catalog! Exiting.'%(SFILT[ii])
                print_err(msg, exit=True)

    elif CAT_BB_IND: # if individual photometric catalog; made in get_sdss.py
        unit = 'nu'
        fd0 = fits.open(DIR_EXTR + CAT_BB_IND)
        hd0 = fd0[1].header
        bunit_bb = float(hd0['bunit'][:5])
        lmbb0 = fd0[1].data['wavelength']
        fbb0 = fd0[1].data['flux'] * bunit_bb
        ebb0 = 1/np.sqrt(fd0[1].data['inverse_variance']) * bunit_bb

        try:
            unit = inputs['UNIT_SPEC']
        except:
            MB.logger.info('No param for UNIT_SPEC is found.')
            MB.logger.info('BB flux unit is assumed to F%s.'%unit)
            pass

        if unit == 'lambda':
            MB.logger.info('Changed BB from Flam to Fnu')
            snbb0 = fbb0/ebb0
            fbb = flamtonu(lmbb0, fbb0, m0set=MB.m0set)
            ebb = fbb/snbb0
        else:
            snbb0 = fbb0/ebb0
            fbb = fbb0
            ebb = ebb0

    else:
        fbb = np.zeros(len(SFILT), dtype=float)
        ebb = np.zeros(len(SFILT), dtype=float)
        for ii in range(len(SFILT)):
            fbb[ii] = 0
            ebb[ii] = -99 #1000

    # Dust flux;
    if MB.f_dust:
        key_id = 'id'
        fdd = ascii.read(CAT_BB_DUST)
        try:
            try:
                id0 = fdd[key_id].astype('str')
            except:
                key_id = 'ID'
                id0 = fdd[key_id].astype('str')

            ii0 = np.where(id0[:]==MB.ID)
            try:
                id = fd0[key_id][ii0]
            except:
                msg = 'Could not find the column for [ID: %s] in the input BB catalog! Exiting.'%(MB.ID)
                print_err(msg, exit=True)
        except:
            return False
            
        id = fdd[key_id]

        fbb_d = np.zeros(len(DFILT), dtype=float)
        ebb_d = np.zeros(len(DFILT), dtype=float)
        for ii in range(len(DFILT)):
            fbb_d[ii] = fdd['F%s'%(DFILT[ii])][ii0]
            ebb_d[ii] = fdd['E%s'%(DFILT[ii])][ii0]

    #################
    # Get morphology;
    #################
    MB.f_prism = False
    MB.file_res = None
    MB.f_diff_conv = False

    if MB.f_spec:
        LSF = get_LSF(inputs, DIR_EXTR, MB.ID, lm)
    else:
        LSF = []
        lm = []
    try:
        if inputs['MORP'] == 'jwst-prism':
            MB.file_res = os.path.join(MB.config_path, 'jwst_nirspec_prism_disp.fits')
            if os.path.exists(MB.file_res):
                MB.f_prism = True
                LSF = []
    except:
        MB.f_prism = False
        pass

    if MB.f_prism:
        try:
            n_diff_conv = int(inputs['DIFF_CONV'])
            if n_diff_conv == 1:
                MB.f_diff_conv = True
                MB.logger.warning('Differential template convolution, `DIFF_CONV`, is requested - this may take a while.')
        except:
            pass

    try:
        x_HI_input = float(inputs['x_HI'])
        print('Neutral fraction, x_HI = %.2f, is provided;'%(x_HI_input))
    except:
        x_HI_input = None

    if MB.SFH_FORM == -99:
        ####################################
        # Start generating templates
        ####################################
        for zz in range(len(Z)):
            for pp in range(len(tau0)):
                Zbest = Z[zz]
                Na = len(age)
                Ntmp = 1

                for nzz, zbest in enumerate(MB.zbests):

                    age_univ= MB.cosmo.age(zbest).value

                    if zz == 0 and pp == 0:
                        if delwave>0:
                            lm0_orig = spechdu['wavelength'][::nthin]
                            lm0 = np.arange(lm0_orig.min(), lm0_orig.max(), delwave)
                        else:
                            lm0 = spechdu['wavelength'][::nthin]
                        if not lammax == None and not MB.f_dust:
                            lm0 = lm0[(lm0 * (zbest+1) < lammax)]

                    lmbest = np.zeros((Ntmp, len(lm0)), dtype=float)
                    fbest = np.zeros((Ntmp, len(lm0)), dtype=float)
                    lmbestbb = np.zeros((Ntmp, len(SFILT)), dtype=float)
                    fbestbb = np.zeros((Ntmp, len(SFILT)), dtype=float)

                    spec_mul = np.zeros((Na, len(lm0)), dtype=float)
                    spec_mul_nu = np.zeros((Na, len(lm0)), dtype=float)
                    spec_mul_nu_conv = np.zeros((Na, len(lm0)), dtype=float)

                    ftmpbb = np.zeros((Na, len(SFILT)), dtype=float)
                    ltmpbb = np.zeros((Na, len(SFILT)), dtype=float)

                    ftmp_nu_int = np.zeros((Na, len(lm)), dtype=float)
                    spec_av_tmp = np.zeros((Na, len(lm)), dtype=float)

                    ms = np.zeros(Na, dtype=float)
                    Ls = np.zeros(Na, dtype=float)
                    tau = np.zeros(Na, dtype=float)
                    sfr = np.zeros(Na, dtype=float)
                    Fuv = np.zeros(Na, dtype=float)
                    ms[:] = mshdu['ms_'+str(zz)][:] # [:] is necessary.
                    Ls[:] = mshdu['Ls_'+str(zz)][:]

                    # Distance;
                    DL = MB.cosmo.luminosity_distance(zbest).value * MB.Mpc_cm # Luminositydistance in cm
                    DL10 = MB.Mpc_cm / 1e6 * 10 # 10pc in cm

                    for ss in range(Na):
                        wave = lm0
                        wavetmp = wave*(1.+zbest)
                        if delwave>0:
                            fint = interpolate.interp1d(lm0_orig, spechdu['fspec_'+str(zz)+'_'+str(ss)+'_'+str(pp)][::nthin], kind='nearest', fill_value="extrapolate")
                            spec_mul[ss,:] = fint(lm0)
                        else:
                            spec_mul[ss,:] = spechdu['fspec_'+str(zz)+'_'+str(ss)+'_'+str(pp)][::nthin] # Lsun/A

                        ###################
                        # IGM attenuation.
                        ###################
                        if f_IGM:
                            # spec_av_tmp = madau_igm_abs(wave, spec_mul[ss,:], zbest, cosmo=MB.cosmo)
                            spec_av_tmp, x_HI = dijkstra_igm_abs(wave, spec_mul[ss,:], zbest, cosmo=MB.cosmo, x_HI=x_HI_input)
                            MB.x_HI = x_HI
                            spec_mul[ss,:] = spec_av_tmp

                        # Flam to Fnu
                        spec_mul_nu[ss,:] = flamtonu(wave, spec_mul[ss,:], m0set=MB.m0set)
                        spec_mul_nu[ss,:] *= MB.Lsun / (4.*np.pi*DL**2/(1.+zbest))
                        # (1.+zbest) takes acount of the change in delta lam by redshifting.
                        # Note that this is valid only when F_nu.
                        # When Flambda, /(1.+zbest) will be *(1.+zbest).

                        spec_mul_nu[ss,:] *= (1./Ls[ss])*tmp_norm # in unit of erg/s/Hz/cm2/ms[ss].
                        ms[ss] *= (1./Ls[ss])*tmp_norm # M/L; 1 unit template has this mass/solar.
                        try:
                            tautmp = af['ML']['realtau_%d'%int(zz)]
                            sfr[ss] = ms[ss] / (tautmp[ss]*1e9) # SFR per unit template, in units of Msolar/yr.
                        except:
                            if zz == 0 and ss == 0:
                                MB.logger.warning('realtau entry not found. SFR is set to 0. (Or rerun maketmp_z0.py)')
                            sfr[ss] = 0

                        if MB.f_spec:
                            ftmp_nu_int[ss,:] = data_int(lm, wavetmp, spec_mul_nu[ss,:])

                        ltmpbb[ss,:], ftmpbb[ss,:] = filconv(SFILT, wavetmp, spec_mul_nu[ss,:], DIR_FILT, MB=MB, f_regist=False)

                        # Convolution has to come after this?
                        if MB.f_spec:
                            spec_mul_nu_conv[ss,:] = convolve_templates(wavetmp, spec_mul_nu[ss], LSF, boundary='extend', 
                                                                        f_prism=MB.f_prism, file_res=MB.file_res, redshift=zbest, f_diff_conv=MB.f_diff_conv)
                        else:
                            spec_mul_nu_conv[ss,:] = spec_mul_nu[ss]

                        ##########################################
                        # Writing out the templates to fits table.
                        ##########################################
                        if ss == 0 and pp == 0 and zz == 0:
                            # First file
                            nd1 = np.arange(0,len(lm),1)
                            nd3 = np.arange(10000,10000+len(ltmpbb[ss,:]),1)
                            nd_ap = np.append(nd1,nd3)
                            lm_ap = np.append(lm, ltmpbb[ss,:])

                            # ASDF
                            tree_spec.update({'wavelength':lm_ap})
                            tree_spec.update({'colnum':nd_ap})

                            # Second file
                            # ASDF
                            nd = np.arange(0,len(wavetmp),1)
                            if zbest == MB.zgal:
                                tree_spec_full.update({'wavelength':wavetmp})
                                tree_spec_full.update({'colnum':nd})

                        # ASDF
                        spec_ap = np.append(ftmp_nu_int[ss,:], ftmpbb[ss,:])
                        tree_spec.update({'fspec_'+str(zz)+'_'+str(ss)+'_'+str(pp): spec_ap})
                        if zbest == MB.zgal:
                            tree_spec_full.update({'fspec_orig_'+str(zz)+'_'+str(ss)+'_'+str(pp): spec_mul_nu[ss,:]})

                        # For nebular library;
                        # For every Z, but not for ss and pp.
                        if fneb == 1 and MB.f_bpass==0 and ss==0 and pp==0:
                            if zz==0:
                                spec_mul_neb = np.zeros((len(Z), len(MB.logUs), len(lm0)), dtype=float)
                                spec_mul_neb_nu = np.zeros((len(Z), len(MB.logUs), len(lm0)), dtype=float)
                                spec_mul_neb_nu_conv = np.zeros((len(Z), len(MB.logUs), len(lm0)), dtype=float)
                                ftmpbb_neb = np.zeros((len(Z), len(MB.logUs), len(SFILT)), dtype=float)
                                ltmpbb_neb = np.zeros((len(Z), len(MB.logUs), len(SFILT)), dtype=float)
                                ftmp_neb_nu_int = np.zeros((len(Z), len(MB.logUs), len(lm)), dtype=float)
                                ms_neb = np.zeros((len(Z), len(MB.logUs)), dtype=float)

                            for uu in range(len(MB.logUs)):
                                if delwave>0:
                                    fint = interpolate.interp1d(lm0_orig, spechdu['flux_nebular_Z%d_logU%d'%(zz,uu)][::nthin], kind='nearest', fill_value="extrapolate")
                                    spec_mul_neb[zz,uu,:] = fint(lm0)
                                else:
                                    spec_mul_neb[zz,uu,:] = spechdu['flux_nebular_Z%d_logU%d'%(zz,uu)][::nthin]
                                
                                con_neb = (spec_mul_neb[zz,uu,:]<0)
                                spec_mul_neb[zz,uu,:][con_neb] = 0
                                
                                if f_IGM:
                                    # spec_neb_av_tmp = madau_igm_abs(wave, spec_mul_neb[zz,uu,:], zbest, cosmo=MB.cosmo)
                                    spec_neb_av_tmp, x_HI = dijkstra_igm_abs(wave, spec_mul_neb[zz,uu,:], zbest, cosmo=MB.cosmo, x_HI=x_HI_input)
                                    # spec_neb_av_tmp = masongronke_igm_abs(wave, spec_mul_neb[zz,uu,:], zbest, cosmo=MB.cosmo)
                                    spec_mul_neb[zz,uu,:] = spec_neb_av_tmp

                                spec_mul_neb_nu[zz,uu,:] = flamtonu(wave, spec_mul_neb[zz,uu,:], m0set=MB.m0set)
                                
                                spec_mul_neb_nu[zz,uu,:] *= MB.Lsun/(4.*np.pi*DL**2/(1.+zbest))
                                
                                spec_mul_neb_nu[zz,uu,:] *= (1./Ls[ss])*tmp_norm # in unit of erg/s/Hz/cm2/ms[ss].
                                ltmpbb_neb[zz,uu,:], ftmpbb_neb[zz,uu,:] = filconv(SFILT, wavetmp, spec_mul_neb_nu[zz,uu,:], DIR_FILT, MB=MB, f_regist=False)
                                ms_neb[zz,uu] *= (1./Ls[ss])*tmp_norm # M/L; 1 unit template has this mass/solar.

                                if MB.f_spec:
                                    ftmp_neb_nu_int[zz,uu,:] = data_int(lm, wavetmp, spec_mul_neb_nu[zz,uu,:])

                                if MB.f_spec:
                                    spec_mul_neb_nu_conv[zz,uu,:] = convolve_templates(wavetmp, spec_mul_neb_nu[zz,uu,:], LSF, boundary='extend', 
                                                                                f_prism=MB.f_prism, file_res=MB.file_res, redshift=zbest, f_diff_conv=MB.f_diff_conv)
                                else:
                                    spec_mul_neb_nu_conv[zz,uu,:] = spec_mul_neb_nu[zz,uu,:]

                                if zbest == MB.zgal:
                                    tree_spec_full.update({'fspec_orig_nebular_Z%d_logU%d'%(zz,uu): spec_mul_neb_nu[zz,uu,:]})
                                    tree_spec_full.update({'fspec_nebular_Z%d_logU%d'%(zz,uu): spec_mul_neb_nu_conv[zz,uu,:]})

                                spec_neb_ap = np.append(ftmp_neb_nu_int[zz,uu,:], ftmpbb_neb[zz,uu,:])
                                tree_spec.update({'fspec_nebular_Z%d_logU%d'%(zz,uu): spec_neb_ap})

                    #########################
                    # Summarize the ML
                    #########################
                    if pp == 0:
                        # ML
                        tree_ML.update({'ML_'+str(zz): ms})
                        # SFR
                        tree_SFR.update({'SFR_'+str(zz): sfr})

                        if fneb == 1 and MB.f_bpass==0:
                            # ML neb
                            tree_ML.update({'ML_neb': ms_neb[zz,:]})

    else:
        ####################################
        # Start generating templates
        ####################################
        print('Tau model comes here;')
        tau = MB.tau
        age = MB.ageparam
        Na = len(age)

        for zz in range(len(Z)):
            Zbest = Z[zz]
            Na = len(age)
            Ntmp = 1
            age_univ= MB.cosmo.age(zbest).value

            for tt in range(len(tau)): # tau
                if zz == 0 and tt == 0:
                    if delwave>0:
                        lm0_orig = spechdu['wavelength'][::nthin]
                        lm0 = np.arange(lm0_orig.min(), lm0_orig.max(), delwave)
                    else:
                        lm0 = spechdu['wavelength'][::nthin]
                    if not lammax == None and not MB.f_dust:
                        lm0 = lm0[(lm0 * (zbest+1) < lammax)]
                    wave = lm0

                lmbest = np.zeros((Ntmp, len(lm0)), dtype=float)
                fbest = np.zeros((Ntmp, len(lm0)), dtype=float)
                lmbestbb = np.zeros((Ntmp, len(SFILT)), dtype=float)
                fbestbb = np.zeros((Ntmp, len(SFILT)), dtype=float)

                spec_mul = np.zeros((Na, len(lm0)), dtype=float)
                spec_mul_nu = np.zeros((Na, len(lm0)), dtype=float)
                spec_mul_nu_conv = np.zeros((Na, len(lm0)), dtype=float)

                ftmpbb = np.zeros((Na, len(SFILT)), dtype=float)
                ltmpbb = np.zeros((Na, len(SFILT)), dtype=float)

                ftmp_nu_int = np.zeros((Na, len(lm)), dtype=float)
                spec_av_tmp = np.zeros((Na, len(lm)), dtype=float)

                ms = np.zeros(Na, dtype=float)
                Ls = np.zeros(Na, dtype=float)
                ms[:] = mshdu['ms_'+str(zz)+'_'+str(tt)][:] # [:] is necessary.
                Ls[:] = mshdu['Ls_'+str(zz)+'_'+str(tt)][:]
                Fuv = np.zeros(Na, dtype=float)

                for ss in range(Na):
                    #print(ss,tt,zz)
                    if ss == 0 and tt == 0 and zz == 0:
                        DL = MB.cosmo.luminosity_distance(zbest).value * MB.Mpc_cm # Luminositydistance in cm
                        wavetmp = wave*(1.+zbest)

                    if delwave>0:
                        fint = interpolate.interp1d(lm0_orig, spechdu['fspec_'+str(zz)+'_'+str(tt)+'_'+str(ss)][::nthin], kind='nearest', fill_value="extrapolate")
                        spec_mul[ss] = fint(lm0)
                    else:
                        spec_mul[ss] = spechdu['fspec_'+str(zz)+'_'+str(tt)+'_'+str(ss)][::nthin]

                    ##################
                    # IGM attenuation.
                    ##################
                    if f_IGM:
                        # spec_av_tmp = madau_igm_abs(wave, spec_mul[ss,:], zbest, cosmo=MB.cosmo)
                        spec_av_tmp, x_HI = dijkstra_igm_abs(wave, spec_mul[ss,:], zbest, cosmo=MB.cosmo, x_HI=x_HI_input)
                    else:
                        spec_av_tmp = spec_mul[ss,:]

                    spec_mul_nu[ss,:] = flamtonu(wave, spec_av_tmp, m0set=MB.m0set)

                    spec_mul_nu[ss,:] *= MB.Lsun/(4.*np.pi*DL**2/(1.+zbest))
                    spec_mul_nu[ss,:] *= (1./Ls[ss])*tmp_norm # in unit of erg/s/Hz/cm2/ms[ss].
                    ms[ss] *= (1./Ls[ss])*tmp_norm # M/L; 1 unit template has this mass in [Msolar].

                    if len(lm)>0:
                        try:
                            spec_mul_nu_conv[ss,:] = convolve(spec_mul_nu[ss,:], LSF, boundary='extend')
                        except:
                            spec_mul_nu_conv[ss,:] = spec_mul_nu[ss,:]
                            if zz==0 and ss==0:
                                print('Kernel is too small. No convolution.')
                    else:
                        spec_mul_nu_conv[ss,:] = spec_mul_nu[ss,:]

                    if MB.f_spec:
                        #ftmp_nu_int[ss,:] = data_int(lm, wavetmp, spec_mul_nu_conv[ss,:])
                        ftmp_nu_int[ss,:] = data_int(lm, wavetmp, spec_mul_nu[ss,:])
                    
                    # Register filter response;
                    ltmpbb[ss,:], ftmpbb[ss,:] = filconv(SFILT, wavetmp, spec_mul_nu[ss,:], DIR_FILT, MB=MB, f_regist=False)

                    ##########################################
                    # Writing out the templates to fits table.
                    ##########################################
                    if ss == 0 and tt == 0 and zz == 0:
                        # First file
                        nd1    = np.arange(0,len(lm),1)
                        nd3    = np.arange(10000,10000+len(ltmpbb[ss,:]),1)
                        nd_ap  = np.append(nd1,nd3)
                        lm_ap  = np.append(lm, ltmpbb[ss,:])

                        # ASDF
                        tree_spec.update({'wavelength':lm_ap})
                        tree_spec.update({'colnum':nd_ap})

                        # Second file
                        # ASDF
                        nd = np.arange(0,len(wavetmp),1)
                        tree_spec_full.update({'wavelength':wavetmp})
                        tree_spec_full.update({'colnum':nd})

                    # ASDF
                    # ???
                    spec_ap = np.append(ftmp_nu_int[ss,:], ftmpbb[ss,:])
                    tree_spec.update({'fspec_'+str(zz)+'_'+str(tt)+'_'+str(ss): spec_ap})
                    tree_spec_full.update({'fspec_orig_'+str(zz)+'_'+str(tt)+'_'+str(ss): spec_mul_nu[ss,:]})                
                    #tree_spec_full.update({'fspec_'+str(zz)+'_'+str(tt)+'_'+str(ss): spec_mul_nu_conv[ss,:]})

                    # Nebular library;
                    if fneb == 1 and MB.f_bpass==0 and ss==0 and tt==0:
                        if zz==0:
                            spec_mul_neb = np.zeros((len(Z), len(MB.logUs), len(lm0)), dtype=float)
                            spec_mul_neb_nu = np.zeros((len(Z), len(MB.logUs), len(lm0)), dtype=float)
                            spec_mul_neb_nu_conv = np.zeros((len(Z), len(MB.logUs), len(lm0)), dtype=float)
                            ftmpbb_neb = np.zeros((len(Z), len(MB.logUs), len(SFILT)), dtype=float)
                            ltmpbb_neb = np.zeros((len(Z), len(MB.logUs), len(SFILT)), dtype=float)
                            ftmp_neb_nu_int = np.zeros((len(Z), len(MB.logUs), len(lm)), dtype=float)

                        for uu in range(len(MB.logUs)):
                            if delwave>0:
                                fint = interpolate.interp1d(lm0_orig, spechdu['flux_nebular_Z%d_logU%d'%(zz,uu)][::nthin], kind='nearest', fill_value="extrapolate")
                                spec_mul_neb[zz,uu,:] = fint(lm0)
                            else:
                                spec_mul_neb[zz,uu,:] = spechdu['flux_nebular_Z%d_logU%d'%(zz,uu)][::nthin]
                            
                            con_neb = (spec_mul_neb[zz,uu,:]<0)
                            spec_mul_neb[zz,uu,:][con_neb] = 0
                            
                            if f_IGM:
                                # spec_neb_av_tmp = madau_igm_abs(wave, spec_mul_neb[zz,uu,:], zbest, cosmo=MB.cosmo)
                                spec_neb_av_tmp, x_HI = dijkstra_igm_abs(wave, spec_mul_neb[zz,uu,:], zbest, cosmo=MB.cosmo, x_HI=x_HI_input)
                                spec_mul_neb[zz,uu,:] = spec_neb_av_tmp

                            spec_mul_neb_nu[zz,uu,:] = flamtonu(wave, spec_mul_neb[zz,uu,:], m0set=MB.m0set)
                            spec_mul_neb_nu[zz,uu,:] *= MB.Lsun/(4.*np.pi*DL**2/(1.+zbest))
                            spec_mul_neb_nu[zz,uu,:] *= (1./Ls[ss])*tmp_norm # in unit of erg/s/Hz/cm2/ms[ss].
                            ltmpbb_neb[zz,uu,:], ftmpbb_neb[zz,uu,:] = filconv(SFILT, wavetmp, spec_mul_neb_nu[zz,uu,:], DIR_FILT, MB=MB, f_regist=False)

                            if MB.f_spec:
                                ftmp_neb_nu_int[zz,uu,:] = data_int(lm, wavetmp, spec_mul_neb_nu[zz,uu,:])

                            if MB.f_spec:
                                spec_mul_neb_nu_conv[zz,uu,:] = convolve_templates(wavetmp, spec_mul_neb_nu[zz,uu,:], LSF, boundary='extend', 
                                                                            f_prism=MB.f_prism, file_res=MB.file_res, redshift=zbest, f_diff_conv=MB.f_diff_conv)
                            else:
                                spec_mul_neb_nu_conv[zz,uu,:] = spec_mul_neb_nu[zz,uu,:]

                            tree_spec_full.update({'fspec_orig_nebular_Z%d_logU%d'%(zz,uu): spec_mul_neb_nu[zz,uu,:]})
                            tree_spec_full.update({'fspec_nebular_Z%d_logU%d'%(zz,uu): spec_mul_neb_nu_conv[zz,uu,:]})

                            spec_neb_ap = np.append(ftmp_neb_nu_int[zz,uu,:], ftmpbb_neb[zz,uu,:])
                            tree_spec.update({'fspec_nebular_Z%d_logU%d'%(zz,uu): spec_neb_ap})

                #########################
                # Summarize the ML
                #########################
                # ASDF
                tree_ML.update({'ML_'+str(zz)+'_'+str(tt): ms})

    #########################
    # Summarize the templates
    #########################
    tree['id'] = MB.ID
    tree['z'] = zbest
    try:
        tree['x_HI'] = x_HI
    except:
        pass

    tree.update({'spec' : tree_spec})
    tree.update({'spec_full' : tree_spec_full})
    tree.update({'ML' : tree_ML})
    tree.update({'SFR' : tree_SFR})

    ######################
    # Add dust component;
    ######################
    if MB.f_dust:
        tree_spec_dust = {}
        tree_spec_dust_full = {}

        if DT0 == DT1:
            Temp = [DT0]
        else:
            Temp = np.arange(DT0,DT1,dDT)

        Mdust_temp = np.zeros(len(Temp),float)
        dellam_d = 1e1
        lambda_d = np.arange(1e3, 1e7, dellam_d)
        
        MB.logger.info('Reading dust table...')
        for tt in range(len(Temp)):
            if tt == 0:
                # For full;
                nd_d  = np.arange(0,len(lambda_d),1)

                # ASDF
                tree_spec_dust_full.update({'wavelength': lambda_d*(1.+zbest)})
                tree_spec_dust_full.update({'colnum': nd_d})

            f_drain = False #True #
            if f_drain:
                #numin, numax, nmodel = 8, 3, 9
                numin, numax, nmodel = tt, MB.dust_numax, MB.dust_nmodel #3, 9
                fnu_d = get_spectrum_draine(lambda_d, DL, zbest, numin, numax, nmodel, DIR_DUST=MB.DIR_DUST, m0set=MB.m0set)
                # This should be in fnu w mzp=25.0
                Mdust_temp[tt] = 1.0
            else:
                if tt == 0:
                    from astropy.modeling import models
                    MB.logger.info('Dust emission based on Modified Blackbody')
                    '''
                    # from Eq.3 of Bianchi 13
                    kabs0 = 4.0 # in cm2/g
                    beta_d = 2.08 #
                    lam0 = 250.*1e4 # mu m to AA
                    '''
                    kb = 1.380649e-23 # Boltzmann constant, in J/K
                    hp = 6.62607015e-34 # Planck constant, in J*s
                    wav = lambda_d * u.AA
                    nures = c / wav.value * 1e-9 # GHz

                    Tcmb = 2.726 * (1+zbest)
                    beta = 1.8
                    kappa = 0.0484 * (nures/345.)**beta # m^2 / kg
                    kappa *= (100)**2 / (1e3) # cm2/g
                    nurest = c / wav.value # Hz

                BT_nu = (2.0 * hp * (nurest)**3. * c**(-2.) / (np.exp(hp*(nurest)/(kb * Temp[tt])) - 1.0))
                BT_nu_cmb = (2.0 * hp * (nurest)**3. * c**(-2.) / (np.exp(hp*(nurest)/(kb * Temp[tt])) - 1.0))
                # J s * (Hz)3 * (AA/s)^-2 = 1e+7 erg * (AA)-2 /s
                denom = BT_nu-BT_nu_cmb

                fnu_d = (1+zbest) / (DL*MB.Mpc_cm)**2 * (kappa * denom)
                # 1/cm2 * cm2/g * (1e+7 erg * (AA)-2 / s) = 1e7 erg /s / g / AA / AA
                fnu_d *= 1e7 * 1e16 # erg /s / g / cm2
                fnu_d *= 1.989e+33 # erg /s / Msun / cm2

                # Into magzp;
                fnu_d = fnutonu(fnu_d, m0set=MB.m0set, m0input=-48.6)
                Mdust_temp[tt] = 1. # Msun/temp, like Mass to light ratio.

            # ASDF
            tree_spec_dust_full.update({'fspec_'+str(tt): fnu_d})

            # Convolution;
            ALLFILT = np.append(SFILT,DFILT)
            ltmpbb_d, ftmpbb_d = filconv(ALLFILT,lambda_d*(1.+zbest),fnu_d,DIR_FILT)
            nd_db = np.arange(0, len(ftmpbb_d), 1)

            if MB.f_spec:
                ftmp_nu_int_d = data_int(lm, lambda_d*(1.+zbest), fnu_d)
                ltmpbb_d = np.append(lm, ltmpbb_d)
                ftmpbb_d = np.append(ftmp_nu_int_d, ftmpbb_d)
                nd_db = np.arange(0, len(ftmpbb_d), 1)

            if tt == 0:
                # For conv;
                # ASDF
                tree_spec_dust.update({'wavelength': ltmpbb_d})
                tree_spec_dust.update({'colnum': nd_db})

            tree_spec_dust.update({'fspec_'+str(tt): ftmpbb_d})
        # Md/temp;
        tree_spec_dust.update({'Mdust': Mdust_temp})
        tree.update({'spec_dust' : tree_spec_dust})
        tree.update({'spec_dust_full' : tree_spec_dust_full})
        MB.logger.info('dust updated.')

    # Save;
    file_asdf = os.path.join(DIR_TMP, 'spec_all_' + MB.ID + '.asdf')
    af = asdf.AsdfFile(tree)
    af.write_to(file_asdf, all_array_compression='zlib')
    # Loading the redshifted template;
    MB.af = asdf.open(file_asdf)
    # Remove file?
    os.system('rm %s'%file_asdf)

    ##########################################
    # For data;
    ##########################################
    if True:
        MB.data = {}
        MB.data['meta'] = data_meta

        # The following files are just temporary.
        file_tmp = 'tmp_library_%s.txt'%MB.ID
        file_tmp2 = 'tmp_library2_%s.txt'%MB.ID
        fw = open(file_tmp,'w')
        fw.write('# BB data (>%d) in this file are not used in fitting.\n'%(ncolbb))

        # this is for spec;
        for ii in range(len(lm)):
            g_offset = 0 #1000 * fgrs[ii]
            if lm[ii]/(1.+zbest) > lamliml and lm[ii]/(1.+zbest) < lamlimu and not np.isnan(fobs[ii]):
                fw.write('%d %.5f %.5e %.5e\n'%(ii+g_offset, lm[ii], fobs[ii], eobs[ii]))
            else:
                fw.write('%d %.5f 0 1000\n'%(ii+g_offset, lm[ii]))

        for ii in range(len(ltmpbb[0,:])):
            if SFILT[ii] in SKIPFILT:# data point to be skiped;
                fw.write('%d %.5f %.5e %.5e\n'%(ii+ncolbb, ltmpbb[0,ii], 0.0, fbb[ii]))
            elif ebb[ii]>ebblim:
                fw.write('%d %.5f 0 1000\n'%(ii+ncolbb, ltmpbb[0,ii]))
            else:
                fw.write('%d %.5f %.5e %.5e\n'%(ii+ncolbb, ltmpbb[0,ii], fbb[ii], ebb[ii]))
        fw.close()

        # register;
        dat = ascii.read(file_tmp, format='no_header')
        NR = dat['col1']
        x = dat['col2']
        fy00 = dat['col3']
        ey00 = dat['col4']
        dict_spec_obs = {'NR':NR, 'x':x, 'fy':fy00, 'ey':ey00}
        MB.data['spec_obs'] = dict_spec_obs

        fw = open(file_tmp,'w')
        if MB.f_dust:
            nbblast = len(ltmpbb[0,:])+len(lm)
            for ii in range(len(ebb_d[:])):
                if ebb_d[ii]>ebblim:
                    fw.write('%d %.5f 0 1000\n'%(ii+ncolbb+nbblast, ltmpbb_d[ii+nbblast]))
                else:
                    fw.write('%d %.5f %.5e %.5e\n'%(ii+ncolbb+nbblast, ltmpbb_d[ii+nbblast], fbb_d[ii], ebb_d[ii]))
        fw.close()
        
        if MB.f_dust:
            dat = ascii.read(file_tmp, format='no_header')
            nr_d = dat['col1']
            x_d = dat['col2']
            fy_d = dat['col3']
            ey_d = dat['col4']
            dict_spec_fir_obs = {'NR':nr_d, 'x':x_d, 'fy':fy_d, 'ey':ey_d}
            MB.data['spec_fir_obs'] = dict_spec_fir_obs

        # BB phot
        MB.has_photometry = False
        fw = open(file_tmp,'w')
        fw_rem = open(file_tmp2, 'w')
        for ii in range(len(ltmpbb[0,:])):
            MB.has_photometry = True
            if SFILT[ii] in SKIPFILT:# data point to be skiped;
                fw.write('%d %.5f %.5e %.5e %.1f %s\n'%(ii+ncolbb, ltmpbb[0,ii], 0.0, fbb[ii], FWFILT[ii]/2., SFILT[ii]))
                fw_rem.write('%d %.5f %.5e %.5e %.1f %s\n'%(ii+ncolbb, ltmpbb[0,ii], fbb[ii], ebb[ii], FWFILT[ii]/2., SFILT[ii]))
            elif ebb[ii]>ebblim:
                fw.write('%d %.5f 0 1000 %.1f %s\n'%(ii+ncolbb, ltmpbb[0,ii], FWFILT[ii]/2., SFILT[ii]))
            elif ebb[ii]<=0:
                fw.write('%d %.5f 0 -99 %.1f %s\n'%(ii+ncolbb, ltmpbb[0,ii], FWFILT[ii]/2., SFILT[ii]))
            else:
                fw.write('%d %.5f %.5e %.5e %.1f %s\n'%(ii+ncolbb, ltmpbb[0,ii], fbb[ii], ebb[ii], FWFILT[ii]/2., SFILT[ii]))
        fw.close()
        fw_rem.close()

        # register;
        if MB.has_photometry:
            dat = ascii.read(file_tmp, format='no_header')
            NRbb = dat['col1']
            xbb  = dat['col2']
            fybb = dat['col3']
            eybb = dat['col4']
            exbb = dat['col5']
            dict_bb_obs = {'NR':NRbb, 'x':xbb, 'fy':fybb, 'ey':eybb, 'ex':exbb}
            MB.data['bb_obs'] = dict_bb_obs
            if len(SKIPFILT)>0:#try:
                dat = ascii.read(file_tmp2, format='no_header')
                NR_ex = dat['col1']
                x_ex = dat['col2']
                fy_ex = dat['col3']
                ey_ex = dat['col4']
                ex_ex = dat['col5']
                dict_bb_obs_removed = {'NR':NR_ex, 'x':x_ex, 'fy':fy_ex, 'ey':ey_ex, 'ex':ex_ex}
                MB.data['bb_obs_removed'] = dict_bb_obs_removed

        # Dust; Not sure where this is being used...
        fw = open(file_tmp,'w')
        if MB.f_dust:
            for ii in range(len(ebb_d[:])):
                if ebb_d[ii]>ebblim:
                    fw.write('%d %.5f 0 1000 %.1f %s\n'%(ii+ncolbb+nbblast, ltmpbb_d[ii+nbblast], DFWFILT[ii]/2., DFILT[ii]))
                else:
                    fw.write('%d %.5f %.5e %.5e %.1f %s\n'%(ii+ncolbb+nbblast, ltmpbb_d[ii+nbblast], fbb_d[ii], ebb_d[ii], DFWFILT[ii]/2., DFILT[ii]))
        fw.close()
        if MB.f_dust:
            dat = ascii.read(file_tmp, format='no_header')
            nr_bb_d = dat['col1']
            x_bb_d = dat['col2']
            fy_bb_d = dat['col3']
            ey_bb_d = dat['col4']
            dict_bb_fir_obs = {'NR':nr_bb_d, 'x':x_bb_d, 'fy':fy_bb_d, 'ey':ey_bb_d}
            MB.data['bb_fir_obs'] = dict_bb_fir_obs

        MB.logger.info('Done making templates at z=%.4f'%zbest)
        os.system('rm %s %s'%(file_tmp, file_tmp2))

    MB.ztemplate = True
    return True


def maketemp_tau(MB, ebblim=1e10, lamliml=0., lamlimu=50000., ncolbb=10000, tau_lim=0.001,
    f_IGM=True, nthin=1, tmp_norm=1e10, delwave=0, lammax=300000):
    '''
    Make SPECTRA at given z and filter set.
    
    Parameters
    ----------
    inputs : str
        Configuration file.
    zbest :float
        Best redshift at this iteration. Templates are generated based on this reshift.
    Z : array
        Stellar phase metallicity in logZsun.
    age : array
        Age, in Gyr.
    fneb : int
        flag for adding nebular emissionself.
    f_IGM : bool
        IGM attenuation. Madau.
    nthin : int
        Thinning templates.
    lammax : float
        Maximum wavelength in RF.
    '''    

    inputs = MB.inputs
    ID = MB.ID
    age = MB.ageparam
    nage = MB.nage
    tau = MB.tau
    Z = MB.Zall
    fneb = MB.fneb
    DIR_TMP = MB.DIR_TMP
    zbest = MB.zgal

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
        DIR_EXTR = MB.DIR_EXTR #inputs['DIR_EXTR']
        if len(DIR_EXTR)==0:
            DIR_EXTR = False
    except:
        DIR_EXTR = False
    DIR_FILT = MB.DIR_FILT #inputs['DIR_FILT']
    try:
        CAT_BB_IND = inputs['CAT_BB_IND']
    except:
        CAT_BB_IND = False
    try:
        CAT_BB = MB.CAT_BB #inputs['CAT_BB']
    except:
        CAT_BB = False

    try:
        SFILT = MB.filts #inputs['FILTER'] # filter band string.
        FWFILT = fil_fwhm(SFILT, DIR_FILT)
    except:
        print('########################')
        print('Filter is not detected!!')
        print('Make sure your \nfilter directory is correct.')
        print('########################')
        sys.exit()
    try:
        SKIPFILT = inputs['SKIPFILT']
        SKIPFILT = [x.strip() for x in SKIPFILT.split(',')]
    except:
        SKIPFILT = []

    # If FIR data;
    if MB.f_dust:
        DFILT = inputs['FIR_FILTER'] # filter band string.
        DFILT = [x.strip() for x in DFILT.split(',')]
        DFWFILT = fil_fwhm(DFILT, DIR_FILT)
        CAT_BB_DUST = inputs['CAT_BB_DUST']
        DT0 = float(inputs['TDUSTMIN'])
        DT1 = float(inputs['TDUSTMAX'])
        dDT = float(inputs['DELTDUST'])
        print('FIR is implemented.\n')
    else:
        print('No FIR is implemented.\n')


    print('############################')
    print('Making templates at z=%.4f'%(zbest))
    print('############################')
    ####################################################
    # Get extracted spectra.
    ####################################################
    MB.f_spec = False
    try:
        spec_files = inputs['SPEC_FILE'] #.replace('$ID','%s'%(ID))
        spec_files = [x.strip() for x in spec_files.split(',')]
        ninp0 = np.zeros(len(spec_files), dtype='int')
        for ff, spec_file in enumerate(spec_files):
            try:
                fd0 = np.loadtxt(DIR_EXTR + spec_file, comments='#')
                lm0tmp = fd0[:,0]
                fobs0 = fd0[:,1]
                eobs0 = fd0[:,2]
                ninp0[ff] = len(lm0tmp)#[con_tmp])
            except Exception:
                print('File, %s/%s, cannot be open.'%(DIR_EXTR,spec_file))
                pass
        # Constructing arrays.
        lm = np.zeros(np.sum(ninp0[:]),dtype=float)
        fobs = np.zeros(np.sum(ninp0[:]),dtype=float)
        eobs = np.zeros(np.sum(ninp0[:]),dtype=float)
        fgrs = np.zeros(np.sum(ninp0[:]),dtype='int')  # FLAG for each grism.
        for ff, spec_file in enumerate(spec_files):
            try:
                fd0 = np.loadtxt(DIR_EXTR + spec_file, comments='#')
                lm0tmp = fd0[:,0]
                fobs0 = fd0[:,1]
                eobs0 = fd0[:,2]
                for ii1 in range(ninp0[ff]):
                    if ff==0:
                        ii = ii1
                    else:
                        ii = ii1 + np.sum(ninp0[:ff])
                    fgrs[ii] = ff
                    lm[ii] = lm0tmp[ii1]
                    fobs[ii] = fobs0[ii1]
                    eobs[ii] = eobs0[ii1]
                MB.f_spec = True
            except Exception:
                pass
    except:
        print('No spec file is provided.')
        pass

    if MB.f_spec:
        nthin = 1

    #############################
    # READ BB photometry from CAT_BB:
    #############################
    if CAT_BB:
        fd0 = ascii.read(CAT_BB)
        id0 = fd0['id'].astype('str')
        ii0 = np.where(id0[:]==ID)
        try:
            id = fd0['id'][ii0]
        except:
            print('Could not find the column for [ID: %s] in the input BB catalog! Exiting.'%(ID))
            return False        
        if len(ii0) == 0:
            print('Could not find the column for [ID: %s] in the input BB catalog! Exiting.'%(ID))
            return False

        fbb = np.zeros(len(SFILT), dtype=float)
        ebb = np.zeros(len(SFILT), dtype=float)

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
            snbb0 = fbb0/ebb0
            fbb = flamtonu(lmbb0, fbb0, m0set=MB.m0set)
            ebb = fbb/snbb0
        else:
            snbb0 = fbb0/ebb0
            fbb = fbb0
            ebb = ebb0

    else:
        fbb = np.zeros(len(SFILT), dtype=float)
        ebb = np.zeros(len(SFILT), dtype=float)
        for ii in range(len(SFILT)):
            fbb[ii] = 0
            ebb[ii] = -99

    # Dust flux;
    if MB.f_dust:
        fdd = ascii.read(CAT_BB_DUST)
        id0 = fdd['id'].astype('str')
        ii0 = np.where(id0[:]==ID)
        try:
            id = fd0['id'][ii0]
        except:
            print('Could not find the column for [ID: %s] in the input BB catalog! Exiting.'%(ID))
            return False

        fbb_d = np.zeros(len(DFILT), dtype=float)
        ebb_d = np.zeros(len(DFILT), dtype=float)
        for ii in range(len(DFILT)):
            fbb_d[ii] = fdd['F%s'%(DFILT[ii])][ii0]
            ebb_d[ii] = fdd['E%s'%(DFILT[ii])][ii0]

    #############################
    # Getting Morphology params.
    #############################
    Amp = 0
    f_morp = False
    if MB.f_spec:
        LSF = get_LSF(inputs, DIR_EXTR, ID, lm)
    else:
        LSF = []
        lm = []
    try:
        if inputs['MORP'] == 'jwst-prism':
            MB.file_res = os.path.join(MB.config_path, 'jwst_nirspec_prism_disp.fits')
            if os.path.exists(MB.file_res):
                MB.f_prism = True
                LSF = []
    except:
        MB.f_prism = False
        pass

    if MB.f_prism:
        MB.f_diff_conv = False
        try:
            n_diff_conv = int(inputs['DIFF_CONV'])
            if n_diff_conv == 1:
                MB.f_diff_conv = True
                print('Differential template convolution is requested - this may take a while.')
        except:
            pass

    try:
        x_HI_input = float(inputs['x_HI'])
    except:
        x_HI_input = None

    ####################################
    # Start generating templates
    ####################################
    for zz in range(len(Z)):
        Zbest = Z[zz]
        Na = len(age)
        Ntmp = 1
        age_univ= MB.cosmo.age(zbest).value #, use_flat=True, **cosmo)

        for tt in range(len(tau)): # tau
            if zz == 0 and tt == 0:
                if delwave>0:
                    lm0_orig = spechdu['wavelength'][::nthin]
                    lm0 = np.arange(lm0_orig.min(), lm0_orig.max(), delwave)
                else:
                    lm0 = spechdu['wavelength'][::nthin]
                if not lammax == None and not MB.f_dust:
                    lm0 = lm0[(lm0 * (zbest+1) < lammax)]
                wave = lm0

            lmbest = np.zeros((Ntmp, len(lm0)), dtype=float)
            fbest = np.zeros((Ntmp, len(lm0)), dtype=float)
            lmbestbb = np.zeros((Ntmp, len(SFILT)), dtype=float)
            fbestbb = np.zeros((Ntmp, len(SFILT)), dtype=float)

            spec_mul = np.zeros((Na, len(lm0)), dtype=float)
            spec_mul_nu = np.zeros((Na, len(lm0)), dtype=float)
            spec_mul_nu_conv = np.zeros((Na, len(lm0)), dtype=float)

            ftmpbb = np.zeros((Na, len(SFILT)), dtype=float)
            ltmpbb = np.zeros((Na, len(SFILT)), dtype=float)

            ftmp_nu_int = np.zeros((Na, len(lm)), dtype=float)
            spec_av_tmp = np.zeros((Na, len(lm)), dtype=float)

            ms = np.zeros(Na, dtype=float)
            Ls = np.zeros(Na, dtype=float)
            ms[:] = mshdu['ms_'+str(zz)+'_'+str(tt)][:] # [:] is necessary.
            Ls[:] = mshdu['Ls_'+str(zz)+'_'+str(tt)][:]
            Fuv = np.zeros(Na, dtype=float)

            for ss in range(Na):
                #print(ss,tt,zz)
                if ss == 0 and tt == 0 and zz == 0:
                    DL = MB.cosmo.luminosity_distance(zbest).value * MB.Mpc_cm # Luminositydistance in cm
                    wavetmp = wave*(1.+zbest)

                if delwave>0:
                    fint = interpolate.interp1d(lm0_orig, spechdu['fspec_'+str(zz)+'_'+str(tt)+'_'+str(ss)][::nthin], kind='nearest', fill_value="extrapolate")
                    spec_mul[ss] = fint(lm0)
                else:
                    spec_mul[ss] = spechdu['fspec_'+str(zz)+'_'+str(tt)+'_'+str(ss)][::nthin]

                ##################
                # IGM attenuation.
                ##################
                if f_IGM:
                    # spec_av_tmp = madau_igm_abs(wave, spec_mul[ss,:], zbest, cosmo=MB.cosmo)
                    spec_av_tmp, x_HI = dijkstra_igm_abs(wave, spec_mul[ss,:], zbest, cosmo=MB.cosmo, x_HI=x_HI_input)
                else:
                    spec_av_tmp = spec_mul[ss,:]

                spec_mul_nu[ss,:] = flamtonu(wave, spec_av_tmp, m0set=MB.m0set)

                spec_mul_nu[ss,:] *= MB.Lsun/(4.*np.pi*DL**2/(1.+zbest))
                spec_mul_nu[ss,:] *= (1./Ls[ss])*tmp_norm # in unit of erg/s/Hz/cm2/ms[ss].
                ms[ss] *= (1./Ls[ss])*tmp_norm # M/L; 1 unit template has this mass in [Msolar].

                if len(lm)>0:
                    try:
                        spec_mul_nu_conv[ss,:] = convolve(spec_mul_nu[ss,:], LSF, boundary='extend')
                    except:
                        spec_mul_nu_conv[ss,:] = spec_mul_nu[ss,:]
                        if zz==0 and ss==0:
                            print('Kernel is too small. No convolution.')
                else:
                    spec_mul_nu_conv[ss,:] = spec_mul_nu[ss,:]

                if MB.f_spec:
                    #ftmp_nu_int[ss,:] = data_int(lm, wavetmp, spec_mul_nu_conv[ss,:])
                    ftmp_nu_int[ss,:] = data_int(lm, wavetmp, spec_mul_nu[ss,:])
                
                # Register filter response;
                ltmpbb[ss,:], ftmpbb[ss,:] = filconv(SFILT, wavetmp, spec_mul_nu[ss,:], DIR_FILT, MB=MB, f_regist=False)


                ##########################################
                # Writing out the templates to fits table.
                ##########################################
                if ss == 0 and tt == 0 and zz == 0:
                    # First file
                    nd1    = np.arange(0,len(lm),1)
                    nd3    = np.arange(10000,10000+len(ltmpbb[ss,:]),1)
                    nd_ap  = np.append(nd1,nd3)
                    lm_ap  = np.append(lm, ltmpbb[ss,:])

                    # ASDF
                    tree_spec.update({'wavelength':lm_ap})
                    tree_spec.update({'colnum':nd_ap})

                    # Second file
                    # ASDF
                    nd = np.arange(0,len(wavetmp),1)
                    tree_spec_full.update({'wavelength':wavetmp})
                    tree_spec_full.update({'colnum':nd})

                # ASDF
                # ???
                spec_ap = np.append(ftmp_nu_int[ss,:], ftmpbb[ss,:])
                tree_spec.update({'fspec_'+str(zz)+'_'+str(tt)+'_'+str(ss): spec_ap})
                tree_spec_full.update({'fspec_orig_'+str(zz)+'_'+str(tt)+'_'+str(ss): spec_mul_nu[ss,:]})                
                #tree_spec_full.update({'fspec_'+str(zz)+'_'+str(tt)+'_'+str(ss): spec_mul_nu_conv[ss,:]})

                # Nebular library;
                if fneb == 1 and MB.f_bpass==0 and ss==0 and tt==0:
                    if zz==0:
                        spec_mul_neb = np.zeros((len(Z), len(MB.logUs), len(lm0)), dtype=float)
                        spec_mul_neb_nu = np.zeros((len(Z), len(MB.logUs), len(lm0)), dtype=float)
                        spec_mul_neb_nu_conv = np.zeros((len(Z), len(MB.logUs), len(lm0)), dtype=float)
                        ftmpbb_neb = np.zeros((len(Z), len(MB.logUs), len(SFILT)), dtype=float)
                        ltmpbb_neb = np.zeros((len(Z), len(MB.logUs), len(SFILT)), dtype=float)
                        ftmp_neb_nu_int = np.zeros((len(Z), len(MB.logUs), len(lm)), dtype=float)

                    for uu in range(len(MB.logUs)):
                        if delwave>0:
                            fint = interpolate.interp1d(lm0_orig, spechdu['flux_nebular_Z%d_logU%d'%(zz,uu)][::nthin], kind='nearest', fill_value="extrapolate")
                            spec_mul_neb[zz,uu,:] = fint(lm0)
                        else:
                            spec_mul_neb[zz,uu,:] = spechdu['flux_nebular_Z%d_logU%d'%(zz,uu)][::nthin]
                        
                        con_neb = (spec_mul_neb[zz,uu,:]<0)
                        spec_mul_neb[zz,uu,:][con_neb] = 0
                        
                        if f_IGM:
                            # spec_neb_av_tmp = madau_igm_abs(wave, spec_mul_neb[zz,uu,:], zbest, cosmo=MB.cosmo)
                            spec_neb_av_tmp, x_HI = dijkstra_igm_abs(wave, spec_mul_neb[zz,uu,:], zbest, cosmo=MB.cosmo, x_HI=x_HI_input)
                            spec_mul_neb[zz,uu,:] = spec_neb_av_tmp

                        spec_mul_neb_nu[zz,uu,:] = flamtonu(wave, spec_mul_neb[zz,uu,:], m0set=MB.m0set)
                        spec_mul_neb_nu[zz,uu,:] *= MB.Lsun/(4.*np.pi*DL**2/(1.+zbest))
                        spec_mul_neb_nu[zz,uu,:] *= (1./Ls[ss])*tmp_norm # in unit of erg/s/Hz/cm2/ms[ss].
                        ltmpbb_neb[zz,uu,:], ftmpbb_neb[zz,uu,:] = filconv(SFILT, wavetmp, spec_mul_neb_nu[zz,uu,:], DIR_FILT, MB=MB, f_regist=False)

                        if MB.f_spec:
                            ftmp_neb_nu_int[zz,uu,:] = data_int(lm, wavetmp, spec_mul_neb_nu[zz,uu,:])

                        if MB.f_spec:
                            spec_mul_neb_nu_conv[zz,uu,:] = convolve_templates(wavetmp, spec_mul_neb_nu[zz,uu,:], LSF, boundary='extend', 
                                                                        f_prism=MB.f_prism, file_res=MB.file_res, redshift=zbest, f_diff_conv=MB.f_diff_conv)
                        else:
                            spec_mul_neb_nu_conv[zz,uu,:] = spec_mul_neb_nu[zz,uu,:]

                        tree_spec_full.update({'fspec_orig_nebular_Z%d_logU%d'%(zz,uu): spec_mul_neb_nu[zz,uu,:]})
                        tree_spec_full.update({'fspec_nebular_Z%d_logU%d'%(zz,uu): spec_mul_neb_nu_conv[zz,uu,:]})

                        spec_neb_ap = np.append(ftmp_neb_nu_int[zz,uu,:], ftmpbb_neb[zz,uu,:])
                        tree_spec.update({'fspec_nebular_Z%d_logU%d'%(zz,uu): spec_neb_ap})

            #########################
            # Summarize the ML
            #########################
            # ASDF
            tree_ML.update({'ML_'+str(zz)+'_'+str(tt): ms})

    #########################
    # Summarize the templates
    #########################
    tree['id'] = ID
    tree['z'] = zbest
    try:
        tree['x_HI'] = x_HI
    except:
        pass
    tree.update({'spec' : tree_spec})
    tree.update({'spec_full' : tree_spec_full})
    tree.update({'ML' : tree_ML})

    ######################
    # Add dust component;
    ######################
    if MB.f_dust:
        tree_spec_dust = {}
        tree_spec_dust_full = {}

        if DT0 == DT1:
            Temp = [DT0]
        else:
            Temp = np.arange(DT0,DT1,dDT)

        dellam_d = 1e3
        lambda_d = np.arange(1e3, 1e7, dellam_d) # RF wavelength, in AA. #* (1.+zbest) # 1um to 1000um; This has to be wide enough, to cut dust contribution at <1um.

        print('Reading dust table...')
        for tt in range(len(Temp)):
            if tt == 0:
                # For full;
                nd_d  = np.arange(0,len(lambda_d),1)

                # ASDF
                tree_spec_dust_full.update({'wavelength': lambda_d*(1.+zbest)})
                tree_spec_dust_full.update({'colnum': nd_d})

            #numin, numax, nmodel = 8, 3, 9
            numin, numax, nmodel = tt, 3, 9
            fnu_d = get_spectrum_draine(lambda_d, DL, zbest, numin, numax, nmodel, DIR_DUST=MB.DIR_DUST, m0set=MB.m0set)

            if False:
                for nn in range(0,11,1):
                    try:
                        fnu_d_tmp = get_spectrum_draine(lambda_d, DL, zbest, numin, numax, nn, DIR_DUST=MB.DIR_DUST, m0set=MB.m0set)
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

            if MB.f_spec:
                ftmp_nu_int_d = data_int(lm, lambda_d*(1.+zbest), fnu_d)
                ltmpbb_d = np.append(lm, ltmpbb_d)
                ftmpbb_d = np.append(ftmp_nu_int_d, ftmpbb_d)
                nd_db = np.arange(0, len(ftmpbb_d), 1)

            if tt == 0:
                # For conv;
                # ASDF
                tree_spec_dust.update({'wavelength': ltmpbb_d})
                tree_spec_dust.update({'colnum': nd_db})

            tree_spec_dust.update({'fspec_'+str(tt): ftmpbb_d})

        tree.update({'spec_dust' : tree_spec_dust})
        tree.update({'spec_dust_full' : tree_spec_dust_full})
        print('dust updated.')

    # Save;
    af = asdf.AsdfFile(tree)
    af.write_to(DIR_TMP + 'spec_all_' + ID + '.asdf', all_array_compression='zlib')

    # Re-register
    MB.af = af

    ##########################################
    # For observation.
    # Write out for the Multi-component fitting.
    ##########################################
    fw = open(DIR_TMP + 'spec_obs_' + ID + '.cat', 'w')
    fw.write('# BB data (>%d) in this file are not used in fitting.\n'%(ncolbb))

    for ii in range(len(lm)):
        g_offset = 1000 * fgrs[ii]
        if lm[ii]/(1.+zbest) > lamliml and lm[ii]/(1.+zbest) < lamlimu:
            fw.write('%d %.5f %.5e %.5e\n'%(ii+g_offset, lm[ii], fobs[ii], eobs[ii]))
        else:
            fw.write('%d %.5f 0 1000\n'%(ii+g_offset, lm[ii]))

    for ii in range(len(ltmpbb[0,:])):
        if SFILT[ii] in SKIPFILT:# data point to be skiped;
            fw.write('%d %.5f %.5e %.5e\n'%(ii+ncolbb, ltmpbb[0,ii], 0.0, fbb[ii]))
        elif  ebb[ii]>ebblim:
            fw.write('%d %.5f 0 1000\n'%(ii+ncolbb, ltmpbb[0,ii]))
        else:
            fw.write('%d %.5f %.5e %.5e\n'%(ii+ncolbb, ltmpbb[0,ii], fbb[ii], ebb[ii]))

    fw.close()    


    fw = open(DIR_TMP + 'spec_dust_obs_' + ID + '.cat', 'w')
    if MB.f_dust:
        nbblast = len(ltmpbb[0,:])+len(lm)
        for ii in range(len(ebb_d[:])):
            if ebb_d[ii]>ebblim:
                fw.write('%d %.5f 0 1000\n'%(ii+ncolbb+nbblast, ltmpbb_d[ii+nbblast]))
            else:
                fw.write('%d %.5f %.5e %.5e\n'%(ii+ncolbb+nbblast, ltmpbb_d[ii+nbblast], fbb_d[ii], ebb_d[ii]))
    fw.close()

    # BB phot
    fw = open(DIR_TMP + 'bb_obs_' + ID + '.cat', 'w')
    fw_rem = open(DIR_TMP + 'bb_obs_' + ID + '_removed.cat', 'w')
    for ii in range(len(ltmpbb[0,:])):
        if SFILT[ii] in SKIPFILT:# data point to be skiped;
            fw.write('%d %.5f %.5e %.5e %.1f %s\n'%(ii+ncolbb, ltmpbb[0,ii], 0.0, fbb[ii], FWFILT[ii]/2., SFILT[ii]))
            fw_rem.write('%d %.5f %.5e %.5e %.1f %s\n'%(ii+ncolbb, ltmpbb[0,ii], fbb[ii], ebb[ii], FWFILT[ii]/2., SFILT[ii]))
        elif ebb[ii]>ebblim:
            fw.write('%d %.5f 0 1000 %.1f %s\n'%(ii+ncolbb, ltmpbb[0,ii], FWFILT[ii]/2., SFILT[ii]))
        elif ebb[ii]<=0:
            fw.write('%d %.5f 0 -99 %.1f %s\n'%(ii+ncolbb, ltmpbb[0,ii], FWFILT[ii]/2., SFILT[ii]))
        else:
            fw.write('%d %.5f %.5e %.5e %.1f %s\n'%(ii+ncolbb, ltmpbb[0,ii], fbb[ii], ebb[ii], FWFILT[ii]/2., SFILT[ii]))
    fw.close()
    fw_rem.close()

    # Dust
    fw = open(DIR_TMP + 'bb_dust_obs_' + ID + '.cat', 'w')
    if MB.f_dust:
        for ii in range(len(ebb_d[:])):
            if  ebb_d[ii]>ebblim:
                fw.write('%d %.5f 0 1000 %.1f %s\n'%(ii+ncolbb+nbblast, ltmpbb_d[ii+nbblast], DFWFILT[ii]/2., DFILT[ii]))
            else:
                fw.write('%d %.5f %.5e %.5e %.1f %s\n'%(ii+ncolbb+nbblast, ltmpbb_d[ii+nbblast], fbb_d[ii], ebb_d[ii], DFWFILT[ii]/2., DFILT[ii]))
    fw.close()

    print('Done making templates at z=%.2f.\n'%zbest)

    return True


def get_spectrum_draine(lambda_d, DL, zbest, numin, numax, ndmodel,
    DIR_DUST='./DL07spec/', phi=0.055, m0set=25.0):
    '''
    Parameters
    ----------
    lambda_d : array
        Wavelength array, in AA.
    phi : float
        Eq.34 of Draine & Li 2007. (default: 0.055g/(ergs/s))
    DL : float
        in cm.

    Returns
    -------
    Interpolated dust emission in Fnu of m0=25.0. In units of Fnu/Msun

    Notes
    -----
    umins = ['0.10', '0.15', '0.20', '0.30', '0.40', '0.50', '0.70', '0.80', '1.00', '1.20',\
            '1.50', '2.00', '2.50', '3.00', '4.00', '5.00', '7.00', '8.00', '10.0', '12.0', '15.0',\
            '20.0', '25.0']
    umaxs = ['1e3', '1e4', '1e5', '1e6', '1e7']

    '''
    from .function import fnutonu
    import scipy.interpolate as interpolate

    Htokg = 1.66054e-27 # kg/H atom
    kgtomsun = 1.989e+30 # kg/Msun
    MsunperH = Htokg / kgtomsun # Msun/H
    HperMsun = kgtomsun / Htokg # N_Hatom per Msun 

    Jytoerg = 1e-23 # erg/s/cm2/Hz / Jy

    umins = ['0.10', '0.15', '0.20', '0.30', '0.40', '0.50', '0.70', '0.80', '1.00', '1.20',\
            '1.50', '2.00', '2.50', '3.00', '4.00', '5.00', '7.00', '8.00', '12.0', '15.0',\
            '20.0', '25.0']
    umaxs = ['1e3', '1e4', '1e5', '1e6', '1e7']
        
    dust_model = DIR_DUST+'draine07_models.txt'
    fd_model = ascii.read(dust_model)

    umin = umins[numin]
    umax = umaxs[numax]
    dmodel = fd_model['name'][ndmodel]

    if ndmodel == 6 or ndmodel == 1:
        data_start = 55
    else:
        data_start = 36

    file_dust = DIR_DUST + 'U%s/U%s_%s_%s.txt'%(umin, umin, umax, dmodel)
    print(file_dust)
    fd = ascii.read(file_dust, data_start=data_start)

    wave = fd['col1'] # in mu m.
    flux = fd['col2'] # nu*Pnu: erg/s H-1
    flux_dens = fd['col3'] # j_nu, emissivity: Jy cm2 sr-1 H-1
    #survey = fd['col4'] #

    fobs = flux_dens * Jytoerg / (4.*np.pi*DL**2/(1.+zbest)) * HperMsun
    # Jy cm2 sr-1 H-1 * erg/s/cm2/Hz / Jy / (cm2 * sr) * (H/Msun) = erg/s/cm2/Hz / Msun
    # i.e. observed flux density from a dust of Msun at the distance of DL.

    fnu = fnutonu(fobs, m0set=m0set, m0input=-48.6)
    # Fnu / Msun

    fint = interpolate.interp1d(wave*1e4, fnu, kind='nearest', fill_value="extrapolate")
    yy_s = fint(lambda_d)
    con_yys = (lambda_d<1e4) # Interpolation cause some error??
    yy_s[con_yys] = 0

    return yy_s


def sim_spec(lmin, fin, sn):
    '''
    SIMULATION of SPECTRA.
    
    Parameters
    ----------
    sn : float array
        
    Returns
    -------
    frand : float array

    erand :  float array


    :func:`get_spectrum_draine`
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


def check_library(MB, af, nround=3):
    '''Check library if it has a consistency setup as input file.

    Returns
    -------
    flag : bool
        
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
    MB.logger.info('Checking the template library...')

    # No. of age;
    if MB.SFH_FORM==-99:
        if len(af['ML']['ms_0']) != len(MB.age):
            MB.logger.error('No of age pixels:', len(MB.age), len(af['ML']['ms_0']))
            flag = False
    else:
        flag = True

    # Matallicity:
    for aa in range(len(Zall)):
        if Zall[aa] != af['Z%d'%(aa)]:
            MB.logger.error('Z:', Zall[aa], af['Z%d'%(aa)])
            flag = False

    if MB.SFH_FORM==-99:
        # Age:
        for aa in range(len(MB.age)):
            if round(MB.age[aa],nround) != round(af['age%d'%(aa)],nround):
                MB.logger.error('age:', MB.age[aa], af['age%d'%(aa)])
                flag = False
        # Tau (e.g. ssp/csp):
        for aa in range(len(MB.tau0)):
            if round(MB.tau0[aa]) != round(af['tau0%d'%(aa)]):
                MB.logger.error('tau0:', MB.tau0[aa], af['tau0%d'%(aa)])
                flag = False
    else:
        # Age:
        for aa in range(len(MB.ageparam)):
            if round(MB.ageparam[aa]) != round(af['age%d'%(aa)]):
                MB.logger.error('age:', MB.ageparam[aa], af['age%d'%(aa)])
                flag = False
        for aa in range(len(MB.tau)):
            if round(MB.tau[aa]) != round(af['tau%d'%(aa)]):
                MB.logger.error('tau:', MB.tau[aa], af['tau%d'%(aa)])
                flag = False

    # IMF:
    if MB.nimf != af['nimf']:
        MB.logger.error('nimf:', MB.nimf, af['nimf'])
        flag = False

    if not flag:
        MB.logger.error('# Specified - Template')

    return flag


def smooth_template_diff(waves, fluxes, Rs, Rs_template, f_diff_conv=False):
    '''from Gina's code
    Parameters
    ----------
    wave, flux, Rs: arrays
        All have the same size.
    sigma_template : float
        Sigma of the template spectrum, in km/s

    Notes
    -----
    fsps templates have a resolution of ~2.5A FWHM from 3750AA - 7200AA restframe, and much lower (R~200 or so, but not actually well defined) outside this range. 
    '''
    from scipy import ndimage
    c_light = 299792.458 # speed of light in km/s

    fluxes_conv = np.zeros(len(fluxes), float)

    if f_diff_conv:

        fluxes_tmp = np.zeros(len(fluxes), float)
        for nw,wave in enumerate(waves):
            fluxes_tmp[:] = 0

            # Find resolution for roman pixel
            if nw == 0:
                delta_wvl = waves[nw+1] - waves[nw]
            else:
                delta_wvl = waves[nw] - waves[nw-1]

            sigma_inst = c/(Rs[nw]*2.355)

            # FWHM_gal = 1e4*np.sqrt(0.97*1.89)/R_median
            sigma_template = c/(Rs_template[nw]*2.355)

            # Smooth template to match roman resolution
            smoothing_sigma = sigma_inst / sigma_template

            # @@@ This part could be shortcut;
            # smoothed = ndimage.gaussian_filter(fluxes_tmp, smoothing_sigma)
            xMof = np.arange(-5, 5.1, .1) # dimension must be even.
            Amp = 1.
            LSF = gauss(xMof, Amp, smoothing_sigma)

            # find indices that include pixel's wavelength and interpolate to find
            # smoothed flux
            fluxes_tmp[nw] = fluxes[nw]
            smoothed = convolve(fluxes_tmp, LSF, boundary='extend')
            fluxes_conv[:] += smoothed[:]

    else:
        delta_wvl = np.nanmedian(np.diff(waves))

        sigma_inst = c/(Rs[:]*2.355)
        sigma_template = c/(Rs_template[:]*2.355)

        # Smooth template to match roman resolution
        smoothing_sigma = np.nanmedian(sigma_inst / sigma_template)
        # fluxes_conv = ndimage.gaussian_filter(fluxes, smoothing_sigma)
        xMof = np.arange(-5, 5.1, .1) # dimension must be even.
        Amp = 1.
        LSF = gauss(xMof, Amp, smoothing_sigma)
        fluxes_conv = convolve(fluxes, LSF, boundary='extend')
        # print('smoothing_sigma is %.2f'%smoothing_sigma)

    return fluxes_conv

        
def get_LSF(inputs, DIR_EXTR, ID, lm, wave_repr=4000, c=3e18,
    sig_temp_def=50., redshift=None):
    '''
    Load Morphology params, and returns LSF

    Parameters
    ----------
    lm : float array
        wavelength array for the observed spectrum, in AA.

    Returns
    -------
    LSF

    '''
    lists_morp = ['moffat', 'gauss', 'jwst-prism']
    Amp = 0
    f_morp = False
    if inputs['MORP'] in lists_morp:
        if inputs['MORP'] in lists_morp[:2]:
            try:
                mor_file = inputs['MORP_FILE'].replace('$ID','%s'%(ID))
                fm = ascii.read(DIR_EXTR + mor_file)
                Amp = fm['A']
                gamma = fm['gamma']
                if inputs['MORP'] == 'moffat':
                    alp = fm['alp']
                else:
                    alp = 0
                f_morp = True
            except Exception:
                msg = '`MORP_FILE` cannot be found.\nNo morphology convolution.'
                print_err(msg, exit=False)
                pass
    else:
        msg = 'MORP Keywords does not match.\nNo morphology convolution.'
        print_err(msg, exit=False)

    ############################
    # Template convolution;
    # fsps templates have a resolution of ~2.5A FWHM from 3750AA - 7200AA restframe, and much lower (R~200 or so, but not actually well defined) outside this range. 
    ############################
    try:
        sig_temp = float(inputs['SIG_TEMP'])
        print('Template is set to %.1f km/s.'%(sig_temp))
    except:
        sig_temp = sig_temp_def
        print('Template resolution is unknown.')
        print('Set to %.1f km/s.'%(sig_temp))

    # @@@ Below assumes a constant R over wavelength range;
    iixlam = np.argmin(np.abs(lm-wave_repr)) # Around 4000 AA
    if iixlam == len(lm)-1:
        dellam = lm[iixlam] - lm[iixlam-1] # AA/pix
    else:
        dellam = lm[iixlam+1] - lm[iixlam] # AA/pix
    R_temp = c / (sig_temp*1e3*1e10)
    sig_temp_pix = np.median(lm) / R_temp / dellam # delta v in pixel;
    sig_inst = 0 #65 #km/s for Manga

    # If grism;
    if f_morp:
        print('\nStarting templates convolution (intrinsic morphology).')
        if gamma>sig_temp_pix:# and False:
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
            msg = 'Something is wrong with the convolution file. Exiting.'
            print_err(msg, exit=True)
            LSF = []

    else: # For slit spectroscopy. To be updated...
        print('Templates convolution (intrinsic velocity).')
        try:
            vdisp = float(inputs['VDISP'])
            dellam = lm[1] - lm[0] # AA/pix
            R_disp = c/(np.sqrt(vdisp**2-sig_inst**2)*1e3*1e10)
            vdisp_pix = np.median(lm) / R_disp / dellam # delta v in pixel;
            print('Templates are convolved at %.2f km/s.'%(vdisp))
            if vdisp_pix-sig_temp_pix>0:
                sig_conv = np.sqrt(vdisp_pix**2-sig_temp_pix**2)
            else:
                sig_conv = 0
        except:
            vdisp = 0.
            # print('Templates are not convolved.')
            sig_conv = 0 #np.sqrt(sig_temp_pix**2)
            pass
        xMof = np.arange(-5, 5.1, .1) # dimension must be even.
        Amp = 1.
        LSF = gauss(xMof, Amp, sig_conv)

    return LSF


def convolve_templates(wave, spec, LSF, boundary='extend', f_prism=False, file_res=None, redshift=None, f_diff_conv=False):
    '''
    file_res : str
        From the official jdocs.
    '''
    if len(LSF) > 1:
        spec_conv = convolve(spec, LSF, boundary='extend')

    elif f_prism:
        fd_res = fits.open(file_res)[1].data
        R_res = fd_res['R']
        wave_res = fd_res['WAVELENGTH'] # in um;
        wave_res *= 1e4
        fint = interpolate.interp1d(wave_res, R_res, kind='nearest', fill_value="extrapolate")
        Rs_res_interp = fint(wave)
        Rs_template = np.zeros(len(wave),float) + 200 # Assuming R=200;
        if redshift != None:
            mask_hr = np.where((wave/(1+redshift) > 3750) & ((wave/(1+redshift) < 7200)))
            Rs_template[mask_hr] = wave[mask_hr] / 2.5
        else:
            mask_hr = None
        # Smooth;
        spec_conv = smooth_template_diff(wave, spec, Rs_res_interp, Rs_template, f_diff_conv=f_diff_conv)

    else:
        spec_conv = spec

    return spec_conv
