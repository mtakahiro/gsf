import numpy as np
import matplotlib.pyplot as plt
import copy
import asdf,sys,os
import scipy.interpolate as interpolate

# import os
from astropy.io import ascii,fits
from astropy.convolution import Gaussian1DKernel, convolve

import gsf
from .function import get_ind,get_imf_str,get_lognorm
from .utils_maketmp import get_nebular_template

INDICES = ['G4300', 'Mgb', 'Fe5270', 'Fe5335', 'NaD', 'Hb', 'Fe4668', 'Fe5015', 'Fe5709', 'Fe5782', 'Mg1', 'Mg2', 'TiO1', 'TiO2']


def validate_and_save_tree(tree, file_out, dir_tmp='templates/', keys=['spec','ML','lick']):
    """"""
    for key in keys:
        if key not in tree:
            return False
    # Save
    af = asdf.AsdfFile(tree)
    af.write_to(os.path.join(dir_tmp, file_out), all_array_compression='zlib')
    return True


def make_tmp_z0(MB, lammin=100, lammax=160000, tau_lim=0.001, force_no_neb=False, Zforce=None, f_mp=True,
                smooth_uv=False):
    '''
    This is for the preparation of default template, with FSPS, at z=0.
    Should be run before SED fitting.

    Parameters
    ----------
    :class:`gsf.fitting.Mainbody` : class
        Mainbody class, that contains attributes.

    lammin : float, optional
        Minimum value of the rest-frame wavelength of the template, in AA.

    lammax : float, optional
        Maximum value of the rest-frame wavelength of the template, in AA.

    tau_lim : float, optional
        Maximum value of tau of the template, in Gyr. Tau smaller than this 
        value would be approximated by SSP.

    force_no_neb : bool
        Turn this on that you are very much sure do not want to include emission line templates, 
        maybe to save some time running z0 module.

    f_mp : bool
        Multiprocessing.
    smooth_uv : bool
        Experimental - smoothing stellar spectra at rf-UV, as they look wiggling...
    '''
    import fsps
    nimf = MB.nimf
    age = MB.age
    tau0 = MB.tau0
    fneb = MB.fneb
    if not force_no_neb:
        fneb = True
    DIR_TMP = MB.DIR_TMP
    Na = len(age)

    if not Zforce == None:
        file_out = 'spec_all_Z%.1f.asdf'%Zforce
    else:
        file_out = 'spec_all.asdf'
    Z = MB.Zall
    NZ = len(Z)
    
    # Current age in Gyr;
    age_univ = MB.cosmo.age(0).value

    MB.logger.warning('Making templates at z=0 - This may take a while.')
    MB.logger.info('IMF=%d'%(nimf))
    
    tree_spec = {}
    tree_ML = {}
    tree_lick = {}

    MB.logger.info('tau is the width of each age bin.')
    tau_age = np.zeros(Na,dtype='float')
    age_age = np.zeros(Na,dtype='float')
    flagz = True
    for zz in range(len(Z)):

        if not Zforce == None and Z[zz] != Zforce:
            continue

        for pp in range(len(tau0)):
            spall = [] # For ssp model
            ms = np.zeros(Na, dtype='float')
            Ls = np.zeros(Na, dtype='float')
            mlost = np.zeros(Na, dtype='float')
            LICK = np.zeros((Na,len(INDICES)), dtype='float')
            tau0_old = 0
            for ss in range(Na):
                #
                # Determining tau for each age bin;
                #
                # if zz == 0 and pp == 0 and ss == 0 and age[ss]<0.01 and MB.fneb:
                #     MB.logger.warning('Your input AGE includes <0.01Gyr --- fsps interpolates spectra, and you may not get accurate SEDs.')

                # 1.Continuous age bin;
                if int(tau0[pp]) == 99:
                    if ss==0:
                        delTl = age[ss]
                        delTu = (age[ss+1]-age[ss])/2.
                        delT = delTu + delTl
                        tautmp = delT
                        agetmp = age[ss]+delTu
                    elif ss == len(age)-1:
                        delTl = (age[ss]-age[ss-1])/2.
                        delTu = delTl
                        delT = delTu + delTl
                        tautmp = delT
                        agetmp = age[ss]+delTu
                    else:
                        delTl = (age[ss]-age[ss-1])/2.
                        delTu = (age[ss+1]-age[ss])/2.
                        delT = delTu + delTl
                        tautmp = delT
                        agetmp = age[ss]+delTu
                # 2.A fixed-age bin;
                elif tau0[pp] > 0.0:
                    tautmp = tau0[pp]
                    agetmp = age[ss] + tautmp/2.
                # 3.SSP;
                else: # =Negative tau;
                    tautmp = tau_lim
                    agetmp = age[ss]

                # Keep tau in header;
                tau_age[ss] = tautmp
                age_age[ss] = agetmp

                #
                # Then, make sps.
                #
                f_add_dust = False # Not yet. Or never...

                if tautmp != tau0_old:

                    if int(tau0[pp]) == 99:
                        MB.logger.info('CSP is applied.')
                        MB.logger.info('At t=%.3f, tau is %.3f Gyr' %(age[ss],tautmp))

                        kwargs = {
                            'compute_vega_mags':False, 'zcontinuous':1, 'imf_type':nimf, 'sfh':1, 'logzsol':Z[zz], 'dust_type':2, 'dust2':0.0, 
                            'tau':20, 'const':0, 'sf_start':0, 'sf_trunc':tautmp, 'tburst':13, 'fburst':0
                            }

                        sptmp = fsps.StellarPopulation(add_neb_emission=0, **kwargs) # Lsun/Hz

                        if fneb:
                            esptmp = fsps.StellarPopulation(add_neb_emission=1, **kwargs)

                        if f_add_dust:
                            dsptmp = fsps.StellarPopulation(duste_gamma=0.01, duste_umin=1.0, duste_qpah=3.5, fagn=0.0, **kwargs)

                        if MB.fagn:
                            asptmp = fsps.StellarPopulation(add_neb_emission=0, fagn=1.0, **kwargs)

                    elif tau0[pp] > 0.0:
                        MB.logger.info('At t=%.3f, fixed tau, %.3f, is applied.'%(age[ss],tautmp))

                        kwargs = {
                            'compute_vega_mags':False, 'zcontinuous':1, 'imf_type':nimf, 'sfh':1, 'logzsol':Z[zz], 'dust_type':2, 'dust2':0.0, 
                            'tau':20, 'const':0, 'sf_start':0, 'sf_trunc':tautmp, 'tburst':13, 'fburst':0
                            }

                        sptmp = fsps.StellarPopulation(add_neb_emission=0, **kwargs) # Lsun/Hz

                        if fneb:
                            esptmp = fsps.StellarPopulation(add_neb_emission=1, **kwargs)

                        if f_add_dust:
                            dsptmp = fsps.StellarPopulation(duste_gamma=0.01, duste_umin=1.0, duste_qpah=3.5, fagn=0.0, **kwargs)

                        if MB.fagn:
                            asptmp = fsps.StellarPopulation(add_neb_emission=0, fagn=1.0, **kwargs)

                    else: # =Negative tau;
                        MB.logger.info('At t=%.3f, SSP (%.3f) is applied.'%(age[ss],tautmp))

                        kwargs = {
                            'compute_vega_mags':False, 'zcontinuous':1, 'imf_type':nimf, 'sfh':0, 'logzsol':Z[zz], 'dust_type':2, 'dust2':0.0, 
                            #'tau':20, 'const':0, 'sf_start':0, 'sf_trunc':tautmp, 'tburst':13, 'fburst':0
                            }

                        sptmp = fsps.StellarPopulation(add_neb_emission=0, **kwargs) # Lsun/Hz

                        if fneb:
                            esptmp = fsps.StellarPopulation(add_neb_emission=1, **kwargs)

                        if f_add_dust:
                            dsptmp = fsps.StellarPopulation(duste_gamma=0.01, duste_umin=1.0, duste_qpah=3.5, fagn=0.0, **kwargs)

                        if MB.fagn:
                            asptmp = fsps.StellarPopulation(add_neb_emission=0, fagn=1.0, **kwargs)
                else:
                    MB.logger.info('At t=%.3f, tau is %.3f Gyr' %(age[ss],tautmp))
                    MB.logger.info('Skip fsps, by using previous library.')

                tau0_old = tautmp
                sp = sptmp
                MB.logger.info('Z:%d/%d, t:%d/%d, %s, %s'%(zz+1, len(Z), pp+1, len(tau0), sp.libraries[0].decode("utf-8") , sp.libraries[1].decode("utf-8")))

                wave0, flux0 = sp.get_spectrum(tage=age[ss], peraa=True) # Lsun/AA

                # Post process RF-UV?
                if smooth_uv and age[ss]<0.01:
                    wave0, flux0 = smooth_spectrum(wave0, flux0, wmin=0, wmax=3000, sigma=10)

                con = (wave0>lammin) & (wave0<lammax)
                wave, flux = wave0[con], flux0[con]
                mlost[ss] = sp.stellar_mass / sp.formed_mass

                # Check UV based SFR?
                # from .function import flamtonu,fnutolam,check_line_man,loadcpkl,get_Fuv,filconv_fast,printProgressBar,filconv,get_uvbeta,print_err
                # fnu = flamtonu(wave, flux, m0set=-48.6)
                # Luv = get_Fuv(wave, fnu, lmin=1600, lmax=2800) # Lsun / delta_wl
                # print(sp.stellar_mass, age[ss], Z[zz], Luv)

                if fneb and pp == 0 and ss == 0:

                    esptmp.params['gas_logz'] = Z[zz] # gas metallicity, assuming = Zstel
                    stellar_mass_tmp = sp.stellar_mass

                    # Loop within logU;
                    for nlogU, logUtmp in enumerate(MB.logUs):

                        esptmp.params['gas_logu'] = logUtmp
                        esp, flux_nebular = get_nebular_template(wave, flux, sp, esptmp, age[ss], lammin, lammax)

                        tree_spec.update({'flux_nebular_Z%d'%zz+'_logU%d'%nlogU: flux_nebular})
                        tree_spec.update({'emline_wavelengths_Z%d'%zz+'_logU%d'%nlogU: esp.emline_wavelengths})
                        tree_spec.update({'emline_luminosity_Z%d'%zz+'_logU%d'%nlogU: esp.emline_luminosity})
                        tree_spec.update({'emline_mass_Z%d'%zz+'_logU%d'%nlogU: esp.stellar_mass - stellar_mass_tmp})

                if MB.fagn and pp == 0 and ss == 0:

                    asptmp.params['gas_logz'] = Z[zz] # gas metallicity, assuming = Zstel
                    stellar_mass_tmp = sp.stellar_mass

                    # Loop within logU;
                    for nAGNTAU, AGNTAUtmp in enumerate(MB.AGNTAUs):
                        asptmp.params['agn_tau'] = AGNTAUtmp
                        asp = asptmp

                        tage_agn = age[ss]

                        ewave0, eflux0 = asp.get_spectrum(tage=tage_agn, peraa=True)
                        if age[ss] != tage_agn:
                            sp_tmp = sp.copy()
                            wave0_tmp, flux0_tmp = sp_tmp.get_spectrum(tage=tage_neb, peraa=True) # Lsun/AA
                            _, flux_tmp = wave0_tmp[con], flux0_tmp[con]
                        else:
                            _, flux_tmp = wave, flux

                        con = (ewave0>lammin) & (ewave0<lammax)
                        flux_agn = eflux0[con] - flux_tmp
                        # Eliminate some negatives. Mostly on <912A;
                        con_neg = flux_agn<0
                        flux_agn[con_neg] = 0

                        tree_spec.update({'flux_agn_Z%d'%zz+'_AGNTAU%d'%nAGNTAU: flux_agn})
                        tree_spec.update({'agn_mass_Z%d'%zz+'_AGNTAU%d'%nAGNTAU: asp.stellar_mass - stellar_mass_tmp})

                if f_add_dust:
                    wave0_d, flux0_d = dsptmp.get_spectrum(tage=age[ss], peraa=True)
                    con = (wave0_d>lammin) & (wave0_d<lammax)
                    wave_d, flux_d = wave0[con], flux0[con]
                    plt.plot(wave0, flux0, linestyle='--')
                    plt.plot(wave0_d, flux0_d, linestyle='--')
                    plt.xlim(4000,100000)
                    plt.xscale('log')
                    plt.show()

                # Mass-Luminosity
                ms[ss] = sp.stellar_mass # Survived mass
                Ls[ss] = 10**sp.log_lbol
                LICK[ss,:] = get_ind(wave, flux)

                if flagz and ss == 0 and pp == 0:
                    # ASDF Big tree;
                    # Create header;
                    tree = {
                        'isochrone': '%s'%(sp.libraries[0].decode("utf-8")),
                        'library': '%s'%(sp.libraries[1].decode("utf-8")),
                        'imf': get_imf_str(nimf),
                        'nimf': nimf,
                        'version_gsf': gsf.__version__
                    }
                    tree.update({'age': MB.age})
                    tree.update({'Z': MB.Zall})
                    if fneb:
                        tree.update({'logUMIN': MB.logUMIN})
                        tree.update({'logUMAX': MB.logUMAX})
                        tree.update({'DELlogU': MB.DELlogU})
                    if MB.fagn:
                        tree.update({'AGNTAUMIN': MB.AGNTAUMIN})
                        tree.update({'AGNTAUMAX': MB.AGNTAUMAX})
                        tree.update({'DELAGNTAU': MB.DELAGNTAU})

                    tree_spec.update({'wavelength': wave})
                    flagz = False

                tree_spec.update({'fspec_'+str(zz)+'_'+str(ss)+'_'+str(pp): flux})

            if pp == 0:
                # use tau0[0] as representative for M/L and index.
                for ll in range(len(INDICES)):
                    # ASDF
                    tree_lick.update({INDICES[ll]+'_'+str(zz): LICK[:,ll]})

                tree_ML.update({'ms_'+str(zz): ms})
                tree_ML.update({'Ls_'+str(zz): Ls})
                tree_ML.update({'frac_mass_survive_'+str(zz): mlost})
                tree_ML.update({'realtau_'+str(zz): ms})

    # Write;
    for aa in range(len(age)):
        tree.update({'realtau%d(Gyr)'%(aa): tau_age[aa]})
    for aa in range(len(age)):
        tree.update({'realage%d(Gyr)'%(aa): age_age[aa]})
    for aa in range(len(age)):
        tree.update({'age%d'%(aa): age[aa]})
    for aa in range(len(Z)):
        tree.update({'Z%d'%(aa): Z[aa]})
    for aa in range(len(tau0)):
        tree.update({'tau0%d'%(aa): tau0[aa]})

    # Index, Mass-to-light;
    tree.update({'spec' : tree_spec})
    tree.update({'ML' : tree_ML})
    tree.update({'lick' : tree_lick})

    # Save
    assert validate_and_save_tree(tree, file_out, dir_tmp=DIR_TMP) == True


def make_tmp_z0_tau(MB, lammin=100, lammax=160000, Zforce=None): 
    '''
    This is for the preparation of default template, with FSPS, at z=0.
    Should be run before SED fitting.

    Parameters
    ----------
    :class:`gsf.fitting.Mainbody` : class
        Mainbody class, that contains attributes.

    lammin : float, optional
        Minimum value of the rest-frame wavelength of the template, in AA.

    lammax : float, optional
        Maximum value of the rest-frame wavelength of the template, in AA.

    tau_lim : float, optional
        Maximum value of tau of the template, in Gyr. Tau smaller than this 
        value would be approximated by SSP.
    '''
    import asdf
    import fsps
    import gsf
    
    nimf = MB.nimf
    age = MB.ageparam # in linear.
    fneb = MB.fneb
    DIR_TMP = MB.DIR_TMP
    Na = len(age)

    tau = MB.tau
    sfh = MB.SFH_FORM

    if not Zforce == None:
        file_out = 'spec_all_Z%.1f.asdf'%Zforce
    else:
        file_out = 'spec_all.asdf'
    Z = MB.Zall
    NZ = len(Z)
    
    # Current age in Gyr;
    age_univ = MB.cosmo.age(0).value

    NZ = len(Z)
    Nt = len(tau)
    Na = len(age)

    ms = np.zeros(Na, dtype='float')
    Ls = np.zeros(Na, dtype='float')

    print('#######################################')
    print('Making templates at z=0, IMF=%d'%(nimf))
    print('#######################################')

    tree_spec = {}
    tree_ML = {}
    tree_lick = {}

    print('tau is the width of each age bin.')
    flagz = True
    for zz in range(len(Z)):
        if not Zforce == None and Z[zz] != Zforce:
            continue
        for ss in range(len(tau)):

            if 10**tau[ss]<0.01:
                # then do SSP
                print('!! tau is <0.01Gyr. SSP is applied. !!') # This corresponds to the min-tau of fsps.
                sp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1, imf_type=nimf, sfh=0, logzsol=Z[zz], dust_type=2, dust2=0.0) # Lsun/Hz
                if fneb == 1:
                    esptmp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1, imf_type=nimf, sfh=0, logzsol=Z[zz], dust_type=2, dust2=0.0, add_neb_emission=1) # Lsun/Hz
                if MB.fagn:
                    asptmp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1, imf_type=nimf, sfh=0, logzsol=Z[zz], dust_type=2, dust2=0.0, fagn=1.0) # Lsun/Hz
            elif sfh<5:
                sp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1, imf_type=nimf, sfh=sfh, logzsol=Z[zz], dust_type=2, dust2=0.0, tau=10**tau[ss], const=0, sf_start=0, sf_trunc=0, tburst=13, fburst=0) # Lsun/Hz
                if fneb == 1:
                    esptmp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1, imf_type=nimf, sfh=sfh, logzsol=Z[zz], dust_type=2, dust2=0.0, tau=10**tau[ss], const=0, sf_start=0, sf_trunc=0, tburst=13, fburst=0, add_neb_emission=1) # Lsun/Hz
                if MB.fagn:
                    asptmp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1, imf_type=nimf, sfh=sfh, logzsol=Z[zz], dust_type=2, dust2=0.0, tau=10**tau[ss], const=0, sf_start=0, sf_trunc=0, tburst=13, fburst=0, fagn=1.0) # Lsun/Hz
            elif sfh==6: # Custom SFH
                sp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=3, imf_type=nimf, sfh=3, dust_type=2, dust2=0.0)
                if fneb == 1:
                    sp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=3, imf_type=nimf, sfh=3, dust_type=2, dust2=0.0, add_neb_emission=1)
                if MB.fagn:
                    asptmp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=3, imf_type=nimf, sfh=3, dust_type=2, dust2=0.0, fagn=1.0)
                print('Log normal is used. !!')

            print('Z:%d/%d, t:%d/%d, %s, %s'%(zz+1, len(Z), ss+1, len(tau), sp.libraries[0].decode("utf-8") , sp.libraries[1].decode("utf-8")))
            ms = np.zeros(Na)
            Ls = np.zeros(Na)
            LICK = np.zeros((Na,len(INDICES)), dtype='float')
            mlost = np.zeros(Na, dtype='float')

            for tt in range(len(age)):

                if zz == 0 and ss == 0 and tt == 0 and age[tt]<0.01:
                    MB.logger.warning('Your input AGE includes <0.01Gyr --- fsps interpolates spectra, and you may not get accurate SEDs.')

                if sfh==6: # Tabular SFH.
                    tuniv_hr = np.arange(0,age_univ,0.01) # in Gyr
                    T0 = np.log(age[tt])
                    sfh_hr_in = get_lognorm(tuniv_hr, tau[ss], T0) # tau in log Gyr
                    zh_hr_in  = tuniv_hr*0 + 10**Z[zz] # metallicity is constant
                    sp.set_tabular_sfh(tuniv_hr, sfh_hr_in, zh_hr_in)
                    wave0, flux0 = sp.get_spectrum(tage=age_univ, peraa=True) # if peraa=True, in unit of L/AA
                else:
                    wave0, flux0 = sp.get_spectrum(tage=age[tt], peraa=True) # if peraa=True, in unit of L/AA

                con = (wave0>lammin) & (wave0<lammax)
                wave, flux = wave0[con], flux0[con]
                ms[tt] = sp.stellar_mass
                if np.isnan(ms[tt]):
                    print('Error at age %.3f and tau %.3f'%(age[tt], tau[ss]))
                    sys.exit()

                Ls[tt] = 10**sp.log_lbol
                LICK[tt,:] = get_ind(wave, flux)
                mlost[tt] = sp.stellar_mass / sp.formed_mass

                if fneb and tt == 0 and ss == 0:

                    esptmp.params['gas_logz'] = Z[zz] # gas metallicity, assuming = Zstel
                    stellar_mass_tmp = sp.stellar_mass

                    # Loop within logU;
                    for nlogU, logUtmp in enumerate(MB.logUs):

                        esptmp.params['gas_logu'] = logUtmp
                        esp, flux_nebular = get_nebular_template(wave, flux, sp, esptmp, age[ss], lammin, lammax)

                        tree_spec.update({'flux_nebular_Z%d'%zz+'_logU%d'%nlogU: flux_nebular})
                        tree_spec.update({'emline_wavelengths_Z%d'%zz+'_logU%d'%nlogU: esp.emline_wavelengths})
                        tree_spec.update({'emline_luminosity_Z%d'%zz+'_logU%d'%nlogU: esp.emline_luminosity})
                        tree_spec.update({'emline_mass_Z%d'%zz+'_logU%d'%nlogU: esp.stellar_mass - stellar_mass_tmp})

                if MB.fagn and tt == 0 and ss == 0:

                    asptmp.params['gas_logz'] = Z[zz] # gas metallicity, assuming = Zstel
                    stellar_mass_tmp = sp.stellar_mass

                    # Loop within logU;
                    for nAGNTAU, AGNTAUtmp in enumerate(MB.AGNTAUs):
                        asptmp.params['agn_tau'] = AGNTAUtmp
                        asp = asptmp

                        tage_agn = age[tt]

                        ewave0, eflux0 = asp.get_spectrum(tage=tage_agn, peraa=True)
                        if age[tt] != tage_agn:
                            sp_tmp = sp.copy()
                            wave0_tmp, flux0_tmp = sp_tmp.get_spectrum(tage=tage_agn, peraa=True) # Lsun/AA
                            _, flux_tmp = wave0_tmp[con], flux0_tmp[con]
                        else:
                            _, flux_tmp = wave, flux

                        con = (ewave0>lammin) & (ewave0<lammax)
                        flux_agn = eflux0[con] - flux_tmp
                        # Eliminate some negatives. Mostly on <912A;
                        con_neg = flux_agn<0
                        flux_agn[con_neg] = 0

                        tree_spec.update({'flux_agn_Z%d'%zz+'_AGNTAU%d'%nAGNTAU: flux_agn})
                        tree_spec.update({'agn_mass_Z%d'%zz+'_AGNTAU%d'%nAGNTAU: asp.stellar_mass - stellar_mass_tmp})

                if flagz and ss == 0 and tt == 0:
                    # ASDF Big tree;
                    # Create header;
                    tree = {
                        'isochrone': '%s'%(sp.libraries[0].decode("utf-8")),
                        'library': '%s'%(sp.libraries[1].decode("utf-8")),
                        'nimf': nimf,
                        'version_gsf': gsf.__version__
                    }
                    tree.update({'age': MB.age})
                    tree.update({'tau': MB.tau})
                    tree.update({'Z': MB.Zall})
                    if fneb == 1:
                        tree.update({'logUMIN': MB.logUMIN})
                        tree.update({'logUMAX': MB.logUMAX})
                        tree.update({'DELlogU': MB.DELlogU})
                    if MB.fagn:
                        tree.update({'AGNTAUMIN': MB.AGNTAUMIN})
                        tree.update({'AGNTAUMAX': MB.AGNTAUMAX})
                        tree.update({'DELAGNTAU': MB.DELAGNTAU})

                    tree_spec.update({'wavelength': wave})
                    flagz = False

                tree_spec.update({'fspec_'+str(zz)+'_'+str(ss)+'_'+str(tt): flux})

            for ll in range(len(INDICES)):
                # ASDF
                tree_lick.update({INDICES[ll]+'_'+str(zz)+'_'+str(ss): LICK[:,ll]})

            tree_ML.update({'ms_'+str(zz)+'_'+str(ss): ms})
            tree_ML.update({'Ls_'+str(zz)+'_'+str(ss): Ls})
            tree_ML.update({'frac_mass_survive_'+str(zz)+'_'+str(ss): mlost})

    # Write;
    for aa in range(len(tau)):
        tree.update({'tau%d'%(aa): tau[aa]})
    for aa in range(len(age)):
        tree.update({'age%d'%(aa): age[aa]})
    for aa in range(len(Z)):
        tree.update({'Z%d'%(aa): Z[aa]})

    # Index, Mass-to-light;
    tree.update({'spec' : tree_spec})
    tree.update({'ML' : tree_ML})
    tree.update({'lick' : tree_lick})

    #
    assert validate_and_save_tree(tree, file_out, dir_tmp=DIR_TMP) == True


def make_tmp_z0_bpass_v2p3(MB, lammin=100, lammax=160000, Zforce=None, Zsun=0.02, 
                           alpha='+00', ):
    '''
    This is for the preparation of default template, with BPASS v2.3 templates, at z=0.
    Should be run before SED fitting.

    Parameters
    ----------
    :class:`gsf.fitting.Mainbody` : class
        Mainbody class, that contains attributes.

    lammin : float, optional
        Minimum value of the rest-frame wavelength of the template, in AA.

    lammax : float, optional
        Maximum value of the rest-frame wavelength of the template, in AA.

    DIR_BPASS : str, optional
        Path to the ditectory where BPASS templates are storesd.

    BPASS_ver : str, optional
        Version of BPASS. Used to identify template files.

    Zsun : float, optional
        Metallicity of templates, in units of absolute value (e.g. Zsun=0.02 for BPASS).
    '''
    nimf = MB.nimf
    if nimf == 0: # Salpeter
        imf_str = '135all_100'
    elif nimf == 1:
        imf_str = '_chab100'
    else:
        imf_str = ''

    if not Zforce == None:
        file_out = 'spec_all_Z%.1f.asdf'%Zforce
    else:
        file_out = 'spec_all.asdf'

    Z = MB.Zall
    age = MB.age
    tau0 = MB.tau0
    fneb = MB.fneb
    if fneb:
        MB.logger.warning('Currently, BPASS does not have option of nebular emission.')
        fneb = False
    DIR_TMP= MB.DIR_TMP

    # binary?
    f_bin = MB.f_bin
    #f_bin = True
    if f_bin==1:
        bin_str = 'bin'
    else:
        bin_str = 'str'

    DIR_LIB = MB.DIR_BPASS + 'BPASS%s/BPASS%s_%s-imf%s/'%(MB.BPASS_ver,MB.BPASS_ver,bin_str,imf_str)

    NZ = len(Z)
    Na = len(age)

    # Current age in Gyr;
    age_univ = MB.cosmo.age(0).value

    print('#######################################')
    print('Making templates at z=0, IMF=%d'%(nimf))
    print('#######################################')

    tree_spec = {}
    tree_ML = {}
    tree_lick = {}

    tau_age = np.zeros(Na,dtype='float')
    age_age = np.zeros(Na,dtype='float')

    flagz = True
    for zz in range(len(Z)):
        if not Zforce == None and Z[zz] != Zforce:
            continue
        #
        # open spectral file;
        #
        if 10**(Z[zz])*Zsun>1e-4:
            zstrtmp = round(10**(Z[zz])*Zsun/10,5)
            zstrtmp2 = '%.6s'%(zstrtmp)
            z_str = zstrtmp2[3:]
            if len(z_str)<3:
                z_str = z_str+'0'
        elif 10**(Z[zz])*Zsun>1e-5:
            z_str = 'em4'
        else:
            z_str = 'em5'

        file_sed = '%sspectra-%s-imf%s.z%s.dat'%(DIR_LIB,bin_str,imf_str,z_str)
        file_stm = '%sstarmass-%s-imf%s.z%s.dat'%(DIR_LIB,bin_str,imf_str,z_str)
        fd_sed = ascii.read(file_sed)
        fd_stm = ascii.read(file_stm)

        wave0 = fd_sed['col1']
        age_stm = fd_stm['col1']
        mass_formed = fd_stm['col2'][0]

        ncols = 52
        nage_temp = np.arange(2,ncols+1,1)
        lage_temp = (6+0.1*(nage_temp-2))
        age_temp = 10**(6+0.1*(nage_temp-2)) # in yr

        # 'tau is the width of each age bin.'
        for pp in range(len(tau0)):
            spall = [] # For sps model
            ms = np.zeros(Na, dtype='float')
            Ls = np.zeros(Na, dtype='float')
            mlost = np.zeros(Na, dtype='float')
            LICK = np.zeros((Na,len(INDICES)), dtype='float')
            tau0_old = 0

            for ss in range(Na):
                mass_formed_tot = 0
                flux0 = np.zeros(len(wave0),'float')
                #
                # Determining tau for each age bin;
                #
                if int(tau0[pp]) == 99:
                    if ss==0:
                        tautmp = age[ss]
                        agetmp = age[ss]/2.
                        con_tau= np.where((age_temp[:]<=age[ss]*1e9) & (age_temp[:]>0))
                        for sstmp in con_tau[0][-1:]:
                            flux0  += fd_sed['col%d'%(sstmp+2)][:]
                            ms[ss] += fd_stm['col2'][sstmp]
                            mass_formed_tot += mass_formed
                    else:
                        tautmp = age[ss] - age[ss-1]
                        agetmp = age[ss] - (age[ss]-age[ss-1])/2.
                        con_tau= np.where((age_temp[:]<=age[ss]*1e9) & (age_temp[:]>age[ss-1]*1e9))
                        for sstmp in con_tau[0][-1:]:
                            flux0  += fd_sed['col%d'%(sstmp+2)][:]
                            ms[ss] += fd_stm['col2'][sstmp]
                            mass_formed_tot += mass_formed

                elif tau0[pp] > 0.0:
                    if ss==0 and age[ss]<tau0[pp]:
                        tautmp = age[ss]
                        agetmp = age[ss]/2.
                        con_tau= np.where((age_temp[:]<=age[ss]*1e9) & (age_temp[:]>0))
                        for sstmp in con_tau[0]:
                            flux0  += fd_sed['col%d'%(sstmp+2)][:]
                            ms[ss] += fd_stm['col2'][sstmp]
                            mass_formed_tot += mass_formed

                    elif (age[ss]-age[ss-1]) < tau0[pp]:
                        tautmp = age[ss] - age[ss-1]
                        agetmp = age[ss] - (age[ss]-age[ss-1])/2.
                        con_tau= np.where((age_temp[:]<=age[ss]*1e9) & (age_temp[:]>age[ss-1]*1e9))
                        for sstmp in con_tau[0]:
                            flux0  += fd_sed['col%d'%(sstmp+2)][:]
                            ms[ss] += fd_stm['col2'][sstmp]
                            mass_formed_tot += mass_formed

                    else:
                        tautmp = tau0[pp]
                        agetmp = age[ss] - tau0[pp]/2.
                        con_tau= np.where((age_temp[:]<=age[ss]*1e9) & (age_temp[:]>(age[ss] - tau0[pp])*1e9))
                        for sstmp in con_tau[0]:
                            flux0  += fd_sed['col%d'%(sstmp+2)][:]
                            ms[ss] += fd_stm['col2'][sstmp]
                            mass_formed_tot += mass_formed

                else: # =Negative tau; SSP
                    iis   = np.argmin(np.abs(age[ss] - age_temp[:]/1e9))
                    iistm = np.argmin(np.abs(age[ss] - 10**age_stm[:]/1e9))
                    if ss==0:
                        tautmp = 10**6.05 / 1e9 # in Gyr
                        agetmp = age[ss]/2.
                    else:
                        tautmp = ( 10**(lage_temp[iis]+0.05) - 10**(lage_temp[iis]-0.05) ) / 1e9 # Gyr
                        agetmp = (age[ss]+age[ss-1])/2.

                    flux0 = fd_sed['col%d'%(iis+2)]
                    ms[ss] = fd_stm['col2'][iistm]
                    mass_formed_tot += mass_formed

                # Keep tau in header;
                tau_age[ss] = tautmp
                age_age[ss] = agetmp

                # Then. add flux if tau > 0.
                con = (wave0>lammin) & (wave0<lammax)
                wave, flux = wave0[con], flux0[con]
                # Temp
                mlost[ss] =  ms[ss] / mass_formed_tot

                Ls[ss] = np.sum(flux0) # BPASS sed is in Lsun.
                LICK[ss,:] = get_ind(wave, flux)

                if flagz and ss == 0 and pp == 0:
                    # ASDF Big tree;
                    # Create header;
                    tree = {
                        'isochrone': 'BPASS',
                        'library': 'BPASS',
                        'nimf': nimf,
                        'version_gsf': gsf.__version__
                    }

                    tree.update({'age': MB.age})
                    tree.update({'Z': MB.Zall})
                    if fneb:
                        tree.update({'logUMIN': MB.logUMIN})
                        tree.update({'logUMAX': MB.logUMAX})
                        tree.update({'DELlogU': MB.DELlogU})
                    if MB.fagn:
                        tree.update({'AGNTAUMIN': MB.AGNTAUMIN})
                        tree.update({'AGNTAUMAX': MB.AGNTAUMAX})
                        tree.update({'DELAGNTAU': MB.DELAGNTAU})

                    # ASDF
                    tree_spec.update({'wavelength': wave})
                    flagz = True

                # ASDF
                tree_spec.update({'fspec_'+str(zz)+'_'+str(ss)+'_'+str(pp): flux})

                if fneb:
                    # ASDF
                    tree_spec.update({'efspec_'+str(zz)+'_'+str(ss)+'_'+str(pp): eflux})

            if pp == 0:
                # use tau0[0] as representative for M/L and index.
                for ll in range(len(INDICES)):
                    # ASDF
                    tree_lick.update({INDICES[ll]+'_'+str(zz): LICK[:,ll]})

                col1 = fits.Column(name='ms_'+str(zz), format='E', unit='Msun', array=ms)
                tree_ML.update({'ms_'+str(zz): ms})
                col2 = fits.Column(name='Ls_'+str(zz), format='E', unit='Lsun', array=Ls)
                tree_ML.update({'Ls_'+str(zz): Ls})
                col3 = fits.Column(name='fm_'+str(zz), format='E', unit='', array=mlost)
                tree_ML.update({'frac_mass_survive_'+str(zz): mlost})
                col4 = fits.Column(name='tau_'+str(zz), format='E', unit='Gyr', array=tau_age)
                tree_ML.update({'realtau_'+str(zz): ms})


    # Write;
    for aa in range(len(age)):
        tree.update({'realtau%d(Gyr)'%(aa): tau_age[aa]})
    for aa in range(len(age)):
        tree.update({'realage%d(Gyr)'%(aa): age_age[aa]})
    for aa in range(len(age)):
        tree.update({'age%d'%(aa): age[aa]})
    for aa in range(len(Z)):
        tree.update({'Z%d'%(aa): Z[aa]})
    for aa in range(len(tau0)):
        tree.update({'tau0%d'%(aa): tau0[aa]})

    # Index, Mass-to-light;
    tree.update({'spec' : tree_spec})
    tree.update({'ML' : tree_ML})
    tree.update({'lick' : tree_lick})

    # Save
    assert validate_and_save_tree(tree, file_out, dir_tmp=DIR_TMP) == True


def make_tmp_z0_bpass(MB, lammin=100, lammax=160000, Zforce=None, Zsun=0.02, 
                      upmass=300, 
                      couple_neb=False, logu_neb=-2.0,
                      age_neb=0.01,
                      ):
    '''
    This is for the preparation of default template, with BPASS templates, at z=0.
    Should be run before SED fitting.

    Parameters
    ----------
    :class:`gsf.fitting.Mainbody` : class
        Mainbody class, that contains attributes.

    lammin : float, optional
        Minimum value of the rest-frame wavelength of the template, in AA.

    lammax : float, optional
        Maximum value of the rest-frame wavelength of the template, in AA.

    DIR_BPASS : str, optional
        Path to the ditectory where BPASS templates are storesd.

    BPASS_ver : str, optional
        Version of BPASS. Used to identify template files.

    Zsun : float, optional
        Metallicity of templates, in units of absolute value (e.g. Zsun=0.02 for BPASS).

    age_neb : float
        age in Gyr, with which the nebular component is calculated. 
        if couple_nebular, this is not effective.

    couple_neb: bool
        if True, it attaches the nebular component to the main flux component. This case, eflux would be 0. Thus, Aneb makes no sense. 
        This case, logU will be fixed.
        else, nebular component is calculated assuming the youngest age among the age bin, and will be controlled by Aneb.
    '''
    nimf = MB.nimf
    if nimf == 0: # Salpeter
        imf_str = '135all_%d'%(upmass)
        imf_str = '135_%d'%(upmass)
    elif nimf == 1:
        imf_str = '_chab%d'%(upmass)
    else:
        imf_str = ''

    if Zforce is not None:
        file_out = 'spec_all_Z%.1f.asdf'%Zforce
    else:
        file_out = 'spec_all.asdf'

    Z = MB.Zall
    age = MB.age
    tau0 = MB.tau0
    DIR_TMP= MB.DIR_TMP

    # binary?
    f_bin = MB.f_bin
    if f_bin==1:
        bin_str = 'bin'
    else:
        bin_str = 'str'
    DIR_LIB = MB.DIR_BPASS + 'BPASS%s/BPASS%s_%s-imf%s/'%(MB.BPASS_ver,MB.BPASS_ver,bin_str,imf_str)

    if MB.fneb:
        if bin_str == 'bin' and imf_str == '135_300':
            MB.logger.info('nebular is on')
            MB.fneb = True
            DIR_LIB_NEB = MB.DIR_BPASS + 'BPASS%s-Cloudy/cloudyspec_outputs/'%(MB.BPASS_ver)
        else:
            MB.logger.error('Currently, BPASS nebular emission is only available for imf135_300 bin.')
            MB.logger.error('Turn `ADD_NEBULAE` off')
            print(bin_str, imf_str)
            MB.fneb = False
            return False

    Na = len(age)

    # Current age in Gyr;
    age_univ = MB.cosmo.age(0).value

    print('#######################################')
    print('Making templates at z=0, IMF=%d'%(nimf))
    print('#######################################')

    tree_spec = {}
    tree_ML = {}
    tree_lick = {}

    tau_age = np.zeros(Na,dtype='float')
    age_age = np.zeros(Na,dtype='float')

    flagz = True
    for zz in range(len(Z)):
        if not Zforce == None and Z[zz] != Zforce:
            continue
        #
        # open spectral file;
        #
        if 10**(Z[zz])*Zsun>1e-3:
            zstrtmp = round(10**(Z[zz])*Zsun/10,5)
            zstrtmp2 = '%.6s'%(zstrtmp)
            z_str = zstrtmp2[3:]
            if len(z_str)<3:
                z_str = z_str+'0'
        elif 10**(Z[zz])*Zsun>1e-4:
            z_str = 'em4'
        else:
            z_str = 'em5'

        file_sed = '%sspectra-%s-imf%s.z%s.dat'%(DIR_LIB,bin_str,imf_str,z_str)
        file_stm = '%sstarmass-%s-imf%s.z%s.dat'%(DIR_LIB,bin_str,imf_str,z_str)
        fd_sed = ascii.read(file_sed)
        fd_stm = ascii.read(file_stm)

        wave0 = fd_sed['col1']
        age_stm = fd_stm['col1']
        mass_formed = fd_stm['col2'][0] # i.e. 1e6 Msun formed at t=1e6yr.

        ncols = 52
        nage_temp = np.arange(2,ncols+1,1)
        lage_temp = (6+0.1*(nage_temp-2))
        age_temp = 10**(6+0.1*(nage_temp-2)) # in yr

        if MB.fneb:
            mstel_emi = 1e6
            ncols_emi = 15
            Lunit_emi = 3.848e33
            nage_emi = np.arange(2,ncols_emi+1,1)
            age_emi = 10**(6+0.1*(nage_emi-2))
            logUs_bpass = np.linspace(-4.0,-1.0,7)
            v_blass_neb = '11'
            logUs_bpass_str = ['v%sg'%v_blass_neb, 'v%sf'%v_blass_neb, 'v%se'%v_blass_neb, 'v%sd'%v_blass_neb, 'v%sc'%v_blass_neb, 'v%sb'%v_blass_neb, 'v%sa'%v_blass_neb]
            if age_neb is not None:
                iix_age_neb = np.argmin(np.abs(age_neb-age))
            else:
                iix_age_neb = 0 # Use the youngest

        # 'tau is the width of each age bin.'
        for pp in range(len(tau0)):
            spall = [] # For sps model
            ms = np.zeros(Na, dtype='float')
            Ls = np.zeros(Na, dtype='float')
            mlost = np.zeros(Na, dtype='float')
            LICK = np.zeros((Na,len(INDICES)), dtype='float')
            # tau0_old = 0

            for ss in range(Na):
                mass_formed_tot = 0
                flux0 = np.zeros(len(wave0),'float')

                #
                # Determining tau for each age bin;
                #
                if int(tau0[pp]) == 99:
                    if ss==0:
                        tautmp = age[ss]
                        agetmp = age[ss]/2.
                        con_tau= np.where((age_temp[:]<=age[ss]*1e9) & (age_temp[:]>0))
                        for sstmp in con_tau[0][-1:]:
                            flux0  += fd_sed['col%d'%(sstmp+2)][:]
                            ms[ss] += fd_stm['col2'][sstmp]
                            mass_formed_tot += mass_formed
                    else:
                        tautmp = age[ss] - age[ss-1]
                        agetmp = age[ss] - (age[ss]-age[ss-1])/2.
                        con_tau= np.where((age_temp[:]<=age[ss]*1e9) & (age_temp[:]>age[ss-1]*1e9))
                        for sstmp in con_tau[0][-1:]:
                            flux0  += fd_sed['col%d'%(sstmp+2)][:]
                            ms[ss] += fd_stm['col2'][sstmp]
                            mass_formed_tot += mass_formed

                elif tau0[pp] > 0.0:
                    if ss==0 and age[ss]<tau0[pp]:
                        tautmp = age[ss]
                        agetmp = age[ss]/2.
                        con_tau= np.where((age_temp[:]<=age[ss]*1e9) & (age_temp[:]>0))
                        for sstmp in con_tau[0]:
                            flux0  += fd_sed['col%d'%(sstmp+2)][:]
                            ms[ss] += fd_stm['col2'][sstmp]
                            mass_formed_tot += mass_formed

                    elif (age[ss]-age[ss-1]) < tau0[pp]:
                        tautmp = age[ss] - age[ss-1]
                        agetmp = age[ss] - (age[ss]-age[ss-1])/2.
                        con_tau= np.where((age_temp[:]<=age[ss]*1e9) & (age_temp[:]>age[ss-1]*1e9))
                        for sstmp in con_tau[0]:
                            flux0  += fd_sed['col%d'%(sstmp+2)][:]
                            ms[ss] += fd_stm['col2'][sstmp]
                            mass_formed_tot += mass_formed

                    else:
                        tautmp = tau0[pp]
                        agetmp = age[ss] - tau0[pp]/2.
                        con_tau= np.where((age_temp[:]<=age[ss]*1e9) & (age_temp[:]>(age[ss] - tau0[pp])*1e9))
                        for sstmp in con_tau[0]:
                            flux0  += fd_sed['col%d'%(sstmp+2)][:]
                            ms[ss] += fd_stm['col2'][sstmp]
                            mass_formed_tot += mass_formed

                else: # =Negative tau; SSP
                    iis   = np.argmin(np.abs(age[ss] - age_temp[:]/1e9))
                    iistm = np.argmin(np.abs(age[ss] - 10**age_stm[:]/1e9))
                    if ss==0:
                        tautmp = 10**6.05 / 1e9 # in Gyr
                        agetmp = age[ss]/2.
                    else:
                        tautmp = ( 10**(lage_temp[iis]+0.05) - 10**(lage_temp[iis]-0.05) ) / 1e9 # Gyr
                        agetmp = (age[ss]+age[ss-1])/2.

                    flux0 = fd_sed['col%d'%(iis+2)]
                    ms[ss] = fd_stm['col2'][iistm]
                    mass_formed_tot += mass_formed

                # Keep tau in header;
                tau_age[ss] = tautmp
                age_age[ss] = agetmp

                # Then. add flux if tau > 0.
                con = (wave0>lammin) & (wave0<lammax)
                wave, flux = wave0[con], flux0[con] # flux is Lsun / AA

                # # Check UV based SFR?
                # from .function import flamtonu,fnutolam,check_line_man,loadcpkl,get_Fuv,filconv_fast,printProgressBar,filconv,get_uvbeta,print_err
                # fnu = flamtonu(wave, flux, m0set=-48.6)
                # Luv = get_Fuv(wave, fnu, lmin=1450, lmax=1650) # Lsun / delta_wl
                # print(mass_formed_tot, age[ss], Z[zz], Luv)

                # Temp
                mlost[ss] = ms[ss] / mass_formed_tot
                Ls[ss] = np.sum(flux0) # BPASS sed is in Lsun.
                LICK[ss,:] = get_ind(wave, flux)

                if flagz and ss == 0 and pp == 0:
                    # ASDF Big tree;
                    # Create header;
                    tree = {
                        'isochrone': 'BPASS',
                        'library': 'BPASS',
                        'nimf': nimf,
                        'version_gsf': gsf.__version__
                    }

                    tree.update({'age': MB.age})
                    tree.update({'Z': MB.Zall})
                    if MB.fneb:
                        tree.update({'logUMIN': MB.logUMIN})
                        tree.update({'logUMAX': MB.logUMAX})
                        tree.update({'DELlogU': MB.DELlogU})
                    if MB.fagn:
                        tree.update({'AGNTAUMIN': MB.AGNTAUMIN})
                        tree.update({'AGNTAUMAX': MB.AGNTAUMAX})
                        tree.update({'DELAGNTAU': MB.DELAGNTAU})

                    # ASDF
                    tree_spec.update({'wavelength': wave})
                    flagz = True

                if not couple_neb:
                    # ASDF
                    tree_spec.update({'fspec_'+str(zz)+'_'+str(ss)+'_'+str(pp): flux})

                    # BPASS neb;
                    if MB.fneb and pp == 0 and ss == iix_age_neb:

                        if zz == 0:
                            MB.logger.info('BPASS nebular component is calculated using age=%.1e'%(age[ss]))

                        for nlogU, logUtmp in enumerate(MB.logUs):
                            # Each file has 16 columns and 30,000 rows. The first column lists a wavelength in angstroms, 
                            # and each remaining column n (n>1) holds the model flux for the population at an age of 10^(6+0.1*(n-2)) years at that wavelength. The range of ages covered is log(age/years)=6.0-7.5
                            # for nlogU, logUtmp in enumerate(MB.logUs):
                            # The units of flux are log_10(ergs/s per Angstrom), normalised for a cluster of 1e6 Msun formed in a single instantaneous burst. The total luminosity of the SED can be simply calculated by summing all the rows together
                            iix = np.argmin(np.abs(logUtmp - logUs_bpass))
                            logu_str = logUs_bpass_str[iix]
                            file_sed_emi = '%scloudyspec_imf%s_z%s_%s_%s.sed'%(DIR_LIB_NEB,imf_str,z_str,bin_str,logu_str)
                            fd_sed_emi = ascii.read(file_sed_emi)
                            con_emi_data = (fd_sed_emi['col1'] != 'Total_Power')
                            fd_sed_emi = fd_sed_emi[con_emi_data]
                            wave0_emi = np.asarray([float(s) for s in fd_sed_emi['col1']])
                            flux0_emi = np.zeros(len(wave0_emi),'float')
                            # Repeat the same but for emission;

                            #
                            # Determining tau for each age bin;
                            #
                            # Only ssp available;
                            iis = np.argmin(np.abs(age[ss] - age_emi[:]/1e9))
                            if iis+2 < ncols_emi:
                                flux0_emi = 10**fd_sed_emi['col%d'%(iis+2)]
                            else:
                                flux0_emi = flux0_emi[:] * 0

                            femi = interpolate.interp1d(wave0_emi, flux0_emi, kind='linear', fill_value="extrapolate")
                            flux_nebular = femi(wave)
                            emline_luminosity = np.sum(flux0_emi)

                            flux_nebular_only = flux_nebular/Lunit_emi-flux
                            con_neg = flux_nebular_only<0
                            flux_nebular_only[con_neg] = 0

                            # ASDF
                            # tree_spec.update({'efspec_'+str(zz)+'_'+str(ss)+'_'+str(pp): eflux})
                            tree_spec.update({'flux_nebular_Z%d'%zz+'_logU%d'%nlogU: flux_nebular_only}) # in Lsun/AA
                            tree_spec.update({'emline_wavelengths_Z%d'%zz+'_logU%d'%nlogU: wave})
                            tree_spec.update({'emline_luminosity_Z%d'%zz+'_logU%d'%nlogU: emline_luminosity}) # in Lsun
                            tree_spec.update({'emline_mass_Z%d'%zz+'_logU%d'%nlogU: mstel_emi}) # in Msun

                else:
                    MB.logUFIX = logu_neb
                    MB.nlogU = 1
                    MB.logUMIN = MB.logUFIX
                    MB.logUMAX = MB.logUFIX
                    MB.DELlogU = 0
                    MB.logUs = np.asarray([MB.logUMAX])

                    iix = np.argmin(np.abs(logu_neb - logUs_bpass))
                    logu_str = logUs_bpass_str[iix]
                    file_sed_emi = '%scloudyspec_imf%s_z%s_%s_%s.sed'%(DIR_LIB_NEB,imf_str,z_str,bin_str,logu_str)
                    fd_sed_emi = ascii.read(file_sed_emi)
                    con_emi_data = (fd_sed_emi['col1'] != 'Total_Power')
                    fd_sed_emi = fd_sed_emi[con_emi_data]
                    wave0_emi = np.asarray([float(s) for s in fd_sed_emi['col1']])
                    flux0_emi = np.zeros(len(wave0_emi),'float')

                    if pp == 0 and ss == 0 and zz == 0:
                        MB.logger.info('BPASS nebular component is calculated using logU=%.1f'%(logu_neb))

                    #
                    # Determining tau for each age bin;
                    #
                    # Only ssp available;
                    iis = np.argmin(np.abs(age[ss] - age_emi[:]/1e9))
                    if iis+2 < ncols_emi:
                        flux0_emi = 10**fd_sed_emi['col%d'%(iis+2)]
                    else:
                        flux0_emi = flux0_emi[:] * 0
                    emline_luminosity = np.nansum(flux0_emi)

                    femi = interpolate.interp1d(wave0_emi, flux0_emi, kind='linear', fill_value="extrapolate")
                    flux_nebular = femi(wave)
                    con_neg = flux_nebular<0
                    flux_nebular[con_neg] = 0

                    flux_nebular_only = flux_nebular/Lunit_emi-flux
                    con_neg = flux_nebular_only<0
                    flux_nebular_only[con_neg] = 0

                    # ASDF
                    L_neb_tmp = np.nansum(flux_nebular/Lunit_emi)
                    tree_spec.update({'fspec_'+str(zz)+'_'+str(ss)+'_'+str(pp): flux_nebular/Lunit_emi * Ls[ss]/L_neb_tmp})
                    # if zz == 0:
                    #     plt.close()
                    #     plt.plot(wave, flux)
                    #     plt.plot(wave, flux_nebular/Lunit_emi * Ls[ss]/L_neb_tmp)
                    #     plt.show()
                    tree_spec.update({'flux_nebular_Z%d'%zz+'_logU%d'%0: flux_nebular_only*0}) # in Lsun/AA
                    tree_spec.update({'emline_wavelengths_Z%d'%zz+'_logU%d'%0: wave})
                    tree_spec.update({'emline_luminosity_Z%d'%zz+'_logU%d'%0: emline_luminosity}) # in Lsun
                    tree_spec.update({'emline_mass_Z%d'%zz+'_logU%d'%0: mstel_emi}) # in Msun

            # plt.legend(loc=0)
            # plt.show()
            # hoge

            # Register M/Ls;
            if pp == 0:
                # use tau0[0] as representative for M/L and index.
                for ll in range(len(INDICES)):
                    # ASDF
                    tree_lick.update({INDICES[ll]+'_'+str(zz): LICK[:,ll]})

                col1 = fits.Column(name='ms_'+str(zz), format='E', unit='Msun', array=ms)
                tree_ML.update({'ms_'+str(zz): ms})
                col2 = fits.Column(name='Ls_'+str(zz), format='E', unit='Lsun', array=Ls)
                tree_ML.update({'Ls_'+str(zz): Ls})
                col3 = fits.Column(name='fm_'+str(zz), format='E', unit='', array=mlost)
                tree_ML.update({'frac_mass_survive_'+str(zz): mlost})
                col4 = fits.Column(name='tau_'+str(zz), format='E', unit='Gyr', array=tau_age)
                tree_ML.update({'realtau_'+str(zz): ms})

    # Write;
    for aa in range(len(age)):
        tree.update({'realtau%d(Gyr)'%(aa): tau_age[aa]})
    for aa in range(len(age)):
        tree.update({'realage%d(Gyr)'%(aa): age_age[aa]})
    for aa in range(len(age)):
        tree.update({'age%d'%(aa): age[aa]})
    for aa in range(len(Z)):
        tree.update({'Z%d'%(aa): Z[aa]})
    for aa in range(len(tau0)):
        tree.update({'tau0%d'%(aa): tau0[aa]})

    # Index, Mass-to-light;
    tree.update({'spec' : tree_spec})
    tree.update({'ML' : tree_ML})
    tree.update({'lick' : tree_lick})

    # Save
    assert validate_and_save_tree(tree, file_out, dir_tmp=DIR_TMP) == True


def make_tmp_z0_general(MB, lammin=100, lammax=160000, Zforce=None, Zsun=0.02, 
                      upmass=300, 
                      couple_neb=False, logu_neb=-2.0,
                      age_neb=0.01,
                      ):
    '''
    This is for the preparation of default template, with general templates, at z=0.
    A list of templates (ascii) should be provided, with the following format:
    # path logZ logT logU Ms
    where logZ is metallicity (log, in units of Zsun), 
    logT is the age (log, in units of Gyr),
    Ms is the stellar mass per the template's flux, in Msun.
    logU is only for nebular component; if set to -99, then the template is treated as stellar.
    Each template file should have the following format:
    # wave flux
    Should be run before SED fitting.

    Parameters
    ----------
    :class:`gsf.fitting.Mainbody` : class
        Mainbody class, that contains attributes.

    lammin : float, optional
        Minimum value of the rest-frame wavelength of the template, in AA.

    lammax : float, optional
        Maximum value of the rest-frame wavelength of the template, in AA.

    DIR_BPASS : str, optional
        Path to the ditectory where BPASS templates are storesd.

    BPASS_ver : str, optional
        Version of BPASS. Used to identify template files.

    Zsun : float, optional
        Metallicity of templates, in units of absolute value (e.g. Zsun=0.02 for BPASS).

    age_neb : float
        age in Gyr, with which the nebular component is calculated. 
        if couple_nebular, this is not effective.

    couple_neb: bool
        if True, it attaches the nebular component to the main flux component. This case, eflux would be 0. Thus, Aneb makes no sense. 
        This case, logU will be fixed.
        else, nebular component is calculated assuming the youngest age among the age bin, and will be controlled by Aneb.
    '''
    nimf = MB.nimf
    if nimf == 0: # Salpeter
        imf_str = '135all_%d'%(upmass)
        imf_str = '135_%d'%(upmass)
    elif nimf == 1:
        imf_str = '_chab%d'%(upmass)
    else:
        imf_str = ''

    if Zforce is not None:
        file_out = 'spec_all_Z%.1f.asdf'%Zforce
    else:
        file_out = 'spec_all.asdf'

    Z = MB.Zall
    age = MB.age
    tau0 = MB.tau0
    fneb = MB.fneb
    DIR_TMP= MB.DIR_TMP

    if fneb:
        MB.logger.warning('Currently, BPASS nebular emission is only available for imf135_300 bin.')
        if bin_str == 'bin' and imf_str == '135_300':
            print('nebular is on')
            fneb = True
            DIR_LIB_NEB = MB.DIR_BPASS + 'BPASS%s-Cloudy/cloudyspec_outputs/'%(MB.BPASS_ver)
        else:
            print('nebular is off')
            print(bin_str, imf_str)
            fneb = False

    NZ = len(Z)
    Na = len(age)

    # Current age in Gyr;
    age_univ = MB.cosmo.age(0).value

    print('#######################################')
    print('Making templates at z=0, IMF=%d'%(nimf))
    print('#######################################')

    tree_spec = {}
    tree_ML = {}
    tree_lick = {}

    tau_age = np.zeros(Na,dtype=float)
    age_age = np.zeros(Na,dtype=float)

    # General templates;
    fd_temp = ascii.read(MB.file_temp)
    logZs_temp = fd_temp['logZ']
    logTs_temp = fd_temp['logT']
    Ms_temp = fd_temp['Ms']

    flagz = True
    for zz in range(len(Z)):
        if not Zforce == None and Z[zz] != Zforce:
            continue

        iiz = np.where(logZs_temp == Z[zz])
        if len(iiz[0]) == 0:
            continue

        #
        # open spectral file;
        #
        if 10**(float(Z[zz]))*Zsun>1e-3:
            zstrtmp = round(10**(Z[zz])*Zsun/10,5)
            zstrtmp2 = '%.6s'%(zstrtmp)
            z_str = zstrtmp2[3:]
            if len(z_str)<3:
                z_str = z_str+'0'
        elif 10**(float(Z[zz]))*Zsun>1e-4:
            z_str = 'em4'
        else:
            z_str = 'em5'

        if fneb:
            mstel_emi = 1e6
            ncols_emi = 15
            Lunit_emi = 3.848e33
            nage_emi = np.arange(2,ncols_emi+1,1)
            age_emi = 10**(6+0.1*(nage_emi-2))
            logUs_bpass = np.linspace(-4.0,-1.0,7)
            v_blass_neb = '11'
            logUs_bpass_str = ['v%sg'%v_blass_neb, 'v%sf'%v_blass_neb, 'v%se'%v_blass_neb, 'v%sd'%v_blass_neb, 'v%sc'%v_blass_neb, 'v%sb'%v_blass_neb, 'v%sa'%v_blass_neb]
            if age_neb is not None:
                iix_age_neb = np.argmin(np.abs(age_neb-age))
            else:
                iix_age_neb = 0 # Use the youngest

        # 'tau is the width of each age bin.'
        wave0 = None
        for pp in range(len(tau0)):
            spall = [] # For sps model
            ms = np.zeros(Na, dtype='float')
            Ls = np.zeros(Na, dtype='float')
            mlost = np.zeros(Na, dtype='float')
            LICK = np.zeros((Na,len(INDICES)), dtype='float')
            # tau0_old = 0

            for ss in range(Na):

                iiz = np.where( (logZs_temp == Z[zz]) & ( np.abs(logTs_temp - np.log10(age[ss]))<0.001)) 

                if len(iiz[0]) == 0:
                    print('Template not found??')
                    hoge
                    # continue
                elif len(iiz[0])>1:
                    print('Duplicated Z/T in the template list??')
                    hoge

                file_sed = '%s'%(fd_temp['file'][iiz[0][0]])
                # file_stm = '%sstarmass-%s-imf%s.z%s.dat'%(DIR_LIB,bin_str,imf_str,z_str)
                fd_sed = ascii.read(file_sed)
                # fd_stm = ascii.read(file_stm)

                if wave0 is None:
                    wave0 = fd_sed['wavelength']
                    wave_tmp = fd_sed['wavelength']
                else:
                    wave_tmp = fd_sed['wavelength']

                # age_stm = fd_temp['logT'][iiz[0][0]]#np.log10(age[ss]) #fd_stm['col1']
                mass_formed = fd_temp['Ms'][iiz[0][0]]#fd_stm['col2'][0]

                ncols = len(MB.age)
                nage_temp = np.arange(2,ncols+1,1)
                lage_temp = np.log10(MB.age) #(6+0.1*(nage_temp-2))
                age_temp = 10**(6+0.1*(nage_temp-2)) # in yr

                mass_formed_tot = 0
                # flux0 = np.zeros(len(wave_tmp),'float')

                #
                # Determining tau for each age bin;
                #
                if int(tau0[pp]) == 99:
                    # @@@ TBD
                    if ss==0:
                        tautmp = age[ss]
                        agetmp = age[ss]/2.
                        con_tau= np.where((age_temp[:]<=age[ss]*1e9) & (age_temp[:]>0))
                        for sstmp in con_tau[0][-1:]:
                            flux0  += fd_sed['col%d'%(sstmp+2)][:]
                            ms[ss] += fd_stm['col2'][sstmp]
                            mass_formed_tot += mass_formed
                    else:
                        tautmp = age[ss] - age[ss-1]
                        agetmp = age[ss] - (age[ss]-age[ss-1])/2.
                        con_tau= np.where((age_temp[:]<=age[ss]*1e9) & (age_temp[:]>age[ss-1]*1e9))
                        for sstmp in con_tau[0][-1:]:
                            flux0  += fd_sed['col%d'%(sstmp+2)][:]
                            ms[ss] += fd_stm['col2'][sstmp]
                            mass_formed_tot += mass_formed

                elif tau0[pp] > 0.0:
                    # @@@ TBD

                    if ss==0 and age[ss]<tau0[pp]:
                        tautmp = age[ss]
                        agetmp = age[ss]/2.
                        con_tau= np.where((age_temp[:]<=age[ss]*1e9) & (age_temp[:]>0))
                        for sstmp in con_tau[0]:
                            flux0  += fd_sed['col%d'%(sstmp+2)][:]
                            ms[ss] += fd_stm['col2'][sstmp]
                            mass_formed_tot += mass_formed

                    elif (age[ss]-age[ss-1]) < tau0[pp]:
                        tautmp = age[ss] - age[ss-1]
                        agetmp = age[ss] - (age[ss]-age[ss-1])/2.
                        con_tau= np.where((age_temp[:]<=age[ss]*1e9) & (age_temp[:]>age[ss-1]*1e9))
                        for sstmp in con_tau[0]:
                            flux0  += fd_sed['col%d'%(sstmp+2)][:]
                            ms[ss] += fd_stm['col2'][sstmp]
                            mass_formed_tot += mass_formed

                    else:
                        tautmp = tau0[pp]
                        agetmp = age[ss] - tau0[pp]/2.
                        con_tau= np.where((age_temp[:]<=age[ss]*1e9) & (age_temp[:]>(age[ss] - tau0[pp])*1e9))
                        for sstmp in con_tau[0]:
                            flux0  += fd_sed['col%d'%(sstmp+2)][:]
                            ms[ss] += fd_stm['col2'][sstmp]
                            mass_formed_tot += mass_formed

                else: # =Negative tau; SSP
                    iis = np.argmin(np.abs(age[ss] - age_temp[:]/1e9))
                    # iistm = np.argmin(np.abs(age[ss] - 10**age_stm[:]/1e9))
                    if ss==0:
                        tautmp = 10**6.05 / 1e9 # in Gyr
                        agetmp = age[ss]/2.
                    else:
                        tautmp = ( 10**(lage_temp[iis]+0.05) - 10**(lage_temp[iis]-0.05) ) / 1e9 # Gyr
                        agetmp = (age[ss]+age[ss-1])/2.

                    # Interp;
                    flux_tmp = fd_sed['flux']
                    fint = interpolate.interp1d(wave_tmp, flux_tmp, kind='linear', fill_value='extrapolate')                    
                    flux0 = fint(wave0) # unit?
                    flux0 /= 3.826e33 # Now in Lsun, same as in BPASS.

                    ms[ss] = mass_formed
                    mass_formed_tot += mass_formed

                # Keep tau in header;
                tau_age[ss] = tautmp
                age_age[ss] = agetmp

                # Then. add flux if tau > 0.
                con = (wave0>lammin) & (wave0<lammax)
                wave, flux = wave0[con], flux0[con]

                # Temp
                mlost[ss] = ms[ss] / mass_formed_tot
                Ls[ss] = np.sum(flux0) # BPASS sed is in Lsun.
                LICK[ss,:] = get_ind(wave, flux)

                if flagz and ss == 0 and pp == 0:
                    # ASDF Big tree;
                    # Create header;
                    tree = {
                        'isochrone': '',
                        'library': '',
                        'nimf': nimf,
                        'version_gsf': gsf.__version__
                    }

                    tree.update({'age': MB.age})
                    tree.update({'Z': MB.Zall})
                    if fneb:
                        tree.update({'logUMIN': MB.logUMIN})
                        tree.update({'logUMAX': MB.logUMAX})
                        tree.update({'DELlogU': MB.DELlogU})
                    if MB.fagn:
                        tree.update({'AGNTAUMIN': MB.AGNTAUMIN})
                        tree.update({'AGNTAUMAX': MB.AGNTAUMAX})
                        tree.update({'DELAGNTAU': MB.DELAGNTAU})

                    # ASDF
                    tree_spec.update({'wavelength': wave})
                    flagz = True

                if not couple_neb:
                    # ASDF
                    tree_spec.update({'fspec_'+str(zz)+'_'+str(ss)+'_'+str(pp): flux})

                    # BPASS neb;
                    if fneb and pp == 0 and ss == iix_age_neb:
                        # @@@ TBD
                        if zz == 0:
                            MB.logger.info('BPASS nebular component is calculated using age=%.1e'%(age[ss]))

                        for nlogU, logUtmp in enumerate(MB.logUs):
                            # Each file has 16 columns and 30,000 rows. The first column lists a wavelength in angstroms, 
                            # and each remaining column n (n>1) holds the model flux for the population at an age of 10^(6+0.1*(n-2)) years at that wavelength. The range of ages covered is log(age/years)=6.0-7.5
                            # for nlogU, logUtmp in enumerate(MB.logUs):
                            # The units of flux are log_10(ergs/s per Angstrom), normalised for a cluster of 1e6 Msun formed in a single instantaneous burst. The total luminosity of the SED can be simply calculated by summing all the rows together
                            iix = np.argmin(np.abs(logUtmp - logUs_bpass))
                            logu_str = logUs_bpass_str[iix]
                            file_sed_emi = '%scloudyspec_imf%s_z%s_%s_%s.sed'%(DIR_LIB_NEB,imf_str,z_str,bin_str,logu_str)
                            fd_sed_emi = ascii.read(file_sed_emi)
                            con_emi_data = (fd_sed_emi['col1'] != 'Total_Power')
                            fd_sed_emi = fd_sed_emi[con_emi_data]
                            wave0_emi = np.asarray([float(s) for s in fd_sed_emi['col1']])
                            flux0_emi = np.zeros(len(wave0_emi),'float')
                            # Repeat the same but for emission;

                            #
                            # Determining tau for each age bin;
                            #
                            # Only ssp available;
                            iis = np.argmin(np.abs(age[ss] - age_emi[:]/1e9))
                            if iis+2 < ncols_emi:
                                flux0_emi = 10**fd_sed_emi['col%d'%(iis+2)]
                            else:
                                flux0_emi = flux0_emi[:] * 0

                            # if zz == 0 and nlogU ==0:
                            #     for _i in range(5):
                            #         plt.plot(wave0_emi, 10**fd_sed_emi['col%d'%(_i+2)], ls=':', alpha=0.5, label='%d'%(_i))

                            # Then. add flux if tau > 0.
                            # con = (wave0>lammin) & (wave0<lammax)
                            # wave, flux = wave0[con], flux0[con]
                            # con_emi = (wave0_emi>lammin) & (wave0_emi<lammax)
                            # ewave, eflux = wave0_emi[con_emi], flux0_emi[con_emi]
                            femi = interpolate.interp1d(wave0_emi, flux0_emi, kind='linear', fill_value="extrapolate")
                            flux_nebular = femi(wave)
                            emline_luminosity = np.sum(flux0_emi)

                            flux_nebular_only = flux_nebular/Lunit_emi-flux
                            con_neg = flux_nebular_only<0
                            flux_nebular_only[con_neg] = 0

                            # ASDF
                            # tree_spec.update({'efspec_'+str(zz)+'_'+str(ss)+'_'+str(pp): eflux})
                            tree_spec.update({'flux_nebular_Z%d'%zz+'_logU%d'%nlogU: flux_nebular_only}) # in Lsun/AA
                            tree_spec.update({'emline_wavelengths_Z%d'%zz+'_logU%d'%nlogU: wave})
                            tree_spec.update({'emline_luminosity_Z%d'%zz+'_logU%d'%nlogU: emline_luminosity}) # in Lsun
                            tree_spec.update({'emline_mass_Z%d'%zz+'_logU%d'%nlogU: mstel_emi}) # in Msun
                            # print('fspec_nebular_Z%d'%zz+'_logU%d'%nlogU, flux_nebular)

                else:
                    MB.logUFIX = logu_neb
                    MB.nlogU = 1
                    MB.logUMIN = MB.logUFIX
                    MB.logUMAX = MB.logUFIX
                    MB.DELlogU = 0
                    MB.logUs = np.asarray([MB.logUMAX])

                    iix = np.argmin(np.abs(logu_neb - logUs_bpass))
                    logu_str = logUs_bpass_str[iix]
                    file_sed_emi = '%scloudyspec_imf%s_z%s_%s_%s.sed'%(DIR_LIB_NEB,imf_str,z_str,bin_str,logu_str)
                    fd_sed_emi = ascii.read(file_sed_emi)
                    con_emi_data = (fd_sed_emi['col1'] != 'Total_Power')
                    fd_sed_emi = fd_sed_emi[con_emi_data]
                    wave0_emi = np.asarray([float(s) for s in fd_sed_emi['col1']])
                    flux0_emi = np.zeros(len(wave0_emi),'float')

                    if pp == 0 and ss == 0 and zz == 0:
                        MB.logger.info('BPASS nebular component is calculated using logU=%.1f'%(logu_neb))

                    #
                    # Determining tau for each age bin;
                    #
                    # Only ssp available;
                    iis = np.argmin(np.abs(age[ss] - age_emi[:]/1e9))
                    if iis+2 < ncols_emi:
                        flux0_emi = 10**fd_sed_emi['col%d'%(iis+2)]
                    else:
                        flux0_emi = flux0_emi[:] * 0
                    emline_luminosity = np.nansum(flux0_emi)

                    femi = interpolate.interp1d(wave0_emi, flux0_emi, kind='linear', fill_value="extrapolate")
                    flux_nebular = femi(wave)
                    con_neg = flux_nebular<0
                    flux_nebular[con_neg] = 0

                    flux_nebular_only = flux_nebular/Lunit_emi-flux
                    con_neg = flux_nebular_only<0
                    flux_nebular_only[con_neg] = 0

                    # ASDF
                    L_neb_tmp = np.nansum(flux_nebular/Lunit_emi)
                    tree_spec.update({'fspec_'+str(zz)+'_'+str(ss)+'_'+str(pp): flux_nebular/Lunit_emi * Ls[ss]/L_neb_tmp})
                    # if zz == 0:
                    #     plt.close()
                    #     plt.plot(wave, flux)
                    #     plt.plot(wave, flux_nebular/Lunit_emi * Ls[ss]/L_neb_tmp)
                    #     plt.show()
                    tree_spec.update({'flux_nebular_Z%d'%zz+'_logU%d'%0: flux_nebular_only*0}) # in Lsun/AA
                    tree_spec.update({'emline_wavelengths_Z%d'%zz+'_logU%d'%0: wave})
                    tree_spec.update({'emline_luminosity_Z%d'%zz+'_logU%d'%0: emline_luminosity}) # in Lsun
                    tree_spec.update({'emline_mass_Z%d'%zz+'_logU%d'%0: mstel_emi}) # in Msun

            # plt.legend(loc=0)
            # plt.show()
            # hoge

            # Register M/Ls;
            if pp == 0:
                # use tau0[0] as representative for M/L and index.
                for ll in range(len(INDICES)):
                    # ASDF
                    tree_lick.update({INDICES[ll]+'_'+str(zz): LICK[:,ll]})

                col1 = fits.Column(name='ms_'+str(zz), format='E', unit='Msun', array=ms)
                tree_ML.update({'ms_'+str(zz): ms})
                col2 = fits.Column(name='Ls_'+str(zz), format='E', unit='Lsun', array=Ls)
                tree_ML.update({'Ls_'+str(zz): Ls})
                col3 = fits.Column(name='fm_'+str(zz), format='E', unit='', array=mlost)
                tree_ML.update({'frac_mass_survive_'+str(zz): mlost})
                col4 = fits.Column(name='tau_'+str(zz), format='E', unit='Gyr', array=tau_age)
                tree_ML.update({'realtau_'+str(zz): ms})

    # Write;
    for aa in range(len(age)):
        tree.update({'realtau%d(Gyr)'%(aa): tau_age[aa]})
    for aa in range(len(age)):
        tree.update({'realage%d(Gyr)'%(aa): age_age[aa]})
    for aa in range(len(age)):
        tree.update({'age%d'%(aa): age[aa]})
    for aa in range(len(Z)):
        tree.update({'Z%d'%(aa): Z[aa]})
    for aa in range(len(tau0)):
        tree.update({'tau0%d'%(aa): tau0[aa]})

    # Index, Mass-to-light;
    tree.update({'spec' : tree_spec})
    tree.update({'ML' : tree_ML})
    tree.update({'lick' : tree_lick})

    # Save
    assert validate_and_save_tree(tree, file_out, dir_tmp=DIR_TMP) == True


def smooth_spectrum(wave, flux, wmin=0, wmax=1750, sigma=30, verbose=False):
    '''
    wave : float array
        in AA
    sigma : float
        in AA
    '''
    con = (wave>wmin) & (wave<wmax)
    
    delwave = np.nanmedian(np.diff(wave[con]))
    stddev = sigma / delwave
    if verbose:
        print('stddev is set to',stddev)
        
    if stddev < 0.1:
        print('stddev is too small (%.2f). No processing.'%stddev)
        return wave, flux
        
    # Create kernel
    g = Gaussian1DKernel(stddev=stddev)

    # Convolve data
    z = convolve(flux[con], g, boundary='extend')
    
    flux[con] = z

    return wave, flux
