import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os
import sys
from astropy.io import ascii

from .function import get_ind

INDICES = ['G4300', 'Mgb', 'Fe5270', 'Fe5335', 'NaD', 'Hb', 'Fe4668', 'Fe5015', 'Fe5709', 'Fe5782', 'Mg1', 'Mg2', 'TiO1', 'TiO2']


def get_lognorm(t, ltau0, T0=-10):
    A   = 1
    tau0= 10**ltau0
    SFR = t * 0 + 1e-20
    conlogn = (t>0)
    SFR[conlogn] = A / np.sqrt(2*np.pi*tau0**2) * np.exp(-(np.log(t[conlogn])-T0)**2/(2*tau0**2)) / t[conlogn]
    return SFR


def make_tmp_z0(MB, lammin=100, lammax=160000, Zforce=None): 
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
    age = MB.ageparam
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
    col01 = [] # For M/L ratio.
    col02 = [] # For templates
    col05 = [] # For spectral indices.

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

                if zz == 0 and ss == 0 and tt == 0 and 10**age[tt]<0.01:
                    MB.logger.warning('Your input AGE includes <0.01Gyr --- fsps interpolates spectra, and you may not get accurate SEDs.')

                if sfh==6: # Tabular SFH.
                    tuniv_hr  = np.arange(0,age_univ,0.01) # in Gyr
                    T0 = np.log(10**age[tt])
                    sfh_hr_in = get_lognorm(tuniv_hr, tau[ss], T0) # tau in log Gyr
                    zh_hr_in  = tuniv_hr*0 + 10**Z[zz] # metallicity is constant
                    sp.set_tabular_sfh(tuniv_hr, sfh_hr_in, zh_hr_in)
                    wave0, flux0 = sp.get_spectrum(tage=age_univ, peraa=True) # if peraa=True, in unit of L/AA
                else:
                    wave0, flux0 = sp.get_spectrum(tage=10**age[tt], peraa=True) # if peraa=True, in unit of L/AA

                con = (wave0>lammin) & (wave0<lammax)
                wave, flux = wave0[con], flux0[con]
                ms[tt] = sp.stellar_mass
                if np.isnan(ms[tt]):
                    print('Error at tau element at',tt)
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
                        esp = esptmp

                        if age[ss]>0.01:
                            tage_neb = 0.01
                            MB.logger.info('Nebular component is calculabed with %.2f Gyr'%tage_neb)
                        else:
                            tage_neb = age[tt]

                        ewave0, eflux0 = esp.get_spectrum(tage=age[tt], peraa=True)
                        con = (ewave0>lammin) & (ewave0<lammax)
                        flux_nebular = eflux0[con]-flux

                        # Eliminate some negatives. Mostly on <912A;
                        con_neg = flux_nebular<0
                        flux_nebular[con_neg] = 0

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

    # Save
    af = asdf.AsdfFile(tree)
    af.write_to(DIR_TMP + file_out, all_array_compression='zlib')
