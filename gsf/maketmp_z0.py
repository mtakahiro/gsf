import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os
from astropy.io import ascii

from .function import get_ind

INDICES = ['G4300', 'Mgb', 'Fe5270', 'Fe5335', 'NaD', 'Hb', 'Fe4668', 'Fe5015', 'Fe5709', 'Fe5782', 'Mg1', 'Mg2', 'TiO1', 'TiO2']

def make_tmp_z0(MB, lammin=100, lammax=160000, tau_lim=0.001):
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
    age  = MB.age
    tau0 = MB.tau0
    fneb = MB.fneb
    logU = MB.logU
    DIR_TMP = MB.DIR_TMP
    Na = len(age)

    # Z needs special care in z0 script, to avoid Zfix.
    if False:
        # If this is implemented, make sure maketemp at z is also consistent.
        Zmax_tmp, Zmin_tmp = float(MB.inputs['ZMAX']), float(MB.inputs['ZMIN'])
        delZ_tmp = float(MB.inputs['DELZ'])
        if Zmax_tmp == Zmin_tmp or delZ_tmp==0:
            delZ_tmp = 0.0001
        Z = np.arange(Zmin_tmp, Zmax_tmp+delZ_tmp, delZ_tmp) # in logZsun
    else:
        Z = MB.Zall
    NZ = len(Z)
    
    # Current age in Gyr;
    age_univ = MB.cosmo.age(0).value

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
    tau_age = np.zeros(Na,dtype='float')
    age_age = np.zeros(Na,dtype='float')
    for zz in range(len(Z)):
        for pp in range(len(tau0)):
            spall = [] # For sps model
            ms = np.zeros(Na, dtype='float')
            Ls = np.zeros(Na, dtype='float')
            mlost = np.zeros(Na, dtype='float')
            LICK = np.zeros((Na,len(INDICES)), dtype='float')
            tau0_old = 0
            for ss in range(Na):
                #
                # Determining tau for each age bin;
                #

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
                        print('CSP is applied.')
                        print('At t=%.3f, tau is %.3f Gyr' %(age[ss],tautmp))
                        sptmp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1, imf_type=nimf, sfh=1, logzsol=Z[zz], dust_type=2, dust2=0.0, tau=20, const=0, sf_start=0, sf_trunc=tautmp, tburst=13, fburst=0) # Lsun/Hz
                        if fneb:
                            esptmp = fsps.StellarPopulation(zcontinuous=1, imf_type=nimf, sfh=1, logzsol=Z[zz], dust_type=2, dust2=0.0, tau=20, const=0, sf_start=0, sf_trunc=tautmp, tburst=13, fburst=0, add_neb_emission=1)
                        if f_add_dust:
                            dsptmp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1, imf_type=nimf, sfh=1, logzsol=Z[zz],
                            dust_type=2, dust2=0.0, tau=20, const=0, sf_start=0, sf_trunc=tautmp, tburst=13, fburst=0,
                            duste_gamma=0.01, duste_umin=1.0, duste_qpah=3.5, fagn=0.0)
                    elif tau0[pp] > 0.0:
                        print('At t=%.3f, fixed tau, %.3f, is applied.'%(age[ss],tautmp))
                        sptmp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1, imf_type=nimf, sfh=1, logzsol=Z[zz], dust_type=2, dust2=0.0, tau=20, const=0, sf_start=0, sf_trunc=tautmp, tburst=13, fburst=0) # Lsun/Hz
                        if fneb:
                            esptmp = fsps.StellarPopulation(zcontinuous=1, imf_type=nimf, sfh=1, logzsol=Z[zz], dust_type=2, dust2=0.0, tau=20, const=0, sf_start=0, sf_trunc=tautmp, tburst=13, fburst=0, add_neb_emission=1)
                        if f_add_dust:
                            dsptmp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1, imf_type=nimf, sfh=1, logzsol=Z[zz], 
                            dust_type=2, dust2=0.0, tau=20, const=0, sf_start=0, sf_trunc=tautmp, tburst=13, fburst=0,
                            duste_gamma=0.01, duste_umin=1.0, duste_qpah=3.5, fagn=0.0)
                    else: # =Negative tau;
                        print('At t=%.3f, SSP (%.3f) is applied.'%(age[ss],tautmp))
                        sptmp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1, imf_type=nimf, sfh=0, logzsol=Z[zz], dust_type=2, dust2=0.0) # Lsun/Hz
                        if fneb:
                            esptmp = fsps.StellarPopulation(zcontinuous=1, imf_type=nimf, sfh=0, logzsol=Z[zz], dust_type=2, dust2=0.0, add_neb_emission=1)
                        if f_add_dust:
                            dsptmp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1, imf_type=nimf, sfh=0, logzsol=Z[zz], dust_type=2, dust2=1.0,
                            tau=20, const=0, sf_start=0, sf_trunc=tautmp, tburst=13, fburst=0,
                            duste_gamma=0.01, duste_umin=1.0, duste_qpah=3.5, fagn=0.0)
                else:
                    print('At t=%.3f, tau is %.3f Gyr' %(age[ss],tautmp))
                    print('Skip fsps, by using previous library.')

                tau0_old = tautmp
                sp = sptmp
                print('Z:%d/%d, t:%d/%d, %s, %s'%(zz+1, len(Z), pp+1, len(tau0), sp.libraries[0].decode("utf-8") , sp.libraries[1].decode("utf-8")))

                wave0, flux0 = sp.get_spectrum(tage=age[ss], peraa=True)
                con = (wave0>lammin) & (wave0<lammax)
                wave, flux = wave0[con], flux0[con]
                mlost[ss] = sp.stellar_mass / sp.formed_mass

                if fneb:
                    esptmp.params['gas_logz'] = Z[zz] # gas metallicity, assuming = Zstel
                    esptmp.params['gas_logu'] = logU # ionization parameter
                    esp = esptmp
                    if ss == 0:
                        print('Nebular lines are also added, with logU=%.2f.'%(logU))
                    ewave0, eflux0 = esp.get_spectrum(tage=age[ss], peraa=True)
                    con = (ewave0>lammin) & (ewave0<lammax)
                    eflux = eflux0[con]
                
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

                if ss == 0 and pp == 0 and zz == 0:
                    # ASDF Big tree;
                    # Create header;
                    tree = {
                        'isochrone': '%s'%(sp.libraries[0].decode("utf-8")),
                        'library': '%s'%(sp.libraries[1].decode("utf-8")),
                        'nimf': nimf,
                        'version_gsf': gsf.__version__
                    }
                    if fneb:
                        tree.update({'logU': logU})

                    col3 = fits.Column(name='wavelength', format='E', unit='AA', array=wave)
                    col02.append(col3)
                    # ASDF
                    tree_spec.update({'wavelength': wave})

                col4 = fits.Column(name='fspec_'+str(zz)+'_'+str(ss)+'_'+str(pp), format='E', unit='Fnu', array=flux)
                col02.append(col4)
                # ASDF
                tree_spec.update({'fspec_'+str(zz)+'_'+str(ss)+'_'+str(pp): flux})

                if fneb:
                    col4e = fits.Column(name='efspec_'+str(zz)+'_'+str(ss)+'_'+str(pp), format='E', unit='Fnu', array=eflux)
                    col02.append(col4e)
                    # ASDF
                    tree_spec.update({'efspec_'+str(zz)+'_'+str(ss)+'_'+str(pp): eflux})


            if pp == 0:
                # use tau0[0] as representative for M/L and index.
                for ll in range(len(INDICES)):
                    col5 = fits.Column(name=INDICES[ll]+'_'+str(zz), format='E', unit='', array=LICK[:,ll])
                    col05.append(col5)
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

                col01.append(col1)
                col01.append(col2)
                col01.append(col3)
                col01.append(col4)

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
    af = asdf.AsdfFile(tree)
    af.write_to(DIR_TMP + 'spec_all.asdf', all_array_compression='zlib')


def make_tmp_z0_bpass(MB, lammin=100, lammax=160000, \
    Zsun=0.02):
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

    BPASS_DIR : str, optional
        Path to the ditectory where BPASS templates are storesd.

    BPASS_ver : str, optional
        Version of BPASS. Used to identify template files.

    Zsun : float, optional
        Metallicity of templates, in units of absolute value (e.g. Zsun=0.02 for BPASS).
    '''
    import asdf
    import gsf

    nimf = MB.nimf
    if nimf == 0: # Salpeter
        imf_str = '135all_100'
    elif nimf == 1:
        imf_str = '_chab100'
    else:
        imf_str = ''

    Z = MB.Zall
    age = MB.age
    tau0 = MB.tau0
    fneb = MB.fneb
    if fneb:
        print('Currently, BPASS does not have option of nebular emission.')
        fneb = False
    logU = MB.logU
    DIR_TMP= MB.DIR_TMP

    # binary?
    f_bin = MB.f_bin
    #f_bin = True
    if f_bin==1:
        bin_str = 'bin'
    else:
        bin_str = 'str'

    DIR_LIB = MB.BPASS_DIR + 'BPASS%s/BPASS%s_%s-imf%s/'%(MB.BPASS_ver,MB.BPASS_ver,bin_str,imf_str)

    NZ = len(Z)
    Na = len(age)

    # Current age in Gyr;
    age_univ = MB.cosmo.age(0).value

    #import matplotlib.pyplot as plt

    print('#######################################')
    print('Making templates at z=0, IMF=%d'%(nimf))
    print('#######################################')
    col01 = [] # For M/L ratio.
    col02 = [] # For templates
    col05 = [] # For spectral indices.
    #col06 = [] # For weird templates for UVJ calculation;

    tree_spec = {}
    tree_ML = {}
    tree_lick = {}

    tau_age = np.zeros(Na,dtype='float')
    age_age = np.zeros(Na,dtype='float')
    for zz in range(len(Z)):
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

        wave0   = fd_sed['col1']
        age_stm = fd_stm['col1']
        mass_formed = fd_stm['col2'][0]

        ncols = 52
        nage_temp = np.arange(2,ncols+1,1)
        lage_temp = (6+0.1*(nage_temp-2))
        age_temp  = 10**(6+0.1*(nage_temp-2)) # in yr

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

                    flux0  = fd_sed['col%d'%(iis+2)] #sp.get_spectrum(tage=age[ss], peraa=True)
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

                if ss == 0 and pp == 0 and zz == 0:
                    # ASDF Big tree;
                    # Create header;
                    tree = {
                        'isochrone': 'BPASS',
                        'library': 'BPASS',
                        'nimf': nimf,
                        'version_gsf': gsf.__version__
                    }
                    if fneb:
                        tree.update({'logU': logU})

                    col3 = fits.Column(name='wavelength', format='E', unit='AA', array=wave)
                    col02.append(col3)

                    # ASDF
                    tree_spec.update({'wavelength': wave})

                col4  = fits.Column(name='fspec_'+str(zz)+'_'+str(ss)+'_'+str(pp), format='E', unit='Fnu', array=flux)
                col02.append(col4)

                # ASDF
                tree_spec.update({'fspec_'+str(zz)+'_'+str(ss)+'_'+str(pp): flux})

                if fneb:
                    col4e = fits.Column(name='efspec_'+str(zz)+'_'+str(ss)+'_'+str(pp), format='E', unit='Fnu', array=eflux)
                    col02.append(col4e)
                    # ASDF
                    tree_spec.update({'efspec_'+str(zz)+'_'+str(ss)+'_'+str(pp): eflux})


            if pp == 0:
                # use tau0[0] as representative for M/L and index.
                for ll in range(len(INDICES)):
                    col5 = fits.Column(name=INDICES[ll]+'_'+str(zz), format='E', unit='', array=LICK[:,ll])
                    col05.append(col5)
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

                col01.append(col1)
                col01.append(col2)
                col01.append(col3)

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
    af = asdf.AsdfFile(tree)
    af.write_to(DIR_TMP + 'spec_all.asdf', all_array_compression='zlib')

