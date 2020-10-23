import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os
from astropy.io import ascii

INDICES = ['G4300', 'Mgb', 'Fe5270', 'Fe5335', 'NaD', 'Hb', 'Fe4668', 'Fe5015', 'Fe5709', 'Fe5782', 'Mg1', 'Mg2', 'TiO1', 'TiO2']

def get_ind(wave,flux):
    '''
    Purpose:
    ========
    Get Lick index for input input
    '''
    lml     = [4268, 5143, 5233, 5305, 5862, 4828, 4628, 4985, 5669, 5742, 4895, 4895, 5818, 6068]
    lmcl    = [4283, 5161, 5246, 5312, 5879, 4848, 4648, 5005, 5689, 5762, 5069, 5154, 5938, 6191]
    lmcr    = [4318, 5193, 5286, 5352, 5911, 4877, 4668, 5925, 5709, 5782, 5134, 5197, 5996, 6274]
    lmr     = [4336, 5206, 5318, 5363, 5950, 4892, 4688, 5945, 5729, 5802, 5366, 5366, 6105, 6417]

    W = np.zeros(len(lml), dtype='float32')
    for ii in range(len(lml)):
        con_cen = (wave>lmcl[ii]) & (wave<lmcr[ii])
        con_sid = ((wave<lmcl[ii]) & (wave>lml[ii])) | ((wave<lmr[ii]) & (wave>lmcr[ii]))

        Ic = np.mean(flux[con_cen])
        Is = np.mean(flux[con_sid])

        delam = lmcr[ii] - lmcl[ii]

        if ii < 10:
            W[ii] = (1. - Ic/Is) * delam
        elif 1. - Ic/Is > 0:
            W[ii] = -2.5 * np.log10(1. - Ic/Is)
        else:
            W[ii] = -99

    # Return equivalent width
    return W


def make_tmp_z0(MB, lammin=100, lammax=160000):
    '''
    Purpose:
    ========
    #
    # This is for the preparation of
    # default template, with FSPS, at z=0.
    #
    # Should be run before SED fitting.
    #
    #

    Input:
    ======
    #
    # nimf (int) : 0:Salpeter, 1:Chabrier, 2:Kroupa, 3:vanDokkum08,...
    # Z (array)  : Stellar phase metallicity in logZsun.
    # age (array): Age, in Gyr.
    # fneb (int) : flag for adding nebular emissionself.
    # logU (float): ionizing parameter, in logU.
    # tau0 (float array): Width of age bin. If you want to fix, put >0.01 (Gyr).
    #  Otherwise, it would be either minimum value (=0.01; if one age bin), or
    #  the width to the next age bin.
    #
    '''
    import asdf
    import fsps
    import gsf

    nimf = MB.nimf
    Z  = MB.Zall #np.arange(-1.2,0.4249,0.05)
    age  = MB.age #[0.01, 0.1, 0.3, 0.7, 1.0, 3.0]
    tau0 = MB.tau0 #[0.01,0.02,0.03],
    fneb = MB.fneb #0,
    logU = MB.logU #-2.5,
    DIR_TMP= MB.DIR_TMP #'./templates/'

    NZ = len(Z)
    Na = len(age)

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
    tau_age = np.zeros(Na,dtype='float64')
    age_age = np.zeros(Na,dtype='float64')
    for zz in range(len(Z)):
        for pp in range(len(tau0)):
            spall = [] # For sps model
            ms = np.zeros(Na, dtype='float32')
            Ls = np.zeros(Na, dtype='float32')
            LICK = np.zeros((Na,len(INDICES)), dtype='float32')
            tau0_old = 0
            for ss in range(Na):
                #
                # Determining tau for each age bin;
                #
                # 1.Continuous age bin;
                if int(tau0[pp]) == 99:
                    if ss==0:
                        tautmp = age[ss]
                        agetmp = age[ss]/2.
                    else:
                        tautmp = age[ss] - age[ss-1]
                        agetmp = (age[ss]+age[ss-1])/2.
                # 2.A fixed-age bin;
                elif tau0[pp] > 0.0:
                    tautmp = tau0[pp]
                    agetmp = age[ss]
                # 3.SSP;
                else: # =Negative tau;
                    tautmp = 0.001
                    agetmp = age[ss]

                # Keep tau in header;
                tau_age[ss] = tautmp
                age_age[ss] = agetmp

                #
                # Then, make sps.
                #
                if tautmp != tau0_old:
                    if int(tau0[pp]) == 99:
                        print('CSP is applied.')
                        print('At t=%.3f, tau is %.3f Gyr' %(age[ss],tautmp))
                        sptmp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1, imf_type=nimf, sfh=1, logzsol=Z[zz], dust_type=2, dust2=0.0, tau=20, const=0, sf_start=0, sf_trunc=tautmp, tburst=13, fburst=0) # Lsun/Hz
                        if fneb == 1:
                            esptmp = fsps.StellarPopulation(zcontinuous=1, imf_type=nimf, sfh=1, logzsol=Z[zz], dust_type=2, dust2=0.0, tau=20, const=0, sf_start=0, sf_trunc=tautmp, tburst=13, fburst=0, add_neb_emission=1)
                    elif tau0[pp] > 0.0:
                        print('At t=%.3f, fixed tau, %.3f, is applied.'%(age[ss],tautmp))
                        sptmp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1, imf_type=nimf, sfh=1, logzsol=Z[zz], dust_type=2, dust2=0.0, tau=20, const=0, sf_start=0, sf_trunc=tautmp, tburst=13, fburst=0) # Lsun/Hz
                        if fneb == 1:
                            esptmp = fsps.StellarPopulation(zcontinuous=1, imf_type=nimf, sfh=1, logzsol=Z[zz], dust_type=2, dust2=0.0, tau=20, const=0, sf_start=0, sf_trunc=tautmp, tburst=13, fburst=0, add_neb_emission=1)
                    else: # =Negative tau;
                        print('At t=%.3f, SSP (%.3f) is applied.'%(age[ss],tautmp))
                        sptmp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1, imf_type=nimf, sfh=0, logzsol=Z[zz], dust_type=2, dust2=0.0) # Lsun/Hz
                        if fneb == 1:
                            esptmp = fsps.StellarPopulation(zcontinuous=1, imf_type=nimf, sfh=0, logzsol=Z[zz], dust_type=2, dust2=0.0, add_neb_emission=1)
                else:
                    print('At t=%.3f, tau is %.3f Gyr' %(age[ss],tautmp))
                    print('Skip fsps, by using previous library.')

                tau0_old = tautmp
                sp = sptmp
                print(zz, sp.libraries[0].decode("utf-8") , sp.libraries[1].decode("utf-8") , pp)

                wave0, flux0 = sp.get_spectrum(tage=age[ss], peraa=True)
                con = (wave0>lammin) & (wave0<lammax)
                wave, flux = wave0[con], flux0[con]

                if fneb == 1:
                    esptmp.params['gas_logz'] = Z[zz] # gas metallicity, assuming = Zstel
                    esptmp.params['gas_logu'] = logU # ionization parameter
                    esp = esptmp
                    print('Nebular lines are also added, with logU=%.2f.'%(logU))
                    ewave0, eflux0 = esp.get_spectrum(tage=age[ss], peraa=True)
                    con = (ewave0>lammin) & (ewave0<lammax)
                    eflux = eflux0[con]

                # Mass-Luminosity
                ms[ss]  = sp.stellar_mass
                Ls[ss]  = 10**sp.log_lbol
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
                    if fneb == 1:
                        tree.update({'logU': logU})

                    col3 = fits.Column(name='wavelength', format='E', unit='AA', array=wave)
                    col02.append(col3)

                    # ASDF
                    tree_spec.update({'wavelength': wave})

                col4  = fits.Column(name='fspec_'+str(zz)+'_'+str(ss)+'_'+str(pp), format='E', unit='Fnu', array=flux)
                col02.append(col4)

                # ASDF
                tree_spec.update({'fspec_'+str(zz)+'_'+str(ss)+'_'+str(pp): flux})

                if fneb == 1:
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
                # ASDF
                tree_ML.update({'ms_'+str(zz): ms})
                col2 = fits.Column(name='Ls_'+str(zz), format='E', unit='Lsun', array=Ls)
                # ASDF
                tree_ML.update({'Ls_'+str(zz): Ls})
                col01.append(col1)
                col01.append(col2)

    # Write;
    for aa in range(len(age)):
        #hdr['hierarch realtau%d(Gyr)'%(aa)] = tau_age[aa]
        tree.update({'realtau%d(Gyr)'%(aa): tau_age[aa]})
    for aa in range(len(age)):
        #hdr['hierarch realage%d(Gyr)'%(aa)] = age_age[aa]
        tree.update({'realage%d(Gyr)'%(aa): age_age[aa]})

    tree.update({'spec' : tree_spec})
    tree.update({'ML' : tree_ML})
    tree.update({'lick' : tree_lick})
    af = asdf.AsdfFile(tree)
    af.write_to(DIR_TMP + 'spec_all.asdf', all_array_compression='zlib')


def make_tmp_z0_bpass(MB, lammin=100, lammax=160000, BPASS_DIR='/astro/udfcen3/Takahiro/BPASS/', BPASS_ver='v2.2.1', Zsun=0.02):
    '''
    #
    # nimf (int) : 0:Salpeter, 1:Chabrier, 2:Kroupa, 3:vanDokkum08,...
    # Z (array)  : Stellar phase metallicity in logZsun.
    # age (array): Age, in Gyr.
    # fneb (int) : flag for adding nebular emissionself.
    # logU (float): ionizing parameter, in logU.
    # tau0 (float array): Width of age bin. If you want to fix, put >0.01 (Gyr).
    #  Otherwise, it would be either minimum value (=0.01; if one age bin), or
    #  the width to the next age bin.
    #
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

    Z    = MB.Zall #np.arange(-1.2,0.4249,0.05)
    age  = MB.age #[0.01, 0.1, 0.3, 0.7, 1.0, 3.0]/Gyr
    tau0 = MB.tau0 #[0.01,0.02,0.03],
    fneb = MB.fneb #0,
    print('Currently, BPASS does not have option of nebular emission.')
    fneb = 0
    logU = MB.logU #-2.5,
    DIR_TMP= MB.DIR_TMP #'./templates/'

    # binary?
    f_bin = MB.f_bin
    #f_bin = True
    if f_bin==1:
        bin_str = 'bin'
    else:
        bin_str = 'str'

    DIR_LIB = BPASS_DIR + 'BPASS%s/BPASS%s_%s-imf%s/'%(BPASS_ver,BPASS_ver,bin_str,imf_str)

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

    tau_age = np.zeros(Na,dtype='float64')
    age_age = np.zeros(Na,dtype='float64')
    for zz in range(len(Z)):
        #
        # open spectral file;
        #
        if 10**(Z[zz])*Zsun>1e-4:
            zstrtmp  = round(10**(Z[zz])*Zsun/10,5)
            zstrtmp2 = '%.6s'%(zstrtmp)
            z_str    = zstrtmp2[3:]
            if len(z_str)<3:
                z_str = z_str+'0'
        elif 10**(Z[zz])*Zsun>1e-5:
            z_str    = 'em4'
        else:
            z_str    = 'em5'

        file_sed = '%sspectra-%s-imf%s.z%s.dat'%(DIR_LIB,bin_str,imf_str,z_str)
        file_stm = '%sstarmass-%s-imf%s.z%s.dat'%(DIR_LIB,bin_str,imf_str,z_str)
        fd_sed = ascii.read(file_sed)
        fd_stm = ascii.read(file_stm)

        wave0   = fd_sed['col1']
        age_stm = fd_stm['col1']

        ncols = 52
        nage_temp = np.arange(2,ncols+1,1)
        lage_temp = (6+0.1*(nage_temp-2))
        age_temp  = 10**(6+0.1*(nage_temp-2)) # in yr

        # 'tau is the width of each age bin.'
        for pp in range(len(tau0)):
            spall = [] # For sps model
            ms = np.zeros(Na, dtype='float32')
            Ls = np.zeros(Na, dtype='float32')
            LICK = np.zeros((Na,len(INDICES)), dtype='float32')
            tau0_old = 0

            for ss in range(Na):
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
                            #print(age[ss],age_temp[sstmp]/1e9,sstmp+2)
                    else:
                        tautmp = age[ss] - age[ss-1]
                        agetmp = age[ss] - (age[ss]-age[ss-1])/2.
                        con_tau= np.where((age_temp[:]<=age[ss]*1e9) & (age_temp[:]>age[ss-1]*1e9))
                        for sstmp in con_tau[0][-1:]:
                            flux0  += fd_sed['col%d'%(sstmp+2)][:]
                            ms[ss] += fd_stm['col2'][sstmp]
                            #print(age[ss],age_temp[sstmp]/1e9,sstmp+2)

                elif tau0[pp] > 0.0:
                    if ss==0 and age[ss]<tau0[pp]:
                        tautmp = age[ss]
                        agetmp = age[ss]/2.
                        con_tau= np.where((age_temp[:]<=age[ss]*1e9) & (age_temp[:]>0))
                        for sstmp in con_tau[0]:
                            flux0  += fd_sed['col%d'%(sstmp+2)][:]
                            ms[ss] += fd_stm['col2'][sstmp]

                    elif (age[ss]-age[ss-1]) < tau0[pp]:
                        tautmp = age[ss] - age[ss-1]
                        agetmp = age[ss] - (age[ss]-age[ss-1])/2.
                        con_tau= np.where((age_temp[:]<=age[ss]*1e9) & (age_temp[:]>age[ss-1]*1e9))
                        for sstmp in con_tau[0]:
                            flux0  += fd_sed['col%d'%(sstmp+2)][:]
                            ms[ss] += fd_stm['col2'][sstmp]

                    else:
                        tautmp = tau0[pp]
                        agetmp = age[ss] - tau0[pp]/2.
                        con_tau= np.where((age_temp[:]<=age[ss]*1e9) & (age_temp[:]>(age[ss] - tau0[pp])*1e9))
                        for sstmp in con_tau[0]:
                            flux0  += fd_sed['col%d'%(sstmp+2)][:]
                            ms[ss] += fd_stm['col2'][sstmp]

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
                    #print(age[ss],age_temp[iis]/1e9,iis+2)

                # Keep tau in header;
                tau_age[ss] = tautmp
                age_age[ss] = agetmp

                # Then. add flux if tau > 0.
                con   = (wave0>lammin) & (wave0<lammax)
                wave, flux = wave0[con], flux0[con]

                Ls[ss]     = np.sum(flux0) # BPASS sed is in Lsun.
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
                    if fneb == 1:
                        tree.update({'logU': logU})

                    col3 = fits.Column(name='wavelength', format='E', unit='AA', array=wave)
                    col02.append(col3)

                    # ASDF
                    tree_spec.update({'wavelength': wave})

                col4  = fits.Column(name='fspec_'+str(zz)+'_'+str(ss)+'_'+str(pp), format='E', unit='Fnu', array=flux)
                col02.append(col4)

                # ASDF
                tree_spec.update({'fspec_'+str(zz)+'_'+str(ss)+'_'+str(pp): flux})

                if fneb == 1:
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
                # ASDF
                tree_ML.update({'ms_'+str(zz): ms})
                col2 = fits.Column(name='Ls_'+str(zz), format='E', unit='Lsun', array=Ls)
                # ASDF
                tree_ML.update({'Ls_'+str(zz): Ls})
                col01.append(col1)
                col01.append(col2)


    # Write;
    for aa in range(len(age)):
        #hdr['hierarch realtau%d(Gyr)'%(aa)] = tau_age[aa]
        tree.update({'realtau%d(Gyr)'%(aa): tau_age[aa]})
    for aa in range(len(age)):
        #hdr['hierarch realage%d(Gyr)'%(aa)] = age_age[aa]
        tree.update({'realage%d(Gyr)'%(aa): age_age[aa]})

    tree.update({'spec' : tree_spec})
    tree.update({'ML' : tree_ML})
    tree.update({'lick' : tree_lick})
    af = asdf.AsdfFile(tree)
    af.write_to(DIR_TMP + 'spec_all.asdf', all_array_compression='zlib')

    '''
    #
    # Create header;
    #
    hdr = fits.Header()
    hdr['COMMENT'] = 'Library:BPASS%s'%(BPASS_ver)
    if fneb == 1:
        hdr['logU'] = logU
    for aa in range(len(age)):
        hdr['hierarch realtau%d(Gyr)'%(aa)] = tau_age[aa]
    for aa in range(len(age)):
        hdr['hierarch realage%d(Gyr)'%(aa)] = age_age[aa]

    colspec = fits.ColDefs(col02)
    hdu2    = fits.BinTableHDU.from_columns(colspec, header=hdr)
    hdu2.writeto(DIR_TMP + 'spec_all.fits', overwrite=True)

    colind = fits.ColDefs(col05)
    hdu5   = fits.BinTableHDU.from_columns(colind, header=hdr)
    hdu5.writeto(DIR_TMP + 'index.fits', overwrite=True)

    col6 = fits.Column(name='tA', format='E', unit='Gyr', array=age[:])
    col01.append(col6)

    colms = fits.ColDefs(col01)
    hdu1  = fits.BinTableHDU.from_columns(colms, header=hdr)
    hdu1.writeto(DIR_TMP + 'ms.fits', overwrite=True)
    '''