# Last update;
# 2017.12.29
#
# This is for the preparation of
# default template, with FSPS, at z=0.
#
# Should be run before SED fitting.
#
# Parameters are; IMF, metallicity (list), SFH (SSP default)
#
import numpy as np
import fsps
from astropy.io import fits
import matplotlib.pyplot as plt
import os

######################
# Fixed Parameters
######################
c      = 3e18
Mpc_cm = 3.08568025e+24 # cm/Mpc
m0set  = 25.0
pixelscale = 0.06 # arcsec/pixel

#######################
# Path
#######################
DIR_TMP = './templates/' # Templates are saved in this directory.
INDICES = ['G4300', 'Mgb', 'Fe5270', 'Fe5335', 'NaD', 'Hb', 'Fe4668', 'Fe5015', 'Fe5709', 'Fe5782', 'Mg1', 'Mg2', 'TiO1', 'TiO2']

#
# Get Lick index
# for input input
#
def get_ind(wave,flux):
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


def make_tmp_z0(nimf=0, Z=np.arange(-1.2,0.4249,0.05), age=[0.01, 0.1, 0.3, 0.7, 1.0, 3.0], lammin = 400, lammax = 80000, tau0 = [0.01,0.02,0.03], fneb=0, logU=-2.5):
    #
    # nimf (int) : 0:Salpeter, 1:Chabrier, 2:Kroupa, 3:vanDokkum08,...
    # Z (array)  : Stellar phase metallicity in logZsun.
    # age (array): Age, in Gyr.
    # fneb (int) : flag for adding nebular emissionself.
    # logU (float): ionizing parameter, in logU.
    #
    NZ = len(Z)
    Na = len(age)

    print('#######################################')
    print('Making templates at z=0, IMF=%d'%(nimf))
    print('#######################################')

    col01 = [] # For M/L ratio.
    col02 = [] # For templates
    col05 = [] # For spectral indices.
    #col06 = [] # For weird templates for UVJ calculation;
    for zz in range(len(Z)):

        for pp in range(len(tau0)):
            spall = [] # For sps model
            ms = np.zeros(Na, dtype='float32')
            Ls = np.zeros(Na, dtype='float32')
            LICK = np.zeros((Na,len(INDICES)), dtype='float32')

            tau0_old = 0
            for ss in range(Na):
                if ss > 0:
                    tautmp = age[ss] - age[ss-1]
                    if tau0[pp] > 0.0:
                        tautmp = tau0[pp]
                else:
                    tautmp = 0.01

                # Short cut.
                if tautmp != tau0_old:
                    if tau0[pp] == 99:
                        tautmp = 0.01
                        print('SSP is applied.')
                        sptmp  = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1, imf_type=nimf, sfh=0, logzsol=Z[zz], dust_type=2, dust2=0.0) # Lsun/Hz
                        if fneb == 1:
                            esptmp = fsps.StellarPopulation(zcontinuous=1, imf_type=nimf, sfh=0, logzsol=Z[zz], dust_type=2, dust2=0.0, add_neb_emission=1)
                    elif tau0[pp] > 0.0:
                        print('Fixed tau, %.3f, is applied.'%(tau0[pp]))
                        sptmp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1, imf_type=nimf, sfh=1, logzsol=Z[zz], dust_type=2, dust2=0.0, tau=20, const=0, sf_start=0, sf_trunc=tautmp, tburst=13, fburst=0) # Lsun/Hz
                        if fneb == 1:
                            esptmp = fsps.StellarPopulation(zcontinuous=1, imf_type=nimf, sfh=1, logzsol=Z[zz], dust_type=2, dust2=0.0, tau=20, const=0, sf_start=0, sf_trunc=tautmp, tburst=13, fburst=0, add_neb_emission=1)
                    else:
                        print('CSP is applied.')
                        print('tau is %.2f Gyr' % tautmp)
                        sptmp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1, imf_type=nimf, sfh=1, logzsol=Z[zz], dust_type=2, dust2=0.0, tau=20, const=0, sf_start=0, sf_trunc=tautmp, tburst=13, fburst=0) # Lsun/Hz
                        if fneb == 1:
                            esptmp = fsps.StellarPopulation(zcontinuous=1, imf_type=nimf, sfh=1, logzsol=Z[zz], dust_type=2, dust2=0.0, tau=20, const=0, sf_start=0, sf_trunc=tautmp, tburst=13, fburst=0, add_neb_emission=1)
                else:
                    print('Skip fsps, by using previous library.')

                tau0_old = tautmp
                sp  = sptmp
                print(zz, sp.libraries[0].decode("utf-8") , sp.libraries[1].decode("utf-8") , pp)

                wave0, flux0 = sp.get_spectrum(tage=age[ss], peraa=True)
                con = (wave0>lammin) & (wave0<lammax)
                wave, flux = wave0[con], flux0[con]

                if fneb == 1:
                    esptmp.params['gas_logz'] = Z[zz] # gas metallicity, assuming = Zstel
                    esptmp.params['gas_logu'] = logU # ionization parameter
                    esp = esptmp
                    print('Nebular is also added, with logU=%.2f.'%(logU))
                    ewave0, eflux0 = esp.get_spectrum(tage=age[ss], peraa=True)
                    con = (ewave0>lammin) & (ewave0<lammax)
                    eflux = eflux0[con]

                #plt.plot(wave, flux, linestyle='-')
                #plt.plot(wave, eflux, linestyle='--')
                #plt.show()

                # Mass-Luminosity
                ms[ss]  = sp.stellar_mass
                Ls[ss]  = 10**sp.log_lbol
                LICK[ss,:] = get_ind(wave, flux)

                if ss == 0 and pp == 0 and zz == 0:
                    col3 = fits.Column(name='wavelength', format='E', unit='AA', array=wave)
                    col02.append(col3)

                col4  = fits.Column(name='fspec_'+str(zz)+'_'+str(ss)+'_'+str(pp), format='E', unit='Fnu', array=flux)
                col02.append(col4)
                if fneb == 1:
                    col4e = fits.Column(name='efspec_'+str(zz)+'_'+str(ss)+'_'+str(pp), format='E', unit='Fnu', array=eflux)
                    col02.append(col4e)

                for ss0 in range(len(age)):
                    if ss == 0 and age[ss0] == age[ss]:
                        wave1, flux1 = sp.get_spectrum(tage=0.01, peraa=True)
                        flux1 /= 10**sp.log_lbol
                        con1 = (wave1>lammin) & (wave1<lammax)
                        col001 = fits.Column(name='fspec_'+str(zz)+'_'+str(ss)+'_'+str(pp)+'_'+str(ss0), format='E', unit='Fnu', array=flux1[con1])
                        col02.append(col001)
                        if fneb == 1:
                            ewave1, eflux1 = esp.get_spectrum(tage=0.01, peraa=True)
                            eflux1 /= 10**esp.log_lbol
                            con1 = (ewave1>lammin) & (ewave1<lammax)
                            col001 = fits.Column(name='efspec_'+str(zz)+'_'+str(ss)+'_'+str(pp)+'_'+str(ss0), format='E', unit='Fnu', array=eflux1[con1])
                            col02.append(col001)

                    if age[ss0] < age[ss]:
                        wave1, flux1 = sp.get_spectrum(tage=age[ss] - age[ss0], peraa=True)
                        flux1 /= 10**sp.log_lbol
                        con1 = (wave1>lammin) & (wave1<lammax)
                        col001 = fits.Column(name='fspec_'+str(zz)+'_'+str(ss)+'_'+str(pp)+'_'+str(ss0), format='E', unit='Fnu', array=flux1[con1])
                        col02.append(col001)
                        if fneb == 1:
                            ewave1, eflux1 = esp.get_spectrum(tage=age[ss] - age[ss0], peraa=True)
                            eflux1 /= 10**esp.log_lbol
                            con1 = (ewave1>lammin) & (ewave1<lammax)
                            col001 = fits.Column(name='efspec_'+str(zz)+'_'+str(ss)+'_'+str(pp)+'_'+str(ss0), format='E', unit='Fnu', array=eflux1[con1])
                            col02.append(col001)



            if pp == 0:
                # use tau0[0] as representative for M/L and index.
                for ll in range(len(INDICES)):
                    col5 = fits.Column(name=INDICES[ll]+'_'+str(zz), format='E', unit='', array=LICK[:,ll])
                    col05.append(col5)

                col1 = fits.Column(name='ms_'+str(zz), format='E', unit='Msun', array=ms)
                col2 = fits.Column(name='Ls_'+str(zz), format='E', unit='Lsun', array=Ls)
                col01.append(col1)
                col01.append(col2)


    # ##############
    # Create header;
    hdr = fits.Header()
    hdr['COMMENT'] = 'Library:%s %s'%(sp.libraries[0].decode("utf-8"), sp.libraries[1].decode("utf-8"))
    if fneb == 1:
        hdr['logU'] = logU
    for pp in range(len(tau0)):
        hdr['Tau%d'%(pp)] = tau0[pp]

    colspec = fits.ColDefs(col02)
    hdu2    = fits.BinTableHDU.from_columns(colspec, header=hdr)
    hdu2.writeto(DIR_TMP + 'spec_all.fits', overwrite=True)

    colind = fits.ColDefs(col05)
    hdu5   = fits.BinTableHDU.from_columns(colind, header=hdr)
    hdu5.writeto(DIR_TMP + 'index.fits', overwrite=True)

    #colspec6 = fits.ColDefs(col06)
    #hdu6     = fits.BinTableHDU.from_columns(colspec6, header=hdr)
    #hdu6.writeto(DIR_TMP + 'spec_all_inv_'+str(zz)+'.fits', overwrite=True)

    #
    col6 = fits.Column(name='tA', format='E', unit='Gyr', array=age[:])
    col01.append(col6)

    colms = fits.ColDefs(col01)
    hdu1  = fits.BinTableHDU.from_columns(colms, header=hdr)
    hdu1.writeto(DIR_TMP + 'ms.fits', overwrite=True)
