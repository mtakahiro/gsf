import numpy as np
from astropy.io import fits
from lmfit import Parameters
import asdf

from .function import filconv, calc_Dn4

def get_param(self, res, fitc, tcalc=1., burnin=-1):
    '''
    Purpose:
    ========
    Write a parameter file.

    '''
    print('##########################')
    print('### Writing parameters ###')
    print('##########################')
    lib_all = self.lib_all

    # Those are from redshiftfit;
    #zrecom = self.zrecom
    #Czrec0 = self.Czrec0
    #Czrec1 = self.Czrec1
    zrecom = self.zgal
    Czrec0 = self.Cz0
    Czrec1 = self.Cz1

    try:
        z_cz = self.z_cz
        scl_cz0 = self.scl_cz0
        scl_cz1 = self.scl_cz1
    except: # When redshiftfit is skipped.
        z_cz = np.asarray([self.zgal,self.zgal,self.zgal])
        scl_cz0 = np.asarray([self.Cz0,self.Cz0,self.Cz0])
        scl_cz1 = np.asarray([self.Cz1,self.Cz1,self.Cz1])

    tau0 = self.tau0
    ID0 = self.ID
    age = self.age
    Zall = self.Zall

    fnc  = self.fnc 
    bfnc = self.bfnc 

    DIR_TMP = self.DIR_TMP

    fil_path = self.DIR_FILT
    nmc  = self.nmc
    ndim = self.ndim
    nwalker = self.nwalk

    #samples = res.chain[:, :, :].reshape((-1, ndim))
    samples = res.flatchain
    if burnin < 0:
         burnin = int(samples.shape[0]/2.)

    ##############################
    # Best parameters
    Amc = np.zeros((len(age),3), dtype='float')
    Ab = np.zeros(len(age), dtype='float')
    Zmc = np.zeros((len(age),3), dtype='float')
    Zb = np.zeros(len(age), dtype='float')
    NZbest = np.zeros(len(age), dtype='int')
    AGEmc = np.zeros((len(age),3), dtype='float')
    TAUmc = np.zeros((len(age),3), dtype='float')
    if self.f_dust:
        Mdustmc = np.zeros(3, dtype='float')
        nTdustmc= np.zeros(3, dtype='float')
        Tdustmc = np.zeros(3, dtype='float')

    # ASDF;
    af = asdf.open(DIR_TMP + 'spec_all_' + ID0 + '.asdf')
    sedpar = af['ML']
    
    ms = np.zeros(len(age), dtype='float')
    try:
        msmc0 = np.zeros(len(res.flatchain['A%d'%self.aamin[0]][burnin:]), dtype='float')
    except:
        msmc0 = np.zeros(len(res.flatchain['Av'][burnin:]), dtype='float')

    for aa in range(len(age)):
        try:
            Ab[aa] = res.params['A'+str(aa)].value
            Amc[aa,:] = np.percentile(res.flatchain['A'+str(aa)][burnin:], [16,50,84])
        except:
            Ab[aa] = -99
            Amc[aa,:] = [-99,-99,-99]
            pass
        if aa == 0 or self.ZEVOL:
            try:
                Zb[aa] = res.params['Z'+str(aa)].value
                Zmc[aa,:] = np.percentile(res.flatchain['Z'+str(aa)][burnin:], [16,50,84])
            except:
                Zb[aa] = self.ZFIX
                Zmc[aa,:] = [self.ZFIX,self.ZFIX,self.ZFIX]

        if self.SFH_FORM == -99:
            NZbest = bfnc.Z2NZ(Zb[aa])
            ms[aa] = sedpar['ML_' +  str(NZbest)][aa]
            try:
                msmc0[:] += 10**res.flatchain['A' + str(aa)][burnin:] * ms[aa]
            except:
                pass
        else:
            taub = res.params['TAU'+str(aa)].value
            ageb = res.params['AGE'+str(aa)].value
            NZbest,NTAU,NAGE = bfnc.Z2NZ(Zb[aa],taub,ageb)
            ms[aa] = sedpar['ML_%d_%d'%(NZbest, NTAU)][NAGE]
            try:
                msmc0[:] += 10**res.flatchain['A' + str(aa)][burnin:] * ms[aa]
            except:
                pass

            AGEmc[aa,:] = np.percentile(res.flatchain['AGE'+str(aa)][burnin:], [16,50,84])
            TAUmc[aa,:] = np.percentile(res.flatchain['TAU'+str(aa)][burnin:], [16,50,84])

    #
    msmc = np.percentile(msmc0, [16,50,84])
    try:
        Avb = res.params['Av'].value
        Avmc = np.percentile(res.flatchain['Av'][burnin:], [16,50,84])
    except:
        Avb = self.AVFIX
        Avmc = [self.AVFIX,self.AVFIX,self.AVFIX]

    AAvmc = [Avmc]
    if self.fzmc:
        zmc = np.percentile(res.flatchain['zmc'][burnin:], [16,50,84])
    else:
        zmc = z_cz

    AA_tmp = np.zeros(len(age), dtype='float')
    ZZ_tmp = np.zeros(len(age), dtype='float')
    NZbest = np.zeros(len(age), dtype='int')

    #
    # Get mcmc model templates, plus some indicies.
    #
    # This is just for here;
    fit_params = Parameters()
    for aa in range(len(age)):
        fit_params.add('A'+str(aa), value=1., min=0, max=10000)
    fit_params.add('Av', value=0., min=0, max=10)
    for aa in range(len(age)):
        try:
            fit_params.add('Z'+str(aa), value=0.0, min=-10, max=10)
        except:
            pass


    ############
    # Get SN.
    ############
    file = DIR_TMP + 'spec_obs_' + ID0 + '.cat'
    fds  = np.loadtxt(file, comments='#')
    nrs  = fds[:,0]
    lams = fds[:,1]
    fsp  = fds[:,2]
    esp  = fds[:,3]

    consp = (nrs<10000) & (lams/(1.+zrecom)>3600) & (lams/(1.+zrecom)<4200) & (esp>0)
    NSN   = len(fsp[consp])
    if NSN>0:
        SN = np.median((fsp/esp)[consp])
    else:
        SN = 0

    ######################
    # Write in Fits table.
    ######################
    # Header
    prihdr = fits.Header()
    prihdr['ID']     = ID0
    prihdr['Cz0']    = Czrec0
    prihdr['Cz1']    = Czrec1
    prihdr['z']      = zrecom
    prihdr['zmc']    = zmc[1]
    prihdr['SN']     = SN
    prihdr['nSN']    = NSN
    prihdr['NDIM']   = ndim
    prihdr['tcalc']  = tcalc
    prihdr['chi2']   = fitc[0]
    prihdr['chi2nu'] = fitc[1]
    prihdr['bic'] = res.bic
    prihdr['nmc'] = nmc
    prihdr['nwalker'] = nwalker
    import gsf
    prihdr['version'] = gsf.__version__
    prihdu = fits.PrimaryHDU(header=prihdr)

    # Data
    col01 = []
    for aa in range(len(age)):
        col50 = fits.Column(name='A'+str(aa), format='E', unit='', array=Amc[aa][:])
        col01.append(col50)

    for aa in range(len(AAvmc)):
        col50 = fits.Column(name='Av'+str(aa), format='E', unit='mag', array=AAvmc[aa][:])
        col01.append(col50)

    for aa in range(len(Zmc)):
        col50 = fits.Column(name='Z'+str(aa), format='E', unit='logZsun', array=Zmc[aa][:])
        col01.append(col50)

    for aa in range(len(AGEmc)):
        col50 = fits.Column(name='AGE'+str(aa), format='E', unit='logGyr', array=AGEmc[aa][:])
        col01.append(col50)
    for aa in range(len(TAUmc)):
        col50 = fits.Column(name='TAU'+str(aa), format='E', unit='logGyr', array=TAUmc[aa][:])
        col01.append(col50)

    if self.f_dust:
        Mdustmc[:] = np.percentile(res.flatchain['MDUST'][burnin:], [16,50,84])
        if self.DT0 == self.DT1 or self.DT0 + self.dDT <= self.DT1:
            nTdustmc[:] = [0,0,0] #np.percentile(res.flatchain['TDUST'][burnin:], [16,50,84])
            Tdustmc[:] = [self.DT0,self.DT0,self.DT0]
        else:
            nTdustmc[:] = np.percentile(res.flatchain['TDUST'][burnin:], [16,50,84])
            Tdustmc[:] = self.DT0 + self.dDT * nTdustmc[:]

        col50 = fits.Column(name='MDUST', format='E', unit='Msun', array=Mdustmc[:])
        col01.append(col50)
        col50 = fits.Column(name='nTDUST', format='E', unit='K', array=nTdustmc[:])
        col01.append(col50)
        col50 = fits.Column(name='TDUST', format='E', unit='K', array=Tdustmc[:])
        col01.append(col50)

    # zmc
    col50 = fits.Column(name='zmc', format='E', unit='', array=zmc[:])
    col01.append(col50)

    # Mass
    col50 = fits.Column(name='ms', format='E', unit='Msun', array=msmc[:])
    col01.append(col50)

    # zmc
    col50 = fits.Column(name='z_cz', format='E', unit='', array=z_cz[:])
    col01.append(col50)

    # Chi
    col50 = fits.Column(name='chi', format='E', unit='', array=fitc[:])
    col01.append(col50)

    # C0 scale
    col50 = fits.Column(name='Cscale0', format='E', unit='', array=scl_cz0[:])
    col01.append(col50)

    # C1 scale
    col50 = fits.Column(name='Cscale1', format='E', unit='', array=scl_cz1[:])
    col01.append(col50)

    colms = fits.ColDefs(col01)
    dathdu = fits.BinTableHDU.from_columns(colms)
    hdu = fits.HDUList([prihdu, dathdu])
    hdu.writeto(self.DIR_OUT + 'summary_' + ID0 + '.fits', overwrite=True)
