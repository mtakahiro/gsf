import numpy as np
from astropy.io import fits
import asdf

from .function import filconv, calc_Dn4

def get_param(self, res, fitc, tcalc=1., burnin=-1):
    '''
    Purpose
    -------
    Write a parameter file.

    '''
    print('##########################')
    print('### Writing parameters ###')
    print('##########################')
    lib_all = self.lib_all

    # Those are from redshiftfit;
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
    af = self.af
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
                ZFIX = self.ZFIX
                Zb[aa] = ZFIX
                Zmc[aa,:] = [ZFIX, ZFIX, ZFIX]
        else:
            Zb[aa] = Zb[0]
            Zmc[aa,:] = Zmc[0,:]

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

    if self.ferr:
        logf = np.percentile(res.flatchain['logf'][burnin:], [16,50,84])
    else:
        logf = [-99,-99,-99]

    AA_tmp = np.zeros(len(age), dtype='float')
    ZZ_tmp = np.zeros(len(age), dtype='float')
    NZbest = np.zeros(len(age), dtype='int')

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
    prihdr['tcalc']  = (tcalc, 'in second')
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
        Mdust_per_temp = self.af['spec_dust']['Mdust']
        if self.DT0 == self.DT1 or self.DT0 + self.dDT >= self.DT1:
            nTdustmc[:] = [0,0,0]
            Tdustmc[:] = [self.DT0,self.DT0,self.DT0]
        else:
            if not self.TDUSTFIX == None:
                nTdustmc[:] = [self.NTDUST,self.NTDUST,self.NTDUST]
            else:
                nTdustmc[:] = np.percentile(res.flatchain['TDUST'][burnin:], [16,50,84])
            Tdustmc[:] = self.DT0 + self.dDT * nTdustmc[:]

        Mdustmc[:] = np.percentile(res.flatchain['MDUST'][burnin:], [16,50,84])
        col50 = fits.Column(name='ADUST', format='E', unit='Msun', array=Mdustmc[:])
        col01.append(col50)

        # Then M2Light ratio;
        Mdustmc[0] += np.log10(Mdust_per_temp[int(nTdustmc[0])])
        Mdustmc[1] += np.log10(Mdust_per_temp[int(nTdustmc[1])])
        Mdustmc[2] += np.log10(Mdust_per_temp[int(nTdustmc[2])])

        col50 = fits.Column(name='MDUST', format='E', unit='Msun', array=Mdustmc[:])
        col01.append(col50)
        col50 = fits.Column(name='nTDUST', format='E', unit='', array=nTdustmc[:])
        col01.append(col50)
        col50 = fits.Column(name='TDUST', format='E', unit='K', array=Tdustmc[:])
        col01.append(col50)

    if self.fneb:
        Anebmc = np.zeros(3, dtype=float)
        logUmc = np.zeros(3, dtype=float)
        Anebmc[:] = np.percentile(res.flatchain['Aneb'][burnin:], [16,50,84])
        if not self.logUFIX == None:
            logUmc[:] = [self.logUFIX,self.logUFIX,self.logUFIX]
        else:
            logUmc[:] = np.percentile(res.flatchain['logU'][burnin:], [16,50,84])
        col50 = fits.Column(name='Aneb', format='E', unit='', array=Anebmc[:])
        col01.append(col50)
        col50 = fits.Column(name='logU', format='E', unit='', array=logUmc[:])
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

    col50 = fits.Column(name='logf', format='E', unit='', array=logf)
    col01.append(col50)

    colms = fits.ColDefs(col01)
    dathdu = fits.BinTableHDU.from_columns(colms)
    hdu = fits.HDUList([prihdu, dathdu])
    hdu.writeto(self.DIR_OUT + 'summary_' + ID0 + '.fits', overwrite=True)



def get_index(mmax=300):
    '''
    Purpose
    -------
    Retrieve spectral indices from each realization.
    '''

    if nmc<mmax:
        mmax = int(nmc/2.)

    # RF color
    uv = np.zeros(int(mmax), dtype='float')
    bv = np.zeros(int(mmax), dtype='float')
    vj = np.zeros(int(mmax), dtype='float')
    zj = np.zeros(int(mmax), dtype='float')

    # Lick indeces
    Dn4  = np.zeros(int(mmax), dtype='float')
    Mgb  = np.zeros(int(mmax), dtype='float')
    Fe52 = np.zeros(int(mmax), dtype='float')
    Fe53 = np.zeros(int(mmax), dtype='float')
    Mg1  = np.zeros(int(mmax), dtype='float')
    Mg2  = np.zeros(int(mmax), dtype='float')
    G4300= np.zeros(int(mmax), dtype='float')
    NaD  = np.zeros(int(mmax), dtype='float')
    Hb   = np.zeros(int(mmax), dtype='float')

    for mm in range(0,mmax,1):
        rn = np.random.randint(len(res.flatchain['A0']))

        for aa in range(len(age)):
            fit_params['A'+str(aa)].value = res.flatchain['A%d'%(aa)][rn]
            try:
                fit_params['Z'+str(aa)].value = res.flatchain['Z%d'%(aa)][rn]
            except:
                pass
        fit_params['Av'].value = res.flatchain['Av'][rn]

        model2, xm_tmp = fnc.tmp04(fit_params, zrecom, lib_all)

        # not necessary here.
        if self.f_dust:
            model2_dust, xm_tmp_dust = fnc.tmp04_dust(ID0, par_tmp, zrecom, lib_all, tau0=tau0)
            model2 = np.append(model2,model2_dust)
            xm_tmp = np.append(xm_tmp,xm_tmp_dust)

        Dn4[mm]= calc_Dn4(xm_tmp, model2, zrecom) # Dust attenuation is not included?
        lmrest = xm_tmp / (1. + zrecom)
        band0  = ['u','b','v','j','sz']
        try:
            lmconv,fconv = filconv(band0, lmrest, model2, fil_path) # model2 in fnu
            uv[mm] = -2.5*np.log10(fconv[0]/fconv[2])
            bv[mm] = -2.5*np.log10(fconv[1]/fconv[2])
            vj[mm] = -2.5*np.log10(fconv[2]/fconv[3])
            zj[mm] = -2.5*np.log10(fconv[4]/fconv[3])
        except:
            uv[mm] = 0
            bv[mm] = 0
            vj[mm] = 0
            zj[mm] = 0

    conper = (Dn4>-99) #(Dn4>0)
    Dnmc = np.percentile(Dn4[conper], [16,50,84])
    uvmc = np.percentile(uv[conper], [16,50,84])
    bvmc = np.percentile(bv[conper], [16,50,84])
    vjmc = np.percentile(vj[conper], [16,50,84])
    zjmc = np.percentile(zj[conper], [16,50,84])

    Mgbmc  = np.percentile(Mgb[conper], [16,50,84])
    Fe52mc = np.percentile(Fe52[conper], [16,50,84])
    Fe53mc = np.percentile(Fe53[conper], [16,50,84])
    G4300mc= np.percentile(G4300[conper], [16,50,84])
    NaDmc  = np.percentile(NaD[conper], [16,50,84])
    Hbmc   = np.percentile(Hb[conper], [16,50,84])
    Mg1mc  = np.percentile(Mg1[conper], [16,50,84])
    Mg2mc  = np.percentile(Mg2[conper], [16,50,84])


    # Dn4000
    col50 = fits.Column(name='Dn4', format='E', unit='', array=Dnmc[:])
    col01.append(col50)


    # U-V
    col50 = fits.Column(name='uv', format='E', unit='mag', array=uvmc[:])
    col01.append(col50)

    # V-J
    col50 = fits.Column(name='vj', format='E', unit='mag', array=vjmc[:])
    col01.append(col50)

    # B-V
    col50 = fits.Column(name='bv', format='E', unit='mag', array=bvmc[:])
    col01.append(col50)

    # z-J
    col50 = fits.Column(name='zj', format='E', unit='mag', array=zjmc[:])
    col01.append(col50)

    # Mgb
    col50 = fits.Column(name='Mgb', format='E', unit='', array=Mgbmc[:])
    col01.append(col50)

    # Fe5270
    col50 = fits.Column(name='Fe5270', format='E', unit='', array=Fe52mc[:])
    col01.append(col50)

    # Fe5335
    col50 = fits.Column(name='Fe5335', format='E', unit='', array=Fe53mc[:])
    col01.append(col50)

    # G4300
    col50 = fits.Column(name='G4300', format='E', unit='', array=G4300mc[:])
    col01.append(col50)

    # NaD
    col50 = fits.Column(name='NaD', format='E', unit='', array=NaDmc[:])
    col01.append(col50)

    # Hb
    col50 = fits.Column(name='Hb', format='E', unit='', array=Hbmc[:])
    col01.append(col50)

    # Mg1
    col50 = fits.Column(name='Mg1', format='E', unit='', array=Mg1mc[:])
    col01.append(col50)

    # Mg2
    col50 = fits.Column(name='Mg2', format='E', unit='', array=Mg2mc[:])
    col01.append(col50)
