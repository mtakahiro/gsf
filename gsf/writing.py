import numpy as np
from astropy.io import fits
from lmfit import Parameters

from .function import filconv, calc_Dn4

def get_param(self, res, fitc, tcalc=1., burnin=-1):
    '''
    '''
    print('##########################')
    print('### Writing parameters ###')
    print('##########################')
    lib_all = self.lib_all
    zrecom = self.zrecom
    Czrec0 = self.Czrec0
    Czrec1 = self.Czrec1
    z_cz   = self.z_cz
    scl_cz0= self.scl_cz0
    scl_cz1= self.scl_cz1
    tau0   = self.tau0

    ID0 = self.ID
    PA0 = self.PA

    age  = self.age
    nage = self.nage
    Zall = self.Zall

    fnc  = self.fnc #Func(ID0, PA0, Zall, nage) # Set up the number of Age/ZZ
    bfnc = self.bfnc #Basic(Zall)

    DIR_TMP = self.DIR_TMP

    fil_path = self.DIR_FILT
    nmc  = self.nmc
    ndim = self.ndim

    samples = res.chain[:, :, :].reshape((-1, ndim))
    if burnin < 0:
         burnin = int(samples.shape[0]/2.)

    ##############################
    # Best parameters
    Amc  = np.zeros((len(age),3), dtype='float32')
    Ab   = np.zeros(len(age), dtype='float32')
    Zmc  = np.zeros((len(age),3), dtype='float32')
    Zb   = np.zeros(len(age), dtype='float32')
    NZbest = np.zeros(len(age), dtype='int')
    if self.f_dust:
        Mdustmc = np.zeros(3, dtype='float32')
        nTdustmc= np.zeros(3, dtype='float32')
        Tdustmc = np.zeros(3, dtype='float32')

    f0     = fits.open(DIR_TMP + 'ms_' + ID0 + '_PA' + PA0 + '.fits')
    sedpar = f0[1]
    ms     = np.zeros(len(age), dtype='float32')
    msmc0  = np.zeros(len(res.flatchain['A0'][burnin:]), dtype='float32')

    for aa in range(len(age)):
        Ab[aa]    = res.params['A'+str(aa)].value
        Amc[aa,:] = np.percentile(res.flatchain['A'+str(aa)][burnin:], [16,50,84])
        try:
            Zb[aa]    = res.params['Z'+str(aa)].value
            Zmc[aa,:] = np.percentile(res.flatchain['Z'+str(aa)][burnin:], [16,50,84])
        except:
            try:
                Zb[aa]    = res.params['Z0'].value
                Zmc[aa,:] = np.percentile(res.flatchain['Z0'][burnin:], [16,50,84])
            except:
                Zb[aa]    = self.ZFIX
                Zmc[aa,:] = [self.ZFIX,self.ZFIX,self.ZFIX]

        NZbest[aa]= bfnc.Z2NZ(Zb[aa])
        ms[aa]    = sedpar.data['ML_' +  str(NZbest[aa])][aa]
        msmc0[:] += res.flatchain['A' + str(aa)][burnin:] * ms[aa]


    msmc  = np.percentile(msmc0, [16,50,84])

    try:
        Avb   = res.params['Av'].value
        Avmc  = np.percentile(res.flatchain['Av'][burnin:], [16,50,84])
    except:
        Avb   = self.AVFIX
        Avmc  = [self.AVFIX,self.AVFIX,self.AVFIX]

    AAvmc = [Avmc]
    try:
        zmc = np.percentile(res.flatchain['zmc'][burnin:], [16,50,84])
    except:
        zmc = z_cz

    AA_tmp = np.zeros(len(age), dtype='float32')
    ZZ_tmp = np.zeros(len(age), dtype='float32')
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
    file = DIR_TMP + 'spec_obs_' + ID0 + '_PA' + PA0 + '.cat'
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
    prihdr['PA']     = PA0
    prihdr['Cz0']    = Czrec0
    prihdr['Cz1']    = Czrec1
    prihdr['z']      = zrecom
    prihdr['SN']     = SN
    prihdr['nSN']    = NSN
    prihdr['NDIM']   = ndim
    prihdr['tcalc']  = tcalc
    prihdr['chi2']   = fitc[0]
    prihdr['chi2nu'] = fitc[1]
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

    if self.f_dust:
        Mdustmc[:]  = np.percentile(res.flatchain['MDUST'][burnin:], [16,50,84])
        nTdustmc[:] = np.percentile(res.flatchain['TDUST'][burnin:], [16,50,84])
        Tdustmc[:]  = self.DT0 + self.dDT * nTdustmc[:]
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

    colms  = fits.ColDefs(col01)
    dathdu = fits.BinTableHDU.from_columns(colms)
    hdu    = fits.HDUList([prihdu, dathdu])
    hdu.writeto('summary_' + ID0 + '_PA' + PA0 + '.fits', overwrite=True)

    ##########
    # LINES
    ##########
    LW, fLW = self.get_lines(self.LW0)
    fw = open('table_' + ID0 + '_PA' + PA0 + '_lines.txt', 'w')
    fw.write('# ID PA WL Fcont50 Fcont16 Fcont84 Fline50 Fline16 Fline84 EW50 EW16 EW84\n')
    for ii in range(len(LW)):
        if fLW[ii] == 1:
            fw.write('%s %s %d %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n'%(ID0, PA0, LW[ii], np.median(Fcont[ii,:]), np.percentile(Fcont[ii,:],16), np.percentile(Fcont[ii,:],84), np.median(Fline[ii,:]), np.percentile(Fline[ii,:],16), np.percentile(Fline[ii,:],84), np.median(EW[ii,:]), np.percentile(EW[ii,:],16), np.percentile(EW[ii,:],84)))
        else:
            fw.write('%s %s %d 0 0 0 0 0 0 0 0 0\n'%(ID0, PA0, LW[ii]))
    fw.close()



def get_index(mmax=300):
    '''
    Purpose:
    ==========
    Retrieve spectral indices from each realization.
    '''

    if nmc<mmax:
        mmax = int(nmc/2.)

    # RF color
    uv = np.zeros(int(mmax), dtype='float32')
    bv = np.zeros(int(mmax), dtype='float32')
    vj = np.zeros(int(mmax), dtype='float32')
    zj = np.zeros(int(mmax), dtype='float32')

    # Lick indeces
    Dn4  = np.zeros(int(mmax), dtype='float32')
    Mgb  = np.zeros(int(mmax), dtype='float32')
    Fe52 = np.zeros(int(mmax), dtype='float32')
    Fe53 = np.zeros(int(mmax), dtype='float32')
    Mg1  = np.zeros(int(mmax), dtype='float32')
    Mg2  = np.zeros(int(mmax), dtype='float32')
    G4300= np.zeros(int(mmax), dtype='float32')
    NaD  = np.zeros(int(mmax), dtype='float32')
    Hb   = np.zeros(int(mmax), dtype='float32')

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
            model2_dust, xm_tmp_dust = fnc.tmp04_dust(ID0, PA0, par_tmp, zrecom, lib_all, tau0=tau0)
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
