# For fitting.
from scipy import asarray as ar,exp
import numpy as np
import sys
from scipy.integrate import simps
import pickle as cPickle

#
c = 3.e18 # A/s
chimax = 1.
mag0 = 25.0
m0set= mag0
d = 10**(73.6/2.5) # From [ergs/s/cm2/A] to [ergs/s/cm2/Hz]

################
# Line library
################
LN0 = ['Mg2', 'Ne5', 'O2', 'Htheta', 'Heta', 'Ne3', 'Hdelta', 'Hgamma', 'Hbeta', 'O3L', 'O3H', 'Mgb', 'Halpha', 'S2L', 'S2H']
LW0 = [2800, 3347, 3727, 3799, 3836, 3869, 4102, 4341, 4861, 4960, 5008, 5175, 6563, 6717, 6731]
fLW = np.zeros(len(LW0), dtype='int') # flag.


def data_int(lmobs, lmtmp, ftmp):
    # lmobs: Observed wavelength.
    # lmtmp, ftmp: Those to be interpolated.
    ftmp_int  = np.interp(lmobs,lmtmp,ftmp) # Interpolate model flux to observed wavelength axis.
    return ftmp_int

def flamtonu(lam, flam):
    Ctmp = lam **2/c * 10**((48.6+m0set)/2.5) #/ delx_org
    fnu  = flam * Ctmp
    return fnu

def fnutolam(lam, fnu):
    Ctmp = lam **2/c * 10**((48.6+m0set)/2.5) #/ delx_org
    flam  = fnu / Ctmp
    return flam

def gauss(x,A,sig):
    return A * np.exp(-0.5*x**2/sig**2)

def moffat(xx, A, x0, gamma, alp):
    yy = A * (1. + (xx-x0)**2/gamma**2)**(-alp)
    return yy

def get_filt(LIBFILT, NFILT):
    #f = open(LIBFILT + '.info', 'r')
    f = open(LIBFILT + '', 'r')

def get_fit(x,y,xer,yer, nsfh='Del.'):
    from lmfit import Model, Parameters, minimize, fit_report, Minimizer

    minsfr = 1e-10
    def SFH_del(t0, tau, A, tt=np.arange(0.,10,0.1)): # tt = x_data
        sfr = np.zeros(len(tt), dtype='float32') + minsfr
        sfr[:] = A * (tt[:]-t0) * np.exp(-(tt[:]-t0)/tau)
        con = (tt[:]-t0<0)
        sfr[:][con] = minsfr
        return sfr

    def SFH_dec(t0, tau, A, tt=np.arange(0.,10,0.1)):
        sfr = np.zeros(len(tt), dtype='float32') + minsfr
        sfr[:] = A * (np.exp(-(tt[:]-t0)/tau))
        con = (tt[:]-t0<0)
        sfr[:][con] = minsfr
        return sfr

    def SFH_cons(t0, tau, A, tt=np.arange(0.,10,0.1)):
        sfr = np.zeros(len(tt), dtype='float32') + minsfr
        sfr[:] = A #* (np.exp(-(tt[:]-t0)/tau))
        con = (tt[:]<t0) | (tt[:]>tau+t0)
        sfr[:][con] = minsfr
        return sfr


    fit_params = Parameters()
    #fit_params.add('t0', value=1., min=0, max=np.max(tt))
    fit_params.add('t0', value=.5, min=0, max=14)
    fit_params.add('tau', value=.1, min=0, max=100)
    fit_params.add('A', value=1, min=0, max=5000)

    def residual(pars):
        vals  = pars.valuesdict()
        t0_tmp, tau_tmp, A_tmp = vals['t0'],vals['tau'],vals['A']

        if nsfh == 'Del.':
            model = SFH_del(t0_tmp, tau_tmp, A_tmp, tt=x)
        elif nsfh == 'Decl.':
            model = SFH_dec(t0_tmp, tau_tmp, A_tmp, tt=x)
        elif nsfh == 'Cons.':
            model = SFH_cons(t0_tmp, tau_tmp, A_tmp, tt=x)

        #con = (model>minsfr)
        con = (model>0)
        #print(model[con])
        #resid = np.abs(model - y)[con] / np.sqrt(yer[con])
        #resid = np.square(model - y)[con] / np.square(yer[con])
        #resid = np.square(np.log10(model[con]) - y[con]) / np.square(yer[con])
        #resid = (np.log10(model[con]) - y[con]) / np.sqrt(yer[con])
        resid = (np.log10(model[con]) - y[con]) / yer[con]
        #print(yer[con])
        #resid = (model - y)[con] / (yer[con])
        # i.e. residual/sigma
        return resid


    out = minimize(residual, fit_params, method='powell')
    #out = minimize(residual, fit_params, method='nelder')
    print(fit_report(out))

    t0    = out.params['t0'].value
    tau   = out.params['tau'].value
    A     = out.params['A'].value
    param = [t0, tau, A]

    keys = fit_report(out).split('\n')
    for key in keys:
        if key[4:7] == 'chi':
            skey = key.split(' ')
            csq  = float(skey[14])
        if key[4:7] == 'red':
            skey = key.split(' ')
            rcsq = float(skey[7])

    return param, rcsq

def savecpkl(data, cpklfile, verbose=True):
    """
    Save data into cpklfile.
    """
    if verbose: print(' => Saving data to cpklfile '+cpklfile)
    f = open(cpklfile,'wb')
    cPickle.dump(data, f, 2)
    f.close()

def dust_MW(lm, fl, Av): # input lm is at RF.
    # By Cardelli89
    Rv = 3.1 #\pm0.80 from Calzetti+00
    Alam = np.zeros(len(lm), dtype='float32')
    for ii0 in range(len(Alam)):
        lmm = lm[ii0]/10000. # in micron
        xx  = 1./lmm
        if xx<=1.1:
            ax = 0.574 * xx**1.61
            bx = -0.527 * xx**1.61
            Alam[ii0] = Av * (ax + bx / Rv)
            #Kl[ii0] = 2.659 * (-2.156 + 1.509/lmm - 0.198/lmm**2 + 0.011/lmm**3) + Rv
        elif xx>1.1 and xx<=3.3:
            yy = (xx - 1.82)
            ax = 1. + 0.17699 * yy - 0.50447 * yy**2 - 0.02427 * yy**3 + 0.72085 * yy**4\
                 + 0.01979 * yy**5 - 0.77530 * yy**6 + 0.32999 * yy**7
            bx = 1.41338 * yy + 2.28305 * yy**2 + 1.07233 * yy**3 - 5.38434 * yy**4\
                 - 0.62251 * yy**5 + 5.30260 * yy**6 - 2.09002 * yy**7
            Alam[ii0] = Av * (ax + bx / Rv)
        elif xx>3.3 and xx<=8.0:
            if xx>5.9 and xx<=8.0:
                Fax = -0.04473 * (xx - 5.9)**2 - 0.009779 * (xx - 5.9)**3
                Fbx = 0.2130 * (xx - 5.9)**2 + 0.1207 * (xx - 5.9)**3
            else:
                Fax = Fbx = 0
            ax = 1.752 - 0.316 * xx - 0.104/((xx-4.67)**2+0.341) + Fax
            bx = -3.090 + 1.825 * xx + 1.206/((xx-4.62)**2+0.263) + Fbx
            Alam[ii0] = Av * (ax + bx / Rv)
        elif xx>8.0:
            Fax = -0.04473 * (xx - 5.9)**2 - 0.009779 * (xx - 5.9)**3
            Fbx = 0.2130 * (xx - 5.9)**2 + 0.1207 * (xx - 5.9)**3
            ax = 1.752 - 0.316 * xx - 0.104/((xx-4.67)**2+0.341) + Fax
            bx = -3.090 + 1.825 * xx + 1.206/((xx-4.62)**2+0.263) + Fbx
            Alam[ii0] = Av * (ax + bx / Rv)
        else:
            Alam[ii0] = 99.
            print('Error in dust attenuation. at', xx, lm[ii0], lmm)
    fl_cor = fl * np.power(10,(-0.4*Alam))
    return fl_cor

# This function is much better than previous,
# but is hard to impliment for the current version.
def dust_MW2(lm, fl, Av, nr): # input lm is at RF.
    Rv = 3.1 #4.05 #\pm0.80 from Calzetti+00
    lmlimu = 3.115 # Upperlimit. 2.2 in Calz+00

    lmm  = lm/10000. # in micron
    xx   = 1./lmm
    con0 = (xx<1.1)
    con1 = (xx>=1.1) & (xx<3.3)
    con2 = (xx>=3.3) & (xx<5.9)
    con3 = (xx>=5.9) & (xx<8.0)
    con4 = (xx>=8.0)

    nr0 = nr[con0]
    nr1 = nr[con1]
    nr2 = nr[con2]
    nr3 = nr[con3]
    nr4 = nr[con4]

    lmm0 = lmm[con0]
    lmm1 = lmm[con1]
    lmm2 = lmm[con2]
    lmm3 = lmm[con3]
    lmm4 = lmm[con4]

    fl0 = fl[con0]
    fl1 = fl[con1]
    fl2 = fl[con2]
    fl3 = fl[con3]
    fl4 = fl[con3]

    ax0 =  0.574 * (1./lmm0)**1.61
    bx0 = -0.527 * (1./lmm0)**1.61

    yy  = ((1./lmm1) - 1.82)
    ax1 = 1. + 0.17699 * yy - 0.50447 * yy**2 - 0.02427 * yy**3 + 0.72085 * yy**4\
          + 0.01979 * yy**5 - 0.77530 * yy**6 + 0.32999 * yy**7
    bx1 = 1.41338 * yy + 2.28305 * yy**2 + 1.07233 * yy**3 - 5.38434 * yy**4\
          - 0.62251 * yy**5 + 5.30260 * yy**6 - 2.09002 * yy**7

    Fax2 = Fbx2 = 0
    ax2  = 1.752 - 0.316 * (1./lmm2) - 0.104/(((1./lmm2)-4.67)**2+0.341) + Fax2
    bx2  = -3.090 + 1.825 * (1./lmm2) + 1.206/(((1./lmm2)-4.62)**2+0.263) + Fbx2

    Fax3 = -0.04473 * ((1./lmm3) - 5.9)**2 - 0.009779 * ((1./lmm3) - 5.9)**3
    Fbx3 = 0.2130 * ((1./lmm3) - 5.9)**2 + 0.1207 * ((1./lmm3) - 5.9)**3
    ax3  = 1.752 - 0.316 * (1./lmm3) - 0.104/(((1./lmm3)-4.67)**2+0.341) + Fax3
    bx3  = -3.090 + 1.825 * (1./lmm3) + 1.206/(((1./lmm3)-4.62)**2+0.263) + Fbx3

    Fax4 = -0.04473 * ((1./lmm4) - 5.9)**2 - 0.009779 * ((1./lmm4) - 5.9)**3
    Fbx4 = 0.2130 * ((1./lmm4) - 5.9)**2 + 0.1207 * ((1./lmm4) - 5.9)**3
    ax4  = 1.752 - 0.316 * (1./lmm4) - 0.104/(((1./lmm4)-4.67)**2+0.341) + Fax4
    bx4  = -3.090 + 1.825 * (1./lmm4) + 1.206/(((1./lmm4)-4.62)**2+0.263) + Fbx4


    #Kl   = np.concatenate([Kl0,Kl1,Kl2,Kl3,Kl4])
    nrd  = np.concatenate([nr0,nr1,nr2,nr3,nr4])
    lmmc = np.concatenate([lmm0,lmm1,lmm2,lmm3,lmm4])
    flc  = np.concatenate([fl0,fl1,fl2,fl3,fl4])
    ax   = np.concatenate([ax0,ax1,ax2,ax3,ax4])
    bx   = np.concatenate([bx0,bx1,bx2,bx3,bx4])

    Alam   = Av * (ax + bx / Rv)
    fl_cor = flc * np.power(10,(-0.4*Alam))

    return fl_cor, lmmc*10000., nrd

# This function is much better than previous,
# but is hard to impliment for the current version.
def dust_calz2(lm, fl, Av, nr):
    #
    # lm (float array) : wavelength, at RF.
    # fl (float array) : fnu
    # Av (float)       : mag
    # nr (int array)   : index, to be used for sorting.
    #
    Rv = 4.05 #\pm0.80 from Calzetti+00
    lmlimu = 3.115 # Upperlimit. 2.2 in Calz+00
    Kl = np.zeros(len(lm), dtype='float32')

    lmm  = lm/10000. # in micron
    con1 = (lmm<=0.63)
    con2 = (lmm>0.63)  & (lmm<=lmlimu)
    con3 = (lmm>lmlimu)

    Kl1 = (2.659 * (-2.156 + 1.509/lmm[con1] - 0.198/lmm[con1]**2 + 0.011/lmm[con1]**3) + Rv)
    Kl2 = (2.659 * (-1.857 + 1.040/lmm[con2]) + Rv)
    Kl3 = (2.659 * (-1.857 + 1.040/lmlimu + lmm[con3] * 0) + Rv)

    #nr0 = nr[con0]
    nr1 = nr[con1]
    nr2 = nr[con2]
    nr3 = nr[con3]

    #lmm0 = lmm[con0]
    lmm1 = lmm[con1]
    lmm2 = lmm[con2]
    lmm3 = lmm[con3]

    #fl0 = fl[con0]
    fl1 = fl[con1]
    fl2 = fl[con2]
    fl3 = fl[con3]

    Kl   = np.concatenate([Kl1,Kl2,Kl3])
    nrd  = np.concatenate([nr1,nr2,nr3])
    lmmc = np.concatenate([lmm1,lmm2,lmm3])
    flc  = np.concatenate([fl1,fl2,fl3])

    Alam   = Kl * Av / Rv
    #fl_cor = flc * np.power(10,(-0.4*Alam))
    fl_cor = flc[:] * 10**(-0.4*Alam[:])

    return fl_cor, lmmc*10000., nrd

'''
def dust_calz3(lm, fl, Av, nr): # input lm is at RF.
    Rv = 4.05 #\pm0.80 from Calzetti+00
    lmlimu = 3.115 # Upperlimit. 2.2 in Calz+00
    Kl = np.zeros(len(lm), dtype='float32')

    lmm  = lm/10000. # in micron
    con1 = (lmm<=0.63)
    con2 = (lmm>0.63)  & (lmm<=lmlimu)
    con3 = (lmm>lmlimu)

    Kl1 = (2.659 * (-2.156 + 1.509/lmm[con1] - 0.198/lmm[con1]**2 + 0.011/lmm[con1]**3) + Rv)
    Kl2 = (2.659 * (-1.857 + 1.040/lmm[con2]) + Rv)
    Kl3 = (2.659 * (-1.857 + 1.040/lmlimu + lmm[con3] * 0) + Rv)

    #nr0 = nr[con0]
    nr1 = nr[con1]
    nr2 = nr[con2]
    nr3 = nr[con3]

    #lmm0 = lmm[con0]
    lmm1 = lmm[con1]
    lmm2 = lmm[con2]
    lmm3 = lmm[con3]

    #fl0 = fl[con0]
    fl1 = fl[con1]
    fl2 = fl[con2]
    fl3 = fl[con3]

    Kl   = np.concatenate([Kl1,Kl2,Kl3])
    nrd  = np.concatenate([nr1,nr2,nr3])
    lmmc = np.concatenate([lmm1,lmm2,lmm3])
    flc  = np.concatenate([fl1,fl2,fl3])

    Alam   = Kl * Av / Rv
    #fl_cor = flc * np.power(10,(-0.4*Alam))
    fl_cor = flc[:] * 10**(-0.4*Alam[:])

    return fl_cor, lmmc*10000., nrd
'''

def dust_calz(lm, fl, Av): # input lm is at RF.
    Rv = 4.05 #\pm0.80 from Calzetti+00
    lmlimu = 3.115 # Upperlimit. 2.2 in Calz+00
    Kl = np.zeros(len(lm), dtype='float32')
    for ii0 in range(len(Kl)):
        lmm = lm[ii0]/10000. # in micron
        if lmm<0.12:
            Kl[ii0] = 2.659 * (-2.156 + 1.509/lmm - 0.198/lmm**2 + 0.011/lmm**3) + Rv
        elif lmm>=0.12 and lmm<=0.63:
            Kl[ii0] = 2.659 * (-2.156 + 1.509/lmm - 0.198/lmm**2 + 0.011/lmm**3) + Rv
        elif lmm>0.63 and lmm<=lmlimu:
            Kl[ii0] = 2.659 * (-1.857 + 1.040/lmm) + Rv
        elif lmm>2.2:
            Kl[ii0] = 2.659 * (-1.857 + 1.040/lmlimu) + Rv
    Alam   = Kl * Av / Rv
    fl_cor = fl * np.power(10,(-0.4*Alam))
    return fl_cor

def check_line(data,wave,wht,model):
    R_grs = 50
    dw  = 10
    ldw = 5
    dlw = R_grs * 2
    lsig = 1.5 # significance of lines.
    ii0 = 0 + dw
    ii9 = len(data) - dw
    wd  = data * wht
    er  = 1./np.sqrt(wht)
    wht2 = wht

    #print('line check.')
    for ii in range(ii0, ii9, 1):
        concont = (((wave>wave[ii]-dw*R_grs) & (wave<wave[ii]-(dw-ldw)*R_grs)) \
                   | ((wave<wave[ii]+dw*R_grs) & ((wave>wave[ii]+(dw-ldw)*R_grs))))

        xcont = wave[concont]
        ycont = data[concont]
        wycont = wht[concont]

        if len(xcont)>5:
            try:
                z = np.polyfit(xcont, ycont, 1, w=wycont)
                p = np.poly1d(z)
                fconttmp  = p(wave[ii])
                fconttmp1 = p(wave[ii-1])
                fconttmp2 = p(wave[ii+1])

                if data[ii] > er[ii]*lsig + fconttmp and data[ii-1] > er[ii-1]*lsig + fconttmp1 and data[ii+1] > er[ii+1]*lsig + fconttmp2:
                    #print wave[ii]/(1.+zgal), dlw/(1+zgal)
                    for jj in range(len(LW)):
                        #print wave[ii],  LW[jj]*(1.+zgal) - dlw, LW[jj]*(1.+zgal) + dlw, LN[jj]
                        if wave[ii] > LW[jj]*(1.+zgal) - dlw\
                           and wave[ii] < LW[jj]*(1.+zgal) + dlw:
                            wht2[ii-dw:ii+dw] *= 0
                            fLW[jj] = 1
                            print(p, LN[jj], fconttmp, data[ii], wave[ii]/(1.+zgal))
                elif wht2[ii] != 0:
                    wht2[ii] = wht[ii]

            except Exception:
                #print('Error in Line Check.')
                pass

    return wht2

# Convolution of templates with filter response curves.
def filconv_cen(band0, l0, f0): # f0 in fnu
    DIR = 'FILT/'
    fnu  = np.zeros(len(band0), dtype='float32')
    lcen = np.zeros(len(band0), dtype='float32')
    for ii in range(len(band0)):
        fd = np.loadtxt(DIR + 'f' + str(band0[ii]) + 'w.fil', comments='#')
        lfil = fd[:,1]
        ffil = fd[:,2]

        lmin  = np.min(lfil)
        lmax = np.max(lfil)
        imin  = 0
        imax = 0

        lcen[ii] = np.sum(lfil*ffil)/np.sum(ffil)

        lamS,spec = l0, f0 #Two columns with wavelength and flux density
        lamF,filt = lfil, ffil #Two columns with wavelength and response in the range [0,1]
        filt_int   = np.interp(lamS,lamF,filt)  #Interpolate Filter to common(spectra) wavelength axis
        filtSpec = filt_int * spec #Calculate throughput
        wht        = 1. #/(er1[con_rf])**2

        if len(lamS)>0: #./3*len(x0[con_org]): # Can be affect results.
            I1  = simps(spec/lamS**2*c*filt_int*lamS,lamS)   #Denominator for Fnu
            I2  = simps(filt_int/lamS,lamS)                  #Numerator
            fnu[ii] = I1/I2/c         #Average flux density
        else:
            I1  = 0
            I2  = 0
            fnu[ii] = 0

    return lcen, fnu

# Convolution of templates with filter response curves.
'''
def filconv(f00, l00, ffil, lfil): # f0 in fnu
    con00 = (l00>3000) & (l00<14000) # For U to J band calculation.
    f0 = f00[con00]
    l0 = l00[con00]

    lmin = np.min(lfil)
    lmax = np.max(lfil)
    imin = 0
    imax = 0

    lamS,spec = l0, f0 #Two columns with wavelength and flux density for filter.
    lamF,filt = lfil, ffil #Two columns with wavelength and response in the range [0,1]
    filt_int  = np.interp(lamS,lamF,filt)  #Interpolate Filter to common(spectra) wavelength axis
    filtSpec  = filt_int * spec #Calculate throughput
    wht       = 1. #/(er1[con_rf])**2

    if len(lamS)>0: #./3*len(x0[con_org]): # Can be affect results.
        I1  = simps(spec/lamS**2*c*filt_int*lamS,lamS)   #Denominator for Fnu
        #I1  = simps(spec*filt_int*lamS,lamS)   #Denominator for Flambda
        I2  = simps(filt_int/lamS,lamS)                  #Numerator
        fnu = I1/I2/c         #Average flux density
    else:
        I1  = 0
        I2  = 0
        fnu = 0

    if fnu>0:
        return fnu
    else:
        return 1e-99
'''

def filconv(band0, l0, f0, DIR): # f0 in fnu
    #home = os.path.expanduser('~')
    #DIR  = home + '/Dropbox/FILT/'
    fnu  = np.zeros(len(band0), dtype='float32')
    lcen = np.zeros(len(band0), dtype='float32')
    fwhm = np.zeros(len(band0), dtype='float32')

    for ii in range(len(band0)):
        fd = np.loadtxt(DIR + band0[ii] + '.fil', comments='#')
        lfil = fd[:,1]
        ffil = fd[:,2]

        lmin  = np.min(lfil)
        lmax  = np.max(lfil)
        imin  = 0
        imax  = 0

        lcen[ii] = np.sum(lfil*ffil)/np.sum(ffil)
        lamS,spec = l0, f0                     # Two columns with wavelength and flux density
        lamF,filt = lfil, ffil                 # Two columns with wavelength and response in the range [0,1]
        filt_int  = np.interp(lamS,lamF,filt)  # Interpolate Filter to common(spectra) wavelength axis
        wht       = 1. #/(er1[con_rf])**2

        if len(lamS)>0: #./3*len(x0[con_org]): # Can be affect results.
            I1  = simps(spec/lamS**2*c*filt_int*lamS,lamS)   #Denominator for Fnu
            I2  = simps(filt_int/lamS,lamS)                  #Numerator
            fnu[ii] = I1/I2/c         #Average flux density
        else:
            I1  = 0
            I2  = 0
            fnu[ii] = 0

    return lcen, fnu

def fil_fwhm(band0, DIR): # f0 in fnu
    #
    # FWHM
    #
    fwhm = np.zeros(len(band0), dtype='float32')
    for ii in range(len(band0)):
        fd = np.loadtxt(DIR + band0[ii] + '.fil', comments='#')
        lfil = fd[:,1]
        ffil = fd[:,2]

        fsum = np.sum(ffil)
        fcum = np.zeros(len(ffil), dtype='float32')
        lam0,lam1 = 0,0

        for jj in range(len(ffil)):
            fcum[jj] = np.sum(ffil[:jj])/fsum
            if lam0 == 0 and fcum[jj]>0.05:
                lam0 = lfil[jj]
            if lam1 == 0 and fcum[jj]>0.95:
                lam1 = lfil[jj]

        fwhm[ii] = lam1 - lam0

    return fwhm

def calc_Dn4(x0, y0, z0):
    con1 = (x0/(1+z0)>3750) & (x0/(1+z0)<3950)
    con2 = (x0/(1+z0)>4050) & (x0/(1+z0)<4250)
    D41  = np.average(y0[con1])
    D42  = np.average(y0[con2])
    if D41>0 and D42>0:
        D4 = D42/D41
        return D4
    else:
        return -99


def gaus(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))


def detect_line(xcont, ycont, wycont, zgal):
    ################
    # Line library
    ################
    LN = ['Mg2', 'Ne5', 'O2', 'Htheta', 'Heta', 'Ne3', 'Hdelta', 'Hgamma', 'Hbeta', 'O3', 'Halpha', 'S2L', 'S2H']
    LW = [2800, 3347, 3727, 3799, 3836, 3869, 4102, 4341, 4861, 4983, 6563, 6717, 6731]
    fLW = np.zeros(len(LW), dtype='int') # flag.

    R_grs = 50
    dw  = 5
    ldw = 5
    dlw = R_grs * 2
    lsig = 3 # significance of lines.
    er   = 1./np.sqrt(wycont)

    con = (xcont<20000)
    z = np.polyfit(xcont[con], ycont[con], 5, w=wycont[con])
    p = np.poly1d(z)
    ypoly = p(xcont)

    wht2   = wycont
    flag_l = 0

    for ii in range(len(xcont)):
        if ycont[ii] > er[ii]*lsig + ypoly[ii] and ycont[ii-1] > er[ii-1]*lsig + ypoly[ii-1] and ycont[ii+1] > er[ii+1]*lsig + ypoly[ii+1] and wht2[ii]:
            for jj in range(len(LW)):
                if xcont[ii]/(1.+zgal) > LW[jj] -  dw and xcont[ii]/(1.+zgal) < LW[jj] +  dw:
                    wht2[ii-dw:ii+dw] *= 0
                    flag_l  = 1
                    fLW[jj] = 1

    return wht2,flag_l


def check_line_cz(ycont,xcont,wycont,model,zgal):
    er   = 1./np.sqrt(wycont)
    try:
        wht2, flag_l = detect_line(xcont, ycont, wycont, zgal)
        if flag_l == 1:
            wycont = wht2
            wht2, flag_l = detect_line(xcont, ycont, wycont,zgal)

    except Exception:
        #print('Error in Line Check.')
        wht2 = wycont
        pass

    z     = np.polyfit(xcont, ycont, 5, w=wht2)
    p     = np.poly1d(z)
    ypoly = p(xcont)

    return wht2, ypoly


def check_line_cz_man(ycont,xcont,wycont,model,zgal,LW=LW0):
    er   = 1./np.sqrt(wycont)
    try:
        wht2, flag_l = detect_line_man(xcont, ycont, wycont, zgal, LW, model)
    except Exception:
        print('Error in Line Check.')
        wht2 = wycont
        pass

    z     = np.polyfit(xcont, ycont, 5, w=wht2)
    p     = np.poly1d(z)
    ypoly = p(xcont)

    return wht2, ypoly

def detect_line_man(xcont, ycont, wycont, zgal, LW, model):
    ################
    # Line library
    ################
    #LN = ['Mg2', 'Ne5', 'O2', 'Htheta', 'Heta', 'Ne3', 'Hdelta', 'Hgamma', 'Hbeta', 'O3', 'Halpha', 'S2L', 'S2H']
    #LW = [2800, 3347, 3727, 3799, 3836, 3869, 4102, 4341, 4861, 4983, 6563, 6717, 6731]
    fLW = np.zeros(len(LW), dtype='int') # flag.

    #R_grs = 45
    #R_grs = 23.0
    R_grs = (xcont[1] - xcont[0])
    dw   = 1
    lsig = 1.5 # significance of lines.
    er   = 1./np.sqrt(wycont)

    con   = (xcont<20000)
    z     = np.polyfit(xcont[con], ycont[con], 5, w=wycont[con])
    p     = np.poly1d(z)
    ypoly = p(xcont)

    wht2   = wycont
    flag_l = 0

    for ii in range(len(xcont)):
        if 1 > 0:
            for jj in range(len(LW)):
                if xcont[ii]/(1.+zgal) > LW[jj] - dw*R_grs and xcont[ii]/(1.+zgal) < LW[jj] + dw*R_grs:
                    wht2[int(ii-dw):int(ii+dw)] *= 0
                    flag_l  = 1

    return wht2,flag_l


def check_line_man(data,xcont,wht,model,zgal,LW=LW0,lsig=1.5):
    #
    # lsig (float): which sigma to detect lines.
    #

    ################
    # Line library
    ################
    #LN = ['Mg2', 'Ne5', 'O2', 'Htheta', 'Heta', 'Ne3', 'Hdelta', 'Hgamma', 'Hbeta', 'O3', 'Halpha', 'S2L', 'S2H']
    #LW = [2800, 3347, 3727, 3799, 3836, 3869, 4102, 4341, 4861, 4983, 6563, 6717, 6731]
    fLW = np.zeros(len(LW), dtype='int') # flag.
    R_grs = (xcont[1] - xcont[0])
    dw    = 1.
    er    = 1./np.sqrt(wht)
    wht2   = wht
    flag_l = 0

    for ii in range(len(xcont)):
        for jj in range(len(LW)):
            if LW[jj]>0:
                if xcont[ii]/(1.+zgal) > LW[jj] - dw*R_grs and xcont[ii]/(1.+zgal) < LW[jj] + dw*R_grs:
                    wht2[int(ii-dw):int(ii+dw)] *= 0
                    flag_l  = 1
    return wht2
