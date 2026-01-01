import copy

def get_nebular_template(wave, flux, sp, esptmp, age, lammin, lammax):
    """"""
    esp = esptmp

    if age>0.01:
        tage_neb = 0.01
        print('Nebular component is calculabed with %.2f Gyr'%tage_neb)
    else:
        tage_neb = age

    ewave0, eflux0 = esp.get_spectrum(tage=tage_neb, peraa=True)
    # print(tage_neb, ewave0.shape, eflux0.shape)
    # import matplotlib.pyplot as plt
    # plt.close()
    # for ii in range(eflux0.shape[0]):
    #     plt.plot(ewave0, eflux0[ii,:])
    # plt.xlim(0,10000)
    # plt.show()
    # hoge

    con = (ewave0>lammin) & (ewave0<lammax)
    if age != tage_neb:
        # sp_tmp = sp.copy()
        sp_tmp = copy.copy(sp)
        wave0_tmp, flux0_tmp = sp_tmp.get_spectrum(tage=tage_neb, peraa=True) # Lsun/AA
        _, flux_tmp = wave0_tmp[con], flux0_tmp[con]
    else:
        _, flux_tmp = wave, flux
        # print(len(ewave0), len(ewave0[con]), len(eflux0), len(flux))

    flux_nebular = eflux0[con] - flux_tmp
    # Eliminate some negatives. Mostly on <912A;
    con_neg = flux_nebular<0
    flux_nebular[con_neg] = 0

    # plt.close()
    # plt.plot(ewave0[con], flux_nebular)
    # plt.xlim(0,10000)
    # plt.show()

    return esp, flux_nebular
