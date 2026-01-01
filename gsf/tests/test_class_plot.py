import numpy as np
import matplotlib
matplotlib.use('Agg')
from gsf.class_plot import PLOT


class MockMBMinimal:
    def __init__(self):
        self.age = np.array([0.5, 1.0, 2.0])
        self.has_ZFIX = False
        self.ZEVOL = True
        self.inputs = {'NIMF': 0}
        self.ID = 'test'
        # minimal attributes used elsewhere if needed


def test_define_axis_sfh_returns_axes():
    mb = MockMBMinimal()
    p = PLOT(mb, f_silence=True)
    axes = p.define_axis_sfh(f_log_sfh=True, skip_zhist=True)
    assert isinstance(axes, dict)
    assert 'ax1' in axes and 'ax2' in axes
    assert axes['ax1'] is not None and axes['ax2'] is not None


def test_sfr_tau_exponential_and_normalization():
    tt = np.linspace(0, 5, 51)
    t0 = 2.0
    tau0 = 0.5
    Mtot = 5.0
    tt_out, yy, yyms = PLOT.sfr_tau(t0, tau0, sfh=1, tt=tt, Mtot=Mtot)
    assert np.allclose(tt_out, tt)
    assert yy.shape == tt.shape
    assert yyms.shape == tt.shape
    # normalization: max of cumulative mass should equal Mtot
    assert np.isclose(np.nanmax(yyms), Mtot)


def test_sfr_tau_lognormal_behavior():
    tt = np.linspace(0.01, 10, 100)
    t0 = 1.0
    tau0 = 0.3
    Mtot = 2.5
    _, yy, yyms = PLOT.sfr_tau(t0, tau0, sfh=6, tt=tt, Mtot=Mtot)
    assert yy.shape == tt.shape
    assert yyms.shape == tt.shape
    assert np.isclose(np.nanmax(yyms), Mtot)
    # SFR should be non-negative
    assert np.all(yy >= 0)
