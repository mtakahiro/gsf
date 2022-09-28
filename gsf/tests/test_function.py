import numpy as np
from gsf.function import SFH_dec


def test_SFH():
    t0 = 1.0
    tau = 0.1
    A = 1.0
    SFH = SFH_dec(t0, tau, A, tt=None, minsfr=1e-10)

    assert np.max(SFH) == A
