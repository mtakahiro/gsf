import numpy as np
from gsf.basic_func import Basic, Basic_tau


class MockMB:
    def __init__(self):
        self.Zall = np.array([0.001, 0.01, 0.02, 0.03])
        self.delZ = 0.01
        self.tau = np.array([0.1, 0.5, 1.0])
        self.deltau = 0.1
        self.ageparam = np.array([1.0, 5.0, 10.0])
        self.delage = 1.0


def test_basic_Z2NZ():
    mb = MockMB()
    b = Basic(mb)
    idx = b.Z2NZ(0.02)
    assert idx == 2


def test_basic_tau_Z2NZ():
    mb = MockMB()
    bt = Basic_tau(mb)
    nz, nt, na = bt.Z2NZ(0.01, 0.5, 5.0)
    assert nz == 1
    assert nt == 1
    assert na == 1
