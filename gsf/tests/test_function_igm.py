import numpy as np
from gsf.function_igm import get_XI, get_nH, get_sig_lya


def test_get_XI_bounds():
    assert get_XI(4.0, zend=5, zstart=8) == 0
    assert get_XI(9.0, zend=5, zstart=8) == 1.0


def test_get_XI_linear():
    xi = get_XI(6.5, zend=5, zstart=8)
    assert abs(xi - 0.5) < 1e-8


def test_get_nH_scalar_and_array():
    # scalar-like input should return numeric value
    n = get_nH(2.0)
    expected = 8.5e-5 * ((1.0 + 2.0) / 8.0) ** 3
    assert abs(n - expected) < 1e-12

    # array input returns array-like with same length
    zarr = np.array([2.0, 3.0])
    narr = get_nH(zarr)
    assert narr.shape == zarr.shape


def test_get_sig_lya_shape_positive():
    lam = np.array([1216.0, 1300.0])
    # call scalar-version since get_sig_lya integrates assuming scalar x
    sigs = np.array([get_sig_lya(l, z_s=6.0) for l in lam])
    assert sigs.shape == lam.shape
    assert np.all(sigs > 0)
