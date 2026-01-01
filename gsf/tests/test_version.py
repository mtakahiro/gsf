from gsf import version


def test_version_present():
    assert hasattr(version, '__version__')
    assert isinstance(version.__version__, str)
    assert hasattr(version, '__version_tuple__')
    assert isinstance(version.__version_tuple__, tuple)
