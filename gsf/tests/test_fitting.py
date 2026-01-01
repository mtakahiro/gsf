import sys
from gsf import gsf
import argparse
import fsps

def test_fitting(monkeypatch):
    parfile = 'gds_43114.input'
    fplt = 0
    z = 1.9
    delwave = 5.0

    called = {}
    def fake_run_gsf_all(parfile_arg, fplt_arg, nthin=None, delwave=None, zman=None):
        called['args'] = (parfile_arg, fplt_arg, nthin, delwave, zman)

    monkeypatch.setattr(gsf, 'run_gsf_all', fake_run_gsf_all)

    gsf.run_gsf_all(parfile, fplt, nthin=1, delwave=delwave, zman=z)

    assert 'args' in called
