import os
import tempfile
import gsf.gsf as gmod


class DummyMainbody:
    def __init__(self, *args, **kwargs):
        # minimal attributes used early in run_gsf_all
        self.flag_class = False
        self.DIR_TMP = tempfile.mkdtemp(prefix='gsf_test_tmp_')


def test_run_gsf_all_returns_false_when_flag_class_false(monkeypatch):
    # Patch read_input to avoid reading files
    monkeypatch.setattr(gmod, 'read_input', lambda parfile: {})
    # Patch Mainbody constructor
    monkeypatch.setattr(gmod, 'Mainbody', DummyMainbody)

    try:
        res = gmod.run_gsf_all('dummy.par', fplt=0, verbose=True)
        assert res is False
    finally:
        # cleanup temp dir if it exists
        if os.path.isdir(DummyMainbody().DIR_TMP):
            os.rmdir(DummyMainbody().DIR_TMP)
