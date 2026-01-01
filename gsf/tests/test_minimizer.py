import numpy as np
from gsf.minimizer import (
    reduce_chisquare,
    reduce_negentropy,
    reduce_cauchylogpdf,
    MinimizerResult,
)


def test_reduce_chisquare():
    r = np.array([1.0, 2.0, -2.0])
    assert reduce_chisquare(r) == 9.0


def test_reduce_metrics_return_float():
    r = np.array([0.1, -0.2, 0.3])
    ne = reduce_negentropy(r)
    cc = reduce_cauchylogpdf(r)
    assert isinstance(ne, float)
    assert isinstance(cc, float)


def test_minimizerresult_calculates_stats():
    mr = MinimizerResult()
    mr.init_vals = [0.0, 0.0]
    mr.residual = np.array([1.0, 2.0, 3.0])
    mr._calculate_statistics()
    assert hasattr(mr, 'chisqr')
    assert mr.ndata == 3
    assert mr.nvarys == 2
    assert mr.nfree == 1
    assert mr.redchi == mr.chisqr / max(1, mr.nfree)
    assert isinstance(mr.aic, float) and isinstance(mr.bic, float)
