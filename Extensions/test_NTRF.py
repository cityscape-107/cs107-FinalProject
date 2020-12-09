from NTRF import NTRF
import pytest

def test_NTRF():
    init_vals={'x1':1,'x2':2,'x3':3}
    assert len(NTRF(init_vals))==3

test_NTRF()