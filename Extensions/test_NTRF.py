from NTRF import NTRF
import pytest
from AD.ADmulti import AD
import numpy as np

def test_NTRF():
    init_vals={'x1':1,'x2':2,'x3':3}
    assert len(NTRF(init_vals))==3

def test_NTRF_base():
    f = lambda x: x**2
    root_finder = NTRF(f, order=['x1'], tol=1e-5, iters=1000)
    print(root_finder.find_roots(initial_point=AD(10, 1, 'x1')))

def test_NTRF2():
    x1 = AD(1, 1, 'x1')
    x2 = AD(2, 1, 'x2')
    x3 = AD(3, 1, 'x3')
    f = x1 * 2 + x2 + x3 / 2
    g = x1 + x2 + x3
    h = x3
    v = AD([f, g, h])



"""
if __name__ == '__main__':
    init_vals={'x1':1,'x2':2,'x3':3}
    # for k,v in init_vals.items():
    #     globals()[k]=AD(v,1,k)
    
    def _function(init_vals):
        for k,v in init_vals.items():
            globals()[k]=AD(v,1,k)
            
    
        f = x1*2 + x2 + x3/2
        g = x1 + x2 + x3
        h = x1*0 + x2*0 + x3
        v = AD([f,g,h])
        
        return v
    
    #print(NTRF(init_vals))
"""
