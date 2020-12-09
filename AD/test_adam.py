from Extensions.adam import *
import pytest

def test_dimensionality_2():
    f = lambda x, y: x + y
   # print('test_dimensionality_2:', produce_random_point(f).shape[0])
    assert produce_random_point(f).shape[0]==2
test_dimensionality_2()

def test_dimensionality_3():
    f = lambda x, y, z: x ** 2 + y ** 2 + z
    assert produce_random_point(f).shape[0]==3
    #print('test_dimensionality_3:', produce_random_point(f))
test_dimensionality_3()

def test_3_dimensions():
    f = lambda x, y, z: x ** 2 + y ** 2 + (z - 2) ** 2
    acc = adam_gd(f, random_restarts=10)
    #print('test_3_dimensions:', acc) 
    assert len(acc)==3
test_3_dimensions()

def test_tunning_true():
    f = lambda x, y, z: x ** 2 + y ** 2 + (z - 2) ** 2
    acc = adam_gd(f, random_restarts=10, tuning=True)
    #print('test_3_dimensions:', acc) 
    assert acc==0
test_tunning_true()

def test_adam_simple():
    f = lambda x: x ** 2 + 2
    assert np.abs(adam_gd(f)) < 1e-3 
    print('test_adam_simple:', adam_gd(f)) # should be close to 0
test_adam_simple()

