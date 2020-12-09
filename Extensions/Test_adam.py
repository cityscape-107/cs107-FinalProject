from Extensions.Adam import *


def test_dimensionality():
    f = lambda x, y: x + y
    print(produce_random_point(f))

def test_dimensionality_2():
    f = lambda x, y: x ** 2 + y ** 2
    print(produce_random_point(f))

def test_3_dimensions():
    f = lambda x, y, z: x ** 2 + y ** 2 + (z - 2) ** 2
    adam = Adam(f, random_restarts=10)
    adam.descent()
    print(adam)

def test_adam_simple():
    f = lambda x: x ** 2 + 2
    print(adam_gd(f)) # should be clse to 0

def test_3_opt():
    f = lambda x, y, z: x ** 2 + y ** 2 + (z - 2) ** 2
    adam = Adam(f, random_restarts=10)
    adam.descent()
    print(adam)
    sgd = sgd(f, random_restarts=10)
    sgd.descent()
    print(sgd)
    rms = RMSProp(f, random_restarts=10)
    rms.descent()
    print(rms)

def test_sa():
    f = lambda x, y, z: x ** 2 + y ** 2 + z ** 2
    adam = Adam(f, random_restarts=10, tuning=True, quadratic_matrix=np.eye(3), verbose=2)
    adam.descent()
    print(adam)
    adam = Adam(f, random_restarts=10, verbose=2)
    adam.descent()
    print(adam)


