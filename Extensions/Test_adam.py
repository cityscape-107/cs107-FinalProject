from Extensions.Adam import *


def test_dimensionality():
    f = lambda x, y: x + y
    print(produce_random_point(f))

def test_dimensionality_2():
    f = lambda x, y: x ** 2 + y ** 2
    print(produce_random_point(f))

def test_3_dimensions():
    f = lambda x, y, z: x ** 2 + y ** 2 + (z - 2) ** 2
    acc = adam_gd(f, random_restarts=10)
    print(acc)

def test_adam_simple():
    f = lambda x: x ** 2 + 2
    print(adam_gd(f)) # should be clse to 0

