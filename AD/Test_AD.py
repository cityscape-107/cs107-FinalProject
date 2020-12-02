from AD_multi import AD
import numpy as np


def test_add():
    x = AD(1, 1, 'x')
    y = AD(2, 2, 'y')
    z = x+y
    print(z)

def test_add_c():
    x = AD(1, 1, 'x')
    y = AD(2, 2, 'y')
    z = x + y + 4
    print(z)

def test_mul():
    x = AD(1, 1, 'x')
    y = AD(2, 2, 'y')
    z = x*y
    print(z)

def test_mul_c():
    x = AD(1, 1, 'x')
    y = AD(2, 2, 'y')
    z = x*y + 4
    print(z)

def test_mul_c2():
    x = AD(1, 1, 'x')
    y = AD(2, 2, 'y')
    z = x * y * 4
    print(z)

def test_mul_c2_addx():
    x = AD(1, 1, 'x')
    y = AD(2, 2, 'y')
    z = x * y * 4 + x
    print(z)

def test_mul_c2_addx2():
    x = AD(1, 1, 'x')
    y = AD(2, 2, 'y')
    z = x * y * 4
    w = z + x
    print(w)

def test_mul_last():
    x = AD(1, 4, 'x')
    y = AD(2, 7, 'y')
    w = x*y
    z = x+y
    v = z+w
    print(v)

def test_mul_last_last():
    x = AD(1, 4, 'x')
    y = AD(2, 7, 'y')
    w = x * y
    z = x + y
    print(w)
    print(z)
    v = z*w
    print(v)


def test_div():
    x = AD(1, 4, 'x')
    y = AD(2, 7, 'y')
    z = x/y
    print(z)

def test_div_hard():
    x = AD(1, 4, 'x')
    y = AD(2, 7, 'y')
    z = x*y
    print(z)
    w = x+y
    print(w)
    print(z/w)

def test_div_3_var():
    x = AD(1, 4, 'x')
    y = AD(2, 7, 'y')
    z = AD(3, 2, 'z')
    w = x*y*z
    a = x + y + z
    print(w)
    print(a)
    print(w/a)



def test_rtrue_div():
    x = AD(1, 4, 'x')
    y = AD(2, 7, 'y')
    z = x+y
    print(2/z)



