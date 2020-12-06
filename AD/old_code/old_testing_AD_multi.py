from AD_multi import AD
import numpy as np


def test_add():
    x = AD(1, 1, 'x')
    y = AD(2, 2, 'y')
    z = x+y
    print(z)

def test_add_c():  # todo: why does it change ?
    x = AD(1, 1, 'x')
    y = AD(2, 2, 'y')
    z1 = x+y
    print(z1)
    z2 = z1+4
    print(z2)

def test_mul():
    # x = AD(1, [1, 0])
    # y = AD(2, [0, 5])
    x = AD(1, 1, 'x')
    y = AD(2, 2, 'y')
    z = x*y
    print(z)

def test_mul_c(): # todo: same
    x = AD(1, 1, 'x')
    y = AD(2, 2, 'y')
    z = x*y + 4
    print(z)

def test_mul_c2():  # todo: same
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
    print(w)
    print(z)
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


def test_pow():
    x = AD(1, 4, 'x')
    z = x**3
    print(z)

def test_pow_2():
    x = AD(1, 4, 'x')
    y = AD(2, 7, 'y')
    z = x**y
    print(z)

def test_pow_3():  # checked, is okay
    x = AD(1, 2, 'x')
    y = AD(2, 1, 'y')
    w = x*y
    v = x+y
    print('Product is ', w)
    print('Sum is ', v)
    z = w**v
    print(z)




def test_multi_dim():
    x = AD(1, 4, 'x')
    y = AD(2, 3, 'y')
    z = AD(1, 7, 'z')
    v = AD([x*y, y+z, z+x])
    print(v)


def test_multi_dim_2():
    x = AD(1, 4, 'x')
    y = AD(2, 3, 'y')
    z = AD(1, 7, 'z')
    print('1 x value is', x)
    print(x*y)
    print(y+z)
    print(z+x)
    print('y value is', y)
    print('2 x value is', x)
    print(x+y)
    v = AD([x * y, y + z, z + x, 4, x+y, x+y])
    print(v)





