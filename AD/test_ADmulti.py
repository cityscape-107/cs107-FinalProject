from ADmulti import AD
import numpy as np
import pytest


# Testing inputs
def test_invalid_der():
    with pytest.raises(TypeError):
        x = AD(0, 'hello', 'x')


def test_der_array():
    x = AD(1, np.array([1]), 'x')
    assert x.val == [1]
    assert x.der == [1]


def test_der_list():
    x = AD(1, [1], 'x')
    assert x.val == [1]
    assert x.der == [1]


def test_invalid_val():
    with pytest.raises(TypeError):
        x = AD('hello', 1, 'x')


def test_names():
    x = AD(1, 1, 'x1')
    z = 2 * x
    v = x - z


def test_add_constant():
    x = AD(1, 1, 'x')
    z = x + 2
    assert z.val == [3]
    np.testing.assert_array_equal(z.der, np.array([1]).reshape(1, -1))
    assert z.name == ['x']


def test_add_constant_to_vec():
    x = AD(1, 1, 'x')
    y = AD(3, 1, 'y')
    w = AD([x + y, y - x]) + 2
    np.testing.assert_array_equal(w.val, np.array([6, 4]).reshape(2, 1))


def test_radd_constant():
    x = AD(1, 1, 'x')
    z = 2 + x
    assert z.val == [3]
    np.testing.assert_array_equal(z.der, np.array([1]).reshape(1, -1))
    assert z.name == ['x']


def test_add():
    x = AD(1, 1, 'x')
    y = AD(2, 2, 'y')
    z = x + y
    assert z.val == [3]
    np.testing.assert_array_equal(z.der, np.array([1, 2]).reshape(1, -1))
    assert z.name == ['x', 'y']


def test_add_c():
    x = AD(1, 1, 'x')
    y = AD(2, 2, 'y')
    z = x + y + 4
    assert z.val == [7]
    np.testing.assert_array_equal(z.der, np.array([1, 2]).reshape(1, -1))
    assert z.name == ['x', 'y']


def test_add_2_vec():
    x = AD(1, 1, 'x')
    y = AD(2, 2, 'y')
    w = AD([x + y, y - x])
    z = np.array([1.0,2.0]).reshape(2,1)
    q= w + z
    np.testing.assert_array_equal(q.val, np.array([[4.0], [3.0]]))


def test_add_vec_str():
    x = AD(1, 1, 'x')
    y = AD(2, 2, 'y')
    w = AD([x + y, y - x])
    z = np.array(['x',2.0]).reshape(2,1)
    with pytest.raises(TypeError):
        w+z


def test_add_2_vec_diff_dim():
    x = AD(1, 1, 'x')
    y = AD(2, 2, 'y')
    w = AD([x + y, y - x])
    z = np.array([1.0,2.0,3.0])
    with pytest.raises(ValueError):
        w + z


def test_add_2_vec_str():
    x = AD(1, 1, 'x')
    y = AD(2, 2, 'y')
    w = AD([x + y, y - x])
    z = np.array(['a','b'])
    with pytest.raises(ValueError):
        q = w + z

def test_add_list():
    x = AD(1, 1, 'x')
    y = AD(2, 2, 'y')
    w = AD([x + y, y - x])
    z = [1,2]
    with pytest.raises(ValueError):
        w + z

def test_order():
    x = AD(1, 1, 'x')
    y = AD(2, 2, 'y')
    z = y + x
    z.sort(['x', 'y'])



def test_sub():
    x = AD(1, 1, 'x')
    y = AD(2, 2, 'y')
    z = y - x
    assert z.val == [1]
    np.testing.assert_array_equal(z.der, np.array([2, -1]).reshape(1, -1))


def test_sub_c():
    y = AD(2, 2, 'y')
    z = y - 1
    assert z.val == [1]
    assert z.der == [2]


def test_rsub():
    x = AD(1, 1, 'x')
    y = AD(2, 2, 'y')
    z = -(y - x)
    assert z.val == [-1]
    np.testing.assert_array_equal(z.der, np.array([-2, 1]).reshape(1, -1))


def test_rsub_c():
    y = AD(2, 2, 'y')
    z = -(y - 1)
    assert z.val == [-1]
    assert z.der == [-2]


def test_rsub_d():
    y = AD(2, 2, 'y')
    z = -2 - y
    assert z.val == [-4]
    assert z.der == [-2]


def test_mul():
    x = AD(1, 1, 'x')
    y = AD(2, 2, 'y')
    z = x * y
    assert z.val == [2]
    np.testing.assert_array_equal(z.der, np.array([2, 2]).reshape(1, -1))


def test_mul_array_str():
    x = AD(1, 1, 'x')
    y=np.array(['1'])
    with pytest.raises(TypeError):
        x * y


def test_mul_c1():
    x = AD(1, 1, 'x')
    y = AD(2, 2, 'y')
    z = x * y + 4
    assert z.val == [6]
    np.testing.assert_array_equal(z.der, np.array([2, 2]).reshape(1, -1))
    assert z.name == ['x', 'y']


def test_mul_c2():
    x = AD(1, 1, 'x')
    y = AD(2, 2, 'y')
    z = x * y * 4
    assert z.val == [8]
    np.testing.assert_array_equal(z.der, np.array([8, 8]).reshape(1, -1))
    assert z.name == ['x', 'y']


def test_mul_c2_addx():
    x = AD(1, 1, 'x')
    y = AD(2, 2, 'y')
    z = x * y * 4 + x
    assert z.val == [9]
    np.testing.assert_array_equal(z.der, np.array([9, 8]).reshape(1, -1))
    assert z.name == ['x', 'y']


def test_mul_last():
    x = AD(1, 4, 'x')
    y = AD(2, 7, 'y')
    w = x * y
    z = x + y
    v = z * w
    assert v.name == ['x', 'y']
    assert v.val == [6]

def test_rmul():
    x = AD(1, 1, 'x')
    z = 2 * x
    assert z.val == [2]
    assert z.der == [2]


def test_div():
    x = AD(1, 4, 'x')
    y = AD(2, 7, 'y')
    z = x / y
    assert z.name == ['x', 'y']
    assert z.val == [0.5]
    np.testing.assert_array_equal(z.der, np.array([2, -7 / 4]).reshape(1, -1))


def test_div_c():
    x = AD(3, 12, 'x')
    z = x / 3
    assert z.val == [1]
    assert z.der == [4]




def test_div_hard():  # we should cover the lines in rtruediv here
    x = AD(1, 4, 'x')
    y = AD(2, 7, 'y')
    z = x * y
    w = x + y
    u = z / w
    assert u.name == ['x', 'y']
    assert u.val == 2 / 3
    np.testing.assert_array_equal(u.der, np.array([16 / 9, 7 / 9]).reshape(1, -1))


def test_div_3_var():
    x = AD(1, 4, 'x')
    y = AD(2, 7, 'y')
    z = AD(3, 2, 'z')
    w = x * y * z
    assert w.name == ['x', 'y', 'z']
    assert w.val == [6]
    np.testing.assert_array_equal(w.der, np.array([24, 21, 4]).reshape(1, -1))


def test_true_div():
    x = AD(1, 3, 'x')
    z = 1 / x
    assert z.name == ['x']
    assert z.val == [1]
    np.testing.assert_array_equal(z.der, np.array([-3]).reshape(1, -1))


def test_rtrue_div():
    x = AD(1, 4, 'x')
    y = AD(2, 7, 'y')
    z = 2 / (x + y)
    assert z.name == ['x', 'y']
    assert z.val == [2 / 3]
    np.testing.assert_array_equal(z.der, np.array([-8 / 9, -14 / 9]).reshape(1, -1))


# Division errors

def test_true_div_zero():
    x = AD(0, 3, 'x')
    with pytest.raises(ZeroDivisionError):
        1 / x

def test_true_div_zero_array():
    x = AD(0, 3, 'x')
    y = AD(0, 3, 'y')
    z = AD([x,y])
    with pytest.raises(ZeroDivisionError):
        z / z


def test_true_div_zero():
    x = AD(5, 3, 'x')
    with pytest.raises(ZeroDivisionError):
        x / 0

def test_true_div_string():
    x = AD(5, 3, 'x')
    with pytest.raises(TypeError):
        x / '0'


def test_rtrue_div_string():
    x = AD(5, 3, 'x')
    with pytest.raises(TypeError):
        'o' / x


def test_rtrue_div_zero_array():
    x = AD(1, 3, 'x')
    y = AD(0, 3, 'y')
    z = AD([x,y])
    with pytest.raises(ZeroDivisionError):
        5 / z

# Power
def test_pow():
    x = AD(1, 4, 'x')
    z = x ** 3
    assert z.name == ['x']
    assert z.val == [1]
    np.testing.assert_array_equal(z.der, np.array([12]).reshape(1, -1))


def test_pow_1():
    x = AD(3, 4, 'x')
    y = AD(2, 7, 'y')
    z = x ** (y + x)
    assert z.name == ['x', 'y']
    assert z.val == [3 ** 5]
    np.testing.assert_array_equal(z.der,
                                  np.array([(4 * np.log(3) + 20 / 3) * 3 ** 5, 7 * np.log(3) * 3 ** 5]).reshape(1, -1))


def test_pow_2():
    x = AD(1, 4, 'x')
    y = AD(2, 7, 'y')
    z = x ** y
    assert z.name == ['x', 'y']
    assert z.val == [1]
    np.testing.assert_array_equal(z.der, np.array([8, 0]).reshape(1, -1))


# Power errors
def test_pow_0_to_neg():
    x = AD(0, 1, 'x')
    with pytest.raises(ZeroDivisionError):
        z = x ** (-5)

def test_pow_0_to_1():
    x = AD(0, 1, 'x')
    with pytest.raises(ZeroDivisionError):
        x ** (1)


def test_neg_pow_0_to_1():
    x = AD(-1, 1, 'x')
    with pytest.raises(ValueError):
        z = x ** 0.5


def test_fn_pow_neg():
    x = AD(-1, 1, 'x')
    n = AD(0.5, 1, 'n')
    with pytest.raises(ValueError):
        x ** n


def test_fn_power_0():
    x = AD(0, 1, 'x')
    with pytest.raises(ZeroDivisionError):
        z = x ** 0

def test_fn_power_0_base_neg_n():
    x = AD(-1, -1, 'x')
    y = AD(0, 1, 'y')
    with pytest.raises(ZeroDivisionError):
        y**x

def test_fn_power_str():
    x = AD(1, 1, 'x')
    with pytest.raises(TypeError):
        x**'st'

def test_fn_rpower_str():
    x = AD(1, 1, 'x')
    y = 's'
    with pytest.raises(TypeError):
        y**x

def test_fn_power_0AD():
    x = AD(0, 1, 'x')
    n = AD(0, 1, 'n')

    with pytest.raises(ZeroDivisionError):
        z = x ** n

def test_fn_power_array():
    x = AD(0, 1, 'x')
    y = AD(0, 1, 'x')
    z = AD([x,y])
    n = AD(0, 1, 'n')

    with pytest.raises(TypeError):
        w = z ** n


def test_fn_power_array2():
    x = AD(0, 1, 'x')
    y = AD(0, 1, 'x')
    z = AD([x,y])
    n = AD(0, 1, 'n')

    with pytest.raises(TypeError):
        w = z ** n



def test_multi_dim():
    x = AD(1, 4, 'x')
    y = AD(2, 3, 'y')
    z = AD(1, 7, 'z')
    v = AD([x * y, y + z, z + x])


def test_multi_dim_2():
    x = AD(1, 4, 'x')
    y = AD(2, 3, 'y')
    z = AD(1, 7, 'z')
    v = AD([x * y, y + z, z + x, 4, x + y, x + y])


# Operations

# lt
def test_lt_values_str():
    x = AD(7, 1, 'x')
    with pytest.raises(TypeError):
       x < 'a'

def test_lt_values_shape():
    x = AD(7, 1, 'x')
    y = AD(2, 1, 'y')
    z = AD([x,y])
    with pytest.raises(AttributeError):
       x < z

def test_lt_values():
    x = AD(2, 1, 'x')
    y = AD(3, 1, 'y')
    assert (x < y) == True
    assert (y < x) == False


def test_lt_different_dim():
    x = AD(1, 1, 'x')
    y = AD(3, 1, 'y')
    z = AD(4, 1, 'z')
    w = x + y
    assert (w < z) == False


def test_lt_different_dim2():
    x = AD(1, 1, 'x')
    y = AD(3, 1, 'y')
    w = AD([x + y, y - x])
    assert (w < 7) == True


def test_lt_values_w_const():
    x = AD(2, 1, 'x')
    assert (x < 3) == True
    assert (x < 1) == False


def test_lt_equal():
    x = AD(1, 1, 'x')
    y = AD(1, 1, 'y')
    assert (x < y) == False


# gt

def test_gt_values_str():
    x = AD(7, 1, 'x')
    with pytest.raises(TypeError):
       x > 'a'


def test_gt_values():
    x = AD(7, 1, 'x')
    y = AD(3, 1, 'y')
    assert (x > y) == True
    assert (y > x) == False


def test_gt_different_dim():
    x = AD(1, 1, 'x')
    y = AD(3, 1, 'y')
    z = AD(4, 1, 'z')
    w = x + y
    assert (w > z) == False


def test_gt_equal():
    x = AD(1, 1, 'x')
    y = AD(1, 1, 'y')
    assert (x > y) == False


def test_gt_different_dim2():
    x = AD(1, 1, 'x')
    y = AD(3, 1, 'y')
    w = AD([x + y, y - x])
    assert (w > 7) == False


def test_gt_values_w_const():
    x = AD(7, 1, 'x')
    assert (x > 3) == True
    assert (x > 11) == False


# le
def test_le_values():
    x = AD(2, 1, 'x')
    y = AD(3, 1, 'y')
    z = AD(3, 1, 'z')
    assert (x <= y) == True
    assert (y <= x) == False
    assert (z <= y) == True


def test_le_different_dim():
    x = AD(1, 1, 'x')
    y = AD(3, 1, 'y')
    z = AD(4, 1, 'z')
    w = x + y
    assert (w <= z) == True


def test_le_different_dim2():
    x = AD(1, 1, 'x')
    y = AD(3, 1, 'y')
    w = AD([x + y, y - x])
    assert (w <= 7) == True


def test_le_values_w_const():
    x = AD(2, 1, 'x')
    assert (x <= 3) == True
    assert (x <= 2) == True
    assert (x <= 1) == False


def test_ge_values():
    x = AD(2, 1, 'x')
    y = AD(3, 1, 'y')
    z = AD(3, 1, 'z')
    assert (y >= x) == True
    assert (x >= y) == False
    assert (z >= y) == True


def test_ge_different_dim():
    x = AD(1, 1, 'x')
    y = AD(3, 1, 'y')
    z = AD(4, 1, 'z')
    w = x + y
    assert (w >= z) == True


def test_ge_different_dim2():
    x = AD(1, 1, 'x')
    y = AD(3, 1, 'y')
    w = AD([x + y, y - x])
    assert (w >= 7) == False


def test_ge_values_w_const():
    x = AD(2, 1, 'x')
    assert (x >= 1) == True
    assert (x >= 2) == True
    assert (x >= 3) == False


# eq
def test_eq_values():
    x = AD(2, 1, 'x')
    y = AD(3, 1, 'y')
    z = AD(3, 1, 'z')
    assert (y == x) == False
    assert (x == y) == False
    assert (z == y) == True


def test_eq_different_dim():
    x = AD(1, 1, 'x')
    y = AD(3, 1, 'y')
    z = AD(4, 1, 'z')
    w = x + y
    assert (w == z) == True


def test_eq_different_dim2():
    x = AD(1, 1, 'x')
    y = AD(3, 1, 'y')
    w = AD([x + y, y - x])
    assert (w == 7) == False


def test_eq_values_w_const():
    x = AD(2, 1, 'x')
    assert (x == 1) == False
    assert (x == 2) == True
    assert (x == 3) == False


# ne
def test_ne_values():
    x = AD(2, 1, 'x')
    y = AD(3, 1, 'y')
    z = AD(3, 1, 'z')
    assert (y != x) == True
    assert (x != y) == True
    assert (z != y) == False


def test_ne_different_dim():
    x = AD(1, 1, 'x')
    y = AD(3, 1, 'y')
    z = AD(4, 1, 'z')
    w = x + y
    assert (w != z) == False


def test_ne_different_dim2():
    x = AD(1, 1, 'x')
    y = AD(3, 1, 'y')
    w = AD([x + y, y - x])
    assert (w != 7) == True


def test_ne_values_w_const():
    x = AD(2, 1, 'x')
    assert (x != 1) == True
    assert (x != 2) == False
    assert (x != 3) == True


def test_exp():
    x = AD(0.5, 1, 'x')
    z = x.exp()
    assert z.val == [np.exp(0.5)]
    assert z.der == [1 * np.exp(0.5)]


# Testing log
def test_log_0():
    x = AD(0, 1, 'x')
    with pytest.raises(ValueError):
        z = x.ln()


def test_log_neg():
    x = AD(-0.3, 1, 'x')
    with pytest.raises(ValueError):
        z = x.ln()


def test_log():
    x = AD(0.5, 1, 'x')
    z = x.ln()
    assert z.val == [np.log(0.5)]
    assert z.der == [1 / 0.5]



def test_sin():
    x = AD(0.5, 1, 'x')
    z = x.sin()
    assert z.val == [np.sin(0.5)]
    assert z.der == [np.cos(0.5)]


def test_cos():
    x = AD(0.5, 1, 'x')
    z = x.cos()
    assert z.val == [np.cos(0.5)]
    assert z.der == [-np.sin(0.5)]


def test_tan():
    x = AD(0.5, 1, 'x')
    z = x.tan()
    assert z.val == [np.tan(0.5)]
    assert z.der == [1 / (np.cos(0.5)) ** 2]


def test_tan_inf():
    x = AD(np.pi / 2, 1, 'x')
    with pytest.raises(ValueError):
        x.tan()


def test_sinh():
    x = AD(0.5, 1, 'x')
    z = x.sinh()
    assert z.val == [np.sinh(0.5)]
    assert z.der == [np.cosh(0.5) * (1)]


def test_cosh():
    x = AD(0.5, 1, 'x')
    z = x.cosh()
    assert z.val == [np.cosh(0.5)]
    assert z.der == [np.sinh(0.5) * (1)]


def test_tanh():
    x = AD(0.5, 1, 'x')
    z = x.tanh()
    assert z.val == [np.tanh(0.5)]
    assert z.der == [(1 / np.cosh(0.5) ** 2) * (1)]



def test_arcsin():
    x = AD(0.5, 3, 'x')
    z = x.arcsin()
    assert z.val == [np.arcsin(0.5)]
    # np.testing.assert_array_equal(z.der, np.array([-3/np.sqrt(1 - 0.5**2)]))
    assert z.der == [3 * (1 - 0.5 ** 2) ** (-0.5)]


def test_arcsin_val_err_1():
    x = AD(2, 1, 'x')
    with pytest.raises(ValueError):
        x.arcsin()


def test_arcsin_val_err_2():
    x = AD(-2, 1, 'x')
    with pytest.raises(ValueError):
        x.arcsin()


def test_arccos():
    x = AD(0.5, 3, 'x')
    z = x.arccos()
    assert z.val == [np.arccos(0.5)]
    # np.testing.assert_array_equal(z.der, np.array([-3/np.sqrt(1 - 0.5**2)]))
    assert z.der == [-3 * (1 - 0.5 ** 2) ** (-0.5)]


def test_arccos_m1():
    x = AD(-1.1, 3, 'x')
    with pytest.raises(ValueError):
        x.arccos()


def test_arccos_p1():
    x = AD(1.1, 3, 'x')
    with pytest.raises(ValueError):
        x.arccos()


def test_arctan():
    x = AD(0.5, 3, 'x')
    z = x.arctan()
    assert z.val == [np.arctan(0.5)]
    assert z.der == [3 * (1 + 0.5 ** 2) ** (-1)]


def test_logistic():
    x = AD(2, 3, 'x')
    z = x.logistic()
    assert z.val == [1 / (1 + np.exp(-2))]
    assert z.der == [3 * np.exp(-2) * (1 + np.exp(-2)) ** (-2)]




def test_sort_type_err():
    x = AD(-2, 1, 'x')
    y = AD(1, 7, 'y')
    z = x + y
    with pytest.raises(TypeError):
        z.sort(1)


def test_beug():
    x = AD(1, 1, 'x')
    y = AD(2, 1, 'y')
    f1 = AD([10 * x, 10 * y])
    f2 = AD([3 * x, 3 * y])
    z = f1 + f2
    np.testing.assert_array_equal(z.val, np.array([13, 26]).reshape(2, 1))
    np.testing.assert_array_equal(z.sort(order=['x', 'y']).der, np.array([[13, 0], [0, 13]]))


def test_sort_type_err2():
    x = AD(-2, 1, 'x')
    y = AD(1, 7, 'y')
    z = x + y
    with pytest.raises(TypeError):
        z.sort([1])


def test_sort_same():
    x = AD(-2, 1, 'x')
    y = AD(1, 7, 'y')
    z = x + y
    z.sort(['x', 'y'])
    assert z.name == ['x', 'y']


def test_rpow_1():
    x = AD(2, 3, 'x')
    z = 2 ** x
    assert z.val == 4
    assert z.der == 4 * np.log(2) * 3


def test_rpow_2():
    x = AD(1, 1, 'x')
    y = AD(2, 3, 'z')
    z = [x, y]
    with pytest.raises(TypeError):
        y = 2 ** z


def test_rpow_3():
    x = AD(1, 1, 'x')
    with pytest.raises(ValueError):
        y = (-2) ** x

def test_rpow_4():
    x = AD(-1, 1, 'x')
    with pytest.raises(ZeroDivisionError):
        y = (0) ** x

def test_rpow_array():
    x = AD(2, 3, 'x')
    z = np.array([2,2,3,4]) ** x
    assert z[0].val == 4
    assert z[0].der == 4 * np.log(2) * 3


def test_sqrt():
    x = AD(0, 1, 'x')
    with pytest.raises(ZeroDivisionError):
        y = x.sqrt()


def test_sqrt1():
    x = AD(-1, 1, 'x')
    with pytest.raises(ValueError):
        y = x.sqrt()


def test_sqrt2():
    x = AD(10, 3, 'x')
    y = x.sqrt()
    assert y.val == np.sqrt(10)
    assert y.der == 0.5 * 3 * 10 ** (-0.5)





def test_order_paul():
    x = AD(1, 1, 'x')
    y = AD(2, 2, 'y')
    z = x + y
    z.sort(['x', 'y'])
    assert z.name == ['x', 'y']



def test_multi_dim_4():
    x = AD(1, 1, 'x')
    y = AD(2, 2, 'y')
    z = AD([y.cos()+x.cos(), y.tan()+x.tanh()])
    z.sort(['x', 'y'])
    np.testing.assert_allclose(z.val, np.array([0.12415547, -1.42344571]).reshape(2, 1), atol=1e-5)
    np.testing.assert_allclose(z.der, np.array([[-0.84147098, -1.81859485], [0.41997434, 11.54879841]]), atol=1e-5)
    assert z.name == ['x', 'y']



