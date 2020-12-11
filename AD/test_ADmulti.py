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
    print(z)
    v = x - z
    print(v)


def test_add_constant():
    x = AD(1, 1, 'x')
    z = x + 2
    assert z.val == [3]
    np.testing.assert_array_equal(z.der, np.array([1]).reshape(1, -1))
    assert z.name == ['x']


#HERE
def test_add_constant_to_vec():
    x = AD(1, 1, 'x')
    y = AD(3, 1, 'y')
    w = AD([x + y, y - x])
    with pytest.raises(TypeError):
        w + 2
    #print(z)
    #assert z.val == [3]
    #np.testing.assert_array_equal(z.der, np.array([1]).reshape(1, -1))
    #assert z.name == ['x']






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


def test_order():
    x = AD(1, 1, 'x')
    y = AD(2, 2, 'y')
    z = y + x
    z.sort(['x', 'y'])
    print(z)


# Subtraction

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


def test_rsub_c():
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


def test_mul_c1():  # todo: same
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

"""
        except AttributeError:  # one of the coefficients of other is None, it is a constant
            if isinstance(other, np.ndarray):
                if len(other.shape) > 1:
                    new_value = np.dot(self.val, other)
                    derivative = np.dot(self.der, other)
                    name = self.name
                    return AD(new_value, derivative, name)
            new_value = self.val * other
            derivative = self.der * other
            name = self.name
        return AD(new_value, derivative, name)


def test_mul_vec():
    x = AD(np.array([1,2]), np.array([1,2]), 'x')
    a=np.array([[1,2],[3,4]])
    w = x*a
    assert w.name == ['x']
    assert w.val == np.dot(x.val,a)
    assert w.der == np.dot(x.der,a)
"""



# Rmul

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
    n = AD(0, 1, 'n')
    with pytest.raises(ZeroDivisionError):
        x ** n


def test_multi_dim():
    x = AD(1, 4, 'x')
    y = AD(2, 3, 'y')
    z = AD(1, 7, 'z')
    v = AD([x * y, y + z, z + x])
    print(v)


def test_multi_dim_2():
    x = AD(1, 4, 'x')
    y = AD(2, 3, 'y')
    z = AD(1, 7, 'z')
    print('1 x value is', x)
    print(x * y)
    print(y + z)
    print(z + x)
    print('y value is', y)
    print('2 x value is', x)
    print(x + y)
    v = AD([x * y, y + z, z + x, 4, x + y, x + y])
    print(v)


# Operations

# lt
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
    with pytest.raises(AttributeError):
        w < z


def test_lt_different_dim2():
    x = AD(1, 1, 'x')
    y = AD(3, 1, 'y')
    w = AD([x + y, y - x])
    with pytest.raises(AttributeError):
        w < 7


def test_lt_values_w_const():
    x = AD(2, 1, 'x')
    assert (x < 3) == True
    assert (x < 1) == False


def test_lt_equal():
    x = AD(1, 1, 'x')
    y = AD(1, 1, 'y')
    assert (x < y) == False


# gt

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
    with pytest.raises(AttributeError):
        w > z


def test_gt_equal():
    x = AD(1, 1, 'x')
    y = AD(1, 1, 'y')
    assert (x > y) == False


def test_gt_different_dim2():
    x = AD(1, 1, 'x')
    y = AD(3, 1, 'y')
    w = AD([x + y, y - x])
    with pytest.raises(AttributeError):
        w > 7


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
    with pytest.raises(AttributeError):
        w <= z


def test_le_different_dim2():
    x = AD(1, 1, 'x')
    y = AD(3, 1, 'y')
    w = AD([x + y, y - x])
    with pytest.raises(AttributeError):
        w <= 7


def test_le_values_w_const():
    x = AD(2, 1, 'x')
    assert (x <= 3) == True
    assert (x <= 2) == True
    assert (x <= 1) == False


# ge
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
    with pytest.raises(AttributeError):
        w >= z


def test_ge_different_dim2():
    x = AD(1, 1, 'x')
    y = AD(3, 1, 'y')
    w = AD([x + y, y - x])
    with pytest.raises(AttributeError):
        w >= 7


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
    with pytest.raises(AttributeError):
        w == z


def test_eq_different_dim2():
    x = AD(1, 1, 'x')
    y = AD(3, 1, 'y')
    w = AD([x + y, y - x])
    with pytest.raises(AttributeError):
        w == 7


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
    # print(y!=x)
    assert (y != x) == True
    assert (x != y) == True
    assert (z != y) == False


def test_ne_different_dim():
    x = AD(1, 1, 'x')
    y = AD(3, 1, 'y')
    z = AD(4, 1, 'z')
    w = x + y
    with pytest.raises(AttributeError):
        w != z


def test_ne_different_dim2():
    x = AD(1, 1, 'x')
    y = AD(3, 1, 'y')
    w = AD([x + y, y - x])
    with pytest.raises(AttributeError):
        w != 7


def test_ne_values_w_const():
    x = AD(2, 1, 'x')
    assert (x != 1) == True
    assert (x != 2) == False
    assert (x != 3) == True


# Functions
# Testing exp

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


# Testing sine
def test_sin():
    x = AD(0.5, 1, 'x')
    z = x.sin()
    assert z.val == [np.sin(0.5)]
    assert z.der == [np.cos(0.5)]


# Testing cosine
def test_cos():
    x = AD(0.5, 1, 'x')
    z = x.cos()
    assert z.val == [np.cos(0.5)]
    assert z.der == [-np.sin(0.5)]


# Testing tan
def test_tan():
    x = AD(0.5, 1, 'x')
    z = x.tan()
    assert z.val == [np.tan(0.5)]
    assert z.der == [1 / (np.cos(0.5)) ** 2]


def test_tan_inf():
    x = AD(np.pi / 2, 1, 'x')
    with pytest.raises(ValueError):
        x.tan()


#    def sinh(self): #hyperbolic sin
# d/dx (sinh x) = cosh x
# sinh x = (e^x - e^(-x))/2  range (-inf, inf)
# val = np.multiply(.5, (np.exp(self.val) - np.exp(np.multiply(-1, self.val))))
# val = np.sinh(self.val)
# der = np.cosh(self.val) * self.der
# return AD(val, der, self.name)

def test_sinh():
    x = AD(0.5, 1, 'x')
    z = x.sinh()
    assert z.val == [np.sinh(0.5)]
    assert z.der == [np.cosh(0.5) * (1)]


def test_cosh():
    x = AD(0.5, 1, 'x')
    z = x.cosh()
    print(z)
    assert z.val == [np.cosh(0.5)]
    assert z.der == [np.sinh(0.5) * (1)]


def test_tanh():
    x = AD(0.5, 1, 'x')
    z = x.tanh()
    print(z)
    assert z.val == [np.tanh(0.5)]
    assert z.der == [(1 / np.cosh(0.5) ** 2) * (1)]


"""
        def tanh(self): #hyperbolic tan
        #d/dx (tanh x) = (sech x)^2 = 1/((cosh x)^2)
        #tanh x = (e^x - e^(-x)) / (e^x + e^(-x))       range (-inf, inf)
        val = np.tanh(self.val)
        der = np.multiply((1/np.power(np.cosh(self.val), 2)), self.der)
        return AD(val, der, self.name)
"""

"""
     def arcsin(self):
<<<<<<< HEAD
        if ((self.val <= -1) or (self.val>=1)):
=======
        if ((self.val <= -1) or (self.val>=1)):
>>>>>>> 047d296ced50bbfd965aff22291a23cda76c6b63
            raise ValueError("Cannot take derivative of arcsin of value outside of range (-1, 1)")
        val = np.arcsin(self.val)
        der = (1/(np.sqrt(1 - np.power(self.val, 2)))) * self.der
        return AD(val, der, self.name)
    def arccos(self):
        if ((self.val <= -1) or (self.val>=1)):
            raise ValueError("Cannot take derivative of arcsin of value outside of range (-1, 1)")
        val = np.arccos(self.val)
        der = -(1/(np.sqrt(1 - np.power(self.val, 2)))) * self.der
        return AD(val, der, self.name)
    def arctan(self):
        val = np.arctan(self.val)
        der = (1/(1 + np.power(self.val, 2))) * self.der
        return AD(val, der, self.name)
    def logistic(self):
       #assuming logistic function = sigmoid function = 1/(1+e^(-x))
        val = 1/(1 + np.exp(-self.val))
        der = np.multiply(val, (1-val))
        return AD(val, der, self.name)
"""


def test_arcsin():
    x = AD(0.5, 3, 'x')
    z = x.arcsin()
    print('x=', x)
    print(z)
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
    print('x=', x)
    print(z)
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
    print(z)
    assert z.val == [np.arctan(0.5)]
    assert z.der == [3 * (1 + 0.5 ** 2) ** (-1)]


def test_logistic():
    x = AD(2, 3, 'x')
    z = x.logistic()
    assert z.val == [1 / (1 + np.exp(-2))]
    assert z.der == [3 * np.exp(-2) * (1 + np.exp(-2)) ** (-2)]


#Sort testing

"""
>>> x = AD(2,1,'x')
		>>> y = AD(3,1,'y')
		>>> z = AD(4,1,'z')
		>>> f = AD([5*x+4*y+3*z, x*y*z])
		>>> print(f)
		Numerical Value is:
		[34.0, 24.0],
		Jacobian is:
		[[5.0, 3.0, 4.0], [12.0, 6.0, 8.0]],
		Name is:
		['x', 'z', 'y']

		>>> f.sort(['x', 'y', 'z'])
		>>> print(f)
		Numerical Value is:
		[34.0, 24.0],
		Jacobian is:
		[[5.0, 4.0, 3.0], [12.0, 8.0, 6.0]],
		Name is:
		['x', 'y', 'z']

"""

def test_sort_type_err():
    x = AD(-2, 1, 'x')
    y = AD(1, 7, 'y')
    z=x+y
    with pytest.raises(TypeError):
        z.sort(1)

def test_sort_type_err2():
    x = AD(-2, 1, 'x')
    y = AD(1, 7, 'y')
    z=x+y
    with pytest.raises(TypeError):
        z.sort([1])

def test_sort_same():
    x = AD(-2, 1, 'x')
    y = AD(1, 7, 'y')
    z=x+y
    z.sort(['x','y'])
    assert z.name==['x','y']


def test_rpow_1():
    x=AD(2,3,'x')
    z=2**x
    assert z.val==4
    assert z.der==4*np.log(2)*3

def test_rpow_2():
    x=AD(1,1,'x')
    y=AD(2,3,'z')
    z=[x,y]
    with pytest.raises(TypeError):
        y=2**z

def test_rpow_3():
    x=AD(1,1,'x')
    with pytest.raises(ValueError):
        y=(-2)**x

def test_rpow_4():
    x=AD(-1,1,'x')
    with pytest.raises(ZeroDivisionError):
        y=(0)**x

def test_sqrt():
    x=AD(0,1,'x')
    with pytest.raises(ValueError):
        y=x.sqrt()

def test_sqrt1():
    x=AD(-1,1,'x')
    with pytest.raises(ValueError):
        y=x.sqrt()    

def test_sqrt2():
    x=AD(10,3,'x')
    y=x.sqrt()
    assert y.val==np.sqrt(10)
    assert y.der==0.5*3*10**(-0.5)
