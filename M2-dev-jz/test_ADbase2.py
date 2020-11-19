#Importing the package
from ADbase2 import AD

#Importing dependencies
import math
import numpy as np
import sys
import pytest

############ Testing for AD #####################

#9-check for non float and no int inputs for vals
<<<<<<< HEAD
#12-check for non float and no int inputs for derivs
#18 - test the printed string is as expected
#29 - check for attribute error if adding a non-ad var
#39 - test for negation as with addition
#49 - subtraction
#52 - check A-B!=B-A but - of the answer
#52 - check 0-A=neg(A)
#60 - attribute error when multiplying by non-ad
=======
def test_val_types():
    with pytest.raises(TypeError):
        x = AD('2',1)
    with pytest.raises(TypeError):
        x = AD(2+2j,1)
#12-check for non float and no int inputs for derivs
def test_der_types():
    with pytest.raises(TypeError):
        x = AD(2,'1')
    with pytest.raises(TypeError):
        x = AD(2,1+1j)
#18 - test the printed string is as expected
def test_repr():
    x = AD(42,43)
    assert repr(x) == 'Numerical Value is:\n{}, \nJacobian is:\n{}'.format([42], [43])

#I don't think we need the following tests.
#They fail because the add/mul functions already handle these cases and don't raise an exception
"""
#29 - check for attribute error if adding a non-ad var
def test_add_non_AD():
    with pytest.raises(AttributeError):
        x = AD(0.3,1) 
        y = x + 7

#60 - attribute error when multiplying by non-ad
def test_mult_non_AD():
    with pytest.raises(AttributeError):
        AD(0.3,1) * 7
"""

#96 - check exponent 
#Leave power testing until we talk with team
#131 - tan value error








#Testing constant input
def test_val_basic():
    x = AD(0.5)
    assert x.val==0.5
    assert x.der==0

#Testing x=val input

def test_val_x():
    x = AD(0.5,1)
    assert x.val==0.5
    assert x.der==1


#Testing sine
def test_sin():
    x = AD(0.5,1)
    z = x.sin()
    assert z.val==np.sin(0.5)
    assert z.der==np.cos(0.5)

#Testing cosine
def test_cos():
    x = AD(0.5,1)
    z = x.cos()
    assert z.val==np.cos(0.5)
    assert z.der==-np.sin(0.5)

#Testing tan
def test_tan():
    x = AD(0.5,1)
    z = x.tan()
    assert z.val==np.tan(0.5)
    assert z.der==1/(np.cos(0.5))**2

#Testing power
def test_power():
    x = AD(0.5,1)
    z = x**3
    assert z.val==0.5**3
    assert z.der==3*(0.5**2)

#Testing log
def test_log():
    x = AD(0.5,1)
    z = x.ln()
    assert z.val==np.log(0.5)
    assert z.der==1/0.5

######## Testing Basic operations #######################


# testing addition
def test_add_const():
    x = AD(3,1)
    z = 3*x + 2
    assert 11 == z.val
    assert 3 == z.der

def test_add_const_rev():
    x = AD(3,1)
    z = 3*x + 2
    z_r = 2 + 3*x
    assert z.val == z_r.val
    assert z.der == z_r.der

def test_add_vars():
    x = AD(3,1)
    z = x + x
    assert x.val + x.val==z.val
    assert x.der + x.der==z.der
    assert 3 + 3==z.val
    assert 1 + 1==z.der


#testing negation
def test_neg():
    x = AD(0.5,1)
    y = -x
    assert y.val == -x.val
    assert y.der == -x.der
    x = AD(0.5)
    y = -x
    assert y.val == -x.val
    assert y.der == -x.der


#testing subtraction
def test_sub_const():
    x = AD(3,1)
    z = 4*x - 2
    assert 10 == z.val
    assert 4 == z.der

def test_sub_const_rev(): #check A-B = -(B-A)
    x = AD(3,1)
    z = 3*x - 2
    z_r = 2 - 3*x
    assert z.val == - z_r.val
    assert z.der == - z_r.der

def test_sub_vars():
    x = AD(3,1)
    z = 5*x - x
    assert 4*x.val == z.val
    assert 4*x.der == z.der
    assert 15-3==z.val
    assert 5-1==z.der

def test_sub_0(): #check 0-A=neg(A)
    x = AD(3,1)
    z = 0 - x
    x_neg = -x
    assert z.val == x_neg.val
    assert z.der == x_neg.der


#testing multiplication
def test_mul_const():
    x = AD(3,1)
    z = x*4
    assert 12 == z.val
    assert 4 == z.der

def test_mul_const_rev(): # check A*B=B*A
    x = AD(3,1)
    z = x*4
    z_r = 4*x
    assert z.val == z_r.val
    assert z.der == z_r.der

def test_mul_vars():
    x = AD(3,1)
    y = AD(4,1)
    z = x*y
    assert x.val*y.val == z.val
    assert 12 == z.val
    assert x.der*y.val + y.der*x.val == z.der
    assert 7 == z.der

def test_mul_vars():
    x = AD(3,1)
    y = AD(4,1)
    z = x*y
    z_r = y*x
    assert z.val == z_r.val
    assert z.der == z_r.der


#testing division 
def test_div_const():
    x = AD(12,1)
    z = x/4
    assert 3 == z.val
    assert 1/4 == z.der

def test_dev_const_rev():
    x = AD(3,1)
    z = 12/x
    assert z.val == 4
    assert z.der == -12/9

def test_div_vars():
    x = AD(12,1)
    y = AD(4,1)
    z = x/y
    assert x.val/y.val == z.val
    assert 3 == z.val
    assert (x.der*y.val - y.der*x.val)/(y.val**2) == z.der
    print('z der=',z.der)
    assert (4-12)/(4**2) == z.der

def test_mul_vars():
    x = AD(3,1)
    y = AD(4,1)
    z = x*y
    z_r = y*x
    assert z.val == z_r.val
    assert z.der == z_r.der


#74 - 0 div attribute error
#74 - 0/other =0


######## Testing combining multiple functions ##########


def test_two_vars():
    x = AD(3,1)
    z = 2*x + 3*x
    assert 2*x.val+3*x.val==z.val
    assert 2*x.der+3*x.der==z.der
    assert 2*3+3*3==z.val
    assert 2*1+3*1==z.der

def test_two_powers():
    x = AD(3,1)
    x1=x**5
    x2=x**3
    z = x1+x2
    assert x1.val+x2.val==z.val
    assert x1.der+x2.der==z.der
    assert (3)**5    +(3)**3    ==z.val
    assert 5*(3**4)  +3*(3**2)  ==z.der





#Log edge cases

def test_log_0():
    x = AD(0,1)
    with pytest.raises(ValueError):
        z = x.ln()

def test_log_neg():
    x = AD(-0.3,1)
    with pytest.raises(ValueError):
        z = x.ln()

""" def test_der_results():
    x = AD(0.5,1)
    z = x**2 + sin(x) + ln(x)
    assert z.der==(2*0.5+np.cos(0.5)+1/0.5)

def test_der_results2():
    def f2(x):
        return x**2 + sin(x) + ln(x)
    x = AD(0.5)
    z=f2(x)
    assert z.der==(2*0.5+np.cos(0.5)+1/0.5)

def test_AD_types():
    with pytest.raises(TypeError):
        AD("name")



def test_val_result():
    x = AD(0.5,1)
    z = x**2 + sin(x) + ln(x)
    assert z.val==(0.5*0.5+np.sin(0.5)+np.log(0.5))

def test_val_result2():
    def f2(x):
        return x**2 + sin(x) + ln(x)
    x = AD(0.5,1)
    z=f2(x)
    assert z.val==(0.5*0.5+np.sin(0.5)+np.log(0.5))



def test_forward_init():
    print("Case: z = x**2 + sin(x) + ln(x)")
    print("Value=",0.5)
    #Initiating the Forward mode
    x = AD(0.5)
    #Function example
    z = x**2 + x.sin() + x.ln()
    print('z.der()=',z.der())
    print('z.val()=',z.val())
    if ((0.5*0.5+np.sin(0.5)+np.log(0.5))-(z.val))<(0.000001):
        print("Pass")
    else:
        print("Fail")

    if ((2*0.5+np.cos(0.5)+1/0.5)-(z.der))<(0.000001):
        print("Pass")
    else:
        print("Fail")
    print("_________________________")

test_forward_init()

def f1(x):
    return x**2

def f2(x):
    return x**2 + x.sin() + x.ln()

def test_forward_fn():
    print("Case: f2(x)=z")
    print("Value=",0.5)
    x = AD(0.5)
    z1=f2(x)
    print('z1.der()=',z1.der())
    print("_________________________")

test_forward_fn()



def NRFM(f,xn,tol):
    #Initialising
    x=AD(xn)
    y=f(x)
    xn=x.val-y.val/y.der


    while abs(x.val-xn)>tol:


        x=AD(xn)
        y=f(x)
        xn=x.val-y.val/y.der

    return xn

def test_f1():
    print("f=x**2")
    x0=0.5
    tol=0.001
    print("x_0:{}, tol:{}".format(x0,tol))
    trial=NRFM(f1,x0,tol)
    print('x=',trial)
    print('f1(x)=',f1(trial))
    print("_________________________")

test_f1()

def test_f2():
    print("f2=x**2 + sin(x) + ln(x)")
    x0=0.05
    tol=0.001
    print("x_0:{}, tol:{}".format(x0,tol))
    trial=NRFM(f2,x0,tol)
    print('x_final=',trial)
    print('f2(x)=',f2(Forward(trial)).val)
    print("_________________________")

test_f2()
"""
