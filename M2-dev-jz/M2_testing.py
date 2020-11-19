#Importing the package
from ADbase2 import AD

#Importing dependencies
import math
import numpy as np
import sys
import pytest


def test_forward_init():
    print("Case: z = x**2 + sin(x) + ln(x)")
    print("Value=",0.5)
    #Initiating the Forward mode
    x = AD(0.5)
    #Function example
    z = x**2 + sin(x) + ln(x)
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
    return x**2 + sin(x) + ln(x)

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


############ Testing for AD #####################
def test_val__basic():
    x = AD(0.5)
    assert z.der==0

def test_val_result():
    x = AD(0.5)
    z = x**2 + sin(x) + ln(x)
    assert z.val==(0.5*0.5+np.sin(0.5)+np.log(0.5))

def test_val_result2():
    def f2(x):
        return x**2 + sin(x) + ln(x)
    x = AD(0.5)
    z=f2(x)
    assert z.val==(0.5*0.5+np.sin(0.5)+np.log(0.5))

def test_der_basic():
    x = AD(0.5)
    assert z.val==0.5

def test_der_results():
    x = AD(0.5)
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

#Values for extra fns
def test_sin():
    x = AD(0.5)
    z = sin(x)
    assert z.val==np.sin(0.5)

def test_power():
    x = AD(0.5)
    z = x**2
    assert z.val==0.5**2

def test_log():
    x = AD(0.5)
    z = ln(x)
    assert z.val==np.log(0.5)

def test_log_0():
    x = AD(0)
    with pytest.raises(ValueError):
        z = ln(x)


#Derivs for extra fns
def test_sin():
    x = AD(0.5)
    z = sin(x)
    assert z.der==np.cos(0.5)

def test_power():
    x = AD(0.5)
    z = x**2
    assert z.der==2*0.5

def test_log():
    x = AD(0.5)
    z = ln(x)
    assert z.der==1/0.5
