#Importing the package
from ADbase2 import AD, constant,power,ln,sin

#Importing dependencies
import math
import numpy as np


def test_forward_init():
    print("Case: z = x**2 + sin(x) + ln(x)")
    print("Value=",0.5)
    #Initiating the Forward mode
    x = AD(0.5)
    #Function example
    z = x**2 + sin(x) + ln(x)
    print('z.der()=',z.der())
    print('z.val()=',z.val())
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


        x=ADxn)
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
