#Importing the package
from brainstorm import Forward, constant,power,ln,sin

#Importing dependencies
import math


def test_forward_init():
    print("Case: z = x**2 + sin(x) + ln(x)")
    print("Value=",0.5)
    #Initiating the Forward mode
    x = Forward(0.5)
    #Function example
    z = x**2 + sin(x) + ln(x)
    print('z.grad()=',z.grad())
    print("_________________________")

test_forward_init()

def f1(x):
    return x**2

def f2(x):
    return x**2 + sin(x) + ln(x)

def test_forward_fn():
    print("Case: f2(x)=z")
    print("Value=",0.5)
    x = Forward(0.5)
    z1=f2(x)
    print('z1.grad()=',z1.grad())
    print("_________________________")

test_forward_fn()

def NRFM(f,xn,tol):


    #Initialising
    x=Forward(xn)
    y=f(x)
    xn=x.value-y.value/y.derivative


    while abs(x.value-xn)>tol:


        x=Forward(xn)
        y=f(x)
        xn=x.value-y.value/y.derivative

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
    print('f2(x)=',f2(Forward(trial)).value)
    print("_________________________")

test_f2()
