#Importing the package
from brainstorm import Forward, constant,power,ln,sin

#Importing dependencies
import math

#Initiating the Forward mode
x = Forward(0.5)
#Function example
z = x**2 + sin(x) + ln(x)
print(z.grad())

def f(x):
    return x**2+sin(x)+ln(x)

def NRFM(f,xn,tol):

    #Initialising
    x=Forward(xn)
    y=f(x)
    xn=x.value-y.value/y.derivative

    while abs(x.value-xn)<tol:
        x=Forward(xn)
        y=f(x)
        xn=x.value-y.value/y.derivative

    return xn
