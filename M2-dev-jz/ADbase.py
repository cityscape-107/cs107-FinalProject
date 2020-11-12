import numpy as np
import math
import pdb
import matplotlib.pyplot as plt

class Var:
    def __init__(self, value):
        self.value = value
        self.children = []
        self.grad_value = None

    def grad(self):
        if self.grad_value is None:
            self.grad_value = sum(weight * var.grad()
                                  for weight, var in self.children)
        return self.grad_value

    #overloading the '+' operator
    def __add__(self, other):
        try: 
            z = Var(self.value + other.value)
            self.children.append((1.0, z))
            other.children.append((1.0, z))
            return z
        except:
            z = Var(self.value + other)
            self.children.append((1, z))
            return z
    
    def __radd__(self, other):
        return self.__add__(other)
    
    #overloading the '-' operator
    def __sub__(self, other):
        try: 
            z = Var(self.value - other.value)
            self.children.append((1.0, z))
            other.children.append((-1.0, z))
            return z
        except:
            z = Var(self.value - other)
            self.children.append((1, z))
            return z
    
    def __rsub__(self, other):
        return self.__sub__(other)
    
    #overloading the '*' operator
    def __mul__(self, other):
        try: 
            other.value
        except:
            z = Var(self.value * other)
            self.children.append((self.value, z))
            return z

        z = Var(self.value * other.value)
        self.children.append((other.value, z))
        other.children.append((self.value, z))
        return z
    
    def __rmul__(self, other):
        return self.__mul__(other)


def sin(x):
    z = Var(math.sin(x.value))
    x.children.append((math.cos(x.value), z))
    return z

def ln(x):
    z = Var(math.log(x.value))
    x.children.append((1/z, z))
    return z

def pow(x, n):
    if (n == 0):
        z = Var(1)
        x.children.append(0, z)
    else:
        z = Var(math.pow(x.value, n))
        x.children.append((n*math.pow(x.value, n-1), z))
    return z


x = Var(0.5)
y = Var(4.2)
z = x * y + pow(x, 2)
z.grad_value = 1.0

print('value of x*y + sin(x) evaluated at x=0.5, y=4.2: {}\nforward pass of our implementation: {}'.format(0.5 * 4.2 + 0.5**2, z.value))