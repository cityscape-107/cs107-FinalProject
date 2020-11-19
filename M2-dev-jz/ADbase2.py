import numpy as np
import math
import pdb
import matplotlib.pyplot as plt

class Var:
    def __init__(self, value, der=[1]):
        self.val = value
        self.der = der
        

    def grad(self):
        if self.grad_value is None:
            self.grad_value = sum(weight * var.grad()
                                  for weight, var in self.children)
        return self.grad_value


    #overloading the '+' operator
    def __add__(self, other):
        try:
            value = Var(self.val + other.val)
            derivative = self.der + other.der

        except AttributeError: # the other variable does not have any 
            value = Var(self.value + other)
            der = self.der

        return Var(value, derivative)

    def __radd__(self, other):
        new_var = Var(self.val, self.der) # create a new variable 
        return new_var.__add__(other)


    def __neg__(self):
        try:
            val = -self.val
            der = -self.der
        except: # for some reason, der is None
            val = -self.val
            der = None
        return Val(val, der)


    #overloading the '-' operator
    def __sub__(self, other):
        return (-other).__add__(self) # the add function already defines a new variable


    def __rsub__(self, other):
        return other.__sub__(self)  # equivalent to -self.__add__(other), but more convenient regarding the commutativity of inputs

    
    #overloading the '*' operator
    def __mul__(self, other):
        try:
            new_value = self.val * other.val
            new_der = self.der*other.val + self.val*other.der
        except AttributeError: # one of the coefficients of other is None, it is a constant
            new_value = self.val*other
            new_der = self.der*other
        return Var(new_value, new_der)

    def __rmul__(self, other):
        return Var(self.val, self.der).__mul__(other)

    
    def __truediv__(self, other):
        try:  # other is an instance of the Var class
            if other.val == 0:
                raise ZeroDivisionError
            new_val = self.val/other.val
            new_der = (self.der*other.val - self.val*other.der)/self.val**2
        except AttributeError: # other is not an instance of the Var class
            if other == 0:
                raise ZeroDivisionError
            new_val = self.val/other
            new_der = self.der/other
        return Var(new_val, new_der)

    
    def __rtruediv__(self, other):
        try:
            if self.val == 0:
                raise ZeroDivisionError
            new_val = other.val/self.val
            new_der = -other.val*self.der/self.val**2
        except AttributeError:
            if self==0:
                raise ZeroDivisionError
            new_val = other.val/self
            new_der = None
        return Var(new_val, new_der)

    
    def __pow__(self, n):
        # we need to deal when: n is an int or n is a Var instance
        try:
            float(n) # n is an int/float
            value = self.val
            if (value < 0 and 0 < n < 1) or (value==0 and n<1):
                raise ValueError('Illegal value and exponent')
            new_val = value**n
            new_der = n*self.der*value**(n-1)
        except: #type of error to be defined
            # n is a Var
            value = self.val
            if (value < 0 and 0 < other.val < 1) or (value == 0 and other.val < 1):
                raise ValueError('Illegal value and exponent')
            new_val = value**other.val
            new_der = other.der*math.log(self.val)*self.val**other.val + other.val*self.der*self.val**(other.val-1) 
        return Var(new_val, new_der)

    def __rpow__(self, n):
        # how to handle negative values ? We cannot compute -2**(2.5), but we can -2**2 and self should be a function? 
        # in this case, n is an integer (otherwise, we are in the case __mul__)
        if n < 0:
            raise ValueError('negative values are not currently supported')
        elif n == 0:
            if self.val < 0:
                raise ValueError('Division by Zero')
            val = 1*(self.val == 0)
            der = 0
        else:
            val = n**self.val
            der = n ** self.val * np.log(n) * self.der
        return Var(val, der)


        
