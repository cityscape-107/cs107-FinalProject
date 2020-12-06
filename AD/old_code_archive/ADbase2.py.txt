import math

import numpy as np


class AD:

    def __init__(self, value, der=0): #changed from der=[0]
        if isinstance(value, float) or isinstance(value, int):
            value = [value]
        elif isinstance(value, np.ndarray):
            value = value  
        else: # NEW
            raise TypeError(f'Value should be int or float and not {type(value)}')

        if isinstance(der, float) or isinstance(der, int):
            der = [der]
        elif isinstance(der, np.ndarray):
            der = der
        else: # NEW
            raise TypeError(f'Derivative should be int or float and not {type(der)}')

        self.val = np.array(value)
        self.der = np.array(der)            

    def __repr__(self):  # todo: pb with __repr__, solved
        val = self.val
        der = self.der
        return 'Numerical Value is:\n{}, \nJacobian is:\n{}'.format(val, der)

    # overloading the '+' operator
    def __add__(self, other):
        try:
            value = self.val + other.val
            derivative = self.der + other.der

        except AttributeError:  # the other variable does not have any
            value = self.val + other
            derivative = self.der
        
        return AD(value, derivative)

    def __radd__(self, other):
        new_var = AD(self.val, self.der)  # create a new variable
        return new_var.__add__(other)

    def __neg__(self):
        try:
            val = -self.val
            der = -self.der
            return AD(val, der)
        except:  # for some reason, der is None
            val = -self.val
            return AD(val)
        
    # overloading the '-' operator
    def __sub__(self, other):
        return self.__add__(-other)  # the add function already defines a new variable

    def __rsub__(self, other):
        return -(self.__sub__(other))

    # overloading the '*' operator
    def __mul__(self, other):
        try:
            new_value = self.val * other.val
            new_der = self.der * other.val + self.val * other.der
        except AttributeError:  # one of the coefficients of other is None, it is a constant
            new_value = self.val * other
            new_der = self.der * other
        return AD(new_value, new_der)

    def __rmul__(self, other):
        return AD(self.val, self.der).__mul__(other)

    def __truediv__(self, other):
        try:  # other is an instance of the AD class
            if other.val == 0 or other.val==np.array([0]):
                raise ZeroDivisionError
            new_val = self.val / other.val
            new_der = (self.der * other.val - self.val * other.der) / other.val ** 2
        except AttributeError:  # other is not an instance of the AD class
            if other == 0:
                raise ZeroDivisionError
            new_val = self.val / other
            new_der = self.der / other
        return AD(new_val, new_der)

    def __rtruediv__(self, other):
        #try:
        if self.val == 0 or self.val==np.array([0]):
  
          raise ZeroDivisionError
        new_val = other / self.val
        new_der = -other * self.der / self.val ** 2
        #except AttributeError:
        #    if self.val == 0:    
        #        raise ZeroDivisionError
        #    new_val = other / self.val
        #    new_der = None
        return AD(new_val, new_der)

    def __pow__(self, n):
        # # we need to deal when: n is an int or n is a AD instance
        if isinstance(n, float) or isinstance(n, int):  # duck typing fails here because of the raised exception
            float(n)  # n is an int/float
            value = self.val
            if value < 0 and 0 < n < 1:
                raise ValueError('Illegal value and exponent')
            if value == 0 and n < 1:
                raise ZeroDivisionError
            new_val = value ** n
            new_der = n * self.der * value ** (n - 1)
        if isinstance(n, AD):  # n is an AD object
            value = self.val
            if (value < 0 and 0 < n.val < 1) or (value == 0 and n.val < 1):
                raise ValueError('Illegal value and exponent')
            new_val = value ** n.val
            new_der = n.der * math.log(self.val) * self.val ** n.val + n.val * self.der * self.val ** (n.val - 1)
        return AD(new_val, new_der)

    def __rpow__(self, n):
        # how to handle negative values ? We cannot compute -2**(2.5), but we can -2**2 and self should be a function?
        # in this case, n is an integer (otherwise, we are in the case __mul__)
        if n < 0:
            raise ValueError('negative values are not currently supported')
        elif n == 0:
            if self.val < 0:
                raise ValueError('Division by Zero')
            val = 1 * (self.val == 0)
            der = 0
        else:
            val = n ** self.val
            der = n ** self.val * math.log(n) * self.der
        return AD(val, der)

    def tan(self):
        nonpoints = map(lambda x: ((x / np.pi) - 0.5) % 1 == 0.00, self.val)
        if any(nonpoints):
            raise ValueError("Math error, Tangent cannot handle i*0.5pi ")
        val = np.tan(self.val)
        #der = np.multiply(np.power(1 / np.cos(self.val), 2), self.der)
        der = np.multiply(1/np.power(np.cos(self.val), 2), self.der)
        return AD(val, der)

    def sin(self):
        val = np.sin(self.val)
        der = np.cos(self.val) * self.der
        return AD(val, der)

    def cos(self):
        val = np.cos(self.val)
        der = -np.sin(self.val) * self.der
        return AD(val, der)

    def exp(self):
        val = np.exp(self.val)
        der = np.multiply(np.exp(self.val), self.der)
        return AD(val, der)


    def ln(self):
        if self.val<=0:
            raise ValueError("Cannot take natural log of zero or negative values")
        val = np.log(self.val)
        der = 1/self.val
        return AD(val, der)

    #add log to other bases
    
    
    
