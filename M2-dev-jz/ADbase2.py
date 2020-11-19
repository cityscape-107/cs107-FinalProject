import numpy as np

class AD:
    def __init__(self, value, der=[1]):
        if isinstance(value, float) or isinstance(value, int):
            value = [value]

        if isinstance(der, float) or isinstance(der, int):
            der = [der]
        self.val = np.array(value)
        self.der = np.array(der)
    

    def __repr__(self):
        return 'Numerical Value is:\n{},\nJacobian is:\n{}'.format(self.val, self.der)
        
    
    #overloading the '+' operator
    def __add__(self, other):
        try:
            value = AD(self.val + other.val)
            derivative = self.der + other.der

        except AttributeError: # the other variable does not have any 
            value = AD(self.value + other)
            der = self.der

        return AD(value, derivative)

    def __radd__(self, other):
        new_var = AD(self.val, self.der) # create a new variable 
        return new_var.__add__(other)


    def __neg__(self):
        try:
            val = -self.val
            der = -self.der
        except: # for some reason, der is None
            val = -self.val
            der = None
        return Var(val, der)


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
        return AD(new_value, new_der)

    def __rmul__(self, other):
        return AD(self.val, self.der).__mul__(other)

    
    def __truediv__(self, other):
        try:  # other is an instance of the AD class
            if other.val == 0:
                raise ZeroDivisionError
            new_val = self.val/other.val
            new_der = (self.der*other.val - self.val*other.der)/self.val**2
        except AttributeError: # other is not an instance of the AD class
            if other == 0:
                raise ZeroDivisionError
            new_val = self.val/other
            new_der = self.der/other
        return AD(new_val, new_der)

    
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
        return AD(new_val, new_der)

    
    def __pow__(self, n):
        # # we need to deal when: n is an int or n is a AD instance
        # try:
        #     float(n) # n is an int/float
        #     value = self.val
        #     if (value < 0 and 0 < n < 1) or (value==0 and n<1):
        #         raise ValueError('Illegal value and exponent')
        #     new_val = value**n
        #     new_der = n*self.der*value**(n-1)
        # except: #type of error to be defined
        #     # n is a AD
        #     value = self.val
        #     if (value < 0 and 0 < other.val < 1) or (value == 0 and other.val < 1):
        #         raise ValueError('Illegal value and exponent')
        #     new_val = value**other.val
        #     new_der = other.der*math.log(self.val)*self.val**other.val + other.val*self.der*self.val**(other.val-1) 
        # return AD(new_val, new_der)
    
    
        value = map(lambda x: x >= 0, self.val)
        if n % 1 != 0 and not all(value):
            raise ValueError("error in pow.")
        elif n < 1 and 0 in self.val:
            raise ValueError("error in pow.")
        val = np.power(self.val, n)
        if len(self.der.shape):
            self_val = np.expand_dims(self.val, 1) if len(self.der.shape) > len(self.val.shape) else self.val
            der = n * np.multiply((self_val ** (n - 1)), self.der)
        else:
            der = None
        return AD(val, der)

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
        return AD(val, der)


        
