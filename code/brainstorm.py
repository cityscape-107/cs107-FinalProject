import math

class Forward:
    def __init__(self, value, constant=False):
        self.value = value
        self.derivative = 1
        if constant:
          self.derivative = 0

    #overloading the '+' operator
    def __add__(self, other):
        z = Forward(self.value + other.value)
        z.derivative = self.derivative +  other.derivative
        return z
    
    def grad(self):
      return self.derivative

    #overloading the '*' operator
    def __mul__(self, other):
        z = Forward(self.value * other.value)
        z.derivative = self.value*other.derivative + self.derivative*other.value
        return z

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self*other
      
    def __pow__(self, n):
        return power(self, n) 

def constant(c):   # to be done later: check for types in order to not create a constant operation but use a float, how to handle constant types ? 
  z = Forward(c.value)
  z.derivative = 0
  return z

def power(x, n):
  z = Forward(math.pow(x.value,n))
  z.derivative = n*math.pow(x.value, n-1)*x.derivative
  return z

def ln(x):
  z = Forward(math.log(x.value))
  z.derivative = (1/x.value)*x.derivative
  return z

def sin(x):
  z = Forward(math.sin(x.value))
  z.derivative = math.cos(x.value)*x.derivative
  return z
