import math
import numpy as np


class AD:
    """
    TO DO! 
    """
    def __init__(self, value, der=0, name='x'):  # changed from der=[0]
        """
        
		INPUTS
		========
		value : int, float, list, or np.array. This defines the value of a variable

		der : This allows manually setting derivative (default=0)

		name: stores the name of the variable
        
		EXAMPLES
		========
		# scalar input
		>>> x = AD(1,1,'x') 
        >>> x
		Numerical Value is:
        [[1.]], 
        Jacobian is:
        [[1.]], 
        Name is:
        ['x']

		# vector input
        >>> x = AD(1,1,'x') 
        >>> y = AD(2,1,'y') 
        >>> z = AD(3,1,'z') 
        >>> v = AD([x+y+z])
        >>> v
        Numerical Value is:
        [6.0], 
        Jacobian is:
        [[1.0, 1.0, 1.0]], 
        Name is:
        ['y', 'x', 'z']
		"""
        self.val = None
        self.der = None
        self.name = None
        if isinstance(value, list):
            names = []
            for AD_function in value:
                try:
                    names.append(AD_function.name)
                except AttributeError:
                    continue  # if just list names never has anything appended
            # unique_names = set(np.asarray(names).flatten())  # reference
            unique_names = list(set(np.array(np.concatenate(names, axis=0 )))) # reference
            global_value = []  # vector of values
            global_jacobian = []  # matrix of derivatives, every derivative of the list should be one row
            for AD_function in value:
                try:
                    # print('The value we are appending is ', AD_function.val)
                    global_value.append(AD_function.val[0][0])
                except AttributeError:
                    global_value.append(AD_function)  # constant
            for AD_function in value:
                AD_jacobian = []
                for var in unique_names:
                    try:
                        # print('We are looking for ', var)
                        # print('Inside', AD_function.name)
                        index = AD_function.name.index(var)  # ['x', 'y'] vs ['y', 'x']
                        # print('It is ', index)
                        # print('The derivative of the AD var is ', AD_function.der)
                        AD_jacobian.append(AD_function.der[0][index])
                    except:
                        AD_jacobian.append(0)
                    # print('...', AD_jacobian)
                global_jacobian.append(AD_jacobian)
            if len(unique_names) != 0:
                self.val = global_value
                self.der = global_jacobian
                self.name = unique_names
            else:
                self.val = np.array(value).reshape(len(value), 1)

        if self.val is None:
            if isinstance(value, float) or isinstance(value, int):
                value = np.array(value, dtype=np.float64).reshape(1, -1)
            elif isinstance(value, np.ndarray):
                value = value.reshape(value.shape[0], 1)
            else:
                raise TypeError(f'Value should be int or float and not {type(value)}')

            self.val = np.array(value, dtype=np.float64)

        if self.der is None:
            if isinstance(der, float) or isinstance(der, int):
                der = np.array(der, dtype=np.float64).reshape(1, -1)
            elif isinstance(der, list) and len(der) == 1:
                der = np.array(der).reshape(len(der), 1)
            elif isinstance(der, np.ndarray) and der.shape[0] == 1:
                der = der.reshape(der.shape[0], -1)
            else:
                raise TypeError(f'Derivative should be int or float and not {type(der)}')

            self.der = np.array(der, dtype=np.float64)

        if self.name is None:
            if isinstance(name, list):
                self.name = name
            else:
                self.name = [name]

    def __repr__(self):
        """
        Prints the Numerical Value, Jacobian, and Name.
        
        INPUTS:
        ======== 
        self: AD object
        
        RETURNS:
        ======== 
        x: AD object with value, Jacobian matrix, and corresponding name(s)
		
		
		EXAMPLES
		========
        >>> x = AD(1,1,'x') 
        >>> y = AD(2,1,'y') 
        >>> z = AD(3,1,'z') 
        >>> v = AD([x+y+z])
        >>> print(v)
        Numerical Value is:
        [6.0], 
        Jacobian is:
        [[1.0, 1.0, 1.0]], 
        Name is:
        ['y', 'x', 'z']
		"""
        val = self.val
        der = self.der
        name = self.name
        return 'Numerical Value is:\n{}, \nJacobian is:\n{}, \nName is:\n{}'.format(val, der, name)

    def __add__(self, other):
        """ 
		Computes the addition of self and other.
		
		INPUTS
		========
		self: AD object
		other: AD object or number
		
		RETURNS
		========
		x: AD object that equals to the sum of self and other
		
		EXAMPLES
		======== 
		>>> x = AD(1,1,'x') + AD(2,1,'x')
		>>> print(x)
		Numerical Value is:
        [[3.]], 
        Jacobian is:
        [[1.]], 
        Name is:
        ['x']

		>>> x = AD(1,1,'x') + 2
		>>> print(x)
        Numerical Value is:
        [[3.]], 
        Jacobian is:
        [[1.]], 
        Name is:
        ['x']

		"""
        try:
            names_1 = self.name.copy()
            names_2 = other.name
            value = self.val + other.val
            derivative = self.der.copy()
            for i, name_2 in enumerate(names_2):
                if name_2 in names_1:
                    index_1 = names_1.index(name_2)
                    derivative[:, index_1] = derivative[:, index_1] + other.der[:, i]
                else:
                    derivative = np.concatenate((derivative, other.der[:, i].reshape(-1, 1)), axis=1)
                    names_1.append(name_2)
            name = names_1
        except AttributeError:
            value = self.val + other
            derivative = self.der
            name = self.name
        return AD(value, derivative, name)

    def __radd__(self, other):
        """
		Computes the addition of other and self.

		INPUTS
		========
		self: AD object
		other: AD object or number
		
		RETURNS
		========
		x: AD object that has a value equals to the sum of other and self
		
		EXAMPLES
		======== 
		>>> x = 2 + AD(1,1,'x') 
		>>> print(x)
        Numerical Value is:
        [[3.]], 
        Jacobian is:
        [[1.]], 
        Name is:
        ['x']

		"""
        new_var = AD(self.val, self.der, self.name)  # create a new variable
        return new_var.__add__(other)

    def __neg__(self):
        """ 
		Computes the negation of self

		INPUTS
		==========
		self: AD object
		
		RETURNS
		======= 
		y: AD object with the negative value & derivative of x

		EXAMPLES
		======== 
		>>> x = AD(2,1,'x')
		>>> print(-x)
		Numerical Value is:
        [[-2.]], 
        Jacobian is:
        [[-1.]], 
        Name is:
        ['x']
		"""
        val = -self.val.copy()
        der = -self.der.copy()
        name = self.name
        return AD(val, der, name)

        # overloading the '-' operator

    def __sub__(self, other):
        """
		Computes self subtracts other

		INPUTS
		==========
		self: AD object
		other: AD object or number
		
		RETURNS
		========
		z: Var object that has the value of self subtracts other
		
		EXAMPLES
		======== 
        >>> x = AD(1,1,'x') 
        >>> y = AD(2,1,'y') 
        >>> print(x-y)
        Numerical Value is:
        [[-1.]], 
        Jacobian is:
        [[ 1. -1.]], 
        Name is:
        ['x', 'y']
        
        >>> print(x-2)
        Numerical Value is:
        [[-1.]], 
        Jacobian is:
        [[1.]], 
        Name is:
        ['x']
		"""
        return self.__add__(-other)

    def __rsub__(self, other):
        """
		Computes self subtracts other

		INPUTS
		==========
		self: AD object
		other: AD object or number
		
		RETURNS
		========
		z: Var object that has the value of other subtracts self
		
		EXAMPLES
		======== 
        >>> x = AD(1,1,'x') 
        >>> y = AD(2,1,'y')
        >>> print(y-x)        
        Numerical Value is:
        [[1.]], 
        Jacobian is:
        [[ 1. -1.]], 
        Name is:
        ['y', 'x']
        
        >>> x = AD(1,1,'x') 
        >>> print(2-x)
        Numerical Value is:
        [[1.]], 
        Jacobian is:
        [[-1.]], 
        Name is:
        ['x']
                
		"""
        return -(self.__sub__(other))

    # overloading the '*' operator
    def __mul__(self, other):
        """ 
		Computes the product of self and other.
		
		INPUTS
		==========
		self: AD object
		other: AD object or number
		
		RETURNS
		========
		z: Var object that has the value and derivative corresponding to self multiplies other
		
		EXAMPLES
		========= 
        >>> x = AD(1,1,'x') 
        >>> y = AD(2,1,'y') 
        >>> print(x*y)
        Numerical Value is:
        [[2.]], 
        Jacobian is:
        [[2. 1.]], 
        Name is:
        ['x', 'y']

        >>> x = AD(1,1,'x') 
        >>> print(x*2)
        Numerical Value is:
        [[2.]], 
        Jacobian is:
        [[2.]], 
        Name is:
        ['x']
        """
        try:
            names_1 = self.name.copy()
            names_2 = other.name.copy()
            new_value = self.val.copy() * other.val.copy()  # numpy array multiplication
            new_names = names_1 + [name for name in other.name if name not in names_1]
            derivative = self.der.copy()
            for name in new_names:
                # print('Current name', name)
                # print('List of names for 1',names_1)
                # print('List of names for 2', names_2)
                if name in names_1 and name in names_2:
                    index_1 = names_1.index(name)
                    index_2 = names_2.index(name)
                    new_der = self.val.copy() * other.der[:, index_2] + self.der[:, index_1] * other.val
                    derivative[:, index_1] = new_der
                if name in names_1 and name not in names_2:
                    # print('I am here')
                    index_1 = names_1.index(name)
                    new_der = self.der[:, index_1] * other.val
                    # print(new_der)
                    derivative[:, index_1] = new_der
                    # print(derivative)
                if name in names_2 and name not in names_1:
                    # print('Now I am here')
                    index_2 = names_2.index(name)
                    new_der = self.val * other.der[:, index_2]
                    # print(new_der)
                    # print('Before the update', derivative)
                    derivative = np.concatenate((derivative, new_der), axis=1)
            name = new_names

        except AttributeError:  # one of the coefficients of other is None, it is a constant
            if isinstance(other, np.ndarray):
                if len(other.shape) > 1:
                    new_value = np.dot(self.val, other)
                    derivative = np.dot(self.der, other)
                    name = self.name
                    return AD(new_value, derivative, name)
            new_value = self.val * other
            derivative = self.der * other
            name = self.name
        return AD(new_value, derivative, name)

    def __rmul__(self, other):
        """ 
		Computes the product of other and self.
		
		INPUTS
		==========
		self: AD object
		other: AD object or number
		
		RETURNS
		========
		z: Var object that has the value and derivative corresponding to other multiplies self
		
		EXAMPLES
		========= 
        >>> x = AD(1,1,'x') 
        >>> y = AD(2,1,'y') 
        >>> print(y*x)
        Numerical Value is:
        [[2.]], 
        Jacobian is:
        [[1. 2.]], 
        Name is:
        ['y', 'x']

        >>> x = AD(1,1,'x') 
        >>> print(2*x)
        Numerical Value is:
        [[2.]], 
        Jacobian is:
        [[2.]], 
        Name is:
        ['x']
		"""
        return AD(self.val, self.der, self.name).__mul__(other)

    def __truediv__(self, other):  # todo: check for forbidden values
		""" 
		Computes self devided by other
		
		INPUTS
		==========
		self: AD object
		other: AD object or number
		
		RETURNS
		========
		z: 
		
		
		EXAMPLES
		========  
        """
        try:
            names_1 = self.name
            names_2 = other.name
            new_value = self.val / other.val  # numpy array multiplication
            new_names = names_1 + [name for name in other.name if
                                   name not in names_1]  # we should raise an error when dividing by 0
            derivative = self.der.copy()
            for name in new_names:
                if name in names_1 and name in names_2:
                    index_1 = names_1.index(name)
                    index_2 = names_2.index(name)
                    new_der = (self.der[:, index_1] * other.val - other.der[:,
                                                                  index_2] * self.val) / other.val ** 2  # with scalars
                    derivative[:, index_1] = new_der
                if name in names_1 and name not in names_2:
                    # print('I am here')
                    index_1 = names_1.index(name)
                    new_der = self.der[:, index_1] / other.val
                    derivative[:, index_1] = new_der
                if name in names_2 and name not in names_1:
                    index_2 = names_2.index(name)
                    new_der = -self.val * other.der[:, index_2] / other.val ** 2
                    derivative = np.concatenate((derivative, new_der), axis=1)
            name = new_names
        except AttributeError:
            new_value = self.val / other
            derivative = self.der / other
            name = self.name
        return AD(new_value, derivative, name)

    def __rtruediv__(self, other):

        if self.val == 0 or self.val == np.array([0]):
            raise ZeroDivisionError
        new_val = other / self.val
        new_der = -other * self.der / self.val ** 2
        names = self.name
        return AD(new_val, new_der, names)

    def __pow__(self, n):

        if isinstance(n, float) or isinstance(n, int):  # duck typing fails here because of the raised exception
            float(n)  # n is an int/float
            value = self.val.copy()
            derivative = self.der.copy()
            names = self.name
            if value < 0 and 0 < n < 1:
                raise ValueError('Illegal value and exponent')
            if value == 0 and n < 1:
                raise ZeroDivisionError
            new_val = value ** n
            for i, name in enumerate(names):
                derivative[:, i] = n * self.der[:, i] * value ** (n - 1)
            return AD(new_val, derivative, names)

        if isinstance(n, AD):  # n is an AD object
            value_base = self.val
            value_exponent = n.val
            if value_base < 0 and 0 < value_exponent < 1:
                raise ValueError('Illegal value and exponent')
            if value_base == 0 and value_exponent < 1:
                raise ZeroDivisionError
            new_val = value_base ** value_exponent
            names_1 = self.name
            names_2 = n.name
            new_names = names_1 + [name for name in names_2 if name not in names_1]
            derivative = self.der.copy()
            for i, name in enumerate(new_names):
                if name in names_1 and name in names_2:
                    index_2 = names_2.index(name)
                    new_der = (n.der[:, index_2] * math.log(value_base) + value_exponent * self.der[:,
                                                                                           i] / value_base) * new_val
                    derivative[:, i] = new_der
                if name in names_1 and name not in names_2:
                    index_1 = names_1.index(name)
                    new_der = self.der[:, index_1] * value_exponent * value_base ** (value_exponent - 1)
                    derivative[:, index_1] = new_der
                if name in names_2 and name not in names_1:
                    index_2 = names_2.index(name)
                    new_der = n.der[:, index_2] * math.log(value_base) * new_val
                    derivative = np.concatenate((derivative, new_der), axis=1)
            return AD(new_val, derivative, new_names)


    def sort(self, order):
        if not isinstance(order, list) and not isinstance(order, np.ndarray):
            raise TypeError('Order should be an array-like composed of strings')
        for string in order:
            if not isinstance(string, str):
                raise TypeError('Order should only be composed of strings')
        if self.name == order:
            return
        final_derivative = self.der.copy()
        for i, variable in enumerate(order):
            index = self.name.index(variable)
            derivative = self.der[:, index]
            final_derivative[:, i] = derivative
        self.der = final_derivative
        self.name = order

    def __lt__(self, other):

        if isinstance(other, AD):

            # Error if object don't have same dim.
            if len(self.name) != len(other.name):
                raise AttributeError('Incoherent dimension input')

            # < acts on scalars and vector
            if self.val < other.val:
                return True
            else:
                return False

            # else:  # need to compare the derivatives

            # else:
            # for i, name in enumerate(self.name):
            # if name in other.name:
            # index_2 = other.name.index(name)
            # der_1 = self.der[:, i]
            # der_2 = other.der[:, index_2]
            # if der_1 < der_2:
            #   return True
            # elif der_1 > der_2:
            #    return False
            # else:
            #    continue
            # else:
            # raise AttributeError('Incoherent dimension input')
            # return False
        else:
            if len(self.name) == 1:
                return self.val < other
            else:
                raise AttributeError('Incoherent dimension input')

    def __gt__(self, other):

        if isinstance(other, AD):
            return other.__lt__(self)
        else:
            if len(self.name) == 1:
                return self.val > other
            else:
                raise AttributeError('Incoherent dimension input')

    def __le__(self, other):

        if isinstance(other, AD):

            # Error if object don't have same dim.
            if len(self.name) != len(other.name):
                raise AttributeError('Incoherent dimension input')

            # < acts on scalars and vector
            if self.val <= other.val:
                return True
            else:
                return False
        else:
            if len(self.name) == 1:
                return self.val <= other
            else:
                raise AttributeError('Incoherent dimension input')

    def __ge__(self, other):

        # raises an error when the two objects do not have the same attributes
        if isinstance(other, AD):
            return other.__le__(self)
        else:
            if len(self.name) == 1:
                return self.val >= other
            else:
                raise AttributeError('Incoherent dimension input')

    def __eq__(self, other):

        if isinstance(other, AD):

            # Error if object don't have same dim.
            if len(self.name) != len(other.name):
                raise AttributeError('Incoherent dimension input')

            # < acts on scalars and vector
            if self.val == other.val:
                return True
            else:
                return False
        else:
            if len(self.name) == 1:
                return self.val == other
            else:
                raise AttributeError('Incoherent dimension input')

    def __ne__(self, other):

        return not self.__eq__(other)

    def tan(self):
        nonpoints = map(lambda x: ((x / np.pi) - 0.5) % 1 == 0.00, self.val)
        if any(nonpoints):
            raise ValueError("Math error, Tangent cannot handle i*0.5pi ")
        val = np.tan(self.val)
        # der = np.multiply(np.power(1 / np.cos(self.val), 2), self.der)
        der = np.multiply(1 / np.power(np.cos(self.val), 2), self.der)
        return AD(val, der, self.name)

    def sin(self):
        val = np.sin(self.val)
        der = np.cos(self.val) * self.der
        return AD(val, der, self.name)

    def cos(self):
        val = np.cos(self.val)
        der = -np.sin(self.val) * self.der
        return AD(val, der, self.name)

    def exp(self):
        val = np.exp(self.val)
        der = np.multiply(np.exp(self.val), self.der)
        return AD(val, der, self.name)

    def ln(self):
        if self.val <= 0:
            raise ValueError("Cannot take natural log of zero or negative values")
        val = np.log(self.val)
        der = 1 / self.val
        return AD(val, der, self.name)

    def sinh(self): #hyperbolic sin
        #d/dx (sinh x) = cosh x
        #sinh x = (e^x - e^(-x))/2  range (-inf, inf)
        #val = np.multiply(.5, (np.exp(self.val) - np.exp(np.multiply(-1, self.val))))
        val = np.sinh(self.val)
        der = np.cosh(self.val) * self.der 
        return AD(val, der, self.name)

    def cosh(self): #hyperbolic cos
        #d/dx (cosh x) = sinh x
        #cosh x = (e^x + e^(-x))/2  range (-inf, inf)
        #val = np.multiply(.5, (np.exp(self.val) + np.exp(np.multiply(-1, self.val))))
        val = np.cosh(self.val)
        der = np.sinh(self.val) * self.der 
        return AD(val, der, self.name)

    def tanh(self): #hyperbolic tan
        #d/dx (tanh x) = (sech x)^2 = 1/((cosh x)^2)
        #tanh x = (e^x - e^(-x)) / (e^x + e^(-x))       range (-inf, inf)
        val = np.tanh(self.val)
        der = (1/np.power(np.cosh(self.val), 2))* self.der
        return AD(val, der, self.name)

    def arcsin(self):
        if ((self.val <= -1) or (self.val>=1)): 
            raise ValueError("Cannot take derivative of arcsin of value outside of range (-1, 1)")
        val = np.arcsin(self.val)
        #der = (1/(np.sqrt(1 - np.power(self.val, 2)))) * self.der
        der = self.der*((1 - self.val**2)**(-0.5))
        return AD(val, der, self.name)

    def arccos(self):
        if ((self.val <= -1) or (self.val>=1)): 
            raise ValueError("Cannot take derivative of arcsin of value outside of range (-1, 1)")
        val = np.arccos(self.val)
        #der = -(1/(np.sqrt(1 - np.power(self.val, 2)))) * self.der
        der = -self.der*((1 - self.val**2)**(-0.5))
        return AD(val, der, self.name)

    def arctan(self):
        val = np.arctan(self.val)
        der = self.der*(1 + self.val**2)**(-1)
        return AD(val, der, self.name)

    def logistic(self): 
       #assuming logistic function = sigmoid function = 1/(1+e^(-x))
        val = 1/(1 + np.exp(-self.val)) 
        der = self.der* np.exp(-self.val)/((1 + np.exp(-self.val))**2)
        return AD(val, der, self.name)

    # add log to other basesimport math
# this is where we are going to work on creating the new class



