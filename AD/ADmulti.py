import math
import numpy as np


class AD:
    """
    This class is used to represent functions (and their variables) for automatic differentiation.
    In order to get the derivative of a function with respect to a specific variable, that variable must be initialized as an AD object before performing any operation in which this variable is involved.

    Vector functions are also initialized as an AD object.

    ...

    Parameters
    ==========
    value : int, float, np.ndarray or list
        Values or functions used to construct a new variable or function object:
          - if value is int or float or np.ndarray         --> variable or scalar function
          - if value is list(int, float, AD or np.ndarray) --> vector function

    der : int, float, np.ndarray or list, optional
        Derivatives with respect to each of the variables of a function (default is 0).
        This parameter is only needed when instantiating variables and not for functions.

    name : str, optional
        Names of the variables (default is 'x').
        This parameter is only needed when instantiating variables and not for functions.

    Attributes
    ==========
    val : np.ndarray (scalar functions) or list (vector functions)
        Value(s) of the variable(s) where the functions are evaluated.
    der : np.ndarray (scalar functions) or list (vector functions)
        Jacobian: Derivative(s) with respect to each of the variable(s)
    name : list of strings
        Name(s) of the variable(s)

    Examples
    ==========
    # Scalar input (x)
    >>> x = AD(2,1,'x')
    >>> f = 7*x + 0.3
    >>> f
    Numerical Value is:
    [[14.3]],
    Jacobian is:
    [[7.]],
    Name is:
    ['x']

    # Vector input (x,y)
    >>> x = AD(2,1,'x')
    >>> y = AD(3,1,'y')
    >>> f = 5*x + 4*y + 0.5
    >>> f
    Numerical Value is:
    [[22.5]],
    Jacobian is:
    [[5. 4.]],
    Name is:
    ['x', 'y']

    # Vector input (x,y) and Vector output (f1,f2,f3)
    >>> x = AD(2,1,'x')
    >>> y = AD(3,1,'y')
    >>> f = AD([5*x+4*y+0.5, 43*x, 7]) #f1,f2,f3 = 5*x+4*y+0.5, 43*x, 7
    >>> f
    Numerical Value is:
    [22.5, 86.0, 7],
    Jacobian is:
    [[5.0, 4.0], [43.0, 0], [0, 0]],
    Name is:
    ['x', 'y']


    Methods
    ==========
    # AD object-related methods
    __init__(self, value, der=0, name='x'):  Constructs the necessary attributes of an AD object representing a variable or a function.
    __repr__(self):  Return the canonical string representation of the object.
    sort(self, order):  Sort the derivatives and variable names of an AD object by the specified order.

    # Basic operations
    __add__(self, other):  Perform addition on an AD object.
    __radd__(self, other):  Perform reverse addition on an AD object.
    __neg__(self):  Perform negation on AD objects.
    __sub__(self, other):  Perform subtraction on an AD object.
    __rsub__(self, other):  Perform reverse subtraction on an AD object.
    __mul__(self, other):  Perform multiplication on an AD object.
    __rmul__(self, other):  Perform reverse multiplication on an AD object.
    __truediv__(self, other):  Perform true division on an AD object.
    __rtruediv__(self, other):  Perform reverse true division on an AD object.
    __pow__(self, n):  Raise an AD object to the power of n.
    __rpow__(self, n):  Raise a number to the power of an AD object.

    # Comparisons
    __lt__(self, other):  Perform "less than" comparison on an AD object.
    __gt__(self, other):  Perform "greater omparison on an AD object.
    __le__(self, other):  Perform "less or equal than" comparison on an AD object.
    __ge__(self, other):  Perform "greater or equal than" comparison on an AD object.
    __eq__(self, other):  Perform "equality" comparison on an AD object.
    __ne__(self, other):  Perform "inequality" comparison on an AD object.

    # Elementary functions
    tan(self):  Compute the tangent of an AD object.
    sin(self):  Compute the sine of an AD object.
    cos(self):  Compute the cosine of an AD object.
    exp(self):  Compute the exponential of an AD object.
    ln(self):  Compute the natural logarithm of an AD object.
    ln_base(self):  Compute the base-specific logarithm of an AD object.
    sinh(self):  Compute the hyperbolic sine of an AD object.
    cosh(self):  Compute the hyperbolic cosine of an AD object.
    tanh(self):  Compute the hyperbolic tangent of an AD object.
    arcsin(self):  Compute the arcsine (inverse of sine) of an AD object.
    arccos(self):  Compute the arccosine (inverse of cosine) of an AD object.
    arctan(self):  Compute the arctangent (inverse of tangent) of an AD object.
    logistic(self):   Apply the sigmoid function to an AD object, defined as: sigmoid(x) =  1/(1+e**(-x))
    sqrt(self): Compute the square root of an AD object.
    """

    def __init__(self, value, der=0, name='x'):
        """
        Constructs the necessary attributes of an AD object representing a variable or a function.

        Parameters
        ----------
        value : int, float, np.ndarray or list
            Values or functions used to construct a new variable or function object
              - if value is int or float or np.ndarray         --> scalar function
              - if value is list(int, float, AD or np.ndarray) --> vector function

        der : int, float, np.ndarray or list, optional
            Derivatives with respect to each of the variables of a function (default is 0).
            This parameter is only needed when instantiating variables and not for functions.

        name : str, optional
            Names of the variables (default is 'x').
            This parameter is only needed when instantiating variables and not for functions.

        Returns
        -------
        AD object representing a variable or a function, with the corresponding derivatives and variable names.

        Examples
        --------
        # Scalar input (x)
        >>> x = AD(2,1,'x')
        >>> x
        Numerical Value is:
        [[2.]],
        Jacobian is:
        [[1.]],
        Name is:
        ['x']
        # Vector input (x,y)
        >>> x = AD(2,1,'x')
        >>> y = AD(3,1,'y')
        >>> f = 5*x + 4*y + 0.5
        >>> f
        Numerical Value is:
        [[22.5]],
        Jacobian is:
        [[5. 4.]],
        Name is:
        ['x', 'y']
        # Vector input (x,y) and Vector output (f1,f2,f3)
        >>> x = AD(2,1,'x')
        >>> y = AD(3,1,'y')
        >>> f = AD([5*x+4*y+0.5, 43*x, 7]) #f1,f2,f3 = 5*x+4*y+0.5, 43*x, 7
        >>> f
        Numerical Value is:
        [22.5, 86.0, 7],
        Jacobian is:
        [[5.0, 4.0], [43.0, 0], [0, 0]],
        Name is:
        ['x', 'y']
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
            try:
                unique_names = list(set(np.array(np.concatenate(names, axis=0))))  # reference
            except ValueError:
                unique_names = set(np.asarray(names).flatten())
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


            # if len(unique_names) != 0:
            self.val = np.array(global_value).reshape(-1, 1)
            self.der = np.array(global_jacobian)
            self.name = unique_names
            # else:
            # self.val = np.array(value).reshape(len(value), 1)

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
            elif isinstance(der, np.ndarray) and len(der.shape) == 2:
                der = der
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
        Get a string representation of the AD object. It prints:
           - the values where the function is evaluated (self.val)
           - the Jacobian (self.der)
           - the names of the function variables (self.name)

        Returns
        -------
        String representing the AD object's values, Jacobian and variable names.

        Example
        -------
        >>> x = AD(2,1,'x')
        >>> y = AD(3,1,'y')
        >>> f = AD([5*x+4*y+0.5, 11*x*y])
        >>> f
        Numerical Value is:
        [22.5, 66.0],
        Jacobian is:
        [[5.0, 4.0], [33.0, 22.0]],
        Name is:
        ['x', 'y']
        """
        val = self.val
        der = self.der
        name = self.name
        return 'Numerical Value is:\n{}, \nJacobian is:\n{}, \nName is:\n{}'.format(val, der, name)

    def sort(self, order):
        """
        Sort the derivatives and variable names of an AD object by the specified order.

        Parameters
        ----------
        order : list(str)
            Names of the variables in the desired order.

        Returns
        -------
        AD object with self.val, self.der and self.name in the desired order.

        Example
        -------
        >>> x = AD(2,1,'x')
        >>> y = AD(3,1,'y')
        >>> z = AD(4,1,'z')
        >>> f = AD([5*x+4*y+3*z, x*y*z])
        >>> print(f)
        >>> f.sort(['x', 'y', 'z'])
        >>> print(f)
        Numerical Value is:
        [34.0, 24.0],
        Jacobian is:
        [[5.0, 3.0, 4.0], [12.0, 6.0, 8.0]],
        Name is:
        ['x', 'z', 'y']
        Numerical Value is:
        [34.0, 24.0],
        Jacobian is:
        [[5.0, 4.0, 3.0], [12.0, 8.0, 6.0]],
        Name is:
        ['x', 'y', 'z']
        """
        if not isinstance(order, list) and not isinstance(order, np.ndarray):
            raise TypeError('Order should be an array-like composed of strings')
        for string in order:
            if not isinstance(string, str):
                raise TypeError('Order should only be composed of strings')
        if self.name == order:
            return self
        for i in range(len(self.der)):
            final_derivative_i = self.der[i].copy()
            for j, variable in enumerate(order):
                index = self.name.index(variable)
                derivative = self.der[i][index].copy()
                final_derivative_i[j] = derivative
            self.der[i] = final_derivative_i
        self.name = order
        return self

    def __add__(self, other):
        """
        Perform addition on an AD object.

        Parameters
        ----------
        other : int, float, np.ndarray, list or AD
            Values to be added to self, which is an AD object.

        Returns
        -------
        AD object representing the result of self+other

        Example
        -------
        >>> x = AD(1,1,'x')
        >>> y = AD(2,1,'y')
        >>> z = AD(3,1,'z')
        >>> x+y+z
        Numerical Value is:
        [[6.]],
        Jacobian is:
        [[1. 1. 1.]],
        Name is:
        ['x', 'y', 'z']

        >>> x = AD(1,1,'x')
        >>> y = AD(2,1,'y')
        >>> z = AD(3,1,'z')
        >>> v = AD([x+y+z, x+2])
        >>> v
        Numerical Value is:
        [6. 3.],
        Jacobian is:
        [[1. 1. 1.]
         [1. 0. 0.]],
        Name is:
        ['x', 'z', 'y']

        >>> x = AD(2,1,'x')
        >>> y = AD(3,1,'y')
        >>> f1 = AD([10*x, 10*y])
        >>> f2 = AD([x, y])
        >>> f1+f2+1000
        Numerical Value is:
        [[1022.]
         [1033.]],
        Jacobian is:
        [[11.  0.]
         [ 0. 11.]],
        Name is:
        ['x', 'y']
        """
        if isinstance(other, AD):
            names_1 = self.name.copy()
            names_2 = other.name
            value = self.val + other.val
            derivative = self.der.copy()
            for i, name_2 in enumerate(names_2):
                if name_2 in names_1:
                    index_1 = names_1.index(name_2)
                    #     if not isinstance(derivative, np.ndarray):
                      #  derivative = np.array(derivative)
                    #  if not isinstance(other.der, np.ndarray):
                      #   other.der = np.array(other.der)
                    derivative[:, index_1] = derivative[:, index_1] + other.der[:, i]
                else:
                    derivative = np.concatenate((derivative, other.der[:, i].reshape(-1, 1)), axis=1)
                    names_1.append(name_2)
            name = names_1
        else:
            if isinstance(other, np.ndarray):
                if other.shape != self.val.shape:
                    raise ValueError('Cannot add arrays of different size')
                else:
                    for v in other.flatten():
                        if type(v) not in [int, float, np.int, np.int8, np.int16, np.int32, np.int64, \
                                           np.uint8, np.uint16, np.uint32, np.uint64, np.float32, np.float64]:
                            raise TypeError('Array entries must be int or float')
                    else:
                        value = self.val + other
            elif isinstance(other, int) or isinstance(other, float):
                value = self.val + other
            else:
                raise ValueError('Other must be an array, AD, int or float')

            derivative = self.der
            name = self.name
        return AD(value, derivative, name)

    def __radd__(self, other):
        """
        Perform reverse addition on an AD object.

        Parameters
        ----------
        other : int, float, np.ndarray, list or AD
            Values to be added to self, which is an AD object.

        Returns
        -------
        AD object representing the result of self+other
        Example
        -------
        >>> x = AD(1,1,'x')
        >>> 2+x
        Numerical Value is:
        [[3.]],
        Jacobian is:
        [[1.]],
        Name is:
        ['x']

        >>> x = AD(1,1,'x')
        >>> y = AD(2,1,'y')
        >>> z = AD([x+y, x])
        >>> 40 + z
        Numerical Value is:
        [[43.]
         [41.]],
        Jacobian is:
        [[1. 1.]
         [1. 0.]],
        Name is:
        ['x', 'y']
        """
        new_var = AD(self.val, self.der, self.name)  # create a new variable
        return new_var.__add__(other)

    def __neg__(self):
        """
        Perform negation on AD objects.

        Returns
        -------
        AD object representing the result of -self

        Example
        -------
        >>> x = AD(2,1,'x')
        >>> -x
        Numerical Value is:
        [[-2.]],
        Jacobian is:
        [[-1.]],
        Name is:
        ['x']

        >> x = AD(2,1,'x')
        >> y = AD(3,1,'y')
        >> z = AD([10*x, 100*y])
        >> -z
        Numerical Value is:
        [[ -20.]
         [-300.]],
        Jacobian is:
        [[ -10.   -0.]
         [  -0. -100.]],
        Name is:
        ['x', 'y']
        """
        val = -self.val.copy()
        der = -self.der.copy()
        name = self.name
        return AD(val, der, name)

        # overloading the '-' operator

    def __sub__(self, other):
        """
        Perform subtraction on an AD object.

        Parameters
        ----------
        other : int, float, np.ndarray, list or AD
            Values to be subtracted from self, which is an AD object.

        Returns
        -------
        AD object representing the result of self-other

        Example
        -------
        >>> x = AD(1,1,'x')
        >>> x-3
        Numerical Value is:
        [[-2.]],
        Jacobian is:
        [[1.]],
        Name is:
        ['x']

        >>> x = AD(1,1,'x')
        >>> y = AD(3,1,'y')
        >>> x-y
        Numerical Value is:
        [[-2.]],
        Jacobian is:
        [[ 1. -1.]],
        Name is:
        ['x', 'y']

        >>> x = AD(2,1,'x')
        >>> y = AD(3,1,'y')
        >>> z = AD([10*x, 100*y])
        >>> z-2
        Numerical Value is:
        [[ 18.]
         [298.]],
        Jacobian is:
        [[ 10.   0.]
         [  0. 100.]],
        Name is:
        ['x', 'y']
        """
        return self.__add__(-other)

    def __rsub__(self, other):
        """
        Perform reverse subtraction on an AD object.

        Parameters
        ----------
        other : int, float, np.ndarray, list or AD
            Values from which self will be subtracted.

        Returns
        -------
        AD object representing the result of other-self

        Example
        -------
        >>> x = AD(1,1,'x')
        >>> 4-x
        Numerical Value is:
        [[3.]],
        Jacobian is:
        [[-1.]],
        Name is:
        ['x']

        >>> x = AD(2,1,'x')
        >>> y = AD(3,1,'y')
        >>> z = AD([10*x, 100*y])
        >>> 2-z
       Numerical Value is:
        [[ -18.]
         [-298.]],
        Jacobian is:
        [[ -10.   -0.]
         [  -0. -100.]],
        Name is:
        ['x', 'y']
        """
        return -(self.__sub__(other))

    # overloading the '*' operator
    def __mul__(self, other):
        """
        Perform multiplication on an AD object.

        Parameters
        ----------
        other : int, float, np.ndarray, list or AD
            Values multiplying self, which is an AD object.

        Returns
        -------
        AD object representing the result of self*other

        Examples
        --------
        >>> x = AD(1,1,'x')
        >>> x*2
        Numerical Value is:
        [[2.]],
        Jacobian is:
        [[2.]],
        Name is:
        ['x']

        >>> x = AD(1,1,'x')
        >>> y = AD(2,1,'y')
        >>> x*y
        Numerical Value is:
        [[2.]],
        Jacobian is:
        [[2. 1.]],
        Name is:
        ['x', 'y']

        >>> x = AD(2,1,'x')
        >>> y = AD(3,1,'y')
        >>> f1 = AD([10*x, 10*y])
        >>> f2 = AD([x, y])
        >>> f1*f2
        Numerical Value is:
        [[40.]
         [90.]],
        Jacobian is:
        [[40.  0.]
         [ 0. 60.]],
        Name is:
        ['x', 'y']
        """
        if isinstance(other, AD):
            names_1 = self.name.copy()
            names_2 = other.name.copy()
            new_value = self.val.copy() * other.val.copy()
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
        else:  # one of the coefficients of other is None, it is a constant
            #if isinstance(other, np.ndarray):
                #for value in other.flatten():
                    #if type(value) not in [int, float, np.int, np.int8, np.int16, np.int32, np.int64, \
                                           #np.uint8, np.uint16, np.uint32, np.uint64, np.float32, np.float64]:
                            #raise TypeError('Arrays must contain only integers or float')
                #if self.val.shape[1] == other.shape[0] or self.val.shape == other.shape:

                    #new_value = np.dot(self.val, other)
                    #print('self.der.shape',self.der.shape)
                    #derivative = np.dot(self.der, other)

                    #print('Here3!')
                    #name = self.name
                #else:
                    #raise ValueError('Input dimension mismatch')
            if isinstance(other, int) or isinstance(other, float):
                new_value = self.val * other
                derivative = self.der * other
                name = self.name
            else:
                raise TypeError('Invalid input type ')
        return AD(new_value, derivative, name)

    def __rmul__(self, other):
        """
        Perform reverse multiplication on an AD object.

        Parameters
        ----------
        other : int, float, np.ndarray, list or AD
            Values to be multiplied by self, which is an AD object.

        Returns
        -------
        AD object representing the result of other*self


        Examples
        --------
        >>> x = AD(2,1,'x')
        >>> 2*x
        Numerical Value is:
        [[4.]],
        Jacobian is:
        [[2.]],
        Name is:
        ['x']

        >>> x = AD(1,1,'x')
        >>> y = AD(3,1,'y')
        >>> z = AD([x+y, x])
        >>> 2*z
        Numerical Value is:
        [[8.]
         [2.]],
        Jacobian is:
        [[2. 2.]
         [2. 0.]],
        Name is:
        ['x', 'y']
        """
        return AD(self.val, self.der, self.name).__mul__(other)

    def __truediv__(self, other):  # todo: check for forbidden values
        """
        Perform true division on an AD object.

        Parameters
        ----------
        other : int, float, np.ndarray, list or AD
            Values dividing self, which is an AD object.

        Returns
        -------
        AD object representing the result of self/other

        Examples
        --------
        >>> x = AD(4,1,'x')
        >>> x/2
        Numerical Value is:
        [[2.]],
        Jacobian is:
        [[0.5]],
        Name is:
        ['x']

        >>> x = AD(2,1,'x')
        >>> y = AD(4,1,'y')
        >>> y/x
        Numerical Value is:
        [[2.]],
        Jacobian is:
        [[ 1. -2.]],
        Name is:
        ['y', 'x']

        >>> x = AD(2,1,'x')
        >>> y = AD(3,1,'y')
        >>> f1 = AD([10*x, 10*y])
        >>> f2 = AD([2, 1])
        >>> f1/f2
        Numerical Value is:
        [[10.]
         [30.]],
        Jacobian is:
        [[ 5.  0.]
         [ 0. 10.]],
        Name is:
        ['x', 'y']
        """
        if isinstance(other, AD):
            names_1 = self.name
            names_2 = other.name
            for val in other.val:
                if val == 0:
                    raise ZeroDivisionError
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
        else:
            #if isinstance(other, np.ndarray):
                #if other.shape != self.val.shape:
                    #raise TypeError('Input dimension mismatch')
               # for v in other.flatten():
                    #if type(v) not in [int, float, np.int, np.int8, np.int16, np.int32, np.int64, \
                                       #np.uint8, np.uint16, np.uint32, np.uint64, np.float32, np.float64]:
                       # raise TypeError('Array entries must be int or float')
               # if np.min(np.abs(other)) == 0:
                    #raise ZeroDivisionError
               # new_value = self.val / other
                #derivative = self.der / other
               # name = self.name
            if isinstance(other, int) or isinstance(other, float):
                if other == 0:
                    raise ZeroDivisionError
                new_value = self.val / other
                derivative = self.der / other
                name = self.name
            else:
                raise TypeError('Wrong input type')
        return AD(new_value, derivative, name)

    def __rtruediv__(self, other):
        """
        Perform reverse true division on an AD object.

        Parameters
        ----------
        other : int, float, np.ndarray, list or AD
            Values to be divided by self, which is an AD object.

        Returns
        -------
        AD object representing the result of other/self

        Examples
        --------
        >>> x = AD(10,1,'x')
        >>> 2/x
        Numerical Value is:
        [[0.2]],
        Jacobian is:
        [[-0.02]],
        Name is:
        ['x']

        >>> x = AD(2,1,'x')
        >>> y = AD(3,1,'y')
        >>> z = x+y
        >>> 1/z
        Numerical Value is:
        [[0.2]],
        Jacobian is:
        [[-0.04 -0.04]],
        Name is:
        ['x', 'y']

        >>> x = AD(2,1,'x')
        >>> y = AD(3,1,'y')
        >>> z = AD([x, y])
        >>> 1/z
        Numerical Value is:
        [[0.5       ]
         [0.33333333]],
        Jacobian is:
        [[-0.25       -0.        ]
         [-0.         -0.11111111]],
        Name is:
        ['x', 'y']
        """
        if np.abs(np.min(self.val)) == 0:
            raise ZeroDivisionError
        if not isinstance(other, int) and not isinstance(other, float):
            raise TypeError('Invalid input type')
        new_val = other / self.val
        new_der = -other * self.der / self.val ** 2
        names = self.name
        return AD(new_val, new_der, names)


    def __pow__(self, n):
        """
        Raise an AD object to the power of n.

        Parameters
        ----------
        n : int, float or AD
            Exponent to which self will be raised. Self is an AD object.

        Returns
        -------
        AD object representing the result of self**other

        Examples
        --------
        >>> x = AD(2,1,'x')
        >>> x**3
        Numerical Value is:
        [[8.]],
        Jacobian is:
        [[12.]],
        Name is:
        ['x']

        >>> x = AD(2,1,'x')
        >>> y = AD(3,1,'y')
        >>> x**y
        Numerical Value is:
        [[8.]],
        Jacobian is:
        [[12.          5.54517744]],
        Name is:
        ['x', 'y']

        >>> x = AD(2,1,'x')
        >>> y = AD(3,1,'y')
        >>> z = x+y
        >>> z**2
        Numerical Value is:
        [[25.]],
        Jacobian is:
        [[10. 10.]],
        Name is:
        ['x', 'y']

        """
        if isinstance(n, float) or isinstance(n, int):
            n = float(n)  # n is an int/float
            value = self.val.copy()
            derivative = self.der.copy()
            names = self.name
            #Taking sqrt of neg
            print('Value=', value)
            print('n=',n)
            print('(value < 0).any() = ', (value < 0).any())
            if (value < 0).any() and 0 < n < 1:
                raise ValueError('Illegal value and exponent')

            if (value == 0).any() and n < 1:
                raise ZeroDivisionError

            if (value==0).any() and ((n==1) or (n==0)):
                raise ZeroDivisionError

            new_val = value ** n
            for i, name in enumerate(names):
                derivative[:, i] = n * self.der[:, i] * value ** (n - 1)
            return AD(new_val, derivative, names)

        elif isinstance(n, AD):  # n is an AD object
            value_base = self.val
            value_exponent = n.val
            if not self.val.shape == n.val.shape:
                raise TypeError("Incoherent dimensions")
                #for v in n.val.flatten():
                #    if type(v) not in [int, float, np.int, np.int8, np.int16, np.int32, np.int64, \
                #                       np.uint8, np.uint16, np.uint32, np.uint64, np.float32, np.float64]:
                #        raise TypeError('Array entries must be int or float')
            for val_base, val_exponent in zip(value_base, value_exponent):
                if val_base < 0 and 0 < val_exponent < 1:
                    raise ValueError('Illegal value and exponent')
                if val_base == 0 and val_exponent < 0:
                    raise ZeroDivisionError
                if val_base == 0 and ((val_exponent==1) or (val_exponent==0)):
                    raise ZeroDivisionError
            new_val = value_base ** value_exponent
            names_1 = self.name
            names_2 = n.name
            new_names = names_1 + [name for name in names_2 if name not in names_1]
            derivative = self.der.copy()
            for i, name in enumerate(new_names):
                if name in names_1 and name in names_2:
                    index_2 = names_2.index(name)
                    new_der = (n.der[:, index_2] * np.log(value_base) + value_exponent * derivative[:,
                                                                                           i] / value_base) * new_val
                    derivative[:, i] = new_der
                if name in names_1 and name not in names_2:
                    index_1 = names_1.index(name)
                    new_der = derivative[:, index_1] * value_exponent * value_base ** (value_exponent - 1)
                    derivative[:, index_1] = new_der
                if name in names_2 and name not in names_1:
                    index_2 = names_2.index(name)
                    new_der = n.der[:, index_2] * np.log(value_base) * new_val
                    derivative = np.concatenate((derivative, new_der), axis=1)
            return AD(new_val, derivative, new_names)
        else:
            raise TypeError('Invalid Input Type for the exponent')


    def __rpow__(self, other):
        """
        Raise a number to the power of an AD object.

        Parameters
        ----------
        other : int or float
            Exponent to which self will be raised. Self is an AD object.

        Returns
        -------
        AD object representing the result of other**self

        Examples
        --------
        >>> x = AD(2,1,'x')
        >>> 3**x
        Numerical Value is:
        [[9.]],
        Jacobian is:
        [[6.23832463]],
        Name is:
        ['x']

        >>> x = AD(2,1,'x')
        >>> y = AD(3,1,'y')
        >>> z = x+y
        >>> 2**z
        Numerical Value is:
        [[32.]],
        Jacobian is:
        [[51.5020132 51.5020132]],
        Name is:
        ['x', 'y']
        """
        value = self.val
        print('type other:', type(other))
        if isinstance(other, int) or isinstance(other, float):
            if other < 0:
                raise ValueError('Inconsistent value found for the base')
            if other == 0 and np.min(value) < 0:
                raise ZeroDivisionError
            der = self.der
            name = self.name
            new_value = other ** value
            new_der = der * np.log(other) * new_value
        #elif isinstance(other, np.ndarray):
            #if other.shape != value.shape:
                #raise ValueError('Invalid dimension')
            #for val in other:
                #if not isinstance(val, int) and not isinstance(val, float):
                    #raise TypeError('Invalid input type')
                #if val < 0:
                    #raise ValueError('Inconsistent value found for the base')
                #if val == 0 and np.min(value) == 0:
                    #raise ZeroDivisionError
            #der = self.der
            #name = self.name
            #new_value = other ** value
            #new_der = der * np.log(value) * new_value
        else:
            raise TypeError('Invalid input type')
        return AD(new_value, new_der, name)


    def __lt__(self, other):
        """
        Perform "less than" comparison on an AD object.

        Parameters
        ----------
        other : int, float, np.ndarray, list or AD
            Value to compare with self, which is an AD object.

        Returns
        -------
        Boolean representing the result of self < other

        Examples
        --------
        >>> x = AD(2,1,'x')
        >>> y = AD(2,1,'y')
        >>> u = x+y
        >>> v = 2*(x+y)
        >>> u < v
        True

        >>> x = AD(2,1,'x')
        >>> y = AD(2,1,'y')
        >>> u = AD([x+y, x+y])
        >>> v = AD([x+y, 2*(x+y)])
        >>> u < v
        True
        """
        if isinstance(other, AD):
            if self.val.shape != other.val.shape:
                raise AttributeError('Incoherent dimension input')
            for val, other_value in zip(self.val, other.val):
                if val >= other_value:
                    return False
            return True
        elif isinstance(other, int) or isinstance(other, float):
            for value in self.val:
                if value >= other:
                    return False
            return True
        else:
            raise TypeError('Invalid input type')



    def __gt__(self, other):
        """
        Perform "greater than" comparison on an AD object.

        Parameters
        ----------
        other : int, float, np.ndarray, list or AD
            Value to compare with self, which is an AD object.

        Returns
        -------
        Boolean representing the result of self > other


        Examples
        --------
        >>> x = AD(2,1,'x')
        >>> y = AD(2,1,'y')
        >>> u = x+y
        >>> v = 2*(x+y)
        >>> v > u
        True

        >>> x = AD(2,1,'x')
        >>> y = AD(2,1,'y')
        >>> u = AD([x+y, x+y])
        >>> v = AD([x+y, 2*(x+y)])
        >>> v > u
        True
        """
        if isinstance(other, AD):
            return other.__lt__(self)
        elif isinstance(other, int) or isinstance(other, float):
            for value in self.val:
                if value <= other:
                    return False
            return True
        else:
            raise TypeError('Invalid input type')


    def __eq__(self, other):
        """
        Perform "equality" comparison on an AD object.

        Parameters
        ----------
        other : int, float, np.ndarray, list or AD
            Value to compare with self, which is an AD object.

        Returns
        -------

        AD object representing the result of self == other

        Boolean representing the result of self <= other

        Examples
        --------
        >>> x = AD(2,1,'x')
        >>> y = AD(2,1,'y')
        >>> u = x+y
        >>> v = 2*(x+y)
        >>> u <= v , u <= v/2
        (True, True)

        >>> x = AD(2,1,'x')
        >>> y = AD(2,1,'y')
        >>> u = AD([x+y, x+y])
        >>> v = AD([x+y, 2*(x+y)])
        >>> u <= v
        True

        """
        return not bool(self.__lt__(other)) and not bool(self.__gt__(other))


    def __le__(self, other):
        """
        Perform "less or equal than" comparison on an AD object.

        Parameters
        ----------
        other : int, float, np.ndarray, list or AD
            Value to compare with self, which is an AD object.

        Returns
        -------

        AD object representing the result of self <= other

        Boolean representing the result of self >= other

        Examples
        --------
        >>> x = AD(2,1,'x')
        >>> y = AD(2,1,'y')
        >>> u = x+y
        >>> v = 2*(x+y)
        >>> v >= u, v/2 >= u
        (True, True)

        >>> x = AD(2,1,'x')
        >>> y = AD(2,1,'y')
        >>> u = AD([x+y, x+y])
        >>> v = AD([x+y, 2*(x+y)])
        >>> v >= u
        True

        """
        print('Eq', bool(self.__eq__(other)))
        print('Le', bool(self.__lt__(other)))
        return (bool(self.__lt__(other)) or bool(self.__eq__(other)))

    def __ge__(self, other):
        """
        Perform "greater or equal than" comparison on an AD object.

        Parameters
        ----------
        other : int, float, np.ndarray, list or AD
            Value to compare with self, which is an AD object.

        Returns
        -------

        AD object representing the result of self >= other

        Boolean representing the result of self == other

        Examples
        --------
        >>> x = AD(2,1,'x')
        >>> y = AD(2,1,'y')
        >>> u = x+y
        >>> v = 2*(x+y)
        >>> v == u, v/2 == u
        (False, True)

        >>> x = AD(2,1,'x')
        >>> y = AD(2,1,'y')
        >>> u = AD([x+y, x+y])
        >>> v = AD([x+y, x+y])
        >>> v == u
        True

        """
        # raises an error when the two objects do not have the same attributes
        return self.__gt__(other) or self.__eq__(other)



    def __ne__(self, other):
        """
        Perform "inequality" comparison on an AD object.

        Parameters
        ----------
        other : int, float, np.ndarray, list or AD
            Value to compare with self, which is an AD object.

        Returns
        -------
        AD object representing the result of self != other

        Examples
        --------
        >>> x = AD(2,1,'x')
        >>> y = AD(2,1,'y')
        >>> u = x+y
        >>> v = 2*(x+y)
        >>> v != u, v/2 != u
        (True, False)

        >>> x = AD(2,1,'x')
        >>> y = AD(2,1,'y')
        >>> u = AD([x+y, x+y])
        >>> v = AD([x+y, x+y])
        >>> v != u
        False
        """
        return not self.__eq__(other)


    def tan(self):
        """
        Compute the tangent of an AD object.

        Returns
        -------
        AD object representing the result of tan(self)

        Examples
        --------
        >>> x = AD(np.pi/4,1,'x')
        >>> x.tan()
        Numerical Value is:
        [[1.]],
        Jacobian is:
        [[2.]],
        Name is:
        ['x']

        >>> x = AD(np.pi/4,1,'x')
        >>> y = AD(np.pi/4,1,'y')
        >>> z = AD([x,y])
        >>> z.sin()
        Numerical Value is:
        [[1.]
         [1.]],
        Jacobian is:
        [[2. 0.]
         [0. 2.]],
        Name is:
        ['x', 'y']
        """
        nonpoints = map(lambda x: ((x / np.pi) - 0.5) % 1 == 0.00, self.val)
        if any(nonpoints):
            raise ValueError("Math error, Tangent cannot handle i*0.5pi ")
        val = np.tan(self.val)
        der = np.multiply(1 / np.power(np.cos(self.val), 2), self.der)
        return AD(val, der, self.name)


    def sin(self):
        """
        Compute the sine of an AD object.

        Returns
        -------
        AD object representing the result of sin(self)

        Examples
        --------
        >>> x = AD(np.pi/4,1,'x')
        >>> x.sin()
        Numerical Value is:
        [[0.70710678]],
        Jacobian is:
        [[0.70710678]],
        Name is:
        ['x']

        >>> x = AD(np.pi/4,1,'x')
        >>> y = AD(np.pi/4,1,'y')
        >>> z = AD([x,y])
        >>> z.sin()
        Numerical Value is:
        [[0.70710678]
         [0.70710678]],
        Jacobian is:
        [[0.70710678 0.        ]
         [0.         0.70710678]],
        Name is:
        ['x', 'y']
        """
        val = np.sin(self.val)
        der = np.cos(self.val) * self.der
        return AD(val, der, self.name)

    def cos(self):
        """
        Compute the cosine of an AD object.

        Returns
        -------
        AD object representing the result of cos(self)

        Examples
        --------
        >>> x = AD(np.pi/4,1,'x')
        >>> x.cos()
        Numerical Value is:
        [[0.70710678]],
        Jacobian is:
        [[-0.70710678]],
        Name is:
        ['x']

        >>> x = AD(np.pi/4,1,'x')
        >>> y = AD(np.pi/4,1,'y')
        >>> z = AD([x,y])
        >>> z.cos()
        Numerical Value is:
        [[0.70710678]
         [0.70710678]],
        Jacobian is:
        [[-0.70710678 -0.        ]
         [-0.         -0.70710678]],
        Name is:
        ['x', 'y']
        """
        val = np.cos(self.val)
        der = -np.sin(self.val) * self.der
        return AD(val, der, self.name)

    def exp(self):
        """
        Compute the exponential of an AD object.

        Returns
        -------
        AD object representing the result of exp(self)

        Examples
        --------
        >>> x = x = AD(2,1,'x')
        >>> x.exp()
        Numerical Value is:
        [[7.3890561]],
        Jacobian is:
        [[7.3890561]],
        Name is:
        ['x']

        >>> x = AD(2,1,'x')
        >>> y = AD(3,1,'y')
        >>> z = AD([x,y])
        >>> z.exp()
        Numerical Value is:
        [[ 7.3890561 ]
         [20.08553692]],
        Jacobian is:
        [[ 7.3890561   0.        ]
         [ 0.         20.08553692]],
        Name is:
        ['x', 'y']
        """
        #val = np.exp(self.val)
        #der = np.multiply(np.exp(self.val), self.der)
        #return AD(val, der, self.name) # return np.exp(1)**self
        return self.__rpow__(np.exp(1))


    def ln(self):
        """
        Compute the natural logarithm of an AD object.

        Returns
        -------
        AD object representing the result of ln(self)

        Examples
        --------
        >>> x = x = AD(2,1,'x')
        >>> x.ln()
        Numerical Value is:
        [[0.69314718]],
        Jacobian is:
        [[0.5]],
        Name is:
        ['x']

        >>> x = AD(2,1,'x')
        >>> y = AD(3,1,'y')
        >>> z = AD([x,y])
        >>> z.ln()
        Numerical Value is:
        [[0.69314718]
         [1.09861229]],
        Jacobian is:
        [[0.5        0.        ]
         [0.         0.33333333]],
        Name is:
        ['x', 'y']
        """

        for val in self.val:
            if val <= 0:
                raise ValueError("Cannot take natural log of zero or negative values")
        val = np.log(self.val)
        der = self.der / self.val
        return AD(val, der, self.name)


    def ln_base(self, base):
        """
        Compute the base-specific logarithm of an AD object.

        Returns
        -------
        AD object representing the result of ln_base(self)

        Examples
        --------
        >>> x = x = AD(8,1,'x')
        >>> x.ln_base(2)
        Numerical Value is:
        [[3.]],
        Jacobian is:
        [[0.18033688]],
        Name is:
        ['x']

        >>> x = AD(8,1,'x')
        >>> y = AD(32,1,'y')
        >>> z = AD([x*y,y])
        >>> z.ln_base(2)
        Numerical Value is:
        [[8.]
         [5.]],
        Jacobian is:
        [[0.18033688 0.36067376]
         [0.         0.04508422]],
        Name is:
        ['x', 'y']
        """
        return self.ln()/np.log(base)


    def sinh(self):  # hyperbolic sin
        """
        Compute the hyperbolic sine of an AD object.

        Returns
        -------
        AD object representing the result of sinh(self)

        Examples
        --------
        >>> x = x = AD(2,1,'x')
        >>> x.sinh()
        Numerical Value is:
        [[3.62686041]],
        Jacobian is:
        [[3.76219569]],
        Name is:
        ['x']

        >>> x = AD(2,1,'x')
        >>> y = AD(3,1,'y')
        >>> z = AD([x,y])
        >>> z.sinh()
        Numerical Value is:
        [[ 3.62686041]
         [10.01787493]],
        Jacobian is:
        [[ 3.76219569  0.        ]
         [ 0.         10.067662  ]],
        Name is:
        ['x', 'y']
        """
        # d/dx (sinh x) = cosh x
        # sinh x = (e^x - e^(-x))/2  range (-inf, inf)
        # val = np.multiply(.5, (np.exp(self.val) - np.exp(np.multiply(-1, self.val))))
        val = np.sinh(self.val)
        der = np.cosh(self.val) * self.der
        return AD(val, der, self.name)

    def cosh(self):  # hyperbolic cos
        """
        Compute the hyperbolic cosine of an AD object.

        Returns
        -------
        AD object representing the result of cosh(self)

        Examples
        --------
        >>> x = x = AD(2,1,'x')
        >>> x.cosh()
        Numerical Value is:
        [[3.76219569]],
        Jacobian is:
        [[3.62686041]],
        Name is:
        ['x']

        >>> x = AD(2,1,'x')
        >>> y = AD(3,1,'y')
        >>> z = AD([x,y])
        >>> z.cosh()
        Numerical Value is:
        [[ 3.62686041]
         [10.01787493]],
        Jacobian is:
        [[ 3.76219569  0.        ]
         [ 0.         10.067662  ]],
        Name is:
        ['x', 'y']
        """
        # d/dx (cosh x) = sinh x
        # cosh x = (e^x + e^(-x))/2  range (-inf, inf)
        # val = np.multiply(.5, (np.exp(self.val) + np.exp(np.multiply(-1, self.val))))
        val = np.cosh(self.val)
        der = np.sinh(self.val) * self.der
        return AD(val, der, self.name)


    def tanh(self):  # hyperbolic tan
        """
        Compute the hyperbolic tangent of an AD object.

        Returns
        -------
        AD object representing the result of tanh(self)

        Examples
        --------
        >>> x = x = AD(2,1,'x')
        >>> x.tanh()
        Numerical Value is:
        [[0.96402758]],
        Jacobian is:
        [[0.07065082]],
        Name is:
        ['x']

        >>> x = AD(2,1,'x')
        >>> y = AD(3,1,'y')
        >>> z = AD([x,y])
        >>> z.tanh()
       Numerical Value is:
        [[0.96402758]
         [0.99505475]],
        Jacobian is:
        [[0.07065082 0.        ]
         [0.         0.00986604]],
        Name is:
        ['x', 'y']
        """
        # d/dx (tanh x) = (sech x)^2 = 1/((cosh x)^2)
        # tanh x = (e^x - e^(-x)) / (e^x + e^(-x))       range (-inf, inf)
        val = np.tanh(self.val)
        der = (1 / np.power(np.cosh(self.val), 2)) * self.der
        # return self.sinh()/self.cosh()
        return AD(val, der, self.name)


    def arcsin(self):
        """
        Compute the arcsine (inverse of sine) of an AD object.

        Returns
        -------
        AD object representing the result of arcsin(self)

        Examples
        --------
        >>> x = AD(-0.5,1,'x')
        >>> x.arcsin()
        Numerical Value is:
        [[-0.52359878]],
        Jacobian is:
        [[1.15470054]],
        Name is:
        ['x']

        >>> x = AD(-0.5,1,'x')
        >>> y = AD(0.5,1,'y')
        >>> z = AD([x,y])
        >>> z.arcsin()
        Numerical Value is:
        [[-0.52359878]
         [ 0.52359878]],
        Jacobian is:
        [[1.15470054 0.        ]
         [0.         1.15470054]],
        Name is:
        ['x', 'y']
        """
        if ((self.val <= -1).any() or (self.val>=1).any()):
           raise ValueError("Cannot take derivative of arcsin of value outside of range (-1, 1)")
        val = np.arcsin(self.val)
        # der = (1/(np.sqrt(1 - np.power(self.val, 2)))) * self.der
        der = self.der * ((1 - self.val ** 2) ** (-0.5))
        return AD(val, der, self.name)


    def arccos(self):
        """
        Compute the arccosine (inverse of cosine) of an AD object.

        Returns
        -------
        AD object representing the result of arccos(self)

        Examples
        --------
        >>> x = AD(-0.5,1,'x')
        >>> x.arccos()
        Numerical Value is:
        [[2.0943951]],
        Jacobian is:
        [[-1.15470054]],
        Name is:
        ['x']

        >>> x = AD(-0.5,1,'x')
        >>> y = AD(0.5,1,'y')
        >>> z = AD([x,y])
        >>> z.arccos()
        Numerical Value is:
        [[2.0943951 ]
         [1.04719755]],
        Jacobian is:
        [[-1.15470054 -0.        ]
         [-0.         -1.15470054]],
        Name is:
        ['x', 'y']
        """
        if (self.val <= -1).any() or (self.val >= 1).any():
            raise ValueError("Cannot take derivative of arcsin of value outside of range (-1, 1)")
        val = np.arccos(self.val)
        # der = -(1/(np.sqrt(1 - np.power(self.val, 2)))) * self.der
        der = -self.der * ((1 - self.val ** 2) ** (-0.5))
        return AD(val, der, self.name)


    def arctan(self):
        """
        Compute the arctangent (inverse of tangent) of an AD object.

        Returns
        -------
        AD object representing the result of arctan(self)

        Examples
        --------
        >>> x = AD(-0.5,1,'x')
        >>> x.arctan()
        Numerical Value is:
        [[-0.46364761]],
        Jacobian is:
        [[0.8]],
        Name is:
        ['x']

        >>> x = AD(-0.5,1,'x')
        >>> y = AD(0.5,1,'y')
        >>> z = AD([x,y])
        >>> z.arctan()
        Numerical Value is:
        [[-0.46364761]
         [ 0.46364761]],
        Jacobian is:
        [[0.8 0. ]
         [0.  0.8]],
        Name is:
        ['x', 'y']
        """
        val = np.arctan(self.val)
        der = self.der * (1 + self.val ** 2) ** (-1)
        return AD(val, der, self.name)


    def logistic(self):
        """
        Apply the sigmoid function to an AD object.

        The sigmoid function of x is defined as:

            sigmoid(x) =  1/(1+e**(-x))

        Returns
        -------
        AD object representing the result of sigmoid(self)

        Examples
        --------
        >>> x = AD(-0.5,1,'x')
        >>> x.logistic()
        Numerical Value is:
        [[0.37754067]],
        Jacobian is:
        [[0.23500371]],
        Name is:
        ['x']

        >>> x = AD(-0.5,1,'x')
        >>> y = AD(0.5,1,'y')
        >>> z = AD([x,y])
        >>> z.logistic()
        Numerical Value is:
        [[0.37754067]
         [0.62245933]],
        Jacobian is:
        [[0.23500371 0.        ]
         [0.         0.23500371]],
        Name is:
        ['x', 'y']
        """
        # assuming logistic function = sigmoid function = 1/(1+e^(-x))
        val = 1 / (1 + np.exp(-self.val))
        der = self.der * np.exp(-self.val) / ((1 + np.exp(-self.val)) ** 2)
        return AD(val, der, self.name)


    def sqrt(self):
        """
        Compute the square root of an AD object.

        Returns
        -------
        AD object representing the result of sigmoid(self)

        Examples
        --------
        >>> x = AD(25,1,'x')
        >>> x.sqrt()
        Numerical Value is:
        [[5.]],
        Jacobian is:
        [[0.1]],
        Name is:
        ['x']

        >>> x = AD(4,1,'x')
        >>> y = AD(9,1,'y')
        >>> z = AD([x,y])
        >>> z.sqrt()
        Numerical Value is:
        [[2.]
         [3.]],
        Jacobian is:
        [[0.25       0.        ]
         [0.         0.16666667]],
        Name is:
        ['x', 'y']
        """

        # if (self.val < 0).any():
          #  raise ValueError('Square root should only be considered for positive numbers')
        # new_val = np.sqrt(self.val.copy())
        # if (self.val == 0).any():
          #  raise ValueError('The derivative of the square root can only be computed for strictly positive numbers')
        # new_der = 1 / 2 * self.der * (self.val ** -0.5)
        #  return AD(new_val, new_der, self.name)
        return self.__pow__(0.5)
