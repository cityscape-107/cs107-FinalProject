import math
# this is where we are going to work on creating the new class
import numpy as np


class AD:

    def __init__(self, value, der=0, name="x"):  # changed from der=[0]
        if isinstance(value, float) or isinstance(value, int):
            value = np.array(value).reshape(1, 1)
        elif isinstance(value, np.ndarray):
            value = value
        else:  # NEW
            raise TypeError(f'Value should be int or float and not {type(value)}')

        # if not isinstance(name, str):
        #    raise TypeError('Please enter a valid name')  # todo: type error on every input of the list

        if isinstance(der, float) or isinstance(der, int):
            der = np.array([der]).reshape(1, 1)
        elif isinstance(der, np.ndarray):
            der = der
        else:  # NEW
            raise TypeError(f'Derivative should be int or float and not {type(der)}')

        self.val = np.array(value, dtype=np.float64)
        self.der = np.array(der, dtype=np.float64)
        if isinstance(name, list):
            self.name = name
        else:
            self.name = [name]

    def __repr__(self):  # todo: pb with __repr__, solved
        val = self.val
        der = self.der
        name = self.name
        return 'Numerical Value is:\n{}, \nJacobian is:\n{}, \nName is:\n{}'.format(val, der, name)

    # overloading the '+' operator
    def __add__(self, other):

        if isinstance(other, AD):
            names_1 = self.name
            names_2 = other.name
            value = self.val + other.val
            derivative = self.der.copy()
            for name_2 in names_2:
                index_2 = names_2.index(name_2)
                if name_2 in names_1:
                    index_1 = names_1.index(name_2)
                    derivative[:, index_1] = derivative[:, index_1] + other.der[:, index_2]
                else:
                    derivative = np.concatenate((derivative, other.der[:, index_2].reshape(-1, 1)), axis=1)
                    names_1.append(name_2)
            name = names_1
        else:
            value = self.val + other
            derivative = self.der
            name = self.name
        return AD(value, derivative, name)

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
        if isinstance(other, AD):
            names_1 = self.name
            names_2 = other.name
            new_value = self.val * other.val  # numpy array multiplication
            new_names = names_1 + [name for name in other.name if name not in names_1]
            derivative = self.der.copy()
            for name in new_names:
                # print('Current name', name)
                # print('List of names for 1',names_1)
                # print('List of names for 2', names_2)
                if name in names_1 and name in names_2:
                    index_1 = names_1.index(name)
                    index_2 = names_2.index(name)
                    new_der = self.val * other.der[:, index_2] + self.der[:, index_1] * other.val
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
                    # print(derivative)
            name = new_names

        else:  # one of the coefficients of other is None, it is a constant
            new_value = self.val * other
            derivative = self.der * other
            name = self.name
        return AD(new_value, derivative, name)

    def __rmul__(self, other):
        return AD(self.val, self.der).__mul__(other)


    def __truediv__(self, other):
        if isinstance(other, AD):
            names_1 = self.name
            names_2 = other.name
            new_value = self.val / other.val  # numpy array multiplication
            new_names = names_1 + [name for name in other.name if name not in names_1]
            derivative = self.der.copy()
            for name in new_names:
                # print('Current name', name)
                # print('List of names for 1', names_1)
                # print('List of names for 2', names_2)
                if name in names_1 and name in names_2:
                    # print('We are here working with ', name)
                    index_1 = names_1.index(name)
                    # print('Index 1', index_1)
                    index_2 = names_2.index(name)
                    # print('Index 2', index_2)
                    new_der = (self.der[:, index_1]*other.val - other.der[:, index_2]*self.val)/other.val**2  # with scalars
                    # print('The derivative according to ', name, 'is ', new_der)
                    # print('Before update', derivative)
                    derivative[:, index_1] = new_der
                    # print(derivative)
                if name in names_1 and name not in names_2:
                    # print('I am here')
                    index_1 = names_1.index(name)
                    new_der = self.der[:, index_1]/other.val
                    # print(new_der)
                    derivative[:, index_1] = new_der
                    # print(derivative)
                if name in names_2 and name not in names_1:
                    # print('Now I am here')
                    index_2 = names_2.index(name)
                    new_der = -self.val*other.der[:, index_2]/other.val**2
                    # print(new_der)
                    # print('Before the update', derivative)
                    derivative = np.concatenate((derivative, new_der), axis=1)
                    # print(derivative)
            name = new_names
        else:
            new_value = self.val/other
            derivative = self.der/other
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
        # der = np.multiply(np.power(1 / np.cos(self.val), 2), self.der)
        der = np.multiply(1 / np.power(np.cos(self.val), 2), self.der)
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
        if self.val <= 0:
            raise ValueError("Cannot take natural log of zero or negative values")
        val = np.log(self.val)
        der = 1 / self.val
        return AD(val, der)

    # add log to other bases
