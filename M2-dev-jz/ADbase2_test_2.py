import math

from ADbase2 import AD


def DeriveAlive_Var():
    '''Test constructor of Var class to ensure proper variable initializations.'''

    def test_scalar_without_bracket():
        x = AD(1)
        assert x.val == 1
        assert x.der == 1

    def test_scalar_with_bracket():
        x = AD([1])
        assert x.val == 1
        assert x.der == 1

    def test_input():
        x = AD(2, 1)
        y = AD(3, 1)
        f = x + y
        print(f)

    def test_with_preset_der():
        der = 3.5
        x = AD(2, der)
        assert x.val == 2
        assert x.der == der

    # Run tests within test_DeriveAliveVar
    test_scalar_without_bracket()
    test_scalar_with_bracket()
    test_input()
    test_with_preset_der()


def AD_add_Var():
    def test_addition_1():
        x = AD(10, 5)
        y = AD(12, 3)
        z = x + y
        assert z.val == [22]
        assert z.der == [8]

    def test_addition_2():
        x = AD(-10, -5)
        y = AD(12, 3)
        z = x + y
        assert z.val == [2]
        assert z.der == [-2]

    def test_addition_3():
        x = AD(-10, -5)
        y = 4
        z = x + y
        assert z.val == [-6]
        assert z.der == [-5]

    def test_addition_4():
        x = AD(-10, -5)
        y = 4
        z = y + x
        assert z.val == [-6]
        assert z.der == [-5]

    test_addition_1()
    test_addition_2()
    test_addition_3()
    test_addition_4()


def AD_neg_VAR():
    def test_negation_1():
        x = AD(12, 20)
        z = -x
        assert z.val == -12
        assert z.der == -20

    def test_negation_2():
        x = AD(12, None)
        y = -x
        assert y.val == -12
        assert y.der == None

    test_negation_1()
    test_negation_2()


def AD_sub_Var():
    def test_substraction_1():
        x = AD(10, 5)
        y = AD(12, 3)
        z = x - y
        assert z.val == [-2]
        assert z.der == [2]

    def test_substraction_2():
        x = AD(-10, -5)
        y = AD(12, 3)
        z = x - y
        assert z.val == [-22]
        assert z.der == [-8]

    def test_substraction_3():
        x = AD(-10, -5)
        y = 4
        z = x - y
        assert z.val == [-14]
        assert z.der == [-5]

    def test_substraction_4():
        x = AD(-10, -5)
        y = 4
        z = y - x
        assert z.val == [14]
        assert z.der == [5]

    test_substraction_1()
    test_substraction_2()
    test_substraction_3()
    test_substraction_4()


def AD_mul_var():
    def test_multiplication_1():
        x = AD(10, 2)
        y = AD(3, -4)
        z = x * y
        assert z.val == 30
        assert z.der == -34

    def test_multiplication_2():
        x = AD(10, 2)
        y = AD(3, -4)
        z = y * x
        assert z.val == 30
        assert z.der == -34

    def test_multilplication_3():
        x = AD(10, 2)
        y = 5
        z = x * y
        assert z.val == 50
        assert z.der == 10

    def test_multilplication_4():
        x = AD(10, 2)
        y = AD(5)
        z = y * x
        assert z.val == 50
        assert z.der == 10

    test_multiplication_1()
    test_multiplication_2()
    test_multilplication_3()
    test_multilplication_4()


def AD_div_var():
    def test_division_1():
        x = AD(10, 3)
        y = AD(2, 4)
        z = x / y
        assert z.val == 5
        assert z.der == -34 / 4

    def test_division_2():
        x = AD(10, 3)
        y = AD(2, 4)
        z = y / x
        assert z.val == 2 / 10
        assert z.der == 34 / 100

    def test_division_3():
        x = AD(10, 3)
        y = AD(0)
        try:
            z = x / y
            print(z)
        except ZeroDivisionError:
            print('Zero Division Error caught')

    def test_division_4():
        x = AD(0)
        y = AD(10, 3)
        z = x / y
        assert z.val == 0
        assert z.der == 0

    def test_division_5():
        x = AD(10, 3)
        y = 4
        z = x / y
        assert z.val == 2.5
        assert z.der == 0.75

    def test_division_6():
        x = AD(10, 3)
        y = 4
        z = y / x
        assert z.val == 0.4
        assert z.der == -0.12

    test_division_1()
    test_division_2()
    test_division_3()
    test_division_4()
    test_division_5()
    test_division_6()


def AD_pow_var():
    def test_pow_1():
        x = AD(4, 3)
        n = 3
        z = x ** n
        assert z.val == 64
        assert z.der == 144

    def test_pow_2():
        x = AD(-3, 2)
        n = 3
        z = x ** n
        assert z.val == -27
        assert z.der == 54

    def test_pow_3():
        try:
            x = AD(0, 12)
            n = -1
            z = x ** n
        except ZeroDivisionError:
            print('Succesfully caught')

    def test_pow_4():
        try:
            x = AD(-1, 13)
            n = 0.5
            z = x ** n
        except ValueError:
            print('Success')

    def test_pow_5():
        x = AD(2, 5)
        y = AD(3, 6)
        z = x ** y
        assert z.val == 8
        assert z.der == (6 * math.log(2) + 3 * 2.5) * 8

    def test_pow_6():
        x = 4
        y = AD(4, 5)
        z = x ** y
        assert z.val == 256
        assert z.der == 5 * math.log(4) * 256

    test_pow_1()
    test_pow_2()
    test_pow_3()
    test_pow_4()
    test_pow_5()
    test_pow_6()


if __name__ == '__main__':
    AD_add_Var()
    AD_sub_Var()
    AD_neg_VAR()
    AD_mul_var()
    AD_div_var()
    AD_pow_var()
