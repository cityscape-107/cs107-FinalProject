from AD.ADmulti import AD
from Extensions.Optimizer import *

"""
############################################################
###################### ROOT-FINDING ########################
############################################################
"""


def test_forward_init():
    print("Case: z = x**2 + sin(x) + ln(x)")
    print("Value=", 0.5)
    # Initiating the Forward mode
    x = AD(0.5, 1)
    # Function example
    z = x ** 2 + x.sin() + x.ln()
    print('z.der()=', z.der)
    print("_________________________")


test_forward_init()


def f1(x):
    return x ** 2


def f2(x):
    return x ** 2 + x.sin() + x.ln()


def test_forward_fn():
    print("Case: f2(x)=z")
    print("Value=", 0.5)
    x = AD(0.5, 1)
    z1 = f2(x)
    print('z1.der=', z1.der)
    print("_________________________")


test_forward_fn()


def NRFM(f, xn, tol):
    # Initializing
    x = AD(xn, 1)
    y = f(x)
    xn = x.val - y.val / y.der

    while abs(x.val - xn) > tol:
        x = AD(xn, 1)
        y = f(x)
        xn = x.val - y.val / y.der

    return xn


def test_f1():
    print("f=x**2")
    x0 = 0.5
    tol = 0.001
    print("x_0:{}, tol:{}".format(x0, tol))
    trial = NRFM(f1, x0, tol)
    print('x=', trial)
    print('f1(x)=', f1(trial))
    print("_________________________")


test_f1()


def test_f2():
    print("f2=x**2 + sin(x) + ln(x)")
    x0 = 0.05
    tol = 0.001
    print("x_0:{}, tol:{}".format(x0, tol))
    trial = NRFM(f2, x0, tol)
    print('x_final=', trial)
    print('f2(x)=', f2(AD(trial, 1)).val)
    print("_________________________")


test_f2()

"""
############################################################
###################### OPTIMIZATION ########################
############################################################
"""


def test_scalar_input():
    f = lambda x: x ** 2 + 2
    adam = Adam(f, random_restarts=10)
    adam.descent()
    z = adam.global_optimizer
    print('The optimal point found for minimization is ', *z, 'which value is ', f(*z))


test_scalar_input()


def test_two_scalar_inputs():
    f = lambda x, y: x ** 2 + 2 + (y - 3) ** 2
    adam = Adam(f, random_restarts=10)
    adam.descent()
    z = adam.global_optimizer
    print('The optimal point found for minimization is ', *z, 'which value is ', f(*z))


test_two_scalar_inputs()


def test_with_init_points():
    f = lambda x, y: x ** 2 + 2 + (y - 3) ** 2
    adam = Adam(f, random_restarts=10, init_points=[4, -2])
    adam.descent()
    z = adam.global_optimizer
    print('The optimal point found for minimization is ', *z, 'which value is ', f(*z))


test_with_init_points()


def test_with_annealing():
    f = lambda x, y, z: x ** 2 + y ** 2 + z ** 2
    adam = Adam(f, random_restarts=10, quadratic_matrix=np.eye(3), tuning=True)
    adam.descent()
    z = adam.global_optimizer
    print('The optimal point found for minimization is ', *z, 'which value is ', f(*z))


test_with_annealing()


def test_vector():
    f = lambda v: np.sum(v ** 2)
    adam = Adam(f, random_restarts=10, dimensions=20)
    adam.descent()
    z = adam.global_optimizer
    print('The optimal point found for minimization is ', *z, 'which value is ', f(np.array(z)))


test_vector()


