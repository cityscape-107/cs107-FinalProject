import numpy as np
import inspect


def produce_temp():
    return [10 / np.log(n) for n in range(20, 500, 20)]

def produce_random_point(cost_function):
    n_args = len(inspect.getargspec(cost_function).args)
    test_dim_input = np.ones(n_args).reshape(n_args, 1)
    for i in range(0, 100):
        try:
            cost_function(*test_dim_input)
            break
        except:
            try:
                cost_function(np.array(*test_dim_input))
                break
            except:
                test_dim_input = np.concatenate((test_dim_input, np.ones(n_args)), axis=1)
                continue
    dimensionality = len(test_dim_input)
    sample_produced = np.random.multivariate_normal(np.zeros(dimensionality), np.eye(dimensionality))
    return sample_produced

# todo : finish writing the simulated annealing function, which will be an extra feature
def annealing(cost_function, temperatures=None, maxsteps=100, init_point=None):
    """ This function allows to produce points inside regions of importance in the optimization landscape of f.
    For now, it only supports quadratic functions."""
    if init_point is None:
        init_point = produce_random_point(cost_function)
    if temperatures is None:
        temperatures = produce_temp()
    for step in range(maxsteps):
        temperature = temperatures[step]
        new_proposal = 'pass'

    return 0