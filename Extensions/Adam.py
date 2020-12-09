import numpy as np
from AD.ADmulti import AD
from Extensions.Sampling_with_Optimization import annealing, produce_random_point


names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w']


def adam_gd(f, max_iter=1e6, init_points=None, tolerance=1e-6, random_restarts=2, beta_1=0.9, beta_2=0.999, step_size=1e-3,
            epsilon=1e-8, tuning=False):
    # docstrings
    # input_testing
    # dimensionality testing
    # one particularity here, is that we functions we will optimize are real value functions
    # so it's always several inputs, one output
    # furthermore, if we have vector inputs, they should always have the same size
    # input for function: either one input vector or several functions
    # justification of the default values
    # init_point should be a list
    """Run the gradient descent algorithm to minimize a function, with the Adam modification for the adaptation
    of the learning rate.

        INPUTS
        =======
        f: function, takes arbitrary number of scalar inputs, or a vector input.
            The function we wish to minimize
        init_points: list, optional, default value is None
            The initial points from which you want to perform your descent. If it not specified by the user,
            the algorithm will create some. By default, these points are drawn from a multivariate gaussian centered
            at the origin of the input space.
        max_iter: int, optional, default value is 1e6
            The max number of updates you wish to perform during your descent algorithm
        tolerance: int, optional, default value is 1e-6
            Threshold on updates performed on the weight parameter. This allows to perform early stopping for the
            algorithm in zones where the weights are not updated anymore
        random_restarts: int, optional, default value is 2
            Number of starts the user wants to restart the descent from a different random point. The final optimal
            point selected will be the best of the different (random_restarts) points
        beta_1: float, optional, default value is 0.9
            Momentum on the first moment of the gradient. Default value selected accordingly to Tensorflow's algorithms
        beta_2: float, optional, default value is 0.99
            Momentum on the second moment of the gradient. Default value selected accordingly to Tensorflow's algorithms
        step_size: float, optional, default value is 1e-3
            Static step size wanted during the descent.
        epsilon: float, optional, default value is 1e-8
            Noise introduced in order to prevent the involved denominators from being 0
        tuning: boolean, optional, default value is False
            Future feature to be developed. This will allow to tune the initial point before starting the descent.
            The algorithm leveraged is simulated annealing, which allow to sample points from the 'interesting' zones
            in the optimization landscape of a function.

        RETURNS
        ========
        final_values: list of values
           Has the form (x, y, z) where this point is the best point found across the different random restarts

        EXAMPLES
        =========
        >>> f = lambda x: x**2+y**2+(z-2)**2
        >>> acc = adam_gd(f, random_restarts=10)
        [-3.1341273654385723e-06, 1.9563233193181366e-59, 1.9998541875049667]
    """
    accumulator = []
    final_values = []
    for i in range(random_restarts):
        try:
            init_point = init_points[i]
        except:
            init_point = produce_random_point(f)
        if tuning:
            init_point = annealing(f, init_point=init_point)
        w = [AD(init_point[i], 1, name=names[i]) for i in range(len(init_point))]  # w should be an AD variable
        m = 0
        v = 0  # same as tf
        for t in range(1, int(max_iter)):
            AD_function = f(*w)
            gradient = AD_function.der
            m = beta_1 * m + (1 - beta_1) * gradient
            v = beta_2 * v + (1 - beta_2) * np.power(gradient, 2)
            m_hat = m / (1 - np.power(beta_1, t))  # unbiased estimator
            v_hat = v / (1 - np.power(beta_2, t))  # unbiased estimator
            update = step_size * m_hat / (np.sqrt(v_hat) + epsilon)
            if np.sqrt(np.sum(update**2)) < tolerance:
                break
            w = w - update  # weight update
            w = w[0]
        final_value = [float(w.val) for w in w]
        accumulator.append(f(*final_value))
        final_values.append(final_value)
    best_value = np.argmin(accumulator)
    return final_values[best_value]


#if __name__ == '__main__':
#    f = lambda x, y, z: x ** 2 + y ** 2 + (z - 2) ** 2
#    acc = adam_gd(f, random_restarts=10)
#    print(acc)

