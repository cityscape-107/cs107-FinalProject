import numpy as np
from AD.ADmulti import AD
import inspect
import math

names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
         'w']


class Optimizer:

    def __init__(self, f, max_iter=1e6, init_points=None, tolerance=1e-6, random_restarts=2, beta_1=0.9, beta_2=0.999,
                 step_size=1e-3, epsilon=1e-8, tuning=False, quadratic_matrix=None, max_epochs=10, verbose=0):
        """f: function, takes arbitrary number of scalar inputs, or a vector input.
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
            in the optimization landscape of a function. Right now, it is efficient for quadratic functions (that have
            the form x^T*A*x
        quadratic_matrix: numpy matrix, optional, default value is None
            The quadratic matrix for the definition of the quadratic form. This will allow to efficiently sample
            points that are interesting in the optimization landscape of the quadratic form.
        max_epochs: int, optional, default value is 100
            The number of times you wish to perform simulated annealing algorithm in order to get a good starting point.
            The higher the better
        verbose: integer, optional, default value is 0
            Value between 0 and 2 describing the amount of information we wish to be print during the optimization
            algorithm, for debugging issues"""
        self.init_function = f
        self.trace_values = []
        self.trace_gradients = []
        self.trace = []
        self.max_iter = max_iter
        self.init = init_points
        self.tol = tolerance
        self.restarts = random_restarts
        self.lr = step_size
        self.beta1 = beta_1
        self.beta2 = beta_2
        self.eps = epsilon
        self.sampling = tuning
        self.global_optimizer = None
        self.covariance = quadratic_matrix
        self.max_epochs = max_epochs
        self.verbose = verbose
        self.num_iterations = []

    def __str__(self):
        raise NotImplementedError

    def produce_random_points(self):
        """
        This function allows to produce random initialization points in order to start the descent algorithm when
        the user does not wish to specify some initialization points. This function infer the dimensionality of
        the inputs and returns points sampled from a gaussian distribution
        :return: init_point, array of shape (1, n_args)
        """
        n_args = len(inspect.getargspec(self.init_function).args)
        test_dim_input = np.ones(n_args).reshape(n_args, 1)
        for i in range(0, 100):
            try:
                self.init_function(*test_dim_input)
                break
            except:
                try:
                    self.init_function(np.array(*test_dim_input))
                    break
                except:
                    test_dim_input = np.concatenate((test_dim_input, np.ones(n_args)), axis=1)
                    continue
        dimensionality = len(test_dim_input)
        sample_produced = np.random.multivariate_normal(np.zeros(dimensionality), np.eye(dimensionality))
        return sample_produced

    def descent(self):
        """Run the gradient descent algorithm to minimize a function, with the Adam modification for the adaptation
        of the learning rate, or the sgd optimizer, depending on the values of beta1 and beta2.
        INPUTS
        =======
        self: an adam instance of the optimizer

        RETURNS
        ========
        final_values: list of values
           Has the form (x, y, z) where this point is the best point found across the different random restarts

        EXAMPLES
        =========
        # >>> f = lambda x: x**2+y**2+(z-2)**2
        # >>> acc = adam_gd(f, random_restarts=10)
        [-3.1341273654385723e-06, 1.9563233193181366e-59, 1.9998541875049667]"""
        accumulator = []
        final_values = []
        #todo: find a way to update the init points without having this line breaking
        for i in range(self.restarts):
            try:
                init_point = self.init[i]
            except:
                init_point = self.produce_random_points()
            if self.sampling:
                init_point = self.annealing(init_point)
            w = [AD(init_point[i], 1, name=names[i]) for i in range(len(init_point))]  # w should be an AD variable
            m = 0
            v = 0  # same as tf
            for t in range(1, int(self.max_iter)):
                AD_function = self.init_function(*w)
                self.trace_values.append(AD_function.val)
                gradient = AD_function.der
                self.trace_gradients.append(gradient)
                if self.beta1 != 0 and self.beta2 != 0:  # want to perform a adam gradient descent
                    m = self.beta1 * m + (1 - self.beta1) * gradient
                    v = self.beta2 * v + (1 - self.beta2) * np.power(gradient, 2)
                    m_hat = m / (1 - np.power(self.beta1, t))  # unbiased estimator
                    v_hat = v / (1 - np.power(self.beta2, t))  # unbiased estimator
                    update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
                if self.beta1 == 0 and self.beta2 != 0:  # want to perform RMS prop
                    v = self.beta2 * v + (1 - self.beta2) * np.power(gradient, 2)
                    v_hat = v / (1 - np.power(self.beta2, t))  # unbiased estimator
                    update = self.lr * gradient / (np.sqrt(v_hat) + self.eps)
                elif self.beta1 == 0 and self.beta2 == 0:  # want to perform single sgd/gd without momentum
                    update = self.lr * gradient
                if np.sqrt(np.sum(update ** 2)) < self.tol:
                    break
                w = w - update  # weight update
                w = w[0]
                self.trace.append(w)
            self.num_iterations.append(t)
            final_value = [float(w.val) for w in w]
            accumulator.append(self.init_function(*final_value))
            final_values.append(final_value)
            if self.verbose >= 2:
                print('Descent algorithm performed, the optimal point found is ' + str(
                    final_value) + 'with a value of ' + str(accumulator[-1]) + 'after ' + str(self.num_iterations[
                          -1]) + 'iterations')

        best_value = np.argmin(accumulator)
        self.global_optimizer = final_values[best_value]
        if self.verbose >= 1:
            print('Descent algorithm performed, the optimal point found across ' + str(
                self.restarts) + 'different descents is ' + str(self.global_optimizer) + 'with a value of ' + str(
                np.min(accumulator)) + 'after ' + str(self.num_iterations[best_value]) + ' iterations')
        return self.trace_values

# todo: change the syntax into a class for annealing
    def annealing(self, init_point):
        """ This function allows to produce points inside regions of importance in the optimization landscape of f.
            It enables to tune the initialized points entered by the user, or it allows to produce good points when the user
            did not input any initialized points. This allows to accelerate the optimization algorithm.
            For now, it only supports quadratic functions."""
        if self.covariance is None:
            raise Exception('You must enter your quadratic form matrix in order to perform efficient sampling')

        temp = 10  # common default value when using this cooling schedule
        length = 500

        def reduce_temp(temp):
            return 0.8 * temp

        def incr_iters(length):
            return int(math.ceil(1.2 * length))

        old_solution = init_point
        old_cost = self.init_function(*old_solution)
        accepted = 0
        total = 0
        accumulator = []
        for epoch in range(self.max_epochs):
            if epoch > 0:
                temp = reduce_temp(temp)
                length = incr_iters(length)
            for it in range(length):
                total += 1
                new_solution = np.random.multivariate_normal(old_solution, self.covariance, size=1).flatten()
                new_cost = self.init_function(*new_solution)
                alpha = min(1, np.exp((old_cost - new_cost) / temp))
                if (new_cost < old_cost) or (np.random.uniform() < alpha):
                    accepted += 1
                    accumulator.append([temp, new_solution, new_cost])

                    old_cost = new_cost
                    old_solution = new_solution

                else:
                    # Keep the old stuff
                    accumulator.append([temp, old_solution, old_cost])
        if self.verbose == 2:
            if accepted * 1. / total < 0.1:
                print('The sampling has not yeat converged, you might want to increase the number of epochs')
            print('Initialization point before simulated annealing: '+str(self.init))
            print('Initialization point after simulated annealing: ' + str(accumulator[-1][1]))
        return accumulator[-1][1]

    def history(self):
        if self.global_optimizer is None:
            raise Exception('Optimization not run yet. Please run your optimizer before accessing to its history')
        return self.__str__()


class Adam(Optimizer):
    """
    This Adam class inherits the Optimizer class, leveraging the default values we defined for the parameters beta1 and
    beta2, which were, for the sake of simplicity, (since the users use mostly adam) the coefficients for the adam optimizer.
    """

    def __str__(self):
        if self.global_optimizer:
            return 'Adam optimizer, which found an optimal weight point of ' + str(
                self.global_optimizer) + ' the value of the objective function at this point is ' + str(
                self.init_function(*self.global_optimizer))
        else:
            return 'Adam optimizer, not yet fitted'


class sgd(Optimizer):
    """
    This sgd class inherits the Optimizer class, redefining the values for beta1 and beta2. For sgd, we do not want
    to leverage momentum, we just want to perform classic gradient descent, with a smaller batch size (addendum to be done later on).
    """

    def __init__(self, f, max_iter=1e6, init_points=None, tolerance=1e-6, random_restarts=2, beta_1=0.9, beta_2=0.999,
                 step_size=1e-3, epsilon=1e-8, tuning=False, verbose=0):
        super(sgd, self).__init__(f, max_iter=1e6, init_points=None, tolerance=1e-6, random_restarts=2, beta_1=0.9,
                                  beta_2=0.999,
                                  step_size=1e-3, epsilon=1e-8, tuning=False, verbose=0)
        self.beta1 = 0
        self.beta2 = 0

    def __str__(self):
        if self.global_optimizer:
            return 'Sgd optimizer, which found an optimal weight point of ' + str(
                self.global_optimizer) + ' the value of the objective function at this point is ' + str(
                self.init_function(*self.global_optimizer))
        else:
            return 'Sgd optimizer, not yet fitted'


class RMSProp(Optimizer):
    """
    This RMSprop class inherits the Optimizer class, redefining the values for beta1. For RMSProp, we do not
    want to put momentum on the gradient update, but which to have an adaptive learning rate depending on the
    accumulated squared gradients until then
    """

    def __init__(self, f, max_iter=1e6, init_points=None, tolerance=1e-6, random_restarts=2, beta_1=0.9, beta_2=0.999,
                 step_size=1e-3, epsilon=1e-8, tuning=False, verbose=0):
        super(RMSProp, self).__init__(f, max_iter=1e6, init_points=None, tolerance=1e-6, random_restarts=2, beta_1=0.9,
                                      beta_2=0.999,
                                      step_size=1e-3, epsilon=1e-8, tuning=False, verbose=0)
        self.beta1 = 0

    def __str__(self):
        if self.global_optimizer:
            return 'RMSProp optimizer, which found an optimal weight point of ' + str(
                self.global_optimizer) + ' the value of the objective function at this point is ' + str(
                self.init_function(*self.global_optimizer))
        else:
            return 'RMSProp optimizer, not yet fitted'


if __name__ == '__main__':
    f = lambda x, y, z: x ** 2 + y ** 2 + z ** 2
    adam = Adam(f, random_restarts=10, tuning=True, quadratic_matrix=np.eye(3), verbose=2)
    adam.descent()
    print(adam)
    adam = Adam(f, random_restarts=10, verbose=2)
    adam.descent()
    print(adam)

