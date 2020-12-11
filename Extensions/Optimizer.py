import numpy as np
from AD.ADmulti import AD
import inspect
import math

names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
         'w']
# todo: names inside the class and initialize with numbers a1


class Optimizer:

    """This class implements different types of Optimizers. Their common paradigm is gradient descent. The
    way these optimizers differ is the way the updates are performed, and whether they include momentum
    control or second moment of the gradient control."""

    def __init__(self, f, max_iter=1e6, init_points=None, dimensions=0, tolerance=1e-6, random_restarts=2, beta_1=0.9, beta_2=0.999,
                 step_size=1e-3, epsilon=1e-8, tuning=False, quadratic_matrix=None, max_epochs=10, verbose=0):
        """f: function, takes arbitrary number of scalar inputs, or a vector input.
            The function we wish to minimize
        init_points: list, optional, default value is None
            The initial points from which you want to perform your descent. If it not specified by the user,
            the algorithm will create some. By default, these points are drawn from a multivariate gaussian centered
            at the origin of the input space.
        dimensions: int, default value is 0
            This allow the optimizer to understand that the input is a vector, and the user specifies the size of the vector
            he wishes to get back from the optimizer. Is not necessary when init_points is input.
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
        if not isinstance(max_iter, int) and not isinstance(max_iter, float):
            raise TypeError('Max iter should be an integer or a float')
        if max_iter < 0:
            raise ValueError('Max iter should be a > 0 integer')
        self.max_iter = max_iter
        if isinstance(init_points, int) or isinstance(init_points, float):
            init_points = [init_points]
        if init_points is not None and not isinstance(init_points, list):
            raise TypeError('Initial points should be a list')
        if init_points is None:
            self.dimension = dimensions
            if self.dimension > 2:
                self.vectorize = True
            else:
                self.vectorize = False
        if init_points is not None:
            try:
                f(*init_points)
                self.vectorize = False
            except TypeError:
                try:
                    f(np.array(init_points))
                    self.vectorize = True
                    self.dimension = len(init_points)
                except TypeError:
                    raise ValueError('Please enter valid input points')
        self.init = init_points
        if not isinstance(tolerance, float) and not isinstance(tolerance, int):
            raise TypeError('Tolerance should be an integer or a float')
        if tolerance < 0:
            raise ValueError('Tolerance should be > 0')
        self.tol = tolerance
        if not isinstance(random_restarts, int):
            raise TypeError('Random restarts should be an integer')
        if random_restarts < 0:
            raise ValueError('Random restarts should be > 0')
        self.restarts = random_restarts
        if not isinstance(step_size, float) and not isinstance(step_size, int):
            raise TypeError('Step size should be an integer or a float')
        if step_size < 0:
            raise ValueError('Step size should be > 0')
        self.lr = step_size
        if not isinstance(beta_1, int) and not isinstance(beta_1, float):
            raise TypeError('Beta_1 should be an integer or a float')
        if beta_1 > 1 or beta_1 < 0:
            raise ValueError('Beta_1 should be an integer or a float between 0 and 1')
        self.beta1 = beta_1
        if not isinstance(beta_2, int) and not isinstance(beta_2, float):
            raise TypeError('Beta_2 should be an integer or a float')
        if beta_2 > 1 or beta_2 < 0:
            raise ValueError('Beta_1 should be an integer or a float between 0 and 1')
        self.beta2 = beta_2
        if not isinstance(epsilon, int) and not isinstance(epsilon, float):
            raise TypeError('Epsilon should be an integer or a float')
        self.eps = epsilon
        if not isinstance(tuning, bool):
            raise TypeError('Tuning should be a boolean')
        self.sampling = tuning
        self.global_optimizer = None
        if not self.sampling and quadratic_matrix is not None:
            raise ValueError('You should set tuning to true when inputing a covariance matrix')
        if quadratic_matrix is not None and not isinstance(quadratic_matrix, np.ndarray):
            raise TypeError('Quadratic matrix should be an array')
        if quadratic_matrix is not None and len(quadratic_matrix.shape) < 2:
            raise ValueError('Quadratic matrix should be a 2D array')
        if quadratic_matrix is not None and init_points is not None:
            if quadratic_matrix.shape[0] != len(init_points):
                raise ValueError('Input dimensions not coherent for cov matrix and init points')
        self.covariance = quadratic_matrix
        if not isinstance(max_epochs, int):
            raise TypeError('Epochs should be an integer')
        if max_epochs < 0:
            raise ValueError('The number of epochs should be an integer > 0')
        self.max_epochs = max_epochs
        if not isinstance(verbose, int):
            raise TypeError('Verbose should be an integer')
        if verbose < 0:
            raise ValueError('The verbose should be an integer > 0')
        self.verbose = verbose
        self.num_iterations = []

    def __str__(self):
        if self.global_optimizer:
            return 'Adam optimizer, which found an optimal weight point of ' + str(
                self.global_optimizer) + ' the value of the objective function at this point is ' + str(
                self.init_function(*self.global_optimizer))
        else:
            return 'Adam optimizer, not yet fitted'

    def produce_random_points(self):
        """This function allows to produce random initialization points in order to start the descent algorithm when
        the user does not wish to specify some initialization points. This function infer the dimensionality of
        the inputs and returns points sampled from a gaussian distribution. Returns, init_point, array of shape (1, n_args)
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
        # >>> sgd = sgd(f)
        # >>> init = sgd.produce_random_points()
        [-1.4738622, 1.0198161, -0.2515112]"""
        if self.vectorize:
            if self.dimension > 2:  # the user did not specify any input points
                dimensionality = self.dimension
                sample_produced = np.random.multivariate_normal(np.zeros(dimensionality), np.eye(dimensionality))
        else:
            n_args = len(inspect.getargspec(self.init_function).args)
            test_dim_input = np.ones(n_args).reshape(n_args, 1)
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
        for _ in range(self.restarts):
            if self.init:
                init_point = self.init
            else:
                init_point = self.produce_random_points()
            if self.sampling:
                init_point = self.annealing(init_point)
            w = [AD(init_point[i], 1, name=names[i]) for i in range(len(init_point))] # w should be an AD variable
            if self.vectorize:
                w = np.array(w)
            m = 0
            v = 0  # same as tf
            for t in range(1, int(self.max_iter)):
                if not self.vectorize:
                    AD_function = self.init_function(*w)
                else:
                    AD_function = self.init_function(w)
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
            if not self.vectorize:
                accumulator.append(self.init_function(*final_value))
            else:
                accumulator.append(self.init_function(np.array(final_value)))
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


    def annealing(self, init_point):
        """ This function allows to produce points inside regions of importance in the optimization landscape of f.
            It enables to tune the initialized points entered by the user, or it allows to produce good points when the user
            did not input any initialized points. This allows to accelerate the optimization algorithm.
            For now, it only supports quadratic functions.
            INPUTS
            =======
            self: an instance of optimizer class
            init_point: the initial point, produced via self.produce_random_points() or input by user

            RETURNS
            ========
            accumulator[-1][-1]: list of values
               Has the form (x, y, z) where this point is the best point found accross the different epochs based
               on random sampling based on simulated annealing.

            EXAMPLES
            =========
            # >>> f = lambda x: x**2+y**2+(z-2)**2
            # >>> adam = Adam(f, random_restarts=10)
            # >>> init_point = adam.produce_random_points()
            # >>> init_point = adam.annealing(init_point)
            # >>> adam.init = init_point    #todo: check if this thing work
            # >>> adam.descent()"""
        if self.covariance is None:
            raise ValueError('You must enter your quadratic form matrix in order to perform efficient sampling')

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
                temp = reduce_temp(temp)  # cooling schedule
                length = incr_iters(length)   # favor exploitation over exploration over time
            for it in range(length):
                total += 1
                new_solution = np.random.multivariate_normal(old_solution, self.covariance, size=1).flatten()
                new_cost = self.init_function(*new_solution)
                alpha = min(1, np.exp((old_cost - new_cost) / temp))
                if (new_cost < old_cost) or (np.random.uniform() < alpha):
                    # print('new', new_cost, 'old', old_cost)
                    # print('Acceptance probability', alpha)
                    # print('new solution', new_solution)
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
        if self.vectorize:
            final_value = self.init_function(np.array(self.global_optimizer))
        else:
            final_value = self.init_function(*self.global_optimizer)
        if self.global_optimizer:
            return 'Adam optimizer, which found an optimal weight point of ' + str(
                self.global_optimizer) + ' the value of the objective function at this point is ' + str(
                final_value)
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


