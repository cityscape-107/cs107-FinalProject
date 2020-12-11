from Extensions.Optimizer import *
import pytest


def test_init_type_error_iter():
    f = lambda x: x
    max_iter = 'hey'
    with pytest.raises(TypeError):
        opt = Optimizer(f=f, max_iter=max_iter)


def test_init_value_error_iter():
    f = lambda x: x
    max_iter = -1
    with pytest.raises(ValueError):
        opt = Optimizer(f=f, max_iter=max_iter)


def test_init_type_error_init_points():
    f = lambda x, y: x + y
    init_points = 'hey'
    with pytest.raises(TypeError):
        opt = Optimizer(f=f, init_points=init_points)


def test_init_without_list():
    f = lambda x: x
    init_point = 2
    opt = Optimizer(f=f, init_points=init_point)


def test_init_value_error_init_points():
    f = lambda x, y: x + y
    init_points = [1]
    with pytest.raises(ValueError):
        opt = Optimizer(f=f, init_points=init_points)


def test_init_type_error_tol():
    f = lambda x: x
    tol = 'hey'
    with pytest.raises(TypeError):
        opt = Optimizer(f=f, tolerance=tol)


def test_init_value_error_tol():
    f = lambda x: x
    tol = -1
    with pytest.raises(ValueError):
        opt = Optimizer(f=f, tolerance=tol)


def test_init_type_error_restarts():
    f = lambda x: x
    restarts = 'hey'
    with pytest.raises(TypeError):
        opt = Optimizer(f=f, random_restarts=restarts)


def test_init_value_error_restarts():
    f = lambda x: x
    restarts = -1
    with pytest.raises(ValueError):
        opt = Optimizer(f=f, random_restarts=restarts)


def test_init_type_error_lr():
    f = lambda x: x
    lr = 'hey'
    with pytest.raises(TypeError):
        opt = Optimizer(f=f, step_size=lr)


def test_init_value_error_lr():
    f = lambda x: x
    lr = -1
    with pytest.raises(ValueError):
        opt = Optimizer(f=f, step_size=lr)


def test_init_type_error_beta1():
    f = lambda x: x
    beta_1 = 'hey'
    with pytest.raises(TypeError):
        opt = Optimizer(f=f, beta_1=beta_1)


def test_init_type_error_beta2():
    f = lambda x: x
    beta_2 = 'hey'
    with pytest.raises(TypeError):
        opt = Optimizer(f=f, beta_2=beta_2)


def test_init_value_error_beta1():
    f = lambda x: x
    beta_1 = -1
    with pytest.raises(ValueError):
        opt = Optimizer(f=f, beta_1=beta_1)


def test_init_value_error_beta2():
    f = lambda x: x
    beta_2 = 3
    with pytest.raises(ValueError):
        opt = Optimizer(f=f, beta_2=beta_2)


def test_init_type_error_epsilon():
    f = lambda x: x
    epsilon = 'hey'
    with pytest.raises(TypeError):
        opt = Optimizer(f=f, epsilon=epsilon)


def test_init_type_error_sampling():
    f = lambda x: x
    sampling = 'hey'
    with pytest.raises(TypeError):
        opt = Optimizer(f=f, tuning=sampling)


def test_init_value_error_sampling():
    f = lambda x: x
    sampling = False
    covariance_matrix = np.eye(2)
    with pytest.raises(ValueError):
        opt = Optimizer(f=f, tuning=sampling, quadratic_matrix=covariance_matrix)


def test_init_dimensions():
    f = lambda x: x
    init_points = [1, 2]
    sampling = True
    cov_matrix = np.eye(2)
    opt = Optimizer(f=f, tuning=sampling, init_points=init_points, quadratic_matrix=cov_matrix)
    assert opt.vectorize == True
    assert opt.dimension == 2


def test_init_quad_type_error():
    f = lambda x: x
    init_points = [1]
    sampling = True
    cov_matrix = [1, 2]
    with pytest.raises(TypeError):
        opt = Optimizer(f=f, tuning=sampling, init_points=init_points, quadratic_matrix=cov_matrix)


def test_init_quad_value_error():
    f = lambda x, y: x + y
    init_points = [1, 2]
    sampling = True
    cov_matrix = np.array([1, 2])
    print(len(cov_matrix.shape))
    with pytest.raises(ValueError):
        opt = Optimizer(f=f, tuning=sampling, init_points=init_points, quadratic_matrix=cov_matrix)


def test_init_quad_value_error_2():
    f = lambda x, y: x + y
    init_points = [1, 2]
    sampling = True
    cov_matrix = np.eye(3)
    print(len(cov_matrix.shape))
    with pytest.raises(ValueError):
        opt = Optimizer(f=f, tuning=sampling, init_points=init_points, quadratic_matrix=cov_matrix)


def test_init_epochs_type_error():
    f = lambda x: x
    epochs = 'hey'
    with pytest.raises(TypeError):
        opt = Optimizer(f=f, max_epochs=epochs)


def test_init_epochs_value_error():
    f = lambda x: x
    epochs = -1
    with pytest.raises(ValueError):
        opt = Optimizer(f=f, max_epochs=epochs)


def test_init_verbose_type_error():
    f = lambda x: x
    verbose = 'hey'
    with pytest.raises(TypeError):
        opt = Optimizer(f=f, verbose=verbose)


def test_init_verbose_value_error():
    f = lambda x: x
    verbose = -1
    with pytest.raises(ValueError):
        opt = Optimizer(f=f, verbose=verbose)


def test_init_without_sampling():
    f = lambda x: x
    max_iter = 10
    init_points = [1]
    tolerance = 1
    random_restarts = 2
    beta_1 = 0.5
    beta_2 = 0.1
    step_size = 1e-2
    epsilon = 1e-4
    tuning = False
    verbose = 0
    opt = Optimizer(f=f, max_iter=max_iter, init_points=init_points, tolerance=tolerance,
                    random_restarts=random_restarts, beta_1=beta_1, beta_2=beta_2, step_size=step_size,
                    epsilon=epsilon, tuning=tuning, verbose=verbose)


def test_produce_random_sample():
    f = lambda x: x ** 2
    Opt = Optimizer(f)
    init_points = Opt.produce_random_points()
    print(init_points)
    assert init_points.shape[0] == 1


def test_dimensionality_2():
    f = lambda x, y: x + y
    Opt = Optimizer(f)
    init_points = Opt.produce_random_points()
    assert init_points.shape[0] == 2


def test_produce_random_sample_2():
    f = lambda x, y, z: x * y + z
    opt = Optimizer(f)
    init_points = opt.produce_random_points()
    assert init_points.shape[0] == 3


def test_descent_1():
    f = lambda x, y, z: x ** 2 + y ** 2 + (z - 2) ** 2
    opt = Optimizer(f)
    opt.descent()
    optimal_point = opt.global_optimizer
    print(optimal_point)
    assert abs(optimal_point[0]) < 1e-3
    assert abs(optimal_point[1]) < 1e-3
    assert abs(optimal_point[2] - 2) < 1e-3


def test_descent_2():
    f = lambda x, y, z: x ** 2 + y ** 2 + z ** 2
    opt = Optimizer(f, quadratic_matrix=np.eye(3), tuning=True)
    opt.descent()
    optimal_point = opt.global_optimizer
    assert abs(optimal_point[0]) < 1e-3
    assert abs(optimal_point[1]) < 1e-3
    assert abs(optimal_point[2]) < 1e-3


def test_descent_2_verbose_1():
    f = lambda x, y, z: x ** 2 + y ** 2 + z ** 2
    opt = Optimizer(f, quadratic_matrix=np.eye(3), tuning=True, verbose=1)
    opt.descent()
    optimal_point = opt.global_optimizer
    assert abs(optimal_point[0]) < 1e-3
    assert abs(optimal_point[1]) < 1e-3
    assert abs(optimal_point[2]) < 1e-3


def test_descent_2_verbose_2():
    f = lambda x, y, z: x ** 2 + y ** 2 + z ** 2
    opt = Optimizer(f, quadratic_matrix=np.eye(3), tuning=True, verbose=2)
    opt.descent()
    optimal_point = opt.global_optimizer
    assert abs(optimal_point[0]) < 1e-3
    assert abs(optimal_point[1]) < 1e-3
    assert abs(optimal_point[2]) < 1e-3


def test_descenterror_cov():
    f = lambda x, y, z: x ** 2 + y ** 2 + z ** 2
    with pytest.raises(ValueError):
        opt = Optimizer(f, quadratic_matrix=None, tuning=True, verbose=2)
        opt.descent()
        optimal_point = opt.global_optimizer


def test_adam_gd():
    f = lambda x: x ** 2 + 2
    adam = Adam(f)
    adam.descent()
    assert np.abs(adam.global_optimizer) < 1e-3


def test_adam():
    f = lambda x, y, z: x ** 2 + y ** 2 + z ** 2
    opt = Adam(f)
    opt.descent()
    np.testing.assert_allclose(opt.global_optimizer, np.array([0, 0, 0]), atol=1e-3)


def test_sgd():
    f = lambda x, y, z: x ** 2 + y ** 2 + z ** 2
    opt = sgd(f)
    opt.descent()
    np.testing.assert_allclose(opt.global_optimizer, np.array([0, 0, 0]), atol=1e-3)


def test_rms():
    f = lambda x, y, z: x ** 2 + y ** 2 + z ** 2
    opt = RMSProp(f)
    opt.descent()
    opt.history()
    np.testing.assert_allclose(opt.global_optimizer, np.array([0, 0, 0]), atol=1e-3)


def test_opt():
    f = lambda x, y, z: x ** 2 + y ** 2 + z ** 2
    opt = Optimizer(f)
    opt.descent()
    np.testing.assert_allclose(opt.global_optimizer, np.array([0, 0, 0]), atol=1e-3)


def test_opt():
    f = lambda x, y, z: x ** 2 + y ** 2 + z ** 2
    opt = Optimizer(f)
    opt.descent()
    print(opt)
    np.testing.assert_allclose(opt.global_optimizer, np.array([0, 0, 0]), atol=1e-3)


def test_3_opt():
    f = lambda x, y, z: x ** 2 + y ** 2 + (z - 2) ** 2
    adam = Adam(f, random_restarts=10)
    print(adam)
    adam.descent()
    print(adam)
    opt_sgd = sgd(f, random_restarts=10)
    print(opt_sgd)
    opt_sgd.descent()
    print(opt_sgd)
    rms = RMSProp(f, random_restarts=10)
    print(rms)
    rms.descent()
    print(rms)
    np.testing.assert_allclose(adam.global_optimizer, opt_sgd.global_optimizer, atol=1e-3)
    np.testing.assert_allclose(adam.global_optimizer, rms.global_optimizer, atol=1e-3)
    np.testing.assert_allclose(adam.global_optimizer, np.array([0, 0, 2]), atol=1e-3)


def test_vector():
    f = lambda v: np.sum(v ** 2)
    adam = Adam(f, random_restarts=10)
    adam.descent()
    assert len(adam.global_optimizer) == 1
    assert np.abs(adam.trace_values[-1]) < 1e-5


def test_vector_1():
    f = lambda v: np.sum(v ** 2)
    adam = Adam(f, random_restarts=10, init_points=[1, 1])
    adam.descent()
    assert len(adam.global_optimizer) == 2
    assert np.abs(adam.trace_values[-1]) < 1e-5


def test_vector_2():
    f = lambda v: np.sum(v ** 2)
    adam = Adam(f, random_restarts=10, init_points=[1, 1, 1, 1, 1, 1, 1, 1, 10])
    adam.descent()
    assert len(adam.global_optimizer) == 9
    assert np.abs(adam.trace_values[-1]) < 1e-5


def test_vector_3():
    f = lambda v: np.sum(v ** 2)
    adam = Adam(f, random_restarts=10, dimensions=10)
    adam.descent()
    optimal_point = adam.global_optimizer
    assert len(optimal_point) == 10
    assert np.abs(adam.trace_values[-1]) < 1e-5

def test_vector_edge():
    f = lambda v: np.sum(v ** 2)
    adam = Adam(f, random_restarts=10, dimensions=1)
    adam.descent()
    optimal_point = adam.global_optimizer
    assert len(optimal_point) == 1
    assert np.abs(adam.trace_values[-1]) < 1e-5


def test_batch_size_1():
    f = lambda v: np.sum(v**2)
    with pytest.raises(ValueError):
        adam = Adam(f, dimensions=1, batch_size=3)

def test_batch_size_2():
    f = lambda v: np.sum(v ** 2)
    with pytest.raises(ValueError):
        adam = Adam(f, dimensions=1, batch_size=3.5)


def test_batch_size_3():
    f = lambda v: np.sum((v-2)**2)
    adam = Adam(f, dimensions=20, batch_size=3)
    adam.descent()
    optimal_point = adam.global_optimizer
    assert len(optimal_point) == 20
    assert np.abs(adam.trace_values[-1]) < 1e-5
