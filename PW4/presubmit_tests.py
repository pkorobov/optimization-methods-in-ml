import nose
from nose.tools import assert_almost_equal, ok_, eq_
from nose.plugins.attrib import attr
from io import StringIO
import numpy as np
import scipy
import scipy.sparse
import scipy.optimize
import sys
import warnings

import optimization
import oracles


def test_python3():
    ok_(sys.version_info > (3, 0))


def test_least_squares_oracle():
    A = np.eye(3)
    b = np.array([1, 2, 3])

    def matvec_Ax(x):
        return A.dot(x)

    def matvec_ATx(x):
        return A.T.dot(x)

    oracle = oracles.LeastSquaresOracle(matvec_Ax, matvec_ATx, b)

    # Checks at point x = [0, 0, 0]
    x = np.zeros(3)
    assert_almost_equal(oracle.func(x), 7.0)
    ok_(np.allclose(oracle.grad(x), np.array([-1., -2., -3.])))
    ok_(isinstance(oracle.grad(x), np.ndarray))

    # Checks at point x = [1, 1, 1]
    x = np.ones(3)
    assert_almost_equal(oracle.func(x), 2.5)
    ok_(np.allclose(oracle.grad(x), np.array([0., -1., -2.])))
    ok_(isinstance(oracle.grad(x), np.ndarray))


def test_least_squares_oracle_2():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([1.0, -1.0])

    def matvec_Ax(x):
        return A.dot(x)

    def matvec_ATx(x):
        return A.T.dot(x)

    oracle = oracles.LeastSquaresOracle(matvec_Ax, matvec_ATx, b)

    # Checks at point x = [1, 2]
    x = np.array([1.0, 2.0])
    assert_almost_equal(oracle.func(x), 80.0)
    ok_(np.allclose(oracle.grad(x), np.array([40.,  56.])))
    ok_(isinstance(oracle.grad(x), np.ndarray))


def test_l1_reg_oracle():
    # h(x) = 1.0 * \|x\|_1
    oracle = oracles.L1RegOracle(1.0)

    # Checks at point x = [0, 0, 0]
    x = np.zeros(3)
    assert_almost_equal(oracle.func(x), 0.0)
    ok_(np.allclose(oracle.prox(x, alpha=1.0), x))
    ok_(np.allclose(oracle.prox(x, alpha=2.0), x))
    ok_(isinstance(oracle.prox(x, alpha=1.0), np.ndarray))

    # Checks at point x = [-3]
    x = np.array([-3.0])
    assert_almost_equal(oracle.func(x), 3.0)
    ok_(np.allclose(oracle.prox(x, alpha=1.0), np.array([-2.0])))
    ok_(np.allclose(oracle.prox(x, alpha=2.0), np.array([-1.0])))
    ok_(isinstance(oracle.prox(x, alpha=1.0), np.ndarray))

    # Checks at point x = [-3, 3]
    x = np.array([-3.0, 3.0])
    assert_almost_equal(oracle.func(x), 6.0)
    ok_(np.allclose(oracle.prox(x, alpha=1.0), np.array([-2.0, 2.0])))
    ok_(np.allclose(oracle.prox(x, alpha=2.0), np.array([-1.0, 1.0])))
    ok_(isinstance(oracle.prox(x, alpha=1.0), np.ndarray))


def test_l1_reg_oracle_2():
    # h(x) = 2.0 * \|x\|_1
    oracle = oracles.L1RegOracle(2.0)

    # Checks at point x = [-3, 3]
    x = np.array([-3.0, 3.0])
    assert_almost_equal(oracle.func(x), 6 * 2.0)
    ok_(np.allclose(oracle.prox(x, alpha=1.0), np.array([-1.0, 1.0])))


def test_lasso_nonsmooth_oracle():
    A = np.eye(2)
    b = np.array([1.0, 2.0])
    oracle = oracles.create_lasso_nonsmooth_oracle(A, b, regcoef=2.0)

    # Checks at point x = [1, 0]
    x = np.array([-3.0, 0.0])
    assert_almost_equal(oracle.func(x), 16.0)
    assert_almost_equal(oracle.duality_gap(x), 14.5)
    # Checks a subgradient
    g = oracle.subgrad(x)
    ok_(isinstance(g, np.ndarray))
    assert_almost_equal(g[0], -6.0)
    assert_almost_equal(g[1], -2.0)


def test_lasso_duality_gap():
    A = np.eye(3)
    b = np.array([1.0, 2.0, 3.0])
    regcoef = 2.0

    # Checks at point x = [0, 0, 0]
    x = np.zeros(3)
    assert_almost_equal(0.77777777777777,
                        oracles.lasso_duality_gap(x, A.dot(x) - b,
                                                  A.T.dot(A.dot(x) - b),
                                                  b, regcoef))

    # Checks at point x = [1, 1, 1]
    x = np.ones(3)
    assert_almost_equal(3.0, oracles.lasso_duality_gap(x, A.dot(x) - b,
                                                       A.T.dot(A.dot(x) - b),
                                                       b, regcoef))


def test_lasso_prox_oracle():
    A = np.eye(2)
    b = np.array([1.0, 2.0])
    oracle = oracles.create_lasso_prox_oracle(A, b, regcoef=1.0)

    # Checks at point x = [-3, 3]
    x = np.array([-3.0, 3.0])
    assert_almost_equal(oracle.func(x), 14.5)
    ok_(np.allclose(oracle.grad(x), np.array([-4.,  1.])))
    ok_(isinstance(oracle.grad(x), np.ndarray))
    ok_(np.allclose(oracle.prox(x, alpha=1.0), np.array([-2.0, 2.0])))
    ok_(np.allclose(oracle.prox(x, alpha=2.0), np.array([-1.0, 1.0])))
    ok_(isinstance(oracle.prox(x, alpha=1.0), np.ndarray))
    assert_almost_equal(oracle.duality_gap(x), 14.53125)


def check_prototype_results(results, groundtruth):
    if groundtruth[0] is not None:
        ok_(np.allclose(np.array(results[0]),
                        np.array(groundtruth[0])))

    if groundtruth[1] is not None:
        eq_(results[1], groundtruth[1])

    if groundtruth[2] is not None:
        ok_(results[2] is not None)
        ok_('time' in results[2])
        ok_('func' in results[2])
        ok_('duality_gap' in results[2])
        eq_(len(results[2]['func']), len(groundtruth[2]))
    else:
        ok_(results[2] is None)


def test_subgradient_prototype():
    method = optimization.subgradient_method

    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([1.0, 2.0])
    oracle = oracles.create_lasso_nonsmooth_oracle(A, b, regcoef=2.0)
    x_0 = np.array([-3.0, 0.0])

    method(oracle, x_0)
    check_prototype_results(method(oracle, x_0, tolerance=1e10),
                            [x_0, 'success', None])
    check_prototype_results(method(oracle, x_0, tolerance=1e10, trace=True),
                            [None, 'success', [0.0]])
    check_prototype_results(method(oracle, x_0, max_iter=1),
                            [None, 'iterations_exceeded', None])
    check_prototype_results(method(oracle, x_0, max_iter=1, trace=True),
                            [None, 'iterations_exceeded', [0.0, 0.0]])
    method(oracle, x_0, alpha_0=1)
    method(oracle, x_0, display=True)
    method(oracle, x_0, 1e-2, 100, 1, True, True)


def test_proximal_gm_prototype():
    method = optimization.proximal_gradient_method

    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([1.0, 2.0])
    oracle = oracles.create_lasso_prox_oracle(A, b, regcoef=2.0)
    x_0 = np.array([-3.0, 0.0])

    method(oracle, x_0)
    method(oracle, x_0, L_0=1)
    check_prototype_results(method(oracle, x_0, tolerance=1e10),
                            [None, 'success', None])
    check_prototype_results(method(oracle, x_0, tolerance=1e10, trace=True),
                            [None, 'success', [0.0]])
    check_prototype_results(method(oracle, x_0, max_iter=1),
                            [None, 'iterations_exceeded', None])
    check_prototype_results(method(oracle, x_0, max_iter=1, trace=True),
                            [None, 'iterations_exceeded', [0.0, 0.0]])
    method(oracle, x_0, display=True)
    method(oracle, x_0, 1, 1e-5, 100, True, True)


def test_proximal_fast_gm_prototype():
    method = optimization.proximal_fast_gradient_method

    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([1.0, 2.0])
    oracle = oracles.create_lasso_prox_oracle(A, b, regcoef=2.0)
    x_0 = np.array([-3.0, 0.0])

    method(oracle, x_0)
    method(oracle, x_0, L_0=1)
    check_prototype_results(method(oracle, x_0, tolerance=1e10),
                            [None, 'success', None])
    check_prototype_results(method(oracle, x_0, tolerance=1e10, trace=True),
                            [None, 'success', [0.0]])
    check_prototype_results(method(oracle, x_0, max_iter=1),
                            [None, 'iterations_exceeded', None])
    check_prototype_results(method(oracle, x_0, max_iter=1, trace=True),
                            [None, 'iterations_exceeded', [0.0, 0.0]])
    method(oracle, x_0, display=True)
    method(oracle, x_0, 1, 1e-5, 100, True, True)


def test_proximal_gm_one_step():
    # Simple smooth quadratic task.
    A = np.eye(2)
    b = np.array([1.0, 0.0])
    oracle = oracles.create_lasso_prox_oracle(A, b, regcoef=0.0)
    x_0 = np.zeros(2)

    x_star, status, hist = optimization.proximal_gradient_method(
                                oracle, x_0, trace=True)
    eq_(status, 'success')
    ok_(np.allclose(x_star, np.array([1.0, 0.0])))
    ok_(np.allclose(np.array(hist['func']), np.array([0.5, 0.0])))


def test_proximal_gm_nonsmooth():
    # Minimize ||x||_1.
    oracle = oracles.create_lasso_prox_oracle(np.zeros([2, 2]),
                                              np.zeros(2),
                                              regcoef=1.0)
    x_0 = np.array([2.0, -1.0])
    x_star, status, hist = optimization.proximal_gradient_method(
                                oracle, x_0, trace=True)
    eq_(status, 'success')
    ok_(np.allclose(x_star, np.array([0.0, 0.0])))
    ok_(np.allclose(np.array(hist['func']), np.array([3.0, 1.0, 0.0])))


def proximal_fast_gm_one_step():
    # Simple smooth quadratic task.
    A = np.eye(2)
    b = np.array([1.0, 0.0])
    oracle = oracles.create_lasso_prox_oracle(A, b, regcoef=0.0)
    x_0 = np.zeros(2)

    x_star, status, hist = optimization.proximal_fast_gradient_method(
                                oracle, x_0, trace=True)
    eq_(status, 'success')
    ok_(np.allclose(x_star, np.array([1.0, 0.0])))
    ok_(np.allclose(np.array(hist['func']), np.array([0.5, 0.0])))


def test_proximal_fast_gm_nonsmooth():
    # Minimize ||x||_1.
    oracle = oracles.create_lasso_prox_oracle(np.zeros([2, 2]),
                                              np.zeros(2),
                                              regcoef=1.0)
    x_0 = np.array([2.0, -1.0])
    x_star, status, hist = optimization.proximal_fast_gradient_method(
                                oracle, x_0, trace=True)
    eq_(status, 'success')
    ok_(np.allclose(x_star, np.array([7.64497775e-06, 0.0])))
    ok_(np.allclose(np.array(hist['func']), np.array([
        [3.0, 1.0, 0.2679491924311227,
         0.099179721203666166, 0.041815858679730096, 0.018850854415891891,
         0.0088302704548977509, 0.0042337678350286671, 0.0020598280587097542,
         0.0010116006643634953, 0.00049984569823222113, 0.00024797029202045391,
         0.00012334133663333596, 6.1457610088328345e-05, 3.0658125993932962e-05,
         1.5305578515886715e-05, 7.6449777454652732e-06]
        ])))


def test_proximal_fast_gm_nonsmooth2():
    oracle = oracles.create_lasso_prox_oracle(np.array([[1, 2, 3], [4, 5, 6]]),
                                              np.array([1, 4]),
                                              regcoef=1.0)
    x_0 = np.array([1, 1, -1])
    x_star, status, hist = optimization.proximal_fast_gradient_method(
                                oracle, x_0, trace=True, max_iter=3)
    eq_(status, 'iterations_exceeded')
    ok_(np.allclose(x_star, np.array([[1.01714486, 1.04658812, -0.85594606]])))
    ok_(np.allclose(np.array(hist['func']), np.array([4.0, 3.219970703125, 3.0946475565433502, 3.0380920410969945])))


def check_equal_histories(test_history, reference_history, atol=1e-3):
    if test_history is None or reference_history is None:
        eq_(test_history, reference_history)
        return

    for key in reference_history.keys():
        eq_(key in test_history, True)
        if key != 'time':
            ok_(np.allclose(
                test_history[key],
                reference_history[key],
                atol=atol))
        else:
            # make sure its length is correct and
            # its values are non-negative and monotonic
            eq_(len(test_history[key]), len(reference_history[key]))
            test_time = np.asarray(test_history['time'])
            ok_(np.all(test_time >= 0))
            ok_(np.all(test_time[1:] - test_time[:-1] >= 0))


def test_subgradient_one_step():
    # Simple smooth quadratic task.
    A = np.eye(2)
    b = np.array([1.0, 0.0])
    oracle = oracles.create_lasso_nonsmooth_oracle(A, b, regcoef=0.0)
    x_0 = np.zeros(2)

    x_star, status, hist = optimization.subgradient_method(
                                oracle, x_0, trace=True)
    eq_(status, 'success')
    ok_(np.allclose(x_star, np.array([1.0, 0.0])))
    ok_(np.allclose(np.array(hist['func']), np.array([0.5, 0.0])))


def test_subgradient_one_step_nonsmooth():
    # Minimize 0.5 * ||x - b||_2^2 + ||x||_1
    # with small tolerance by one step.
    A = np.eye(2)
    b = np.array([3.0, 3.0])
    oracle = oracles.create_lasso_nonsmooth_oracle(A, b, regcoef=1.0)
    x_0 = np.ones(2)
    x_star, status, hist = optimization.subgradient_method(
                                oracle, x_0, tolerance=1e-1, trace=True)
    eq_(status, 'success')
    ok_(np.allclose(x_star, np.array([1.70710678,  1.70710678])))
    ok_(np.allclose(np.array(hist['func']), np.array([6.0, 5.085786437626])))
