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
    ok_(sys.version_info >= (3, 4))


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


def test_barrier_prototype():
    method = optimization.barrier_method_lasso
    A = np.eye(2)
    b = np.array([1.0, 2.0])
    reg_coef = 2.0
    x_0 = np.array([10.0, 10.0])
    u_0 = np.array([11.0, 11.0])
    ldg = oracles.lasso_duality_gap

    method(A, b, reg_coef, x_0, u_0, lasso_duality_gap=ldg)
    check_prototype_results(method(A, b, reg_coef, x_0, u_0,
                                   lasso_duality_gap=ldg, tolerance=1e10),
                            [(x_0, u_0), 'success', None])
    check_prototype_results(method(A, b, reg_coef, x_0, u_0,
                                   lasso_duality_gap=ldg, tolerance=1e10,
                                   trace=True),
                            [(x_0, u_0), 'success', [0.0]])
    check_prototype_results(method(A, b, reg_coef, x_0, u_0,
                                   lasso_duality_gap=ldg, max_iter=1,
                                   trace=True),
                            [None, 'iterations_exceeded', [0.0, 0.0]])
    method(A, b, reg_coef, x_0, u_0, lasso_duality_gap=ldg,
           tolerance_inner=1e-8)
    method(A, b, reg_coef, x_0, u_0, lasso_duality_gap=ldg, max_iter=1)
    method(A, b, reg_coef, x_0, u_0, lasso_duality_gap=ldg, max_iter_inner=2)
    method(A, b, reg_coef, x_0, u_0, lasso_duality_gap=ldg, t_0=1)
    method(A, b, reg_coef, x_0, u_0, lasso_duality_gap=ldg, gamma=10)
    method(A, b, reg_coef, x_0, u_0, lasso_duality_gap=ldg, c1=1e-4)
    method(A, b, reg_coef, x_0, u_0, lasso_duality_gap=ldg, trace=True)
    method(A, b, reg_coef, x_0, u_0, lasso_duality_gap=ldg, display=True)
    method(A, b, reg_coef, x_0, u_0, 1e-5, 1e-8, 100, 20, 1, 10, 1e-4, ldg,
           True, True)


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


def test_barrier_univariate():
    # Minimize 1/2 * x^2 + 3.0 * u,
    #     s.t. -u <= x <= u

    A = np.array([[1.0]])
    b = np.array([0.0])
    reg_coef = 3.0

    x_0 = np.array([1.0])
    u_0 = np.array([2.0])

    print("Univariate")
    (x_star, u_star), message, history = optimization.barrier_method_lasso(
        A, b, reg_coef, x_0, u_0,
        lasso_duality_gap=oracles.lasso_duality_gap,
        trace=True)
    eq_(message, "success")
    ok_(np.allclose(
        x_star,
        np.array([-7.89796404e-07])))
    ok_(isinstance(x_star, np.ndarray))
    ok_(np.allclose(
        u_star,
        np.array([0.66666568])))
    ok_(isinstance(u_star, np.ndarray))
    check_equal_histories(
        history,
        {'time': [None] * 2,
         'duality_gap': [4.0, 2.3693898355252885e-06]})


def test_barrier_one_step():
    # Simple 2-dimensional problem with identity matrix.
    A = np.eye(2)
    b = np.array([1.0, 0.0])
    reg_coef = 1.0

    x_0 = np.array([0.0, 1.0])
    u_0 = np.array([2.0, 2.0])

    (x_star, u_star), message, history = optimization.barrier_method_lasso(
        A, b, reg_coef, x_0, u_0,
        lasso_duality_gap=oracles.lasso_duality_gap,
        trace=True)
    eq_(message, "success")
    ok_(np.allclose(
        x_star,
        np.array([0.00315977, 0.0])))
    ok_(isinstance(x_star, np.ndarray))
    ok_(np.allclose(
        u_star,
        np.array([3.16979051e-03, 2.00000000e-05])))
    ok_(isinstance(u_star, np.ndarray))
    check_equal_histories(
        history,
        {'time': [None] * 7,
         'duality_gap': [
             2.0,
             0.47457244767635376,
             0.083152839863428252,
             0.0094880329978710432,
             0.0009840651705023129,
             9.9498755346316692e-05,
             9.9841761475039092e-06]})
