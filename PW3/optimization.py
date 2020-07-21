from collections import defaultdict
import numpy as np
from numpy.linalg import norm, solve
from time import time
from datetime import datetime
from utils import get_line_search_tool
from functools import partial
from oracles import BarrierLassoOracle, lasso_duality_gap
import itertools
import scipy


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    """
    Newton's optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively. If the Hessian
        returned by the oracle is not positive-definite method stops with message="newton_direction_error"
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'newton_direction_error': in case of failure of solving linear system with Hessian matrix (e.g. non-invertible matrix).
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = newton(oracle, np.zeros(5), line_search_options={'method': 'Constant', 'c': 1.0})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    if display:
        print("x_0: {}".format(x_k))

    grad_norm_0 = np.linalg.norm(oracle.grad(x_0))
    start = datetime.now()
    for iter in range(max_iter):

        grad = oracle.grad(x_k)
        grad_norm = np.linalg.norm(grad)
        t = (datetime.now() - start).total_seconds()

        if trace:
            history['time'].append(t)
            if x_0.size <= 2:
                history['x'].append(x_k)
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(grad_norm)

        if not (np.isfinite(x_k).all() and
                np.isfinite(oracle.func(x_k)) and
                np.isfinite(oracle.grad(x_k)).all()):
            return x_k, 'computational_error', history

        if grad_norm ** 2 <= tolerance * grad_norm_0 ** 2:
            return x_k, 'success', history

        try:
            L = scipy.linalg.cho_factor(oracle.hess(x_k))
        except:
            return x_k, 'newton_direction_error', history

        d_k = -scipy.linalg.cho_solve(L, grad)
        n = x_k.size // 2
        x, u = x_k[:n], x_k[n:]
        d_x, d_u = d_k[:n], d_k[n:]

        max_alpha = line_search_tool.alpha_0
        if np.sum(d_x > d_u):
            max_alpha = min(((u - x)[d_x > d_u] / (d_x - d_u)[d_x > d_u]).min() * 0.99, max_alpha)  # u - x > 0
        if np.sum(-d_x > d_u):
            max_alpha = min(((u + x)[-d_x > d_u] / -(d_x + d_u)[-d_x > d_u]).min() * 0.99, max_alpha)  # u + x > 0
        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=max_alpha)
        x_k = x_k + alpha * d_k
        print(f"Newton iteration = {iter}, d_k = {d_k}, alpha_k = {alpha}")

        if display:
            print("x_k: {}, d_k: {}, alpha: {}".format(x_k, d_k, alpha))

    if display:
        print("x_star: {}".format(x_k))

    grad = oracle.grad(x_k)
    grad_norm = np.linalg.norm(grad)
    t = (datetime.now() - start).total_seconds()

    if trace:
        history['time'].append(t)
        if x_0.size <= 2:
            history['x'].append(x_k)
        history['func'].append(oracle.func(x_k))
        history['grad_norm'].append(grad_norm)

    if not (np.isfinite(x_k).all() and
            np.isfinite(oracle.func(x_k)) and
            np.isfinite(oracle.grad(x_k)).all()):
        return x_k, 'computational_error', history

    if grad_norm ** 2 <= tolerance * grad_norm_0 ** 2:
        return x_k, 'success', history

    return x_k, 'iterations_exceeded', history


def barrier_method_lasso(A, b, reg_coef, x_0, u_0, tolerance=1e-5,
                         tolerance_inner=1e-8, max_iter=100,
                         max_iter_inner=20, t_0=1, gamma=10,
                         c1=1e-4, lasso_duality_gap=None,
                         trace=False, display=False):
    """
    Log-barrier method for solving the problem:
        minimize    f(x, u) := 1/2 * ||Ax - b||_2^2 + reg_coef * \sum_i u_i
        subject to  -u_i <= x_i <= u_i.

    The method constructs the following barrier-approximation of the problem:
        phi_t(x, u) := t * f(x, u) - sum_i( log(u_i + x_i) + log(u_i - x_i) )
    and minimize it as unconstrained problem by Newton's method.

    In the outer loop `t` is increased and we have a sequence of approximations
        { phi_t(x, u) } and solutions { (x_t, u_t)^{*} } which converges in `t`
    to the solution of the original problem.

    Parameters
    ----------
    A : np.array
        Feature matrix for the regression problem.
    b : np.array
        Given vector of responses.
    reg_coef : float
        Regularization coefficient.
    x_0 : np.array
        Starting value for x in optimization algorithm.
    u_0 : np.array
        Starting value for u in optimization algorithm.
    tolerance : float
        Epsilon value for the outer loop stopping criterion:
        Stop the outer loop (which iterates over `k`) when
            `duality_gap(x_k) <= tolerance`
    tolerance_inner : float
        Epsilon value for the inner loop stopping criterion.
        Stop the inner loop (which iterates over `l`) when
            `|| \nabla phi_t(x_k^l) ||_2^2 <= tolerance_inner * \| \nabla \phi_t(x_k) \|_2^2 `
    max_iter : int
        Maximum number of iterations for interior point method.
    max_iter_inner : int
        Maximum number of iterations for inner Newton's method.
    t_0 : float
        Starting value for `t`.
    gamma : float
        Multiplier for changing `t` during the iterations:
        t_{k + 1} = gamma * t_k.
    c1 : float
        Armijo's constant for line search in Newton's method.
    lasso_duality_gap : callable object or None.
        If calable the signature is lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef)
        Returns duality gap value for esimating the progress of method.
    trace : bool
        If True, the progress information is appended into history dictionary
        during training. Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    (x_star, u_star) : tuple of np.array
        The point found by the optimization procedure.
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every **outer** iteration of the algorithm
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """

    def update_history(t, gap, z_k):
        history['time'].append(t)
        if x_0.size <= 2:
            history['x'].append(x_k)
        history['func'].append(oracle.func(z_k))
        if lasso_duality_gap:
            history['duality_gap'].append(gap)

    history = defaultdict(list) if trace else None
    oracle = BarrierLassoOracle(A, b, reg_coef, t_0)
    x_k, u_k, t_k = x_0, u_0, t_0
    n = A.shape[1]
    start = datetime.now()
    for k in range(max_iter):

        z_k = np.hstack([x_k, u_k])
        if not (np.isfinite(z_k).all() and
                np.isfinite(oracle.func(z_k)) and
                np.isfinite(oracle.grad(z_k)).all()):
            return x_k, 'computational_error', history

        if lasso_duality_gap:
            Ax_b = A @ x_k - b
            ATAx_b = A.T @ Ax_b
            gap = lasso_duality_gap(x_k, Ax_b, ATAx_b, b, reg_coef)
            if gap <= tolerance:
                break
        else:
            gap = None

        if display:
            print(x_k)

        if trace:
            t = (datetime.now() - start).total_seconds()
            z_k = np.hstack([x_k, u_k])
            update_history(t, gap, z_k)

        if gap <= tolerance:
            return (x_k, u_k), 'success', history
        z_k = np.hstack([x_k, u_k])
        z_k, message, hist = newton(oracle, z_k,
                                    tolerance_inner, max_iter_inner,
                                    line_search_options={'c1': c1},
                                    trace=True)
        x_k, u_k = z_k[:n], z_k[n:]
        t_k = min(n / tolerance + 1, t_k * gamma)
        oracle.update_tau(t_k)

    z_k = np.hstack([x_k, u_k])
    if not (np.isfinite(z_k).all() and
            np.isfinite(oracle.func(z_k)) and
            np.isfinite(oracle.grad(z_k)).all()):
        return x_k, 'computational_error', history

    if lasso_duality_gap:
        Ax_b = A @ x_k - b
        ATAx_b = A.T @ Ax_b
        gap = lasso_duality_gap(x_k, Ax_b, ATAx_b, b, reg_coef)

    if trace:
        t = (datetime.now() - start).total_seconds()
        z_k = np.hstack([x_k, u_k])
        update_history(t, gap, z_k)

    if display:
        print(x_k)

    if gap <= tolerance:
        return (x_k, u_k), 'success', history
    return (x_k, u_k), 'iterations_exceeded', history
