from collections import defaultdict, deque  # Use this for effective implementation of L-BFGS
from datetime import datetime
import numpy as np
from utils import get_line_search_tool
import itertools
from functools import partial
import copy

np.set_printoptions(suppress=True)


def conjugate_gradients(matvec, b, x_0, tolerance=1e-4, max_iter=None, trace=False, display=False):
    """
    Solves system Ax=b using Conjugate Gradients method.

    Parameters
    ----------
    matvec : function
        Implement matrix-vector product of matrix A and arbitrary vector x
    b : 1-dimensional np.array
        Vector b for the system.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
        Stop optimization procedure and return x_k when:
         ||Ax_k - b||_2 <= tolerance * ||b||_2
    max_iter : int, or None
        Maximum number of iterations. if max_iter=None, set max_iter to n, where n is
        the dimension of the space
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display:  bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['residual_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """

    def grad(x):
        return matvec(x) - b

    def update_history(t_k, res_norm_k, x_k):
        history['time'].append(t_k)
        history['residual_norm'].append(res_norm_k)
        if x_0.size <= 2:
            history['x'].append(x_k)

    if display:
        print(f"x_0 = {x_0}")

    history = defaultdict(list) if trace else None
    start = datetime.now()

    b_norm = np.linalg.norm(b)
    x_k = np.copy(x_0)
    g_k = grad(x_k)
    d_k = -g_k
    res_norm = np.linalg.norm(g_k)

    if trace:
        t = (datetime.now() - start).total_seconds()
        update_history(t, res_norm, x_k)

    if res_norm <= tolerance * b_norm:
        return x_k, 'success', history

    if display:
        print(f"x_0 = {x_0}")

    i = 0
    while max_iter is None or (max_iter is not None and i < max_iter):
        x_k = x_k + np.dot(g_k, g_k) / np.dot(matvec(d_k), d_k) * d_k

        if display:
            print(f"x_{i+1} = {x_k}")

        g_k_plus_1 = grad(x_k)
        res_norm = np.linalg.norm(g_k_plus_1)
        if trace:
            t = (datetime.now() - start).total_seconds()
            update_history(t, res_norm, x_k)

        if np.linalg.norm(g_k_plus_1) <= tolerance * b_norm:
            return x_k, 'success', history

        d_k = -g_k_plus_1 + np.dot(g_k_plus_1, g_k_plus_1) / np.dot(g_k, g_k) * d_k
        g_k = g_k_plus_1
        i += 1

    if max_iter is not None and i >= max_iter:
        return x_k, 'iterations_exceeded', history
    return x_k, 'success', history


def lbfgs(oracle, x_0, tolerance=1e-4, max_iter=500, memory_size=10,
          line_search_options=None, display=False, trace=False):
    """
    Limited-memory Broyden–Fletcher–Goldfarb–Shanno's method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented for computing
        function value and its gradient respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    memory_size : int
        The length of directions history in L-BFGS method.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """

    def bfgs_multiply(v, H, gamma_0):
        if not H:
            return gamma_0 * v
        s, y = H.pop()
        v_ = v - np.dot(s, v) / np.dot(y, s) * y
        z = bfgs_multiply(v_, H, gamma_0)
        return z + (np.dot(s, v) - np.dot(y, z)) / np.dot(y, s) * s

    def lbfgs_direction(g_k, H_k):
        if H_k:
            s, y = H_k[-1]
            gamma_0 = np.dot(y, s) / np.dot(y, y)
        else:
            gamma_0 = 1.0
        return bfgs_multiply(-g_k, copy.copy(H_k), gamma_0)

    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    if display:
        print("x_0: {}".format(x_k))

    g_norm_0 = np.linalg.norm(oracle.grad(x_0))
    H_k = deque(maxlen=memory_size)
    start = datetime.now()
    for iter in range(max_iter):

        g_k = oracle.grad(x_k)
        g_norm = np.linalg.norm(g_k)
        t = (datetime.now() - start).total_seconds()

        if trace:
            history['time'].append(t)
            if x_0.size <= 2:
                history['x'].append(x_k)
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(g_norm)

        if g_norm ** 2 <= tolerance * g_norm_0 ** 2:
            return x_k, 'success', history

        # search of direction
        d_k = lbfgs_direction(g_k, H_k)
        alpha = line_search_tool.line_search(oracle, x_k, d_k)

        H_k.append([alpha * d_k, oracle.grad(x_k + alpha * d_k) - oracle.grad(x_k)])
        x_k = x_k + alpha * d_k

        if display:
            print(f"fx_k: {x_k}, d_k: {d_k}, alpha: {alpha}")

    if display:
        print(f"x_star: {x_k}")

    g = oracle.grad(x_k)
    g_norm = np.linalg.norm(g)
    t = (datetime.now() - start).total_seconds()

    if trace:
        history['time'].append(t)
        if x_0.size <= 2:
            history['x'].append(x_k)
        history['func'].append(oracle.func(x_k))
        history['grad_norm'].append(g_norm)

    if g_norm ** 2 <= tolerance * g_norm_0 ** 2:
        return x_k, 'success', history
    return x_k, 'iterations_exceeded', history


def hessian_free_newton(oracle, x_0, tolerance=1e-4, max_iter=500,
                        line_search_options=None, display=False, trace=False):
    """
    Hessian Free method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess_vec() methods implemented for computing
        function value, its gradient and matrix product of the Hessian times vector respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    if display:
        print("x_0: {}".format(x_k))

    g_norm_0 = np.linalg.norm(oracle.grad(x_0))
    start = datetime.now()
    for iter in range(max_iter):

        g_k = oracle.grad(x_k)
        g_norm = np.linalg.norm(g_k)
        t = (datetime.now() - start).total_seconds()

        if trace:
            history['time'].append(t)
            if x_0.size <= 2:
                history['x'].append(x_k)
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(g_norm)

        if not (np.isfinite(x_k).all() and
                np.isfinite(oracle.func(x_k)) and
                np.isfinite(oracle.grad(x_k)).all()):
            return x_k, 'computational_error', history

        if g_norm ** 2 <= tolerance * g_norm_0 ** 2:
            return x_k, 'success', history

        # search of direction
        eta_k = min(0.5, np.linalg.norm(g_k) ** 0.5)
        d_k = conjugate_gradients(partial(oracle.hess_vec, x_k), -g_k, -g_k, eta_k)[0]
        while np.dot(d_k, g_k) >= 0:
            eta_k *= 0.1
            d_k = conjugate_gradients(partial(oracle.hess_vec, x_k), d_k, np.zeros(x_0.shape), eta_k)[0]

        alpha = line_search_tool.line_search(oracle, x_k, d_k)
        x_k = x_k + alpha * d_k

        if display:
            print(f"fx_k: {x_k}, d_k: {d_k}, alpha: {alpha}, eta: {eta_k}")

    if display:
        print(f"x_star: {x_k}")

    g = oracle.grad(x_k)
    g_norm = np.linalg.norm(g)
    t = (datetime.now() - start).total_seconds()

    if trace:
        history['time'].append(t)
        if x_0.size <= 2:
            history['x'].append(x_k)
        history['func'].append(oracle.func(x_k))
        history['grad_norm'].append(g_norm)

    if g_norm ** 2 <= tolerance * g_norm_0 ** 2:
        return x_k, 'success', history

    return x_k, 'iterations_exceeded', history


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradien descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
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
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c1': 1e-4})
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
        d_k = -oracle.grad(x_k)
        grad_norm = np.linalg.norm(d_k)
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

        alpha = line_search_tool.line_search(oracle, x_k, d_k)
        x_k = x_k + alpha * d_k

        if display:
            print("x_k: {}, d_k: {}, alpha: {}".format(x_k, d_k, alpha))

    if display:
        print("x_star: {}".format(x_k))

    d_k = -oracle.grad(x_k)
    grad_norm = np.linalg.norm(d_k)

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
