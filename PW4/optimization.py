from collections import defaultdict
import numpy as np
from datetime import datetime
from time import time


def subgradient_method(oracle, x_0, tolerance=1e-2, max_iter=1000, alpha_0=1,
                       display=False, trace=False):
    """
    Subgradient descent method for nonsmooth convex optimization.

    Parameters
    ----------
    oracle : BaseNonsmoothConvexOracle-descendant object
        Oracle with .func() and .subgrad() methods implemented for computing
        function value and its one (arbitrary) subgradient respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    alpha_0 : float
        Initial value for the sequence of step-sizes.
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
            - history['func'] : list of function values phi(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0)
    min_point = np.copy(x_0)
    if display:
        print("x_0: {}".format(x_k))

    start = datetime.now()
    for iter in range(max_iter):

        d_k = -oracle.subgrad(x_k)
        if np.linalg.norm(d_k) > 0:
            d_k /= np.linalg.norm(d_k)
        t = (datetime.now() - start).total_seconds()

        if oracle.func(x_k) < oracle.func(min_point):
            min_point = x_k

        duality_gap = oracle.duality_gap(min_point)
        if trace:
            history['time'].append(t)
            if x_0.size <= 2:
                history['x'].append(min_point)
            history['func'].append(oracle.func(min_point))
            history['duality_gap'].append(duality_gap)

        if duality_gap <= tolerance:
            return x_k, 'success', history

        alpha = alpha_0 / np.sqrt(1 + iter * 100)
        x_k = x_k + alpha * d_k

        if display:
            print("x_k: {}, d_k: {}".format(x_k, d_k))

    if display:
        print("x_star: {}".format(x_k))

    if oracle.func(x_k) < oracle.func(min_point):
        min_point = x_k

    duality_gap = oracle.duality_gap(min_point)
    if trace:
        history['time'].append(t)
        if x_0.size <= 2:
            history['x'].append(min_point)
        history['func'].append(oracle.func(min_point))
        history['duality_gap'].append(duality_gap)

    if duality_gap <= tolerance:
        return min_point, 'success', history

    return min_point, 'iterations_exceeded', history


def proximal_gradient_method(oracle, x_0, L_0=1, tolerance=1e-5,
                             max_iter=1000, trace=False, display=False):
    """
    Gradient method for composite optimization.

    Parameters
    ----------
    oracle : BaseCompositeOracle-descendant object
        Oracle with .func() and .grad() and .prox() methods implemented
        for computing function value, its gradient and proximal mapping
        respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    L_0 : float
        Initial value for adaptive line-search.
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
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
            - history['func'] : list of objective function values phi(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0)
    L = L_0

    if display:
        print("x_0: {}".format(x_k))

    start = datetime.now()
    for iter in range(max_iter):

        duality_gap = oracle.duality_gap(x_k)
        t = (datetime.now() - start).total_seconds()

        if trace:
            history['time'].append(t)
            if x_0.size <= 2:
                history['x'].append(x_k)
            history['func'].append(oracle.func(x_k))
            history['duality_gap'].append(duality_gap)

        if duality_gap <= tolerance:
            return x_k, 'success', history

        int_steps = 1
        while True:
            alpha = 1 / L
            y = oracle.prox(x_k - alpha * oracle.grad(x_k), alpha)
            if oracle.f(y) <= oracle.f(x_k) + oracle.grad(x_k) @ (y - x_k) + L / 2 * (y - x_k) @ (y - x_k):
                break
            L *= 2
            int_steps += 1
        x_k = y
        L = np.maximum(L_0, L / 2)

        if display:
            print("x_k: {}, alpha: {}".format(x_k, alpha))

        if trace:
            history['int_steps'].append(int_steps)

    if display:
        print("x_star: {}".format(x_k))

    duality_gap = oracle.duality_gap(x_k)

    if trace:
        history['time'].append(t)
        if x_0.size <= 2:
            history['x'].append(x_k)
        history['func'].append(oracle.func(x_k))
        history['duality_gap'].append(duality_gap)

    if duality_gap <= tolerance:
        return x_k, 'success', history

    return x_k, 'iterations_exceeded', history


def proximal_fast_gradient_method(oracle, x_0, L_0=1.0, tolerance=1e-5,
                                  max_iter=1000, trace=False, display=False):
    """
    Fast gradient method for composite minimization.

    Parameters
    ----------
    oracle : BaseCompositeOracle-descendant object
        Oracle with .func() and .grad() and .prox() methods implemented
        for computing function value, its gradient and proximal mapping
        respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    L_0 : float
        Initial value for adaptive line-search.
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
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
            - history['func'] : list of objective function values phi(best_point) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps for best point on every step of the algorithm
    """
    history = defaultdict(list) if trace else None

    L = L_0
    A_k = 0
    x_k = np.copy(x_0)
    v_k = np.copy(x_0)
    min_point = np.copy(x_0)

    if display:
        print("x_0: {}".format(x_k))

    start = datetime.now()

    weighted_grad_sum = 0.

    for iter in range(max_iter):

        duality_gap = oracle.duality_gap(x_k)
        t = (datetime.now() - start).total_seconds()

        if trace:
            history['time'].append(t)
            if x_0.size <= 2:
                history['x'].append(min_point)
            history['func'].append(oracle.func(min_point))
            history['duality_gap'].append(duality_gap)

        if duality_gap <= tolerance:
            return x_k, 'success', history

        int_steps = 1
        while True:
            a_k = (1 + np.sqrt(1 + 4 * L * A_k)) / (2 * L)
            A_k_next = A_k + a_k
            y_k = (A_k * x_k + a_k * v_k) / A_k_next

            cur_weighted_grad_sum = weighted_grad_sum + a_k * oracle.grad(y_k)

            v_k_next = oracle.prox(x_0 - cur_weighted_grad_sum, A_k_next)
            x_k_next = (A_k * x_k + a_k * v_k_next) / A_k_next

            if oracle.func(y_k) < oracle.func(min_point):
                min_point = y_k

            if oracle.func(x_k_next) < oracle.func(min_point):
                min_point = x_k_next

            if oracle.f(x_k_next) <= oracle.f(y_k) + oracle.grad(y_k) @ (x_k_next - y_k) + \
                    L / 2 * (x_k_next - y_k) @ (x_k_next - y_k):
                break
            L *= 2
            int_steps += 1
        L /= 2

        A_k = A_k_next
        v_k = v_k_next
        x_k = x_k_next
        weighted_grad_sum = cur_weighted_grad_sum

        if display:
            print("x_k: {}, y_k: {}".format(x_k, y_k))

        if trace:
            history['int_steps'].append(int_steps)

    if display:
        print("x_star: {}".format(x_k))

    duality_gap = oracle.duality_gap(min_point)
    if trace:
        history['time'].append(t)
        if x_0.size <= 2:
            history['x'].append(min_point)
        history['func'].append(oracle.func(min_point))
        history['duality_gap'].append(duality_gap)

    if duality_gap <= tolerance:
        return min_point, 'success', history

    return min_point, 'iterations_exceeded', history