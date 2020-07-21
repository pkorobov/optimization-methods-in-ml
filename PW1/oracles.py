import numpy as np
import scipy
from scipy.special import expit
from scipy.sparse import diags

class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        return np.logaddexp(0., -self.b * self.matvec_Ax(x)).mean() + self.regcoef / 2 * np.linalg.norm(x) ** 2

    def grad(self, x):
        return -self.matvec_ATx((expit(-self.b * self.matvec_Ax(x)) * self.b).T).T / self.b.size + self.regcoef * x

    def hess(self, x):
        sigma = expit(-self.b * self.matvec_Ax(x))
        s = sigma * (1 - sigma) / self.b.size
        return self.matmat_ATsA(s) + self.regcoef * np.eye(x.size)


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle for logistic regression with l2 regularization
    with optimized *_directional methods (are used in line_search).

    For explanation see LogRegL2Oracle.
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
        self.Ax = None
        self.x = None
        self.dir_Ax = None
        self.dir_x = None
        self.Ad = None
        self.d = None

    def precompute(self, x, d=None, alpha=None):
        if self.dir_x is not None and np.array_equal(x, self.dir_x):
            self.x = self.dir_x
            self.Ax = self.dir_Ax

        if self.x is None or not np.array_equal(self.x, x):
            self.x = x
            self.Ax = self.matvec_Ax(x)

        if d is not None:
            if self.d is None or not np.array_equal(self.d, d):
                self.d = d
                self.Ad = self.matvec_Ax(d)
            if self.dir_x is None or not np.array_equal(x + alpha * d, self.dir_x):
                self.dir_x = x + alpha * d
                self.dir_Ax = self.Ax + alpha * self.Ad

    def func(self, x):
        self.precompute(x)
        return np.logaddexp(0., -self.b * self.Ax).mean() + self.regcoef / 2 * np.linalg.norm(x) ** 2

    def grad(self, x):
        self.precompute(x)
        return -self.matvec_ATx((expit(-self.b * self.Ax) * self.b).T).T / self.b.size + self.regcoef * x

    def hess(self, x):
        self.precompute(x)
        sigma = expit(-self.b * self.Ax)
        s = sigma * (1 - sigma) / self.b.size
        return self.matmat_ATsA(s) + self.regcoef * np.eye(x.size)

    def func_directional(self, x, d, alpha):
        self.precompute(x, d, alpha)
        return np.logaddexp(0., -self.b * self.dir_Ax).mean() + self.regcoef / 2 * np.linalg.norm(x + alpha * d) ** 2

    def grad_directional(self, x, d, alpha):
        self.precompute(x, d, alpha)
        return -(expit(-self.b * self.dir_Ax) * self.b) @ self.Ad / self.b.size + self.regcoef * (x + alpha * d) @ d


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    def matvec_Ax(x):
        return A @ x

    def matvec_ATx(x):
        return A.T @ x

    def matmat_ATsA(s):
        return A.T @ diags(s) @ A

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise 'Unknown oracle_type=%s' % oracle_type
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    I = np.eye(x.size)
    grad = np.zeros(x.shape)
    for i in range(x.size):
        grad[i] = (func(x + eps * I[i]) - func(x)) / eps
    return grad


def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i)
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    hess = np.zeros((x.size, x.size))
    I = np.eye(x.size)
    for i in range(x.size):
        for j in range(x.size):
            hess[i, j] = (func(x + eps * I[i] + eps * I[j]) -
                          func(x + eps * I[i]) - func(x + eps * I[j]) +
                          func(x)) / eps ** 2
    return hess
