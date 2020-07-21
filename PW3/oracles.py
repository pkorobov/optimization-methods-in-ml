import numpy as np
import scipy
from scipy.special import expit
from scipy.sparse import diags, eye, bmat


class BarrierLassoOracle:
    """
    Oracle for linear regression with l1 regularization:
         func(x) = 1/2 sum_i (a_i^T x - b_i)^2 + regcoef ||x||_1.

    Let A and b be parameters of the linear regression (feature matrix
    and labels vector respectively).

    Parameters
    ----------
        A -- matrix of features
        b -- target vector
        regcoef -- regularization coefficient
        t -- barrier method t_k variable
    """
    def __init__(self, A, b, regcoef, t):
        self.A = A
        self.b = b
        self.regcoef = regcoef
        self.t = t
        self.n = A.shape[1]

    def func(self, z):
        x = z[:self.n]
        u = z[self.n:]
        return self.t * (0.5 * np.sum((self.A @ x - self.b) ** 2) + self.regcoef * np.sum(u)) - np.log(u - x).sum() - np.log(u + x).sum()

    def grad(self, z):
        x = z[:self.n]
        u = z[self.n:]
        d1 = 1 / (x + u)
        d2 = 1 / (u - x)
        grad_x = self.t * self.A.T @ (self.A @ x - self.b) - d1 + d2
        grad_u = self.regcoef * self.t * np.ones(u.shape) - d1 - d2
        return np.hstack([grad_x, grad_u])

    def hess(self, z):
        x = z[:self.n]
        u = z[self.n:]
        denum = (u ** 2 - x ** 2) ** 2
        D1 = 2 * diags((x ** 2 + u ** 2) / denum)
        D2 = -4 * diags(x * u / denum)
        return bmat([[self.t * self.A.T @ self.A + D1, D2], [D2, D1]]).toarray()

    def hess_vec(self, z, v):
        # return self.hess(z) @ v

        x = z[:self.n]
        u = z[self.n:]
        v_x = v[:self.n]
        v_u = v[self.n:]

        denum = (u ** 2 - x ** 2) ** 2
        D1 = 2 * (x ** 2 + u ** 2) / denum
        D2 = -4 * x * u / denum

        hess_v = np.hstack([self.t * self.A.T @ (self.A @ v_x), np.zeros(v_u.shape)]) + \
                 np.hstack([D1 * v_x + D2 * v_u, D2 * v_x + D1 * v_u])
        return hess_v

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

    def update_tau(self, t):
        self.t = t


def lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef):
    """
    Estimates f(x) - f* via duality gap for
        f(x) := 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
    """

    mu = Ax_b * np.minimum(1.0, regcoef / np.linalg.norm(ATAx_b, ord=np.inf))
    return 0.5 * np.linalg.norm(Ax_b) ** 2 + regcoef * np.linalg.norm(x, ord=1) + \
           0.5 * np.linalg.norm(mu) ** 2 + b @ mu
