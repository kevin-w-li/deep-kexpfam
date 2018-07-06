from scipy.linalg import solve_triangular

from kernel_hmc.tools.math import qmult
from kernel_hmc.tools.assertions import assert_positive_int
import numpy as np


def log_gaussian_pdf(x, mu=None, Sigma=None, is_cholesky=False, compute_grad=False):
    if mu is None:
        mu = np.zeros(len(x))
    if Sigma is None:
        Sigma = np.eye(len(mu))
    
    if is_cholesky is False:
        L = np.linalg.cholesky(Sigma)
    else:
        L = Sigma
    
    assert len(x) == Sigma.shape[0]
    assert len(x) == Sigma.shape[1]
    assert len(x) == len(mu)
    
    # solve y=K^(-1)x = L^(-T)L^(-1)x
    x = np.array(x - mu)
    y = solve_triangular(L, x.T, lower=True)
    y = solve_triangular(L.T, y, lower=False)
    
    if not compute_grad:
        log_determinant_part = -np.sum(np.log(np.diag(L)))
        quadratic_part = -0.5 * x.dot(y)
        const_part = -0.5 * len(L) * np.log(2 * np.pi)
        
        return const_part + log_determinant_part + quadratic_part
    else:
        return -y

def sample_gaussian(N, mu=np.zeros(2), Sigma=np.eye(2), is_cholesky=False):
    D = len(mu)
    assert len(mu.shape) == 1
    assert len(Sigma.shape) == 2
    assert D == Sigma.shape[0]
    assert D == Sigma.shape[1]
    
    if is_cholesky is False:
        L = np.linalg.cholesky(Sigma)
    else:
        L = Sigma
    
    return L.dot(np.random.randn(D, N)).T + mu

class GaussianBase(object):
    def __init__(self, D=1):
        assert_positive_int(D)
        self.D = D
    
    def log_pdf(self, x):
        raise NotImplementedError()
    
    def grad(self, x):
        raise NotImplementedError()
    
    def sample(self):
        raise NotImplementedError()

class IsotropicZeroMeanGaussian(GaussianBase):
    def __init__(self, sigma=1., D=1):
        self.sigma = sigma
        GaussianBase.__init__(self, D)
    
    def log_pdf(self, x):
        D = len(x)
        const_part = -0.5 * D * np.log(2 * np.pi)
        quadratic_part = -np.dot(x, x) / (2 * (self.sigma ** 2))
        log_determinant_part = -D * np.log(self.sigma)
        return const_part + log_determinant_part + quadratic_part
    
    def grad(self, x):
        return -x / (self.sigma ** 2)
    
    def sample(self):
        return np.random.randn(self.D) * self.sigma

class GammaEigenvalueRotatedGaussian(GaussianBase):
    def __init__(self, gamma_shape=1., D=1):
        GaussianBase.__init__(self, D)
        
        # place a gamma on the Eigenvalues of a Gaussian covariance
        EVs = np.random.gamma(shape=gamma_shape, size=D)
        
        # random orthogonal matrix to rotate
        Q = qmult(np.eye(D))
        Sigma = Q.T.dot(np.diag(EVs)).dot(Q)
        
        # Cholesky of random covariance
        self.L = np.linalg.cholesky(Sigma)
        
    def log_pdf(self, x):
        return log_gaussian_pdf(x, Sigma=self.L, is_cholesky=True, compute_grad=False)
    
    def grad(self, x):
        return log_gaussian_pdf(x, Sigma=self.L, is_cholesky=True, compute_grad=True)
    
    def sample(self, N):
        return sample_gaussian(N=N, mu=np.zeros(self.D), Sigma=self.L, is_cholesky=True)
