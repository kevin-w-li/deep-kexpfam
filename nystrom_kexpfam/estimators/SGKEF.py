from abc import abstractmethod

import numpy as np
from nystrom_kexpfam.estimators.WrappedEstimatorBase import WrappedEstimatorBase
from nystrom_kexpfam.log import logger
import shogun as sg


class SGKEF(WrappedEstimatorBase):
    """
    Full kernel exponential family model with Gaussian kernel exp(|x-y|^2 / \sigma )
    Fitting complexity O(n^3 d^3)
    """
    def __init__(self, sigma, lmbda, num_threads=1):
        super(SGKEF, self).__init__(num_threads)
        
        self.sigma = sigma
        self.lmbda = lmbda
        self.num_threads = num_threads

    def _instantiate(self, X):
        logger.info("Using Shogun with %d threads." % self.num_threads)
        sg.get_global_parallel().set_num_threads(self.num_threads)
        
        # Shogun uses column major format, so transpose X
        return sg.KernelExpFamily(X.T, self.sigma, self.lmbda)
    
    def fit(self, X):
        self.est = self._instantiate(X)
        self.est.fit()
    
    def grad(self, X_test=None):
        if X_test is not None:
            self.est.set_data(X_test.T)
            
        return self.est.grad_multiple().T
    
    def score(self, X_test=None):
        if X_test is not None:
            self.est.set_data(X_test.T)
            
        return self.est.score()

class SGKEFNy(SGKEF):
    """
    Nystrom kernel exponential family, using m uniformly chosen data points (all components) as basis,
    with Gaussian kernel exp(|x-y|^2 / \sigma ).
    
    Fitting complexity O(n m^2 d^3)
    """
    def __init__(self, m, sigma, lmbda, num_threads=1):
        super(SGKEFNy, self).__init__(sigma, lmbda, num_threads)
        
        self.m = m
    
    @abstractmethod
    def _instantiate(self, X):
        return sg.KernelExpFamilyNystrom(X.T, self.m, self.sigma, self.lmbda)

class SGKEFNyFixedL2_1e_5(SGKEFNy):
    """
    Nystrom kernel exponential family, using m uniformly chosen data points (all components) as basis,
    with Gaussian kernel exp(|x-y|^2 / \sigma ).
    
    Uses a 1e-5 L2 regulariser (in addition to the RKHS reg. lambda)
    
    Fitting complexity O(n m^2 d^3)
    """
    def __init__(self, m, sigma, lmbda, num_threads=1):
        super(SGKEFNyFixedL2_1e_5, self).__init__(m, sigma, lmbda, num_threads)

    @abstractmethod
    def _instantiate(self, X):
        sg.get_global_io().set_loglevel(1)
        return sg.KernelExpFamilyNystrom(X.T, self.m, self.sigma, self.lmbda, 1e-5)

class SGKEFNyDFixedL2_1e_5(SGKEFNy):
    """
    Nystrom kernel exponential family, using m*D uniformly data point components as basis,
    with Gaussian kernel exp(|x-y|^2 / \sigma ).
    
    Uses a 1e-5 L2 regulariser (in addition to the RKHS reg. lambda)
    
    Fitting complexity O(n m^2 d^3)
    """
    def __init__(self, m, sigma, lmbda, num_threads=1):
        super(SGKEFNyDFixedL2_1e_5, self).__init__(m, sigma, lmbda, num_threads)
        
    @abstractmethod
    def _instantiate(self, X):
        N = X.shape[0]
        D = X.shape[1]
        self.basis_mask = self._create_basis_mask(N, D)
        return sg.KernelExpFamilyNystrom(X.T, self.basis_mask.T, self.sigma, self.lmbda, 1e-5)

    def _create_mask(self, num_elements, N, D):
        mask = np.zeros(N * D, dtype=np.bool)
        mask[np.random.choice(N * D, num_elements, replace=False)] = True
        mask = mask.reshape(N, D)
        
        return mask

    @abstractmethod
    def _create_basis_mask(self, N, D):
        return self._create_mask(self.m * D, N, D)


class SGKEFNyD2FixedL2_1e_5(SGKEFNyDFixedL2_1e_5):
    """
    Nystrom kernel exponential family, using m uniformly data point components as basis,
    with Gaussian kernel exp(|x-y|^2 / \sigma ).
    
    Uses a 1e-5 L2 regulariser (in addition to the RKHS reg. lambda)
    
    Fitting complexity O(n m^2)
    """
    def __init__(self, m, sigma, lmbda, num_threads=1):
        super(SGKEFNyD2FixedL2_1e_5, self).__init__(m, sigma, lmbda, num_threads)

    @abstractmethod
    def _create_basis_mask(self, N, D):
        return self._create_mask(self.m, N, D)
