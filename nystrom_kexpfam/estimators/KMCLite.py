from kernel_exp_family.estimators.lite.gaussian import KernelExpLiteGaussian
from nystrom_kexpfam.estimators.WrappedEstimatorBase import WrappedEstimatorBase

import numpy as np

class KMCLite(WrappedEstimatorBase):
    """
    KMCLite with Gaussian kernel exp(|x-y|^2 / \sigma )
    
    Fitting complexity O(m^3 + dn^2)
    """
    def __init__(self, m, sigma, lmbda):
        # number of threads has to be controlled via numpy/lapack
        super(KMCLite, self).__init__()
        
        self.sigma = sigma
        self.lmbda = lmbda
        self.m = m

    def _instantiate(self, D):
        return KernelExpLiteGaussian(self.sigma, self.lmbda, D, self.m,
                                     reg_f_norm=True, reg_alpha_norm=True)
    def fit(self, X):
        D = X.shape[1]
        assert D == X.shape[1]
        self.est = self._instantiate(D)
        self.est.fit(X)
    
    def grad(self, X_test):
        return np.array([self.est.grad(x_test) for x_test in X_test])

    def score(self, X_test):
        return self.est.objective(X_test)
