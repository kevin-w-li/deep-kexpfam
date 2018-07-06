from abc import abstractmethod

import numpy as np
from nystrom_kexpfam.data_generators.Base import DataGenerator
from nystrom_kexpfam.density import rings_log_pdf_grad, rings_sample, rings_log_pdf


class Ring(DataGenerator):
    def __init__(self, D, sigma, N_train, N_test):
        assert D >= 2
        
        super(Ring, self).__init__(D, N_train, N_test)
        
        self.sigma = sigma
        self.radia = np.array([1, 3, 5])
        self.name  = "ring"
        
    def grad_multiple(self, X):
        return rings_log_pdf_grad(X, self.sigma, self.radia)

    def logpdf(self, X):
        return rings_log_pdf(X, self.sigma, self.radia)

    def log_pdf(self,X):
        return self.logpdf(self,X)
    
    @abstractmethod
    def sample(self, N):
        samples = rings_sample(N, self.D, self.sigma, self.radia)
        return samples
    
    @abstractmethod
    def get_params(self):
        params = super(Ring, self).get_params()
        params["sigma"] = self.sigma
        return params

class RingNoGradient(Ring):
    def __init__(self, D, sigma, N_train, N_test):
        super(RingNoGradient, self).__init__(D, sigma, N_train, N_test)
    
    def grad_multiple(self, X):
        raise NotImplementedError
