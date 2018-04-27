from nystrom_kexpfam.autoencoder import DenoisingAutoencoder
from nystrom_kexpfam.estimators.WrappedEstimatorBase import WrappedEstimatorBase
from nystrom_kexpfam.log import logger


class DAE(WrappedEstimatorBase):
    def __init__(self, m, sigma_noise, max_iterations, num_noise_levels, num_threads=1):
        super(DAE, self).__init__(num_threads)
        
        self.sigma_noise = sigma_noise
        self.m = m
        self.max_iterations = max_iterations
        self.num_noise_levels = num_noise_levels

    def fit(self, X):
        logger.info("Using TensorFlow with %d threads." % self.num_threads)
        self.est = DenoisingAutoencoder(self.m, self.sigma_noise, self.max_iterations,
                                   self.num_noise_levels, num_threads=self.num_threads)
        self.est.fit(X)
    def grad(self, X_test):
        return self.est.grad(X_test)
    
    def score(self, X_test):
        return self.est.score(X_test)
