from abc import abstractmethod
from scipy.linalg import cho_solve

import numpy as np
from nystrom_kexpfam.data_generators.Base import DataGenerator
from nystrom_kexpfam.density import log_gaussian_pdf_isotropic, log_gaussian_pdf, \
    sample_gaussian
from nystrom_kexpfam.mathematics import qmult, log_sum_exp, hypercube


class GaussianBase(DataGenerator):
    def __init__(self, D, N_train, N_test):
        super(GaussianBase, self).__init__(D, N_train, N_test)

class IsotropicZeroMeanGaussian(GaussianBase):
    def __init__(self, D, sigma, N_train, N_test):
        super(GaussianBase, self).__init__(D, N_train, N_test)
        self.sigma = sigma
    
    def log_pdf(self, x):
        return log_gaussian_pdf_isotropic(x, self.sigma)
    
    def grad(self, x):
        return log_gaussian_pdf_isotropic(x, self.sigma, compute_grad=True)

    def grad_multiple(self, X):
        return IsotropicZeroMeanGaussian.grad(self, X)
    
    def sample(self, N=None):
        if N is None:
            N=self.N_train
        return np.random.randn(N, self.D) * self.sigma
    
    def get_mean(self):
        return np.zeros(self.D)

class IsotropicGaussian(IsotropicZeroMeanGaussian):
    def __init__(self, mu, sigma, N_train, N_test):
        super(IsotropicGaussian, self).__init__(len(mu), sigma, N_train, N_test)
        self.mu=mu
    
    def log_pdf(self, x):
        log_gaussian_pdf_isotropic(x, self.sigma, self.mu)
    
    def grad(self, x):
        log_gaussian_pdf_isotropic(x, self.sigma, self.mu, compute_grad=True)

    def grad_multiple(self, X):
        return IsotropicGaussian.grad(self, X)
    
    def sample(self, N):
        return IsotropicZeroMeanGaussian.sample(self, N) - self.mu

    def get_mean(self):
        return self.mu.copy()

class FullGaussian(GaussianBase):
    def __init__(self, mu=np.zeros(2), Sigma=np.eye(2), is_cholesky=False):
        self.mu=mu
        if is_cholesky:
            self.L = Sigma
        else:
            self.L = np.linalg.cholesky(Sigma)
    
    def log_pdf(self, x):
        return log_gaussian_pdf(x-self.mu, Sigma=self.L, is_cholesky=True, compute_grad=False)
    
    def grad(self, x):
        return log_gaussian_pdf(x-self.mu, Sigma=self.L, is_cholesky=True, compute_grad=True)

    def grad_multiple(self, X):
        X_centered = X-self.mu
        return np.array([log_gaussian_pdf(x, Sigma=self.L, is_cholesky=True, compute_grad=True) for x in X_centered])
    
    def sample(self, N):
        return sample_gaussian(N=N, mu=self.mu, Sigma=self.L, is_cholesky=True)

    def get_mean(self):
        return self.mu.copy()

class GammaEigenvalueRotatedGaussian(FullGaussian):
    def __init__(self, gamma_shape=1., D=1):
        super(FullGaussian, self).__init__(D)
        
        # Eigenvalues of covariance
        EVs = np.random.gamma(shape=gamma_shape, size=D)
        
        # random orthogonal matrix to rotate
        Q = qmult(np.eye(D))
        Sigma = Q.T.dot(np.diag(EVs)).dot(Q)
        
        FullGaussian.__init__(self, mu=np.zeros(D), Sigma=Sigma, is_cholesky=False)

class Mixture(DataGenerator):
    def __init__(self, D, components, weights):
        assert len(components)>0
        assert len(components) == len(weights)
        
        for component in components:
            if hasattr(component, 'D'):
                assert component.D == D
        
        self.D = D
        self.components = components
        self.log_weights = np.log(weights)
        self.weights = weights

    def sample(self, N):
        comp_inds = np.random.choice(len(self.components), N,
                                     p=self.weights)
        samples = np.zeros((N,self.D))
        for i in range(N):
            samples[i] = np.squeeze(self.components[comp_inds[i]].sample(1))
        
        return samples
        
    def log_pdf(self, x):
        log_pdfs = np.array([c.log_pdf(x) for c in self.components])
        return log_sum_exp(self.log_weights + log_pdfs)
    
    def log_pdf_multiple(self, X):
        return np.array([self.log_pdf(x) for x in X])
    
    def get_mean(self):
        means = np.array([c.get_mean() for c in self.components])
        return np.average(means, weights=self.weights, axis=0)

class GaussianGridWrapped(DataGenerator):
    def __init__(self, D, sigma, N_train, N_test):
        super(GaussianGridWrapped, self).__init__(D, N_train, N_test)
        self.gaussian_grid = GaussianGrid(D, sigma)

    @abstractmethod
    def sample(self, N):
        return self.gaussian_grid.sample(N)
    
    def log_pdf(self, x):
        return self.gaussian_grid.log_pdf(x)
    
    def grad(self, x):
        return self.gaussian_grid.grad(x)
    
    @abstractmethod
    def get_params(self):
        params = super(GaussianGridWrapped, self).get_params()
        params['sigma'] = self.gaussian_grid.sigma
        return params

class GaussianGridWrappedNoGradient(GaussianGridWrapped):
    def __init__(self, D, sigma, N_train, N_test):
        super(GaussianGridWrappedNoGradient, self).__init__(D, sigma, N_train, N_test)
    
    def grad_multiple(self, X):
        raise NotImplementedError

class Gaussian2Mixture(Mixture):
    def __init__(self, D, N_train, N_test, offset=4):
        components = np.array([np.ones(D)*offset, -np.ones(D)*offset])
        weights = np.ones(D)*0.5
        super(Gaussian2Mixture, self).__init__(D, components, weights)

        self.offset = offset
        
    @abstractmethod
    def get_params(self):
        params = super(Gaussian2Mixture, self).get_params()
        params['offset'] = self.offset
        return params

class GaussianGrid(Mixture):
    def __init__(self, D, sigma, sep = 1, weights=None, num_components=None):
        mus = np.array(hypercube(D))
        mus *= sep
        
        if num_components is None:
            num_components = D
        inds = np.random.permutation(len(mus))[:num_components]
        mus = mus[inds]
        mus = mus - mus.mean(0)
        
        self.sigma=sigma
        self.name = "grid"
        
        Sigma = np.eye(D) * sigma
        components = []
        for mu in mus:
            mu = np.squeeze(mu)
            component = FullGaussian(mu=mu, Sigma=Sigma,
                                        is_cholesky=True)
            components += [component]
        
        if weights is None:
            weights = np.ones(len(components))
            weights /= np.sum(weights)
        
        Mixture.__init__(self, D, components, weights)

    def grad(self, x):
        log_pdf_components = np.array([c.log_pdf(x) for c in self.components])
        log_pdf = log_sum_exp(self.log_weights + log_pdf_components)
        neg_log_neg_ratios = log_pdf_components - log_pdf

        # optimization: only compute gradients for coefficients that won't underflow
        log_eps = np.log(np.finfo(np.float32).eps)
        grad_non_zero = neg_log_neg_ratios>log_eps
    
        gs_inner = np.zeros((len(self.components), self.D))
        for k in range(len(self.components)):
            if grad_non_zero[k]:
                c = self.components[k]
                gs_inner[k] = -cho_solve((c.L, True), x-c.mu)


        return np.dot(gs_inner[grad_non_zero].T, np.exp(neg_log_neg_ratios[grad_non_zero]+self.log_weights[grad_non_zero]))

    def grad_multiple(self, X):
        return np.array([self.grad(x) for x in X])

class Dataset(DataGenerator):
    def __init__(self, fname):
        self.fname = fname
    
    @abstractmethod
    def _load_dataset(self):
        X = np.load(self.fname)
        assert len(X) > 1
        return X
    
    def sample_train_test(self, N_train, N_test=None):
        X = self._load_dataset()
        
        assert (type(N_train) == type(N_test)) or \
                ((type(N_train)==np.float) and N_test is None)
        if type(N_train) is np.float:
            assert N_train>0 and N_train<1
            
            N_train = np.max(1, int(np.round(len(X) * N_train)))
        
        perm = np.random.permutation(len(X))
        
        X_train = X[perm[:N_train]]
        X_test = X[perm[N_train:]]
        
        return X_train, X_test
