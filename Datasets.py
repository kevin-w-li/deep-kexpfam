import numpy as np
from scipy.special import logsumexp
from scipy.integrate import quad
from scipy.stats import norm
from multiprocessing import Pool
from nystrom_kexpfam.density import rings_log_pdf_grad, rings_sample, rings_log_pdf
from Utils import support_1d
from sklearn.metrics.pairwise import euclidean_distances

class Dataset(object):

    def sample(self, n):
        raise NotImplementedError

    def sample_two(self, n1, n2):
        raise NotImplementedError


class ToyDataset(Dataset):

    def sample(self, n):
        raise NotImplementedError

    def sample_two(self, n1, n2):
        return self.sample(n1), self.sample(n2)

    def logpdf_multiple(self, x):
        raise NotImplementedError

    def logpdf(self, x):
        return support_1d(self.logpdf_multiple, x)

    def log_pdf(self, x):
        return support_1d(self.logpdf_multiple, x)

    def log_pdf_multile(self, x):
        return self.logpdf_multiple(x)


    def dlogpdf(self, x):
        return grad_multiple(x)

    def grad_multiple(self, x):
        raise NotImplementedError

    def grad(self, x):
        return grad_multiple(self.logpdf, x)



class Spiral(ToyDataset):
    
    def __init__(self, sigma=0.1, D = 2, eps=2.0, r_scale=1.5, starts=np.array([-0.05,0.05,2.0/3,4.0/3]) * np.pi, 
                length=np.pi):

        self.sigma = sigma
        self.L= length
        self.r_scale = r_scale
        self.D = D
        self.eps = eps # add a small noise at the center of spiral
        self.starts= starts
        self.nstart= len(starts)
        self.name = "spiral"
        self.has_grad = False

    def _branch_params(self, a, start):
        
        n = len(a)
        a = self.L * ((a)**(1.0/self.eps))+ start
        r = (a-start)*self.r_scale
        
        m = np.zeros((n, self.D))
        s = np.ones((n, self.D)) * self.sigma
        
        m[:,0] = r * np.cos(a)
        m[:,1:] = (r * np.sin(a))[:,None]
        s[:,:] = (a[:,None]-start)/self.L * self.sigma

        return m, s

    def _branch_params_one(self, a, start):
        
        a = self.L * ((a)**(1.0/self.eps))+ start
        r = (a-start)*self.r_scale
        
        m = np.zeros((self.D))
        s = np.ones((self.D)) * self.sigma
        
        m[0] = r * np.cos(a)
        m[1] = r * np.sin(a)
        s[:2] = (a-start)/self.L * self.sigma

        return m, s

    def sample(self, n):
        
        data = np.zeros((n+self.nstart, self.D))
        batch_size = np.floor_divide(n+self.nstart,self.nstart)
        
        for si, s in enumerate(self.starts):
            m = np.floor_divide(n,self.nstart)
            data[si*batch_size:(si+1)*batch_size] = self.sample_branch(batch_size, s)
        return  data[:n,:]

        
        
    def sample_branch(self, n, start):
        
        a = np.random.uniform(0,1,n)

        m, s = self._branch_params(a, start) 

        data = m + np.random.randn(n, self.D) * s
        return data

    def _conditional_pdf(self, a, x):
        
        n = x.shape[0]
        p = np.array((n,self.nstart))

        for si, s in enumerate(self.starts):
            
            m, s = self._branch_params(a, s)
            pdf[:,si] = norm.logpdf(x, loc = m, scale = s).sum(1)
            pdf[:,si] -= np.log(self.nstart)

        return np.sum(np.exp(pdf), 1)

    def _conditional_pdf_one(self, a, x):
        
        pdf = np.zeros((self.nstart))

        for si, s in enumerate(self.starts):
            
            m, s = self._branch_params_one(a, s)
            pdf[si] = norm.logpdf(x, loc = m, scale = s).sum()
            pdf[si] -= np.log(self.nstart)

        return np.sum(np.exp(pdf))

    def _conditional_dpdf_one_dim(self, a, x, D):

        dpdf = np.zeros((self.nstart))
        
        for si, s in enumerate(self.starts):
            
            m, s = self._branch_params_one(a, s)
            dpdf[si] = np.exp(norm.logpdf(x, loc = m, scale = s).sum()) * ( - x[D] + m[D]) / s[D]**2
            dpdf[si] /= self.nstart

        return dpdf.sum()

    def pdf_one(self, x, *args, **kwargs):
        
        return quad(self._conditional_pdf_one, 0, 1, x, *args, **kwargs)[0]

    def dpdf_one(self, x, *args, **kwargs):
        
        dpdf = np.zeros(self.D)
        for d in range(self.D):
            dpdf[d] = quad(self._conditional_dpdf_one_dim, 0, 1, (x, d), *args, **kwargs)[0]
        return dpdf

    def grad_one(self, x, *args, **kwargs):
        
        return self.dpdf_one(x, *args, **kwargs) / self.pdf_one(x, *args, **kwargs)


class Funnel(ToyDataset):
    
    def __init__(self, sigma=2.0, D=2, lim=10.0):
    
        self.sigma=sigma
        self.D=D
        self.lim=lim
        self.low_lim = 0.000
        self.thresh   = lambda x: np.clip(np.exp(-x), self.low_lim, self.lim)
        self.name="funnel"
        self.has_grad = True
        
        
    def sample(self, n):
        
        data = np.random.randn(n, self.D)
        data[:,0]  *= self.sigma
        v =  self.thresh(data[:,0:1])
        data[:,1:] = data[:,1:] * np.sqrt(v)
        return data
    
    def grad_multiple(self, x):
        
        N = x.shape[0]
        grad = np.zeros((N, self.D))
        x1 = x[:,0]
        
        v = np.exp(-x1)
        
        dv  = -1*v
        dlv = -np.ones_like(v)
        
        dv[(v) < self.low_lim] = 0
        dv[(v) > self.lim] = 0
        
        dlv[(v) < self.low_lim] = 0
        dlv[(v) > self.lim] = 0
        
        grad[:,0] = -x1/self.sigma**2 - (self.D-1)/2.0 * dlv - 0.5*(x[:,1:]**2).sum(1) * (-dv)/v/v
        grad[:,1:]= - x[:,1:] / self.thresh(x1)[:,None]
        return grad
    
    def logpdf_multiple(self, x):
        v = self.thresh(x[:,0])
        return norm.logpdf(x[:,0], 0, self.sigma) + norm.logpdf(x[:,1:], 0, np.sqrt(v)[:,None]).sum(1)

class Ring(ToyDataset):


    def __init__(self, sigma=0.1, D=2):

        assert D >= 2
        
        self.sigma = sigma
        self.D = D
        self.radia = np.array([1, 3, 5])
        self.name  = "ring"
        self.has_grad = True
        
    def grad_multiple(self, X):
        return rings_log_pdf_grad(X, self.sigma, self.radia)

    def logpdf_multiple(self, X):
        return rings_log_pdf(X, self.sigma, self.radia)

    def sample(self, N):
        samples = rings_sample(N, self.D, self.sigma, self.radia)
        return samples

class Banana(ToyDataset):
    
    def __init__(self, bananicity = 0.03, sigma=10, D=2):
        self.bananicity = bananicity
        self.sigma = sigma
        self.D = D
        self.name = "banana"
        self.has_grad = True

    def logpdf_multiple(self,x):
        x = np.atleast_2d(x)
        assert x.shape[1] == self.D
        logp =  norm.logpdf(x[:,0], 0, self.sigma) + \
                norm.logpdf(x[:,1], self.bananicity * (x[:,0]**2-self.sigma**2), 1)
        if self.D > 2:
            logp += norm.logpdf(x[:,2:], 0,1).sum(1)

        return logp

    def sample(self, n):
        
        X = np.random.randn(n, self.D)
        X[:, 0] = self.sigma * X[:, 0]
        X[:, 1] = X[:, 1] + self.bananicity * (X[:, 0] ** 2 - self.sigma**2)
        if self.D > 2:
            X[:,2:] = np.random.randn(n, self.D - 2)
        
        return X

    def grad_multiple(self, x):
        
        x = np.atleast_2d(x)
        assert x.shape[1] == self.D

        grad = np.zeros(x.shape)
        
        quad = x[:,1] - self.bananicity * (x[:,0]**2 - self.sigma**2)
        grad[:,0] = -x[:,0]/self.sigma**2 + quad * 2 * self.bananicity * x[:,0]
        grad[:,1] = -quad
        if self.D > 2:
            grad[:,2:] = -x[:, 2:]
        return grad

class RealDataset(Dataset):


    def __init__(self, idx=None, valid_thresh=1e-6, noise_std = 0.0):

        if idx is not None:
            self.data = self.data[:,idx]
        np.random.shuffle(self.data)
        self.noise_std = noise_std

        self.valid_thresh = valid_thresh
        self.close_mat = euclidean_distances(self.data) > valid_thresh

        self.N, self.D = self.data.shape

        self.data -= self.data.mean(0, keepdims=True)
        self.data /= self.data.std(0,keepdims=True)

    def sample(self, n):

        n = min(n, self.N)
        idx = np.random.choice(self.N,n,replace=False)
        d = self.data[idx]
        if self.noise_std != 0:
            d += np.random.randn(n, self.D) * self.noise_std
        return d
    
    def sample_two(self, n1, n2):

        n = min(n1+n2, self.N)
        idx = np.random.choice(self.N,n1,replace=False)
        s1 = self.data[idx]
        valid_idx = np.where(np.prod(self.close_mat[idx], 0))[0]
        idx = np.random.choice(valid_idx, n2)
        s2 = self.data[idx]

        if self.noise_std != 0:
            s1 += np.random.randn(n1, self.D) * self.noise_std
            s2 += np.random.randn(n2, self.D) * self.noise_std
            
        return s1, s2
     

class WhiteWine(RealDataset):
    
    def __init__(self, idx=None, valid_thresh = 1e-6, noise_std=0.0):
        self.data = np.loadtxt("data/winequality-white.csv", delimiter=";", skiprows=1)[:,:-1]
        self.name="Wine"
        
        super(WhiteWine, self).__init__(idx=idx, valid_thresh=valid_thresh, noise_std = noise_std)
    
class RedWine(RealDataset):
    
    def __init__(self, idx=None, valid_thresh = 1e-6, noise_std=0.0):
        self.data = np.loadtxt("data/winequality-red.csv", delimiter=";", skiprows=1)[:,:-1]
        self.name="Wine"
        
        super(RedWine, self).__init__(idx=idx, valid_thresh=valid_thresh, noise_std = noise_std)
    
