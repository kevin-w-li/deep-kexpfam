import numpy as np
from scipy.special import logsumexp
from scipy.integrate import quad
from scipy.stats import norm
from multiprocessing import Pool


class Dataset():

    def sample(self, n):
        raise NotImplementedError
    def logpdf(self, n):
        raise NotImplementedError


class Spiral(Dataset):
    
    def __init__(self, sigma=0.1, eps=0.1, D = 2, r_scale=1, starts=[0.0], length=2*np.pi):
        
        self.sigma = sigma
        self.L= length
        self.r_scale = r_scale
        self.D = D
        self.eps = eps # add a small noise at the center of spiral
        self.starts= starts
        self.nstart= len(starts)
        self.name = "spiral"

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


class Funnel(Dataset):
    
    def __init__(self, sigma=3.0, D=2, lim=np.inf):
    
        self.sigma=sigma
        self.D=D
        self.lim=lim
        self.low_lim = 0.000
        self.thresh   = lambda x: np.clip(np.exp(-x), self.low_lim, self.lim)
        self.name="funnel"
        
        
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
    
    def logpdf(self, x):
        v = self.thresh(x[:,0])
        return norm.logpdf(x[:,0], 0, self.sigma) + norm.logpdf(x[:,1:], 0, np.sqrt(v)[:,None]).sum(1)
