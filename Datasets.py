import numpy as np


class Dataset():

    def sample(self, n):
        raise NotImplementedError
    def logpdf(self, n):
        raise NotImplementedError


class Spiral(Dataset):
    
    def __init__(self, sigma=0.1, D = 2, r_scale=1, starts=[0.0], length=2*np.pi):
        
        self.sigma = sigma
        self.L= length
        self.r_scale = r_scale
        self.D = D
        self.center_eps = 0.01 # add a small noise at the center of spiral
        self.starts= starts
        self.nstart= len(starts)
        self.name = "spiral"

    def sample(self, n):
        
        data = np.zeros((n+self.nstart, self.D))
        batch_size = np.floor_divide(n+self.nstart,self.nstart)
        
        for si, s in enumerate(self.starts):
            m = np.floor_divide(n,self.nstart)
            data[si*batch_size:(si+1)*batch_size] = self.sample_branch(batch_size, s)
        return  data[:n,:]
        
    def sample_branch(self, n, start):
        
        a = np.random.uniform(0,1,n)
        a = self.L * np.cbrt(a) + start

        r = (a-start)*self.r_scale
        
        data = np.zeros((n,self.D))
        data[:,0] = r * np.cos(a)
        data[:,1] = r * np.sin(a)
    
        data += np.random.randn(*data.shape)* self.sigma * ((a-start)/self.L+self.center_eps)[:,None]

        return data

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
