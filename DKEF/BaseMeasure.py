import tensorflow as tf
import numpy as np
from Utils import *

class BaseMeasureBase(object):
    
    def __init__(self):
        raise(NotImplementedError)
    
    def get_grad(self):
        raise(NotImplementedError)
        
    def get_sec(self):
        raise(NotImplementedError)
       
class GaussianBase(BaseMeasureBase):
    
    def __init__(self, D, sigma=2.0, trainable=True):

        with tf.name_scope("GaussianBase"):
            self.mu    = tf.Variable([0]*D, dtype=FDTYPE, name="mu", trainable=trainable) 
            self.sigma = tf.Variable([sigma]*D,  dtype=FDTYPE, name="sigma", trainable=trainable)
            self.beta  = tf.exp(tf.Variable([0]*D, dtype=FDTYPE, name="beta", trainable=trainable)) + 1

    def get_fun(self, data):
        
        return - 0.5 * tf.reduce_sum(tf.abs(data - self.mu)**self.beta / self.sigma**2, -1)

    def get_grad(self, data):
        d = data - self.mu 
        return - self.beta * tf.abs(d) ** (self.beta-1) * tf.sign(d) / tf.square(self.sigma) / 2.0

    def get_sec(self, data):
        
        return - self.beta * (self.beta-1) * tf.abs(data - self.mu) ** (self.beta-2) / tf.square(self.sigma) / 2.0

    def get_grad_fun(self, data):
        
        d = data - self.mu
        sigma2 = tf.square(self.sigma)
        f = -0.5 * tf.reduce_sum(d**self.beta / sigma2,-1)
        g = - self.beta * tf.abs(d) ** (self.beta-1) * tf.sign(d) / sigma2 / 2.0
        return g, f

    def get_sec_grad(self, data):
        
        sigma2 = tf.square(self.sigma)
        d = data - self.mu
        g = - self.beta * tf.abs(d) ** (self.beta-1) * tf.sign(d) / sigma2 / 2.0
        s = - self.beta * (self.beta-1) * tf.abs(d) ** (self.beta-2) / sigma2 / 2.0
        return s, g

    def get_hess(self, data):
        

        sigma2 = tf.square(self.sigma)
        d = data - self.mu
        s = - self.beta * (self.beta-1) * tf.abs(d) ** (self.beta-2) / sigma2 / 2.0
        h = tf.matrix_diag(s)

        return h

    def get_sec_grad_fun(self, data):
        
        sigma2 = tf.square(self.sigma)
        d = data - self.mu
        f = -0.5 * tf.reduce_sum(tf.abs(d)**self.beta / sigma2,-1)
        g = - self.beta * tf.abs(d) ** (self.beta-1) * tf.sign(s) / sigma2 / 2.0
        s = - self.beta * (self.beta-1) * tf.abs(d) ** (self.beta-2) / sigma2 / 2.0
        return s, g, f

    def get_hess_grad_fun(self, data):
        
        sigma2 = tf.square(self.sigma)
        d = data - self.mu
        f = -0.5 * tf.reduce_sum(tf.abs(d)**self.beta / sigma2,-1)
        g = - self.beta * tf.abs(d) ** (self.beta-1) * tf.sign(d) / sigma2 / 2.0
        s = - self.beta * (self.beta-1) * tf.abs(d) ** (self.beta-2) / sigma2 / 2.0
        h = tf.matrix_diag(s)
        return h, g, f

    def get_hess_grad(self, data):
        
        sigma2 = tf.square(self.sigma)
        d = data - self.mu
        g = - self.beta * tf.abs(d) ** (self.beta-1) * tf.sign(d) / sigma2 / 2.0
        s = - self.beta * (self.beta-1) * tf.abs(d) ** (self.beta-2) / sigma2 / 2.0
        h = tf.matrix_diag(s)
        return h, g

    def sample_logq(self, n):
        
        # https://sccn.ucsd.edu/wiki/Generalized_Gaussian_Probability_Density_Function#Generating_Random_Samples
        # their beta is our gamma
        # their rho is our beta

        beta = (1./(2*self.sigma**2))**(2./self.beta)
        rho   = self.beta
        s = tf.abs(tf.random_gamma([n], 1.0/rho)) ** (1.0 / rho)
        s = s * (tf.floor( tf.random_uniform([n, self.D]) + 0.5) * 2 - 1)
        s = self.mu + 1. / tf.sqrt(beta) * s
        
        print s

        logq  = 0.5 * tf.log(beta) - tf.log(2.0) - tf.lgamma ( 1 + 1. / rho ) - beta ** (rho/2) * tf.abs(s-self.mu)**rho
        logq  = tf.reduce_sum(logq, -1)
        return s, logq

class DeepBase(BaseMeasureBase):
    
    def __init__(self, Ds, init_weight_std):
 
        assert Ds[-1] == (1,)
        with tf.name_scope("DeepBase"):
            layers = []
            for i in range(len(Ds)-1):
                if i != len(Ds)-2:
                    layers += LinearSoftNetwork(Ds[i], Ds[i+1], init_weight_std = init_weight_std),
                else:
                    layers += LinearSoftNetwork(Ds[i], Ds[i+1], init_weight_std = init_weight_std),


        self.network = DeepNetwork(layers, ndim_out = (1,), add_skip=False)

    def get_fun(self, data):
        f = self.network.forward_tensor(data)
        return tf.squeeze(f, 1)

    def get_grad(self, data):
        g = self.network.get_grad_data(data)[0]
        return tf.squeeze(g, 0)

    def get_sec(self, data):
        s = self.network.get_sec_data(data)[0]
        return tf.squeeze(s, 0)

    def get_grad_fun(self, data):
        g, f = self.network.get_grad_data(data)[:2]
        return tf.squeeze(g, 0), tf.squeeze(f, 1)
        
    def get_sec_grad(self, data):
        s, g = self.network.get_sec_grad_data(data)[:2]
        return tf.squeeze(s, 0), tf.squeeze(g, 0)

    def get_hess(self, data):
        h = self.network.get_hess_grad_data(data)[0]
        return tf.squeeze(h, 0)

    def get_sec_grad_fun(self, data):
        s,g,f = self.network.get_sec_grad_data(data)[:3]
        return tf.squeeze(s,0), tf.squeeze(g,0), tf.squeeze(f,1)

    def get_hess_grad_fun(self, data):
        h,g,f = self.network.get_hess_grad_data(data)[:3]
        return tf.squeeze(h,0), tf.squeeze(g,0), tf.squeeze(f,1)

    def get_hess_grad(self, data):
        hg = self.network.get_hess_grad_data(data)[:2]
        return (tf.squeeze(v,0) for v in hg)


class MixtureBase(BaseMeasureBase):
    
    def __init__(self, measures):
        self.measures = measures

    def get_fun(self, data):
        return tf.reduce_sum([m.get_fun(data) for m in self.measures], 0)

    def get_grad(self, data):
        return tf.reduce_sum([m.get_grad(data) for m in self.measures], 0)

    def get_sec(self, data):
        return tf.reduce_sum([m.get_sec(data) for m in self.measures], 0)

    def get_grad_fun(self, data):
        gs = []
        fs = []
        for m in self.measures:
            g, f = m.get_grad_fun(data)
            gs += g,
            fs += f,
        return tf.reduce_sum(gs, 0), tf.reduce_sum(fs, 0)
        
    def get_sec_grad(self, data):
        gs = []
        fs = []
        for m in self.measures:
            g, f = m.get_sec_grad(data)
            gs += g,
            fs += f,
        return tf.reduce_sum(gs, 0), tf.reduce_sum(fs, 0)

    def get_hess(self, data):
        return tf.reduce_sum([m.get_hess(data) for m in self.measures], 0)

    def get_sec_grad_fun(self, data):
        ss = []
        gs = []
        fs = []
        for m in self.measures:
            s, g, f = m.get_sec_grad_fun(data)
            gs += g,
            fs += f,
            ss += s,
        return tf.reduce_sum(ss,0), tf.reduce_sum(gs, 0), tf.reduce_sum(fs, 0)

    def get_hess_grad_fun(self, data):
        ss = []
        gs = []
        fs = []
        for m in self.measures:
            s, g, f = m.get_hess_grad_fun(data)
            gs += g,
            fs += f,
            ss += s,
        return tf.reduce_sum(ss,0), tf.reduce_sum(gs, 0), tf.reduce_sum(fs, 0)

    def get_hess_grad(self, data):
        hs = []
        gs = []
        for m in self.measures:
            h, g = m.get_sec_grad(data)
            gs += g,
            hs += h,
        return tf.reduce_sum(hs, 0), tf.reduce_sum(gs, 0)
