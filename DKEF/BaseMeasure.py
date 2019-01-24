import tensorflow as tf
import numpy as np
from Utils import *

class BaseMeasure(object):
    
    def __init__(self):
        raise(NotImplementedError)
    
    def get_grad(self):
        raise(NotImplementedError)
        
    def get_sec(self):
        raise(NotImplementedError)
       
class GaussianBase(BaseMeasure):
    
    def __init__(self, D, sigma=2.0, beta = 2.0, trainable=True):
        
        assert beta >= 1.0
        with tf.name_scope("GaussianBase"):
            self.mu    = tf.Variable([0]*D, dtype=FDTYPE, name="mu", trainable=trainable) 
            self.sigma = tf.Variable([sigma]*D,  dtype=FDTYPE, name="sigma", trainable=trainable)
            self.beta  = tf.exp(tf.Variable([0]*D, dtype=FDTYPE, name="beta", trainable=trainable)) + beta - 1.0
        self.D = D

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
        # their rho is our beta
        # their beta is our (1/(2 * sigma^2))^(2/beta)
        t_rho = self.beta
        t_beta = (1. / (2 * self.sigma**2)) ** (2. / t_rho)

        s = tf.abs(tf.random_gamma([n], 1. / t_rho)) ** (1. / t_rho)
        s = s * (tf.floor(tf.random_uniform([n, self.D]) + 0.5) * 2 - 1)
        s = self.mu + 1. / tf.sqrt(t_beta) * s

        logq = self.get_log_normaliser() + self.get_fun(s)
        return s, logq

    def get_log_normaliser(self):
        # https://sccn.ucsd.edu/wiki/Generalized_Gaussian_Probability_Density_Function#Density_Function
        # each of our dimensions is   exp(- |x - mu|^beta / (2 * sigma^2) )
        t_beta = (1./(2*self.sigma**2))**(2./self.beta)
        t_rho = self.beta
        return tf.reduce_sum(0.5 * tf.log(t_beta) - tf.log(2.) - tf.lgamma(1 + 1. / t_rho))

class MixtureBase(BaseMeasure):
    
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
