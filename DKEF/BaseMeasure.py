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
    
    def __init__(self, D, sigma=2.0, trainable=False):

        with tf.name_scope("GaussianBase"):
            self.mu    = tf.Variable([0], dtype=FDTYPE, name="mu", trainable=trainable) 
            self.sigma = tf.Variable([sigma], dtype=FDTYPE, name="sigma", trainable=trainable)

    def get_fun(self, data):
        
        return - 0.5 * tf.reduce_sum((data - self.mu)**2 / self.sigma**2, -1)

    def get_grad(self, data):
        
        return - (data - self.mu) / tf.square(self.sigma)

    def get_sec(self, data):
        
        return -1.0/tf.square(self.sigma) * tf.ones_like(data)

    def get_grad_fun(self, data):
        
        d = (data - self.mu)
        sigma2 = tf.square(self.sigma)
        f = -0.5 * tf.reduce_sum(tf.square(d) / sigma2,-1)
        g = -d / sigma2
        return g, f

    def get_sec_grad(self, data):
        
        sigma2 = tf.square(self.sigma)
        d = (data - self.mu)
        g = -d / sigma2
        s = -1.0/sigma2 * tf.ones_like(data)
        return s, g

    def get_hess(self, data):

        h = -1.0/tf.square(self.sigma) * tf.eye(tf.shape(data)[-1], dtype=FDTYPE, batch_shape=[1])

        return h

    def get_sec_grad_fun(self, data):
        
        sigma2 = tf.square(self.sigma)
        d = (data - self.mu)
        f = -0.5 * tf.reduce_sum(tf.square(d) / sigma2, -1)
        g = -d / sigma2
        s = -1.0/sigma2 * tf.ones_like(data)
        return s, g, f

    def get_hess_grad_fun(self, data):
        
        sigma2 = tf.square(self.sigma)
        d = (data - self.mu)
        f = -0.5 * tf.reduce_sum(tf.square(d) / sigma2, -1)
        g = -d / sigma2
        h = -1.0/sigma2 * tf.eye(tf.shape(data)[-1], dtype=FDTYPE, batch_shape=[1])
        return h, g, f

    def get_hess_grad(self, data):
        
        sigma2 = tf.square(self.sigma)
        d = (data - self.mu)
        g = -d / sigma2
        h = -1.0/sigma2 * tf.eye(tf.shape(data)[-1], dtype=FDTYPE, batch_shape=[1])
        return h, g

