import tensorflow as tf
import numpy as np
from collections import OrderedDict
import operator
import itertools
from time import time
import warnings

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

def construct_index(dim,s="o", n=1):
    ''' construct string for use in tf.einsum as it does not support...'''
    s = ord(s)
    return ''.join([str(unichr(i+s)) for i in range(len(dim)*n)])

FDTYPE="float32"

c = 10000
nl   = lambda x: tf.log(1+tf.exp(x))
dnl  = lambda x: 1/(1+tf.exp(-x))
d2nl = lambda x: tf.exp(-x)/tf.square(1+tf.exp(-x))

'''
nl   = lambda x: tf.where(x<c, tf.log(1+tf.exp(x)), x)
dnl  = lambda x: tf.where(x<-c, tf.zeros_like(x), 1/(1+tf.exp(-x)))
d2nl = lambda x: tf.where(tf.logical_and(-c<x, x<c), tf.exp(-x)/tf.square(1+tf.exp(-x)), tf.zeros_like(x))

nl   = lambda x: tf.where(x<0, tf.exp(0.5*x)-1, tf.where(x<c, tf.log(1+tf.exp(x)), x)-np.log(2))
dnl  = lambda x: tf.where(x<0, 0.50*tf.exp(0.5*x), 1/(1+tf.exp(-x)))
d2nl = lambda x: tf.where(x<0, 0.25*tf.exp(0.5*x), 
                            tf.where(x<c, tf.exp(-x)/tf.square(1+tf.exp(-x)), tf.zeros_like(x)))
'''

def pow_10(x, name, **kwargs): 

    var = tf.Variable(x, dtype=FDTYPE, name="log_" + name, **kwargs)

    return tf.pow(np.array(10,dtype=FDTYPE), var, name=name)

# =====================            
# Kernel related
# =====================            

class LiteModel:


    def __init__(self, kernel, alpha = None, points = None, 
                init_log_lam = 0.0, log_lam_weights=-3, noise_std=0.0, 
                simple_lite=False, lam = None, base=False):
        
        self.kernel = kernel
        self.alpha  = alpha
        self.base   = base
        
        if simple_lite:
            assert lam is not None
            self.lam_norm = lam
            self.lam_alpha = lam
            self.lam_curve = tf.constant(0.0, dtype=FDTYPE, name="lam_curve")
            self.lam_weights = tf.constant(0.0, dtype=FDTYPE, name="lam_weights")
            self.noise_std   = tf.constant(0.0, dtype=FDTYPE, name="noise_std")
        else:
            with tf.name_scope("regularizers"):
                self.lam_norm    = pow_10(init_log_lam, "lam_norm", trainable=True)
                self.lam_alpha   = pow_10(init_log_lam, "lam_alpha", trainable=True)
                self.lam_curve   = pow_10(init_log_lam, "lam_curve", trainable=True)
                self.lam_weights = pow_10(log_lam_weights, "lam_weights", trainable=False)
                self.lam_kde     = pow_10(0, "lam_kde", trainable=False)
                self.noise_std   = noise_std

        if points is not None:
            self.set_points(points)

        if base:
            self.base = GaussianBase(self.ndim_in[0], 2)
        if alpha is None:
            self.alpha = tf.zeros([1], dtype=FDTYPE)


    def _score_statistics(self, data=None, add_noise=False, take_mean=True):
        
        ''' compute the vector b and matrix C
            Y: the input data to the lite model to fit
        '''
        if data is None: 
            data = tf.placeholder(FDTYPE, shape = (None,) + self.ndim_in)
        
        if self.noise_std > 0 and add_noise:
            data = data + self.noise_std * tf.random_normal(tf.shape(data))

        d2kdx2, dkdx = self.kernel.get_sec_grad(self.X, data)
        npoint = tf.shape(self.X)[0]
        ndata  = tf.shape(data)[0]
        
        
        # score     = (alpha * H + qH) + [ (0.5 * alpha * G2 * alpha) + (alpha * G * qG) + (0.5*qG2) ]
        # curvature = (0.5 * alpha * H2 * alpha) + (alpha * HqH)  + (0.5 * qH2)

        H = tf.einsum("ijk->ij", 
                      d2kdx2)
        
        G2 = tf.einsum('ikl,jkl->ijk',
                      dkdx, dkdx)
        
        H2 = tf.einsum('ikl,jkl->ijk',
                      d2kdx2, d2kdx2)

        if take_mean:
            H = tf.reduce_mean(H,1)
            G2 = tf.reduce_mean(G2,2)
            H2 = tf.reduce_mean(H2,2)

        if self.base:

            d2qdx2, dqdx = self.base.get_sec_grad(data)

            GqG  = tf.reduce_sum(dqdx * dkdx, -1)
            qG2 = tf.reduce_sum(tf.square(dqdx), -1)
            qH = tf.reduce_sum(d2qdx2,          -1)

            HqH = tf.reduce_sum(d2qdx2*d2kdx2,    -1)
            qH2 = tf.reduce_sum(tf.square(d2qdx2), -1)

        else:

            GqG= tf.zeros([npoint, ndata], dtype=FDTYPE)
            qG2= tf.zeros([1], dtype=FDTYPE)
            qH = tf.zeros([1], dtype=FDTYPE)

            HqH= tf.zeros([npoint, ndata], dtype=FDTYPE)
            qH2 = tf.zeros([1], dtype=FDTYPE)
            
        if take_mean:

            GqG  = tf.reduce_mean(GqG, -1)
            qG2 = tf.reduce_mean(qG2)
            qH = tf.reduce_mean(qH)

            HqH= tf.reduce_mean(HqH,-1)
            qH2 = tf.reduce_mean(qH2)

        return H, G2, H2, GqG, qG2, qH, HqH, qH2, data
    
    def individual_score(self, data, alpha=None, add_noise=False):
        
        H, G2, H2, GqG, qG2, qH, HqH, qH2, data = self._score_statistics(data=data, add_noise=add_noise, take_mean=False)

        if alpha is None:
            alpha = self.alpha

        s2 = tf.einsum('i,ij->j', alpha, H) + qH
        s1 = 0.5 * (tf.einsum('i,ijk,j->k', alpha, G2, alpha) + qG2) + tf.einsum("i,ij->j", alpha , GqG)
        score  =  s1 + s2

        return score

    def score(self, data=None, alpha=None, add_noise=False):

        H, G2, H2, GqG, qG2, qH, HqH, qH2, data = self._score_statistics(data=data, add_noise=add_noise)

        if alpha is None:
            alpha = self.alpha

        s2 = tf.einsum('i,i->', alpha, H) + qH
        s1 = 0.5 * (tf.einsum('i,ij,j', alpha, G2, alpha) + qG2) + tf.einsum("i,i->", alpha, GqG)
        score  =  s1 + s2

        return score, H, G2, H2, GqG, qG2, qH, HqH, qH2, data

    def kde_loss(self, data, kde):

        kde_delta = kde[:,None] - kde[None,:]
        delta = self.evaluate_gram(self.X, data)[:,:]
        delta = tf.einsum("i,ijk->jk", self.alpha, delta[:,:,None] - delta[:,None,:])
        if self.base:
            q0 = self.base.get_fun(data)
            delta = delta + (q0[:,None] - q0[None,:])
        loss = tf.reduce_sum(tf.square(delta - kde_delta)) / tf.cast((tf.reduce_prod(tf.shape(kde_delta))), FDTYPE)
        return loss


    def opt_alpha(self, data=None, kde=None):
        # score     = (alpha * H + qH) + [ (0.5 * alpha * G2 * alpha) + (alpha * G * qG) + (0.5*qG2) ]
        # curvature = (0.5 * alpha * H2 * alpha) + (alpha * H * qH)  + (0.5 * qH2)

        H, G2, H2, GqG, qG2, qH, HqH, qH2, data = self._score_statistics(data=data)

        quad =  (G2 + 
                self.K*self.lam_norm+
                tf.eye(self.npoint, dtype=FDTYPE)*self.lam_alpha+
                H2 * self.lam_curve)
        
        lin  =  -(H + GqG + 
                HqH * self.lam_curve)
        
        if kde is not None:
            kde  = kde[:100]
            kde_delta = kde[:,None] - kde[None,:]
            if self.base:
                q0 = self.base.get_fun(data[:100])
                kde_delta = kde_delta - (q0[:,None] - q0[None,:])
            delta = self.evaluate_gram(self.X, data[:100])
            delta = delta[:,:,None] - delta[:,None,:]
            npair = tf.cast(tf.reduce_prod(tf.shape(kde_delta)), FDTYPE)
            quad  = quad + self.lam_kde * tf.einsum("mij,nij->mn", delta, delta) / npair
            lin   = lin  + self.lam_kde * tf.einsum("mij,ij->m", delta, kde_delta) / npair
        
        alpha = tf.matrix_solve(quad, lin[:,None])[:,0]
        return alpha,H, G2, H2, GqG, qG2, qH, HqH, qH2, data 


    def opt_score(self, data=None, alpha=None, kde=None):
        '''
        compute regularised score and returns a handle for assign optimal alpha
        '''
        if alpha is None:
            alpha = self.alpha
        
        alpha_opt, H, G2, H2, GqG, qG2, qH, HqH, qH2, data  = self.opt_alpha(data, kde)
        alpha_assign_op = tf.assign(alpha, alpha_opt)

        s2     =  tf.einsum('i,i->', alpha, H) + qH
        s1     =  0.5 * (tf.einsum('i,ij,j', alpha, G2, alpha) + qG2) + tf.einsum("i,i->", alpha, GqG)

        r_norm =  self.get_fun_rkhs_norm()
        l_norm =  self.get_fun_l2_norm()
        curve  =  0.5 * (tf.einsum('i,ij,j', alpha, H2, alpha) + qH2) + tf.einsum("i,i->", alpha, HqH)
        w_norm =  self.get_weights_norm()

        score  =  s1 + s2 + 0.5 * (self.lam_norm  * r_norm + 
                                   self.lam_alpha * l_norm+
                                   self.lam_curve * curve
                                   )
        if kde is not None:
            score = score + 0.5 * self.lam_kde * self.kde_loss(data, kde)

        return alpha_assign_op, score, data
        
    def val_score(self, train_data=None, valid_data=None, test_data=None, train_kde=None, valid_kde=None):
        

        self.alpha, H, G2, H2, GqG, qG2, qH, HqH, qH2, train_data = self.opt_alpha(train_data, train_kde)

        #  ====== validation ======
        score, H, G2, H2, GqG, qG2, qH, HqH, qH2, valid_data = self.score(data=valid_data, alpha=self.alpha, 
                                                add_noise=True)

        if test_data is not None: 
            test_score = self.score(data=test_data, alpha=self.alpha, 
                                                    add_noise=True)[0]
        else:
            test_score = tf.constant(0.0, dtype=FDTYPE)

        r_norm =  self.get_fun_rkhs_norm()
        l_norm =  self.get_fun_l2_norm()
        curve  =  0.5 * (tf.einsum('i,ij,j', self.alpha, H2, self.alpha) + qH2) + tf.einsum("i,i->", self.alpha, HqH)
        w_norm =  self.get_weights_norm()
        loss   =  score + 0.5 * (  w_norm * self.lam_weights
                                   #r_norm * tf.stop_gradient(self.lam_norm)+
                                   #l_norm * tf.stop_gradient(self.lam_alpha)+
                                   #curve  * tf.stop_gradient(self.lam_curve)
                                   )
        if valid_kde is not None:
            k_loss =  self.kde_loss(valid_data, valid_kde)
            loss = loss + 0.5 * self.lam_kde * k_loss
        else:   
            k_loss = tf.zeros([], dtype=FDTYPE)


        return loss, score, train_data, valid_data, r_norm, l_norm, curve, w_norm, k_loss, test_score
        
    def step_score(self, train_data=None, test_data=None):
        
        score, H, G, C, qH, qG, qC, valid_data = self.score(data=train_data, alpha=self.alpha, 
                                                add_noise=True)

        if test_data is not None: 
            test_score = self.score(data=test_data, alpha=self.alpha, 
                                                    add_noise=False)[0]
        else:
            test_score = tf.constant(0.0, dtype=FDTYPE)

        r_norm =  self.get_fun_rkhs_norm()
        l_norm =  self.get_fun_l2_norm()
        curve = tf.einsum('i,ij,j', self.alpha, C, self.alpha) + qC
        w_norm =  self.get_weights_norm()
        loss   =  score + 0.5 * (curve * self.lam_curve + 
                                 w_norm * self.lam_weights)

        return loss, score, train_data, valid_data, r_norm, l_norm, curve, w_norm, test_score
    
    def set_points(self, points):
        
        self.X = points
        self.npoint = tf.shape(points)[0]
        self.ndim_in = tuple( points.shape[1:].as_list() )
        self.K = self.kernel.get_gram_matrix(self.X, self.X)

    def evaluate_gram(self, X, Y):
        return self.kernel.get_gram_matrix(X, Y)

    def evaluate_fun(self, data, alpha=None):

        if alpha is None:
            alpha = self.alpha
        gram = self.kernel.get_gram_matrix(self.X, data)
        
        fv = tf.tensordot(alpha, gram, [[0],[0]])

        if self.base:
            fv = fv + self.base.get_fun(data)
            
        return fv

    def evaluate_grad(self, data, alpha=None):

        if alpha is None:
            alpha = self.alpha
        grad = self.kernel.get_grad(self.X, data)
        gv   = tf.tensordot(alpha, grad, axes=[[0],[0]])

        if self.base:
            gv = gv + self.base.get_grad(data)

        return gv

    def evaluate_hess(self, data, alpha=None):
        
        if alpha is None:
            alpha = self.alpha
        hess = self.kernel.get_hess(self.X, data)
        hv   = tf.tensordot(alpha, hess, axes=[[0],[0]])

        if self.base:
            hv   = hv + self.base.get_hess(data)
        return hv

    def evaluate_grad_fun(self, data, alpha=None):
        
        if alpha is None:
            alpha = self.alpha
        grad, gram = self.kernel.get_grad_gram(self.X, data)
        grad = tf.tensordot(alpha, grad, axes=[[0],[0]])
        fun  = tf.tensordot(alpha, gram, axes=[[0],[0]])

        if self.base:
            qgrad, qfun = self.base.get_grad_fun(data)
            grad = grad + qgrad
            fun  = fun  + qfun

        return grad, fun

    def evaluate_hess_grad_fun(self, data, alpha=None): 

        if alpha is None:
            alpha = self.alpha
        hess, grad, gram = self.kernel.get_hess_grad_gram(self.X, data)
        hess = tf.tensordot(alpha, hess, axes=[[0],[0]])
        grad = tf.tensordot(alpha, grad, axes=[[0],[0]])
        fun  = tf.tensordot(alpha, gram, axes=[[0],[0]])

        if self.base:
            qhess, qgrad, qfun = self.base.get_hess_grad_fun(data)
            grad = grad + qgrad
            fun  = fun  + qfun
            hess = hess + qhess

        return hess, grad, fun
        

    def get_fun_l2_norm(self):
        
        return tf.reduce_sum(tf.square(self.alpha))

    def get_fun_rkhs_norm(self, K=None):
        return tf.einsum('i,ij,j', self.alpha, self.K, self.alpha)

    def get_weights_norm(self):
        return self.kernel.get_weights_norm()


class BaseMeasure(object):
    
    def __init__(self):
        raise(NotImplementedError)
    
    def get_grad(self):
        raise(NotImplementedError)
        
    def get_sec(self):
        raise(NotImplementedError)
       
class GaussianBase(BaseMeasure):
    
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


class Kernel:

    def __init__(self):
        raise(NotImplementedError)

    def get_weights_norm(self):
        # no weights by default
        return tf.constant(0.0, dtype=FDTYPE)

    def _net_forward(self, X):
        # no network by default
        return X

    def get_gram_matrix(self, X, Y):
        raise(NotImplementedError)

    def get_grad(self, X, Y):
        raise(NotImplementedError)

    def get_hess(self, X, Y):
        raise(NotImplementedError)

    def get_sec_grad(self, X, Y):
        raise(NotImplementedError)

    def get_gram_sec_grad(self, X, Y):
        raise(NotImplementedError)

    def get_hess_grad(self, X, Y):
        raise(NotImplementedError)

    def get_grad_gram(self, X, Y):
        raise(NotImplementedError)

    def get_hess_grad_gram(self, X, Y):
        raise(NotImplementedError)

    def get_two_grad_cross_hess(self, X, Y):
        raise(NotImplementedError)

class MixtureKernel(Kernel):
    
    def __init__(self, kernels, props):
        
        assert len(props) == len(kernels)
        self.kernels=kernels
        self.props = props
        self.nkernel = len(kernels)

    def get_gram_matrix(self, X, Y):

        out = tf.zeros([], dtype=FDTYPE)

        for ki in range(self.nkernel):
            out = out + self.kernels[ki].get_gram_matrix(X, Y) * self.props[ki]

        return out

    def get_sec_grad(self, X, Y):
        
        grad = tf.zeros([], dtype=FDTYPE)
        sec  = tf.zeros([], dtype=FDTYPE)

        for ki in range(self.nkernel):
            s, g  = self.kernels[ki].get_sec_grad(X, Y)
            grad  = grad + g * self.props[ki]
            sec   = sec  + s * self.props[ki]
        return sec, grad


    def get_grad(self, X, Y):

        out = tf.zeros([], dtype=FDTYPE)

        for ki in range(self.nkernel):
            out = out + self.kernels[ki].get_grad(X, Y) * self.props[ki]
        return out

    def get_grad_gram(self, X, Y):

        grad = tf.zeros([], dtype=FDTYPE)
        gram = tf.zeros([], dtype=FDTYPE)

        for ki in range(self.nkernel):
            g, k  = self.kernels[ki].get_grad_gram(X, Y)
            grad  = grad + g * self.props[ki]
            gram  = gram + k * self.props[ki]
        return  grad, gram

    def get_hess(self, X, Y):

        out = tf.zeros([], dtype=FDTYPE)
        for ki in range(self.nkernel):
            out = out + self.kernels[ki].get_hess(X, Y) * self.props[ki]
        return out

    def get_hess_grad(self, X, Y):

        hess = tf.zeros([], dtype=FDTYPE)
        grad = tf.zeros([], dtype=FDTYPE)

        for ki in range(self.nkernel):
            h, g  = self.kernels[ki].get_hess_grad(X, Y)
            hess  = hess + h * self.props[ki]
            grad  = grad + g * self.props[ki]
        return  hess, grad


    def get_hess_grad_gram(self, X, Y):

        hess = tf.zeros([], dtype=FDTYPE)
        grad = tf.zeros([], dtype=FDTYPE)
        gram = tf.zeros([], dtype=FDTYPE)

        for ki in range(self.nkernel):
            h, g, k  = self.kernels[ki].get_hess_grad_gram(X, Y)
            hess  = hess + h * self.props[ki]
            grad  = grad + g * self.props[ki]
            gram  = gram + k * self.props[ki]
        return  hess, grad, gram

    def get_weights_norm(self):

        out = tf.zeros([], dtype=FDTYPE)

        for ki in range(self.nkernel):
            out = out + self.kernels[ki].get_weights_norm()
        return out
        

class NetworkKernel(Kernel):
    
    def __init__(self, network):
        
        self.network = network
        self.out_size = np.prod(self.network.ndim_out)

    def get_gram_matrix(self, X, Y):

        X = self.network.forward_tensor(X) 
        Y = self.network.forward_tensor(Y)
        
        X = tf.reshape(X, [-1, self.out_size])
        Y = tf.reshape(Y, [-1, self.out_size])
        
        return tf.matmul(X, Y, transpose_b=True)

    def get_sec_grad(self, X, Y):

        X = self.network.forward_tensor(X)
        d2Y, dY, _, _ = self.network.get_sec_grad_data(Y)

        input_idx = construct_index(self.network.ndim_in)

        X = tf.reshape(X, [-1, self.out_size])

        dY  = tf.reshape(dY,  (self.out_size, -1) + self.network.ndim_in)
        d2Y = tf.reshape(d2Y, (self.out_size, -1) + self.network.ndim_in)
        
        grad = tf.einsum('ij,jk'+input_idx+'->ik' + input_idx, X, dY)
        sec  = tf.einsum('ij,jk'+input_idx+'->ik' + input_idx, X, d2Y)

        return sec, grad


    def get_grad(self, X, Y):

        X = self.network.forward_tensor(X)
        dY, _, _ = self.network.get_grad_data(Y)

        input_idx = construct_index(self.network.ndim_in)

        X = tf.reshape(X, [-1, self.out_size])
        dY  = tf.reshape(dY,  (self.out_size, -1) + self.network.ndim_in)
        
        grad = tf.einsum('ij,jk'+input_idx+'->ik' + input_idx, X, dY)

        return grad

    def get_hess(self, X, Y):

        X = self.network.forward_tensor(X)
        d2Y, _, _, _ = self.network.get_hess_grad_data(Y)

        input_idx = construct_index(self.network.ndim_in, n=2)

        X = tf.reshape(X, [-1, self.out_size])
        d2Y  = tf.reshape(d2Y,  (self.out_size, -1) + self.network.ndim_in*2)
        
        hess = tf.einsum('ij,jk'+input_idx+'->ik' + input_idx, X, d2Y)

        return hess

    def get_hess_grad(self, X, Y):

        X = self.network.forward_tensor(X)
        d2Y, dY, _, _ = self.network.get_hess_grad_data(Y)

        input_idx_1 = construct_index(self.network.ndim_in, n=1)
        input_idx_2 = construct_index(self.network.ndim_in, n=2)

        X = tf.reshape(X, [-1, self.out_size])
        dY  = tf.reshape(dY,  (self.out_size, -1) + self.network.ndim_in)
        d2Y  = tf.reshape(d2Y,  (self.out_size, -1) + self.network.ndim_in*2)
        
        grad = tf.einsum('ij,jk'+input_idx_1+'->ik' + input_idx_1, X, d2Y)
        hess = tf.einsum('ij,jk'+input_idx_2+'->ik' + input_idx_2, X, d2Y)

        return hess, grad

    def get_weights_norm(self):
        return self.network.get_weights_norm()


class CompositeKernel(Kernel):
    
    def __init__(self, kernel, network):

        self.kernel = kernel
        self.network = network

    def _net_forward(self, data):

        X = self.network.forward_tensor(data)
        return X

    def get_gram_matrix(self, X, Y):
        
        X = self._net_forward(X)
        Y = self._net_forward(Y)

        return self.kernel.get_gram_matrix(X, Y)

    def get_sec_grad(self, X, Y):

        X = self._net_forward(X)
        d2ydx2, dydx, Y, _ = self.network.get_sec_grad_data(data=Y)
        hessK, gradK = self.kernel.get_hess_grad(X, Y)

        input_idx = construct_index(self.network.ndim_in)

        dkdx = tf.einsum('ijk,kj'+input_idx+'->ij'+input_idx,
                        gradK, dydx)

        s2 = tf.einsum('ijkl,kj'+input_idx + '->ijl'+input_idx, 
                            hessK, dydx) 
        s2 = tf.einsum('ijl'+input_idx+',lj'+input_idx + '->ij'+input_idx, 
                            s2, dydx) 
        s1 = tf.einsum('ijk,kj'+input_idx + '->ij'+input_idx, 
                            gradK, d2ydx2)
        d2kdx2 = s1 + s2

        return d2kdx2, dkdx

    def get_grad(self, X, Y):

        X = self._net_forward(X)
        dydx, Y, _ = self.network.get_grad_data(data=Y)
        gradK = self.kernel.get_grad(X, Y)

        input_idx = construct_index(self.network.ndim_in)

        dkdx = tf.einsum('ijk,kj'+input_idx+'->ij'+input_idx,
                        gradK, dydx)

        return dkdx

    def get_grad_gram(self, X, Y):

        X = self._net_forward(X)
        d2ydx2, dydx, Y, _ = self.network.get_sec_grad_data(data=Y)
        hessK, gradK = self.kernel.get_hess_grad(X, Y)

        input_idx = construct_index(self.network.ndim_in)

        dkdx = tf.einsum('ijk,kj'+input_idx+'->ij'+input_idx,
                        gradK, dydx)
        
        gram = self.kernel.get_gram_matrix(X, Y)
        return dkdx, gram

    def get_hess(self, X, Y):

        X = self._net_forward(X)
        d2ydx2, dydx, Y, _ = self.network.get_hess_grad_data(data=Y)
        hessK, gradK = self.kernel.get_hess_grad(X, Y)

        input_idx = construct_index(self.network.ndim_in, n=2)
        input_idx_1 = input_idx[:len(self.network.ndim_in)]
        input_idx_2 = input_idx[len(self.network.ndim_in):]

        d2kdx2 = tf.einsum('ijkl,kj'+input_idx_1+',lj'+input_idx_2+'->ij'+input_idx,
                        hessK, dydx, dydx) + \
                 tf.einsum('ijk,kj'+input_idx+"->ij"+input_idx, gradK, d2ydx2)

        return d2kdx2

    def get_hess_grad(self, X, Y):

        X = self._net_forward(X)

        d2ydx2, dydx, Y, _ = self.network.get_hess_grad_data(data=Y)
        hessK, gradK = self.kernel.get_hess_grad(X, Y)

        input_idx = construct_index(self.network.ndim_in, n=2)
        input_idx_1 = input_idx[:len(self.network.ndim_in)]
        input_idx_2 = input_idx[len(self.network.ndim_in):]


        dkdx = tf.einsum('ijk,kj'+input_idx_1+'->ij'+input_idx_1,
                        gradK, dydx)

        d2kdx2 = tf.einsum('ijkl,kj'+input_idx_1+',lj'+input_idx_2+'->ij'+input_idx,
                        hessK, dydx, dydx) + \
                 tf.einsum('ijk,kj'+input_idx+"->ij"+input_idx, gradK, d2kdx2)

        return d2kdx2, dkdx

    def get_hess_grad_gram(self, X, Y):

        X = self._net_forward(X)

        d2ydx2, dydx, Y, _ = self.network.get_hess_grad_data(data=Y)
        hessK, gradK = self.kernel.get_hess_grad(X, Y)

        input_idx = construct_index(self.network.ndim_in, n=2)
        input_idx_1 = input_idx[:len(self.network.ndim_in)]
        input_idx_2 = input_idx[len(self.network.ndim_in):]


        dkdx = tf.einsum('ijk,kj'+input_idx_1+'->ij'+input_idx_1,
                        gradK, dydx)

        d2kdx2 = tf.einsum('ijkl,kj'+input_idx_1+',lj'+input_idx_2+'->ij'+input_idx,
                        hessK, dydx, dydx) + \
                 tf.einsum('ijk,kj'+input_idx+"->ij"+input_idx, gradK, d2ydx2)

        gram = self.kernel.get_gram_matrix(X, Y)

        return d2kdx2, dkdx, gram


    def get_weights_norm(self):
        return self.network.get_weights_norm()

class GaussianKernel(Kernel):

    '''
    Output and derivatives of Gaussain kernels
    X: the data points that define the function, rank 2
    Y: input data, rank 2
    '''

    def __init__(self, sigma = 1.0, trainable=True):
        if isinstance(sigma, float):
            with tf.name_scope("GaussianKernel"):
                self.sigma  = pow_10(sigma, "sigma", trainable=trainable)
        elif type(sigma)==tf.Tensor:
            self.sigma = sigma
        else:
            raise NameError("sigma should be a float or tf.Tensor")
        self.pdist2 = None

    def get_pdist2(self, X, Y):
        
        if X.shape.ndims==1:
            X = X[None,:]
        if Y.shape.ndims==1:
            Y = Y[None,:]
        assert X.shape[1] == Y.shape[1]
        pdist2 = tf.reduce_sum(tf.square(X), axis=1, keep_dims=True)
        pdist2 -= 2.0*tf.matmul(X, Y, transpose_b = True)
        pdist2 += tf.matrix_transpose(tf.reduce_sum(tf.square(Y), axis=1, keep_dims=True))
        self.pdist2 = pdist2
        return pdist2

    def get_gram_matrix(self, X, Y):

        pdist2 = self.get_pdist2(X, Y)
        sigma = self.sigma
        gram = tf.exp(-0.5/sigma*pdist2)
        return gram

    def get_grad(self, X, Y):
        ''' first derivative of the kernel on the second input, dk(x, y)/dy'''

        gram = self.get_gram_matrix(X, Y)[:,:,None]

        # D contrains the vector difference between pairs of x_m and y_i
        # divided by sigma
        D = (tf.expand_dims(X, 1) - tf.expand_dims(Y, 0))/self.sigma

        # K is a vector that has derivatives on all points
        K = (gram * D)

        return K
        
    def get_hess(self, X, Y):

        ''' 
            first derivative of the kernel on the second input, d2k(x, y)/dy2
        '''
        gram = self.get_gram_matrix(X, Y)[:,:,None, None]
        # the first term
        D = ( tf.expand_dims(X, 1) - tf.expand_dims(Y, 0) )/self.sigma
        D2 = tf.einsum('ijk,ijl->ijkl', D, D)
        I  = tf.eye( D.shape[-1].value, dtype=FDTYPE)/self.sigma

        # K is a vector that has the hessian on all points
        K = gram * (D2 - I)

        return K

    def get_sec_grad(self, X, Y):

        gram = self.get_gram_matrix(X, Y)

        # D contrains the vector difference between pairs of x_m and y_i
        # divided by sigma
        D = (tf.expand_dims(X, 1) - tf.expand_dims(Y, 0))/self.sigma

        # K is a vector that has derivatives on all points
        K1 = (gram[:,:,None]* D)
        
        D2 = tf.square(D)
        I  = tf.ones( D.shape[-1].value, dtype=FDTYPE)/self.sigma

        # K is a vector that has the hessian on all points
        K2 = gram[:,:,None] * (D2 - I)

        return K2, K1

    def get_grad_gram(self, X, Y):

        gram = self.get_gram_matrix(X, Y)

        # D contrains the vector difference between pairs of x_m and y_i
        # divided by sigma
        D = (tf.expand_dims(X, 1) - tf.expand_dims(Y, 0))/self.sigma

        # K is a vector that has derivatives on all points
        K1 = (gram[:,:,None]* D)
        
        return  K1, gram

    def get_sec_grad_gram(self, X, Y):

        gram = self.get_gram_matrix(X, Y)

        # D contrains the vector difference between pairs of x_m and y_i
        # divided by sigma
        D = (tf.expand_dims(X, 1) - tf.expand_dims(Y, 0))/self.sigma

        # K is a vector that has derivatives on all points
        K1 = (gram[:,:,None]* D)
        
        D2 = tf.square(D)
        I  = tf.ones( D.shape[-1].value, dtype=FDTYPE)/self.sigma

        # K is a vector that has the hessian on all points
        K2 = gram[:,:,None] * (D2 - I)

        return  K2, K1, gram

    def get_hess_grad(self, X, Y):

        '''
        compute the first derivatives and hessian using one computation of gram matrix 
        '''

        gram = self.get_gram_matrix(X, Y)

        # D contrains the vector difference between pairs of x_m and y_i
        # divided by sigma
        D = (tf.expand_dims(X, 1) - tf.expand_dims(Y, 0))/self.sigma

        # K is a vector that has derivatives on all points
        K1 = (gram[:,:,None]* D)
        
        D2 = tf.einsum('ijk,ijl->ijkl', D, D)
        I  = tf.eye( D.shape[-1].value, dtype=FDTYPE)/self.sigma

        # K is a vector that has the hessian on all points
        K2 = gram[:,:,None,None] * (D2 - I)

        return K2, K1

    def get_hess_grad_gram(self, X, Y):

        gram = self.get_gram_matrix(X, Y)

        # D contrains the vector difference between pairs of x_m and y_i
        # divided by sigma
        D = (tf.expand_dims(X, 1) - tf.expand_dims(Y, 0))/self.sigma

        # K is a vector that has derivatives on all points
        K1 = (gram[:,:,None]* D)
        
        D2 = tf.einsum('ijk,ijl->ijkl', D, D)
        I  = tf.eye( D.shape[-1].value, dtype=FDTYPE)/self.sigma

        # K is a vector that has the hessian on all points
        K2 = gram[:,:,None,None] * (D2 - I)

        return K2, K1, gram

    def get_two_grad_cross_hess(self, X, Y):
        '''
        compute the derivatives of d2k/dx_idy_j, used for MSD
        and also the derivatives w.r.t x and y
        '''

        gram = self.get_gram_matrix(X, Y)
        
        D = (tf.expand_dims(X, 1) - tf.expand_dims(Y, 0))/self.sigma
        # dk_dy
        K2 = (gram[:,:,None] * D)
        # dk_dx
        K1 = -K2

        D2 = tf.einsum('ijk,ijl->ijkl', D, D)
        I  = tf.eye( D.shape[-1].valu, dtype=FDTYPE )/self.sigma

        K3 = gram[:,:,None,None] * (I - D2)

        return K1, K2, K3, gram

class RationalQuadraticKernel:

    def __init__(self, sigma, power=2, trainable=True):
        
        with tf.name_scope("RQKernel"):
            self.sigma  = pow_10(sigma, "sigma", trainable=trainable)
            self.power  = power

    def get_inner(self, X, Y):
        if X.shape.ndims==1:
            X = X[None,:]
        if Y.shape.ndims==1:
            Y = Y[None,:]

        inner = tf.reduce_sum( (X[:,None,:] - Y[None,:,:])**2, -1)
        return (1+inner/self.sigma * 0.5)
        
    def get_gram_matrix(self, X, Y):
        
        inner = self.get_inner(X,Y)
        return inner**(-self.power)
        
    def get_grad(self, X, Y):

        inner = self.get_inner(X,Y)[:,:,None]
        return self.power/self.sigma* inner**(-self.power-1) * (X[:,None,:]-Y[None,:,:])

    def get_hess(self, X, Y):

        s = self.sigma
        p = self.power

        inner = self.get_inner(X,Y)
        diff2 = (X[:,None,:]-Y[None,:,:])
        diff2 = diff2[:,:,:,None] * diff2[:,:,None,:]

        hess  = (p * (p+1) / tf.square(s) * inner**(-p-2))[...,None,None] * diff2
        hess  = hess - (p/s*(inner)**(-p-1))[...,None,None] * tf.eye(tf.shape(X)[-1], batch_shape=[1,1])

        return hess

    def get_hess_grad(self, X, Y):

        s = self.sigma
        p = self.power

        inner = self.get_inner(X,Y)
        diff1 = (X[:,None,:]-Y[None,:,:])
        diff2 = diff1[:,:,:,None] * diff1[:,:,None,:]
        grad  =  self.power/self.sigma* inner[...,None]**(-self.power-1) * diff1

        hess  = (p * (p+1) / tf.square(s) * inner**(-p-2))[...,None,None] * diff2
        hess  = hess - (p/s*(inner)**(-p-1))[...,None,None] * tf.eye(tf.shape(X)[-1], dtype=FDTYPE, batch_shape=[1,1])

        return hess, grad

    def get_hess_grad_gram(self, X, Y):
        
        s = self.sigma
        p = self.power

        inner = self.get_inner(X,Y)
        gram  = inner**(-self.power)

        diff1 = (X[:,None,:]-Y[None,:,:])
        diff2 = diff1[:,:,:,None] * diff1[:,:,None,:]

        grad  =  self.power/self.sigma* inner[...,None]**(-self.power-1) * diff1

        hess  = (p * (p+1) / tf.square(s) * inner**(-p-2))[...,None,None] * diff2
        hess  = hess - (p/s*(inner)**(-p-1))[...,None,None] * tf.eye(tf.shape(X)[-1], dtype=FDTYPE, batch_shape=[1,1])

        return hess, grad, gram


class PolynomialKernel(Kernel):

    def __init__(self, d, c=1.0):
        
        self.d = d
        with tf.name_scope("GaussianKernel"):
            self.c = tf.Variable(c, name="c", dtype=FDTYPE)

    def get_inner(self, X, Y):
        if X.shape.ndims==1:
            X = X[None,:]
        if Y.shape.ndims==1:
            Y = Y[None,:]

        inner = tf.matmul(X, Y, transpose_b=True)
        return inner
        
    def get_gram_matrix(self, X, Y):
        
        inner = self.get_inner(X,Y)
        return (inner+self.c)**self.d
        
    def get_grad(self, X, Y):

        inner = self.get_inner(X,Y)[:,:,None]
        return self.d * (inner+self.c)**(self.d-1) * X[:,None,:] 

    def get_hess(self, X, Y):

        if self.d == 1:
            return tf.zeros((X.shape[0], Y.shape[0], Y.shape[1], Y.shape[1]))
        else:
            inner = self.get_inner(X,Y)[:,:,None,None]
            return self.d*(self.d-1)*(inner+self.c)**(self.d-2) * (X[:,None,None,:]*X[:,None,:,None])

    def get_grad_gram(self, X, Y):

        inner = self.get_inner(X,Y)
        grad = self.d * (inner[...,None]+self.c)**(self.d-1) * X[:,None,:] 

        return grad, (inner+self.c)**self.d

    def get_sec_grad(self, X, Y):

        inner = self.get_inner(X,Y)[:,:,None]
        K1 = self.d * (inner+self.c)**(self.d-1) * X[:,None,:] 

        if self.d == 1:
            K2 = tf.zeros((X.shape[0], Y.shape[0], Y.shape[1]))
        else:
            K2 = self.d*(self.d-1)*(inner+self.c)**(self.d-2) * (X[:,None,:]**2)

        return K2, K1

    def get_hess_grad(self, X, Y):

        inner = self.get_inner(X,Y)[:,:,None]
        K1 = self.d * (inner+self.c)**(self.d-1) * X[:,None,:] 

        if self.d == 1:
            K2 = tf.zeros((X.shape[0], Y.shape[0], Y.shape[1], Y.shape[1]))
        else:
            inner = inner[:,:,:,None]
            K2 = self.d*(self.d-1)*(inner+self.c)**(self.d-2) * (X[:,None,None,:] * X[:,None,:,None])

        return K2, K1

    def get_hess_grad_gram(self, X, Y):

        inner = self.get_inner(X,Y)
        K =   (inner+self.c)**self.d

        inner = inner[:,:,None]
        K1 = self.d * (inner+self.c)**(self.d-1) * X[:,None,:] 

        if self.d == 1:
            K2 = tf.zeros((X.shape[0], Y.shape[0], Y.shape[1], Y.shape[1]))
        else:
            inner = inner[:,:,:,None]
            K2 = self.d*(self.d-1)*(inner+self.c)**(self.d-2) * (X[:,None,None,:] * X[:,None,:,None])

        return K2, K1, K




# =====================            
# Network related
# =====================            
class SumOutputNet:
    
    def __init__(self, net):

        '''
        Container object that stores variables in a network that sums the output over input batch

        sum_output  :   [  sum_j y_0(x_j), sum_j y_1(x_j), sum_j y_2(x_j), .. ,
                        sum_j y_ndim_out(x_j) for j in batch]
                        Create dummy variable so that gradient can be computed. This has the same length
                        as the number of outputs.
                        It works because the following
                        Let the output be y_i(x_j), each data x_j is paired with parameter copy w_j,
                        The i'th value of this variable, denoted by sum_i, is then sum_j y_i(x_j)
                        Then d(sum_i) / d(w_j) = d(y_i(x_j))

        output      :   output of the data through the network without sum
        batch       :   tensorflow placeholder for input batch
        batch_split :   one symbol for each data
        param_copy  :   copy of param_dict that is going to be paired with each data 
        
        '''

        # parameters are duplicated (not copied in memory) to facilitate parallel gradient over input batch
        # it creates different symbols that points to the same parameters via the tf.identity function
        # each copy of the parameter will be paired with a single input data for forward computation
        self.param_copy = [OrderedDict(
                        zip( net.param.keys(), 
                             [tf.identity(v) for v in net.param.values()]
                           )
                    ) for _ in xrange(net.batch_size)]

        # create input placeholder that has batch_size
        self.batch = tf.placeholder(FDTYPE, shape = (net.batch_size,) + net.ndim_in)
        # split it into different symbols to be paired with parameter copies
        self.batch_split = tf.split(self.batch, net.batch_size, axis = 0)

        # output for each data
        self.output = [ net.forward_tensor(one_data[0],param=p) 
                            for (one_data, p) in zip(self.batch_split,  self.param_copy) ]

        # sum the output over batch sum_output[i] = sum_j y_i(x_j)
        self.sum_output = tf.reduce_sum(self.output, axis=0)

class Network(object):
    '''
    Network should be implemented as subclass of Network

    Define a forward_tensor method which takes in data as tf.Tensor (placeholder) of shape
        [ndata, ...], e.g. [ndata, C, H, W] for conv net

    and return the output as a vector
        [ndata, ndim_out]
    This class contains methods that computes gradients w.r.t. data and parameter (may not be required)
    and second derivative w.r.t. data
    
    '''
   
    def __init__(self, ndim_in, ndim_out, init_mean, init_weight_std, scope, keep_prob):

        self.ndim_out  = ndim_out
        self.ndim_in = ndim_in

        with tf.name_scope(scope):
            W = tf.Variable(init_mean + np.random.randn(*(ndim_out + ndim_in))*init_weight_std,
                            name="W", dtype=FDTYPE)
            b = tf.Variable(init_mean + np.random.randn(1, *ndim_out)*init_weight_std,
                            name="b", dtype=FDTYPE)

        self.param = OrderedDict([('W', W), ('b', b)])
        self.scope=scope

        self.keep_prob = keep_prob


    def reshape_data_array(self, x):
        ''' check if input is 1-D and augment if so'''
        if x.ndim == len(self.ndim_in):
            if x.shape == self.ndim_in:
                x = x[None,:]
                single = True
            else:
                raise NameError('input dimension is wrong')
        else:
            if x.shape[1:] != self.ndim_in:
                raise NameError('input dimension is wrong')
            else:
                single = False
        return x, single

    def reshape_data_tensor(self, x):
        ''' check if input is 1-D and augment if so '''
        if x.shape.ndims == len(self.ndim_in):
            if x.shape == self.ndim_in:
                x = tf.expand_dims(x, 0)
                single = True
            else:
                raise NameError('input dimension is wrong')
        else:
            if x.shape[1:] != self.ndim_in:
                raise NameError('input dimension is wrong')
            else:
                single = False
        return x, single
        

    def forward_array(self, data):
        ''' Output of the network given data array as np.array '''
        raise NotImplementedError('should implement in individual networks')

    def forward_tensor(self, data):
        ''' Output of the network given data array as tf.Tensor '''
        raise NotImplementedError('should implement in individual networks')

    def get_weights_norm(self):

        return sum(map(lambda p: tf.reduce_sum(tf.square(p)), self.param.values()))

    def get_grad_param(self, data):

        ''' Compute the derivative of each output to all the parameters
            and store in dictionary, also return network output

            data:   numpy array
            return: grad_value_dict{param_name=>gradients of shape (ndim_out, ninput, param.shape)}
                    network output of shape (ninput, ndim_out)
        '''

        # reshape data to see if there is a single input
        data, single   = self.reshape_data_array(data)
        ninput = data.shape[0]
        # number of batches
        nbatch = ninput/self.batch_size

        son = SumOutputNet(self)
        
        # rearrange the parameter handles into a single array so tf.gradients can consume. It is arrange like:
        #         [ w_1_copy1, w_1_copy_2, w_1_copy_3, ..., w_nparam_copy_batch_size ]
        batch_param = [ son.param_copy[i][k] for k in self.param.keys() for i in xrange(self.batch_size) ]

        # grad will be arranged by [ ndim_out [len(batch_param)] ]
        grad   = [ list(tf.gradients(son.sum_output[oi], batch_param)) for oi in xrange(self.ndim_out)]

        # results stores grad over multiple batches
        results = []
        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        sess.run(init)

        for bi in xrange(nbatch):
            # get a batch
            batch_data = data[bi*self.batch_size:(bi+1)*self.batch_size]
            # run and append results, the first argument is [[w_1_copies], [w_2_copies], network output]
            results.append(sess.run(grad + [son.output], feed_dict = {son.batch : batch_data}))

        # create output dictionary what will be indexed by parameter name.
        # grad_value_dict[parameter_name][output_idx][input_idx][param_dims]
        grad_value_dict = OrderedDict.fromkeys(self.param.keys())
       
        # results are arranged as follows
        # results   [batch_idx  [  [output [param_1_copy_1, param_1_copy_2, ... param_nparam_copy_nbatch]],
        #                           networkoutput
        #                       ]
        #           ]
        # first fix a k, then form an array of [ndim_out x batch_size*nbatch]
        for ki, k in enumerate(self.param.keys()):
            grad_k = np.array([      
                        np.concatenate([
                            results[bi][oi][ki*self.batch_size : (ki+1)*self.batch_size ] for bi in xrange(nbatch)
                        ]) for oi in xrange(self.ndim_out)
                     ])
            grad_value_dict[k] = grad_k

        
        # extract the output by using the last index of each batch
        output_value = np.concatenate([results[bi][-1] for bi in xrange(nbatch)])
        return grad_value_dict, output_value

    def get_grad_data(self):

        ''' get first derivative with respect to data,
            return gradient node, output node and input (feed) node
        '''
        
        t0 = time()
        son = SumOutputNet(self)
        grad = []
        t0 = time()
        t1 = time()
        print 'building grad output'
        for oi in xrange(self.ndim_out):
            g = tf.gradients(son.sum_output[oi], son.batch)[0]
            grad.append(g)
            print '\r%3d out of %3d took %5.3f sec' % ( oi, self.ndim_out, time()-t1),
            ti = time()
        grad   = tf.stack(grad)
        print 'building grad output took %.3f sec' % (time() - t0)

        return grad, tf.stack(son.output), son.batch

    def get_sec_grad_data(self):

        ''' get second derivative node with respect to data,
            return second derivative node, gradient node, network output node and input (feed) node
        '''

        son = SumOutputNet(self)
        # sum over batch again
        grad = tf.stack([ tf.gradients(son.sum_output[oi], son.batch)[0] for oi in xrange(self.ndim_out)])

        sec = []

        sum_grad = tf.reduce_sum(grad, 1)
        g_flat = tf.reshape(sum_grad, [self.ndim_out, -1])

        # this is the linear index for the input dimensions
        raveled_index = np.arange(np.prod(self.ndim_in)).astype('int32')
        unraveled_index = np.array(np.unravel_index(raveled_index, self.ndim_in))

        for oi in xrange(self.ndim_out):
            t0 = time()
            print 'building output %d out of %d' % (oi+1, self.ndim_out)

            # tf.gather_nd(A, idx) takes tensor A and return elements whose indices are specified in idx
            # like [A[i] for i in idx] 
            # The following line takes derivative of each the gradient at oi'th output dimension
            # w.r.t. each input dimension, stack over input dimension
            #   idx looks like:
            #   [   [0, input_dim_1, input_dim_2, ... ,],
            #       [1, input_dim_1, input_dim_2, ... ,],
            #       ...
            #       [batch_size, input_dim_1, input_dim_2, ... ,]
            #   ]
            # And the result below is of shape [prod(self.ndim_in), batch_size]
            sec_oi_ii = tf.stack([ tf.gather_nd(
                                        self._grad_zero(g_flat[oi, i], [son.batch])[0],
                                        np.c_[  np.arange(self.batch_size)[:,None], 
                                                np.tile(unraveled_index[:,i],[self.batch_size,1])]
                                        )
                                    for i in raveled_index])
            print 'took %.3f sec' % (time()-t0)
            sec.append(sec_oi_ii)
        sec_stack = tf.stack(sec)
        # make the output shape [ ndim_out, batch_size, prod(self.ndim_in) ]
        sec_stack = tf.transpose(sec_stack, perm=[0,2,1])
        sec_stack = tf.reshape(sec_stack, (self.ndim_out, self.batch_size,) +  self.ndim_in)
        return sec_stack, grad, tf.stack(son.output), son.batch

    @staticmethod
    def _grad_zero(f, x):
        # x has to be a list
        grad = tf.gradients(f, x)
        for gi, g in enumerate(grad):
            if g is None:
                grad[gi] = tf.zeros_like(x[gi])
        return grad 

    def get_param_size(self):
        
        size = 0.0
        for _, v in self.param.items():
            size += tf.reduce_sum(tf.square(v))
        return size 

def add_dropout(layer, out, *args):
    
    mask = layer.keep_prob
    mask += tf.random_uniform(tf.shape(out), dtype=FDTYPE, seed=1)
    mask = tf.floor(mask)
    out = out / layer.keep_prob * mask

    # number of output dims
    m_out = len(layer.ndim_out)

    # number of output dims
    m_in = len(layer.ndim_in)

    # total number of elements per data point
    d = np.prod(layer.ndim_out)
    
    data_shape = tf.shape(out)
    d_mask = tf.transpose(mask, perm = range(1, m_out+1) + [0])

    ds = []
    for d in args:
        exp_d_mask = tf.identity(d_mask)
        for i in range(len(d.shape) - len(out.shape)):
            exp_d_mask = tf.expand_dims(exp_d_mask, -1)
        ds.append(d * exp_d_mask / layer.keep_prob)

    return out, ds

class LinearSoftNetwork(Network):

    ''' y =  ReLU( W \cdot x + b ) '''

    def __init__(self, ndim_in, ndim_out, init_weight_std = 1.0, init_mean = 0.0, scope="fc1", keep_prob=None):
        
        super(LinearSoftNetwork, self).__init__(ndim_in, ndim_out, init_mean, init_weight_std, scope, keep_prob)
        self.nl = nl
        self.dnl = dnl
        self.d2nl = d2nl

    def forward_array(self, data, param = None):
        
        if param is None:
            param = self.param
        data, single = self.reshape_data_array(data)
            
        data_tensor  = tf.placeholder(FDTYPE, shape= (None, ) + self.ndim_in)

        W = param['W']
        b = param['b']
        out = tf.matmul(data_tensor, W,  transpose_b = True)
        out += b
        out = self.nl(out)

        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        sess.run(init)
        out_value = sess.run(out, feed_dict={data_tensor: data})

        if self.keep_prob is not None:
            out_value *= np.random.rand(*out_value.shape)<sess.run(self.keep_prob)

        if single:
            out_value = out_value[0]
        return out_value

    def forward_tensor(self, data, param = None):
        
        if param is None:
            param = self.param
        data, single = self.reshape_data_tensor(data)
            
        W = param['W']
        b = param['b']
        out = tf.matmul(data, W,  transpose_b = True)
        out += b
        out = self.nl(out)

        if self.keep_prob is not None:
            out = add_dropout(self, out)[0]

        if single:
            out = out[0]
        return out

    def get_grad_data(self, data=None):

        param = self.param
        
        if data is None:
            data = tf.placeholder(FDTYPE, shape = (None,) + self.ndim_in)
        data, single = self.reshape_data_tensor(data)

        W = param['W']
        b = param['b']
        lin_out = tf.matmul(data, W,  transpose_b = True) + b
        out  = self.nl(lin_out)
        grad = self.dnl(lin_out)[:,:,None] * W[None,:,:]
        grad = tf.transpose(grad, [1,0,2])

        if self.keep_prob is not None:
            out, ds = add_dropout(self, out, grad)
            grad = ds[0]

        return grad, out, data
  
    def get_sec_grad_data(self, data=None):

        if data is None:
            data = tf.placeholder(FDTYPE, shape = (None,) + self.ndim_in)
        param = self.param

        # create input placeholder that has batch_size
        data, single = self.reshape_data_tensor(data)
        W = param['W']
        b = param['b']

        lin_out = tf.matmul(data, W,  transpose_b = True) + b
        out  = self.nl(lin_out)

        grad = self.dnl(lin_out)[:,:,None] * W[None,:,:]
        grad = tf.transpose(grad, [1,0,2])

        sec = self.d2nl(lin_out)[:,:,None] * tf.square(W[None,:,:])
        sec = tf.transpose(sec, [1,0,2])

        if self.keep_prob is not None:
            out, ds = add_dropout(self, out, sec, grad)
            sec, grad = ds

        return sec, grad, out, data

    def get_hess_grad_data(self, data = None):

        if data is None:
            data = tf.placeholder(FDTYPE, shape = (None,) + self.ndim_in)

        param = self.param

        # create input placeholder that has batch_size
        data, single = self.reshape_data_tensor(data)
        W = param['W']
        b = param['b']

        lin_out = tf.matmul(data, W,  transpose_b = True) + b
        out  = self.nl(lin_out)

        grad = self.dnl(lin_out)[:,:,None] * W[None,:,:]
        grad = tf.transpose(grad, [1,0,2])

        hess = self.d2nl(lin_out)[:,:,None,None] * W[None,:,:,None] * W[None,:,None,:]
        hess = tf.transpose(hess, [1,0,2,3])

        if self.keep_prob is not None:
            out, ds = add_dropout(self, out, hess, grad)
            hess, grad = ds

        return hess, grad, out, data

class DenseLinearSoftNetwork(Network):

    ''' y =  ReLU( W \cdot x + b ) '''

    def __init__(self, ndim_in, ndim_out, init_weight_std = 1.0, init_mean = 0.0, scope="skip"):
        
        self.ndim_out  = ndim_out
        self.ndim_in = ndim_in
        self.nin     = len(ndim_in)

        self.param = OrderedDict()

        with tf.name_scope(scope):

            for i in range(self.nin):
                W = tf.Variable(init_mean + np.random.randn(*(ndim_out + ndim_in[i]))*init_weight_std,
                                name="W"+str(i), dtype=FDTYPE)
                self.param["W"+str(i)] = W

            self.param["b"] = tf.Variable(init_mean + np.random.randn(1, *ndim_out)*init_weight_std,
                            name="b", dtype=FDTYPE)

        self.scope=scope
        self.nl = nl
        self.dnl = dnl
        self.d2nl = d2nl
        
    def forward_array(self, data, param = None):
        
        if param is None:
            param = self.param
        
        
        data_tensor  = [tf.placeholder(FDTYPE, shape= (None, ) + self.ndim_in[i]) for i in range(self.nin)]
        
        out = 0.0
        for i in range(self.nin):
            W = param['W'+str(i)]
            out += tf.matmul(data_tensor[i], W,  transpose_b = True)

        b = param['b']
        out += b
        out = self.nl(out)

        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        sess.run(init)

        out_value = sess.run(out, feed_dict = dict(zip(data_tensor, data)) )

        return out_value

    def forward_tensor(self, data, param = None):
        
        if param is None:
            param = self.param
            
        out = 0.0
        for i in range(self.nin):
            W = param['W'+str(i)]
            out += tf.matmul(data[i], W,  transpose_b = True)
        b = param['b']
        out += b
        out = self.nl(out)

        return out

    def get_grad_data(self, data=None):

        param = self.param
        
        if data is None:
            data = [tf.placeholder(FDTYPE, shape= (None, ) + self.ndim_in[i]) for i in range(self.nin)]

        lin_out = 0.0
        for i in range(self.nin):
            W = param['W'+str(i)]
            lin_out += tf.matmul(data[i], W,  transpose_b = True)
        b = param['b']
        lin_out += b

        out  = self.nl(lin_out)
        grad = [self.dnl(lin_out)[:,:,None] * param['W'+str(i)][None,:,:] for i in range(self.nin)]
        grad = [tf.transpose(grad[i], [1,0,2]) for i in range(self.nin)]

        return grad, out, data
  
    def get_sec_grad_data(self, data=None):

        param = self.param

        if data is None:
            data = [tf.placeholder(FDTYPE, shape= (None, ) + self.ndim_in[i]) for i in range(self.nin)]

        lin_out = 0.0
        for i in range(self.nin):
            W = param['W'+str(i)]
            lin_out += tf.matmul(data[i], W,  transpose_b = True)
        b = param['b']
        lin_out += b
        out  = self.nl(lin_out)

        grad = [self.dnl(lin_out)[:,:,None] * param["W"+str(i)][None,:,:] for i in range(self.nin)]
        grad = [tf.transpose(grad[i], [1,0,2]) for i in range(self.nin)]

        sec  = [self.d2nl(lin_out)[:,:,None] * tf.square(param["W"+str(i)][None,:,:]) for i in range(self.nin)]
        sec  = [tf.transpose(sec[i], [1,0,2]) for i in range(self.nin)]
        return sec, grad, out, data

    def get_hess_cross_grad_data(self, data = None):

        if data is None:
            data = [tf.placeholder(FDTYPE, shape= (None, ) + self.ndim_in[i]) for i in range(self.nin)]

        param = self.param

        lin_out = 0.0
        for i in range(self.nin):
            W = param['W'+str(i)]
            lin_out += tf.matmul(data[i], W,  transpose_b = True)
        b = param['b']
        lin_out += b
        out  = self.nl(lin_out)

        grad = [self.dnl(lin_out)[:,:,None] * param["W"+str(i)][None,:,:] for i in range(self.nin)]
        grad = [tf.transpose(grad[i], [1,0,2]) for i in range(self.nin)]

        hess  = [self.d2nl(lin_out)[:,:,None,None] * param["W"+str(i)][None,:,:,None] * param["W"+str(i)][None,:,None,:] 
                    for i in range(self.nin)]
        hess  = [tf.transpose(hess[i], [1,0,2,3]) for i in range(self.nin)]

        cross = self.d2nl(lin_out)[:,:,None,None] * param["W"+str(0)][None,:,:,None] * param["W"+str(1)][None,:,None,:] 
        cross = tf.transpose(cross, [1,0,2,3])

        return hess, cross, grad, out, data

class DeepNetwork(Network):

    def __init__(self, layers, init_mean = 0.0, init_weight_std = 1.0, ndim_out = None, add_skip=False):

        self.ndim_in  = layers[0].ndim_in
        if add_skip:
            assert ndim_out is not None
            self.ndim_out = ndim_out
        else:
            self.ndim_out = layers[-1].ndim_out
        self.nlayer   = len(layers)

        self.layers   = layers
        self.param    = {}
        self.add_skip = add_skip

        for i in range(self.nlayer-1):
            assert layers[i+1].ndim_in == layers[i].ndim_out
        
        layer_count = 1
        for i, l in enumerate(layers):
            if type(l) == DropoutNetwork:
                continue
            layer_str = str(layer_count)
            p = l.param
            for k, v in p.items():
                self.param[k+layer_str] = v

            layer_count += 1
        
        if self.add_skip:

            self.skip_layer = DenseLinearSoftNetwork([self.ndim_in, layers[-1].ndim_out], self.ndim_out, 
                                init_mean=init_mean, init_weight_std=init_weight_std, scope="skip")

            for k, v in self.skip_layer.param.items():
                self.param[k+"skip"] = v

    def forward_tensor(self, data):
        
        d = data

        for i in range(self.nlayer):
            d = self.layers[i].forward_tensor(d) 
        
        if self.add_skip:
            d = self.skip_layer.forward_tensor([data,d])

        return d

    def forward_array(self, data):

        d = data
        for i in range(self.nlayer):
            d = self.layers[i].forward_array(d) 

        if self.add_skip:
            d = self.skip_layer.forward_array([data,d])
        return d


    def get_grad_data(self, data = None):
        
        if data is None:
            data = tf.placeholder(FDTYPE, shape = (None,) + self.ndim_in)

        grad, out, _ = self.layers[0].get_grad_data(data)

        i_idx_l = construct_index(self.layers[0].ndim_in, s="o")

        for i in range(1,self.nlayer):

            layer = self.layers[i]

            i_idx_h = construct_index(layer.ndim_in, s="j")
            o_idx_h = construct_index(layer.ndim_out, s="a")

            this_grad, out, _ = layer.get_grad_data(out)

            grad = tf.einsum(o_idx_h+"i"+i_idx_h+","\
                            +i_idx_h+"i"+i_idx_l+"->"\
                            +o_idx_h+"i"+i_idx_l,  this_grad, grad)

            i_idx_l = construct_index(layer.ndim_in, s="o")
        
        if self.add_skip:
            
            layer = self.skip_layer
            skip_grad, skip_out, _ = layer.get_grad_data([data,out])

            i_idx_h = construct_index(layer.ndim_in[1], s="j")
            o_idx_h = construct_index(layer.ndim_out, s="a")

            grad = tf.einsum(o_idx_h+"i"+i_idx_h+","\
                            +i_idx_h+"i"+i_idx_l+"->"\
                            +o_idx_h+"i"+i_idx_l,  skip_grad[1], grad)

            grad = grad + skip_grad[0]
            out  = skip_out

        return grad, out, data
            
    def get_sec_grad_data(self, data = None):
        
        if data is None:
            data = tf.placeholder(FDTYPE, shape = (None,) + self.ndim_in, name="input")

        sec, grad, out, _ = self.layers[0].get_sec_grad_data(data)

        i_idx_l = construct_index(self.layers[0].ndim_in, s="o")

        for i in range(1,self.nlayer):


            layer = self.layers[i]

            i_idx_h = construct_index(layer.ndim_in, s="j", n=2)
            i_idx_h_1 = i_idx_h[:len(i_idx_h)/2]
            i_idx_h_2 = i_idx_h[len(i_idx_h)/2:]

            o_idx_h = construct_index(layer.ndim_out, s="a")

            this_hess, this_grad, out, _ = self.layers[i].get_hess_grad_data(out)

            sec =  tf.einsum(o_idx_h+"i"+i_idx_h+","+
                             i_idx_h_1+"i"+i_idx_l+","+
                             i_idx_h_2+"i"+i_idx_l+"->"+
                             o_idx_h + "i"+i_idx_l,  this_hess, grad, grad) + \
                   tf.einsum(o_idx_h+"i"+i_idx_h_1+","+
                             i_idx_h_1+"i"+i_idx_l+"->"+
                             o_idx_h+"i"+i_idx_l,  this_grad, sec) 

            grad = tf.einsum(o_idx_h+"i"+i_idx_h_1+","\
                            +i_idx_h_1+"i"+i_idx_l+"->"\
                            +o_idx_h+"i"+i_idx_l,  this_grad, grad)

            i_idx_l = construct_index(self.layers[i-1].ndim_in, s="o",n=1)

        if self.add_skip:

            layer = self.skip_layer

            skip_sec, skip_grad, skip_out, _ = layer.get_sec_grad_data([data,out])

            i_idx_h   = construct_index(layer.ndim_in[1], s="j", n=2)
            i_idx_h_1 = i_idx_h[:len(i_idx_h)/2]
            i_idx_h_2 = i_idx_h[len(i_idx_h)/2:]
            i_idx_h_c_1   = construct_index(layer.ndim_in[0], s="j", n=1)
            i_idx_h_c_2   = construct_index(layer.ndim_in[1], s="p", n=1)
            o_idx_h   = construct_index(layer.ndim_out, s="a")

            this_hess, this_cross, _, _, _ = layer.get_hess_cross_grad_data([data,out])

            sec =  tf.einsum(o_idx_h+"i"+i_idx_h+","+
                             i_idx_h_1+"i"+i_idx_l+","+
                             i_idx_h_2+"i"+i_idx_l+"->"+
                             o_idx_h + "i"+i_idx_l,  this_hess[1], grad, grad) + \
                   tf.einsum(o_idx_h+"i"+i_idx_h_1+","+
                             i_idx_h_1+"i"+i_idx_l+"->"+
                             o_idx_h+"i"+i_idx_l,  skip_grad[1], sec) + \
                   tf.einsum(o_idx_h+"i"+i_idx_h_c_1+i_idx_h_c_2+","+
                             i_idx_h_c_2+"i"+i_idx_h_c_1+"->"+
                             o_idx_h+"i"+i_idx_h_c_1, this_cross, grad) * 2

            grad = tf.einsum(o_idx_h+"i"+i_idx_h_1+","\
                            +i_idx_h_1+"i"+i_idx_l+"->"\
                            +o_idx_h+"i"+i_idx_l,  skip_grad[1], grad)

            sec  = sec  + skip_sec[0]
            grad = grad + skip_grad[0]
            out  = skip_out

        return sec, grad, out, data

    def get_hess_grad_data(self, data = None):

        if data is None:
            data = tf.placeholder(FDTYPE, shape = (None,) + self.ndim_in, name="input")

        hess, grad, out, _ = self.layers[0].get_hess_grad_data(data)

        i_idx_l = construct_index(self.layers[0].ndim_in, s="o",n=2)
        i_idx_l_1 = i_idx_l[:len(i_idx_l)/2]
        i_idx_l_2 = i_idx_l[len(i_idx_l)/2:]

        for i in range(1,self.nlayer):


            layer = self.layers[i]

            i_idx_h = construct_index(layer.ndim_in, s="j", n=2)
            i_idx_h_1 = i_idx_h[:len(i_idx_h)/2]
            i_idx_h_2 = i_idx_h[len(i_idx_h)/2:]

            o_idx_h = construct_index(layer.ndim_out, s="a")

            this_hess, this_grad, out, _ = self.layers[i].get_hess_grad_data(out)

            hess =  tf.einsum(o_idx_h +"i"+i_idx_h  +","+
                             i_idx_h_1+"i"+i_idx_l_1+","+
                             i_idx_h_2+"i"+i_idx_l_2+"->"+
                             o_idx_h  +"i"+i_idx_l,  this_hess, grad, grad) + \
                   tf.einsum(o_idx_h  +"i"+i_idx_h_1+","+
                             i_idx_h_1+"i"+i_idx_l  +"->"+
                             o_idx_h  +"i"+i_idx_l,  this_grad, hess)

            grad = tf.einsum(o_idx_h+"i"+i_idx_h_1+","\
                            +i_idx_h_1+"i"+i_idx_l_1+"->"\
                            +o_idx_h+"i"+i_idx_l_1,  this_grad, grad)

            i_idx_l = construct_index(layer.ndim_in, s="o",n=2)
            i_idx_l_1 = i_idx_l[:len(i_idx_l)/2]
            i_idx_l_2 = i_idx_l[len(i_idx_l)/2:]

        if self.add_skip:

            layer = self.skip_layer

            skip_hess, skip_cross, skip_grad, skip_out, _ = layer.get_hess_cross_grad_data([data,out])

            i_idx_h = construct_index(layer.ndim_in[1], s="j", n=2)
            i_idx_h_1 = i_idx_h[:len(i_idx_h)/2]
            i_idx_h_2 = i_idx_h[len(i_idx_h)/2:]
            i_idx_h_c_1   = construct_index(layer.ndim_in[0], s="q", n=1)
            i_idx_h_c_2   = construct_index(layer.ndim_in[1], s="u", n=1)
            o_idx_h = construct_index(layer.ndim_out, s="a")


            hess_cross = tf.einsum(o_idx_h+"i"+i_idx_h_c_1+i_idx_h_c_2+","+
                             i_idx_h_c_2+"i"+i_idx_h_1+"->"+
                             o_idx_h+"i"+i_idx_h_c_1+i_idx_h_1, skip_cross, grad)

            hess =  tf.einsum(o_idx_h +"i"+i_idx_h  +","+
                             i_idx_h_1+"i"+i_idx_l_1+","+
                             i_idx_h_2+"i"+i_idx_l_2+"->"+
                             o_idx_h  +"i"+i_idx_l,  skip_hess[1], grad, grad) + \
                   tf.einsum(o_idx_h  +"i"+i_idx_h_1+","+
                             i_idx_h_1+"i"+i_idx_l  +"->"+
                             o_idx_h  +"i"+i_idx_l,  skip_grad[1], hess) + \
                   hess_cross + tf.transpose(hess_cross, [0,1,3,2])

            grad = tf.einsum(o_idx_h+"i"+i_idx_h_1+","\
                            +i_idx_h_1+"i"+i_idx_l_1+"->"\
                            +o_idx_h+"i"+i_idx_l_1,  skip_grad[1], grad)

            hess = hess + skip_hess[0]
            grad = grad + skip_grad[0]
            out  = skip_out

        return hess, grad, out, data
        

class LinearNetwork(Network):

    ''' y =  W \cdot x + b '''

    def __init__(self, ndim_in, ndim_out, batch_size = 2, init_weight_std = 1.0, init_mean = 0.0, identity=False,
                scope="fc1", keep_prob=None):
        super(LinearNetwork, self).__init__(ndim_in, ndim_out, init_mean, init_weight_std, scope, keep_prob) 
        self.batch_size = batch_size


    def forward_array(self, data, param = None):
        
        if param is None:
            param = self.param
        data, single = self.reshape_data_array(data)
            
        data_tensor  = tf.placeholder(FDTYPE, shape= (None, ) + self.ndim_in)

        W = param['W']
        b = param['b']
        out = tf.matmul(data_tensor, W,  transpose_b = True)
        out += b


        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        sess.run(init)
        out_value = sess.run(out, feed_dict={data_tensor: data})
        if single:
            out_value = out_value[0]
        return out_value

    def forward_tensor(self, data, param = None):
        
        if param is None:
            param = self.param
        data, single = self.reshape_data_tensor(data)
            
        W = param['W']
        b = param['b']
        out = tf.matmul(data, W,  transpose_b = True)

        out += b

        if single:
            out = out[0]
        return out

    def get_grad_data(self, data=None):

        param = self.param

        # create input placeholder that has batch_size
        if data is None:
            data = tf.placeholder(FDTYPE, shape = (None,) + self.ndim_in)
        data, single = self.reshape_data_tensor(data)
        W = param['W']
        output = self.forward_tensor(data, param)
        N = tf.shape(data)[0]
        grad = tf.tile(W[None,:,:], [N, 1, 1])
        grad = tf.transpose(grad, [1,0,2])
        return grad, output, data

    def get_sec_grad_data(self, data=None):

        if data is None:
            data = tf.placeholder(FDTYPE, shape = (None,) + self.ndim_in)
        param = self.param

        # create input placeholder that has batch_size
        #data = tf.placeholder(FDTYPE, shape = (self.batch_size,) + self.ndim_in)
        data, single = self.reshape_data_tensor(data)
        W = param['W']
        out  = self.forward_tensor(data, param)
        N = tf.shape(data)[0]
        grad = tf.tile(W[None,:,:], [N, 1, 1])
        grad = tf.transpose(grad, [1,0,2])

        sec = tf.zeros_like(grad)
        return sec, grad, out, data

    def get_hess_grad_data(self, data=None):

        if data is None:
            data = tf.placeholder(FDTYPE, shape = (None,) + self.ndim_in)
        param = self.param

        # create input placeholder that has batch_size
        #data = tf.placeholder(FDTYPE, shape = (self.batch_size,) + self.ndim_in)
        data, single = self.reshape_data_tensor(data)
        W = param['W']
        out  = self.forward_tensor(data, param)
        N = tf.shape(data)[0]
        grad = tf.tile(W[None,:,:], [N, 1, 1])
        grad = tf.transpose(grad, [1,0,2])

        sec = tf.zeros([N, self.ndim_out, self.ndim_in, self.ndim_in], dtype=FDTYPE)
        return sec, grad, out, data

        

class SquareNetwork(Network):
    
    ''' y =  ( W \cdot x + b ) ** 2 
        for testing automatic differentiation
    '''

    def __init__(self, ndim_in, ndim_out, batch_size = 2):
        
        self.ndim_out  = ndim_out
        self.ndim_in = ndim_in
        self.batch_size = batch_size
        W     = tf.Variable(np.random.randn(ndim_out, *ndim_in).astype(FDTYPE))
        b      = tf.Variable(np.random.randn(1, ndim_out).astype(FDTYPE))
        self.param = OrderedDict([('W', W), ('b', b)])
        self.out   = None

    def forward_array(self, data, param = None):
        
        if param is None:
            param = self.param
        data, single = self.reshape_data_array(data)
            
        data_tensor  = tf.placeholder(FDTYPE, shape= (None, ) + self.ndim_in)

        W = param['W']
        b = param['b']
        out = tf.matmul(data_tensor, W,  transpose_b = True)
        out += b
        out = tf.square(out)
        
        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        sess.run(init)
        out_value = sess.run(out, feed_dict={data_tensor: data})
        if single:
            out_value = out_value[0]
        return out_value

    def forward_tensor(self, data, param = None):
        
        if param is None:
            param = self.param
        data, single = self.reshape_data_tensor(data)
            
        W = param['W']
        b = param['b']
        out = tf.matmul(data, W,  transpose_b = True)
        out += b
        out = tf.square(out)

        if single:
            out = out[0]
        return out


class LinearReLUNetwork(Network):

    ''' y =  ReLU( W \cdot x + b ) '''

    def __init__(self, ndim_in, ndim_out, batch_size = 2, 
                init_weight_std = 1.0, init_mean = 0.0, 
                grads = [0.0, 1.0]):
        
        self.ndim_out  = ndim_out
        self.ndim_in = ndim_in
        self.batch_size = batch_size
        W   = tf.Variable(init_mean + np.random.randn(*ndim_out + ndim_in).astype(FDTYPE))*init_weight_std
        b   = tf.Variable(np.random.randn(1, *ndim_out).astype(FDTYPE))*init_weight_std
        self.grads = grads
        self.param = OrderedDict([('W', W), ('b', b)])
        self.out   = None

    def forward_array(self, data, param = None):
        
        if param is None:
            param = self.param
        data, single = self.reshape_data_array(data)
            
        data_tensor  = tf.placeholder(FDTYPE, shape= (None, ) + self.ndim_in)

        W = param['W']
        b = param['b']
        out = tf.matmul(data_tensor, W,  transpose_b = True)
        out += b
        out = tf.maximum(out*(self.grads[1]), out*(self.grads[0]))

        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        sess.run(init)
        out_value = sess.run(out, feed_dict={data_tensor: data})
        if single:
            out_value = out_value[0]
        return out_value

    def forward_tensor(self, data, param = None):
        
        if param is None:
            param = self.param
        data, single = self.reshape_data_tensor(data)
            
        W = param['W']
        b = param['b']
        out = tf.matmul(data, W,  transpose_b = True)
        out += b
        out = tf.maximum(out * self.grads[1], out * self.grads[0])

        if single:
            out = out[0]
        return out
   
    def get_grad_data(self, data):

        if data is None:
            data = tf.placeholder(FDTYPE, shape = (None) + self.ndim_in)
        param = self.param

        # create input placeholder that has batch_size
        data, single = self.reshape_data_tensor(data)
        W = param['W']
        out =   self.forward_tensor(data, param)
        out =   tf.maximum(out*(self.grads[1]), out*(self.grads[0]))
        grad = (self.grads[1] * tf.cast(out > 0, FDTYPE ) + \
                self.grads[0] * tf.cast(out <=0, FDTYPE )) [:,:,None] * \
                W[None,:,:]
        grad = tf.transpose(grad, [1,0,2])
        return grad, out, data

class ConvNetwork(Network):

    ''' one layer convolution network'''

    def __init__(self, ndim_in, nfil, size, stride=1, batch_size = 2):
        
        self.ndim_in = ndim_in
        self.batch_size = batch_size

        self.nfil    = nfil
        self.size    = size
        self.stride  = stride

        self.ndim_out = (  (ndim_in[1] - size) / stride + 1) **2 * nfil

        W      = tf.Variable(np.random.randn( * ((self.size,self.size)+ndim_in[0:1] + (nfil,))).astype(FDTYPE))
        b      = tf.Variable(np.random.randn(self.ndim_out).astype(FDTYPE))
        self.param = OrderedDict([('W', W), ('b', b)])
        self.out   = None

    def forward_array(self, data, param = None):
        
        if param is None:
            param = self.param
        data, single = self.reshape_data_array(data)
        ndata = data.shape[0]
        data_tensor  = tf.placeholder(FDTYPE, shape= (ndata, ) + self.ndim_in)

        W = param['W']
        b = param['b']

        conv = tf.nn.conv2d(data_tensor, W,
                            [1,1]+[self.stride]*2, 'VALID', data_format='NCHW')
        conv = tf.reshape(conv,[ndata,-1])
        conv = conv + b
        out = tf.nn.relu(conv)

        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        sess.run(init)
        out_value = sess.run(out, feed_dict={data_tensor: data})
        if single:
            out_value = out_value[0]
        return out_value

    def forward_tensor(self, data, param = None):
        
        if param is None:
            param = self.param
        data, single = self.reshape_data_tensor(data)
        ndata = data.shape[0].value
            
        W = param['W']
        b = param['b']

        conv = tf.nn.conv2d(data, W,
                            [1,1]+[self.stride]*2, 'VALID', data_format='NCHW')
        conv = tf.reshape(conv,[ndata,-1])
        conv = conv + b[None,:]
        out = tf.nn.relu(conv)

        if single:
            out = out[0]
        return out


class DropoutNetwork(Network):

    def __init__(self, ndim_in, p = 0.5, mode="test"):

        self.ndim_in = ndim_in
        self.ndim_out = ndim_in
        self.mode = mode

        self.p = p
        self.no_need_proc = (self.p == 1.0 or self.mode == "test")
        self.mask = None
        self.param=dict()

    def forward_array(self, data, mask = None):
        
        # only for testing purposes
        if mask is None:
            mask = (np.random.rand(*self.data.shape)<self.p)
        
        if self.no_need_proc:
            return data
        else:
            return data * mask / self.p

    def forward_tensor(self, data, mask=None):
        
        if self.no_need_proc:
            self.mask = tf.ones(tf.shape(data))
            return data 

        data, single = self.reshape_data_tensor(data)

        data_shape = tf.shape(data)
        batch_size = data_shape[0]
        m = len(self.ndim_in)

        if mask is None:
            mask = self.p
            # mask += tf.tile(tf.random_uniform((1,)+self.ndim_in, seed=2018), multiples=[batch_size] + [1]*m)
            mask += tf.random_uniform(data_shape, dtype=FDTYPE)
            mask = tf.floor(mask)

        out = data / self.p * mask
        self.mask = mask

        if single:
            out = out[0]

        return out

    def get_grad_data(self, data = None):
            
        if data is None:
            data = tf.placeholder(FDTYPE, shape = (None,) + self.ndim_in)
            
        data_shape = tf.shape(data)
        batch_size = data_shape[0]

        # number of dims
        m = len(self.ndim_in)
        # total number of elements per data point
        d = np.prod(self.ndim_in)

        if self.no_need_proc:

            grad = tf.eye(d, batch_shape=(batch_size,), dtype=FDTYPE)
            out  = tf.identity(data)
            self.mask = tf.ones(tf.shape(data), dtype=FDTYPE)
            
        else:

            mask = tf.constant(self.p, dtype=FDTYPE)
            #mask += tf.tile(tf.random_uniform((1,)+self.ndim_in, seed=2018), multiples=[batch_size]+[1]*m)
            mask += tf.random_uniform(data_shape, dtype=FDTYPE)
            mask = tf.floor(mask)
            out = data / self.p * mask
            
            grad = tf.matrix_diag(tf.reshape(mask,[batch_size, np.prod(self.ndim_in)]))
            grad = grad / self.p
            self.mask = mask
        
        grad = tf.reshape(grad, (batch_size,) + self.ndim_in + self.ndim_in)
        grad = tf.transpose(grad, perm = range(1,m+1) + [0,] + range(m+1,2*m+1) )

        return grad, out, data

    def get_hess_grad_data(self, data=None):

        grad, out, data = self.get_grad_data(data=data)
        data_shape = tf.shape(data)
        batch_size = data_shape[0]

        #hess = tf.zeros(np.ones(m+1+m*2), dtype=FDTYPE)
        hess = tf.zeros(self.ndim_in + (batch_size,) + self.ndim_in + self.ndim_in, dtype=FDTYPE)

        return hess, grad, out, data

###########################
###########################
### OTHER STUFF ########### 
###########################
###########################


class KernelNetMSD:

    ''' 
        A class that combines an ordinary kernel and network that finds the 
        maximum Stein discrepency (MSD) of data under a particular parametric (probablistic)
        model, potentially unnormalized, whose gradients on the data (score) can be computed 

        This gradient is required to calculate MSD
    '''

    def __init__(self, kernel, network):
        
        self.batch_size = network.batch_size
        self.ndim       = network.ndim_out
        self.ndim_in    = network.ndim_in
        self.network    = network
        assert self.ndim == self.network.ndim_out
        self.kernel  = kernel

    def MSD_V(self):

        dp_dx = tf.placeholder(FDTYPE, shape = (self.batch_size,) + self.ndim_in)
        dp_dy = tf.placeholder(FDTYPE, shape = (self.batch_size,) + self.ndim_in)

        dZX_dX, ZX, X = self.network.get_grad_data()
        dZY_dY, ZY, Y = self.network.get_grad_data()

        dk_dZX, dk_dZY, d2k_dZXdZY, gram  = self.kernel.get_two_grad_cross_hess(ZX, ZY)
        
        input_idx = construct_index(self.network.ndim_in)

        dk_dX = tf.einsum('ijk,ki'+input_idx+'->ij'+input_idx,
                        dk_dZX, dZX_dX)
        dk_dY = tf.einsum('ijk,kj'+input_idx+'->ij'+input_idx,
                        dk_dZY, dZY_dY)
        d2k_dXdY = tf.einsum('ijkl,ki' + input_idx + ',lj'+input_idx + '->ij'+input_idx, d2k_dZXdZY, dZX_dX, dZY_dY)
        print d2k_dZXdZY
        print dZX_dX
        print dZY_dY
        print d2k_dXdY 
        h = tf.einsum('i'+input_idx +  ',j'+input_idx + '->ij', dp_dx, dp_dy) * gram + \
            tf.einsum('j'+input_idx + ',ij'+input_idx + '->ij', dp_dy, dk_dX) + \
            tf.einsum('i'+input_idx + ',ij'+input_idx + '->ij', dp_dx, dk_dY) + \
            tf.reduce_sum(d2k_dXdY, range(2,len(self.ndim_in)+2))

        print h

        h = tf.reduce_sum(h)
        print h
        return h

class KernelNetModel():
    ''' A class that combines an ordinary kernel on the features
        extracted by a net

        This is used to build an infinite dimensional exp-fam model
        on the data, fitted using score matching

        There are two components: 
            kernel, responsible for computing the gram matrix
            network, responsible for network forward and derivatives

        This object then computes the derivatives of the function defined by the kernels
        and points that set up this function dk(x_i, y)/dy, which is then used to compute the score

    ''' 
    
    def __init__(self, kernel, network, alpha = None, points = None):

        ''' 
            ndim  : dimentionality of input kernel, an integer
            ndim_in: shape of input data to the network, a tuple
            X     : points that define the RKHS function
        '''
        warnings.warn("Deprecated, use KernelModel and define the a CompositeKernel", DeprecationWarning)
        self.alpha   = alpha
        self.ndim   = network.ndim_out
        self.ndim_in= network.ndim_in
        self.network = network
        self.kernel  = kernel
        if points is not None:
            self.set_points(points)

    def _net_forward(self, data):

        Y = self.network.forward_tensor(data)
        return Y


    def set_points(self, points):
        ''' This sets up the set of points used by model x_i's
            Input is a set of images that will first be processed by network
            These are stored inside the model as parameters
        '''
        
        assert points.shape[1:] == self.ndim_in, 'input shape of points not the same as predefiend'
        self.X = self._net_forward(points)
        self.K = self.kernel.get_gram_matrix(self.X, self.X)
        self.npoint = points.shape[0].value

    def evaluate_kernel_fun(self, Y):
        '''
        takes in input vector Y (output of the network) of shape (ninput x ndim) 
        and return the function defined by the lite model, linear combination of kernel
        functions 
        
        sum_m alpha_m * k(x_m, y_i)

        '''
        if Y.shape.ndims == 1:
            Y = tf.expand_dims(Y,0)
        
        K = self.kernel.get_gram_matrix(self.X, Y)

        return tf.einsum('i,ij', self.alpha, K)

    def evaluate_fun(self, data):

        y = self._net_forward(data)
        fv = self.evaluate_kernel_fun(y)
        return fv

    def evaluate_gram(self, points, data):
        
        x = self._net_forward(points)
        y = self._net_forward(data)
        return self.kernel.get_gram_matrix(x, y)


    def evaluate_grad(self, data):

        dydx, y, _ = self.network.get_grad_data(data)
        gradK = self.kernel.get_grad(self.X, y)

        input_idx = construct_index(self.network.ndim_in)

        gv = tf.einsum('i,ijk', self.alpha, gradK)

        gv = tf.einsum('jk,kj'+input_idx+'->j'+input_idx,
                        gv, dydx)

        return gv

    def get_fun_l2_norm(self):
        
        return tf.reduce_sum(tf.square(self.alpha))

    def get_fun_rkhs_norm(self, K=None):
        
        return tf.einsum('i,ij,j', self.alpha, self.K, self.alpha)
        
    def _score_statistics(self):


        d2ydx2, dydx, y, data = self.network.get_sec_grad_data()
        hessK, gradK = self.kernel.get_hess_grad(self.X, y)

        input_idx = construct_index(self.network.ndim_in)

        dkdx = tf.einsum('ijk,kj'+input_idx+'->ij'+input_idx,
                        gradK, dydx)
        '''
        d2kdx2 = tf.einsum('ijkl,klj'+input_idx + '->ij'+input_idx, 
                            hessK, dydx[None,...] * dydx[:,None,...]) + \
                 tf.einsum('ijk,kj'+input_idx + '->ij'+input_idx, 
                            gradK, d2ydx2)

        '''
        s2 = tf.einsum('ijkl,kj'+input_idx + '->ijl'+input_idx, 
                            hessK, dydx) 
        s2 = tf.einsum('ijl'+input_idx+',lj'+input_idx + '->ij'+input_idx, 
                            s2, dydx) 
        s1 = tf.einsum('ijk,kj'+input_idx + '->ij'+input_idx, 
                            gradK, d2ydx2)
        d2kdx2 = s1 + s2

        H = tf.einsum('ij'+input_idx+"->ij", 
                      d2kdx2)
        H = tf.reduce_mean(H,1)
        
        G = tf.einsum('ik'+input_idx+',jk'+input_idx+'->ijk',
                      dkdx, dkdx)
        G = tf.reduce_mean(G,2)
        
        C = tf.einsum('ik'+input_idx+',jk'+input_idx+'->ijk',
                      d2kdx2, d2kdx2)
        C = tf.reduce_mean(C,2)

        return H, G, C, data


