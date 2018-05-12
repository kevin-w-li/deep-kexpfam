import tensorflow as tf
import numpy as np
from collections import OrderedDict
import operator
import itertools
import time
import warnings

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

def construct_index(dim,s="o", n=1):
    ''' construct string for use in tf.einsum as it does not support...'''
    s = ord(s)
    return ''.join([str(unichr(i+s)) for i in range(len(dim)*n)])

FDTYPE="float32"

# =====================            
# Kernel related
# =====================            

class LiteModel:


    def __init__(self, kernel, alpha = None, points = None):
        
        self.kernel = kernel
        self.alpha   = alpha
        if points is not None:
            self.set_points(points)

    def _score_statistics(self, data=None):
        
        ''' compute the vector b and matrix C
            Y: the input data to the lite model to fit
        '''
        if data is None: 
            data = tf.placeholder(FDTYPE, shape = (None,) + self.ndim_in)

        d2kdx2, dkdx = self.kernel.get_sec_grad(self.X, data)

        H = tf.einsum("ijk->ij", 
                      d2kdx2)
        H = tf.reduce_mean(H,1)
        
        G = tf.einsum('ikl,jkl->ijk',
                      dkdx, dkdx)
        G = tf.reduce_mean(G,2)
        
        C = tf.einsum('ikl,jkl->ijk',
                      d2kdx2, d2kdx2)
        C = tf.reduce_mean(C,2)

        return H, G, C, data
    

    def score(self, data=None, alpha=None):

        H, G, C, data = self._score_statistics(data=data)

        if alpha is None:
            alpha = self.alpha

        s2 = tf.einsum('i,i->', alpha, H)
        s1 = 0.5 * tf.einsum('i,ij,j', alpha, G, alpha)
        score  =  s1+s2

        return score, H, G, C, data

    def opt_score(self, data=None, lam_norm=0.0, lam_alpha=0.0, lam_curve=0.0, lam_weights=0.0):
        '''
        compute regularised score and returns a handle for assign optimal alpha
        '''

        score, H, G, C, data = self.score(data=data)

        r_norm =  self.get_fun_rkhs_norm()
        l_norm =  self.get_fun_l2_norm()

        alpha_sg = self.alpha

        curve  =  tf.einsum('i,ij,j', alpha_sg, C, alpha_sg)
        
        w_norm = self.get_weights_norm()
        loss   =  score + 0.5 * (   lam_norm  * r_norm + 
                                    lam_alpha * l_norm + 
                                    curve * lam_curve + 
                                    lam_weights * w_norm)

        alpha = tf.matrix_solve(G + 
                                self.K*lam_norm + 
                                tf.eye(self.npoint, dtype=FDTYPE)*lam_alpha + 
                                C * lam_curve, 
                                -H[:,None])[:,0]
        alpha_assign_opt = tf.assign(self.alpha, alpha)

        return alpha_assign_opt, loss, score, data, r_norm, l_norm
        
    def val_score(self, train_data=None, val_data=None, 
                    lam_norm=0.0, lam_alpha=0.0, lam_curve=0.0, lam_weights=0.0):

        H, G, C, train_data = self._score_statistics(data=train_data)

        #self.alpha = tf.ones(npoint)
        self.alpha = tf.matrix_solve(G + self.K*lam_norm + 
                                    tf.eye(self.npoint, dtype=FDTYPE)*lam_alpha + 
                                    C * lam_curve, -H[:,None])[:,0]
        
        score, H, G, C, val_data = self.score(data=val_data,  alpha=self.alpha)

        r_norm =  self.get_fun_rkhs_norm()
        l_norm =  self.get_fun_l2_norm()
        curve = tf.einsum('i,ij,j', self.alpha, C, self.alpha)
        w_norm =  self.get_weights_norm()
        loss   =  score + 0.5 * w_norm * lam_weights

        return loss, score, train_data, val_data, r_norm, l_norm, curve, w_norm

    def step_score(self, lam_norm=0.0, lam_alpha=0.0, lam_curve=0.0, lam_weights=0.0):
        
        assert self.alpha is not None
        self.alpha = tf.stop_gradient(self.alpha)

        score, H, G, C, train_data = self.score()
        score_sg, H, G, C, val_data = self.score(alpha=self.alpha)

        r_norm =  self.get_fun_rkhs_norm()
        l_norm =  self.get_fun_l2_norm()
        curve  =  tf.einsum('i,ij,j', self.alpha, C, self.alpha)

        loss   =  score + 0.5 * (   lam_norm  * r_norm + 
                                    lam_alpha * l_norm + 
                                    curve * lam_curve + 
                                    weights * lam_weights)
        loss_sg =  score_sg

        return loss, loss_sg, score, train_data, val_data, r_norm, l_norm, curve


    def set_points(self, points):
        
        self.X = points
        self.npoint = points.shape[0].value
        self.ndim_in = tuple( points.shape[1:].as_list() )
        self.K = self.kernel.get_gram_matrix(self.X, self.X)

    def evaluate_gram(self, X, Y):
        return self.kernel.get_gram_matrix(X, Y)

    def evaluate_fun(self, data):

        gram = self.kernel.get_gram_matrix(self.X, data)

        return tf.tensordot(self.alpha, gram, [[0],[0]])

    def evaluate_grad(self, data):

        grad = self.kernel.get_grad(self.X, data)
        return tf.tensordot(self.alpha, grad, axes=[[0],[0]])

    def evaluate_hess(self, data):
        
        hess = self.kernel.get_hess(self.X, data)
        return tf.tensordot(self.alpha, hess, axes=[[0],[0]])

    def evaluate_hess_grad_fun(self, data):
        
        hess, grad, gram = self.kernel.get_hess_grad_gram(self.X, data)
        hess = tf.tensordot(self.alpha, hess, axes=[[0],[0]])
        grad = tf.tensordot(self.alpha, grad, axes=[[0],[0]])
        fun  = tf.tensordot(self.alpha, gram, axes=[[0],[0]])

        return hess, grad, fun
        

    def get_fun_l2_norm(self):
        
        return tf.reduce_sum(self.alpha**2)

    def get_fun_rkhs_norm(self, K=None):
        
        return tf.einsum('i,ij,j', self.alpha, self.K, self.alpha)

    def get_weights_norm(self):
        return self.kernel.get_weights_norm() 

class Kernel:

    def __init__(self,):
        raise(NotImplementedError)

    def get_weights_norm(self):
        # no weights by default
        return 0.0

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
        out = 0
        for ki in range(self.nkernel):
            out = out + self.kernels[ki].get_gram_matrix(X, Y) * self.props[ki]

        return out

    def get_sec_grad(self, X, Y):
        
        grad = 0
        sec  = 0
        for ki in range(self.nkernel):
            s, g  = self.kernels[ki].get_sec_grad(X, Y)
            grad  = grad + g * self.props[ki]
            sec   = sec  + s * self.props[ki]

        return grad, sec


    def get_grad(self, X, Y):

        out = 0
        for ki in range(self.nkernel):
            out = out + self.kernels[ki].get_grad(X, Y) * self.props[ki]
        return out

    def get_hess(self, X, Y):

        out = 0
        for ki in range(self.nkernel):
            out = out + self.kernels[ki].get_hess(X, Y) * self.props[ki]
        return out

    def get_weights_norm(self):
        out = 0
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

    def __init__(self, sigma = 1.0):

        self.sigma = sigma
        self.pdist2 = None

    def get_pdist2(self, X, Y):
        
        if X.shape.ndims==1:
            X = X[None,:]
        if Y.shape.ndims==1:
            Y = Y[None,:]
        assert X.shape[1] == Y.shape[1]
        pdist2 = tf.reduce_sum(X**2, axis=1, keep_dims=True)
        pdist2 -= 2.0*tf.matmul(X, Y, transpose_b = True)
        pdist2 += tf.matrix_transpose(tf.reduce_sum(Y**2, axis=1, keep_dims=True))
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
        
        D2 = D**2
        I  = tf.ones( D.shape[-1].value )/self.sigma

        # K is a vector that has the hessian on all points
        K2 = gram[:,:,None] * (D2 - I)

        return K2, K1

    def get_sec_grad_gram(self, X, Y):

        gram = self.get_gram_matrix(X, Y)

        # D contrains the vector difference between pairs of x_m and y_i
        # divided by sigma
        D = (tf.expand_dims(X, 1) - tf.expand_dims(Y, 0))/self.sigma

        # K is a vector that has derivatives on all points
        K1 = (gram[:,:,None]* D)
        
        D2 = D**2
        I  = tf.ones( D.shape[-1].value )/self.sigma

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

    def __init__(self, ndim, d, c=1):
        
        self.d = d
        self.c = c

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

    def get_hess_grad(self, X, Y):

        inner = self.get_inner(X,Y)[:,:,None]
        K1 = self.d * (inner+self.c)**(self.d-1) * X[:,None,:] 

        if self.d == 1:
            K2 = tf.zeros((X.shape[0], Y.shape[0], Y.shape[1], Y.shape[1]))
        else:
            inner = inner[:,:,:,None]
            K2 = self.d*(self.d-1)*(inner+self.c)**(self.d-2) * (X[:,None,None,:] * X[:,None,:,None])

        return K2, K1



class PolynomialKernel:

    def __init__(self, d, c=1):
        
        self.d = d
        self.c = c

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

class Network:
    '''
    Network should be implemented as subclass of Network

    Define a forward_tensor method which takes in data as tf.Tensor (placeholder) of shape
        [ndata, ...], e.g. [ndata, C, H, W] for conv net

    and return the output as a vector
        [ndata, ndim_out]
    This class contains methods that computes gradients w.r.t. data and parameter (may not be required)
    and second derivative w.r.t. data
    
    '''
   

    # input shape should be a tuple
    ndim_in = None
    # output shape is a scalar. Output assumed to be 1-D
    ndim_out= None
    # how much data to process for gradients, only for gradients and second gradients now.
    batch_size   = None
    # network parameter as a dictionary
    param   = None
    # empty now...
    out     = None

    def __init__(self, ndim_in, ndim):

        raise NotImplementedError('should implement individual networks')

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

        return sum(map(lambda p: tf.reduce_sum(p**2), self.param.values()))

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
        
        t0 = time.time()
        son = SumOutputNet(self)
        grad = []
        t0 = time.time()
        t1 = time.time()
        print 'building grad output'
        for oi in xrange(self.ndim_out):
            g = tf.gradients(son.sum_output[oi], son.batch)[0]
            grad.append(g)
            print '\r%3d out of %3d took %5.3f sec' % ( oi, self.ndim_out, time.time()-t1),
            ti = time.time()
        grad   = tf.stack(grad)
        print 'building grad output took %.3f sec' % (time.time() - t0)

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
            t0 = time.time()
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
            print 'took %.3f sec' % (time.time()-t0)
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
            size += tf.reduce_sum(v**2.0) 
        return size 

class LinearSoftNetwork(Network):

    ''' y =  ReLU( W \cdot x + b ) '''

    def __init__(self, ndim_in, ndim_out, 
                init_std = 1.0, init_mean = 0.0, 
                #nl   = lambda x: tf.where(x<30, tf.nn.softplus(x), x),
                #dnl  = lambda x: 1/(1+tf.exp(-x)),
                #d2nl = lambda x: tf.where(tf.logical_and(-30<x, x<30), tf.exp(x-2*tf.log(1+tf.exp(x))), tf.zeros_like(x))):
                nl   = tf.nn.softplus,
                dnl  = lambda x: 1/(1+tf.exp(-x)),
                d2nl = lambda x: tf.exp(x-2*tf.log(1+tf.exp(x)))):
        
        self.ndim_out  = ndim_out
        self.ndim_in = ndim_in
        W = tf.Variable(init_mean + np.random.randn(*(ndim_out + ndim_in)).astype(FDTYPE))*init_std
        b = tf.Variable(np.random.randn(1, *ndim_out).astype(FDTYPE))*init_std

        self.nl = nl
        self.dnl = dnl
        self.d2nl = d2nl

        self.param = OrderedDict([('W', W), ('b', b)])

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

        sec = self.d2nl(lin_out)[:,:,None] * W[None,:,:]**2
        sec = tf.transpose(sec, [1,0,2])
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

        return hess, grad, out, data

class DeepNetwork(Network):

    def __init__(self, layers):

        self.layers   = layers
        self.nlayer   = len(layers)
        self.param    = {}
        for i in range(self.nlayer-1):
            assert layers[i+1].ndim_in == layers[i].ndim_out

        for i, l in enumerate(layers):
            layer_str = str(i+1)
            p = l.param
            for k, v in p.items():
                self.param[k+layer_str] = v

        self.ndim_in  = self.layers[0].ndim_in
        self.ndim_out = self.layers[-1].ndim_out
        
    def forward_tensor(self, data):
        
        d = data

        for i in range(self.nlayer):
            d = self.layers[i].forward_tensor(d) 
        return d

    def forward_array(self, data):

        d = data
        for i in range(self.nlayer):
            d = self.layers[i].forward_array(d) 
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

        return grad, out, data
            
    def get_sec_grad_data(self, data = None):
        
        if data is None:
            data = tf.placeholder(FDTYPE, shape = (None,) + self.ndim_in, name="input")

        sec, grad, out, _ = self.layers[0].get_sec_grad_data(data)


        for i in range(1,self.nlayer):

            i_idx_l = construct_index(self.layers[i-1].ndim_in, s="o",n=1)

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

        return sec, grad, out, data

    def get_hess_grad_data(self, data = None):

        if data is None:
            data = tf.placeholder(FDTYPE, shape = (None,) + self.ndim_in, name="input")

        hess, grad, out, _ = self.layers[0].get_hess_grad_data(data)

        for i in range(1,self.nlayer):

            i_idx_l = construct_index(self.layers[i-1].ndim_in, s="o",n=2)
            i_idx_l_1 = i_idx_l[:len(i_idx_l)/2]
            i_idx_l_2 = i_idx_l[len(i_idx_l)/2:]

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


        return hess, grad, out, data
        

class LinearNetwork(Network):

    ''' y =  W \cdot x + b '''

    def __init__(self, ndim_in, ndim_out, batch_size = 2, init_std = 1.0, init_mean = 0.0, identity=False):
        
        self.ndim_out  = ndim_out
        self.ndim_in = ndim_in
        self.batch_size = batch_size
        if identity:
            W   = tf.constant(np.eye(ndim_in[0]).astype(FDTYPE))
            b   = tf.constant(np.zeros((1,ndim_in[0])).astype(FDTYPE))
        else:
            W   = tf.Variable(init_mean + np.random.randn(ndim_out, *ndim_in).astype(FDTYPE))*init_std
            b   = tf.Variable(np.random.randn(1, ndim_out).astype(FDTYPE))*init_std
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
            data = tf.placeholder(FDTYPE, shape = (None) + self.ndim_in)
        data, single = self.reshape_data_tensor(data)
        W = param['W']
        output = self.forward_tensor(data, param)
        grad = tf.tile(W[None,:,:], [self.batch_size, 1, 1])
        grad = tf.transpose(grad, [1,0,2])
        return grad, output, data

    def get_sec_grad_data(self, data=None):

        if data is None:
            data = tf.placeholder(FDTYPE, shape = (None) + self.ndim_in)
        param = self.param

        # create input placeholder that has batch_size
        #data = tf.placeholder(FDTYPE, shape = (self.batch_size,) + self.ndim_in)
        data, single = self.reshape_data_tensor(data)
        W = param['W']
        out  = self.forward_tensor(data, param)
        grad = tf.tile(W[None,:,:], [self.batch_size, 1, 1])
        grad = tf.transpose(grad, [1,0,2])

        sec = tf.zeros_like(grad)
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
        out = out ** 2.0
        
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
        out = out ** 2.0

        if single:
            out = out[0]
        return out


class LinearReLUNetwork(Network):

    ''' y =  ReLU( W \cdot x + b ) '''

    def __init__(self, ndim_in, ndim_out, batch_size = 2, 
                init_std = 1.0, init_mean = 0.0, 
                grads = [0.0, 1.0]):
        
        self.ndim_out  = ndim_out
        self.ndim_in = ndim_in
        self.batch_size = batch_size
        W   = tf.Variable(init_mean + np.random.randn(*ndim_out + ndim_in).astype(FDTYPE))*init_std
        b   = tf.Variable(np.random.randn(1, *ndim_out).astype(FDTYPE))*init_std
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
        data = tf.placeholder(FDTYPE, shape = (self.batch_size,) + self.ndim_in)
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
        
        return tf.reduce_sum(self.alpha**2)

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


