import tensorflow as tf
import numpy as np
from Utils import *

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

        input_idx = onstruct_index(self.network.ndim_in)

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
        d2Y, _, _ = self.network.get_hess_grad_data(Y)

        input_idx = construct_index(self.network.ndim_in, n=2)

        X = tf.reshape(X, [-1, self.out_size])
        d2Y  = tf.reshape(d2Y,  (self.out_size, -1) + self.network.ndim_in*2)
        
        hess = tf.einsum('ij,jk'+input_idx+'->ik' + input_idx, X, d2Y)

        return hess

    def get_hess_grad(self, X, Y):

        X = self.network.forward_tensor(X)
        d2Y, dY, _ = self.network.get_hess_grad_data(Y)

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
        d2ydx2, dydx, Y  = self.network.get_sec_grad_data(Y)
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
        dydx, Y = self.network.get_grad_data(Y)
        gradK = self.kernel.get_grad(X, Y)

        input_idx = construct_index(self.network.ndim_in)

        dkdx = tf.einsum('ijk,kj'+input_idx+'->ij'+input_idx,
                        gradK, dydx)

        return dkdx

    def get_grad_gram(self, X, Y):

        X = self._net_forward(X)
        d2ydx2, dydx = self.network.get_sec_grad_data(Y)
        hessK, gradK = self.kernel.get_hess_grad(X, Y)

        input_idx = construct_index(self.network.ndim_in)

        dkdx = tf.einsum('ijk,kj'+input_idx+'->ij'+input_idx,
                        gradK, dydx)
        
        gram = self.kernel.get_gram_matrix(X, Y)
        return dkdx, gram

    def get_hess(self, X, Y):

        X = self._net_forward(X)
        d2ydx2, dydx, Y = self.network.get_hess_grad_data(Y)
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

        d2ydx2, dydx, Y = self.network.get_hess_grad_data(Y)
        hessK, gradK = self.kernel.get_hess_grad(X, Y)

        input_idx = construct_index(self.network.ndim_in, n=2)
        input_idx_1 = input_idx[:len(self.network.ndim_in)]
        input_idx_2 = input_idx[len(self.network.ndim_in):]


        dkdx = tf.einsum('ijk,kj'+input_idx_1+'->ij'+input_idx_1,
                        gradK, dydx)

        d2kdx2 = tf.einsum('ijkl,kj'+input_idx_1+',lj'+input_idx_2+'->ij'+input_idx,
                        hessK, dydx, dydx) + \
                 tf.einsum('ijk,kj'+input_idx+"->ij"+input_idx, gradK, d2ydx2)

        return d2kdx2, dkdx

    def get_hess_grad_gram(self, X, Y):

        X = self._net_forward(X)

        d2ydx2, dydx, Y = self.network.get_hess_grad_data(Y)
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
        with tf.name_scope("PolynomialKernel"):
            self.c = tf.Variable(c, name="c", dtype=FDTYPE, trainable=False)

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
            K2 = tf.zeros((tf.shape(X)[0], tf.shape(Y)[0], tf.shape(Y)[1]), dtype=FDTYPE)
        else:
            K2 = self.d*(self.d-1)*(inner+self.c)**(self.d-2) * (X[:,None,:]**2)

        return K2, K1

    def get_hess_grad(self, X, Y):

        inner = self.get_inner(X,Y)[:,:,None]
        K1 = self.d * (inner+self.c)**(self.d-1) * X[:,None,:] 

        if self.d == 1:
            K2 = tf.zeros((tf.shape(X)[0], tf.shape(Y)[0], tf.shape(Y)[1], tf.shape(Y)[1]), dtype=FDTYPE)
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
            K2 = tf.zeros((tf.shape(X)[0], tf.shape(Y)[0], tf.shape(Y)[1], tf.shape(Y)[1]), dtype=FDTYPE)
        else:
            inner = inner[:,:,:,None]
            K2 = self.d*(self.d-1)*(inner+self.c)**(self.d-2) * (X[:,None,None,:] * X[:,None,:,None])

        return K2, K1, K


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
        d2Y, _, _ = self.network.get_hess_grad_data(Y)

        input_idx = construct_index(self.network.ndim_in, n=2)

        X = tf.reshape(X, [-1, self.out_size])
        d2Y  = tf.reshape(d2Y,  (self.out_size, -1) + self.network.ndim_in*2)
        
        hess = tf.einsum('ij,jk'+input_idx+'->ik' + input_idx, X, d2Y)

        return hess

    def get_hess_grad(self, X, Y):

        X = self.network.forward_tensor(X)
        d2Y, dY, _ = self.network.get_hess_grad_data(Y)

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

