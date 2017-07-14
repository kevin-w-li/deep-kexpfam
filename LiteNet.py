import tensorflow as tf
import numpy as np

class ExponentialKernel:

    def __init__(self, sigma = 1.0, X = None):
            
        if X is not None:
            self.X = X
            self.npoint = X.shape[0].value
            self.ndim = X.shape[1].value
        else:
            self.X  = None
            self.npoint = None
            self.ndim = None
        self.sigma = 2.0
        self.pdist2 = None

    def set_points(self, X):
        
        self.X = X
        self.npoint = X.shape[0].value
        self.ndim = X.shape[1].value

    def get_pdist2(self, Y, X=None):

        if X is None:
            X = self.X
        pdist2 = tf.reduce_sum(X**2, axis=1, keep_dims=True)
        pdist2 -= 2.0*tf.matmul(X, Y, transpose_b = True)
        pdist2 += tf.matrix_transpose(tf.reduce_sum(Y**2, axis=1, keep_dims=True))
        return pdist2

    def gram_matrix(self,Y, X = None, sigma = None):
        
        if X is None:
            X = self.X
        self.pdist2 = self.get_pdist2(Y)
        pdist2 = self.pdist2
        sigma = self.sigma if sigma == None else sigma

        return tf.exp(-1.0/sigma*pdist2)


class LiteModel:
    
    def __init__(self, kernel, network):
        
        self.npoint = kernel.npoint
        self.ndim   = kernel.ndim
        self.ndim_in= network.ndim_in
        self.ndata  =  network.ndata
        self.network = network
        self.kernel  = kernel
        self.X       = self.kernel.X

    def set_points(self, X):
        '''This sets up the set of points used by model x_i's
        Input is a set of images that will first be processed by network
        These are stored inside the model as parameters
        '''
        self.kernel.set_points(self.network.forward(X))
        self.ndim = self.kernel.ndim
        self.npoint = self.kernel.npoint
        self.X      = self.kernel.X

    def score_original(self, data):

        ''' This computs the score objective without the network-dependent 
            base measure q_0
        '''

        sigma = self.kernel.sigma
        N = self.ndata
        alpha = self.kernel.alpha
        ndim  = self.ndim

        # output of the network, same as x's
        Y = self.network.forward(data)

        # compute gram matrix of network output with points
        K = self.kernel.gram_matrix(Y)
        D = 2.0/sigma*self.kernel.pdist2 - ndim
        t = tf.reduce_sum(tf.expand_dims(alpha,1) * K * D) * 2.0 / N / sigma
        
        # X_il
        # Y_jl
        X = self.X
        # C = (X[:,None,:] - x[None,:,:]), C_ijl
        S = tf.expand_dims(X,1) - tf.expand_dims(Y,0)

        # alpha_exp = alpha[:,None, NOne], alpha_exp_i..
        alpha_exp = tf.expand_dims(tf.expand_dims(alpha, -1),-1)

        # K_exp = K[:,:,None], K_ij.
        K_exp     = tf.expand_dims(K, -1)

        J += 2.0/N/(sigma**2)*tf.reduce_sum(
                            tf.reduce_sum(alpha_exp * (S) * K_exp, 0
                        )**2)

        return J

    def _process_data(self, data):

        if data is not None:
                Y = self.network.forward(data)
        else:
            if self.X is None:
                raise NameError('run set_point first to set the kernel points, or provide it as input')
            Y = self.X
        return Y

    def score(self, data=None):
        
        Y = self._process_data(data)

        # alpha as vectors
        # alpha as vectors
        alpha = self.kernel.alpha
        alpha_V = tf.expand_dims(alpha,1)
        alpha_T = tf.expand_dims(alpha,0)

        sigma = self.kernel.sigma

        b, C = self._lite_score_stats(Y)

        J = 2.0 / self.ndata / sigma    * tf.einsum('i,i'   , alpha, b) + \
            2.0 / self.ndata / sigma**2 * tf.einsum('i,ij,j', alpha, C, alpha)

        return J

    def _lite_score_stats(self, Y):
        
        ''' compute the vector b and matrix C
            Y: the input data to the lite model to fit
        '''

        X = self.X
        sigma = self.kernel.sigma
        N = self.ndata
        D = self.ndim

        K= self.kernel.gram_matrix(Y)
        K_exp = tf.expand_dims(K, -1)

        S = tf.expand_dims(X,1) - tf.expand_dims(Y,0)

        b =  -D * tf.reduce_sum(K,1)
        b += 2/sigma * tf.reduce_sum( K * tf.reduce_sum(S**2, 2), 1)

        # term inside the square that multiplies alpha_i
        A   = S * K_exp
        SA  = tf.einsum('ikl,jkl->ij', A, A)
        C   = (SA)

        return b, C


    def lite_fit(self, data=None, lam = 0.01):

        npoint = self.npoint
        Y = self._process_data(data)
        b, C = self._lite_score_stats(Y)
       
        alpha_hat =  -self.kernel.sigma/2.0 * \
                        tf.matrix_solve( C, tf.expand_dims(b,-1) )
        self.kernel.alpha = tf.assign(self.kernel.alpha, tf.squeeze(alpha_hat))
      
        return self.kernel.alpha



        
        

class Network:

    def __init__(self, ndata, ndim_in, ndim):
        
        self.ndata = ndata
        self.ndim  = ndim
        self.ndim_in = ndim_in


    def forward(self, data):
        return data




        
        
