import tensorflow as tf

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
        self.alpha   = None

    def set_points(self, X):
        self.kernel.set_points(self.network.forward(X))
        self.ndim = self.kernel.ndim
        self.npoint = self.kernel.npoint

    def score(self, data):
        sigma = self.kernel.sigma
        Y = self.network.forward(data)
        G = self.kernel.gram_matrix(Y)
        D = 2.0/sigma*self.kernel.pdist2 - self.ndim
        # J = tf.reduce_sum(D) * 2 / self.ndata / sigma
        J = tf.reduce_sum(tf.expand_dims(self.alpha,1) * G * D) * 2.0 / self.ndata / sigma

        return J
            
            
class Network:

    def __init__(self, ndata, ndim_in, ndim):
        
        self.ndata = ndata
        self.ndim  = ndim
        self.ndim_in = ndim_in


    def forward(self, data):
        return data




        
        
