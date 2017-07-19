import tensorflow as tf
import numpy as np
from collections import OrderedDict
import operator
import itertools

class GaussianKernel:

    def __init__(self, sigma = 1.0, X = None, alpha = None):

        '''
        X: points that define the kernel, X.shape = (npoint, ndim)
        '''
            
        if X is not None:
            self.X = X
            self.npoint = X.shape[0].value
            self.ndim = X.shape[1].value
        else:
            self.X  = None
            self.npoint = None
            self.ndim = None
        if alpha is not None: 
            assert alpha.shape[0] == X.shape[0]
            self.alpha  = alpha
        else:
            self.alpha  = None
        self.sigma = 2.0
        self.pdist2 = None
        self.grad   = None
        self.hess   = None

    def set_points(self, X, alpha = None):
        
        self.X = X
        self.npoint = X.shape[0].value
        self.ndim = X.shape[1].value
        if alpha is not None:
            assert alpha.shape[0] == X.shape[0]
            self.alpha = alpha

    def get_pdist2(self, Y):
        
        X = self.X
        pdist2 = tf.reduce_sum(X**2, axis=1, keep_dims=True)
        pdist2 -= 2.0*tf.matmul(X, Y, transpose_b = True)
        pdist2 += tf.matrix_transpose(tf.reduce_sum(Y**2, axis=1, keep_dims=True))
        self.pdist2 = pdist2
        return pdist2

    def set_pdist2(self, Y):

        self.get_pdist2(Y)
        
    def get_gram_matrix(self,Y=None):
        
        if Y is not None:
            self.pdist2 = self.get_pdist2(Y)
        else:
            if self.pdist2 is None:
                self.set_pdist2(Y)
        pdist2 = self.pdist2
        sigma = self.sigma
        gram = tf.exp(-0.5/sigma*pdist2)
        return gram

    def evaluate_fun(self, Y):
        '''
        takes in input vector Y of shape (ninput x ndim) and return the 
        function defined by the lite model, linear combination of kernel
        functions 
        
        sum_m alpha_m * k(x_m, y_i)
        
        '''
        if Y.shape.ndims == 1:
            Y = tf.expand_dims(Y,0)
        
        gram = self.get_gram_matrix(Y)

        return tf.einsum('i,ij', self.alpha, gram)
        
    def get_grad(self, Y):

        ''' first derivative of the function of lite model '''

        X = self.X
        gram = self.get_gram_matrix(Y)[:,:,None]

        # D contrains the vector difference between pairs of x_m and y_i
        # divided by sigma
        D = (tf.expand_dims(X, 1) - tf.expand_dims(Y, 0))/self.sigma

        # summing the first axes, 1 means the first 1 axes...
        grad = tf.tensordot(self.alpha, gram * D, 1)

        return grad
    
    def get_hess(self, Y):
        '''
        gram matrix, multiplied by ( outer products of x_m - y_i, minus
                                     identity matrix of shape ndim x ndim)
        '''
        X = self.X
        gram = self.get_gram_matrix(Y)[:,:,None, None]
        # the first term
        D = ( tf.expand_dims(X, 1) - tf.expand_dims(Y, 0) )/self.sigma
        D2 = tf.einsum('ijk,ijl->ijkl', D, D)
        I  = tf.eye( D.shape[-1].value )/self.sigma
        I_exp = tf.expand_dims(tf.expand_dims(I, 0), 0)
        hess  = tf.tensordot(self.alpha, gram * (D2 - I), 1)

        return hess
        
        

            

        
class KernelNetModel:
    
    def __init__(self, kernel, network):
        
        self.npoint = kernel.npoint
        self.ndim   = kernel.ndim
        self.ndim_in= network.ndim_in
        self.network = network
        self.kernel  = kernel
        self.X       = self.kernel.X

    def set_points(self, X):
        '''This sets up the set of points used by model x_i's
        Input is a set of images that will first be processed by network
        These are stored inside the model as parameters
        '''
        self.kernel.set_points(self.network.forward_tensor(X))
        self.ndim = self.kernel.ndim
        self.npoint = self.kernel.npoint
        self.X      = self.kernel.X

    def score_original(self, data):

        ''' This computs the score objective without the network-dependent 
            base measure q_0
        '''

        sigma = self.kernel.sigma*1
        N = data.shape[0].value
        alpha = self.kernel.alpha
        ndim  = self.ndim

        # output of the network, same as x's
        Y = self.network.forward_tensor(data)

        # compute gram matrix of network output with points
        K = self.kernel.get_gram_matrix(Y)
        D = 1.0/sigma*self.kernel.pdist2 - ndim
        J = tf.reduce_sum(tf.expand_dims(alpha,1) * K * D) * 1.0 / N / sigma
        
        # X_il
        # Y_jl
        X = self.X
        # C = (X[:,None,:] - x[None,:,:]), C_ijl
        S = tf.expand_dims(X,1) - tf.expand_dims(Y,0)

        # alpha_exp = alpha[:,None, NOne], alpha_exp_i..
        alpha_exp = tf.expand_dims(tf.expand_dims(alpha, -1),-1)

        # K_exp = K[:,:,None], K_ij.
        K_exp     = tf.expand_dims(K, -1)

        J += 0.5/N/(sigma**2)*tf.reduce_sum(
                            tf.reduce_sum(alpha_exp * (S) * K_exp, 0
                        )**2)

        return J

    def _process_data(self, data):

        if data is not None:
                Y = self.network.forward_tensor(data)
        else:
            if self.X is None:
                raise NameError('run set_point first to set the kernel points, or provide it as input')
            Y = self.X
        return Y

    def kernel_score(self, data=None):
        
        Y = self._process_data(data)
        ndata = data.shape.dims[0].value

        # alpha as vectors
        # alpha as vectors
        alpha = self.kernel.alpha
        alpha_V = tf.expand_dims(alpha,1)
        alpha_T = tf.expand_dims(alpha,0)

        sigma = self.kernel.sigma

        b, C = self._kernel_score_stats(Y)

        J = 1.0 / ndata / sigma    * tf.einsum('i,i'   , alpha, b) + \
            0.5 / ndata / sigma**2 * tf.einsum('i,ij,j', alpha, C, alpha)

        return J

    def _kernel_score_stats(self, Y):
        
        ''' compute the vector b and matrix C
            Y: the input data to the lite model to fit
        '''

        X = self.X
        sigma = self.kernel.sigma
        N = Y.shape[0]
        D = self.ndim

        K= self.kernel.get_gram_matrix(Y)
        K_exp = tf.expand_dims(K, -1)

        S = tf.expand_dims(X,1) - tf.expand_dims(Y,0)

        b =  -D * tf.reduce_sum(K,1)
        b += 1/sigma * tf.reduce_sum( K * tf.reduce_sum(S**2, 2), 1)

        # term inside the square that multiplies alpha_i
        A   = S * K_exp
        SA  = tf.einsum('ikl,jkl->ij', A, A)
        C   = (SA)

        return b, C


    def kernel_fit(self, data=None, lam = 0.01):

        npoint = self.npoint
        Y = self._process_data(data)
        b, C = self._kernel_score_stats(Y)
       
        alpha_hat =  -self.kernel.sigma * \
                        tf.matrix_solve( C, tf.expand_dims(b,-1) )
        self.kernel.alpha = tf.assign(self.kernel.alpha, tf.squeeze(alpha_hat))
      
        return self.kernel.alpha


    def kernel_grad(self, data):
        Y = self.net.forward_tensor(data)
        return self.kernel.grad(Y)
        

class Network:
    
    ndim_in = None
    ndim_out= None
    batch_size   = None
    param   = None
    out     = None

    def __init__(self, ndim_in, ndim):

        raise NotImplementedError('should implement individual networks')

    def reshape_data_array(self, x):
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
        raise NotImplementedError('should implement individual networks')

    def forward_tensor(self, data):
        raise NotImplementedError('should implement individual networks')

    def get_grad_param(self, data):

        ''' Compute the derivative of each output to all the parameters
        data: numpy array

        '''

        # reshape data to see if there is a single input
        data, single   = self.reshape_data_array(data)
        ninput = data.shape[0]
        # number of batches
        nbatch = ninput/self.batch_size

        # parameters are duplicated (not copied in memory) to facilitate parallel gradient over input batch
        # it creates different symbols that points to the same parameters via the tf.identity function
        # each copy of the parameter will be paired with a single input data for forward computation
        param_copy = [OrderedDict(
                        zip( self.param.keys(), 
                             [tf.identity(v) for v in self.param.values()]
                           )
                    ) for _ in xrange(self.batch_size)]
       
        # rearrange the parameter handles into a single array so tf.gradients can consume. It is arrange like:
        #         [ w_1_copy1, w_1_copy_2, w_1_copy_3, ..., w_nparam_copy_batch_size ]
        batch_param = [ param_copy[i][k] for k in self.param.keys() for i in xrange(self.batch_size) ]

        # create input placeholder that has batch_size
        one_batch = tf.placeholder(tf.float32, shape = (self.batch_size,) + self.ndim_in)

        # split it into different symbols to be paired with parameter copies
        one_batch_split = tf.split(one_batch, self.batch_size, axis = 0)
    
        # compute output of each data which is paired with a copy of parameter dictionary
        output = [ self.forward_tensor(one_data[0],param=p) 
                            for (one_data, p) in zip(one_batch_split,  param_copy) ]

        # Create dummy variable so that gradient can be computed. This has the same length
        # as the number of outputs.
        # It works because the following
        # Let the output be y_i(x_j), each data x_j is paired with parameter copy w_j,
        # The i'th value of this variable, denoted by sum_i, is then sum_j y_i(x_j)
        # Then d(sum_i) / d(w_j) = d(y_i(x_j))
        # sum_output = [ sum_j y_0(x_j), sum_j y_1(x_j), sum_j y_2(x_j), .. , sum_j y_ndim_out(x_j) ]
        sum_output = tf.reduce_sum(output, axis=0)

        # Essentailly create ndim_out number of graphs that share parameters, each element is the dummpy
        # variable that sums each output over input data batch
        # grad will be arranged by [ number of output [len(batch_param)] ]
        grad   = [ list(tf.gradients(sum_output[oi], batch_param)) for oi in xrange(self.ndim_out)]

        # After evaluating the gradient, arrange the gradients into a dictionary so that copies of parameters
        # are indexed by the same parameter name. This is done by looking for the gradient w.r.t. the correct 
        # chunk in each tf.gradient output where parameters of the same name reside as in batch_param
        grad_dict = OrderedDict.fromkeys(self.param.keys())
        for ki, k in enumerate(self.param.keys()):
            # look for the correct chunk in each output dimension
            grad_k = [grad[oi][ki*self.batch_size:(ki+1)*self.batch_size] for oi in xrange(self.ndim_out)]
            # flatten(grad_k)
            grad_dict[k] = list(itertools.chain.from_iterable(grad_k))

        # results stores grad over multiple batches
        results = []
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        for bi in xrange(nbatch):
            # get a batch
            batch_data = data[bi*self.batch_size:(bi+1)*self.batch_size]
            # run and append results, the first argument is [[w_1_copies], [w_2_copies], network output]
            results.append(sess.run(grad_dict.values() + [output], feed_dict = {one_batch : batch_data}))
        
        # create output dictionary what will be indexed by parameter name.
        # grad_value_dict[parameter_name][output_idx][input_idx][param_dims]
        grad_value_dict = OrderedDict.fromkeys(self.param.keys())
       
        # results are arranged as follows
        # results[batch_idx][param][w_copyies that corresponds to each input data within batch]
        # first fix a k, then form an array of [ndim_out x batch_size*nbatch]
        for ki, k in enumerate(self.param.keys()):
            grad_k = np.array([      
                        np.concatenate([
                            results[bi][ki][oi*self.batch_size:(oi+1)*self.batch_size] for bi in xrange(nbatch)
                        ]) for oi in xrange(self.ndim_out)
                     ])
            grad_value_dict[k] = grad_k
        
        # extract the output by using the last index of each batch
        output_value = np.concatenate([results[bi][-1] for bi in xrange(nbatch)])
        return grad_value_dict, output_value

    def get_grad_param_one_data(self, data):

        if data.shape.ndims == 1 and data.shape[0] == self.ndim_in:
            data = tf.expand_dims(data, 0)
        return get_grad_param(data)


    def get_sec_grad_data(self, data):

        raise NotImplementedError('should implement individual networks')

class LinearNetwork(Network):

    def __init__(self, ndim_in, ndim_out, batch_size = 2):
        
        self.ndim_out  = ndim_out
        self.ndim_in = ndim_in
        self.batch_size = batch_size
        W     = tf.Variable(np.random.randn(ndim_out, *ndim_in).astype('float32'))
        b      = tf.Variable(np.random.randn(1, ndim_out).astype('float32'))
        self.param = OrderedDict([('W', W), ('b', b)])
        self.out   = None

    def forward_array(self, data, param = None):
        
        if param is None:
            param = self.param
        data, single = self.reshape_data_array(data)
            
        data_tensor  = tf.placeholder('float32', shape= (None, ) + self.ndim_in)

        W = param['W']
        b = param['b']
        out = tf.matmul(data_tensor, W,  transpose_b = True)
        out += b
        sess = tf.Session()
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
    def get_grad_param_one_out(self, data):

        
        out = self.out

        grad = lambda oi: zip(*[tf.gradients(out[di, oi], [W, b]) for di in range(ninput)])

        grad_dict = lambda i: OrderedDict( zip(self.param.keys(), grad(i)) )

        return grad_dict

    def get_sec_grad(self, data):

        if self.out is None:
            self.forward_tensor(data)
        
        out = self.out
        ninput = out.shape.as_list()[0]
        grad = tf.stack([tf.gradients(out[i], [W])[0] for i in np.arange(ninput)])
        return grad


