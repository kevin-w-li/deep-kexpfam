import tensorflow as tf
import numpy as np
from collections import OrderedDict
import operator
import itertools
from time import time
import warnings

config = tf.ConfigProto()
config.gpu_options.allow_growth=True


c = 30
'''
nl   = lambda x: tf.log(1+tf.exp(x))
dnl  = lambda x: 1/(1+tf.exp(-x))
d2nl = lambda x: tf.exp(-x)/tf.square(1+tf.exp(-x))

'''
nl   = lambda x: tf.where(x<c, tf.log(1+tf.exp(x)), x)
dnl  = lambda x: tf.where(x<-c, tf.zeros_like(x), 1/(1+tf.exp(-x)))
d2nl = lambda x: tf.where(tf.logical_and(-c<x, x<c), tf.exp(-x)/tf.square(1+tf.exp(-x)), tf.zeros_like(x))

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

    def __init__(self, ndim_in, ndim_out, init_weight_std = 1.0, init_mean = 0.0, scope="skip", nl_type=None):
        
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

        if nl_type == "linear":
            self.nl = lambda x: x
            self.dnl = lambda x: tf.zeros_like(x)
            self.d2nl= lambda x: tf.zeros_like(x)
        else:
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

    def __init__(self, layers, init_mean = 0.0, init_weight_std = 1.0, ndim_out = None, add_skip=False, nl_type=None):

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
                                init_mean=init_mean, init_weight_std=init_weight_std, scope="skip", nl_type=nl_type)

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


