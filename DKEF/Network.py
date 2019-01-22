import tensorflow as tf
import numpy as np
from collections import OrderedDict
from time import time
from settings import FDTYPE
from Utils import construct_index


c = 30

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

    def get_grad_data(self, data):

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

    def get_grad_data(self, data):

        param = self.param
        
        if data is None:
            data = tf.placeholder(FDTYPE, shape = (None,) + self.ndim_in)
        data, single = self.reshape_data_tensor(data)

        W = param['W']
        b = param['b']
        lin_out = tf.matmul(data, W,  transpose_b = True)
        lin_out += b
        out = self.nl(lin_out)

        grad = self.dnl(lin_out)[:,:,None] * W[None,:,:]
        grad = tf.transpose(grad, [1,0,2])

        if self.keep_prob is not None:
            out, ds = add_dropout(self, out, grad)
            grad = ds[0]

        return grad, out
  
    def get_sec_grad_data(self, data):

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

        return sec, grad, out

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

        return hess, grad, out

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
            self.dnl = lambda x: tf.ones_like(x)
            self.d2nl= lambda x: tf.zeros_like(x)
        else:
            self.nl = nl
            self.dnl = dnl
            self.d2nl = d2nl
        
    def forward_array(self, data):
        
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

    def forward_tensor(self, data):
        
        param = self.param
            
        out = 0.0
        for i in range(self.nin):
            W = param['W'+str(i)]
            out += tf.matmul(data[i], W,  transpose_b = True)
        b = param['b']
        out += b
        out = self.nl(out)

        return out

    def get_grad_data(self, data):

        param = self.param
        
        lin_out = 0.0
        for i in range(self.nin):
            W = param['W'+str(i)]
            lin_out += tf.matmul(data[i], W,  transpose_b = True)
        b = param['b']
        lin_out += b

        out  = self.nl(lin_out)
        grad = [self.dnl(lin_out)[:,:,None] * param['W'+str(i)][None,:,:] for i in range(self.nin)]
        grad = [tf.transpose(grad[i], [1,0,2]) for i in range(self.nin)]

        return grad, out
  
    def get_sec_grad_data(self, data):

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

        sec  = [self.d2nl(lin_out)[:,:,None] * tf.square(param["W"+str(i)][None,:,:]) for i in range(self.nin)]
        sec  = [tf.transpose(sec[i], [1,0,2]) for i in range(self.nin)]
        return sec, grad, out

    def get_hess_cross_grad_data(self, data):

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

        return hess, cross, grad, out

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


    def get_grad_data(self, data):
        
        if data is None:
            data = tf.placeholder(FDTYPE, shape = (None,) + self.ndim_in)

        grad, out = self.layers[0].get_grad_data(data)

        i_idx_l = construct_index(self.layers[0].ndim_in, s="o")

        for i in range(1,self.nlayer):

            layer = self.layers[i]

            i_idx_h = construct_index(layer.ndim_in, s="j")
            o_idx_h = construct_index(layer.ndim_out, s="a")

            this_grad, out = layer.get_grad_data(out)

            grad = tf.einsum(o_idx_h+"i"+i_idx_h+","\
                            +i_idx_h+"i"+i_idx_l+"->"\
                            +o_idx_h+"i"+i_idx_l,  this_grad, grad)

            i_idx_l = construct_index(layer.ndim_in, s="o")
        
        if self.add_skip:
            
            layer = self.skip_layer
            skip_grad, skip_out = layer.get_grad_data([data,out])

            i_idx_h = construct_index(layer.ndim_in[1], s="j")
            o_idx_h = construct_index(layer.ndim_out, s="a")

            grad = tf.einsum(o_idx_h+"i"+i_idx_h+","\
                            +i_idx_h+"i"+i_idx_l+"->"\
                            +o_idx_h+"i"+i_idx_l,  skip_grad[1], grad)

            grad = grad + skip_grad[0]
            out  = skip_out

        return grad, out
            
    def get_sec_grad_data(self, data):
        
        if data is None:
            data = tf.placeholder(FDTYPE, shape = (None,) + self.ndim_in, name="input")

        sec, grad, out = self.layers[0].get_sec_grad_data(data)

        i_idx_l = construct_index(self.layers[0].ndim_in, s="o")

        for i in range(1,self.nlayer):


            layer = self.layers[i]

            i_idx_h = construct_index(layer.ndim_in, s="j", n=2)
            i_idx_h_1 = i_idx_h[:len(i_idx_h)/2]
            i_idx_h_2 = i_idx_h[len(i_idx_h)/2:]

            o_idx_h = construct_index(layer.ndim_out, s="a")

            this_hess, this_grad, out = self.layers[i].get_hess_grad_data(out)

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

            skip_sec, skip_grad, skip_out = layer.get_sec_grad_data([data,out])

            i_idx_h   = construct_index(layer.ndim_in[1], s="j", n=2)
            i_idx_h_1 = i_idx_h[:len(i_idx_h)/2]
            i_idx_h_2 = i_idx_h[len(i_idx_h)/2:]
            i_idx_h_c_1   = construct_index(layer.ndim_in[0], s="j", n=1)
            i_idx_h_c_2   = construct_index(layer.ndim_in[1], s="p", n=1)
            o_idx_h   = construct_index(layer.ndim_out, s="a")

            this_hess, this_cross, _, _ = layer.get_hess_cross_grad_data([data,out])

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

        return sec, grad, out

    def get_hess_grad_data(self, data):

        if data is None:
            data = tf.placeholder(FDTYPE, shape = (None,) + self.ndim_in, name="input")

        hess, grad, out = self.layers[0].get_hess_grad_data(data)

        i_idx_l = construct_index(self.layers[0].ndim_in, s="o",n=2)
        i_idx_l_1 = i_idx_l[:len(i_idx_l)/2]
        i_idx_l_2 = i_idx_l[len(i_idx_l)/2:]

        for i in range(1,self.nlayer):


            layer = self.layers[i]

            i_idx_h = construct_index(layer.ndim_in, s="j", n=2)
            i_idx_h_1 = i_idx_h[:len(i_idx_h)/2]
            i_idx_h_2 = i_idx_h[len(i_idx_h)/2:]

            o_idx_h = construct_index(layer.ndim_out, s="a")

            this_hess, this_grad, out = self.layers[i].get_hess_grad_data(out)

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

            skip_hess, skip_cross, skip_grad, skip_out = layer.get_hess_cross_grad_data([data,out])

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

        return hess, grad, out
        

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

    def get_grad_data(self, data):

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
        return grad, output

    def get_sec_grad_data(self, data):

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
        return sec, grad, out

    def get_hess_grad_data(self, data):

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
        return sec, grad, out

        

