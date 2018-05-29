import tensorflow as tf
from LiteNet import *
import unittest
import numpy as np
import time



@unittest.skip('does not work yet')
class test_PolynomialKernel(unittest.TestCase):


    ndim = 4
    nx = 2
    ny  = 3

    def setUp(self):
        self.kernel = PolynomialKernel(self.ndim,2)
        self.X = tf.constant(np.random.randn(self.nx, self.ndim).astype(FDTYPE))
        self.Y = tf.constant(np.random.randn(self.ny, self.ndim).astype(FDTYPE))

        init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(init)

    def test_get_gram_matrix(self):
        gram = self.kernel.get_gram_matrix(self.X, self.Y)
        x = self.X.eval()
        y = self.Y.eval()
        res = gram.eval()

        true_res = (np.dot(x, y.T) + self.kernel.c) ** self.kernel.d
        assert np.allclose(res, true_res)

    def test_get_grad(self):
    
        grad = self.kernel.get_grad(self.X, self.Y)
        grad = grad.eval()

        gram = self.kernel.get_gram_matrix(self.X, self.Y)
        grad_real = np.empty((self.nx,self.ny, self.ndim))
        
        for xi in range(self.nx):
            for yi in range(self.ny):
                   grad_real[xi, yi] = tf.stack(tf.gradients(gram[xi, yi], self.Y)[0][yi]).eval()

        assert np.allclose(grad, grad_real), np.linalg.norm(grad_real-grad)
           
    def test_get_hess(self):
    
        hess = self.kernel.get_hess(self.X, self.Y)
        hess = hess.eval()

        hess_real = np.empty((self.nx,self.ny, self.ndim, self.ndim))
        
        for xi in range(self.nx):
            this_x = self.X[xi]
            for yi in range(self.ny):
                this_y = self.Y[yi]
                gram = self.kernel.get_gram_matrix(this_x, this_y)
                hess_real[xi, yi] = tf.hessians(gram, this_y)[0].eval()
        assert np.allclose(hess, hess_real), np.linalg.norm(hess_real-hess)

    def test_get_sec_grad(self):
    
        sec,grad = self.kernel.get_sec_grad(self.X, self.Y)
        grad = grad.eval()
        sec = sec.eval()

        grad_real = self.kernel.get_grad(self.X, self.Y)
        grad_real = grad_real.eval()

        sec_real = np.empty((self.nx,self.ny, self.ndim))
        
        for xi in range(self.nx):
            this_x = self.X[xi]
            for yi in range(self.ny):
                this_y = self.Y[yi]
                gram = self.kernel.get_gram_matrix(this_x, this_y)
                sec_real[xi, yi] = np.diagonal(tf.hessians(gram, this_y)[0].eval())

        assert np.allclose(grad, grad_real), np.linalg.norm(grad_real-grad)
        assert np.allclose(sec, sec_real), np.linalg.norm(sec_real-sec)

    def test_get_hess_grad(self):

        hess, grad = self.kernel.get_hess_grad(self.X, self.Y)
        grad = grad.eval()
        hess = hess.eval()
        
        grad_real = self.kernel.get_grad(self.X, self.Y)
        grad_real = grad_real.eval()

        hess_real = self.kernel.get_hess(self.X, self.Y)
        hess_real = hess_real.eval()

        assert np.allclose(grad, grad_real), np.linalg.norm(grad_real-grad)
        assert np.allclose(hess, hess_real), np.linalg.norm(hess_real-hess)

@unittest.skip('pass')
class test_GaussianKernel(unittest.TestCase):


    ndim = 4
    nx = 2
    ny  = 3

    def setUp(self):
        self.kernel = GaussianKernel(sigma=1000)
        self.X = tf.constant(np.random.randn(self.nx, self.ndim).astype(FDTYPE))
        self.Y = tf.constant(np.random.randn(self.ny, self.ndim).astype(FDTYPE))

        init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(init)

    def test_get_gram_matrix(self):
        gram = self.kernel.get_gram_matrix(self.X, self.Y)
        x = self.X.eval()
        y = self.Y.eval()
        res = gram.eval()

        true_res = np.array([[np.linalg.norm(x[i]-y[j])**2 for j in range(y.shape[0]) ]for i in range(x.shape[0])])
        true_res = np.exp(-0.5/self.kernel.sigma*true_res)
        assert np.allclose(res, true_res)

    def test_get_grad(self):
    
        grad = self.kernel.get_grad(self.X, self.Y)
        grad = grad.eval()

        gram = self.kernel.get_gram_matrix(self.X, self.Y)
        grad_real = np.empty((self.nx,self.ny, self.ndim))
        
        for xi in range(self.nx):
            for yi in range(self.ny):
                   grad_real[xi, yi] = tf.stack(tf.gradients(gram[xi, yi], self.Y)[0][yi]).eval()

        assert np.allclose(grad, grad_real), np.linalg.norm(grad_real-grad)
           
    def test_get_hess(self):
    
        hess = self.kernel.get_hess(self.X, self.Y)
        hess = hess.eval()

        hess_real = np.empty((self.nx,self.ny, self.ndim, self.ndim))
        
        for xi in range(self.nx):
            this_x = self.X[xi]
            for yi in range(self.ny):
                this_y = self.Y[yi]
                gram = self.kernel.get_gram_matrix(this_x, this_y)
                hess_real[xi, yi] = tf.hessians(gram, this_y)[0].eval()
        assert np.allclose(hess, hess_real), np.linalg.norm(hess_real-hess)

    def test_get_hess_grad(self):

        hess, grad = self.kernel.get_hess_grad(self.X, self.Y)
        grad = grad.eval()
        hess = hess.eval()
        
        grad_real = self.kernel.get_grad(self.X, self.Y)
        grad_real = grad_real.eval()

        hess_real = self.kernel.get_hess(self.X, self.Y)
        hess_real = hess_real.eval()

        assert np.allclose(grad, grad_real), np.linalg.norm(grad_real-grad)
        assert np.allclose(hess, hess_real), np.linalg.norm(hess_real-hess)

    @unittest.skip('does not work yet')
    def test_get_two_grad_cross_hess(self):
            
        dk_dx, dk_dy, d2k_dxdy, _ = self.kernel.get_two_grad_cross_hess(self.X, self.Y)
        d2k_dxdy = d2k_dxdy.eval()
        dk_dx = dk_dx.eval()
        dk_dy = dk_dy.eval()

        d2k_dxdy_real = np.empty(d2k_dxdy.shape)
        dk_dx_real = np.empty(dk_dx.shape)
        dk_dy_real = np.empty(dk_dy.shape)

        # For each pair of x and y vectors, compute the entry of sum_i d2k/dx_1dy_1
        for xi in range(self.nx):

            this_x = self.X[xi]

            for yi in range(self.ny):

                this_y = self.Y[yi]
                
                gram = self.kernel.get_gram_matrix(this_x, this_y)
                
                this_dk_dx = tf.gradients(gram, this_x)[0]
                dk_dx_real[xi, yi] = this_dk_dx.eval()
                this_dk_dy = tf.gradients(gram, this_y)[0]
                dk_dy_real[xi, yi] = this_dk_dy.eval()
                
                # loop over ndim to get cross gradients
                for di in range(self.ndim):
                    d2k_dxdy_real[xi, yi, di] = tf.gradients(this_dk_dx[di], this_y)[0].eval()

        assert np.allclose(d2k_dxdy, d2k_dxdy_real)
        assert np.allclose(dk_dy, dk_dy_real)
        assert np.allclose(dk_dx, dk_dx_real)


@unittest.skip('does not work yet')
class test_LinearNetwork(unittest.TestCase):
    
    ndim_in = (3,)
    ndim_out = (4,)
    ndata  = 10
    batch_size = 5

    def setUp(self):
        
        # fake data and points
        self.data =   np.random.randn(self.ndata, *self.ndim_in).astype(FDTYPE)
        self.data_tensor = tf.constant(self.data)
        
        # build network
        self.network = LinearNetwork(self.ndim_in, self.ndim_out, 
                                     batch_size = self.batch_size)

        self.sess = tf.InteractiveSession()

        init = tf.global_variables_initializer()

        self.sess.run(init)


    def test_forward_array(self):
        out = self.network.forward_array(self.data)
        W = self.network.param['W'].eval()
        b = self.network.param['b'].eval()
        data = self.data
        out_real = b + data.dot(W.T)
        assert np.allclose(out, out_real), np.linalg.norm(out-out_real)**2

    @unittest.skip('not required')
    def test_get_grad_param(self):
        grad_dict, output = self.network.get_grad_param(self.data)
        grad_dict_true = OrderedDict.fromkeys(self.network.param)
        output_true    = self.network.forward_array(self.data)
        for k in grad_dict_true:
            grad_dict_true[k] = np.zeros([self.network.ndim_out[0], self.ndata] + self.network.param[k].shape.dims)

        for di in xrange(self.ndata):
            for oi in xrange(self.network.ndim_out[0]):
                grad_dict_true['W'][oi, di, oi] = self.data[di]
                grad_dict_true['b'][oi, di, 0, oi] = 1

        assert np.allclose(grad_dict_true['W'], grad_dict['W'])
        assert np.allclose(grad_dict_true['b'], grad_dict['b'])
        assert np.allclose(output_true, output)

    def test_get_grad_data(self):

        # reshape data to see if there is a single input
        data, single   = self.network.reshape_data_array(self.data)
        ninput = data.shape[0]

        # number of batches

        grad, _, feed= self.network.get_grad_data()
        grad_data = self.sess.run(grad, feed_dict={feed: data})

        grad_data_real = np.empty(self.ndim_out + ( self.ndata, ) + self.ndim_in)
        W = self.network.param['W'].eval()
        for oi in xrange(self.ndim_out[0]):
            for di in xrange(self.ndata):
                grad_data_real[oi, di] = W[oi]
        assert np.allclose(grad_data, grad_data_real)



@unittest.skip('does not work yet')
class test_SquareNetwork(unittest.TestCase):
    
    ndim_in = (5,)
    ndim_out = (3,)
    ndata  = 10
    batch_size = 1

    def setUp(self):
        
        # fake data and points
        self.data =   np.random.randn(self.ndata, *self.ndim_in).astype(FDTYPE)
        self.data_tensor = tf.constant(self.data)
        
        # build network
        self.network = SquareNetwork(self.ndim_in, self.ndim_out, 
                                     batch_size = self.batch_size)
        self.sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        self.sess.run(init)


    def test_forward_array(self):
        out = self.network.forward_array(self.data)
        W = self.network.param['W'].eval()
        b = self.network.param['b'].eval()
        data = self.data
        out_real = (b + data.dot(W.T)) ** 2.0
        assert np.allclose(out, out_real), np.linalg.norm(out-out_real)**2

    @unittest.skip('not required')
    def test_get_grad_param(self):

        W = self.network.param['W'].eval()
        b = self.network.param['b'].eval()
        data = self.data

        grad_dict, output = self.network.get_grad_param(data)
        grad_dict_true = OrderedDict.fromkeys(self.network.param)
        output_true    = self.network.forward_array(data)
        for k in grad_dict_true:
            grad_dict_true[k] = np.zeros([self.network.ndim_out[0], self.ndata] + self.network.param[k].shape.dims)

        lin = data.dot(W.T)+b
        for oi in xrange(self.network.ndim_out[0]):
            for di in xrange(self.ndata):
                grad_dict_true['W'][oi, di, oi] = 2*lin[di, oi]*data[di]
                grad_dict_true['b'][oi, di, 0, oi] = 2*lin[di, oi]

        assert np.allclose(output_true, output)
        assert np.allclose(grad_dict_true['W'], grad_dict['W'])
        assert np.allclose(grad_dict_true['b'], grad_dict['b'])

    def test_get_grad_data(self):

        # reshape data to see if there is a single input
        data, single   = self.network.reshape_data_array(self.data)
        ninput = data.shape[0]

        # number of batches
        batch_size = self.network.batch_size
        nbatch = ninput/self.network.batch_size
        output_true    = self.network.forward_array(self.data)
        
        # run the model
        grad, _, feed= self.network.get_grad_data()
        results = []
        for bi in xrange(nbatch):
            batch_data = data[bi*batch_size:(bi+1)*batch_size]
            results.append(self.sess.run(grad, feed_dict={feed: batch_data}))
        grad_data = np.concatenate(results, axis=1)
        
        # construct the real 
        grad_data_real = np.empty((self.ndim_out[0], self.ndata) + self.ndim_in)
        W = self.network.param['W'].eval()
        b = self.network.param['b'].eval()
        lin = data.dot(W.T)+b

        for oi in xrange(self.ndim_out[0]):
            for di in xrange(self.ndata):
                    grad_data_real[oi, di] = 2*W[oi]*lin[di, oi]
        assert np.allclose(grad_data, grad_data_real)

    '''    
    def test_get_sec_data(self):
        

        # reshape data to see if there is a single input
        data, single   = self.network.reshape_data_array(self.data)
        ninput = data.shape[0]

        # number of batches
        batch_size = self.network.batch_size
        nbatch = ninput/self.network.batch_size
        output_true    = self.network.forward_array(self.data)

        # run the model
        sec, _, _, feed = self.network.get_sec_data()
        results = []
        for bi in xrange(nbatch):
            batch_data = data[bi*batch_size:(bi+1)*batch_size]
            results.append(self.sess.run(sec, feed_dict={feed: batch_data}))
        sec_data = np.concatenate(results, axis=1)

        # construct the real 
        sec_data_real = np.empty((self.ndim_out[0], self.ndata) + self.ndim_in)
        W = self.network.param['W'].eval()
        b = self.network.param['b'].eval()
        lin = data.dot(W.T)+b

        for oi in xrange(self.ndim_out[0]):
            sec_data_real[oi] = 2*W[oi]**2
        
        assert np.allclose(sec_data_real, sec_data)
    '''    

@unittest.skip('does not work yet')
class test_LinearReLUNetwork(unittest.TestCase):
    
    ndim_in = (10,)
    ndim_out =(5,)
    ndata  = 1000
    batch_size = 5

    def setUp(self):
        
        # fake data and points
        self.data =   np.random.randn(self.ndata, *self.ndim_in).astype(FDTYPE)
        self.data_tensor = tf.constant(self.data)
        
        # build network
        self.network = LinearReLUNetwork(self.ndim_in, self.ndim_out, 
                                     batch_size = self.batch_size, lin_grads=[0.2,1.0])

        self.sess = tf.InteractiveSession()

        init = tf.global_variables_initializer()

        self.sess.run(init)


    def test_forward_array(self):
        out = self.network.forward_array(self.data)
        W = self.network.param['W'].eval()
        b = self.network.param['b'].eval()
        data = self.data
        out_real = b + data.dot(W.T)
        out_real = ((out_real>0)*self.network.lin_grads[1] + (out_real<=0)*self.network.lin_grads[0])*out_real
        assert np.allclose(out, out_real, atol=1e-4), np.max(np.sqrt((out-out_real)**2))

    def test_forward_tensor(self):
        out = self.network.forward_tensor(self.data_tensor).eval()
        out_real = self.network.forward_array(self.data)
        assert np.allclose(out, out_real, atol=1e-4), np.max(np.sqrt((out-out_real)**2))
       
    '''
    @unittest.skip('not required')
    def test_get_grad_param(self):
        grad_dict, output = self.network.get_grad_param(self.data)
        grad_dict_true = OrderedDict.fromkeys(self.network.param)
        output_true    = self.network.forward_array(self.data)
        for k in grad_dict_true:
            grad_dict_true[k] = np.zeros([self.network.ndim_out, self.ndata] + self.network.param[k].shape.dims)

        for di in xrange(self.ndata):
            for oi in xrange(self.network.ndim_out):
                grad_dict_true['W'][oi, di, oi] = self.data[di] * (output[di,oi]>0)
                grad_dict_true['b'][oi, di, 0, oi] = 1 * (output[di,oi]>0)

        assert np.allclose(grad_dict_true['W'], grad_dict['W'])
        assert np.allclose(grad_dict_true['b'], grad_dict['b'])
        assert np.allclose(output_true, output)
    '''

    def test_get_grad_data(self):
        
        # reshape data to see if there is a single input
        data, single   = self.network.reshape_data_array(self.data)
        ninput = data.shape[0]

        # number of batches
        batch_size = self.network.batch_size
        nbatch = ninput/self.network.batch_size

        grad, _, feed= self.network.get_grad_data()
        t0 = time.time()
        results = []
        for bi in xrange(nbatch):
            batch_data = data[bi*batch_size:(bi+1)*batch_size]
            results.append(self.sess.run(grad, feed_dict={feed: batch_data}))
        grad_data = np.concatenate(results, axis=1)

        output_true = self.network.forward_array(self.data)

        grad_data_real = np.empty(self.ndim_out +( self.ndata, ) + self.ndim_in)
        W = self.network.param['W'].eval()
        for oi in xrange(self.ndim_out[0]):
            for di in xrange(self.ndata):
                grad_data_real[oi, di] = W[oi] * ( (output_true[di, oi]>0 )*self.network.lin_grads[1] +\
                                                   (output_true[di, oi]<=0)*self.network.lin_grads[0]) 
        assert np.allclose(grad_data, grad_data_real)
    
    '''    
    def test_get_sec_data(self):

        data, single   = self.network.reshape_data_array(self.data)
        ninput = data.shape[0]

        # number of batches
        batch_size = self.network.batch_size

        sec, _, _, feed = self.network.get_sec_data()
        batch_data = data[0:batch_size]
        results = self.sess.run(sec, feed_dict={feed: batch_data})
        assert results.prod() == 0
        assert results.shape == (self.ndim_out[0], self.batch_size,) + self.ndim_in
    '''    
@unittest.skip('does not work yet')
class test_LinearSoftNetwork(unittest.TestCase):
    
    ndim_in = (3,)
    ndim_out =(2,)
    ndata  = 5

    def setUp(self):
        
        # fake data and points
        self.data =   np.random.randn(self.ndata, *self.ndim_in).astype(FDTYPE)*10
        self.data_tensor = tf.constant(self.data)
        
        # build network
        self.network = LinearSoftNetwork(self.ndim_in, self.ndim_out, init_std = 1)

        self.sess = tf.InteractiveSession()

        init = tf.global_variables_initializer()

        self.sess.run(init)


    def test_forward_array(self):
        out = self.network.forward_array(self.data)
        W = self.network.param['W'].eval()
        b = self.network.param['b'].eval()
        data = self.data
        out_real = np.log(1+np.exp(b + data.dot(W.T)))
        inf_idx = np.isinf(out_real)
        out_real[inf_idx] = (b + data.dot(W.T))[inf_idx]
        assert np.all(np.isfinite(out))
        assert np.allclose(out, out_real, atol=2e-6), np.max(np.sqrt((out-out_real)**2))

    def test_forward_tensor(self):
        out = self.network.forward_tensor(self.data_tensor).eval() 
        out_real = self.network.forward_array(self.data)
        assert np.allclose(out, out_real, atol=1e-6), np.max(np.sqrt((out-out_real)**2))
       
    '''
    @unittest.skip('not required')
    def test_get_grad_param(self):
        grad_dict, output = self.network.get_grad_param(self.data)
        grad_dict_true = OrderedDict.fromkeys(self.network.param)
        output_true    = self.network.forward_array(self.data)
        for k in grad_dict_true:
            grad_dict_true[k] = np.zeros([self.network.ndim_out, self.ndata] + self.network.param[k].shape.dims)

        for di in xrange(self.ndata):
            for oi in xrange(self.network.ndim_out):
                grad_dict_true['W'][oi, di, oi] = self.data[di] * (output[di,oi]>0)
                grad_dict_true['b'][oi, di, 0, oi] = 1 * (output[di,oi]>0)

        assert np.allclose(grad_dict_true['W'], grad_dict['W'])
        assert np.allclose(grad_dict_true['b'], grad_dict['b'])
        assert np.allclose(output_true, output)
    '''

    def test_get_grad_data(self):

        data, single   = self.network.reshape_data_array(self.data)
        ninput = data.shape[0]

        grad, _, feed = self.network.get_grad_data()
        grad_data = self.sess.run(grad, feed_dict={feed: data})

        grad_real = np.empty((self.ndim_out + (self.ndata,) + self.ndim_in))

        for di in range(self.data.shape[0]):

            this_data = self.data_tensor[di]
            out = self.network.forward_tensor(this_data)

            for oi in xrange(self.ndim_out[0]):

                g = tf.gradients(out[oi], this_data)[0].eval()
                grad_real[oi, di, :] = g

        assert np.all(np.isfinite(grad_data))
        assert np.allclose(grad_data, grad_real), np.linalg.norm(grad_real-grad_data)

    def test_get_sec_grad_data(self):

        data, single   = self.network.reshape_data_array(self.data)
        ninput = data.shape[0]

        # number of batches

        sec, grad, _, feed = self.network.get_sec_grad_data()
        sec_data = self.sess.run(sec, feed_dict={feed: data})

        sec_real = np.empty((self.ndim_out + (self.ndata,) + self.ndim_in))
        for oi in range(self.network.ndim_out[0]):
            for di in range(self.data.shape[0]):

                this_data = self.data_tensor[di]
                out = self.network.forward_tensor(this_data)[oi,...]
                sec = tf.hessians(out, this_data)[0].eval()
                sec_real[oi, di, :] = np.diagonal(sec)

        assert np.all(np.isfinite(sec_data))
        assert np.allclose(sec_real, sec_data), np.linalg.norm(sec_real-sec_data)

    def test_get_hess_grad_data(self):

        data, single   = self.network.reshape_data_array(self.data)
        ninput = data.shape[0]

        # number of batches

        hess, grad, _, feed = self.network.get_hess_grad_data()
        hess_data = self.sess.run(hess, feed_dict={feed: data})

        grad_data = self.sess.run(grad, feed_dict={feed: data})
        grad, _, feed = self.network.get_grad_data()
        grad_real = self.sess.run(grad, feed_dict={feed: data})
        

        hess_real = np.empty((self.ndim_out + (self.ndata,) + self.ndim_in*2))
        for oi in range(self.network.ndim_out[0]):
            for di in range(self.data.shape[0]):

                this_data = self.data_tensor[di]
                out = self.network.forward_tensor(this_data)[oi,...]
                h = tf.hessians(out, this_data)[0].eval()
                hess_real[oi, di, :, :] = h

        assert np.all(np.isfinite(hess_data))
        assert np.allclose(hess_real, hess_data), np.linalg.norm(hess_real-hess_data)
        assert np.allclose(grad_data, grad_real)

#@unittest.skip('not required')
class test_DenseLinearSoftNetwork(unittest.TestCase):
    
    ndim_in = [(3,), (3,)]
    ndim_out =(2,)
    ndata  = 3

    def setUp(self):
        np.random.seed(1) 
        # fake data and points
        self.nin = len(self.ndim_in)
        self.data = [np.random.randn(self.ndata, *self.ndim_in[i]).astype(FDTYPE)*0.1 for i in range(self.nin)]
        self.data_tensor = [tf.constant(self.data[i]) for i in range(self.nin)]
        
        # build network
        self.network = DenseLinearSoftNetwork(self.ndim_in, self.ndim_out, init_std = 1)

        self.sess = tf.InteractiveSession()

        init = tf.global_variables_initializer()

        self.sess.run(init)


    def test_forward_array(self):

        out = self.network.forward_array(self.data)

        b = self.network.param['b'].eval()

        data = self.data
        
        out_real = 0
        for i in range(self.nin):
            W = self.network.param['W'+str(i)].eval()
            out_real += data[i].dot(W.T)
        out_real += b
        #out_real = np.where(out_real<0, np.exp(0.5*out_real), np.log(1+np.exp(out_real)))-1
        out_real  = np.log(1+np.exp(out_real)) - 1
        assert np.all(np.isfinite(out))
        assert np.allclose(out, out_real, atol=2e-6), np.max(np.sqrt((out-out_real)**2))

    def test_forward_tensor(self):
        out = self.network.forward_tensor(self.data_tensor).eval() 
        out_real = self.network.forward_array(self.data)
        assert np.allclose(out, out_real, atol=1e-6), np.max(np.sqrt((out-out_real)**2))
       
    def test_get_grad_data(self):


        grad, _, feed = self.network.get_grad_data()
        grad_data = self.sess.run(grad, feed_dict=dict(zip(feed, self.data)))

        grad_real = [np.empty((self.ndim_out + (self.ndata,) + self.ndim_in[i])) for i in range(self.nin)]

        for di in range(self.ndata):

            this_data = [self.data_tensor[i][di][None,:] for i in range(self.nin)]
            out = self.network.forward_tensor(this_data)[0]

            for oi in xrange(self.ndim_out[0]):

                g = self.sess.run(tf.gradients(out[oi], this_data))

                for i in range(self.nin):
                    grad_real[i][oi, di, :] = g[i]

        assert np.all(np.isfinite(grad_data))
        assert np.allclose(grad_data, grad_real), np.linalg.norm(grad_real-grad_data)

    def test_get_sec_grad_data(self):

        # number of batches

        sec, grad, _, feed = self.network.get_sec_grad_data()
        sec_data = self.sess.run(sec, feed_dict=dict(zip(feed, self.data)))

        sec_real = [np.empty((self.ndim_out + (self.ndata,) + self.ndim_in[i])) for i in range(self.nin)]

        for di in range(self.ndata):

            this_data_flat = [ self.data_tensor[i][di] for i in range(self.nin) ]
            this_data = [ this_data_flat[i][None,:] for i in range(self.nin) ]
            out = self.network.forward_tensor(this_data)[0]

            for oi in range(self.network.ndim_out[0]):

                sec = self.sess.run(tf.hessians(out[oi], this_data_flat))
                for i in range(self.nin):
                    sec_real[i][oi, di, :] = np.diagonal(sec[i])
       
        assert np.all(np.isfinite(sec_data))
        assert np.allclose(sec_real, sec_data), np.linalg.norm(sec_real-sec_data)

    def test_get_hess_grad_data(self):


        hess, _, grad, _, feed = self.network.get_hess_cross_grad_data()
        hess_data = self.sess.run(hess, feed_dict=dict(zip(feed, self.data)))

        grad_data = self.sess.run(grad, feed_dict=dict(zip(feed, self.data)))
        grad, _, feed = self.network.get_grad_data()
        grad_real = self.sess.run(grad, feed_dict=dict(zip(feed, self.data)))
        

        hess_real = [np.empty((self.ndim_out + (self.ndata,) + self.ndim_in[i]*2)) for i in range(self.nin)]

        for di in range(self.ndata):

            this_data_flat = [ self.data_tensor[i][di] for i in range(self.nin) ]
            this_data = [ this_data_flat[i][None,:] for i in range(self.nin) ]
            out = self.network.forward_tensor(this_data)[0]

            for oi in range(self.network.ndim_out[0]):

                h = self.sess.run(tf.hessians(out[oi], this_data_flat))
                for i in range(self.nin):
                    hess_real[i][oi, di, :, :] = h[i]
        
        assert np.all(np.isfinite(hess_data))
        assert np.allclose(hess_real, hess_data), np.linalg.norm(hess_real-hess_data)
        assert np.allclose(grad_data, grad_real)

@unittest.skip('not required')
class test_DeepNetwork(unittest.TestCase):
    
    ndim_in = (3,)
    ndim_2 =  (2,)
    ndim_3 =  (2,)

    ndata  = 3

    def setUp(self):
        
        # fake data and points
        self.data =   np.random.randn(self.ndata, *self.ndim_in).astype(FDTYPE)*0.1
        self.data_tensor = tf.constant(self.data)
        
        # build network
        self.ndim_out = (2,)
        layer_1 = LinearSoftNetwork(self.ndim_in, self.ndim_2, init_std = 1)
        layer_2 = LinearSoftNetwork(self.ndim_2, self.ndim_3, init_std = 1)
        layer_3 = LinearSoftNetwork(self.ndim_3, self.ndim_out, init_std = 1)

        self.network = DeepNetwork([layer_1], ndim_out = (2,), add_skip=False)
        self.ndim_out = self.network.ndim_out

        self.sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def test_forward_tensor(self):
        
        out = self.network.forward_tensor(self.data_tensor).eval()
        out_real = self.network.forward_array(self.data)

        assert np.allclose(out, out_real), np.max(np.abs(out-out_real))

    def test_forward_data(self):
        
        out = self.network.forward_array(self.data)
        
        out_real = self.data
        for l in self.network.layers:
            out_real = l.forward_array(out_real)
        
        if self.network.add_skip:
            out_real = self.network.skip_layer.forward_array([self.data, out_real])
        
        assert np.allclose(out, out_real), np.max(np.abs(out-out_real))

    def test_get_grad_data(self):

        grad, _, feed = self.network.get_grad_data()
        grad_data = self.sess.run(grad, feed_dict={feed: self.data})
        print self.data.shape

        grad_real = np.empty((self.ndim_out + (self.ndata,) + self.ndim_in))

        for di in range(self.data.shape[0]):

            this_data = self.data_tensor[di][None,:]
            out = self.network.forward_tensor(this_data)[0]

            for oi in xrange(self.ndim_out[0]):

                g = tf.gradients(out[oi], this_data)[0].eval()
                grad_real[oi, di, :] = g

        assert np.all(np.isfinite(grad_data))
        assert np.allclose(grad_data, grad_real, atol=1e-6, rtol=1e-4), np.linalg.norm(grad_real-grad_data)/np.linalg.norm(grad_real)

    def test_get_sec_data(self):

        # number of batches

        sec, _, _, feed = self.network.get_sec_grad_data()
        sec_data = self.sess.run(sec, feed_dict={feed: self.data})

        sec_real = np.empty((self.ndim_out + (self.ndata,) + self.ndim_in))
        for oi in range(self.network.ndim_out[0]):
            for di in range(self.data.shape[0]):

                this_data_flat = self.data_tensor[di]
                this_data = this_data_flat[None,:]
                out = self.network.forward_tensor(this_data)[0,oi]
                sec = tf.hessians(out, this_data_flat)[0].eval()
                sec_real[oi, di, :] = np.diagonal(sec)
        assert np.all(np.isfinite(sec_data))
        assert np.allclose(sec_real, sec_data, atol=1e-6, rtol=1e-4), np.linalg.norm(sec_real-sec_data)/np.linalg.norm(sec_real)

    def test_get_hess_grad_data(self):

        # number of batches

        hess, _ , _, feed = self.network.get_hess_grad_data()
        hess_data = self.sess.run(hess, feed_dict={feed: self.data})

        hess_real = np.empty((self.ndim_out + (self.ndata,) + self.ndim_in*2))

        for oi in range(self.network.ndim_out[0]):
            for di in range(self.data.shape[0]):
                this_data_flat = self.data_tensor[di]
                this_data = this_data_flat[None,:]
                out = self.network.forward_tensor(this_data)[0,oi]
                hess_real[oi, di]= tf.hessians(out, this_data_flat)[0].eval()

        assert np.all(np.isfinite(hess_data))
        assert np.allclose(hess_real, hess_data, atol=1e-6, rtol=1e-4), np.linalg.norm(hess_real-hess_data)/np.linalg.norm(hess_real)


#@unittest.skip('not required')
class test_LiteModel(unittest.TestCase):

    ndata  = 2
    ndim_in = (5,)
    npoint = 5
    ndim_out = (5,)

    def setUp(self):
        

        # fake data and points
        self.data =   np.random.randn(self.ndata , *self.ndim_in).astype(FDTYPE)
        self.data_tensor    =   tf.constant(self.data)

        self.data_val =   np.random.randn(self.ndata , *self.ndim_in).astype(FDTYPE)

        self.points = np.random.randn(self.npoint, *self.ndim_in).astype(FDTYPE)
        self.points_tensor  =   tf.constant(self.points)

        # setup lite model with fake alpha and points
        kernel  = GaussianKernel(1.0)
        network_1 = LinearSoftNetwork(self.ndim_in, self.ndim_out, init_std  = 1.0)
        network_2 = LinearSoftNetwork(self.ndim_out, self.ndim_out, init_std = 1.0)
        network_3 = LinearSoftNetwork(self.ndim_out, self.ndim_out, init_std = 1.0)
        network   = DeepNetwork([network_1, network_2], ndim_out = self.ndim_out, add_skip=True)
        #kernel    = NetworkKernel(network)
        kernel  =   CompositeKernel(kernel, network)
        kernel  =   MixtureKernel([kernel, GaussianKernel(1.0)], [1.0,1.0])

        alpha   = tf.Variable( np.random.randn(self.npoint).astype(FDTYPE))

        self.model = LiteModel(kernel, alpha=alpha, init_log_lam=0.0)
        self.model.set_points( self.points_tensor )
 
        # compute and store the features 
        init = tf.global_variables_initializer()

        self.sess = tf.InteractiveSession()
        self.sess.run(init)
    
    def test_val_score(self):

        train_data = tf.placeholder(FDTYPE, shape = (None,) + self.ndim_in)
        val_data = tf.placeholder(FDTYPE, shape = (None,) + self.ndim_in)
        score, train_data, val_data = self.model.val_score(train_data=train_data, val_data=val_data)[1:4]

        feed_dict = { train_data   : self.data,
                      val_data: self.data_val}

        score = self.sess.run(score, feed_dict=feed_dict)
        
        score_real = 0.0

        this_data_flat = tf.placeholder(FDTYPE, np.prod(self.ndim_in))
        this_data      = tf.reshape(this_data_flat, (1,)+self.ndim_in)
        f = self.model.evaluate_fun(this_data)
        one_data_score = tf.trace(tf.hessians(f, this_data_flat)[0]) + \
                         0.5 * tf.reduce_sum(tf.gradients(f, this_data_flat)[0]**2)

        for yi in range(self.ndata):
            score_real += one_data_score.eval({
                                  train_data: self.data,
                                  this_data: self.data_val[yi:yi+1]})

        score_real/=self.ndata
        assert np.allclose(score, score_real, atol=1e-6, rtol=1e-4), (score, score_real)

    def test_score(self):

        score, train_data, val_data = self.model.val_score()[1:4]
        feed_dict = { train_data   : self.data,
                      val_data: self.data}
        score = self.sess.run(score, feed_dict=feed_dict)
        
        score_real = 0.0

        this_data_flat = tf.placeholder(FDTYPE, np.prod(self.ndim_in))
        this_data      = tf.reshape(this_data_flat, (1,)+self.ndim_in)

        f = self.model.evaluate_fun(this_data)
        one_data_score = tf.trace(tf.hessians(f, this_data_flat)[0]) + \
                         0.5 * tf.reduce_sum(tf.gradients(f, this_data_flat)[0]**2)

        for yi in range(self.ndata):
            score_real += one_data_score.eval({
                                  train_data: self.data,
                                  this_data: self.data[yi:yi+1]})
        score_real/=self.ndata
        assert np.allclose(score, score_real, atol=1e-6, rtol=1e-4), (score, score_real)

    def test_opt_score(self):
        
        # build assign operator
        alpha_assign_opt, loss, score, data = self.model.opt_score()[:4]

        initial_score_value = \
                self.sess.run(score, feed_dict={data: self.data})

        self.sess.run(alpha_assign_opt, feed_dict={data:self.data})

        # assign random deviations to alpha
        assign_op = tf.assign_add(self.model.alpha, tf.random_normal([self.npoint], stddev=0.1, dtype=FDTYPE))
        
        # evaluate the optimal score after assigning optimal alpha
        opt_score_value = self.sess.run(score, feed_dict={data : self.data})

        for i in range(10):
            self.sess.run(assign_op)
            score_value = self.sess.run(score, feed_dict={data : self.data})
            assert opt_score_value<=score_value

    def test_evaluate_grad(self):
        
        
        grad_data = self.model.evaluate_grad(self.data_tensor).eval()
        grad_real = np.empty( (self.ndata,) + self.ndim_in)

        for di in range(self.data.shape[0]):

            this_data_flat = self.data_tensor[di]
            this_data = this_data_flat[None,:]
            out = self.model.evaluate_fun(this_data)

            g = tf.gradients(out, this_data_flat)[0].eval()
            grad_real[di, :] = g

        assert np.all(np.isfinite(grad_data))
        assert np.allclose(grad_data, grad_real, atol=1e-6, rtol=1e-4), np.linalg.norm(grad_real-grad_data)/np.linalg.norm(grad_real)
        

    def test_evaluate_hess(self):
        
        
        hess_data = self.model.evaluate_hess(self.data_tensor).eval()
        hess_real = np.empty( (self.ndata,) + self.ndim_in*2)

        for di in range(self.data.shape[0]):

            this_data_flat = self.data_tensor[di]
            this_data = this_data_flat[None,:]
            out = self.model.evaluate_fun(this_data)

            g = tf.hessians(out, this_data_flat)[0].eval()
            hess_real[di] = g

        assert np.all(np.isfinite(hess_data))
        assert np.allclose(hess_data, hess_real, atol=1e-6, rtol=1e-4), np.linalg.norm(hess_real-hess_data)/np.linalg.norm(hess_real)

    def test_evaluate_hess_grad_fun(self):
        
        h,g,f = self.model.evaluate_hess_grad_fun(self.data_tensor)

        h_true = self.model.evaluate_hess(self.data_tensor).eval()
        g_true = self.model.evaluate_grad(self.data_tensor).eval()
        f_true = self.model.evaluate_fun(self.data_tensor).eval()

        h = h.eval()
        g = g.eval()
        f = f.eval()
        
        assert np.all(np.isfinite(h))
        assert np.all(np.isfinite(g))
        assert np.all(np.isfinite(f))
        
        assert np.allclose(f, f_true)
        assert np.allclose(g, g_true)
        assert np.allclose(h, h_true)

        
        
        hess_data = self.model.evaluate_hess(self.data_tensor).eval()
        hess_real = np.empty( (self.ndata,) + self.ndim_in*2)

        for di in range(self.data.shape[0]):

            this_data_flat = self.data_tensor[di]
            this_data = this_data_flat[None,:]
            out = self.model.evaluate_fun(this_data)

            g = tf.hessians(out, this_data_flat)[0].eval()
            hess_real[di] = g

        assert np.all(np.isfinite(hess_data))
        assert np.allclose(hess_data, hess_real, atol=1e-6, rtol=1e-4), np.linalg.norm(hess_real-hess_data)/np.linalg.norm(hess_real)

###########################
###########################
### OTHER STUFF ########### 
###########################
###########################



@unittest.skip('does not work yet')
class test_ConvNetwork(unittest.TestCase):

    '''
        ndim_in: nchannel x size x size
        nfil   : number of filters
        filter_size: length of a square filter
        stride : 

        input: CWH
        filter: nchannel x size x size x nfil 
    '''
    
    ndim_in  = (1,10,10)
    nfil     = 3
    filter_size = 4
    stride   = 3
    ndata  = 10
    batch_size = 5
    flatten=True

    def setUp(self):
        
        # fake data and points
        self.data =   np.random.randn(self.ndata, *self.ndim_in).astype(FDTYPE)
        self.data_tensor = tf.constant(self.data)
        self.size_out = (self.ndim_in[-1] - self.filter_size)/self.stride + 1
        
        # build network
        self.network = ConvNetwork( self.ndim_in, self.nfil, self.filter_size, 
                                    self.stride, self.batch_size, 
                                    lin_grads = [0.2, 1.0])
        self.ndim_out = self.network.ndim_out
        self.sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def test_forward_array(self):

        W = self.network.param['W'].eval()
        W = np.moveaxis(W, [0,1,2,3],[2,3,1,0])
        b = self.network.param['b'].eval()
        data = self.data
        
        size_out = self.size_out
        output = np.empty((self.ndata, self.nfil, size_out, size_out))
        for i in range(size_out):
            for j in range(size_out):
                patch = data[:,:,
                            i*self.stride:i*self.stride+self.filter_size,
                            j*self.stride:j*self.stride+self.filter_size]
                output[:,:,i,j] = np.einsum('ikmn,jkmn->ij', patch, W)
        
        output += b
        if self.network.flatten:
            output = output.reshape(self.ndata, -1)
        output_true = np.maximum(self.network.lin_grads[0] * output, self.network.lin_grads[1]*output)
        output = self.network.forward_array(self.data)

        assert np.allclose(output,  output_true, atol=1e-5), np.abs(output-output_true).max()
    
    @unittest.skipIf(flatten is False, 'skip test for non-flat conv output')
    def test_get_grad_data(self):

        # reshape data to see if there is a single input
        data, _   = self.network.reshape_data_array(self.data)
        ninput = data.shape[0]

        # number of batches
        batch_size = self.network.batch_size
        nbatch = ninput/batch_size
        
        grad, out, data = self.network.get_grad_data()
        grad_val = np.zeros((self.ndim_out + (self.ndata,)+self.ndim_in))

        t0 = time.time()

        for bi in range(nbatch):
            grad_val[:, bi*batch_size : (bi+1)*batch_size, ...] = \
                grad.eval({data:self.data[bi*batch_size:(bi+1)*batch_size]})
        print 'get_grad: ', time.time()-t0
        
        grad_real_val = np.zeros(self.ndim_out + (self.ndata,) + self.ndim_in)
        out = self.network.forward_tensor(self.data_tensor)
        for di in range(self.ndata):
            for oi in range(self.ndim_out[0]):
                g = tf.gradients(out[di,oi], self.data_tensor)[0][di]
                grad_real_val[oi, di] = g.eval() 

        assert np.allclose(grad_val, grad_real_val)
       

    @unittest.skip('derivatie w.r.t param not needed')
    def test_get_grad_param(self):

        W = self.network.param['W'].eval()
        b = self.network.param['b'].eval()
        data = self.data

        grad_dict, output = self.network.get_grad_param(data)
        assert grad_dict['W'].shape == (self.ndim_out, self.ndata, 
                                        self.filter_size, self.filter_size, 
                                        self.ndim_in[0], self.nfil)


    '''
    def test_get_sec_data(self):

        # reshape data to see if there is a single input
        data, single   = self.network.reshape_data_array(self.data)
        ninput = data.shape[0]

        # number of batches
        batch_size = self.network.batch_size
        nbatch = ninput/self.network.batch_size
        output_true    = self.network.forward_array(self.data)
        sec, _, _, feed = self.network.get_sec_data()

        # run the model
        results = [] 
        t0 = time.time()
        for bi in xrange(nbatch):
            batch_data = data[bi*batch_size:(bi+1)*batch_size]
            results.append(self.sess.run(sec, feed_dict={feed: batch_data}))
        sec_data = np.concatenate(results, axis=1)
        print 'second derivative: %d data took %.3f sec' % (ninput, time.time()-t0)

    '''





@unittest.skip('deprecated')
class test_KernelNetModel(unittest.TestCase):

    ndata  = 10
    ndim_in = (7,)
    filter_size = 3
    stride = 2
    ndim = (6,)
    npoint = 5

    def setUp(self):
        

        # fake data and points
        self.data =   np.random.randn(self.ndata , *self.ndim_in).astype(FDTYPE)
        self.data_tensor    =   tf.constant(self.data)
        self.points = np.random.randn(self.npoint, *self.ndim_in).astype(FDTYPE)
        self.points_tensor  =   tf.constant(self.points)

        network = LinearSoftNetwork(self.ndim_in, self.ndim)

        # setup lite model with fake alpha and points
        kernel  = GaussianKernel(1)
        alpha   = tf.Variable( np.random.randn(self.npoint).astype(FDTYPE)*0.1 )

        self.model = KernelNetModel(kernel, network, alpha)
        self.model.set_points( self.points_tensor )
        self.X = self.model.X
 
        # compute and store the features 
        self.Y = self.model.network.forward_tensor(self.data_tensor)
        init = tf.global_variables_initializer()

        self.sess = tf.InteractiveSession()
        self.sess.run(init)
    
    @unittest.skip("")
    def test_kernel_score(self):

        # fake coefficients
        score = self.model.kernel_score(self.data_tensor)

        alpha = self.model.alpha.eval()
        X = self.X.eval()
        Y = self.Y.eval()
        true_score = 0.0

        sigma = self.model.kernel.sigma
        for l in range(self.ndim[0]):
            for i in range(self.npoint):
                for j in range(self.ndata):
                    true_score += alpha[i]*np.exp(-np.linalg.norm(X[i]-Y[j])**2/sigma/2.0) * (-1 + 1/sigma * (X[i,l]-Y[j,l])**2)
        
        for l in range(self.ndim[0]):
            for i in range(self.ndata):
                sq = 0
                for j in range(self.npoint):
                    sq += alpha[j]*(X[j,l]-Y[i,l]) * np.exp(-np.linalg.norm(X[j]-Y[i])**2/sigma/2.0) 
                true_score += sq**2.0/(sigma)/2.0
        true_score *=  1.0 / self.ndata/sigma
        assert np.allclose(score.eval(), true_score), (score.eval, true_score)



unittest.main()
