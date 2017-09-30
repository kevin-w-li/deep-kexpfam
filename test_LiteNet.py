import tensorflow as tf
from LiteNet import *
import unittest
import numpy as np
import time

@unittest.skip('does not work yet')
class test_GaussianKernel(unittest.TestCase):


    ndim = 4
    nx = 2
    ny  = 3

    def setUp(self):
        self.kernel = GaussianKernel(self.ndim, sigma=3.2)
        self.X = tf.constant(np.random.randn(self.nx, self.ndim).astype('float32'))
        self.Y = tf.constant(np.random.randn(self.ny, self.ndim).astype('float32'))

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


@unittest.skip('does not work yet')
class test_LinearNetwork(unittest.TestCase):
    
    ndim_in = (3,)
    ndim_out = 4
    ndata  = 10
    batch_size = 5

    def setUp(self):
        
        # fake data and points
        self.data =   np.random.randn(self.ndata, *self.ndim_in).astype('float32')
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
            grad_dict_true[k] = np.zeros([self.network.ndim_out, self.ndata] + self.network.param[k].shape.dims)

        for di in xrange(self.ndata):
            for oi in xrange(self.network.ndim_out):
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
        batch_size = self.network.batch_size
        nbatch = ninput/self.network.batch_size

        grad, _, feed= self.network.get_grad_data()
        results = []
        for bi in xrange(nbatch):
            batch_data = data[bi*batch_size:(bi+1)*batch_size]
            results.append(self.sess.run(grad, feed_dict={feed: batch_data}))
        grad_data = np.concatenate(results, axis=1)

        grad_data_real = np.empty((self.ndim_out, self.ndata) + self.ndim_in)
        W = self.network.param['W'].eval()
        for oi in xrange(self.ndim_out):
            for di in xrange(self.ndata):
                grad_data_real[oi, di] = W[oi]
        assert np.allclose(grad_data, grad_data_real)

    def test_get_sec_data(self):

        data, single   = self.network.reshape_data_array(self.data)
        ninput = data.shape[0]

        # number of batches
        batch_size = self.network.batch_size

        sec, _, _, feed = self.network.get_sec_data()
        batch_data = data[0:batch_size]
        results = self.sess.run(sec, feed_dict={feed: batch_data})
        assert results.prod() == 0
        assert results.shape == (self.ndim_out, self.batch_size,) + self.ndim_in



@unittest.skip('does not work yet')
class test_SquareNetwork(unittest.TestCase):
    
    ndim_in = (5,)
    ndim_out = 3
    ndata  = 10
    batch_size = 1

    def setUp(self):
        
        # fake data and points
        self.data =   np.random.randn(self.ndata, *self.ndim_in).astype('float32')
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
            grad_dict_true[k] = np.zeros([self.network.ndim_out, self.ndata] + self.network.param[k].shape.dims)

        lin = data.dot(W.T)+b
        for oi in xrange(self.network.ndim_out):
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
        grad_data_real = np.empty((self.ndim_out, self.ndata) + self.ndim_in)
        W = self.network.param['W'].eval()
        b = self.network.param['b'].eval()
        lin = data.dot(W.T)+b

        for oi in xrange(self.ndim_out):
            for di in xrange(self.ndata):
                    grad_data_real[oi, di] = 2*W[oi]*lin[di, oi]
        assert np.allclose(grad_data, grad_data_real)

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
        sec_data_real = np.empty((self.ndim_out, self.ndata) + self.ndim_in)
        W = self.network.param['W'].eval()
        b = self.network.param['b'].eval()
        lin = data.dot(W.T)+b

        for oi in xrange(self.ndim_out):
            sec_data_real[oi] = 2*W[oi]**2
        
        assert np.allclose(sec_data_real, sec_data)

# @unittest.skip('does not work yet')
class test_LinearReLUNetwork(unittest.TestCase):
    
    ndim_in = (10,)
    ndim_out = 5
    ndata  = 1000
    batch_size = 5

    def setUp(self):
        
        # fake data and points
        self.data =   np.random.randn(self.ndata, *self.ndim_in).astype('float32')
        self.data_tensor = tf.constant(self.data)
        
        # build network
        self.network = LinearReLUNetwork(self.ndim_in, self.ndim_out, 
                                     batch_size = self.batch_size, grads=[0.05,1.0])

        self.sess = tf.InteractiveSession()

        init = tf.global_variables_initializer()

        self.sess.run(init)


    def test_forward_array(self):
        out = self.network.forward_array(self.data)
        W = self.network.param['W'].eval()
        b = self.network.param['b'].eval()
        data = self.data
        out_real = b + data.dot(W.T)
        out_real = ((out_real>0)*self.network.grads[1] + (out_real<=0)*self.network.grads[0])*out_real
        assert np.allclose(out, out_real, atol=1e-4), np.max(np.sqrt((out-out_real)**2))

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

        grad_data_real = np.empty((self.ndim_out, self.ndata) + self.ndim_in)
        W = self.network.param['W'].eval()
        for oi in xrange(self.ndim_out):
            for di in xrange(self.ndata):
                grad_data_real[oi, di] = W[oi] * ( (output_true[di, oi]>0 )*self.network.grads[1] +\
                                                   (output_true[di, oi]<=0)*self.network.grads[0]) 
        assert np.allclose(grad_data, grad_data_real)
    
    @unittest.skip('not required')
    def test_get_sec_data(self):

        data, single   = self.network.reshape_data_array(self.data)
        ninput = data.shape[0]

        # number of batches
        batch_size = self.network.batch_size

        sec, _, _, feed = self.network.get_sec_data()
        batch_data = data[0:batch_size]
        results = self.sess.run(sec, feed_dict={feed: batch_data})
        assert results.prod() == 0
        assert results.shape == (self.ndim_out, self.batch_size,) + self.ndim_in

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
    
    ndim_in  = (1,4,4)
    nfil     = 2
    filter_size = 2
    stride   = 2
    ndata  = 4
    batch_size = 2

    def setUp(self):
        
        # fake data and points
        self.data =   np.random.randn(self.ndata, *self.ndim_in).astype('float32')
        self.data_tensor = tf.constant(self.data)
        self.size_out = (self.ndim_in[-1] - self.filter_size)/self.stride + 1
        self.ndim_out = self.size_out ** 2 * self.nfil
        
        # build network
        self.network = ConvNetwork(self.ndim_in, self.nfil, self.filter_size, self.stride, self.batch_size)
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
        output = output.reshape(self.ndata, -1)
        output += b
        output_true = np.maximum(0, output)
        output = self.network.forward_array(self.data)

        assert np.allclose(output,  output_true, atol=1e-5), np.abs(output-output_true).max()

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
        t0 = time.time()
        for bi in xrange(nbatch):
            batch_data = data[bi*batch_size:(bi+1)*batch_size]
            results.append(self.sess.run(grad, feed_dict={feed: batch_data}))
        grad_data = np.concatenate(results, axis=1)
        print 'computing gradient took %.3f' % (time.time() - t0)
        assert grad_data.shape == (self.ndim_out, self.ndata, ) + self.ndim_in

    @unittest.skip('derivatie w.r.t param not needed')
    def test_get_grad_param(self):

        W = self.network.param['W'].eval()
        b = self.network.param['b'].eval()
        data = self.data

        grad_dict, output = self.network.get_grad_param(data)
        assert grad_dict['W'].shape == (self.ndim_out, self.ndata, 
                                        self.filter_size, self.filter_size, 
                                        self.ndim_in[0], self.nfil)

    @unittest.skip('derivatie w.r.t param not needed')
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


# @unittest.skip('')
class test_KernelNetModel(unittest.TestCase):

    ndata  = 10
    ndim_in = (6,)
    nfil = 5
    ndim    = 5
    npoint = 3

    def setUp(self):
        

        # fake data and points
        self.data =   np.random.randn(self.ndata, *self.ndim_in).astype('float32')
        self.data_tensor    =   tf.constant(self.data)
        self.points = np.random.randn(self.npoint, *self.ndim_in).astype('float32')
        self.points_tensor  = tf.constant(self.points)
        
        # build network
        network = LinearNetwork(self.ndim_in, self.ndim, batch_size = self.ndata)
        # network = ConvNetwork(self.ndim_in, self.nfil, self.ndim_in[-1],
        #                            batch_size = self.ndata)

        kernel = GaussianKernel(network.ndim_out)

        # setup lite model with fake alpha and points
        alpha = tf.Variable( np.random.randn(self.npoint).astype('float32') )
        self.model = KernelNetModel(kernel, network, alpha)
        self.model.set_points( self.points_tensor )
        self.X = self.model.X
        
        # compute and store the features 
        self.Y = self.model.network.forward_tensor(self.data_tensor)
        init = tf.global_variables_initializer()

        self.sess = tf.InteractiveSession()
        self.sess.run(init)

    def test_original_score(self):

        score = self.model.kernel_score(self.data_tensor)
        score_o = self.model.score_original(self.data_tensor)
        assert np.allclose(score.eval(), score_o.eval())

    def test_kernel_score(self):

        # fake coefficients
        score = self.model.kernel_score(self.data_tensor)

        alpha = self.model.alpha.eval()
        X = self.X.eval()
        Y = self.Y.eval()
        true_score = 0.0
        sigma = self.model.kernel.sigma
        for l in range(self.ndim):
            for i in range(self.npoint):
                for j in range(self.ndata):
                    true_score += alpha[i]*np.exp(-np.linalg.norm(X[i]-Y[j])**2/sigma/2.0) * (-1 + 1/sigma * (X[i,l]-Y[j,l])**2)
        
        for l in range(self.ndim):
            for i in range(self.ndata):
                sq = 0
                for j in range(self.npoint):
                    sq += alpha[j]*(X[j,l]-Y[i,l]) * np.exp(-np.linalg.norm(X[j]-Y[i])**2/sigma/2.0) 
                true_score += sq**2.0/(sigma)/2.0
        true_score *=  1.0 / self.ndata/sigma
        assert np.allclose(score.eval(), true_score), (score.eval, true_score)

    def test_kernel_fit(self):
        
        scores = []
        for i in range(10):
            self.model.alpha = tf.assign(self.model.alpha, np.random.randn(self.npoint))
            scores.append( self.model.kernel_score(self.data_tensor).eval() )

        self.model.kernel_fit(self.data_tensor, lam=0.0)
        score_opt = self.model.kernel_score(self.data_tensor).eval()

        assert np.all(score_opt<=scores), (score_opt, min(scores))

    def test_evaluate_kernel_fun(self):
        
        f = self.model.evaluate_kernel_fun(self.Y).eval()

        Y_real = self.model.network.forward_array(self.data)
        K_real = np.empty((self.npoint, self.ndata))
        X_real = self.model.X.eval()
        sigma  = self.model.kernel.sigma

        for xi in range(self.npoint):
            for yi in range(self.ndata):
                K_real[xi, yi] = np.exp(-np.sum((X_real[xi]-Y_real[yi])**2)/2.0/sigma) 
        f_real = self.model.alpha.eval().dot(K_real)

        assert np.allclose(f_real, f), np.linalg.norm(f_real - f)

    @unittest.skip('')
    def test_get_kernel_fun_grad(self):
    
        grad, _ = self.model.get_kernel_fun_grad(self.Y)
        grad = grad.eval()
        values = self.model.evaluate_kernel_fun(self.Y)
        grad_real = np.empty(self.Y.shape)
        
        for vi in range(values.shape[0]):
           grad_real[vi] = tf.stack(tf.gradients(values[vi], self.Y)[0][vi]).eval()
           
        assert np.allclose(grad, grad_real)


    
    @unittest.skip('')
    def test_get_kernel_fun_hess(self):
        
        hess, _ = self.model.get_kernel_fun_hess(self.Y)
        hess = hess.eval()
        ndim = self.model.ndim
        ninput = self.Y.shape[0]

        # hessian is of shape (ninput x ndim x ndim)
        hess_real = np.empty( (ninput, ndim, ndim) )
        for vi in range(ninput):
            this_y = self.Y[vi]
            values = self.model.evaluate_kernel_fun(this_y)
            hess_real[vi] = tf.hessians(values[0], this_y)[0].eval()
        assert np.allclose(hess, hess_real)

    def test_score(self):

        score, points, data = self.model.score()
        feed_dict = {points : self.points,
                     data   : self.data}
        score = self.sess.run(score, feed_dict=feed_dict)
        
        score_real = 0.0

        this_data_flat = tf.placeholder('float32', np.prod(self.ndim_in))
        this_data      = tf.reshape(this_data_flat, (1,)+self.ndim_in)
        this_y    = self.model.network.forward_tensor(this_data)
        f = self.model.evaluate_kernel_fun(this_y)
        one_data_score = tf.reduce_sum(tf.diag_part(tf.hessians(f, this_data_flat)[0])) + \
                         0.5 * tf.reduce_sum(tf.gradients(f, this_data_flat)[0])**2
        for yi in range(self.ndata):
            score_real += one_data_score.eval({points:self.points,
                                  data:  self.data,
                                  this_data: self.data[yi:yi+1]})
        np.allclose(score, score_real)
        

    @unittest.skip('')
    def test_get_opt_alpha(self):
        alpha, opt_score, points, data = self.model.get_opt_alpha()
        alpha_value, opt_score_value = \
                    self.sess.run([alpha, opt_score], feed_dict={points: self.points, data: self.data})
        print 'alpha, opt_score: ', opt_score_value

unittest.main()
