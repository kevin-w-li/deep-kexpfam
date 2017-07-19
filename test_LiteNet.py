import tensorflow as tf
from LiteNet import *
import unittest
import numpy as np

class test_LiteNet(unittest.TestCase):


    def setUp(self):
        self.kernel = GaussianKernel()
        self.X = tf.constant(np.random.randn(3,4).astype('float32'))
        self.Y = tf.constant(np.random.randn(2,4).astype('float32'))
        self.alpha = tf.constant(np.random.randn(3).astype('float32'))
        self.kernel.set_points(self.X, alpha = self.alpha)

        init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(init)

    def test_get_gram_matrix(self):
        result = self.kernel.get_gram_matrix(self.Y)
        a = self.X.eval()
        x = self.X.eval()
        y = self.Y.eval()
        res = result.eval()

        true_res = np.array([[np.linalg.norm(x[i]-y[j])**2 for j in range(y.shape[0]) ]for i in range(x.shape[0])])
        true_res = np.exp(-0.5/self.kernel.sigma*true_res)
        assert np.allclose(res, true_res)

    def test_evaluate_fun(self):
        
        self.kernel.set_pdist2(self.Y)
        value = self.kernel.evaluate_fun(self.Y).eval()
        gram = self.kernel.get_gram_matrix().eval()
        alpha = self.alpha.eval()
        assert value.shape[0] == self.Y.shape[0]
        assert np.allclose(value, alpha.dot(gram))
        

    def test_get_grad(self):

        grad = self.kernel.get_grad(self.Y).eval()
        values = self.kernel.evaluate_fun(self.Y)
        grad_real = np.empty(self.Y.shape)
        
        for vi in range(values.shape[0]):
           grad_real[vi] = tf.stack(tf.gradients(values[vi], self.Y)[0][vi]).eval()
           
        assert np.allclose(grad, grad_real)

    def test_get_hess(self):
        
        hess = self.kernel.get_hess(self.Y).eval()
        ndim = self.kernel.ndim
        ninput = self.Y.shape[0]

        # hessian is of shape (ninput x ndim x ndim)
        hess_real = np.empty( (ninput, ndim, ndim) )
        for vi in range(ninput):
            this_y = self.Y[vi]
            values = self.kernel.evaluate_fun(this_y)
            hess_real[vi] = tf.stack(tf.hessians(values[0], this_y)[0]).eval()
        assert np.allclose(hess, hess_real)

class test_LinearNetwork(unittest.TestCase):
    
    ndim_in = (5,)
    ndim_out = 4
    ndata  = 6

    def setUp(self):
        
        # fake data and points
        self.data =   np.random.randn(self.ndata, *self.ndim_in).astype('float32')
        self.data_tensor = tf.constant(self.data)
        
        # build network
        self.network = LinearNetwork(self.ndim_in, self.ndim_out)

        self.sess = tf.InteractiveSession()

        init = tf.global_variables_initializer()

        self.sess.run(init)


    def test_forward_array(self):
        out = self.network.forward_array(self.data)
        W = self.network.param['W'].eval()
        b = self.network.param['b'].eval()
        data = self.data
        out_real = b + data.dot(W.T)
        assert np.allclose(out, out_real)

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

class test_kernel_score(unittest.TestCase):

    ndata  = 2
    ndim_in = (5,)
    ndim    = 4
    npoint = 3

    def setUp(self):
        
        kernel = GaussianKernel()

        # fake data and points
        self.data =   np.random.randn(self.ndata, *self.ndim_in).astype('float32')
        self.data_tensor =   tf.constant(self.data)
        self.points = tf.constant(np.random.randn(self.npoint, *self.ndim_in).astype('float32'))
        
        # build network
        network = LinearNetwork(self.ndim_in, self.ndim)

        # setup lite model with fake alpha and points
        self.model = KernelNetModel(kernel, network)
        self.model.kernel.alpha = tf.Variable( np.random.randn(self.npoint).astype('float32') )
        self.model.set_points(self.points)
        
        # compute and store the features 
        self.Y = self.model.network.forward_array(self.data)
        self.X = self.model.kernel.X
        self.model.X = self.model.kernel.X
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

        alpha = self.model.kernel.alpha.eval()
        X = self.X.eval()
        Y = self.Y
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
        assert np.allclose(score.eval(), true_score)

    def test_kernel_fit(self):
        
        scores = []
        for i in range(10):
            self.model.kernel.alpha = tf.assign(self.model.kernel.alpha, np.random.randn(self.npoint))
            scores.append( self.model.kernel_score(self.data_tensor).eval() )

        self.model.kernel_fit(self.data_tensor, lam=0.0)
        score_opt = self.model.kernel_score(self.data_tensor).eval()

        assert np.all(score_opt<=scores)


unittest.main()
