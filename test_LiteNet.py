import tensorflow as tf
from LiteNet import *
import unittest
import numpy as np


class test_LiteNet(unittest.TestCase):


    def setUp(self):
        self.kernel = ExponentialKernel()
        self.X = tf.constant(np.random.randn(3,4))
        self.Y = tf.constant(np.random.randn(2,4))
        self.kernel.set_points(self.X)

    def test_gram_matrix(self):
        result = self.kernel.gram_matrix(self.Y)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            x = self.X.eval()
            y = self.Y.eval()
            res = result.eval()

            true_res = np.array([[np.linalg.norm(x[i]-y[j])**2 for j in range(y.shape[0]) ]for i in range(x.shape[0])])
            true_res = np.exp(-1.0/self.kernel.sigma*true_res)
            assert np.allclose(res, true_res)


class test_score(unittest.TestCase):

    ndata  = 2
    ndim_in = 5
    ndim    = 5
    npoint = 3

    def setUp(self):
        
        kernel = ExponentialKernel()

        # fake data and points
        self.data =   tf.constant(np.random.randn(self.ndata, self.ndim_in).astype('float32'))
        self.points = tf.constant(np.random.randn(self.npoint, self.ndim_in).astype('float32'))
        
        # build network
        network = Network(self.ndata, self.ndim_in, self.ndim)

        # setup lite model with fake alpha and points
        self.model = LiteModel(kernel, network)
        self.model.kernel.alpha = tf.Variable( np.random.randn(self.npoint).astype('float32') )
        self.model.set_points(self.points)
        
        # compute and store the features 
        self.Y = self.model.network.forward(self.data)
        self.X = self.model.kernel.X
        self.model.X = self.model.kernel.X
        init = tf.global_variables_initializer()

        self.sess = tf.InteractiveSession()
        self.sess.run(init)

    def test_score(self):

        # fake coefficients
        score = self.model.score(self.data)

        alpha = self.model.kernel.alpha.eval()
        X = self.X.eval()
        Y = self.Y.eval()
        true_score = 0.0
        sigma = self.model.kernel.sigma
        for l in range(self.ndim):
            for i in range(self.npoint):
                for j in range(self.ndata):
                    true_score += alpha[i]*np.exp(-np.linalg.norm(X[i]-Y[j])**2/sigma) * (-1 + 2/sigma * (X[i,l]-Y[j,l])**2)
        
        for l in range(self.ndim):
            for i in range(self.ndata):
                sq = 0
                for j in range(self.npoint):
                    sq += alpha[j]*(X[j,l]-Y[i,l]) * np.exp(-np.linalg.norm(X[j]-Y[i])**2/sigma) 
                true_score += sq**2.0/sigma
        true_score *=  2 / self.ndata/sigma
        assert np.allclose(score.eval(), true_score)

    def test_lite_fit(self):
        
        scores = []
        for i in range(10):
            self.model.kernel.alpha = tf.assign(self.model.kernel.alpha, np.random.randn(self.npoint))
            scores.append( self.model.score(self.data).eval() )

        self.model.lite_fit(self.data, lam=0.0)
        score_opt = self.model.score(self.data).eval()

        assert np.all(score_opt<=scores)

unittest.main()
