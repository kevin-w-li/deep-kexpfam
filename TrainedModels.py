import numpy as np
import tensorflow as tf
from LiteNet import *


class TrainedDeepLite(object):

    def __init__(self, fn, rand_train_data, config=None):
        

        if config is None:
            config = tf.ConfigProto(device_count={"GPU":0})
            config.gpu_options.allow_growth=True

        tf.reset_default_graph() 
        param_str = fn.split("_")
        self.D = int(param_str[1][1:])
        self.nlayer = int(param_str[2][1:])
        self.ndim   = (int(param_str[3][2:]),)
        self.npoint = int(param_str[4][2:])

        self.train_data = tf.placeholder(FDTYPE, shape=(self.npoint, self.D), name="train_data")
        self.test_data  = tf.placeholder(FDTYPE, shape=(None, self.D), name="test_data")
        self.rand_train_data = rand_train_data[:self.npoint]

        points  = tf.identity(self.train_data)
        self.alpha   = tf.Variable(np.zeros(self.npoint), dtype=FDTYPE, name="alpha")

        kernel  = GaussianKernel(1.0)
        if self.nlayer>0:

            layer_1 = LinearSoftNetwork((self.D,), self.ndim, 
                                        init_std=1, scope="fc1")
            layers = [LinearSoftNetwork(self.ndim, self.ndim, 
                                        init_std=1,  scope="fc"+str(i+2)) for i in range(self.nlayer-2)]
            network = DeepNetwork([layer_1, ] + layers, ndim_out = self.ndim, init_std = 1, add_skip=self.nlayer>1)
            kernel = CompositeKernel(kernel, network)
            prop = 1/(1+tf.exp(-tf.Variable(0, dtype=FDTYPE)))
            kernel = MixtureKernel([kernel, GaussianKernel(0)], [1.0-prop, prop])

        self.kn = LiteModel(kernel, points=points, init_log_lam=0, log_lam_weights=-3, alpha=self.alpha)
        self.kn.npoint = self.npoint
        
        self.alpha_assign_opt, self.train_score = self.kn.opt_score(data=self.train_data)

        self.hv, self.gv, self.fv = self.kn.evaluate_hess_grad_fun(self.test_data)
        
        self.sess = tf.Session(config=config)

        ckpt = "ckpts/"+fn+".ckpt"

        saver = tf.train.Saver()
        saver.restore(self.sess, ckpt)
        self.sess.run(self.alpha_assign_opt, feed_dict={self.train_data:self.rand_train_data})
        
    
    def __enter__(self):
        return self
    
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.sess.close()
    
    def grad(self, data, batch_size=100):
        
        neval = data.shape[0]
        nbatch = neval/batch_size
        value = np.empty((neval, self.D))

        for i in range(nbatch):
            batch = data[i*batch_size:(i+1)*batch_size]
            value[i*batch_size:(i+1)*batch_size,:] = \
                self.sess.run(self.gv, feed_dict={self.test_data:batch, self.train_data: self.rand_train_data})   
        return value
        
    def fun(self, data, batch_size=100):
        
        neval = data.shape[0]
        nbatch = neval/batch_size
        value = np.empty((neval))
        
        for i in range(nbatch):

            batch = data[i*batch_size:(i+1)*batch_size]
            value[i*batch_size:(i+1)*batch_size] = \
                self.sess.run(self.fv, feed_dict={self.test_data:batch, self.train_data: self.rand_train_data}) 
        return value
        
    def hess(self, data, batch_size=100):
        
        neval = data.shape[0]
        nbatch = neval/batch_size
        value = np.empty((neval, self.D, self.D))

        for i in range(nbatch):
            batch = data[i*batch_size:(i+1)*batch_size]
            value[i*batch_size:(i+1)*batch_size] = \
                self.sess.run(self.hv, feed_dict={self.test_data:batch, self.train_data: self.rand_train_data}) 
        return value
    
    def retrain(self, rand_train_data):
        
        self.sess.run(self.alpha_assign_opt, feed_dict={self.train_data:rand_train_data})

    
