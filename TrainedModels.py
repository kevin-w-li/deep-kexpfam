import numpy as np
import tensorflow as tf
from LiteNet import *
from Utils import support_1d
from kernel_hmc.mini_mcmc.mini_mcmc import mini_mcmc
from kernel_hmc.densities.gaussian import IsotropicZeroMeanGaussian
from kernel_hmc.proposals.kmc import KMCStatic



# restore variables while ignore shape constraints
# https://github.com/tensorflow/tensorflow/issues/312
# RalphMao
def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)


class TrainedDeepLite(object):

    def __init__(self, fn, rand_train_data, config=None, ndims = None, points_type="train", npoint=None):
        

        if config is None:
            config = tf.ConfigProto(device_count={"GPU":0})
            config.gpu_options.allow_growth=True

        tf.reset_default_graph() 
        param_str = fn.split("_")
        self.D = int(param_str[1][1:])
        self.nlayer = int(param_str[2][1:])
        if ndims is None:
            self.ndims   = [(int(param_str[3][2:]),)] * self.nlayer
        else:
            self.ndims   = ndims

        if npoint is None:
            self.npoint = int(param_str[4][2:])
            assert points_type in ["train", "opt"]
        else:   
            self.npoint = npoint

        self.train_data = tf.placeholder(FDTYPE, shape=(None, self.D), name="train_data")
        self.test_data  = tf.placeholder(FDTYPE, shape=(None, self.D), name="test_data")
        
        if points_type == "train":
            self.points  = tf.identity(self.train_data, name="points")
            self.rand_train_data = rand_train_data[:self.npoint]
            self.rand_points     = self.rand_train_data
        elif points_type == "opt":
            self.rand_train_data = rand_train_data
            self.points  = tf.Variable(np.zeros((self.npoint, self.D)), dtype=FDTYPE, name="points")
        elif points_type == "sep":
            self.rand_train_data = rand_train_data
            randint = np.random.randint(rand_train_data.shape[0], size=self.npoint)
            self.rand_points  = rand_train_data[randint,:]
            self.points  = tf.constant(self.rand_points, dtype=FDTYPE)
        else:
            raise NameError("points_type needs to be 'train', 'opt', or 'sep'")
            
        
        if points_type == "opt":
            self.alpha   = tf.Variable(tf.zeros(self.npoint), dtype=FDTYPE, name="alpha")
        else:
            self.alpha   = tf.Variable(tf.zeros(self.npoint), dtype=FDTYPE, name="pv")

        kernel  = GaussianKernel(sigma=0)
        if self.nlayer>0:

            layer_1 = LinearSoftNetwork((self.D,), self.ndims[0], 
                                        init_std=1, scope="fc1")
            layers = [LinearSoftNetwork(self.ndims[i], self.ndims[i+1], 
                                        init_std=1,  scope="fc"+str(i+2)) for i in range(self.nlayer-2)]
            network = DeepNetwork([layer_1, ] + layers, ndim_out = self.ndims[-1], init_std = 1, add_skip=self.nlayer>1)
            kernel = CompositeKernel(kernel, network)
            prop = 1/(1+tf.exp(-tf.Variable(0, dtype=FDTYPE)))
            kernel = MixtureKernel([kernel, GaussianKernel(sigma=0)], [1.0-prop, prop])

        self.kn = LiteModel(kernel, points=self.points, init_log_lam=0, log_lam_weights=-3, alpha=self.alpha)

        self.kn.npoint = self.npoint
        
        if points_type in ["train", "sep"]:
            self.alpha_assign_opt, self.train_score = self.kn.opt_score(data=self.train_data)[:2]

        self.hv, self.gv, self.fv = self.kn.evaluate_hess_grad_fun(self.test_data)
        
        self.sess = tf.Session(config=config)
        self.kmc = None

        ckpt = "ckpts/"+fn+".ckpt"

        #saver = tf.train.Saver(allow_empty=True)
        #saver.restore(self.sess, ckpt)

        optimistic_restore(self.sess, ckpt)
        

        if points_type == "train":
            self.sess.run(self.alpha_assign_opt, feed_dict={self.train_data:self.rand_train_data})
        elif points_type == "sep":
            self.sess.run(self.alpha_assign_opt, feed_dict={self.points     : self.rand_points,
                                                            self.train_data : self.rand_train_data})
        elif points_type == "opt":
            self.sess.run(self.alpha_assign_opt, feed_dict={self.train_data:self.rand_train_data})
            self.points = self.sess.run(self.points)
            
        self.min_log_pdf = -np.inf
        
    
    def __enter__(self):
        return self
    
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.sess.close()

    def grad(self, data):
        return support_1d(lambda x: self.grad_multiple(x, batch_size=1), data)

    def log_pdf(self, data):
        return support_1d(lambda x: self.fun_multiple(x, batch_size=1), data)
    
    def grad_multiple(self, data, batch_size=100):
        
        neval = data.shape[0]
        nbatch = neval/batch_size
        value = np.empty((neval, self.D))

        for i in range(nbatch):
            batch = data[i*batch_size:(i+1)*batch_size]
            value[i*batch_size:(i+1)*batch_size,:] = \
                self.sess.run(self.gv, feed_dict={self.test_data:batch, self.train_data: self.rand_train_data})   
        return value
        
    def fun_multiple(self, data, batch_size=100):
        
        neval = data.shape[0]
        nbatch = neval/batch_size
        value = np.empty((neval))
        
        for i in range(nbatch):

            batch = data[i*batch_size:(i+1)*batch_size]
            value[i*batch_size:(i+1)*batch_size] = \
                self.sess.run(self.fv, feed_dict={self.test_data:batch, self.train_data: self.rand_train_data}) 
        value[value<self.min_log_pdf] = -np.inf
        return value
        
    def hess_multiple(self, data, batch_size=100):
        
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

    def setup_mcmc(self, sigma=1.0, num_steps_min=1, num_steps_max=10, step_size_min=0.01, step_size_max=0.1,
                    min_log_pdf=0.0):
        
        momentum = IsotropicZeroMeanGaussian(sigma=sigma, D=self.D)
        self.kmc = KMCStatic(self, momentum, num_steps_min, num_steps_max, 
                            step_size_min, step_size_max)
        self.min_log_pdf = min_log_pdf

    def sample(self, N, start):
        assert self.kmc is not None
        samples, proposals, accepted, acc_prob, log_pdf, times, step_sizes = mini_mcmc(self.kmc, start, N, self.D)
        return samples

