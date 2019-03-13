import numpy as np
import tensorflow as tf
from DKEF import *
from Utils import support_1d

from tqdm import tqdm_notebook, tqdm
from collections import OrderedDict
import warnings
from scipy.misc import logsumexp
from sklearn.cluster import KMeans
from tensorflow.contrib.opt import ScipyOptimizerInterface

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from time import time

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


class DeepLiteMixture(object):
    
    def __init__(self, targets, **kwargs):
        self.targets = targets
        self.n_clusters = targets.n_clusters
        self.lites = []
        self.props = targets.props

        for i in range(self.n_clusters):
            dl = DeepLite(targets.ps[i], fn_ext = "mc%d"%(i), **kwargs)
            self.lites.append(dl)

    def fit(self, **kwargs):
        for i in range(self.n_clusters):
            self.lites[i].fit(**kwargs)

    def fit_alpha(self, **kwargs):
        for i in range(self.n_clusters):
            self.lites[i].fit_alpha(**kwargs)

    def fit_kernel(self, **kwargs):
        for i in range(self.n_clusters):
            self.lites[i].fit_kernel(**kwargs)

    def set_test(self, *args):
        for i in range(self.n_clusters):
            self.lites[i].set_test(*args)

    def set_train(self, *args):
        for i in range(self.n_clusters):
            self.lites[i].set_train(*args)
        
    def estimate_normaliser(self, **kwargs):
        for i in range(self.n_clusters):
            self.lites[i].estimate_normaliser(**kwargs)

    def eval(self, data, **kwargs):
        ll = np.zeros((data.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            l = self.lites[i]
            d = self.targets.ps[i].trans(data) 
            ll[:,i] = l.eval(d, bar=False, **kwargs) + np.linalg.slogdet(self.targets.ps[i].W)[1]
        loglik = logsumexp(ll, 1, b=self.props)
        return loglik

    def save(self):
        for i in range(self.n_clusters):
            l = self.lites[i].save()
        return self.lites[0].default_file_name()

    def load(self):
        for i in range(self.n_clusters):
            self.lites[i].load()

    def default_file_name(self):
        return self.lites[0].default_file_name()

    def eval_grad(self, data, **kwargs):
        
        ll = np.zeros((data.shape[0], self.n_clusters))
        g  = np.zeros((data.shape)+(self.n_clusters,))

        for i in range(self.n_clusters):
            l = self.lites[i]
            d = self.targets.ps[i].trans(data) 
            ll[:,i] = l.eval(d, bar=False, **kwargs) + np.linalg.slogdet(self.targets.ps[i].W)[1]
            g[...,i]  = l.grad(d).dot((self.targets.ps[i].W.T))

        loglik = logsumexp(ll, 1, b=self.props)
        
        jll = ll + np.log(self.props)
        jll -= np.max(jll, -1, keepdims=True)
        
        w = np.exp(jll)
        w = w / w.sum(-1, keepdims=True)
        g = np.einsum('ik,ijk->ij', w, g)

        return loglik, g

    def grad(self, data, **kwargs):
        return self.eval_grad(data, **kwargs)[1]

class DeepLite(object):

    def __init__(self, target, nlayer=3, nneuron=30,
                    noise_std=0.0, points_std=0.0,
                    init_weight_std=1.0, init_log_sigma=[0.0], init_log_lam=-2.0, log_lam_weights=-6,
                    seed=None, keep_prob = 1.0, mixture_kernel=False, base=True,
                    npoint=300, ntrain=100, nvalid=100, nbatch=1,points_type="opt", clip_score=False,
                    step_size=1e-3, niter=10000, patience=200, kernel_type="gaussian",
                    final_step_size = 1e-3, final_ntrain=200, final_nvalid=200, final_niter=1000,
                    gpu_count=1, cpu_count=None, fn_ext = "", train_stage = 0, nl_type = "linear", add_skip=True, curve_penalty=True,
                    ):
        
        self.target = target

        
        self.fn_ext = fn_ext
        self.seed = seed
        self.train_stage = train_stage
        
        self.model_params = dict( nlayer          = nlayer, 
                                    nneuron        = nneuron, 
                                    noise_std      = noise_std,
                                    points_std     = points_std,
                                    init_weight_std= init_weight_std,
                                    init_log_sigma = init_log_sigma,
                                    init_log_lam   = init_log_lam,
                                    log_lam_weights= log_lam_weights,
                                    ndims = [(nneuron,)] * nlayer,
                                    _keep_prob = keep_prob,
                                    npoint = npoint,
                                    mixture_kernel = mixture_kernel,
                                    base           = base,
                                    kernel_type    = kernel_type,
                                    nl_type       = nl_type,
                                    add_skip       = add_skip
                                )
        if nlayer == 0:
            self.model_params["ndims"] = [0,]
        self.train_params   = dict( step_size = step_size,
                                    niter = niter,
                                    ntrain = ntrain,
                                    nvalid = nvalid,
                                    patience=patience,
                                    points_type = points_type,
                                    clip_score = clip_score,
                                    nbatch = nbatch,
                                    curve_penalty = curve_penalty,
                                    final_step_size = final_step_size,
                                    final_ntrain    = final_ntrain,
                                    final_nvalid    = final_nvalid,
                                    final_niter     = final_niter
                                    )
        
        self.states = OrderedDict()
        self.final_states = OrderedDict()
            
        self.state_hist = OrderedDict()
        self.final_state_hist = OrderedDict()
        
        self.target = target
        
        if niter is not None:
            self.niter = niter
        else:
            self.niter = 0

        self.build_model(gpu_count, cpu_count=cpu_count, train_stage=train_stage)

        self.logZ = None
        
    def build_model(self, gpu_count=1, cpu_count=None, train_stage=0):
        
        self.graph = tf.Graph()
        with self.graph.as_default(): 
            self.ops = dict()
            

            np.random.seed(self.seed)
            tf.set_random_seed(self.seed)
            
            target = self.target
            
            nneuron = self.model_params["nneuron"]
            nlayer  = self.model_params["nlayer"]
            ndims   = self.model_params["ndims"]
            kernel_type   = self.model_params["kernel_type"]
            add_skip= self.model_params["add_skip"]
            nl_type= self.model_params["nl_type"]

            mixture_kernel = self.model_params["mixture_kernel"]
            
            keep_prob = tf.Variable(1.0, dtype=FDTYPE, trainable=False, name="keep_prob")
            
            self.ops["set_dropout"] = tf.assign(keep_prob, self.model_params["_keep_prob"])
            self.ops["set_keepall"] = tf.assign(keep_prob, 1.0)
                    
            noise_std = self.model_params["noise_std"]
            points_std= self.model_params["points_std"]

            init_log_sigma= self.model_params["init_log_sigma"]
            init_weight_std = self.model_params["init_weight_std"]

            init_log_lam = self.model_params["init_log_lam"]
            log_lam_weights = self.model_params["log_lam_weights"]
            base    = self.model_params["base"]

            npoint  = self.model_params["npoint"]
            ntrain  = self.train_params["ntrain"]
            nbatch  = self.train_params["nbatch"]

            clip_score  = self.train_params["clip_score"]
            points_type = self.train_params["points_type"]
            curve_penalty = self.train_params["curve_penalty"]

            if self.target.nkde:
                self.train_kde = tf.placeholder(FDTYPE, shape=(None,), name="train_kde")
                self.valid_kde = tf.placeholder(FDTYPE, shape=(None,), name="valid_kde")
            else:
                self.valid_kde = None
                self.train_kde = None
            
            train_data  = tf.placeholder(FDTYPE, shape=(None, target.D), name="train_data")
            if points_type != "feat":
                test_points = tf.placeholder(FDTYPE, shape=(None, target.D), name="test_points")

            if points_type == "fixed":
                points  = tf.Variable(target.sample(npoint) + np.random.randn(npoint,target.D)*points_std, dtype=FDTYPE, name="points", trainable=False)
            elif points_type == "opt":
                    points  = tf.Variable(target.sample(npoint) + np.random.randn(npoint,target.D)*points_std, dtype=FDTYPE, name="points", trainable=True)
            elif points_type == "tied":
                points  = tf.identity(train_data, name="points")
            elif points_type == "kmeans":
                kmeans  = KMeans(n_clusters=npoint, random_state=self.seed).fit(self.target.sample(min(5000, self.target.N)))
                points  = tf.Variable(kmeans.cluster_centers_ + np.random.randn(npoint,target.D)*points_std, dtype=FDTYPE, name="points", trainable=False)
            elif points_type == "feat":
                points  = tf.Variable(np.random.exponential(size=((npoint,) + ndims[-1]))*points_std, dtype=FDTYPE, name="points", trainable=True)
            else:
                raise NameError(points_type + " is not a valid points type")
                
            valid_data  = tf.placeholder(FDTYPE, shape=(None, target.D), name="valid_data")
            test_data = tf.placeholder(FDTYPE, shape=(None, target.D), name="test_data")

            kernels = []
            sigmas  = []
            props   = []
            net_outs = []
            kernel_grams = []
            nkernel = len(init_log_sigma)

            add_skip = nlayer>1 if add_skip else False

            for i in range(len(init_log_sigma)):
                
                if kernel_type=="gaussian":
                    kernel  = GaussianKernel(init_log_sigma[i],   trainable=True)
                    sigma   = kernel.sigma

                elif kernel_type == "linear":
                    kernel  = PolynomialKernel(1.0,0.0)
                    sigma   = tf.constant(0.0, dtype=FDTYPE)

                prop    = tf.exp(-tf.Variable(0.0, dtype=FDTYPE, trainable=nkernel!=1))

                if nlayer>0:

                    layers = []
                    
                    layer = LinearSoftNetwork((target.D,), ndims[0], 
                                                init_weight_std=init_weight_std/np.sqrt(ndims[0][0]), scope="fc1", keep_prob=keep_prob)
                    layers.append(layer)
                    
                    for i in range(nlayer-2):
                        layer = LinearSoftNetwork(ndims[i], ndims[i+1], 
                                                    init_weight_std=init_weight_std/np.sqrt(ndims[i][0]),  scope="fc"+str(i+2), keep_prob=keep_prob)
                        layers.append(layer)

                    network = DeepNetwork(layers, ndim_out = ndims[-1], init_weight_std = init_weight_std/np.sqrt(ndims[-1][0]), add_skip=add_skip, nl_type=nl_type)
                    net_outs.append(network.forward_tensor(test_data))
                    kernel = CompositeKernel(kernel, network, feat_points=points_type=="feat")

                kernels.append(kernel)
                #sigmas.append(sigma)
                props.append(prop)
                if points_type != "feat":
                    kernel_grams.append(kernel.get_gram_matrix(test_points, test_data))

            self.ops["net_outs"]  = net_outs
            self.ops["kernel_grams"] = kernel_grams
            prop_sum = tf.reduce_sum(props)

            for i in range(nkernel-1):
                props[i] = props[i]/prop_sum

            props[-1]  = 1-tf.reduce_sum(props[:-1])

            kernel    = MixtureKernel( kernels, props )
            
            self.alpha   = tf.Variable(tf.zeros(npoint), dtype=FDTYPE, name="alpha_eval", trainable=False)

            kn = LiteModel(kernel, npoint, points=points, alpha=self.alpha,
                            init_log_lam=init_log_lam, log_lam_weights=log_lam_weights, 
                            feat_points=points_type=="feat",
                            noise_std=noise_std, base=base, curve_penalty=curve_penalty, feat_dim=ndims[-1])

            loss, score, _, _, r_norm, l_norm, curve, w_norm, k_loss, test_score, \
                self.states["outlier"], save_alpha = \
                kn.val_score(train_data=train_data, valid_data=valid_data, test_data = test_data, 
                            train_kde=self.train_kde, valid_kde=self.valid_kde, clip_score=clip_score,
                            add_noise=True)


            if train_stage <=0 :
                # kernel learning
                optimizer = tf.train.AdamOptimizer(self.train_params["step_size"])
                raw_gradients, variables = zip(*optimizer.compute_gradients(loss))
                gradients = [ tf.where(tf.is_nan(g), tf.zeros_like(g), g) for g in raw_gradients if g is not None]

                accum_gradients = [tf.Variable(tf.zeros_like(g), trainable=False) for g in gradients]
                self.ops["zero_op"]  = [ag.assign(tf.zeros_like(ag)) for ag in accum_gradients]

                self.ops["accum_op"] = [accum_gradients[i].assign_add(g/nbatch) for i, g in enumerate(gradients)]
                gradients, self.states["grad_norm"] = tf.clip_by_global_norm(accum_gradients, 100.0)
                self.ops["train_step"] = [optimizer.apply_gradients( zip(gradients, variables) ), save_alpha]
            
            if train_stage <= 1:
                
                # regularizer learning for final fit
                G2  = tf.Variable(tf.zeros((npoint, npoint)), dtype=FDTYPE, name="G2", trainable=False)
                H2  = tf.Variable(tf.zeros((npoint, npoint)), dtype=FDTYPE, name="H2", trainable=False)
                
                H  = tf.Variable(tf.zeros((npoint)), dtype=FDTYPE, name="H", trainable=False)
                GqG  = tf.Variable(tf.zeros((npoint)), dtype=FDTYPE, name="GqG", trainable=False)
                HqH  = tf.Variable(tf.zeros((npoint)), dtype=FDTYPE, name="HqH", trainable=False)
                self.ops["zero_quad_lin"]  = [ag.assign(tf.zeros_like(ag)) for ag in [G2, H2, H, GqG, HqH]]
                 
                _, b_H, b_G2, b_H2, b_GqG, _, _, b_HqH, _, _  = kn.opt_alpha(data=train_data, add_noise=True)

                nt = tf.cast(tf.shape(train_data)[0], FDTYPE)
                n_acc_quad_lin= tf.placeholder(FDTYPE, shape=(), name="n_acc")
                acc_G2  = tf.assign_add(G2, b_G2 * nt / n_acc_quad_lin)
                acc_H   = tf.assign_add(H, b_H * nt / n_acc_quad_lin)
                acc_GqG = tf.assign_add(GqG, b_GqG * nt / n_acc_quad_lin)
                acc_H2  = tf.assign_add(H2, b_H2 * nt / n_acc_quad_lin)
                acc_HqH = tf.assign_add(HqH, b_HqH * nt / n_acc_quad_lin)
                self.ops["acc_quad_lin"] = [acc_G2, acc_H2, acc_H, acc_GqG, acc_HqH]
                self.train_params["n_acc_quad_lin"] = n_acc_quad_lin
                
                quad = (G2 + 
                        H2 * kn.lam_curve + 
                        tf.eye(npoint, dtype=FDTYPE)*kn.lam_alpha
                        )
                lin  =  -(
                        HqH * kn.lam_curve +
                        H + GqG
                        )
                
                alpha = tf.matrix_solve(quad, lin[:,None])[:,0]
                save_alpha = tf.assign(self.alpha,alpha)

                lambdas = [v for v in tf.trainable_variables() if "regularizers" in v.name or "Base" in v.name]
                
                final_score = kn.score(valid_data, alpha = alpha, add_noise=True)[0]
                final_optimizer = tf.train.AdamOptimizer(self.train_params["final_step_size"])
                raw_gradients, variables = zip(*final_optimizer.compute_gradients(final_score, var_list = lambdas))
                gradients, self.final_states["grad_norm"] = tf.clip_by_global_norm(raw_gradients, 100.0)
                self.ops["train_lambdas_B"]  = [final_optimizer.apply_gradients(zip(gradients, variables)), save_alpha]
                self.ops["train_lambdas_CG"] = ScipyOptimizerInterface(final_score, var_list = lambdas,
                                                        method="CG")
               
                self.ops["assign_alpha"]  = [tf.assign(self.alpha, alpha), alpha]

            self.min_log_pdf = -np.inf
            hv, gv, fv = kn.evaluate_hess_grad_fun(test_data, alpha=self.alpha)
            
            # add for logr 
            q_std = tf.placeholder(dtype=FDTYPE, shape=[], name="q_std")
            n_rand = tf.placeholder(dtype="int32", shape=[], name="n_rand")
            le_cutoff = tf.placeholder(dtype=FDTYPE, shape=[], name="le_cutoff")
            self.nodes= dict(q_std=q_std, n_rand=n_rand, le_cutoff=le_cutoff)
            rand_norm = tf.random_normal((n_rand, self.target.D), mean=0.0, stddev=q_std, dtype=FDTYPE)
            rand_fv   = kn.evaluate_fun(rand_norm, alpha = self.alpha)
            logq = (-.5 * self.target.D * tf.log(2 * np.pi * q_std**2)
                         - tf.reduce_sum(rand_norm ** 2, axis=1) / (2 * q_std**2))

            self.nodes["logr"] = rand_fv - logq
            self.nodes["lse_logr"] = tf.reduce_logsumexp(self.nodes["logr"])
            self.nodes["lse_2logr"] = tf.reduce_logsumexp(2 * self.nodes["logr"])
            self.nodes["logr_le"] = tf.count_nonzero(self.nodes["logr"] <= le_cutoff)

            assert len(kn.base.measures) == 1
            q_sample, q_logq = kn.base.measures[0].sample_logq(n_rand)
            q_rand_fv = kn.evaluate_fun(q_sample, alpha=self.alpha)
            q_rand_kfv = kn.evaluate_kernel_fun(q_sample, alpha=self.alpha)
            
            self.nodes["q_sample"] = q_sample
            self.nodes["q_logq"] = q_logq
            self.nodes["q_rand_fv"]= q_rand_fv
            self.nodes["q_rand_kfv"]= q_rand_kfv
            
            self.nodes["q_logr"] = q_rand_fv - q_logq
            self.nodes["q_lse_logr"] = tf.reduce_logsumexp(self.nodes["q_logr"])
            self.nodes["q_lse_2logr"] = tf.reduce_logsumexp(2 * self.nodes["q_logr"])
            self.nodes["q_logr_le"] = tf.count_nonzero(self.nodes["q_logr"] <= le_cutoff)

            self.nodes["q_logr_lowerbound"] = (
                tf.reduce_sum(tf.minimum(self.alpha, 0))
                - kn.base.measures[0].get_log_normaliser())

            sc         = kn.individual_score(test_data, alpha=self.alpha)[0]
            self.ops["hv"] = hv
            self.ops["gv"] = gv
            self.ops["fv"] = fv
            self.ops["sc"] = sc
            
            dc = {"GPU": gpu_count}
            config_args = {'device_count': dc}
            if cpu_count is not None:
                dc['CPU'] = cpu_count
                config_args['intra_op_parallelism_threads'] = cpu_count
                config_args['inter_op_parallelism_threads'] = 1  # think this is right...
            config = tf.ConfigProto(**config_args)
            config.gpu_options.allow_growth=True

            #Visualise the kernel with random initialization

            sess = tf.Session(config=config)
            init = tf.global_variables_initializer()
            sess.run(init)

            self.saver = tf.train.Saver()

            self.sess = sess
            self.kn   = kn

            self.train_data = train_data
            self.valid_data = valid_data    
            self.test_data  = test_data
            self.points     = points

            if points_type != "feat":
                self.test_points= test_points
            
            self.states["score"]   = score
            self.states["loss"]    = loss
            #self.states["sigmas"]   = sigmas
            self.states["props"]    = props
            self.states["r_norm"]  = r_norm
            self.states["l_norm"]  = l_norm
            self.states["curve"]   = curve
            self.states["w_norm"]  = w_norm
            self.states["k_loss"]  = k_loss

            self.states["lam_norm"]     = self.kn.lam_norm
            self.states["lam_curve"]    = self.kn.lam_curve
            self.states["lam_alpha"]    = self.kn.lam_alpha
            self.states["lam_kde"]      = self.kn.lam_kde
            self.states["test_score"]   = test_score

            for k in self.states:
                if k not in self.state_hist:
                    self.state_hist[k] = []

            
            if train_stage<=1:
                self.final_states["score"] = final_score
                self.final_states["lam_norm"]     = self.kn.lam_norm
                self.final_states["lam_curve"]    = self.kn.lam_curve
                self.final_states["lam_alpha"]    = self.kn.lam_alpha
                self.final_states["lam_kde"]      = self.kn.lam_kde

                #self.final_states["sigmas"]   = sigmas
                self.final_states["props"]    = props
                self.final_states["r_norm"]  = r_norm
                self.final_states["l_norm"]  = l_norm
                self.final_states["curve"]   = curve
                self.final_states["w_norm"]  = w_norm
                self.final_states["k_loss"]  = k_loss
            for k in self.final_states:
                if k not in self.final_state_hist:
                    self.final_state_hist[k] = []
        
    def step(self, feed):
        

        nbatch = self.train_params["nbatch"]
        ntrain =self.train_params["ntrain"]
        nvalid =self.train_params["nvalid"]

        self.sess.run(self.ops["zero_op"])
        feed[self.train_data], valid_data, train_kde, valid_kde = self.target.stream_two(ntrain, nvalid*nbatch)
        
        if self.target.nkde:
            feed[self.train_kde] = train_kde

        for i in range(nbatch):
            feed[self.valid_data] = valid_data[i * nvalid : (i+1) * nvalid]

            if self.target.nkde:
                feed[self.valid_kde] = valid_kde[i * nvalid : (i+1) * nvalid]

            self.sess.run(self.ops["accum_op"], feed_dict=feed)

        res = self.sess.run([self.ops["train_step"],  self.states.values()[:-1]], feed_dict=feed)[1]
        
        final_ntrain = self.train_params["final_ntrain"]
        final_nvalid  = self.train_params["final_nvalid"]
        final_niter  = self.train_params["final_niter"]
        n_acc_quad_lin = self.train_params["n_acc_quad_lin"]
        n_acc = min(self.target.N,1000)
        nbatch = int(np.ceil(n_acc* 1.0 / final_ntrain))
        
        self.sess.run(self.ops["zero_quad_lin"])
        train_data_for_test = self.target.sample(nbatch * final_ntrain)
        for i in range(nbatch):
            data = train_data_for_test[i*final_ntrain:(i+1)*final_ntrain]
            self.sess.run(self.ops["acc_quad_lin"], feed_dict={self.train_data:data,
                                                               n_acc_quad_lin:n_acc })
        self.sess.run(self.ops["assign_alpha"])

        ntest_batch = int(np.ceil(self.target.nvalid/200.0))
        test_score = 0
        for i in range(ntest_batch):
            test_data = self.target.valid_data[i*200:(i+1)*200]
            test_score += self.sess.run( self.states["test_score"], 
                            feed_dict={self.test_data:test_data}) * test_data.shape[0] / self.target.nvalid

        res.append(test_score)

        for ki, k in enumerate(self.state_hist.keys()):
            self.state_hist[k].append(res[ki])

    def fit(self, stage = 0, kernel_kwargs={}, alpha_kwargs={}, norm_kwargs={}):

        assert (stage >=0) and (stage < 3)
        kernel_res = None
        alpha_res = None
        logZ = None
        if stage <=0:
            kernel_res = self.fit_kernel(**kernel_kwargs)
        if stage<=1:
            alpha_res  = self.fit_alpha(**alpha_kwargs)
        if stage >= 2:
            self.load()
        #if stage <= 2:
        #    logZ = self.estimate_normaliser(**norm_kwargs)
        return kernel_res, alpha_res, logZ

    def fit_kernel(self, **kwargs):
        
        train_data = self.train_data
        valid_data = self.valid_data
        
        sess = self.sess
        target = self.target

        for k,v in kwargs.items():
            if k in self.train_params:
                self.train_params[k] = v
        
        ntrain = self.train_params["ntrain"]
        nvalid = self.train_params["nvalid"]
        niter  = self.train_params["niter"]
        patience = self.train_params["patience"]
        nbatch = self.train_params["nbatch"]
        
        feed={}
        
        t0 = time()
        last_time = t0
        
        self.set_train()

        last_epoch = 0
        best_score = np.inf
        wait_window = 0
        with tqdm(range(niter+1), ncols=100, desc="trainining kernel", postfix=[dict(loss="%.3f" % 0.0, test="%.3f" % 0.0)]) as tr:    

            for i in tr: 

                res = self.step(feed)

                block_score  = self.state_hist["score"][i-min(i, 30):]
                block_test_score  = self.state_hist["test_score"][i-min(i, 30):]


                block_score_mean = np.mean(block_score)
                block_test_score_mean = np.mean(block_test_score)

                tr.postfix[0]["loss"] = "%.3f" % block_score_mean
                tr.postfix[0]["test"] = "%.3f" % best_score
                tr.update()

                t0 = time()

                epoch = i
                
                cs_mean = np.mean(self.state_hist["score"][-100:])
                if i > 100:
                    cs_std  = np.std(self.state_hist["score"][-100:])
                else:   
                    cs_std   = np.inf
                current_test_score = self.state_hist["test_score"][-1]

                if current_test_score < best_score:
                    best_score = min(best_score, current_test_score)
                    if patience>0:
                        wait_window = 0
                        self.save()
                        found_best = True
                else:
                    found_best = False

                if epoch > last_epoch:
                    last_epoch = epoch
                    if patience>0:
                        if found_best:
                            wait_window = 0
                        else:
                            wait_window += 1
                        
                        if wait_window == patience:
                            break

        if patience>0:
            self.load()
        else:
            self.save()
        print "best score: %.5f" % best_score
        
        return self.state_hist

    def get_logr(self, n, std=2.0):
        
        return self.sess.run(self.nodes["logr"], feed_dict={self.nodes["n_rand"]: n, self.nodes["q_std"]: std})

    def get_q_logr(self, n):
        
        return self.sess.run(self.nodes["q_logr"], feed_dict={self.nodes["n_rand"]: n})
        
    def fit_alpha(self, **kwargs):

        for k,v in kwargs.items():
            if k in self.train_params:
                self.train_params[k] = v
        
        feed = {}
        
        final_ntrain = self.train_params["final_ntrain"]
        final_nvalid  = self.train_params["final_nvalid"]
        final_niter  = self.train_params["final_niter"]
        n_acc_quad_lin = self.train_params["n_acc_quad_lin"]
        nbatch = int(np.ceil(self.target.N * 1.0 / final_ntrain))
        
        self.sess.run(self.ops["zero_quad_lin"])
        for i in tqdm(range(nbatch), ncols=100, desc="accumulating stats"):
            data = self.target.data[i*final_ntrain:(i+1)*final_ntrain]
            self.sess.run(self.ops["acc_quad_lin"], feed_dict={self.train_data:data, 
                                                               n_acc_quad_lin:self.target.N})
    
        pointer = 0
        lam_names = [k for k in self.states.keys() if "lam" in k ]
        lambdas = [ self.states[ln] for ln in lam_names ]

        for i in tqdm(range(final_niter), ncols=100, desc="fitting lambda"):
            pointer += final_nvalid
            if pointer >= self.target.nvalid:
                pointer = 0
                np.random.shuffle(self.target.valid_data)
            
            data = self.target.valid_data[pointer:pointer+final_nvalid]
            feed[self.valid_data] = data
            l = self.sess.run([self.ops["train_lambdas_B"], self.final_states.values()], feed_dict=feed)[1]

            for li, ln in enumerate(self.final_states.keys()):
                self.final_state_hist[ln].append(l[li])
        
        nbatch = int(np.ceil(1.0*self.target.nvalid/final_nvalid))
        s = 0
        for i in range(nbatch):
            d = self.target.valid_data[i*final_nvalid:(i+1)*final_nvalid]
            feed[self.valid_data] = d
            s += self.sess.run(self.final_states["score"], feed_dict=feed) * d.shape[0] / self.target.nvalid
        print "final validation score: %.3f" % s
        self.sess.run(self.ops["assign_alpha"])
        self.save()
        return self.final_state_hist

    def fit_alpha_CG(self, **kwargs):

        for k,v in kwargs.items():
            if k in self.train_params:
                self.train_params[k] = v
        
        feed = {}
        
        final_ntrain = self.train_params["final_ntrain"]
        final_nvalid  = self.train_params["final_nvalid"]
        final_niter  = self.train_params["final_niter"]
        nbatch = int(np.ceil(self.target.N * 1.0 / final_ntrain))

        for i in tqdm(range(nbatch), ncols=100, desc="accumulating stats"):
            data = self.target.data[i*final_ntrain:(i+1)*final_ntrain]
            self.sess.run(self.ops["acc_quad_lin"], feed_dict={self.train_data:data})
    
        self.ops["train_lambdas_CG"].minimize(self.sess, feed_dict={self.valid_data : self.target.valid_data})
        self.sess.run(self.ops["assign_alpha"])

        final_state_vals = self.sess.run(self.final_states.values()[2:])
        nbatch = int(np.ceil(1.0*self.target.nvalid/final_nvalid))
        s = 0
        for i in range(nbatch):
            d = self.target.valid_data[i*final_nvalid:(i+1)*final_nvalid]
            feed[self.valid_data] = d
            s += self.sess.run(self.final_states["score"], feed_dict=feed) * d.shape[0] / self.target.nvalid

        final_state_vals = [0,s]+ final_state_vals
        return dict(zip(self.final_state_hist.keys(), final_state_vals))

    def set_test(self, rebuild=False, gpu_count=None, cpu_count=None, train_stage=0):

        if rebuild:
            assert gpu_count is not None, "specify number of gpu"
            if isinstance(self.seed, int):
                self.save()
                self.build_model(gpu_count, cpu_count=cpu_count, train_stage=train_stage)
                self.load()
            else:   
                self.save("tmp")
                self.build_model(gpu_count, cpu_count=cpu_count, train_stage=train_stage)
                self.load("tmp")
            
        self.sess.run(self.ops["set_keepall"])
        
    def set_train(self, rebuild=False, gpu_count=None, cpu_count=None):
        if rebuild:
            assert gpu_count is not None, "specify number of gpu"
            self.save("tmp")
            self.build_model(gpu_count, cpu_count=cpu_count)
            self.load("tmp")
        self.sess.run(self.ops["set_dropout"])
        
    def default_file_name(self):

        file_name = "%s_D%02d_l%d_nd%d_np%d_nt%d_nv%d_pt%s_ss%d_ni%d_n%02d_k%d_m%d_b%d_p%d_nk%d_cl%d_cu%d" % \
            (self.target.name[0], self.target.D, self.model_params["nlayer"], 
             np.prod(self.model_params["ndims"][0]), 
             self.model_params["npoint"], self.train_params["ntrain"], 
             self.train_params["nvalid"], self.train_params["points_type"][0],
             int(np.around(self.train_params["step_size"]*10000)), 
             self.niter,
             self.model_params["noise_std"]*100,
             self.model_params["_keep_prob"]*10,
             self.model_params["mixture_kernel"],
             self.model_params["base"],
             self.train_params["patience"],
             len(self.model_params["init_log_sigma"]),
             self.train_params["clip_score"],
             self.train_params["curve_penalty"])

        if self.model_params["kernel_type"] == "linear":
            file_name += "_lin"

        if isinstance(self.target.N_prop, float):
            file_name += "_pr%d" % (self.target.N_prop*100)
            
        if len(self.fn_ext)!=0:
            file_name += "_" + self.fn_ext
        
        if isinstance(self.seed, int) :
            file_name += "_s%02d" % self.seed

        return file_name

    def save(self, file_name=None):
        if file_name is None:
            file_name = self.default_file_name()

        save_path = self.saver.save(self.sess, "ckpts/"+file_name+".ckpt")
        return file_name

    def load(self, file_name=None):
            
        if file_name is None:
            file_name = self.default_file_name()
        ckpt = "ckpts/"+file_name+".ckpt"
        with self.graph.as_default():
            optimistic_restore(self.sess, ckpt)

    def grad(self, data):
        return support_1d(lambda x: self.grad_multiple(x, batch_size=1), data)

    def log_pdf(self, data):
        return support_1d(lambda x: self.fun_multiple(x, batch_size=1), data)

    def score_multiple(self, data, batch_size=100):

        neval = data.shape[0]
        nbatch = int(np.ceil(1.0*neval/batch_size))

        value = np.zeros(0)

        for i in range(nbatch):
            batch = data[i*batch_size:(i+1)*batch_size]
            r = self.sess.run(self.ops["sc"], feed_dict={self.test_data:batch})
            value = np.append(value, r)

        return value

    def grad_multiple(self, data, batch_size=100):

        neval = data.shape[0]
        nbatch = int(np.ceil(1.0*neval/batch_size))

        value = np.zeros((0, self.target.D))

        for i in range(nbatch):
            batch = data[i*batch_size:(i+1)*batch_size]
            value = np.concatenate([value, 
                self.sess.run(self.ops["gv"], feed_dict={self.test_data:batch})], axis=0)

        return value

    def fun_multiple(self, data, batch_size=100):

        neval = data.shape[0]
        nbatch = int(np.ceil(1.0*neval/batch_size))
        value = np.zeros(0)

        for i in range(nbatch):

            batch = data[i*batch_size:(i+1)*batch_size]
            value = np.append(value, self.sess.run(self.ops["fv"], feed_dict={self.test_data:batch}) )
        value[value<self.min_log_pdf] = -np.inf
        return value

    def hess_multiple(self, data, batch_size=100):

        neval = data.shape[0]
        nbatch = int(np.ceil(1.0*neval/batch_size))

        value = np.zeros((0, self.target.D, self.target.D))

        for i in range(nbatch):
            batch = data[i*batch_size:(i+1)*batch_size]
            r = self.sess.run(self.ops["hv"], feed_dict={self.test_data:batch})
            value = np.concatenate([value, r], axis=0)

        return value

    def estimate_normaliser(self, n=10**8, batch_size=10**3, std=2.0, budget=120, bar = True):
        
        if n==0:
            assert self.logZ is not None
            return self.logZ

        nbatch = int(np.ceil(n*1.0/batch_size))
        
        t0 = time()
        S = -np.inf
        if bar:
            iterable = tqdm(range(nbatch), ncols=100, desc="estimating logZ")
        else:
            iterable = range(nbatch)
        for i in iterable:
            lse_logr = self.sess.run(self.nodes["lse_logr"], feed_dict={self.nodes["n_rand"]: batch_size, 
                                                                     self.nodes["q_std"] : std})
            S = logsumexp([S,lse_logr])
            if time()-t0>budget:
                break
            
        self.logZ = S - np.log((i+1)*batch_size)
        self.Z = np.exp(self.logZ)
        return self.logZ

    def q_estimate_normaliser(self, n=10**8, batch_size=10**3, std=2.0, budget=120, bar = True):
        
        if n==0:
            assert self.logZ is not None
            return self.logZ

        nbatch = int(np.ceil(n*1.0/batch_size))
        
        t0 = time()
        S = -np.inf
        if bar:
            iterable = tqdm(range(nbatch), ncols=100, desc="estimating logZ")
        else:
            iterable = range(nbatch)
        for i in iterable:
            lse_logr = self.sess.run(self.nodes["q_lse_logr"], feed_dict={self.nodes["n_rand"]: batch_size})
            S = logsumexp([S,lse_logr])
            if time()-t0>budget:
                break
            
        self.logZ = S - np.log((i+1)*batch_size)
        self.Z = np.exp(self.logZ)
        return self.logZ

    
    def eval(self, data, batch_size = 1000, **kwargs):
        kwargs["batch_size"] = batch_size
        self.q_estimate_normaliser(**kwargs)
        
        n = data.shape[0]
        assert self.target.D == data.shape[1]
        logp = self.fun_multiple(data, batch_size = batch_size)
        logp -= self.logZ
        return logp
