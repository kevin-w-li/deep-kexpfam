import numpy as np
import tensorflow as tf
from LiteNet import *
from Utils import support_1d
'''
from kernel_hmc.mini_mcmc.mini_mcmc import mini_mcmc
from kernel_hmc.densities.gaussian import IsotropicZeroMeanGaussian
from kernel_hmc.proposals.kmc import KMCStatic
'''
from tqdm import tqdm_notebook, tqdm
from collections import OrderedDict
import warnings
from scipy.stats import norm
from scipy.misc import logsumexp
from sklearn.cluster import KMeans
from nuts.emcee_nuts import NUTSSampler


from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

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

    def set_test(self, *args):
        for i in range(self.n_clusters):
            self.lites[i].set_test(*args)

    def set_train(self, *args):
        for i in range(self.n_clusters):
            self.lites[i].set_train(*args)
        
    def fit_alpha(self, n):
        for i in range(self.n_clusters):
            self.lites[i].fit_alpha(n)
            
    def estimate_normaliser(self, **kwargs):
        for i in range(self.n_clusters):
            self.lites[i].estimate_normaliser(**kwargs)

    def estimate_data_lik(self, data, **kwargs):
        ll = np.zeros((data.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            l = self.lites[i]
            d = self.targets.ps[i].trans(data) 
            if l.logZ is None:
                l.estimate_normaliser(**kwargs)
            ll[:,i] = l.estimate_data_lik(d, **kwargs) + np.linalg.slogdet(self.targets.ps[i].W)[1]
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
        

class DeepLite(object):

    def __init__(self, target, nlayer=3, nneuron=30,
                    noise_std=0.0, points_std=0.2,
                    init_weight_std=1.0, init_log_sigma=[0.0], init_log_lam=-2.0, log_lam_weights=-100,
                    seed=None, keep_prob = 1.0, mixture_kernel=False, base=True,
                    npoint=300, ntrain=300, nvalid=300, points_type="fixed", clip_score=False,
                    step_size=1e-2, niter=None, patience=None, kernel_type="gaussian",
                    gpu_count=1, fn_ext = "", trainable=True
                    ):        
        
        self.target = target
        
        self.fn_ext = fn_ext
        self.seed = seed
        self.trainable = trainable
        
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
                                    kernel_type    = kernel_type
                                )
        if nlayer == 0:
            self.model_params["ndims"] = [0,]
        self.train_params   = dict( _step_size = step_size,
                                    niter = niter,
                                    ntrain = ntrain,
                                    nvalid = nvalid,
                                    patience=patience,
                                    points_type = points_type,
                                    clip_score = clip_score
                                    )
        
        self.states = OrderedDict()
        
        self.state_hist = OrderedDict()
        
        for k in self.states.keys():
            self.state_hist[k] = []
            
        
        self.target = target
        
        if niter is not None:
            self.niter = niter
        else:
            self.niter = 0

        self.build_model(gpu_count)

        self.logZ = None
        
    def build_model(self, gpu_count=1):
        
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

            npoint  = self.model_params["npoint"]
            ntrain  = self.train_params["ntrain"]
            clip_score  = self.train_params["clip_score"]
            base    = self.model_params["base"]
            points_type = self.train_params["points_type"]
            if self.target.nkde:
                self.train_kde = tf.placeholder(FDTYPE, shape=(None,), name="train_kde")
                self.valid_kde = tf.placeholder(FDTYPE, shape=(None,), name="valid_kde")
            else:
                self.valid_kde = None
                self.train_kde = None
            
            self.train_params["step_size"]  = tf.placeholder(FDTYPE, shape=[], name="step_size")
            
            train_data  = tf.placeholder(FDTYPE, shape=(None, target.D), name="train_data")
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

                    network = DeepNetwork(layers, ndim_out = ndims[-1], init_weight_std = init_weight_std/np.sqrt(ndims[-1][0]), add_skip=nlayer>1)
                    net_outs.append(network.forward_tensor(test_data))
                    kernel = CompositeKernel(kernel, network)

                kernels.append(kernel)
                sigmas.append(sigma)
                props.append(prop)
                kernel_grams.append(kernel.get_gram_matrix(test_points, test_data))

            self.ops["net_outs"]  = net_outs
            self.ops["kernel_grams"] = kernel_grams
            prop_sum = tf.reduce_sum(props)

            for i in range(nkernel-1):
                props[i] = props[i]/prop_sum

            props[-1]  = 1-tf.reduce_sum(props[:-1])

            kernel    = MixtureKernel( kernels, props )

            kn = LiteModel(kernel, npoint, points=points, 
                            init_log_lam=init_log_lam, log_lam_weights=log_lam_weights, 
                            noise_std=noise_std, base=base)

            loss, score, _, _, r_norm, l_norm, curve, w_norm, k_loss, test_score, \
                self.states["outlier"], save_alpha = \
                kn.val_score(train_data=train_data, valid_data=valid_data, test_data = test_data, 
                            train_kde=self.train_kde, valid_kde=self.valid_kde, clip_score=clip_score)

            optimizer = tf.train.AdamOptimizer(self.train_params["step_size"])

            if self.trainable:
                raw_gradients, variables = zip(*optimizer.compute_gradients(loss))
                gradients = [ tf.where(tf.is_nan(g), tf.zeros_like(g), g) for g in raw_gradients if g is not None]
                gradients, self.states["grad_norm"] = tf.clip_by_global_norm(gradients, 100.0)

                accum_gradients = [tf.Variable(tf.zeros_like(g), trainable=False) for g in gradients]
                self.ops["zero_op"]  = [ag.assign(tf.zeros_like(ag)) for ag in accum_gradients]

                nbatch = tf.placeholder(FDTYPE, shape=[], name="nbatch")
                self.train_params["nbatch"] = nbatch

                self.ops["accum_op"] = [accum_gradients[i].assign_add(g/nbatch) for i, g in enumerate(gradients)]
                self.ops["train_step"] = [optimizer.apply_gradients( zip(gradients, variables) ), save_alpha]
            
            lambdas = [v for v in tf.trainable_variables() if "regularizers" in v.name]
            if len(lambdas)>0:
                self.ops["train_lambdas"] = optimizer.minimize(loss, var_list = lambdas)

            self.alpha   = tf.Variable(tf.zeros(npoint), dtype=FDTYPE, name="alpha_eval", trainable=False)
            
            
            quad  = tf.Variable(tf.zeros((npoint, npoint)), dtype=FDTYPE, name="quad", trainable=False)
            lin   = tf.Variable(tf.zeros(npoint), dtype=FDTYPE, name="lin", trainable=False)
            
            
            ndata = self.target.N + self.target.nvalid
            ndata = int(np.floor(ndata/ntrain)*ntrain)
            
            nbatch_final = ndata / ntrain
            self.train_params["nbatch_final"] = nbatch_final
            
            batch_quad, batch_lin = \
                kn.opt_alpha(data=train_data, kde = self.train_kde,)[-2:]

            acc_quad = tf.assign_add(quad, batch_quad/nbatch_final)
            acc_lin  = tf.assign_add(lin , batch_lin /nbatch_final)

            self.ops["acc_quad_lin"] = [acc_quad, acc_lin]

            self.ops["fit_alpha"] = tf.assign(self.alpha, tf.matrix_solve(quad, lin[:,None])[:,0])


            self.min_log_pdf = -np.inf
            hv, gv, fv = kn.evaluate_hess_grad_fun(test_data, alpha=self.alpha)
            sc         = kn.individual_score(test_data, alpha=self.alpha)[0]
            self.ops["hv"] = hv
            self.ops["gv"] = gv
            self.ops["fv"] = fv
            self.ops["sc"] = sc
            
            config = tf.ConfigProto(device_count={"GPU":gpu_count})
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
            self.test_points= test_points
            
            self.states["score"]   = score
            self.states["loss"]    = loss
            self.states["sigmas"]   = sigmas
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
        
    def step(self, feed):
        

        nbatch = self.train_params["_nbatch"]
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

        res = self.sess.run([self.ops["train_step"]] + self.states.values()[:-1], feed_dict=feed)[1:]

        ntest_batch = self.target.nvalid/100
        test_score = 0
        for i in range(ntest_batch):
            test_data = self.target.valid_data[i*100:(i+1)*100]
            test_score += self.sess.run( self.states["test_score"], 
                                        feed_dict={self.test_data:test_data}) / ntest_batch

        res.append(test_score)
        
        return res

    def fit

    def fit_kernel(self, niter = None, ntrain = None, nvalid=None, nbatch=1, patience=200,
            step_size=None, verbose = False, print_time_interval=10,
           print_iteration_interval=200, true_grad_fun=None):
        
        train_data = self.train_data
        valid_data = self.valid_data
        
        sess = self.sess
        target = self.target
        
        if ntrain is not None:
            self.train_params["ntrain"] = ntrain
        if nvalid is not None:
            self.train_params["nvalid"] = nvalid
        if niter is not None:
            self.train_params["niter"] = niter
        if patience is not None:
            self.train_params["patience"] = patience
            
        self.train_params["patience"] = patience
        
        feed={}
        
        if not isinstance(step_size, float) :
            feed[self.train_params["step_size"]] = self.train_params["_step_size"]
        else:
            feed[self.train_params["step_size"]] = step_size

        feed[self.train_params["nbatch"]] = nbatch
        self.train_params["_nbatch"] = nbatch

        t0 = time()
        last_time = t0
        
        self.set_train()

        last_epoch = 0
        best_score = np.inf
        wait_window = 0
        with tqdm(range(niter+1), ncols=100, desc="trainining kernel", postfix=[dict(loss="%.3f" % 0.0, test="%.3f" % 0.0)]) as tr:    

            for i in tr: 

                res = self.step(feed)

                for ki, k in enumerate(self.state_hist.keys()):
                    self.state_hist[k].append(res[ki])

                block_score  = self.state_hist["score"][i-min(i, 30):]
                block_test_score  = self.state_hist["test_score"][i-min(i, 30):]


                block_score_mean = np.mean(block_score)
                block_test_score_mean = np.mean(block_test_score)

                tr.postfix[0]["loss"] = "%.3f" % block_score_mean
                tr.postfix[0]["test"] = "%.3f" % best_score
                tr.update()

                t0 = time()

                epoch = i
                
                current_score = self.state_hist["test_score"][-1]

                if current_score < best_score:
                    best_score = min(best_score, current_score)
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


                if ((time() - last_time) > print_time_interval or \
                        i % min(niter, print_iteration_interval) == 0 ) and verbose:

                    block_score_mean = np.mean(block_score)
                    block_score_std  = np.std(block_score)

                    last_time = time()

                    #grad_vals, global_norm_val = self.sess.run([raw_gradients, global_norm], feed_dict=feed)

                    tqdm.write( '==================' )
                    tqdm.write( 'Iteration %5d, score: %5.3g +- %5.3g, time taken %.2f' % (i, block_score_mean, block_score_std, time()-t0) )

                    if true_grad_fun is not None:
                        tg = true_grad_fun(feed[self.test_data])
                        mg = self.gv.eval(feed)
                        tqdm.write( 'true score %.6g' % 0.5 * np.mean((tg, mg)**2))

                    state_str = ""

                    for ki, k in enumerate(self.state_hist.keys()):
                        state_str += "%10s = %10.5g" % (k, self.state_hist[k][-1])
                        if (ki+1) % 4 == 0:
                            state_str += "\n"
                    tqdm.write(state_str)
        
        if patience>0:
            self.load()
        else:
            self.save()
        print "best score: %.5f" % best_score
        '''
        data = self.final_train_data(min(self.target.N, 5000))
        feed[self.train_data] = data
        feed[self.valid_data] = self.target.valid_data
        for i in tqdm(range(100), desc="lamdas", ncols=100):

            res = self.sess.run([self.ops["train_lambdas"]] + self.states.values(), feed_dict=feed)[1:]

            for ki, k in enumerate(self.states.keys()):
                self.state_hist[k].append(res[ki])
        print "best score: %.5f" % self.state_hist["test_score"][-1]
        '''
        
        return self.state_hist

    def final_train_data(self, ndata):
        data = self.target.data[:ndata]
        if self.target.nkde:
            kde  = self.target.kde_logp[:ndata]
        else:
            kde  = None
        return data, kde
    
    def fit_alpha(self):
        
        nf = self.train_params["nbatch_final"]
        nt = self.train_params["ntrain"]
        
        train_valid_data = np.r_[self.target.data, self.target.valid_data]
        for i in range(nf):
            data = train_valid_data[i*nt:(i+1)*nt]
            self.sess.run(self.ops["acc_quad_lin"], feed_dict={self.train_data:data})

        self.sess.run(self.ops["fit_alpha"])

        
    def set_test(self, rebuild=False, gpu_count=None):

        if rebuild:
            assert gpu_count is not None, "specify number of gpu"
            if isinstance(self.seed, int):
                self.save()
                self.build_model(gpu_count)
                self.load()
            else:   
                self.save("tmp")
                self.build_model(gpu_count)
                self.load("tmp")
            
        self.sess.run(self.ops["set_keepall"])
        
    def set_train(self, rebuild=False, gpu_count=None):
        if rebuild:
            assert gpu_count is not None, "specify number of gpu"
            self.save("tmp")
            self.build_model(gpu_count)
            self.load("tmp")
        self.sess.run(self.ops["set_dropout"])
        
    def default_file_name(self):

        file_name = "%s_D%02d_l%d_nd%d_np%d_nt%d_nv%d_pt%s_ss%d_ni%d_n%02d_k%d_m%d_b%d_p%d_nk%d_c%d" % \
            (self.target.name[0], self.target.D, self.model_params["nlayer"], 
             np.prod(self.model_params["ndims"][0]), 
             self.model_params["npoint"], self.train_params["ntrain"], 
             self.train_params["nvalid"], self.train_params["points_type"][0],
             int(np.around(self.train_params["_step_size"]*10000)), 
             self.niter,
             self.model_params["noise_std"]*100,
             self.model_params["_keep_prob"]*10,
             self.model_params["mixture_kernel"],
             self.model_params["base"],
             self.train_params["patience"],
             len(self.model_params["init_log_sigma"]),
             self.train_params["clip_score"])

        if self.model_params["kernel_type"] == "linear":
            file_name += "_lin"

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
        nbatch = neval/batch_size

        value = np.empty((neval))

        for i in range(nbatch):
            batch = data[i*batch_size:(i+1)*batch_size]
            value[i*batch_size:(i+1)*batch_size] = \
                self.sess.run(self.ops["sc"], feed_dict={self.test_data:batch})

        return value

    def grad_multiple(self, data, batch_size=100):

        neval = data.shape[0]
        nbatch = neval/batch_size

        value = np.empty((neval, self.target.D))

        for i in range(nbatch):
            batch = data[i*batch_size:(i+1)*batch_size]
            value[i*batch_size:(i+1)*batch_size,:] = \
                self.sess.run(self.ops["gv"], feed_dict={self.test_data:batch})

        return value

    def fun_multiple(self, data, batch_size=100):

        neval = data.shape[0]
        nbatch = neval/batch_size

        value = np.empty((neval))

        for i in range(nbatch):

            batch = data[i*batch_size:(i+1)*batch_size]
            value[i*batch_size:(i+1)*batch_size] = \
                self.sess.run(self.ops["fv"], feed_dict={self.test_data:batch})
        value[value<self.min_log_pdf] = -np.inf
        return value

    def hess_multiple(self, data, batch_size=100):

        neval = data.shape[0]
        nbatch = neval/batch_size

        value = np.empty((neval, self.target.D, self.target.D))

        for i in range(nbatch):
            batch = data[i*batch_size:(i+1)*batch_size]
            value[i*batch_size:(i+1)*batch_size] = \
                self.sess.run(self.ops["vv"], feed_dict={self.test_data:batch})

        value = value.reshape(neval, np.prod(self.target.D), np.prod(self.target.D))
        return value

    def setup_mcmc(self, sigma=1.0, num_steps_min=1, num_steps_max=10, step_size_min=0.01, step_size_max=0.1,
                    min_log_pdf=0.0):

        momentum = IsotropicZeroMeanGaussian(sigma=sigma, D=np.prod(self.target.D))
        self.kmc = KMCStatic(self, momentum, num_steps_min, num_steps_max,
                            step_size_min, step_size_max)
        self.min_log_pdf = min_log_pdf

    def sample(self, N, nchain=1, thin=1, burn=0):
        
        samples = []

        assert N % nchain == 0

        for i in tqdm(range(nchain), ncols=100, desc="one chain"):
            s = self.sample_one_chain(N/nchain*thin+burn, self.target.sample(1)[0])
            samples.append(s["samples"][burn::thin])

        return np.concatenate(samples, axis=0)


    def sample_one_chain(self, N, start):
        assert self.kmc is not None

        samples, proposals, accepted, acc_prob, log_pdf, times, step_sizes = \
            mini_mcmc(self.kmc, start, N+1, np.prod(self.target.D))
        
        return dict(samples=samples, proposals=proposals, accepted=accepted, acc_prob=acc_prob,
            log_pdf=log_pdf, times=times, step_sizes=step_sizes)

    def nuts(self, N, thin=10, nchain=10, burn=1000, delta = 0.7):
        
        samples = []

        assert N % nchain == 0

        for i in tqdm(range(nchain), ncols=100, desc="one chain"):
            s = self.nuts_one_chain(N*thin/nchain, self.target.sample(1)[0], burn, delta)
            samples.append(s[::thin])

        return np.concatenate(samples, axis=0)

    def nuts_one_chain(self, nsample, theta0, Madapt, delta):
        
        def lnprobfn(x):
            x = np.atleast_2d(x)
            return self.fun_multiple(x, batch_size=1)[0]

        def gradfn(x):
            x = np.atleast_2d(x)
            return self.grad_multiple(x, batch_size=1)[0]

        sampler = NUTSSampler(self.target.D, lnprobfn, gradfn)
        samples = sampler.run_mcmc(theta0, nsample, Madapt, delta)

        return samples

    def estimate_normaliser(self, n=10**5, batch_size=10**5, std=1.0):
        
        s = np.random.randn(n, self.target.D) * std
        logq = norm.logpdf(s, loc=0, scale=std).sum(-1)
        logp = self.fun_multiple(s, batch_size=batch_size)
        self.logZ = logsumexp(logp-logq) - np.log(n)
        self.Z = np.exp(self.logZ)
        return self.logZ

    
    def estimate_data_lik(self, data, batch_size = 1000, **kwargs):
        if self.logZ is None:
            self.estimate_normaliser(**kwargs)
        
        n = data.shape[0]
        assert self.target.D == data.shape[1]
        logp = self.fun_multiple(data, batch_size = batch_size)
        logp -= self.logZ
        return logp


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

        self.test_data  = tf.placeholder(FDTYPE, shape=(None, self.D), name="test_data")
        
        if points_type == "train":
            self.rand_train_data = rand_train_data[:self.npoint]
            self.rand_points     = self.rand_train_data
            self.train_data = tf.Variable(self.rand_train_data, name="train_data", dtype=FDTYPE)
            self.points  = tf.identity(self.train_data, name="points")
        elif points_type == "opt":
            self.rand_train_data = rand_train_data
            self.points     = tf.Variable(np.zeros((self.npoint, self.D)), dtype=FDTYPE, name="points")
            self.train_data = tf.Variable(self.rand_train_data, dtype=FDTYPE, name="train_data")
        elif points_type == "sep":
            self.rand_train_data = rand_train_data
            randint = np.random.randint(rand_train_data.shape[0], size=self.npoint)
            self.rand_points  = rand_train_data[randint,:]
            self.points     = tf.Variable(self.rand_points, dtype=FDTYPE, name="points")
            self.train_data = tf.Variable(self.rand_train_data, dtype=FDTYPE, name="train_data")
        else:
            raise NameError("points_type needs to be 'train', 'opt', or 'sep'")
            
        
        self.alpha   = tf.Variable(tf.zeros(self.npoint), dtype=FDTYPE, name="alpha")

        kernel  = GaussianKernel(sigma=1.0)
        if self.nlayer>0:

            layer_1 = LinearSoftNetwork((self.D,), self.ndims[0], 
                                        init_weight_std=1, scope="fc1")
            layers = [LinearSoftNetwork(self.ndims[i], self.ndims[i+1], 
                                        init_weight_std=1,  scope="fc"+str(i+2)) for i in range(self.nlayer-2)]
            network = DeepNetwork([layer_1, ] + layers, ndim_out = self.ndims[-1], init_weight_std = 1, add_skip=self.nlayer>1)
            kernel = CompositeKernel(kernel, network)
            prop = 1/(1+tf.exp(-tf.Variable(0, dtype=FDTYPE)))
            kernel = MixtureKernel([kernel, GaussianKernel(sigma=1.0)], [1.0-prop, prop])

        self.kn = LiteModel(kernel, points=self.points, init_log_lam=0, log_lam_weights=-3, alpha=self.alpha)

        self.kn.npoint = self.npoint
        
        self.hv, self.gv, self.fv = self.kn.evaluate_hess_grad_fun(self.test_data)
        self.alpha_assign_op = self.kn.opt_score(data=self.train_data)
        
        init = tf.global_variables_initializer()
        
        self.sess = tf.Session(config=config)
        self.kmc = None
        self.sess.run(init)

        ckpt = "ckpts/"+fn+".ckpt"

        #saver = tf.train.Saver(allow_empty=True)
        #saver.restore(self.sess, ckpt)

        optimistic_restore(self.sess, ckpt)

        self.sess.run(self.alpha_assign_op)
            
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
                self.sess.run(self.gv, feed_dict={self.test_data:batch})   
        return value
        
    def fun_multiple(self, data, batch_size=100):
        
        neval = data.shape[0]
        nbatch = neval/batch_size
        value = np.empty((neval))
        
        for i in range(nbatch):

            batch = data[i*batch_size:(i+1)*batch_size]
            value[i*batch_size:(i+1)*batch_size] = \
                self.sess.run(self.fv, feed_dict={self.test_data:batch}) 
        value[value<self.min_log_pdf] = -np.inf
        return value
        
    def hess_multiple(self, data, batch_size=100):
        
        neval = data.shape[0]
        nbatch = neval/batch_size
        value = np.empty((neval, self.D, self.D))

        for i in range(nbatch):
            batch = data[i*batch_size:(i+1)*batch_size]
            value[i*batch_size:(i+1)*batch_size] = \
                self.sess.run(self.hv, feed_dict={self.test_data:batch}) 
        return value
    
    def retrain(self, rand_train_data):
        
        self.sess.run(self.alpha_assign_op, feed_dict={self.train_data:rand_train_data})

    def setup_mcmc(self, sigma=1.0, num_steps_min=1, num_steps_max=10, step_size_min=0.01, step_size_max=0.1,
                    min_log_pdf=0.0):
        
        momentum = IsotropicZeroMeanGaussian(sigma=sigma, D=self.D)
        self.kmc = KMCStatic(self, momentum, num_steps_min, num_steps_max, 
                            step_size_min, step_size_max)
        self.min_log_pdf = min_log_pdf

    def sample(self, N, start):
        assert self.kmc is not None
        samples, proposals, accepted, acc_prob, log_pdf, times, step_sizes = mini_mcmc(self.kmc, start, N+1, self.D)
        return samples

