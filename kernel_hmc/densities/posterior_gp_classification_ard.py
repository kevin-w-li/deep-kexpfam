import os
import urllib

from kernel_hmc.densities.gaussian import log_gaussian_pdf
from kernel_hmc.tools.file import sha1sum
from kernel_hmc.tools.log import logger
try:
    import modshogun as sg
except ImportError:
    import shogun as sg
import numpy as np
import scipy as sp


class PseudoMarginalHyperparameters(object):
    """
    Class to represent a GP's marginal posterior distribution of hyperparameters
    
    p(theta|y) \propto p(y|theta) p(theta)
    
    as an MCMC target. The p(y|theta) function is an unbiased estimate.
    Hyperparameters are the length scales of a Gaussian ARD kernel.
    
    Uses the Shogun machine learning toolbox for GP inference.
    """
    def __init__(self, X, y, n_importance, prior_log_pdf, ridge=0., num_shogun_threads=1):
        self.n_importance = n_importance
        self.prior_log_pdf = prior_log_pdf
        self.ridge = ridge
        self.X = X
        self.y = y
        
        self.num_shogun_threads = num_shogun_threads
    
        # tell shogun to use 1 thread only
        logger.debug("Using Shogun with %d threads" % self.num_shogun_threads)
        sg.ZeroMean().parallel.set_num_threads(self.num_shogun_threads)
    
        # shogun representation of data
        self.sg_labels = sg.BinaryLabels(self.y)
        self.sg_feats_train = sg.RealFeatures(self.X.T)
        
        # ARD: set theta, which is in log-scale, as kernel weights
        D = X.shape[1]
        theta_start = np.ones(D)
        
        self.sg_mean = sg.ZeroMean()
        self.sg_likelihood = sg.LogitLikelihood()
        
    def log_pdf(self, theta):
        self.sg_kernel = sg.GaussianARDKernel()
        exp_theta = np.exp(theta)
        if np.any(exp_theta<=0):
            exp_theta[exp_theta<=0]=np.finfo('d').eps
        self.sg_kernel.set_vector_weights(exp_theta)
        inference = sg.EPInferenceMethod(
#         inference=sg.SingleLaplacianInferenceMethod(
                                        self.sg_kernel,
                                        self.sg_feats_train,
                                        self.sg_mean,
                                        self.sg_labels,
                                        self.sg_likelihood)

        # fix kernel scaling for now
        inference.set_scale(1.)
        
        log_ml_estimate = inference.get_marginal_likelihood_estimate(self.n_importance, self.ridge)
        
        # prior is also in log-domain, so no exp of theta
        log_prior = self.prior_log_pdf(theta)
        result = log_ml_estimate + log_prior
            
        return result
    
def log_prior_log_pdf(x):
    D = len(x)
    return log_gaussian_pdf(x, mu=0.*np.ones(D), Sigma=np.eye(D) * 5)

class GlassPosterior(object):
    def __init__(self, n_importance=100, ridge=1e-3, prior_log_pdf=log_prior_log_pdf):
        self.n_importance = n_importance
        self.ridge = ridge
        self.prior_log_pdf = prior_log_pdf
    
    @staticmethod
    def _load_glass_data(data_dir=os.sep.join([os.path.expanduser('~'), "data"])):
        filename = os.sep.join([data_dir, "glass.data"])
        
        try:
            data = np.loadtxt(filename, delimiter=",")
        except IOError:
            # make sure dir exists
            try:
                os.makedirs(data_dir)
            except OSError:
                pass
            
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
            logger.warning("%s not found. Trying to download from %s" % (filename, url))
            urllib.urlretrieve (url, filename)
            
            # try again
            try:
                data = np.loadtxt(filename, delimiter=",")
            except IOError:
                raise RuntimeError("Download failed. Please download manually.")
        
        # make sure file is as expected
        s_reference = "eb292f3709b6fbbeb18a34f95e2293470cbe58ed"
        logger.info("Asserting sha1sum(%s)==%s" % (filename, s_reference))
        s = sha1sum(filename)
        if s != s_reference:
            raise RuntimeError("sha1sum(%s) is %s while reference is %s" % (filename,s, s_reference))
        
        
        # create a binary "window glass" vs "non-window glass" labelling
        lab = data[:, -1]
        lab = np.array([1. if x <= 4 else -1.0 for x in lab])
        
        # cut off ids and labeling
        data = data[:, 1:-1]
        
        return data, lab
    
    def set_up(self):
        # load data using kameleon-mcmc code
        logger.info("Loading data")
        X, y = GlassPosterior._load_glass_data()
    
        # normalise and whiten dataset, as done in kameleon-mcmc code
        logger.info("Whitening data")
        X -= np.mean(X, 0)
        L = np.linalg.cholesky(np.cov(X.T))
        X = sp.linalg.solve_triangular(L, X.T, lower=True).T
        
        # build target, as in kameleon-mcmc code
        self.gp_posterior = PseudoMarginalHyperparameters(X, y,
                                                          self.n_importance,
                                                          self.prior_log_pdf,
                                                          self.ridge,
                                                          num_shogun_threads=1)
    
    def log_pdf(self, theta):
        if not hasattr(self, "gp_posterior"):
            raise RuntimeError("Call set_up method first.")
        
        return self.gp_posterior.log_pdf(theta)
