from scipy.spatial.distance import cdist, squareform, pdist

from kernel_hmc.densities.gaussian import sample_gaussian, log_gaussian_pdf
from kernel_hmc.proposals.base import ProposalBase, standard_sqrt_schedule
from kernel_hmc.tools.log import Log
import numpy as np


logger = Log.get_logger()

# low rank update depends on "cholupdate" optional dependency
try:
    from choldate._choldate import cholupdate
    cholupdate_available = True
except ImportError:
    cholupdate_available = False
    logger.warning("Package cholupdate not available. Adaptive Metropolis falls back to (more expensive) re-estimation of covariance.")

if cholupdate_available:
    def rank_one_update_mean_covariance_cholesky_lmbda(u, lmbda=.1, mean=None, cov_L=None, nu2=1., gamma2=None):
        """
        Returns updated mean and Cholesky of sum of outer products following a
        (1-lmbda)*old + lmbda* step_size*uu^T+lmbda*gamm2*I
        rule
        
        Optional: If gamma2 is given, an isotropic term gamma2 * I is added to the uu^T part
        
        where old mean and cov_L=Cholesky(old) (lower Cholesky) are given.
        
        Performs efficient rank-one updates of the Cholesky directly.
        """
        assert lmbda >= 0 and lmbda <= 1
        assert u.ndim == 1
        D = len(u)
        
        # check if first term
        if mean is None or cov_L is None :
            # in that case, zero mean and scaled identity matrix
            mean = np.zeros(D)
            cov_L = np.eye(D) * nu2
        else:
            assert len(mean) == D
            assert mean.ndim == 1
            assert cov_L.ndim == 2
            assert cov_L.shape[0] == D
            assert cov_L.shape[1] == D
        
        # update mean
        updated_mean = (1 - lmbda) * mean + lmbda * u
        
        # update Cholesky: first downscale existing Cholesky
        update_cov_L = np.sqrt(1 - lmbda) * cov_L.T
        
        # rank-one update of the centered new vector
        update_vec = np.sqrt(lmbda) * np.sqrt(nu2) * (u - mean)
        cholupdate(update_cov_L, update_vec)
        
        # optional: add isotropic term if specified, requires looping rank-one updates over
        # all basis vectors e_1, ..., e_D
        if gamma2 is not None:
            e_d = np.zeros(D)
            for d in range(D):
                e_d[:] = 0
                e_d[d] = np.sqrt(gamma2)
                
                # could do a Cholesky update, but this routine does a loop over dimensions
                # where the vector only has one non-zero component
                # That is O(D^2) and therefore not efficient when used in a loop
                cholupdate(update_cov_L, np.sqrt(lmbda) * e_d)
                
                # TODO:
                # in contrast, can do a simplified update when knowing that e_d is sparse
                # manual Cholesky update (only doing the d-th component of algorithm on
                # https://en.wikipedia.org/wiki/Cholesky_decomposition#Rank-one_update
    #             # wiki (MB) code:
    #             r = sqrt(L(k,k)^2 + x(k)^2);
    #             c = r / L(k, k);
    #             s = x(k) / L(k, k);
    #             L(k, k) = r;
    #             L(k+1:n,k) = (L(k+1:n,k) + s*x(k+1:n)) / c;
    #             x(k+1:n) = c*x(k+1:n) - s*L(k+1:n,k);
    
        # since cholupdate works on transposed version
        update_cov_L = update_cov_L.T
        
        # done updating Cholesky
        
        return updated_mean, update_cov_L

def rank_update_mean_covariance_cholesky_lmbda_naive(u, lmbda=.1, mean=None, cov_L=None, nu2=1., gamma2=None):
    """
    Returns updated mean and Cholesky of sum of outer products following a
    (1-lmbda)*old + lmbda* step_size*uu^T
    rule
    
    Optional: If gamma2 is given, an isotropic term gamma2 * I is added to the uu^T part
    
    where old mean and cov_L=Cholesky(old) (lower Cholesky) are given.
    
    Naive version that re-computes the Cholesky factorisation
    """
    assert lmbda >= 0 and lmbda <= 1
    assert u.ndim == 1
    D = len(u)
    
    # check if first term
    if mean is None or cov_L is None :
        # in that case, zero mean and scaled identity matrix
        mean = np.zeros(D)
        cov_L = np.eye(D) * nu2
    else:
        assert len(mean) == D
        assert mean.ndim == 1
        assert cov_L.ndim == 2
        assert cov_L.shape[0] == D
        assert cov_L.shape[1] == D
    
    # update mean
    updated_mean = (1 - lmbda) * mean + lmbda * u
    
    # centered new vector
    update_vec = u - mean

    # reconstruct covariance, update
    update_cov = np.dot(cov_L, cov_L.T)
    update_cov = (1 - lmbda)*update_cov + lmbda*nu2*np.outer(update_vec, update_vec)
    
    # optional: add isotropic term if specified
    if gamma2 is not None:
        update_cov += np.eye(update_cov.shape[0])*gamma2
    
    # re-compute Cholesky
    update_cov_L = np.linalg.cholesky(update_cov)
    
    return updated_mean, update_cov_L


class AdaptiveMetropolis(ProposalBase):
    """
    Implements the adaptive MH.
    
    If "cholupdate" package is available, 
    performs efficient low-rank updates of Cholesky factor of covariance,
    costing O(d^2) computation.
    
    Otherwise, covariance is is simply updated every iteration and its Cholesky
    factorisation is re-computed every time, costing O(d^3) computation.
    """
    
    def __init__(self, target, D, step_size=1., gamma2=0.1,
                 adaptation_schedule=standard_sqrt_schedule, acc_star=0.234):
        ProposalBase.__init__(self, target, D, step_size, adaptation_schedule, acc_star)
        
        self.gamma2 = gamma2
        
        # initialise as scaled isotropic, otherwise Cholesky updates fail
        self.mu = np.zeros(self.D)
        self.L_C = np.eye(self.D) * np.sqrt(self.step_size)    

    def set_batch_covariance(self, Z):
        self.mu = np.mean(Z, axis=0)
        self.L_C = np.linalg.cholesky(self.step_size*np.cov(Z.T)+np.eye(Z.shape[1])*self.gamma2)
    
    def update(self, samples, acc_probs):
        self.t += 1
        
        z_new = samples[-1]
        previous_accpept_prob = acc_probs[-1]
        if self.adaptation_schedule is not None:
            # generate updating weight
            lmbda = self.adaptation_schedule(self.t)
            
            logger.debug("Updating covariance using lmbda=%.3f" % lmbda)
            if cholupdate_available:
                # low-rank update of Cholesky, costs O(d^2) only, adding exploration noise on the fly
                logger.debug("O(d^2) Low rank update of Cholesky of covariance")
                self.mu, self.L_C = rank_one_update_mean_covariance_cholesky_lmbda(z_new,
                                                                                   lmbda,
                                                                                   self.mu,
                                                                                   self.L_C,
                                                                                   self.step_size,
                                                                                   self.gamma2)
            else:
                # low-rank update of Cholesky, naive costs O(d^3), adding exploration noise on the fly
                logger.debug("O(d^3) Low rank update of Cholesky of covariance")
                self.mu, self.L_C = rank_update_mean_covariance_cholesky_lmbda_naive(z_new,
                                                                                   lmbda,
                                                                                   self.mu,
                                                                                   self.L_C,
                                                                                   self.step_size,
                                                                                   self.gamma2)
            
            # update scalling parameter if wanted
            if self.acc_star is not None:
                self._update_scaling(lmbda, previous_accpept_prob)
    
    def proposal(self, current, current_log_pdf):
        """
        Returns a sample from the proposal centred at current, acceptance probability,
        and its log-pdf under the target.
        """
        if current_log_pdf is None:
            current_log_pdf = self.target.log_pdf(current)

        # generate proposal
        proposal = sample_gaussian(N=1, mu=current, Sigma=self.L_C, is_cholesky=True)[0]
        proposal_log_pdf = self.target.log_pdf(proposal)
        
        # compute acceptance prob, proposals probability cancels due to symmetry
        acc_log_prob = np.min([0, proposal_log_pdf - current_log_pdf])
        
        # probability of proposing current when would be sitting at proposal is symmetric
        return proposal, np.exp(acc_log_prob), proposal_log_pdf
    

class StandardMetropolis(AdaptiveMetropolis):
    """
    Implements the adaptive MH with a isotropic proposal covariance.
    """
    
    def __init__(self, target, D, step_size=1.,
                 adaptation_schedule=standard_sqrt_schedule, acc_star=0.234):
        AdaptiveMetropolis.__init__(self, target, D, step_size, 0.0,
                                    adaptation_schedule, acc_star)

    def proposal(self, current, current_log_pdf):
        """
        Returns a sample from the proposal centred at current, acceptance probability,
        and its log-pdf under the target.
        """
        if current_log_pdf is None:
            current_log_pdf = self.target.log_pdf(current)

        # generate proposal
        proposal = sample_gaussian(N=1, mu=current, Sigma=np.eye(self.D) * np.sqrt(self.step_size), is_cholesky=True)[0]
        proposal_log_pdf = self.target.log_pdf(proposal)
        
        # compute acceptance prob, proposals probability cancels due to symmetry
        acc_log_prob = np.min([0, proposal_log_pdf - current_log_pdf])
        
        # probability of proposing current when would be sitting at proposal is symmetric
        return proposal, np.exp(acc_log_prob), proposal_log_pdf
    
def gamma_median_heuristic(Z, num_subsample=1000):
    """
    Computes the median pairwise distance in a random sub-sample of Z.
    Returns a \gamma for k(x,y)=\exp(-\gamma ||x-y||^2), according to the median heuristc,
    i.e. it corresponds to \sigma in k(x,y)=\exp(-0.5*||x-y||^2 / \sigma^2) where
    \sigma is the median distance. \gamma = 0.5/(\sigma^2)
    """
    inds = np.random.permutation(len(Z))[:np.max([num_subsample, len(Z)])]
    dists = squareform(pdist(Z[inds], 'sqeuclidean'))
    median_dist = np.median(dists[dists > 0])
    sigma = np.sqrt(0.5 * median_dist)
    gamma = 0.5 / (sigma ** 2)
    
    return gamma

class KernelAdaptiveMetropolis(ProposalBase):
    def __init__(self, target, D, N, kernel_sigma=1., minimum_size_sigma_learning=100,
                 step_size=1., gamma2=0.1, adaptation_schedule=standard_sqrt_schedule, acc_star=0.234):
        ProposalBase.__init__(self, target, D, step_size, adaptation_schedule, acc_star)
        
        self.kernel_sigma = kernel_sigma
        self.minimum_size_sigma_learning = minimum_size_sigma_learning
        self.N = N
        self.gamma2 = gamma2
        self.Z = np.zeros((0, D))

    def set_batch_covariance(self, Z):
        self.Z = Z
    
    def update(self, samples, acc_probs):
        self.t += 1
        
        previous_accpept_prob = acc_probs[-1]
        if self.adaptation_schedule is not None:
            # generate updating probability
            lmbda = self.adaptation_schedule(self.t)
            
            if np.random.rand() < lmbda:
                # update sub-sample of chain history
                self.Z = samples[np.random.permutation(len(samples))[:self.N]]
                logger.info("Updated chain history sub-sample of size %d with probability lmbda=%.3f" % (self.N, lmbda))
                
                if self.minimum_size_sigma_learning < len(self.Z):
                    # re-compute median heuristic for kernel
                    self.kernel_sigma = 1./gamma_median_heuristic(self.Z, self.N)
                    logger.info("Re-computed kernel bandwith using median heuristic to sigma=%.3f" % self.kernel_sigma)
            
                # update scaling parameter if wanted
                if self.acc_star is not None:
                    self._update_scaling(lmbda, previous_accpept_prob)

    def proposal(self, current, current_log_pdf):
        """
        Returns a sample from the proposal centred at current, acceptance probability,
        and its log-pdf under the target.
        """
        if current_log_pdf is None:
            current_log_pdf = self.target.log_pdf(current)
        
        L_R = self.construct_proposal_covariance_(current)
        proposal = sample_gaussian(N=1, mu=current, Sigma=L_R, is_cholesky=True)[0]
        proposal_log_prob = log_gaussian_pdf(proposal, current, L_R, is_cholesky=True)
        proposal_log_pdf = self.target.log_pdf(proposal)
        
        # probability of proposing y when would be sitting at proposal
        L_R_inv = self.construct_proposal_covariance_(proposal)
        proopsal_log_prob_inv = log_gaussian_pdf(current, proposal, L_R_inv, is_cholesky=True)
        
        log_acc_prob = proposal_log_pdf - current_log_pdf + proopsal_log_prob_inv - proposal_log_prob
        
        return proposal, np.exp(log_acc_prob), proposal_log_pdf
    
    def construct_proposal_covariance_(self, y):
        """
        Helper method to compute Cholesky factor of the Gaussian Kameleon proposal centred at y.
        """
        R = self.gamma2 * np.eye(self.D)
        
        if len(self.Z) > 0:
            # the code is parametrised in gamma=1./sigma
            kernel_gamma = 1./self.kernel_sigma
            # k(y,z) = exp(-gamma ||y-z||)
            # d/dy k(y,z) = k(y,z) * (-gamma * d/dy||y-z||^2)
            #             = 2 * k(y,z) * (-gamma * ||y-z||^2)
            #             = 2 * k(y,z) * (gamma * ||z-y||^2)
            
            # gaussian kernel gradient, same as in kameleon-mcmc package, but without python overhead
            sq_dists = cdist(y[np.newaxis, :], self.Z, 'sqeuclidean')
            k = np.exp(-kernel_gamma * sq_dists)
            neg_differences = self.Z - y
            G = 2 * kernel_gamma * (k.T * neg_differences)
            
            # Kameleon
            G *= 2  # = M
            # R = gamma^2 I + \eta^2 * M H M^T
            H = np.eye(len(self.Z)) - 1.0 / len(self.Z)
            R += self.step_size * G.T.dot(H.dot(G))
        
        L_R = np.linalg.cholesky(R)
        
        return L_R
    
