from kernel_hmc.proposals.base import standard_sqrt_schedule
from kernel_hmc.proposals.hmc import HMCBase
from kernel_hmc.tools.assertions import assert_implements_log_pdf_and_grad
from kernel_hmc.tools.log import Log
import numpy as np


logger = Log.get_logger()

class KMCStatic(HMCBase):
    """
    """
    
    def __init__(self, surrogate, momentum, num_steps_min=10, num_steps_max=100, step_size_min=0.05,
                 step_size_max=0.3, adaptation_schedule=None, acc_star=0.7):
        """
        """
        HMCBase.__init__(self,  momentum, num_steps_min, num_steps_max, step_size_min, step_size_max,
                         adaptation_schedule, acc_star)
        
        assert_implements_log_pdf_and_grad(surrogate)
        
        self.surrogate = surrogate
        self.target = surrogate
    
    def accept_prob_log_pdf(self, current, q, p0_log_pdf, p_log_pdf, current_log_pdf):
        # same as super-class, but with original target
        #kernel_target = self.target
        #self.target = self.orig_target
        
        acc_prob, log_pdf_q = HMCBase.accept_prob_log_pdf(self, current, q, p0_log_pdf, p_log_pdf, current_log_pdf)
        
        # restore target
        # self.target = kernel_target
        
        return acc_prob, log_pdf_q

class KMC(KMCStatic):
    def __init__(self, surrogate, momentum, num_steps_min=10, num_steps_max=100, step_size_min=0.05,
                 step_size_max=0.3, adaptation_schedule=standard_sqrt_schedule, acc_star=0.6):
        KMCStatic.__init__(self, surrogate,
                           momentum, num_steps_min, num_steps_max, step_size_min, step_size_max,
                           adaptation_schedule, acc_star)
        
        # can the surrogate be online updated?
        self.surrogate_has_update_fit = hasattr(surrogate, 'update_fit')
        
    def update(self, samples, acc_probs):
        self.t += 1
        
        z_new = samples[-1][np.newaxis, :]
        previous_accpept_prob = acc_probs[-1]
        
        if self.adaptation_schedule is not None:
            # generate updating weight
            lmbda = self.adaptation_schedule(self.t)
            
            if np.random.rand() <= lmbda:
                if self.surrogate_has_update_fit:
                    logger.info("Updating surrogate (was probability lmbda=%.3f)" % lmbda)
                    self.surrogate.update_fit(z_new)
                else:
                    logger.info("Re-fitting surrogate (was probability lmbda=%.3f)" % lmbda)
                    self.surrogate.fit(samples)
            
            if self.acc_star is not None:
                self._update_scaling(lmbda, previous_accpept_prob)
