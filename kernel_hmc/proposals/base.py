from kernel_hmc.tools.assertions import assert_implements_log_pdf_and_grad
from kernel_hmc.tools.log import Log
import numpy as np


logger = Log.get_logger()

def standard_sqrt_schedule(t):
    return 1. / np.sqrt(t + 1)

class ProposalBase(object):
    def __init__(self, D, step_size, adaptation_schedule, acc_star):
        self.D = D
        self.step_size = step_size
        self.adaptation_schedule = adaptation_schedule
        self.acc_star = acc_star
        
        self.t = 0
        
        # some sanity checks
        assert acc_star is None or acc_star > 0 and acc_star < 1
        if adaptation_schedule is not None:
            lmbdas = np.array([adaptation_schedule(t) for t in  np.arange(100)])
            assert np.all(lmbdas >= 0)
            assert np.allclose(np.sort(lmbdas)[::-1], lmbdas)
    
    def initialise(self):
        pass
    
    def proposal(self):
        pass
    
    def update(self, samples, acc_probs):
        self.t += 1
        
        previous_accpept_prob = acc_probs[-1]
        
        if self.adaptation_schedule is not None and self.acc_star is not None:
            # always update with weight
            lmbda = self.adaptation_schedule(self.t)
            self._update_scaling(lmbda, previous_accpept_prob)
    
    def _update_scaling(self, lmbda, accept_prob):
        # difference desired and actuall acceptance rate
        diff = accept_prob - self.acc_star
        
        new_log_step_size = np.log(self.step_size) + lmbda * diff
        new_step_size = np.exp(new_log_step_size)
        
        logger.debug("Acc. prob. diff. was %.3f-%.3f=%.3f. Updating step-size from %s to %s." % \
                     (accept_prob, self.acc_star, diff, self.step_size, new_step_size))

        self.step_size = new_step_size
