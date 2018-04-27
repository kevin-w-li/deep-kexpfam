from abc import abstractmethod
import os

import numpy as np
from nystrom_kexpfam.data_generators.Base import DataGenerator
from nystrom_kexpfam.log import logger
from nystrom_kexpfam.posterior_gp_classification_ard import GlassPosterior


class Dataset(DataGenerator):
    """
    Wrapper for datasets that allows sampling training/test data from a fixed file
    """
    def __init__(self, fname, N_train, N_test):
        assert N_train + N_test == 1.0
        assert N_train > 0 and N_test > 0
        
        super(Dataset, self).__init__(None, N_train, N_test)
        
        self.fname = fname
    
    @abstractmethod
    def _load_file_as_array(self):
        loaded = np.load(os.path.expanduser(self.fname))
        if isinstance(loaded, np.ndarray):
            return loaded.astype(np.float64)
        else:
            return ["X_train"].astype(np.float64)
    
    def sample_train_test(self):
        data = self._load_file_as_array()
        
        N = np.shape(data)[0]
        D = np.shape(data)[1]
        N_train = int(N * self.N_train)
        
        logger.info("Loaded %dx%d array. Subsampling %d for training." % \
                     (N, D, N_train))
        
        inds = np.random.permutation(N)
        
        X_train = data[inds[:N_train]]
        X_test = data[inds[N_train:]]
        
        return X_train, X_test
    
class LogPDFDataset(Dataset):
    """
    Extends datasets to potentially support a log-pdf method (if possible)
    """
    def __init__(self, fname, N_train):
        assert N_train > 0 and N_train < 1
        super(LogPDFDataset, self).__init__(fname, N_train, 1.0 - N_train)
    
    def log_pdf(self, x):
        return NotImplementedError
    
    def sample(self):
        return self.sample_train_test()[0]

class GlassPosteriorDataset(LogPDFDataset):
    """
    Dataset that contains (thinned) samples from the posterior over HP hyperparameters
    over the UCL glass dataset, as described by "Gradient-free Hamiltonian Monte Carlo
    with efficient kernel exponential families" by Strathmann et al.
    
    Adds a log-pdf method based on importance sampling with a EP importance proposal,
    which is computed using Shogun.
    
    Obviously only works if file with samples to match the posterior that
    is used to estimate the log_pdf (default settings).
    """
    def __init__(self, fname, N_train):
        super(GlassPosteriorDataset, self).__init__(fname, N_train)
    
        self.target = GlassPosterior()
        self.target.set_up()
        
    def log_pdf(self, x):
        return self.target.log_pdf(x)
    
