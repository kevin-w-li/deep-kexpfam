from abc import abstractmethod
import numpy as np

class DataGenerator(object):
    def __init__(self, D, N_train, N_test):
        self.D = D
        self.N_train = N_train
        self.N_test = N_test
    
    def log_pdf(self, x):
        raise NotImplementedError()
    
    @abstractmethod
    def grad(self, x):
        raise NotImplementedError()
    
    def grad_multiple(self, X):
        return np.array([self.grad(x) for x in X])
    
    @abstractmethod
    def sample(self, N):
        raise NotImplementedError()
    
    def sample_train_test(self):
        X_train = self.sample(self.N_train)
        X_test = self.sample(self.N_test)
        
        return X_train, X_test
    
    @abstractmethod
    def get_params(self):
        return {"D": self.D,
                "N_train": self.N_train,
                "N_test": self.N_test}
    
    def to_string(self):
        return self.__class__.__name__ + "_" + "_".join(["%s=%s" % (k,v) for (k,v)in self.get_params().items()])
