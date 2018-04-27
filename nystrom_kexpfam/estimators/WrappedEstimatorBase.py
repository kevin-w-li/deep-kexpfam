import numpy as np

class WrappedEstimatorBase(object):
    def __init__(self, num_threads=1):
        self.num_threads = num_threads

    def fit(self, X):
        raise NotImplementedError
    
    def predict(self, X_test):
        raise NotImplementedError
    
    def score(self, X_test):
        raise NotImplementedError
    
#     def fit_and_compute_grad(self, X, X_test):
#         self.fit(X)
#         return self.grad(X_test)
#     
#     def fit_and_compute_score(self, X, X_test):
#         self.fit(X)
#         return self.score(X_test)
#     
#     def fit_and_grad_handle(self, X):
#         self.fit(X)
#         
#         def grad_handle_anon(x):
#             return self.grad(np.atleast_2d(x)).ravel()
#         
#         return grad_handle_anon
    
    