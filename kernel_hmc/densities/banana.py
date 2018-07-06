rom theano import function
import theano

import numpy as np
import theano.tensor as T

def log_banana_pdf_theano_expr(x, bananicity, V):
    transformed = x.copy()
    transformed = T.set_subtensor(transformed[1], x[1] - bananicity * ((x[0] ** 2) - V))
    transformed = T.set_subtensor(transformed[0], x[0] / T.sqrt(V))
    
    log_determinant_part = 0.
    quadratic_part = -0.5 * transformed.dot(transformed)
    const_part = -0.5 * x.shape[0] * np.log(2 * np.pi)
    
    banana_log_pdf_expr = const_part + log_determinant_part + quadratic_part
    return banana_log_pdf_expr

# build theano functions for log-pdf and gradient
x = T.dvector('x')
bananicity = T.dscalar('bananicity')
V = T.dscalar('V')
banana_log_pdf_expr = log_banana_pdf_theano_expr(x, bananicity, V)
banana_log_pdf_theano = function([x, bananicity, V], banana_log_pdf_expr)
banana_log_pdf_grad_theano = function([x, bananicity, V], theano.gradient.jacobian(banana_log_pdf_expr, x))

def log_banana_pdf(x, bananicity=0.03, V=100, compute_grad=False):
    if not compute_grad:
        return np.float64(banana_log_pdf_theano(x, bananicity, V))
    else:
        return np.float64(banana_log_pdf_grad_theano(x, bananicity, V))

def sample_banana(N, D, bananicity=0.03, V=100):
    X = np.random.randn(N, 2)
    X[:, 0] = np.sqrt(V) * X[:, 0]
    X[:, 1] = X[:, 1] + bananicity * (X[:, 0] ** 2 - V)
    if D > 2:
        X = np.hstack((X, np.random.randn(N, D - 2)))
    
    return X

class Banana(object):
    def __init__(self, D=2, bananicity=0.03, V=100):
        self.D = D
        self.bananicity = bananicity
        self.V = V
    
    def log_pdf(self, x):
        return log_banana_pdf(x, self.bananicity, self.V, compute_grad=False)
    
    def grad(self, x):
        return log_banana_pdf(x, self.bananicity, self.V, compute_grad=True)
    
    def set_up(self):
        pass
