import numpy as np
from scipy.optimize import approx_fprime, check_grad

def lite_score(X, x, alpha, sigma):
    '''
    This is the score function under the lite model
    '''
    if X.ndim == 1:
        X = X[None,:]
    if x.ndim == 1:
        x = x[None,:]

    N = x.shape[0]
    D = X.shape[1]

    C = (X[:,None,:] - x[None,:,:])
    K = np.sum(C**2,(2))
    K /= -sigma
    np.exp(K, K)

    C = np.sum(-1+2.0/sigma*C**2, 2)
    J = 2.0/N/sigma * alpha.dot(K * C).sum()

    # Original implementation
    # J2 = 0
    # for i in range(D):
    #      X2 = X[:,i]**2
    #      x2 = x[:,i]**2
    #      xi = x[:,i]
    #      Xi = X[:,i]
    #      J2 += 2.0/sigma * (alpha.dot(K).dot(x2) + (alpha*X2).dot(K).sum() - 2*(alpha*Xi).dot(K).dot(xi))\
    #              - alpha.dot(K).sum()
    # J2 *= 2.0/N/sigma
    # print J, J2

    C = (X[:,None,:] - x[None,:,:])
    J += 2.0/N/(sigma**2)*np.sum(np.sum(alpha[:,None,None] * (C) * K[:,:,None], 0)**2) # FIXME +=

    # Original implementation
    # J2 = 0
    # for i in range(D):
    #     xi = x[:,i]
    #     Xi = X[:,i] 
    #     J2 += ((alpha*Xi).dot(K) - xi*(alpha.dot(K))).dot((alpha*Xi).dot(K)-(alpha.dot(K))*xi)
    # J2 *= 2.0/N/sigma**2
    # print J, J2
    return J

def gaussian_kernel_prime(xi, xj, sigma, W):

def score_prime(X, x, alpha, sigma):
    
    I = X.shape[0]
    J = x.shape[0]




def network(fun,W, X):
    output = fun(W.dot(X))
    return output

def network_prime(fun_prime, W, X):
    return X[:,None,:] * fun_prime(W.dot(X.T)).T[:,:,None]
    
def model_score(fun, W, X, x, alpha, sigma):
    CX = network(fun, W, X)
    Cx = network(fun, W, x)
    return lite_score(CX, Cx, alpha, sigma)

np.random.seed(53)
X = np.random.randn(10,5)
x = np.random.randn(5,2)
W = np.random.randn(4,X.shape[1])
alpha = np.random.rand(100)
sigma = 2.0

fun = lambda x: np.tanh(x)
fun_prime = lambda x: 1-np.tanh(x)**2

net_prime = np.zeros((X.shape[0], W.shape[0], W.shape[1]))
for i in range(X.shape[0]):
    for j in range(W.shape[0]):
        net_prime[i,j,:] +=  approx_fprime(W[j,:], lambda x: network(fun, x, X[i,:]), 1e-8)
print np.allclose(net_prime, network_prime(fun_prime, W, X))


