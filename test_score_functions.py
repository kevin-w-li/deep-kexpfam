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

def gaussian_kernel_prime(ci, cj, dci_dtheta, dcj_dtheta, sigma, W):
    '''
    Derivative of Gaussian kernel on top of a network w.r.t network parameter
    '''
    A = np.exp(-1.0/sigma*np.linalg.norm(ci-cj)**2)*(-2.0/sigma*(ci-cj)*(dci_dtheta-dcj_dtheta))
    return A

def score_prime(X, x, alpha, sigma):
    
    I = X.shape[0]
    J = x.shape[0]

def network(fun,W, X):
    '''
    a simple network with nonlinearity fun and weight matrix W
    '''
    if X.ndim == 1:
        X = X[None,:]
    output = fun(W.dot(X.T))
    return output

def network_prime(fun_prime, W, X):
    '''
    The derivative of the network for a batch of 
    Since the network is only one layer, d(out)_i by dW is a matrix with many zeros, so dout_dW
    is just arranged in a matrix
    X:      ndata x ndim
    outptu: ndata x W.shape[0] x W.shap[1]

    '''
    return X[:,None,:] * fun_prime(W.dot(X.T)).T[:,:,None]
    
def model_score(fun, W, X, x, alpha, sigma):
    '''
    The score of the full model
    '''
    CX = network(fun, W, X)
    Cx = network(fun, W, x)
    return lite_score(CX, Cx, alpha, sigma)


def gaussian_kernel(xi, xj, W, sigma):
    '''
    Outputs Guassian kernel on network outputs
    '''
    W = W.reshape(4,xi.shape[1])
    ci = network(fun, W, xi)
    cj = network(fun, W, xj)
    return np.exp(-np.linalg.norm(ci-cj)**2/sigma)

np.random.seed(53)
X = np.random.randn(10,5)
x = np.random.randn(2,5)
W = np.random.randn(4,X.shape[1])
alpha = np.random.rand(100)
sigma = 2.0

fun = lambda x: np.tanh(x)
fun_prime = lambda x: 1-np.tanh(x)**2


# check network derivative
net_prime = np.zeros((X.shape[0], W.shape[0], W.shape[1]))
for i in range(X.shape[0]):
    for j in range(W.shape[0]):
        net_prime[i,j,:] +=  approx_fprime(W[j,:], lambda x: network(fun, x, X[i,:]), 1e-8)
print np.allclose(net_prime, network_prime(fun_prime, W, X))

# check gaussian kernel derivative w.r.t weights W
xi = X[0:1,:]
xj = x[0:1,:]
kernel_prime = approx_fprime(W.reshape(-1), lambda W: gaussian_kernel(xi,xj, W, sigma), 1e-8)
ci = network(fun, W, xi)
cj = network(fun, W, xj)
dci_dtheta = network_prime(fun_prime, W, xi)
dcj_dtheta = network_prime(fun_prime, W, xj)
print np.allclose(kernel_prime, gaussian_kernel_prime(ci, cj, dci_dtheta, dcj_dtheta, sigma, W).reshape(-1))
