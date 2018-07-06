from scipy.linalg.decomp_cholesky import cho_solve
from scipy.misc import logsumexp

import numpy as np


def log_gaussian_pdf(x, mu=None, Sigma=None, is_cholesky=False, compute_grad=False):
    if mu is None:
        mu = np.zeros(len(x))
    if Sigma is None:
        Sigma = np.eye(len(mu))
    
    if is_cholesky is False:
        L = np.linalg.cholesky(Sigma)
    else:
        L = Sigma
    
    assert len(x) == Sigma.shape[0]
    assert len(x) == Sigma.shape[1]
    assert len(x) == len(mu)
    
    # solve y=K^(-1)x = L^(-T)L^(-1)x
    x = np.array(x - mu)
    y = cho_solve((L, True), x)
    # y = solve_triangular(L, x.T, lower=True)
    # y = solve_triangular(L.T, y, lower=False)
    
    if not compute_grad:
        log_determinant_part = -np.sum(np.log(np.diag(L)))
        quadratic_part = -0.5 * x.dot(y)
        const_part = -0.5 * len(L) * np.log(2 * np.pi)
        
        return const_part + log_determinant_part + quadratic_part
    else:
        return -y
    
def log_gaussian_pdf_isotropic(x, sigma, mu=None, compute_grad=False):
    if mu is not None:
        x = x - mu
    if compute_grad:
        return -(x) / (sigma ** 2)
    else:
        D = len(x)
        const_part = -0.5 * D * np.log(2 * np.pi)
        quadratic_part = -np.dot(x, x) / (2 * (sigma ** 2))
        log_determinant_part = -D * np.log(sigma)
        return const_part + log_determinant_part + quadratic_part

def sample_gaussian(N, mu=np.zeros(2), Sigma=np.eye(2), is_cholesky=False):
    mu = np.atleast_1d(mu)
    D = len(mu)
    assert len(mu.shape) == 1
    assert len(Sigma.shape) == 2
    assert D == Sigma.shape[0]
    assert D == Sigma.shape[1]
    
    if is_cholesky is False:
        L = np.linalg.cholesky(Sigma)
    else:
        L = Sigma
    
    return L.dot(np.random.randn(D, N)).T + mu


def rings_sample(N, D, sigma=0.1, radia=np.array([1, 3])):
    assert D >= 2
    
    angles = np.random.rand(N) * 2 * np.pi
    noise = np.random.randn(N) * sigma
    
    weights = 2 * np.pi * radia
    weights /= np.sum(weights)
    
    radia_inds = np.random.choice(len(radia), N, p=weights)
    radius_samples = radia[radia_inds] + noise
    
    xs = (radius_samples) * np.sin(angles)
    ys = (radius_samples) * np.cos(angles)
    X = np.vstack((xs, ys)).T.reshape(N, 2)
    
    result = np.zeros((N, D))
    result[:, :2] = X
    if D > 2:
        result[:, 2:] = np.random.randn(N, D - 2) * sigma
    return result

def rings_log_pdf_grad(X, sigma=0.1, radia=np.array([1, 3])):
    weights = 2 * np.pi * radia
    weights /= np.sum(weights)
    
    norms = np.linalg.norm(X[:, :2], axis=1)

    result = np.zeros(np.shape(X))

    grads = []
    for i in range(len(X)):
        log_pdf_components = -0.5 * (norms[i] - radia) ** 2 / (sigma ** 2)
        log_pdf = logsumexp(log_pdf_components + np.log(weights))
        neg_log_neg_ratios = log_pdf_components - log_pdf

        gs_inner = np.zeros((len(radia), 1))
        for k in range(len(gs_inner)):
            gs_inner[k] = -(norms[i] - radia[k]) / (sigma ** 2)

        grad_1d = np.dot(gs_inner.T, np.exp(neg_log_neg_ratios + np.log(weights)))
        angle = np.arctan2(X[i, 1], X[i, 0])
        grad_2d = np.array([np.cos(angle), np.sin(angle)]) * grad_1d
        grads += [grad_2d]
    
    result[:, :2] = np.array(grads)
    if X.shape[1] > 2:
        # standard normal log pdf gradient
        result[:, 2:] = -X[:, 2:] / (sigma ** 2)
    
    return result

def rings_log_pdf(X, sigma=0.1, radia=np.array([1, 3])):

    weights = 2 * np.pi * radia
    weights /= np.sum(weights)
    
    norms = np.linalg.norm(X[:, :2], axis=1)

    result = np.zeros(np.shape(X)[0])

    for i in range(len(X)):
        log_pdf_components = -0.5 * (norms[i] - radia) ** 2 / (sigma ** 2) - \
                              0.5 * np.log(2*np.pi*sigma**2) - \
                              np.log(2*np.pi * radia)
        result[i] = logsumexp(log_pdf_components + np.log(weights))
    
    if X.shape[1] > 2:
        # stand+rd normal log pdf gradient
        result += np.sum(-0.5*np.log(2*np.pi*sigma**2) -0.5 * (X[:, 2:]**2) / (sigma ** 2),1)
    
    return result
