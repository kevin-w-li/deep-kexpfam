import numpy as np
import numpy.linalg as nl

def log_sum_exp(X):
    """
    Computes log sum_i exp(X_i).
    Useful if you want to solve log \int f(x)p(x) dx
    where you have samples from p(x) and can compute log f(x)
    """
    X0 = X.min()
    X_without_X0 = np.delete(X, X.argmin())
    
    return X0 + np.log(1. + np.sum(np.exp(X_without_X0 - X0)))

def log_mean_exp(X):
    """
    Computes log 1/n sum_i exp(X_i).
    Useful if you want to solve log \int f(x)p(x) dx
    where you have samples from p(x) and can compute log f(x)
    """
    
    return log_sum_exp(X) - np.log(len(X))

def avg_prob_of_log_probs(X):
    """
    Given a set of log-probabilities, this computes log-mean-exp of them.
    Careful checking is done to prevent buffer overflows
    Similar to calling (but overflow-safe): log_mean_exp(log_prob)
    """
    
    # extract inf inds (no need to delete X0 from X here)
    X0 = X.min()
    inf_inds = np.isinf(np.exp(X - X0))
    
    # remove these numbers
    X_without_inf = X[~inf_inds]
    
    # return exp-log-mean-exp on shortened array
    avg_prob_without_inf = np.exp(log_mean_exp(X_without_inf))
    
    # re-normalise by the full length, which is equivalent to adding a zero probability observation
    renormaliser = float(len(X_without_inf)) / len(X)
    avg_prob_without_inf = avg_prob_without_inf * renormaliser
    
    return avg_prob_without_inf


def qmult(b):
    """
    QMULT  Pre-multiply by random orthogonal matrix.
       QMULT(A) is Q*A where Q is a random real orthogonal matrix from
       the Haar distribution, of dimension the number of rows in A.
       Special case: if A is a scalar then QMULT(A) is the same as
                     QMULT(EYE(A)).

       Called by RANDSVD.

       Reference:
       G.W. Stewart, The efficient generation of random
       orthogonal matrices with an application to condition estimators,
       SIAM J. Numer. Anal., 17 (1980), 403-409.
    """
    try:
        n = b.shape[0]
        a = b.copy()
    except AttributeError:
        n = b
        a = np.eye(n)

    d = np.zeros(n)

    for k in range(n - 2, -1, -1):
        # Generate random Householder transformation.
        x = np.random.randn(n - k)
        s = nl.norm(x)
        # Modification to make sign(0) == 1
        sgn = np.sign(x[0]) + float(x[0] == 0)
        s = sgn * s
        d[k] = -sgn
        x[0] = x[0] + s
        beta = s * x[0]

        # Apply the transformation to a
        y = np.dot(x, a[k:n, :])
        a[k:n, :] = a[k:n, :] - np.outer(x, (y / beta))

    # Tidy up signs.
    for i in range(n - 1):
        a[i, :] = d[i] * a[i, :]

    # Now randomly change the sign (Gaussian dist)
    a[n - 1, :] = a[n - 1, :] * np.sign(np.random.randn())

    return a

def score(est_grad, true_grad):
    ndim = est_grad.ndim
    assert ndim == true_grad.ndim and ndim <= 2 and ndim > 0
    
    if ndim == 2:
        return np.mean(np.sum((est_grad - true_grad) ** 2, axis=1))
    else:
        return np.sum((est_grad - true_grad) ** 2)

def hypercube(D):
    if D <= 1:
        return [[0], [1]]
    else:
        so_far = hypercube(D - 1)
        result = [each + [0] for each in so_far]
        result += [each + [1] for each in so_far]
        return result
