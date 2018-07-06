import numpy as np


def autocorr(x):
    """
    Computes the ( normalised) auto-correlation function of a
    one dimensional sequence of numbers.
    
    Utilises the numpy correlate function that is based on an efficient
    convolution implementation.
    
    Inputs:
    x - one dimensional numpy array
    
    Outputs:
    Vector of autocorrelation values for a lag from zero to max possible
    """
    
    # normalise, compute norm
    xunbiased = x - np.mean(x)
    xnorm = np.sum(xunbiased ** 2)
    
    # convolve with itself
    acor = np.correlate(xunbiased, xunbiased, mode='same')
    
    # use only second half, normalise
    acor = acor[len(acor) / 2:] / xnorm
    
    return acor