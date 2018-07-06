import numpy as np

def support_1d(fun, x):
    assert 1<=x.ndim<=2
    return fun(x) if x.ndim == 2 else fun(x[None,:])[0]
