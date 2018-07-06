import numpy as np

def assert_array_shape(a, ndim=None, shape=None, dims={}):
    if not type(a) is np.ndarray:
        raise TypeError("Provided object type (%s) is not nunpy.array." % str(type(a)))
    
    if ndim is not None:
        if not a.ndim == ndim:
            raise ValueError("Provided array dimensions (%d) are not as expected (%d)." % (a.ndim, ndim))
    
    if shape is not None:
        if not np.all(a.shape==shape):
            raise ValueError("Provided array size (%s) are not as expected (%s)." % (str(a.shape), shape))
    
    for k,v in dims.items():
        if not a.shape[k] == v:
            raise ValueError("Provided array's %d-th dimension's size (%d) is not as expected (%d)." % (k, a.shape[k], v))

def assert_positive_int(i):
    if not issubclass(type(i), np.int):
        raise TypeError("Provided argument (%s) must be npumpy.int." % str(type(i)))
    
    if not i>0:
        raise ValueError("Provided integer (%d) must be positive." % i)

def assert_positive_float(f):
    if not issubclass(type(f), np.float):
        raise TypeError("Provided argument (%s) must be numpy.float." % str(type(f)))
    
    if not f>0:
        raise ValueError("Provided float (%f) must be positive." % f)

def assert_implements_log_pdf_and_grad(density, assert_log_pdf=True, assert_grad=True):
    if assert_log_pdf:
        if not hasattr(density, 'log_pdf') or not callable(density.log_pdf):
            raise ValueError("Density object does not implement log_pdf method")
    
    if assert_grad:
        if not hasattr(density, 'grad') or not callable(density.grad):
            raise ValueError("Density object does not implement grad method")

def assert_inout_log_pdf_and_grad(density, D, assert_log_pdf=True, assert_grad=True):
    x = np.random.randn(D)
    
    if assert_log_pdf:
        result = density.log_pdf(x)
        
        if not issubclass(type(result), np.float):
            raise ValueError("Density object's log_pdf does not return numpy.float64 but %s" % str(type(result)))
    
    if assert_grad:
        grad = density.grad(x)
        assert_array_shape(grad, ndim=1, shape=(D,))