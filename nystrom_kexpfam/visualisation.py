import numpy as np
import matplotlib.pyplot as plt
from itertools import product

def visualise_array_2d(Xs, Ys, A, samples=None, ax=None):
    # visualise found fit
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    vmin = np.nanmin(A)
    vmax = np.nanmax(A)
    heatmap = ax.pcolor(Xs, Ys, A.T, cmap='viridis', vmin=vmin, vmax=vmax)
    heatmap.cmap.set_under('white')
    
    colorbar = plt.colorbar(heatmap, ax=ax)
    colorbar.set_clim(vmin=vmin, vmax=vmax)
    
    if samples is not None:
        ax.scatter(samples[:, 0], samples[:, 1], c='r', s=1);

def pdf_grid(Xs, Ys, est, ref_vec=None, x_ind=0, y_ind=1, kind="grad_norm"):
    n_x = len(Xs)
    n_y = len(Ys)
    
    if ref_vec is None or len(ref_vec) == 2:
        X_test = np.array(list(product(Xs, Ys)))
    else:
        assert len(ref_vec) >= 2
        X_test = np.tile(ref_vec, (n_x * n_y, 1))
        X_test[:, np.array([x_ind, y_ind])] = np.array(list(product(Xs, Ys)))

    est.set_data(X_test.T)
    
    if kind == "grad_norm":
        gradients = est.grad_multiple()
        if not ref_vec is None:
            gradients = gradients[np.array([x_ind, y_ind]), :]

        gradient_norms = np.sum(gradients ** 2, axis=0)
        return gradient_norms.reshape(n_x, n_y)
    elif kind == "log_pdf":
        log_pdfs = est.log_pdf_multiple()
        return log_pdfs.reshape(n_x, n_y)
    else:
        raise ValueError("Wrong kind: %s" % kind)


def visualise_fit_2d(est, X=None, Xs=None, Ys=None, res=50, ref_vec=None,
                     x_ind=0, y_ind=1, ax=None, kind="grad_norm"):
    
    # visualise found fit
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    if Xs is None:
        x_min = -5
        x_max = 5
        if not X is None:
            x_min = np.min(X[:, x_ind])
            x_max = np.max(X[:, x_ind])
            delta = x_max - x_min
            x_min -= delta / 10.
            x_max += delta / 10.


    if Ys is None:
        y_min = -5
        y_max = 5
        if not X is None:
            y_min = np.min(X[:, y_ind])
            y_max = np.max(X[:, y_ind])
            delta = y_max - y_min
            y_min -= delta / 10.
            y_max += delta / 10.
            
    xy_max = np.max([x_max, y_max])
    xy_min = np.min([x_min, y_min])
    Xs = np.linspace(xy_min, xy_max, res)
    Ys = np.linspace(xy_min, xy_max, res)

    G = pdf_grid(Xs, Ys, est, ref_vec, x_ind, y_ind, kind)
     
    """
    plt.subplot(121)
    visualise_array_2d(Xs, Ys, D, X[:,np.array([x_ind,y_ind])])
    plt.axes().set_aspect('equal')
    plt.title("log pdf")
    #plt.colorbar()
    
    plt.subplot(122)
    """
     
    visualise_array_2d(Xs, Ys, G, X[:, np.array([x_ind, y_ind])], ax=ax)
    # plt.colorbar()
 
    
