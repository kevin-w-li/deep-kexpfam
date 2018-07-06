import numpy as np
import matplotlib.pyplot as plt
from LiteNet import FDTYPE
import tensorflow as tf

def plot_dataset(p, plot_size, ngrid, n=500, sample_params=dict(), dlogpdf_params = dict(), quiver_params=dict()):

    eval_grid = np.linspace(-plot_size/2,plot_size/2,ngrid) 

    eval_points = np.array([[xv,yv] + [0.01]*(p.D-2)
        for xv in eval_grid
                for yv in eval_grid])
    #eval_points = np.random.randn(ngrid, D)


    rand_train_data = p.sample(n)

    fig, axes = plt.subplots(1,2,figsize=(8,4), sharex=True, sharey=True)

    logpdf = p.logpdf(eval_points)
    dlogpdf = p.grad_multiple(eval_points)
    logpdf = logpdf.reshape(ngrid,ngrid)
    dlogpdf = dlogpdf.reshape(ngrid,ngrid,-1)
    logpdf -= logpdf.max()+20
    pdf    = np.exp(logpdf)
    pdf /= pdf.sum()

    ax = axes[0]

    ax.scatter(rand_train_data[:,0],rand_train_data[:,1], 2, "r", **sample_params)
    ax.set_xlim([-plot_size/2,plot_size/2])
    ax.set_ylim([-plot_size/2,plot_size/2])
    ax.set_aspect("equal")
    
    ax.pcolor(eval_grid, eval_grid, pdf.T,  zorder=0)
    ax.set_title("pdf")
    ax = axes[1]

    g_int = 10

    ax.pcolor(eval_grid, eval_grid, logpdf.T, **dlogpdf_params)
    if quiver_params is not None:
        ax.quiver(eval_grid[::g_int], eval_grid[::g_int], dlogpdf[::g_int,::g_int,0].T, 
                   dlogpdf[::g_int,::g_int,1].T, **quiver_params)
    ax.scatter(rand_train_data[:,0],rand_train_data[:,1], 2, "r", **sample_params)
    ax.set_title("logpdf")
    ax.set_xlim([-plot_size/2,plot_size/2])
    ax.set_ylim([-plot_size/2,plot_size/2])
    ax.set_aspect("equal")

    return fig, axes, rand_train_data, eval_grid, eval_points

def visualize_kernel(kn_model, grid_one, N, points = np.array([[0,0.0]]),**kwargs):
    
    '''
    Plot effective kernels 
    '''
    ngrid = len(grid_one)
    npoint = points.shape[0]
    D = points.shape[1]
    grid_one = grid_one.astype(FDTYPE)
    
    points = tf.constant(points, dtype=FDTYPE)
    
    grid = np.meshgrid(grid_one,grid_one)
    grid = np.stack(grid, 2).reshape(-1,2)

    grid = tf.constant(np.c_[grid, np.zeros((grid.shape[0],D-2), dtype="float32")])

    K = kn_model.kn.evaluate_gram(points, grid)
    K_eval = kn_model.sess.run(K).reshape(npoint, ngrid,ngrid)
    
    for i in range(npoint):
        
        plt.contour(grid_one, grid_one, K_eval[i], N, 
                    vmin=K_eval[:-1,:-1].min(), vmax=K_eval[:-1,:-1].max(), **kwargs)
            
    plt.gca().set_aspect("equal")
    return K_eval
