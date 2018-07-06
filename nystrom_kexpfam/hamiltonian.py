import numpy as np
from nystrom_kexpfam.density import log_gaussian_pdf, sample_gaussian
from nystrom_kexpfam.mathematics import log_mean_exp
from nystrom_kexpfam.visualisation import visualise_array_2d


def compute_avg_acceptance(q0, logq, dlogq, sigma_p, num_steps, step_size, plot=False,
                           X=None, est_grad=None, plot_i=None, plot_j=None, ax=None, plot_only_trajectory=False):
    """
    Computes HMC trajectory using provided gradient handle, and computes acceptance
    rate using provided log_pdf handle.
    
    HMC momentum is an isotropic Gaussian with specified variance.
    
    @param q0 starting state
    @param logq log_pdf handle
    @param dlogq gradient of log_pdf handle
    @param sigma_p standard deviation momentium
    @param num_steps number of leapfrog steps
    @param step_size step size in leapfrog integrator
    @param plot visualise trajectory (all following parameters need to be provided)
    """
    
    D = len(q0)
    
    # momentum
    L_p = np.linalg.cholesky(np.eye(D) * sigma_p)
    logp = lambda x: log_gaussian_pdf(x, Sigma=L_p, compute_grad=False, is_cholesky=True)
    dlogp = lambda x: log_gaussian_pdf(x, Sigma=L_p, compute_grad=True, is_cholesky=True)
    p_sample = lambda: sample_gaussian(N=1, mu=np.zeros(D), Sigma=L_p, is_cholesky=True)[0]
    
    # starting state
    p0 = p_sample()
    
    # integrate HMC trajectory
    Qs, Ps = leapfrog(q0, dlogq, p0, dlogp, step_size, num_steps)
    
    # compute average acceptance probabilities (using true log_pdf, not estimated one)
    log_acc = compute_log_accept_pr(q0, p0, Qs, Ps, logq, logp)
    acc_mean = np.exp(log_mean_exp(log_acc))
    
    if plot:
        import matplotlib.pyplot as plt
        if ax is None:
            fig = plt.figure(figsize=(16, 4))
            ax = fig.add_subplot(111)
            
        x_min = np.min([np.min(Qs[:, plot_i]), np.min(X[:, plot_i])])
        x_max = np.max([np.max(Qs[:, plot_i]), np.max(X[:, plot_i])])
        y_min = np.min([np.min(Qs[:, plot_j]), np.min(X[:, plot_j])])
        y_max = np.max([np.max(Qs[:, plot_j]), np.max(X[:, plot_j])])
        if not plot_only_trajectory:
            # build grid that covers trajectory and data
            Xs = np.linspace(x_min, x_max)
            Ys = np.linspace(y_min, y_max)
            XX, YY = np.meshgrid(Xs, Ys)
            X_grid = np.array([XX.ravel(), YY.ravel()]).T
            
            # compute estimated gradients on grid
            grad_grid = est_grad(X_grid)
            grad_grid_norm = np.linalg.norm(grad_grid, axis=1)
            
            visualise_array_2d(Xs, Ys, grad_grid_norm.reshape(len(Ys), len(Xs)).T,
                               samples=X, ax=ax)
        else:
            ax.plot(X[:, plot_i], X[:, plot_j], 'b.')
        ax.plot(Qs[:, plot_i], Qs[:, plot_j], 'r-')
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.plot(q0[plot_i], q0[plot_j], "r*", markersize=15)
        ax.plot(Qs[-1, plot_i], Qs[-1, plot_j], "b*", markersize=15)
        ax.set_title("Estimated gradient of log-density and HMC trajectory from random data point")
        ax.set_xlabel("Component %d" % plot_i)
        ax.set_ylabel("Component %d" % plot_j)
        plt.show()
        
    return acc_mean

def leapfrog(q, dlogq, p, dlogp, step_size, num_steps):
    # for storing trajectory
    Ps = np.zeros((num_steps + 1, len(p)))
    Qs = np.zeros(Ps.shape)
    
    # create copy of state
    p = np.array(p.copy())
    q = np.array(q.copy())
    Ps[0] = p
    Qs[0] = q
    
    # half momentum update
    p = p - (step_size / 2) * -dlogq(q)
    
    # alternate full variable and momentum updates
    for i in range(num_steps):
        q = q + step_size * -dlogp(p)
        Qs[i + 1] = q

        # precompute since used for two half-steps
        dlogq_eval = dlogq(q)

        #  first half momentum update
        p = p - (step_size / 2) * -dlogq_eval
        
        # store p as now fully updated
        Ps[i + 1] = p
        
        # second half momentum update
        if i != num_steps - 1:
            p = p - (step_size / 2) * -dlogq_eval

    return Qs, Ps

def compute_hamiltonian(Qs, Ps, logq, logp):
    assert len(Ps) == len(Qs)
    return np.asarray([-logq(Qs[i]) - logp(Ps[i]) for i in range(len(Qs))])

def compute_log_accept_pr(q0, p0, Qs, Ps, logq, logp):
    H0 = compute_hamiltonian(np.atleast_2d(q0), np.atleast_2d(p0), logq, logp)
    H = compute_hamiltonian(Qs, Ps, logq, logp)
    
    return np.minimum(np.zeros(H.shape), -H + H0)

