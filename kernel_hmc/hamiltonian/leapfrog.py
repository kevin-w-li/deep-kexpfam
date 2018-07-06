from kernel_hmc.densities.gaussian import sample_gaussian
import numpy as np


def leapfrog(q, dlogq, p, dlogp, step_size=0.3, num_steps=1):
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

def leapfrog_no_storing(q, dlogq, p, dlogp, step_size=0.3, num_steps=1):
    # create copy of state
    p = np.array(p.copy())
    q = np.array(q.copy())
    
    # half momentum update
    p = p - (step_size / 2) * -dlogq(q)
    
    # alternate full variable and momentum updates
    for i in range(num_steps):
        q = q + step_size * -dlogp(p)

        # precompute since used for two half-steps
        dlogq_eval = dlogq(q)

        #  first half momentum update
        p = p - (step_size / 2) * -dlogq_eval
        
        # second half momentum update
        if i != num_steps - 1:
            p = p - (step_size / 2) * -dlogq_eval

    return q, p

def leapfrog_friction_habc_no_storing(c, V, q, dlogq, p, dlogp, step_size=0.3, num_steps=1):
    """
    MATLAB code by Chen et al
    
    function [ newx ] = sghmc( U, gradU, m, dt, nstep, x, C, V )
    %% SGHMC using gradU, for nstep, starting at position x
    
    p = randn( size(x) ) * sqrt( m );
    B = 0.5 * V * dt; 
    D = sqrt( 2 * (C-B) * dt );
    
    for i = 1 : nstep
        p = p - gradU( x ) * dt  - p * C * dt  + randn(1)*D;
        x = x + p./m * dt;
    end
    newx = x;
    end
    """
    
    # friction term (as in HABC)
    D = len(q)
    B = 0.5 * V * step_size
    C = np.eye(D) * c + V
    L_friction = np.linalg.cholesky(2 * step_size * (C - B))
    zeros_D = np.zeros(D)
    
    # create copy of state
    p = np.array(p.copy())
    q = np.array(q.copy())
    
    # alternate full momentum and variable updates
    for _ in range(num_steps):
        friction = sample_gaussian(N=1, mu=zeros_D, Sigma=L_friction, is_cholesky=True)[0]

        # just like normal momentum update but with friction
        p = p - step_size * -dlogq(q) - step_size * C.dot(-dlogp(p)) + friction

        # normal position update
        q = q + step_size * -dlogp(p)
        

    return q, p
