import numpy as np

def support_1d(fun, x):
    assert 1<=x.ndim<=2
    return fun(x) if x.ndim == 2 else fun(x[None,:])[0]

def get_grid(r, i, j, cond):


    grid = np.meshgrid(r,r)

    grid = np.stack(grid,2)
    grid = grid.reshape(-1,2)
    
    num_point = len(grid)
    grid_cond = np.tile(cond[None,:], [num_point, 1])
    
    grid_cond[:,i] = grid[:,0]
    grid_cond[:,j] = grid[:,1]
    return grid_cond

