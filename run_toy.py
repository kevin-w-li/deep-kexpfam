import sys

from Datasets import * 

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, linewidth=120)
from LiteModels import DeepLite
from Datasets import *

from tqdm import trange
import seaborn as sns
import pandas as pd

from maf import experiments as ex

from KCEF.estimators.kcef import KCEF_Gaussian
from KCEF.tools import median_heuristic
from Utils import get_grid

idx_i, idx_j = 0,1

ngrid = 100
eval_grid = np.linspace(-8,8,ngrid)

cond_values = np.zeros(2)
eval_points = get_grid(eval_grid, idx_i, idx_j, cond_values)

n_hiddens = 30
n_layers = 5
n_comps = 10

act_fun = 'tanh'
mode = 'random'

N = 5000
seed = 12
niter = 10000
ntrain = 100
nvalid = 100
npoint = 200
patience =30
nneuron = 15
nlayer = 3

dname = sys.argv[1]

p = load_data(sys.argv[1],D=2,  N=N, seed=seed, rotate=dname=="funnel")
   
grid_points  = get_grid(np.linspace(-5,5,4), 0, 1, np.zeros(2))
grid_data_one= np.linspace(-8,8,100)
grid_data    = get_grid(grid_data_one, idx_i, idx_j, np.zeros(2))

print "====================================" 
print p.name
# deep lite
dl_model = DeepLite(p, npoint=npoint, nlayer=nlayer, nneuron=nneuron, init_log_lam=-3, points_std=0.0, keep_prob=1.0, init_weight_std=1.0, noise_std=0.0, points_type="fixed", log_lam_weights=-6, step_size=1e-3, mixture_kernel=False, init_log_sigma=np.linspace(0,1,1), base=True, niter=niter, ntrain=ntrain, nvalid=nvalid, patience=patience, seed=10,gpu_count=1, fn_ext = "curve")
res = dl_model.fit(niter=niter, ntrain=ntrain, nvalid=nvalid, nbatch=1, patience=patience)
dl_model.set_test(True, 0) 
dl_model.fit_alpha()
dl_model.logZ = None
dl_model.estimate_normaliser(n=2*10**6, batch_size=10**5, std=p.test_data.std()*2.0)
test_loglik = dl_model.estimate_data_lik(p.test_data, batch_size=p.test_data.shape[0])
dl_model_loglik = test_loglik.mean()
print "dl_model", dl_model_loglik

fv = dl_model.fun_multiple(eval_points)
dl_model_logpdf = fv.copy() - fv.max()
dl_model_kernel_vals = dl_model.sess.run(dl_model.ops["kernel_grams"][0], feed_dict={dl_model.test_points: grid_points, dl_model.test_data: grid_data})
np.savez("figs/dl_model_%s_data"%p.name, 
    eval_points = eval_points, eval_grid = eval_grid,
    grid_points=grid_points, rid_data_one=grid_data_one, grid_data = grid_data,
    dl_model_kernel_vals=dl_model_kernel_vals, dl_model_logpdf=dl_model_logpdf, dl_model_loglik=dl_model_loglik)
    

'''
# simple_lite
sl_model = DeepLite(p, npoint=npoint, nlayer=0, nneuron=nneuron, init_log_lam=-3, points_std=0.0, keep_prob=1.0, init_weight_std=1.0, noise_std=0.0,
                    points_type="fixed", log_lam_weights=-6, step_size=1e-3, mixture_kernel=False, init_log_sigma=np.linspace(0,1,1), base=True,
                    niter=niter, ntrain=ntrain, nvalid=nvalid, patience=patience, seed=seed,gpu_count=1)
res = sl_model.fit(niter=niter, ntrain=ntrain, nvalid=nvalid,ntest=500, nbatch=1, patience=patience)

sl_model.set_test(True, 0) 
sl_model.fit_alpha(5000)
    
sl_model.logZ = None
sl_model.estimate_normaliser(n=2*10**6, batch_size=10**5, std=p.test_data.std()*2.0)
test_loglik = sl_model.estimate_data_lik(p.test_data, batch_size=p.test_data.shape[0])
sl_model_loglik = test_loglik.mean()
print "simple lite", sl_model_loglik
fv = sl_model.fun_multiple(eval_points)
sl_model_logpdf = fv.copy() - fv.max()
sl_model_kernel_vals = sl_model.sess.run(sl_model.ops["kernel_grams"][0], feed_dict={sl_model.test_points: grid_points, sl_model.test_data: grid_data})

np.savez("figs/sl_model_%s_data"%p.name, 
    eval_points = eval_points, eval_grid = eval_grid,
    grid_points=grid_points, rid_data_one=grid_data_one, grid_data = grid_data,
    sl_model_kernel_vals=sl_model_kernel_vals, sl_model_logpdf=sl_model_logpdf, sl_model_loglik=sl_model_loglik)
    

# linear_lite
ll_model = DeepLite(p, npoint=npoint, nlayer=nlayer, nneuron=50, init_log_lam=-3, points_std=0.0, keep_prob=1.0, init_weight_std=1.0, noise_std=0.0,
                    points_type="fixed", log_lam_weights=-6, step_size=1e-3, mixture_kernel=False, init_log_sigma=np.linspace(0,1,1), base=True,
                    niter=niter, ntrain=ntrain, nvalid=nvalid, patience=patience, seed=seed,gpu_count=1, kernel_type="linear")
res = ll_model.fit(niter=niter, ntrain=ntrain, nvalid=nvalid,ntest=500, nbatch=1, patience=patience)

ll_model.set_test(True, 0) 
ll_model.fit_alpha(5000)
    
ll_model.logZ = None
ll_model.estimate_normaliser(n=2*10**6, batch_size=10**5, std=p.test_data.std()*2.0)
test_loglik = ll_model.estimate_data_lik(p.test_data, batch_size=p.test_data.shape[0])
ll_model_loglik = test_loglik.mean()
print "linear lite", ll_model_loglik
fv = ll_model.fun_multiple(eval_points)
ll_model_logpdf = fv.copy() - fv.max()
ll_model_kernel_vals = ll_model.sess.run(ll_model.ops["kernel_grams"][0], feed_dict={ll_model.test_points: grid_points, ll_model.test_data: grid_data})

np.savez("figs/ll_model_%s_data"%p.name, 
    eval_points = eval_points, eval_grid = eval_grid,
    grid_points=grid_points, rid_data_one=grid_data_one, grid_data = grid_data,
    ll_model_kernel_vals=ll_model_kernel_vals, ll_model_logpdf=ll_model_logpdf, ll_model_loglik=ll_model_loglik)
    
# huge_lite
hl_model = DeepLite(p, npoint=npoint, nlayer=nlayer, nneuron=50, init_log_lam=-3, points_std=0.0, keep_prob=1.0, init_weight_std=1.0, noise_std=0.0,
                    points_type="fixed", log_lam_weights=-6, step_size=1e-3, mixture_kernel=False, init_log_sigma=np.linspace(0,1,1), base=True,
                    niter=niter, ntrain=ntrain, nvalid=nvalid, patience=patience, seed=seed,gpu_count=1)
res = hl_model.fit(niter=niter, ntrain=ntrain, nvalid=nvalid,ntest=500, nbatch=1, patience=patience)

hl_model.set_test(True, 0) 
hl_model.fit_alpha(5000)
    
hl_model.logZ = None
hl_model.estimate_normaliser(n=2*10**6, batch_size=10**5, std=p.test_data.std()*2.0)
test_loglik = hl_model.estimate_data_lik(p.test_data, batch_size=p.test_data.shape[0])
hl_model_loglik = test_loglik.mean()
print "huge lite", hl_model_loglik
fv = hl_model.fun_multiple(eval_points)
hl_model_logpdf = fv.copy() - fv.max()
hl_model_kernel_vals = hl_model.sess.run(hl_model.ops["kernel_grams"][0], feed_dict={hl_model.test_points: grid_points, hl_model.test_data: grid_data})

np.savez("figs/hl_model_%s_data"%p.name, 
    eval_points = eval_points, eval_grid = eval_grid,
    grid_points=grid_points, rid_data_one=grid_data_one, grid_data = grid_data,
    hl_model_kernel_vals=hl_model_kernel_vals, hl_model_logpdf=hl_model_logpdf, hl_model_loglik=hl_model_loglik)

# KCEF
est = KCEF_Gaussian(graph_type = 'full', d = p.D, graph = None)
res = est.optimize_cv_score(p.data[:1000])
est.update_parameters(res)
est.fit(p.data[:1000])
kcef_loglik = est.logpdf_multiple(p.test_data).mean()
print "kcef", kcef_loglik
fv = est.logpdf_multiple(eval_points)
kcef_logpdf = fv.copy() - fv.max()

np.savez("figs/kcef_%s_data"%p.name, 
    eval_points = eval_points, eval_grid = eval_grid,
    grid_points=grid_points, rid_data_one=grid_data_one, grid_data = grid_data,
    kcef_logpdf=kcef_logpdf, kcef_loglik=kcef_loglik)
    

# maf 
maf_data_obj = ex.load_data(p.name.lower(), D=p.D, noise_std=0, seed=seed, itanh=False, whiten=False, N=N, rotate=dname=="funnel")


mog_made_model = ex.train_mog_made([n_hiddens]*2, act_fun, n_comps, mode)
ex.save_model(mog_made_model, "mog_made", mode, [n_hiddens]*2, act_fun, n_comps, False)
mog_made_model = ex.load_model("mog_made", mode, [n_hiddens]*2, act_fun, n_comps, False)
mog_made_pdf = mog_made_model.eval(eval_points)
mog_made_loglik = mog_made_model.eval(p.test_data).mean()
print "mog_made %.5f" % mog_made_loglik

made_model = ex.train_made([n_hiddens]*2, act_fun, mode)
ex.save_model(made_model, "made", mode, [n_hiddens]*2, act_fun, n_comps, False)
made_model = ex.load_model("made", mode, [n_hiddens]*2, act_fun, n_comps, False)
made_pdf = made_model.eval(eval_points)
made_loglik = made_model.eval(p.test_data).mean()
print "made %.5f" % made_loglik

nvp_model =ex.train_realnvp([n_hiddens]*2, 'tanh', 'relu', n_layers)
ex.save_model(nvp_model, "nvp", mode, [n_hiddens]*2, act_fun, n_comps, False)
nvp_model = ex.load_model("nvp", mode, [n_hiddens]*2, act_fun, n_comps, False)
nvp_pdf = nvp_model.eval(eval_points)
nvp_loglik = nvp_model.eval(p.test_data).mean()
print "nvp %.5f" % nvp_loglik


maf_model = ex.train_maf([n_hiddens]*2, act_fun, n_layers, mode)
ex.save_model(maf_model, "maf", mode, [n_hiddens]*2, act_fun, n_comps, False)
maf_model = ex.load_model("maf", mode, [n_hiddens]*2, act_fun, n_comps, False)
maf_pdf = maf_model.eval(eval_points)
maf_loglik = maf_model.eval(p.test_data).mean()
print "maf %.5f" % maf_loglik

mog_maf_model = ex.train_maf_on_made([n_hiddens]*2, act_fun, n_layers, n_comps, mode)
ex.save_model(mog_maf_model, "mog_maf", mode, [n_hiddens]*2, act_fun, n_comps, False)
mog_maf_model = ex.load_model("mog_maf", mode, [n_hiddens]*2, act_fun, n_comps, False)
mog_maf_pdf = mog_maf_model.eval(eval_points)
mog_maf_loglik = mog_maf_model.eval(p.test_data).mean()
print "mog_maf %.5f" % mog_maf_loglik

np.savez("figs/others_%s_data"%p.name, 
    eval_points = eval_points, eval_grid = eval_grid,
    grid_points=grid_points, rid_data_one=grid_data_one, grid_data = grid_data,
    made_pdf=made_pdf, made_mog_pdf=mog_made_pdf, nvp_pdf=nvp_pdf, maf_pdf=maf_pdf, maf_mog_pdf=mog_maf_pdf,
    made_loglik=made_loglik, made_mog_loglik=mog_made_loglik, nvp_loglik=nvp_loglik, maf_loglik=maf_loglik, maf_mog_loglik=mog_maf_loglik,)

np.savez("figs/%s_data"%p.name, 
    eval_points = eval_points, eval_grid = eval_grid,
    grid_points=grid_points, rid_data_one=grid_data_one, grid_data = grid_data,
    dl_model_kernel_vals=dl_model_kernel_vals, dl_model_logpdf=dl_model_logpdf, dl_model_loglik=dl_model_loglik,
    sl_model_kernel_vals=sl_model_kernel_vals, sl_model_logpdf=sl_model_logpdf, sl_model_loglik=sl_model_loglik,
    ll_model_kernel_vals=ll_model_kernel_vals, ll_model_logpdf=ll_model_logpdf, ll_model_loglik=ll_model_loglik,
    hl_model_kernel_vals=hl_model_kernel_vals, hl_model_logpdf=hl_model_logpdf, hl_model_loglik=hl_model_loglik,
    kcef_logpdf=kcef_logpdf, kcef_loglik=kcef_loglik,
    made_pdf=made_pdf, made_mog_pdf=mog_made_pdf, nvp_pdf=nvp_pdf, maf_pdf=maf_pdf, maf_mog_pdf=mog_maf_pdf,
    made_loglik=made_loglik, made_mog_loglik=mog_made_loglik, nvp_loglik=nvp_loglik, maf_loglik=maf_loglik, maf_mog_loglik=mog_maf_loglik,)
'''
