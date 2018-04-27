import matplotlib as mpl
import os, sys
mpl.use('Agg')
from LiteNet import *
import tensorflow as tf
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from tqdm import trange

from nystrom_kexpfam.data_generators.Ring import Ring
from nystrom_kexpfam.visualisation import visualise_array_2d

data_name="Ring"

D = 2
res=40
nrep = 10
ntrain = 500
ntest = ntrain*10
test_batch_size = ntrain*10

p = Ring(D=D, sigma=0.1, N_train=10000, N_test=10000)
def gen_data(n):
    
    rand_data = p.sample(n)
    return rand_data

X, Y = np.linspace(-2,2,res, dtype="float32"), np.linspace(-6,-2,res, dtype="float32")

np.random.seed(1)

sigma = tf.placeholder("float32", shape=[])
lam   = tf.placeholder("float32", shape=[])
alpha = tf.Variable(np.zeros(ntrain, dtype="float32"))
points = tf.placeholder(shape=(ntrain, D),  dtype="float32")
test_data = tf.placeholder("float32", shape=(test_batch_size, D), name="1")

kernel = GaussianKernel(sigma)

kn = KernelModel(kernel, alpha = alpha)
kn.set_points(points)
alpha_opt, _, score, train_data = kn.opt_score(lam=lam, lam2=lam)[:4]
gv = kn.evaluate_grad(test_data)

nX = len(X)
nY = len(Y)
scores = np.zeros((nX, nY, nrep))


config = tf.ConfigProto(device_count={"GPU":1})
config.gpu_options.allow_growth=True
init = tf.global_variables_initializer()


rand_train_data = gen_data(ntrain)
rand_test_data = gen_data(ntrain)


with tf.Session(config=config) as sess:
    
    sess.run(init)

    for xi in trange(len(X), desc="x loop", leave=True):
        for yi in trange(len(Y), desc="y loop", leave=True):
            for r in trange(nrep, desc="r loop", leave=True):
                x = X[xi]
                y = Y[yi]

                #print "\r xi=%2d yi=%2d ri=%2d" % (xi, yi, r),

                rand_train_data = gen_data(ntrain)
                feed = {train_data: rand_train_data.astype("float32"), 
                        points:rand_train_data.astype("float32"),
                        sigma: 10**(x), lam:10**(y)}
                
                sess.run(alpha_opt, feed_dict=feed)

                nbatch = ntest/test_batch_size
                
                for i in range(nbatch):
                    d = gen_data(test_batch_size)
                    feed[test_data] = d
                    gt = p.grad_multiple(d)
                    ge = gv.eval(feed_dict=feed)
                    scores[xi,yi,r] += 0.5 * np.mean(np.sum((gt-ge)**2,1))
                scores[xi,yi,r] /= nbatch
    alpha_val = alpha.eval()

scores = scores.mean(2)

best_ind = np.unravel_index(np.nanargmin(scores), scores.shape)
fig,ax = plt.subplots(figsize=(6,5))
visualise_array_2d(X, Y, np.log10(scores), ax=ax)
dX = X[1]-X[0]
dY = Y[1]-Y[0]
plt.plot(X[best_ind[0]]+dX/2, Y[best_ind[1]]+dY/2, 'ro', markersize=15)

ax.set_title("D=%d, score, best=%.2f\nbest X=%.4f, best Y=%.4f\nexp values: %.4f, %.4f" % \
                     (D, np.nanmin(scores), X[best_ind[0]], Y[best_ind[1]],
                        10**(X[best_ind[0]]), 10**(Y[best_ind[1]]))
            )
ax.set_xlabel(r"log $\sigma$" )
ax.set_ylabel(r"log $\lambda$")

file_name = "%s_D%d_rep%d_res%d" % (data_name, D, nrep, res)

fig.savefig("figs/param_search/"+file_name+".pdf")

with h5.File('results/param_search/'+file_name+".h5", 'w') as f:
    f.create_dataset("scores", data = scores)
    f.create_dataset("sigs", data = X)
    f.create_dataset("lams", data = Y)


    
