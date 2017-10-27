from Blocks import *
from LiteNet import *
import tensorflow as tf
from scipy.spatial.distance import pdist
import time
from plot_utils import *
np.set_printoptions(linewidth=140, precision=5)

test = 0

if test:

    world_dim = 9
    world_shape = [world_dim,]*2
    nsample = 10
    n_test_sample = 10
    batch_size = 5
    niter = 10
    test_iter = 10

    # set network parameters
    nfil_1 = 50
    size_1 = 5
    stride_1 = 2
    ndim_out_2 = 300
    ndim_out = 50
else:

    world_dim = 9
    world_shape = [world_dim,]*2
    nsample = 1000
    n_test_sample = 1000
    batch_size = 100
    niter = 10000
    test_iter = 50

    # set network parameters
    nfil_1 = 20
    size_1 = world_dim
    stride_1 = 1
    ndim_out_2 = 20
    ndim_out = 100

exp_param_dict = dict(
    worldshape = world_shape,
    ntrain = nsample,
    ntest = n_test_sample,
    niter = niter,
    batch_size = batch_size,
    )

exp_param_str = '_'.join(( k+str(v) for k, v in exp_param_dict.items()))

# this is used for training
world_1 = World(world_shape)
world_1.append(Rectangle([(world_dim-1)/2]*2, 5, 5))
world_1.append(Rectangle([(world_dim-1)/2]*2, 3, 3))
world_1.likelihood = Gaussian_Likelihood(1.0)
world_1.potentials.append(Collision_Potential(000))
world_1.finalize_model()

# this is used for validation
world_2 = World(world_shape)
world_2.append(Rectangle([(world_dim-1)/2]*2, 5, 5))
world_2.append(Rectangle([(world_dim-1)/2]*2, 3, 3))
world_2.likelihood = Gaussian_Likelihood(1.0)
world_2.potentials.append(Collision_Potential(-np.inf))
world_2.finalize_model()

# this is used for testing
world_3 = World(world_shape)
world_3.append(Rectangle([(world_dim-1)/2]*2, 5, 5))
world_3.append(Rectangle([(world_dim-1)/2]*2, 3, 3))
world_3.likelihood = Gaussian_Likelihood(1.0)
world_3.potentials.append(Collision_Potential(000))
world_3.finalize_model()

print 'sample val'
images_2  = world_2.sample_valid(n_test_sample)[0]
dloglik_2 = world_1.dloglik(images_2)
print 'sample test'
images_3  = world_3.sample_valid(n_test_sample)[0]
dloglik_3 = world_3.dloglik(images_3)

images_2  = images_2[:,None,:,:]
dloglik_2 = dloglik_2[:,None,:,:]
images_3  = images_3[:,None,:,:]
dloglik_3 = dloglik_3[:,None,:,:]

print images_2.mean()
print images_3.mean()

print dloglik_2.mean()
print dloglik_3.mean()

layer_1 = ConvNetwork((1,)+tuple(world_shape), nfil_1, size_1,
                            stride=stride_1,
                            batch_size = batch_size,
                            flatten=True)
ndim_2 = ((world_shape[0] - size_1 ) / stride_1 + 1)**2 * nfil_1

# layer_2 = LinearReLUNetwork((ndim_2,), (ndim_out_2,), 
#                            batch_size = batch_size, lin_grads=[1.0,1.0])

# layer_3 = LinearReLUNetwork(layer_2.ndim_out, (ndim_out,), 
#                            batch_size = batch_size, lin_grads=[0.0,1.0])

# network = DeepNetwork([layer_1, layer_2, layer_3])
network = DeepNetwork([layer_1])
network_out = network.forward_array(images_2[:200])
print network_out.shape
med = np.median(pdist(network_out))

kernel = GaussianKernel(network.ndim_out[0], sigma = 2*med**2)
# setup lite model
MSD = KernelNetMSD(kernel, network)

h, X, Y, dlogp_dx, dlogp_dy = MSD.MSD_V()


def give_data(world_data, world_model):

    count = 0 

    while True:
        if count % test_iter == 0:    
            images          = world_data.sample_valid(nsample)[0][:,None,:,:]
            dloglik         = world_model.dloglik(images[:,0,:,:])[:,None,:,:]

        rand_idx      = np.random.choice(nsample, batch_size)
        dlogp_dx_val  = dloglik[rand_idx]
        X_val         = images[rand_idx]

        rand_idx      = np.random.choice(nsample, batch_size)
        dlogp_dy_val  = dloglik[rand_idx]
        Y_val         = images[rand_idx]
         
        feed_dict = {   X: X_val,
                        Y: Y_val,
                        dlogp_dx: dlogp_dx_val, 
                        dlogp_dy: dlogp_dy_val}
        count += 1
        yield feed_dict

train_step = tf.train.MomentumOptimizer(1.0, 0.9).minimize(-h)

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True

init = tf.global_variables_initializer()

with tf.Session(config=sess_config) as sess:
    sess.run(init)
    data_gen = give_data(world_2, world_1)
    t0 = time.time()
    scores = np.zeros((niter/test_iter+1, 3))
    weights = []
    for i in range(niter):
        
        feed_dict = data_gen.next()

        sess.run(train_step, feed_dict = feed_dict) 

        # print validation and regenerate data
        if i% test_iter == 0 or i == niter:

            print '===== iter: %d out of %.d, time taken: %.3f' %( i, niter, time.time()-t0)
            t0 = time.time()
            scores[i/test_iter,0] = h.eval(feed_dict = feed_dict)
            print 'train h: %.5f' % scores[i/test_iter,0]

            h_val = []
            h_test = []
            
            # loop over all data to generate random batches and compute V-stats, and average
            for bi in range(n_test_sample/batch_size):
                for bj in range(n_test_sample/batch_size):

                    dlogp_dx_val  = dloglik_2[bi*batch_size:(bi+1)*batch_size]
                    X_val         = images_2[ bi*batch_size:(bi+1)*batch_size]
                    dlogp_dy_val  = dloglik_2[bj*batch_size:(bj+1)*batch_size]
                    Y_val         = images_2[ bj*batch_size:(bj+1)*batch_size]
                    
                    feed_dict = {   X: X_val,
                                    Y: Y_val,
                                    dlogp_dx: dlogp_dx_val, 
                                    dlogp_dy: dlogp_dy_val}
                    h_val.append(h.eval(feed_dict = feed_dict))

                    dlogp_dx_val  = dloglik_3[bi*batch_size:(bi+1)*batch_size]
                    X_val         = images_3[ bi*batch_size:(bi+1)*batch_size]
                    dlogp_dy_val  = dloglik_3[bj*batch_size:(bj+1)*batch_size]
                    Y_val         = images_3[ bj*batch_size:(bj+1)*batch_size]
                    
                    feed_dict = {   X: X_val,
                                    Y: Y_val,
                                    dlogp_dx: dlogp_dx_val, 
                                    dlogp_dy: dlogp_dy_val}
                    h_test.append(h.eval(feed_dict = feed_dict))

            val_score  = np.mean(h_val) 
            test_score = np.mean(h_test)

            scores[i/test_iter,1] = val_score
            scores[i/test_iter,2] = test_score

            print 'val   h: %.5f' % val_score
            print 'test  h: %.5f' % test_score
            print 'test time taken: %.3f' %( time.time()-t0)
            t0 = time.time()
            
        # save weights of first layer
        if ( i % (niter/5) ) == 0:
            print np.array([np.linalg.norm(MSD.network.param['layer1W'].eval()[:,:,0,i]) for i in range(nfil_1)])
            W = MSD.network.param['layer1W'].eval()
            weights.append(W)

plot_filters(weights, 'MSD/weights_%s' % exp_param_str)

labels = ['train', 'val', 'test']
fig, axes = plt.subplots(1)
ax = axes
for i in range(scores.shape[1]):
    ax.plot(scores[:,i], label=labels[i])
plt.legend(loc='best')
fig.savefig('figs/MSD/scores_%s.pdf' % exp_param_str)
