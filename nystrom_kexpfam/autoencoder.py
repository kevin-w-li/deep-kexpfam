import numpy as np
import tensorflow as tf


class DenoisingAutoencoder():
    """
    Custom tensorflow implementation of the denoising autoencoder by Alain and Bengio.
    
    This implementation follow the paper in the sense of using tanh encoding and linear de-coding layers.
    Furthermore, we use BFGS to optimize the parameters.
    Optimization stars with a larger noise level, which is decresed in multiple steps.
    """
    def __init__(self, m, sigma_noise, max_iterations, num_noise_levels, seed=None, num_threads=1):
        self.m = m
        self.max_iterations = max_iterations
        self.sigma_noise = sigma_noise
        self.num_noise_levels = num_noise_levels
        self.seed = seed
        self.num_threads = num_threads
        
        self.tf_config = tf.ConfigProto(intra_op_parallelism_threads=self.num_threads)
        
    def _build_graph(self, X_input=None):
        m = self.m
        D = self.D
        if X_input is None:
            X_input = tf.placeholder(tf.float32, [None, D], name='X_corrupted')

        vs = {}
        vs['W'] = tf.Variable(tf.truncated_normal(shape=[D, m], stddev=0.1), name='W')
        vs['bias_hidden'] = tf.Variable(tf.constant(0.1, shape=[m]), name='bias_hidden')

        vs['V'] = tf.Variable(tf.truncated_normal(shape=[m, D], stddev=0.1), name='V')
        vs['bias_visible'] = tf.Variable(tf.constant(0.1, shape=[D]), name='bias_visible')
        
        activation_enc = tf.matmul(X_input, vs['W']) + vs['bias_hidden']
        encoded = tf.nn.tanh(activation_enc)
        activation_dec = tf.matmul(encoded, vs['V']) + vs['bias_visible']
        # return tf.nn.tanh(activation_dec)
        reconstructed = activation_dec

        return X_input, vs, reconstructed
    
    def _get_vals(self, vs, session):
        names, variables = zip(*vs.items())
        return {n: v for n, v in zip(names, session.run(variables))}

    def _feed_dict(self, vs, vals):
        return {vs[n]: val for n, val in vals.items()}

    def fit(self, X):
        self.D = D = X.shape[1]
        N = X.shape[0]
        
        tf.reset_default_graph()
        g = tf.Graph()
        with g.as_default():
            if self.seed is not None:
                tf.set_random_seed(self.seed)
            
            X_input, vs, reconstructed = self._build_graph()
            X_orig = tf.placeholder(tf.float32, [None, D], name='X_orig')
            cost = tf.sqrt(
                        tf.reduce_mean(
                        tf.square(
                        tf.subtract(X_orig, reconstructed))))
        
            optimizer = tf.contrib.opt.ScipyOptimizerInterface(cost,
                                               options={'maxiter': self.max_iterations})
            
            with tf.Session(config=self.tf_config) as session:

                session.run(tf.global_variables_initializer())

                # the last one is just 1, to set the noise level to the self.sigma_noise
                noise_inflations = 10 ** np.arange(self.num_noise_levels - 1, -1, step=-1)
                
                for noise_inflation in noise_inflations:
                    optimizer.minimize(session, feed_dict={
                        X_orig: X,
                        X_input: X + np.random.randn(N, D) * self.sigma_noise * noise_inflation,
                    })

                self.vals = self._get_vals(vs, session)
    
    def reconstruct(self, X):
        assert self.D == X.shape[1]
        
        X_input, vs, reconstructed = self._build_graph()
        feed_dict = self._feed_dict(vs, self.vals)
        feed_dict[X_input] = X
        
        with tf.Session(config=self.tf_config) as session:
            session.run(tf.global_variables_initializer())
            return session.run(reconstructed, feed_dict=feed_dict)
    
    def grad(self, X):
        reconstructed = self.reconstruct(X)
        # use last noise level which is the one for reconstruction
        return (reconstructed - X) / self.sigma_noise ** 2

    def score(self, X):
        D = X.shape[1]
        X_single = tf.placeholder(tf.float32, [D], name='X_corrupted')
        _, vs, recon = self._build_graph(tf.expand_dims(X_single, 0))
        recon = tf.squeeze(recon, 0)

        grad = (recon - X_single) / self.sigma_noise ** 2
        hess_diag = tf.concat([
            tf.gradients(grad[i], X_single)[0][i:i + 1] for i in range(D)], 0)
        score = tf.nn.l2_loss(grad) + tf.reduce_sum(hess_diag)

        feed_dict = self._feed_dict(vs, self.vals)

        with tf.Session(config=self.tf_config) as session:
            session.run(tf.global_variables_initializer())
            scores = np.empty(len(X))
            for i, x in enumerate(X):
                feed_dict[X_single] = x
                scores[i] = session.run(score, feed_dict=feed_dict)
            return scores.mean()

