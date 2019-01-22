import tensorflow as tf
import numpy as np
from Utils import *
from BaseMeasure import *
from Network import *
from Kernel import *

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

class LiteModel:


    def __init__(self, kernel, npoint, alpha = None, points = None, 
                init_log_lam = 0.0, log_lam_weights=-3, noise_std=0.0, 
                simple_lite=False, lam = None, base=False, curve_penalty=False):
        
        self.kernel = kernel
        self.base   = base

        if simple_lite:
            assert lam is not None
            self.lam_norm = lam
            self.lam_alpha = lam
            self.lam_curve = tf.constant(0.0, dtype=FDTYPE, name="lam_curve")
            self.lam_weights = tf.constant(0.0, dtype=FDTYPE, name="lam_weights")
            self.noise_std   = tf.constant(0.0, dtype=FDTYPE, name="noise_std")
        else:
            with tf.name_scope("regularizers"):
                self.lam_norm    = pow_10(-1000, "lam_norm", trainable=False)
                self.lam_alpha   = pow_10(init_log_lam, "lam_alpha", trainable=True)
                if curve_penalty:
                    self.lam_curve   = pow_10(init_log_lam, "lam_curve", trainable=True)
                else:
                    self.lam_curve   = pow_10(-1000, "lam_curve", trainable=False)
                self.lam_weights = pow_10(log_lam_weights, "lam_weights", trainable=False)
                self.lam_kde     = pow_10(-1000, "lam_kde", trainable=False)
                self.noise_std   = noise_std

        if points is not None:
            self.set_points(points)

        if alpha is None:
            self.alpha = tf.Variable(tf.zeros(npoint), dtype=FDTYPE, trainable=False, name="kn_alpha")
        else:
            self.alpha = alpha
        

        if base:
            self.base = GaussianBase(self.ndim_in[0], 2)

    def _score_statistics(self, data, add_noise=False, take_mean=True):
        
        ''' compute the vector b and matrix C
            Y: the input data to the lite model to fit
        '''
        
        if self.noise_std > 0 and add_noise:
            data = data + self.noise_std * tf.random_normal(tf.shape(data), dtype=FDTYPE)

        d2kdx2, dkdx = self.kernel.get_sec_grad(self.X, data)
        npoint = tf.shape(self.X)[0]
        ndata  = tf.shape(data)[0]
        
        
        # score     = (alpha * H + qH) + [ (0.5 * alpha * G2 * alpha) + (alpha * G * qG) + (0.5*qG2) ]
        # curvature = (0.5 * alpha * H2 * alpha) + (alpha * HqH)  + (0.5 * qH2)

        H = tf.einsum("ijk->ij", 
                      d2kdx2)
        
        G2 = tf.einsum('ikl,jkl->ijk',
                      dkdx, dkdx)
        
        H2 = tf.einsum('ikl,jkl->ijk',
                     d2kdx2, d2kdx2)


        if take_mean:
            H = tf.reduce_mean(H,1)
            G2 = tf.reduce_mean(G2,2)
            H2 = tf.reduce_mean(H2,2)

        if self.base:

            d2qdx2, dqdx = self.base.get_sec_grad(data)

            GqG = tf.reduce_sum(dqdx * dkdx, -1)
            qG2 = tf.reduce_sum(tf.square(dqdx), -1)
            qH  = tf.reduce_sum(d2qdx2,          -1)

            HqH = tf.einsum('ijk->ij', d2qdx2*d2kdx2)
            qH2 = tf.einsum('ij->i', tf.square(d2qdx2))

        else:

            GqG= tf.zeros([npoint, ndata], dtype=FDTYPE)
            qG2= tf.zeros([1], dtype=FDTYPE)
            qH = tf.zeros([1], dtype=FDTYPE)

            HqH= tf.zeros([npoint, ndata], dtype=FDTYPE)
            qH2 = tf.zeros([1], dtype=FDTYPE)
            
        if take_mean:

            GqG  = tf.reduce_mean(GqG, -1)
            qG2 = tf.reduce_mean(qG2)
            qH = tf.reduce_mean(qH)

            HqH= tf.reduce_mean(HqH,-1)
            qH2 = tf.reduce_mean(qH2)

        return H, G2, H2, GqG, qG2, qH, HqH, qH2
    
    def individual_score(self, data, alpha=None, add_noise=False):
        
        H, G2, H2, GqG, qG2, qH, HqH, qH2 = self._score_statistics(data, add_noise=add_noise, take_mean=False)

        if alpha is None:
            alpha = self.alpha

        s2 = tf.einsum('i,ij->j', alpha, H) + qH
        s1 = 0.5 * (tf.einsum('i,ijk,j->k', alpha, G2, alpha) + qG2) + tf.einsum("i,ij->j", alpha , GqG)
        score  =  s1 + s2

        return score, H, G2, H2, GqG, qG2, qH, HqH, qH2

    def score(self, data, alpha=None, add_noise=False):

        H, G2, H2, GqG, qG2, qH, HqH, qH2 = self._score_statistics(data, add_noise=add_noise)

        if alpha is None:
            alpha = self.alpha

        s2 = tf.einsum('i,i->', alpha, H) + qH
        s1 = 0.5 * (tf.einsum('i,ij,j', alpha, G2, alpha) + qG2) + tf.einsum("i,i->", alpha, GqG)
        score  =  s1 + s2

        return score, H, G2, H2, GqG, qG2, qH, HqH, qH2

    def curve(self, data, alpha=None, add_noise=False):

        # curvature = (0.5 * alpha * H2 * alpha) + (alpha * H * qH)  + (0.5 * qH2)
        H, G2, H2, GqG, qG2, qH, HqH, qH2  = self._score_statistics(data, add_noise=add_noise)

        if alpha is None:
            alpha = self.alpha

        curve = 0.5 * tf.einsum("i,ij,j->", alpha, H2, alpha) + tf.einsum('i,i->', alpha, HqH) + 0.5 * qH2
        return curve 


    def kde_loss(self, data, kde):

        kde_delta = kde[:,None] - kde[None,:]
        delta = self.evaluate_gram(self.X, data)[:,:]
        delta = tf.einsum("i,ijk->jk", self.alpha, delta[:,:,None] - delta[:,None,:])
        if self.base:
            q0 = self.base.get_fun(data)
            delta = delta + (q0[:,None] - q0[None,:])
        loss = tf.reduce_sum(tf.square(delta - kde_delta)) / tf.cast((tf.reduce_prod(tf.shape(kde_delta))), FDTYPE)
        return loss


    def opt_alpha(self, data, kde=None, add_noise=False):
        # score     = (alpha * H + qH) + [ (0.5 * alpha * G2 * alpha) + (alpha * G * qG) + (0.5*qG2) ]
        # curvature = (0.5 * alpha * H2 * alpha) + (alpha * H * qH)  + (0.5 * qH2)

        H, G2, H2, GqG, qG2, qH, HqH, qH2 = self._score_statistics(data, add_noise=add_noise)

        ndata = tf.cast(tf.shape(data)[0], FDTYPE)

        quad =  (G2 + 
                H2 * self.lam_curve + 
                tf.eye(self.npoint, dtype=FDTYPE)*self.lam_alpha
                )
        
        lin  =  -(
                HqH * self.lam_curve + 
                H + GqG
                )
        
        if kde is not None:
            kde  = kde[:100]
            kde_delta = kde[:,None] - kde[None,:]
            if self.base:
                q0 = self.base.get_fun(data[:100])
                kde_delta = kde_delta - (q0[:,None] - q0[None,:])
            delta = self.evaluate_gram(self.X, data[:100])
            delta = delta[:,:,None] - delta[:,None,:]
            npair = tf.cast(tf.reduce_prod(tf.shape(kde_delta)), FDTYPE)
            quad  = quad + self.lam_kde * tf.einsum("mij,nij->mn", delta, delta) / npair
            lin   = lin  + self.lam_kde * tf.einsum("mij,ij->m", delta, kde_delta) / npair
        
                
        alpha = tf.matrix_solve(quad, lin[:,None])[:,0]

        return alpha, H, G2, H2, GqG, qG2, qH, HqH, qH2


    def opt_score(self, data, alpha=None, kde=None, add_noise=False):
        '''
        compute regularised score and returns a handle for assign optimal alpha
        '''
        if alpha is None:
            alpha = self.alpha

        alpha_opt, H, G2, H2, GqG, qG2, qH, HqH, qH2  = self.opt_alpha(data, kde, add_noise=add_noise)
        alpha_assign_op = tf.assign(alpha, alpha_opt)

        s2     =  tf.einsum('i,i->', alpha, H) + qH
        s1     =  0.5 * (tf.einsum('i,ij,j', alpha, G2, alpha) + qG2) + tf.einsum("i,i->", alpha, GqG)

        r_norm =  self.get_fun_rkhs_norm()
        l_norm =  self.get_fun_l2_norm()
        curve  =  0.5 * (tf.einsum('i,ij,j', alpha, H2, alpha) + qH2) + tf.einsum("i,i->", alpha, HqH)
        w_norm =  self.get_weights_norm()

        score  =  s1 + s2 + 0.5 * (#self.lam_norm  * r_norm + 
                                   self.lam_curve * curve+
                                   self.lam_alpha * l_norm
                                   )
        if kde is not None:
            score = score + 0.5 * self.lam_kde * self.kde_loss(data, kde)

        return alpha_assign_op, score
        
    def val_score(self, train_data, valid_data, test_data=None, train_kde=None, valid_kde=None, clip_score=False, add_noise=False):

        alpha, H, G2, H2, GqG, qG2, qH, HqH, qH2 = self.opt_alpha(train_data, 
                                                                        train_kde, add_noise=add_noise)
        save_alpha = tf.assign(self.alpha, alpha)

        #  ====== validation ======
        score, H, G2, H2, GqG, qG2, qH, HqH, qH2 = self.individual_score(
                                            valid_data, alpha=alpha, add_noise=add_noise)
        
        score_mean = tf.reduce_mean(score)
        score_std  = tf.sqrt(tf.reduce_mean(score**2) - score_mean**2)
        count = tf.reduce_sum(tf.cast(score < score_mean-3*score_std, "int32"))

        if clip_score:
            score = tf.clip_by_value(score, score_mean-3*score_std,np.inf)

        score = tf.reduce_mean(score)

        if test_data is not None: 
            test_score = self.score(test_data, alpha=self.alpha, add_noise=add_noise)[0]
        else:
            test_score = tf.constant(0.0, dtype=FDTYPE)

        #r_norm =  self.get_fun_rkhs_norm(self.alpha)
        r_norm = tf.constant(0.0)
        l_norm =  self.get_fun_l2_norm(self.alpha)
        curve  =  0.5 * (tf.einsum('i,ij,j', self.alpha, tf.reduce_mean(H2,2), self.alpha) + 
                    tf.reduce_mean(qH2)) + tf.einsum("i,i->", self.alpha, tf.reduce_mean(HqH,1))

        w_norm =  self.get_weights_norm()
        loss   =  score + 0.5 * (  w_norm * self.lam_weights )
        if valid_kde is not None:
            k_loss =  self.kde_loss(valid_data, valid_kde)
            loss = loss + 0.5 * self.lam_kde * k_loss
        else:   
            k_loss = tf.zeros([], dtype=FDTYPE)


        return loss, score, r_norm, l_norm, curve, w_norm, k_loss, test_score, count, \
                save_alpha
        
    def set_points(self, points):
        
        self.X = points
        self.npoint = tf.shape(points)[0]
        self.ndim_in = tuple( points.shape[1:].as_list() )
        self.K = self.kernel.get_gram_matrix(self.X, self.X)

    def evaluate_gram(self, X, Y):
        return self.kernel.get_gram_matrix(X, Y)

    def evaluate_fun(self, data, alpha=None):

        if alpha is None:
            alpha = self.alpha
        gram = self.kernel.get_gram_matrix(self.X, data)
        
        fv = tf.tensordot(alpha, gram, [[0],[0]])

        if self.base:
            fv = fv + self.base.get_fun(data)
            
        return fv

    def evaluate_grad(self, data, alpha=None):

        if alpha is None:
            alpha = self.alpha
        grad = self.kernel.get_grad(self.X, data)
        gv   = tf.tensordot(alpha, grad, axes=[[0],[0]])

        if self.base:
            gv = gv + self.base.get_grad(data)

        return gv

    def evaluate_hess(self, data, alpha=None):
        
        if alpha is None:
            alpha = self.alpha
        hess = self.kernel.get_hess(self.X, data)
        hv   = tf.tensordot(alpha, hess, axes=[[0],[0]])

        if self.base:
            hv   = hv + self.base.get_hess(data)
        return hv

    def evaluate_grad_fun(self, data, alpha=None):
        
        if alpha is None:
            alpha = self.alpha
        grad, gram = self.kernel.get_grad_gram(self.X, data)
        grad = tf.tensordot(alpha, grad, axes=[[0],[0]])
        fun  = tf.tensordot(alpha, gram, axes=[[0],[0]])

        if self.base:
            qgrad, qfun = self.base.get_grad_fun(data)
            grad = grad + qgrad
            fun  = fun  + qfun

        return grad, fun

    def evaluate_hess_grad_fun(self, data, alpha=None): 

        if alpha is None:
            alpha = self.alpha
        hess, grad, gram = self.kernel.get_hess_grad_gram(self.X, data)
        hess = tf.tensordot(alpha, hess, axes=[[0],[0]])
        grad = tf.tensordot(alpha, grad, axes=[[0],[0]])
        fun  = tf.tensordot(alpha, gram, axes=[[0],[0]])

        if self.base:
            qhess, qgrad, qfun = self.base.get_hess_grad_fun(data)
            grad = grad + qgrad
            fun  = fun  + qfun
            hess = hess + qhess

        return hess, grad, fun
        

    def get_fun_l2_norm(self, alpha=None):
        if alpha is None:
            alpha = self.alpha
        return tf.reduce_sum(tf.square(alpha))

    def get_fun_rkhs_norm(self, alpha=None):
        if alpha is None:
            alpha = self.alpha
        return tf.einsum('i,ij,j', alpha, self.K, alpha)

    def get_weights_norm(self):
        return self.kernel.get_weights_norm()

