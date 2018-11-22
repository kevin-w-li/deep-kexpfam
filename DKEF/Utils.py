import tensorflow as tf
import numpy as np
from settings import FDTYPE

def construct_index(dim,s="o", n=1):
    ''' construct string for use in tf.einsum as it does not support...'''
    s = ord(s)
    return ''.join([str(unichr(i+s)) for i in range(len(dim)*n)])

c = 30
'''
nl   = lambda x: tf.log(1+tf.exp(x))
dnl  = lambda x: 1/(1+tf.exp(-x))
d2nl = lambda x: tf.exp(-x)/tf.square(1+tf.exp(-x))

'''
nl   = lambda x: tf.where(x<c, tf.log(1+tf.exp(x)), x)
dnl  = lambda x: tf.where(x<-c, tf.zeros_like(x), 1/(1+tf.exp(-x)))
d2nl = lambda x: tf.where(tf.logical_and(-c<x, x<c), tf.exp(-x)/tf.square(1+tf.exp(-x)), tf.zeros_like(x))

nl   = lambda x: tf.where(x<0, tf.zeros_like(x), 0.5*tf.square(x))
dnl  = lambda x: tf.where(x<0, tf.zeros_like(x), x)
d2nl = lambda x: tf.where(x<0, tf.zeros_like(x), tf.ones_like(x))

def pow_10(x, name, **kwargs): 

    var = tf.Variable(x, dtype=FDTYPE, name="log_" + name, **kwargs)

    return tf.pow(10.0, var, name=name)

