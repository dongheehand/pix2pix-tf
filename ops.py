import tensorflow as tf
import numpy as np


def _batch_norm(name, x, dim, BN_decay = 0.999, BN_epsilon = 1e-3):
    
    beta = tf.get_variable(name = name + "beta", shape = dim, dtype = tf.float32,
                           initializer = tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable(name + "gamma", dim, tf.float32, 
                            initializer = tf.constant_initializer(1.0, tf.float32))
    
    mean, variance = tf.nn.moments(x, axes = [0, 1, 2])

    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_epsilon)

def _conv(name, x, filter_size, in_filters, out_filters,  strides = 2, batch_norm = True):

    with tf.variable_scope(name):
        
        kernel = tf.get_variable('filter', [filter_size, filter_size, in_filters, out_filters],tf.float32, initializer = tf.random_normal_initializer(stddev = 0.02))
        bias = tf.get_variable('bias',[out_filters],tf.float32, initializer = tf.random_normal_initializer(stddev = 0.02))
        
        x = tf.nn.conv2d(x, kernel, [1,strides,strides,1], padding='SAME') + bias
        
        if batch_norm:
            x = _batch_norm('batch_norm', x, out_filters)
        
        return x

def _deconv(name, x, filter_size, in_filters, out_filters, fraction = 2, drop_out = False, drop_out_rate = 0.5, batch_norm = True):
    
    with tf.variable_scope(name):
        
        kernel = tf.get_variable('filter', [filter_size, filter_size, out_filters, in_filters], tf.float32, initializer=tf.random_normal_initializer(stddev = 0.02) )
        bias = tf.get_variable('bias', [out_filters], tf.float32, initializer = tf.random_normal_initializer(stddev = 0.02))
        size = tf.shape(x)
        output_shape = tf.stack([size[0], size[1] * fraction, size[2] * fraction, out_filters])
        
        x = tf.nn.conv2d_transpose(x, kernel, output_shape, [1, fraction, fraction, 1], padding = "SAME")
        if batch_norm :
            x = _batch_norm('batch_norm', x, out_filters)
        if drop_out :
            x = tf.nn.dropout(x, drop_out_rate)
        
        return x

