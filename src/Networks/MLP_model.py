"""
The model is a multiclass perceptron.
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import math
import json
from utils_init import *
from utils_nn_model import *


def sigmoid(x):
    return float(1./(1.+np.exp(-x)))

mask_initial_value=0.0
scaling = 1. / sigmoid(mask_initial_value)


class Model(object):
  def __init__(self, num_classes, batch_size, network_size, pool_size, subset_ratio, num_features, dropout = 1, l2 = 0, l0 = 0, rho=0, image_size=None, ticket=False, stored_weights=None):
    self.dropout = dropout
    self.subset_ratio = subset_ratio
    self.rho =  rho
    self.l2 = l2
    self.ticket = ticket

    w_vars, b_vars, stable_var, sparse_vars = init_vars(len(network_size) + 1)
    weights, biases, stab_weight, sparse_weights = init_weights(w_vars, b_vars, stable_var, sparse_vars)
    if stored_weights is not None:
      weights, biases, stab_weight, sparse_weights = reset_stored_weights(stored_weights)


    layer_sizes = [num_features] + network_size + [num_classes]
    layer_names = ["x_input"] + [str("h") + str(l) for l in range(len(network_size)+1)]
    mask_names = [w_vars[l] + str("_masked") for l in range(len(w_vars))]
    norm_names = [w_vars[l] + str("_norm") for l in range(len(w_vars))]


    self.x_input = tf.placeholder(tf.float32, shape = [None, num_features])
    self.y_input = tf.placeholder(tf.int64, shape = [None])
    self.temp = tf.placeholder(tf.float32)



    # DEFINE VARIABLES

    #Stability Variable
    setattr(self, stable_var, self._bias_variable([], stab_weight)) 

    for i in range(len(w_vars)):
      #Matrix Variables
      setattr(self, w_vars[i], self._weight_variable([layer_sizes[i] , layer_sizes[i+1]], weights[i])) 
      #Vector Variables
      setattr(self, b_vars[i], self._bias_variable([layer_sizes[i+1]], biases[i])) 
      #Sparsity Varibles
      setattr(self, sparse_vars[i], self.init_mask_weight([layer_sizes[i] , layer_sizes[i+1]], sparse_weights[i]))

      

    # DEFINE LAYERS
    for l in range(len(w_vars)):

      if l0 > 0:

        #Compute mask
        mask = None

        if self.ticket:
          mask = tf.cast(tf.math.greater(tf.constant([mask_initial_value]), getattr(self, sparse_vars[l])), tf.float32)
        else:
          mask = scaling*tf.sigmoid(self.temp * getattr(self, sparse_vars[l]))

        #Compute masked weights and l1 norm
        W_masked = tf.multiply(getattr(self, w_vars[l]), mask)
        W_norm = tf.reduce_sum(mask)
        setattr(self, mask_names[l], W_masked)
        setattr(self, norm_names[l], W_norm)

        if l < len(w_vars) -1:
          hidden_layer = tf.nn.relu(tf.matmul(getattr(self, layer_names[l]), getattr(self, mask_names[l])) + getattr(self, b_vars[l]))
          setattr(self, layer_names[l+1], hidden_layer) 
        else:
          hidden_layer = tf.matmul(getattr(self, layer_names[l]), getattr(self, mask_names[l])) + getattr(self, b_vars[l])
          setattr(self, layer_names[l+1], hidden_layer) 

      else:
        
        if l < len(w_vars) -1:
          hidden_layer = tf.nn.relu(tf.matmul(getattr(self, layer_names[l]), getattr(self, w_vars[l])) + getattr(self, b_vars[l]))
          setattr(self, layer_names[l+1], hidden_layer)
        else:
          hidden_layer = tf.matmul(getattr(self, layer_names[l]), getattr(self, w_vars[l])) + getattr(self, b_vars[l])
          setattr(self, layer_names[l+1], hidden_layer)
          
     
      if l < len(network_size):
        setattr(self, layer_names[l+1], tf.nn.dropout(getattr(self,  layer_names[l+1]), self.dropout)) 


    #Compute standard Cross Entropy
    self.pre_softmax = getattr(self, layer_names[len(layer_sizes)-1])
    y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_input, logits=self.pre_softmax)
    self.logits = tf.nn.softmax(self.pre_softmax)
    self.xent = tf.reduce_mean(y_xent)


    #Compute robust cross-entropy.
    data_range = tf.range(tf.shape(self.y_input)[0])
    indices = tf.map_fn(lambda n: tf.stack([tf.cast(self.y_input[n], tf.int32), n]), data_range)
    pre_softmax_t = tf.transpose(self.pre_softmax)
    self.nom_exponent = pre_softmax_t -  tf.gather_nd(pre_softmax_t, indices)

    sum_exps = 0
    for i in range(num_classes):
      grad = tf.gradients(self.nom_exponent[i], self.x_input)
      exponent = rho*tf.reduce_sum(tf.abs(grad[0]), axis=1) + self.nom_exponent[i]
      sum_exps+=tf.exp(exponent)
    robust_y_xent = tf.log(sum_exps)
    self.robust_xent = tf.reduce_mean(robust_y_xent)

    #Compute stable cross-entropy.
    self.stable_data_loss = tf.nn.relu(y_xent - getattr(self, stable_var))
    self.stable_xent = getattr(self, stable_var) + 1/(self.subset_ratio) * tf.reduce_mean(self.stable_data_loss)

    #Compute stable robust cross-entropy.
    self.rob_stable_data_loss = tf.nn.relu(robust_y_xent - getattr(self, stable_var))
    self.robust_stable_xent = getattr(self, stable_var) + 1/(self.subset_ratio) * tf.reduce_mean(self.rob_stable_data_loss)

    #Compute regularization terms
    l0_regularizer = 0
    l2_regularizer = 0

    for i in range(len(w_vars)):
      if l0 > 0:
        l0_regularizer += getattr(self, norm_names[i])
        l2_regularizer += tf.reduce_sum(tf.square(getattr(self, mask_names[i])))
      else:
        l2_regularizer += tf.reduce_sum(tf.square(getattr(self, w_vars[i])))

    self.regularizer = l2*l2_regularizer + l0*l0_regularizer


    #Evaluation
    self.y_pred = tf.argmax(self.pre_softmax, 1)
    correct_prediction = tf.equal(self.y_pred, self.y_input)
    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    self.gr = tf.gradients(self.xent, self.log_a_W1)



  @staticmethod
  def _weight_variable(shape, initial = None):
      if initial is None:
        W0 = tf.glorot_uniform_initializer()
        return tf.get_variable(shape=shape, initializer=W0, name=str(np.random.randint(1e10)))

      else:
        W0 = tf.constant(initial, shape = shape, dtype=tf.float32)
        return tf.Variable(W0)

  @staticmethod
  def _bias_variable(shape, initial = None):
      if initial is None:
        b0 = tf.constant(0.1, shape = shape)
        return tf.Variable(b0)
      else:
        b0 = tf.constant(initial, shape = shape, dtype=tf.float32)
        return tf.Variable(b0)
  

  @staticmethod
  def init_mask_weight(shape, initial = None):
      if initial is None:
        a0 = tf.constant(mask_initial_value, shape = shape, dtype=tf.float32)
        #a0 = tf.Variable(tf.random_normal(shape, mean=0.0, stddev=0.01))
        return tf.Variable(a0)
      else:
        a0 = tf.constant(initial, shape = shape, dtype=tf.float32)
        return tf.Variable(a0)



  def hard_sigmoid(x):
    return tf.minimum(tf.maximum(x, tf.zeros_like(x)), tf.ones_like(x))

