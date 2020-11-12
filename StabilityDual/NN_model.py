"""
The model is a multiclass perceptron for 10 classes.
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import json

from l0_regularization import *#get_l0_norm



class Model(object):
  def __init__(self, num_subsets, batch_size, l1_size, l2_size, subset_ratio, num_features, dropout = 0, l2 = 0, l0 = 0, eps=0, reg_stability = 0):
    self.subset_size = int(subset_ratio*batch_size)
    self.num_subsets = num_subsets
    self.dropout = dropout
    self.subset_ratio = subset_ratio
    self.x_input = tf.placeholder(tf.float32, shape = [None, num_features])
    self.y_input = tf.placeholder(tf.int64, shape = [None])
    self.eps =  eps

    # Stability dual variable
    self.theta = tf.Variable(tf.constant(0.1))

    # Perceptron's fully connected layer.
    self.W1 = self._weight_variable([num_features, l1_size])
    self.b1 = self._bias_variable([l1_size])

    if l0 > 0:

      #self.mask_W1 = get_l0_mask(self.W1, "W1")
      self.W1_masked = tf.Variable(tf.zeros(self.W1.get_shape()), name="W1_m")#self._weight_variable([num_features, l1_size])
      self.W1_masked, self.l0_norm_W1 = get_l0_norm_full(self.W1, "W1")
      #self.W1 = self.W1 + self.W1_masked
      self.h1 = tf.nn.relu(tf.matmul(self.x_input, self.W1_masked) + self.b1)
    else:

      self.h1 = tf.nn.relu(tf.matmul(self.x_input, self.W1) + self.b1)
      self.h1 = tf.nn.dropout(self.h1, self.dropout)

    self.W2 = self._weight_variable([l1_size, l2_size])
    self.b2 = self._bias_variable([l2_size])


    if l0 > 0:
      self.mask_W2 = get_l0_mask(self.W2, "W2")
      self.W2_masked, self.l0_norm_W2 = get_l0_norm_full(self.W2, "W2")
      self.h2 = tf.nn.relu(tf.matmul(self.h1, self.W2_masked) + self.b2)
    else:
      self.h2 = tf.nn.relu(tf.matmul(self.h1, self.W2) + self.b2)
      self.h2 = tf.nn.dropout(self.h2, self.dropout)

    self.W3 = self._weight_variable([l2_size, 10])
    self.b3 = self._bias_variable([10])

    if l0 > 0:
      self.mask_W3 = get_l0_mask(self.W3, "W3")
      self.W3_masked, self.l0_norm_W3 = get_l0_norm_full(self.W3, "W3")
      self.pre_softmax = tf.matmul(self.h2, self.W3_masked) + self.b3

    else:
      self.pre_softmax = tf.matmul(self.h2, self.W3) + self.b3

    #Prediction 
    y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.y_input, logits=self.pre_softmax)
    self.logits = tf.nn.softmax(self.pre_softmax)
    self.xent = tf.reduce_mean(y_xent)
    self.y_pred = tf.argmax(self.pre_softmax, 1)

    #Compute objective value for robust cross-entropy.
    data_range = tf.range(tf.shape(self.y_input)[0])
    indices = tf.map_fn(lambda n: tf.stack([tf.cast(self.y_input[n], tf.int32), n]), data_range)
    pre_softmax_t = tf.transpose(self.pre_softmax)
    self.nom_exponent = pre_softmax_t -  tf.gather_nd(pre_softmax_t, indices)

    sum_exps = 0
    for i in range(10):
      grad = tf.gradients(self.nom_exponent[i], self.x_input)
      exponent = eps*tf.reduce_sum(tf.abs(grad[0]), axis=1) + self.nom_exponent[i]
      sum_exps+=tf.exp(exponent)

    robust_y_xent = tf.log(sum_exps)
    self.robust_xent = tf.reduce_mean(robust_y_xent)

    #Compute objective value for stable cross-entropy using dual formulation.
    self.stable_data_loss = tf.nn.relu(y_xent - self.theta)
    self.dual_xent = self.theta + 1/(self.subset_ratio) * tf.reduce_mean(self.stable_data_loss)

    #Compute objective value for stable robust cross-entropy using dual formulation.
    self.rob_stable_data_loss = tf.nn.relu(robust_y_xent - self.theta)
    self.robust_stable_xent = self.theta + 1/(self.subset_ratio) * tf.reduce_mean(self.rob_stable_data_loss)

    #Compute objective value for max cross-entropy using MC formulation.
    max_subset_xent = 0.0

    for k in range(self.num_subsets):
      perm = np.arange(batch_size)
      np.random.shuffle(perm)
      subset_y_xent = tf.gather(y_xent, perm[:self.subset_size])
      max_subset_xent = tf.maximum(max_subset_xent, tf.reduce_mean(subset_y_xent))

    self.MC_xent = max_subset_xent

    #Compute regularizer
    #self.regularizer = #l2*(tf.reduce_sum(tf.square(self.b2))+ tf.reduce_sum(tf.square(self.b1)) +
                        #tf.reduce_sum(tf.square(self.b3)))#+tf.reduce_sum(tf.square(self.W1)) +
                           #tf.reduce_sum(tf.square(self.W2)+tf.reduce_sum(tf.square(self.W3))))
    self.regularizer = l2*(tf.reduce_sum(tf.square(self.W1)) + tf.reduce_sum(tf.square(self.W2))
                              + tf.reduce_sum(tf.square(self.W3)))
    if l0 > 0:
      self.regularizer = self.regularizer + l0 * (self.l0_norm_W1 + self.l0_norm_W2 + self.l0_norm_W3)

    if reg_stability > 0 :
      self.regularizer = self.regularizer + reg_stability * tf.math.reduce_std(self.h2)
    #Evaluation
    correct_prediction = tf.equal(self.y_pred, self.y_input)
    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


  @staticmethod
  def _weight_variable(shape):
      initial = tf.glorot_uniform_initializer()
      return tf.get_variable(shape=shape, initializer=initial, name=str(np.random.randint(1e10)))

  @staticmethod
  def _bias_variable(shape):
      initial = tf.constant(0.1, shape = shape)
      return tf.Variable(initial)

