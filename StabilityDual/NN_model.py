"""
The model is a multiclass perceptron for 10 classes.
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import math

#from l0_regularization import *#get_l0_norm

limit_0 = -0.1
limit_1 = 1.1
temperature = 2 / 3
epsilon = 1e-6
lambd = 1
W = [[None, None], [None, None], [None, None], None, [None, None, None]]

class Model(object):
  def __init__(self, num_classes, num_subsets, batch_size, l1_size, l2_size, subset_ratio, num_features, dropout = 1, l2 = 0, l0 = 0, eps=0, reg_stability = 0, weights = W):
    self.subset_size = int(subset_ratio*batch_size)
    self.num_subsets = num_subsets
    self.dropout = dropout
    self.subset_ratio = subset_ratio
    self.x_input = tf.placeholder(tf.float32, shape = [None, num_features])
    self.y_input = tf.placeholder(tf.int64, shape = [None])
    self.eps =  eps
    self.l2 = l2
    W_1, b_1 = weights[0]
    W_2, b_2 = weights[1]
    W_3, b_3 = weights[2]
    theta = weights[3]
    log_a1, log_a2, log_a3 = weights[4]


    # Stability dual variable
    self.theta = self._bias_variable([], theta)

    # Perceptron's fully connected layer.
    self.W1 = self._weight_variable([num_features, l1_size], W_1)
    self.b1 = self._bias_variable([l1_size], b_1)

    # For L0 reg
    # initialize log a from normal distribution
    #### TODO ####
    self.log_a_W1 = self._log_a_variable(self.W1.get_shape(), log_a1) # , name="log_a_W1")

    if l0 > 0:
      self.W1_masked, self.l0_norm_W1 = self.get_l0_norm_full(self.W1, self.log_a_W1)
      self.h1 = tf.nn.relu(tf.matmul(self.x_input, self.W1_masked) + self.b1)

    #### END TODO ####
    else:
      self.h1 = tf.nn.relu(tf.matmul(self.x_input, self.W1) + self.b1)
    self.h1 = tf.nn.dropout(self.h1, self.dropout)

    self.W2 = self._weight_variable([l1_size, l2_size], W_2)
    self.b2 = self._bias_variable([l2_size], b_2)
    #self.log_a_W2 = self._weight_variable([l1_size, l2_size])#tf.get_variable(tf.random_normal(self.W2.get_shape(), mean=0.0, stddev=0.01))#, name="log_a_W2")

    #### TODO ####
    self.log_a_W2 = self._log_a_variable(self.W2.get_shape(), log_a2)

    if l0 > 0:
      self.W2_masked, self.l0_norm_W2 = self.get_l0_norm_full(self.W2, self.log_a_W2)
      self.h2 = tf.nn.relu(tf.matmul(self.h1, self.W2_masked) + self.b2)

    #### END TODO ####
    else:
      self.h2 = tf.nn.relu(tf.matmul(self.h1, self.W2) + self.b2)
    self.h2 = tf.nn.dropout(self.h2, self.dropout)

    self.W3 = self._weight_variable([l2_size, num_classes], W_3)
    self.b3 = self._bias_variable([num_classes], b_3)

    #### TODO ####
    self.log_a_W3 = self._log_a_variable(self.W3.get_shape(), log_a3)#, name="log_a_W3")

    if l0 > 0:
      self.W3_masked, self.l0_norm_W3 = self.get_l0_norm_full(self.W3, self.log_a_W3)
      self.pre_softmax = tf.matmul(self.h2, self.W3_masked) + self.b3
    #### END TODO ####
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
    for i in range(num_classes):
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
    self.regularizer = self.l2*(tf.reduce_sum(tf.square(self.W1)) + tf.reduce_sum(tf.square(self.W2))
                              + tf.reduce_sum(tf.square(self.W3)))
    #### TODO ####
    if l0 > 0:
      self.regularizer = l0 * (self.l0_norm_W1 + self.l0_norm_W2 + self.l0_norm_W3)
      if l2 > 0:
        self.regularizer += self.l2 * (tf.reduce_sum(tf.square(self.W1_masked)) + tf.reduce_sum(tf.square(self.W2_masked))
              + tf.reduce_sum(tf.square(self.W3_masked)))



    #### END TODO ####

    if reg_stability > 0 :
      self.regularizer = self.regularizer + reg_stability * tf.math.reduce_std(self.h2)
    #Evaluation
    correct_prediction = tf.equal(self.y_pred, self.y_input)
    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    if l0 > 0:
      tf.summary.histogram("Weights W1", self.W1_masked)
      tf.summary.histogram("Weights W2", self.W2_masked)
      tf.summary.histogram("Weights W3", self.W3_masked)
      tf.summary.scalar("W1 absolute value avg", tf.reduce_mean(tf.math.abs(self.W1_masked)))
      tf.summary.scalar("W1 absolute value std", tf.math.reduce_std(tf.math.abs(self.W1_masked)))
      tf.summary.scalar("W2 absolute value avg", tf.reduce_mean(tf.math.abs(self.W2_masked)))
      tf.summary.scalar("W2 absolute value std", tf.math.reduce_std(tf.math.abs(self.W2_masked)))
      tf.summary.scalar("W3 absolute value avg", tf.reduce_mean(tf.math.abs(self.W3_masked)))
      tf.summary.scalar("W3 absolute value std", tf.math.reduce_std(tf.math.abs(self.W3_masked)))
    else:
      tf.summary.histogram("Weights W1", self.W1)
      tf.summary.histogram("Weights W2", self.W2)
      tf.summary.histogram("Weights W3", self.W3)
      tf.summary.scalar("W1 absolute value avg", tf.reduce_mean(tf.math.abs(self.W1)))
      tf.summary.scalar("W1 absolute value std", tf.math.reduce_std(tf.math.abs(self.W1)))
      tf.summary.scalar("W2 absolute value avg", tf.reduce_mean(tf.math.abs(self.W2)))
      tf.summary.scalar("W2 absolute value std", tf.math.reduce_std(tf.math.abs(self.W2)))
      tf.summary.scalar("W3 absolute value avg", tf.reduce_mean(tf.math.abs(self.W3)))
      tf.summary.scalar("W3 absolute value std", tf.math.reduce_std(tf.math.abs(self.W3)))

    tf.summary.histogram("Pre_softmax Test", self.pre_softmax)
    tf.summary.histogram("Post_softmax Test", self.logits)
    tf.summary.histogram("Post_softmax Norm Test", tf.norm(self.logits, axis=1))

    tf.summary.scalar("Accuracy Test", self.accuracy)
    tf.summary.scalar("Xent Loss Test", self.xent)
    tf.summary.scalar("MC_Xent Loss Test", self.MC_xent)
    tf.summary.scalar("Dual Xent Loss Test", self.dual_xent)


    self.summary = tf.summary.merge_all()


  @staticmethod
  def _weight_variable(shape, initial = None):
      if initial is None:
        W0 = tf.glorot_uniform_initializer()
        return tf.get_variable(shape=shape, initializer=W0, name=str(np.random.randint(1e10)))
      #in case for normal init
      #initial = tf.glorot_normal_initializer()
      #return tf.get_variable(shape=shape, initializer=initial, name=str(np.random.randint(1e10)))
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
  def _log_a_variable(shape, initial = None):
      if initial is None:
        a0 = tf.Variable(tf.random_normal(shape, mean=0.0, stddev=0.01))
        return a0
      else:
        a0 = tf.constant(initial, shape = shape, dtype=tf.float32)
        return tf.Variable(a0)

#### TODO ####
  def get_l0_norm_full(self, x, log_a):

    shape = x.get_shape()

    # sample u
    # TODO Change 1e-6
    u = tf.random_uniform(shape)

    # compute hard concrete distribution
    # i.e., implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution
    y = tf.sigmoid((tf.log(u) - tf.log(1.0 - u) + log_a) / temperature)
    # stretch hard concrete distribution
    s_bar = y * (limit_1 - limit_0) + limit_0

    # compute differentiable l0 norm ; cdf_qz
    # Implements the CDF of the 'stretched' concrete distribution
    #if self.l2 > 0:
      #q0 = tf.clip_by_value(
        #tf.sigmoid(temperature * math.log(-limit_0 / limit_1)-log_a),
        #epsilon, 1-epsilon)
      #logpw_col = -tf.reduce_sum(- (.5 * self.l2 * tf.square(x)) - lambd, 0)
      #l0_norm = tf.reduce_sum((1 - q0) * logpw_col)
    #  #logpb = -tf.reduce_sum((1 - q0) * (.5 * l2 * tf.pow(self.bias, 2) - lambd))
    #  #logpw #+ logpb
    #else:
    l0_norm = tf.reduce_sum(tf.clip_by_value(
        tf.sigmoid(log_a - temperature * math.log(-limit_0 / limit_1)),
        epsilon, 1-epsilon))

    # get mask for calculating sparse version of tensor
    mask = hard_sigmoid(s_bar)

    # return masked version of tensor and l0 norm
    return tf.multiply(x, mask), l0_norm

def hard_sigmoid(x):
    return tf.minimum(tf.maximum(x, tf.zeros_like(x)), tf.ones_like(x))

#### END TODO ####
