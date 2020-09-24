"""
The model is a multiclass perceptron for 10 classes.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import json


class Model(object):
  def __init__(self, num_subsets, subset_batch_size):
    self.x_input = tf.placeholder(tf.float32, shape = [None, 784])
    self.y_input = tf.placeholder(tf.int64, shape = [None])

    N = tf.shape(self.x_input)[0]

    # Perceptron's fully connected layer.
    self.W1 = self._weight_variable([784, 512])
    self.b1 = self._bias_variable([512])
    self.h1 = tf.nn.relu(tf.matmul(self.x_input, self.W1) + self.b1)

    self.W2 = self._weight_variable([512, 256])
    self.b2 = self._bias_variable([256])
    self.h2 = tf.nn.relu(tf.matmul(self.h1, self.W2) + self.b2)

    self.W3 = self._weight_variable([256, 10])
    self.b3 = self._bias_variable([10])
    self.pre_softmax = tf.matmul(self.h2, self.W3) + self.b3

    #Prediction 
    y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.y_input, logits=self.pre_softmax)
    self.logits = tf.nn.softmax(self.pre_softmax)
    self.xent = tf.reduce_sum(y_xent)
    self.y_pred = tf.argmax(self.pre_softmax, 1)

    #Split cross entropies for each subset and get maximum
    split_filter = tf.reshape(tf.ones(subset_batch_size), [1, subset_batch_size, 1,1])
    split_strides = [1, 1, subset_batch_size, 1]
    splits_xent = tf.nn.conv2d(tf.reshape(y_xent, [1, 1, N, 1]), split_filter, split_strides, 'SAME')
    self.max_xent = tf.reduce_max(splits_xent)

    #Evaluation
    correct_prediction = tf.equal(self.y_pred, self.y_input)
    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


  @staticmethod
  def _weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1, seed=0)
      return tf.Variable(initial)

  @staticmethod
  def _bias_variable(shape):
      initial = tf.constant(0.1, shape = shape)
      return tf.Variable(initial)

