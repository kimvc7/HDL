# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 20:31:36 2020

@author: Michael
"""

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
  def __init__(self, num_subsets, batch_size, subset_ratio, pixels_x, pixels_y, num_channels,dropout = 0, l2 = 0):
    self.subset_size = int(subset_ratio*batch_size)
    self.num_subsets = num_subsets
    self.dropout = dropout
    self.subset_ratio = subset_ratio
    self.x_input = tf.placeholder(tf.float32, shape = [None, pixels_x, pixels_y, num_channels])
    self.y_input = tf.placeholder(tf.int64, shape = [None])
    self.dropout = dropout

    # Stability dual variable
    self.theta = tf.Variable(tf.constant(1.0))

    self.conv11 = tf.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape = (pixels_x, pixels_y, num_channels))
    self.conv12 = tf.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')
    self.maxpool1 = tf.layers.MaxPooling2D((2,2),(2,2),padding = 'same')
    self.vgg1_output = self.maxpool1(self.conv12(self.conv11(self.x_input)))
    self.conv21 = tf.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')
    self.conv22 = tf.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')
    self.maxpool2 = tf.layers.MaxPooling2D((2,2),(2,2),padding = 'same')
    self.vgg2_output = self.maxpool2(self.conv22(self.conv21(self.vgg1_output)))
    self.conv31 = tf.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')
    self.conv32 = tf.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')
    self.maxpool3 = tf.layers.MaxPooling2D((2,2),(2,2),padding = 'same')    
    self.vgg3_output = self.maxpool3(self.conv32(self.conv31(self.vgg2_output)))

    self.flatten = tf.layers.Flatten()
    
    self.vgg3_flat = self.flatten(self.vgg3_output)
    
    self.fc1 = tf.layers.Dense(128, activation='relu', kernel_initializer='glorot_uniform')

    # Perceptron's fully connected layer.
    self.h1 = tf.nn.dropout(self.fc1(self.vgg3_flat), self.dropout)


    self.fc2 = tf.layers.Dense(10, activation='relu', kernel_initializer='glorot_uniform')

    self.pre_softmax = tf.nn.dropout(self.fc2(self.h1), self.dropout)

    #Prediction 
    y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.y_input, logits=self.pre_softmax)
    self.logits = tf.nn.softmax(self.pre_softmax)
    self.xent = tf.reduce_mean(y_xent)
    self.y_pred = tf.argmax(self.pre_softmax, 1)

    #Compute objective value for max cross-entropy using dual formulation.
    self.stable_data_loss = tf.nn.relu(y_xent - self.theta)
    self.dual_xent = self.theta + 1/(self.subset_ratio) * tf.reduce_mean(self.stable_data_loss)

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

    #Evaluation
    correct_prediction = tf.equal(self.y_pred, self.y_input)
    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
