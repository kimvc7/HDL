# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import json

W = [None, None, None, None, None, None, None, None, None]


class Model(object):
  def __init__(self, num_subsets, batch_size, cnn_size, fc_size, subset_ratio, pixels_x, pixels_y, num_channels, dropout = 0, l2 = 0, theta = True, eps=0, weights = W):
    self.subset_size = int(subset_ratio*batch_size)
    self.num_subsets = num_subsets
    self.dropout = dropout
    self.subset_ratio = subset_ratio
    self.x_input = tf.placeholder(tf.float32, shape = [None, pixels_x, pixels_y, num_channels])
    self.y_input = tf.placeholder(tf.int64, shape = [None])
    self.dropout = dropout
    self.regularizer = tf.keras.regularizers.l2(l2)
    self.eps =  eps


    #Weight Initializers
    c11 = tf.constant_initializer(weights[0]) if weights[0] is not None else 'he_uniform'
    c12 = tf.constant_initializer(weights[1]) if weights[1] is not None else 'he_uniform'
    c21 = tf.constant_initializer(weights[2]) if weights[2] is not None else 'he_uniform'
    c22 = tf.constant_initializer(weights[3]) if weights[3] is not None else 'he_uniform'
    c31 = tf.constant_initializer(weights[4]) if weights[4] is not None else 'he_uniform'
    c32 = tf.constant_initializer(weights[5]) if weights[5] is not None else 'he_uniform'
    fc1 = tf.constant_initializer(weights[6]) if weights[6] is not None else 'glorot_uniform'
    fc2 = tf.constant_initializer(weights[7]) if weights[7] is not None else 'glorot_uniform'
    theta_val = tf.constant(weights[8], dtype=tf.float32) if weights[8] is not None else tf.constant(1.0)

    # Stability dual variable
    self.theta = tf.Variable(theta_val, trainable = theta)

    self.conv11 = tf.layers.Conv2D(cnn_size, (3, 3), activation='relu', kernel_initializer=c11, padding='same', kernel_regularizer = self.regularizer, input_shape = (pixels_x, pixels_y, num_channels))
    self.conv12 = tf.layers.Conv2D(cnn_size, (3, 3), activation='relu', kernel_initializer=c12, padding='same', kernel_regularizer = self.regularizer)
    self.maxpool1 = tf.layers.MaxPooling2D((2,2),(2,2),padding = 'same')
    self.vgg1_output = self.maxpool1(self.conv12(self.conv11(self.x_input)))
    self.conv21 = tf.layers.Conv2D(cnn_size, (3, 3), activation='relu', kernel_initializer=c21, padding='same', kernel_regularizer = self.regularizer)
    self.conv22 = tf.layers.Conv2D(cnn_size, (3, 3), activation='relu', kernel_initializer=c22, padding='same', kernel_regularizer = self.regularizer)
    self.maxpool2 = tf.layers.MaxPooling2D((2,2),(2,2),padding = 'same')
    self.vgg2_output = self.maxpool2(self.conv22(self.conv21(self.vgg1_output)))
    self.conv31 = tf.layers.Conv2D(cnn_size, (3, 3), activation='relu', kernel_initializer=c31, padding='same', kernel_regularizer = self.regularizer)
    self.conv32 = tf.layers.Conv2D(cnn_size, (3, 3), activation='relu', kernel_initializer=c32, padding='same', kernel_regularizer = self.regularizer)
    self.maxpool3 = tf.layers.MaxPooling2D((2,2),(2,2),padding = 'same')    
    self.vgg3_output = self.maxpool3(self.conv32(self.conv31(self.vgg2_output)))

    self.flatten = tf.layers.Flatten()
    
    self.vgg3_flat = self.flatten(self.vgg3_output)
    
    self.fc1 = tf.layers.Dense(fc_size, activation='relu', kernel_initializer=fc1, kernel_regularizer = self.regularizer)

    # Perceptron's fully connected layer.
    self.h1 = tf.nn.dropout(self.fc1(self.vgg3_flat), self.dropout)


    self.fc2 = tf.layers.Dense(10, activation='sigmoid', kernel_initializer=fc2, kernel_regularizer = self.regularizer)

    self.pre_softmax = tf.nn.dropout(self.fc2(self.h1), self.dropout)
    self.regularizer = 0


    #Compute linear approximation for robust cross-entropy.
    data_range = tf.range(tf.shape(self.y_input)[0])
    indices = tf.map_fn(lambda n: tf.stack([tf.cast(self.y_input[n], tf.int32), n]), data_range)
    pre_softmax_t = tf.transpose(self.pre_softmax)
    self.nom_exponent = pre_softmax_t -  tf.gather_nd(pre_softmax_t, indices)
    
    sum_exps=0
    for i in range(10):
      grad = tf.gradients(self.nom_exponent[i], self.x_input)
      exponent = self.eps*tf.reduce_sum(tf.abs(grad[0]), axis=1) + self.nom_exponent[i]
      sum_exps+=tf.exp(exponent)
    robust_y_xent = tf.log(sum_exps)
    self.robust_xent = tf.reduce_mean(robust_y_xent)

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


    #Compute objective value for stable robust cross-entropy using dual formulation.
    self.rob_stable_data_loss = tf.nn.relu(robust_y_xent - self.theta)
    self.robust_stable_xent = self.theta + 1/(self.subset_ratio) * tf.reduce_mean(self.rob_stable_data_loss)

    #Evaluation
    correct_prediction = tf.equal(self.y_pred, self.y_input)
    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
