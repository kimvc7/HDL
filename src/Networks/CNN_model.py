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


#Model Initialization
limit_0, limit_1, temperature, epsilon, lambd = init_sparsity_constants()

class Model(object):
	def __init__(self, num_classes, batch_size, cnn_size, pool_size, subset_ratio, num_features, dropout = 1, l2 = 0, l0 = 0, rho=0, image_size=None, stored_weights=None):
		self.dropout = dropout
		self.subset_ratio = subset_ratio
		self.rho =  rho
		self.l2 = l2

		w_vars, b_vars, stable_var, sparse_vars = init_vars(len(cnn_size) + 1)
		weights, biases, stab_weight, sparse_weights = init_weights(w_vars, b_vars, stable_var, sparse_vars)
		if stored_weights is not None:
			weights, biases, stab_weight, sparse_weights = reset_stored_weights(stored_weights)

		if image_size is None:
			image_size = [int(math.sqrt(num_features)), int(math.sqrt(num_features)), 1]

		reduced_size = [image_size[0], image_size[1]]
		for pool in pool_size:
			if pool:
				reduced_size = [math.ceil(reduced_size[0]/2), math.ceil(reduced_size[1]/2)]


		layer_sizes = [image_size] + cnn_size + [[num_classes]]
		layer_names = ["x_image"] + [str("h") + str(l) for l in range(len(cnn_size)+1)]
		mask_names = [w_vars[l] + str("_masked") for l in range(len(w_vars))]
		norm_names = [w_vars[l] + str("_norm") for l in range(len(w_vars))]

		self.x_input = tf.placeholder(tf.float32, shape = [None, num_features])
		self.y_input = tf.placeholder(tf.int64, shape = [None])
		self.x_image = tf.reshape(self.x_input, [-1, image_size[0], image_size[1], image_size[2]])


		# DEFINE VARIABLES

		#Stability Variable
		setattr(self, stable_var, self._bias_variable([], stab_weight)) 

		for i in range(len(w_vars)):


			if len(layer_sizes[i+1])>1:
				#Matrix Variables
				setattr(self, w_vars[i], self._weight_variable([layer_sizes[i+1][0] , layer_sizes[i+1][1], layer_sizes[i][2], layer_sizes[i+1][2]], weights[i])) 
				#Vector Variables
				setattr(self, b_vars[i], self._bias_variable([layer_sizes[i+1][2]], biases[i])) 
				#Sparsity Varibles
				setattr(self, sparse_vars[i], self._log_a_variable([layer_sizes[i+1][0] , layer_sizes[i+1][1], layer_sizes[i][2], layer_sizes[i+1][2]], sparse_weights[i]))

			else:

				#First Dense Layer
				if len(layer_sizes[i])>1:
					#Matrix Variables
					setattr(self, w_vars[i], self._weight_variable([reduced_size[0]*reduced_size[1]*layer_sizes[i][2], layer_sizes[i+1][0]], weights[i])) 
					#Vector Variables
					setattr(self, b_vars[i], self._bias_variable([layer_sizes[i+1][0]], biases[i])) 
					#Sparsity Varibles
					setattr(self, sparse_vars[i], self._log_a_variable([reduced_size[0]*reduced_size[1]*layer_sizes[i][2], layer_sizes[i+1][0]], sparse_weights[i]))

				#Other dense layers
				else:

					#Matrix Variables
					setattr(self, w_vars[i], self._weight_variable([layer_sizes[i][0] , layer_sizes[i+1][0]], weights[i])) 
					#Vector Variables
					setattr(self, b_vars[i], self._bias_variable([layer_sizes[i+1][0]], biases[i])) 
					#Sparsity Varibles
					setattr(self, sparse_vars[i], self._log_a_variable([layer_sizes[i][0] , layer_sizes[i+1][0]], sparse_weights[i]))

			

		# DEFINE LAYERS
		previous_layer = layer_names[0]
		for l in range(len(w_vars)):

			if l < len(cnn_size) and 0 < l:
				if len(layer_sizes[l+1]) == 1 and len(layer_sizes[l]) >1: #Flatten for first dense layer
					flat_layer = tf.reshape(getattr(self, previous_layer), [-1, reduced_size[0]*reduced_size[1]*layer_sizes[l][2]])
					setattr(self, "flat", flat_layer) 
					previous_layer = "flat"

			if l0 > 0:
				W_masked, W_norm = self.get_l0_norm_full(getattr(self, w_vars[l]), getattr(self, sparse_vars[l]))
				setattr(self, mask_names[l], W_masked)
				setattr(self, norm_names[l], W_norm)
				hidden_layer = tf.nn.relu(self._operation(getattr(self, previous_layer), getattr(self, mask_names[l]), len(layer_sizes[l+1])) + getattr(self, b_vars[l]))
				setattr(self, layer_names[l+1], hidden_layer) 
			else:
				hidden_layer = tf.nn.relu(self._operation(getattr(self, previous_layer), getattr(self, w_vars[l]), len(layer_sizes[l+1])) + getattr(self, b_vars[l]))
				setattr(self, layer_names[l+1], hidden_layer) 
			
			previous_layer = layer_names[l+1]

			if l < len(cnn_size):

				if pool_size[l]:
					pool_layer = self._max_pool_2x2(getattr(self, layer_names[l+1]))
					setattr(self, layer_names[l+1]+"_pool", pool_layer) 
					previous_layer = layer_names[l+1] + "_pool"

				setattr(self, layer_names[l+1] + "_dropout", tf.nn.dropout(getattr(self,  previous_layer), self.dropout)) 
				previous_layer = layer_names[l+1] + "_dropout"


		#Compute standard Cross Entropy
		self.pre_softmax = getattr(self, previous_layer)
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


	@staticmethod
	def _weight_variable(shape, initial = None):
			if initial is None:
				W0 = tf.glorot_uniform_initializer()
				return tf.get_variable(shape=shape, initializer=W0, name=str(np.random.randint(1e10)))
				#return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

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
	def _conv2d(x, W):
			return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

	@staticmethod
	def _operation(x, W, i):
			if i>1:
				return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
			else:
				return tf.matmul(x, W)


	@staticmethod
	def _max_pool_2x2(x):
			return tf.nn.max_pool(x,
														ksize = [1,2,2,1],
														strides=[1,2,2,1],
														padding='SAME')

	@staticmethod
	def _log_a_variable(shape, initial = None):
			if initial is None:
				a0 = tf.Variable(tf.random_normal(shape, mean=0.0, stddev=0.01))
				return a0
			else:
				a0 = tf.constant(initial, shape = shape, dtype=tf.float32)
				return tf.Variable(a0)

	def get_l0_norm_full(self, x, log_a):

		shape = x.get_shape()
		# sample u
		u = tf.random_uniform(shape)
		# compute hard concrete distribution
		# i.e., implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution
		y = tf.sigmoid((tf.log(u) - tf.log(1.0 - u) + log_a) / temperature)
		# stretch hard concrete distribution
		s_bar = y * (limit_1 - limit_0) + limit_0
		l0_norm = tf.reduce_sum(tf.clip_by_value(
				tf.sigmoid(log_a - temperature * math.log(-limit_0 / limit_1)),
				epsilon, 1-epsilon))

		# get mask for calculating sparse version of tensor
		mask = hard_sigmoid(s_bar)
		# return masked version of tensor and l0 norm
		return tf.multiply(x, mask), l0_norm

def hard_sigmoid(x):
		return tf.minimum(tf.maximum(x, tf.zeros_like(x)), tf.ones_like(x))

