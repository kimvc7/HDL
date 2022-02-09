import json

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def init_sparsity_constants():
	limit_0 = -0.1
	limit_1 = 1.1
	temperature = 2 / 3
	epsilon = 1e-6
	lambd = 1
	return limit_0, limit_1, temperature, epsilon, lambd


def init_weights(network_vars_w, network_vars_b, stable_var, sparse_vars):
	weights = [None for i in range(len(network_vars_w))]
	biases = [None for i in range(len(network_vars_b))]
	theta = None
	sparse_weights = [None for i in range(len(sparse_vars))]
	return weights, biases, theta, sparse_weights

def reset_stored_weights(stored_weights):
	weights = stored_weights['network_weights']
	biases = stored_weights['network_biases']
	stab_weight = stored_weights['stability_variable']
	sparse_weights = stored_weights['sparsity_variables']
	return weights, biases, stab_weight, sparse_weights 


