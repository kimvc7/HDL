import json
import utils_init 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

mask_initial_value = 0

def init_sparsity_constants():
    lambd = 1e-8
    mask_initial_value = 0
    return lambd, mask_initial_value


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

def _prune(mask_weights):
    prunned_weights = []
    for mask_weight in mask_weights:
        prunned = np.clip(mask_weight, a_min=-100000, a_max=mask_initial_value)
        prunned_weights.append(prunned)
    return prunned_weights

def store_network(model, args, sess, nat_dict, prune):
    network_size = list(utils_init.NN[args.network_type])
    w_vars, b_vars, stable_var, sparse_vars = utils_init.init_vars(len(network_size)+1)
    mask_names = [w_vars[l] + str("_masked") for l in range(len(w_vars))]

    weights = [sess.run(getattr(model, w_vars[i]), feed_dict= nat_dict) for i in range(len(w_vars))]
    biases = [sess.run(getattr(model, b_vars[i]), feed_dict= nat_dict) for i in range(len(b_vars))]
    stab_weight = sess.run(getattr(model, stable_var))
    sparse_weights = [sess.run(getattr(model, sparse_vars[i]), feed_dict= nat_dict) for i in range(len(sparse_vars))]

    if prune:
        sparse_weights = _prune(sparse_weights)

    stored_weights = {}
    stored_weights['network_weights'] = weights
    stored_weights['network_biases'] = biases
    stored_weights['stability_variable'] = stab_weight
    stored_weights['sparsity_variables'] = sparse_weights

    return stored_weights


