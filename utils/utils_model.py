import numpy as np
import json
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

with open('config.json') as config_file:
    config = json.load(config_file)

from utils_MLP_model import init_MLP_vars

w_vars, b_vars, stable_var, sparse_vars = init_MLP_vars()
mask_names = [w_vars[l] + str("_masked") for l in range(len(w_vars))]




def get_loss(model, args):
    loss = model.xent + model.regularizer
    if args.rho and not args.is_stable:
        loss = model.robust_xent

    elif args.is_stable:
        if args.rho:
            loss = model.robust_stable_xent
        else:
            loss = model.stable_xent

    return loss


def create_dict(args, num_classes, num_features, train_shape, test_size):
    dict_exp = {}
    dict_exp['logits_acc'] = np.zeros((config['num_experiments'], test_size[0], num_classes))
    layer_sizes = [num_features] +  args.network_size + [num_classes]

    for i in range(len(w_vars)):
        dict_exp[w_vars[i]] = np.zeros((config['num_experiments'], layer_sizes[i], layer_sizes[i+1]))
        dict_exp[b_vars[i]] = np.zeros((config['num_experiments'], layer_sizes[i+1]))
        dict_exp[sparse_vars[i]] = np.zeros((config['num_experiments'], layer_sizes[i], layer_sizes[i+1]))
        dict_exp[w_vars[i] + '_nonzero'] = np.zeros(config['num_experiments'])
        dict_exp[w_vars[i] + '_killed_neurons'] = np.zeros(config['num_experiments'])
        dict_exp[w_vars[i] + '_killed_input_features'] = np.zeros(config['num_experiments'])

    dict_exp[stable_var] = np.zeros(config['num_experiments'])
    dict_exp['preds'] = np.zeros((config['num_experiments'], test_size[0]))
    dict_exp['test_accs'] = np.zeros(config['num_experiments'])
    dict_exp['iterations'] = np.zeros(config['num_experiments'])
    dict_exp['adv_test_accs'] = {eps_test: np.zeros(config['num_experiments']) for eps_test in args.robust_test}

    return dict_exp


def update_dict(dict_exp, args, sess, model, test_dict, experiment):

    dict_exp[stable_var][experiment] = sess.run(getattr(model, stable_var))

    for i in range(len(w_vars)):
        dict_exp[b_vars[i]][experiment] = sess.run(getattr(model, b_vars[i]))
        dict_exp[sparse_vars[i]][experiment] = sess.run(getattr(model, sparse_vars[i]))


        if args.l0 > 0:
            W_masked =  sess.run(getattr(model, mask_names[i]))
            dict_exp[w_vars[i] + '_nonzero'][experiment] = sum( W_masked.reshape(-1)  >0)/ W_masked.reshape(-1).shape[0] 
            dict_exp[w_vars[i] + '_killed_neurons'][experiment] = sum(np.sum(W_masked, axis=0) == 0)
            dict_exp[w_vars[i] + '_killed_input_features'][experiment] =  sum(np.sum(W_masked, axis=1) == 0)
            dict_exp[w_vars[i]][experiment] = W_masked
        else:
            W = sess.run(getattr(model, w_vars[i]))
            dict_exp[w_vars[i]][experiment] = W
            dict_exp[w_vars[i] + '_nonzero'][experiment] = sum(W.reshape(-1) > 0) / W.reshape(-1).shape[0]


    dict_exp['logits_acc'][experiment] = sess.run(model.logits, feed_dict=test_dict)
    dict_exp['preds'][experiment] = sess.run(model.y_pred, feed_dict=test_dict)


    return dict_exp

def get_best_model(dict_exp, experiment, args, num_classes, batch_size, subset_ratio, num_features, spec, network_module):
    spec.loader.exec_module(network_module)
    
    weights = [dict_exp[w][experiment] for w in w_vars]
    biases = [dict_exp[b][experiment] for b in b_vars]
    stab_weight = dict_exp[stable_var][experiment]
    sparse_weights = [dict_exp[sparse_a][experiment] for sparse_a in sparse_vars]

    stored_weights = {}
    stored_weights['network_weights'] = weights
    stored_weights['network_biases'] = biases
    stored_weights['stability_variable'] = stab_weight
    stored_weights['sparsity_variables'] = sparse_weights

    best_model = network_module.Model(num_classes, batch_size, args.network_size, subset_ratio, num_features, args.dropout, args.l2, args.l0, args.rho , stored_weights)
    return best_model
    


