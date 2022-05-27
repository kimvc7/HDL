from datetime import datetime
import numpy as np
import csv
from utils import total_gini
import tensorflow.compat.v1 as tf
import json
from pgd_attack import LinfPGDAttack
import utils_init 

with open('config.json') as config_file:
    config = json.load(config_file)


def print_metrics(sess, model, nat_dict, val_dict, test_dict, ii, args, summary_writer, experiment, global_step):
    network_size = list(utils_init.NN[args.network_type])
    w_vars, b_vars, stable_var, sparse_vars = utils_init.init_vars(len(network_size)+1)
    mask_names = [w_vars[l] + str("_masked") for l in range(len(w_vars))]


    nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
    test_acc = sess.run(model.accuracy, feed_dict=test_dict)
    val_acc = sess.run(model.accuracy, feed_dict=val_dict)
    nat_xent = sess.run(model.xent, feed_dict=nat_dict)
    stable_xent = sess.run(model.stable_xent, feed_dict=nat_dict)
    robust_xent = sess.run(model.robust_xent, feed_dict=nat_dict)
    robust_stable_xent = sess.run(model.robust_stable_xent, feed_dict=nat_dict)
    stable_var = sess.run(getattr(model, stable_var), feed_dict=nat_dict)


    print('Step {}:    ({})'.format(ii, datetime.now()))
    print('    training nat accuracy {:.4}'.format(nat_acc * 100))
    print('    validation nat accuracy {:.4}'.format(val_acc * 100))
    print('    Nat Xent {:.4}'.format(nat_xent))

    if args.is_stable:
        print('    Stability Variable {:.4}'.format(stable_var ))
        print('    Stable Xent {:.4}'.format(stable_xent))

    if args.rho > 0 :
        print('    Robust Xent {:.4}'.format(robust_xent))
        if args.is_stable:
            print('    Robust Stable Xent {:.4}'.format(robust_stable_xent))



    for i in range(len(w_vars)):
        

        if args.l0 > 0:
            W_masked =  sess.run(getattr(model, mask_names[i]), feed_dict= nat_dict)
            print(' #   Killed neurons - ' + w_vars[i], sum(np.sum(W_masked.reshape(-1, W_masked.shape[-1]), axis=0) == 0))
            print(' #   Killed input neurons - ' + w_vars[i], sum(np.sum(W_masked.reshape(-1, W_masked.shape[-1]), axis=1) == 0))
            print('    Non zero features percentage - ' + w_vars[i] , 100*(sum( W_masked.reshape(-1) != 0)/ W_masked.reshape(-1).shape[0]), "%")
        

    regularizer = sess.run(model.regularizer, feed_dict=nat_dict)
    print('    Regularizer', regularizer)



    summary = tf.Summary(value=[
          tf.Summary.Value(tag='Train Xent', simple_value= nat_xent),
          tf.Summary.Value(tag='Val Acc', simple_value= val_acc),
          tf.Summary.Value(tag='Train Acc', simple_value= nat_acc),
          tf.Summary.Value(tag='Train Stable Xent', simple_value= stable_xent),
          tf.Summary.Value(tag='Train Robust Stable Xent', simple_value= robust_stable_xent),
          tf.Summary.Value(tag='Test Acc', simple_value= test_acc)])
    
    summary_writer.add_summary(summary, global_step.eval(sess))

    for i in range(len(w_vars)):
        if args.l0 > 0:
            W_masked =  sess.run(getattr(model, mask_names[i]), feed_dict= nat_dict)
            summary_sparse = tf.Summary(value=[
                tf.Summary.Value(tag=w_vars[i] + '_killed_neurons', simple_value=sum(np.sum(W_masked.reshape(-1, W_masked.shape[-1]), axis=0) == 0)),
                tf.Summary.Value(tag=w_vars[i] + '_killed_inputs', simple_value=sum(np.sum(W_masked.reshape(-1, W_masked.shape[-1]), axis=1) == 0)),
                tf.Summary.Value(tag=w_vars[i] + '_nonzero', simple_value=sum( W_masked.reshape(-1)  != 0)/ W_masked.reshape(-1).shape[0])])

            summary_writer.add_summary(summary_sparse, global_step.eval(sess))
    



def update_best_acc(args, best_model, x_test, y_test, experiment, dict_exp):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        clip = True
        if "uci" in args.data_set:
            clip = False

        for rho_test in args.robust_test:
            attack = LinfPGDAttack(best_model, rho_test, config['k'], config['a'], 
            config['random_start'], config['loss_func'], clip)
            x_test_adv = attack.perturb(x_test, y_test, sess)
            adv_dict = {best_model.x_input: x_test_adv, best_model.y_input: y_test, best_model.temp: 1}
            dict_exp['adv_test_accs'][rho_test][experiment] = sess.run(best_model.accuracy, feed_dict=adv_dict)

        test_dict = {best_model.x_input: x_test, best_model.y_input: y_test, best_model.temp: 1}
        test_acc = sess.run(best_model.accuracy, feed_dict=test_dict)
        dict_exp['test_acc'][experiment] = test_acc*100

    return dict_exp


def print_stability_measures(dict_exp, args, num_experiments, batch_size, subset_ratio, max_train_steps, network_path):
    network_size = list(utils_init.NN[args.network_type])
    w_vars, b_vars, stable_var, sparse_vars = utils_init.init_vars(len(network_size)+1)
    
    avg_test_acc = np.array([float(k) for k in dict_exp['test_acc']]).mean()
    std = np.array([float(k) for k in dict_exp['test_acc']]).std()
    logit_stability = np.mean(np.std(dict_exp['logits_acc'], axis=0), axis=0)
    gini_stability = total_gini(dict_exp['preds'].transpose())


    print('  Average testing accuracy {:.4}'.format(avg_test_acc * 100))
    print('  Individual accuracies: \n', dict_exp['test_acc'])
    print('  Adv testing accuracies', dict_exp['adv_test_accs'])
    print('  Stability values', dict_exp[stable_var])
    print('  Test Accuracy std {:.2}'.format(std))
    print("  Logits std", np.mean(np.mean(np.std(dict_exp['logits_acc'], axis=0), axis=0)))
    print("  Gini stability", gini_stability)


    weights_stability = print_layer_stability(dict_exp, num_experiments, args)
    weights_nonzero = [np.mean(dict_exp[w_vars[i]+ '_nonzero']) for i in range(len(w_vars))]

    for i in range(len(w_vars)):
        print(w_vars[i] + ' non zero percentage', weights_nonzero[i])
    

    file = open(str('results_' + network_path + args.data_set + '.csv'), 'a+', newline='')

    file_read = open(str('results_' + network_path + args.data_set + '.csv'), "r")
    one_char = file_read.read(1)

    writer = csv.writer(file)

    if not len(one_char):
        headers = []
        headers += ['num_experiments', 'batch_size', 'subset_ratio', 'max_train_steps']
        headers += ['test accuracy '+ str(i) for i in range(num_experiments)]
        headers += ['# Iterations '+ str(i) for i in range(num_experiments)]

        for key in dict_exp:
            if key not in w_vars+ b_vars+ [stable_var]+ sparse_vars + ['adv_test_accs', 'preds']:
                headers +=  ['Avg '+str(key)]

        headers += ['Avg test adversarial acc for rho = '+ str(rho) for rho in  args.robust_test]
        headers += ['is_stable', 'rho', 'train_size', 'l2', 'l0', 'network_size', 'learning rate']
        headers += [w_vars[i] + ' Stability' for i in range(len(w_vars))]
        headers += ['test_acc_std', 'logit_stability', 'gini_stability' ] 
        writer.writerow(headers)

    with file:

        cols = []

        cols += [num_experiments, batch_size, subset_ratio, max_train_steps]
        cols += [dict_exp['test_acc'][i] for i in range(num_experiments)]
        cols += [dict_exp['iterations'][i] for i in range(num_experiments)]
        
        for key in dict_exp:
            if key not in w_vars+ b_vars+ [stable_var]+ sparse_vars + ['adv_test_accs', 'preds']:
                cols += [np.mean(dict_exp[key])]

        cols += [np.mean(dict_exp['adv_test_accs'][rho]) for rho in args.robust_test]
        cols += [args.is_stable, args.rho,  args.train_size, args.l2, args.l0, network_size, args.lr]
        cols += weights_stability
        cols += [std, logit_stability, gini_stability ] 

        print(cols)
        writer.writerow(cols)


def print_layer_stability(dict_exp, num_experiments, args):
    network_size = list(utils_init.NN[args.network_type])
    w_vars, b_vars, stable_var, sparse_vars = utils_init.init_vars(len(network_size)+1)

    stabilities = []
    for i in range(len(w_vars)):
        w_i = [dict_exp[w_vars[i]][experiment].reshape(-1) for experiment in range(num_experiments)]
        w_stability = np.mean(np.std(w_i, axis=0), axis=0)
        print(w_vars[i] + " std", w_stability)
        stabilities = stabilities + [w_stability]
    return stabilities
